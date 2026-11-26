"""Main observe-decide-act loop."""

from __future__ import annotations

import io
import math
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path

from gameagent.agent import ActionValidator
from gameagent.models import (
    Action, ActionType, CaptureAdapter, ControlAdapter, ModelClient, Decision, ExecutionResult,
)
from gameagent.storage import EpisodeLogger


@dataclass
class RunnerOptions:
    tick_interval_ms: int = 800
    settle_after_action_ms: int = 800
    pre_action_refresh: bool = False
    max_pre_action_frame_change: float = 0.08
    max_episode_steps: int = 1000
    step_budget_mode: str = 'observations'
    max_idle_seconds: float = 120.0
    emergency_stop_path: str | None = "./STOP"
    dry_run: bool = False


class _StopRequested(BaseException):
    """Bypass adapters' ordinary Exception retry/fallback handlers on cancellation."""


class AgentRunner:
    def __init__(
        self,
        capture: CaptureAdapter,
        model: ModelClient,
        control: ControlAdapter,
        validator: ActionValidator,
        logger: EpisodeLogger,
        options: RunnerOptions,
    ) -> None:
        self.capture = capture
        self.model = model
        self.control = control
        self.validator = validator
        self.logger = logger
        self.options = options
        if options.step_budget_mode not in {'observations', 'actions'}:
            raise ValueError('step_budget_mode must be observations or actions')
        if not math.isfinite(options.max_idle_seconds) or options.max_idle_seconds <= 0:
            raise ValueError('max_idle_seconds must be positive')
        self._stop = False
        self._interruptible = False
        self._previous_handlers: dict = {}
        self._inflight: dict = {}

    def run(self) -> int:
        self._install_signal_handlers()
        finish_reason = 'completed'
        try:
            return self._run_episode()
        except (_StopRequested, KeyboardInterrupt):
            self._stop = True
            finish_reason = 'interrupted'
            print('\n[gameagent] stop requested; saving logs and finalizing video...', flush=True)
            return 130
        except BaseException:
            finish_reason = 'error'
            raise
        finally:
            # Further Ctrl+C must not interrupt JSON writes or the MP4 trailer.
            self._interruptible = False
            try:
                if self._inflight and not self._inflight.get('logged'):
                    try:
                        self._log_interrupted_step(finish_reason)
                    except Exception as exc:
                        print(f'[gameagent] interrupted frame logging failed: {exc}', flush=True)
                if finish_reason != 'completed' or not getattr(self.logger, 'finish_reason', None):
                    self.logger.finish_reason = finish_reason
                print('[gameagent] finalizing video and logs; please wait...', flush=True)
                self.logger.finalize()
                print('[gameagent] run finished; returning to terminal.', flush=True)
            finally:
                for signum, handler in self._previous_handlers.items():
                    signal.signal(signum, handler)

    def _run_episode(self) -> int:
        previous_action: Action | None = None
        previous_execution_ok: bool | None = None
        skipped_action_frame_id: int | None = None
        blocker_recoveries: list[tuple[bytes, Action]] = []
        print(f"[gameagent] run_dir={self.logger.path}")
        print(f"[gameagent] rule/action trace={self.logger.rule_action_trace_path}")
        print(f"[gameagent] model={self.model.model_name}")
        print(f"[gameagent] budget={self.options.max_episode_steps} {self.options.step_budget_mode}; "
              f"idle_timeout={self.options.max_idle_seconds:g}s", flush=True)
        self.logger.start_video_recording()
        action_steps = 0
        frame_id = 0
        budget_steps = 0
        idle_since = time.monotonic()
        self.logger.action_steps = 0
        self.logger.step_budget_mode = self.options.step_budget_mode
        while budget_steps < self.options.max_episode_steps:
            self._check_stop()
            if (self.options.step_budget_mode == 'actions'
                    and time.monotonic() - idle_since >= self.options.max_idle_seconds):
                return self._idle_timeout()
            frame_id += 1
            self._inflight = {'frame_id': frame_id, 'phase': 'capture', 'logged': False}

            step_started = time.perf_counter()
            observation = self._call_interruptible(self.capture.capture, frame_id, previous_action)
            self._inflight['observation'] = observation
            observation.metadata['dry_run'] = self.options.dry_run
            observation.metadata['previous_execution_ok'] = previous_execution_ok
            if skipped_action_frame_id is not None:
                observation.metadata["skipped_action_frame_id"] = skipped_action_frame_id
                skipped_action_frame_id = None
            self._inflight['phase'] = 'decision'
            decision = self._call_interruptible(self.model.decide, observation)
            self._inflight['decision'] = decision
            self._check_stop()
            # Network/model time also counts toward the idle deadline. Check
            # after the response, before executing a now-too-late action.
            idle_expired = (self.options.step_budget_mode == 'actions'
                            and time.monotonic() - idle_since >= self.options.max_idle_seconds)
            if idle_expired:
                decision = replace(decision, action=Action(type=ActionType.NOOP,
                    reason='idle timeout: action not executed'), intent='idle timeout')
                self._inflight['decision'] = decision
            decision.action = self.validator.validate(decision.action, observation)

            action_started_at = time.time()
            self._inflight['action_started_at'] = action_started_at
            intended_action = decision.action
            frame_change: float | None = None
            stale_observation = False
            stale_recovered = False
            recovery_action: Action | None = None
            recovery_change: float | None = None
            if self.options.pre_action_refresh and _is_screen_action(intended_action):
                self._inflight['phase'] = 'refresh'
                refreshed = self._call_interruptible(self.capture.capture, frame_id, previous_action)
                frame_change = _frame_change_ratio(
                    observation.image_bytes,
                    refreshed.image_bytes,
                )
                stale_observation = (
                    frame_change is not None
                    and frame_change > self.options.max_pre_action_frame_change
                )
                if stale_observation and not self.options.dry_run:
                    for blocker_image, blocker_action in reversed(blocker_recoveries):
                        blocker_change = _frame_change_ratio(
                            blocker_image,
                            refreshed.image_bytes,
                        )
                        if blocker_change is None or blocker_change > 0.03:
                            continue
                        print(
                            "[gameagent] known blocking screen appeared; "
                            f"recovering with {blocker_action.type.value}",
                            flush=True,
                        )
                        recovery_action = blocker_action
                        self._inflight['phase'] = 'recovery_action'
                        self._inflight['recovery_action'] = blocker_action.to_dict()
                        recovery_result = self._call_interruptible(
                            self.control.execute, blocker_action, refreshed)
                        self._check_stop()
                        if recovery_result.ok:
                            self._call_interruptible(time.sleep, 0.5)
                            self._inflight['phase'] = 'refresh'
                            recovered = self._call_interruptible(self.capture.capture, frame_id, previous_action)
                            recovery_change = _frame_change_ratio(
                                observation.image_bytes,
                                recovered.image_bytes,
                            )
                            stale_recovered = (
                                recovery_change is not None
                                and recovery_change
                                <= self.options.max_pre_action_frame_change
                            )
                            if stale_recovered:
                                stale_observation = False
                                print(
                                    "[gameagent] blocking screen dismissed; "
                                    "executing original action",
                                    flush=True,
                                )
                        break
                if stale_observation:
                    print(
                        "[gameagent] stale observation: "
                        f"frame={frame_id} change={frame_change:.3f}; action skipped",
                        flush=True,
                    )
                    decision.action = Action(
                        type=ActionType.NOOP,
                        reason="stale_observation: screen changed before action",
                    )

            self._check_stop()
            self._inflight['phase'] = 'action'
            self._inflight['action_started_at'] = time.time()
            if self.options.dry_run or stale_observation:
                result_action = Action(
                    type=ActionType.NOOP,
                    reason=(
                        "stale_observation: skipped outdated action"
                        if stale_observation
                        else f"dry-run skipped {decision.action.type.value}"
                    ),
                )
                result = self._call_interruptible(self.control.execute, result_action, observation)
            else:
                result = self._call_interruptible(self.control.execute, decision.action, observation)
            self._inflight['result'] = result
            previous_execution_ok = result.ok and not self.options.dry_run and not stale_observation
            action_finished_at = time.time()
            # Count non-wait inputs actually attempted, including execution
            # failures. Dry-run counts eligible planned inputs so it is bounded.
            counted_action = (_is_screen_action(decision.action) and not stale_observation)
            if counted_action:
                action_steps += 1
                idle_since = time.monotonic()
            budget_steps = (action_steps if self.options.step_budget_mode == 'actions' else frame_id)
            self.logger.action_steps = action_steps

            if (
                not stale_observation
                and result.ok
                and observation.image_bytes
                and _is_blocker_dismiss_decision(decision)
            ):
                blocker_recoveries.append((observation.image_bytes, intended_action))
                blocker_recoveries = blocker_recoveries[-5:]

            self._inflight['phase'] = 'logging'
            self.logger.log_step(
                observation=observation,
                decision=decision,
                result=result,
                extra={
                    "dry_run": self.options.dry_run,
                    "settle_after_action_ms": self.options.settle_after_action_ms,
                    "action_started_at": action_started_at,
                    "action_finished_at": action_finished_at,
                    "stale_observation": stale_observation,
                    "pre_action_frame_change": frame_change,
                    "stale_recovered": stale_recovered,
                    "recovery_action": recovery_action.to_dict() if recovery_action else None,
                    "post_recovery_frame_change": recovery_change,
                    "intended_action": intended_action.to_dict(),
                    "action_step": action_steps,
                    "observation_step": frame_id,
                    "counted_action": counted_action,
                    "step_budget_mode": self.options.step_budget_mode,
                },
            )
            self._inflight['logged'] = True
            self._check_stop()
            if stale_observation:
                previous_action = None
                skipped_action_frame_id = frame_id
            else:
                previous_action = decision.action

            print(
                "[gameagent] "
                f"step={budget_steps}/{self.options.max_episode_steps} observation={frame_id} "
                f"action={decision.action.type.value} "
                f"x={decision.action.x} y={decision.action.y} ok={result.ok} "
                f"intent={decision.intent[:80]}"
            )
            response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            pipeline = response.get('pipeline') or {}
            routing = pipeline.get('screen_routing') or {}
            if routing.get('stop_requested') is True:
                print(f'[gameagent] automatic stop: {routing.get("stop_reason") or "recovery exhausted"}; '
                      'saving video and exiting', flush=True)
                self.logger.finish_reason = 'unknown_recovery_exhausted'
                self.logger.stop_reason = routing.get('stop_reason')
                return 2
            if idle_expired:
                return self._idle_timeout()
            if budget_steps >= self.options.max_episode_steps:
                break
            self._inflight['phase'] = 'settling'
            self._call_interruptible(self._wait_before_next_observation, step_started,
                                     counted_action and not self.options.dry_run)

        print(f"[gameagent] max_episode_steps reached ({self.options.step_budget_mode}); "
              f"actions={action_steps} observations={frame_id}", flush=True)
        return 0

    def _idle_timeout(self) -> int:
        self.logger.finish_reason = 'idle_timeout'
        self.logger.stop_reason = f'no actionable input for {self.options.max_idle_seconds:g} seconds'
        print(f'[gameagent] automatic stop: {self.logger.stop_reason}; saving video and exiting', flush=True)
        return 2

    def _check_stop(self) -> None:
        if self._should_stop():
            raise _StopRequested()

    def _call_interruptible(self, function, *args):
        self._interruptible = True
        try:
            self._check_stop()
            return function(*args)
        finally:
            self._interruptible = False

    def _log_interrupted_step(self, reason: str) -> None:
        current = self._inflight
        observation = current.get('observation')
        if observation is None:
            return
        phase = current['phase']
        planned = current.get('decision')
        execution_uncertain = phase in {'action', 'recovery_action'}
        result = current.get('result') or ExecutionResult(
            ok=False, message=f'{reason} during {phase}',
            metadata={'execution_status': 'unknown' if execution_uncertain else 'not_executed'},
        )
        decision = planned or Decision(action=Action(type=ActionType.NOOP),
                                       intent=f'{reason} while waiting for model')
        if not execution_uncertain and not current.get('result'):
            decision = replace(decision, action=Action(type=ActionType.NOOP, reason=f'{reason}: not executed'))
        self.logger.log_step(observation, decision, result, extra={
            'interrupted': True, 'interrupted_phase': phase, 'termination_reason': reason,
            'execution_uncertain': execution_uncertain, 'dry_run': self.options.dry_run,
            'intended_action': planned.action.to_dict() if planned else None,
            'recovery_action': current.get('recovery_action'),
            'action_started_at': current.get('action_started_at', time.time()),
            'action_finished_at': time.time(),
        })
        current['logged'] = True

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        stop_path = self.options.emergency_stop_path
        return bool(stop_path and Path(stop_path).exists())

    def _wait_before_next_observation(self, step_started: float, after_input: bool = True) -> None:
        """Settle only after input; WAIT already slept in the control adapter.

        Preserve any explicitly configured tick interval for all observations.
        The Candy action-budget configuration uses tick_interval_ms=0.
        """

        settle_s = max(self.options.settle_after_action_ms, 0) / 1000 if after_input else 0
        if settle_s > 0:
            time.sleep(settle_s)

        interval_s = max(self.options.tick_interval_ms, 0) / 1000
        elapsed_s = time.perf_counter() - step_started
        remaining_s = interval_s - elapsed_s
        if remaining_s > 0:
            time.sleep(remaining_s)

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: object) -> None:
            self._stop = True
            if self._interruptible:
                raise _StopRequested()

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop)


def _is_screen_action(action: Action) -> bool:
    return action.type in {
        ActionType.TAP,
        ActionType.DOUBLE_TAP,
        ActionType.SWIPE,
        ActionType.LONG_PRESS,
        ActionType.BACK,
        ActionType.HOME,
    }


def _is_blocker_dismiss_decision(decision: object) -> bool:
    text = " ".join(
        str(getattr(decision, field, "") or "")
        for field in ("observation_summary", "intent")
    ).lower()
    action = getattr(decision, "action", None)
    reason = str(getattr(action, "reason", "") or "").lower()
    combined = f"{text} {reason}"
    blocker_words = ("popup", "overlay", "tutorial", "팝업", "오버레이", "튜토리얼")
    dismiss_words = ("dismiss", "close", "닫기", "닫", "해제")
    return any(word in combined for word in blocker_words) and any(
        word in combined for word in dismiss_words
    )


def _frame_change_ratio(before: bytes | None, after: bytes | None) -> float | None:
    """Return normalized mean pixel change after downsampling two screenshots."""
    if not before or not after:
        return None
    try:
        from PIL import Image, ImageChops, ImageStat

        with Image.open(io.BytesIO(before)) as first_image:
            first = first_image.convert("RGB").resize((96, 160))
        with Image.open(io.BytesIO(after)) as second_image:
            second = second_image.convert("RGB").resize((96, 160))
    except (ModuleNotFoundError, OSError):
        return None
    if first.size != second.size:
        return 1.0
    channel_means = ImageStat.Stat(ImageChops.difference(first, second)).mean
    return sum(channel_means) / (len(channel_means) * 255.0)
