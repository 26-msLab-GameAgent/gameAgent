"""Main observe-decide-act loop."""

from __future__ import annotations

import io
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, CaptureAdapter, ControlAdapter, ModelClient
from gameagent.storage import EpisodeLogger


@dataclass
class RunnerOptions:
    tick_interval_ms: int = 800
    settle_after_action_ms: int = 800
    pre_action_refresh: bool = False
    max_pre_action_frame_change: float = 0.08
    max_episode_steps: int = 1000
    emergency_stop_path: str | None = "./STOP"
    dry_run: bool = False


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
        self._stop = False

    def run(self) -> int:
        self._install_signal_handlers()
        previous_action: Action | None = None
        skipped_action_frame_id: int | None = None
        blocker_recoveries: list[tuple[bytes, Action]] = []
        print(f"[gameagent] run_dir={self.logger.path}")
        print(f"[gameagent] rule/action trace={self.logger.rule_action_trace_path}")
        print(f"[gameagent] model={self.model.model_name}")
        self.logger.start_video_recording()

        for frame_id in range(1, self.options.max_episode_steps + 1):
            if self._should_stop():
                print("[gameagent] stop requested")
                self.logger.finalize()
                return 0

            step_started = time.perf_counter()
            observation = self.capture.capture(frame_id, previous_action)
            if skipped_action_frame_id is not None:
                observation.metadata["skipped_action_frame_id"] = skipped_action_frame_id
                skipped_action_frame_id = None
            decision = self.model.decide(observation)
            decision.action = self.validator.validate(decision.action, observation)

            action_started_at = time.time()
            intended_action = decision.action
            frame_change: float | None = None
            stale_observation = False
            stale_recovered = False
            recovery_action: Action | None = None
            recovery_change: float | None = None
            if self.options.pre_action_refresh and _is_screen_action(intended_action):
                refreshed = self.capture.capture(frame_id, previous_action)
                frame_change = _frame_change_ratio(
                    observation.image_bytes,
                    refreshed.image_bytes,
                )
                stale_observation = (
                    frame_change is not None
                    and frame_change > self.options.max_pre_action_frame_change
                )
                if stale_observation:
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
                        recovery_result = self.control.execute(blocker_action, refreshed)
                        if recovery_result.ok:
                            time.sleep(0.5)
                            recovered = self.capture.capture(frame_id, previous_action)
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

            if self.options.dry_run or stale_observation:
                result_action = Action(
                    type=ActionType.NOOP,
                    reason=(
                        "stale_observation: skipped outdated action"
                        if stale_observation
                        else f"dry-run skipped {decision.action.type.value}"
                    ),
                )
                result = self.control.execute(result_action, observation)
            else:
                result = self.control.execute(decision.action, observation)
            action_finished_at = time.time()

            if (
                not stale_observation
                and result.ok
                and observation.image_bytes
                and _is_blocker_dismiss_decision(decision)
            ):
                blocker_recoveries.append((observation.image_bytes, intended_action))
                blocker_recoveries = blocker_recoveries[-5:]

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
                },
            )
            if stale_observation:
                previous_action = None
                skipped_action_frame_id = frame_id
            else:
                previous_action = decision.action

            print(
                "[gameagent] "
                f"step={frame_id} action={decision.action.type.value} "
                f"x={decision.action.x} y={decision.action.y} ok={result.ok} "
                f"intent={decision.intent[:80]}"
            )
            self._wait_before_next_observation(step_started)

        print("[gameagent] max_episode_steps reached")
        self.logger.finalize()
        return 0

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        stop_path = self.options.emergency_stop_path
        return bool(stop_path and Path(stop_path).exists())

    def _wait_before_next_observation(self, step_started: float) -> None:
        """Wait after action execution before capturing the next screenshot."""

        settle_s = max(self.options.settle_after_action_ms, 0) / 1000
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

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)


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
