"""Main observe-decide-act loop."""

from __future__ import annotations

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
        print(f"[gameagent] run_dir={self.logger.path}")
        print(f"[gameagent] rule/action trace={self.logger.rule_action_trace_path}")
        print(f"[gameagent] model={self.model.model_name}")

        for frame_id in range(1, self.options.max_episode_steps + 1):
            if self._should_stop():
                print("[gameagent] stop requested")
                return 0

            step_started = time.perf_counter()
            observation = self.capture.capture(frame_id, previous_action)
            decision = self.model.decide(observation)
            decision.action = self.validator.validate(decision.action, observation)

            if self.options.dry_run:
                result_action = Action(
                    type=ActionType.NOOP,
                    reason=f"dry-run skipped {decision.action.type.value}",
                )
                result = self.control.execute(result_action, observation)
            else:
                result = self.control.execute(decision.action, observation)

            self.logger.log_step(
                observation=observation,
                decision=decision,
                result=result,
                extra={
                    "dry_run": self.options.dry_run,
                    "settle_after_action_ms": self.options.settle_after_action_ms,
                },
            )
            previous_action = decision.action

            print(
                "[gameagent] "
                f"step={frame_id} action={decision.action.type.value} "
                f"x={decision.action.x} y={decision.action.y} ok={result.ok} "
                f"intent={decision.intent[:80]}"
            )
            self._wait_before_next_observation(step_started)

        print("[gameagent] max_episode_steps reached")
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
