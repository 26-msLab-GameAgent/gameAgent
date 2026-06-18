"""Mock control adapter that records actions without touching a device."""

from __future__ import annotations

import time

from gameagent.models import Action, ExecutionResult, Observation


class MockControlAdapter:
    def execute(self, action: Action, observation: Observation) -> ExecutionResult:
        started = time.perf_counter()
        if action.type.value == "wait":
            time.sleep(max(action.duration_ms, 0) / 1000)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ExecutionResult(
            ok=True,
            message=f"mock executed {action.type.value}",
            latency_ms=latency_ms,
            metadata={"source": "mock", "frame_id": observation.frame_id},
        )

