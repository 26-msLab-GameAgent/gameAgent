"""Mock model client for local smoke tests."""

from __future__ import annotations

from gameagent.models import Action, ActionType, Decision, Observation


class MockModelClient:
    @property
    def model_name(self) -> str:
        return "mock_policy"

    def decide(self, observation: Observation) -> Decision:
        if observation.frame_id % 2 == 0:
            action = Action(
                type=ActionType.TAP,
                x=max(observation.width // 2, 1),
                y=max(observation.height // 2, 1),
                duration_ms=80,
                reason="smoke-test center tap",
            )
            intent = "exercise tap path"
        else:
            action = Action(type=ActionType.WAIT, duration_ms=100, reason="let the screen update")
            intent = "exercise wait path"
        return Decision(
            action=action,
            observation_summary=f"mock frame {observation.frame_id}",
            intent=intent,
            confidence=1.0,
            raw_response={"source": "mock"},
            model_name=self.model_name,
        )

