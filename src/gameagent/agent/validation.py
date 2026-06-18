"""Action validation and normalization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from gameagent.models import Action, ActionType, Observation


@dataclass
class ActionValidator:
    allowed_actions: set[ActionType] = field(default_factory=lambda: set(ActionType))
    clamp_coordinates: bool = True
    max_repeated_action_count: int = 8
    _recent: deque[str] = field(default_factory=deque)

    def validate(self, action: Action, observation: Observation) -> Action:
        if action.type not in self.allowed_actions:
            return Action(
                type=ActionType.NOOP,
                reason=f"action type is not allowed: {action.type.value}",
            )

        if action.type in {ActionType.TAP, ActionType.LONG_PRESS} and (
            action.x is None or action.y is None
        ):
            return Action(
                type=ActionType.NOOP,
                reason=f"{action.type.value} action missing x/y",
            )
        if action.type == ActionType.SWIPE and (
            action.x is None or action.y is None or action.x2 is None or action.y2 is None
        ):
            return Action(type=ActionType.NOOP, reason="swipe action missing coordinates")

        normalized = self._clamp(action, observation) if self.clamp_coordinates else action
        signature = _signature(normalized)
        self._recent.append(signature)
        while len(self._recent) > self.max_repeated_action_count:
            self._recent.popleft()
        if (
            self.max_repeated_action_count > 0
            and len(self._recent) == self.max_repeated_action_count
            and len(set(self._recent)) == 1
        ):
            self._recent.clear()
            return Action(type=ActionType.NOOP, reason="repeated action guard triggered")
        return normalized

    def _clamp(self, action: Action, observation: Observation) -> Action:
        width = max(observation.width, 1)
        height = max(observation.height, 1)
        action.x = _clamp_int(action.x, 0, width - 1)
        action.y = _clamp_int(action.y, 0, height - 1)
        action.x2 = _clamp_int(action.x2, 0, width - 1)
        action.y2 = _clamp_int(action.y2, 0, height - 1)
        action.duration_ms = max(int(action.duration_ms), 0)
        return action


def allowed_actions_from_config(values: list[str] | None) -> set[ActionType]:
    if not values:
        return set(ActionType)
    return {ActionType(value) for value in values}


def _clamp_int(value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    return max(low, min(high, int(value)))


def _signature(action: Action) -> str:
    return (
        f"{action.type.value}:{action.x}:{action.y}:{action.x2}:{action.y2}:"
        f"{action.duration_ms}"
    )
