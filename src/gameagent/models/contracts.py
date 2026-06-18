"""Core contracts shared by the runner, adapters, and model clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class ActionType(str, Enum):
    TAP = "tap"
    SWIPE = "swipe"
    LONG_PRESS = "long_press"
    WAIT = "wait"
    BACK = "back"
    HOME = "home"
    NOOP = "noop"


@dataclass
class Action:
    type: ActionType
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    duration_ms: int = 80
    reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            type=ActionType(data.get("type", "noop")),
            x=_optional_int(data.get("x")),
            y=_optional_int(data.get("y")),
            x2=_optional_int(data.get("x2")),
            y2=_optional_int(data.get("y2")),
            duration_ms=int(data.get("duration_ms", 80)),
            reason=data.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass
class Observation:
    frame_id: int
    timestamp: float
    width: int
    height: int
    image_bytes: bytes | None = None
    image_path: str | None = None
    previous_action: Action | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "image_path": self.image_path,
            "previous_action": self.previous_action.to_dict()
            if self.previous_action is not None
            else None,
            "metadata": self.metadata,
        }


@dataclass
class Decision:
    action: Action
    observation_summary: str = ""
    intent: str = ""
    confidence: float = 0.0
    raw_response: dict[str, Any] | str | None = None
    model_name: str = "unknown"
    latency_ms: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_name: str = "unknown") -> "Decision":
        action_data = data.get("action", {"type": "noop", "reason": "missing action"})
        return cls(
            action=Action.from_dict(action_data),
            observation_summary=str(data.get("observation_summary", "")),
            intent=str(data.get("intent", "")),
            confidence=float(data.get("confidence", 0.0)),
            raw_response=data,
            model_name=str(data.get("model_name", model_name)),
            latency_ms=int(data.get("latency_ms", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "observation_summary": self.observation_summary,
            "intent": self.intent,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ExecutionResult:
    ok: bool
    message: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaptureAdapter(Protocol):
    def capture(self, frame_id: int, previous_action: Action | None = None) -> Observation:
        """Capture one frame from the environment."""


class ControlAdapter(Protocol):
    def execute(self, action: Action, observation: Observation) -> ExecutionResult:
        """Execute one validated action."""


class ModelClient(Protocol):
    @property
    def model_name(self) -> str:
        """Human-readable model name."""

    def decide(self, observation: Observation) -> Decision:
        """Return the next decision for the current observation."""


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
