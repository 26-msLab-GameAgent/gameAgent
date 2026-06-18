"""Mock frame capture that works in every environment."""

from __future__ import annotations

import base64
import time

from gameagent.models import Action, Observation


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGA"
    "WjR9awAAAABJRU5ErkJggg=="
)


class MockCaptureAdapter:
    def __init__(self, width: int = 720, height: int = 1280) -> None:
        self.width = width
        self.height = height

    def capture(self, frame_id: int, previous_action: Action | None = None) -> Observation:
        return Observation(
            frame_id=frame_id,
            timestamp=time.time(),
            width=self.width,
            height=self.height,
            image_bytes=_TINY_PNG,
            previous_action=previous_action,
            metadata={"source": "mock"},
        )

