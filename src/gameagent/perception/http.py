"""HTTP frame capture adapter for remote environments."""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from typing import Any

from gameagent.models import Action, Observation


class HttpFrameCaptureAdapter:
    def __init__(self, endpoint: str, timeout_s: float = 10.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def capture(self, frame_id: int, previous_action: Action | None = None) -> Observation:
        payload = {
            "frame_id": frame_id,
            "previous_action": previous_action.to_dict() if previous_action else None,
        }
        started = time.perf_counter()
        response = _post_json(self.endpoint, payload, self.timeout_s)
        latency_ms = int((time.perf_counter() - started) * 1000)
        image_bytes = base64.b64decode(response["image_base64"]) if response.get("image_base64") else None
        return Observation(
            frame_id=frame_id,
            timestamp=time.time(),
            width=int(response.get("width", 0)),
            height=int(response.get("height", 0)),
            image_bytes=image_bytes,
            image_path=response.get("image_path"),
            previous_action=previous_action,
            metadata={
                "source": "http_frame_stream",
                "capture_latency_ms": latency_ms,
                **dict(response.get("metadata", {})),
            },
        )


def _post_json(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

