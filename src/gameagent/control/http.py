"""HTTP action bridge for remote environments."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from gameagent.models import Action, ExecutionResult, Observation


class HttpControlAdapter:
    def __init__(self, endpoint: str, timeout_s: float = 10.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def execute(self, action: Action, observation: Observation) -> ExecutionResult:
        payload = {
            "frame_id": observation.frame_id,
            "screen": {"width": observation.width, "height": observation.height},
            "action": action.to_dict(),
        }
        started = time.perf_counter()
        response = _post_json(self.endpoint, payload, self.timeout_s)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ExecutionResult(
            ok=bool(response.get("ok", True)),
            message=str(response.get("message", "remote action executed")),
            latency_ms=latency_ms,
            metadata=dict(response.get("metadata", {})),
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

