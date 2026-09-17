"""HTTP model client for local or remote inference servers."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any

from gameagent.models import Action, ActionType, Decision, Observation


class HttpModelClient:
    def __init__(
        self,
        endpoint: str,
        model_name: str = "remote_inference",
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.endpoint = endpoint
        self._model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model_name

    def decide(self, observation: Observation) -> Decision:
        payload = {
            "frame_id": observation.frame_id,
            "timestamp": observation.timestamp,
            "screen": {"width": observation.width, "height": observation.height},
            "image_base64": base64.b64encode(observation.image_bytes).decode("ascii")
            if observation.image_bytes
            else None,
            "image_path": observation.image_path,
            "previous_action": observation.previous_action.to_dict()
            if observation.previous_action
            else None,
            "metadata": observation.metadata,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                started = time.perf_counter()
                data = _post_json(self.endpoint, payload, self.timeout_s)
                latency_ms = int((time.perf_counter() - started) * 1000)
                decision = Decision.from_dict(data, model_name=self.model_name)
                decision.latency_ms = decision.latency_ms or latency_ms
                return decision
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        return Decision(
            action=Action(
                type=ActionType.WAIT,
                duration_ms=1000,
                reason=f"remote inference failed: {last_error}",
            ),
            observation_summary="remote inference error",
            intent="모델 응답 오류로 잠시 기다립니다.",
            confidence=0.0,
            raw_response=str(last_error),
            model_name=self.model_name,
        )


def _post_json(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body}") from exc
