"""Small HTTP bridge for testing full-remote mode."""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from gameagent.perception.mock import _TINY_PNG


class MockBridgeHandler(BaseHTTPRequestHandler):
    server_version = "GameAgentMockBridge/0.1"

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/v1/frame":
                self._send_json(_frame_response(payload))
            elif self.path == "/v1/decide":
                self._send_json(_decision_response(payload))
            elif self.path == "/v1/action":
                self._send_json({"ok": True, "message": "mock bridge executed action"})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[mock-bridge] {self.address_string()} {fmt % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _frame_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": payload.get("frame_id", 0),
        "width": 720,
        "height": 1280,
        "image_base64": base64.b64encode(_TINY_PNG).decode("ascii"),
        "metadata": {"source": "mock_bridge"},
    }


def _decision_response(payload: dict[str, Any]) -> dict[str, Any]:
    frame_id = int(payload.get("frame_id", 0))
    if frame_id % 2 == 0:
        action = {"type": "tap", "x": 360, "y": 640, "duration_ms": 80}
        intent = "remote mock center tap"
    else:
        action = {"type": "wait", "duration_ms": 100, "reason": "remote mock wait"}
        intent = "remote mock wait"
    return {
        "observation_summary": f"remote mock frame {frame_id}",
        "intent": intent,
        "confidence": 1.0,
        "action": action,
        "model_name": "mock_bridge_policy",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gameagent-mock-bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), MockBridgeHandler)
    print(f"[mock-bridge] listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

