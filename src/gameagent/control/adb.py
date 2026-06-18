"""ADB touch control for BlueStacks or Android devices."""

from __future__ import annotations

import shutil
import subprocess
import time
import os
from pathlib import Path

from gameagent.models import Action, ActionType, ExecutionResult, Observation


class AdbControlAdapter:
    def __init__(
        self,
        device_id: str = "auto",
        adb_path: str = "adb",
        min_action_interval_ms: int = 250,
        timeout_s: float = 10.0,
        adb_server_socket: str | None = None,
    ) -> None:
        self.device_id = device_id
        self.adb_path = adb_path
        self.min_action_interval_ms = min_action_interval_ms
        self.timeout_s = timeout_s
        self.adb_server_socket = adb_server_socket
        self._last_action_at = 0.0

    def execute(self, action: Action, observation: Observation) -> ExecutionResult:
        _ensure_adb(self.adb_path)
        self._respect_interval()

        if action.type == ActionType.NOOP:
            return ExecutionResult(ok=True, message=action.reason or "noop")
        if action.type == ActionType.WAIT:
            time.sleep(max(action.duration_ms, 0) / 1000)
            return ExecutionResult(ok=True, message="wait complete")

        cmd = self._command_for(action)
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
            env=_adb_env(self.adb_server_socket),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._last_action_at = time.perf_counter()
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            return ExecutionResult(
                ok=False,
                message=f"ADB action failed: {stderr or proc.returncode}",
                latency_ms=latency_ms,
            )
        return ExecutionResult(
            ok=True,
            message=f"executed {action.type.value}",
            latency_ms=latency_ms,
            metadata={"device_id": self.device_id, "frame_id": observation.frame_id},
        )

    def _command_for(self, action: Action) -> list[str]:
        base = self._base_cmd() + ["shell", "input"]
        if action.type == ActionType.TAP:
            _require_xy(action)
            return base + ["tap", str(action.x), str(action.y)]
        if action.type == ActionType.LONG_PRESS:
            _require_xy(action)
            duration = max(action.duration_ms, 500)
            return base + [
                "swipe",
                str(action.x),
                str(action.y),
                str(action.x),
                str(action.y),
                str(duration),
            ]
        if action.type == ActionType.SWIPE:
            _require_xy(action)
            if action.x2 is None or action.y2 is None:
                raise ValueError("swipe action requires x2 and y2")
            return base + [
                "swipe",
                str(action.x),
                str(action.y),
                str(action.x2),
                str(action.y2),
                str(max(action.duration_ms, 1)),
            ]
        if action.type == ActionType.BACK:
            return base + ["keyevent", "KEYCODE_BACK"]
        if action.type == ActionType.HOME:
            return base + ["keyevent", "KEYCODE_HOME"]
        raise ValueError(f"Unsupported ADB action: {action.type.value}")

    def _base_cmd(self) -> list[str]:
        if self.device_id and self.device_id != "auto":
            return [self.adb_path, "-s", self.device_id]
        return [self.adb_path]

    def _respect_interval(self) -> None:
        elapsed_ms = (time.perf_counter() - self._last_action_at) * 1000
        wait_ms = self.min_action_interval_ms - elapsed_ms
        if wait_ms > 0:
            time.sleep(wait_ms / 1000)


def _ensure_adb(adb_path: str) -> None:
    if shutil.which(adb_path) is None:
        raise RuntimeError(f"ADB executable not found: {adb_path}")


def _adb_env(adb_server_socket: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if adb_server_socket:
        env["ADB_SERVER_SOCKET"] = adb_server_socket
    android_dir = Path.home() / ".android"
    try:
        android_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path.cwd() / ".android"
        fallback.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(Path.cwd())
        env["ANDROID_USER_HOME"] = str(fallback)
    return env


def _require_xy(action: Action) -> None:
    if action.x is None or action.y is None:
        raise ValueError(f"{action.type.value} action requires x and y")
