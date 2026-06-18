"""ADB-based screenshot capture for BlueStacks or Android devices."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import os
from pathlib import Path

from gameagent.models import Action, Observation


class AdbCaptureAdapter:
    def __init__(
        self,
        device_id: str = "auto",
        adb_path: str = "adb",
        width: int | None = None,
        height: int | None = None,
        timeout_s: float = 10.0,
        capture_method: str = "auto",
        adb_server_socket: str | None = None,
    ) -> None:
        self.device_id = device_id
        self.adb_path = adb_path
        self.width = width
        self.height = height
        self.timeout_s = timeout_s
        self.capture_method = capture_method
        self.adb_server_socket = adb_server_socket

    def capture(self, frame_id: int, previous_action: Action | None = None) -> Observation:
        _ensure_adb(self.adb_path)
        started = time.perf_counter()
        method = self.capture_method
        if method == "exec_out":
            image_bytes = self._capture_exec_out()
        elif method == "file_pull":
            image_bytes = self._capture_file_pull(frame_id)
        elif method == "auto":
            try:
                image_bytes = self._capture_exec_out()
                method = "exec_out"
            except Exception:
                image_bytes = self._capture_file_pull(frame_id)
                method = "file_pull"
        else:
            raise ValueError(f"Unknown ADB capture method: {method}")
        latency_ms = int((time.perf_counter() - started) * 1000)

        width, height = _png_size(image_bytes)
        return Observation(
            frame_id=frame_id,
            timestamp=time.time(),
            width=self.width or width,
            height=self.height or height,
            image_bytes=image_bytes,
            previous_action=previous_action,
            metadata={
                "source": "adb_screencap",
                "device_id": self.device_id,
                "capture_method": method,
                "capture_latency_ms": latency_ms,
                "raw_width": width,
                "raw_height": height,
            },
        )

    def _base_cmd(self) -> list[str]:
        if self.device_id and self.device_id != "auto":
            return [self.adb_path, "-s", self.device_id]
        return [self.adb_path]

    def _capture_exec_out(self) -> bytes:
        proc = _run_adb(
            self._base_cmd() + ["exec-out", "screencap", "-p"],
            timeout_s=self.timeout_s,
            adb_server_socket=self.adb_server_socket,
        )
        if proc.returncode != 0 or not proc.stdout:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ADB exec-out screencap failed: {stderr or proc.returncode}")
        return proc.stdout

    def _capture_file_pull(self, frame_id: int) -> bytes:
        remote_path = f"/sdcard/gameagent_frame_{frame_id}.png"
        with tempfile.NamedTemporaryFile(prefix="gameagent_frame_", suffix=".png") as fh:
            proc = _run_adb(
                self._base_cmd() + ["shell", "screencap", "-p", remote_path],
                timeout_s=self.timeout_s,
                adb_server_socket=self.adb_server_socket,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ADB shell screencap failed: {stderr or proc.returncode}")

            proc = _run_adb(
                self._base_cmd() + ["pull", remote_path, fh.name],
                timeout_s=self.timeout_s,
                adb_server_socket=self.adb_server_socket,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ADB pull screencap failed: {stderr or proc.returncode}")

            _run_adb(
                self._base_cmd() + ["shell", "rm", "-f", remote_path],
                timeout_s=5,
                adb_server_socket=self.adb_server_socket,
            )
            return Path(fh.name).read_bytes()


def _run_adb(
    cmd: list[str],
    timeout_s: float,
    adb_server_socket: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        env=_adb_env(adb_server_socket),
    )


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


def _png_size(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    return 0, 0
