"""Append-only episode logging."""

from __future__ import annotations

import json
import atexit
import os
import signal
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gameagent.models import Decision, ExecutionResult, Observation


class EpisodeLogger:
    def __init__(
        self,
        run_dir: str | Path = "./runs",
        save_frames: bool = True,
        save_model_raw_response: bool = True,
        timestamped_run_dir: bool = True,
        numbered_run_dir: bool = False,
        video_recording: bool = False,
        adb_path: str = "adb",
        adb_device_id: str = "auto",
        adb_server_socket: str | None = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.session_id = stamp
        if numbered_run_dir:
            self.path = _next_numbered_run_path(Path(run_dir))
        elif timestamped_run_dir:
            self.path = Path(run_dir) / stamp
        else:
            self.path = Path(run_dir)
        self.frames_dir = self.path / "frames"
        self.save_frames = save_frames
        self.save_model_raw_response = save_model_raw_response
        self.path.mkdir(parents=True, exist_ok=True)
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.path / "steps.jsonl"
        self.rule_action_trace_path = self.path / "rule_action_trace.jsonl"
        self.demonstration_json_path = self.path / "demonstration.json"
        self.demonstration_video_path = self.path / "demonstration.mp4"
        self._video_origin: float | None = None
        self._screen: tuple[int, int] | None = None
        self._demonstration_actions: list[dict[str, Any]] = []
        self._video_frames: list[tuple[Path, float]] = []
        self.video_recording = video_recording
        self.adb_path = adb_path
        self.adb_device_id = adb_device_id
        self.adb_server_socket = adb_server_socket
        self._recording_stop = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._recording_process: subprocess.Popen[bytes] | None = None
        self._recording_raw_path = self.path / ".screenrecord.h264"
        self._continuous_video_created = False
        self._finalized = False
        atexit.register(self.finalize)

    def start_video_recording(self) -> None:
        """Record the device continuously, independently of agent decision latency."""
        if not self.video_recording or self._recording_thread is not None:
            return
        self._video_origin = time.time()
        self._recording_thread = threading.Thread(
            target=self._record_video_segments,
            name=f"gameagent-video-{self.session_id}",
            daemon=True,
        )
        self._recording_thread.start()
        print(
            f"[gameagent] continuous video recording started: {self.demonstration_video_path}",
            flush=True,
        )

    def _adb_base_cmd(self) -> list[str]:
        command = [self.adb_path]
        if self.adb_device_id and self.adb_device_id != "auto":
            command.extend(["-s", self.adb_device_id])
        return command

    def _adb_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.adb_server_socket:
            environment["ADB_SERVER_SOCKET"] = self.adb_server_socket
        return environment

    def _record_video_segments(self) -> None:
        # Write the elementary H.264 stream directly to one host file. Unlike an
        # MP4 on the device, this remains recoverable even if screenrecord or ADB
        # is interrupted before it can write a container trailer.
        with self._recording_raw_path.open("ab") as video_output:
            while not self._recording_stop.is_set():
                command = self._adb_base_cmd() + [
                    "exec-out",
                    "screenrecord",
                    "--output-format=h264",
                    "--time-limit",
                    "175",
                    "-",
                ]
                try:
                    self._recording_process = subprocess.Popen(
                        command,
                        stdout=video_output,
                        stderr=subprocess.PIPE,
                        env=self._adb_env(),
                    )
                    return_code = self._recording_process.wait()
                    video_output.flush()
                    if return_code != 0 and not self._recording_stop.is_set():
                        error = (self._recording_process.stderr.read() or b"").decode(
                            "utf-8", errors="replace"
                        ).strip()
                        print(
                            f"[gameagent] continuous H.264 recording failed: "
                            f"{error or return_code}",
                            flush=True,
                        )
                        break
                except Exception as exc:
                    print(f"[gameagent] continuous video recording failed: {exc}", flush=True)
                    break
                finally:
                    self._recording_process = None

    def _stop_video_recording(self) -> bool:
        thread = self._recording_thread
        if thread is None:
            return False
        self._recording_stop.set()
        process = self._recording_process
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
        thread.join(timeout=10)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        self._recording_thread = None
        return self._convert_raw_recording()

    def _convert_raw_recording(self) -> bool:
        if not self._recording_raw_path.exists() or self._recording_raw_path.stat().st_size == 0:
            return False
        try:
            self._encode_raw_h264()
            return self.demonstration_video_path.exists()
        except Exception as exc:
            print(f"[gameagent] converting continuous video failed: {exc}", flush=True)
            return False
        finally:
            self._recording_raw_path.unlink(missing_ok=True)

    def _encode_raw_h264(self) -> None:
        from fractions import Fraction
        from itertools import chain

        import av

        with av.open(str(self._recording_raw_path), format="h264") as source:
            frames = iter(source.decode(video=0))
            first_frame = next(frames)
            with av.open(str(self.demonstration_video_path), mode="w") as output:
                output_stream = output.add_stream("libx264", rate=30)
                output_stream.width = first_frame.width
                output_stream.height = first_frame.height
                output_stream.pix_fmt = "yuv420p"
                for index, frame in enumerate(chain((first_frame,), frames)):
                    frame.pts = index
                    frame.time_base = Fraction(1, 30)
                    for packet in output_stream.encode(frame):
                        output.mux(packet)
                for packet in output_stream.encode():
                    output.mux(packet)

    def log_step(
        self,
        observation: Observation,
        decision: Decision,
        result: ExecutionResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        image_path = observation.image_path
        if self.save_frames and observation.image_bytes:
            frame_path = self.frames_dir / f"{self.session_id}_{observation.frame_id:06d}.png"
            frame_path.write_bytes(observation.image_bytes)
            image_path = str(frame_path)
            if self._video_origin is None:
                self._video_origin = observation.timestamp
            self._video_frames.append((frame_path, observation.timestamp))
        self._screen = (observation.width, observation.height)
        obs_log = observation.to_log_dict()
        obs_log["image_path"] = image_path
        decision_log = decision.to_dict()
        if not self.save_model_raw_response:
            decision_log["raw_response"] = None
        record = {
            "session_id": self.session_id,
            "observation": obs_log,
            "decision": decision_log,
            "execution": result.to_dict(),
            "extra": extra or {},
        }
        with self.steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._append_rule_action_trace(observation, decision, result, image_path)
        self._append_demonstration_action(observation, decision, extra or {})
        self._write_demonstration_json(status="recording")

    def _append_demonstration_action(
        self,
        observation: Observation,
        decision: Decision,
        extra: dict[str, Any],
    ) -> None:
        action = decision.action
        if action.type.value not in {"tap", "double_tap", "swipe"} or self._video_origin is None:
            return
        started = float(extra.get("action_started_at", observation.timestamp))
        finished = float(extra.get("action_finished_at", started + action.duration_ms / 1000))
        start_ms = max(0, round((started - self._video_origin) * 1000))
        end_ms = max(start_ms, round((finished - self._video_origin) * 1000))
        width, height = self._screen or (observation.width, observation.height)

        def point(x: int | None, y: int | None) -> dict[str, Any]:
            px, py = int(x or 0), int(y or 0)
            return {
                "x": px,
                "y": py,
                "x_norm": round(px / max(1, width - 1), 6),
                "y_norm": round(py / max(1, height - 1), 6),
            }

        record: dict[str, Any] = {
            "index": len(self._demonstration_actions),
            "type": action.type.value,
            "timestamp": datetime.fromtimestamp(started, timezone.utc).astimezone().isoformat(
                timespec="milliseconds"
            ),
            "duration_ms": max(1, round((finished - started) * 1000)),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "video_start_ms": start_ms,
            "video_end_ms": end_ms,
            "source": "autonomous_agent",
            "reason": action.reason,
        }
        if action.type.value in {"tap", "double_tap"}:
            record["position"] = point(action.x, action.y)
        else:
            record["start"] = point(action.x, action.y)
            record["end"] = point(action.x2, action.y2)
        self._demonstration_actions.append(record)

    def _write_demonstration_json(self, status: str) -> None:
        if self._video_origin is None or self._screen is None:
            return
        width, height = self._screen
        document = {
            "schema_version": "1.0",
            "session": {
                "started_at": datetime.fromtimestamp(
                    self._video_origin, timezone.utc
                ).astimezone().isoformat(timespec="milliseconds"),
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "capture_method": "autonomous_agent_frames",
                "screen": {"width": width, "height": height},
                "coordinate_system": "Android display pixels; origin is top-left",
                "video": {
                    "path": self.demonstration_video_path.name,
                    "format": "mp4",
                    "capture_method": (
                        "adb_screenrecord"
                        if self._continuous_video_created
                        else "timestamped_agent_frames"
                    ),
                    "started_offset_ms": 0,
                    "status": status,
                },
            },
            "actions": self._demonstration_actions,
        }
        temporary = self.demonstration_json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.demonstration_json_path)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._recording_thread is not None:
            self._continuous_video_created = self._stop_video_recording()
            if self._continuous_video_created:
                self._write_demonstration_json(status="finished")
                print(
                    f"[gameagent] continuous demonstration: "
                    f"{self.demonstration_json_path} + {self.demonstration_video_path}",
                    flush=True,
                )
                return
            print(
                "[gameagent] continuous recording unavailable; falling back to step frames",
                flush=True,
            )
        if len(self._video_frames) < 1:
            return
        try:
            self._encode_video_with_pyav()
            self._write_demonstration_json(status="finished")
            print(
                f"[gameagent] 0730 demonstration: {self.demonstration_json_path} + "
                f"{self.demonstration_video_path}",
                flush=True,
            )
            return
        except Exception as pyav_error:
            print(f"[gameagent] PyAV video encoding failed: {pyav_error}", flush=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[gameagent] ffmpeg not found; demonstration video was not created")
            self._write_demonstration_json(status="frames_only")
            return
        manifest = self.path / "demonstration_frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        for index, (frame, timestamp) in enumerate(self._video_frames):
            escaped = str(frame.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            if index + 1 < len(self._video_frames):
                duration = max(0.04, self._video_frames[index + 1][1] - timestamp)
            else:
                duration = 1.5
            lines.append(f"duration {duration:.6f}")
        last = str(self._video_frames[-1][0].resolve()).replace("'", "'\\''")
        lines.append(f"file '{last}'")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(manifest),
                "-vsync", "vfr", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.demonstration_video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            self._write_demonstration_json(status="finished")
            print(
                f"[gameagent] 0730 demonstration: {self.demonstration_json_path} + "
                f"{self.demonstration_video_path}",
                flush=True,
            )
        else:
            self._write_demonstration_json(status="video_error")
            print(f"[gameagent] ffmpeg failed: {proc.stderr.strip()}", flush=True)

    def _encode_video_with_pyav(self) -> None:
        from fractions import Fraction

        import av
        from PIL import Image

        origin = self._video_frames[0][1]
        timeline = list(self._video_frames)
        timeline.append((self._video_frames[-1][0], self._video_frames[-1][1] + 1.5))
        with av.open(str(self.demonstration_video_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            with Image.open(timeline[0][0]) as first_image:
                stream.width, stream.height = first_image.size
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1000)
            for path, timestamp in timeline:
                with Image.open(path) as image:
                    frame = av.VideoFrame.from_image(image.convert("RGB"))
                frame.pts = max(0, round((timestamp - origin) * 1000))
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def _append_rule_action_trace(
        self,
        observation: Observation,
        decision: Decision,
        result: ExecutionResult,
        image_path: str | None,
    ) -> None:
        raw_response = decision.raw_response
        pipeline = (
            raw_response.get("pipeline", {})
            if isinstance(raw_response, dict)
            else {}
        )
        if not isinstance(pipeline, dict):
            pipeline = {}

        trace = {
            "session_id": self.session_id,
            "frame_id": observation.frame_id,
            "timestamp": observation.timestamp,
            "image_path": image_path,
            "perception": pipeline.get("perception"),
            "previous_outcome": pipeline.get("previous_outcome"),
            "interaction_memory": pipeline.get("interaction_memory"),
            "rule_memory": pipeline.get("rule_memory"),
            "plan": pipeline.get("plan"),
            "stage_timings_ms": pipeline.get("stage_timings_ms"),
            "decision": {
                "intent": decision.intent,
                "confidence": decision.confidence,
                "action": decision.action.to_dict(),
            },
            "execution": result.to_dict(),
        }
        with self.rule_action_trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _next_numbered_run_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in root.glob("run_[0-9][0-9][0-9][0-9]"):
        try:
            numbers.append(int(path.name.removeprefix("run_")))
        except ValueError:
            continue
    number = max(numbers, default=0) + 1
    while True:
        candidate = root / f"run_{number:04d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1
