"""Append-only episode logging."""

from __future__ import annotations

import json
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
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(run_dir) / stamp
        self.frames_dir = self.path / "frames"
        self.save_frames = save_frames
        self.save_model_raw_response = save_model_raw_response
        self.path.mkdir(parents=True, exist_ok=True)
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.path / "steps.jsonl"

    def log_step(
        self,
        observation: Observation,
        decision: Decision,
        result: ExecutionResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        image_path = observation.image_path
        if self.save_frames and observation.image_bytes:
            frame_path = self.frames_dir / f"{observation.frame_id:06d}.png"
            frame_path.write_bytes(observation.image_bytes)
            image_path = str(frame_path)
        obs_log = observation.to_log_dict()
        obs_log["image_path"] = image_path
        decision_log = decision.to_dict()
        if not self.save_model_raw_response:
            decision_log["raw_response"] = None
        record = {
            "observation": obs_log,
            "decision": decision_log,
            "execution": result.to_dict(),
            "extra": extra or {},
        }
        with self.steps_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

