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
        timestamped_run_dir: bool = True,
        numbered_run_dir: bool = False,
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
            "rule_memory": pipeline.get("rule_memory"),
            "plan": pipeline.get("plan"),
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
