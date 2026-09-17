"""Local Qwen-VL decision server.

This server receives a frame from the gameagent runner and returns a structured
action. It can run in two modes:

- real mode: load Qwen2.5-VL and ask it to produce action JSON
- mock mode: deterministic wait/tap responses for wiring tests
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import mimetypes
import os
import re
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    FastAPI = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]

    class HTTPException(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(f"{status_code}: {detail}")

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(default_factory: Any) -> Any:  # type: ignore[no-redef]
        return default_factory()


ActionName = Literal[
    "tap", "double_tap", "swipe", "long_press", "wait", "back", "home", "noop"
]
AgentName = Literal["perception", "rule_learner", "planner", "policy"]
QWEN_VL_MODEL_IDS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}
PIPELINE_AGENTS: tuple[AgentName, ...] = ("perception", "rule_learner", "planner", "policy")
HOSTED_PROVIDERS = {"openai", "gemini", "anthropic"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLOBAL_RULE_PATH = PROJECT_ROOT / "configs" / "profiles" / "global_rule.yaml"


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}" if self.provider in HOSTED_PROVIDERS else self.model


class Screen(BaseModel):
    width: int
    height: int


class PreviousAction(BaseModel):
    type: str
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    duration_ms: int | None = None
    reason: str | None = None


class DecideRequest(BaseModel):
    frame_id: int
    screen: Screen
    image_base64: str | None = None
    image_path: str | None = None
    previous_action: PreviousAction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionServer:
    def __init__(
        self,
        model_id: str,
        mock: bool = False,
        max_new_tokens: int = 1024,
        max_pixels: int = 589824,
        temperature: float = 0.0,
        profile_path: str | None = None,
        rule_memory_path: str | None = None,
        agent_mode: str = "pipeline",
        agent_model_ids: dict[str, str] | None = None,
        rule_learning_interval: int = 3,
        tutorial_enabled: bool = False,
    ) -> None:
        self.model_id = model_id
        self.agent_model_refs = {
            agent: _parse_model_ref((agent_model_ids or {}).get(agent, model_id))
            for agent in PIPELINE_AGENTS
        }
        self.agent_model_ids = {agent: ref.key for agent, ref in self.agent_model_refs.items()}
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.temperature = temperature
        self.profile = _load_profile_with_global_rules(
            profile_path,
            tutorial_model_ref=self.agent_model_refs["perception"],
            auto_tutorial=tutorial_enabled and not mock,
        )
        self.rule_memory_path = Path(rule_memory_path).resolve() if rule_memory_path else None
        self.agent_mode = agent_mode
        self.rule_learning_interval = max(1, int(rule_learning_interval))
        self._loaded = False
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._process_vision_info: Any = None
        self._history: list[dict[str, Any]] = []
        self._interaction_memory: list[dict[str, Any]] = []
        self._pending_interactions: list[dict[str, Any]] = []
        self._pending_rule_outcomes: list[dict[str, Any]] = []
        self._verified_outcomes_since_rule_learning = 0
        self._rule_memory: dict[str, Any] = self._load_rule_memory()
        self._failed_swipes_by_board: dict[str, set[str]] = {}
        self._failed_cells_by_board: dict[str, set[str]] = {}
        self._viewport_recovery: dict[str, Any] | None = None
        self._stage_progress: dict[str, Any] = _empty_stage_progress()

    def decide(self, req: DecideRequest) -> dict[str, Any]:
        if self.mock:
            return self._mock_decide(req)
        if req.frame_id <= 1:
            self._history.clear()
            self._interaction_memory.clear()
            self._pending_interactions.clear()
            self._pending_rule_outcomes.clear()
            self._verified_outcomes_since_rule_learning = 0
            self._rule_memory = self._load_rule_memory()
            self._failed_swipes_by_board.clear()
            self._failed_cells_by_board.clear()
            self._viewport_recovery = None
            self._stage_progress = _empty_stage_progress()

        self._discard_skipped_interaction(req)

        image_path = self._materialize_image(req)
        image_fingerprint = _image_fingerprint(image_path)
        board_signature = _candy_board_signature(self.profile, image_path)
        self._record_previous_action_outcome(req, board_signature)
        started = time.perf_counter()
        try:
            if self.agent_mode == "pipeline":
                decision = self._decide_pipeline(
                    req,
                    image_path,
                    image_fingerprint,
                    board_signature,
                    started,
                )
                self._remember(req, decision)
                return decision

            raw_text = self._generate(req, image_path)
            parsed = _parse_model_json(raw_text)
            action = _normalize_action(parsed.get("action", {}), req.screen)
            confidence = float(parsed.get("confidence", 0.0))
            if (
                action.get("type") in {"tap", "double_tap", "long_press"}
                and action.get("x") == 0
                and action.get("y") == 0
                and confidence <= 0.05
            ):
                action = {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "model returned an unsafe placeholder coordinate",
                }
            latency_ms = int((time.perf_counter() - started) * 1000)
            if board_signature and action.get("type") in {
                "tap", "double_tap", "swipe", "wait", "noop"
            }:
                fallback = _fallback_decision(
                    self.profile,
                    image_path,
                    req,
                    raw_text,
                    latency_ms,
                    self._history,
                    image_fingerprint,
                    board_signature,
                    self._failed_swipes_by_board,
                    self._failed_cells_by_board,
                )
                if fallback is not None and fallback.get("action", {}).get("type") == "swipe":
                    fallback["model_name"] = f"{self.model_id}+candy_crush_grid_fallback"
                    self._remember(req, fallback)
                    return fallback
            decision = {
                "observation_summary": str(parsed.get("observation_summary", "")),
                "intent": str(parsed.get("intent", "")),
                "confidence": confidence,
                "action": action,
                "model_name": self.model_id,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "image_fingerprint": image_fingerprint,
                "board_signature": board_signature,
            }
            self._remember(req, decision)
            return decision
        except ValueError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            fallback = _fallback_decision(
                self.profile,
                image_path,
                req,
                raw_text if "raw_text" in locals() else "",
                latency_ms,
                self._history,
                image_fingerprint,
                board_signature,
                self._failed_swipes_by_board,
                self._failed_cells_by_board,
            )
            if fallback is not None:
                fallback["error"] = str(exc)
                self._remember(req, fallback)
                return fallback
            decision = {
                "observation_summary": "unparseable model output",
                "intent": "화면 전환 또는 판단 불가 상태라 잠시 기다립니다.",
                "confidence": 0.0,
                "action": {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "model output was not valid action JSON",
                },
                "model_name": self.model_id,
                "latency_ms": latency_ms,
                "raw_text": raw_text if "raw_text" in locals() else "",
                "error": str(exc),
            }
            self._remember(req, decision)
            return decision
        finally:
            if image_path.name.startswith("gameagent_frame_"):
                image_path.unlink(missing_ok=True)

    def _decide_pipeline(
        self,
        req: DecideRequest,
        image_path: Path,
        image_fingerprint: list[int] | None,
        board_signature: str | None,
        started: float,
    ) -> dict[str, Any]:
        raw_perception = ""
        raw_rule_learning = ""
        raw_plan = ""
        raw_policy = ""
        perception: dict[str, Any] | None = None
        outcome: dict[str, Any] | None = None
        plan: dict[str, Any] | None = None
        stage_timings_ms: dict[str, int] = {}
        active_stage = "perception"
        _trace_pipeline_frame(req, self.agent_model_ids)
        try:
            stage_started = time.perf_counter()
            _trace_pipeline_waiting("1/4 화면 인식", self.agent_model_ids["perception"])
            raw_perception, perception = self._generate_json_with_retry(
                req,
                image_path,
                _perception_prompt(req, self.profile, self._history),
                agent_name="perception",
                include_image=True,
            )
            stage_timings_ms["perception"] = _trace_pipeline_timing(
                req.frame_id, "perception", stage_started
            )
            _trace_pipeline_value("1/4 화면 인식 결과", perception)
            _record_completed_stage_from_perception(self._stage_progress, perception)

            stage_started = time.perf_counter()
            outcome = self._verify_previous_outcome(req, image_path, perception)
            stage_timings_ms["previous_action_verification"] = _trace_pipeline_timing(
                req.frame_id,
                "previous_action_verification",
                stage_started,
                status="completed" if outcome is not None else "skipped",
            )
            if outcome is not None:
                _trace_pipeline_value("이전 행동 결과 검증", outcome)

            active_stage = "rule_learner"
            stage_started = time.perf_counter()
            rule_learning_due = (
                self._verified_outcomes_since_rule_learning
                >= self.rule_learning_interval
            )
            if rule_learning_due:
                _trace_pipeline_waiting(
                    "2/4 룰 학습", self.agent_model_ids["rule_learner"]
                )
                raw_rule_learning = self._generate_with_prompt(
                    req,
                    image_path,
                    _rule_learner_prompt(
                        req,
                        self.profile,
                        self._history,
                        perception,
                        self._rule_memory,
                        outcome,
                        self._pending_rule_outcomes,
                    ),
                    agent_name="rule_learner",
                    include_image=False,
                )
                try:
                    rule_memory = _normalize_rule_memory(
                        _parse_model_json(raw_rule_learning)
                    )
                except ValueError as rule_error:
                    print(
                        "[PIPELINE] rule JSON was incomplete; preserving saved rules and "
                        f"continuing: {rule_error}",
                        flush=True,
                    )
                else:
                    self._rule_memory = rule_memory
                    self._save_rule_memory()
                    self._verified_outcomes_since_rule_learning = 0
                    self._pending_rule_outcomes.clear()
                _trace_pipeline_value("2/4 모델이 작성한 룰 메모리", self._rule_memory)
            else:
                print(
                    "[PIPELINE] 2/4 룰 학습 보류 "
                    f"({self._verified_outcomes_since_rule_learning}/"
                    f"{self.rule_learning_interval} verified outcomes)",
                    flush=True,
                )
            stage_timings_ms["rule_learning"] = _trace_pipeline_timing(
                req.frame_id,
                "rule_learning",
                stage_started,
                status="completed" if rule_learning_due else "skipped",
            )

            active_stage = "planner"
            stage_started = time.perf_counter()
            if self._viewport_recovery is not None:
                plan = _viewport_recovery_plan(self._viewport_recovery)
                raw_plan = json.dumps(plan, ensure_ascii=False)
                print(
                    "[PIPELINE] 3/4 viewport recovery active; suppressing normal gameplay plan",
                    flush=True,
                )
            else:
                _trace_pipeline_waiting("3/4 룰 기반 계획", self.agent_model_ids["planner"])
                planner_prompt = _planner_prompt(
                    req,
                    self.profile,
                    self._history,
                    perception,
                    self._rule_memory,
                    outcome,
                    self._interaction_memory,
                    self._stage_progress,
                )
                raw_plan, plan = self._generate_json_with_retry(
                    req,
                    image_path,
                    planner_prompt,
                    agent_name="planner",
                    include_image=True,
                )
                repeated_stage = _plan_reenters_completed_stage(
                    plan, self._stage_progress
                )
                if repeated_stage is not None:
                    print(
                        f"[PROGRESS] rejected completed stage target {repeated_stage}; replanning",
                        flush=True,
                    )
                    replan_prompt = (
                        planner_prompt
                        + "\n\nVALIDATION RETRY: Your previous plan selected the already "
                        f"completed stage {repeated_stage}. Choose a visible uncompleted "
                        "progress node instead. A new series may restart its ordinal at 1. "
                        "Do not select a completed stage unless replay is explicitly required."
                    )
                    raw_plan, plan = self._generate_json_with_retry(
                        req,
                        image_path,
                        replan_prompt,
                        agent_name="planner",
                        include_image=True,
                    )
                    if _plan_reenters_completed_stage(plan, self._stage_progress) is not None:
                        plan = {
                            "screen_mode": "progress_ui",
                            "current_goal": "Avoid completed stage replay",
                            "strategy": "Reobserve available uncompleted progress nodes",
                            "desired_action": "wait",
                            "target_description": "none",
                            "success_check": "Uncompleted node becomes identifiable",
                            "risk": "none",
                            "confidence": 0.0,
                        }
                        raw_plan = json.dumps(plan, ensure_ascii=False)
            stage_timings_ms["planning"] = _trace_pipeline_timing(
                req.frame_id,
                "planning",
                stage_started,
                status="viewport_recovery" if self._viewport_recovery is not None else "completed",
            )
            _trace_pipeline_value("3/4 룰을 보고 세운 계획", plan)

            active_stage = "policy"
            stage_started = time.perf_counter()
            _trace_pipeline_waiting("4/4 실제 행동 결정", self.agent_model_ids["policy"])
            raw_policy, policy = self._generate_json_with_retry(
                req,
                image_path,
                _policy_prompt(
                    req,
                    self.profile,
                    self._history,
                    perception,
                    plan,
                    outcome,
                    self._interaction_memory,
                ),
                agent_name="policy",
                include_image=True,
            )
            action = _normalize_action(policy.get("action", {}), req.screen)
            if self._viewport_recovery is not None and _is_viewport_recovery_plan(plan):
                self._record_recovery_policy_action(action)
            confidence = float(policy.get("confidence", plan.get("confidence", 0.0)))
            if _repeats_blocked_action(action, outcome):
                action = {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "outcome verifier blocked retry of failed action",
                }
                confidence = 0.0
            if (
                action.get("type") in {"tap", "double_tap", "long_press"}
                and action.get("x") == 0
                and action.get("y") == 0
                and confidence <= 0.05
            ):
                action = {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "policy returned an unsafe placeholder coordinate",
                }
            stage_timings_ms["action_decision"] = _trace_pipeline_timing(
                req.frame_id, "action_decision", stage_started
            )
            _trace_pipeline_value(
                "4/4 최종 행동",
                {
                    "intent": policy.get("intent", ""),
                    "confidence": confidence,
                    "action": action,
                },
            )

            latency_ms = int((time.perf_counter() - started) * 1000)
            stage_timings_ms["pipeline_total"] = latency_ms
            _trace_pipeline_timing_summary(req.frame_id, stage_timings_ms)
            if board_signature and action.get("type") in {
                "tap", "double_tap", "swipe", "wait", "noop"
            }:
                fallback = _fallback_decision(
                    self.profile,
                    image_path,
                    req,
                    raw_policy,
                    latency_ms,
                    self._history,
                    image_fingerprint,
                    board_signature,
                    self._failed_swipes_by_board,
                    self._failed_cells_by_board,
                )
                if fallback is not None and fallback.get("action", {}).get("type") == "swipe":
                    fallback["model_name"] = f"{self._pipeline_model_name()}+fallback"
                    fallback["pipeline"] = {
                        "perception": perception,
                        "rule_memory": self._rule_memory,
                        "plan": plan,
                        "policy": policy,
                        "previous_outcome": outcome,
                        "interaction_memory": self._interaction_memory,
                        "raw_perception": raw_perception,
                        "raw_rule_learning": raw_rule_learning,
                        "raw_plan": raw_plan,
                        "raw_policy": raw_policy,
                        "rule_learning_due": rule_learning_due,
                        "rule_learning_progress": self._verified_outcomes_since_rule_learning,
                        "stage_timings_ms": stage_timings_ms,
                    }
                    return fallback

            return {
                "observation_summary": str(
                    policy.get("observation_summary")
                    or perception.get("screen_state")
                    or perception.get("summary")
                    or ""
                ),
                "intent": str(
                    policy.get("intent")
                    or plan.get("next_goal")
                    or plan.get("strategy")
                    or ""
                ),
                "confidence": confidence,
                "action": action,
                "model_name": self._pipeline_model_name(),
                "latency_ms": latency_ms,
                "raw_text": raw_policy,
                "image_fingerprint": image_fingerprint,
                "board_signature": board_signature,
                "pipeline": {
                    "perception": perception,
                    "rule_memory": self._rule_memory,
                    "plan": plan,
                    "policy": policy,
                    "previous_outcome": outcome,
                    "interaction_memory": self._interaction_memory,
                    "raw_perception": raw_perception,
                    "raw_rule_learning": raw_rule_learning,
                    "raw_plan": raw_plan,
                    "raw_policy": raw_policy,
                    "rule_learning_due": rule_learning_due,
                    "rule_learning_progress": self._verified_outcomes_since_rule_learning,
                    "stage_progress": self._stage_progress,
                    "stage_timings_ms": stage_timings_ms,
                },
            }
        except ValueError as exc:
            failed_stage_key = {
                "perception": "perception",
                "rule_learner": "rule_learning",
                "planner": "planning",
                "policy": "action_decision",
            }.get(active_stage, active_stage)
            if failed_stage_key not in stage_timings_ms:
                stage_timings_ms[failed_stage_key] = _trace_pipeline_timing(
                    req.frame_id, failed_stage_key, stage_started, status="error"
                )
            _trace_pipeline_value(
                "PIPELINE 오류",
                {"stage": active_stage, "error": str(exc)},
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            stage_timings_ms["pipeline_total"] = latency_ms
            _trace_pipeline_timing_summary(req.frame_id, stage_timings_ms)
            return {
                "observation_summary": "pipeline parse error",
                "intent": "모델 단계 응답 오류로 잠시 기다립니다.",
                "confidence": 0.0,
                "action": {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "pipeline output was not valid JSON",
                },
                "model_name": self._pipeline_model_name(),
                "latency_ms": latency_ms,
                "raw_text": raw_policy or raw_plan or raw_perception,
                "error": str(exc),
                "pipeline": {
                    "perception": perception,
                    "previous_outcome": outcome,
                    "interaction_memory": self._interaction_memory,
                    "plan": plan,
                    "raw_perception": raw_perception,
                    "raw_rule_learning": raw_rule_learning,
                    "raw_plan": raw_plan,
                    "raw_policy": raw_policy,
                    "rule_memory": self._rule_memory,
                    "stage_timings_ms": stage_timings_ms,
                },
            }

    def _load_rule_memory(self) -> dict[str, Any]:
        if self.rule_memory_path is None or not self.rule_memory_path.exists():
            return _empty_rule_memory()
        try:
            document = json.loads(self.rule_memory_path.read_text(encoding="utf-8"))
            raw_memory = document.get("rule_memory", document)
            memory = _normalize_rule_memory(dict(raw_memory))
            print(f"[RULES] loaded {self.rule_memory_path}", flush=True)
            return memory
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load rule memory {self.rule_memory_path}: {exc}"
            ) from exc

    def _save_rule_memory(self) -> None:
        if self.rule_memory_path is None:
            return
        self.rule_memory_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "updated_at": time.time(),
            "profile": self.profile.get("name") if self.profile else None,
            "rule_memory": self._rule_memory,
        }
        temporary = self.rule_memory_path.with_suffix(self.rule_memory_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.rule_memory_path)
        print(f"[RULES] saved {self.rule_memory_path}", flush=True)

    def _remember(self, req: DecideRequest, decision: dict[str, Any]) -> None:
        action = decision.get("action", {})
        entry = {
                "frame_id": req.frame_id,
                "summary": decision.get("observation_summary", ""),
                "intent": decision.get("intent", ""),
                "action": {
                    "type": action.get("type"),
                    "x": action.get("x"),
                    "y": action.get("y"),
                    "x2": action.get("x2"),
                    "y2": action.get("y2"),
                    "reason": action.get("reason"),
                },
                "image_fingerprint": decision.get("image_fingerprint"),
                "board_signature": decision.get("board_signature"),
                "perception": decision.get("pipeline", {}).get("perception"),
                "plan": decision.get("pipeline", {}).get("plan"),
                "outcome": None,
            }
        self._history.append(entry)
        self._history = self._history[-30:]
        if action.get("type") in {
            "tap", "double_tap", "swipe", "long_press", "back"
        }:
            self._pending_interactions.append(entry)
            self._pending_interactions = self._pending_interactions[-20:]

    def _discard_skipped_interaction(self, req: DecideRequest) -> None:
        """Remove an action the runtime skipped because its source frame became stale."""
        skipped = req.metadata.get("skipped_action_frame_id")
        if skipped is None:
            return
        try:
            skipped_frame_id = int(skipped)
        except (TypeError, ValueError):
            return
        self._pending_interactions = [
            item for item in self._pending_interactions
            if item.get("frame_id") != skipped_frame_id
        ]
        for item in self._history:
            if item.get("frame_id") == skipped_frame_id:
                item["action"] = {
                    "type": "noop",
                    "x": None,
                    "y": None,
                    "x2": None,
                    "y2": None,
                    "reason": "stale_observation: runtime skipped outdated action",
                }
                item["outcome"] = None
        print(
            f"[PIPELINE] discarded skipped stale action for frame {skipped_frame_id}",
            flush=True,
        )

    def _verify_previous_outcome(
        self,
        req: DecideRequest,
        image_path: Path,
        current_perception: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Judge game progress separately from low-level control execution."""
        if not self._pending_interactions:
            return None
        previous = self._pending_interactions[0]
        previous_plan = previous.get("plan") or {}
        raw = self._generate_with_prompt(
            req,
            image_path,
            _outcome_verifier_prompt(
                req,
                previous,
                current_perception,
                previous_plan,
                self.profile,
            ),
            agent_name="rule_learner",
            include_image=False,
        )
        try:
            outcome = _normalize_action_outcome(_parse_model_json(raw))
        except ValueError as exc:
            outcome = {
                "status": "inconclusive",
                "expected_result": str(previous_plan.get("success_check", "")),
                "observed_change": "verifier output could not be parsed",
                "evidence": str(exc)[:160],
                "failure_reason": "",
                "retry_recommendation": "reobserve",
                "confidence": 0.0,
            }
        outcome["frame_id"] = previous.get("frame_id")
        outcome["action"] = dict(previous.get("action") or {})
        self._update_viewport_recovery(previous, outcome)
        previous["outcome"] = outcome
        self._interaction_memory.append(outcome)
        self._interaction_memory = self._interaction_memory[-20:]
        self._pending_rule_outcomes.append(outcome)
        self._verified_outcomes_since_rule_learning += 1
        self._pending_interactions.pop(0)
        return outcome

    def _update_viewport_recovery(
        self,
        previous: dict[str, Any],
        outcome: dict[str, Any],
    ) -> None:
        plan = previous.get("plan") or {}
        if self._viewport_recovery is not None and _is_viewport_recovery_plan(plan):
            self._viewport_recovery["attempts"] = int(
                self._viewport_recovery.get("attempts", 0)
            ) + 1
            if outcome.get("status") == "success":
                print("[RECOVERY] viewport restored; resuming normal planning", flush=True)
                self._viewport_recovery = None
            elif self._viewport_recovery["attempts"] >= 4:
                print(
                    "[RECOVERY] maximum pan attempts reached; resuming from current view",
                    flush=True,
                )
                self._viewport_recovery = None
            return

        action = previous.get("action") or {}
        if not _outcome_reports_unintended_viewport_motion(outcome, action, plan):
            return
        self._viewport_recovery = {
            "trigger_frame": previous.get("frame_id"),
            "attempts": 0,
            "stalls": 0,
            "direction": _opposite_swipe_direction(action),
            "evidence": str(outcome.get("observed_change", ""))[:240],
        }
        self._failed_swipes_by_board.clear()
        self._failed_cells_by_board.clear()
        print(
            "[RECOVERY] unintended viewport motion detected; normal planning paused",
            flush=True,
        )

    def _record_recovery_policy_action(self, action: dict[str, Any]) -> None:
        if self._viewport_recovery is None:
            return
        if action.get("type") not in {"wait", "noop"}:
            self._viewport_recovery["stalls"] = 0
            return
        stalls = int(self._viewport_recovery.get("stalls", 0)) + 1
        self._viewport_recovery["stalls"] = stalls
        if stalls >= 3:
            print(
                "[RECOVERY] no executable recovery target after 3 frames; "
                "resuming from current view",
                flush=True,
            )
            self._viewport_recovery = None

    def _record_previous_action_outcome(
        self,
        req: DecideRequest,
        current_board_signature: str | None,
    ) -> None:
        if not current_board_signature or not req.previous_action or not self._history:
            return
        previous = req.previous_action
        if previous.type not in {"swipe", "tap", "double_tap"}:
            return
        if previous.x is None or previous.y is None:
            return
        last = self._history[-1]
        previous_board_signature = last.get("board_signature")
        if previous_board_signature != current_board_signature:
            return
        failed_cells = self._failed_cells_by_board.setdefault(current_board_signature, set())
        failed_cells.add(_coordinate_cell_bucket_signature(int(previous.x), int(previous.y)))
        if previous.type in {"tap", "double_tap"}:
            return
        if previous.x2 is None or previous.y2 is None:
            return
        signature = _coordinate_swipe_signature(
            int(previous.x),
            int(previous.y),
            int(previous.x2),
            int(previous.y2),
        )
        failed = self._failed_swipes_by_board.setdefault(current_board_signature, set())
        failed.add(signature)
        failed.add(
            _coordinate_bucket_signature(
                int(previous.x),
                int(previous.y),
                int(previous.x2),
                int(previous.y2),
            )
        )
        failed_cells.add(_coordinate_cell_bucket_signature(int(previous.x2), int(previous.y2)))
        if len(self._failed_swipes_by_board) > 20:
            oldest_key = next(iter(self._failed_swipes_by_board))
            self._failed_swipes_by_board.pop(oldest_key, None)
            self._failed_cells_by_board.pop(oldest_key, None)

    def _mock_decide(self, req: DecideRequest) -> dict[str, Any]:
        if req.frame_id % 2 == 0:
            action = {
                "type": "tap",
                "x": req.screen.width // 2,
                "y": req.screen.height // 2,
                "duration_ms": 80,
            }
            intent = "mock center tap"
        else:
            action = {"type": "wait", "duration_ms": 300, "reason": "mock wait"}
            intent = "mock wait for next frame"
        return {
            "observation_summary": f"mock frame {req.frame_id}",
            "intent": intent,
            "confidence": 1.0,
            "action": action,
            "model_name": "mock_vlm_server",
        }

    def _load(self, model_id: str) -> dict[str, Any]:
        cached = self._model_cache.get(model_id)
        if cached is not None:
            self._loaded = True
            return cached
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - depends on local VLM env
            raise RuntimeError(
                "Qwen-VL dependencies are missing. Activate the VLM environment or run "
                "with --mock."
            ) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        self._process_vision_info = process_vision_info
        cached = {"model": model, "processor": processor}
        self._model_cache[model_id] = cached
        self._loaded = True
        return cached

    def _generate(self, req: DecideRequest, image_path: Path) -> str:
        prompt = _decision_prompt(req, self.profile, self._history)
        return self._generate_with_prompt(
            req,
            image_path,
            prompt,
            agent_name="policy",
            include_image=True,
        )

    def _generate_with_prompt(
        self,
        req: DecideRequest,
        image_path: Path,
        prompt: str,
        agent_name: AgentName,
        include_image: bool,
    ) -> str:
        model_ref = self.agent_model_refs[agent_name]
        if model_ref.provider in HOSTED_PROVIDERS:
            return _generate_hosted(
                model_ref=model_ref,
                prompt=prompt,
                image_path=image_path,
                include_image=include_image,
                max_tokens=self.max_new_tokens,
                timeout_s=120.0,
            )

        model_id = model_ref.model
        runtime = self._load(model_id)
        model = runtime["model"]
        processor = runtime["processor"]
        content: list[dict[str, Any]] = []
        if include_image:
            content.append(
                {
                    "type": "image",
                    "image": str(image_path),
                    "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if include_image:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generation_kwargs.update({"do_sample": False})

        generated_ids = model.generate(**inputs, **generation_kwargs)
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        return str(decoded).strip()

    def _generate_json_with_retry(
        self,
        req: DecideRequest,
        image_path: Path,
        prompt: str,
        agent_name: AgentName,
        include_image: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Generate structured output, retrying once after prose/truncation."""
        raw = self._generate_with_prompt(
            req,
            image_path,
            prompt,
            agent_name=agent_name,
            include_image=include_image,
        )
        try:
            return raw, _parse_model_json(raw)
        except ValueError as exc:
            print(
                f"[PIPELINE] {agent_name} JSON invalid; retrying once with compact-output constraint: {exc}",
                flush=True,
            )

        retry_prompt = (
            prompt
            + "\n\nRETRY: Your previous response was invalid or truncated. "
            "Return the requested result again as exactly one compact JSON object. "
            "Start with { and end with }. Do not output analysis, reasoning, markdown, "
            "code fences, or text before or after the JSON. If uncertain, encode that "
            "uncertainty inside the requested JSON instead of explaining it."
        )
        retry_raw = self._generate_with_prompt(
            req,
            image_path,
            retry_prompt,
            agent_name=agent_name,
            include_image=include_image,
        )
        return retry_raw, _parse_model_json(retry_raw)

    def _pipeline_model_name(self) -> str:
        compact = ",".join(
            f"{agent}={self.agent_model_ids[agent]}" for agent in PIPELINE_AGENTS
        )
        return f"pipeline({compact})"

    def _materialize_image(self, req: DecideRequest) -> Path:
        if req.image_base64:
            raw = base64.b64decode(req.image_base64)
            fh = tempfile.NamedTemporaryFile(
                prefix="gameagent_frame_",
                suffix=".png",
                delete=False,
            )
            with fh:
                fh.write(raw)
            return Path(fh.name)
        if req.image_path:
            return Path(req.image_path)
        raise HTTPException(status_code=400, detail="image_base64 or image_path is required")


def _trace_pipeline_frame(req: DecideRequest, model_ids: dict[str, str]) -> None:
    previous = req.previous_action.model_dump() if req.previous_action else None
    print("\n" + "=" * 78, flush=True)
    print(f"[PIPELINE] frame={req.frame_id} screen={req.screen.width}x{req.screen.height}", flush=True)
    print(
        "[PIPELINE] models="
        + ", ".join(f"{name}={model_ids[name]}" for name in PIPELINE_AGENTS),
        flush=True,
    )
    print(
        "[PIPELINE] 이전 행동="
        + json.dumps(previous, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _trace_pipeline_waiting(label: str, model_id: str) -> None:
    print(f"\n[PIPELINE] {label} 요청 중... model={model_id}", flush=True)


def _trace_pipeline_value(label: str, value: Any) -> None:
    print(f"[PIPELINE] {label}", flush=True)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def _trace_pipeline_timing(
    frame_id: int,
    stage: str,
    started: float,
    *,
    status: str = "completed",
) -> int:
    duration_ms = int((time.perf_counter() - started) * 1000)
    ended_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(
        f"[TIMING] timestamp={ended_at} frame={frame_id} stage={stage} "
        f"duration_ms={duration_ms} status={status}",
        flush=True,
    )
    return duration_ms


def _trace_pipeline_timing_summary(
    frame_id: int, stage_timings_ms: dict[str, int]
) -> None:
    values = " ".join(
        f"{stage}={duration_ms}ms"
        for stage, duration_ms in stage_timings_ms.items()
    )
    print(f"[TIMING] frame={frame_id} summary {values}", flush=True)


def _decision_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    history_prompt = _history_prompt(history)
    return f"""
You are controlling a mobile game through touch actions only.
Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Recent agent history:
{history_prompt}

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Look at the screenshot and choose exactly one next action.
Use only facts and explicit instructions visible in the screenshot and the recorded
history. Do not use prior knowledge about games or common UI conventions.

Decision policy:
- Act only when visible evidence or recorded action-result evidence supports the action.
- Otherwise choose wait.
- Coordinates must use the current screen coordinate system: x in [0, {req.screen.width - 1}], y in [0, {req.screen.height - 1}].
- Do not use placeholder coordinates such as x=0,y=0. Only use x=0,y=0 if the real visible target is exactly at the top-left corner.
- If confidence is 0.0 or very low, the action must be wait or noop, not tap/swipe.
- Use action type "tap" for touching/clicking a button. Never use "click".

Allowed actions:
- tap: {{"type":"tap","x":int,"y":int,"duration_ms":80}}
- double_tap: {{"type":"double_tap","x":int,"y":int,"duration_ms":120}}
- swipe: {{"type":"swipe","x":int,"y":int,"x2":int,"y2":int,"duration_ms":int}}
- long_press: {{"type":"long_press","x":int,"y":int,"duration_ms":int}}
- wait: {{"type":"wait","duration_ms":int,"reason":str}}
- back: {{"type":"back","reason":str}}
- noop: {{"type":"noop","reason":str}}

Return only valid compact JSON. Do not include markdown.
Keep observation_summary under 12 words.
Keep intent as one short Korean or English sentence that says where you tap and why.
The JSON schema is:
{{
  "observation_summary": "brief state",
  "intent": "Wait because no safe action is visible.",
  "confidence": 0.2,
  "action": {{"type": "wait", "duration_ms": 800, "reason": "no safe action visible"}}
}}
""".strip()


def _outcome_verifier_prompt(
    req: DecideRequest,
    previous: dict[str, Any],
    current_perception: dict[str, Any],
    previous_plan: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> str:
    return f"""
You are the outcome verifier for a mobile game agent. Judge whether the previous
action achieved its intended in-game result. Control execution is not game success.
Use only the before/after observations and expected visible result below.

Screen size: {req.screen.width}x{req.screen.height}.
Previous perception: {json.dumps(previous.get("perception"), ensure_ascii=False)}
Previous action actually executed: {json.dumps(previous.get("action"), ensure_ascii=False)}
Previous intent: {json.dumps(previous.get("intent", ""), ensure_ascii=False)}
Expected visible result: {json.dumps(previous_plan.get("success_check", ""), ensure_ascii=False)}
Current perception: {json.dumps(current_perception, ensure_ascii=False)}

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Status rules:
- success: expected progress is visibly satisfied.
- failure: the stable screen visibly contradicts the expected result or shows rejection/locked feedback.
- partial: useful progress occurred but the expected result is not fully satisfied.
- inconclusive: animation, insufficient evidence, or observations cannot be compared safely.
- Never call an action successful merely because the touch command executed.
- Use do_not_retry only for a failure under the same visible state and target.

Return only valid compact JSON:
{{
  "status": "success | failure | partial | inconclusive",
  "expected_result": "brief expected result",
  "observed_change": "brief literal before/after change",
  "evidence": "brief visible evidence",
  "failure_reason": "brief reason or empty",
  "retry_recommendation": "allow | change_target | change_method | do_not_retry | reobserve",
  "confidence": 0.0
}}
""".strip()


def _perception_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    previous_perception = history[-1].get("perception") if history else None
    return f"""
You are the VLM perception module for a mobile game agent.
Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Previous perception: {json.dumps(previous_perception, ensure_ascii=False)}.

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Describe only facts visible in the screenshot. Compare against the supplied
previous perception when available. Do not choose an action, infer a
game rule, make a plan, or use prior knowledge about how games usually work.

Return only valid compact JSON:
{{
  "summary": "brief literal visual description",
  "visible_text": "visible text or none",
  "objects": "visible objects and approximate locations",
  "screen_change_from_previous": "visible change or unknown",
  "confidence": 0.0
}}
""".strip()


def _rule_learner_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    perception: dict[str, Any],
    rule_memory: dict[str, Any],
    previous_outcome: dict[str, Any] | None = None,
    interaction_memory: list[dict[str, Any]] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    return f"""
You are the rule learner module for a mobile game agent.
Your only job is to maintain compact game-rule memory.
Do not choose actions. Do not choose coordinates. Do not make the next plan.

Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Recent agent history:
{_history_prompt(history)}

Perception JSON:
{json.dumps(perception, ensure_ascii=False)}

Verified previous action outcome:
{json.dumps(previous_outcome, ensure_ascii=False)}

Recent structured interaction outcomes:
{_interaction_memory_prompt(interaction_memory)}

Current rule memory:
{json.dumps(rule_memory, ensure_ascii=False)}

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Rule learning rules:
- Treat the verified outcome as the authoritative result of the previous game action.
- Update rules only from visible tutorial/objective text or verified action-result evidence.
- Preserve useful existing rules unless contradicted.
- Increase confidence only when the recent history shows evidence.
- Keep uncertain ideas under hypotheses, not confirmed_rules.
- Do not use prior knowledge about games or common UI conventions.
- When no confirmed rule supports progress, create one falsifiable hypothesis from
  visible objects, text, spatial patterns, or observed screen changes.
- A new hypothesis must name what visible interaction could test it and what
  subsequent screen change would count as evidence.
- Keep this memory compact so future planner prompts stay useful.
- Do not output any action, desired_action, target coordinate, or plan.

Return only valid compact JSON:
{{
  "objective": "known objective or unknown",
  "confirmed_rules": [
    {{"rule": "under 14 words", "confidence": 0.0, "evidence": "under 12 words"}}
  ],
  "hypotheses": [
    {{"rule": "under 14 words", "confidence": 0.0, "needs_test": true, "evidence": "under 12 words"}}
  ],
  "failed_patterns": [
    {{"pattern": "under 12 words", "evidence": "under 12 words"}}
  ],
  "updated_reason": "under 16 words"
}}
""".strip()


def _planner_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    perception: dict[str, Any],
    rule_memory: dict[str, Any],
    previous_outcome: dict[str, Any] | None = None,
    interaction_memory: list[dict[str, Any]] | None = None,
    stage_progress: dict[str, Any] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    return f"""
You are the planner module for a mobile game agent.
You do not choose exact coordinates. Decide the current objective and action strategy.
Do not update rule memory. Do not read pixels directly beyond the supplied perception.

Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Recent agent history:
{_history_prompt(history)}

Perception JSON:
{json.dumps(perception, ensure_ascii=False)}

Learned rule memory:
{json.dumps(rule_memory, ensure_ascii=False)}

Verified previous action outcome:
{json.dumps(previous_outcome, ensure_ascii=False)}

Recent structured interaction outcomes:
{_interaction_memory_prompt(interaction_memory)}

Structured stage progress for this run:
{json.dumps(stage_progress or _empty_stage_progress(), ensure_ascii=False)}

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Planning rules:
- Base the plan only on the supplied perception and learned rule memory.
- Do not use prior knowledge about games, UI conventions, or likely button meanings.
- A confirmed rule may be used directly.
- When no confirmed rule applies, select one hypothesis grounded in a currently
  visible object and plan one reversible experiment to test it.
- State the tested hypothesis and expected visible result in strategy/success_check.
- Do not repeat an experiment when history already shows that it produced no change.
- If a matching recent outcome is failure, change the target or interaction method.
- If the previous outcome is inconclusive, reobserve instead of declaring a rule.
- Do not select a stage identity listed under completed. Prefer a visible uncompleted
  node in the same series with a higher ordinal. If no such node exists, a visible
  uncompleted node in a different series may restart at ordinal 1.
- Replaying a completed stage is allowed only when visible text explicitly requires it.
- Choose wait only when there is no visible target grounded in a rule or hypothesis.
- Do not output tap/swipe coordinates.

Return only valid compact JSON:
{{
  "screen_mode": "gameplay | progress_ui | popup | loading | unsafe | unknown",
  "current_goal": "under 8 words",
  "strategy": "under 12 words",
  "desired_action": "tap | double_tap | swipe | long_press | wait | back | noop",
  "target_description": "under 12 words, no exact coordinates",
  "success_check": "under 10 words",
  "risk": "under 8 words or none",
  "confidence": 0.0
}}
""".strip()


def _policy_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    perception: dict[str, Any],
    plan: dict[str, Any],
    previous_outcome: dict[str, Any] | None = None,
    interaction_memory: list[dict[str, Any]] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    return f"""
You are the low-level touch policy module.
Choose exactly one executable touch action from the screenshot, perception, and plan.
Do not update rules, reinterpret the game objective, or create a new plan.

Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Recent agent history:
{_history_prompt(history)}

Perception JSON:
{json.dumps(perception, ensure_ascii=False)}

Planner JSON:
{json.dumps(plan, ensure_ascii=False)}

Verified previous action outcome:
{json.dumps(previous_outcome, ensure_ascii=False)}

Recent structured interaction outcomes:
{_interaction_memory_prompt(interaction_memory)}

Authoritative game profile and safety rules:
{_profile_prompt(profile)}

Policy rules:
- Execute only planner.desired_action and planner.target_description.
- Do not reproduce an action marked do_not_retry under a visibly unchanged state.
- Do not use prior knowledge about games, UI conventions, or likely button meanings.
- If the planner's target is not visibly identifiable, return wait.
- Coordinates must be exact screen coordinates: x in [0, {req.screen.width - 1}], y in [0, {req.screen.height - 1}].
- Never use placeholder coordinates such as x=0,y=0.
- For tap/double_tap/long_press, choose the center of the visible target.
- For swipe, use x,y as the drag start and x2,y2 as the drag end.
- If exact coordinates are uncertain, return wait instead of inventing a coordinate.
- If confidence is below 0.2, return wait or noop.

Allowed actions:
- tap: {{"type":"tap","x":int,"y":int,"duration_ms":80}}
- double_tap: {{"type":"double_tap","x":int,"y":int,"duration_ms":120}}
- swipe: {{"type":"swipe","x":int,"y":int,"x2":int,"y2":int,"duration_ms":int}}
- long_press: {{"type":"long_press","x":int,"y":int,"duration_ms":int}}
- wait: {{"type":"wait","duration_ms":int,"reason":str}}
- back: {{"type":"back","reason":str}}
- noop: {{"type":"noop","reason":str}}

Return only valid compact JSON:
{{
  "observation_summary": "under 8 words",
  "intent": "under 16 words saying what and why",
  "confidence": 0.2,
  "action": {{"type": "wait", "duration_ms": 800, "reason": "no safe coordinate"}}
}}
""".strip()


def _empty_rule_memory() -> dict[str, Any]:
    return {
        "objective": "unknown",
        "confirmed_rules": [],
        "hypotheses": [],
        "failed_patterns": [],
        "updated_reason": "empty memory",
    }


def _normalize_action_outcome(data: dict[str, Any]) -> dict[str, Any]:
    status = str(data.get("status", "inconclusive")).strip().lower()
    if status not in {"success", "failure", "partial", "inconclusive"}:
        status = "inconclusive"
    retry = str(data.get("retry_recommendation", "reobserve")).strip().lower()
    if retry not in {"allow", "change_target", "change_method", "do_not_retry", "reobserve"}:
        retry = "reobserve"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "status": status,
        "expected_result": str(data.get("expected_result", ""))[:240],
        "observed_change": str(data.get("observed_change", ""))[:320],
        "evidence": str(data.get("evidence", ""))[:240],
        "failure_reason": str(data.get("failure_reason", ""))[:240],
        "retry_recommendation": retry,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _repeats_blocked_action(
    action: dict[str, Any],
    outcome: dict[str, Any] | None,
) -> bool:
    if not outcome or outcome.get("status") != "failure":
        return False
    if outcome.get("retry_recommendation") != "do_not_retry":
        return False
    previous = outcome.get("action") or {}
    keys = ("type", "x", "y", "x2", "y2")
    return all(action.get(key) == previous.get(key) for key in keys)


def _normalize_rule_memory(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": _short_text(data.get("objective"), "unknown", 120),
        "confirmed_rules": _normalize_rule_items(
            data.get("confirmed_rules"),
            allowed_keys=("rule", "confidence", "evidence"),
            limit=8,
        ),
        "hypotheses": _normalize_rule_items(
            data.get("hypotheses"),
            allowed_keys=("rule", "confidence", "needs_test", "evidence"),
            limit=8,
        ),
        "failed_patterns": _normalize_rule_items(
            data.get("failed_patterns"),
            allowed_keys=("pattern", "evidence"),
            limit=6,
        ),
        "updated_reason": _short_text(data.get("updated_reason"), "", 160),
    }


def _normalize_rule_items(
    value: Any,
    allowed_keys: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for raw_item in value[:limit]:
        if isinstance(raw_item, str):
            key = "rule" if "rule" in allowed_keys else allowed_keys[0]
            raw_item = {key: raw_item}
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, Any] = {}
        for key in allowed_keys:
            if key not in raw_item:
                continue
            if key == "confidence":
                item[key] = _clamped_float(raw_item[key], 0.0, 0.0, 1.0)
            elif key == "needs_test":
                item[key] = bool(raw_item[key])
            else:
                item[key] = _short_text(raw_item[key], "", 160)
        if item:
            items.append(item)
    return items


def _clamped_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


def _short_text(value: Any, default: str, limit: int) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text[:limit]


def _load_profile(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    raw = profile_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
        return dict(loaded or {})
    except ModuleNotFoundError:
        return json.loads(raw)


def _merge_profiles(
    global_profile: dict[str, Any] | None,
    game_profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge common rules first, then append/override with game-specific values."""
    if not global_profile and not game_profile:
        return None
    merged: dict[str, Any] = dict(global_profile or {})
    for key, game_value in (game_profile or {}).items():
        common_value = merged.get(key)
        if isinstance(common_value, list) and isinstance(game_value, list):
            merged[key] = [*common_value, *game_value]
        elif isinstance(common_value, dict) and isinstance(game_value, dict):
            merged[key] = {**common_value, **game_value}
        else:
            merged[key] = game_value
    return merged


def _load_profile_with_global_rules(
    game_profile_path: str | Path | None,
    global_rule_path: str | Path = DEFAULT_GLOBAL_RULE_PATH,
    tutorial_model_ref: ModelRef | None = None,
    auto_tutorial: bool = False,
) -> dict[str, Any] | None:
    global_path = Path(global_rule_path).resolve()
    game_path = Path(game_profile_path).resolve() if game_profile_path else None
    global_profile = _load_profile(global_path) if global_path.exists() else None
    if game_path == global_path:
        return global_profile
    game_profile = _load_profile(game_path) if game_path else None
    if game_profile and game_path and auto_tutorial:
        tutorial_prior = _load_or_create_tutorial_prior(
            game_path,
            tutorial_model_ref,
        )
        game_profile = _merge_profiles(game_profile, tutorial_prior)
    return _merge_profiles(global_profile, game_profile)


def _load_or_create_tutorial_prior(
    game_profile_path: Path,
    model_ref: ModelRef | None,
) -> dict[str, Any] | None:
    """Load a game prior by convention, generating it from a demo pair when needed."""
    game_id = game_profile_path.stem
    prior_path = game_profile_path.parent / "tutorials" / f"{game_id}.yaml"
    existing = _load_profile(prior_path) if prior_path.exists() else None
    existing_rules = existing.get("demonstration_rules") if existing else None
    force_regenerate = bool(existing.get("regenerate")) if existing else False
    if existing and not force_regenerate and (
        existing.get("status") == "ready"
        or isinstance(existing_rules, list) and bool(existing_rules)
    ):
        print(f"[TUTORIAL] loaded prior {prior_path}", flush=True)
        return existing

    demonstration = _find_tutorial_demonstration(game_id)
    if demonstration is None:
        if not prior_path.exists():
            _write_tutorial_prior(
                prior_path,
                {
                    "status": "awaiting_demonstration",
                    "game_id": game_id,
                    "demonstration_rules": [],
                    "tutorial_sequence": [],
                    "gesture_priors": [],
                },
            )
            print(f"[TUTORIAL] created empty prior {prior_path}", flush=True)
        return existing

    video_path, actions_path = demonstration
    if model_ref is None or model_ref.provider not in HOSTED_PROVIDERS:
        print(
            f"[TUTORIAL] demo found for {game_id}, but no hosted API model is configured",
            flush=True,
        )
        return existing

    print(
        f"[TUTORIAL] generating {game_id} prior from {video_path.name} + {actions_path.name}",
        flush=True,
    )
    try:
        generated = _generate_tutorial_prior(
            game_id,
            video_path,
            actions_path,
            model_ref,
        )
    except Exception as exc:
        print(f"[TUTORIAL] prior generation failed: {type(exc).__name__}: {exc}", flush=True)
        return existing
    _write_tutorial_prior(prior_path, generated)
    print(f"[TUTORIAL] saved generated prior {prior_path}", flush=True)
    return generated


def _find_tutorial_demonstration(game_id: str) -> tuple[Path, Path] | None:
    canonical = PROJECT_ROOT / "tutorials" / game_id
    search_roots = [canonical]
    legacy_root = PROJECT_ROOT.parent
    if legacy_root.exists():
        candidates = [path for path in legacy_root.iterdir() if path.is_dir()]
        normalized_game = re.sub(r"[^a-z0-9]", "", game_id.lower())
        game_tokens = [token for token in re.split(r"[^a-z0-9]+", game_id.lower()) if token]
        scored: list[tuple[float, Path]] = []
        for path in candidates:
            name = re.sub(r"(?:tutorial|tuto)$", "", path.name.lower())
            normalized_name = re.sub(r"[^a-z0-9]", "", name)
            name_tokens = [token for token in re.split(r"[^a-z0-9]+", name) if token]
            scores = [difflib.SequenceMatcher(None, normalized_game, normalized_name).ratio()]
            scores.extend(
                difflib.SequenceMatcher(None, left, right).ratio()
                for left in game_tokens
                for right in name_tokens
            )
            score = max(scores)
            if score >= 0.7:
                scored.append((score, path))
        search_roots.extend(path for _, path in sorted(scored, reverse=True))

    for directory in search_roots:
        if not directory.is_dir():
            continue
        videos = sorted(directory.glob("*.mp4"))
        actions = sorted(directory.glob("*.json"))
        for video in videos:
            matching = next((item for item in actions if item.stem == video.stem), None)
            if matching:
                return video.resolve(), matching.resolve()
        if len(videos) == 1 and len(actions) == 1:
            return videos[0].resolve(), actions[0].resolve()
    return None


def _generate_tutorial_prior(
    game_id: str,
    video_path: Path,
    actions_path: Path,
    model_ref: ModelRef,
) -> dict[str, Any]:
    document = json.loads(actions_path.read_text(encoding="utf-8"))
    actions = list(document.get("actions") or [])
    if not actions:
        raise ValueError("tutorial action JSON has no actions")
    sample_count = min(12, len(actions))
    sample_indexes = sorted(
        {round(index * (len(actions) - 1) / max(sample_count - 1, 1)) for index in range(sample_count)}
    )
    selected = [actions[index] for index in sample_indexes]
    tap_bursts = _tutorial_tap_bursts(actions)
    compound_actions = _normalize_tutorial_compound_actions(actions)
    with tempfile.TemporaryDirectory(prefix="gameagent_tutorial_") as temporary:
        contact_path = Path(temporary) / "tutorial_contact.jpg"
        _build_tutorial_contact_sheet(video_path, selected, contact_path)
        compact_actions = [_compact_tutorial_action(action) for action in selected]
        prompt = f"""You analyze a verified human tutorial playthrough for game {game_id}.
The contact sheet contains before/after frames for selected actions in chronological order.
Selected action metadata:
{json.dumps(compact_actions, ensure_ascii=False)}
Detected same-position rapid tap bursts (the recorder stores each tap separately):
{json.dumps(tap_bursts, ensure_ascii=False)}
Normalized action timeline (same-position rapid tap pairs become double_tap):
{json.dumps(compound_actions, ensure_ascii=False)}

Infer only rules supported by visible transitions and normalized recorded actions. Describe objects and
UI states semantically; never prescribe recorded absolute coordinates. Separate the ordered
tutorial flow from general game rules and gesture timing. A normalized double_tap is an observed
input gesture, not a hardcoded rule for any object category; infer its target and effect only from
the corresponding visual before/after frames. Return strict JSON:
{{
  "demonstration_rules": ["concise verified rule"],
  "tutorial_sequence": ["ordered state -> action -> expected result"],
  "gesture_priors": ["timing or drag technique"],
  "demonstration_evidence_policy": [
    "Do not overturn this prior because of stale observations or overlay-blocked input"
  ]
}}
"""
        raw = _generate_hosted(
            model_ref=model_ref,
            prompt=prompt,
            image_path=contact_path,
            include_image=True,
            max_tokens=2048,
            timeout_s=120.0,
        )
    parsed = _parse_model_json(raw)
    prior: dict[str, Any] = {
        "status": "ready",
        "game_id": game_id,
        "generated_at": time.time(),
        "generated_from": {
            "video": str(video_path),
            "actions": str(actions_path),
            "model": model_ref.key,
            "action_count": len(actions),
        },
    }
    for key in (
        "demonstration_rules",
        "tutorial_sequence",
        "gesture_priors",
        "demonstration_evidence_policy",
    ):
        values = parsed.get(key)
        prior[key] = [str(value)[:500] for value in values] if isinstance(values, list) else []
    if not prior["demonstration_rules"]:
        raise ValueError("tutorial API returned no demonstration rules")
    return prior


def _compact_tutorial_action(action: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "index": action.get("index"),
        "type": action.get("type"),
        "video_start_ms": action.get("video_start_ms"),
        "video_end_ms": action.get("video_end_ms"),
        "duration_ms": action.get("duration_ms"),
    }
    for key in ("position", "start", "end"):
        point = action.get(key)
        if isinstance(point, dict):
            compact[key] = {"x_norm": point.get("x_norm"), "y_norm": point.get("y_norm")}
    return compact


def _tutorial_tap_bursts(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bursts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for action in actions:
        point = action.get("position")
        if action.get("type") != "tap" or not isinstance(point, dict):
            if len(current) >= 2:
                bursts.append(current)
            current = []
            continue
        if current:
            previous = current[-1]
            previous_point = previous["position"]
            interval = int(action.get("start_ms", 0)) - int(previous.get("start_ms", 0))
            dx = float(point.get("x_norm", 0)) - float(previous_point.get("x_norm", 0))
            dy = float(point.get("y_norm", 0)) - float(previous_point.get("y_norm", 0))
            if interval > 900 or (dx * dx + dy * dy) ** 0.5 > 0.03:
                if len(current) >= 2:
                    bursts.append(current)
                current = []
        current.append(action)
    if len(current) >= 2:
        bursts.append(current)
    return [
        {
            "first_index": burst[0].get("index"),
            "last_index": burst[-1].get("index"),
            "tap_count": len(burst),
            "duration_ms": int(burst[-1].get("start_ms", 0))
            - int(burst[0].get("start_ms", 0)),
            "position_norm": {
                "x": burst[0]["position"].get("x_norm"),
                "y": burst[0]["position"].get("y_norm"),
            },
        }
        for burst in bursts
    ]


def _normalize_tutorial_compound_actions(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert recorder-level rapid tap pairs into semantic double_tap actions."""
    normalized: list[dict[str, Any]] = []
    index = 0
    while index < len(actions):
        first = actions[index]
        if index + 1 < len(actions):
            second = actions[index + 1]
            first_point = first.get("position")
            second_point = second.get("position")
            if (
                first.get("type") == "tap"
                and second.get("type") == "tap"
                and isinstance(first_point, dict)
                and isinstance(second_point, dict)
            ):
                interval = int(second.get("start_ms", 0)) - int(first.get("start_ms", 0))
                dx = float(first_point.get("x_norm", 0)) - float(
                    second_point.get("x_norm", 0)
                )
                dy = float(first_point.get("y_norm", 0)) - float(
                    second_point.get("y_norm", 0)
                )
                if 0 <= interval <= 900 and (dx * dx + dy * dy) ** 0.5 <= 0.03:
                    normalized.append(
                        {
                            "type": "double_tap",
                            "source_indices": [first.get("index"), second.get("index")],
                            "interval_ms": interval,
                            "position": {
                                "x_norm": first_point.get("x_norm"),
                                "y_norm": first_point.get("y_norm"),
                            },
                            "video_start_ms": first.get("video_start_ms"),
                            "video_end_ms": second.get("video_end_ms"),
                        }
                    )
                    index += 2
                    continue
        normalized.append(_compact_tutorial_action(first))
        index += 1
    return normalized


def _build_tutorial_contact_sheet(
    video_path: Path,
    actions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import av
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyAV and Pillow are required for tutorial analysis") from exc

    targets: list[tuple[float, str]] = []
    for action in actions:
        index = action.get("index", "?")
        start = max(0.0, float(action.get("video_start_ms", 0)) / 1000 - 0.2)
        end = max(start, float(action.get("video_end_ms", 0)) / 1000 + 0.8)
        targets.extend([(start, f"{index} before"), (end, f"{index} after")])
    targets.sort()

    captured: list[tuple[str, Any]] = []
    target_index = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            timestamp = float(frame.time or 0.0)
            while target_index < len(targets) and timestamp >= targets[target_index][0]:
                captured.append((targets[target_index][1], frame.to_image().convert("RGB")))
                target_index += 1
            if target_index >= len(targets):
                break
    if len(captured) != len(targets):
        raise ValueError("tutorial video ended before all sampled actions")

    tile_width, tile_height = 180, 320
    columns = 6
    rows = (len(captured) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(captured):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + 22)
        sheet.paste(image.resize((tile_width, tile_height)), (x, y))
        draw.text((x + 4, y + tile_height + 3), label, fill="black")
    sheet.save(output_path, format="JPEG", quality=88)


def _write_tutorial_prior(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    except ModuleNotFoundError:
        text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _profile_prompt(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "Your goal is to actively progress the game."

    lines = [
        f"You are playing: {profile.get('name', 'unknown game')}.",
        f"Main objective: {profile.get('objective', 'progress the game')}.",
    ]
    for title, key in [
        ("Global play rules", "play_rule"),
        ("High-priority strategy", "priorities"),
        ("Screen rules", "screen_rules"),
        ("Match rules", "match_rules"),
        ("Action rules", "action_rules"),
        ("Verified human-demonstration rules", "demonstration_rules"),
        ("Verified tutorial sequence", "tutorial_sequence"),
        ("Human gesture priors", "gesture_priors"),
        ("Demonstration evidence policy", "demonstration_evidence_policy"),
        ("Prohibited actions", "avoid"),
        ("Battle tactics", "battle_tactics"),
    ]:
        values = profile.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{title}:")
            lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def _history_prompt(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "- none"
    lines = []
    for item in history[-6:]:
        action = item.get("action", {})
        outcome = item.get("outcome") or {}
        lines.append(
            "- "
            f"frame {item.get('frame_id')}: "
            f"{action.get('type')}({action.get('x')},{action.get('y')}) "
            f"intent={item.get('intent', '')} "
            f"outcome={outcome.get('status', 'pending')} "
            f"retry={outcome.get('retry_recommendation', '')}"
        )
    return "\n".join(lines)


def _interaction_memory_prompt(memory: list[dict[str, Any]] | None) -> str:
    if not memory:
        return "- none"
    lines = []
    for item in memory[-6:]:
        action = item.get("action") or {}
        lines.append(
            "- "
            f"frame {item.get('frame_id')}: "
            f"{action.get('type')}({action.get('x')},{action.get('y')}"
            f"->{action.get('x2')},{action.get('y2')}) "
            f"status={item.get('status')} "
            f"observed={item.get('observed_change', '')} "
            f"retry={item.get('retry_recommendation', '')}"
        )
    return "\n".join(lines)


def _empty_stage_progress() -> dict[str, Any]:
    return {"completed": [], "last_completed": None}


def _canonical_stage_series(series: str) -> str:
    normalized = series.strip().casefold()
    if normalized in {"level", "stage", "레벨", "스테이지"}:
        return "level"
    return normalized


def _extract_stage_labels(text: str) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    ignored_series = {
        "p", "dx", "dy", "xv", "yv", "prs", "size", "score", "coin", "gem", "점수"
    }
    for match in re.finditer(r"(?<![\w])([A-Za-z가-힣]+)\s*[-:#]?\s*(\d+)(?![\w])", text):
        series = match.group(1).strip()
        if series.casefold() in ignored_series:
            continue
        ordinal = int(match.group(2))
        if ordinal > 100000:
            continue
        labels.append(
            {
                "raw_label": f"{series} {ordinal}",
                "series": _canonical_stage_series(series),
                "ordinal": ordinal,
            }
        )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for label in labels:
        unique[(str(label["series"]), int(label["ordinal"]))] = label
    return list(unique.values())


def _record_completed_stage_from_perception(
    progress: dict[str, Any],
    perception: dict[str, Any],
) -> None:
    summary = str(perception.get("summary", ""))
    visible_text = str(perception.get("visible_text", ""))
    combined = f"{summary} {visible_text}"
    lower = combined.lower()
    completion_markers = (
        "level-completion",
        "level completion",
        "stage-completion",
        "stage completion",
        "level complete",
        "stage complete",
        "victory",
        "three stars",
        "3 stars",
        "레벨 완료",
        "스테이지 완료",
        "클리어 완료",
    )
    if not any(marker in lower for marker in completion_markers):
        return
    labels = _extract_stage_labels(combined)
    if not labels:
        return
    preferred = next(
        (
            label
            for label in labels
            if str(label["series"]) == "level"
        ),
        labels[0],
    )
    completed = progress.setdefault("completed", [])
    identity = (preferred["series"], preferred["ordinal"])
    if not any(
        (item.get("series"), item.get("ordinal")) == identity
        for item in completed
        if isinstance(item, dict)
    ):
        completed.append(preferred)
        print(f"[PROGRESS] recorded completed stage {preferred['raw_label']}", flush=True)
    progress["last_completed"] = preferred


def _plan_reenters_completed_stage(
    plan: dict[str, Any],
    progress: dict[str, Any],
) -> str | None:
    plan_text = " ".join(
        str(plan.get(key, ""))
        for key in ("current_goal", "strategy", "target_description")
    )
    lower = plan_text.lower()
    if any(marker in lower for marker in ("replay", "play again", "retry level", "재도전", "다시 플레이")):
        return None
    completed = {
        (str(item.get("series")), int(item.get("ordinal")))
        for item in progress.get("completed", [])
        if isinstance(item, dict)
        and item.get("series") is not None
        and item.get("ordinal") is not None
    }
    for label in _extract_stage_labels(plan_text):
        if (str(label["series"]), int(label["ordinal"])) in completed:
            return str(label["raw_label"])
    return None


def _is_viewport_recovery_plan(plan: dict[str, Any]) -> bool:
    return str(plan.get("screen_mode", "")).lower() == "recovery"


def _outcome_reports_unintended_viewport_motion(
    outcome: dict[str, Any],
    action: dict[str, Any],
    plan: dict[str, Any],
) -> bool:
    """Detect a manipulation swipe that moved the viewport instead of its target."""
    if action.get("type") != "swipe" or _is_viewport_recovery_plan(plan):
        return False
    if outcome.get("status") not in {"failure", "partial"}:
        return False
    intended = " ".join(
        str(plan.get(key, ""))
        for key in ("current_goal", "strategy", "target_description")
    ).lower()
    if any(word in intended for word in ("pan camera", "move camera", "recenter", "화면 이동")):
        return False
    evidence = " ".join(
        str(outcome.get(key, ""))
        for key in ("observed_change", "failure_reason", "evidence")
    ).lower()
    motion_markers = (
        "camera panned",
        "camera moved",
        "map panned",
        "map moved",
        "view shifted",
        "view moved",
        "island shifted",
        "whole island shifted",
        "screen drag",
        "화면이 이동",
        "화면 이동",
        "맵이 이동",
        "카메라 이동",
    )
    return any(marker in evidence for marker in motion_markers)


def _opposite_swipe_direction(action: dict[str, Any]) -> str:
    try:
        dx = int(action.get("x2")) - int(action.get("x"))
        dy = int(action.get("y2")) - int(action.get("y"))
    except (TypeError, ValueError):
        return "opposite the observed viewport shift"
    if abs(dx) >= abs(dy):
        return "left" if dx > 0 else "right"
    return "up" if dy > 0 else "down"


def _viewport_recovery_plan(state: dict[str, Any]) -> dict[str, Any]:
    direction = str(state.get("direction") or "opposite the observed viewport shift")
    attempts = int(state.get("attempts", 0))
    return {
        "screen_mode": "recovery",
        "current_goal": "Restore stable viewport",
        "strategy": f"Pan {direction} moderately; continue until fully restored",
        "desired_action": "swipe",
        "target_description": (
            f"safe empty movable background, swipe {direction}; avoid all objects"
        ),
        "success_check": "Original full play area is centered",
        "risk": "Do not grab game objects",
        "confidence": max(0.35, 0.75 - attempts * 0.1),
        "recovery": {
            "active": True,
            "trigger_frame": state.get("trigger_frame"),
            "attempt": attempts + 1,
        },
    }


def _parse_model_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                repaired = _repair_partial_decision(match.group(0))
                if repaired is not None:
                    repaired["raw_repair_note"] = "parsed from incomplete JSON"
                    return repaired
        repaired = _repair_partial_decision(text)
        if repaired is not None:
            repaired["raw_repair_note"] = "parsed from incomplete text"
            return repaired
        raise ValueError(f"Model did not return JSON: {raw_text[:500]}")


def _repair_partial_decision(text: str) -> dict[str, Any] | None:
    action_type = _regex_value(text, r'"type"\s*:\s*"([^"]+)"')
    x = _regex_int(text, r'"x"\s*:\s*(-?\d+)')
    y = _regex_int(text, r'"y"\s*:\s*(-?\d+)')
    duration_ms = _regex_int(text, r'"duration_ms"\s*:\s*(\d+)') or 80
    if action_type in {"click", "touch"}:
        action_type = "tap"
    if action_type == "tap" and x is not None and y is not None:
        return {
            "observation_summary": _regex_value(
                text, r'"observation_summary"\s*:\s*"([^"]*)"'
            )
            or "",
            "intent": _regex_value(text, r'"intent"\s*:\s*"([^"]*)"') or "",
            "confidence": _regex_float(text, r'"confidence"\s*:\s*([0-9.]+)') or 0.5,
            "action": {
                "type": "tap",
                "x": x,
                "y": y,
                "duration_ms": duration_ms,
            },
        }
    return None


def _regex_value(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def _regex_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.DOTALL)
    return int(match.group(1)) if match else None


def _regex_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.DOTALL)
    return float(match.group(1)) if match else None


def _normalize_action(action: dict[str, Any], screen: Screen) -> dict[str, Any]:
    action_type = str(action.get("type", "noop"))
    if action_type in {"click", "touch"}:
        action_type = "tap"
    allowed = {
        "tap", "double_tap", "swipe", "long_press", "wait", "back", "home", "noop"
    }
    if action_type not in allowed:
        return {"type": "noop", "reason": f"unsupported model action: {action_type}"}

    normalized = dict(action)
    normalized["type"] = action_type
    if action_type in {"tap", "double_tap", "long_press"} and (
        normalized.get("x") is None or normalized.get("y") is None
    ):
        return {"type": "noop", "reason": f"{action_type} missing x/y"}
    if action_type == "swipe" and (
        normalized.get("x") is None
        or normalized.get("y") is None
        or normalized.get("x2") is None
        or normalized.get("y2") is None
    ):
        return {"type": "noop", "reason": "swipe missing coordinates"}
    for key in ("x", "x2"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = max(0, min(int(normalized[key]), screen.width - 1))
    for key in ("y", "y2"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = max(0, min(int(normalized[key]), screen.height - 1))
    if "duration_ms" in normalized and normalized["duration_ms"] is not None:
        normalized["duration_ms"] = max(0, int(normalized["duration_ms"]))
    return normalized


def _fallback_decision(
    profile: dict[str, Any] | None,
    image_path: Path,
    req: DecideRequest,
    raw_text: str,
    latency_ms: int,
    history: list[dict[str, Any]] | None = None,
    image_fingerprint: list[int] | None = None,
    board_signature: str | None = None,
    failed_swipes_by_board: dict[str, set[str]] | None = None,
    failed_cells_by_board: dict[str, set[str]] | None = None,
) -> dict[str, Any] | None:
    if not profile or profile.get("fallback_planner") != "candy_crush_grid":
        return None
    progress_action = _plan_candy_crush_progress_tap(image_path)
    if progress_action is not None:
        progress_action = _normalize_action(progress_action, req.screen)
        return {
            "observation_summary": "level progress screen",
            "intent": progress_action.get("reason", "다음 단계로 진행합니다."),
            "confidence": 0.75,
            "action": progress_action,
            "model_name": "candy_crush_ui_fallback",
            "latency_ms": latency_ms,
            "raw_text": raw_text,
            "image_fingerprint": image_fingerprint,
            "board_signature": board_signature,
        }
    action = _plan_candy_crush_swipe(
        profile,
        image_path,
        history,
        image_fingerprint,
        board_signature,
        failed_swipes_by_board,
        failed_cells_by_board,
    )
    if action is None:
        return {
            "observation_summary": "candy board visible",
            "intent": "유효한 3매치 후보를 찾지 못해 잠시 기다립니다.",
            "confidence": 0.2,
            "action": {"type": "wait", "duration_ms": 500, "reason": "no grid match found"},
            "model_name": "candy_crush_grid_fallback",
            "latency_ms": latency_ms,
            "raw_text": raw_text,
            "image_fingerprint": image_fingerprint,
            "board_signature": board_signature,
        }
    action = _normalize_action(action, req.screen)
    return {
        "observation_summary": "candy board visible",
        "intent": action.get("reason", "3매치를 만들기 위해 인접 캔디를 스와이프합니다."),
        "confidence": 0.65,
        "action": action,
        "model_name": "candy_crush_grid_fallback",
        "latency_ms": latency_ms,
        "raw_text": raw_text,
        "image_fingerprint": image_fingerprint,
        "board_signature": board_signature,
    }


def _plan_candy_crush_progress_tap(image_path: Path) -> dict[str, Any] | None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None

    button = _find_large_green_button(image)
    if button is None:
        return None
    x, y, width, height = button
    if width < image.width * 0.12 or height < image.height * 0.035:
        return None
    return {
        "type": "tap",
        "x": x,
        "y": y,
        "duration_ms": 80,
        "reason": "레벨 완료 또는 진행 화면의 초록색 다음 버튼을 누릅니다.",
    }


def _plan_candy_crush_swipe(
    profile: dict[str, Any],
    image_path: Path,
    history: list[dict[str, Any]] | None = None,
    image_fingerprint: list[int] | None = None,
    board_signature: str | None = None,
    failed_swipes_by_board: dict[str, set[str]] | None = None,
    failed_cells_by_board: dict[str, set[str]] | None = None,
) -> dict[str, Any] | None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None

    grid_data = _choose_candy_grid(profile, image)
    if grid_data is None:
        return None
    rows, cols, centers, labels = grid_data

    blocked = _recent_swipe_signatures(history, image_fingerprint, board_signature)
    if board_signature and failed_swipes_by_board:
        blocked.update(failed_swipes_by_board.get(board_signature, set()))
    failed_cells = (
        failed_cells_by_board.get(board_signature, set())
        if board_signature and failed_cells_by_board
        else set()
    )
    recent_cells = _recent_swipe_cells(history)
    candidates: list[tuple[int, int, int, int, int, str]] = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                row2 = row + dr
                col2 = col + dc
                if row2 >= rows or col2 >= cols:
                    continue
                if labels[row][col] == "unknown" or labels[row2][col2] == "unknown":
                    continue
                swapped = [line[:] for line in labels]
                swapped[row][col], swapped[row2][col2] = swapped[row2][col2], swapped[row][col]
                info = _combined_match_info(swapped, row, col, row2, col2)
                score = info["score"]
                if score <= 0:
                    continue
                score -= _recent_cell_penalty(row, col, row2, col2, recent_cells)
                score -= _failed_cell_penalty(centers, row, col, row2, col2, failed_cells)
                candidates.append((score, row, col, row2, col2, str(info["label"])))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = None
    for candidate in candidates:
        _, row, col, row2, col2, _label = candidate
        signatures = _candidate_swipe_signatures(centers, row, col, row2, col2)
        if signatures.isdisjoint(blocked):
            best = candidate
            break
    if best is None:
        best = candidates[0]
    _, row, col, row2, col2, match_label = best
    x, y = centers[row][col]
    x2, y2 = centers[row2][col2]
    return {
        "type": "swipe",
        "x": x,
        "y": y,
        "x2": x2,
        "y2": y2,
        "duration_ms": 220,
        "reason": f"{row + 1}행 {col + 1}열 캔디를 {row2 + 1}행 {col2 + 1}열과 바꿔 {match_label}를 만듭니다.",
    }


def _candy_board_signature(profile: dict[str, Any] | None, image_path: Path) -> str | None:
    if not profile or profile.get("fallback_planner") != "candy_crush_grid":
        return None
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None

    grid_data = _choose_candy_grid(profile, image)
    if grid_data is None:
        return None
    _rows, _cols, _centers, labels_grid = grid_data
    labels = [label for row in labels_grid for label in row]
    return "|".join(labels)


def _choose_candy_grid(
    profile: dict[str, Any],
    image: Any,
) -> tuple[int, int, list[list[tuple[int, int]]], list[list[str]]] | None:
    best: tuple[int, int, int, list[list[tuple[int, int]]], list[list[str]]] | None = None
    for grid in _board_grid_candidates(profile):
        rows = int(grid.get("rows", 0))
        cols = int(grid.get("cols", 0))
        bbox = grid.get("bbox") or {}
        if rows <= 0 or cols <= 0:
            continue
        left = int(bbox.get("left", 0))
        top = int(bbox.get("top", 0))
        right = int(bbox.get("right", image.width))
        bottom = int(bbox.get("bottom", image.height))
        if right <= left or bottom <= top:
            continue
        cell_w = (right - left) / cols
        cell_h = (bottom - top) / rows
        centers = [
            [
                (int(left + (col + 0.5) * cell_w), int(top + (row + 0.5) * cell_h))
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        labels = [
            [
                _classify_candy_color(_sample_rgb(image, *centers[row][col]))
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        flat = [label for row in labels for label in row]
        known = sum(1 for label in flat if label != "unknown")
        candidate_count = _count_match_candidates(labels)
        score = known * 10 + candidate_count * 25
        if known / max(len(flat), 1) < 0.45:
            continue
        if best is None or score > best[0]:
            best = (score, rows, cols, centers, labels)
    if best is None:
        return None
    _score, rows, cols, centers, labels = best
    return rows, cols, centers, labels


def _board_grid_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for value in profile.get("board_grids") or []:
        if isinstance(value, dict):
            candidates.append(value)
    default = profile.get("board_grid")
    if isinstance(default, dict):
        candidates.append(default)
    return candidates


def _sample_rgb(image: Any, x: int, y: int, radius: int = 14) -> tuple[int, int, int]:
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(image.width, x + radius + 1)
    bottom = min(image.height, y + radius + 1)
    pixels = list(image.crop((left, top, right, bottom)).getdata())
    count = max(len(pixels), 1)
    return (
        sum(pixel[0] for pixel in pixels) // count,
        sum(pixel[1] for pixel in pixels) // count,
        sum(pixel[2] for pixel in pixels) // count,
    )


def _classify_candy_color(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    if red > 170 and green > 120 and blue < 110:
        return "yellow"
    if red > 170 and 70 <= green <= 165 and blue < 100:
        return "orange"
    if red > 150 and green < 110 and blue < 110:
        return "red"
    if green > 130 and red < 140 and blue < 140:
        return "green"
    if red > 120 and blue > 120 and green < 130:
        return "purple"
    return "unknown"


def _find_large_green_button(image: Any) -> tuple[int, int, int, int] | None:
    scale = 4
    left = int(image.width * 0.10)
    top = int(image.height * 0.50)
    right = int(image.width * 0.72)
    bottom = int(image.height * 0.99)
    roi = image.crop((left, top, right, bottom))
    small = roi.resize((max(1, roi.width // scale), max(1, roi.height // scale)))
    width, height = small.size
    mask = [[False for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            red, green, blue = small.getpixel((x, y))
            mask[y][x] = green > 135 and red < 110 and blue < 130 and green > red * 1.45

    seen = [[False for _ in range(width)] for _ in range(height)]
    best: tuple[int, int, int, int, int] | None = None
    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y][start_x] or not mask[start_y][start_x]:
                continue
            stack = [(start_x, start_y)]
            seen[start_y][start_x] = True
            count = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            while stack:
                x, y = stack.pop()
                count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny][nx]
                        and mask[ny][nx]
                    ):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            comp_w = (max_x - min_x + 1) * scale
            comp_h = (max_y - min_y + 1) * scale
            if count < 250 or comp_w < 120 or comp_h < 35:
                continue
            if best is None or count > best[0]:
                best = (count, min_x, min_y, max_x, max_y)

    if best is None:
        return None
    _, min_x, min_y, max_x, max_y = best
    center_x = left + int((min_x + max_x + 1) * scale / 2)
    center_y = top + int((min_y + max_y + 1) * scale / 2)
    comp_w = (max_x - min_x + 1) * scale
    comp_h = (max_y - min_y + 1) * scale
    return center_x, center_y, comp_w, comp_h


def _count_match_candidates(labels: list[list[str]]) -> int:
    rows = len(labels)
    cols = len(labels[0]) if rows else 0
    count = 0
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                row2 = row + dr
                col2 = col + dc
                if row2 >= rows or col2 >= cols:
                    continue
                if labels[row][col] == "unknown" or labels[row2][col2] == "unknown":
                    continue
                swapped = [line[:] for line in labels]
                swapped[row][col], swapped[row2][col2] = swapped[row2][col2], swapped[row][col]
                if int(_combined_match_info(swapped, row, col, row2, col2)["score"]) > 0:
                    count += 1
    return count


def _recent_swipe_signatures(
    history: list[dict[str, Any]] | None,
    image_fingerprint: list[int] | None = None,
    board_signature: str | None = None,
) -> set[str]:
    if not history:
        return set()
    signatures: set[str] = set()
    recent_history = history[-20:]
    for idx, item in enumerate(recent_history):
        same_screen = _fingerprints_similar(
            image_fingerprint,
            item.get("image_fingerprint"),
        )
        same_board = bool(board_signature and board_signature == item.get("board_signature"))
        is_most_recent = idx == len(recent_history) - 1
        if not same_board and not same_screen and not is_most_recent:
            continue
        action = item.get("action", {})
        if action.get("type") != "swipe":
            continue
        x = action.get("x")
        y = action.get("y")
        x2 = action.get("x2")
        y2 = action.get("y2")
        if None in {x, y, x2, y2}:
            continue
        signatures.add(_coordinate_swipe_signature(int(x), int(y), int(x2), int(y2)))
        signatures.add(_coordinate_bucket_signature(int(x), int(y), int(x2), int(y2)))
    return signatures


def _recent_swipe_cells(history: list[dict[str, Any]] | None) -> list[tuple[int, int]]:
    if not history:
        return []
    cells: list[tuple[int, int]] = []
    for item in history[-12:]:
        action = item.get("action", {})
        if action.get("type") != "swipe":
            continue
        for x_key, y_key in (("x", "y"), ("x2", "y2")):
            x = action.get(x_key)
            y = action.get(y_key)
            if x is None or y is None:
                continue
            cells.append(_coordinate_to_grid_cell(int(x), int(y)))
    return cells


def _recent_cell_penalty(
    row: int,
    col: int,
    row2: int,
    col2: int,
    recent_cells: list[tuple[int, int]],
) -> int:
    if not recent_cells:
        return 0
    penalty = 0
    for recent_row, recent_col in recent_cells:
        distance = min(
            abs(row - recent_row) + abs(col - recent_col),
            abs(row2 - recent_row) + abs(col2 - recent_col),
        )
        if distance == 0:
            penalty += 45
        elif distance == 1:
            penalty += 25
        elif distance == 2:
            penalty += 10
    return penalty


def _failed_cell_penalty(
    centers: list[list[tuple[int, int]]],
    row: int,
    col: int,
    row2: int,
    col2: int,
    failed_cells: set[str],
) -> int:
    if not failed_cells:
        return 0
    x, y = centers[row][col]
    x2, y2 = centers[row2][col2]
    penalty = 0
    for point in (
        _coordinate_cell_bucket_signature(x, y),
        _coordinate_cell_bucket_signature(x2, y2),
    ):
        if point in failed_cells:
            penalty += 180
    return penalty


def _image_fingerprint(image_path: Path) -> list[int] | None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None
    try:
        image = Image.open(image_path).convert("L").resize((8, 8))
    except OSError:
        return None
    pixels = list(image.getdata())
    average = sum(pixels) / max(len(pixels), 1)
    return [1 if pixel >= average else 0 for pixel in pixels]


def _fingerprints_similar(
    current: list[int] | None,
    previous: object,
    max_distance: int = 4,
) -> bool:
    if not isinstance(current, list) or not isinstance(previous, list):
        return False
    if len(current) != len(previous):
        return False
    distance = sum(1 for left, right in zip(current, previous) if left != right)
    return distance <= max_distance


def _coordinate_swipe_signature(x: int, y: int, x2: int, y2: int) -> str:
    row, col = _coordinate_to_grid_cell(x, y)
    row2, col2 = _coordinate_to_grid_cell(x2, y2)
    return _grid_swipe_signature(row, col, row2, col2)


def _coordinate_bucket_signature(x: int, y: int, x2: int, y2: int) -> str:
    first = _coordinate_cell_bucket(x, y)
    second = _coordinate_cell_bucket(x2, y2)
    a, b = sorted([first, second])
    return f"coord:{a[0]}:{a[1]}-{b[0]}:{b[1]}"


def _coordinate_cell_bucket_signature(x: int, y: int) -> str:
    bucket = _coordinate_cell_bucket(x, y)
    return f"cell:{bucket[0]}:{bucket[1]}"


def _coordinate_cell_bucket(x: int, y: int) -> tuple[int, int]:
    return round(x / 40), round(y / 40)


def _candidate_swipe_signatures(
    centers: list[list[tuple[int, int]]],
    row: int,
    col: int,
    row2: int,
    col2: int,
) -> set[str]:
    x, y = centers[row][col]
    x2, y2 = centers[row2][col2]
    return {
        _grid_swipe_signature(row, col, row2, col2),
        _coordinate_swipe_signature(x, y, x2, y2),
        _coordinate_bucket_signature(x, y, x2, y2),
    }


def _coordinate_to_grid_cell(x: int, y: int) -> tuple[int, int]:
    if 900 <= x <= 1500 and 120 <= y <= 950:
        col = round((x - 915 - 560 / 10) / (560 / 5))
        row = round((y - 150 - 780 / 14) / (780 / 7))
        return row, col
    col = round((x - 805 - 780 / 14) / (780 / 7))
    row = round((y - 40 - 785 / 14) / (785 / 7))
    return row, col


def _grid_swipe_signature(row: int, col: int, row2: int, col2: int) -> str:
    first = (row, col)
    second = (row2, col2)
    a, b = sorted([first, second])
    return f"{a[0]}:{a[1]}-{b[0]}:{b[1]}"


def _match_score(labels: list[list[str]], row: int, col: int) -> int:
    return int(_single_match_info(labels, row, col)["score"])


def _combined_match_info(
    labels: list[list[str]],
    row: int,
    col: int,
    row2: int,
    col2: int,
) -> dict[str, int | str]:
    first = _single_match_info(labels, row, col)
    second = _single_match_info(labels, row2, col2)
    score = int(first["score"]) + int(second["score"])
    max_run = max(int(first["max_run"]), int(second["max_run"]))
    line_count = int(first["line_count"]) + int(second["line_count"])
    if score <= 0:
        return {"score": 0, "max_run": 0, "line_count": 0, "label": "유효하지 않은 매치"}

    if max_run >= 5:
        score += 120
        label = "5매치"
    elif line_count >= 2:
        score += 90
        label = "가로/세로 동시 매치"
    elif max_run >= 4:
        score += 55
        label = "4매치"
    else:
        label = "3매치"
    return {
        "score": score,
        "max_run": max_run,
        "line_count": line_count,
        "label": label,
    }


def _single_match_info(labels: list[list[str]], row: int, col: int) -> dict[str, int]:
    target = labels[row][col]
    if target == "unknown":
        return {"score": 0, "max_run": 0, "line_count": 0}
    rows = len(labels)
    cols = len(labels[0]) if rows else 0

    horizontal = 1
    left = col - 1
    while left >= 0 and labels[row][left] == target:
        horizontal += 1
        left -= 1
    right = col + 1
    while right < cols and labels[row][right] == target:
        horizontal += 1
        right += 1

    vertical = 1
    up = row - 1
    while up >= 0 and labels[up][col] == target:
        vertical += 1
        up -= 1
    down = row + 1
    while down < rows and labels[down][col] == target:
        vertical += 1
        down += 1

    score = 0
    line_count = 0
    if horizontal >= 3:
        line_count += 1
        score += horizontal * horizontal * 10
    if vertical >= 3:
        line_count += 1
        score += vertical * vertical * 10
    max_run = max(horizontal if horizontal >= 3 else 0, vertical if vertical >= 3 else 0)
    return {"score": score, "max_run": max_run, "line_count": line_count}


def create_app(server: DecisionServer) -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Activate gameagent_vlm or install the vlm extras."
        )
    app = FastAPI(title="GameAgent Local VLM Server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mock": server.mock,
            "model_id": server.model_id,
            "agent_model_ids": server.agent_model_ids,
            "agent_mode": server.agent_mode,
            "loaded": server._loaded,
            "loaded_model_ids": sorted(server._model_cache),
        }

    @app.post("/v1/decide")
    def decide(req: DecideRequest) -> dict[str, Any]:
        try:
            decision = server.decide(req)
            action = decision["action"]
            print(
                f"[VLM] frame={req.frame_id} action={action.get('type')} "
                f"x={action.get('x')} y={action.get('y')} "
                f"intent={str(decision.get('intent', ''))[:120]}",
                flush=True,
            )
            return decision
        except Exception as exc:
            print(f"[VLM] unhandled decide error: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _agent_model_ids_from_args(
    args: argparse.Namespace,
    default_model_id: str,
    configured_models: dict[str, Any] | None = None,
) -> dict[str, str]:
    model_ids: dict[str, str] = {}
    configured_models = configured_models or {}
    for agent in PIPELINE_AGENTS:
        arg_prefix = agent
        explicit_model = getattr(args, f"{arg_prefix}_model_id")
        model_size = getattr(args, f"{arg_prefix}_model_size")
        if explicit_model:
            model_ids[agent] = _resolve_local_model_ref(str(explicit_model))
        elif model_size:
            model_ids[agent] = QWEN_VL_MODEL_IDS[str(model_size)]
        elif configured_models.get(agent):
            model_ids[agent] = _resolve_local_model_ref(str(configured_models[agent]))
        else:
            model_ids[agent] = default_model_id
    return model_ids


def _parse_model_ref(value: str) -> ModelRef:
    resolved = _resolve_local_model_ref(value)
    lower = resolved.lower()
    for prefix, provider in [
        ("openai:", "openai"),
        ("gpt:", "openai"),
        ("gemini:", "gemini"),
        ("google:", "gemini"),
        ("anthropic:", "anthropic"),
        ("claude:", "anthropic"),
    ]:
        if lower.startswith(prefix):
            model = resolved.split(":", 1)[1].strip()
            if not model:
                raise ValueError(f"{prefix} model ref requires a model name")
            return ModelRef(provider=provider, model=model)
    return ModelRef(provider="hf", model=resolved)


def _resolve_local_model_ref(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    if text in QWEN_VL_MODEL_IDS:
        return QWEN_VL_MODEL_IDS[text]
    if lower.startswith("qwen:"):
        size = text.split(":", 1)[1]
        if size not in QWEN_VL_MODEL_IDS:
            raise ValueError(f"Unknown Qwen model size: {size}")
        return QWEN_VL_MODEL_IDS[size]
    if lower.startswith("hf:"):
        model_id = text.split(":", 1)[1].strip()
        if not model_id:
            raise ValueError("hf: model ref requires a Hugging Face model id")
        return model_id
    return text


def _generate_hosted(
    model_ref: ModelRef,
    prompt: str,
    image_path: Path,
    include_image: bool,
    max_tokens: int,
    timeout_s: float,
) -> str:
    image_b64 = _image_base64(image_path) if include_image else None
    media_type = _media_type(image_path)
    if model_ref.provider == "openai":
        return _generate_openai(model_ref.model, prompt, image_b64, media_type, max_tokens, timeout_s)
    if model_ref.provider == "gemini":
        return _generate_gemini(model_ref.model, prompt, image_b64, media_type, max_tokens, timeout_s)
    if model_ref.provider == "anthropic":
        return _generate_anthropic(model_ref.model, prompt, image_b64, media_type, max_tokens, timeout_s)
    raise ValueError(f"Unsupported hosted provider: {model_ref.provider}")


def _generate_openai(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _required_env("OPENAI_API_KEY", "openai")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if image_b64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_b64}",
            }
        )
    data = _post_json(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/responses",
        {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
        },
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    text = data.get("output_text")
    if isinstance(text, str):
        return text.strip()
    return _extract_openai_output_text(data)


def _generate_gemini(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _env_first("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY")
    input_items: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64:
        input_items.append({"type": "image", "data": image_b64, "mime_type": media_type})
    data = _post_json(
        os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip()
        + "/interactions",
        {
            "model": model,
            "input": input_items,
            "generation_config": {"max_output_tokens": max_tokens},
        },
        {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    text = data.get("output_text")
    if isinstance(text, str):
        return text.strip()
    return _extract_gemini_output_text(data)


def _generate_anthropic(
    model: str,
    prompt: str,
    image_b64: str | None,
    media_type: str,
    max_tokens: int,
    timeout_s: float,
) -> str:
    api_key = _env_first("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("anthropic provider requires ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    content: list[dict[str, Any]] = []
    if image_b64:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    data = _post_json(
        os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        + "/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    return _extract_anthropic_output_text(data)


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body}") from exc


def _image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _media_type(image_path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(image_path))
    return guessed or "image/png"


def _required_env(name: str, provider: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{provider} provider requires {name}")
    return value


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _extract_openai_output_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise ValueError(f"OpenAI response did not contain output text: {str(data)[:500]}")
    return "\n".join(parts).strip()


def _extract_gemini_output_text(data: dict[str, Any]) -> str:
    for key in ("text", "response_text"):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip()
    output = data.get("output")
    if isinstance(output, list):
        parts = [
            item.get("text")
            for item in output
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    raise ValueError(f"Gemini response did not contain output text: {str(data)[:500]}")


def _extract_anthropic_output_text(data: dict[str, Any]) -> str:
    parts = [
        item.get("text")
        for item in data.get("content", []) or []
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    if not parts:
        raise ValueError(f"Anthropic response did not contain output text: {str(data)[:500]}")
    return "\n".join(parts).strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gameagent-vlm-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--pipeline-config",
        default=None,
        help="YAML/JSON file containing default_model and per-agent model refs",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(QWEN_VL_MODEL_IDS),
        default="7B",
        help="Qwen2.5-VL size shortcut; ignored when --model-id is set",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "default model ref: qwen:3B, qwen:7B, hf:<model-id>, openai:<model>, "
            "gemini:<model>, anthropic:<model>, claude:<model>, or bare HF model id"
        ),
    )
    for agent in PIPELINE_AGENTS:
        dashed = agent.replace("_", "-")
        parser.add_argument(
            f"--{dashed}-model-id",
            default=None,
            help=(
                f"model ref for the {dashed} agent: qwen:3B, qwen:7B, hf:<model-id>, "
                "openai:<model>, gemini:<model>, anthropic:<model>, claude:<model>, "
                "or bare HF model id"
            ),
        )
        parser.add_argument(
            f"--{dashed}-model-size",
            choices=sorted(QWEN_VL_MODEL_IDS),
            default=None,
            help=f"Qwen2.5-VL size shortcut for the {dashed} agent",
        )
    parser.add_argument("--mock", action="store_true", help="run deterministic test policy")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="maximum output tokens per pipeline stage (default: 1024)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=589824,
        help="maximum image pixels sent to Qwen-VL; lower values reduce memory use",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profile", default=None, help="game knowledge profile YAML/JSON")
    parser.add_argument(
        "--tutorial",
        action="store_true",
        help=(
            "load or generate the convention-based tutorial prior; when omitted, "
            "use only global and game profile rules"
        ),
    )
    parser.add_argument(
        "--rule-memory-path",
        default=None,
        help="persist learned rule memory as JSON and load it on the next server run",
    )
    parser.add_argument(
        "--rule-learning-interval",
        type=int,
        default=3,
        help="update persistent rules after this many verified actions (default: 3)",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["pipeline", "direct"],
        default="pipeline",
        help="pipeline separates perception/planning/policy; direct asks for action in one pass",
    )
    args = parser.parse_args(argv)

    if FastAPI is None or uvicorn is None:
        raise SystemExit(
            "FastAPI/uvicorn are not installed. Run `conda env create -f environment.yml` "
            "and `conda activate gameagent_vlm`, or install the `vlm` extras."
        )

    pipeline_config = _load_profile(args.pipeline_config) or {}
    configured_default = pipeline_config.get("default_model")
    configured_agents = pipeline_config.get("agents", {})
    if not isinstance(configured_agents, dict):
        raise SystemExit("pipeline config 'agents' must be a mapping")
    default_model_id = _resolve_local_model_ref(
        str(args.model_id or configured_default or QWEN_VL_MODEL_IDS[args.model_size])
    )
    agent_model_ids = _agent_model_ids_from_args(
        args,
        default_model_id,
        configured_agents,
    )

    rule_memory_path = args.rule_memory_path
    if rule_memory_path is None and args.profile:
        game_id = Path(args.profile).stem
        rule_memory_path = str(PROJECT_ROOT / "state" / game_id / "rules.json")

    server = DecisionServer(
        model_id=default_model_id,
        mock=args.mock,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        profile_path=args.profile,
        rule_memory_path=rule_memory_path,
        agent_mode=args.agent_mode,
        agent_model_ids=agent_model_ids,
        rule_learning_interval=args.rule_learning_interval,
        tutorial_enabled=args.tutorial,
    )
    app = create_app(server)
    print(
        f"[VLM] listening on http://{args.host}:{args.port} "
        f"model={default_model_id} "
        f"agent_models={agent_model_ids} "
        f"mode={args.agent_mode} mock={args.mock}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
