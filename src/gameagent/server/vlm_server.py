"""Local Qwen-VL decision server.

This server receives a frame from the gameagent runner and returns a structured
action. It can run in two modes:

- real mode: load Qwen2.5-VL and ask it to produce action JSON
- mock mode: deterministic wait/tap responses for wiring tests
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gameagent.server.hosted import generate_hosted

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


ActionName = Literal["tap", "swipe", "long_press", "wait", "back", "home", "noop"]
AgentName = Literal["perception", "rule_learner", "planner", "policy"]
QWEN_VL_MODEL_IDS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}
PIPELINE_AGENTS: tuple[AgentName, ...] = ("perception", "rule_learner", "planner", "policy")
HOSTED_PROVIDERS = {"openai", "openrouter", "gemini", "anthropic", "ollama"}


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
        max_new_tokens: int = 128,
        max_pixels: int = 2073600,
        temperature: float = 0.0,
        profile_path: str | None = None,
        agent_mode: str = "pipeline",
        agent_model_ids: dict[str, str] | None = None,
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
        self.profile = _load_profile(profile_path)
        self.agent_mode = agent_mode
        self._loaded = False
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._process_vision_info: Any = None
        self._history: list[dict[str, Any]] = []
        self._rule_memory: dict[str, Any] = _empty_rule_memory()
        self._failed_swipes_by_board: dict[str, set[str]] = {}
        self._failed_cells_by_board: dict[str, set[str]] = {}

    def decide(self, req: DecideRequest) -> dict[str, Any]:
        if self.mock:
            return self._mock_decide(req)
        if req.frame_id <= 1:
            self._history.clear()
            self._rule_memory = _empty_rule_memory()
            self._failed_swipes_by_board.clear()
            self._failed_cells_by_board.clear()

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
                action.get("type") in {"tap", "long_press"}
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
            if board_signature and action.get("type") in {"tap", "swipe", "wait", "noop"}:
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
        active_stage = "perception"
        _trace_pipeline_frame(req, self.agent_model_ids)
        try:
            _trace_pipeline_waiting("1/4 화면 인식", self.agent_model_ids["perception"])
            raw_perception = self._generate_with_prompt(
                req,
                image_path,
                _perception_prompt(req),
                agent_name="perception",
                include_image=True,
            )
            perception = _parse_model_json(raw_perception)
            _trace_pipeline_value("1/4 화면 인식 결과", perception)

            active_stage = "rule_learner"
            _trace_pipeline_waiting("2/4 룰 학습", self.agent_model_ids["rule_learner"])
            raw_rule_learning = self._generate_with_prompt(
                req,
                image_path,
                _rule_learner_prompt(
                    req,
                    self.profile,
                    self._history,
                    perception,
                    self._rule_memory,
                ),
                agent_name="rule_learner",
                include_image=False,
            )
            rule_memory = _normalize_rule_memory(_parse_model_json(raw_rule_learning))
            self._rule_memory = rule_memory
            _trace_pipeline_value("2/4 모델이 작성한 룰 메모리", self._rule_memory)

            active_stage = "planner"
            _trace_pipeline_waiting("3/4 룰 기반 계획", self.agent_model_ids["planner"])
            raw_plan = self._generate_with_prompt(
                req,
                image_path,
                _planner_prompt(
                    req,
                    self.profile,
                    self._history,
                    perception,
                    self._rule_memory,
                ),
                agent_name="planner",
                include_image=True,
            )
            plan = _parse_model_json(raw_plan)
            _trace_pipeline_value("3/4 룰을 보고 세운 계획", plan)

            active_stage = "policy"
            _trace_pipeline_waiting("4/4 실제 행동 결정", self.agent_model_ids["policy"])
            raw_policy = self._generate_with_prompt(
                req,
                image_path,
                _policy_prompt(req, self.profile, self._history, perception, plan),
                agent_name="policy",
                include_image=True,
            )
            policy = _parse_model_json(raw_policy)
            action = _normalize_action(policy.get("action", {}), req.screen)
            confidence = float(policy.get("confidence", plan.get("confidence", 0.0)))
            if (
                action.get("type") in {"tap", "long_press"}
                and action.get("x") == 0
                and action.get("y") == 0
                and confidence <= 0.05
            ):
                action = {
                    "type": "wait",
                    "duration_ms": 800,
                    "reason": "policy returned an unsafe placeholder coordinate",
                }
            _trace_pipeline_value(
                "4/4 최종 행동",
                {
                    "intent": policy.get("intent", ""),
                    "confidence": confidence,
                    "action": action,
                },
            )

            latency_ms = int((time.perf_counter() - started) * 1000)
            if board_signature and action.get("type") in {"tap", "swipe", "wait", "noop"}:
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
                        "raw_perception": raw_perception,
                        "raw_rule_learning": raw_rule_learning,
                        "raw_plan": raw_plan,
                        "raw_policy": raw_policy,
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
                    "raw_perception": raw_perception,
                    "raw_rule_learning": raw_rule_learning,
                    "raw_plan": raw_plan,
                    "raw_policy": raw_policy,
                },
            }
        except ValueError as exc:
            _trace_pipeline_value(
                "PIPELINE 오류",
                {"stage": active_stage, "error": str(exc)},
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
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
                    "raw_perception": raw_perception,
                    "raw_rule_learning": raw_rule_learning,
                    "raw_plan": raw_plan,
                    "raw_policy": raw_policy,
                    "rule_memory": self._rule_memory,
                },
            }

    def _remember(self, req: DecideRequest, decision: dict[str, Any]) -> None:
        action = decision.get("action", {})
        self._history.append(
            {
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
            }
        )
        self._history = self._history[-30:]

    def _record_previous_action_outcome(
        self,
        req: DecideRequest,
        current_board_signature: str | None,
    ) -> None:
        if not current_board_signature or not req.previous_action or not self._history:
            return
        previous = req.previous_action
        if previous.type not in {"swipe", "tap"}:
            return
        if previous.x is None or previous.y is None:
            return
        last = self._history[-1]
        previous_board_signature = last.get("board_signature")
        if previous_board_signature != current_board_signature:
            return
        failed_cells = self._failed_cells_by_board.setdefault(current_board_signature, set())
        failed_cells.add(_coordinate_cell_bucket_signature(int(previous.x), int(previous.y)))
        if previous.type == "tap":
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
            return generate_hosted(
                provider=model_ref.provider,
                model=model_ref.model,
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


def _perception_prompt(req: DecideRequest) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    return f"""
You are the VLM perception module for a mobile game agent.
Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.

Describe only facts visible in the screenshot. Do not choose an action, infer a
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

Current rule memory:
{json.dumps(rule_memory, ensure_ascii=False)}

Rule learning rules:
- Update rules only from visible tutorial/objective text or repeated action-result evidence.
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

Planning rules:
- Base the plan only on the supplied perception and learned rule memory.
- Do not use prior knowledge about games, UI conventions, or likely button meanings.
- A confirmed rule may be used directly.
- When no confirmed rule applies, select one hypothesis grounded in a currently
  visible object and plan one reversible experiment to test it.
- State the tested hypothesis and expected visible result in strategy/success_check.
- Do not repeat an experiment when history already shows that it produced no change.
- Choose wait only when there is no visible target grounded in a rule or hypothesis.
- Do not output tap/swipe coordinates.

Return only valid compact JSON:
{{
  "screen_mode": "gameplay | progress_ui | popup | loading | unsafe | unknown",
  "current_goal": "under 8 words",
  "strategy": "under 12 words",
  "desired_action": "tap | swipe | long_press | wait | back | noop",
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

Policy rules:
- Execute only planner.desired_action and planner.target_description.
- Do not use prior knowledge about games, UI conventions, or likely button meanings.
- If the planner's target is not visibly identifiable, return wait.
- Coordinates must be exact screen coordinates: x in [0, {req.screen.width - 1}], y in [0, {req.screen.height - 1}].
- Never use placeholder coordinates such as x=0,y=0.
- For tap/long_press, choose the center of the visible target.
- For swipe, use x,y as the drag start and x2,y2 as the drag end.
- If exact coordinates are uncertain, return wait instead of inventing a coordinate.
- If confidence is below 0.2, return wait or noop.

Allowed actions:
- tap: {{"type":"tap","x":int,"y":int,"duration_ms":80}}
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


def _load_profile(path: str | None) -> dict[str, Any] | None:
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


def _profile_prompt(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "Your goal is to actively progress the game."

    lines = [
        f"You are playing: {profile.get('name', 'unknown game')}.",
        f"Main objective: {profile.get('objective', 'progress the game')}.",
    ]
    for title, key in [
        ("High-priority strategy", "priorities"),
        ("Screen rules", "screen_rules"),
        ("Match rules", "match_rules"),
        ("Action rules", "action_rules"),
        ("Avoid unless explicitly needed", "avoid"),
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
        lines.append(
            "- "
            f"frame {item.get('frame_id')}: "
            f"{action.get('type')}({action.get('x')},{action.get('y')}) "
            f"intent={item.get('intent', '')}"
        )
    return "\n".join(lines)


def _parse_model_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
    allowed = {"tap", "swipe", "long_press", "wait", "back", "home", "noop"}
    if action_type not in allowed:
        return {"type": "noop", "reason": f"unsupported model action: {action_type}"}

    normalized = dict(action)
    normalized["type"] = action_type
    if action_type in {"tap", "long_press"} and (
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
        ("openrouter:", "openrouter"),
        ("openai:", "openai"),
        ("gpt:", "openai"),
        ("gemini:", "gemini"),
        ("google:", "gemini"),
        ("anthropic:", "anthropic"),
        ("claude:", "anthropic"),
        ("ollama:", "ollama"),
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
            "default model ref: qwen:3B, qwen:7B, hf:<model-id>, "
            "openrouter:<provider/model>, openai:<model>, gemini:<model>, "
            "anthropic:<model>, claude:<model>, ollama:<model>, or bare HF model id"
        ),
    )
    for agent in PIPELINE_AGENTS:
        dashed = agent.replace("_", "-")
        parser.add_argument(
            f"--{dashed}-model-id",
            default=None,
            help=(
                f"model ref for the {dashed} agent: qwen:3B, qwen:7B, hf:<model-id>, "
                "openrouter:<provider/model>, openai:<model>, gemini:<model>, "
                "anthropic:<model>, claude:<model>, ollama:<model>, or bare HF model id"
            ),
        )
        parser.add_argument(
            f"--{dashed}-model-size",
            choices=sorted(QWEN_VL_MODEL_IDS),
            default=None,
            help=f"Qwen2.5-VL size shortcut for the {dashed} agent",
        )
    parser.add_argument("--mock", action="store_true", help="run deterministic test policy")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=2073600,
        help=(
            "maximum image pixels sent to Qwen-VL; defaults to 1920x1080, "
            "lower values reduce memory use"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profile", default=None, help="game knowledge profile YAML/JSON")
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

    server = DecisionServer(
        model_id=default_model_id,
        mock=args.mock,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        profile_path=args.profile,
        agent_mode=args.agent_mode,
        agent_model_ids=agent_model_ids,
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
