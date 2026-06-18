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


ActionName = Literal["tap", "swipe", "long_press", "wait", "back", "home", "noop"]
QWEN_VL_MODEL_IDS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}


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
        max_pixels: int = 589824,
        temperature: float = 0.0,
        profile_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.temperature = temperature
        self.profile = _load_profile(profile_path)
        self._loaded = False
        self._model: Any = None
        self._processor: Any = None
        self._process_vision_info: Any = None
        self._history: list[dict[str, Any]] = []

    def decide(self, req: DecideRequest) -> dict[str, Any]:
        if self.mock:
            return self._mock_decide(req)
        if req.frame_id <= 1:
            self._history.clear()

        image_path = self._materialize_image(req)
        image_fingerprint = _image_fingerprint(image_path)
        started = time.perf_counter()
        try:
            raw_text = self._generate(req, image_path)
            parsed = _parse_model_json(raw_text)
            action = _normalize_action(parsed.get("action", {}), req.screen)
            latency_ms = int((time.perf_counter() - started) * 1000)
            decision = {
                "observation_summary": str(parsed.get("observation_summary", "")),
                "intent": str(parsed.get("intent", "")),
                "confidence": float(parsed.get("confidence", 0.0)),
                "action": action,
                "model_name": self.model_id,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "image_fingerprint": image_fingerprint,
            }
            self._remember(req, decision)
            return decision
        except ValueError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
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
            }
        )
        self._history = self._history[-30:]

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

    def _load(self) -> None:
        if self._loaded:
            return
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
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._process_vision_info = process_vision_info
        self._loaded = True

    def _generate(self, req: DecideRequest, image_path: Path) -> str:
        self._load()
        prompt = _decision_prompt(req, self.profile, self._history)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generation_kwargs.update({"do_sample": False})

        generated_ids = self._model.generate(**inputs, **generation_kwargs)
        decoded = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        return str(decoded).strip()

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


def _decision_prompt(
    req: DecideRequest,
    profile: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    previous = req.previous_action.model_dump() if req.previous_action else None
    profile_prompt = _profile_prompt(profile)
    history_prompt = _history_prompt(history)
    return f"""
You are controlling a mobile game through touch actions only.
Screen size: {req.screen.width}x{req.screen.height}.
Previous action: {json.dumps(previous, ensure_ascii=False)}.
Recent agent history:
{history_prompt}

{profile_prompt}

Look at the screenshot and choose exactly one next action.
If tutorial, help, mission, rule, or objective text is visible, read it and use it as game knowledge for this decision. Mention the useful part briefly in the intent when it affects the action.

Decision policy:
- Follow the game profile priorities before generic UI instincts.
- For board or puzzle games, follow the profile's move/action rules before tapping generic UI.
- Prefer a useful tap over waiting.
- Tap obvious primary buttons, start buttons, next buttons, confirmation buttons, close buttons, reward buttons, menu buttons, red-dot notification buttons, quest buttons, and highlighted UI elements.
- If a dialog, popup, tutorial prompt, or reward screen is visible, choose the most likely continue/confirm/close/claim action.
- If the screen looks like a loading screen, transition animation, or no actionable UI is visible, choose wait.
- If the screen is confusing, blank, changing, or you cannot identify a safe action, return a wait action JSON.
- Do not choose wait just because you are uncertain. If there is any plausible actionable UI, tap the most likely target.
- Avoid account, payment, purchase, login, or personal information actions. Choose noop or back for those screens.
- Coordinates must use the current screen coordinate system: x in [0, {req.screen.width - 1}], y in [0, {req.screen.height - 1}].
- Use action type "tap" for touching/clicking a button. Never use "click".
- If the previous action was a tap at nearly the same coordinates and the screen still looks similar, do not repeat the same tap. Choose another visible target, wait briefly, or use back if appropriate.
- If recent history shows the same OK/confirm/start tap was already tried twice, do not tap it again unless the screen clearly changed.
- Treat noop or wait after a tap as a sign that the previous tap may not have progressed. Re-evaluate the screen and choose a different progress action if visible.

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
  "intent": "Tap the start button to begin battle.",
  "confidence": 0.0,
  "action": {{"type": "tap", "x": 0, "y": 0, "duration_ms": 80}}
}}
""".strip()


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
    return None


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
            "loaded": server._loaded,
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
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gameagent-vlm-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--model-size",
        choices=sorted(QWEN_VL_MODEL_IDS),
        default="7B",
        help="Qwen2.5-VL size shortcut; ignored when --model-id is set",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="explicit Hugging Face model id; overrides --model-size",
    )
    parser.add_argument("--mock", action="store_true", help="run deterministic test policy")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=589824,
        help="maximum image pixels sent to Qwen-VL; lower values reduce memory use",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profile", default=None, help="game knowledge profile YAML/JSON")
    args = parser.parse_args(argv)

    if FastAPI is None or uvicorn is None:
        raise SystemExit(
            "FastAPI/uvicorn are not installed. Run `conda env create -f environment.yml` "
            "and `conda activate gameagent_vlm`, or install the `vlm` extras."
        )

    server = DecisionServer(
        model_id=args.model_id or QWEN_VL_MODEL_IDS[args.model_size],
        mock=args.mock,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        profile_path=args.profile,
    )
    app = create_app(server)
    print(
        f"[VLM] listening on http://{args.host}:{args.port} "
        f"model={args.model_id} mock={args.mock}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
