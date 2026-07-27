import base64
from types import SimpleNamespace
from unittest.mock import patch

from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, Observation
from gameagent.server.hosted import (
    _generate_ollama,
    _apply_openrouter_model_options,
    _extract_anthropic_output_text,
    _extract_gemini_output_text,
    _extract_ollama_output_text,
    _extract_openai_output_text,
    _extract_openrouter_output_text,
    _is_openrouter_claude,
    _is_openrouter_gemini,
)
from gameagent.server.vlm_server import (
    DecisionServer,
    _normalize_rule_memory,
    _parse_model_ref,
    _resolve_local_model_ref,
)


def test_decision_server_materializes_base64_image():
    image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    request = SimpleNamespace(
        image_path=None,
        image_base64=base64.b64encode(image).decode("ascii"),
    )
    server = DecisionServer(model_id="qwen:7B", mock=True)

    image_path = server._materialize_image(request)
    try:
        assert image_path.read_bytes() == image
    finally:
        image_path.unlink(missing_ok=True)


def test_decision_server_keeps_full_hd_qwen_input_by_default():
    server = DecisionServer(model_id="qwen:7B", mock=True)

    assert server.max_pixels == 1920 * 1080


def test_action_validator_clamps_coordinates():
    validator = ActionValidator(allowed_actions={ActionType.TAP})
    obs = Observation(frame_id=1, timestamp=0.0, width=100, height=200)
    action = validator.validate(Action(type=ActionType.TAP, x=999, y=-5), obs)

    assert action.type == ActionType.TAP
    assert action.x == 99
    assert action.y == 0


def test_action_validator_blocks_disallowed_action():
    validator = ActionValidator(allowed_actions={ActionType.WAIT})
    obs = Observation(frame_id=1, timestamp=0.0, width=100, height=200)
    action = validator.validate(Action(type=ActionType.TAP, x=50, y=50), obs)

    assert action.type == ActionType.NOOP


def test_rule_memory_normalization_keeps_compact_schema():
    memory = _normalize_rule_memory(
        {
            "objective": "clear the board",
            "confirmed_rules": [
                {"rule": "match three pieces", "confidence": "0.8", "evidence": "score changed"},
                {"rule": "bad confidence", "confidence": "not-a-number"},
            ],
            "hypotheses": ["locked pieces cannot move"],
            "failed_patterns": [{"pattern": "repeat same tap", "evidence": "no change"}],
        }
    )

    assert memory["objective"] == "clear the board"
    assert memory["confirmed_rules"][0]["confidence"] == 0.8
    assert memory["confirmed_rules"][1]["confidence"] == 0.0
    assert memory["hypotheses"][0]["rule"] == "locked pieces cannot move"
    assert memory["failed_patterns"][0]["pattern"] == "repeat same tap"


def test_local_model_ref_resolution():
    assert _resolve_local_model_ref("qwen:3B") == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert _resolve_local_model_ref("hf:org/model") == "org/model"
    assert _resolve_local_model_ref("org/model") == "org/model"


def test_hosted_model_ref_parsing():
    openrouter = _parse_model_ref("openrouter:~openai/gpt-latest")
    assert openrouter.provider == "openrouter"
    assert openrouter.model == "~openai/gpt-latest"
    assert _parse_model_ref("openai:gpt-5.6").provider == "openai"
    assert _parse_model_ref("gpt:gpt-5.6").model == "gpt-5.6"
    assert _parse_model_ref("gemini:gemini-3.5-flash").provider == "gemini"
    assert _parse_model_ref("claude:claude-opus-4-8").provider == "anthropic"
    ollama = _parse_model_ref("ollama:llava:7b")
    assert ollama.provider == "ollama"
    assert ollama.model == "llava:7b"
    assert ollama.key == "ollama:llava:7b"


def test_openrouter_claude_detection():
    assert _is_openrouter_claude("~anthropic/claude-sonnet-latest")
    assert _is_openrouter_claude("anthropic/claude-opus-4.8")
    assert not _is_openrouter_claude("~openai/gpt-latest")


def test_openrouter_gemini_detection():
    assert _is_openrouter_gemini("~google/gemini-flash-latest")
    assert _is_openrouter_gemini("google/gemini-3.5-flash")
    assert not _is_openrouter_gemini("~anthropic/claude-sonnet-latest")


def test_openrouter_claude_reserves_json_output_budget():
    payload = {"max_tokens": 1024}
    _apply_openrouter_model_options(payload, "~anthropic/claude-sonnet-latest")

    assert payload["max_tokens"] == 4096
    assert payload["reasoning"] == {"max_tokens": 1024, "exclude": True}
    assert payload["response_format"] == {"type": "json_object"}


def test_ollama_reserves_json_output_budget():
    with patch(
        "gameagent.server.hosted._post_json",
        return_value={"message": {"content": '{"status":"ok"}'}},
    ) as post_json:
        output = _generate_ollama("llava:7b", "return JSON", None, 128, 30.0)

    assert output == '{"status":"ok"}'
    payload = post_json.call_args.args[1]
    assert payload["format"] == "json"
    assert payload["options"]["num_predict"] == 512


def test_hosted_response_text_extraction():
    assert (
        _extract_openrouter_output_text(
            {"choices": [{"message": {"content": "openrouter"}}]}
        )
        == "openrouter"
    )
    assert (
        _extract_openai_output_text(
            {"output": [{"content": [{"type": "output_text", "text": "openai"}]}]}
        )
        == "openai"
    )
    assert _extract_gemini_output_text({"output": [{"text": "gemini"}]}) == "gemini"
    assert _extract_ollama_output_text({"message": {"content": "ollama"}}) == "ollama"
    assert (
        _extract_anthropic_output_text({"content": [{"type": "text", "text": "anthropic"}]})
        == "anthropic"
    )
