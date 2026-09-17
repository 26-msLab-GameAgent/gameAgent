from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, Observation
from gameagent.server.vlm_server import (
    _load_profile_with_global_rules,
    _merge_profiles,
    _normalize_action_outcome,
    _normalize_rule_memory,
    _parse_model_ref,
    _plan_reenters_completed_stage,
    _profile_prompt,
    _record_completed_stage_from_perception,
    _repeats_blocked_action,
    _resolve_local_model_ref,
    _outcome_reports_unintended_viewport_motion,
    _opposite_swipe_direction,
    _viewport_recovery_plan,
)
from gameagent.runtime.runner import _frame_change_ratio


def _solid_png(color: tuple[int, int, int]) -> bytes:
    import io
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_frame_change_ratio_detects_large_screen_change():
    black = _solid_png((0, 0, 0))
    white = _solid_png((255, 255, 255))

    assert _frame_change_ratio(black, black) == 0.0
    assert _frame_change_ratio(black, white) == 1.0


def test_global_rules_are_prepended_to_game_profile():
    merged = _merge_profiles(
        {"play_rule": ["continue stages"], "avoid": ["no purchases"]},
        {"name": "Example Game", "avoid": ["no risky taps"]},
    )

    assert merged == {
        "name": "Example Game",
        "play_rule": ["continue stages"],
        "avoid": ["no purchases", "no risky taps"],
    }
    prompt = _profile_prompt(merged)
    assert "Global play rules:\n- continue stages" in prompt
    assert "Prohibited actions:\n- no purchases\n- no risky taps" in prompt


def test_testy_profile_loads_human_demonstration_prior_when_enabled():
    profile = _load_profile_with_global_rules(
        "configs/profiles/testy_travel.yaml",
        auto_tutorial=True,
    )

    assert profile is not None
    assert any(
        "dragging the center" in rule
        for rule in profile["demonstration_rules"]
    )
    prompt = _profile_prompt(profile)
    assert "Verified human-demonstration rules" in prompt
    assert "Human gesture priors" in prompt


def test_profile_skips_human_demonstration_prior_by_default():
    profile = _load_profile_with_global_rules("configs/profiles/testy_travel.yaml")

    assert profile is not None
    assert "demonstration_rules" not in profile
    assert "Verified human-demonstration rules" not in _profile_prompt(profile)


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
    assert _parse_model_ref("openai:gpt-5.6").provider == "openai"
    assert _parse_model_ref("gpt:gpt-5.6").model == "gpt-5.6"
    assert _parse_model_ref("gemini:gemini-3.5-flash").provider == "gemini"
    assert _parse_model_ref("claude:claude-opus-4-8").provider == "anthropic"


def test_action_outcome_normalization_is_bounded_and_safe():
    outcome = _normalize_action_outcome(
        {
            "status": "FAILURE",
            "expected_result": "two objects merge",
            "observed_change": "objects remained",
            "retry_recommendation": "do_not_retry",
            "confidence": 3,
        }
    )

    assert outcome["status"] == "failure"
    assert outcome["retry_recommendation"] == "do_not_retry"
    assert outcome["confidence"] == 1.0


def test_invalid_action_outcome_defaults_to_inconclusive():
    outcome = _normalize_action_outcome(
        {"status": "maybe", "retry_recommendation": "repeat_forever"}
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["retry_recommendation"] == "reobserve"


def test_do_not_retry_outcome_blocks_exact_failed_action():
    action = {"type": "tap", "x": 10, "y": 20, "x2": None, "y2": None}
    outcome = {
        "status": "failure",
        "retry_recommendation": "do_not_retry",
        "action": dict(action),
    }

    assert _repeats_blocked_action(action, outcome)
    assert not _repeats_blocked_action({**action, "x": 11}, outcome)
    assert not _repeats_blocked_action(
        action, {**outcome, "status": "inconclusive"}
    )


def test_unintended_camera_pan_enters_generic_recovery():
    action = {"type": "swipe", "x": 400, "y": 600, "x2": 1300, "y2": 620}
    plan = {
        "screen_mode": "gameplay",
        "current_goal": "Merge three matching objects",
        "strategy": "Drag one object onto the pair",
    }
    outcome = {
        "status": "failure",
        "observed_change": "No merge; camera panned and the whole island shifted right",
    }

    assert _outcome_reports_unintended_viewport_motion(outcome, action, plan)
    assert _opposite_swipe_direction(action) == "left"
    recovery = _viewport_recovery_plan(
        {"trigger_frame": 4, "attempts": 1, "direction": "left"}
    )
    assert recovery["screen_mode"] == "recovery"
    assert recovery["desired_action"] == "swipe"
    assert "left" in recovery["target_description"]


def test_intended_camera_pan_does_not_reenter_recovery():
    action = {"type": "swipe", "x": 1300, "y": 600, "x2": 900, "y2": 600}
    plan = {
        "screen_mode": "gameplay",
        "current_goal": "Restore full board view",
        "strategy": "Pan camera back to center",
    }
    outcome = {"status": "partial", "observed_change": "camera panned left"}

    assert not _outcome_reports_unintended_viewport_motion(outcome, action, plan)


def test_completed_stage_is_recorded_and_reentry_rejected():
    progress = {"completed": [], "last_completed": None}
    _record_completed_stage_from_perception(
        progress,
        {
            "summary": "A level-completion popup with three stars",
            "visible_text": "레벨 2 | 다음",
        },
    )

    assert progress["last_completed"]["series"] == "level"
    assert progress["last_completed"]["ordinal"] == 2
    repeated = _plan_reenters_completed_stage(
        {
            "current_goal": "Start Level 2",
            "strategy": "Select the next level in ascending order",
            "target_description": "Level 2 button",
        },
        progress,
    )
    assert repeated == "Level 2"


def test_new_series_may_restart_at_one():
    progress = {
        "completed": [{"raw_label": "초원 3", "series": "초원", "ordinal": 3}],
        "last_completed": {"raw_label": "초원 3", "series": "초원", "ordinal": 3},
    }
    plan = {
        "current_goal": "Start 화산 1",
        "strategy": "Enter newly unlocked series",
        "target_description": "화산 1 node",
    }

    assert _plan_reenters_completed_stage(plan, progress) is None
