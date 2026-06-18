from gameagent.agent import ActionValidator
from gameagent.models import Action, ActionType, Observation


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

