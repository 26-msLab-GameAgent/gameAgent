import json
import pytest
from gameagent.server.screen_router import ScreenRoutingServer, load_routing, normalize_screen
from gameagent.server.vlm_server import DecisionServer, DecideRequest, Screen


def classification(state='P'):
    return dict(state=state, confidence=.95, evidence='visible screen',
                board_bbox=[10, 10, 90, 90], board_confidence=.95, controls=[])


def control(role='close', ident='x', **changes):
    result = dict(id=ident, role=role, bbox=[70, 10, 90, 30],
                  confidence=.95, enabled=True, free=True, label='X')
    result.update(changes)
    return result


def setup_server(monkeypatch, tmp_path, raw, settings=None, responses=None):
    outputs = {'screen_classifier': raw, **(responses or {})}
    calls = []
    def generate(self, req, path, prompt, agent_name, include_image):
        calls.append(agent_name)
        result = outputs[agent_name]
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    server = ScreenRoutingServer(model_id='openai:test-board', screen_routing=settings)
    from PIL import Image
    image = tmp_path/'frame.png'
    Image.new('RGB', (100, 100)).save(image)
    req = DecideRequest(frame_id=1, screen=Screen(width=100, height=100), image_path=str(image))
    return server, req, calls


def test_popup_only_closes_and_skips_board(monkeypatch, tmp_path):
    raw = classification(); raw['controls'] = [control('purchase', 'buy'), control()]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert result['action']['x'] == 80
    assert calls == ['screen_classifier']
    assert server._pending_interactions == []
    annotation = result['pipeline']['screen_routing']
    assert annotation['hosted_call_count'] == 0
    assert annotation['reviewed'] is False
    assert annotation['raw_classification'] == raw


@pytest.mark.parametrize('state', ['L', 'P'])
def test_no_control_waits(monkeypatch, tmp_path, state):
    server, req, calls = setup_server(monkeypatch, tmp_path, classification(state))
    assert server.decide(req)['action']['type'] == 'wait'
    assert calls == ['screen_classifier']


@pytest.mark.parametrize('change', [
    {'free': False}, {'enabled': False}, {'confidence': .2},
    {'bbox': [-1, 0, 10, 10]}, {'bbox': [0, 0, float('nan'), 10]},
    {'role': 'start'}, {'role': 'purchase'},
])
def test_unusable_popup_target_never_tapped(monkeypatch, tmp_path, change):
    raw = classification(); raw['controls'] = [control(**change)]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw)
    assert server.decide(req)['action']['type'] == 'wait'


@pytest.mark.parametrize('state,role', [('S', 'start'), ('C', 'next')])
def test_semantic_ui_rules(monkeypatch, tmp_path, state, role):
    raw = classification(state); raw['controls'] = [control(role)]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    assert server.decide(req)['action']['type'] == 'tap'
    assert calls == ['screen_classifier']


def test_ambiguous_stage_waits(monkeypatch, tmp_path):
    raw = classification('S'); raw['controls'] = [control('stage', 'a'), control('stage', 'b')]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw)
    assert server.decide(req)['action']['type'] == 'wait'


@pytest.mark.parametrize('target,expected', [('a', 'tap'), ('invented', 'wait')])
def test_selector_only_uses_grounded_ids(monkeypatch, tmp_path, target, expected):
    raw = classification('S'); raw['controls'] = [control('stage', 'a'), control('stage', 'b')]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'states': {'S': {'model': 'qwen:3B'}}},
        {'screen_S': {'target_id': target, 'confidence': .9}})
    assert server.decide(req)['action']['type'] == expected
    assert calls == ['screen_classifier', 'screen_S']


@pytest.mark.parametrize('x,expected', [(30, 'swipe'), (99, 'wait')])
def test_real_board_pipeline_and_final_guard(monkeypatch, tmp_path, x, expected):
    server, req, calls = setup_server(monkeypatch, tmp_path, classification('B'), responses={
        'perception': {'summary': 'stable board', 'visible_text': 'none', 'confidence': .9},
        'planner': {'screen_mode': 'gameplay', 'desired_action': 'swipe',
                    'target_description': 'two board pieces', 'confidence': .9},
        'policy': {'action': {'type': 'swipe', 'x': 20, 'y': 20, 'x2': x, 'y2': 20},
                   'confidence': .9},
    })
    result = server.decide(req)
    assert result['action']['type'] == expected
    assert calls == ['screen_classifier', 'perception', 'planner', 'policy']
    assert result['pipeline']['screen_routing']['hosted_call_count'] == 3
    if expected == 'wait':
        assert result['rejected_action']['x2'] == 99
        assert not server._pending_interactions


def test_classifier_error_can_disable_unknown_recovery(monkeypatch, tmp_path):
    server, req, calls = setup_server(monkeypatch, tmp_path, RuntimeError('offline'),
                                    {'unknown_recovery': {'enabled': False}})
    assert server.decide(req)['action']['type'] == 'wait'
    assert calls == ['screen_classifier']


@pytest.mark.parametrize('changes', [
    {'confidence': .1}, {'board_bbox': None}, {'board_confidence': .1}, {'state': 'nonsense'},
])
def test_uncertainty_becomes_unknown(changes):
    raw = classification('B'); raw.update(changes)
    assert normalize_screen(raw, Screen(width=100, height=100), .8)['state'] == 'U'


def test_config_cannot_enable_popup_purchase_or_swipe():
    with pytest.raises(ValueError):
        load_routing({'states': {'P': {'roles': ['purchase']}}})
    with pytest.raises(ValueError):
        load_routing({'states': {'P': {'allowed_actions': ['swipe']}}})


def test_trace_keeps_annotation(tmp_path):
    from gameagent.storage.episode import EpisodeLogger
    from gameagent.models import Observation, Decision, Action, ActionType, ExecutionResult
    logger = object.__new__(EpisodeLogger)
    logger.session_id = 'test'
    logger.rule_action_trace_path = tmp_path/'trace.jsonl'
    annotation = {'state': 'P', 'reviewed': False}
    logger._append_rule_action_trace(
        Observation(frame_id=1, timestamp=1., width=100, height=100),
        Decision(action=Action(type=ActionType.WAIT), raw_response={'pipeline': {'screen_routing': annotation}}),
        ExecutionResult(ok=True), 'frame.png')
    trace = json.loads(logger.rule_action_trace_path.read_text())
    assert trace['screen_routing'] == annotation
    assert trace['image_path'] == 'frame.png'


def stage(number):
    return {'series': 'level', 'ordinal': number, 'confidence': .95}


@pytest.mark.parametrize('role', ['stage', 'start'])
@pytest.mark.parametrize('number,expected', [(1, 'wait'), (2, 'tap'), (3, 'wait'), (None, 'wait')])
def test_clear_to_stage_guard_including_single_button(monkeypatch, tmp_path, role, number, expected):
    raw = classification('C'); raw['stage'] = stage(1)
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    server.decide(req)
    assert server._stage_progress['last_completed']['ordinal'] == 1
    raw.clear(); raw.update(classification('S'))
    raw['controls'] = [control(role, stage=stage(number) if number else None)]
    req.frame_id = 2
    result = server.decide(req)
    assert result['action']['type'] == expected
    assert calls == ['screen_classifier', 'screen_classifier']
    guard = result['pipeline']['screen_routing']['classification']['stage_guard']
    assert guard['expected'] == {'series': 'level', 'ordinal': 2}
    assert result['pipeline']['screen_routing']['stage_progress']['last_completed']['ordinal'] == 1


def test_start_dialog_uses_visible_stage_and_new_run_resets(monkeypatch, tmp_path):
    raw = classification('C'); raw['stage'] = stage(1)
    server, req, _ = setup_server(monkeypatch, tmp_path, raw)
    server.decide(req)
    raw.clear(); raw.update(classification('S'))
    raw['visible_text'] = '레벨 2'
    raw['controls'] = [control('start')]
    req.frame_id = 2
    assert server.decide(req)['action']['type'] == 'tap'
    raw['visible_text'] = '레벨 1'
    req.frame_id = 3
    assert server.decide(req)['action']['type'] == 'wait'
    req.frame_id = 1
    assert server.decide(req)['action']['type'] == 'tap'
    assert server._stage_progress['completed'] == []


def test_model_cannot_select_filtered_stage(monkeypatch, tmp_path):
    raw = classification('C'); raw['stage'] = stage(1)
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'states': {'S': {'model': 'qwen:3B'}}},
        {'screen_S': {'target_id': 'old', 'confidence': .95}})
    server.decide(req)
    raw.clear(); raw.update(classification('S'))
    raw['controls'] = [control('stage', 'old', stage=stage(1)),
                       control('stage', 'next-a', stage=stage(2)),
                       control('stage', 'next-b', stage=stage(2))]
    req.frame_id = 2
    assert server.decide(req)['action']['type'] == 'wait'
    assert calls[-1] == 'screen_S'


def test_progress_advances_to_three_after_two_and_never_regresses(monkeypatch, tmp_path):
    raw = classification('C'); raw['stage'] = stage(2)
    server, req, _ = setup_server(monkeypatch, tmp_path, raw)
    server.decide(req)
    raw['stage'] = stage(1); req.frame_id = 2
    server.decide(req)
    raw.clear(); raw.update(classification('S'))
    raw['controls'] = [control('stage', '2', stage=stage(2)), control('stage', '3', stage=stage(3))]
    req.frame_id = 3
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert '(3)' in result['action']['reason']


def test_uncertain_completion_is_not_remembered(monkeypatch, tmp_path):
    raw = classification('C'); raw['stage'] = {**stage(1), 'confidence': .1}
    server, req, _ = setup_server(monkeypatch, tmp_path, raw)
    server.decide(req)
    assert server._stage_progress['completed'] == []


def test_runtime_config_uses_api_classifier_and_local_board(monkeypatch, tmp_path):
    from gameagent.server.vlm_server import _load_profile
    server, req, _ = setup_server(monkeypatch, tmp_path, classification('B'), responses={
        'perception': {'summary': 'stable board', 'confidence': .9},
        'planner': {'screen_mode': 'gameplay', 'desired_action': 'swipe', 'confidence': .9},
        'policy': {'action': {'type': 'swipe', 'x': 200, 'y': 200, 'x2': 300, 'y2': 200},
                   'confidence': .9},
    })
    cfg = _load_profile('configs/model_pipelines/all_qwen_local.yaml')
    server = ScreenRoutingServer(model_id=cfg['default_model'], agent_model_ids=cfg['agents'],
        screen_routing=_load_profile('configs/screen_routing/default.yaml'))
    result = server.decide(req)
    assert result['action']['type'] == 'swipe'
    trace = result['pipeline']['screen_routing']
    assert trace['hosted_call_count'] == 1
    for role in ('perception', 'rule_learner', 'planner', 'policy'):
        assert server.agent_model_refs[role].provider not in {'openai', 'gemini', 'anthropic'}
    assert trace['calls'][0]['role'] == 'screen_classifier'
    assert all('Qwen3-VL-8B' in c['model'] for c in trace['calls'][1:])


@pytest.mark.parametrize('role', ['start', 'confirm', 'next', 'continue', 'skip', 'close', 'dismiss'])
def test_tutorial_buttons_use_fast_path_without_stage_identity(monkeypatch, tmp_path, role):
    raw = classification('T')
    raw['controls'] = [control(role, 'tutorial_button')]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    req.frame_id = 2
    server._stage_progress = {'completed': [{'series': 'level', 'ordinal': 1}],
                              'last_completed': {'series': 'level', 'ordinal': 1}}
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert 'T:' in result['action']['reason']
    assert calls == ['screen_classifier']
    assert server._pending_interactions == []
    assert server._stage_progress['last_completed']['ordinal'] == 1
    assert result['pipeline']['screen_routing']['state'] == 'T'
    assert result['pipeline']['screen_routing']['taxonomy_version'] == 'SBPCTLU-v2'


@pytest.mark.parametrize('button', [None, control('purchase'), control('confirm', free=False),
                                   control('start', enabled=False), control('next', confidence=.1)])
def test_tutorial_without_usable_button_waits(monkeypatch, tmp_path, button):
    raw = classification('T'); raw['controls'] = [button] if button else []
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    assert server.decide(req)['action']['type'] == 'wait'
    assert calls == ['screen_classifier']


def test_tutorial_prefers_start_over_close(monkeypatch, tmp_path):
    raw = classification('T')
    raw['controls'] = [control('close', 'x'), control('start', 'start_game')]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw)
    result = server.decide(req)
    assert '(start_game)' in result['action']['reason']
    assert calls == ['screen_classifier']
