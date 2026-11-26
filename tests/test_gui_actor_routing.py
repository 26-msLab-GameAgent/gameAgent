import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gameagent.server.screen_router import load_routing
from test_screen_router import classification, control, setup_server, stage


def mock_grounder(monkeypatch, change=None):
    requests = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        requests.append((request.full_url, payload, timeout))
        import base64
        response = {
            'request_id': payload['request_id'], 'coord_mode': 'norm1',
            'image_width': 100, 'image_height': 100,
            'image_sha256': hashlib.sha256(base64.b64decode(payload['screenshot_base64'])).hexdigest(),
            'candidates': [
                {'rank': 2, 'x': .85, 'y': .25, 'pixel_x': 85, 'pixel_y': 25, 'score': .9},
                {'rank': 1, 'x': .79, 'y': .21, 'pixel_x': 79, 'pixel_y': 21, 'score': .07},
            ],
        }
        if change:
            change(response)
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr('gameagent.clients.gui_actor.urlopen', urlopen)
    return requests


@pytest.mark.parametrize('state,role', [('S', 'start'), ('C', 'next'), ('P', 'close'), ('T', 'confirm')])
def test_semantic_ui_uses_gui_actor_without_selector_or_board(monkeypatch, tmp_path, state, role):
    requests = mock_grounder(monkeypatch)
    raw = classification(state)
    raw['controls'] = [control(role, bbox=None, grounding_description='upper right OK button')]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    result = server.decide(req)
    assert result['action']['x'] == 79  # Rank 1, not the largest score or bbox midpoint.
    assert result['action']['y'] == 21
    assert calls == ['screen_classifier']
    assert requests[0][0] == 'http://127.0.0.1:8010/v1/ground'
    assert requests[0][1]['instruction'] == 'upper right OK button'
    trace = result['pipeline']['screen_routing']
    assert trace['route'] == 'ui_gui_actor'
    assert trace['grounding_call_count'] == 1
    assert trace['grounding_calls'][0]['selected_rank'] == 1
    assert not server._pending_interactions


@pytest.mark.parametrize('change', [
    lambda r: r.update(image_width=200),
    lambda r: r.update(image_sha256='wrong'),
    lambda r: r.update(request_id='stale'),
    lambda r: r.update(coord_mode='pixels'),
    lambda r: r.update(candidates=[]),
    lambda r: r.update(candidates=[{'rank': 1, 'x': .5, 'y': .5, 'pixel_x': 500, 'pixel_y': 50}]),
    lambda r: r.update(candidates=[{'rank': 1, 'x': float('nan'), 'y': .5, 'pixel_x': 50, 'pixel_y': 50}]),
    lambda r: r.update(candidates=[{'rank': 1, 'x': .1, 'y': .1, 'pixel_x': 80, 'pixel_y': 20}]),
])
def test_invalid_grounding_never_clicks(monkeypatch, tmp_path, change):
    mock_grounder(monkeypatch, change)
    raw = classification('P'); raw['controls'] = [control()]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert result['pipeline']['screen_routing']['route'] == 'ui_grounding_error'


def test_grounder_failure_is_bounded_and_logged(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise TimeoutError('offline')
    monkeypatch.setattr('gameagent.clients.gui_actor.urlopen', fail)
    raw = classification('T'); raw['controls'] = [control('confirm')]
    server, req, calls = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert calls == ['screen_classifier']
    assert 'TimeoutError' in result['pipeline']['screen_routing']['grounding_calls'][0]['error']


def test_optional_bbox_rejects_other_button(monkeypatch, tmp_path):
    mock_grounder(monkeypatch)
    raw = classification('P'); raw['controls'] = [control(bbox=[10, 10, 30, 30])]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    assert server.decide(req)['action']['type'] == 'wait'


def test_stage_guard_before_gui_and_classifier_tiebreaker(monkeypatch, tmp_path):
    requests = mock_grounder(monkeypatch)
    raw = classification('C'); raw['stage'] = stage(1)
    server, req, calls = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    server.decide(req)
    raw.clear(); raw.update(classification('S'))
    raw['controls'] = [control('stage', 'old', stage=stage(1), bbox=None),
                       control('stage', 'next', stage=stage(2), bbox=None)]
    raw['preferred_control_id'] = 'old'
    req.frame_id = 2
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert result['pipeline']['screen_routing']['grounding_calls'][0]['target_id'] == 'next'
    assert len(requests) == 1
    assert 'stage 2' in requests[0][1]['instruction']
    assert calls == ['screen_classifier', 'screen_classifier']


@pytest.mark.parametrize('button', [control('purchase'), control(free=False), control(enabled=False)])
def test_disallowed_targets_do_not_reach_grounder(monkeypatch, tmp_path, button):
    requests = mock_grounder(monkeypatch)
    raw = classification('P'); raw['controls'] = [button]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw, {'gui_actor': {'enabled': True}})
    assert server.decide(req)['action']['type'] == 'wait'
    assert requests == []


def recovery_responses(action=None):
    return {
        'perception': {'summary': 'retry popup', 'confidence': .9},
        'planner': {'screen_mode': 'popup', 'desired_action': 'tap',
                    'target_description': 'free retry', 'confidence': .9},
        'policy': {'action': action or {'type': 'tap', 'x': 95, 'y': 5}, 'confidence': .9},
        'rule_learner': {'status': 'failure', 'retry_recommendation': 'change_target', 'confidence': .9},
    }


@pytest.mark.parametrize('raw', [classification('U'), {**classification('B'), 'confidence': .1},
                                RuntimeError('classifier offline')])
def test_unknown_runs_actual_vlm_pipeline_without_board_rect(monkeypatch, tmp_path, raw):
    server, req, calls = setup_server(monkeypatch, tmp_path, raw, responses=recovery_responses())
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert result['action']['x'] == 95  # Outside the former B bbox is allowed on U.
    assert calls == ['screen_classifier', 'perception', 'planner', 'policy']
    assert result['pipeline']['screen_routing']['route'] == 'unknown_pipeline'
    assert len(server._pending_interactions) == 1


def test_unknown_back_and_outcome_verification(monkeypatch, tmp_path):
    raw = classification('U')
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
                                    responses=recovery_responses({'type': 'back'}))
    assert server.decide(req)['action']['type'] == 'back'
    req.frame_id = 2
    result = server.decide(req)
    assert 'rule_learner' in calls
    assert result['pipeline']['previous_outcome'] is not None


def test_persistent_unknown_exhausts_and_resets(monkeypatch, tmp_path):
    raw = classification('U')
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'unknown_recovery': {'max_attempts': 2}}, recovery_responses({'type': 'wait'}))
    for frame in (1, 2):
        req.frame_id = frame
        assert not server.decide(req)['pipeline']['screen_routing']['stop_requested']
    req.frame_id = 3
    result = server.decide(req)
    assert result['pipeline']['screen_routing']['stop_requested']
    assert calls.count('perception') == 2
    raw['state'] = 'L'; req.frame_id = 4
    assert server.decide(req)['pipeline']['screen_routing']['unknown_recovery']['attempts'] == 0
    raw['state'] = 'U'; req.frame_id = 5
    assert server.decide(req)['pipeline']['screen_routing']['route'] == 'unknown_pipeline'
    req.frame_id = 1
    assert server.decide(req)['pipeline']['screen_routing']['unknown_recovery']['attempts'] == 1


def test_unknown_vlm_outage_also_exhausts(monkeypatch, tmp_path):
    server, req, calls = setup_server(monkeypatch, tmp_path, classification('U'),
        {'unknown_recovery': {'max_attempts': 2}}, {'perception': TimeoutError('offline')})
    for frame in (1, 2, 3):
        req.frame_id = frame
        result = server.decide(req)
    assert result['pipeline']['screen_routing']['stop_requested']
    assert calls.count('perception') == 2


def test_board_never_calls_gui_or_solver_when_deferred(monkeypatch, tmp_path):
    requests = mock_grounder(monkeypatch)
    server, req, calls = setup_server(monkeypatch, tmp_path, classification('B'),
        {'gui_actor': {'enabled': True}, 'enable_board_solver': False},
        recovery_responses({'type': 'tap', 'x': 50, 'y': 50}))
    assert server.decide(req)['pipeline']['screen_routing']['route'] == 'board_pipeline'
    assert not requests
    assert calls == ['screen_classifier', 'perception', 'planner', 'policy']


def test_runner_stops_and_logs_exhausted_recovery(monkeypatch, tmp_path):
    from gameagent.runtime.runner import AgentRunner, RunnerOptions
    from gameagent.models import Decision, Observation, ExecutionResult
    from gameagent.agent import ActionValidator
    events = []
    decision = Decision.from_dict({'action': {'type': 'wait', 'duration_ms': 0},
        'pipeline': {'screen_routing': {'stop_requested': True}}})
    logger = SimpleNamespace(path=tmp_path, rule_action_trace_path=tmp_path/'trace',
        start_video_recording=lambda: None, log_step=lambda **kw: events.append('logged'),
        finalize=lambda: events.append('finalized'))
    runner = AgentRunner(
        SimpleNamespace(capture=lambda frame, previous: Observation(frame, 0., 100, 100)),
        SimpleNamespace(model_name='mock', decide=lambda observation: decision),
        SimpleNamespace(execute=lambda action, observation: ExecutionResult(ok=True)),
        ActionValidator(), logger,
        RunnerOptions(max_episode_steps=10, settle_after_action_ms=0, tick_interval_ms=0))
    monkeypatch.setattr(runner, '_install_signal_handlers', lambda: None)
    assert runner.run() == 2
    assert events == ['logged', 'finalized']


@pytest.mark.parametrize('settings', [
    {'gui_actor': {'timeout_s': 0}}, {'gui_actor': {'top_k': 6}},
    {'gui_actor': {'url': 'file:///tmp/socket'}}, {'unknown_recovery': {'max_attempts': 0}},
])
def test_invalid_configuration_rejected(settings):
    with pytest.raises(ValueError):
        load_routing(settings)


def test_real_level11_candidate_rechecked_before_grounding(monkeypatch, tmp_path):
    raw = json.loads((Path(__file__).parent/'fixtures/candy_level11_low_confidence.json').read_text())
    requests = mock_grounder(monkeypatch)
    confirmed = {**raw['controls'][0], 'confidence': .93, 'label': '11',
                 'bbox': [70, 10, 90, 30],
                 'stage': stage(11), 'grounding_description': 'blue level tile labeled 11'}
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'gui_actor': {'enabled': True}, 'ui_recheck_model': 'qwen:8B'},
        {'screen_ui_recheck': {'controls': [confirmed], 'evidence': '11 visible on blue tile'}})
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert calls == ['screen_classifier', 'screen_ui_recheck']
    assert len(requests) == 1
    trace = result['pipeline']['screen_routing']
    assert trace['raw_classification']['controls'][0]['confidence'] == .62
    assert trace['classification']['controls'][0]['confidence'] == .93
    assert trace['classification']['ui_recheck']['confirmed_ids'] == ['current_level_tile']


@pytest.mark.parametrize('changes', [{'confidence': .74}, {'free': False}, {'enabled': False},
                                   {'role': 'purchase'}, {'id': 'invented'}, {'bbox': None}])
def test_ui_recheck_does_not_bypass_guards(monkeypatch, tmp_path, changes):
    requests = mock_grounder(monkeypatch)
    raw = classification('S')
    raw['controls'] = [control('stage', confidence=.62, bbox=None)]
    confirmed = {**raw['controls'][0], 'confidence': .95, 'bbox': [70, 10, 90, 30], **changes}
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'gui_actor': {'enabled': True}, 'ui_recheck_model': 'qwen:8B'},
        {'screen_ui_recheck': {'controls': [confirmed]}})
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert result['pipeline']['screen_routing']['route'] == 'ui_recheck_wait'
    assert requests == []


def test_ui_recheck_still_obeys_stage_order(monkeypatch, tmp_path):
    requests = mock_grounder(monkeypatch)
    raw = classification('S'); raw['controls'] = [control('stage', confidence=.62, bbox=None)]
    server, req, _ = setup_server(monkeypatch, tmp_path, raw,
        {'gui_actor': {'enabled': True}, 'ui_recheck_model': 'qwen:8B'},
        {'screen_ui_recheck': {'controls': [control('stage', stage=stage(10))]}})
    req.frame_id = 2
    server._stage_progress = {'completed': [{'series': 'level', 'ordinal': 10}],
                              'last_completed': {'series': 'level', 'ordinal': 10}}
    result = server.decide(req)
    assert result['pipeline']['screen_routing']['route'] == 'stage_guard'
    assert result['action']['type'] == 'wait'
    assert not requests


@pytest.mark.parametrize('state,controls', [('P', []), ('S', []),
                                         ('P', [control('purchase', confidence=.6)]),
                                         ('S', [control('stage', enabled=False, confidence=.6)])])
def test_ui_recheck_does_not_invent_missing_or_forbidden_controls(monkeypatch, tmp_path, state, controls):
    raw = classification(state); raw['controls'] = controls
    server, req, calls = setup_server(monkeypatch, tmp_path, raw,
        {'gui_actor': {'enabled': True}, 'ui_recheck_model': 'qwen:8B'})
    assert server.decide(req)['action']['type'] == 'wait'
    assert calls == ['screen_classifier']
