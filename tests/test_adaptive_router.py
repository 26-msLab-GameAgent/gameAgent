import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gameagent.server.adaptive_router import AdaptiveRoutingServer
from gameagent.server.fast_screens import VerifiedScreens, thumbnail
from gameagent.server.screen_router import normalize_screen
from gameagent.server.vlm_server import DecisionServer, DecideRequest, PreviousAction, Screen
from test_screen_router import classification, control


def server_fixture(monkeypatch, tmp_path, raw, adaptive=None, gui=False):
    calls = []
    def generate(self, req, path, prompt, agent_name, include_image):
        calls.append(agent_name)
        assert agent_name == 'screen_classifier', 'Legacy multi-call pipeline must never run'
        if isinstance(raw, Exception):
            raise raw
        return json.dumps(raw)
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    server = AdaptiveRoutingServer(model_id='openai:test', screen_routing={
        'classifier_model': 'openai:test', 'gui_actor': {'enabled': gui},
        'adaptive': {'templates': str(tmp_path/'missing.json'), **(adaptive or {})}})
    path = tmp_path/'frame.png'
    Image.new('RGB', (100, 100), 'black').save(path)
    req = DecideRequest(frame_id=1, screen=Screen(width=100, height=100), image_path=str(path))
    return server, req, calls


def advance(req, result):
    req.frame_id += 1
    req.previous_action = PreviousAction(**result['action'])


def trace(result):
    return result['pipeline']['screen_routing']


def test_low_confidence_animation_waits_without_extra_calls(monkeypatch, tmp_path):
    raw = {**classification('L'), 'confidence': .6}
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw)
    result = server.decide(req)
    assert trace(result)['state'] == 'L'
    assert result['action']['type'] == 'wait'
    assert result['action']['duration_ms'] == 200
    assert calls == ['screen_classifier']


def test_state_confidence_does_not_remove_button_safety():
    screen = Screen(width=100, height=100)
    raw = {**classification('S'), 'confidence': .2,
           'controls': [control('start', free=False), control('start', 'low', confidence=.1)]}
    state = normalize_screen(raw, screen, .8, preserve_state=True)
    assert state['state'] == 'S'
    assert state['controls'] == []
    raw = {**classification('B'), 'board_bbox': None, 'confidence': .2}
    assert normalize_screen(raw, screen, .8, preserve_state=True)['state'] == 'U'


@pytest.mark.parametrize('raw', [classification('U'), TimeoutError('offline')])
def test_unknown_bounded_without_legacy_pipeline(monkeypatch, tmp_path, raw):
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                       {'max_recovery_calls': 2, 'reobserve_steps': 1})
    for _ in range(8):
        result = server.decide(req)
        if trace(result)['stop_requested']:
            break
        advance(req, result)
    assert trace(result)['stop_requested']
    assert len(calls) == 2


def test_label_changes_do_not_reset_budget(monkeypatch, tmp_path):
    raw = classification('U')
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                       {'max_recovery_calls': 2, 'reobserve_steps': 1})
    for i in range(10):
        raw['state'] = 'L' if i % 2 else 'U'
        result = server.decide(req)
        if trace(result)['stop_requested']:
            break
        advance(req, result)
    assert trace(result)['stop_requested']
    assert len(calls) == 2


def test_retry_uses_visible_free_button_and_counts_executed_actions(monkeypatch, tmp_path):
    raw = classification('R')
    raw['controls'] = [control('retry', 'retry', label='재시도')]
    raw['stage'] = {'series': 'level', 'ordinal': 10, 'confidence': .99}
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                       {'max_retries_per_stage': 1, 'reobserve_steps': 1})
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert trace(result)['retry_counts']['10'] == 0  # not executed yet
    advance(req, result)
    server.decide(req)  # short observation verifies the prior execution
    assert server._retry_counts['10'] == 1
    req.frame_id += 1
    result = server.decide(req)
    assert trace(result)['stop_requested']


@pytest.mark.parametrize('metadata', [{'dry_run': True}, {'previous_execution_ok': False},
                                     {'skipped_action_frame_id': 1}])
def test_skipped_retry_does_not_consume_retry_budget(monkeypatch, tmp_path, metadata):
    raw = classification('R'); raw['controls'] = [control('retry')]
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw)
    result = server.decide(req)
    advance(req, result); req.metadata = metadata
    server.decide(req)
    assert sum(server._retry_counts.values()) == 0


def test_paid_retry_never_clicked(monkeypatch, tmp_path):
    raw = classification('R'); raw['controls'] = [control('retry', free=False)]
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw)
    assert server.decide(req)['action']['type'] == 'wait'


def test_local_ui_goes_directly_to_gui_actor(monkeypatch, tmp_path):
    raw = classification('C'); raw['controls'] = [control('next')]
    raw['stage'] = {'series': 'level', 'ordinal': 1, 'confidence': 1}
    raw['evidence'] = 'verified test template'
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw, gui=True)
    server.fast_screens.classify = lambda *a, **kw: deepcopy(raw)
    grounded = []
    def ground(*args):
        grounded.append(args)
        return {'type': 'tap', 'x': 80, 'y': 20, 'duration_ms': 80}, {}
    server.gui_actor = SimpleNamespace(ground=ground)
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert trace(result)['classification_source'] == 'local_template'
    assert calls == [] and len(grounded) == 1


def test_template_failure_escalates_instead_of_looping(monkeypatch, tmp_path):
    raw = classification('S'); raw['controls'] = [control('start')]
    raw['evidence'] = 'verified test template'
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                       {'reobserve_steps': 1, 'max_same_ui_attempts': 1})
    server.fast_screens.classify = lambda *a, **kw: deepcopy(raw)
    result = server.decide(req)
    advance(req, result)
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    advance(req, result)
    server.decide(req)
    assert calls == ['screen_classifier']


def test_ui_progress_survives_intermediate_animation(monkeypatch, tmp_path):
    raw = classification('S'); raw['controls'] = [control('start')]
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                       {'reobserve_steps': 1})
    result = server.decide(req)
    advance(req, result)
    Image.new('RGB', (100, 100), 'red').save(req.image_path)
    result = server.decide(req)  # deferred L, should not forget pending S tap
    advance(req, result)
    raw.update(classification('B'))
    result = server.decide(req)
    assert trace(result)['unknown_recovery']['calls_without_progress'] == 0
    assert result['pipeline']['previous_outcome']['status'] == 'success'


@pytest.mark.parametrize('action', [
    {'type': 'swipe', 'x': 80, 'y': 50, 'x2': 99, 'y2': 50},
    {'type': 'swipe', 'x': 30, 'y': 30, 'x2': 50, 'y2': 50},
    {'type': 'tap', 'x': 50, 'y': 50},
    {'type': 'swipe', 'x': 15, 'y': 50, 'x2': 85, 'y2': 50},
])
def test_single_call_board_action_validated(monkeypatch, tmp_path, action):
    raw = {**classification('B'), 'fallback_action': action}
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw)
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert calls == ['screen_classifier']


def test_stable_confirmed_board_skips_semantic_api(monkeypatch, tmp_path):
    raw = classification('B')
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw)
    server._solver_profile = {'fallback_planner': 'candy_crush_grid'}
    monkeypatch.setattr('gameagent.server.adaptive_router._candy_board_signature', lambda *a: 'red|green')
    monkeypatch.setattr('gameagent.server.adaptive_router._plan_candy_crush_swipe', lambda *a:
                        {'type': 'swipe', 'x': 30, 'y': 50, 'x2': 50, 'y2': 50, 'duration_ms': 220})
    result = server.decide(req)
    advance(req, result)
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'local_board'
    assert calls == ['screen_classifier']


def test_changed_chrome_forces_reobservation(monkeypatch, tmp_path):
    server, req, calls = server_fixture(monkeypatch, tmp_path, classification('B'))
    result = server.decide(req)
    Image.new('RGB', (100, 100), 'white').save(req.image_path)
    advance(req, result)
    result = server.decide(req)
    assert trace(result)['classification_source'] != 'local_board'
    assert result['action']['type'] == 'wait'


def stub_board(monkeypatch, tmp_path, adaptive=None):
    server, req, calls = server_fixture(monkeypatch, tmp_path, classification('B'), adaptive)
    signature = ['red|green|yellow']
    monkeypatch.setattr('gameagent.server.adaptive_router._candy_board_signature',
                        lambda *a: signature[0])
    monkeypatch.setattr('gameagent.server.adaptive_router._plan_candy_crush_swipe', lambda *a:
                        {'type': 'swipe', 'x': 30, 'y': 50, 'x2': 50, 'y2': 50, 'duration_ms': 220})
    server._solver_profile = {'fallback_planner': 'candy_crush_grid'}
    return server, req, calls, signature


def test_semantic_board_progress_ignores_whole_screen_motion(monkeypatch, tmp_path):
    server, req, calls, signature = stub_board(monkeypatch, tmp_path,
        {'local_board': False, 'reobserve_steps': 1, 'max_recovery_calls': 1})
    monkeypatch.setattr('gameagent.server.adaptive_router.stationary', lambda *a: False)
    result = server.decide(req)
    for _ in range(5):
        advance(req, result)
        signature[0] = 'green|red|yellow' if signature[0].startswith('red') else 'red|green|yellow'
        result = server.decide(req)
        assert trace(result)['classification_source'] == 'deferred'
        assert 'classifier not called' in result['intent']
        advance(req, result)
        result = server.decide(req)
        assert result['pipeline']['previous_outcome']['status'] == 'success'
        assert trace(result)['unknown_recovery']['calls_without_progress'] == 0
        assert not trace(result)['stop_requested']
    assert len(calls) == 6  # Normal classification calls cannot exhaust U's budget.


def test_local_grid_reuse_ignores_animated_hud(monkeypatch, tmp_path):
    from PIL import ImageDraw
    server, req, calls, _ = stub_board(monkeypatch, tmp_path)
    result = server.decide(req)
    advance(req, result)
    # Motion outside the board, in the HUD region excluded by the chrome check.
    with Image.open(req.image_path) as im:
        ImageDraw.Draw(im).rectangle((0, 40, 5, 80), fill='white')
        im.save(req.image_path)
    result = server.decide(req)
    assert not trace(result)['stability']['whole_screen']
    assert trace(result)['stability']['local_grid']
    assert trace(result)['classification_source'] == 'local_board'
    assert len(calls) == 1


def test_changed_local_grid_requires_confirmation_then_resets_budget(monkeypatch, tmp_path):
    server, req, calls, signature = stub_board(monkeypatch, tmp_path)
    result = server.decide(req)
    advance(req, result)
    signature[0] = 'green|red|yellow'
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'deferred'
    assert result['pipeline']['previous_outcome'] is None
    # Verified local progress must be checked even if a prior U exhausted its budget.
    server._recovery_calls = server.adaptive['max_recovery_calls']
    advance(req, result)
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'local_board'
    assert result['pipeline']['previous_outcome']['status'] == 'success'
    assert not trace(result)['stop_requested']
    assert server._recovery_calls == 0
    assert len(calls) == 1
    # Do not carry old waits over to the next locally issued swipe.
    assert server._deferred == 0
    advance(req, result)
    signature[0] = 'yellow|red|green'
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'deferred'
    assert len(calls) == 1


def test_grid_labels_do_not_hide_a_new_board_overlay(monkeypatch, tmp_path):
    from PIL import ImageDraw
    server, req, _, _ = stub_board(monkeypatch, tmp_path)
    result = server.decide(req)
    advance(req, result)
    with Image.open(req.image_path) as im:
        ImageDraw.Draw(im).rectangle((30, 30, 70, 70), fill='white')
        im.save(req.image_path)
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'deferred'
    assert result['action']['type'] == 'wait'


@pytest.mark.parametrize('new_signature,metadata', [
    ('red|unknown|yellow', {}),
    ('red|green', {}),
    ('green|red|yellow', {'previous_execution_ok': False}),
    ('green|red|yellow', {'dry_run': True}),
    ('green|red|yellow', {'skipped_action_frame_id': 1}),
])
def test_uncertain_or_unexecuted_grid_change_is_not_progress(monkeypatch, tmp_path, new_signature, metadata):
    server, req, _, signature = stub_board(monkeypatch, tmp_path,
                                          {'local_board': False, 'reobserve_steps': 1})
    result = server.decide(req)
    advance(req, result)
    req.metadata = metadata
    signature[0] = new_signature
    result = server.decide(req)
    advance(req, result)
    result = server.decide(req)
    assert result['pipeline']['previous_outcome'] is None
    assert trace(result)['unknown_recovery']['no_progress_actions'] == (0 if metadata else 1)


def test_unchanged_board_does_not_stop_at_disabled_no_progress_limit(monkeypatch, tmp_path):
    server, req, _, _ = stub_board(monkeypatch, tmp_path, {'max_no_progress_steps': 5})
    for _ in range(7):
        result = server.decide(req)
        advance(req, result)
    assert not trace(result)['stop_requested']
    assert trace(result)['stop_reason'] is None
    assert trace(result)['unknown_recovery']['calls_without_progress'] == 0


def test_reobservation_never_consumes_no_progress_actions(monkeypatch, tmp_path):
    server, req, _, _ = stub_board(monkeypatch, tmp_path,
        {'local_board': False, 'reobserve_steps': 20, 'max_no_progress_steps': 1})
    result = server.decide(req)
    for _ in range(15):
        advance(req, result)
        result = server.decide(req)
        assert trace(result)['classification_source'] == 'deferred'
        assert trace(result)['unknown_recovery']['no_progress_actions'] == 1
        assert not trace(result)['stop_requested']


def moves_board_fixture(monkeypatch, tmp_path):
    raw = {**classification('B'), 'moves_remaining': {'value': 15, 'confidence': .99},
           'fallback_action': {'type': 'swipe', 'x': 30, 'y': 50, 'x2': 50, 'y2': 50}}
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw,
                                        {'local_board': False, 'reobserve_steps': 1})
    signature = [None]
    monkeypatch.setattr('gameagent.server.adaptive_router._candy_board_signature', lambda *a: signature[0])
    return server, req, raw, signature, calls


def next_semantic(server, req, result):
    advance(req, result)
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'deferred'
    advance(req, result)
    return server.decide(req)


def test_counter_decrease_proves_fallback_swipe_progress_without_grid(monkeypatch, tmp_path):
    server, req, raw, _, calls = moves_board_fixture(monkeypatch, tmp_path)
    result = server.decide(req)
    for moves in (14, 13, 12, 11, 10):
        assert result['action']['type'] == 'swipe'
        raw['moves_remaining']['value'] = moves
        result = next_semantic(server, req, result)
        assert 'remaining moves decreased' in result['pipeline']['previous_outcome']['evidence']
        assert trace(result)['unknown_recovery']['no_progress_actions'] == 0
        assert not trace(result)['stop_requested']
    assert len(calls) == 6


@pytest.mark.parametrize('counter', [None, 14, {'value': '14', 'confidence': 1},
    {'value': True, 'confidence': 1}, {'value': 14, 'confidence': .5},
    {'value': 14, 'confidence': float('nan')}, {'value': -1, 'confidence': 1},
    {'value': 14, 'confidence': True}, {'value': 14, 'confidence': 2}])
def test_counter_requires_explicit_valid_confident_reading(monkeypatch, tmp_path, counter):
    server, req, raw, _, _ = moves_board_fixture(monkeypatch, tmp_path)
    result = server.decide(req)
    raw.update(moves_remaining=counter, visible_text='14 목표 5 3 3')
    result = next_semantic(server, req, result)
    assert trace(result)['board_progress']['moves_remaining'] is None
    assert result['pipeline']['previous_outcome'] is None


@pytest.mark.parametrize('metadata', [{'dry_run': True}, {'previous_execution_ok': False},
                                     {'skipped_action_frame_id': 1}])
def test_counter_change_does_not_credit_unexecuted_swipe(monkeypatch, tmp_path, metadata):
    server, req, raw, _, _ = moves_board_fixture(monkeypatch, tmp_path)
    result = server.decide(req)
    req.metadata = metadata
    raw['moves_remaining']['value'] = 14
    result = next_semantic(server, req, result)
    assert result['pipeline']['previous_outcome'] is None
    assert trace(result)['unknown_recovery']['no_progress_actions'] == 0


@pytest.mark.parametrize('moves', [15, 16, 5])
def test_counter_increase_or_implausible_drop_is_not_progress(monkeypatch, tmp_path, moves):
    server, req, raw, _, _ = moves_board_fixture(monkeypatch, tmp_path)
    result = server.decide(req)
    raw['moves_remaining']['value'] = moves
    result = next_semantic(server, req, result)
    assert result['pipeline']['previous_outcome'] is None
    assert trace(result)['unknown_recovery']['no_progress_actions'] == 1


def test_missing_grid_does_not_erase_last_confirmed_signature(monkeypatch, tmp_path):
    server, req, raw, signature, _ = moves_board_fixture(monkeypatch, tmp_path)
    raw['moves_remaining'] = None
    signature[0] = 'red|green|yellow'
    result = server.decide(req)
    signature[0] = None
    result = next_semantic(server, req, result)
    assert server._pending_effect['signature'] == 'red|green|yellow'
    signature[0] = 'green|red|yellow'
    result = next_semantic(server, req, result)
    assert result['pipeline']['previous_outcome']['status'] == 'success'
    assert trace(result)['unknown_recovery']['no_progress_actions'] == 0


def test_stage_change_clears_old_grid_and_counter_baselines(monkeypatch, tmp_path):
    server, req, raw, signature, _ = moves_board_fixture(monkeypatch, tmp_path)
    raw['stage'] = {'series': 'level', 'ordinal': 1, 'confidence': 1}
    signature[0] = 'red|green|yellow'
    result = server.decide(req)
    raw['stage']['ordinal'] = 2
    raw['moves_remaining']['value'] = 14
    signature[0] = 'green|red|yellow'
    result = next_semantic(server, req, result)
    assert result['pipeline']['previous_outcome'] is None


def test_cached_board_does_not_reuse_stale_moves_reading(monkeypatch, tmp_path):
    raw = {**classification('B'), 'moves_remaining': {'value': 15, 'confidence': .99}}
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw)
    server._solver_profile = {'fallback_planner': 'candy_crush_grid'}
    monkeypatch.setattr('gameagent.server.adaptive_router._candy_board_signature', lambda *a: 'red|green')
    monkeypatch.setattr('gameagent.server.adaptive_router._plan_candy_crush_swipe', lambda *a:
        {'type': 'swipe', 'x': 30, 'y': 50, 'x2': 50, 'y2': 50})
    result = server.decide(req)
    advance(req, result)
    result = server.decide(req)
    assert trace(result)['classification_source'] == 'local_board'
    assert trace(result)['board_progress']['moves_remaining'] is None
    assert trace(result)['board_progress']['swipes_since_counter_read'] == 1


def test_new_session_resets_recovery(monkeypatch, tmp_path):
    server, req, _ = server_fixture(monkeypatch, tmp_path, classification('U'),
                                    {'max_recovery_calls': 1, 'reobserve_steps': 1})
    for _ in range(3):
        result = server.decide(req); advance(req, result)
    assert trace(result)['stop_requested']
    req.frame_id = 1; req.previous_action = None
    assert not trace(server.decide(req))['stop_requested']


def test_templates_keep_level_identity(tmp_path):
    # Synthetic input keeps this regression independent of historical gameplay.
    import base64
    import io
    image = io.BytesIO()
    Image.new('RGB', (320, 192), 'white').save(image, format='PNG')
    path = tmp_path / 'templates.json'
    path.write_text(json.dumps({'templates': [{
        'name': 'synthetic-level', 'screen': [1920, 1080], 'anchors': [],
        'image': base64.b64encode(image.getvalue()).decode(),
        'classification': {**classification('S'),
                           'stage': {'series': 'level', 'ordinal': 7, 'confidence': 1}},
    }]}))
    matcher = VerifiedScreens(path)
    assert matcher.entries
    for entry, pixels in matcher.entries:
        found = matcher.classify(pixels, Screen(width=1920, height=1080))
        assert found['state'] == entry['classification']['state']
        assert found.get('stage') == entry['classification'].get('stage')
    # Unknown screens must not acquire a click target.
    assert matcher.classify(np.zeros((192,320,3)), Screen(width=1920,height=1080)) is None


def test_real_map_response_without_bbox_reaches_gui_actor(monkeypatch, tmp_path):
    from pathlib import Path
    from test_gui_actor_routing import mock_grounder
    raw = json.loads((Path(__file__).parent/'fixtures/candy_map_missing_bbox.json').read_text())
    requests = mock_grounder(monkeypatch)
    server, req, calls = server_fixture(monkeypatch, tmp_path, raw, gui=True)
    result = server.decide(req)
    assert result['action']['type'] == 'tap'
    assert (result['action']['x'], result['action']['y']) == (79, 21)
    assert calls == ['screen_classifier'] and len(requests) == 1
    assert trace(result)['route'] == 'ui_gui_actor'
    assert trace(result)['grounding_calls'][0]['target_id'] == 'lvl5'
    assert 'stage 5' in requests[0][1]['instruction']
    assert server._pending_effect['role'] == 'stage'
    assert server._pending_effect['ui_key'][1] == 5


@pytest.mark.parametrize('mode,role', [('S','start'), ('C','next'), ('T','confirm'),
                                      ('P','close'), ('R','retry')])
def test_semantic_ui_without_bbox_keeps_outcome_tracking(monkeypatch, tmp_path, mode, role):
    from test_gui_actor_routing import mock_grounder
    requests = mock_grounder(monkeypatch)
    raw = classification(mode)
    raw['controls'] = [control(role, bbox=None, grounding_description='visible green button')]
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw, gui=True)
    result = server.decide(req)
    assert result['action']['type'] == 'tap' and len(requests) == 1
    assert server._pending_effect['role'] == role
    # Observe exactly the same state after a successful dispatch: count the failure.
    server.fast_screens.classify = lambda *a, **kw: deepcopy(raw)
    previous_key = server._pending_effect['ui_key']
    advance(req, result)
    server.decide(req)
    assert server._failed_ui[previous_key] == 1


@pytest.mark.parametrize('changes', [{'free': False}, {'enabled': False}, {'confidence': .1},
                                    {'role': 'purchase'}, {'bbox': [-1, 0, 20, 20]}])
def test_optional_bbox_does_not_bypass_control_checks(monkeypatch, tmp_path, changes):
    from test_gui_actor_routing import mock_grounder
    requests = mock_grounder(monkeypatch)
    raw = classification('P')
    raw['controls'] = [control(bbox=None, grounding_description='close button', **{
        k:v for k,v in changes.items() if k not in {'bbox', 'role'}})]
    raw['controls'][0].update(changes)
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw, gui=True)
    assert server.decide(req)['action']['type'] == 'wait'
    assert requests == []


def test_provided_bbox_still_constrains_gui_actor(monkeypatch, tmp_path):
    from test_gui_actor_routing import mock_grounder
    mock_grounder(monkeypatch)
    raw = classification('P')
    raw['controls'] = [control(bbox=[10, 10, 30, 30])]
    server, req, _ = server_fixture(monkeypatch, tmp_path, raw, gui=True)
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert trace(result)['route'] == 'ui_grounding_error'
