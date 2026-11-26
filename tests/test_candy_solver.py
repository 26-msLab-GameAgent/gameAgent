"""Regression for the real screenshot that previously produced a zero-length swipe."""
from pathlib import Path
import json

import pytest
from PIL import Image

from gameagent.server.screen_router import ScreenRoutingServer, guard_board
from gameagent.server.vlm_server import (
    DecisionServer, DecideRequest, Screen, _load_profile, _choose_candy_grid,
    _plan_candy_crush_swipe, _scale_relative_coordinates, _uses_relative_coordinates,
    _parse_model_ref, _combined_match_info,
)

FRAME = Path(__file__).parent / 'fixtures/candy_split_board.png'


def test_solver_selects_adjacent_cells_that_make_a_match():
    profile = _load_profile('configs/profiles/candy_crush_soda.yaml')
    with Image.open(FRAME) as image:
        rows, cols, centers, labels = _choose_candy_grid(profile, image)
    action = _plan_candy_crush_swipe(profile, FRAME)
    cells = {point: (r, c) for r, row in enumerate(centers) for c, point in enumerate(row)}
    r, c = cells[(action['x'], action['y'])]
    r2, c2 = cells[(action['x2'], action['y2'])]
    assert abs(r-r2) + abs(c-c2) == 1
    labels[r][c], labels[r2][c2] = labels[r2][c2], labels[r][c]
    assert _combined_match_info(labels, r, c, r2, c2)['score'] > 0
    # Visually verified blue candies in the first column form a five-candy line.
    assert (r, c, r2, c2) == (2, 0, 2, 1)
    assert all(labels[y][0] == 'blue' for y in range(5))


def make_server(monkeypatch, state='B', bbox=None, solver=None):
    calls = []
    def generate(self, req, image_path, prompt, agent_name, include_image):
        calls.append(agent_name)
        if agent_name == 'screen_classifier':
            return json.dumps({'state': state, 'confidence': .99, 'evidence': 'stable board',
                'board_bbox': bbox or [460, 145, 1462, 936], 'board_confidence': .99,
                'controls': []})
        return json.dumps({'summary': 'board', 'confidence': .9, 'screen_mode': 'gameplay',
                           'desired_action': 'wait', 'action': {'type': 'wait'}})
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    cfg = _load_profile('configs/model_pipelines/all_qwen_local.yaml')
    routing = _load_profile('configs/screen_routing/default.yaml')
    if solver is not None:
        routing['enable_board_solver'] = solver
    server = ScreenRoutingServer(model_id=cfg['default_model'], agent_model_ids=cfg['agents'],
        profile_path='configs/profiles/candy_crush_soda.yaml',
        screen_routing=routing)
    req = DecideRequest(frame_id=1, screen=Screen(width=1920, height=1080), image_path=str(FRAME))
    return server, req, calls


def test_solver_on_real_frame_skips_all_qwen_calls(monkeypatch):
    server, req, calls = make_server(monkeypatch)
    result = server.decide(req)
    assert calls == ['screen_classifier']
    assert result['action']['type'] == 'swipe'
    trace = result['pipeline']['screen_routing']
    assert trace['route'] == 'board_solver'
    assert trace['hosted_call_count'] == 1
    assert result['board_signature']
    assert result['action']['x'] != result['action']['x2']


@pytest.mark.parametrize('state', ['S', 'P', 'C', 'T', 'L', 'U'])
def test_nonboard_never_runs_solver_even_with_visible_board(monkeypatch, state):
    server, req, calls = make_server(monkeypatch, state)
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    expected = ['screen_classifier', 'perception', 'planner', 'policy'] if state == 'U' else ['screen_classifier']
    assert calls == expected
    assert result['pipeline']['screen_routing']['route'] != 'board_solver'


def test_explicitly_disabled_solver_uses_vlm_even_for_readable_board(monkeypatch):
    server, req, calls = make_server(monkeypatch, solver=False)
    assert server.decide(req)['pipeline']['screen_routing']['route'] == 'board_pipeline'
    assert calls == ['screen_classifier', 'perception', 'planner', 'policy']


def test_solver_action_is_subject_to_board_region_guard(monkeypatch):
    server, req, calls = make_server(monkeypatch, bbox=[1000, 145, 1462, 936])
    result = server.decide(req)
    assert result['action']['type'] == 'wait'
    assert result['pipeline']['screen_routing']['guard_rejection']
    assert calls == ['screen_classifier']


def test_unreadable_board_rechecks_then_uses_qwen(monkeypatch):
    import gameagent.server.screen_router as router
    server, req, calls = make_server(monkeypatch)
    monkeypatch.setattr(router, '_candy_board_signature', lambda *a: None)
    for frame_id in range(1, 4):
        req.frame_id = frame_id
        assert server.decide(req)['pipeline']['screen_routing']['route'] == 'board_solver_recheck'
    assert calls == ['screen_classifier'] * 3
    req.frame_id = 4
    result = server.decide(req)
    assert result['pipeline']['screen_routing']['route'] == 'board_pipeline'
    assert calls[-3:] == ['perception', 'planner', 'policy']
    req.frame_id = 1
    assert server.decide(req)['pipeline']['screen_routing']['route'] == 'board_solver_recheck'


def test_qwen3_coordinates_are_scaled_and_zero_swipes_rejected():
    assert _uses_relative_coordinates(_parse_model_ref('qwen:8B'))
    assert not _uses_relative_coordinates(_parse_model_ref('qwen:3B'))
    assert not _uses_relative_coordinates(_parse_model_ref('openai:any'))
    a = _scale_relative_coordinates({'type': 'swipe', 'x': 250, 'y': 500, 'x2': 500, 'y2': 500},
                                     Screen(width=1920, height=1080))
    assert (a['x'], a['y'], a['x2'], a['y2']) == (480, 540, 960, 540)
    assert guard_board({**a, 'x2': a['x']}, {'board_bbox': [0, 0, 1919, 1079]}, ['swipe'])
