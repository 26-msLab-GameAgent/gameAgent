import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from gameagent.solvers import build
from gameagent.solvers.registry import LocalSolver, validate_result, write_json
from gameagent.solvers.worker import check_source
from gameagent.server.adaptive_router import AdaptiveRoutingServer
from gameagent.server.vlm_server import DecisionServer, DecideRequest, PreviousAction, Screen

SOURCE = '''
def observe(image, context):
    value = image.getpixel((50, 50))[0]
    return {"signature": str(value), "confidence": 1.0,
            "board_bbox": [10, 10, 90, 90], "state": value}
def choose_action(state, context):
    return {"type": "tap", "x": 50, "y": 50, "duration_ms": 80}
'''


def candidate(tmp_path, source=SOURCE):
    (tmp_path / 'solver.py').write_text(source)
    path = tmp_path / 'manifest.json'
    write_json(path, {'game': 'test', 'version': 'test-v1', 'status': 'candidate',
                     'sha256': hashlib.sha256(source.encode()).hexdigest()})
    return path


def test_worker_local_perception_and_action(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (100, 100), (123, 0, 0)).save(path)
    solver = LocalSolver(candidate(tmp_path))
    result = solver.infer(path, {'width': 100, 'height': 100})
    assert result['observation']['signature'] == '123'
    assert result['action']['type'] == 'tap'


@pytest.mark.parametrize('source', [SOURCE + '\nimport socket', SOURCE + '\nopen("x")',
                                   SOURCE + '\nimport os', SOURCE + '\neval("1")'])
def test_no_network_or_io_source(source):
    with pytest.raises(ValueError):
        check_source(source)


@pytest.mark.parametrize('action', [dict(type='tap', x=999, y=1), dict(type='home'),
                                   dict(type='tap', x=True, y=20), dict(type='wait', duration_ms=-1)])
def test_reject_bad_actions(action):
    with pytest.raises(ValueError):
        validate_result({'observation': {'signature': 'a', 'confidence': 1,
                        'board_bbox': [10, 10, 90, 90]}, 'action': action}, 100, 100)


def test_worker_errors_disable_candidate(tmp_path):
    source = SOURCE.replace('value = image.getpixel((50, 50))[0]', 'value = 1 / 0')
    solver = LocalSolver(candidate(tmp_path, source))
    image = tmp_path / 'frame.png'
    Image.new('RGB', (100, 100)).save(image)
    for _ in range(3):
        assert solver.infer(image, {'width': 100, 'height': 100}) is None
    assert json.loads(solver.path.read_text())['status'] == 'failed'


def frame_result(mode='B', route='generated_solver', source='semantic'):
    return {'action': {'type': 'tap', 'x': 50, 'y': 50} if mode == 'B' else {'type': 'wait'},
            'pipeline': {'screen_routing': {'state': mode, 'route': route,
                         'classification_source': source, 'classification': {'confidence': 1}, 'errors': []}}}


def observation(solver, signature):
    solver.last = {'observation': {'signature': signature, 'confidence': 1,
                                  'board_bbox': [10, 10, 90, 90]}}


@pytest.mark.parametrize('metadata', [{'dry_run': True, 'previous_execution_ok': True},
                                     {'previous_execution_ok': False}, {},
                                     {'previous_execution_ok': True, 'skipped_action_frame_id': 1}])
def test_unexecuted_actions_never_promote(tmp_path, metadata):
    solver = LocalSolver(candidate(tmp_path))
    req = SimpleNamespace(frame_id=1, previous_action=None, metadata={})
    observation(solver, '1')
    solver.record(req, frame_result(), tmp_path/'frame.png')
    req.frame_id = 2
    req.previous_action = PreviousAction(type='tap', x=50, y=50)
    req.metadata = metadata
    observation(solver, '2')
    solver.record(req, frame_result('C', 'ui'), tmp_path/'frame.png')
    assert solver.stats['executed'] == 0
    assert solver.manifest['status'] == 'candidate'


def test_promotion_requires_execution_changes_and_semantic_completion(tmp_path):
    solver = LocalSolver(candidate(tmp_path))
    req = SimpleNamespace(frame_id=1, previous_action=None, metadata={})
    for n in range(4):
        observation(solver, str(n))
        solver.record(req, frame_result(), tmp_path/'frame.png')
        req.frame_id += 1
        req.previous_action = PreviousAction(type='tap', x=50, y=50)
        req.metadata = {'previous_execution_ok': True}
    assert solver.manifest['status'] == 'candidate'
    solver.record(req, frame_result('C', 'ui'), tmp_path/'frame.png')
    assert solver.manifest['status'] == 'active'


def test_router_generated_policy_and_vlm_ui_fallback(monkeypatch, tmp_path):
    manifest = candidate(tmp_path)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name': 'test', 'solver_first': True, 'generated_solver': str(manifest)})
    raw = {'state': 'B', 'confidence': 1, 'evidence': 'board',
           'board_bbox': [10, 10, 90, 90], 'board_confidence': 1, 'controls': []}
    calls = []
    def generate(self, req, path, prompt, agent_name, include_image):
        calls.append(agent_name)
        return json.dumps(raw)
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model': 'openai:test', 'adaptive': {'templates': str(tmp_path/'none.json'),
                                                      'orb_references': None}})
    image = tmp_path/'frame.png'
    Image.new('RGB', (100, 100), (200, 0, 0)).save(image)
    req = DecideRequest(frame_id=1, image_path=str(image), screen=Screen(width=100, height=100))
    first = server.decide(req)
    assert first['action']['type'] == 'tap'
    assert first['pipeline']['screen_routing']['route'] == 'generated_solver'
    assert calls == ['screen_classifier']
    req.frame_id = 2
    req.previous_action = PreviousAction(**first['action'])
    req.metadata = {'previous_execution_ok': True}
    second = server.decide(req)
    assert second['pipeline']['screen_routing']['classification_source'] == 'local_board'
    assert calls == ['screen_classifier']
    server._board_age = server.adaptive['board_refresh_steps']
    server._deferred = server.adaptive['reobserve_steps']
    raw.update(state='P', board_bbox=None)
    req.frame_id = 3
    third = server.decide(req)
    assert third['pipeline']['screen_routing']['state'] == 'P'
    assert third['action']['type'] == 'wait'
    assert calls == ['screen_classifier', 'screen_classifier']


@pytest.mark.parametrize('use_orb', [False, True])
def test_generation_builds_both_backends_without_candy_references(monkeypatch, tmp_path, use_orb):
    (tmp_path / 'configs').symlink_to(build.ROOT / 'configs', target_is_directory=True)
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'api_generate', lambda *args: {'source': SOURCE, 'objective': 'test'})
    refs = tmp_path / 'orb' if use_orb else None
    manifest = build.generate('test_game', 'tap red', model='test', base_url='https://unused', orb_references=refs)
    assert json.loads(manifest.read_text())['status'] == 'candidate'
    for backend in ('local', 'hosted'):
        routing = json.loads(manifest.with_name(f'routing_{backend}.json').read_text())
        assert routing['adaptive']['orb_references'] == (str(refs) if refs else None)
        assert routing['enable_board_solver'] is True


def test_source_tamper_invalidates_validation(tmp_path):
    manifest = candidate(tmp_path)
    manifest.with_name('solver.py').write_text(SOURCE + '\n# changed')
    with pytest.raises(ValueError, match='differs'):
        LocalSolver(manifest)


def test_delayed_board_change_counts_once_and_fallback_cannot_claim_win(tmp_path):
    solver = LocalSolver(candidate(tmp_path))
    req = SimpleNamespace(frame_id=1, previous_action=None, metadata={})
    observation(solver, 'a')
    solver.record(req, frame_result(), tmp_path/'frame.png')
    req.frame_id = 2
    req.previous_action = PreviousAction(type='tap', x=50, y=50)
    req.metadata = {'previous_execution_ok': True}
    solver.record(req, frame_result('L', 'reobserve'), tmp_path/'frame.png')
    assert solver.stats['executed'] == 1
    req.frame_id = 3
    req.previous_action = PreviousAction(type='wait')
    observation(solver, 'b')
    result = frame_result('B', 'board_reobserve')
    result['action'] = {'type': 'wait'}
    solver.record(req, result, tmp_path/'frame.png')
    assert solver.stats['changed'] == 1
    req.frame_id = 4
    solver.record(req, result, tmp_path/'frame.png')
    assert solver.stats['changed'] == 1
    # A VLM fallback input breaks solver-only attribution of a subsequent win.
    req.frame_id = 5
    solver.record(req, frame_result('B', 'board_single_call'), tmp_path/'frame.png')
    req.frame_id = 6
    req.previous_action = PreviousAction(type='tap', x=50, y=50)
    solver.record(req, frame_result('C', 'ui'), tmp_path/'frame.png')
    assert solver.stats['completions'] == 0


@pytest.mark.parametrize('orb_mode', ['default', 'explicit', 'disabled'])
def test_bootstrap_repairs_then_reuses_active(monkeypatch, tmp_path, orb_mode):
    from gameagent.solvers import cli
    rules = tmp_path/'rules.txt'
    rules.write_text('human rules')
    generated, trialed = [], []
    refs = tmp_path / 'orb'
    refs.mkdir()
    Image.new('RGB', (100, 100)).save(refs / 'board.png')
    expected = {'orb_references': refs if orb_mode == 'explicit' else
                False if orb_mode == 'disabled' else tmp_path/'assets/router_references/test'}
    def fake_generate(*a, **kw):
        assert kw.get('orb_references') == expected.get('orb_references')
        generated.append(kw.get('previous'))
        folder = tmp_path / f'version{len(generated)}'
        folder.mkdir()
        return candidate(folder)
    def fake_trial(manifest, *a):
        trialed.append(manifest)
        return 'active' if len(trialed) == 2 else 'candidate'
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    def prepare(*a, **kw):
        assert kw == expected
        return tmp_path/'capture.png'
    monkeypatch.setattr(cli, 'prepare_reference', prepare)
    monkeypatch.setattr(cli, 'generate', fake_generate)
    monkeypatch.setattr(cli, 'trial', fake_trial)
    cli.main(['bootstrap', '--game', 'test', '--instructions', str(rules),
              '--model', 'test', '--runtime-config', 'unused.json', '--attempts', '3']
             + (['--orb-references', 'orb'] if orb_mode == 'explicit' else
                ['--no-orb'] if orb_mode == 'disabled' else []))
    assert len(generated) == 2
    assert generated[1] == trialed[0]


@pytest.mark.parametrize('mode', ['reuse', 'failed_run', 'candidate', 'stale_gate', 'fresh', 'stop'])
def test_bootstrap_only_repairs_unvalidated_solver(monkeypatch, tmp_path, mode):
    from gameagent.solvers import cli
    folder = tmp_path/'solvers/test/20260930'
    folder.mkdir(parents=True)
    manifest = candidate(folder)
    data = json.loads(manifest.read_text())
    data['status'] = 'candidate' if mode == 'candidate' else 'active'
    data['perception_validation'] = {'status': 'passed', 'sha256':
        'stale' if mode == 'stale_gate' else data['sha256']}
    write_json(manifest, data)
    calls = []
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    def play(path, *args, **kwargs):
        calls.append('play')
        assert path == manifest
        if mode == 'stop':
            (tmp_path/'STOP').touch()
            return 'candidate'
        if mode == 'failed_run':
            data['status'] = 'candidate'
            write_json(manifest, data)
            return 'candidate'
        return 'active'
    def prepare(*args, **kwargs):
        calls.append('prepare')
        return tmp_path/'frame.png'
    def generate(*args, **kwargs):
        calls.append('generate')
        assert kwargs['previous'] == (None if mode == 'fresh' else manifest)
        raise RuntimeError('test repair reached')
    monkeypatch.setattr(cli, 'trial', play)
    monkeypatch.setattr(cli, 'prepare_reference', prepare)
    monkeypatch.setattr(cli, 'generate', generate)
    argv = ['bootstrap', '--game', 'test', '--model', 'test', '--runtime-config', 'unused']
    if mode not in {'reuse', 'stop'}:
        argv += ['--video', 'unused.mp4']
    if mode == 'fresh':
        argv += ['--fresh-solver']
    if mode == 'reuse':
        cli.main(argv)
        assert calls == ['play']
    elif mode == 'stop':
        with pytest.raises(SystemExit, match='Stopped by STOP'):
            cli.main(argv)
        assert calls == ['play']
    else:
        with pytest.raises(SystemExit, match='test repair reached'):
            cli.main(argv)
        assert calls == (['play'] if mode == 'failed_run' else []) + ['prepare', 'generate']


def test_trial_stop_prevents_another_generation(monkeypatch, tmp_path):
    from gameagent.solvers import cli
    rules = tmp_path/'rules.txt'
    rules.write_text('human rules')
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'prepare_reference', lambda *a, **kw: tmp_path/'capture.png')
    manifest = candidate(tmp_path)
    monkeypatch.setattr(cli, 'generate', lambda *a, **kw: manifest)
    def stop(*args):
        (tmp_path/'STOP').touch()
        return 'candidate'
    monkeypatch.setattr(cli, 'trial', stop)
    with pytest.raises(SystemExit, match='Stopped by STOP'):
        cli.main(['bootstrap', '--game', 'test', '--instructions', str(rules),
                  '--model', 'test', '--runtime-config', 'unused.json'])


def test_invalid_generated_code_is_repaired_before_trial(monkeypatch, tmp_path):
    (tmp_path / 'configs').symlink_to(build.ROOT / 'configs', target_is_directory=True)
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    calls = []
    def generate(parts, *args):
        calls.append(len(parts))
        return {'source': 'invalid python !' if len(calls) == 1 else SOURCE, 'objective': 'test'}
    monkeypatch.setattr(build, 'api_generate', generate)
    manifest = build.generate('test_game', 'tap red', model='test', base_url='https://unused')
    assert manifest.exists()
    assert len(calls) == 2 and calls[1] > calls[0]


@pytest.mark.parametrize('key', ['secret-키값', 'secret\n', ' secret', 'secret\t', 'secret\x7f'])
def test_invalid_credentials_never_send_request_or_retry(monkeypatch, key):
    from gameagent.solvers.credentials import CredentialError
    monkeypatch.setenv('SOLVER_API_KEY', key)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'valid-fallback')
    monkeypatch.setattr(build, 'evidence', lambda *args: [])
    calls = []
    real_generate = build.api_generate
    def counted(*args):
        calls.append(True)
        return real_generate(*args)
    monkeypatch.setattr(build, 'api_generate', counted)
    monkeypatch.setattr(build, 'urlopen', lambda *a, **kw: pytest.fail('must not send request'))
    with pytest.raises(CredentialError) as error:
        build.generate('test', 'rules', model='test', base_url='https://unused')
    assert len(calls) == 1
    assert 'SOLVER_API_KEY' in str(error.value)
    assert key not in str(error.value)


def test_credential_priority_and_missing_key():
    from gameagent.solvers.credentials import CredentialError, select_api_key
    names = ('SOLVER_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY')
    assert select_api_key(names, {names[0]: 'first', names[1]: 'second'}) == 'first'
    assert select_api_key(names, {names[0]: '', names[1]: 'second'}) == 'second'
    with pytest.raises(CredentialError, match='Set an API key'):
        select_api_key(names, {})


def test_cli_key_error_is_actionable_without_traceback(monkeypatch, tmp_path):
    from gameagent.solvers import cli
    from gameagent.solvers.credentials import CredentialError
    rules = tmp_path/'rules.txt'
    rules.write_text('rules')
    def fail(*args, **kwargs):
        raise CredentialError('SOLVER_API_KEY is invalid')
    monkeypatch.setattr(cli, 'generate', fail)
    with pytest.raises(SystemExit, match='API key configuration error: SOLVER_API_KEY'):
        cli.main(['generate', '--game', 'test', '--instructions', str(rules), '--model', 'test'])


@pytest.mark.parametrize('model', ['', 'export SOLVER_MODELCluade Opus 5', 'Claude Opus 5', '한글'])
def test_invalid_model_fails_before_video_or_network(monkeypatch, model):
    monkeypatch.setattr(build, 'evidence', lambda *a: pytest.fail('do not decode video'))
    with pytest.raises(build.GenerationRequestError, match='Invalid --model'):
        build.generate('test', '', video='unused.mp4', model=model, base_url='https://unused')


def test_http_error_reports_reason_without_secrets_or_retries(monkeypatch):
    import io
    from urllib.error import HTTPError
    key = 'synthetic-private-key'
    monkeypatch.setenv('SOLVER_API_KEY', key)
    monkeypatch.setattr(build, 'evidence', lambda *a: [])
    calls = []
    def reject(*args, **kwargs):
        calls.append(True)
        body = json.dumps({'error': {'message': 'Invalid model; ' + key}}).encode()
        raise HTTPError('https://unused', 400, 'Bad Request', {}, io.BytesIO(body))
    monkeypatch.setattr(build, 'urlopen', reject)
    with pytest.raises(build.GenerationRequestError) as error:
        build.generate('test', 'rules', model='test', base_url='https://unused')
    assert len(calls) == 1
    assert 'HTTP 400: Invalid model' in str(error.value)
    assert key not in str(error.value)


@pytest.mark.parametrize('failure', ['partial', 'disconnected'])
def test_http_401_survives_broken_error_body(monkeypatch, failure):
    import io
    from http.client import IncompleteRead
    from urllib.error import HTTPError
    key = 'synthetic-private-key'
    monkeypatch.setenv('SOLVER_API_KEY', key)
    class BrokenBody(io.BytesIO):
        def read(self, *args):
            if failure == 'partial':
                raise IncompleteRead(json.dumps({'error': {'message': 'Invalid key ' + key}}).encode())
            raise ConnectionResetError('disconnected')
    body = BrokenBody()
    calls = []
    def reject(*args, **kwargs):
        calls.append(True)
        raise HTTPError('https://unused', 401, 'Unauthorized', {}, body)
    monkeypatch.setattr(build, 'urlopen', reject)
    with pytest.raises(build.GenerationRequestError, match='HTTP 401') as error:
        build.api_generate([], 'test', 'https://unused')
    assert key not in str(error.value)
    assert len(calls) == 1 and body.closed


def test_successful_generation_request_preserves_multimodal_format(monkeypatch):
    import io
    monkeypatch.setenv('SOLVER_API_KEY', 'synthetic-key')
    content = [{'type': 'text', 'text': 'rules'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,test'}}]
    def respond(req, **kwargs):
        payload = json.loads(req.data)
        assert payload['model'] == 'anthropic/claude-opus-5'
        assert payload['messages'][1]['content'] == content
        return io.BytesIO(json.dumps({'choices': [{'message': {'content':
            json.dumps({'source': SOURCE, 'objective': 'test'})}}]}).encode())
    monkeypatch.setattr(build, 'urlopen', respond)
    assert build.api_generate(content, 'anthropic/claude-opus-5', 'https://unused')['source'] == SOURCE


@pytest.mark.parametrize('payload', [None, {}, {'choices': []}, {'choices': [None]},
    {'choices': [{'message': None}]}, {'choices': [{'message': {'content': None}}]},
    {'choices': [{'message': {'content': ''}}]},
    {'choices': [{'message': {'content': 42}}]},
    {'choices': [{'message': {'content': None, 'reasoning': 'private reasoning'}}]}])
def test_empty_or_malformed_api_response_is_retryable(payload):
    with pytest.raises(build.GenerationResponseError):
        build.response_text(payload, 'synthetic-key')


def test_truncation_is_not_mistaken_for_valid_solver():
    with pytest.raises(build.GenerationRequestError, match='finish_reason=length'):
        build.response_text({'choices': [{'finish_reason': 'length',
                             'message': {'content': '{"source":"partial"}'}}]}, 'key')


def test_refusal_and_embedded_error_stop_without_retry():
    with pytest.raises(build.GenerationRequestError, match='refused or filtered'):
        build.response_text({'choices': [{'message': {'content': None, 'refusal': 'no'}}]}, 'key')
    with pytest.raises(build.GenerationRequestError) as err:
        build.response_text({'error': {'message': 'Invalid credential secret-key'}}, 'secret-key')
    assert 'secret-key' not in str(err.value)


def test_text_blocks_supported_without_exposing_reasoning():
    payload = {'choices': [{'message': {'content': [
        {'type': 'reasoning', 'text': 'private reasoning'},
        {'type': 'text', 'text': '{"source":'}, {'type': 'text', 'text': '"code"}'}]}}]}
    assert build.response_text(payload, 'key') == '{"source":"code"}'


@pytest.mark.parametrize('succeeds', [True, False])
def test_null_api_content_retries_bounded_without_saving_bad_solver(monkeypatch, tmp_path, succeeds):
    import io
    (tmp_path/'configs').symlink_to(build.ROOT/'configs', target_is_directory=True)
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'evidence', lambda *a: [])
    monkeypatch.setenv('SOLVER_API_KEY', 'synthetic-key')
    calls = []
    def respond(*args, **kwargs):
        calls.append(True)
        content = json.dumps({'source': SOURCE, 'objective': 'test'}) if succeeds and len(calls) == 2 else None
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                                                  'message': {'content': content}}]}).encode())
    monkeypatch.setattr(build, 'urlopen', respond)
    if succeeds:
        manifest = build.generate('test', 'rules', model='test', base_url='https://unused')
        assert manifest.exists() and len(calls) == 2
    else:
        with pytest.raises(build.GenerationRequestError, match='after 3 responses'):
            build.generate('test', 'rules', model='test', base_url='https://unused')
        assert len(calls) == 3
        assert not (tmp_path/'solvers').exists()


def test_length_stops_after_one_paid_request(monkeypatch, tmp_path):
    import io
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setenv('SOLVER_API_KEY', 'synthetic-key')
    monkeypatch.setattr(build, 'evidence', lambda *a: [])
    calls = []
    def respond(req, **kwargs):
        body = json.loads(req.data)
        assert body['max_tokens'] == 32000
        assert body['reasoning'] == {'effort': 'low'}
        calls.append(True)
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'length',
                         'message': {'content': None}}]}).encode())
    monkeypatch.setattr(build, 'urlopen', respond)
    with pytest.raises(build.GenerationRequestError, match='automatic retries stopped'):
        build.generate('test', 'rules', model='test', base_url='https://unused',
                       max_output_tokens=32000, reasoning_effort='low')
    assert len(calls) == 1
    assert not (tmp_path/'solvers').exists()


def test_generation_attempts_caps_empty_responses(monkeypatch):
    calls = []
    monkeypatch.setattr(build, 'evidence', lambda *a: [])
    def empty(*args):
        calls.append(True)
        raise build.GenerationResponseError('empty')
    monkeypatch.setattr(build, 'api_generate', empty)
    with pytest.raises(build.GenerationRequestError, match='after 1 responses'):
        build.generate('test', 'rules', model='test', base_url='https://unused', generation_attempts=1)
    assert len(calls) == 1


def test_cli_generation_budget_forwarded(monkeypatch, tmp_path):
    from gameagent.solvers import cli
    rules = tmp_path/'rules.txt'
    rules.write_text('rules')
    seen = {}
    def generate(*args, **kwargs):
        seen.update(kwargs)
        return candidate(tmp_path)
    monkeypatch.setattr(cli, 'generate', generate)
    cli.main(['generate', '--game', 'test', '--instructions', str(rules), '--model', 'test',
              '--max-output-tokens', '32000', '--reasoning-effort', 'low', '--generation-attempts', '1'])
    assert seen['max_output_tokens'] == 32000
    assert seen['reasoning_effort'] == 'low'
    assert seen['generation_attempts'] == 1


def test_dual_trials_use_distinct_device_server_stop_and_memory(monkeypatch, tmp_path):
    from gameagent.solvers import cli
    from gameagent.models import load_config
    from pathlib import Path
    seen = []
    class Sock:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            seen.append(('bind', address))
    class Server:
        def __init__(self, command, **kwargs):
            seen.append(('server', command))
        def poll(self):
            return None
        def terminate(self):
            pass
        def wait(self, **kwargs):
            return 0
    class Health:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def run(command, **kwargs):
        config = load_config(command[command.index('--config') + 1])
        seen.append(('runner', config))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('socket.socket', Sock)
    monkeypatch.setattr(cli.subprocess, 'Popen', Server)
    monkeypatch.setattr(cli.subprocess, 'run', run)
    monkeypatch.setattr(cli, 'urlopen', lambda *a, **k: Health())
    monkeypatch.setattr(cli, 'server_env', lambda *a: {})
    configs = []
    for name, device, port in [('tasty_travel', '127.0.0.1:17555', 18096),
                                ('candy_crush_soda', '127.0.0.1:16555', 18097)]:
        folder = tmp_path/name
        folder.mkdir()
        manifest = candidate(folder)
        original = cli.ROOT/'configs'/f'{name}.yaml'
        # Keep test independent from actual STOP files in the user's workspace.
        import yaml
        value = yaml.safe_load(original.read_text())
        value['runtime']['emergency_stop_path'] = str(tmp_path/f'STOP_{name}')
        value['capture']['device_id'] = device
        value['control']['device_id'] = device
        config_path = tmp_path/f'{name}.json'
        write_json(config_path, value)
        assert cli.trial(manifest, 'hosted', config_path, 1, port) == 'candidate'
        config = seen[-1][1]
        assert config.capture['device_id'] == device
        assert config.control['device_id'] == device
        assert config.model['endpoint'] == f'http://127.0.0.1:{port}/v1/decide'
        assert config.runtime['emergency_stop_path'] == str(tmp_path/f'STOP_{name}')
        command = [v for k, v in seen if k == 'server'][-1]
        assert command[command.index('--rule-memory-path') + 1] == str(folder/'rules.json')
        configs.append(config)
    assert configs[0].storage['run_dir'] != configs[1].storage['run_dir']


def test_busy_port_leaves_running_trial_files_untouched(monkeypatch, tmp_path):
    import errno
    from gameagent.solvers import cli
    manifest = candidate(tmp_path)
    config = manifest.with_name('trial_runtime.json')
    config.write_text('running config')
    class BusySocket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            raise OSError(errno.EADDRINUSE, 'Address already in use')
    monkeypatch.setattr('socket.socket', BusySocket)
    with pytest.raises(SystemExit, match='Port 18097 is already in use'):
        cli.trial(manifest, 'hosted', tmp_path/'unused.yaml', 100, 18097)
    assert config.read_text() == 'running config'


@pytest.mark.parametrize('fallback,expected', [
    ({'type': 'swipe', 'x': 30, 'y': 30, 'x2': 50, 'y2': 30}, 'swipe'),
    ({'type': 'swipe', 'x': 30, 'y': 30, 'x2': 98, 'y2': 30}, 'wait'),
    ({'type': 'swipe', 'x': 30, 'y': 30, 'x2': 30, 'y2': 30}, 'wait'),
    ({'type': 'home'}, 'wait'),
    (None, 'wait'),
])
def test_rejected_solver_uses_same_frame_fallback_with_guards(monkeypatch, tmp_path, fallback, expected):
    # Incorrect local board includes a point beyond the independently classified board.
    source = SOURCE.replace('[10, 10, 90, 90]', '[0, 0, 99, 99]').replace('"x": 50', '"x": 98')
    manifest = candidate(tmp_path, source)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name': 'test', 'solver_first': True, 'generated_solver': str(manifest)})
    calls = []
    raw = {'state': 'B', 'confidence': 1, 'evidence': 'board', 'board_bbox': [10, 10, 90, 90],
           'board_confidence': 1, 'controls': [], 'fallback_action': fallback}
    def generate(*args, **kwargs):
        calls.append(True)
        return json.dumps(raw)
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model': 'openai:test', 'adaptive': {'templates': str(tmp_path/'none.json'),
                                                      'orb_references': None}})
    image = tmp_path/'frame.png'
    Image.new('RGB', (100, 100), (200, 0, 0)).save(image)
    req = DecideRequest(frame_id=1, image_path=str(image), screen=Screen(width=100, height=100))
    result = server.decide(req)
    trace = result['pipeline']['screen_routing']
    assert result['action']['type'] == expected
    assert 'action leaves classified board region' in trace['errors']
    assert trace['route'] == ('board_single_call' if expected == 'swipe' else 'board_reobserve')
    assert len(calls) == 1  # Reuse the current response, no extra VLM request.
    assert server.generated_solver.pending is None  # Fallback is not solver validation evidence.
    if expected == 'swipe':
        assert result['action']['x2'] == 50


def test_rejected_local_solver_reobserves_without_reusing_stale_fallback(monkeypatch, tmp_path):
    manifest = candidate(tmp_path)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name': 'test', 'solver_first': True, 'generated_solver': str(manifest)})
    calls = []
    raw = {'state': 'B', 'confidence': 1, 'evidence': 'board', 'board_bbox': [10, 10, 90, 90],
           'board_confidence': 1, 'controls': [],
           'fallback_action': {'type': 'swipe', 'x': 30, 'y': 30, 'x2': 50, 'y2': 30}}
    def generate(*args, **kwargs):
        calls.append(True)
        return json.dumps(raw)
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', generate)
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model': 'openai:test', 'adaptive': {'templates': str(tmp_path/'none.json'),
                                                      'orb_references': None}})
    image = tmp_path/'frame.png'
    Image.new('RGB', (100, 100), (200, 0, 0)).save(image)
    req = DecideRequest(frame_id=1, image_path=str(image), screen=Screen(width=100, height=100))
    first = server.decide(req)
    # Simulate an out-of-board local action on the next stable cached board.
    monkeypatch.setattr(server, '_board_action', lambda *a: {'type': 'tap', 'x': 98, 'y': 50})
    req.frame_id = 2
    req.previous_action = PreviousAction(**first['action'])
    result = server.decide(req)
    trace = result['pipeline']['screen_routing']
    assert trace['classification_source'] == 'local_board'
    assert result['action']['type'] == 'wait'
    assert trace['route'] == 'board_reobserve'
    assert server._board_age == server.adaptive['board_refresh_steps']
    assert len(calls) == 1


def test_failed_generated_action_and_identical_vlm_fallback_are_both_blocked(monkeypatch,tmp_path):
    manifest=candidate(tmp_path)
    profile=tmp_path/'profile.json'
    write_json(profile, {'name':'test','solver_first':True,'generated_solver':str(manifest)})
    raw={'state':'B','confidence':1,'evidence':'board','board_bbox':[10,10,90,90],
         'board_confidence':1,'controls':[], 'fallback_action':{'type':'tap','x':50,'y':50}}
    monkeypatch.setattr(DecisionServer,'_generate_with_prompt',lambda *a,**kw:json.dumps(raw))
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'none.json'),
        'orb_references':None,'board_refresh_steps':1}})
    image=tmp_path/'frame.png'; Image.new('RGB',(100,100),(200,0,0)).save(image)
    previous=None
    for frame in range(1,5):
        req=DecideRequest(frame_id=frame,image_path=str(image),screen=Screen(width=100,height=100),
            previous_action=previous,metadata={'previous_execution_ok':True})
        result=server.decide(req)
        if frame==1:
            assert result['action']['type']=='tap'
        else:
            assert result['action']['type'] not in {'tap','swipe'}
        previous=PreviousAction(**{k:v for k,v in result['action'].items()
                                   if k in {'type','x','y','x2','y2'}})
    assert server.generated_solver.failed_actions['200']


def test_generation_contract_lists_actual_source_restrictions():
    from gameagent.solvers.worker import FORBIDDEN, IO_ATTRIBUTES
    assert all(name in build.CONTRACT for name in FORBIDDEN | IO_ATTRIBUTES)
    assert 'mapping.get(key, default)' in build.CONTRACT
    with pytest.raises(ValueError, match='forbidden name: getattr'):
        check_source(SOURCE + '\nx = getattr(image, "width", 100)')


@pytest.mark.parametrize('confidence,stops',[(1.0,True),(.7,False)])
def test_ui_preparation_never_executes_board_fallback(monkeypatch,tmp_path,confidence,stops):
    profile=tmp_path/'profile.json'
    write_json(profile,{'name':'test','prepare_board_only':True})
    raw={'state':'B','confidence':confidence,'evidence':'board','board_bbox':[10,10,90,90],
         'board_confidence':confidence,'controls':[],
         'fallback_action':{'type':'swipe','x':30,'y':30,'x2':50,'y2':30}}
    monkeypatch.setattr(DecisionServer,'_generate_with_prompt',lambda *a,**kw:json.dumps(raw))
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','enable_board_solver':False,
        'adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    image=tmp_path/'frame.png';Image.new('RGB',(100,100),(200,0,0)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert result['action']['type'] in {'noop','wait'}
    trace=result['pipeline']['screen_routing']
    assert trace['stop_requested']==stops
    if stops:
        assert trace['stop_reason']=='UI preparation complete: gameplay screen reached'


@pytest.mark.parametrize('use_orb', [False, True])
def test_prepare_reference_keeps_runtime_device_and_returns_reached_frame(monkeypatch,tmp_path,use_orb):
    from gameagent.solvers import cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    from gameagent.server import vlm_server
    monkeypatch.setattr(vlm_server,'_load_profile',lambda *a:{'adaptive':{}})
    refs = tmp_path / 'orb' if use_orb else None
    def trial(manifest,backend,config,steps,port,*,ui_only):
        assert (backend,config,steps,port,ui_only)==('hosted','candy-config',40,18097,True)
        assert json.loads(manifest.with_name('profile.json').read_text())['prepare_board_only']
        for route_backend in ('hosted', 'local'):
            routing = json.loads(manifest.with_name(f'routing_{route_backend}.json').read_text())
            assert routing['adaptive']['orb_references'] == (str(refs) if refs else None)
        run=manifest.parent/'runs'/'test';run.mkdir(parents=True)
        image=run/'board.png';Image.new('RGB',(100,100)).save(image)
        (run/'steps.jsonl').write_text(json.dumps({'observation':{'image_path':str(image)},
            'decision':{'raw_response':{'pipeline':{'screen_routing':{
             'stop_reason':'UI preparation complete: gameplay screen reached','stop_requested':True,'state':'B'}}}}})+'\n')
    monkeypatch.setattr(cli,'trial',trial)
    assert cli.prepare_reference('test','candy-config','hosted',18097,orb_references=refs).is_file()


def test_ui_preparation_still_dispatches_tutorial_controls(monkeypatch,tmp_path):
    profile=tmp_path/'profile.json'
    write_json(profile,{'name':'test','prepare_board_only':True})
    raw={'state':'T','confidence':1,'evidence':'tutorial with continue button',
         'board_bbox':None,'controls':[{'id':'continue','role':'continue','label':'Continue',
         'confidence':1,'enabled':True,'free':True,'bbox':[10,10,90,90]}]}
    monkeypatch.setattr(DecisionServer,'_generate_with_prompt',lambda *a,**kw:json.dumps(raw))
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','enable_board_solver':False,
        'adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    calls=[]
    def ui_action(*args):
        calls.append(True)
        return {'type':'tap','x':50,'y':50},'ui_gui_actor'
    monkeypatch.setattr(server,'_ui_action',ui_action)
    image=tmp_path/'frame.png';Image.new('RGB',(100,100)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert calls and result['action']['type']=='tap'
    assert not result['pipeline']['screen_routing']['stop_requested']


@pytest.mark.parametrize('reason,exit_code,success',[
    ('UI preparation complete: gameplay screen reached',2,True),
    ('semantic recovery budget exhausted',2,False),
    ('UI preparation complete: gameplay screen reached',1,False),
])
def test_trial_handles_preparation_exit_code_with_recorded_evidence(monkeypatch,tmp_path,reason,exit_code,success):
    from gameagent.solvers import cli
    from gameagent.server import vlm_server
    manifest=tmp_path/'manifest.json';write_json(manifest,{'status':'ui_preparation'})
    monkeypatch.setattr(vlm_server,'_load_profile',lambda *a:{'model':{},'storage':{},'runtime':{}})
    monkeypatch.setattr(cli,'server_env',lambda *a:{})
    import socket
    class Socket:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def bind(self,*a): pass
    monkeypatch.setattr(socket,'socket',lambda *a,**kw:Socket())
    class Server:
        def poll(self): return None
        def terminate(self): pass
        def wait(self,**kw): pass
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**kw:Server())
    class Health:
        status=200
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setattr(cli,'urlopen',lambda *a,**kw:Health())
    def run(*a,**kw):
        folder=tmp_path/'runs'/'test';folder.mkdir(parents=True)
        image=folder/'board.png';image.write_bytes(b'test')
        (folder/'steps.jsonl').write_text(json.dumps({'observation':{'image_path':str(image)},
            'decision':{'raw_response':{'pipeline':{'screen_routing':{
            'state':'B','stop_requested':True,'stop_reason':reason}}}}})+'\n')
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(cli.subprocess,'run',run)
    if success:
        assert cli.trial(manifest,'hosted','unused',40,0,ui_only=True)=='ui_preparation'
    else:
        with pytest.raises(RuntimeError,match='play runner exited'):
            cli.trial(manifest,'hosted','unused',40,0,ui_only=True)


@pytest.mark.parametrize('source,confidence,complete',[
    ('orb',.99,True),('local_template',.99,True),('orb',.7,False),('local_board',.99,False),
])
def test_ui_preparation_accepts_direct_local_board_classification(monkeypatch,tmp_path,source,confidence,complete):
    profile=tmp_path/'profile.json'
    write_json(profile,{'name':'test','prepare_board_only':True})
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','enable_board_solver':False,
        'adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    raw={'state':'B','confidence':confidence,'evidence':'recognized board',
         'board_bbox':[10,10,90,90],'board_confidence':confidence,'controls':[],
         '_classification_source':source,
         'fallback_action':{'type':'swipe','x':30,'y':30,'x2':50,'y2':30}}
    monkeypatch.setattr(server.fast_screens,'classify',lambda *a,**kw:dict(raw))
    monkeypatch.setattr(server,'_semantic',lambda *a,**kw:pytest.fail('no extra VLM call for recognized B'))
    image=tmp_path/'frame.png';Image.new('RGB',(100,100)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    trace=result['pipeline']['screen_routing']
    assert trace['stop_requested']==complete
    assert result['action']['type'] in {'wait','noop'}
    if complete:
        assert trace['stop_reason']=='UI preparation complete: gameplay screen reached'


def test_worker_attribute_failure_feedback_is_not_silently_empty(tmp_path):
    source=SOURCE.replace('return {"type": "tap", "x": 50, "y": 50, "duration_ms": 80}',
        'return {"type": "tap", "x": 70 if context.failed_actions else 50, "y": 50, "duration_ms": 80}')
    solver=LocalSolver(candidate(tmp_path,source))
    image=tmp_path/'board.png';Image.new('RGB',(100,100),(123,0,0)).save(image)
    solver.failed_actions={'123':[{'type':'tap','x':50,'y':50}]}
    assert solver.infer(image,{'width':100,'height':100})['action']['x']==70


def test_orb_hit_cannot_starve_recovery_after_local_policy_failure(monkeypatch,tmp_path):
    manifest=candidate(tmp_path)
    profile=tmp_path/'profile.json';write_json(profile,{'name':'test','generated_solver':str(manifest),'solver_first':True})
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'none.json'),
        'orb_references':None,'reobserve_steps':1,'max_recovery_calls':2}})
    raw={'state':'B','confidence':1,'evidence':'board','board_bbox':[10,10,90,90],
         'board_confidence':1,'controls':[]}
    monkeypatch.setattr(server.fast_screens,'classify',lambda *a,**kw:{**raw,'_classification_source':'orb'})
    calls=[]
    def semantic(*a,**kw):
        calls.append(True)
        return {**raw,'fallback_action':None}
    monkeypatch.setattr(server,'_semantic',semantic)
    image=tmp_path/'frame.png';Image.new('RGB',(100,100),(123,0,0)).save(image)
    previous=None; stopped=False
    for frame in range(1,16):
        result=server.decide(DecideRequest(frame_id=frame,image_path=str(image),screen=Screen(width=100,height=100),
            previous_action=previous,metadata={'previous_execution_ok':True}))
        if frame>1: assert result['action']['type'] in {'wait','noop'}
        previous=PreviousAction(**{k:v for k,v in result['action'].items() if k in {'type','x','y','x2','y2'}})
        if result['pipeline']['screen_routing']['stop_requested']:
            stopped=True;break
    assert stopped and len(calls)==2


@pytest.mark.parametrize('fallback', [True, False])
def test_missing_perception_after_input_releases_watch_for_recovery(monkeypatch, tmp_path, fallback):
    source = SOURCE.replace('    value =', "    if context.get('frame_id', 0) > 1:\n        return None\n    value =")
    manifest = candidate(tmp_path, source)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name':'test', 'generated_solver':str(manifest), 'solver_first':True})
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model':'openai:test', 'adaptive':{'templates':str(tmp_path/'none.json'),
        'orb_references':None, 'reobserve_steps':1, 'max_recovery_calls':2}})
    raw = {'state':'B', 'confidence':1, 'evidence':'board', 'board_bbox':[10,10,90,90],
           'board_confidence':1, 'controls':[]}
    monkeypatch.setattr(server.fast_screens, 'classify', lambda *a, **kw:{**raw, '_classification_source':'orb'})
    calls = []
    def semantic(*a, **kw):
        calls.append(True)
        return {**raw, 'fallback_action':{'type':'tap','x':30,'y':30} if fallback else None}
    monkeypatch.setattr(server, '_semantic', semantic)
    image = tmp_path/'frame.png'
    Image.new('RGB', (100,100), (123,0,0)).save(image)
    previous = None
    for frame in range(1,16):
        result = server.decide(DecideRequest(frame_id=frame, image_path=str(image),
            screen=Screen(width=100,height=100), previous_action=previous,
            metadata={'previous_execution_ok':True}))
        trace = result['pipeline']['screen_routing']
        previous = PreviousAction(**{k:v for k,v in result['action'].items() if k in {'type','x','y','x2','y2'}})
        if trace['route'] == 'board_single_call' or trace['stop_requested']:
            break
    assert server.generated_solver.action_watch is None
    assert server.generated_solver.failed_actions == {}  # Unknown outcome is not a confirmed failure.
    if fallback:
        assert len(calls) == 1 and result['action']['x'] == 30
        assert trace['route'] == 'board_single_call'
        assert 'perception unavailable' in trace['fallback_reason']
    else:
        assert len(calls) == 2 and trace['stop_requested']


def test_failure_feedback_tracks_cells_despite_coordinate_jitter():
    from gameagent.solvers.worker import SolverContext,align_failed_actions
    failed={'signature':'same','action':{'type':'swipe','x':50,'y':30,'x2':50,'y2':60}}
    ctx=SolverContext(failed_actions=[failed])
    state={'signature':'same','grid':{'centers':[[[52,32]],[[52,63]]]}}
    align_failed_actions(ctx,state)
    assert ctx.failed_actions[0]==failed
    assert ctx.failed_actions[1]['action']=={'type':'swipe','x':52,'y':32,'x2':52,'y2':63}
    other=SolverContext(failed_actions=[failed])
    align_failed_actions(other,{**state,'signature':'changed'})
    assert len(other.failed_actions)==1


def test_local_alternative_cannot_cancel_confirmed_failure_recovery(monkeypatch,tmp_path):
    source=SOURCE.replace('"x": 50, "y": 50, "duration_ms": 80}',
                          '"x": 70 if context.failed_actions else 50, "y": 50, "duration_ms": 80}')
    manifest=candidate(tmp_path,source)
    profile=tmp_path/'profile.json';write_json(profile,{'name':'test','generated_solver':str(manifest),'solver_first':True})
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    raw={'state':'B','confidence':1,'evidence':'board','board_bbox':[10,10,90,90],
         'board_confidence':1,'controls':[],'_classification_source':'orb'}
    monkeypatch.setattr(server.fast_screens,'classify',lambda *a,**kw:dict(raw))
    calls=[]
    def semantic(*a,**kw):
        calls.append(True)
        return {**raw, 'fallback_action': {'type':'tap','x':30,'y':30}}
    monkeypatch.setattr(server,'_semantic',semantic)
    image=tmp_path/'frame.png';Image.new('RGB',(100,100),(123,0,0)).save(image)
    previous=None
    for frame in range(1,7):
        result=server.decide(DecideRequest(frame_id=frame,image_path=str(image),screen=Screen(width=100,height=100),
            previous_action=previous,metadata={'previous_execution_ok':True}))
        previous=PreviousAction(**{k:v for k,v in result['action'].items() if k in {'type','x','y','x2','y2'}})
        if result['pipeline']['screen_routing']['route']=='board_single_call':
            break
    assert result['action']['type']=='tap' and result['action']['x']==30
    trace=result['pipeline']['screen_routing']
    assert trace['action_origin']=='vlm_fallback'
    assert 'ineffective' in trace['fallback_reason']
    assert len(calls)==1
    assert server.generated_solver.stats['vlm_fallback_executed']==0
    assert server.generated_solver.pending is None
    server.decide(DecideRequest(frame_id=frame+1,image_path=str(image),screen=Screen(width=100,height=100),
        previous_action=previous,metadata={'previous_execution_ok':True}))
    assert server.generated_solver.stats['vlm_fallback_executed']==1
    assert server.generated_solver.manifest['last_trial_result']=='failed_vlm_fallback'
    assert server.generated_solver.manifest['repair_required'] is True
    events = [json.loads(line) for line in manifest.with_name('validation.jsonl').read_text().splitlines()]
    recovery = next(e for e in events if e['kind']=='local_solver_trial_failed')
    assert recovery['action_origin']=='vlm_fallback'
    assert recovery['local_action']['x']==70
    assert recovery['action']['x']==30
    assert list(manifest.with_name('samples').glob('vlm_fallback_*.png'))


def test_ambiguous_tutorial_state_can_supply_reference_without_board_input(monkeypatch,tmp_path):
    profile=tmp_path/'profile.json';write_json(profile,{'name':'test','prepare_board_only':True})
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','enable_board_solver':False,
        'adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    raw={'state':'B','confidence':.72,'board_confidence':.82,'board_bbox':[10,10,90,90],
         'evidence':'visible board with tap hint, no blocking panel','controls':[],
         'fallback_action':{'type':'tap','x':50,'y':50}}
    monkeypatch.setattr(server,'_semantic',lambda *a,**kw:raw)
    image=tmp_path/'frame.png';Image.new('RGB',(100,100)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    trace=result['pipeline']['screen_routing']
    assert result['action']['type']=='noop' and trace['stop_requested']
    assert trace['stop_reason']=='UI preparation: provisional board reference captured; validation required'
    from gameagent.solvers.cli import prepared_frame
    steps=tmp_path/'steps.jsonl'
    steps.write_text(json.dumps({'observation':{'image_path':str(image)},'decision':{'raw_response':result}})+'\n')
    assert prepared_frame([steps])==image


@pytest.mark.parametrize('outside', [False, True])
def test_semantic_recovery_when_grid_observer_returns_none(monkeypatch, tmp_path, outside):
    manifest = candidate(tmp_path, 'def observe(image, context):\n    return None\ndef choose_action(state, context):\n    return None\n')
    profile = tmp_path/'profile.json'
    write_json(profile, {'name':'test','solver_first':True,'generated_solver':str(manifest)})
    raw = {'state':'B','confidence':1,'evidence':'visible board','board_bbox':[10,10,90,90],
           'board_confidence':1,'controls':[],
           'fallback_action':{'type':'swipe','x':30,'y':30,'x2':98 if outside else 50,'y2':30}}
    monkeypatch.setattr(DecisionServer, '_generate_with_prompt', lambda *a, **kw: json.dumps(raw))
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    server.generated_solver.game_spec = {'representation':'grid'}
    image = tmp_path/'frame.png'
    Image.new('RGB',(100,100),(200,0,0)).save(image)
    result = server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert result['action']['type'] == ('wait' if outside else 'swipe')
    assert result['pipeline']['screen_routing']['route'] == ('board_reobserve' if outside else 'board_single_call')
    if not outside:
        assert server.generated_solver.fallback_pending is not None
        assert server.generated_solver.pending is None
    # Local proposals remain blocked without local perception.
    assert server.generated_solver.guard_repeated_action(raw['fallback_action']) is not None


def test_semantic_recovery_ignores_local_access_but_preserves_repeat_guard(tmp_path):
    solver = LocalSolver(candidate(tmp_path))
    solver.game_spec = {'representation':'grid'}
    action = {'type':'tap','x':50,'y':50}
    solver.last = {'observation':{'signature':'board', 'grid':{'rows':1,'cols':1,
        'centers':[[[50,50]]], 'access':[['unknown']], 'labels':[['unknown']]}}}
    assert solver.guard_repeated_action(action) is not None
    assert solver.guard_repeated_action(action, semantic_fallback=True) is None
    solver.failed_actions['board'] = [action]
    assert 'ineffective' in solver.guard_repeated_action(action, semantic_fallback=True)


def test_validation_captures_none_return_condition_and_measured_values(tmp_path):
    source = '''
import numpy as np
def observe(image, context):
    rows = 7
    cols = 15
    stats = np.array([[0, 0, 50, 50, 107780]])
    i = 0
    limit = 27663.6
    if float(stats[i, 4]) > limit:
        return None
    return None
def choose_action(state, context):
    return None
'''
    solver = LocalSolver(candidate(tmp_path, source))
    image = tmp_path/'frame.png'; Image.new('RGB', (100,100)).save(image)
    result = solver.infer(image, {'width':100,'height':100}, validation=True)
    entry = result['diagnostics']['returns'][-1]
    assert entry['function'] == 'observe'
    assert entry['condition'] == 'float(stats[i, 4]) > limit'
    assert entry['values']['cols'] == 15
    assert entry['values']['stats[i]']['values'][-1] == 107780
    assert entry['values']['limit'] == 27663.6
    plain = solver.infer(image, {'width':100,'height':100})
    assert plain == {'observation':None, 'action':None}
    solver.manifest['status'] = 'perception_failed'
    assert solver.infer(image, {'width':100,'height':100}) is None
    replay = solver.infer(image, {'width':100,'height':100}, validation=True)
    assert replay['diagnostics']['returns'][-1]['values']['cols'] == 15


def test_diagnostics_preserve_helper_conditions_and_successful_output(tmp_path):
    source = SOURCE.replace('    value =', '    check(image)\n    value =') + '''
def check(image):
    bright = 5.72
    if bright > 4.5:
        return True
    return False
'''
    solver = LocalSolver(candidate(tmp_path, source))
    image = tmp_path/'frame.png'; Image.new('RGB',(100,100),(123,0,0)).save(image)
    normal = solver.infer(image, {'width':100,'height':100})
    traced = solver.infer(image, {'width':100,'height':100}, validation=True)
    assert normal == traced
    # An implicit return is explained too, with the helper's evidence retained.
    implicit = source[:source.index('def choose_action')] + '''
def choose_action(state, context):
    pass
''' + source[source.index('def check'):]
    solver = LocalSolver(candidate(tmp_path, implicit))
    traced = solver.infer(image, {'width':100,'height':100}, validation=True)
    assert traced['diagnostics']['phase'] == 'choose_action'
    helper = next(r for r in traced['diagnostics']['returns'] if r['function'] == 'check')
    assert helper['condition'] == 'bright > 4.5'
    assert helper['values']['bright'] == 5.72


def test_repair_prompt_preserves_structured_failure_and_image(monkeypatch, tmp_path):
    previous = candidate(tmp_path)
    image = tmp_path/'failed.png'; Image.new('RGB',(100,100)).save(image)
    failure = {'id':'image_0','scale':1,'status':'failed','image_path':str(image),
               'errors':['solver missed a visible board'],
               'diagnostics':{'phase':'observe','returns':[{'line':198,'condition':'area > limit',
                    'values':{'area':107780,'limit':27663.6}}]}}
    write_json(tmp_path/'perception_report.json', {'results':[failure], 'board_examples':1})
    class Captured(Exception):
        pass
    def api(parts, *a, **kw):
        text = '\n'.join(p.get('text','') for p in parts)
        assert 'Failed perception case; runtime return conditions' in text
        assert '107780' in text and '27663.6' in text and 'area > limit' in text
        assert any(p.get('type') == 'image_url' for p in parts)
        raise Captured()
    monkeypatch.setattr(build, 'api_generate', api)
    with pytest.raises(Captured):
        build.generate('test','match candies',previous=previous,model='test',base_url='https://unused')


def test_manual_rules_are_reloaded_for_generation_and_repair(monkeypatch, tmp_path):
    rules = tmp_path/'configs'/'solver_rules'; rules.mkdir(parents=True)
    common = rules/'common.md'; common.write_text('No color-only popup rule')
    (rules/'test.md').write_text('Only board swaps')
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    previous = candidate(tmp_path)
    seen = []
    class Captured(Exception):
        pass
    def api(parts, *a, **kw):
        seen.append(parts[0]['text'])
        raise Captured()
    monkeypatch.setattr(build, 'api_generate', api)
    with pytest.raises(Captured):
        build.generate('test','extra rule',model='test',base_url='https://unused')
    common.write_text('Updated structural popup rule')
    with pytest.raises(Captured):
        build.generate('test','extra rule',model='test',base_url='https://unused',previous=previous)
    assert 'No color-only popup rule' in seen[0]
    assert 'Updated structural popup rule' in seen[1]
    assert 'No color-only popup rule' not in seen[1]
    assert all('Only board swaps' in s and 'extra rule' in s for s in seen)


def test_unchanged_move_counter_prefers_vlm_over_valid_local_candidate(monkeypatch, tmp_path):
    manifest = candidate(tmp_path)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name':'test', 'generated_solver':str(manifest), 'solver_first':True})
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model':'openai:test', 'adaptive':{'templates':str(tmp_path/'none.json'), 'orb_references':None}})
    raw = {'state':'B', 'confidence':1, 'evidence':'board', 'board_bbox':[10,10,90,90],
           'board_confidence':1, 'controls':[], 'moves_remaining':{'value':20,'confidence':1},
           'fallback_action':{'type':'swipe', 'x':30, 'y':30, 'x2':50, 'y2':30}}
    calls = []
    def semantic(*a, **kw):
        calls.append(True)
        return dict(raw)
    monkeypatch.setattr(server, '_semantic', semantic)
    server._last_moves_remaining = 20
    server._swipes_since_moves = 2
    image = tmp_path/'frame.png'
    Image.new('RGB', (100,100), (123,0,0)).save(image)
    result = server.decide(DecideRequest(frame_id=3, image_path=str(image), screen=Screen(width=100,height=100)))
    trace = result['pipeline']['screen_routing']
    assert len(calls) == 1
    assert trace['route'] == 'board_single_call'
    assert trace['action_origin'] == 'vlm_fallback'
    assert 'counter unchanged' in trace['fallback_reason']
    assert result['action']['type'] == 'swipe'
    assert server.generated_solver.pending is None


def test_orb_unknown_reaches_vlm_classification(monkeypatch, tmp_path):
    manifest = candidate(tmp_path)
    profile = tmp_path/'profile.json'
    write_json(profile, {'name':'test', 'generated_solver':str(manifest), 'solver_first':True})
    server = AdaptiveRoutingServer(model_id='openai:test', profile_path=str(profile), screen_routing={
        'classifier_model':'openai:test', 'adaptive':{'templates':str(tmp_path/'none.json'), 'orb_references':None}})
    monkeypatch.setattr(server.fast_screens.orb, 'classify', lambda *a, **kw: None)
    calls=[]
    def semantic(*a, **kw):
        calls.append(True)
        return {'state':'B','confidence':1,'evidence':'board','board_bbox':[10,10,90,90],
                'board_confidence':1,'controls':[]}
    monkeypatch.setattr(server, '_semantic', semantic)
    image=tmp_path/'frame.png'
    Image.new('RGB',(100,100),(123,0,0)).save(image)
    result=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert calls == [True]
    assert result['pipeline']['screen_routing']['classification_source']=='semantic'


@pytest.mark.parametrize('mode', ['auto','fresh','explicit'])
def test_cli_selects_latest_candidate_unless_overridden(monkeypatch,tmp_path,mode):
    from gameagent.solvers import cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    old = tmp_path/'solvers'/'test'/'20260101T000000-old'
    new = tmp_path/'solvers'/'test'/'20260102T000000-new'
    for folder in (old,new):
        folder.mkdir(parents=True)
        manifest=candidate(folder)
        data=json.loads(manifest.read_text());data.update(game='test',status='perception_failed')
        write_json(manifest,data)
    # Ignore a newer incomplete directory, and other games.
    (tmp_path/'solvers'/'test'/'20260103T000000-incomplete').mkdir()
    assert cli.latest_solver('test') == new/'manifest.json'
    seen=[]
    def generate(*a,**kw):
        seen.append(kw['previous'])
        result=tmp_path/'result.json';write_json(result,{'status':'candidate'})
        return result
    monkeypatch.setattr(cli,'generate',generate)
    options=[] if mode=='auto' else ['--fresh-solver'] if mode=='fresh' else ['--previous',str(old/'manifest.json')]
    cli.main(['generate','--game','test','--model','test','--video','unused.mp4']+options)
    assert seen == [new/'manifest.json' if mode=='auto' else None if mode=='fresh' else old/'manifest.json']


def order_result():
    return {'observation':{'signature':'board|order1-ready','confidence':.95,
        'board_bbox':[10,40,90,95], 'order_controls':[{'id':'order1','bbox':[20,10,40,25],
        'ready':True,'checkmark_visible':True,'order_background_green':True}]},
        'action':{'type':'tap','x':30,'y':15,'duration_ms':80,'target':'sell_order','target_id':'order1'}}


@pytest.mark.parametrize('mutation', ['other_game','swipe','not_ready','no_check','no_green','wrong_id','outside','large_box'])
def test_order_sale_rejects_unrelated_outside_input(mutation):
    result=order_result();game='tasty_travel'
    action=result['action'];control=result['observation']['order_controls'][0]
    if mutation=='other_game': game='candy_crush_soda'
    elif mutation=='swipe': action.update(type='swipe',x2=50,y2=60)
    elif mutation=='not_ready': control['ready']=False
    elif mutation=='no_check': control['checkmark_visible']=False
    elif mutation=='no_green': control['order_background_green']=False
    elif mutation=='wrong_id': action['target_id']='another'
    elif mutation=='outside': action['x']=50
    else: control['bbox']=[0,0,99,39]
    with pytest.raises(ValueError): validate_result(result,100,100,game=game)


def test_tasty_local_order_sale_passes_router_and_repeat_guard(monkeypatch,tmp_path):
    result=order_result()
    source='def observe(image, context):\n    return '+repr(result['observation'])+'\ndef choose_action(state, context):\n    return '+repr(result['action'])+'\n'
    manifest=candidate(tmp_path,source)
    data=json.loads(manifest.read_text());data['game']='tasty_travel';write_json(manifest,data)
    profile=tmp_path/'profile.json'
    write_json(profile,{'name':'test','solver_first':True,'generated_solver':str(manifest)})
    raw={'state':'B','confidence':1,'evidence':'board','board_bbox':[10,40,90,95],
         'board_confidence':1,'controls':[]}
    monkeypatch.setattr(DecisionServer,'_generate_with_prompt',lambda *a,**kw:json.dumps(raw))
    server=AdaptiveRoutingServer(model_id='openai:test',profile_path=str(profile),screen_routing={
        'classifier_model':'openai:test','adaptive':{'templates':str(tmp_path/'none.json'),'orb_references':None}})
    image=tmp_path/'frame.png';Image.new('RGB',(100,100),(200,0,0)).save(image)
    output=server.decide(DecideRequest(frame_id=1,image_path=str(image),screen=Screen(width=100,height=100)))
    assert output['action']['type']=='tap' and output['action']['y']==15
    assert output['pipeline']['screen_routing']['route']=='generated_solver'
    solver=server.generated_solver
    solver.game_spec={'representation':'grid'}
    solver.action_watch=None
    assert solver.guard_repeated_action(result['action']) is None
    solver.failed_actions[result['observation']['signature']]=[result['action']]
    assert solver.guard_repeated_action(result['action']) is not None
    assert server._guard_board_action({**result['action'],'x':50},raw,['tap']) is not None
