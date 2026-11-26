import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import v2s


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    # Copy under a path containing spaces to exercise portability, not just the original host.
    root = tmp_path / 'portable V2S'
    root.mkdir()
    for name in ('configs', 'pretrained', 'assets', 'src'):
        shutil.copytree(v2s.ROOT / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(v2s.ROOT / 'v2s.py', root / 'v2s.py')
    monkeypatch.setattr(v2s, 'ROOT', root)
    return root


def config():
    return dict(device='emulator-5554', adb_path='adb', adb_server_socket=None,
                play_model='test/vision', gui_actor_url='http://localhost:8010')


def fingerprints(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('game,variant', [('candy_crush_soda','default'),
    ('candy_crush_soda','completion'), ('game_2048','default'), ('tasty_travel','default')])
def test_relocated_play_isolated_and_server_loadable(relocated, game, variant):
    from gameagent.server.adaptive_router import AdaptiveRoutingServer
    before = fingerprints(relocated / 'pretrained')
    v2s.apply_settings(config())
    folder = v2s.prepare_play(game, variant, None, config())
    route = json.loads((folder / 'routing_hosted.json').read_text())
    server = AdaptiveRoutingServer('openai:test/vision', mock=True,
        profile_path=str(folder / 'profile.json'), screen_routing=route)
    assert server.generated_solver.manifest['game'] == game
    assert server.generated_solver.path == folder / 'manifest.json'
    assert route['gui_actor']['url'] == config()['gui_actor_url']
    assert route['classifier_model'] == 'openai:test/vision'
    runtime = json.loads((folder / 'runtime.json').read_text())
    assert runtime['capture']['device_id'] == runtime['control']['device_id'] == 'emulator-5554'
    assert runtime['runtime']['emergency_stop_path'] == str(relocated / 'STOP')
    # A trial is allowed to update its copy, but never the shipped solver.
    (folder / 'manifest.json').write_text('{}')
    assert fingerprints(relocated / 'pretrained') == before


def test_generation_is_fresh_and_alias_normalized(relocated):
    args = v2s.parser().parse_args(['bootstrap', '--game', 'testy_travel', '--model', 'test/vision'])
    command = v2s.bootstrap_args(args, config())
    assert command[command.index('--game')+1] == 'tasty_travel'
    assert '--fresh-solver' in command and '--previous' not in command
    assert command[command.index('--backend')+1] == 'hosted'
    assert Path(command[command.index('--runtime-config')+1]).is_file()


def test_reject_wrong_game_and_tampered_solver(relocated):
    source = relocated / 'pretrained/candy_crush_soda/default'
    with pytest.raises(ValueError, match='게임'):
        v2s.prepare_play('game_2048','default',str(source/'manifest.json'),config())
    with (source/'solver.py').open('a') as f:
        f.write('\n# changed\n')
    with pytest.raises(ValueError, match='differs'):
        v2s.prepare_play('candy_crush_soda','default',None,config())


def test_bootstrap_cannot_mutate_bundled_solver(relocated):
    args = v2s.parser().parse_args(['bootstrap','--game','2048','--model','test/vision',
        '--previous',str(relocated/'pretrained/game_2048/default/manifest.json')])
    with pytest.raises(ValueError,match='V2S/solvers'):
        v2s.bootstrap_args(args,config())


def test_no_original_paths_or_results_in_bundles():
    for p in (v2s.ROOT/'pretrained').rglob('*'):
        if not p.is_file():continue
        assert p.suffix not in {'.png','.mp4','.jsonl','.log'}
        assert '/data/project/' not in p.read_text()
        assert '/home/ub0311' not in p.read_text()


def test_offline_doctor_in_separate_relocated_process(relocated):
    result = subprocess.run([sys.executable,str(relocated/'v2s.py'),'doctor','--offline'],
        cwd=relocated.parent, capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert '정상' in result.stdout
