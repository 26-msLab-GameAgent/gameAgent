#!/usr/bin/env python3
"""Portable launcher for the three-game 0928 distribution."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = 'openai/gpt-6-astra'
sys.path.insert(0, str(ROOT / 'src'))
GAMES = {'candy_crush_soda': 'candy_crush_soda', '2048': 'game_2048',
         'game_2048': 'game_2048', 'testy_travel': 'tasty_travel', 'tasty_travel': 'tasty_travel'}
PORTS = {'candy_crush_soda': 18096, 'game_2048': 18095, 'tasty_travel': 18098}
ARTIFACTS = ('solver.py', 'manifest.json', 'profile.json', 'game_spec.json')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def settings():
    path = ROOT / 'configs/local.json'
    if not path.is_file():
        raise ValueError('먼저 python v2s.py configure --device DEVICE 을 실행하세요. 기본 모델은 GPT-6 Astra입니다.')
    return json.loads(path.read_text())


def read_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text())


def apply_settings(config):
    """Keep generation and UI preparation on the same configured hosted backend."""
    import yaml
    for name in ('adaptive.yaml', 'adaptive_local_qwen.yaml'):
        path = ROOT / 'configs/screen_routing' / name
        route = read_yaml(path)
        route['gui_actor'].update(enabled=True, url=config['gui_actor_url'])
        if name == 'adaptive.yaml':
            route['classifier_model'] = 'openai:' + config['play_model']
        path.write_text(yaml.safe_dump(route, sort_keys=False))
    path = ROOT / 'configs/model_pipelines/adaptive_hosted.yaml'
    path.write_text(yaml.safe_dump({'default_model': 'openai:' + config['play_model'], 'agents': {}}, sort_keys=False))


def runtime_config(game, config, output):
    data = read_yaml(ROOT / 'configs' / (game + '.yaml'))
    for key in ('capture', 'control'):
        data[key].update(device_id=config['device'], adb_path=config['adb_path'],
                         adb_server_socket=config.get('adb_server_socket'))
    data['runtime']['emergency_stop_path'] = str(ROOT / 'STOP')
    data['storage']['run_dir'] = str(ROOT / 'outputs' / game)
    data['model']['endpoint'] = f'http://127.0.0.1:{PORTS[game]}/v1/decide'
    write_json(output, data)
    return output


def check_solver(folder, game):
    from gameagent.solvers.registry import LocalSolver, perception_allows_trial
    solver = LocalSolver(folder / 'manifest.json')
    if solver.manifest.get('game') != game:
        raise ValueError('선택한 게임과 manifest의 game이 다릅니다.')
    if not perception_allows_trial(solver.manifest):
        raise ValueError('인식 검증을 통과하지 못한 솔버입니다. bootstrap으로 생성/검증하세요.')
    if solver.manifest.get('status') not in {'candidate', 'active'}:
        raise ValueError('실행 가능한 candidate/active 솔버가 아닙니다.')
    return solver.manifest


def prepare_play(game, variant, manifest, config):
    """Never let trial rewrite the bundled or user-selected source artifact."""
    source = Path(manifest).expanduser().resolve().parent if manifest else ROOT / 'pretrained' / game / variant
    if not (source / 'manifest.json').is_file():
        raise ValueError(f'솔버를 찾을 수 없습니다: {source}')
    check_solver(source, game)
    folder = ROOT / 'outputs' / game / uuid4().hex[:12]
    folder.mkdir(parents=True)
    for name in ARTIFACTS:
        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
    profile = json.loads((folder / 'profile.json').read_text())
    profile['generated_solver'] = str(folder / 'manifest.json')
    profile.pop('gui_rules', None)
    if game == 'tasty_travel':
        profile['gui_rules'] = str(ROOT / 'configs/gui_rules/tasty_travel.yaml')
    write_json(folder / 'profile.json', profile)
    route = read_yaml(ROOT / 'configs/screen_routing/adaptive.yaml')
    refs = ROOT / 'assets/router_references' / game
    route['adaptive'].update(templates=str(folder / 'ui_templates.json'),
                             orb_references=str(refs) if refs.is_dir() else None)
    write_json(folder / 'routing_hosted.json', route)
    runtime_config(game, config, folder / 'runtime.json')
    write_json(folder / 'origin.json', {'source_manifest': str(source / 'manifest.json')})
    return folder


def bootstrap_args(args, config):
    game = GAMES[args.game]
    runtime = runtime_config(game, config, ROOT / 'state/runtime' / (game + '.json'))
    result = ['bootstrap', '--game', game, '--backend', 'hosted', '--runtime-config', str(runtime),
              '--port', str(args.port or PORTS[game]), '--model', args.model,
              '--steps', str(args.steps), '--attempts', str(args.attempts),
              '--max-output-tokens', str(args.max_output_tokens),
              '--generation-attempts', str(args.generation_attempts),
              '--max-generation-calls', str(args.max_generation_calls),
              '--max-repair-seconds', str(args.max_repair_seconds)]
    if args.previous:
        previous = Path(args.previous).expanduser().resolve()
        data = json.loads(previous.read_text())
        if data.get('game') != game:
            raise ValueError('--previous의 게임이 다릅니다.')
        # Bootstrap can mutate a reused validated manifest; only generated V2S versions qualify.
        if not previous.is_relative_to(ROOT / 'solvers'):
            raise ValueError('--previous에는 V2S/solvers 안에서 생성한 manifest만 지정하세요.')
        result += ['--previous', str(previous)]
    else:
        result += ['--fresh-solver']
    # Built-in common + per-game rules are included by the original generator.
    instructions = Path(args.instructions).expanduser().resolve() if args.instructions else ROOT / 'configs/solver_rules' / (game + '.md')
    result += ['--instructions', str(instructions)]
    if args.reasoning_effort:
        result += ['--reasoning-effort', args.reasoning_effort]
    for image in args.image:
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'입력 파일이 없습니다: {path}')
        result += ['--image', str(path)]
    if args.video:
        path = Path(args.video).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'영상이 없습니다: {path}')
        result += ['--video', str(path)]
    if args.no_orb:
        result += ['--no-orb']
    return result


def doctor(config, offline):
    for module in ('PIL', 'numpy', 'cv2', 'yaml', 'fastapi', 'uvicorn'):
        importlib.import_module(module)
    catalog = json.loads((ROOT / 'pretrained/catalog.json').read_text())
    for game, variants in catalog.items():
        for variant in variants:
            check_solver(ROOT / 'pretrained' / game / variant, game)
    print('Python 의존성 및 포함된 솔버 4개 무결성: 정상')
    if offline:
        print('오프라인 검사: ADB, GUI Actor, API 연결은 검사하지 않았습니다.')
        return
    from gameagent.solvers.credentials import select_api_key
    select_api_key(('OPENROUTER_API_KEY', 'OPENAI_API_KEY'))
    env = dict(os.environ)
    if config.get('adb_server_socket'):
        env['ADB_SERVER_SOCKET'] = config['adb_server_socket']
    result = subprocess.run([config['adb_path'], '-s', config['device'], 'get-state'],
                            env=env, check=True, capture_output=True, text=True, timeout=15)
    if result.stdout.strip() != 'device':
        raise ValueError('ADB 기기 상태가 device가 아닙니다.')
    with urlopen(config['gui_actor_url'].rstrip('/') + '/v1/health', timeout=10) as response:
        if response.status != 200:
            raise ValueError('GUI Actor가 준비되지 않았습니다.')
    print('ADB 기기 및 GUI Actor health: 정상. API 키 형식 확인 완료(유료 모델 호출은 하지 않음).')


def parser():
    p = argparse.ArgumentParser(description='V2S: 기존 솔버 실행 / 새 솔버 생성')
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('configure', help='기기·플레이 모델·GUI Actor 설정')
    c.add_argument('--device', required=True, help='adb devices에 표시되는 serial')
    c.add_argument('--adb-path', default='adb')
    c.add_argument('--adb-server-socket')
    c.add_argument('--play-model', default=DEFAULT_MODEL,
                   help=f'플레이용 OpenRouter 모델 ID (기본값: {DEFAULT_MODEL})')
    c.add_argument('--gui-actor-url', default='http://127.0.0.1:8010')
    d = sub.add_parser('doctor', help='설치·기기 연결 확인')
    d.add_argument('--offline', action='store_true')
    sub.add_parser('list', help='포함된 솔버 버전과 선정 이유')
    for name in ('play', 'bootstrap'):
        c = sub.add_parser(name, help='기존 솔버 실행' if name == 'play' else '새 솔버 생성·검증·실행')
        c.add_argument('--game', required=True, choices=sorted(GAMES))
        c.add_argument('--steps', type=int, default=50)
        c.add_argument('--port', type=int)
        if name == 'play':
            choice = c.add_mutually_exclusive_group()
            choice.add_argument('--variant', choices=['default', 'completion'], default='default')
            choice.add_argument('--manifest', help='새로 생성한 solver의 manifest.json')
        else:
            c.add_argument('--model', default=DEFAULT_MODEL,
                           help=f'생성용 OpenRouter 모델 ID (기본값: {DEFAULT_MODEL})')
            c.add_argument('--video')
            c.add_argument('--image', action='append', default=[])
            c.add_argument('--instructions')
            c.add_argument('--previous', help='V2S에서 생성한 후보를 이어서 개선')
            c.add_argument('--attempts', type=int, default=3)
            c.add_argument('--max-output-tokens', type=int, default=32000)
            c.add_argument('--generation-attempts', type=int, default=1)
            c.add_argument('--max-generation-calls', type=int, default=12)
            c.add_argument('--max-repair-seconds', type=int, default=1800)
            c.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'])
            c.add_argument('--no-orb', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'list':
        print((ROOT / 'pretrained/catalog.json').read_text())
        return
    if args.command == 'configure':
        if args.play_model.startswith('openai:') or '/' not in args.play_model:
            raise ValueError('--play-model에는 provider/model 형식의 OpenRouter 모델 ID를 넣으세요.')
        if not args.gui_actor_url.startswith(('http://', 'https://')):
            raise ValueError('--gui-actor-url은 http(s) URL이어야 합니다.')
        data = {k: getattr(args, k) for k in ('device', 'adb_path', 'adb_server_socket', 'play_model', 'gui_actor_url')}
        apply_settings(data)
        write_json(ROOT / 'configs/local.json', data)
        print('설정 저장: configs/local.json (API 키는 저장하지 않음)')
        return
    if args.command == 'doctor':
        doctor({} if args.offline else settings(), args.offline)
        return
    if args.steps < 1 or (args.port is not None and not 1 <= args.port <= 65535):
        raise ValueError('--steps는 양수, --port는 1~65535여야 합니다.')
    if (ROOT / 'STOP').exists():
        raise ValueError('STOP 파일이 있습니다. 재개하려면 삭제하세요.')
    config = settings()
    from gameagent.solvers.credentials import select_api_key
    select_api_key(('OPENROUTER_API_KEY', 'OPENAI_API_KEY'))
    apply_settings(config)
    if args.command == 'play':
        game = GAMES[args.game]
        folder = prepare_play(game, args.variant, args.manifest, config)
        print(f'실행 사본: {folder}', flush=True)
        command = ['trial', '--manifest', str(folder / 'manifest.json'), '--backend', 'hosted',
                   '--runtime-config', str(folder / 'runtime.json'), '--steps', str(args.steps),
                   '--port', str(args.port or PORTS[game])]
    else:
        command = bootstrap_args(args, config)
    from gameagent.solvers import cli
    os.chdir(ROOT)
    cli.main(command)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'V2S: {exc}') from None
