# GameAgent Skeleton

터미널에서 실행하는 모바일 게임 자동 플레이 에이전트의 배포용 골격입니다.

핵심 목표는 로컬 PC, 원격 서버, BlueStacks/ADB 연결 방식이 달라도 코드를 직접 고치지 않고 설정 파일과 실행 인자만 바꿔 같은 루프를 돌리는 것입니다.

## Core Loop

```text
capture screenshot
  -> normalize observation
  -> ask VLM/LLM policy
  -> validate structured action
  -> execute through ADB/control adapter
  -> wait until the screen settles
  -> log transition
  -> repeat
```

## Directory Map

```text
configs/                  실행 환경별 설정 예시
configs/profiles/         게임별 규칙/목표 프로필 템플릿
docs/                     아키텍처와 런타임 설명
scripts/                  모델 smoke test 등 보조 스크립트
src/gameagent/
  agent/                  planner/policy loop 확장 지점
  clients/                remote inference client
  control/                ADB touch/swipe/back adapter
  models/                 shared schemas and contracts
  perception/             ADB screenshot capture
  runtime/                CLI runner and wiring
  server/                 local VLM decision server
  storage/                episode logs and frame snapshots
  telemetry/              debug events and metrics hooks
tests/                    contract tests
```

## Quick Start

Mock 루프로 배선만 확인:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli run \
  --config configs/mock.example.yaml \
  --steps 3
```

ADB/BlueStacks 연결 확인:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli doctor
```

로컬 VLM 서버 실행:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --model-size 7B \
  --max-pixels 2073600 \
  --max-new-tokens 128 \
  --profile configs/profiles/example_game_profile.yaml
```

에이전트 실행:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli run \
  --config configs/local_vlm_bluestacks_adbserver.example.yaml \
  --steps 100
```

메모리를 줄이고 싶으면 `--model-size 3B`를 사용합니다. 직접 Hugging Face 모델을 지정하려면 `--model-id`를 넘기면 됩니다.

## Game Profiles

게임별 목표, 규칙, 금지 행동은 `configs/profiles/*.yaml`에 따로 작성합니다. 배포본에는 특정 게임 프로필을 포함하지 않고, `example_game_profile.yaml`만 제공합니다.

## Generated Files

실행 로그와 캡처 프레임은 `agent_runs/` 아래에 생성됩니다. ADB platform-tools, 모델 캐시, 실행 로그, 스크린샷은 배포 대상에 포함하지 않습니다.
