# GameAgent

UI 없이 터미널에서 실행하는 게임 자동 플레이 에이전트의 아키텍처 골격입니다.

목표는 BlueStacks 같은 모바일 에뮬레이터, 로컬 PC, 원격 서버, 커맨드 실행 환경이 달라도 코드를 바꾸지 않고 설정과 어댑터만 바꿔서 같은 루프를 돌리는 것입니다.

## Core Loop

```text
capture one screenshot
  -> normalize observation
  -> ask model pipeline
     -> perception: read the visible screen state
     -> rule learner: update compact game-rule memory
     -> planner: choose the current objective and strategy
     -> policy: choose one executable touch action
  -> validate action
  -> execute through device adapter
  -> wait for the game screen to settle
  -> log transition
  -> repeat
```

## Design Principles

- 런타임 환경 차이는 `CaptureAdapter`, `ControlAdapter`, `ModelClient`로 숨긴다.
- 게임별 로직은 코어 루프에 넣지 않고 `profiles/` 또는 설정으로 분리한다.
- 판단 pipeline의 각 agent는 perception, rule learning, planning, policy 중 자기 역할만 수행한다.
- VLM/LLM 응답은 바로 실행하지 않고 action schema 검증을 거친다.
- 프레임, 모델 응답, 실행 액션, 결과를 모두 episode 단위로 기록한다.
- 키보드 입력은 기본 경로에서 제외하고, 터치/스와이프/대기 중심의 모바일 액션을 표준 액션으로 둔다.

## Directory Map

```text
configs/                  실행 환경별 설정 예시
docs/                     아키텍처, 런타임, 레퍼런스 문서
src/gameagent/
  agent/                  planner, policy loop, action validation
  clients/                OpenAI, local VLM, remote inference client
  control/                ADB/BlueStacks/remote control adapters
  models/                 shared schemas and typed contracts
  perception/             frame capture, preprocessing, OCR hooks
  runtime/                CLI runner, dependency wiring, lifecycle
  server/                 future frame/action API server
  storage/                episode logs, frame snapshots, replay data
  telemetry/              metrics, traces, debug events
tests/                    contract and loop tests
```

## Current Status

이 저장소는 터미널에서 실행 가능한 기본 구현을 포함합니다.

설치 없이 바로 smoke test:

```bash
cd /data/project/ssh010214/nexon/gameagent
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/mock.example.yaml --steps 3
```

ADB/BlueStacks 점검:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli doctor
```

BlueStacks가 ADB에 잡힌 상태에서 실제 터치 루프:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/local_bluestacks.example.yaml
```

이 예시는 모델을 `mock`으로 둬서 ADB 연결과 터치 경로를 먼저 확인합니다.

게임은 로컬, 모델 판단은 원격 서버:

```bash
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/remote_model.example.yaml
```

BlueStacks 화면을 로컬 Qwen-VL 서버가 보고 액션을 결정하는 모드:

```bash
conda env create -f environment.yml
conda activate gameagent_vlm
PYTHONPATH=src python -m gameagent.server.vlm_server --port 18081
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/local_vlm_bluestacks.example.yaml
```

Battle Cats 지식 프로필을 넣어 더 계획적으로 플레이:

```bash
PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18081 \
  --max-pixels 2073600 \
  --max-new-tokens 128 \
  --profile configs/profiles/battle_cats.yaml
```

메모리를 줄이고 싶으면 `--model-size 3B`로 바꿉니다. 직접 모델 ID를 지정하려면
`--model-id Qwen/Qwen2.5-VL-7B-Instruct`처럼 넣을 수 있고, 이 경우 `--model-size`보다 우선합니다.

Agent별 모델 ablation을 하려면 pipeline 단계마다 다른 모델 ref를 줄 수 있습니다.
지원되는 ref 형식은 `qwen:3B`, `qwen:7B`, `hf:<model-id>`, bare Hugging Face model id,
`openrouter:<provider/model>`, `openai:<model>`, `gemini:<model>`,
`anthropic:<model>`, `claude:<model>`입니다.

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --model-id qwen:7B \
  --perception-model-id qwen:3B \
  --rule-learner-model-id qwen:3B \
  --planner-model-id qwen:7B \
  --policy-model-id qwen:7B
```

Hosted 모델을 섞으려면 provider별 API key를 환경변수로 줍니다.

OpenRouter는 `OPENROUTER_API_KEY` 하나로 지원 모델을 바꿔 사용할 수 있습니다.

```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
echo
export OPENROUTER_API_KEY

PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --model-id 'openrouter:~openai/gpt-latest' \
  --rule-learner-model-id qwen:7B
```

OpenRouter 모델 ref 예시는 `openrouter:~openai/gpt-latest`,
`openrouter:~google/gemini-pro-latest`,
`openrouter:~anthropic/claude-sonnet-latest`입니다.

```bash
OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=... \
PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --perception-model-id openai:gpt-5.6 \
  --rule-learner-model-id claude:claude-opus-4-8 \
  --planner-model-id gemini:gemini-3.5-flash \
  --policy-model-id qwen:7B
```

처음에는 모델 없이 wiring만 확인할 수 있습니다.

```bash
PYTHONPATH=src python -m gameagent.server.vlm_server --port 18081 --mock
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/local_vlm_bluestacks.example.yaml --steps 3
```

실행 로그와 프레임은 기본적으로 `agent_runs/` 아래에 저장됩니다.

테스트는 프로젝트 루트에서 별도 `PYTHONPATH` 설정 없이 실행할 수 있습니다.

```bash
pytest -q
```

캡처/조작/모델 판단을 모두 원격 브리지로 보내는 모드:

```bash
PYTHONPATH=src python -m gameagent.server.mock_bridge --port 8080
PYTHONPATH=src python -m gameagent.runtime.cli run --config configs/full_remote.example.yaml
```

## Remote Model API

`remote_inference` 서버는 `POST /v1/decide`에서 아래 형태를 받습니다.

```json
{
  "frame_id": 1,
  "screen": {"width": 720, "height": 1280},
  "image_base64": "...",
  "previous_action": null
}
```

응답은 아래처럼 structured decision이면 됩니다.

```json
{
  "observation_summary": "main menu",
  "intent": "start the next action",
  "confidence": 0.7,
  "action": {"type": "tap", "x": 360, "y": 900, "duration_ms": 80}
}
```
