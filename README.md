# GameAgent

UI 없이 터미널에서 실행하는 게임 자동 플레이 에이전트의 아키텍처 골격입니다.

`0714/gameagent`는 사람 플레이 시연 없이 현재 화면만 보고 규칙을 갱신하며
플레이하는 버전입니다. `0723`은 사람 플레이 JSON을 학습하는 별도 버전입니다.

## 2048 자동 플레이와 영구 룰 메모리

첫 번째 터미널에서 2048 전용 OpenRouter 서버를 실행합니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_openrouter_2048_server.sh
```

두 번째 터미널에서 에이전트를 실행합니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_2048_agent.sh 100
```

2048에서 관찰해 갱신한 룰은 `state/2048/rules.json`에 매 단계 자동 저장됩니다.
서버를 종료하고 나중에 다시 실행해도 이 파일을 자동으로 읽어 이전 룰을
계속 사용합니다. 화면과 전체 실행 기록은 `agent_runs/2048/`에 별도로 저장되어
다른 게임의 룰 및 기록과 섞이지 않습니다.

## Testy Travel 자동 플레이

첫 번째 터미널에서 OpenRouter 모델 서버를 실행하고, 프롬프트에 키를
붙여넣습니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_openrouter_testy_travel_server.sh
```

이 서버는 OpenRouter의 `anthropic/claude-opus-5`를 인식, 규칙 학습, 계획,
행동 단계에 사용합니다. 정상 시작되면 `[VLM] listening on
http://127.0.0.1:18094`가 출력됩니다.

두 번째 터미널에서 자동 플레이를 시작합니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_testy_travel_agent.sh 100
```

에이전트는 시작 전에 서버의 `/health`를 확인합니다. 연결되지 않았다면 플레이를
시작하지 않고 즉시 어느 서버 스크립트를 실행해야 하는지 출력합니다.

이 프로필에는 게임별 좌표나 규칙을 하드코딩하지 않았습니다. 현재 화면의
튜토리얼과 행동 결과를 바탕으로 규칙을 갱신하며, 실제 해상도는 ADB PNG에서
자동 인식합니다. 기록은 `agent_runs/testy_travel/` 아래에 저장됩니다.
첫 프레임은 인식, 규칙 갱신, 계획, 행동의 모델 요청 4개를 실행합니다. 두 번째
프레임부터는 직전 행동의 게임 내 성공 여부를 판정하는 outcome verifier가 추가되어
모델 요청 5개를 순차 실행하므로
첫 행동까지 시간이 걸릴 수 있습니다. 두 번째 터미널은 최대 10분간 서버 응답을
기다리며, 동일 프레임을 자동으로 중복 요청하지 않습니다.

## 행동 결과 검증

파이프라인은 ADB 명령 실행 성공과 게임 진행 성공을 구분합니다. Planner가 기록한
`success_check`, 행동 전 perception, 다음 프레임 perception을 outcome verifier가
비교하여 직전 행동을 `success`, `failure`, `partial`, `inconclusive` 중 하나로
판정합니다. 판정과 재시도 정책은 `rule_action_trace.jsonl`의
`previous_outcome`과 `interaction_memory`에 기록되며 Rule Learner, Planner,
Policy의 다음 판단에 전달됩니다. 동일 상태에서 재시도하지 말아야 할 실패 행동은
서버에서도 한 번 더 차단합니다.

## OpenRouter로 Candy Crush 자동 플레이

첫 번째 터미널에서 모델 서버를 실행합니다. 키 입력 프롬프트가 나타나면
OpenRouter 키를 붙여넣고 Enter를 누릅니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_openrouter_candy_server.sh
```

서버가 실행된 상태에서 두 번째 터미널로 에이전트를 시작합니다.

```bash
cd /data/project/gameagent_yb/0902/gameagent
./run_candy_agent.sh 100
```

화면 해상도는 ADB 스크린샷의 PNG 헤더에서 자동으로 읽으므로 설정에
`1920x1080` 같은 고정값을 넣지 않습니다. 캡처는 먼저 `exec-out`을 시도하고,
실패하면 기기 파일 생성 후 pull 방식으로 자동 전환합니다. 실행 기록과 화면은
`agent_runs/`에 저장됩니다.

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
cd /data/project/sink0324/gameagent
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
  --max-new-tokens 1024 \
  --profile configs/profiles/battle_cats.yaml
```

Candy Crush Soda 지식 프로필을 넣어 스와이프 퍼즐로 플레이:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --model-size 7B \
  --max-pixels 2073600 \
  --max-new-tokens 1024 \
  --profile configs/profiles/candy_crush_soda.yaml

PYTHONPATH=src python -m gameagent.runtime.cli run \
  --config configs/local_vlm_bluestacks_adbserver.example.yaml \
  --steps 20
```

Merge Dragons 지식 프로필을 넣어 초록 활성 영역 안에서만 드래그 머지를 수행:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m gameagent.server.vlm_server \
  --port 18082 \
  --model-size 7B \
  --max-pixels 2073600 \
  --max-new-tokens 128 \
  --profile configs/profiles/merge_dragons.yaml

PYTHONPATH=src python -m gameagent.runtime.cli run \
  --config configs/local_vlm_bluestacks_adbserver.example.yaml \
  --steps 20
```

메모리를 줄이고 싶으면 `--model-size 3B`로 바꿉니다. 직접 모델 ID를 지정하려면
`--model-id Qwen/Qwen2.5-VL-7B-Instruct`처럼 넣을 수 있고, 이 경우 `--model-size`보다 우선합니다.

Agent별 모델 ablation을 하려면 pipeline 단계마다 다른 모델 ref를 줄 수 있습니다.
지원되는 ref 형식은 `qwen:3B`, `qwen:7B`, `hf:<model-id>`, bare Hugging Face model id,
`openai:<model>`, `gemini:<model>`, `anthropic:<model>`, `claude:<model>`입니다.

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

## 자동 플레이를 0730 시연 데이터로 재사용

각 자동 플레이 실행 폴더에는 기존 로그와 함께 다음 두 파일이 생성됩니다.

```text
demonstration.json
demonstration.mp4
```

JSON에는 실제 실행된 tap/swipe의 절대 시각, 영상 기준 시작/종료 밀리초,
기기 좌표와 정규화 좌표가 기록됩니다. MP4는 에이전트가 ADB로 관측한 화면들을
실제 관측 타임스탬프 간격에 맞춰 VFR 영상으로 인코딩합니다. 프로세스가 정상
종료되거나 `Ctrl+C`로 중지될 때 PyAV로 영상이 완성됩니다.

두 파일을 이름을 유지한 채 `0730` 설정의 demonstrations 디렉터리로 복사하면
사람 플레이 JSON+MP4 쌍 대신 사용할 수 있습니다. 예:

```bash
cp agent_runs/2048/<실행시각>/demonstration.json /data/project/gameagent_yb/0730/2048/agent_play_001.json
cp agent_runs/2048/<실행시각>/demonstration.mp4 /data/project/gameagent_yb/0730/2048/agent_play_001.mp4
```

작업공간에는 `0728` 디렉터리가 없으며, JSON+영상 시연을 처리하는 구현은
`0730`입니다.

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
