# Architecture

## 1. Problem Shape

에이전트는 게임 화면 이미지를 계속 관찰하고, VLM/LLM 또는 별도 policy가 다음 행동을 결정한 뒤, BlueStacks에 터치 기반 조작을 적용합니다. 로컬에서 돌 수도 있고, 원격 서버에서 추론만 맡길 수도 있으므로 모든 외부 의존성은 교체 가능한 포트로 둡니다.

## 2. Reference-Inspired Structure

### NitroGen에서 가져올 점

NitroGen은 여러 게임에서 동작하는 vision-action foundation model 방향입니다. 이 프로젝트에 바로 대형 모델 학습을 넣지는 않지만, 다음 설계 원칙은 가져옵니다.

- action vocabulary를 통일한다.
- 프레임과 액션을 transition dataset으로 저장한다.
- 게임별로 다른 조작을 모델이 직접 알게 하기보다 normalized action으로 맞춘다.
- 나중에 imitation learning 또는 behavior cloning 데이터로 재사용 가능한 로그를 남긴다.

### Voyager에서 가져올 점

Voyager는 LLM 기반 embodied agent 구조에서 자동 curriculum, skill library, 반복 피드백을 강조합니다. 이 프로젝트에서는 다음처럼 축소 적용합니다.

- skill은 "반복 터치 패턴"이나 "메뉴 이동 루틴"으로 저장한다.
- 실패한 episode에서 reflection note를 남겨 다음 prompt/context에 넣을 수 있게 한다.
- 장기 목표와 즉시 조작을 분리한다.

### ReAct/DEPS 계열에서 가져올 점

화면을 보고 바로 action만 뽑으면 디버깅이 어렵습니다. 따라서 모델 출력은 `observation_summary`, `intent`, `action`으로 나눕니다.

```json
{
  "observation_summary": "current visible game state",
  "intent": "why this action is useful",
  "action": {
    "type": "tap",
    "x": 540,
    "y": 1620,
    "duration_ms": 80
  }
}
```

## 3. Layers

### Runtime Layer

터미널 실행 진입점입니다.

- 설정 파일 로드
- 어댑터 생성
- episode 시작/종료
- loop rate, timeout, stop condition 관리

### Perception Layer

화면을 가져오고 모델 입력으로 정규화합니다.

- ADB screencap
- local window capture
- remote stream capture
- resize, crop, color conversion
- optional OCR/object detection hook

### Agent Layer

정책 판단과 조작 결정을 담당합니다.

- perception agent: 화면 상태와 주요 시각 정보를 읽되 rule/plan/action은 만들지 않는다.
- rule learner agent: 관찰과 최근 history에서 게임 rule memory를 갱신하되 plan/action은 만들지 않는다.
- planner agent: profile과 rule memory를 바탕으로 현재 목표와 전략을 정하되 좌표는 만들지 않는다.
- policy agent: planner의 전략을 실제 touch action 하나로 변환하되 rule/plan은 갱신하지 않는다.
- action schema validation
- cooldown/debounce
- safety guardrail

기본 `pipeline` 모드에서는 같은 VLM을 역할별 프롬프트로 순차 호출합니다. Ablation을 위해
`--perception-model-id`, `--rule-learner-model-id`, `--planner-model-id`,
`--policy-model-id`로 agent별 로컬 모델 ref를 다르게 줄 수 있습니다.

### Model Client Layer

추론 백엔드를 교체 가능하게 둡니다.

- hosted VLM API
- local VLM server
- remote inference server
- replay/mock model for tests

### Control Layer

실제 게임에 액션을 적용합니다.

- BlueStacks through ADB
- direct Android device through ADB
- remote control bridge
- future desktop mouse adapter

기본 액션은 키보드가 아니라 모바일 터치 입력입니다.

### Storage/Telemetry Layer

에이전트가 왜 그렇게 행동했는지 추적할 수 있게 남깁니다.

- frames
- observations
- model raw response
- validated action
- execution result
- latency/cost
- episode metadata

## 4. Main Contracts

### Observation

```text
frame_id
timestamp
image_uri or image_bytes
screen_width
screen_height
optional ocr/text detections
optional previous_action
```

### Action

```text
tap(x, y, duration_ms)
swipe(x1, y1, x2, y2, duration_ms)
long_press(x, y, duration_ms)
wait(duration_ms)
back()
home()
noop(reason)
```

### Decision

```text
observation_id
intent
action
confidence
raw_model_response
model_name
latency_ms
```

## 5. Deployment Modes

### Local All-in-One

```text
BlueStacks + ADB + capture + model client + runner
```

가장 먼저 구현하기 좋은 모드입니다.

### Local Game, Remote Model

```text
BlueStacks + local capture/control + remote inference server
```

이미지는 로컬에서 캡처하고 모델 판단만 원격으로 보냅니다.

### Remote Worker

```text
remote emulator/device + remote capture/control + local CLI or orchestrator
```

장기적으로 여러 게임 인스턴스를 돌릴 때 사용합니다.

## 6. Safety and Robustness

- 모델 출력은 반드시 schema validation을 통과해야 한다.
- 좌표는 현재 화면 크기 기준으로 clamp한다.
- 같은 액션 반복은 cooldown 정책으로 제어한다.
- emergency stop 파일 또는 CLI signal로 즉시 중단 가능하게 한다.
- 계정/결제/개인정보 화면으로 보이는 상태에서는 기본적으로 `noop` 또는 stop을 선택할 수 있게 guard를 둔다.
