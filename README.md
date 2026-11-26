# V2S — 화면을 보고 게임별 solver를 생성하고 실행하기

`0928`의 코드를 처음 사용하는 사람을 위한 독립 실행 폴더입니다. 게임 화면을 캡처하고, 로컬 solver가 보드 상태를 해석해 행동을 선택한 뒤 ADB로 입력합니다. 메뉴·튜토리얼·팝업은 화면 라우터와 GUI Actor가 처리하며, 로컬 solver가 처리하지 못하는 보드에서는 VLM 대체 행동을 사용할 수 있습니다.

이 문서는 다음 두 경로를 설명합니다.

1. **이미 포함된 solver로 실행**: 새 코드를 생성하지 않고 선택된 solver를 재사용합니다.
2. **solver부터 새로 생성**: 게임 규칙과 현재 화면, 선택적으로 영상·스크린샷을 입력해 생성 → 인식 검증 → 실제 실행 → 제한된 수정을 수행합니다.

대상은 **Candy Crush Soda, 2048, Tasty Travel** 세 게임입니다. 요청한 `testy_travel` 이름도 실행 명령에서 사용할 수 있습니다. 기존 `0928` 폴더를 수정하거나 참조할 필요 없이 이 폴더 전체를 복사해서 사용합니다. Python 패키지만 따로 복사하지 마세요. 코드가 프로젝트 안의 설정과 solver를 함께 사용합니다.

## 1. 포함 범위와 기본 solver

| 명령에서 쓰는 게임 이름 | 내부 이름 | 기본 solver 원본 버전 | 선정 기준 및 한계 |
| --- | --- | --- | --- |
| `candy_crush_soda` | 동일 | `20260929T204651-6c68d645` | 로컬 보드 진행과 낮은 VLM 대체 비중을 우선한 후보. 인식 검증은 통과했으나 완전한 로컬 실행 검증은 미통과 |
| `2048` | `game_2048` | `20260930T155300-27d4d185` | `active`, 로컬 진행 검증 통과. 2048 타일 달성이나 모든 보드에서의 성공을 뜻하지는 않음 |
| `testy_travel` | `tasty_travel` | `20260930T124249-21906c4a` | 기록된 trial에서 로컬 입력 대비 VLM 대체 비중이 낮았던 후보. 인식 검증은 `provisional`이며 성공 보장 모델이 아님 |

Candy에는 완료 감지가 있었던 `20260930T181524-cd4ba527`도 `completion` 대안으로 포함했습니다. 기본 버전보다 VLM 대체 행동 의존이 높아 기본값으로 선택하지 않았습니다.

**‘가장 좋았던’의 기준:** 동일한 조건에서 모든 버전을 재평가한 순위가 아니라, 남아 있는 실행 기록에서 인식 검증 여부·로컬 보드 진행·VLM 대체 비중을 비교해 고른 재사용 후보입니다. 최신 날짜만으로 선택하지 않았습니다. Tasty는 짧은 실행끼리 비교한 잠정 선택입니다.

solver의 `solver.py`는 원본과 SHA-256이 같습니다. 실행에 필요한 manifest/profile과 인식 검증 상태만 함께 보관했습니다. 과거 실험 로그, 플레이 영상, 실행 스크린샷, 전체 결과 보고서는 포함하지 않았습니다. Candy의 ORB 기준 이미지는 화면 분류에 쓰는 **입력 자산**이므로 포함했습니다. 과거에 학습된 화면 템플릿과 규칙 메모리는 재사용하지 않고 실행별로 새로 쌓습니다.

### Tasty Travel에서 먼저 알아둘 점

`0928`의 실행기는 조작 가능 칸을 나타내는 `grid.access`를 요구하지만, 과거 Tasty solver 일부는 이 정보를 출력하지 않습니다. 포함한 과거 후보도 이 호환성 한계가 있습니다. 따라서 코드가 로드되어 실행되더라도 로컬 입력이 차단되고 VLM 대체 행동이 사용될 수 있습니다. `Missing cell interactability; board input blocked`가 보이면 이를 네트워크 오류로 해석하지 마세요. 안정적인 로컬 solver가 필요하면 아래 **경로 B**로 현재 출력 계약에 맞는 후보를 새로 생성·검증하세요. 생성 자체가 성공을 보장하지는 않습니다.

## 2. 설치 및 공통 준비

### 2.1 실행 환경

- Linux 또는 WSL의 Python **3.10 이상**. 아래 명령은 Bash 기준입니다.
- 게임이 설치된 Android 기기 또는 Android 에뮬레이터와 ADB 연결.
- **OpenRouter의 GPT-6 Astra (`openai/gpt-6-astra`)** 접근 권한 및 API 키.
- **GUI Actor Grounder** 서버. 이 배포본은 `0928`의 UI 좌표 추정 경로를 유지합니다. GUI Actor를 직접 호스팅한다면 해당 배포 이미지와 호환되는 NVIDIA GPU·드라이버·Docker/NVIDIA Container Toolkit이 필요합니다. 이미 실행 중인 서버를 사용하면 에이전트 머신에 GPU가 없어도 됩니다.

게임 앱, 계정, GUI Actor 가중치와 Docker 이미지, API 키는 이 폴더에 포함하지 않습니다. 본문은 hosted VLM 경로를 사용하므로 에이전트 환경에 로컬 Qwen이나 PyTorch를 설치할 필요는 없습니다.

```bash
cd /path/to/V2S
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python v2s.py doctor --offline
python v2s.py list
```

`/path/to/V2S`는 복사한 실제 경로로 바꾸세요. 현재 서버에서는 `/data/project/gameagent_yb/V2S`입니다. 다른 환경에 같은 이름의 `gameagent` 패키지가 있다면 별도 가상환경을 사용하세요.

### 2.2 ADB 연결

Android SDK Platform-Tools의 `adb`를 설치하고 실행 머신에서 기기가 보이는지 확인합니다.

```bash
adb devices -l
```

USB 기기는 USB 디버깅을 허용해야 합니다. TCP 연결 에뮬레이터는 해당 제품에 설정된 주소와 포트를 사용합니다. 아래 주소는 **예시**입니다.

```bash
adb connect 127.0.0.1:5555
adb -s 127.0.0.1:5555 get-state
```

Linux 서버에서 Windows의 BlueStacks를 조작한다면 `127.0.0.1`은 Linux 서버 자신입니다. 먼저 SSH 터널 등으로 Windows 에뮬레이터의 ADB 포트를 실행 머신에 연결하고, `adb devices -l`에 실제 표시된 serial을 사용하세요. 다른 ADB 서버를 사용하는 경우 `configure --adb-server-socket tcp:HOST:PORT`를 지정할 수 있습니다.

### 2.3 GUI Actor 준비

이미 서버가 있다면 다음을 확인하고 2.4로 넘어갑니다.

```bash
curl --fail http://127.0.0.1:8010/v1/health
```

직접 설치하려면 별도로 제공받은 GUI Actor 배포 이미지와 가중치가 필요합니다. `services/gui_actor/`에는 compose 설정과 호출 예제만 있습니다. 이미지 다운로드 주소나 비공개 체크포인트를 자동으로 내려받는 기능은 없습니다.

아래는 **RTX 3090/4090용 배포 파일을 갖고 있는 경우**입니다. `GUI_ACTOR_ASSETS`를 실제 아카이브 위치로 바꾸세요.

```bash
export GUI_ACTOR_ASSETS=/path/to/gui-actor-release
export GUI_ACTOR_MODELS_ROOT="$HOME/gui-actor-models"
mkdir -p "$GUI_ACTOR_MODELS_ROOT"

zstd -dc "$GUI_ACTOR_ASSETS/gui-actor-grounder-0.1.0-rtx3090-rtx4090-cu126.docker.tar.zst" | docker load
zstd -dc "$GUI_ACTOR_ASSETS/gui-actor-base-7b-qwen2.5-vl.tar.zst" | tar -xf - -C "$GUI_ACTOR_MODELS_ROOT"
zstd -dc "$GUI_ACTOR_ASSETS/gui-actor-ours-checkpoint-4623.tar.zst" | tar -xf - -C "$GUI_ACTOR_MODELS_ROOT"

cp services/gui_actor/.env.release.example services/gui_actor/.env
```

`services/gui_actor/.env`의 `GUI_ACTOR_MODELS_ROOT`를 `base/`, `ours/`가 들어 있는 실제 절대 경로로 수정합니다. `GROUNDING_IMAGE`는 GPU에 맞는 배포 태그로 지정합니다. 기본 예시는 `gui-actor-grounder:0.1.0-rtx4090-cu126`이며, `GUI_ACTOR_MODEL_PROFILE=ours`를 사용합니다.

```bash
docker compose --env-file services/gui_actor/.env \
  -f services/gui_actor/compose.release.yaml up -d

docker compose --env-file services/gui_actor/.env \
  -f services/gui_actor/compose.release.yaml logs -f grounder
```

모델 로딩이 완료되면 다른 터미널에서 `/v1/health`를 확인하세요. 다른 GPU용 이미지가 필요하면 해당 장비용 배포본을 사용해야 합니다. 가중치와 이미지 이용 조건은 제공된 라이선스를 따릅니다.

### 2.4 모델·기기 설정

생성과 플레이 모두 **GPT-6 Astra**를 기본으로 사용합니다. 기존 OpenRouter 연결을 유지하며 모델 ID는 `openai/gpt-6-astra`입니다. OpenAI의 공식 모델명은 [`gpt-6-astra`](https://developers.openai.com/api/docs/models/gpt-6-astra)입니다. 아래 예제는 이 모델을 명시적으로 지정합니다.

```bash
export PLAY_MODEL='openai/gpt-6-astra'
export GENERATION_MODEL="$PLAY_MODEL"

python v2s.py configure \
  --device 127.0.0.1:5555 \
  --adb-path adb \
  --play-model "$PLAY_MODEL" \
  --gui-actor-url http://127.0.0.1:8010

unset SOLVER_API_KEY OPENAI_API_KEY
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
python v2s.py doctor
```

`--play-model`은 화면 분류·보드 복구용, `bootstrap --model`은 새 solver 생성용입니다. 두 옵션을 생략해도 모두 `openai/gpt-6-astra`를 사용합니다. 모델 ID에는 실행기 내부 표기인 `openai:`를 추가로 붙이지 않습니다. `configure`는 `configs/local.json`에 기기·모델·서버 주소를 저장하고 API 키는 저장하지 않습니다. 기존 설정을 다른 모델로 저장했다면 `python v2s.py configure --device 실제_SERIAL`을 다시 실행하세요. 새 터미널에서는 가상환경 활성화와 API 키 환경변수 설정이 필요합니다.

`doctor`는 의존성·solver 무결성, ADB 상태와 GUI Actor health를 확인합니다. API 키의 형식만 검사하고 모델에 실제 요청을 보내지 않으므로 모델 접근 권한·잔액·응답 품질까지 검증하지는 않습니다.

## 3. 경로 A — 이미 있는 solver로 실행하기

기기에서 원하는 게임을 열고 가능하면 조작 가능한 보드 화면까지 이동합니다. 동일 기기에 두 에이전트를 동시에 실행하지 마세요.

```bash
# Candy Crush Soda 기본 후보
python v2s.py play --game candy_crush_soda --steps 50

# 2048
python v2s.py play --game 2048 --steps 50

# Tasty Travel: 잠정 검증 후보이며 위 호환성 제한 참고
python v2s.py play --game testy_travel --steps 50
```

위 명령은 **게임별로 하나씩** 실행합니다. 자동으로 해당 게임의 로컬 결정 서버를 시작하고 ADB 실행기를 연결한 뒤, 실행 종료 시 결정 서버를 종료합니다. 별도로 `serve` 명령을 실행할 필요가 없습니다. GUI Actor 서버는 별도 서비스로 계속 실행됩니다.

Candy의 완료 감지 대안은 다음처럼 선택합니다.

```bash
python v2s.py play --game candy_crush_soda --variant completion --steps 50
```

- `pretrained/`의 solver는 그대로 두고 `outputs/<내부 게임 이름>/<실행 ID>/`에 실행 사본을 만듭니다.
- `play`는 solver 생성·자동 수정을 하지 않습니다. 다만 화면 분류·GUI 처리·보드 복구에는 API 호출이 발생합니다.
- `--steps 50`은 기본적으로 입력 행동 예산입니다. 대기·재관찰을 포함한 스크린샷 수와 같지 않으며, 무진행 제한 등으로 일찍 종료될 수 있습니다.
- 기본 결정 서버 포트는 Candy `18096`, 2048 `18095`, Tasty `18098`입니다. 충돌하면 `--port 18196`처럼 바꿉니다.
- 실행 결과의 `candidate`/`active`는 로컬 solver 검증 상태입니다. API 대체 행동이 포함된 플레이 진행과 순수 로컬 검증 통과는 별개입니다.

## 4. 경로 B — solver부터 새로 만들기

### 4.1 입력 준비

기본 규칙은 `configs/solver_rules/common.md`와 게임별 `.md`에서 읽습니다. 게임이 설치된 기기에 연결되어 있어야 하며, 현재 보드 화면을 인식 검증 자료로 자동 캡처합니다. 메뉴에 있으면 UI 준비 단계를 통해 보드 진입을 시도합니다.

추가 입력으로 다음을 사용할 수 있습니다.

- `--video /path/to/tutorial.mp4`: 로컬 튜토리얼 영상. 최대 16개 프레임을 샘플링하며 오디오는 사용하지 않습니다.
- `--image /path/to/board.png`: 보드·규칙 화면. 옵션을 여러 번 넣을 수 있습니다.
- `--instructions /path/to/rules.md`: UTF-8 추가 규칙. 내장 공통·게임별 규칙과 함께 전달됩니다.

과거 플레이 영상이나 튜토리얼 영상은 배포본에 포함하지 않았습니다. 영상 없이도 내장 규칙과 현재 화면으로 시작할 수 있습니다.

### 4.2 새로 생성 → 검증 → 실행

```bash
# 기본 solver를 재사용하지 않고 새 후보부터 시작
python v2s.py bootstrap \
  --game candy_crush_soda \
  --model "$GENERATION_MODEL" \
  --steps 50 \
  --attempts 3

# 영상과 추가 보드 화면이 있을 때
python v2s.py bootstrap \
  --game 2048 \
  --model "$GENERATION_MODEL" \
  --video /path/to/2048_tutorial.mp4 \
  --image /path/to/2048_board.png \
  --steps 50 \
  --attempts 3

# Tasty Travel을 현재 출력 계약으로 새로 생성
python v2s.py bootstrap \
  --game testy_travel \
  --model "$GENERATION_MODEL" \
  --steps 50 \
  --attempts 3
```

이 명령들도 하나씩 실행합니다. 이 경로는 실제 기기를 조작합니다. UI 준비 → 코드 생성 → 로컬 검사 → 인식 검증 → 실제 trial → 필요 시 수정 순서이며, 인식 검증에 실패하면 보드 실행을 막고 예산 안에서 다음 후보를 시도합니다. Tasty의 `provisional`은 라벨 일치가 충분히 검증되지 않은 시험 실행을 허용하는 상태입니다.

기본 생성 설정은 다음과 같습니다.

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--attempts` | `3` | 후보 생성·수정 단계 수 |
| `--max-output-tokens` | `32000` | 개별 생성 응답의 출력 한도. 사용하는 모델 한도에 맞게 조정 |
| `--generation-attempts` | `1` | 후보 코드 응답 형식 오류 시 생성 호출 시도 수 |
| `--max-generation-calls` | `12` | 코드·규격·인식 참조 생성 API 호출 예산 |
| `--max-repair-seconds` | `1800` | 새 수정 단계를 시작할 수 있는 시간 예산. 진행 중인 요청은 끝날 수 있음 |

위 호출 예산에는 플레이 중 화면 판단과 GUI 규칙 평가 호출이 포함되지 않습니다. 모델이 지원하는 경우에만 `--reasoning-effort low`, `medium`, `high`를 지정하세요. 출력이 잘리면 보고된 오류를 확인하고 출력 한도·모델 설정을 조정합니다.

새 후보는 `solvers/<내부 게임 이름>/<버전>/`에 저장됩니다. 터미널에 출력되는 `manifest.json` 경로와 검증 상태를 확인하세요. 후보 예산이 소진되면 실패 상태로 종료될 수 있으며, 파일이 만들어졌다고 검증을 통과한 것은 아닙니다.

### 4.3 새로 만든 solver를 나중에 재사용

생성 과정에서 나온 실제 버전 경로로 바꿉니다.

```bash
python v2s.py play \
  --game 2048 \
  --manifest solvers/game_2048/실제_버전/manifest.json \
  --steps 50
```

게임이 일치하고 인식 검증이 허용된 `candidate`/`active`만 실행할 수 있습니다. 재실행 시에도 원본 후보를 별도 실행 폴더에 복사합니다.

기존 생성 후보를 이어서 **수정·검증**하려면 다음처럼 실행합니다. `--previous`가 없으면 항상 새 후보부터 시작합니다.

```bash
python v2s.py bootstrap \
  --game testy_travel \
  --model "$GENERATION_MODEL" \
  --previous solvers/tasty_travel/실제_버전/manifest.json \
  --steps 50 --attempts 3
```

`--previous`는 이 V2S의 `solvers/` 안에서 생성한 후보만 받습니다. 원본 bootstrap의 재사용 조건을 만족하는 검증된 후보는 먼저 실행하고, 필요하면 수정합니다. `pretrained/` 배포 원본을 수정 대상으로 사용하지 않습니다.

## 5. 중지·문제 해결

**중지:** 실행 터미널에서 `Ctrl+C`를 누르거나 다른 터미널에서 V2S 루트에 `touch STOP`을 실행합니다. STOP은 실행 루프/단계 경계에서 확인하므로 진행 중인 API 요청을 즉시 취소하는 기능은 아닙니다. 재개할 때 `rm STOP`으로 파일을 지웁니다.

| 증상 | 확인 방법 |
| --- | --- |
| 기기를 찾지 못함 / unauthorized | `adb devices -l`, USB 디버깅 승인, serial·ADB 경로·터널 확인 후 `configure` 재실행 |
| GUI Actor 연결 실패 | `/v1/health`, 컨테이너 로그, 포트와 `--gui-actor-url` 확인 |
| API 인증·모델 오류 | 키를 다시 입력하고 현재 계정에서 해당 vision 모델을 사용할 수 있는지 확인 |
| `Perception validation ...` 오류 | 생성 후보의 인식 보고서 확인. 보드가 잘 보이는 현재 화면·추가 이미지로 새 후보 생성 |
| Tasty의 `Missing cell interactability` | 과거 solver의 출력 계약 제한. 차단을 해제하거나 검증 상태를 수동 변경하지 말고 경로 B에서 새 후보 생성 |
| Candy의 화면 분류가 맞지 않음 | 다른 해상도·언어·버전일 수 있음. 새 후보 생성 시 `--no-orb`로 기존 기준 이미지 매칭을 끄고 비교 |
| 계속 같은 행동 / 무진행 종료 | 실행 폴더의 `trial_server.log`, `validation.jsonl`, `runs/` 확인. solver 인식 범위·입력 기기·애니메이션 대기 확인 |
| 포트 사용 중 | 기존 실행을 종료하거나 `--port` 변경 |

과거 solver는 당시 게임 화면에 맞춰 만들어졌습니다. 현재 앱 버전·해상도·레벨에서도 같은 성능을 낸다는 보장은 없습니다. 기본 환경이 달라졌다면 경로 B로 새 후보를 생성하는 편이 적합합니다.

## 6. 폴더 구조 및 검증

```text
V2S/
├── README.md                  # 이 문서
├── v2s.py                     # configure / doctor / list / play / bootstrap
├── pyproject.toml
├── src/gameagent/             # 0928 실행기·라우터·생성·검증 코드
├── configs/                   # 세 게임 설정, 규칙, 모델/라우팅 설정
├── pretrained/                # 수정하지 않는 배포 solver 4개 및 출처
├── assets/router_references/  # Candy ORB 화면 분류 입력 자산
├── services/gui_actor/        # GUI Actor 배포 설정·라이선스
└── tests/                     # 원본 회귀 테스트와 배포/경로 검증 테스트
```

처음 실행하면 `outputs/`(기존 solver 실행), `solvers/`(새 생성 후보), `state/`(준비 화면·수정 예산 상태)가 추가됩니다. 이 파일들은 사용자가 실행하면서 새로 생기는 자료이며 기존 실험 결과는 배포하지 않습니다.

```bash
python -m pytest -q
```

테스트는 실제 기기 입력이나 유료 API 호출 없이 수행합니다. 포함된 solver 무결성, 이동한 폴더에서의 로딩, 배포 원본 보존, 새 생성 경로와 기존 회귀 테스트를 검사합니다. 배포 정리 시 Python 3.10 환경에서 414개 테스트가 통과했습니다. 실제 ADB 게임 플레이와 외부 모델 호출은 사용자의 기기·GUI Actor·API 설정 후 확인해야 합니다.
