# Runtime Modes

## Config-First Runtime

코드는 환경을 모르게 하고, 설정이 어떤 어댑터를 쓸지 결정합니다.

```yaml
runtime:
  mode: local_bluestacks
  tick_interval_ms: 800
  max_episode_steps: 1000

capture:
  adapter: adb_screencap
  device_id: auto
  resize:
    width: 720
    height: 1280

control:
  adapter: adb_touch
  device_id: auto
  coordinate_space: device

model:
  provider: openai
  model: gpt-4o-mini
```

## Adapter Matrix

| Need | Local BlueStacks | Android Device | Remote Server |
| --- | --- | --- | --- |
| Capture | `adb_screencap` | `adb_screencap` | `http_frame_stream` |
| Control | `adb_touch` | `adb_touch` | `http_action_bridge` |
| Model | `hosted_vlm` or `local_vlm` | same | `remote_inference` |
| Storage | local files | local files | object store later |

## CLI Shape

초기 CLI는 다음 정도면 충분합니다.

```bash
gameagent run --config configs/local_bluestacks.yaml
gameagent dry-run --config configs/local_bluestacks.yaml --frames samples/menu.png
gameagent replay --episode runs/2026-06-14/example
```

## Stop Conditions

- max steps reached
- user interrupt
- emergency stop file detected
- repeated invalid model response
- repeated same action with no visual change
- sensitive screen guard triggered

