#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ub0311/.conda/envs/gameagent_vlm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "gameagent_vlm Python not found: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

echo "[Testy Travel] checking model server http://127.0.0.1:18094/health"
if ! HEALTH_RESPONSE="$(curl --fail --silent --show-error --max-time 5 http://127.0.0.1:18094/health)"; then
  echo "[Testy Travel] model server is not reachable." >&2
  echo "Start ./run_openrouter_testy_travel_server.sh in another terminal first." >&2
  exit 1
fi
echo "[Testy Travel] server connected: $HEALTH_RESPONSE"
STEPS=100
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  STEPS="$1"
  shift
fi

exec "$PYTHON_BIN" -m gameagent.runtime.cli run \
  --config configs/testy_travel.yaml \
  --steps "$STEPS" \
  "$@"
