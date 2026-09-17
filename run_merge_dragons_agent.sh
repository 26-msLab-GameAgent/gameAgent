#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ub0311/.conda/envs/gameagent_vlm/bin/python"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

echo "[Merge Dragons] checking http://127.0.0.1:5585/health"
if ! HEALTH_RESPONSE="$(curl --fail --silent --show-error --max-time 5 http://127.0.0.1:5585/health)"; then
  echo "Start ./run_openrouter_merge_dragons_server.sh first." >&2
  exit 1
fi
echo "[Merge Dragons] server connected: $HEALTH_RESPONSE"
STEPS=100
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  STEPS="$1"
  shift
fi
exec "$PYTHON_BIN" -m gameagent.runtime.cli run \
  --config configs/merge_dragons_openrouter.yaml \
  --steps "$STEPS" \
  "$@"
