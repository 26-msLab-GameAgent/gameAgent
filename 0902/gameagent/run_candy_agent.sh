#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ub0311/.conda/envs/gameagent_vlm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "gameagent_vlm Python not found: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/src"
cd "$PROJECT_DIR"
STEPS=100
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  STEPS="$1"
  shift
fi
exec "$PYTHON_BIN" -m gameagent.runtime.cli run \
  --config configs/local_vlm_bluestacks.example.yaml \
  --steps "$STEPS" \
  "$@"
