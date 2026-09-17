#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ub0311/.conda/envs/gameagent_vlm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "gameagent_vlm Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  read -r -s -p "OpenRouter API key: " OPENROUTER_API_KEY
  echo
fi
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "OpenRouter API key is required." >&2
  exit 1
fi

export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m gameagent.server.vlm_server \
  --port 18093 \
  --pipeline-config configs/model_pipelines/all_claude_opus_openrouter.yaml \
  --max-new-tokens 2048 \
  --profile configs/profiles/candy_crush_soda.yaml \
  --rule-memory-path state/candy_crush_soda/rules.json
