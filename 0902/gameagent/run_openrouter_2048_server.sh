#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ub0311/.conda/envs/gameagent_vlm/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "gameagent_vlm Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[2048] OPENROUTER_API_KEY is not set."
  echo "[2048] Paste the OpenRouter key below and press Enter (input is hidden)."
  printf "OpenRouter API key: "
  read -r -s OPENROUTER_API_KEY
  echo
fi
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "OpenRouter API key is required." >&2
  exit 1
fi

export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
echo "[2048] starting server on http://127.0.0.1:18095"
echo "[2048] OpenRouter model: anthropic/claude-opus-5"
echo "[2048] persistent rules: $PROJECT_DIR/state/2048/rules.json"
exec "$PYTHON_BIN" -m gameagent.server.vlm_server \
  --port 18095 \
  --pipeline-config configs/model_pipelines/all_claude_opus_openrouter.yaml \
  --max-new-tokens 2048 \
  --profile configs/profiles/game_2048.yaml \
  --rule-memory-path state/2048/rules.json
