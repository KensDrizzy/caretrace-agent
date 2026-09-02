#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-8080}"
# Let pydantic-settings load AI_PROVIDER/OLLAMA_* from .env or its code defaults.
# Setting these here would override the repository's .env at process level.
UVICORN_BIN="$(command -v uvicorn || true)"
if [ -z "$UVICORN_BIN" ] && [ -x ".venv/bin/uvicorn" ]; then
  UVICORN_BIN=".venv/bin/uvicorn"
fi
if [ -z "$UVICORN_BIN" ]; then
  echo "Cannot find uvicorn. Activate the virtualenv or install project dependencies." >&2
  exit 1
fi
exec "$UVICORN_BIN" app.main:app --host "$HOST" --port "$PORT"
