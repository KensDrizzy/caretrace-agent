#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-8080}"
# Let pydantic-settings load AI_PROVIDER/OLLAMA_* from .env or its code defaults.
# Setting these here would override the repository's .env at process level.
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
