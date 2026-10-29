#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 本机运行才能连接本机 Chrome 的 loopback 调试服务。
# 不改动现有 Docker 服务，使用 .env 已配置的 MySQL/Redis。
if [[ ! -x .venv/bin/python ]]; then
  echo '请先创建 .venv 并安装 requirements.txt。' >&2
  exit 1
fi
if [[ -x /opt/homebrew/opt/node@22/bin/node ]]; then
  export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${CARETRACE_WEB_PORT:-8081}"
