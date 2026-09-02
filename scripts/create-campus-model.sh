#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${CAMPUS_MODEL_NAME:-caretrace-qwen3-14b-ft:latest}"
MODEL_DIR="${CAMPUS_MODEL_DIR:-$ROOT_DIR/finetuning/qwen3-14b-lora-full}"
GGUF_FILE="${CAMPUS_MODEL_FILE:-caretrace-qwen3-14b-ft-q4_k_m.gguf}"
UPSTREAM_GGUF="${UPSTREAM_GGUF:-}"

OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || true)}"
if [ -z "$OLLAMA_BIN" ] && [ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
  OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
fi
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Cannot find Ollama. Set OLLAMA_BIN or install Ollama." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$GGUF_FILE" ]; then
  if [ -n "$UPSTREAM_GGUF" ] && [ -f "$UPSTREAM_GGUF" ]; then
    ln -sf "$UPSTREAM_GGUF" "$MODEL_DIR/$GGUF_FILE"
  else
    echo "Missing GGUF model file: $MODEL_DIR/$GGUF_FILE" >&2
    echo "Put it there or set UPSTREAM_GGUF to an existing GGUF path." >&2
    exit 1
  fi
fi

"$OLLAMA_BIN" create "$MODEL_NAME" -f "$MODEL_DIR/Modelfile"
echo "Created $MODEL_NAME"
