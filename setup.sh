#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "==> Creating virtual environment with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

mkdir -p models/kokoro-mlx outputs references

echo "==> Downloading Kokoro v1.0 voicepacks & ONNX fallback..."
curl -fSL -C - -o models/voices-v1.0.bin \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
curl -fSL -C - -o models/kokoro-v1.0.onnx \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"

echo "==> Downloading Kokoro-82M Apple Silicon MLX Metal weights..."
curl -fsSL -o models/kokoro-mlx/config.json \
  "https://huggingface.co/mlx-community/Kokoro-82M-bf16/resolve/main/config.json"
curl -fSL -C - -o models/kokoro-mlx/kokoro-v1_0.safetensors \
  "https://huggingface.co/mlx-community/Kokoro-82M-bf16/resolve/main/kokoro-v1_0.safetensors"

echo "==> Done! Launch Voice Studio with: ./run.sh"
echo "    (Chatterbox MLX downloads automatically on first switch to Chatterbox MLX.)"
