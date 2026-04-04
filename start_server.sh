#!/usr/bin/env bash
#
# start_server.sh — Start the DotsOCR API server
#
# The API server will:
#   1. Launch a vLLM backend (if not already running)
#   2. Expose an HTTP API on the configured port (default 8100)
#
# Environment variables (all optional):
#   VLLM_HOST     — vLLM host (default: localhost)
#   VLLM_PORT     — vLLM base port (default: 8000); instances use 8000, 8001, ...
#   VLLM_GPU      — GPU(s) for vLLM, comma-separated (default: 0)
#                   e.g. "0,1,2,3" launches 4 instances for pipeline parallelism
#   MODEL_PATH    — Path to model weights (default: ./weights/DotsOCR_1_5)
#   MODEL_NAME    — Served model name (default: model)
#   API_PORT      — Port for the HTTP API (default: 8300)
#
# Usage:
#   ./start_server.sh                           # single GPU, foreground
#   VLLM_GPU=0,1,2,3 ./start_server.sh         # 4 GPUs, foreground
#   ./start_server.sh --background              # background (logs to api_server.log)

set -euo pipefail
cd "$(dirname "$0")"

export VLLM_HOST="${VLLM_HOST:-localhost}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_GPU="${VLLM_GPU:-0}"
export MODEL_PATH="${MODEL_PATH:-./weights/DotsOCR_1_5}"
export MODEL_NAME="${MODEL_NAME:-model}"
export API_PORT="${API_PORT:-8300}"

# Ensure the model directory's parent is on PYTHONPATH (for custom model code)
MODEL_ABS="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
export PYTHONPATH="${MODEL_ABS%/*}:${PYTHONPATH:-}"

IFS=',' read -ra GPU_LIST <<< "$VLLM_GPU"
NUM_GPUS=${#GPU_LIST[@]}

echo "=== DotsOCR API Server ==="
echo "  API port:    $API_PORT"
echo "  vLLM base:   $VLLM_HOST:$VLLM_PORT"
echo "  Model:       $MODEL_PATH"
echo "  GPUs:        $VLLM_GPU ($NUM_GPUS instance(s))"
for i in "${!GPU_LIST[@]}"; do
    echo "    GPU ${GPU_LIST[$i]} → port $((VLLM_PORT + i))"
done
echo ""

if [[ "${1:-}" == "--background" ]]; then
    nohup python api_server.py > api_server.log 2>&1 &
    echo $! > api_server.pid
    echo "Started in background (PID $(cat api_server.pid)). Logs: api_server.log"
else
    python api_server.py
fi
