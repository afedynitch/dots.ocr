#!/usr/bin/env bash
#
# stop_server.sh — Stop the DotsOCR API server (and its managed vLLM backend)
#
# This script sends SIGTERM to the API server process recorded in api_server.pid.
# The API server's shutdown hook will also terminate the vLLM subprocess.

set -euo pipefail
cd "$(dirname "$0")"

PID_FILE="api_server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No $PID_FILE found. Is the server running?"
    exit 1
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping DotsOCR API server (PID $PID)..."
    kill "$PID"
    # Wait up to 30 seconds for graceful shutdown
    for i in $(seq 1 30); do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force-killing PID $PID..."
        kill -9 "$PID"
    fi
    echo "Stopped."
else
    echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
