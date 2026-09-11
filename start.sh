#!/bin/bash
set -euo pipefail

# Change to the directory containing the server.py file
cd "$(dirname "$0")" || exit 1

# Add the current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Keep uvicorn as PID 1 so Docker/systemd can stop and restart it cleanly.
exec python -m uvicorn server:app --host 0.0.0.0 --port "${QWEN_PORT:-7860}"
