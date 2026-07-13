#!/usr/bin/env bash
set -euo pipefail

cd /data/users/zhoushuo/NCC-Trotter-Compensation
mkdir -p logs scaling/expected/data
tmux kill-session -t step-proxy-a100-remaining 2>/dev/null || true
tmux new-session -d -s step-proxy-a100-remaining \
  'cd /data/users/zhoushuo/NCC-Trotter-Compensation && PYTHONUNBUFFERED=1 .venv/bin/python scaling/expected/NCC_channel_step_proxy_r_min_cached.py --batch-file scaling/expected/step_proxy_trace_A100_remaining_batch.json --norm trace --no-require-evolution-cache 2>&1 | tee logs/step_proxy_trace_A100_remaining_20260712.log'
tmux ls
