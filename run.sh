#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRIALS="${TRIALS:-1000}"
REPEATS="${REPEATS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-data}"

NS=(4) # 5 6 7)
TS=(0.2 0.4 0.6 0.8 1.0 1.2 2.0)
EPSILONS=(0.00125 0.0025 0.005 0.01 0.02 0.04 0.1)
SCRIPTS=(NCC_original_sampling_r.py NCC_log_sampling_r.py)

mkdir -p "$OUT_DIR"

for script in "${SCRIPTS[@]}"; do
  for n in "${NS[@]}"; do
    for t in "${TS[@]}"; do
      for eps in "${EPSILONS[@]}"; do
        echo "[$(date '+%F %T')] running $script N=$n T=$t eps=$eps trials=$TRIALS repeats=$REPEATS"
        "$PYTHON_BIN" "$script" \
          --N "$n" \
          --T "$t" \
          --epsilon "$eps" \
          --trials "$TRIALS" \
          --repeats "$REPEATS" \
          --out-dir "$OUT_DIR"
      done
    done
  done

  NS=(5 6 7)
  for n in "${NS[@]}"; do
    t=1.0
    eps=0.01
    echo "[$(date '+%F %T')] running $script N=$n T=$t eps=$eps trials=$TRIALS repeats=$REPEATS"
    "$PYTHON_BIN" "$script" \
      --N "$n" \
      --T "$t" \
      --epsilon "$eps" \
      --trials "$TRIALS" \
      --repeats "$REPEATS" \
      --out-dir "$OUT_DIR"
  done
done

