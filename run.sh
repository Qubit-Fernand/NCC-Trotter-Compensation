#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRIALS="${TRIALS:-1000}"
REPEATS="${REPEATS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-data}"
R_MAX="${R_MAX:-1024}"
SAMPLING_SCRIPT="${SAMPLING_SCRIPT:-NCC_sampling_r.py}"

NS=(4 5 6 7 8)
TS=(0.2 0.4 0.6 0.8 1.0 1.2 2.0)
EPSILONS=(0.00125 0.0025 0.005 0.01 0.02 0.04 0.1)
MODES=(original log)
FIXED_N="${FIXED_N:-4}"
FIXED_T="${FIXED_T:-1.0}"
FIXED_EPSILON="${FIXED_EPSILON:-0.01}"

mkdir -p "$OUT_DIR"

run_case() {
  local mode="$1"
  local n="$2"
  local t="$3"
  local eps="$4"
  echo "[$(date '+%F %T')] running $SAMPLING_SCRIPT mode=$mode N=$n T=$t eps=$eps trials=$TRIALS repeats=$REPEATS r_max=$R_MAX"
  "$PYTHON_BIN" "$SAMPLING_SCRIPT" \
    --mode "$mode" \
    --N "$n" \
    --T "$t" \
    --epsilon "$eps" \
    --trials "$TRIALS" \
    --repeats "$REPEATS" \
    --r-max "$R_MAX" \
    --out-dir "$OUT_DIR"
}

for mode in "${MODES[@]}"; do
  # T sweep at fixed N and epsilon.
  for t in "${TS[@]}"; do
    run_case "$mode" "$FIXED_N" "$t" "$FIXED_EPSILON"
  done

  # Epsilon sweep at fixed N and T.
  for eps in "${EPSILONS[@]}"; do
    run_case "$mode" "$FIXED_N" "$FIXED_T" "$eps"
  done

  # Higher-N baseline sweep at fixed T and epsilon.
  for n in "${NS[@]}"; do
    run_case "$mode" "$n" "$FIXED_T" "$FIXED_EPSILON"
  done
done
