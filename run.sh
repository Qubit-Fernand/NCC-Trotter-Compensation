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
TS=(0.2 0.4 0.8 1.0 2.0 4.0 8.0)
EPSILONS=(0.001 0.0025 0.005 0.01 0.02 0.04 0.1)
MODES=(original log)
FIXED_N="${FIXED_N:-8}"
FIXED_T="${FIXED_T:-2.0}"
FIXED_EPSILON="${FIXED_EPSILON:-0.001}"

usage() {
  cat <<'EOF'
Usage: bash run.sh [--mode MODE[,MODE...]]

Modes:
  original
  log

Examples:
  bash run.sh
  bash run.sh --mode log
  bash run.sh --mode original,log
EOF
}

parse_modes() {
  local raw="$1"
  local mode
  local parsed=()
  IFS=',' read -r -a parsed <<< "$raw"
  MODES=()
  for mode in "${parsed[@]}"; do
    case "$mode" in
      original|log)
        MODES+=("$mode")
        ;;
      *)
        echo "Unknown mode: $mode" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
  if [ "${#MODES[@]}" -eq 0 ]; then
    echo "No valid modes provided." >&2
    usage >&2
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      if [ "$#" -lt 2 ]; then
        echo "--mode requires an argument." >&2
        usage >&2
        exit 1
      fi
      parse_modes "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

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
