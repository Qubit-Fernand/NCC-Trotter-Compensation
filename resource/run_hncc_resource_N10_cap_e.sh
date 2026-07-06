#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

PYTHON="/opt/anaconda3/envs/AntiEntropy/bin/python"
SCRIPT="scripts_cap_e_N10/hncc_resource_estimate.py"
OUT_DIR="figures_cap_e_N10"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/hncc_resource_N10_cap_e.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

: > "${LOG_FILE}"
{
  echo "[$(date)] start N=10 cap=e run"
  echo "Working directory: $(pwd)"
  echo "Python: ${PYTHON}"
} | tee -a "${LOG_FILE}"

# Current plotted line elements:
#   plain Suzuki-Trotter: S2, S4, S6
#   Richardson extrapolation: RE2-3, RE4-3
#   unitary-level NCC: NCC2
#   HNCC: HNCC2
set +e
/usr/bin/time -l "${PYTHON}" -u "${SCRIPT}" \
  --N 10 \
  --orders 2 4 6 \
  --ncc-orders 2 \
  --hncc-orders 2 \
  --extrap-bases 2 4 \
  --extrap-points 3 \
  --time 1.0 \
  --eps 1e-3 8e-4 6e-4 4e-4 3e-4 2e-4 1e-4 \
  --time-values 0.25 0.35 0.5 0.7 1.0 1.4 2.0 2.8 4.0 \
  --fixed-eps 1e-3 \
  --sampling-overhead-cap 2.718281828459045 \
  --x-min 1e-3 \
  --x-max 1.0 \
  --num-x 40 \
  --scan both \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
echo "[$(date)] finished with exit status ${status}" | tee -a "${LOG_FILE}"
exit "${status}"
