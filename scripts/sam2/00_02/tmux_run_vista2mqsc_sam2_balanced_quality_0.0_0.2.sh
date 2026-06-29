#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_ROOT="key_logs/vista2mqsc_sam2_0.0_0.2_${TS}"
mkdir -p "${LOG_ROOT}"

BALANCED_SESSION="${BALANCED_SESSION:-vista2mqsc_sam2_balanced_cuda1}"
QUALITY_SESSION="${QUALITY_SESSION:-vista2mqsc_sam2_quality_cuda2}"

tmux new-session -d -s "${BALANCED_SESSION}" \
  "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES=1 bash scripts/sam2/00_02/run_vista2mqsc_sam2_all_0.0_0.2.sh balanced detailed '${TS}' 2>&1 | tee '${LOG_ROOT}/tmux_balanced_cuda1.log'"

tmux new-session -d -s "${QUALITY_SESSION}" \
  "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES=2 bash scripts/sam2/00_02/run_vista2mqsc_sam2_all_0.0_0.2.sh quality detailed '${TS}' 2>&1 | tee '${LOG_ROOT}/tmux_quality_cuda2.log'"

{
  echo "timestamp=${TS}"
  echo "balanced_session=${BALANCED_SESSION}"
  echo "quality_session=${QUALITY_SESSION}"
  echo "balanced_cuda=1"
  echo "quality_cuda=2"
  echo "balanced_log=${LOG_ROOT}/tmux_balanced_cuda1.log"
  echo "quality_log=${LOG_ROOT}/tmux_quality_cuda2.log"
} > "${LOG_ROOT}/tmux_run_args.txt"

echo "Started ${BALANCED_SESSION} and ${QUALITY_SESSION}"
echo "Log root: ${LOG_ROOT}"
tmux list-sessions | grep -E "(${BALANCED_SESSION}|${QUALITY_SESSION})"
