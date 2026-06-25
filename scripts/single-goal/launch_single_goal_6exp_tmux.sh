#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TAG="${1:-full}"
TMUX_BIN=(tmux)
if [[ -n "${SINGLE_GOAL_TMUX_SOCKET:-}" ]]; then
  TMUX_BIN=(tmux -S "${SINGLE_GOAL_TMUX_SOCKET}")
fi

start_session () {
  local SESSION_NAME="$1"
  local CUDA_ID="$2"
  local RUN_CMD="$3"

  if "${TMUX_BIN[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}"
    return 0
  fi

  "${TMUX_BIN[@]}" new-session -d -s "${SESSION_NAME}" \
    "cd '${PROJECT_ROOT}' && unset LD_LIBRARY_PATH && unset __EGL_VENDOR_LIBRARY_FILENAMES && hash -r && export CUDA_VISIBLE_DEVICES='${CUDA_ID}' && ${RUN_CMD}"
  echo "started tmux session ${SESSION_NAME} on CUDA ${CUDA_ID}"
}

start_session \
  "single_goal_baseline_detailed_cuda1_${RUN_TAG}" \
  "1" \
  "bash scripts/single-goal/run_baseline_single_goal_all_20x0.05.sh detailed '${RUN_TAG}'"

start_session \
  "single_goal_baseline_concise_cuda1_${RUN_TAG}" \
  "1" \
  "bash scripts/single-goal/run_baseline_single_goal_all_20x0.05.sh concise '${RUN_TAG}'"

start_session \
  "single_goal_ours_fastsam_detailed_cuda2_${RUN_TAG}" \
  "2" \
  "bash scripts/single-goal/run_vista2mqsc_single_goal_all_20x0.05.sh detailed '${RUN_TAG}'"

start_session \
  "single_goal_ours_fastsam_concise_cuda2_${RUN_TAG}" \
  "2" \
  "bash scripts/single-goal/run_vista2mqsc_single_goal_all_20x0.05.sh concise '${RUN_TAG}'"

start_session \
  "single_goal_ours_sam_detailed_cuda3_${RUN_TAG}" \
  "3" \
  "bash scripts/single-goal/run_vista2mqsc_sam_single_goal_all_20x0.05.sh detailed '${RUN_TAG}'"

start_session \
  "single_goal_ours_sam_concise_cuda3_${RUN_TAG}" \
  "3" \
  "bash scripts/single-goal/run_vista2mqsc_sam_single_goal_all_20x0.05.sh concise '${RUN_TAG}'"

echo "Use '${TMUX_BIN[*]} ls' to check sessions and '${TMUX_BIN[*]} attach -t <session>' to watch logs."
