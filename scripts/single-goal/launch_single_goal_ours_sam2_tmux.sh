#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Usage:
#   bash scripts/single-goal/launch_single_goal_ours_sam2_tmux.sh [detailed|concise] [stable_tag] [balanced|quality|sanity]
DESC_MODE="${1:-detailed}"
RUN_TAG="${2:-full}"
SAM2_LEVEL_PRESET="${3:-balanced}"
CUDA_ID="${CUDA_ID:-1}"
VLM_MODEL_NAME="${MQSC_R1_VLM_MODEL:-${VLM_MODEL:-qwen-vl-plus}}"

if [ "${DESC_MODE}" != "detailed" ] && [ "${DESC_MODE}" != "concise" ]; then
  echo "Invalid first arg: ${DESC_MODE}. Must be 'detailed' or 'concise'."
  exit 1
fi
if [ "${SAM2_LEVEL_PRESET}" != "balanced" ] && [ "${SAM2_LEVEL_PRESET}" != "quality" ] && [ "${SAM2_LEVEL_PRESET}" != "sanity" ]; then
  echo "Invalid SAM2_LEVEL_PRESET: ${SAM2_LEVEL_PRESET}. Must be balanced, quality, or sanity."
  exit 1
fi

TMUX_BIN=(tmux)
if [[ -n "${SINGLE_GOAL_TMUX_SOCKET:-}" ]]; then
  TMUX_BIN=(tmux -S "${SINGLE_GOAL_TMUX_SOCKET}")
fi

SESSION_NAME="${SESSION_NAME:-single_goal_ours_sam2_${SAM2_LEVEL_PRESET}_${DESC_MODE}_cuda${CUDA_ID}_${RUN_TAG}}"
RUN_CMD="bash scripts/single-goal/run_vista2mqsc_sam2_single_goal_all_20x0.05.sh '${DESC_MODE}' '${RUN_TAG}' '${SAM2_LEVEL_PRESET}'"

if "${TMUX_BIN[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  exit 0
fi

"${TMUX_BIN[@]}" new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && unset LD_LIBRARY_PATH && unset __EGL_VENDOR_LIBRARY_FILENAMES && hash -r && export CUDA_VISIBLE_DEVICES='${CUDA_ID}' && export MQSC_R1_VLM_MODEL='${VLM_MODEL_NAME}' && export VLM_MODEL='${VLM_MODEL_NAME}' && ${RUN_CMD}"

echo "started tmux session ${SESSION_NAME} on CUDA ${CUDA_ID}"
echo "VLM model: ${VLM_MODEL_NAME}"
echo "SAM2 preset: ${SAM2_LEVEL_PRESET}"
echo "Output dir: output_logs/anchor/single_goal/vista2mqsc_sam2/${SAM2_LEVEL_PRESET}-${DESC_MODE}-${RUN_TAG}"
echo "Use '${TMUX_BIN[*]} attach -t ${SESSION_NAME}' to watch logs."
