#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Launch four disjoint workers for the existing SAM2.1 balanced detailed qwen run.
# Existing completed JSON shards are reused by the Python resume logic.
DESC_MODE="${DESC_MODE:-detailed}"
RUN_TAG="${RUN_TAG:-qwen_vl_plus_full}"
SAM2_LEVEL_PRESET="${SAM2_LEVEL_PRESET:-balanced}"
VLM_MODEL_NAME="${MQSC_R1_VLM_MODEL:-${VLM_MODEL:-qwen-vl-plus}}"

TMUX_BIN=(tmux)
if [[ -n "${SINGLE_GOAL_TMUX_SOCKET:-}" ]]; then
  TMUX_BIN=(tmux -S "${SINGLE_GOAL_TMUX_SOCKET}")
fi

if [ "${DESC_MODE}" != "detailed" ]; then
  echo "This parallel launch plan is intended for detailed mode only; got DESC_MODE=${DESC_MODE}"
  exit 1
fi
if [ "${SAM2_LEVEL_PRESET}" != "balanced" ]; then
  echo "This parallel launch plan is intended for SAM2 balanced only; got SAM2_LEVEL_PRESET=${SAM2_LEVEL_PRESET}"
  exit 1
fi

OUT_DIR="output_logs/anchor/single_goal/vista2mqsc_sam2/${SAM2_LEVEL_PRESET}-${DESC_MODE}-${RUN_TAG}"
mkdir -p "${OUT_DIR}/worker_logs"

WORKER_IDS=(w0 w1 w2 w3)
CUDA_IDS=(1 3 3 3)
SHARD_INDICES=(
  "2,5,8,11,14,17"
  "3,7,12,16"
  "4,9,13,18"
  "6,10,15,19"
)

{
  echo "parallel_plan=sam2_balanced_single_goal_parallel4"
  echo "desc_mode=${DESC_MODE}"
  echo "run_tag=${RUN_TAG}"
  echo "sam2_level_preset=${SAM2_LEVEL_PRESET}"
  echo "vlm_model=${VLM_MODEL_NAME}"
  echo "output_dir=${OUT_DIR}"
  echo "worker_count=4"
  for i in "${!WORKER_IDS[@]}"; do
    echo "worker_${WORKER_IDS[$i]}_cuda=${CUDA_IDS[$i]}"
    echo "worker_${WORKER_IDS[$i]}_shard_indices=${SHARD_INDICES[$i]}"
  done
} > "${OUT_DIR}/parallel4_plan.txt"

start_worker () {
  local WORKER_ID="$1"
  local CUDA_ID="$2"
  local SHARDS="$3"
  local SESSION_NAME="single_goal_sam2_balanced_${WORKER_ID}_cuda${CUDA_ID}_${RUN_TAG}"

  if "${TMUX_BIN[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}"
    return 0
  fi

  "${TMUX_BIN[@]}" new-session -d -s "${SESSION_NAME}" \
    "cd '${PROJECT_ROOT}' && unset LD_LIBRARY_PATH && unset __EGL_VENDOR_LIBRARY_FILENAMES && hash -r && export CUDA_VISIBLE_DEVICES='${CUDA_ID}' && export MQSC_R1_VLM_MODEL='${VLM_MODEL_NAME}' && export VLM_MODEL='${VLM_MODEL_NAME}' && export SAM2_WORKER_ID='${WORKER_ID}' && export SAM2_SHARD_INDICES='${SHARDS}' && bash scripts/single-goal/run_vista2mqsc_sam2_single_goal_all_20x0.05.sh '${DESC_MODE}' '${RUN_TAG}' '${SAM2_LEVEL_PRESET}'"
  echo "started ${SESSION_NAME}: CUDA=${CUDA_ID} shards=${SHARDS}"
}

for i in "${!WORKER_IDS[@]}"; do
  start_worker "${WORKER_IDS[$i]}" "${CUDA_IDS[$i]}" "${SHARD_INDICES[$i]}"
done

echo "Output dir: ${OUT_DIR}"
echo "Plan: ${OUT_DIR}/parallel4_plan.txt"
echo "Worker logs: ${OUT_DIR}/worker_logs/"
