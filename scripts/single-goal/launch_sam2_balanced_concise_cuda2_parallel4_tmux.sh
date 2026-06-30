#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Launch four disjoint workers for SAM2.1 balanced concise qwen-vl-plus.
DESC_MODE="${DESC_MODE:-concise}"
RUN_TAG="${RUN_TAG:-qwen_vl_plus_full}"
SAM2_LEVEL_PRESET="${SAM2_LEVEL_PRESET:-balanced}"
VLM_MODEL_NAME="${MQSC_R1_VLM_MODEL:-${VLM_MODEL:-qwen-vl-plus}}"

TMUX_BIN=(tmux)
if [[ -n "${SINGLE_GOAL_TMUX_SOCKET:-}" ]]; then
  TMUX_BIN=(tmux -S "${SINGLE_GOAL_TMUX_SOCKET}")
fi

if [ "${DESC_MODE}" != "concise" ]; then
  echo "This parallel launch plan is intended for concise mode only; got DESC_MODE=${DESC_MODE}"
  exit 1
fi
if [ "${SAM2_LEVEL_PRESET}" != "balanced" ]; then
  echo "This parallel launch plan is intended for SAM2 balanced only; got SAM2_LEVEL_PRESET=${SAM2_LEVEL_PRESET}"
  exit 1
fi

OUT_DIR="output_logs/anchor/single_goal/vista2mqsc_sam2/${SAM2_LEVEL_PRESET}-${DESC_MODE}-${RUN_TAG}"
mkdir -p "${OUT_DIR}/worker_logs"

WORKER_IDS=(w0 w1 w2 w3)
CUDA_IDS=(2 2 2 2)
SHARD_INDICES=(
  "0,1,5,9,11"
  "6,7,8,16,19"
  "2,12,13,15,18"
  "3,4,10,14,17"
)
EXPECTED_TASK_COUNTS=(2394 2390 2398 2238)

{
  echo "parallel_plan=sam2_balanced_single_goal_concise_cuda2_parallel4"
  echo "desc_mode=${DESC_MODE}"
  echo "run_tag=${RUN_TAG}"
  echo "sam2_level_preset=${SAM2_LEVEL_PRESET}"
  echo "vlm_model=${VLM_MODEL_NAME}"
  echo "output_dir=${OUT_DIR}"
  echo "worker_count=4"
  echo "shard_count=20"
  echo "expected_total_tasks=9420"
  echo "split_strategy=greedy_balance_by_single_goal_task_count"
  for i in "${!WORKER_IDS[@]}"; do
    echo "worker_${WORKER_IDS[$i]}_cuda=${CUDA_IDS[$i]}"
    echo "worker_${WORKER_IDS[$i]}_shard_indices=${SHARD_INDICES[$i]}"
    echo "worker_${WORKER_IDS[$i]}_expected_tasks=${EXPECTED_TASK_COUNTS[$i]}"
  done
} > "${OUT_DIR}/parallel4_plan.txt"

start_worker () {
  local WORKER_ID="$1"
  local CUDA_ID="$2"
  local SHARDS="$3"
  local SESSION_NAME="single_goal_sam2_balanced_concise_${WORKER_ID}_cuda${CUDA_ID}_${RUN_TAG}"

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
