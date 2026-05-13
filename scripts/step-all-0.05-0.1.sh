#!/bin/bash
set -eo pipefail

_SAVED_ARGV=("$@")
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

RUN_CMD="$(printf '%q ' "$0" "$@")"
RUN_CMD="${RUN_CMD% }"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# STEP VLM calls should bypass server proxy by default.
if [ "${STEP_NO_PROXY:-1}" = "1" ]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="*"
  export no_proxy="*"
fi

# Usage:
#   bash scripts/step-all-0.05-0.1.sh [detailed|concise] [optional_tag]
DESC_MODE="${1:-detailed}"
USER_TAG="${2:-}"
if [ "${DESC_MODE}" != "detailed" ] && [ "${DESC_MODE}" != "concise" ]; then
  echo "Invalid first arg: ${DESC_MODE}. Must be 'detailed' or 'concise'."
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-${DESC_MODE}"
if [ -n "${USER_TAG}" ]; then
  RUN_TAG="${RUN_TAG}-${USER_TAG}"
fi

OUT_DIR="output_logs/anchor/step_all_0.05_0.1/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

STEP_SEED="${STEP_SEED:-1234}"
STEP_VLM_MODEL="${STEP_VLM_MODEL:-gpt-4o-mini}"
STEP_VLM_MAX_RETRIES="${STEP_VLM_MAX_RETRIES:-3}"
STEP_VLM_RETRY_SLEEP_SEC="${STEP_VLM_RETRY_SLEEP_SEC:-2.0}"
STEP_DISABLE_VLM_FLAG=""
if [ "${STEP_DISABLE_VLM:-0}" = "1" ]; then
  STEP_DISABLE_VLM_FLAG="--step_disable_vlm"
fi
STEP_USE_PROXY_FLAG=""
if [ "${STEP_NO_PROXY:-1}" = "0" ]; then
  STEP_USE_PROXY_FLAG="--step_use_proxy"
fi

echo ">>> STEP all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> STEP module: anchor-only pre-step then full-description navigation"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> STEP seed: ${STEP_SEED}"
echo ">>> STEP VLM model: ${STEP_VLM_MODEL}"
echo ">>> STEP no proxy: ${STEP_NO_PROXY:-1}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/refhm3d-nav-sequence-analyze-anchor-step-refine1.py"
  echo "module=step_anchor_only_prestep_then_full_description"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0.05-0.1"
  echo "slice_step=0.05"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "seed=${STEP_SEED}"
  echo "uses_vlm=true"
  echo "step_prompt_version=step_anchor_decompose_v4_no_target_query"
  echo "step_apply_task_levels=room,region,instance"
  echo "step_vlm_model=${STEP_VLM_MODEL}"
  echo "step_vlm_max_retries=${STEP_VLM_MAX_RETRIES}"
  echo "step_vlm_retry_sleep_sec=${STEP_VLM_RETRY_SLEEP_SEC}"
  echo "step_no_proxy=${STEP_NO_PROXY:-1}"
  echo "step_disable_vlm=${STEP_DISABLE_VLM:-0}"
  echo "step_max_object_anchors=5"
  echo "step_max_attribute_anchors=4"
  echo "decision_log_interval=0"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/step-all-0.05-0.1.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON
  local EFF_JSON
  local DESC_FLAG=""
  if [ "${DESC_MODE}" = "concise" ]; then
    OUT_JSON="${OUT_DIR}/refhm3d_seq_step_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_step_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_step_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_step_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi
  if [ -f "${EFF_JSON}" ] && [ ! -s "${EFF_JSON}" ]; then
    rm -f "${EFF_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-step-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --seed "${STEP_SEED}" \
    --step_vlm_model "${STEP_VLM_MODEL}" \
    --step_vlm_max_retries "${STEP_VLM_MAX_RETRIES}" \
    --step_vlm_retry_sleep_sec "${STEP_VLM_RETRY_SLEEP_SEC}" \
    --step_max_object_anchors 5 \
    --step_max_attribute_anchors 4 \
    --step_apply_task_levels "room,region,instance" \
    ${STEP_DISABLE_VLM_FLAG} \
    ${STEP_USE_PROXY_FLAG} \
    --decision_log_interval 0 \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.1

echo ">>> Done STEP all-task [0.05,0.1]. Output: ${OUT_DIR}"
