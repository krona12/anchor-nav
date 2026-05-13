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

# Usage:
#   bash scripts/run_mqsc_instance_0.05_0.1.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/anchor/mqsc_instance_0.05_0.1/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

MQSC_SEED="${MQSC_SEED:-1234}"
MQSC_TOP_K="${MQSC_TOP_K:-8}"
MQSC_TEMPERATURE="${MQSC_TEMPERATURE:-1.0}"
MQSC_CLUSTER_EPS="${MQSC_CLUSTER_EPS:-1.2}"
MQSC_MIN_REGION_COVERAGE="${MQSC_MIN_REGION_COVERAGE:-0.5}"
MQSC_MIN_TARGET_PROB="${MQSC_MIN_TARGET_PROB:-0.05}"
MQSC_MIN_REGION_MARGIN="${MQSC_MIN_REGION_MARGIN:-0.05}"
MQSC_MIN_SELECTED_GAIN="${MQSC_MIN_SELECTED_GAIN:--0.02}"
MQSC_VLM_MODEL="${MQSC_VLM_MODEL:-${VLM_MODEL:-gpt-4o-mini}}"
MQSC_VLM_MAX_RETRIES="${MQSC_VLM_MAX_RETRIES:-3}"
MQSC_VLM_RETRY_SLEEP_SEC="${MQSC_VLM_RETRY_SLEEP_SEC:-1.0}"
MQSC_DISABLE_VLM="${MQSC_DISABLE_VLM:-0}"
MQSC_DISABLE_NO_PROXY="${MQSC_DISABLE_NO_PROXY:-0}"
MQSC_DISABLE_HEURISTIC_DECOMPOSE="${MQSC_DISABLE_HEURISTIC_DECOMPOSE:-0}"
MQSC_DISABLE_DEBUG_JSON="${MQSC_DISABLE_DEBUG_JSON:-0}"

echo ">>> MQSC refine1 instance-only run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> MQSC seed: ${MQSC_SEED}"
echo ">>> MQSC VLM model: ${MQSC_VLM_MODEL}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/refhm3d-nav-sequence-analyze-anchor-mqsc-refine1.py"
  echo "module=mqsc_role_aware_region_consensus"
  echo "task_levels=instance"
  echo "slice_range=0.05-0.1"
  echo "slice_step=0.05"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "seed=${MQSC_SEED}"
  echo "mqsc_top_k=${MQSC_TOP_K}"
  echo "mqsc_temperature=${MQSC_TEMPERATURE}"
  echo "mqsc_cluster_eps=${MQSC_CLUSTER_EPS}"
  echo "mqsc_min_region_coverage=${MQSC_MIN_REGION_COVERAGE}"
  echo "mqsc_min_target_prob=${MQSC_MIN_TARGET_PROB}"
  echo "mqsc_min_region_margin=${MQSC_MIN_REGION_MARGIN}"
  echo "mqsc_min_selected_gain=${MQSC_MIN_SELECTED_GAIN}"
  echo "mqsc_vlm_model=${MQSC_VLM_MODEL}"
  echo "mqsc_vlm_max_retries=${MQSC_VLM_MAX_RETRIES}"
  echo "mqsc_vlm_retry_sleep_sec=${MQSC_VLM_RETRY_SLEEP_SEC}"
  echo "mqsc_disable_vlm=${MQSC_DISABLE_VLM}"
  echo "mqsc_disable_no_proxy=${MQSC_DISABLE_NO_PROXY}"
  echo "mqsc_disable_heuristic_decompose=${MQSC_DISABLE_HEURISTIC_DECOMPOSE}"
  echo "mqsc_disable_debug_json=${MQSC_DISABLE_DEBUG_JSON}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_mqsc_instance_0.05_0.1.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_mqsc_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_mqsc_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_mqsc_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_mqsc_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi
  if [ -f "${EFF_JSON}" ] && [ ! -s "${EFF_JSON}" ]; then
    rm -f "${EFF_JSON}"
  fi

  local VLM_FLAG=""
  local NO_PROXY_FLAG=""
  local HEURISTIC_FLAG=""
  local DEBUG_JSON_FLAG=""
  if [ "${MQSC_DISABLE_VLM}" = "1" ]; then
    VLM_FLAG="--mqsc_disable_vlm"
  fi
  if [ "${MQSC_DISABLE_NO_PROXY}" = "1" ]; then
    NO_PROXY_FLAG="--mqsc_disable_no_proxy"
  fi
  if [ "${MQSC_DISABLE_HEURISTIC_DECOMPOSE}" = "1" ]; then
    HEURISTIC_FLAG="--mqsc_disable_heuristic_decompose"
  fi
  if [ "${MQSC_DISABLE_DEBUG_JSON}" = "1" ]; then
    DEBUG_JSON_FLAG="--mqsc_disable_debug_json"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-mqsc-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --seed "${MQSC_SEED}" \
    --mqsc_top_k "${MQSC_TOP_K}" \
    --mqsc_temperature "${MQSC_TEMPERATURE}" \
    --mqsc_cluster_eps "${MQSC_CLUSTER_EPS}" \
    --mqsc_min_region_coverage "${MQSC_MIN_REGION_COVERAGE}" \
    --mqsc_min_target_prob "${MQSC_MIN_TARGET_PROB}" \
    --mqsc_min_region_margin "${MQSC_MIN_REGION_MARGIN}" \
    --mqsc_min_selected_gain "${MQSC_MIN_SELECTED_GAIN}" \
    --mqsc_vlm_model "${MQSC_VLM_MODEL}" \
    --mqsc_vlm_max_retries "${MQSC_VLM_MAX_RETRIES}" \
    --mqsc_vlm_retry_sleep_sec "${MQSC_VLM_RETRY_SLEEP_SEC}" \
    ${VLM_FLAG} \
    ${NO_PROXY_FLAG} \
    ${HEURISTIC_FLAG} \
    ${DEBUG_JSON_FLAG} \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.1

echo ">>> Done mqsc instance-only [0.05,0.1]. Output: ${OUT_DIR}"
