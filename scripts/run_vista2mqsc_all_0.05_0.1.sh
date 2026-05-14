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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/run_vista2mqsc_all_0.05_0.1.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/anchor/vista2mqsc_all_0.05_0.1/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

MQSC_R1_SEED="${MQSC_R1_SEED:-1234}"
MQSC_R1_TOP_K="${MQSC_R1_TOP_K:-8}"
MQSC_R1_TEMPERATURE="${MQSC_R1_TEMPERATURE:-1.0}"
MQSC_R1_CLUSTER_EPS="${MQSC_R1_CLUSTER_EPS:-1.2}"
MQSC_R1_MIN_REGION_COVERAGE="${MQSC_R1_MIN_REGION_COVERAGE:-0.5}"
MQSC_R1_MIN_TARGET_PROB="${MQSC_R1_MIN_TARGET_PROB:-0.05}"
MQSC_R1_MIN_REGION_MARGIN="${MQSC_R1_MIN_REGION_MARGIN:-0.05}"
MQSC_R1_MIN_SELECTED_GAIN="${MQSC_R1_MIN_SELECTED_GAIN:--0.02}"
MQSC_R1_VLM_MODEL="${MQSC_R1_VLM_MODEL:-${VLM_MODEL:-gpt-4o-mini}}"
MQSC_R1_VLM_MAX_RETRIES="${MQSC_R1_VLM_MAX_RETRIES:-3}"
MQSC_R1_VLM_RETRY_SLEEP_SEC="${MQSC_R1_VLM_RETRY_SLEEP_SEC:-1.0}"
MQSC_R1_DISABLE_VLM="${MQSC_R1_DISABLE_VLM:-0}"
MQSC_R1_DISABLE_NO_PROXY="${MQSC_R1_DISABLE_NO_PROXY:-0}"
MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE="${MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE:-0}"
MQSC_R1_DISABLE_DEBUG_JSON="${MQSC_R1_DISABLE_DEBUG_JSON:-0}"

VISTALS_ENABLE_VVD_REPLACEMENT="${VISTALS_ENABLE_VVD_REPLACEMENT:-1}"
VISTALS_DISABLE_VISIBLE_BASELINE_GUARD="${VISTALS_DISABLE_VISIBLE_BASELINE_GUARD:-1}"
VISTALS_TARGET_SAMPLE_COUNT="${VISTALS_TARGET_SAMPLE_COUNT:-300}"
VISTALS_SCENE_SAMPLE_COUNT="${VISTALS_SCENE_SAMPLE_COUNT:-0}"
VISTALS_MAX_RAY_SAMPLE_COUNT="${VISTALS_MAX_RAY_SAMPLE_COUNT:-300}"
VISTALS_MIN_VISIBILITY_SCORE="${VISTALS_MIN_VISIBILITY_SCORE:-0.02}"
VISTALS_RADIAL_STEP_M="${VISTALS_RADIAL_STEP_M:-0.10}"
VISTALS_ANGLE_STEP_DEG="${VISTALS_ANGLE_STEP_DEG:-10.0}"

echo ">>> Vista2MQSC refine1 all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> MQSC-R1 VLM model: ${MQSC_R1_VLM_MODEL}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py"
  echo "module=vista2mqsc_mqsc_r1_then_vista_ls"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0.05-0.1"
  echo "seed=${MQSC_R1_SEED}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "mqsc_r1_top_k=${MQSC_R1_TOP_K}"
  echo "mqsc_r1_cluster_eps=${MQSC_R1_CLUSTER_EPS}"
  echo "mqsc_r1_min_region_coverage=${MQSC_R1_MIN_REGION_COVERAGE}"
  echo "mqsc_r1_min_target_prob=${MQSC_R1_MIN_TARGET_PROB}"
  echo "mqsc_r1_min_region_margin=${MQSC_R1_MIN_REGION_MARGIN}"
  echo "mqsc_r1_min_selected_gain=${MQSC_R1_MIN_SELECTED_GAIN}"
  echo "mqsc_r1_vlm_model=${MQSC_R1_VLM_MODEL}"
  echo "vistals_enable_vvd_replacement=${VISTALS_ENABLE_VVD_REPLACEMENT}"
  echo "vistals_disable_visible_baseline_guard=${VISTALS_DISABLE_VISIBLE_BASELINE_GUARD}"
  echo "vistals_target_sample_count=${VISTALS_TARGET_SAMPLE_COUNT}"
  echo "vistals_scene_sample_count=${VISTALS_SCENE_SAMPLE_COUNT}"
  echo "vistals_max_ray_sample_count=${VISTALS_MAX_RAY_SAMPLE_COUNT}"
  echo "vistals_min_visibility_score=${VISTALS_MIN_VISIBILITY_SCORE}"
  echo "vistals_radial_step_m=${VISTALS_RADIAL_STEP_M}"
  echo "vistals_angle_step_deg=${VISTALS_ANGLE_STEP_DEG}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_vista2mqsc_all_0.05_0.1.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vista2mqsc_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vista2mqsc_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vista2mqsc_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vista2mqsc_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
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
  local VISTALS_REPLACE_FLAG=""
  local VISTALS_BASELINE_GUARD_FLAG=""
  if [ "${MQSC_R1_DISABLE_VLM}" = "1" ]; then VLM_FLAG="--mqsc_r1_disable_vlm"; fi
  if [ "${MQSC_R1_DISABLE_NO_PROXY}" = "1" ]; then NO_PROXY_FLAG="--mqsc_r1_disable_no_proxy"; fi
  if [ "${MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE}" = "1" ]; then HEURISTIC_FLAG="--mqsc_r1_disable_heuristic_decompose"; fi
  if [ "${MQSC_R1_DISABLE_DEBUG_JSON}" = "1" ]; then DEBUG_JSON_FLAG="--mqsc_r1_disable_debug_json"; fi
  if [ "${VISTALS_ENABLE_VVD_REPLACEMENT}" = "1" ]; then VISTALS_REPLACE_FLAG="--vistals_enable_vvd_replacement"; fi
  if [ "${VISTALS_DISABLE_VISIBLE_BASELINE_GUARD}" = "1" ]; then VISTALS_BASELINE_GUARD_FLAG="--vistals_disable_visible_baseline_guard"; fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --seed "${MQSC_R1_SEED}" \
    --mqsc_r1_top_k "${MQSC_R1_TOP_K}" \
    --mqsc_r1_temperature "${MQSC_R1_TEMPERATURE}" \
    --mqsc_r1_cluster_eps "${MQSC_R1_CLUSTER_EPS}" \
    --mqsc_r1_min_region_coverage "${MQSC_R1_MIN_REGION_COVERAGE}" \
    --mqsc_r1_min_target_prob "${MQSC_R1_MIN_TARGET_PROB}" \
    --mqsc_r1_min_region_margin "${MQSC_R1_MIN_REGION_MARGIN}" \
    --mqsc_r1_min_selected_gain "${MQSC_R1_MIN_SELECTED_GAIN}" \
    --mqsc_r1_vlm_model "${MQSC_R1_VLM_MODEL}" \
    --mqsc_r1_vlm_max_retries "${MQSC_R1_VLM_MAX_RETRIES}" \
    --mqsc_r1_vlm_retry_sleep_sec "${MQSC_R1_VLM_RETRY_SLEEP_SEC}" \
    ${VLM_FLAG} \
    ${NO_PROXY_FLAG} \
    ${HEURISTIC_FLAG} \
    ${DEBUG_JSON_FLAG} \
    ${VISTALS_REPLACE_FLAG} \
    ${VISTALS_BASELINE_GUARD_FLAG} \
    --vistals_target_sample_count "${VISTALS_TARGET_SAMPLE_COUNT}" \
    --vistals_scene_sample_count "${VISTALS_SCENE_SAMPLE_COUNT}" \
    --vistals_max_ray_sample_count "${VISTALS_MAX_RAY_SAMPLE_COUNT}" \
    --vistals_min_visibility_score "${VISTALS_MIN_VISIBILITY_SCORE}" \
    --vistals_radial_step_m "${VISTALS_RADIAL_STEP_M}" \
    --vistals_angle_step_deg "${VISTALS_ANGLE_STEP_DEG}" \
    --vistals_apply_task_levels "object,room,region,instance" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.1

echo ">>> Done Vista2MQSC all-task [0.05,0.1]. Output: ${OUT_DIR}"
