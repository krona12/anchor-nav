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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_ovon_vista2mqsc_all_0.0_1.0.sh [optional_tag]
USER_TAG="${1:-}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-ovon"
if [ -n "${USER_TAG}" ]; then
  RUN_TAG="${RUN_TAG}-${USER_TAG}"
fi

OUT_DIR="output_logs/anchor/ovon_vista2mqsc_all_0.0_1.0/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

DATA_SET_PATH="${DATA_SET_PATH:-/disks/amax_robot_dataset/embodied/embodied_bench_data/our-set/ovon_full_set.json}"
NAVIGATION_DATA_PATH="${NAVIGATION_DATA_PATH:-/disks/amax_robot_dataset/embodied/embodied_bench_data/ovon/}"
HM3D_DATA_BASE_PATH="${HM3D_DATA_BASE_PATH:-/home/chenlin/krona/MTU3D/datascene}"
PQ3D_STAGE1_PATH="${PQ3D_STAGE1_PATH:-/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all}"
PQ3D_STAGE2_PATH="${PQ3D_STAGE2_PATH:-/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-ovon}"

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
QUIET_NAV_STEPS="${QUIET_NAV_STEPS:-0}"
DECISION_LOG_INTERVAL="${DECISION_LOG_INTERVAL:-0}"

echo ">>> OVON Vista2MQSC refine1 run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> MQSC-R1 VLM model: ${MQSC_R1_VLM_MODEL}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/ovon-nav-vista2mqsc-refine1.py"
  echo "module=vista2mqsc_mqsc_r1_then_vista_ls"
  echo "dataset=ovon"
  echo "slice_range=0.0-1.0"
  echo "seed=${MQSC_R1_SEED}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "data_set_path=${DATA_SET_PATH}"
  echo "navigation_data_path=${NAVIGATION_DATA_PATH}"
  echo "hm3d_data_base_path=${HM3D_DATA_BASE_PATH}"
  echo "pq3d_stage1_path=${PQ3D_STAGE1_PATH}"
  echo "pq3d_stage2_path=${PQ3D_STAGE2_PATH}"
  echo "mqsc_r1_top_k=${MQSC_R1_TOP_K}"
  echo "mqsc_r1_temperature=${MQSC_R1_TEMPERATURE}"
  echo "mqsc_r1_cluster_eps=${MQSC_R1_CLUSTER_EPS}"
  echo "mqsc_r1_min_region_coverage=${MQSC_R1_MIN_REGION_COVERAGE}"
  echo "mqsc_r1_min_target_prob=${MQSC_R1_MIN_TARGET_PROB}"
  echo "mqsc_r1_min_region_margin=${MQSC_R1_MIN_REGION_MARGIN}"
  echo "mqsc_r1_min_selected_gain=${MQSC_R1_MIN_SELECTED_GAIN}"
  echo "mqsc_r1_vlm_model=${MQSC_R1_VLM_MODEL}"
  echo "mqsc_r1_vlm_max_retries=${MQSC_R1_VLM_MAX_RETRIES}"
  echo "mqsc_r1_vlm_retry_sleep_sec=${MQSC_R1_VLM_RETRY_SLEEP_SEC}"
  echo "mqsc_r1_disable_vlm=${MQSC_R1_DISABLE_VLM}"
  echo "mqsc_r1_disable_no_proxy=${MQSC_R1_DISABLE_NO_PROXY}"
  echo "mqsc_r1_disable_heuristic_decompose=${MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE}"
  echo "mqsc_r1_disable_debug_json=${MQSC_R1_DISABLE_DEBUG_JSON}"
  echo "vistals_enable_vvd_replacement=${VISTALS_ENABLE_VVD_REPLACEMENT}"
  echo "vistals_disable_visible_baseline_guard=${VISTALS_DISABLE_VISIBLE_BASELINE_GUARD}"
  echo "vistals_target_sample_count=${VISTALS_TARGET_SAMPLE_COUNT}"
  echo "vistals_scene_sample_count=${VISTALS_SCENE_SAMPLE_COUNT}"
  echo "vistals_max_ray_sample_count=${VISTALS_MAX_RAY_SAMPLE_COUNT}"
  echo "vistals_min_visibility_score=${VISTALS_MIN_VISIBILITY_SCORE}"
  echo "vistals_radial_step_m=${VISTALS_RADIAL_STEP_M}"
  echo "vistals_angle_step_deg=${VISTALS_ANGLE_STEP_DEG}"
  echo "quiet_nav_steps=${QUIET_NAV_STEPS}"
  echo "decision_log_interval=${DECISION_LOG_INTERVAL}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_ovon_vista2mqsc_all_0.0_1.0.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON="${OUT_DIR}/ovon_vista2mqsc_refine1_${START_RATIO}_${END_RATIO}.json"
  local EFF_JSON="${OUT_DIR}/ovon_vista2mqsc_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"

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
  local QUIET_FLAG=""
  if [ "${MQSC_R1_DISABLE_VLM}" = "1" ]; then VLM_FLAG="--mqsc_r1_disable_vlm"; fi
  if [ "${MQSC_R1_DISABLE_NO_PROXY}" = "1" ]; then NO_PROXY_FLAG="--mqsc_r1_disable_no_proxy"; fi
  if [ "${MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE}" = "1" ]; then HEURISTIC_FLAG="--mqsc_r1_disable_heuristic_decompose"; fi
  if [ "${MQSC_R1_DISABLE_DEBUG_JSON}" = "1" ]; then DEBUG_JSON_FLAG="--mqsc_r1_disable_debug_json"; fi
  if [ "${VISTALS_ENABLE_VVD_REPLACEMENT}" = "1" ]; then VISTALS_REPLACE_FLAG="--vistals_enable_vvd_replacement"; fi
  if [ "${VISTALS_DISABLE_VISIBLE_BASELINE_GUARD}" = "1" ]; then VISTALS_BASELINE_GUARD_FLAG="--vistals_disable_visible_baseline_guard"; fi
  if [ "${QUIET_NAV_STEPS}" = "1" ]; then QUIET_FLAG="--quiet_nav_steps"; fi

  python3 hm3d-online/ovon-nav-vista2mqsc-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    --data_set_path "${DATA_SET_PATH}" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --hm3d_data_base_path "${HM3D_DATA_BASE_PATH}" \
    --pq3d_stage1_path "${PQ3D_STAGE1_PATH}" \
    --pq3d_stage2_path "${PQ3D_STAGE2_PATH}" \
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
    ${QUIET_FLAG} \
    --decision_log_interval "${DECISION_LOG_INTERVAL}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.0 1.0

echo ">>> Done OVON Vista2MQSC [0.0,1.0]. Output: ${OUT_DIR}"
