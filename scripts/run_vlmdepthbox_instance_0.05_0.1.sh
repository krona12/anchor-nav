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

# Usage:
#   bash scripts/run_vlmdepthbox_instance_0.05_0.1.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/anchor/vlmdepthbox_instance_0.05_0.1/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

VLM_MODEL="${VLM_MODEL:-gpt-4o-mini}"
VLM_TIMEOUT="${VLM_TIMEOUT:-60}"
VLMDEPTHBOX_TARGET_MODE="${VLMDEPTHBOX_TARGET_MODE:-xz_baseline_y}"
BBOX_INNER_FRACTION="${BBOX_INNER_FRACTION:-0.0}"

echo ">>> VLMDepthBox refine1 instance-only run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> VLM model: ${VLM_MODEL}"
echo ">>> VLM timeout: ${VLM_TIMEOUT}"
echo ">>> VLMDepthBox target mode: ${VLMDEPTHBOX_TARGET_MODE}"
echo ">>> BBox inner fraction: ${BBOX_INNER_FRACTION}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "task_levels=instance"
  echo "slice_range=0.05-0.1"
  echo "slice_step=0.05"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "vlm_model=${VLM_MODEL}"
  echo "vlm_timeout=${VLM_TIMEOUT}"
  echo "vlmdepthbox_top_k=5"
  echo "vlmdepthbox_target_mode=${VLMDEPTHBOX_TARGET_MODE}"
  echo "bbox_inner_fraction=${BBOX_INNER_FRACTION}"
  echo "frontier_visit_resolution_m=0.1"
  echo "non_baseline_policy=best_index_non1_uses_bbox_depth_projection;best_index_1_keeps_baseline_object_position"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_vlmdepthbox_instance_0.05_0.1.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vlmdepthbox_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vlmdepthbox_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vlmdepthbox_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vlmdepthbox_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi
  if [ -f "${EFF_JSON}" ] && [ ! -s "${EFF_JSON}" ]; then
    rm -f "${EFF_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmdepthbox-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --vlmdepthbox_top_k 5 \
    --vlmdepthbox_target_mode "${VLMDEPTHBOX_TARGET_MODE}" \
    --bbox_inner_fraction "${BBOX_INNER_FRACTION}" \
    --vlmdepthbox_vlm_model "${VLM_MODEL}" \
    --vlmdepthbox_vlm_timeout "${VLM_TIMEOUT}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.1

echo ">>> Done vlmdepthbox instance-only [0.05,0.1]. Output: ${OUT_DIR}"
