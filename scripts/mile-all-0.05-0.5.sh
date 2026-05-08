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
#   bash scripts/mile-all-0.05-0.5.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/anchor/mile_all_0.05_0.5/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> Mile all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> Mile module: MSGNav-style VVD only (no VLM call)"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0.05-0.5"
  echo "slice_step=0.05"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "mile_module=lastmile_vvd_only"
  echo "uses_vlm=false"
  echo "mile_decision_radius_m=0.75"
  echo "mile_candidate_view_count=20"
  echo "mile_camera_height_m=1.31"
  echo "mile_max_snap_distance_m=0.60"
  echo "mile_scene_sample_count=0"
  echo "mile_occlusion_radius_m=0.05"
  echo "mile_max_ray_sample_count=1000"
  echo "mile_min_visibility_score=0.0"
  echo "effectiveness_threshold_m=1.0"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/mile-all-0.05-0.5.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_mile_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_mile_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_mile_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_mile_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi
  if [ -f "${EFF_JSON}" ] && [ ! -s "${EFF_JSON}" ]; then
    rm -f "${EFF_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --mile_decision_radius_m 0.75 \
    --mile_candidate_view_count 20 \
    --mile_camera_height_m 1.31 \
    --mile_max_snap_distance_m 0.60 \
    --mile_scene_sample_count 0 \
    --mile_occlusion_radius_m 0.05 \
    --mile_max_ray_sample_count 1000 \
    --mile_min_visibility_score 0.0 \
    --effectiveness_threshold_m 1.0 \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.5

echo ">>> Done mile all-task [0.05,0.5]. Output: ${OUT_DIR}"
