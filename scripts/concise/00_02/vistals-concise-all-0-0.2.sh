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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/use-local-nvidia-580.126.09.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/concise/00_02/vistals-concise-all-0-0.2.sh [optional_tag]
DESC_MODE="concise"
USER_TAG="${1:-}"

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
if ! NVIDIA_SMI_OUTPUT="$("${NVIDIA_SMI_BIN}" 2>&1)"; then
  echo "nvidia-smi failed before starting Habitat-Sim:"
  echo "${NVIDIA_SMI_OUTPUT}"
  if [ -r /proc/driver/nvidia/version ]; then
    echo "Loaded kernel module:"
    cat /proc/driver/nvidia/version
  fi
  echo "Habitat-Sim EGL rendering requires matching NVIDIA kernel/user-space driver libraries."
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-${DESC_MODE}"
if [ -n "${USER_TAG}" ]; then
  RUN_TAG="${RUN_TAG}-${USER_TAG}"
fi

OUT_DIR="output_logs/anchor/vistals_all_0_0.2/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

VISTALS_SEED="${VISTALS_SEED:-1234}"

echo ">>> VISTA-LS concise all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> VISTA-LS module: level-set medial-center target adjustment + planner filter"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> VISTA-LS seed: ${VISTALS_SEED}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/refhm3d-nav-sequence-analyze-anchor-vistals-refine1.py"
  echo "module=vista_ls_level_set_medial_center"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0-0.2"
  echo "slice_step=0.2"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "seed=${VISTALS_SEED}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "local_nvidia_root=${LOCAL_NVIDIA_ROOT:-}"
  echo "egl_vendor_library_filenames=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "uses_vlm=false"
  echo "vistals_candidate_generator=polar_level_set_grid"
  echo "vistals_camera_height_m=1.50"
  echo "vistals_max_snap_distance_m=0.60"
  echo "vistals_scene_sample_count=0"
  echo "vistals_occlusion_radius_m=0.05"
  echo "vistals_target_sample_count=300"
  echo "vistals_max_ray_sample_count=300"
  echo "vistals_min_visibility_score=0.02"
  echo "vistals_visibility_tie_epsilon=0.0"
  echo "legacy_vista_candidate_args=not_used_by_vista_ls"
  echo "vistals_r_min_m=0.30"
  echo "vistals_r_max_m=1.30"
  echo "vistals_radial_step_m=0.10"
  echo "vistals_angle_step_deg=10.0"
  echo "vistals_shell_min_m=0.35"
  echo "vistals_shell_max_m=1.20"
  echo "vistals_relaxed_shell_min_m=0.30"
  echo "vistals_relaxed_shell_max_m=1.35"
  echo "vistals_min_clearance_m=0.10"
  echo "vistals_min_component_size=3"
  echo "vistals_size_tie_ratio=0.85"
  echo "vistals_enable_vvd_replacement=true"
  echo "vistals_disable_visible_baseline_guard=true"
  echo "vistals_apply_task_levels=object,room,region,instance"
  echo "vistals_apply_non_final_object_decisions=false"
  echo "vistals_enable_planner_filter=true"
  echo "vistals_planner_repair_radii_m=0.20,0.40,0.60,0.80"
  echo "vistals_planner_candidates_per_radius=16"
  echo "decision_log_interval=0"
  echo "effectiveness_threshold_m=1.0"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/vistals-concise-all-0-0.2.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vistals_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vistals_refine1_effectiveness_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vistals_refine1_${START_RATIO}_${END_RATIO}.json"
    EFF_JSON="${OUT_DIR}/refhm3d_seq_vistals_refine1_effectiveness_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi
  if [ -f "${EFF_JSON}" ] && [ ! -s "${EFF_JSON}" ]; then
    rm -f "${EFF_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vistals-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --seed "${VISTALS_SEED}" \
    --vistals_camera_height_m 1.50 \
    --vistals_max_snap_distance_m 0.60 \
    --vistals_target_sample_count 300 \
    --vistals_scene_sample_count 0 \
    --vistals_occlusion_radius_m 0.05 \
    --vistals_max_ray_sample_count 300 \
    --vistals_min_visibility_score 0.02 \
    --vistals_visibility_tie_epsilon 0.0 \
    --vistals_r_min_m 0.30 \
    --vistals_r_max_m 1.30 \
    --vistals_radial_step_m 0.10 \
    --vistals_angle_step_deg 10.0 \
    --vistals_shell_min_m 0.35 \
    --vistals_shell_max_m 1.20 \
    --vistals_relaxed_shell_min_m 0.30 \
    --vistals_relaxed_shell_max_m 1.35 \
    --vistals_min_clearance_m 0.10 \
    --vistals_min_component_size 3 \
    --vistals_size_tie_ratio 0.85 \
    --vistals_enable_vvd_replacement \
    --vistals_disable_visible_baseline_guard \
    --vistals_enable_planner_filter \
    --vistals_planner_repair_radii_m "0.20,0.40,0.60,0.80" \
    --vistals_planner_candidates_per_radius 16 \
    --effectiveness_threshold_m 1.0 \
    --decision_log_interval 0 \
    --output_log_dir "${OUT_DIR}"
}

run_one 0 0.2

echo ">>> Done VISTA-LS all-task [0,0.2]. Output: ${OUT_DIR}"
