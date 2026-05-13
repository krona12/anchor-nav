#!/usr/bin/env bash
set -euo pipefail
cd /home/chenlin/krona/anchor-nav
_SAVED_ARGV=("$@")
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="output_process/mile-tiebreak-minimal-${TS}"
mkdir -p "${OUT_DIR}"
python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py \
  --start_ratio 0.05 \
  --end_ratio 0.5 \
  --task_levels "object,room,region,instance" \
  --navigation_data_path "/home/chenlin/krona/anchor-nav/LangMap_Annotations" \
  --hm3d_data_base_path "/home/chenlin/krona/MTU3D/datascene" \
  --pq3d_stage1_path "/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all" \
  --pq3d_stage2_path "/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-goat" \
  --seed 1234 \
  --mile_decision_radius_m 0.75 \
  --mile_candidate_radii_m "0.5,0.75" \
  --mile_candidate_view_count 20 \
  --mile_camera_height_m 1.31 \
  --mile_max_snap_distance_m 0.60 \
  --mile_scene_sample_count 0 \
  --mile_occlusion_radius_m 0.05 \
  --mile_max_ray_sample_count 1000 \
  --mile_min_visibility_score 0.0 \
  --mile_visibility_tie_epsilon 0.0 \
  --mile_apply_task_levels "object,room,region,instance" \
  --effectiveness_threshold_m 1.0 \
  --max_eval_tasks 20 \
  --output_log_dir "${OUT_DIR}" \
  "$@"

echo "OUT_DIR=${OUT_DIR}"
LOG="$(ls -1t "${OUT_DIR}"/refhm3d-nav-sequence-analyze-anchor-mile-refine1-*.log | head -1)"
echo "LOG=${LOG}"
grep -E "selection_policy|visibility_tie_epsilon|\\[mile-refine1\\]\\[module\\]|\\[Metrics\\]|module-status-counts|case-counts|Traceback|module-error|follow-error" "${LOG}" | tail -80 || true
