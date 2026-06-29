#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

_SAVED_ARGV=("$@")
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one device, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

for arg in "$@"; do
  if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
    exec python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-tffs-vista2mqsc-sequence-refine1.py --help
  fi
done

SCENE_NAME="${SCENE_NAME:-00800-TEEsavR23oF}"
EPISODE_ID="${EPISODE_ID:-0}"
MAX_TASKS_PER_EPISODE="${MAX_TASKS_PER_EPISODE:-1}"
MAX_STEPS="${MAX_STEPS:-120}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-scene_${SCENE_NAME}_ep${EPISODE_ID}_tasks${MAX_TASKS_PER_EPISODE}_cuda${CUDA_VISIBLE_DEVICES}}"
OUT_DIR="${OUT_DIR:-output_logs/anchor/tffs_vista2mqsc_single/${RUN_TAG}}"

TFFS_VLM_MODEL="${TFFS_VLM_MODEL:-${VLM_MODEL:-gpt-4o-mini}}"
TFFS_TIMEOUT_SEC="${TFFS_TIMEOUT_SEC:-120}"
MQSC_R1_VLM_MODEL="${MQSC_R1_VLM_MODEL:-${VLM_MODEL:-gpt-4o-mini}}"

mkdir -p "${OUT_DIR}"

{
  echo "script=$0"
  echo "scene_name=${SCENE_NAME}"
  echo "episode_id=${EPISODE_ID}"
  echo "max_tasks_per_episode=${MAX_TASKS_PER_EPISODE}"
  echo "max_steps=${MAX_STEPS}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "python=hm3d-online/refhm3d-nav-sequence-analyze-anchor-tffs-vista2mqsc-sequence-refine1.py"
  echo "module=tffs_frontier_override_then_vista2mqsc_final_decision"
  echo "tffs_vlm_model=${TFFS_VLM_MODEL}"
  echo "tffs_vlm_call_interval=5"
  echo "tffs_timeout_sec=${TFFS_TIMEOUT_SEC}"
  echo "mqsc_r1_vlm_model=${MQSC_R1_VLM_MODEL}"
  echo "notes=real SIM/PQ3D/TFFS/VLM/MQSC/VISTA-LS run; no smoke; no mock"
} > "${OUT_DIR}/run_args.txt"

python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-tffs-vista2mqsc-sequence-refine1.py \
  --start_ratio 0.0 \
  --end_ratio 1.0 \
  --scene_name "${SCENE_NAME}" \
  --episode_id "${EPISODE_ID}" \
  --max_scenes 1 \
  --max_episodes_per_scene 1 \
  --max_tasks_per_episode "${MAX_TASKS_PER_EPISODE}" \
  --max_steps "${MAX_STEPS}" \
  --task_levels "object,room,region,instance" \
  --navigation_data_path "${ROOT_DIR}/LangMap_Annotations" \
  --tffs_vlm_model "${TFFS_VLM_MODEL}" \
  --tffs_vlm_call_interval 5 \
  --tffs_timeout_sec "${TFFS_TIMEOUT_SEC}" \
  --mqsc_r1_vlm_model "${MQSC_R1_VLM_MODEL}" \
  --vistals_enable_vvd_replacement \
  --vistals_disable_visible_baseline_guard \
  --vistals_apply_task_levels "object,room,region,instance" \
  --output_log_dir "${OUT_DIR}" \
  "$@"

echo ">>> TFFS + Vista2MQSC single real run complete: ${OUT_DIR}"
