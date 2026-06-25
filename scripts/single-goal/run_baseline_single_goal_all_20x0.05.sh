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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "${USE_LOCAL_NVIDIA_580:-0}" = "1" ] && [ -f "${PROJECT_ROOT}/scripts/use-local-nvidia-580.126.09.sh" ]; then
  source "${PROJECT_ROOT}/scripts/use-local-nvidia-580.126.09.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="${NAVIGATION_DATA_PATH:-/home/chenlin/krona/anchor-nav/LangMap_Annotations}"

# Usage:
#   bash scripts/single-goal/run_baseline_single_goal_all_20x0.05.sh [detailed|concise] [stable_tag]
DESC_MODE="${1:-detailed}"
USER_TAG="${2:-full}"
if [ "${DESC_MODE}" != "detailed" ] && [ "${DESC_MODE}" != "concise" ]; then
  echo "Invalid first arg: ${DESC_MODE}. Must be 'detailed' or 'concise'."
  exit 1
fi

RUN_TAG="${DESC_MODE}-${USER_TAG}"
OUT_DIR="output_logs/anchor/single_goal/baseline/${RUN_TAG}"
LOG_DIR="${OUT_DIR}/shard_logs"
mkdir -p "${LOG_DIR}"

BASELINE_SEED="${BASELINE_SEED:-1234}"

echo ">>> Single-goal baseline run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> Baseline seed: ${BASELINE_SEED}"

{
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/refhm3d-nav-single-goal-baseline.py"
  echo "method=baseline"
  echo "segmenter=fastsam"
  echo "task_levels=object,room,region,instance"
  echo "single_goal_total_expected=9420"
  echo "slice_range=0.0-1.0"
  echo "slice_step=0.05"
  echo "num_shards_total=20"
  echo "schedule=sequential_shards_resume_by_output_json"
  echo "seed=${BASELINE_SEED}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "ld_library_path=${LD_LIBRARY_PATH:-}"
  echo "egl_vendor_library_filenames=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "use_local_nvidia_580=${USE_LOCAL_NVIDIA_580:-0}"
  echo "navigation_data_path=${NAVIGATION_DATA_PATH}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_baseline_single_goal_all_20x0.05.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local DESC_FLAG=""
  local OUT_JSON
  local SHARD_LOG="${LOG_DIR}/baseline_${DESC_MODE}_${START_RATIO}_${END_RATIO}.log"

  if [ "${DESC_MODE}" = "concise" ]; then
    OUT_JSON="${OUT_DIR}/refhm3d_single_goal_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_single_goal_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi

  echo ">>> [baseline ${DESC_MODE}] shard ${START_RATIO}-${END_RATIO}" | tee -a "${SHARD_LOG}"
  python3 hm3d-online/refhm3d-nav-single-goal-baseline.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --seed "${BASELINE_SEED}" \
    --output_log_dir "${OUT_DIR}" 2>&1 | tee -a "${SHARD_LOG}"
}

STARTS=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
ENDS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
for i in "${!STARTS[@]}"; do
  run_one "${STARTS[$i]}" "${ENDS[$i]}"
done

echo ">>> Done single-goal baseline ${DESC_MODE}. Output: ${OUT_DIR}"
