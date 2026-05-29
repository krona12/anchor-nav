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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/concise/00_02/baseline-concise-all-0-0.2.sh [optional_tag]
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

OUT_DIR="output_logs/baseline_all_0_0.2/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> Baseline concise all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "task_levels=object,room,region,instance"
  echo "slice_range=0-0.2"
  echo "slice_step=0.2"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "local_nvidia_root=${LOCAL_NVIDIA_ROOT:-}"
  echo "egl_vendor_library_filenames=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/baseline-concise-all-0-0.2.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON
  local DESC_FLAG=""
  if [ "${DESC_MODE}" = "concise" ]; then
    OUT_JSON="${OUT_DIR}/refhm3d_seq_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-baseline.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0 0.2

echo ">>> Done baseline all-task [0,0.2]. Output: ${OUT_DIR}"
