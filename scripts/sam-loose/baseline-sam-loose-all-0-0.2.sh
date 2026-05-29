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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
export SAM_CHECKPOINT="${SAM_CHECKPOINT:-/home/chenlin/krona/anchor-nav/hm3d-online/SAM/sam_vit_h_4b8939.pth}"
export SAM_POINTS_PER_BATCH="${SAM_POINTS_PER_BATCH:-64}"
export SAM_LEVEL_PRESET="loose"
LOCAL_NVIDIA_ROOT="${LOCAL_NVIDIA_ROOT:-/home/chenlin/krona/anchor-nav/local_nvidia_580_126_09/root}"
USE_LOCAL_NVIDIA_580_126_09="${USE_LOCAL_NVIDIA_580_126_09:-auto}"
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

if [ "${USE_LOCAL_NVIDIA_580_126_09}" = "1" ] || \
   { [ "${USE_LOCAL_NVIDIA_580_126_09}" = "auto" ] && ! nvidia-smi >/dev/null 2>&1; }; then
  LOCAL_NVIDIA_LIB="${LOCAL_NVIDIA_ROOT}/usr/lib/x86_64-linux-gnu"
  LOCAL_NVIDIA_EGL_JSON="${LOCAL_NVIDIA_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  if [ -d "${LOCAL_NVIDIA_LIB}" ] && [ -f "${LOCAL_NVIDIA_EGL_JSON}" ]; then
    export LD_LIBRARY_PATH="${LOCAL_NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-${LOCAL_NVIDIA_EGL_JSON}}"
    export NVIDIA_SMI_BIN="${LOCAL_NVIDIA_ROOT}/usr/bin/nvidia-smi"
    echo ">>> Using local NVIDIA 580.126.09 user-space libraries: ${LOCAL_NVIDIA_LIB}"
  fi
fi

# Usage:
#   bash scripts/sam-loose/baseline-sam-loose-all-0-0.2.sh [detailed|concise] [optional_tag]
DESC_MODE="${1:-detailed}"
USER_TAG="${2:-}"
if [ "${DESC_MODE}" != "detailed" ] && [ "${DESC_MODE}" != "concise" ]; then
  echo "Invalid first arg: ${DESC_MODE}. Must be 'detailed' or 'concise'."
  exit 1
fi

if [ ! -f "${SAM_CHECKPOINT}" ]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}"
  exit 1
fi

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

OUT_DIR="output_logs/baseline_sam_loose_all_0_0.2/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> SAM-loose baseline all-task run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> SAM_CHECKPOINT: ${SAM_CHECKPOINT}"
echo ">>> SAM_LEVEL_PRESET: ${SAM_LEVEL_PRESET}"

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
  echo "segmenter=sam"
  echo "sam_level_preset=${SAM_LEVEL_PRESET}"
  echo "sam_model_type=${SAM_MODEL_TYPE}"
  echo "sam_checkpoint=${SAM_CHECKPOINT}"
  echo "sam_points_per_batch=${SAM_POINTS_PER_BATCH}"
  echo "hf_endpoint=${HF_ENDPOINT}"
  echo "use_local_nvidia_580_126_09=${USE_LOCAL_NVIDIA_580_126_09}"
  echo "local_nvidia_root=${LOCAL_NVIDIA_ROOT}"
  echo "egl_vendor_library_filenames=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/baseline-sam-loose-all-0-0.2.sh.snapshot"
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

  python3 hm3d-online/refhm3d-nav-sequence-baseline-sam.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "object,room,region,instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0 0.2

echo ">>> Done SAM-loose baseline all-task [0,0.2]. Output: ${OUT_DIR}"
