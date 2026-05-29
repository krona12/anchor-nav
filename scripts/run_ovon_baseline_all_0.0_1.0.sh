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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export OVON_FAIL_FAST="${OVON_FAIL_FAST:-1}"

export OVON_USE_LOCAL_NVIDIA="${OVON_USE_LOCAL_NVIDIA:-1}"
if [ "${OVON_USE_LOCAL_NVIDIA}" != "0" ] && [ -f "scripts/use-local-nvidia-580.126.09.sh" ]; then
  # The host currently has a 580.126 kernel driver with newer 580.159 user-space
  # libraries. Habitat EGL needs matching user-space libraries to create a context.
  source "scripts/use-local-nvidia-580.126.09.sh"
fi

# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_ovon_baseline_all_0.0_1.0.sh [optional_tag]
USER_TAG="${1:-}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-ovon-baseline"
if [ -n "${USER_TAG}" ]; then
  RUN_TAG="${RUN_TAG}-${USER_TAG}"
fi

OUT_DIR="output_logs/baseline/ovon_all_0.0_1.0/${RUN_TAG}"
mkdir -p "${OUT_DIR}" output_dirs

export OVON_START_RATIO="${OVON_START_RATIO:-0.0}"
export OVON_END_RATIO="${OVON_END_RATIO:-1.0}"
export OVON_DATA_SET_PATH="${OVON_DATA_SET_PATH:-/disks/amax_robot_dataset/embodied/embodied_bench_data/our-set/ovon_full_set.json}"
export OVON_NAVIGATION_DATA_PATH="${OVON_NAVIGATION_DATA_PATH:-/disks/amax_robot_dataset/embodied/embodied_bench_data/ovon/}"
export OVON_HM3D_DATA_BASE_PATH="${OVON_HM3D_DATA_BASE_PATH:-/home/chenlin/krona/MTU3D/datascene}"
export OVON_PQ3D_STAGE1_PATH="${OVON_PQ3D_STAGE1_PATH:-/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all}"
export OVON_PQ3D_STAGE2_PATH="${OVON_PQ3D_STAGE2_PATH:-/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-ovon}"
export OVON_OUTPUT_PATH="${OVON_OUTPUT_PATH:-${OUT_DIR}/ovon_mtu3d_baseline_${OVON_START_RATIO}_${OVON_END_RATIO}.json}"

echo ">>> OVON MTU3D baseline run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> OVON output JSON: ${OVON_OUTPUT_PATH}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=hm3d-online/ovon-nav.py"
  echo "module=mtu3d_baseline"
  echo "dataset=ovon"
  echo "slice_range=${OVON_START_RATIO}-${OVON_END_RATIO}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "ovon_fail_fast=${OVON_FAIL_FAST}"
  echo "ovon_use_local_nvidia=${OVON_USE_LOCAL_NVIDIA}"
  echo "local_nvidia_root=${LOCAL_NVIDIA_ROOT:-}"
  echo "ovon_data_set_path=${OVON_DATA_SET_PATH}"
  echo "ovon_navigation_data_path=${OVON_NAVIGATION_DATA_PATH}"
  echo "ovon_hm3d_data_base_path=${OVON_HM3D_DATA_BASE_PATH}"
  echo "ovon_pq3d_stage1_path=${OVON_PQ3D_STAGE1_PATH}"
  echo "ovon_pq3d_stage2_path=${OVON_PQ3D_STAGE2_PATH}"
  echo "ovon_output_path=${OVON_OUTPUT_PATH}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_ovon_baseline_all_0.0_1.0.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

python3 -u hm3d-online/ovon-nav.py 2>&1 | tee "${OUT_DIR}/run.log"

echo ">>> Done OVON MTU3D baseline [${OVON_START_RATIO},${OVON_END_RATIO}]. Output: ${OUT_DIR}"
