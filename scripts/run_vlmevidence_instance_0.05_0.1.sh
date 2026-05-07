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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

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

OUT_DIR="output_logs/anchor/vlmevidence_instance_0.05_0.1/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

VLM_MODEL="${VLM_MODEL:-gpt-4o-mini}"

echo ">>> VLMEvidence refine1 instance-only run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> VLM model: ${VLM_MODEL}"

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
  echo "panorama_subsample_frames=12"
  echo "vlmevidence_top_k=5"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_vlmevidence_instance_0.05_0.1.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local DESC_FLAG=""
  if [ "${DESC_MODE}" = "concise" ]; then
    DESC_FLAG="--concise_description"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --panorama_subsample_frames 12 \
    --vlmevidence_vlm_model "${VLM_MODEL}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.05 0.1

echo ">>> Done vlmevidence instance-only [0.05,0.1]. Output: ${OUT_DIR}"
