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

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/run_baseline_instance_substitue_0_0.5.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/baseline_substitue_instance_0_0.5/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

VLM_MODEL="${VLM_MODEL:-gpt-4o-mini}"

echo ">>> Baseline substitue instance-only run tag: ${RUN_TAG}"
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
  echo "slice_range=0.0-0.5"
  echo "slice_step=0.05"
  echo "num_shards_total=10"
  echo "schedule=sequential_10_shards_single_pipeline"
  echo "vlm_model=${VLM_MODEL}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_baseline_instance_substitue_0_0.5.sh.snapshot"
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

  python3 hm3d-online/refhm3d-nav-sequence-baseline-substitue.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "instance" \
    --vlm_model "${VLM_MODEL}" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.0 0.05
run_one 0.05 0.1
run_one 0.1 0.15
run_one 0.15 0.2
run_one 0.2 0.25
run_one 0.25 0.3
run_one 0.3 0.35
run_one 0.35 0.4
run_one 0.4 0.45
run_one 0.45 0.5

echo ">>> Done baseline substitue instance-only [0,0.5]. Output: ${OUT_DIR}"
