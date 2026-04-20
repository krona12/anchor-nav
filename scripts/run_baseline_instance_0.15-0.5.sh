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

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/run_baseline_instance_0.15-0.5.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/baseline_instance_0.15_0.5/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> Baseline instance-only run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "task_levels=instance"
  echo "slice_range=0.15-0.5"
  echo "slice_step=0.05"
  echo "num_shards_total=7"
  echo "schedule=parallel_7_shards_single_pipeline"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_baseline_instance_0.15-0.5.sh.snapshot"

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
    --task_levels "instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.15 0.2 &
run_one 0.2 0.25 &
run_one 0.25 0.3 &
run_one 0.3 0.35 &
run_one 0.35 0.4 &
run_one 0.4 0.45 &
run_one 0.45 0.5 &

wait
echo ">>> Done baseline instance-only [0.15,0.5] with 7 parallel shards. Output: ${OUT_DIR}"
