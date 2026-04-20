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

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

# Usage:
#   bash scripts/run_semantic_instance_0_0.1.sh [detailed|concise] [optional_tag]
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

OUT_DIR="output_logs/anchor/semantic_instance_0_0.05/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

SEMANTIC_TOP_K="${SEMANTIC_TOP_K:-10}"
SEMANTIC_TOP_M="${SEMANTIC_TOP_M:-5}"
SEMANTIC_PROB_TEMPERATURE="${SEMANTIC_PROB_TEMPERATURE:-0.07}"
SEMANTIC_VLM_MODEL="${SEMANTIC_VLM_MODEL:-gpt-4o-mini}"
SEMANTIC_CLIP_DEVICE="${SEMANTIC_CLIP_DEVICE:-cuda}"

echo ">>> Semantic instance-only run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "task_levels=instance"
  echo "semantic_levels=instance"
  echo "SEMANTIC_TOP_K=${SEMANTIC_TOP_K}"
  echo "SEMANTIC_TOP_M=${SEMANTIC_TOP_M}"
  echo "SEMANTIC_PROB_TEMPERATURE=${SEMANTIC_PROB_TEMPERATURE}"
  echo "SEMANTIC_VLM_MODEL=${SEMANTIC_VLM_MODEL}"
  echo "SEMANTIC_CLIP_DEVICE=${SEMANTIC_CLIP_DEVICE}"
  echo "slice_range=0.0-0.05"
  echo "slice_step=0.05"
  echo "num_shards_total=1"
  echo "schedule=single_shard_single_pipeline"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_semantic_instance_0_0.1.sh.snapshot"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON
  local DESC_FLAG=""
  if [ "${DESC_MODE}" = "concise" ]; then
    OUT_JSON="${OUT_DIR}/refhm3d_seq_semantic_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_semantic_refine1_${START_RATIO}_${END_RATIO}.json"
  fi

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    rm -f "${OUT_JSON}"
  fi

  python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-semantic-refine1.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --semantic_levels "instance" \
    --semantic_top_k "${SEMANTIC_TOP_K}" \
    --semantic_top_m "${SEMANTIC_TOP_M}" \
    --semantic_prob_temperature "${SEMANTIC_PROB_TEMPERATURE}" \
    --semantic_vlm_model "${SEMANTIC_VLM_MODEL}" \
    --semantic_clip_device "${SEMANTIC_CLIP_DEVICE}" \
    --output_log_dir "${OUT_DIR}"
}

run_one 0.0 0.05

echo ">>> Done semantic instance-only [0,0.05]. Output: ${OUT_DIR}"
