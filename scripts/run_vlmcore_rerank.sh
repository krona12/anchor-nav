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

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

# Usage:
#   bash scripts/run_vlmcore_rerank.sh [detailed|concise] [optional_tag]
#
# Anchor 物体目标 VLM 重排序（anchor_nav.rerank），不含 VLMDecisionCorrector。
# 默认仅对 instance 级别在 object-commit 时调用 rerank；可用环境变量改参：
#   RERANK_LEVELS RERANK_TOP_K RERANK_MIN_RGB_CAND
#   VLM_BASE_URL VLM_MODEL
# 可选：RERANK_IO_DIR 指定每次 VLM 调用的图片/文本落盘根目录（磁盘占用大，默认不设）。
# 图片调试落盘由 refine1 的 --rerank_save_image_log 控制，默认 0（不写 rerank_vlm_io/）；
# 需要时：export RERANK_SAVE_IMAGE_LOG=1 或传 --rerank_save_image_log 1
# 注意：若曾多进程并行，勿共用一个 RERANK_LOG_JSONL；本脚本单线时每分片仍有独立 jsonl。
#
# 数据集分片与 run_vlmcore_refine1.sh 相同（0.05 步长共 10 片），但此处**单线顺序**跑分片，
# 避免同时两个进程打爆本机 vLLM。
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

OUT_DIR="output_logs/anchor/rerank/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

RERANK_LEVELS="${RERANK_LEVELS:-instance}"
RERANK_TOP_K="${RERANK_TOP_K:-8}"
RERANK_MIN_RGB_CAND="${RERANK_MIN_RGB_CAND:-2}"
RERANK_SAVE_IMAGE_LOG="${RERANK_SAVE_IMAGE_LOG:-0}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLM_MODEL="${VLM_MODEL:-gpt-4o-mini}"

echo ">>> Anchor rerank run tag: ${RUN_TAG}"
echo ">>> Description mode: ${DESC_MODE}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> Rerank: levels=${RERANK_LEVELS} top_k=${RERANK_TOP_K} min_rgb_cand=${RERANK_MIN_RGB_CAND} save_image_log=${RERANK_SAVE_IMAGE_LOG}"
echo ">>> VLM: base_url=${VLM_BASE_URL} model=${VLM_MODEL}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "pwd=$(pwd)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "argv=$*"
  echo "RERANK_LEVELS=${RERANK_LEVELS}"
  echo "RERANK_TOP_K=${RERANK_TOP_K}"
  echo "RERANK_MIN_RGB_CAND=${RERANK_MIN_RGB_CAND}"
  echo "RERANK_SAVE_IMAGE_LOG=${RERANK_SAVE_IMAGE_LOG}"
  echo "VLM_BASE_URL=${VLM_BASE_URL}"
  echo "VLM_MODEL=${VLM_MODEL}"
  echo "RERANK_IO_DIR=${RERANK_IO_DIR:-}"
  echo "slice_step=0.05"
  echo "slice_range=0.0-0.5"
  echo "num_shards_total=10"
  echo "schedule=sequential_10_shards_single_pipeline"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_vlmcore_rerank.sh.snapshot"
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
    OUT_JSON="${OUT_DIR}/refhm3d_seq_rerank_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_rerank_refine1_${START_RATIO}_${END_RATIO}.json"
  fi

  local MAX_RETRY=3
  local COUNT=0

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    echo ">>> Found empty result file, removing: ${OUT_JSON}"
    rm -f "${OUT_JSON}"
  fi

  while true; do
    echo ">>> Starting anchor rerank refine1 ${DESC_MODE} ${START_RATIO}-${END_RATIO} (attempt $((COUNT+1)))..."
    python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-rerank-refine1.py \
      --start_ratio "${START_RATIO}" \
      --end_ratio "${END_RATIO}" \
      ${DESC_FLAG} \
      --rerank_levels "${RERANK_LEVELS}" \
      --rerank_top_k "${RERANK_TOP_K}" \
      --rerank_min_rgb_cand "${RERANK_MIN_RGB_CAND}" \
      --rerank_save_image_log "${RERANK_SAVE_IMAGE_LOG}" \
      --vlm_base_url "${VLM_BASE_URL}" \
      --vlm_model "${VLM_MODEL}" \
      --navigation_data_path "${NAVIGATION_DATA_PATH}" \
      --output_log_dir "${OUT_DIR}"
    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
      echo ">>> ${START_RATIO}-${END_RATIO} finished successfully!"
      break
    fi

    COUNT=$((COUNT+1))
    if [ ${COUNT} -ge ${MAX_RETRY} ]; then
      echo ">>> ${START_RATIO}-${END_RATIO} failed after ${MAX_RETRY} attempts."
      break
    fi

    echo ">>> ${START_RATIO}-${END_RATIO} crashed (exit ${EXIT_CODE}), retrying..."
    sleep 3
  done
}

echo ">>> Batch 1/5: ratios [0, 0.1) — sequential (single pipeline)"
run_one 0.0 0.05
run_one 0.05 0.1

echo ">>> Batch 2/5: ratios [0.1, 0.2) — sequential"
run_one 0.1 0.15
run_one 0.15 0.2

echo ">>> Batch 3/5: ratios [0.2, 0.3) — sequential"
run_one 0.2 0.25
run_one 0.25 0.3

echo ">>> Batch 4/5: ratios [0.3, 0.4) — sequential"
run_one 0.3 0.35
run_one 0.35 0.4

echo ">>> Batch 5/5: ratios [0.4, 0.5] — sequential"
run_one 0.4 0.45
run_one 0.45 0.5

echo ">>> All anchor rerank refine1 splits done (10 shards sequential). Output: ${OUT_DIR}"
