#!/bin/bash
set -eo pipefail

# 必须先清空 "$@" 再 source conda，否则第一个参数（detailed|concise）会被 activate 当成环境名。
_SAVED_ARGV=("$@")
set --
source "/home/zhaochaoyang/miniforge3/bin/activate"
set +u
conda activate envnameba
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

# Usage:
#   bash scripts/run_vlmcore_refine1.sh [detailed|concise] [optional_tag]
#
# VLM 纯决策纠偏（与 vlm_decision_corrector 一致）：
#   - 须 --vlm_mode sync才会在当轮合并 VLM（async 不合并，避免错配）。
#   - frontier + found_target 且 confidence >=阈值 → 强制 commit（记忆 top / 位姿兜底）。
#   - object-commit + not found 且 confidence >= 压制阈值 → 改最近 frontier。
# 可选环境变量（批量改参不必改脚本）：
#   VLM_MODE VLM_CONF_THRESHOLD VLM_STRIDE VLM_MIN_DECISION_NUM
#   VLM_BASE_URL VLM_MODEL
#
# 数据集分片：步长 0.05，覆盖 [0, 0.5)，共 10 个分片。
# 调度：顺序 5 批，每批 2 个分片并行（减轻 VLM/仿真同时压力）；
#   批1 [0,0.1)  | 批2 [0.1,0.2) | 批3 [0.2,0.3) | 批4 [0.3,0.4) | 批5 [0.4,0.5]。
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

OUT_DIR="output_logs/anchor/vlmcor/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
export VLM_CORRECTOR_LOG_JSONL="${OUT_DIR}/vlm_corrector_calls.jsonl"
export VLM_CORRECTOR_LOG_RAW_MAX="${VLM_CORRECTOR_LOG_RAW_MAX:-2000}"

VLM_MODE="${VLM_MODE:-sync}"
VLM_CONF_THRESHOLD="${VLM_CONF_THRESHOLD:-0.9}"
VLM_STRIDE="${VLM_STRIDE:-1}"
VLM_MIN_DECISION_NUM="${VLM_MIN_DECISION_NUM:-2}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLM_MODEL="${VLM_MODEL:-Qwen2.5-VL-32B-Instruct}"

echo ">>> VLMCore run tag: ${RUN_TAG}"
echo ">>> Description mode: ${DESC_MODE}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> VLM: mode=${VLM_MODE} conf_threshold=${VLM_CONF_THRESHOLD} stride=${VLM_STRIDE} min_decision_num=${VLM_MIN_DECISION_NUM}"

# Save run metadata for reproducibility
{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "pwd=$(pwd)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "argv=$*"
  echo "VLM_CORRECTOR_LOG_JSONL=${VLM_CORRECTOR_LOG_JSONL}"
  echo "VLM_CORRECTOR_LOG_RAW_MAX=${VLM_CORRECTOR_LOG_RAW_MAX}"
  echo "VLM_MODE=${VLM_MODE}"
  echo "VLM_CONF_THRESHOLD=${VLM_CONF_THRESHOLD}"
  echo "VLM_STRIDE=${VLM_STRIDE}"
  echo "VLM_MIN_DECISION_NUM=${VLM_MIN_DECISION_NUM}"
  echo "VLM_BASE_URL=${VLM_BASE_URL}"
  echo "VLM_MODEL=${VLM_MODEL}"
  echo "slice_step=0.05"
  echo "slice_range=0.0-0.5"
  echo "num_shards_total=10"
  echo "schedule=sequential_5_batches_parallel_2_per_batch"
} > "${OUT_DIR}/run_args.txt"

# Save an exact copy of current launcher script
cp "$0" "${OUT_DIR}/run_vlmcore_refine1.sh.snapshot"

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON
  local DESC_FLAG=""
  if [ "${DESC_MODE}" = "concise" ]; then
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vlmcor_refine1_concisedesc_${START_RATIO}_${END_RATIO}.json"
    DESC_FLAG="--concise_description"
  else
    OUT_JSON="${OUT_DIR}/refhm3d_seq_vlmcor_refine1_${START_RATIO}_${END_RATIO}.json"
  fi

  local MAX_RETRY=3
  local COUNT=0

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    echo ">>> Found empty result file, removing: ${OUT_JSON}"
    rm -f "${OUT_JSON}"
  fi

  while true; do
    echo ">>> Starting vlmcore refine1 ${DESC_MODE} ${START_RATIO}-${END_RATIO} (attempt $((COUNT+1)))..."
    python3 hm3d-online/refhm3d-nav-sequence-analyze-vlmcore-refine1.py \
      --start_ratio "${START_RATIO}" \
      --end_ratio "${END_RATIO}" \
      ${DESC_FLAG} \
      --enable_vlm_corrector \
      --vlm_mode "${VLM_MODE}" \
      --vlm_stride "${VLM_STRIDE}" \
      --vlm_min_decision_num "${VLM_MIN_DECISION_NUM}" \
      --vlm_conf_threshold "${VLM_CONF_THRESHOLD}" \
      --vlm_base_url "${VLM_BASE_URL}" \
      --vlm_model "${VLM_MODEL}" \
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

echo ">>> Batch 1/5: ratios [0, 0.1) — 2 shards parallel"
run_one 0.0 0.05 &
run_one 0.05 0.1 &
wait

echo ">>> Batch 2/5: ratios [0.1, 0.2) — 2 shards parallel"
run_one 0.1 0.15 &
run_one 0.15 0.2 &
wait

echo ">>> Batch 3/5: ratios [0.2, 0.3) — 2 shards parallel"
run_one 0.2 0.25 &
run_one 0.25 0.3 &
wait

echo ">>> Batch 4/5: ratios [0.3, 0.4) — 2 shards parallel"
run_one 0.3 0.35 &
run_one 0.35 0.4 &
wait

echo ">>> Batch 5/5: ratios [0.4, 0.5] — 2 shards parallel"
run_one 0.4 0.45 &
run_one 0.45 0.5 &
wait

echo ">>> All vlmcore refine1 splits done (5 batches × 2 parallel). Output: ${OUT_DIR}"
