#!/bin/bash
set -eo pipefail

# 必须先清空 "$@" 再 source conda：否则 bash 会把 $1 传给被 source 的 activate，
# 导致 `bash scripts/run_baseline.sh concise` 被误当成 `conda activate concise`。
_SAVED_ARGV=("$@")
set --
source "/home/zhaochaoyang/miniforge3/bin/activate"
set +u
conda activate envnameba
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

# Usage:
#   bash scripts/run_baseline.sh [detailed|concise] [optional_user_tag]
#
# 与 refhm3d-nav-sequence-baseline.py 一致：concise 时传 --concise_description，
# 输出文件名为 refhm3d_seq_concisedesc_{start}_{end}.json，否则为 refhm3d_seq_{start}_{end}.json。
#
# 数据集分片：步长 0.1，覆盖 start_ratio∈[0, 0.5) 即 [0,0.1)…[0.4,0.5)，共 5 路并行。
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

OUT_DIR="output_logs/baseline/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> Baseline run tag: ${RUN_TAG}"
echo ">>> Description mode: ${DESC_MODE}"
echo ">>> Output dir: ${OUT_DIR}"

# Save run metadata for reproducibility（与 run_vlmcore_refine1.sh 对齐）
{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "pwd=$(pwd)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "argv=$*"
  echo "python=hm3d-online/refhm3d-nav-sequence-baseline.py"
  echo "json_naming=concise->refhm3d_seq_concisedesc_{start}_{end}.json; detailed->refhm3d_seq_{start}_{end}.json"
  echo "json_fields_note=每条结果含 task_level: object|room|region|instance"
  echo "slice_step=0.1"
  echo "slice_range=0.0-0.5"
  echo "num_parallel_splits=5"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_baseline.sh.snapshot"

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

  local MAX_RETRY=3
  local COUNT=0

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    echo ">>> Found empty result file, removing: ${OUT_JSON}"
    rm -f "${OUT_JSON}"
  fi

  while true; do
    echo ">>> Starting baseline ${DESC_MODE} ${START_RATIO}-${END_RATIO} (attempt $((COUNT+1)))..."
    python3 hm3d-online/refhm3d-nav-sequence-baseline.py \
      --start_ratio "${START_RATIO}" \
      --end_ratio "${END_RATIO}" \
      ${DESC_FLAG} \
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

run_one 0.0 0.1 &
run_one 0.1 0.2 &
run_one 0.2 0.3 &
run_one 0.3 0.4 &
run_one 0.4 0.5 &

wait
echo ">>> All baseline ${DESC_MODE} splits done. Output: ${OUT_DIR}"
