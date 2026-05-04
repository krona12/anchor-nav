#!/bin/bash
# Baseline instance-only：按 episode 序列比例切片 [0.05,0.5]，步长 0.05，共 9 段 → 9 个 refhm3d_seq_*.json
# 用法：bash scripts/run_baseline_instance_0.05_0.5_sweep.sh [detailed|concise] [optional_tag]
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
NAVIGATION_DATA_PATH="/home/chenlin/krona/anchor-nav/LangMap_Annotations"

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

OUT_DIR="output_logs/baseline_instance_0.05_0.5/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

echo ">>> Baseline instance-only sweep tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "task_levels=instance"
  echo "slice_bins=9"
  echo "slice_pattern=[0.05,0.1),[0.1,0.15),...,[0.45,0.5] (each width 0.05)"
  echo "num_json_expected=9"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_baseline_instance_0.05_0.5_sweep.sh.snapshot"
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

  echo ">>> Running slice ${START_RATIO} .. ${END_RATIO} -> ${OUT_JSON}"
  python3 hm3d-online/refhm3d-nav-sequence-baseline.py \
    --start_ratio "${START_RATIO}" \
    --end_ratio "${END_RATIO}" \
    ${DESC_FLAG} \
    --task_levels "instance" \
    --navigation_data_path "${NAVIGATION_DATA_PATH}" \
    --output_log_dir "${OUT_DIR}"
}

# 9 段：[0.05,0.1], [0.1,0.15], …, [0.45,0.5]（与 run_baseline_instance_0.05_0.1.sh 中单段逻辑相同，仅循环多次）
for i in $(seq 0 8); do
  read -r START_RATIO END_RATIO < <(python3 -c "i=int('${i}'); s=0.05*(i+1); e=0.05*(i+2); print(s, e)")
  run_one "${START_RATIO}" "${END_RATIO}"
done

echo ">>> Done baseline instance-only sweep [0.05,0.5] in 9 slices. Output: ${OUT_DIR}"
ls -1 "${OUT_DIR}"/refhm3d*.json 2>/dev/null || true
