#!/bin/bash
set -eo pipefail

source "/home/zhaochaoyang/miniforge3/bin/activate"
set +u
conda activate envnameba
set -u

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:$PYTHONPATH
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

run_one () {
  local START_RATIO="$1"
  local END_RATIO="$2"
  local OUT_JSON="output_logs/baseline/refhm3d_seq_concisedesc_${START_RATIO}_${END_RATIO}.json"

  local MAX_RETRY=3
  local COUNT=0

  if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
    echo ">>> Found empty result file, removing: ${OUT_JSON}"
    rm -f "${OUT_JSON}"
  fi

  while true; do
    echo ">>> Starting baseline concise ${START_RATIO}-${END_RATIO} (attempt $((COUNT+1)))..."
    python3 hm3d-online/refhm3d-nav-sequence-baseline.py \
      --start_ratio "${START_RATIO}" \
      --end_ratio "${END_RATIO}" \
      --concise_description
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

wait
echo ">>> All concise baseline splits done."