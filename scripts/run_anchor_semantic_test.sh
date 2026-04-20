#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_anchor_semantic_test.sh <scene_name> <episode_id> [task_id] [num_tasks]
# 示例：
#   bash scripts/run_anchor_semantic_test.sh 00800-TEEsavR23oF 0 0 1

SCENE_NAME="${1:-}"
EPISODE_ID="${2:-}"
TASK_ID="${3:-0}"
NUM_TASKS="${4:-3}"

if [[ -z "${SCENE_NAME}" || -z "${EPISODE_ID}" ]]; then
  echo "Usage: bash scripts/run_anchor_semantic_test.sh <scene_name> <episode_id> [task_id] [num_tasks]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/opt/conda/etc/profile.d/conda.sh"
  # conda 的 deactivate hook 在 `set -u` 下可能访问未定义变量（如 CONDA_BACKUP_CXX）
  # 这里临时关闭 nounset，激活完成后再恢复。
  set +u
  conda activate mtu3d || true
  set -u
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/hm3d-online:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

RUN_TAG="$(date +%Y%m%d-%H%M%S)-semantic-test"
echo ">>> run_tag=${RUN_TAG}"
echo ">>> scene=${SCENE_NAME} episode=${EPISODE_ID} task_id=${TASK_ID} num_tasks=${NUM_TASKS}"

python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-semantic.py \
  --scene_name "${SCENE_NAME}" \
  --episode_id "${EPISODE_ID}" \
  --task_id "${TASK_ID}" \
  --num_tasks "${NUM_TASKS}" \
  --description_mode detailed \
  --semantic_levels instance \
  --semantic_top_k 10 \
  --semantic_top_m 5 \
  --semantic_prob_temperature 0.07 \
  --semantic_vlm_model "${SEMANTIC_VLM_MODEL:-gpt-4o-mini}" \
  --run_tag "${RUN_TAG}"
