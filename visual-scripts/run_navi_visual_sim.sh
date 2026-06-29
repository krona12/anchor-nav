#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SIM_ARGS=("$@")
set --
source /opt/conda/bin/activate
set +u
conda activate mtu3d
set -u
set -- "${SIM_ARGS[@]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one device, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

TOPDOWN_CAM_RES="${TOPDOWN_CAM_RES:-384}"
TOPDOWN_CAM_HEIGHT="${TOPDOWN_CAM_HEIGHT:-2.0}"
TOPDOWN_CAM_HFOV="${TOPDOWN_CAM_HFOV:-90.0}"

exec python -u visual-scripts/navi-visual/code/sim.py \
  --topdown_cam_res "${TOPDOWN_CAM_RES}" \
  --topdown_cam_height "${TOPDOWN_CAM_HEIGHT}" \
  --topdown_cam_hfov "${TOPDOWN_CAM_HFOV}" \
  "$@"
