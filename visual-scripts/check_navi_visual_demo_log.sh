#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHECK_ARGS=("$@")
set --
source /opt/conda/bin/activate
set +u
conda activate mtu3d
set -u
set -- "${CHECK_ARGS[@]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

exec python -u visual-scripts/navi-visual/code/check_demo_log.py "$@"
