#!/usr/bin/env bash
# run_vis_topdown_rgb.sh - Render photorealistic top-down RGB for a HM3D scene.
#
# Usage (from repo root):
#   bash visual-scripts/run_vis_topdown_rgb.sh --scene_name 00802-wcojb4TFT35
#   bash visual-scripts/run_vis_topdown_rgb.sh --scene_name 00802-wcojb4TFT35 \
#       --resolution 2048 --hfov 60 --output_dir visual-output/topdown_rgb
#
# All extra args are forwarded to vis_topdown_rgb.py.

set -eo pipefail

_SAVED_ARGV=("$@")
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

echo ">>> vis_topdown_rgb  args: $*"
python3 visual-scripts/vis_topdown_rgb.py "$@"
echo ">>> Done."
