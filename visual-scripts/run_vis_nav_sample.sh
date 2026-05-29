#!/usr/bin/env bash
# run_vis_nav_sample.sh - Launch visualization for one navigation episode.
#
# Usage:
#   bash visual-scripts/run_vis_nav_sample.sh \
#       --scene_name 00802-wcojb4TFT35 \
#       --episode_id 12 \
#       [--task_ids 0,2] \
#       [--output_vis_dir visual-output/my_vis]
#
# All extra arguments are forwarded to vis_nav_sample.py.
# Run from the repository root (anchor-nav/).

set -eo pipefail

# ── Activate conda environment ──────────────────────────────────────────────
_SAVED_ARGV=("$@")
set --
source "/opt/conda/bin/activate"
set +u
conda activate mtu3d
set -u
set -- "${_SAVED_ARGV[@]}"
unset _SAVED_ARGV

# ── Environment variables ────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

echo ">>> vis_nav_sample  CUDA=${CUDA_VISIBLE_DEVICES}  args: $*"
python3 visual-scripts/vis_nav_sample.py "$@"
echo ">>> Done."
