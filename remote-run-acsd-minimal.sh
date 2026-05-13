#!/usr/bin/env bash
set -euo pipefail
cd /home/chenlin/krona/anchor-nav
source /opt/conda/bin/activate
conda activate mtu3d
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-acsd.py \
  --scene_name 00803-k1cupFYWXJ6 \
  --episode_id 19 \
  --task_id 0 \
  --num_tasks 1 \
  --max_steps 180 \
  --run_tag acsd-vlm-anchors-object-only-minimal \
  --output_root output_process
