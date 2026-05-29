#!/usr/bin/env bash
# Run the decision-level visualization for:
#   scene 00844-q5QZSEeHe5g
#   episodes_by_instance_level episode_id 122
#   instance_id armchair_906
#
# Extra arguments are forwarded to vis_nav_sample.py.

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
export PYTHONUNBUFFERED=1
export VIS_NAV_DEBUG_IMPORTS="${VIS_NAV_DEBUG_IMPORTS:-0}"
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
LOCAL_NVIDIA_ROOT="${LOCAL_NVIDIA_ROOT:-/home/chenlin/krona/anchor-nav/local_nvidia_580_126_09/root}"
LOCAL_NVIDIA_LIB="${LOCAL_NVIDIA_ROOT}/usr/lib/x86_64-linux-gnu"
LOCAL_NVIDIA_EGL_JSON="${LOCAL_NVIDIA_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [ -d "${LOCAL_NVIDIA_LIB}" ] && [ -f "${LOCAL_NVIDIA_EGL_JSON}" ]; then
  export LD_LIBRARY_PATH="${LOCAL_NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${LOCAL_NVIDIA_EGL_JSON}"
  export PATH="${LOCAL_NVIDIA_ROOT}/usr/bin${PATH:+:${PATH}}"
  export NVIDIA_SMI_BIN="${LOCAL_NVIDIA_ROOT}/usr/bin/nvidia-smi"
  echo ">>> Using local NVIDIA userspace: ${LOCAL_NVIDIA_LIB}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
mkdir -p "${MPLCONFIGDIR}"

echo ">>> visualizing instance armchair_906 on CUDA=${CUDA_VISIBLE_DEVICES}"
python3 -u visual-scripts/vis_nav_sample.py \
  --scene_name 00844-q5QZSEeHe5g \
  --navigation_type instance \
  --episode_id 122 \
  --instance_id armchair_906 \
  --output_vis_dir visual-output/vis_armchair_906 \
  "$@"
echo ">>> Done."
