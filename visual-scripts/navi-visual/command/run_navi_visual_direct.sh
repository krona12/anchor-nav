#!/usr/bin/env bash
# Direct launcher for visual-scripts/navi-visual/code/navigate_to_target.py.
#
# Matches the project batch scripts: activates the mtu3d conda env and sets the
# same Habitat runtime variables before launching the direct target runner.
#
# Usage:
#   bash visual-scripts/navi-visual/command/run_navi_visual_direct.sh [detailed|concise] [optional_tag] [-- extra python args]
#
# Defaults target the current armchair_906 visualization case:
#   scene=00844-q5QZSEeHe5g navigation_type=instance episode=122 instance=armchair_906
#
# Override defaults with env vars, for example:
#   SCENE_NAME=00802-wcojb4TFT35 EPISODE_ID=12 NAVIGATION_TYPE=sequence TASK_ID=0 \
#     bash visual-scripts/navi-visual/command/run_navi_visual_direct.sh detailed test

set -eo pipefail

_SAVED_ARGV=("$@")
RUN_CMD="$(printf '%q ' "$0" "${_SAVED_ARGV[@]}")"
RUN_CMD="${RUN_CMD% }"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [ "${NAVI_VISUAL_SKIP_ENV_SETUP:-0}" != "1" ]; then
  _NAVI_VISUAL_ARGV=("$@")
  set --
  source "/opt/conda/bin/activate"
  set +u
  conda activate "${CONDA_ENV_NAME:-mtu3d}"
  set -u
  set -- "${_NAVI_VISUAL_ARGV[@]}"
  unset _NAVI_VISUAL_ARGV
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}"
USE_LOCAL_NVIDIA_580="${USE_LOCAL_NVIDIA_580:-0}"
LOCAL_NVIDIA_ROOT="${LOCAL_NVIDIA_ROOT:-${REPO_ROOT}/local_nvidia_580_126_09/root}"
LOCAL_NVIDIA_LIB="${LOCAL_NVIDIA_ROOT}/usr/lib/x86_64-linux-gnu"
LOCAL_NVIDIA_EGL_JSON="${LOCAL_NVIDIA_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if [ "${USE_LOCAL_NVIDIA_580}" = "1" ] && [ -d "${LOCAL_NVIDIA_LIB}" ] && [ -f "${LOCAL_NVIDIA_EGL_JSON}" ]; then
  export LD_LIBRARY_PATH="${LOCAL_NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${LOCAL_NVIDIA_EGL_JSON}"
  export PATH="${LOCAL_NVIDIA_ROOT}/usr/bin${PATH:+:${PATH}}"
  export NVIDIA_SMI_BIN="${LOCAL_NVIDIA_ROOT}/usr/bin/nvidia-smi"
  echo ">>> Using local NVIDIA userspace: ${LOCAL_NVIDIA_LIB}"
fi
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

DESC_MODE="detailed"
if [ "${1:-}" = "detailed" ] || [ "${1:-}" = "concise" ]; then
  DESC_MODE="$1"
  shift
elif [ -n "${1:-}" ] && [ "${1:0:2}" != "--" ]; then
  echo "Invalid first arg: ${1}. Must be 'detailed' or 'concise'."
  exit 1
fi

USER_TAG=""
if [ "$#" -gt 0 ] && [ "${1:-}" != "--" ]; then
  USER_TAG="$1"
  shift
fi
if [ "${1:-}" = "--" ]; then
  shift
fi
EXTRA_ARGS=("$@")

TS="$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${TS}-${DESC_MODE}"
if [ -n "${USER_TAG}" ]; then
  RUN_TAG="${RUN_TAG}-${USER_TAG}"
fi

SCENE_NAME="${SCENE_NAME:-00844-q5QZSEeHe5g}"
NAVIGATION_TYPE="${NAVIGATION_TYPE:-instance}"
EPISODE_ID="${EPISODE_ID:-122}"
INSTANCE_ID="${INSTANCE_ID:-armchair_906}"
TASK_ID="${TASK_ID:-0}"

NAVIGATION_DATA_PATH="${NAVIGATION_DATA_PATH:-${REPO_ROOT}/LangMap_Annotations}"
HM3D_DATA_BASE_PATH="${HM3D_DATA_BASE_PATH:-${REPO_ROOT}/datascene}"
PQ3D_STAGE1_PATH="${PQ3D_STAGE1_PATH:-${REPO_ROOT}/checkpoint/stage1-pretrain-all}"
PQ3D_STAGE2_PATH="${PQ3D_STAGE2_PATH:-${REPO_ROOT}/checkpoint/stage2-fine-tune-goat}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/visual-scripts/navi-visual/logs}"
OUT_DIR="${OUT_DIR:-${LOG_ROOT}/direct/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Headless direct run: no GUI window. LIVE_DIR is still passed through so the
# underlying navigator can refresh live RGB/topdown images when available.
HEADLESS="${HEADLESS:-1}"
LIVE_DIR="${LIVE_DIR:-${LOG_ROOT}/live}"

MAX_ROUNDS="${MAX_ROUNDS:-4}"
SEGMENT_ADVANCE_M="${SEGMENT_ADVANCE_M:-1.0}"
ARRIVE_THRESH_M="${ARRIVE_THRESH_M:-0.7}"
SEQUENCE_TASK_COUNT="${SEQUENCE_TASK_COUNT:-1}"

mkdir -p "${OUT_DIR}"

echo ">>> Navi visual direct run tag: ${RUN_TAG}"
echo ">>> Output dir: ${OUT_DIR}"
echo ">>> Headless: ${HEADLESS} (direct target runner, no GUI window)"
echo ">>> Live view: ${LIVE_DIR}/rgb.png  ${LIVE_DIR}/topdown.png"
echo ">>> Scene: ${SCENE_NAME}"
echo ">>> Navigation: ${NAVIGATION_TYPE} episode=${EPISODE_ID} task=${TASK_ID} instance=${INSTANCE_ID}"
echo ">>> Direct params: max_rounds=${MAX_ROUNDS} segment_advance_m=${SEGMENT_ADVANCE_M} arrive_thresh_m=${ARRIVE_THRESH_M}"
echo ">>> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ">>> Python: ${PYTHON_BIN} ($(command -v "${PYTHON_BIN}"))"

{
  echo "timestamp=${TS}"
  echo "run_tag=${RUN_TAG}"
  echo "desc_mode=${DESC_MODE}"
  echo "user_tag=${USER_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
  echo "python=visual-scripts/navi-visual/code/navigate_to_target.py"
  echo "scene_name=${SCENE_NAME}"
  echo "navigation_type=${NAVIGATION_TYPE}"
  echo "episode_id=${EPISODE_ID}"
  echo "instance_id=${INSTANCE_ID}"
  echo "task_id=${TASK_ID}"
  echo "navigation_data_path=${NAVIGATION_DATA_PATH}"
  echo "hm3d_data_base_path=${HM3D_DATA_BASE_PATH}"
  echo "pq3d_stage1_path=${PQ3D_STAGE1_PATH}"
  echo "pq3d_stage2_path=${PQ3D_STAGE2_PATH}"
  echo "log_root=${LOG_ROOT}"
  echo "output_dir=${OUT_DIR}"
  echo "headless=${HEADLESS}"
  echo "live_dir=${LIVE_DIR}"
  echo "max_rounds=${MAX_ROUNDS}"
  echo "segment_advance_m=${SEGMENT_ADVANCE_M}"
  echo "arrive_thresh_m=${ARRIVE_THRESH_M}"
  echo "sequence_task_count=${SEQUENCE_TASK_COUNT}"
  echo "use_local_nvidia_580=${USE_LOCAL_NVIDIA_580}"
  echo "local_nvidia_root=${LOCAL_NVIDIA_ROOT}"
  echo "egl_vendor_library_filenames=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "conda_env=${CONDA_DEFAULT_ENV:-}"
  echo "python_bin=${PYTHON_BIN}"
  echo "python_path=$(command -v "${PYTHON_BIN}")"
  echo "extra_args=${EXTRA_ARGS[*]:-}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_navi_visual_direct.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

PY_ARGS=(
  "visual-scripts/navi-visual/code/navigate_to_target.py"
  --scene_name "${SCENE_NAME}"
  --navigation_type "${NAVIGATION_TYPE}"
  --episode_id "${EPISODE_ID}"
  --task_id "${TASK_ID}"
  --logs_dir "${OUT_DIR}"
  --live_dir "${LIVE_DIR}"
  --max_rounds "${MAX_ROUNDS}"
  --segment_advance_m "${SEGMENT_ADVANCE_M}"
  --arrive_thresh_m "${ARRIVE_THRESH_M}"
  --sequence_task_count "${SEQUENCE_TASK_COUNT}"
)

if [ "${DESC_MODE}" = "concise" ]; then
  PY_ARGS+=(--concise_description)
fi
if [ -n "${INSTANCE_ID}" ]; then
  PY_ARGS+=(--instance_id "${INSTANCE_ID}")
fi

{
  echo "#!/usr/bin/env bash"
  printf '%q ' "${PYTHON_BIN}" -u "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'
} > "${OUT_DIR}/python_command.sh"
chmod +x "${OUT_DIR}/python_command.sh"

"${PYTHON_BIN}" -u "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUT_DIR}/console.log"

echo ">>> Done navi visual direct. Output: ${OUT_DIR}"
