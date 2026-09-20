#!/usr/bin/env bash
# MTU3D on LangMap. Example: bash scripts/run_template.sh --end_ratio 0.05
# Override paths using the Python entrypoint's CLI options; see --help.
# Set ENTRYPOINT to one of the other three retained refhm3d scripts to reuse.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hm3d-online:$PROJECT_ROOT/hm3d-online/FastSAM${PYTHONPATH:+:$PYTHONPATH}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
export YOLO_VERBOSE=False
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# Conda base may restrict EGL discovery to its Mesa vendor directory.
# Habitat's CUDA renderer needs the system NVIDIA vendor; honor explicit overrides.
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES+x}" && -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

ENTRYPOINT="${ENTRYPOINT:-refhm3d-nav-sequence-baseline.py}"
case "$ENTRYPOINT" in
    refhm3d-nav-sequence.py|refhm3d-nav-sequence-baseline.py|refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py|refhm3d-nav-sequence-analyze-anchor-vista2mqsc-sequence-refine1.py) ;;
    *) echo "Unsupported ENTRYPOINT: $ENTRYPOINT" >&2; exit 2 ;;
esac

if [[ -n "${MTU3D_PYTHON:-}" ]]; then
    runner=("$MTU3D_PYTHON")
else
    runner=(conda run --no-capture-output -n "${MTU3D_ENV:-envname}" python)
fi

exec "${runner[@]}" "$PROJECT_ROOT/hm3d-online/$ENTRYPOINT" \
    --start_ratio "${START_RATIO:-0.0}" \
    --end_ratio "${END_RATIO:-1.0}" \
    "$@"
