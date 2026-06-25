#!/bin/bash
export MODEL_NAME="gemini-2.5-flash"
export MODEL_SLUG="gemini-2.5-flash"
export CUDA_DEVICE="0"
export TMUX_SESSION="sam-vista2mqsc-modelcmp-00_005-gemini25flash-cuda0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_vista2mqsc_sam_0.0_0.05_model.sh" "$@"
