#!/bin/bash
export MODEL_NAME="qwen-vl-plus"
export MODEL_SLUG="qwen-vl-plus"
export CUDA_DEVICE="3"
export TMUX_SESSION="sam-vista2mqsc-modelcmp-00_005-qwenvlplus-cuda3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_vista2mqsc_sam_0.0_0.05_model.sh" "$@"
