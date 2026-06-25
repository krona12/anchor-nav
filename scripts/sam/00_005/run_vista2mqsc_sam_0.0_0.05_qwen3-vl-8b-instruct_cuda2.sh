#!/bin/bash
export MODEL_NAME="qwen3-vl-8b-instruct"
export MODEL_SLUG="qwen3-vl-8b-instruct"
export CUDA_DEVICE="2"
export TMUX_SESSION="sam-vista2mqsc-modelcmp-00_005-qwen3vl8b-cuda2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_vista2mqsc_sam_0.0_0.05_model.sh" "$@"
