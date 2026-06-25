#!/bin/bash
export MODEL_NAME="gpt-4o-mini"
export MODEL_SLUG="gpt-4o-mini"
export CUDA_DEVICE="0"
export TMUX_SESSION="sam-vista2mqsc-modelcmp-00_005-gpt4omini-cuda0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_vista2mqsc_sam_0.0_0.05_model.sh" "$@"
