#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_navi_visual_direct_check.py" "$@"
