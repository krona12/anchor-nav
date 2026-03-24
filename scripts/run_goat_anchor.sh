#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[AnchorNav] ROOT_DIR=${ROOT_DIR}"
echo "[AnchorNav] Output => output_dirs/goat-anchor-test.json"
python hm3d-online/goat-nav-anchor.py

