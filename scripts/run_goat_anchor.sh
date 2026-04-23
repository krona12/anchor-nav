#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_CMD="$(printf '%q ' "$0" "$@")"
RUN_CMD="${RUN_CMD% }"
RUN_TAG="$(date +%Y%m%d-%H%M%S)-goat-anchor"
OUT_DIR="output_dirs/goat-anchor/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

{
  echo "timestamp=$(date +%Y%m%d-%H%M%S)"
  echo "run_tag=${RUN_TAG}"
  echo "script=$0"
  echo "command=${RUN_CMD}"
} > "${OUT_DIR}/run_args.txt"

cp "$0" "${OUT_DIR}/run_goat_anchor.sh.snapshot"
{
  echo "#!/usr/bin/env bash"
  printf '%s\n' "${RUN_CMD}"
} > "${OUT_DIR}/run_command.sh"
chmod +x "${OUT_DIR}/run_command.sh"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[AnchorNav] ROOT_DIR=${ROOT_DIR}"
echo "[AnchorNav] Output => output_dirs/goat-anchor-test.json"
echo "[AnchorNav] Metadata => ${OUT_DIR}"
python hm3d-online/goat-nav-anchor.py

