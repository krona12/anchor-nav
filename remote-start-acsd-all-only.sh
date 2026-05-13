set -euo pipefail
cd /home/chenlin/krona/anchor-nav
mkdir -p output_logs/tmux
chmod +x scripts/run_acsd_all_0.05_0.1.sh

TS="$(date +%Y%m%d-%H%M%S)"
ACSD_SESSION="acsd-all-00501-anchor-room-only-${TS}"
ACSD_LOG="output_logs/tmux/${ACSD_SESSION}.log"

tmux new-session -d -s "${ACSD_SESSION}" "bash -lc 'cd /home/chenlin/krona/anchor-nav && CUDA_VISIBLE_DEVICES=1 bash scripts/run_acsd_all_0.05_0.1.sh detailed anchor-room-only-tmux-all-${TS}; rc=\$?; echo ACSD_ALL_EXIT=\$rc; exit \$rc' > ${ACSD_LOG} 2>&1"

echo "ACSD_SESSION=${ACSD_SESSION}"
echo "ACSD_LOG=${ACSD_LOG}"
