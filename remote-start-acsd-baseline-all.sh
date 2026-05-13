set -euo pipefail
cd /home/chenlin/krona/anchor-nav
mkdir -p output_logs/tmux
chmod +x scripts/run_acsd_all_0.05_0.1.sh scripts/run_baseline_all_0.05_0.1.sh

TS="$(date +%Y%m%d-%H%M%S)"
ACSD_SESSION="acsd-all-00501-vlm-anchors-object-only-${TS}"
BASE_SESSION="baseline-all-00501-live-metrics-${TS}"
ACSD_LOG="output_logs/tmux/${ACSD_SESSION}.log"
BASE_LOG="output_logs/tmux/${BASE_SESSION}.log"

tmux new-session -d -s "${BASE_SESSION}" "bash -lc 'cd /home/chenlin/krona/anchor-nav && bash scripts/run_baseline_all_0.05_0.1.sh detailed tmux-all-${TS}; rc=\$?; echo BASELINE_ALL_EXIT=\$rc; exit \$rc' > ${BASE_LOG} 2>&1"
tmux new-session -d -s "${ACSD_SESSION}" "bash -lc 'cd /home/chenlin/krona/anchor-nav && bash scripts/run_acsd_all_0.05_0.1.sh detailed vlm-anchors-object-only-tmux-all-${TS}; rc=\$?; echo ACSD_ALL_EXIT=\$rc; exit \$rc' > ${ACSD_LOG} 2>&1"

echo "ACSD_SESSION=${ACSD_SESSION}"
echo "ACSD_LOG=${ACSD_LOG}"
echo "BASELINE_SESSION=${BASE_SESSION}"
echo "BASELINE_LOG=${BASE_LOG}"
