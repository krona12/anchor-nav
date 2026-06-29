#!/bin/bash
set -eo pipefail

LOG_ROOT="${1:?Usage: monitor_vista2mqsc_sam2_tmux.sh <key_log_root>}"
BALANCED_SESSION="${BALANCED_SESSION:-vista2mqsc_sam2_balanced_cuda1}"
QUALITY_SESSION="${QUALITY_SESSION:-vista2mqsc_sam2_quality_cuda2}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
REQUIRED_OK="${REQUIRED_OK:-3}"

balanced_out_dir="$(find output_logs/anchor/vista2mqsc_sam2_balanced_all_0.0_0.2 -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)"
quality_out_dir="$(find output_logs/anchor/vista2mqsc_sam2_quality_all_0.0_0.2 -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)"
monitor_log="${LOG_ROOT}/monitor.log"

check_one () {
  local name="$1"
  local session="$2"
  local out_dir="$3"
  local json_path="${out_dir}/refhm3d_seq_vista2mqsc_refine1_0.0_0.2.json"
  local live_log="${out_dir}/vista2mqsc_live_metrics_0.0_0.2.log"

  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "${name}: session_missing ${session}"
    return 1
  fi
  if [ -n "${out_dir}" ] && find "${out_dir}" -type f \( -name '*.log' -o -name 'run_stdout_stderr.log' \) -print0 \
      | xargs -0 grep -E -i -m 1 'Traceback|OutOfMemory|CUDA out of memory|RuntimeError|nvidia-smi failed|Driver/library version mismatch|cannot import name ._C.|Skipping the post-processing step' >/tmp/sam2_monitor_err.$$ 2>/dev/null; then
    echo "${name}: error_found $(cat /tmp/sam2_monitor_err.$$)"
    rm -f /tmp/sam2_monitor_err.$$
    return 1
  fi
  rm -f /tmp/sam2_monitor_err.$$

  local rows="0"
  if [ -f "${json_path}" ]; then
    rows="$(python3 - "${json_path}" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    if isinstance(data, list):
        print(len(data))
    elif isinstance(data, dict):
        print(len(data.get("sequence", data.get("results", []))))
    else:
        print(0)
except Exception:
    print(0)
PY
)"
  fi
  local live_lines="0"
  if [ -f "${live_log}" ]; then
    live_lines="$(wc -l < "${live_log}")"
  fi
  echo "${name}: ok session=${session} rows=${rows} live_lines=${live_lines} out_dir=${out_dir}"
  return 0
}

ok_count=0
iteration=0
while [ "${ok_count}" -lt "${REQUIRED_OK}" ]; do
  iteration=$((iteration + 1))
  iteration_log="$(mktemp)"
  balanced_ok=1
  quality_ok=1
  {
    echo "=== monitor iteration ${iteration} $(date '+%Y-%m-%d %H:%M:%S') ==="
    if check_one balanced "${BALANCED_SESSION}" "${balanced_out_dir}"; then
      balanced_ok=0
    else
      balanced_ok=1
    fi
    if check_one quality "${QUALITY_SESSION}" "${quality_out_dir}"; then
      quality_ok=0
    else
      quality_ok=1
    fi
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  } > "${iteration_log}"
  cat "${iteration_log}" | tee -a "${monitor_log}"
  rm -f "${iteration_log}"

  if [ "${balanced_ok}" -eq 0 ] && [ "${quality_ok}" -eq 0 ]; then
    ok_count=$((ok_count + 1))
  else
    ok_count=0
  fi
  if [ "${ok_count}" -lt "${REQUIRED_OK}" ]; then
    sleep "${INTERVAL_SEC}"
  fi
done

echo "monitor completed after ${REQUIRED_OK} consecutive ok checks" | tee -a "${monitor_log}"
