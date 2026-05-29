#!/usr/bin/env bash
set -uo pipefail

SESSION="${1:-bs_sam_0_0_2_cuda3_580126}"
RUN_DIR="${2:-}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
HEALTHY_REQUIRED="${HEALTHY_REQUIRED:-2}"
STALE_SECONDS="${STALE_SECONDS:-1800}"
ACTIVE_CPU_PERCENT="${ACTIVE_CPU_PERCENT:-20}"
MAX_ACTIVE_STALE_SECONDS="${MAX_ACTIVE_STALE_SECONDS:-7200}"
MONITOR_LOG="${MONITOR_LOG:-output_logs/monitors/${SESSION}-$(date +%Y%m%d-%H%M%S).log}"

mkdir -p "$(dirname "${MONITOR_LOG}")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MONITOR_LOG}"
}

find_run_dir() {
  if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
    printf '%s\n' "${RUN_DIR}"
    return 0
  fi

  find output_logs/baseline_sam_all_0_0.2 -maxdepth 1 -type d -name '*cuda3-580126*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
}

find_latest_log() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name 'refhm3d-nav-sequence-baseline-*.log' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
}

descendants_of() {
  local root="$1"
  local child
  for child in $(pgrep -P "${root}" 2>/dev/null || true); do
    printf '%s\n' "${child}"
    descendants_of "${child}"
  done
}

find_target_pid() {
  local pane_pid="$1"
  local pid
  for pid in "${pane_pid}" $(descendants_of "${pane_pid}"); do
    if ps -p "${pid}" -o args= 2>/dev/null | grep -q 'refhm3d-nav-sequence-baseline-sam.py'; then
      printf '%s\n' "${pid}"
      return 0
    fi
  done
}

bad_log_patterns() {
  local file="$1"
  tail -n 500 "${file}" 2>/dev/null | grep -E \
    'Traceback \(most recent call last\)|CUDA out of memory|CUBLAS_STATUS|CUDNN_STATUS|RuntimeError:|WindowlessContext|EGL.*(failed|unable)|Driver/library version mismatch|Segmentation fault|Killed|Error executing job|ModuleNotFoundError|ImportError'
}

check_once() {
  local run_dir latest_log pane_pid target_pid now log_mtime log_age pcpu stat output_json

  run_dir="$(find_run_dir)"
  if [ -z "${run_dir}" ] || [ ! -d "${run_dir}" ]; then
    log "FAIL run_dir not found"
    return 1
  fi

  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    log "FAIL tmux session not found: ${SESSION}"
    return 1
  fi

  pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' 2>/dev/null | head -n 1)"
  if [ -z "${pane_pid}" ]; then
    log "FAIL no pane pid for session: ${SESSION}"
    return 1
  fi

  target_pid="$(find_target_pid "${pane_pid}")"
  if [ -z "${target_pid}" ]; then
    log "FAIL target python process not found under pane pid ${pane_pid}"
    return 1
  fi

  latest_log="$(find_latest_log "${run_dir}")"
  if [ -z "${latest_log}" ] || [ ! -f "${latest_log}" ]; then
    log "FAIL baseline log not found in ${run_dir}"
    return 1
  fi

  if bad_log_patterns "${latest_log}" > "${MONITOR_LOG}.last_error"; then
    log "FAIL error pattern found in ${latest_log}"
    sed 's/^/[error] /' "${MONITOR_LOG}.last_error" | tee -a "${MONITOR_LOG}"
    return 1
  fi

  now="$(date +%s)"
  log_mtime="$(stat -c %Y "${latest_log}")"
  log_age="$((now - log_mtime))"
  pcpu="$(ps -p "${target_pid}" -o pcpu= 2>/dev/null | awk '{printf "%.0f", $1}')"
  stat="$(ps -p "${target_pid}" -o stat= 2>/dev/null | awk '{print $1}')"
  output_json="$(find "${run_dir}" -maxdepth 1 -type f -name '*.json' -printf '%f ' 2>/dev/null)"

  if [ "${log_age}" -gt "${MAX_ACTIVE_STALE_SECONDS}" ]; then
    log "FAIL log stale for ${log_age}s > ${MAX_ACTIVE_STALE_SECONDS}s: ${latest_log}"
    return 1
  fi

  if [ "${log_age}" -gt "${STALE_SECONDS}" ] && [ "${pcpu}" -lt "${ACTIVE_CPU_PERCENT}" ]; then
    log "FAIL log stale for ${log_age}s and process CPU is only ${pcpu}%"
    return 1
  fi

  log "OK session=${SESSION} pid=${target_pid} stat=${stat} cpu=${pcpu}% log_age=${log_age}s run_dir=${run_dir} json=${output_json:-none}"
  return 0
}

log "monitor started session=${SESSION} interval=${INTERVAL_SECONDS}s healthy_required=${HEALTHY_REQUIRED}"
healthy_count=0
while [ "${healthy_count}" -lt "${HEALTHY_REQUIRED}" ]; do
  if check_once; then
    healthy_count="$((healthy_count + 1))"
    log "healthy_count=${healthy_count}/${HEALTHY_REQUIRED}"
  else
    log "monitor detected unhealthy state; leaving target session untouched"
    exit 2
  fi

  if [ "${healthy_count}" -lt "${HEALTHY_REQUIRED}" ]; then
    sleep "${INTERVAL_SECONDS}"
  fi
done

log "monitor finished after ${HEALTHY_REQUIRED} consecutive healthy checks; target session left running"
