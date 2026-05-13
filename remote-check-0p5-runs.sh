#!/usr/bin/env bash
set -euo pipefail

cd /home/chenlin/krona/anchor-nav

BASE_ROOT="output_logs/baseline_all_0.05_0.5"
MILE_ROOT="output_logs/anchor/mile_all_0.05_0.5"

echo "=== tmux ==="
tmux list-sessions 2>/dev/null | grep -E 'baseline-all-0_05-0_5|mile-all-0_05-0_5' || true
BASE_SESSION="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E '^baseline-all-0_05-0_5' | sort | tail -1 || true)"
MILE_SESSION="$(tmux list-sessions -F '#S' 2>/dev/null | grep -E '^mile-all-0_05-0_5' | sort | tail -1 || true)"
echo "baseline_session=${BASE_SESSION}"
echo "mile_session=${MILE_SESSION}"

BASE_DIR="$(
  find "${BASE_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | grep '0p5-baseline-rerun1' \
    | sort \
    | tail -1 || true
)"
if [ -z "${BASE_DIR}" ]; then
  BASE_DIR="$(find "${BASE_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
fi

MILE_DIR="$(
  find "${MILE_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | grep -E 'msgnav-vvd-parity-height15-firstmax-0p5|all-object-vvd-metricfix-0p5' \
    | sort \
    | tail -1 || true
)"
if [ -z "${MILE_DIR}" ]; then
  MILE_DIR="$(find "${MILE_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
fi

echo "=== dirs ==="
echo "baseline_dir=${BASE_DIR}"
echo "mile_dir=${MILE_DIR}"

BASE_LOG=""
MILE_LOG=""
if [ -n "${BASE_DIR}" ]; then
  BASE_LOG="$(find "${BASE_DIR}" -maxdepth 1 -type f -name 'refhm3d-nav-sequence-baseline-*.log' | sort | tail -1 || true)"
fi
if [ -n "${MILE_DIR}" ]; then
  MILE_LOG="$(find "${MILE_DIR}" -maxdepth 1 -type f -name 'refhm3d-nav-sequence-analyze-anchor-mile-refine1-*.log' | sort | tail -1 || true)"
fi

echo "baseline_log=${BASE_LOG}"
echo "mile_log=${MILE_LOG}"

echo "=== baseline pane tail ==="
if [ -n "${BASE_SESSION}" ]; then
  tmux capture-pane -pt "${BASE_SESSION}" -S -35 2>/dev/null || true
else
  echo "no baseline tmux session"
fi

echo "=== mile pane tail ==="
if [ -n "${MILE_SESSION}" ]; then
  tmux capture-pane -pt "${MILE_SESSION}" -S -35 2>/dev/null || true
else
  echo "no mile tmux session"
fi

echo "=== parsed metrics ==="
python3 - "$BASE_LOG" "$MILE_LOG" <<'PY'
import re
import sys
from pathlib import Path

base_log = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
mile_log = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None

base_re = re.compile(
    r"\[Metrics\]\s+sequence count:\s*(\d+),\s*invalid_count:\s*(\d+),\s*"
    r"follower_error_count:\s*(\d+),\s*avg_sr:\s*([0-9.]+),\s*avg_spl:\s*([0-9.]+)"
)
mile_re = re.compile(
    r"\[Metrics\]\s+sequence count=(\d+),\s*avg_sr=([0-9.]+),\s*avg_spl=([0-9.]+)"
)
status_re = re.compile(r"\[mile-refine1\]\[module-status-counts\]\s*(\{.*\})")

def read_text(path):
    if path and path.exists():
        return path.read_text(errors="replace")
    return ""

base_text = read_text(base_log)
mile_text = read_text(mile_log)

base_metrics = []
for m in base_re.finditer(base_text):
    base_metrics.append({
        "count": int(m.group(1)),
        "invalid_count": int(m.group(2)),
        "follower_error_count": int(m.group(3)),
        "avg_sr": float(m.group(4)),
        "avg_spl": float(m.group(5)),
    })

mile_metrics = []
for m in mile_re.finditer(mile_text):
    mile_metrics.append({
        "count": int(m.group(1)),
        "avg_sr": float(m.group(2)),
        "avg_spl": float(m.group(3)),
    })

latest_status = None
for m in status_re.finditer(mile_text):
    latest_status = m.group(1)

traceback = "Traceback" in mile_text[-20000:] or "Traceback" in base_text[-20000:]
finished = "[MileRefine1] run finished" in mile_text or ">>> Done mile all-task [0.05,0.5]" in mile_text

print(f"baseline_metrics_count={len(base_metrics)}")
print(f"mile_metrics_count={len(mile_metrics)}")
print(f"mile_finished={finished}")
print(f"traceback_recent={traceback}")
print(f"mile_status_latest={latest_status}")

if mile_metrics:
    mile = mile_metrics[-1]
    print(f"mile_latest count={mile['count']} avg_sr={mile['avg_sr']:.6f} avg_spl={mile['avg_spl']:.6f}")
else:
    mile = None
    print("mile_latest none")

if base_metrics:
    base_latest = base_metrics[-1]
    print(
        f"baseline_latest count={base_latest['count']} invalid_count={base_latest['invalid_count']} "
        f"follower_error_count={base_latest['follower_error_count']} "
        f"avg_sr={base_latest['avg_sr']:.6f} avg_spl={base_latest['avg_spl']:.6f}"
    )
else:
    print("baseline_latest none")

if mile and base_metrics:
    exact = [b for b in base_metrics if b["count"] == mile["count"]]
    if not exact:
        available = ",".join(str(b["count"]) for b in base_metrics[-8:])
        print(
            f"aligned mode=exact_missing mile_count={mile['count']} "
            f"baseline_recent_counts=[{available}] comparable=False strict_mile_below_baseline=unknown"
        )
    else:
        aligned = exact[-1]
        sr_delta = mile["avg_sr"] - aligned["avg_sr"]
        spl_delta = mile["avg_spl"] - aligned["avg_spl"]
        strict_bad = (sr_delta < 0.0) or (spl_delta < 0.0)
        print(
            f"aligned mode=exact baseline_count={aligned['count']} "
            f"baseline_sr={aligned['avg_sr']:.6f} baseline_spl={aligned['avg_spl']:.6f} "
            f"delta_sr={sr_delta:+.6f} delta_spl={spl_delta:+.6f} "
            f"comparable=True strict_mile_below_baseline={strict_bad}"
        )
PY

echo "=== error markers ==="
if [ -n "${MILE_LOG}" ]; then
  grep -nE 'Traceback|mile_error|follow-error|GreedyFollowerError' "${MILE_LOG}" | tail -40 || true
fi
