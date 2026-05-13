set -euo pipefail
cd /home/chenlin/krona/anchor-nav

echo "TMUX_MATCH_BEGIN"
tmux ls 2>/dev/null | grep -E "^(acsd-all-00501|baseline-all-00501)-" || true
echo "TMUX_MATCH_END"

latest_session() {
  local prefix="$1"
  tmux list-sessions -F "#{session_created} #{session_name}" 2>/dev/null \
    | awk -v p="$prefix" '$2 ~ "^" p {print $0}' \
    | sort -n \
    | tail -1 \
    | awk '{print $2}'
}

ACSD_SESSION="$(latest_session acsd-all-00501 || true)"
BASE_SESSION="$(latest_session baseline-all-00501 || true)"
ACSD_LOG=""
BASE_LOG=""
if [ -n "$ACSD_SESSION" ]; then ACSD_LOG="output_logs/tmux/${ACSD_SESSION}.log"; fi
if [ -n "$BASE_SESSION" ]; then BASE_LOG="output_logs/tmux/${BASE_SESSION}.log"; fi

session_status() {
  local name="$1"
  if [ -z "$name" ]; then
    echo "SESSION_STATUS name= active=False rc=missing out='no matching session'"
    return
  fi
  if tmux has-session -t "$name" 2>/tmp/tmux_has.err; then
    echo "SESSION_STATUS name=$name active=True rc=0 out=''"
  else
    local out
    out="$(cat /tmp/tmux_has.err 2>/dev/null || true)"
    echo "SESSION_STATUS name=$name active=False rc=1 out=$(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<<"$out")"
  fi
}
session_status "$ACSD_SESSION"
session_status "$BASE_SESSION"
echo "ACSD_LOG=${ACSD_LOG}"
echo "BASELINE_LOG=${BASE_LOG}"

tail_block() {
  local label="$1"
  local path="$2"
  echo "${label}_TAIL_BEGIN"
  if [ -n "$path" ] && [ -f "$path" ]; then
    tail -35 "$path"
  else
    echo "missing_log=$path"
  fi
  echo "${label}_TAIL_END"
}
tail_block ACSD "$ACSD_LOG"
tail_block BASELINE "$BASE_LOG"

echo "ERROR_SCAN_BEGIN"
for path in "$ACSD_LOG" "$BASE_LOG"; do
  if [ -n "$path" ] && [ -f "$path" ]; then
    echo "ERR_LOG=$path"
    grep -En "Traceback|RuntimeError|ACSD_FATAL|decision-error|EXIT=[1-9][0-9]*|ALL_EXIT=[1-9][0-9]*|Killed|CUDA out of memory|No space left" "$path" \
      | grep -Ev "ACSD_ERROR|vlm_anchor_extraction_failed_use_baseline|ACSD anchor extractor VLM call failed" \
      | grep -Ev "ACSD_FOLLOW_ERROR" \
      | tail -30 || true
  fi
done
echo "ERROR_SCAN_END"

extract_out_dir() {
  local log="$1"
  if [ -z "$log" ] || [ ! -f "$log" ]; then return 0; fi
  grep -E ">>> Output dir:" "$log" | tail -1 | sed -E 's/^.*>>> Output dir:[[:space:]]*//'
}

ACSD_OUT="$(extract_out_dir "$ACSD_LOG")"
BASE_OUT="$(extract_out_dir "$BASE_LOG")"

latest_output_dir() {
  local root="$1"
  local metric_file="$2"
  python3 - "$root" "$metric_file" <<'PY'
import os, sys
root, metric = sys.argv[1], sys.argv[2]
if not os.path.isdir(root):
    sys.exit(0)
dirs = [
    os.path.join(root, x)
    for x in os.listdir(root)
    if os.path.isdir(os.path.join(root, x))
]
dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
for d in dirs:
    if os.path.isfile(os.path.join(d, metric)):
        print(d)
        sys.exit(0)
if dirs:
    print(dirs[0])
PY
}

if [ -z "$ACSD_OUT" ]; then
  ACSD_OUT="$(latest_output_dir output_logs/anchor/acsd_all_0.05_0.1 refhm3d_seq_acsd_refine1_0.05_0.1.json)"
fi
if [ -z "$BASE_OUT" ]; then
  BASE_OUT="$(latest_output_dir output_logs/baseline_all_0.05_0.1 refhm3d_seq_0.05_0.1.json)"
fi

echo "ACSD_OUTPUT_DIR=${ACSD_OUT}"
echo "BASELINE_OUTPUT_DIR=${BASE_OUT}"
echo "RUN_CONFIG_BEGIN"
if [ -n "$ACSD_OUT" ] && [ -f "$ACSD_OUT/run_args.txt" ]; then
  echo "ACSD_RUN_ARGS=${ACSD_OUT}/run_args.txt"
  grep -E "^(run_tag|seed|correction_margin|object_correction_min_baseline_score|acsd_top_k|task_levels|slice_range)=" "$ACSD_OUT/run_args.txt" || true
fi
if [ -n "$BASE_OUT" ] && [ -f "$BASE_OUT/run_args.txt" ]; then
  echo "BASELINE_RUN_ARGS=${BASE_OUT}/run_args.txt"
  grep -E "^(run_tag|seed|task_levels|slice_range)=" "$BASE_OUT/run_args.txt" || true
fi
echo "RUN_CONFIG_END"

python3 - <<PY
import json, os, math
acsd_out = ${ACSD_OUT@Q}
base_out = ${BASE_OUT@Q}
acsd_path = os.path.join(acsd_out, "refhm3d_seq_acsd_refine1_0.05_0.1.json") if acsd_out else ""
base_path = os.path.join(base_out, "refhm3d_seq_0.05_0.1.json") if base_out else ""

def load_rows(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for k in ("sequence", "results", "records", "data", "episodes"):
        if isinstance(data.get(k), list):
            return data[k]
    raise RuntimeError(f"unsupported metrics shape for {path}: {list(data)[:20]}")

def key(r):
    return (
        str(r.get("scene_name") or r.get("scene") or ""),
        str(r.get("episode_id") if r.get("episode_id") is not None else r.get("episode")),
        str(r.get("task_id") if r.get("task_id") is not None else r.get("task")),
        str(r.get("task_level") or ""),
    )

def metric(rows, name):
    if not rows:
        return None
    vals = [float(r.get(name, r.get(name.upper(), 0)) or 0) for r in rows]
    return sum(vals) / len(vals)

def summary(path, rows):
    if rows is None:
        return {"path": path, "exists": False, "rows": 0}
    return {
        "path": path,
        "exists": True,
        "rows": len(rows),
        "avg_sr": metric(rows, "sr"),
        "avg_spl": metric(rows, "spl"),
    }

acsd_rows = load_rows(acsd_path)
base_rows = load_rows(base_path)
print("METRICS_BEGIN")
print("ACSD_METRICS", json.dumps(summary(acsd_path, acsd_rows), sort_keys=True))
print("BASELINE_METRICS", json.dumps(summary(base_path, base_rows), sort_keys=True))
if acsd_rows is not None and base_rows is not None:
    b = {}
    dup = []
    for r in base_rows:
        k = key(r)
        if k in b:
            dup.append(k)
        b[k] = r
    matched = [(r, b[key(r)]) for r in acsd_rows if key(r) in b]
    missing = [key(r) for r in acsd_rows if key(r) not in b]
    ar = [x[0] for x in matched]
    br = [x[1] for x in matched]
    a_sr = metric(ar, "sr")
    a_spl = metric(ar, "spl")
    b_sr = metric(br, "sr")
    b_spl = metric(br, "spl")
    print("ALIGNED_METRICS", json.dumps({
        "matched_rows": len(matched),
        "acsd_total_rows": len(acsd_rows),
        "baseline_total_rows": len(base_rows),
        "acsd_aligned_avg_sr": a_sr,
        "acsd_aligned_avg_spl": a_spl,
        "baseline_aligned_avg_sr": b_sr,
        "baseline_aligned_avg_spl": b_spl,
        "acsd_below_aligned_baseline_sr": (a_sr is not None and b_sr is not None and a_sr < b_sr),
        "acsd_below_aligned_baseline_spl": (a_spl is not None and b_spl is not None and a_spl < b_spl),
        "missing_baseline_keys": missing[:10],
        "duplicate_baseline_keys": dup[:10],
        "acsd_path": acsd_path,
        "baseline_path": base_path,
    }, sort_keys=True))
else:
    print("ALIGNED_METRICS", json.dumps({"matched_rows": 0, "reason": "waiting_for_both_metrics_files"}, sort_keys=True))
if acsd_out:
    effect = os.path.join(acsd_out, "refhm3d_seq_acsd_refine1_effectiveness_0.05_0.1.json")
    if os.path.isfile(effect):
        with open(effect, "r", encoding="utf-8") as f:
            e = json.load(f)
        print("ACSD_EFFECTIVENESS", json.dumps({
            "path": effect,
            "case_counts": e.get("case_counts"),
            "module_status_counts": e.get("module_status_counts"),
            "rows": len(e.get("records", [])) if isinstance(e.get("records"), list) else 0,
        }, sort_keys=True))
print("METRICS_END")
PY
