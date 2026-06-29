#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


BAD_LOG_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|out of memory|Error 803|NVML|"
    r"ModuleNotFoundError|FileNotFoundError|mqsc_exception_fallback|heuristic_fallback|"
    r"VLMDisabled|401 Client Error|403 Client Error|429 Client Error",
    re.IGNORECASE,
)


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _safe_read(path: Path, tail_chars: int | None = None) -> str:
    text = path.read_text(errors="replace")
    if tail_chars is not None and len(text) > tail_chars:
        return text[-tail_chars:]
    return text


def check_health(
    *,
    session: str,
    run_dir: Path,
    expected_model: str,
    expected_preset: str,
    expected_cuda: str,
    require_output_json: bool,
    require_mqsc_vlm: bool,
    stale_sec: int,
    recent_sec: int,
) -> Dict[str, Any]:
    now = time.time()
    errors: List[str] = []
    notes: List[str] = []

    if _run(["tmux", "has-session", "-t", session]).returncode != 0:
        errors.append("tmux session missing")
    else:
        notes.append("tmux session alive")

    ps = _run(["ps", "-eo", "pid,ppid,stat,etime,cmd"]).stdout
    proc_lines = [
        line
        for line in ps.splitlines()
        if "refhm3d-nav-sequence-sam2-runner.py" in line and str(run_dir) in line
    ]
    if not proc_lines:
        errors.append("sam2 single-goal python process not found")
    else:
        cmdline = proc_lines[0]
        if f"--mqsc_r1_vlm_model {expected_model}" not in cmdline:
            errors.append(f"python command missing --mqsc_r1_vlm_model {expected_model}")
        if "refhm3d-nav-single-goal-analyze-anchor-vista2mqsc-refine1.py" not in cmdline:
            errors.append("python target is not single-goal vista2mqsc")
        notes.append("python process alive with expected model")

    run_args = run_dir / "run_args.txt"
    if not run_args.exists():
        errors.append("run_args.txt missing")
    else:
        text = _safe_read(run_args)
        required = [
            "segmenter=sam2.1",
            f"sam2_level_preset={expected_preset}",
            f"mqsc_r1_vlm_model={expected_model}",
            f"cuda_visible_devices={expected_cuda}",
            "num_shards_total=20",
            "schedule=sequential_shards_resume_by_output_json",
        ]
        for item in required:
            if item not in text:
                errors.append(f"run_args missing {item}")
        notes.append("run_args records sam2/qwen/cuda/shards")

    log = run_dir / "run_stdout_stderr.log"
    if not log.exists():
        errors.append("run_stdout_stderr.log missing")
        log_text = ""
    else:
        age = now - log.stat().st_mtime
        if age > stale_sec:
            errors.append(f"run log stale age_sec={age:.0f}")
        log_text = _safe_read(log, tail_chars=300_000)
        markers = [
            f"[preflight] VLM reachable model={expected_model}",
            "[sam2-runner]",
            "[sam2-baseline]",
            f"level_preset={expected_preset}",
            "profiles=['object', 'room', 'region', 'instance']",
            f"mqsc_r1_vlm_model={expected_model}",
        ]
        for marker in markers:
            if marker not in log_text:
                errors.append(f"log missing marker: {marker}")
        step_count = len(re.findall(r"\[vista2mqsc-refine1\]\[step\]", log_text))
        final_count = len(re.findall(r"\[vista2mqsc-refine1\]\[final\]", log_text))
        metrics_count = len(re.findall(r"\[VISTA2MQSC_LIVE_METRICS\]", log_text))
        task_start_count = len(re.findall(r"\[vista2mqsc-refine1\]\[task-start\]", log_text))
        if step_count < 1:
            errors.append("no navigation step lines yet")
        notes.append(
            f"progress task_start={task_start_count} step={step_count} "
            f"final={final_count} metrics={metrics_count}"
        )

    bad_hits: List[str] = []
    if run_dir.exists():
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                if now - path.stat().st_mtime > recent_sec:
                    continue
                txt = _safe_read(path, tail_chars=120_000)
            except Exception:
                continue
            match = BAD_LOG_RE.search(txt)
            if match:
                bad_hits.append(f"{path}: {match.group(0)}")
    if bad_hits:
        errors.extend(["bad log keyword " + item for item in bad_hits[:8]])
    else:
        notes.append("no recent error/fallback keywords")

    valid_json = 0
    row_count = 0
    for path in sorted(run_dir.glob("refhm3d_single_goal_vista2mqsc_refine1_*.json")):
        if "effectiveness" in path.name:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"invalid output json {path}: {type(exc).__name__}")
            continue
        sequence = payload.get("sequence", [])
        if isinstance(sequence, list):
            valid_json += 1
            row_count += len(sequence)
    if valid_json:
        notes.append(f"valid output json shards={valid_json} rows={row_count}")
    else:
        notes.append("output json not written yet")
    if require_output_json and row_count < 1:
        errors.append("required output json row not available yet")

    mqsc_files = (
        sorted((run_dir / "process").glob("**/mqsc-r1/dec_*_mqsc_r1.json"))
        if (run_dir / "process").exists()
        else []
    )
    if mqsc_files:
        checked = 0
        bad_sources: List[str] = []
        for path in mqsc_files[-10:]:
            try:
                payload = json.loads(path.read_text())
            except Exception as exc:
                errors.append(f"invalid mqsc json {path}: {type(exc).__name__}")
                continue
            decomp = payload.get("decomposition", {})
            source = decomp.get("source")
            if source != "vlm":
                bad_sources.append(
                    f"{path}: source={source!r} error={decomp.get('error_type')!r}"
                )
            checked += 1
        if bad_sources:
            errors.extend(["mqsc decomposition not vlm " + item for item in bad_sources[:5]])
        else:
            notes.append(f"mqsc debug source=vlm checked={checked}")
    else:
        notes.append("mqsc debug not available yet; preflight verified qwen-vl-plus")
        if require_mqsc_vlm:
            errors.append("required mqsc debug source=vlm not available yet")

    return {
        "healthy": not errors,
        "errors": errors,
        "notes": notes,
        "session": session,
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Health check for SAM2.1 single-goal qwen run.")
    parser.add_argument("--check", type=int, default=0)
    parser.add_argument(
        "--session",
        default="single_goal_ours_sam2_balanced_detailed_cuda1_qwen_vl_plus_full",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output_logs/anchor/single_goal/vista2mqsc_sam2/balanced-detailed-qwen_vl_plus_full"),
    )
    parser.add_argument("--expected-model", default="qwen-vl-plus")
    parser.add_argument("--expected-preset", default="balanced")
    parser.add_argument("--expected-cuda", default="1")
    parser.add_argument("--require-output-json", action="store_true")
    parser.add_argument("--require-mqsc-vlm", action="store_true")
    parser.add_argument("--stale-sec", type=int, default=900)
    parser.add_argument("--recent-sec", type=int, default=900)
    args = parser.parse_args()

    result = check_health(
        session=args.session,
        run_dir=args.run_dir,
        expected_model=args.expected_model,
        expected_preset=args.expected_preset,
        expected_cuda=args.expected_cuda,
        require_output_json=bool(args.require_output_json),
        require_mqsc_vlm=bool(args.require_mqsc_vlm),
        stale_sec=int(args.stale_sec),
        recent_sec=int(args.recent_sec),
    )
    if args.check:
        result["check"] = int(args.check)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
