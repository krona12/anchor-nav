#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN_DIR = Path("output_logs/anchor/single_goal/vista2mqsc_sam2/balanced-concise-qwen_vl_plus_full")
PLAN = {
    "w0": {"cuda": "2", "shards": [0, 1, 5, 9, 11], "expected": 2394},
    "w1": {"cuda": "2", "shards": [6, 7, 8, 16, 19], "expected": 2390},
    "w2": {"cuda": "2", "shards": [2, 12, 13, 15, 18], "expected": 2398},
    "w3": {"cuda": "2", "shards": [3, 4, 10, 14, 17], "expected": 2238},
}
STARTS = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
ENDS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def json_count(index: int) -> int | None:
    path = RUN_DIR / f"refhm3d_single_goal_vista2mqsc_refine1_concisedesc_{STARTS[index]}_{ENDS[index]}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return -1
    rows = payload.get("sequence", [])
    return len(rows) if isinstance(rows, list) else -1


def main() -> int:
    shard_to_worker: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    missing: list[int] = []
    for worker, cfg in PLAN.items():
        for shard in cfg["shards"]:
            if shard in shard_to_worker:
                duplicates.setdefault(shard, [shard_to_worker[shard]]).append(worker)
            shard_to_worker[shard] = worker
    for shard in range(20):
        if shard not in shard_to_worker:
            missing.append(shard)

    sessions = run(["tmux", "ls"])
    ps = run(["ps", "-eo", "pid,ppid,stat,etime,cmd"])
    print(f"run_dir={RUN_DIR}")
    print(f"duplicate_shards={duplicates or {}}")
    print(f"missing_shards={missing}")
    print("| worker | expected CUDA | shards | expected tasks | tmux alive | process alive | shard row counts |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for worker, cfg in PLAN.items():
        session_prefix = f"single_goal_sam2_balanced_concise_{worker}_cuda{cfg['cuda']}_qwen_vl_plus_full"
        tmux_alive = session_prefix in sessions
        proc_alive = (
            session_prefix in ps
            or f"balanced-concise-qwen_vl_plus_full/worker_logs/worker_{worker}_stdout_stderr.log" in ps
            or session_prefix in sessions
        )
        counts = ",".join(f"{i}:{json_count(i)}" for i in cfg["shards"])
        print(
            f"| {worker} | {cfg['cuda']} | {','.join(map(str, cfg['shards']))} | "
            f"{cfg['expected']} | {tmux_alive} | {proc_alive} | {counts} |"
        )

    existing_counts = [f"{i}:{json_count(i)}" for i in range(20) if json_count(i) is not None]
    print("\nExisting shard counts:")
    print(", ".join(existing_counts) if existing_counts else "(none yet)")
    return 1 if duplicates or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
