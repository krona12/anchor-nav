#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN_DIR = Path("output_logs/anchor/single_goal/vista2mqsc_sam2/balanced-detailed-qwen_vl_plus_full")
PLAN = {
    "w0": {"cuda": "1", "shards": [2, 5, 8, 11, 14, 17]},
    "w1": {"cuda": "3", "shards": [3, 7, 12, 16]},
    "w2": {"cuda": "3", "shards": [4, 9, 13, 18]},
    "w3": {"cuda": "3", "shards": [6, 10, 15, 19]},
}
STARTS = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]
ENDS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def json_count(index: int) -> int | None:
    path = RUN_DIR / f"refhm3d_single_goal_vista2mqsc_refine1_{STARTS[index]}_{ENDS[index]}.json"
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
    for worker, cfg in PLAN.items():
        for shard in cfg["shards"]:
            if shard in shard_to_worker:
                duplicates.setdefault(shard, [shard_to_worker[shard]]).append(worker)
            shard_to_worker[shard] = worker

    sessions = run(["tmux", "ls"])
    ps = run(["ps", "-eo", "pid,ppid,stat,etime,cmd"])
    print(f"run_dir={RUN_DIR}")
    print(f"duplicate_shards={duplicates or {}}")
    print("| worker | expected CUDA | shards | tmux alive | process alive | shard row counts |")
    print("|---|---:|---:|---:|---:|---:|")
    for worker, cfg in PLAN.items():
        session_prefix = f"single_goal_sam2_balanced_{worker}_cuda{cfg['cuda']}_qwen_vl_plus_full"
        tmux_alive = session_prefix in sessions
        proc_alive = f"SAM2_WORKER_ID='{worker}'" in ps or f"SAM2_WORKER_ID={worker}" in ps or f"worker_{worker}_stdout" in ps
        counts = ",".join(f"{i}:{json_count(i)}" for i in cfg["shards"])
        print(f"| {worker} | {cfg['cuda']} | {','.join(map(str, cfg['shards']))} | {tmux_alive} | {proc_alive} | {counts} |")

    print("\\nExisting shard counts:")
    print(", ".join(f"{i}:{json_count(i)}" for i in range(20) if json_count(i) is not None))
    return 1 if duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
