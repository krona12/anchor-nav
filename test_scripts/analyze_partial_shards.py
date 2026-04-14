#!/usr/bin/env python3
"""
对“未跑完整区间”的分片结果做对比分析。

默认比较：
1) output_logs/anchor/rerank_instance_0_0.5/20260413-171525-detailed
2) output_logs/baseline_instance_0_0.5/20260413-150752-detailed

默认只统计 0.0_0.05 与 0.05_0.1 两个分片，并输出：
- 每组总体 n / SR / SPL
- 每个 task_level 的 n / SR / SPL
- rerank - baseline 的差值
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

DEFAULT_RERANK_DIR = (
    "output_logs/anchor/rerank_instance_0_0.5/20260413-171525-detailed"
)
DEFAULT_BASELINE_DIR = (
    "output_logs/baseline_instance_0_0.5/20260413-150752-detailed"
)
DEFAULT_SHARDS = ("0.0_0.05", "0.05_0.1")
LEVEL_ORDER = ("object", "room", "region", "instance", "_missing")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_bool_sr(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return bool(v)


def _load_sequence(json_path: Path) -> List[Dict[str, Any]]:
    raw = json_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    seq = data.get("sequence", [])
    if not isinstance(seq, list):
        return []
    return [x for x in seq if isinstance(x, dict)]


def _task_level(task: Dict[str, Any]) -> str:
    lv = task.get("task_level", task.get("task_type"))
    if lv is None or lv == "":
        return "_missing"
    return str(lv)


def _metrics(tasks: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(tasks)
    if n == 0:
        return {"n": 0.0, "sr": 0.0, "spl": 0.0}
    sr = sum(1.0 for t in tasks if _to_bool_sr(t.get("sr"))) / n
    spl = sum(float(t.get("spl", 0.0) or 0.0) for t in tasks) / n
    return {"n": float(n), "sr": sr, "spl": spl}


def _metrics_by_level(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        buckets.setdefault(_task_level(t), []).append(t)
    ordered_levels = [lv for lv in LEVEL_ORDER if lv in buckets]
    ordered_levels.extend(sorted(lv for lv in buckets if lv not in ordered_levels))
    return {lv: _metrics(buckets[lv]) for lv in ordered_levels}


def _collect_tasks(run_dir: Path, prefix: str, shards: Sequence[str]) -> List[Dict[str, Any]]:
    all_tasks: List[Dict[str, Any]] = []
    for shard in shards:
        path = run_dir / f"{prefix}{shard}.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少分片文件: {path}")
        all_tasks.extend(_load_sequence(path))
    return all_tasks


def _format_overall(name: str, m: Dict[str, float]) -> str:
    return f"{name:<10} n={int(m['n']):>4}  SR={100*m['sr']:>6.2f}%  SPL={m['spl']:.6f}"


def _format_level_row(level: str, m: Dict[str, float], m2: Dict[str, float] | None = None) -> str:
    if m2 is None:
        return (
            f"  {level:<10} n={int(m['n']):>4}  SR={100*m['sr']:>6.2f}%  SPL={m['spl']:.6f}"
        )
    return (
        f"  {level:<10} n={int(m['n']):>4}  SR={100*m['sr']:>6.2f}%  SPL={m['spl']:.6f}"
        f"   |   ΔSR={100*(m['sr']-m2['sr']):>+6.2f}%  ΔSPL={(m['spl']-m2['spl']):+.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze partial shard results (0-0.1 by default).")
    parser.add_argument("--root", type=Path, default=_project_root(), help="项目根目录")
    parser.add_argument("--rerank-dir", default=DEFAULT_RERANK_DIR, help="rerank 结果目录（相对 --root）")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR, help="baseline 结果目录（相对 --root）")
    parser.add_argument(
        "--shards",
        nargs="+",
        default=list(DEFAULT_SHARDS),
        help="要统计的分片后缀列表，如: 0.0_0.05 0.05_0.1",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选：把结构化结果写到 JSON 文件",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    rerank_dir = (root / args.rerank_dir).resolve()
    baseline_dir = (root / args.baseline_dir).resolve()

    rerank_tasks = _collect_tasks(rerank_dir, "refhm3d_seq_rerank_refine1_", args.shards)
    baseline_tasks = _collect_tasks(baseline_dir, "refhm3d_seq_", args.shards)

    rerank_overall = _metrics(rerank_tasks)
    baseline_overall = _metrics(baseline_tasks)
    rerank_by_level = _metrics_by_level(rerank_tasks)
    baseline_by_level = _metrics_by_level(baseline_tasks)

    lines: List[str] = []
    lines.append(f"root={root}")
    lines.append(f"shards={list(args.shards)}")
    lines.append("")
    lines.append("== Overall ==")
    lines.append(_format_overall("rerank", rerank_overall))
    lines.append(_format_overall("baseline", baseline_overall))
    lines.append(
        f"delta      n={int(rerank_overall['n']):>4}  "
        f"SR={100*(rerank_overall['sr']-baseline_overall['sr']):>+6.2f}%  "
        f"SPL={(rerank_overall['spl']-baseline_overall['spl']):+.6f}"
    )
    lines.append("")
    lines.append("== By task_level (rerank vs baseline) ==")

    all_levels = list(LEVEL_ORDER)
    all_levels.extend(
        sorted(
            {
                *rerank_by_level.keys(),
                *baseline_by_level.keys(),
            }
            - set(all_levels)
        )
    )
    for lv in all_levels:
        if lv not in rerank_by_level and lv not in baseline_by_level:
            continue
        r = rerank_by_level.get(lv, {"n": 0.0, "sr": 0.0, "spl": 0.0})
        b = baseline_by_level.get(lv, {"n": 0.0, "sr": 0.0, "spl": 0.0})
        lines.append(_format_level_row(lv, r, b))

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.json_out is not None:
        payload = {
            "root": str(root),
            "shards": list(args.shards),
            "rerank_dir": str(rerank_dir),
            "baseline_dir": str(baseline_dir),
            "rerank_overall": rerank_overall,
            "baseline_overall": baseline_overall,
            "rerank_by_level": rerank_by_level,
            "baseline_by_level": baseline_by_level,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[analyze_partial_shards] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
