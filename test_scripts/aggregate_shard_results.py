#!/usr/bin/env python3
"""
汇总多个分片 JSON（RefHM3D sequence 格式）的指标。

每个 JSON 顶层一般为 {"sequence": [ { "sr", "spl", "task_time_sec", ... }, ... ]}；
vlmcor/refine 等可在每条里带 "task_level": object|room|region|instance。

将同一目录下匹配的多个分片合并后计算：任务数、成功率 SR、平均 SPL、平均/总耗时。
可选按 task_level 分层统计（baseline 老结果无 task_level 时会归入 _missing）。

用法示例（在仓库根目录 MTU3D 下）：
  python3 test_scripts/aggregate_shard_results.py
  python3 test_scripts/aggregate_shard_results.py --groups baseline_concise
  python3 test_scripts/aggregate_shard_results.py --verbose
  python3 test_scripts/aggregate_shard_results.py --by-level
  python3 test_scripts/aggregate_shard_results.py --verbose --by-level
  python3 test_scripts/aggregate_shard_results.py --root /path/to/MTU3D --json-out /tmp/summary.json

  自定义目录（可多次），格式 name|相对root的路径|glob：
  python3 test_scripts/aggregate_shard_results.py --by-level \\
    --spec 'run_a|output_logs/baseline/20260411-xxx-detailed|refhm3d_seq_*.json'
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LEVEL_ORDER = ("object", "room", "region", "instance", "_missing")
REASON_ORDER = (
    "model_decision",
    "frontier_exhausted",
    "timeout",
    "decision_error",
    "path_error",
    "unknown",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_bool_sr(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return bool(v)


def _records_from_result_dict(data: Any) -> List[Dict[str, Any]]:
    """从结果 JSON 顶层提取所有子任务记录（兼容仅含 sequence 或其它 goal_type 列表）。"""
    if not isinstance(data, dict):
        return []
    out: List[Dict[str, Any]] = []
    seq = data.get("sequence")
    if isinstance(seq, list):
        out.extend(x for x in seq if isinstance(x, dict))
        if out:
            return out
    for _k, v in data.items():
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict) and "sr" in x:
                    out.append(x)
    return out


def _load_records(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return [], "empty_file"
        data = json.loads(raw)
    except OSError as e:
        return [], f"os_error:{e}"
    except json.JSONDecodeError as e:
        return [], f"json_error:{e}"
    out = _records_from_result_dict(data)
    if not out and isinstance(data, dict):
        return [], "missing_or_invalid_sequence"
    return out, None


def _task_level(task: Dict[str, Any]) -> str:
    """层级字段：vlmcor 为 task_level；旧 baseline 无则 _missing。"""
    lv = task.get("task_level", task.get("task_type"))
    if lv is None or lv == "":
        return "_missing"
    return str(lv)


def _metrics_by_level(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        lv = _task_level(t)
        buckets.setdefault(lv, []).append(t)
    out: Dict[str, Dict[str, float]] = {}
    keys = list(LEVEL_ORDER)
    for k in buckets:
        if k not in keys:
            keys.append(k)
    for lv in keys:
        if lv not in buckets:
            continue
        out[lv] = _metrics_from_tasks(buckets[lv])
    return out


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _termination_reason(task: Dict[str, Any]) -> str:
    reason = task.get("termination_reason")
    if reason is None or reason == "":
        return "unknown"
    return str(reason)


def _reason_sr_matrix(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for t in tasks:
        reason = _termination_reason(t)
        if reason not in matrix:
            matrix[reason] = {"count": 0, "sr_true": 0, "sr_false": 0}
        matrix[reason]["count"] += 1
        if _as_bool_sr(t.get("sr")):
            matrix[reason]["sr_true"] += 1
        else:
            matrix[reason]["sr_false"] += 1
    ordered: Dict[str, Dict[str, int]] = {}
    for k in REASON_ORDER:
        if k in matrix:
            ordered[k] = matrix[k]
    for k in sorted(matrix.keys()):
        if k not in ordered:
            ordered[k] = matrix[k]
    return ordered


def _decision_diagnostics(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(tasks)
    reason_matrix = _reason_sr_matrix(tasks)

    # 语义相关关键格子：
    # 模型主动判断（可认为“模型认为看到了目标”） vs SR
    model_row = reason_matrix.get("model_decision", {"count": 0, "sr_true": 0, "sr_false": 0})
    frontier_row = reason_matrix.get("frontier_exhausted", {"count": 0, "sr_true": 0, "sr_false": 0})

    # final prob 直接阈值统计（<=0.5 为模型偏向 object）
    prob_values: List[float] = []
    prob_le_05_sr_true = 0
    prob_le_05_sr_false = 0
    prob_gt_05_sr_true = 0
    prob_gt_05_sr_false = 0
    for t in tasks:
        p = _to_float_or_none(t.get("final_goto_frontier_prob"))
        if p is None:
            continue
        prob_values.append(p)
        sr = _as_bool_sr(t.get("sr"))
        if p <= 0.5:
            if sr:
                prob_le_05_sr_true += 1
            else:
                prob_le_05_sr_false += 1
        else:
            if sr:
                prob_gt_05_sr_true += 1
            else:
                prob_gt_05_sr_false += 1

    prob_stats: Dict[str, Any] = {
        "available_count": len(prob_values),
        "le_0_5": {
            "count": prob_le_05_sr_true + prob_le_05_sr_false,
            "sr_true": prob_le_05_sr_true,
            "sr_false": prob_le_05_sr_false,
        },
        "gt_0_5": {
            "count": prob_gt_05_sr_true + prob_gt_05_sr_false,
            "sr_true": prob_gt_05_sr_true,
            "sr_false": prob_gt_05_sr_false,
        },
    }
    if prob_values:
        prob_stats.update(
            {
                "mean": sum(prob_values) / len(prob_values),
                "min": min(prob_values),
                "max": max(prob_values),
            }
        )

    return {
        "n_tasks": n,
        "reason_sr_matrix": reason_matrix,
        "semantic_quality": {
            "correct_navigate_to_object_count": model_row["sr_true"],
            "model_false_positive_count": model_row["sr_false"],
            "model_decision_total": model_row["count"],
            "frontier_exhausted_success_count": frontier_row["sr_true"],
            "frontier_exhausted_fail_count": frontier_row["sr_false"],
            "frontier_exhausted_total": frontier_row["count"],
        },
        "final_prob_stats": prob_stats,
    }


def _decision_diagnostics_by_level(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        lv = _task_level(t)
        buckets.setdefault(lv, []).append(t)
    out: Dict[str, Dict[str, Any]] = {}
    keys = list(LEVEL_ORDER)
    for k in buckets:
        if k not in keys:
            keys.append(k)
    for lv in keys:
        if lv in buckets:
            out[lv] = _decision_diagnostics(buckets[lv])
    return out


def _metrics_from_tasks(tasks: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(tasks)
    if n == 0:
        return {
            "n_tasks": 0.0,
            "sr": 0.0,
            "spl": 0.0,
            "avg_task_time_sec": 0.0,
            "sum_task_time_sec": 0.0,
        }
    sr_sum = sum(1.0 for t in tasks if _as_bool_sr(t.get("sr")))
    spl_sum = sum(float(t.get("spl", 0.0) or 0.0) for t in tasks)
    time_sum = sum(float(t.get("task_time_sec", 0.0) or 0.0) for t in tasks)
    return {
        "n_tasks": float(n),
        "sr": sr_sum / n,
        "spl": spl_sum / n,
        "avg_task_time_sec": time_sum / n,
        "sum_task_time_sec": time_sum,
    }


@dataclass
class GroupSpec:
    name: str
    rel_dir: str
    glob: str


# 默认三组：与当前实验目录一致；可按需改脚本或改用 --config
DEFAULT_GROUPS: List[GroupSpec] = [
    GroupSpec(
        name="baseline_concise",
        rel_dir="output_logs/baseline",
        glob="refhm3d_seq_concisedesc_*.json",
    ),
    GroupSpec(
        name="baseline_detailed_20260410-220519",
        rel_dir="output_logs/baseline/20260410-220519",
        glob="refhm3d_seq_*.json",
    ),
    GroupSpec(
        name="vlmcor_20260411-000136-detailed",
        rel_dir="output_logs/anchor/vlmcor/20260411-000136-detailed",
        glob="refhm3d_seq_vlmcor_refine1_*.json",
    ),
]


def _parse_spec_line(line: str) -> GroupSpec:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 3:
        raise ValueError(
            f"--spec 需三段 name|rel_dir|glob，当前: {line!r}"
        )
    return GroupSpec(name=parts[0], rel_dir=parts[1], glob=parts[2])


def _collect_files(root: Path, spec: GroupSpec) -> List[Path]:
    d = (root / spec.rel_dir).resolve()
    if not d.is_dir():
        return []
    paths = sorted(d.glob(spec.glob))
    return [p for p in paths if p.is_file()]


def _format_level_table_rows(by_level: Dict[str, Dict[str, float]], indent: str = "  ") -> List[str]:
    lines = [
        f"{indent}{'task_level':<12} {'tasks':>8} {'SR%':>8} {'SPL':>10} {'avg_t(s)':>10} {'sum_t(s)':>12}"
    ]
    order = [k for k in LEVEL_ORDER if k in by_level]
    order.extend(sorted(k for k in by_level if k not in order))
    for lv in order:
        m = by_level[lv]
        n = int(m["n_tasks"])
        sr_pct = 100.0 * m["sr"] if n else 0.0
        lines.append(
            f"{indent}{lv:<12} {n:>8} {sr_pct:>7.2f}% {m['spl']:>10.6f} "
            f"{m['avg_task_time_sec']:>10.2f} {m['sum_task_time_sec']:>12.1f}"
        )
    return lines


def run_group(root: Path, spec: GroupSpec, verbose: bool, by_level: bool) -> Dict[str, Any]:
    files = _collect_files(root, spec)
    merged: List[Dict[str, Any]] = []
    per_file: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for fp in files:
        seq, err = _load_records(fp)
        if err:
            warnings.append(f"{fp.name}: {err}")
            per_file.append(
                {
                    "file": str(fp.relative_to(root)),
                    "n_tasks": 0,
                    "error": err,
                }
            )
            continue
        merged.extend(seq)
        m = _metrics_from_tasks(seq)
        entry: Dict[str, Any] = {
            "file": str(fp.relative_to(root)),
            "n_tasks": int(m["n_tasks"]),
            "sr": m["sr"],
            "spl": m["spl"],
            "avg_task_time_sec": m["avg_task_time_sec"],
            "sum_task_time_sec": m["sum_task_time_sec"],
        }
        if by_level:
            entry["by_level"] = _metrics_by_level(seq)
            entry["decision_diagnostics_by_level"] = _decision_diagnostics_by_level(seq)
        entry["decision_diagnostics"] = _decision_diagnostics(seq)
        per_file.append(entry)

    agg = _metrics_from_tasks(merged)
    merged_by_level = _metrics_by_level(merged)
    merged_decision_diag = _decision_diagnostics(merged)
    merged_decision_diag_by_level = _decision_diagnostics_by_level(merged) if by_level else {}
    return {
        "name": spec.name,
        "rel_dir": spec.rel_dir,
        "glob": spec.glob,
        "n_files": len(files),
        "n_files_ok": sum(1 for x in per_file if "error" not in x),
        "warnings": warnings,
        "merged": {
            "n_tasks": int(agg["n_tasks"]),
            "sr": agg["sr"],
            "spl": agg["spl"],
            "avg_task_time_sec": agg["avg_task_time_sec"],
            "sum_task_time_sec": agg["sum_task_time_sec"],
        },
        "merged_by_level": merged_by_level,
        "merged_decision_diagnostics": merged_decision_diag,
        "merged_decision_diagnostics_by_level": merged_decision_diag_by_level,
        "per_file": per_file if verbose else [],
    }


def _format_reason_sr_rows(reason_sr: Dict[str, Dict[str, int]], indent: str = "  ") -> List[str]:
    lines = [f"{indent}{'reason':<20} {'count':>8} {'sr_true':>8} {'sr_false':>9} {'sr%':>8}"]
    for reason, row in reason_sr.items():
        c = int(row.get("count", 0))
        t = int(row.get("sr_true", 0))
        f = int(row.get("sr_false", 0))
        sr_pct = (100.0 * t / c) if c > 0 else 0.0
        lines.append(f"{indent}{reason:<20} {c:>8} {t:>8} {f:>9} {sr_pct:>7.2f}%")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate RefHM3D shard JSON metrics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=_project_root(),
        help="项目根目录（含 output_logs），默认为本仓库 MTU3D 根",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个分片文件的子统计",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="将完整结果写入 JSON 文件",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        metavar="NAME",
        help="只统计指定组（可多个）；不传则统计 DEFAULT_GROUPS 中全部。组名见脚本内 GroupSpec.name",
    )
    parser.add_argument(
        "--by-level",
        action="store_true",
        help="按 task_level（object/room/region/instance）输出分层指标；无该字段时计入 _missing",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        metavar="NAME|REL_DIR|GLOB",
        help="自定义一组（相对 --root 的目录 + glob）。可多次；若提供则优先使用，不再用 --groups / 默认列表",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    if args.spec:
        try:
            specs = [_parse_spec_line(s) for s in args.spec]
        except ValueError as e:
            print(f"[aggregate_shard_results] {e}", file=sys.stderr)
            return 2
    elif args.groups:
        wanted = set(args.groups)
        specs = [s for s in DEFAULT_GROUPS if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            print(f"[aggregate_shard_results] 未知组名（已忽略）: {sorted(missing)}", file=sys.stderr)
        if not specs:
            print("[aggregate_shard_results] 没有匹配的组，可用组名:", [s.name for s in DEFAULT_GROUPS], file=sys.stderr)
            return 2
    else:
        specs = list(DEFAULT_GROUPS)

    results: List[Dict[str, Any]] = []
    for spec in specs:
        results.append(run_group(root, spec, verbose=args.verbose, by_level=args.by_level))

    lines: List[str] = []
    lines.append(f"project_root={root}")
    lines.append("")
    lines.append(
        f"{'group':<40} {'shards':>7} {'tasks':>8} {'SR%':>8} {'SPL':>10} {'avg_t(s)':>10} {'sum_t(s)':>12}"
    )
    lines.append("-" * 100)

    for r in results:
        m = r["merged"]
        n_tasks = int(m["n_tasks"])
        sr_pct = 100.0 * m["sr"] if n_tasks else 0.0
        lines.append(
            f"{r['name']:<40} {r['n_files']:>7} {n_tasks:>8} {sr_pct:>7.2f}% {m['spl']:>10.6f} "
            f"{m['avg_task_time_sec']:>10.2f} {m['sum_task_time_sec']:>12.1f}"
        )
        for w in r.get("warnings", []):
            lines.append(f"  [warn] {w}")
        if args.by_level and r.get("merged_by_level"):
            lines.append("  [merged by task_level]")
            lines.extend(_format_level_table_rows(r["merged_by_level"], indent="    "))
        diag = r.get("merged_decision_diagnostics", {})
        if diag:
            sq = diag.get("semantic_quality", {})
            prob = diag.get("final_prob_stats", {})
            lines.append("  [decision diagnostics overall]")
            lines.append(
                "    semantic_quality: "
                f"correct_navigate_to_object={sq.get('correct_navigate_to_object_count', 0)}, "
                f"model_false_positive={sq.get('model_false_positive_count', 0)}, "
                f"frontier_exhausted_success={sq.get('frontier_exhausted_success_count', 0)}, "
                f"frontier_exhausted_fail={sq.get('frontier_exhausted_fail_count', 0)}"
            )
            lines.append(
                "    final_prob_stats: "
                f"available={prob.get('available_count', 0)}, "
                f"le_0.5={prob.get('le_0_5', {}).get('count', 0)}, "
                f"gt_0.5={prob.get('gt_0_5', {}).get('count', 0)}, "
                f"mean={prob.get('mean', 0.0):.6f}" if prob.get("available_count", 0) else
                "    final_prob_stats: available=0"
            )
            lines.extend(
                _format_reason_sr_rows(
                    diag.get("reason_sr_matrix", {}),
                    indent="    ",
                )
            )
        if args.by_level and r.get("merged_decision_diagnostics_by_level"):
            lines.append("  [decision diagnostics by task_level]")
            for lv in [k for k in LEVEL_ORDER if k in r["merged_decision_diagnostics_by_level"]]:
                diag_lv = r["merged_decision_diagnostics_by_level"][lv]
                sq = diag_lv.get("semantic_quality", {})
                lines.append(
                    f"    - {lv}: "
                    f"correct_navigate_to_object={sq.get('correct_navigate_to_object_count', 0)}, "
                    f"model_false_positive={sq.get('model_false_positive_count', 0)}, "
                    f"frontier_exhausted_success={sq.get('frontier_exhausted_success_count', 0)}, "
                    f"frontier_exhausted_fail={sq.get('frontier_exhausted_fail_count', 0)}"
                )
                lines.extend(_format_reason_sr_rows(diag_lv.get("reason_sr_matrix", {}), indent="      "))
        if args.verbose and r.get("per_file"):
            for pf in r["per_file"]:
                if "error" in pf:
                    lines.append(f"    - {pf['file']} ERROR {pf['error']}")
                else:
                    lines.append(
                        f"    - {pf['file']}: tasks={pf['n_tasks']} "
                        f"SR={100*pf['sr']:.2f}% SPL={pf['spl']:.6f}"
                    )
                    if args.by_level and pf.get("by_level"):
                        lines.extend(_format_level_table_rows(pf["by_level"], indent="        "))
                    if pf.get("decision_diagnostics"):
                        sq = pf["decision_diagnostics"].get("semantic_quality", {})
                        lines.append(
                            f"      decision_quality: correct_navigate_to_object={sq.get('correct_navigate_to_object_count', 0)} "
                            f"model_false_positive={sq.get('model_false_positive_count', 0)}"
                        )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    sys.stdout.write(text)

    if args.json_out is not None:
        out_payload = {
            "project_root": str(root),
            "groups": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[aggregate_shard_results] wrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
