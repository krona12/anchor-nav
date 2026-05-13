#!/usr/bin/env python3
"""
Analyze VISTA module effectiveness from a vista-refine1 run log.

The default input is the 20260512 vista_all_0.05_0.1 run. The script can work
from the log alone, and will automatically use result/effectiveness JSON files
from the same directory when they are present for more precise task-level stats.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / (
    "output_logs/anchor/vista_all_0.05_0.1/"
    "20260512-205908-detailed-vista-all-00501/"
    "refhm3d-nav-sequence-analyze-anchor-vista-refine1-"
    "20260512-205927-562428-pid807201.log"
)

CASE_ORDER = ("00", "01", "10", "11")
LEVEL_ORDER = ("object", "room", "region", "instance", "_missing")
EPS = 1e-9

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).replace("\r", "").strip()


def _parse_scalar(raw: str) -> Any:
    value = raw.rstrip(",")
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value in {"None", "none", "null"}:
        return None
    lowered = value.lower()
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    if lowered == "nan":
        return float("nan")
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"")
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_kv(payload: str) -> Dict[str, Any]:
    return {match.group(1): _parse_scalar(match.group(2)) for match in KV_RE.finditer(payload)}


def _parse_literal_dict(line: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"(\{.*\})", line)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _case_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text in CASE_ORDER:
        return text
    try:
        number = int(text)
    except ValueError:
        return text
    if 0 <= number <= 1:
        return f"0{number}"
    return str(number)


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{100.0 * count / total:.2f}%"


def _mean(values: Iterable[float]) -> float:
    nums = list(values)
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(num):
        return "nan"
    if math.isinf(num):
        return "inf" if num > 0 else "-inf"
    return f"{num:.{digits}f}"


def _format_counter(counter: Mapping[str, int], order: Sequence[str]) -> str:
    keys = list(order)
    keys.extend(sorted(str(k) for k in counter if str(k) not in keys))
    return ", ".join(f"{key}:{int(counter.get(key, 0))}" for key in keys)


def parse_log(log_path: Path) -> Dict[str, Any]:
    module_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    case_snapshots: List[Dict[str, Any]] = []
    status_snapshots: List[Dict[str, Any]] = []
    metrics_snapshots: List[Dict[str, Any]] = []
    live_snapshots: List[Dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = _clean_line(raw_line)
            if "[vista-refine1][module]" in line:
                payload = line.split("[vista-refine1][module]", 1)[1]
                row = _parse_kv(payload)
                row["_line"] = lineno
                row["is_final"] = row.get("final", True) is not False
                module_rows.append(row)
            elif "[vista-refine1][task-summary]" in line:
                payload = line.split("[vista-refine1][task-summary]", 1)[1]
                row = _parse_kv(payload)
                row["_line"] = lineno
                task_rows.append(row)
            elif "[vista-refine1][case-counts]" in line:
                parsed = _parse_literal_dict(line)
                if parsed is not None:
                    case_snapshots.append(parsed)
            elif "[vista-refine1][module-status-counts]" in line:
                parsed = _parse_literal_dict(line)
                if parsed is not None:
                    status_snapshots.append(parsed)
            elif "[Metrics]" in line and "sequence count=" in line:
                payload = line.split("[Metrics]", 1)[1].replace("sequence count=", "count=", 1)
                row = _parse_kv(payload)
                row["_line"] = lineno
                metrics_snapshots.append(row)
            elif "[VISTA_LIVE_METRICS]" in line:
                payload = line.split("[VISTA_LIVE_METRICS]", 1)[1]
                row = _parse_kv(payload)
                row["_line"] = lineno
                live_snapshots.append(row)

    return {
        "module_rows": module_rows,
        "task_rows": task_rows,
        "case_snapshots": case_snapshots,
        "status_snapshots": status_snapshots,
        "metrics_snapshots": metrics_snapshots,
        "live_snapshots": live_snapshots,
    }


def _auto_result_json(log_path: Path) -> Optional[Path]:
    candidates = [
        p
        for p in sorted(log_path.parent.glob("refhm3d_seq_vista_refine1_*.json"))
        if "effectiveness" not in p.name
    ]
    return candidates[0] if len(candidates) == 1 else None


def _auto_effectiveness_json(log_path: Path) -> Optional[Path]:
    candidates = sorted(log_path.parent.glob("refhm3d_seq_vista_refine1_effectiveness_*.json"))
    return candidates[0] if len(candidates) == 1 else None


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _normalize_log_task(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scene_name": row.get("scene"),
        "episode_id": row.get("ep"),
        "task_id": row.get("task"),
        "task_level": row.get("level", "_missing"),
        "sr": _safe_float(row.get("SR")),
        "spl": _safe_float(row.get("SPL")),
        "steps_total": _safe_int(row.get("steps")),
        "decisions": _safe_int(row.get("decisions")),
        "end_reason": row.get("end_reason"),
        "vista_case": _case_value(row.get("case")),
        "vista_helpful": row.get("helpful"),
    }


def _normalize_module(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scene_name": row.get("scene"),
        "episode_id": row.get("ep"),
        "task_id": row.get("task"),
        "decision_num": row.get("dec"),
        "is_final": bool(row.get("is_final", True)),
        "called": row.get("called"),
        "source": row.get("source"),
        "correction_applied": row.get("correction_applied"),
        "viewpoint_applied": row.get("viewpoint_applied"),
        "efficiency_score": row.get("efficiency_score"),
        "rejected_reason": row.get("rejected_reason"),
        "case_metric": row.get("case_metric"),
        "case": _case_value(row.get("case")),
        "baseline_gt_viewpoint_in_1m": row.get("baseline_gt_viewpoint_in_1m"),
        "corrected_gt_viewpoint_in_1m": row.get("corrected_gt_viewpoint_in_1m"),
        "baseline_vp_geo": row.get("baseline_vp_geo"),
        "corrected_vp_geo": row.get("corrected_vp_geo"),
        "object_l2_case": _case_value(row.get("object_l2_case")),
        "baseline_obj_l2": row.get("baseline_obj_l2"),
        "corrected_obj_l2": row.get("corrected_obj_l2"),
        "_line": row.get("_line"),
    }


def _task_key(record: Mapping[str, Any]) -> Tuple[str, int, int]:
    return (
        str(record.get("scene_name", record.get("scene", ""))),
        _safe_int(record.get("episode_id", record.get("ep"))),
        _safe_int(record.get("task_id", record.get("task"))),
    )


def _ordered_group_keys(keys: Iterable[str], preferred: Sequence[str]) -> List[str]:
    seen = {str(k) for k in keys}
    ordered = [key for key in preferred if key in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def summarize_tasks(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(tasks)
    sr_values = [_safe_float(t.get("sr")) for t in tasks]
    spl_values = [_safe_float(t.get("spl")) for t in tasks]
    steps = [_safe_float(t.get("steps_total")) for t in tasks if t.get("steps_total") is not None]
    decisions = [_safe_float(t.get("decisions")) for t in tasks if t.get("decisions") is not None]
    cases = Counter(str(t.get("vista_case")) for t in tasks if t.get("vista_case") is not None)
    helpful = Counter(str(t.get("vista_helpful")) for t in tasks)
    return {
        "count": n,
        "sr": _mean(sr_values),
        "spl": _mean(spl_values),
        "success_count": sum(1 for v in sr_values if v >= 0.5),
        "avg_steps": _mean(steps),
        "avg_decisions": _mean(decisions),
        "case_counts": dict(cases),
        "helpful_counts": dict(helpful),
    }


def group_task_metrics(tasks: Sequence[Mapping[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        value = task.get(key_name)
        buckets[str(value if value not in (None, "") else "_missing")].append(task)
    order = LEVEL_ORDER if key_name == "task_level" else CASE_ORDER
    return {key: summarize_tasks(buckets[key]) for key in _ordered_group_keys(buckets.keys(), order)}


def _distance_cmp(baseline: Any, corrected: Any) -> Optional[str]:
    base = _safe_float(baseline, default=float("nan"))
    corr = _safe_float(corrected, default=float("nan"))
    if math.isnan(base) or math.isnan(corr):
        return None
    if math.isinf(base) and math.isinf(corr):
        return "tied"
    if math.isinf(base) and not math.isinf(corr):
        return "improved"
    if not math.isinf(base) and math.isinf(corr):
        return "worsened"
    if corr < base - EPS:
        return "improved"
    if corr > base + EPS:
        return "worsened"
    return "tied"


def distance_quality(rows: Sequence[Mapping[str, Any]], baseline_key: str, corrected_key: str) -> Dict[str, Any]:
    counts = Counter()
    finite_deltas: List[float] = []
    for row in rows:
        cmp_result = _distance_cmp(row.get(baseline_key), row.get(corrected_key))
        if cmp_result is None:
            continue
        counts[cmp_result] += 1
        base = _safe_float(row.get(baseline_key), default=float("nan"))
        corr = _safe_float(row.get(corrected_key), default=float("nan"))
        if math.isfinite(base) and math.isfinite(corr):
            finite_deltas.append(corr - base)
    total = sum(counts.values())
    return {
        "count": total,
        "improved": counts["improved"],
        "worsened": counts["worsened"],
        "tied": counts["tied"],
        "mean_delta_finite": _mean(finite_deltas),
        "min_delta_finite": min(finite_deltas) if finite_deltas else 0.0,
        "max_delta_finite": max(finite_deltas) if finite_deltas else 0.0,
    }


def case_effectiveness(case_counts: Mapping[str, int]) -> Dict[str, Any]:
    counts = Counter({str(k): int(v) for k, v in case_counts.items()})
    total = sum(counts.values())
    baseline_success = counts["10"] + counts["11"]
    corrected_success = counts["01"] + counts["11"]
    return {
        "count": total,
        "case_counts": {case: counts[case] for case in CASE_ORDER},
        "baseline_success": baseline_success,
        "corrected_success": corrected_success,
        "rescued_01": counts["01"],
        "regressed_10": counts["10"],
        "unchanged_success_11": counts["11"],
        "unchanged_fail_00": counts["00"],
        "net_success_delta": corrected_success - baseline_success,
        "baseline_success_rate": baseline_success / total if total else 0.0,
        "corrected_success_rate": corrected_success / total if total else 0.0,
    }


def summarize_modules(module_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [_normalize_module(row) for row in module_rows]
    case_counts = Counter(str(row.get("case")) for row in rows if row.get("case") is not None)
    object_case_counts = Counter(
        str(row.get("object_l2_case")) for row in rows if row.get("object_l2_case") is not None
    )
    efficiency_values = [
        _safe_float(row.get("efficiency_score"))
        for row in rows
        if row.get("efficiency_score") is not None and math.isfinite(_safe_float(row.get("efficiency_score")))
    ]
    return {
        "count": len(rows),
        "final_count": sum(1 for row in rows if row.get("is_final")),
        "non_final_count": sum(1 for row in rows if not row.get("is_final")),
        "called_count": sum(1 for row in rows if row.get("called") is True),
        "correction_applied_count": sum(1 for row in rows if row.get("correction_applied") is True),
        "viewpoint_applied_count": sum(1 for row in rows if row.get("viewpoint_applied") is True),
        "rejected_count": sum(1 for row in rows if row.get("rejected_reason") is not None),
        "case_counts": dict(case_counts),
        "object_l2_case_counts": dict(object_case_counts),
        "viewpoint_geo_quality": distance_quality(rows, "baseline_vp_geo", "corrected_vp_geo"),
        "object_l2_quality": distance_quality(rows, "baseline_obj_l2", "corrected_obj_l2"),
        "efficiency_score_mean": _mean(efficiency_values),
        "efficiency_score_min": min(efficiency_values) if efficiency_values else 0.0,
        "efficiency_score_max": max(efficiency_values) if efficiency_values else 0.0,
    }


def summarize_status_counts(status_counts: Mapping[str, Any]) -> Dict[str, int]:
    return {str(k): _safe_int(v) for k, v in status_counts.items()}


def final_task_records(
    parsed_log: Mapping[str, Any],
    result_json: Optional[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    if result_json is not None and isinstance(result_json.get("sequence"), list):
        rows = [row for row in result_json["sequence"] if isinstance(row, dict)]
        return rows, "result_json"
    return [_normalize_log_task(row) for row in parsed_log["task_rows"]], "log_task_summary"


def task_distance_rows(tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                **dict(task),
                "baseline_vp_geo": task.get("baseline_target_to_gt_viewpoint_geo"),
                "corrected_vp_geo": task.get("corrected_target_to_gt_viewpoint_geo"),
                "baseline_obj_l2": task.get("baseline_target_to_goal_l2"),
                "corrected_obj_l2": task.get("corrected_target_to_goal_l2"),
            }
        )
    return rows


def build_report(
    *,
    log_path: Path,
    result_json_path: Optional[Path],
    effectiveness_json_path: Optional[Path],
    parsed_log: Mapping[str, Any],
    result_json: Optional[Mapping[str, Any]],
    effectiveness_json: Optional[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    tasks, task_source = final_task_records(parsed_log, result_json)
    module_rows = [_normalize_module(row) for row in parsed_log["module_rows"]]
    final_module_rows = [row for row in module_rows if row.get("is_final")]

    if effectiveness_json and isinstance(effectiveness_json.get("case_counts"), dict):
        aggregate_case_counts = {
            str(k): _safe_int(v) for k, v in effectiveness_json.get("case_counts", {}).items()
        }
        aggregate_case_source = "effectiveness_json"
    elif parsed_log["case_snapshots"]:
        aggregate_case_counts = {
            str(k): _safe_int(v) for k, v in parsed_log["case_snapshots"][-1].items()
        }
        aggregate_case_source = "log_last_case_counts"
    else:
        aggregate_case_counts = dict(Counter(str(row.get("case")) for row in final_module_rows if row.get("case")))
        aggregate_case_source = "final_module_rows"

    if effectiveness_json and isinstance(effectiveness_json.get("module_status_counts"), dict):
        status_counts = summarize_status_counts(effectiveness_json.get("module_status_counts", {}))
        status_source = "effectiveness_json"
    elif parsed_log["status_snapshots"]:
        status_counts = summarize_status_counts(parsed_log["status_snapshots"][-1])
        status_source = "log_last_module_status_counts"
    else:
        status_counts = {}
        status_source = "unavailable"

    task_rows_for_distance = task_distance_rows(tasks)
    final_distance_source = "result_json" if task_source == "result_json" else "module_log_rounded_values"
    if task_source != "result_json":
        task_rows_for_distance = final_module_rows

    key_to_task = {_task_key(task): task for task in tasks}
    examples = []
    for row in final_module_rows:
        base = _safe_float(row.get("baseline_vp_geo"), default=float("nan"))
        corr = _safe_float(row.get("corrected_vp_geo"), default=float("nan"))
        if not (math.isfinite(base) and math.isfinite(corr)):
            continue
        task = key_to_task.get(_task_key(row), {})
        examples.append(
            {
                "scene_name": row.get("scene_name"),
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id"),
                "task_level": task.get("task_level"),
                "case": row.get("case"),
                "sr": task.get("sr"),
                "spl": task.get("spl"),
                "helpful": task.get("vista_helpful"),
                "baseline_vp_geo": base,
                "corrected_vp_geo": corr,
                "delta_vp_geo": corr - base,
                "line": row.get("_line"),
            }
        )
    examples_sorted = sorted(examples, key=lambda item: item["delta_vp_geo"])

    return {
        "meta": {
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "log_path": str(log_path),
            "result_json_path": None if result_json_path is None else str(result_json_path),
            "effectiveness_json_path": None if effectiveness_json_path is None else str(effectiveness_json_path),
            "task_source": task_source,
            "aggregate_case_source": aggregate_case_source,
            "status_source": status_source,
            "final_distance_source": final_distance_source,
        },
        "log_counts": {
            "module_lines": len(module_rows),
            "final_module_lines": len(final_module_rows),
            "task_summary_lines": len(parsed_log["task_rows"]),
            "case_count_snapshots": len(parsed_log["case_snapshots"]),
            "module_status_snapshots": len(parsed_log["status_snapshots"]),
            "metrics_snapshots": len(parsed_log["metrics_snapshots"]),
            "live_metrics_snapshots": len(parsed_log["live_snapshots"]),
            "last_metrics": parsed_log["metrics_snapshots"][-1] if parsed_log["metrics_snapshots"] else None,
            "last_live_metrics": parsed_log["live_snapshots"][-1] if parsed_log["live_snapshots"] else None,
        },
        "tasks": {
            "overall": summarize_tasks(tasks),
            "by_level": group_task_metrics(tasks, "task_level"),
            "by_case": group_task_metrics(tasks, "vista_case"),
            "by_helpful": group_task_metrics(tasks, "vista_helpful"),
        },
        "vista_module": summarize_modules(module_rows),
        "aggregate_case_effectiveness": case_effectiveness(aggregate_case_counts),
        "module_status_counts": status_counts,
        "final_viewpoint_geo_quality": distance_quality(task_rows_for_distance, "baseline_vp_geo", "corrected_vp_geo"),
        "final_object_l2_quality": distance_quality(task_rows_for_distance, "baseline_obj_l2", "corrected_obj_l2"),
        "top_improved_by_viewpoint_geo": examples_sorted[:top_k],
        "top_worsened_by_viewpoint_geo": list(reversed(examples_sorted[-top_k:])) if top_k > 0 else [],
    }


def _append_task_metric_line(lines: List[str], name: str, metrics: Mapping[str, Any]) -> None:
    n = _safe_int(metrics.get("count"))
    success = _safe_int(metrics.get("success_count"))
    lines.append(
        f"{name:<12} n={n:>3}  SR={100 * _safe_float(metrics.get('sr')):>6.2f}%"
        f" ({success}/{n})  SPL={_safe_float(metrics.get('spl')):.6f}"
        f"  steps={_safe_float(metrics.get('avg_steps')):.1f}"
        f"  decisions={_safe_float(metrics.get('avg_decisions')):.1f}"
    )


def _append_quality_line(lines: List[str], name: str, quality: Mapping[str, Any]) -> None:
    total = _safe_int(quality.get("count"))
    improved = _safe_int(quality.get("improved"))
    worsened = _safe_int(quality.get("worsened"))
    tied = _safe_int(quality.get("tied"))
    lines.append(
        f"{name:<18} improved={improved:>3} ({_pct(improved, total)})"
        f"  worsened={worsened:>3} ({_pct(worsened, total)})"
        f"  tied={tied:>3} ({_pct(tied, total)})"
        f"  mean_delta={_fmt_float(quality.get('mean_delta_finite'), 4)}m"
    )


def _format_example(item: Mapping[str, Any]) -> str:
    return (
        f"scene={item.get('scene_name')} ep={item.get('episode_id')} task={item.get('task_id')}"
        f" level={item.get('task_level')} case={item.get('case')}"
        f" delta={_fmt_float(item.get('delta_vp_geo'), 3)}m"
        f" vp={_fmt_float(item.get('baseline_vp_geo'), 3)}->{_fmt_float(item.get('corrected_vp_geo'), 3)}"
        f" SR={_fmt_float(item.get('sr'), 1)} helpful={item.get('helpful')}"
        f" line={item.get('line')}"
    )


def format_report(report: Mapping[str, Any], top_k: int) -> str:
    lines: List[str] = []
    meta = report["meta"]
    log_counts = report["log_counts"]

    lines.append("== Inputs ==")
    lines.append(f"log={meta['log_path']}")
    lines.append(f"result_json={meta['result_json_path']}")
    lines.append(f"effectiveness_json={meta['effectiveness_json_path']}")
    lines.append(f"task_source={meta['task_source']}  case_source={meta['aggregate_case_source']}")
    lines.append("")

    lines.append("== Parsed log ==")
    lines.append(
        f"module_lines={log_counts['module_lines']} final_module_lines={log_counts['final_module_lines']} "
        f"task_summary_lines={log_counts['task_summary_lines']} "
        f"metrics_snapshots={log_counts['metrics_snapshots']}"
    )
    last_metrics = log_counts.get("last_metrics")
    if isinstance(last_metrics, Mapping):
        lines.append(
            f"last_metrics: count={last_metrics.get('count')} "
            f"avg_sr={_fmt_float(last_metrics.get('avg_sr'))} avg_spl={_fmt_float(last_metrics.get('avg_spl'))}"
        )
    lines.append("")

    lines.append("== Task outcome ==")
    _append_task_metric_line(lines, "overall", report["tasks"]["overall"])
    lines.append("")
    lines.append("By level:")
    for level, metrics in report["tasks"]["by_level"].items():
        _append_task_metric_line(lines, level, metrics)
    lines.append("")

    case_eff = report["aggregate_case_effectiveness"]
    total = _safe_int(case_eff.get("count"))
    lines.append("== VISTA threshold effectiveness (case=baseline/corrected in 1m, GT viewpoint geo) ==")
    lines.append(f"cases: {_format_counter(case_eff['case_counts'], CASE_ORDER)}  total={total}")
    lines.append(
        f"baseline_success={case_eff['baseline_success']}/{total} "
        f"({_pct(case_eff['baseline_success'], total)})  "
        f"corrected_success={case_eff['corrected_success']}/{total} "
        f"({_pct(case_eff['corrected_success'], total)})  "
        f"net={case_eff['net_success_delta']:+d}"
    )
    lines.append(
        f"rescued(01)={case_eff['rescued_01']}  regressed(10)={case_eff['regressed_10']}  "
        f"both_success(11)={case_eff['unchanged_success_11']}  both_fail(00)={case_eff['unchanged_fail_00']}"
    )
    lines.append("")

    lines.append("== Distance quality ==")
    _append_quality_line(lines, "final viewpoint", report["final_viewpoint_geo_quality"])
    _append_quality_line(lines, "final object L2", report["final_object_l2_quality"])
    lines.append("")

    module = report["vista_module"]
    lines.append("== Module status ==")
    lines.append(
        f"called={module['called_count']}/{module['count']}  "
        f"applied={module['correction_applied_count']}/{module['count']}  "
        f"viewpoint_applied={module['viewpoint_applied_count']}/{module['count']}  "
        f"rejected={module['rejected_count']}"
    )
    lines.append(
        f"module cases: {_format_counter(module['case_counts'], CASE_ORDER)}  "
        f"object_l2_cases: {_format_counter(module['object_l2_case_counts'], CASE_ORDER)}"
    )
    lines.append(
        f"efficiency_score mean={_fmt_float(module['efficiency_score_mean'])} "
        f"min={_fmt_float(module['efficiency_score_min'])} max={_fmt_float(module['efficiency_score_max'])}"
    )
    if report["module_status_counts"]:
        status = ", ".join(f"{key}:{value}" for key, value in sorted(report["module_status_counts"].items()))
        lines.append(f"status_counts({meta['status_source']}): {status}")
    lines.append("")

    lines.append("== Task split by case ==")
    for case, metrics in report["tasks"]["by_case"].items():
        _append_task_metric_line(lines, str(case), metrics)
    lines.append("")
    lines.append("== Task split by helpful flag ==")
    for flag, metrics in report["tasks"]["by_helpful"].items():
        _append_task_metric_line(lines, str(flag), metrics)

    if top_k > 0:
        lines.append("")
        lines.append(f"== Top {top_k} improved by viewpoint geo ==")
        for item in report["top_improved_by_viewpoint_geo"]:
            lines.append(_format_example(item))
        lines.append("")
        lines.append(f"== Top {top_k} worsened by viewpoint geo ==")
        for item in report["top_worsened_by_viewpoint_geo"]:
            lines.append(_format_example(item))

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze vista-refine1 module effectiveness from a run log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="vista-refine1 log path")
    parser.add_argument("--result-json", type=Path, default=None, help="optional result JSON path")
    parser.add_argument("--effectiveness-json", type=Path, default=None, help="optional effectiveness JSON path")
    parser.add_argument("--no-auto-json", action="store_true", help="do not auto-load JSON files beside the log")
    parser.add_argument("--json-out", type=Path, default=None, help="optional path for structured JSON report")
    parser.add_argument("--top-k", type=int, default=5, help="number of improved/worsened examples to print")
    args = parser.parse_args()

    log_path = _resolve_path(args.log)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    result_json_path = _resolve_path(args.result_json) if args.result_json is not None else None
    effectiveness_json_path = (
        _resolve_path(args.effectiveness_json) if args.effectiveness_json is not None else None
    )
    if not args.no_auto_json:
        result_json_path = result_json_path or _auto_result_json(log_path)
        effectiveness_json_path = effectiveness_json_path or _auto_effectiveness_json(log_path)

    parsed_log = parse_log(log_path)
    result_json = _load_json(result_json_path)
    effectiveness_json = _load_json(effectiveness_json_path)

    report = build_report(
        log_path=log_path,
        result_json_path=result_json_path,
        effectiveness_json_path=effectiveness_json_path,
        parsed_log=parsed_log,
        result_json=result_json,
        effectiveness_json=effectiveness_json,
        top_k=max(0, int(args.top_k)),
    )
    print(format_report(report, max(0, int(args.top_k))), end="")

    if args.json_out is not None:
        json_out = _resolve_path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[analyze_vista_effectiveness] wrote {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
