#!/usr/bin/env python3
"""
Systematic module-effectiveness analysis for the RefHM3D sequence runs.

This script builds on compare_refhm3d_sequence_experiments.py and focuses on:
  - correction/call/application rates
  - target-distance improvement and 1m case transitions
  - real SR/SPL wins and losses vs Baseline on matched tasks
  - end-to-end task-time overhead vs Baseline
  - module-specific signals for VISTA-LS, MQSC-R1, and Vista2MQSC

Large effectiveness JSON files are not loaded into memory. When enabled, the
script streams only top-level module_info elapsed_ms lines.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from compare_refhm3d_sequence_experiments import (
    DEFAULT_EXPERIMENTS,
    LEVEL_ORDER,
    bool_sr,
    float_value,
    load_experiment,
    project_root,
    resolve_path,
    task_key,
)


CASE_ORDER = ("00", "01", "10", "11", "missing")
SOURCE_TOP_K = 8


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.2f}pp"


def f6(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.6f}"


def f2(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def format_seconds(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.2f}s"


def ratio(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.2f}x"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    sep_line = " | ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rendered]
    return "\n".join([header_line, sep_line, *body])


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def as_bool(value: Any) -> bool:
    return bool_sr(value)


def has_field(record: Dict[str, Any], *names: str) -> bool:
    return any(name in record and record.get(name) is not None for name in names)


def selected_l2(record: Dict[str, Any]) -> Optional[float]:
    for name in ("selected_target_to_goal_l2", "corrected_target_to_goal_l2"):
        if record.get(name) is not None:
            return float_value(record.get(name))
    return None


def baseline_l2(record: Dict[str, Any]) -> Optional[float]:
    if record.get("baseline_target_to_goal_l2") is not None:
        return float_value(record.get("baseline_target_to_goal_l2"))
    return None


def l2_valid(record: Dict[str, Any]) -> bool:
    if record.get("baseline_target_to_goal_l2_valid") is False:
        return False
    if record.get("selected_target_to_goal_l2_valid") is False:
        return False
    return baseline_l2(record) is not None and selected_l2(record) is not None


def distance_delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return before - after


def case_from_distances(before: Optional[float], after: Optional[float], threshold: float = 1.0) -> str:
    if before is None or after is None:
        return "missing"
    return ("1" if before <= threshold else "0") + ("1" if after <= threshold else "0")


def status_counts(records: Sequence[Dict[str, Any]], field: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        if field in record:
            value = record.get(field)
            out[str(value)] += 1
    return out


def top_counter(counter: Counter[str], limit: int = SOURCE_TOP_K) -> str:
    if not counter:
        return "-"
    parts = [f"{key}:{count}" for key, count in counter.most_common(limit)]
    if len(counter) > limit:
        parts.append(f"+{len(counter) - limit} more")
    return ", ".join(parts)


def task_time(record: Dict[str, Any]) -> Optional[float]:
    if record.get("task_time_sec") is None:
        return None
    return float_value(record.get("task_time_sec"))


def task_time_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    values = [task_time(r) for r in records]
    times = [v for v in values if v is not None]
    return {
        "count": len(times),
        "mean": mean(times),
        "median": median(times),
        "p95": percentile(times, 0.95),
        "sum_hours": sum(times) / 3600.0 if times else 0.0,
    }


def field_rate(records: Sequence[Dict[str, Any]], field: str) -> Tuple[int, int, float]:
    available = [r for r in records if field in r]
    count = sum(1 for r in available if as_bool(r.get(field)))
    return count, len(available), count / len(available) if available else 0.0


def module_basic_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    called, called_denom, called_rate = field_rate(records, "module_hook_called")
    applied, applied_denom, applied_rate_tasks = field_rate(records, "module_hook_applied")
    applied_rate_called = applied / called if called else 0.0
    helpful_true = sum(1 for r in records if r.get("module_helpful") is True)
    helpful_false = sum(1 for r in records if r.get("module_helpful") is False)
    helpful_known = helpful_true + helpful_false

    target_valid = [r for r in records if l2_valid(r)]
    pairs: List[Tuple[float, float, float]] = []
    for record in target_valid:
        before = baseline_l2(record)
        after = selected_l2(record)
        delta = distance_delta(before, after)
        if before is not None and after is not None and delta is not None:
            pairs.append((before, after, delta))
    deltas = [delta for _, _, delta in pairs]
    finite_pairs = [(before, after, delta) for before, after, delta in pairs if math.isfinite(before) and math.isfinite(after) and math.isfinite(delta)]
    improved = sum(1 for d in deltas if d > 1e-6)
    worsened = sum(1 for d in deltas if d < -1e-6)
    unchanged = len(deltas) - improved - worsened

    case_counts: Counter[str] = Counter()
    for record in target_valid:
        case_counts[case_from_distances(baseline_l2(record), selected_l2(record))] += 1
    for key in CASE_ORDER:
        case_counts.setdefault(key, 0)

    return {
        "tasks": n,
        "called": called,
        "called_denom": called_denom,
        "called_rate": called_rate,
        "applied": applied,
        "applied_denom": applied_denom,
        "applied_rate_tasks": applied_rate_tasks,
        "applied_rate_called": applied_rate_called,
        "helpful_true": helpful_true,
        "helpful_false": helpful_false,
        "helpful_known": helpful_known,
        "helpful_rate_known": helpful_true / helpful_known if helpful_known else 0.0,
        "target_valid": len(target_valid),
        "target_finite_pairs": len(finite_pairs),
        "target_improved": improved,
        "target_worsened": worsened,
        "target_unchanged": unchanged,
        "target_improved_rate": improved / len(target_valid) if target_valid else 0.0,
        "target_worsened_rate": worsened / len(target_valid) if target_valid else 0.0,
        "mean_target_delta_m": mean([delta for _, _, delta in finite_pairs]),
        "median_target_delta_m": median([delta for _, _, delta in finite_pairs]),
        "mean_baseline_l2_m": mean([before for before, _, _ in finite_pairs]),
        "mean_selected_l2_m": mean([after for _, after, _ in finite_pairs]),
        "target_case_counts": dict(case_counts),
        "fix_01": case_counts["01"],
        "break_10": case_counts["10"],
    }


def matched_records(
    experiment_records: Sequence[Dict[str, Any]],
    baseline_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_by_key = {task_key(record): record for record in baseline_records}
    exp: List[Dict[str, Any]] = []
    base: List[Dict[str, Any]] = []
    for record in experiment_records:
        base_record = baseline_by_key.get(task_key(record))
        if base_record is not None:
            exp.append(record)
            base.append(base_record)
    return exp, base


def impact_vs_baseline(
    exp_records: Sequence[Dict[str, Any]],
    base_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    exp, base = matched_records(exp_records, base_records)
    wins = 0
    losses = 0
    both_success = 0
    both_fail = 0
    spl_deltas: List[float] = []
    time_deltas: List[float] = []
    ratios: List[float] = []
    for e, b in zip(exp, base):
        e_sr = as_bool(e.get("sr"))
        b_sr = as_bool(b.get("sr"))
        if e_sr and not b_sr:
            wins += 1
        elif b_sr and not e_sr:
            losses += 1
        elif e_sr and b_sr:
            both_success += 1
        else:
            both_fail += 1
        spl_deltas.append(float_value(e.get("spl")) - float_value(b.get("spl")))
        e_time = task_time(e)
        b_time = task_time(b)
        if e_time is not None and b_time is not None:
            time_deltas.append(e_time - b_time)
            if b_time > 1e-9:
                ratios.append(e_time / b_time)
    n = len(exp)
    exp_sr = sum(1 for r in exp if as_bool(r.get("sr"))) / n if n else 0.0
    base_sr = sum(1 for r in base if as_bool(r.get("sr"))) / n if n else 0.0
    exp_spl = mean([float_value(r.get("spl")) for r in exp])
    base_spl = mean([float_value(r.get("spl")) for r in base])
    return {
        "tasks": n,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "both_success": both_success,
        "both_fail": both_fail,
        "exp_sr": exp_sr,
        "base_sr": base_sr,
        "sr_delta": exp_sr - base_sr,
        "exp_spl": exp_spl,
        "base_spl": base_spl,
        "spl_delta": exp_spl - base_spl,
        "mean_spl_delta": mean(spl_deltas),
        "median_spl_delta": median(spl_deltas),
        "mean_time_delta_sec": mean(time_deltas),
        "median_time_delta_sec": median(time_deltas),
        "p95_time_delta_sec": percentile(time_deltas, 0.95),
        "mean_time_ratio": mean(ratios),
        "median_time_ratio": median(ratios),
    }


def subset_by_field(records: Sequence[Dict[str, Any]], field: str, desired: bool) -> List[Dict[str, Any]]:
    return [record for record in records if field in record and as_bool(record.get(field)) is desired]


def impact_subsets(exp_records: Sequence[Dict[str, Any]], base_records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    def target_delta(record: Dict[str, Any]) -> Optional[float]:
        if not l2_valid(record):
            return None
        return distance_delta(baseline_l2(record), selected_l2(record))

    subsets = {
        "module_applied": subset_by_field(exp_records, "module_hook_applied", True),
        "module_not_applied": subset_by_field(exp_records, "module_hook_applied", False),
        "target_improved": [r for r in exp_records if (target_delta(r) or 0.0) > 1e-6],
        "target_worsened": [r for r in exp_records if (target_delta(r) or 0.0) < -1e-6],
    }
    return {name: impact_vs_baseline(rows, base_records) for name, rows in subsets.items()}


def by_level_effectiveness(records: Sequence[Dict[str, Any]], base_records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_level[str(record.get("task_level", "_missing"))].append(record)
    out: Dict[str, Dict[str, Any]] = {}
    for level in LEVEL_ORDER:
        if level in by_level:
            out[level] = {
                "basic": module_basic_stats(by_level[level]),
                "impact": impact_vs_baseline(by_level[level], base_records),
                "time": task_time_stats(by_level[level]),
            }
    return out


def module_specific_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for field in (
        "vistals_called",
        "vistals_correction_applied",
        "vistals_viewpoint_applied",
        "vistals_applied",
        "mqsc_r1_called",
        "mqsc_r1_applied",
        "planner_filter_adjustment_applied",
        "planner_filter_prevented_follower_error",
        "planner_filter_unresolved_follower_error_risk",
    ):
        count, denom, rate = field_rate(records, field)
        if denom:
            out[field] = {"count": count, "denom": denom, "rate": rate}

    if any("vistals_case" in r for r in records):
        counter = Counter(str(r.get("vistals_case", "missing")) for r in records)
        out["vistals_case_counts"] = dict(counter)

    if any("module_reason" in r for r in records):
        out["module_reason_counts"] = dict(Counter(str(r.get("module_reason")) for r in records))

    if any("vistals_target_source" in r for r in records):
        out["vistals_target_source_counts"] = dict(Counter(str(r.get("vistals_target_source")) for r in records))

    if any("vistals_input_slot_source" in r for r in records):
        out["vistals_input_slot_source_counts"] = dict(Counter(str(r.get("vistals_input_slot_source")) for r in records))

    viewpoint_records = [
        r
        for r in records
        if r.get("baseline_target_to_gt_viewpoint_geo") is not None
        and r.get("corrected_target_to_gt_viewpoint_geo") is not None
    ]
    if viewpoint_records:
        before = [float_value(r.get("baseline_target_to_gt_viewpoint_geo")) for r in viewpoint_records]
        after = [float_value(r.get("corrected_target_to_gt_viewpoint_geo")) for r in viewpoint_records]
        pairs = [(b, a, b - a) for b, a in zip(before, after)]
        finite_pairs = [(b, a, d) for b, a, d in pairs if math.isfinite(b) and math.isfinite(a) and math.isfinite(d)]
        deltas = [d for _, _, d in pairs]
        cases: Counter[str] = Counter(case_from_distances(b, a) for b, a in zip(before, after))
        out["viewpoint_geo"] = {
            "count": len(viewpoint_records),
            "finite_pairs": len(finite_pairs),
            "mean_before_m": mean([b for b, _, _ in finite_pairs]),
            "mean_after_m": mean([a for _, a, _ in finite_pairs]),
            "mean_delta_m": mean([d for _, _, d in finite_pairs]),
            "median_delta_m": median([d for _, _, d in finite_pairs]),
            "improved": sum(1 for d in deltas if d > 1e-6),
            "worsened": sum(1 for d in deltas if d < -1e-6),
            "case_counts": dict(cases),
        }

    return out


def effectiveness_json_paths(root: Path, experiment: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for shard in experiment["shards"]:
        json_path = resolve_path(root, shard["json_path"])
        candidates = sorted(json_path.parent.glob("*effectiveness*.json"))
        paths.extend(candidates)
    return paths


def stream_top_level_elapsed_ms(path: Path) -> List[float]:
    """Read module_info elapsed_ms only, avoiding nested MQSC/VISTA-LS elapsed fields."""
    elapsed: List[float] = []
    pattern = re.compile(r'^\s{12}"elapsed_ms":\s*([0-9.eE+-]+)')
    if not path.is_file():
        return elapsed
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                elapsed.append(float(match.group(1)))
    return elapsed


def elapsed_stats(paths: Sequence[Path]) -> Dict[str, Any]:
    values: List[float] = []
    scanned: List[str] = []
    for path in paths:
        elapsed = stream_top_level_elapsed_ms(path)
        if elapsed:
            values.extend(elapsed)
            scanned.append(str(path))
    seconds = [v / 1000.0 for v in values]
    return {
        "count": len(seconds),
        "mean_sec": mean(seconds),
        "median_sec": median(seconds),
        "p95_sec": percentile(seconds, 0.95),
        "sum_hours": sum(seconds) / 3600.0 if seconds else 0.0,
        "scanned_files": scanned,
    }


def build_analysis(root: Path, tasks_per_episode: int, scan_elapsed: bool) -> Dict[str, Any]:
    experiments = [load_experiment(root, spec, tasks_per_episode) for spec in DEFAULT_EXPERIMENTS]
    baseline = experiments[0]
    base_records = baseline["records"]

    analyses: List[Dict[str, Any]] = []
    for experiment in experiments:
        records = experiment["records"]
        analysis = {
            "name": experiment["name"],
            "status": experiment["status"],
            "tasks": experiment["tasks"],
            "expected_tasks": experiment["expected_tasks"],
            "basic": module_basic_stats(records),
            "time": task_time_stats(records),
            "impact_vs_baseline": impact_vs_baseline(records, base_records),
            "impact_subsets_vs_baseline": impact_subsets(records, base_records),
            "by_level": by_level_effectiveness(records, base_records),
            "specific": module_specific_stats(records),
            "elapsed_ms_from_effectiveness": {"count": 0, "mean_sec": float("nan"), "median_sec": float("nan"), "p95_sec": float("nan"), "sum_hours": 0.0, "scanned_files": []},
        }
        if scan_elapsed and experiment["name"] in {"MQSC-R1", "Vista2MQSC"}:
            analysis["elapsed_ms_from_effectiveness"] = elapsed_stats(effectiveness_json_paths(root, experiment))
        analyses.append(analysis)

    return {
        "root": str(root),
        "tasks_per_episode": tasks_per_episode,
        "scan_elapsed": scan_elapsed,
        "experiments": analyses,
    }


def metric_design_text() -> str:
    return "\n".join(
        [
            "Brainstormed Analysis Axes",
            "- coverage: did the module get called often enough to matter, and did it actually apply a change",
            "- correction quality: among valid target-distance pairs, did the proposed target move closer or farther from the goal",
            "- threshold correction: did the module convert target distance across the 1m success-relevant boundary, including 00/01/10/11 cases",
            "- real outcome impact: on exactly matched tasks, how often did the module turn a Baseline failure into success or a success into failure",
            "- negative correction: target_worsened, SR_losses, and 10 cases are tracked separately because average gains can hide regressions",
            "- runtime cost: compare end-to-end task_time_sec against Baseline and, where available, stream module_info elapsed_ms from effectiveness logs",
            "- level sensitivity: repeat the above by object/room/region/instance to find where the module is useful or risky",
            "- routing/composition: for Vista2MQSC, separate MQSC-R1 semantic target usage from VISTA-LS viewpoint usage through module_reason and source fields",
            "",
            "Metric Design",
            "- correction call rate: module_hook_called / tasks",
            "- correction application rate: module_hook_applied / tasks and / called",
            "- target distance delta: baseline_target_to_goal_l2 - selected/corrected_target_to_goal_l2; positive means the module moved target closer to goal",
            "- Infinity distances are counted for fix/break/improve/worsen decisions, but finite-distance means use only finite before/after pairs",
            "- target 1m case: 00 outside->outside, 01 outside->inside fix, 10 inside->outside break, 11 inside->inside keep",
            "- real task impact: matched-task SR win/loss vs Baseline, plus SPL/time deltas",
            "- time: end-to-end task_time_sec is the trusted comparable runtime; MQSC-R1/Vista2MQSC module_info elapsed_ms is streamed from effectiveness JSON when available",
            "- VISTA-LS partial note: 0.2-1.0 hit OOM, so its combined rows cover 2436/3600 tasks and are not a complete full-set estimate",
        ]
    )


def build_markdown(payload: Dict[str, Any]) -> str:
    experiments = payload["experiments"]
    lines: List[str] = []
    lines.append(f"project_root={payload['root']}")
    lines.append(f"tasks_per_episode={payload['tasks_per_episode']}")
    lines.append(f"scan_elapsed={payload['scan_elapsed']}")
    lines.append("")
    lines.append(metric_design_text())
    lines.append("")

    overview_rows: List[List[str]] = []
    for exp in experiments:
        basic = exp["basic"]
        impact = exp["impact_vs_baseline"]
        time = exp["time"]
        elapsed = exp["elapsed_ms_from_effectiveness"]
        overview_rows.append(
            [
                exp["name"],
                exp["status"],
                f"{exp['tasks']}/{exp['expected_tasks']}",
                pct(basic["called_rate"]),
                pct(basic["applied_rate_tasks"]),
                pct(basic["applied_rate_called"]),
                pct(basic["helpful_rate_known"]) if basic["helpful_known"] else "NA",
                pct(basic["target_improved_rate"]),
                pct(basic["target_worsened_rate"]),
                f6(basic["mean_target_delta_m"]),
                str(impact["wins"]),
                str(impact["losses"]),
                str(impact["net_wins"]),
                pp(impact["sr_delta"]),
                f6(impact["spl_delta"]),
                format_seconds(time["mean"]),
                format_seconds(impact["mean_time_delta_sec"]),
                ratio(impact["mean_time_ratio"]),
                format_seconds(elapsed["mean_sec"]),
            ]
        )
    lines.append("Module Effectiveness Overview")
    lines.append(
        format_table(
            (
                "experiment",
                "status",
                "tasks",
                "call",
                "apply/tasks",
                "apply/called",
                "helpful",
                "target+",
                "target-",
                "finite_mean_delta_m",
                "SR_wins",
                "SR_losses",
                "net",
                "dSR",
                "dSPL",
                "avg_time",
                "dtime",
                "time_ratio",
                "module_elapsed",
            ),
            overview_rows,
        )
    )
    lines.append("")

    case_rows: List[List[str]] = []
    for exp in experiments:
        counts = exp["basic"]["target_case_counts"]
        case_rows.append(
            [
                exp["name"],
                str(exp["basic"]["target_valid"]),
                str(exp["basic"]["target_finite_pairs"]),
                *(str(counts.get(case_name, 0)) for case_name in CASE_ORDER),
                str(exp["basic"]["fix_01"]),
                str(exp["basic"]["break_10"]),
                f6(exp["basic"]["mean_baseline_l2_m"]),
                f6(exp["basic"]["mean_selected_l2_m"]),
            ]
        )
    lines.append("Target 1m Case And Distance")
    lines.append(
        format_table(
            ("experiment", "valid", "finite_pairs", *CASE_ORDER, "fix_01", "break_10", "finite_mean_before_l2", "finite_mean_after_l2"),
            case_rows,
        )
    )
    lines.append("")

    subset_rows: List[List[str]] = []
    for exp in experiments:
        if exp["name"] == "Baseline":
            continue
        for subset_name, impact in exp["impact_subsets_vs_baseline"].items():
            subset_rows.append(
                [
                    exp["name"],
                    subset_name,
                    str(impact["tasks"]),
                    pp(impact["sr_delta"]),
                    f6(impact["spl_delta"]),
                    str(impact["wins"]),
                    str(impact["losses"]),
                    str(impact["net_wins"]),
                    format_seconds(impact["mean_time_delta_sec"]),
                    ratio(impact["mean_time_ratio"]),
                ]
            )
    lines.append("Matched-Task Impact By Subset")
    lines.append(
        format_table(
            ("experiment", "subset", "tasks", "dSR", "dSPL", "wins", "losses", "net", "dtime", "time_ratio"),
            subset_rows,
        )
    )
    lines.append("")

    time_rows: List[List[str]] = []
    for exp in experiments:
        impact = exp["impact_vs_baseline"]
        time = exp["time"]
        elapsed = exp["elapsed_ms_from_effectiveness"]
        time_rows.append(
            [
                exp["name"],
                str(time["count"]),
                format_seconds(time["mean"]),
                format_seconds(time["median"]),
                format_seconds(time["p95"]),
                f2(time["sum_hours"]),
                format_seconds(impact["mean_time_delta_sec"]),
                format_seconds(impact["median_time_delta_sec"]),
                format_seconds(impact["p95_time_delta_sec"]),
                ratio(impact["mean_time_ratio"]),
                str(elapsed["count"]),
                format_seconds(elapsed["mean_sec"]),
                format_seconds(elapsed["median_sec"]),
                format_seconds(elapsed["p95_sec"]),
                f2(elapsed["sum_hours"]),
            ]
        )
    lines.append("Time Cost")
    lines.append(
        format_table(
            (
                "experiment",
                "tasks",
                "avg_task",
                "med_task",
                "p95_task",
                "sum_h",
                "avg_dtime",
                "med_dtime",
                "p95_dtime",
                "avg_ratio",
                "elapsed_n",
                "elapsed_avg",
                "elapsed_med",
                "elapsed_p95",
                "elapsed_h",
            ),
            time_rows,
        )
    )
    lines.append("")

    level_rows: List[List[str]] = []
    for exp in experiments:
        for level in LEVEL_ORDER:
            level_data = exp["by_level"].get(level)
            if not level_data:
                continue
            basic = level_data["basic"]
            impact = level_data["impact"]
            time = level_data["time"]
            level_rows.append(
                [
                    exp["name"],
                    level,
                    str(basic["tasks"]),
                    pct(basic["applied_rate_tasks"]),
                    pct(basic["target_improved_rate"]),
                    pct(basic["target_worsened_rate"]),
                    f6(basic["mean_target_delta_m"]),
                    pp(impact["sr_delta"]),
                    f6(impact["spl_delta"]),
                    str(impact["wins"]),
                    str(impact["losses"]),
                    format_seconds(time["mean"]),
                    format_seconds(impact["mean_time_delta_sec"]),
                ]
            )
    lines.append("By Level Effectiveness")
    lines.append(
        format_table(
            ("experiment", "level", "tasks", "apply", "target+", "target-", "finite_mean_delta_m", "dSR", "dSPL", "wins", "losses", "avg_time", "dtime"),
            level_rows,
        )
    )
    lines.append("")

    specific_rows: List[List[str]] = []
    for exp in experiments:
        spec = exp["specific"]
        fields = [
            "vistals_called",
            "vistals_correction_applied",
            "vistals_viewpoint_applied",
            "vistals_applied",
            "mqsc_r1_called",
            "mqsc_r1_applied",
            "planner_filter_adjustment_applied",
            "planner_filter_prevented_follower_error",
            "planner_filter_unresolved_follower_error_risk",
        ]
        values = []
        for field in fields:
            item = spec.get(field)
            values.append(f"{item['count']}/{item['denom']} ({pct(item['rate'])})" if item else "-")
        specific_rows.append([exp["name"], *values])
    lines.append("Module-Specific Rates")
    lines.append(format_table(("experiment", "vistals_called", "vistals_corr", "vistals_vp", "vistals_applied", "mqsc_called", "mqsc_applied", "planner_adj", "planner_prevent", "planner_unresolved"), specific_rows))
    lines.append("")

    dist_rows: List[List[str]] = []
    for exp in experiments:
        spec = exp["specific"]
        if spec.get("vistals_case_counts"):
            dist_rows.append([exp["name"], "vistals_case", top_counter(Counter(spec["vistals_case_counts"]))])
        if spec.get("module_reason_counts"):
            dist_rows.append([exp["name"], "module_reason", top_counter(Counter(spec["module_reason_counts"]))])
        if spec.get("vistals_target_source_counts"):
            dist_rows.append([exp["name"], "vistals_target_source", top_counter(Counter(spec["vistals_target_source_counts"]))])
        if spec.get("vistals_input_slot_source_counts"):
            dist_rows.append([exp["name"], "vistals_input_slot_source", top_counter(Counter(spec["vistals_input_slot_source_counts"]))])
        if spec.get("viewpoint_geo"):
            vp = spec["viewpoint_geo"]
            dist_rows.append(
                [
                    exp["name"],
                    "viewpoint_geo",
                    f"n={vp['count']}, finite_pairs={vp['finite_pairs']}, improved={vp['improved']}, worsened={vp['worsened']}, mean_before={f6(vp['mean_before_m'])}, mean_after={f6(vp['mean_after_m'])}, mean_delta={f6(vp['mean_delta_m'])}, cases={top_counter(Counter(vp['case_counts']))}",
                ]
            )
    lines.append("Distributions And VISTA-LS Viewpoint Signals")
    lines.append(format_table(("experiment", "signal", "distribution"), dist_rows))
    lines.append("")

    elapsed_rows = []
    for exp in experiments:
        elapsed = exp["elapsed_ms_from_effectiveness"]
        for path in elapsed.get("scanned_files", []):
            elapsed_rows.append([exp["name"], path])
    if elapsed_rows:
        lines.append("Effectiveness JSON Files Scanned For elapsed_ms")
        lines.append(format_table(("experiment", "path"), elapsed_rows))

    return "\n".join(lines)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze module effectiveness for RefHM3D sequence runs.")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--tasks-per-episode", type=int, default=5)
    parser.add_argument("--scan-effectiveness-elapsed", action="store_true", help="Stream effectiveness JSON files to summarize module_info elapsed_ms.")
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    payload = build_analysis(root, args.tasks_per_episode, args.scan_effectiveness_elapsed)
    markdown = build_markdown(payload)
    print(markdown)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n", encoding="utf-8")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
