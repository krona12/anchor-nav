#!/usr/bin/env python3
"""
Compare completed SAM2.1 detailed single-goal shards against baseline detailed.

Default scope:
- baseline: output_logs/anchor/single_goal/baseline/detailed-full/refhm3d_single_goal_*.json
- ours: output_logs/anchor/single_goal/vista2mqsc_sam2/balanced-detailed-qwen_vl_plus_full/refhm3d_single_goal_vista2mqsc_refine1_*.json

The script aligns tasks on the same sample key, computes SR/SPL on the common
set, projects a global head-vs-long-tail object-category split onto that set,
and writes JSON/JSONL/Markdown outputs under stastics_scripts/logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_LEVELS = ("object", "room", "region", "instance")
EPS = 1e-6

DEFAULT_BASELINE_PATTERN = (
    "output_logs/anchor/single_goal/baseline/detailed-full/"
    "refhm3d_single_goal_*.json"
)
DEFAULT_OURS_PATTERN = (
    "output_logs/anchor/single_goal/vista2mqsc_sam2/"
    "balanced-detailed-qwen_vl_plus_full/"
    "refhm3d_single_goal_vista2mqsc_refine1_*.json"
)
DEFAULT_CATEGORY_ROOT = "LangMap_Annotations"
DEFAULT_OUTPUT_DIR = "stastics_scripts/logs"
DEFAULT_TAG = "sam2_1_detailed_vs_baseline_detailed_common"


JsonDict = Dict[str, Any]
RecordKey = Tuple[str, str, str, int, int]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def success_value(record: Mapping[str, Any]) -> bool:
    return is_true(record.get("sr"))


def spl_value(record: Mapping[str, Any]) -> float:
    return safe_float(record.get("spl"), 0.0)


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def pct(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{100.0 * value:+.2f} pp"


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def ratio(numer: int, denom: int) -> float:
    return numer / denom if denom else float("nan")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def should_skip_result_file(path: Path, include_concise: bool) -> bool:
    name = path.name.lower()
    if "effectiveness" in name:
        return True
    if not include_concise and "concisedesc" in name:
        return True
    return False


def expand_patterns(patterns: Sequence[str], include_concise: bool) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if not path.is_file() or should_skip_result_file(path, include_concise):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def parse_records(path: Path) -> List[JsonDict]:
    data = load_json(path)
    records: List[JsonDict] = []
    if isinstance(data, list):
        raw_records = data
    elif isinstance(data, dict):
        raw_records = data.get("sequence")
        if not isinstance(raw_records, list):
            raw_records = []
            for level in TASK_LEVELS:
                value = data.get(level)
                if isinstance(value, list):
                    raw_records.extend(value)
    else:
        raw_records = []

    for row in raw_records:
        if not isinstance(row, dict):
            continue
        if row.get("valid_for_metric", True) is False:
            continue
        record = dict(row)
        record["_source_file"] = str(path.relative_to(PROJECT_ROOT))
        records.append(record)
    return records


def record_level(record: Mapping[str, Any]) -> str:
    level = record.get("task_level") or record.get("navigation_type") or "_missing"
    return str(level)


def record_key(record: Mapping[str, Any]) -> RecordKey:
    level = record_level(record)
    navigation_type = str(record.get("navigation_type") or level)
    scene_name = str(record.get("scene_name") or "")
    episode_id = safe_int(record.get("episode_id"))
    task_id = safe_int(record.get("task_id"))
    return (level, navigation_type, scene_name, episode_id, task_id)


def key_to_string(key: RecordKey) -> str:
    level, navigation_type, scene_name, episode_id, task_id = key
    return f"{level}|{navigation_type}|{scene_name}|ep{episode_id}|task{task_id}"


def metric_signature(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        record.get("sr"),
        record.get("spl"),
        record.get("end_reason"),
        record.get("object_category"),
        record.get("task_level"),
        record.get("navigation_type"),
    )


def load_index(files: Sequence[Path]) -> Tuple[Dict[RecordKey, JsonDict], JsonDict]:
    index: Dict[RecordKey, JsonDict] = {}
    duplicates: List[JsonDict] = []
    record_counts: Dict[str, int] = {}

    for path in files:
        records = parse_records(path)
        record_counts[str(path.relative_to(PROJECT_ROOT))] = len(records)
        for record in records:
            key = record_key(record)
            if key in index:
                old = index[key]
                duplicates.append(
                    {
                        "key": key_to_string(key),
                        "kept_source": old.get("_source_file"),
                        "dropped_source": record.get("_source_file"),
                        "same_metric_signature": metric_signature(old) == metric_signature(record),
                    }
                )
                continue
            index[key] = record

    meta = {
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "file_count": len(files),
        "record_counts": record_counts,
        "indexed_record_count": len(index),
        "duplicate_count": len(duplicates),
        "duplicate_examples": duplicates[:20],
    }
    return index, meta


def task_category_from_annotation(
    task: Mapping[str, Any],
    goals_by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    category = task.get("object_category")
    if category:
        return str(category)
    target_ids = task.get("target_object_ids")
    if isinstance(target_ids, list):
        for target_id in target_ids:
            goal = goals_by_id.get(str(target_id))
            if goal and goal.get("object_category"):
                return str(goal["object_category"])
    instance_id = task.get("instance_id")
    if instance_id:
        goal = goals_by_id.get(str(instance_id))
        if goal and goal.get("object_category"):
            return str(goal["object_category"])
    return None


def load_global_category_counts(category_root: Path) -> Tuple[Counter[str], JsonDict]:
    counts: Counter[str] = Counter()
    scene_files = sorted(category_root.rglob("*.json.gz"))
    level_counts: Counter[str] = Counter()
    missing_category = 0

    for path in scene_files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene_data = json.load(handle)
        goals_by_id = {
            str(goal.get("object_id")): goal
            for goal in scene_data.get("goals", [])
            if isinstance(goal, dict)
        }
        level_to_key = {
            "object": "episodes_by_object_level",
            "room": "episodes_by_room_level",
            "region": "episodes_by_region_level",
            "instance": "episodes_by_instance_level",
        }
        for level, key in level_to_key.items():
            episodes = scene_data.get(key, [])
            if not isinstance(episodes, list):
                continue
            for task in episodes:
                if not isinstance(task, dict):
                    continue
                category = task_category_from_annotation(task, goals_by_id)
                if category:
                    counts[category] += 1
                    level_counts[level] += 1
                else:
                    missing_category += 1

    meta = {
        "source": str(category_root.relative_to(PROJECT_ROOT))
        if category_root.is_relative_to(PROJECT_ROOT)
        else str(category_root),
        "scene_file_count": len(scene_files),
        "task_count": sum(counts.values()),
        "level_counts": dict(level_counts),
        "category_count": len(counts),
        "missing_category_count": missing_category,
    }
    return counts, meta


def fallback_category_counts(records: Iterable[Mapping[str, Any]]) -> Tuple[Counter[str], JsonDict]:
    counts: Counter[str] = Counter()
    for record in records:
        category = record.get("object_category")
        if category:
            counts[str(category)] += 1
    return counts, {
        "source": "fallback: loaded baseline detailed records",
        "task_count": sum(counts.values()),
        "category_count": len(counts),
        "missing_category_count": 0,
    }


def build_head_tail_split(
    counts: Counter[str],
    fraction: float,
) -> Tuple[set[str], JsonDict]:
    if not counts:
        return set(), {
            "fraction": fraction,
            "category_count": 0,
            "head_category_count": 0,
            "head_task_count": 0,
            "long_tail_task_count": 0,
            "head_task_coverage": float("nan"),
            "top_categories": [],
        }

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    head_count = max(1, math.ceil(len(ordered) * fraction))
    head_categories = {category for category, _ in ordered[:head_count]}
    head_tasks = sum(counts[category] for category in head_categories)
    total_tasks = sum(counts.values())
    meta = {
        "fraction": fraction,
        "category_count": len(ordered),
        "head_category_count": head_count,
        "long_tail_category_count": len(ordered) - head_count,
        "task_count": total_tasks,
        "head_task_count": head_tasks,
        "long_tail_task_count": total_tasks - head_tasks,
        "head_task_coverage": head_tasks / total_tasks if total_tasks else float("nan"),
        "long_tail_task_coverage": (total_tasks - head_tasks) / total_tasks if total_tasks else float("nan"),
        "top_categories": [{"category": category, "count": count} for category, count in ordered[:20]],
    }
    return head_categories, meta


def finite_distance_pair(record: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    if record.get("baseline_target_to_goal_l2_valid") is False:
        return None
    if record.get("selected_target_to_goal_l2_valid") is False:
        return None
    before = safe_float(record.get("baseline_target_to_goal_l2"), float("nan"))
    after = safe_float(record.get("selected_target_to_goal_l2"), float("nan"))
    if not (math.isfinite(before) and math.isfinite(after)):
        return None
    return before, after


def threshold_case(before: float, after: float, threshold: float) -> str:
    return ("1" if before <= threshold else "0") + ("1" if after <= threshold else "0")


def build_paired_rows(
    baseline_index: Mapping[RecordKey, JsonDict],
    ours_index: Mapping[RecordKey, JsonDict],
    head_categories: set[str],
    target_threshold_m: float,
) -> List[JsonDict]:
    common_keys = sorted(set(baseline_index).intersection(ours_index))
    rows: List[JsonDict] = []
    for key in common_keys:
        baseline = baseline_index[key]
        ours = ours_index[key]
        category = str(ours.get("object_category") or baseline.get("object_category") or "_missing")
        bucket = "head" if category in head_categories else "long_tail"
        if category == "_missing":
            bucket = "unknown"

        baseline_sr = success_value(baseline)
        ours_sr = success_value(ours)
        dist_pair = finite_distance_pair(ours)
        target_delta = None
        target_case = None
        target_delta_label = "missing"
        if dist_pair is not None:
            before, after = dist_pair
            target_delta = before - after
            target_case = threshold_case(before, after, target_threshold_m)
            if target_delta > EPS:
                target_delta_label = "improved"
            elif target_delta < -EPS:
                target_delta_label = "worsened"
            else:
                target_delta_label = "tied"

        rows.append(
            {
                "key": key_to_string(key),
                "task_level": key[0],
                "navigation_type": key[1],
                "scene_name": key[2],
                "episode_id": key[3],
                "task_id": key[4],
                "object_category": category,
                "category_bucket": bucket,
                "baseline_sr": baseline_sr,
                "baseline_spl": spl_value(baseline),
                "baseline_end_reason": baseline.get("end_reason"),
                "baseline_steps_total": baseline.get("steps_total"),
                "baseline_task_time_sec": baseline.get("task_time_sec"),
                "ours_sr": ours_sr,
                "ours_spl": spl_value(ours),
                "ours_end_reason": ours.get("end_reason"),
                "ours_steps_total": ours.get("steps_total"),
                "ours_task_time_sec": ours.get("task_time_sec"),
                "outcome_case": (
                    ("1" if baseline_sr else "0") + ("1" if ours_sr else "0")
                ),
                "sr_delta": int(ours_sr) - int(baseline_sr),
                "spl_delta": spl_value(ours) - spl_value(baseline),
                "module_hook_called": safe_int(ours.get("module_hook_called")),
                "module_hook_applied": safe_int(ours.get("module_hook_applied")),
                "module_helpful": ours.get("module_helpful"),
                "module_reason": ours.get("module_reason"),
                "mqsc_r1_applied": is_true(ours.get("mqsc_r1_applied")),
                "vistals_applied": is_true(ours.get("vistals_applied")),
                "vistals_input_slot_source": ours.get("vistals_input_slot_source"),
                "baseline_target_to_goal_l2": ours.get("baseline_target_to_goal_l2"),
                "selected_target_to_goal_l2": ours.get("selected_target_to_goal_l2"),
                "target_distance_delta_m": target_delta,
                "target_distance_delta_label": target_delta_label,
                "target_threshold_case": target_case,
                "baseline_source_file": baseline.get("_source_file"),
                "ours_source_file": ours.get("_source_file"),
            }
        )
    return rows


def method_metrics(rows: Sequence[Mapping[str, Any]], prefix: str) -> JsonDict:
    n = len(rows)
    sr_field = f"{prefix}_sr"
    spl_field = f"{prefix}_spl"
    successes = sum(1 for row in rows if row.get(sr_field) is True)
    return {
        "n": n,
        "success_count": successes,
        "sr": successes / n if n else float("nan"),
        "spl": mean(safe_float(row.get(spl_field), 0.0) for row in rows),
    }


def outcome_flip_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n = len(rows)
    counts = Counter(str(row.get("outcome_case")) for row in rows)
    both_wrong = counts["00"]
    rescued = counts["01"]
    regressed = counts["10"]
    both_correct = counts["11"]
    baseline_wrong = both_wrong + rescued
    baseline_correct = regressed + both_correct
    ours_wrong = both_wrong + regressed
    ours_correct = rescued + both_correct
    return {
        "n": n,
        "case_counts": {
            "00_both_wrong": both_wrong,
            "01_baseline_wrong_ours_correct": rescued,
            "10_baseline_correct_ours_wrong": regressed,
            "11_both_correct": both_correct,
        },
        "baseline_wrong_count": baseline_wrong,
        "baseline_correct_count": baseline_correct,
        "ours_wrong_count": ours_wrong,
        "ours_correct_count": ours_correct,
        "correction_rate_over_baseline_wrong": ratio(rescued, baseline_wrong),
        "regression_rate_over_baseline_correct": ratio(regressed, baseline_correct),
        "net_success_delta_count": rescued - regressed,
        "net_sr_delta": ratio(rescued - regressed, n),
    }


def target_correction_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    valid = [row for row in rows if row.get("target_distance_delta_m") is not None]
    applied_valid = [row for row in valid if safe_int(row.get("module_hook_applied")) > 0]
    deltas = [safe_float(row.get("target_distance_delta_m")) for row in valid]
    applied_deltas = [safe_float(row.get("target_distance_delta_m")) for row in applied_valid]
    label_counts = Counter(str(row.get("target_distance_delta_label")) for row in valid)
    applied_label_counts = Counter(str(row.get("target_distance_delta_label")) for row in applied_valid)
    case_counts = Counter(str(row.get("target_threshold_case")) for row in valid if row.get("target_threshold_case") is not None)
    for case in ("00", "01", "10", "11"):
        case_counts.setdefault(case, 0)

    outside_before = case_counts["00"] + case_counts["01"]
    inside_before = case_counts["10"] + case_counts["11"]
    return {
        "n": len(rows),
        "valid_l2_count": len(valid),
        "missing_l2_count": len(rows) - len(valid),
        "module_applied_valid_l2_count": len(applied_valid),
        "improved_count": label_counts["improved"],
        "worsened_count": label_counts["worsened"],
        "tied_count": label_counts["tied"],
        "improved_rate_over_valid_l2": ratio(label_counts["improved"], len(valid)),
        "worsened_rate_over_valid_l2": ratio(label_counts["worsened"], len(valid)),
        "applied_improved_count": applied_label_counts["improved"],
        "applied_worsened_count": applied_label_counts["worsened"],
        "applied_tied_count": applied_label_counts["tied"],
        "applied_improved_rate": ratio(applied_label_counts["improved"], len(applied_valid)),
        "applied_worsened_rate": ratio(applied_label_counts["worsened"], len(applied_valid)),
        "mean_target_distance_delta_m": mean(deltas),
        "median_target_distance_delta_m": median(deltas),
        "mean_applied_target_distance_delta_m": mean(applied_deltas),
        "target_threshold_case_counts": {
            "00_both_outside": case_counts["00"],
            "01_baseline_outside_selected_inside": case_counts["01"],
            "10_baseline_inside_selected_outside": case_counts["10"],
            "11_both_inside": case_counts["11"],
        },
        "threshold_fix_rate_over_baseline_outside": ratio(case_counts["01"], outside_before),
        "threshold_break_rate_over_baseline_inside": ratio(case_counts["10"], inside_before),
    }


def comparison_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    baseline = method_metrics(rows, "baseline")
    ours = method_metrics(rows, "ours")
    return {
        "n": len(rows),
        "baseline": baseline,
        "ours": ours,
        "delta": {
            "sr": ours["sr"] - baseline["sr"] if rows else float("nan"),
            "spl": ours["spl"] - baseline["spl"] if rows else float("nan"),
            "success_count": ours["success_count"] - baseline["success_count"],
        },
        "outcome_flips": outcome_flip_summary(rows),
        "target_correction": target_correction_summary(rows),
    }


def group_rows(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "_missing")].append(row)
    return grouped


def ordered_group_summary(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    preferred_order: Sequence[str] = (),
) -> JsonDict:
    grouped = group_rows(rows, field)
    keys = [key for key in preferred_order if key in grouped]
    keys.extend(sorted(set(grouped) - set(keys)))
    return {key: comparison_summary(grouped[key]) for key in keys}


def nested_level_bucket_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    by_level = group_rows(rows, "task_level")
    for level in [*TASK_LEVELS, *sorted(set(by_level) - set(TASK_LEVELS))]:
        if level not in by_level:
            continue
        out[level] = ordered_group_summary(by_level[level], "category_bucket", ("head", "long_tail", "unknown"))
    return out


def top_outcome_examples(rows: Sequence[Mapping[str, Any]], case: str, limit: int) -> List[JsonDict]:
    filtered = [row for row in rows if row.get("outcome_case") == case]
    filtered.sort(key=lambda row: abs(safe_float(row.get("spl_delta"), 0.0)), reverse=True)
    return [
        {
            "key": row.get("key"),
            "task_level": row.get("task_level"),
            "object_category": row.get("object_category"),
            "baseline_spl": row.get("baseline_spl"),
            "ours_spl": row.get("ours_spl"),
            "spl_delta": row.get("spl_delta"),
            "module_reason": row.get("module_reason"),
            "target_distance_delta_m": row.get("target_distance_delta_m"),
            "baseline_end_reason": row.get("baseline_end_reason"),
            "ours_end_reason": row.get("ours_end_reason"),
        }
        for row in filtered[:limit]
    ]


def top_distance_examples(rows: Sequence[Mapping[str, Any]], label: str, limit: int) -> List[JsonDict]:
    filtered = [row for row in rows if row.get("target_distance_delta_label") == label]
    reverse = label == "improved"
    filtered.sort(key=lambda row: safe_float(row.get("target_distance_delta_m"), 0.0), reverse=reverse)
    return [
        {
            "key": row.get("key"),
            "task_level": row.get("task_level"),
            "object_category": row.get("object_category"),
            "target_distance_delta_m": row.get("target_distance_delta_m"),
            "baseline_target_to_goal_l2": row.get("baseline_target_to_goal_l2"),
            "selected_target_to_goal_l2": row.get("selected_target_to_goal_l2"),
            "module_reason": row.get("module_reason"),
            "outcome_case": row.get("outcome_case"),
        }
        for row in filtered[:limit]
    ]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def summary_metric_row(name: str, stats: Mapping[str, Any]) -> List[str]:
    baseline = stats["baseline"]
    ours = stats["ours"]
    delta = stats["delta"]
    return [
        name,
        str(stats["n"]),
        f"{pct(baseline['sr'])} ({baseline['success_count']}/{baseline['n']})",
        fmt_float(baseline["spl"], 4),
        f"{pct(ours['sr'])} ({ours['success_count']}/{ours['n']})",
        fmt_float(ours["spl"], 4),
        pp(delta["sr"]),
        fmt_float(delta["spl"], 4),
    ]


def correction_row(name: str, stats: Mapping[str, Any]) -> List[str]:
    flips = stats["outcome_flips"]
    cases = flips["case_counts"]
    return [
        name,
        str(stats["n"]),
        str(cases["01_baseline_wrong_ours_correct"]),
        pct(flips["correction_rate_over_baseline_wrong"]),
        str(cases["10_baseline_correct_ours_wrong"]),
        pct(flips["regression_rate_over_baseline_correct"]),
        str(flips["net_success_delta_count"]),
        pp(flips["net_sr_delta"]),
    ]


def target_row(name: str, stats: Mapping[str, Any]) -> List[str]:
    target = stats["target_correction"]
    cases = target["target_threshold_case_counts"]
    return [
        name,
        str(target["valid_l2_count"]),
        f"{target['improved_count']} ({pct(target['improved_rate_over_valid_l2'])})",
        f"{target['worsened_count']} ({pct(target['worsened_rate_over_valid_l2'])})",
        fmt_float(target["mean_target_distance_delta_m"], 3),
        fmt_float(target["median_target_distance_delta_m"], 3),
        str(cases["01_baseline_outside_selected_inside"]),
        str(cases["10_baseline_inside_selected_outside"]),
    ]


def build_markdown(payload: Mapping[str, Any]) -> str:
    meta = payload["meta"]
    category = payload["category_split"]
    common = payload["common_sample"]
    overall = payload["metrics"]["overall"]
    by_level = payload["metrics"]["by_level"]
    by_bucket = payload["metrics"]["by_category_bucket"]

    lines: List[str] = []
    lines.append("# SAM2.1 detailed vs baseline detailed")
    lines.append("")
    lines.append("## Technical summary")
    lines.append("")
    lines.append(
        "- 本报告只比较 baseline detailed 与 SAM2.1 detailed 的共集样本；"
        f"共集 n={common['common_count']}，baseline-only n={common['baseline_only_count']}，"
        f"SAM2.1-only n={common['ours_only_count']}。"
    )
    lines.append(
        "- 共集总体："
        f"baseline SR={pct(overall['baseline']['sr'])}, SPL={fmt_float(overall['baseline']['spl'], 4)}；"
        f"SAM2.1 SR={pct(overall['ours']['sr'])}, SPL={fmt_float(overall['ours']['spl'], 4)}；"
        f"差值 SR={pp(overall['delta']['sr'])}, SPL={fmt_float(overall['delta']['spl'], 4)}。"
    )
    flips = overall["outcome_flips"]
    lines.append(
        "- 结果层面："
        f"SAM2.1 救回 baseline 错误 {flips['case_counts']['01_baseline_wrong_ours_correct']} 条，"
        f"把 baseline 正确样本变错 {flips['case_counts']['10_baseline_correct_ours_wrong']} 条，"
        f"净成功数变化 {flips['net_success_delta_count']}。"
    )
    target = overall["target_correction"]
    lines.append(
        "- 目标距离层面："
        f"有效 L2 对 n={target['valid_l2_count']}；"
        f"selected target 更接近目标 {target['improved_count']} 条 "
        f"({pct(target['improved_rate_over_valid_l2'])})，"
        f"更远 {target['worsened_count']} 条 "
        f"({pct(target['worsened_rate_over_valid_l2'])})。"
    )
    lines.append("")

    lines.append("## Scope and definitions")
    lines.append("")
    lines.append(f"- Generated at: `{meta['created_at']}`")
    lines.append(f"- Baseline files: {meta['baseline']['file_count']} files")
    lines.append(f"- SAM2.1 files: {meta['ours']['file_count']} files")
    lines.append(
        "- Sample key: `(task_level, navigation_type, scene_name, episode_id, task_id)`."
    )
    lines.append(
        f"- Head categories: top {category['head_category_count']} / {category['category_count']} "
        f"object categories by full LangMap task frequency "
        f"({pct(category['head_task_coverage'])} of full tasks)."
    )
    lines.append(
        f"- Long-tail categories: remaining {category['long_tail_category_count']} categories "
        f"({pct(category['long_tail_task_coverage'])} of full tasks)."
    )
    lines.append(
        "- Outcome cases use `baseline_sr -> ours_sr`: "
        "`00` both wrong, `01` rescue, `10` regression, `11` both correct."
    )
    lines.append(
        "- Target correction uses `baseline_target_to_goal_l2 - selected_target_to_goal_l2`; "
        "positive means the SAM2.1 selected target is closer to the goal."
    )
    lines.append("")

    lines.append("## Overall SR/SPL")
    lines.append("")
    rows = [summary_metric_row("overall", overall)]
    for level, stats in by_level.items():
        rows.append(summary_metric_row(level, stats))
    lines.append(
        markdown_table(
            ["segment", "n", "baseline SR", "baseline SPL", "SAM2.1 SR", "SAM2.1 SPL", "SR delta", "SPL delta"],
            rows,
        )
    )
    lines.append("")

    lines.append("## Head vs long-tail SR/SPL")
    lines.append("")
    bucket_rows = [
        summary_metric_row(bucket, stats)
        for bucket, stats in by_bucket.items()
    ]
    lines.append(
        markdown_table(
            ["bucket", "n", "baseline SR", "baseline SPL", "SAM2.1 SR", "SAM2.1 SPL", "SR delta", "SPL delta"],
            bucket_rows,
        )
    )
    lines.append("")

    lines.append("## Outcome correction by level")
    lines.append("")
    correction_rows = [correction_row("overall", overall)]
    for level, stats in by_level.items():
        correction_rows.append(correction_row(level, stats))
    lines.append(
        markdown_table(
            [
                "segment",
                "n",
                "rescue 01",
                "correction rate",
                "regression 10",
                "regression rate",
                "net success",
                "net SR delta",
            ],
            correction_rows,
        )
    )
    lines.append("")

    lines.append("## Target-distance correction by level")
    lines.append("")
    target_rows = [target_row("overall", overall)]
    for level, stats in by_level.items():
        target_rows.append(target_row(level, stats))
    lines.append(
        markdown_table(
            [
                "segment",
                "valid L2 n",
                "closer",
                "farther",
                "mean delta m",
                "median delta m",
                f"outside->inside @{meta['target_threshold_m']}m",
                f"inside->outside @{meta['target_threshold_m']}m",
            ],
            target_rows,
        )
    )
    lines.append("")

    lines.append("## Limitations and checks")
    lines.append("")
    lines.append(
        "- 当前 SAM2.1 balanced detailed 只覆盖已完成 shard；因此所有方法对比都是共集样本上的描述性统计，不是全量 9420 条任务估计。"
    )
    lines.append(
        "- head/long-tail 的类别集合来自 `LangMap_Annotations` 全量任务频率，但 SR/SPL 只在当前共集里投影计算。"
    )
    if meta["baseline"]["duplicate_count"] or meta["ours"]["duplicate_count"]:
        lines.append(
            f"- 发现重复 sample key：baseline={meta['baseline']['duplicate_count']}，"
            f"SAM2.1={meta['ours']['duplicate_count']}；脚本保留排序后的第一条，并在 JSON metadata 里记录示例。"
        )
    else:
        lines.append("- 未发现重复 sample key。")
    lines.append(
        "- `module_helpful`/L2 纠偏是事后诊断，说明目标点更近或更远；它不等同于最终导航 SR。"
    )
    lines.append("")

    lines.append("## Metrics to confirm before adding")
    lines.append("")
    for item in payload["suggested_optional_metrics"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append(f"- Summary JSON: `{meta['summary_json']}`")
    lines.append(f"- Paired samples JSONL: `{meta['paired_rows_jsonl']}`")
    lines.append(f"- Analysis Markdown: `{meta['analysis_md']}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, allow_nan=True)
            handle.write("\n")


def build_payload(args: argparse.Namespace, summary_json: Path, paired_jsonl: Path, analysis_md: Path) -> Tuple[JsonDict, List[JsonDict]]:
    baseline_files = expand_patterns(args.baseline_pattern, args.include_concise)
    ours_files = expand_patterns(args.ours_pattern, args.include_concise)
    if not baseline_files:
        raise FileNotFoundError("No baseline JSON files matched the configured pattern(s).")
    if not ours_files:
        raise FileNotFoundError("No SAM2.1 JSON files matched the configured pattern(s).")

    baseline_index, baseline_meta = load_index(baseline_files)
    ours_index, ours_meta = load_index(ours_files)

    category_root = resolve_path(args.category_root)
    if category_root.is_dir():
        category_counts, category_source_meta = load_global_category_counts(category_root)
    else:
        category_counts, category_source_meta = fallback_category_counts(baseline_index.values())
        category_source_meta["warning"] = f"category root not found: {category_root}"
    head_categories, category_split = build_head_tail_split(category_counts, args.head_fraction)
    category_split.update(category_source_meta)

    paired_rows = build_paired_rows(
        baseline_index,
        ours_index,
        head_categories,
        args.target_threshold_m,
    )
    common_keys = set(baseline_index).intersection(ours_index)
    baseline_only = set(baseline_index) - set(ours_index)
    ours_only = set(ours_index) - set(baseline_index)

    metrics = {
        "overall": comparison_summary(paired_rows),
        "by_level": ordered_group_summary(paired_rows, "task_level", TASK_LEVELS),
        "by_category_bucket": ordered_group_summary(paired_rows, "category_bucket", ("head", "long_tail", "unknown")),
        "by_level_and_category_bucket": nested_level_bucket_summary(paired_rows),
    }

    payload: JsonDict = {
        "meta": {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "command": " ".join(sys.argv),
            "baseline_patterns": args.baseline_pattern,
            "ours_patterns": args.ours_pattern,
            "include_concise": args.include_concise,
            "head_fraction": args.head_fraction,
            "target_threshold_m": args.target_threshold_m,
            "baseline": baseline_meta,
            "ours": ours_meta,
            "summary_json": str(summary_json.relative_to(PROJECT_ROOT)),
            "paired_rows_jsonl": str(paired_jsonl.relative_to(PROJECT_ROOT)),
            "analysis_md": str(analysis_md.relative_to(PROJECT_ROOT)),
        },
        "common_sample": {
            "baseline_indexed_count": len(baseline_index),
            "ours_indexed_count": len(ours_index),
            "common_count": len(common_keys),
            "baseline_only_count": len(baseline_only),
            "ours_only_count": len(ours_only),
            "baseline_only_examples": [key_to_string(key) for key in sorted(baseline_only)[:20]],
            "ours_only_examples": [key_to_string(key) for key in sorted(ours_only)[:20]],
        },
        "category_split": category_split,
        "metrics": metrics,
        "examples": {
            "rescued_01": top_outcome_examples(paired_rows, "01", args.example_limit),
            "regressed_10": top_outcome_examples(paired_rows, "10", args.example_limit),
            "target_distance_most_improved": top_distance_examples(paired_rows, "improved", args.example_limit),
            "target_distance_most_worsened": top_distance_examples(paired_rows, "worsened", args.example_limit),
        },
        "suggested_optional_metrics": [
            "runtime/cost: compare task_time_sec and module elapsed_ms by level and category bucket",
            "module routing: split by mqsc_r1_applied, vistals_applied, module_reason, and vistals_input_slot_source",
            "category-level gain/loss: top object categories by rescue/regression and SR delta, with minimum-n filtering",
            "failure-mode shift: end_reason transition matrix from baseline to SAM2.1",
            "SPL conditional on success: efficiency among successful tasks only, separated from SR changes",
        ],
    }
    return payload, paired_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM2.1 detailed single-goal results against baseline detailed on common samples."
    )
    parser.add_argument(
        "--baseline-pattern",
        action="append",
        default=None,
        help="Glob pattern relative to repo root. Can be provided multiple times.",
    )
    parser.add_argument(
        "--ours-pattern",
        action="append",
        default=None,
        help="Glob pattern relative to repo root. Can be provided multiple times.",
    )
    parser.add_argument(
        "--category-root",
        default=DEFAULT_CATEGORY_ROOT,
        help="LangMap annotation root used to define global head/top-20%% categories.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Output filename stem.",
    )
    parser.add_argument(
        "--head-fraction",
        type=float,
        default=0.20,
        help="Fraction of object categories treated as head categories.",
    )
    parser.add_argument(
        "--target-threshold-m",
        type=float,
        default=1.0,
        help="Distance threshold for outside/inside target correction cases.",
    )
    parser.add_argument(
        "--include-concise",
        action="store_true",
        help="Include concisedesc files if the patterns match them. Default excludes concise.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=10,
        help="Number of example rows to keep for each example list.",
    )
    args = parser.parse_args()
    if args.baseline_pattern is None:
        args.baseline_pattern = [DEFAULT_BASELINE_PATTERN]
    if args.ours_pattern is None:
        args.ours_pattern = [DEFAULT_OURS_PATTERN]
    return args


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / f"{args.tag}.json"
    paired_jsonl = output_dir / f"{args.tag}_paired_samples.jsonl"
    analysis_md = output_dir / f"{args.tag}.md"

    payload, paired_rows = build_payload(args, summary_json, paired_jsonl, analysis_md)
    write_json(summary_json, payload)
    write_jsonl(paired_jsonl, paired_rows)
    analysis_md.write_text(build_markdown(payload), encoding="utf-8")

    overall = payload["metrics"]["overall"]
    print(f"summary_json={summary_json.relative_to(PROJECT_ROOT)}")
    print(f"paired_rows_jsonl={paired_jsonl.relative_to(PROJECT_ROOT)}")
    print(f"analysis_md={analysis_md.relative_to(PROJECT_ROOT)}")
    print(
        "overall "
        f"n={overall['n']} "
        f"baseline_sr={pct(overall['baseline']['sr'])} "
        f"ours_sr={pct(overall['ours']['sr'])} "
        f"sr_delta={pp(overall['delta']['sr'])} "
        f"baseline_spl={fmt_float(overall['baseline']['spl'], 4)} "
        f"ours_spl={fmt_float(overall['ours']['spl'], 4)} "
        f"spl_delta={fmt_float(overall['delta']['spl'], 4)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
