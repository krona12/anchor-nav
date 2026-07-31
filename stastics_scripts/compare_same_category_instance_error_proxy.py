#!/usr/bin/env python3
"""
Estimate same-category wrong-instance errors from current single-goal outputs.

Important limitation:
The existing baseline result JSON/logs do not store final agent positions or
final baseline target positions. For matched SAM2.1 detailed rows, the SAM2.1
result does store:
  - baseline_target_position: the unmodified final PQ3D target before refinement
  - selected_target_position: the refined final target used by ours

This script therefore computes a target-selection proxy, not the exact final
navigation endpoint metric. A row is counted as proxy same-category
wrong-instance when:
  - the method failed SR,
  - its final target point is within threshold of a same-category non-target
    instance anchor,
  - and it is not within threshold of any target instance anchor.

By default, instance anchors are object centers plus annotated view points.
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
TASK_LEVEL_TO_ANN_KEY = {
    "object": "episodes_by_object_level",
    "room": "episodes_by_room_level",
    "region": "episodes_by_region_level",
    "instance": "episodes_by_instance_level",
}
DEFAULT_BASELINE_PATTERN = (
    "output_logs/anchor/single_goal/baseline/detailed-full/"
    "refhm3d_single_goal_*.json"
)
DEFAULT_OURS_PATTERN = (
    "output_logs/anchor/single_goal/vista2mqsc_sam2/"
    "balanced-detailed-qwen_vl_plus_full/"
    "refhm3d_single_goal_vista2mqsc_refine1_*.json"
)
DEFAULT_OUTPUT_DIR = "stastics_scripts/logs"
DEFAULT_TAG = "sam2_1_same_category_instance_error_proxy"


RecordKey = Tuple[str, str, str, int, int]
JsonDict = Dict[str, Any]


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


def fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{100.0 * value:.2f}%"


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


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def should_skip(path: Path) -> bool:
    name = path.name.lower()
    return "effectiveness" in name or "concisedesc" in name


def expand_patterns(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if not path.is_file() or should_skip(path):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_level(record: Mapping[str, Any]) -> str:
    return str(record.get("task_level") or record.get("navigation_type") or "_missing")


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


def parse_records(path: Path) -> List[JsonDict]:
    data = load_json(path)
    if not isinstance(data, dict):
        return []
    raw_records = data.get("sequence")
    if not isinstance(raw_records, list):
        raw_records = []
        for level in TASK_LEVELS:
            value = data.get(level)
            if isinstance(value, list):
                raw_records.extend(value)

    records: List[JsonDict] = []
    for row in raw_records:
        if not isinstance(row, dict):
            continue
        if row.get("valid_for_metric", True) is False:
            continue
        record = dict(row)
        record["_source_file"] = str(path.relative_to(PROJECT_ROOT))
        records.append(record)
    return records


def load_index(files: Sequence[Path]) -> Tuple[Dict[RecordKey, JsonDict], JsonDict]:
    index: Dict[RecordKey, JsonDict] = {}
    duplicates: List[JsonDict] = []
    record_counts: Dict[str, int] = {}
    for path in files:
        records = parse_records(path)
        rel = str(path.relative_to(PROJECT_ROOT))
        record_counts[rel] = len(records)
        for record in records:
            key = record_key(record)
            if key in index:
                duplicates.append(
                    {
                        "key": key_to_string(key),
                        "kept_source": index[key].get("_source_file"),
                        "dropped_source": record.get("_source_file"),
                    }
                )
                continue
            index[key] = record
    return index, {
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "file_count": len(files),
        "record_counts": record_counts,
        "indexed_record_count": len(index),
        "duplicate_count": len(duplicates),
        "duplicate_examples": duplicates[:20],
    }


class AnnotationCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: Dict[str, JsonDict] = {}

    def get(self, scene_name: str) -> JsonDict:
        if scene_name not in self.cache:
            path = self.root / f"{scene_name}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.cache[scene_name] = json.load(handle)
        return self.cache[scene_name]


def goals_by_id(scene_data: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(goal.get("object_id")): goal
        for goal in scene_data.get("goals", [])
        if isinstance(goal, dict) and goal.get("object_id") is not None
    }


def task_for_record(scene_data: Mapping[str, Any], level: str, episode_id: int) -> Optional[Mapping[str, Any]]:
    ann_key = TASK_LEVEL_TO_ANN_KEY.get(level)
    if ann_key is None:
        return None
    tasks = scene_data.get(ann_key, [])
    if not isinstance(tasks, list) or episode_id < 0 or episode_id >= len(tasks):
        return None
    task = tasks[episode_id]
    return task if isinstance(task, dict) else None


def task_target_ids(task: Mapping[str, Any]) -> set[str]:
    target_ids = task.get("target_object_ids")
    if isinstance(target_ids, list):
        return {str(value) for value in target_ids}
    instance_id = task.get("instance_id")
    return {str(instance_id)} if instance_id else set()


def task_category(
    task: Mapping[str, Any],
    target_ids: set[str],
    goal_index: Mapping[str, Mapping[str, Any]],
    fallback: Any,
) -> str:
    if task.get("object_category"):
        return str(task["object_category"])
    for target_id in target_ids:
        goal = goal_index.get(target_id)
        if goal and goal.get("object_category"):
            return str(goal["object_category"])
    return str(fallback or "_missing")


def point3(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        return None
    if len(value) < 3:
        return None
    point = (safe_float(value[0], float("nan")), safe_float(value[1], float("nan")), safe_float(value[2], float("nan")))
    if not all(math.isfinite(x) for x in point):
        return None
    return point


def goal_anchor_points(goal: Mapping[str, Any], point_set: str) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    if point_set in {"center", "hybrid"}:
        center = point3(goal.get("position"))
        if center is not None:
            points.append(center)
    if point_set in {"viewpoint", "hybrid"}:
        view_points = goal.get("view_points", [])
        if isinstance(view_points, list):
            for view_point in view_points:
                if not isinstance(view_point, dict):
                    continue
                agent_state = view_point.get("agent_state", {})
                if not isinstance(agent_state, dict):
                    continue
                point = point3(agent_state.get("position"))
                if point is not None:
                    points.append(point)
    return points


def nearest_distance(
    point: Tuple[float, float, float],
    goals: Sequence[Mapping[str, Any]],
    point_set: str,
) -> Tuple[float, Optional[str], int]:
    best_dist = float("inf")
    best_id: Optional[str] = None
    point_count = 0
    for goal in goals:
        anchors = goal_anchor_points(goal, point_set)
        point_count += len(anchors)
        for anchor in anchors:
            dist = math.dist(point, anchor)
            if dist < best_dist:
                best_dist = dist
                best_id = str(goal.get("object_id"))
    return best_dist, best_id, point_count


def classify_method_point(
    *,
    point_value: Any,
    sr: bool,
    scene_data: Mapping[str, Any],
    level: str,
    episode_id: int,
    category_fallback: Any,
    threshold_m: float,
    point_set: str,
) -> JsonDict:
    point = point3(point_value)
    goal_index = goals_by_id(scene_data)
    task = task_for_record(scene_data, level, episode_id)
    if task is None:
        return {"classifiable": False, "reason": "missing_annotation_task"}

    target_ids = task_target_ids(task)
    category = task_category(task, target_ids, goal_index, category_fallback)
    goals = [goal for goal in scene_data.get("goals", []) if isinstance(goal, dict)]
    target_goals = [goal_index[target_id] for target_id in target_ids if target_id in goal_index]
    same_category_non_target_goals = [
        goal
        for goal in goals
        if str(goal.get("object_category")) == category and str(goal.get("object_id")) not in target_ids
    ]

    if point is None:
        return {
            "classifiable": False,
            "reason": "missing_method_point",
            "category": category,
            "target_instance_count": len(target_goals),
            "same_category_non_target_count": len(same_category_non_target_goals),
        }

    target_dist, target_id, target_anchor_count = nearest_distance(point, target_goals, point_set)
    non_target_dist, non_target_id, non_target_anchor_count = nearest_distance(
        point,
        same_category_non_target_goals,
        point_set,
    )
    target_hit = target_dist <= threshold_m
    non_target_hit = non_target_dist <= threshold_m
    ambiguous = target_hit and non_target_hit
    proxy_wrong = (not sr) and non_target_hit and not target_hit

    return {
        "classifiable": True,
        "reason": "ok",
        "category": category,
        "target_instance_count": len(target_goals),
        "same_category_non_target_count": len(same_category_non_target_goals),
        "target_anchor_count": target_anchor_count,
        "same_category_non_target_anchor_count": non_target_anchor_count,
        "min_target_anchor_distance_m": target_dist,
        "min_target_anchor_object_id": target_id,
        "min_same_category_non_target_anchor_distance_m": non_target_dist,
        "min_same_category_non_target_object_id": non_target_id,
        "target_anchor_hit": target_hit,
        "same_category_non_target_anchor_hit": non_target_hit,
        "ambiguous_target_and_non_target_hit": ambiguous,
        "proxy_same_category_wrong_instance": proxy_wrong,
        "sr": sr,
    }


def build_rows(
    baseline_index: Mapping[RecordKey, JsonDict],
    ours_index: Mapping[RecordKey, JsonDict],
    annotation_cache: AnnotationCache,
    threshold_m: float,
    point_set: str,
) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for key in sorted(set(baseline_index).intersection(ours_index)):
        level, navigation_type, scene_name, episode_id, task_id = key
        baseline = baseline_index[key]
        ours = ours_index[key]
        scene_data = annotation_cache.get(scene_name)
        category_fallback = ours.get("object_category") or baseline.get("object_category")

        baseline_cls = classify_method_point(
            point_value=ours.get("baseline_target_position"),
            sr=is_true(baseline.get("sr")),
            scene_data=scene_data,
            level=level,
            episode_id=episode_id,
            category_fallback=category_fallback,
            threshold_m=threshold_m,
            point_set=point_set,
        )
        ours_cls = classify_method_point(
            point_value=ours.get("selected_target_position"),
            sr=is_true(ours.get("sr")),
            scene_data=scene_data,
            level=level,
            episode_id=episode_id,
            category_fallback=category_fallback,
            threshold_m=threshold_m,
            point_set=point_set,
        )
        category = str(baseline_cls.get("category") or ours_cls.get("category") or category_fallback or "_missing")

        rows.append(
            {
                "key": key_to_string(key),
                "task_level": level,
                "navigation_type": navigation_type,
                "scene_name": scene_name,
                "episode_id": episode_id,
                "task_id": task_id,
                "object_category": category,
                "baseline_sr": is_true(baseline.get("sr")),
                "baseline_spl": safe_float(baseline.get("spl"), 0.0),
                "ours_sr": is_true(ours.get("sr")),
                "ours_spl": safe_float(ours.get("spl"), 0.0),
                "baseline_proxy": baseline_cls,
                "ours_proxy": ours_cls,
                "baseline_target_position_source": "ours_result.baseline_target_position",
                "ours_target_position_source": "ours_result.selected_target_position",
                "baseline_source_file": baseline.get("_source_file"),
                "ours_source_file": ours.get("_source_file"),
            }
        )
    return rows


def method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> JsonDict:
    proxy_field = f"{method}_proxy"
    sr_field = f"{method}_sr"
    classifiable = [row for row in rows if row.get(proxy_field, {}).get("classifiable")]
    alt_available = [
        row
        for row in classifiable
        if safe_int(row.get(proxy_field, {}).get("same_category_non_target_count")) > 0
    ]
    failures_alt = [row for row in alt_available if not bool(row.get(sr_field))]
    wrong = [
        row
        for row in classifiable
        if row.get(proxy_field, {}).get("proxy_same_category_wrong_instance")
    ]
    wrong_alt = [
        row
        for row in wrong
        if safe_int(row.get(proxy_field, {}).get("same_category_non_target_count")) > 0
    ]
    non_target_hits = [
        row
        for row in classifiable
        if row.get(proxy_field, {}).get("same_category_non_target_anchor_hit")
    ]
    ambiguous = [
        row
        for row in classifiable
        if row.get(proxy_field, {}).get("ambiguous_target_and_non_target_hit")
    ]
    no_alt = len(classifiable) - len(alt_available)
    reasons = Counter(str(row.get(proxy_field, {}).get("reason")) for row in rows)
    non_target_distances = [
        safe_float(row.get(proxy_field, {}).get("min_same_category_non_target_anchor_distance_m"), float("nan"))
        for row in classifiable
        if math.isfinite(
            safe_float(row.get(proxy_field, {}).get("min_same_category_non_target_anchor_distance_m"), float("nan"))
        )
    ]
    return {
        "rows": len(rows),
        "classifiable_count": len(classifiable),
        "not_classifiable_count": len(rows) - len(classifiable),
        "not_classifiable_reasons": dict(reasons),
        "same_category_non_target_available_count": len(alt_available),
        "no_same_category_non_target_count": no_alt,
        "failure_count_with_alt_available": len(failures_alt),
        "proxy_wrong_instance_count": len(wrong),
        "proxy_wrong_instance_count_alt_available": len(wrong_alt),
        "proxy_wrong_instance_rate_over_all_rows": ratio(len(wrong), len(rows)),
        "proxy_wrong_instance_rate_over_classifiable": ratio(len(wrong), len(classifiable)),
        "proxy_wrong_instance_rate_over_alt_available": ratio(len(wrong_alt), len(alt_available)),
        "proxy_wrong_instance_rate_over_failures_with_alt_available": ratio(len(wrong_alt), len(failures_alt)),
        "same_category_non_target_anchor_hit_count": len(non_target_hits),
        "ambiguous_target_and_non_target_hit_count": len(ambiguous),
        "mean_nearest_non_target_anchor_distance_m": mean(non_target_distances),
        "median_nearest_non_target_anchor_distance_m": median(non_target_distances),
    }


def comparison_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    baseline_wrong = {
        row["key"]
        for row in rows
        if row.get("baseline_proxy", {}).get("proxy_same_category_wrong_instance")
    }
    ours_wrong = {
        row["key"]
        for row in rows
        if row.get("ours_proxy", {}).get("proxy_same_category_wrong_instance")
    }
    common_keys = {str(row["key"]) for row in rows}
    return {
        "rows": len(rows),
        "baseline": method_summary(rows, "baseline"),
        "ours": method_summary(rows, "ours"),
        "paired_wrong_instance_cases": {
            "both_wrong_instance": len(baseline_wrong & ours_wrong),
            "baseline_wrong_only": len(baseline_wrong - ours_wrong),
            "ours_wrong_only": len(ours_wrong - baseline_wrong),
            "neither_wrong_instance": len(common_keys - (baseline_wrong | ours_wrong)),
            "net_ours_minus_baseline_wrong_count": len(ours_wrong) - len(baseline_wrong),
        },
    }


def group_rows(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "_missing")].append(row)
    return grouped


def by_level_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped = group_rows(rows, "task_level")
    keys = [key for key in TASK_LEVELS if key in grouped]
    keys.extend(sorted(set(grouped) - set(keys)))
    return {key: comparison_summary(grouped[key]) for key in keys}


def example_rows(rows: Sequence[Mapping[str, Any]], method: str, limit: int) -> List[JsonDict]:
    proxy_field = f"{method}_proxy"
    filtered = [
        row
        for row in rows
        if row.get(proxy_field, {}).get("proxy_same_category_wrong_instance")
    ]
    filtered.sort(
        key=lambda row: safe_float(
            row.get(proxy_field, {}).get("min_same_category_non_target_anchor_distance_m"),
            float("inf"),
        )
    )
    examples: List[JsonDict] = []
    for row in filtered[:limit]:
        proxy = row.get(proxy_field, {})
        examples.append(
            {
                "key": row.get("key"),
                "task_level": row.get("task_level"),
                "object_category": row.get("object_category"),
                f"{method}_sr": row.get(f"{method}_sr"),
                "nearest_wrong_instance_id": proxy.get("min_same_category_non_target_object_id"),
                "nearest_wrong_instance_distance_m": proxy.get("min_same_category_non_target_anchor_distance_m"),
                "nearest_target_instance_id": proxy.get("min_target_anchor_object_id"),
                "nearest_target_distance_m": proxy.get("min_target_anchor_distance_m"),
            }
        )
    return examples


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")


def scrub_nonfinite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: scrub_nonfinite(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [scrub_nonfinite(item) for item in value]
    return value


def method_row(label: str, stats: Mapping[str, Any]) -> List[str]:
    return [
        label,
        str(stats["classifiable_count"]),
        str(stats["same_category_non_target_available_count"]),
        str(stats["failure_count_with_alt_available"]),
        str(stats["proxy_wrong_instance_count"]),
        fmt_pct(stats["proxy_wrong_instance_rate_over_classifiable"]),
        fmt_pct(stats["proxy_wrong_instance_rate_over_alt_available"]),
        fmt_pct(stats["proxy_wrong_instance_rate_over_failures_with_alt_available"]),
        str(stats["ambiguous_target_and_non_target_hit_count"]),
    ]


def level_row(level: str, stats: Mapping[str, Any]) -> List[str]:
    paired = stats["paired_wrong_instance_cases"]
    return [
        level,
        str(stats["rows"]),
        str(stats["baseline"]["proxy_wrong_instance_count"]),
        str(stats["ours"]["proxy_wrong_instance_count"]),
        str(paired["baseline_wrong_only"]),
        str(paired["ours_wrong_only"]),
        str(paired["both_wrong_instance"]),
        str(paired["net_ours_minus_baseline_wrong_count"]),
    ]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def build_markdown(payload: Mapping[str, Any]) -> str:
    overall = payload["metrics"]["overall"]
    baseline = overall["baseline"]
    ours = overall["ours"]
    paired = overall["paired_wrong_instance_cases"]
    meta = payload["meta"]

    lines: List[str] = []
    lines.append("# Same-category wrong-instance proxy")
    lines.append("")
    lines.append("## Feasibility decision")
    lines.append("")
    lines.append(
        "- Exact final-navigation wrong-instance rate is not recoverable from the existing baseline result JSON/logs because they do not store final agent position or final baseline target for successful rows."
    )
    lines.append(
        "- A target-selection proxy is computable on the SAM2.1/common rows because SAM2.1 detailed results store both the unmodified `baseline_target_position` and the refined `selected_target_position`."
    )
    lines.append(
        "- This report computes that proxy only; use it to compare target choice confusion, not as a replacement for exact endpoint-based navigation analysis."
    )
    lines.append("")
    lines.append("## Proxy definition")
    lines.append("")
    lines.append(
        f"- Threshold: `{meta['threshold_m']}m`; anchor set: `{meta['point_set']}`."
    )
    lines.append(
        "- A method counts as proxy same-category wrong-instance when it failed SR, its final target point is within threshold of a same-category non-target instance anchor, and it is not within threshold of any target instance anchor."
    )
    lines.append(
        "- Target instances and non-target same-category instances are read from `LangMap_Annotations` by `(scene_name, task_level, episode_id)`."
    )
    lines.append("")
    lines.append("## Overall comparison")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "method",
                "classifiable",
                "alt category inst. available",
                "failures with alt",
                "wrong-inst count",
                "rate / classifiable",
                "rate / alt",
                "rate / failures+alt",
                "ambiguous hits",
            ],
            [method_row("baseline proxy", baseline), method_row("ours proxy", ours)],
        )
    )
    lines.append("")
    lines.append(
        f"Paired switch counts: baseline-only wrong={paired['baseline_wrong_only']}, "
        f"ours-only wrong={paired['ours_wrong_only']}, both={paired['both_wrong_instance']}, "
        f"net ours-baseline={paired['net_ours_minus_baseline_wrong_count']}."
    )
    lines.append("")
    lines.append("## By level")
    lines.append("")
    level_rows = [level_row(level, stats) for level, stats in payload["metrics"]["by_level"].items()]
    lines.append(
        markdown_table(
            [
                "level",
                "rows",
                "baseline wrong",
                "ours wrong",
                "baseline-only",
                "ours-only",
                "both",
                "net ours-baseline",
            ],
            level_rows,
        )
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- For exact endpoint measurement, future runs should save `final_agent_position`, `final_snapped_target_position`, `target_object_ids`, and nearest target/non-target same-category object ids/distances in every result row."
    )
    lines.append(
        "- If you want a stricter interpretation, rerun this script with `--point-set center` or `--point-set viewpoint`; the default `hybrid` treats either object center or annotated view point as an instance anchor."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary JSON: `{meta['summary_json']}`")
    lines.append(f"- Classified rows JSONL: `{meta['classified_rows_jsonl']}`")
    lines.append(f"- Analysis Markdown: `{meta['analysis_md']}`")
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace, summary_json: Path, rows_jsonl: Path, analysis_md: Path) -> Tuple[JsonDict, List[JsonDict]]:
    baseline_files = expand_patterns(args.baseline_pattern)
    ours_files = expand_patterns(args.ours_pattern)
    if not baseline_files:
        raise FileNotFoundError("No baseline JSON files matched.")
    if not ours_files:
        raise FileNotFoundError("No SAM2.1 JSON files matched.")

    baseline_index, baseline_meta = load_index(baseline_files)
    ours_index, ours_meta = load_index(ours_files)
    annotation_cache = AnnotationCache(resolve_path(args.category_root))
    rows = build_rows(
        baseline_index,
        ours_index,
        annotation_cache,
        threshold_m=args.threshold_m,
        point_set=args.point_set,
    )

    payload: JsonDict = {
        "meta": {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "command": " ".join(sys.argv),
            "baseline_patterns": args.baseline_pattern,
            "ours_patterns": args.ours_pattern,
            "category_root": args.category_root,
            "threshold_m": args.threshold_m,
            "point_set": args.point_set,
            "baseline": baseline_meta,
            "ours": ours_meta,
            "common_count": len(rows),
            "summary_json": str(summary_json.relative_to(PROJECT_ROOT)),
            "classified_rows_jsonl": str(rows_jsonl.relative_to(PROJECT_ROOT)),
            "analysis_md": str(analysis_md.relative_to(PROJECT_ROOT)),
        },
        "feasibility": {
            "exact_endpoint_metric_computable_from_current_artifacts": False,
            "exact_blockers": [
                "baseline JSON/logs do not store final_agent_position",
                "baseline JSON/logs do not store successful-row final target_position/snapped target",
                "SAM2.1 JSON stores final selected target but not final_agent_position",
            ],
            "computed_metric": "target-selection proxy",
            "baseline_proxy_source": "baseline_target_position stored in SAM2.1 detailed result rows",
            "ours_proxy_source": "selected_target_position stored in SAM2.1 detailed result rows",
        },
        "metrics": {
            "overall": comparison_summary(rows),
            "by_level": by_level_summary(rows),
        },
        "examples": {
            "baseline_proxy_wrong_instance": example_rows(rows, "baseline", args.example_limit),
            "ours_proxy_wrong_instance": example_rows(rows, "ours", args.example_limit),
        },
    }
    return scrub_nonfinite(payload), scrub_nonfinite(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a same-category wrong-instance target-selection proxy."
    )
    parser.add_argument("--baseline-pattern", action="append", default=None)
    parser.add_argument("--ours-pattern", action="append", default=None)
    parser.add_argument("--category-root", default="LangMap_Annotations")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--threshold-m", type=float, default=0.25)
    parser.add_argument(
        "--point-set",
        choices=("center", "viewpoint", "hybrid"),
        default="hybrid",
        help="Which per-instance anchors to test against.",
    )
    parser.add_argument("--example-limit", type=int, default=10)
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
    rows_jsonl = output_dir / f"{args.tag}_classified_rows.jsonl"
    analysis_md = output_dir / f"{args.tag}.md"

    payload, rows = build_payload(args, summary_json, rows_jsonl, analysis_md)
    write_json(summary_json, payload)
    write_jsonl(rows_jsonl, rows)
    analysis_md.write_text(build_markdown(payload), encoding="utf-8")

    overall = payload["metrics"]["overall"]
    baseline = overall["baseline"]
    ours = overall["ours"]
    paired = overall["paired_wrong_instance_cases"]
    print(f"summary_json={summary_json.relative_to(PROJECT_ROOT)}")
    print(f"classified_rows_jsonl={rows_jsonl.relative_to(PROJECT_ROOT)}")
    print(f"analysis_md={analysis_md.relative_to(PROJECT_ROOT)}")
    print(
        "proxy "
        f"n={overall['rows']} "
        f"baseline_wrong={baseline['proxy_wrong_instance_count']} "
        f"ours_wrong={ours['proxy_wrong_instance_count']} "
        f"net={paired['net_ours_minus_baseline_wrong_count']} "
        f"baseline_rate_fail_alt={fmt_pct(baseline['proxy_wrong_instance_rate_over_failures_with_alt_available'])} "
        f"ours_rate_fail_alt={fmt_pct(ours['proxy_wrong_instance_rate_over_failures_with_alt_available'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
