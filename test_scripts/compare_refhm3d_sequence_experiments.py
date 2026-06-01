#!/usr/bin/env python3
"""
Compare RefHM3D sequence experiment shards.

The default manifest is intentionally explicit: it records the result JSON and
log file used for each shard, so the reported metrics have a traceable source.

Metrics:
  - task-level SR/SPL overall and by task_level
  - sequence-level @4/@5 over complete 5-task episodes
  - shard completeness and fatal log markers
  - deltas vs the Baseline run, plus overlap deltas for partial runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LEVEL_ORDER = ("object", "room", "region", "instance")
FATAL_PATTERNS = (
    "Traceback",
    "OutOfMemoryError",
    "CUDA out of memory",
    "KeyboardInterrupt",
    "Segmentation fault",
    "Killed",
)


@dataclass(frozen=True)
class ShardSpec:
    split: str
    expected_tasks: int
    json_path: str
    log_path: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    shards: Tuple[ShardSpec, ...]


DEFAULT_EXPERIMENTS: Tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="Baseline",
        shards=(
            ShardSpec(
                split="0.0-0.2",
                expected_tasks=700,
                json_path="output_logs/baseline_all_0_0.2/20260513-112352-detailed/refhm3d_seq_0.0_0.2.json",
                log_path="output_logs/baseline_all_0_0.2/20260513-112352-detailed/refhm3d-nav-sequence-baseline-20260513-112413-119890-pid1086119.log",
            ),
            ShardSpec(
                split="0.2-1.0",
                expected_tasks=2900,
                json_path="output_logs/baseline_all_0.2_1.0/20260513-101905-detailed/refhm3d_seq_0.2_1.0.json",
                log_path="output_logs/baseline_all_0.2_1.0/20260513-101905-detailed/refhm3d-nav-sequence-baseline-20260513-101924-534262-pid862786.log",
            ),
        ),
    ),
    ExperimentSpec(
        name="+ VISTA-LS",
        shards=(
            ShardSpec(
                split="0.0-0.2",
                expected_tasks=700,
                json_path="output_logs/anchor/vistals_all_0_0.2/20260513-112500-detailed-fast-grid-v2-cuda2/refhm3d_seq_vistals_refine1_0.0_0.2.json",
                log_path="output_logs/anchor/vistals_all_0_0.2/20260513-112500-detailed-fast-grid-v2-cuda2/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-112519-263530-pid1088698.log",
            ),
            ShardSpec(
                split="0.2-1.0",
                expected_tasks=2900,
                json_path="output_logs/anchor/vistals_all_0.2_1.0/20260513-102052-detailed/refhm3d_seq_vistals_refine1_0.2_1.0.json",
                log_path="output_logs/anchor/vistals_all_0.2_1.0/20260513-102052-detailed/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-102111-463001-pid868899.log",
            ),
        ),
    ),
    ExperimentSpec(
        name="MQSC-R1",
        shards=(
            ShardSpec(
                split="0.0-0.2",
                expected_tasks=700,
                json_path="output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d_seq_mqsc_r1_refine1_0.0_0.2.json",
                log_path="output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260514-144811-731948-pid1525210.log",
            ),
            ShardSpec(
                split="0.2-1.0",
                expected_tasks=2900,
                json_path="output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_mqsc_r1_refine1_0.2_1.0.json",
                log_path="output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260516-113222-410036-pid56740.log",
            ),
        ),
    ),
    ExperimentSpec(
        name="Vista2MQSC",
        shards=(
            ShardSpec(
                split="0.0-0.2",
                expected_tasks=700,
                json_path="output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d_seq_vista2mqsc_refine1_0.0_0.2.json",
                log_path="output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260514-150659-226187-pid1684292.log",
            ),
            ShardSpec(
                split="0.2-1.0",
                expected_tasks=2900,
                json_path="output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_vista2mqsc_refine1_0.2_1.0.json",
                log_path="output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260516-113222-409168-pid56747.log",
            ),
        ),
    ),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def bool_sr(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes", "y"}
    return False


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def records_from_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        seq = data.get("sequence")
        if isinstance(seq, list):
            return [x for x in seq if isinstance(x, dict)]
        out: List[Dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                out.extend(x for x in value if isinstance(x, dict) and "sr" in x)
        return out
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def task_key(record: Dict[str, Any]) -> Tuple[str, int, int]:
    scene = record.get("scene_name", record.get("scene", record.get("scene_id", "")))
    episode = record.get("episode_id", record.get("episode", record.get("episode_index", -1)))
    task = record.get("task_id", record.get("task", -1))
    return str(scene), int(episode), int(task)


def episode_key(record: Dict[str, Any]) -> Tuple[str, int]:
    scene, episode, _task = task_key(record)
    return scene, episode


def task_level(record: Dict[str, Any]) -> str:
    level = record.get("task_level", record.get("level", record.get("task_type", "_missing")))
    return str(level) if level not in (None, "") else "_missing"


def metric_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    sr_count = sum(1 for r in records if bool_sr(r.get("sr")))
    spl_sum = sum(float_value(r.get("spl")) for r in records)
    time_values = [float_value(r.get("task_time_sec")) for r in records if r.get("task_time_sec") is not None]
    return {
        "tasks": n,
        "successes": sr_count,
        "sr": sr_count / n if n else 0.0,
        "spl": spl_sum / n if n else 0.0,
        "avg_task_time_sec": sum(time_values) / len(time_values) if time_values else 0.0,
    }


def by_level_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[task_level(record)].append(record)
    ordered = list(LEVEL_ORDER) + sorted(k for k in buckets if k not in LEVEL_ORDER)
    return {level: metric_summary(buckets[level]) for level in ordered if level in buckets}


def dedup_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        key = task_key(record)
        if key in seen:
            duplicates += 1
        seen[key] = record
    return [seen[k] for k in sorted(seen)], duplicates


def sequence_summary(records: Sequence[Dict[str, Any]], tasks_per_episode: int) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        scene, episode, task = task_key(record)
        grouped[(scene, episode)][task] = record

    complete: List[Dict[int, Dict[str, Any]]] = []
    partial = 0
    for tasks in grouped.values():
        expected_ids = set(range(tasks_per_episode))
        if expected_ids.issubset(tasks.keys()):
            complete.append(tasks)
        else:
            partial += 1

    success_counts = [
        sum(1 for task_id in range(tasks_per_episode) if bool_sr(tasks[task_id].get("sr")))
        for tasks in complete
    ]
    n = len(success_counts)
    at4 = sum(1 for count in success_counts if count >= 4)
    at5 = sum(1 for count in success_counts if count >= 5)
    return {
        "episode_groups": len(grouped),
        "complete_episodes": n,
        "partial_episodes": partial,
        "at4_count": at4,
        "at4": at4 / n if n else 0.0,
        "at5_count": at5,
        "at5": at5 / n if n else 0.0,
        "avg_successes_per_episode": sum(success_counts) / n if n else 0.0,
    }


def complete_episode_keys(records: Sequence[Dict[str, Any]], tasks_per_episode: int) -> set[Tuple[str, int]]:
    grouped: Dict[Tuple[str, int], set[int]] = defaultdict(set)
    for record in records:
        scene, episode, task = task_key(record)
        grouped[(scene, episode)].add(task)
    expected_ids = set(range(tasks_per_episode))
    return {key for key, task_ids in grouped.items() if expected_ids.issubset(task_ids)}


def filter_records_by_keys(records: Sequence[Dict[str, Any]], keys: Iterable[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    wanted = set(keys)
    return [record for record in records if task_key(record) in wanted]


def filter_records_by_episode_keys(
    records: Sequence[Dict[str, Any]],
    keys: Iterable[Tuple[str, int]],
) -> List[Dict[str, Any]]:
    wanted = set(keys)
    return [record for record in records if episode_key(record) in wanted]


def scan_log(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": path.is_file(),
        "run_finished": False,
        "fatal_markers": [],
        "max_metrics_sequence_count": None,
        "max_live_rows": None,
    }
    if not path.is_file():
        return info

    sequence_counts: List[int] = []
    live_rows: List[int] = []
    fatal_hits: List[str] = []
    metric_re = re.compile(r"sequence count[:=]\s*(\d+)")
    live_re = re.compile(r"\brows=(\d+)\b|\bcount=(\d+)\b")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if "run finished" in line:
                info["run_finished"] = True
            metric_match = metric_re.search(line)
            if metric_match:
                sequence_counts.append(int(metric_match.group(1)))
            if "LIVE_METRICS" in line:
                live_match = live_re.search(line)
                if live_match:
                    live_rows.append(int(next(x for x in live_match.groups() if x is not None)))
            for pattern in FATAL_PATTERNS:
                if pattern in line:
                    fatal_hits.append(f"{line_no}:{pattern}")
                    break

    info["fatal_markers"] = fatal_hits
    info["max_metrics_sequence_count"] = max(sequence_counts) if sequence_counts else None
    info["max_live_rows"] = max(live_rows) if live_rows else None
    return info


def load_experiment(root: Path, spec: ExperimentSpec, tasks_per_episode: int) -> Dict[str, Any]:
    shard_reports: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    expected_total = 0

    for shard in spec.shards:
        expected_total += shard.expected_tasks
        json_path = resolve_path(root, shard.json_path)
        log_path = resolve_path(root, shard.log_path)
        shard_records = records_from_json(json_path) if json_path.is_file() else []
        records.extend(shard_records)
        unique_records, duplicate_count = dedup_records(shard_records)
        log_info = scan_log(log_path)
        shard_reports.append(
            {
                "split": shard.split,
                "expected_tasks": shard.expected_tasks,
                "json_path": relpath(root, json_path),
                "log_path": relpath(root, log_path),
                "json_exists": json_path.is_file(),
                "log_exists": log_path.is_file(),
                "tasks": len(shard_records),
                "unique_tasks": len(unique_records),
                "duplicates": duplicate_count,
                "metric": metric_summary(shard_records),
                "sequence": sequence_summary(shard_records, tasks_per_episode),
                "log": log_info,
            }
        )

    unique_records, duplicate_count = dedup_records(records)
    fatal_count = sum(len(shard["log"]["fatal_markers"]) for shard in shard_reports)
    complete = len(records) == expected_total and duplicate_count == 0 and fatal_count == 0
    partial = len(records) < expected_total or any(shard["tasks"] < shard["expected_tasks"] for shard in shard_reports)
    if complete:
        status = "complete"
    elif partial and fatal_count:
        status = "partial_error"
    elif partial:
        status = "partial"
    elif fatal_count:
        status = "log_error"
    else:
        status = "check"

    return {
        "name": spec.name,
        "expected_tasks": expected_total,
        "tasks": len(records),
        "unique_tasks": len(unique_records),
        "duplicates": duplicate_count,
        "status": status,
        "records": records,
        "metric": metric_summary(records),
        "by_level": by_level_summary(records),
        "sequence": sequence_summary(records, tasks_per_episode),
        "shards": shard_reports,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def number(value: float) -> str:
    return f"{value:.6f}"


def delta_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.2f}pp"


def delta_float(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


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


def comparison_rows(
    experiments: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> List[List[str]]:
    rows: List[List[str]] = []
    base_metric = baseline["metric"]
    base_seq = baseline["sequence"]
    for exp in experiments:
        metric = exp["metric"]
        seq = exp["sequence"]
        rows.append(
            [
                exp["name"],
                exp["status"],
                f"{exp['tasks']}/{exp['expected_tasks']}",
                pct(metric["sr"]),
                number(metric["spl"]),
                delta_pp(metric["sr"] - base_metric["sr"]) if exp is not baseline else "-",
                delta_float(metric["spl"] - base_metric["spl"]) if exp is not baseline else "-",
                f"{seq['complete_episodes']}",
                pct(seq["at4"]),
                pct(seq["at5"]),
                delta_pp(seq["at4"] - base_seq["at4"]) if exp is not baseline else "-",
                delta_pp(seq["at5"] - base_seq["at5"]) if exp is not baseline else "-",
            ]
        )
    return rows


def level_rows(experiments: Sequence[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for level in LEVEL_ORDER:
        for metric_name in ("SR", "SPL"):
            row = [level, metric_name]
            for exp in experiments:
                level_metric = exp["by_level"].get(level)
                if not level_metric:
                    row.append("-")
                elif metric_name == "SR":
                    row.append(f"{pct(level_metric['sr'])} (n={level_metric['tasks']})")
                else:
                    row.append(number(level_metric["spl"]))
            rows.append(row)
    return rows


def overlap_report(
    experiment: Dict[str, Any],
    baseline: Dict[str, Any],
    tasks_per_episode: int,
) -> Optional[Dict[str, Any]]:
    if experiment["tasks"] >= experiment["expected_tasks"]:
        return None

    exp_records = experiment["records"]
    base_records = baseline["records"]
    exp_keys = {task_key(record) for record in exp_records}
    base_overlap = filter_records_by_keys(base_records, exp_keys)
    exp_overlap = filter_records_by_keys(exp_records, {task_key(record) for record in base_overlap})

    exp_episode_keys = complete_episode_keys(exp_overlap, tasks_per_episode)
    base_episode_keys = complete_episode_keys(base_overlap, tasks_per_episode)
    shared_episode_keys = exp_episode_keys & base_episode_keys
    exp_seq_records = filter_records_by_episode_keys(exp_overlap, shared_episode_keys)
    base_seq_records = filter_records_by_episode_keys(base_overlap, shared_episode_keys)

    return {
        "name": experiment["name"],
        "task_overlap_count": len(exp_overlap),
        "baseline_metric": metric_summary(base_overlap),
        "experiment_metric": metric_summary(exp_overlap),
        "complete_episode_overlap_count": len(shared_episode_keys),
        "baseline_sequence": sequence_summary(base_seq_records, tasks_per_episode),
        "experiment_sequence": sequence_summary(exp_seq_records, tasks_per_episode),
    }


def build_report(root: Path, experiments: Sequence[Dict[str, Any]], tasks_per_episode: int) -> str:
    baseline = experiments[0]
    lines: List[str] = []
    lines.append(f"project_root={root}")
    lines.append(f"tasks_per_episode={tasks_per_episode}")
    lines.append("")
    lines.append("Overall")
    lines.append(
        format_table(
            (
                "experiment",
                "status",
                "tasks",
                "SR",
                "SPL",
                "dSR_vs_base",
                "dSPL_vs_base",
                "episodes",
                "@4",
                "@5",
                "d@4_vs_base",
                "d@5_vs_base",
            ),
            comparison_rows(experiments, baseline),
        )
    )
    lines.append("")
    lines.append("By Level")
    lines.append(format_table(("level", "metric", *(exp["name"] for exp in experiments)), level_rows(experiments)))

    lines.append("")
    lines.append("Shard Sources")
    shard_rows: List[List[str]] = []
    for exp in experiments:
        for shard in exp["shards"]:
            log = shard["log"]
            fatal = len(log["fatal_markers"])
            shard_rows.append(
                [
                    exp["name"],
                    shard["split"],
                    f"{shard['tasks']}/{shard['expected_tasks']}",
                    str(shard["unique_tasks"]),
                    str(shard["duplicates"]),
                    "yes" if log["run_finished"] else "no",
                    str(log["max_metrics_sequence_count"]),
                    str(log["max_live_rows"]),
                    str(fatal),
                    shard["json_path"],
                    shard["log_path"],
                ]
            )
    lines.append(
        format_table(
            (
                "experiment",
                "split",
                "tasks",
                "unique",
                "dups",
                "finished",
                "log_seq",
                "live_rows",
                "fatal",
                "json",
                "log",
            ),
            shard_rows,
        )
    )

    overlap_rows: List[List[str]] = []
    for exp in experiments[1:]:
        overlap = overlap_report(exp, baseline, tasks_per_episode)
        if overlap is None:
            continue
        base_metric = overlap["baseline_metric"]
        exp_metric = overlap["experiment_metric"]
        base_seq = overlap["baseline_sequence"]
        exp_seq = overlap["experiment_sequence"]
        overlap_rows.append(
            [
                overlap["name"],
                str(overlap["task_overlap_count"]),
                pct(base_metric["sr"]),
                pct(exp_metric["sr"]),
                delta_pp(exp_metric["sr"] - base_metric["sr"]),
                number(base_metric["spl"]),
                number(exp_metric["spl"]),
                delta_float(exp_metric["spl"] - base_metric["spl"]),
                str(overlap["complete_episode_overlap_count"]),
                pct(base_seq["at4"]),
                pct(exp_seq["at4"]),
                delta_pp(exp_seq["at4"] - base_seq["at4"]),
                pct(base_seq["at5"]),
                pct(exp_seq["at5"]),
                delta_pp(exp_seq["at5"] - base_seq["at5"]),
            ]
        )
    if overlap_rows:
        lines.append("")
        lines.append("Partial Overlap Vs Baseline")
        lines.append(
            format_table(
                (
                    "experiment",
                    "tasks",
                    "base_SR",
                    "exp_SR",
                    "dSR",
                    "base_SPL",
                    "exp_SPL",
                    "dSPL",
                    "episodes",
                    "base_@4",
                    "exp_@4",
                    "d@4",
                    "base_@5",
                    "exp_@5",
                    "d@5",
                ),
                overlap_rows,
            )
        )

    return "\n".join(lines)


def strip_records_for_json(experiment: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in experiment.items() if k != "records"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RefHM3D sequence experiment shards.")
    parser.add_argument("--root", type=Path, default=project_root(), help="Repository root.")
    parser.add_argument("--tasks-per-episode", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    experiments = [load_experiment(root, spec, args.tasks_per_episode) for spec in DEFAULT_EXPERIMENTS]

    if not experiments or experiments[0]["name"] != "Baseline":
        print("The first default experiment must be Baseline.", file=sys.stderr)
        return 2

    report = build_report(root, experiments, args.tasks_per_episode)
    print(report)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report + "\n", encoding="utf-8")

    if args.json_out:
        payload = {
            "root": str(root),
            "tasks_per_episode": args.tasks_per_episode,
            "experiments": [strip_records_for_json(exp) for exp in experiments],
            "partial_overlap_vs_baseline": [
                overlap_report(exp, experiments[0], args.tasks_per_episode)
                for exp in experiments[1:]
                if overlap_report(exp, experiments[0], args.tasks_per_episode) is not None
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
