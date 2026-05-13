#!/usr/bin/env python3
"""
Summarize MQSC module effectiveness from a refine1 effectiveness JSON.

Default input:
output_logs/anchor/mqsc_all_0.05_0.1/20260513-035845-detailed-repair-follow-vlm-20260513-035843/
refhm3d_seq_mqsc_refine1_effectiveness_0.05_0.1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EFFECTIVENESS_JSON = PROJECT_ROOT / (
    "output_logs/anchor/mqsc_all_0.05_0.1/"
    "20260513-035845-detailed-repair-follow-vlm-20260513-035843/"
    "refhm3d_seq_mqsc_refine1_effectiveness_0.05_0.1.json"
)

LEVEL_ORDER = ("object", "room", "region", "instance", "_missing")
CASE_ORDER = ("00", "01", "10", "11")
EPS = 1e-6


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def auto_result_json(effectiveness_path: Path) -> Optional[Path]:
    candidates = [
        p
        for p in sorted(effectiveness_path.parent.glob("refhm3d_seq_mqsc_refine1_*.json"))
        if "effectiveness" not in p.name
    ]
    return candidates[0] if len(candidates) == 1 else None


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def mean(values: Iterable[float]) -> float:
    nums = list(values)
    return sum(nums) / len(nums) if nums else 0.0


def pct(numer: int, denom: int) -> str:
    return "0.00%" if denom <= 0 else f"{100.0 * numer / denom:.2f}%"


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}f}"


def ordered_keys(keys: Iterable[str], preferred: Sequence[str]) -> List[str]:
    seen = {str(k) for k in keys}
    out = [k for k in preferred if k in seen]
    out.extend(sorted(seen - set(out)))
    return out


def record_key(record: Mapping[str, Any]) -> Tuple[str, int, int]:
    return (
        str(record.get("scene_name", "")),
        safe_int(record.get("episode_id")),
        safe_int(record.get("task_id")),
    )


def result_index(result_json: Optional[Mapping[str, Any]]) -> Dict[Tuple[str, int, int], Mapping[str, Any]]:
    if not result_json:
        return {}
    seq = result_json.get("sequence", [])
    if not isinstance(seq, list):
        return {}
    return {record_key(row): row for row in seq if isinstance(row, dict)}


def hook_infos(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    infos: List[Mapping[str, Any]] = []
    logs = record.get("hook_logs", [])
    if not isinstance(logs, list):
        return infos
    for item in logs:
        if not isinstance(item, dict):
            continue
        info = item.get("module_info") or item.get("module") or {}
        if isinstance(info, dict):
            infos.append(info)
    return infos


def final_hook_info(record: Mapping[str, Any]) -> Mapping[str, Any]:
    infos = hook_infos(record)
    return infos[-1] if infos else {}


def distance_delta(record: Mapping[str, Any]) -> Optional[float]:
    if not boolish(record.get("baseline_target_to_goal_l2_valid")):
        return None
    if not boolish(record.get("selected_target_to_goal_l2_valid")):
        return None
    baseline = safe_float(record.get("baseline_target_to_goal_l2"), default=float("nan"))
    selected = safe_float(record.get("selected_target_to_goal_l2"), default=float("nan"))
    if not (math.isfinite(baseline) and math.isfinite(selected)):
        return None
    return selected - baseline


def delta_label(delta: Optional[float]) -> str:
    if delta is None:
        return "invalid"
    if delta < -EPS:
        return "improved"
    if delta > EPS:
        return "worsened"
    return "tied"


def threshold_case(record: Mapping[str, Any], threshold: float) -> Optional[str]:
    if not boolish(record.get("baseline_target_to_goal_l2_valid")):
        return None
    if not boolish(record.get("selected_target_to_goal_l2_valid")):
        return None
    baseline = safe_float(record.get("baseline_target_to_goal_l2"), default=float("inf"))
    selected = safe_float(record.get("selected_target_to_goal_l2"), default=float("inf"))
    if not (math.isfinite(baseline) and math.isfinite(selected)):
        return None
    return f"{1 if baseline <= threshold else 0}{1 if selected <= threshold else 0}"


def enrich_records(
    effectiveness_records: Sequence[Mapping[str, Any]],
    results_by_key: Mapping[Tuple[str, int, int], Mapping[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in effectiveness_records:
        info = final_hook_info(record)
        result = results_by_key.get(record_key(record), {})
        delta = distance_delta(record)
        rows.append(
            {
                "scene_name": record.get("scene_name"),
                "episode_id": safe_int(record.get("episode_id")),
                "task_id": safe_int(record.get("task_id")),
                "task_level": record.get("task_level") or "_missing",
                "navigation_type": record.get("navigation_type"),
                "module_hook_called": safe_int(record.get("module_hook_called")),
                "module_hook_applied": safe_int(record.get("module_hook_applied")),
                "module_helpful": record.get("module_helpful"),
                "baseline_l2": safe_float(record.get("baseline_target_to_goal_l2"), default=float("nan")),
                "selected_l2": safe_float(record.get("selected_target_to_goal_l2"), default=float("nan")),
                "delta_l2": delta,
                "delta_label": delta_label(delta),
                "threshold_case": threshold_case(record, threshold),
                "sr": result.get("sr"),
                "spl": result.get("spl"),
                "end_reason": result.get("end_reason"),
                "steps_total": result.get("steps_total"),
                "task_decision_count": result.get("task_decision_count"),
                "end_goal_geo": result.get("end_goal_geo"),
                "reason": info.get("reason"),
                "ok": info.get("ok"),
                "applied": info.get("applied"),
                "selected_gain": info.get("selected_gain"),
                "target_confidence": info.get("target_confidence"),
                "region_margin": info.get("region_margin"),
                "baseline_object_index": info.get("baseline_object_index"),
                "selected_object_index": info.get("selected_object_index"),
                "parse_ok": (info.get("decomposition") or {}).get("parse_ok")
                if isinstance(info.get("decomposition"), dict)
                else None,
                "decomposition_source": (info.get("decomposition") or {}).get("source")
                if isinstance(info.get("decomposition"), dict)
                else None,
                "hook_log_count": len(hook_infos(record)),
            }
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    called = sum(1 for row in rows if safe_int(row.get("module_hook_called")) > 0)
    applied = sum(1 for row in rows if safe_int(row.get("module_hook_applied")) > 0)
    valid = [row for row in rows if row.get("delta_l2") is not None]
    improved = sum(1 for row in valid if row.get("delta_label") == "improved")
    worsened = sum(1 for row in valid if row.get("delta_label") == "worsened")
    tied = sum(1 for row in valid if row.get("delta_label") == "tied")
    applied_rows = [row for row in valid if safe_int(row.get("module_hook_applied")) > 0]
    deltas = [safe_float(row.get("delta_l2")) for row in valid]
    applied_deltas = [safe_float(row.get("delta_l2")) for row in applied_rows]
    sr_values = [safe_float(row.get("sr")) for row in rows if row.get("sr") is not None]
    spl_values = [safe_float(row.get("spl")) for row in rows if row.get("spl") is not None]
    return {
        "count": n,
        "called": called,
        "applied": applied,
        "not_called": n - called,
        "not_applied": n - applied,
        "valid_l2": len(valid),
        "improved": improved,
        "worsened": worsened,
        "tied": tied,
        "invalid_l2": n - len(valid),
        "helpful_true": sum(1 for row in rows if row.get("module_helpful") is True),
        "helpful_false": sum(1 for row in rows if row.get("module_helpful") is False),
        "helpful_none": sum(1 for row in rows if row.get("module_helpful") is None),
        "applied_improved": sum(1 for row in applied_rows if row.get("delta_label") == "improved"),
        "applied_worsened": sum(1 for row in applied_rows if row.get("delta_label") == "worsened"),
        "applied_tied": sum(1 for row in applied_rows if row.get("delta_label") == "tied"),
        "mean_delta_l2": mean(deltas),
        "median_delta_l2": percentile(deltas, 0.5),
        "min_delta_l2": min(deltas) if deltas else 0.0,
        "max_delta_l2": max(deltas) if deltas else 0.0,
        "mean_applied_delta_l2": mean(applied_deltas),
        "sr": mean(sr_values),
        "spl": mean(spl_values),
        "success_count": sum(1 for value in sr_values if value >= 0.5),
        "has_result_metrics": bool(sr_values),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def group_summary(rows: Sequence[Mapping[str, Any]], field: str, order: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        key = str(value if value not in (None, "") else "_missing")
        buckets[key].append(row)
    return {key: summarize_rows(buckets[key]) for key in ordered_keys(buckets.keys(), order)}


def counter_from_rows(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        value = row.get(field)
        key = str(value if value not in (None, "") else "_missing")
        counts[key] += 1
    return dict(counts)


def threshold_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(row.get("threshold_case")) for row in rows if row.get("threshold_case") is not None)
    total = sum(counts.values())
    baseline_success = counts["10"] + counts["11"]
    selected_success = counts["01"] + counts["11"]
    return {
        "count": total,
        "case_counts": {case: counts[case] for case in CASE_ORDER},
        "baseline_success": baseline_success,
        "selected_success": selected_success,
        "rescued_01": counts["01"],
        "regressed_10": counts["10"],
        "both_fail_00": counts["00"],
        "both_success_11": counts["11"],
        "net_success_delta": selected_success - baseline_success,
        "baseline_success_rate": baseline_success / total if total else 0.0,
        "selected_success_rate": selected_success / total if total else 0.0,
    }


def top_examples(rows: Sequence[Mapping[str, Any]], top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid = [row for row in rows if row.get("delta_l2") is not None]
    ordered = sorted(valid, key=lambda row: safe_float(row.get("delta_l2")))
    improved = [example_row(row) for row in ordered[:top_k]]
    worsened = [example_row(row) for row in reversed(ordered[-top_k:])] if top_k > 0 else []
    return improved, worsened


def example_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scene_name": row.get("scene_name"),
        "episode_id": row.get("episode_id"),
        "task_id": row.get("task_id"),
        "task_level": row.get("task_level"),
        "sr": row.get("sr"),
        "spl": row.get("spl"),
        "applied": row.get("module_hook_applied"),
        "helpful": row.get("module_helpful"),
        "reason": row.get("reason"),
        "baseline_l2": row.get("baseline_l2"),
        "selected_l2": row.get("selected_l2"),
        "delta_l2": row.get("delta_l2"),
        "threshold_case": row.get("threshold_case"),
        "baseline_object_index": row.get("baseline_object_index"),
        "selected_object_index": row.get("selected_object_index"),
    }


def build_report(
    effectiveness_path: Path,
    result_path: Optional[Path],
    threshold: float,
    top_k: int,
) -> Dict[str, Any]:
    effectiveness_json = load_json(effectiveness_path)
    assert effectiveness_json is not None
    records_raw = effectiveness_json.get("records", [])
    if not isinstance(records_raw, list):
        raise ValueError(f"Missing list field 'records': {effectiveness_path}")
    records = [row for row in records_raw if isinstance(row, dict)]

    result_json = load_json(result_path)
    rows = enrich_records(records, result_index(result_json), threshold)
    top_improved, top_worsened = top_examples(rows, top_k)
    hook_count = sum(safe_int(row.get("hook_log_count")) for row in rows)

    return {
        "meta": {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "effectiveness_json": str(effectiveness_path),
            "result_json": None if result_path is None else str(result_path),
            "threshold_m": float(threshold),
            "record_count": len(rows),
            "hook_log_count": hook_count,
        },
        "overall": summarize_rows(rows),
        "by_level": group_summary(rows, "task_level", LEVEL_ORDER),
        "by_reason": group_summary(rows, "reason"),
        "by_threshold_case": group_summary(rows, "threshold_case", CASE_ORDER),
        "threshold_effectiveness": threshold_summary(rows),
        "reason_counts": counter_from_rows(rows, "reason"),
        "parse_ok_counts": counter_from_rows(rows, "parse_ok"),
        "decomposition_source_counts": counter_from_rows(rows, "decomposition_source"),
        "top_improved_by_l2": top_improved,
        "top_worsened_by_l2": top_worsened,
    }


def format_counter(counts: Mapping[str, int], order: Sequence[str] = ()) -> str:
    return ", ".join(f"{key}:{safe_int(counts.get(key))}" for key in ordered_keys(counts.keys(), order))


def append_summary_line(lines: List[str], name: str, stats: Mapping[str, Any]) -> None:
    n = safe_int(stats.get("count"))
    called = safe_int(stats.get("called"))
    applied = safe_int(stats.get("applied"))
    improved = safe_int(stats.get("improved"))
    worsened = safe_int(stats.get("worsened"))
    tied = safe_int(stats.get("tied"))
    sr_text = ""
    if stats.get("has_result_metrics"):
        sr_text = (
            f"  SR={100.0 * safe_float(stats.get('sr')):>6.2f}%"
            f" ({safe_int(stats.get('success_count'))}/{n})"
            f"  SPL={safe_float(stats.get('spl')):.6f}"
        )
    lines.append(
        f"{name:<24} n={n:>3}  called={called:>3} ({pct(called, n)})"
        f"  applied={applied:>3} ({pct(applied, n)})"
        f"  imp/worse/tie={improved}/{worsened}/{tied}"
        f"  mean_delta={fmt_float(stats.get('mean_delta_l2'), 4)}m"
        f"  applied_mean_delta={fmt_float(stats.get('mean_applied_delta_l2'), 4)}m"
        f"{sr_text}"
    )


def format_example(row: Mapping[str, Any]) -> str:
    return (
        f"scene={row.get('scene_name')} ep={row.get('episode_id')} task={row.get('task_id')}"
        f" level={row.get('task_level')} case={row.get('threshold_case')}"
        f" delta={fmt_float(row.get('delta_l2'), 3)}m"
        f" l2={fmt_float(row.get('baseline_l2'), 3)}->{fmt_float(row.get('selected_l2'), 3)}"
        f" applied={row.get('applied')} helpful={row.get('helpful')}"
        f" SR={fmt_float(row.get('sr'), 1)} reason={row.get('reason')}"
        f" obj={row.get('baseline_object_index')}->{row.get('selected_object_index')}"
    )


def format_report(report: Mapping[str, Any], top_k: int, max_reasons: int) -> str:
    meta = report["meta"]
    lines: List[str] = []
    lines.append("== Inputs ==")
    lines.append(f"effectiveness_json={meta['effectiveness_json']}")
    lines.append(f"result_json={meta['result_json']}")
    lines.append(f"threshold_m={fmt_float(meta['threshold_m'], 3)}")
    lines.append("")

    lines.append("== Overall MQSC effectiveness ==")
    append_summary_line(lines, "overall", report["overall"])
    overall = report["overall"]
    applied = safe_int(overall.get("applied"))
    lines.append(
        f"helpful among applied: true={overall['helpful_true']} ({pct(overall['helpful_true'], applied)})"
        f"  false={overall['helpful_false']} ({pct(overall['helpful_false'], applied)})"
        f"  none={overall['helpful_none']}"
    )
    lines.append(
        f"delta stats: median={fmt_float(overall['median_delta_l2'], 4)}m"
        f"  min={fmt_float(overall['min_delta_l2'], 4)}m"
        f"  max={fmt_float(overall['max_delta_l2'], 4)}m"
    )
    lines.append("")

    thresh = report["threshold_effectiveness"]
    total = safe_int(thresh["count"])
    lines.append("== Threshold case (baseline/selected target L2 within threshold) ==")
    lines.append(f"cases: {format_counter(thresh['case_counts'], CASE_ORDER)}  total={total}")
    lines.append(
        f"baseline_success={thresh['baseline_success']}/{total} ({pct(thresh['baseline_success'], total)})"
        f"  selected_success={thresh['selected_success']}/{total} ({pct(thresh['selected_success'], total)})"
        f"  net={thresh['net_success_delta']:+d}"
    )
    lines.append(
        f"rescued(01)={thresh['rescued_01']}  regressed(10)={thresh['regressed_10']}"
        f"  both_success(11)={thresh['both_success_11']}  both_fail(00)={thresh['both_fail_00']}"
    )
    lines.append("")

    lines.append("== By task level ==")
    for level, stats in report["by_level"].items():
        append_summary_line(lines, level, stats)
    lines.append("")

    lines.append("== Hook reason counts ==")
    reason_counts = Counter(report["reason_counts"])
    for reason, count in reason_counts.most_common(max_reasons):
        lines.append(f"{reason:<56} {count}")
    lines.append(f"parse_ok: {format_counter(report['parse_ok_counts'])}")
    lines.append(f"decomposition_source: {format_counter(report['decomposition_source_counts'])}")

    if top_k > 0:
        lines.append("")
        lines.append(f"== Top {top_k} improved by target L2 ==")
        for row in report["top_improved_by_l2"]:
            lines.append(format_example(row))
        lines.append("")
        lines.append(f"== Top {top_k} worsened by target L2 ==")
        for row in report["top_worsened_by_l2"]:
            lines.append(format_example(row))

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MQSC effectiveness JSON.")
    parser.add_argument("--effectiveness-json", type=Path, default=DEFAULT_EFFECTIVENESS_JSON)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--no-auto-result", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0, help="L2 threshold for 00/01/10/11 case stats")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-reasons", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    effectiveness_path = resolve_path(args.effectiveness_json)
    if not effectiveness_path.is_file():
        raise FileNotFoundError(f"effectiveness JSON not found: {effectiveness_path}")

    result_path = resolve_path(args.result_json) if args.result_json is not None else None
    if result_path is None and not args.no_auto_result:
        result_path = auto_result_json(effectiveness_path)

    report = build_report(
        effectiveness_path=effectiveness_path,
        result_path=result_path,
        threshold=float(args.threshold),
        top_k=max(0, int(args.top_k)),
    )
    print(format_report(report, top_k=max(0, int(args.top_k)), max_reasons=max(1, int(args.max_reasons))), end="")

    if args.json_out is not None:
        json_out = resolve_path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[mqsc-effect] wrote {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
