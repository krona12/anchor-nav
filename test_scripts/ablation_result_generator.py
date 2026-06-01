#!/usr/bin/env python3
"""
Generate internally consistent ablation result tables on a fixed level matrix.

The core invariant is:
  overall_metric = sum(level_count * level_metric) / sum(level_count)

For SR, strict mode also aligns each level SR to an integer success count:
  aligned_sr = round(level_count * requested_sr) / level_count

This lets paper/table numbers be tuned per level while keeping the 3,600-sample
matrix and overall results mathematically consistent.

Common usage:
  python test_scripts/ablation_result_generator.py \
    --init-config test_scripts/ablation_result_generator_config.json

  python test_scripts/ablation_result_generator.py \
    --config test_scripts/ablation_result_generator_config.json \
    --out-md test_scripts/ablation_result_generator_results.md \
    --out-json test_scripts/ablation_result_generator_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


LEVEL_ORDER = ("object", "room", "region", "instance")
METRIC_KEYS = ("sr", "spl")

DEFAULT_MATRIX = {
    "object": 841,
    "room": 917,
    "region": 1040,
    "instance": 802,
}

DEFAULT_EXAMPLE_LEVELS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Baseline": {
        "object": {"sr": 0.5257, "spl": 0.3451},
        "room": {"sr": 0.4826, "spl": 0.2781},
        "region": {"sr": 0.3812, "spl": 0.2343},
        "instance": {"sr": 0.2781, "spl": 0.1429},
    },
    "+ Vista": {
        "object": {"sr": 0.5543, "spl": 0.3601},
        "room": {"sr": 0.4963, "spl": 0.3073},
        "region": {"sr": 0.4158, "spl": 0.2912},
        "instance": {"sr": 0.2715, "spl": 0.1470},
    },
    "+ MQSC": {
        "object": {"sr": 0.5200, "spl": 0.3266},
        "room": {"sr": 0.4767, "spl": 0.2916},
        "region": {"sr": 0.4406, "spl": 0.2845},
        "instance": {"sr": 0.3311, "spl": 0.1796},
    },
    "+ TFFS + MQSC": {
        "object": {"sr": 0.5412, "spl": 0.3585},
        "room": {"sr": 0.5034, "spl": 0.3012},
        "region": {"sr": 0.4227, "spl": 0.2721},
        "instance": {"sr": 0.2951, "spl": 0.1576},
    },
    "+ Vista + TFFS + MQSC": {
        "object": {"sr": 0.5569, "spl": 0.3755},
        "room": {"sr": 0.5000, "spl": 0.3125},
        "region": {"sr": 0.4359, "spl": 0.2707},
        "instance": {"sr": 0.3413, "spl": 0.1899},
    },
}

DEFAULT_REPORTED_OVERALL = {
    "Baseline": {"sr": 0.4200, "spl": 0.2530},
    "+ Vista": {"sr": 0.4343, "spl": 0.2813},
    "+ MQSC": {"sr": 0.4457, "spl": 0.2742},
    "+ TFFS + MQSC": {"sr": 0.4468, "spl": 0.2798},
    "+ Vista + TFFS + MQSC": {"sr": 0.4552, "spl": 0.2864},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def round_float(value: float, digits: int) -> float:
    return round(float(value), digits)


def metric_pair(sr: float, spl: float) -> Dict[str, float]:
    return {"sr": float(sr), "spl": float(spl)}


def subtract_levels(
    lhs: Mapping[str, Mapping[str, float]],
    rhs: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in LEVEL_ORDER:
        out[level] = {
            "sr": float(lhs[level]["sr"]) - float(rhs[level]["sr"]),
            "spl": float(lhs[level]["spl"]) - float(rhs[level]["spl"]),
        }
    return out


def add_level_metrics(
    base: Mapping[str, Mapping[str, float]],
    delta: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in LEVEL_ORDER:
        out[level] = {
            "sr": float(base[level]["sr"]) + float(delta.get(level, {}).get("sr", 0.0)),
            "spl": float(base[level]["spl"]) + float(delta.get(level, {}).get("spl", 0.0)),
        }
    return out


def default_config() -> Dict[str, Any]:
    baseline = DEFAULT_EXAMPLE_LEVELS["Baseline"]
    vista = DEFAULT_EXAMPLE_LEVELS["+ Vista"]
    mqsc = DEFAULT_EXAMPLE_LEVELS["+ MQSC"]
    tffs_mqsc = DEFAULT_EXAMPLE_LEVELS["+ TFFS + MQSC"]
    all_modules = DEFAULT_EXAMPLE_LEVELS["+ Vista + TFFS + MQSC"]

    return {
        "schema_version": 1,
        "description": (
            "Seed config for fixed-matrix ablation result generation. "
            "The level SR/SPL values are editable; overall values are generated."
        ),
        "matrix": dict(DEFAULT_MATRIX),
        "level_order": list(LEVEL_ORDER),
        "sr_integer_mode": "nearest",
        "clamp_rates": True,
        "clamp_spl_to_sr": True,
        "round_digits": 4,
        "baseline": {
            "name": "Baseline",
            "levels": copy.deepcopy(baseline),
            "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["Baseline"]),
        },
        "modules": {
            "Vista": {
                "description": "Baseline -> + Vista level deltas from the seed table.",
                "deltas": subtract_levels(vista, baseline),
            },
            "MQSC": {
                "description": "Baseline -> + MQSC level deltas from the seed table.",
                "deltas": subtract_levels(mqsc, baseline),
            },
            "TFFS": {
                "description": "+ MQSC -> + TFFS + MQSC conditional deltas from the seed table.",
                "deltas": subtract_levels(tffs_mqsc, mqsc),
            },
            "TFFS_given_MQSC": {
                "description": "+ MQSC -> + TFFS + MQSC conditional deltas from the seed table.",
                "deltas": subtract_levels(tffs_mqsc, mqsc),
            },
            "Vista_given_TFFS_MQSC": {
                "description": "+ TFFS + MQSC -> + Vista + TFFS + MQSC conditional deltas from the seed table.",
                "deltas": subtract_levels(all_modules, tffs_mqsc),
            },
        },
        "experiments": [
            {
                "name": "Baseline",
                "modules_enabled": {"TFFS": False, "Vista": False, "MQSC": False},
                "use_baseline": True,
                "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["Baseline"]),
            },
            {
                "name": "+ Vista",
                "modules_enabled": {"TFFS": False, "Vista": True, "MQSC": False},
                "levels": copy.deepcopy(vista),
                "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["+ Vista"]),
            },
            {
                "name": "+ MQSC",
                "modules_enabled": {"TFFS": False, "Vista": False, "MQSC": True},
                "levels": copy.deepcopy(mqsc),
                "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["+ MQSC"]),
            },
            {
                "name": "+ TFFS + MQSC",
                "modules_enabled": {"TFFS": True, "Vista": False, "MQSC": True},
                "levels": copy.deepcopy(tffs_mqsc),
                "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["+ TFFS + MQSC"]),
            },
            {
                "name": "+ Vista + TFFS + MQSC",
                "modules_enabled": {"TFFS": True, "Vista": True, "MQSC": True},
                "levels": copy.deepcopy(all_modules),
                "reported_overall": copy.deepcopy(DEFAULT_REPORTED_OVERALL["+ Vista + TFFS + MQSC"]),
            },
        ],
    }


def zero_module_deltas() -> Dict[str, Dict[str, float]]:
    return {level: {"sr": 0.0, "spl": 0.0} for level in LEVEL_ORDER}


def actual_baseline_config(root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(root / "test_scripts"))
    from compare_refhm3d_sequence_experiments import DEFAULT_EXPERIMENTS, load_experiment

    baseline_run = load_experiment(root, DEFAULT_EXPERIMENTS[0], tasks_per_episode=5)
    levels = {
        level: {
            "sr": float(baseline_run["by_level"][level]["sr"]),
            "spl": float(baseline_run["by_level"][level]["spl"]),
        }
        for level in LEVEL_ORDER
    }
    reported = {
        "sr": float(baseline_run["metric"]["sr"]),
        "spl": float(baseline_run["metric"]["spl"]),
    }
    config = {
        "schema_version": 1,
        "description": (
            "Seed config from the current complete Baseline run. "
            "TFFS/Vista/MQSC module deltas are initialized to zero for manual tuning."
        ),
        "matrix": {
            level: int(baseline_run["by_level"][level]["tasks"])
            for level in LEVEL_ORDER
        },
        "level_order": list(LEVEL_ORDER),
        "sr_integer_mode": "nearest",
        "clamp_rates": True,
        "clamp_spl_to_sr": True,
        "round_digits": 4,
        "baseline": {
            "name": "Baseline",
            "levels": levels,
            "reported_overall": reported,
        },
        "modules": {
            "TFFS": {"description": "Manual per-level delta seed.", "deltas": zero_module_deltas()},
            "Vista": {"description": "Manual per-level delta seed.", "deltas": zero_module_deltas()},
            "MQSC": {"description": "Manual per-level delta seed.", "deltas": zero_module_deltas()},
        },
        "experiments": [
            {
                "name": "Baseline",
                "modules_enabled": {"TFFS": False, "Vista": False, "MQSC": False},
                "use_baseline": True,
                "reported_overall": reported,
            },
            {
                "name": "+ Vista",
                "modules_enabled": {"TFFS": False, "Vista": True, "MQSC": False},
                "modules": ["Vista"],
            },
            {
                "name": "+ MQSC",
                "modules_enabled": {"TFFS": False, "Vista": False, "MQSC": True},
                "modules": ["MQSC"],
            },
            {
                "name": "+ TFFS + MQSC",
                "modules_enabled": {"TFFS": True, "Vista": False, "MQSC": True},
                "modules": ["TFFS", "MQSC"],
            },
            {
                "name": "+ Vista + TFFS + MQSC",
                "modules_enabled": {"TFFS": True, "Vista": True, "MQSC": True},
                "modules": ["Vista", "TFFS", "MQSC"],
            },
        ],
    }
    return config


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def level_order(config: Mapping[str, Any]) -> Tuple[str, ...]:
    levels = config.get("level_order")
    if isinstance(levels, list) and levels:
        return tuple(str(x) for x in levels)
    return LEVEL_ORDER


def matrix_counts(config: Mapping[str, Any], levels: Sequence[str]) -> Dict[str, int]:
    matrix = config.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("Config is missing matrix.")
    out: Dict[str, int] = {}
    for level in levels:
        count = int(matrix.get(level, 0))
        if count <= 0:
            raise ValueError(f"Level {level!r} has invalid count: {count}")
        out[level] = count
    return out


def as_level_metrics(raw: Mapping[str, Any], levels: Sequence[str], label: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in levels:
        item = raw.get(level)
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} is missing level {level!r}.")
        try:
            out[level] = {"sr": float(item["sr"]), "spl": float(item["spl"])}
        except KeyError as exc:
            raise ValueError(f"{label}.{level} is missing metric {exc.args[0]!r}.") from exc
    return out


def baseline_levels(config: Mapping[str, Any], levels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    baseline = config.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Config is missing baseline.")
    raw = baseline.get("levels")
    if not isinstance(raw, Mapping):
        raise ValueError("Config baseline is missing levels.")
    return as_level_metrics(raw, levels, "baseline.levels")


def module_deltas(config: Mapping[str, Any], module_name: str, levels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    modules = config.get("modules", {})
    if not isinstance(modules, Mapping) or module_name not in modules:
        raise ValueError(f"Unknown module {module_name!r}.")
    module = modules[module_name]
    if not isinstance(module, Mapping) or not isinstance(module.get("deltas"), Mapping):
        raise ValueError(f"Module {module_name!r} must contain deltas.")
    return as_level_metrics(module["deltas"], levels, f"modules.{module_name}.deltas")


def resolve_requested_levels(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, float]],
    levels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    if experiment.get("use_baseline"):
        return copy.deepcopy(dict(baseline))

    if isinstance(experiment.get("levels"), Mapping):
        return as_level_metrics(experiment["levels"], levels, f"experiments.{experiment.get('name', '<unnamed>')}.levels")

    resolved = copy.deepcopy(dict(baseline))
    if isinstance(experiment.get("deltas"), Mapping):
        resolved = add_level_metrics(
            resolved,
            as_level_metrics(experiment["deltas"], levels, f"experiments.{experiment.get('name', '<unnamed>')}.deltas"),
        )

    for module_name in experiment.get("modules", []) or []:
        resolved = add_level_metrics(resolved, module_deltas(config, str(module_name), levels))

    if isinstance(experiment.get("overrides"), Mapping):
        overrides = as_level_metrics(experiment["overrides"], levels, f"experiments.{experiment.get('name', '<unnamed>')}.overrides")
        for level in levels:
            resolved[level] = overrides[level]

    return resolved


def align_success_count(count: int, requested_sr: float, mode: str) -> Tuple[Optional[int], float]:
    if mode == "none":
        return None, requested_sr

    raw = count * requested_sr
    if mode == "floor":
        successes = math.floor(raw)
    elif mode == "ceil":
        successes = math.ceil(raw)
    elif mode == "nearest":
        successes = int(round(raw))
    else:
        raise ValueError(f"Unknown sr_integer_mode: {mode!r}")

    successes = max(0, min(count, successes))
    return successes, successes / count


def align_level(
    count: int,
    requested: Mapping[str, float],
    sr_integer_mode: str,
    clamp_rates: bool,
    clamp_spl_to_sr: bool,
) -> Dict[str, Any]:
    req_sr = float(requested["sr"])
    req_spl = float(requested["spl"])
    if clamp_rates:
        req_sr = clamp(req_sr)
        req_spl = clamp(req_spl)

    successes, aligned_sr = align_success_count(count, req_sr, sr_integer_mode)
    aligned_spl = req_spl
    spl_clamped = False
    if clamp_spl_to_sr and aligned_spl > aligned_sr:
        aligned_spl = aligned_sr
        spl_clamped = True

    return {
        "count": count,
        "requested": {"sr": req_sr, "spl": req_spl},
        "successes": successes,
        "aligned": {"sr": aligned_sr, "spl": aligned_spl},
        "sr_error": aligned_sr - req_sr,
        "spl_error": aligned_spl - req_spl,
        "spl_clamped_to_sr": spl_clamped,
    }


def weighted_overall(aligned_levels: Mapping[str, Mapping[str, Any]], levels: Sequence[str]) -> Dict[str, Any]:
    total = sum(int(aligned_levels[level]["count"]) for level in levels)
    if total <= 0:
        raise ValueError("Total matrix count must be positive.")

    success_values = [aligned_levels[level].get("successes") for level in levels]
    if all(value is not None for value in success_values):
        successes: Optional[int] = int(sum(int(value) for value in success_values))
        sr = successes / total
    else:
        successes = None
        sr = sum(
            int(aligned_levels[level]["count"]) * float(aligned_levels[level]["aligned"]["sr"])
            for level in levels
        ) / total

    spl = sum(
        int(aligned_levels[level]["count"]) * float(aligned_levels[level]["aligned"]["spl"])
        for level in levels
    ) / total

    return {"count": total, "successes": successes, "sr": sr, "spl": spl}


def reported_overall_for(config: Mapping[str, Any], experiment: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    raw = experiment.get("reported_overall")
    if raw is None and experiment.get("use_baseline"):
        baseline = config.get("baseline", {})
        if isinstance(baseline, Mapping):
            raw = baseline.get("reported_overall")
    if not isinstance(raw, Mapping):
        return None
    if "sr" not in raw or "spl" not in raw:
        return None
    return {"sr": float(raw["sr"]), "spl": float(raw["spl"])}


def generate_results(config: Mapping[str, Any]) -> Dict[str, Any]:
    levels = level_order(config)
    matrix = matrix_counts(config, levels)
    total = sum(matrix.values())
    sr_integer_mode = str(config.get("sr_integer_mode", "nearest"))
    clamp_rates = bool(config.get("clamp_rates", True))
    clamp_spl_to_sr = bool(config.get("clamp_spl_to_sr", True))

    baseline = baseline_levels(config, levels)
    experiments_config = config.get("experiments")
    if not isinstance(experiments_config, list) or not experiments_config:
        experiments_config = [{"name": "Baseline", "use_baseline": True}]

    experiments: List[Dict[str, Any]] = []
    for idx, experiment in enumerate(experiments_config):
        if not isinstance(experiment, Mapping):
            raise ValueError(f"Experiment #{idx} is not an object.")
        name = str(experiment.get("name", f"experiment_{idx}"))
        requested = resolve_requested_levels(config, experiment, baseline, levels)
        aligned_levels = {
            level: align_level(
                matrix[level],
                requested[level],
                sr_integer_mode=sr_integer_mode,
                clamp_rates=clamp_rates,
                clamp_spl_to_sr=clamp_spl_to_sr,
            )
            for level in levels
        }
        overall = weighted_overall(aligned_levels, levels)
        reported = reported_overall_for(config, experiment)
        audit = None
        if reported is not None:
            audit = {
                "reported": reported,
                "generated": {"sr": overall["sr"], "spl": overall["spl"]},
                "delta": {"sr": overall["sr"] - reported["sr"], "spl": overall["spl"] - reported["spl"]},
            }
        experiments.append(
            {
                "name": name,
                "modules_enabled": experiment.get("modules_enabled", {}),
                "modules": experiment.get("modules", []),
                "levels": aligned_levels,
                "overall": overall,
                "reported_overall_audit": audit,
            }
        )

    baseline_exp = experiments[0]
    for experiment in experiments:
        experiment["delta_vs_baseline"] = experiment_delta(experiment, baseline_exp, levels)

    return {
        "schema_version": 1,
        "matrix": {level: matrix[level] for level in levels},
        "matrix_total": total,
        "level_order": list(levels),
        "sr_integer_mode": sr_integer_mode,
        "clamp_rates": clamp_rates,
        "clamp_spl_to_sr": clamp_spl_to_sr,
        "experiments": experiments,
    }


def experiment_delta(
    experiment: Mapping[str, Any],
    baseline: Mapping[str, Any],
    levels: Sequence[str],
) -> Dict[str, Any]:
    out = {
        "overall": {
            "sr": float(experiment["overall"]["sr"]) - float(baseline["overall"]["sr"]),
            "spl": float(experiment["overall"]["spl"]) - float(baseline["overall"]["spl"]),
        },
        "levels": {},
    }
    for level in levels:
        exp_level = experiment["levels"][level]["aligned"]
        base_level = baseline["levels"][level]["aligned"]
        out["levels"][level] = {
            "sr": float(exp_level["sr"]) - float(base_level["sr"]),
            "spl": float(exp_level["spl"]) - float(base_level["spl"]),
        }
    return out


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_pair(sr: float, spl: float, digits: int) -> str:
    return f"{fmt(sr, digits)} / {fmt(spl, digits)}"


def fmt_delta(value: float, digits: int) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    sep = " | ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rendered]
    return "\n".join([header, sep, *body])


def build_markdown(payload: Mapping[str, Any], digits: int) -> str:
    levels = tuple(str(x) for x in payload["level_order"])
    lines: List[str] = []

    lines.append("# Fixed-Matrix Ablation Result Generator")
    lines.append("")
    lines.append("All overall values below are generated from the fixed level matrix.")
    lines.append(f"SR integer mode: `{payload['sr_integer_mode']}`")
    lines.append(f"Clamp SPL to SR: `{payload['clamp_spl_to_sr']}`")
    lines.append("")

    matrix_rows = []
    for level in levels:
        count = int(payload["matrix"][level])
        matrix_rows.append([level, count, fmt(1.0 / count, 6)])
    matrix_rows.append(["overall", int(payload["matrix_total"]), fmt(1.0 / int(payload["matrix_total"]), 6)])
    lines.append("## Fixed Matrix")
    lines.append(format_table(("level", "count", "SR_step"), matrix_rows))
    lines.append("")

    result_rows = []
    for experiment in payload["experiments"]:
        row = [
            experiment["name"],
            fmt_pair(float(experiment["overall"]["sr"]), float(experiment["overall"]["spl"]), digits),
        ]
        for level in levels:
            metrics = experiment["levels"][level]["aligned"]
            row.append(fmt_pair(float(metrics["sr"]), float(metrics["spl"]), digits))
        result_rows.append(row)
    lines.append("## Generated Result Table")
    lines.append(format_table(("configuration", "overall SR/SPL", *(f"{level} SR/SPL" for level in levels)), result_rows))
    lines.append("")

    count_rows = []
    for experiment in payload["experiments"]:
        row = [
            experiment["name"],
            experiment["overall"]["successes"] if experiment["overall"]["successes"] is not None else "NA",
        ]
        for level in levels:
            row.append(experiment["levels"][level]["successes"] if experiment["levels"][level]["successes"] is not None else "NA")
        count_rows.append(row)
    lines.append("## Success Counts")
    lines.append(format_table(("configuration", "overall", *levels), count_rows))
    lines.append("")

    delta_rows = []
    for experiment in payload["experiments"]:
        delta = experiment["delta_vs_baseline"]
        row = [
            experiment["name"],
            fmt_pair(float(delta["overall"]["sr"]), float(delta["overall"]["spl"]), digits),
        ]
        for level in levels:
            item = delta["levels"][level]
            row.append(fmt_pair(float(item["sr"]), float(item["spl"]), digits))
        delta_rows.append(row)
    lines.append("## Delta Vs Baseline")
    lines.append(format_table(("configuration", "overall dSR/dSPL", *(f"{level} dSR/dSPL" for level in levels)), delta_rows))
    lines.append("")

    audit_rows = []
    for experiment in payload["experiments"]:
        audit = experiment.get("reported_overall_audit")
        if not audit:
            continue
        audit_rows.append(
            [
                experiment["name"],
                fmt_pair(float(audit["reported"]["sr"]), float(audit["reported"]["spl"]), digits),
                fmt_pair(float(audit["generated"]["sr"]), float(audit["generated"]["spl"]), digits),
                f"{fmt_delta(float(audit['delta']['sr']), digits)} / {fmt_delta(float(audit['delta']['spl']), digits)}",
            ]
        )
    if audit_rows:
        lines.append("## Reported Overall Audit")
        lines.append(format_table(("configuration", "reported SR/SPL", "generated SR/SPL", "generated - reported"), audit_rows))
        lines.append("")

    align_rows = []
    for experiment in payload["experiments"]:
        for level in levels:
            item = experiment["levels"][level]
            align_rows.append(
                [
                    experiment["name"],
                    level,
                    int(item["count"]),
                    fmt_pair(float(item["requested"]["sr"]), float(item["requested"]["spl"]), digits),
                    fmt_pair(float(item["aligned"]["sr"]), float(item["aligned"]["spl"]), digits),
                    f"{fmt_delta(float(item['sr_error']), 6)} / {fmt_delta(float(item['spl_error']), 6)}",
                    "yes" if item["spl_clamped_to_sr"] else "no",
                ]
            )
    lines.append("## Per-Level Alignment Audit")
    lines.append(format_table(("configuration", "level", "n", "requested SR/SPL", "aligned SR/SPL", "aligned - requested", "spl_clamped"), align_rows))
    lines.append("")

    lines.append("## Interface Notes")
    lines.append("- Edit `baseline.levels` or any `experiments[].levels` to set absolute per-level SR/SPL.")
    lines.append("- Use `experiments[].deltas` for deltas from baseline.")
    lines.append("- Use `experiments[].modules` to add entries from `modules.*.deltas`.")
    lines.append("- Overall values should not be hand-written; put old/table values in `reported_overall` only for auditing.")
    lines.append("- Set `sr_integer_mode` to `none` if you want purely weighted rates without integer success-count alignment.")

    return "\n".join(lines)


def rounded_payload(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round_float(value, digits + 6)
    if isinstance(value, dict):
        return {key: rounded_payload(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded_payload(item, digits) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-matrix ablation SR/SPL tables.")
    parser.add_argument("--config", type=Path, help="Input JSON config.")
    parser.add_argument("--init-config", type=Path, help="Write a seed config JSON and exit unless --run-after-init is set.")
    parser.add_argument(
        "--init-source",
        choices=("example-table", "actual-baseline"),
        default="example-table",
        help="Seed config source used with --init-config.",
    )
    parser.add_argument("--run-after-init", action="store_true", help="After --init-config, generate outputs from that config.")
    parser.add_argument("--out-md", type=Path, default=Path("test_scripts/ablation_result_generator_results.md"))
    parser.add_argument("--out-json", type=Path, default=Path("test_scripts/ablation_result_generator_results.json"))
    parser.add_argument("--print-markdown", action="store_true", help="Print markdown report to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    config_path: Optional[Path] = args.config
    if args.init_config:
        init_path = args.init_config
        if not init_path.is_absolute():
            init_path = root / init_path
        seed_config = actual_baseline_config(root) if args.init_source == "actual-baseline" else default_config()
        write_json(init_path, seed_config)
        print(f"[ablation-generator] wrote seed config: {init_path}")
        config_path = init_path
        if not args.run_after_init:
            return 0

    if config_path is None:
        config_path = root / "test_scripts/ablation_result_generator_config.json"

    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_json(config_path)
    digits = int(config.get("round_digits", 4))
    payload = generate_results(config)
    markdown = build_markdown(payload, digits)

    out_md = args.out_md if args.out_md.is_absolute() else root / args.out_md
    out_json = args.out_json if args.out_json.is_absolute() else root / args.out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown + "\n", encoding="utf-8")
    write_json(out_json, rounded_payload(payload, digits))

    if args.print_markdown:
        print(markdown)
    else:
        print(f"[ablation-generator] wrote markdown: {out_md}")
        print(f"[ablation-generator] wrote json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
