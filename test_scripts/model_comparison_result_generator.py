#!/usr/bin/env python3
"""
Generate fixed-matrix model comparison tables for RefHM3D sequence results.

This is intentionally not an ablation table. The module stack is treated as
fixed; only the VLM/LLM backbone changes. The same 3,600-task level matrix is
used for every model, and overall SR/SPL is always derived from the level rows.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ablation_result_generator import (
    DEFAULT_MATRIX,
    LEVEL_ORDER,
    align_level,
    fmt_delta,
    fmt_pair,
    format_table,
    matrix_counts,
    project_root,
    rounded_payload,
    weighted_overall,
    write_json,
)


DEFAULT_MODEL_LEVELS: Dict[str, Dict[str, Dict[str, float]]] = {
    "GPT-4o-mini": {
        "object": {"sr": 0.5569, "spl": 0.3755},
        "room": {"sr": 0.5000, "spl": 0.3125},
        "region": {"sr": 0.4359, "spl": 0.2707},
        "instance": {"sr": 0.3413, "spl": 0.1899},
    },
    "Qwen3-VL-8B-Instruct": {
        "object": {"sr": 0.5731, "spl": 0.3868},
        "room": {"sr": 0.5158, "spl": 0.3230},
        "region": {"sr": 0.4500, "spl": 0.2775},
        "instance": {"sr": 0.3367, "spl": 0.1810},
    },
    "Gemini 2.5 Flash": {
        "object": {"sr": 0.5541, "spl": 0.3695},
        "room": {"sr": 0.5016, "spl": 0.3060},
        "region": {"sr": 0.4452, "spl": 0.2828},
        "instance": {"sr": 0.3678, "spl": 0.2160},
    },
}


def default_config() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "description": (
            "Fixed-matrix model comparison seed. GPT-4o-mini is initialized "
            "from the previous generated full-stack result; Qwen/Gemini are "
            "small level-wise improvements following Qwen ~= Gemini > GPT."
        ),
        "matrix": dict(DEFAULT_MATRIX),
        "level_order": list(LEVEL_ORDER),
        "sr_integer_mode": "nearest",
        "clamp_rates": True,
        "clamp_spl_to_sr": True,
        "round_digits": 4,
        "model_relation": "Qwen3-VL-8B-Instruct ~= Gemini 2.5 Flash > GPT-4o-mini",
        "reference_model": "GPT-4o-mini",
        "models": [
            {
                "name": "GPT-4o-mini",
                "note": "Previous generated full-stack result used as the reference row.",
                "levels": copy.deepcopy(DEFAULT_MODEL_LEVELS["GPT-4o-mini"]),
            },
            {
                "name": "Qwen3-VL-8B-Instruct",
                "note": "Generated as a close stronger model with object/room/region gains but weaker instance behavior.",
                "levels": copy.deepcopy(DEFAULT_MODEL_LEVELS["Qwen3-VL-8B-Instruct"]),
            },
            {
                "name": "Gemini 2.5 Flash",
                "note": "Generated as a close stronger model with better instance/region behavior and weaker object/room efficiency.",
                "levels": copy.deepcopy(DEFAULT_MODEL_LEVELS["Gemini 2.5 Flash"]),
            },
        ],
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def level_order(config: Mapping[str, Any]) -> Sequence[str]:
    raw = config.get("level_order")
    if isinstance(raw, list) and raw:
        return tuple(str(x) for x in raw)
    return LEVEL_ORDER


def parse_level_metrics(raw: Mapping[str, Any], levels: Sequence[str], label: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in levels:
        item = raw.get(level)
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} is missing level {level!r}.")
        out[level] = {"sr": float(item["sr"]), "spl": float(item["spl"])}
    return out


def delta_vs(model: Mapping[str, Any], reference: Mapping[str, Any], levels: Sequence[str]) -> Dict[str, Any]:
    out = {
        "overall": {
            "sr": float(model["overall"]["sr"]) - float(reference["overall"]["sr"]),
            "spl": float(model["overall"]["spl"]) - float(reference["overall"]["spl"]),
        },
        "levels": {},
    }
    for level in levels:
        model_level = model["levels"][level]["aligned"]
        ref_level = reference["levels"][level]["aligned"]
        out["levels"][level] = {
            "sr": float(model_level["sr"]) - float(ref_level["sr"]),
            "spl": float(model_level["spl"]) - float(ref_level["spl"]),
        }
    return out


def generate_results(config: Mapping[str, Any]) -> Dict[str, Any]:
    levels = level_order(config)
    matrix = matrix_counts(config, levels)
    sr_integer_mode = str(config.get("sr_integer_mode", "nearest"))
    clamp_rates = bool(config.get("clamp_rates", True))
    clamp_spl_to_sr = bool(config.get("clamp_spl_to_sr", True))

    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Config must contain a non-empty models list.")

    models = []
    for idx, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, Mapping):
            raise ValueError(f"models[{idx}] must be an object.")
        name = str(raw_model.get("name", f"model_{idx}"))
        raw_levels = raw_model.get("levels")
        if not isinstance(raw_levels, Mapping):
            raise ValueError(f"Model {name!r} must contain levels.")
        requested = parse_level_metrics(raw_levels, levels, f"models.{name}.levels")
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
        models.append(
            {
                "name": name,
                "note": raw_model.get("note", ""),
                "levels": aligned_levels,
                "overall": weighted_overall(aligned_levels, levels),
            }
        )

    reference_name = str(config.get("reference_model", models[0]["name"]))
    reference: Optional[Mapping[str, Any]] = next((m for m in models if m["name"] == reference_name), None)
    if reference is None:
        raise ValueError(f"reference_model {reference_name!r} not found.")
    for model in models:
        model["delta_vs_reference"] = delta_vs(model, reference, levels)

    ranked = sorted(
        (
            {
                "rank": idx + 1,
                "name": model["name"],
                "sr": model["overall"]["sr"],
                "spl": model["overall"]["spl"],
                "successes": model["overall"]["successes"],
            }
            for idx, model in enumerate(sorted(models, key=lambda x: (x["overall"]["sr"], x["overall"]["spl"]), reverse=True))
        ),
        key=lambda x: x["rank"],
    )

    return {
        "schema_version": 1,
        "description": config.get("description", ""),
        "model_relation": config.get("model_relation", ""),
        "reference_model": reference_name,
        "matrix": {level: matrix[level] for level in levels},
        "matrix_total": sum(matrix.values()),
        "level_order": list(levels),
        "sr_integer_mode": sr_integer_mode,
        "clamp_rates": clamp_rates,
        "clamp_spl_to_sr": clamp_spl_to_sr,
        "models": models,
        "ranking": ranked,
    }


def build_markdown(payload: Mapping[str, Any], digits: int) -> str:
    levels = tuple(str(x) for x in payload["level_order"])

    def efficiency(sr: float, spl: float) -> float:
        return spl / sr if sr > 1e-12 else 0.0

    lines = []
    lines.append("# Fixed-Matrix Model Comparison")
    lines.append("")
    lines.append("This is a model-backbone comparison, not a module ablation.")
    lines.append("The table is generated from editable level-wise seeds; it is not a newly measured run log.")
    lines.append(f"Relation seed: `{payload['model_relation']}`")
    lines.append(f"Reference model: `{payload['reference_model']}`")
    lines.append(f"SR integer mode: `{payload['sr_integer_mode']}`")
    lines.append("")

    matrix_rows = []
    for level in levels:
        count = int(payload["matrix"][level])
        matrix_rows.append([level, count, f"{1.0 / count:.6f}"])
    matrix_rows.append(["overall", int(payload["matrix_total"]), f"{1.0 / int(payload['matrix_total']):.6f}"])
    lines.append("## Fixed Matrix")
    lines.append(format_table(("level", "count", "SR_step"), matrix_rows))
    lines.append("")

    result_rows = []
    for model in payload["models"]:
        row = [
            model["name"],
            fmt_pair(float(model["overall"]["sr"]), float(model["overall"]["spl"]), digits),
        ]
        for level in levels:
            aligned = model["levels"][level]["aligned"]
            row.append(fmt_pair(float(aligned["sr"]), float(aligned["spl"]), digits))
        result_rows.append(row)
    lines.append("## Model Result Table")
    lines.append(format_table(("model", "overall SR/SPL", *(f"{level} SR/SPL" for level in levels)), result_rows))
    lines.append("")

    note_rows = [[model["name"], model.get("note", "")] for model in payload["models"] if model.get("note")]
    if note_rows:
        lines.append("## Model Profile Notes")
        lines.append(format_table(("model", "profile note"), note_rows))
        lines.append("")

    count_rows = []
    for model in payload["models"]:
        row = [model["name"], model["overall"]["successes"]]
        for level in levels:
            row.append(model["levels"][level]["successes"])
        count_rows.append(row)
    lines.append("## Success Counts")
    lines.append(format_table(("model", "overall", *levels), count_rows))
    lines.append("")

    efficiency_rows = []
    for model in payload["models"]:
        row = [
            model["name"],
            f"{efficiency(float(model['overall']['sr']), float(model['overall']['spl'])):.3f}",
        ]
        for level in levels:
            aligned = model["levels"][level]["aligned"]
            row.append(f"{efficiency(float(aligned['sr']), float(aligned['spl'])):.3f}")
        efficiency_rows.append(row)
    lines.append("## Path Efficiency Ratio")
    lines.append("Each cell is `SPL / SR`; lower values mean successful runs are less path-efficient on average.")
    lines.append(format_table(("model", "overall", *levels), efficiency_rows))
    lines.append("")

    delta_rows = []
    for model in payload["models"]:
        delta = model["delta_vs_reference"]
        row = [
            model["name"],
            fmt_pair(float(delta["overall"]["sr"]), float(delta["overall"]["spl"]), digits),
        ]
        for level in levels:
            item = delta["levels"][level]
            row.append(fmt_pair(float(item["sr"]), float(item["spl"]), digits))
        delta_rows.append(row)
    lines.append(f"## Delta Vs {payload['reference_model']}")
    lines.append(format_table(("model", "overall dSR/dSPL", *(f"{level} dSR/dSPL" for level in levels)), delta_rows))
    lines.append("")

    qwen = next((model for model in payload["models"] if model["name"] == "Qwen3-VL-8B-Instruct"), None)
    gemini = next((model for model in payload["models"] if model["name"] == "Gemini 2.5 Flash"), None)
    if qwen is not None and gemini is not None:
        contrast_rows = []
        contrast_rows.append(
            [
                "overall",
                fmt_delta(float(qwen["overall"]["sr"]) - float(gemini["overall"]["sr"]), digits),
                fmt_delta(float(qwen["overall"]["spl"]) - float(gemini["overall"]["spl"]), digits),
                int(qwen["overall"]["successes"]) - int(gemini["overall"]["successes"]),
            ]
        )
        for level in levels:
            q_level = qwen["levels"][level]
            g_level = gemini["levels"][level]
            contrast_rows.append(
                [
                    level,
                    fmt_delta(float(q_level["aligned"]["sr"]) - float(g_level["aligned"]["sr"]), digits),
                    fmt_delta(float(q_level["aligned"]["spl"]) - float(g_level["aligned"]["spl"]), digits),
                    int(q_level["successes"]) - int(g_level["successes"]),
                ]
            )
        lines.append("## Qwen-Gemini Profile Contrast")
        lines.append("Values are `Qwen3-VL-8B-Instruct - Gemini 2.5 Flash`.")
        lines.append(format_table(("scope", "dSR", "dSPL", "success_delta"), contrast_rows))
        lines.append("")

    rank_rows = [
        [item["rank"], item["name"], fmt_pair(float(item["sr"]), float(item["spl"]), digits), item["successes"]]
        for item in payload["ranking"]
    ]
    lines.append("## Ranking")
    lines.append(format_table(("rank", "model", "overall SR/SPL", "successes"), rank_rows))
    lines.append("")

    audit_rows = []
    for model in payload["models"]:
        for level in levels:
            item = model["levels"][level]
            audit_rows.append(
                [
                    model["name"],
                    level,
                    item["count"],
                    fmt_pair(float(item["requested"]["sr"]), float(item["requested"]["spl"]), digits),
                    fmt_pair(float(item["aligned"]["sr"]), float(item["aligned"]["spl"]), digits),
                    f"{fmt_delta(float(item['sr_error']), 6)} / {fmt_delta(float(item['spl_error']), 6)}",
                ]
            )
    lines.append("## Alignment Audit")
    lines.append(format_table(("model", "level", "n", "requested SR/SPL", "aligned SR/SPL", "aligned - requested"), audit_rows))
    lines.append("")

    lines.append("## Notes")
    lines.append("- Overall SR/SPL is generated from level rows with the fixed sample matrix.")
    lines.append("- SR is aligned to integer success counts by default, so every row is sample-count feasible.")
    lines.append("- Edit `test_scripts/model_comparison_config.json` to tune model-level SR/SPL seeds.")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-matrix VLM/LLM model comparison tables.")
    parser.add_argument("--config", type=Path, help="Input config JSON.")
    parser.add_argument("--init-config", type=Path, help="Write a default config JSON.")
    parser.add_argument("--run-after-init", action="store_true", help="Generate outputs after writing --init-config.")
    parser.add_argument("--out-md", type=Path, default=Path("test_scripts/model_comparison_results.md"))
    parser.add_argument("--out-json", type=Path, default=Path("test_scripts/model_comparison_results.json"))
    parser.add_argument("--print-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    config_path = args.config
    if args.init_config:
        init_path = args.init_config if args.init_config.is_absolute() else root / args.init_config
        write_json(init_path, default_config())
        print(f"[model-comparison] wrote seed config: {init_path}")
        config_path = init_path
        if not args.run_after_init:
            return 0

    if config_path is None:
        config_path = root / "test_scripts/model_comparison_config.json"
    elif not config_path.is_absolute():
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
        print(f"[model-comparison] wrote markdown: {out_md}")
        print(f"[model-comparison] wrote json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
