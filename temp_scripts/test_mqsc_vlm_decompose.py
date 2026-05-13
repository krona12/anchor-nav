from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
HM3D_ONLINE = ROOT / "hm3d-online"
for path in (ROOT, HM3D_ONLINE):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from anchor_nav.mqsc import MqscConfig, build_role_queries, decompose_navigation_text


def build_sentence(
    task_type: str,
    cur_task: Dict[str, Any],
    *,
    goals_map: Dict[str, Any],
    region_map: Dict[str, Any],
    concise: bool,
) -> Tuple[str, str]:
    if task_type == "object":
        return cur_task["object_category"], cur_task["object_category"]
    if task_type == "room":
        return f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}", cur_task["object_category"]
    if task_type == "region":
        region_info = region_map[cur_task["region_id"]]
        desc = (
            region_info.get("shortest_description")
            or region_info.get("concise_description")
            or region_info.get("detailed_description")
            or ""
        ) if concise else (
            region_info.get("comprehensive_description")
            or region_info.get("detailed_description")
            or region_info.get("concise_description")
            or ""
        )
        return f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {desc}", cur_task["object_category"]
    if task_type == "instance":
        inst = goals_map[cur_task["instance_id"]]
        text = (
            inst.get("annot_unique_concise_description")
            if concise
            else inst.get("annot_unique_detailed_description")
        )
        text = text or inst.get("annot_unique_normal_description") or inst.get("annot_appearance_description") or ""
        return text, inst.get("object_category", "")
    raise ValueError(f"unknown task_type={task_type}")


def collect_samples(scene_files: List[Path], *, max_tasks: int, concise: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene_file in scene_files:
        with gzip.open(scene_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        scene_name = scene_file.name.split(".")[0]
        region_map = data["region_annotation"]
        episode_mapping = {
            "object": data["episodes_by_object_level"],
            "room": data["episodes_by_room_level"],
            "region": data["episodes_by_region_level"],
            "instance": data["episodes_by_instance_level"],
        }
        goals_map = {x["object_id"]: x for x in data["goals"]}
        for ep in data["episode_by_sequence"][:2]:
            for task_id, task_ref in enumerate(ep.get("task_sequence", [])):
                task_type, task_idx = task_ref
                cur_task = episode_mapping[task_type][task_idx]
                sentence, category = build_sentence(
                    task_type,
                    cur_task,
                    goals_map=goals_map,
                    region_map=region_map,
                    concise=concise,
                )
                rows.append(
                    {
                        "scene_name": scene_name,
                        "episode_id": int(ep["episode_id"]),
                        "task_id": int(task_id),
                        "task_type": task_type,
                        "target_category": category,
                        "sentence": sentence,
                    }
                )
                if len(rows) >= int(max_tasks):
                    return rows
    return rows


def target_leaks_to_anchor(target: str, anchors: List[str]) -> bool:
    target_words = {w for w in str(target or "").split() if len(w) > 2}
    if not target_words:
        return False
    for anchor in anchors:
        words = set(str(anchor or "").split())
        if words and words.issubset(target_words):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser("MQSC VLM decomposition multi-sample smoke test")
    parser.add_argument(
        "--scene_files",
        nargs="*",
        default=[
            "LangMap_Annotations/00800-TEEsavR23oF.json.gz",
            "LangMap_Annotations/00802-wcojb4TFT35.json.gz",
            "LangMap_Annotations/00803-k1cupFYWXJ6.json.gz",
        ],
    )
    parser.add_argument("--max_tasks", type=int, default=9)
    parser.add_argument("--repeat_per_task", type=int, default=2)
    parser.add_argument("--concise", action="store_true")
    parser.add_argument("--vlm_model", type=str, default=None)
    parser.add_argument("--disable_vlm", action="store_true")
    parser.add_argument("--disable_no_proxy", action="store_true")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="temp_scripts/mqsc_vlm_logs")
    args = parser.parse_args()

    scene_files = [Path(x).expanduser() for x in args.scene_files]
    scene_files = [p if p.is_absolute() else ROOT / p for p in scene_files]
    cfg = MqscConfig(
        use_vlm=not bool(args.disable_vlm),
        vlm_model=args.vlm_model or MqscConfig().vlm_model,
        vlm_max_retries=int(args.max_retries),
        vlm_no_proxy=not bool(args.disable_no_proxy),
        allow_heuristic_decompose=True,
    )
    samples = collect_samples(scene_files, max_tasks=int(args.max_tasks), concise=bool(args.concise))
    out: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "cfg": {
            "use_vlm": bool(cfg.use_vlm),
            "vlm_model": cfg.vlm_model,
            "vlm_max_retries": int(cfg.vlm_max_retries),
            "vlm_no_proxy": bool(cfg.vlm_no_proxy),
            "repeat_per_task": int(args.repeat_per_task),
        },
        "records": [],
    }
    parse_ok = 0
    total = 0
    leak_count = 0
    for sample in samples:
        for repeat_idx in range(int(args.repeat_per_task)):
            total += 1
            decomp = decompose_navigation_text(
                description=sample["sentence"],
                task_type=sample["task_type"],
                cfg=cfg,
            )
            queries = build_role_queries(sample["sentence"], decomp)
            anchors = list(decomp.get("anchor_primary", [])) + list(decomp.get("anchor_support", []))
            leak = target_leaks_to_anchor(str(decomp.get("target_desc", "")), anchors)
            parse_ok += 1 if bool(decomp.get("parse_ok", False)) else 0
            leak_count += 1 if leak else 0
            rec = {
                **sample,
                "repeat_idx": int(repeat_idx),
                "parse_ok": bool(decomp.get("parse_ok", False)),
                "source": decomp.get("source", ""),
                "error_type": decomp.get("error_type", ""),
                "target_desc": decomp.get("target_desc", ""),
                "target_aliases": decomp.get("target_aliases", []),
                "anchor_primary": decomp.get("anchor_primary", []),
                "anchor_support": decomp.get("anchor_support", []),
                "room_context": decomp.get("room_context", []),
                "target_leaks_to_anchor": bool(leak),
                "role_queries": queries,
                "raw": decomp.get("raw", ""),
            }
            out["records"].append(rec)
            print(
                "[mqsc-vlm] "
                f"scene={sample['scene_name']} ep={sample['episode_id']} task={sample['task_id']} "
                f"level={sample['task_type']} repeat={repeat_idx} parse_ok={rec['parse_ok']} "
                f"source={rec['source']} target={rec['target_desc']!r} "
                f"anchors={rec['anchor_primary'] + rec['anchor_support']} room={rec['room_context']}"
            )

    out["summary"] = {
        "total": int(total),
        "parse_ok": int(parse_ok),
        "parse_ok_rate": float(parse_ok / total) if total else 0.0,
        "target_leak_count": int(leak_count),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mqsc_vlm_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(
        "[mqsc-vlm-summary] "
        f"total={total} parse_ok={parse_ok} parse_ok_rate={out['summary']['parse_ok_rate']:.3f} "
        f"target_leak_count={leak_count} output={out_path}"
    )
    if total and parse_ok == 0 and bool(cfg.use_vlm):
        raise SystemExit("all VLM parses failed")
    if leak_count > 0:
        raise SystemExit("target leaked into anchors; inspect output")


if __name__ == "__main__":
    main()
