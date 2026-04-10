import argparse
import datetime
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple


LEVELS = ("object", "room", "region", "instance")


def load_result_records(path: Path) -> Dict[Tuple[str, str, int, int], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "sequence" not in data or not isinstance(data["sequence"], list):
        raise ValueError(f"Invalid result format in {path}: expect dict with key 'sequence' as list")

    records: Dict[Tuple[str, str, int, int], dict] = {}
    for item in data["sequence"]:
        key = (
            str(item["scene_name"]),
            str(item["navigation_type"]),
            int(item["episode_id"]),
            int(item["task_id"]),
        )
        if key in records:
            raise ValueError(f"Duplicate result key found in {path}: {key}")
        records[key] = item
    return records


def build_scene_file_index(langmap_root: Path) -> Dict[str, Path]:
    scene_files = sorted(langmap_root.rglob("*.json.gz"))
    if not scene_files:
        raise FileNotFoundError(f"No *.json.gz scene files found under: {langmap_root}")

    index: Dict[str, Path] = {}
    for scene_file in scene_files:
        scene_name = scene_file.name.replace(".json.gz", "")
        if scene_name in index:
            raise ValueError(f"Duplicate scene file for scene '{scene_name}': {index[scene_name]} and {scene_file}")
        index[scene_name] = scene_file
    return index


def load_scene_episode_level_map(scene_file: Path) -> Dict[int, List[List]]:
    with gzip.open(scene_file, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    episodes = scene_data["episode_by_sequence"]
    mapping: Dict[int, List[List]] = {}
    for ep in episodes:
        ep_id = int(ep["episode_id"])
        mapping[ep_id] = ep["task_sequence"]
    return mapping


def resolve_level(
    key: Tuple[str, str, int, int],
    scene_index: Dict[str, Path],
    scene_cache: Dict[str, Dict[int, List[List]]],
) -> str:
    scene_name, _, episode_id, task_id = key
    if scene_name not in scene_index:
        raise KeyError(f"Scene '{scene_name}' not found under LangMap annotations")
    if scene_name not in scene_cache:
        scene_cache[scene_name] = load_scene_episode_level_map(scene_index[scene_name])

    episode_map = scene_cache[scene_name]
    if episode_id not in episode_map:
        raise KeyError(f"Episode {episode_id} not found in scene '{scene_name}'")
    task_sequence = episode_map[episode_id]
    if task_id < 0 or task_id >= len(task_sequence):
        raise IndexError(f"Task index {task_id} out of range in scene '{scene_name}' episode {episode_id}")

    level = str(task_sequence[task_id][0])
    if level not in LEVELS:
        raise ValueError(f"Unexpected level '{level}' for key {key}")
    return level


def summarize(records: List[dict]) -> dict:
    count = len(records)
    if count == 0:
        return {"count": 0, "sr_mean": 0.0, "spl_mean": 0.0, "sr_sum": 0.0, "spl_sum": 0.0}
    sr_values = [float(r["sr"]) for r in records]
    spl_values = [float(r["spl"]) for r in records]
    return {
        "count": count,
        "sr_mean": sum(sr_values) / count,
        "spl_mean": sum(spl_values) / count,
        "sr_sum": sum(sr_values),
        "spl_sum": sum(spl_values),
    }


def summarize_paired(baseline_records: List[dict], anchor_records: List[dict]) -> dict:
    if len(baseline_records) != len(anchor_records):
        raise ValueError("Paired summary requires same record count")
    count = len(baseline_records)
    if count == 0:
        return {
            "count": 0,
            "delta_sr_mean": 0.0,
            "delta_spl_mean": 0.0,
            "anchor_better_sr_count": 0,
            "anchor_better_spl_count": 0,
        }

    delta_sr = [float(a["sr"]) - float(b["sr"]) for b, a in zip(baseline_records, anchor_records)]
    delta_spl = [float(a["spl"]) - float(b["spl"]) for b, a in zip(baseline_records, anchor_records)]
    return {
        "count": count,
        "delta_sr_mean": sum(delta_sr) / count,
        "delta_spl_mean": sum(delta_spl) / count,
        "anchor_better_sr_count": sum(1 for x in delta_sr if x > 0),
        "anchor_better_spl_count": sum(1 for x in delta_spl if x > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs anchor results by task level")
    parser.add_argument(
        "--baseline_result",
        type=str,
        default="output_logs/baseline/refhm3d_seq_0.0_0.4.json",
        help="Baseline result JSON path",
    )
    parser.add_argument(
        "--anchor_result",
        type=str,
        default="output_logs/anchor/gap/refhm3d_seq_anchor_gap_0.0_0.4.json",
        help="Anchor result JSON path",
    )
    parser.add_argument(
        "--langmap_root",
        type=str,
        default="LangMap_Annotations",
        help="LangMap annotation root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_logs/comparative_results",
        help="Output directory for comparison JSON",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Manual tag for output filename",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_result).expanduser().resolve()
    anchor_path = Path(args.anchor_result).expanduser().resolve()
    langmap_root = Path(args.langmap_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    baseline_records_map = load_result_records(baseline_path)
    anchor_records_map = load_result_records(anchor_path)

    baseline_keys = set(baseline_records_map.keys())
    anchor_keys = set(anchor_records_map.keys())
    common_keys = sorted(baseline_keys & anchor_keys)
    baseline_only_keys = sorted(baseline_keys - anchor_keys)
    anchor_only_keys = sorted(anchor_keys - baseline_keys)

    scene_index = build_scene_file_index(langmap_root)
    scene_cache: Dict[str, Dict[int, List[List]]] = {}

    baseline_by_level = {lvl: [] for lvl in LEVELS}
    anchor_by_level = {lvl: [] for lvl in LEVELS}
    paired_baseline_by_level = {lvl: [] for lvl in LEVELS}
    paired_anchor_by_level = {lvl: [] for lvl in LEVELS}

    for key, item in baseline_records_map.items():
        level = resolve_level(key, scene_index, scene_cache)
        baseline_by_level[level].append(item)

    for key, item in anchor_records_map.items():
        level = resolve_level(key, scene_index, scene_cache)
        anchor_by_level[level].append(item)

    for key in common_keys:
        level = resolve_level(key, scene_index, scene_cache)
        paired_baseline_by_level[level].append(baseline_records_map[key])
        paired_anchor_by_level[level].append(anchor_records_map[key])

    baseline_all = list(baseline_records_map.values())
    anchor_all = list(anchor_records_map.values())
    paired_baseline_all = [baseline_records_map[k] for k in common_keys]
    paired_anchor_all = [anchor_records_map[k] for k in common_keys]

    result = {
        "meta": {
            "tag": args.tag,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "baseline_result": str(baseline_path),
            "anchor_result": str(anchor_path),
            "langmap_root": str(langmap_root),
            "level_names": list(LEVELS),
        },
        "record_alignment": {
            "baseline_count": len(baseline_keys),
            "anchor_count": len(anchor_keys),
            "common_count": len(common_keys),
            "baseline_only_count": len(baseline_only_keys),
            "anchor_only_count": len(anchor_only_keys),
        },
        "metrics": {
            "baseline": {
                "overall": summarize(baseline_all),
                "by_level": {lvl: summarize(baseline_by_level[lvl]) for lvl in LEVELS},
            },
            "anchor": {
                "overall": summarize(anchor_all),
                "by_level": {lvl: summarize(anchor_by_level[lvl]) for lvl in LEVELS},
            },
            "paired_comparison_on_common_records": {
                "overall": summarize_paired(paired_baseline_all, paired_anchor_all),
                "by_level": {
                    lvl: summarize_paired(paired_baseline_by_level[lvl], paired_anchor_by_level[lvl])
                    for lvl in LEVELS
                },
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"{args.tag}_{ts}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] comparison saved to: {output_path}")
    print(
        "[Summary] "
        f"baseline={len(baseline_keys)}, anchor={len(anchor_keys)}, common={len(common_keys)}, "
        f"baseline_only={len(baseline_only_keys)}, anchor_only={len(anchor_only_keys)}"
    )


if __name__ == "__main__":
    main()
