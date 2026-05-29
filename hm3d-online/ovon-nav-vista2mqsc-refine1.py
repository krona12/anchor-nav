"""OVON Vista2MQSC refine1 batch evaluation.

This is the OVON counterpart of
``refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py``.  It keeps the
OVON baseline episode flow from ``ovon-nav.py`` and applies the existing
MQSC-R1 -> VISTA-LS hook only when PQ3D makes a final object decision.
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import gzip
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.embodied_utils.simulator import HabitatSimulator
from data_utils import PQ3DModel
from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, map_coors_to_pixel, pixel_to_map_coors


def _load_vista2mqsc_helpers() -> Any:
    helper_path = SCRIPT_DIR / "refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py"
    spec = importlib.util.spec_from_file_location("_vista2mqsc_refine1_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Vista2MQSC helper script: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_vista2mqsc_refine1_helpers", module)
    spec.loader.exec_module(module)
    return module


VISTA2MQSC = _load_vista2mqsc_helpers()


class _TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _tqdm_print(msg: str) -> None:
    tqdm.write(msg, file=sys.stdout)


def _setup_run_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"ovon-nav-vista2mqsc-refine1-{ts}-pid{os.getpid()}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[OVONVista2MQSC] logging enabled -> {log_path.resolve()}")

    def _cleanup() -> None:
        try:
            print(f"[OVONVista2MQSC] run finished, log saved -> {log_path.resolve()}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        value = float(x)
        return value if math.isfinite(value) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    _tqdm_print(f"[OVONVista2MQSC] seed={int(seed)}")


def _parse_float_list(text: str, *, arg_name: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if len(values) == 0 or any(x <= 0.0 for x in values):
        raise RuntimeError(f"{arg_name} must contain positive comma-separated floats, got {text!r}")
    return values


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _split_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _load_navigation_data(navigation_data_path: Path, hm3d_data_base_path: Path, split_list: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    navigation_data_dict: Dict[str, Dict[str, Any]] = {split: {} for split in split_list}
    raw_scan_ids = {
        d.name
        for d in hm3d_data_base_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    }
    for split in split_list:
        data_dir = navigation_data_path / split / "content"
        if not data_dir.exists():
            raise FileNotFoundError(f"OVON navigation content directory not found: {data_dir}")
        file_list = sorted(f for f in os.listdir(data_dir) if f and not f.startswith(".") and f.endswith(".json.gz"))
        for file_name in file_list:
            simplified_scan_id = file_name.split(".")[0]
            matches = sorted(pa for pa in raw_scan_ids if simplified_scan_id in pa)
            if not matches:
                raise FileNotFoundError(
                    f"Cannot map OVON scene file {file_name} to HM3D scene under {hm3d_data_base_path}"
                )
            raw_scan_id = matches[0]
            with gzip.open(data_dir / file_name, "rt", encoding="utf-8") as f:
                data = json.load(f)
            navigation_data_dict[split][raw_scan_id] = {
                "episodes": data["episodes"],
                "goals_by_category": {
                    k.split("glb_")[-1]: v
                    for k, v in data["goals_by_category"].items()
                },
            }
    return navigation_data_dict


def _init_result_dict(path: Path, split_list: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for split in split_list:
            data.setdefault(split, [])
        return data
    return {split: [] for split in split_list}


def _existing_episode_keys(result_dict: Mapping[str, Any], split_list: Iterable[str]) -> set:
    keys = set()
    for split in split_list:
        for row in result_dict.get(split, []):
            if not isinstance(row, dict):
                continue
            keys.add((split, row.get("scan_id"), row.get("episode_index")))
    return keys


def _metric_rows(result_dict: Mapping[str, Any], split_list: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in split_list:
        for row in result_dict.get(split, []):
            if isinstance(row, dict):
                rows.append({**row, "split": split})
    return rows


def _metric_snapshot(result_dict: Mapping[str, Any], split_list: Iterable[str]) -> Dict[str, Any]:
    rows = _metric_rows(result_dict, split_list)
    by_split = {}
    for split in split_list:
        bucket = [r for r in rows if r.get("split") == split]
        by_split[split] = {
            "count": int(len(bucket)),
            "sr": float(np.mean([float(r.get("sr", 0.0)) for r in bucket])) if bucket else 0.0,
            "spl": float(np.mean([float(r.get("spl", 0.0)) for r in bucket])) if bucket else 0.0,
        }
    return {
        "count": int(len(rows)),
        "sr": float(np.mean([float(r.get("sr", 0.0)) for r in rows])) if rows else 0.0,
        "spl": float(np.mean([float(r.get("spl", 0.0)) for r in rows])) if rows else 0.0,
        "by_split": by_split,
    }


def _print_metrics(result_dict: Mapping[str, Any], split_list: Iterable[str]) -> None:
    snap = _metric_snapshot(result_dict, split_list)
    _tqdm_print(
        f"[OVON_METRICS] count={snap['count']} sr={snap['sr']:.6f} spl={snap['spl']:.6f} "
        f"by_split={json.dumps(snap['by_split'], ensure_ascii=False, sort_keys=True)}"
    )
    for split in split_list:
        category_sr_spl = defaultdict(lambda: {"sr": 0.0, "spl": 0.0, "count": 0})
        for row in result_dict.get(split, []):
            category = str(row.get("object_category", ""))
            category_sr_spl[category]["sr"] += float(row.get("sr", 0.0))
            category_sr_spl[category]["spl"] += float(row.get("spl", 0.0))
            category_sr_spl[category]["count"] += 1
        for category, metrics in sorted(category_sr_spl.items()):
            count = max(1, int(metrics["count"]))
            _tqdm_print(
                f"[OVON_METRICS] split={split} category={category} "
                f"count={metrics['count']} sr={metrics['sr'] / count:.6f} spl={metrics['spl'] / count:.6f}"
            )


def _append_live_metrics(metrics_log_path: Path, result_dict: Mapping[str, Any], split_list: Iterable[str], latest: Dict[str, Any]) -> None:
    snap = _metric_snapshot(result_dict, split_list)
    line = (
        f"[OVON_VISTA2MQSC_LIVE_METRICS] count={snap['count']} sr={snap['sr']:.6f} spl={snap['spl']:.6f} "
        f"by_split={json.dumps(snap['by_split'], ensure_ascii=False, sort_keys=True)}"
    )
    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "latest": latest,
        "metrics": snap,
    }
    _tqdm_print(line)
    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _goal_positions(goals: Iterable[Mapping[str, Any]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for goal in goals:
        pos = goal.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            out.append(np.asarray(pos[:3], dtype=float).reshape(3))
    return out


def _view_points(goals: Iterable[Mapping[str, Any]]) -> List[Any]:
    return [
        view_point["agent_state"]["position"]
        for goal in goals
        for view_point in goal.get("view_points", [])
        if isinstance(view_point, dict) and "agent_state" in view_point
    ]


def _compute_ovon_metrics(
    *,
    path_finder: Any,
    start_position: Any,
    agent_position: Any,
    view_points: List[Any],
    episode_cum_distance: float,
    success_distance: float,
    forced_failure: bool,
) -> Tuple[float, float, float, float]:
    if forced_failure or len(view_points) == 0:
        return 0.0, 0.0, float("inf"), float("inf")
    sp = habitat_sim.MultiGoalShortestPath()
    sp.requested_start = start_position
    sp.requested_ends = view_points
    if path_finder.find_path(sp):
        start_end_geo_distance = float(sp.geodesic_distance)
    else:
        start_end_geo_distance = float("inf")

    ep = habitat_sim.MultiGoalShortestPath()
    ep.requested_start = agent_position
    ep.requested_ends = view_points
    if path_finder.find_path(ep):
        agent_end_geo_distance = float(ep.geodesic_distance)
    else:
        agent_end_geo_distance = float("inf")

    if start_end_geo_distance == float("inf"):
        sr, spl = 1.0, 1.0
    elif agent_end_geo_distance == float("inf"):
        sr, spl = 0.0, 0.0
    else:
        sr = 1.0 if agent_end_geo_distance <= float(success_distance) else 0.0
        denom = max(float(start_end_geo_distance), float(episode_cum_distance), 1e-12)
        spl = float(sr * start_end_geo_distance / denom)
    return float(sr), float(spl), float(start_end_geo_distance), float(agent_end_geo_distance)


def configure_vista2mqsc(args: argparse.Namespace) -> None:
    VISTA2MQSC.MQSC_R1_CFG = VISTA2MQSC.MqscR1Config(
        top_k=int(args.mqsc_r1_top_k),
        temperature=float(args.mqsc_r1_temperature),
        cluster_eps=float(args.mqsc_r1_cluster_eps),
        min_region_coverage=float(args.mqsc_r1_min_region_coverage),
        min_target_prob=float(args.mqsc_r1_min_target_prob),
        min_region_margin=float(args.mqsc_r1_min_region_margin),
        min_selected_gain=float(args.mqsc_r1_min_selected_gain),
        use_vlm=not bool(args.mqsc_r1_disable_vlm),
        vlm_model=str(args.mqsc_r1_vlm_model),
        vlm_max_retries=int(args.mqsc_r1_vlm_max_retries),
        vlm_retry_sleep_sec=float(args.mqsc_r1_vlm_retry_sleep_sec),
        vlm_no_proxy=not bool(args.mqsc_r1_disable_no_proxy),
        allow_heuristic_decompose=not bool(args.mqsc_r1_disable_heuristic_decompose),
        write_debug_json=not bool(args.mqsc_r1_disable_debug_json),
    )
    VISTA2MQSC.VISTALS_CFG = VISTA2MQSC.VistaLsConfig(
        candidate_radii_m=_parse_float_list(args.vistals_candidate_radii_m, arg_name="--vistals_candidate_radii_m"),
        candidate_view_count=int(args.vistals_candidate_view_count),
        enable_vvd_replacement=bool(args.vistals_enable_vvd_replacement),
        prefer_visible_baseline=not bool(args.vistals_disable_visible_baseline_guard),
        camera_height_m=float(args.vistals_camera_height_m),
        max_snap_distance_m=float(args.vistals_max_snap_distance_m),
        target_sample_count=int(args.vistals_target_sample_count),
        scene_sample_count=int(args.vistals_scene_sample_count),
        max_ray_sample_count=int(args.vistals_max_ray_sample_count),
        occlusion_radius_m=float(args.vistals_occlusion_radius_m),
        min_visibility_score=float(args.vistals_min_visibility_score),
        visibility_tie_epsilon=float(args.vistals_visibility_tie_epsilon),
        path_efficiency_exponent=float(args.vistals_path_efficiency_exponent),
        r_min_m=float(args.vistals_r_min_m),
        r_max_m=float(args.vistals_r_max_m),
        radial_step_m=float(args.vistals_radial_step_m),
        angle_step_deg=float(args.vistals_angle_step_deg),
        shell_min_m=float(args.vistals_shell_min_m),
        shell_max_m=float(args.vistals_shell_max_m),
        relaxed_shell_min_m=float(args.vistals_relaxed_shell_min_m),
        relaxed_shell_max_m=float(args.vistals_relaxed_shell_max_m),
        min_clearance_m=float(args.vistals_min_clearance_m),
        min_component_size=int(args.vistals_min_component_size),
        size_tie_ratio=float(args.vistals_size_tie_ratio),
    )
    VISTA2MQSC.VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS = {"object"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run OVON Vista2MQSC refine1 batch evaluation")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=0.2)
    parser.add_argument("--splits", type=str, default="val_seen,val_seen_synonyms,val_unseen")
    parser.add_argument(
        "--data_set_path",
        type=str,
        default="/disks/amax_robot_dataset/embodied/embodied_bench_data/our-set/ovon_full_set.json",
    )
    parser.add_argument(
        "--navigation_data_path",
        type=str,
        default="/disks/amax_robot_dataset/embodied/embodied_bench_data/ovon/",
    )
    parser.add_argument("--hm3d_data_base_path", type=str, default="/home/chenlin/krona/MTU3D/datascene")
    parser.add_argument("--pq3d_stage1_path", type=str, default="/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-ovon")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--output_log_dir", type=str, default="output_logs/anchor/ovon_vista2mqsc")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--visible_radius", type=float, default=3.0)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet_nav_steps", action="store_true")
    parser.add_argument("--decision_log_interval", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mqsc_r1_top_k", type=int, default=8)
    parser.add_argument("--mqsc_r1_temperature", type=float, default=1.0)
    parser.add_argument("--mqsc_r1_cluster_eps", type=float, default=1.2)
    parser.add_argument("--mqsc_r1_min_region_coverage", type=float, default=0.5)
    parser.add_argument("--mqsc_r1_min_target_prob", type=float, default=0.05)
    parser.add_argument("--mqsc_r1_min_region_margin", type=float, default=0.05)
    parser.add_argument("--mqsc_r1_min_selected_gain", type=float, default=-0.02)
    parser.add_argument("--mqsc_r1_vlm_model", type=str, default=os.environ.get("VLM_MODEL", VISTA2MQSC.MqscR1Config().vlm_model))
    parser.add_argument("--mqsc_r1_vlm_max_retries", type=int, default=3)
    parser.add_argument("--mqsc_r1_vlm_retry_sleep_sec", type=float, default=1.0)
    parser.add_argument("--mqsc_r1_disable_vlm", action="store_true")
    parser.add_argument("--mqsc_r1_disable_no_proxy", action="store_true")
    parser.add_argument("--mqsc_r1_disable_heuristic_decompose", action="store_true")
    parser.add_argument("--mqsc_r1_disable_debug_json", action="store_true")
    parser.add_argument("--vistals_candidate_radii_m", type=str, default="0.5,0.75")
    parser.add_argument("--vistals_candidate_view_count", type=int, default=20)
    parser.add_argument("--vistals_enable_vvd_replacement", action="store_true")
    parser.add_argument("--vistals_disable_visible_baseline_guard", action="store_true")
    parser.add_argument("--vistals_camera_height_m", type=float, default=1.50)
    parser.add_argument("--vistals_max_snap_distance_m", type=float, default=0.60)
    parser.add_argument("--vistals_target_sample_count", type=int, default=300)
    parser.add_argument("--vistals_scene_sample_count", type=int, default=0)
    parser.add_argument("--vistals_max_ray_sample_count", type=int, default=300)
    parser.add_argument("--vistals_occlusion_radius_m", type=float, default=0.05)
    parser.add_argument("--vistals_min_visibility_score", type=float, default=0.02)
    parser.add_argument("--vistals_visibility_tie_epsilon", type=float, default=0.0)
    parser.add_argument("--vistals_path_efficiency_exponent", type=float, default=0.15)
    parser.add_argument("--vistals_r_min_m", type=float, default=0.30)
    parser.add_argument("--vistals_r_max_m", type=float, default=1.30)
    parser.add_argument("--vistals_radial_step_m", type=float, default=0.10)
    parser.add_argument("--vistals_angle_step_deg", type=float, default=10.0)
    parser.add_argument("--vistals_shell_min_m", type=float, default=0.35)
    parser.add_argument("--vistals_shell_max_m", type=float, default=1.20)
    parser.add_argument("--vistals_relaxed_shell_min_m", type=float, default=0.30)
    parser.add_argument("--vistals_relaxed_shell_max_m", type=float, default=1.35)
    parser.add_argument("--vistals_min_clearance_m", type=float, default=0.10)
    parser.add_argument("--vistals_min_component_size", type=int, default=3)
    parser.add_argument("--vistals_size_tie_ratio", type=float, default=0.85)
    return parser


def run_episode(
    *,
    args: argparse.Namespace,
    split: str,
    cur_data: Mapping[str, Any],
    navigation_data_dict: Mapping[str, Mapping[str, Any]],
    pq3d_model: Any,
    output_log_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scene_id = str(cur_data["scan_id"])
    episode_index = int(cur_data["episode_index"])
    object_category = str(cur_data["object_category"])
    scene_nav = navigation_data_dict[split][scene_id]
    cur_episode = scene_nav["episodes"][episode_index]
    episode_id = _safe_int(cur_episode.get("episode_id", episode_index), episode_index)
    if str(cur_episode.get("object_category")) != object_category:
        raise RuntimeError(
            f"OVON object category mismatch split={split} scene={scene_id} episode_index={episode_index}: "
            f"dataset={object_category!r} nav={cur_episode.get('object_category')!r}"
        )
    goals = scene_nav["goals_by_category"].get(object_category)
    if not goals:
        raise RuntimeError(
            f"OVON goals missing split={split} scene={scene_id} episode_index={episode_index} category={object_category!r}"
        )

    pq3d_model.reset()
    start_position = cur_episode["start_position"]
    start_rotation = cur_episode["start_rotation"]
    episode_dir = output_log_dir / "process" / f"split={split}" / f"scene={scene_id}" / f"episode_index={episode_index}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    sim_settings = OmegaConf.load(str(Path(args.sim_config).expanduser()))
    agent_setting = OmegaConf.load(str(Path(args.agent_config).expanduser()))
    sim_settings["scene"] = VISTA2MQSC.resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), scene_id)
    abstract_sim = HabitatSimulator(sim_settings, agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    try:
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels = convert_meters_to_pixel(9, 512, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(float(args.visible_radius), 512, sim)

        total_steps = 0
        decision_num = 0
        episode_cum_distance = 0.0
        prev_agent_state = agent.get_state()
        visited_frontier_set = set()
        goto_color_list: List[np.ndarray] = []
        goto_depth_list: List[np.ndarray] = []
        goto_agent_state_list: List[Any] = []
        baseline_final_target_pos: Optional[np.ndarray] = None
        final_selected_target_pos: Optional[np.ndarray] = None
        hook_called = 0
        hook_applied = 0
        hook_logs: List[Dict[str, Any]] = []
        task_pq3d_object_gap: Optional[float] = None
        end_reason = "max_steps"
        forced_failure = False
        t0 = time.perf_counter()

        while total_steps < int(args.max_steps):
            color_list: List[np.ndarray] = []
            depth_list: List[np.ndarray] = []
            agent_state_list: List[Any] = []
            if len(goto_color_list) > 6:
                step = max(1, len(goto_color_list) // 6)
                goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
            color_list.extend(goto_color_list)
            depth_list.extend(goto_depth_list)
            agent_state_list.extend(goto_agent_state_list)

            t_scan = time.perf_counter()
            scan_rgb, scan_depth, scan_states, fog_of_war_mask, total_steps = VISTA2MQSC._capture_scan_frames(
                sim=sim,
                agent=agent,
                top_down_map=top_down_map,
                fog_of_war_mask=fog_of_war_mask,
                visibility_dist_in_pixels=visibility_dist_in_pixels,
                total_steps=total_steps,
                max_steps=int(args.max_steps),
            )
            scan_ms = (time.perf_counter() - t_scan) * 1000.0
            color_list.extend(scan_rgb)
            depth_list.extend(scan_depth)
            agent_state_list.extend(scan_states)
            if total_steps >= int(args.max_steps):
                break

            t_frontier = time.perf_counter()
            agent_state = agent.get_state()
            frontier_waypoints = detect_frontier_waypoints(
                top_down_map,
                fog_of_war_mask,
                area_thres_in_pixels,
                xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                enable_visualization=False,
            )
            if len(frontier_waypoints) == 0:
                frontier_waypoints = []
            else:
                frontier_waypoints = pixel_to_map_coors(frontier_waypoints[:, ::-1], agent_state.position, top_down_map, sim)
            frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]
            frontier_ms = (time.perf_counter() - t_frontier) * 1000.0

            t_pq = time.perf_counter()
            target_position, is_final = pq3d_model.decision(
                color_list,
                depth_list,
                agent_state_list,
                frontier_waypoints,
                object_category,
                decision_num,
            )
            pq_ms = (time.perf_counter() - t_pq) * 1000.0
            if not bool(args.quiet_nav_steps):
                _tqdm_print(
                    f"[ovon-vista2mqsc][step] split={split} scene={scene_id} episode_index={episode_index} "
                    f"dec={decision_num} scan_ms={scan_ms:.0f} frontier_ms={frontier_ms:.0f} pq3d_ms={pq_ms:.0f} "
                    f"frames={len(color_list)} frontiers={len(frontier_waypoints)} final={bool(is_final)}"
                )
            if int(args.decision_log_interval) > 0 and decision_num % int(args.decision_log_interval) == 0:
                _tqdm_print(
                    f"[ovon-vista2mqsc][decision] split={split} scene={scene_id} episode_index={episode_index} "
                    f"dec={decision_num} target={np.asarray(target_position, dtype=float).reshape(-1)[:3].tolist()} "
                    f"final={bool(is_final)}"
                )

            used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
            aux = dict(getattr(pq3d_model, "last_decision_aux", {}) or {})
            module_info: Dict[str, Any] = {
                "ok": True,
                "module": "vista2mqsc",
                "called": False,
                "applied": False,
                "reason": "non_final_decision",
                "target_before": used_target.tolist(),
                "target_after": used_target.tolist(),
            }
            if bool(is_final):
                baseline_final_target_pos = used_target.copy()
                task_pq3d_object_gap = float(aux.get("object_top1_top2_logit_gap", 0.0))
                hook_called += 1
                used_target, module_info = VISTA2MQSC.vista2mqsc_refine_hook(
                    sentence=object_category,
                    task_type="object",
                    scene_name=scene_id,
                    episode_id=int(episode_id),
                    task_id=int(episode_index),
                    decision_num=int(decision_num),
                    is_final=True,
                    pq3d_model=pq3d_model,
                    target_position=used_target,
                    decision_aux=aux,
                    output_dir=episode_dir,
                    path_finder=path_finder,
                    agent_position_xyz=np.asarray(agent.get_state().position, dtype=float).reshape(3),
                )
                _write_json(episode_dir / "vista2mqsc" / f"dec_{decision_num:03d}_vista2mqsc.json", module_info)
                if bool(module_info.get("applied", False)):
                    hook_applied += 1
                final_selected_target_pos = np.asarray(used_target, dtype=float).reshape(3).copy()
                hook_logs.append(
                    {
                        "split": split,
                        "scan_id": scene_id,
                        "episode_index": int(episode_index),
                        "episode_id": int(episode_id),
                        "decision_num": int(decision_num),
                        "sentence": object_category,
                        "module_info": module_info,
                    }
                )
                _tqdm_print(
                    f"[ovon-vista2mqsc][final] split={split} scene={scene_id} episode_index={episode_index} "
                    f"dec={decision_num} hook_called={hook_called} hook_applied={hook_applied} "
                    f"reason={module_info.get('reason')}"
                )
            else:
                visited_frontier_set.add(tuple(np.round(used_target, 1)))

            (
                goto_color_list,
                goto_depth_list,
                goto_agent_state_list,
                prev_agent_state,
                total_steps,
                episode_cum_distance,
                follow_log,
            ) = VISTA2MQSC._follow_target(
                path_finder=path_finder,
                agent=agent,
                sim=sim,
                target=used_target,
                prev_agent_state=prev_agent_state,
                total_steps=total_steps,
                max_steps=int(args.max_steps),
                episode_cum_distance=float(episode_cum_distance),
            )
            _write_json(
                episode_dir / f"dec_{decision_num:03d}_vista2mqsc.json",
                {
                    "split": split,
                    "scan_id": scene_id,
                    "episode_index": int(episode_index),
                    "episode_id": int(episode_id),
                    "object_category": object_category,
                    "decision_num": int(decision_num),
                    "is_final": bool(is_final),
                    "target_used": np.asarray(used_target, dtype=float).reshape(3).tolist(),
                    "pq3d_aux": aux,
                    "module": module_info,
                    "module_info": module_info,
                    "follow": follow_log,
                    "steps_total_after_follow": int(total_steps),
                },
            )
            decision_num += 1
            if not bool(follow_log.get("ok", False)):
                if not bool(is_final):
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))
                    _tqdm_print(
                        f"[ovon-vista2mqsc][frontier-follow-retry] split={split} scene={scene_id} "
                        f"episode_index={episode_index} dec={decision_num - 1} "
                        f"error={follow_log.get('error_type')} candidates={follow_log.get('candidate_attempt_count')}"
                    )
                    continue
                end_reason = "follower_error"
                forced_failure = True
                break
            if bool(is_final):
                end_reason = "final_decision"
                break

        task_time = float(time.perf_counter() - t0)
        agent_state = agent.get_state()
        goal_positions = _goal_positions(goals)
        view_points = _view_points(goals)
        sr, spl, start_end_geo_distance, agent_end_geo_distance = _compute_ovon_metrics(
            path_finder=path_finder,
            start_position=start_position,
            agent_position=agent_state.position,
            view_points=view_points,
            episode_cum_distance=float(episode_cum_distance),
            success_distance=float(args.success_distance),
            forced_failure=bool(forced_failure),
        )

        baseline_target_to_goal_l2 = float("inf")
        selected_target_to_goal_l2 = float("inf")
        module_helpful = None
        if goal_positions:
            if baseline_final_target_pos is not None:
                baseline_target_to_goal_l2 = float(
                    min(float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions)
                )
            if final_selected_target_pos is not None:
                selected_target_to_goal_l2 = float(
                    min(float(np.linalg.norm(final_selected_target_pos - gp)) for gp in goal_positions)
                )
            if hook_applied > 0 and np.isfinite(baseline_target_to_goal_l2) and np.isfinite(selected_target_to_goal_l2):
                module_helpful = bool(selected_target_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6)

        final_module_info = hook_logs[-1].get("module_info", {}) if hook_logs else {}
        row = {
            "split": split,
            "scan_id": scene_id,
            "episode_index": int(episode_index),
            "episode_id": int(episode_id),
            "sr": float(sr),
            "spl": float(spl),
            "object_category": object_category,
            "sentence": object_category,
            "task_time_sec": float(task_time),
            "steps_total": int(total_steps),
            "decision_count": int(decision_num),
            "end_reason": end_reason,
            "start_goal_geo": float(start_end_geo_distance),
            "end_goal_geo": float(agent_end_geo_distance),
            "episode_cum_distance": float(episode_cum_distance),
            "module_name": "vista2mqsc",
            "module_hook_called": int(hook_called),
            "module_hook_applied": int(hook_applied),
            "module_helpful": module_helpful,
            "module_reason": final_module_info.get("reason"),
            "mqsc_r1_called": bool(final_module_info.get("mqsc_r1_called", False)),
            "mqsc_r1_applied": bool(final_module_info.get("mqsc_r1_applied", False)),
            "vistals_called": bool(final_module_info.get("vistals_called", False)),
            "vistals_applied": bool(final_module_info.get("vistals_applied", False)),
            "vistals_input_slot_index": final_module_info.get("vistals_input_slot_index"),
            "vistals_input_slot_source": final_module_info.get("vistals_input_slot_source"),
            "goal_positions": [gp.tolist() for gp in goal_positions],
            "baseline_target_position": None if baseline_final_target_pos is None else baseline_final_target_pos.tolist(),
            "selected_target_position": None if final_selected_target_pos is None else final_selected_target_pos.tolist(),
            "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
            "selected_target_to_goal_l2": float(selected_target_to_goal_l2),
            "baseline_target_to_goal_l2_valid": bool(np.isfinite(baseline_target_to_goal_l2)),
            "selected_target_to_goal_l2_valid": bool(np.isfinite(selected_target_to_goal_l2)),
            "pq3d_object_top1_top2_logit_gap": task_pq3d_object_gap,
            "agent_start_position": np.asarray(start_position, dtype=float).reshape(3).tolist(),
            "agent_end_position": np.asarray(agent_state.position, dtype=float).reshape(3).tolist(),
        }
        effectiveness_record = {
            "split": split,
            "scan_id": scene_id,
            "episode_index": int(episode_index),
            "episode_id": int(episode_id),
            "object_category": object_category,
            "module_name": "vista2mqsc",
            "module_hook_called": int(hook_called),
            "module_hook_applied": int(hook_applied),
            "module_helpful": module_helpful,
            "module_reason": final_module_info.get("reason"),
            "mqsc_r1_called": bool(final_module_info.get("mqsc_r1_called", False)),
            "mqsc_r1_applied": bool(final_module_info.get("mqsc_r1_applied", False)),
            "vistals_called": bool(final_module_info.get("vistals_called", False)),
            "vistals_applied": bool(final_module_info.get("vistals_applied", False)),
            "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
            "selected_target_to_goal_l2": float(selected_target_to_goal_l2),
            "baseline_target_to_goal_l2_valid": bool(np.isfinite(baseline_target_to_goal_l2)),
            "selected_target_to_goal_l2_valid": bool(np.isfinite(selected_target_to_goal_l2)),
            "hook_logs": hook_logs,
        }
        _write_json(episode_dir / "summary.json", row)
        _write_json(episode_dir / "effectiveness.json", effectiveness_record)
        _tqdm_print(
            f"[ovon-vista2mqsc] split={split} scene={scene_id} episode_index={episode_index} "
            f"SR={sr:.1f} SPL={spl:.4f} time={task_time:.3f}s steps={total_steps} decisions={decision_num} "
            f"hook_called={hook_called} hook_applied={hook_applied} helpful={module_helpful} "
            f"reason={final_module_info.get('reason')}"
        )
        return row, effectiveness_record
    finally:
        sim.close()


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 <= float(args.start_ratio) <= float(args.end_ratio) <= 1.0):
        raise RuntimeError(f"invalid ratio range: start={args.start_ratio} end={args.end_ratio}")

    split_list = _split_arg(args.splits)
    if not split_list:
        raise RuntimeError("--splits resolved to empty list")
    output_log_dir = Path(os.path.expanduser(args.output_log_dir))
    output_log_dir.mkdir(parents=True, exist_ok=True)
    _setup_run_logging(output_log_dir)
    _set_seed(int(args.seed))
    configure_vista2mqsc(args)
    _write_json(output_log_dir / "run_args.json", vars(args))

    _tqdm_print(
        f"[OVONVista2MQSC] cfg start_ratio={args.start_ratio} end_ratio={args.end_ratio} splits={split_list} "
        f"max_steps={args.max_steps} decision_num_min={args.decision_num_min} output_log_dir={output_log_dir} "
        f"mqsc_r1_top_k={VISTA2MQSC.MQSC_R1_CFG.top_k} mqsc_r1_eps={VISTA2MQSC.MQSC_R1_CFG.cluster_eps} "
        f"mqsc_r1_vlm={VISTA2MQSC.MQSC_R1_CFG.use_vlm} mqsc_r1_vlm_model={VISTA2MQSC.MQSC_R1_CFG.vlm_model} "
        f"vistals_enable_vvd_replacement={VISTA2MQSC.VISTALS_CFG.enable_vvd_replacement} "
        f"vistals_radial_step={VISTA2MQSC.VISTALS_CFG.radial_step_m} vistals_angle_step={VISTA2MQSC.VISTALS_CFG.angle_step_deg}"
    )

    navigation_data_path = Path(os.path.expanduser(args.navigation_data_path))
    hm3d_data_base_path = Path(os.path.expanduser(args.hm3d_data_base_path))
    data_set_path = Path(os.path.expanduser(args.data_set_path))
    navigation_data_dict = _load_navigation_data(navigation_data_path, hm3d_data_base_path, split_list)
    with open(data_set_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    selected_data: Dict[str, List[Dict[str, Any]]] = {}
    for split in split_list:
        episodes = list(data_set.get(split, []))
        start = int(float(args.start_ratio) * len(episodes))
        end = int(float(args.end_ratio) * len(episodes))
        selected_data[split] = episodes[start:end]
        _tqdm_print(
            f"[OVONVista2MQSC] selected split={split} episodes={len(selected_data[split])} "
            f"range=[{start},{end}) total={len(episodes)}"
        )

    if bool(args.dry_run):
        sample = next((eps[0] for eps in selected_data.values() if eps), None)
        if sample is not None:
            scene_id = str(sample["scan_id"])
            scene_path = VISTA2MQSC.resolve_scene_path(str(hm3d_data_base_path), scene_id)
            _tqdm_print(f"[OVONVista2MQSC][dry-run] sample_scene={scene_id} scene_path={scene_path}")
        _tqdm_print("[OVONVista2MQSC][dry-run] data/config validation complete")
        return

    out_name = f"ovon_vista2mqsc_refine1_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"ovon_vista2mqsc_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
    output_path = output_log_dir / out_name
    effectiveness_path = output_log_dir / eff_name
    metrics_log_path = output_log_dir / f"ovon_vista2mqsc_live_metrics_{args.start_ratio}_{args.end_ratio}.log"
    _tqdm_print(f"[OVONVista2MQSC] output_json={output_path}")
    _tqdm_print(f"[OVONVista2MQSC] effectiveness_json={effectiveness_path}")
    _tqdm_print(f"[OVONVista2MQSC] live_metrics_log={metrics_log_path}")

    result_dict = _init_result_dict(output_path, split_list)
    existing_episodes = _existing_episode_keys(result_dict, split_list)
    if effectiveness_path.exists():
        with open(effectiveness_path, "r", encoding="utf-8") as f:
            effectiveness_dict = json.load(f)
        effectiveness_dict.setdefault("records", [])
    else:
        effectiveness_dict = {"records": []}
    if existing_episodes:
        _print_metrics(result_dict, split_list)
        _append_live_metrics(metrics_log_path, result_dict, split_list, latest={"event": "resume_existing_output"})

    pq3d_model = PQ3DModel(
        os.path.expanduser(args.pq3d_stage1_path),
        os.path.expanduser(args.pq3d_stage2_path),
        min_decision_num=int(args.decision_num_min),
    )

    for split in split_list:
        for cur_data in tqdm(selected_data[split], desc=f"*** OVON {split} ***"):
            key = (split, cur_data.get("scan_id"), cur_data.get("episode_index"))
            if key in existing_episodes:
                continue
            row, effectiveness_record = run_episode(
                args=args,
                split=split,
                cur_data=cur_data,
                navigation_data_dict=navigation_data_dict,
                pq3d_model=pq3d_model,
                output_log_dir=output_log_dir,
            )
            result_dict.setdefault(split, []).append(row)
            effectiveness_dict.setdefault("records", []).append(effectiveness_record)
            existing_episodes.add(key)
            _write_json(output_path, result_dict)
            _write_json(effectiveness_path, effectiveness_dict)
            _append_live_metrics(
                metrics_log_path,
                result_dict,
                split_list,
                latest={
                    "split": split,
                    "scan_id": row["scan_id"],
                    "episode_index": int(row["episode_index"]),
                    "sr": float(row["sr"]),
                    "spl": float(row["spl"]),
                },
            )
            _print_metrics(result_dict, split_list)

    _write_json(output_path, result_dict)
    _write_json(effectiveness_path, effectiveness_dict)
    _print_metrics(result_dict, split_list)


if __name__ == "__main__":
    main()
