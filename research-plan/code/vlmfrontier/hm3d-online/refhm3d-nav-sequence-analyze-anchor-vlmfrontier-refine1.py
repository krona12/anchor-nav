from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from anchor_nav.vlmfrontier import (
    vlmfrontierConfig,
    build_frontier_effectiveness_record,
    correct_frontier_with_vlm,
    register_new_object_panorama_frames,
)
from common.embodied_utils.simulator import HabitatSimulator
from data_utils import PQ3DModel
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL


def _tqdm_print(msg: str) -> None:
    tqdm.write(msg, file=sys.stdout)


class _TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"refhm3d-nav-sequence-analyze-anchor-vlmfrontier-refine1-{ts}-pid{os.getpid()}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[vlmfrontierRefine1] logging enabled -> {log_path.resolve()}")

    def _cleanup() -> None:
        try:
            print(f"[vlmfrontierRefine1] run finished, log saved -> {log_path.resolve()}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def _resolve_scene_mesh(scene_root: Path, scene_name: str) -> Path:
    sid = scene_name.split("-")[-1]
    cands = [
        scene_root / scene_name / f"{sid}.basis.glb",
        scene_root / scene_name / f"{sid}.glb",
        scene_root / scene_name / f"{sid}.basis.scene_instance.json",
        scene_root / scene_name / f"{sid}.scene_instance.json",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve scene asset for {scene_name} under {scene_root}")


def _build_sentence(
    task_type: str,
    cur_task: Dict[str, Any],
    goals_map: Dict[str, Any],
    region_map: Dict[str, Any],
    concise: bool,
) -> str:
    if task_type == "object":
        return cur_task["object_category"]
    if task_type == "room":
        return f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
    if task_type == "region":
        region_info = region_map[cur_task["region_id"]]
        if concise:
            desc = (
                region_info.get("shortest_description")
                or region_info.get("concise_description")
                or region_info.get("detailed_description")
                or ""
            )
        else:
            desc = (
                region_info.get("comprehensive_description")
                or region_info.get("detailed_description")
                or region_info.get("concise_description")
                or ""
            )
        return f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {desc}"
    if task_type == "instance":
        inst = goals_map[cur_task["instance_id"]]
        if concise:
            return inst.get("annot_unique_concise_description") or ""
        return (
            inst.get("annot_unique_detailed_description")
            or inst.get("annot_unique_normal_description")
            or inst.get("annot_appearance_description")
            or ""
        )
    raise ValueError(f"unknown task_type={task_type}")


def _capture_scan_frames(
    *,
    sim: Any,
    agent: Any,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    vis_dist: int,
    total_steps: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], np.ndarray, int]:
    scan_rgb: List[np.ndarray] = []
    scan_depth: List[np.ndarray] = []
    scan_state: List[Any] = []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        st_now = agent.get_state()
        rgb = obs["color_sensor"][:, :, :3]
        dep = obs["depth_sensor"][:, :]
        scan_rgb.append(rgb)
        scan_depth.append(dep)
        scan_state.append(st_now)
        fog[:] = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=fog,
            current_point=map_coors_to_pixel(st_now.position, top_down_map, sim),
            current_angle=get_polar_angle(st_now),
            fov=42,
            max_line_len=vis_dist,
            enable_debug_visualization=False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return scan_rgb, scan_depth, scan_state, fog, total_steps


def _follow_target(
    *,
    pf: Any,
    agent: Any,
    sim: Any,
    used_target: np.ndarray,
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float, Dict[str, Any]]:
    start_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    agent_island = pf.get_island(agent.get_state().position)
    target_nav = pf.snap_point(point=np.asarray(used_target, dtype=float).reshape(3), island_index=agent_island)
    target_nav_arr = np.asarray(target_nav, dtype=float).reshape(3)
    follower = habitat_sim.GreedyGeodesicFollower(
        pf,
        agent,
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )
    actions = follower.find_path(target_nav)
    if actions is None:
        actions = []

    goto_rgb: List[np.ndarray] = []
    goto_depth: List[np.ndarray] = []
    goto_state: List[Any] = []
    executed_actions: List[str] = []
    for a in actions:
        if not a:
            continue
        obs = sim.step(action=a)
        st2 = agent.get_state()
        goto_rgb.append(obs["color_sensor"][:, :, :3])
        goto_depth.append(obs["depth_sensor"][:, :])
        goto_state.append(st2)
        executed_actions.append(str(a))
        episode_cum_distance += float(np.linalg.norm(st2.position - prev_agent_state.position))
        prev_agent_state = st2
        total_steps += 1
        if total_steps >= int(max_steps):
            break

    end_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    follow_info = {
        "raw_target": np.asarray(used_target, dtype=float).reshape(3).tolist(),
        "snapped_target": target_nav_arr.tolist(),
        "agent_island": int(agent_island),
        "start_position": start_position.tolist(),
        "end_position": end_position.tolist(),
        "path_action_count": int(len([a for a in actions if a])),
        "executed_action_count": int(len(executed_actions)),
        "truncated_by_max_steps": bool(len(executed_actions) < len([a for a in actions if a])),
    }
    return goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, float(episode_cum_distance), follow_info


def _print_follow_exception(*, pf: Any, sim: Any, agent: Any, target: np.ndarray, exc: Exception) -> None:
    agent_state = agent.get_state()
    agent_island = pf.get_island(agent_state.position)
    target_on_navmesh = pf.snap_point(point=np.asarray(target, dtype=float).reshape(3), island_index=agent_island)
    _tqdm_print(f"GreedyGeodesicFollower error: {type(exc).__name__}")
    if not pf.is_navigable(target_on_navmesh):
        _tqdm_print("Target is not navigable")
    if not pf.is_navigable(agent_state.position):
        _tqdm_print("Agent is not navigable")
    path = habitat_sim.ShortestPath()
    path.requested_start = agent_state.position
    path.requested_end = target_on_navmesh
    if sim.pathfinder.find_path(path):
        _tqdm_print(f"geodesic_distance: {path.geodesic_distance}")
    else:
        _tqdm_print("cannt find path")


def _goal_positions(goals: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for g in goals:
        pos = g.get("position", [])
        if isinstance(pos, list) and len(pos) >= 3:
            out.append(np.asarray(pos, dtype=float).reshape(3))
    return out


def _view_points(goals: Sequence[Dict[str, Any]]) -> List[List[float]]:
    return [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]


def _geo_dist_to_viewpoints(pf: Any, start_pos: Sequence[float], view_points: Sequence[Sequence[float]]) -> float:
    if len(view_points) == 0:
        return float("inf")
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = start_pos
    path.requested_ends = list(view_points)
    return float(path.geodesic_distance) if pf.find_path(path) else float("inf")


def _frontier_visit_key(point: Sequence[float], resolution_m: float = 0.1) -> Tuple[int, int, int]:
    arr = np.asarray(point, dtype=float).reshape(3)
    return tuple(np.rint(arr / float(resolution_m)).astype(int).tolist())


def _nearest_goal_dist(point: Optional[np.ndarray], goals: Sequence[np.ndarray]) -> float:
    if point is None or len(goals) == 0:
        return float("inf")
    p = np.asarray(point, dtype=float).reshape(3)
    return float(min(float(np.linalg.norm(p - g)) for g in goals))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _sequence_compute_metric_results(result_dict: Dict[str, Any]) -> None:
    rows = result_dict.get("sequence", [])
    if not rows:
        _tqdm_print("[Metrics] sequence count=0")
        return
    avg_sr = float(np.mean([float(x.get("sr", 0.0)) for x in rows]))
    avg_spl = float(np.mean([float(x.get("spl", 0.0)) for x in rows]))
    avg_time = float(np.mean([float(x.get("task_time_sec", 0.0)) for x in rows]))
    _tqdm_print(f"[Metrics] sequence count={len(rows)}, avg_sr={avg_sr:.6f}, avg_spl={avg_spl:.6f}, avg_task_time_sec={avg_time:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser("RefHM3D anchor vlmfrontier refine1 batch analysis")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=0.2)
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
    parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
    parser.add_argument("--sim_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    parser.add_argument("--agent_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))
    parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/vlmfrontier"))
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--vlmfrontier_top_k", type=int, default=5)
    parser.add_argument("--vlmfrontier_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
    parser.add_argument("--vlmfrontier_vlm_timeout", type=int, default=60)
    parser.add_argument("--panorama_subsample_frames", type=int, default=12)
    parser.add_argument("--effectiveness_threshold_m", type=float, default=1.0)
    parser.add_argument("--quiet_nav_steps", action="store_true")
    args = parser.parse_args()

    output_log_dir = Path(args.output_log_dir).expanduser().resolve()
    _setup_run_logging(output_log_dir)
    cfg = vlmfrontierConfig(
        top_k=int(args.vlmfrontier_top_k),
        vlm_model=str(args.vlmfrontier_vlm_model),
        vlm_timeout=int(args.vlmfrontier_vlm_timeout),
        panorama_subsample_frames=int(args.panorama_subsample_frames),
    )
    _tqdm_print(
        f"[vlmfrontierRefine1] cfg top_k={cfg.top_k} model={cfg.vlm_model} timeout={cfg.vlm_timeout} "
        f"pano_frames={cfg.panorama_subsample_frames} start_ratio={args.start_ratio} end_ratio={args.end_ratio}"
    )

    navigation_data_root = Path(args.navigation_data_path).expanduser().resolve()
    scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
    if not scene_data_paths:
        raise FileNotFoundError(f"No *.json.gz found under navigation_data_path={navigation_data_root}")
    scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]

    out_name = f"refhm3d_seq_vlmfrontier_refine1_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_vlmfrontier_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
    if args.concise_description:
        out_name = f"refhm3d_seq_vlmfrontier_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
        eff_name = f"refhm3d_seq_vlmfrontier_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    output_path = output_log_dir / out_name
    effectiveness_path = output_log_dir / eff_name

    if output_path.exists():
        result_dict = json.load(open(output_path, "r", encoding="utf-8"))
        existing_episodes = {
            "_".join([r["scene_name"], r["navigation_type"], str(r["episode_id"])])
            for rows in result_dict.values()
            for r in rows
        }
    else:
        result_dict = {"sequence": []}
        existing_episodes = set()

    if effectiveness_path.exists():
        effectiveness_dict = json.load(open(effectiveness_path, "r", encoding="utf-8"))
    else:
        effectiveness_dict = {"records": []}

    pq3d = PQ3DModel(
        str(Path(args.pq3d_stage1_path).expanduser().resolve()),
        str(Path(args.pq3d_stage2_path).expanduser().resolve()),
        min_decision_num=int(args.decision_num_min),
    )

    for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
        scene_name = scene_data_path.name.split(".")[0]
        with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
            scene_data = json.load(f)
        region_map = scene_data["region_annotation"]
        episode_mapping = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }
        goals_map = {x["object_id"]: x for x in scene_data["goals"]}

        for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
            pq3d.reset()
            decision_num = 0
            visited_frontier: set = set()
            panorama_frames_by_slot: Dict[int, List[np.ndarray]] = {}
            episode_id = cur_episode["episode_id"]
            navigation_type = cur_episode["navigation_type"]
            episode_key = "_".join([scene_name, navigation_type, str(episode_id)])
            if episode_key in existing_episodes:
                continue

            sim_settings = OmegaConf.load(str(Path(args.sim_config).expanduser().resolve()))
            agent_settings = OmegaConf.load(str(Path(args.agent_config).expanduser().resolve()))
            sim_settings["scene"] = str(_resolve_scene_mesh(Path(args.hm3d_data_base_path).expanduser().resolve(), scene_name))
            abstract_sim = HabitatSimulator(sim_settings, agent_settings)
            sim = abstract_sim.simulator
            agent = abstract_sim.agent
            pf = sim.pathfinder
            st = habitat_sim.AgentState()
            st.position = cur_episode["start_position"]
            st.rotation = cur_episode["start_rotation"]
            agent.set_state(st)

            map_resolution = 512
            top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
            fog = np.zeros_like(top_down_map)
            area_thr = convert_meters_to_pixel(9, map_resolution, sim)
            vis_dist = convert_meters_to_pixel(3.0, map_resolution, sim)
            out_episode_dir = output_log_dir / "process" / f"scene={scene_name}" / f"episode={episode_id}"

            for idx, task_ref in enumerate(cur_episode["task_sequence"]):
                task_type, task_idx = task_ref
                if task_type != "instance":
                    continue
                task_t0 = time.perf_counter()
                cur_task = episode_mapping[task_type][task_idx]
                sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=bool(args.concise_description))
                goals_ids = list(cur_task.get("target_object_ids", []))
                goals = [goals_map[x] for x in goals_ids if x in goals_map]
                goal_positions = _goal_positions(goals)
                view_points = _view_points(goals)
                goal_category = goals[0]["object_category"] if goals else cur_task.get("object_category", "")
                out_task = out_episode_dir / f"task={idx}"
                out_task.mkdir(parents=True, exist_ok=True)
                _tqdm_print(f"[vlmfrontier-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} sentence={sentence!r}")

                total_steps = 0
                prev_agent_state = agent.get_state()
                sub_episode_start_position = prev_agent_state.position
                episode_cum_distance = 0.0
                goto_rgb: List[np.ndarray] = []
                goto_depth: List[np.ndarray] = []
                goto_state: List[Any] = []
                baseline_final_target: Optional[np.ndarray] = None
                corrected_final_target: Optional[np.ndarray] = None
                final_vlmfrontier_info: Dict[str, Any] = {"vlmfrontier_called": False}
                final_effectiveness: Optional[Dict[str, Any]] = None
                final_follow_info: Optional[Dict[str, Any]] = None
                task_effective_logs: List[Dict[str, Any]] = []
                task_end_reason = "max_steps"

                while total_steps < int(args.max_steps):
                    color_list: List[np.ndarray] = []
                    depth_list: List[np.ndarray] = []
                    state_list: List[Any] = []
                    if len(goto_rgb) > 6:
                        step = max(1, len(goto_rgb) // 6)
                        goto_rgb = [goto_rgb[i] for i in range(0, len(goto_rgb), step)][:6]
                        goto_depth = [goto_depth[i] for i in range(0, len(goto_depth), step)][:6]
                        goto_state = [goto_state[i] for i in range(0, len(goto_state), step)][:6]
                    color_list.extend(goto_rgb)
                    depth_list.extend(goto_depth)
                    state_list.extend(goto_state)

                    scan_rgb, scan_depth, scan_state, fog, total_steps = _capture_scan_frames(
                        sim=sim,
                        agent=agent,
                        top_down_map=top_down_map,
                        fog=fog,
                        vis_dist=vis_dist,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                    )
                    color_list.extend(scan_rgb)
                    depth_list.extend(scan_depth)
                    state_list.extend(scan_state)
                    if total_steps >= int(args.max_steps):
                        break

                    st_now = agent.get_state()
                    fw = detect_frontier_waypoints(
                        top_down_map,
                        fog,
                        area_thr,
                        xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1],
                        enable_visualization=False,
                    )
                    if len(fw) == 0:
                        raw_frontiers: List[np.ndarray] = []
                    else:
                        raw_frontiers = list(pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim))
                    frontiers: List[np.ndarray] = []
                    filtered_frontiers: List[Dict[str, Any]] = []
                    for w in raw_frontiers:
                        key = _frontier_visit_key(w)
                        if key in visited_frontier:
                            filtered_frontiers.append({"key": list(key), "point": np.asarray(w, dtype=float).reshape(3).tolist()})
                            continue
                        frontiers.append(w)
                    frontier_filter_info = {
                        "raw_frontier_count": int(len(raw_frontiers)),
                        "frontier_count_after_visited_filter": int(len(frontiers)),
                        "visited_frontier_count": int(len(visited_frontier)),
                        "filtered_as_visited": filtered_frontiers,
                    }

                    dec_dir = out_task / f"dec_{decision_num:03d}"
                    dec_dir.mkdir(parents=True, exist_ok=True)
                    prev_object_count = int(
                        np.asarray(getattr(pq3d.representation_manager, "object_score", np.zeros((0,))), dtype=float)
                        .reshape(-1)
                        .shape[0]
                    )
                    target, is_final = pq3d.decision(
                        color_list,
                        depth_list,
                        state_list,
                        frontiers,
                        sentence,
                        decision_num,
                        analysis_output_dir=str(dec_dir.resolve()),
                    )
                    register_info = register_new_object_panorama_frames(
                        rep=pq3d.representation_manager,
                        prev_object_count=prev_object_count,
                        color_list=color_list,
                        panorama_frames_by_slot=panorama_frames_by_slot,
                    )
                    baseline_target = np.asarray(target, dtype=float).reshape(3)
                    aux = dict(getattr(pq3d, "last_decision_aux", {}) or {})
                    if not args.quiet_nav_steps:
                        _tqdm_print(
                            f"[vlmfrontier-refine1][decision] scene={scene_name} ep={episode_id} task={idx} "
                            f"dec={decision_num} final={bool(is_final)} frontiers={len(frontiers)}/{len(raw_frontiers)} "
                            f"filtered_visited={len(filtered_frontiers)} baseline_target={baseline_target.tolist()}"
                        )

                    corrected_target = baseline_target.copy()
                    vlmfrontier_info: Dict[str, Any] = {"vlmfrontier_called": False}
                    effectiveness: Optional[Dict[str, Any]] = None

                    if bool(is_final):
                        baseline_final_target = baseline_target.copy()
                        corrected_target, vlmfrontier_info = correct_frontier_with_vlm(
                            description=sentence,
                            rep=pq3d.representation_manager,
                            decision_aux=aux,
                            stage2_json_path=dec_dir / "stage2_decision.json",
                            baseline_target_xyz=baseline_target,
                            panorama_frames_by_slot=panorama_frames_by_slot,
                            output_dir=dec_dir / "vlmfrontier",
                            cfg=cfg,
                        )
                        effectiveness = build_frontier_effectiveness_record(
                            baseline_target_xyz=baseline_target,
                            corrected_target_xyz=corrected_target,
                            goal_positions_xyz=goal_positions,
                            threshold_m=float(args.effectiveness_threshold_m),
                        )
                        corrected_final_target = corrected_target.copy()
                        final_vlmfrontier_info = vlmfrontier_info
                        final_effectiveness = effectiveness
                        task_effective_logs.append(
                            {
                                "scene_name": scene_name,
                                "episode_id": int(episode_id),
                                "task_id": int(idx),
                                "decision_num": int(decision_num),
                                "sentence": sentence,
                                "vlmfrontier": vlmfrontier_info,
                                "effectiveness": effectiveness,
                            }
                        )
                        _tqdm_print(
                            f"[vlmfrontier-refine1][module] scene={scene_name} ep={episode_id} task={idx} "
                            f"dec={decision_num} best_index={vlmfrontier_info['selection']['best_index']} "
                            f"correction_applied={vlmfrontier_info['correction_applied']} "
                            f"panorama_vetoed={vlmfrontier_info.get('panorama_vetoed_correction', False)} "
                            f"selected_target_used={vlmfrontier_info.get('selected_target_used', False)} "
                            f"case={effectiveness['case']} baseline_dist={effectiveness['baseline_nearest_goal_dist_m']:.3f} "
                            f"corrected_dist={effectiveness['corrected_nearest_goal_dist_m']:.3f}"
                        )
                        used_target = corrected_target.copy()
                        try:
                            goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _follow_target(
                                pf=pf,
                                agent=agent,
                                sim=sim,
                                used_target=used_target,
                                prev_agent_state=prev_agent_state,
                                total_steps=total_steps,
                                max_steps=int(args.max_steps),
                                episode_cum_distance=float(episode_cum_distance),
                            )
                            final_follow_info = follow_info
                            task_end_reason = "final_decision"
                        except Exception as exc:
                            _print_follow_exception(pf=pf, sim=sim, agent=agent, target=used_target, exc=exc)
                            follow_info = None
                            task_end_reason = "follow_error"
                        _write_json(
                            dec_dir / "vlmfrontier_step_summary.json",
                            {
                                "task_id": int(idx),
                                "decision_num": int(decision_num),
                                "is_final": True,
                                "baseline_target": baseline_target.tolist(),
                                "corrected_target": corrected_target.tolist(),
                                "used_target": used_target.tolist(),
                                "pq3d_last_decision_aux": aux,
                                "frontier_filter_info": frontier_filter_info,
                                "register_info": register_info,
                                "follow_info": follow_info,
                                "vlmfrontier": vlmfrontier_info,
                                "effectiveness": effectiveness,
                            },
                        )
                        decision_num += 1
                        break

                    used_target = baseline_target.copy()
                    used_frontier_key = _frontier_visit_key(used_target)
                    visited_frontier.add(used_frontier_key)
                    try:
                        goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _follow_target(
                            pf=pf,
                            agent=agent,
                            sim=sim,
                            used_target=used_target,
                            prev_agent_state=prev_agent_state,
                            total_steps=total_steps,
                            max_steps=int(args.max_steps),
                            episode_cum_distance=float(episode_cum_distance),
                        )
                    except Exception as exc:
                        _print_follow_exception(pf=pf, sim=sim, agent=agent, target=used_target, exc=exc)
                        follow_info = None
                        task_end_reason = "follow_error"
                    _write_json(
                        dec_dir / "vlmfrontier_step_summary.json",
                        {
                            "task_id": int(idx),
                            "decision_num": int(decision_num),
                            "is_final": False,
                            "baseline_target": baseline_target.tolist(),
                            "used_target": used_target.tolist(),
                            "used_frontier_key": list(used_frontier_key),
                            "pq3d_last_decision_aux": aux,
                            "frontier_filter_info": frontier_filter_info,
                            "register_info": register_info,
                            "follow_info": follow_info,
                            "vlmfrontier": vlmfrontier_info,
                        },
                    )
                    decision_num += 1
                    if follow_info is None:
                        break

                task_time = float(time.perf_counter() - task_t0)
                end_state = agent.get_state()
                start_goal_geo = _geo_dist_to_viewpoints(pf, sub_episode_start_position, view_points)
                end_goal_geo = _geo_dist_to_viewpoints(pf, end_state.position, view_points)
                if np.isinf(start_goal_geo) or np.isinf(end_goal_geo):
                    sr = 0.0
                    spl = 0.0
                else:
                    sr = 1.0 if end_goal_geo <= float(args.success_distance) else 0.0
                    spl = float(sr * start_goal_geo / max(start_goal_geo, max(episode_cum_distance, 1e-12)))

                baseline_target_to_goal_l2 = _nearest_goal_dist(baseline_final_target, goal_positions)
                corrected_target_to_goal_l2 = _nearest_goal_dist(corrected_final_target, goal_positions)
                correction_applied = bool(final_vlmfrontier_info.get("correction_applied", False))
                selected_target_used = bool(final_vlmfrontier_info.get("selected_target_used", False))
                vlmfrontier_helpful = None
                if correction_applied and np.isfinite(baseline_target_to_goal_l2) and np.isfinite(corrected_target_to_goal_l2):
                    vlmfrontier_helpful = bool(corrected_target_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6)

                row = {
                    "scene_name": scene_name,
                    "episode_id": int(episode_id),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "sr": float(sr),
                    "spl": float(spl),
                    "object_category": goal_category,
                    "task_time_sec": task_time,
                    "steps_total": int(total_steps),
                    "decisions": int(decision_num),
                    "end_reason": task_end_reason,
                    "start_goal_geo": float(start_goal_geo),
                    "end_goal_geo": float(end_goal_geo),
                    "episode_cum_distance": float(episode_cum_distance),
                    "goal_positions": [g.tolist() for g in goal_positions],
                    "baseline_target_position": None if baseline_final_target is None else baseline_final_target.tolist(),
                    "corrected_target_position": None if corrected_final_target is None else corrected_final_target.tolist(),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "corrected_target_to_goal_l2": float(corrected_target_to_goal_l2),
                    "vlmfrontier_called": bool(final_vlmfrontier_info.get("vlmfrontier_called", False)),
                    "vlmfrontier_best_index": (final_vlmfrontier_info.get("selection") or {}).get("best_index"),
                    "vlmfrontier_match_type": (final_vlmfrontier_info.get("selection") or {}).get("match_type"),
                    "vlmfrontier_correction_applied": correction_applied,
                    "vlmfrontier_selected_target_used": selected_target_used,
                    "vlmfrontier_panorama_vetoed": bool(final_vlmfrontier_info.get("panorama_vetoed_correction", False)),
                    "vlmfrontier_panorama_has_task_object": None
                    if final_vlmfrontier_info.get("panorama_verify") is None
                    else bool(final_vlmfrontier_info["panorama_verify"].get("has_task_object", False)),
                    "vlmfrontier_case": None if final_effectiveness is None else final_effectiveness.get("case"),
                    "vlmfrontier_helpful": vlmfrontier_helpful,
                    "final_follow_info": final_follow_info,
                }
                result_dict.setdefault(navigation_type, []).append(row)
                effectiveness_dict.setdefault("records", []).append(
                    {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(idx),
                        "task_level": task_type,
                        "navigation_type": navigation_type,
                        "sentence": sentence,
                        "effectiveness": final_effectiveness,
                        "vlmfrontier": final_vlmfrontier_info,
                        "task_effective_logs": task_effective_logs,
                    }
                )
                _write_json(out_task / "summary.json", row)
                _tqdm_print(
                    f"[vlmfrontier-refine1][task-summary] scene={scene_name} ep={episode_id} task={idx} "
                    f"SR={sr:.1f} SPL={spl:.4f} steps={total_steps} decisions={decision_num} "
                    f"end_reason={task_end_reason} case={row['vlmfrontier_case']} helpful={vlmfrontier_helpful}"
                )

            sim.close()
            _write_json(output_path, result_dict)
            _write_json(effectiveness_path, effectiveness_dict)
            _sequence_compute_metric_results(result_dict)

    _sequence_compute_metric_results(result_dict)


if __name__ == "__main__":
    main()

