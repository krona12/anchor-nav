from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

_HM3D_ONLINE = Path(__file__).resolve().parent
_MTU3D_ROOT = _HM3D_ONLINE.parent
for _p in (_HM3D_ONLINE, _MTU3D_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from anchor_nav.vlmpanobank import (
    VLMPanoBankConfig,
    build_effectiveness_record,
    correct_final_decision_with_vlmpanobank,
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


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float]:
    agent_island = pf.get_island(agent.get_state().position)
    target_nav = pf.snap_point(point=used_target, island_index=agent_island)
    follower = habitat_sim.GreedyGeodesicFollower(
        pf,
        agent,
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )
    try:
        actions = follower.find_path(target_nav)
    except Exception:
        actions = []
    goto_rgb: List[np.ndarray] = []
    goto_depth: List[np.ndarray] = []
    goto_state: List[Any] = []
    for a in actions:
        if not a:
            continue
        obs = sim.step(action=a)
        st2 = agent.get_state()
        goto_rgb.append(obs["color_sensor"][:, :, :3])
        goto_depth.append(obs["depth_sensor"][:, :])
        goto_state.append(st2)
        episode_cum_distance += float(np.linalg.norm(st2.position - prev_agent_state.position))
        prev_agent_state = st2
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("Anchor vlmpanobank minimal analysis")
    parser.add_argument("--scene_name", type=str, default="00802-wcojb4TFT35")
    parser.add_argument("--episode_id", type=int, default=17)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--navigation_data_path", type=str, default="LangMap_Annotations")
    parser.add_argument("--hm3d_data_base_path", type=str, default="datascene")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--pq3d_stage1_path", type=str, default="checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="checkpoint/stage2-fine-tune-goat")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--vlmpanobank_top_k", type=int, default=5)
    parser.add_argument("--vlmpanobank_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
    parser.add_argument("--vlmpanobank_vlm_timeout", type=int, default=60)
    parser.add_argument("--panorama_subsample_frames", type=int, default=12)
    parser.add_argument("--effectiveness_threshold_m", type=float, default=1.0)
    parser.add_argument("--output_root", type=str, default="./output_process")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-anchor-vlmpanobank")
    cfg = VLMPanoBankConfig(
        top_k=int(args.vlmpanobank_top_k),
        vlm_model=str(args.vlmpanobank_vlm_model),
        vlm_timeout=int(args.vlmpanobank_vlm_timeout),
        panorama_subsample_frames=int(args.panorama_subsample_frames),
    )
    _write_json(
        out_root / "run_args.json",
        {
            "run_tag": run_tag,
            "scene_name": args.scene_name,
            "episode_id": int(args.episode_id),
            "task_id": int(args.task_id),
            "num_tasks": int(args.num_tasks),
            "description_mode": args.description_mode,
            "vlmpanobank": cfg.__dict__,
            "output_root": str(out_root),
        },
    )
    print(
        f"[vlmpanobank] run_tag={run_tag} scene={args.scene_name} episode={args.episode_id} "
        f"task_id={args.task_id} num_tasks={args.num_tasks} output={out_root}"
    )

    scene_file = (project_root / args.navigation_data_path / f"{args.scene_name}.json.gz").resolve()
    with gzip.open(scene_file, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    region_map = scene_data["region_annotation"]
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    goals_map = {x["object_id"]: x for x in scene_data["goals"]}
    eps = [e for e in scene_data["episode_by_sequence"] if int(e["episode_id"]) == int(args.episode_id)][0]
    task_sequence = eps["task_sequence"]
    task_end = min(int(args.task_id) + int(args.num_tasks), len(task_sequence))

    sim_settings = OmegaConf.load(str((project_root / args.sim_config).resolve()))
    agent_settings = OmegaConf.load(str((project_root / args.agent_config).resolve()))
    sim_settings["scene"] = str(_resolve_scene_mesh((project_root / args.hm3d_data_base_path).resolve(), args.scene_name))
    abstract_sim = HabitatSimulator(sim_settings, agent_settings)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    pf = sim.pathfinder
    st = habitat_sim.AgentState()
    st.position = eps["start_position"]
    st.rotation = eps["start_rotation"]
    agent.set_state(st)

    pq3d = PQ3DModel(
        str((project_root / args.pq3d_stage1_path).resolve()),
        str((project_root / args.pq3d_stage2_path).resolve()),
        min_decision_num=args.decision_num_min,
    )
    pq3d.reset()

    map_resolution = 512
    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, map_resolution, sim)
    vis_dist = convert_meters_to_pixel(3.0, map_resolution, sim)
    visited_frontier: set = set()
    panorama_frames_by_slot: Dict[int, List[np.ndarray]] = {}
    summaries: List[Dict[str, Any]] = []

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
        goals_ids = list(cur_task.get("target_object_ids", []))
        goals = [goals_map[x] for x in goals_ids if x in goals_map]
        goal_positions = _goal_positions(goals)
        view_points = _view_points(goals)
        out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
        print(f"[vlmpanobank][task-start] task={loop_tid} level={task_type} sentence={sentence!r}")

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        start_goal_geo = _geo_dist_to_viewpoints(pf, sub_episode_start_position, view_points)
        episode_cum_distance = 0.0
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
                raw_frontiers = []
            else:
                raw_frontiers = list(pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim))
            frontiers = []
            filtered_frontiers = []
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

            dec_dir = _ensure_dir(out_task / f"dec_{decision_num:03d}")
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
            print(
                f"[vlmpanobank][decision] task={loop_tid} dec={decision_num} final={bool(is_final)} "
                f"frontiers={len(frontiers)}/{len(raw_frontiers)} "
                f"filtered_visited={len(filtered_frontiers)} baseline_target={baseline_target.tolist()} "
                f"new_slots={register_info['registered_slots']}"
            )

            corrected_target = baseline_target.copy()
            vlmpanobank_info: Dict[str, Any] = {"vlmpanobank_called": False}
            effectiveness: Optional[Dict[str, Any]] = None

            if bool(is_final):
                corrected_target, vlmpanobank_info = correct_final_decision_with_vlmpanobank(
                    description=sentence,
                    rep=pq3d.representation_manager,
                    decision_aux=aux,
                    stage2_json_path=dec_dir / "stage2_decision.json",
                    baseline_target_xyz=baseline_target,
                    panorama_frames_by_slot=panorama_frames_by_slot,
                    output_dir=dec_dir / "vlmpanobank",
                    cfg=cfg,
                )
                effectiveness = build_effectiveness_record(
                    baseline_target_xyz=baseline_target,
                    corrected_target_xyz=corrected_target,
                    goal_positions_xyz=goal_positions,
                    threshold_m=float(args.effectiveness_threshold_m),
                )
                print(
                    f"[vlmpanobank][module] task={loop_tid} dec={decision_num} called=True "
                    f"best_index={vlmpanobank_info['selection']['best_index']} "
                    f"selected_slot={vlmpanobank_info['selected_slot_index']} "
                    f"correction_applied={vlmpanobank_info['correction_applied']} "
                    f"panorama_vetoed={vlmpanobank_info.get('panorama_vetoed_correction', False)} "
                    f"selected_target_used={vlmpanobank_info.get('selected_target_used', False)} "
                    f"panorama_has_task_object="
                    f"{None if vlmpanobank_info['panorama_verify'] is None else vlmpanobank_info['panorama_verify']['has_task_object']}"
                )
                print(
                    f"[vlmpanobank][effectiveness] task={loop_tid} dec={decision_num} "
                    f"case={effectiveness['case']} baseline_in_1m={int(effectiveness['baseline_in_1m'])} "
                    f"corrected_in_1m={int(effectiveness['corrected_in_1m'])} "
                    f"baseline_dist={effectiveness['baseline_nearest_goal_dist_m']:.3f} "
                    f"corrected_dist={effectiveness['corrected_nearest_goal_dist_m']:.3f}"
                )
                used_target = corrected_target.copy()
                goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                    pf=pf,
                    agent=agent,
                    sim=sim,
                    used_target=used_target,
                    prev_agent_state=prev_agent_state,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                    episode_cum_distance=float(episode_cum_distance),
                )
                task_end_reason = "final_decision"
                _write_json(
                    dec_dir / "vlmpanobank_step_summary.json",
                    {
                        "task_id": int(loop_tid),
                        "decision_num": int(decision_num),
                        "is_final": True,
                        "baseline_target": baseline_target.tolist(),
                        "corrected_target": corrected_target.tolist(),
                        "used_target": used_target.tolist(),
                        "pq3d_last_decision_aux": aux,
                        "frontier_filter_info": frontier_filter_info,
                        "register_info": register_info,
                        "vlmpanobank": vlmpanobank_info,
                        "effectiveness": effectiveness,
                    },
                )
                decision_num += 1
                break

            used_target = baseline_target.copy()
            used_frontier_key = _frontier_visit_key(used_target)
            visited_frontier.add(used_frontier_key)
            goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                pf=pf,
                agent=agent,
                sim=sim,
                used_target=used_target,
                prev_agent_state=prev_agent_state,
                total_steps=total_steps,
                max_steps=int(args.max_steps),
                episode_cum_distance=float(episode_cum_distance),
            )
            _write_json(
                dec_dir / "vlmpanobank_step_summary.json",
                {
                    "task_id": int(loop_tid),
                        "decision_num": int(decision_num),
                        "is_final": False,
                        "baseline_target": baseline_target.tolist(),
                        "used_target": used_target.tolist(),
                        "used_frontier_key": list(used_frontier_key),
                        "pq3d_last_decision_aux": aux,
                        "frontier_filter_info": frontier_filter_info,
                        "register_info": register_info,
                        "vlmpanobank": vlmpanobank_info,
                    },
            )
            decision_num += 1

        end_state = agent.get_state()
        end_goal_geo = _geo_dist_to_viewpoints(pf, end_state.position, view_points)
        if np.isinf(start_goal_geo) or np.isinf(end_goal_geo):
            sr = 0.0
            spl = 0.0
        else:
            sr = 1.0 if end_goal_geo <= float(args.success_distance) else 0.0
            spl = float(sr * start_goal_geo / max(start_goal_geo, max(episode_cum_distance, 1e-12)))
        summary = {
            "run_tag": run_tag,
            "scene_name": args.scene_name,
            "episode_id": int(args.episode_id),
            "task_id": int(loop_tid),
            "task_level": task_type,
            "steps_total": int(total_steps),
            "decisions": int(decision_num),
            "end_reason": task_end_reason,
            "sr": float(sr),
            "spl": float(spl),
            "start_goal_geo": float(start_goal_geo),
            "end_goal_geo": float(end_goal_geo),
            "episode_cum_distance": float(episode_cum_distance),
        }
        summaries.append(summary)
        _write_json(out_task / "summary.json", summary)
        print(
            f"[vlmpanobank][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} SR={sr:.1f} SPL={spl:.4f}"
        )

    _write_json(
        out_root / "run_summary.json",
        {
            "run_tag": run_tag,
            "tasks": len(summaries),
            "avg_sr": float(np.mean([s["sr"] for s in summaries])) if summaries else 0.0,
            "avg_spl": float(np.mean([s["spl"] for s in summaries])) if summaries else 0.0,
            "summaries": summaries,
        },
    )
    sim.close()
    print(f"[vlmpanobank] done output={out_root}")


if __name__ == "__main__":
    main()




