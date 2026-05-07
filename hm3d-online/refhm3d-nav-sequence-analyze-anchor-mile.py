from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

_REFINE1_PATH = _HM3D_ONLINE / "refhm3d-nav-sequence-analyze-anchor-mile-refine1.py"
_spec = importlib.util.spec_from_file_location("_mile_refine1_helpers", _REFINE1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load helper script: {_REFINE1_PATH}")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

from anchor_nav.mile import (
    MileConfig,
    build_effectiveness_record,
    correct_final_decision_with_mile,
)
from common.embodied_utils.simulator import HabitatSimulator
from data_utils import PQ3DModel
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    map_coors_to_pixel,
    pixel_to_map_coors,
)


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _object_slot_info(rep: Any, prev_object_count: int) -> Dict[str, Any]:
    cur_count = int(np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0])
    prev = int(prev_object_count)
    if cur_count < prev:
        raise RuntimeError(f"object slot count shrank from {prev} to {cur_count}")
    return {
        "prev_object_count": prev,
        "cur_object_count": cur_count,
        "new_object_slots": [int(x) for x in range(prev, cur_count)],
    }


def main() -> None:
    parser = argparse.ArgumentParser("Anchor mile minimal analysis")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
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
    parser.add_argument("--effectiveness_threshold_m", type=float, default=1.0)
    parser.add_argument("--mile_decision_radius_m", type=float, default=0.75)
    parser.add_argument("--mile_candidate_view_count", type=int, default=20)
    parser.add_argument("--mile_camera_height_m", type=float, default=1.31)
    parser.add_argument("--mile_max_snap_distance_m", type=float, default=0.60)
    parser.add_argument("--mile_scene_sample_count", type=int, default=0)
    parser.add_argument("--mile_max_ray_sample_count", type=int, default=1000)
    parser.add_argument("--mile_occlusion_radius_m", type=float, default=0.05)
    parser.add_argument("--mile_min_visibility_score", type=float, default=0.0)
    parser.add_argument("--output_root", type=str, default="./output_process")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-anchor-mile")
    cfg = MileConfig(
        decision_radius_m=float(args.mile_decision_radius_m),
        candidate_view_count=int(args.mile_candidate_view_count),
        camera_height_m=float(args.mile_camera_height_m),
        max_snap_distance_m=float(args.mile_max_snap_distance_m),
        scene_sample_count=int(args.mile_scene_sample_count),
        max_ray_sample_count=int(args.mile_max_ray_sample_count),
        occlusion_radius_m=float(args.mile_occlusion_radius_m),
        min_visibility_score=float(args.mile_min_visibility_score),
    )
    _helpers._write_json(
        out_root / "run_args.json",
        {
            "run_tag": run_tag,
            "scene_name": args.scene_name,
            "episode_id": int(args.episode_id),
            "task_id": int(args.task_id),
            "num_tasks": int(args.num_tasks),
            "description_mode": args.description_mode,
            "mile": cfg.__dict__,
            "output_root": str(out_root),
        },
    )
    print(
        f"[mile] run_tag={run_tag} scene={args.scene_name} episode={args.episode_id} "
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
    eval_goals_map = {x["object_id"]: x for x in scene_data["goals"]}
    language_goals_map = _helpers._language_only_goals_map(scene_data["goals"])
    eps = [e for e in scene_data["episode_by_sequence"] if int(e["episode_id"]) == int(args.episode_id)][0]
    task_sequence = eps["task_sequence"]
    task_end = min(int(args.task_id) + int(args.num_tasks), len(task_sequence))

    sim_settings = OmegaConf.load(str((project_root / args.sim_config).resolve()))
    agent_settings = OmegaConf.load(str((project_root / args.agent_config).resolve()))
    sim_settings["scene"] = str(_helpers._resolve_scene_mesh((project_root / args.hm3d_data_base_path).resolve(), args.scene_name))
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
        min_decision_num=int(args.decision_num_min),
    )
    pq3d.reset()

    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, 512, sim)
    vis_dist = convert_meters_to_pixel(3.0, 512, sim)
    visited_frontier: set = set()
    summaries: List[Dict[str, Any]] = []
    case_counts = {"00": 0, "01": 0, "10": 0, "11": 0}

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _helpers._build_sentence(task_type, cur_task, language_goals_map, region_map, concise=(args.description_mode == "concise"))
        eval_goals: Optional[List[Dict[str, Any]]] = None
        goal_positions: List[np.ndarray] = []
        view_points: List[List[float]] = []
        out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
        print(f"[mile][task-start] task={loop_tid} level={task_type} sentence={sentence!r}")

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        start_goal_geo = float("inf")
        episode_cum_distance = 0.0
        task_end_reason = "max_steps"
        baseline_final_target: Optional[np.ndarray] = None
        corrected_final_target: Optional[np.ndarray] = None
        final_effectiveness: Optional[Dict[str, Any]] = None
        final_mile_info: Dict[str, Any] = {"mile_called": False}

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

            scan_rgb, scan_depth, scan_state, fog, total_steps = _helpers._capture_scan_frames(
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

            st_now = agent.get_state()
            fw = detect_frontier_waypoints(
                top_down_map,
                fog,
                area_thr,
                xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1],
                enable_visualization=False,
            )
            raw_frontiers = [] if len(fw) == 0 else list(pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim))
            frontiers: List[np.ndarray] = []
            filtered_frontiers: List[Dict[str, Any]] = []
            for w in raw_frontiers:
                key = _helpers._frontier_visit_key(w)
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
                np.asarray(getattr(pq3d.representation_manager, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0]
            )
            target, is_final = pq3d.decision(
                color_list,
                depth_list,
                state_list,
                frontiers,
                sentence,
                decision_num,
            )
            register_info = _object_slot_info(pq3d.representation_manager, prev_object_count)
            baseline_target = np.asarray(target, dtype=float).reshape(3)
            aux = dict(getattr(pq3d, "last_decision_aux", {}) or {})
            print(
                f"[mile][decision] task={loop_tid} dec={decision_num} final={bool(is_final)} "
                f"frontiers={len(frontiers)}/{len(raw_frontiers)} filtered_visited={len(filtered_frontiers)} "
                f"baseline_target={baseline_target.tolist()} new_slots={register_info['new_object_slots']}"
            )

            corrected_target = baseline_target.copy()
            mile_info: Dict[str, Any] = {"mile_called": False}
            effectiveness: Optional[Dict[str, Any]] = None
            if bool(is_final):
                baseline_final_target = baseline_target.copy()
                corrected_target, mile_info = correct_final_decision_with_mile(
                    rep=pq3d.representation_manager,
                    decision_aux=aux,
                    baseline_target_xyz=baseline_target,
                    output_dir=dec_dir / "mile",
                    path_finder=pf,
                    agent_position_xyz=agent.get_state().position,
                    cfg=cfg,
                )
                corrected_target, mile_info = _helpers._apply_followability_filter(
                    mile_info=mile_info,
                    baseline_target=baseline_target,
                    pf=pf,
                    agent=agent,
                )
                eval_goals, goal_positions, view_points, _ = _helpers._eval_goal_bundle(cur_task, eval_goals_map)
                effectiveness = build_effectiveness_record(
                    baseline_target_xyz=baseline_target,
                    corrected_target_xyz=corrected_target,
                    goal_positions_xyz=goal_positions,
                    threshold_m=float(args.effectiveness_threshold_m),
                )
                effectiveness = _helpers._augment_viewpoint_effectiveness(
                    effectiveness,
                    pf=pf,
                    baseline_target_xyz=baseline_target,
                    corrected_target_xyz=corrected_target,
                    view_points=view_points,
                    threshold_m=float(args.effectiveness_threshold_m),
                )
                mile_info["effectiveness"] = effectiveness
                _helpers._write_json(dec_dir / "mile" / "mile_decision.json", mile_info)
                case = str(effectiveness["case"])
                case_counts[case] = int(case_counts.get(case, 0)) + 1
                corrected_final_target = corrected_target.copy()
                final_effectiveness = effectiveness
                final_mile_info = mile_info
                print(
                    f"[mile][module] task={loop_tid} dec={decision_num} called=True "
                    f"source={mile_info['target_source']} "
                    f"selected_slot={mile_info['selected_slot_index']} correction_applied={mile_info['correction_applied']} "
                    f"viewpoint_applied={mile_info['viewpoint_correction_applied']} "
                    f"followability_ok={mile_info.get('followability_selected_precheck', {}).get('precheck_ok', None)}"
                )
                print(
                    f"[mile][effectiveness] task={loop_tid} dec={decision_num} "
                    f"case_metric={effectiveness['case_metric']} case={case} "
                    f"baseline_gt_viewpoint_in_1m={int(effectiveness['baseline_gt_viewpoint_in_1m'])} "
                    f"corrected_gt_viewpoint_in_1m={int(effectiveness['corrected_gt_viewpoint_in_1m'])} "
                    f"baseline_vp_geo={effectiveness['baseline_nearest_gt_viewpoint_geo_m']:.3f} "
                    f"corrected_vp_geo={effectiveness['corrected_nearest_gt_viewpoint_geo_m']:.3f} "
                    f"object_l2_case={effectiveness['object_l2_case']} "
                    f"baseline_obj_l2={effectiveness['baseline_nearest_goal_dist_m']:.3f} "
                    f"corrected_obj_l2={effectiveness['corrected_nearest_goal_dist_m']:.3f}"
                )
                used_target = corrected_target.copy()
                try:
                    goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _helpers._follow_target(
                        pf=pf,
                        agent=agent,
                        sim=sim,
                        top_down_map=top_down_map,
                        fog=fog,
                        vis_dist=vis_dist,
                        used_target=used_target,
                        prev_agent_state=prev_agent_state,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                        episode_cum_distance=float(episode_cum_distance),
                    )
                    task_end_reason = "final_decision"
                except _helpers.FollowerNavigationError as exc:
                    follow_info = dict(exc.info)
                    follow_info.update({"decision_num": int(decision_num), "is_final": True, "stage": "final_follow"})
                    task_end_reason = "follower_error"
                    _helpers._write_json(out_task / "follow_error.json", {"follow_info": follow_info})
                    print(
                        f"[mile][follow-error] task={loop_tid} dec={decision_num} final=True "
                        f"error={follow_info.get('error_type')} path_found={follow_info.get('shortest_path_found')} "
                        f"geo={follow_info.get('shortest_path_geodesic_distance')}"
                    )
                _helpers._write_json(
                    dec_dir / "mile_step_summary.json",
                    {
                        "task_id": int(loop_tid),
                        "task_level": task_type,
                        "decision_num": int(decision_num),
                        "is_final": True,
                        "baseline_target": baseline_target.tolist(),
                        "corrected_target": corrected_target.tolist(),
                        "used_target": used_target.tolist(),
                        "pq3d_last_decision_aux": aux,
                        "frontier_filter_info": frontier_filter_info,
                        "register_info": register_info,
                        "follow_info": follow_info,
                        "mile": mile_info,
                        "effectiveness": effectiveness,
                    },
                )
                decision_num += 1
                break

            used_target = baseline_target.copy()
            used_frontier_key = _helpers._frontier_visit_key(used_target)
            visited_frontier.add(used_frontier_key)
            try:
                goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _helpers._follow_target(
                    pf=pf,
                    agent=agent,
                    sim=sim,
                    top_down_map=top_down_map,
                    fog=fog,
                    vis_dist=vis_dist,
                    used_target=used_target,
                    prev_agent_state=prev_agent_state,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                    episode_cum_distance=float(episode_cum_distance),
                )
            except _helpers.FollowerNavigationError as exc:
                follow_info = dict(exc.info)
                follow_info.update({"decision_num": int(decision_num), "is_final": False, "stage": "frontier_follow"})
                task_end_reason = "follower_error"
                _helpers._write_json(out_task / "follow_error.json", {"follow_info": follow_info})
                print(
                    f"[mile][follow-error] task={loop_tid} dec={decision_num} final=False "
                    f"error={follow_info.get('error_type')} path_found={follow_info.get('shortest_path_found')} "
                    f"geo={follow_info.get('shortest_path_geodesic_distance')}"
                )
                _helpers._write_json(
                    dec_dir / "mile_step_summary.json",
                    {
                        "task_id": int(loop_tid),
                        "task_level": task_type,
                        "decision_num": int(decision_num),
                        "is_final": False,
                        "baseline_target": baseline_target.tolist(),
                        "used_target": used_target.tolist(),
                        "used_frontier_key": list(used_frontier_key),
                        "pq3d_last_decision_aux": aux,
                        "frontier_filter_info": frontier_filter_info,
                        "register_info": register_info,
                        "follow_info": follow_info,
                        "mile": mile_info,
                    },
                )
                decision_num += 1
                break
            _helpers._write_json(
                dec_dir / "mile_step_summary.json",
                {
                    "task_id": int(loop_tid),
                    "task_level": task_type,
                    "decision_num": int(decision_num),
                    "is_final": False,
                    "baseline_target": baseline_target.tolist(),
                    "used_target": used_target.tolist(),
                    "used_frontier_key": list(used_frontier_key),
                    "pq3d_last_decision_aux": aux,
                    "frontier_filter_info": frontier_filter_info,
                    "register_info": register_info,
                    "follow_info": follow_info,
                    "mile": mile_info,
                },
            )
            decision_num += 1

        end_state = agent.get_state()
        if eval_goals is None:
            eval_goals, goal_positions, view_points, _ = _helpers._eval_goal_bundle(cur_task, eval_goals_map)
        start_goal_geo = _helpers._geo_dist_to_viewpoints(pf, sub_episode_start_position, view_points)
        end_goal_geo = _helpers._geo_dist_to_viewpoints(pf, end_state.position, view_points)
        baseline_target_to_gt_viewpoint_geo = (
            _helpers._geo_dist_to_viewpoints(pf, baseline_final_target, view_points)
            if baseline_final_target is not None
            else float("inf")
        )
        corrected_target_to_gt_viewpoint_geo = (
            _helpers._geo_dist_to_viewpoints(pf, corrected_final_target, view_points)
            if corrected_final_target is not None
            else float("inf")
        )
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
            "eval_only_goal_positions": [g.tolist() for g in goal_positions],
            "eval_only_gt_viewpoint_count": int(len(view_points)),
            "baseline_target_position": None if baseline_final_target is None else baseline_final_target.tolist(),
            "corrected_target_position": None if corrected_final_target is None else corrected_final_target.tolist(),
            "baseline_target_to_gt_viewpoint_geo": float(baseline_target_to_gt_viewpoint_geo),
            "corrected_target_to_gt_viewpoint_geo": float(corrected_target_to_gt_viewpoint_geo),
            "mile_case": None if final_effectiveness is None else final_effectiveness.get("case"),
            "mile_case_metric": None if final_effectiveness is None else final_effectiveness.get("case_metric"),
            "mile_target_source": final_mile_info.get("target_source"),
        }
        summaries.append(summary)
        _helpers._write_json(out_task / "summary.json", summary)
        print(
            f"[mile][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} SR={sr:.1f} SPL={spl:.4f} "
            f"case={summary['mile_case']}"
        )

    _helpers._write_json(
        out_root / "run_summary.json",
        {
            "run_tag": run_tag,
            "tasks": len(summaries),
            "case_counts": case_counts,
            "avg_sr": float(np.mean([s["sr"] for s in summaries])) if summaries else 0.0,
            "avg_spl": float(np.mean([s["spl"] for s in summaries])) if summaries else 0.0,
            "summaries": summaries,
        },
    )
    sim.close()
    print(f"[mile] done output={out_root}")


if __name__ == "__main__":
    main()
