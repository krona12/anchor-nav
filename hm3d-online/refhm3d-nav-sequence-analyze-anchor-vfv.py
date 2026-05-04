"""RefHM3D：单场景 VFV 分析（Phase1 全景验证 + Phase2 PQ3D 锚点查询）。

锚点过滤规则在 ``anchor_nav.vfv.decompose_target_anchor`` 的 prompt 中；本脚本不对锚点做额外语义过滤。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from anchor_nav.posnode import _save_rgb_jpg, stitch_panorama
from anchor_nav.vfv import (
    build_query_fn_from_pq3d_stage2,
    decompose_target_anchor,
    select_best_anchor_object,
    verify_description_visible,
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


def _build_sentence(task_type: str, cur_task: Dict[str, Any], goals_map: Dict[str, Any], region_map: Dict[str, Any], concise: bool) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        "Anchor VFV analyze（Phase1+Phase2）。锚点约束见 anchor_nav.vfv.decompose_target_anchor 内 VLM prompt。"
    )
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default="LangMap_Annotations")
    parser.add_argument("--hm3d_data_base_path", type=str, default="datascene")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--pq3d_stage1_path", type=str, default="checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="checkpoint/stage2-fine-tune-goat")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
    parser.add_argument("--anchor_top_k", type=int, default=16)
    parser.add_argument("--panorama_subsample_frames", type=int, default=12)
    parser.add_argument(
        "--vfv_skip_verify_if_object_logit_gap",
        type=float,
        default=-1.0,
        help=">=0：PQ3D 物体 top1-top2 logit 差 >= 该值则跳过全景+VLM；-1 关闭",
    )
    parser.add_argument(
        "--vfv_min_remaining_steps",
        type=int,
        default=0,
        help="Phase1 跟点后剩余步数低于该值则跳过验证；0 关闭",
    )
    parser.add_argument("--vfv_verify_parse_attempts", type=int, default=2)
    parser.add_argument("--output_root", type=str, default="./output_logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)
    print(
        f"[vfv] run_tag={run_tag} anchor_top_k={args.anchor_top_k} pano_frames={args.panorama_subsample_frames} "
        f"skip_gap>={args.vfv_skip_verify_if_object_logit_gap} min_rem={args.vfv_min_remaining_steps}"
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
    query_fn = build_query_fn_from_pq3d_stage2(pq3d)

    map_resolution = 512
    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, map_resolution, sim)
    vis_dist = convert_meters_to_pixel(3.0, map_resolution, sim)
    visited_frontier: set = set()

    out_root = None
    if output_enabled:
        out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-anchor-vfv")

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
        decomp = decompose_target_anchor(sentence, args.vlm_model)
        print(
            f"[vfv][task-start] task={loop_tid} level={task_type} "
            f"target_desc={decomp.target_desc!r} anchor_desc={decomp.anchor_desc!r} parse_ok={decomp.parse_ok}"
        )

        goals_ids = list(cur_task.get("target_object_ids", []))
        goals = [goals_map[x] for x in goals_ids if x in goals_map]
        goal_positions = [
            np.asarray(g.get("position", []), dtype=float).reshape(3)
            for g in goals
            if isinstance(g, dict) and len(g.get("position", [])) >= 3
        ]

        out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}") if out_root else None
        pano_dir = _ensure_dir(out_task / "panorama") if out_task else None

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        episode_cum_distance = 0.0
        task_end_reason = "max_steps"

        while total_steps < args.max_steps:
            color_list, depth_list, state_list = [], [], []
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
            if total_steps >= args.max_steps:
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
                frontiers = []
            else:
                frontiers = pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim)
            frontiers = [w for w in frontiers if tuple(np.round(w, 1)) not in visited_frontier]

            target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence, decision_num)
            used_target = np.asarray(target, dtype=float).reshape(3).copy()
            print(
                f"[vfv][decision] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline_target={used_target.tolist()} final={bool(is_final)}"
            )

            vfv_log: Dict[str, Any] = {"phase": "phase1", "triggered": False}

            if is_final:
                phase1_pos = used_target.copy()
                aux = getattr(pq3d, "last_decision_aux", {}) or {}
                gap = float(aux.get("object_top1_top2_logit_gap", 0.0))

                goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                    pf=pf,
                    agent=agent,
                    sim=sim,
                    used_target=phase1_pos,
                    prev_agent_state=prev_agent_state,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                    episode_cum_distance=float(episode_cum_distance),
                )

                remaining_after_p1 = int(args.max_steps) - int(total_steps)
                skip_gap = (
                    float(args.vfv_skip_verify_if_object_logit_gap) >= 0.0
                    and gap >= float(args.vfv_skip_verify_if_object_logit_gap)
                )
                skip_steps = (
                    int(args.vfv_min_remaining_steps) > 0
                    and remaining_after_p1 < int(args.vfv_min_remaining_steps)
                )
                skip_verify = skip_gap or skip_steps

                if skip_verify:
                    verify = {
                        "visible": True,
                        "skipped": True,
                        "skip_reason": "object_logit_gap" if skip_gap else "remaining_steps",
                        "object_top1_top2_logit_gap": gap,
                        "remaining_steps_after_phase1_follow": remaining_after_p1,
                        "full_match": True,
                        "strong_anchor_match": False,
                        "confidence": "high",
                        "reason": "vfv_verify_skipped_deployable",
                        "raw": "",
                        "parse_attempts": 0,
                    }
                    vfv_log = {
                        "phase": "phase1",
                        "triggered": False,
                        "verify_skipped": True,
                        "phase1_target": phase1_pos.tolist(),
                        "verify": verify,
                        "panorama_path": None,
                        "decompose": {
                            "target_desc": decomp.target_desc,
                            "anchor_desc": decomp.anchor_desc,
                            "anchor_descs": list(decomp.anchor_descs),
                        },
                    }
                    used_target = phase1_pos.copy()
                else:
                    verify_rgb, _, _, fog, total_steps = _capture_scan_frames(
                        sim=sim,
                        agent=agent,
                        top_down_map=top_down_map,
                        fog=fog,
                        vis_dist=vis_dist,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                    )
                    verify_rgb = list(reversed(verify_rgb))
                    if int(args.panorama_subsample_frames) < len(verify_rgb):
                        step = max(1, len(verify_rgb) // int(args.panorama_subsample_frames))
                        verify_rgb = [verify_rgb[i] for i in range(0, len(verify_rgb), step)][: int(args.panorama_subsample_frames)]
                    pano = stitch_panorama(verify_rgb)
                    pano_path = None
                    if pano_dir is not None:
                        pano_path_obj = pano_dir / f"vfv_dec_{int(decision_num):03d}.jpg"
                        _save_rgb_jpg(pano, pano_path_obj)
                        pano_path = str(pano_path_obj)
                    verify = verify_description_visible(
                        description=sentence,
                        image_path=str(pano_path) if pano_path is not None else "",
                        vlm_model=args.vlm_model,
                        target_desc=decomp.target_desc,
                        anchor_hints=decomp.anchor_descs,
                        max_parse_attempts=int(args.vfv_verify_parse_attempts),
                    ) if pano_path is not None else {"visible": False, "reason": "no_output_dir"}

                    vfv_log = {
                        "phase": "phase1",
                        "triggered": bool(not verify.get("visible", False)),
                        "verify_skipped": False,
                        "phase1_target": phase1_pos.tolist(),
                        "verify": verify,
                        "panorama_path": pano_path,
                        "decompose": {
                            "target_desc": decomp.target_desc,
                            "anchor_desc": decomp.anchor_desc,
                            "anchor_descs": list(decomp.anchor_descs),
                        },
                    }

                    if not bool(verify.get("visible", False)):
                        anchor_info = select_best_anchor_object(
                            anchor_descs=decomp.anchor_descs,
                            query_fn=query_fn,
                            rep=pq3d.representation_manager,
                            top_k=int(args.anchor_top_k),
                        )
                        vfv_log["phase"] = "phase2"
                        vfv_log["anchor_query"] = anchor_info
                        if bool(anchor_info.get("ok", False)):
                            used_target = np.asarray(anchor_info["anchor_position"], dtype=float).reshape(3).copy()
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
                        else:
                            used_target = phase1_pos.copy()
                    else:
                        used_target = phase1_pos.copy()

                if out_task is not None:
                    with open(out_task / f"dec_{decision_num:03d}_vfv.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "task_id": int(loop_tid),
                                "decision_num": int(decision_num),
                                "target_used": used_target.tolist(),
                                "vfv": vfv_log,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                task_end_reason = "final_decision"
                decision_num += 1
                break

            visited_frontier.add(tuple(np.round(used_target, 1)))
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

            if out_task is not None:
                with open(out_task / f"dec_{decision_num:03d}_vfv.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": int(loop_tid),
                            "decision_num": int(decision_num),
                            "is_final": False,
                            "target_used": used_target.tolist(),
                            "vfv": vfv_log,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            decision_num += 1

        agent_state = agent.get_state()
        view_points = [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]
        if len(view_points) > 0:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = sub_episode_start_position
            path.requested_ends = view_points
            start_end_geo_distance = float(path.geodesic_distance) if pf.find_path(path) else float("inf")
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            agent_end_geo_distance = float(path.geodesic_distance) if pf.find_path(path) else float("inf")
        else:
            start_end_geo_distance = float("inf")
            agent_end_geo_distance = float("inf")

        if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
            sr = 0.0
            spl = 0.0
        else:
            sr = 1.0 if agent_end_geo_distance <= float(args.success_distance) else 0.0
            spl = float(sr * start_end_geo_distance / max(start_end_geo_distance, max(episode_cum_distance, 1e-12)))

        print(
            f"[vfv][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} SR={sr:.1f} SPL={spl:.4f}"
        )

    sim.close()
    print("[vfv] done")


if __name__ == "__main__":
    main()

