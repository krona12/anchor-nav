from __future__ import annotations

import argparse
import datetime as _dt
import gzip
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

from anchor_nav.tri_query import TriQueryConfig, build_query_fn_from_pq3d_stage2, run_tri_query
from anchor_nav.vote import build_target_anchor_prompt, parse_target_anchors_from_vlm_raw
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
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL, chat


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


def _extract_target_anchor(description: str, model: str) -> Dict[str, str]:
    raw = chat(text=build_target_anchor_prompt(description), image_path=None, model=model, max_tokens=128)
    main_target, anchors, _, relation, anchor_types = parse_target_anchors_from_vlm_raw(
        raw,
        nearby_anchor_vote_weight=1.0,
        secondary_anchor_vote_weight=1.0,
    )
    nearest_anchor = ""
    for i, a in enumerate(anchors):
        if i < len(anchor_types) and str(anchor_types[i]).strip().lower() == "nearby":
            nearest_anchor = str(a).strip()
            break
    if not nearest_anchor and len(anchors) > 0:
        nearest_anchor = str(anchors[0]).strip()
    return {
        "main_target": str(main_target).strip(),
        "nearest_anchor": str(nearest_anchor).strip(),
        "anchors": [str(x) for x in anchors],
        "anchor_types": [str(x) for x in anchor_types],
        "spatial_relation": str(relation),
        "raw": str(raw),
    }


def _find_oracle_object_index(rep: Any, goal_positions: List[np.ndarray]) -> Optional[int]:
    if len(goal_positions) == 0:
        return None
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] < 3:
        return None
    box_nav = np.asarray(box[:, :3], dtype=float).copy()
    box_nav[:, [1, 2]] = box_nav[:, [2, 1]]
    d_all = [np.linalg.norm(box_nav - gp[None, :], axis=1) for gp in goal_positions]
    d_min = np.min(np.stack(d_all, axis=0), axis=0)
    return int(np.argmin(d_min))


def _query_top_hit_rank(topk: List[Dict[str, Any]], obj_idx: int) -> Optional[int]:
    for i, rec in enumerate(topk, start=1):
        if int(rec.get("object_index", -1)) == int(obj_idx):
            return int(i)
    return None


def main() -> None:
    parser = argparse.ArgumentParser("Anchor Tri-Query analyze（仅 tri_query 模块）")
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
    parser.add_argument("--tri_top_k", type=int, default=8)
    parser.add_argument("--tri_sigma_anchor", type=float, default=2.0)
    parser.add_argument("--tri_sigma_full", type=float, default=3.0)
    parser.add_argument("--final_topn_log", type=int, default=5)
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_logs",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)
    tri_cfg = TriQueryConfig(
        top_k=int(args.tri_top_k),
        sigma_anchor=float(args.tri_sigma_anchor),
        sigma_full=float(args.tri_sigma_full),
    )
    print(f"[tri-query] run_tag={run_tag} cfg={tri_cfg}")

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
        out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-anchor-tri-query")

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
        print(f"[tri-query][task-start] task={loop_tid} level={task_type} desc={sentence!r}")
        goals_ids = list(cur_task.get("target_object_ids", []))
        goals = [goals_map[x] for x in goals_ids if x in goals_map]
        goal_positions = [
            np.asarray(g.get("position", []), dtype=float).reshape(3)
            for g in goals
            if isinstance(g, dict) and len(g.get("position", [])) >= 3
        ]

        out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}") if out_root else None

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        episode_cum_distance = 0.0
        task_end_reason = "max_steps"
        tri_used = 0
        tri_attempts = 0

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

            for _ in range(12):
                obs = sim.step(action="turn_left")
                rgb = obs["color_sensor"][:, :, :3]
                dep = obs["depth_sensor"][:, :]
                st_now = agent.get_state()
                color_list.append(rgb)
                depth_list.append(dep)
                state_list.append(st_now)
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
                if total_steps >= args.max_steps:
                    break
            if total_steps >= args.max_steps:
                break

            st_now = agent.get_state()
            fw = detect_frontier_waypoints(top_down_map, fog, area_thr, xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1], enable_visualization=False)
            if len(fw) == 0:
                frontiers = []
            else:
                frontiers = pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim)
            frontiers = [w for w in frontiers if tuple(np.round(w, 1)) not in visited_frontier]

            target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence, decision_num)
            used_target = np.asarray(target, dtype=float).reshape(3).copy()
            print(
                f"[tri-query][decision] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline_target={used_target.tolist()} final={bool(is_final)}"
            )

            tri_log: Dict[str, Any] = {"ok": False}
            if is_final:
                tri_attempts += 1
                t_vlm = time.perf_counter()
                parsed = _extract_target_anchor(sentence, args.vlm_model)
                vlm_ms = (time.perf_counter() - t_vlm) * 1000.0
                print(
                    f"[tri-query][vlm] task={loop_tid} dec={decision_num} elapsed_ms={vlm_ms:.1f} "
                    f"main_target={parsed['main_target']!r} nearest_anchor={parsed['nearest_anchor']!r} "
                    f"anchors={parsed['anchors']} anchor_types={parsed['anchor_types']}"
                )
                tri_log = run_tri_query(
                    description=sentence,
                    rep=pq3d.representation_manager,
                    query_fn=query_fn,
                    main_target=parsed["main_target"],
                    nearest_anchor=parsed["nearest_anchor"],
                    cfg=tri_cfg,
                )
                if tri_log.get("ok"):
                    tri_used += 1
                    used_target = np.asarray(tri_log["target_xyz"], dtype=float).reshape(3)
                    final_topn = list(tri_log.get("final_ranking", []))[: max(1, int(args.final_topn_log))]
                    topn_str = ", ".join(
                        f"rank={i+1}/obj={int(x['object_index'])}/final={float(x['final_score']):.4f}/sem={float(x['semantic_score']):.4f}/spa={float(x['spatial_consensus']):.4f}"
                        for i, x in enumerate(final_topn)
                    )
                    print(f"[tri-query][final-top] task={loop_tid} dec={decision_num} top{len(final_topn)}=[{topn_str}]")

                    oracle_obj_idx = _find_oracle_object_index(pq3d.representation_manager, goal_positions)
                    if oracle_obj_idx is not None:
                        qlogs = tri_log.get("query_logs", {})
                        full_rank = _query_top_hit_rank(list(qlogs.get("full_topk", [])), oracle_obj_idx)
                        target_rank = _query_top_hit_rank(list(qlogs.get("target_topk", [])), oracle_obj_idx)
                        anchor_rank = _query_top_hit_rank(list(qlogs.get("anchor_topk", [])), oracle_obj_idx)
                        hit_query = []
                        if full_rank is not None:
                            hit_query.append(f"full@{full_rank}")
                        if target_rank is not None:
                            hit_query.append(f"target@{target_rank}")
                        if anchor_rank is not None:
                            hit_query.append(f"anchor@{anchor_rank}")
                        print(
                            f"[tri-query][gt-hit] task={loop_tid} dec={decision_num} oracle_obj={oracle_obj_idx} "
                            f"in_queries={len(hit_query)>0} where={hit_query}"
                        )
                        tri_log["gt_hit"] = {
                            "oracle_object_index": int(oracle_obj_idx),
                            "in_any_query_topk": bool(len(hit_query) > 0),
                            "in_full_topk_rank": full_rank,
                            "in_target_topk_rank": target_rank,
                            "in_anchor_topk_rank": anchor_rank,
                            "where": hit_query,
                        }
                else:
                    print(f"[tri-query][run-fail] task={loop_tid} dec={decision_num} reason={tri_log.get('reason')}")

            if not is_final:
                visited_frontier.add(tuple(np.round(used_target, 1)))

            agent_island = pf.get_island(agent.get_state().position)
            target_nav = pf.snap_point(point=used_target, island_index=agent_island)
            follower = habitat_sim.GreedyGeodesicFollower(pf, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
            try:
                actions = follower.find_path(target_nav)
            except Exception:
                actions = []
            goto_rgb, goto_depth, goto_state = [], [], []
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
                if total_steps >= args.max_steps:
                    break

            if out_task is not None:
                with open(out_task / f"dec_{decision_num:03d}_tri_query.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": int(loop_tid),
                            "decision_num": int(decision_num),
                            "is_final": bool(is_final),
                            "target_used": used_target.tolist(),
                            "tri_query": tri_log,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            decision_num += 1
            if is_final:
                task_end_reason = "final_decision"
                break

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
            f"[tri-query][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} tri_attempts={tri_attempts} tri_used={tri_used} "
            f"SR={sr:.1f} SPL={spl:.4f} start_goal_geo={start_end_geo_distance:.3f} end_goal_geo={agent_end_geo_distance:.3f}"
        )

    sim.close()
    print("[tri-query] done")


if __name__ == "__main__":
    main()
