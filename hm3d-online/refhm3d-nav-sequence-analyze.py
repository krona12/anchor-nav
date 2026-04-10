"""
Analyze a single RefHM3D LangMap sequence task with rich artifacts.

This script is based on `hm3d-online/refhm3d-nav-sequence.py`, but:
- runs a single (scene, episode_id, task_id)
- writes a structured JSONL trace + verbose log file
- saves images: RGB/depth frames, topdown map, fog-of-war, frontier markers
- dumps PQ3D merge/memory statistics if available (from PQ3DModel.last_decision_stats)

Outputs are written under `output_process/<run_tag>/...`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

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


class TeeStdout(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        n = 0
        for st in self.streams:
            try:
                n = st.write(s)
                st.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:  # type: ignore[override]
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    # rgb: HxWx3 uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _imwrite_depth(path: Path, depth: np.ndarray, max_m: float = 10.0) -> None:
    # depth: HxW float (meters)
    d = np.clip(depth, 0.0, max_m)
    d = (255.0 * (d / max_m)).astype(np.uint8)
    cv2.imwrite(str(path), d)


def _render_topdown(
    top_down_map: np.ndarray,
    fog_of_war_mask: np.ndarray,
    agent_pos: np.ndarray,
    agent_rot,
    sim,
    frontier_waypoints: List[np.ndarray],
    target_position: Optional[np.ndarray],
) -> np.ndarray:
    # Base map -> color
    base = maps.colorize_topdown_map(top_down_map)
    out = base.copy()

    # Fog overlay: explored=1 (or >0). We dim unexplored to emphasize frontiers.
    fog = (fog_of_war_mask > 0).astype(np.uint8)
    unexplored = (fog == 0)
    out[unexplored] = (out[unexplored] * 0.35).astype(out.dtype)

    # Agent marker
    apix = map_coors_to_pixel(agent_pos, top_down_map, sim)  # (row, col)
    cv2.circle(out, (int(apix[1]), int(apix[0])), 4, (255, 255, 255), -1)

    # Frontiers
    for wp in frontier_waypoints or []:
        pp = map_coors_to_pixel(wp, top_down_map, sim)
        cv2.circle(out, (int(pp[1]), int(pp[0])), 2, (0, 255, 255), -1)

    # Target (if any)
    if target_position is not None and np.all(np.isfinite(target_position)):
        tp = map_coors_to_pixel(target_position, top_down_map, sim)
        cv2.circle(out, (int(tp[1]), int(tp[0])), 4, (0, 0, 255), -1)

    return out


def _resolve_scene_mesh(scene_root: Path, scene_name: str) -> Path:
    clean_scene_id = scene_name.split("-")[-1]
    candidates = [
        scene_root / scene_name / f"{clean_scene_id}.basis.glb",
        scene_root / scene_name / f"{clean_scene_id}.glb",
        scene_root / scene_name / f"{clean_scene_id}.basis.scene_instance.json",
        scene_root / scene_name / f"{clean_scene_id}.scene_instance.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: allow non-basis naming in older script layout
    fallback = scene_root / scene_name / f"{clean_scene_id}.basis.glb"
    return fallback


def _feature_summary(vec: np.ndarray, keep_dims: int = 8) -> Dict[str, Any]:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"dim": 0, "norm_l2": 0.0, "head": []}
    return {
        "dim": int(arr.size),
        "norm_l2": float(np.linalg.norm(arr)),
        "head": [float(x) for x in arr[:keep_dims]],
    }


def main() -> None:
    parser = argparse.ArgumentParser("Analyze one scene/episode/task with artifacts")
    parser.add_argument("--scene_name", type=str, required=True, help="e.g. 00800-TEEsavR23oF")
    parser.add_argument("--episode_id", type=int, required=True, help="0..19 within the scene file")
    parser.add_argument("--task_id", type=int, required=True, help="0..4 within a sequence episode")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument(
        "--description_mode",
        type=str,
        choices=["detailed", "concise"],
        default="detailed",
        help="严格使用数据集里的描述字段：detailed 或 concise。",
    )
    parser.add_argument(
        "--description_override",
        type=str,
        default=None,
        help="手动覆盖任务描述文本；设置后将直接替换 sentence（用于对照实验）。",
    )

    # dataset / assets (defaults aligned to your current repo layout, but overridable)
    parser.add_argument("--navigation_data_path", type=str, default="LangMap_Annotations")
    parser.add_argument("--hm3d_data_base_path", type=str, default="datascene")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--pq3d_stage1_path", type=str, default="checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="checkpoint/stage2-fine-tune-goat")

    # rollout controls
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--visible_radius", type=float, default=3.0)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--map_resolution", type=int, default=512)
    parser.add_argument("--success_distance", type=float, default=0.25)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    out_root = _ensure_dir(project_root / "output_process" / run_tag)
    out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={args.task_id}")
    out_logs = _ensure_dir(out_task / "logs")
    out_decisions = _ensure_dir(out_task / "decisions")
    out_maps = _ensure_dir(out_task / "maps")

    # tee stdout to file
    log_path = out_logs / "run.log"
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = TeeStdout(sys.__stdout__, log_f)  # type: ignore[assignment]
    sys.stderr = TeeStdout(sys.__stderr__, log_f)  # type: ignore[assignment]

    print(f"[analyze] run_tag={run_tag}")
    print(f"[analyze] output_dir={out_task}")
    print(f"[analyze] log_path={log_path}")

    navigation_data_path = (project_root / args.navigation_data_path).resolve()
    hm3d_data_base_path = (project_root / args.hm3d_data_base_path).resolve()
    scene_file = navigation_data_path / f"{args.scene_name}.json.gz"
    if not scene_file.exists():
        raise FileNotFoundError(f"scene annotation not found: {scene_file}")

    # load scene annotations
    with gzip.open(scene_file, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    region_to_annot_dict = scene_data["region_annotation"]
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    # pick episode
    all_eps = scene_data["episode_by_sequence"]
    chosen_ep = None
    for ep in all_eps:
        if int(ep["episode_id"]) == int(args.episode_id):
            chosen_ep = ep
            break
    if chosen_ep is None:
        raise ValueError(f"episode_id={args.episode_id} not found in {scene_file} (have {[e['episode_id'] for e in all_eps]})")

    # pick task
    if not (0 <= args.task_id < len(chosen_ep["task_sequence"])):
        raise ValueError(f"task_id out of range: {args.task_id}, len(task_sequence)={len(chosen_ep['task_sequence'])}")
    task_type, task_idx = chosen_ep["task_sequence"][args.task_id]
    cur_task = episode_mapping[task_type][task_idx]

    goals_ids = cur_task["target_object_ids"]
    goals = [all_navigation_goals_dict[x] for x in goals_ids]
    description_field = "detailed"
    if task_type == "object":
        sentence = cur_task["object_category"]
        goal_category = cur_task["object_category"]
        description_field = "object_category"
    elif task_type == "room":
        sentence = f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
        goal_category = cur_task["object_category"]
        description_field = "object_category + room_name"
    elif task_type == "region":
        region_info = region_to_annot_dict[cur_task["region_id"]]
        if args.description_mode == "concise":
            region_desc = region_info.get("concise_description") or ""
            description_field = "region_annotation.concise_description"
        else:
            # 优先使用 detailed_description；如果缺失再退化到 comprehensive
            region_desc = (
                region_info.get("detailed_description")
                or region_info.get("comprehensive_description")
                or ""
            )
            description_field = "region_annotation.detailed_description"
        sentence = (
            f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {region_desc}"
        )
        goal_category = cur_task["object_category"]
    elif task_type == "instance":
        inst = all_navigation_goals_dict[cur_task["instance_id"]]
        if args.description_mode == "concise":
            sentence = inst.get("annot_unique_concise_description") or ""
            description_field = "goals.annot_unique_concise_description"
        else:
            sentence = (
                inst.get("annot_unique_detailed_description")
                or inst.get("annot_unique_normal_description")
                or ""
            )
            description_field = "goals.annot_unique_detailed_description"
        goal_category = goals[0]["object_category"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # manual prompt override for controlled ablation
    if args.description_override is not None and args.description_override.strip():
        sentence = args.description_override.strip()
        description_field = "manual_override"

    print(f"[analyze] scene={args.scene_name} episode_id={args.episode_id} task_id={args.task_id}")
    print(f"[analyze] task_type={task_type} goal_category={goal_category}")
    print(f"[analyze] description_mode={args.description_mode} field={description_field}")
    if description_field == "manual_override":
        print("[analyze] description_override is enabled")
    print(f"[analyze] sentence={sentence}")
    print(f"[analyze] goal_object_ids={goals_ids}")

    # init simulator
    sim_settings = OmegaConf.load(str((project_root / args.sim_config).resolve()))
    goat_agent_setting = OmegaConf.load(str((project_root / args.agent_config).resolve()))
    sim_settings["scene"] = str(_resolve_scene_mesh(hm3d_data_base_path, args.scene_name))
    abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    path_finder = sim.pathfinder

    # init agent at episode start
    agent_state = habitat_sim.AgentState()
    agent_state.position = chosen_ep["start_position"]
    agent_state.rotation = chosen_ep["start_rotation"]
    agent.set_state(agent_state)

    # maps
    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=args.map_resolution, draw_border=False)
    fog_of_war_mask = np.zeros_like(top_down_map)
    area_thres_in_pixels = convert_meters_to_pixel(9, args.map_resolution, sim)
    visibility_dist_in_pixels = convert_meters_to_pixel(args.visible_radius, args.map_resolution, sim)

    # model
    pq3d_model = PQ3DModel(
        str((project_root / args.pq3d_stage1_path).resolve()),
        str((project_root / args.pq3d_stage2_path).resolve()),
        min_decision_num=args.decision_num_min,
    )
    pq3d_model.reset()

    # GT viewpoints for success + distances
    view_points = [
        vp["agent_state"]["position"]
        for g in goals
        for vp in g.get("view_points", [])
        if "agent_state" in vp and "position" in vp["agent_state"]
    ]

    def geo_dist(a, ends) -> float:
        if not ends:
            return float("inf")
        sp = habitat_sim.MultiGoalShortestPath()
        sp.requested_start = a
        sp.requested_ends = ends
        if path_finder.find_path(sp):
            return float(sp.geodesic_distance)
        return float("inf")

    trace_path = out_logs / "trace.jsonl"
    print(f"[analyze] trace_jsonl={trace_path}")

    # rollout loop (single task)
    total_steps = 0
    rotation_steps = 0
    decision_num = 0
    visited_frontier_set = set()
    episode_cum_distance = 0.0
    prev_agent_state = agent.get_state()
    sub_episode_start_position = prev_agent_state.position
    start_goal_geo = geo_dist(sub_episode_start_position, view_points)
    print(f"[analyze] start_position={np.asarray(sub_episode_start_position).round(4).tolist()} start_goal_geo={start_goal_geo:.3f}")

    goto_color_list: List[np.ndarray] = []
    goto_depth_list: List[np.ndarray] = []
    goto_agent_state_list: List[Any] = []

    while total_steps < args.max_steps:
        # build frame batch = (goto frames subsampled) + (12-turn panorama)
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

        action_list = ["turn_left"] * 12
        for action in action_list:
            obs = sim.step(action=action)
            rgb = obs["color_sensor"][:, :, :3]
            dep = obs["depth_sensor"][:, :]
            st = agent.get_state()
            color_list.append(rgb)
            depth_list.append(dep)
            agent_state_list.append(st)
            fog_of_war_mask[:] = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=fog_of_war_mask,
                current_point=map_coors_to_pixel(st.position, top_down_map, sim),
                current_angle=get_polar_angle(st),
                fov=42,
                max_line_len=visibility_dist_in_pixels,
                enable_debug_visualization=False,
            )
            total_steps += 1
            rotation_steps += 1
            if total_steps >= args.max_steps:
                break
        if total_steps >= args.max_steps:
            break

        # frontier detection
        st_now = agent.get_state()
        frontier_waypoints = detect_frontier_waypoints(
            top_down_map,
            fog_of_war_mask,
            area_thres_in_pixels,
            xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1],
            enable_visualization=False,
        )
        if len(frontier_waypoints) == 0:
            frontier_waypoints = []
        else:
            frontier_waypoints = frontier_waypoints[:, ::-1]
            frontier_waypoints = pixel_to_map_coors(frontier_waypoints, st_now.position, top_down_map, sim)
        frontier_waypoints = [wp for wp in frontier_waypoints if tuple(np.round(wp, 1)) not in visited_frontier_set]

        # save pre-decision artifacts
        dec_dir = _ensure_dir(out_decisions / f"dec_{decision_num:03d}")
        frames_dir = _ensure_dir(dec_dir / "frames")
        for i, (rgb, dep) in enumerate(zip(color_list, depth_list)):
            _imwrite_rgb(frames_dir / f"rgb_{i:02d}.jpg", rgb)
            _imwrite_depth(frames_dir / f"depth_{i:02d}.png", dep)
        # topdown/fog/frontier before decision
        td = _render_topdown(
            top_down_map=top_down_map,
            fog_of_war_mask=fog_of_war_mask,
            agent_pos=np.asarray(st_now.position),
            agent_rot=st_now.rotation,
            sim=sim,
            frontier_waypoints=frontier_waypoints,
            target_position=None,
        )
        cv2.imwrite(str(dec_dir / "topdown_pre.png"), cv2.cvtColor(td, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(dec_dir / "fog.png"), (fog_of_war_mask > 0).astype(np.uint8) * 255)

        # decision
        try:
            target_position, is_final_decision = pq3d_model.decision(
                color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num
            )
        except Exception as e:
            print(f"[analyze] decision error: {e}")
            break

        # decision stats from PQ3DModel (if available)
        decision_stats = getattr(pq3d_model, "last_decision_stats", None)
        with open(dec_dir / "decision_stats.json", "w", encoding="utf-8") as f:
            json.dump(decision_stats or {}, f, ensure_ascii=False, indent=2)

        # robust decision meta log: always available even when decision_stats is empty
        frontier_candidates = [np.asarray(wp).astype(float).tolist() for wp in frontier_waypoints]
        with open(dec_dir / "frontier_candidates.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "decision_num": int(decision_num),
                    "num_frontiers": int(len(frontier_candidates)),
                    "frontier_candidates": frontier_candidates,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # top-k object/frontier dump for diagnostics
        rep = pq3d_model.representation_manager
        obj_boxes = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
        obj_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float)
        obj_feats = np.asarray(getattr(rep, "object_feat", np.zeros((0, 0))), dtype=float)
        obj_vocab_feats = np.asarray(getattr(rep, "open_vocab_feat", np.zeros((0, 0))), dtype=float)
        obj_count = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float)

        if obj_scores.ndim == 0:
            obj_scores = obj_scores.reshape(-1)
        order = np.argsort(-obj_scores) if obj_scores.size > 0 else np.array([], dtype=int)
        order = order[:15]
        object_topk = []
        for rank, oid in enumerate(order.tolist()):
            box = obj_boxes[oid] if oid < len(obj_boxes) else np.zeros((6,))
            center = [float(box[0]), float(box[1]), float(box[2])] if box.shape[0] >= 3 else []
            object_topk.append(
                {
                    "rank": int(rank + 1),
                    "object_id_in_memory": int(oid),
                    "score": float(obj_scores[oid]),
                    "count": float(obj_count[oid]) if oid < len(obj_count) else None,
                    "center_xyz": center,
                    "box_xyzwhd": [float(x) for x in box.tolist()] if box.size > 0 else [],
                    "object_feat_summary": _feature_summary(obj_feats[oid]) if oid < len(obj_feats) else {"dim": 0, "norm_l2": 0.0, "head": []},
                    "open_vocab_feat_summary": _feature_summary(obj_vocab_feats[oid]) if oid < len(obj_vocab_feats) else {"dim": 0, "norm_l2": 0.0, "head": []},
                }
            )

        agent_pos_np = np.asarray(st_now.position, dtype=float).reshape(3)
        frontier_topk = []
        for i, fp in enumerate(frontier_candidates[:15]):
            p = np.asarray(fp, dtype=float).reshape(3)
            dist = float(np.linalg.norm(p - agent_pos_np))
            frontier_topk.append(
                {
                    "rank": int(i + 1),
                    "frontier_id_in_round": int(i),
                    "position_xyz": [float(x) for x in p.tolist()],
                    # baseline does not expose explicit frontier confidence here
                    "score": None,
                    "distance_to_agent": dist,
                }
            )

        topk_payload = {
            "decision_num": int(decision_num),
            "object_topk_by_score": object_topk,
            "frontier_topk": frontier_topk,
        }
        with open(dec_dir / "decision_topk.json", "w", encoding="utf-8") as f:
            json.dump(topk_payload, f, ensure_ascii=False, indent=2)

        # mark visited frontier if frontier decision
        if not is_final_decision:
            visited_frontier_set.add(tuple(np.round(target_position, 1)))

        # save post-decision topdown (with target)
        td2 = _render_topdown(
            top_down_map=top_down_map,
            fog_of_war_mask=fog_of_war_mask,
            agent_pos=np.asarray(st_now.position),
            agent_rot=st_now.rotation,
            sim=sim,
            frontier_waypoints=frontier_waypoints,
            target_position=np.asarray(target_position),
        )
        cv2.imwrite(str(dec_dir / "topdown_post.png"), cv2.cvtColor(td2, cv2.COLOR_RGB2BGR))

        # compute distances & log entry
        cur_goal_geo = geo_dist(agent.get_state().position, view_points)
        entry = {
            "run_tag": run_tag,
            "scene_name": args.scene_name,
            "episode_id": int(args.episode_id),
            "task_id": int(args.task_id),
            "task_type": task_type,
            "description_mode": args.description_mode,
            "description_field": description_field,
            "description_override": args.description_override,
            "sentence": sentence,
            "decision_num": int(decision_num),
            "total_steps": int(total_steps),
            "rotation_steps": int(rotation_steps),
            "agent_position": np.asarray(agent.get_state().position).astype(float).tolist(),
            "target_position": np.asarray(target_position).astype(float).tolist(),
            "is_final_decision": bool(is_final_decision),
            "decision_target_type": "object" if bool(is_final_decision) else "frontier",
            "num_frontiers": int(len(frontier_waypoints)) if frontier_waypoints is not None else 0,
            "frontier_candidates": frontier_candidates,
            "visited_frontier_size": int(len(visited_frontier_set)),
            "memory_objects": int(getattr(pq3d_model.representation_manager, "object_box", np.zeros((0, 6))).shape[0]),
            "object_topk_by_score": object_topk,
            "frontier_topk": frontier_topk,
            "start_goal_geo": start_goal_geo,
            "cur_goal_geo": cur_goal_geo,
            "episode_cum_distance": float(episode_cum_distance),
        }
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(
            f"[analyze] dec={decision_num} steps={total_steps} frontiers={len(frontier_waypoints)} "
            f"mem_objs={entry['memory_objects']} final={is_final_decision} "
            f"cur_goal_geo={cur_goal_geo:.3f} target={np.round(target_position,3).tolist()}"
        )

        decision_num += 1

        # follow to target (GT geodesic follower)
        st_now = agent.get_state()
        agent_island = path_finder.get_island(st_now.position)
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
        )
        try:
            goto_actions = follower.find_path(target_on_navmesh)
        except Exception:
            goto_actions = []

        with open(dec_dir / "decision_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "decision_num": int(decision_num),
                    "decision_target_type": "object" if bool(is_final_decision) else "frontier",
                    "is_final_decision": bool(is_final_decision),
                    "target_position": np.asarray(target_position).astype(float).tolist(),
                    "target_on_navmesh": np.asarray(target_on_navmesh).astype(float).tolist(),
                    "agent_position_before_goto": np.asarray(st_now.position).astype(float).tolist(),
                    "num_frontiers": int(len(frontier_candidates)),
                    "frontier_candidates": frontier_candidates,
                    "goto_actions_len": int(len(goto_actions)),
                    "goto_actions": [str(a) for a in goto_actions],
                    "memory_objects": int(getattr(pq3d_model.representation_manager, "object_box", np.zeros((0, 6))).shape[0]),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        goto_color_list = []
        goto_depth_list = []
        goto_agent_state_list = []
        for a in goto_actions:
            if not a:
                continue
            obs = sim.step(action=a)
            st = agent.get_state()
            goto_color_list.append(obs["color_sensor"][:, :, :3])
            goto_depth_list.append(obs["depth_sensor"][:, :])
            goto_agent_state_list.append(st)
            fog_of_war_mask[:] = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=fog_of_war_mask,
                current_point=map_coors_to_pixel(st.position, top_down_map, sim),
                current_angle=get_polar_angle(st),
                fov=42,
                max_line_len=visibility_dist_in_pixels,
                enable_debug_visualization=False,
            )
            total_steps += 1
            if a in ["turn_left", "turn_right"]:
                rotation_steps += 1
            episode_cum_distance += float(np.linalg.norm(st.position - prev_agent_state.position))
            prev_agent_state = st
            if total_steps >= args.max_steps:
                break

        if is_final_decision:
            break

    # final metrics (for this single task)
    end_state = agent.get_state()
    end_goal_geo = geo_dist(end_state.position, view_points)
    sr = float(np.isfinite(end_goal_geo) and (end_goal_geo <= args.success_distance))
    spl = 0.0
    if np.isfinite(start_goal_geo) and start_goal_geo > 0:
        spl = float(sr * start_goal_geo / max(start_goal_geo, episode_cum_distance))

    summary = {
        "run_tag": run_tag,
        "scene_name": args.scene_name,
        "episode_id": int(args.episode_id),
        "task_id": int(args.task_id),
        "task_type": task_type,
        "description_mode": args.description_mode,
        "description_field": description_field,
        "description_override": args.description_override,
        "goal_category": goal_category,
        "sentence": sentence,
        "goal_object_ids": goals_ids,
        "start_position": np.asarray(sub_episode_start_position).astype(float).tolist(),
        "end_position": np.asarray(end_state.position).astype(float).tolist(),
        "start_goal_geo": start_goal_geo,
        "end_goal_geo": end_goal_geo,
        "sr": sr,
        "spl": spl,
        "episode_cum_distance": float(episode_cum_distance),
        "decisions": int(decision_num),
        "steps_total": int(total_steps),
        "trace_jsonl": str(trace_path),
    }
    with open(out_task / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[analyze] summary:", json.dumps(summary, ensure_ascii=False))

    # dump final topdown
    td_final = _render_topdown(
        top_down_map=top_down_map,
        fog_of_war_mask=fog_of_war_mask,
        agent_pos=np.asarray(end_state.position),
        agent_rot=end_state.rotation,
        sim=sim,
        frontier_waypoints=[],
        target_position=None,
    )
    cv2.imwrite(str(out_maps / "topdown_final.png"), cv2.cvtColor(td_final, cv2.COLOR_RGB2BGR))

    sim.close()
    log_f.close()


if __name__ == "__main__":
    main()

