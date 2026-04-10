from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
from anchor_nav.vlm_decision_corrector import AsyncVLMDecisionCorrector, CorrectorConfig, VLMDecisionCorrector


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


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


def _make_2x2_tile(imgs: List[np.ndarray]) -> np.ndarray:
    assert len(imgs) == 4
    h, w, _ = imgs[0].shape
    out = np.zeros((h * 2, w * 2, 3), dtype=imgs[0].dtype)
    out[0:h, 0:w] = imgs[0]
    out[0:h, w:2 * w] = imgs[1]
    out[h:2 * h, 0:w] = imgs[2]
    out[h:2 * h, w:2 * w] = imgs[3]
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser("Single-sample anchor-vlm analyze")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--description_override", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--enable_output_logs", action="store_true", help="开启落盘日志/图片输出（默认关闭）")

    parser.add_argument("--navigation_data_path", type=str, default="LangMap_Annotations")
    parser.add_argument("--hm3d_data_base_path", type=str, default="datascene")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--pq3d_stage1_path", type=str, default="checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="checkpoint/stage2-fine-tune-goat")

    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--map_resolution", type=int, default=512)
    parser.add_argument("--visible_radius", type=float, default=3.0)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)

    parser.add_argument("--enable_vlm_corrector", action="store_true")
    parser.add_argument("--vlm_stride", type=int, default=1)
    parser.add_argument("--vlm_min_decision_num", type=int, default=2)
    parser.add_argument("--vlm_conf_threshold", type=float, default=0.2)
    parser.add_argument("--vlm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--vlm_mode", choices=["async", "sync"], default="async")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = bool(args.enable_output_logs)

    if output_enabled:
        out_root = _ensure_dir(project_root / "output_process" / f"{run_tag}-anchor-vlm")
        out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={args.task_id}")
        out_dec = _ensure_dir(out_task / "decisions")
        out_logs = _ensure_dir(out_task / "logs")
        trace_path = out_logs / "trace.jsonl"
    else:
        out_task = None
        out_dec = None
        trace_path = None

    print(f"[anchor-vlm] run_tag={run_tag}")
    print(f"[anchor-vlm] output_enabled={output_enabled}")
    print(f"[anchor-vlm] vlm_mode={args.vlm_mode}")

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
    task_type, task_idx = eps["task_sequence"][args.task_id]
    cur_task = episode_mapping[task_type][task_idx]
    sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
    if args.description_override and args.description_override.strip():
        sentence = args.description_override.strip()
    print(f"[anchor-vlm] sentence={sentence}")
    goals_ids = cur_task["target_object_ids"]
    goals = [goals_map[x] for x in goals_ids]

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

    corrector = None
    async_corrector = None
    if args.enable_vlm_corrector:
        corrector = VLMDecisionCorrector(
            CorrectorConfig(
                enabled=True,
                stride=args.vlm_stride,
                min_decision_num=args.vlm_min_decision_num,
                confidence_threshold=args.vlm_conf_threshold,
                base_url=args.vlm_base_url,
                model=args.vlm_model,
            )
        )
        async_corrector = AsyncVLMDecisionCorrector(corrector)

    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=args.map_resolution, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, args.map_resolution, sim)
    vis_dist = convert_meters_to_pixel(args.visible_radius, args.map_resolution, sim)

    view_points = [
        vp["agent_state"]["position"]
        for g in goals
        for vp in g.get("view_points", [])
        if "agent_state" in vp and "position" in vp["agent_state"]
    ]

    def geo_dist(start_pos, ends) -> float:
        if not ends:
            return float("inf")
        sp = habitat_sim.MultiGoalShortestPath()
        sp.requested_start = start_pos
        sp.requested_ends = ends
        if pf.find_path(sp):
            return float(sp.geodesic_distance)
        return float("inf")

    total_steps = 0
    decision_num = 0
    visited_frontier = set()
    goto_rgb: List[np.ndarray] = []
    goto_depth: List[np.ndarray] = []
    goto_state: List[Any] = []
    prev_obj_count = np.asarray(getattr(pq3d.representation_manager, "object_count", np.zeros((0,))), dtype=float)
    prev_agent_state = agent.get_state()
    sub_episode_start_position = prev_agent_state.position
    start_goal_geo = geo_dist(sub_episode_start_position, view_points)
    episode_cum_distance = 0.0
    vlm_force_total = 0
    vlm_calls_total = 0
    task_t0 = _dt.datetime.now().timestamp()

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
        fw = detect_frontier_waypoints(
            top_down_map, fog, area_thr, xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1], enable_visualization=False
        )
        if len(fw) == 0:
            frontiers = []
        else:
            fw = fw[:, ::-1]
            frontiers = pixel_to_map_coors(fw, st_now.position, top_down_map, sim)
        frontiers = [w for w in frontiers if tuple(np.round(w, 1)) not in visited_frontier]

        dec_dir = None
        if output_enabled:
            dec_dir = _ensure_dir(out_dec / f"dec_{decision_num:03d}")
            frames_dir = _ensure_dir(dec_dir / "frames")
            for i, rgb in enumerate(color_list):
                _imwrite_rgb(frames_dir / f"rgb_{i:02d}.jpg", rgb)

        target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence, decision_num)
        baseline_type = "object" if is_final else "frontier"

        rep = pq3d.representation_manager
        obj_boxes = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
        obj_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1)
        obj_counts = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float).reshape(-1)
        cur_n = len(obj_scores)
        prev_n = len(prev_obj_count)
        new_ids = list(range(prev_n, cur_n)) if cur_n > prev_n else []
        updated_ids = [i for i in range(min(prev_n, cur_n)) if obj_counts[i] > prev_obj_count[i]]
        current_pool = sorted(set(new_ids + updated_ids))
        current_pool = sorted(current_pool, key=lambda i: float(obj_scores[i]), reverse=True)[:15]
        current_candidates = [
            {
                "object_id_in_memory": int(i),
                "score": float(obj_scores[i]),
                "count": float(obj_counts[i]),
                "center_xyz": [float(x) for x in obj_boxes[i, :3].tolist()] if i < len(obj_boxes) else [],
            }
            for i in current_pool
        ]

        # build VLM tiles from last 12 pano frames
        tiles = []
        pano = color_list[-12:]
        tmp_tile_dir = None
        if len(pano) == 12:
            if output_enabled:
                tmp_tile_dir = dec_dir
            else:
                tmp_tile_dir = Path(tempfile.mkdtemp(prefix="anchor_vlm_tiles_"))
            for t in range(3):
                tile = _make_2x2_tile(pano[t * 4 : (t + 1) * 4])
                tile_path = tmp_tile_dir / f"vlm_tile_{t}.jpg"
                _imwrite_rgb(tile_path, tile)
                tiles.append(tile_path)

        vlm_info = {"vlm_called": False}
        vlm_async_source_decision = None
        vlm_async_source_target = None
        vlm_async_ready = False
        if async_corrector is not None and args.vlm_mode == "async":
            polled = async_corrector.poll_ready()
            vlm_async_ready = bool(polled.get("ready", False))
            vlm_info = polled.get("vlm_info", {"vlm_called": False})
            vlm_async_source_decision = polled.get("source_decision_num", None)
            vlm_async_source_target = polled.get("source_agent_position", None)
        elif corrector is not None and args.vlm_mode == "sync" and corrector.should_call(decision_num):
            vlm_async_ready = True
            vlm_async_source_decision = int(decision_num)
            vlm_async_source_target = [float(x) for x in np.asarray(st_now.position, dtype=float).tolist()]
            try:
                vlm_info = corrector.evaluate(
                    description=sentence,
                    decision_num=int(decision_num),
                    baseline_target_type=baseline_type,
                    num_frontiers=int(len(frontiers)),
                    memory_objects=int(cur_n),
                    candidate_objects=current_candidates,
                    image_tile_paths=tiles,
                )
            except Exception as e:
                vlm_info = {"vlm_called": True, "error": str(e), "force_object_query": False}
            vlm_calls_total += 1

        corrected = False
        corrected_target = np.asarray(target, dtype=float)
        corrected_final = bool(is_final)
        if vlm_info.get("force_object_query", False):
            src_target = None
            if isinstance(vlm_async_source_target, list) and len(vlm_async_source_target) == 3:
                src_target = np.asarray(vlm_async_source_target, dtype=float)
            if src_target is not None:
                corrected_target = src_target
                corrected_final = True
                corrected = True
                vlm_force_total += 1
            elif len(current_pool) > 0:
                # fallback for robustness if source target was not recorded
                chosen = current_pool[0]
                corrected_target = np.asarray(obj_boxes[chosen, :3], dtype=float)
                corrected_final = True
                corrected = True
                vlm_force_total += 1

        meta = {
            "decision_num": int(decision_num),
            "baseline_target_type": baseline_type,
            "baseline_target_position": np.asarray(target, dtype=float).tolist(),
            "baseline_is_final": bool(is_final),
            "corrected": corrected,
            "corrected_target_position": corrected_target.tolist(),
            "corrected_is_final": corrected_final,
            "num_frontiers": int(len(frontiers)),
            "memory_objects": int(cur_n),
            "new_ids": new_ids,
            "updated_ids": updated_ids,
            "current_top_candidates": current_candidates,
            "vlm_info": vlm_info,
            "vlm_async_ready": bool(vlm_async_ready),
            "vlm_async_source_decision": vlm_async_source_decision,
            "vlm_async_source_agent_position": vlm_async_source_target,
            "start_goal_geo": start_goal_geo,
            "cur_goal_geo_before_goto": geo_dist(agent.get_state().position, view_points),
            "episode_cum_distance": float(episode_cum_distance),
        }
        if output_enabled:
            with open(dec_dir / "decision_meta_anchor_vlm.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        if not corrected_final:
            visited_frontier.add(tuple(np.round(corrected_target, 1)))
        agent_island = pf.get_island(st_now.position)
        target_nav = pf.snap_point(point=corrected_target, island_index=agent_island)
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
            fog[:] = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=fog,
                current_point=map_coors_to_pixel(st2.position, top_down_map, sim),
                current_angle=get_polar_angle(st2),
                fov=42,
                max_line_len=vis_dist,
                enable_debug_visualization=False,
            )
            total_steps += 1
            episode_cum_distance += float(np.linalg.norm(st2.position - prev_agent_state.position))
            prev_agent_state = st2
            if total_steps >= args.max_steps:
                break

        print(
            f"[anchor-vlm] dec={decision_num} frontiers={len(frontiers)} "
            f"baseline={baseline_type} corrected={corrected} final={corrected_final} "
            f"memory={cur_n} current_pool={len(current_pool)} "
            f"vlm_called={vlm_info.get('vlm_called', False)} "
            f"vlm_async_ready={vlm_async_ready} "
            f"vlm_src_dec={vlm_async_source_decision} "
            f"vlm_src_tgt={np.round(np.asarray(vlm_async_source_target, dtype=float), 3).tolist() if isinstance(vlm_async_source_target, list) and len(vlm_async_source_target) == 3 else None} "
            f"vlm_force={vlm_info.get('force_object_query', False)} "
            f"vlm_found={vlm_info.get('found_target', False)} "
            f"vlm_conf={round(float(vlm_info.get('confidence', 0.0)), 3) if vlm_info.get('vlm_called', False) else 0.0} "
            f"vlm_ms={round(float(vlm_info.get('elapsed_ms', 0.0)), 1) if vlm_info.get('vlm_called', False) else 0.0} "
            f"vlm_reason={str(vlm_info.get('reason', ''))[:120]}"
        )

        # schedule VLM for current decision
        scheduled = False
        if async_corrector is not None and args.vlm_mode == "async":
            src_target_xyz = [float(x) for x in np.asarray(st_now.position, dtype=float).tolist()]
            scheduled = async_corrector.submit_if_needed(
                decision_num=int(decision_num),
                description=sentence,
                baseline_target_type=baseline_type,
                num_frontiers=int(len(frontiers)),
                memory_objects=int(cur_n),
                candidate_objects=current_candidates,
                image_tile_paths=tiles,
                source_agent_position=src_target_xyz,
                cleanup_tile_paths=tiles if not output_enabled else [],
                cleanup_dir=tmp_tile_dir if not output_enabled else None,
            )
            if scheduled:
                vlm_calls_total += 1
        if not scheduled and (not output_enabled) and len(tiles) > 0:
            # if this round does not schedule async call, temp tiles can be removed immediately
            for p in tiles:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                if tmp_tile_dir is not None:
                    tmp_tile_dir.rmdir()
            except Exception:
                pass

        prev_obj_count = obj_counts.copy()
        decision_num += 1
        if corrected_final:
            break

    end_state = agent.get_state()
    end_goal_geo = geo_dist(end_state.position, view_points)
    sr = float(np.isfinite(end_goal_geo) and (end_goal_geo <= args.success_distance))
    spl = 0.0
    if np.isfinite(start_goal_geo) and start_goal_geo > 0:
        spl = float(sr * start_goal_geo / max(start_goal_geo, episode_cum_distance))
    summary = {
        "run_tag": run_tag,
        "scene_name": args.scene_name,
        "episode_id": args.episode_id,
        "task_id": args.task_id,
        "steps_total": int(total_steps),
        "decisions": int(decision_num),
        "goal_object_ids": goals_ids,
        "start_goal_geo": start_goal_geo,
        "end_goal_geo": end_goal_geo,
        "sr": sr,
        "spl": spl,
        "episode_cum_distance": float(episode_cum_distance),
        "success_distance": float(args.success_distance),
        "end_position": np.asarray(end_state.position, dtype=float).tolist(),
        "trace_jsonl": str(trace_path) if trace_path is not None else None,
        "enable_vlm_corrector": bool(args.enable_vlm_corrector),
        "enable_output_logs": output_enabled,
        "vlm_mode": args.vlm_mode,
        "vlm_calls_total": int(vlm_calls_total),
        "vlm_force_total": int(vlm_force_total),
        "task_time_sec": float(_dt.datetime.now().timestamp() - task_t0),
    }
    if output_enabled:
        with open(out_task / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[anchor-vlm] summary:", json.dumps(summary, ensure_ascii=False))
    print(
        f"[anchor-vlm] task_time scene={args.scene_name} episode={args.episode_id} "
        f"task={args.task_id} sec={summary['task_time_sec']:.3f}"
    )
    if async_corrector is not None:
        async_corrector.close()
    sim.close()


if __name__ == "__main__":
    main()

