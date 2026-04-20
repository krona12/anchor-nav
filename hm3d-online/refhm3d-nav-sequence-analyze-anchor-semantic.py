from __future__ import annotations

import sys
from pathlib import Path

_HM3D_ONLINE = Path(__file__).resolve().parent
_MTU3D_ROOT = _HM3D_ONLINE.parent
for _p in (_HM3D_ONLINE, _MTU3D_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse
import datetime as _dt
import gzip
import json
import os
from typing import Any, Dict, List

import cv2
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

from anchor_nav.semantic_enhance import (
    SemanticEnhanceConfig,
    parse_levels_csv,
    semantic_enhance_object_target,
    should_run_semantic_enhance,
)
from anchor_nav.validate import ValidateConfig, validate_after_arrival
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


def main() -> None:
    parser = argparse.ArgumentParser("Anchor-Semantic analyze（is_final 后 semantic-enhance）")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument("--detailed_description", action="store_true")
    parser.add_argument("--description_override", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

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

    parser.add_argument("--semantic_levels", type=str, default="instance")
    parser.add_argument("--semantic_top_k", type=int, default=8)
    parser.add_argument("--semantic_top_m", type=int, default=5)
    parser.add_argument("--semantic_prob_temperature", type=float, default=0.07)
    parser.add_argument("--semantic_vlm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_clip_model_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--semantic_clip_device", type=str, default="cuda")
    parser.add_argument("--semantic_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
    parser.add_argument("--disable_validate", action="store_true")
    parser.add_argument("--validate_vlm_model", type=str, default="")
    parser.add_argument("--validate_num_views", type=int, default=12)
    parser.add_argument("--validate_group_size", type=int, default=4)
    parser.add_argument("--validate_max_tokens", type=int, default=128)
    parser.add_argument("--validate_max_distance_cm", type=float, default=50.0)
    args = parser.parse_args()

    if args.concise_description and args.detailed_description:
        parser.error("不要同时指定 --concise_description 与 --detailed_description")
    if args.concise_description:
        args.description_mode = "concise"
    if args.detailed_description:
        args.description_mode = "detailed"

    if args.semantic_api_key:
        os.environ["ZZZ_API_KEY"] = args.semantic_api_key

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)

    sem_cfg = SemanticEnhanceConfig(
        enabled_levels=parse_levels_csv(args.semantic_levels),
        top_k=int(args.semantic_top_k),
        top_m=int(args.semantic_top_m),
        prob_temperature=float(args.semantic_prob_temperature),
        clip_model_path=str(args.semantic_clip_model_path),
        clip_device=str(args.semantic_clip_device),
        vlm_model=str(args.semantic_vlm_model),
        override_final_on_apply=False,
    )
    validate_cfg = ValidateConfig(
        enabled=(not bool(args.disable_validate)),
        vlm_model=(str(args.validate_vlm_model).strip() or str(args.semantic_vlm_model)),
        num_views=int(args.validate_num_views),
        group_size=int(args.validate_group_size),
        max_tokens=int(args.validate_max_tokens),
        max_distance_cm=float(args.validate_max_distance_cm),
    )
    print(f"[anchor-semantic] run_tag={run_tag}")
    print(
        f"[anchor-semantic] semantic_levels={sem_cfg.enabled_levels} top_k={sem_cfg.top_k} "
        f"top_m={sem_cfg.top_m} temp={sem_cfg.prob_temperature} vlm_model={sem_cfg.vlm_model}"
    )
    print(f"[anchor-semantic] semantic_override_final={sem_cfg.override_final_on_apply}")
    print(
        f"[anchor-semantic] validate_enabled={validate_cfg.enabled} "
        f"validate_model={validate_cfg.vlm_model} views={validate_cfg.num_views} group={validate_cfg.group_size} "
        f"max_distance_cm={validate_cfg.max_distance_cm}"
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
    if int(args.task_id) >= len(task_sequence):
        raise ValueError(f"task_id {args.task_id} >= len(task_sequence)={len(task_sequence)}")

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

    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=args.map_resolution, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, args.map_resolution, sim)
    vis_dist = convert_meters_to_pixel(args.visible_radius, args.map_resolution, sim)

    visited_frontier: set = set()
    decision_num = 0
    total_steps = 0

    if output_enabled:
        out_root = _ensure_dir(project_root / "output_process" / f"{run_tag}-anchor-semantic")
        _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}")
    else:
        out_root = None

    all_summaries: List[Dict[str, Any]] = []

    for loop_tid in range(int(args.task_id), task_end):
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
        if args.description_override and args.description_override.strip():
            sentence = args.description_override.strip()
        print(f"[anchor-semantic] --- task {loop_tid} level={task_type} sentence={sentence[:120]!r} ---")

        goals_ids = cur_task["target_object_ids"]
        goals = [goals_map[x] for x in goals_ids]
        goal_category = str(cur_task.get("object_category") or goals[0].get("object_category") or "")
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

        if output_enabled:
            out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
            out_dec = _ensure_dir(out_task / "decisions")
            out_logs = _ensure_dir(out_task / "logs")
            trace_path = out_logs / "trace.jsonl"
            os.environ["SEMANTIC_ENHANCE_LOG_JSONL"] = str(out_logs / "semantic_enhance_vlm.jsonl")
        else:
            out_task = None
            out_dec = None
            trace_path = None
            if "SEMANTIC_ENHANCE_LOG_JSONL" in os.environ:
                del os.environ["SEMANTIC_ENHANCE_LOG_JSONL"]

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        start_goal_geo = geo_dist(sub_episode_start_position, view_points)
        episode_cum_distance = 0.0
        semantic_attempts = 0
        semantic_applied = 0
        vlm_elapsed_ms_total = 0.0
        vlm_elapsed_ms_count = 0
        validate_attempts = 0
        validate_passed = 0
        validate_failed = 0
        validation_retry_used = False
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
            if output_enabled and out_dec is not None:
                dec_dir = _ensure_dir(out_dec / f"dec_{decision_num:03d}")
                frames_dir = _ensure_dir(dec_dir / "frames")
                for i, rgb in enumerate(color_list):
                    _imwrite_rgb(frames_dir / f"rgb_{i:02d}.jpg", rgb)
                os.environ["SEMANTIC_ENHANCE_IO_DIR"] = str(_ensure_dir(dec_dir / "semantic_enhance_io"))
            elif "SEMANTIC_ENHANCE_IO_DIR" in os.environ:
                del os.environ["SEMANTIC_ENHANCE_IO_DIR"]

            target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence, decision_num)
            baseline_type = "object" if is_final else "frontier"
            target_before_sem = np.asarray(target, dtype=float).reshape(3).copy()

            sinfo: Dict[str, Any] = {"skipped": "deferred_until_validate_fail"}

            meta = {
                "decision_num": int(decision_num),
                "baseline_target_type": baseline_type,
                "baseline_target_before_semantic": target_before_sem.tolist(),
                "target_position_used_for_goto": np.asarray(target, dtype=float).tolist(),
                "baseline_is_final": bool(is_final),
                "num_frontiers": int(len(frontiers)),
                "semantic_info": sinfo,
                "last_decision_aux": getattr(pq3d, "last_decision_aux", {}),
                "start_goal_geo": start_goal_geo,
                "cur_goal_geo_before_goto": geo_dist(agent.get_state().position, view_points),
                "episode_cum_distance": float(episode_cum_distance),
            }
            if output_enabled and dec_dir is not None:
                with open(dec_dir / "decision_meta_anchor_semantic.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                if trace_path is not None:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            corrected_target = np.asarray(target, dtype=float)
            corrected_final = bool(is_final)
            if not corrected_final:
                visited_frontier.add(tuple(np.round(corrected_target, 1)))
            agent_island = pf.get_island(st_now.position)
            target_nav = pf.snap_point(point=corrected_target, island_index=agent_island)
            follower = habitat_sim.GreedyGeodesicFollower(
                pf, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
            )
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
                f"[anchor-semantic] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline={baseline_type} final={corrected_final} semantic_applied={sinfo.get('semantic_applied', False)}"
            )
            if corrected_final and validate_cfg.enabled and (not validation_retry_used):
                validate_attempts += 1
                validate_io_dir = None
                if output_enabled and out_logs is not None:
                    validate_io_dir = _ensure_dir(out_logs / "validate_io")
                validate_info = validate_after_arrival(
                    sim=sim,
                    description=sentence,
                    cfg=validate_cfg,
                    io_dir=validate_io_dir,
                    jsonl_log_path=(str(out_logs / "validate_vlm.jsonl") if (output_enabled and out_logs is not None) else ""),
                )
                total_steps += int(validate_info.get("num_steps", 0))
                if bool(validate_info.get("contains_target", False)):
                    validate_passed += 1
                    print(
                        f"[anchor-semantic] validate-pass task={loop_tid} dec={decision_num} "
                        f"target={goal_category!r} elapsed_ms={validate_info.get('elapsed_ms')}"
                    )
                else:
                    validate_failed += 1
                    validation_retry_used = True
                    corrected_final = False
                    retry_target = np.asarray(target_before_sem, dtype=float).reshape(3).copy()
                    if should_run_semantic_enhance(task_type, sem_cfg):
                        semantic_attempts += 1
                        retry_target, sinfo = semantic_enhance_object_target(
                            description=sentence,
                            rep=pq3d.representation_manager,
                            baseline_target_xyz=target_before_sem,
                            decision_aux=getattr(pq3d, "last_decision_aux", {}),
                            cfg=sem_cfg,
                        )
                        if sinfo.get("semantic_applied"):
                            semantic_applied += 1
                        if sinfo.get("elapsed_ms") is not None:
                            vlm_elapsed_ms_total += float(sinfo["elapsed_ms"])
                            vlm_elapsed_ms_count += 1
                    else:
                        sinfo = {"skipped": "task_level_not_enabled"}
                    print(
                        f"[anchor-semantic] validate-fail -> semantic retry task={loop_tid} dec={decision_num} "
                        f"target={goal_category!r} elapsed_ms={validate_info.get('elapsed_ms')}"
                    )
                    # 失败后立刻执行一次 semantic 目标导航；本次不再 target 确认
                    st_retry = agent.get_state()
                    retry_island = pf.get_island(st_retry.position)
                    retry_nav = pf.snap_point(point=np.asarray(retry_target, dtype=float), island_index=retry_island)
                    retry_follower = habitat_sim.GreedyGeodesicFollower(
                        pf, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
                    )
                    try:
                        retry_actions = retry_follower.find_path(retry_nav)
                    except Exception:
                        retry_actions = []
                    goto_rgb, goto_depth, goto_state = [], [], []
                    for a_retry in retry_actions:
                        if not a_retry:
                            continue
                        obs_retry = sim.step(action=a_retry)
                        st_retry2 = agent.get_state()
                        goto_rgb.append(obs_retry["color_sensor"][:, :, :3])
                        goto_depth.append(obs_retry["depth_sensor"][:, :])
                        goto_state.append(st_retry2)
                        fog[:] = reveal_fog_of_war(
                            top_down_map=top_down_map,
                            current_fog_of_war_mask=fog,
                            current_point=map_coors_to_pixel(st_retry2.position, top_down_map, sim),
                            current_angle=get_polar_angle(st_retry2),
                            fov=42,
                            max_line_len=vis_dist,
                            enable_debug_visualization=False,
                        )
                        total_steps += 1
                        episode_cum_distance += float(np.linalg.norm(st_retry2.position - prev_agent_state.position))
                        prev_agent_state = st_retry2
                        if total_steps >= args.max_steps:
                            break
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
            "task_id": loop_tid,
            "task_level": task_type,
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
            "semantic_levels": list(sem_cfg.enabled_levels),
            "semantic_attempts": int(semantic_attempts),
            "semantic_applied": int(semantic_applied),
            "vlm_elapsed_ms_total": float(vlm_elapsed_ms_total),
            "vlm_elapsed_ms_avg": (float(vlm_elapsed_ms_total / vlm_elapsed_ms_count) if vlm_elapsed_ms_count > 0 else None),
            "vlm_elapsed_count": int(vlm_elapsed_ms_count),
            "validate_enabled": bool(validate_cfg.enabled),
            "validate_attempts": int(validate_attempts),
            "validate_passed": int(validate_passed),
            "validate_failed": int(validate_failed),
            "validation_retry_used": bool(validation_retry_used),
            "task_time_sec": float(_dt.datetime.now().timestamp() - task_t0),
        }
        all_summaries.append(summary)
        if output_enabled and out_task is not None:
            with open(out_task / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[anchor-semantic] summary:", json.dumps(summary, ensure_ascii=False))

    if output_enabled and out_root is not None:
        run_summary_path = out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / "run_summary.json"
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_tag": run_tag,
                    "scene_name": args.scene_name,
                    "episode_id": args.episode_id,
                    "tasks_ran": len(all_summaries),
                    "summaries": all_summaries,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[anchor-semantic] run_summary -> {run_summary_path}")

    sim.close()


if __name__ == "__main__":
    main()
