"""
单场景 / 单 episode 调试脚本：仅启用 anchor_nav.rerank（首检 RGB + 描述 → VLM 重选物体目标）。
不含 VLMDecisionCorrector / resolve_navigation_after_vlm。

默认保存调试图：每步全景帧、各记忆槽「首次检测」完整 RGB、送入 rerank 的候选图。
用 --quiet 关闭落盘。

默认使用 detailed 文本描述；需要精简时用 --description_mode concise 或 --concise_description。

送入 VLM 的候选图数量由 --rerank_top_k 控制（默认 8）；另可将按 score 前 N 个首检图导出到 rerank_topN_extra_log/。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 从 MTU3D 根目录运行 `python hm3d-online/本脚本.py` 时，需同时能 import
# `common`（在仓库根）与 `data_utils`（在 hm3d-online）。
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

from anchor_nav.rerank import (
    RerankConfig,
    list_rerank_candidate_memory_ids,
    parse_levels_csv,
    rerank_object_target,
    should_run_rerank,
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


def _save_memory_first_rgb(rep: Any, out_dir: Path) -> None:
    """保存当前所有记忆槽对应的「首次检测」完整 RGB（与 merge_utils.object_first_rgb 对齐）。"""
    d = _ensure_dir(out_dir)
    rgb_list = getattr(rep, "object_first_rgb", None) or []
    for i, img in enumerate(rgb_list):
        if img is None:
            continue
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            continue
        _imwrite_rgb(d / f"mem_{i:03d}_first_detection.jpg", img[:, :, :3])


def _save_rerank_candidates(rep: Any, rerank_cfg: RerankConfig, out_dir: Path) -> List[int]:
    """保存即将送入 VLM 的候选顺序（与 rerank_object_target 一致）。"""
    d = _ensure_dir(out_dir)
    cand = list_rerank_candidate_memory_ids(rep, top_k=rerank_cfg.top_k)
    rgb_list = getattr(rep, "object_first_rgb", None) or []
    for rank, mid in enumerate(cand, start=1):
        if mid >= len(rgb_list) or rgb_list[mid] is None:
            continue
        _imwrite_rgb(d / f"rank_{rank:02d}_mem_{mid:03d}.jpg", rgb_list[mid])
    return cand


def _save_rerank_extra_topn_log(rep: Any, out_dir: Path, n: int) -> None:
    """
    额外日志：按 object_score 取至多 n 个「有首检 RGB」的记忆槽，导出图片 + manifest。
    与送入 VLM 的 top_k 独立，用于核对 score 排序下的前若干张首检图。
    """
    if n <= 0:
        return
    d = _ensure_dir(out_dir)
    cand = list_rerank_candidate_memory_ids(rep, top_k=n)
    rgb_list = getattr(rep, "object_first_rgb", None) or []
    scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1)
    rows: List[Dict[str, Any]] = []
    for rank, mid in enumerate(cand, start=1):
        if mid < len(rgb_list) and rgb_list[mid] is not None and isinstance(rgb_list[mid], np.ndarray):
            _imwrite_rgb(d / f"rank_{rank:02d}_mem_{mid:03d}.jpg", rgb_list[mid][:, :, :3])
        rows.append(
            {
                "rank": rank,
                "memory_index": int(mid),
                "object_score": float(scores[mid]) if mid < len(scores) else None,
                "image_saved": mid < len(rgb_list) and rgb_list[mid] is not None,
            }
        )
    manifest = {
        "sort": "object_score_desc_then_first_rgb_only",
        "top_n": int(n),
        "exported_count": len(cand),
        "candidates": rows,
    }
    (d / "rerank_top_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Anchor-Rerank analyze（仅 rerank 模块）")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, default=0, help="起始 task 下标")
    parser.add_argument("--num_tasks", type=int, default=10, help="连续跑几条子任务")
    parser.add_argument(
        "--description_mode",
        choices=["detailed", "concise"],
        default="detailed",
        help="region/instance 等使用详细或精简标注；默认 detailed",
    )
    parser.add_argument(
        "--concise_description",
        action="store_true",
        help="显式使用精简描述（等同 --description_mode concise）",
    )
    parser.add_argument(
        "--detailed_description",
        action="store_true",
        help="使用详细描述（等同 --description_mode detailed）",
    )
    parser.add_argument("--description_override", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不保存调试图与 trace（默认会保存）",
    )

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

    parser.add_argument("--rerank_levels", type=str, default="instance", help="逗号分隔: object,room,region,instance")
    parser.add_argument("--rerank_top_k", type=int, default=8, help="送入 VLM 的首检图数量上限（默认 8）")
    parser.add_argument(
        "--rerank_extra_log_top_n",
        type=int,
        default=15,
        help="额外导出：按 score 排序的前 N 个有首检 RGB 的图与 manifest，目录名 rerank_top{N}_extra_log（默认 15）",
    )
    parser.add_argument("--rerank_min_rgb_cand", type=int, default=2)
    parser.add_argument("--vlm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="Qwen2.5-VL-32B-Instruct")
    args = parser.parse_args()
    if args.concise_description and args.detailed_description:
        parser.error("不要同时指定 --concise_description 与 --detailed_description")
    if args.concise_description:
        args.description_mode = "concise"
    if args.detailed_description:
        args.description_mode = "detailed"

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)

    rerank_cfg = RerankConfig(
        enabled_levels=parse_levels_csv(args.rerank_levels),
        top_k=int(args.rerank_top_k),
        min_candidates_with_rgb=int(args.rerank_min_rgb_cand),
        base_url=args.vlm_base_url,
        model=args.vlm_model,
    )
    extra_log_n = max(0, int(args.rerank_extra_log_top_n))

    print(f"[anchor-rerank] run_tag={run_tag}")
    print(f"[anchor-rerank] output_enabled={output_enabled} rerank_levels={rerank_cfg.enabled_levels}")
    print(f"[anchor-rerank] description_mode={args.description_mode}")
    print(f"[anchor-rerank] rerank_top_k={args.rerank_top_k} extra_log_top_n={extra_log_n}")
    print(f"[anchor-rerank] task_id={args.task_id} num_tasks={args.num_tasks}")

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
        out_root = _ensure_dir(project_root / "output_process" / f"{run_tag}-anchor-rerank")
        _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}")
    else:
        out_root = None

    all_summaries: List[Dict[str, Any]] = []

    for loop_tid in range(int(args.task_id), task_end):
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(
            task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise")
        )
        if args.description_override and args.description_override.strip():
            sentence = args.description_override.strip()
        print(f"[anchor-rerank] --- task {loop_tid} level={task_type} sentence={sentence[:120]!r} ---")

        goals_ids = cur_task["target_object_ids"]
        goals = [goals_map[x] for x in goals_ids]

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
            os.environ["RERANK_LOG_JSONL"] = str(out_logs / "rerank_vlm.jsonl")
        else:
            out_task = None
            out_dec = None
            trace_path = None
            if "RERANK_LOG_JSONL" in os.environ:
                del os.environ["RERANK_LOG_JSONL"]

        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        prev_obj_count = np.asarray(getattr(pq3d.representation_manager, "object_count", np.zeros((0,))), dtype=float)
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        start_goal_geo = geo_dist(sub_episode_start_position, view_points)
        episode_cum_distance = 0.0
        rerank_attempts = 0
        rerank_applied = 0
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
            target_before_rerank = np.asarray(target, dtype=float).reshape(3).copy()

            rep = pq3d.representation_manager
            obj_boxes = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
            obj_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1)
            obj_counts = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float).reshape(-1)
            cur_n = len(obj_scores)

            rinf: Dict[str, Any] = {}
            if is_final and should_run_rerank(task_type, rerank_cfg) and output_enabled and dec_dir is not None:
                _save_memory_first_rgb(rep, dec_dir / "memory_first_rgb_all_slots")
                _save_rerank_candidates(rep, rerank_cfg, dec_dir / "rerank_candidates_to_vlm")
                if extra_log_n > 0:
                    _save_rerank_extra_topn_log(
                        rep,
                        dec_dir / f"rerank_top{extra_log_n}_extra_log",
                        extra_log_n,
                    )

            if is_final and should_run_rerank(task_type, rerank_cfg):
                rerank_attempts += 1
                new_tp, rinf = rerank_object_target(
                    description=sentence,
                    rep=rep,
                    baseline_target_xyz=target_before_rerank,
                    decision_aux=getattr(pq3d, "last_decision_aux", {}),
                    cfg=rerank_cfg,
                )
                if rinf.get("rerank_applied"):
                    target = new_tp
                    rerank_applied += 1
                    print(
                        f"[anchor-rerank] rerank applied mem={rinf.get('chosen_memory_index')} "
                        f"baseline_mem={rinf.get('baseline_memory_index')} reason={str(rinf.get('reason', ''))[:100]!r} "
                        f"artifact_dir={rinf.get('artifact_dir')}"
                    )
            else:
                rinf = {"skipped": "not_object_final_or_level"}

            meta = {
                "decision_num": int(decision_num),
                "baseline_target_type": baseline_type,
                "baseline_target_before_rerank": target_before_rerank.tolist(),
                "target_position_used_for_goto": np.asarray(target, dtype=float).tolist(),
                "baseline_is_final": bool(is_final),
                "num_frontiers": int(len(frontiers)),
                "memory_objects": int(cur_n),
                "object_scores_snapshot": obj_scores.tolist() if cur_n else [],
                "rerank_info": rinf,
                "last_decision_aux": getattr(pq3d, "last_decision_aux", {}),
                "start_goal_geo": start_goal_geo,
                "cur_goal_geo_before_goto": geo_dist(agent.get_state().position, view_points),
                "episode_cum_distance": float(episode_cum_distance),
            }
            if output_enabled and dec_dir is not None:
                with open(dec_dir / "decision_meta_anchor_rerank.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                if trace_path is not None:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            corrected_final = bool(is_final)
            corrected_target = np.asarray(target, dtype=float)

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
                f"[anchor-rerank] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline={baseline_type} final={corrected_final} memory={cur_n} "
                f"rerank_applied={rinf.get('rerank_applied', False)}"
            )

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
            "rerank_levels": list(rerank_cfg.enabled_levels),
            "rerank_attempts": int(rerank_attempts),
            "rerank_applied": int(rerank_applied),
            "task_time_sec": float(_dt.datetime.now().timestamp() - task_t0),
        }
        all_summaries.append(summary)
        if output_enabled and out_task is not None:
            with open(out_task / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[anchor-rerank] summary:", json.dumps(summary, ensure_ascii=False))

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
        print(f"[anchor-rerank] run_summary -> {run_summary_path}")

    sim.close()


if __name__ == "__main__":
    main()
