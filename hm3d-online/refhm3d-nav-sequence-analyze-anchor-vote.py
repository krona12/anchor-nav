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
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

from anchor_nav.vote import (
    VoteConfig,
    VoteState,
    build_refined_query_prompt,
    build_target_anchor_prompt,
    parse_refined_query_from_vlm_raw,
    parse_target_anchors_from_vlm_raw,
    pq3d_stage2_object_logits,
    run_position_vote_with_pq3d_stage2,
    update_bindings_for_new_objects,
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
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL, chat


def _pq3d_logit_at(logits: np.ndarray, idx: int) -> float:
    if logits.size == 0 or idx < 0 or idx >= int(logits.shape[0]):
        return float("nan")
    return float(logits[int(idx)])


def _extract_target_anchors(
    description: str, model: str, vote_cfg: VoteConfig
) -> Tuple[str, List[str], List[float], str, List[str]]:
    prompt = build_target_anchor_prompt(description)
    raw = chat(text=prompt, image_path=None, model=model, max_tokens=128)
    return parse_target_anchors_from_vlm_raw(
        raw,
        nearby_anchor_vote_weight=float(vote_cfg.primary_anchor_vote_weight),
        secondary_anchor_vote_weight=float(vote_cfg.other_anchor_vote_weight),
    )


def _extract_refined_navigation_query(description: str, model: str) -> Dict[str, str]:
    raw = chat(text=build_refined_query_prompt(description), image_path=None, model=model, max_tokens=128)
    mt, ka, rq = parse_refined_query_from_vlm_raw(raw)
    return {"main_target": mt, "key_anchor": ka, "refined_query": rq, "raw": raw}


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
    parser = argparse.ArgumentParser("Anchor-Vote analyze（仅 vote 模块）")
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
    parser.add_argument("--vote_query_top_k", type=int, default=5)
    parser.add_argument("--vote_node_top_k", type=int, default=5)
    parser.add_argument("--vote_node_min_dist_m", type=float, default=1.0)
    parser.add_argument("--vote_target_weight", type=float, default=0.05)
    parser.add_argument("--vote_primary_anchor_weight", type=float, default=0.05)
    parser.add_argument("--vote_other_anchor_weight", type=float, default=0.05)
    parser.add_argument(
        "--vote_anchor_aggregate_scale",
        type=float,
        default=1.0,
        help="锚点在 per-anchor 权重上的额外缩放",
    )
    parser.add_argument(
        "--vote_rank_decay_gamma",
        type=float,
        default=0.75,
        help="排名衰减系数 γ：第 k 名 × γ^k；1.0 为旧行为",
    )
    parser.add_argument(
        "--vote_substitute_weight",
        type=float,
        default=0.9,
        help="refined≠main 时 substitute top-k 基础增益（再乘 γ^rank）",
    )
    parser.add_argument(
        "--vote_object_score",
        type=str,
        default="pick",
        choices=("pick", "target", "max"),
        help="胜出节点内选物体：pick=Stage2+refined；target=Stage2+main_target；max=logits 取大",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)
    vote_cfg = VoteConfig(
        node_min_dist_m=float(args.vote_node_min_dist_m),
        query_top_k=int(args.vote_query_top_k),
        node_pick_top_k=int(args.vote_node_top_k),
        softmax_temp=0.07,
        target_vote_weight=float(args.vote_target_weight),
        primary_anchor_vote_weight=float(args.vote_primary_anchor_weight),
        other_anchor_vote_weight=float(args.vote_other_anchor_weight),
        anchor_vote_aggregate_scale=float(args.vote_anchor_aggregate_scale),
        query_rank_decay_gamma=float(args.vote_rank_decay_gamma),
        refined_substitute_vote_weight=float(args.vote_substitute_weight),
        object_pick_score=str(args.vote_object_score),
    )
    print(f"[anchor-vote] run_tag={run_tag} vote_cfg={vote_cfg}")

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

    if output_enabled:
        out_root = _ensure_dir(project_root / "output_process" / f"{run_tag}-anchor-vote")
    else:
        out_root = None

    vote_state = VoteState()
    visited_frontier: set = set()

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        original_sentence = _build_sentence(
            task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise")
        )
        refined_pack: Optional[Dict[str, str]] = None
        sentence_nav = original_sentence
        try:
            refined_pack = _extract_refined_navigation_query(original_sentence, args.vlm_model)
            sentence_nav = str(refined_pack["refined_query"])
            print(
                f"[anchor-vote][refined] task={loop_tid} original={original_sentence!r} "
                f"refined_query={sentence_nav!r} main_target={refined_pack['main_target']!r} "
                f"key_anchor={refined_pack['key_anchor']!r}"
            )
        except Exception as e:
            print(f"[anchor-vote][refined-fail] task={loop_tid} use original desc for PQ3D, err={e!r}")
        print(f"[anchor-vote][task-start] task={loop_tid} level={task_type} desc={original_sentence!r}")
        goals_ids = list(cur_task.get("target_object_ids", []))
        goals = [goals_map[x] for x in goals_ids if x in goals_map]
        goal_category = str(cur_task.get("object_category", goals[0]["object_category"] if len(goals) > 0 else "unknown"))
        goal_positions = [
            np.asarray(g.get("position", []), dtype=float).reshape(3)
            for g in goals
            if isinstance(g, dict) and len(g.get("position", [])) >= 3
        ]
        if output_enabled:
            out_task = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
        else:
            out_task = None

        prev_count = int(len(getattr(pq3d.representation_manager, "object_count", [])))
        goto_rgb: List[np.ndarray] = []
        goto_depth: List[np.ndarray] = []
        goto_state: List[Any] = []
        task_end_reason = "max_steps"
        vote_applied = 0
        vote_attempts = 0
        vlm_extract_elapsed_ms_total = 0.0
        prev_agent_state = agent.get_state()
        sub_episode_start_position = prev_agent_state.position
        episode_cum_distance = 0.0
        final_selected_object_index: Optional[int] = None
        final_selected_object_pos: Optional[np.ndarray] = None
        baseline_final_target_pos: Optional[np.ndarray] = None
        baseline_final_target_obj_index: Optional[int] = None

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

            target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence_nav, decision_num)
            mem_count = int(len(np.asarray(getattr(pq3d.representation_manager, "object_count", np.zeros((0,))), dtype=float)))
            print(
                f"[anchor-vote][decision] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline_target={np.asarray(target, dtype=float).reshape(-1)[:3].tolist()} "
                f"final={bool(is_final)} memory_objects={mem_count}"
            )

            # update object-position bindings after perception/merge
            cur_count = int(len(np.asarray(getattr(pq3d.representation_manager, "object_count", np.zeros((0,))), dtype=float)))
            rep_for_bind = pq3d.representation_manager
            box_for_bind = np.asarray(getattr(rep_for_bind, "object_box", np.zeros((0, 6))), dtype=float)
            object_positions_xyz: Dict[int, Any] = {}
            if box_for_bind.ndim == 2 and box_for_bind.shape[0] >= cur_count and box_for_bind.shape[1] >= 3:
                for oi in range(0, cur_count):
                    obj_xyz = np.asarray(box_for_bind[int(oi), :3], dtype=float).reshape(3).copy()
                    obj_xyz[[1, 2]] = obj_xyz[[2, 1]]
                    object_positions_xyz[int(oi)] = obj_xyz
            bind_records = update_bindings_for_new_objects(
                vote_state,
                prev_object_count=prev_count,
                cur_object_count=cur_count,
                agent_position_xyz=agent.get_state().position,
                object_positions_xyz=object_positions_xyz,
                cfg=vote_cfg,
            )
            prev_count = cur_count

            used_target = np.asarray(target, dtype=float).reshape(3).copy()
            vote_info: Dict[str, Any] = {}
            if is_final:
                baseline_final_target_pos = used_target.copy()
                rep0 = pq3d.representation_manager
                box0 = np.asarray(getattr(rep0, "object_box", np.zeros((0, 6))), dtype=float)
                if box0.ndim == 2 and box0.shape[0] > 0 and box0.shape[1] >= 3:
                    bxyz = np.asarray(used_target, dtype=float).reshape(3).copy()
                    bxyz[[1, 2]] = bxyz[[2, 1]]
                    d_baseline = np.linalg.norm(box0[:, :3] - bxyz[None, :], axis=1)
                    baseline_final_target_obj_index = int(np.argmin(d_baseline))
                vote_attempts += 1
                t_vlm = time.perf_counter()
                main_target, anchors, anchor_weights, spatial_relation, anchor_types = _extract_target_anchors(
                    original_sentence, args.vlm_model, vote_cfg
                )
                refined_query = (
                    str(refined_pack["refined_query"]).strip() if refined_pack else main_target
                )
                vlm_ms = (time.perf_counter() - t_vlm) * 1000.0
                vlm_extract_elapsed_ms_total += vlm_ms
                print(
                    f"[anchor-vote][vlm] task={loop_tid} dec={decision_num} elapsed_ms={vlm_ms:.1f} "
                    f"main_target={main_target!r} anchors={anchors} anchor_types={anchor_types} "
                    f"anchor_weights={anchor_weights} relation={spatial_relation!r} "
                    f"refined_query={refined_query!r}"
                )
                vote_info = run_position_vote_with_pq3d_stage2(
                    vote_state,
                    pq3d_model=pq3d,
                    main_target=main_target,
                    anchors=anchors,
                    anchor_weights=anchor_weights,
                    cfg=vote_cfg,
                    candidate_object_indices=[int(x.get("object_index")) for x in bind_records],
                    refined_pick_text=refined_query,
                )
                if vote_info.get("ok"):
                    chosen = int(vote_info["chosen_object_index"])
                    oracle_meta: Optional[Dict[str, Any]] = None
                    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
                    if box.ndim == 2 and box.shape[0] > 0 and box.shape[1] >= 3 and len(goal_positions) > 0:
                        box_nav = np.asarray(box[:, :3], dtype=float).copy()
                        box_nav[:, [1, 2]] = box_nav[:, [2, 1]]
                        d_all = [np.linalg.norm(box_nav - gp[None, :], axis=1) for gp in goal_positions]
                        d_min = np.min(np.stack(d_all, axis=0), axis=0)
                        oracle_idx = int(np.argmin(d_min))
                        oracle_dist = float(d_min[oracle_idx])
                        _log_pick_o = pq3d_stage2_object_logits(pq3d, refined_query)
                        oracle_score = _pq3d_logit_at(_log_pick_o, oracle_idx)
                        if oracle_score != oracle_score:
                            oracle_score = -1e6
                        oracle_node = vote_state.object_node.get(int(oracle_idx), None)
                        best_node = vote_info.get("best_node_id", None)
                        oracle_in_best = (
                            oracle_node is not None and best_node is not None and int(oracle_node) == int(best_node)
                        )
                        bn_list = list(vote_info.get("best_node_objects") or [])
                        pick_pool_n = len(vote_info.get("pick_pool") or [])
                        oracle_first_rgb_path: Optional[str] = None
                        if out_task is not None:
                            rep_o = pq3d.representation_manager
                            fr_o = list(getattr(rep_o, "object_first_rgb", None) or [])
                            if 0 <= oracle_idx < len(fr_o):
                                rgb_o = np.asarray(fr_o[oracle_idx])
                                if rgb_o.ndim == 3 and rgb_o.shape[2] >= 3:
                                    odir = _ensure_dir(
                                        out_task / "vote_candidate_first_rgb" / f"dec_{decision_num:03d}"
                                    )
                                    op = odir / f"oracle_obj{oracle_idx}_first_rgb.jpg"
                                    _imwrite_rgb(op, rgb_o[:, :, :3])
                                    oracle_first_rgb_path = str(op)
                        oracle_meta = {
                            "object_index": int(oracle_idx),
                            "dist_to_goal": float(oracle_dist),
                            "target_score": float(oracle_score),
                            "node_id": None if oracle_node is None else int(oracle_node),
                            "oracle_in_best_node": bool(oracle_in_best),
                            "first_rgb_path": oracle_first_rgb_path,
                        }
                        print(
                            f"[anchor-vote][diagnose] task={loop_tid} dec={decision_num} "
                            f"oracle_obj={oracle_idx} oracle_dist_to_goal={oracle_dist:.3f} "
                            f"oracle_score={oracle_score:.4f} oracle_node={oracle_node} "
                            f"best_node={best_node} oracle_in_best_node={oracle_in_best} "
                            f"best_node_object_count={len(bn_list)} node_pick_top_k={vote_cfg.node_pick_top_k} "
                            f"pick_pool_len={pick_pool_n} oracle_first_rgb={oracle_first_rgb_path!r}"
                        )
                    if chosen < len(box):
                        used_target = np.asarray(box[chosen, :3], dtype=float).reshape(3).copy()
                        used_target[[1, 2]] = used_target[[2, 1]]
                        vote_applied += 1
                        final_selected_object_index = int(chosen)
                        final_selected_object_pos = used_target.copy()
                print(
                    f"[anchor-vote][vote] task={loop_tid} dec={decision_num} ok={vote_info.get('ok')} "
                    f"best_node={vote_info.get('best_node_id')} chosen_obj={vote_info.get('chosen_object_index')}"
                )
                if vote_info.get("ok"):
                    qlogs = vote_info.get("query_logs", [])
                    for qi, qrec in enumerate(qlogs):
                        tk = qrec.get("topk", [])
                        if qrec.get("query_type") in ("refined_global_topk", "refined_substitute_topk"):
                            topk = tk[:5]
                            topk_str = ", ".join(
                                f"obj={x.get('object_index')} score={float(x.get('score', 0.0)):.4f} "
                                f"node={x.get('node_id')} vw={x.get('vote_weight')}"
                                for x in topk
                            )
                            print(
                                f"[anchor-vote][vote-query] task={loop_tid} dec={decision_num} q{qi} "
                                f"type={qrec.get('query_type')} text={qrec.get('query_text')!r} top5=[{topk_str}]"
                            )
                        elif qrec.get("query_type") == "refined_global_topk_skipped":
                            print(
                                f"[anchor-vote][vote-query] task={loop_tid} dec={decision_num} q{qi} "
                                f"type={qrec.get('query_type')} reason={qrec.get('reason')} "
                                f"text={qrec.get('query_text')!r}"
                            )
                        else:
                            topk = tk[:3]
                            topk_str = ", ".join(
                                f"obj={x.get('object_index')} score={float(x.get('score', 0.0)):.4f} node={x.get('node_id')}"
                                for x in topk
                            )
                            eff = qrec.get("anchor_effective_vote_weight")
                            eff_s = "" if eff is None else f" eff_w={float(eff):.4f}"
                            print(
                                f"[anchor-vote][vote-query] task={loop_tid} dec={decision_num} q{qi} "
                                f"type={qrec.get('query_type')} weight={qrec.get('query_weight')}{eff_s} "
                                f"text={qrec.get('query_text')!r} top3=[{topk_str}]"
                            )
                    node_votes = vote_info.get("node_votes", [])[:3]
                    node_str = ", ".join(
                        f"node={x.get('node_id')} votes={x.get('vote_count')} score_sum={float(x.get('vote_score_sum', 0.0)):.4f}"
                        for x in node_votes
                    )
                    print(
                        f"[anchor-vote][vote-node] task={loop_tid} dec={decision_num} top_nodes=[{node_str}] "
                        f"pick_probs={vote_info.get('pick_probs', [])} "
                        f"selected_from_candidates={vote_info.get('selected_from_candidates')}"
                    )
                    print(
                        f"[anchor-vote][vote-target] task={loop_tid} dec={decision_num} used_target={used_target.tolist()}"
                    )
                    if out_task is not None:
                        rep_rgb = pq3d.representation_manager
                        first_rgbs = list(getattr(rep_rgb, "object_first_rgb", None) or [])
                        cand_dir = _ensure_dir(out_task / "vote_candidate_first_rgb" / f"dec_{decision_num:03d}")
                        cand_records: List[Dict[str, Any]] = []
                        for rank_i, item in enumerate(vote_info.get("pick_pool", []), start=1):
                            obj_i = int(item.get("object_index", -1))
                            score_i = float(item.get("score", 0.0))
                            prob_i = 0.0
                            probs = vote_info.get("pick_probs", [])
                            if isinstance(probs, list) and (rank_i - 1) < len(probs):
                                prob_i = float(probs[rank_i - 1])
                            rec = {
                                "rank": int(rank_i),
                                "object_index": int(obj_i),
                                "score": float(score_i),
                                "prob": float(prob_i),
                                "image_path": None,
                            }
                            if 0 <= obj_i < len(first_rgbs):
                                rgb0 = np.asarray(first_rgbs[obj_i])
                                if rgb0.ndim == 3 and rgb0.shape[2] >= 3:
                                    img_path = cand_dir / f"top{rank_i:02d}_obj{obj_i}_first_rgb.jpg"
                                    _imwrite_rgb(img_path, rgb0[:, :, :3])
                                    rec["image_path"] = str(img_path)
                            cand_records.append(rec)
                        bn_objs = list(vote_info.get("best_node_objects") or [])
                        with open(cand_dir / "index.json", "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "task_id": int(loop_tid),
                                    "decision_num": int(decision_num),
                                    "node_pick_top_k": int(vote_cfg.node_pick_top_k),
                                    "best_node_object_count": len(bn_objs),
                                    "pick_pool_count": len(cand_records),
                                    "pick_pool": cand_records,
                                    "chosen_object_index": vote_info.get("chosen_object_index"),
                                    "oracle": oracle_meta,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                else:
                    print(
                        f"[anchor-vote][vote-fail] task={loop_tid} dec={decision_num} reason={vote_info.get('reason')}"
                    )

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
                rec = {
                    "decision_num": int(decision_num),
                    "is_final": bool(is_final),
                    "bind_records": bind_records,
                    "vote_info": vote_info,
                    "target_used": used_target.tolist(),
                    "nodes": [x.tolist() for x in vote_state.nodes],
                    "object_node": {str(k): int(v) for k, v in vote_state.object_node.items()},
                }
                with open(out_task / f"dec_{decision_num:03d}_vote.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                if len(color_list) > 0:
                    _imwrite_rgb(out_task / f"dec_{decision_num:03d}_last_rgb.jpg", color_list[-1])

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
            if pf.find_path(path):
                start_end_geo_distance = float(path.geodesic_distance)
            else:
                start_end_geo_distance = float("inf")
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if pf.find_path(path):
                agent_end_geo_distance = float(path.geodesic_distance)
            else:
                agent_end_geo_distance = float("inf")
        else:
            start_end_geo_distance = float("inf")
            agent_end_geo_distance = float("inf")
        if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
            sr = 0.0
            spl = 0.0
        else:
            sr = 1.0 if agent_end_geo_distance <= float(args.success_distance) else 0.0
            spl = float(sr * start_end_geo_distance / max(start_end_geo_distance, max(episode_cum_distance, 1e-12)))
        end_position = np.asarray(agent_state.position, dtype=float).reshape(3)
        min_end_to_goal_l2 = float("inf")
        min_selected_obj_to_goal_l2 = float("inf")
        min_baseline_target_to_goal_l2 = float("inf")
        nearest_goal_position = None
        if len(goal_positions) > 0:
            d_end = [float(np.linalg.norm(end_position - gp)) for gp in goal_positions]
            min_end_idx = int(np.argmin(d_end))
            min_end_to_goal_l2 = float(d_end[min_end_idx])
            nearest_goal_position = goal_positions[min_end_idx]
            if final_selected_object_pos is not None:
                d_sel = [float(np.linalg.norm(final_selected_object_pos - gp)) for gp in goal_positions]
                min_selected_obj_to_goal_l2 = float(min(d_sel))
            if baseline_final_target_pos is not None:
                d_base = [float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions]
                min_baseline_target_to_goal_l2 = float(min(d_base))

        print(
            f"[anchor-vote][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} vote_attempts={vote_attempts} "
            f"vote_applied={vote_applied} vlm_elapsed_ms_total={vlm_extract_elapsed_ms_total:.1f} "
            f"SR={sr:.1f} SPL={spl:.4f} start_goal_geo={start_end_geo_distance:.3f} end_goal_geo={agent_end_geo_distance:.3f} "
            f"goal_pos={None if nearest_goal_position is None else nearest_goal_position.tolist()} "
            f"baseline_obj_idx={baseline_final_target_obj_index} "
            f"baseline_target_pos={None if baseline_final_target_pos is None else baseline_final_target_pos.tolist()} "
            f"dist(baseline_target,goal)={min_baseline_target_to_goal_l2:.3f} "
            f"selected_obj_idx={final_selected_object_index} "
            f"selected_obj_pos={None if final_selected_object_pos is None else final_selected_object_pos.tolist()} "
            f"dist(selected_obj,goal)={min_selected_obj_to_goal_l2:.3f} dist(end,goal)={min_end_to_goal_l2:.3f}"
        )
        if out_task is not None:
            summary = {
                "task_id": int(loop_tid),
                "task_level": str(task_type),
                "description": str(original_sentence),
                "refined_query_for_pq3d": str(sentence_nav),
                "object_category": str(goal_category),
                "goal_object_ids": [str(x) for x in goals_ids],
                "goal_positions": [gp.tolist() for gp in goal_positions],
                "steps_total": int(total_steps),
                "decisions": int(decision_num),
                "end_reason": str(task_end_reason),
                "success_distance": float(args.success_distance),
                "start_goal_geo": float(start_end_geo_distance),
                "end_goal_geo": float(agent_end_geo_distance),
                "episode_cum_distance": float(episode_cum_distance),
                "sr": float(sr),
                "spl": float(spl),
                "end_position": end_position.tolist(),
                "baseline_target_object_index": None if baseline_final_target_obj_index is None else int(baseline_final_target_obj_index),
                "baseline_target_position": None if baseline_final_target_pos is None else baseline_final_target_pos.tolist(),
                "baseline_target_to_goal_l2": float(min_baseline_target_to_goal_l2),
                "selected_object_index": None if final_selected_object_index is None else int(final_selected_object_index),
                "selected_object_position": None if final_selected_object_pos is None else final_selected_object_pos.tolist(),
                "selected_object_to_goal_l2": float(min_selected_obj_to_goal_l2),
                "end_to_goal_l2": float(min_end_to_goal_l2),
                "vote_attempts": int(vote_attempts),
                "vote_applied": int(vote_applied),
                "vlm_elapsed_ms_total": float(vlm_extract_elapsed_ms_total),
            }
            with open(out_task / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    sim.close()
    print("[anchor-vote] done")


if __name__ == "__main__":
    main()
