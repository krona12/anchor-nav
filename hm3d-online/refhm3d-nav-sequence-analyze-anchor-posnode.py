from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from anchor_nav.posnode import (
    _save_rgb_jpg,
    MergeTracker,
    PosNodeRegistry,
    build_selection_trace,
    build_query_fn_from_pq3d_stage2,
    decompose_description,
    query_registry_with_vlm,
    registry_snapshot,
    select_from_topk,
    stitch_panorama,
    update_panorama_node,
    validate_cooccur_nodes_with_image,
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


def _dump_panorama_debug(
    *,
    out_task: Path,
    decision_num: int,
    scan_rgb: List[np.ndarray],
    scan_rgb_reversed: List[np.ndarray],
) -> None:
    dbg_dir = _ensure_dir(out_task / "panorama_debug" / f"dec_{int(decision_num):03d}")
    capture_files = []
    reverse_files = []
    for i, rgb in enumerate(scan_rgb):
        fp = dbg_dir / f"capture_order_{int(i):02d}.jpg"
        _save_rgb_jpg(np.asarray(rgb, dtype=np.uint8), fp)
        capture_files.append(str(fp))
    for i, rgb in enumerate(scan_rgb_reversed):
        fp = dbg_dir / f"reverse_order_{int(i):02d}.jpg"
        _save_rgb_jpg(np.asarray(rgb, dtype=np.uint8), fp)
        reverse_files.append(str(fp))

    if len(scan_rgb) > 0:
        _save_rgb_jpg(stitch_panorama(scan_rgb), dbg_dir / "stitched_capture_order.jpg")
    if len(scan_rgb_reversed) > 0:
        _save_rgb_jpg(stitch_panorama(scan_rgb_reversed), dbg_dir / "stitched_reverse_order.jpg")

    meta = {
        "decision_num": int(decision_num),
        "capture_count": int(len(scan_rgb)),
        "capture_indices": [int(i) for i in range(len(scan_rgb))],
        "reverse_source_indices": [int(i) for i in list(reversed(range(len(scan_rgb))))],
        "capture_files": capture_files,
        "reverse_files": reverse_files,
        "stitched_capture_order": str(dbg_dir / "stitched_capture_order.jpg"),
        "stitched_reverse_order": str(dbg_dir / "stitched_reverse_order.jpg"),
    }
    with open(dbg_dir / "order_debug.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _render_topdown_with_agent(
    *,
    top_down_map: np.ndarray,
    fog_mask: np.ndarray,
    agent_rc: np.ndarray,
    radius: int = 6,
) -> np.ndarray:
    try:
        base = maps.colorize_topdown_map(top_down_map, fog_mask)
        rgb = np.asarray(base[:, :, :3], dtype=np.uint8).copy()
    except Exception:
        m = np.asarray(top_down_map, dtype=float)
        mn = float(np.min(m)) if m.size > 0 else 0.0
        mx = float(np.max(m)) if m.size > 0 else 1.0
        g = ((m - mn) / max(mx - mn, 1e-6) * 180.0).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=2)
        explored = np.asarray(fog_mask) > 0
        rgb[~explored] = np.array([18, 18, 18], dtype=np.uint8)

    h, w = rgb.shape[:2]
    rr = int(np.clip(int(agent_rc[0]), 0, h - 1))
    cc = int(np.clip(int(agent_rc[1]), 0, w - 1))
    rad = int(max(2, radius))
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            if dr * dr + dc * dc > rad * rad:
                continue
            r = rr + dr
            c = cc + dc
            if 0 <= r < h and 0 <= c < w:
                rgb[r, c, :] = np.array([255, 36, 36], dtype=np.uint8)
    # Add a bright center to make marker visually "bold".
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r = rr + dr
            c = cc + dc
            if 0 <= r < h and 0 <= c < w:
                rgb[r, c, :] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _dump_topdown_debug(
    *,
    out_task: Path,
    decision_num: int,
    top_down_map: np.ndarray,
    fog_mask: np.ndarray,
    agent_rc: np.ndarray,
) -> Path:
    dbg_dir = _ensure_dir(out_task / "topdown_debug")
    rgb = _render_topdown_with_agent(
        top_down_map=top_down_map,
        fog_mask=fog_mask,
        agent_rc=agent_rc,
        radius=6,
    )
    out_path = dbg_dir / f"dec_{int(decision_num):03d}_topdown.jpg"
    _save_rgb_jpg(rgb, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser("Anchor PosNode analyze（仅 posnode 模块）")
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
    parser.add_argument("--posnode_top_k", type=int, default=16)
    parser.add_argument("--panorama_update_interval", type=int, default=2)
    parser.add_argument("--panorama_subsample_frames", type=int, default=12)
    parser.add_argument("--visible_object_max_dist", type=float, default=6.0)
    parser.add_argument("--posnode_min_move_dist", type=float, default=0.4)
    parser.add_argument("--auto_vlm_after_decision", type=int, default=6)
    parser.add_argument("--auto_vlm_interval", type=int, default=2)
    parser.add_argument("--disable_auto_vlm_frontier", action="store_true")
    parser.add_argument("--disable_early_final_on_cooccur", action="store_true")
    parser.add_argument("--dump_panorama_debug", action="store_true", default=True)
    parser.add_argument("--no_dump_panorama_debug", action="store_false", dest="dump_panorama_debug")
    parser.add_argument("--dump_topdown_debug", action="store_true", default=True)
    parser.add_argument("--no_dump_topdown_debug", action="store_false", dest="dump_topdown_debug")
    parser.add_argument("--output_root", type=str, default="./output_logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    output_enabled = not bool(args.quiet)
    print(
        f"[posnode] run_tag={run_tag} top_k={args.posnode_top_k} "
        f"update_interval={args.panorama_update_interval} pano_frames={args.panorama_subsample_frames} "
        f"min_move_dist={args.posnode_min_move_dist} auto_vlm_frontier={not args.disable_auto_vlm_frontier} "
        f"auto_after={args.auto_vlm_after_decision} auto_interval={args.auto_vlm_interval} "
        f"dump_panorama_debug={bool(args.dump_panorama_debug)} "
        f"dump_topdown_debug={bool(args.dump_topdown_debug)}"
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
        out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-anchor-posnode")

    registry = PosNodeRegistry()
    merge_tracker = MergeTracker()

    for loop_tid in range(int(args.task_id), task_end):
        total_steps = 0
        decision_num = 0
        task_type, task_idx = task_sequence[loop_tid]
        cur_task = episode_mapping[task_type][task_idx]
        sentence = _build_sentence(task_type, cur_task, goals_map, region_map, concise=(args.description_mode == "concise"))
        decomp = decompose_description(sentence, args.vlm_model)
        print(
            f"[posnode][task-start] task={loop_tid} level={task_type} desc={sentence!r} "
            f"target_desc={decomp.get('target_desc')!r} anchor_desc={decomp.get('anchor_desc')!r}"
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
        posnode_attempts = 0
        posnode_used = 0
        auto_vlm_attempts = 0
        auto_vlm_applied = 0
        baseline_final_target_pos = None
        final_selected_object_pos = None

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

            scan_rgb: List[np.ndarray] = []
            for _ in range(12):
                obs = sim.step(action="turn_left")
                rgb = obs["color_sensor"][:, :, :3]
                dep = obs["depth_sensor"][:, :]
                st_now = agent.get_state()
                scan_rgb.append(rgb)
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

            scan_rgb_reversed = list(reversed(scan_rgb))
            if bool(args.dump_panorama_debug) and out_task is not None:
                _dump_panorama_debug(
                    out_task=out_task,
                    decision_num=int(decision_num),
                    scan_rgb=scan_rgb,
                    scan_rgb_reversed=scan_rgb_reversed,
                )
                print(
                    f"[posnode][pano-debug] task={loop_tid} dec={decision_num} "
                    f"dump_dir={out_task / 'panorama_debug' / f'dec_{int(decision_num):03d}'}"
                )

            st_now = agent.get_state()
            agent_rc = map_coors_to_pixel(st_now.position, top_down_map, sim)
            if bool(args.dump_topdown_debug) and out_task is not None:
                td_path = _dump_topdown_debug(
                    out_task=out_task,
                    decision_num=int(decision_num),
                    top_down_map=top_down_map,
                    fog_mask=fog,
                    agent_rc=np.asarray(agent_rc, dtype=int).reshape(2),
                )
                print(f"[posnode][topdown-debug] task={loop_tid} dec={decision_num} path={td_path}")
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
                f"[posnode][decision] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                f"baseline_target={used_target.tolist()} final={bool(is_final)}"
            )

            if int(args.panorama_update_interval) > 0 and ((decision_num + 1) % int(args.panorama_update_interval) == 0):
                try:
                    scan_rgb_panorama = list(reversed(scan_rgb))
                    node_log = update_panorama_node(
                        agent_pos=np.asarray(agent.get_state().position, dtype=float).reshape(3),
                        color_list=scan_rgb_panorama,
                        rep=pq3d.representation_manager,
                        registry=registry,
                        merge_tracker=merge_tracker,
                        vlm_model=args.vlm_model,
                        step_index=int(total_steps),
                        panorama_dir=pano_dir,
                        max_visible_dist=float(args.visible_object_max_dist),
                        panorama_subsample_frames=int(args.panorama_subsample_frames),
                        min_move_dist_to_add=float(args.posnode_min_move_dist),
                    )
                    if bool(node_log.get("ok", False)):
                        print(
                            f"[posnode][registry-add] task={loop_tid} dec={decision_num} "
                            f"node={node_log.get('node_index')} vlm_names={node_log.get('vlm_names', [])[:5]} "
                            f"visible_count={len(node_log.get('visible_indices', []))} "
                            f"pano_frames={node_log.get('panorama_frames_used')}"
                        )
                    else:
                        print(
                            f"[posnode][registry-skip] task={loop_tid} dec={decision_num} "
                            f"reason={node_log.get('reason')} move_dist={node_log.get('move_dist')} "
                            f"min_move={node_log.get('min_move_dist_to_add')}"
                        )
                    if out_task is not None:
                        with open(
                            out_task / f"registry_node_{int(node_log.get('node_index', -1)):03d}.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(
                                {
                                    "task_id": int(loop_tid),
                                    "decision_num": int(decision_num),
                                    "node_log": node_log,
                                    "registry_tail": registry_snapshot(
                                        registry=registry,
                                        merge_tracker=merge_tracker,
                                        rep=pq3d.representation_manager,
                                        max_nodes=8,
                                    ),
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                except Exception as e:
                    print(f"[posnode][registry-add-fail] task={loop_tid} dec={decision_num} err={e!r}")

            posnode_log: Dict[str, Any] = {"ok": False}
            if (not is_final) and (not bool(args.disable_auto_vlm_frontier)):
                if int(decision_num) >= int(args.auto_vlm_after_decision) and (
                    int(decision_num) % max(1, int(args.auto_vlm_interval)) == 0
                ):
                    auto_vlm_attempts += 1
                    query_result_auto = query_registry_with_vlm(
                        description=sentence,
                        registry=registry,
                        vlm_model=args.vlm_model,
                        decomp=decomp,
                    )
                    mode_auto = str(query_result_auto.get("mode", "fallback"))
                    if mode_auto == "co_occur" and len(query_result_auto.get("matched_nodes", [])) > 0:
                        verify_auto = validate_cooccur_nodes_with_image(
                            description=sentence,
                            target_desc=str(query_result_auto.get("target_desc", "")),
                            anchor_descs=[str(x) for x in query_result_auto.get("anchor_descs", [])],
                            matched_nodes=list(query_result_auto.get("matched_nodes", [])),
                            vlm_model=args.vlm_model,
                        )
                        if bool(verify_auto.get("valid", False)):
                            valid_offsets = set(int(x) for x in verify_auto.get("valid_node_offsets", []))
                            matched_nodes_all = list(query_result_auto.get("matched_nodes", []))
                            matched_idx_all = list(query_result_auto.get("matched_node_indices", []))
                            matched_nodes_valid = [n for j, n in enumerate(matched_nodes_all) if j in valid_offsets]
                            matched_idx_valid = [int(matched_idx_all[j]) for j in range(len(matched_idx_all)) if j in valid_offsets]
                            query_result_auto_eff = dict(query_result_auto)
                            query_result_auto_eff["matched_nodes"] = matched_nodes_valid
                            query_result_auto_eff["matched_node_indices"] = matched_idx_valid
                            target_text_auto = str(query_result_auto_eff.get("target_desc", "")).strip() or sentence
                            topk_auto = list(query_fn(target_text_auto, int(args.posnode_top_k)))
                            chosen_idx_auto = select_from_topk(
                                topk=topk_auto,
                                query_result=query_result_auto_eff,
                                merge_tracker=merge_tracker,
                                rep=pq3d.representation_manager,
                            )
                            rep_box_auto = np.asarray(
                                getattr(pq3d.representation_manager, "object_box", np.zeros((0, 6))), dtype=float
                            )
                            if (
                                rep_box_auto.ndim == 2
                                and rep_box_auto.shape[1] >= 3
                                and 0 <= int(chosen_idx_auto) < int(rep_box_auto.shape[0])
                            ):
                                chosen_xyz_auto = np.asarray(rep_box_auto[int(chosen_idx_auto), :3], dtype=float).reshape(3).copy()
                                chosen_xyz_auto[[1, 2]] = chosen_xyz_auto[[2, 1]]
                                used_target = chosen_xyz_auto.copy()
                                auto_vlm_applied += 1
                                if not bool(args.disable_early_final_on_cooccur):
                                    is_final = True
                                posnode_log["auto_frontier"] = {
                                    "triggered": True,
                                    "mode": mode_auto,
                                    "early_final_applied": bool(not args.disable_early_final_on_cooccur),
                                    "matched_node_indices": matched_idx_valid,
                                    "chosen_object_index": int(chosen_idx_auto),
                                    "verify": verify_auto,
                                }
                                print(
                                    f"[posnode][auto-frontier] task={loop_tid} dec={decision_num} mode={mode_auto} "
                                    f"chosen={int(chosen_idx_auto)}"
                                )
                            else:
                                posnode_log["auto_frontier"] = {
                                    "triggered": False,
                                    "mode": mode_auto,
                                    "reason": "chosen_idx_invalid_after_verify",
                                    "verify": verify_auto,
                                }
                        else:
                            posnode_log["auto_frontier"] = {
                                "triggered": False,
                                "mode": mode_auto,
                                "reason": "verify_rejected",
                                "verify": verify_auto,
                            }
                    else:
                        posnode_log["auto_frontier"] = {
                            "triggered": False,
                            "mode": mode_auto,
                            "reason": "mode_not_co_occur_or_no_match",
                            "matched_node_indices": query_result_auto.get("matched_node_indices", []),
                        }
            if is_final:
                baseline_final_target_pos = used_target.copy()
                posnode_attempts += 1
                final_verify_info: Dict[str, Any] = {}
                query_result = query_registry_with_vlm(
                    description=sentence,
                    registry=registry,
                    vlm_model=args.vlm_model,
                    decomp=decomp,
                )
                query_result_eff = dict(query_result)
                if str(query_result.get("mode", "fallback")) == "co_occur" and len(query_result.get("matched_nodes", [])) > 0:
                    verify_final = validate_cooccur_nodes_with_image(
                        description=sentence,
                        target_desc=str(query_result.get("target_desc", "")),
                        anchor_descs=[str(x) for x in query_result.get("anchor_descs", [])],
                        matched_nodes=list(query_result.get("matched_nodes", [])),
                        vlm_model=args.vlm_model,
                    )
                    final_verify_info = dict(verify_final)
                    if bool(verify_final.get("valid", False)):
                        valid_offsets = set(int(x) for x in verify_final.get("valid_node_offsets", []))
                        matched_nodes_all = list(query_result.get("matched_nodes", []))
                        matched_idx_all = list(query_result.get("matched_node_indices", []))
                        query_result_eff["matched_nodes"] = [n for j, n in enumerate(matched_nodes_all) if j in valid_offsets]
                        query_result_eff["matched_node_indices"] = [
                            int(matched_idx_all[j]) for j in range(len(matched_idx_all)) if j in valid_offsets
                        ]
                    else:
                        # Verification rejected: fallback to baseline top1.
                        query_result_eff["mode"] = "fallback"
                        query_result_eff["matched_nodes"] = []
                        query_result_eff["matched_node_indices"] = []

                target_text = str(query_result_eff.get("target_desc", "")).strip() or sentence
                topk = list(query_fn(target_text, int(args.posnode_top_k)))
                selection_trace = build_selection_trace(
                    topk=topk,
                    query_result=query_result_eff,
                    merge_tracker=merge_tracker,
                    rep=pq3d.representation_manager,
                )
                chosen_idx = select_from_topk(
                    topk=topk,
                    query_result=query_result_eff,
                    merge_tracker=merge_tracker,
                    rep=pq3d.representation_manager,
                )
                top1_idx = int(topk[0][0]) if len(topk) > 0 else -1
                rep_box = np.asarray(getattr(pq3d.representation_manager, "object_box", np.zeros((0, 6))), dtype=float)
                if rep_box.ndim == 2 and rep_box.shape[1] >= 3 and 0 <= chosen_idx < int(rep_box.shape[0]):
                    chosen_xyz = np.asarray(rep_box[chosen_idx, :3], dtype=float).reshape(3).copy()
                    chosen_xyz[[1, 2]] = chosen_xyz[[2, 1]]
                    if chosen_idx != top1_idx:
                        posnode_used += 1
                        used_target = chosen_xyz
                        final_selected_object_pos = chosen_xyz.copy()
                    else:
                        used_target = baseline_final_target_pos.copy()
                        final_selected_object_pos = baseline_final_target_pos.copy()
                else:
                    used_target = baseline_final_target_pos.copy()
                    final_selected_object_pos = baseline_final_target_pos.copy()

                posnode_log = {
                    "ok": True,
                    "query_result": {
                        "mode": query_result_eff.get("mode"),
                        "target_desc": query_result_eff.get("target_desc"),
                        "anchor_desc": query_result_eff.get("anchor_desc"),
                        "anchor_descs": query_result_eff.get("anchor_descs", []),
                        "matched_node_indices": query_result_eff.get("matched_node_indices", []),
                    },
                    "topk": [{"object_index": int(i), "score": float(s)} for i, s in topk],
                    "selection_trace": selection_trace,
                    "chosen_object_index": int(chosen_idx),
                    "top1_object_index": int(top1_idx),
                    "registry_nodes_total": int(len(registry.nodes)),
                    "final_verify": final_verify_info,
                }
                print(
                    f"[posnode][final] task={loop_tid} dec={decision_num} mode={query_result_eff.get('mode')} "
                    f"chosen={chosen_idx} top1={top1_idx} used={chosen_idx != top1_idx}"
                )

            if not is_final:
                visited_frontier.add(tuple(np.round(used_target, 1)))

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
                with open(out_task / f"dec_{decision_num:03d}_posnode.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": int(loop_tid),
                            "decision_num": int(decision_num),
                            "is_final": bool(is_final),
                            "target_used": used_target.tolist(),
                            "registry_nodes_total": int(len(registry.nodes)),
                            "registry_tail": registry_snapshot(
                                registry=registry,
                                merge_tracker=merge_tracker,
                                rep=pq3d.representation_manager,
                                max_nodes=12,
                            ),
                            "auto_vlm_attempts": int(auto_vlm_attempts),
                            "auto_vlm_applied": int(auto_vlm_applied),
                            "posnode": posnode_log,
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
            f"[posnode][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
            f"decisions={decision_num} end_reason={task_end_reason} posnode_attempts={posnode_attempts} "
            f"posnode_used={posnode_used} auto_vlm_attempts={auto_vlm_attempts} auto_vlm_applied={auto_vlm_applied} "
            f"registry_nodes={len(registry.nodes)} "
            f"SR={sr:.1f} SPL={spl:.4f}"
        )

    sim.close()
    print("[posnode] done")


if __name__ == "__main__":
    main()
