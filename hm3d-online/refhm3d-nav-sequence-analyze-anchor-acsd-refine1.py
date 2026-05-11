"""RefHM3D ACSD refine1 batch run.

ACSD now performs VLM instruction decomposition and object-only
origin-anchor reranking. Exploration frontiers are passed through unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import habitat_sim
import numpy as np
import torch
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_MINIMAL_PATH = SCRIPT_DIR / "refhm3d-nav-sequence-analyze-anchor-acsd.py"
_SPEC = importlib.util.spec_from_file_location("_acsd_minimal_helpers", _MINIMAL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load ACSD helper script: {_MINIMAL_PATH}")
_helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_helpers)

from anchor_nav.acsd import ACSDConfig, AnchorConditionedSoftDecomposition
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


def _tqdm_print(msg: str) -> None:
    tqdm.write(msg, file=sys.stdout)


def _sequence_compute_metric_results(result_dict: Dict[str, Any]) -> None:
    rows = result_dict.get("sequence", [])
    if not rows:
        _tqdm_print("[Metrics] sequence count=0")
        return
    avg_sr = float(np.mean([float(x.get("sr", 0.0)) for x in rows]))
    avg_spl = float(np.mean([float(x.get("spl", 0.0)) for x in rows]))
    avg_time = float(np.mean([float(x.get("task_time_sec", 0.0)) for x in rows]))
    _tqdm_print(
        f"[Metrics] sequence count={len(rows)}, avg_sr={avg_sr:.6f}, "
        f"avg_spl={avg_spl:.6f}, avg_task_time_sec={avg_time:.3f}"
    )


def _set_reproducibility_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _tqdm_print(f"[ACSDRefine1] reproducibility_seed={seed}")


def _load_json_if_exists(path: Path, default: Any) -> Any:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _existing_task_keys(result_dict: Mapping[str, Any]) -> set:
    keys = set()
    for rows in result_dict.values():
        if not isinstance(rows, list):
            continue
        for r in rows:
            keys.add(
                "_".join(
                    [
                        str(r["scene_name"]),
                        str(r["navigation_type"]),
                        str(r["episode_id"]),
                        str(r["task_id"]),
                        str(r.get("task_level", "")),
                    ]
                )
            )
    return keys


def _compare_and_log(
    *,
    acsd: AnchorConditionedSoftDecomposition,
    episode_id: int,
    decision_num: int,
    baseline: Mapping[str, Any],
    corrected: Mapping[str, Any],
    goals: List[np.ndarray],
    decomposition: Mapping[str, Any],
    correction_applied: bool,
    correction_rejected: bool,
    correction_reason: str,
) -> Dict[str, Any]:
    compare = acsd.build_compare_record(
        episode_id=episode_id,
        step_id=decision_num,
        baseline_position=baseline["position"],
        corrected_position=corrected["position"],
        goal_positions=[g.tolist() for g in goals],
        threshold_m=1.0,
    )
    _tqdm_print(
        acsd.format_call_log(
            episode_id=episode_id,
            step_id=decision_num,
            decomposition=decomposition,
            baseline=baseline,
            corrected=corrected,
            correction_applied=bool(correction_applied),
            correction_reason=correction_reason,
        )
    )
    _tqdm_print(acsd.format_compare_log(compare))
    acsd.update_summary_from_decision(
        correction_applied=bool(correction_applied),
        correction_rejected=bool(correction_rejected),
    )
    return compare


def main() -> None:
    parser = argparse.ArgumentParser("RefHM3D ACSD refine1 batch")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=0.2)
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
    parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
    parser.add_argument("--sim_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    parser.add_argument("--agent_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))
    parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/acsd"))
    parser.add_argument("--task_levels", type=str, default="instance")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
    parser.add_argument("--vlm_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
    parser.add_argument("--acsd_top_k", type=int, default=4)
    parser.add_argument("--correction_margin", type=float, default=0.015)
    parser.add_argument("--object_correction_min_baseline_score", type=float, default=0.90)
    parser.add_argument("--quiet_nav_steps", action="store_true")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible baseline/ACSD comparison")
    args = parser.parse_args()

    if args.vlm_api_key:
        os.environ["ZZZ_API_KEY"] = args.vlm_api_key

    output_log_dir = Path(args.output_log_dir).expanduser().resolve()
    _helpers._setup_run_logging(output_log_dir)
    _set_reproducibility_seed(args.seed)
    enabled_task_levels = {x.strip() for x in str(args.task_levels).split(",") if x.strip()}
    if not enabled_task_levels:
        raise RuntimeError("--task_levels resolved to empty set")

    acsd = AnchorConditionedSoftDecomposition(
        ACSDConfig(
            vlm_model=args.vlm_model,
            object_top_k=int(args.acsd_top_k),
            correction_margin=float(args.correction_margin),
            object_correction_min_baseline_score=float(args.object_correction_min_baseline_score),
        )
    )
    _tqdm_print(
        f"[ACSDRefine1] cfg levels={sorted(enabled_task_levels)} start_ratio={args.start_ratio} "
        f"end_ratio={args.end_ratio} top_k={args.acsd_top_k} "
        f"correction_margin={args.correction_margin} "
        f"object_correction_min_baseline_score={args.object_correction_min_baseline_score} "
        f"mode=vlm_decompose_object_only"
    )

    navigation_data_root = Path(args.navigation_data_path).expanduser().resolve()
    scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
    if not scene_data_paths:
        raise FileNotFoundError(f"No *.json.gz found under navigation_data_path={navigation_data_root}")
    scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)) : int(args.end_ratio * len(scene_data_paths))]

    out_name = f"refhm3d_seq_acsd_refine1_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_acsd_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
    if args.concise_description:
        out_name = f"refhm3d_seq_acsd_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
        eff_name = f"refhm3d_seq_acsd_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    output_path = output_log_dir / out_name
    effectiveness_path = output_log_dir / eff_name

    result_dict = _load_json_if_exists(output_path, {"sequence": []})
    existing_tasks = _existing_task_keys(result_dict)
    effectiveness_dict = _load_json_if_exists(
        effectiveness_path,
        {
            "records": [],
            "case_counts": {"00": 0, "01": 0, "10": 0, "11": 0},
            "module_status_counts": {},
        },
    )

    pq3d = PQ3DModel(
        str(Path(args.pq3d_stage1_path).expanduser().resolve()),
        str(Path(args.pq3d_stage2_path).expanduser().resolve()),
        min_decision_num=int(args.decision_num_min),
    )

    for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
        scene_name = scene_data_path.name.split(".")[0]
        with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
            scene_data = json.load(f)
        region_map = scene_data["region_annotation"]
        episode_mapping = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }
        goals_map = {x["object_id"]: x for x in scene_data["goals"]}

        for _, episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
            pq3d.reset()
            visited_frontier = set()
            episode_decision_num = 0
            start_position = episode["start_position"]
            start_rotation = episode["start_rotation"]
            episode_id, navigation_type = int(episode["episode_id"]), str(episode["navigation_type"])

            sim_settings = OmegaConf.load(str(Path(args.sim_config).expanduser().resolve()))
            agent_settings = OmegaConf.load(str(Path(args.agent_config).expanduser().resolve()))
            sim_settings["scene"] = str(
                _helpers._resolve_scene_mesh(Path(args.hm3d_data_base_path).expanduser().resolve(), scene_name)
            )
            abstract_sim = HabitatSimulator(sim_settings, agent_settings)
            sim = abstract_sim.simulator
            agent = abstract_sim.agent
            state = habitat_sim.AgentState()
            state.position = start_position
            state.rotation = start_rotation
            agent.set_state(state)
            path_finder = sim.pathfinder
            top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
            fog = np.zeros_like(top_down_map)
            area_thr = convert_meters_to_pixel(9, 512, sim)
            vis_dist = convert_meters_to_pixel(3.0, 512, sim)

            try:
                for task_id, task_pair in enumerate(episode["task_sequence"]):
                    task_t0 = time.perf_counter()
                    task_type, task_idx = task_pair
                    if task_type not in enabled_task_levels:
                        continue
                    task_key = "_".join([scene_name, navigation_type, str(episode_id), str(task_id), str(task_type)])
                    if task_key in existing_tasks:
                        continue
                    cur_task = episode_mapping[task_type][task_idx]
                    sentence = _helpers._build_sentence(
                        task_type,
                        cur_task,
                        goals_map,
                        region_map,
                        concise=bool(args.concise_description),
                    )
                    goals = _helpers._goal_positions(cur_task, goals_map)
                    goal_category = goals_map[cur_task["target_object_ids"][0]]["object_category"]
                    decomposition = acsd.decompose_instruction(sentence)
                    _tqdm_print(
                        f"[ACSDRefine1][task-start] scene={scene_name} ep={episode_id} task={task_id} "
                        f"level={task_type} room={decomposition.get('room_anchor', '')!r} "
                        f"anchors={decomposition.get('object_anchors', [])!r} "
                        f"fallback={bool(decomposition.get('acsd_fallback_to_baseline', False))}"
                    )

                    task_dir = output_log_dir / "process" / f"scene={scene_name}" / f"episode={episode_id}" / f"task={task_id}"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    total_steps = 0
                    task_decision_count = 0
                    goto_rgb: List[np.ndarray] = []
                    goto_depth: List[np.ndarray] = []
                    goto_states: List[Any] = []
                    prev_agent_state = agent.get_state()
                    sub_episode_start = prev_agent_state.position
                    episode_cum_distance = 0.0
                    task_end_reason = "max_steps"
                    final_compare = None
                    final_correction_applied = False
                    follower_error_info = None

                    while total_steps < int(args.max_steps):
                        color_list: List[np.ndarray] = []
                        depth_list: List[np.ndarray] = []
                        state_list: List[Any] = []
                        if len(goto_rgb) > 6:
                            step = max(1, len(goto_rgb) // 6)
                            goto_rgb = [goto_rgb[i] for i in range(0, len(goto_rgb), step)][:6]
                            goto_depth = [goto_depth[i] for i in range(0, len(goto_depth), step)][:6]
                            goto_states = [goto_states[i] for i in range(0, len(goto_states), step)][:6]
                        color_list.extend(goto_rgb)
                        depth_list.extend(goto_depth)
                        state_list.extend(goto_states)

                        for _ in range(12):
                            obs = sim.step(action="turn_left")
                            color = obs["color_sensor"][:, :, :3]
                            depth = obs["depth_sensor"][:, :]
                            agent_state = agent.get_state()
                            color_list.append(color)
                            depth_list.append(depth)
                            state_list.append(agent_state)
                            fog = reveal_fog_of_war(
                                top_down_map=top_down_map,
                                current_fog_of_war_mask=fog,
                                current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                                current_angle=get_polar_angle(agent_state),
                                fov=42,
                                max_line_len=vis_dist,
                                enable_debug_visualization=False,
                            )
                            total_steps += 1

                        agent_state = agent.get_state()
                        fw = detect_frontier_waypoints(
                            top_down_map,
                            fog,
                            area_thr,
                            xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                            enable_visualization=False,
                        )
                        if len(fw) == 0:
                            frontiers: List[np.ndarray] = []
                        else:
                            frontiers = list(pixel_to_map_coors(fw[:, ::-1], agent_state.position, top_down_map, sim))
                        frontiers = [w for w in frontiers if tuple(np.round(w, 1)) not in visited_frontier]

                        decision_num = int(episode_decision_num)
                        dec_dir = task_dir / f"dec_{decision_num:03d}"
                        dec_dir.mkdir(parents=True, exist_ok=True)
                        target_position, is_final = pq3d.decision(
                            color_list,
                            depth_list,
                            state_list,
                            frontiers,
                            sentence,
                            decision_num,
                        )
                        stage2_payload = getattr(pq3d, "last_stage2_decision", None)
                        if not isinstance(stage2_payload, dict) or not stage2_payload:
                            raise RuntimeError("PQ3D did not expose last_stage2_decision after decision()")
                        _write_json(dec_dir / "stage2_decision.json", stage2_payload)
                        stage2 = acsd.load_stage2_decision(dec_dir / "stage2_decision.json")
                        baseline = acsd.build_baseline_decision(
                            target_position=target_position,
                            is_object_decision=bool(is_final),
                            decision_aux=dict(getattr(pq3d, "last_decision_aux", {}) or {}),
                            stage2=stage2,
                        )
                        if not bool(args.quiet_nav_steps):
                            _tqdm_print(
                                f"[ACSDRefine1][step] scene={scene_name} ep={episode_id} task={task_id} "
                                f"dec={decision_num} baseline_type={baseline['type']} frontiers={len(frontiers)}"
                            )

                        if baseline["type"] == "object":
                            process = acsd.process_decision(
                                instruction=sentence,
                                baseline_decision=baseline,
                                observation_context={"representation_manager": pq3d.representation_manager},
                                candidate_context={"stage2": stage2, "decomposition": decomposition, "output_dir": str(dec_dir / "acsd")},
                                episode_context={
                                    "scene_name": scene_name,
                                    "episode_id": episode_id,
                                    "task_id": task_id,
                                    "task_level": task_type,
                                },
                                gt_context={"goal_positions": [g.tolist() for g in goals]},
                            )
                            for log_line in process.get("logs", []):
                                _tqdm_print(str(log_line))
                            corrected = dict(process["corrected"])
                            correction_applied = bool(process["correction_applied"])
                            correction_rejected = bool(process.get("correction_rejected", False))
                            correction_reason = str(process["correction_reason"])
                        elif baseline["type"] == "frontier":
                            corrected = {
                                "type": "frontier",
                                "position": list(baseline["position"]),
                                "score": float(baseline["score"]),
                            }
                            process = {
                                "module_enabled": True,
                                "decomposition": decomposition,
                                "baseline": dict(baseline),
                                "corrected": corrected,
                                "object_rerank": None,
                                "compare": None,
                                "logs": [],
                                "correction_applied": False,
                                "correction_rejected": False,
                                "correction_reason": "frontier_passthrough_no_acsd_object_correction",
                            }
                            correction_applied = False
                            correction_rejected = False
                            correction_reason = "frontier_passthrough_no_acsd_object_correction"
                        else:
                            raise RuntimeError(f"unknown baseline decision type: {baseline['type']!r}")
                        follow_failed = False
                        follow_target = np.asarray(corrected["position"], dtype=float).reshape(3)
                        if corrected["type"] == "frontier":
                            visited_frontier.add(tuple(np.round(follow_target, 1)))
                        episode_decision_num += 1
                        task_decision_count += 1

                        agent_island = path_finder.get_island(agent_state.position)
                        target_nav = path_finder.snap_point(point=follow_target, island_index=agent_island)
                        follower = habitat_sim.GreedyGeodesicFollower(
                            path_finder,
                            agent,
                            forward_key="move_forward",
                            left_key="turn_left",
                            right_key="turn_right",
                        )
                        try:
                            action_list = follower.find_path(target_nav)
                        except Exception as exc:
                            follower_error_info = {
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "raw_target": follow_target.tolist(),
                                "snapped_target": np.asarray(target_nav, dtype=float).reshape(3).tolist(),
                                "decision_num": int(decision_num),
                                "baseline_type": str(baseline["type"]),
                                "corrected_type": str(corrected["type"]),
                                "baseline_position": list(baseline["position"]),
                                "corrected_position": list(corrected["position"]),
                                "correction_applied": bool(correction_applied),
                            }
                            if "slot_index" in corrected:
                                follower_error_info["slot_index"] = int(corrected["slot_index"])
                            path = habitat_sim.ShortestPath()
                            path.requested_start = agent_state.position
                            path.requested_end = target_nav
                            if sim.pathfinder.find_path(path):
                                follower_error_info["shortest_path_found"] = True
                                follower_error_info["shortest_path_geodesic_distance"] = float(path.geodesic_distance)
                            else:
                                follower_error_info["shortest_path_found"] = False
                                follower_error_info["shortest_path_geodesic_distance"] = float("inf")
                            task_end_reason = "follower_error"
                            follow_failed = True
                            action_list = []
                            _tqdm_print(
                                f"[ACSD_FOLLOW_ERROR] scene={scene_name} ep={episode_id} task={task_id} "
                                f"dec={decision_num} type={corrected['type']} error={type(exc).__name__} "
                                f"path_found={follower_error_info['shortest_path_found']}"
                            )

                        goto_rgb = []
                        goto_depth = []
                        goto_states = []
                        for action in action_list:
                            if action:
                                obs = sim.step(action=action)
                                agent_state = agent.get_state()
                                color = obs["color_sensor"][:, :, :3]
                                depth = obs["depth_sensor"][:, :]
                                goto_rgb.append(color)
                                goto_depth.append(depth)
                                goto_states.append(agent_state)
                                fog = reveal_fog_of_war(
                                    top_down_map=top_down_map,
                                    current_fog_of_war_mask=fog,
                                    current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                                    current_angle=get_polar_angle(agent_state),
                                    fov=42,
                                    max_line_len=vis_dist,
                                    enable_debug_visualization=False,
                                )
                                total_steps += 1
                                episode_cum_distance += float(np.linalg.norm(agent_state.position - prev_agent_state.position))
                                prev_agent_state = agent_state

                        compare = _compare_and_log(
                            acsd=acsd,
                            episode_id=episode_id,
                            decision_num=decision_num,
                            baseline=baseline,
                            corrected=corrected,
                            goals=goals,
                            decomposition=decomposition,
                            correction_applied=bool(correction_applied),
                            correction_rejected=bool(correction_rejected),
                            correction_reason=correction_reason,
                        )
                        final_compare = compare
                        final_correction_applied = bool(correction_applied)
                        _write_json(
                            dec_dir / "acsd_decision.json",
                            {
                                "scene_name": scene_name,
                                "episode_id": int(episode_id),
                                "task_id": int(task_id),
                                "decision_num": int(decision_num),
                                "instruction": sentence,
                                "decomposition": decomposition,
                                "baseline": baseline,
                                "corrected": corrected,
                                "compare": compare,
                                "process": process,
                                "correction_applied": bool(correction_applied),
                                "correction_rejected": bool(correction_rejected),
                                "correction_reason": correction_reason,
                            },
                        )
                        if follow_failed:
                            break
                        if corrected["type"] == "object":
                            task_end_reason = "final_decision"
                            break

                    task_time = float(time.perf_counter() - task_t0)
                    agent_state = agent.get_state()
                    view_points = [
                        vp["agent_state"]["position"]
                        for gid in cur_task["target_object_ids"]
                        for vp in goals_map[gid].get("view_points", [])
                    ]
                    sp = habitat_sim.MultiGoalShortestPath()
                    sp.requested_start = sub_episode_start
                    sp.requested_ends = view_points
                    start_end_geo = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
                    ep = habitat_sim.MultiGoalShortestPath()
                    ep.requested_start = agent_state.position
                    ep.requested_ends = view_points
                    end_geo = float(ep.geodesic_distance) if path_finder.find_path(ep) else float("inf")
                    if np.isinf(start_end_geo) or np.isinf(end_geo):
                        sr, spl = 0.0, 0.0
                    else:
                        sr = 1.0 if end_geo <= float(args.success_distance) else 0.0
                        spl = float(sr * start_end_geo / max(start_end_geo, episode_cum_distance, 1e-12))
                    if task_end_reason == "follower_error":
                        sr, spl = 0.0, 0.0

                    row = {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(task_id),
                        "task_level": task_type,
                        "navigation_type": navigation_type,
                        "sr": float(sr),
                        "spl": float(spl),
                        "object_category": str(goal_category),
                        "task_time_sec": float(task_time),
                        "steps_total": int(total_steps),
                        "decisions": int(task_decision_count),
                        "episode_decision_num_end": int(episode_decision_num),
                        "end_reason": task_end_reason,
                        "start_goal_geo": float(start_end_geo),
                        "end_goal_geo": float(end_geo),
                        "episode_cum_distance": float(episode_cum_distance),
                        "acsd_case": None if final_compare is None else final_compare.get("case"),
                        "acsd_correction_applied": bool(final_correction_applied),
                        "follower_error_info": follower_error_info,
                    }
                    result_dict.setdefault("sequence", []).append(row)
                    existing_tasks.add(task_key)
                    effectiveness_dict.setdefault("records", []).append(
                        {
                            **row,
                            "instruction": sentence,
                            "goal_positions": [g.tolist() for g in goals],
                            "final_compare": final_compare,
                        }
                    )
                    if final_compare is not None:
                        case = str(final_compare["case"])
                        effectiveness_dict.setdefault("case_counts", {"00": 0, "01": 0, "10": 0, "11": 0})
                        effectiveness_dict["case_counts"][case] = int(effectiveness_dict["case_counts"].get(case, 0)) + 1
                    ms = effectiveness_dict.setdefault("module_status_counts", {})
                    if final_correction_applied:
                        ms["acsd_applied"] = int(ms.get("acsd_applied", 0)) + 1
                    else:
                        ms["acsd_kept_baseline"] = int(ms.get("acsd_kept_baseline", 0)) + 1

                    _write_json(task_dir / "task_summary.json", row)
                    _write_json(output_path, result_dict)
                    _write_json(effectiveness_path, effectiveness_dict)
                    _tqdm_print(
                        f"[ACSDRefine1][task-summary] scene={scene_name} ep={episode_id} task={task_id} "
                        f"SR={sr:.1f} SPL={spl:.4f} steps={total_steps} decisions={task_decision_count} "
                        f"episode_decision_num_end={episode_decision_num} "
                        f"case={row['acsd_case']} correction_applied={final_correction_applied}"
                    )
            finally:
                sim.close()
                _write_json(output_path, result_dict)
                _write_json(effectiveness_path, effectiveness_dict)
                _sequence_compute_metric_results(result_dict)
                _tqdm_print(acsd.format_summary_log())

    _write_json(output_path, result_dict)
    _write_json(effectiveness_path, effectiveness_dict)
    _sequence_compute_metric_results(result_dict)
    _tqdm_print(acsd.format_summary_log())


if __name__ == "__main__":
    main()
