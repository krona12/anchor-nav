"""RefHM3D minimal ACSD test.

This script is intentionally strict: ACSD input or VLM failures raise instead of
falling back to baseline. It writes process artifacts under output_process.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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


class _TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = log_dir / f"refhm3d-nav-sequence-analyze-anchor-acsd-{ts}-pid{os.getpid()}.log"
    fp = open(path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, fp)
    sys.stderr = _TeeStream(old_err, fp)
    print(f"[ACSDTest] logging enabled -> {path.resolve()}")

    import atexit

    def _cleanup() -> None:
        try:
            print(f"[ACSDTest] run finished, log saved -> {path.resolve()}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            fp.close()

    atexit.register(_cleanup)
    return path


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_scene_mesh(scene_root: Path, scene_name: str) -> Path:
    sid = scene_name.split("-")[-1]
    candidates = [
        scene_root / scene_name / f"{sid}.basis.glb",
        scene_root / scene_name / f"{sid}.glb",
        scene_root / scene_name / f"{sid}.basis.scene_instance.json",
        scene_root / scene_name / f"{sid}.scene_instance.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve scene asset for {scene_name} under {scene_root}")


def _build_sentence(
    task_type: str,
    cur_task: Mapping[str, Any],
    goals_map: Mapping[str, Any],
    region_map: Mapping[str, Any],
    concise: bool,
) -> str:
    if task_type == "object":
        return str(cur_task["object_category"])
    if task_type == "room":
        return f"{cur_task['object_category']} in the {str(cur_task['room_name']).lower()}"
    if task_type == "region":
        region_info = region_map[cur_task["region_id"]]
        if concise:
            desc = (
                region_info.get("shortest_description")
                or region_info.get("concise_description")
                or region_info.get("detailed_description")
                or ""
            )
        else:
            desc = (
                region_info.get("comprehensive_description")
                or region_info.get("detailed_description")
                or region_info.get("concise_description")
                or ""
            )
        return f"{cur_task['object_category']} in the {str(region_info['region_category']).lower()} that has {desc}"
    if task_type == "instance":
        inst = goals_map[cur_task["instance_id"]]
        if concise:
            return str(inst.get("annot_unique_concise_description") or "")
        return str(
            inst.get("annot_unique_detailed_description")
            or inst.get("annot_unique_normal_description")
            or inst.get("annot_appearance_description")
            or ""
        )
    raise ValueError(f"unknown task_type={task_type!r}")


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
    rgb_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    state_list: List[Any] = []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        state = agent.get_state()
        rgb_list.append(obs["color_sensor"][:, :, :3])
        depth_list.append(obs["depth_sensor"][:, :])
        state_list.append(state)
        fog[:] = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=fog,
            current_point=map_coors_to_pixel(state.position, top_down_map, sim),
            current_angle=get_polar_angle(state),
            fov=42,
            max_line_len=vis_dist,
            enable_debug_visualization=False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return rgb_list, depth_list, state_list, fog, total_steps


def _follow_target(
    *,
    path_finder: Any,
    agent: Any,
    sim: Any,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    vis_dist: int,
    target: Sequence[float],
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float]:
    target_arr = np.asarray(target, dtype=float).reshape(3)
    agent_island = path_finder.get_island(agent.get_state().position)
    target_nav = path_finder.snap_point(point=target_arr, island_index=agent_island)
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
        raise RuntimeError(f"ACSD follow_target failed for target={target_arr.tolist()}: {exc!r}") from exc
    rgb_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    state_list: List[Any] = []
    for action in action_list:
        if not action:
            continue
        obs = sim.step(action=action)
        state = agent.get_state()
        rgb_list.append(obs["color_sensor"][:, :, :3])
        depth_list.append(obs["depth_sensor"][:, :])
        state_list.append(state)
        fog[:] = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=fog,
            current_point=map_coors_to_pixel(state.position, top_down_map, sim),
            current_angle=get_polar_angle(state),
            fov=42,
            max_line_len=vis_dist,
            enable_debug_visualization=False,
        )
        episode_cum_distance += float(np.linalg.norm(state.position - prev_agent_state.position))
        prev_agent_state = state
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return rgb_list, depth_list, state_list, prev_agent_state, total_steps, float(episode_cum_distance)


def _goal_positions(cur_task: Mapping[str, Any], goals_map: Mapping[str, Any]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for gid in cur_task.get("target_object_ids", []):
        g = goals_map.get(gid)
        if isinstance(g, Mapping) and len(g.get("position", [])) >= 3:
            out.append(np.asarray(g["position"], dtype=float).reshape(3))
    if len(out) == 0:
        raise RuntimeError("ACSD minimal test requires GT goal positions for ACSD_COMPARE")
    return out


def _highest_stage2_frontier(stage2: Mapping[str, Any]) -> Dict[str, Any]:
    frs = stage2.get("frontier_candidates")
    if not isinstance(frs, list) or len(frs) == 0:
        raise RuntimeError("ACSD object rejection requires frontier candidates, but stage2 has none")
    ranked = sorted(frs, key=lambda x: float(x["og3d_logit"]), reverse=True)
    top = ranked[0]
    return {
        "type": "frontier",
        "position": [float(x) for x in top["center_habitat_xyz"]],
        "score": float(top["og3d_logit"]),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("RefHM3D ACSD minimal test")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
    parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
    parser.add_argument("--sim_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    parser.add_argument("--agent_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))
    parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
    parser.add_argument("--vlm_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
    parser.add_argument("--acsd_top_k", type=int, default=4)
    parser.add_argument("--acsd_verify_attempts", type=int, default=1, help="Retained for compatibility; component 4 is removed.")
    parser.add_argument("--output_root", type=str, default=str(PROJECT_ROOT / "output_process"))
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    if args.vlm_api_key:
        os.environ["ZZZ_API_KEY"] = args.vlm_api_key

    run_tag = args.run_tag or f"{_now_tag()}-acsd-minimal"
    out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / run_tag)
    _setup_run_logging(out_root)
    print(
        f"[ACSDTest] run_tag={run_tag} scene={args.scene_name} episode={args.episode_id} "
        f"task_id={args.task_id} num_tasks={args.num_tasks} top_k={args.acsd_top_k} "
        f"selected_candidates={args.acsd_verify_attempts} component4_removed=True"
    )

    acsd = AnchorConditionedSoftDecomposition(
        ACSDConfig(
            vlm_model=args.vlm_model,
            object_top_k=int(args.acsd_top_k),
            verify_attempts=int(args.acsd_verify_attempts),
        )
    )

    scene_file = Path(args.navigation_data_path).expanduser().resolve() / f"{args.scene_name}.json.gz"
    if not scene_file.is_file():
        raise FileNotFoundError(f"scene data missing: {scene_file}")
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
    episode_matches = [e for e in scene_data["episode_by_sequence"] if int(e["episode_id"]) == int(args.episode_id)]
    if len(episode_matches) != 1:
        raise RuntimeError(f"expected one episode_id={args.episode_id}, found {len(episode_matches)}")
    episode = episode_matches[0]
    task_sequence = episode["task_sequence"]
    task_end = min(int(args.task_id) + int(args.num_tasks), len(task_sequence))

    sim_settings = OmegaConf.load(str(Path(args.sim_config).expanduser().resolve()))
    agent_settings = OmegaConf.load(str(Path(args.agent_config).expanduser().resolve()))
    sim_settings["scene"] = str(_resolve_scene_mesh(Path(args.hm3d_data_base_path).expanduser().resolve(), args.scene_name))
    abstract_sim = HabitatSimulator(sim_settings, agent_settings)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    path_finder = sim.pathfinder
    state = habitat_sim.AgentState()
    state.position = episode["start_position"]
    state.rotation = episode["start_rotation"]
    agent.set_state(state)

    pq3d = PQ3DModel(
        str(Path(args.pq3d_stage1_path).expanduser().resolve()),
        str(Path(args.pq3d_stage2_path).expanduser().resolve()),
        min_decision_num=int(args.decision_num_min),
    )
    pq3d.reset()

    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
    fog = np.zeros_like(top_down_map)
    area_thr = convert_meters_to_pixel(9, 512, sim)
    vis_dist = convert_meters_to_pixel(3.0, 512, sim)
    visited_frontier: set = set()
    task_records: List[Dict[str, Any]] = []

    try:
        for loop_tid in range(int(args.task_id), int(task_end)):
            task_type, task_idx = task_sequence[loop_tid]
            cur_task = episode_mapping[task_type][task_idx]
            sentence = _build_sentence(
                task_type,
                cur_task,
                goals_map,
                region_map,
                concise=(args.description_mode == "concise"),
            )
            goals = _goal_positions(cur_task, goals_map)
            decomp_t0 = time.perf_counter()
            decomposition = acsd.decompose_instruction(sentence)
            decomp_ms = (time.perf_counter() - decomp_t0) * 1000.0
            print(
                f"[ACSDTest][task-start] task={loop_tid} level={task_type} "
                f"target={decomposition['target_object']!r} room={decomposition.get('room_anchor', '')!r} "
                f"anchors={decomposition.get('object_anchors', [])!r} decomp_ms={decomp_ms:.1f}"
            )
            print(f"[ACSDTest][instruction] {sentence}")

            task_dir = _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
            total_steps = 0
            decision_num = 0
            goto_rgb: List[np.ndarray] = []
            goto_depth: List[np.ndarray] = []
            goto_states: List[Any] = []
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0.0
            task_end_reason = "max_steps"

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

                scan_rgb, scan_depth, scan_states, fog, total_steps = _capture_scan_frames(
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
                state_list.extend(scan_states)
                if total_steps >= int(args.max_steps):
                    break

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

                dec_dir = _ensure_dir(task_dir / f"dec_{decision_num:03d}")
                target_position, is_final = pq3d.decision(
                    color_list,
                    depth_list,
                    state_list,
                    frontiers,
                    sentence,
                    decision_num,
                    analysis_output_dir=str(dec_dir),
                )
                stage2 = acsd.load_stage2_decision(dec_dir / "stage2_decision.json")
                aux = dict(getattr(pq3d, "last_decision_aux", {}) or {})
                baseline = acsd.build_baseline_decision(
                    target_position=target_position,
                    is_object_decision=bool(is_final),
                    decision_aux=aux,
                    stage2=stage2,
                )
                print(
                    f"[ACSDTest][decision] task={loop_tid} dec={decision_num} "
                    f"baseline_type={baseline['type']} frontiers={len(frontiers)} pos={baseline['position']}"
                )

                process = acsd.process_decision(
                    instruction=sentence,
                    baseline_decision=baseline,
                    observation_context={"representation_manager": pq3d.representation_manager},
                    candidate_context={"stage2": stage2, "decomposition": decomposition, "output_dir": str(dec_dir / "acsd")},
                    episode_context={"scene_name": args.scene_name, "episode_id": args.episode_id, "task_id": loop_tid},
                    gt_context={"goal_positions": [g.tolist() for g in goals]},
                )

                corrected: Dict[str, Any]
                verification: Dict[str, Any]
                correction_reason: str
                correction_rejected = False

                if baseline["type"] == "frontier":
                    corrected = dict(process["corrected"])
                    verification = dict(process["verification"])
                    correction_reason = str(process["frontier_prior"]["reason"])
                    correction_applied = bool(process["frontier_prior"]["correction_applied"])
                    visited_frontier.add(tuple(np.round(np.asarray(corrected["position"], dtype=float), 1)))
                    goto_rgb, goto_depth, goto_states, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                        path_finder=path_finder,
                        agent=agent,
                        sim=sim,
                        top_down_map=top_down_map,
                        fog=fog,
                        vis_dist=vis_dist,
                        target=corrected["position"],
                        prev_agent_state=prev_agent_state,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                        episode_cum_distance=float(episode_cum_distance),
                    )
                else:
                    selected_candidates = list(process["object_rerank"]["selected_candidates"])
                    if len(selected_candidates) < 1:
                        raise RuntimeError("ACSD object rerank returned zero selected candidates")
                    verifier_records: List[Dict[str, Any]] = []
                    candidate = selected_candidates[0]
                    corrected = dict(process["corrected"])
                    goto_rgb, goto_depth, goto_states, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                        path_finder=path_finder,
                        agent=agent,
                        sim=sim,
                        top_down_map=top_down_map,
                        fog=fog,
                        vis_dist=vis_dist,
                        target=corrected["position"],
                        prev_agent_state=prev_agent_state,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                        episode_cum_distance=float(episode_cum_distance),
                    )
                    verification = {
                        "vlm_called": False,
                        "verified": False,
                        "confidence": 0.0,
                        "reason": "component4_removed_by_user_request",
                        "matched_target": False,
                        "matched_anchor": False,
                        "matched_relation": False,
                    }
                    verifier_records.append(
                        {
                            "attempt": 0,
                            "slot_index": int(candidate["slot_index"]),
                            "candidate": dict(candidate),
                            "verification": verification,
                        }
                    )
                    print(
                        f"[ACSDTest][component4-removed] task={loop_tid} dec={decision_num} "
                        f"slot={candidate['slot_index']} reason={verification['reason']}"
                    )
                    corrected["verifier_records"] = verifier_records
                    correction_applied = bool(
                        np.linalg.norm(
                            np.asarray(corrected["position"], dtype=float).reshape(3)
                            - np.asarray(baseline["position"], dtype=float).reshape(3)
                        )
                        > 1e-6
                    )
                    correction_rejected = False
                    correction_reason = (
                        "ACSD component 4 removed by user request; "
                        f"{process['object_rerank']['reason']}; "
                        f"selected_slot={int(corrected['slot_index'])}"
                    )
                    task_end_reason = "final_decision"

                compare = acsd.build_compare_record(
                    episode_id=args.episode_id,
                    step_id=decision_num,
                    baseline_position=baseline["position"],
                    corrected_position=corrected["position"],
                    goal_positions=[g.tolist() for g in goals],
                    threshold_m=1.0,
                )
                print(
                    acsd.format_call_log(
                        episode_id=args.episode_id,
                        step_id=decision_num,
                        decomposition=decomposition,
                        baseline=baseline,
                        corrected=corrected,
                        correction_applied=bool(correction_applied),
                        correction_reason=correction_reason,
                        verification=verification,
                    )
                )
                print(acsd.format_compare_log(compare))
                acsd.update_summary_from_decision(
                    correction_applied=bool(correction_applied),
                    correction_rejected=bool(correction_rejected),
                    verification=verification,
                )
                decision_record = {
                    "task_id": int(loop_tid),
                    "decision_num": int(decision_num),
                    "instruction": sentence,
                    "decomposition": decomposition,
                    "baseline": baseline,
                    "corrected": corrected,
                    "verification": verification,
                    "compare": compare,
                    "process": process,
                    "correction_applied": bool(correction_applied),
                    "correction_rejected": bool(correction_rejected),
                    "correction_reason": correction_reason,
                }
                _write_json(dec_dir / "acsd_decision.json", decision_record)

                decision_num += 1
                if baseline["type"] == "object" and corrected["type"] == "object":
                    break

            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for gid in cur_task["target_object_ids"] for vp in goals_map[gid].get("view_points", [])]
            sp = habitat_sim.MultiGoalShortestPath()
            sp.requested_start = sub_episode_start_position
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
            task_summary = {
                "scene_name": args.scene_name,
                "episode_id": int(args.episode_id),
                "task_id": int(loop_tid),
                "task_level": task_type,
                "instruction": sentence,
                "steps_total": int(total_steps),
                "decisions": int(decision_num),
                "end_reason": task_end_reason,
                "sr": float(sr),
                "spl": float(spl),
                "start_goal_geo": float(start_end_geo),
                "end_goal_geo": float(end_geo),
                "goal_positions": [g.tolist() for g in goals],
            }
            task_records.append(task_summary)
            _write_json(task_dir / "task_summary.json", task_summary)
            print(
                f"[ACSDTest][task-summary] task={loop_tid} level={task_type} steps={total_steps} "
                f"decisions={decision_num} end_reason={task_end_reason} SR={sr:.1f} SPL={spl:.4f}"
            )

    finally:
        sim.close()

    _write_json(out_root / "run_summary.json", {"tasks": task_records, "acsd_summary": acsd.summary})
    print(acsd.format_summary_log())


if __name__ == "__main__":
    main()
