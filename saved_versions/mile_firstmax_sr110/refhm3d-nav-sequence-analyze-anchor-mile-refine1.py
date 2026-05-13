from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
import torch
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from anchor_nav.mile import (
    MileConfig,
    MileRejectedError,
    correct_final_decision_with_mile,
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


def _tqdm_print(msg: str) -> None:
    tqdm.write(msg, file=sys.stdout)


class _TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = log_dir / f"refhm3d-nav-sequence-analyze-anchor-mile-refine1-{ts}-pid{os.getpid()}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[MileRefine1] logging enabled -> {log_path.resolve()}")

    def _cleanup() -> None:
        try:
            print(f"[MileRefine1] run finished, log saved -> {log_path.resolve()}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def set_reproducibility_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _tqdm_print(f"[mile-refine1] reproducibility_seed={seed}")


def _object_slot_info(rep: Any, prev_object_count: int) -> Dict[str, Any]:
    cur_count = int(np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0])
    prev = int(prev_object_count)
    if cur_count < prev:
        raise RuntimeError(f"object slot count shrank from {prev} to {cur_count}")
    return {
        "prev_object_count": prev,
        "cur_object_count": cur_count,
        "new_object_slots": [int(x) for x in range(prev, cur_count)],
    }


class FollowerNavigationError(RuntimeError):
    def __init__(self, info: Dict[str, Any]):
        self.info = info
        msg = (
            f"{info.get('error_type', 'FollowerNavigationError')} while following "
            f"target={info.get('raw_target')} snapped={info.get('snapped_target')} "
            f"path_found={info.get('shortest_path_found')} geo={info.get('shortest_path_geodesic_distance')}"
        )
        super().__init__(msg)


def _path_diagnostics(pf: Any, start: np.ndarray, end: np.ndarray) -> Dict[str, Any]:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=float).reshape(3)
    path.requested_end = np.asarray(end, dtype=float).reshape(3)
    found = bool(pf.find_path(path))
    return {
        "shortest_path_found": found,
        "shortest_path_geodesic_distance": float(path.geodesic_distance) if found else float("inf"),
    }


def _greedy_follower_precheck(
    *,
    pf: Any,
    agent: Any,
    raw_target: Sequence[float],
) -> Dict[str, Any]:
    start_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    target_raw = np.asarray(raw_target, dtype=float).reshape(3)
    agent_island = int(pf.get_island(agent.get_state().position))
    target_nav = pf.snap_point(point=target_raw, island_index=agent_island)
    target_nav_arr = np.asarray(target_nav, dtype=float).reshape(3)
    info: Dict[str, Any] = {
        "raw_target": target_raw.tolist(),
        "snapped_target": target_nav_arr.tolist(),
        "agent_island": int(agent_island),
        "start_position": start_position.tolist(),
        "agent_is_navigable": bool(pf.is_navigable(start_position)),
        "target_is_navigable": bool(pf.is_navigable(target_nav_arr)),
    }
    info.update(_path_diagnostics(pf, start_position, target_nav_arr))
    try:
        follower = habitat_sim.GreedyGeodesicFollower(
            pf,
            agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        actions = follower.find_path(target_nav)
        if actions is None:
            info.update(
                {
                    "precheck_ok": False,
                    "error_type": "GreedyGeodesicFollowerReturnedNone",
                    "error_message": "GreedyGeodesicFollower returned None",
                }
            )
        else:
            non_stop_actions = [a for a in actions if a]
            info.update(
                {
                    "precheck_ok": True,
                    "path_action_count": int(len(non_stop_actions)),
                }
            )
    except Exception as exc:
        info.update(
            {
                "precheck_ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return info


def _select_followable_navigation_target(
    *,
    pf: Any,
    agent: Any,
    raw_target: Sequence[float],
    repair_radii_m: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
    candidates_per_radius: int = 16,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target = np.asarray(raw_target, dtype=float).reshape(3)
    base_pre = _greedy_follower_precheck(pf=pf, agent=agent, raw_target=target)
    records: List[Dict[str, Any]] = [
        {
            "candidate_role": "original",
            "raw_target": target.tolist(),
            "radius_m": 0.0,
            "angle_rad": None,
            "precheck": base_pre,
            "selected": bool(base_pre.get("precheck_ok", False)),
        }
    ]
    if bool(base_pre.get("precheck_ok", False)):
        return target.copy(), {
            "called": True,
            "adjustment_applied": False,
            "selected_role": "original",
            "selected_target": target.tolist(),
            "original_precheck_ok": True,
            "records": records,
        }

    agent_island = int(pf.get_island(agent.get_state().position))
    followable: List[Tuple[float, float, int, np.ndarray, Dict[str, Any]]] = []
    seen = {tuple(np.round(np.asarray(base_pre.get("snapped_target", target), dtype=float), 3).tolist())}
    n = max(4, int(candidates_per_radius))
    for radius in repair_radii_m:
        r = float(radius)
        if r <= 0:
            continue
        for i in range(n):
            theta = 2.0 * np.pi * float(i) / float(n)
            raw = target + np.array([r * np.cos(theta), 0.0, r * np.sin(theta)], dtype=float)
            snapped = np.asarray(pf.snap_point(point=raw, island_index=agent_island), dtype=float).reshape(3)
            key = tuple(np.round(snapped, 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            same_island = bool(np.all(np.isfinite(snapped)) and pf.is_navigable(snapped) and int(pf.get_island(snapped)) == agent_island)
            rec: Dict[str, Any] = {
                "candidate_role": "local_repair",
                "candidate_index": int(i),
                "radius_m": float(r),
                "angle_rad": float(theta),
                "raw_target": raw.tolist(),
                "snapped_target": snapped.tolist() if np.all(np.isfinite(snapped)) else None,
                "snap_distance_m": float(np.linalg.norm(snapped - raw)) if np.all(np.isfinite(snapped)) else float("inf"),
                "same_island": bool(same_island),
                "l2_to_original_m": float(np.linalg.norm(snapped - target)) if np.all(np.isfinite(snapped)) else float("inf"),
            }
            if not same_island:
                rec["precheck"] = {"precheck_ok": False, "error_type": "not_same_navigable_island"}
                rec["selected"] = False
                records.append(rec)
                continue
            pre = _greedy_follower_precheck(pf=pf, agent=agent, raw_target=snapped)
            rec["precheck"] = pre
            rec["selected"] = False
            records.append(rec)
            if bool(pre.get("precheck_ok", False)):
                followable.append(
                    (
                        float(rec["l2_to_original_m"]),
                        float(pre.get("shortest_path_geodesic_distance", float("inf"))),
                        int(pre.get("path_action_count", 10**9)),
                        snapped.copy(),
                        rec,
                    )
                )

    if not followable:
        return target.copy(), {
            "called": True,
            "adjustment_applied": False,
            "selected_role": "original_unfollowable",
            "selected_target": target.tolist(),
            "original_precheck_ok": False,
            "rejected_reason": "no_local_followable_candidate",
            "records": records,
        }

    followable.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, selected, selected_rec = followable[0]
    selected_rec["selected"] = True
    return selected.copy(), {
        "called": True,
        "adjustment_applied": True,
        "selected_role": "local_repair",
        "selected_target": selected.tolist(),
        "original_precheck_ok": False,
        "selected_l2_to_original_m": float(np.linalg.norm(selected - target)),
        "selected_radius_m": float(selected_rec.get("radius_m", 0.0)),
        "selected_angle_rad": selected_rec.get("angle_rad"),
        "followable_candidate_count": int(len(followable)),
        "records": records,
    }


def _apply_followability_filter(
    *,
    mile_info: Dict[str, Any],
    baseline_target: np.ndarray,
    pf: Any,
    agent: Any,
    cfg: MileConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    baseline = np.asarray(baseline_target, dtype=float).reshape(3)
    updated = dict(mile_info)
    if not bool(updated.get("correction_applied", False)):
        updated["followability_policy"] = "noop_no_extra_follower_precheck"
        return baseline.copy(), updated
    if not bool(getattr(cfg, "enable_vvd_replacement", False)):
        updated.update(
            {
                "target_source": "vvd_diagnostic_explicit_keep_baseline",
                "viewpoint_correction_applied": False,
                "correction_applied": False,
                "corrected_target_xyz": baseline.tolist(),
                "rejected_reason": "vvd_replacement_disabled_followability_repair_only",
                "followability_policy": "vvd_diagnostic_only_then_navigation_target_repair",
            }
        )
        return baseline.copy(), updated
    updated["baseline_followability_precheck"] = _greedy_follower_precheck(
        pf=pf,
        agent=agent,
        raw_target=baseline,
    )

    visibility = dict(updated.get("visibility", {}) or {})
    records = list(visibility.get("visibility_records", []) or [])
    min_score = max(0.0, float(visibility.get("min_visibility_score", 0.0) or 0.0))
    tie_epsilon = max(0.0, float(getattr(cfg, "visibility_tie_epsilon", 0.0)))
    prechecks: List[Dict[str, Any]] = []
    followable: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]] = []
    sorted_records = sorted(
        records,
        key=lambda x: (-float(x.get("visibility_score", 0.0)), int(x.get("accepted_rank", x.get("candidate_index", 0)))),
    )
    for rec in sorted_records:
        score = float(rec.get("visibility_score", 0.0))
        if score <= min_score:
            continue
        vp = rec.get("viewpoint_xyz")
        if vp is None:
            continue
        target = np.asarray(vp, dtype=float).reshape(3)
        pre = _greedy_follower_precheck(pf=pf, agent=agent, raw_target=target)
        pre.update(
            {
                "candidate_index": int(rec.get("candidate_index", -1)),
                "accepted_rank": int(rec.get("accepted_rank", -1)),
                "visibility_score": float(score),
                "agent_l2_distance_m": float(rec.get("agent_l2_distance_m", float("inf"))),
                "baseline_l2_distance_m": float(np.linalg.norm(target - baseline)),
            }
        )
        prechecks.append(pre)
        if bool(pre.get("precheck_ok", False)):
            followable.append((rec, pre, target))

    updated["followability_prechecks"] = prechecks
    if not followable:
        updated.update(
            {
                "target_source": "vvd_rejected_no_followable_candidate_explicit_keep_baseline",
                "viewpoint_correction_applied": False,
                "correction_applied": False,
                "corrected_target_xyz": baseline.tolist(),
                "rejected_reason": "no_followable_vvd_candidate",
            }
        )
        visibility["followability_selected_candidate_index"] = None
        visibility["followability_selected_visibility_score"] = None
        updated["visibility"] = visibility
        return baseline.copy(), updated

    selected_rec, selected_pre, selected_target = followable[0]
    max_followable_visibility = float(max(float(item[0].get("visibility_score", 0.0)) for item in followable))
    baseline_followable_items = [
        item for item in followable
        if bool(item[0].get("is_baseline_candidate", False))
        and float(item[0].get("visibility_score", 0.0)) > min_score
    ]
    if bool(getattr(cfg, "prefer_visible_baseline", True)) and baseline_followable_items:
        baseline_rec, baseline_pre, baseline_selected_target = baseline_followable_items[0]
        updated.update(
            {
                "target_source": "baseline_positive_visibility_explicit_keep_baseline",
                "viewpoint_correction_applied": False,
                "correction_applied": False,
                "corrected_target_xyz": baseline.tolist(),
                "followability_selected_precheck": baseline_pre,
                "followability_selection_policy": "msgnav_first_max_followable_visibility_no_baseline_threshold",
                "followability_max_visibility_score": float(max_followable_visibility),
                "followability_baseline_visibility_score": float(baseline_rec.get("visibility_score", 0.0)),
                "followability_candidate_count": int(len(followable)),
                "rejected_reason": "baseline_followable_and_positive_visibility",
            }
        )
        visibility["followability_selected_candidate_index"] = -1
        visibility["followability_selected_accepted_rank"] = int(baseline_rec.get("accepted_rank", -1))
        visibility["followability_selected_visibility_score"] = float(baseline_rec.get("visibility_score", 0.0))
        visibility["followability_selected_viewpoint_xyz"] = baseline_selected_target.tolist()
        visibility["followability_selection_policy"] = "msgnav_first_max_followable_visibility_no_baseline_threshold"
        visibility["followability_max_visibility_score"] = float(max_followable_visibility)
        visibility["followability_baseline_visibility_score"] = float(baseline_rec.get("visibility_score", 0.0))
        visibility["followability_candidate_count"] = int(len(followable))
        updated["visibility"] = visibility
        return baseline.copy(), updated
    if bool(selected_rec.get("is_baseline_candidate", False)):
        updated.update(
            {
                "target_source": "baseline_visibility_candidate_explicit_keep_baseline",
                "viewpoint_correction_applied": False,
                "correction_applied": False,
                "corrected_target_xyz": baseline.tolist(),
                "followability_selected_precheck": selected_pre,
                "followability_selection_policy": "msgnav_first_max_followable_visibility_no_baseline_threshold",
                "followability_max_visibility_score": float(max_followable_visibility),
                "followability_visibility_tie_epsilon": float(tie_epsilon),
                "followability_candidate_count": int(len(followable)),
                "rejected_reason": "baseline_candidate_selected_by_visibility_efficiency",
            }
        )
        visibility["followability_selected_candidate_index"] = -1
        visibility["followability_selected_accepted_rank"] = int(selected_rec.get("accepted_rank", -1))
        visibility["followability_selected_visibility_score"] = float(selected_rec.get("visibility_score", 0.0))
        visibility["followability_selected_viewpoint_xyz"] = selected_target.tolist()
        visibility["followability_selection_policy"] = "msgnav_first_max_followable_visibility_no_baseline_threshold"
        visibility["followability_max_visibility_score"] = float(max_followable_visibility)
        visibility["followability_visibility_tie_epsilon"] = float(tie_epsilon)
        visibility["followability_candidate_count"] = int(len(followable))
        updated["visibility"] = visibility
        return baseline.copy(), updated

    correction_applied = bool(np.linalg.norm(selected_target - baseline) > 1e-6)
    updated.update(
        {
            "target_source": "baseline_vvd_viewpoint_followability_checked",
            "viewpoint_correction_applied": True,
            "correction_applied": bool(correction_applied),
            "corrected_target_xyz": selected_target.tolist(),
            "followability_selected_precheck": selected_pre,
            "followability_selection_policy": "msgnav_first_max_followable_visibility_no_baseline_threshold",
            "followability_max_visibility_score": float(max_followable_visibility),
            "followability_visibility_tie_epsilon": float(tie_epsilon),
            "followability_candidate_count": int(len(followable)),
        }
    )
    visibility["followability_selected_candidate_index"] = int(selected_rec.get("candidate_index", -1))
    visibility["followability_selected_accepted_rank"] = int(selected_rec.get("accepted_rank", -1))
    visibility["followability_selected_visibility_score"] = float(selected_rec.get("visibility_score", 0.0))
    visibility["followability_selected_viewpoint_xyz"] = selected_target.tolist()
    visibility["followability_selection_policy"] = "msgnav_first_max_followable_visibility_no_baseline_threshold"
    visibility["followability_max_visibility_score"] = float(max_followable_visibility)
    visibility["followability_visibility_tie_epsilon"] = float(tie_epsilon)
    visibility["followability_candidate_count"] = int(len(followable))
    updated["visibility"] = visibility
    return selected_target.copy(), updated


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
        if concise:
            desc = region_info["concise_description"]
        else:
            desc = region_info["detailed_description"]
        return f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {desc}"
    if task_type == "instance":
        inst = goals_map[cur_task["instance_id"]]
        if concise:
            return inst["annot_unique_concise_description"]
        return inst["annot_unique_detailed_description"]
    raise ValueError(f"unknown task_type={task_type}")


_LANGUAGE_GOAL_FIELDS = (
    "object_id",
    "object_category",
    "annot_unique_concise_description",
    "annot_unique_detailed_description",
    "annot_unique_normal_description",
    "annot_appearance_description",
)


def _language_only_goals_map(goals: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        g["object_id"]: {k: g[k] for k in _LANGUAGE_GOAL_FIELDS if k in g}
        for g in goals
    }


def _eval_goal_bundle(
    cur_task: Dict[str, Any],
    eval_goals_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[List[float]], str]:
    goal_ids = list(cur_task.get("target_object_ids", []))
    goals = [eval_goals_map[x] for x in goal_ids if x in eval_goals_map]
    goal_positions = _goal_positions(goals)
    view_points = _view_points(goals)
    goal_category = goals[0]["object_category"] if goals else cur_task.get("object_category", "")
    return goals, goal_positions, view_points, goal_category


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
    return scan_rgb, scan_depth, scan_state, fog, total_steps


def _follow_target(
    *,
    pf: Any,
    agent: Any,
    sim: Any,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    vis_dist: int,
    used_target: np.ndarray,
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float, Dict[str, Any]]:
    follow_info: Dict[str, Any] = {}
    try:
        start_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
        raw_target = np.asarray(used_target, dtype=float).reshape(3)
        agent_island = int(pf.get_island(agent.get_state().position))
        target_nav = pf.snap_point(point=raw_target, island_index=agent_island)
        target_nav_arr = np.asarray(target_nav, dtype=float).reshape(3)
        follow_info = {
            "raw_target": raw_target.tolist(),
            "snapped_target": target_nav_arr.tolist(),
            "agent_island": int(agent_island),
            "start_position": start_position.tolist(),
            "agent_is_navigable": bool(pf.is_navigable(start_position)),
            "target_is_navigable": bool(pf.is_navigable(target_nav_arr)),
        }
        follower = habitat_sim.GreedyGeodesicFollower(
            pf,
            agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        actions = follower.find_path(target_nav)
    except FollowerNavigationError:
        raise
    except Exception as exc:
        follow_info.update({"error_type": type(exc).__name__, "error_message": str(exc)})
        if "start_position" in follow_info and "snapped_target" in follow_info:
            follow_info.update(
                _path_diagnostics(
                    pf,
                    np.asarray(follow_info["start_position"], dtype=float),
                    np.asarray(follow_info["snapped_target"], dtype=float),
                )
            )
        raise FollowerNavigationError(follow_info) from exc
    if actions is None:
        follow_info.update(
            {
                "error_type": "GreedyGeodesicFollowerReturnedNone",
                "error_message": "GreedyGeodesicFollower returned None",
            }
        )
        raise FollowerNavigationError(follow_info)

    goto_rgb: List[np.ndarray] = []
    goto_depth: List[np.ndarray] = []
    goto_state: List[Any] = []
    executed_actions: List[str] = []
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
        executed_actions.append(str(a))
        episode_cum_distance += float(np.linalg.norm(st2.position - prev_agent_state.position))
        prev_agent_state = st2
        total_steps += 1

    end_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    non_stop_actions = [a for a in actions if a]
    follow_info.update(
        {
            "end_position": end_position.tolist(),
            "path_action_count": int(len(non_stop_actions)),
            "executed_action_count": int(len(executed_actions)),
            "truncated_by_max_steps": bool(len(executed_actions) < len(non_stop_actions)),
        }
    )
    return goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, float(episode_cum_distance), follow_info


def _goal_positions(goals: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for g in goals:
        pos = g.get("position", [])
        if isinstance(pos, list) and len(pos) >= 3:
            out.append(np.asarray(pos, dtype=float).reshape(3))
    return out


def _view_points(goals: Sequence[Dict[str, Any]]) -> List[List[float]]:
    return [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]


def _geo_dist_to_viewpoints(pf: Any, start_pos: Sequence[float], view_points: Sequence[Sequence[float]]) -> float:
    if len(view_points) == 0:
        return float("inf")
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = start_pos
    path.requested_ends = list(view_points)
    return float(path.geodesic_distance) if pf.find_path(path) else float("inf")


def _frontier_visit_key(point: Sequence[float], resolution_m: float = 0.1) -> Tuple[float, float, float]:
    arr = np.asarray(point, dtype=float).reshape(3)
    return tuple(np.round(arr, 1).astype(float).tolist())


def _nearest_goal_dist(point: Optional[np.ndarray], goals: Sequence[np.ndarray]) -> float:
    if point is None or len(goals) == 0:
        return float("inf")
    p = np.asarray(point, dtype=float).reshape(3)
    return float(min(float(np.linalg.norm(p - g)) for g in goals))


def build_effectiveness_record(
    *,
    baseline_target_xyz: np.ndarray,
    corrected_target_xyz: np.ndarray,
    goal_positions_xyz: Sequence[np.ndarray],
    threshold_m: float = 1.0,
) -> Dict[str, Any]:
    baseline_dist = _nearest_goal_dist(np.asarray(baseline_target_xyz, dtype=float).reshape(3), goal_positions_xyz)
    corrected_dist = _nearest_goal_dist(np.asarray(corrected_target_xyz, dtype=float).reshape(3), goal_positions_xyz)
    baseline_ok = bool(baseline_dist <= float(threshold_m))
    corrected_ok = bool(corrected_dist <= float(threshold_m))
    return {
        "threshold_m": float(threshold_m),
        "baseline_nearest_goal_dist_m": float(baseline_dist),
        "corrected_nearest_goal_dist_m": float(corrected_dist),
        "baseline_in_1m": bool(baseline_ok),
        "corrected_in_1m": bool(corrected_ok),
        "case": f"{1 if baseline_ok else 0}{1 if corrected_ok else 0}",
        "evaluation_only": True,
    }


def _augment_viewpoint_effectiveness(
    effectiveness: Dict[str, Any],
    *,
    pf: Any,
    baseline_target_xyz: np.ndarray,
    corrected_target_xyz: np.ndarray,
    view_points: Sequence[Sequence[float]],
    threshold_m: float,
) -> Dict[str, Any]:
    baseline_geo = _geo_dist_to_viewpoints(pf, baseline_target_xyz, view_points)
    corrected_geo = _geo_dist_to_viewpoints(pf, corrected_target_xyz, view_points)
    baseline_ok = bool(baseline_geo <= float(threshold_m))
    corrected_ok = bool(corrected_geo <= float(threshold_m))
    viewpoint_case = f"{1 if baseline_ok else 0}{1 if corrected_ok else 0}"
    effectiveness.update(
        {
            "case_metric": "gt_viewpoint_geodesic",
            "object_l2_case": str(effectiveness.get("case")),
            "baseline_nearest_gt_viewpoint_geo_m": float(baseline_geo),
            "corrected_nearest_gt_viewpoint_geo_m": float(corrected_geo),
            "baseline_gt_viewpoint_in_1m": bool(baseline_ok),
            "corrected_gt_viewpoint_in_1m": bool(corrected_ok),
            "viewpoint_case": viewpoint_case,
            "case": viewpoint_case,
        }
    )
    return effectiveness


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _sequence_compute_metric_results(result_dict: Dict[str, Any]) -> None:
    rows = result_dict.get("sequence", [])
    if not rows:
        _tqdm_print("[Metrics] sequence count=0")
        return
    avg_sr = float(np.mean([float(x.get("sr", 0.0)) for x in rows]))
    avg_spl = float(np.mean([float(x.get("spl", 0.0)) for x in rows]))
    avg_time = float(np.mean([float(x.get("task_time_sec", 0.0)) for x in rows]))
    _tqdm_print(f"[Metrics] sequence count={len(rows)}, avg_sr={avg_sr:.6f}, avg_spl={avg_spl:.6f}, avg_task_time_sec={avg_time:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser("RefHM3D anchor mile refine1 batch analysis")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=0.2)
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
    parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
    parser.add_argument("--sim_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    parser.add_argument("--agent_config", type=str, default=str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))
    parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/mile"))
    parser.add_argument("--task_levels", type=str, default="object,room,region,instance")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--effectiveness_threshold_m", type=float, default=1.0)
    parser.add_argument("--mile_decision_radius_m", type=float, default=0.75)
    parser.add_argument("--mile_candidate_radii_m", type=str, default="0.5,0.75")
    parser.add_argument("--mile_candidate_view_count", type=int, default=20)
    parser.add_argument("--mile_enable_vvd_replacement", action="store_true")
    parser.add_argument("--mile_disable_visible_baseline_guard", action="store_true")
    parser.add_argument("--mile_camera_height_m", type=float, default=1.50)
    parser.add_argument("--mile_max_snap_distance_m", type=float, default=0.60)
    parser.add_argument("--mile_scene_sample_count", type=int, default=0)
    parser.add_argument("--mile_max_ray_sample_count", type=int, default=1000)
    parser.add_argument("--mile_occlusion_radius_m", type=float, default=0.05)
    parser.add_argument("--mile_min_visibility_score", type=float, default=0.0)
    parser.add_argument("--mile_visibility_tie_epsilon", type=float, default=0.0)
    parser.add_argument("--mile_apply_task_levels", type=str, default="object,room,region,instance")
    parser.add_argument("--mile_apply_non_final_object_decisions", action="store_true")
    parser.add_argument("--mile_enable_navigation_target_repair", action="store_true")
    parser.add_argument("--max_eval_tasks", type=int, default=0)
    parser.add_argument("--quiet_nav_steps", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    output_log_dir = Path(args.output_log_dir).expanduser().resolve()
    _setup_run_logging(output_log_dir)
    set_reproducibility_seed(args.seed)
    enabled_task_levels = {x.strip() for x in str(args.task_levels).split(",") if x.strip()}
    if not enabled_task_levels:
        raise RuntimeError("--task_levels resolved to empty set")
    mile_apply_task_levels = {x.strip() for x in str(args.mile_apply_task_levels).split(",") if x.strip()}
    if not mile_apply_task_levels:
        raise RuntimeError("--mile_apply_task_levels resolved to empty set")
    candidate_radii = tuple(float(x.strip()) for x in str(args.mile_candidate_radii_m).split(",") if x.strip())
    if len(candidate_radii) == 0 or any(x <= 0.0 for x in candidate_radii):
        raise RuntimeError(f"--mile_candidate_radii_m must contain positive radii, got {args.mile_candidate_radii_m!r}")
    cfg = MileConfig(
        decision_radius_m=float(args.mile_decision_radius_m),
        candidate_radii_m=candidate_radii,
        candidate_view_count=int(args.mile_candidate_view_count),
        enable_vvd_replacement=bool(args.mile_enable_vvd_replacement),
        prefer_visible_baseline=not bool(args.mile_disable_visible_baseline_guard),
        camera_height_m=float(args.mile_camera_height_m),
        max_snap_distance_m=float(args.mile_max_snap_distance_m),
        scene_sample_count=int(args.mile_scene_sample_count),
        max_ray_sample_count=int(args.mile_max_ray_sample_count),
        occlusion_radius_m=float(args.mile_occlusion_radius_m),
        min_visibility_score=float(args.mile_min_visibility_score),
        visibility_tie_epsilon=float(args.mile_visibility_tie_epsilon),
    )
    _tqdm_print(
        f"[MileRefine1] cfg levels={sorted(enabled_task_levels)} module=lastmile_vvd_only "
        f"radius={cfg.decision_radius_m} candidate_radii={list(cfg.candidate_radii_m)} "
        f"view_count={cfg.candidate_view_count} "
        f"camera_height={cfg.camera_height_m} max_snap={cfg.max_snap_distance_m} "
        f"scene_sample_count={cfg.scene_sample_count} occlusion_radius={cfg.occlusion_radius_m} "
        f"max_ray_samples={cfg.max_ray_sample_count} min_visibility={cfg.min_visibility_score} "
        f"visibility_tie_epsilon={cfg.visibility_tie_epsilon} "
        f"selection_policy=msgnav_first_max_followable_visibility_no_baseline_threshold "
        f"enable_vvd_replacement={bool(cfg.enable_vvd_replacement)} "
        f"prefer_visible_baseline={bool(cfg.prefer_visible_baseline)} "
        f"mile_apply_task_levels={sorted(mile_apply_task_levels)} "
        f"mile_apply_non_final_object_decisions={bool(args.mile_apply_non_final_object_decisions)} "
        f"mile_enable_navigation_target_repair={bool(args.mile_enable_navigation_target_repair)} "
        f"start_ratio={args.start_ratio} end_ratio={args.end_ratio} max_eval_tasks={args.max_eval_tasks}"
    )

    navigation_data_root = Path(args.navigation_data_path).expanduser().resolve()
    scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
    if not scene_data_paths:
        raise FileNotFoundError(f"No *.json.gz found under navigation_data_path={navigation_data_root}")
    scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]

    out_name = f"refhm3d_seq_mile_refine1_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_mile_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
    if args.concise_description:
        out_name = f"refhm3d_seq_mile_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
        eff_name = f"refhm3d_seq_mile_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    output_path = output_log_dir / out_name
    effectiveness_path = output_log_dir / eff_name

    if output_path.exists():
        result_dict = json.load(open(output_path, "r", encoding="utf-8"))
        existing_tasks = {
            "_".join(
                [
                    r["scene_name"],
                    r["navigation_type"],
                    str(r["episode_id"]),
                    str(r["task_id"]),
                    str(r.get("task_level", "")),
                ]
            )
            for rows in result_dict.values()
            for r in rows
        }
    else:
        result_dict = {"sequence": []}
        existing_tasks = set()

    if effectiveness_path.exists():
        effectiveness_dict = json.load(open(effectiveness_path, "r", encoding="utf-8"))
    else:
        effectiveness_dict = {
            "records": [],
            "case_counts": {"00": 0, "01": 0, "10": 0, "11": 0},
            "module_status_counts": {
                "mile_applied": 0,
                "mile_kept_baseline": 0,
                "mile_rejected": 0,
                "mile_error": 0,
                "follower_error": 0,
            },
        }

    pq3d = PQ3DModel(
        str(Path(args.pq3d_stage1_path).expanduser().resolve()),
        str(Path(args.pq3d_stage2_path).expanduser().resolve()),
        min_decision_num=int(args.decision_num_min),
    )

    max_eval_tasks = max(0, int(args.max_eval_tasks))

    for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
        if max_eval_tasks and len(result_dict.get("sequence", [])) >= max_eval_tasks:
            break
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
        eval_goals_map = {x["object_id"]: x for x in scene_data["goals"]}
        language_goals_map = _language_only_goals_map(scene_data["goals"])

        for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
            if max_eval_tasks and len(result_dict.get("sequence", [])) >= max_eval_tasks:
                break
            pq3d.reset()
            decision_num = 0
            visited_frontier: set = set()
            episode_id = cur_episode["episode_id"]
            navigation_type = cur_episode["navigation_type"]

            sim_settings = OmegaConf.load(str(Path(args.sim_config).expanduser().resolve()))
            agent_settings = OmegaConf.load(str(Path(args.agent_config).expanduser().resolve()))
            sim_settings["scene"] = str(_resolve_scene_mesh(Path(args.hm3d_data_base_path).expanduser().resolve(), scene_name))
            abstract_sim = HabitatSimulator(sim_settings, agent_settings)
            sim = abstract_sim.simulator
            agent = abstract_sim.agent
            pf = sim.pathfinder
            st = habitat_sim.AgentState()
            st.position = cur_episode["start_position"]
            st.rotation = cur_episode["start_rotation"]
            agent.set_state(st)

            top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
            fog = np.zeros_like(top_down_map)
            area_thr = convert_meters_to_pixel(9, 512, sim)
            vis_dist = convert_meters_to_pixel(3.0, 512, sim)
            out_episode_dir = output_log_dir / "process" / f"scene={scene_name}" / f"episode={episode_id}"

            for idx, task_ref in enumerate(cur_episode["task_sequence"]):
                if max_eval_tasks and len(result_dict.get("sequence", [])) >= max_eval_tasks:
                    break
                task_type, task_idx = task_ref
                if task_type not in enabled_task_levels:
                    continue
                task_key = "_".join([scene_name, navigation_type, str(episode_id), str(idx), str(task_type)])
                if task_key in existing_tasks:
                    _tqdm_print(f"[mile-refine1][skip] already processed {task_key}")
                    continue
                task_t0 = time.perf_counter()
                cur_task = episode_mapping[task_type][task_idx]
                sentence = _build_sentence(task_type, cur_task, language_goals_map, region_map, concise=bool(args.concise_description))
                eval_goals: Optional[List[Dict[str, Any]]] = None
                goal_positions: List[np.ndarray] = []
                view_points: List[List[float]] = []
                goal_category = cur_task.get("object_category", "")
                out_task = out_episode_dir / f"task={idx}"
                out_task.mkdir(parents=True, exist_ok=True)
                _tqdm_print(f"[mile-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} level={task_type} sentence={sentence!r}")

                total_steps = 0
                prev_agent_state = agent.get_state()
                sub_episode_start_position = prev_agent_state.position
                episode_cum_distance = 0.0
                goto_rgb: List[np.ndarray] = []
                goto_depth: List[np.ndarray] = []
                goto_state: List[Any] = []
                baseline_final_target: Optional[np.ndarray] = None
                corrected_final_target: Optional[np.ndarray] = None
                final_mile_info: Dict[str, Any] = {"mile_called": False}
                final_effectiveness: Optional[Dict[str, Any]] = None
                final_follow_info: Optional[Dict[str, Any]] = None
                task_effective_logs: List[Dict[str, Any]] = []
                task_end_reason = "max_steps"

                while total_steps < int(args.max_steps):
                    color_list: List[np.ndarray] = []
                    depth_list: List[np.ndarray] = []
                    state_list: List[Any] = []
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

                    st_now = agent.get_state()
                    fw = detect_frontier_waypoints(
                        top_down_map,
                        fog,
                        area_thr,
                        xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1],
                        enable_visualization=False,
                    )
                    raw_frontiers = [] if len(fw) == 0 else list(pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim))
                    frontiers: List[np.ndarray] = []
                    filtered_frontiers: List[Dict[str, Any]] = []
                    for w in raw_frontiers:
                        key = _frontier_visit_key(w)
                        if key in visited_frontier:
                            filtered_frontiers.append({"key": list(key), "point": np.asarray(w, dtype=float).reshape(3).tolist()})
                            continue
                        frontiers.append(w)
                    frontier_filter_info = {
                        "raw_frontier_count": int(len(raw_frontiers)),
                        "frontier_count_after_visited_filter": int(len(frontiers)),
                        "visited_frontier_count": int(len(visited_frontier)),
                        "filtered_as_visited": filtered_frontiers,
                    }

                    dec_dir = out_task / f"dec_{decision_num:03d}"
                    dec_dir.mkdir(parents=True, exist_ok=True)
                    prev_object_count = int(
                        np.asarray(getattr(pq3d.representation_manager, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0]
                    )
                    target, is_final = pq3d.decision(
                        color_list,
                        depth_list,
                        state_list,
                        frontiers,
                        sentence,
                        decision_num,
                    )
                    register_info = _object_slot_info(pq3d.representation_manager, prev_object_count)
                    baseline_target = np.asarray(target, dtype=float).reshape(3)
                    aux = dict(getattr(pq3d, "last_decision_aux", {}) or {})
                    if not args.quiet_nav_steps:
                        _tqdm_print(
                            f"[mile-refine1][decision] scene={scene_name} ep={episode_id} task={idx} "
                            f"dec={decision_num} final={bool(is_final)} frontiers={len(frontiers)}/{len(raw_frontiers)} "
                            f"baseline_target={baseline_target.tolist()}"
                        )

                    corrected_target = baseline_target.copy()
                    mile_info: Dict[str, Any] = {"mile_called": False}
                    effectiveness: Optional[Dict[str, Any]] = None
                    is_object_decision = bool(aux.get("is_object_decision", False))

                    mile_level_enabled = task_type in mile_apply_task_levels
                    if (
                        bool(args.mile_apply_non_final_object_decisions)
                        and is_object_decision
                        and mile_level_enabled
                        and not bool(is_final)
                    ):
                        try:
                            corrected_target, mile_info = correct_final_decision_with_mile(
                                rep=pq3d.representation_manager,
                                decision_aux=aux,
                                baseline_target_xyz=baseline_target,
                                output_dir=dec_dir / "mile",
                                path_finder=pf,
                                agent_position_xyz=agent.get_state().position,
                                cfg=cfg,
                            )
                            corrected_target, mile_info = _apply_followability_filter(
                                mile_info=mile_info,
                                baseline_target=baseline_target,
                                pf=pf,
                                agent=agent,
                                cfg=cfg,
                            )
                        except MileRejectedError as exc:
                            mile_info = {
                                "mile_called": True,
                                "target_source": "mile_rejected",
                                "correction_applied": False,
                                "viewpoint_correction_applied": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                            corrected_target = baseline_target.copy()
                            effectiveness_dict.setdefault("module_status_counts", {})
                            effectiveness_dict["module_status_counts"]["mile_rejected"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_rejected", 0)
                            ) + 1
                            _write_json(dec_dir / "mile" / "mile_rejected.json", mile_info)
                        except Exception as exc:
                            mile_info = {
                                "mile_called": True,
                                "target_source": "mile_error",
                                "correction_applied": False,
                                "viewpoint_correction_applied": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                            effectiveness_dict.setdefault("module_status_counts", {})
                            effectiveness_dict["module_status_counts"]["mile_error"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_error", 0)
                            ) + 1
                            _write_json(dec_dir / "mile" / "mile_error.json", mile_info)
                            _tqdm_print(
                                f"[mile-refine1][module-error] scene={scene_name} ep={episode_id} task={idx} "
                                f"dec={decision_num} final=False error={type(exc).__name__}: {exc}"
                            )
                            raise

                        eval_goals, goal_positions, view_points, goal_category = _eval_goal_bundle(cur_task, eval_goals_map)
                        effectiveness = build_effectiveness_record(
                            baseline_target_xyz=baseline_target,
                            corrected_target_xyz=corrected_target,
                            goal_positions_xyz=goal_positions,
                            threshold_m=float(args.effectiveness_threshold_m),
                        )
                        effectiveness = _augment_viewpoint_effectiveness(
                            effectiveness,
                            pf=pf,
                            baseline_target_xyz=baseline_target,
                            corrected_target_xyz=corrected_target,
                            view_points=view_points,
                            threshold_m=float(args.effectiveness_threshold_m),
                        )
                        mile_info["effectiveness"] = effectiveness
                        _write_json(dec_dir / "mile" / "mile_decision.json", mile_info)
                        case = str(effectiveness["case"])
                        effectiveness_dict.setdefault("case_counts", {"00": 0, "01": 0, "10": 0, "11": 0})
                        effectiveness_dict["case_counts"][case] = int(effectiveness_dict["case_counts"].get(case, 0)) + 1
                        effectiveness_dict.setdefault("module_status_counts", {})
                        if bool(mile_info.get("correction_applied", False)):
                            effectiveness_dict["module_status_counts"]["mile_applied"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_applied", 0)
                            ) + 1
                        else:
                            effectiveness_dict["module_status_counts"]["mile_kept_baseline"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_kept_baseline", 0)
                            ) + 1
                        task_effective_logs.append(
                            {
                                "scene_name": scene_name,
                                "episode_id": int(episode_id),
                                "task_id": int(idx),
                                "task_level": task_type,
                                "decision_num": int(decision_num),
                                "sentence": sentence,
                                "is_final": False,
                                "mile": mile_info,
                                "effectiveness": effectiveness,
                            }
                        )
                        _tqdm_print(
                            f"[mile-refine1][module] scene={scene_name} ep={episode_id} task={idx} dec={decision_num} "
                            f"final=False called={mile_info['mile_called']} source={mile_info['target_source']} "
                            f"correction_applied={mile_info['correction_applied']} "
                            f"viewpoint_applied={mile_info['viewpoint_correction_applied']} "
                            f"followability_ok={mile_info.get('followability_selected_precheck', {}).get('precheck_ok', None)} "
                            f"rejected_reason={mile_info.get('rejected_reason', None)} "
                            f"case_metric={effectiveness['case_metric']} case={case} "
                            f"baseline_gt_viewpoint_in_1m={effectiveness['baseline_gt_viewpoint_in_1m']} "
                            f"corrected_gt_viewpoint_in_1m={effectiveness['corrected_gt_viewpoint_in_1m']} "
                            f"baseline_vp_geo={effectiveness['baseline_nearest_gt_viewpoint_geo_m']:.3f} "
                            f"corrected_vp_geo={effectiveness['corrected_nearest_gt_viewpoint_geo_m']:.3f} "
                            f"object_l2_case={effectiveness['object_l2_case']} "
                            f"baseline_obj_l2={effectiveness['baseline_nearest_goal_dist_m']:.3f} "
                            f"corrected_obj_l2={effectiveness['corrected_nearest_goal_dist_m']:.3f}"
                        )

                    if bool(is_final):
                        baseline_final_target = baseline_target.copy()
                        if is_object_decision and mile_level_enabled:
                            try:
                                corrected_target, mile_info = correct_final_decision_with_mile(
                                    rep=pq3d.representation_manager,
                                    decision_aux=aux,
                                    baseline_target_xyz=baseline_target,
                                    output_dir=dec_dir / "mile",
                                    path_finder=pf,
                                    agent_position_xyz=agent.get_state().position,
                                    cfg=cfg,
                                )
                                corrected_target, mile_info = _apply_followability_filter(
                                    mile_info=mile_info,
                                    baseline_target=baseline_target,
                                    pf=pf,
                                    agent=agent,
                                    cfg=cfg,
                                )
                            except MileRejectedError as exc:
                                mile_info = {
                                    "mile_called": True,
                                    "target_source": "mile_rejected",
                                    "correction_applied": False,
                                    "viewpoint_correction_applied": False,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                                final_mile_info = mile_info
                                corrected_target = baseline_target.copy()
                                effectiveness_dict.setdefault("module_status_counts", {})
                                effectiveness_dict["module_status_counts"]["mile_rejected"] = int(
                                    effectiveness_dict["module_status_counts"].get("mile_rejected", 0)
                                ) + 1
                                _write_json(dec_dir / "mile" / "mile_rejected.json", mile_info)
                                _write_json(
                                    dec_dir / "mile_step_summary.json",
                                    {
                                        "task_id": int(idx),
                                        "task_level": task_type,
                                        "decision_num": int(decision_num),
                                        "is_final": True,
                                        "baseline_target": baseline_target.tolist(),
                                        "corrected_target": corrected_target.tolist(),
                                        "used_target": corrected_target.tolist(),
                                        "pq3d_last_decision_aux": aux,
                                        "frontier_filter_info": frontier_filter_info,
                                        "register_info": register_info,
                                        "follow_info": None,
                                        "mile": mile_info,
                                        "effectiveness": None,
                                    },
                                )
                                _tqdm_print(
                                    f"[mile-refine1][module-rejected] scene={scene_name} ep={episode_id} task={idx} "
                                    f"dec={decision_num} explicit_noop=True use_baseline_target=True "
                                    f"error={type(exc).__name__}: {exc}"
                                )
                            except Exception as exc:
                                mile_info = {
                                    "mile_called": True,
                                    "target_source": "mile_error",
                                    "correction_applied": False,
                                    "viewpoint_correction_applied": False,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                                effectiveness_dict.setdefault("module_status_counts", {})
                                effectiveness_dict["module_status_counts"]["mile_error"] = int(
                                    effectiveness_dict["module_status_counts"].get("mile_error", 0)
                                ) + 1
                                _write_json(dec_dir / "mile" / "mile_error.json", mile_info)
                                _tqdm_print(
                                    f"[mile-refine1][module-error] scene={scene_name} ep={episode_id} task={idx} "
                                    f"dec={decision_num} error={type(exc).__name__}: {exc}"
                                )
                                raise
                        elif is_object_decision and not mile_level_enabled:
                            mile_info = {
                                "mile_called": True,
                                "target_source": "task_level_disabled_explicit_keep_baseline",
                                "correction_applied": False,
                                "viewpoint_correction_applied": False,
                                "rejected_reason": f"task_level_not_enabled:{task_type}",
                                "mile_apply_task_levels": sorted(mile_apply_task_levels),
                            }
                            corrected_target = baseline_target.copy()
                        else:
                            mile_info = {
                                "mile_called": False,
                                "target_source": "non_object_decision_explicit_keep_baseline",
                                "correction_applied": False,
                                "viewpoint_correction_applied": False,
                                "rejected_reason": "decision_aux_is_not_object_decision",
                            }
                            corrected_target = baseline_target.copy()
                        eval_goals, goal_positions, view_points, goal_category = _eval_goal_bundle(cur_task, eval_goals_map)
                        effectiveness = build_effectiveness_record(
                            baseline_target_xyz=baseline_target,
                            corrected_target_xyz=corrected_target,
                            goal_positions_xyz=goal_positions,
                            threshold_m=float(args.effectiveness_threshold_m),
                        )
                        effectiveness = _augment_viewpoint_effectiveness(
                            effectiveness,
                            pf=pf,
                            baseline_target_xyz=baseline_target,
                            corrected_target_xyz=corrected_target,
                            view_points=view_points,
                            threshold_m=float(args.effectiveness_threshold_m),
                        )
                        mile_info["effectiveness"] = effectiveness
                        _write_json(dec_dir / "mile" / "mile_decision.json", mile_info)
                        case = str(effectiveness["case"])
                        effectiveness_dict.setdefault("case_counts", {"00": 0, "01": 0, "10": 0, "11": 0})
                        effectiveness_dict["case_counts"][case] = int(effectiveness_dict["case_counts"].get(case, 0)) + 1
                        effectiveness_dict.setdefault("module_status_counts", {})
                        if bool(mile_info.get("correction_applied", False)):
                            effectiveness_dict["module_status_counts"]["mile_applied"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_applied", 0)
                            ) + 1
                        else:
                            effectiveness_dict["module_status_counts"]["mile_kept_baseline"] = int(
                                effectiveness_dict["module_status_counts"].get("mile_kept_baseline", 0)
                            ) + 1
                        corrected_final_target = corrected_target.copy()
                        final_mile_info = mile_info
                        final_effectiveness = effectiveness
                        task_effective_logs.append(
                            {
                                "scene_name": scene_name,
                                "episode_id": int(episode_id),
                                "task_id": int(idx),
                                "task_level": task_type,
                                "decision_num": int(decision_num),
                                "sentence": sentence,
                                "mile": mile_info,
                                "effectiveness": effectiveness,
                            }
                        )
                        _tqdm_print(
                            f"[mile-refine1][module] scene={scene_name} ep={episode_id} task={idx} dec={decision_num} "
                            f"called={mile_info['mile_called']} source={mile_info['target_source']} "
                            f"correction_applied={mile_info['correction_applied']} "
                            f"viewpoint_applied={mile_info['viewpoint_correction_applied']} "
                            f"followability_ok={mile_info.get('followability_selected_precheck', {}).get('precheck_ok', None)} "
                            f"rejected_reason={mile_info.get('rejected_reason', None)} "
                            f"case_metric={effectiveness['case_metric']} case={case} "
                            f"baseline_gt_viewpoint_in_1m={effectiveness['baseline_gt_viewpoint_in_1m']} "
                            f"corrected_gt_viewpoint_in_1m={effectiveness['corrected_gt_viewpoint_in_1m']} "
                            f"baseline_vp_geo={effectiveness['baseline_nearest_gt_viewpoint_geo_m']:.3f} "
                            f"corrected_vp_geo={effectiveness['corrected_nearest_gt_viewpoint_geo_m']:.3f} "
                            f"object_l2_case={effectiveness['object_l2_case']} "
                            f"baseline_obj_l2={effectiveness['baseline_nearest_goal_dist_m']:.3f} "
                            f"corrected_obj_l2={effectiveness['corrected_nearest_goal_dist_m']:.3f}"
                        )
                        if bool(args.mile_enable_navigation_target_repair):
                            used_target, nav_adjust_info = _select_followable_navigation_target(
                                pf=pf,
                                agent=agent,
                                raw_target=corrected_target,
                            )
                        else:
                            used_target = corrected_target.copy()
                            nav_adjust_info = {
                                "called": False,
                                "adjustment_applied": False,
                                "selected_role": "repair_disabled",
                                "selected_target": used_target.tolist(),
                                "rejected_reason": "mile_enable_navigation_target_repair_false",
                            }
                        if bool(nav_adjust_info.get("adjustment_applied", False)) or not bool(nav_adjust_info.get("original_precheck_ok", True)):
                            _tqdm_print(
                                f"[mile-refine1][nav-adjust] scene={scene_name} ep={episode_id} task={idx} "
                                f"dec={decision_num} final=True applied={nav_adjust_info.get('adjustment_applied')} "
                                f"selected_role={nav_adjust_info.get('selected_role')} "
                                f"selected_l2={nav_adjust_info.get('selected_l2_to_original_m')} "
                                f"followable_candidates={nav_adjust_info.get('followable_candidate_count', 0)} "
                                f"reason={nav_adjust_info.get('rejected_reason')}"
                            )
                        try:
                            goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _follow_target(
                                pf=pf,
                                agent=agent,
                                sim=sim,
                                top_down_map=top_down_map,
                                fog=fog,
                                vis_dist=vis_dist,
                                used_target=used_target,
                                prev_agent_state=prev_agent_state,
                                total_steps=total_steps,
                                max_steps=int(args.max_steps),
                                episode_cum_distance=float(episode_cum_distance),
                            )
                            task_end_reason = "final_decision"
                        except FollowerNavigationError as exc:
                            follow_info = dict(exc.info)
                            follow_info.update(
                                {
                                    "decision_num": int(decision_num),
                                    "is_final": True,
                                    "stage": "final_follow",
                                    "message": str(exc),
                                }
                            )
                            task_end_reason = "follower_error"
                            effectiveness_dict.setdefault("module_status_counts", {})
                            effectiveness_dict["module_status_counts"]["follower_error"] = int(
                                effectiveness_dict["module_status_counts"].get("follower_error", 0)
                            ) + 1
                            _write_json(
                                out_task / "follow_error.json",
                                {
                                    "scene_name": scene_name,
                                    "episode_id": int(episode_id),
                                    "task_id": int(idx),
                                    "task_level": task_type,
                                    "follow_info": follow_info,
                                },
                            )
                            _tqdm_print(
                                f"[mile-refine1][follow-error] scene={scene_name} ep={episode_id} task={idx} "
                                f"dec={decision_num} final=True error={follow_info.get('error_type')} "
                                f"path_found={follow_info.get('shortest_path_found')} "
                                f"geo={follow_info.get('shortest_path_geodesic_distance')}"
                            )
                        final_follow_info = follow_info
                        _write_json(
                            dec_dir / "mile_step_summary.json",
                            {
                                "task_id": int(idx),
                                "task_level": task_type,
                                "decision_num": int(decision_num),
                                "is_final": True,
                                "baseline_target": baseline_target.tolist(),
                                "corrected_target": corrected_target.tolist(),
                                "used_target": used_target.tolist(),
                                "pq3d_last_decision_aux": aux,
                                "frontier_filter_info": frontier_filter_info,
                                "register_info": register_info,
                                "navigation_target_adjustment": nav_adjust_info,
                                "follow_info": follow_info,
                                "mile": mile_info,
                                "effectiveness": effectiveness,
                            },
                        )
                        decision_num += 1
                        break

                    if bool(args.mile_enable_navigation_target_repair):
                        used_target, nav_adjust_info = _select_followable_navigation_target(
                            pf=pf,
                            agent=agent,
                            raw_target=corrected_target,
                        )
                    else:
                        used_target = corrected_target.copy()
                        nav_adjust_info = {
                            "called": False,
                            "adjustment_applied": False,
                            "selected_role": "repair_disabled",
                            "selected_target": used_target.tolist(),
                            "rejected_reason": "mile_enable_navigation_target_repair_false",
                        }
                    if bool(nav_adjust_info.get("adjustment_applied", False)) or not bool(nav_adjust_info.get("original_precheck_ok", True)):
                        _tqdm_print(
                            f"[mile-refine1][nav-adjust] scene={scene_name} ep={episode_id} task={idx} "
                            f"dec={decision_num} final=False applied={nav_adjust_info.get('adjustment_applied')} "
                            f"selected_role={nav_adjust_info.get('selected_role')} "
                            f"selected_l2={nav_adjust_info.get('selected_l2_to_original_m')} "
                            f"followable_candidates={nav_adjust_info.get('followable_candidate_count', 0)} "
                            f"reason={nav_adjust_info.get('rejected_reason')}"
                        )
                    baseline_frontier_key = _frontier_visit_key(baseline_target)
                    used_frontier_key = _frontier_visit_key(used_target)
                    visited_frontier.add(baseline_frontier_key)
                    visited_frontier.add(used_frontier_key)
                    visited_update_info = {
                        "baseline_frontier_key": list(baseline_frontier_key),
                        "used_frontier_key": list(used_frontier_key),
                        "added_baseline_frontier_key": True,
                        "added_used_frontier_key": bool(used_frontier_key != baseline_frontier_key),
                        "policy": "mark_baseline_and_used_target_for_nonfinal_decision",
                    }
                    try:
                        goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_info = _follow_target(
                            pf=pf,
                            agent=agent,
                            sim=sim,
                            top_down_map=top_down_map,
                            fog=fog,
                            vis_dist=vis_dist,
                            used_target=used_target,
                            prev_agent_state=prev_agent_state,
                            total_steps=total_steps,
                            max_steps=int(args.max_steps),
                            episode_cum_distance=float(episode_cum_distance),
                        )
                    except FollowerNavigationError as exc:
                        follow_info = dict(exc.info)
                        follow_info.update(
                            {
                                "decision_num": int(decision_num),
                                "is_final": False,
                                "stage": "frontier_follow",
                                "message": str(exc),
                            }
                        )
                        final_follow_info = follow_info
                        task_end_reason = "follower_error"
                        effectiveness_dict.setdefault("module_status_counts", {})
                        effectiveness_dict["module_status_counts"]["follower_error"] = int(
                            effectiveness_dict["module_status_counts"].get("follower_error", 0)
                        ) + 1
                        _write_json(
                            out_task / "follow_error.json",
                            {
                                "scene_name": scene_name,
                                "episode_id": int(episode_id),
                                "task_id": int(idx),
                                "task_level": task_type,
                                "follow_info": follow_info,
                            },
                        )
                        _write_json(
                            dec_dir / "mile_step_summary.json",
                            {
                                "task_id": int(idx),
                                "task_level": task_type,
                                "decision_num": int(decision_num),
                                "is_final": False,
                                "baseline_target": baseline_target.tolist(),
                                "corrected_target": corrected_target.tolist(),
                                "used_target": used_target.tolist(),
                                "used_frontier_key": list(used_frontier_key),
                                "visited_update": visited_update_info,
                                "pq3d_last_decision_aux": aux,
                                "frontier_filter_info": frontier_filter_info,
                                "register_info": register_info,
                                "navigation_target_adjustment": nav_adjust_info,
                                "follow_info": follow_info,
                                "mile": mile_info,
                                "effectiveness": effectiveness,
                            },
                        )
                        _tqdm_print(
                            f"[mile-refine1][follow-error] scene={scene_name} ep={episode_id} task={idx} "
                            f"dec={decision_num} final=False error={follow_info.get('error_type')} "
                            f"path_found={follow_info.get('shortest_path_found')} "
                            f"geo={follow_info.get('shortest_path_geodesic_distance')}"
                        )
                        decision_num += 1
                        break
                    _write_json(
                        dec_dir / "mile_step_summary.json",
                        {
                            "task_id": int(idx),
                            "task_level": task_type,
                            "decision_num": int(decision_num),
                            "is_final": False,
                            "baseline_target": baseline_target.tolist(),
                            "corrected_target": corrected_target.tolist(),
                            "used_target": used_target.tolist(),
                            "used_frontier_key": list(used_frontier_key),
                            "visited_update": visited_update_info,
                            "pq3d_last_decision_aux": aux,
                            "frontier_filter_info": frontier_filter_info,
                            "register_info": register_info,
                            "navigation_target_adjustment": nav_adjust_info,
                            "follow_info": follow_info,
                            "mile": mile_info,
                            "effectiveness": effectiveness,
                        },
                    )
                    decision_num += 1

                task_time = float(time.perf_counter() - task_t0)
                end_state = agent.get_state()
                if eval_goals is None:
                    eval_goals, goal_positions, view_points, goal_category = _eval_goal_bundle(cur_task, eval_goals_map)
                start_goal_geo = _geo_dist_to_viewpoints(pf, sub_episode_start_position, view_points)
                end_goal_geo = _geo_dist_to_viewpoints(pf, end_state.position, view_points)
                if np.isinf(start_goal_geo) or np.isinf(end_goal_geo):
                    raw_sr = 0.0
                    raw_spl = 0.0
                else:
                    raw_sr = 1.0 if end_goal_geo <= float(args.success_distance) else 0.0
                    raw_spl = float(raw_sr * start_goal_geo / max(start_goal_geo, max(episode_cum_distance, 1e-12)))
                if task_end_reason == "follower_error":
                    sr = 0.0
                    spl = 0.0
                else:
                    sr = float(raw_sr)
                    spl = float(raw_spl)

                baseline_target_to_goal_l2 = _nearest_goal_dist(baseline_final_target, goal_positions)
                corrected_target_to_goal_l2 = _nearest_goal_dist(corrected_final_target, goal_positions)
                baseline_target_to_gt_viewpoint_geo = (
                    _geo_dist_to_viewpoints(pf, baseline_final_target, view_points)
                    if baseline_final_target is not None
                    else float("inf")
                )
                corrected_target_to_gt_viewpoint_geo = (
                    _geo_dist_to_viewpoints(pf, corrected_final_target, view_points)
                    if corrected_final_target is not None
                    else float("inf")
                )
                correction_applied = bool(final_mile_info.get("correction_applied", False))
                mile_helpful = None
                if (
                    correction_applied
                    and np.isfinite(baseline_target_to_gt_viewpoint_geo)
                    and np.isfinite(corrected_target_to_gt_viewpoint_geo)
                ):
                    mile_helpful = bool(corrected_target_to_gt_viewpoint_geo < baseline_target_to_gt_viewpoint_geo - 1e-6)

                row = {
                    "scene_name": scene_name,
                    "episode_id": int(episode_id),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "sr": float(sr),
                    "spl": float(spl),
                    "raw_sr_from_end_position": float(raw_sr),
                    "raw_spl_from_end_position": float(raw_spl),
                    "metric_policy": "follower_error_forces_zero_like_baseline",
                    "object_category": goal_category,
                    "task_time_sec": task_time,
                    "steps_total": int(total_steps),
                    "decisions": int(decision_num),
                    "end_reason": task_end_reason,
                    "start_goal_geo": float(start_goal_geo),
                    "end_goal_geo": float(end_goal_geo),
                    "episode_cum_distance": float(episode_cum_distance),
                    "eval_only_goal_positions": [g.tolist() for g in goal_positions],
                    "eval_only_gt_viewpoint_count": int(len(view_points)),
                    "baseline_target_position": None if baseline_final_target is None else baseline_final_target.tolist(),
                    "corrected_target_position": None if corrected_final_target is None else corrected_final_target.tolist(),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "corrected_target_to_goal_l2": float(corrected_target_to_goal_l2),
                    "baseline_target_to_gt_viewpoint_geo": float(baseline_target_to_gt_viewpoint_geo),
                    "corrected_target_to_gt_viewpoint_geo": float(corrected_target_to_gt_viewpoint_geo),
                    "mile_called": bool(final_mile_info.get("mile_called", False)),
                    "mile_target_source": final_mile_info.get("target_source"),
                    "mile_correction_applied": bool(final_mile_info.get("correction_applied", False)),
                    "mile_viewpoint_applied": bool(final_mile_info.get("viewpoint_correction_applied", False)),
                    "mile_selected_slot_index": final_mile_info.get("selected_slot_index"),
                    "mile_case": None if final_effectiveness is None else final_effectiveness.get("case"),
                    "mile_helpful": mile_helpful,
                    "final_follow_info": final_follow_info,
                }
                result_dict.setdefault("sequence", []).append(row)
                effectiveness_dict.setdefault("records", []).append(
                    {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(idx),
                        "task_level": task_type,
                        "navigation_type": navigation_type,
                        "sentence": sentence,
                        "effectiveness": final_effectiveness,
                        "mile": final_mile_info,
                        "task_effective_logs": task_effective_logs,
                    }
                )
                _write_json(out_task / "summary.json", row)
                _tqdm_print(
                    f"[mile-refine1][task-summary] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                    f"SR={sr:.1f} SPL={spl:.4f} steps={total_steps} decisions={decision_num} "
                    f"end_reason={task_end_reason} case={row['mile_case']} helpful={mile_helpful}"
                )
                _write_json(output_path, result_dict)
                _write_json(effectiveness_path, effectiveness_dict)

            sim.close()
            _write_json(output_path, result_dict)
            _write_json(effectiveness_path, effectiveness_dict)
            _sequence_compute_metric_results(result_dict)
            _tqdm_print(f"[mile-refine1][case-counts] {effectiveness_dict.get('case_counts', {})}")
            _tqdm_print(f"[mile-refine1][module-status-counts] {effectiveness_dict.get('module_status_counts', {})}")

    _sequence_compute_metric_results(result_dict)
    _tqdm_print(f"[mile-refine1][case-counts] {effectiveness_dict.get('case_counts', {})}")
    _tqdm_print(f"[mile-refine1][module-status-counts] {effectiveness_dict.get('module_status_counts', {})}")


if __name__ == "__main__":
    main()
