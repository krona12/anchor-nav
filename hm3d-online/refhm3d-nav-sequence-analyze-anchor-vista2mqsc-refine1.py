"""RefHM3D Vista2MQSC refine1 batch evaluation.

This batch script keeps the template refine1 scaffold and plugs in the
combined MQSC-R1 + VISTA-LS policy only at final PQ3D object decisions:

- scene slicing via ``--start_ratio`` / ``--end_ratio``
- episode resume via the output JSON
- sequence navigation over tasks in an episode
- PQ3D + frontier exploration loop
- per-decision JSON logs under ``process/``
- task metrics, live metrics, and final JSON summaries
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def _tqdm_print(msg: str) -> None:
    """Print through tqdm so progress bars do not swallow log lines."""
    tqdm.write(msg, file=sys.stdout)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from anchor_nav.mqsc_r1 import MqscR1Config, run_mqsc_r1_refine
from anchor_nav.vista_ls import VistaLsConfig, VistaLsRejectedError, correct_final_decision_with_vistals


TASK_LEVEL_ORDER = ("object", "room", "region", "instance")
MQSC_R1_CFG = MqscR1Config()
VISTALS_CFG = VistaLsConfig()
VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS = set(TASK_LEVEL_ORDER)


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


def _setup_run_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-vista2mqsc-refine1-{ts}-pid{os.getpid()}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[Vista2MQSCRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup() -> None:
        try:
            print(f"[Vista2MQSCRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        value = float(x)
        return value if math.isfinite(value) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    _tqdm_print(f"[Vista2MQSCRefine1] seed={int(seed)}")


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [scene_dir / f"{short_scene_name}.basis.glb", scene_dir / f"{short_scene_name}.glb"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name}; checked={candidates}")


def build_sentence(
    task_type: str,
    cur_task: Dict[str, Any],
    *,
    all_navigation_goals_dict: Dict[str, Any],
    region_to_annot_dict: Dict[str, Any],
    concise_description: bool,
) -> Tuple[str, str]:
    if task_type == "object":
        return cur_task["object_category"], cur_task["object_category"]
    if task_type == "room":
        return f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}", cur_task["object_category"]
    if task_type == "region":
        region_info = region_to_annot_dict[cur_task["region_id"]]
        desc = (
            region_info.get("shortest_description")
            or region_info.get("concise_description")
            or region_info.get("detailed_description")
            or ""
        ) if concise_description else (
            region_info.get("comprehensive_description")
            or region_info.get("detailed_description")
            or region_info.get("concise_description")
            or ""
        )
        sentence = f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {desc}"
        return sentence, cur_task["object_category"]
    if task_type == "instance":
        inst = all_navigation_goals_dict[cur_task["instance_id"]]
        sentence = (
            inst.get("annot_unique_concise_description")
            if concise_description
            else inst.get("annot_unique_detailed_description")
        )
        sentence = sentence or inst.get("annot_unique_normal_description") or inst.get("annot_appearance_description") or ""
        return sentence, inst.get("object_category", "")
    raise ValueError(f"unknown task_type={task_type}")


def _iter_metric_rows(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = result_dict.get("sequence", [])
    if isinstance(rows, list) and len(rows) > 0:
        source = [x for x in rows if isinstance(x, dict)]
    else:
        source = []
        for value in result_dict.values():
            if not isinstance(value, list):
                continue
            for row in value:
                if isinstance(row, dict):
                    source.append(row)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in source:
        key = (
            row.get("scene_name"),
            row.get("navigation_type"),
            row.get("episode_id"),
            row.get("task_id"),
            row.get("task_level"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def metric_snapshot(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    rows = _iter_metric_rows(result_dict)
    if not rows:
        return {
            "count": 0,
            "sr": 0.0,
            "spl": 0.0,
            "avg_task_time_sec": 0.0,
            "by_level": {lv: {"count": 0, "sr": 0.0, "spl": 0.0} for lv in TASK_LEVEL_ORDER},
        }
    by_level: Dict[str, Dict[str, Any]] = {}
    for lv in TASK_LEVEL_ORDER:
        bucket = [x for x in rows if str(x.get("task_level", "")) == lv]
        by_level[lv] = {
            "count": int(len(bucket)),
            "sr": float(np.mean([float(x.get("sr", 0.0)) for x in bucket])) if bucket else 0.0,
            "spl": float(np.mean([float(x.get("spl", 0.0)) for x in bucket])) if bucket else 0.0,
        }
    return {
        "count": int(len(rows)),
        "sr": float(np.mean([float(x.get("sr", 0.0)) for x in rows])),
        "spl": float(np.mean([float(x.get("spl", 0.0)) for x in rows])),
        "avg_task_time_sec": float(np.mean([float(x.get("task_time_sec", 0.0)) for x in rows])),
        "by_level": by_level,
    }


def sequence_compute_metric_results(result_dict: Dict[str, Any]) -> None:
    snap = metric_snapshot(result_dict)
    _tqdm_print(
        f"[Metrics] sequence count={snap['count']}, avg_sr={snap['sr']:.6f}, "
        f"avg_spl={snap['spl']:.6f}, avg_task_time_sec={snap['avg_task_time_sec']:.3f}, "
        f"by_level={json.dumps(snap['by_level'], ensure_ascii=False, sort_keys=True)}"
    )


def append_live_metrics(metrics_log_path: Path, result_dict: Dict[str, Any], *, latest: Dict[str, Any]) -> None:
    snap = metric_snapshot(result_dict)
    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "latest": latest,
        "metrics": snap,
    }
    line = (
        f"[VISTA2MQSC_LIVE_METRICS] count={snap['count']} sr={snap['sr']:.6f} spl={snap['spl']:.6f} "
        f"by_level={json.dumps(snap['by_level'], ensure_ascii=False, sort_keys=True)}"
    )
    _tqdm_print(line)
    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _capture_scan_frames(
    *,
    sim: Any,
    agent: Any,
    top_down_map: np.ndarray,
    fog_of_war_mask: np.ndarray,
    visibility_dist_in_pixels: int,
    total_steps: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], np.ndarray, int]:
    scan_rgb, scan_depth, scan_state = [], [], []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        agent_state = agent.get_state()
        rgb = obs["color_sensor"][:, :, :3]
        dep = obs["depth_sensor"][:, :]
        scan_rgb.append(rgb)
        scan_depth.append(dep)
        scan_state.append(agent_state)
        fog_of_war_mask = reveal_fog_of_war(
            top_down_map,
            fog_of_war_mask,
            map_coors_to_pixel(agent_state.position, top_down_map, sim),
            get_polar_angle(agent_state),
            42,
            visibility_dist_in_pixels,
            False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return scan_rgb, scan_depth, scan_state, fog_of_war_mask, total_steps


def _shortest_path_diagnostics(path_finder: Any, start: np.ndarray, end: np.ndarray) -> Dict[str, Any]:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=float).reshape(3)
    path.requested_end = np.asarray(end, dtype=float).reshape(3)
    found = bool(path_finder.find_path(path))
    return {
        "shortest_path_found": found,
        "shortest_path_geodesic_distance": float(path.geodesic_distance) if found else float("inf"),
    }


def _candidate_follow_targets(
    *,
    path_finder: Any,
    raw_target: np.ndarray,
    start_position: np.ndarray,
    agent_island: int,
) -> List[Dict[str, Any]]:
    raw = np.asarray(raw_target, dtype=float).reshape(3)
    start = np.asarray(start_position, dtype=float).reshape(3)
    seen = set()
    candidates: List[Dict[str, Any]] = []

    def add_candidate(point: np.ndarray, source: str, radius: float) -> None:
        try:
            snapped = np.asarray(path_finder.snap_point(point=point, island_index=agent_island), dtype=float).reshape(3)
        except Exception:
            return
        key = tuple(np.round(snapped, 3).astype(float).tolist())
        if key in seen:
            return
        seen.add(key)
        diag = _shortest_path_diagnostics(path_finder, start, snapped)
        candidates.append(
            {
                "source": source,
                "radius": float(radius),
                "raw_candidate": np.asarray(point, dtype=float).reshape(3).tolist(),
                "snapped_target": snapped.tolist(),
                "target_l2_from_raw": float(np.linalg.norm(snapped - raw)),
                **diag,
            }
        )

    add_candidate(raw, "direct", 0.0)
    rings = (0.35, 0.5, 0.75, 1.0, 1.25, 1.5)
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    for radius in rings:
        for angle in angles:
            point = np.asarray(
                [raw[0] + radius * math.cos(float(angle)), start[1], raw[2] + radius * math.sin(float(angle))],
                dtype=float,
            )
            add_candidate(point, "repair_ring", radius)

    direct = [c for c in candidates if c["source"] == "direct"]
    repair = [c for c in candidates if c["source"] != "direct"]
    repair.sort(
        key=lambda c: (
            not bool(c.get("shortest_path_found", False)),
            float(c.get("target_l2_from_raw", float("inf"))),
            float(c.get("shortest_path_geodesic_distance", float("inf"))),
        )
    )
    return direct + repair


def _find_follow_actions_with_repair(
    *,
    path_finder: Any,
    agent: Any,
    raw_target: np.ndarray,
    start_position: np.ndarray,
    agent_island: int,
) -> Tuple[List[Any], Dict[str, Any], List[Dict[str, Any]]]:
    candidates = _candidate_follow_targets(
        path_finder=path_finder,
        raw_target=raw_target,
        start_position=start_position,
        agent_island=agent_island,
    )
    attempts: List[Dict[str, Any]] = []
    last_error: Dict[str, Any] = {
        "error_type": "NoFollowCandidate",
        "error_message": "no candidate target could be generated",
    }
    for cand in candidates:
        attempt = dict(cand)
        try:
            follower = habitat_sim.GreedyGeodesicFollower(
                path_finder,
                agent,
                forward_key="move_forward",
                left_key="turn_left",
                right_key="turn_right",
            )
            action_list = follower.find_path(np.asarray(cand["snapped_target"], dtype=float))
            if action_list is None:
                raise RuntimeError("GreedyGeodesicFollower returned None")
            non_stop_actions = [a for a in action_list if a]
            if (
                len(non_stop_actions) == 0
                and float(cand.get("shortest_path_geodesic_distance", float("inf"))) > 0.15
            ):
                raise RuntimeError("GreedyGeodesicFollower returned zero actions for nonzero path")
            attempt.update({"ok": True, "path_action_count": int(len(non_stop_actions))})
            attempts.append(attempt)
            return list(action_list), attempt, attempts
        except Exception as exc:
            last_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            attempt.update({"ok": False, **last_error})
            attempts.append(attempt)
    failed = dict(candidates[0]) if candidates else {}
    failed.update({"ok": False, **last_error, "candidate_count": int(len(candidates))})
    return [], failed, attempts


def _follow_target(
    *,
    path_finder: Any,
    agent: Any,
    sim: Any,
    target: np.ndarray,
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float, Dict[str, Any]]:
    start_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    agent_island = int(path_finder.get_island(agent.get_state().position))
    raw_target = np.asarray(target, dtype=float).reshape(3)
    follow_log: Dict[str, Any] = {
        "ok": True,
        "raw_target": raw_target.tolist(),
        "agent_island": int(agent_island),
        "start_position": start_position.tolist(),
        "action_count": 0,
        "repair_applied": False,
    }
    try:
        action_list, chosen, attempts = _find_follow_actions_with_repair(
            path_finder=path_finder,
            agent=agent,
            raw_target=raw_target,
            start_position=start_position,
            agent_island=agent_island,
        )
        if not bool(chosen.get("ok", False)):
            follow_log.update(
                {
                    "ok": False,
                    "error_type": chosen.get("error_type"),
                    "error_message": chosen.get("error_message"),
                    "candidate_attempt_count": int(len(attempts)),
                    "candidate_attempts_head": attempts[:12],
                }
            )
            action_list = []
        else:
            action_list = list(action_list)
        follow_log.update(
            {
                "snapped_target": chosen.get("snapped_target"),
                "candidate_source": chosen.get("source"),
                "candidate_radius": chosen.get("radius"),
                "repair_applied": bool(chosen.get("source") != "direct"),
                "target_l2_from_raw": chosen.get("target_l2_from_raw"),
                "shortest_path_found": chosen.get("shortest_path_found"),
                "shortest_path_geodesic_distance": chosen.get("shortest_path_geodesic_distance"),
                "candidate_attempt_count": int(len(attempts)),
                "candidate_attempts_head": attempts[:12],
            }
        )
    except Exception as e:
        follow_log.update(
            {
                "ok": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        action_list = []

    goto_color_list, goto_depth_list, goto_state_list = [], [], []
    for action in action_list:
        if not action:
            continue
        obs = sim.step(action=action)
        state = agent.get_state()
        goto_color_list.append(obs["color_sensor"][:, :, :3])
        goto_depth_list.append(obs["depth_sensor"][:, :])
        goto_state_list.append(state)
        total_steps += 1
        episode_cum_distance += float(np.linalg.norm(state.position - prev_agent_state.position))
        prev_agent_state = state
        follow_log["action_count"] = int(follow_log["action_count"]) + 1
        if total_steps >= int(max_steps):
            break
    return goto_color_list, goto_depth_list, goto_state_list, prev_agent_state, total_steps, float(episode_cum_distance), follow_log


def mqsc_r1_refine_hook(
    *,
    sentence: str,
    task_type: str,
    scene_name: str,
    episode_id: int,
    task_id: int,
    decision_num: int,
    is_final: bool,
    pq3d_model: Any,
    target_position: np.ndarray,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run MQSC-R1 only at the refine1 final-decision hook."""
    target = np.asarray(target_position, dtype=float).reshape(3).copy()
    if not bool(is_final):
        return target, {
            "ok": True,
            "module": "mqsc-r1",
            "applied": False,
            "reason": "non_final_decision",
            "target_before": target.tolist(),
            "target_after": target.tolist(),
        }
    new_target, module_info = run_mqsc_r1_refine(
        sentence=sentence,
        task_type=task_type,
        pq3d_model=pq3d_model,
        target_position=target,
        cfg=MQSC_R1_CFG,
        output_dir=output_dir,
        decision_num=decision_num,
        context={
            "scene_name": scene_name,
            "episode_id": int(episode_id),
            "task_id": int(task_id),
            "decision_num": int(decision_num),
            "is_final": bool(is_final),
            "called": True,
        },
    )
    module_info.update(
        {
            "scene_name": scene_name,
            "episode_id": int(episode_id),
            "task_id": int(task_id),
            "decision_num": int(decision_num),
            "is_final": bool(is_final),
        }
    )
    return np.asarray(new_target, dtype=float).reshape(3), module_info


def _parse_float_list(text: str, *, arg_name: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if len(values) == 0 or any(x <= 0.0 for x in values):
        raise RuntimeError(f"{arg_name} must contain positive comma-separated floats, got {text!r}")
    return values


def _valid_object_index(value: Any) -> Optional[int]:
    try:
        idx = int(value)
    except Exception:
        return None
    return idx if idx >= 0 else None


def vista2mqsc_refine_hook(
    *,
    sentence: str,
    task_type: str,
    scene_name: str,
    episode_id: int,
    task_id: int,
    decision_num: int,
    is_final: bool,
    pq3d_model: Any,
    target_position: np.ndarray,
    decision_aux: Dict[str, Any],
    output_dir: Path,
    path_finder: Any,
    agent_position_xyz: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run MQSC-R1 semantic consensus, then VISTA-LS viewpoint correction."""
    baseline_target = np.asarray(target_position, dtype=float).reshape(3).copy()
    info: Dict[str, Any] = {
        "ok": True,
        "module": "vista2mqsc",
        "called": bool(is_final),
        "applied": False,
        "reason": "non_final_decision" if not bool(is_final) else "final_decision_no_module_applied",
        "target_before": baseline_target.tolist(),
        "target_after": baseline_target.tolist(),
        "baseline_target_before_mqsc": baseline_target.tolist(),
        "policy": "MQSC-R1 object-only spatial consensus followed by VISTA-LS level-set viewpoint correction",
        "task_level": task_type,
        "mqsc_r1_called": False,
        "mqsc_r1_applied": False,
        "vistals_called": False,
        "vistals_applied": False,
        "vistals_level_enabled": bool(task_type in VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS),
    }
    if not bool(is_final):
        return baseline_target, info

    hook_t0 = time.perf_counter()
    mqsc_t0 = time.perf_counter()
    mqsc_target, mqsc_info = mqsc_r1_refine_hook(
        sentence=sentence,
        task_type=task_type,
        scene_name=scene_name,
        episode_id=int(episode_id),
        task_id=int(task_id),
        decision_num=int(decision_num),
        is_final=True,
        pq3d_model=pq3d_model,
        target_position=baseline_target,
        output_dir=output_dir,
    )
    mqsc_info["called"] = True
    mqsc_info["elapsed_ms"] = float((time.perf_counter() - mqsc_t0) * 1000.0)

    baseline_idx = _valid_object_index(
        mqsc_info.get("baseline_object_index", decision_aux.get("real_object_decision_idx"))
    )
    selected_idx = _valid_object_index(mqsc_info.get("selected_object_index"))
    mqsc_applied = bool(mqsc_info.get("applied", False))
    vistals_slot_idx = selected_idx if mqsc_applied and selected_idx is not None else baseline_idx
    vistals_level_enabled = bool(task_type in VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS)
    is_object_decision = bool(decision_aux.get("is_object_decision", False))

    corrected_target = np.asarray(mqsc_target, dtype=float).reshape(3).copy()
    vistals_info: Dict[str, Any] = {
        "vistals_called": False,
        "correction_applied": False,
        "viewpoint_correction_applied": False,
        "target_source": "vistals_not_called",
        "reason": "vistals_not_called",
    }
    if vistals_level_enabled and is_object_decision and vistals_slot_idx is not None:
        vistals_aux = dict(decision_aux)
        vistals_aux["is_object_decision"] = True
        vistals_aux["real_object_decision_idx"] = int(vistals_slot_idx)
        vistals_baseline_target_source = "mqsc_r1_target" if mqsc_applied else "pq3d_baseline_target"
        vistals_t0 = time.perf_counter()
        try:
            _tqdm_print(
                f"[vista2mqsc-refine1][vistals-start] scene={scene_name} ep={episode_id} task={task_id} "
                f"dec={decision_num} slot={vistals_slot_idx} mqsc_applied={mqsc_applied}"
            )
            corrected_target, vistals_info = correct_final_decision_with_vistals(
                rep=pq3d_model.representation_manager,
                decision_aux=vistals_aux,
                baseline_target_xyz=corrected_target,
                output_dir=output_dir / "vista2mqsc" / f"dec_{int(decision_num):03d}_vistals",
                path_finder=path_finder,
                agent_position_xyz=agent_position_xyz,
                cfg=VISTALS_CFG,
            )
            vistals_info["elapsed_ms"] = float((time.perf_counter() - vistals_t0) * 1000.0)
            vistals_info["vistals_input_slot_source"] = "mqsc_r1_selected_object" if mqsc_applied else "pq3d_baseline_object"
            vistals_info["vistals_input_slot_index"] = int(vistals_slot_idx)
            vistals_info["vistals_baseline_target_source"] = vistals_baseline_target_source
        except VistaLsRejectedError as exc:
            vistals_info = {
                "vistals_called": True,
                "target_source": "vistals_rejected",
                "correction_applied": False,
                "viewpoint_correction_applied": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_ms": float((time.perf_counter() - vistals_t0) * 1000.0),
                "vistals_input_slot_index": int(vistals_slot_idx),
                "vistals_baseline_target_source": vistals_baseline_target_source,
            }
            corrected_target = np.asarray(mqsc_target, dtype=float).reshape(3).copy()
            _write_json(output_dir / "vista2mqsc" / f"dec_{int(decision_num):03d}_vistals_rejected.json", vistals_info)
        except Exception as exc:
            vistals_info = {
                "vistals_called": True,
                "target_source": "vistals_error",
                "correction_applied": False,
                "viewpoint_correction_applied": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_ms": float((time.perf_counter() - vistals_t0) * 1000.0),
                "vistals_input_slot_index": int(vistals_slot_idx),
                "vistals_baseline_target_source": vistals_baseline_target_source,
            }
            corrected_target = np.asarray(mqsc_target, dtype=float).reshape(3).copy()
            _write_json(output_dir / "vista2mqsc" / f"dec_{int(decision_num):03d}_vistals_error.json", vistals_info)
    else:
        if not vistals_level_enabled:
            vistals_info["reason"] = "task_level_not_enabled_for_vistals"
        elif not is_object_decision:
            vistals_info["reason"] = "pq3d_decision_aux_not_object"
        else:
            vistals_info["reason"] = "no_valid_object_slot_for_vistals"

    vistals_applied = bool(vistals_info.get("correction_applied", False))
    applied = bool(mqsc_applied or vistals_applied)
    if mqsc_applied and vistals_applied:
        reason = "mqsc_r1_semantic_target_then_vistals_viewpoint"
    elif mqsc_applied:
        reason = "mqsc_r1_semantic_target_only"
    elif vistals_applied:
        reason = "vistals_viewpoint_only"
    else:
        reason = "baseline_target_kept"

    info.update(
        {
            "applied": bool(applied),
            "reason": reason,
            "target_after": np.asarray(corrected_target, dtype=float).reshape(3).tolist(),
            "target_after_mqsc_r1": np.asarray(mqsc_target, dtype=float).reshape(3).tolist(),
            "vistals_baseline_target_source": "mqsc_r1_target" if mqsc_applied else "pq3d_baseline_target",
            "elapsed_ms": float((time.perf_counter() - hook_t0) * 1000.0),
            "mqsc_r1_called": True,
            "mqsc_r1_applied": bool(mqsc_applied),
            "mqsc_r1_selected_object_index": selected_idx,
            "mqsc_r1_baseline_object_index": baseline_idx,
            "vistals_called": bool(vistals_info.get("vistals_called", False)),
            "vistals_applied": bool(vistals_applied),
            "vistals_level_enabled": bool(vistals_level_enabled),
            "vistals_input_slot_index": vistals_slot_idx,
            "vistals_input_slot_source": "mqsc_r1_selected_object" if mqsc_applied else "pq3d_baseline_object",
            "mqsc_r1": mqsc_info,
            "vistals": vistals_info,
        }
    )
    return np.asarray(corrected_target, dtype=float).reshape(3), info


def _existing_episode_keys(result_dict: Dict[str, Any]) -> set:
    return {
        "_".join([str(r.get("scene_name")), str(r.get("navigation_type")), str(r.get("episode_id"))])
        for r in _iter_metric_rows(result_dict)
    }


def _append_result_row(result_dict: Dict[str, Any], navigation_type: str, row: Dict[str, Any]) -> None:
    result_dict.setdefault("sequence", []).append(row)
    if str(navigation_type) != "sequence":
        result_dict.setdefault(navigation_type, []).append(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RefHM3D Vista2MQSC refine1 batch evaluation. "
            "Final PQ3D object decisions are first refined by MQSC-R1, then adjusted by VISTA-LS."
        )
    )
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=0.2)
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
    parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
    parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/vista2mqsc"))
    parser.add_argument(
        "--task_levels",
        type=str,
        default="object,room,region,instance",
        help="Comma-separated task levels.",
    )
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--quiet_nav_steps",
        action="store_true",
        help="Do not print every scan/frontier/PQ3D timing line.",
    )
    parser.add_argument(
        "--decision_log_interval",
        type=int,
        default=0,
        help="Print target detail every N decisions; <=0 disables detail.",
    )
    parser.add_argument("--mqsc_r1_top_k", type=int, default=8)
    parser.add_argument("--mqsc_r1_temperature", type=float, default=1.0)
    parser.add_argument("--mqsc_r1_cluster_eps", type=float, default=1.2)
    parser.add_argument("--mqsc_r1_min_region_coverage", type=float, default=0.5)
    parser.add_argument("--mqsc_r1_min_target_prob", type=float, default=0.05)
    parser.add_argument("--mqsc_r1_min_region_margin", type=float, default=0.05)
    parser.add_argument("--mqsc_r1_min_selected_gain", type=float, default=-0.02)
    parser.add_argument("--mqsc_r1_vlm_model", type=str, default=os.environ.get("VLM_MODEL", MqscR1Config().vlm_model))
    parser.add_argument("--mqsc_r1_vlm_max_retries", type=int, default=3)
    parser.add_argument("--mqsc_r1_vlm_retry_sleep_sec", type=float, default=1.0)
    parser.add_argument("--mqsc_r1_disable_vlm", action="store_true")
    parser.add_argument("--mqsc_r1_disable_no_proxy", action="store_true")
    parser.add_argument("--mqsc_r1_disable_heuristic_decompose", action="store_true")
    parser.add_argument("--mqsc_r1_disable_debug_json", action="store_true")
    parser.add_argument("--vistals_candidate_radii_m", type=str, default="0.5,0.75")
    parser.add_argument("--vistals_candidate_view_count", type=int, default=20)
    parser.add_argument("--vistals_enable_vvd_replacement", action="store_true")
    parser.add_argument("--vistals_disable_visible_baseline_guard", action="store_true")
    parser.add_argument("--vistals_camera_height_m", type=float, default=1.50)
    parser.add_argument("--vistals_max_snap_distance_m", type=float, default=0.60)
    parser.add_argument("--vistals_target_sample_count", type=int, default=300)
    parser.add_argument("--vistals_scene_sample_count", type=int, default=0)
    parser.add_argument("--vistals_max_ray_sample_count", type=int, default=300)
    parser.add_argument("--vistals_occlusion_radius_m", type=float, default=0.05)
    parser.add_argument("--vistals_min_visibility_score", type=float, default=0.02)
    parser.add_argument("--vistals_visibility_tie_epsilon", type=float, default=0.0)
    parser.add_argument("--vistals_path_efficiency_exponent", type=float, default=0.15)
    parser.add_argument("--vistals_r_min_m", type=float, default=0.30)
    parser.add_argument("--vistals_r_max_m", type=float, default=1.30)
    parser.add_argument("--vistals_radial_step_m", type=float, default=0.10)
    parser.add_argument("--vistals_angle_step_deg", type=float, default=10.0)
    parser.add_argument("--vistals_shell_min_m", type=float, default=0.35)
    parser.add_argument("--vistals_shell_max_m", type=float, default=1.20)
    parser.add_argument("--vistals_relaxed_shell_min_m", type=float, default=0.30)
    parser.add_argument("--vistals_relaxed_shell_max_m", type=float, default=1.35)
    parser.add_argument("--vistals_min_clearance_m", type=float, default=0.10)
    parser.add_argument("--vistals_min_component_size", type=int, default=3)
    parser.add_argument("--vistals_size_tie_ratio", type=float, default=0.85)
    parser.add_argument("--vistals_apply_task_levels", type=str, default="object,room,region,instance")
    args = parser.parse_args()

    global MQSC_R1_CFG, VISTALS_CFG, VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS
    MQSC_R1_CFG = MqscR1Config(
        top_k=int(args.mqsc_r1_top_k),
        temperature=float(args.mqsc_r1_temperature),
        cluster_eps=float(args.mqsc_r1_cluster_eps),
        min_region_coverage=float(args.mqsc_r1_min_region_coverage),
        min_target_prob=float(args.mqsc_r1_min_target_prob),
        min_region_margin=float(args.mqsc_r1_min_region_margin),
        min_selected_gain=float(args.mqsc_r1_min_selected_gain),
        use_vlm=not bool(args.mqsc_r1_disable_vlm),
        vlm_model=str(args.mqsc_r1_vlm_model),
        vlm_max_retries=int(args.mqsc_r1_vlm_max_retries),
        vlm_retry_sleep_sec=float(args.mqsc_r1_vlm_retry_sleep_sec),
        vlm_no_proxy=not bool(args.mqsc_r1_disable_no_proxy),
        allow_heuristic_decompose=not bool(args.mqsc_r1_disable_heuristic_decompose),
        write_debug_json=not bool(args.mqsc_r1_disable_debug_json),
    )
    VISTALS_CFG = VistaLsConfig(
        candidate_radii_m=_parse_float_list(args.vistals_candidate_radii_m, arg_name="--vistals_candidate_radii_m"),
        candidate_view_count=int(args.vistals_candidate_view_count),
        enable_vvd_replacement=bool(args.vistals_enable_vvd_replacement),
        prefer_visible_baseline=not bool(args.vistals_disable_visible_baseline_guard),
        camera_height_m=float(args.vistals_camera_height_m),
        max_snap_distance_m=float(args.vistals_max_snap_distance_m),
        target_sample_count=int(args.vistals_target_sample_count),
        scene_sample_count=int(args.vistals_scene_sample_count),
        max_ray_sample_count=int(args.vistals_max_ray_sample_count),
        occlusion_radius_m=float(args.vistals_occlusion_radius_m),
        min_visibility_score=float(args.vistals_min_visibility_score),
        visibility_tie_epsilon=float(args.vistals_visibility_tie_epsilon),
        path_efficiency_exponent=float(args.vistals_path_efficiency_exponent),
        r_min_m=float(args.vistals_r_min_m),
        r_max_m=float(args.vistals_r_max_m),
        radial_step_m=float(args.vistals_radial_step_m),
        angle_step_deg=float(args.vistals_angle_step_deg),
        shell_min_m=float(args.vistals_shell_min_m),
        shell_max_m=float(args.vistals_shell_max_m),
        relaxed_shell_min_m=float(args.vistals_relaxed_shell_min_m),
        relaxed_shell_max_m=float(args.vistals_relaxed_shell_max_m),
        min_clearance_m=float(args.vistals_min_clearance_m),
        min_component_size=int(args.vistals_min_component_size),
        size_tie_ratio=float(args.vistals_size_tie_ratio),
    )
    VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS = {
        x.strip() for x in str(args.vistals_apply_task_levels).split(",") if x.strip()
    }
    if not VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS:
        raise RuntimeError("--vistals_apply_task_levels resolved to empty set")

    output_log_dir = os.path.expanduser(args.output_log_dir)
    os.makedirs(output_log_dir, exist_ok=True)
    _setup_run_logging(output_log_dir)
    _set_seed(int(args.seed))

    enabled_task_levels = {x.strip() for x in str(args.task_levels).split(",") if x.strip()}
    if not enabled_task_levels:
        enabled_task_levels = set(TASK_LEVEL_ORDER)
    _tqdm_print(
        f"[Vista2MQSCRefine1] cfg start_ratio={args.start_ratio} end_ratio={args.end_ratio} "
        f"levels={sorted(enabled_task_levels)} max_steps={args.max_steps} "
        f"decision_num_min={args.decision_num_min} success_distance={args.success_distance} "
        f"quiet_nav_steps={bool(args.quiet_nav_steps)} output_log_dir={output_log_dir} "
        f"mqsc_r1_top_k={MQSC_R1_CFG.top_k} mqsc_r1_eps={MQSC_R1_CFG.cluster_eps} "
        f"mqsc_r1_vlm={MQSC_R1_CFG.use_vlm} mqsc_r1_vlm_model={MQSC_R1_CFG.vlm_model} "
        f"vistals_enable_vvd_replacement={VISTALS_CFG.enable_vvd_replacement} "
        f"vistals_apply_task_levels={sorted(VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS)} "
        f"vistals_radial_step={VISTALS_CFG.radial_step_m} vistals_angle_step={VISTALS_CFG.angle_step_deg}"
    )
    _write_json(Path(output_log_dir) / "run_args.json", vars(args))

    navigation_data_root = Path(os.path.expanduser(args.navigation_data_path))
    scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
    if not scene_data_paths:
        raise FileNotFoundError(f"No *.json.gz found under navigation_data_path={navigation_data_root}")
    scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]
    _tqdm_print(f"[Vista2MQSCRefine1] selected_scenes={len(scene_data_paths)}")

    out_name = f"refhm3d_seq_vista2mqsc_refine1_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_vista2mqsc_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
    if args.concise_description:
        out_name = f"refhm3d_seq_vista2mqsc_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
        eff_name = f"refhm3d_seq_vista2mqsc_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    output_path = Path(output_log_dir) / out_name
    effectiveness_path = Path(output_log_dir) / eff_name
    metrics_log_path = Path(output_log_dir) / f"vista2mqsc_live_metrics_{args.start_ratio}_{args.end_ratio}.log"
    _tqdm_print(f"[Vista2MQSCRefine1] output_json={output_path}")
    _tqdm_print(f"[Vista2MQSCRefine1] effectiveness_json={effectiveness_path}")
    _tqdm_print(f"[Vista2MQSCRefine1] live_metrics_log={metrics_log_path}")

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            result_dict = json.load(f)
        if "sequence" not in result_dict:
            result_dict["sequence"] = _iter_metric_rows(result_dict)
        existing_episodes = _existing_episode_keys(result_dict)
        sequence_compute_metric_results(result_dict)
        append_live_metrics(metrics_log_path, result_dict, latest={"event": "resume_existing_output"})
    else:
        result_dict = {"sequence": []}
        existing_episodes = set()

    if effectiveness_path.exists():
        with open(effectiveness_path, "r", encoding="utf-8") as f:
            effectiveness_dict = json.load(f)
        effectiveness_dict.setdefault("records", [])
    else:
        effectiveness_dict = {"records": []}

    pq3d_model = PQ3DModel(
        os.path.expanduser(args.pq3d_stage1_path),
        os.path.expanduser(args.pq3d_stage2_path),
        min_decision_num=int(args.decision_num_min),
    )

    for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
        scene_name = scene_data_path.name.split(".")[0]
        with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
            scene_data = json.load(f)
        region_to_annot_dict = scene_data.get("region_annotation", {})
        episode_mapping = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }
        all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

        for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
            pq3d_model.reset()
            decision_num = 0
            visited_frontier_set = set()
            start_position = cur_episode["start_position"]
            start_rotation = cur_episode["start_rotation"]
            episode_id, navigation_type = cur_episode["episode_id"], cur_episode["navigation_type"]
            episode_key = "_".join([scene_name, navigation_type, str(episode_id)])
            if episode_key in existing_episodes:
                continue

            sim_settings = OmegaConf.load("configs/habitat/goat_sim_config.yaml")
            goat_agent_setting = OmegaConf.load("configs/habitat/goat_agent_config.yaml")
            sim_settings["scene"] = resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), scene_name)
            abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
            sim = abstract_sim.simulator
            agent = abstract_sim.agent
            agent_state = habitat_sim.AgentState()
            agent_state.position = start_position
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)
            path_finder = sim.pathfinder
            top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
            fog_of_war_mask = np.zeros_like(top_down_map)
            area_thres_in_pixels = convert_meters_to_pixel(9, 512, sim)
            visibility_dist_in_pixels = convert_meters_to_pixel(3.0, 512, sim)
            out_episode_dir = Path(output_log_dir) / "process" / f"scene={scene_name}" / f"episode={episode_id}"
            out_episode_dir.mkdir(parents=True, exist_ok=True)

            try:
                for idx, cur_task_ref in enumerate(cur_episode["task_sequence"]):
                    task_t0 = time.perf_counter()
                    task_type, task_idx = cur_task_ref
                    if task_type not in enabled_task_levels:
                        continue
                    cur_task = episode_mapping[task_type][task_idx]
                    goals = [all_navigation_goals_dict[x] for x in cur_task["target_object_ids"]]
                    goal_positions = [
                        np.asarray(g.get("position", []), dtype=float).reshape(3)
                        for g in goals
                        if isinstance(g, dict) and len(g.get("position", [])) >= 3
                    ]
                    sentence, goal_category = build_sentence(
                        task_type,
                        cur_task,
                        all_navigation_goals_dict=all_navigation_goals_dict,
                        region_to_annot_dict=region_to_annot_dict,
                        concise_description=bool(args.concise_description),
                    )
                    _tqdm_print(
                        f"[vista2mqsc-refine1][task-start] scene={scene_name} ep={episode_id} "
                        f"task={idx} level={task_type} decision_start={decision_num}"
                    )
                    _tqdm_print(f"[vista2mqsc-refine1][task-desc] {sentence}")
                    _tqdm_print(
                        f"[vista2mqsc-refine1][nav] task={idx} entering navigation loop; "
                        f"each step = 12 turns + frontier + PQ3D. Vista2MQSC runs only on final object decisions."
                    )

                    total_steps = 0
                    episode_cum_distance = 0.0
                    prev_agent_state = agent.get_state()
                    sub_episode_start_position = np.asarray(prev_agent_state.position, dtype=float).copy()
                    goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
                    baseline_final_target_pos: Optional[np.ndarray] = None
                    final_selected_target_pos: Optional[np.ndarray] = None
                    hook_called = 0
                    hook_applied = 0
                    hook_logs: List[Dict[str, Any]] = []
                    task_pq3d_object_gap: Optional[float] = None
                    task_end_reason = "max_steps"
                    task_decision_start = int(decision_num)

                    task_dir = out_episode_dir / f"task={idx}"
                    task_dir.mkdir(parents=True, exist_ok=True)

                    while total_steps < int(args.max_steps):
                        color_list, depth_list, agent_state_list = [], [], []
                        if len(goto_color_list) > 6:
                            step = max(1, len(goto_color_list) // 6)
                            goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                            goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                            goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
                        color_list.extend(goto_color_list)
                        depth_list.extend(goto_depth_list)
                        agent_state_list.extend(goto_agent_state_list)

                        t_scan = time.perf_counter()
                        scan_rgb, scan_depth, scan_states, fog_of_war_mask, total_steps = _capture_scan_frames(
                            sim=sim,
                            agent=agent,
                            top_down_map=top_down_map,
                            fog_of_war_mask=fog_of_war_mask,
                            visibility_dist_in_pixels=visibility_dist_in_pixels,
                            total_steps=total_steps,
                            max_steps=int(args.max_steps),
                        )
                        scan_ms = (time.perf_counter() - t_scan) * 1000.0
                        color_list.extend(scan_rgb)
                        depth_list.extend(scan_depth)
                        agent_state_list.extend(scan_states)
                        if total_steps >= int(args.max_steps):
                            break

                        t_frontier = time.perf_counter()
                        agent_state = agent.get_state()
                        frontier_waypoints = detect_frontier_waypoints(
                            top_down_map,
                            fog_of_war_mask,
                            area_thres_in_pixels,
                            xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                            enable_visualization=False,
                        )
                        if len(frontier_waypoints) == 0:
                            frontier_waypoints = []
                        else:
                            frontier_waypoints = pixel_to_map_coors(
                                frontier_waypoints[:, ::-1], agent_state.position, top_down_map, sim
                            )
                        frontier_waypoints = [
                            w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set
                        ]
                        frontier_ms = (time.perf_counter() - t_frontier) * 1000.0

                        t_pq = time.perf_counter()
                        target_position, is_final = pq3d_model.decision(
                            color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num
                        )
                        pq_ms = (time.perf_counter() - t_pq) * 1000.0
                        if not bool(args.quiet_nav_steps):
                            _tqdm_print(
                                f"[vista2mqsc-refine1][step] task={idx} dec={decision_num} "
                                f"scan_ms={scan_ms:.0f} frontier_ms={frontier_ms:.0f} pq3d_ms={pq_ms:.0f} "
                                f"frames={len(color_list)} frontiers={len(frontier_waypoints)} final={bool(is_final)}"
                            )
                        if int(args.decision_log_interval) > 0 and (
                            decision_num % int(args.decision_log_interval) == 0
                        ):
                            _tqdm_print(
                                f"[vista2mqsc-refine1][decision] task={idx} dec={decision_num} "
                                f"target={np.asarray(target_position, dtype=float).reshape(-1)[:3].tolist()} "
                                f"final={bool(is_final)}"
                            )

                        used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                        aux = getattr(pq3d_model, "last_decision_aux", {}) or {}
                        module_info: Dict[str, Any] = {
                            "ok": True,
                            "module": "vista2mqsc",
                            "called": False,
                            "applied": False,
                            "reason": "non_final_decision",
                            "target_before": used_target.tolist(),
                            "target_after": used_target.tolist(),
                        }
                        if bool(is_final):
                            baseline_final_target_pos = used_target.copy()
                            task_pq3d_object_gap = float(aux.get("object_top1_top2_logit_gap", 0.0))
                            hook_called += 1
                            used_target, module_info = vista2mqsc_refine_hook(
                                sentence=sentence,
                                task_type=task_type,
                                scene_name=scene_name,
                                episode_id=int(episode_id),
                                task_id=int(idx),
                                decision_num=int(decision_num),
                                is_final=bool(is_final),
                                pq3d_model=pq3d_model,
                                target_position=used_target,
                                decision_aux=aux,
                                output_dir=task_dir,
                                path_finder=path_finder,
                                agent_position_xyz=np.asarray(agent.get_state().position, dtype=float).reshape(3),
                            )
                            _write_json(task_dir / "vista2mqsc" / f"dec_{decision_num:03d}_vista2mqsc.json", module_info)
                            if bool(module_info.get("applied", False)):
                                hook_applied += 1
                            final_selected_target_pos = used_target.copy()
                            hook_logs.append(
                                {
                                    "scene_name": scene_name,
                                    "episode_id": int(episode_id),
                                    "task_id": int(idx),
                                    "decision_num": int(decision_num),
                                    "sentence": sentence,
                                    "module_info": module_info,
                                }
                            )
                            _tqdm_print(
                                f"[vista2mqsc-refine1][final] task={idx} dec={decision_num} "
                                f"hook_called={hook_called} hook_applied={hook_applied} "
                                f"reason={module_info.get('reason')}"
                            )
                        else:
                            visited_frontier_set.add(tuple(np.round(used_target, 1)))

                        (
                            goto_color_list,
                            goto_depth_list,
                            goto_agent_state_list,
                            prev_agent_state,
                            total_steps,
                            episode_cum_distance,
                            follow_log,
                        ) = _follow_target(
                            path_finder=path_finder,
                            agent=agent,
                            sim=sim,
                            target=used_target,
                            prev_agent_state=prev_agent_state,
                            total_steps=total_steps,
                            max_steps=int(args.max_steps),
                            episode_cum_distance=float(episode_cum_distance),
                        )

                        _write_json(
                            task_dir / f"dec_{decision_num:03d}_vista2mqsc.json",
                            {
                                "task_id": int(idx),
                                "decision_num": int(decision_num),
                                "is_final": bool(is_final),
                                "target_used": used_target.tolist(),
                                "pq3d_aux": aux,
                                "module": module_info,
                                "module_info": module_info,
                                "follow": follow_log,
                                "steps_total_after_follow": int(total_steps),
                            },
                        )
                        decision_num += 1
                        if not bool(follow_log.get("ok", False)):
                            if not bool(is_final):
                                visited_frontier_set.add(tuple(np.round(used_target, 1)))
                                _tqdm_print(
                                    f"[vista2mqsc-refine1][frontier-follow-retry] task={idx} dec={decision_num - 1} "
                                    f"error={follow_log.get('error_type')} candidates={follow_log.get('candidate_attempt_count')}"
                                )
                                continue
                            task_end_reason = "follower_error"
                            break
                        if bool(is_final):
                            task_end_reason = "final_decision"
                            break

                    task_time = float(time.perf_counter() - task_t0)
                    agent_state = agent.get_state()
                    view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
                    sp = habitat_sim.MultiGoalShortestPath()
                    sp.requested_start = sub_episode_start_position
                    sp.requested_ends = view_points
                    start_end_geo_distance = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
                    ep = habitat_sim.MultiGoalShortestPath()
                    ep.requested_start = agent_state.position
                    ep.requested_ends = view_points
                    agent_end_geo_distance = float(ep.geodesic_distance) if path_finder.find_path(ep) else float("inf")
                    if (
                        task_end_reason == "follower_error"
                        or np.isinf(start_end_geo_distance)
                        or np.isinf(agent_end_geo_distance)
                    ):
                        sr, spl = 0.0, 0.0
                    else:
                        sr = 1.0 if agent_end_geo_distance <= float(args.success_distance) else 0.0
                        spl = float(
                            sr
                            * start_end_geo_distance
                            / max(start_end_geo_distance, max(float(episode_cum_distance), 1e-12))
                        )

                    baseline_target_to_goal_l2 = float("inf")
                    selected_target_to_goal_l2 = float("inf")
                    module_helpful = None
                    if len(goal_positions) > 0:
                        if baseline_final_target_pos is not None:
                            baseline_target_to_goal_l2 = float(
                                min(float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions)
                            )
                        if final_selected_target_pos is not None:
                            selected_target_to_goal_l2 = float(
                                min(float(np.linalg.norm(final_selected_target_pos - gp)) for gp in goal_positions)
                            )
                        if hook_applied > 0 and np.isfinite(baseline_target_to_goal_l2) and np.isfinite(selected_target_to_goal_l2):
                            module_helpful = bool(selected_target_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6)

                    final_module_info = hook_logs[-1].get("module_info", {}) if hook_logs else {}
                    row = {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(idx),
                        "task_level": task_type,
                        "navigation_type": navigation_type,
                        "sr": float(sr),
                        "spl": float(spl),
                        "object_category": goal_category,
                        "sentence": sentence,
                        "task_time_sec": float(task_time),
                        "steps_total": int(total_steps),
                        "decisions_total_episode_counter": int(decision_num),
                        "task_decision_start": int(task_decision_start),
                        "task_decision_count": int(decision_num - task_decision_start),
                        "end_reason": task_end_reason,
                        "start_goal_geo": float(start_end_geo_distance),
                        "end_goal_geo": float(agent_end_geo_distance),
                        "episode_cum_distance": float(episode_cum_distance),
                        "module_name": "vista2mqsc",
                        "module_hook_called": int(hook_called),
                        "module_hook_applied": int(hook_applied),
                        "module_helpful": module_helpful,
                        "module_reason": final_module_info.get("reason"),
                        "mqsc_r1_called": bool(final_module_info.get("mqsc_r1_called", False)),
                        "mqsc_r1_applied": bool(final_module_info.get("mqsc_r1_applied", False)),
                        "vistals_called": bool(final_module_info.get("vistals_called", False)),
                        "vistals_applied": bool(final_module_info.get("vistals_applied", False)),
                        "vistals_input_slot_index": final_module_info.get("vistals_input_slot_index"),
                        "vistals_input_slot_source": final_module_info.get("vistals_input_slot_source"),
                        "goal_positions": [gp.tolist() for gp in goal_positions],
                        "baseline_target_position": None
                        if baseline_final_target_pos is None
                        else baseline_final_target_pos.tolist(),
                        "selected_target_position": None
                        if final_selected_target_pos is None
                        else final_selected_target_pos.tolist(),
                        "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                        "selected_target_to_goal_l2": float(selected_target_to_goal_l2),
                        "baseline_target_to_goal_l2_valid": bool(np.isfinite(baseline_target_to_goal_l2)),
                        "selected_target_to_goal_l2_valid": bool(np.isfinite(selected_target_to_goal_l2)),
                        "pq3d_object_top1_top2_logit_gap": task_pq3d_object_gap,
                    }
                    _append_result_row(result_dict, navigation_type, row)
                    effectiveness_record = {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(idx),
                        "task_level": task_type,
                        "navigation_type": navigation_type,
                        "module_name": "vista2mqsc",
                        "module_hook_called": int(hook_called),
                        "module_hook_applied": int(hook_applied),
                        "module_helpful": module_helpful,
                        "module_reason": final_module_info.get("reason"),
                        "mqsc_r1_called": bool(final_module_info.get("mqsc_r1_called", False)),
                        "mqsc_r1_applied": bool(final_module_info.get("mqsc_r1_applied", False)),
                        "vistals_called": bool(final_module_info.get("vistals_called", False)),
                        "vistals_applied": bool(final_module_info.get("vistals_applied", False)),
                        "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                        "selected_target_to_goal_l2": float(selected_target_to_goal_l2),
                        "baseline_target_to_goal_l2_valid": bool(np.isfinite(baseline_target_to_goal_l2)),
                        "selected_target_to_goal_l2_valid": bool(np.isfinite(selected_target_to_goal_l2)),
                        "hook_logs": hook_logs,
                    }
                    effectiveness_dict["records"].append(effectiveness_record)
                    _write_json(task_dir / "summary.json", row)
                    _write_json(task_dir / "effectiveness.json", effectiveness_record)

                    _tqdm_print(
                        f"[vista2mqsc-refine1] scene={scene_name} ep={episode_id} task={idx} "
                        f"SR={sr:.1f} SPL={spl:.4f} time={task_time:.3f}s steps={total_steps} "
                        f"task_decisions={decision_num - task_decision_start} "
                        f"hook_called={hook_called} hook_applied={hook_applied} helpful={module_helpful} "
                        f"reason={final_module_info.get('reason')}"
                    )
                    append_live_metrics(
                        metrics_log_path,
                        result_dict,
                        latest={
                            "scene_name": scene_name,
                            "episode_id": int(episode_id),
                            "task_id": int(idx),
                            "task_level": task_type,
                            "sr": float(sr),
                            "spl": float(spl),
                        },
                    )
            finally:
                sim.close()

            existing_episodes.add(episode_key)
            _write_json(output_path, result_dict)
            _write_json(effectiveness_path, effectiveness_dict)
            sequence_compute_metric_results(result_dict)

    _write_json(output_path, result_dict)
    _write_json(effectiveness_path, effectiveness_dict)
    sequence_compute_metric_results(result_dict)


if __name__ == "__main__":
    main()
