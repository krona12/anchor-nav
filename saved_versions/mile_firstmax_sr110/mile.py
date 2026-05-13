from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MileConfig:
    decision_radius_m: float = 0.75
    candidate_radii_m: Tuple[float, ...] = (0.5, 0.75)
    candidate_view_count: int = 20
    enable_vvd_replacement: bool = False
    prefer_visible_baseline: bool = True
    camera_height_m: float = 1.50
    max_snap_distance_m: float = 0.60
    target_sample_count: int = 1000
    scene_sample_count: int = 0
    max_ray_sample_count: int = 1000
    occlusion_radius_m: float = 0.05
    min_visibility_score: float = 0.0
    visibility_tie_epsilon: float = 0.02
    rng_seed: int = 17


class MileDecisionError(RuntimeError):
    pass


class MileInputError(MileDecisionError):
    pass


class MileRejectedError(MileDecisionError):
    pass


def _as_np3(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise RuntimeError(f"{name} must have at least 3 values, got shape={arr.shape}")
    out = arr[:3].astype(float)
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"{name} contains non-finite values: {out!r}")
    return out


def _model_xyz_to_habitat_xyz(points_model: np.ndarray) -> np.ndarray:
    arr = np.asarray(points_model, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise RuntimeError(f"model point array must have >=3 columns, got shape={arr.shape}")
    return arr[:, [0, 2, 1]].astype(float)


def _object_points_habitat(*, rep: Any, slot_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_cloud = np.asarray(getattr(rep, "point_cloud", np.zeros((0, 6))), dtype=float)
    object_mask = np.asarray(getattr(rep, "object_mask", np.zeros((0, 0))))
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise RuntimeError(f"representation point_cloud has invalid shape={point_cloud.shape}")
    if object_mask.ndim != 2:
        raise RuntimeError(f"representation object_mask has invalid shape={object_mask.shape}")
    if object_mask.shape[0] != point_cloud.shape[0]:
        raise RuntimeError(f"point_cloud/object_mask row mismatch: {point_cloud.shape} vs {object_mask.shape}")
    slot = int(slot_index)
    if slot < 0 or slot >= object_mask.shape[1]:
        raise RuntimeError(f"slot_index={slot} out of object_mask range={object_mask.shape[1]}")
    mask = object_mask[:, slot].astype(bool)
    if int(mask.sum()) == 0:
        raise MileRejectedError(f"zero_object_points slot_index={slot}")
    obj = _model_xyz_to_habitat_xyz(point_cloud[mask, :3])
    blockers = _model_xyz_to_habitat_xyz(point_cloud[~mask, :3])
    scene_all = _model_xyz_to_habitat_xyz(point_cloud[:, :3])
    return obj, blockers, scene_all


def _sample_rows(points: np.ndarray, max_count: int, rng: np.random.RandomState) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] <= int(max_count):
        return arr
    idx = rng.choice(arr.shape[0], size=int(max_count), replace=False)
    return arr[idx]


def _generate_view_candidates(
    *,
    center_xyz: np.ndarray,
    agent_xyz: np.ndarray,
    path_finder: Any,
    cfg: MileConfig,
) -> Tuple[List[Tuple[int, np.ndarray]], List[Dict[str, Any]]]:
    center = _as_np3(center_xyz, name="object_center")
    agent = _as_np3(agent_xyz, name="agent_xyz")
    n = int(cfg.candidate_view_count)
    if n <= 0:
        raise RuntimeError(f"candidate_view_count must be positive, got {n}")
    radii = tuple(float(r) for r in getattr(cfg, "candidate_radii_m", ()) if float(r) > 0.0)
    if len(radii) == 0:
        radii = (float(cfg.decision_radius_m),)
    agent_island = int(path_finder.get_island(agent))
    accepted: List[Tuple[int, np.ndarray]] = []
    records: List[Dict[str, Any]] = []
    accepted_keys = set()

    def _point_key(p: np.ndarray) -> Tuple[float, float, float]:
        return tuple(float(x) for x in np.round(np.asarray(p, dtype=float).reshape(3), 2))

    def _angle_diff(a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    cand_idx = 0
    for radius_idx, radius in enumerate(radii):
        for i in range(n):
            theta = 2.0 * math.pi * float(i) / float(n)
            raw = np.array(
                [
                    center[0] + radius * math.cos(theta),
                    agent[1],
                    center[2] + radius * math.sin(theta),
                ],
                dtype=float,
            )
            raw_navigable = bool(np.all(np.isfinite(raw)) and path_finder.is_navigable(raw))
            raw_same_island = bool(raw_navigable and int(path_finder.get_island(raw)) == agent_island)
            key = _point_key(raw)
            duplicate = key in accepted_keys
            accepted_now = bool(raw_navigable and raw_same_island and not duplicate)
            records.append(
                {
                    "candidate_index": int(cand_idx),
                    "radius_index": int(radius_idx),
                    "angle_index": int(i),
                    "radius_m": float(radius),
                    "angle_rad": float(theta),
                    "raw_xyz": raw.tolist(),
                    "raw_navigable": bool(raw_navigable),
                    "raw_same_island": bool(raw_same_island),
                    "raw_duplicate": bool(duplicate),
                    "snapped_xyz": None,
                    "snap_distance_m": None,
                    "snapped_navigable": None,
                    "snapped_same_island": None,
                    "snapped_preserves_sector": None,
                    "accepted": bool(accepted_now),
                    "accepted_stage": "raw_navigable" if accepted_now else None,
                }
            )
            if accepted_now:
                accepted_keys.add(key)
                accepted.append((int(cand_idx), raw.copy()))
            cand_idx += 1

    for rec in records:
        if bool(rec.get("accepted", False)):
            continue
        raw = np.asarray(rec["raw_xyz"], dtype=float).reshape(3)
        snapped = np.asarray(path_finder.snap_point(raw, island_index=agent_island), dtype=float).reshape(3)
        snap_dist = float(np.linalg.norm(snapped - raw)) if np.all(np.isfinite(snapped)) else float("inf")
        snapped_navigable = bool(np.all(np.isfinite(snapped)) and path_finder.is_navigable(snapped))
        snapped_same_island = bool(snapped_navigable and int(path_finder.get_island(snapped)) == agent_island)
        within_snap_limit = bool(snap_dist <= float(cfg.max_snap_distance_m))
        raw_angle = float(rec["angle_rad"])
        snapped_vec = snapped - center
        snapped_angle = math.atan2(float(snapped_vec[2]), float(snapped_vec[0])) if np.all(np.isfinite(snapped_vec)) else float("nan")
        preserves_sector = bool(np.isfinite(snapped_angle) and _angle_diff(snapped_angle, raw_angle) <= (math.pi / float(n)))
        duplicate = _point_key(snapped) in accepted_keys if np.all(np.isfinite(snapped)) else True
        accepted_now = bool(snapped_navigable and snapped_same_island and within_snap_limit and preserves_sector and not duplicate)
        rec.update(
            {
                "snapped_xyz": snapped.tolist() if np.all(np.isfinite(snapped)) else None,
                "snap_distance_m": float(snap_dist),
                "max_snap_distance_m": float(cfg.max_snap_distance_m),
                "snapped_navigable": bool(snapped_navigable),
                "snapped_same_island": bool(snapped_same_island),
                "snapped_preserves_sector": bool(preserves_sector),
                "snapped_duplicate": bool(duplicate),
                "snapped_within_limit": bool(within_snap_limit),
                "accepted": bool(accepted_now),
                "accepted_stage": "snapped_nearest_navmesh" if accepted_now else None,
            }
        )
        if accepted_now:
            accepted_keys.add(_point_key(snapped))
            accepted.append((int(rec["candidate_index"]), snapped.copy()))
    return accepted, records


def _visibility_scores(
    *,
    viewpoints: Sequence[Tuple[int, np.ndarray]],
    target_points: np.ndarray,
    scene_points: np.ndarray,
    agent_position_xyz: Sequence[float],
    cfg: MileConfig,
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    target = _sample_rows(target_points, int(cfg.target_sample_count), rng)
    scene = np.asarray(scene_points, dtype=float)
    agent = _as_np3(agent_position_xyz, name="agent_position_xyz")
    scene_sample_limit = int(cfg.scene_sample_count)
    if scene_sample_limit > 0:
        scene = _sample_rows(scene, scene_sample_limit, rng)
    if target.shape[0] == 0:
        raise RuntimeError("target point sample is empty")
    tree = cKDTree(scene) if scene.shape[0] > 0 else None
    out: List[Dict[str, Any]] = []
    tau = float(cfg.occlusion_radius_m)
    if tau <= 0.0:
        raise RuntimeError(f"occlusion_radius_m must be positive, got {cfg.occlusion_radius_m}")
    max_samples = int(cfg.max_ray_sample_count)
    if max_samples <= 0:
        raise RuntimeError(f"max_ray_sample_count must be positive, got {cfg.max_ray_sample_count}")
    for accepted_rank, (raw_candidate_index, vp) in enumerate(viewpoints):
        view = _as_np3(vp, name="viewpoint")
        camera_view = view + np.array([0.0, float(cfg.camera_height_m), 0.0], dtype=float)
        ray = target - camera_view.reshape(1, 3)
        dist = np.linalg.norm(ray, axis=1)
        valid = dist > 1e-6
        if not np.any(valid):
            visible_count = 0
            target_count = int(target.shape[0])
            score = 0.0
        elif tree is None:
            visible_count = int(valid.sum())
            target_count = int(valid.sum())
            score = 1.0
        else:
            t_valid = target[valid]
            ray_valid = ray[valid]
            dist_valid = dist[valid]
            visible_flags: List[bool] = []
            for target_point, direction_vec, view_distance in zip(t_valid, ray_valid, dist_valid):
                view_distance_f = float(view_distance)
                sample_count = min(max_samples, max(2, int(view_distance_f / tau) + 1))
                direction = direction_vec / view_distance_f
                ray_start = tau
                ray_stop = view_distance_f - tau
                if ray_stop <= ray_start:
                    visible_flags.append(True)
                    continue
                sample_count = min(max_samples, max(2, int((ray_stop - ray_start) / tau) + 1))
                distances_along_ray = np.linspace(ray_start, ray_stop, num=sample_count, dtype=float)
                q = camera_view.reshape(1, 3) + distances_along_ray.reshape(-1, 1) * direction.reshape(1, 3)
                d, _ = tree.query(q, k=1)
                visible_flags.append(bool(not np.any(d < tau)))
            visible_count = int(np.asarray(visible_flags, dtype=bool).sum())
            target_count = int(t_valid.shape[0])
            score = float(visible_count / max(target_count, 1))
        out.append(
            {
                "candidate_index": int(raw_candidate_index),
                "accepted_rank": int(accepted_rank),
                "viewpoint_xyz": view.tolist(),
                "camera_viewpoint_xyz": camera_view.tolist(),
                "agent_l2_distance_m": float(np.linalg.norm(view - agent)),
                "visibility_score": float(score),
                "visible_target_points": int(visible_count),
                "target_sample_count": int(target_count),
                "scene_sample_count": int(scene.shape[0]),
                "scene_sample_limit": int(scene_sample_limit),
                "scene_points_include_target": False,
                "camera_height_m": float(cfg.camera_height_m),
                "occlusion_radius_m": float(cfg.occlusion_radius_m),
                "max_ray_sample_count": int(cfg.max_ray_sample_count),
            }
        )
    return out


def visibility_based_viewpoint_decision(
    *,
    rep: Any,
    selected_slot_index: int,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: MileConfig,
    baseline_target_xyz: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(cfg.rng_seed) + int(selected_slot_index))
    try:
        object_points, blocker_points, scene_points = _object_points_habitat(rep=rep, slot_index=int(selected_slot_index))
    except MileRejectedError as exc:
        return {
            "called": True,
            "selected_slot_index": int(selected_slot_index),
            "object_point_count": 0,
            "scene_point_count": None,
            "scene_points_include_target": None,
            "object_center_xyz": None,
            "decision_radius_m": float(cfg.decision_radius_m),
            "candidate_view_count": int(cfg.candidate_view_count),
            "camera_height_m": float(cfg.camera_height_m),
            "candidate_records": [],
            "viewpoint_applied": False,
            "rejected_reason": str(exc),
            "best_viewpoint_xyz": None,
            "best_visibility_score": None,
            "visibility_records": [],
            "min_visibility_score": float(cfg.min_visibility_score),
        }
    sampled_object_points = _sample_rows(object_points, int(cfg.target_sample_count), rng)
    center = sampled_object_points.mean(axis=0)
    candidates, candidate_records = _generate_view_candidates(
        center_xyz=center,
        agent_xyz=_as_np3(agent_position_xyz, name="agent_position_xyz"),
        path_finder=path_finder,
        cfg=cfg,
    )
    info: Dict[str, Any] = {
        "called": True,
        "selected_slot_index": int(selected_slot_index),
        "object_point_count": int(object_points.shape[0]),
        "target_decision_sample_count": int(sampled_object_points.shape[0]),
        "scene_point_count": int(scene_points.shape[0]),
        "blocker_point_count": int(blocker_points.shape[0]),
        "scene_points_include_target": False,
        "object_center_xyz": center.tolist(),
        "decision_radius_m": float(cfg.decision_radius_m),
        "candidate_view_count": int(cfg.candidate_view_count),
        "candidate_radii_m": [float(x) for x in tuple(getattr(cfg, "candidate_radii_m", ()) or (cfg.decision_radius_m,))],
        "camera_height_m": float(cfg.camera_height_m),
        "candidate_records": candidate_records,
        "visibility_tie_epsilon": float(cfg.visibility_tie_epsilon),
        }
    if len(candidates) == 0:
        info.update(
            {
                "viewpoint_applied": False,
                "rejected_reason": "no_navigable_candidate_viewpoint",
                "best_viewpoint_xyz": None,
                "best_visibility_score": None,
                "visibility_records": [],
            }
        )
        return info
    baseline_candidate_record: Optional[Dict[str, Any]] = None
    if baseline_target_xyz is not None:
        agent = _as_np3(agent_position_xyz, name="agent_position_xyz")
        agent_island = int(path_finder.get_island(agent))
        baseline_raw = _as_np3(baseline_target_xyz, name="baseline_target_xyz")
        baseline_nav = np.asarray(path_finder.snap_point(baseline_raw, island_index=agent_island), dtype=float).reshape(3)
        baseline_nav_ok = bool(np.all(np.isfinite(baseline_nav)) and path_finder.is_navigable(baseline_nav))
        baseline_same_island = bool(baseline_nav_ok and int(path_finder.get_island(baseline_nav)) == agent_island)
        baseline_candidate_record = {
            "candidate_index": -1,
            "raw_xyz": baseline_raw.tolist(),
            "snapped_xyz": baseline_nav.tolist() if np.all(np.isfinite(baseline_nav)) else None,
            "raw_navigable": bool(np.all(np.isfinite(baseline_raw)) and path_finder.is_navigable(baseline_raw)),
            "raw_same_island": False,
            "snap_distance_m": float(np.linalg.norm(baseline_nav - baseline_raw)) if np.all(np.isfinite(baseline_nav)) else float("inf"),
            "snapped_navigable": bool(baseline_nav_ok),
            "snapped_same_island": bool(baseline_same_island),
            "accepted": bool(baseline_same_island),
            "accepted_stage": "baseline_snapped_navmesh" if baseline_same_island else None,
            "is_baseline_candidate": True,
        }
        info["baseline_candidate_record"] = baseline_candidate_record
        if baseline_same_island:
            candidates.append((-1, baseline_nav.copy()))
    visibility_records = _visibility_scores(
        viewpoints=candidates,
        target_points=sampled_object_points,
        scene_points=blocker_points,
        agent_position_xyz=agent_position_xyz,
        cfg=cfg,
        rng=rng,
    )
    for rec in visibility_records:
        if int(rec.get("candidate_index", 0)) == -1:
            rec["is_baseline_candidate"] = True
            rec["baseline_raw_target_xyz"] = _as_np3(baseline_target_xyz, name="baseline_target_xyz").tolist() if baseline_target_xyz is not None else None
        else:
            rec["is_baseline_candidate"] = False
    candidate_scores = np.asarray([float(x["visibility_score"]) for x in visibility_records], dtype=float)
    best_local = int(np.argmax(candidate_scores))
    best = visibility_records[best_local]
    score = float(best["visibility_score"])
    min_score = max(0.0, float(cfg.min_visibility_score))
    rejected_reason: Optional[str] = None if score > min_score else "no_positive_visibility_score"
    applied = rejected_reason is None
    info.update(
        {
            "viewpoint_applied": bool(applied),
            "rejected_reason": rejected_reason,
            "best_viewpoint_xyz": best["viewpoint_xyz"] if applied else None,
            "best_visibility_score": float(score),
            "best_candidate_index": int(best.get("candidate_index", -1)),
            "best_candidate_agent_l2_distance_m": float(best.get("agent_l2_distance_m", float("inf"))),
            "selection_policy": "msgnav_first_max_visibility",
            "visibility_records": visibility_records,
            "min_visibility_score": float(cfg.min_visibility_score),
            "visibility_tie_epsilon": float(cfg.visibility_tie_epsilon),
        }
    )
    return info


def correct_final_decision_with_mile(
    *,
    rep: Any,
    decision_aux: Mapping[str, Any],
    baseline_target_xyz: np.ndarray,
    output_dir: Path,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: MileConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    missing_keys = [k for k in ("is_object_decision", "real_object_decision_idx") if k not in decision_aux]
    if missing_keys:
        raise MileInputError(f"decision_aux missing required key(s): {missing_keys}")
    if not bool(decision_aux["is_object_decision"]):
        raise MileInputError("mile was called for a non-object final decision")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_memory_index = int(decision_aux["real_object_decision_idx"])
    baseline_target = _as_np3(baseline_target_xyz, name="baseline_target_xyz")

    visibility = visibility_based_viewpoint_decision(
        rep=rep,
        selected_slot_index=int(baseline_memory_index),
        path_finder=path_finder,
        agent_position_xyz=agent_position_xyz,
        cfg=cfg,
        baseline_target_xyz=baseline_target,
    )
    target_source = "baseline_vvd_viewpoint"
    if not bool(visibility["viewpoint_applied"]):
        failure_info = {
            "mile_called": True,
            "module_scope": "lastmile_vvd_only",
            "decision_policy": "MSGNav-style visibility_based_viewpoint_decision on baseline final object slot",
            "uses_language_or_vision_model": False,
            "baseline_memory_index": int(baseline_memory_index),
            "baseline_target_xyz": baseline_target.tolist(),
            "selected_slot_index": int(baseline_memory_index),
            "visibility": visibility,
            "target_source": "vvd_no_visible_candidate_explicit_keep_baseline",
            "viewpoint_correction_applied": False,
            "correction_applied": False,
            "corrected_target_xyz": baseline_target.tolist(),
            "rejected_reason": visibility.get("rejected_reason"),
        }
        with open(output_dir / "mile_decision.json", "w", encoding="utf-8") as f:
            json.dump(failure_info, f, ensure_ascii=False, indent=2)
        return baseline_target.copy(), failure_info
    corrected_target = _as_np3(visibility["best_viewpoint_xyz"], name="best_viewpoint_xyz")

    correction_applied = bool(np.linalg.norm(corrected_target - baseline_target) > 1e-6)
    info = {
        "mile_called": True,
        "module_scope": "lastmile_vvd_only",
        "decision_policy": "MSGNav-style visibility_based_viewpoint_decision on baseline final object slot",
        "uses_language_or_vision_model": False,
        "baseline_memory_index": int(baseline_memory_index),
        "baseline_target_xyz": baseline_target.tolist(),
        "selected_slot_index": int(baseline_memory_index),
        "visibility": visibility,
        "target_source": target_source,
        "viewpoint_correction_applied": bool(visibility.get("viewpoint_applied", False)),
        "correction_applied": bool(correction_applied),
        "corrected_target_xyz": corrected_target.tolist(),
    }
    with open(output_dir / "mile_decision.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return corrected_target, info


__all__ = [
    "MileConfig",
    "MileDecisionError",
    "MileInputError",
    "MileRejectedError",
    "correct_final_decision_with_mile",
    "visibility_based_viewpoint_decision",
]
