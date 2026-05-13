from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VistaLsConfig:
    decision_radius_m: float = 0.75
    candidate_radii_m: Tuple[float, ...] = (0.5, 0.75)
    candidate_view_count: int = 20
    enable_vvd_replacement: bool = False
    prefer_visible_baseline: bool = True
    camera_height_m: float = 1.50
    max_snap_distance_m: float = 0.60
    target_sample_count: int = 300
    scene_sample_count: int = 0
    max_ray_sample_count: int = 1000
    occlusion_radius_m: float = 0.05
    min_visibility_score: float = 0.02
    visibility_tie_epsilon: float = 0.0
    path_efficiency_exponent: float = 0.15
    r_min_m: float = 0.30
    r_max_m: float = 1.30
    radial_step_m: float = 0.05
    angle_step_deg: float = 5.0
    shell_min_m: float = 0.35
    shell_max_m: float = 1.20
    relaxed_shell_min_m: float = 0.30
    relaxed_shell_max_m: float = 1.35
    min_clearance_m: float = 0.10
    min_component_size: int = 3
    size_tie_ratio: float = 0.85
    rng_seed: int = 17


class VistaLsDecisionError(RuntimeError):
    pass


class VistaLsInputError(VistaLsDecisionError):
    pass


class VistaLsRejectedError(VistaLsDecisionError):
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
        raise VistaLsRejectedError(f"zero_object_points slot_index={slot}")
    obj = _model_xyz_to_habitat_xyz(point_cloud[mask, :3])
    blockers = _model_xyz_to_habitat_xyz(point_cloud[~mask, :3])
    scene_all = _model_xyz_to_habitat_xyz(point_cloud[:, :3])
    return obj, blockers, scene_all


def _sample_rows(points: np.ndarray, max_count: int, rng: np.random.RandomState) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if max_count <= 0 or arr.shape[0] <= int(max_count):
        return arr
    idx = rng.choice(arr.shape[0], size=int(max_count), replace=False)
    return arr[idx]


def _point_key(point: np.ndarray, decimals: int = 3) -> Tuple[float, float, float]:
    return tuple(float(x) for x in np.round(np.asarray(point, dtype=float).reshape(3), int(decimals)))


def _snap_point(path_finder: Any, point: np.ndarray, island_index: int) -> np.ndarray:
    try:
        snapped = path_finder.snap_point(point=point, island_index=int(island_index))
    except TypeError:
        try:
            snapped = path_finder.snap_point(point, island_index=int(island_index))
        except TypeError:
            snapped = path_finder.snap_point(point)
    return np.asarray(snapped, dtype=float).reshape(3)


def _safe_get_island(path_finder: Any, point: np.ndarray) -> int:
    return int(path_finder.get_island(np.asarray(point, dtype=float).reshape(3)))


def _safe_is_navigable(path_finder: Any, point: np.ndarray) -> bool:
    arr = np.asarray(point, dtype=float).reshape(3)
    return bool(np.all(np.isfinite(arr)) and path_finder.is_navigable(arr))


def _shortest_path_distance(path_finder: Any, start: np.ndarray, end: np.ndarray) -> float:
    start_arr = _as_np3(start, name="path_start")
    end_arr = _as_np3(end, name="path_end")
    try:
        import habitat_sim  # type: ignore

        path = habitat_sim.ShortestPath()
        path.requested_start = start_arr
        path.requested_end = end_arr
        if bool(path_finder.find_path(path)):
            return float(path.geodesic_distance)
    except Exception:
        pass
    for attr in ("geodesic_distance", "distance"):
        fn = getattr(path_finder, attr, None)
        if fn is None:
            continue
        try:
            value = float(fn(start_arr, end_arr))
            if np.isfinite(value):
                return value
        except Exception:
            continue
    return float(np.linalg.norm(start_arr - end_arr))


def _polar_radii(cfg: VistaLsConfig) -> np.ndarray:
    r_min = float(cfg.r_min_m)
    r_max = float(cfg.r_max_m)
    step = float(cfg.radial_step_m)
    if r_min <= 0.0 or r_max <= 0.0 or r_max < r_min:
        raise RuntimeError(f"invalid VISTA-LS radius bounds: r_min={r_min} r_max={r_max}")
    if step <= 0.0:
        raise RuntimeError(f"radial_step_m must be positive, got {step}")
    count = int(math.floor((r_max - r_min) / step + 1e-9)) + 1
    radii = r_min + np.arange(count, dtype=float) * step
    if radii.size == 0 or radii[-1] < r_max - 1e-6:
        radii = np.append(radii, r_max)
    return radii


def _angle_count(cfg: VistaLsConfig) -> int:
    step = float(cfg.angle_step_deg)
    if step <= 0.0:
        raise RuntimeError(f"angle_step_deg must be positive, got {step}")
    return max(4, int(math.ceil(360.0 / step)))


def _generate_level_set_candidates(
    *,
    center_xyz: np.ndarray,
    agent_xyz: np.ndarray,
    path_finder: Any,
    cfg: VistaLsConfig,
) -> Tuple[List[Tuple[int, np.ndarray]], List[Dict[str, Any]], int]:
    center = _as_np3(center_xyz, name="object_center")
    agent = _as_np3(agent_xyz, name="agent_xyz")
    radii = _polar_radii(cfg)
    n_angles = _angle_count(cfg)
    agent_island = _safe_get_island(path_finder, agent)
    accepted: List[Tuple[int, np.ndarray]] = []
    records: List[Dict[str, Any]] = []
    accepted_keys = set()
    candidate_index = 0

    for radius_index, radius in enumerate(radii):
        for angle_index in range(n_angles):
            theta = 2.0 * math.pi * float(angle_index) / float(n_angles)
            raw = np.array(
                [
                    center[0] + float(radius) * math.cos(theta),
                    agent[1],
                    center[2] + float(radius) * math.sin(theta),
                ],
                dtype=float,
            )
            raw_navigable = _safe_is_navigable(path_finder, raw)
            raw_same_island = bool(raw_navigable and _safe_get_island(path_finder, raw) == agent_island)
            selected_point: Optional[np.ndarray] = raw.copy() if raw_same_island else None
            accepted_stage: Optional[str] = "raw_navigable_same_island" if raw_same_island else None
            snapped: Optional[np.ndarray] = None
            snap_distance = None
            snapped_navigable = None
            snapped_same_island = None
            snapped_within_limit = None

            if selected_point is None:
                try:
                    snapped = _snap_point(path_finder, raw, agent_island)
                    snap_distance = float(np.linalg.norm(snapped - raw)) if np.all(np.isfinite(snapped)) else float("inf")
                    snapped_navigable = _safe_is_navigable(path_finder, snapped)
                    snapped_same_island = bool(
                        snapped_navigable and _safe_get_island(path_finder, snapped) == agent_island
                    )
                    snapped_within_limit = bool(snap_distance <= float(cfg.max_snap_distance_m))
                    if snapped_same_island and snapped_within_limit:
                        selected_point = snapped.copy()
                        accepted_stage = "snapped_navigable_same_island"
                except Exception:
                    snapped = None
                    snap_distance = float("inf")
                    snapped_navigable = False
                    snapped_same_island = False
                    snapped_within_limit = False

            duplicate = True
            if selected_point is not None and np.all(np.isfinite(selected_point)):
                key = _point_key(selected_point)
                duplicate = key in accepted_keys
                if not duplicate:
                    accepted_keys.add(key)
                    accepted.append((int(candidate_index), selected_point.copy()))
            accepted_now = bool(selected_point is not None and not duplicate)
            records.append(
                {
                    "candidate_index": int(candidate_index),
                    "radius_index": int(radius_index),
                    "angle_index": int(angle_index),
                    "radius_m": float(radius),
                    "angle_rad": float(theta),
                    "raw_xyz": raw.tolist(),
                    "raw_navigable": bool(raw_navigable),
                    "raw_same_island": bool(raw_same_island),
                    "snapped_xyz": snapped.tolist() if snapped is not None and np.all(np.isfinite(snapped)) else None,
                    "snap_distance_m": snap_distance,
                    "snapped_navigable": snapped_navigable,
                    "snapped_same_island": snapped_same_island,
                    "snapped_within_limit": snapped_within_limit,
                    "accepted": bool(accepted_now),
                    "reachable_ok": bool(accepted_now),
                    "accepted_stage": accepted_stage if accepted_now else None,
                    "duplicate": bool(duplicate),
                    "is_baseline_candidate": False,
                }
            )
            candidate_index += 1
    return accepted, records, n_angles


def _visibility_scores(
    *,
    viewpoints: Sequence[Tuple[int, np.ndarray]],
    target_points: np.ndarray,
    scene_points: np.ndarray,
    agent_position_xyz: Sequence[float],
    cfg: VistaLsConfig,
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
            for direction_vec, view_distance in zip(ray_valid, dist_valid):
                view_distance_f = float(view_distance)
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


def _target_surface_distance(view: np.ndarray, target_points: np.ndarray) -> float:
    target = np.asarray(target_points, dtype=float)
    if target.shape[0] == 0:
        return float("inf")
    ground = target[:, [0, 2]]
    view_ground = np.asarray(view, dtype=float).reshape(3)[[0, 2]]
    return float(np.min(np.linalg.norm(ground - view_ground.reshape(1, 2), axis=1)))


def _clearance_distance(view: np.ndarray, blocker_tree: Optional[cKDTree]) -> float:
    if blocker_tree is None:
        return float("inf")
    view_ground = np.asarray(view, dtype=float).reshape(3)[[0, 2]]
    dist, _ = blocker_tree.query(view_ground.reshape(1, 2), k=1)
    return float(np.asarray(dist, dtype=float).reshape(-1)[0])


def _augment_records_for_level_set(
    *,
    visibility_records: List[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
    target_points: np.ndarray,
    blocker_points: np.ndarray,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: VistaLsConfig,
) -> List[Dict[str, Any]]:
    by_index = {int(rec["candidate_index"]): rec for rec in candidate_records}
    blocker_ground = np.asarray(blocker_points, dtype=float)[:, [0, 2]] if blocker_points.shape[0] > 0 else np.zeros((0, 2))
    blocker_tree = cKDTree(blocker_ground) if blocker_ground.shape[0] > 0 else None
    agent = _as_np3(agent_position_xyz, name="agent_position_xyz")
    for rec in visibility_records:
        cand_idx = int(rec.get("candidate_index", -1))
        meta = by_index.get(cand_idx, {})
        for key in (
            "radius_index",
            "angle_index",
            "radius_m",
            "angle_rad",
            "raw_xyz",
            "accepted_stage",
            "reachable_ok",
            "is_baseline_candidate",
        ):
            if key in meta:
                rec[key] = meta[key]
        view = _as_np3(rec["viewpoint_xyz"], name="viewpoint_xyz")
        surface_dist = _target_surface_distance(view, target_points)
        clearance = _clearance_distance(view, blocker_tree)
        rec.update(
            {
                "target_surface_distance_m": float(surface_dist),
                "clearance_m": float(clearance),
                "shell_ok": bool(float(cfg.shell_min_m) <= surface_dist <= float(cfg.shell_max_m)),
                "clearance_ok": bool(clearance >= float(cfg.min_clearance_m)),
                "component_id": None,
                "component_size": 0,
                "medial_distance_to_boundary": None,
                "selected_by_vista_ls": False,
                "path_geodesic_distance_m": float(_shortest_path_distance(path_finder, agent, view)),
            }
        )
    return visibility_records


def _grid_angle_delta(a: int, b: int, angle_count: int) -> int:
    raw = abs(int(a) - int(b))
    return min(raw, int(angle_count) - raw)


def _are_component_neighbors(a: Dict[str, Any], b: Dict[str, Any], *, angle_count: int, grid_step_m: float) -> bool:
    if a.get("radius_index") is None or b.get("radius_index") is None:
        return False
    dr = abs(int(a["radius_index"]) - int(b["radius_index"]))
    da = _grid_angle_delta(int(a["angle_index"]), int(b["angle_index"]), int(angle_count))
    if dr <= 1 and da <= 1 and (dr + da) > 0:
        return True
    pa = _as_np3(a["viewpoint_xyz"], name="viewpoint_a")[[0, 2]]
    pb = _as_np3(b["viewpoint_xyz"], name="viewpoint_b")[[0, 2]]
    return bool(np.linalg.norm(pa - pb) <= 1.5 * max(float(grid_step_m), 1e-6))


def _feasible_records_for_attempt(
    records: Sequence[Dict[str, Any]],
    *,
    min_visibility: float,
    shell_min: float,
    shell_max: float,
    require_clearance: bool,
) -> List[int]:
    out: List[int] = []
    for idx, rec in enumerate(records):
        if bool(rec.get("is_baseline_candidate", False)):
            continue
        view = rec.get("viewpoint_xyz")
        if view is None:
            continue
        visibility_ok = float(rec.get("visibility_score", 0.0)) >= float(min_visibility)
        shell_dist = float(rec.get("target_surface_distance_m", float("inf")))
        shell_ok = bool(float(shell_min) <= shell_dist <= float(shell_max))
        clearance_ok = bool(rec.get("clearance_ok", False)) or not bool(require_clearance)
        rec["attempt_visibility_ok"] = bool(visibility_ok)
        rec["attempt_shell_ok"] = bool(shell_ok)
        rec["attempt_clearance_ok"] = bool(clearance_ok)
        if visibility_ok and shell_ok and clearance_ok:
            out.append(int(idx))
    return out


def _connected_components(
    records: Sequence[Dict[str, Any]],
    feasible_indices: Sequence[int],
    *,
    angle_count: int,
    grid_step_m: float,
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    feasible = [int(i) for i in feasible_indices]
    feasible_set = set(feasible)
    adjacency: Dict[int, List[int]] = {idx: [] for idx in feasible}
    for pos, idx_a in enumerate(feasible):
        for idx_b in feasible[pos + 1:]:
            if _are_component_neighbors(records[idx_a], records[idx_b], angle_count=angle_count, grid_step_m=grid_step_m):
                adjacency[idx_a].append(idx_b)
                adjacency[idx_b].append(idx_a)
    seen = set()
    components: List[List[int]] = []
    for start in feasible:
        if start in seen:
            continue
        q: Deque[int] = deque([start])
        seen.add(start)
        comp: List[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt in feasible_set and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        components.append(comp)
    return components, adjacency


def _assign_medial_distances(
    records: List[Dict[str, Any]],
    component: Sequence[int],
    adjacency: Mapping[int, Sequence[int]],
    *,
    angle_count: int,
    max_radius_index: int,
) -> Dict[int, int]:
    comp = [int(i) for i in component]
    comp_set = set(comp)
    key_to_idx = {
        (int(records[idx].get("radius_index", -999)), int(records[idx].get("angle_index", -999))): int(idx)
        for idx in comp
    }
    boundary: List[int] = []
    for idx in comp:
        rec = records[idx]
        r_idx = int(rec.get("radius_index", -999))
        a_idx = int(rec.get("angle_index", -999))
        is_boundary = False
        for dr in (-1, 0, 1):
            for da in (-1, 0, 1):
                if dr == 0 and da == 0:
                    continue
                rr = r_idx + dr
                aa = (a_idx + da) % int(angle_count)
                if rr < 0 or rr > int(max_radius_index) or (rr, aa) not in key_to_idx:
                    is_boundary = True
                    break
            if is_boundary:
                break
        if is_boundary:
            boundary.append(idx)
    if not boundary:
        boundary = comp[:]

    dist = {idx: 10**9 for idx in comp}
    q: Deque[int] = deque()
    for idx in boundary:
        dist[idx] = 0
        q.append(idx)
    while q:
        cur = q.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt not in comp_set:
                continue
            if dist[nxt] > dist[cur] + 1:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    for idx in comp:
        value = int(dist[idx] if dist[idx] < 10**9 else 0)
        records[idx]["medial_distance_to_boundary"] = value
    return {idx: int(records[idx].get("medial_distance_to_boundary") or 0) for idx in comp}


def _component_summary(
    records: Sequence[Dict[str, Any]],
    component: Sequence[int],
    component_id: int,
) -> Dict[str, Any]:
    visibility = [float(records[idx].get("visibility_score", 0.0)) for idx in component]
    geos = [float(records[idx].get("path_geodesic_distance_m", float("inf"))) for idx in component]
    finite_geos = [x for x in geos if np.isfinite(x)]
    return {
        "component_id": int(component_id),
        "size": int(len(component)),
        "visibility_p75": float(np.percentile(visibility, 75)) if visibility else 0.0,
        "visibility_mean": float(np.mean(visibility)) if visibility else 0.0,
        "path_geodesic_mean_m": float(np.mean(finite_geos)) if finite_geos else float("inf"),
        "candidate_indices": [int(records[idx].get("candidate_index", -1)) for idx in component],
    }


def _select_component_medial_record(
    records: List[Dict[str, Any]],
    component: Sequence[int],
) -> int:
    ranked = sorted(
        [int(idx) for idx in component],
        key=lambda idx: (
            -int(records[idx].get("medial_distance_to_boundary") or 0),
            -float(records[idx].get("visibility_score", 0.0)),
            float(records[idx].get("path_geodesic_distance_m", float("inf"))),
            int(records[idx].get("accepted_rank", records[idx].get("candidate_index", 0))),
        ),
    )
    return int(ranked[0])


def _fallback_visibility_selection(
    records: List[Dict[str, Any]],
    *,
    cfg: VistaLsConfig,
) -> Tuple[Optional[int], str]:
    candidates = [idx for idx, rec in enumerate(records) if rec.get("viewpoint_xyz") is not None]
    if not candidates:
        return None, "no_vista_ls_or_fallback_candidate"
    baseline_candidates = [
        idx for idx in candidates
        if bool(records[idx].get("is_baseline_candidate", False))
        and float(records[idx].get("visibility_score", 0.0)) > max(0.0, float(cfg.min_visibility_score))
    ]
    if bool(cfg.prefer_visible_baseline) and baseline_candidates:
        return int(baseline_candidates[0]), "fallback_visible_baseline_guard"
    ranked = sorted(
        candidates,
        key=lambda idx: (
            -float(records[idx].get("visibility_score", 0.0)),
            float(records[idx].get("path_geodesic_distance_m", float("inf"))),
            int(records[idx].get("accepted_rank", records[idx].get("candidate_index", 0))),
        ),
    )
    best = int(ranked[0])
    if float(records[best].get("visibility_score", 0.0)) <= max(0.0, float(cfg.min_visibility_score)):
        return None, "no_positive_visibility_score"
    return best, "fallback_first_max_visibility"


def _select_vista_ls_record(
    records: List[Dict[str, Any]],
    *,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: VistaLsConfig,
    angle_count: int,
    max_radius_index: int,
) -> Tuple[Optional[int], Dict[str, Any]]:
    del path_finder, agent_position_xyz
    attempts = [
        {
            "stage": "strict",
            "min_visibility": float(cfg.min_visibility_score),
            "shell_min": float(cfg.shell_min_m),
            "shell_max": float(cfg.shell_max_m),
            "require_clearance": True,
        },
        {
            "stage": "relax_visibility",
            "min_visibility": 0.0,
            "shell_min": float(cfg.shell_min_m),
            "shell_max": float(cfg.shell_max_m),
            "require_clearance": True,
        },
        {
            "stage": "relax_shell",
            "min_visibility": 0.0,
            "shell_min": float(cfg.relaxed_shell_min_m),
            "shell_max": float(cfg.relaxed_shell_max_m),
            "require_clearance": True,
        },
        {
            "stage": "relax_clearance",
            "min_visibility": 0.0,
            "shell_min": float(cfg.relaxed_shell_min_m),
            "shell_max": float(cfg.relaxed_shell_max_m),
            "require_clearance": False,
        },
    ]
    attempt_logs: List[Dict[str, Any]] = []
    grid_step = max(float(cfg.radial_step_m), 1e-6)
    selected_idx: Optional[int] = None
    selected_summary: Dict[str, Any] = {}

    for attempt in attempts:
        feasible = _feasible_records_for_attempt(
            records,
            min_visibility=float(attempt["min_visibility"]),
            shell_min=float(attempt["shell_min"]),
            shell_max=float(attempt["shell_max"]),
            require_clearance=bool(attempt["require_clearance"]),
        )
        components, adjacency = _connected_components(records, feasible, angle_count=angle_count, grid_step_m=grid_step)
        min_size = max(1, int(cfg.min_component_size))
        large_components = [comp for comp in components if len(comp) >= min_size]
        selectable_components = large_components

        summaries = [_component_summary(records, comp, cid) for cid, comp in enumerate(components)]
        attempt_log = {
            "stage": attempt["stage"],
            "min_visibility": float(attempt["min_visibility"]),
            "shell_min": float(attempt["shell_min"]),
            "shell_max": float(attempt["shell_max"]),
            "require_clearance": bool(attempt["require_clearance"]),
            "feasible_candidate_count": int(len(feasible)),
            "component_count": int(len(components)),
            "selectable_component_count": int(len(selectable_components)),
            "min_component_size": int(min_size),
            "component_summaries": summaries[:20],
        }
        attempt_logs.append(attempt_log)
        if not selectable_components:
            continue

        max_size = max(len(comp) for comp in selectable_components)
        size_floor = max(1, int(math.ceil(max_size * max(0.0, min(1.0, float(cfg.size_tie_ratio))))))
        size_close_components = [comp for comp in selectable_components if len(comp) >= size_floor]
        component_with_id = [
            (components.index(comp), comp, _component_summary(records, comp, components.index(comp)))
            for comp in size_close_components
        ]
        component_with_id.sort(
            key=lambda item: (
                -float(item[2]["visibility_p75"]),
                float(item[2]["path_geodesic_mean_m"]),
                -int(item[2]["size"]),
                int(item[0]),
            )
        )
        selected_component_id, selected_component, component_stats = component_with_id[0]
        for component_id, component in enumerate(components):
            for idx in component:
                records[idx]["component_id"] = int(component_id)
                records[idx]["component_size"] = int(len(component))
        _assign_medial_distances(
            records,
            selected_component,
            adjacency,
            angle_count=angle_count,
            max_radius_index=max_radius_index,
        )
        selected_idx = _select_component_medial_record(records, selected_component)
        records[selected_idx]["selected_by_vista_ls"] = True
        selected_summary = {
            "selection_policy": "vista_ls_level_set_medial_center",
            "vista_ls_enabled": True,
            "vista_ls_component_count": int(len(components)),
            "vista_ls_selected_component_id": int(selected_component_id),
            "vista_ls_selected_component_size": int(len(selected_component)),
            "vista_ls_selected_medial_radius": int(records[selected_idx].get("medial_distance_to_boundary") or 0),
            "vista_ls_fallback_used": bool(str(attempt["stage"]) != "strict"),
            "vista_ls_fallback_stage": str(attempt["stage"]),
            "vista_ls_rejected_reason": None,
            "vista_ls_attempts": attempt_logs,
            "vista_ls_selected_component_stats": component_stats,
        }
        return selected_idx, selected_summary

    fallback_idx, reason = _fallback_visibility_selection(records, cfg=cfg)
    if fallback_idx is not None:
        records[fallback_idx]["selected_by_vista_ls"] = False
    return fallback_idx, {
        "selection_policy": "vista_ls_fallback_first_max_visibility",
        "vista_ls_enabled": True,
        "vista_ls_component_count": 0,
        "vista_ls_selected_component_id": None,
        "vista_ls_selected_component_size": 0,
        "vista_ls_selected_medial_radius": None,
        "vista_ls_fallback_used": True,
        "vista_ls_fallback_stage": reason,
        "vista_ls_rejected_reason": None if fallback_idx is not None else reason,
        "vista_ls_attempts": attempt_logs,
    }


def visibility_based_viewpoint_decision(
    *,
    rep: Any,
    selected_slot_index: int,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: VistaLsConfig,
    baseline_target_xyz: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(cfg.rng_seed) + int(selected_slot_index))
    try:
        object_points, blocker_points, scene_points = _object_points_habitat(rep=rep, slot_index=int(selected_slot_index))
    except VistaLsRejectedError as exc:
        return {
            "called": True,
            "selected_slot_index": int(selected_slot_index),
            "object_point_count": 0,
            "scene_point_count": None,
            "scene_points_include_target": None,
            "object_center_xyz": None,
            "camera_height_m": float(cfg.camera_height_m),
            "candidate_records": [],
            "viewpoint_applied": False,
            "rejected_reason": str(exc),
            "best_viewpoint_xyz": None,
            "best_visibility_score": None,
            "visibility_records": [],
            "min_visibility_score": float(cfg.min_visibility_score),
            "selection_policy": "vista_ls_level_set_medial_center",
            "vista_ls_enabled": True,
            "vista_ls_rejected_reason": str(exc),
        }

    sampled_object_points = _sample_rows(object_points, int(cfg.target_sample_count), rng)
    if sampled_object_points.shape[0] == 0:
        return {
            "called": True,
            "selected_slot_index": int(selected_slot_index),
            "object_point_count": int(object_points.shape[0]),
            "scene_point_count": int(scene_points.shape[0]),
            "scene_points_include_target": False,
            "object_center_xyz": None,
            "camera_height_m": float(cfg.camera_height_m),
            "candidate_records": [],
            "viewpoint_applied": False,
            "rejected_reason": "empty_target_point_sample",
            "best_viewpoint_xyz": None,
            "best_visibility_score": None,
            "visibility_records": [],
            "min_visibility_score": float(cfg.min_visibility_score),
            "selection_policy": "vista_ls_level_set_medial_center",
            "vista_ls_enabled": True,
            "vista_ls_rejected_reason": "empty_target_point_sample",
        }
    center = sampled_object_points.mean(axis=0)
    candidates, candidate_records, angle_count = _generate_level_set_candidates(
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
        "camera_height_m": float(cfg.camera_height_m),
        "target_sample_count": int(cfg.target_sample_count),
        "candidate_records": candidate_records,
        "candidate_count": int(len(candidates)),
        "vista_ls_angle_count": int(angle_count),
        "vista_ls_r_min_m": float(cfg.r_min_m),
        "vista_ls_r_max_m": float(cfg.r_max_m),
        "vista_ls_radial_step_m": float(cfg.radial_step_m),
        "vista_ls_angle_step_deg": float(cfg.angle_step_deg),
        "vista_ls_shell_min_m": float(cfg.shell_min_m),
        "vista_ls_shell_max_m": float(cfg.shell_max_m),
        "vista_ls_min_clearance_m": float(cfg.min_clearance_m),
        "vista_ls_min_component_size": int(cfg.min_component_size),
        "vista_ls_size_tie_ratio": float(cfg.size_tie_ratio),
    }
    if len(candidates) == 0:
        info.update(
            {
                "viewpoint_applied": False,
                "rejected_reason": "no_reachable_level_set_candidate",
                "best_viewpoint_xyz": None,
                "best_visibility_score": None,
                "visibility_records": [],
                "selection_policy": "vista_ls_level_set_medial_center",
                "vista_ls_enabled": True,
                "vista_ls_component_count": 0,
                "vista_ls_fallback_used": False,
                "vista_ls_rejected_reason": "no_reachable_level_set_candidate",
            }
        )
        return info

    baseline_candidate_record: Optional[Dict[str, Any]] = None
    if baseline_target_xyz is not None:
        agent = _as_np3(agent_position_xyz, name="agent_position_xyz")
        agent_island = _safe_get_island(path_finder, agent)
        baseline_raw = _as_np3(baseline_target_xyz, name="baseline_target_xyz")
        baseline_nav = _snap_point(path_finder, baseline_raw, agent_island)
        baseline_nav_ok = _safe_is_navigable(path_finder, baseline_nav)
        baseline_same_island = bool(baseline_nav_ok and _safe_get_island(path_finder, baseline_nav) == agent_island)
        baseline_candidate_record = {
            "candidate_index": -1,
            "radius_index": None,
            "angle_index": None,
            "raw_xyz": baseline_raw.tolist(),
            "snapped_xyz": baseline_nav.tolist() if np.all(np.isfinite(baseline_nav)) else None,
            "raw_navigable": _safe_is_navigable(path_finder, baseline_raw),
            "raw_same_island": False,
            "snap_distance_m": float(np.linalg.norm(baseline_nav - baseline_raw)) if np.all(np.isfinite(baseline_nav)) else float("inf"),
            "snapped_navigable": bool(baseline_nav_ok),
            "snapped_same_island": bool(baseline_same_island),
            "accepted": bool(baseline_same_island),
            "reachable_ok": bool(baseline_same_island),
            "accepted_stage": "baseline_snapped_navmesh" if baseline_same_island else None,
            "is_baseline_candidate": True,
        }
        info["baseline_candidate_record"] = baseline_candidate_record
        if baseline_same_island:
            candidate_records.append(baseline_candidate_record)
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
            rec["baseline_raw_target_xyz"] = (
                _as_np3(baseline_target_xyz, name="baseline_target_xyz").tolist()
                if baseline_target_xyz is not None
                else None
            )
        else:
            rec["is_baseline_candidate"] = False
    visibility_records = _augment_records_for_level_set(
        visibility_records=visibility_records,
        candidate_records=candidate_records,
        target_points=sampled_object_points,
        blocker_points=blocker_points,
        path_finder=path_finder,
        agent_position_xyz=agent_position_xyz,
        cfg=cfg,
    )
    max_radius_index = max([int(rec.get("radius_index", 0)) for rec in candidate_records if rec.get("radius_index") is not None] or [0])
    selected_idx, ls_info = _select_vista_ls_record(
        visibility_records,
        path_finder=path_finder,
        agent_position_xyz=agent_position_xyz,
        cfg=cfg,
        angle_count=angle_count,
        max_radius_index=max_radius_index,
    )
    info.update(ls_info)
    if selected_idx is None:
        info.update(
            {
                "viewpoint_applied": False,
                "rejected_reason": str(ls_info.get("vista_ls_rejected_reason") or "no_vista_ls_feasible_component"),
                "best_viewpoint_xyz": None,
                "best_visibility_score": None,
                "best_candidate_index": None,
                "best_candidate_agent_l2_distance_m": None,
                "visibility_records": visibility_records,
                "min_visibility_score": float(cfg.min_visibility_score),
            }
        )
        return info

    best = visibility_records[int(selected_idx)]
    score = float(best.get("visibility_score", 0.0))
    selected_positive_visibility = bool(score > 0.0)
    applied = bool(selected_positive_visibility)
    if bool(best.get("is_baseline_candidate", False)):
        applied = bool(selected_positive_visibility)
    info.update(
        {
            "viewpoint_applied": bool(applied),
            "rejected_reason": None if applied else "no_positive_visibility_score",
            "best_viewpoint_xyz": best["viewpoint_xyz"] if applied else None,
            "best_visibility_score": float(score),
            "best_candidate_index": int(best.get("candidate_index", -1)),
            "best_candidate_agent_l2_distance_m": float(best.get("agent_l2_distance_m", float("inf"))),
            "best_path_geodesic_distance_m": float(best.get("path_geodesic_distance_m", float("inf"))),
            "best_target_surface_distance_m": float(best.get("target_surface_distance_m", float("inf"))),
            "best_clearance_m": float(best.get("clearance_m", float("inf"))),
            "best_positive_visibility": bool(selected_positive_visibility),
            "visibility_records": visibility_records,
            "min_visibility_score": float(cfg.min_visibility_score),
            "visibility_tie_epsilon": float(cfg.visibility_tie_epsilon),
        }
    )
    return info


def correct_final_decision_with_vistals(
    *,
    rep: Any,
    decision_aux: Mapping[str, Any],
    baseline_target_xyz: np.ndarray,
    output_dir: Path,
    path_finder: Any,
    agent_position_xyz: Sequence[float],
    cfg: VistaLsConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    missing_keys = [k for k in ("is_object_decision", "real_object_decision_idx") if k not in decision_aux]
    if missing_keys:
        raise VistaLsInputError(f"decision_aux missing required key(s): {missing_keys}")
    if not bool(decision_aux["is_object_decision"]):
        raise VistaLsInputError("vista-ls was called for a non-object final decision")
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
    target_source = "vista_ls_level_set_target"
    if not bool(cfg.enable_vvd_replacement):
        failure_info = {
            "ok": True,
            "module": "vista-ls",
            "applied": False,
            "reason": "vistals_target_adjustment_disabled",
            "target_before": baseline_target.tolist(),
            "target_after": baseline_target.tolist(),
            "vistals_called": True,
            "module_scope": "visibility_informed_level_set_target_adjustment",
            "decision_policy": "VISTA-LS diagnostic only; replacement disabled",
            "uses_language_or_vision_model": False,
            "baseline_memory_index": int(baseline_memory_index),
            "baseline_target_xyz": baseline_target.tolist(),
            "selected_slot_index": int(baseline_memory_index),
            "visibility": visibility,
            "target_source": "vistals_diagnostic_explicit_keep_baseline",
            "viewpoint_correction_applied": False,
            "correction_applied": False,
            "corrected_target_xyz": baseline_target.tolist(),
            "rejected_reason": "vistals_enable_vvd_replacement_false",
        }
        with open(output_dir / "vistals_decision.json", "w", encoding="utf-8") as f:
            json.dump(failure_info, f, ensure_ascii=False, indent=2)
        return baseline_target.copy(), failure_info

    if not bool(visibility["viewpoint_applied"]):
        failure_info = {
            "ok": True,
            "module": "vista-ls",
            "applied": False,
            "reason": str(visibility.get("rejected_reason")),
            "target_before": baseline_target.tolist(),
            "target_after": baseline_target.tolist(),
            "vistals_called": True,
            "module_scope": "visibility_informed_level_set_target_adjustment",
            "decision_policy": "VISTA-LS level-set target adjustment on baseline final object slot",
            "uses_language_or_vision_model": False,
            "baseline_memory_index": int(baseline_memory_index),
            "baseline_target_xyz": baseline_target.tolist(),
            "selected_slot_index": int(baseline_memory_index),
            "visibility": visibility,
            "target_source": "vistals_no_feasible_component_explicit_keep_baseline",
            "viewpoint_correction_applied": False,
            "correction_applied": False,
            "corrected_target_xyz": baseline_target.tolist(),
            "rejected_reason": visibility.get("rejected_reason"),
        }
        with open(output_dir / "vistals_decision.json", "w", encoding="utf-8") as f:
            json.dump(failure_info, f, ensure_ascii=False, indent=2)
        return baseline_target.copy(), failure_info

    corrected_target = _as_np3(visibility["best_viewpoint_xyz"], name="best_viewpoint_xyz")
    if int(visibility.get("best_candidate_index", 0)) == -1:
        corrected_target = baseline_target.copy()
        target_source = "baseline_visibility_candidate_explicit_keep_baseline"
    elif str(visibility.get("selection_policy", "")).startswith("vista_ls_fallback"):
        target_source = "vista_ls_fallback_target"

    correction_applied = bool(np.linalg.norm(corrected_target - baseline_target) > 1e-6)
    info = {
        "ok": True,
        "module": "vista-ls",
        "applied": bool(correction_applied),
        "reason": "selected_vista_ls_level_set_target" if correction_applied else "selected_baseline_equivalent_target",
        "target_before": baseline_target.tolist(),
        "target_after": corrected_target.tolist(),
        "vistals_called": True,
        "module_scope": "visibility_informed_level_set_target_adjustment",
        "decision_policy": "VISTA-LS level-set medial-center target adjustment on baseline object slot",
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
    with open(output_dir / "vistals_decision.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return corrected_target, info


__all__ = [
    "VistaLsConfig",
    "VistaLsDecisionError",
    "VistaLsInputError",
    "VistaLsRejectedError",
    "correct_final_decision_with_vistals",
    "visibility_based_viewpoint_decision",
]
