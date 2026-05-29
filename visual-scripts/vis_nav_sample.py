"""vis_nav_sample.py - Visualize a single navigation episode (Vista2MQSC refine1).

Replays the full navigation loop for one specified episode and saves visual
outputs at every decision step.

Output layout under --output_vis_dir:
  scene=<S>/navigation_type=<TYPE>/episode=<E>/task=<T>/
    topdown_maps/   dec_NNN_topdown.png    occupancy + fog-of-war + agent + target
    topdown_rgb/    dec_NNN_rgb_floor.png  colorized RGB floor plan (same overlays)
    frontiers/      dec_NNN_frontiers.png  frontier candidates on RGB floor
    decisions/dec_NNN/
      panorama_12_frames/              12 scan RGB/depth frames
      panorama_rgb_contact_sheet.png   compact panorama overview
      topdown_fog.png                  fog-of-war top-down map
      topdown_full.png                 fully revealed top-down map
      frontiers_on_topdown.png         current frontiers and selected frontier
      birdseye_rgb.png                 extra top-down RGB camera render
      decision_target_facing_rgb.png   current camera pose turned toward target
      frontier_facing_rgb/             temporary views facing frontier candidates
      decision.json                    positions, frontiers, PQ3D aux, follow info
    trajectory/     task_trajectory.png    full path: start, stops, goal objects
    summary.json    per-task metrics

Usage (run from repo root with mtu3d env active):
  python visual-scripts/vis_nav_sample.py \\
      --scene_name 00802-wcojb4TFT35 \\
      --episode_id 12 \\
      [--task_ids 0,2] \\
      [--output_vis_dir visual-output/vis_sample] \\
      [--hm3d_data_base_path datascene] \\
      [--navigation_data_path LangMap_Annotations]

Instance-level example:
  bash visual-scripts/run_vis_instance_armchair_906.sh
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _debug_import(msg: str) -> None:
    if os.environ.get("VIS_NAV_DEBUG_IMPORTS", "0") == "1":
        print(f"[vis-import] {msg}", file=sys.stderr, flush=True)


_debug_import("import cv2")
import cv2
_debug_import("import habitat_sim")
import habitat_sim
_debug_import("import matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
_debug_import("import numpy")
import numpy as np
_debug_import("import habitat maps")
from habitat.utils.visualizations import maps
_debug_import("import omegaconf")
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"
if str(HM3D_ONLINE) not in sys.path:
    sys.path.insert(0, str(HM3D_ONLINE))

_debug_import("import HabitatSimulator")
from common.embodied_utils.simulator import HabitatSimulator
_debug_import("import frontier_utils")
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)
_debug_import("imports complete")

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for trajectory overlays (RGB uint8)
# ──────────────────────────────────────────────────────────────────────────────
CLR_START      = (0,   200,   0)    # green  – episode start
CLR_END        = (200,   0,   0)    # red    – agent final position
CLR_GOAL       = (255, 165,   0)    # orange – ground-truth goal object
CLR_DECISION   = (0,   120, 255)    # blue   – non-final decision targets
CLR_FINAL_DEC  = (220,   0, 220)    # magenta– final (vista-stop) decision
CLR_FRONTIER   = (0,   200, 200)    # cyan   – frontier waypoints
CLR_AGENT      = (255, 255,   0)    # yellow – current agent position
CLR_PATH       = (200, 200,   0)    # yellow-ish – trajectory path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
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


def _pos_to_pixel(position: np.ndarray, top_down_map: np.ndarray, sim: Any) -> Tuple[int, int]:
    """World XYZ → (row, col) pixel in top_down_map."""
    px = map_coors_to_pixel(position, top_down_map, sim)
    return int(px[0]), int(px[1])


def _draw_circle(img: np.ndarray, rc: Tuple[int, int], color: Tuple, radius: int = 6) -> None:
    cv2.circle(img, (rc[1], rc[0]), radius, color, -1)


def _draw_star(img: np.ndarray, rc: Tuple[int, int], color: Tuple, size: int = 10) -> None:
    pts = np.array([
        [rc[1], rc[0] - size],
        [rc[1] + int(size * 0.35), rc[0] + size],
        [rc[1] - int(size * 0.9), rc[0] - int(size * 0.3)],
        [rc[1] + int(size * 0.9), rc[0] - int(size * 0.3)],
        [rc[1] - int(size * 0.35), rc[0] + size],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)


def _draw_agent_arrow(img: np.ndarray, rc: Tuple[int, int], angle_rad: float, color: Tuple, size: int = 10) -> None:
    r, c = rc
    tip_r = r - int(size * math.cos(angle_rad))
    tip_c = c + int(size * math.sin(angle_rad))
    cv2.arrowedLine(img, (c, r), (tip_c, tip_r), color, 2, tipLength=0.4)


def _base_rgb_floor(top_down_map: np.ndarray, fog: np.ndarray) -> np.ndarray:
    """Return a fresh colorized RGB floor map (uint8, H×W×3)."""
    return maps.colorize_topdown_map(top_down_map, fog).copy()


def _clamp_rc(rc: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    r = max(0, min(int(rc[0]), shape[0] - 1))
    c = max(0, min(int(rc[1]), shape[1] - 1))
    return r, c


def _save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    dep = np.asarray(depth, dtype=np.float32)
    dep = np.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
    valid = dep[dep > 0]
    if valid.size == 0:
        norm = np.zeros_like(dep, dtype=np.uint8)
    else:
        hi = float(np.percentile(valid, 95))
        hi = max(hi, 1e-6)
        norm = np.clip(dep / hi, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _contact_sheet(images: List[np.ndarray], cols: int = 4, thumb_w: int = 240) -> Optional[np.ndarray]:
    if not images:
        return None
    thumbs = []
    for idx, img in enumerate(images):
        rgb = np.asarray(img[:, :, :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        thumb_h = max(1, int(round(h * float(thumb_w) / max(float(w), 1.0))))
        thumb = cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{idx:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(thumb, f"{idx:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    rows = int(math.ceil(len(thumbs) / float(cols)))
    th, tw = thumbs[0].shape[:2]
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb
    return sheet


def _sample_indices(n: int, max_items: int) -> List[int]:
    if n <= 0 or max_items <= 0:
        return []
    if n <= max_items:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, max_items)})


def _copy_agent_state(src: Any) -> habitat_sim.AgentState:
    dst = habitat_sim.AgentState()
    dst.position = np.asarray(src.position, dtype=float).reshape(3)
    dst.rotation = src.rotation
    return dst


def _rotation_xyzw(rotation: Any) -> Any:
    try:
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    except Exception:
        try:
            vals = list(rotation)
            return [float(v) for v in vals]
        except Exception:
            return str(rotation)


def _look_at_quat_xz(origin: np.ndarray, target: np.ndarray) -> Any:
    origin = np.asarray(origin, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    dx = float(target[0] - origin[0])
    dz = float(target[2] - origin[2])
    if math.hypot(dx, dz) < 1e-6:
        yaw = 0.0
    else:
        # Habitat's forward camera looks along local -Z. This yaw rotates -Z
        # toward the target in the horizontal XZ plane.
        yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _render_rgb_facing_target(
    *,
    sim: Any,
    agent: Any,
    base_state: Any,
    target: np.ndarray,
) -> np.ndarray:
    saved_state = _copy_agent_state(agent.get_state())
    face_state = habitat_sim.AgentState()
    face_state.position = np.asarray(base_state.position, dtype=float).reshape(3)
    face_state.rotation = _look_at_quat_xz(face_state.position, np.asarray(target, dtype=float).reshape(3))
    try:
        agent.set_state(face_state)
        obs = sim.get_sensor_observations()
        return np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy()
    finally:
        agent.set_state(saved_state)


def _build_birdseye_renderer(
    *,
    scene_path: str,
    resolution: int,
    sensor_height_m: float,
    hfov_deg: float,
) -> Tuple[Any, Any]:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = False

    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "birdseye_rgb"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    cam.resolution = [int(resolution), int(resolution)]
    cam.position = [0.0, float(sensor_height_m), 0.0]
    cam.orientation = [-math.pi / 2.0, 0.0, 0.0]
    cam.hfov = float(hfov_deg)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]
    renderer = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    renderer_agent = renderer.initialize_agent(0)
    return renderer, renderer_agent


def _save_birdseye_rgb(
    *,
    renderer: Optional[Any],
    renderer_agent: Optional[Any],
    agent_state: Any,
    out_path: Path,
) -> None:
    if renderer is None or renderer_agent is None:
        return
    state = habitat_sim.AgentState()
    state.position = np.asarray(agent_state.position, dtype=float).reshape(3)
    state.rotation = agent_state.rotation
    renderer_agent.set_state(state)
    obs = renderer.get_sensor_observations()
    img = np.asarray(obs["birdseye_rgb"][:, :, :3], dtype=np.uint8).copy()
    _save_png(img, out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Per-decision visualization save functions
# ──────────────────────────────────────────────────────────────────────────────

def _overlay_common(
    img: np.ndarray,
    top_down_map: np.ndarray,
    sim: Any,
    agent_state: Any,
    target: np.ndarray,
    is_final: bool,
    path_pixels: List[Tuple[int, int]],
) -> np.ndarray:
    """Draw agent position, movement path and current target onto img (in-place copy)."""
    img = img.copy()
    shape = img.shape[:2]

    # Draw travelled path
    for prev, nxt in zip(path_pixels[:-1], path_pixels[1:]):
        cv2.line(img, (prev[1], prev[0]), (nxt[1], nxt[0]), CLR_PATH, 1)

    # Current agent
    agent_rc = _clamp_rc(_pos_to_pixel(agent_state.position, top_down_map, sim), shape)
    angle = float(get_polar_angle(agent_state))
    _draw_agent_arrow(img, agent_rc, angle, CLR_AGENT, size=10)

    # Current target
    target_rc = _clamp_rc(_pos_to_pixel(target, top_down_map, sim), shape)
    clr = CLR_FINAL_DEC if is_final else CLR_DECISION
    cv2.drawMarker(img, (target_rc[1], target_rc[0]), clr,
                   cv2.MARKER_CROSS, 14, 2)

    return img


def save_topdown_map(
    *,
    out_dir: Path,
    dec_num: int,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    agent_state: Any,
    target: np.ndarray,
    is_final: bool,
    path_pixels: List[Tuple[int, int]],
    sim: Any,
) -> None:
    """Occupancy map (grayscale with fog) + overlays."""
    rgb = _base_rgb_floor(top_down_map, fog)
    rgb = _overlay_common(rgb, top_down_map, sim, agent_state, target, is_final, path_pixels)
    _save_png(rgb, out_dir / "topdown_maps" / f"dec_{dec_num:03d}_topdown.png")


def save_rgb_floor(
    *,
    out_dir: Path,
    dec_num: int,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    agent_state: Any,
    target: np.ndarray,
    is_final: bool,
    path_pixels: List[Tuple[int, int]],
    sim: Any,
) -> None:
    """Fully-revealed RGB floor map (no fog) + overlays."""
    full_fog = np.ones_like(fog)  # reveal everything
    rgb = _base_rgb_floor(top_down_map, full_fog)
    rgb = _overlay_common(rgb, top_down_map, sim, agent_state, target, is_final, path_pixels)
    _save_png(rgb, out_dir / "topdown_rgb" / f"dec_{dec_num:03d}_rgb_floor.png")


def save_frontiers(
    *,
    out_dir: Path,
    dec_num: int,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    agent_state: Any,
    target: np.ndarray,
    is_final: bool,
    path_pixels: List[Tuple[int, int]],
    frontier_waypoints: List[np.ndarray],
    sim: Any,
) -> None:
    """Frontier candidates overlaid on the fog-of-war RGB map."""
    rgb = _base_rgb_floor(top_down_map, fog)
    rgb = _overlay_common(rgb, top_down_map, sim, agent_state, target, is_final, path_pixels)
    shape = rgb.shape[:2]

    # Draw each frontier
    for fw in frontier_waypoints:
        fw_arr = np.asarray(fw, dtype=float).reshape(3)
        fw_rc = _clamp_rc(_pos_to_pixel(fw_arr, top_down_map, sim), shape)
        _draw_circle(rgb, fw_rc, CLR_FRONTIER, radius=4)

    # Legend (top-left block)
    legend_items = [
        (CLR_AGENT,    "agent"),
        (CLR_DECISION, "frontier target" if not is_final else "nav target"),
        (CLR_FINAL_DEC,"final decision"),
        (CLR_FRONTIER, "frontier candidates"),
    ]
    y = 8
    for clr, label in legend_items:
        cv2.rectangle(rgb, (4, y), (16, y + 10), clr, -1)
        cv2.putText(rgb, label, (20, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
        y += 16

    _save_png(rgb, out_dir / "frontiers" / f"dec_{dec_num:03d}_frontiers.png")


def _nearest_frontier_index(target: np.ndarray, frontier_waypoints: List[np.ndarray], max_xz_dist: float = 0.75) -> Optional[int]:
    if not frontier_waypoints:
        return None
    target = np.asarray(target, dtype=float).reshape(3)
    distances = [
        float(np.linalg.norm(np.asarray(fw, dtype=float).reshape(3)[[0, 2]] - target[[0, 2]]))
        for fw in frontier_waypoints
    ]
    idx = int(np.argmin(distances))
    return idx if distances[idx] <= float(max_xz_dist) else None


def _draw_goal_positions(
    img: np.ndarray,
    *,
    top_down_map: np.ndarray,
    sim: Any,
    goal_positions: List[np.ndarray],
) -> None:
    shape = img.shape[:2]
    for gp in goal_positions:
        gp_rc = _clamp_rc(_pos_to_pixel(np.asarray(gp, dtype=float).reshape(3), top_down_map, sim), shape)
        _draw_star(img, gp_rc, CLR_GOAL, size=9)


def _draw_frontier_positions(
    img: np.ndarray,
    *,
    top_down_map: np.ndarray,
    sim: Any,
    frontier_waypoints: List[np.ndarray],
    selected_frontier_idx: Optional[int],
) -> None:
    shape = img.shape[:2]
    for idx, fw in enumerate(frontier_waypoints):
        fw_rc = _clamp_rc(_pos_to_pixel(np.asarray(fw, dtype=float).reshape(3), top_down_map, sim), shape)
        radius = 6 if selected_frontier_idx == idx else 4
        _draw_circle(img, fw_rc, CLR_FRONTIER, radius=radius)
        if selected_frontier_idx == idx:
            cv2.drawMarker(img, (fw_rc[1], fw_rc[0]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)


def save_decision_maps(
    *,
    dec_dir: Path,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    agent_state: Any,
    target: np.ndarray,
    is_final: bool,
    path_pixels: List[Tuple[int, int]],
    frontier_waypoints: List[np.ndarray],
    selected_frontier_idx: Optional[int],
    goal_positions: List[np.ndarray],
    sim: Any,
) -> None:
    dec_dir.mkdir(parents=True, exist_ok=True)
    for name, fog_mask in [("topdown_fog.png", fog), ("topdown_full.png", np.ones_like(fog))]:
        rgb = _base_rgb_floor(top_down_map, fog_mask)
        rgb = _overlay_common(rgb, top_down_map, sim, agent_state, target, is_final, path_pixels)
        _draw_goal_positions(rgb, top_down_map=top_down_map, sim=sim, goal_positions=goal_positions)
        _save_png(rgb, dec_dir / name)

    rgb = _base_rgb_floor(top_down_map, fog)
    rgb = _overlay_common(rgb, top_down_map, sim, agent_state, target, is_final, path_pixels)
    _draw_goal_positions(rgb, top_down_map=top_down_map, sim=sim, goal_positions=goal_positions)
    _draw_frontier_positions(
        rgb,
        top_down_map=top_down_map,
        sim=sim,
        frontier_waypoints=frontier_waypoints,
        selected_frontier_idx=selected_frontier_idx,
    )
    _save_png(rgb, dec_dir / "frontiers_on_topdown.png")


def save_panorama_frames(dec_dir: Path, scan_rgb: List[np.ndarray], scan_depth: List[np.ndarray]) -> None:
    pano_dir = dec_dir / "panorama_12_frames"
    pano_dir.mkdir(parents=True, exist_ok=True)
    rgb_for_sheet: List[np.ndarray] = []
    depth_for_sheet: List[np.ndarray] = []
    for idx, rgb in enumerate(scan_rgb):
        rgb_img = np.asarray(rgb[:, :, :3], dtype=np.uint8)
        rgb_for_sheet.append(rgb_img)
        _save_png(rgb_img, pano_dir / f"frame_{idx:02d}_rgb.png")
    for idx, dep in enumerate(scan_depth):
        dep_rgb = _depth_to_rgb(dep)
        depth_for_sheet.append(dep_rgb)
        _save_png(dep_rgb, pano_dir / f"frame_{idx:02d}_depth.png")
    rgb_sheet = _contact_sheet(rgb_for_sheet, cols=4, thumb_w=240)
    if rgb_sheet is not None:
        _save_png(rgb_sheet, dec_dir / "panorama_rgb_contact_sheet.png")
    depth_sheet = _contact_sheet(depth_for_sheet, cols=4, thumb_w=240)
    if depth_sheet is not None:
        _save_png(depth_sheet, dec_dir / "panorama_depth_contact_sheet.png")


def save_frontier_facing_views(
    *,
    dec_dir: Path,
    sim: Any,
    agent: Any,
    agent_state: Any,
    target: np.ndarray,
    frontier_waypoints: List[np.ndarray],
    selected_frontier_idx: Optional[int],
    max_frontiers: int,
) -> List[Dict[str, Any]]:
    out_dir = dec_dir / "frontier_facing_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Dict[str, Any]] = []
    order: List[int] = []
    if selected_frontier_idx is not None:
        order.append(int(selected_frontier_idx))
    for idx in range(len(frontier_waypoints)):
        if idx not in order:
            order.append(idx)
    if max_frontiers > 0:
        order = order[: int(max_frontiers)]

    for rank, idx in enumerate(order):
        fw = np.asarray(frontier_waypoints[idx], dtype=float).reshape(3)
        img = _render_rgb_facing_target(sim=sim, agent=agent, base_state=agent_state, target=fw)
        name = f"frontier_{idx:02d}_facing_rgb.png"
        if selected_frontier_idx == idx:
            name = f"selected_frontier_{idx:02d}_facing_rgb.png"
            _save_png(img, dec_dir / "selected_frontier_facing_rgb.png")
        _save_png(img, out_dir / name)
        saved.append(
            {
                "rank": int(rank),
                "frontier_index": int(idx),
                "is_selected": bool(selected_frontier_idx == idx),
                "position": fw.tolist(),
                "image": str((out_dir / name).relative_to(dec_dir)),
            }
        )

    target_img = _render_rgb_facing_target(sim=sim, agent=agent, base_state=agent_state, target=target)
    _save_png(target_img, dec_dir / "decision_target_facing_rgb.png")
    return saved


def save_follow_frames(dec_dir: Path, goto_rgb: List[np.ndarray], max_saved: int) -> List[int]:
    indices = _sample_indices(len(goto_rgb), int(max_saved))
    if not indices:
        return []
    out_dir = dec_dir / "follow_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_imgs = []
    for out_idx, frame_idx in enumerate(indices):
        img = np.asarray(goto_rgb[frame_idx][:, :, :3], dtype=np.uint8)
        sampled_imgs.append(img)
        _save_png(img, out_dir / f"follow_{out_idx:02d}_src_{frame_idx:04d}.png")
    sheet = _contact_sheet(sampled_imgs, cols=4, thumb_w=240)
    if sheet is not None:
        _save_png(sheet, dec_dir / "follow_rgb_contact_sheet.png")
    return indices


class _DropTaskLevelMaskGenerator:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("task_level", None)
        return self.inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# ──────────────────────────────────────────────────────────────────────────────
# End-of-task trajectory figure
# ──────────────────────────────────────────────────────────────────────────────

def save_trajectory(
    *,
    out_dir: Path,
    task_id: int,
    top_down_map: np.ndarray,
    fog_final: np.ndarray,
    sim: Any,
    start_position: np.ndarray,
    end_position: np.ndarray,
    goal_positions: List[np.ndarray],
    path_pixels: List[Tuple[int, int]],
    decision_pixels: List[Tuple[int, int]],   # non-final decisions
    final_dec_pixels: List[Tuple[int, int]],  # final (vista-stop) decisions
    sentence: str,
    sr: float,
    spl: float,
    task_level: str,
) -> None:
    full_fog = np.ones_like(fog_final)
    rgb = _base_rgb_floor(top_down_map, full_fog)
    shape = rgb.shape[:2]

    # Draw path
    for prev, nxt in zip(path_pixels[:-1], path_pixels[1:]):
        cv2.line(rgb, (prev[1], prev[0]), (nxt[1], nxt[0]), CLR_PATH, 2)

    # Non-final decision positions (small blue circles)
    for rc in decision_pixels:
        rc = _clamp_rc(rc, shape)
        _draw_circle(rgb, rc, CLR_DECISION, radius=5)

    # Final (vista-stop) decision positions (magenta cross+circle)
    for rc in final_dec_pixels:
        rc = _clamp_rc(rc, shape)
        _draw_circle(rgb, rc, CLR_FINAL_DEC, radius=7)
        cv2.drawMarker(rgb, (rc[1], rc[0]), CLR_FINAL_DEC, cv2.MARKER_CROSS, 16, 2)

    # Goal objects (orange star)
    for gp in goal_positions:
        gp_arr = np.asarray(gp, dtype=float).reshape(3)
        gp_rc = _clamp_rc(_pos_to_pixel(gp_arr, top_down_map, sim), shape)
        _draw_star(rgb, gp_rc, CLR_GOAL, size=10)

    # Start position (green circle)
    st_arr = np.asarray(start_position, dtype=float).reshape(3)
    st_rc = _clamp_rc(_pos_to_pixel(st_arr, top_down_map, sim), shape)
    _draw_circle(rgb, st_rc, CLR_START, radius=8)
    cv2.putText(rgb, "S", (st_rc[1] - 5, st_rc[0] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # End position (red circle)
    en_arr = np.asarray(end_position, dtype=float).reshape(3)
    en_rc = _clamp_rc(_pos_to_pixel(en_arr, top_down_map, sim), shape)
    _draw_circle(rgb, en_rc, CLR_END, radius=8)
    cv2.putText(rgb, "E", (en_rc[1] - 5, en_rc[0] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Matplotlib figure with legend and title
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.axis("off")

    title = (
        f"Task {task_id} [{task_level}]  SR={sr:.1f}  SPL={spl:.4f}\n"
        f"{sentence[:100]}{'...' if len(sentence) > 100 else ''}"
    )
    ax.set_title(title, fontsize=9, wrap=True)

    legend_patches = [
        mpatches.Patch(color=np.array(CLR_START) / 255,    label="Start"),
        mpatches.Patch(color=np.array(CLR_END) / 255,      label="End (agent final)"),
        mpatches.Patch(color=np.array(CLR_GOAL) / 255,     label="Goal object"),
        mpatches.Patch(color=np.array(CLR_DECISION) / 255, label="Decision (frontier)"),
        mpatches.Patch(color=np.array(CLR_FINAL_DEC) / 255,label="Vista stop (final)"),
        mpatches.Patch(color=np.array(CLR_PATH) / 255,     label="Agent path"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              framealpha=0.8, ncol=2)

    out_path = out_dir / "trajectory" / f"task_{task_id:02d}_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] trajectory -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Nav helpers (copied verbatim from main script)
# ──────────────────────────────────────────────────────────────────────────────

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
        ri = region_to_annot_dict[cur_task["region_id"]]
        desc = (
            ri.get("shortest_description") or ri.get("concise_description") or ri.get("detailed_description") or ""
        ) if concise_description else (
            ri.get("comprehensive_description") or ri.get("detailed_description") or ri.get("concise_description") or ""
        )
        return f"{cur_task['object_category']} in the {ri['region_category'].lower()} that has {desc}", cur_task["object_category"]
    if task_type == "instance":
        inst = all_navigation_goals_dict[cur_task["instance_id"]]
        sentence = (
            inst.get("annot_unique_concise_description") if concise_description
            else inst.get("annot_unique_detailed_description")
        )
        sentence = sentence or inst.get("annot_unique_normal_description") or inst.get("annot_appearance_description") or ""
        return sentence, inst.get("object_category", "")
    raise ValueError(f"unknown task_type={task_type}")


def _capture_scan_frames(
    *, sim, agent, top_down_map, fog_of_war_mask, visibility_dist_in_pixels, total_steps, max_steps
):
    scan_rgb, scan_depth, scan_state = [], [], []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        agent_state = agent.get_state()
        scan_rgb.append(obs["color_sensor"][:, :, :3])
        scan_depth.append(obs["depth_sensor"][:, :])
        scan_state.append(agent_state)
        fog_of_war_mask = reveal_fog_of_war(
            top_down_map, fog_of_war_mask,
            map_coors_to_pixel(agent_state.position, top_down_map, sim),
            get_polar_angle(agent_state), 42, visibility_dist_in_pixels, False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return scan_rgb, scan_depth, scan_state, fog_of_war_mask, total_steps


def _find_follow_actions(*, path_finder, agent, raw_target, start_position, agent_island):
    raw = np.asarray(raw_target, dtype=float).reshape(3)
    start = np.asarray(start_position, dtype=float).reshape(3)
    rings = (0.0, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5)
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    seen: Set = set()
    candidates = []

    def _add(point, source, radius):
        try:
            snapped = np.asarray(path_finder.snap_point(point=point, island_index=agent_island), dtype=float).reshape(3)
        except Exception:
            return
        key = tuple(np.round(snapped, 3).tolist())
        if key in seen:
            return
        seen.add(key)
        candidates.append((snapped, source, radius))

    _add(raw, "direct", 0.0)
    for radius in rings[1:]:
        for angle in angles:
            _add(np.array([raw[0] + radius * math.cos(float(angle)), start[1], raw[2] + radius * math.sin(float(angle))]), "ring", radius)

    for snapped, source, radius in candidates:
        try:
            follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
            action_list = follower.find_path(snapped)
            if action_list is None:
                raise RuntimeError("None action_list")
            non_stop = [a for a in action_list if a]
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(start, dtype=float)
            path.requested_end = snapped
            if len(non_stop) == 0 and (path_finder.find_path(path) and path.geodesic_distance > 0.15):
                raise RuntimeError("zero actions for nonzero path")
            return list(action_list)
        except Exception:
            continue
    return []


def _follow_target(*, path_finder, agent, sim, target, prev_agent_state, total_steps, max_steps, episode_cum_distance):
    start_position = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    agent_island = int(path_finder.get_island(agent.get_state().position))
    action_list = _find_follow_actions(
        path_finder=path_finder, agent=agent,
        raw_target=np.asarray(target, dtype=float).reshape(3),
        start_position=start_position, agent_island=agent_island,
    )
    goto_rgb, goto_depth, goto_states = [], [], []
    for action in action_list:
        if not action:
            continue
        obs = sim.step(action=action)
        state = agent.get_state()
        goto_rgb.append(obs["color_sensor"][:, :, :3])
        goto_depth.append(obs["depth_sensor"][:, :])
        goto_states.append(state)
        total_steps += 1
        episode_cum_distance += float(np.linalg.norm(state.position - prev_agent_state.position))
        prev_agent_state = state
        if total_steps >= int(max_steps):
            break
    return goto_rgb, goto_depth, goto_states, prev_agent_state, total_steps, float(episode_cum_distance)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

TASK_LEVEL_ORDER = ("object", "room", "region", "instance")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize a single navigation episode (Vista2MQSC).")
    # Required target
    ap.add_argument("--scene_name",  required=True,  help="e.g. 00802-wcojb4TFT35")
    ap.add_argument("--episode_id",  required=True,  type=int)
    ap.add_argument(
        "--navigation_type",
        default="sequence",
        choices=["sequence", "object", "room", "region", "instance"],
        help="Use sequence for episode_by_sequence, or one task level for direct level episodes.",
    )
    ap.add_argument("--instance_id", default="", help="Optional guard for --navigation_type instance, e.g. armchair_906")
    ap.add_argument("--task_ids",    default="",     help="Comma-separated task indices to render (default: all)")
    # Paths
    ap.add_argument("--navigation_data_path", default=str(PROJECT_ROOT / "LangMap_Annotations"))
    ap.add_argument("--hm3d_data_base_path",  default=str(PROJECT_ROOT / "datascene"))
    ap.add_argument("--pq3d_stage1_path",     default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    ap.add_argument("--pq3d_stage2_path",     default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    ap.add_argument("--output_vis_dir",       default=str(PROJECT_ROOT / "visual-output" / "vis_sample"))
    # Nav params
    ap.add_argument("--max_steps",          type=int,   default=400)
    ap.add_argument("--decision_num_min",   type=int,   default=3)
    ap.add_argument("--success_distance",   type=float, default=0.25)
    ap.add_argument("--concise_description", action="store_true")
    ap.add_argument("--task_levels",        default="object,room,region,instance")
    # Extra visualization controls
    ap.add_argument("--max_frontier_facing", type=int, default=12, help="Max frontier-facing RGB renders per decision.")
    ap.add_argument("--max_saved_follow_frames", type=int, default=24, help="Max sampled follow RGB frames per decision.")
    ap.add_argument("--skip_birdseye_rgb", action="store_true", help="Skip extra Habitat top-down RGB camera renders.")
    ap.add_argument("--birdseye_resolution", type=int, default=768)
    ap.add_argument("--birdseye_height_m", type=float, default=7.5)
    ap.add_argument("--birdseye_hfov", type=float, default=75.0)
    args = ap.parse_args()

    enabled_task_levels = {x.strip() for x in args.task_levels.split(",") if x.strip()}
    filter_task_ids: Optional[Set[int]] = None
    if args.task_ids.strip():
        filter_task_ids = {int(x) for x in args.task_ids.split(",") if x.strip()}

    # ── Load scene data ──────────────────────────────────────────────────────
    nav_root = Path(os.path.expanduser(args.navigation_data_path))
    scene_gz = nav_root / f"{args.scene_name}.json.gz"
    if not scene_gz.exists():
        # recursive search
        found = list(nav_root.rglob(f"{args.scene_name}.json.gz"))
        if not found:
            raise FileNotFoundError(f"Scene data not found: {scene_gz}")
        scene_gz = found[0]

    print(f"[vis] Loading scene data: {scene_gz}")
    with gzip.open(scene_gz, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)

    region_to_annot_dict = scene_data.get("region_annotation", {})
    episode_mapping = {
        "object":   scene_data["episodes_by_object_level"],
        "room":     scene_data["episodes_by_room_level"],
        "region":   scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    all_nav_goals = {x["object_id"]: x for x in scene_data["goals"]}

    # Find the target episode. The user's armchair_906 example is an
    # episodes_by_instance_level record, not an episode_by_sequence record.
    target_ep = None
    if args.navigation_type == "sequence":
        for ep in scene_data["episode_by_sequence"]:
            if int(ep["episode_id"]) == args.episode_id:
                target_ep = ep
                break
    else:
        for ep in episode_mapping[args.navigation_type]:
            if int(ep["episode_id"]) != args.episode_id:
                continue
            if args.instance_id.strip() and ep.get("instance_id") != args.instance_id.strip():
                continue
            target_ep = ep
            break
    if target_ep is None:
        guard = f" instance_id={args.instance_id}" if args.instance_id.strip() else ""
        raise ValueError(
            f"{args.navigation_type} episode {args.episode_id}{guard} not found in scene {args.scene_name}"
        )

    # ── Setup simulator ──────────────────────────────────────────────────────
    sim_cfg  = OmegaConf.load(str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    agent_cfg = OmegaConf.load(str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))
    scene_path = resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), args.scene_name)
    sim_cfg["scene"] = scene_path
    print(f"[vis] Building HabitatSimulator: {scene_path}", flush=True)
    abstract_sim = HabitatSimulator(sim_cfg, agent_cfg)
    sim   = abstract_sim.simulator
    agent = abstract_sim.agent
    print("[vis] HabitatSimulator ready", flush=True)

    # Place agent at episode start
    agent_state = habitat_sim.AgentState()
    agent_state.position = target_ep["start_position"]
    agent_state.rotation = target_ep["start_rotation"]
    agent.set_state(agent_state)
    path_finder = sim.pathfinder

    top_down_map    = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
    fog_of_war_mask = np.zeros_like(top_down_map)
    area_thres_px   = convert_meters_to_pixel(9,   512, sim)
    vis_dist_px     = convert_meters_to_pixel(3.0, 512, sim)

    # ── PQ3D model ───────────────────────────────────────────────────────────
    print("[vis] Importing PQ3DModel", flush=True)
    from data_utils import PQ3DModel as _PQ3DModel

    print("[vis] Loading PQ3D model", flush=True)
    pq3d_model = _PQ3DModel(
        os.path.expanduser(args.pq3d_stage1_path),
        os.path.expanduser(args.pq3d_stage2_path),
        min_decision_num=int(args.decision_num_min),
    )
    pq3d_model.reset()
    pq3d_model.mask_generator = _DropTaskLevelMaskGenerator(pq3d_model.mask_generator)
    print("[vis] PQ3D model ready", flush=True)

    visited_frontier_set: Set = set()
    decision_num = 0
    episode_vis_dir = (
        Path(os.path.expanduser(args.output_vis_dir))
        / f"scene={args.scene_name}"
        / f"navigation_type={args.navigation_type}"
        / f"episode={args.episode_id}"
    )
    episode_vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vis] Output directory: {episode_vis_dir}")

    bird_renderer, bird_renderer_agent = None, None
    if not bool(args.skip_birdseye_rgb):
        bird_renderer, bird_renderer_agent = _build_birdseye_renderer(
            scene_path=scene_path,
            resolution=int(args.birdseye_resolution),
            sensor_height_m=float(args.birdseye_height_m),
            hfov_deg=float(args.birdseye_hfov),
        )

    try:
        if args.navigation_type == "sequence":
            task_iter = list(enumerate(target_ep["task_sequence"]))
        else:
            task_iter = [(0, [args.navigation_type, None])]

        for idx, cur_task_ref in task_iter:
            if args.navigation_type == "sequence":
                task_type, task_idx = cur_task_ref
                cur_task = episode_mapping[task_type][task_idx]
            else:
                task_type, task_idx = args.navigation_type, None
                cur_task = target_ep
            if task_type not in enabled_task_levels:
                continue
            if filter_task_ids is not None and idx not in filter_task_ids:
                continue

            goals      = [all_nav_goals[x] for x in cur_task["target_object_ids"]]
            goal_positions = [
                np.asarray(g.get("position", []), dtype=float).reshape(3)
                for g in goals
                if isinstance(g, dict) and len(g.get("position", [])) >= 3
            ]
            sentence, goal_category = build_sentence(
                task_type, cur_task,
                all_navigation_goals_dict=all_nav_goals,
                region_to_annot_dict=region_to_annot_dict,
                concise_description=bool(args.concise_description),
            )
            print(f"\n[vis] Task {idx} [{task_type}]: {sentence}")

            task_dir_name = f"task={idx}" if args.navigation_type == "sequence" else f"{task_type}_episode={args.episode_id}"
            task_vis_dir = episode_vis_dir / task_dir_name

            # Per-task tracking
            total_steps       = 0
            episode_cum_dist  = 0.0
            prev_agent_state  = agent.get_state()
            sub_start_pos     = np.asarray(prev_agent_state.position, dtype=float).copy()
            task_decision_start = decision_num
            task_end_reason   = "max_steps"

            # Trajectory tracking
            path_pixels:       List[Tuple[int, int]] = [_pos_to_pixel(sub_start_pos, top_down_map, sim)]
            decision_pixels:   List[Tuple[int, int]] = []
            final_dec_pixels:  List[Tuple[int, int]] = []
            sr, spl = 0.0, 0.0

            goto_rgb_list, goto_depth_list, goto_states_list = [], [], []

            while total_steps < int(args.max_steps):
                dec_dir = task_vis_dir / "decisions" / f"dec_{decision_num:03d}"
                print(f"  [vis] dec={decision_num:03d} loop start steps={total_steps}", flush=True)
                color_list, depth_list, state_list = [], [], []
                if len(goto_rgb_list) > 6:
                    step = max(1, len(goto_rgb_list) // 6)
                    goto_rgb_list   = [goto_rgb_list[i]   for i in range(0, len(goto_rgb_list),   step)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                    goto_states_list = [goto_states_list[i] for i in range(0, len(goto_states_list), step)][:6]
                color_list.extend(goto_rgb_list)
                depth_list.extend(goto_depth_list)
                state_list.extend(goto_states_list)

                # Scan
                print(f"  [vis] dec={decision_num:03d} scan start", flush=True)
                scan_rgb, scan_depth, scan_states, fog_of_war_mask, total_steps = _capture_scan_frames(
                    sim=sim, agent=agent, top_down_map=top_down_map,
                    fog_of_war_mask=fog_of_war_mask,
                    visibility_dist_in_pixels=vis_dist_px,
                    total_steps=total_steps, max_steps=int(args.max_steps),
                )
                print(
                    f"  [vis] dec={decision_num:03d} scan done frames={len(scan_rgb)} steps={total_steps}",
                    flush=True,
                )
                save_panorama_frames(dec_dir, scan_rgb, scan_depth)
                color_list.extend(scan_rgb)
                depth_list.extend(scan_depth)
                state_list.extend(scan_states)
                if total_steps >= int(args.max_steps):
                    break

                # Frontiers
                print(f"  [vis] dec={decision_num:03d} frontier start", flush=True)
                cur_agent_state = agent.get_state()
                frontier_waypoints_raw = detect_frontier_waypoints(
                    top_down_map, fog_of_war_mask, area_thres_px,
                    xy=map_coors_to_pixel(cur_agent_state.position, top_down_map, sim)[::-1],
                    enable_visualization=False,
                )
                if len(frontier_waypoints_raw) > 0:
                    frontier_waypoints = list(pixel_to_map_coors(
                        frontier_waypoints_raw[:, ::-1], cur_agent_state.position, top_down_map, sim
                    ))
                else:
                    frontier_waypoints = []
                frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]
                print(
                    f"  [vis] dec={decision_num:03d} frontier done count={len(frontier_waypoints)}",
                    flush=True,
                )

                # PQ3D decision
                print(
                    f"  [vis] dec={decision_num:03d} pq3d start frames={len(color_list)} frontiers={len(frontier_waypoints)}",
                    flush=True,
                )
                target_position, is_final = pq3d_model.decision(
                    color_list, depth_list, state_list, frontier_waypoints, sentence, decision_num
                )
                print(f"  [vis] dec={decision_num:03d} pq3d done final={bool(is_final)}", flush=True)
                used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                pq3d_aux = getattr(pq3d_model, "last_decision_aux", {}) or {}
                selected_frontier_idx = _nearest_frontier_index(used_target, frontier_waypoints)

                print(f"  dec={decision_num:03d}  final={is_final}  target={used_target.tolist()}"
                      f"  frontiers={len(frontier_waypoints)}")

                # Track decisions
                t_rc = _pos_to_pixel(used_target, top_down_map, sim)
                if is_final:
                    final_dec_pixels.append(t_rc)
                else:
                    decision_pixels.append(t_rc)
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))

                # ── Visualization saves ──────────────────────────────────────
                vis_kw = dict(
                    out_dir=task_vis_dir,
                    dec_num=decision_num,
                    top_down_map=top_down_map,
                    fog=fog_of_war_mask.copy(),
                    agent_state=cur_agent_state,
                    target=used_target,
                    is_final=bool(is_final),
                    path_pixels=list(path_pixels),
                    sim=sim,
                )
                save_topdown_map(**vis_kw)
                save_rgb_floor(**vis_kw)
                save_frontiers(**vis_kw, frontier_waypoints=frontier_waypoints)

                save_panorama_frames(dec_dir, scan_rgb, scan_depth)
                save_decision_maps(
                    dec_dir=dec_dir,
                    top_down_map=top_down_map,
                    fog=fog_of_war_mask.copy(),
                    agent_state=cur_agent_state,
                    target=used_target,
                    is_final=bool(is_final),
                    path_pixels=list(path_pixels),
                    frontier_waypoints=frontier_waypoints,
                    selected_frontier_idx=selected_frontier_idx,
                    goal_positions=goal_positions,
                    sim=sim,
                )
                _save_birdseye_rgb(
                    renderer=bird_renderer,
                    renderer_agent=bird_renderer_agent,
                    agent_state=cur_agent_state,
                    out_path=dec_dir / "birdseye_rgb.png",
                )
                facing_records = save_frontier_facing_views(
                    dec_dir=dec_dir,
                    sim=sim,
                    agent=agent,
                    agent_state=cur_agent_state,
                    target=used_target,
                    frontier_waypoints=frontier_waypoints,
                    selected_frontier_idx=selected_frontier_idx,
                    max_frontiers=int(args.max_frontier_facing),
                )

                decision_payload = {
                    "scene_name": args.scene_name,
                    "navigation_type": args.navigation_type,
                    "episode_id": int(args.episode_id),
                    "instance_id": cur_task.get("instance_id"),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "sentence": sentence,
                    "decision_num": int(decision_num),
                    "is_final": bool(is_final),
                    "steps_total_after_scan": int(total_steps),
                    "agent_position": np.asarray(cur_agent_state.position, dtype=float).reshape(3).tolist(),
                    "agent_rotation_xyzw": _rotation_xyzw(cur_agent_state.rotation),
                    "target_used": used_target.tolist(),
                    "selected_frontier_idx": selected_frontier_idx,
                    "frontiers": [
                        {
                            "index": int(i),
                            "position": np.asarray(fw, dtype=float).reshape(3).tolist(),
                            "is_selected": bool(selected_frontier_idx == i),
                        }
                        for i, fw in enumerate(frontier_waypoints)
                    ],
                    "frontier_facing_views": facing_records,
                    "goal_positions": [gp.tolist() for gp in goal_positions],
                    "goal_object_ids": list(cur_task.get("target_object_ids", [])),
                    "pq3d_aux": pq3d_aux,
                    "outputs": {
                        "decision_dir": str(dec_dir.relative_to(episode_vis_dir)),
                        "panorama_rgb_contact_sheet": "panorama_rgb_contact_sheet.png",
                        "panorama_depth_contact_sheet": "panorama_depth_contact_sheet.png",
                        "topdown_fog": "topdown_fog.png",
                        "topdown_full": "topdown_full.png",
                        "frontiers_on_topdown": "frontiers_on_topdown.png",
                        "birdseye_rgb": None if bool(args.skip_birdseye_rgb) else "birdseye_rgb.png",
                        "decision_target_facing_rgb": "decision_target_facing_rgb.png",
                    },
                }
                _write_json(dec_dir / "decision.json", decision_payload)

                # Follow
                (
                    goto_rgb_list, goto_depth_list, goto_states_list,
                    prev_agent_state, total_steps, episode_cum_dist,
                ) = _follow_target(
                    path_finder=path_finder, agent=agent, sim=sim,
                    target=used_target, prev_agent_state=prev_agent_state,
                    total_steps=total_steps, max_steps=int(args.max_steps),
                    episode_cum_distance=float(episode_cum_dist),
                )
                # Update path pixels after follow
                for s in goto_states_list:
                    path_pixels.append(_pos_to_pixel(np.asarray(s.position, dtype=float), top_down_map, sim))
                saved_follow_indices = save_follow_frames(
                    dec_dir,
                    goto_rgb_list,
                    max_saved=int(args.max_saved_follow_frames),
                )
                decision_payload["follow"] = {
                    "rgb_frame_count": int(len(goto_rgb_list)),
                    "saved_rgb_indices": [int(x) for x in saved_follow_indices],
                    "steps_total_after_follow": int(total_steps),
                    "episode_cum_distance_after_follow": float(episode_cum_dist),
                    "end_position_after_follow": np.asarray(agent.get_state().position, dtype=float).reshape(3).tolist(),
                }
                _write_json(dec_dir / "decision.json", decision_payload)

                decision_num += 1
                if bool(is_final):
                    task_end_reason = "final_decision"
                    break

            # ── Post-task metrics ────────────────────────────────────────────
            agent_state_end = agent.get_state()
            view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
            sp = habitat_sim.MultiGoalShortestPath()
            sp.requested_start = sub_start_pos
            sp.requested_ends  = view_points
            start_end_geo = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
            ep_path = habitat_sim.MultiGoalShortestPath()
            ep_path.requested_start = agent_state_end.position
            ep_path.requested_ends  = view_points
            end_geo = float(ep_path.geodesic_distance) if path_finder.find_path(ep_path) else float("inf")
            if not np.isinf(start_end_geo) and not np.isinf(end_geo):
                sr = 1.0 if end_geo <= float(args.success_distance) else 0.0
                spl = float(sr * start_end_geo / max(start_end_geo, max(float(episode_cum_dist), 1e-12)))

            summary = {
                "scene_name": args.scene_name,
                "episode_id": args.episode_id,
                "task_id":    idx,
                "task_level": task_type,
                "sentence":   sentence,
                "goal_category": goal_category,
                "sr":  float(sr),
                "spl": float(spl),
                "end_reason": task_end_reason,
                "start_goal_geo":     float(start_end_geo),
                "end_goal_geo":       float(end_geo),
                "episode_cum_distance": float(episode_cum_dist),
                "total_steps":        int(total_steps),
                "task_decisions":     int(decision_num - task_decision_start),
                "goal_positions":     [gp.tolist() for gp in goal_positions],
            }
            _write_json(task_vis_dir / "summary.json", summary)
            print(f"  SR={sr:.1f}  SPL={spl:.4f}  end={task_end_reason}  steps={total_steps}")

            # ── Trajectory figure ────────────────────────────────────────────
            save_trajectory(
                out_dir=task_vis_dir,
                task_id=idx,
                top_down_map=top_down_map,
                fog_final=fog_of_war_mask.copy(),
                sim=sim,
                start_position=sub_start_pos,
                end_position=np.asarray(agent_state_end.position, dtype=float),
                goal_positions=goal_positions,
                path_pixels=path_pixels,
                decision_pixels=decision_pixels,
                final_dec_pixels=final_dec_pixels,
                sentence=sentence,
                sr=float(sr),
                spl=float(spl),
                task_level=task_type,
            )

    finally:
        if bird_renderer is not None:
            bird_renderer.close()
        sim.close()

    print(f"\n[vis] Done. All outputs under: {episode_vis_dir}")


if __name__ == "__main__":
    main()
