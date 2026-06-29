"""Module-process visualization for the three anchor_nav decision modules.

This driver produces faithful per-module PROCESS LOGS (prompts, VLM input/output,
decisions, and visualizations). TFFS is routed through the real
hm3d-online/anchor_nav/tffs.py implementation; MQSC-R1 and VISTA-LS still use
lightweight visual process reconstructions where full PQ3D state is unavailable.
At sampled decision points it dumps a self-contained log folder per module:

  modules/tffs/dec_XXX/      Task-Facing Frontier Selection  (real frontier rerank, VLM per frontier view)
  modules/mqsc_r1/dec_XXX/   MQSC-R1 spatial consensus        (text decomposition VLM + footprint clustering)
  modules/vista_ls/final/    VISTA-LS viewpoint correction    (ring candidates: unreachable / too-close / bad-view / selected)

Every prompt and raw response is written to disk. The route follows the TFFS
selected frontier between decision rounds when available, so the trajectory log
reflects the frontier selector.

Every visualization carries a legend explaining its markers/colors.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import magnum as mn
import numpy as np
import quaternion

CODE_DIR = Path(__file__).resolve().parent
TELEOP_PATH = CODE_DIR / "interactive_vista2mqsc_teleop.py"
# CODE_DIR is .../visual-scripts/navi-visual/code -> repo root is parents[2].
PROJECT_ROOT = CODE_DIR.parents[2]
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"
ANCHOR_NAV = PROJECT_ROOT / "hm3d-online" / "anchor_nav"

# Make `vlm` resolve to hm3d-online/anchor_nav/vlm (the client the user requires).
for _p in (HM3D_ONLINE, ANCHOR_NAV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vlm.client as vlm_client  # noqa: E402  -> anchor_nav/vlm/client.py
import tffs  # noqa: E402  (prompt builder + pure scoring helpers only)
import mqsc_r1  # noqa: E402  (decomposition prompt + clustering primitives)
import vista_ls  # noqa: E402  (VistaLsConfig)
import habitat_sim.utils.common as hsu  # noqa: E402
from pic.joint import _save_rgb_jpg, _subsample_frames_evenly, stitch_panorama  # noqa: E402
from frontier_utils import get_polar_angle, map_coors_to_pixel  # noqa: E402


def _load_teleop() -> Any:
    spec = importlib.util.spec_from_file_location("teleop_mod", str(TELEOP_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["teleop_mod"] = module
    spec.loader.exec_module(module)
    return module


_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@contextlib.contextmanager
def _no_proxy():
    saved = {k: os.environ.get(k) for k in _PROXY_KEYS}
    no_saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    try:
        for k in _PROXY_KEYS:
            os.environ.pop(k, None)
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
        yield
    finally:
        for k, v in {**saved, **no_saved}.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def call_vlm(
    *, prompt: str, image_path: Optional[str], tag: str, model: str = "gpt-4o-mini",
    max_tokens: int = 256, simulated_fn=None,
) -> Dict[str, Any]:
    """Call the anchor_nav VLM client directly and capture full I/O."""
    rec: Dict[str, Any] = {
        "tag": tag,
        "vlm_client_file": vlm_client.__file__,
        "model": model,
        "prompt": prompt,
        "image_path": str(image_path) if image_path else None,
        "raw_response": "",
        "source": "",
        "ok": False,
        "error": "",
    }
    try:
        with _no_proxy():
            raw = vlm_client.chat(text=prompt, image_path=image_path, model=model, max_tokens=max_tokens)
        rec.update({"raw_response": str(raw), "source": "anchor_nav_vlm_real", "ok": True})
    except Exception as exc:  # network/key/etc. -> labelled simulated fallback
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raw = simulated_fn() if simulated_fn is not None else "{}"
        rec.update({"raw_response": str(raw), "source": "simulated_fallback", "ok": False})
    return rec


# ----------------------------- drawing helpers -----------------------------
def _draw_legend(img_bgr: np.ndarray, title: str, entries: List[Tuple[Tuple[int, int, int], str]]) -> None:
    """Draw a titled legend box (top-left) with colored swatches + labels."""
    pad, sw, line_h = 8, 16, 20
    n = len(entries) + 1
    box_w = 12 + sw + 8 + 250
    box_h = pad * 2 + n * line_h
    x0, y0 = 6, 6
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.82, img_bgr, 0.18, 0, img_bgr)
    cv2.rectangle(img_bgr, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), 1)
    y = y0 + pad + 14
    cv2.putText(img_bgr, title, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    y += line_h
    for color, label in entries:
        cv2.rectangle(img_bgr, (x0 + 10, y - 12), (x0 + 10 + sw, y), color, -1)
        cv2.rectangle(img_bgr, (x0 + 10, y - 12), (x0 + 10 + sw, y), (40, 40, 40), 1)
        cv2.putText(img_bgr, label, (x0 + 10 + sw + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h


def _save_raw_and_legend(img_markers: np.ndarray, base_path: Path, title: str,
                         entries: List[Tuple[Tuple[int, int, int], str]]) -> None:
    """Save the marker image twice: the raw (no-legend) version for the user to
    relabel, and a copy with the legend drawn on top."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base_path.with_name(base_path.name + ".png")), img_markers)
    leg = img_markers.copy()
    _draw_legend(leg, title, entries)
    cv2.imwrite(str(base_path.with_name(base_path.name + "_legend.png")), leg)


def _fill_border_connected_black_bgr(img_bgr: np.ndarray, fill: Tuple[int, int, int] = (184, 184, 184)) -> Tuple[np.ndarray, int]:
    """Replace simulator no-geometry background touching the image edge.

    Downward RGB sensors can render empty space outside the captured mesh as
    pure black.  Keep real dark content inside the scene, but neutralize the
    edge-connected background so the local topdown RGB is not mistaken for a
    black-bordered map.
    """
    img = np.asarray(img_bgr, dtype=np.uint8).copy()
    near_black = (img.max(axis=2) <= 8).astype(np.uint8)
    if near_black.size == 0 or int(near_black.sum()) == 0:
        return img, 0
    h, w = near_black.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    work = near_black.copy()
    seeds: List[Tuple[int, int]] = []
    xs = np.where(work[0, :] > 0)[0]
    seeds.extend((int(x), 0) for x in xs)
    xs = np.where(work[h - 1, :] > 0)[0]
    seeds.extend((int(x), h - 1) for x in xs)
    ys = np.where(work[:, 0] > 0)[0]
    seeds.extend((0, int(y)) for y in ys)
    ys = np.where(work[:, w - 1] > 0)[0]
    seeds.extend((w - 1, int(y)) for y in ys)
    for seed in seeds:
        if work[seed[1], seed[0]] > 0:
            cv2.floodFill(work, mask, seed, 2)
    edge_background = work == 2
    replaced = int(edge_background.sum())
    if replaced > 0:
        img[edge_background] = fill
    return img, replaced


def render_topdown_cam(nav: Any, sim: Any, base_paths: List[Path], log: List[str],
                       candidates: Tuple[float, ...] = (2.5, 2.3, 2.1, 1.9, 1.7, 1.5)) -> Optional[Dict[str, Any]]:
    """Render a colored top-down RGBD camera view (robot-centric, looks straight
    down). Adaptively picks the HIGHEST slice height that is not cut by the
    ceiling. The output is intentionally clean: no legend panel is burned into
    the photo, so white boxes/edges cannot be mistaken for simulator artifacts.
    Saves, per output dir:
      <base>_rgb.png       (plain colored local bird's-eye)
      <base>_depth.png     (colorized depth top-down)
      <base>_info.json     (chosen height diagnostics)
    """
    sensors = sim.get_agent(0)._sensors
    if "topdown_rgb" not in sensors or "topdown_depth" not in sensors:
        return None
    chosen: Optional[float] = None
    med = 0.0
    obs = None
    for H in candidates:
        sensors["topdown_rgb"].node.translation = mn.Vector3(0.0, float(H), 0.0)
        sensors["topdown_depth"].node.translation = mn.Vector3(0.0, float(H), 0.0)
        obs = sim.get_sensor_observations()
        d = np.asarray(obs["topdown_depth"], dtype=np.float32)
        v = d[d > 0]
        med = float(np.median(v)) if v.size else 0.0
        if med >= 0.6 * float(H):  # not cut by the ceiling (furniture may still lower it)
            chosen = float(H)
            break
    if chosen is None:
        chosen = float(candidates[-1])  # lowest slice already rendered
    rgb_bgr = cv2.cvtColor(np.asarray(obs["topdown_rgb"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
    rgb_bgr, edge_black_fill_px = _fill_border_connected_black_bgr(rgb_bgr)
    depth = np.asarray(obs["topdown_depth"], dtype=np.float32)
    dep_vis = cv2.cvtColor(_VIS._depth_to_rgb(depth), cv2.COLOR_RGB2BGR)
    slice_ok = bool(med >= 0.5 * chosen)  # only flags true ceiling occlusion
    for bp in base_paths:
        bp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(bp.with_name(bp.name + "_rgb.png")), rgb_bgr)
        cv2.imwrite(str(bp.with_name(bp.name + "_depth.png")), dep_vis)
        _write_json(
            bp.with_name(bp.name + "_info.json"),
            {
                "source": "robot_center_downward_rgbd_camera",
                "slice_height_m": float(chosen),
                "median_depth_m": float(med),
                "slice_ok": bool(slice_ok),
                "edge_connected_black_fill_px": int(edge_black_fill_px),
                "uncovered_fill": "neutral_gray_for_edge_connected_no_geometry_black",
                "note": "Clean local RGB photo; robot marker is stored in decision metadata, not drawn into the image.",
            },
        )
    log.append(
        f"[topdown_cam] slice_H={chosen:.1f} median_depth={med:.2f} "
        f"slice_ok={slice_ok} edge_black_fill_px={edge_black_fill_px} -> {len(base_paths)} dir(s)"
    )
    return {
        "slice_height_m": chosen,
        "median_depth_m": med,
        "slice_ok": slice_ok,
        "edge_connected_black_fill_px": edge_black_fill_px,
    }


def _base_topdown_bgr(nav: Any) -> np.ndarray:
    rgb = nav.VIS_NAV._base_rgb_floor(nav.top_down_map, nav.fog) if hasattr(nav, "VIS_NAV") else None
    if rgb is None:
        rgb = _VIS._base_rgb_floor(nav.top_down_map, nav.fog)
    return cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)


def _global_topdown_bgr(nav: Any, *, fog: bool) -> np.ndarray:
    fog_mask = nav.fog if fog else np.ones_like(nav.fog)
    rgb = nav.VIS_NAV._base_rgb_floor(nav.top_down_map, fog_mask) if hasattr(nav, "VIS_NAV") else None
    if rgb is None:
        rgb = _VIS._base_rgb_floor(nav.top_down_map, fog_mask)
    return cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)


def render_gray_topdown_rgb(
    nav: Any,
    sim: Any,
    *,
    fog: Optional[np.ndarray] = None,
    agent_state: Optional[Any] = None,
    target: Optional[np.ndarray] = None,
    is_final: bool = False,
    frontiers: Optional[List[np.ndarray]] = None,
    selected_frontier_idx: Optional[int] = None,
    draw_path: bool = True,
) -> np.ndarray:
    """Render the single TopDown Map type used by the demo.

    The base theme is intentionally only black / gray / white:
      - black: non-navigable / outside the current floor map
      - gray: navigable but unseen
      - white: navigable and explored

    Small colored overlays are limited to navigation markers. This avoids the
    older confusing family of topdown_map_rgb/navmesh_rgb/frontiers maps.
    """
    top = np.asarray(nav.top_down_map)
    fog_mask = np.asarray(nav.fog if fog is None else fog)
    navigable = top > 0
    explored = np.logical_and(navigable, fog_mask > 0)
    unexplored = np.logical_and(navigable, fog_mask <= 0)

    rgb = np.zeros((*top.shape[:2], 3), dtype=np.uint8)
    rgb[~navigable] = (28, 28, 28)
    rgb[unexplored] = (112, 112, 112)
    rgb[explored] = (238, 238, 238)
    shape = rgb.shape[:2]

    if draw_path:
        for prev, nxt in zip(getattr(nav, "path_pixels", [])[:-1], getattr(nav, "path_pixels", [])[1:]):
            cv2.line(rgb, (prev[1], prev[0]), (nxt[1], nxt[0]), (20, 120, 255), 2)

    goal_positions = list(getattr(getattr(nav, "ctx", None), "goal_positions", []) or [])
    for gp in goal_positions:
        _VIS._draw_star(rgb, _rc(gp, nav, sim), (255, 165, 0), size=9)

    if frontiers is not None:
        for idx, fw in enumerate(frontiers):
            color = (255, 0, 255) if selected_frontier_idx is not None and int(idx) == int(selected_frontier_idx) else (170, 170, 170)
            _VIS._draw_circle(rgb, _rc(fw, nav, sim), color, radius=6 if color == (255, 0, 255) else 4)

    used_target = target
    if used_target is None:
        used_target = getattr(nav, "current_target", None)
    if used_target is not None:
        trc = _VIS._clamp_rc(_rc(np.asarray(used_target, dtype=float), nav, sim), shape)
        color = (255, 60, 60) if bool(is_final) else (0, 120, 255)
        cv2.drawMarker(rgb, (trc[1], trc[0]), color, cv2.MARKER_CROSS, 18, 2)

    st = agent_state if agent_state is not None else nav.agent.get_state()
    arc = _VIS._clamp_rc(_rc(np.asarray(st.position, dtype=float), nav, sim), shape)
    _VIS._draw_agent_arrow(rgb, arc, float(get_polar_angle(st)), (255, 0, 0), size=11)
    return rgb


def save_decision_topdown_map(
    *,
    nav: Any,
    sim: Any,
    out_dir: Path,
    agent_state: Optional[Any],
    target: Optional[np.ndarray],
    is_final: bool,
    frontiers: Optional[List[np.ndarray]],
    selected_frontier_idx: Optional[int],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = render_gray_topdown_rgb(
        nav,
        sim,
        fog=nav.fog.copy(),
        agent_state=agent_state,
        target=target,
        is_final=bool(is_final),
        frontiers=frontiers,
        selected_frontier_idx=selected_frontier_idx,
    )
    cv2.imwrite(str(out_dir / "topdown_map.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    info = {
        "source": "single_gray_topdown_theme",
        "file": "topdown_map.png",
        "theme": {
            "black": "non_navigable_or_outside_current_floor",
            "gray": "navigable_unexplored",
            "white": "navigable_explored",
        },
        "overlays": {
            "red_arrow": "agent_position_and_heading",
            "blue_line": "path",
            "orange_star": "goal",
            "magenta_circle": "selected_frontier",
            "red_cross": "final_target",
            "blue_cross": "non_final_target",
        },
    }
    _write_json(out_dir / "topdown_map_info.json", info)
    return info


def _world_from_map_rc(nav: Any, sim: Any, rc: Tuple[int, int], y: float) -> np.ndarray:
    rz, rx = _M.maps.from_grid(
        int(rc[0]),
        int(rc[1]),
        (nav.top_down_map.shape[0], nav.top_down_map.shape[1]),
        sim,
    )
    return np.asarray([float(rx), float(y), float(rz)], dtype=float)


def _capture_topdown_rgb_tile(
    *,
    nav: Any,
    sim: Any,
    center_xyz: np.ndarray,
    height_candidates: Tuple[float, ...] = (2.2, 2.0, 1.8, 1.6, 1.4),
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    sensors = sim.get_agent(0)._sensors
    if "topdown_rgb" not in sensors or "topdown_depth" not in sensors:
        return None, {"ok": False, "reason": "topdown_rgbd_sensors_missing"}
    old_state = _M._state_copy(nav.agent.get_state())
    chosen = None
    med = 0.0
    obs = None
    try:
        st = _M.habitat_sim.AgentState()
        st.position = np.asarray(center_xyz, dtype=float).reshape(3)
        # Use a fixed world yaw for every tile. The downward sensor is attached
        # to the agent, so keeping the live yaw would rotate each tile
        # differently before we paste it into the global map.
        st.rotation = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        nav.agent.set_state(st)
        for H in height_candidates:
            sensors["topdown_rgb"].node.translation = mn.Vector3(0.0, float(H), 0.0)
            sensors["topdown_depth"].node.translation = mn.Vector3(0.0, float(H), 0.0)
            obs = sim.get_sensor_observations()
            d = np.asarray(obs["topdown_depth"], dtype=np.float32)
            valid = d[d > 0]
            med = float(np.median(valid)) if valid.size else 0.0
            if med >= 0.45 * float(H):
                chosen = float(H)
                break
        if obs is None:
            return None, {"ok": False, "reason": "sensor_observation_failed"}
        if chosen is None:
            chosen = float(height_candidates[-1])
        rgb_bgr = cv2.cvtColor(np.asarray(obs["topdown_rgb"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        depth = np.asarray(obs["topdown_depth"], dtype=np.float32)
        sensor_state = nav.agent.get_state().sensor_states.get("topdown_rgb")
        if sensor_state is None:
            return None, {"ok": False, "reason": "topdown_rgb_sensor_state_missing"}
        payload = {
            "rgb_bgr": rgb_bgr,
            "depth": depth,
            "camera_position": np.asarray(sensor_state.position, dtype=float).reshape(3).copy(),
            "camera_rotation_matrix": quaternion.as_rotation_matrix(sensor_state.rotation).astype(np.float32),
            "hfov_deg": float(getattr(nav.args, "topdown_cam_hfov", 90.0)),
        }
        return payload, {
            "ok": True,
            "slice_height_m": float(chosen),
            "median_depth_m": float(med),
        }
    finally:
        nav.agent.set_state(old_state)


def _project_rgbd_tile_to_map(
    *,
    tile: Dict[str, Any],
    map_shape: Tuple[int, int],
    sim: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(tile["rgb_bgr"], dtype=np.uint8)
    depth = np.asarray(tile["depth"], dtype=np.float32)
    h, w = depth.shape[:2]
    if rgb.shape[:2] != (h, w):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    hfov = float(tile.get("hfov_deg", 90.0))
    fx = (float(w) / 2.0) / math.tan(math.radians(hfov) / 2.0)
    fy = fx
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    valid = np.isfinite(depth) & (depth > 1e-4)
    vv, uu = np.where(valid)
    if uu.size == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    d = depth[vv, uu].astype(np.float32)
    x_cam = ((uu.astype(np.float32) - cx) / fx) * d
    y_cam = -((vv.astype(np.float32) - cy) / fy) * d
    z_cam = -d
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=0)
    Rcw = np.asarray(tile["camera_rotation_matrix"], dtype=np.float32).reshape(3, 3)
    C = np.asarray(tile["camera_position"], dtype=np.float32).reshape(3, 1)
    pts_world = (Rcw @ pts_cam) + C

    lower, upper = sim.pathfinder.get_bounds()
    lower = np.asarray(lower, dtype=np.float32).reshape(3)
    upper = np.asarray(upper, dtype=np.float32).reshape(3)
    grid_r = abs(float(upper[2] - lower[2])) / float(map_shape[0])
    grid_c = abs(float(upper[0] - lower[0])) / float(map_shape[1])
    rr = ((pts_world[2] - lower[2]) / max(grid_r, 1e-9)).astype(np.int32)
    cc = ((pts_world[0] - lower[0]) / max(grid_c, 1e-9)).astype(np.int32)
    colors_all = rgb[vv, uu]
    valid_color = np.max(colors_all, axis=1) > 8
    inside = (rr >= 0) & (rr < int(map_shape[0])) & (cc >= 0) & (cc < int(map_shape[1])) & valid_color
    if not np.any(inside):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )

    rr = rr[inside]
    cc = cc[inside]
    colors = colors_all[inside]
    du = uu[inside].astype(np.float32) - cx
    dv = vv[inside].astype(np.float32) - cy
    score = -(du * du + dv * dv)
    return rr, cc, colors, score.astype(np.float32)


def _fill_uncovered_by_dilation(mosaic: np.ndarray, filled: np.ndarray, max_iter: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    out = np.asarray(mosaic, dtype=np.uint8).copy()
    known = (np.asarray(filled) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for _ in range(int(max_iter)):
        holes = known == 0
        if not bool(np.any(holes)):
            break
        dilated_img = cv2.dilate(out, kernel)
        dilated_known = cv2.dilate(known, kernel)
        take = np.logical_and(holes, dilated_known > 0)
        if not bool(np.any(take)):
            break
        out[take] = dilated_img[take]
        known[take] = 1
    return out, known


def render_global_topdown_scene_rgb(nav: Any, sim: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a full-map RGB bird's-eye mosaic from many local downward RGB views."""
    cache = getattr(nav, "_global_topdown_scene_rgb_cache", None)
    if isinstance(cache, dict) and cache.get("shape") == list(nav.top_down_map.shape[:2]):
        return np.asarray(cache["image_bgr"], dtype=np.uint8).copy(), dict(cache.get("info", {}))

    base = _global_topdown_bgr(nav, fog=False)
    sensors = sim.get_agent(0)._sensors
    if "topdown_rgb" not in sensors or "topdown_depth" not in sensors:
        info = {"ok": False, "reason": "topdown_rgbd_sensors_missing", "source": "navmesh_color_fallback"}
        return base, info

    mpp = float(getattr(nav, "VIS_meters_per_px", 0.05))
    fov = float(getattr(nav.args, "topdown_cam_hfov", 90.0))
    tile_h = float(getattr(nav.args, "topdown_cam_height", 2.0))
    coverage_m = 2.0 * tile_h * math.tan(math.radians(fov) / 2.0)
    half_px = max(12, int(round((coverage_m * 0.5) / max(mpp, 1e-6))))
    step_px = max(10, int(round(half_px * 0.65)))
    max_tiles = 220

    navigable = np.asarray(nav.top_down_map) > 0
    ys, xs = np.where(navigable)
    if ys.size == 0:
        info = {"ok": False, "reason": "empty_topdown_map", "source": "navmesh_color_fallback"}
        return base, info
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())

    def _grid_centers(lo: int, hi: int, step: int, limit: int) -> List[int]:
        vals = list(range(int(lo), int(hi) + 1, int(step)))
        if len(vals) == 0 or vals[-1] != int(hi):
            vals.append(int(hi))
        return sorted(set(max(0, min(int(v), int(limit) - 1)) for v in vals))

    rows = _grid_centers(r0, r1, step_px, nav.top_down_map.shape[0])
    cols = _grid_centers(c0, c1, step_px, nav.top_down_map.shape[1])
    nav_coords = np.stack([ys.astype(np.int32), xs.astype(np.int32)], axis=1)
    centers_set = set()
    for r in rows:
        for c in cols:
            rr = max(0, min(int(r), navigable.shape[0] - 1))
            cc = max(0, min(int(c), navigable.shape[1] - 1))
            centers_set.add((rr, cc))
            if navigable[rr, cc]:
                continue
            # Grid intersections often land on furniture/holes while nearby
            # corridors are navigable. Snap the sampling center to the nearest
            # navigable pixel so narrow but valid regions receive RGB coverage.
            d2 = (nav_coords[:, 0] - rr) ** 2 + (nav_coords[:, 1] - cc) ** 2
            j = int(np.argmin(d2))
            if float(d2[j]) <= float(step_px * step_px):
                centers_set.add((int(nav_coords[j, 0]), int(nav_coords[j, 1])))
    centers = sorted(centers_set)
    if len(centers) > max_tiles:
        stride = int(math.ceil(math.sqrt(float(len(centers)) / float(max_tiles))))
        centers = centers[::stride]

    mosaic = np.zeros_like(base, dtype=np.uint8)
    mosaic_score = np.full(base.shape[:2], -np.inf, dtype=np.float32)
    filled = np.zeros(nav.top_down_map.shape[:2], dtype=np.uint8)
    frame_filled = np.zeros(nav.top_down_map.shape[:2], dtype=np.uint8)
    start_y = float(nav.agent.get_state().position[1])
    tile_records: List[Dict[str, Any]] = []

    for ti, rc in enumerate(centers):
        raw_center = _world_from_map_rc(nav, sim, rc, start_y)
        try:
            center = np.asarray(sim.pathfinder.snap_point(raw_center), dtype=float).reshape(3)
        except Exception:
            center = raw_center
        if not np.all(np.isfinite(center)):
            tile_records.append({
                "ok": False,
                "reason": "nonfinite_snap_point",
                "tile_index": int(ti),
                "center_rc": [int(rc[0]), int(rc[1])],
                "raw_center_xyz": raw_center.tolist(),
            })
            continue
        tile, rec = _capture_topdown_rgb_tile(nav=nav, sim=sim, center_xyz=center)
        rec.update({"tile_index": int(ti), "center_rc": [int(rc[0]), int(rc[1])], "center_xyz": center.tolist()})
        try:
            rt = _rc(center, nav, sim)
            rec["roundtrip_rc"] = [int(rt[0]), int(rt[1])]
            rec["roundtrip_error_px"] = float(math.hypot(float(rt[0] - rc[0]), float(rt[1] - rc[1])))
        except Exception as exc:
            rec["roundtrip_error"] = f"{type(exc).__name__}: {exc}"
        tile_records.append(rec)
        if tile is None:
            continue
        rr, cc, colors, score = _project_rgbd_tile_to_map(tile=tile, map_shape=base.shape[:2], sim=sim)
        rec["projected_px"] = int(rr.size)
        if rr.size == 0:
            continue
        take = score > mosaic_score[rr, cc]
        if not np.any(take):
            continue
        order = np.argsort(score[take], kind="mergesort")
        rr_t = rr[take][order]
        cc_t = cc[take][order]
        colors_t = colors[take][order]
        score_t = score[take][order]
        mosaic[rr_t, cc_t] = colors_t
        mosaic_score[rr_t, cc_t] = score_t
        frame_filled[rr_t, cc_t] = 1
        filled[np.logical_and(frame_filled > 0, navigable)] = 1

    fill_ratio = float(filled[navigable].mean()) if np.any(navigable) else 0.0
    projected_frame_fill_ratio = float(frame_filled.mean())
    projected_px = int(frame_filled.sum())
    if projected_px > 0:
        clean = np.full_like(base, 184, dtype=np.uint8)
        clean[np.asarray(nav.top_down_map) > 0] = (216, 216, 216)
        clean[frame_filled > 0] = mosaic[frame_filled > 0]
        mosaic = clean
    else:
        mosaic = np.full_like(base, 184, dtype=np.uint8)
    frame_fill_ratio = float(frame_filled.mean())
    info = {
        "ok": True,
        "source": "global_topdown_rgb_tile_mosaic",
        "tile_count": int(len(tile_records)),
        "successful_tile_count": int(sum(1 for r in tile_records if r.get("ok"))),
        "coverage_m": float(coverage_m),
        "half_patch_px": int(half_px),
        "step_px": int(step_px),
        "filled_navigable_ratio": float(fill_ratio),
        "projected_frame_ratio": float(projected_frame_fill_ratio),
        "filled_frame_ratio": float(frame_fill_ratio),
        "fallback_filled_px": 0,
        "dilation_filled_px": 0,
        "inpainted_uncovered_px": 0,
        "mpp": float(mpp),
        "paste_mode": "rgbd_projected_best_center_no_patch_fallback",
        "uncovered_fill": "neutral_gray_no_dilation_no_inpaint",
        "fixed_tile_agent_rotation": "identity_quaternion_world_yaw",
        "records": tile_records[:120],
    }
    nav._global_topdown_scene_rgb_cache = {
        "shape": list(nav.top_down_map.shape[:2]),
        "image_bgr": mosaic.copy(),
        "info": info,
    }
    return mosaic, info


def save_global_topdown_maps(
    *,
    nav: Any,
    sim: Any,
    out_dir: Path,
    title: str,
    frontiers: Optional[List[np.ndarray]] = None,
    selected_frontier_idx: Optional[int] = None,
    agent_state: Optional[Any] = None,
    log: Optional[List[str]] = None,
) -> None:
    """Save the two global map artifacts used in the demo.

    1. ``topdown_map.png``: the single gray TopDown Map theme.
    2. ``topdown_scene_rgb.png``: a clean full-scene RGB bird's-eye projection.

    Older duplicate names such as topdown_map_rgb/topdown_navmesh_rgb are no
    longer emitted because they did not actually encode different image types.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    save_decision_topdown_map(
        nav=nav,
        sim=sim,
        out_dir=out_dir,
        agent_state=agent_state,
        target=getattr(nav, "current_target", None),
        is_final=bool(getattr(nav, "current_target_is_final", False)),
        frontiers=frontiers,
        selected_frontier_idx=selected_frontier_idx,
    )
    scene_rgb, scene_info = render_global_topdown_scene_rgb(nav, sim)
    _write_json(out_dir / "topdown_scene_rgb_info.json", scene_info)
    cv2.imwrite(str(out_dir / "topdown_scene_rgb.png"), scene_rgb)
    if log is not None:
        log.append(
            f"[topdown_global] {out_dir}: saved gray topdown map + clean scene RGB "
            f"tiles={scene_info.get('successful_tile_count', 0)}/{scene_info.get('tile_count', 0)} "
            f"nav_fill={float(scene_info.get('filled_navigable_ratio', 0.0)):.2f} "
            f"frame_fill={float(scene_info.get('filled_frame_ratio', 0.0)):.2f}"
        )


_VIS = None  # set in main from teleop module


def _rc(pos, nav, sim) -> Tuple[int, int]:
    return _VIS._clamp_rc(_M._pos_to_pixel(np.asarray(pos, dtype=float), nav.top_down_map, sim), nav.top_down_map.shape[:2])


def _forward_xz(state) -> np.ndarray:
    f = hsu.quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
    return np.asarray([f[0], f[2]], dtype=float)


_M = None  # teleop module handle, set in main


# ----------------------------- Real TFFS rerank -----------------------------
def simulate_tffs(*, nav, sim, ctx, goal, dec_dir: Path, frontiers, views, agent_state, log) -> Dict[str, Any]:
    cfg = tffs.TffsConfig(vlm_call_interval=5)
    agent_xyz = np.asarray(agent_state.position, dtype=float).reshape(3)
    # Frontier prior for this visualization driver: closer-to-goal frontiers get
    # a stronger baseline logit, then real TFFS can rerank with current views.
    dists = np.asarray([np.linalg.norm((np.asarray(f) - goal)[[0, 2]]) for f in frontiers], dtype=float)
    logits = (-dists).astype(float)
    baseline_idx = int(np.argmax(logits)) if len(frontiers) else -1

    selected_idx, result = tffs.run_tffs_rerank(
        frontier_candidates=frontiers,
        task_text=ctx.sentence,
        frontier_logits=logits,
        panorama_views=views,
        agent_pose={"position": agent_xyz.tolist(), "heading_xz": _forward_xz(agent_state).tolist()},
        branch_is_frontier=True,
        baseline_frontier_index=baseline_idx,
        cfg=cfg,
        context={
            "decision_num": int(getattr(nav, "decision_num", 0) if hasattr(nav, "decision_num") else 0),
            "global_step": int(getattr(nav, "step_count", 0)),
            "scene_name": ctx.scene_name,
            "episode_id": int(ctx.episode_id),
            "task_id": int(ctx.task_id),
            "frontier_logits_source": "visual_sim_negative_distance_to_goal_prior",
        },
    )
    result["module"] = "tffs"
    result["real_tffs_file"] = str(Path(tffs.__file__).resolve())
    result["frontier_distance_to_goal_m"] = [float(x) for x in dists.tolist()]
    _write_json(dec_dir / "tffs_decision.json", result)

    prompt_dir = dec_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for score in list(result.get("vlm_scores", [])):
        fi = int(score.get("frontier_index", -1))
        if fi < 0:
            continue
        if score.get("prompt"):
            with open(prompt_dir / f"frontier_{fi:02d}_prompt.txt", "w", encoding="utf-8") as f:
                f.write(str(score.get("prompt", "")))
        with open(prompt_dir / f"frontier_{fi:02d}_response.txt", "w", encoding="utf-8") as f:
            f.write(str(score.get("raw", "")))

    # Visualization (with legend).
    img = _base_topdown_bgr(nav)
    fused = np.asarray(result.get("fused_score", []), dtype=float).reshape(-1)
    rerank_idx = int(result.get("rerank_frontier_index", baseline_idx))
    selected_idx = int(result.get("selected_frontier_index", selected_idx))
    applied = bool(result.get("tffs_applied", False))
    for gp in ctx.goal_positions:
        _VIS._draw_star(img, _rc(gp, nav, sim), (0, 165, 255), size=10)
    for fi, f in enumerate(frontiers):
        rcix = _rc(f, nav, sim)
        col = (180, 180, 180)  # default frontier gray
        if fi == baseline_idx:
            col = (0, 220, 220)  # baseline = yellow
        if fi == selected_idx and applied:
            col = (255, 0, 255)  # TFS-selected = magenta
        r = 7 if (fi == selected_idx or fi == baseline_idx) else 5
        _VIS._draw_circle(img, rcix, col, radius=r)
        label_score = float(fused[fi]) if fi < fused.size else float("nan")
        cv2.putText(img, f"{fi}:{label_score:.2f}", (rcix[1] + 8, rcix[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    arc = _rc(agent_xyz, nav, sim)
    _VIS._draw_agent_arrow(img, arc, float(get_polar_angle(agent_state)), (255, 0, 0), size=12)
    _save_raw_and_legend(img, dec_dir / "tffs_frontiers_topdown", "TFS frontier rerank", [
        ((0, 165, 255), "goal / target object"),
        ((255, 0, 0), "agent (pos + heading)"),
        ((0, 220, 220), "baseline frontier (max logit)"),
        ((255, 0, 255), "TFS-selected frontier (VLM rerank)"),
        ((180, 180, 180), "other frontier  [label = idx:fused_score]"),
    ])
    save_global_topdown_maps(
        nav=nav,
        sim=sim,
        out_dir=dec_dir,
        title="TFFS global top-down",
        frontiers=list(frontiers),
        selected_frontier_idx=selected_idx if 0 <= selected_idx < len(frontiers) else None,
        agent_state=agent_state,
    )
    log.append(f"[tffs] {dec_dir.name}: frontiers={len(frontiers)} baseline={baseline_idx} "
               f"rerank={rerank_idx} selected={selected_idx} applied={applied} "
               f"gate={result.get('gate_reason', '')} vlm_calls={result.get('vlm_call_count', 0)}")
    return result


# ----------------------------- MQSC-R1 simulation -----------------------------
def _simulated_decompose(sentence: str) -> str:
    words = [w for w in sentence.lower().replace(",", " ").split() if len(w) > 2]
    return json.dumps({
        "target_desc": " ".join(words[:3]) if words else "target object",
        "target_aliases": [], "anchor_primary": words[3:5], "anchor_support": words[5:8],
        "room_context": [], "relations": ["near"],
    })


def simulate_mqsc(*, nav, sim, ctx, goal, dec_dir: Path, agent_state, decompose_cache: Dict[str, Any], log) -> Dict[str, Any]:
    cfg = mqsc_r1.MqscR1Config()
    # 1. VLM text decomposition (cached across rounds; sentence is fixed).
    if "vlm" not in decompose_cache:
        prompt = mqsc_r1.build_decomposition_prompt(ctx.sentence, task_type=ctx.task_level)
        vlm_rec = call_vlm(prompt=prompt, image_path=None, tag="mqsc_decompose",
                           simulated_fn=lambda: _simulated_decompose(ctx.sentence))
        try:
            roles = mqsc_r1.parse_json_object(vlm_rec["raw_response"])
        except Exception as exc:
            roles = {"parse_error": f"{type(exc).__name__}: {exc}"}
        decompose_cache["vlm"] = vlm_rec
        decompose_cache["roles"] = roles
    vlm_rec = decompose_cache["vlm"]
    roles = decompose_cache["roles"]

    # 2. Synthesize candidate object decisions (no PQ3D in this imitation):
    #    a tight cluster near the goal + scattered distractors. The clustering
    #    primitives below are the REAL mqsc_r1 functions.
    pf = sim.pathfinder
    rng = np.random.RandomState(int(ctx.episode_id) + 7)
    cands: List[np.ndarray] = []
    sources: List[str] = []

    def add(p, src):
        try:
            sp = np.asarray(pf.snap_point(p), dtype=float).reshape(3)
            if np.all(np.isfinite(sp)):
                cands.append(sp)
                sources.append(src)
        except Exception:
            pass

    add(goal, "target_cluster")
    for _ in range(2):
        off = rng.uniform(-0.7, 0.7, size=2)
        add(np.array([goal[0] + off[0], goal[1], goal[2] + off[1]]), "target_cluster")
    for _ in range(2):
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(3.0, 5.0)
        add(np.array([goal[0] + rad * math.cos(ang), goal[1], goal[2] + rad * math.sin(ang)]), "distractor")

    xy = np.asarray([[c[0], c[2]] for c in cands], dtype=float)
    radius = np.full((len(cands),), 0.30, dtype=float)
    n = len(cands)
    # Fake logits: a lone distractor gets the single highest logit; the cluster
    # has higher AGGREGATE consensus -> demonstrates MQSC overriding the baseline.
    logits = np.array([1.1, 0.9, 0.8] + [1.6, 0.5][: max(0, n - 3)], dtype=float)[:n]
    probs = mqsc_r1._softmax(logits, cfg.temperature)
    comps = mqsc_r1._connected_components(list(range(n)), xy, radius, float(cfg.cluster_eps))

    comp_scores = []
    for comp in comps:
        noisy_or = mqsc_r1._noisy_or([float(probs[i]) for i in comp])
        compact = mqsc_r1._compactness(comp, xy, sigma=max(float(cfg.cluster_eps), 0.5))
        comp_scores.append({"members": [int(i) for i in comp], "size": len(comp),
                            "noisy_or_prob": float(noisy_or), "compactness": float(compact),
                            "consensus": float(noisy_or * compact)})
    baseline_idx = int(np.argmax(logits))
    best_comp = max(comp_scores, key=lambda c: c["consensus"]) if comp_scores else {"members": []}
    consensus_idx = int(max(best_comp["members"], key=lambda i: float(probs[i]))) if best_comp["members"] else baseline_idx
    applied = bool(consensus_idx != baseline_idx)

    result = {
        "module": "mqsc_r1",
        "prompt_version": mqsc_r1.PROMPT_VERSION,
        "sentence": ctx.sentence,
        "decomposition": {"vlm": vlm_rec, "parsed_roles": _jsonable(roles)},
        "cluster_eps_m": cfg.cluster_eps,
        "candidates": [
            {"object_index": i, "xyz": cands[i].tolist(), "source": sources[i],
             "footprint_radius_m": float(radius[i]), "logit": float(logits[i]), "prob": float(probs[i])}
            for i in range(n)
        ],
        "clusters": comp_scores,
        "baseline_object_index": baseline_idx,
        "selected_object_index": consensus_idx,
        "mqsc_applied": applied,
        "reason": "mqsc_selected_higher_consensus_target" if applied else "baseline_kept",
    }
    _write_json(dec_dir / "mqsc_r1_decision.json", result)
    with open(dec_dir / "decomposition_prompt.txt", "w") as f:
        f.write(vlm_rec["prompt"])
    with open(dec_dir / "decomposition_response.txt", "w") as f:
        f.write(vlm_rec["raw_response"])

    # Visualization (with legend): footprints colored per cluster.
    img = _base_topdown_bgr(nav)
    cluster_colors = [(0, 200, 0), (200, 120, 0), (0, 120, 200), (160, 0, 160)]
    member_to_comp = {}
    for ci, comp in enumerate(comps):
        for mi in comp:
            member_to_comp[mi] = ci
    for i in range(n):
        rcix = _rc(cands[i], nav, sim)
        col = cluster_colors[member_to_comp.get(i, 0) % len(cluster_colors)]
        rpx = max(4, int(radius[i] / nav.VIS_meters_per_px)) if hasattr(nav, "VIS_meters_per_px") else 8
        cv2.circle(img, (rcix[1], rcix[0]), rpx, col, 2)
        _VIS._draw_circle(img, rcix, col, radius=3)
        tag = []
        if i == baseline_idx:
            tag.append("base")
        if i == consensus_idx:
            tag.append("MQSC")
        label = f"{i}:{probs[i]:.2f}" + (("[" + ",".join(tag) + "]") if tag else "")
        cv2.putText(img, label, (rcix[1] + 8, rcix[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    if 0 <= baseline_idx < n:
        b = _rc(cands[baseline_idx], nav, sim)
        cv2.drawMarker(img, (b[1], b[0]), (0, 220, 220), cv2.MARKER_TILTED_CROSS, 18, 2)
    if 0 <= consensus_idx < n:
        s = _rc(cands[consensus_idx], nav, sim)
        cv2.drawMarker(img, (s[1], s[0]), (255, 0, 255), cv2.MARKER_STAR, 20, 2)
    _save_raw_and_legend(img, dec_dir / "mqsc_r1_clusters_topdown", "MQSC-R1 spatial consensus", [
        ((0, 200, 0), "cluster A (connected, footprint dist<=eps)"),
        ((200, 120, 0), "cluster B"),
        ((0, 120, 200), "cluster C / singletons"),
        ((0, 220, 220), "baseline pick (max single logit)"),
        ((255, 0, 255), "MQSC consensus pick"),
    ])
    log.append(f"[mqsc_r1] {dec_dir.name}: cands={n} clusters={len(comps)} "
               f"baseline={baseline_idx} consensus={consensus_idx} applied={applied} "
               f"vlm_source={vlm_rec['source']}")
    return result


# ----------------------------- VISTA-LS simulation -----------------------------
def _los_visibility(p_from: np.ndarray, p_to: np.ndarray, nav, sim, stop_margin_m: float = 0.45) -> float:
    """Fraction of the line-of-sight that is navigable, trimmed to stop just
    short of the object center (the target object itself occupies non-navigable
    cells, so a full ray to the center would always look 'occluded')."""
    p_from = np.asarray(p_from, dtype=float).reshape(3)
    p_to = np.asarray(p_to, dtype=float).reshape(3)
    dist = float(np.linalg.norm((p_to - p_from)[[0, 2]]))
    if dist < 1e-6:
        return 1.0
    frac = max(0.0, (dist - float(stop_margin_m)) / dist)
    p_stop = p_from + (p_to - p_from) * frac
    a = _rc(p_from, nav, sim)
    b = _rc(p_stop, nav, sim)
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
    rr = np.clip(np.linspace(a[0], b[0], n).astype(int), 0, nav.top_down_map.shape[0] - 1)
    cc = np.clip(np.linspace(a[1], b[1], n).astype(int), 0, nav.top_down_map.shape[1] - 1)
    nav_cells = (nav.top_down_map[rr, cc] > 0).astype(float)
    return float(nav_cells.mean())


def simulate_vista_ls(*, nav, sim, goal, agent_state, out_dir: Path, log) -> Dict[str, Any]:
    cfg = vista_ls.VistaLsConfig()
    pf = sim.pathfinder
    agent_xyz = np.asarray(agent_state.position, dtype=float).reshape(3)
    center = np.asarray(goal, dtype=float).reshape(3)
    # Colored top-down RGBD camera view at the final stop.
    render_topdown_cam(nav, sim, [out_dir / "topdown_cam"], log)
    try:
        agent_island = int(pf.get_island(agent_xyz))
    except Exception:
        agent_island = -1
    # Ring radii (to object center). The real module scores distance to the object
    # SURFACE, so we subtract a nominal footprint half-extent; inner rings then
    # surface the "too close" category, outer rings the "too far" one. Coarse
    # 30-deg angular step (vs the module's 5-deg) keeps the illustration readable.
    obj_radius_m = 0.35  # nominal target footprint (no PQ3D object point cloud in this imitation)
    radii = [0.30, 0.45, 0.60, 0.85, 1.10, 1.65]
    n_ang = 12
    records: List[Dict[str, Any]] = []
    for ri, radius in enumerate(radii):
        for ai in range(n_ang):
            theta = 2.0 * math.pi * ai / n_ang
            raw = np.array([center[0] + radius * math.cos(theta), agent_xyz[1],
                            center[2] + radius * math.sin(theta)], dtype=float)
            navigable = bool(pf.is_navigable(raw))
            point = raw.copy()
            snapped = None
            if not navigable:
                try:
                    snapped = np.asarray(pf.snap_point(raw), dtype=float).reshape(3)
                    if np.all(np.isfinite(snapped)) and float(np.linalg.norm(snapped - raw)) <= cfg.max_snap_distance_m:
                        point = snapped.copy()
                        navigable = bool(pf.is_navigable(point))
                except Exception:
                    snapped = None
            same_island = False
            reachable = False
            geo = float("inf")
            if navigable:
                try:
                    same_island = bool(pf.get_island(point) == agent_island)
                except Exception:
                    same_island = False
                spath = __import__("habitat_sim").ShortestPath()
                spath.requested_start = agent_xyz
                spath.requested_end = point
                reachable = bool(pf.find_path(spath)) and same_island
                geo = float(spath.geodesic_distance) if reachable else float("inf")
            center_dist = float(np.linalg.norm((point - center)[[0, 2]]))
            surface_dist = max(0.0, center_dist - obj_radius_m)
            visibility = _los_visibility(point, center, nav, sim) if navigable else 0.0

            # Categorize (priority order matches the module's feasibility gates).
            if not navigable or not reachable:
                category = "unreachable"
            elif surface_dist < cfg.shell_min_m:
                category = "too_close"
            elif surface_dist > cfg.shell_max_m:
                category = "too_far"
            elif visibility < 0.7:
                category = "bad_viewpoint"
            else:
                category = "feasible"
            records.append({
                "candidate_index": len(records), "radius_index": ri, "angle_index": ai,
                "radius_m": float(radius), "angle_rad": float(theta),
                "xyz": point.tolist(), "navigable": navigable, "reachable": reachable,
                "same_island": same_island, "geodesic_distance_m": geo,
                "center_distance_m": center_dist, "surface_distance_m": surface_dist,
                "visibility_score": round(visibility, 3), "category": category,
            })

    feasible = [r for r in records if r["category"] == "feasible"]
    selected = None
    if feasible:
        # Best viewpoint: max visibility, tie-break shorter path.
        selected = max(feasible, key=lambda r: (r["visibility_score"], -r["geodesic_distance_m"]))
        selected["selected_by_vista_ls"] = True

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    result = {
        "module": "vista_ls",
        "policy": "level-set ring sampling + reachability/shell/visibility viewpoint scoring",
        "config": {"candidate_radii_m": list(cfg.candidate_radii_m), "shell_min_m": cfg.shell_min_m,
                   "shell_max_m": cfg.shell_max_m, "min_visibility_score": cfg.min_visibility_score,
                   "max_snap_distance_m": cfg.max_snap_distance_m,
                   "nominal_object_radius_m": obj_radius_m,
                   "illustration_radii_m": radii, "illustration_angle_count": n_ang},
        "object_center_xyz": center.tolist(),
        "agent_position_xyz": agent_xyz.tolist(),
        "candidate_count": len(records),
        "category_counts": counts,
        "selected_viewpoint": selected,
        "candidates": records,
    }
    _write_json(out_dir / "vista_ls_decision.json", result)

    # Visualization (with legend): points colored by category, selected = star.
    img = _base_topdown_bgr(nav)
    cat_color = {
        "unreachable": (0, 0, 230), "too_close": (0, 140, 255), "too_far": (120, 120, 120),
        "bad_viewpoint": (0, 220, 220), "feasible": (0, 200, 0),
    }
    cand_px: List[Tuple[int, int]] = []
    center_px = _rc(center, nav, sim)
    _VIS._draw_star(img, center_px, (0, 165, 255), size=11)
    cand_px.append(center_px)
    for r in records:
        rcix = _rc(r["xyz"], nav, sim)
        cand_px.append(rcix)
        _VIS._draw_circle(img, rcix, cat_color.get(r["category"], (180, 180, 180)), radius=4)
    if selected is not None:
        s = _rc(selected["xyz"], nav, sim)
        cv2.drawMarker(img, (s[1], s[0]), (255, 0, 255), cv2.MARKER_STAR, 22, 3)
    arc = _rc(agent_xyz, nav, sim)
    cand_px.append(arc)
    _VIS._draw_agent_arrow(img, arc, float(get_polar_angle(agent_state)), (255, 0, 0), size=12)
    legend = [
        ((0, 165, 255), "object center (target)"),
        ((0, 0, 230), "unreachable (not navigable / no path)"),
        ((0, 140, 255), "too close (surface < shell_min=0.35m)"),
        ((0, 220, 220), "bad viewpoint (occluded, low visibility)"),
        ((0, 200, 0), "feasible candidate"),
        ((255, 0, 255), "VISTA-LS selected (best viewpoint)"),
        ((255, 0, 0), "agent (pos + heading)"),
    ]
    # Zoomed crop around the candidate ring (the points are tiny on the full map).
    rows = [p[0] for p in cand_px]
    cols = [p[1] for p in cand_px]
    pad = 18
    r0 = max(0, min(rows) - pad); r1 = min(img.shape[0], max(rows) + pad)
    c0 = max(0, min(cols) - pad); c1 = min(img.shape[1], max(cols) + pad)
    crop = img[r0:r1, c0:c1].copy()  # crop from the marker image BEFORE any legend
    if crop.size > 0:
        scale = max(1.0, 540.0 / max(crop.shape[1], 1))
        zoom = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        _save_raw_and_legend(zoom, out_dir / "vista_ls_candidates_zoom", "VISTA-LS viewpoint selection (zoom)", legend)
    _save_raw_and_legend(img, out_dir / "vista_ls_candidates_topdown", "VISTA-LS viewpoint selection", legend)

    # ---- Final target RGB result + point sampling projected onto the camera image ----
    import quaternion as _q
    agent = nav.agent

    def _herr() -> float:
        st = agent.get_state()
        f = hsu.quat_rotate_vector(st.rotation, np.array([0.0, 0.0, -1.0]))
        to = center - np.asarray(st.position, dtype=float).reshape(3)
        return math.degrees(math.atan2(float(f[0]) * float(to[2]) - float(f[2]) * float(to[0]),
                                       float(f[0]) * float(to[0]) + float(f[2]) * float(to[2])))

    if abs(_herr()) > 15.0:
        before = abs(_herr())
        nav.step_action("turn_left", 1, status_prefix="vista_face")
        turn = "turn_left" if abs(_herr()) < before else "turn_right"
        for _ in range(11):
            if abs(_herr()) <= 15.0:
                break
            nav.step_action(turn, 1, status_prefix="vista_face")
    st = agent.get_state()
    to = center - np.asarray(st.position, dtype=float).reshape(3)
    planar = float(np.linalg.norm(to[[0, 2]]))
    drop = float(st.position[1]) + 1.31 - float(center[1])
    n_tilt = max(0, min(3, int(round(math.degrees(math.atan2(max(drop, 0.0), max(planar, 1e-3))) / 30.0))))
    for _ in range(n_tilt):
        nav.step_action("look_down", 1, status_prefix="vista_tilt")

    obs = sim.get_sensor_observations()
    rgb_bgr = cv2.cvtColor(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
    H, W = rgb_bgr.shape[:2]
    cv2.imwrite(str(out_dir / "vista_ls_target_rgb_raw.png"), rgb_bgr)  # the extracted target photo

    sst = agent.get_state().sensor_states["color_sensor"]
    C = np.asarray(sst.position, dtype=float).reshape(3)
    Rcw = _q.as_rotation_matrix(sst.rotation)  # camera->world
    hfov = float(nav.args.hfov if nav.args.hfov > 0 else 42.0)
    fx = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    fy, cx, cy = fx, W / 2.0, H / 2.0

    def _project(P):
        pc = Rcw.T @ (np.asarray(P, dtype=float).reshape(3) - C)  # world -> camera
        depth = -float(pc[2])  # habitat camera looks down -Z
        if depth <= 1e-6:
            return None
        u = cx + fx * (float(pc[0]) / depth)
        v = cy - fy * (float(pc[1]) / depth)
        return (u, v)

    proj = rgb_bgr.copy()
    drawn = 0
    for r in records:
        uv = _project(r["xyz"])
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            continue
        cv2.circle(proj, (int(uv[0]), int(uv[1])), 6, cat_color.get(r["category"], (180, 180, 180)), -1)
        cv2.circle(proj, (int(uv[0]), int(uv[1])), 6, (20, 20, 20), 1)
        drawn += 1
    uvc = _project(center)
    if uvc and 0 <= uvc[0] < W and 0 <= uvc[1] < H:
        cv2.drawMarker(proj, (int(uvc[0]), int(uvc[1])), (0, 165, 255), cv2.MARKER_STAR, 20, 2)
    if selected is not None:
        uvs = _project(selected["xyz"])
        if uvs and 0 <= uvs[0] < W and 0 <= uvs[1] < H:
            cv2.drawMarker(proj, (int(uvs[0]), int(uvs[1])), (255, 0, 255), cv2.MARKER_STAR, 24, 3)
    _save_raw_and_legend(proj, out_dir / "vista_ls_target_rgb_points", "VISTA-LS points on target RGB", [
        ((0, 165, 255), "object center (target)"),
        ((0, 0, 230), "unreachable"),
        ((0, 140, 255), "too close"),
        ((0, 220, 220), "bad viewpoint"),
        ((0, 200, 0), "feasible candidate"),
        ((255, 0, 255), "VISTA-LS selected (best viewpoint)"),
    ])
    result["target_rgb"] = {
        "raw_photo": "vista_ls_target_rgb_raw.png",
        "points_overlay": "vista_ls_target_rgb_points.png (+ _legend)",
        "projected_candidate_count": int(drawn),
        "camera_position_xyz": C.tolist(), "hfov_deg": hfov, "resolution_wh": [W, H],
    }
    _write_json(out_dir / "vista_ls_decision.json", result)
    log.append(f"[vista_ls] candidates={len(records)} counts={counts} "
               f"selected={'yes' if selected else 'none'} target_rgb_points={drawn}")
    return result


# ----------------------------- scan + navigation -----------------------------
def scan_and_capture(nav, sim, pano_dir: Path) -> List[Dict[str, Any]]:
    pano_dir.mkdir(parents=True, exist_ok=True)
    views: List[Dict[str, Any]] = []
    scan_rgb: List[np.ndarray] = []
    for i in range(12):
        nav.step_action("turn_left", 1, status_prefix="scan")
        rgb, _depth, state = nav.context_buffer[-1]
        rgb_arr = np.asarray(rgb[:, :, :3], dtype=np.uint8).copy()
        scan_rgb.append(rgb_arr)
        path = pano_dir / f"view_{i:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
        views.append({"view_index": i, "image_path": str(path),
                      "yaw": float(get_polar_angle(state)),
                      "heading_xz": _forward_xz(state).tolist(),
                      "state": state})
    # The stitched ring view follows the same convention as VFV/PosNode:
    # current decision's final 12 turn_left frames, reversed before stitching.
    pano_frames = list(reversed(scan_rgb[-12:]))
    sampled = _subsample_frames_evenly(pano_frames, max_frames=12)
    if len(sampled) > 0:
        _save_rgb_jpg(stitch_panorama(sampled), pano_dir / "current_decision_panorama_vfv_order.jpg")
    return views


def main() -> None:
    global _M, _VIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--segment_advance_m", type=float, default=3.0)
    ap.add_argument("--arrive_thresh_m", type=float, default=0.7)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--logs_dir", default=str(CODE_DIR.parent / "logs" / "module_sim"))
    ap.add_argument("--live_dir", default=str(CODE_DIR.parent / "logs" / "live"))
    cli = ap.parse_args()

    _M = _load_teleop()
    _VIS = _M.VIS_NAV

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cli.logs_dir) / f"run={run_id}_ep{cli.episode_id}_{cli.instance_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod_dir = out_dir / "modules"
    (mod_dir / "tffs").mkdir(parents=True, exist_ok=True)
    (mod_dir / "mqsc_r1").mkdir(parents=True, exist_ok=True)
    (mod_dir / "vista_ls").mkdir(parents=True, exist_ok=True)

    sys.argv = ["teleop", "--scene_name", cli.scene_name, "--episode_id", str(cli.episode_id),
                "--navigation_type", cli.navigation_type, "--instance_id", cli.instance_id,
                "--task_id", str(cli.task_id), "--headless", "--disable_pq3d",
                "--enable_topdown_cam", "--topdown_cam_height", "2.0",
                "--logs_dir", str(out_dir), "--live_dir", str(cli.live_dir)]
    args = _M.parse_args()
    ctx = _M.load_task_context(args)
    scene_path = _M._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    sim, agent = _M.build_interactive_simulator(args, scene_path)
    nav = _M.InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)
    nav.VIS_NAV = _VIS
    nav.VIS_meters_per_px = float(_M.maps.calculate_meters_per_pixel(int(args.map_resolution), sim=sim))

    goal = np.asarray(ctx.goal_positions[0], dtype=float).reshape(3)
    log: List[str] = []
    print(f"[module_sim] sentence: {ctx.sentence}", flush=True)
    print(f"[module_sim] out_dir : {out_dir}", flush=True)
    print(f"[module_sim] vlm client: {vlm_client.__file__}", flush=True)

    def planar_to_goal():
        return float(np.linalg.norm((np.asarray(agent.get_state().position) - goal)[[0, 2]]))

    decompose_cache: Dict[str, Any] = {}
    rounds = 0
    while planar_to_goal() > cli.arrive_thresh_m and rounds < cli.max_rounds:
        dtag = f"dec_{rounds:03d}"
        nav.decision_num = int(rounds)
        print(f"[module_sim] === round {rounds}: scan + modules @ to_goal={planar_to_goal():.2f}m ===", flush=True)
        pano_dir = mod_dir / "tffs" / dtag / "panorama"
        views = scan_and_capture(nav, sim, pano_dir)
        frontiers = nav.detect_frontiers()
        agent_state = agent.get_state()
        # Colored top-down RGBD camera view for this decision (one per module).
        render_topdown_cam(nav, sim, [mod_dir / "tffs" / dtag / "topdown_cam",
                                      mod_dir / "mqsc_r1" / dtag / "topdown_cam"], log)
        tffs_result: Optional[Dict[str, Any]] = None
        if len(frontiers) >= 2:
            tffs_result = simulate_tffs(nav=nav, sim=sim, ctx=ctx, goal=goal, dec_dir=mod_dir / "tffs" / dtag,
                                        frontiers=frontiers, views=views, agent_state=agent_state, log=log)
        else:
            log.append(f"[tffs] {dtag}: skipped (only {len(frontiers)} frontier)")
            _write_json(mod_dir / "tffs" / dtag / "tffs_decision.json", {
                "module": "tffs",
                "skipped": True,
                "skip_reason": f"only_{len(frontiers)}_frontier",
                "decision_idx": int(rounds),
                "frontier_count": int(len(frontiers)),
                "real_tffs_file": str(Path(tffs.__file__).resolve()),
                "vlm_interval_allowed": False,
                "vlm_call_count": 0,
                "gate_reason": "skipped_insufficient_frontiers",
            })
            save_global_topdown_maps(
                nav=nav, sim=sim, out_dir=mod_dir / "tffs" / dtag,
                title="TFFS skipped global top-down", frontiers=list(frontiers),
                selected_frontier_idx=None, agent_state=agent_state,
            )
        simulate_mqsc(nav=nav, sim=sim, ctx=ctx, goal=goal, dec_dir=mod_dir / "mqsc_r1" / dtag,
                      agent_state=agent_state, decompose_cache=decompose_cache, log=log)
        save_global_topdown_maps(
            nav=nav, sim=sim, out_dir=mod_dir / "mqsc_r1" / dtag,
            title="MQSC-R1 global top-down", frontiers=list(frontiers),
            selected_frontier_idx=None, agent_state=agent_state,
        )

        segment_target = goal.copy()
        segment_is_final = True
        if tffs_result is not None:
            try:
                chosen_i = int(tffs_result.get("selected_frontier_index", -1))
            except Exception:
                chosen_i = -1
            if 0 <= chosen_i < len(frontiers):
                segment_target = np.asarray(frontiers[chosen_i], dtype=float).reshape(3).copy()
                segment_is_final = False

        nav.current_target = segment_target.copy()
        nav.current_target_is_final = bool(segment_is_final)
        actions, _follow = nav._plan_follow_actions(segment_target)
        seg_start = np.asarray(agent.get_state().position, dtype=float).reshape(3)
        for action in actions:
            if not action:
                continue
            nav.step_action(str(action), 1, status_prefix=f"goto[r{rounds}]")
            adv = float(np.linalg.norm((np.asarray(agent.get_state().position) - seg_start)[[0, 2]]))
            if planar_to_goal() <= cli.arrive_thresh_m or adv >= cli.segment_advance_m:
                break
        rounds += 1

    # VISTA-LS at the final stop (viewpoint correction toward the object).
    stop_reason = "arrived" if planar_to_goal() <= cli.arrive_thresh_m else "max_rounds_reached"
    print(f"[module_sim] stop_reason={stop_reason} (to_goal={planar_to_goal():.2f}m). Running final panorama + VISTA-LS viewpoint sim.", flush=True)
    final_pano_dir = mod_dir / "final_panorama"
    final_views = scan_and_capture(nav, sim, final_pano_dir)
    log.append(f"[final_panorama] stop_reason={stop_reason} views={len(final_views)} dir={final_pano_dir}")
    simulate_vista_ls(nav=nav, sim=sim, goal=goal, agent_state=agent.get_state(),
                      out_dir=mod_dir / "vista_ls" / "final", log=log)
    save_global_topdown_maps(
        nav=nav, sim=sim, out_dir=mod_dir / "vista_ls" / "final",
        title="VISTA-LS final global top-down", frontiers=list(getattr(nav, "current_frontiers", [])),
        selected_frontier_idx=None, agent_state=agent.get_state(), log=log,
    )

    nav.save_trajectory_snapshot("route_start_to_goal.png")
    # Add a legend to the trajectory image (the review noted it had none).
    traj_path = out_dir / "trajectory" / "route_start_to_goal.png"
    if traj_path.exists():
        timg = cv2.imread(str(traj_path))
        # Colors match VIS_NAV CLR_* converted RGB->BGR (only markers actually drawn here).
        _draw_legend(timg, "Trajectory overview", [
            ((0, 200, 0), "start position"),
            ((0, 165, 255), "goal / target object"),
            ((0, 200, 200), "walked path (line)"),
            ((0, 255, 255), "agent end (pos + heading arrow)"),
        ])
        cv2.imwrite(str(out_dir / "trajectory" / "route_start_to_goal_legend.png"), timg)

    _write_json(out_dir / "module_sim_summary.json", {
        "scene_name": ctx.scene_name, "episode_id": int(ctx.episode_id), "sentence": ctx.sentence,
        "decision_rounds": rounds, "final_planar_distance_to_goal_m": planar_to_goal(),
        "stop_reason": stop_reason,
        "vlm_client_file": vlm_client.__file__,
        "topdown_map": _jsonable(getattr(nav, "topdown_map_info", {})),
        "modules": {"tffs": "modules/tffs/dec_XXX", "mqsc_r1": "modules/mqsc_r1/dec_XXX",
                    "final_panorama": "modules/final_panorama", "vista_ls": "modules/vista_ls/final"},
        "log": log,
    })
    sim.close()
    print("[module_sim] DONE. Summary:", flush=True)
    for line in log:
        print("   " + line, flush=True)
    print(f"[module_sim] outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
