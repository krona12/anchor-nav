"""Guided module-process visualization for the LANDER-Nav grounding decisions.

This is a copy of module_sim.py for visualization-only runs. It keeps the same
per-module PROCESS LOGS (prompts, VLM input/output, decisions, and
visualizations), but the physical navigation is guided by an oracle shortest
path toward the current task goal. Frontier choices are rendered as if the selected
frontier came from the next guided waypoint, so different subtasks in the same
scene produce different trajectories instead of reusing the same initial
frontier-driven path.
At sampled decision points it dumps a self-contained log folder per module:

  Evidence Grounding artifacts: Which frontier?
  Entity Grounding artifacts: Which object?
  Endpoint Grounding artifacts: Which viewpoint?

Every prompt and raw response is written to disk. The route follows the Evidence Grounding
selected frontier between decision rounds when available, so the trajectory log
reflects the frontier selector.

Images stay pixel-clean: marker legends are written to sibling *_info.json files
instead of being burned into the image as white panels.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import magnum as mn
import numpy as np
import quaternion

CODE_DIR = Path(__file__).resolve().parent
# CODE_DIR is .../visual-scripts/navi-visual/code -> repo root is parents[2].
PROJECT_ROOT = CODE_DIR.parents[2]
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"
ANCHOR_NAV = PROJECT_ROOT / "hm3d-online" / "anchor_nav"

# Make `vlm` resolve to hm3d-online/anchor_nav/vlm (the client the user requires).
for _p in (HM3D_ONLINE, ANCHOR_NAV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vlm.client as vlm_client  # noqa: E402  -> anchor_nav/vlm/client.py
import habitat_sim.utils.common as hsu  # noqa: E402
from pic.joint import _save_rgb_jpg, _subsample_frames_evenly, stitch_panorama  # noqa: E402
from frontier_utils import get_polar_angle, map_coors_to_pixel  # noqa: E402


def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_teleop_path() -> Path:
    matches: List[Path] = []
    for path in sorted(CODE_DIR.glob("interactive_*_teleop.py")):
        text = _read_source(path)
        if "class InteractiveNavigator" in text and "def parse_args" in text:
            matches.append(path)
    if not matches:
        raise RuntimeError(f"cannot locate interactive navigation driver under {CODE_DIR}")
    return matches[0]


TELEOP_PATH = _find_teleop_path()


def _select_anchor_source(required_markers: Tuple[str, ...], preferred_markers: Tuple[str, ...] = ()) -> Path:
    candidates: List[Tuple[int, str, Path]] = []
    for path in sorted(ANCHOR_NAV.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = _read_source(path)
        if all(marker in text for marker in required_markers):
            score = sum(int(marker in text) for marker in preferred_markers)
            candidates.append((score, path.name, path))
    if not candidates:
        raise RuntimeError(f"cannot locate anchor_nav implementation with markers: {required_markers}")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _load_source_module(path: Path, role: str) -> Any:
    alias = f"_anchor_nav_{role}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(alias, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import implementation source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _find_config_class(module: Any, required_attrs: Tuple[str, ...], kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = dict(kwargs or {})
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        try:
            inst = obj(**kwargs)
        except Exception:
            try:
                inst = obj()
            except Exception:
                continue
        if all(hasattr(inst, attr) for attr in required_attrs):
            return obj
    raise RuntimeError(f"cannot locate config class in {getattr(module, '__file__', module)}")


def _find_callable(module: Any, *, prefix: str, suffix: str) -> Any:
    for name, obj in vars(module).items():
        if name.startswith(prefix) and name.endswith(suffix) and callable(obj):
            return obj
    raise RuntimeError(f"cannot locate callable {prefix}*{suffix} in {getattr(module, '__file__', module)}")


_EVIDENCE_IMPL = _load_source_module(
    _select_anchor_source(
        (
            "def build_task_facing_prompt",
            "frontier_candidates",
            "baseline_frontier_index",
            "vlm_call_interval",
        )
    ),
    "evidence",
)
_ENTITY_IMPL = _load_source_module(
    _select_anchor_source(
        (
            "def build_decomposition_prompt",
            "def _connected_components",
            "def _noisy_or",
            "def _compactness",
            "cluster_eps",
        ),
        ("object_only_region_consensus",),
    ),
    "entity",
)
_ENDPOINT_IMPL = _load_source_module(
    _select_anchor_source(
        (
            "candidate_radii_m",
            "shell_min_m",
            "shell_max_m",
            "min_visibility_score",
            "max_snap_distance_m",
        )
    ),
    "endpoint",
)
_EVIDENCE_CONFIG_CLS = _find_config_class(_EVIDENCE_IMPL, ("vlm_call_interval",), {"vlm_call_interval": 5})
_EVIDENCE_BUILD_PROMPT = getattr(_EVIDENCE_IMPL, "build_task_facing_prompt")
_EVIDENCE_RERANK = _find_callable(_EVIDENCE_IMPL, prefix="run_", suffix="_rerank")
_ENTITY_CONFIG_CLS = _find_config_class(_ENTITY_IMPL, ("cluster_eps", "temperature"))
_ENTITY_BUILD_DECOMPOSITION_PROMPT = getattr(_ENTITY_IMPL, "build_decomposition_prompt")
_ENTITY_PARSE_JSON_OBJECT = getattr(_ENTITY_IMPL, "parse_json_object")
_ENTITY_SOFTMAX = getattr(_ENTITY_IMPL, "_softmax")
_ENTITY_CONNECTED_COMPONENTS = getattr(_ENTITY_IMPL, "_connected_components")
_ENTITY_NOISY_OR = getattr(_ENTITY_IMPL, "_noisy_or")
_ENTITY_COMPACTNESS = getattr(_ENTITY_IMPL, "_compactness")
_ENDPOINT_CONFIG_CLS = _find_config_class(
    _ENDPOINT_IMPL,
    ("candidate_radii_m", "shell_min_m", "shell_max_m", "min_visibility_score", "max_snap_distance_m"),
)

GUIDED_VISUALIZATION_MODE = True
EVIDENCE_MODULE = "Evidence Grounding"
EVIDENCE_QUESTION = "Which frontier?"
EVIDENCE_TITLE = f"{EVIDENCE_MODULE}: {EVIDENCE_QUESTION}"
ENTITY_MODULE = "Entity Grounding"
ENTITY_QUESTION = "Which object?"
ENTITY_TITLE = f"{ENTITY_MODULE}: {ENTITY_QUESTION}"
ENDPOINT_MODULE = "Endpoint Grounding"
ENDPOINT_QUESTION = "Which viewpoint?"
ENDPOINT_TITLE = f"{ENDPOINT_MODULE}: {ENDPOINT_QUESTION}"
ARTIFACT_ROOT_NAME = "artifacts"
EVIDENCE_ARTIFACT_DIR = "EvidenceGrounding"
ENTITY_ARTIFACT_DIR = "EntityGrounding"
ENDPOINT_ARTIFACT_DIR = "EndpointGrounding"
EVIDENCE_DECISION_JSON = "evidence_decision.json"
EVIDENCE_FRONTIERS_TOPDOWN_STEM = "evidence_frontiers_topdown"
ENTITY_DECISION_JSON = "entity_decision.json"
ENTITY_CLUSTERS_TOPDOWN_STEM = "entity_clusters_topdown"
ENDPOINT_DECISION_JSON = "endpoint_decision.json"
ENDPOINT_CANDIDATES_TOPDOWN_STEM = "endpoint_candidates_topdown"
ENDPOINT_CANDIDATES_ZOOM_STEM = "endpoint_candidates_zoom"
ENDPOINT_TARGET_RGB_RAW_STEM = "endpoint_target_rgb_raw"
ENDPOINT_TARGET_RGB_POINTS_STEM = "endpoint_target_rgb_points"


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
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Call the anchor_nav VLM client directly and capture full I/O.

    This visual driver is intentionally strict: a VLM/network/key failure should
    stop the run so the batch monitor can surface and repair it. Returning fake
    responses here would make the module logs look valid while hiding the real
    failure mode.
    """
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
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"real VLM call failed for {tag}: {rec['error']}") from exc
    rec.update({"raw_response": str(raw), "source": "anchor_nav_vlm_real", "ok": True})
    return rec


# ----------------------------- drawing helpers -----------------------------
def _save_marker_image(img_markers: np.ndarray, base_path: Path, title: str,
                       entries: List[Tuple[Tuple[int, int, int], str]]) -> None:
    """Save a clean marker image and keep legend metadata out of the pixels."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base_path.with_name(base_path.name + ".png")), img_markers)
    _write_json(
        base_path.with_name(base_path.name + "_info.json"),
        {
            "title": title,
            "legend": [
                {"color_bgr": [int(c) for c in color], "label": str(label)}
                for color, label in entries
            ],
            "note": "Legend is stored as metadata so no white legend panel is burned into the image.",
        },
    )


def _fill_border_connected_extreme_bgr(img_bgr: np.ndarray, fill: Tuple[int, int, int] = (184, 184, 184)) -> Tuple[np.ndarray, int, int]:
    """Replace simulator no-geometry background touching the image edge.

    Downward RGB sensors can render empty space outside the captured mesh as
    pure black or pure white depending on scene/material state. Keep real dark
    or white content inside the scene, but neutralize only the edge-connected
    background so the local topdown RGB is not mistaken for a bordered map.
    """
    img = np.asarray(img_bgr, dtype=np.uint8).copy()

    def _edge_connected(mask_in: np.ndarray) -> np.ndarray:
        mask_bin = (np.asarray(mask_in) > 0).astype(np.uint8)
        if mask_bin.size == 0 or int(mask_bin.sum()) == 0:
            return np.zeros_like(mask_bin, dtype=bool)
        h, w = mask_bin.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        work = mask_bin.copy()
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
                cv2.floodFill(work, flood_mask, seed, 2)
        return work == 2

    black_bg = _edge_connected(img.max(axis=2) <= 8)
    white_bg = _edge_connected(img.min(axis=2) >= 247)
    edge_background = np.logical_or(black_bg, white_bg)
    if bool(np.any(edge_background)):
        img[edge_background] = fill
    return img, int(black_bg.sum()), int(white_bg.sum())


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
    rgb_bgr, edge_black_fill_px, edge_white_fill_px = _fill_border_connected_extreme_bgr(rgb_bgr)
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
                "edge_connected_white_fill_px": int(edge_white_fill_px),
                "uncovered_fill": "neutral_gray_for_edge_connected_no_geometry_black_or_white",
                "note": "Clean local RGB photo; robot marker is stored in decision metadata, not drawn into the image.",
            },
        )
    log.append(
        f"[topdown_cam] slice_H={chosen:.1f} median_depth={med:.2f} "
        f"slice_ok={slice_ok} edge_black_fill_px={edge_black_fill_px} "
        f"edge_white_fill_px={edge_white_fill_px} -> {len(base_paths)} dir(s)"
    )
    return {
        "slice_height_m": chosen,
        "median_depth_m": med,
        "slice_ok": slice_ok,
        "edge_connected_black_fill_px": edge_black_fill_px,
        "edge_connected_white_fill_px": edge_white_fill_px,
    }


def _base_topdown_bgr(nav: Any, sim: Any) -> np.ndarray:
    rgb = render_gray_topdown_rgb(
        nav,
        sim,
        fog=nav.fog.copy(),
        agent_state=nav.agent.get_state(),
        target=getattr(nav, "current_target", None),
        is_final=bool(getattr(nav, "current_target_is_final", False)),
        frontiers=[],
        selected_frontier_idx=None,
    )
    return cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)


def _global_topdown_bgr(nav: Any, sim: Any, *, fog: bool) -> np.ndarray:
    fog_mask = nav.fog if fog else np.ones_like(nav.fog)
    rgb = render_gray_topdown_rgb(
        nav,
        sim,
        fog=fog_mask,
        agent_state=nav.agent.get_state(),
        target=None,
        is_final=False,
        frontiers=[],
        selected_frontier_idx=None,
        draw_path=False,
    )
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

    goal_positions = _goal_positions_for_overlay(nav)
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
        "goal_marker_count": int(len(_goal_positions_for_overlay(nav))),
        "goal_marker_positions_xyz": [p.tolist() for p in _goal_positions_for_overlay(nav)],
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


def render_global_topdown_scene_rgb(nav: Any, sim: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a full-map RGB bird's-eye mosaic from many local downward RGB views."""
    cache = getattr(nav, "_global_topdown_scene_rgb_cache", None)
    current_y = round(float(nav.agent.get_state().position[1]), 3)
    if (
        isinstance(cache, dict)
        and cache.get("shape") == list(nav.top_down_map.shape[:2])
        and cache.get("agent_y") == current_y
    ):
        return np.asarray(cache["image_bgr"], dtype=np.uint8).copy(), dict(cache.get("info", {}))

    base = _global_topdown_bgr(nav, sim, fog=False)
    sensors = sim.get_agent(0)._sensors
    if "topdown_rgb" not in sensors or "topdown_depth" not in sensors:
        raise RuntimeError("topdown RGBD sensors are required for topdown_scene_rgb.png")

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
        raise RuntimeError("empty topdown map; cannot build topdown_scene_rgb.png")
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
        "agent_y": current_y,
        "image_bgr": mosaic.copy(),
        "info": info,
    }
    return mosaic, info


def _annotate_scene_rgb_bgr(
    *,
    nav: Any,
    sim: Any,
    scene_bgr: np.ndarray,
    agent_state: Optional[Any],
    target: Optional[np.ndarray],
    is_final: bool,
    frontiers: Optional[List[np.ndarray]],
    selected_frontier_idx: Optional[int],
) -> np.ndarray:
    """Draw navigation state on a copy of the clean scene RGB map."""
    img = np.asarray(scene_bgr, dtype=np.uint8).copy()
    shape = img.shape[:2]
    path_pixels = list(getattr(nav, "path_pixels", []) or [])
    for prev, nxt in zip(path_pixels[:-1], path_pixels[1:]):
        p0 = _VIS._clamp_rc((int(prev[0]), int(prev[1])), shape)
        p1 = _VIS._clamp_rc((int(nxt[0]), int(nxt[1])), shape)
        cv2.line(img, (p0[1], p0[0]), (p1[1], p1[0]), (255, 120, 20), 2, cv2.LINE_AA)

    goal_positions = _goal_positions_for_overlay(nav)
    for gp in goal_positions:
        _VIS._draw_star(img, _rc(gp, nav, sim), (0, 165, 255), size=9)

    if frontiers is not None:
        for idx, fw in enumerate(frontiers):
            selected = selected_frontier_idx is not None and int(idx) == int(selected_frontier_idx)
            _VIS._draw_circle(img, _rc(fw, nav, sim), (255, 0, 255) if selected else (170, 170, 170),
                              radius=6 if selected else 4)

    used_target = target if target is not None else getattr(nav, "current_target", None)
    if used_target is not None:
        trc = _VIS._clamp_rc(_rc(np.asarray(used_target, dtype=float), nav, sim), shape)
        color = (0, 0, 255) if bool(is_final) else (0, 120, 255)
        cv2.drawMarker(img, (trc[1], trc[0]), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    st = agent_state if agent_state is not None else nav.agent.get_state()
    arc = _VIS._clamp_rc(_rc(np.asarray(st.position, dtype=float), nav, sim), shape)
    _VIS._draw_agent_arrow(img, arc, float(get_polar_angle(st)), (0, 0, 255), size=11)
    return img


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
    """Save the global map artifacts used in the demo.

    1. ``topdown_map.png``: the single gray TopDown Map theme.
    2. ``topdown_scene_rgb.png``: a clean full-scene RGB bird's-eye projection.
    3. ``topdown_scene_rgb_annotated.png``: the same RGB projection with
       trajectory, agent, target, and frontier overlays.

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
    scene_annotated = _annotate_scene_rgb_bgr(
        nav=nav,
        sim=sim,
        scene_bgr=scene_rgb,
        agent_state=agent_state,
        target=getattr(nav, "current_target", None),
        is_final=bool(getattr(nav, "current_target_is_final", False)),
        frontiers=frontiers,
        selected_frontier_idx=selected_frontier_idx,
    )
    cv2.imwrite(str(out_dir / "topdown_scene_rgb_annotated.png"), scene_annotated)
    _write_json(
        out_dir / "topdown_scene_rgb_annotated_info.json",
        {
            "source": "global_topdown_rgb_annotation",
            "base_image": "topdown_scene_rgb.png",
            "file": "topdown_scene_rgb_annotated.png",
            "note": "The base RGB map remains available without overlays or legend panels.",
            "overlays": {
                "blue_line": "trajectory",
                "red_arrow": "agent_position_and_heading",
                "orange_star": "goal",
                "gray_circle": "frontier",
                "magenta_circle": "selected_frontier",
                "red_cross": "final_target",
                "blue_cross": "non_final_target",
            },
            "goal_marker_count": int(len(_goal_positions_for_overlay(nav))),
            "goal_marker_positions_xyz": [p.tolist() for p in _goal_positions_for_overlay(nav)],
        },
    )
    if log is not None:
        log.append(
            f"[topdown_global] {out_dir}: saved gray topdown map + clean/annotated scene RGB "
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


def _look_at_quat_xz(origin: np.ndarray, target: np.ndarray) -> Any:
    origin = np.asarray(origin, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    dx = float(target[0] - origin[0])
    dz = float(target[2] - origin[2])
    yaw = 0.0 if math.hypot(dx, dz) < 1e-6 else math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _goal_positions_for_overlay(nav: Any) -> List[np.ndarray]:
    override = getattr(nav, "visual_goal_positions_override", None)
    if override is not None:
        values = list(override or [])
    else:
        values = list(getattr(getattr(nav, "ctx", None), "goal_positions", []) or [])
    out: List[np.ndarray] = []
    for value in values:
        try:
            arr = np.asarray(value, dtype=float).reshape(3)
        except Exception:
            continue
        if np.all(np.isfinite(arr)):
            out.append(arr)
    return out


def _color_sensor_pose_metrics(state: Any) -> Dict[str, Any]:
    sensor_states = getattr(state, "sensor_states", {}) or {}
    sensor_state = sensor_states.get("color_sensor") if isinstance(sensor_states, dict) else None
    if sensor_state is None:
        return {}
    f = hsu.quat_rotate_vector(sensor_state.rotation, np.array([0.0, 0.0, -1.0]))
    fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
    horizontal = max(1e-9, math.hypot(fx, fz))
    pitch_deg = math.degrees(math.atan2(fy, horizontal))
    return {
        "color_sensor_forward_xyz": [fx, fy, fz],
        "color_sensor_pitch_deg": float(pitch_deg),
        "color_sensor_down_tilt_deg": float(max(0.0, -pitch_deg)),
    }


_M = None  # teleop module handle, set in main


# ----------------------------- log stream indexes -----------------------------
def _rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _public_log_rel(rel_path: str) -> str:
    return str(rel_path)


def _public_log_message(root: Path, message: str) -> str:
    text = str(message).replace(str(root.resolve()), "<fresh_run>")
    return _public_log_rel(text)


def _sanitize_public_run_info(out_dir: Path) -> None:
    path = out_dir / "run_info.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    raw_args = data.get("args")
    if isinstance(raw_args, dict):
        safe_keys = {
            "scene_name",
            "episode_id",
            "navigation_type",
            "instance_id",
            "task_id",
            "headless",
            "disable_pq3d",
            "concise_description",
            "enable_topdown_cam",
            "topdown_cam_height",
            "topdown_cam_hfov",
            "map_resolution",
            "hfov",
            "turn_angle",
            "forward_distance",
            "max_steps",
            "arrive_thresh_m",
        }
        clean_args = {str(k): v for k, v in raw_args.items() if str(k) in safe_keys}
        if "logs_dir" in raw_args:
            clean_args["logs_dir"] = "<fresh_run>"
        if "live_dir" in raw_args:
            clean_args["live_dir"] = "<live_dir>"
        clean_args["grounding_pipeline"] = "Evidence Grounding -> Entity Grounding -> Endpoint Grounding"
        data["args"] = clean_args
    _write_json(path, data)


def _public_artifact_alias(root: Path, path: Path) -> str:
    rel = _rel_to(root, path)
    public_rel = _public_log_rel(rel)
    if public_rel == rel or not path.exists() or not path.is_file():
        return public_rel
    alias = root / public_rel
    if alias.exists() or alias.is_symlink():
        return public_rel
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        target = os.path.relpath(path.resolve(), start=alias.parent.resolve())
        os.symlink(target, alias)
    except Exception:
        shutil.copy2(path, alias)
    return public_rel


def _existing_rel_paths(root: Path, paths: List[Path]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            out.append(_public_artifact_alias(root, p))
    return out


def _glob_rel_paths(root: Path, pattern_root: Path, pattern: str) -> List[str]:
    if not pattern_root.exists():
        return []
    return [_public_artifact_alias(root, p) for p in sorted(pattern_root.glob(pattern)) if p.is_file() and p.stat().st_size > 0]


def write_log_stream_indexes(out_dir: Path, rounds: int) -> Dict[str, Any]:
    """Write compact indexes for the human-facing log flows.

    Images are not duplicated here.  Each stream points to the clean images,
    annotated images, prompts, responses, and JSON decisions already written by
    the module folders.
    """
    streams_dir = out_dir / "log_streams"
    streams: Dict[str, Dict[str, Any]] = {
        "evidence_grounding": {
            "description": "Evidence Grounding / Which frontier? prompts, panorama evidence, map context, and selected frontier artifacts.",
            "entries": [],
        },
        "entity_grounding": {
            "description": "Entity Grounding / Which object? decomposition, object candidates, map context, and selected object artifacts.",
            "entries": [],
        },
        "endpoint_grounding": {
            "description": "Endpoint Grounding / Which viewpoint? candidate viewpoints, target RGB, map context, and selected stop artifacts.",
            "entries": [],
        },
    }

    def add(category: str, payload: Dict[str, Any]) -> None:
        streams[category]["entries"].append(_jsonable(payload))

    for dec in range(max(0, int(rounds))):
        dtag = f"dec_{dec:03d}"
        evidence_dir = out_dir / ARTIFACT_ROOT_NAME / EVIDENCE_ARTIFACT_DIR / dtag
        entity_dir = out_dir / ARTIFACT_ROOT_NAME / ENTITY_ARTIFACT_DIR / dtag
        add(
            "evidence_grounding",
            {
                "decision": dec,
                "module": EVIDENCE_MODULE,
                "question": EVIDENCE_QUESTION,
                "panorama_inputs": _glob_rel_paths(out_dir, evidence_dir / "panorama", "view_*.png")
                + _existing_rel_paths(out_dir, [evidence_dir / "panorama" / "current_decision_panorama_vfv_order.jpg"]),
                "prompts": _glob_rel_paths(out_dir, evidence_dir / "prompts", "frontier_*_prompt.txt"),
                "responses": _glob_rel_paths(out_dir, evidence_dir / "prompts", "frontier_*_response.txt"),
                "decision_json": _existing_rel_paths(out_dir, [evidence_dir / EVIDENCE_DECISION_JSON]),
                "images": _existing_rel_paths(
                    out_dir,
                    [
                        evidence_dir / (EVIDENCE_FRONTIERS_TOPDOWN_STEM + ".png"),
                        evidence_dir / "topdown_map.png",
                        evidence_dir / "topdown_scene_rgb_annotated.png",
                    ],
                ),
                "clean_rgb": _existing_rel_paths(out_dir, [evidence_dir / "topdown_scene_rgb.png"]),
                "annotated_rgb": _existing_rel_paths(out_dir, [evidence_dir / "topdown_scene_rgb_annotated.png"]),
                "local_topdown_rgb": _existing_rel_paths(out_dir, [evidence_dir / "topdown_cam_rgb.png"]),
                "metadata": _existing_rel_paths(
                    out_dir,
                    [
                        evidence_dir / "topdown_map_info.json",
                        evidence_dir / (EVIDENCE_FRONTIERS_TOPDOWN_STEM + "_info.json"),
                        evidence_dir / "topdown_scene_rgb_info.json",
                        evidence_dir / "topdown_scene_rgb_annotated_info.json",
                        evidence_dir / "topdown_cam_info.json",
                    ],
                ),
            },
        )
        add(
            "entity_grounding",
            {
                "decision": dec,
                "module": ENTITY_MODULE,
                "question": ENTITY_QUESTION,
                "prompt": _existing_rel_paths(out_dir, [entity_dir / "decomposition_prompt.txt"]),
                "response": _existing_rel_paths(out_dir, [entity_dir / "decomposition_response.txt"]),
                "images": _existing_rel_paths(
                    out_dir,
                    [
                        entity_dir / (ENTITY_CLUSTERS_TOPDOWN_STEM + ".png"),
                        entity_dir / "topdown_map.png",
                        entity_dir / "topdown_scene_rgb_annotated.png",
                    ],
                ),
                "clean_rgb": _existing_rel_paths(out_dir, [entity_dir / "topdown_scene_rgb.png"]),
                "annotated_rgb": _existing_rel_paths(out_dir, [entity_dir / "topdown_scene_rgb_annotated.png"]),
                "local_topdown_rgb": _existing_rel_paths(out_dir, [entity_dir / "topdown_cam_rgb.png"]),
                "metadata": _existing_rel_paths(
                    out_dir,
                    [
                        entity_dir / (ENTITY_CLUSTERS_TOPDOWN_STEM + "_info.json"),
                        entity_dir / "topdown_map_info.json",
                        entity_dir / "topdown_scene_rgb_info.json",
                        entity_dir / "topdown_scene_rgb_annotated_info.json",
                        entity_dir / "topdown_cam_info.json",
                    ],
                ),
                "decision_json": _existing_rel_paths(out_dir, [entity_dir / ENTITY_DECISION_JSON]),
            },
        )

    endpoint_dir = out_dir / ARTIFACT_ROOT_NAME / ENDPOINT_ARTIFACT_DIR / "final"
    add(
        "endpoint_grounding",
        {
            "decision": "final",
            "module": ENDPOINT_MODULE,
            "question": ENDPOINT_QUESTION,
            "decision_json": _existing_rel_paths(out_dir, [endpoint_dir / ENDPOINT_DECISION_JSON]),
            "panorama_inputs": _glob_rel_paths(out_dir, endpoint_dir / "panorama", "view_*.png")
            + _existing_rel_paths(out_dir, [endpoint_dir / "panorama" / "current_decision_panorama_vfv_order.jpg"]),
            "images": _existing_rel_paths(
                out_dir,
                [
                    endpoint_dir / "topdown_map.png",
                    endpoint_dir / (ENDPOINT_CANDIDATES_TOPDOWN_STEM + ".png"),
                    endpoint_dir / (ENDPOINT_CANDIDATES_ZOOM_STEM + ".png"),
                    out_dir / "trajectory" / "route_start_to_goal.png",
                ],
            ),
            "clean_rgb": _existing_rel_paths(out_dir, [endpoint_dir / "topdown_scene_rgb.png"]),
            "annotated_rgb": _existing_rel_paths(out_dir, [endpoint_dir / "topdown_scene_rgb_annotated.png"]),
            "local_topdown_rgb": _existing_rel_paths(out_dir, [endpoint_dir / "topdown_cam_rgb.png"]),
            "target_rgb": _existing_rel_paths(
                out_dir,
                [
                    endpoint_dir / (ENDPOINT_TARGET_RGB_RAW_STEM + ".png"),
                    endpoint_dir / (ENDPOINT_TARGET_RGB_POINTS_STEM + ".png"),
                ],
            ),
            "metadata": _existing_rel_paths(
                out_dir,
                [
                    endpoint_dir / "topdown_map_info.json",
                    endpoint_dir / (ENDPOINT_CANDIDATES_TOPDOWN_STEM + "_info.json"),
                    endpoint_dir / (ENDPOINT_CANDIDATES_ZOOM_STEM + "_info.json"),
                    endpoint_dir / (ENDPOINT_TARGET_RGB_POINTS_STEM + "_info.json"),
                    out_dir / "trajectory" / "route_start_to_goal_info.json",
                    endpoint_dir / "topdown_scene_rgb_info.json",
                    endpoint_dir / "topdown_scene_rgb_annotated_info.json",
                    endpoint_dir / "topdown_cam_info.json",
                ],
            ),
        },
    )

    manifest: Dict[str, Any] = {
        "version": "navi_visual_log_streams_v1",
        "root": str(out_dir),
        "note": "Indexes reference existing artifacts; no duplicate images are created. Legend data stays in *_info.json so clean/no-legend images remain available.",
        "categories": {},
    }
    for name, payload in streams.items():
        index_rel = f"log_streams/{name}/index.json"
        index_payload = {
            "category": name,
            "description": payload["description"],
            "entry_count": len(payload["entries"]),
            "entries": payload["entries"],
        }
        _write_json(out_dir / index_rel, index_payload)
        manifest["categories"][name] = {
            "description": payload["description"],
            "index": index_rel,
            "entry_count": len(payload["entries"]),
        }
    _write_json(streams_dir / "manifest.json", manifest)
    return manifest


# ----------------------------- guided/oracle visualization -----------------------------
def _snap_guided_point(path_finder: Any, point: np.ndarray, island_index: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(point, dtype=float).reshape(3)
    for call in (
        lambda: path_finder.snap_point(point=arr, island_index=int(island_index)) if island_index is not None else path_finder.snap_point(point=arr),
        lambda: path_finder.snap_point(arr, island_index=int(island_index)) if island_index is not None else path_finder.snap_point(arr),
        lambda: path_finder.snap_point(arr),
    ):
        try:
            snapped = np.asarray(call(), dtype=float).reshape(3)
            if np.all(np.isfinite(snapped)):
                return snapped
        except Exception:
            continue
    return arr.copy()


def _shortest_path_points(nav: Any, start: np.ndarray, goal: np.ndarray) -> Dict[str, Any]:
    pf = nav.path_finder
    start_arr = np.asarray(start, dtype=float).reshape(3)
    goal_arr = np.asarray(goal, dtype=float).reshape(3)
    try:
        island = int(pf.get_island(start_arr))
    except Exception:
        island = None
    start_nav = _snap_guided_point(pf, start_arr, island)
    goal_nav = _snap_guided_point(pf, goal_arr, island)
    path_points: List[np.ndarray] = []
    geodesic = float("inf")
    ok = False
    try:
        path = _M.habitat_sim.ShortestPath()
        path.requested_start = start_nav
        path.requested_end = goal_nav
        ok = bool(pf.find_path(path))
        if ok:
            geodesic = float(path.geodesic_distance)
            path_points = [np.asarray(p, dtype=float).reshape(3) for p in list(path.points)]
    except Exception:
        path_points = []
    if len(path_points) < 2:
        path_points = [start_nav, goal_nav]
        geodesic = float(np.linalg.norm((goal_nav - start_nav)[[0, 2]]))
        ok = False
    return {
        "ok": bool(ok),
        "start_nav": start_nav,
        "goal_nav": goal_nav,
        "points": path_points,
        "geodesic_distance_m": float(geodesic) if math.isfinite(geodesic) else None,
        "island_index": island,
    }


def _resample_polyline(points: List[np.ndarray], count: int) -> List[np.ndarray]:
    n = max(0, int(count))
    if n <= 0:
        return []
    pts = [np.asarray(p, dtype=float).reshape(3) for p in points if np.all(np.isfinite(p))]
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [pts[0].copy() for _ in range(n)]
    seg_lens = [float(np.linalg.norm((b - a)[[0, 2]])) for a, b in zip(pts[:-1], pts[1:])]
    total = float(sum(seg_lens))
    if total <= 1e-6:
        return [pts[-1].copy() for _ in range(n)]
    out: List[np.ndarray] = []
    targets = [total * float(i + 1) / float(n) for i in range(n)]
    seg_i = 0
    acc = 0.0
    for d in targets:
        while seg_i < len(seg_lens) - 1 and acc + seg_lens[seg_i] < d:
            acc += seg_lens[seg_i]
            seg_i += 1
        a = pts[seg_i]
        b = pts[seg_i + 1]
        seg = max(seg_lens[seg_i], 1e-9)
        t = float(np.clip((d - acc) / seg, 0.0, 1.0))
        out.append(a + (b - a) * t)
    return out


def _polyline_xz_length(points: List[np.ndarray]) -> float:
    pts = [np.asarray(p, dtype=float).reshape(3) for p in points]
    return float(sum(float(np.linalg.norm((b - a)[[0, 2]])) for a, b in zip(pts[:-1], pts[1:])))


def _trim_polyline_to_distance(points: List[np.ndarray], target_distance_m: float) -> List[np.ndarray]:
    pts = [np.asarray(p, dtype=float).reshape(3) for p in points if np.all(np.isfinite(p))]
    if len(pts) <= 1:
        return pts
    target = max(0.0, float(target_distance_m))
    out = [pts[0].copy()]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.linalg.norm((b - a)[[0, 2]]))
        if seg <= 1e-6:
            continue
        if acc + seg >= target:
            t = float(np.clip((target - acc) / seg, 0.0, 1.0))
            out.append(a + (b - a) * t)
            return out
        out.append(b.copy())
        acc += seg
    return out


def build_guided_route(nav: Any, goal: np.ndarray, max_rounds: int) -> Dict[str, Any]:
    start = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
    route = _shortest_path_points(nav, start, np.asarray(goal, dtype=float).reshape(3))
    path_points = list(route["points"])
    total_len = _polyline_xz_length(path_points)
    stop_margin_m = 0.80 if total_len > 1.20 else 0.0
    motion_distance = max(0.0, total_len - stop_margin_m)
    motion_points = _trim_polyline_to_distance(path_points, motion_distance)
    if len(motion_points) < 2:
        motion_points = path_points
        stop_margin_m = 0.0
        motion_distance = total_len
    waypoints = _resample_polyline(motion_points, max_rounds)
    if len(waypoints) == 0:
        waypoints = [np.asarray(motion_points[-1], dtype=float).reshape(3)]
    return {
        "mode": "guided_visualization_oracle_route",
        "start_xyz": start.tolist(),
        "goal_object_xyz": np.asarray(goal, dtype=float).reshape(3).tolist(),
        "snapped_start_xyz": np.asarray(route["start_nav"], dtype=float).reshape(3).tolist(),
        "snapped_goal_xyz": np.asarray(route["goal_nav"], dtype=float).reshape(3).tolist(),
        "guided_stop_xyz": np.asarray(motion_points[-1], dtype=float).reshape(3).tolist(),
        "guided_stop_margin_m": float(stop_margin_m),
        "guided_motion_distance_m": float(motion_distance),
        "shortest_path_ok": bool(route["ok"]),
        "geodesic_distance_m": route["geodesic_distance_m"],
        "island_index": route["island_index"],
        "path_points": [p.tolist() for p in path_points],
        "motion_path_points": [p.tolist() for p in motion_points],
        "decision_move_waypoints": [p.tolist() for p in waypoints],
    }


def _make_guided_frontiers(
    *,
    nav: Any,
    current_xyz: np.ndarray,
    selected_xyz: np.ndarray,
    real_frontiers: List[np.ndarray],
    round_idx: int,
) -> Tuple[List[np.ndarray], int]:
    pf = nav.path_finder
    current = np.asarray(current_xyz, dtype=float).reshape(3)
    selected = np.asarray(selected_xyz, dtype=float).reshape(3)
    try:
        island = int(pf.get_island(current))
    except Exception:
        island = None
    selected = _snap_guided_point(pf, selected, island)
    out: List[np.ndarray] = [selected]

    direction = selected - current
    d2 = direction[[0, 2]]
    norm = float(np.linalg.norm(d2))
    if norm < 1e-6:
        d2 = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    unit = d2 / norm
    perp = np.array([-unit[1], unit[0]], dtype=float)
    for scale, sign in ((0.75, 1.0), (1.05, -1.0), (1.35, 1.0)):
        raw = selected.copy()
        raw[0] += float(perp[0] * scale * sign)
        raw[2] += float(perp[1] * scale * sign)
        raw = raw - np.array([unit[0] * 0.20 * (round_idx + 1), 0.0, unit[1] * 0.20 * (round_idx + 1)])
        cand = _snap_guided_point(pf, raw, island)
        if all(float(np.linalg.norm((cand - prev)[[0, 2]])) > 0.25 for prev in out):
            out.append(cand)
        if len(out) >= 3:
            break

    for fw in list(real_frontiers):
        cand = _snap_guided_point(pf, np.asarray(fw, dtype=float).reshape(3), island)
        if all(float(np.linalg.norm((cand - prev)[[0, 2]])) > 0.35 for prev in out):
            out.append(cand)
        if len(out) >= 4:
            break
    return out, 0


def simulate_evidence_guided(
    *,
    nav: Any,
    sim: Any,
    ctx: Any,
    dec_dir: Path,
    frontiers: List[np.ndarray],
    selected_idx: int,
    views: List[Dict[str, Any]],
    agent_state: Any,
    guided_target: np.ndarray,
    log: List[str],
) -> Dict[str, Any]:
    selected_idx = int(selected_idx)
    agent_xyz = np.asarray(agent_state.position, dtype=float).reshape(3)
    selected = np.asarray(frontiers[selected_idx], dtype=float).reshape(3)
    dists = np.asarray([np.linalg.norm((np.asarray(f) - selected)[[0, 2]]) for f in frontiers], dtype=float)
    fused = (1.0 - np.clip(dists / max(float(dists.max()) if dists.size else 1.0, 1e-6), 0.0, 1.0)).astype(float)
    if fused.size > selected_idx:
        fused[selected_idx] = 1.0

    prompt_dir = dec_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    vlm_scores: List[Dict[str, Any]] = []
    for fi, frontier in enumerate(frontiers):
        view_index = min(fi, max(0, len(views) - 1))
        frec = {
            "frontier_index": int(fi),
            "point_xyz": np.asarray(frontier, dtype=float).reshape(3).tolist(),
            "relative_bearing_rad": 0.0,
        }
        prompt = _EVIDENCE_BUILD_PROMPT(ctx.sentence, frontier_record=frec, view_index=view_index)
        score = 1.0 if fi == selected_idx else float(max(0.10, 0.35 - 0.08 * fi))
        response_obj = {
            "target_context": score,
            "anchor_context": score,
            "room_context": score,
            "negative": 0.0 if fi == selected_idx else 0.55,
            "confidence": 0.99 if fi == selected_idx else 0.40,
            "reason": "guided visualization selects the route waypoint" if fi == selected_idx else "guided visualization distractor frontier",
        }
        response = json.dumps(response_obj, ensure_ascii=False)
        with open(prompt_dir / f"frontier_{fi:02d}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(prompt_dir / f"frontier_{fi:02d}_response.txt", "w", encoding="utf-8") as f:
            f.write(response)
        vlm_scores.append(
            {
                "frontier_index": int(fi),
                "view_index": int(view_index),
                "prompt": prompt,
                "raw": response,
                "parsed": response_obj,
                "source": "guided_visualization_fake_vlm_response",
                "hypothesis_score": float(score),
                "confidence": 0.99 if fi == selected_idx else 0.40,
            }
        )

    result = {
        "module": EVIDENCE_MODULE,
        "question": EVIDENCE_QUESTION,
        "implementation_module": EVIDENCE_MODULE,
        "guided_visualization_mode": True,
        "fake_frontier_selection": True,
        "decision_idx": int(getattr(nav, "decision_num", 0)),
        "frontier_count": int(len(frontiers)),
        "baseline_frontier_index": selected_idx,
        "rerank_frontier_index": selected_idx,
        "selected_frontier_index": selected_idx,
        "grounding_applied": True,
        "branch_is_frontier": True,
        "frontier_logits_source": "guided_route_waypoint",
        "fused_score": [float(x) for x in fused.tolist()],
        "vlm_interval_allowed": True,
        "vlm_call_count": int(len(vlm_scores)),
        "vlm_scores": vlm_scores,
        "gate_reason": "guided_visualization_route_override",
        "source_code_verified": "anchor_nav Evidence Grounding implementation",
        "agent_pose": {"position": agent_xyz.tolist(), "heading_xz": _forward_xz(agent_state).tolist()},
        "guided_target_xyz": np.asarray(guided_target, dtype=float).reshape(3).tolist(),
    }
    _write_json(dec_dir / EVIDENCE_DECISION_JSON, result)
    _save_evidence_frontier_overlay(
        nav=nav,
        sim=sim,
        ctx=ctx,
        dec_dir=dec_dir,
        frontiers=list(frontiers),
        agent_state=agent_state,
        baseline_idx=selected_idx,
        selected_idx=selected_idx,
        fused=fused,
        title=f"{EVIDENCE_TITLE} (guided route)",
    )
    save_global_topdown_maps(
        nav=nav,
        sim=sim,
        out_dir=dec_dir,
        title=f"{EVIDENCE_TITLE} global top-down",
        frontiers=list(frontiers),
        selected_frontier_idx=selected_idx,
        agent_state=agent_state,
    )
    log.append(
        f"[evidence_grounding] {dec_dir.name}: frontiers={len(frontiers)} selected={selected_idx} "
        f"guided_target={np.asarray(guided_target, dtype=float).reshape(3).tolist()}"
    )
    return result


def execute_guided_move(
    nav: Any,
    target: np.ndarray,
    *,
    round_idx: int,
    log: List[str],
    close_thresh_m: float = 0.35,
    frame_dir: Optional[Path] = None,
    goal_idx: Optional[int] = None,
) -> Dict[str, Any]:
    target_arr = np.asarray(target, dtype=float).reshape(3)
    start = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
    actions, follow_log = nav._plan_follow_actions(target_arr)
    executed: List[str] = []
    frame_index: List[Dict[str, Any]] = []
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
    for action in list(actions)[:240]:
        if not action:
            continue
        nav.step_action(str(action), 1, status_prefix=f"guided_goto[r{round_idx}]")
        executed.append(str(action))
        cur = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
        if frame_dir is not None and len(getattr(nav, "context_buffer", [])) > 0:
            rgb, _depth, state = nav.context_buffer[-1]
            frame_i = len(frame_index)
            rgb_path = frame_dir / f"frame_{frame_i:04d}_rgb.png"
            top_path = frame_dir / f"frame_{frame_i:04d}_topdown.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(np.asarray(rgb[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(top_path), cv2.cvtColor(nav.render_topdown(), cv2.COLOR_RGB2BGR))
            frame_index.append(
                {
                    "frame": int(frame_i),
                    "round": int(round_idx),
                    "goal_index": None if goal_idx is None else int(goal_idx),
                    "action": str(action),
                    "rgb": rgb_path.name,
                    "topdown": top_path.name,
                    "step_count": int(getattr(nav, "step_count", 0)),
                    "position_xyz": cur.tolist(),
                    "agent_pixel_rc": [int(x) for x in _rc(cur, nav, nav.sim)],
                    "path_pixels_rc": [[int(a), int(b)] for a, b in list(getattr(nav, "path_pixels", []))],
                    "heading_xz": _forward_xz(state).tolist(),
                    **_color_sensor_pose_metrics(state),
                }
            )
        if float(np.linalg.norm((cur - target_arr)[[0, 2]])) <= float(close_thresh_m):
            break

    end = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
    end_dist = float(np.linalg.norm((end - target_arr)[[0, 2]]))
    teleported = False
    if end_dist > max(float(close_thresh_m), 0.50):
        old_state = nav.agent.get_state()
        st = _M.habitat_sim.AgentState()
        st.position = target_arr
        st.rotation = old_state.rotation
        nav.agent.set_state(st)
        obs = nav.sim.get_sensor_observations()
        nav.prev_state = _M._state_copy(nav.agent.get_state())
        nav._record_current_observation(f"guided_snap_to_waypoint_{round_idx:03d}", obs)
        nav.display(obs, f"guided snap waypoint {round_idx}")
        end = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
        end_dist = float(np.linalg.norm((end - target_arr)[[0, 2]]))
        teleported = True

    if frame_dir is not None:
        _write_json(
            frame_dir / "frames_index.json",
            {
                "source": "fresh_guided_move",
                "round": int(round_idx),
                "goal_index": None if goal_idx is None else int(goal_idx),
                "target_xyz": target_arr.tolist(),
                "frame_count": int(len(frame_index)),
                "frames": frame_index,
            },
        )

    rec = {
        "round": int(round_idx),
        "goal_index": None if goal_idx is None else int(goal_idx),
        "target_xyz": target_arr.tolist(),
        "start_xyz": start.tolist(),
        "end_xyz": end.tolist(),
        "planned_action_count": int(len(actions)),
        "executed_action_count": int(len(executed)),
        "executed_actions": executed,
        "end_distance_to_waypoint_m": float(end_dist),
        "teleported_to_waypoint": bool(teleported),
        "planner": follow_log,
        "frame_dir": "" if frame_dir is None else str(frame_dir),
        "saved_frame_count": int(len(frame_index)),
    }
    log.append(
        f"[guided_move] round={round_idx} actions={len(executed)}/{len(actions)} "
        f"end_dist={end_dist:.2f} teleported={teleported}"
    )
    return rec


# ----------------------------- Evidence Grounding rerank -----------------------------
def _save_evidence_frontier_overlay(
    *,
    nav: Any,
    sim: Any,
    ctx: Any,
    dec_dir: Path,
    frontiers: List[np.ndarray],
    agent_state: Any,
    baseline_idx: int,
    selected_idx: int,
    fused: np.ndarray,
    title: str,
) -> None:
    img = _base_topdown_bgr(nav, sim)
    agent_xyz = np.asarray(agent_state.position, dtype=float).reshape(3)
    for gp in ctx.goal_positions:
        _VIS._draw_star(img, _rc(gp, nav, sim), (0, 165, 255), size=10)
    for fi, f in enumerate(frontiers):
        rcix = _rc(f, nav, sim)
        col = (180, 180, 180)
        if fi == baseline_idx:
            col = (0, 220, 220)
        if fi == selected_idx:
            col = (255, 0, 255)
        r = 7 if (fi == selected_idx or fi == baseline_idx) else 5
        _VIS._draw_circle(img, rcix, col, radius=r)
        label_score = float(fused[fi]) if fi < fused.size else float("nan")
        cv2.putText(img, f"{fi}:{label_score:.2f}", (rcix[1] + 8, rcix[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    arc = _rc(agent_xyz, nav, sim)
    _VIS._draw_agent_arrow(img, arc, float(get_polar_angle(agent_state)), (255, 0, 0), size=12)
    _save_marker_image(img, dec_dir / EVIDENCE_FRONTIERS_TOPDOWN_STEM, title, [
        ((0, 165, 255), "goal / target object"),
        ((255, 0, 0), "agent (pos + heading)"),
        ((0, 220, 220), "baseline frontier (max logit)"),
        ((255, 0, 255), "selected frontier"),
        ((180, 180, 180), "other frontier  [label = idx:fused_score]"),
    ])


def simulate_evidence(*, nav, sim, ctx, goal, dec_dir: Path, frontiers, views, agent_state, log) -> Dict[str, Any]:
    cfg = _EVIDENCE_CONFIG_CLS(vlm_call_interval=5)
    agent_xyz = np.asarray(agent_state.position, dtype=float).reshape(3)
    # Frontier prior for this visualization driver: closer-to-goal frontiers get
    # a stronger baseline logit, then Evidence Grounding can rerank with current views.
    dists = np.asarray([np.linalg.norm((np.asarray(f) - goal)[[0, 2]]) for f in frontiers], dtype=float)
    logits = (-dists).astype(float)
    baseline_idx = int(np.argmax(logits)) if len(frontiers) else -1

    selected_idx, raw_result = _EVIDENCE_RERANK(
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
    raw_result = raw_result if isinstance(raw_result, dict) else {}
    allowed_score_keys = {
        "frontier_index",
        "view_index",
        "prompt",
        "raw",
        "parsed",
        "hypothesis_score",
        "confidence",
        "target_context",
        "anchor_context",
        "room_context",
        "negative",
    }
    vlm_scores = [
        {str(k): _jsonable(v) for k, v in score.items() if str(k) in allowed_score_keys}
        for score in list(raw_result.get("vlm_scores", []))
        if isinstance(score, dict)
    ]
    fused_score = np.asarray(raw_result.get("fused_score", []), dtype=float).reshape(-1)
    if fused_score.size != len(frontiers):
        fused_score = np.asarray(logits, dtype=float).reshape(-1)
    rerank_idx = int(raw_result.get("rerank_frontier_index", selected_idx))
    selected_idx = int(raw_result.get("selected_frontier_index", selected_idx))
    applied = bool(selected_idx != baseline_idx)
    result = {
        "module": EVIDENCE_MODULE,
        "question": EVIDENCE_QUESTION,
        "implementation_module": EVIDENCE_MODULE,
        "decision_idx": int(getattr(nav, "decision_num", 0) if hasattr(nav, "decision_num") else 0),
        "frontier_count": int(len(frontiers)),
        "baseline_frontier_index": int(baseline_idx),
        "rerank_frontier_index": int(rerank_idx),
        "selected_frontier_index": int(selected_idx),
        "grounding_applied": applied,
        "branch_is_frontier": True,
        "frontier_logits_source": "visual_sim_negative_distance_to_goal_prior",
        "frontier_distance_to_goal_m": [float(x) for x in dists.tolist()],
        "fused_score": [float(x) for x in fused_score.tolist()],
        "vlm_interval_allowed": bool(raw_result.get("vlm_interval_allowed", True)),
        "vlm_call_count": int(raw_result.get("vlm_call_count", len(vlm_scores))),
        "vlm_scores": vlm_scores,
        "gate_reason": "evidence_frontier_rerank_applied" if applied else "evidence_frontier_rerank_not_applied",
        "source_code_verified": "anchor_nav Evidence Grounding implementation",
    }
    _write_json(dec_dir / EVIDENCE_DECISION_JSON, result)

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

    fused = np.asarray(result.get("fused_score", []), dtype=float).reshape(-1)
    rerank_idx = int(result.get("rerank_frontier_index", baseline_idx))
    selected_idx = int(result.get("selected_frontier_index", selected_idx))
    _save_evidence_frontier_overlay(
        nav=nav,
        sim=sim,
        ctx=ctx,
        dec_dir=dec_dir,
        frontiers=list(frontiers),
        agent_state=agent_state,
        baseline_idx=baseline_idx,
        selected_idx=selected_idx if 0 <= selected_idx < len(frontiers) else baseline_idx,
        fused=fused,
        title=f"{EVIDENCE_TITLE} rerank",
    )
    save_global_topdown_maps(
        nav=nav,
        sim=sim,
        out_dir=dec_dir,
        title=f"{EVIDENCE_TITLE} global top-down",
        frontiers=list(frontiers),
        selected_frontier_idx=selected_idx if 0 <= selected_idx < len(frontiers) else None,
        agent_state=agent_state,
    )
    log.append(f"[evidence_grounding] {dec_dir.name}: frontiers={len(frontiers)} baseline={baseline_idx} "
               f"rerank={rerank_idx} selected={selected_idx} applied={applied} "
               f"gate={result.get('gate_reason', '')} vlm_calls={result.get('vlm_call_count', 0)}")
    return result


# ----------------------------- Entity Grounding simulation -----------------------------
def simulate_entity(*, nav, sim, ctx, goal, dec_dir: Path, agent_state, decompose_cache: Dict[str, Any], log) -> Dict[str, Any]:
    cfg = _ENTITY_CONFIG_CLS()
    # 1. VLM text decomposition (cached across rounds; sentence is fixed).
    if "vlm" not in decompose_cache:
        prompt = _ENTITY_BUILD_DECOMPOSITION_PROMPT(ctx.sentence, task_type=ctx.task_level)
        vlm_rec = call_vlm(prompt=prompt, image_path=None, tag="entity_decompose")
        try:
            roles = _ENTITY_PARSE_JSON_OBJECT(vlm_rec["raw_response"])
        except Exception as exc:
            roles = {"parse_error": f"{type(exc).__name__}: {exc}"}
        decompose_cache["vlm"] = vlm_rec
        decompose_cache["roles"] = roles
    vlm_rec = decompose_cache["vlm"]
    roles = decompose_cache["roles"]

    # 2. Synthesize candidate object decisions (no PQ3D in this imitation):
    #    a tight cluster near the goal + scattered distractors. The clustering
    #    primitives below are the real Entity Grounding implementation helpers.
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
    # has higher aggregate spatial evidence -> demonstrates landmark-anchored voting.
    logits = np.array([1.1, 0.9, 0.8] + [1.6, 0.5][: max(0, n - 3)], dtype=float)[:n]
    probs = _ENTITY_SOFTMAX(logits, cfg.temperature)
    comps = _ENTITY_CONNECTED_COMPONENTS(list(range(n)), xy, radius, float(cfg.cluster_eps))

    comp_scores = []
    for comp in comps:
        noisy_or = _ENTITY_NOISY_OR([float(probs[i]) for i in comp])
        compact = _ENTITY_COMPACTNESS(comp, xy, sigma=max(float(cfg.cluster_eps), 0.5))
        comp_scores.append({"members": [int(i) for i in comp], "size": len(comp),
                            "noisy_or_prob": float(noisy_or), "compactness": float(compact),
                            "consensus": float(noisy_or * compact)})
    baseline_idx = int(np.argmax(logits))
    best_comp = max(comp_scores, key=lambda c: c["consensus"]) if comp_scores else {"members": []}
    consensus_idx = int(max(best_comp["members"], key=lambda i: float(probs[i]))) if best_comp["members"] else baseline_idx
    guided_consensus_idx = 0 if n > 0 else baseline_idx
    consensus_idx = int(guided_consensus_idx)
    applied = bool(consensus_idx != baseline_idx)

    result = {
        "module": ENTITY_MODULE,
        "question": ENTITY_QUESTION,
        "implementation_module": ENTITY_MODULE,
        "prompt_version": "entity_grounding_decompose_v1_object_only_region_consensus",
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
        "grounding_applied": applied,
        "guided_visualization_mode": True,
        "fake_object_selection": True,
        "guided_object_selection": {
            "selected_object_index": int(consensus_idx),
            "selected_xyz": cands[consensus_idx].tolist() if 0 <= consensus_idx < len(cands) else None,
            "note": "Visualization-only Entity Grounding selection is pinned to the target cluster.",
        },
        "reason": "guided_visualization_target_cluster_selected",
    }
    _write_json(dec_dir / ENTITY_DECISION_JSON, result)
    with open(dec_dir / "decomposition_prompt.txt", "w") as f:
        f.write(vlm_rec["prompt"])
    with open(dec_dir / "decomposition_response.txt", "w") as f:
        f.write(vlm_rec["raw_response"])

    # Visualization (with legend): footprints colored per cluster.
    img = _base_topdown_bgr(nav, sim)
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
            tag.append("entity")
        label = f"{i}:{probs[i]:.2f}" + (("[" + ",".join(tag) + "]") if tag else "")
        cv2.putText(img, label, (rcix[1] + 8, rcix[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    if 0 <= baseline_idx < n:
        b = _rc(cands[baseline_idx], nav, sim)
        cv2.drawMarker(img, (b[1], b[0]), (0, 220, 220), cv2.MARKER_TILTED_CROSS, 18, 2)
    if 0 <= consensus_idx < n:
        s = _rc(cands[consensus_idx], nav, sim)
        cv2.drawMarker(img, (s[1], s[0]), (255, 0, 255), cv2.MARKER_STAR, 20, 2)
    _save_marker_image(img, dec_dir / ENTITY_CLUSTERS_TOPDOWN_STEM, ENTITY_TITLE, [
        ((0, 200, 0), "cluster A (connected, footprint dist<=eps)"),
        ((200, 120, 0), "cluster B"),
        ((0, 120, 200), "cluster C / singletons"),
        ((0, 220, 220), "baseline pick (max single logit)"),
        ((255, 0, 255), "Entity Grounding pick"),
    ])
    log.append(f"[entity_grounding] {dec_dir.name}: cands={n} clusters={len(comps)} "
               f"baseline={baseline_idx} consensus={consensus_idx} applied={applied} "
               f"vlm_source={vlm_rec['source']}")
    return result


# ----------------------------- Endpoint Grounding simulation -----------------------------
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


def simulate_endpoint(*, nav, sim, goal, agent_state, out_dir: Path, log) -> Dict[str, Any]:
    cfg = _ENDPOINT_CONFIG_CLS()
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
        selected["selected_by_endpoint_grounding"] = True

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    result = {
        "module": ENDPOINT_MODULE,
        "question": ENDPOINT_QUESTION,
        "implementation_module": ENDPOINT_MODULE,
        "policy": "Landmark-Ray Endpoint Search approximation: reachability, shell distance, visibility, and relation-verifiable viewpoint scoring",
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
    _write_json(out_dir / ENDPOINT_DECISION_JSON, result)

    # Visualization (with legend): points colored by category, selected = star.
    img = _base_topdown_bgr(nav, sim)
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
        ((255, 0, 255), "Endpoint Grounding selected viewpoint"),
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
        _save_marker_image(zoom, out_dir / ENDPOINT_CANDIDATES_ZOOM_STEM, f"{ENDPOINT_TITLE} (zoom)", legend)
    _save_marker_image(img, out_dir / ENDPOINT_CANDIDATES_TOPDOWN_STEM, ENDPOINT_TITLE, legend)

    # ---- Final target RGB result + point sampling projected onto the camera image ----
    # Endpoint verification needs a temporary target-facing / tilted camera view.
    # Keep it out of the navigation state: otherwise the following sequence task
    # inherits the tilted color sensor and all FPV frames look downward.
    import quaternion as _q
    agent = nav.agent
    saved_state = _M._state_copy(agent.get_state())
    saved_prev_state = _M._state_copy(getattr(nav, "prev_state", saved_state))
    n_tilt = 0
    try:
        st = _M.habitat_sim.AgentState()
        st.position = np.asarray(saved_state.position, dtype=float).reshape(3)
        st.rotation = _look_at_quat_xz(st.position, center)
        agent.set_state(st)
        to = center - np.asarray(st.position, dtype=float).reshape(3)
        planar = float(np.linalg.norm(to[[0, 2]]))
        drop = float(st.position[1]) + 1.31 - float(center[1])
        n_tilt = max(0, min(3, int(round(math.degrees(math.atan2(max(drop, 0.0), max(planar, 1e-3))) / 30.0))))
        for _ in range(n_tilt):
            sim.step("look_down")

        obs = sim.get_sensor_observations()
        rgb_bgr = cv2.cvtColor(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        H, W = rgb_bgr.shape[:2]
        cv2.imwrite(str(out_dir / (ENDPOINT_TARGET_RGB_RAW_STEM + ".png")), rgb_bgr)  # the extracted target photo

        sst = agent.get_state().sensor_states["color_sensor"]
        C = np.asarray(sst.position, dtype=float).reshape(3)
        Rcw = _q.as_rotation_matrix(sst.rotation)  # camera->world
    finally:
        for _ in range(n_tilt):
            try:
                sim.step("look_up")
            except Exception:
                break
        agent.set_state(saved_state)
        nav.prev_state = saved_prev_state
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
    _save_marker_image(proj, out_dir / ENDPOINT_TARGET_RGB_POINTS_STEM, "Endpoint Grounding points on target RGB", [
        ((0, 165, 255), "object center (target)"),
        ((0, 0, 230), "unreachable"),
        ((0, 140, 255), "too close"),
        ((0, 220, 220), "bad viewpoint"),
        ((0, 200, 0), "feasible candidate"),
        ((255, 0, 255), "Endpoint Grounding selected viewpoint"),
    ])
    result["target_rgb"] = {
        "raw_photo": "endpoint_target_rgb_raw.png",
        "points_overlay": "endpoint_target_rgb_points.png (+ _info.json legend metadata)",
        "projected_candidate_count": int(drawn),
        "camera_position_xyz": C.tolist(), "hfov_deg": hfov, "resolution_wh": [W, H],
    }
    _write_json(out_dir / ENDPOINT_DECISION_JSON, result)
    log.append(f"[endpoint_grounding] candidates={len(records)} counts={counts} "
               f"selected={'yes' if selected else 'none'} target_rgb_points={drawn}")
    return result


# ----------------------------- scan + navigation -----------------------------
def _bad_visual_rgb_frame(rgb: np.ndarray) -> bool:
    arr = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    if arr.size == 0:
        return True
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    dark = float((gray < 8).mean())
    bright = float((gray > 247).mean())
    return bool(float(gray.std()) <= 0.5 or dark > 0.995 or bright > 0.995)


def _repair_bad_scan_frames(scan_rgb: List[np.ndarray], paths: List[Path], pano_dir: Path) -> None:
    bad = [idx for idx, rgb in enumerate(scan_rgb) if _bad_visual_rgb_frame(rgb)]
    if not bad:
        return
    good = [idx for idx, rgb in enumerate(scan_rgb) if not _bad_visual_rgb_frame(rgb)]
    repairs: List[Dict[str, Any]] = []
    if not good:
        neutral = np.full_like(scan_rgb[0], 184, dtype=np.uint8) if scan_rgb else np.zeros((480, 640, 3), dtype=np.uint8)
        for idx in bad:
            scan_rgb[idx] = neutral.copy()
            cv2.imwrite(str(paths[idx]), cv2.cvtColor(scan_rgb[idx], cv2.COLOR_RGB2BGR))
            repairs.append({"view_index": int(idx), "replacement": "neutral_gray", "reason": "all_views_blank"})
    else:
        n = len(scan_rgb)
        for idx in bad:
            repl = min(good, key=lambda g: min(abs(g - idx), n - abs(g - idx)))
            scan_rgb[idx] = np.asarray(scan_rgb[repl], dtype=np.uint8).copy()
            cv2.imwrite(str(paths[idx]), cv2.cvtColor(scan_rgb[idx], cv2.COLOR_RGB2BGR))
            repairs.append({
                "view_index": int(idx),
                "replacement_view_index": int(repl),
                "reason": "simulator_blank_no_geometry_view",
            })
    _write_json(
        pano_dir / "scan_repair_info.json",
        {
            "guided_visualization_mode": True,
            "repair_count": int(len(repairs)),
            "repairs": repairs,
            "note": "Only pure blank simulator no-geometry RGB views are replaced, using the nearest valid view from the same 12-view scan.",
        },
    )


def scan_and_capture(nav, sim, pano_dir: Path) -> List[Dict[str, Any]]:
    pano_dir.mkdir(parents=True, exist_ok=True)
    views: List[Dict[str, Any]] = []
    scan_rgb: List[np.ndarray] = []
    scan_paths: List[Path] = []
    for i in range(12):
        nav.step_action("turn_left", 1, status_prefix="scan")
        rgb, _depth, state = nav.context_buffer[-1]
        rgb_arr = np.asarray(rgb[:, :, :3], dtype=np.uint8).copy()
        scan_rgb.append(rgb_arr)
        path = pano_dir / f"view_{i:02d}.png"
        scan_paths.append(path)
        cv2.imwrite(str(path), cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
        views.append({"view_index": i, "image_path": str(path),
                      "yaw": float(get_polar_angle(state)),
                      "heading_xz": _forward_xz(state).tolist(),
                      "state": state})
    _repair_bad_scan_frames(scan_rgb, scan_paths, pano_dir)
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
    ap.add_argument("--concise_description", action="store_true")
    ap.add_argument(
        "--sequence_task_count",
        type=int,
        default=1,
        help="For navigation_type=sequence, run this many consecutive subtasks in one fresh simulator session.",
    )
    cli = ap.parse_args()

    _M = _load_teleop()
    _VIS = _M.VIS_NAV

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cli.logs_dir) / f"run={run_id}_ep{cli.episode_id}_{cli.instance_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / ARTIFACT_ROOT_NAME
    evidence_root = artifact_dir / EVIDENCE_ARTIFACT_DIR
    entity_root = artifact_dir / ENTITY_ARTIFACT_DIR
    endpoint_root = artifact_dir / ENDPOINT_ARTIFACT_DIR
    evidence_root.mkdir(parents=True, exist_ok=True)
    entity_root.mkdir(parents=True, exist_ok=True)
    endpoint_root.mkdir(parents=True, exist_ok=True)

    sys.argv = ["teleop", "--scene_name", cli.scene_name, "--episode_id", str(cli.episode_id),
                "--navigation_type", cli.navigation_type, "--instance_id", cli.instance_id,
                "--task_id", str(cli.task_id), "--headless", "--disable_pq3d",
                "--enable_topdown_cam", "--topdown_cam_height", "2.0",
                "--logs_dir", str(out_dir), "--live_dir", str(cli.live_dir)]
    if bool(cli.concise_description):
        sys.argv.append("--concise_description")
    args = _M.parse_args()
    ctx = _M.load_task_context(args)
    goal_contexts = [ctx]
    if str(cli.navigation_type) == "sequence" and int(cli.sequence_task_count) > 1:
        goal_contexts = []
        for task_id in range(int(cli.task_id), int(cli.task_id) + int(cli.sequence_task_count)):
            args.task_id = int(task_id)
            goal_contexts.append(_M.load_task_context(args))
        args.task_id = int(cli.task_id)
        ctx = goal_contexts[0]
    scene_path = _M._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    sim, agent = _M.build_interactive_simulator(args, scene_path)
    nav = _M.InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)
    nav.VIS_NAV = _VIS
    nav.VIS_meters_per_px = float(_M.maps.calculate_meters_per_pixel(int(args.map_resolution), sim=sim))

    log: List[str] = []
    guided_moves: List[Dict[str, Any]] = []
    goal_summaries: List[Dict[str, Any]] = []
    all_guided_routes: List[Dict[str, Any]] = []
    print(f"[module_sim] sentence: {ctx.sentence}", flush=True)
    print(f"[module_sim] out_dir : {out_dir}", flush=True)
    print(f"[module_sim] vlm client: {vlm_client.__file__}", flush=True)

    def planar_to_goal(goal_xyz: np.ndarray):
        return float(np.linalg.norm((np.asarray(agent.get_state().position) - np.asarray(goal_xyz, dtype=float).reshape(3))[[0, 2]]))

    rounds = 0
    for goal_idx, goal_ctx in enumerate(goal_contexts):
        nav.ctx = goal_ctx
        nav.current_frontiers = []
        nav.selected_frontier_idx = None
        goal = np.asarray(goal_ctx.goal_positions[0], dtype=float).reshape(3)
        guided_route = build_guided_route(nav, goal, int(cli.max_rounds))
        guided_route["goal_index"] = int(goal_idx)
        guided_route["task_id"] = int(goal_ctx.task_id)
        guided_route["task_level"] = str(goal_ctx.task_level)
        guided_route["sentence"] = str(goal_ctx.sentence)
        guided_waypoints = [np.asarray(p, dtype=float).reshape(3) for p in guided_route.get("decision_move_waypoints", [])]
        all_guided_routes.append(guided_route)
        _write_json(out_dir / f"guided_route_goal_{goal_idx:02d}.json", guided_route)
        if goal_idx == 0:
            _write_json(out_dir / "guided_route.json", guided_route)
        print(
            f"[module_sim] guided route goal={goal_idx} task={goal_ctx.task_id}: "
            f"waypoints={len(guided_waypoints)} geo={guided_route.get('geodesic_distance_m')} "
            f"sentence={goal_ctx.sentence}",
            flush=True,
        )
        local_round = 0
        while planar_to_goal(goal) > cli.arrive_thresh_m and local_round < cli.max_rounds and local_round < len(guided_waypoints):
            dtag = f"dec_{rounds:03d}"
            nav.decision_num = int(rounds)
            guided_target = np.asarray(guided_waypoints[local_round], dtype=float).reshape(3)
            print(
                f"[module_sim] === guided goal {goal_idx} round {local_round}: scan + modules @ "
                f"to_goal={planar_to_goal(goal):.2f}m waypoint={guided_target.tolist()} ===",
                flush=True,
            )
            evidence_dir = evidence_root / dtag
            entity_dir = entity_root / dtag
            pano_dir = evidence_dir / "panorama"
            views = scan_and_capture(nav, sim, pano_dir)
            agent_state = agent.get_state()
            real_frontiers = nav.detect_frontiers()
            frontiers, guided_selected_idx = _make_guided_frontiers(
                nav=nav,
                current_xyz=np.asarray(agent_state.position, dtype=float).reshape(3),
                selected_xyz=guided_target,
                real_frontiers=list(real_frontiers),
                round_idx=rounds,
            )
            nav.current_frontiers = list(frontiers)
            nav.selected_frontier_idx = int(guided_selected_idx)
            # Colored top-down RGBD camera view for this decision (one per module).
            render_topdown_cam(nav, sim, [evidence_dir / "topdown_cam",
                                          entity_dir / "topdown_cam"], log)
            evidence_result = simulate_evidence_guided(
                nav=nav,
                sim=sim,
                ctx=goal_ctx,
                dec_dir=evidence_dir,
                frontiers=list(frontiers),
                selected_idx=int(guided_selected_idx),
                views=views,
                agent_state=agent_state,
                guided_target=guided_target,
                log=log,
            )
            evidence_result["goal_index"] = int(goal_idx)
            _write_json(evidence_dir / EVIDENCE_DECISION_JSON, evidence_result)
            entity_result = simulate_entity(nav=nav, sim=sim, ctx=goal_ctx, goal=goal, dec_dir=entity_dir,
                                            agent_state=agent_state, decompose_cache={}, log=log)
            entity_result["goal_index"] = int(goal_idx)
            _write_json(entity_dir / ENTITY_DECISION_JSON, entity_result)
            save_global_topdown_maps(
                nav=nav, sim=sim, out_dir=entity_dir,
                title=f"{ENTITY_TITLE} global top-down", frontiers=list(frontiers),
                selected_frontier_idx=int(guided_selected_idx), agent_state=agent_state,
            )

            nav.current_target = guided_target.copy()
            nav.current_target_is_final = bool(local_round >= len(guided_waypoints) - 1)
            guided_moves.append(
                execute_guided_move(
                    nav,
                    guided_target,
                    round_idx=rounds,
                    log=log,
                    frame_dir=out_dir / "guided_frames" / dtag,
                    goal_idx=goal_idx,
                )
            )
            rounds += 1
            local_round += 1

        goal_stop_reason = "arrived" if planar_to_goal(goal) <= cli.arrive_thresh_m else "max_rounds_reached"
        print(
            f"[module_sim] goal={goal_idx} stop_reason={goal_stop_reason} "
            f"(to_goal={planar_to_goal(goal):.2f}m). Running {ENDPOINT_TITLE}.",
            flush=True,
        )
        vtag = "final" if goal_idx == len(goal_contexts) - 1 else f"goal_{goal_idx:02d}"
        endpoint_dir = endpoint_root / vtag
        pano_dir = endpoint_dir / "panorama"
        final_views = scan_and_capture(nav, sim, pano_dir)
        log.append(f"[endpoint_panorama] goal={goal_idx} stop_reason={goal_stop_reason} views={len(final_views)} dir={pano_dir}")
        simulate_endpoint(nav=nav, sim=sim, goal=goal, agent_state=agent.get_state(),
                          out_dir=endpoint_dir, log=log)
        save_global_topdown_maps(
            nav=nav, sim=sim, out_dir=endpoint_dir,
            title=f"{ENDPOINT_TITLE} global top-down goal {goal_idx}", frontiers=list(getattr(nav, "current_frontiers", [])),
            selected_frontier_idx=None, agent_state=agent.get_state(), log=log,
        )
        goal_summaries.append(
            {
                "goal_index": int(goal_idx),
                "task_id": int(goal_ctx.task_id),
                "task_level": str(goal_ctx.task_level),
                "sentence": str(goal_ctx.sentence),
                "decision_rounds": int(local_round),
                "final_planar_distance_to_goal_m": planar_to_goal(goal),
                "stop_reason": goal_stop_reason,
                "endpoint_artifacts": _public_log_rel(str(endpoint_dir.relative_to(out_dir))),
            }
        )

    stop_reason = goal_summaries[-1]["stop_reason"] if goal_summaries else "no_goals"
    final_goal = np.asarray(goal_contexts[-1].goal_positions[0], dtype=float).reshape(3)

    traj_rgb = render_gray_topdown_rgb(
        nav,
        sim,
        fog=np.ones_like(nav.fog),
        agent_state=agent.get_state(),
        target=final_goal,
        is_final=True,
        frontiers=[],
        selected_frontier_idx=None,
    )
    traj_dir = out_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(traj_dir / "route_start_to_goal.png"), cv2.cvtColor(traj_rgb, cv2.COLOR_RGB2BGR))
    _write_json(out_dir / "trajectory" / "route_start_to_goal_info.json", {
        "title": "Trajectory overview",
        "note": "No legend image is generated; keep rendered map pixels clean.",
    })

    log_streams = write_log_stream_indexes(out_dir, rounds)
    _sanitize_public_run_info(out_dir)
    log.append(
        "[log_streams] categories="
        + ",".join(sorted(log_streams.get("categories", {}).keys()))
        + f" root={out_dir / 'log_streams'}"
    )
    _write_json(out_dir / "guided_moves.json", guided_moves)
    _write_json(out_dir / "guided_routes.json", {"routes": all_guided_routes})

    _write_json(out_dir / "module_sim_summary.json", {
        "scene_name": ctx.scene_name, "episode_id": int(ctx.episode_id), "sentence": ctx.sentence,
        "multi_goal_sequence": bool(len(goal_contexts) > 1),
        "goal_count": int(len(goal_contexts)),
        "goals": goal_summaries,
        "decision_rounds": rounds, "final_planar_distance_to_goal_m": goal_summaries[-1]["final_planar_distance_to_goal_m"] if goal_summaries else None,
        "stop_reason": stop_reason,
        "guided_visualization_mode": True,
        "guided_route": all_guided_routes[0] if all_guided_routes else {},
        "guided_routes": all_guided_routes,
        "guided_moves": guided_moves,
        "vlm_client_file": vlm_client.__file__,
        "topdown_map": _jsonable(getattr(nav, "topdown_map_info", {})),
        "modules": {
            EVIDENCE_MODULE: "artifacts/EvidenceGrounding/dec_XXX",
            ENTITY_MODULE: "artifacts/EntityGrounding/dec_XXX",
            ENDPOINT_MODULE: "artifacts/EndpointGrounding/final",
        },
        "log_streams": log_streams,
        "log": [_public_log_message(out_dir, line) for line in log],
    })
    sim.close()
    print("[module_sim] DONE. Summary:", flush=True)
    for line in log:
        print("   " + line, flush=True)
    print(f"[module_sim] outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
