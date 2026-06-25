"""Interactive RefHM3D teleop + Vista2MQSC-style decision visualization.

This script is intentionally a hybrid:

1. Live keyboard teleop. In headless mode (the default for this project) the
   keys are read directly from the terminal (stdin), so no GUI window is
   needed -- this works over SSH. The live camera RGB view and the live
   TopDownMap are continuously written to fixed image files (see --live_dir)
   so they can be watched in any auto-refreshing image viewer.
2. A manual "decision round" key that preserves the original navigation
   scaffold: 12-view scan, frontier extraction, PQ3D decision, optional
   Vista2MQSC final-decision refinement, and structured logs.
3. Per-decision visual artifacts: top-down trajectory, frontier map,
   explored/unexplored maps, panorama frames, and frontier-facing RGB views.

Default logs go to:
  visual-scripts/navi-visual/logs

Live view images (overwritten in place, headless mode):
  <live_dir>/rgb.png        current camera RGB view (with status overlay)
  <live_dir>/topdown.png    current TopDownMap

Typical run:
  python visual-scripts/navi-visual/code/interactive_vista2mqsc_teleop.py \
      --scene_name 00844-q5QZSEeHe5g \
      --episode_id 122 \
      --navigation_type instance \
      --headless

Keys (type the letter in the terminal, no Enter needed):
  w/e      move forward 0.25m / 1.0m
  a/d      turn left / right
  s        turn around
  o/p      look up / down
  r        run one 12-view frontier + decision round and save logs
  f        auto-follow the latest decision target
  k        save a manual snapshot
  h        print controls
  q/esc    quit
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import importlib.util
import json
import math
import os
import select
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
VISUAL_SCRIPTS = PROJECT_ROOT / "visual-scripts"
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"

for _p in (PROJECT_ROOT, HM3D_ONLINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from frontier_utils import (  # noqa: E402
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VIS_NAV = _load_module_from_path("vis_nav_sample_helpers", VISUAL_SCRIPTS / "vis_nav_sample.py")
VISTA2MQSC_PATH = HM3D_ONLINE / "refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py"


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


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _save_gray(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(gray, dtype=np.uint8))


def _state_copy(state: Any) -> habitat_sim.AgentState:
    copied = habitat_sim.AgentState()
    copied.position = np.asarray(state.position, dtype=float).reshape(3)
    copied.rotation = state.rotation
    # Preserve per-sensor poses: PQ3D's decision() reads
    # agent_state.sensor_states['color_sensor'] to build the point cloud, so a
    # snapshot that drops sensor_states silently breaks the model. agent.get_state()
    # returns fresh pose objects each call, so a shallow dict copy is a safe snapshot.
    try:
        sensor_states = getattr(state, "sensor_states", None)
        if sensor_states:
            copied.sensor_states = dict(sensor_states)
    except Exception:
        pass
    return copied


def _rotation_xyzw(rotation: Any) -> Any:
    return VIS_NAV._rotation_xyzw(rotation)


def _pos_to_pixel(position: np.ndarray, top_down_map: np.ndarray, sim: Any) -> Tuple[int, int]:
    return VIS_NAV._pos_to_pixel(position, top_down_map, sim)


def _clamp_rc(rc: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    return VIS_NAV._clamp_rc(rc, shape)


def _resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    return VIS_NAV.resolve_scene_path(hm3d_root, scene_name)


@dataclass
class TaskContext:
    scene_name: str
    episode_id: int
    navigation_type: str
    task_id: int
    task_level: str
    sentence: str
    goal_category: str
    start_position: Sequence[float]
    start_rotation: Any
    cur_task: Dict[str, Any]
    target_episode: Dict[str, Any]
    goal_positions: List[np.ndarray]
    goal_object_ids: List[Any]


def _find_scene_data_file(root: Path, scene_name: str) -> Path:
    direct = root / f"{scene_name}.json.gz"
    if direct.exists():
        return direct
    found = sorted(root.rglob(f"{scene_name}.json.gz"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Scene annotation not found for {scene_name} under {root}")


def load_task_context(args: argparse.Namespace) -> TaskContext:
    nav_root = Path(os.path.expanduser(args.navigation_data_path))
    scene_gz = _find_scene_data_file(nav_root, args.scene_name)
    with gzip.open(scene_gz, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)

    region_to_annot = scene_data.get("region_annotation", {})
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    goals_map = {x["object_id"]: x for x in scene_data["goals"]}

    target_ep: Optional[Dict[str, Any]] = None
    if args.navigation_type == "sequence":
        for ep in scene_data["episode_by_sequence"]:
            if int(ep["episode_id"]) == int(args.episode_id):
                target_ep = ep
                break
        if target_ep is None:
            raise ValueError(f"sequence episode {args.episode_id} not found for {args.scene_name}")
        task_id = int(args.task_id)
        if task_id < 0 or task_id >= len(target_ep["task_sequence"]):
            raise ValueError(f"--task_id must be in [0,{len(target_ep['task_sequence']) - 1}]")
        task_level, task_idx = target_ep["task_sequence"][task_id]
        cur_task = episode_mapping[task_level][task_idx]
    else:
        for ep in episode_mapping[args.navigation_type]:
            if int(ep["episode_id"]) != int(args.episode_id):
                continue
            if args.instance_id.strip() and ep.get("instance_id") != args.instance_id.strip():
                continue
            target_ep = ep
            break
        if target_ep is None:
            guard = f" instance_id={args.instance_id}" if args.instance_id.strip() else ""
            raise ValueError(f"{args.navigation_type} episode {args.episode_id}{guard} not found for {args.scene_name}")
        task_id = int(args.task_id)
        task_level = args.navigation_type
        cur_task = target_ep

    sentence, goal_category = VIS_NAV.build_sentence(
        task_level,
        cur_task,
        all_navigation_goals_dict=goals_map,
        region_to_annot_dict=region_to_annot,
        concise_description=bool(args.concise_description),
    )
    goal_ids = list(cur_task.get("target_object_ids", []))
    goals = [goals_map[x] for x in goal_ids if x in goals_map]
    goal_positions = [
        np.asarray(g.get("position", []), dtype=float).reshape(3)
        for g in goals
        if isinstance(g, dict) and len(g.get("position", [])) >= 3
    ]

    return TaskContext(
        scene_name=args.scene_name,
        episode_id=int(args.episode_id),
        navigation_type=args.navigation_type,
        task_id=task_id,
        task_level=task_level,
        sentence=sentence,
        goal_category=goal_category,
        start_position=target_ep["start_position"],
        start_rotation=target_ep["start_rotation"],
        cur_task=cur_task,
        target_episode=target_ep,
        goal_positions=goal_positions,
        goal_object_ids=goal_ids,
    )


def _camera_spec(uuid: str, sensor_type: Any, sensor_cfg: Any) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [int(sensor_cfg["height"]), int(sensor_cfg["width"])]
    spec.position = list(sensor_cfg["position"])
    spec.hfov = float(sensor_cfg["hfov"])
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return spec


def build_interactive_simulator(args: argparse.Namespace, scene_path: str) -> Tuple[Any, Any]:
    sim_settings = OmegaConf.load(str(PROJECT_ROOT / "configs/habitat/goat_sim_config.yaml"))
    agent_settings = OmegaConf.load(str(PROJECT_ROOT / "configs/habitat/goat_agent_config.yaml"))

    if int(args.rgb_width) > 0:
        agent_settings["rgb_sensor"]["width"] = int(args.rgb_width)
        agent_settings["depth_sensor"]["width"] = int(args.rgb_width)
    if int(args.rgb_height) > 0:
        agent_settings["rgb_sensor"]["height"] = int(args.rgb_height)
        agent_settings["depth_sensor"]["height"] = int(args.rgb_height)
    if float(args.hfov) > 0:
        agent_settings["rgb_sensor"]["hfov"] = float(args.hfov)
        agent_settings["depth_sensor"]["hfov"] = float(args.hfov)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.default_agent_id = int(sim_settings.get("default_agent", 0))
    sim_cfg.allow_sliding = bool(sim_settings.get("allow_sliding", False))
    sim_cfg.random_seed = int(args.seed)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = float(agent_settings["height"])
    agent_cfg.radius = float(agent_settings["radius"])
    agent_cfg.action_space = {
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=float(agent_settings["turn_angle"]))
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=float(agent_settings["turn_angle"]))
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=float(agent_settings["tilt_angle"]))
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=float(agent_settings["tilt_angle"]))
        ),
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=float(agent_settings["step_size"]))
        ),
    }
    agent_cfg.sensor_specifications = [
        _camera_spec("color_sensor", habitat_sim.SensorType.COLOR, agent_settings["rgb_sensor"]),
        _camera_spec("depth_sensor", habitat_sim.SensorType.DEPTH, agent_settings["depth_sensor"]),
    ]
    # Optional robot-centric top-down RGBD camera: a downward-looking sensor a
    # fixed height above the agent, so we get a *colored* bird's-eye of the real
    # scene around the robot (the navmesh top-down map has no color). Height must
    # stay below the room ceiling so the slice shows the floor/furniture, not the
    # ceiling; tune via --topdown_cam_height.
    if bool(getattr(args, "enable_topdown_cam", False)):
        td_h = float(getattr(args, "topdown_cam_height", 2.0))
        td_fov = float(getattr(args, "topdown_cam_hfov", 90.0))
        td_res = int(getattr(args, "topdown_cam_res", 512))
        for uuid, stype in (("topdown_rgb", habitat_sim.SensorType.COLOR),
                            ("topdown_depth", habitat_sim.SensorType.DEPTH)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = stype
            spec.resolution = [td_res, td_res]
            spec.position = [0.0, td_h, 0.0]
            spec.orientation = [-math.pi / 2.0, 0.0, 0.0]  # pitch -90deg -> look straight down
            spec.hfov = td_fov
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            agent_cfg.sensor_specifications.append(spec)

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    nav_cfg = habitat_sim.NavMeshSettings()
    nav_cfg.set_defaults()
    nav_cfg.agent_height = float(sim_settings.get("agent_height", agent_settings["height"]))
    nav_cfg.agent_radius = float(sim_settings.get("agent_radius", agent_settings["radius"]))
    nav_cfg.agent_max_climb = float(sim_settings.get("agent_max_climb", 0.1))
    nav_cfg.cell_height = float(sim_settings.get("cell_height", 0.05))
    sim.recompute_navmesh(sim.pathfinder, nav_cfg)
    agent = sim.initialize_agent(int(sim_settings.get("default_agent", 0)))
    return sim, agent


def _frontier_visit_key(point: Sequence[float], resolution_m: float = 0.1) -> Tuple[int, int, int]:
    arr = np.asarray(point, dtype=float).reshape(3)
    return tuple(int(round(float(x) / float(resolution_m))) for x in arr)


def _sample_context(buffer: Deque[Tuple[np.ndarray, np.ndarray, Any]], max_frames: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
    items = list(buffer)
    if int(max_frames) <= 0 or len(items) == 0:
        return [], [], []
    if len(items) > int(max_frames):
        idxs = sorted({int(round(x)) for x in np.linspace(0, len(items) - 1, int(max_frames))})
        items = [items[i] for i in idxs]
    rgb = [x[0] for x in items]
    depth = [x[1] for x in items]
    states = [x[2] for x in items]
    return rgb, depth, states


def save_exploration_maps(dec_dir: Path, top_down_map: np.ndarray, fog: np.ndarray) -> None:
    navigable = np.asarray(top_down_map) > 0
    explored = np.logical_and(navigable, np.asarray(fog) > 0)
    unexplored = np.logical_and(navigable, np.asarray(fog) == 0)

    _save_gray(dec_dir / "explored_map.png", explored.astype(np.uint8) * 255)
    _save_gray(dec_dir / "unexplored_map.png", unexplored.astype(np.uint8) * 255)

    rgb = np.zeros((*top_down_map.shape, 3), dtype=np.uint8)
    rgb[~navigable] = (20, 20, 20)
    rgb[unexplored] = (70, 70, 70)
    rgb[explored] = (210, 210, 210)
    _save_rgb(dec_dir / "explored_unexplored_map.png", rgb)


def _nearest_component_label(labels: np.ndarray, start_rc: Tuple[int, int]) -> int:
    ys, xs = np.where(labels > 0)
    if ys.size == 0:
        return 0
    sr, sc = int(start_rc[0]), int(start_rc[1])
    j = int(np.argmin((ys - sr) ** 2 + (xs - sc) ** 2))
    return int(labels[int(ys[j]), int(xs[j])])


def _clean_topdown_floor_mask(
    raw_view: np.ndarray,
    *,
    start_rc: Tuple[int, int],
    mpp: float,
    agent_radius_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove raster slivers and keep the start-floor connected component."""
    raw_mask = np.asarray(raw_view) > 0
    raw_area = int(raw_mask.sum())
    if raw_area <= 0:
        return np.zeros_like(raw_view, dtype=np.uint8), {
            "raw_area_px": 0,
            "clean_area_px": 0,
            "component_label": 0,
            "morph_radius_px": 0,
            "reason": "empty_raw",
        }

    # Top-down rasterization can leave one-pixel bridges that the agent cannot
    # actually traverse. Opening by roughly the agent radius removes passages
    # narrower than the physical body, then dilation restores normal room area.
    erode_px = max(1, int(round(float(agent_radius_m) / max(float(mpp), 1e-6))))
    best_mask: Optional[np.ndarray] = None
    best_info: Dict[str, Any] = {}
    for radius_px in (erode_px, max(1, erode_px // 2), 1, 0):
        if radius_px > 0:
            k = 2 * int(radius_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            opened = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        else:
            opened = raw_mask
        n_labels, labels = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            continue
        sr = max(0, min(int(start_rc[0]), labels.shape[0] - 1))
        sc = max(0, min(int(start_rc[1]), labels.shape[1] - 1))
        label = int(labels[sr, sc])
        if label == 0:
            label = _nearest_component_label(labels, (sr, sc))
        comp = labels == int(label)
        comp_area = int(comp.sum())
        if comp_area <= 0:
            continue
        best_mask = comp
        best_info = {
            "raw_area_px": raw_area,
            "clean_area_px": comp_area,
            "component_label": int(label),
            "connected_component_count": int(n_labels - 1),
            "morph_radius_px": int(radius_px),
            "morph_radius_m": float(radius_px) * float(mpp),
            "reason": "start_component_after_width_filter",
        }
        # If the full-radius opening preserved a plausible amount, keep it.
        if comp_area >= max(64, int(0.20 * raw_area)):
            break

    if best_mask is None:
        return raw_mask.astype(np.uint8), {
            "raw_area_px": raw_area,
            "clean_area_px": raw_area,
            "component_label": 0,
            "morph_radius_px": 0,
            "reason": "component_filter_failed_raw_fallback",
        }
    return best_mask.astype(np.uint8), best_info


def build_floor_topdown_map(
    sim: Any,
    agent: Any,
    map_resolution: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Build the current-floor navmap for fog/frontier rendering.

    Match habitat_teleop_cn_explained.py:get_topdown_map_visualize exactly in
    the behavior that matters for visualization: slice the pathfinder at the
    current agent/floor y and use that raw top-down view as the displayed map.
    """
    pf = sim.pathfinder
    mpp = maps.calculate_meters_per_pixel(int(map_resolution), sim=sim)
    agent_y = float(agent.get_state().position[1])
    agent_radius_m = 0.17
    try:
        agent_radius_m = float(agent.agent_config.radius)
    except Exception:
        pass

    raw = np.ascontiguousarray(pf.get_topdown_view(mpp, agent_y), dtype=np.uint8)
    # Compute the start pixel from the ACTUAL view shape (it is non-square, e.g.
    # 512x577). Using a square (map_resolution, map_resolution) placeholder here
    # mis-scales the column by the x/z aspect ratio, anchoring the connected-
    # component diagnostic on the wrong cell.
    start_rc = _pos_to_pixel(np.asarray(agent.get_state().position, dtype=float), raw, sim)
    raw_area = int((raw > 0).sum())
    if raw_area <= 0:
        fallback = np.ascontiguousarray(
            maps.get_topdown_map_from_sim(sim, map_resolution=int(map_resolution), draw_border=False),
            dtype=np.uint8,
        )
        return fallback, agent_y, {
            "strategy": "fallback_standard_topdown",
            "mpp": float(mpp),
            "agent_y": float(agent_y),
            "agent_radius_m": float(agent_radius_m),
            "candidates": [],
        }

    clean, clean_info = _clean_topdown_floor_mask(
        raw,
        start_rc=start_rc,
        mpp=float(mpp),
        agent_radius_m=float(agent_radius_m),
    )
    info = {
        "strategy": "reference_agent_current_height_raw_pathfinder_view",
        "mpp": float(mpp),
        "agent_y": float(agent_y),
        "agent_radius_m": float(agent_radius_m),
        "chosen_height_m": float(agent_y),
        "chosen_delta_from_agent_y_m": 0.0,
        "raw_area_px": raw_area,
        "chosen_area_px": raw_area,
        "raw_shape": list(raw.shape),
        "start_rc": [int(start_rc[0]), int(start_rc[1])],
        "diagnostic_start_component_filter_not_applied": _jsonable(clean_info),
        "reference": "visual-scripts/habitat_teleop_cn_explained.py:get_topdown_map_visualize uses pathfinder.get_topdown_view(meters_per_pixel, state.position[1])",
    }
    return np.ascontiguousarray(raw, dtype=np.uint8), float(agent_y), info


class InteractiveNavigator:
    def __init__(self, args: argparse.Namespace, ctx: TaskContext, sim: Any, agent: Any, scene_path: str, out_dir: Path) -> None:
        self.args = args
        self.ctx = ctx
        self.sim = sim
        self.agent = agent
        self.scene_path = scene_path
        self.out_dir = out_dir
        self.path_finder = sim.pathfinder

        live_dir = getattr(args, "live_dir", "") or str(out_dir / "live")
        self.live_dir = Path(os.path.expanduser(live_dir))
        self.live_dir.mkdir(parents=True, exist_ok=True)

        # Place the agent at the episode start BEFORE building the top-down map:
        # get_topdown_map_from_sim slices the navmesh at the *current* agent
        # height, and the agent spawns on a default (often different) floor. If
        # the map is built first, the start ends up on a non-navigable cell,
        # which makes map_coors_to_pixel land on an obstacle and reveal_fog_of_war
        # reveal nothing (fog stays empty). The original pipeline sets the start
        # first for exactly this reason.
        state = habitat_sim.AgentState()
        state.position = list(ctx.start_position)
        state.rotation = ctx.start_rotation
        self.agent.set_state(state)

        self.top_down_map, self.topdown_slice_height, self.topdown_map_info = build_floor_topdown_map(
            sim, self.agent, int(args.map_resolution)
        )
        self.fog = np.zeros_like(self.top_down_map)
        self.area_thres_px = convert_meters_to_pixel(float(args.frontier_area_m2), int(args.map_resolution), sim)
        self.vis_dist_px = convert_meters_to_pixel(float(args.visible_radius), int(args.map_resolution), sim)

        self.step_count = 0
        self.decision_num = 0
        self.episode_cum_distance = 0.0
        self.prev_state = _state_copy(self.agent.get_state())
        self.start_position = np.asarray(self.prev_state.position, dtype=float).reshape(3).copy()

        start_rc = _pos_to_pixel(self.start_position, self.top_down_map, self.sim)
        self.path_pixels: List[Tuple[int, int]] = [start_rc]
        self.decision_pixels: List[Tuple[int, int]] = []
        self.final_decision_pixels: List[Tuple[int, int]] = []
        self.current_frontiers: List[np.ndarray] = []
        self.current_target: Optional[np.ndarray] = None
        self.current_target_is_final = False
        self.selected_frontier_idx: Optional[int] = None
        self.visited_frontiers: Set[Tuple[int, int, int]] = set()
        self.context_buffer: Deque[Tuple[np.ndarray, np.ndarray, Any]] = deque(maxlen=int(args.context_buffer_size))

        self.pq3d_model: Optional[Any] = None
        self.vista2mqsc: Optional[Any] = None
        self.latest_decision_payload: Optional[Dict[str, Any]] = None
        self.latest_decision_path: Optional[Path] = None
        self.latest_decision_dir: Optional[Path] = None

        self._record_current_observation("init")
        self._write_run_info()

    def _write_run_info(self) -> None:
        _write_json(
            self.out_dir / "run_info.json",
            {
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "scene_path": self.scene_path,
                "args": vars(self.args),
                "topdown_map": _jsonable(getattr(self, "topdown_map_info", {})),
                "task": {
                    "scene_name": self.ctx.scene_name,
                    "episode_id": self.ctx.episode_id,
                    "navigation_type": self.ctx.navigation_type,
                    "task_id": self.ctx.task_id,
                    "task_level": self.ctx.task_level,
                    "sentence": self.ctx.sentence,
                    "goal_category": self.ctx.goal_category,
                    "goal_object_ids": self.ctx.goal_object_ids,
                    "goal_positions": [x.tolist() for x in self.ctx.goal_positions],
                },
            },
        )

    def ensure_pq3d(self) -> Any:
        if bool(self.args.disable_pq3d):
            return None
        if self.pq3d_model is None:
            print("[nav-visual] Loading PQ3DModel ...", flush=True)
            from data_utils import PQ3DModel

            self.pq3d_model = PQ3DModel(
                os.path.expanduser(self.args.pq3d_stage1_path),
                os.path.expanduser(self.args.pq3d_stage2_path),
                min_decision_num=int(self.args.decision_num_min),
            )
            self.pq3d_model.reset()
            if hasattr(self.pq3d_model, "mask_generator"):
                self.pq3d_model.mask_generator = VIS_NAV._DropTaskLevelMaskGenerator(self.pq3d_model.mask_generator)
            print("[nav-visual] PQ3DModel ready.", flush=True)
        return self.pq3d_model

    def ensure_vista2mqsc(self) -> Any:
        if self.vista2mqsc is None:
            print("[nav-visual] Loading Vista2MQSC refine hook ...", flush=True)
            self.vista2mqsc = _load_module_from_path("vista2mqsc_refine1_interactive", VISTA2MQSC_PATH)
            levels = {x.strip() for x in str(self.args.vistals_apply_task_levels).split(",") if x.strip()}
            if levels:
                self.vista2mqsc.VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS = levels
            if hasattr(self.vista2mqsc, "MqscR1Config"):
                self.vista2mqsc.MQSC_R1_CFG = self.vista2mqsc.MqscR1Config(
                    use_vlm=bool(self.args.mqsc_use_vlm),
                    vlm_model=str(self.args.mqsc_vlm_model),
                    write_debug_json=True,
                )
            print("[nav-visual] Vista2MQSC refine hook ready.", flush=True)
        return self.vista2mqsc

    def _record_current_observation(self, event: str, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if obs is None:
            obs = self.sim.get_sensor_observations()
        state = self.agent.get_state()
        rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy()
        depth = np.asarray(obs["depth_sensor"][:, :], dtype=np.float32).copy()

        self.fog = reveal_fog_of_war(
            top_down_map=self.top_down_map,
            current_fog_of_war_mask=self.fog,
            current_point=map_coors_to_pixel(state.position, self.top_down_map, self.sim),
            current_angle=get_polar_angle(state),
            fov=float(self.args.hfov if self.args.hfov > 0 else 42.0),
            max_line_len=self.vis_dist_px,
            enable_debug_visualization=False,
        )

        rc = _pos_to_pixel(np.asarray(state.position, dtype=float), self.top_down_map, self.sim)
        if len(self.path_pixels) == 0 or rc != self.path_pixels[-1]:
            self.path_pixels.append(rc)
        self.context_buffer.append((rgb, depth, _state_copy(state)))
        return {
            "event": event,
            "step_count": int(self.step_count),
            "position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
            "rotation_xyzw": _rotation_xyzw(state.rotation),
        }

    def render_topdown(self) -> np.ndarray:
        rgb = VIS_NAV._base_rgb_floor(self.top_down_map, self.fog)
        shape = rgb.shape[:2]

        for prev, nxt in zip(self.path_pixels[:-1], self.path_pixels[1:]):
            cv2.line(rgb, (prev[1], prev[0]), (nxt[1], nxt[0]), VIS_NAV.CLR_PATH, 2)

        for gp in self.ctx.goal_positions:
            gp_rc = _clamp_rc(_pos_to_pixel(np.asarray(gp, dtype=float), self.top_down_map, self.sim), shape)
            VIS_NAV._draw_star(rgb, gp_rc, VIS_NAV.CLR_GOAL, size=9)

        for idx, fw in enumerate(self.current_frontiers):
            fw_rc = _clamp_rc(_pos_to_pixel(np.asarray(fw, dtype=float), self.top_down_map, self.sim), shape)
            radius = 6 if self.selected_frontier_idx == idx else 4
            VIS_NAV._draw_circle(rgb, fw_rc, VIS_NAV.CLR_FRONTIER, radius=radius)

        if self.current_target is not None:
            t_rc = _clamp_rc(_pos_to_pixel(self.current_target, self.top_down_map, self.sim), shape)
            color = VIS_NAV.CLR_FINAL_DEC if self.current_target_is_final else VIS_NAV.CLR_DECISION
            cv2.drawMarker(rgb, (t_rc[1], t_rc[0]), color, cv2.MARKER_CROSS, 18, 2)

        state = self.agent.get_state()
        agent_rc = _clamp_rc(_pos_to_pixel(np.asarray(state.position, dtype=float), self.top_down_map, self.sim), shape)
        VIS_NAV._draw_agent_arrow(rgb, agent_rc, float(get_polar_angle(state)), VIS_NAV.CLR_AGENT, size=11)
        return rgb

    def _compose_frames(self, obs: Dict[str, Any], status: str = "") -> Tuple[np.ndarray, np.ndarray]:
        rgb_bgr = cv2.cvtColor(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        top_bgr = cv2.cvtColor(self.render_topdown(), cv2.COLOR_RGB2BGR)

        lines = [
            f"scene={self.ctx.scene_name} ep={self.ctx.episode_id} task={self.ctx.task_id} {self.ctx.task_level}",
            f"dec={self.decision_num} steps={self.step_count} frontiers={len(self.current_frontiers)}",
            status or "w/e/a/d/s/o/p teleop | r decision | f follow | k snapshot | h help | q quit",
        ]
        y = 24
        for line in lines:
            cv2.putText(rgb_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rgb_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return rgb_bgr, top_bgr

    def _atomic_imwrite(self, path: Path, img: np.ndarray) -> None:
        # Write to a temp file then rename so a watching viewer never reads a
        # half-written frame. Keep the original extension on the temp file so
        # cv2.imwrite can infer the encoder.
        tmp = path.with_suffix(".tmp" + path.suffix)
        cv2.imwrite(str(tmp), img)
        os.replace(str(tmp), str(path))

    def _write_live_frames(self, rgb_bgr: np.ndarray, top_bgr: np.ndarray) -> None:
        self._atomic_imwrite(self.live_dir / "rgb.png", rgb_bgr)
        self._atomic_imwrite(self.live_dir / "topdown.png", top_bgr)

    def display(self, obs: Optional[Dict[str, Any]] = None, status: str = "") -> None:
        if obs is None:
            obs = self.sim.get_sensor_observations()
        rgb_bgr, top_bgr = self._compose_frames(obs, status)
        if bool(self.args.headless):
            self._write_live_frames(rgb_bgr, top_bgr)
        else:
            cv2.imshow("Habitat Teleop RGB", rgb_bgr)
            cv2.imshow("Navi Visual Topdown", top_bgr)

    def step_action(self, action: str, repeat: int = 1, status_prefix: str = "teleop") -> None:
        for _ in range(int(repeat)):
            obs = self.sim.step(action=action)
            state_now = self.agent.get_state()
            self.episode_cum_distance += float(np.linalg.norm(state_now.position - self.prev_state.position))
            self.prev_state = _state_copy(state_now)
            self.step_count += 1
            self._record_current_observation(action, obs)
            self.display(obs, f"{status_prefix}: {action}")
            if not bool(self.args.headless):
                cv2.waitKey(1)

    def detect_frontiers(self) -> List[np.ndarray]:
        state = self.agent.get_state()
        raw = detect_frontier_waypoints(
            self.top_down_map,
            self.fog,
            self.area_thres_px,
            xy=map_coors_to_pixel(state.position, self.top_down_map, self.sim)[::-1],
            enable_visualization=False,
        )
        if len(raw) > 0:
            frontiers = list(pixel_to_map_coors(raw[:, ::-1], state.position, self.top_down_map, self.sim))
        else:
            frontiers = []
        filtered: List[np.ndarray] = []
        for fw in frontiers:
            key = _frontier_visit_key(fw)
            if key not in self.visited_frontiers:
                filtered.append(np.asarray(fw, dtype=float).reshape(3))
        self.current_frontiers = filtered
        return filtered

    def _capture_scan_frames(self, dec_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        scan_rgb: List[np.ndarray] = []
        scan_depth: List[np.ndarray] = []
        scan_states: List[Any] = []
        for view_idx in range(12):
            obs = self.sim.step(action="turn_left")
            self.step_count += 1
            state = self.agent.get_state()
            rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy()
            dep = np.asarray(obs["depth_sensor"][:, :], dtype=np.float32).copy()
            scan_rgb.append(rgb)
            scan_depth.append(dep)
            scan_states.append(_state_copy(state))
            self._record_current_observation(f"scan_view_{view_idx:02d}", obs)
            self.display(obs, f"decision scan {view_idx + 1}/12")
            if not bool(self.args.headless):
                cv2.waitKey(max(1, int(self.args.scan_wait_ms)))

        VIS_NAV.save_panorama_frames(dec_dir, scan_rgb, scan_depth)
        return scan_rgb, scan_depth, scan_states

    def _fallback_decision(self, frontiers: List[np.ndarray]) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        if len(frontiers) > 0:
            agent_pos = np.asarray(self.agent.get_state().position, dtype=float).reshape(3)
            dists = [float(np.linalg.norm(np.asarray(fw)[[0, 2]] - agent_pos[[0, 2]])) for fw in frontiers]
            idx = int(np.argmin(dists))
            return np.asarray(frontiers[idx], dtype=float).reshape(3), False, {
                "fallback": "nearest_frontier",
                "selected_frontier_idx": idx,
                "reason": "PQ3D disabled or unavailable",
            }
        return np.asarray(self.agent.get_state().position, dtype=float).reshape(3), True, {
            "fallback": "current_position",
            "reason": "no frontier and PQ3D disabled or unavailable",
        }

    def run_decision_round(self) -> None:
        dec_num = int(self.decision_num)
        dec_dir = self.out_dir / "decisions" / f"dec_{dec_num:03d}"
        dec_dir.mkdir(parents=True, exist_ok=True)
        print(f"[nav-visual] decision {dec_num:03d}: scan/frontier/decision start", flush=True)

        context_rgb, context_depth, context_states = _sample_context(self.context_buffer, int(self.args.max_context_frames))
        scan_rgb, scan_depth, scan_states = self._capture_scan_frames(dec_dir)

        color_list = list(context_rgb) + scan_rgb
        depth_list = list(context_depth) + scan_depth
        state_list = list(context_states) + scan_states

        frontiers = self.detect_frontiers()
        t_pq = time.perf_counter()
        pq3d_aux: Dict[str, Any] = {}
        try:
            pq3d = self.ensure_pq3d()
            if pq3d is None:
                target, is_final, pq3d_aux = self._fallback_decision(frontiers)
            else:
                target, is_final = pq3d.decision(
                    color_list,
                    depth_list,
                    state_list,
                    frontiers,
                    self.ctx.sentence,
                    dec_num,
                    task_level=self.ctx.task_level,
                )
                pq3d_aux = getattr(pq3d, "last_decision_aux", {}) or {}
        except TypeError:
            pq3d = self.ensure_pq3d()
            target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, self.ctx.sentence, dec_num)
            pq3d_aux = getattr(pq3d, "last_decision_aux", {}) or {}
        except Exception as exc:
            print(f"[nav-visual] PQ3D decision failed, fallback to frontier: {type(exc).__name__}: {exc}", flush=True)
            target, is_final, pq3d_aux = self._fallback_decision(frontiers)
            pq3d_aux["pq3d_error_type"] = type(exc).__name__
            pq3d_aux["pq3d_error_message"] = str(exc)
        pq_ms = (time.perf_counter() - t_pq) * 1000.0

        used_target = np.asarray(target, dtype=float).reshape(3).copy()
        module_info: Dict[str, Any] = {
            "module": "vista2mqsc",
            "called": False,
            "applied": False,
            "reason": "disabled_or_non_final",
            "target_before": used_target.tolist(),
            "target_after": used_target.tolist(),
        }
        if bool(is_final) and bool(self.args.enable_vista2mqsc_refine) and self.pq3d_model is not None:
            try:
                hook = self.ensure_vista2mqsc()
                used_target, module_info = hook.vista2mqsc_refine_hook(
                    sentence=self.ctx.sentence,
                    task_type=self.ctx.task_level,
                    scene_name=self.ctx.scene_name,
                    episode_id=int(self.ctx.episode_id),
                    task_id=int(self.ctx.task_id),
                    decision_num=dec_num,
                    is_final=True,
                    pq3d_model=self.pq3d_model,
                    target_position=used_target,
                    decision_aux=pq3d_aux,
                    output_dir=dec_dir,
                    path_finder=self.path_finder,
                    agent_position_xyz=np.asarray(self.agent.get_state().position, dtype=float).reshape(3),
                )
            except Exception as exc:
                module_info = {
                    "module": "vista2mqsc",
                    "called": True,
                    "applied": False,
                    "reason": "hook_error_baseline_target_kept",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "target_before": np.asarray(target, dtype=float).reshape(3).tolist(),
                    "target_after": used_target.tolist(),
                }
                _write_json(dec_dir / "vista2mqsc_error.json", module_info)
                print(f"[nav-visual] Vista2MQSC hook failed: {type(exc).__name__}: {exc}", flush=True)

        self.current_target = np.asarray(used_target, dtype=float).reshape(3)
        self.current_target_is_final = bool(is_final)
        self.selected_frontier_idx = VIS_NAV._nearest_frontier_index(self.current_target, frontiers)

        target_rc = _pos_to_pixel(self.current_target, self.top_down_map, self.sim)
        if bool(is_final):
            self.final_decision_pixels.append(target_rc)
        else:
            self.decision_pixels.append(target_rc)
            self.visited_frontiers.add(_frontier_visit_key(self.current_target))

        state = self.agent.get_state()
        VIS_NAV.save_topdown_map(
            out_dir=self.out_dir,
            dec_num=dec_num,
            top_down_map=self.top_down_map,
            fog=self.fog.copy(),
            agent_state=state,
            target=self.current_target,
            is_final=bool(is_final),
            path_pixels=list(self.path_pixels),
            sim=self.sim,
        )
        VIS_NAV.save_rgb_floor(
            out_dir=self.out_dir,
            dec_num=dec_num,
            top_down_map=self.top_down_map,
            fog=self.fog.copy(),
            agent_state=state,
            target=self.current_target,
            is_final=bool(is_final),
            path_pixels=list(self.path_pixels),
            sim=self.sim,
        )
        VIS_NAV.save_frontiers(
            out_dir=self.out_dir,
            dec_num=dec_num,
            top_down_map=self.top_down_map,
            fog=self.fog.copy(),
            agent_state=state,
            target=self.current_target,
            is_final=bool(is_final),
            path_pixels=list(self.path_pixels),
            frontier_waypoints=frontiers,
            sim=self.sim,
        )
        VIS_NAV.save_decision_maps(
            dec_dir=dec_dir,
            top_down_map=self.top_down_map,
            fog=self.fog.copy(),
            agent_state=state,
            target=self.current_target,
            is_final=bool(is_final),
            path_pixels=list(self.path_pixels),
            frontier_waypoints=frontiers,
            selected_frontier_idx=self.selected_frontier_idx,
            goal_positions=self.ctx.goal_positions,
            sim=self.sim,
        )
        save_exploration_maps(dec_dir, self.top_down_map, self.fog.copy())
        facing_records = VIS_NAV.save_frontier_facing_views(
            dec_dir=dec_dir,
            sim=self.sim,
            agent=self.agent,
            agent_state=state,
            target=self.current_target,
            frontier_waypoints=frontiers,
            selected_frontier_idx=self.selected_frontier_idx,
            max_frontiers=int(self.args.max_frontier_facing),
        )

        payload = {
            "scene_name": self.ctx.scene_name,
            "episode_id": int(self.ctx.episode_id),
            "navigation_type": self.ctx.navigation_type,
            "task_id": int(self.ctx.task_id),
            "task_level": self.ctx.task_level,
            "sentence": self.ctx.sentence,
            "decision_num": dec_num,
            "is_final": bool(is_final),
            "step_count_after_scan": int(self.step_count),
            "pq3d_ms": float(pq_ms),
            "context_frame_count": int(len(context_rgb)),
            "scan_frame_count": int(len(scan_rgb)),
            "agent_position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
            "agent_rotation_xyzw": _rotation_xyzw(state.rotation),
            "target_before_refine": np.asarray(target, dtype=float).reshape(3).tolist(),
            "target_used": self.current_target.tolist(),
            "selected_frontier_idx": self.selected_frontier_idx,
            "frontiers": [
                {
                    "index": int(i),
                    "position": np.asarray(fw, dtype=float).reshape(3).tolist(),
                    "is_selected": bool(self.selected_frontier_idx == i),
                    "visited_key": list(_frontier_visit_key(fw)),
                }
                for i, fw in enumerate(frontiers)
            ],
            "frontier_facing_views": facing_records,
            "visited_frontier_count": int(len(self.visited_frontiers)),
            "goal_positions": [x.tolist() for x in self.ctx.goal_positions],
            "goal_object_ids": list(self.ctx.goal_object_ids),
            "pq3d_aux": pq3d_aux,
            "module_info": module_info,
            "outputs": {
                "decision_dir": str(dec_dir.relative_to(self.out_dir)),
                "panorama_dir": "panorama_12_frames",
                "topdown_fog": "topdown_fog.png",
                "topdown_full": "topdown_full.png",
                "frontiers_on_topdown": "frontiers_on_topdown.png",
                "explored_map": "explored_map.png",
                "unexplored_map": "unexplored_map.png",
                "explored_unexplored_map": "explored_unexplored_map.png",
                "decision_target_facing_rgb": "decision_target_facing_rgb.png",
            },
        }
        self.latest_decision_payload = payload
        self.latest_decision_dir = dec_dir
        self.latest_decision_path = dec_dir / "decision.json"
        _write_json(self.latest_decision_path, payload)
        self.save_trajectory_snapshot("trajectory_latest.png")

        print(
            f"[nav-visual] decision {dec_num:03d}: final={bool(is_final)} "
            f"frontiers={len(frontiers)} target={self.current_target.tolist()} log={self.latest_decision_path}",
            flush=True,
        )
        self.decision_num += 1

    def _plan_follow_actions(self, target: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        start_position = np.asarray(self.agent.get_state().position, dtype=float).reshape(3)
        agent_island = int(self.path_finder.get_island(start_position))
        try:
            module = self.ensure_vista2mqsc() if bool(self.args.use_vista2mqsc_follower) else None
            if module is not None and hasattr(module, "_find_follow_actions_with_repair"):
                actions, chosen, attempts = module._find_follow_actions_with_repair(
                    path_finder=self.path_finder,
                    agent=self.agent,
                    raw_target=np.asarray(target, dtype=float).reshape(3),
                    start_position=start_position,
                    agent_island=agent_island,
                )
                return list(actions), {
                    "ok": bool(chosen.get("ok", False)),
                    "planner": "vista2mqsc_repair",
                    "chosen": chosen,
                    "candidate_attempt_count": int(len(attempts)),
                    "candidate_attempts_head": attempts[:12],
                }
        except Exception as exc:
            return [], {
                "ok": False,
                "planner": "vista2mqsc_repair",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        actions = VIS_NAV._find_follow_actions(
            path_finder=self.path_finder,
            agent=self.agent,
            raw_target=np.asarray(target, dtype=float).reshape(3),
            start_position=start_position,
            agent_island=agent_island,
        )
        return list(actions), {"ok": True, "planner": "vis_nav_sample_follower", "action_count": len(actions)}

    def follow_latest_target(self) -> None:
        if self.current_target is None:
            print("[nav-visual] No decision target yet. Press r first.", flush=True)
            return
        actions, follow_log = self._plan_follow_actions(self.current_target)
        if not actions:
            print(f"[nav-visual] No follow actions: {follow_log}", flush=True)
            if self.latest_decision_payload is not None and self.latest_decision_path is not None:
                self.latest_decision_payload["follow"] = follow_log
                _write_json(self.latest_decision_path, self.latest_decision_payload)
            return

        follow_rgb: List[np.ndarray] = []
        executed: List[Any] = []
        for action in actions:
            if not action:
                continue
            obs = self.sim.step(action=action)
            executed.append(action)
            follow_rgb.append(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy())
            state_now = self.agent.get_state()
            self.episode_cum_distance += float(np.linalg.norm(state_now.position - self.prev_state.position))
            self.prev_state = _state_copy(state_now)
            self.step_count += 1
            self._record_current_observation(f"follow_{action}", obs)
            self.display(obs, f"follow target: {action}")
            if not bool(self.args.headless):
                cv2.waitKey(max(1, int(self.args.follow_wait_ms)))
            if int(self.args.max_follow_actions) > 0 and len(executed) >= int(self.args.max_follow_actions):
                follow_log["truncated_by_max_follow_actions"] = True
                break

        follow_log.update(
            {
                "ok": bool(follow_log.get("ok", True)),
                "executed_action_count": int(len(executed)),
                "executed_actions": [str(x) for x in executed],
                "end_position": np.asarray(self.agent.get_state().position, dtype=float).reshape(3).tolist(),
                "step_count_after_follow": int(self.step_count),
                "episode_cum_distance": float(self.episode_cum_distance),
            }
        )
        if self.latest_decision_dir is not None:
            saved = VIS_NAV.save_follow_frames(
                self.latest_decision_dir,
                follow_rgb,
                max_saved=int(self.args.max_saved_follow_frames),
            )
            follow_log["saved_rgb_indices"] = [int(x) for x in saved]
        if self.latest_decision_payload is not None and self.latest_decision_path is not None:
            self.latest_decision_payload["follow"] = follow_log
            _write_json(self.latest_decision_path, self.latest_decision_payload)
        self.save_trajectory_snapshot("trajectory_latest.png")
        print(f"[nav-visual] Follow done actions={len(executed)}", flush=True)

    def save_manual_snapshot(self) -> None:
        snap_id = int(time.time() * 1000)
        out = self.out_dir / "manual_snapshots" / f"snapshot_{snap_id}"
        out.mkdir(parents=True, exist_ok=True)
        obs = self.sim.get_sensor_observations()
        state = self.agent.get_state()
        _save_rgb(out / "rgb.png", np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8))
        depth_rgb = VIS_NAV._depth_to_rgb(np.asarray(obs["depth_sensor"][:, :], dtype=np.float32))
        _save_rgb(out / "depth.png", depth_rgb)
        _save_rgb(out / "topdown.png", self.render_topdown())
        save_exploration_maps(out, self.top_down_map, self.fog.copy())
        _write_json(
            out / "snapshot.json",
            {
                "step_count": int(self.step_count),
                "decision_num_next": int(self.decision_num),
                "position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
                "rotation_xyzw": _rotation_xyzw(state.rotation),
                "frontier_count": int(len(self.current_frontiers)),
                "target": None if self.current_target is None else self.current_target.tolist(),
            },
        )
        print(f"[nav-visual] Snapshot saved: {out}", flush=True)

    def save_trajectory_snapshot(self, name: str = "trajectory_latest.png") -> None:
        rgb = VIS_NAV._base_rgb_floor(self.top_down_map, np.ones_like(self.fog))
        shape = rgb.shape[:2]
        for prev, nxt in zip(self.path_pixels[:-1], self.path_pixels[1:]):
            cv2.line(rgb, (prev[1], prev[0]), (nxt[1], nxt[0]), VIS_NAV.CLR_PATH, 2)
        for rc in self.decision_pixels:
            VIS_NAV._draw_circle(rgb, _clamp_rc(rc, shape), VIS_NAV.CLR_DECISION, radius=5)
        for rc in self.final_decision_pixels:
            rc = _clamp_rc(rc, shape)
            VIS_NAV._draw_circle(rgb, rc, VIS_NAV.CLR_FINAL_DEC, radius=7)
            cv2.drawMarker(rgb, (rc[1], rc[0]), VIS_NAV.CLR_FINAL_DEC, cv2.MARKER_CROSS, 16, 2)
        for gp in self.ctx.goal_positions:
            VIS_NAV._draw_star(rgb, _clamp_rc(_pos_to_pixel(gp, self.top_down_map, self.sim), shape), VIS_NAV.CLR_GOAL, size=10)
        VIS_NAV._draw_circle(rgb, _clamp_rc(_pos_to_pixel(self.start_position, self.top_down_map, self.sim), shape), VIS_NAV.CLR_START, radius=8)
        end_state = self.agent.get_state()
        end_rc = _clamp_rc(_pos_to_pixel(np.asarray(end_state.position, dtype=float), self.top_down_map, self.sim), shape)
        VIS_NAV._draw_agent_arrow(rgb, end_rc, float(get_polar_angle(end_state)), VIS_NAV.CLR_AGENT, size=12)
        _save_rgb(self.out_dir / "trajectory" / name, rgb)

    def finalize(self) -> None:
        self.save_trajectory_snapshot("trajectory_final.png")
        end_state = self.agent.get_state()
        summary = {
            "scene_name": self.ctx.scene_name,
            "episode_id": int(self.ctx.episode_id),
            "navigation_type": self.ctx.navigation_type,
            "task_id": int(self.ctx.task_id),
            "task_level": self.ctx.task_level,
            "sentence": self.ctx.sentence,
            "steps": int(self.step_count),
            "decisions": int(self.decision_num),
            "episode_cum_distance": float(self.episode_cum_distance),
            "start_position": self.start_position.tolist(),
            "end_position": np.asarray(end_state.position, dtype=float).reshape(3).tolist(),
            "goal_positions": [x.tolist() for x in self.ctx.goal_positions],
            "outputs": {
                "trajectory_final": "trajectory/trajectory_final.png",
                "decisions_dir": "decisions",
                "manual_snapshots_dir": "manual_snapshots",
            },
        }
        _write_json(self.out_dir / "summary.json", summary)


def print_controls() -> None:
    print(
        "\nControls:\n"
        "  w       move forward 0.25m\n"
        "  e       move forward 1.0m\n"
        "  a/d     turn left/right\n"
        "  s       turn around\n"
        "  o/p     look up/down\n"
        "  r       run 12-view scan + frontier + decision, save logs\n"
        "  f       auto-follow latest decision target\n"
        "  k       save manual snapshot\n"
        "  h       print this help\n"
        "  q/esc   quit\n",
        flush=True,
    )


def _norm_key(key: int) -> str:
    if key < 0:
        return ""
    if key in (27,):
        return "q"
    low = key & 0xFF
    if 0 < low < 256:
        ch = chr(low).lower()
        if ch.isprintable():
            return ch
    return ""


class StdinKeyReader:
    """Read single keypresses from the terminal without requiring Enter.

    Used in headless mode so teleop works over SSH with no GUI window. Falls
    back to a disabled state if stdin is not an interactive terminal.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._fd = None
        self._old = None
        try:
            import termios  # noqa: F401
            import tty

            self._termios = termios
            self._fd = sys.stdin.fileno()
            if not os.isatty(self._fd):
                return
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # cbreak keeps Ctrl-C working
            self.enabled = True
        except Exception:
            self.enabled = False

    def get_key(self, timeout: float) -> str:
        if not self.enabled:
            return ""
        try:
            ready, _, _ = select.select([self._fd], [], [], max(0.0, float(timeout)))
            if not ready:
                return ""
            ch = os.read(self._fd, 1).decode("utf-8", "ignore")
        except Exception:
            return ""
        if ch == "\x1b":
            # Could be a bare ESC (quit) or the start of an arrow/escape
            # sequence -- drain and ignore the latter.
            extra, _, _ = select.select([self._fd], [], [], 0.0)
            if extra:
                try:
                    os.read(self._fd, 8)
                except Exception:
                    pass
                return ""
            return "q"
        if ch in ("\r", "\n"):
            return ""
        ch = ch.lower()
        return ch if ch.isprintable() else ""

    def restore(self) -> None:
        if self._old is not None and self._fd is not None:
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)
            except Exception:
                pass


def dispatch_key(nav: "InteractiveNavigator", key: str) -> bool:
    """Handle one teleop key. Returns False when the user asked to quit."""
    if key == "q":
        return False
    if key == "h":
        print_controls()
    elif key == "w":
        nav.step_action("move_forward", 1)
    elif key == "e":
        nav.step_action("move_forward", 4)
    elif key == "a":
        nav.step_action("turn_left", 1)
    elif key == "d":
        nav.step_action("turn_right", 1)
    elif key == "s":
        nav.step_action("turn_left", 6, status_prefix="turn-around")
    elif key == "o":
        nav.step_action("look_up", 1)
    elif key == "p":
        nav.step_action("look_down", 1)
    elif key == "r":
        nav.run_decision_round()
    elif key == "f":
        nav.follow_latest_target()
    elif key == "k":
        nav.save_manual_snapshot()
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive teleop with RefHM3D frontier/PQ3D/Vista2MQSC decision logging.")
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance", choices=["sequence", "object", "room", "region", "instance"])
    ap.add_argument("--instance_id", default="")
    ap.add_argument("--task_id", type=int, default=0, help="Sequence task index; ignored for direct object/room/region/instance episodes.")
    ap.add_argument("--concise_description", action="store_true")

    ap.add_argument("--navigation_data_path", default=str(PROJECT_ROOT / "LangMap_Annotations"))
    ap.add_argument("--hm3d_data_base_path", default=str(PROJECT_ROOT / "datascene"))
    ap.add_argument("--pq3d_stage1_path", default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
    ap.add_argument("--pq3d_stage2_path", default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
    ap.add_argument("--logs_dir", default=str(VISUAL_SCRIPTS / "navi-visual" / "logs"))

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--map_resolution", type=int, default=512)
    ap.add_argument("--visible_radius", type=float, default=3.0)
    ap.add_argument("--frontier_area_m2", type=float, default=9.0)
    ap.add_argument("--decision_num_min", type=int, default=3)
    ap.add_argument("--max_context_frames", type=int, default=6)
    ap.add_argument("--context_buffer_size", type=int, default=80)

    ap.add_argument("--rgb_width", type=int, default=640)
    ap.add_argument("--rgb_height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=42.0)
    ap.add_argument("--wait_ms", type=int, default=15)
    ap.add_argument("--scan_wait_ms", type=int, default=35)
    ap.add_argument("--follow_wait_ms", type=int, default=20)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--enable_topdown_cam", action="store_true",
                    help="Add a downward RGBD camera above the agent for a colored top-down scene view.")
    ap.add_argument("--topdown_cam_height", type=float, default=2.0,
                    help="Height (m) of the top-down camera above the agent; must stay below the ceiling slice.")
    ap.add_argument("--topdown_cam_hfov", type=float, default=90.0)
    ap.add_argument("--topdown_cam_res", type=int, default=512)
    ap.add_argument(
        "--live_dir",
        default="",
        help="Fixed directory for live rgb.png / topdown.png views (headless mode). "
        "Defaults to <logs_dir>/.../live inside the run output dir.",
    )

    ap.add_argument("--disable_pq3d", action="store_true", help="Use nearest-frontier fallback instead of loading PQ3D.")
    ap.set_defaults(enable_vista2mqsc_refine=True)
    ap.add_argument("--enable_vista2mqsc_refine", dest="enable_vista2mqsc_refine", action="store_true", help="Run Vista2MQSC final-decision hook after PQ3D final decisions.")
    ap.add_argument("--disable_vista2mqsc_refine", dest="enable_vista2mqsc_refine", action="store_false", help="Skip the Vista2MQSC final-decision hook.")
    ap.add_argument("--use_vista2mqsc_follower", action="store_true", help="Use the robust repaired follower from the Vista2MQSC script.")
    ap.add_argument("--vistals_apply_task_levels", default="object,room,region,instance")
    ap.add_argument("--mqsc_use_vlm", action="store_true", help="Allow MQSC-R1 to call its VLM decomposition path.")
    ap.add_argument("--mqsc_vlm_model", default=os.environ.get("VLM_MODEL", "gpt-4o-mini"))

    ap.add_argument("--max_frontier_facing", type=int, default=12)
    ap.add_argument("--max_saved_follow_frames", type=int, default=24)
    ap.add_argument("--max_follow_actions", type=int, default=0, help="0 means no cap.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ctx = load_task_context(args)
    scene_path = _resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(os.path.expanduser(args.logs_dir))
        / f"run={run_id}"
        / f"scene={ctx.scene_name}"
        / f"navigation_type={ctx.navigation_type}"
        / f"episode={ctx.episode_id}"
        / f"task={ctx.task_id:02d}_{ctx.task_level}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nav-visual] scene_path={scene_path}", flush=True)
    print(f"[nav-visual] task={ctx.task_level} sentence={ctx.sentence}", flush=True)
    print(f"[nav-visual] logs={out_dir}", flush=True)
    print_controls()

    sim, agent = build_interactive_simulator(args, scene_path)
    nav = InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)

    key_reader: Optional[StdinKeyReader] = None
    if bool(args.headless):
        print(f"[nav-visual] live rgb view : {nav.live_dir / 'rgb.png'}", flush=True)
        print(f"[nav-visual] live topdown   : {nav.live_dir / 'topdown.png'}", flush=True)
        key_reader = StdinKeyReader()
        if not key_reader.enabled:
            print(
                "[nav-visual] stdin is not an interactive terminal; cannot read keys. "
                "Saving one snapshot and exiting.",
                flush=True,
            )

    try:
        # Prime the live view / GUI window before waiting for the first key.
        nav.display()
        while True:
            if bool(args.headless):
                if key_reader is None or not key_reader.enabled:
                    nav.save_manual_snapshot()
                    break
                key = key_reader.get_key(max(0.001, int(args.wait_ms) / 1000.0))
            else:
                obs = sim.get_sensor_observations()
                nav.display(obs)
                key = _norm_key(cv2.waitKeyEx(max(1, int(args.wait_ms))))

            if not key:
                continue
            if not dispatch_key(nav, key):
                break
    finally:
        try:
            nav.finalize()
        finally:
            sim.close()
            if key_reader is not None:
                key_reader.restore()
            if not bool(args.headless):
                cv2.destroyAllWindows()

    print(f"[nav-visual] Done. Logs saved under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
