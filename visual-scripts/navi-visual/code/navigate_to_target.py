"""Baseline-style direct navigation to the target with decision-pause logs.

No VLM is called here, and no Evidence/Entity/Endpoint module folders are
created.  The script only simulates the baseline rhythm: pause at each decision,
do a 12-view panorama, save top-down/RGB maps, then quietly move along an oracle
shortest-path waypoint toward the task target.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[2]
TELEOP_PATH = CODE_DIR / "interactive_vista2mqsc_teleop.py"
GUIDED_HELPERS_PATH = CODE_DIR / "module_sim_guided.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _glob_rel(root: Path, base: Path, pattern: str) -> List[str]:
    if not base.exists():
        return []
    return [_rel(root, p) for p in sorted(base.glob(pattern)) if p.exists() and p.stat().st_size > 0]


def _existing_rel(root: Path, paths: List[Path]) -> List[str]:
    return [_rel(root, p) for p in paths if p.exists() and p.stat().st_size > 0]


def _planar_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a, dtype=float).reshape(3) - np.asarray(b, dtype=float).reshape(3))[[0, 2]]))


def _direct_arrive_thresh(cli: argparse.Namespace) -> float:
    return float(max(0.20, min(0.35, float(cli.arrive_thresh_m))))


def _retarget_route_to_snapped_goal(route: Dict[str, Any], helpers: Any, max_rounds: int) -> List[np.ndarray]:
    path_points = [np.asarray(p, dtype=float).reshape(3) for p in route.get("path_points", [])]
    snapped_goal = np.asarray(route.get("snapped_goal_xyz", route.get("goal_object_xyz")), dtype=float).reshape(3)
    if not path_points:
        path_points = [snapped_goal.copy()]
    if _planar_dist(path_points[-1], snapped_goal) > 1e-4:
        path_points.append(snapped_goal.copy())

    waypoint_count = max(1, int(max_rounds))
    waypoints = helpers._resample_polyline(path_points, waypoint_count)
    if not waypoints:
        waypoints = [snapped_goal.copy()]
    waypoints[-1] = snapped_goal.copy()

    original_guided_stop = route.get("guided_stop_xyz")
    route["original_guided_stop_xyz"] = original_guided_stop
    route["original_guided_stop_margin_m"] = route.get("guided_stop_margin_m")
    route["direct_decision_viewpoint_order"] = "move_to_sampled_viewpoint_then_pause_scan"
    route["direct_goal_nav_xyz"] = snapped_goal.tolist()
    route["guided_stop_xyz"] = snapped_goal.tolist()
    route["guided_stop_margin_m"] = 0.0
    try:
        route["guided_motion_distance_m"] = float(helpers._polyline_xz_length(path_points))
    except Exception:
        route["guided_motion_distance_m"] = route.get("geodesic_distance_m")
    route["motion_path_points"] = [p.tolist() for p in path_points]
    route["decision_move_waypoints"] = [p.tolist() for p in waypoints]
    return [np.asarray(p, dtype=float).reshape(3) for p in waypoints]


def _select_reachable_goal(nav: Any, ctx: Any, helpers: Any) -> tuple[np.ndarray, Dict[str, Any]]:
    start = np.asarray(nav.agent.get_state().position, dtype=float).reshape(3)
    candidates = [np.asarray(p, dtype=float).reshape(3) for p in list(getattr(ctx, "goal_positions", []) or [])]
    if not candidates:
        raise RuntimeError("task has no goal positions")
    scored: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        route = helpers._shortest_path_points(nav, start, cand)
        geo = route.get("geodesic_distance_m")
        geo_f = float(geo) if geo is not None and np.isfinite(float(geo)) else float("inf")
        scored.append(
            {
                "index": int(idx),
                "goal_xyz": cand.tolist(),
                "shortest_path_ok": bool(route.get("ok")),
                "geodesic_distance_m": None if not np.isfinite(geo_f) else float(geo_f),
                "snapped_goal_xyz": np.asarray(route.get("goal_nav"), dtype=float).reshape(3).tolist(),
                "island_index": route.get("island_index"),
            }
        )
    reachable = [row for row in scored if row["shortest_path_ok"] and row["geodesic_distance_m"] is not None]
    if not reachable:
        raise RuntimeError(f"no reachable goal candidate from start for task_id={ctx.task_id}: {scored}")
    chosen = min(reachable, key=lambda row: float(row["geodesic_distance_m"]))
    return np.asarray(chosen["goal_xyz"], dtype=float).reshape(3), {
        "strategy": "nearest_reachable_goal_candidate_by_geodesic",
        "selected_goal_index": int(chosen["index"]),
        "selected_goal": chosen,
        "candidate_count": int(len(scored)),
        "reachable_candidate_count": int(len(reachable)),
        "candidates": scored,
    }


def _build_log_streams(out_dir: Path, rounds: int) -> Dict[str, Any]:
    streams_dir = out_dir / "log_streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    categories: Dict[str, Dict[str, Any]] = {}

    def add_category(name: str, description: str, entries: List[Dict[str, Any]]) -> None:
        index_rel = f"log_streams/{name}/index.json"
        _write_json(
            out_dir / index_rel,
            {
                "category": name,
                "description": description,
                "entry_count": len(entries),
                "entries": entries,
            },
        )
        categories[name] = {
            "description": description,
            "index": index_rel,
            "entry_count": len(entries),
        }

    decision_entries: List[Dict[str, Any]] = []
    topdown_entries: List[Dict[str, Any]] = []
    rgb_entries: List[Dict[str, Any]] = []
    for dec in range(int(rounds)):
        dtag = f"dec_{dec:03d}"
        dec_dir = out_dir / "baseline_decisions" / dtag
        pano_dir = dec_dir / "panorama"
        common = {
            "decision": dec,
            "decision_json": _existing_rel(out_dir, [dec_dir / "decision.json"]),
            "panorama_inputs": _glob_rel(out_dir, pano_dir, "view_*.png")
            + _existing_rel(out_dir, [pano_dir / "current_decision_panorama_vfv_order.jpg"]),
        }
        decision_entries.append(
            {
                **common,
                "policy": "direct_oracle_move_to_viewpoint_then_baseline_pause_scan",
                "no_vlm": True,
                "no_module_grounding": True,
                "move_frames": _existing_rel(out_dir, [out_dir / "guided_frames" / dtag / "frames_index.json"]),
            }
        )
        topdown_entries.append(
            {
                **common,
                "images": _existing_rel(
                    out_dir,
                    [
                        dec_dir / "topdown_map.png",
                        dec_dir / "topdown_scene_rgb_annotated.png",
                        dec_dir / "topdown_cam_rgb.png",
                    ],
                ),
                "image_roles": {
                    "topdown_map.png": "fog_topdown_map_with_navigation_overlays",
                    "topdown_scene_rgb_annotated.png": "global_scene_rgb_overhead_photo_mosaic_with_overlays",
                    "topdown_cam_rgb.png": "local_robot_centered_downward_rgb_photo",
                },
                "metadata": _existing_rel(
                    out_dir,
                    [
                        dec_dir / "topdown_map_info.json",
                        dec_dir / "topdown_scene_rgb_annotated_info.json",
                        dec_dir / "topdown_cam_info.json",
                    ],
                ),
            }
        )
        rgb_entries.append(
            {
                **common,
                "clean_rgb": _existing_rel(out_dir, [dec_dir / "topdown_scene_rgb.png"]),
                "annotated_rgb": _existing_rel(out_dir, [dec_dir / "topdown_scene_rgb_annotated.png"]),
                "local_topdown_rgb": _existing_rel(out_dir, [dec_dir / "topdown_cam_rgb.png"]),
                "metadata": _existing_rel(
                    out_dir,
                    [dec_dir / "topdown_scene_rgb_info.json", dec_dir / "topdown_scene_rgb_annotated_info.json"],
                ),
            }
        )

    final_dir = out_dir / "final"
    final_pano = final_dir / "panorama"
    topdown_entries.append(
        {
            "decision": "final",
            "images": _existing_rel(
                out_dir,
                [
                    final_dir / "topdown_map.png",
                    final_dir / "topdown_scene_rgb_annotated.png",
                    final_dir / "topdown_cam_rgb.png",
                    out_dir / "trajectory" / "route_start_to_goal.png",
                ],
            ),
            "metadata": _existing_rel(
                out_dir,
                [
                    final_dir / "topdown_map_info.json",
                    final_dir / "topdown_scene_rgb_annotated_info.json",
                    final_dir / "topdown_cam_info.json",
                    out_dir / "trajectory" / "route_start_to_goal_info.json",
                ],
            ),
        }
    )
    rgb_entries.append(
        {
            "decision": "final",
            "clean_rgb": _existing_rel(out_dir, [final_dir / "topdown_scene_rgb.png"]),
            "annotated_rgb": _existing_rel(out_dir, [final_dir / "topdown_scene_rgb_annotated.png"]),
            "local_topdown_rgb": _existing_rel(out_dir, [final_dir / "topdown_cam_rgb.png"]),
            "target_facing_rgb": _existing_rel(out_dir, [final_dir / "target_facing_rgb.png"]),
            "panorama_inputs": _glob_rel(out_dir, final_pano, "view_*.png")
            + _existing_rel(out_dir, [final_pano / "current_decision_panorama_vfv_order.jpg"]),
            "metadata": _existing_rel(
                out_dir,
                [
                    final_dir / "topdown_scene_rgb_info.json",
                    final_dir / "topdown_scene_rgb_annotated_info.json",
                    final_dir / "target_facing_rgb_info.json",
                ],
            ),
        }
    )
    nav_entries = [
        {
            "guided_route": _existing_rel(out_dir, [out_dir / "guided_route.json"]),
            "guided_moves": _existing_rel(out_dir, [out_dir / "guided_moves.json"]),
            "summary": _existing_rel(out_dir, [out_dir / "direct_navigation_summary.json"]),
            "trajectory": _existing_rel(out_dir, [out_dir / "trajectory" / "route_start_to_goal.png"]),
            "policy": "hard_direct_navigation_to_target_waypoints",
        }
    ]

    add_category(
        "baseline_decision_pause_scan",
        "Baseline-style pauses at sampled route viewpoints: hard direct move, 12-view scan, frontier observation, and waypoint metadata. No VLM/module call.",
        decision_entries,
    )
    add_category("topdown_map_process", "Gray top-down map changes with trajectory, position, frontiers, and direct target waypoint.", topdown_entries)
    add_category("rgb_map_process", "Global scene RGB overhead mosaics, local topdown RGB photos, and final target-facing RGB photo.", rgb_entries)
    add_category("direct_navigation", "Oracle-guided physical navigation frames and route summaries.", nav_entries)

    manifest = {
        "version": "navi_visual_direct_log_streams_v1",
        "root": str(out_dir),
        "note": "Direct baseline visualization only: no VLM, no Evidence/Entity/Endpoint modules. Clean no-legend images are retained.",
        "categories": categories,
    }
    _write_json(streams_dir / "manifest.json", manifest)
    return manifest


def _save_direct_topdown_maps(
    *,
    helpers: Any,
    nav: Any,
    sim: Any,
    out_dir: Path,
    agent_state: Any,
    target: Optional[np.ndarray],
    is_final: bool,
    frontiers: List[np.ndarray],
    selected_frontier_idx: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    helpers.save_decision_topdown_map(
        nav=nav,
        sim=sim,
        out_dir=out_dir,
        agent_state=agent_state,
        target=target,
        is_final=bool(is_final),
        frontiers=frontiers,
        selected_frontier_idx=selected_frontier_idx,
    )

    scene_bgr, scene_info = helpers.render_global_topdown_scene_rgb(nav, sim)
    cv2.imwrite(str(out_dir / "topdown_scene_rgb.png"), scene_bgr)
    _write_json(
        out_dir / "topdown_scene_rgb_info.json",
        {
            **scene_info,
            "file": "topdown_scene_rgb.png",
            "semantic_type": "global_scene_rgb_overhead_photo_mosaic",
            "note": "Clean full-scene RGB overhead mosaic. This is not a fog/topdown occupancy map and has no overlays or legend.",
        }
    )
    annotated_bgr = helpers._annotate_scene_rgb_bgr(
        nav=nav,
        sim=sim,
        scene_bgr=scene_bgr,
        agent_state=agent_state,
        target=target,
        is_final=bool(is_final),
        frontiers=frontiers,
        selected_frontier_idx=selected_frontier_idx,
    )
    cv2.imwrite(str(out_dir / "topdown_scene_rgb_annotated.png"), annotated_bgr)
    _write_json(
        out_dir / "topdown_scene_rgb_annotated_info.json",
        {
            "source": "global_topdown_rgb_annotation",
            "base_image": "topdown_scene_rgb.png",
            "file": "topdown_scene_rgb_annotated.png",
            "semantic_type": "global_scene_rgb_overhead_photo_mosaic_with_navigation_overlays",
            "note": "Annotated direct-baseline RGB overhead mosaic: trajectory, current pose, target waypoint, and observed frontiers.",
            "overlays": {
                "blue_line": "trajectory",
                "red_arrow": "agent_position_and_heading",
                "orange_star": "goal",
                "gray_circle": "observed_frontier",
                "magenta_circle": "baseline_visual_selected_frontier",
                "red_cross": "final_target",
                "blue_cross": "non_final_target",
            },
            "goal_marker_count": int(len(helpers._goal_positions_for_overlay(nav))),
            "goal_marker_positions_xyz": [p.tolist() for p in helpers._goal_positions_for_overlay(nav)],
        },
    )


def _save_target_facing_rgb(
    *,
    helpers: Any,
    nav: Any,
    sim: Any,
    out_dir: Path,
    target: np.ndarray,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = nav.agent
    saved_state = helpers._M._state_copy(agent.get_state())
    saved_prev_state = helpers._M._state_copy(getattr(nav, "prev_state", saved_state))
    applied_tilt = 0
    tilt_action = None
    target_arr = np.asarray(target, dtype=float).reshape(3)
    try:
        st = helpers._M.habitat_sim.AgentState()
        st.position = np.asarray(saved_state.position, dtype=float).reshape(3)
        st.rotation = helpers._look_at_quat_xz(st.position, target_arr)
        agent.set_state(st)

        camera_y = float(st.position[1]) + 1.31
        planar = max(1e-3, _planar_dist(st.position, target_arr))
        pitch_deg = math.degrees(math.atan2(float(target_arr[1]) - camera_y, planar))
        tilt_steps = int(round(abs(pitch_deg) / 30.0))
        applied_tilt = max(0, min(3, tilt_steps))
        tilt_action = "look_up" if pitch_deg > 0 else "look_down"
        for _ in range(applied_tilt):
            sim.step(tilt_action)

        obs = sim.get_sensor_observations()
        rgb_bgr = cv2.cvtColor(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        raw_path = out_dir / "target_facing_rgb.png"
        cv2.imwrite(str(raw_path), rgb_bgr)
        sensor_state = agent.get_state().sensor_states.get("color_sensor")
        info = {
            "source": "final_agent_color_sensor_look_at_goal_object",
            "file": raw_path.name,
            "semantic_type": "final_target_facing_rgb_photo",
            "target_xyz": target_arr.tolist(),
            "agent_position_xyz": np.asarray(saved_state.position, dtype=float).reshape(3).tolist(),
            "yaw_targeting": "look_at_goal_object_xz",
            "pitch_to_target_deg": float(pitch_deg),
            "tilt_action": tilt_action,
            "tilt_steps": int(applied_tilt),
            "resolution_hw": list(rgb_bgr.shape[:2]),
        }
        if sensor_state is not None:
            info["camera_position_xyz"] = np.asarray(sensor_state.position, dtype=float).reshape(3).tolist()
        _write_json(out_dir / "target_facing_rgb_info.json", info)
        return info
    finally:
        if applied_tilt and tilt_action is not None:
            undo = "look_down" if tilt_action == "look_up" else "look_up"
            for _ in range(applied_tilt):
                try:
                    sim.step(undo)
                except Exception:
                    break
        agent.set_state(saved_state)
        nav.prev_state = saved_prev_state


def _ensure_guided_frame_record(
    *,
    helpers: Any,
    nav: Any,
    sim: Any,
    frame_dir: Path,
    round_idx: int,
    goal_idx: int,
    target: np.ndarray,
    move_rec: Dict[str, Any],
) -> Dict[str, Any]:
    if int(move_rec.get("saved_frame_count", 0) or 0) > 0:
        return move_rec
    frame_dir.mkdir(parents=True, exist_ok=True)
    obs = sim.get_sensor_observations()
    state = nav.agent.get_state()
    rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8)
    rgb_path = frame_dir / "frame_0000_rgb.png"
    top_path = frame_dir / "frame_0000_topdown.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(top_path), cv2.cvtColor(nav.render_topdown(), cv2.COLOR_RGB2BGR))
    pos = np.asarray(state.position, dtype=float).reshape(3)
    frame = {
        "frame": 0,
        "round": int(round_idx),
        "goal_index": int(goal_idx),
        "action": "hold_no_move_needed",
        "rgb": rgb_path.name,
        "topdown": top_path.name,
        "step_count": int(getattr(nav, "step_count", 0)),
        "position_xyz": pos.tolist(),
        "agent_pixel_rc": [int(x) for x in helpers._rc(pos, nav, sim)],
        "path_pixels_rc": [[int(a), int(b)] for a, b in list(getattr(nav, "path_pixels", []))],
        "heading_xz": helpers._forward_xz(state).tolist(),
        **helpers._color_sensor_pose_metrics(state),
    }
    _write_json(
        frame_dir / "frames_index.json",
        {
            "source": "direct_baseline_hold_frame",
            "round": int(round_idx),
            "goal_index": int(goal_idx),
            "target_xyz": np.asarray(target, dtype=float).reshape(3).tolist(),
            "frame_count": 1,
            "frames": [frame],
        },
    )
    updated = dict(move_rec)
    updated["saved_frame_count"] = 1
    updated["hold_frame_added"] = True
    return updated


def _build_teleop_args(cli: argparse.Namespace, out_dir: Path) -> List[str]:
    args = [
        "teleop",
        "--scene_name",
        str(cli.scene_name),
        "--episode_id",
        str(cli.episode_id),
        "--navigation_type",
        str(cli.navigation_type),
        "--instance_id",
        str(cli.instance_id),
        "--task_id",
        str(cli.task_id),
        "--headless",
        "--disable_pq3d",
        "--disable_vista2mqsc_refine",
        "--enable_topdown_cam",
        "--topdown_cam_height",
        "2.0",
        "--logs_dir",
        str(out_dir),
        "--live_dir",
        str(Path(cli.live_dir).expanduser()),
    ]
    if bool(cli.concise_description):
        args.append("--concise_description")
    return args


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline direct target navigation with decision-pause visualization logs.")
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--arrive_thresh_m", type=float, default=0.7)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--logs_dir", default=str(CODE_DIR.parent / "logs" / "direct"))
    ap.add_argument("--live_dir", default=str(CODE_DIR.parent / "logs" / "live"))
    ap.add_argument("--concise_description", action="store_true")
    ap.add_argument("--sequence_task_count", type=int, default=1)
    cli = ap.parse_args()

    teleop = _load_module(TELEOP_PATH, "direct_teleop_mod")
    helpers = _load_module(GUIDED_HELPERS_PATH, "direct_guided_helpers")
    helpers._M = teleop
    helpers._VIS = teleop.VIS_NAV

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cli.logs_dir).expanduser() / f"run={run_id}_ep{cli.episode_id}_{cli.instance_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_argv = list(sys.argv)
    sys.argv = _build_teleop_args(cli, out_dir)
    try:
        args = teleop.parse_args()
    finally:
        sys.argv = old_argv

    ctx = teleop.load_task_context(args)
    scene_path = teleop._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    sim, agent = teleop.build_interactive_simulator(args, scene_path)
    nav = teleop.InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)
    nav.VIS_NAV = teleop.VIS_NAV
    nav.VIS_meters_per_px = float(teleop.maps.calculate_meters_per_pixel(int(args.map_resolution), sim=sim))

    goal, goal_selection = _select_reachable_goal(nav, ctx, helpers)
    nav.visual_goal_positions_override = [goal.copy()]
    route = helpers.build_guided_route(nav, goal, int(cli.max_rounds))
    waypoints = _retarget_route_to_snapped_goal(route, helpers, int(cli.max_rounds))
    goal_nav = np.asarray(route.get("direct_goal_nav_xyz", route.get("snapped_goal_xyz", goal)), dtype=float).reshape(3)
    arrive_nav_thresh = _direct_arrive_thresh(cli)
    route.update(
        {
            "direct_navigation_mode": True,
            "baseline_pause_scan_mode": True,
            "no_vlm": True,
            "module_artifacts_enabled": False,
            "direct_arrive_thresh_m": float(arrive_nav_thresh),
            "goal_selection": goal_selection,
            "task_id": int(ctx.task_id),
            "task_level": str(ctx.task_level),
            "sentence": str(ctx.sentence),
        }
    )
    _write_json(out_dir / "guided_route.json", route)

    log: List[str] = []
    decisions: List[Dict[str, Any]] = []
    moves: List[Dict[str, Any]] = []
    print(f"[direct] sentence: {ctx.sentence}", flush=True)
    print(f"[direct] out_dir : {out_dir}", flush=True)
    print(
        f"[direct] route decision_viewpoints={len(waypoints)} geo={route.get('geodesic_distance_m')} "
        f"goal_nav={goal_nav.tolist()} arrive_nav_thresh={arrive_nav_thresh:.2f}",
        flush=True,
    )

    rounds = 0
    while rounds < int(cli.max_rounds) and rounds < len(waypoints):
        dtag = f"dec_{rounds:03d}"
        dec_dir = out_dir / "baseline_decisions" / dtag
        pano_dir = dec_dir / "panorama"
        target = waypoints[rounds]
        nav.decision_num = int(rounds)
        nav.current_target = target.copy()
        nav.current_target_is_final = bool(rounds >= len(waypoints) - 1)

        print(
            f"[direct] === decision {rounds}: move-to-viewpoint then pause+scan "
            f"to_nav_goal={_planar_dist(agent.get_state().position, goal_nav):.2f}m waypoint={target.tolist()} ===",
            flush=True,
        )
        pre_move_state = agent.get_state()
        frame_dir = out_dir / "guided_frames" / dtag
        move_rec = helpers.execute_guided_move(
            nav,
            target,
            round_idx=rounds,
            log=log,
            frame_dir=frame_dir,
            goal_idx=0,
            close_thresh_m=arrive_nav_thresh,
        )
        move_rec = _ensure_guided_frame_record(
            helpers=helpers,
            nav=nav,
            sim=sim,
            frame_dir=frame_dir,
            round_idx=rounds,
            goal_idx=0,
            target=target,
            move_rec=move_rec,
        )
        moves.append(move_rec)

        decision_state = agent.get_state()
        views = helpers.scan_and_capture(nav, sim, pano_dir)
        frontiers = nav.detect_frontiers()
        selected_frontier_idx: Optional[int] = None
        if frontiers:
            selected_frontier_idx = int(np.argmin([_planar_dist(np.asarray(f), target) for f in frontiers]))
        nav.current_frontiers = list(frontiers)
        nav.selected_frontier_idx = selected_frontier_idx
        helpers.render_topdown_cam(nav, sim, [dec_dir / "topdown_cam"], log)
        _save_direct_topdown_maps(
            helpers=helpers,
            nav=nav,
            sim=sim,
            out_dir=dec_dir,
            agent_state=agent.get_state(),
            target=target,
            is_final=bool(rounds >= len(waypoints) - 1),
            frontiers=list(frontiers),
            selected_frontier_idx=selected_frontier_idx,
        )
        try:
            nav.decision_pixels.append(helpers._rc(np.asarray(agent.get_state().position, dtype=float), nav, sim))
        except Exception:
            pass
        decision = {
            "decision_index": int(rounds),
            "policy": "hard_direct_navigation_to_viewpoint_then_baseline_pause_scan",
            "no_vlm": True,
            "no_module_grounding": True,
            "sentence": str(ctx.sentence),
            "pre_move_position_xyz": np.asarray(pre_move_state.position, dtype=float).reshape(3).tolist(),
            "decision_position_xyz": np.asarray(decision_state.position, dtype=float).reshape(3).tolist(),
            "post_scan_position_xyz": np.asarray(agent.get_state().position, dtype=float).reshape(3).tolist(),
            "goal_object_xyz": goal.tolist(),
            "goal_navigation_xyz": goal_nav.tolist(),
            "selected_oracle_waypoint_xyz": target.tolist(),
            "distance_to_goal_object_at_decision_m": _planar_dist(agent.get_state().position, goal),
            "distance_to_navigation_goal_at_decision_m": _planar_dist(agent.get_state().position, goal_nav),
            "move_to_decision_viewpoint": move_rec,
            "frontier_count_observed": int(len(frontiers)),
            "baseline_visual_selected_frontier_index": selected_frontier_idx,
            "frontiers_xyz": [np.asarray(f, dtype=float).reshape(3).tolist() for f in frontiers],
            "panorama": _rel(out_dir, pano_dir / "current_decision_panorama_vfv_order.jpg"),
            "topdown_map": "topdown_map.png",
            "topdown_scene_rgb": "topdown_scene_rgb.png",
            "topdown_scene_rgb_annotated": "topdown_scene_rgb_annotated.png",
            "actual_navigation_override": "direct_oracle_shortest_path_to_decision_viewpoint_before_scan",
        }
        _write_json(dec_dir / "decision.json", decision)
        decisions.append(decision)

        rounds += 1
        if _planar_dist(agent.get_state().position, goal_nav) <= arrive_nav_thresh:
            break

    final_goal_correction: Optional[Dict[str, Any]] = None
    if _planar_dist(agent.get_state().position, goal_nav) > arrive_nav_thresh:
        print(
            f"[direct] final direct correction to snapped goal: "
            f"to_nav_goal={_planar_dist(agent.get_state().position, goal_nav):.2f}m",
            flush=True,
        )
        frame_dir = out_dir / "guided_frames" / "final_direct_to_goal"
        final_goal_correction = helpers.execute_guided_move(
            nav,
            goal_nav,
            round_idx=rounds,
            log=log,
            frame_dir=frame_dir,
            goal_idx=0,
            close_thresh_m=arrive_nav_thresh,
        )
        final_goal_correction = _ensure_guided_frame_record(
            helpers=helpers,
            nav=nav,
            sim=sim,
            frame_dir=frame_dir,
            round_idx=rounds,
            goal_idx=0,
            target=goal_nav,
            move_rec=final_goal_correction,
        )
        final_goal_correction["correction_type"] = "post_decision_direct_to_snapped_goal"
        moves.append(final_goal_correction)

    final_nav_dist = _planar_dist(agent.get_state().position, goal_nav)
    stop_reason = "arrived_at_snapped_goal" if final_nav_dist <= arrive_nav_thresh else "navigation_goal_unreached"
    print(
        f"[direct] final pause+scan stop_reason={stop_reason} "
        f"to_nav_goal={final_nav_dist:.2f}m to_object={_planar_dist(agent.get_state().position, goal):.2f}m",
        flush=True,
    )
    final_dir = out_dir / "final"
    final_views = helpers.scan_and_capture(nav, sim, final_dir / "panorama")
    target_facing_info = _save_target_facing_rgb(
        helpers=helpers,
        nav=nav,
        sim=sim,
        out_dir=final_dir,
        target=goal,
    )
    helpers.render_topdown_cam(nav, sim, [final_dir / "topdown_cam"], log)
    nav.current_target = goal.copy()
    nav.current_target_is_final = True
    _save_direct_topdown_maps(
        helpers=helpers,
        nav=nav,
        sim=sim,
        out_dir=final_dir,
        agent_state=agent.get_state(),
        target=goal,
        is_final=True,
        frontiers=list(getattr(nav, "current_frontiers", [])),
        selected_frontier_idx=None,
    )
    try:
        nav.final_decision_pixels.append(helpers._rc(np.asarray(agent.get_state().position, dtype=float), nav, sim))
    except Exception:
        pass

    traj_dir = out_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    nav.save_trajectory_snapshot("route_start_to_goal.png")
    _write_json(
        traj_dir / "route_start_to_goal_info.json",
        {
            "title": "Direct baseline route to target",
            "note": "No legend image is generated; clean rendered map pixels are preserved.",
        },
    )
    _write_json(out_dir / "guided_moves.json", moves)
    log_streams = _build_log_streams(out_dir, rounds)

    end_pos = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    summary = {
        "scene_name": ctx.scene_name,
        "episode_id": int(ctx.episode_id),
        "navigation_type": str(ctx.navigation_type),
        "task_id": int(ctx.task_id),
        "task_level": str(ctx.task_level),
        "sentence": ctx.sentence,
        "direct_navigation_mode": True,
        "baseline_pause_scan_mode": True,
        "guided_visualization_mode": False,
        "no_vlm": True,
        "module_artifacts_enabled": False,
        "decision_rounds": int(rounds),
        "stop_reason": stop_reason,
        "start_position": np.asarray(nav.start_position, dtype=float).reshape(3).tolist(),
        "end_position": end_pos.tolist(),
        "goal_object_position": goal.tolist(),
        "goal_selection": goal_selection,
        "final_planar_distance_to_goal_m": _planar_dist(end_pos, goal),
        "goal_navigation_position": goal_nav.tolist(),
        "final_planar_distance_to_navigation_goal_m": _planar_dist(end_pos, goal_nav),
        "direct_arrive_thresh_m": float(arrive_nav_thresh),
        "final_goal_correction": final_goal_correction,
        "guided_route": route,
        "guided_moves": moves,
        "decisions": decisions,
        "final_panorama_view_count": int(len(final_views)),
        "final_target_facing_rgb": target_facing_info,
        "topdown_map": getattr(nav, "topdown_map_info", {}),
        "log_streams": log_streams,
        "log": log,
    }
    _write_json(out_dir / "direct_navigation_summary.json", summary)
    _write_json(out_dir / "navigation_summary.json", summary)
    # Keep this compatibility name so existing batch status readers can display
    # one summary path, but mark clearly that this is not a module run.
    _write_json(out_dir / "module_sim_summary.json", summary)

    sim.close()
    print("[direct] DONE.", flush=True)
    print(f"[direct] outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
