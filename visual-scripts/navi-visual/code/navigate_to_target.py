"""Manual-style direct navigation to the episode target object.

Reuses the headless InteractiveNavigator from interactive_vista2mqsc_teleop.py.
It (1) does a short 360 "exploration" spin at the start, (2) reads the target
object coordinates directly, then (3) navigates straight to the target using the
same GreedyGeodesicFollower the teleop "follow" key uses.

Along the way it records, for every primitive step:
  * the live camera RGB view and the live TopDownMap (overwritten in --live_dir)
  * a numbered side-by-side composite frame under navigation_frames/
  * the equivalent teleop key (move_forward->w, turn_left->a, turn_right->d)

Outputs (under the run log dir):
  navigation_frames/NNN_<phase>_<key>.png   per-step camera|topdown composite
  trajectory/route_start_to_goal.png        final direct route (path/start/goal)
  manual_navigation_keys.json               full + compressed key sequence
  navigation_summary.json                   distances, positions, route points
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
TELEOP_PATH = CODE_DIR / "interactive_vista2mqsc_teleop.py"


def _load_teleop() -> Any:
    spec = importlib.util.spec_from_file_location("teleop_mod", str(TELEOP_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["teleop_mod"] = module  # needed so dataclasses can resolve types
    spec.loader.exec_module(module)
    return module


ACTION_TO_KEY = {"move_forward": "w", "turn_left": "a", "turn_right": "d", "look_up": "o", "look_down": "p"}


def _compress_keys(keys: List[str]) -> str:
    if not keys:
        return ""
    out = []
    prev = keys[0]
    count = 1
    for k in keys[1:]:
        if k == prev:
            count += 1
        else:
            out.append(f"{prev}x{count}" if count > 1 else prev)
            prev, count = k, 1
    out.append(f"{prev}x{count}" if count > 1 else prev)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--explore_turns", type=int, default=12, help="number of 30deg left turns for the pretend-explore spin")
    ap.add_argument("--logs_dir", default=str(CODE_DIR.parent / "logs" / "manual_nav"))
    ap.add_argument("--live_dir", default=str(CODE_DIR.parent / "logs" / "live"))
    cli = ap.parse_args()

    m = _load_teleop()

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cli.logs_dir) / f"run={run_id}_ep{cli.episode_id}_{cli.instance_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a full args namespace via the teleop parser so every attribute the
    # navigator expects exists, then run headless with PQ3D disabled (direct nav).
    sys.argv = [
        "teleop",
        "--scene_name", cli.scene_name,
        "--episode_id", str(cli.episode_id),
        "--navigation_type", cli.navigation_type,
        "--instance_id", cli.instance_id,
        "--task_id", str(cli.task_id),
        "--headless",
        "--disable_pq3d",
        "--logs_dir", str(out_dir),
        "--live_dir", str(cli.live_dir),
    ]
    args = m.parse_args()

    ctx = m.load_task_context(args)
    scene_path = m._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    sim, agent = m.build_interactive_simulator(args, scene_path)
    nav = m.InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)

    frames_dir = out_dir / "navigation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    start_pos = np.asarray(agent.get_state().position, dtype=float).reshape(3).copy()
    goal = np.asarray(ctx.goal_positions[0], dtype=float).reshape(3)

    print(f"[nav] sentence : {ctx.sentence}", flush=True)
    print(f"[nav] start    : {start_pos.tolist()}", flush=True)
    print(f"[nav] goal(obj): {goal.tolist()}", flush=True)
    print(f"[nav] out_dir  : {out_dir}", flush=True)

    frame_idx = [0]
    keys: List[str] = []

    def save_frame(phase: str, key: str) -> None:
        obs = sim.get_sensor_observations()
        rgb_bgr, top_bgr = nav._compose_frames(obs, status=f"{phase}:{key}")
        h = rgb_bgr.shape[0]
        scale = h / top_bgr.shape[0]
        top_resized = cv2.resize(top_bgr, (int(top_bgr.shape[1] * scale), h))
        composite = cv2.hconcat([rgb_bgr, top_resized])
        cv2.imwrite(str(frames_dir / f"{frame_idx[0]:03d}_{phase}_{key}.png"), composite)
        frame_idx[0] += 1

    # Frame 0: initial state.
    save_frame("init", "-")

    # 1) Pretend-explore: a full 360 spin (12 x 30deg left turns).
    print(f"[nav] exploration spin: {cli.explore_turns} x turn_left", flush=True)
    for _ in range(int(cli.explore_turns)):
        nav.step_action("turn_left", 1, status_prefix="explore")
        keys.append("a")
        save_frame("explore", "a")

    # 2) Plan the direct route to the target object and execute it.
    nav.current_target = goal.copy()
    nav.current_target_is_final = True
    actions, follow_log = nav._plan_follow_actions(goal)
    print(f"[nav] planner: ok={follow_log.get('ok')} planner={follow_log.get('planner')} actions={len(actions)}", flush=True)

    nav_keys: List[str] = []
    for action in actions:
        if not action:
            continue
        nav.step_action(str(action), 1, status_prefix="goto")
        key = ACTION_TO_KEY.get(str(action), "?")
        keys.append(key)
        nav_keys.append(key)
        save_frame("goto", key)

    # 2b) Turn to face the target so the final camera view centers the object.
    import math
    import habitat_sim.utils.common as _hsu

    def _signed_heading_err_deg() -> float:
        st = agent.get_state()
        fwd = _hsu.quat_rotate_vector(st.rotation, np.array([0.0, 0.0, -1.0]))
        to = goal - np.asarray(st.position, dtype=float).reshape(3)
        fx, fz, gx, gz = float(fwd[0]), float(fwd[2]), float(to[0]), float(to[2])
        return math.degrees(math.atan2(fx * gz - fz * gx, fx * gx + fz * gz))

    face_keys: List[str] = []
    if abs(_signed_heading_err_deg()) > 20.0:
        before = abs(_signed_heading_err_deg())
        nav.step_action("turn_left", 1, status_prefix="face")
        keys.append("a"); face_keys.append("a"); save_frame("face", "a")
        turn_action = "turn_left" if abs(_signed_heading_err_deg()) < before else "turn_right"
        for _ in range(11):
            if abs(_signed_heading_err_deg()) <= 20.0:
                break
            nav.step_action(turn_action, 1, status_prefix="face")
            k = ACTION_TO_KEY[turn_action]
            keys.append(k); face_keys.append(k); save_frame("face", k)
    print(f"[nav] face-target turns: {len(face_keys)} (residual heading err={_signed_heading_err_deg():.1f} deg)", flush=True)

    # 2c) Tilt the camera down to center the close, low target object.
    tilt_keys: List[str] = []
    st = agent.get_state()
    to = goal - np.asarray(st.position, dtype=float).reshape(3)
    planar = float(np.linalg.norm(to[[0, 2]]))
    drop = float(st.position[1]) + 1.31 - float(goal[1])  # camera height - goal height
    depression_deg = math.degrees(math.atan2(max(drop, 0.0), max(planar, 1e-3)))
    n_tilt = int(round(depression_deg / 30.0))  # tilt_angle is 30 deg
    n_tilt = max(0, min(3, n_tilt))
    for _ in range(n_tilt):
        nav.step_action("look_down", 1, status_prefix="face")
        keys.append("p"); tilt_keys.append("p"); save_frame("face", "p")
    print(f"[nav] look-down tilts: {n_tilt} (depression={depression_deg:.1f} deg)", flush=True)

    end_pos = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    planar_to_goal = float(np.linalg.norm((end_pos - goal)[[0, 2]]))

    # 3) Final direct route image + reports.
    nav.save_trajectory_snapshot("route_start_to_goal.png")
    nav.finalize()

    # Geodesic reference path for reporting.
    import habitat_sim

    island = int(sim.pathfinder.get_island(start_pos))
    snapped_goal = np.asarray(sim.pathfinder.snap_point(point=goal, island_index=island), dtype=float).reshape(3)
    sp = habitat_sim.ShortestPath()
    sp.requested_start = start_pos
    sp.requested_end = snapped_goal
    has_path = bool(sim.pathfinder.find_path(sp))
    geodesic = float(sp.geodesic_distance) if has_path else None
    route_points = [np.asarray(p, dtype=float).reshape(3).tolist() for p in (sp.points or [])] if has_path else []

    keys_payload = {
        "sentence": ctx.sentence,
        "target_instance": cli.instance_id,
        "explore_keys": ["a"] * int(cli.explore_turns),
        "face_target_keys": face_keys,
        "look_down_keys": tilt_keys,
        "navigation_keys": nav_keys,
        "all_keys_in_order": keys,
        "all_keys_compressed": _compress_keys(keys),
        "navigation_keys_compressed": _compress_keys(nav_keys),
        "key_legend": {"w": "move_forward 0.25m", "a": "turn_left 30deg", "d": "turn_right 30deg",
                       "o": "look_up", "p": "look_down", "s": "turn_around(=a x6)", "e": "move_forward 1m(=w x4)"},
        "total_key_count": len(keys),
    }
    with open(out_dir / "manual_navigation_keys.json", "w", encoding="utf-8") as f:
        json.dump(keys_payload, f, ensure_ascii=False, indent=2)

    summary = {
        "scene_name": ctx.scene_name,
        "episode_id": int(ctx.episode_id),
        "sentence": ctx.sentence,
        "target_instance": cli.instance_id,
        "start_position": start_pos.tolist(),
        "goal_object_position": goal.tolist(),
        "snapped_goal_position": snapped_goal.tolist(),
        "end_position": end_pos.tolist(),
        "final_planar_distance_to_goal_m": planar_to_goal,
        "geodesic_distance_start_to_goal_m": geodesic,
        "geodesic_route_points": route_points,
        "step_count": int(nav.step_count),
        "navigation_action_count": len(nav_keys),
        "planner": follow_log,
        "frames_dir": str(frames_dir.relative_to(out_dir)),
        "route_image": "trajectory/route_start_to_goal.png",
    }
    with open(out_dir / "navigation_summary.json", "w", encoding="utf-8") as f:
        json.dump(m._jsonable(summary), f, ensure_ascii=False, indent=2)

    sim.close()
    print(f"[nav] DONE. end={end_pos.tolist()} planar_to_goal={planar_to_goal:.3f}m "
          f"geodesic={geodesic} frames={frame_idx[0]} keys={len(keys)}", flush=True)
    print(f"[nav] route image: {out_dir / 'trajectory' / 'route_start_to_goal.png'}", flush=True)


if __name__ == "__main__":
    main()
