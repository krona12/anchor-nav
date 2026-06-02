"""Simulated *real* navigation flow that explores its way to the target.

Unlike navigate_to_target.py (which walks a near-straight follower route), this
reproduces the original RefHM3D decision loop so the exploration looks real:

  repeat until close to the target object:
    1. DECISION ROUND  -- stop and scan 12 views (full 360 turn). This reveals
       the fog-of-war in every direction, so the TopDownMap "shadow" grows at
       every stop (the narrow 42deg FOV barely updates it during plain forward
       motion -- that was the "shadow not updating" symptom).
    2. VLE DECISION    -- detect frontiers + run the PQ3D model decision
       (frontier / final), logged exactly like the batch pipeline.
    3. ADVANCE         -- step a chunk along the known geodesic route toward the
       real target, updating fog continuously.
  then face the target and tilt down so it is centered.

A side-by-side "camera | TopDownMap" composite is saved for EVERY primitive
step (scan views included) so the fog/shadow update is visible frame by frame.

Outputs (under the run log dir):
  exploration_frames/NNNNN_<event>.png   per-step camera|topdown composite
  decisions/dec_XXX/...                   per-decision scan + frontier + maps
  trajectory/route_start_to_goal.png      final route (path/start/goal)
  manual_navigation_keys.json             full + per-phase key sequence
  navigation_summary.json                 distances, decisions, route points
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
from typing import Any, List, Optional

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
TELEOP_PATH = CODE_DIR / "interactive_vista2mqsc_teleop.py"

ACTION_TO_KEY = {"move_forward": "w", "turn_left": "a", "turn_right": "d", "look_up": "o", "look_down": "p"}


def _load_teleop() -> Any:
    spec = importlib.util.spec_from_file_location("teleop_mod", str(TELEOP_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["teleop_mod"] = module
    spec.loader.exec_module(module)
    return module


def _compress(keys: List[str]) -> str:
    if not keys:
        return ""
    out, prev, n = [], keys[0], 1
    for k in keys[1:]:
        if k == prev:
            n += 1
        else:
            out.append(f"{prev}x{n}" if n > 1 else prev)
            prev, n = k, 1
    out.append(f"{prev}x{n}" if n > 1 else prev)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--segment_advance_m", type=float, default=2.0, help="planar distance to advance between decision rounds")
    ap.add_argument("--arrive_thresh_m", type=float, default=0.7, help="stop when this close (planar) to the target")
    ap.add_argument("--max_rounds", type=int, default=12)
    ap.add_argument("--disable_pq3d", action="store_true", help="use nearest-frontier fallback instead of the PQ3D VLE model")
    ap.add_argument("--logs_dir", default=str(CODE_DIR.parent / "logs" / "explore_nav"))
    ap.add_argument("--live_dir", default=str(CODE_DIR.parent / "logs" / "live"))
    cli = ap.parse_args()

    m = _load_teleop()

    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cli.logs_dir) / f"run={run_id}_ep{cli.episode_id}_{cli.instance_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "teleop",
        "--scene_name", cli.scene_name,
        "--episode_id", str(cli.episode_id),
        "--navigation_type", cli.navigation_type,
        "--instance_id", cli.instance_id,
        "--task_id", str(cli.task_id),
        "--headless",
        "--disable_vista2mqsc_refine",  # PQ3D alone does the VLE decisions
        "--logs_dir", str(out_dir),
        "--live_dir", str(cli.live_dir),
    ]
    if cli.disable_pq3d:
        argv.append("--disable_pq3d")
    sys.argv = argv
    args = m.parse_args()

    ctx = m.load_task_context(args)
    scene_path = m._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    sim, agent = m.build_interactive_simulator(args, scene_path)
    nav = m.InteractiveNavigator(args, ctx, sim, agent, scene_path, out_dir)

    frames_dir = out_dir / "exploration_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    start_pos = np.asarray(agent.get_state().position, dtype=float).reshape(3).copy()
    goal = np.asarray(ctx.goal_positions[0], dtype=float).reshape(3)

    print(f"[explore] sentence : {ctx.sentence}", flush=True)
    print(f"[explore] start    : {start_pos.tolist()}", flush=True)
    print(f"[explore] goal(obj): {goal.tolist()}", flush=True)
    print(f"[explore] out_dir  : {out_dir}", flush=True)

    frame_idx = [0]

    def save_frame(event: str) -> None:
        if len(nav.context_buffer) == 0:
            return
        rgb = nav.context_buffer[-1][0]
        d = float(np.linalg.norm((np.asarray(agent.get_state().position) - goal)[[0, 2]]))
        rgb_bgr, top_bgr = nav._compose_frames({"color_sensor": rgb}, status=f"{event} | to_goal={d:.2f}m")
        h = rgb_bgr.shape[0]
        scale = h / top_bgr.shape[0]
        top_resized = cv2.resize(top_bgr, (int(top_bgr.shape[1] * scale), h))
        composite = cv2.hconcat([rgb_bgr, top_resized])
        cv2.imwrite(str(frames_dir / f"{frame_idx[0]:05d}_{event}.png"), composite)
        frame_idx[0] += 1

    # Capture a composite frame after EVERY sim observation (scan views + every
    # move step) by wrapping the navigator's per-step recorder.
    _orig_record = nav._record_current_observation

    def _record_and_capture(event: str, obs: Optional[dict] = None) -> dict:
        info = _orig_record(event, obs)
        save_frame(event)
        return info

    nav._record_current_observation = _record_and_capture  # type: ignore[assignment]

    save_frame("init")

    keys: List[str] = []
    decision_keys: List[str] = []
    move_keys: List[str] = []
    decisions_log: List[dict] = []

    def planar_to_goal() -> float:
        return float(np.linalg.norm((np.asarray(agent.get_state().position) - goal)[[0, 2]]))

    rounds = 0
    while planar_to_goal() > cli.arrive_thresh_m and rounds < cli.max_rounds:
        # 1) + 2) Stop, scan 12 views (fog updates 360deg), run PQ3D/VLE decision.
        print(f"[explore] === round {rounds}: decision scan @ to_goal={planar_to_goal():.2f}m ===", flush=True)
        nav.run_decision_round()
        keys.extend(["a"] * 12)
        decision_keys.extend(["a"] * 12)
        if nav.latest_decision_payload is not None:
            p = nav.latest_decision_payload
            decisions_log.append({
                "decision_num": p.get("decision_num"),
                "is_final": p.get("is_final"),
                "frontier_count": len(p.get("frontiers", [])),
                "selected_frontier_idx": p.get("selected_frontier_idx"),
                "pq3d_target": p.get("target_used"),
                "to_goal_after_scan_m": planar_to_goal(),
            })

        # 3) Advance a chunk along the geodesic route toward the real target.
        nav.current_target = goal.copy()
        nav.current_target_is_final = True
        actions, follow_log = nav._plan_follow_actions(goal)
        seg_start = np.asarray(agent.get_state().position, dtype=float).reshape(3)
        advanced = 0.0
        n_moves = 0
        for action in actions:
            if not action:
                continue
            nav.step_action(str(action), 1, status_prefix=f"goto[r{rounds}]")
            key = ACTION_TO_KEY.get(str(action), "?")
            keys.append(key)
            move_keys.append(key)
            n_moves += 1
            advanced = float(np.linalg.norm((np.asarray(agent.get_state().position) - seg_start)[[0, 2]]))
            if planar_to_goal() <= cli.arrive_thresh_m or advanced >= cli.segment_advance_m:
                break
        print(f"[explore] round {rounds}: advanced {advanced:.2f}m in {n_moves} steps "
              f"(planner={follow_log.get('planner')}, actions={len(actions)})", flush=True)
        rounds += 1

    # Final decision scan at the target, for completeness.
    print(f"[explore] arrived (to_goal={planar_to_goal():.2f}m). Final decision scan.", flush=True)
    nav.run_decision_round()
    keys.extend(["a"] * 12)
    decision_keys.extend(["a"] * 12)

    # Face the target, then tilt down to center the close, low object.
    import habitat_sim.utils.common as _hsu

    def _heading_err_deg() -> float:
        st = agent.get_state()
        f = _hsu.quat_rotate_vector(st.rotation, np.array([0.0, 0.0, -1.0]))
        to = goal - np.asarray(st.position, dtype=float).reshape(3)
        return math.degrees(math.atan2(float(f[0]) * float(to[2]) - float(f[2]) * float(to[0]),
                                       float(f[0]) * float(to[0]) + float(f[2]) * float(to[2])))

    face_keys: List[str] = []
    if abs(_heading_err_deg()) > 20.0:
        before = abs(_heading_err_deg())
        nav.step_action("turn_left", 1, status_prefix="face")
        keys.append("a"); face_keys.append("a")
        turn = "turn_left" if abs(_heading_err_deg()) < before else "turn_right"
        for _ in range(11):
            if abs(_heading_err_deg()) <= 20.0:
                break
            nav.step_action(turn, 1, status_prefix="face")
            k = ACTION_TO_KEY[turn]
            keys.append(k); face_keys.append(k)

    st = agent.get_state()
    to = goal - np.asarray(st.position, dtype=float).reshape(3)
    planar = float(np.linalg.norm(to[[0, 2]]))
    drop = float(st.position[1]) + 1.31 - float(goal[1])
    n_tilt = max(0, min(3, int(round(math.degrees(math.atan2(max(drop, 0.0), max(planar, 1e-3))) / 30.0))))
    tilt_keys: List[str] = []
    for _ in range(n_tilt):
        nav.step_action("look_down", 1, status_prefix="face")
        keys.append("p"); tilt_keys.append("p")

    end_pos = np.asarray(agent.get_state().position, dtype=float).reshape(3)
    planar_final = float(np.linalg.norm((end_pos - goal)[[0, 2]]))

    nav.save_trajectory_snapshot("route_start_to_goal.png")
    nav.finalize()

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
        "phase_keys": {
            "decision_scans (each round = a x12)": decision_keys,
            "movement": move_keys,
            "face_target": face_keys,
            "look_down": tilt_keys,
        },
        "all_keys_in_order": keys,
        "all_keys_compressed": _compress(keys),
        "movement_keys_compressed": _compress(move_keys),
        "key_legend": {"w": "move_forward 0.25m", "a": "turn_left 30deg", "d": "turn_right 30deg",
                       "o": "look_up", "p": "look_down"},
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
        "final_planar_distance_to_goal_m": planar_final,
        "geodesic_distance_start_to_goal_m": geodesic,
        "geodesic_route_points": route_points,
        "decision_round_count": int(nav.decision_num),
        "step_count": int(nav.step_count),
        "total_frames": int(frame_idx[0]),
        "decisions": decisions_log,
        "frames_dir": str(frames_dir.relative_to(out_dir)),
        "route_image": "trajectory/route_start_to_goal.png",
    }
    with open(out_dir / "navigation_summary.json", "w", encoding="utf-8") as f:
        json.dump(m._jsonable(summary), f, ensure_ascii=False, indent=2)

    sim.close()
    print(f"[explore] DONE. decisions={nav.decision_num} steps={nav.step_count} frames={frame_idx[0]} "
          f"end={end_pos.tolist()} planar_to_goal={planar_final:.3f}m geodesic={geodesic}", flush=True)
    print(f"[explore] route image: {out_dir / 'trajectory' / 'route_start_to_goal.png'}", flush=True)


if __name__ == "__main__":
    main()
