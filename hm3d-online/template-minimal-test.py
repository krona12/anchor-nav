"""RefHM3D minimal template test.

This script keeps the single-scene / single-episode evaluation structure from
``refhm3d-nav-sequence-analyze-anchor-vfv.py`` while removing the VFV module
logic. It is intended as a clean template for plugging in a new module at the
decision hook without changing the surrounding navigation, metric, and logging
scaffold.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

_HM3D_ONLINE = Path(__file__).resolve().parent
_MTU3D_ROOT = _HM3D_ONLINE.parent
for _p in (_HM3D_ONLINE, _MTU3D_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

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


TASK_LEVEL_ORDER = ("object", "room", "region", "instance")


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


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _setup_run_logging(log_dir: Path) -> Path:
    _ensure_dir(log_dir)
    log_path = log_dir / f"template-minimal-test-{_now_tag()}-pid{os.getpid()}.log"
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[template] logging enabled -> {log_path.resolve()}")

    def _cleanup() -> None:
        try:
            print(f"[template] run finished, log saved -> {log_path.resolve()}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            log_fp.close()

    atexit.register(_cleanup)
    return log_path


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
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
    print(f"[template] seed={int(seed)}")


def _resolve_scene_mesh(scene_root: Path, scene_name: str) -> Path:
    sid = scene_name.split("-")[-1]
    cands = [
        scene_root / scene_name / f"{sid}.basis.glb",
        scene_root / scene_name / f"{sid}.glb",
        scene_root / scene_name / f"{sid}.basis.scene_instance.json",
        scene_root / scene_name / f"{sid}.scene_instance.json",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve scene asset for {scene_name} under {scene_root}")


def _build_sentence(
    task_type: str,
    cur_task: Dict[str, Any],
    goals_map: Dict[str, Any],
    region_map: Dict[str, Any],
    concise: bool,
) -> str:
    if task_type == "object":
        return cur_task["object_category"]
    if task_type == "room":
        return f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
    if task_type == "region":
        region_info = region_map[cur_task["region_id"]]
        desc = (
            region_info.get("shortest_description")
            or region_info.get("concise_description")
            or region_info.get("detailed_description")
            or ""
        ) if concise else (
            region_info.get("comprehensive_description")
            or region_info.get("detailed_description")
            or region_info.get("concise_description")
            or ""
        )
        return f"{cur_task['object_category']} in the {region_info['region_category'].lower()} that has {desc}"
    if task_type == "instance":
        inst = goals_map[cur_task["instance_id"]]
        if concise:
            return inst.get("annot_unique_concise_description") or ""
        return (
            inst.get("annot_unique_detailed_description")
            or inst.get("annot_unique_normal_description")
            or inst.get("annot_appearance_description")
            or ""
        )
    raise ValueError(f"unknown task_type={task_type}")


def _capture_scan_frames(
    *,
    sim: Any,
    agent: Any,
    top_down_map: np.ndarray,
    fog: np.ndarray,
    vis_dist: int,
    total_steps: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], np.ndarray, int]:
    scan_rgb: List[np.ndarray] = []
    scan_depth: List[np.ndarray] = []
    scan_state: List[Any] = []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        st_now = agent.get_state()
        rgb = obs["color_sensor"][:, :, :3]
        dep = obs["depth_sensor"][:, :]
        scan_rgb.append(rgb)
        scan_depth.append(dep)
        scan_state.append(st_now)
        fog[:] = reveal_fog_of_war(
            top_down_map=top_down_map,
            current_fog_of_war_mask=fog,
            current_point=map_coors_to_pixel(st_now.position, top_down_map, sim),
            current_angle=get_polar_angle(st_now),
            fov=42,
            max_line_len=vis_dist,
            enable_debug_visualization=False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return scan_rgb, scan_depth, scan_state, fog, total_steps


def _follow_target(
    *,
    pf: Any,
    agent: Any,
    sim: Any,
    used_target: np.ndarray,
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float, Dict[str, Any]]:
    agent_island = pf.get_island(agent.get_state().position)
    target_nav = pf.snap_point(point=used_target, island_index=agent_island)
    follower = habitat_sim.GreedyGeodesicFollower(
        pf,
        agent,
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )
    follow_log: Dict[str, Any] = {
        "ok": True,
        "raw_target": np.asarray(used_target, dtype=float).reshape(3).tolist(),
        "snapped_target": np.asarray(target_nav, dtype=float).reshape(3).tolist(),
        "action_count": 0,
    }
    try:
        actions = follower.find_path(target_nav)
    except Exception as e:
        follow_log.update({"ok": False, "error_type": type(e).__name__, "error_message": str(e)})
        actions = []

    goto_rgb: List[np.ndarray] = []
    goto_depth: List[np.ndarray] = []
    goto_state: List[Any] = []
    for a in actions:
        if not a:
            continue
        obs = sim.step(action=a)
        st2 = agent.get_state()
        goto_rgb.append(obs["color_sensor"][:, :, :3])
        goto_depth.append(obs["depth_sensor"][:, :])
        goto_state.append(st2)
        episode_cum_distance += float(np.linalg.norm(st2.position - prev_agent_state.position))
        prev_agent_state = st2
        total_steps += 1
        follow_log["action_count"] = int(follow_log["action_count"]) + 1
        if total_steps >= int(max_steps):
            break
    return goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_log


def _summarize_tasks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        by_level = {lv: {"count": 0, "sr": 0.0, "spl": 0.0} for lv in TASK_LEVEL_ORDER}
        return {"count": 0, "sr": 0.0, "spl": 0.0, "by_level": by_level}

    sr = float(np.mean([float(x.get("sr", 0.0)) for x in rows]))
    spl = float(np.mean([float(x.get("spl", 0.0)) for x in rows]))
    by_level: Dict[str, Dict[str, float]] = {}
    for lv in TASK_LEVEL_ORDER:
        bucket = [x for x in rows if str(x.get("task_level", "")) == lv]
        by_level[lv] = {
            "count": int(len(bucket)),
            "sr": float(np.mean([float(x.get("sr", 0.0)) for x in bucket])) if bucket else 0.0,
            "spl": float(np.mean([float(x.get("spl", 0.0)) for x in bucket])) if bucket else 0.0,
        }
    return {"count": int(len(rows)), "sr": sr, "spl": spl, "by_level": by_level}


def main() -> None:
    parser = argparse.ArgumentParser(
        "Minimal RefHM3D template test: PQ3D + frontier navigation scaffold, without VFV module logic."
    )
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--description_mode", choices=["detailed", "concise"], default="detailed")
    parser.add_argument("--task_levels", type=str, default="object,room,region,instance")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--navigation_data_path", type=str, default="LangMap_Annotations")
    parser.add_argument("--hm3d_data_base_path", type=str, default="datascene")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--pq3d_stage1_path", type=str, default="checkpoint/stage1-pretrain-all")
    parser.add_argument("--pq3d_stage2_path", type=str, default="checkpoint/stage2-fine-tune-goat")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--success_distance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_root", type=str, default="./output_logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_tag = args.run_tag or _now_tag()
    out_root: Optional[Path] = None
    if not bool(args.quiet):
        out_root = _ensure_dir(Path(args.output_root).expanduser().resolve() / f"{run_tag}-template-minimal-test")
        _setup_run_logging(out_root)

    _set_seed(int(args.seed))
    enabled_task_levels = {x.strip() for x in str(args.task_levels).split(",") if x.strip()}
    if not enabled_task_levels:
        enabled_task_levels = set(TASK_LEVEL_ORDER)
    print(
        f"[template] run_tag={run_tag} scene={args.scene_name} episode={args.episode_id} "
        f"task_range=[{args.task_id},{int(args.task_id) + int(args.num_tasks)}) "
        f"levels={sorted(enabled_task_levels)} output={str(out_root) if out_root else '(quiet)'}"
    )
    if out_root is not None:
        _write_json(out_root / "run_args.json", vars(args))

    scene_file = (project_root / args.navigation_data_path / f"{args.scene_name}.json.gz").resolve()
    with gzip.open(scene_file, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    region_map = scene_data["region_annotation"]
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    goals_map = {x["object_id"]: x for x in scene_data["goals"]}
    eps_matches = [e for e in scene_data["episode_by_sequence"] if int(e["episode_id"]) == int(args.episode_id)]
    if not eps_matches:
        raise RuntimeError(f"episode_id={args.episode_id} not found in {scene_file}")
    eps = eps_matches[0]
    task_sequence = eps["task_sequence"]
    task_end = min(int(args.task_id) + int(args.num_tasks), len(task_sequence))

    sim_settings = OmegaConf.load(str((project_root / args.sim_config).resolve()))
    agent_settings = OmegaConf.load(str((project_root / args.agent_config).resolve()))
    sim_settings["scene"] = str(_resolve_scene_mesh((project_root / args.hm3d_data_base_path).resolve(), args.scene_name))
    abstract_sim = HabitatSimulator(sim_settings, agent_settings)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    pf = sim.pathfinder
    st = habitat_sim.AgentState()
    st.position = eps["start_position"]
    st.rotation = eps["start_rotation"]
    agent.set_state(st)

    summaries: List[Dict[str, Any]] = []
    try:
        pq3d = PQ3DModel(
            str((project_root / args.pq3d_stage1_path).resolve()),
            str((project_root / args.pq3d_stage2_path).resolve()),
            min_decision_num=args.decision_num_min,
        )
        pq3d.reset()

        map_resolution = 512
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
        fog = np.zeros_like(top_down_map)
        area_thr = convert_meters_to_pixel(9, map_resolution, sim)
        vis_dist = convert_meters_to_pixel(3.0, map_resolution, sim)
        visited_frontier: set = set()

        for loop_tid in range(int(args.task_id), task_end):
            total_steps = 0
            decision_num = 0
            task_type, task_idx = task_sequence[loop_tid]
            if task_type not in enabled_task_levels:
                print(f"[template][task-skip] task={loop_tid} level={task_type} disabled")
                continue

            cur_task = episode_mapping[task_type][task_idx]
            sentence = _build_sentence(
                task_type,
                cur_task,
                goals_map,
                region_map,
                concise=(args.description_mode == "concise"),
            )
            print(f"[template][task-start] task={loop_tid} level={task_type} sentence={sentence!r}")

            goals_ids = list(cur_task.get("target_object_ids", []))
            goals = [goals_map[x] for x in goals_ids if x in goals_map]
            out_task = (
                _ensure_dir(out_root / f"scene={args.scene_name}" / f"episode={args.episode_id}" / f"task={loop_tid}")
                if out_root is not None
                else None
            )

            goto_rgb: List[np.ndarray] = []
            goto_depth: List[np.ndarray] = []
            goto_state: List[Any] = []
            prev_agent_state = agent.get_state()
            sub_episode_start_position = np.asarray(prev_agent_state.position, dtype=float).copy()
            episode_cum_distance = 0.0
            task_end_reason = "max_steps"
            decision_records: List[Dict[str, Any]] = []

            while total_steps < int(args.max_steps):
                color_list: List[np.ndarray] = []
                depth_list: List[np.ndarray] = []
                state_list: List[Any] = []
                if len(goto_rgb) > 6:
                    step = max(1, len(goto_rgb) // 6)
                    goto_rgb = [goto_rgb[i] for i in range(0, len(goto_rgb), step)][:6]
                    goto_depth = [goto_depth[i] for i in range(0, len(goto_depth), step)][:6]
                    goto_state = [goto_state[i] for i in range(0, len(goto_state), step)][:6]
                color_list.extend(goto_rgb)
                depth_list.extend(goto_depth)
                state_list.extend(goto_state)

                scan_rgb, scan_depth, scan_state, fog, total_steps = _capture_scan_frames(
                    sim=sim,
                    agent=agent,
                    top_down_map=top_down_map,
                    fog=fog,
                    vis_dist=vis_dist,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                )
                color_list.extend(scan_rgb)
                depth_list.extend(scan_depth)
                state_list.extend(scan_state)
                if total_steps >= int(args.max_steps):
                    break

                st_now = agent.get_state()
                fw = detect_frontier_waypoints(
                    top_down_map,
                    fog,
                    area_thr,
                    xy=map_coors_to_pixel(st_now.position, top_down_map, sim)[::-1],
                    enable_visualization=False,
                )
                if len(fw) == 0:
                    frontiers = []
                else:
                    frontiers = pixel_to_map_coors(fw[:, ::-1], st_now.position, top_down_map, sim)
                frontiers = [w for w in frontiers if tuple(np.round(w, 1)) not in visited_frontier]

                target, is_final = pq3d.decision(color_list, depth_list, state_list, frontiers, sentence, decision_num)
                used_target = np.asarray(target, dtype=float).reshape(3).copy()
                aux = getattr(pq3d, "last_decision_aux", {}) or {}
                print(
                    f"[template][decision] task={loop_tid} dec={decision_num} frontiers={len(frontiers)} "
                    f"target={used_target.tolist()} final={bool(is_final)}"
                )

                if not bool(is_final):
                    visited_frontier.add(tuple(np.round(used_target, 1)))

                goto_rgb, goto_depth, goto_state, prev_agent_state, total_steps, episode_cum_distance, follow_log = _follow_target(
                    pf=pf,
                    agent=agent,
                    sim=sim,
                    used_target=used_target,
                    prev_agent_state=prev_agent_state,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                    episode_cum_distance=float(episode_cum_distance),
                )

                record = {
                    "task_id": int(loop_tid),
                    "decision_num": int(decision_num),
                    "module": "none",
                    "module_hook_note": "Insert module logic here if needed; template currently uses raw PQ3D target.",
                    "is_final": bool(is_final),
                    "frontier_count": int(len(frontiers)),
                    "target_used": used_target.tolist(),
                    "pq3d_aux": aux,
                    "follow": follow_log,
                    "steps_total_after_follow": int(total_steps),
                }
                decision_records.append(record)
                if out_task is not None:
                    _write_json(out_task / f"dec_{decision_num:03d}_template.json", record)

                decision_num += 1
                if not bool(follow_log.get("ok", False)):
                    task_end_reason = "follower_error"
                    break
                if bool(is_final):
                    task_end_reason = "final_decision"
                    break

            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])]
            if len(view_points) > 0:
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = sub_episode_start_position
                path.requested_ends = view_points
                start_end_geo_distance = float(path.geodesic_distance) if pf.find_path(path) else float("inf")
                path = habitat_sim.MultiGoalShortestPath()
                path.requested_start = agent_state.position
                path.requested_ends = view_points
                agent_end_geo_distance = float(path.geodesic_distance) if pf.find_path(path) else float("inf")
            else:
                start_end_geo_distance = float("inf")
                agent_end_geo_distance = float("inf")

            if task_end_reason == "follower_error" or np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
                sr = 0.0
                spl = 0.0
            else:
                sr = 1.0 if agent_end_geo_distance <= float(args.success_distance) else 0.0
                spl = float(sr * start_end_geo_distance / max(start_end_geo_distance, max(episode_cum_distance, 1e-12)))

            row = {
                "scene_name": args.scene_name,
                "episode_id": int(args.episode_id),
                "task_id": int(loop_tid),
                "task_level": task_type,
                "sentence": sentence,
                "sr": float(sr),
                "spl": float(spl),
                "start_end_geo_distance": float(start_end_geo_distance),
                "agent_end_geo_distance": float(agent_end_geo_distance),
                "episode_cum_distance": float(episode_cum_distance),
                "steps_total": int(total_steps),
                "decisions": int(decision_num),
                "end_reason": task_end_reason,
                "decision_records": decision_records,
            }
            summaries.append(row)
            if out_task is not None:
                _write_json(out_task / "summary.json", row)

            live = _summarize_tasks(summaries)
            print(
                f"[template][task-summary] task={loop_tid} level={task_type} steps_total={total_steps} "
                f"decisions={decision_num} end_reason={task_end_reason} SR={sr:.1f} SPL={spl:.4f}"
            )
            print(
                f"[template][live-metrics] count={live['count']} sr={live['sr']:.6f} spl={live['spl']:.6f} "
                f"by_level={json.dumps(live['by_level'], ensure_ascii=False, sort_keys=True)}"
            )
    finally:
        sim.close()

    metrics = _summarize_tasks(summaries)
    run_summary = {
        "run_tag": run_tag,
        "script": str(Path(__file__).resolve()),
        "scene_name": args.scene_name,
        "episode_id": int(args.episode_id),
        "task_start": int(args.task_id),
        "task_end": int(task_end),
        "metrics": metrics,
        "tasks": summaries,
    }
    if out_root is not None:
        _write_json(out_root / "run_summary.json", run_summary)
    print(
        f"[template][run-summary] count={metrics['count']} sr={metrics['sr']:.6f} "
        f"spl={metrics['spl']:.6f} by_level={json.dumps(metrics['by_level'], ensure_ascii=False, sort_keys=True)}"
    )
    print("[template] done")


if __name__ == "__main__":
    main()
