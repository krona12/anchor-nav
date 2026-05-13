import gzip
import os
import sys
import atexit
import datetime
import random
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from habitat.utils.visualizations import maps
import json
import habitat_sim
import numpy as np
import torch
from omegaconf import OmegaConf
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)
import cv2
from data_utils import PQ3DModel
from tqdm import tqdm
import time
import argparse


def sequence_compute_metric_results(result_dict: dict) -> None:
    sequence_results = result_dict.get("sequence", [])
    total_count = len(sequence_results)
    if total_count == 0:
        print("[Metrics] sequence count: 0")
        return

    # Navigation follower errors are valid failed trials and must remain in
    # the denominator. Decision/model exceptions are raised before appending
    # a row, so they cannot silently pollute SR/SPL.
    total_sr = sum(float(item.get("sr", 0)) for item in sequence_results)
    total_spl = sum(float(item.get("spl", 0)) for item in sequence_results)
    avg_sr = total_sr / total_count
    avg_spl = total_spl / total_count
    invalid_count = sum(1 for item in sequence_results if not bool(item.get("valid_for_metric", True)))
    follower_error_count = sum(1 for item in sequence_results if item.get("end_reason") == "follower_error")
    print(
        f"[Metrics] sequence count: {total_count}, invalid_count: {invalid_count}, "
        f"follower_error_count: {follower_error_count}, avg_sr: {avg_sr:.6f}, avg_spl: {avg_spl:.6f}"
    )


TASK_LEVEL_ORDER = ("object", "room", "region", "instance")


def _baseline_metric_snapshot(result_dict: dict) -> dict:
    rows = result_dict.get("sequence", [])
    avg_sr = sum(float(item.get("sr", 0.0)) for item in rows) / len(rows) if rows else 0.0
    avg_spl = sum(float(item.get("spl", 0.0)) for item in rows) / len(rows) if rows else 0.0
    by_level = {}
    for item in rows:
        level = str(item.get("task_level", "unknown"))
        bucket = by_level.setdefault(level, {"count": 0, "sr_sum": 0.0, "spl_sum": 0.0})
        bucket["count"] += 1
        bucket["sr_sum"] += float(item.get("sr", 0.0))
        bucket["spl_sum"] += float(item.get("spl", 0.0))

    ordered_levels = list(TASK_LEVEL_ORDER)
    ordered_levels.extend(sorted(level for level in by_level if level not in TASK_LEVEL_ORDER))
    by_level_out = {}
    for level in ordered_levels:
        bucket = by_level.get(level, {"count": 0, "sr_sum": 0.0, "spl_sum": 0.0})
        count = int(bucket["count"])
        by_level_out[level] = {
            "count": count,
            "avg_sr": float(bucket["sr_sum"] / count) if count else 0.0,
            "avg_spl": float(bucket["spl_sum"] / count) if count else 0.0,
        }

    return {
        "rows": len(rows),
        "avg_sr": float(avg_sr),
        "avg_spl": float(avg_spl),
        "follower_error_count": sum(1 for item in rows if item.get("end_reason") == "follower_error"),
        "by_level": by_level_out,
    }


def _format_baseline_metric_snapshot(snapshot: dict) -> str:
    level_parts = []
    for level in TASK_LEVEL_ORDER:
        metrics = snapshot["by_level"].get(level, {"count": 0, "avg_sr": 0.0, "avg_spl": 0.0})
        level_parts.append(
            f"{level}:n={metrics['count']},sr={metrics['avg_sr']:.6f},spl={metrics['avg_spl']:.6f}"
        )
    return (
        f"rows={snapshot['rows']} avg_sr={snapshot['avg_sr']:.6f} avg_spl={snapshot['avg_spl']:.6f} "
        f"follower_error_count={snapshot['follower_error_count']} levels=[{'; '.join(level_parts)}]"
    )


def baseline_live_metric_log(
    result_dict: dict,
    *,
    scene_name: str,
    episode_id: int,
    task_id: int,
    task_level: str,
    metrics_log_path: str,
) -> None:
    snapshot = _baseline_metric_snapshot(result_dict)
    snapshot.update(
        {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "latest_scene": scene_name,
            "latest_episode": int(episode_id),
            "latest_task": int(task_id),
            "latest_level": task_level,
        }
    )
    formatted = _format_baseline_metric_snapshot(snapshot)
    print(
        "[BASELINE_LIVE_METRICS] "
        f"{formatted} "
        f"latest_scene={scene_name} latest_episode={episode_id} latest_task={task_id} latest_level={task_level} "
        f"by_level={json.dumps(snapshot['by_level'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write("[BASELINE_LIVE_METRICS] " + formatted + "\n")
        f.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def set_reproducibility_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[baseline] reproducibility_seed={seed}")


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [
        scene_dir / f"{short_scene_name}.basis.glb",
        scene_dir / f"{short_scene_name}.glb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Scene asset not found for {scene_name}. Checked: {[str(x) for x in candidates]}"
    )


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-baseline-{ts}-pid{pid}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    print(f"[Baseline] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[Baseline] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fp.close()

    atexit.register(_cleanup)


program_start = time.time()
old_print = print


def print(*args, **kwargs):
    now = time.time()
    elapsed = now - program_start
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    old_print(f"Elapsed {elapsed_str}", *args, **kwargs)


parser = argparse.ArgumentParser(description="Run RefHM3D baseline evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0, help="Dataset start ratio")
parser.add_argument("--end_ratio", type=float, default=0.2, help="Dataset end ratio")
parser.add_argument("--concise_description", action="store_true", help="Use concise descriptions")
parser.add_argument(
    "--navigation_data_path",
    type=str,
    default="/home/chenlin/krona/anchor-nav/LangMap_Annotations",
    help="Path to RefHM3D sequence dataset root (recursive search for *.json.gz)",
)
parser.add_argument(
    "--hm3d_data_base_path",
    type=str,
    default="/home/chenlin/krona/MTU3D/datascene",
    help="Path to HM3D scene folder",
)
parser.add_argument(
    "--pq3d_stage1_path",
    type=str,
    default="/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all",
)
parser.add_argument(
    "--pq3d_stage2_path",
    type=str,
    default="/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-goat",
)
parser.add_argument(
    "--output_log_dir",
    type=str,
    default="./output_logs/baseline",
    help="Output directory for both logs and metric json",
)
parser.add_argument(
    "--task_levels",
    type=str,
    default="object,room,region,instance",
    help="Comma-separated task levels to execute, e.g. instance or region,instance",
)
parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible baseline/ACSD comparison")
args = parser.parse_args()

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)
set_reproducibility_seed(args.seed)
print(
    "[baseline] 输出 JSON 每条记录含 task_level（object|room|region|instance），"
    "与 vlmcore refine 及 test_scripts/aggregate_shard_results.py --by-level 一致"
)
enabled_task_levels = {x.strip() for x in args.task_levels.split(",") if x.strip()}
if not enabled_task_levels:
    enabled_task_levels = {"object", "room", "region", "instance"}
print(f"[baseline] enabled_task_levels={sorted(enabled_task_levels)}")

black_task_ids = []
print(f"NUMBER OF BLACK IDS: {len(black_task_ids)}")

hm3d_data_base_path = os.path.expanduser(args.hm3d_data_base_path)
pq3d_stage1_path = os.path.expanduser(args.pq3d_stage1_path)
pq3d_stage2_path = os.path.expanduser(args.pq3d_stage2_path)
enable_visualization = False
decision_num_min = 3
visible_radius = 3

concise_description_tag = args.concise_description
start_ratio, end_ratio = args.start_ratio, args.end_ratio
navigation_data_path = os.path.expanduser(args.navigation_data_path)
navigation_data_root = Path(navigation_data_path)
os.makedirs(output_log_dir, exist_ok=True)
if concise_description_tag:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_concisedesc_{start_ratio}_{end_ratio}.json")
else:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_{start_ratio}_{end_ratio}.json")
metrics_log_path = os.path.join(output_log_dir, f"baseline_live_metrics_{start_ratio}_{end_ratio}.log")
print(f"[baseline] live metrics log -> {os.path.abspath(metrics_log_path)}")

scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
if len(scene_data_paths) == 0:
    raise FileNotFoundError(
        "No *.json.gz files found recursively under navigation_data_path: "
        f"{navigation_data_path}. "
        "Please pass the RefHM3D sequence dataset root folder."
    )
num_scene = len(scene_data_paths)
scene_data_paths = scene_data_paths[int(start_ratio * num_scene):int(end_ratio * num_scene)]
scene_data_list_for_print = [p.name for p in scene_data_paths]
print(f"\n\nTotal selected number of scenes {len(scene_data_paths)}: {scene_data_list_for_print}\n\n")

if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    sequence_compute_metric_results(result_dict)
    baseline_live_metric_log(
        result_dict,
        scene_name="resume_existing_output",
        episode_id=-1,
        task_id=-1,
        task_level="resume",
        metrics_log_path=metrics_log_path,
    )
    existing_episodes = {
        "_".join([result["scene_name"], result["navigation_type"], str(result["episode_id"])])
        for goal_type in result_dict
        for result in result_dict[goal_type]
    }
else:
    existing_episodes = {}
    result_dict = {"sequence": []}

pq3d_model = PQ3DModel(pq3d_stage1_path, pq3d_stage2_path, min_decision_num=decision_num_min)

for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
    scene_data_file = scene_data_path.name
    scene_name = scene_data_file.split(".")[0]
    with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
        region_to_annot_dict = scene_data["region_annotation"]
        episode_mapping = {
            "object": scene_data["episodes_by_object_level"],
            "room": scene_data["episodes_by_room_level"],
            "region": scene_data["episodes_by_region_level"],
            "instance": scene_data["episodes_by_instance_level"],
        }
        all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    print(
        f"\n\n\n\n\n*************************** Processing {scene_data_file}, "
        f"Total Episode {len(scene_data['episode_by_sequence'])} ***************************"
    )
    for episode_idx, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
        pq3d_model.reset()
        global_color_list = []
        decision_num = 0
        visited_frontier_set = set()

        start_position = cur_episode["start_position"]
        start_rotation = cur_episode["start_rotation"]
        episode_id, navigation_type = cur_episode["episode_id"], cur_episode["navigation_type"]
        if "_".join([scene_name, navigation_type, str(episode_id)]) in existing_episodes:
            print("_".join([scene_name, navigation_type, str(episode_id)]), " already processed, skipped")
            continue

        sim_settings = OmegaConf.load("configs/habitat/goat_sim_config.yaml")
        goat_agent_setting = OmegaConf.load("configs/habitat/goat_agent_config.yaml")
        sim_settings["scene"] = resolve_scene_path(hm3d_data_base_path, scene_name)
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder

        map_resolution = 512
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels = convert_meters_to_pixel(9, map_resolution, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)

        for idx, cur_task in enumerate(cur_episode["task_sequence"]):
            task_type, task_idx = cur_task
            if task_type not in enabled_task_levels:
                continue
            cur_task = episode_mapping[task_type][task_idx]

            goals_ids = cur_task["target_object_ids"]
            assert len(goals_ids) > 0, f"{'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])} should have at least one goal"
            goals = [all_navigation_goals_dict[x] for x in goals_ids]
            if task_type == "object":
                sentence = cur_task["object_category"]
                goal_category = cur_task["object_category"]
            elif task_type == "room":
                sentence = f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
                goal_category = cur_task["object_category"]
            elif task_type == "region":
                region_desc = (
                    region_to_annot_dict[cur_task["region_id"]]["concise_description"]
                    if concise_description_tag
                    else region_to_annot_dict[cur_task["region_id"]]["detailed_description"]
                )
                sentence = (
                    f"{cur_task['object_category']} in the {region_to_annot_dict[cur_task['region_id']]['region_category'].lower()} "
                    f"that has {region_desc}"
                )
                goal_category = cur_task["object_category"]
            elif task_type == "instance":
                sentence = (
                    all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"]
                    if concise_description_tag
                    else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
                )
                goal_category = goals[0]["object_category"]
            print(f"\n\nBegin to process [{'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])}] type: [{task_type}], Question: [{sentence}]\n")

            total_steps, rotation_steps = 0, 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            print(f"Current start position is {sub_episode_start_position}")
            episode_cum_distance = 0
            goto_color_list = []
            goto_depth_list = []
            goto_agent_state_list = []
            task_end_reason = "max_steps"
            follower_error_info = None

            t_episode_start = time.perf_counter()
            while total_steps < 400:
                color_list = []
                depth_list = []
                agent_state_list = []
                if len(goto_color_list) > 6:
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)

                action_list = ["turn_left"] * 12
                for action in action_list:
                    obervations = sim.step(action=action)
                    color = obervations["color_sensor"][:, :, :3]
                    color_list.append(color)
                    global_color_list.append(color)
                    depth = obervations["depth_sensor"][:, :]
                    depth_list.append(depth)
                    agent_state = agent.get_state()
                    agent_state_list.append(agent_state)
                    if enable_visualization:
                        cv2.imwrite("color.png", color)
                    fog_of_war_mask = reveal_fog_of_war(
                        top_down_map=top_down_map,
                        current_fog_of_war_mask=fog_of_war_mask,
                        current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                        current_angle=get_polar_angle(agent_state),
                        fov=42,
                        max_line_len=visibility_dist_in_pixels,
                        enable_debug_visualization=enable_visualization,
                    )
                    total_steps += 1
                    rotation_steps += 1

                agent_state = agent.get_state()
                frontier_waypoints = detect_frontier_waypoints(
                    top_down_map,
                    fog_of_war_mask,
                    area_thres_in_pixels,
                    xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                    enable_visualization=enable_visualization,
                )
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = frontier_waypoints[:, ::-1]
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
                frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set]

                try:
                    target_position, is_final_decision = pq3d_model.decision(
                        color_list,
                        depth_list,
                        agent_state_list,
                        frontier_waypoints,
                        sentence,
                        decision_num,
                    )
                except Exception as e:
                    error_info = {
                        "scene_name": scene_name,
                        "episode_id": int(episode_id),
                        "task_id": int(idx),
                        "task_level": task_type,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "decision_num": int(decision_num),
                        "steps_total": int(total_steps),
                    }
                    error_path = os.path.join(output_log_dir, "baseline_decision_error.json")
                    with open(error_path, "w", encoding="utf-8") as f:
                        json.dump(error_info, f, ensure_ascii=False, indent=2)
                    print(
                        f"[baseline][decision-error] scene={scene_name} episode_id={episode_id} "
                        f"task_id={idx} task_level={task_type} error={type(e).__name__}: {e}"
                    )
                    raise RuntimeError(
                        f"baseline decision failed for scene={scene_name} episode={episode_id} task={idx}; "
                        f"wrote {error_path}"
                    ) from e
                decision_num += 1
                if not is_final_decision:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except Exception as e:
                    follower_error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "raw_target": np.asarray(target_position, dtype=float).reshape(3).tolist(),
                        "snapped_target": np.asarray(target_on_navmesh, dtype=float).reshape(3).tolist(),
                        "decision_num": int(decision_num - 1),
                        "is_final": bool(is_final_decision),
                    }
                    if not path_finder.is_navigable(target_on_navmesh):
                        print("Target is not navigable")
                    if not path_finder.is_navigable(agent_state.position):
                        print("Agent is not navigable")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent_state.position
                    path.requested_end = target_on_navmesh
                    if sim.pathfinder.find_path(path):
                        print(f"geodesic_distance: {path.geodesic_distance}")
                        follower_error_info["shortest_path_found"] = True
                        follower_error_info["shortest_path_geodesic_distance"] = float(path.geodesic_distance)
                    else:
                        print("cannt find path")
                        follower_error_info["shortest_path_found"] = False
                        follower_error_info["shortest_path_geodesic_distance"] = float("inf")
                    print(
                        f"[baseline][follow-error] scene={scene_name} episode_id={episode_id} task_id={idx} "
                        f"task_level={task_type} final={bool(is_final_decision)} "
                        f"error={follower_error_info['error_type']} path_found={follower_error_info.get('shortest_path_found')}"
                    )
                    action_list = []
                    task_end_reason = "follower_error"
                    break

                goto_color_list = []
                goto_depth_list = []
                goto_agent_state_list = []
                for action in action_list:
                    if action:
                        obervations = sim.step(action=action)
                        global_color_list.append(obervations["color_sensor"][:, :, :3])
                        agent_state = agent.get_state()
                        color = obervations["color_sensor"][:, :, :3]
                        depth = obervations["depth_sensor"][:, :]
                        goto_color_list.append(color)
                        goto_depth_list.append(depth)
                        goto_agent_state_list.append(agent_state)
                        fog_of_war_mask = reveal_fog_of_war(
                            top_down_map=top_down_map,
                            current_fog_of_war_mask=fog_of_war_mask,
                            current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                            current_angle=get_polar_angle(agent_state),
                            fov=42,
                            max_line_len=visibility_dist_in_pixels,
                            enable_debug_visualization=enable_visualization,
                        )
                        total_steps += 1
                        if action in ["turn_left", "turn_right"]:
                            rotation_steps += 1
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                if is_final_decision:
                    task_end_reason = "final_decision"
                    break

            t_episode_end = time.perf_counter()
            episode_time = float(t_episode_end - t_episode_start)
            if enable_visualization:
                height, width, layers = global_color_list[0].shape
                video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height))
                for color_frame in global_color_list:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                    video.write(color_frame)
                video.release()
                pq3d_model.representation_manager.save_colored_point_cloud()

            agent_state = agent.get_state()
            view_points = [view_point["agent_state"]["position"] for goal in goals for view_point in goal["view_points"]]
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = sub_episode_start_position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                start_end_geo_distance = path.geodesic_distance
            else:
                print(f"Goal is not navigable: {'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])}")
                start_end_geo_distance = np.inf
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                agent_end_geo_distance = path.geodesic_distance
            else:
                agent_end_geo_distance = np.inf

            if start_end_geo_distance == np.inf:
                raw_sr = 0
                raw_spl = 0
            elif agent_end_geo_distance == np.inf:
                raw_sr = 0
                raw_spl = 0
            else:
                raw_sr = bool(agent_end_geo_distance <= 0.25)
                raw_spl = raw_sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)
            valid_for_metric = True
            sr = 0 if task_end_reason == "follower_error" else raw_sr
            spl = 0 if task_end_reason == "follower_error" else raw_spl

            result_dict[navigation_type].append(
                {
                    "scene_name": scene_name,
                    "episode_id": episode_id,
                    "task_id": idx,
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "sr": sr,
                    "spl": spl,
                    "raw_sr_from_end_position": raw_sr,
                    "raw_spl_from_end_position": raw_spl,
                    "valid_for_metric": bool(valid_for_metric),
                    "end_reason": task_end_reason,
                    "steps_total": int(total_steps),
                    "decisions": int(decision_num),
                    "follower_error_info": follower_error_info,
                    "object_category": goal_category,
                    "task_time_sec": episode_time,
                }
            )
            print(
                f"===Episode_id {episode_id} task_id {idx} task_level={task_type}===\n"
                f"SR: {sr}, SPL: {spl}, Object category: {goal_category}, goal type: {navigation_type}===\n"
            )
            print(
                f"[baseline] task_level={task_type} scene={scene_name} episode={episode_id} "
                f"task={idx} sec={episode_time:.3f}"
            )
            with open(output_path, "w") as f:
                json.dump(result_dict, f)
            baseline_live_metric_log(
                result_dict,
                scene_name=scene_name,
                episode_id=int(episode_id),
                task_id=int(idx),
                task_level=str(task_type),
                metrics_log_path=metrics_log_path,
            )

        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)
    sequence_compute_metric_results(result_dict)
sequence_compute_metric_results(result_dict)
