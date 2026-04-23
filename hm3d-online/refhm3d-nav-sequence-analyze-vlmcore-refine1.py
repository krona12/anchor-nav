import gzip
import os
import sys
import atexit
import datetime
import tempfile
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

from anchor_nav.vlm_decision_corrector import (
    AsyncVLMDecisionCorrector,
    CorrectorConfig,
    VLMDecisionCorrector,
    resolve_navigation_after_vlm,
)


def sequence_compute_metric_results(result_dict: dict) -> None:
    sequence_results = result_dict.get("sequence", [])
    total_count = len(sequence_results)
    if total_count == 0:
        print("[Metrics] sequence count: 0")
        return

    total_sr = sum(float(item.get("sr", 0)) for item in sequence_results)
    total_spl = sum(float(item.get("spl", 0)) for item in sequence_results)
    total_task_time = sum(float(item.get("task_time_sec", 0.0)) for item in sequence_results)
    avg_sr = total_sr / total_count
    avg_spl = total_spl / total_count
    avg_task_time = total_task_time / total_count
    print(
        f"[Metrics] sequence count: {total_count}, avg_sr: {avg_sr:.6f}, "
        f"avg_spl: {avg_spl:.6f}, avg_task_time_sec: {avg_task_time:.3f}"
    )


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [scene_dir / f"{short_scene_name}.basis.glb", scene_dir / f"{short_scene_name}.glb"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name}. Checked: {[str(x) for x in candidates]}")


def make_2x2_tile(imgs):
    assert len(imgs) == 4
    h, w, _ = imgs[0].shape
    out = np.zeros((h * 2, w * 2, 3), dtype=imgs[0].dtype)
    out[0:h, 0:w] = imgs[0]
    out[0:h, w : 2 * w] = imgs[1]
    out[h : 2 * h, 0:w] = imgs[2]
    out[h : 2 * h, w : 2 * w] = imgs[3]
    return out


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
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-vlmcore-refine1-{ts}-pid{pid}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    print(f"[VLMCore] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[VLMCore] run finished, log saved -> {os.path.abspath(log_path)}")
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


def geo_dist(path_finder, start_pos, ends) -> float:
    if not ends:
        return float("inf")
    sp = habitat_sim.MultiGoalShortestPath()
    sp.requested_start = start_pos
    sp.requested_ends = ends
    if path_finder.find_path(sp):
        return float(sp.geodesic_distance)
    return float("inf")


parser = argparse.ArgumentParser(description="Run RefHM3D VLMCore refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0, help="Dataset start ratio")
parser.add_argument("--end_ratio", type=float, default=0.2, help="Dataset end ratio")
parser.add_argument("--concise_description", action="store_true", help="Use concise descriptions")
parser.add_argument("--navigation_data_path", type=str, default="/home/chenlin/krona/anchor-nav/LangMap_Annotations")
parser.add_argument("--hm3d_data_base_path", type=str, default="/home/chenlin/krona/MTU3D/datascene")
parser.add_argument("--pq3d_stage1_path", type=str, default="/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all")
parser.add_argument("--pq3d_stage2_path", type=str, default="/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-goat")
parser.add_argument("--output_log_dir", type=str, default="./output_logs/anchor/vlmcor")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--enable_vlm_corrector", action="store_true")
parser.add_argument("--vlm_mode", choices=["async", "sync"], default="sync")
parser.add_argument("--vlm_stride", type=int, default=1)
parser.add_argument("--vlm_min_decision_num", type=int, default=2)
parser.add_argument("--vlm_conf_threshold", type=float, default=0.9)
parser.add_argument(
    "--vlm_suppress_commit_conf_threshold",
    type=float,
    default=None,
    help="压制 baseline 错误 object-commit 时的 conf 下限；默认与 vlm_conf_threshold 相同",
)
parser.add_argument("--vlm_base_url", type=str, default="http://127.0.0.1:8000/v1")
parser.add_argument("--vlm_model", type=str, default="Qwen2.5-VL-32B-Instruct")
args = parser.parse_args()

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)
print(f"[VLMCore] vlm_mode={args.vlm_mode}, enable_vlm_corrector={args.enable_vlm_corrector}")

hm3d_data_base_path = os.path.expanduser(args.hm3d_data_base_path)
pq3d_stage1_path = os.path.expanduser(args.pq3d_stage1_path)
pq3d_stage2_path = os.path.expanduser(args.pq3d_stage2_path)
enable_visualization = False
decision_num_min = 3
visible_radius = 3
success_distance = 0.25

concise_description_tag = args.concise_description
start_ratio, end_ratio = args.start_ratio, args.end_ratio
navigation_data_path = os.path.expanduser(args.navigation_data_path)
navigation_data_root = Path(navigation_data_path)
os.makedirs(output_log_dir, exist_ok=True)
if concise_description_tag:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_vlmcor_refine1_concisedesc_{start_ratio}_{end_ratio}.json")
else:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_vlmcor_refine1_{start_ratio}_{end_ratio}.json")

scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
if len(scene_data_paths) == 0:
    raise FileNotFoundError(f"No *.json.gz files found under {navigation_data_path}")
num_scene = len(scene_data_paths)
scene_data_paths = scene_data_paths[int(start_ratio * num_scene) : int(end_ratio * num_scene)]

if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    sequence_compute_metric_results(result_dict)
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
    scene_name = scene_data_path.name.split(".")[0]
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

    for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
        pq3d_model.reset()
        decision_num = 0
        visited_frontier_set = set()

        start_position = cur_episode["start_position"]
        start_rotation = cur_episode["start_rotation"]
        episode_id, navigation_type = cur_episode["episode_id"], cur_episode["navigation_type"]
        if "_".join([scene_name, navigation_type, str(episode_id)]) in existing_episodes:
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

        corrector = None
        async_corrector = None
        if args.enable_vlm_corrector:
            corrector = VLMDecisionCorrector(
                CorrectorConfig(
                    enabled=True,
                    stride=args.vlm_stride,
                    min_decision_num=args.vlm_min_decision_num,
                    confidence_threshold=args.vlm_conf_threshold,
                    suppress_commit_conf_threshold=args.vlm_suppress_commit_conf_threshold,
                    base_url=args.vlm_base_url,
                    model=args.vlm_model,
                )
            )
            if args.vlm_mode == "async":
                async_corrector = AsyncVLMDecisionCorrector(corrector)

        for idx, cur_task in enumerate(cur_episode["task_sequence"]):
            task_t0 = time.perf_counter()
            task_type, task_idx = cur_task
            cur_task = episode_mapping[task_type][task_idx]
            goals_ids = cur_task["target_object_ids"]
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
                sentence = f"{cur_task['object_category']} in the {region_to_annot_dict[cur_task['region_id']]['region_category'].lower()} that has {region_desc}"
                goal_category = cur_task["object_category"]
            else:
                sentence = (
                    all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"]
                    if concise_description_tag
                    else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
                )
                goal_category = goals[0]["object_category"]
            use_vlm_for_task = bool(args.enable_vlm_corrector and task_type in ["region", "instance"])

            total_steps = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0.0
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            prev_obj_count = np.asarray(getattr(pq3d_model.representation_manager, "object_count", np.zeros((0,))), dtype=float)
            vlm_calls_total = 0
            vlm_force_total = 0

            while total_steps < args.max_steps:
                color_list, depth_list, agent_state_list = [], [], []
                if len(goto_color_list) > 6:
                    step = max(1, len(goto_color_list) // 6)
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)

                for _ in range(12):
                    obs = sim.step(action="turn_left")
                    color = obs["color_sensor"][:, :, :3]
                    depth = obs["depth_sensor"][:, :]
                    agent_state = agent.get_state()
                    color_list.append(color)
                    depth_list.append(depth)
                    agent_state_list.append(agent_state)
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
                    if total_steps >= args.max_steps:
                        break
                if total_steps >= args.max_steps:
                    break

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
                frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]

                target_position, is_final_decision = pq3d_model.decision(
                    color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num
                )
                baseline_type = "object" if is_final_decision else "frontier"
                decision_num += 1

                rep = pq3d_model.representation_manager
                obj_boxes = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
                obj_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1)
                obj_counts = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float).reshape(-1)
                cur_n = len(obj_scores)
                prev_n = len(prev_obj_count)
                new_ids = list(range(prev_n, cur_n)) if cur_n > prev_n else []
                updated_ids = [i for i in range(min(prev_n, cur_n)) if obj_counts[i] > prev_obj_count[i]]
                current_pool = sorted(set(new_ids + updated_ids))
                current_pool = sorted(current_pool, key=lambda i: float(obj_scores[i]), reverse=True)[:15]
                current_candidates = [
                    {
                        "object_id_in_memory": int(i),
                        "score": float(obj_scores[i]),
                        "count": float(obj_counts[i]),
                        "center_xyz": [float(x) for x in obj_boxes[i, :3].tolist()] if i < len(obj_boxes) else [],
                    }
                    for i in current_pool
                ]

                tiles = []
                tmp_tile_dir = Path(tempfile.mkdtemp(prefix="vlmcor_tiles_"))
                pano = color_list[-12:]
                if len(pano) == 12:
                    for t in range(3):
                        tile = make_2x2_tile(pano[t * 4 : (t + 1) * 4])
                        tile_path = tmp_tile_dir / f"vlm_tile_{t}.jpg"
                        cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                        tiles.append(tile_path)

                vlm_info = {"vlm_called": False}
                vlm_src_pos = None
                if async_corrector is not None and use_vlm_for_task:
                    polled = async_corrector.poll_ready()
                    vlm_info = polled.get("vlm_info", {"vlm_called": False})
                    vlm_src_pos = polled.get("source_agent_position", None)
                elif corrector is not None and use_vlm_for_task and corrector.should_call(decision_num - 1):
                    try:
                        vlm_info = corrector.evaluate(
                            description=sentence,
                            decision_num=decision_num - 1,
                            baseline_target_type=baseline_type,
                            num_frontiers=len(frontier_waypoints),
                            memory_objects=cur_n,
                            candidate_objects=current_candidates,
                            image_tile_paths=tiles,
                        )
                    except Exception as e:
                        vlm_info = {"vlm_called": True, "error": str(e)}
                    vlm_src_pos = [float(x) for x in np.asarray(agent_state.position, dtype=float).tolist()]
                    vlm_calls_total += 1

                mem_top = None
                if len(current_pool) > 0 and current_pool[0] < len(obj_boxes):
                    mem_top = obj_boxes[current_pool[0], :3]
                _cfg = (
                    corrector.cfg
                    if corrector is not None
                    else CorrectorConfig(
                        confidence_threshold=args.vlm_conf_threshold,
                        suppress_commit_conf_threshold=args.vlm_suppress_commit_conf_threshold,
                    )
                )
                merged = resolve_navigation_after_vlm(
                    baseline_is_final=bool(is_final_decision),
                    baseline_target=target_position,
                    frontiers=frontier_waypoints,
                    agent_position=agent_state.position,
                    vlm_info=vlm_info,
                    cfg=_cfg,
                    memory_top_xyz=mem_top,
                    async_agent_xyz=vlm_src_pos,
                    sync_vlm_round=(args.vlm_mode == "sync"),
                )
                corrected_target = np.asarray(merged["corrected_target"], dtype=float)
                corrected_final = bool(merged["corrected_final"])
                if merged.get("vlm_force_applied"):
                    vlm_force_total += 1

                if async_corrector is not None and use_vlm_for_task:
                    scheduled = async_corrector.submit_if_needed(
                        decision_num=decision_num - 1,
                        description=sentence,
                        baseline_target_type=baseline_type,
                        num_frontiers=len(frontier_waypoints),
                        memory_objects=cur_n,
                        candidate_objects=current_candidates,
                        image_tile_paths=tiles,
                        source_agent_position=[float(x) for x in np.asarray(agent_state.position, dtype=float).tolist()],
                        cleanup_tile_paths=tiles,
                        cleanup_dir=tmp_tile_dir,
                    )
                    if scheduled:
                        vlm_calls_total += 1
                    else:
                        for p in tiles:
                            p.unlink(missing_ok=True)
                        tmp_tile_dir.rmdir()
                else:
                    for p in tiles:
                        p.unlink(missing_ok=True)
                    tmp_tile_dir.rmdir()

                if not corrected_final:
                    visited_frontier_set.add(tuple(np.round(corrected_target, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=corrected_target, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(
                    path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
                )
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except Exception:
                    action_list = []

                goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
                for action in action_list:
                    if not action:
                        continue
                    obs = sim.step(action=action)
                    agent_state = agent.get_state()
                    goto_color_list.append(obs["color_sensor"][:, :, :3])
                    goto_depth_list.append(obs["depth_sensor"][:, :])
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
                    episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                    prev_agent_state = agent_state
                    if total_steps >= args.max_steps:
                        break

                prev_obj_count = obj_counts.copy()
                if corrected_final:
                    break

            task_time = float(time.perf_counter() - task_t0)
            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
            start_end_geo_distance = geo_dist(path_finder, sub_episode_start_position, view_points)
            agent_end_geo_distance = geo_dist(path_finder, agent_state.position, view_points)
            if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
                sr, spl = 0, 0
            else:
                sr = agent_end_geo_distance <= success_distance
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            result_dict[navigation_type].append(
                {
                    "scene_name": scene_name,
                    "episode_id": episode_id,
                    "task_id": idx,
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "sr": sr,
                    "spl": spl,
                    "object_category": goal_category,
                    "task_time_sec": task_time,
                    "steps_total": int(total_steps),
                    "decisions": int(decision_num),
                    "start_goal_geo": float(start_end_geo_distance),
                    "end_goal_geo": float(agent_end_geo_distance),
                    "episode_cum_distance": float(episode_cum_distance),
                    "vlm_mode": args.vlm_mode,
                    "vlm_task_enabled": bool(use_vlm_for_task),
                    "vlm_calls_total": int(vlm_calls_total),
                    "vlm_force_total": int(vlm_force_total),
                }
            )
            print(
                f"[vlmcore-refine1] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                f"SR={sr} SPL={spl:.4f} time={task_time:.3f}s "
                f"steps={total_steps} decisions={decision_num} vlm_calls={vlm_calls_total} vlm_force={vlm_force_total}"
            )

        if async_corrector is not None:
            async_corrector.close()
        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
