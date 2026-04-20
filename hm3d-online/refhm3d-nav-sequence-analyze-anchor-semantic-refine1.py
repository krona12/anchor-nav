import gzip
import os
import sys
import atexit
import datetime
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
from data_utils import PQ3DModel
from tqdm import tqdm
import time
import argparse

from anchor_nav.semantic_enhance import (
    SemanticEnhanceConfig,
    parse_levels_csv,
    semantic_enhance_object_target,
    should_run_semantic_enhance,
)
from anchor_nav.validate import ValidateConfig, validate_after_arrival


def sequence_compute_metric_results(result_dict: dict) -> None:
    sequence_results = result_dict.get("sequence", [])
    total_count = len(sequence_results)
    if total_count == 0:
        print("[Metrics] sequence count: 0")
        return
    total_sr = sum(float(item.get("sr", 0)) for item in sequence_results)
    total_spl = sum(float(item.get("spl", 0)) for item in sequence_results)
    total_task_time = sum(float(item.get("task_time_sec", 0.0)) for item in sequence_results)
    print(
        f"[Metrics] sequence count: {total_count}, avg_sr: {total_sr/total_count:.6f}, "
        f"avg_spl: {total_spl/total_count:.6f}, avg_task_time_sec: {total_task_time/total_count:.3f}"
    )


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [scene_dir / f"{short_scene_name}.basis.glb", scene_dir / f"{short_scene_name}.glb"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name}. Checked: {[str(x) for x in candidates]}")


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
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-semantic-refine1-{ts}-pid{pid}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    print(f"[SemanticRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[SemanticRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
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


parser = argparse.ArgumentParser(description="Run RefHM3D anchor semantic refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0)
parser.add_argument("--end_ratio", type=float, default=0.2)
parser.add_argument("--concise_description", action="store_true")
parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/semantic"))
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--semantic_levels", type=str, default="instance")
parser.add_argument("--semantic_top_k", type=int, default=10)
parser.add_argument("--semantic_top_m", type=int, default=5)
parser.add_argument("--semantic_prob_temperature", type=float, default=0.07)
parser.add_argument("--semantic_vlm_model", type=str, default="gpt-4o-mini")
parser.add_argument("--semantic_clip_model_path", type=str, default="openai/clip-vit-large-patch14")
parser.add_argument("--semantic_clip_device", type=str, default="cuda")
parser.add_argument("--semantic_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
parser.add_argument("--disable_validate", action="store_true")
parser.add_argument("--validate_vlm_model", type=str, default="")
parser.add_argument("--validate_num_views", type=int, default=12)
parser.add_argument("--validate_group_size", type=int, default=4)
parser.add_argument("--validate_max_tokens", type=int, default=128)
parser.add_argument("--validate_max_distance_cm", type=float, default=50.0)
args = parser.parse_args()

if args.semantic_api_key:
    os.environ["ZZZ_API_KEY"] = args.semantic_api_key

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)

sem_cfg = SemanticEnhanceConfig(
    enabled_levels=parse_levels_csv(args.semantic_levels),
    top_k=int(args.semantic_top_k),
    top_m=int(args.semantic_top_m),
    prob_temperature=float(args.semantic_prob_temperature),
    clip_model_path=str(args.semantic_clip_model_path),
    clip_device=str(args.semantic_clip_device),
    vlm_model=str(args.semantic_vlm_model),
    override_final_on_apply=False,
)
validate_cfg = ValidateConfig(
    enabled=(not bool(args.disable_validate)),
    vlm_model=(str(args.validate_vlm_model).strip() or str(args.semantic_vlm_model)),
    num_views=int(args.validate_num_views),
    group_size=int(args.validate_group_size),
    max_tokens=int(args.validate_max_tokens),
    max_distance_cm=float(args.validate_max_distance_cm),
)
print(
    f"[SemanticRefine1] semantic_levels={sorted(sem_cfg.enabled_levels)} top_k={sem_cfg.top_k} "
    f"top_m={sem_cfg.top_m} temp={sem_cfg.prob_temperature} vlm_model={sem_cfg.vlm_model}"
)
print(
    f"[SemanticRefine1] validate_enabled={validate_cfg.enabled} validate_model={validate_cfg.vlm_model} "
    f"views={validate_cfg.num_views} group={validate_cfg.group_size} max_distance_cm={validate_cfg.max_distance_cm}"
)
enabled_task_levels = {"instance"}
print(f"[SemanticRefine1] enabled_task_levels={sorted(enabled_task_levels)}")

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
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_semantic_refine1_concisedesc_{start_ratio}_{end_ratio}.json")
else:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_semantic_refine1_{start_ratio}_{end_ratio}.json")

scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
if len(scene_data_paths) == 0:
    raise FileNotFoundError(f"No *.json.gz files found under {navigation_data_path}")
num_scene = len(scene_data_paths)
scene_data_paths = scene_data_paths[int(start_ratio * num_scene): int(end_ratio * num_scene)]

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

        for idx, cur_task in enumerate(cur_episode["task_sequence"]):
            task_t0 = time.perf_counter()
            task_type, task_idx = cur_task
            if task_type not in enabled_task_levels:
                continue
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
            print(
                f"[semantic-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} "
                f"nav_type={navigation_type} level={task_type}"
            )
            print(f"[semantic-refine1][task-desc] {sentence}")

            total_steps = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0.0
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            semantic_attempts = 0
            semantic_applied = 0
            semantic_vlm_elapsed_ms_total = 0.0
            semantic_vlm_elapsed_ms_last = None
            validate_attempts = 0
            validate_passed = 0
            validate_failed = 0
            validation_retry_used = False

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
                decision_num += 1

                rep = pq3d_model.representation_manager
                target_before = np.asarray(target_position, dtype=float).reshape(3).copy()
                corrected_target = target_before.copy()
                corrected_final = bool(is_final_decision)
                sinfo = {"skipped": "deferred_until_validate_fail"}

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

                if corrected_final and validate_cfg.enabled and (not validation_retry_used):
                    validate_attempts += 1
                    validate_io_dir = (
                        Path(output_log_dir)
                        / "validate_io"
                        / f"scene={scene_name}"
                        / f"episode={episode_id}"
                        / f"task={idx}"
                        / f"decision={decision_num-1:03d}"
                    )
                    validate_io_dir.mkdir(parents=True, exist_ok=True)
                    validate_info = validate_after_arrival(
                        sim=sim,
                        description=sentence,
                        cfg=validate_cfg,
                        io_dir=validate_io_dir,
                        jsonl_log_path=str(Path(output_log_dir) / "validate_vlm_refine1.jsonl"),
                    )
                    total_steps += int(validate_info.get("num_steps", 0))
                    if bool(validate_info.get("contains_target", False)):
                        validate_passed += 1
                        print(
                            f"[semantic-refine1][validate-pass] scene={scene_name} ep={episode_id} task={idx} "
                            f"decision={decision_num-1} target={goal_category!r} elapsed_ms={validate_info.get('elapsed_ms')}"
                        )
                    else:
                        validate_failed += 1
                        validation_retry_used = True
                        corrected_final = False
                        retry_target = np.asarray(target_before, dtype=float).reshape(3).copy()
                        if should_run_semantic_enhance(task_type, sem_cfg):
                            semantic_attempts += 1
                            print(
                                f"[semantic-refine1][vlm-call] scene={scene_name} ep={episode_id} task={idx} "
                                f"decision={decision_num-1} level={task_type} reason=validate_fail"
                            )
                            retry_target, sinfo = semantic_enhance_object_target(
                                description=sentence,
                                rep=rep,
                                baseline_target_xyz=target_before,
                                decision_aux=getattr(pq3d_model, "last_decision_aux", {}),
                                cfg=sem_cfg,
                            )
                            if sinfo.get("semantic_applied"):
                                semantic_applied += 1
                            _elapsed_ms = sinfo.get("elapsed_ms", None)
                            if _elapsed_ms is not None:
                                _elapsed_ms = float(_elapsed_ms)
                                semantic_vlm_elapsed_ms_last = _elapsed_ms
                                semantic_vlm_elapsed_ms_total += _elapsed_ms
                            print(
                                f"[semantic-refine1][vlm-result] applied={bool(sinfo.get('semantic_applied'))} "
                                f"chosen_mem={sinfo.get('chosen_memory_index')} "
                                f"score={sinfo.get('chosen_weighted_score')} elapsed_ms={sinfo.get('elapsed_ms')}"
                            )
                        else:
                            sinfo = {"skipped": "task_level_not_enabled"}
                        print(
                            f"[semantic-refine1][validate-fail] scene={scene_name} ep={episode_id} task={idx} "
                            f"decision={decision_num-1} -> semantic retry without target confirmation"
                        )
                        # 失败后立即执行一次 semantic 目标导航；后续不再进行 target 确认
                        retry_state = agent.get_state()
                        retry_island = path_finder.get_island(retry_state.position)
                        retry_navmesh = path_finder.snap_point(point=np.asarray(retry_target, dtype=float), island_index=retry_island)
                        retry_follower = habitat_sim.GreedyGeodesicFollower(
                            path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
                        )
                        try:
                            retry_actions = retry_follower.find_path(retry_navmesh)
                        except Exception:
                            retry_actions = []
                        goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
                        for retry_action in retry_actions:
                            if not retry_action:
                                continue
                            obs_retry = sim.step(action=retry_action)
                            retry_state2 = agent.get_state()
                            goto_color_list.append(obs_retry["color_sensor"][:, :, :3])
                            goto_depth_list.append(obs_retry["depth_sensor"][:, :])
                            goto_agent_state_list.append(retry_state2)
                            fog_of_war_mask = reveal_fog_of_war(
                                top_down_map=top_down_map,
                                current_fog_of_war_mask=fog_of_war_mask,
                                current_point=map_coors_to_pixel(retry_state2.position, top_down_map, sim),
                                current_angle=get_polar_angle(retry_state2),
                                fov=42,
                                max_line_len=visibility_dist_in_pixels,
                                enable_debug_visualization=enable_visualization,
                            )
                            total_steps += 1
                            episode_cum_distance += np.linalg.norm(retry_state2.position - prev_agent_state.position)
                            prev_agent_state = retry_state2
                            if total_steps >= args.max_steps:
                                break

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

            if navigation_type not in result_dict:
                result_dict[navigation_type] = []
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
                    "semantic_levels": sorted(sem_cfg.enabled_levels),
                    "semantic_attempts": int(semantic_attempts),
                    "semantic_applied": int(semantic_applied),
                    "semantic_vlm_elapsed_ms_total": float(semantic_vlm_elapsed_ms_total),
                    "semantic_vlm_elapsed_ms_avg": (
                        float(semantic_vlm_elapsed_ms_total / semantic_attempts) if semantic_attempts > 0 else None
                    ),
                    "semantic_vlm_elapsed_ms_last": semantic_vlm_elapsed_ms_last,
                    "validate_enabled": bool(validate_cfg.enabled),
                    "validate_attempts": int(validate_attempts),
                    "validate_passed": int(validate_passed),
                    "validate_failed": int(validate_failed),
                    "validation_retry_used": bool(validation_retry_used),
                }
            )
            print(
                f"[semantic-refine1] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                f"SR={sr} SPL={spl:.4f} time={task_time:.3f}s steps={total_steps} "
                f"decisions={decision_num} semantic_attempts={semantic_attempts} semantic_applied={semantic_applied}"
            )

        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
