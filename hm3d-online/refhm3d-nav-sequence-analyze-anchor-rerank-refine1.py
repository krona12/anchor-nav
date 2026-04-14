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
import random
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

from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL
from anchor_nav.rerank import (
    confirm_target_visible_from_rgb_views,
    RerankConfig,
    list_rerank_candidate_memory_ids,
    parse_levels_csv,
    rerank_memory_target,
    rerank_object_target,
    should_run_rerank,
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
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-rerank-refine1-{ts}-pid{pid}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    print(f"[RerankRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[RerankRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
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


parser = argparse.ArgumentParser(description="Run RefHM3D anchor rerank refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0, help="Dataset start ratio")
parser.add_argument("--end_ratio", type=float, default=0.2, help="Dataset end ratio")
parser.add_argument("--concise_description", action="store_true", help="Use concise descriptions")
parser.add_argument(
    "--task_levels",
    type=str,
    default="object,room,region,instance",
    help="Comma-separated task levels to execute, e.g. instance or region,instance",
)
parser.add_argument("--navigation_data_path", type=str, default="/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D/LangMap_Annotations")
parser.add_argument("--hm3d_data_base_path", type=str, default="/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D/datascene")
parser.add_argument("--pq3d_stage1_path", type=str, default="/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D/checkpoint/stage1-pretrain-all")
parser.add_argument("--pq3d_stage2_path", type=str, default="/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D/checkpoint/stage2-fine-tune-goat")
parser.add_argument("--output_log_dir", type=str, default="/home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D/output_logs/anchor/rerank")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument(
    "--rerank_levels",
    type=str,
    default="instance",
    help="Comma-separated task levels to run rerank on (e.g. region,instance)",
)
parser.add_argument("--rerank_top_k", type=int, default=8)
parser.add_argument("--rerank_min_rgb_cand", type=int, default=2, help="Min candidates with first-sight RGB to call VLM")
parser.add_argument(
    "--disable_frontier_vlm_rescue",
    action="store_true",
    help="Disable optional frontier rescue: after frontier streak, probabilistically call VLM for memory target",
)
parser.add_argument("--frontier_streak_trigger", type=int, default=5)
parser.add_argument("--frontier_vlm_prob_step", type=float, default=0.05)
parser.add_argument("--disable_vlm_post_arrival_confirm", action="store_true")
parser.add_argument("--vlm_post_arrival_confirm_max_images", type=int, default=4)
parser.add_argument("--vlm_post_arrival_approach_max_steps", type=int, default=30)
parser.add_argument("--vlm_base_url", type=str, default="http://127.0.0.1:8000/v1")
parser.add_argument("--vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
parser.add_argument(
    "--vlm_api_key",
    type=str,
    default=os.environ.get("ZZZ_API_KEY", ""),
    help="VLM API key for hm3d-online/vlm/client.py (writes env ZZZ_API_KEY when provided)",
)
parser.add_argument(
    "--rerank_save_image_log",
    type=int,
    default=0,
    choices=(0, 1),
    help="是否落盘 rerank_vlm_io/ 下 input 图片等调试文件：0=否（默认），1=是",
)
args = parser.parse_args()
os.environ["RERANK_SKIP_IO_ARTIFACTS"] = "0" if args.rerank_save_image_log else "1"
if args.vlm_api_key:
    os.environ["ZZZ_API_KEY"] = args.vlm_api_key

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)

cfg_kwargs = {
    "enabled_levels": parse_levels_csv(args.rerank_levels),
    "top_k": int(args.rerank_top_k),
    "min_candidates_with_rgb": int(args.rerank_min_rgb_cand),
}
# 兼容新旧 rerank 配置：旧版可能有 base_url/api_key，新版可能由 hm3d-online/vlm/client.py 固定。
_fields = getattr(RerankConfig, "__dataclass_fields__", {})
if "base_url" in _fields:
    cfg_kwargs["base_url"] = args.vlm_base_url
if "model" in _fields:
    cfg_kwargs["model"] = args.vlm_model
if "api_key" in _fields and args.vlm_api_key:
    cfg_kwargs["api_key"] = args.vlm_api_key
rerank_cfg = RerankConfig(**cfg_kwargs)
print(
    f"[RerankRefine1] rerank_levels={sorted(rerank_cfg.enabled_levels)} "
    f"top_k={rerank_cfg.top_k} min_rgb_cand={rerank_cfg.min_candidates_with_rgb} "
    f"rerank_save_image_log={args.rerank_save_image_log} "
    f"vlm_model={args.vlm_model}"
)
frontier_vlm_rescue_enabled = not bool(args.disable_frontier_vlm_rescue)
print(
    f"[RerankRefine1] frontier_vlm_rescue_enabled={frontier_vlm_rescue_enabled} "
    f"frontier_streak_trigger={args.frontier_streak_trigger} frontier_vlm_prob_step={args.frontier_vlm_prob_step}"
)
vlm_post_arrival_confirm_enabled = not bool(args.disable_vlm_post_arrival_confirm)
print(
    f"[RerankRefine1] vlm_post_arrival_confirm_enabled={vlm_post_arrival_confirm_enabled} "
    f"confirm_max_images={args.vlm_post_arrival_confirm_max_images}"
)
enabled_task_levels = {x.strip() for x in args.task_levels.split(",") if x.strip()}
if not enabled_task_levels:
    enabled_task_levels = {"object", "room", "region", "instance"}
print(f"[RerankRefine1] enabled_task_levels={sorted(enabled_task_levels)}")

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

# 分片独立 jsonl，避免多进程同目录冲突；可用环境变量覆盖
_shard_tag = f"{start_ratio}_{end_ratio}"
_default_rerank_log = os.path.join(output_log_dir, f"rerank_vlm_refine1_{_shard_tag}.jsonl")
if os.environ.get("RERANK_LOG_JSONL", "").strip():
    pass
else:
    os.environ["RERANK_LOG_JSONL"] = _default_rerank_log

if concise_description_tag:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_rerank_refine1_concisedesc_{start_ratio}_{end_ratio}.json")
else:
    output_path = os.path.join(output_log_dir, f"refhm3d_seq_rerank_refine1_{start_ratio}_{end_ratio}.json")

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
                f"[rerank-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} "
                f"nav_type={navigation_type} level={task_type}"
            )
            print(f"[rerank-refine1][task-desc] {sentence}")

            total_steps = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0.0
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            prev_obj_count = np.asarray(getattr(pq3d_model.representation_manager, "object_count", np.zeros((0,))), dtype=float)
            rerank_attempts = 0
            rerank_applied = 0
            rerank_vlm_elapsed_ms_total = 0.0
            rerank_vlm_elapsed_ms_last = None
            frontier_streak = 0
            frontier_rescue_attempts = 0
            frontier_rescue_applied = 0

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
                obj_counts = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float).reshape(-1)
                cur_n = len(obj_counts)

                target_before_rerank = np.asarray(target_position, dtype=float).reshape(3).copy()
                corrected_target = target_before_rerank.copy()
                corrected_final = bool(is_final_decision)
                rinf: dict = {}
                vlm_adjusted_target = False
                if is_final_decision:
                    frontier_streak = 0
                else:
                    frontier_streak += 1

                if is_final_decision and should_run_rerank(task_type, rerank_cfg):
                    rerank_attempts += 1
                    print(
                        f"[rerank-refine1][vlm-call] scene={scene_name} ep={episode_id} task={idx} "
                        f"decision={decision_num-1} level={task_type} memory_objects={cur_n}"
                    )
                    new_tp, rinf = rerank_object_target(
                        description=sentence,
                        rep=rep,
                        baseline_target_xyz=target_before_rerank,
                        decision_aux=getattr(pq3d_model, "last_decision_aux", {}),
                        cfg=rerank_cfg,
                    )
                    _elapsed_ms = rinf.get("elapsed_ms", None)
                    if _elapsed_ms is not None:
                        _elapsed_ms = float(_elapsed_ms)
                        rerank_vlm_elapsed_ms_last = _elapsed_ms
                        rerank_vlm_elapsed_ms_total += _elapsed_ms
                    print(
                        f"[rerank-refine1][vlm-result] applied={bool(rinf.get('rerank_applied'))} "
                        f"best_index={rinf.get('best_index_1based')} chosen_mem={rinf.get('chosen_memory_index')} "
                        f"baseline_mem={rinf.get('baseline_memory_index')} "
                        f"elapsed_ms={rinf.get('elapsed_ms')} "
                        f"reason={str(rinf.get('reason', ''))[:160]!r}"
                    )
                    if rinf.get("rerank_applied"):
                        rerank_applied += 1
                        corrected_target = np.asarray(new_tp, dtype=float).reshape(3)
                        vlm_adjusted_target = True
                elif (
                    (not is_final_decision)
                    and frontier_vlm_rescue_enabled
                    and should_run_rerank(task_type, rerank_cfg)
                    and frontier_streak >= int(args.frontier_streak_trigger)
                ):
                    cand_count = len(list_rerank_candidate_memory_ids(rep, top_k=rerank_cfg.top_k))
                    rescue_prob = min(
                        1.0,
                        max(
                            0.0,
                            0.4
                            + float(args.frontier_vlm_prob_step)
                            * float(frontier_streak - int(args.frontier_streak_trigger)),
                        ),
                    )
                    if cand_count >= rerank_cfg.min_candidates_with_rgb and random.random() < rescue_prob:
                        frontier_rescue_attempts += 1
                        print(
                            f"[rerank-refine1][frontier-vlm-call] scene={scene_name} ep={episode_id} task={idx} "
                            f"decision={decision_num-1} streak={frontier_streak} prob={rescue_prob:.2f} "
                            f"memory_candidates={cand_count}"
                        )
                        new_tp, rinf = rerank_memory_target(
                            description=sentence,
                            rep=rep,
                            cfg=rerank_cfg,
                            baseline_memory_index=int(getattr(pq3d_model, "last_decision_aux", {}).get("real_object_decision_idx", -1)),
                        )
                        _elapsed_ms = rinf.get("elapsed_ms", None)
                        if _elapsed_ms is not None:
                            _elapsed_ms = float(_elapsed_ms)
                            rerank_vlm_elapsed_ms_last = _elapsed_ms
                            rerank_vlm_elapsed_ms_total += _elapsed_ms
                        corrected_target = np.asarray(new_tp, dtype=float).reshape(3)
                        frontier_rescue_applied += 1
                        vlm_adjusted_target = True
                        print(
                            f"[rerank-refine1][frontier-vlm-result] chosen_mem={rinf.get('chosen_memory_index')} "
                            f"elapsed_ms={rinf.get('elapsed_ms')} reason={str(rinf.get('reason', ''))[:160]!r}"
                        )

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

                if vlm_adjusted_target and vlm_post_arrival_confirm_enabled and total_steps < args.max_steps:
                    confirm_rgb = []
                    for _ in range(12):
                        obs = sim.step(action="turn_left")
                        agent_state = agent.get_state()
                        confirm_rgb.append(obs["color_sensor"][:, :, :3])
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
                    if len(confirm_rgb) > 0:
                        cinfo = confirm_target_visible_from_rgb_views(
                            description=sentence,
                            rgb_views=confirm_rgb,
                            cfg=rerank_cfg,
                            max_images=int(args.vlm_post_arrival_confirm_max_images),
                        )
                        print(
                            f"[rerank-refine1][post-arrival-confirm] has_target={cinfo.get('has_target')} "
                            f"elapsed_ms={cinfo.get('elapsed_ms')} reason={str(cinfo.get('reason', ''))[:120]!r}"
                        )
                        if cinfo.get("has_target"):
                            chosen_mid = rinf.get("chosen_memory_index", None)
                            if chosen_mid is None:
                                cand1 = list_rerank_candidate_memory_ids(rep, top_k=1)
                                chosen_mid = int(cand1[0]) if len(cand1) > 0 else None
                            if chosen_mid is not None:
                                mid = int(chosen_mid)
                                obj_box_now = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
                                if mid < len(obj_box_now):
                                    approach_target = obj_box_now[mid, :3].copy()
                                    approach_target[[1, 2]] = approach_target[[2, 1]]
                                    print(
                                        f"[rerank-refine1][post-arrival-approach] mem={mid} "
                                        f"max_steps={args.vlm_post_arrival_approach_max_steps}"
                                    )
                                    cur_state = agent.get_state()
                                    try:
                                        island_idx = path_finder.get_island(cur_state.position)
                                        approach_nav = path_finder.snap_point(
                                            point=np.asarray(approach_target, dtype=float), island_index=island_idx
                                        )
                                        approach_follower = habitat_sim.GreedyGeodesicFollower(
                                            path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
                                        )
                                        approach_actions = approach_follower.find_path(approach_nav) or []
                                    except Exception:
                                        approach_actions = []
                                    used = 0
                                    for a2 in approach_actions:
                                        if not a2 or used >= int(args.vlm_post_arrival_approach_max_steps):
                                            break
                                        obs2 = sim.step(action=a2)
                                        agent_state = agent.get_state()
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
                                        used += 1
                                        if total_steps >= args.max_steps:
                                            break
                                    print(f"[rerank-refine1][post-arrival-approach] used_steps={used}")
                            corrected_final = True

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
                    "rerank_levels": sorted(rerank_cfg.enabled_levels),
                    "rerank_attempts": int(rerank_attempts),
                    "rerank_applied": int(rerank_applied),
                    "frontier_rescue_attempts": int(frontier_rescue_attempts),
                    "frontier_rescue_applied": int(frontier_rescue_applied),
                    "rerank_vlm_elapsed_ms_total": float(rerank_vlm_elapsed_ms_total),
                    "rerank_vlm_elapsed_ms_avg": (
                        float(rerank_vlm_elapsed_ms_total / rerank_attempts) if rerank_attempts > 0 else None
                    ),
                    "rerank_vlm_elapsed_ms_last": rerank_vlm_elapsed_ms_last,
                }
            )
            print(
                f"[rerank-refine1] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                f"SR={sr} SPL={spl:.4f} time={task_time:.3f}s "
                f"steps={total_steps} decisions={decision_num} rerank_attempts={rerank_attempts} rerank_applied={rerank_applied}"
            )

        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
