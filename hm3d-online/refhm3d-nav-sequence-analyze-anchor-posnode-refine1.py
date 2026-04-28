import argparse
import atexit
import datetime
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anchor_nav.posnode import (
    MergeTracker,
    PosNodeRegistry,
    build_query_fn_from_pq3d_stage2,
    build_selection_trace,
    decompose_description,
    query_registry_with_vlm,
    registry_snapshot,
    select_from_topk,
    update_panorama_node,
    validate_cooccur_nodes_with_image,
)
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
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [scene_dir / f"{short_scene_name}.basis.glb", scene_dir / f"{short_scene_name}.glb"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name}")


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
        f"[Metrics] sequence count={total_count}, avg_sr={total_sr/total_count:.6f}, "
        f"avg_spl={total_spl/total_count:.6f}, avg_task_time_sec={total_task_time/total_count:.3f}"
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
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-posnode-refine1-{ts}-pid{os.getpid()}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[PosNodeRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[PosNodeRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


parser = argparse.ArgumentParser(description="Run RefHM3D anchor posnode refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0)
parser.add_argument("--end_ratio", type=float, default=0.2)
parser.add_argument("--concise_description", action="store_true")
parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/posnode"))
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--posnode_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
parser.add_argument("--posnode_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
parser.add_argument("--posnode_top_k", type=int, default=16)
parser.add_argument("--panorama_update_interval", type=int, default=2)
parser.add_argument("--panorama_subsample_frames", type=int, default=12)
parser.add_argument("--visible_object_max_dist", type=float, default=6.0)
parser.add_argument("--posnode_min_move_dist", type=float, default=0.4)
parser.add_argument("--auto_vlm_after_decision", type=int, default=6)
parser.add_argument("--auto_vlm_interval", type=int, default=2)
parser.add_argument("--disable_auto_vlm_frontier", action="store_true")
parser.add_argument("--disable_early_final_on_cooccur", action="store_true")
parser.add_argument(
    "--decision_log_interval",
    type=int,
    default=0,
    help="每隔 N 次 decision 打印一次明细；<=0 表示关闭明细",
)
args = parser.parse_args()

if args.posnode_api_key:
    os.environ["ZZZ_API_KEY"] = args.posnode_api_key

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)
print(
    f"[PosNodeRefine1] cfg top_k={args.posnode_top_k} "
    f"update_interval={args.panorama_update_interval} pano_frames={args.panorama_subsample_frames} "
    f"visible_max_dist={args.visible_object_max_dist} min_move_dist={args.posnode_min_move_dist} "
    f"auto_vlm_frontier={not args.disable_auto_vlm_frontier} "
    f"auto_after={args.auto_vlm_after_decision} auto_interval={args.auto_vlm_interval}"
)

enabled_task_levels = {"instance"}
success_distance = 0.25
decision_num_min = 3
visible_radius = 3

navigation_data_root = Path(os.path.expanduser(args.navigation_data_path))
scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]
os.makedirs(output_log_dir, exist_ok=True)
out_name = f"refhm3d_seq_posnode_refine1_{args.start_ratio}_{args.end_ratio}.json"
eff_name = f"refhm3d_seq_posnode_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
if args.concise_description:
    out_name = f"refhm3d_seq_posnode_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_posnode_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
output_path = os.path.join(output_log_dir, out_name)
effectiveness_path = os.path.join(output_log_dir, eff_name)

if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    existing_episodes = {"_".join([r["scene_name"], r["navigation_type"], str(r["episode_id"])]) for k in result_dict for r in result_dict[k]}
else:
    result_dict = {"sequence": []}
    existing_episodes = set()

if os.path.exists(effectiveness_path):
    effectiveness_dict = json.load(open(effectiveness_path, "r"))
else:
    effectiveness_dict = {"records": []}

pq3d_model = PQ3DModel(
    os.path.expanduser(args.pq3d_stage1_path),
    os.path.expanduser(args.pq3d_stage2_path),
    min_decision_num=decision_num_min,
)

for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
    scene_name = scene_data_path.name.split(".")[0]
    with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
        pq3d_model.reset()
        query_fn = build_query_fn_from_pq3d_stage2(pq3d_model)
        decision_num = 0
        visited_frontier_set = set()
        registry = PosNodeRegistry()  # per-episode reset, cross-task keep
        merge_tracker = MergeTracker()
        start_position = cur_episode["start_position"]
        start_rotation = cur_episode["start_rotation"]
        episode_id, navigation_type = cur_episode["episode_id"], cur_episode["navigation_type"]
        episode_key = "_".join([scene_name, navigation_type, str(episode_id)])
        if episode_key in existing_episodes:
            continue

        sim_settings = OmegaConf.load("configs/habitat/goat_sim_config.yaml")
        goat_agent_setting = OmegaConf.load("configs/habitat/goat_agent_config.yaml")
        sim_settings["scene"] = resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), scene_name)
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels = convert_meters_to_pixel(9, 512, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, 512, sim)
        out_episode_dir = Path(output_log_dir) / "process" / f"scene={scene_name}" / f"episode={episode_id}"
        out_episode_dir.mkdir(parents=True, exist_ok=True)

        for idx, cur_task in enumerate(cur_episode["task_sequence"]):
            task_t0 = time.perf_counter()
            task_type, task_idx = cur_task
            if task_type not in enabled_task_levels:
                continue
            cur_task = episode_mapping[task_type][task_idx]
            goals = [all_navigation_goals_dict[x] for x in cur_task["target_object_ids"]]
            goal_positions = [
                np.asarray(g.get("position", []), dtype=float).reshape(3)
                for g in goals
                if isinstance(g, dict) and len(g.get("position", [])) >= 3
            ]
            original_sentence = (
                all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"]
                if args.concise_description
                else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
            )
            decomp_t0 = time.perf_counter()
            decomp = decompose_description(original_sentence, args.posnode_vlm_model)
            decomp_ms = (time.perf_counter() - decomp_t0) * 1000.0
            goal_category = goals[0]["object_category"]
            print(
                f"[posnode-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                f"target_desc={decomp.get('target_desc')!r} anchor_desc={decomp.get('anchor_desc')!r} decomp_ms={decomp_ms:.1f}"
            )
            print(f"[posnode-refine1][task-desc] {original_sentence}")

            total_steps = 0
            episode_cum_distance = 0.0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            posnode_attempts = 0
            posnode_used = 0
            auto_vlm_attempts = 0
            auto_vlm_applied = 0
            posnode_vlm_elapsed_ms_total = float(decomp_ms)
            baseline_final_target_pos: Optional[np.ndarray] = None
            final_selected_object_pos: Optional[np.ndarray] = None
            task_effective_logs: List[Dict[str, Any]] = []
            node_update_count = 0

            task_dir = out_episode_dir / f"task={idx}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pano_dir = task_dir / "panorama"
            pano_dir.mkdir(parents=True, exist_ok=True)

            while total_steps < int(args.max_steps):
                color_list, depth_list, agent_state_list = [], [], []
                if len(goto_color_list) > 6:
                    step = max(1, len(goto_color_list) // 6)
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)
                scan_rgb = []
                for _ in range(12):
                    obs = sim.step(action="turn_left")
                    agent_state = agent.get_state()
                    rgb = obs["color_sensor"][:, :, :3]
                    dep = obs["depth_sensor"][:, :]
                    scan_rgb.append(rgb)
                    color_list.append(rgb)
                    depth_list.append(dep)
                    agent_state_list.append(agent_state)
                    fog_of_war_mask = reveal_fog_of_war(
                        top_down_map,
                        fog_of_war_mask,
                        map_coors_to_pixel(agent_state.position, top_down_map, sim),
                        get_polar_angle(agent_state),
                        42,
                        visibility_dist_in_pixels,
                        False,
                    )
                    total_steps += 1
                    if total_steps >= int(args.max_steps):
                        break
                if total_steps >= int(args.max_steps):
                    break
                agent_state = agent.get_state()
                frontier_waypoints = detect_frontier_waypoints(
                    top_down_map,
                    fog_of_war_mask,
                    area_thres_in_pixels,
                    xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                    enable_visualization=False,
                )
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints[:, ::-1], agent_state.position, top_down_map, sim)
                frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]
                target_position, is_final = pq3d_model.decision(
                    color_list, depth_list, agent_state_list, frontier_waypoints, original_sentence, decision_num
                )
                if int(args.decision_log_interval) > 0 and (decision_num % int(args.decision_log_interval) == 0):
                    print(
                        f"[posnode-refine1][decision] task={idx} dec={decision_num} frontiers={len(frontier_waypoints)} "
                        f"baseline_target={np.asarray(target_position, dtype=float).reshape(-1)[:3].tolist()} final={bool(is_final)}"
                    )
                decision_num += 1

                if int(args.panorama_update_interval) > 0 and ((decision_num % int(args.panorama_update_interval)) == 0):
                    try:
                        scan_rgb_panorama = list(reversed(scan_rgb))
                        t_vlm_node = time.perf_counter()
                        node_log = update_panorama_node(
                            agent_pos=np.asarray(agent.get_state().position, dtype=float).reshape(3),
                            color_list=scan_rgb_panorama,
                            rep=pq3d_model.representation_manager,
                            registry=registry,
                            merge_tracker=merge_tracker,
                            vlm_model=args.posnode_vlm_model,
                            step_index=int(total_steps),
                            panorama_dir=pano_dir,
                            max_visible_dist=float(args.visible_object_max_dist),
                            panorama_subsample_frames=int(args.panorama_subsample_frames),
                            min_move_dist_to_add=float(args.posnode_min_move_dist),
                        )
                        node_vlm_ms = (time.perf_counter() - t_vlm_node) * 1000.0
                        if bool(node_log.get("ok", False)):
                            posnode_vlm_elapsed_ms_total += node_vlm_ms
                            node_update_count += 1
                        node_tag = int(node_log.get("node_index", -1))
                        if bool(node_log.get("skipped", False)):
                            node_tag = int(decision_num - 1)
                        with open(task_dir / f"registry_node_{node_tag:03d}.json", "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "task_id": int(idx),
                                    "decision_num": int(decision_num - 1),
                                    "vlm_elapsed_ms": float(node_vlm_ms),
                                    "node_log": node_log,
                                    "registry_tail": registry_snapshot(
                                        registry=registry,
                                        merge_tracker=merge_tracker,
                                        rep=pq3d_model.representation_manager,
                                        max_nodes=8,
                                    ),
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        if bool(node_log.get("ok", False)):
                            print(
                                f"[posnode-refine1][registry-add] task={idx} dec={decision_num-1} node={node_log.get('node_index')} "
                                f"visible={len(node_log.get('visible_indices', []))} pano_frames={node_log.get('panorama_frames_used')} "
                                f"vlm_ms={node_vlm_ms:.1f}"
                            )
                        else:
                            print(
                                f"[posnode-refine1][registry-skip] task={idx} dec={decision_num-1} "
                                f"reason={node_log.get('reason')} move_dist={node_log.get('move_dist')} "
                                f"min_move={node_log.get('min_move_dist_to_add')}"
                            )
                    except Exception as e:
                        print(f"[posnode-refine1][registry-add-fail] task={idx} dec={decision_num-1} err={e!r}")

                used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                posnode_info: Dict[str, Any] = {"ok": False}
                if (not is_final) and (not bool(args.disable_auto_vlm_frontier)):
                    if int(decision_num - 1) >= int(args.auto_vlm_after_decision) and (
                        int(decision_num - 1) % max(1, int(args.auto_vlm_interval)) == 0
                    ):
                        auto_vlm_attempts += 1
                        t_vlm_auto = time.perf_counter()
                        query_result_auto = query_registry_with_vlm(
                            description=original_sentence,
                            registry=registry,
                            vlm_model=args.posnode_vlm_model,
                            decomp=decomp,
                        )
                        auto_vlm_ms = (time.perf_counter() - t_vlm_auto) * 1000.0
                        posnode_vlm_elapsed_ms_total += auto_vlm_ms
                        mode_auto = str(query_result_auto.get("mode", "fallback"))
                        if mode_auto == "co_occur" and len(query_result_auto.get("matched_nodes", [])) > 0:
                            verify_auto = validate_cooccur_nodes_with_image(
                                description=original_sentence,
                                target_desc=str(query_result_auto.get("target_desc", "")),
                                anchor_descs=[str(x) for x in query_result_auto.get("anchor_descs", [])],
                                matched_nodes=list(query_result_auto.get("matched_nodes", [])),
                                vlm_model=args.posnode_vlm_model,
                            )
                            if bool(verify_auto.get("valid", False)):
                                valid_offsets = set(int(x) for x in verify_auto.get("valid_node_offsets", []))
                                matched_nodes_all = list(query_result_auto.get("matched_nodes", []))
                                matched_idx_all = list(query_result_auto.get("matched_node_indices", []))
                                matched_nodes_valid = [n for j, n in enumerate(matched_nodes_all) if j in valid_offsets]
                                matched_idx_valid = [int(matched_idx_all[j]) for j in range(len(matched_idx_all)) if j in valid_offsets]
                                query_result_auto_eff = dict(query_result_auto)
                                query_result_auto_eff["matched_nodes"] = matched_nodes_valid
                                query_result_auto_eff["matched_node_indices"] = matched_idx_valid
                                target_text_auto = str(query_result_auto_eff.get("target_desc", "")).strip() or original_sentence
                                topk_auto = list(query_fn(target_text_auto, int(args.posnode_top_k)))
                                chosen_idx_auto = select_from_topk(
                                    topk=topk_auto,
                                    query_result=query_result_auto_eff,
                                    merge_tracker=merge_tracker,
                                    rep=pq3d_model.representation_manager,
                                )
                                rep_box_auto = np.asarray(
                                    getattr(pq3d_model.representation_manager, "object_box", np.zeros((0, 6))), dtype=float
                                )
                                if (
                                    rep_box_auto.ndim == 2
                                    and rep_box_auto.shape[1] >= 3
                                    and 0 <= int(chosen_idx_auto) < int(rep_box_auto.shape[0])
                                ):
                                    chosen_xyz_auto = np.asarray(rep_box_auto[int(chosen_idx_auto), :3], dtype=float).reshape(3).copy()
                                    chosen_xyz_auto[[1, 2]] = chosen_xyz_auto[[2, 1]]
                                    used_target = chosen_xyz_auto.copy()
                                    auto_vlm_applied += 1
                                    if not bool(args.disable_early_final_on_cooccur):
                                        is_final = True
                                    posnode_info["auto_frontier"] = {
                                        "triggered": True,
                                        "mode": mode_auto,
                                        "early_final_applied": bool(not args.disable_early_final_on_cooccur),
                                        "matched_node_indices": matched_idx_valid,
                                        "chosen_object_index": int(chosen_idx_auto),
                                        "vlm_query_elapsed_ms": float(auto_vlm_ms),
                                        "verify": verify_auto,
                                    }
                                    print(
                                        f"[posnode-refine1][auto-frontier] task={idx} dec={decision_num-1} "
                                        f"mode={mode_auto} chosen={int(chosen_idx_auto)} vlm_ms={auto_vlm_ms:.1f}"
                                    )
                                else:
                                    posnode_info["auto_frontier"] = {
                                        "triggered": False,
                                        "mode": mode_auto,
                                        "reason": "chosen_idx_invalid_after_verify",
                                        "vlm_query_elapsed_ms": float(auto_vlm_ms),
                                        "verify": verify_auto,
                                    }
                            else:
                                posnode_info["auto_frontier"] = {
                                    "triggered": False,
                                    "mode": mode_auto,
                                    "reason": "verify_rejected",
                                    "vlm_query_elapsed_ms": float(auto_vlm_ms),
                                    "verify": verify_auto,
                                }
                        else:
                            posnode_info["auto_frontier"] = {
                                "triggered": False,
                                "mode": mode_auto,
                                "reason": "mode_not_co_occur_or_no_match",
                                "matched_node_indices": query_result_auto.get("matched_node_indices", []),
                                "vlm_query_elapsed_ms": float(auto_vlm_ms),
                            }
                if is_final:
                    baseline_final_target_pos = used_target.copy()
                    posnode_attempts += 1
                    t_vlm_query = time.perf_counter()
                    query_result = query_registry_with_vlm(
                        description=original_sentence,
                        registry=registry,
                        vlm_model=args.posnode_vlm_model,
                        decomp=decomp,
                    )
                    query_vlm_ms = (time.perf_counter() - t_vlm_query) * 1000.0
                    posnode_vlm_elapsed_ms_total += query_vlm_ms
                    query_result_eff = dict(query_result)
                    final_verify_info: Dict[str, Any] = {}
                    if str(query_result.get("mode", "fallback")) == "co_occur" and len(query_result.get("matched_nodes", [])) > 0:
                        verify_final = validate_cooccur_nodes_with_image(
                            description=original_sentence,
                            target_desc=str(query_result.get("target_desc", "")),
                            anchor_descs=[str(x) for x in query_result.get("anchor_descs", [])],
                            matched_nodes=list(query_result.get("matched_nodes", [])),
                            vlm_model=args.posnode_vlm_model,
                        )
                        final_verify_info = dict(verify_final)
                        if bool(verify_final.get("valid", False)):
                            valid_offsets = set(int(x) for x in verify_final.get("valid_node_offsets", []))
                            matched_nodes_all = list(query_result.get("matched_nodes", []))
                            matched_idx_all = list(query_result.get("matched_node_indices", []))
                            query_result_eff["matched_nodes"] = [n for j, n in enumerate(matched_nodes_all) if j in valid_offsets]
                            query_result_eff["matched_node_indices"] = [
                                int(matched_idx_all[j]) for j in range(len(matched_idx_all)) if j in valid_offsets
                            ]
                        else:
                            query_result_eff["mode"] = "fallback"
                            query_result_eff["matched_nodes"] = []
                            query_result_eff["matched_node_indices"] = []

                    target_text = str(query_result_eff.get("target_desc", "")).strip() or original_sentence
                    topk = list(query_fn(target_text, int(args.posnode_top_k)))
                    selection_trace = build_selection_trace(
                        topk=topk,
                        query_result=query_result_eff,
                        merge_tracker=merge_tracker,
                        rep=pq3d_model.representation_manager,
                    )
                    chosen_idx = select_from_topk(
                        topk=topk,
                        query_result=query_result_eff,
                        merge_tracker=merge_tracker,
                        rep=pq3d_model.representation_manager,
                    )
                    top1_idx = int(topk[0][0]) if len(topk) > 0 else -1
                    rep = pq3d_model.representation_manager
                    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
                    if box.ndim == 2 and box.shape[0] > 0 and box.shape[1] >= 3 and 0 <= chosen_idx < int(box.shape[0]):
                        chosen_xyz = np.asarray(box[chosen_idx, :3], dtype=float).reshape(3).copy()
                        chosen_xyz[[1, 2]] = chosen_xyz[[2, 1]]
                    else:
                        chosen_xyz = baseline_final_target_pos.copy()
                    corrected = bool(chosen_idx != top1_idx)
                    if corrected:
                        used_target = chosen_xyz.copy()
                        final_selected_object_pos = chosen_xyz.copy()
                        posnode_used += 1
                    else:
                        used_target = baseline_final_target_pos.copy()
                        final_selected_object_pos = baseline_final_target_pos.copy()
                    posnode_info = {
                        "ok": True,
                        "query_result": {
                            "mode": query_result_eff.get("mode"),
                            "target_desc": query_result_eff.get("target_desc"),
                            "anchor_desc": query_result_eff.get("anchor_desc"),
                            "anchor_descs": query_result_eff.get("anchor_descs", []),
                            "matched_node_indices": query_result_eff.get("matched_node_indices", []),
                        },
                        "topk": [{"object_index": int(i), "score": float(s)} for i, s in topk],
                        "selection_trace": selection_trace,
                        "chosen_object_index": int(chosen_idx),
                        "top1_object_index": int(top1_idx),
                        "vlm_query_elapsed_ms": float(query_vlm_ms),
                        "registry_nodes_total": int(len(registry.nodes)),
                        "vlm_correction_applied": bool(corrected),
                        "vlm_correction_mode": str(query_result_eff.get("mode", "fallback")),
                        "final_verify": final_verify_info,
                    }
                    print(
                        f"[posnode-refine1][final] task={idx} dec={decision_num-1} mode={query_result_eff.get('mode')} "
                        f"chosen={chosen_idx} top1={top1_idx} corrected={corrected} vlm_ms={query_vlm_ms:.1f}"
                    )

                    task_effective_logs.append(
                        {
                            "scene_name": scene_name,
                            "episode_id": int(episode_id),
                            "task_id": int(idx),
                            "decision_num": int(decision_num - 1),
                            "original_sentence": original_sentence,
                            "decompose": decomp,
                            "posnode_ok": True,
                            "posnode_info": posnode_info,
                        }
                    )
                else:
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=used_target, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
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
                    total_steps += 1
                    episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                    prev_agent_state = agent_state
                    if total_steps >= int(args.max_steps):
                        break

                with open(task_dir / f"dec_{decision_num-1:03d}_posnode.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": int(idx),
                            "decision_num": int(decision_num - 1),
                            "is_final": bool(is_final),
                            "target_used": used_target.tolist(),
                            "registry_nodes_total": int(len(registry.nodes)),
                            "registry_tail": registry_snapshot(
                                registry=registry,
                                merge_tracker=merge_tracker,
                                rep=pq3d_model.representation_manager,
                                max_nodes=12,
                            ),
                            "auto_vlm_attempts": int(auto_vlm_attempts),
                            "auto_vlm_applied": int(auto_vlm_applied),
                            "posnode": posnode_info,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                if is_final:
                    break

            task_time = float(time.perf_counter() - task_t0)
            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
            sp = habitat_sim.MultiGoalShortestPath()
            sp.requested_start = sub_episode_start_position
            sp.requested_ends = view_points
            start_end_geo_distance = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
            ep = habitat_sim.MultiGoalShortestPath()
            ep.requested_start = agent_state.position
            ep.requested_ends = view_points
            agent_end_geo_distance = float(ep.geodesic_distance) if path_finder.find_path(ep) else float("inf")
            if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
                sr, spl = 0, 0
            else:
                sr = agent_end_geo_distance <= success_distance
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            baseline_target_to_goal_l2 = float("inf")
            selected_object_to_goal_l2 = float("inf")
            vlm_correction_helpful = None
            if len(goal_positions) > 0:
                if baseline_final_target_pos is not None:
                    d0 = [float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions]
                    baseline_target_to_goal_l2 = float(min(d0))
                if final_selected_object_pos is not None:
                    d1 = [float(np.linalg.norm(final_selected_object_pos - gp)) for gp in goal_positions]
                    selected_object_to_goal_l2 = float(min(d1))
                # No VLM-applied target override -> effectiveness is undefined (None), not False.
                has_vlm_applied = bool(int(posnode_used) > 0 or int(auto_vlm_applied) > 0)
                if has_vlm_applied and np.isfinite(baseline_target_to_goal_l2) and np.isfinite(selected_object_to_goal_l2):
                    vlm_correction_helpful = bool(selected_object_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6)
                else:
                    vlm_correction_helpful = None

            result_dict.setdefault(navigation_type, []).append(
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
                    "posnode_attempts": int(posnode_attempts),
                    "posnode_used": int(posnode_used),
                    "auto_vlm_attempts": int(auto_vlm_attempts),
                    "auto_vlm_applied": int(auto_vlm_applied),
                    "node_update_count": int(node_update_count),
                    "posnode_vlm_elapsed_ms_total": float(posnode_vlm_elapsed_ms_total),
                    "goal_positions": [gp.tolist() for gp in goal_positions],
                    "baseline_target_position": None if baseline_final_target_pos is None else baseline_final_target_pos.tolist(),
                    "selected_object_position": None if final_selected_object_pos is None else final_selected_object_pos.tolist(),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
                    "vlm_correction_helpful": vlm_correction_helpful,
                }
            )
            effectiveness_dict["records"].append(
                {
                    "scene_name": scene_name,
                    "episode_id": int(episode_id),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "posnode_attempts": int(posnode_attempts),
                    "posnode_used": int(posnode_used),
                    "auto_vlm_attempts": int(auto_vlm_attempts),
                    "auto_vlm_applied": int(auto_vlm_applied),
                    "node_update_count": int(node_update_count),
                    "posnode_vlm_elapsed_ms_total": float(posnode_vlm_elapsed_ms_total),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
                    "vlm_correction_helpful": vlm_correction_helpful,
                    "task_effective_logs": task_effective_logs,
                }
            )
            print(
                f"[posnode-refine1] scene={scene_name} ep={episode_id} task={idx} SR={sr} SPL={spl:.4f} "
                f"time={task_time:.3f}s steps={total_steps} decisions={decision_num} "
                f"posnode_attempts={posnode_attempts} posnode_used={posnode_used} "
                f"auto_vlm_attempts={auto_vlm_attempts} auto_vlm_applied={auto_vlm_applied} "
                f"node_updates={node_update_count} "
                f"dist(baseline_target,goal)={baseline_target_to_goal_l2:.3f} "
                f"dist(posnode_object,goal)={selected_object_to_goal_l2:.3f} helpful={vlm_correction_helpful}"
            )

        sim.close()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f)
        with open(effectiveness_path, "w", encoding="utf-8") as f:
            json.dump(effectiveness_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
