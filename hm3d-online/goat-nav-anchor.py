from collections import defaultdict
import atexit
from datetime import datetime
import gzip
import os
import sys
import habitat
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import OmegaConf
import torch
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
import random

from anchor_nav import (
    AnchorNavContext,
    AnchorPQ3DModel,
    load_anchor_plugin_config,
    on_after_decision,
    on_after_merge,
    on_episode_start,
    on_sub_episode_start,
)
from anchor_nav.vlm_adapter import probe_vlm_or_raise


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


def _setup_run_logging():
    log_dir = os.path.join("output_dirs", "anchor_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"goat-nav-anchor-{ts}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)
    print(f"[AnchorNav] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[AnchorNav] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fp.close()

    atexit.register(_cleanup)


_setup_run_logging()

# hyperparameter
data_set_path = "/disks/amax_robot_dataset/embodied/embodied_bench_data/our-set/goat_full_set.json"
navigation_data_path = "/disks/amax_robot_dataset/embodied/embodied_bench_data/goat/"
hm3d_data_base_path = "/home/chenlin/krona/MTU3D/datascene"
pq3d_stage1_path = "/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all"
pq3d_stage2_path = "/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-goat"
output_path = "./output_dirs/goat-anchor-test.json"
enable_visualization = False
decision_num_min = 3
visible_radius = 3

# load navigation data
navigation_data_dict = {"val_seen": {}, "val_seen_synonyms": {}, "val_unseen": {}}
split_list = ["val_seen", "val_seen_synonyms", "val_unseen"]
raw_scan_ids = set([d for d in os.listdir(hm3d_data_base_path) if os.path.isdir(os.path.join(hm3d_data_base_path, d))])
for split in split_list:
    data_dir = os.path.join(navigation_data_path, split, "content")
    file_list = [f for f in os.listdir(data_dir) if f[0] != "."]
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            simplified_scan_id = file_name.split(".")[0]
            raw_scan_id = [pa for pa in raw_scan_ids if simplified_scan_id in pa][0]
            new_data = {}
            new_data["episodes"] = data["episodes"]
            new_data["goals_by_category"] = dict([(k.split("glb_")[-1], v) for k, v in data["goals"].items()])
            navigation_data_dict[split][raw_scan_id] = new_data

# load image feature
image_feat_dir = os.path.join("/disks/amax_robot_dataset/embodied/embodied_scan_vle_data", "goat-clip-feat")
image_feat_dict = {"val_seen": {}, "val_seen_synonyms": {}, "val_unseen": {}}
for split in split_list:
    file_list = os.listdir(os.path.join(image_feat_dir, split))
    for f_name in file_list:
        image_feat = torch.load(os.path.join(image_feat_dir, split, f_name), map_location="cpu")
        image_feat_dict[split][f_name.split(".")[0]] = image_feat

# load data set
data_set = json.load(open(data_set_path, "r"))

# record result
if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
else:
    result_dict = {
        "val_seen": {"object": [], "description": [], "image": []},
        "val_seen_synonyms": {"object": [], "description": [], "image": []},
        "val_unseen": {"object": [], "description": [], "image": []},
    }

# filter out data in dataset which is already in result dict
for split in split_list:
    existing_episodes = {(result["scan_id"], result["episode_index"]) for goal_type in result_dict[split] for result in result_dict[split][goal_type]}
    data_set[split] = [episode for episode in data_set[split] if (episode["scan_id"], episode["episode_index"]) not in existing_episodes]

# load model
pq3d_model = AnchorPQ3DModel(pq3d_stage1_path, pq3d_stage2_path, min_decision_num=decision_num_min)
plugin_cfg_path = os.path.join(os.path.dirname(__file__), "inference_config_anchor.yaml")
plugins = load_anchor_plugin_config(plugin_cfg_path)
anchor_ctx = AnchorNavContext(plugin_flags=plugins)
print(f"[AnchorNav] plugin flags: {plugins}")
probe_vlm_or_raise()

for split in split_list:
    for cur_data in data_set[split]:
        scene_id = cur_data["scan_id"]
        clean_scene_id = scene_id.split("-")[-1]
        scene_dir = os.path.join(hm3d_data_base_path, scene_id)
        scene_path_candidates = [
            os.path.join(scene_dir, f"{clean_scene_id}.basis.glb"),
            os.path.join(scene_dir, f"{clean_scene_id}.glb"),
        ]
        scene_path = None
        for candidate in scene_path_candidates:
            if os.path.exists(candidate):
                scene_path = candidate
                break
        if scene_path is None:
            print(f"Skip episode because scene file does not exist. Checked: {scene_path_candidates}")
            continue
        episode_index = cur_data["episode_index"]
        cur_episode = navigation_data_dict[split][scene_id]["episodes"][episode_index]

        pq3d_model.reset()

        all_descs = []
        has_description_task = False
        for task in cur_episode["tasks"]:
            gc, gt = task[0], task[1]
            if gt == "description":
                has_description_task = True
                g_obj_id = task[2]
                g_list = [g for g in navigation_data_dict[split][scene_id]["goals_by_category"][gc] if g["object_id"] == g_obj_id]
                all_descs.append(g_list[0]["lang_desc"] if g_list else gc)
            else:
                all_descs.append(gc)
        on_episode_start(
            anchor_ctx,
            all_task_descriptions=all_descs,
            pq3d_model=pq3d_model,
            has_description_task=has_description_task,
        )

        start_position = cur_episode["start_position"]
        start_rotation = cur_episode["start_rotation"]

        sim_settings = OmegaConf.load("configs/habitat/goat_sim_config.yaml")
        goat_agent_setting = OmegaConf.load("configs/habitat/goat_agent_config.yaml")
        sim_settings["scene"] = scene_path
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

        decision_num = 0
        global_color_list = []
        visited_frontier_set = set()

        for sub_episode_index in range(len(cur_episode["tasks"])):
            on_sub_episode_start(anchor_ctx, sub_episode_index)

            cur_sub_episode = cur_episode["tasks"][sub_episode_index]
            if len(cur_sub_episode) == 3:
                goal_category, goal_type, goal_object_id = cur_sub_episode
            else:
                goal_category, goal_type, goal_object_id, goal_image_id = cur_sub_episode

            if goal_type == "object":
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]["goals_by_category"][goal_category]]
            elif goal_type == "description":
                goals = [g for g in navigation_data_dict[split][scene_id]["goals_by_category"][goal_category] if g["object_id"] == goal_object_id]
                sentence = goals[0]["lang_desc"]
                assert len(goals) == 1
            elif goal_type == "image":
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]["goals_by_category"][goal_category] if g["object_id"] == goal_object_id]
                goal_image_feat = image_feat_dict[split][scene_id][int(goal_object_id.split("_")[1])][goal_image_id]
                assert len(goals) == 1
            else:
                raise ValueError(f"unknown goal type: {goal_type}")
            print(sentence)

            total_steps = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            episode_cum_distance = 0
            goto_color_list = []
            goto_depth_list = []
            goto_agent_state_list = []

            while total_steps < 500:
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
                    observations = sim.step(action=action)
                    color = observations["color_sensor"][:, :, :3]
                    color_list.append(color)
                    global_color_list.append(color)
                    depth = observations["depth_sensor"][:, :]
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
                    if goal_type == "image":
                        target_position, is_final_decision = pq3d_model.decision(
                            color_list,
                            depth_list,
                            agent_state_list,
                            frontier_waypoints,
                            sentence,
                            decision_num,
                            goal_image_feat,
                        )
                    else:
                        target_position, is_final_decision = pq3d_model.decision(
                            color_list,
                            depth_list,
                            agent_state_list,
                            frontier_waypoints,
                            sentence,
                            decision_num,
                        )
                except Exception as e:
                    print(f"Error in decision making, episode_id: {cur_episode['episode_id']}, scene_id: {scene_id}, {e}")
                    sys.exit(1)
                    break

                on_after_merge(anchor_ctx, pq3d_model.representation_manager)
                target_position, is_final_decision = on_after_decision(
                    anchor_ctx,
                    target_position=target_position,
                    is_final_decision=is_final_decision,
                    representation_manager=pq3d_model.representation_manager,
                    frontier_waypoints=frontier_waypoints,
                    color_list=color_list,
                    agent_state_list=agent_state_list,
                    sub_episode_index=sub_episode_index,
                    pq3d_model=pq3d_model,
                )
                decision_num += 1

                if not is_final_decision:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(
                    path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
                )
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except Exception:
                    if not path_finder.is_navigable(target_on_navmesh):
                        print("Target is not navigable")
                    if not path_finder.is_navigable(agent_state.position):
                        print("Agent is not navigable")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = agent_state.position
                    path.requested_end = target_on_navmesh
                    if sim.pathfinder.find_path(path):
                        print(f"geodesic_distance: {path.geodesic_distance}")
                    else:
                        print("cannt find path")
                    action_list = []
                    break

                goto_color_list = []
                goto_depth_list = []
                goto_agent_state_list = []
                for action in action_list:
                    if action:
                        observations = sim.step(action=action)
                        global_color_list.append(observations["color_sensor"][:, :, :3])
                        agent_state = agent.get_state()
                        color = observations["color_sensor"][:, :, :3]
                        goto_color_list.append(color)
                        depth = observations["depth_sensor"][:, :]
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
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state

                if is_final_decision:
                    break

            if enable_visualization and global_color_list:
                height, width, _ = global_color_list[0].shape
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
                print("goal is not navigatable")
                start_end_geo_distance = np.inf

            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                agent_end_geo_distance = path.geodesic_distance
            else:
                agent_end_geo_distance = np.inf

            if start_end_geo_distance == np.inf:
                sr = 1
                spl = 1
            elif agent_end_geo_distance == np.inf:
                sr = 0
                spl = 0
            else:
                sr = agent_end_geo_distance <= 0.25
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            result_dict[split][goal_type].append(
                {
                    "scan_id": scene_id,
                    "episode_index": episode_index,
                    "sub_episode_index": sub_episode_index,
                    "sr": sr,
                    "spl": spl,
                    "object_category": goal_category,
                }
            )
            print(
                f"SR: {sr}, SPL: {spl}, Agent start position: {start_position}, Agent position: {agent_state.position}, "
                f"Goal positions: {[g['position'] for g in goals]}, Object category: {goal_category}, "
                f"Decision number: {decision_num}, goal type: {goal_type}"
            )

        with open(output_path, "w") as f:
            json.dump(result_dict, f)

# Calculate and print average SPL and SR for each split and goal type.
for split in split_list:
    for goal_type in ["object", "description", "image"]:
        total_sr = 0
        total_spl = 0
        category_sr_spl = defaultdict(lambda: {"sr": 0, "spl": 0, "count": 0})
        count = 0
        for result in result_dict[split][goal_type]:
            total_sr += result["sr"]
            total_spl += result["spl"]
            category = result["object_category"]
            category_sr_spl[category]["sr"] += result["sr"]
            category_sr_spl[category]["spl"] += result["spl"]
            category_sr_spl[category]["count"] += 1
            count += 1
        avg_sr = total_sr / count if count > 0 else 0
        avg_spl = total_spl / count if count > 0 else 0
        print(f"Split: {split}, Goal Type: {goal_type}, Average SR: {avg_sr}, Average SPL: {avg_spl}")

for split in split_list:
    total_sr = 0
    total_spl = 0
    count = 0
    for goal_type in ["object", "description", "image"]:
        for result in result_dict[split][goal_type]:
            total_sr += result["sr"]
            total_spl += result["spl"]
            count += 1
    avg_sr = total_sr / count if count > 0 else 0
    avg_spl = total_spl / count if count > 0 else 0
    print(f"Split: {split}, Overall Average SR: {avg_sr}, Overall Average SPL: {avg_spl}")

