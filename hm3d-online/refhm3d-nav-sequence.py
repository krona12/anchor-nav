from collections import defaultdict
import gzip
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from omegaconf import OmegaConf
import torch
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import visualize_numpy_data, convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
import cv2
from data_utils import PQ3DModel
from tqdm import tqdm
import builtins
import datetime
import time

##### ************** Personal Modification **********
# from memory import DynamicMemory

# record elapsed time.
import time
program_start = time.time()
old_print = print
def print(*args, **kwargs):
    now = time.time()
    elapsed = now - program_start
    # 格式化成 HH:MM:SS
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    time_str = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    # old_print(f"{time_str} | Elapsed {elapsed_str}", *args, **kwargs)
    old_print(f"Elapsed {elapsed_str}", *args, **kwargs)


import argparse
from utils import sequence_compute_metric_results

parser = argparse.ArgumentParser(description="Run RefHM3D evaluation with hyperparameters")
parser.add_argument("--start_ratio", type=float, default=0.0, help="Dataset start ratio (e.g., 0.0)")
parser.add_argument("--end_ratio", type=float, default=0.2, help="Dataset end ratio (e.g., 0.2)")
parser.add_argument("--concise_description", action="store_true", help="Use concise descriptions instead of detailed ones")
args = parser.parse_args()


black_task_ids = \
   ['00800-TEEsavR23oF_room_74', '00810-CrMo8WxCyVb_object_30', '00810-CrMo8WxCyVb_object_31', '00810-CrMo8WxCyVb_region_75', '00810-CrMo8WxCyVb_region_76', '00810-CrMo8WxCyVb_room_53', '00810-CrMo8WxCyVb_room_54', '00820-mL8ThkuaVTM_object_29', '00820-mL8ThkuaVTM_region_47', '00820-mL8ThkuaVTM_room_39', '00823-7MXmsvcQjpJ_room_100', '00823-7MXmsvcQjpJ_room_101', '00823-7MXmsvcQjpJ_room_74', '00823-7MXmsvcQjpJ_room_98', '00823-7MXmsvcQjpJ_room_99', '00829-QaLdnwvtxbs_room_14', '00829-QaLdnwvtxbs_room_16', '00829-QaLdnwvtxbs_room_21', '00829-QaLdnwvtxbs_room_22', '00829-QaLdnwvtxbs_room_23', '00829-QaLdnwvtxbs_room_24', '00829-QaLdnwvtxbs_room_25', '00829-QaLdnwvtxbs_room_5', '00829-QaLdnwvtxbs_room_7', '00829-QaLdnwvtxbs_room_9', '00832-qyAac8rV8Zk_region_18', '00832-qyAac8rV8Zk_room_18', '00839-zt1RVoi7PcG_object_25', '00839-zt1RVoi7PcG_region_70', '00839-zt1RVoi7PcG_room_54', '00862-LT9Jq6dN3Ea_object_46', '00862-LT9Jq6dN3Ea_region_135', '00862-LT9Jq6dN3Ea_region_37', '00862-LT9Jq6dN3Ea_room_21', '00862-LT9Jq6dN3Ea_room_93', '00871-VBzV5z6i1WS_object_21', '00871-VBzV5z6i1WS_object_28', '00871-VBzV5z6i1WS_object_29', '00871-VBzV5z6i1WS_object_38', '00871-VBzV5z6i1WS_object_8', '00871-VBzV5z6i1WS_region_32', '00871-VBzV5z6i1WS_region_58', '00871-VBzV5z6i1WS_region_67', '00871-VBzV5z6i1WS_region_68', '00871-VBzV5z6i1WS_region_77', '00871-VBzV5z6i1WS_room_21', '00871-VBzV5z6i1WS_room_39', '00871-VBzV5z6i1WS_room_46', '00871-VBzV5z6i1WS_room_47', '00871-VBzV5z6i1WS_room_56', '00873-bxsVRursffK_object_29', '00873-bxsVRursffK_region_52', '00873-bxsVRursffK_room_39', '00876-mv2HUxq3B53_object_16', '00876-mv2HUxq3B53_object_31', '00876-mv2HUxq3B53_object_40', '00876-mv2HUxq3B53_region_33', '00876-mv2HUxq3B53_region_51', '00876-mv2HUxq3B53_region_69', '00876-mv2HUxq3B53_region_81', '00876-mv2HUxq3B53_region_92', '00876-mv2HUxq3B53_region_93', '00876-mv2HUxq3B53_room_19', '00876-mv2HUxq3B53_room_31', '00876-mv2HUxq3B53_room_45', '00876-mv2HUxq3B53_room_50', '00876-mv2HUxq3B53_room_59', '00877-4ok3usBNeis_object_44', '00877-4ok3usBNeis_region_58', '00877-4ok3usBNeis_region_8', '00877-4ok3usBNeis_room_52', '00877-4ok3usBNeis_room_9', '00878-XB4GS9ShBRE_object_18', '00878-XB4GS9ShBRE_region_33', '00878-XB4GS9ShBRE_room_32', '00880-Nfvxx8J5NCo_object_13', '00880-Nfvxx8J5NCo_object_30', '00880-Nfvxx8J5NCo_object_31', '00880-Nfvxx8J5NCo_region_22', '00880-Nfvxx8J5NCo_region_45', '00880-Nfvxx8J5NCo_region_46', '00880-Nfvxx8J5NCo_room_23', '00880-Nfvxx8J5NCo_room_44', '00880-Nfvxx8J5NCo_room_45', '00890-6s7QHgap2fW_region_19', '00890-6s7QHgap2fW_room_16', '00891-cvZr5TUy5C5_object_12', '00891-cvZr5TUy5C5_object_43', '00891-cvZr5TUy5C5_object_44', '00891-cvZr5TUy5C5_region_109', '00891-cvZr5TUy5C5_region_110', '00891-cvZr5TUy5C5_region_111', '00891-cvZr5TUy5C5_region_112', '00891-cvZr5TUy5C5_region_39', '00891-cvZr5TUy5C5_region_40', '00891-cvZr5TUy5C5_region_44', '00891-cvZr5TUy5C5_room_34', '00891-cvZr5TUy5C5_room_35', '00891-cvZr5TUy5C5_room_36', '00891-cvZr5TUy5C5_room_92', '00891-cvZr5TUy5C5_room_93', '00891-cvZr5TUy5C5_room_94', '00891-cvZr5TUy5C5_room_95', '00891-cvZr5TUy5C5_room_96', '00891-cvZr5TUy5C5_room_97', '00894-HY1NcmCgn3n_instance_7']
print(f"NUMBER OF BLACK IDS: {len(black_task_ids)}")


ckpt = "goat" # ovon  goat
# hyperparameter
hm3d_data_base_path = "../temp_datasets/hm3d/val"
pq3d_stage1_path = "checkpoint/stage1-pretrain-all"
pq3d_stage2_path = f"checkpoint/stage2-fine-tune-{ckpt}"
enable_visualization = False
decision_num_min = 3
visible_radius = 3
# ''' trigger to select what lang version '''
use_api = False

##########################
# ADELAIDE: LOAD OUR DATA
##########################
concise_description_tag = args.concise_description  # comprehensive text or concise text
start_ratio, end_ratio = args.start_ratio, args.end_ratio
folder_name = f"output_dirs_newjson_{ckpt}ckpt_newspl"
folder_name = f"toy"
os.makedirs(folder_name, exist_ok=True)
if concise_description_tag:
    output_path = f"./{folder_name}/refhm3d_seq_concisedesc_{start_ratio}_{end_ratio}.json"
else:
    output_path = f"./{folder_name}/refhm3d_seq_{start_ratio}_{end_ratio}.json"
# ***************************
# LOAD DATASET
# ******************************
navigation_data_path = "../temp_datasets/refhm3d_final_new"
scene_data_list = sorted([x for x in os.listdir(navigation_data_path) if x.endswith(".json.gz")])
num_scene = len(scene_data_list)
scene_data_list = scene_data_list[int(start_ratio * num_scene):int(end_ratio * num_scene)]
print(f"\n\nTotal selected number of scenes {len(scene_data_list)}: {scene_data_list}\n\n")
##########################


# record result
if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    sequence_compute_metric_results(result_dict)
    existing_episodes = {"_".join([result['scene_name'], result['navigation_type'], str(result['episode_id'])])
                         for goal_type in result_dict for result in result_dict[goal_type]}
else:
    existing_episodes = {}
    result_dict = {'sequence': []}

# load pq3d model
pq3d_model = PQ3DModel(pq3d_stage1_path, pq3d_stage2_path, min_decision_num=decision_num_min)

for scene_data_file in tqdm(scene_data_list, desc=f"*** Scene ***"):
    ##########################
    # ADELAIDE: LOAD OUR DATA [same for different models]
    ##########################
    scene_name = scene_data_file.split(".")[0]  # 00800-TEEsavR23oF
    with gzip.open(os.path.join(navigation_data_path, scene_data_file), 'rt', encoding='utf-8') as f:
        scene_data = json.load(f)
        region_to_annot_dict = scene_data['region_annotation']
        episode_mapping = {"object": scene_data['episodes_by_object_level'],
                           "room": scene_data['episodes_by_room_level'],
                           "region": scene_data['episodes_by_region_level'],
                           "instance": scene_data['episodes_by_instance_level']}
        all_navigation_goals_dict = {x['object_id']: x for x in scene_data["goals"]}  # object_id: object_info dictionary

    # ************* For each episode *************
    print(f"\n\n\n\n\n*************************** Processing {scene_data_file}, Total Episode {len(scene_data['episode_by_sequence'])} ***************************")
    for episode_idx, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc=f"=== Episode ==="):
        # reset pq3d
        pq3d_model.reset()
        global_color_list = []
        decision_num = 0
        visited_frontier_set = set()
        action_ct = 0

        # load target
        start_position = cur_episode['start_position']
        start_rotation = cur_episode['start_rotation']
        episode_id, navigation_type = cur_episode['episode_id'], cur_episode["navigation_type"]
        if "_".join([scene_name, navigation_type, str(episode_id)]) in existing_episodes:
            print("_".join([scene_name, navigation_type, str(episode_id)]), " already processed, skipped")
            continue

        ''' get simulator '''
        sim_settings = OmegaConf.load('configs/habitat/goat_sim_config.yaml')
        goat_agent_setting = OmegaConf.load('configs/habitat/goat_agent_config.yaml')
        sim_settings['scene'] = os.path.join(hm3d_data_base_path, scene_name, f"{scene_name.split('-')[-1]}.basis.glb")
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder

        # get fronier param
        map_resolution = 512
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution,  draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map) # (512, 528)
        area_thres_in_pixels =  convert_meters_to_pixel(9, map_resolution, sim)  # 217
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)

        for idx, cur_task in enumerate(cur_episode['task_sequence']):
            ''' get single task info [same as prev episode info] '''
            task_type, task_idx = cur_task
            cur_task = episode_mapping[task_type][task_idx]

            # GET TARGET GOALS & INSTRUCTION
            goals_ids = cur_task["target_object_ids"]
            assert len(goals_ids) > 0, f"{'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])} should have at least one goal"
            goals = [all_navigation_goals_dict[x] for x in goals_ids]
            if task_type == 'object':
                sentence = cur_task['object_category']
                goal_category = cur_task['object_category']
            elif task_type == 'room':
                sentence = f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
                goal_category = cur_task['object_category']
            elif task_type == 'region':
                region_desc = region_to_annot_dict[cur_task['region_id']]['shortest_description'] if concise_description_tag \
                    else region_to_annot_dict[cur_task['region_id']]['comprehensive_description']
                sentence = (f"{cur_task['object_category']} in the {region_to_annot_dict[cur_task['region_id']]['region_category'].lower()} "
                            f"that has {region_desc}")
                goal_category = cur_task['object_category']
            elif task_type == 'instance':
                sentence = all_navigation_goals_dict[cur_task['instance_id']]['annot_unique_concise_description'] if concise_description_tag \
                    else all_navigation_goals_dict[cur_task['instance_id']]['annot_unique_normal_description']
                goal_category = goals[0]['object_category']
            print(f"\n\nBegin to process [{'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])}] type: [{task_type}], Question: [{sentence}]\n")

            # start decision
            # sub_episdoe global parameter
            total_steps, rotation_steps = 0, 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            print(f"Current start position is {sub_episode_start_position}")
            episode_cum_distance = 0
            # loop parameter
            goto_color_list = []
            goto_depth_list = []
            goto_agent_state_list = []
            # visited frontier
            t_episode_start = time.perf_counter()
            while total_steps < 400:
                '''
                1. observe and get frontier points based on top-down map and explore mask
                2. model to predict target point to go
                3. use simulator to automatically generate actions
                '''
                # spin around
                color_list = []
                depth_list = []
                agent_state_list = []
                # subsample at most 6 frames from goto list, use interval sample
                if len(goto_color_list) > 6:
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)

                ''' **** spin each 30 degree to get panorama view (each turn is a step, but wont let redundancy?) ***** '''
                action_list = ['turn_left'] * 12
                for action in action_list:
                    obervations = sim.step(action=action)
                    color = obervations['color_sensor'][:, :, :3] # (h,w,4) 0-255
                    color_list.append(color)
                    global_color_list.append(color)
                    depth = obervations['depth_sensor'][:, :] # (h,w) float
                    depth_list.append(depth)
                    agent_state = agent.get_state()
                    agent_state_list.append(agent_state)
                    if enable_visualization:
                        # Save the current color image to color.png
                        cv2.imwrite('color.png', color)
                    # top-down 栅格地图上，根据智能体当前位置与朝向，画出一块 视野扇形区域 after observation
                    fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                    total_steps += 1
                    rotation_steps += 1

                agent_state = agent.get_state()
                # compute frontier [fog_of_war_mask is explored_map]
                frontier_waypoints = detect_frontier_waypoints(top_down_map, fog_of_war_mask, area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization)
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = frontier_waypoints[:, ::-1]
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
                # filter out visited frontier (list of coords)
                # [array([    -6.4429,      3.1134,     -3.8503], dtype=float32), array([    -6.4959,      3.1134,     -3.7231], dtype=float32)]
                frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set]

                # decision
                try:
                    target_position, is_final_decision = pq3d_model.decision(color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num)
                except Exception as e:
                    print(f"Error in decision making, episode_id: {episode_id}, task_id: {idx}, scene_id: {scene_name}, {e}")
                    # sys.exit(1)
                    break
                decision_num += 1
                # add frontier to visited frontier
                if not is_final_decision:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))

                # goto 把目标点对齐到可走网格上，并用 Habitat 自带的贪心测地方向跟随器来产生动作，让智能体朝目标前进
                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                try:
                    # TODO: seems it use GT path directly move to waypoint OR a past position
                    action_list = follower.find_path(target_on_navmesh)
                except:
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
                        obervations = sim.step(action=action)
                        global_color_list.append(obervations['color_sensor'][:, :, :3])
                        agent_state = agent.get_state()
                        color = obervations['color_sensor'][:, :, :3] # (h,w,4) 0-255
                        depth = obervations['depth_sensor'][:, :] # (h,w) float
                        goto_color_list.append(color)
                        goto_depth_list.append(depth)
                        goto_agent_state_list.append(agent_state)
                        fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                        total_steps += 1
                        if action in ['turn_left', 'turn_right']:
                            rotation_steps += 1
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                # break on final decision
                if is_final_decision:
                    break

            t_episode_end = time.perf_counter()
            episode_time = float(t_episode_end - t_episode_start)
            # Save color_list to video
            if enable_visualization:
                height, width, layers = global_color_list[0].shape
                video = cv2.VideoWriter(f'video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
                for color_frame in global_color_list:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                    video.write(color_frame)
                video.release()
                pq3d_model.representation_manager.save_colored_point_cloud()

            # compute metric
            agent_state = agent.get_state()
            view_points = [
                        view_point["agent_state"]["position"]
                        for goal in goals
                        for view_point in goal["view_points"]
            ]
            # computer start end geodesic distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = sub_episode_start_position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                start_end_geo_distance = path.geodesic_distance
            else:
                print(f"Goal is not navigable: {'_'.join([scene_name, navigation_type, str(episode_id), str(idx)])}")
                start_end_geo_distance = np.inf
            # compute agent current distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):  # TODO: add distance to surface
                agent_end_geo_distance = path.geodesic_distance  # GEO distance between current position and target position
            else:
                agent_end_geo_distance = np.inf

            # compute success rate
            if start_end_geo_distance == np.inf:  # TODO: WTF???????
                sr = 0
                spl = 0
            elif agent_end_geo_distance == np.inf:
                sr = 0
                spl = 0
            else:
                sr = agent_end_geo_distance <= 0.25
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            
            # store results
            current_rot = agent_state.rotation
            result_dict[navigation_type].append(
                {'scene_name': scene_name, 'episode_id': episode_id, 'task_id': idx, 'navigation_type': navigation_type,
                 'sr': sr, 'spl': spl, 'object_category': goal_category,
                 })

            print(f"===Episode_id {episode_id} task_id {idx}===\nSR: {sr}, SPL: {spl}, Object category: {goal_category}, goal type: {navigation_type}===\n")

        # store results after each episode
        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)
    sequence_compute_metric_results(result_dict)
sequence_compute_metric_results(result_dict)