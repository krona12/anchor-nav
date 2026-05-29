"""
Habitat-Sim 交互式场景浏览与全景截图脚本。

功能概述：
1. 加载 HM3D / Habitat 场景；
2. 随机初始化 agent 到可导航位置；
3. 使用键盘控制 agent 在场景中移动、转向、抬头/低头；
4. 实时显示 RGB 视角和 top-down map；
5. 按 k 键保存当前位置的三向全景截图；
6. 按 n / m 键在楼层之间切换。

注意：
- 本脚本主要用于 demo、可视化和论文 figure 数据采集。
- 代码默认依赖本地 HM3D 数据路径、Habitat-Sim、Habitat-Lab、OpenCV GUI。
- 运行环境需要支持 cv2.imshow，例如本地桌面环境；纯服务器终端可能无法显示窗口。
"""

import os, sys
import json
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import GPUtil
import argparse
from collections import Counter
import shutil
from typing import Dict, Iterable, List, Optional, Set
from typing import Dict, Union, cast
import numpy as np
import cv2
import time
from datetime import datetime
import imageio

import habitat
from habitat.tasks.utils import compute_pixel_coverage
from habitat.config.default import get_agent_config, get_config
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    HabitatSimSemanticSensorConfig,
)
import habitat_sim
from habitat_sim.simulator import Simulator
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import get_topdown_map

print("habitat-lab:", habitat.__version__)
print("habitat-sim:", habitat_sim.__version__)

# 关闭 Habitat / Magnum 的大部分日志，避免终端输出太乱。
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# ********** Custom package ************
# 自定义工具函数：
# - is_on_ceiling: 判断物体是否在天花板上；
# - most_common_value: 统计最常见值，一般用于估计楼层高度。
from low_level_utils import is_on_ceiling, most_common_value


def get_floor_height(sim, search_center: np.ndarray) -> float:
    """估计某个 3D 点所在的地面高度。

    思路：
    1. 先把物体 bbox 中心点 snap 到 navmesh 上；
    2. 如果中心点低于 snap 后的可导航点，就逐步向下搜索；
    3. 返回最近的可导航地面高度。

    参数：
        sim: Habitat-Sim simulator。
        search_center: 通常是物体 bbox 的中心点，shape 为 (3,)。

    返回：
        snapped[1]: 估计出的 floor height。
    """
    point = np.asarray(search_center)[:, None]
    snapped = sim.pathfinder.snap_point(point)

    # 如果物体中心点低于 snap 后的位置，则逐步向下尝试，最多向下 trace 2m。
    tries = 0
    while point[1, 0] < snapped[1]:
        point[1, 0] -= 0.05
        snapped = sim.pathfinder.snap_point(point)
        tries += 1
        if tries > 40:  # trace 2.0m down.
            break
    return snapped[1]


def get_floor_heights_fast(sim):
    """快速采样场景中的可能楼层高度。

    做法：
    - 从 navmesh 中随机采样 2000 个可导航点；
    - 收集它们的 y 坐标；
    - round 到 0.01m 后去重，作为候选楼层高度。

    这里不是严格的楼层分割，只是为了交互浏览时快速切换楼层。
    """
    heights = []
    for _ in range(2000):
        p = sim.pathfinder.get_random_navigable_point()
        heights.append(round(p[1], 2))
    heights = list(set(heights))
    heights.sort()
    return heights


def generate_habitat_sim(i: int, scene: str, args, sensor_height=1.31, resolution=[480, 640]):
    """创建并初始化 Habitat-Sim simulator。

    主要配置：
    - 选择 GPU；
    - 设置 scene dataset config；
    - 创建 RGB 和 depth sensor；
    - 设置 agent 的动作空间；
    - 重新生成 navmesh。

    参数：
        i: 当前任务编号，用于选择 GPU。
        scene: 场景 id，例如 "LT9Jq6dN3Ea"。
        args: 命令行参数对象。
        sensor_height: 相机高度。
        resolution: 图像分辨率，格式 [H, W]。

    返回：
        sim: 初始化好的 habitat_sim.Simulator。
    """
    root_path, split, num_gpus, tasks_per_gpu = args.root_path, args.split, args.num_gpus, args.tasks_per_gpu

    # 有语义 mask 的配置。
    SCENE_CFG = os.path.join(root_path, "hm3d_annotated_basis.scene_dataset_config.json")
    # 无语义 mask 的配置。当前代码会覆盖上一行，因此实际使用的是这个配置。
    SCENE_CFG = os.path.join(root_path, "hm3d_basis.scene_dataset_config.json")

    # 优先选择空闲显存最多的 GPU；如果没有可用 GPU，则按任务编号轮转。
    deviceIds = GPUtil.getAvailable(order="memory", limit=1, maxLoad=1.0, maxMemory=1.0)
    if i < num_gpus * tasks_per_gpu or len(deviceIds) == 0:
        deviceId = i % num_gpus
    else:
        deviceId = deviceIds[0]

    # Simulator 基础配置。
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = (deviceId)
    # sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim_cfg.enable_hbao = True
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY
    sim_cfg.create_renderer = True
    sim_cfg.scene_dataset_config_file = SCENE_CFG
    sim_cfg.scene_id = scene

    sensor_specs = []

    # 当前只使用 color 和 depth。semantic sensor 暂时没有启用。
    # 如果需要语义图，可以把 semantic sensor 加回来。
    # for name, sensor_type in (
    #         zip(["color", "depth", "semantic"], [habitat_sim.SensorType.COLOR, habitat_sim.SensorType.DEPTH, habitat_sim.SensorType.SEMANTIC])):
    for name, sensor_type in (zip(["color", "depth"], [habitat_sim.SensorType.COLOR, habitat_sim.SensorType.DEPTH])):
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = f"{name}"
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = resolution  # [480, 640]、[512, 512]、[680, 1200] 等都可以。
        sensor_spec.position = [0.0, sensor_height, 0.0]
        sensor_spec.hfov = 120
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)

    # Agent 配置。
    # 注意：agent 的 height/radius 与 navmesh 设置会影响可导航区域和碰撞。
    agent_cfg = AgentConfiguration(
        height=1.41,
        radius=0.17,
        sensor_specifications=sensor_specs,
        action_space={
            "look_up": habitat_sim.ActionSpec("look_up", habitat_sim.ActuationSpec(amount=30),),
            "look_down": habitat_sim.ActionSpec("look_down", habitat_sim.ActuationSpec(amount=30),),
            "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=30.0),),
            "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=30.0),),
            "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25),),
        },
    )

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"

    # 重新计算 navmesh。
    # 这里把 agent_radius 设得很小，是为了尽量捕获/靠近更多 instance，方便 demo 和截图。
    # 如果用于真实导航评估，建议使用和 agent_cfg.radius 一致的半径，例如 0.17。
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = 0.01  # should be 0.17 (but I assign 0.01 to ensure capture all instances)
    navmesh_settings.agent_height = 1.41  # TODO: seems should all be small
    navmesh_success = sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings  # , include_static_objects=False
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    assert navmesh_success, "Failed to build the navmesh!"
    return sim


def get_topdown_map_(sim, height=None, meters_per_pixel=0.05):
    """生成某个高度切片上的 top-down map。

    map 中的原始类别：
    - invalid
    - free
    - occupied

    这里将其重新映射成 RGB 颜色，方便保存和可视化。
    """
    if height is None:
        height = sim.get_agent_state().position[1]

    topdown_map = get_topdown_map(
        sim.pathfinder,
        height,
        map_resolution=1024,
        meters_per_pixel=meters_per_pixel,
    )
    # topdown_map = maps.get_topdown_map_from_sim(cast("HabitatSim", sim), meters_per_pixel=0.025)

    recolor_map = np.array(
        [[255, 255, 255],  # invalid: 白色
         [128, 128, 128],  # free: 灰色
         [0, 0, 0]]        # occupied: 黑色
    )
    topdown_map = recolor_map[topdown_map]
    return topdown_map


def get_topdown_map_visualize(sim, agent, trajectory, shortest_path=None, meters_per_pixel=0.025):
    """生成用于交互窗口显示的 top-down map，并画出轨迹。

    显示内容：
    - 黑色背景：不可通行区域；
    - 灰色区域：可通行区域；
    - 蓝色线：agent 当前已走过的轨迹；
    - 绿色线：可选的 shortest path；
    - 红色方块：起点和当前终点。
    """
    state = agent.get_state()
    height = state.position[1]

    topdown_bool = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    h, w = topdown_bool.shape

    # 官方风格：黑色背景。
    topdown = np.zeros((h, w, 3), dtype=np.uint8)

    # free space 显示为灰色。
    topdown[topdown_bool] = (200, 200, 200)
    bounds = sim.pathfinder.get_bounds()
    min_bound = bounds[0]

    def world_to_map(pos):
        """将 Habitat 世界坐标转换为 top-down map 像素坐标。

        Habitat 世界坐标通常使用 x-z 平面表示地面位置，y 表示高度。
        """
        x = int((pos[0] - min_bound[0]) / meters_per_pixel)
        y = int((pos[2] - min_bound[2]) / meters_per_pixel)
        return x, y

    # 画 agent 的历史轨迹：蓝色实线。
    if len(trajectory) > 1:
        pts = []
        for p in trajectory:
            x, y = world_to_map(p)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(
            topdown, [pts], isClosed=False, color=(255, 0, 0), thickness=3,
        )

    # 画 shortest path：绿色实线。
    if shortest_path is not None and len(shortest_path) > 1:
        pts = []
        for p in shortest_path:
            x, y = world_to_map(p)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(
            topdown, [pts], isClosed=False, color=(0, 255, 0), thickness=3,
        )

    # 画起点：红色方块。
    if len(trajectory) > 0:
        x, y = world_to_map(trajectory[0])
        cv2.rectangle(
            topdown, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), -1,
        )

    # 画当前点 / 终点：红色方块。
    if len(trajectory) > 0:
        x, y = world_to_map(trajectory[-1])
        cv2.rectangle(
            topdown, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), -1,
        )

    return topdown


def capture_panorama(sim, agent, step_id, output_dir):
    """在当前 agent 位置保存三张不同朝向的 RGB 图像。

    当前实现：
    - 先保存当前朝向；
    - 每保存一张后，连续 turn_left 4 次；
    - 每次 turn_left 是 30 度，所以 4 次是 120 度；
    - 共保存 3 张图，覆盖 360 度。

    输出文件：
        step_XXX_view_0.png
        step_XXX_view_1.png
        step_XXX_view_2.png
    """
    # 保存当前状态，之后恢复原始朝向。
    state = agent.get_state()
    original_rotation = state.rotation

    for view_id in range(0, 3):
        # 获取当前传感器观测并保存 RGB 图。
        obs = sim.get_sensor_observations()
        rgb = obs["color"][:, :, :3]
        filename = f"{output_dir}/step_{step_id:03d}_view_{view_id}.png"
        imageio.imwrite(filename, rgb, compress_level=0)

        # 向左转 120 度，准备保存下一个方向。
        for _ in range(4):
            sim.step("turn_left")
        print("saved:", filename)

    # 恢复原始朝向，避免截图改变 agent 后续控制方向。
    state.rotation = original_rotation
    agent.set_state(state)


def teleport_to_floor(sim, agent, target_height):
    """将 agent teleport 到接近指定高度的某个可导航点。

    用于楼层切换：
    - 随机采样可导航点；
    - 找到 y 坐标最接近 target_height 的点；
    - 设置 agent 位置到该点。
    """
    best = None
    best_diff = 1e9

    # 随机采样多个 navmesh 点，找和目标楼层高度最接近的点。
    for _ in range(2000):
        p = sim.pathfinder.get_random_navigable_point()
        diff = abs(p[1] - target_height)
        if diff < best_diff:
            best = p
            best_diff = diff

        # 如果已经足够接近目标楼层，就提前停止。
        if diff < 0.05:
            break

    if best is None:
        print("No navmesh point found")
        return False

    state = agent.get_state()
    state.position = best
    agent.set_state(state)
    print("Teleported to:", best)
    return True


def get_lower_floor(current_height, floor_heights):
    """返回比当前高度低的最近楼层高度。"""
    lower = [h for h in floor_heights if h < current_height - 0.2]
    if not lower:
        return None
    return max(lower)


def get_higher_floor(current_height, floor_heights):
    """返回比当前高度高的最近楼层高度。"""
    higher = [h for h in floor_heights if h > current_height + 0.2]
    if not higher:
        return None
    return min(higher)


def get_objects_for_scene(args) -> None:
    """加载一个场景，并进入键盘交互控制循环。

    参数 args 是一个 tuple：
        (scene_name, outpath, args, device_id)

    交互按键：
        w: 前进 0.25m
        e: 连续前进 4 次，即约 1m
        a: 左转 30 度
        d: 右转 30 度
        s: 连续左转 6 次，即 180 度
        o: 抬头 30 度
        p: 低头 30 度
        k: 保存当前位置三向全景
        n: 切到更高楼层
        m: 切到更低楼层
        q: 退出
    """
    # "00009-vLpv2VX547B"
    scene_name, outpath, args, device_id = args

    # 输出目录加入当前时间，避免覆盖历史 demo 截图。
    outpath = os.path.join(outpath, scene_name, datetime.now().strftime("%m%d%H"))
    os.makedirs(outpath, exist_ok=True)

    # 从类似 "00862-LT9Jq6dN3Ea" 中取出 scene key。
    scene_key = os.path.basename(scene_name.split('-')[-1]).split(".")[0]
    print(" ***** Begin to process scene {}. *****".format(scene_name))

    # 加载 sim，并采样候选楼层高度。
    sim = generate_habitat_sim(device_id, scene_key, args, sensor_height=args.sensor_height, resolution=[720, 960])
    sampled_floor_heights = get_floor_heights_fast(sim)

    # 随机采样一个合法初始位置。
    # 要求：
    # - 不是 NaN；
    # - 在 navmesh 上；
    # - 所在 island 足够大，避免出生在很小的孤立区域。
    flag = True
    while flag:
        start_position = sim.pathfinder.get_random_navigable_point().astype(np.float32)
        if (start_position is None or np.any(np.isnan(start_position)) or not sim.pathfinder.is_navigable(start_position)):
            continue
        if sim.pathfinder.island_radius(start_position) < 1.5:
            continue
        flag = False

    pathfinder = sim.pathfinder
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = start_position
    agent.set_state(agent_state)
    print("Start position: {}".format(start_position))

    trajectory, step_id = [], 0

    # 主交互循环：持续显示当前 RGB 观察和 top-down map，并读取键盘输入。
    while True:
        # 获取当前 observation。
        obs = sim.get_sensor_observations()
        rgb_bgr = cv2.cvtColor(obs['color'][:, :, :3], cv2.COLOR_RGB2BGR)

        # 如果之后需要同时显示 depth，可以启用下面这段。
        # depth_norm = np.nan_to_num(obs['depth'])
        # if depth_norm.max() > depth_norm.min():
        #     depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
        # depth_norm = (depth_norm * 255).astype(np.uint8)
        # depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        # combined = np.hstack([rgb_bgr, depth_bgr])

        # 记录轨迹并生成 top-down 可视化。
        state = agent.get_state()
        trajectory.append(state.position.copy())
        topdown = get_topdown_map_visualize(sim, agent, trajectory)

        # 显示当前 RGB 观察和 top-down map。
        cv2.imshow("Habitat Teleop", rgb_bgr)
        cv2.imshow("Topdown Map", topdown)

        # 捕获键盘输入。
        key = cv2.waitKey(1) & 0xFF

        if key == ord('w'):
            # 前进一步，动作长度由 action_space 里的 amount=0.25 决定。
            sim.step("move_forward")

        elif key == ord('e'):
            # 快速前进：连续走 4 步，约 1m。
            for _ in range(4):
                sim.step("move_forward")

        elif key == ord('a'):
            # 左转 30 度。
            sim.step("turn_left")

        elif key == ord('d'):
            # 右转 30 度。
            sim.step("turn_right")

        elif key == ord('s'):
            # 原地掉头：左转 6 次，每次 30 度，共 180 度。
            for _ in range(6):
                sim.step("turn_left")

        elif key == ord('o'):
            # 相机抬头。
            sim.step("look_up")

        elif key == ord('p'):
            # 相机低头。
            sim.step("look_down")

        elif key == ord('k'):
            # 保存当前位置的三向全景截图。
            capture_panorama(sim, agent, step_id, outpath)

        elif key == ord('n'):
            # 切换到更高楼层。
            state = agent.get_state()
            current_height = state.position[1]
            target = get_higher_floor(current_height, sampled_floor_heights)
            if target is not None:
                teleport_to_floor(sim, agent, target)
            else:
                print("Already at highest floor")

        elif key == ord('m'):
            # 切换到更低楼层。
            state = agent.get_state()
            current_height = state.position[1]
            target = get_lower_floor(current_height, sampled_floor_heights)
            if target is not None:
                teleport_to_floor(sim, agent, target)
            else:
                print("Already at lowest floor")

        elif key == ord('q'):
            # 退出交互。
            break

        step_id += 1

    cv2.destroyAllWindows()

    # 以下代码原本用于分析每个 region 的楼层高度，并导出每个楼层的 top-down map。
    # 当前主流程没有启用，保留为后续分析用。
    # ''' analyze height of each floor '''
    # min_bound, max_bound = sim.pathfinder.get_bounds()
    # min_height, max_height = min_bound[1], max_bound[1]
    # scene = sim.semantic_scene
    #
    # ''' analyze each region '''
    # print("Analyzing region.")
    # total_heights = []
    # for region in scene.regions:
    #     region_key = "region{}".format(region.id)
    #     region_max_xyz, region_min_xyz = None, None
    #     est_region_heights, nav_obj_num = [], 0
    #     for obj in region.objects:
    #         oid = obj.semantic_id
    #         ocl = obj.category.name().lower()
    #         if is_on_ceiling(sim, obj.aabb, ocl):
    #             continue
    #         obj_center = obj.aabb.center
    #         obj_floor_height = get_floor_height(sim, obj_center)
    #         if not np.isnan(obj_floor_height):
    #             est_region_heights.append(obj_floor_height)
    #     est_region_heights = [x for x in est_region_heights if not np.isnan(x)]
    #     est_region_height = most_common_value(est_region_heights)
    #     if est_region_height is not None:
    #         total_heights.append(est_region_height)
    #
    # for height in total_heights:
    #     map_img = get_topdown_map_(sim, height=height, meters_per_pixel=0.025)
    #     os.makedirs(os.path.join(args.outpath, scene_name), exist_ok=True)
    #     cv2.imwrite(os.path.join(args.outpath, scene_name, f'{height:.1f}.png'), map_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sh", "--sensor_height",
        default=1.,
        help="sensor height for observation",
        type=float,
    )

    parser.add_argument(
        "-s", "--scene",
        default='val/00862-LT9Jq6dN3Ea',
        help="scene path under split, e.g., val/00862-LT9Jq6dN3Ea",
        type=str,
    )  # val/00821-eF36g7L6Z9M   train/00155-iLDo95ZbDJq

    parser.add_argument(
        "-o", "--outpath",
        default="/home/bo/Documents/Datasets/CondVLN/hm3d_object_views/demo",
        help="output path for saved panorama images",
        type=str,
    )

    parser.add_argument(
        "--tasks_per_gpu",
        default=1,
        help="number of scenes/tasks per GPU",
        type=int,
    )

    args = parser.parse_args()

    # 当前脚本默认只使用 1 张 GPU。
    args.num_gpus = 1

    # 从 scene 参数中解析 split，例如 val / train。
    args.split = args.scene.split('/')[0]

    # HM3D 本地数据根目录。运行前需要确认该路径存在。
    args.root_path = "/home/bo/Documents/Datasets/HM3D/data/scene_datasets/hm3d"
    assert os.path.exists(args.root_path), "root_path does not exist"

    os.makedirs(args.outpath, exist_ok=True)

    # 只处理一个 scene。
    # args.scene = "val/00862-LT9Jq6dN3Ea" 时，
    # args.scene.split('/')[1] = "00862-LT9Jq6dN3Ea"。
    get_objects_for_scene((args.scene.split('/')[1], args.outpath, args, 0))
