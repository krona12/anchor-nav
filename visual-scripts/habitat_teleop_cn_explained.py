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
import gzip
import json
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import numpy as np
import cv2
import time
from datetime import datetime
import imageio.v2 as imageio

try:
    import GPUtil
except Exception:
    class GPUtil:
        @staticmethod
        def getAvailable(*_args, **_kwargs):
            return []

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
try:
    from low_level_utils import is_on_ceiling, most_common_value
except Exception:
    # 只有旧版楼层分析注释代码会用到这两个函数；object-fetch 自动拍摄
    # 不依赖它们，所以本地没有 low_level_utils 时仍然允许运行。
    def is_on_ceiling(*_args, **_kwargs):
        return False

    def most_common_value(values):
        return Counter(values).most_common(1)[0][0] if values else None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_SCRIPTS_DIR = PROJECT_ROOT / "visual-scripts"


def _jsonable(value: Any) -> Any:
    """把 numpy / Path 等对象转换成 json.dump 可以处理的类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _parse_vec3(text: str) -> Optional[np.ndarray]:
    text = str(text or "").strip()
    if not text:
        return None
    vals = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {text}")
    return np.asarray(vals, dtype=float).reshape(3)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.asarray(rgb, dtype=np.uint8), compress_level=0)


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """把深度图归一化成便于查看的灰度 RGB 图。"""
    dep = np.asarray(depth, dtype=np.float32)
    dep = np.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
    valid = dep[dep > 0.0]
    if valid.size == 0:
        return np.zeros((*dep.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((dep - lo) / (hi - lo), 0.0, 1.0)
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    """根据 scene id 找到本地 glb 文件。

    支持两种常见目录：
    - datascene/00844-q5QZSEeHe5g/q5QZSEeHe5g.glb
    - HM3D 原始数据目录中带 *.basis.glb 的结构
    """
    raw = Path(os.path.expanduser(scene_name))
    if raw.exists() and raw.suffix == ".glb":
        return str(raw)

    root = Path(os.path.expanduser(hm3d_root))
    scene_leaf = scene_name.split("/")[-1]
    short_scene_name = scene_leaf.split("-")[-1]
    scene_dirs: List[Path] = [root / scene_leaf]

    if "-" not in scene_leaf:
        scene_dirs.extend(sorted(root.glob(f"*-{short_scene_name}")))
    if not scene_dirs[0].exists():
        scene_dirs.extend(sorted(root.rglob(scene_leaf)))

    checked: List[str] = []
    for scene_dir in scene_dirs:
        candidates = [
            scene_dir / f"{short_scene_name}.basis.glb",
            scene_dir / f"{short_scene_name}.glb",
        ]
        candidates.extend(sorted(scene_dir.glob("*.basis.glb")))
        candidates.extend(sorted(scene_dir.glob("*.glb")))
        for candidate in candidates:
            checked.append(str(candidate))
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(f"Scene asset not found for {scene_name}; checked={checked[:20]}")


def _find_scene_data_file(root: Path, scene_name: str) -> Path:
    direct = root / f"{scene_name}.json.gz"
    if direct.exists():
        return direct
    found = sorted(root.rglob(f"{scene_name}.json.gz"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Scene annotation not found for {scene_name} under {root}")


def load_object_fetch_task(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """读取指定 episode / instance 的目标物体记录。"""
    scene_name = args.scene_name or args.scene.split("/")[-1]
    nav_root = Path(os.path.expanduser(args.navigation_data_path))
    scene_gz = _find_scene_data_file(nav_root, scene_name)
    with gzip.open(scene_gz, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)

    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    if args.navigation_type not in episode_mapping:
        raise ValueError(f"object-fetch only supports object/room/region/instance, got {args.navigation_type}")

    target_ep = None
    for ep in episode_mapping[args.navigation_type]:
        if int(ep["episode_id"]) != int(args.episode_id):
            continue
        if args.instance_id.strip() and ep.get("instance_id") != args.instance_id.strip():
            continue
        target_ep = ep
        break
    if target_ep is None:
        guard = f" instance_id={args.instance_id}" if args.instance_id.strip() else ""
        raise ValueError(f"{args.navigation_type} episode {args.episode_id}{guard} not found in {scene_gz}")

    goals_map = {g["object_id"]: g for g in scene_data["goals"]}
    goal_ids = list(target_ep.get("target_object_ids", []))
    if not goal_ids:
        raise ValueError(f"Episode {args.episode_id} has no target_object_ids")
    goal_id = args.instance_id.strip() or str(goal_ids[0])
    if goal_id not in goals_map:
        raise KeyError(f"Goal object {goal_id} not found in scene goals")
    return target_ep, goals_map[goal_id], str(scene_gz)


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
    root_path = os.path.expanduser(getattr(args, "root_path", ""))
    split = getattr(args, "split", "")
    num_gpus = max(1, int(getattr(args, "num_gpus", 1)))
    tasks_per_gpu = max(1, int(getattr(args, "tasks_per_gpu", 1)))

    # 有语义 mask 的配置。
    SCENE_CFG = os.path.join(root_path, "hm3d_annotated_basis.scene_dataset_config.json")
    # 无语义 mask 的配置。当前代码会覆盖上一行，因此实际使用的是这个配置。
    SCENE_CFG = os.path.join(root_path, "hm3d_basis.scene_dataset_config.json")
    scene_dataset_config = SCENE_CFG if root_path and os.path.exists(SCENE_CFG) else ""

    # 新增：如果当前仓库只有 datascene/<scene>/<scene>.glb，没有 HM3D
    # scene_dataset_config.json，就直接把 scene_id 指向 glb 文件。
    scene_id = scene
    if os.path.exists(os.path.expanduser(str(scene))):
        scene_id = os.path.expanduser(str(scene))
    elif not scene_dataset_config:
        hm3d_base = os.path.expanduser(getattr(args, "hm3d_data_base_path", root_path))
        scene_id = resolve_scene_path(hm3d_base, scene)

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
    if hasattr(sim_cfg, "enable_hbao"):
        sim_cfg.enable_hbao = True
    if hasattr(sim_cfg, "override_scene_light_defaults"):
        sim_cfg.override_scene_light_defaults = True
    if hasattr(sim_cfg, "scene_light_setup"):
        sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY
    sim_cfg.create_renderer = True
    if scene_dataset_config:
        sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = scene_id

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


def _rotation_xyzw(rotation: Any) -> Any:
    try:
        return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
    except Exception:
        try:
            vals = list(rotation)
            return [float(v) for v in vals]
        except Exception:
            return str(rotation)


def _look_at_quat_xz(origin: np.ndarray, target: np.ndarray, yaw_offset_deg: float = 0.0) -> List[float]:
    """让 agent 在 XZ 平面上面向 target。

    Habitat 相机默认朝向 local -Z，因此这里的 yaw 公式与普通 +Z 前向
    坐标系略有不同。
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    dx = float(target[0] - origin[0])
    dz = float(target[2] - origin[2])
    if math.hypot(dx, dz) < 1e-6:
        yaw = 0.0
    else:
        yaw = math.atan2(-dx, -dz)
    yaw += math.radians(float(yaw_offset_deg))
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _snap_point(pathfinder: Any, point: Sequence[float], island_index: Optional[int] = None) -> np.ndarray:
    raw = np.asarray(point, dtype=float).reshape(3)
    try:
        if island_index is None:
            snapped = pathfinder.snap_point(point=raw)
        else:
            snapped = pathfinder.snap_point(point=raw, island_index=int(island_index))
    except TypeError:
        snapped = pathfinder.snap_point(raw)
    snapped = np.asarray(snapped, dtype=float).reshape(3)
    if np.any(np.isnan(snapped)):
        raise ValueError(f"snap_point returned NaN for {raw.tolist()}")
    return snapped


def _unit_xz(vec: np.ndarray) -> Optional[np.ndarray]:
    out = np.asarray([float(vec[0]), 0.0, float(vec[2])], dtype=float)
    norm = float(np.linalg.norm(out[[0, 2]]))
    if norm < 1e-6:
        return None
    return out / norm


def _set_agent_pose(agent: Any, position: Sequence[float], rotation: Any) -> None:
    state = AgentState()
    state.position = np.asarray(position, dtype=float).reshape(3)
    state.rotation = rotation
    agent.set_state(state)


def _render_current_rgb_depth(sim: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["color"][:, :, :3], dtype=np.uint8).copy()
    depth = None
    if "depth" in obs:
        depth = np.asarray(obs["depth"], dtype=np.float32).copy()
    return rgb, depth


def render_pose_rgb_depth(
    sim: Any,
    agent: Any,
    position: Sequence[float],
    target: Sequence[float],
    *,
    yaw_offset_deg: float = 0.0,
    look_down_steps: int = 0,
    fixed_rotation: Any = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """设置位姿并渲染一张照片。look_down 用动作实现，渲染后撤销。"""
    rotation = fixed_rotation if fixed_rotation is not None else _look_at_quat_xz(
        np.asarray(position, dtype=float), np.asarray(target, dtype=float), yaw_offset_deg
    )
    _set_agent_pose(agent, position, rotation)

    look_down_steps = max(0, int(look_down_steps))
    for _ in range(look_down_steps):
        sim.step("look_down")
    rgb, depth = _render_current_rgb_depth(sim)
    for _ in range(look_down_steps):
        sim.step("look_up")

    state = agent.get_state()
    pose = {
        "position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
        "rotation_xyzw": _rotation_xyzw(state.rotation),
        "yaw_offset_deg": float(yaw_offset_deg),
        "look_down_steps": int(look_down_steps),
    }
    return rgb, depth, pose


def plan_object_fetch_pose(
    sim: Any,
    target_ep: Dict[str, Any],
    goal: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """为“到目标点，然后后退几格拍照”选择最终可导航拍摄点。"""
    pathfinder = sim.pathfinder
    start = np.asarray(target_ep["start_position"], dtype=float).reshape(3)
    target_center = np.asarray(goal["position"], dtype=float).reshape(3)
    step_size = float(args.object_fetch_step_size)
    back_steps = int(args.object_fetch_back_steps)
    back_distance = max(0.1, float(back_steps) * step_size)

    try:
        start_snap = _snap_point(pathfinder, start)
    except Exception:
        start_snap = start
    try:
        island_index = int(pathfinder.get_island(start_snap))
    except Exception:
        island_index = None

    try:
        target_snap = _snap_point(pathfinder, target_center, island_index)
    except Exception:
        target_snap = _snap_point(pathfinder, target_center)

    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[float, float, float]] = set()

    def add_candidate(
        *,
        source: str,
        direction_seed: np.ndarray,
        iou: float = 0.0,
        raw_view_position: Optional[np.ndarray] = None,
        view_index: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> None:
        direction = _unit_xz(direction_seed - target_snap)
        if direction is None:
            return
        desired = target_snap + direction * back_distance
        desired[1] = target_snap[1]
        try:
            snapped = _snap_point(pathfinder, desired, island_index)
        except Exception:
            try:
                snapped = _snap_point(pathfinder, desired)
            except Exception:
                return
        key = tuple(np.round(snapped, 3).tolist())
        if key in seen:
            return
        seen.add(key)
        snap_error = float(np.linalg.norm((snapped - desired)[[0, 2]]))
        dist_to_target = float(np.linalg.norm((snapped - target_center)[[0, 2]]))
        dist_to_target_snap = float(np.linalg.norm((snapped - target_snap)[[0, 2]]))
        score = float(iou) * 10.0 - abs(dist_to_target_snap - back_distance) - 0.25 * snap_error
        if source.startswith("annotation"):
            score += 0.25
        candidates.append(
            {
                "source": source,
                "view_index": view_index,
                "iou": float(iou),
                "radius": None if radius is None else float(radius),
                "raw_view_position": None if raw_view_position is None else raw_view_position.tolist(),
                "desired_backoff_position": desired.tolist(),
                "snapped_position": snapped.tolist(),
                "snap_error_m": snap_error,
                "distance_to_target_center_m": dist_to_target,
                "distance_to_target_snap_m": dist_to_target_snap,
                "score": score,
            }
        )

    view_points = list(goal.get("view_points", []))
    for idx, vp in enumerate(view_points):
        agent_state = vp.get("agent_state", {}) if isinstance(vp, dict) else {}
        raw_pos = np.asarray(agent_state.get("position", []), dtype=float)
        if raw_pos.size != 3:
            continue
        add_candidate(
            source="annotation_viewpoint",
            view_index=idx,
            direction_seed=raw_pos.reshape(3),
            raw_view_position=raw_pos.reshape(3),
            iou=float(vp.get("iou", 0.0)),
        )

    # 如果 annotation 视点不可用，就围绕目标点生成一圈候选拍摄点。
    for radius in (back_distance, 0.5, 0.75, 1.0, 1.25, 1.5):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            seed = target_snap + np.array([radius * math.cos(float(angle)), 0.0, radius * math.sin(float(angle))])
            add_candidate(source="ring_fallback", direction_seed=seed, radius=float(radius))

    if not candidates:
        raise RuntimeError("No valid object-fetch camera candidates found around target")

    candidates.sort(key=lambda x: float(x["score"]), reverse=True)
    best = candidates[0]
    final_position = np.asarray(best["snapped_position"], dtype=float).reshape(3)

    camera_y = float(final_position[1]) + float(args.sensor_height)
    planar = float(np.linalg.norm((target_center - final_position)[[0, 2]]))
    depression_deg = math.degrees(math.atan2(max(camera_y - float(target_center[1]), 0.0), max(planar, 1e-3)))
    suggested_look_down_steps = max(0, min(2, int(round(depression_deg / 30.0))))

    best_annotation_pose = None
    if view_points:
        vp0 = view_points[0].get("agent_state", {})
        if len(vp0.get("position", [])) == 3:
            raw_pos = np.asarray(vp0["position"], dtype=float).reshape(3)
            try:
                ann_pos = _snap_point(pathfinder, raw_pos, island_index)
            except Exception:
                ann_pos = raw_pos
            best_annotation_pose = {
                "position": ann_pos.tolist(),
                "rotation": vp0.get("rotation"),
                "iou": float(view_points[0].get("iou", 0.0)),
            }

    return {
        "start_position": start.tolist(),
        "start_snap_position": start_snap.tolist(),
        "target_center": target_center.tolist(),
        "target_snap_position": target_snap.tolist(),
        "island_index": island_index,
        "back_steps": back_steps,
        "step_size_m": step_size,
        "back_distance_m": back_distance,
        "selected": best,
        "final_position": final_position.tolist(),
        "depression_deg": float(depression_deg),
        "suggested_look_down_steps": int(suggested_look_down_steps),
        "best_annotation_pose": best_annotation_pose,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:10],
    }


def resolve_focus_pose(
    *,
    sim: Any,
    scene_name: str,
    goal_id: str,
    goal: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """解析用户指定/本任务默认的重点拍摄位姿。

    对 armchair_906，本函数默认使用用户在截图中画红圈的站位，并让相机
    朝向带针织毯子和红枕的可见椅面中心，而不是只朝 annotation bbox 中心。
    """
    position = _parse_vec3(getattr(args, "object_fetch_focus_position", ""))
    aim = _parse_vec3(getattr(args, "object_fetch_focus_aim", ""))
    source = "cli"

    if position is None and scene_name == "00844-q5QZSEeHe5g" and goal_id == "armchair_906":
        position = np.asarray([1.7048323154449463, 0.15387701988220215, 2.420600175857544], dtype=float)
        source = "user_red_circle_default"
    if aim is None and scene_name == "00844-q5QZSEeHe5g" and goal_id == "armchair_906":
        # goal['position'] is close to the object center, but visually the
        # blanket/chair face is slightly toward +X and -Z from that point.
        aim = np.asarray(goal["position"], dtype=float).reshape(3) + np.asarray([0.45, 0.0, -0.05], dtype=float)

    if position is None:
        return None
    if aim is None:
        aim = np.asarray(goal["position"], dtype=float).reshape(3)

    try:
        snapped = _snap_point(sim.pathfinder, position)
    except Exception:
        snapped = position

    return {
        "source": source,
        "position": snapped.tolist(),
        "raw_position": position.tolist(),
        "aim": aim.tolist(),
        "look_down_steps": int(getattr(args, "object_fetch_focus_look_down_steps", 1)),
    }


def save_object_fetch_topdown(
    sim: Any,
    out_path: Path,
    *,
    start: Sequence[float],
    target_center: Sequence[float],
    target_snap: Sequence[float],
    final_position: Sequence[float],
    meters_per_pixel: float = 0.025,
) -> None:
    """保存一张 top-down 图，标出 start / object / target snap / photo pose。"""
    start = np.asarray(start, dtype=float).reshape(3)
    target_center = np.asarray(target_center, dtype=float).reshape(3)
    target_snap = np.asarray(target_snap, dtype=float).reshape(3)
    final_position = np.asarray(final_position, dtype=float).reshape(3)

    height = float(target_snap[1])
    topdown_bool = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    h, w = topdown_bool.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[topdown_bool] = (210, 210, 210)

    bounds = sim.pathfinder.get_bounds()
    min_bound = np.asarray(bounds[0], dtype=float)

    def world_to_px(pos: np.ndarray) -> Tuple[int, int]:
        col = int((float(pos[0]) - float(min_bound[0])) / meters_per_pixel)
        row = int((float(pos[2]) - float(min_bound[2])) / meters_per_pixel)
        col = max(0, min(col, w - 1))
        row = max(0, min(row, h - 1))
        return col, row

    start_px = world_to_px(start)
    target_px = world_to_px(target_center)
    target_snap_px = world_to_px(target_snap)
    final_px = world_to_px(final_position)

    cv2.line(rgb, start_px, target_snap_px, (80, 130, 255), 2)
    cv2.line(rgb, target_snap_px, final_px, (30, 220, 120), 2)
    cv2.circle(rgb, start_px, 7, (40, 120, 255), -1)
    cv2.circle(rgb, target_snap_px, 7, (255, 40, 180), -1)
    cv2.drawMarker(rgb, target_px, (255, 170, 0), cv2.MARKER_STAR, 20, 2)
    cv2.drawMarker(rgb, final_px, (0, 255, 120), cv2.MARKER_TRIANGLE_UP, 18, 2)
    cv2.putText(rgb, "start", (start_px[0] + 8, start_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 120, 255), 1)
    cv2.putText(rgb, "object", (target_px[0] + 8, target_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 0), 1)
    cv2.putText(rgb, "photo", (final_px[0] + 8, final_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
    _save_rgb(out_path, rgb)


def save_contact_sheet(path: Path, records: List[Dict[str, Any]], thumb_w: int = 320) -> None:
    images = []
    for rec in records:
        rgb = imageio.imread(rec["rgb_path"])
        scale = float(thumb_w) / float(rgb.shape[1])
        thumb_h = max(1, int(round(rgb.shape[0] * scale)))
        thumb = cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        label = str(rec["name"])
        cv2.rectangle(thumb, (0, 0), (thumb_w, 28), (0, 0, 0), -1)
        cv2.putText(thumb, label[:42], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        images.append(thumb)
    if not images:
        return
    cols = min(3, len(images))
    rows = int(math.ceil(len(images) / cols))
    h = max(img.shape[0] for img in images)
    sheet = np.zeros((rows * h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * h:r * h + img.shape[0], c * thumb_w:c * thumb_w + thumb_w] = img
    _save_rgb(path, sheet)


def run_object_fetch_task(args: argparse.Namespace) -> Path:
    """执行本次具体任务：episode 122 / armchair_906 的目标点拍摄。"""
    scene_name = args.scene_name or args.scene.split("/")[-1]
    target_ep, goal, annotation_path = load_object_fetch_task(args)
    scene_path = resolve_scene_path(args.hm3d_data_base_path, scene_name)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    goal_id = str(goal["object_id"])
    out_dir = Path(os.path.expanduser(args.object_fetch_outdir)) / f"run={run_id}_ep{int(args.episode_id)}_{goal_id}"
    photos_dir = out_dir / "photos"
    maps_dir = out_dir / "maps"
    photos_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[object-fetch] scene_path={scene_path}", flush=True)
    print(f"[object-fetch] episode={args.episode_id} instance={goal_id}", flush=True)
    print(f"[object-fetch] output={out_dir}", flush=True)

    sim = generate_habitat_sim(
        0,
        scene_path,
        args,
        sensor_height=float(args.sensor_height),
        resolution=[int(args.fetch_rgb_height), int(args.fetch_rgb_width)],
    )
    agent = sim.get_agent(0)

    try:
        _set_agent_pose(agent, target_ep["start_position"], target_ep["start_rotation"])
        plan = plan_object_fetch_pose(sim, target_ep, goal, args)
        target_center = np.asarray(plan["target_center"], dtype=float).reshape(3)
        final_position = np.asarray(plan["final_position"], dtype=float).reshape(3)
        focus_pose = resolve_focus_pose(
            sim=sim,
            scene_name=scene_name,
            goal_id=goal_id,
            goal=goal,
            args=args,
        )

        saved_records: List[Dict[str, Any]] = []

        def save_variant(name: str, position: Sequence[float], *, yaw_offset_deg: float = 0.0,
                         look_down_steps: int = 0, fixed_rotation: Any = None) -> None:
            rgb, depth, pose = render_pose_rgb_depth(
                sim,
                agent,
                position,
                target_center,
                yaw_offset_deg=yaw_offset_deg,
                look_down_steps=look_down_steps,
                fixed_rotation=fixed_rotation,
            )
            rgb_path = photos_dir / f"{name}_rgb.png"
            _save_rgb(rgb_path, rgb)
            depth_path = None
            if depth is not None:
                depth_path = photos_dir / f"{name}_depth.png"
                _save_rgb(depth_path, _depth_to_rgb(depth))
            saved_records.append(
                {
                    "name": name,
                    "rgb_path": str(rgb_path),
                    "depth_path": None if depth_path is None else str(depth_path),
                    "pose": pose,
                }
            )
            print(f"[object-fetch] saved {rgb_path}", flush=True)

        # 记录 annotation 中 IoU 最高的原始推荐视点，便于和 backoff 逻辑对照。
        ann_pose = plan.get("best_annotation_pose")
        if ann_pose and ann_pose.get("rotation") is not None:
            save_variant(
                "00_annotation_best_view",
                ann_pose["position"],
                fixed_rotation=ann_pose["rotation"],
            )

        # 到目标点附近后，从目标点沿最佳视线方向退后几格，正对目标拍摄。
        save_variant("01_backoff_facing_target", final_position)

        suggested_look_down_steps = int(plan["suggested_look_down_steps"])
        look_down_steps = max(0, min(int(args.object_fetch_max_look_down_steps), suggested_look_down_steps))
        if look_down_steps > 0:
            save_variant("02_backoff_facing_target_soft_tilt", final_position, look_down_steps=look_down_steps)

        # 微调角度，给后续人工挑选照片留出余量。默认 10 度比 15 度更平稳，
        # 仍然能让 05 号图保留目标椅子的倾斜观察角。
        focus_yaw = float(args.object_fetch_focus_yaw_deg)
        focus_yaw_tag = str(int(abs(focus_yaw))) if abs(focus_yaw - int(focus_yaw)) < 1e-6 else f"{abs(focus_yaw):.1f}".replace(".", "p")
        save_variant(f"03_backoff_yaw_left_{focus_yaw_tag}", final_position, yaw_offset_deg=focus_yaw)
        save_variant(f"04_backoff_yaw_right_{focus_yaw_tag}", final_position, yaw_offset_deg=-focus_yaw)
        if focus_pose is not None:
            focus_position = np.asarray(focus_pose["position"], dtype=float).reshape(3)
            focus_aim = np.asarray(focus_pose["aim"], dtype=float).reshape(3)
            focus_look_down_steps = max(0, int(focus_pose["look_down_steps"]))
            save_variant(
                "05_redcircle_front_facing_soft_tilt",
                focus_position,
                look_down_steps=focus_look_down_steps,
                fixed_rotation=_look_at_quat_xz(focus_position, focus_aim),
            )
            save_variant(
                "06_redcircle_front_facing_level",
                focus_position,
                fixed_rotation=_look_at_quat_xz(focus_position, focus_aim),
            )
        elif look_down_steps > 0:
            save_variant(f"05_backoff_yaw_left_{focus_yaw_tag}_soft_tilt", final_position, yaw_offset_deg=focus_yaw, look_down_steps=look_down_steps)
            save_variant(f"06_backoff_yaw_right_{focus_yaw_tag}_soft_tilt", final_position, yaw_offset_deg=-focus_yaw, look_down_steps=look_down_steps)

        save_object_fetch_topdown(
            sim,
            maps_dir / "object_fetch_topdown.png",
            start=plan["start_position"],
            target_center=plan["target_center"],
            target_snap=plan["target_snap_position"],
            final_position=plan["final_position"],
        )
        save_contact_sheet(photos_dir / "object_fetch_contact_sheet.png", saved_records)

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": {
                "scene_name": scene_name,
                "scene_path": scene_path,
                "annotation_path": annotation_path,
                "navigation_type": args.navigation_type,
                "episode_id": int(args.episode_id),
                "instance_id": goal_id,
                "target_object_ids": target_ep.get("target_object_ids", []),
                "object_category": goal.get("object_category"),
                "description": goal.get("annot_unique_detailed_description")
                    or goal.get("annot_unique_concise_description")
                    or goal.get("annot_appearance_description"),
            },
            "logic": {
                "intent": "move to the target object's navigable point, back off several grid steps, face the object, and save micro-adjusted photos",
                "back_steps": int(args.object_fetch_back_steps),
                "step_size_m": float(args.object_fetch_step_size),
                "back_distance_m": float(plan["back_distance_m"]),
                "angle_micro_adjustments_deg": [0, float(focus_yaw), -float(focus_yaw)],
                "suggested_look_down_steps": int(suggested_look_down_steps),
                "max_look_down_steps": int(args.object_fetch_max_look_down_steps),
                "look_down_steps": int(look_down_steps),
            },
            "plan": plan,
            "focus_pose": focus_pose,
            "saved_photos": saved_records,
            "outputs": {
                "photos_dir": str(photos_dir),
                "contact_sheet": str(photos_dir / "object_fetch_contact_sheet.png"),
                "topdown": str(maps_dir / "object_fetch_topdown.png"),
                "summary": str(out_dir / "object_fetch_summary.json"),
            },
        }
        _write_json(out_dir / "object_fetch_summary.json", summary)
    finally:
        sim.close()

    print(f"[object-fetch] DONE. photos={photos_dir}", flush=True)
    return out_dir


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
        default=1.31,
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
        default=str(VISUAL_SCRIPTS_DIR / "object-fetch" / "teleop-demo"),
        help="output path for saved panorama images",
        type=str,
    )

    parser.add_argument(
        "--root_path",
        default=str(PROJECT_ROOT / "datascene"),
        help="HM3D scene root or local datascene root",
        type=str,
    )

    parser.add_argument(
        "--hm3d_data_base_path",
        default=str(PROJECT_ROOT / "datascene"),
        help="local scene asset root, e.g. datascene/",
        type=str,
    )

    parser.add_argument(
        "--navigation_data_path",
        default=str(PROJECT_ROOT / "LangMap_Annotations"),
        help="RefHM3D / LangMap annotation root",
        type=str,
    )

    parser.add_argument(
        "--scene_name",
        default="00844-q5QZSEeHe5g",
        help="scene id for object-fetch mode",
        type=str,
    )

    parser.add_argument(
        "--navigation_type",
        default="instance",
        choices=["object", "room", "region", "instance"],
        help="navigation task level for object-fetch mode",
        type=str,
    )

    parser.add_argument(
        "--episode_id",
        default=122,
        help="episode id for object-fetch mode",
        type=int,
    )

    parser.add_argument(
        "--instance_id",
        default="armchair_906",
        help="target instance id for object-fetch mode",
        type=str,
    )

    parser.add_argument(
        "--object_fetch",
        action="store_true",
        help="run automatic target-point/backoff photo capture instead of keyboard teleop",
    )

    parser.add_argument(
        "--object_fetch_outdir",
        default=str(VISUAL_SCRIPTS_DIR / "object-fetch"),
        help="output log root for object-fetch photos and metadata",
        type=str,
    )

    parser.add_argument(
        "--object_fetch_back_steps",
        default=5,
        help="number of 0.25m-style grid steps to back away from the target before shooting",
        type=int,
    )

    parser.add_argument(
        "--object_fetch_max_look_down_steps",
        default=1,
        help="cap look-down actions for a softer robot-like tilted view",
        type=int,
    )

    parser.add_argument(
        "--object_fetch_focus_yaw_deg",
        default=10.0,
        help="small yaw offset used by the 03/04 micro-adjusted comparison views",
        type=float,
    )

    parser.add_argument(
        "--object_fetch_focus_position",
        default="",
        help="optional manual focus camera position as x,y,z; armchair_906 defaults to the user red-circle floor point",
        type=str,
    )

    parser.add_argument(
        "--object_fetch_focus_aim",
        default="",
        help="optional manual focus aim point as x,y,z; armchair_906 defaults to the visible knitted-blanket chair center",
        type=str,
    )

    parser.add_argument(
        "--object_fetch_focus_look_down_steps",
        default=1,
        help="look-down actions for the manual focus 05 image",
        type=int,
    )

    parser.add_argument(
        "--object_fetch_step_size",
        default=0.25,
        help="meters per backoff step",
        type=float,
    )

    parser.add_argument(
        "--fetch_rgb_width",
        default=960,
        help="object-fetch output RGB width",
        type=int,
    )

    parser.add_argument(
        "--fetch_rgb_height",
        default=720,
        help="object-fetch output RGB height",
        type=int,
    )

    parser.add_argument(
        "--num_gpus",
        default=1,
        help="number of GPUs available for simulator creation",
        type=int,
    )

    parser.add_argument(
        "--tasks_per_gpu",
        default=1,
        help="number of scenes/tasks per GPU",
        type=int,
    )

    args = parser.parse_args()

    # 当前脚本默认只使用 1 张 GPU。
    args.num_gpus = max(1, int(args.num_gpus))

    # 从 scene 参数中解析 split，例如 val / train。
    args.split = args.scene.split('/')[0] if "/" in args.scene else ""

    args.root_path = os.path.expanduser(args.root_path)
    args.hm3d_data_base_path = os.path.expanduser(args.hm3d_data_base_path)
    args.navigation_data_path = os.path.expanduser(args.navigation_data_path)
    args.outpath = os.path.expanduser(args.outpath)
    args.object_fetch_outdir = os.path.expanduser(args.object_fetch_outdir)
    assert os.path.exists(args.root_path), "root_path does not exist"

    if args.object_fetch:
        run_object_fetch_task(args)
        sys.exit(0)

    os.makedirs(args.outpath, exist_ok=True)

    # 只处理一个 scene。
    # args.scene = "val/00862-LT9Jq6dN3Ea" 时，
    # args.scene.split('/')[1] = "00862-LT9Jq6dN3Ea"。
    get_objects_for_scene((args.scene.split('/')[-1], args.outpath, args, 0))
