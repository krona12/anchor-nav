from typing import List, Optional, Tuple

import cv2
import numpy as np
import quaternion


IMG_WIDTH = 640
IMG_HEIGHT = 360
HFOV_DEG = 42.0


def make_intrinsic_hfov(hfov_deg: float = HFOV_DEG, width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> np.ndarray:
    intrinsic = np.eye(4)
    hfov = np.radians(hfov_deg)
    aspect = width / height
    intrinsic[0, 0] = 1.0 / np.tan(hfov / 2.0)
    intrinsic[1, 1] = 1.0 / np.tan(hfov / 2.0) / aspect
    return intrinsic


def _sensor_cam_to_world(agent_state) -> np.ndarray:
    sensor_state = agent_state.sensor_states["color_sensor"]
    rot = quaternion.as_rotation_matrix(sensor_state.rotation)
    pos = sensor_state.position
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform


def project_world_to_pixel(
    world_xyz: np.ndarray,
    agent_state,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
) -> Optional[Tuple[int, int]]:
    intrinsic = make_intrinsic_hfov(HFOV_DEG, width, height)
    world_to_cam = np.linalg.inv(_sensor_cam_to_world(agent_state))
    point_world = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
    point_cam = world_to_cam @ point_world

    # Habitat camera looks toward -Z.
    depth = -point_cam[2]
    if depth <= 1e-2:
        return None

    u_ndc = (point_cam[0] / depth) * intrinsic[0, 0]
    v_ndc = (point_cam[1] / depth) * intrinsic[1, 1]
    px = int((u_ndc + 1.0) * 0.5 * width)
    py = int((1.0 - v_ndc) * 0.5 * height)
    if px < 0 or px >= width or py < 0 or py >= height:
        return None
    return (px, py)


def draw_som_circles(
    image_rgb: np.ndarray,
    points_2d: List[Optional[Tuple[int, int]]],
    labels: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = image_rgb.copy()
    for point, label in zip(points_2d, labels):
        if point is None:
            continue
        px, py = point
        cv2.circle(out, (px, py), 14, color, 2)
        cv2.putText(out, str(label), (px - 5, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out

