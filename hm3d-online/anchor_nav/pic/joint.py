from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def stitch_panorama(
    color_list: List[np.ndarray],
    *,
    camera_fov_deg: float = 42.0,
    yaw_step_deg: float = 30.0,
    crop_overlap: bool = True,
) -> np.ndarray:
    if len(color_list) == 0:
        raise ValueError("empty color_list")
    imgs = []
    h0, w0 = color_list[0].shape[:2]
    keep_ratio = 1.0
    if crop_overlap and float(camera_fov_deg) > 1e-6:
        keep_ratio = float(yaw_step_deg) / float(camera_fov_deg)
        keep_ratio = float(max(0.2, min(1.0, keep_ratio)))
    for x in color_list:
        arr = np.asarray(x)
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue
        if arr.shape[0] != h0 or arr.shape[1] != w0:
            continue
        rgb = np.asarray(arr[:, :, :3], dtype=np.uint8)
        if keep_ratio < 0.999:
            keep_w = int(round(float(w0) * keep_ratio))
            keep_w = int(max(1, min(w0, keep_w)))
            x0 = int((w0 - keep_w) // 2)
            x1 = int(x0 + keep_w)
            rgb = rgb[:, x0:x1, :]
        imgs.append(rgb)
    if len(imgs) == 0:
        raise ValueError("no valid rgb frames for panorama")
    return np.concatenate(imgs, axis=1)


def _subsample_frames_evenly(color_list: List[np.ndarray], max_frames: int) -> List[np.ndarray]:
    if max_frames <= 0:
        return []
    n = len(color_list)
    if n <= max_frames:
        return list(color_list)
    # Use fixed-step sampling to preserve angular continuity.
    # Example: n=12, max_frames=6 -> [0, 2, 4, 6, 8, 10]
    step = float(n) / float(max_frames)
    idx = []
    used = set()
    for k in range(int(max_frames)):
        i = int(np.floor(k * step))
        i = min(max(i, 0), n - 1)
        if i in used:
            # Fallback: take next unused index in order.
            j = i
            while j < n and j in used:
                j += 1
            if j >= n:
                j = i
                while j >= 0 and j in used:
                    j -= 1
            if j < 0 or j >= n:
                continue
            i = int(j)
        used.add(i)
        idx.append(i)
    idx = sorted(idx)
    return [color_list[int(i)] for i in idx]


def _save_rgb_jpg(rgb: np.ndarray, out_path: Path) -> None:
    import cv2

    bgr = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1]
    cv2.imwrite(str(out_path), bgr)
