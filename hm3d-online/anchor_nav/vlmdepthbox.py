from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import quaternion

from data_utils import convert_from_uvd, make_intrinsic_hfov
from vlm.client import DEFAULT_MODEL, chat_messages


@dataclass(frozen=True)
class VLMDepthBoxConfig:
    top_k: int = 5
    vlm_model: str = DEFAULT_MODEL
    vlm_timeout: int = 60
    bbox_inner_fraction: float = 0.0
    target_mode: str = "xz_baseline_y"


@dataclass(frozen=True)
class ObjectObservation:
    slot_index: int
    source_frame_index: int
    rgb: np.ndarray
    depth: np.ndarray
    pose_mat: np.ndarray
    agent_position_xyz: Tuple[float, float, float]
    sensor_position_xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class DepthBoxCandidate:
    rank: int
    slot_index: int
    og3d_logit: float
    merged_object_score: float
    center_habitat_xyz: Tuple[float, float, float]
    observation: ObjectObservation


SELECT_BOX_SYSTEM_PROMPT = (
    "You are a precise object localization assistant for embodied navigation. "
    "Return strict JSON only. Numeric bounding boxes must use pixel coordinates."
)


SELECT_BOX_USER_TEMPLATE = """Navigation task:
{description}

ROLE:
You choose the candidate image that best corresponds to the target instance, then draw one tight box
around the physical target region that should be localized with the paired depth map.

INPUT:
There are exactly {k} RGB images in detector logit order.
- Image 1 is the detector baseline top-1 object.
- Images 2..{k} are lower-ranked object candidates.
- Each image has a same-frame depth map and camera pose in the program.
- The images shown to you include a coordinate grid overlay.
- Image size is width={width}, height={height}. Pixel origin is the top-left corner.
- If you choose Image 1, the program will keep the detector baseline object position.
- If you choose Image 2..{k}, the program will navigate autonomously to your bbox depth projection.

TASK:
1. Select the image that best matches the exact task target or its most diagnostic anchor region.
2. For that selected image, output one tight bbox_xyxy around the target region to be projected by depth.

ATTENTION:
1. Do not box the whole room, wall, floor, or a large context area.
2. Do not prefer an image only because it is clearer, more centered, or shows a complete distant view.
3. Prefer a box on the actual target surface/object/anchor described by the task, especially a region that
   can be navigated to. For counters, tables, shelves, cabinets, beds, sofas, and other support surfaces,
   box the described target surface region near the discriminative objects/anchors.
4. If the target is partly occluded but its location is clearer and closer, a partial box is valid.
5. The bbox must stay inside the selected image and tightly cover the visible target region.
6. Read the grid labels before writing bbox_xyxy. Do not output a generic placeholder box.
7. The bbox center must lie on the named target region, not on ceiling, plain wall, unrelated appliance,
   empty floor, or unrelated furniture unless that is the target.
8. If the selected image shows the target only in a small/partial area, use a small tight box on that area.
9. For a counter with bananas, the box must cover the visible counter/bananas pixels, not the nearby cabinets,
   doorway, fridge, ceiling, or the broad kitchen context.

Return strict JSON with exact keys:
{{
  "best_index": <integer 1..{k}>,
  "bbox_xyxy": [<x1>, <y1>, <x2>, <y2>],
  "target_region": "<short phrase naming what is inside the box>",
  "confidence": "high|medium|low",
  "visible_constraints": ["<visible task constraint>", "..."],
  "missing_or_uncertain_constraints": ["<missing or uncertain constraint>", "..."],
  "reason": "<brief English reason>"
}}
"""


def parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"VLM response is not a JSON object: {raw!r}")
    return parsed


def require_int_in_range(parsed: Mapping[str, Any], key: str, lo: int, hi: int) -> int:
    value = parsed[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < lo or value > hi:
        raise RuntimeError(f"unexpected {key}={value!r}, expected integer {lo}..{hi}")
    return int(value)


def require_choice(parsed: Mapping[str, Any], key: str, allowed: Sequence[str]) -> str:
    value = parsed[key]
    if not isinstance(value, str) or value not in set(allowed):
        raise RuntimeError(f"unexpected {key}={value!r}, expected one of {sorted(allowed)!r}")
    return value


def require_string_list(parsed: Mapping[str, Any], key: str) -> List[str]:
    value = parsed[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise RuntimeError(f"unexpected {key}={value!r}, expected list of strings")
    return [str(x) for x in value]


def require_bbox(parsed: Mapping[str, Any], key: str, width: int, height: int) -> Tuple[int, int, int, int]:
    value = parsed[key]
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError(f"unexpected {key}={value!r}, expected [x1,y1,x2,y2]")
    vals: List[int] = []
    for x in value:
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise RuntimeError(f"unexpected bbox value {x!r}, expected number")
        vals.append(int(round(float(x))))
    x1, y1, x2, y2 = vals
    if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height or x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"bbox {vals!r} outside image bounds width={width}, height={height} or has non-positive area"
        )
    return x1, y1, x2, y2


def save_rgb_jpg(rgb: np.ndarray, path: Path, *, quality: int = 92) -> None:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB image HxWx3, got shape={arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def save_depth_vis(depth: np.ndarray, path: Path) -> None:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected depth HxW, got shape={arr.shape}")
    valid = arr[arr > 0]
    if valid.size == 0:
        raise RuntimeError("cannot visualize depth: no positive pixels")
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    vis = (norm * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def rgb_to_jpeg_data_url(rgb: np.ndarray) -> str:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB image HxWx3, got shape={arr.shape}")
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_coordinate_grid_overlay(rgb: np.ndarray, *, step: int = 40) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(rgb)[:, :, :3], dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB image HxWx3, got shape={arr.shape}")
    height, width = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    line_color = (0, 255, 255)
    text_color = (0, 0, 255)
    for x in range(0, width, int(step)):
        cv2.line(bgr, (x, 0), (x, height - 1), line_color, 1)
        cv2.putText(bgr, str(x), (x + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
    for y in range(0, height, int(step)):
        cv2.line(bgr, (0, y), (width - 1, y), line_color, 1)
        cv2.putText(bgr, str(y), (2, min(height - 4, y + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
    cv2.rectangle(bgr, (0, 0), (width - 1, height - 1), (255, 0, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pose_from_agent_state(agent_state: Any) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    sensor_state = agent_state.sensor_states["color_sensor"]
    sensor_rot = quaternion.as_rotation_matrix(sensor_state.rotation)
    sensor_pos = np.asarray(sensor_state.position, dtype=np.float64).reshape(3)
    pose_mat = np.eye(4, dtype=np.float64)
    pose_mat[:3, :3] = sensor_rot
    pose_mat[:3, 3] = sensor_pos
    agent_pos = np.asarray(agent_state.position, dtype=np.float64).reshape(3)
    return pose_mat, (float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2]))


def _find_exact_frame_index(first_rgb: np.ndarray, color_list: Sequence[np.ndarray]) -> int:
    target = np.ascontiguousarray(np.asarray(first_rgb)[:, :, :3], dtype=np.uint8)
    diffs: List[Tuple[int, float]] = []
    for idx, frame in enumerate(color_list):
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[:2] != target.shape[:2] or arr.shape[2] < 3:
            continue
        cand = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
        if np.array_equal(cand, target):
            return int(idx)
        diffs.append((int(idx), float(np.mean(np.abs(cand.astype(np.int16) - target.astype(np.int16))))))
    best = sorted(diffs, key=lambda x: x[1])[:5]
    raise RuntimeError(f"could not align object_first_rgb to any current raw frame; best_mean_abs_diffs={best}")


def _state_summary(agent_state: Any, pose_mat: np.ndarray) -> Dict[str, Any]:
    sensor_pos = pose_mat[:3, 3]
    return {
        "agent_position_xyz": [float(x) for x in np.asarray(agent_state.position, dtype=float).reshape(3)],
        "sensor_position_xyz": [float(x) for x in sensor_pos],
        "sensor_rotation_matrix": [[float(v) for v in row] for row in pose_mat[:3, :3]],
    }


def register_new_object_observations(
    *,
    rep: Any,
    prev_object_count: int,
    color_list: Sequence[np.ndarray],
    depth_list: Sequence[np.ndarray],
    agent_state_list: Sequence[Any],
    object_observations_by_slot: MutableMapping[int, ObjectObservation],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if len(color_list) != len(depth_list) or len(color_list) != len(agent_state_list):
        raise RuntimeError(
            f"frame/depth/state length mismatch: color={len(color_list)} depth={len(depth_list)} "
            f"state={len(agent_state_list)}"
        )
    cur_count = int(np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0])
    prev = int(prev_object_count)
    if cur_count < prev:
        raise RuntimeError(f"object slot count shrank from {prev} to {cur_count}; depth observation cache is invalid")
    rgb_list = getattr(rep, "object_first_rgb", None)
    if rgb_list is None:
        raise RuntimeError("representation_manager.object_first_rgb is missing")
    if len(rgb_list) < cur_count:
        raise RuntimeError(f"object_first_rgb length {len(rgb_list)} < object count {cur_count}")

    records: List[Dict[str, Any]] = []
    for slot in range(prev, cur_count):
        first_rgb = rgb_list[slot]
        if first_rgb is None:
            raise RuntimeError(f"new object slot {slot} has no object_first_rgb")
        frame_index = _find_exact_frame_index(first_rgb, color_list)
        pose_mat, agent_pos = pose_from_agent_state(agent_state_list[frame_index])
        depth = np.ascontiguousarray(np.asarray(depth_list[frame_index], dtype=np.float32))
        rgb = np.ascontiguousarray(np.asarray(first_rgb)[:, :, :3], dtype=np.uint8)
        if depth.ndim != 2 or depth.shape[:2] != rgb.shape[:2]:
            raise RuntimeError(
                f"slot {slot} depth/rgb shape mismatch: depth={depth.shape}, rgb={rgb.shape}"
            )
        sensor_pos = tuple(float(x) for x in pose_mat[:3, 3].reshape(3))
        obs = ObjectObservation(
            slot_index=int(slot),
            source_frame_index=int(frame_index),
            rgb=rgb.copy(),
            depth=depth.copy(),
            pose_mat=pose_mat.copy(),
            agent_position_xyz=agent_pos,
            sensor_position_xyz=(sensor_pos[0], sensor_pos[1], sensor_pos[2]),
        )
        object_observations_by_slot[int(slot)] = obs
        rec = {
            "slot_index": int(slot),
            "source_frame_index": int(frame_index),
            "agent_position_xyz": [float(x) for x in obs.agent_position_xyz],
            "sensor_position_xyz": [float(x) for x in obs.sensor_position_xyz],
        }
        if output_dir is not None:
            slot_dir = Path(output_dir) / f"slot_{slot:03d}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = slot_dir / "first_rgb.jpg"
            depth_path = slot_dir / "first_depth.npy"
            depth_vis_path = slot_dir / "first_depth_vis.jpg"
            pose_path = slot_dir / "first_pose.npy"
            state_path = slot_dir / "first_state.json"
            save_rgb_jpg(obs.rgb, rgb_path)
            np.save(depth_path, obs.depth)
            np.save(pose_path, obs.pose_mat)
            save_depth_vis(obs.depth, depth_vis_path)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(_state_summary(agent_state_list[frame_index], obs.pose_mat), f, ensure_ascii=False, indent=2)
            rec.update(
                {
                    "first_rgb_path": str(rgb_path),
                    "first_depth_path": str(depth_path),
                    "first_depth_vis_path": str(depth_vis_path),
                    "first_pose_path": str(pose_path),
                    "first_state_path": str(state_path),
                }
            )
        records.append(rec)

    if output_dir is not None:
        out_path = Path(output_dir) / "object_observation_registration.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prev_object_count": int(prev),
                    "cur_object_count": int(cur_count),
                    "registered_slots": [int(x) for x in range(prev, cur_count)],
                    "records": records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return {
        "prev_object_count": int(prev),
        "cur_object_count": int(cur_count),
        "registered_slots": [int(x) for x in range(prev, cur_count)],
        "records": records,
    }


def load_topk_depthbox_candidates_from_stage2(
    *,
    stage2_json_path: Path,
    object_observations_by_slot: Mapping[int, ObjectObservation],
    top_k: int,
    baseline_memory_index: int,
) -> List[DepthBoxCandidate]:
    with open(stage2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objs = data.get("object_candidates")
    if not isinstance(objs, list):
        raise RuntimeError(f"stage2 object_candidates missing or invalid: {stage2_json_path}")
    ranked = sorted(objs, key=lambda x: float(x["og3d_logit"]), reverse=True)
    k = int(top_k)
    if len(ranked) < k:
        raise RuntimeError(f"stage2 has {len(ranked)} object candidates, need top_k={k}")
    candidates: List[DepthBoxCandidate] = []
    for rank, rec in enumerate(ranked[:k], start=1):
        slot = int(rec["slot_index"])
        if slot not in object_observations_by_slot:
            raise RuntimeError(f"slot {slot} has no saved first RGB/depth/pose observation")
        center = rec["center_habitat_xyz"]
        if not isinstance(center, list) or len(center) < 3:
            raise RuntimeError(f"candidate slot {slot} missing center_habitat_xyz")
        candidates.append(
            DepthBoxCandidate(
                rank=int(rank),
                slot_index=slot,
                og3d_logit=float(rec["og3d_logit"]),
                merged_object_score=float(rec["merged_object_score"]),
                center_habitat_xyz=(float(center[0]), float(center[1]), float(center[2])),
                observation=object_observations_by_slot[slot],
            )
        )
    if int(candidates[0].slot_index) != int(baseline_memory_index):
        raise RuntimeError(
            "baseline memory index mismatch: "
            f"stage2 rank1 slot={candidates[0].slot_index}, aux real_object_decision_idx={baseline_memory_index}"
        )
    return candidates


def call_vlm_select_and_box(
    *,
    description: str,
    candidates: Sequence[DepthBoxCandidate],
    cfg: VLMDepthBoxConfig,
) -> Dict[str, Any]:
    if len(candidates) == 0:
        raise RuntimeError("no candidates supplied to VLM depth box selector")
    h, w = candidates[0].observation.rgb.shape[:2]
    content: List[Dict[str, Any]] = []
    for cand in candidates:
        rgb = cand.observation.rgb
        if rgb.shape[:2] != (h, w):
            raise RuntimeError(f"candidate image shape mismatch: expected {(h, w)}, got {rgb.shape[:2]}")
        content.append({"type": "image_url", "image_url": {"url": rgb_to_jpeg_data_url(make_coordinate_grid_overlay(rgb))}})
    content.append(
        {
            "type": "text",
            "text": SELECT_BOX_USER_TEMPLATE.format(
                description=description.strip(),
                k=len(candidates),
                width=int(w),
                height=int(h),
            ),
        }
    )
    t0 = time.perf_counter()
    raw = chat_messages(
        messages=[
            {"role": "system", "content": SELECT_BOX_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=cfg.vlm_model,
        max_tokens=512,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    best = require_int_in_range(parsed, "best_index", 1, len(candidates))
    bbox = require_bbox(parsed, "bbox_xyxy", int(w), int(h))
    confidence = require_choice(parsed, "confidence", ("high", "medium", "low"))
    visible_constraints = require_string_list(parsed, "visible_constraints")
    missing_or_uncertain_constraints = require_string_list(parsed, "missing_or_uncertain_constraints")
    return {
        "raw": raw,
        "parsed": parsed,
        "best_index": int(best),
        "bbox_xyxy": [int(x) for x in bbox],
        "target_region": str(parsed.get("target_region", "")),
        "confidence": confidence,
        "visible_constraints": visible_constraints,
        "missing_or_uncertain_constraints": missing_or_uncertain_constraints,
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def estimate_xyz_from_bbox(
    *,
    observation: ObjectObservation,
    bbox_xyxy: Sequence[int],
    bbox_inner_fraction: float = 0.0,
) -> Dict[str, Any]:
    rgb = np.asarray(observation.rgb)
    depth_m = np.asarray(observation.depth, dtype=np.float32)
    if rgb.ndim != 3 or depth_m.ndim != 2 or rgb.shape[:2] != depth_m.shape[:2]:
        raise RuntimeError(f"rgb/depth shape mismatch: rgb={rgb.shape}, depth={depth_m.shape}")
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = [int(x) for x in bbox_xyxy]
    if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height or x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid bbox {bbox_xyxy!r} for image width={width}, height={height}")

    frac = float(bbox_inner_fraction)
    if frac < 0.0 or frac >= 0.5:
        raise RuntimeError(f"bbox_inner_fraction must be in [0,0.5), got {frac}")
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    ix1 = int(round(x1 + bw * frac))
    ix2 = int(round(x2 - bw * frac))
    iy1 = int(round(y1 + bh * frac))
    iy2 = int(round(y2 - bh * frac))
    if ix2 <= ix1 or iy2 <= iy1:
        raise RuntimeError(f"inner bbox collapsed from bbox={bbox_xyxy!r} frac={frac}")

    xs = np.arange(ix1, ix2 + 1, dtype=np.int32)
    ys = np.arange(iy1, iy2 + 1, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    depth_crop_m = depth_m[grid_y, grid_x]
    valid = depth_crop_m > 0
    if not np.any(valid):
        raise RuntimeError(f"bbox {bbox_xyxy!r} has no positive depth pixels")

    depth_valid_mm = depth_crop_m[valid].astype(np.float64) * 1000.0
    x_valid = grid_x[valid]
    y_valid = grid_y[valid]

    # Match data_utils.PQ3DModel.decision exactly: hfov=42, aspect=640/360,
    # w_ind linspace(-1,1,width), h_ind linspace(1,-1,height), then convert_from_uvd.
    intrinsic = make_intrinsic_hfov(42, 640 / 360)
    w_ind = np.linspace(-1, 1, width)
    h_ind = np.linspace(1, -1, height)
    u = w_ind[x_valid]
    v = h_ind[y_valid]
    xyz = convert_from_uvd(u, v, depth_valid_mm, intrinsic, observation.pose_mat)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] != 3:
        raise RuntimeError(f"unexpected projected xyz shape={xyz.shape}")
    median_xyz = np.median(xyz, axis=0)
    mean_xyz = np.mean(xyz, axis=0)
    depth_values_m = depth_valid_mm / 1000.0
    return {
        "estimated_target_xyz": [float(x) for x in median_xyz.reshape(3)],
        "mean_target_xyz": [float(x) for x in mean_xyz.reshape(3)],
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "inner_bbox_xyxy": [int(ix1), int(iy1), int(ix2), int(iy2)],
        "valid_depth_pixel_count": int(depth_valid_mm.shape[0]),
        "total_inner_bbox_pixel_count": int(depth_crop_m.size),
        "depth_m_min": float(np.min(depth_values_m)),
        "depth_m_p10": float(np.percentile(depth_values_m, 10)),
        "depth_m_median": float(np.median(depth_values_m)),
        "depth_m_p90": float(np.percentile(depth_values_m, 90)),
        "depth_m_max": float(np.max(depth_values_m)),
        "projection_alignment": {
            "intrinsic_hfov": 42,
            "aspect_ratio": 640 / 360,
            "u_grid": "np.linspace(-1, 1, width)",
            "v_grid": "np.linspace(1, -1, height)",
            "depth_unit_before_convert_from_uvd": "millimeters",
        },
    }


def draw_bbox_overlay(rgb: np.ndarray, bbox_xyxy: Sequence[int], path: Path, label: str) -> None:
    arr = np.ascontiguousarray(np.asarray(rgb)[:, :, :3], dtype=np.uint8).copy()
    x1, y1, x2, y2 = [int(x) for x in bbox_xyxy]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(bgr, label[:48], (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def build_effectiveness_record(
    *,
    baseline_target_xyz: np.ndarray,
    corrected_target_xyz: np.ndarray,
    goal_positions_xyz: Sequence[np.ndarray],
    threshold_m: float = 1.0,
) -> Dict[str, Any]:
    if len(goal_positions_xyz) == 0:
        raise RuntimeError("cannot build effectiveness record without goal positions")
    baseline = np.asarray(baseline_target_xyz, dtype=float).reshape(3)
    corrected = np.asarray(corrected_target_xyz, dtype=float).reshape(3)
    goals = [np.asarray(g, dtype=float).reshape(3) for g in goal_positions_xyz]
    bdist = float(min(np.linalg.norm(baseline - g) for g in goals))
    cdist = float(min(np.linalg.norm(corrected - g) for g in goals))
    bdist_xz = float(min(np.linalg.norm(baseline[[0, 2]] - g[[0, 2]]) for g in goals))
    cdist_xz = float(min(np.linalg.norm(corrected[[0, 2]] - g[[0, 2]]) for g in goals))
    b_ok = bool(bdist <= float(threshold_m))
    c_ok = bool(cdist <= float(threshold_m))
    b_ok_xz = bool(bdist_xz <= float(threshold_m))
    c_ok_xz = bool(cdist_xz <= float(threshold_m))
    return {
        "threshold_m": float(threshold_m),
        "baseline_nearest_goal_dist_m": bdist,
        "corrected_nearest_goal_dist_m": cdist,
        "baseline_topdown_xz_nearest_goal_dist_m": bdist_xz,
        "corrected_topdown_xz_nearest_goal_dist_m": cdist_xz,
        "baseline_within_threshold": b_ok,
        "corrected_within_threshold": c_ok,
        "case": ("1" if b_ok else "0") + ("1" if c_ok else "0"),
        "baseline_topdown_xz_within_threshold": b_ok_xz,
        "corrected_topdown_xz_within_threshold": c_ok_xz,
        "case_topdown_xz": ("1" if b_ok_xz else "0") + ("1" if c_ok_xz else "0"),
    }


def choose_navigation_target_from_projection(
    *,
    projected_target_xyz: np.ndarray,
    baseline_target_xyz: np.ndarray,
    selected_observation: ObjectObservation,
    target_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    projected = np.asarray(projected_target_xyz, dtype=float).reshape(3)
    baseline = np.asarray(baseline_target_xyz, dtype=float).reshape(3)
    mode = str(target_mode)
    if mode == "full_xyz":
        used = projected.copy()
    elif mode == "xz_baseline_y":
        used = np.asarray([projected[0], baseline[1], projected[2]], dtype=float)
    elif mode == "xz_agent_y":
        used = np.asarray([projected[0], selected_observation.agent_position_xyz[1], projected[2]], dtype=float)
    else:
        raise RuntimeError(f"unknown target_mode={target_mode!r}")
    return used, {
        "target_mode": mode,
        "coordinate_note": "Habitat uses x-z as the ground plane and y as height; planar modes keep projected x,z and replace y.",
        "full_projected_target_xyz": [float(x) for x in projected],
        "baseline_target_xyz": [float(x) for x in baseline],
        "selected_observation_agent_y": float(selected_observation.agent_position_xyz[1]),
        "used_navigation_target_xyz": [float(x) for x in used],
    }


def correct_final_decision_with_vlmdepthbox(
    *,
    description: str,
    decision_aux: Mapping[str, Any],
    stage2_json_path: Path,
    baseline_target_xyz: np.ndarray,
    object_observations_by_slot: Mapping[int, ObjectObservation],
    output_dir: Path,
    cfg: VLMDepthBoxConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not bool(decision_aux["is_object_decision"]):
        raise RuntimeError("VLMDepthBox was called for a non-object decision")
    baseline_memory_index = int(decision_aux["real_object_decision_idx"])
    candidates = load_topk_depthbox_candidates_from_stage2(
        stage2_json_path=stage2_json_path,
        object_observations_by_slot=object_observations_by_slot,
        top_k=int(cfg.top_k),
        baseline_memory_index=baseline_memory_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_records: List[Dict[str, Any]] = []
    for cand in candidates:
        stem = f"candidate_rank{cand.rank:02d}_slot{cand.slot_index:03d}"
        rgb_path = output_dir / f"{stem}_first_rgb.jpg"
        grid_path = output_dir / f"{stem}_first_rgb_grid.jpg"
        depth_path = output_dir / f"{stem}_first_depth.npy"
        depth_vis_path = output_dir / f"{stem}_first_depth_vis.jpg"
        pose_path = output_dir / f"{stem}_first_pose.npy"
        save_rgb_jpg(cand.observation.rgb, rgb_path)
        save_rgb_jpg(make_coordinate_grid_overlay(cand.observation.rgb), grid_path)
        np.save(depth_path, cand.observation.depth)
        np.save(pose_path, cand.observation.pose_mat)
        save_depth_vis(cand.observation.depth, depth_vis_path)
        candidate_records.append(
            {
                "rank": int(cand.rank),
                "slot_index": int(cand.slot_index),
                "og3d_logit": float(cand.og3d_logit),
                "merged_object_score": float(cand.merged_object_score),
                "center_habitat_xyz": [float(x) for x in cand.center_habitat_xyz],
                "source_frame_index": int(cand.observation.source_frame_index),
                "sensor_position_xyz": [float(x) for x in cand.observation.sensor_position_xyz],
                "first_rgb_path": str(rgb_path),
                "first_rgb_grid_path": str(grid_path),
                "first_depth_path": str(depth_path),
                "first_depth_vis_path": str(depth_vis_path),
                "first_pose_path": str(pose_path),
            }
        )

    selection = call_vlm_select_and_box(description=description, candidates=candidates, cfg=cfg)
    selected = candidates[int(selection["best_index"]) - 1]
    projection = estimate_xyz_from_bbox(
        observation=selected.observation,
        bbox_xyxy=selection["bbox_xyxy"],
        bbox_inner_fraction=float(cfg.bbox_inner_fraction),
    )
    projected_target = np.asarray(projection["estimated_target_xyz"], dtype=float).reshape(3)
    box_navigation_target, target_mode_info = choose_navigation_target_from_projection(
        projected_target_xyz=projected_target,
        baseline_target_xyz=np.asarray(baseline_target_xyz, dtype=float).reshape(3),
        selected_observation=selected.observation,
        target_mode=str(cfg.target_mode),
    )
    autonomous_navigation_used = int(selection["best_index"]) != 1
    baseline_target = np.asarray(baseline_target_xyz, dtype=float).reshape(3)
    final_navigation_target = box_navigation_target if autonomous_navigation_used else baseline_target.copy()
    overlay_path = output_dir / f"selected_rank{selected.rank:02d}_slot{selected.slot_index:03d}_bbox_overlay.jpg"
    draw_bbox_overlay(
        selected.observation.rgb,
        selection["bbox_xyxy"],
        overlay_path,
        f"rank{selected.rank} slot{selected.slot_index}",
    )
    info = {
        "vlmdepthbox_called": True,
        "baseline_target_xyz": baseline_target.tolist(),
        "full_projected_target_xyz": projected_target.tolist(),
        "box_navigation_target_xyz": box_navigation_target.tolist(),
        "corrected_target_xyz": final_navigation_target.tolist(),
        "autonomous_navigation_used": bool(autonomous_navigation_used),
        "used_target_source": "vlmdepthbox_projected_bbox" if autonomous_navigation_used else "baseline_object_position",
        "candidate_records": candidate_records,
        "selection": selection,
        "selected_rank": int(selected.rank),
        "selected_slot_index": int(selected.slot_index),
        "selected_candidate_center_habitat_xyz": [float(x) for x in selected.center_habitat_xyz],
        "bbox_projection": projection,
        "target_mode_info": target_mode_info,
        "bbox_overlay_path": str(overlay_path),
        "used_depth_box_target": bool(autonomous_navigation_used),
    }
    with open(output_dir / "vlmdepthbox_decision.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return final_navigation_target, info
