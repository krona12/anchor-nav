from collections import Counter
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .proj_utils import IMG_HEIGHT, IMG_WIDTH, draw_som_circles, project_world_to_pixel
from .types import GoalAnchor
from .vlm_adapter import VLM_MODEL_BEST, VLM_MODEL_FAST, call_vlm_image, extract_json


CLIP_GAP_THRESHOLD = 0.15
PHASE2_CONF_THRESHOLD = 0.6
PHASE2_TOP_K = 5


def phase1_override_frontier(
    frontier_waypoints: List[np.ndarray],
    color_list: List[np.ndarray],
    agent_state_list: List,
    missing_anchor_names: List[str],
    agent_position: np.ndarray,
) -> Optional[np.ndarray]:
    if not frontier_waypoints or not missing_anchor_names:
        return None

    best_frame_for_frontier: List[Optional[int]] = []
    best_pixel_for_frontier = []
    for frontier in frontier_waypoints:
        best_idx = None
        best_pixel = None
        best_center_dist = float("inf")
        for frame_idx, agent_state in enumerate(agent_state_list):
            pixel = project_world_to_pixel(frontier, agent_state, IMG_WIDTH, IMG_HEIGHT)
            if pixel is None:
                continue
            center_dist = np.linalg.norm(np.array(pixel) - np.array([IMG_WIDTH // 2, IMG_HEIGHT // 2]))
            if center_dist < best_center_dist:
                best_center_dist = center_dist
                best_idx = frame_idx
                best_pixel = pixel
        best_frame_for_frontier.append(best_idx)
        best_pixel_for_frontier.append(best_pixel)

    visible_ids = [idx for idx, px in enumerate(best_pixel_for_frontier) if px is not None]
    if not visible_ids:
        return None

    frame_count = Counter(best_frame_for_frontier[idx] for idx in visible_ids if best_frame_for_frontier[idx] is not None)
    selected_frames = [fi for fi, _ in frame_count.most_common(3)]
    if not selected_frames:
        return None

    annotated = []
    visible_labels = []
    for fi in selected_frames:
        image = color_list[fi].copy()
        frame_visible = [idx for idx in visible_ids if best_frame_for_frontier[idx] == fi]
        if not frame_visible:
            continue
        points = [best_pixel_for_frontier[idx] for idx in frame_visible]
        labels = [str(idx + 1) for idx in frame_visible]
        image = draw_som_circles(image, points, labels, color=(0, 255, 0))
        annotated.append(image)
        visible_labels.extend(frame_visible)

    if not annotated:
        return None
    unique_visible = sorted(set(visible_labels))

    prompt = (
        "You are helping indoor exploration. Missing anchors are: "
        + ", ".join(missing_anchor_names)
        + ".\nThe image contains numbered circles for candidate frontier directions. "
        + "Give a score 0.0-1.0 for each visible number by likelihood of finding the missing anchors. "
        + 'Return ONLY JSON: {"scores": [..]} where the score order follows the visible numbers in ascending order.'
    )

    try:
        raw = call_vlm_image(annotated, prompt, model=VLM_MODEL_FAST)
        if raw is None:
            return None
        parsed = extract_json(raw)
        scores = list(parsed.get("scores", []))
    except Exception as err:
        print(f"[AnchorNav/ADO] phase1 vlm failed: {err}")
        return None

    if len(scores) < len(unique_visible):
        scores = scores + [0.5] * (len(unique_visible) - len(scores))
    scores = scores[: len(unique_visible)]

    best_frontier = None
    best_utility = -1.0
    for rank, frontier_idx in enumerate(unique_visible):
        score = float(scores[rank])
        dist = float(np.linalg.norm(frontier_waypoints[frontier_idx] - agent_position)) + 1e-3
        utility = score / dist
        if utility > best_utility:
            best_utility = utility
            best_frontier = frontier_waypoints[frontier_idx]
    return best_frontier


def _encode_text(clip_text_model, clip_tokenizer, text: str) -> torch.Tensor:
    tokens = clip_tokenizer([text], truncation=True, padding=True, return_tensors="pt")
    device = next(clip_text_model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = clip_text_model(**tokens)
        pooled = out.pooler_output
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        pooled = F.normalize(pooled, dim=-1)
    return pooled.squeeze(0).detach().cpu()


def phase2_override_target(
    representation_manager,
    goal_anchor: GoalAnchor,
    registry,
    color_list: List[np.ndarray],
    agent_state_list: List,
    clip_text_model,
    clip_tokenizer,
) -> Optional[np.ndarray]:
    open_vocab_feat = getattr(representation_manager, "open_vocab_feat", None)
    object_box = getattr(representation_manager, "object_box", None)
    if open_vocab_feat is None or object_box is None or len(open_vocab_feat) == 0:
        return None

    feat = torch.from_numpy(np.asarray(open_vocab_feat)).float()
    feat = F.normalize(feat, dim=-1)
    text = _encode_text(clip_text_model, clip_tokenizer, goal_anchor.target)
    sims = (feat @ text).numpy()
    topk = min(PHASE2_TOP_K, len(sims))
    candidate_ids = np.argsort(sims)[-topk:][::-1]
    if len(candidate_ids) == 0:
        return None
    if len(candidate_ids) == 1:
        return np.asarray(object_box)[candidate_ids[0]][:3].copy()

    if float(sims[candidate_ids[0]] - sims[candidate_ids[1]]) > CLIP_GAP_THRESHOLD:
        return np.asarray(object_box)[candidate_ids[0]][:3].copy()

    latest_agent_state = agent_state_list[-1]
    latest_frame = color_list[-1].copy()

    candidate_pixels = []
    for rank, idx in enumerate(candidate_ids):
        center = np.asarray(object_box)[idx][:3]
        pixel = project_world_to_pixel(center, latest_agent_state, IMG_WIDTH, IMG_HEIGHT)
        candidate_pixels.append(pixel)
        latest_frame = draw_som_circles(latest_frame, [pixel], [str(rank + 1)], color=(0, 200, 255))

    found_anchors = goal_anchor.found_anchors(registry)
    for anchor in found_anchors:
        pos = registry.get_position(anchor.name)
        if pos is None:
            continue
        pixel = project_world_to_pixel(pos, latest_agent_state, IMG_WIDTH, IMG_HEIGHT)
        if pixel is None:
            continue
        cv2.rectangle(latest_frame, (pixel[0] - 10, pixel[1] - 10), (pixel[0] + 10, pixel[1] + 10), (0, 255, 0), 2)
        cv2.putText(latest_frame, anchor.name, (pixel[0] + 12, pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    prompt = (
        f'Target object: "{goal_anchor.target}".\n'
        "Green boxes are known reference anchors. Orange circles are candidate targets.\n"
        'Return ONLY JSON: {"best_match": <number or null>, "confidence": <0-1>}.'
    )
    try:
        raw = call_vlm_image([latest_frame], prompt, model=VLM_MODEL_BEST)
        if raw is None:
            return np.asarray(object_box)[candidate_ids[0]][:3].copy()
        parsed = extract_json(raw)
        best_num = parsed.get("best_match")
        confidence = float(parsed.get("confidence", 0.0))
        if best_num is not None and confidence > PHASE2_CONF_THRESHOLD:
            chosen = int(best_num) - 1
            if 0 <= chosen < len(candidate_ids):
                return np.asarray(object_box)[candidate_ids[chosen]][:3].copy()
    except Exception as err:
        print(f"[AnchorNav/ADO] phase2 vlm failed: {err}")

    return np.asarray(object_box)[candidate_ids[0]][:3].copy()

