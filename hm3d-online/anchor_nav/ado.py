from collections import Counter
from typing import List, Optional, Tuple
import re
import os
import time

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
SAVE_DEBUG_IMAGES = os.environ.get("ANCHORNAV_SAVE_DEBUG_IMAGES", "1").strip().lower() not in {"0", "false", "off", "no"}
DEBUG_IMG_DIR = os.path.join("output_dirs", "anchor_logs", "phase_images")
_DEBUG_IMG_SEQ = 0


def _save_debug_image(image_rgb: np.ndarray, phase: str, tag: str) -> None:
    global _DEBUG_IMG_SEQ
    if not SAVE_DEBUG_IMAGES:
        return
    os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
    _DEBUG_IMG_SEQ += 1
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}-{phase}-{_DEBUG_IMG_SEQ:06d}-{tag}.jpg"
    path = os.path.join(DEBUG_IMG_DIR, filename)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def _normalize_json_dict(parsed, required_keys: List[str], context: str) -> dict:
    """
    Extract a dict carrying required keys from:
    - dict
    - list[dict, ...] (choose the first dict that contains all required keys)
    """
    if isinstance(parsed, dict):
        candidate = parsed
    elif isinstance(parsed, list):
        candidate = None
        for item in parsed:
            if isinstance(item, dict) and all(key in item for key in required_keys):
                candidate = item
                break
        if candidate is None:
            raise RuntimeError(
                f"{context} expects dict JSON with keys {required_keys}, "
                f"but list has no matching item. Value={parsed}"
            )
    else:
        raise RuntimeError(f"{context} expects dict/list JSON but got type={type(parsed)}. Value={parsed}")

    missing = [k for k in required_keys if k not in candidate]
    if missing:
        raise RuntimeError(f"{context} missing required keys: {missing}. Value={candidate}")
    return candidate


def _extract_best_match_and_confidence(parsed: dict, candidate_count: int) -> Tuple[Optional[int], float]:
    best_num = parsed.get("best_match")
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    if isinstance(best_num, str):
        match = re.search(r"\d+", best_num)
        best_num = int(match.group(0)) if match else None
    elif isinstance(best_num, (int, float)):
        best_num = int(best_num)
    else:
        best_num = None

    if best_num is not None and not (1 <= best_num <= candidate_count):
        best_num = None
    return best_num, confidence


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
        _save_debug_image(image, "phase1", f"frame{fi}-visible{len(frame_visible)}")
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
            raise RuntimeError("[AnchorNav/ADO] phase1 VLM returned empty response")
        parsed = extract_json(raw)
        parsed = _normalize_json_dict(parsed, required_keys=["scores"], context="[AnchorNav/ADO] phase1")
        scores = list(parsed["scores"])
    except Exception as err:
        raise RuntimeError(f"[AnchorNav/ADO] phase1 VLM failed: {err}") from err

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

    # Build multi-frame annotations so numbered circles are visible whenever candidates appear in any frame.
    frame_candidate_pixels = {}
    frame_visible_counts = []
    for frame_idx, agent_state in enumerate(agent_state_list):
        pixels = []
        visible = 0
        for idx in candidate_ids:
            center = np.asarray(object_box)[idx][:3]
            pixel = project_world_to_pixel(center, agent_state, IMG_WIDTH, IMG_HEIGHT)
            pixels.append(pixel)
            if pixel is not None:
                visible += 1
        frame_candidate_pixels[frame_idx] = pixels
        frame_visible_counts.append((frame_idx, visible))

    frame_visible_counts.sort(key=lambda x: x[1], reverse=True)
    selected_frame_indices = [idx for idx, cnt in frame_visible_counts[:3] if cnt > 0]
    if not selected_frame_indices:
        raise RuntimeError(
            "[AnchorNav/ADO] phase2 no candidate markers visible in any frame, "
            "cannot provide numbered circles to VLM"
        )

    annotated_images = []
    total_visible = 0
    found_anchors = goal_anchor.found_anchors(registry)
    for frame_idx in selected_frame_indices:
        frame = color_list[frame_idx].copy()
        agent_state = agent_state_list[frame_idx]
        points = frame_candidate_pixels[frame_idx]
        labels = [str(rank + 1) for rank in range(len(candidate_ids))]
        frame = draw_som_circles(frame, points, labels, color=(0, 200, 255))
        visible_here = sum(1 for p in points if p is not None)
        total_visible += visible_here

        for anchor in found_anchors:
            pos = registry.get_position(anchor.name)
            if pos is None:
                continue
            pixel = project_world_to_pixel(pos, agent_state, IMG_WIDTH, IMG_HEIGHT)
            if pixel is None:
                continue
            cv2.rectangle(frame, (pixel[0] - 10, pixel[1] - 10), (pixel[0] + 10, pixel[1] + 10), (0, 255, 0), 2)
            cv2.putText(frame, anchor.name, (pixel[0] + 12, pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        annotated_images.append(frame)

    print(
        f"[AnchorNav/ADO] phase2 markers visible: frames={selected_frame_indices}, "
        f"visible_total={total_visible}, candidates={len(candidate_ids)}"
    )

    prompt = (
        f'Target object: "{goal_anchor.target}".\n'
        "Green boxes are known reference anchors. Orange circles are candidate targets.\n"
        'Return ONLY JSON: {"best_match": <number or null>, "confidence": <0-1>}.'
    )
    for i, img in enumerate(annotated_images):
        _save_debug_image(img, "phase2", f"target-{goal_anchor.target}-view{i}")
    raw = None
    try:
        raw = call_vlm_image(annotated_images, prompt, model=VLM_MODEL_BEST)
        if raw is None:
            raise RuntimeError("[AnchorNav/ADO] phase2 VLM returned empty response")
        parsed = extract_json(raw)
        parsed = _normalize_json_dict(parsed, required_keys=["best_match", "confidence"], context="[AnchorNav/ADO] phase2")
        best_num, confidence = _extract_best_match_and_confidence(parsed, len(candidate_ids))
        if best_num is not None:
            chosen = best_num - 1
            if confidence < PHASE2_CONF_THRESHOLD:
                print(
                    f"[AnchorNav/ADO] phase2 low confidence accepted: "
                    f"best_match={best_num}, confidence={confidence:.3f} < {PHASE2_CONF_THRESHOLD}"
                )
            return np.asarray(object_box)[candidate_ids[chosen]][:3].copy()

        # Retry once with stricter instruction instead of immediate failure.
        retry_prompt = (
            f'Target object: "{goal_anchor.target}".\n'
            "Green boxes are known reference anchors. Orange circles are candidate targets.\n"
            f"You MUST choose exactly one candidate number from 1 to {len(candidate_ids)}.\n"
            'Return ONLY JSON object: {"best_match": <integer>, "confidence": <0-1>}. '
            "No explanation, no markdown."
        )
        print("[AnchorNav/ADO] phase2 first response missing valid best_match, retrying once")
        for i, img in enumerate(annotated_images):
            _save_debug_image(img, "phase2-retry", f"target-{goal_anchor.target}-view{i}")
        raw_retry = call_vlm_image(annotated_images, retry_prompt, model=VLM_MODEL_BEST)
        if raw_retry is None:
            raise RuntimeError("[AnchorNav/ADO] phase2 retry VLM returned empty response")
        parsed_retry = extract_json(raw_retry)
        parsed_retry = _normalize_json_dict(
            parsed_retry,
            required_keys=["best_match", "confidence"],
            context="[AnchorNav/ADO] phase2 retry",
        )
        best_num_retry, confidence_retry = _extract_best_match_and_confidence(parsed_retry, len(candidate_ids))
        if best_num_retry is not None:
            chosen = best_num_retry - 1
            if confidence_retry < PHASE2_CONF_THRESHOLD:
                print(
                    f"[AnchorNav/ADO] phase2 retry low confidence accepted: "
                    f"best_match={best_num_retry}, confidence={confidence_retry:.3f} < {PHASE2_CONF_THRESHOLD}"
                )
            return np.asarray(object_box)[candidate_ids[chosen]][:3].copy()
        raise RuntimeError(
            "[AnchorNav/ADO] phase2 retry still missing valid best_match. "
            f"first_raw={raw} retry_raw={raw_retry}"
        )
    except Exception as err:
        raise RuntimeError(f"[AnchorNav/ADO] phase2 VLM failed: {err}; raw={raw}") from err

