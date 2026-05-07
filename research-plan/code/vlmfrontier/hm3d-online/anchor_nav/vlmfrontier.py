from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import quaternion

from anchor_nav.posnode import stitch_panorama
from vlm.client import BASE_URL, DEFAULT_MODEL, chat_messages


@dataclass(frozen=True)
class VLMFrontierConfig:
    vlm_model: str = DEFAULT_MODEL
    vlm_timeout: int = 60
    view_count: int = 12
    semantic_weight: float = 0.45
    logit_weight: float = 0.55
    min_score_delta: float = 0.12


@dataclass(frozen=True)
class FrontierCandidate:
    frontier_index: int
    center_habitat_xyz: Tuple[float, float, float]
    og3d_logit: float


SYSTEM_PROMPT = (
    "You are a conservative semantic frontier advisor for embodied navigation. "
    "You do not choose coordinates. You only score panorama directions for whether exploring that direction "
    "is likely to reveal the target object or its diagnostic anchors. Return strict JSON only."
)

USER_TEMPLATE = """Navigation task:
{description}

INPUT:
The image is a stitched 360-degree panorama from the agent's current position. It is composed of {view_count}
numbered views in left-to-right order. Each view label is printed in the image.

TASK:
Score each view direction for whether moving/exploring through that direction is semantically promising for the
task target or its anchors. Use visible objects, room layout, doorways/open passages, and common indoor priors.

RULES:
1. Do not claim the target is present unless it is clearly visible.
2. A doorway/open passage toward a likely room can receive high room_potential even when the target is not visible.
3. If a direction only shows a dead end, plain wall, irrelevant clutter, or already-observed irrelevant area, mark avoid=true.
4. Keep scores conservative. Use 0.0..1.0 numbers.
5. Score every view index 1..{view_count}; do not omit any view.

Return strict JSON with exact keys:
{{
  "directions": [
    {{
      "view_index": <integer 1..{view_count}>,
      "target_potential": <number 0.0..1.0>,
      "anchor_potential": <number 0.0..1.0>,
      "room_potential": <number 0.0..1.0>,
      "avoid": <true_or_false>,
      "reason": "<brief reason>"
    }}
  ],
  "best_view_indices": [<integer 1..{view_count}>, "..."],
  "reason": "<brief global reason>"
}}
"""


def parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"VLM response is not a JSON object: {raw!r}")
    return parsed


def _as_rgb_uint8(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise RuntimeError(f"expected RGB image HxWx3, got shape={arr.shape}")
    return np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)


def current_decision_panorama_frames(color_list: Sequence[np.ndarray], view_count: int) -> List[np.ndarray]:
    if len(color_list) < int(view_count):
        raise RuntimeError(f"need at least {view_count} color frames for frontier panorama, got {len(color_list)}")
    frames = list(color_list[-int(view_count):])
    frames = list(reversed(frames))
    return [_as_rgb_uint8(x) for x in frames]


def register_new_object_panorama_frames(
    *,
    rep: Any,
    prev_object_count: int,
    color_list: Sequence[np.ndarray],
    panorama_frames_by_slot: Dict[int, List[np.ndarray]],
    view_count: int = 12,
) -> Dict[str, Any]:
    cur_count = int(np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0])
    prev = int(prev_object_count)
    if cur_count < prev:
        raise RuntimeError(f"object slot count shrank from {prev} to {cur_count}; panorama slot map is invalid")
    frames = current_decision_panorama_frames(color_list, int(view_count))
    new_slots = list(range(prev, cur_count))
    for slot in new_slots:
        panorama_frames_by_slot[int(slot)] = [x.copy() for x in frames]
    return {
        "prev_object_count": prev,
        "cur_object_count": cur_count,
        "registered_slots": [int(x) for x in new_slots],
        "registered_frame_count": int(len(frames)),
    }


def _label_frame(rgb: np.ndarray, label: str) -> np.ndarray:
    out = _as_rgb_uint8(rgb).copy()
    cv2.rectangle(out, (8, 8), (150, 50), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_rgb_jpg(rgb: np.ndarray, path: Path, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(_as_rgb_uint8(rgb), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def jpeg_data_url_from_file(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_frontier_candidates(stage2_json_path: Path) -> List[FrontierCandidate]:
    with open(stage2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("frontier_candidates")
    if not isinstance(rows, list):
        raise RuntimeError(f"stage2 frontier_candidates missing or invalid: {stage2_json_path}")
    out: List[FrontierCandidate] = []
    for rec in rows:
        center = rec["center_habitat_xyz"]
        if not isinstance(center, list) or len(center) < 3:
            raise RuntimeError(f"frontier candidate missing center_habitat_xyz: {rec!r}")
        out.append(
            FrontierCandidate(
                frontier_index=int(rec["frontier_index"]),
                center_habitat_xyz=(float(center[0]), float(center[1]), float(center[2])),
                og3d_logit=float(rec["og3d_logit"]),
            )
        )
    if len(out) == 0:
        raise RuntimeError("no frontier candidates to rerank")
    return out


def _normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _agent_forward_xz(agent_state: Any) -> np.ndarray:
    rot = quaternion.as_rotation_matrix(agent_state.rotation)
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=float)
    xz = np.array([float(forward[0]), float(forward[2])], dtype=float)
    n = float(np.linalg.norm(xz))
    if n < 1e-12:
        raise RuntimeError("agent forward vector is degenerate")
    return xz / n


def frontier_view_index(
    *,
    agent_state: Any,
    frontier_xyz: Sequence[float],
    view_count: int,
) -> int:
    agent_pos = np.asarray(agent_state.position, dtype=float).reshape(3)
    frontier_pos = np.asarray(frontier_xyz, dtype=float).reshape(3)
    vec = np.array([float(frontier_pos[0] - agent_pos[0]), float(frontier_pos[2] - agent_pos[2])], dtype=float)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        raise RuntimeError("frontier coincides with agent position; cannot compute bearing")
    vec = vec / n
    fwd = _agent_forward_xz(agent_state)
    cross = fwd[0] * vec[1] - fwd[1] * vec[0]
    dot = float(np.clip(float(np.dot(fwd, vec)), -1.0, 1.0))
    rel = math.atan2(cross, dot)
    # rel=-pi..pi. Map to 1..view_count in the same left-to-right panorama order used by reversed scan frames.
    u = (rel + math.pi) / (2.0 * math.pi)
    idx0 = int(np.clip(round(u * (int(view_count) - 1)), 0, int(view_count) - 1))
    return int(idx0 + 1)


def call_vlm_score_directions(
    *,
    description: str,
    panorama_path: Path,
    cfg: VLMFrontierConfig,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": jpeg_data_url_from_file(panorama_path)}},
        {"type": "text", "text": USER_TEMPLATE.format(description=description.strip(), view_count=int(cfg.view_count))},
    ]
    t0 = time.perf_counter()
    raw = chat_messages(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=cfg.vlm_model,
        max_tokens=1024,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    rows = parsed["directions"]
    if not isinstance(rows, list):
        raise RuntimeError("VLM directions must be a list")
    scores = np.zeros(int(cfg.view_count), dtype=float)
    records: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError(f"direction row is not object: {row!r}")
        vi = int(row["view_index"])
        if vi < 1 or vi > int(cfg.view_count):
            raise RuntimeError(f"view_index out of range: {vi}")
        seen.add(vi)
        target_p = float(row["target_potential"])
        anchor_p = float(row["anchor_potential"])
        room_p = float(row["room_potential"])
        avoid = bool(row["avoid"])
        if min(target_p, anchor_p, room_p) < 0.0 or max(target_p, anchor_p, room_p) > 1.0:
            raise RuntimeError(f"direction potentials must be 0..1: {row!r}")
        score = 0.55 * target_p + 0.30 * anchor_p + 0.15 * room_p
        if avoid:
            score *= 0.35
        scores[vi - 1] = float(score)
        records.append(
            {
                "view_index": int(vi),
                "target_potential": float(target_p),
                "anchor_potential": float(anchor_p),
                "room_potential": float(room_p),
                "avoid": bool(avoid),
                "semantic_score": float(score),
                "reason": str(row.get("reason", "")),
            }
        )
    missing = [i for i in range(1, int(cfg.view_count) + 1) if i not in seen]
    if missing:
        raise RuntimeError(f"VLM omitted view indices: {missing}")
    return {
        "raw": raw,
        "parsed": parsed,
        "direction_scores": [float(x) for x in scores.tolist()],
        "direction_records": records,
        "best_view_indices": parsed.get("best_view_indices", []),
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def correct_frontier_with_vlm(
    *,
    description: str,
    agent_state: Any,
    color_list: Sequence[np.ndarray],
    stage2_json_path: Path,
    baseline_target_xyz: np.ndarray,
    output_dir: Path,
    cfg: VLMFrontierConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    frontiers = load_frontier_candidates(stage2_json_path)
    frames = current_decision_panorama_frames(color_list, int(cfg.view_count))
    labeled = [_label_frame(frame, f"VIEW {i}") for i, frame in enumerate(frames, start=1)]
    panorama = stitch_panorama(labeled)
    output_dir.mkdir(parents=True, exist_ok=True)
    panorama_path = output_dir / "current_labeled_frontier_panorama.jpg"
    save_rgb_jpg(panorama, panorama_path)

    vlm = call_vlm_score_directions(description=description, panorama_path=panorama_path, cfg=cfg)
    dir_scores = np.asarray(vlm["direction_scores"], dtype=float).reshape(-1)
    norm_logits = _normalize([f.og3d_logit for f in frontiers])

    rows: List[Dict[str, Any]] = []
    final_scores: List[float] = []
    for i, f in enumerate(frontiers):
        vi = frontier_view_index(agent_state=agent_state, frontier_xyz=f.center_habitat_xyz, view_count=int(cfg.view_count))
        semantic = float(dir_scores[vi - 1])
        final = float(cfg.logit_weight) * float(norm_logits[i]) + float(cfg.semantic_weight) * semantic
        final_scores.append(final)
        rows.append(
            {
                "frontier_index": int(f.frontier_index),
                "center_habitat_xyz": [float(x) for x in f.center_habitat_xyz],
                "og3d_logit": float(f.og3d_logit),
                "normalized_logit": float(norm_logits[i]),
                "mapped_view_index": int(vi),
                "semantic_score": float(semantic),
                "final_score": float(final),
            }
        )

    selected_i = int(np.argmax(np.asarray(final_scores, dtype=float)))
    selected = frontiers[selected_i]
    baseline = np.asarray(baseline_target_xyz, dtype=float).reshape(3)
    baseline_i = min(
        range(len(frontiers)),
        key=lambda j: float(np.linalg.norm(np.asarray(frontiers[j].center_habitat_xyz, dtype=float).reshape(3) - baseline)),
    )
    score_delta = float(final_scores[selected_i] - final_scores[baseline_i])
    correction_applied = bool(selected_i != baseline_i and score_delta >= float(cfg.min_score_delta))
    corrected = (
        np.asarray(selected.center_habitat_xyz, dtype=float).reshape(3)
        if correction_applied
        else baseline.copy()
    )
    info = {
        "vlmfrontier_called": True,
        "base_url": BASE_URL,
        "vlm_model": cfg.vlm_model,
        "baseline_target_xyz": [float(x) for x in baseline.tolist()],
        "selected_target_xyz": [float(x) for x in np.asarray(selected.center_habitat_xyz, dtype=float).reshape(3).tolist()],
        "corrected_target_xyz": [float(x) for x in corrected.tolist()],
        "baseline_frontier_index": int(frontiers[baseline_i].frontier_index),
        "selected_frontier_index": int(selected.frontier_index),
        "score_delta": float(score_delta),
        "correction_applied": bool(correction_applied),
        "correction_rejected_reason": "accepted" if correction_applied else f"score_delta_below_threshold_or_same:{score_delta:.3f}",
        "panorama_path": str(panorama_path),
        "vlm_direction_scores": vlm,
        "frontier_records": rows,
    }
    with open(output_dir / "vlmfrontier_decision.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return corrected, info


def nearest_goal_distance(position_xyz: np.ndarray, goal_positions_xyz: Sequence[np.ndarray]) -> float:
    if len(goal_positions_xyz) == 0:
        raise RuntimeError("no goal positions available for frontier effectiveness logging")
    p = np.asarray(position_xyz, dtype=float).reshape(3)
    dists = [float(np.linalg.norm(p - np.asarray(g, dtype=float).reshape(3))) for g in goal_positions_xyz]
    return float(min(dists))


def build_frontier_effectiveness_record(
    *,
    baseline_target_xyz: np.ndarray,
    corrected_target_xyz: np.ndarray,
    goal_positions_xyz: Sequence[np.ndarray],
    threshold_m: float = 1.0,
) -> Dict[str, Any]:
    baseline_dist = nearest_goal_distance(baseline_target_xyz, goal_positions_xyz)
    corrected_dist = nearest_goal_distance(corrected_target_xyz, goal_positions_xyz)
    baseline_ok = bool(baseline_dist <= float(threshold_m))
    corrected_ok = bool(corrected_dist <= float(threshold_m))
    return {
        "threshold_m": float(threshold_m),
        "baseline_nearest_goal_dist_m": float(baseline_dist),
        "corrected_nearest_goal_dist_m": float(corrected_dist),
        "baseline_in_1m": bool(baseline_ok),
        "corrected_in_1m": bool(corrected_ok),
        "case": f"{1 if baseline_ok else 0}{1 if corrected_ok else 0}",
    }
