from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from anchor_nav.posnode import _subsample_frames_evenly, stitch_panorama
from vlm.client import BASE_URL, DEFAULT_MODEL, chat_messages


@dataclass(frozen=True)
class VLMTop5Config:
    top_k: int = 5
    vlm_model: str = DEFAULT_MODEL
    vlm_timeout: int = 60
    panorama_subsample_frames: int = 12


@dataclass(frozen=True)
class ObjectCandidate:
    rank: int
    slot_index: int
    og3d_logit: float
    merged_object_score: float
    center_habitat_xyz: Tuple[float, float, float]
    first_rgb: np.ndarray


SELECT_SYSTEM_PROMPT = (
    "You compare ordered candidate object images for one embodied navigation task. "
    "Choose the candidate that best matches the exact target INSTANCE, not merely the object category. "
    "Return strict JSON only."
)

SELECT_USER_TEMPLATE = """Navigation task:
{description}

There are exactly {k} candidate object images, in the same order as the detector's baseline logit ranking.
Image 1 is the baseline top-1 object. Images 2..{k} are the next object candidates.

Before choosing, infer the task's discriminative visual constraints:
- the main target object category and attributes;
- nearby anchors and spatial relations;
- distinctive scene context that disambiguates this instance from similar objects.

Choose the single image that best matches the COMPLETE target instance. Penalize candidates that only match a generic
category or one common cue (for example, a table with a lamp) but miss stronger instance-specific context such as nearby
chairs, piano, wall art, materials, colors, or spatial relations. If multiple images show the same generic object type,
prefer the one with more of the unique anchors and relations from the full task description.

Return strict JSON with exact keys:
{{"best_index": <integer 1..{k}>, "match_type": "exact_target|partial_target|context_only", "reason": "<brief English reason naming the decisive constraints and any missing constraints>"}}
"""

PANORAMA_SYSTEM_PROMPT = (
    "You verify a stitched panorama for an embodied navigation task. Return strict JSON only."
)

PANORAMA_USER_TEMPLATE = """Navigation task:
{description}

This stitched panorama is the full 360-degree scan from the decision step associated with the selected candidate object image.
Decide whether the panorama contains the task's target INSTANCE. Set has_task_object=true only if the target object and
enough discriminative context from the task are visible to distinguish it from similar objects elsewhere. Do not count a
mere room label or a generic category match as sufficient.

Return strict JSON with exact keys:
{{"has_task_object": <true_or_false>, "confidence": "high|medium|low", "reason": "<brief English reason>"}}
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


def current_decision_panorama_frames(color_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    if len(color_list) < 12:
        raise RuntimeError(f"need at least 12 color frames for decision panorama, got {len(color_list)}")
    frames = list(color_list[-12:])
    frames = list(reversed(frames))
    return [np.ascontiguousarray(np.asarray(x)[:, :, :3], dtype=np.uint8) for x in frames]


def register_new_object_panorama_frames(
    *,
    rep: Any,
    prev_object_count: int,
    color_list: Sequence[np.ndarray],
    panorama_frames_by_slot: Dict[int, List[np.ndarray]],
) -> Dict[str, Any]:
    cur_count = int(np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1).shape[0])
    prev = int(prev_object_count)
    if cur_count < prev:
        raise RuntimeError(f"object slot count shrank from {prev} to {cur_count}; panorama slot map is invalid")
    frames = current_decision_panorama_frames(color_list)
    new_slots = list(range(prev, cur_count))
    for slot in new_slots:
        panorama_frames_by_slot[int(slot)] = [x.copy() for x in frames]
    return {
        "prev_object_count": prev,
        "cur_object_count": cur_count,
        "registered_slots": [int(x) for x in new_slots],
        "registered_frame_count": int(len(frames)),
    }


def stitch_and_save_panorama(
    *,
    frames: Sequence[np.ndarray],
    output_path: Path,
    max_frames: int,
) -> Path:
    sampled = _subsample_frames_evenly(list(frames), max_frames=int(max_frames))
    if len(sampled) == 0:
        raise RuntimeError("no frames left after panorama subsampling")
    pano = stitch_panorama(sampled)
    save_rgb_jpg(pano, output_path, quality=92)
    return output_path


def _candidate_from_stage2_record(
    *,
    rec: Mapping[str, Any],
    rank: int,
    rep: Any,
) -> ObjectCandidate:
    slot = int(rec["slot_index"])
    rgb_list = getattr(rep, "object_first_rgb", None)
    if rgb_list is None:
        raise RuntimeError("representation_manager.object_first_rgb is missing")
    if slot < 0 or slot >= len(rgb_list):
        raise RuntimeError(f"candidate slot {slot} out of object_first_rgb range {len(rgb_list)}")
    first_rgb = rgb_list[slot]
    if first_rgb is None:
        raise RuntimeError(f"candidate slot {slot} has no first-detection RGB")
    first_rgb_arr = np.asarray(first_rgb)
    if first_rgb_arr.ndim != 3 or first_rgb_arr.shape[2] < 3:
        raise RuntimeError(f"candidate slot {slot} has invalid first RGB shape {first_rgb_arr.shape}")
    center = rec["center_habitat_xyz"]
    if not isinstance(center, list) or len(center) < 3:
        raise RuntimeError(f"candidate slot {slot} missing center_habitat_xyz")
    return ObjectCandidate(
        rank=int(rank),
        slot_index=slot,
        og3d_logit=float(rec["og3d_logit"]),
        merged_object_score=float(rec["merged_object_score"]),
        center_habitat_xyz=(float(center[0]), float(center[1]), float(center[2])),
        first_rgb=np.ascontiguousarray(first_rgb_arr[:, :, :3], dtype=np.uint8),
    )


def load_topk_candidates_from_stage2(
    *,
    stage2_json_path: Path,
    rep: Any,
    top_k: int,
    baseline_memory_index: int,
) -> List[ObjectCandidate]:
    with open(stage2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objs = data.get("object_candidates")
    if not isinstance(objs, list):
        raise RuntimeError(f"stage2 object_candidates missing or invalid: {stage2_json_path}")
    ranked = sorted(objs, key=lambda x: float(x["og3d_logit"]), reverse=True)
    k = int(top_k)
    if len(ranked) < k:
        raise RuntimeError(f"stage2 has {len(ranked)} object candidates, need top_k={k}")
    candidates = [
        _candidate_from_stage2_record(rec=rec, rank=rank, rep=rep)
        for rank, rec in enumerate(ranked[:k], start=1)
    ]
    if int(candidates[0].slot_index) != int(baseline_memory_index):
        raise RuntimeError(
            "baseline memory index mismatch: "
            f"stage2 rank1 slot={candidates[0].slot_index}, aux real_object_decision_idx={baseline_memory_index}"
        )
    return candidates


def call_vlm_choose_candidate(
    *,
    description: str,
    candidates: Sequence[ObjectCandidate],
    cfg: VLMTop5Config,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for cand in candidates:
        content.append({"type": "image_url", "image_url": {"url": rgb_to_jpeg_data_url(cand.first_rgb)}})
    content.append({"type": "text", "text": SELECT_USER_TEMPLATE.format(description=description.strip(), k=len(candidates))})
    t0 = time.perf_counter()
    raw = chat_messages(
        messages=[
            {"role": "system", "content": SELECT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=cfg.vlm_model,
        max_tokens=256,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    best = int(parsed["best_index"])
    if best < 1 or best > len(candidates):
        raise RuntimeError(f"VLM best_index out of range: {best}, expected 1..{len(candidates)}")
    match_type = str(parsed["match_type"])
    if match_type not in {"exact_target", "partial_target", "context_only"}:
        raise RuntimeError(f"unexpected VLM match_type={match_type!r}")
    return {
        "raw": raw,
        "parsed": parsed,
        "best_index": best,
        "match_type": match_type,
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def call_vlm_verify_panorama(
    *,
    description: str,
    panorama_path: Path,
    cfg: VLMTop5Config,
) -> Dict[str, Any]:
    data = panorama_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": PANORAMA_USER_TEMPLATE.format(description=description.strip())},
    ]
    t0 = time.perf_counter()
    raw = chat_messages(
        messages=[
            {"role": "system", "content": PANORAMA_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model=cfg.vlm_model,
        max_tokens=192,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    has_task_object = bool(parsed["has_task_object"])
    confidence = str(parsed["confidence"])
    if confidence not in {"high", "medium", "low"}:
        raise RuntimeError(f"unexpected panorama confidence={confidence!r}")
    return {
        "raw": raw,
        "parsed": parsed,
        "has_task_object": has_task_object,
        "confidence": confidence,
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def correct_final_decision_with_vlmtop5(
    *,
    description: str,
    rep: Any,
    decision_aux: Mapping[str, Any],
    stage2_json_path: Path,
    baseline_target_xyz: np.ndarray,
    panorama_frames_by_slot: Mapping[int, Sequence[np.ndarray]],
    output_dir: Path,
    cfg: VLMTop5Config,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not bool(decision_aux["is_object_decision"]):
        raise RuntimeError("VLMTop5 was called for a non-object decision")
    baseline_memory_index = int(decision_aux["real_object_decision_idx"])
    candidates = load_topk_candidates_from_stage2(
        stage2_json_path=stage2_json_path,
        rep=rep,
        top_k=int(cfg.top_k),
        baseline_memory_index=baseline_memory_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_records: List[Dict[str, Any]] = []
    for cand in candidates:
        img_path = output_dir / f"candidate_rank{cand.rank:02d}_slot{cand.slot_index:03d}_first_rgb.jpg"
        save_rgb_jpg(cand.first_rgb, img_path)
        candidate_records.append(
            {
                "rank": int(cand.rank),
                "slot_index": int(cand.slot_index),
                "og3d_logit": float(cand.og3d_logit),
                "merged_object_score": float(cand.merged_object_score),
                "center_habitat_xyz": [float(x) for x in cand.center_habitat_xyz],
                "first_rgb_path": str(img_path),
            }
        )

    selection = call_vlm_choose_candidate(description=description, candidates=candidates, cfg=cfg)
    selected = candidates[int(selection["best_index"]) - 1]
    correction_applied = int(selection["best_index"]) != 1
    panorama_verify = None
    selected_panorama_path = None
    panorama_vetoed_correction = False

    if correction_applied:
        if int(selected.slot_index) not in panorama_frames_by_slot:
            raise RuntimeError(f"missing panorama frames for selected slot {selected.slot_index}")
        selected_panorama_path_obj = output_dir / (
            f"selected_rank{selected.rank:02d}_slot{selected.slot_index:03d}_stitched_panorama.jpg"
        )
        selected_panorama_path = stitch_and_save_panorama(
            frames=panorama_frames_by_slot[int(selected.slot_index)],
            output_path=selected_panorama_path_obj,
            max_frames=int(cfg.panorama_subsample_frames),
        )
        panorama_verify = call_vlm_verify_panorama(
            description=description,
            panorama_path=selected_panorama_path,
            cfg=cfg,
        )
        panorama_vetoed_correction = not bool(panorama_verify["has_task_object"])

    use_selected = bool(correction_applied) and not bool(panorama_vetoed_correction)
    corrected_target = np.asarray(selected.center_habitat_xyz, dtype=float).reshape(3) if use_selected else np.asarray(
        baseline_target_xyz, dtype=float
    ).reshape(3)
    info = {
        "vlmtop5_called": True,
        "base_url": BASE_URL,
        "vlm_model": cfg.vlm_model,
        "top_k": int(cfg.top_k),
        "baseline_memory_index": int(baseline_memory_index),
        "baseline_target_xyz": [float(x) for x in np.asarray(baseline_target_xyz, dtype=float).reshape(3)],
        "candidate_records": candidate_records,
        "selection": selection,
        "selected_rank": int(selected.rank),
        "selected_slot_index": int(selected.slot_index),
        "selected_target_xyz": [float(x) for x in selected.center_habitat_xyz],
        "correction_applied": bool(correction_applied),
        "panorama_vetoed_correction": bool(panorama_vetoed_correction),
        "selected_target_used": bool(use_selected),
        "panorama_verify": panorama_verify,
        "selected_panorama_path": str(selected_panorama_path) if selected_panorama_path is not None else None,
        "corrected_target_xyz": [float(x) for x in corrected_target.tolist()],
    }
    with open(output_dir / "vlmtop5_decision.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return corrected_target, info


def nearest_goal_distance(position_xyz: np.ndarray, goal_positions_xyz: Sequence[np.ndarray]) -> float:
    if len(goal_positions_xyz) == 0:
        raise RuntimeError("no goal positions available for effectiveness logging")
    p = np.asarray(position_xyz, dtype=float).reshape(3)
    dists = [float(np.linalg.norm(p - np.asarray(g, dtype=float).reshape(3))) for g in goal_positions_xyz]
    return float(min(dists))


def build_effectiveness_record(
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
