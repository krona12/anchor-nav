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
class VLMEvidenceConfig:
    top_k: int = 5
    vlm_model: str = DEFAULT_MODEL
    vlm_timeout: int = 60
    panorama_subsample_frames: int = 12
    evidence_delta: float = 0.20


@dataclass(frozen=True)
class ObjectCandidate:
    rank: int
    slot_index: int
    og3d_logit: float
    merged_object_score: float
    center_habitat_xyz: Tuple[float, float, float]
    first_rgb: np.ndarray


SELECT_SYSTEM_PROMPT = (
    "You are an evidence auditor for embodied navigation. "
    "Rank candidate object views by instance-level evidence, but only allow an override when the non-baseline "
    "candidate satisfies decisive task constraints that Image 1 misses. Return strict JSON only."
)

SELECT_USER_TEMPLATE = """Navigation task:
{description}

ROLE:
You are verifying object candidates for an embodied navigation agent. The agent will navigate to the
selected object's 3D location, so choosing the wrong repeated instance is harmful.

INPUT:
There are exactly {k} candidate object images in detector logit order.
- Image 1 is the baseline top-1 candidate.
- Images 2..{k} are lower-ranked candidates.

TASK:
Answer three separate questions:
1. Constraint extraction: what target, attributes, anchors, and spatial relations matter?
2. Evidence audit: which candidate best matches the exact target INSTANCE?
3. Safety gating: is that candidate safe to use instead of Image 1?

ATTENTION:
1. First extract the discriminative constraints from the task: target category, attributes, nearby anchors,
   spatial relations, materials, colors, patterns, and room context.
2. A generic category match is not enough. For example, "a table with a lamp" is weaker than a table that also
   matches the described chair, picture, piano, wall, material, color, or relative placement.
3. best_index is the visually best candidate, even if it is not safe to override Image 1. Do not force
   best_index=1 just because the override is uncertain.
4. safe_to_override_image1 is a separate boolean. Set it true only when a non-1 candidate is exact_target,
   high confidence, and visibly satisfies important task constraints that Image 1 misses.
5. Repeated objects are common. For carpets, couches, armchairs, pictures, tables, bins, and plants, require
   specific anchors or spatial relations before setting safe_to_override_image1=true.
6. Use "exact_target" only when the target object and enough unique constraints are visible to distinguish this
   instance from similar objects. Use "partial_target" for correct category but missing key constraints. Use
   "context_only" when only room/anchor context or a generic category cue is visible. Use "not_target" when
   the image does not show the target category or useful target evidence.
7. selection_confidence describes the override decision, not the visual ranking. It may be "high" only when
   safe_to_override_image1=true is supported by strict instance-level evidence.

Return strict JSON with exact keys:
{{
  "best_index": <integer 1..{k}>,
  "match_type": "exact_target|partial_target|context_only|not_target",
  "selection_confidence": "high|medium|low",
  "image1_match_type": "exact_target|partial_target|context_only|not_target",
  "best_non1_index": <integer 2..{k}>,
  "best_non1_match_type": "exact_target|partial_target|context_only|not_target",
  "safe_to_override_image1": <true_or_false>,
  "switch_is_strictly_better_than_image1": <true_or_false>,
  "evidence_score_image1": <number 0.0..1.0>,
  "evidence_score_best": <number 0.0..1.0>,
  "strict_advantage_constraints": ["<constraint that the non-1 candidate satisfies better than Image 1>", "..."],
  "image1_missing_constraints": ["<constraint missing or weaker in Image 1>", "..."],
  "decisive_constraints": ["<visible constraint>", "..."],
  "missing_or_uncertain_constraints": ["<missing or uncertain constraint>", "..."],
  "task_constraints": {{
    "target": "<main target phrase>",
    "attributes": ["<attribute>", "..."],
    "anchors": ["<nearby or scene anchor>", "..."],
    "relations": ["<spatial relation>", "..."]
  }},
  "reason": "<brief English reason: name the visually best image, compare it to Image 1, and justify safe_to_override_image1>"
}}
"""

PANORAMA_SYSTEM_PROMPT = (
    "You are a conservative second-stage verifier for an embodied navigation candidate. "
    "You may veto a candidate if the panorama only shows a generic category or lacks instance-level evidence. "
    "Return strict JSON only."
)

PANORAMA_USER_TEMPLATE = """Navigation task:
{description}

ROLE:
You are verifying whether a selected candidate's 360-degree panorama supports overriding the baseline top-1 object.

INPUT:
This stitched panorama is the full 360-degree scan from the decision step associated with the selected candidate.

TASK:
Decide whether this panorama contains the exact target INSTANCE from the navigation task.

ATTENTION:
1. Set has_task_object=true only if the target object and enough discriminative task constraints are visible.
2. Do not accept a mere room label, a generic object category, or a common nearby anchor as sufficient.
3. For repeated objects such as carpets, couches, armchairs, pictures, tables, bins, and plants, require distinctive
   anchors or spatial relations from the task. If the instance cannot be distinguished, set has_task_object=false.
4. confidence may be "high" only when the panorama supports an exact instance match. Use "medium" or "low" when
   the target category appears but key instance constraints are missing or ambiguous.

Return strict JSON with exact keys:
{{
  "has_task_object": <true_or_false>,
  "instance_match": "exact_target|generic_category|insufficient_context",
  "confidence": "high|medium|low",
  "visible_constraints": ["<visible task constraint>", "..."],
  "missing_constraints": ["<missing or ambiguous task constraint>", "..."],
  "reason": "<brief English reason>"
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


def require_choice(parsed: Mapping[str, Any], key: str, allowed: Sequence[str]) -> str:
    value = parsed[key]
    if not isinstance(value, str) or value not in set(allowed):
        raise RuntimeError(f"unexpected {key}={value!r}, expected one of {sorted(allowed)!r}")
    return value


def require_bool(parsed: Mapping[str, Any], key: str) -> bool:
    value = parsed[key]
    if not isinstance(value, bool):
        raise RuntimeError(f"unexpected {key}={value!r}, expected JSON boolean")
    return bool(value)


def require_int_in_range(parsed: Mapping[str, Any], key: str, lo: int, hi: int) -> int:
    value = parsed[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < lo or value > hi:
        raise RuntimeError(f"unexpected {key}={value!r}, expected integer {lo}..{hi}")
    return int(value)


def require_float_in_range(parsed: Mapping[str, Any], key: str, lo: float, hi: float) -> float:
    value = parsed[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < lo or float(value) > hi:
        raise RuntimeError(f"unexpected {key}={value!r}, expected number {lo}..{hi}")
    return float(value)


def require_string_list(parsed: Mapping[str, Any], key: str) -> List[str]:
    value = parsed[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise RuntimeError(f"unexpected {key}={value!r}, expected list of strings")
    return [str(x) for x in value]


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
    cfg: VLMEvidenceConfig,
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
        max_tokens=512,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    best = require_int_in_range(parsed, "best_index", 1, len(candidates))
    best_non1 = require_int_in_range(parsed, "best_non1_index", 2, len(candidates))
    target_match_types = ("exact_target", "partial_target", "context_only", "not_target")
    match_type = require_choice(parsed, "match_type", target_match_types)
    image1_match_type = require_choice(parsed, "image1_match_type", target_match_types)
    best_non1_match_type = require_choice(parsed, "best_non1_match_type", target_match_types)
    selection_confidence = require_choice(parsed, "selection_confidence", ("high", "medium", "low"))
    safe_to_override_image1 = require_bool(parsed, "safe_to_override_image1")
    switch_is_strictly_better = require_bool(parsed, "switch_is_strictly_better_than_image1")
    evidence_score_image1 = require_float_in_range(parsed, "evidence_score_image1", 0.0, 1.0)
    evidence_score_best = require_float_in_range(parsed, "evidence_score_best", 0.0, 1.0)
    strict_advantage_constraints = require_string_list(parsed, "strict_advantage_constraints")
    image1_missing_constraints = require_string_list(parsed, "image1_missing_constraints")
    decisive_constraints = require_string_list(parsed, "decisive_constraints")
    missing_or_uncertain_constraints = require_string_list(parsed, "missing_or_uncertain_constraints")
    return {
        "raw": raw,
        "parsed": parsed,
        "best_index": best,
        "match_type": match_type,
        "selection_confidence": selection_confidence,
        "image1_match_type": image1_match_type,
        "best_non1_index": best_non1,
        "best_non1_match_type": best_non1_match_type,
        "safe_to_override_image1": safe_to_override_image1,
        "switch_is_strictly_better_than_image1": switch_is_strictly_better,
        "evidence_score_image1": evidence_score_image1,
        "evidence_score_best": evidence_score_best,
        "evidence_score_delta": float(evidence_score_best - evidence_score_image1),
        "strict_advantage_constraints": strict_advantage_constraints,
        "image1_missing_constraints": image1_missing_constraints,
        "decisive_constraints": decisive_constraints,
        "missing_or_uncertain_constraints": missing_or_uncertain_constraints,
        "task_constraints": parsed.get("task_constraints", {}),
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def call_vlm_verify_panorama(
    *,
    description: str,
    panorama_path: Path,
    cfg: VLMEvidenceConfig,
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
        max_tokens=384,
        temperature=0.0,
        timeout=int(cfg.vlm_timeout),
    )
    parsed = parse_json_object(raw)
    has_task_object = require_bool(parsed, "has_task_object")
    instance_match = require_choice(parsed, "instance_match", ("exact_target", "generic_category", "insufficient_context"))
    confidence = require_choice(parsed, "confidence", ("high", "medium", "low"))
    visible_constraints = require_string_list(parsed, "visible_constraints")
    missing_constraints = require_string_list(parsed, "missing_constraints")
    return {
        "raw": raw,
        "parsed": parsed,
        "has_task_object": has_task_object,
        "instance_match": instance_match,
        "confidence": confidence,
        "visible_constraints": visible_constraints,
        "missing_constraints": missing_constraints,
        "reason": str(parsed.get("reason", "")),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def correct_final_decision_with_vlmevidence(
    *,
    description: str,
    rep: Any,
    decision_aux: Mapping[str, Any],
    stage2_json_path: Path,
    baseline_target_xyz: np.ndarray,
    panorama_frames_by_slot: Mapping[int, Sequence[np.ndarray]],
    output_dir: Path,
    cfg: VLMEvidenceConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not bool(decision_aux["is_object_decision"]):
        raise RuntimeError("vlmevidence was called for a non-object decision")
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
    candidate_switch_requested = int(selection["best_index"]) != 1
    selection_gate_passed = bool(
        candidate_switch_requested
        and selection["match_type"] == "exact_target"
        and selection["selection_confidence"] == "high"
        and selection["safe_to_override_image1"]
        and selection["switch_is_strictly_better_than_image1"]
        and float(selection["evidence_score_delta"]) >= float(cfg.evidence_delta)
        and len(selection["strict_advantage_constraints"]) > 0
    )
    panorama_verify = None
    selected_panorama_path = None
    panorama_vetoed_correction = False
    panorama_gate_passed = None

    if selection_gate_passed:
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
        panorama_gate_passed = bool(
            panorama_verify["has_task_object"]
            and panorama_verify["instance_match"] == "exact_target"
            and panorama_verify["confidence"] == "high"
        )
        panorama_vetoed_correction = not bool(panorama_gate_passed)

    use_selected = bool(selection_gate_passed and panorama_gate_passed)
    correction_applied = bool(use_selected)
    if use_selected:
        correction_rejected_reason = "accepted"
    elif not candidate_switch_requested:
        correction_rejected_reason = "kept_baseline_best_index_1"
    elif not selection_gate_passed:
        correction_rejected_reason = (
            "selection_gate_failed:"
            f"match_type={selection['match_type']},"
            f"confidence={selection['selection_confidence']},"
            f"safe_to_override={selection['safe_to_override_image1']},"
            f"strictly_better={selection['switch_is_strictly_better_than_image1']},"
            f"evidence_delta={selection['evidence_score_delta']:.3f},"
            f"strict_advantage_count={len(selection['strict_advantage_constraints'])}"
        )
    else:
        correction_rejected_reason = (
            "panorama_gate_failed:"
            f"has_task_object={panorama_verify['has_task_object']},"
            f"instance_match={panorama_verify['instance_match']},"
            f"confidence={panorama_verify['confidence']}"
        )
    corrected_target = np.asarray(selected.center_habitat_xyz, dtype=float).reshape(3) if use_selected else np.asarray(
        baseline_target_xyz, dtype=float
    ).reshape(3)
    info = {
        "vlmevidence_called": True,
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
        "candidate_switch_requested": bool(candidate_switch_requested),
        "selection_gate_passed": bool(selection_gate_passed),
        "panorama_gate_passed": None if panorama_gate_passed is None else bool(panorama_gate_passed),
        "correction_rejected_reason": correction_rejected_reason,
        "correction_applied": bool(correction_applied),
        "panorama_vetoed_correction": bool(panorama_vetoed_correction),
        "selected_target_used": bool(use_selected),
        "panorama_verify": panorama_verify,
        "selected_panorama_path": str(selected_panorama_path) if selected_panorama_path is not None else None,
        "corrected_target_xyz": [float(x) for x in corrected_target.tolist()],
    }
    with open(output_dir / "vlmevidence_decision.json", "w", encoding="utf-8") as f:
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

