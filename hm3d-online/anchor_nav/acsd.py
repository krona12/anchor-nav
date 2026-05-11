from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vlm.client import chat


def _normalize_phrase(text: Any) -> str:
    out = str(text or "").strip().lower()
    out = re.sub(r"[^a-z0-9\s\-]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _as_np3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise RuntimeError(f"{name} must contain at least 3 values, got shape={arr.shape}")
    out = arr[:3].astype(float)
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"{name} contains non-finite values: {out!r}")
    return out


def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    quote = ""
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object_strict(raw: Any, *, source: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError(f"{source} returned empty response")
    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    balanced = _extract_balanced_json(text)
    if balanced:
        candidates.append(balanced)
    if text.startswith("{"):
        candidates.append(text)
    errors: List[str] = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(obj, dict):
            raise RuntimeError(f"{source} JSON is not an object: {type(obj).__name__}")
        return obj
    raise RuntimeError(f"{source} JSON parse failed; errors={errors}; raw={text[:1000]!r}")


def _require_str(obj: Mapping[str, Any], key: str) -> str:
    if key not in obj:
        raise KeyError(f"missing required string field: {key}")
    value = str(obj[key]).strip()
    return value


def _require_float(obj: Mapping[str, Any], key: str, *, lo: float = -math.inf, hi: float = math.inf) -> float:
    if key not in obj:
        raise KeyError(f"missing required float field: {key}")
    value = float(obj[key])
    if not math.isfinite(value) or value < lo or value > hi:
        raise RuntimeError(f"{key} must be finite in [{lo},{hi}], got {value!r}")
    return value


def _require_bool(obj: Mapping[str, Any], key: str) -> bool:
    if key not in obj:
        raise KeyError(f"missing required bool field: {key}")
    value = obj[key]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("true", "yes", "1"):
            return True
        if val in ("false", "no", "0"):
            return False
    raise RuntimeError(f"{key} must be bool-like, got {value!r}")


def _require_str_list(obj: Mapping[str, Any], key: str) -> List[str]:
    if key not in obj:
        raise KeyError(f"missing required list field: {key}")
    raw = obj[key]
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuntimeError(f"{key} must be a list, got {type(raw).__name__}")
    out: List[str] = []
    for item in raw:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _target_hint_from_instruction(instruction: str) -> str:
    text = _normalize_phrase(instruction)
    text = re.sub(r"^(find|go to|navigate to|look for|the|a|an)\s+", "", text).strip()
    if not text:
        return ""
    rel_alt = (
        " next to ",
        " in front of ",
        " on top of ",
        " beside ",
        " between ",
        " against ",
        " near ",
        " above ",
        " below ",
        " under ",
        " inside ",
        " behind ",
        " with ",
        " on ",
        " in ",
    )
    cut = len(text)
    for token in rel_alt:
        pos = text.find(token)
        if pos >= 0:
            cut = min(cut, pos)
    hint = text[:cut].strip()
    if len(hint.split()) > 8:
        return ""
    return hint


@dataclass
class ACSDConfig:
    vlm_model: str = "gpt-4o-mini"
    object_top_k: int = 4
    max_object_anchors: int = 3
    vlm_max_retries: int = 3
    vlm_retry_sleep_sec: float = 3.0
    correction_margin: float = 0.015
    baseline_weight: float = 0.42
    target_weight: float = 0.18
    anchor_weight: float = 0.22
    object_correction_min_baseline_score: float = 0.90
    adaptive_level_policy: bool = True


class AnchorConditionedSoftDecomposition:
    def __init__(self, cfg: Optional[ACSDConfig] = None) -> None:
        self.cfg = cfg or ACSDConfig()
        self.summary = {
            "case_10": 0,
            "case_01": 0,
            "case_11": 0,
            "case_00": 0,
            "total_compared": 0,
            "correction_applied": 0,
            "correction_rejected": 0,
        }

    def _chat_strict(self, *, text: str, image_path: Any, max_tokens: int, source: str) -> str:
        attempts = max(1, int(self.cfg.vlm_max_retries))
        errors: List[str] = []
        for attempt in range(1, attempts + 1):
            try:
                return chat(text=text, image_path=image_path, model=self.cfg.vlm_model, max_tokens=max_tokens)
            except Exception as exc:
                errors.append(f"attempt={attempt}/{attempts} {type(exc).__name__}: {exc}")
                if attempt < attempts:
                    time.sleep(float(self.cfg.vlm_retry_sleep_sec))
        raise RuntimeError(f"{source} VLM call failed after {attempts} attempts: {' | '.join(errors)}")

    def decompose_instruction(self, instruction: str) -> Dict[str, Any]:
        text = str(instruction or "").strip()
        if not text:
            text = ""
        prompt = (
            "You are the ACSD anchor extractor for indoor navigation.\n\n"
            "Task:\n"
            "Extract only supporting object anchors and room/environment context from one navigation instruction.\n"
            "Return strict JSON only. Do not include markdown, comments, explanations, or extra keys.\n\n"
            "Definitions:\n"
            "- object_anchors: nearby/supporting physical objects that help disambiguate the target location.\n"
            "- room_anchor: the room, region, or environment context, if explicitly stated or strongly implied.\n"
            "- Infer the target internally so you can exclude it, but do not output the target.\n\n"
            "Rules:\n"
            "1. object_anchors must contain only physical supporting objects/furniture/fixtures, not the target itself.\n"
            "2. Do not put room/environment words in object_anchors.\n"
            "3. Prefer specific object nouns over long descriptions. Each object_anchor must be 1-5 words.\n"
            "4. Exclude generic environment words from object_anchors: room, area, space, place, environment, "
            "corner, side, wall, floor, ceiling, doorway, entrance, hallway, corridor.\n"
            "5. Use lowercase normalized English phrases.\n"
            "6. If no supporting object anchor exists, return [] for object_anchors.\n"
            "7. If no room/environment context exists, return \"\" for room_anchor.\n"
            "8. If the instruction is only a target object phrase, object_anchors must be [].\n"
            "9. Do not guess hidden objects that are not stated or strongly implied.\n\n"
            "Return exactly this JSON object:\n"
            "{\n"
            "  \"room_anchor\": string,\n"
            "  \"object_anchors\": [string]\n"
            "}\n\n"
            f"Instruction: {text}"
        )
        try:
            raw = self._chat_strict(text=prompt, image_path=None, max_tokens=256, source="ACSD anchor extractor")
            parsed = parse_json_object_strict(raw, source="ACSD anchor extractor")
            required = {"room_anchor", "object_anchors"}
            got = set(parsed.keys())
            if got != required:
                raise RuntimeError(f"ACSD anchor extractor schema mismatch; missing={sorted(required - got)} extra={sorted(got - required)}")
            raw_anchors = [_normalize_phrase(x) for x in _require_str_list(parsed, "object_anchors") if _normalize_phrase(x)]
            room_anchor = _normalize_phrase(parsed.get("room_anchor", ""))
            source = "vlm_anchor_extractor"
            error = ""
        except Exception as exc:
            raw = ""
            raw_anchors = []
            room_anchor = ""
            source = "vlm_anchor_extractor_failed"
            error = f"{type(exc).__name__}: {exc}"
        out = {
            "full_instruction": text,
            "room_anchor": room_anchor,
            "object_anchors": [],
            "anchor_prompt": "",
            "raw": str(raw),
            "decompose_source": source,
            "decompose_error": error,
            "anchor_policy": "vlm_extracted_soft_topk_not_all_required",
            "acsd_fallback_to_baseline": bool(error),
        }
        out["object_anchors"] = self._active_object_anchors(
            {
                "target_object": _target_hint_from_instruction(text),
                "room_anchor": out["room_anchor"],
                "object_anchors": raw_anchors,
            }
        )
        anchor_parts = [out["room_anchor"]] if out["room_anchor"] else []
        anchor_parts.extend(out["object_anchors"])
        out["anchor_prompt"] = ", ".join(anchor_parts)
        return out

    def load_stage2_decision(self, stage2_json_path: Path) -> Dict[str, Any]:
        path = Path(stage2_json_path)
        if not path.is_file():
            raise FileNotFoundError(f"ACSD requires stage2_decision.json, missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(f"stage2 decision JSON is not object: {path}")
        chosen = data.get("chosen")
        if not isinstance(chosen, dict):
            raise RuntimeError(f"stage2 decision missing chosen: {path}")
        branch = str(chosen.get("branch", "")).strip()
        if branch not in ("object", "frontier"):
            raise RuntimeError(f"stage2 chosen.branch invalid: {branch!r}")
        for key in ("object_candidates", "frontier_candidates"):
            if key not in data or not isinstance(data[key], list):
                raise RuntimeError(f"stage2 decision missing list field {key}: {path}")
        return data

    def build_baseline_decision(
        self,
        *,
        target_position: Sequence[float],
        is_object_decision: bool,
        decision_aux: Mapping[str, Any],
        stage2: Mapping[str, Any],
    ) -> Dict[str, Any]:
        decision_type = "object" if bool(is_object_decision) else "frontier"
        chosen = stage2.get("chosen")
        if not isinstance(chosen, Mapping):
            raise RuntimeError("stage2 missing chosen for baseline decision")
        if str(chosen.get("branch")) != decision_type:
            raise RuntimeError(
                f"baseline decision type mismatch: returned={decision_type}, stage2={chosen.get('branch')!r}"
            )
        score = None
        if decision_type == "object":
            idx = int(decision_aux["real_object_decision_idx"])
            for rec in stage2["object_candidates"]:
                if int(rec["slot_index"]) == idx:
                    score = float(rec["og3d_logit"])
                    break
            if score is None:
                raise RuntimeError(f"baseline object slot {idx} missing from stage2 object_candidates")
        else:
            idx = int(chosen["frontier_argmax_index"])
            for rec in stage2["frontier_candidates"]:
                if int(rec["frontier_index"]) == idx:
                    score = float(rec["og3d_logit"])
                    break
            if score is None:
                raise RuntimeError(f"baseline frontier index {idx} missing from stage2 frontier_candidates")
        return {
            "type": decision_type,
            "position": _as_np3(target_position, name="baseline target").tolist(),
            "score": float(score),
            "decision_aux": dict(decision_aux),
        }

    def _object_candidates_from_stage2(
        self,
        *,
        stage2: Mapping[str, Any],
        rep: Any,
        top_k: int,
        require_images: bool,
    ) -> List[Dict[str, Any]]:
        objs = stage2.get("object_candidates")
        if not isinstance(objs, list) or len(objs) == 0:
            raise RuntimeError("ACSD object reranker requires non-empty stage2 object_candidates")
        ranked = sorted(objs, key=lambda x: float(x["og3d_logit"]), reverse=True)
        k = min(int(top_k), len(ranked))
        if k <= 0:
            raise RuntimeError(f"ACSD object_top_k must be positive, got {top_k}")
        rgb_list = getattr(rep, "object_first_rgb", None)
        if require_images and rgb_list is None:
            raise RuntimeError("representation_manager.object_first_rgb is missing")
        out: List[Dict[str, Any]] = []
        for rank, rec in enumerate(ranked[:k], start=1):
            slot = int(rec["slot_index"])
            center = _as_np3(rec["center_habitat_xyz"], name=f"object candidate {slot} center")
            item = {
                "rank": int(rank),
                "slot_index": slot,
                "center_habitat_xyz": center.tolist(),
                "og3d_logit": float(rec["og3d_logit"]),
                "merged_object_score": float(rec["merged_object_score"]),
            }
            if require_images:
                if slot < 0 or slot >= len(rgb_list):
                    raise RuntimeError(f"object_first_rgb slot {slot} out of range length={len(rgb_list)}")
                rgb = rgb_list[slot]
                if rgb is None:
                    raise RuntimeError(f"object_first_rgb for slot {slot} is None")
                arr = np.asarray(rgb)
                if arr.ndim != 3 or arr.shape[2] < 3:
                    raise RuntimeError(f"object_first_rgb slot {slot} invalid shape={arr.shape}")
                item["first_rgb"] = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
            out.append(item)
        return out

    def _active_object_anchors(self, decomposition: Mapping[str, Any]) -> List[str]:
        raw = decomposition.get("object_anchors", [])
        if not isinstance(raw, list):
            raise RuntimeError("ACSD decomposition object_anchors must be a list")
        target = _normalize_phrase(decomposition.get("target_object", ""))
        room_anchor = _normalize_phrase(decomposition.get("room_anchor", ""))
        generic = {
            "room",
            "area",
            "space",
            "place",
            "environment",
            "location",
            "region",
            "corner",
            "side",
            "wall",
            "floor",
            "ceiling",
            "doorway",
            "entrance",
            "hallway",
            "corridor",
            "upstairs",
            "downstairs",
        }
        anchors: List[str] = []
        for item in raw:
            text = _normalize_phrase(item)
            if not text:
                continue
            words = text.split()
            if len(words) > 5:
                continue
            if text in generic:
                continue
            if room_anchor and text == room_anchor:
                continue
            if target:
                target_words = target.split()
                anchor_words = text.split()
                if text == target or text in target or target in text:
                    continue
                if anchor_words and target_words and anchor_words[-1] == target_words[-1]:
                    overlap = len(set(anchor_words) & set(target_words))
                    if overlap >= max(1, len(anchor_words) - 1):
                        continue
            if text not in anchors:
                anchors.append(text)
            if len(anchors) >= int(self.cfg.max_object_anchors):
                break
        return anchors

    @staticmethod
    def _weighted_normalized_score(parts: Sequence[Tuple[float, float]]) -> float:
        total_weight = float(sum(float(w) for w, _ in parts))
        if total_weight <= 0:
            raise RuntimeError("ACSD score normalization received zero total weight")
        score = sum(float(w) * float(v) for w, v in parts) / total_weight
        if not np.isfinite(score):
            raise RuntimeError(f"ACSD weighted score is non-finite: {score!r}")
        return float(max(0.0, min(1.0, score)))

    def _level_policy(self, task_level: Any) -> Dict[str, Any]:
        lvl = _normalize_phrase(task_level)
        policy: Dict[str, Any] = {
            "task_level": lvl,
            "margin": float(self.cfg.correction_margin),
            "min_candidate_baseline_score": float(self.cfg.object_correction_min_baseline_score),
            "baseline_weight": float(self.cfg.baseline_weight),
            "target_weight": float(self.cfg.target_weight),
            "anchor_weight": float(self.cfg.anchor_weight),
            "min_target_advantage": 0.08,
            "min_anchor_advantage": 0.30,
            "min_target_advantage_when_anchor_only": 0.08,
            "require_room_or_strong_target": False,
            "reason": "global_default",
        }
        if not bool(self.cfg.adaptive_level_policy):
            return policy
        if lvl == "object":
            policy.update(
                {
                    "margin": min(float(self.cfg.correction_margin), 0.002),
                    "min_candidate_baseline_score": min(float(self.cfg.object_correction_min_baseline_score), 0.70),
                    "baseline_weight": 0.38,
                    "target_weight": 0.30,
                    "anchor_weight": 0.24,
                    "min_target_advantage": 0.06,
                    "min_anchor_advantage": 0.28,
                    "min_target_advantage_when_anchor_only": 0.08,
                    "reason": "object_permissive_target_anchor",
                }
            )
        elif lvl == "region":
            policy.update(
                {
                    "margin": min(float(self.cfg.correction_margin), 0.003),
                    "min_candidate_baseline_score": min(float(self.cfg.object_correction_min_baseline_score), 0.68),
                    "baseline_weight": 0.30,
                    "target_weight": 0.34,
                    "anchor_weight": 0.30,
                    "min_target_advantage": 0.45,
                    "min_anchor_advantage": 0.60,
                    "min_target_advantage_when_anchor_only": 0.18,
                    "reason": "region_strong_target_or_context_rescue",
                }
            )
        elif lvl == "room":
            policy.update(
                {
                    "margin": max(float(self.cfg.correction_margin), 0.008),
                    "min_candidate_baseline_score": max(float(self.cfg.object_correction_min_baseline_score), 0.82),
                    "baseline_weight": 0.48,
                    "target_weight": 0.30,
                    "anchor_weight": 0.14,
                    "min_target_advantage": 0.12,
                    "min_anchor_advantage": 0.35,
                    "min_target_advantage_when_anchor_only": 0.12,
                    "reason": "room_conservative_context_safety",
                }
            )
        elif lvl == "instance":
            policy.update(
                {
                    "margin": max(float(self.cfg.correction_margin), 0.010),
                    "min_candidate_baseline_score": max(float(self.cfg.object_correction_min_baseline_score), 0.96),
                    "baseline_weight": 0.54,
                    "target_weight": 0.32,
                    "anchor_weight": 0.08,
                    "min_target_advantage": 0.18,
                    "min_anchor_advantage": 0.45,
                    "min_target_advantage_when_anchor_only": 0.18,
                    "require_room_or_strong_target": True,
                    "reason": "instance_conservative_identity_safety",
                }
            )
        return policy

    def _score_candidates_with_anchors(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        decomposition: Mapping[str, Any],
        task_level: Any = "",
    ) -> Dict[int, Dict[str, Any]]:
        if len(candidates) == 0:
            raise RuntimeError("ACSD anchor scorer requires non-empty candidates")
        base_scores = self._minmax_scores([float(c["og3d_logit"]) for c in candidates])
        merged_scores = self._minmax_scores([float(c["merged_object_score"]) for c in candidates])
        centers = [_as_np3(c["center_habitat_xyz"], name=f"object candidate {c.get('slot_index')} center") for c in candidates]
        active_anchors = self._active_object_anchors(decomposition)
        room_active = bool(str(decomposition.get("room_anchor", "")).strip())
        lvl = _normalize_phrase(task_level)
        # A room name alone is too coarse for room-level final decisions
        # (e.g. "blanket in the bedroom" can point to many plausible objects).
        # Let region tasks use room context as a weak spatial prior, but require
        # explicit supporting objects for room and instance corrections.
        room_context_eligible = bool(room_active and lvl == "region")
        correction_eligible = bool(active_anchors or room_context_eligible)
        constraint_active = bool(correction_eligible or room_active)
        if len(candidates) < 2 and active_anchors:
            raise RuntimeError("ACSD anchor scoring requires at least two object candidates when anchors are active")
        raw_anchor_values: List[float] = []
        for idx, _ in enumerate(candidates):
            proximity_values: List[float] = []
            density_num = 0.0
            density_den = 0.0
            for j, other in enumerate(centers):
                if j == idx:
                    continue
                dist = float(np.linalg.norm(centers[idx][[0, 2]] - other[[0, 2]]))
                kernel = math.exp(-dist / 2.5)
                density_num += float(merged_scores[j]) * kernel
                density_den += kernel
                proximity_values.append(float(merged_scores[j]) / (1.0 + dist))
            if proximity_values:
                nearest_context = float(max(proximity_values))
                density_context = float(density_num / max(density_den, 1e-12))
                raw_anchor_values.append(float(0.65 * density_context + 0.35 * nearest_context))
            elif active_anchors:
                raise RuntimeError("ACSD anchor scoring had no neighbor object candidates")
            else:
                raw_anchor_values.append(0.5)
        anchor_scores = (
            self._minmax_scores(raw_anchor_values)
            if correction_eligible
            else [0.5 for _ in candidates]
        )
        by_rank: Dict[int, Dict[str, Any]] = {}
        for idx, cand in enumerate(candidates):
            target_match = float(merged_scores[idx])
            rank = int(cand["rank"])
            by_rank[rank] = {
                "target_match_score": float(max(0.0, min(1.0, target_match))),
                "anchor_match_score": float(max(0.0, min(1.0, anchor_scores[idx]))),
                "active_object_anchors": list(active_anchors),
                "room_context_eligible": bool(room_context_eligible),
                "constraint_active": bool(constraint_active),
                "correction_eligible": bool(correction_eligible),
                "target_match_source": "merged_object_score_semantic_labels_unavailable",
                "reason": (
                    "object_only_acsd_origin_anchor_rerank; "
                    f"task_level={lvl}; constraint_active={constraint_active}; "
                    f"correction_eligible={correction_eligible}; room_context_eligible={room_context_eligible}; "
                    f"active_object_anchors={active_anchors}; "
                    "anchor_policy=soft_topk_not_all_required; score_normalization=minmax_neutral_0.5; "
                    "vlm_decomposition_only"
                ),
            }
        return by_rank

    @staticmethod
    def _minmax_scores(values: Sequence[float]) -> List[float]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] == 0:
            raise RuntimeError("cannot normalize empty score list")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"non-finite scores for normalization: {arr!r}")
        lo = float(arr.min())
        hi = float(arr.max())
        if abs(hi - lo) < 1e-9:
            return [0.5 for _ in arr]
        return [float((v - lo) / (hi - lo)) for v in arr]

    def rerank_object_candidates(
        self,
        *,
        baseline_decision: Mapping[str, Any],
        stage2: Mapping[str, Any],
        rep: Any,
        decomposition: Mapping[str, Any],
        output_dir: Path,
        task_level: Any = "",
    ) -> Dict[str, Any]:
        if str(baseline_decision["type"]) != "object":
            raise RuntimeError("rerank_object_candidates called for non-object baseline decision")
        candidates = self._object_candidates_from_stage2(
            stage2=stage2,
            rep=rep,
            top_k=int(self.cfg.object_top_k),
            require_images=False,
        )
        anchor_scores = self._score_candidates_with_anchors(
            candidates=candidates,
            decomposition=decomposition,
            task_level=task_level,
        )
        base_scores = self._minmax_scores([float(c["og3d_logit"]) for c in candidates])
        policy = self._level_policy(task_level)
        ranked: List[Dict[str, Any]] = []
        for cand, base in zip(candidates, base_scores):
            extra = anchor_scores[int(cand["rank"])]
            parts: List[Tuple[float, float]] = [
                (float(policy["baseline_weight"]), float(base)),
                (float(policy["target_weight"]), float(extra["target_match_score"])),
            ]
            if bool(extra.get("correction_eligible", False)):
                parts.extend(
                    [
                        (float(policy["anchor_weight"]), float(extra["anchor_match_score"])),
                    ]
                )
            final = self._weighted_normalized_score(parts)
            item = dict(cand)
            item.update(extra)
            item["baseline_object_score"] = float(base)
            item["acsd_score"] = float(final)
            item["level_policy"] = dict(policy)
            ranked.append(item)
        ranked.sort(key=lambda x: float(x["acsd_score"]), reverse=True)
        baseline_slot = int(baseline_decision["decision_aux"]["real_object_decision_idx"])
        baseline_rank = next((x for x in ranked if int(x["slot_index"]) == baseline_slot), None)
        if baseline_rank is None:
            raise RuntimeError(f"baseline object slot {baseline_slot} not present after reranking")
        best = ranked[0]
        level_name = str(policy.get("task_level", ""))
        room_anchor = str(decomposition.get("room_anchor", "")).strip()
        best_baseline_distance = float(
            np.linalg.norm(
                _as_np3(best["center_habitat_xyz"], name=f"best object candidate {best.get('slot_index')} center")
                - _as_np3(
                    baseline_rank["center_habitat_xyz"],
                    name=f"baseline object candidate {baseline_slot} center",
                )
            )
        )
        room_local_target_override_gate = False
        if False and level_name == "room" and room_anchor and not bool(baseline_rank.get("correction_eligible", False)):
            baseline_center = _as_np3(
                baseline_rank["center_habitat_xyz"],
                name=f"baseline object candidate {baseline_slot} center",
            )
            local_target_candidates: List[Tuple[float, float, float, Dict[str, Any]]] = []
            baseline_target = float(baseline_rank["target_match_score"])
            baseline_acsd = float(baseline_rank["acsd_score"])
            for cand in ranked:
                if int(cand["slot_index"]) == baseline_slot:
                    continue
                dist = float(
                    np.linalg.norm(
                        _as_np3(cand["center_habitat_xyz"], name=f"object candidate {cand.get('slot_index')} center")
                        - baseline_center
                    )
                )
                strong_target_local = bool(
                    dist <= 0.85
                    and float(cand["baseline_object_score"]) >= 0.70
                    and float(cand["target_match_score"]) >= 0.90
                    and baseline_target <= 0.40
                    and float(cand["target_match_score"]) - baseline_target >= 0.45
                    and float(cand["acsd_score"]) >= baseline_acsd + 0.02
                )
                high_conf_local = bool(
                    dist <= 0.85
                    and float(cand["baseline_object_score"]) >= 0.90
                    and float(cand["target_match_score"]) >= 0.96
                    and baseline_target <= 0.60
                    and float(cand["target_match_score"]) - baseline_target >= 0.40
                    and float(cand["acsd_score"]) >= baseline_acsd + 0.08
                )
                if strong_target_local or high_conf_local:
                    local_target_candidates.append(
                        (
                            float(cand["target_match_score"]),
                            float(cand["acsd_score"]),
                            -dist,
                            cand,
                        )
                    )
            if local_target_candidates:
                local_target_candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
                best = local_target_candidates[0][3]
                room_local_target_override_gate = True
        target_advantage = float(best["target_match_score"]) - float(baseline_rank["target_match_score"])
        anchor_advantage = float(best["anchor_match_score"]) - float(baseline_rank["anchor_match_score"])
        semantic_gate = bool(
            target_advantage >= float(policy["min_target_advantage"])
            or (
                anchor_advantage >= float(policy["min_anchor_advantage"])
                and target_advantage >= float(policy["min_target_advantage_when_anchor_only"])
            )
        )
        room_or_identity_gate = True
        if bool(policy.get("require_room_or_strong_target", False)):
            room_or_identity_gate = bool(room_anchor) or target_advantage >= 0.65
        candidate_confidence_gate = float(best["baseline_object_score"]) >= float(policy["min_candidate_baseline_score"])
        region_context_override_gate = bool(
            level_name == "region"
            and float(best["baseline_object_score"]) >= 0.45
            and float(best["target_match_score"]) >= 0.50
            and float(best["anchor_match_score"]) >= 0.65
            and float(baseline_rank["baseline_object_score"]) < 0.95
            and float(baseline_rank["target_match_score"]) < 0.95
            and float(baseline_rank["anchor_match_score"]) <= 0.12
            and target_advantage <= -0.10
            and anchor_advantage >= 0.55
        )
        region_target_anchor_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.35
            and float(best["target_match_score"]) >= 0.88
            and float(best["anchor_match_score"]) >= 0.60
            and float(baseline_rank["target_match_score"]) <= 0.90
            and float(baseline_rank["baseline_object_score"]) < 0.98
            and (
                target_advantage >= 0.45
                or (target_advantage >= 0.18 and anchor_advantage >= 0.60)
            )
        )
        region_context_localization_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.45
            and float(best["target_match_score"]) >= 0.85
            and float(best["anchor_match_score"]) >= 0.85
            and float(baseline_rank["target_match_score"]) >= 0.80
            and target_advantage >= 0.05
            and anchor_advantage >= 0.60
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.05
        )
        region_high_conf_target_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.95
            and float(best["target_match_score"]) >= 0.85
            and float(baseline_rank["target_match_score"]) <= 0.90
            and target_advantage >= 0.18
            and (
                float(best["anchor_match_score"]) >= 0.60
                or anchor_advantage >= -0.20
                or best_baseline_distance <= 0.85
            )
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.04
        )
        region_exact_target_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.90
            and float(best["target_match_score"]) >= 0.98
            and float(baseline_rank["target_match_score"]) <= 0.05
            and target_advantage >= 0.95
            and (
                float(best["anchor_match_score"]) >= 0.60
                or anchor_advantage >= -0.20
                or best_baseline_distance <= 0.85
            )
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.15
        )
        region_low_raw_exact_target_rescue_gate = bool(
            False
            and
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and 0.35 <= float(best["baseline_object_score"]) < 0.90
            and float(best["target_match_score"]) >= 0.96
            and float(baseline_rank["target_match_score"]) <= 0.05
            and target_advantage >= 0.90
            and float(best["anchor_match_score"]) >= 0.60
            and anchor_advantage >= -0.15
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.08
        )
        region_mid_conf_target_context_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.80
            and float(best["target_match_score"]) >= 0.95
            and float(baseline_rank["target_match_score"]) <= 0.20
            and target_advantage >= 0.75
            and float(best["anchor_match_score"]) >= 0.80
            and anchor_advantage >= 0.0
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.15
        )
        region_lower_conf_target_context_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and 0.62 <= float(best["baseline_object_score"]) < 0.80
            and float(best["target_match_score"]) >= 0.92
            and float(baseline_rank["target_match_score"]) <= 0.10
            and target_advantage >= 0.80
            and float(best["anchor_match_score"]) >= 0.72
            and anchor_advantage >= 0.10
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.20
        )
        region_low_conf_target_anchor_rescue_gate = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and 0.55 <= float(best["baseline_object_score"]) < 0.68
            and float(best["target_match_score"]) >= 0.98
            and float(baseline_rank["target_match_score"]) >= 0.70
            and 0.12 <= target_advantage <= 0.35
            and float(best["anchor_match_score"]) >= 0.55
            and float(baseline_rank["anchor_match_score"]) <= 0.10
            and anchor_advantage >= 0.55
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.10
        )
        region_target_context_advantage_gate = bool(
            level_name == "region"
            and (
                (
                    target_advantage >= 0.45
                    and (
                        float(best["anchor_match_score"]) >= 0.60
                        or anchor_advantage >= -0.20
                        or best_baseline_distance <= 0.85
                    )
                )
                or (target_advantage >= 0.18 and anchor_advantage >= 0.60)
            )
        )
        region_anchor_regression_nonlocal_block = bool(
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and target_advantage >= 0.45
            and float(best["anchor_match_score"]) < 0.60
            and anchor_advantage < -0.20
            and best_baseline_distance > 0.85
        )
        # Human navigation often accepts a lower raw PQ3D logit for region-level
        # targets when the candidate is strongly supported by both the target
        # noun and the surrounding context cluster. Keep this escape hatch
        # narrow, otherwise room/instance corrections become noisy.
        if (
            level_name == "region"
            and float(best["baseline_object_score"]) >= 0.08
            and region_target_context_advantage_gate
        ):
            candidate_confidence_gate = True
        if (
            region_context_override_gate
            or region_target_anchor_rescue_gate
            or region_context_localization_rescue_gate
            or region_high_conf_target_rescue_gate
            or region_exact_target_rescue_gate
            or region_low_raw_exact_target_rescue_gate
            or region_mid_conf_target_context_rescue_gate
            or region_lower_conf_target_context_rescue_gate
            or region_low_conf_target_anchor_rescue_gate
        ):
            semantic_gate = True
            candidate_confidence_gate = True
        if (
            level_name == "instance"
            and float(best["baseline_object_score"]) >= 0.80
            and target_advantage >= 0.55
            and anchor_advantage >= -0.40
        ):
            candidate_confidence_gate = True
        instance_target_context_rescue_gate = bool(
            level_name == "instance"
            and bool(room_anchor)
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.58
            and float(best["target_match_score"]) >= 0.86
            and float(baseline_rank["target_match_score"]) <= 0.55
            and target_advantage >= 0.35
            and anchor_advantage >= -0.10
        )
        instance_anchor_identity_rescue_gate = bool(
            level_name == "instance"
            and bool(room_anchor)
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.88
            and float(best["target_match_score"]) >= 0.86
            and float(baseline_rank["target_match_score"]) >= 0.65
            and target_advantage >= 0.18
            and anchor_advantage >= 0.45
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.015
        )
        if instance_target_context_rescue_gate or instance_anchor_identity_rescue_gate:
            semantic_gate = True
            candidate_confidence_gate = True
        room_target_override_gate = bool(
            False
            and
            level_name == "room"
            and bool(room_anchor)
            and not bool(baseline_rank.get("correction_eligible", False))
            and float(best["baseline_object_score"]) >= 0.90
            and float(baseline_rank["baseline_object_score"]) < 0.95
            and float(best["target_match_score"]) >= 0.90
            and float(baseline_rank["target_match_score"]) <= 0.40
            and target_advantage >= 0.45
        )
        room_exact_target_rescue_gate = bool(
            False
            and
            level_name == "room"
            and bool(room_anchor)
            and int(best["slot_index"]) != baseline_slot
            and not bool(baseline_rank.get("correction_eligible", False))
            and float(best["baseline_object_score"]) >= 0.94
            and float(best["target_match_score"]) >= 0.98
            and float(baseline_rank["target_match_score"]) <= 0.05
            and target_advantage >= 0.95
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.20
        )
        region_anchor_substitute_rescue_gate = bool(
            False
            and
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= 0.50
            and float(best["target_match_score"]) >= 0.88
            and float(baseline_rank["target_match_score"]) >= 0.90
            and target_advantage >= -0.10
            and float(best["anchor_match_score"]) >= 0.95
            and anchor_advantage >= 0.80
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.06
        )
        region_context_probe_rescue_gate = bool(
            False
            and
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and 0.90 <= float(best["baseline_object_score"]) <= 0.93
            and 0.70 <= float(best["anchor_match_score"]) <= 0.85
            and float(best["target_match_score"]) >= 0.80
            and -0.20 <= target_advantage <= -0.12
            and 0.35 <= anchor_advantage <= 0.60
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + 0.03
        )
        if room_target_override_gate or room_local_target_override_gate or room_exact_target_rescue_gate:
            semantic_gate = True
            candidate_confidence_gate = True
        if region_anchor_substitute_rescue_gate or region_context_probe_rescue_gate:
            semantic_gate = True
            candidate_confidence_gate = True
        if (
            level_name == "region"
            and int(best["slot_index"]) != baseline_slot
            and anchor_advantage < float(policy["min_anchor_advantage"])
            and float(best["baseline_object_score"]) < 0.95
            and not region_low_raw_exact_target_rescue_gate
            and not region_mid_conf_target_context_rescue_gate
            and not region_lower_conf_target_context_rescue_gate
            and not region_low_conf_target_anchor_rescue_gate
        ):
            candidate_confidence_gate = False
        margin_gate = float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + float(policy["margin"])
        if (
            level_name == "region"
            and region_target_context_advantage_gate
        ):
            margin_gate = True
        if (
            region_context_override_gate
            or region_target_anchor_rescue_gate
            or region_context_localization_rescue_gate
            or region_high_conf_target_rescue_gate
            or region_exact_target_rescue_gate
            or region_low_raw_exact_target_rescue_gate
            or region_mid_conf_target_context_rescue_gate
            or region_lower_conf_target_context_rescue_gate
            or region_low_conf_target_anchor_rescue_gate
        ):
            margin_gate = True
        if region_anchor_substitute_rescue_gate or region_context_probe_rescue_gate:
            margin_gate = True
        if level_name == "instance" and target_advantage >= 0.55 and float(best["baseline_object_score"]) >= 0.80:
            margin_gate = True
        if instance_target_context_rescue_gate or instance_anchor_identity_rescue_gate:
            margin_gate = True
        if room_target_override_gate or room_local_target_override_gate or room_exact_target_rescue_gate:
            margin_gate = True
        if region_anchor_regression_nonlocal_block:
            semantic_gate = False
            candidate_confidence_gate = False
            margin_gate = False
        correction_eligibility_gate = (
            bool(baseline_rank.get("correction_eligible", False))
            or region_target_anchor_rescue_gate
            or region_context_localization_rescue_gate
            or region_high_conf_target_rescue_gate
            or region_exact_target_rescue_gate
            or region_low_raw_exact_target_rescue_gate
            or region_mid_conf_target_context_rescue_gate
            or region_lower_conf_target_context_rescue_gate
            or region_low_conf_target_anchor_rescue_gate
            or region_anchor_substitute_rescue_gate
            or region_context_probe_rescue_gate
            or instance_target_context_rescue_gate
            or instance_anchor_identity_rescue_gate
            or room_target_override_gate
            or room_local_target_override_gate
            or room_exact_target_rescue_gate
        )
        correction_applied = bool(
            correction_eligibility_gate
            and
            int(best["slot_index"]) != baseline_slot
            and candidate_confidence_gate
            and margin_gate
            and semantic_gate
            and room_or_identity_gate
        )
        selected_order = list(ranked)
        if correction_applied:
            selected_slot = int(best["slot_index"])
            selected_order.sort(key=lambda x: (int(x["slot_index"]) != selected_slot, -float(x["acsd_score"])))
        else:
            selected_order.sort(key=lambda x: (int(x["slot_index"]) != baseline_slot, -float(x["acsd_score"])))
        return {
            "called": True,
            "baseline_slot_index": int(baseline_slot),
            "candidate_count": int(len(ranked)),
            "ranked_candidates": ranked,
            "selected_candidates": selected_order[:1],
            "correction_applied_pre_follow": bool(correction_applied),
            "level_policy": dict(policy),
            "target_advantage": float(target_advantage),
            "anchor_advantage": float(anchor_advantage),
            "best_baseline_distance": float(best_baseline_distance),
            "semantic_gate": bool(semantic_gate),
            "region_context_override_gate": bool(region_context_override_gate),
            "region_target_anchor_rescue_gate": bool(region_target_anchor_rescue_gate),
            "region_context_localization_rescue_gate": bool(region_context_localization_rescue_gate),
            "region_high_conf_target_rescue_gate": bool(region_high_conf_target_rescue_gate),
            "region_exact_target_rescue_gate": bool(region_exact_target_rescue_gate),
            "region_low_raw_exact_target_rescue_gate": bool(region_low_raw_exact_target_rescue_gate),
            "region_mid_conf_target_context_rescue_gate": bool(region_mid_conf_target_context_rescue_gate),
            "region_lower_conf_target_context_rescue_gate": bool(region_lower_conf_target_context_rescue_gate),
            "region_low_conf_target_anchor_rescue_gate": bool(region_low_conf_target_anchor_rescue_gate),
            "region_target_context_advantage_gate": bool(region_target_context_advantage_gate),
            "region_anchor_regression_nonlocal_block": bool(region_anchor_regression_nonlocal_block),
            "region_anchor_substitute_rescue_gate": bool(region_anchor_substitute_rescue_gate),
            "region_context_probe_rescue_gate": bool(region_context_probe_rescue_gate),
            "instance_target_context_rescue_gate": bool(instance_target_context_rescue_gate),
            "instance_anchor_identity_rescue_gate": bool(instance_anchor_identity_rescue_gate),
            "room_target_override_gate": bool(room_target_override_gate),
            "room_local_target_override_gate": bool(room_local_target_override_gate),
            "room_exact_target_rescue_gate": bool(room_exact_target_rescue_gate),
            "correction_eligibility_gate": bool(correction_eligibility_gate),
            "room_or_identity_gate": bool(room_or_identity_gate),
            "candidate_confidence_gate": bool(candidate_confidence_gate),
            "margin_gate": bool(margin_gate),
            "reason": (
                "top ACSD candidate exceeds baseline by adaptive level-aware origin-anchor margin"
                if correction_applied
                else (
                    "baseline retained; no separable VLM object anchors or relations for object correction"
                    if not bool(baseline_rank.get("correction_eligible", False))
                    else "baseline retained because adaptive level-aware margin, target/context gate, or identity safety gate was insufficient"
                )
            ),
        }

    def process_decision(
        self,
        *,
        instruction: str,
        baseline_decision: Mapping[str, Any],
        observation_context: Mapping[str, Any],
        candidate_context: Mapping[str, Any],
        episode_context: Mapping[str, Any],
        gt_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        decomposition = candidate_context.get("decomposition")
        if not isinstance(decomposition, Mapping):
            decomposition = self.decompose_instruction(instruction)
        stage2 = candidate_context.get("stage2")
        if not isinstance(stage2, Mapping):
            raise RuntimeError("ACSD process_decision requires candidate_context['stage2']")
        if bool(decomposition.get("acsd_fallback_to_baseline", False)):
            corrected = {
                "type": str(baseline_decision["type"]),
                "position": list(baseline_decision["position"]),
                "score": float(baseline_decision["score"]),
            }
            aux = baseline_decision.get("decision_aux", {})
            if isinstance(aux, Mapping) and "real_object_decision_idx" in aux:
                corrected["slot_index"] = int(aux["real_object_decision_idx"])
            return {
                "module_enabled": True,
                "decomposition": decomposition,
                "baseline": dict(baseline_decision),
                "corrected": corrected,
                "object_rerank": None,
                "compare": None,
                "logs": [
                    f"[ACSD_ERROR] decompose_source={decomposition.get('decompose_source')} "
                    f"error={decomposition.get('decompose_error')}"
                ],
                "correction_applied": False,
                "correction_rejected": True,
                "correction_reason": f"vlm_anchor_extraction_failed_use_baseline: {decomposition.get('decompose_error')}",
            }
        if baseline_decision["type"] == "frontier":
            corrected = {
                "type": "frontier",
                "position": list(baseline_decision["position"]),
                "score": float(baseline_decision["score"]),
            }
            return {
                "module_enabled": True,
                "decomposition": decomposition,
                "baseline": dict(baseline_decision),
                "corrected": corrected,
                "object_rerank": None,
                "compare": None,
                "logs": [],
                "correction_applied": False,
                "correction_rejected": False,
                "correction_reason": "frontier_decision_kept_without_acsd_correction",
            }
        if baseline_decision["type"] != "object":
            raise RuntimeError(f"unknown baseline decision type: {baseline_decision['type']!r}")
        object_rerank = self.rerank_object_candidates(
            baseline_decision=baseline_decision,
            stage2=stage2,
            rep=observation_context["representation_manager"],
            decomposition=decomposition,
            output_dir=Path(str(candidate_context.get("output_dir", ""))),
            task_level=episode_context.get("task_level", candidate_context.get("task_level", "")),
        )
        selected = object_rerank["selected_candidates"][0]
        correction_applied = bool(object_rerank["correction_applied_pre_follow"])
        baseline_slot = int(object_rerank["baseline_slot_index"])
        baseline_candidate = next(
            (x for x in object_rerank["ranked_candidates"] if int(x["slot_index"]) == baseline_slot),
            None,
        )
        if baseline_candidate is None:
            raise RuntimeError(f"baseline object slot {baseline_slot} missing from ranked candidates")
        corrected_candidate = selected if correction_applied else baseline_candidate
        corrected = {
            "type": "object",
            "position": (
                list(selected["center_habitat_xyz"])
                if correction_applied
                else list(baseline_decision["position"])
            ),
            "score": float(corrected_candidate["acsd_score"]),
            "slot_index": int(corrected_candidate["slot_index"]),
        }
        return {
            "module_enabled": True,
            "decomposition": decomposition,
            "baseline": dict(baseline_decision),
            "corrected": corrected,
            "object_rerank": object_rerank,
            "compare": None,
            "logs": [],
            "correction_applied": bool(correction_applied),
            "correction_rejected": False,
            "correction_reason": f"{object_rerank['reason']}; selected_slot={int(corrected['slot_index'])}",
        }

    def build_compare_record(
        self,
        *,
        episode_id: Any,
        step_id: Any,
        baseline_position: Sequence[float],
        corrected_position: Sequence[float],
        goal_positions: Sequence[Sequence[float]],
        threshold_m: float = 1.0,
    ) -> Dict[str, Any]:
        if len(goal_positions) == 0:
            raise RuntimeError("ACSD_COMPARE requires at least one GT goal position")
        baseline = _as_np3(baseline_position, name="baseline compare position")
        corrected = _as_np3(corrected_position, name="corrected compare position")
        goals = [_as_np3(g, name="goal position") for g in goal_positions]
        bdist = float(min(np.linalg.norm(baseline - g) for g in goals))
        cdist = float(min(np.linalg.norm(corrected - g) for g in goals))
        b_ok = int(bdist <= float(threshold_m))
        c_ok = int(cdist <= float(threshold_m))
        case = f"{b_ok}{c_ok}"
        if case not in ("10", "01", "11", "00"):
            raise RuntimeError(f"invalid ACSD compare case: {case}")
        self.summary[f"case_{case}"] += 1
        self.summary["total_compared"] += 1
        return {
            "episode_id": episode_id,
            "step_id": step_id,
            "baseline_dist_to_gt": float(bdist),
            "acsd_dist_to_gt": float(cdist),
            "baseline_within_1m": int(b_ok),
            "acsd_within_1m": int(c_ok),
            "case": case,
        }

    def update_summary_from_decision(
        self,
        *,
        correction_applied: bool,
        correction_rejected: bool,
    ) -> None:
        if correction_applied:
            self.summary["correction_applied"] += 1
        if correction_rejected:
            self.summary["correction_rejected"] += 1

    @staticmethod
    def format_call_log(
        *,
        episode_id: Any,
        step_id: Any,
        decomposition: Mapping[str, Any],
        baseline: Mapping[str, Any],
        corrected: Mapping[str, Any],
        correction_applied: bool,
        correction_reason: str,
    ) -> str:
        return (
            "[ACSD_CALL] "
            f"episode_id={episode_id} "
            f"step_id={step_id} "
            "module_enabled=True "
            f"instruction={json.dumps(str(decomposition['full_instruction']), ensure_ascii=False)} "
            f"room_anchor={json.dumps(str(decomposition.get('room_anchor', '')), ensure_ascii=False)} "
            f"object_anchors={json.dumps(list(decomposition.get('object_anchors', [])), ensure_ascii=False)} "
            f"baseline_decision_type={baseline['type']} "
            f"baseline_position={json.dumps(baseline['position'])} "
            f"baseline_score={float(baseline['score']):.6f} "
            f"corrected_decision_type={corrected['type']} "
            f"corrected_position={json.dumps(corrected['position'])} "
            f"corrected_score={float(corrected.get('score', 0.0)):.6f} "
            f"correction_applied={bool(correction_applied)} "
            f"correction_reason={json.dumps(str(correction_reason), ensure_ascii=False)} "
            f"anchor_policy={json.dumps(str(decomposition.get('anchor_policy', '')), ensure_ascii=False)} "
            f"decompose_source={json.dumps(str(decomposition.get('decompose_source', '')), ensure_ascii=False)} "
            f"acsd_fallback_to_baseline={bool(decomposition.get('acsd_fallback_to_baseline', False))} "
            f"decompose_error={json.dumps(str(decomposition.get('decompose_error', '')), ensure_ascii=False)}"
        )

    @staticmethod
    def format_compare_log(compare: Mapping[str, Any]) -> str:
        return (
            "[ACSD_COMPARE] "
            f"episode_id={compare['episode_id']} "
            f"step_id={compare['step_id']} "
            f"baseline_dist_to_gt={float(compare['baseline_dist_to_gt']):.6f} "
            f"acsd_dist_to_gt={float(compare['acsd_dist_to_gt']):.6f} "
            f"baseline_within_1m={int(compare['baseline_within_1m'])} "
            f"acsd_within_1m={int(compare['acsd_within_1m'])} "
            f"case={compare['case']}"
        )

    def format_summary_log(self) -> str:
        s = self.summary
        return (
            "[ACSD_SUMMARY] "
            f"case_10={int(s['case_10'])} "
            f"case_01={int(s['case_01'])} "
            f"case_11={int(s['case_11'])} "
            f"case_00={int(s['case_00'])} "
            f"total_compared={int(s['total_compared'])} "
            f"correction_applied={int(s['correction_applied'])} "
            f"correction_rejected={int(s['correction_rejected'])}"
        )


__all__ = [
    "ACSDConfig",
    "AnchorConditionedSoftDecomposition",
    "parse_json_object_strict",
]
