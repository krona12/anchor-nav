from __future__ import annotations

import json
import math
import re
import tempfile
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


@dataclass
class ACSDConfig:
    vlm_model: str = "gpt-4o-mini"
    object_top_k: int = 4
    max_object_anchors: int = 3
    verify_attempts: int = 2
    vlm_max_retries: int = 3
    vlm_retry_sleep_sec: float = 3.0
    correction_margin: float = 0.015
    frontier_correction_margin: float = 0.04
    baseline_weight: float = 0.42
    target_weight: float = 0.18
    anchor_weight: float = 0.22
    relation_weight: float = 0.18
    frontier_baseline_weight: float = 0.55
    frontier_anchor_weight: float = 0.35
    frontier_commonsense_weight: float = 0.10
    object_correction_min_baseline_score: float = 0.90
    enable_frontier_correction: bool = False


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
            "vlm_verified_true": 0,
            "vlm_verified_false": 0,
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

    @staticmethod
    def _rule_decompose_instruction(instruction: str) -> Optional[Dict[str, Any]]:
        text = str(instruction or "").strip()
        norm = _normalize_phrase(text)
        if not norm:
            return None
        relation_tokens = [
            "next to",
            "beside",
            "between",
            "against",
            "near",
            "above",
            "below",
            "under",
            "on top of",
            "on",
            "inside",
            "in front of",
            "behind",
            "with",
            "in",
        ]
        room_words = [
            "bedroom",
            "bathroom",
            "kitchen",
            "living room",
            "dining room",
            "hall",
            "hallway",
            "office",
            "study",
            "garage",
            "laundry room",
            "closet",
            "entryway",
            "lounge",
            "room",
        ]
        colors = [
            "white",
            "black",
            "gray",
            "grey",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "pink",
            "orange",
            "brown",
            "beige",
            "teal",
            "gold",
            "silver",
            "wooden",
            "metal",
            "stone",
            "glass",
        ]
        room_anchor = ""
        for room in room_words:
            if re.search(rf"\b{re.escape(room)}\b", norm):
                room_anchor = room
                break

        first_rel_pos = len(norm)
        first_rel = ""
        for token in relation_tokens:
            m = re.search(rf"\b{re.escape(token)}\b", norm)
            if m and m.start() < first_rel_pos:
                first_rel_pos = m.start()
                first_rel = token
        head = norm[:first_rel_pos].strip() if first_rel else norm

        target_object = re.sub(r"^(find|go to|navigate to|look for|the|a|an)\s+", "", head).strip()
        target_object = re.sub(r"\s+", " ", target_object).strip()
        if not target_object:
            return None
        if target_object == norm:
            # Full-sentence target is not an acceptable structured parse.
            words = target_object.split()
            if len(words) > 7:
                return None

        attributes = [c for c in colors if re.search(rf"\b{re.escape(c)}\b", norm)]
        spatial_relations: List[str] = []
        object_anchors: List[str] = []
        nested_rel_pattern = (
            r"\s+\b(?:next to|beside|between|against|near|above|below|under|on top of|on|inside|"
            r"in front of|behind|with|in)\b\s+"
        )
        rel_alt = (
            r"next to|in front of|on top of|beside|between|against|near|above|below|"
            r"under|inside|behind|with|on|in"
        )
        rel_pattern = rf"\b({rel_alt})\b\s+(.+?)(?=\s+\b(?:{rel_alt})\b\s+|[,.;]|$)"
        for m in re.finditer(rel_pattern, norm):
            rel = m.group(1).strip()
            phrase = m.group(2).strip()
            if not phrase:
                continue
            phrase = re.split(r"\b(?:and then|while|where)\b", phrase)[0].strip()
            relation_text = f"{rel} {phrase}".strip()
            if relation_text and relation_text not in spatial_relations:
                spatial_relations.append(relation_text)
            if rel in ("with",) and phrase.startswith(("red ", "white ", "black ", "gray ", "grey ", "blue ", "green ")):
                continue
            if rel == "in" and any(room in phrase for room in room_words):
                continue
            cleaned = re.split(nested_rel_pattern, phrase, maxsplit=1)[0].strip()
            cleaned = re.split(r"\s+\b(?:and|or)\b\s+", cleaned, maxsplit=1)[0].strip()
            cleaned = re.sub(r"^(the|a|an|some)\s+", "", cleaned).strip()
            cleaned = re.sub(r"\b(?:area|space|place|room)\b$", "", cleaned).strip()
            if any(room == cleaned for room in room_words):
                continue
            if len(cleaned.split()) > 5:
                continue
            if target_object in cleaned or cleaned in target_object:
                continue
            if cleaned and cleaned not in object_anchors and cleaned != target_object:
                object_anchors.append(cleaned)
        if room_anchor and room_anchor not in object_anchors:
            anchor_prompt_parts = [room_anchor] + object_anchors
        else:
            anchor_prompt_parts = list(object_anchors)
        anchor_prompt = ", ".join(anchor_prompt_parts)
        relation_prompt = "; ".join(spatial_relations) if spatial_relations else target_object
        verification_prompt = (
            f"Does this candidate contain the {target_object}"
            + (f" in or near {anchor_prompt}" if anchor_prompt else "")
            + (f" with relations: {relation_prompt}?" if relation_prompt else "?")
        )
        return {
            "full_instruction": text,
            "target_object": target_object,
            "room_anchor": room_anchor,
            "object_anchors": object_anchors[:3],
            "attributes": attributes,
            "spatial_relations": spatial_relations,
            "target_prompt": target_object,
            "anchor_prompt": anchor_prompt,
            "relation_prompt": relation_prompt,
            "verification_prompt": verification_prompt,
            "raw": "rule_decompose_instruction",
            "decompose_source": "rule",
        }

    def decompose_instruction(self, instruction: str) -> Dict[str, Any]:
        text = str(instruction or "").strip()
        if not text:
            raise ValueError("instruction is empty")
        rule = self._rule_decompose_instruction(text)
        if rule is not None:
            return rule
        prompt = (
            "You decompose fine-grained navigation instructions into semantic constraints.\n"
            "Return strict JSON only with exactly these fields:\n"
            "{\n"
            "  \"full_instruction\": string,\n"
            "  \"target_object\": string,\n"
            "  \"room_anchor\": string,\n"
            "  \"object_anchors\": [string],\n"
            "  \"attributes\": [string],\n"
            "  \"spatial_relations\": [string],\n"
            "  \"target_prompt\": string,\n"
            "  \"anchor_prompt\": string,\n"
            "  \"relation_prompt\": string,\n"
            "  \"verification_prompt\": string\n"
            "}\n"
            "Rules:\n"
            "- target_object is the primary object to find, not a room or relation.\n"
            "- room_anchor is empty only if no room/region context exists.\n"
            "- object_anchors are supporting objects, furniture, fixtures, or distinctive objects.\n"
            "- attributes are visual modifiers such as color, material, size, count.\n"
            "- spatial_relations are concise relation phrases involving target and anchors.\n"
            "- Do not copy the full instruction as target_object.\n\n"
            f"Instruction: {text}"
        )
        raw = self._chat_strict(text=prompt, image_path=None, max_tokens=512, source="ACSD decomposer")
        parsed = parse_json_object_strict(raw, source="ACSD decomposer")
        out = {
            "full_instruction": _require_str(parsed, "full_instruction") or text,
            "target_object": _normalize_phrase(_require_str(parsed, "target_object")),
            "room_anchor": _normalize_phrase(parsed.get("room_anchor", "")),
            "object_anchors": self._active_object_anchors(
                {"object_anchors": [_normalize_phrase(x) for x in _require_str_list(parsed, "object_anchors") if _normalize_phrase(x)]}
            ),
            "attributes": [_normalize_phrase(x) for x in _require_str_list(parsed, "attributes") if _normalize_phrase(x)],
            "spatial_relations": [str(x).strip() for x in _require_str_list(parsed, "spatial_relations") if str(x).strip()],
            "target_prompt": _require_str(parsed, "target_prompt"),
            "anchor_prompt": _require_str(parsed, "anchor_prompt"),
            "relation_prompt": _require_str(parsed, "relation_prompt"),
            "verification_prompt": _require_str(parsed, "verification_prompt"),
            "raw": str(raw),
            "decompose_source": "vlm",
        }
        if not out["target_object"]:
            raise ValueError(f"ACSD decomposer failed to extract target_object for instruction={text!r}")
        if not out["target_prompt"].strip():
            raise RuntimeError("ACSD decomposer returned empty target_prompt")
        if not out["verification_prompt"].strip():
            raise RuntimeError("ACSD decomposer returned empty verification_prompt")
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
        if len(out) < int(self.cfg.verify_attempts):
            raise RuntimeError(
                f"ACSD object rerank selection policy needs {self.cfg.verify_attempts} object candidates, got {len(out)}"
            )
        return out

    def _frontier_candidates_from_stage2(self, *, stage2: Mapping[str, Any]) -> List[Dict[str, Any]]:
        frs = stage2.get("frontier_candidates")
        if not isinstance(frs, list) or len(frs) == 0:
            raise RuntimeError("ACSD frontier prior requires non-empty stage2 frontier_candidates")
        out: List[Dict[str, Any]] = []
        for rec in frs:
            center = _as_np3(rec["center_habitat_xyz"], name=f"frontier candidate {rec.get('frontier_index')} center")
            out.append(
                {
                    "frontier_index": int(rec["frontier_index"]),
                    "center_habitat_xyz": center.tolist(),
                    "og3d_logit": float(rec["og3d_logit"]),
                }
            )
        return out

    def _active_object_anchors(self, decomposition: Mapping[str, Any]) -> List[str]:
        raw = decomposition.get("object_anchors", [])
        if not isinstance(raw, list):
            raise RuntimeError("ACSD decomposition object_anchors must be a list")
        anchors: List[str] = []
        for item in raw:
            text = _normalize_phrase(item)
            if not text:
                continue
            words = text.split()
            if len(words) > 5:
                continue
            if text in {"room", "area", "space", "place"}:
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

    def _score_candidates_with_rules(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        decomposition: Mapping[str, Any],
    ) -> Dict[int, Dict[str, Any]]:
        if len(candidates) == 0:
            raise RuntimeError("ACSD rule scorer requires non-empty candidates")
        base_scores = self._minmax_scores([float(c["og3d_logit"]) for c in candidates])
        merged_scores = self._minmax_scores([float(c["merged_object_score"]) for c in candidates])
        centers = [_as_np3(c["center_habitat_xyz"], name=f"object candidate {c.get('slot_index')} center") for c in candidates]
        active_anchors = self._active_object_anchors(decomposition)
        relation_active = bool(list(decomposition.get("spatial_relations", [])))
        room_active = bool(str(decomposition.get("room_anchor", "")).strip())
        correction_eligible = bool(active_anchors or relation_active)
        constraint_active = bool(correction_eligible or room_active)
        if len(candidates) < 2 and (active_anchors or relation_active):
            raise RuntimeError("ACSD relation scoring requires at least two object candidates when anchors/relations are active")
        raw_anchor_values: List[float] = []
        for idx, _ in enumerate(candidates):
            proximity_values: List[float] = []
            for j, other in enumerate(centers):
                if j == idx:
                    continue
                dist = float(np.linalg.norm(centers[idx][[0, 2]] - other[[0, 2]]))
                proximity_values.append(float(merged_scores[j]) / (1.0 + dist))
            if proximity_values:
                raw_anchor_values.append(float(max(proximity_values)))
            elif active_anchors or relation_active:
                raise RuntimeError("ACSD relation scoring had no neighbor object candidates")
            else:
                raw_anchor_values.append(0.5)
        anchor_scores = (
            self._minmax_scores(raw_anchor_values)
            if (active_anchors or relation_active)
            else [0.5 for _ in candidates]
        )
        raw_relation_values = [
            float(0.65 * float(anchor_scores[idx]) + 0.35 * float(merged_scores[idx]))
            for idx, _ in enumerate(candidates)
        ]
        relation_scores = (
            self._minmax_scores(raw_relation_values)
            if (active_anchors or relation_active)
            else [0.5 for _ in candidates]
        )
        by_rank: Dict[int, Dict[str, Any]] = {}
        for idx, cand in enumerate(candidates):
            target_match = float(merged_scores[idx])
            rank = int(cand["rank"])
            by_rank[rank] = {
                "target_match_score": float(max(0.0, min(1.0, target_match))),
                "anchor_match_score": float(max(0.0, min(1.0, anchor_scores[idx]))),
                "relation_score": float(max(0.0, min(1.0, relation_scores[idx]))),
                "active_object_anchors": list(active_anchors),
                "constraint_active": bool(constraint_active),
                "correction_eligible": bool(correction_eligible),
                "target_match_source": "merged_object_score_semantic_labels_unavailable",
                "reason": (
                    "rule_only_object_rerank_component4_removed; "
                    f"constraint_active={constraint_active}; correction_eligible={correction_eligible}; "
                    f"active_object_anchors={active_anchors}; "
                    "anchor_policy=soft_topk_not_all_required; score_normalization=minmax_neutral_0.5; "
                    "no candidate VLM scorer called"
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
    ) -> Dict[str, Any]:
        if str(baseline_decision["type"]) != "object":
            raise RuntimeError("rerank_object_candidates called for non-object baseline decision")
        candidates = self._object_candidates_from_stage2(
            stage2=stage2,
            rep=rep,
            top_k=int(self.cfg.object_top_k),
            require_images=False,
        )
        rule_scores = self._score_candidates_with_rules(candidates=candidates, decomposition=decomposition)
        base_scores = self._minmax_scores([float(c["og3d_logit"]) for c in candidates])
        ranked: List[Dict[str, Any]] = []
        for cand, base in zip(candidates, base_scores):
            extra = rule_scores[int(cand["rank"])]
            parts: List[Tuple[float, float]] = [
                (float(self.cfg.baseline_weight), float(base)),
                (float(self.cfg.target_weight), float(extra["target_match_score"])),
            ]
            if bool(extra.get("correction_eligible", False)):
                parts.extend(
                    [
                        (float(self.cfg.anchor_weight), float(extra["anchor_match_score"])),
                        (float(self.cfg.relation_weight), float(extra["relation_score"])),
                    ]
                )
            final = self._weighted_normalized_score(parts)
            item = dict(cand)
            item.update(extra)
            item["baseline_object_score"] = float(base)
            item["acsd_score"] = float(final)
            ranked.append(item)
        ranked.sort(key=lambda x: float(x["acsd_score"]), reverse=True)
        baseline_slot = int(baseline_decision["decision_aux"]["real_object_decision_idx"])
        baseline_rank = next((x for x in ranked if int(x["slot_index"]) == baseline_slot), None)
        if baseline_rank is None:
            raise RuntimeError(f"baseline object slot {baseline_slot} not present after reranking")
        best = ranked[0]
        correction_applied = bool(
            bool(baseline_rank.get("correction_eligible", False))
            and
            int(best["slot_index"]) != baseline_slot
            and float(best["baseline_object_score"]) >= float(self.cfg.object_correction_min_baseline_score)
            and float(best["acsd_score"]) >= float(baseline_rank["acsd_score"]) + float(self.cfg.correction_margin)
        )
        selected_order = list(ranked)
        if not correction_applied:
            selected_order.sort(key=lambda x: (int(x["slot_index"]) != baseline_slot, -float(x["acsd_score"])))
        return {
            "called": True,
            "baseline_slot_index": int(baseline_slot),
            "candidate_count": int(len(ranked)),
            "ranked_candidates": ranked,
            "selected_candidates": selected_order[: int(self.cfg.verify_attempts)],
            "correction_applied_pre_verification": bool(correction_applied),
            "reason": (
                "top ACSD rule-only candidate exceeds baseline by margin"
                if correction_applied
                else (
                    "baseline retained; no separable object anchors or relations for safe rule-only correction"
                    if not bool(baseline_rank.get("correction_eligible", False))
                    else "baseline retained because normalized ACSD margin or candidate target-confidence gate was insufficient"
                )
            ),
            "component4_removed": True,
            "vlm_candidate_scorer_called": False,
        }

    def score_frontier_with_anchors(
        self,
        *,
        baseline_decision: Mapping[str, Any],
        stage2: Mapping[str, Any],
        decomposition: Mapping[str, Any],
    ) -> Dict[str, Any]:
        candidates = self._frontier_candidates_from_stage2(stage2=stage2)
        object_candidates = stage2.get("object_candidates")
        if not isinstance(object_candidates, list) or len(object_candidates) == 0:
            raise RuntimeError("ACSD frontier prior requires object_candidates as observed semantic context")
        obj_centers = [_as_np3(o["center_habitat_xyz"], name="object candidate center") for o in object_candidates]
        obj_logits = self._minmax_scores([float(o["og3d_logit"]) for o in object_candidates])
        base_scores = self._minmax_scores([float(f["og3d_logit"]) for f in candidates])
        target_object = str(decomposition.get("target_object", ""))
        room_anchor = str(decomposition.get("room_anchor", ""))
        active_anchors = self._active_object_anchors(decomposition)
        relation_active = bool(list(decomposition.get("spatial_relations", [])))
        target_room_bonus = self._target_room_commonsense(target_object, room_anchor)

        raw_anchor_values: List[float] = []
        for cand in candidates:
            p = _as_np3(cand["center_habitat_xyz"], name="frontier center")
            proximity_values: List[float] = []
            for obj_center, obj_logit in zip(obj_centers, obj_logits):
                dist = float(np.linalg.norm(p[[0, 2]] - obj_center[[0, 2]]))
                proximity_values.append(float(obj_logit) / (1.0 + dist))
            if not proximity_values:
                raise RuntimeError("frontier anchor relevance had empty proximity list")
            raw_anchor_values.append(float(max(proximity_values)))
        anchor_scores = (
            self._minmax_scores(raw_anchor_values)
            if (active_anchors or relation_active)
            else [0.5 for _ in candidates]
        )

        scored: List[Dict[str, Any]] = []
        for idx, (cand, base) in enumerate(zip(candidates, base_scores)):
            parts: List[Tuple[float, float]] = [(float(self.cfg.frontier_baseline_weight), float(base))]
            if active_anchors or relation_active:
                parts.append((float(self.cfg.frontier_anchor_weight), float(anchor_scores[idx])))
            if room_anchor:
                parts.append((float(self.cfg.frontier_commonsense_weight), float(target_room_bonus)))
            score = self._weighted_normalized_score(parts)
            rec = dict(cand)
            rec.update(
                {
                    "baseline_frontier_score": float(base),
                    "room_anchor_relevance": float(target_room_bonus),
                    "object_anchor_relevance": float(anchor_scores[idx]),
                    "target_anchor_commonsense_score": float(target_room_bonus),
                    "acsd_score": float(score),
                    "active_object_anchors": list(active_anchors),
                    "anchor_policy": "soft_topk_not_all_required",
                    "anchor_relevance_source": "stage2_object_candidate_proximity_normalized_semantic_labels_unavailable",
                }
            )
            scored.append(rec)
        scored.sort(key=lambda x: float(x["acsd_score"]), reverse=True)
        baseline_position = _as_np3(baseline_decision["position"], name="baseline frontier position")
        baseline_idx = self._match_frontier_index_by_position(scored, baseline_position)
        baseline_row = next((x for x in scored if int(x["frontier_index"]) == int(baseline_idx)), None)
        if baseline_row is None:
            raise RuntimeError(f"baseline frontier index {baseline_idx} disappeared after scoring")
        best = scored[0]
        correction = bool(
            bool(self.cfg.enable_frontier_correction)
            and
            (active_anchors or relation_active)
            and len(scored) > 1
            and
            int(best["frontier_index"]) != int(baseline_idx)
            and float(best["acsd_score"]) >= float(baseline_row["acsd_score"]) + float(self.cfg.frontier_correction_margin)
        )
        chosen = best if correction else baseline_row
        if correction:
            reason = "anchor-conditioned frontier rerank applied"
        elif not bool(self.cfg.enable_frontier_correction):
            reason = "baseline frontier kept; frontier prior scored but correction disabled for baseline-parity safety"
        elif not (active_anchors or relation_active):
            reason = "baseline frontier kept; no separable object anchors for soft frontier rerank"
        else:
            reason = "baseline frontier kept by normalized ACSD margin"
        return {
            "called": True,
            "baseline_frontier_index": int(baseline_idx),
            "scored_frontiers": scored,
            "selected_frontier": chosen,
            "correction_applied": bool(correction),
            "reason": reason,
        }

    @staticmethod
    def _match_frontier_index_by_position(scored: Sequence[Mapping[str, Any]], pos: np.ndarray) -> int:
        dists = []
        for rec in scored:
            p = _as_np3(rec["center_habitat_xyz"], name="frontier center")
            dists.append((float(np.linalg.norm(p - pos)), int(rec["frontier_index"])))
        if not dists:
            raise RuntimeError("cannot match baseline frontier from empty candidate list")
        dists.sort(key=lambda x: x[0])
        return int(dists[0][1])

    @staticmethod
    def _target_room_commonsense(target_object: str, room_anchor: str) -> float:
        target = _normalize_phrase(target_object)
        room = _normalize_phrase(room_anchor)
        if not room:
            return 0.5
        table = {
            "bedroom": ("bed", "lamp", "nightstand", "pillow", "dresser", "wardrobe", "blanket"),
            "bathroom": ("toilet", "sink", "shower", "bathtub", "towel", "mirror"),
            "kitchen": ("stove", "oven", "microwave", "sink", "refrigerator", "counter", "cabinet"),
            "living room": ("sofa", "couch", "tv", "television", "coffee table", "chair"),
            "dining room": ("dining table", "chair", "table", "cabinet"),
            "office": ("desk", "monitor", "chair", "computer", "bookshelf"),
        }
        for key, words in table.items():
            if key in room:
                return 1.0 if any(w in target for w in words) else 0.65
        return 0.6

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
        output_dir = Path(str(candidate_context.get("output_dir", tempfile.mkdtemp(prefix="acsd_"))))
        if baseline_decision["type"] == "frontier":
            frontier_prior = self.score_frontier_with_anchors(
                baseline_decision=baseline_decision,
                stage2=stage2,
                decomposition=decomposition,
            )
            selected = frontier_prior["selected_frontier"]
            frontier_correction_applied = bool(frontier_prior["correction_applied"])
            corrected = {
                "type": "frontier",
                "position": (
                    list(selected["center_habitat_xyz"])
                    if frontier_correction_applied
                    else list(baseline_decision["position"])
                ),
                "score": float(selected["acsd_score"]),
            }
            verification = {
                "vlm_called": False,
                "verified": False,
                "confidence": 0.0,
                "reason": "not_applicable_frontier_decision",
                "matched_target": False,
                "matched_anchor": False,
                "matched_relation": False,
            }
            return {
                "module_enabled": True,
                "decomposition": decomposition,
                "baseline": dict(baseline_decision),
                "corrected": corrected,
                "frontier_prior": frontier_prior,
                "object_rerank": None,
                "verification": verification,
                "compare": None,
                "logs": [],
            }
        if baseline_decision["type"] != "object":
            raise RuntimeError(f"unknown baseline decision type: {baseline_decision['type']!r}")
        object_rerank = self.rerank_object_candidates(
            baseline_decision=baseline_decision,
            stage2=stage2,
            rep=observation_context["representation_manager"],
            decomposition=decomposition,
            output_dir=output_dir,
        )
        selected = object_rerank["selected_candidates"][0]
        correction_applied = bool(object_rerank["correction_applied_pre_verification"])
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
        verification = {
            "vlm_called": False,
            "verified": False,
            "confidence": 0.0,
            "reason": "component4_removed_by_user_request",
            "matched_target": False,
            "matched_anchor": False,
            "matched_relation": False,
        }
        return {
            "module_enabled": True,
            "decomposition": decomposition,
            "baseline": dict(baseline_decision),
            "corrected": corrected,
            "frontier_prior": None,
            "object_rerank": object_rerank,
            "verification": verification,
            "compare": None,
            "logs": [],
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
        verification: Mapping[str, Any],
    ) -> None:
        if correction_applied:
            self.summary["correction_applied"] += 1
        if correction_rejected:
            self.summary["correction_rejected"] += 1
        if bool(verification.get("vlm_called", True)):
            if bool(verification.get("verified", False)):
                self.summary["vlm_verified_true"] += 1
            else:
                self.summary["vlm_verified_false"] += 1

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
        verification: Mapping[str, Any],
    ) -> str:
        return (
            "[ACSD_CALL] "
            f"episode_id={episode_id} "
            f"step_id={step_id} "
            "module_enabled=True "
            f"instruction={json.dumps(str(decomposition['full_instruction']), ensure_ascii=False)} "
            f"target_object={json.dumps(str(decomposition['target_object']), ensure_ascii=False)} "
            f"room_anchor={json.dumps(str(decomposition.get('room_anchor', '')), ensure_ascii=False)} "
            f"object_anchors={json.dumps(list(decomposition.get('object_anchors', [])), ensure_ascii=False)} "
            f"attributes={json.dumps(list(decomposition.get('attributes', [])), ensure_ascii=False)} "
            f"spatial_relations={json.dumps(list(decomposition.get('spatial_relations', [])), ensure_ascii=False)} "
            f"baseline_decision_type={baseline['type']} "
            f"baseline_position={json.dumps(baseline['position'])} "
            f"baseline_score={float(baseline['score']):.6f} "
            f"corrected_decision_type={corrected['type']} "
            f"corrected_position={json.dumps(corrected['position'])} "
            f"corrected_score={float(corrected.get('score', 0.0)):.6f} "
            f"correction_applied={bool(correction_applied)} "
            f"correction_reason={json.dumps(str(correction_reason), ensure_ascii=False)} "
            f"verifier_verified={bool(verification.get('verified', False))} "
            f"verifier_confidence={float(verification.get('confidence', 0.0)):.6f} "
            f"verifier_reason={json.dumps(str(verification.get('reason', '')), ensure_ascii=False)}"
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
            f"correction_rejected={int(s['correction_rejected'])} "
            f"vlm_verified_true={int(s['vlm_verified_true'])} "
            f"vlm_verified_false={int(s['vlm_verified_false'])}"
        )


__all__ = [
    "ACSDConfig",
    "AnchorConditionedSoftDecomposition",
    "parse_json_object_strict",
]
