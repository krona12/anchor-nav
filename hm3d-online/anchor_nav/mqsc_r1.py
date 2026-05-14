from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vlm.client import DEFAULT_MODEL, chat


MODULE_NAME = "mqsc-r1"
PROMPT_VERSION = "mqsc_r1_decompose_v1_object_only_region_consensus"


@dataclass
class MqscR1Config:
    top_k: int = 8
    temperature: float = 1.0
    cluster_eps: float = 1.2
    min_region_coverage: float = 0.5
    min_target_prob: float = 0.05
    min_region_margin: float = 0.05
    min_selected_gain: float = -0.02
    use_vlm: bool = True
    vlm_model: str = DEFAULT_MODEL
    vlm_max_retries: int = 3
    vlm_retry_sleep_sec: float = 1.0
    vlm_no_proxy: bool = True
    allow_heuristic_decompose: bool = True
    write_debug_json: bool = True
    exclude_room_context_query: bool = True
    excluded_consensus_roles: Tuple[str, ...] = ("room_context",)
    role_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "full": 0.8,
            "target": 1.35,
            "anchor_primary": 0.9,
            "anchor_support": 0.7,
        }
    )


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextmanager
def no_proxy_env(enabled: bool = True):
    if not enabled:
        yield
        return
    old = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS + ("NO_PROXY", "no_proxy")}
    try:
        for key in _PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_phrase(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        phrase = _normalize_phrase(value)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= int(limit):
            break
    return out


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x) for x in value if str(x).strip()]


def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for idx in range(start, len(text)):
        ch = text[idx]
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
                return text[start : idx + 1]
    return None


def parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    candidates: List[str] = []
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1).strip())
    balanced = _extract_balanced_json(text)
    if balanced:
        candidates.append(balanced)
    if text.startswith("{"):
        candidates.append(text)
    seen = set()
    errors: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("MQSC VLM response is not a JSON object: " + " | ".join(errors))


def build_decomposition_prompt(description: str, task_type: str = "") -> str:
    return (
        "You decompose a RefHM3D navigation instruction for multi-query 3D object grounding.\n"
        "The downstream algorithm will query a detector separately with full task, target, anchors, and room context.\n\n"
        "Return strict JSON only with this exact schema:\n"
        "{"
        "\"target_desc\": string, "
        "\"target_aliases\": [string], "
        "\"anchor_primary\": [string], "
        "\"anchor_support\": [string], "
        "\"room_context\": [string], "
        "\"relations\": [string]"
        "}\n\n"
        "Definitions and hard rules:\n"
        "1. target_desc is the primary object to navigate to, not a location or anchor.\n"
        "2. target_aliases are short alternative names for the same target object; keep 0-3.\n"
        "3. anchor_primary are the strongest nearby objects/furniture/fixtures that identify the target area; keep 0-2.\n"
        "4. anchor_support are other useful nearby concrete objects or visual area cues; keep 0-4.\n"
        "5. room_context contains room or region labels only when explicit or strongly implied; keep 0-2.\n"
        "6. Do not put the target object itself into anchor_primary, anchor_support, or room_context.\n"
        "7. Avoid generic structural anchors such as wall, floor, ceiling, door, window unless modified into a distinctive object phrase.\n"
        "8. Use lowercase concise noun phrases. No instructions, no sentences, no markdown.\n\n"
        f"task_type: {task_type}\n"
        f"navigation_instruction: {description}"
    )


def _heuristic_decompose(description: str, task_type: str = "") -> Dict[str, Any]:
    text = str(description or "").strip()
    norm = _normalize_phrase(text)
    target = norm
    anchors: List[str] = []
    room_context: List[str] = []
    relations: List[str] = []

    region_match = re.match(r"^\s*(.+?)\s+in\s+the\s+(.+?)\s+that\s+has\s+(.+)$", text, flags=re.IGNORECASE)
    room_match = re.match(r"^\s*(.+?)\s+in\s+the\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    room_with_context_match = re.match(
        r"^\s*(.+?)\s+in\s+(?:the\s+)?(.+?)(?:\s+with\s+|\s+near\s+|\s+beside\s+|\s+next\s+to\s+)(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if region_match:
        target = _normalize_phrase(region_match.group(1))
        room_context.append(region_match.group(2))
        anchors.extend(re.split(r",|;|\band\b|\bwith\b", region_match.group(3), flags=re.IGNORECASE))
    elif room_match:
        target = _normalize_phrase(room_match.group(1))
        room_context.append(room_match.group(2))
    elif room_with_context_match:
        target = _normalize_phrase(room_with_context_match.group(1))
        room_context.append(room_with_context_match.group(2))
        anchors.extend(
            re.split(r",|;|\band\b|\bor\b|\bwith\b", room_with_context_match.group(3), flags=re.IGNORECASE)
        )
    else:
        split = re.split(
            r"\bnear\b|\bbeside\b|\bnext to\b|\bon top of\b|\bon\b|\bunder\b|\bbelow\b|\babove\b|\bbetween\b|\bwith\b|\bin\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if split:
            target = _normalize_phrase(split[0])
        if len(split) > 1:
            anchors.extend(re.split(r",|;|\band\b|\bor\b", split[1], flags=re.IGNORECASE))
        relations.extend(re.findall(r"\b(near|beside|next to|on top of|on|under|below|above|between)\b", text, flags=re.IGNORECASE))

    target = target or norm or "object"
    anchors_clean = [a for a in _dedupe(anchors, limit=6) if a and a != target and target not in a]
    return {
        "target_desc": target,
        "target_aliases": [],
        "anchor_primary": anchors_clean[:2],
        "anchor_support": anchors_clean[2:6],
        "room_context": _dedupe(room_context, limit=2),
        "relations": _dedupe(relations, limit=4),
        "raw": "",
        "source": "heuristic_fallback",
        "parse_ok": False,
        "parse_attempts": 0,
        "error_type": "",
        "error_message": "",
        "errors": [],
    }


def sanitize_query_spec(parsed: Mapping[str, Any], *, description: str, task_type: str = "") -> Dict[str, Any]:
    target = _normalize_phrase(parsed.get("target_desc") or "")
    if not target:
        target = _heuristic_decompose(description, task_type).get("target_desc", "object")
    aliases = [x for x in _dedupe(_as_list(parsed.get("target_aliases")), limit=3) if x != target]
    anchor_primary = _dedupe(_as_list(parsed.get("anchor_primary")), limit=2)
    anchor_support = _dedupe(_as_list(parsed.get("anchor_support")), limit=4)
    room_context = _dedupe(_as_list(parsed.get("room_context")), limit=2)
    target_terms = set(target.split())

    def _not_target(phrase: str) -> bool:
        if phrase == target:
            return False
        words = set(phrase.split())
        if target_terms and words and words.issubset(target_terms):
            return False
        return True

    anchor_primary = [x for x in anchor_primary if _not_target(x)]
    anchor_support = [x for x in anchor_support if _not_target(x) and x not in anchor_primary]
    room_context = [x for x in room_context if _not_target(x)]
    return {
        "ok": True,
        "module": MODULE_NAME,
        "prompt_version": PROMPT_VERSION,
        "description": str(description),
        "task_type": str(task_type),
        "target_desc": target,
        "target_aliases": aliases,
        "anchor_primary": anchor_primary,
        "anchor_support": anchor_support,
        "room_context": room_context,
        "relations": _dedupe(_as_list(parsed.get("relations")), limit=6),
        "raw": str(parsed.get("raw", "")),
        "source": str(parsed.get("source", "vlm")),
        "parse_ok": bool(parsed.get("parse_ok", parsed.get("source", "vlm") == "vlm")),
        "parse_attempts": int(parsed.get("parse_attempts", 0)),
        "error_type": str(parsed.get("error_type", "")),
        "error_message": str(parsed.get("error_message", "")),
        "errors": list(parsed.get("errors", [])) if isinstance(parsed.get("errors", []), list) else [],
    }


def decompose_navigation_text(
    *,
    description: str,
    task_type: str = "",
    cfg: Optional[MqscR1Config] = None,
) -> Dict[str, Any]:
    cfg = cfg or MqscR1Config()
    prompt = build_decomposition_prompt(description, task_type)
    errors: List[Dict[str, str]] = []
    last_raw = ""
    if bool(cfg.use_vlm):
        for attempt in range(max(1, int(cfg.vlm_max_retries))):
            try:
                with no_proxy_env(bool(cfg.vlm_no_proxy)):
                    raw = chat(text=prompt, image_path=None, model=cfg.vlm_model, max_tokens=384)
                last_raw = str(raw)
                parsed = parse_json_object(last_raw)
                parsed.update(
                    {
                        "raw": last_raw,
                        "source": "vlm",
                        "parse_ok": True,
                        "parse_attempts": int(attempt + 1),
                        "error_type": "",
                        "error_message": "",
                        "errors": [],
                    }
                )
                return sanitize_query_spec(parsed, description=description, task_type=task_type)
            except Exception as exc:
                errors.append({"error_type": type(exc).__name__, "error_message": str(exc)})
                if attempt + 1 < max(1, int(cfg.vlm_max_retries)):
                    time.sleep(max(0.0, float(cfg.vlm_retry_sleep_sec)))
    else:
        errors.append({"error_type": "VLMDisabled", "error_message": "MQSC VLM decomposition disabled"})

    fallback = _heuristic_decompose(description, task_type)
    fallback.update(
        {
            "raw": last_raw,
            "source": "heuristic_fallback_after_vlm_error" if bool(cfg.use_vlm) else "heuristic_fallback_vlm_disabled",
            "parse_ok": False,
            "parse_attempts": int(len(errors)),
            "error_type": errors[-1]["error_type"] if errors else "",
            "error_message": errors[-1]["error_message"] if errors else "",
            "errors": errors,
        }
    )
    out = sanitize_query_spec(fallback, description=description, task_type=task_type)
    out["ok"] = bool(cfg.allow_heuristic_decompose)
    return out


def build_role_queries(description: str, query_spec: Mapping[str, Any]) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = [{"role": "full", "text": str(description).strip()}]
    target_parts = [query_spec.get("target_desc", "")]
    target_parts.extend(_as_list(query_spec.get("target_aliases")))
    target_query = ", ".join(_dedupe(target_parts, limit=4))
    if target_query:
        queries.append({"role": "target", "text": target_query})
    anchor_primary = ", ".join(_dedupe(_as_list(query_spec.get("anchor_primary")), limit=2))
    if anchor_primary:
        queries.append({"role": "anchor_primary", "text": anchor_primary})
    anchor_support = ", ".join(_dedupe(_as_list(query_spec.get("anchor_support")), limit=4))
    if anchor_support:
        queries.append({"role": "anchor_support", "text": anchor_support})
    # MQSC-R1 treats room labels as diagnostic-only because the VLE/Stage2
    # object grounding model is object-centric and tends to turn room words
    # into noisy object priors.

    out: List[Dict[str, str]] = []
    seen = set()
    for q in queries:
        text = str(q.get("text", "")).strip()
        role = str(q.get("role", ""))
        key = (role, _normalize_phrase(text))
        if not text or key in seen:
            continue
        seen.add(key)
        out.append({"role": role, "text": text})
    return out


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=float)
    temp = max(float(temperature), 1e-6)
    z = arr / temp
    z = z - np.max(z)
    exp = np.exp(np.clip(z, -60.0, 60.0))
    denom = float(np.sum(exp))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(arr, dtype=float) / max(1, arr.size)
    return exp / denom


def model_box_to_habitat_xyz(box: Sequence[float]) -> np.ndarray:
    b = np.asarray(box, dtype=float).reshape(-1)
    if b.size < 3:
        return np.zeros(3, dtype=float)
    return np.asarray([b[0], b[2], b[1]], dtype=float)


def footprint_xy_and_radius(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    xy = boxes[:, [0, 1]].astype(float)
    if boxes.shape[1] >= 5:
        radius = 0.5 * np.maximum(np.abs(boxes[:, 3]), np.abs(boxes[:, 4]))
    else:
        radius = np.full((boxes.shape[0],), 0.25, dtype=float)
    radius = np.clip(radius, 0.05, 2.5)
    return xy, radius


def pq3d_stage2_object_logits(pq3d_model: Any, sentence: str) -> np.ndarray:
    import torch
    from torch.utils.data import default_collate

    from data.datasets.constant import PromptType
    from data_utils import batch_to_cuda

    rep = getattr(pq3d_model, "representation_manager", None)
    tokenizer = getattr(pq3d_model, "tokenizer", None)
    stage2 = getattr(pq3d_model, "pq3d_stage2", None)
    if rep is None or tokenizer is None or stage2 is None:
        return np.zeros((0,), dtype=np.float64)

    query_box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=np.float32)
    if query_box.size == 0 or query_box.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    query_feat = np.asarray(getattr(rep, "object_feat", np.zeros((0, 768))), dtype=np.float32)
    query_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=np.float32).reshape(-1)
    obj_openvocab_feat = np.asarray(getattr(rep, "open_vocab_feat", np.zeros((0, 768))), dtype=np.float32)
    n = int(query_box.shape[0])
    if query_feat.shape[0] != n or obj_openvocab_feat.shape[0] != n:
        return np.zeros((n,), dtype=np.float64)
    if query_scores.shape[0] < n:
        query_scores = np.pad(query_scores, (0, n - query_scores.shape[0]), constant_values=1.0)
    query_scores = query_scores[:n]

    obj_boxes = torch.from_numpy(query_box).float()
    obj_locs = obj_boxes.clone()
    obj_scores = torch.from_numpy(query_scores).float()
    obj_pad_masks = torch.ones(n, dtype=torch.bool)
    real_obj_pad_masks = torch.ones(n, dtype=torch.bool)
    seg_center = obj_locs.clone()
    seg_pad_masks = obj_pad_masks.clone()
    mv_seg_fts = torch.from_numpy(query_feat).float()
    mv_seg_pad_masks = obj_pad_masks.clone()
    vocab_seg_fts = torch.from_numpy(obj_openvocab_feat).float()
    vocab_seg_pad_masks = obj_pad_masks.clone()

    encoded_input = tokenizer([sentence], add_special_tokens=True, truncation=True)
    tokenized_txt = encoded_input.input_ids[0]
    prompt = torch.FloatTensor(tokenized_txt)
    prompt_pad_masks = torch.ones((len(tokenized_txt))).bool()

    data_dict = {
        "query_pad_masks": obj_pad_masks.clone(),
        "query_locs": obj_locs.clone(),
        "query_scores": obj_scores.clone(),
        "real_obj_pad_masks": real_obj_pad_masks,
        "seg_center": seg_center,
        "seg_pad_masks": seg_pad_masks,
        "mv_seg_fts": mv_seg_fts,
        "mv_seg_pad_masks": mv_seg_pad_masks,
        "vocab_seg_fts": vocab_seg_fts,
        "vocab_seg_pad_masks": vocab_seg_pad_masks,
        "obj_labels": torch.zeros(n, dtype=torch.long),
        "tgt_object_id": torch.LongTensor([]),
        "decision_label": 1,
        "prompt": prompt,
        "prompt_pad_masks": prompt_pad_masks,
        "prompt_type": PromptType.TXT,
    }
    batch = batch_to_cuda(default_collate([data_dict]))
    stage2.eval()
    with torch.no_grad():
        stage2_output = stage2(batch)
    logits = stage2_output["og3d_logits"].detach().float().cpu().numpy().reshape(-1)
    mask = stage2_output["real_obj_pad_masks"].bool().detach().cpu().numpy().reshape(-1)
    return logits[mask].astype(np.float64)


def _footprint_distance(i: int, j: int, xy: np.ndarray, radius: np.ndarray) -> float:
    return max(0.0, float(np.linalg.norm(xy[i] - xy[j])) - float(radius[i]) - float(radius[j]))


def _connected_components(candidate_object_ids: Sequence[int], xy: np.ndarray, radius: np.ndarray, eps: float) -> List[List[int]]:
    ids = [int(x) for x in candidate_object_ids]
    if not ids:
        return []
    parent = {obj_id: obj_id for obj_id in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a_idx, oid_a in enumerate(ids):
        for oid_b in ids[a_idx + 1 :]:
            if _footprint_distance(oid_a, oid_b, xy, radius) <= float(eps):
                union(oid_a, oid_b)
    comps: Dict[int, List[int]] = {}
    for oid in ids:
        comps.setdefault(find(oid), []).append(oid)
    return [sorted(v) for v in comps.values()]


def _noisy_or(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    prod = 1.0
    for value in values:
        p = float(np.clip(value, 0.0, 1.0))
        prod *= 1.0 - p
    return float(1.0 - prod)


def _compactness(object_ids: Sequence[int], xy: np.ndarray, sigma: float) -> float:
    if len(object_ids) <= 1:
        return 1.0
    pts = xy[list(object_ids)]
    center = np.mean(pts, axis=0)
    mean_sq = float(np.mean(np.sum((pts - center) ** 2, axis=1)))
    return float(math.exp(-mean_sq / max(1e-6, float(sigma) ** 2)))


def _relation_support(target_id: int, region_object_ids: Sequence[int], roles_by_object: Mapping[int, set], xy: np.ndarray, radius: np.ndarray) -> float:
    supports: List[float] = []
    for oid in region_object_ids:
        if int(oid) == int(target_id):
            continue
        roles = roles_by_object.get(int(oid), set())
        if not roles.intersection({"anchor_primary", "anchor_support", "full"}):
            continue
        d = _footprint_distance(int(target_id), int(oid), xy, radius)
        supports.append(math.exp(-(d * d) / (2.0 * 1.2 * 1.2)))
    return float(max(supports) if supports else 0.0)


def run_mqsc_r1_refine(
    *,
    sentence: str,
    task_type: str,
    pq3d_model: Any,
    target_position: Sequence[float],
    cfg: Optional[MqscR1Config] = None,
    output_dir: Optional[Path] = None,
    decision_num: Optional[int] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = cfg or MqscR1Config()
    target_before = np.asarray(target_position, dtype=float).reshape(3).copy()
    info: Dict[str, Any] = {
        "ok": True,
        "module": MODULE_NAME,
        "applied": False,
        "reason": "",
        "target_before": target_before.tolist(),
        "target_after": target_before.tolist(),
        "prompt_version": PROMPT_VERSION,
    }
    if context:
        info.update({str(k): v for k, v in context.items()})
    try:
        rep = getattr(pq3d_model, "representation_manager", None)
        boxes = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
        object_scores = np.asarray(getattr(rep, "object_score", np.zeros((0,))), dtype=float).reshape(-1)
        object_counts = np.asarray(getattr(rep, "object_count", np.zeros((0,))), dtype=float).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 3 or boxes.shape[0] == 0:
            info.update({"reason": "no_memory_objects", "n_objects": 0})
            return target_before, info
        n = int(boxes.shape[0])
        if object_scores.shape[0] < n:
            object_scores = np.pad(object_scores, (0, n - object_scores.shape[0]), constant_values=1.0)
        object_scores = object_scores[:n]
        if object_counts.shape[0] < n:
            object_counts = np.pad(object_counts, (0, n - object_counts.shape[0]), constant_values=1.0)
        object_counts = object_counts[:n]

        decomp = decompose_navigation_text(description=sentence, task_type=task_type, cfg=cfg)
        queries = build_role_queries(sentence, decomp)
        info["decomposition"] = decomp
        info["queries"] = queries
        info["r1_policy"] = {
            "exclude_room_context_query": bool(cfg.exclude_room_context_query),
            "excluded_consensus_roles": list(cfg.excluded_consensus_roles),
            "room_context_diagnostic_only": _as_list(decomp.get("room_context")),
            "reason": "stage2_object_grounding_is_object_centric_room_words_are_noisy_priors",
        }
        info["n_objects"] = n
        if len(queries) < 2:
            info["reason"] = "insufficient_role_queries"
            return target_before, info

        xy, radius = footprint_xy_and_radius(boxes)
        top_k = min(max(1, int(cfg.top_k)), n)
        hits: List[Dict[str, Any]] = []
        query_summaries: List[Dict[str, Any]] = []
        logits_by_role: Dict[str, np.ndarray] = {}
        probs_by_role: Dict[str, np.ndarray] = {}

        for q in queries:
            role = str(q["role"])
            text = str(q["text"])
            try:
                logits = pq3d_stage2_object_logits(pq3d_model, text)
            except Exception as exc:
                query_summaries.append(
                    {
                        "role": role,
                        "text": text,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            logits = np.asarray(logits, dtype=float).reshape(-1)[:n]
            if logits.size != n:
                query_summaries.append(
                    {
                        "role": role,
                        "text": text,
                        "ok": False,
                        "error_type": "LogitShapeMismatch",
                        "error_message": f"got={logits.size} expected={n}",
                    }
                )
                continue
            probs = _softmax(logits, cfg.temperature)
            logits_by_role[role] = logits
            probs_by_role[role] = probs
            order = np.argsort(-probs)[:top_k]
            query_summaries.append(
                {
                    "role": role,
                    "text": text,
                    "ok": True,
                    "top_indices": [int(x) for x in order],
                    "top_probs": [float(probs[int(x)]) for x in order],
                    "top_logits": [float(logits[int(x)]) for x in order],
                }
            )
            for rank, oid in enumerate(order):
                oid_i = int(oid)
                hits.append(
                    {
                        "object_id": oid_i,
                        "rank": int(rank + 1),
                        "role": role,
                        "query": text,
                        "prob": float(probs[oid_i]),
                        "logit": float(logits[oid_i]),
                        "xy": [float(xy[oid_i, 0]), float(xy[oid_i, 1])],
                        "footprint_radius": float(radius[oid_i]),
                        "box_model_xzydxdzdy": boxes[oid_i].tolist(),
                        "center_habitat_xyz": model_box_to_habitat_xyz(boxes[oid_i]).tolist(),
                        "merged_object_score": float(object_scores[oid_i]),
                        "object_count": float(object_counts[oid_i]),
                    }
                )
        info["query_summaries"] = query_summaries
        info["candidate_hits"] = hits
        if "target" not in probs_by_role:
            info["reason"] = "target_query_failed"
            return target_before, info
        if not hits:
            info["reason"] = "no_candidate_hits"
            return target_before, info

        candidate_ids = sorted({int(h["object_id"]) for h in hits})
        comps = _connected_components(candidate_ids, xy, radius, float(cfg.cluster_eps))
        hit_probs_by_role_obj: Dict[str, Dict[int, float]] = {}
        roles_by_object: Dict[int, set] = {}
        for hit in hits:
            role = str(hit["role"])
            oid = int(hit["object_id"])
            roles_by_object.setdefault(oid, set()).add(role)
            hit_probs_by_role_obj.setdefault(role, {})
            hit_probs_by_role_obj[role][oid] = max(float(hit["prob"]), hit_probs_by_role_obj[role].get(oid, 0.0))

        required_roles = ["full", "target"]
        if decomp.get("anchor_primary"):
            required_roles.append("anchor_primary")
        if decomp.get("anchor_support"):
            required_roles.append("anchor_support")
        excluded_roles = set(str(x) for x in getattr(cfg, "excluded_consensus_roles", ()))
        required_roles = [r for r in required_roles if r not in excluded_roles]
        required_roles = [r for r in required_roles if any(q["role"] == r for q in queries)]

        regions: List[Dict[str, Any]] = []
        for ridx, obj_ids in enumerate(comps):
            role_evidence: Dict[str, float] = {}
            represented = 0
            for role in required_roles:
                vals = [hit_probs_by_role_obj.get(role, {}).get(int(oid), 0.0) for oid in obj_ids]
                p_region = _noisy_or(vals)
                role_evidence[role] = p_region
                if p_region > 1e-9:
                    represented += 1
            coverage = float(represented / max(1, len(required_roles)))
            compact = _compactness(obj_ids, xy, sigma=max(float(cfg.cluster_eps), 0.5))
            score = 0.0
            for role, value in role_evidence.items():
                score += float(cfg.role_weights.get(role, 0.5)) * math.log(1e-6 + float(value))
            score += 1.25 * coverage + 0.25 * compact
            role_counts: Dict[str, int] = {}
            for oid in obj_ids:
                for role in roles_by_object.get(int(oid), set()):
                    role_counts[role] = role_counts.get(role, 0) + 1
            if coverage < 0.5 and max(role_counts.values() or [0]) >= 3:
                score -= 0.2
            regions.append(
                {
                    "region_id": int(ridx),
                    "object_ids": [int(x) for x in obj_ids],
                    "score": float(score),
                    "coverage": float(coverage),
                    "compactness": float(compact),
                    "role_evidence": {k: float(v) for k, v in role_evidence.items()},
                    "role_counts": {k: int(v) for k, v in sorted(role_counts.items())},
                }
            )
        regions.sort(key=lambda x: float(x["score"]), reverse=True)
        info["regions"] = regions
        if not regions:
            info["reason"] = "no_regions"
            return target_before, info

        best = regions[0]
        second_score = float(regions[1]["score"]) if len(regions) > 1 else float("-inf")
        margin = float(best["score"] - second_score) if np.isfinite(second_score) else float("inf")
        best_ids = [int(x) for x in best["object_ids"]]
        target_probs = probs_by_role["target"]
        full_probs = probs_by_role.get("full", np.zeros_like(target_probs))
        baseline_idx = int((getattr(pq3d_model, "last_decision_aux", {}) or {}).get("real_object_decision_idx", -1))

        target_pool = [oid for oid in best_ids if "target" in roles_by_object.get(int(oid), set())]
        if not target_pool:
            info.update({"reason": "best_region_has_no_target_role_object", "best_region": best, "region_margin": margin})
            return target_before, info

        selected_records: List[Dict[str, Any]] = []
        for oid in target_pool:
            rel = _relation_support(int(oid), best_ids, roles_by_object, xy, radius)
            merged = float(np.clip(object_scores[int(oid)], 1e-6, None))
            count = float(max(1.0, object_counts[int(oid)]))
            target_score = (
                1.45 * math.log(1e-6 + float(target_probs[int(oid)]))
                + 0.75 * math.log(1e-6 + float(full_probs[int(oid)]))
                + 0.45 * rel
                + 0.05 * math.log(merged)
                + 0.03 * math.log(count)
            )
            selected_records.append(
                {
                    "object_id": int(oid),
                    "target_score": float(target_score),
                    "target_prob": float(target_probs[int(oid)]),
                    "full_prob": float(full_probs[int(oid)]),
                    "relation_support": float(rel),
                    "merged_object_score": float(merged),
                    "object_count": float(count),
                    "center_habitat_xyz": model_box_to_habitat_xyz(boxes[int(oid)]).tolist(),
                }
            )
        selected_records.sort(key=lambda x: float(x["target_score"]), reverse=True)
        selected = selected_records[0]
        selected_idx = int(selected["object_id"])
        baseline_target_score = None
        if 0 <= baseline_idx < n:
            baseline_rel = _relation_support(baseline_idx, best_ids, roles_by_object, xy, radius)
            baseline_target_score = float(
                1.45 * math.log(1e-6 + float(target_probs[baseline_idx]))
                + 0.75 * math.log(1e-6 + float(full_probs[baseline_idx]))
                + 0.45 * baseline_rel
            )
        selected_gain = (
            float(selected["target_score"]) - float(baseline_target_score)
            if baseline_target_score is not None
            else float("inf")
        )
        selected_xyz = model_box_to_habitat_xyz(boxes[selected_idx])
        info.update(
            {
                "best_region": best,
                "region_margin": float(margin),
                "selected_candidates": selected_records,
                "selected_object_index": int(selected_idx),
                "baseline_object_index": int(baseline_idx),
                "baseline_target_score": baseline_target_score,
                "selected_gain": float(selected_gain),
                "selected_target_position": selected_xyz.tolist(),
                "target_confidence": float(selected["target_prob"]),
            }
        )

        fail_reasons: List[str] = []
        if float(best["coverage"]) < float(cfg.min_region_coverage):
            fail_reasons.append("coverage_below_threshold")
        if float(selected["target_prob"]) < float(cfg.min_target_prob):
            fail_reasons.append("target_prob_below_threshold")
        if np.isfinite(margin) and margin < float(cfg.min_region_margin):
            fail_reasons.append("region_margin_below_threshold")
        if selected_gain < float(cfg.min_selected_gain):
            fail_reasons.append("selected_gain_below_threshold")
        if selected_idx == baseline_idx:
            fail_reasons.append("selected_matches_baseline")

        if fail_reasons:
            info["reason"] = ",".join(fail_reasons)
            return target_before, info

        info.update(
            {
                "applied": True,
                "reason": "mqsc_selected_higher_consensus_target",
                "target_after": selected_xyz.tolist(),
            }
        )
        return selected_xyz.astype(float), info
    except Exception as exc:
        info.update(
            {
                "ok": False,
                "applied": False,
                "reason": "mqsc_exception_fallback_baseline",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "target_after": target_before.tolist(),
            }
        )
        return target_before, info
    finally:
        if output_dir is not None and bool(cfg.write_debug_json):
            try:
                mqsc_dir = Path(output_dir) / MODULE_NAME
                mqsc_dir.mkdir(parents=True, exist_ok=True)
                dec = "unknown" if decision_num is None else f"{int(decision_num):03d}"
                with open(mqsc_dir / f"dec_{dec}_mqsc_r1.json", "w", encoding="utf-8") as f:
                    json.dump(_jsonable(info), f, ensure_ascii=False, indent=2)
            except Exception:
                pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, set):
        return sorted(_jsonable(x) for x in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "MODULE_NAME",
    "PROMPT_VERSION",
    "MqscR1Config",
    "build_decomposition_prompt",
    "build_role_queries",
    "decompose_navigation_text",
    "model_box_to_habitat_xyz",
    "pq3d_stage2_object_logits",
    "run_mqsc_r1_refine",
    "sanitize_query_spec",
]
