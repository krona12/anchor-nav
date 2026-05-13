from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from vlm.client import DEFAULT_MODEL, chat


MODULE_NAME = "step"
PROMPT_VERSION = "step_anchor_decompose_v4_no_target_query"


@dataclass
class StepConfig:
    vlm_model: str = DEFAULT_MODEL
    use_vlm: bool = True
    no_proxy: bool = True
    max_retries: int = 3
    retry_sleep_sec: float = 2.0
    max_object_anchors: int = 5
    max_attribute_anchors: int = 4
    allow_heuristic_prestep: bool = False


_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_GENERIC_WORDS = {
    "a",
    "an",
    "and",
    "area",
    "for",
    "in",
    "near",
    "of",
    "or",
    "room",
    "space",
    "the",
    "to",
    "with",
}


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


def normalize_phrase(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"STEP VLM response is not a JSON object: {raw!r}")
    return parsed


def _as_clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        phrase = normalize_phrase(item)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def _target_head_terms(target_category: str) -> List[str]:
    category = normalize_phrase(target_category)
    if not category:
        return []
    terms: List[str] = []
    chunks = re.split(r"\b(?:and|or)\b|[,&+]", category)
    for chunk in chunks:
        words = [w for w in normalize_phrase(chunk).split() if w not in _GENERIC_WORDS and len(w) > 1]
        if words:
            terms.append(words[-1])
    if not terms:
        words = [w for w in category.split() if w not in _GENERIC_WORDS and len(w) > 1]
        if words:
            terms.append(words[-1])
    seen = set()
    out = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def phrase_mentions_target(phrase: str, target_category: str) -> bool:
    text = normalize_phrase(phrase)
    category = normalize_phrase(target_category)
    if not text or not category:
        return False
    if category and re.search(rf"\b{re.escape(category)}s?\b", text):
        return True
    for term in _target_head_terms(category):
        if len(term) <= 2:
            continue
        if re.search(rf"\b{re.escape(term)}s?\b", text):
            return True
    return False


def _filter_safe_phrases(phrases: Iterable[str], target_category: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for phrase in phrases:
        clean = normalize_phrase(phrase)
        if not clean or clean in seen:
            continue
        if phrase_mentions_target(clean, target_category):
            continue
        seen.add(clean)
        out.append(clean)
    return out


def compose_anchor_query(
    *,
    room_anchors: Sequence[str],
    object_anchors: Sequence[str],
    attribute_anchors: Sequence[str],
    max_object_anchors: int = 5,
    max_attribute_anchors: int = 4,
) -> str:
    parts: List[str] = []
    for bucket, limit in (
        (room_anchors, 2),
        (object_anchors, max(0, int(max_object_anchors))),
        (attribute_anchors, max(0, int(max_attribute_anchors))),
    ):
        for phrase in bucket[:limit]:
            clean = normalize_phrase(phrase)
            if clean and clean not in parts:
                parts.append(clean)
    return ", ".join(parts)


def build_step_decomposition_prompt(*, description: str, task_level: str, target_category: str) -> str:
    return (
        "You are decomposing RefHM3D navigation text for a two-step navigation experiment.\n"
        "Step 1 uses an anchor-only query for exactly one navigation decision. Step 2 uses the full original description.\n"
        "The step-1 query must help move toward the target area without mentioning the target object.\n\n"
        "Inputs: target_category, task_level, navigation_text.\n\n"
        "Return fields:\n"
        "- target_desc: main target object phrase.\n"
        "- object_anchors: safe nearby objects/furniture/fixtures that are NOT the target.\n"
        "- room_anchors: safe room/region names.\n"
        "- attribute_anchors: safe visual/spatial area attributes, not target appearance.\n"
        "- anchor_query: a short comma-separated noun phrase list made from room_anchors + object_anchors + attribute_anchors.\n"
        "- should_prestep: true when anchor_query is non-empty, including room-only queries like \"bedroom\".\n\n"
        "Hard constraints:\n"
        "1. Never include target_category, its plural/singular form, or target_desc words in object_anchors, attribute_anchors, or anchor_query.\n"
        "2. Delete every source phrase that names/describes the target itself, even inside a region inventory after \"that has\".\n"
        "3. anchor_query must NOT be a question or instruction. No \"what is\", \"find\", \"go to\", \"look for\".\n"
        "4. anchor_query should be concise: prefer room first, then up to 5 strongest object anchors, then up to 4 area attributes.\n"
        "5. Use lowercase strings. Return strict JSON only.\n\n"
        "Schema exactly:\n"
        "{\"target_desc\": string, \"object_anchors\": [string], \"room_anchors\": [string], "
        "\"attribute_anchors\": [string], \"anchor_query\": string, \"should_prestep\": boolean}\n\n"
        f"target_category: {target_category}\n"
        f"task_level: {task_level}\n"
        f"navigation_text: {description}"
    )


def _fallback_decomposition(description: str, task_level: str, target_category: str) -> Dict[str, Any]:
    desc = str(description or "").strip()
    target = normalize_phrase(target_category)
    room_anchors: List[str] = []
    object_anchors: List[str] = []
    attribute_anchors: List[str] = []

    region_match = re.match(r"^\s*(.+?)\s+in\s+the\s+(.+?)\s+that\s+has\s+(.+)$", desc, flags=re.IGNORECASE)
    room_match = re.match(r"^\s*(.+?)\s+in\s+the\s+(.+?)\s*$", desc, flags=re.IGNORECASE)
    if region_match:
        if not target:
            target = normalize_phrase(region_match.group(1))
        room_anchors.append(region_match.group(2))
        chunks = re.split(r",|;|\band\b", region_match.group(3), flags=re.IGNORECASE)
        object_anchors.extend(chunks)
    elif room_match:
        if not target:
            target = normalize_phrase(room_match.group(1))
        room_anchors.append(room_match.group(2))
    else:
        if not target:
            before_context = re.split(r"\bin\b|\bwith\b|\bnear\b|\bbeside\b", desc, maxsplit=1, flags=re.IGNORECASE)[0]
            target = normalize_phrase(before_context)
        room_bits = re.findall(r"\bin\s+([a-z][a-z\s\-]+?)\s+with\b", desc, flags=re.IGNORECASE)
        if not room_bits:
            room_bits = re.findall(r"\bin\s+([a-z][a-z\s\-]+)$", desc, flags=re.IGNORECASE)
        room_anchors.extend(room_bits[:1])
        tail = re.split(r"\bwith\b|\bnear\b|\bbeside\b|\bbelow\b|\babove\b|\bbetween\b|\bby\b|,", desc, flags=re.IGNORECASE)
        object_anchors.extend(tail[1:])

    return {
        "target_desc": target,
        "object_anchors": object_anchors,
        "room_anchors": room_anchors,
        "attribute_anchors": attribute_anchors,
        "raw": "",
        "source": "heuristic_fallback",
    }


def sanitize_decomposition(
    parsed: Dict[str, Any],
    *,
    description: str,
    task_level: str,
    target_category: str,
    cfg: Optional[StepConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or StepConfig()
    target_desc = normalize_phrase(parsed.get("target_desc") or target_category)
    object_anchors = _filter_safe_phrases(_as_clean_list(parsed.get("object_anchors")), target_category)
    room_anchors = _filter_safe_phrases(_as_clean_list(parsed.get("room_anchors")), target_category)
    attribute_anchors = _filter_safe_phrases(_as_clean_list(parsed.get("attribute_anchors")), target_category)
    anchor_query = compose_anchor_query(
        room_anchors=room_anchors,
        object_anchors=object_anchors,
        attribute_anchors=attribute_anchors,
        max_object_anchors=cfg.max_object_anchors,
        max_attribute_anchors=cfg.max_attribute_anchors,
    )
    return {
        "ok": True,
        "module": MODULE_NAME,
        "prompt_version": PROMPT_VERSION,
        "task_level": str(task_level),
        "description": str(description),
        "target_category": normalize_phrase(target_category),
        "target_desc": target_desc,
        "object_anchors": object_anchors,
        "room_anchors": room_anchors,
        "attribute_anchors": attribute_anchors,
        "anchor_query": anchor_query,
        "should_prestep": bool(anchor_query),
        "raw_anchor_query": normalize_phrase(parsed.get("anchor_query", "")),
        "raw": str(parsed.get("raw", "")),
        "source": str(parsed.get("source", "vlm")),
        "error": str(parsed.get("error", "")),
        "parse_ok": bool(parsed.get("parse_ok", parsed.get("source", "vlm") == "vlm")),
        "parse_attempts": int(parsed.get("parse_attempts", 0)),
        "image_path": parsed.get("image_path", None),
        "error_type": str(parsed.get("error_type", "")),
        "error_message": str(parsed.get("error_message", "")),
        "errors": parsed.get("errors", []),
    }


def decompose_navigation_description(
    *,
    description: str,
    task_level: str,
    target_category: str,
    cfg: Optional[StepConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or StepConfig()
    prompt = build_step_decomposition_prompt(
        description=description,
        task_level=task_level,
        target_category=target_category,
    )
    error_records: List[Dict[str, str]] = []
    last_raw = ""
    if bool(cfg.use_vlm):
        for attempt in range(max(1, int(cfg.max_retries))):
            try:
                with no_proxy_env(bool(cfg.no_proxy)):
                    raw = chat(text=prompt, image_path=None, model=cfg.vlm_model, max_tokens=256)
                last_raw = str(raw)
                parsed = _parse_json_object(raw)
                parsed["raw"] = raw
                parsed["source"] = "vlm"
                parsed["parse_ok"] = True
                parsed["parse_attempts"] = int(attempt + 1)
                parsed["image_path"] = None
                parsed["error_type"] = ""
                parsed["error_message"] = ""
                parsed["errors"] = []
                out = sanitize_decomposition(
                    parsed,
                    description=description,
                    task_level=task_level,
                    target_category=target_category,
                    cfg=cfg,
                )
                out["vlm_attempts"] = int(attempt + 1)
                return out
            except Exception as exc:
                error_records.append({"error_type": type(exc).__name__, "error_message": str(exc)})
                if attempt + 1 < max(1, int(cfg.max_retries)):
                    time.sleep(max(0.0, float(cfg.retry_sleep_sec)))
    else:
        error_records.append({"error_type": "VLMDisabled", "error_message": "STEP VLM decomposition disabled"})

    fallback = _fallback_decomposition(description, task_level, target_category)
    fallback["source"] = "heuristic_fallback_after_vlm_error" if bool(cfg.use_vlm) else "heuristic_fallback_vlm_disabled"
    fallback["raw"] = last_raw
    fallback["parse_ok"] = False
    fallback["parse_attempts"] = int(len(error_records))
    fallback["image_path"] = None
    fallback["errors"] = error_records
    fallback["error_type"] = error_records[-1]["error_type"] if error_records else ""
    fallback["error_message"] = error_records[-1]["error_message"] if error_records else ""
    if error_records:
        fallback["error"] = " | ".join(f"{e['error_type']}: {e['error_message']}" for e in error_records)
    out = sanitize_decomposition(
        fallback,
        description=description,
        task_level=task_level,
        target_category=target_category,
        cfg=cfg,
    )
    out["ok"] = (not bool(error_records)) or bool(cfg.allow_heuristic_prestep)
    out["vlm_attempts"] = int(len(error_records))
    out["fallback_anchor_query"] = out.get("anchor_query", "")
    if not bool(cfg.allow_heuristic_prestep):
        out["anchor_query"] = ""
        out["should_prestep"] = False
    return out
