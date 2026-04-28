from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from anchor_nav.posnode import build_query_fn_from_pq3d_stage2
from vlm.client import chat


@dataclass
class VfvDecompose:
    target_desc: str
    anchor_desc: str
    raw: str
    parse_ok: bool = True


def _normalize_phrase(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_balanced_brace_object(text: str) -> Optional[str]:
    """从任意文本中提取第一个花括号平衡的 JSON 对象子串（粗略处理字符串内括号）。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    quote: Optional[str] = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif quote and ch == quote:
                in_str = False
                quote = None
            continue
        if ch == '"' or ch == "'":
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


def _try_parse_json_obj(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """解析 VLM 返回中的 JSON 对象；失败返回 None（不回溯抛错）。"""
    text = (raw or "").strip()
    if not text:
        return None
    candidates: List[str] = []
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())
    bal = _extract_balanced_brace_object(text)
    if bal:
        candidates.append(bal)
    if text.lstrip().startswith("{"):
        candidates.append(text)
    # 去重保序
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_json_obj(raw: str) -> Dict[str, Any]:
    """兼容旧调用：解析失败时返回空 dict。"""
    out = _try_parse_json_obj(raw)
    return out if out is not None else {}


def decompose_target_anchor(
    description: str,
    vlm_model: str,
    *,
    max_attempts: int = 3,
) -> VfvDecompose:
    prompt = (
        "Decompose the navigation description into target and ONE explicit anchor.\n"
        "Rules:\n"
        "1) target_desc: object to find, short noun phrase with key modifiers.\n"
        "2) anchor_desc: the most visible and explicit anchor object near/related to target.\n"
        "3) Do not use room words as anchor (e.g., bedroom, bathroom).\n"
        "4) If no explicit anchor, use empty string.\n"
        "Return strict JSON only: {\"target_desc\":\"...\",\"anchor_desc\":\"...\"}\n\n"
        f"Description: {description}"
    )
    last_raw = ""
    for attempt in range(max(1, int(max_attempts))):
        try:
            raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=128)
        except Exception:
            raw = ""
        last_raw = str(raw or "")
        parsed = _try_parse_json_obj(last_raw)
        if parsed is None:
            continue
        target_desc = _normalize_phrase(str(parsed.get("target_desc", "")))
        anchor_desc = _normalize_phrase(str(parsed.get("anchor_desc", "")))
        if target_desc:
            return VfvDecompose(
                target_desc=target_desc,
                anchor_desc=anchor_desc,
                raw=last_raw,
                parse_ok=True,
            )
    # 回退：整句描述作为 target，避免任务中断；phase2 无 anchor 时 select_anchor 会跳过
    fb = _normalize_phrase(description)
    if not fb:
        fb = _normalize_phrase(str(description or "object")) or "object"
    return VfvDecompose(
        target_desc=fb,
        anchor_desc="",
        raw=(last_raw + "|decompose_fallback_parse_failed") if last_raw else "decompose_fallback_parse_failed",
        parse_ok=False,
    )


def verify_description_visible(
    *,
    description: str,
    image_path: str,
    vlm_model: str,
) -> Dict[str, Any]:
    prompt = (
        "Check whether the full navigation description is visually supported by this panorama image.\n"
        "Be conservative: if uncertain, answer false.\n"
        "Return strict JSON only: "
        "{\"visible\": true/false, \"confidence\": \"high|medium|low\", \"reason\": \"...\"}\n\n"
        f"Description: {description}"
    )
    raw = chat(text=prompt, image_path=image_path, model=vlm_model, max_tokens=128)
    parsed = _try_parse_json_obj(raw)
    if parsed is None:
        return {
            "visible": False,
            "confidence": "low",
            "reason": "verify_parse_failed",
            "raw": str(raw),
        }
    return {
        "visible": bool(parsed.get("visible", False)),
        "confidence": str(parsed.get("confidence", "")),
        "reason": str(parsed.get("reason", "")),
        "raw": str(raw),
    }


def select_anchor_object(
    *,
    anchor_desc: str,
    query_fn: Any,
    rep: Any,
    top_k: int = 16,
) -> Dict[str, Any]:
    anchor = _normalize_phrase(anchor_desc)
    if not anchor:
        return {"ok": False, "reason": "empty_anchor_desc"}
    topk = list(query_fn(anchor, int(top_k)))
    if len(topk) == 0:
        return {"ok": False, "reason": "empty_anchor_query"}
    idx = int(topk[0][0])
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[1] < 3 or idx < 0 or idx >= int(box.shape[0]):
        return {"ok": False, "reason": "invalid_anchor_index", "anchor_index": int(idx)}
    anchor_xyz = np.asarray(box[idx, :3], dtype=float).reshape(3).copy()
    anchor_xyz[[1, 2]] = anchor_xyz[[2, 1]]
    return {
        "ok": True,
        "anchor_index": int(idx),
        "anchor_position": anchor_xyz.tolist(),
        "anchor_topk": [{"object_index": int(i), "score": float(s)} for i, s in topk],
    }


__all__ = [
    "VfvDecompose",
    "build_query_fn_from_pq3d_stage2",
    "decompose_target_anchor",
    "verify_description_visible",
    "select_anchor_object",
]

