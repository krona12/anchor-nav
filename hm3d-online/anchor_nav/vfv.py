from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from anchor_nav.posnode import build_query_fn_from_pq3d_stage2
from vlm.client import chat


@dataclass
class VfvDecompose:
    target_desc: str
    anchor_desc: str
    raw: str
    anchor_descs: List[str] = field(default_factory=list)
    parse_ok: bool = True


def _normalize_phrase(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def dedupe_anchor_phrases(phrases: Sequence[str]) -> List[str]:
    """规范化并去重，保序；语义过滤依赖 decompose 的 VLM prompt。"""
    out: List[str] = []
    seen: set = set()
    for ph in phrases:
        n = _normalize_phrase(str(ph))
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _extract_balanced_brace_object(text: str) -> Optional[str]:
    """从任意文本中提取第一个花括号平衡的 JSON 对象子串（粗略处理字符串内括号）。

    注：非标准 JSON（如单引号）若值内含未转义撇号（it's），可能破坏字符串状态；
    上层另有 _try_parse_json_obj 的多候选兜底。
    """
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
    # 规则写全在 prompt 中，由 VLM 遵守；此处仅做规范化去重
    prompt = (
        "Decompose the navigation task into a target and OBJECT anchors (for re-querying a detector).\n"
        "Rules:\n"
        "1) target_desc: the primary object to find; short noun phrase with key modifiers.\n"
        "2) anchor_descs: 0 to 3 DISTINCTIVE object anchors (furniture, appliances, lamps, decor, "
        "fixtures that identify a location). Order by usefulness for navigation (best first).\n"
        "3) ENVIRONMENT / STRUCTURE — DO NOT use as an anchor (single token OR short phrase that is only structure):\n"
        "   - Single-word bans: door, doors, wall, walls, ceiling, ceilings, floor, floors, ground, "
        "window, windows, hallway, hallways, corridor, corridors, room, rooms, entryway, passage, "
        "stair, stairs, staircase (use compound objects instead, see rule 6).\n"
        "   - Phrases of 2–3 words whose LAST word is one of the above environment tokens "
        "(e.g. bad: 'white door', 'wood floor', 'bathroom door'; ok to omit such anchors).\n"
        "   - Two-word phrases whose FIRST word is one of those environment tokens "
        "(e.g. bad: 'window sill') unless the phrase clearly names a distinctive object.\n"
        "4) ROOM / AREA LABELS — Never use alone as an anchor (e.g. bedroom, kitchen, dining room, bathroom).\n"
        "5) HIGHLY AMBIGUOUS OBJECT HEAD NOUNS — Do not output an anchor whose meaning collapses to "
        "these categories alone OR whose LAST word (head noun) is one of these "
        "(even in longer phrases): plant, plants, vase, vases, bin, bins, mat, mats, rug, rugs, "
        "pillow, pillows, cushion, cushions, towel, towels, curtain, curtains, decoration, decorations.\n"
        "   Examples to EXCLUDE: 'decorative plant', 'two white pillows', 'blue towels'. "
        "Prefer anchors where the last noun is distinctive (e.g. 'oak bookshelf').\n"
        "6) STAIRS — Bad alone ('stairs'); ALLOW compounds when distinctive: "
        "'stair railing', 'staircase landing', 'metal banister'.\n"
        "7) Prefer concrete instances ('blue armchair', 'kitchen island') over vague references.\n"
        "8) Return anchor_descs only; do not repeat the same anchor twice. Do not use bare surfaces as anchors.\n"
        "Return strict JSON only: {\"target_desc\":\"...\",\"anchor_descs\":[\"...\",\"...\"]}\n\n"
        f"Description: {description}"
    )
    last_raw = ""
    for attempt in range(max(1, int(max_attempts))):
        try:
            raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=256)
        except Exception:
            raw = ""
        last_raw = str(raw or "")
        parsed = _try_parse_json_obj(last_raw)
        if parsed is None:
            continue
        target_desc = _normalize_phrase(str(parsed.get("target_desc", "")))
        raw_list: List[str] = []
        ad = parsed.get("anchor_descs")
        if isinstance(ad, list):
            raw_list.extend(str(x) for x in ad if str(x).strip())
        # 仅当新字段未给出锚点时再读 anchor_desc
        if not raw_list:
            legacy = str(parsed.get("anchor_desc", "") or "").strip()
            if legacy:
                raw_list.append(legacy)
        anchor_descs = dedupe_anchor_phrases(raw_list)
        anchor_primary = anchor_descs[0] if anchor_descs else ""
        if target_desc:
            return VfvDecompose(
                target_desc=target_desc,
                anchor_desc=anchor_primary,
                anchor_descs=list(anchor_descs),
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
        anchor_descs=[],
        raw=(last_raw + "|decompose_fallback_parse_failed") if last_raw else "decompose_fallback_parse_failed",
        parse_ok=False,
    )


def verify_description_visible(
    *,
    description: str,
    image_path: str,
    vlm_model: str,
    target_desc: str = "",
    anchor_hints: Optional[Sequence[str]] = None,
    max_parse_attempts: int = 2,
) -> Dict[str, Any]:
    hints = dedupe_anchor_phrases(list(anchor_hints or []))
    td = _normalize_phrase(target_desc)
    hint_lines = "\n".join(f"- {h}" for h in hints) if hints else "(none)"
    prompt = (
        "You are given a navigation task and one stitched panorama from the agent.\n"
        "Decide ONLY whether full_match is true: i.e. the primary TARGET object is adequately "
        "visible so that no extra navigation pivot (Phase2) is needed.\n\n"
        "Rules for full_match:\n"
        "- full_match=true ONLY if the PRIMARY TARGET (see target phrase / task) is clearly and "
        "unambiguously visible, identifiable at close range (roughly within 1–2 m / clearly "
        "reachable in the scene), and consistent with the task wording.\n"
        "- If the target is far, heavily occluded, ambiguous, only partially seen, or you are unsure, "
        "set full_match=false.\n"
        "- The anchor list below is INFORMATIONAL ONLY (scene context for you). It MUST NOT be used "
        "to set full_match=true. Seeing anchors alone (e.g. vanity, curtain, bed) without the target "
        "clearly meeting the criteria above always means full_match=false.\n"
        "- Be conservative: when in doubt, full_match=false and use confidence=low.\n\n"
        "Diagnostic fields (for logging only; they do NOT replace full_match): still report "
        "strong_anchor_match=true if a distinctive anchor from the hints is clearly visible even "
        "when the target is not; set matched_anchor to a short phrase or empty.\n\n"
        "Return strict JSON only:\n"
        "{\"full_match\": true/false, \"strong_anchor_match\": true/false, "
        "\"matched_anchor\": \"short phrase or empty\", "
        "\"confidence\": \"high|medium|low\", \"reason\": \"...\"}\n\n"
        f"Task description: {description}\n"
        f"Target phrase (primary object to judge): {td or '(extract from description)'}\n"
        f"Anchor hints (informational only, do not use alone to pass):\n{hint_lines}\n"
    )
    parsed: Optional[Dict[str, Any]] = None
    raw = ""
    attempts_used = 0
    for attempt in range(max(1, int(max_parse_attempts))):
        attempts_used = attempt + 1
        try:
            raw = chat(text=prompt, image_path=image_path, model=vlm_model, max_tokens=256)
        except Exception:
            raw = ""
        parsed = _try_parse_json_obj(raw)
        if parsed is not None:
            break
    if parsed is None:
        return {
            "visible": False,
            "full_match": False,
            "strong_anchor_match": False,
            "full_match_ok": False,
            "anchor_ok": False,
            "matched_anchor": "",
            "confidence": "low",
            "reason": "verify_parse_failed",
            "raw": str(raw),
            "parse_attempts": int(attempts_used),
        }
    full_match = bool(parsed.get("full_match", False))
    strong = bool(parsed.get("strong_anchor_match", False))
    conf = str(parsed.get("confidence", "low") or "low").lower()
    matched = str(parsed.get("matched_anchor", "") or "").strip()
    # 兼容旧字段 visible
    if "full_match" not in parsed and "strong_anchor_match" not in parsed:
        full_match = bool(parsed.get("visible", False))
        strong = False
    # visible：仅 full_match + 中高置信；strong_anchor_match 仅作诊断，不再用来 pass Phase1 / 压制 Phase2
    full_match_ok = bool(full_match and conf in ("high", "medium"))
    anchor_ok = bool(strong and conf in ("high", "medium"))
    visible = bool(full_match_ok)
    return {
        "visible": visible,
        "full_match": full_match,
        "strong_anchor_match": strong,
        "full_match_ok": full_match_ok,
        "anchor_ok": anchor_ok,
        "matched_anchor": matched,
        "confidence": conf,
        "reason": str(parsed.get("reason", "")),
        "raw": str(raw),
        "parse_attempts": int(attempts_used),
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


def select_best_anchor_object(
    *,
    anchor_descs: Sequence[str],
    query_fn: Any,
    rep: Any,
    top_k: int = 16,
) -> Dict[str, Any]:
    """
    依次尝试多个锚点 PQ3D 查询；取**第一个**查询成功（有有效检测）的锚点。
    不在不同查询间比较 Stage2 raw score（各查询 softmax 分母不同，不可比）。
    """
    ordered = dedupe_anchor_phrases(list(anchor_descs))
    tried: List[Dict[str, Any]] = []

    for ad in ordered:
        res = select_anchor_object(anchor_desc=ad, query_fn=query_fn, rep=rep, top_k=int(top_k))
        if not res.get("ok"):
            tried.append({"anchor_desc": ad, **{k: v for k, v in res.items() if k != "tried"}})
            continue
        top0 = res.get("anchor_topk") or []
        pq = float(top0[0]["score"]) if top0 else 0.0
        return {**res, "anchor_desc_used": ad, "pq_top1_score": pq}

    return {
        "ok": False,
        "reason": "no_anchor_query_succeeded",
        "tried": tried,
    }


__all__ = [
    "VfvDecompose",
    "build_query_fn_from_pq3d_stage2",
    "decompose_target_anchor",
    "verify_description_visible",
    "select_anchor_object",
    "select_best_anchor_object",
    "dedupe_anchor_phrases",
]

