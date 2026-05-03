from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from anchor_nav.posnode import build_query_fn_from_pq3d_stage2
from vlm.client import chat


# 禁用作为 PQ3D 问询锚点的泛环境词（门/墙/地面等），优先保留具体物体。
# 易与场景中多处实例混淆的词（参考 exp_vfv_analysis：如 plant 导致 Phase2 错位）
_AMBIGUOUS_OBJECT_LEXEMES = frozenset(
    {
        "plant",
        "plants",
        "vase",
        "vases",
        "bin",
        "bins",
        "mat",
        "mats",
        "rug",
        "rugs",
        "pillow",
        "pillows",
        "cushion",
        "cushions",
        "towel",
        "towels",
        "curtain",
        "curtains",
        "decoration",
        "decorations",
        # 单独出现时缺乏区分度；复合短语（如 stair railing）末词通常不是这些词
        "stair",
        "stairs",
        "staircase",
    }
)

_GENERIC_ENV_TOKENS = frozenset(
    {
        "door",
        "doors",
        "wall",
        "walls",
        "ceiling",
        "ceilings",
        "floor",
        "floors",
        "ground",
        "window",
        "windows",
        "hallway",
        "hallways",
        "corridor",
        "corridors",
        "room",
        "rooms",
        "entryway",
        "passage",
    }
)


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


def _is_ambiguous_object_lexeme(phrase: str) -> bool:
    """末尾词或单词短语为泛指的物体类别时，不宜单独作为 Phase2 PQ3D 锚点。"""
    p = _normalize_phrase(phrase)
    if not p:
        return True
    parts = p.split()
    if len(parts) == 1:
        return parts[0] in _AMBIGUOUS_OBJECT_LEXEMES
    last = parts[-1]
    if last in _AMBIGUOUS_OBJECT_LEXEMES:
        return True
    return False


def _is_generic_environment_anchor(phrase: str) -> bool:
    """是否为缺乏区分度的环境/结构类描述（不宜作为重新导航的 PQ3D 锚点问询）。"""
    p = _normalize_phrase(phrase)
    if not p:
        return True
    parts = p.split()
    if len(parts) == 1:
        return parts[0] in _GENERIC_ENV_TOKENS
    # 短语以泛环境词结尾且较短：如 "white door", "wood floor"
    last = parts[-1]
    if last in _GENERIC_ENV_TOKENS and len(parts) <= 3:
        return True
    if parts[0] in _GENERIC_ENV_TOKENS and len(parts) <= 2:
        return True
    return False


def filter_anchor_candidates(phrases: Sequence[str]) -> List[str]:
    """去重并剔除泛环境锚点，保留顺序。"""
    out: List[str] = []
    seen: set = set()
    for ph in phrases:
        n = _normalize_phrase(str(ph))
        if not n or n in seen:
            continue
        if _is_generic_environment_anchor(n):
            continue
        if _is_ambiguous_object_lexeme(n):
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
    prompt = (
        "Decompose the navigation task into a target and OBJECT anchors (for re-querying a detector).\n"
        "Rules:\n"
        "1) target_desc: the primary object to find; short noun phrase with key modifiers.\n"
        "2) anchor_descs: 0 to 3 DISTINCTIVE movable/object anchors (furniture, appliances, lamps, "
        "decor, fixtures that identify location). Ordered by usefulness for navigation.\n"
        "3) NEVER use generic environment/structure as anchors: door, wall, ceiling, floor, window, "
        "hallway, corridor, room, stairs (as the whole anchor), ground, passage—unless paired with a "
        "clear object (e.g. ok: 'office desk', bad: 'door', bad: 'wood floor').\n"
        "4) Do not use room-type words alone as anchors (e.g. bedroom, kitchen) without a specific object.\n"
        "5) Prefer physical objects over bare surfaces or boundaries.\n"
        "6) anchor_descs should name concrete instances (e.g. 'blue armchair', 'kitchen island') rather than vague references.\n"
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
        # 仅当新字段未给出锚点时再读 anchor_desc，避免重复或与 filter 后的列表冲突（如泛词被重新塞入）
        if not raw_list:
            legacy = str(parsed.get("anchor_desc", "") or "").strip()
            if legacy:
                raw_list.append(legacy)
        anchor_descs = filter_anchor_candidates(raw_list)
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
    hints = filter_anchor_candidates(list(anchor_hints or []))
    td = _normalize_phrase(target_desc)
    hint_lines = "\n".join(f"- {h}" for h in hints) if hints else "(none)"
    prompt = (
        "You are given a navigation task and a panorama. Decide if the view SUPPORTS the task well enough "
        "that no extra navigation pivot is needed.\n"
        "Success if EITHER:\n"
        "(A) full_match: the FULL task description is visually supported (target situation is credible in the image); OR\n"
        "(B) strong_anchor_match: the PRIMARY target is NOT clearly visible, BUT a DISTINCTIVE object anchor "
        "from the task is clearly and unambiguously visible (you can name which object).\n"
        "For (B), only count anchors that are specific objects (furniture, appliances, props)—not bare walls/doors/floor/ceiling.\n"
        "Be conservative on (A); for (B) require clear identification, not guess.\n"
        "If the panorama is blurry/occluded, use low confidence and avoid strong_anchor_match.\n"
        "Return strict JSON only:\n"
        "{\"full_match\": true/false, \"strong_anchor_match\": true/false, "
        "\"matched_anchor\": \"short phrase or empty\", "
        "\"confidence\": \"high|medium|low\", \"reason\": \"...\"}\n\n"
        f"Task description: {description}\n"
        f"Target phrase (primary object): {td or '(extract from description)'}\n"
        f"Preferred object anchors (non-environment):\n{hint_lines}\n"
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
    # full_match 与 strong_anchor 均要求中高置信，避免低置信 full_match 误判「已到达」
    full_match_ok = full_match and conf in ("high", "medium")
    anchor_ok = strong and conf in ("high", "medium")
    visible = bool(full_match_ok or anchor_ok)
    return {
        "visible": visible,
        "full_match": full_match,
        "strong_anchor_match": strong,
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
    ordered = filter_anchor_candidates(list(anchor_descs))
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
    "filter_anchor_candidates",
]

