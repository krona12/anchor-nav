from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anchor_nav.pic.joint import _save_rgb_jpg, _subsample_frames_evenly, stitch_panorama
from vlm.client import chat


QueryFn = Callable[[str, int], Sequence[Tuple[int, float]]]

# 注册节点处「局部可见」半径（米）：过大则 candidate 近似全场景；需与 get_visible_object_indices 默认一致。
DEFAULT_VISIBLE_MAX_DIST_M = 2.5
# 最终决策时与节点位置对齐的空间候选半径（通常与上相同或略小）
DEFAULT_SPATIAL_NEAR_DIST_M = 2.5


@dataclass
class PosNode:
    pos: np.ndarray
    step_index: int
    vlm_names: List[str]
    object_indices: List[int]
    panorama_path: Optional[str] = None


@dataclass
class MergeTracker:
    merge_map: Dict[int, int] = field(default_factory=dict)

    def record_merge(self, old_idx: int, new_idx: int) -> None:
        old_i = int(old_idx)
        new_i = int(new_idx)
        if old_i == new_i:
            return
        self.merge_map[old_i] = new_i

    def resolve(self, idx: int) -> int:
        cur = int(idx)
        seen = set()
        while cur in self.merge_map and cur not in seen:
            seen.add(cur)
            cur = int(self.merge_map[cur])
        return int(cur)

    def resolve_all(self, indices: List[int], valid_set: Optional[set] = None) -> List[int]:
        out: List[int] = []
        used = set()
        for i in indices:
            r = self.resolve(int(i))
            if valid_set is not None and r not in valid_set:
                continue
            if r in used:
                continue
            used.add(r)
            out.append(r)
        return out


@dataclass
class PosNodeRegistry:
    nodes: List[PosNode] = field(default_factory=list)

    def add(self, node: PosNode) -> None:
        self.nodes.append(node)

    def query_co_occurrence(self, target_desc: str, anchor_desc: str) -> List[PosNode]:
        t = _normalize_phrase(target_desc)
        a = _normalize_phrase(anchor_desc)
        if not t or not a:
            return []
        out = []
        for n in self.nodes:
            has_t = any(_phrase_match(t, x) for x in n.vlm_names)
            has_a = any(_phrase_match(a, x) for x in n.vlm_names)
            if has_t and has_a:
                out.append(n)
        return out

    def query_single(self, desc: str) -> List[PosNode]:
        q = _normalize_phrase(desc)
        if not q:
            return []
        out = []
        for n in self.nodes:
            if any(_phrase_match(q, x) for x in n.vlm_names):
                out.append(n)
        return out

    def resolve_object_indices(
        self,
        node: PosNode,
        merge_tracker: MergeTracker,
        *,
        rep: Any,
    ) -> List[int]:
        box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
        valid_set = set(range(int(box.shape[0])))
        return merge_tracker.resolve_all([int(x) for x in node.object_indices], valid_set=valid_set)


def _normalize_phrase(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _phrase_match(query: str, text: str) -> bool:
    q = _normalize_phrase(query)
    t = _normalize_phrase(text)
    if not q or not t:
        return False
    return q in t or t in q


def _parse_json_obj(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _parse_json_list(raw: str) -> List[str]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    arr = json.loads(text)
    if not isinstance(arr, list):
        raise RuntimeError(f"VLM list parse failed, raw={raw!r}")
    out = []
    seen = set()
    for x in arr:
        s = _normalize_phrase(str(x))
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def get_visible_object_indices(
    agent_pos: np.ndarray,
    rep: Any,
    max_dist: float = DEFAULT_VISIBLE_MAX_DIST_M,
) -> List[int]:
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] < 3:
        return []
    centers = np.asarray(box[:, :3], dtype=float).copy()
    centers[:, [1, 2]] = centers[:, [2, 1]]
    p = np.asarray(agent_pos, dtype=float).reshape(3)
    dists = np.linalg.norm(centers - p[None, :], axis=1)
    return [int(i) for i in np.where(dists < float(max_dist))[0]]


def candidate_indices_for_node(
    node: PosNode,
    merge_tracker: MergeTracker,
    rep: Any,
    *,
    near_dist_m: float = DEFAULT_SPATIAL_NEAR_DIST_M,
) -> List[int]:
    """
    合并「历史上记录的可见 index」与「当前 RepresentationManager 下、以节点位置为中心的局部可见 index」。

    RepresentationManager 在全局 top-k 裁剪时会重排物体列下标，PosNode 内缓存的 object_indices 可能整体失效；
    仅用 resolve_merge 无法恢复。此处用节点拍摄全景时的位置对**当前** object_box 做距离过滤，
    得到与当下索引对齐的候选集；若与 resolve 后的历史索引有交集则优先取交，否则用空间候选。
    """
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0:
        return []
    valid_set = set(range(int(box.shape[0])))
    hist = merge_tracker.resolve_all([int(x) for x in node.object_indices], valid_set=valid_set)
    hist_set = {int(x) for x in hist}
    pos = np.asarray(node.pos, dtype=float).reshape(3)
    fresh = get_visible_object_indices(pos, rep, max_dist=float(near_dist_m))
    fresh_set = {int(x) for x in fresh}

    if len(hist_set) > 0 and len(fresh_set) > 0:
        inter = hist_set & fresh_set
        chosen = inter if len(inter) > 0 else fresh_set
    elif len(fresh_set) > 0:
        chosen = fresh_set
    else:
        chosen = hist_set
    return sorted(chosen)


def gather_posnode_candidate_indices(
    matched_nodes: List[PosNode],
    merge_tracker: MergeTracker,
    rep: Any,
    *,
    near_dist_m: float = DEFAULT_SPATIAL_NEAR_DIST_M,
) -> List[int]:
    s = set()
    for node in matched_nodes:
        for x in candidate_indices_for_node(node, merge_tracker, rep, near_dist_m=near_dist_m):
            s.add(int(x))
    return sorted(s)


# 不宜作为 posnode 锚点的泛词 / 房间标签（参考实验诊断与 VFV anchor 过滤）
_BAD_POSNODE_ANCHOR_LEXEMES = frozenset(
    {
        "window",
        "windows",
        "door",
        "doors",
        "wall",
        "walls",
        "floor",
        "floors",
        "ceiling",
        "ceilings",
        "room",
        "rooms",
        "area",
        "space",
        "hallway",
        "corridor",
        "kitchen",
        "bedroom",
        "bathroom",
        "dining",
        "region",
    }
)


def _is_bad_posnode_anchor_phrase(phrase: str) -> bool:
    p = _normalize_phrase(str(phrase))
    if not p:
        return True
    parts = p.split()
    if len(parts) == 1:
        return parts[0] in _BAD_POSNODE_ANCHOR_LEXEMES
    if parts[-1] in ("room", "area", "space") and len(parts) <= 3:
        return True
    if re.search(
        r"^(dining room|living room|bed ?room|master bedroom|bathroom|kitchen|hallway|corridor)$",
        p,
    ):
        return True
    return False


def decompose_description(description: str, vlm_model: str) -> Dict[str, Any]:
    prompt = (
        "Decompose this navigation description into target and anchor.\n"
        "Rules:\n"
        "1. target: object to navigate TO (single noun phrase with modifiers).\n"
        "2. anchors: list 0-3 nearby reference object phrases.\n"
        "3. Multi-anchor descriptions are common. Keep all reasonable nearby anchors in anchor_descs.\n"
        "4. If no clear nearby anchor, use empty list.\n"
        "3. Keep phrases concrete and visually grounded. Avoid room-only words (e.g., 'bathroom') as anchor.\n"
        "5. Order anchors by confidence, highest first.\n"
        "Return strict JSON: {\"target_desc\": \"...\", \"anchor_descs\": [\"...\"], \"anchor_desc\": \"...\"}\n"
        "anchor_desc is optional backward-compat field; if provided, it should be the best anchor.\n\n"
        f"Description: {description}"
    )
    raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=96)
    parsed = _parse_json_obj(raw)
    target_desc = _normalize_phrase(str(parsed.get("target_desc", "")))
    anchor_descs_raw = parsed.get("anchor_descs", [])
    if not isinstance(anchor_descs_raw, list):
        anchor_descs_raw = []
    anchor_desc_legacy = _normalize_phrase(str(parsed.get("anchor_desc", "")))
    anchor_descs: List[str] = []
    seen = set()
    for x in anchor_descs_raw:
        s = _normalize_phrase(str(x))
        if not s or s in seen:
            continue
        seen.add(s)
        anchor_descs.append(s)
    if anchor_desc_legacy and anchor_desc_legacy not in seen:
        anchor_descs.append(anchor_desc_legacy)
    anchor_descs = [a for a in anchor_descs if not _is_bad_posnode_anchor_phrase(a)]
    if len(anchor_descs) == 0 and anchor_desc_legacy and not _is_bad_posnode_anchor_phrase(anchor_desc_legacy):
        anchor_descs = [anchor_desc_legacy]
    anchor_descs = anchor_descs[:3]
    best_anchor = anchor_descs[0] if len(anchor_descs) > 0 else ""
    return {
        "target_desc": target_desc,
        "anchor_desc": _normalize_phrase(best_anchor),
        "anchor_descs": anchor_descs,
        "raw": str(raw),
    }


def update_panorama_node(
    *,
    agent_pos: np.ndarray,
    color_list: List[np.ndarray],
    rep: Any,
    registry: PosNodeRegistry,
    merge_tracker: MergeTracker,
    vlm_model: str,
    step_index: int,
    panorama_dir: Optional[Path] = None,
    max_visible_dist: float = DEFAULT_VISIBLE_MAX_DIST_M,
    panorama_subsample_frames: int = 12,
    min_move_dist_to_add: float = 0.4,
) -> Dict[str, Any]:
    del merge_tracker  # reserved for future merge callback integration
    cur_pos = np.asarray(agent_pos, dtype=float).reshape(3)
    if len(registry.nodes) > 0:
        prev_pos = np.asarray(registry.nodes[-1].pos, dtype=float).reshape(3)
        move_dist = float(np.linalg.norm(cur_pos - prev_pos))
        if move_dist < float(min_move_dist_to_add):
            return {
                "ok": False,
                "skipped": True,
                "reason": "movement_below_threshold",
                "move_dist": float(move_dist),
                "min_move_dist_to_add": float(min_move_dist_to_add),
                "registry_nodes_total": int(len(registry.nodes)),
            }

    sampled = _subsample_frames_evenly(color_list, max_frames=int(panorama_subsample_frames))
    panorama = stitch_panorama(sampled)
    pano_path: Optional[str] = None
    if panorama_dir is not None:
        panorama_dir.mkdir(parents=True, exist_ok=True)
        pano_file = panorama_dir / f"pano_step_{int(step_index):05d}.jpg"
        _save_rgb_jpg(panorama, pano_file)
        pano_path = str(pano_file)
        image_path = pano_file
    else:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fp:
            tmp_path = Path(fp.name)
        _save_rgb_jpg(panorama, tmp_path)
        image_path = tmp_path
        pano_path = str(tmp_path)

    prompt = (
        "List distinct visible object nouns from this 360 panorama image captured at ONE current agent position.\n"
        "Rules:\n"
        "1. Output only concrete objects that are clearly visible.\n"
        "2. Do NOT output room types or vague regions (e.g., bathroom, corner, area).\n"
        "3. Use short noun phrases, deduplicate synonyms, max 20 items.\n"
        "Return strict JSON array only, e.g. "
        "[\"round wooden table\", \"gray couch\", \"potted plant\"]."
    )
    raw = chat(text=prompt, image_path=str(image_path), model=vlm_model, max_tokens=128)
    vlm_names = _parse_json_list(raw)
    visible_indices = get_visible_object_indices(np.asarray(agent_pos, dtype=float).reshape(3), rep, max_dist=max_visible_dist)
    node = PosNode(
        pos=cur_pos.copy(),
        step_index=int(step_index),
        vlm_names=vlm_names,
        object_indices=[int(x) for x in visible_indices],
        panorama_path=pano_path,
    )
    registry.add(node)
    return {
        "ok": True,
        "node_index": int(len(registry.nodes) - 1),
        "step_index": int(step_index),
        "panorama_frames_used": int(len(sampled)),
        "move_dist": None if len(registry.nodes) <= 1 else float(np.linalg.norm(cur_pos - np.asarray(registry.nodes[-2].pos, dtype=float).reshape(3))),
        "min_move_dist_to_add": float(min_move_dist_to_add),
        "vlm_names": vlm_names,
        "visible_indices": [int(x) for x in visible_indices],
        "panorama_path": pano_path,
    }


def build_node_summary(registry: PosNodeRegistry, max_nodes: int = 24) -> str:
    if len(registry.nodes) == 0:
        return "(empty)"
    nodes = registry.nodes[-int(max_nodes):]
    lines = []
    base_idx = len(registry.nodes) - len(nodes)
    for i, node in enumerate(nodes):
        idx = base_idx + i
        names = ", ".join(node.vlm_names[:10]) if len(node.vlm_names) > 0 else "(none)"
        p = np.asarray(node.pos, dtype=float).reshape(3)
        lines.append(f"[node_{idx}] pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) objects: {names}")
    return "\n".join(lines)


def query_registry_with_vlm(
    *,
    description: str,
    registry: PosNodeRegistry,
    vlm_model: str,
    decomp: Optional[Dict[str, Any]] = None,
    max_nodes: int = 24,
) -> Dict[str, Any]:
    if decomp is None:
        decomp = decompose_description(description, vlm_model)
    target_desc = _normalize_phrase(decomp.get("target_desc", ""))
    anchor_desc = _normalize_phrase(decomp.get("anchor_desc", ""))
    anchor_descs = [_normalize_phrase(str(x)) for x in list(decomp.get("anchor_descs", []))]
    anchor_descs = [x for x in anchor_descs if x]
    if len(anchor_descs) == 0 and anchor_desc:
        anchor_descs = [anchor_desc]
    if len(registry.nodes) == 0:
        return {
            "mode": "fallback",
            "matched_nodes": [],
            "target_desc": target_desc,
            "anchor_desc": anchor_desc,
            "anchor_descs": anchor_descs,
        }

    summary = build_node_summary(registry, max_nodes=max_nodes)
    anchor_descs_text = ", ".join([f"'{x}'" for x in anchor_descs]) if len(anchor_descs) > 0 else "(none)"
    prompt = (
        f"Navigation task: find '{target_desc}' near ANY of anchors: {anchor_descs_text}.\n"
        f"Observation history:\n{summary}\n\n"
        "Decision rules (be conservative):\n"
        "1. Put index in co_occur only if BOTH target and at least one anchor are explicitly present in the same node object list.\n"
        "2. If uncertain, return empty arrays. Prefer precision over recall.\n"
        "3. anchor_only or target_only can contain at most one index and only when uniquely supported.\n"
        "4. Do NOT infer by room context or commonsense; rely on listed objects only.\n\n"
        "Q1: Which node indices show BOTH the target and ANY anchor together? (Return [] if none)\n"
        "Q2: Which node index shows anchor-only evidence and is unique? (Return [] otherwise)\n"
        "Q3: Which node index shows the target only and is unique? (Return [] otherwise)\n"
        "Return strict JSON only: "
        "{\"co_occur\": [...], \"anchor_only\": [...], \"target_only\": [...]}"
    )
    raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=128)
    parsed = _parse_json_obj(raw)
    co_occur = [int(x) for x in parsed.get("co_occur", []) if isinstance(x, (int, float))]
    anchor_only = [int(x) for x in parsed.get("anchor_only", []) if isinstance(x, (int, float))]
    target_only = [int(x) for x in parsed.get("target_only", []) if isinstance(x, (int, float))]
    n_nodes = len(registry.nodes)
    co_occur = [i for i in co_occur if 0 <= i < n_nodes]
    anchor_only = [i for i in anchor_only if 0 <= i < n_nodes]
    target_only = [i for i in target_only if 0 <= i < n_nodes]

    if len(co_occur) > 0:
        return {
            "mode": "co_occur",
            "matched_nodes": [registry.nodes[i] for i in co_occur],
            "matched_node_indices": co_occur,
            "target_desc": target_desc,
            "anchor_desc": anchor_desc,
            "anchor_descs": anchor_descs,
            "raw": raw,
        }
    if len(anchor_only) == 1:
        return {
            "mode": "anchor_only",
            "matched_nodes": [registry.nodes[anchor_only[0]]],
            "matched_node_indices": anchor_only,
            "target_desc": target_desc,
            "anchor_desc": anchor_desc,
            "anchor_descs": anchor_descs,
            "raw": raw,
        }
    if len(target_only) == 1:
        return {
            "mode": "target_only",
            "matched_nodes": [registry.nodes[target_only[0]]],
            "matched_node_indices": target_only,
            "target_desc": target_desc,
            "anchor_desc": anchor_desc,
            "anchor_descs": anchor_descs,
            "raw": raw,
        }
    return {
        "mode": "fallback",
        "matched_nodes": [],
        "matched_node_indices": [],
        "target_desc": target_desc,
        "anchor_desc": anchor_desc,
        "anchor_descs": anchor_descs,
        "raw": raw,
    }


def validate_cooccur_nodes_with_image(
    *,
    description: str,
    target_desc: str,
    anchor_descs: List[str],
    matched_nodes: List[PosNode],
    vlm_model: str,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    for i, node in enumerate(matched_nodes):
        pano = str(node.panorama_path or "").strip()
        if not pano:
            records.append(
                {
                    "node_offset": int(i),
                    "valid": False,
                    "reason": "missing_panorama_path",
                    "matched_anchor": "",
                    "confidence": "low",
                }
            )
            continue
        pano_path = Path(pano).expanduser()
        if not pano_path.is_absolute():
            pano_path = (Path.cwd() / pano_path).resolve()
        if not pano_path.exists():
            records.append(
                {
                    "node_offset": int(i),
                    "valid": False,
                    "reason": "panorama_not_found",
                    "matched_anchor": "",
                    "confidence": "low",
                    "panorama_path": str(pano_path),
                }
            )
            continue
        anchors_text = ", ".join([f"'{x}'" for x in anchor_descs if str(x).strip()]) or "(none)"
        prompt = (
            "Verify this candidate panorama for navigation.\n"
            f"Full description: {description}\n"
            f"Target: '{target_desc}'\n"
            f"Anchors (any one is acceptable): {anchors_text}\n\n"
            "Decision rule:\n"
            "Return valid=true ONLY if the panorama clearly supports that target and at least one anchor are both present.\n"
            "If uncertain or ambiguous, return valid=false.\n"
            "Return strict JSON only: "
            "{\"valid\": true/false, \"matched_anchor\": \"...\", \"confidence\": \"high|medium|low\", \"reason\": \"...\"}"
        )
        raw = chat(text=prompt, image_path=str(pano_path), model=vlm_model, max_tokens=128)
        try:
            parsed = _parse_json_obj(raw)
            is_valid = bool(parsed.get("valid", False))
            rec = {
                "node_offset": int(i),
                "valid": bool(is_valid),
                "matched_anchor": _normalize_phrase(str(parsed.get("matched_anchor", ""))),
                "confidence": str(parsed.get("confidence", "")),
                "reason": str(parsed.get("reason", "")),
                "panorama_path": str(pano_path),
            }
            records.append(rec)
            if is_valid:
                valid_indices.append(int(i))
        except Exception:
            records.append(
                {
                    "node_offset": int(i),
                    "valid": False,
                    "matched_anchor": "",
                    "confidence": "low",
                    "reason": "verify_parse_failed",
                    "panorama_path": str(pano_path),
                }
            )
    return {
        "ok": True,
        "valid": bool(len(valid_indices) > 0),
        "valid_node_offsets": valid_indices,
        "records": records,
    }


def select_from_topk(
    *,
    topk: List[Tuple[int, float]],
    query_result: Dict[str, Any],
    merge_tracker: MergeTracker,
    rep: Any,
) -> int:
    if len(topk) == 0:
        raise ValueError("empty topk")
    matched_nodes = list(query_result.get("matched_nodes", []))
    if len(matched_nodes) == 0:
        return int(topk[0][0])

    candidate_set = set(gather_posnode_candidate_indices(matched_nodes, merge_tracker, rep))
    for obj_idx, _ in topk:
        if int(obj_idx) in candidate_set:
            return int(obj_idx)
    return int(topk[0][0])


def select_nearest_object_from_nodes(
    *,
    matched_nodes: List[PosNode],
    merge_tracker: MergeTracker,
    rep: Any,
    agent_pos_xyz: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if len(matched_nodes) == 0:
        return None
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] < 3:
        return None
    candidate_set = set(gather_posnode_candidate_indices(matched_nodes, merge_tracker, rep))
    if len(candidate_set) == 0:
        return None

    box_nav = np.asarray(box[:, :3], dtype=float).copy()
    box_nav[:, [1, 2]] = box_nav[:, [2, 1]]
    p = np.asarray(agent_pos_xyz, dtype=float).reshape(3)
    cand = sorted(list(candidate_set))
    d = [float(np.linalg.norm(box_nav[i] - p)) for i in cand]
    j = int(np.argmin(np.asarray(d, dtype=float)))
    chosen = int(cand[j])
    chosen_dist = float(d[j])
    chosen_xyz = np.asarray(box[chosen, :3], dtype=float).reshape(3).copy()
    chosen_xyz[[1, 2]] = chosen_xyz[[2, 1]]
    return {
        "chosen_object_index": int(chosen),
        "chosen_object_position": chosen_xyz.tolist(),
        "distance_to_agent": float(chosen_dist),
        "candidate_set": cand,
    }


def build_selection_trace(
    *,
    topk: List[Tuple[int, float]],
    query_result: Dict[str, Any],
    merge_tracker: MergeTracker,
    rep: Any,
) -> Dict[str, Any]:
    if len(topk) == 0:
        return {
            "mode": str(query_result.get("mode", "fallback")),
            "candidate_set": [],
            "chosen_object_index": None,
            "chosen_rank": None,
            "fallback_to_top1": True,
        }
    mode = str(query_result.get("mode", "fallback"))
    matched_nodes = list(query_result.get("matched_nodes", []))
    candidate_set = set()
    if len(matched_nodes) > 0:
        candidate_set.update(gather_posnode_candidate_indices(matched_nodes, merge_tracker, rep))
    chosen = int(topk[0][0])
    chosen_rank = 1
    fallback = True
    if len(candidate_set) > 0:
        for r, (obj_idx, _) in enumerate(topk, start=1):
            if int(obj_idx) in candidate_set:
                chosen = int(obj_idx)
                chosen_rank = int(r)
                fallback = False
                break
    return {
        "mode": mode,
        "matched_node_indices": list(query_result.get("matched_node_indices", [])),
        "candidate_set": sorted([int(x) for x in candidate_set]),
        "chosen_object_index": int(chosen),
        "chosen_rank": int(chosen_rank),
        "fallback_to_top1": bool(fallback),
    }


def registry_snapshot(
    *,
    registry: PosNodeRegistry,
    merge_tracker: MergeTracker,
    rep: Any,
    max_nodes: int = 12,
) -> List[Dict[str, Any]]:
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    valid_set = set(range(int(box.shape[0])))
    out: List[Dict[str, Any]] = []
    start = max(0, len(registry.nodes) - int(max_nodes))
    for idx in range(start, len(registry.nodes)):
        node = registry.nodes[idx]
        resolved = merge_tracker.resolve_all(node.object_indices, valid_set=valid_set)
        spatial = candidate_indices_for_node(node, merge_tracker, rep)
        p = np.asarray(node.pos, dtype=float).reshape(3)
        out.append(
            {
                "node_index": int(idx),
                "step_index": int(node.step_index),
                "pos": [float(p[0]), float(p[1]), float(p[2])],
                "vlm_names": [str(x) for x in node.vlm_names],
                "object_indices": [int(x) for x in node.object_indices],
                "resolved_object_indices": [int(x) for x in resolved],
                "spatial_candidate_indices": spatial,
                "panorama_path": node.panorama_path,
            }
        )
    return out


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

    query_locs = obj_locs.clone()
    query_pad_masks = obj_pad_masks.clone()
    query_scores_t = obj_scores.clone()
    obj_labels = torch.zeros(n, dtype=torch.long)
    tgt_object_id = torch.LongTensor([])

    encoded_input = tokenizer([sentence], add_special_tokens=True, truncation=True)
    tokenized_txt = encoded_input.input_ids[0]
    prompt = torch.FloatTensor(tokenized_txt)
    prompt_pad_masks = torch.ones((len(tokenized_txt))).bool()

    data_dict = {
        "query_pad_masks": query_pad_masks,
        "query_locs": query_locs,
        "query_scores": query_scores_t,
        "real_obj_pad_masks": real_obj_pad_masks,
        "seg_center": seg_center,
        "seg_pad_masks": seg_pad_masks,
        "mv_seg_fts": mv_seg_fts,
        "mv_seg_pad_masks": mv_seg_pad_masks,
        "vocab_seg_fts": vocab_seg_fts,
        "vocab_seg_pad_masks": vocab_seg_pad_masks,
        "obj_labels": obj_labels,
        "tgt_object_id": tgt_object_id,
        "decision_label": 1,
        "prompt": prompt,
        "prompt_pad_masks": prompt_pad_masks,
        "prompt_type": PromptType.TXT,
    }
    batch = default_collate([data_dict])
    batch = batch_to_cuda(batch)
    stage2.eval()
    with torch.no_grad():
        stage2_output = stage2(batch)
    logits = stage2_output["og3d_logits"].detach().float().cpu().numpy().reshape(-1)
    mask = stage2_output["real_obj_pad_masks"].bool().detach().cpu().numpy().reshape(-1)
    return logits[mask].astype(np.float64)


def build_query_fn_from_pq3d_stage2(pq3d_model: Any) -> QueryFn:
    def _query_fn(text: str, top_k: int) -> Sequence[Tuple[int, float]]:
        scores = pq3d_stage2_object_logits(pq3d_model, text)
        if scores.size == 0:
            return []
        k = max(1, int(top_k))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    return _query_fn

