from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import re


# Query callback:
# input text query, output ranked [(object_index, score), ...] in descending order.
QueryFn = Callable[[str, int], Sequence[Tuple[int, float]]]
TextEncoderFn = Callable[[str], np.ndarray]


@dataclass
class VoteConfig:
    node_min_dist_m: float = 1.0
    query_top_k: int = 5
    node_pick_top_k: int = 5
    softmax_temp: float = 0.07
    target_vote_weight: float = 0.05
    primary_anchor_vote_weight: float = 0.05
    other_anchor_vote_weight: float = 0.05
    max_nearby_anchors: int = 1
    max_secondary_anchors: int = 2
    # Substitute (refined != main_target): top refined_vote_top_k hits, vote weight refined_substitute_vote_weight × γ**rank.
    refined_vote_top_k: int = 5
    refined_substitute_vote_weight: float = 0.9
    # Extra scale on anchor queries only (per-anchor weights above are the 0.05 channel vs 0.9/0.05).
    anchor_vote_aggregate_scale: float = 1.0
    # Per ranked slot: multiply contribution by gamma**rank (0-based). 1.0 = same weight for every top-k hit (legacy).
    query_rank_decay_gamma: float = 0.75
    # Within best_node: "pick" (refined text), "target" (main_target), or "max" of both (Stage2 logits).
    object_pick_score: str = "pick"


def normalize_anchors_with_weights(
    anchors: Sequence[str],
    anchor_types: Sequence[str],
    *,
    max_nearby: int = 1,
    max_secondary: int = 2,
    nearby_weight: float = 0.8,
    secondary_weight: float = 0.5,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Normalize anchors for voting:
    - Keep at most `max_nearby` anchors with type "nearby"
    - Keep at most `max_secondary` anchors with non-nearby type (normalized to "scene")
    - Return aligned (anchors, weights, types)
    """
    nearby_items: List[Tuple[str, str]] = []
    secondary_items: List[Tuple[str, str]] = []
    seen = set()
    for i, raw in enumerate(anchors):
        a = str(raw).strip()
        if not a:
            continue
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        t = "nearby"
        if i < len(anchor_types):
            tt = str(anchor_types[i]).strip().lower()
            if tt == "nearby":
                t = "nearby"
            else:
                t = "scene"
        if t == "nearby":
            nearby_items.append((a, t))
        else:
            secondary_items.append((a, "scene"))
    kept = nearby_items[: int(max_nearby)] + secondary_items[: int(max_secondary)]
    out_anchors = [x[0] for x in kept]
    out_types = [x[1] for x in kept]
    out_weights = [float(nearby_weight if t == "nearby" else secondary_weight) for t in out_types]
    return out_anchors, out_weights, out_types


def build_refined_query_prompt(description: str) -> str:
    """Same contract as refhm3d-nav-sequence-baseline-substitue extract_target_anchor_query_from_description."""
    return (
        "Refine the navigation description for robust object query.\n"
        "Keep ONLY:\n"
        "1) main_target: exact target object phrase\n"
        "2) key_anchor: at most one spatially tight anchor object that is directly linked to main_target.\n"
        "Drop broad scene/global context and weak anchors.\n"
        "Return strict JSON only: "
        "{\"main_target\": \"...\", \"key_anchor\": \"... or empty\", \"refined_query\": \"...\"}.\n"
        "If no reliable anchor, set key_anchor to empty and refined_query=main_target.\n\n"
        f"Description: {description}"
    )


def parse_refined_query_from_vlm_raw(raw: str) -> Tuple[str, str, str]:
    parsed = _parse_json_obj(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    key_anchor = str(parsed.get("key_anchor", "")).strip()
    refined_query = str(parsed.get("refined_query", "")).strip()
    if not main_target:
        raise RuntimeError(f"empty main_target in refined query, raw={raw!r}")
    if not refined_query:
        refined_query = main_target if not key_anchor else f"{main_target} near {key_anchor}"
    return main_target, key_anchor, refined_query


def build_target_anchor_prompt(description: str) -> str:
    return (
        "Extract navigation target and anchors from the description.\n"
        "Return strict JSON only: "
        "{\"main_target\":\"...\", \"anchors\":[...], \"anchor_types\":[...], \"spatial_relation\":\"... or null\"}.\n\n"
        "Hard rules:\n"
        "1) main_target MUST be a single object noun phrase (with material/color/style modifiers if present).\n"
        "2) main_target MUST NOT contain relational or scene clauses: do NOT include words/phrases like "
        "\"with\", \"in\", \"near\", \"between\", \"beside\", \"under\", \"above\", \"next to\", \"in bedroom\", etc.\n"
        "3) Nearby relation object(s) go to anchors, not main_target.\n"
        "4) anchor_types aligned to anchors, each is either \"nearby\" or \"scene\".\n"
        "5) Keep ONLY one nearby anchor and at most two secondary(scene) anchors.\n\n"
        "Examples:\n"
        "- Input: \"white table with pink chair in bedroom\"\n"
        "  Output main_target: \"white table\"\n"
        "  Output anchors: [\"pink chair\", \"bedroom\"]\n"
        "  Output anchor_types: [\"nearby\", \"scene\"]\n"
        "- Input: \"black wall-mounted tv above fireplace in living room\"\n"
        "  Output main_target: \"black wall-mounted tv\"\n"
        "  Output anchors: [\"fireplace\", \"living room\"]\n"
        "  Output anchor_types: [\"nearby\", \"scene\"]\n\n"
        f"Description: {description}"
    )


def _parse_json_obj(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def parse_target_anchors_from_vlm_raw(
    raw: str,
    *,
    nearby_anchor_vote_weight: float,
    secondary_anchor_vote_weight: float,
) -> Tuple[str, List[str], List[float], str, List[str]]:
    parsed = _parse_json_obj(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    anchors_raw = parsed.get("anchors", [])
    anchor_types_raw = parsed.get("anchor_types", [])
    relation = str(parsed.get("spatial_relation", "") or "").strip()
    anchors: List[str] = []
    anchor_types: List[str] = []
    if isinstance(anchors_raw, list):
        for i, x in enumerate(anchors_raw):
            s = str(x).strip()
            if not s:
                continue
            if s.lower() in {"null", "none"}:
                continue
            if s.lower() == main_target.lower():
                continue
            if s.lower() in {a.lower() for a in anchors}:
                continue
            t = "nearby"
            if isinstance(anchor_types_raw, list) and i < len(anchor_types_raw):
                tt = str(anchor_types_raw[i]).strip().lower()
                if tt in {"nearby", "scene"}:
                    t = tt
            anchors.append(s)
            anchor_types.append(t)
    if not main_target:
        raise RuntimeError(f"empty main_target, raw={raw!r}")
    anchors, anchor_weights, anchor_types = normalize_anchors_with_weights(
        anchors,
        anchor_types,
        max_nearby=1,
        max_secondary=2,
        nearby_weight=float(nearby_anchor_vote_weight),
        secondary_weight=float(secondary_anchor_vote_weight),
    )
    return main_target, anchors, anchor_weights, relation, anchor_types


@dataclass
class VoteState:
    # node centers in map/world xyz
    nodes: List[np.ndarray] = field(default_factory=list)
    # object index -> node id
    object_node: Dict[int, int] = field(default_factory=dict)
    # object index -> distance(object anchor position, bound node center)
    object_node_dist: Dict[int, float] = field(default_factory=dict)
    # object index -> first seen position
    object_first_position: Dict[int, np.ndarray] = field(default_factory=dict)


def _as_xyz(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"position must have >=3 dims, got shape={arr.shape}")
    return arr[:3].copy()


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    y = x / t
    y = y - np.max(y)
    e = np.exp(y)
    return e / max(float(np.sum(e)), 1e-12)


def _normalize_scores_within_query(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def assign_or_create_node(state: VoteState, position_xyz: Any, cfg: VoteConfig) -> int:
    pos = _as_xyz(position_xyz)
    if len(state.nodes) == 0:
        state.nodes.append(pos)
        return 0
    centers = np.stack(state.nodes, axis=0)
    d = np.linalg.norm(centers - pos[None, :], axis=1)
    j = int(np.argmin(d))
    if float(d[j]) > float(cfg.node_min_dist_m):
        state.nodes.append(pos)
        return len(state.nodes) - 1
    return j


def bind_object_to_position(
    state: VoteState,
    *,
    object_index: int,
    position_xyz: Any,
    cfg: VoteConfig,
) -> Dict[str, Any]:
    """
    Bind object to nearest node. If a newly observed position is closer than previous binding, rebind.
    """
    pos = _as_xyz(position_xyz)
    node_id = assign_or_create_node(state, pos, cfg)
    node_center = state.nodes[node_id]
    dist = float(np.linalg.norm(node_center - pos))

    prev_node = state.object_node.get(int(object_index), None)
    prev_dist = state.object_node_dist.get(int(object_index), None)

    rebind = False
    if prev_node is None:
        rebind = True
    elif prev_dist is None or dist < float(prev_dist):
        rebind = True

    if rebind:
        state.object_node[int(object_index)] = int(node_id)
        state.object_node_dist[int(object_index)] = float(dist)
        if int(object_index) not in state.object_first_position:
            state.object_first_position[int(object_index)] = pos

    return {
        "object_index": int(object_index),
        "node_id": int(state.object_node.get(int(object_index), node_id)),
        "distance_to_node": float(state.object_node_dist.get(int(object_index), dist)),
        "created_or_rebound": bool(rebind),
        "prev_node": None if prev_node is None else int(prev_node),
        "prev_dist": None if prev_dist is None else float(prev_dist),
    }


def _bind_object_to_nearest_existing_node(
    state: VoteState,
    *,
    object_index: int,
    position_xyz: Any,
) -> Dict[str, Any]:
    pos = _as_xyz(position_xyz)
    if len(state.nodes) == 0:
        raise ValueError("no existing nodes to bind object")
    centers = np.stack(state.nodes, axis=0)
    d = np.linalg.norm(centers - pos[None, :], axis=1)
    node_id = int(np.argmin(d))
    dist = float(d[node_id])

    prev_node = state.object_node.get(int(object_index), None)
    prev_dist = state.object_node_dist.get(int(object_index), None)

    rebind = False
    if prev_node is None:
        rebind = True
    elif prev_dist is None or dist < float(prev_dist):
        rebind = True

    if rebind:
        state.object_node[int(object_index)] = int(node_id)
        state.object_node_dist[int(object_index)] = float(dist)
        if int(object_index) not in state.object_first_position:
            state.object_first_position[int(object_index)] = pos

    return {
        "object_index": int(object_index),
        "node_id": int(state.object_node.get(int(object_index), node_id)),
        "distance_to_node": float(state.object_node_dist.get(int(object_index), dist)),
        "created_or_rebound": bool(rebind),
        "prev_node": None if prev_node is None else int(prev_node),
        "prev_dist": None if prev_dist is None else float(prev_dist),
    }


def update_bindings_for_new_objects(
    state: VoteState,
    *,
    prev_object_count: int,
    cur_object_count: int,
    agent_position_xyz: Any,
    object_positions_xyz: Optional[Dict[int, Any]] = None,
    cfg: VoteConfig,
) -> List[Dict[str, Any]]:
    """
    Caller can invoke this after each perceive/merge step.
    Node generation rule:
    - Nodes are generated only from agent positions (at most one new node per call).
    Object binding rule:
    - New objects bind to nearest existing node using object position when available;
      otherwise use current agent position as fallback.
    """
    records: List[Dict[str, Any]] = []
    # Keep "position-node" semantics: node comes from agent position only.
    assign_or_create_node(state, agent_position_xyz, cfg)
    # Rebind existing objects to closer nodes when better positions are available.
    if object_positions_xyz is not None:
        for oi, pos in object_positions_xyz.items():
            if int(oi) < int(prev_object_count):
                _bind_object_to_nearest_existing_node(
                    state,
                    object_index=int(oi),
                    position_xyz=pos,
                )
    for obj_idx in range(int(prev_object_count), int(cur_object_count)):
        bind_pos = agent_position_xyz
        if object_positions_xyz is not None and int(obj_idx) in object_positions_xyz:
            bind_pos = object_positions_xyz[int(obj_idx)]
        rec = _bind_object_to_nearest_existing_node(
            state,
            object_index=int(obj_idx),
            position_xyz=bind_pos,
        )
        records.append(rec)
    return records


def run_position_vote(
    state: VoteState,
    *,
    query_fn: QueryFn,
    main_target: str,
    anchors: Sequence[str],
    anchor_weights: Optional[Sequence[float]] = None,
    cfg: VoteConfig,
    candidate_object_indices: Optional[Sequence[int]] = None,
    candidate_target_scores: Optional[Dict[int, float]] = None,
    refined_pick_text: Optional[str] = None,
    candidate_pick_scores: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Spatial vote:
    1) Target query (target_vote_weight); each ranked hit contributes × query_rank_decay_gamma**rank.
    2) If refined_pick_text differs from main_target: substitute top-k **before** anchors; each hit contributes
       refined_substitute_vote_weight × γ**rank (same γ as target/anchors). Skipped when refined equals main_target.
    3) Anchor queries: per-anchor weight × anchor_vote_aggregate_scale × γ**rank.
    4) Pick node: max ``vote_score_sum`` first, tie-break ``vote_count``, then smaller node_id.
    5) In best node, pick object per object_pick_score; ties → smaller object_index.

    NOTE:
    - To use PQ3D Stage2 without re-running perception, pass a query_fn over current memory (recommended).
    """
    if not main_target.strip():
        raise ValueError("main_target is empty")
    if len(state.object_node) == 0:
        return {"ok": False, "reason": "empty_object_node_binding"}
    candidate_set = None
    if candidate_object_indices is not None:
        candidate_set = {int(x) for x in candidate_object_indices}
        if len(candidate_set) == 0:
            return {
                "ok": False,
                "reason": "empty_candidate_object_indices",
                "candidate_object_indices": [],
            }

    anchor_list = [str(a).strip() for a in anchors if str(a).strip()]
    weights_list: List[float] = []
    if anchor_weights is None:
        for i, _ in enumerate(anchor_list):
            weights_list.append(float(cfg.primary_anchor_vote_weight if i == 0 else cfg.other_anchor_vote_weight))
    else:
        for i, _ in enumerate(anchor_list):
            if i < len(anchor_weights):
                weights_list.append(float(anchor_weights[i]))
            else:
                weights_list.append(float(cfg.other_anchor_vote_weight))
    anc_scale = max(0.0, float(getattr(cfg, "anchor_vote_aggregate_scale", 1.0)))
    rank_gamma = max(0.0, min(1.0, float(getattr(cfg, "query_rank_decay_gamma", 1.0))))

    def _rank_decay_mult(rank_i: int) -> float:
        return float(rank_gamma) ** int(rank_i)

    def _candidate_allowed(oi: int) -> bool:
        return candidate_set is None or int(oi) in candidate_set

    # vote[node]: vote_count, vote_score_sum — per-query min-max norm on scores.
    vote_count: Dict[int, float] = {}
    vote_score_sum: Dict[int, float] = {}
    query_logs: List[Dict[str, Any]] = []
    target_score_map: Dict[int, float] = {}

    def _accumulate_query(q_type: str, q_text: str, q_weight: float) -> None:
        ranked = list(query_fn(q_text, int(cfg.query_top_k)))
        norm_scores = _normalize_scores_within_query([float(x[1]) for x in ranked])
        rec = {
            "query_type": q_type,
            "query_text": q_text,
            "query_weight": float(q_weight),
            "query_rank_decay_gamma": float(rank_gamma),
            "topk": [],
        }
        for rank_i, (obj_idx, score) in enumerate(ranked):
            oi = int(obj_idx)
            s = float(score)
            sn = float(norm_scores[rank_i]) if rank_i < len(norm_scores) else 0.0
            rm = _rank_decay_mult(rank_i)
            node_id = state.object_node.get(oi, None)
            allowed = _candidate_allowed(oi)
            rec["topk"].append(
                {
                    "object_index": oi,
                    "score": s,
                    "score_norm": sn,
                    "rank_decay_mult": float(rm),
                    "node_id": None if node_id is None else int(node_id),
                    "candidate_allowed": bool(allowed),
                }
            )
            if node_id is None or not allowed:
                continue
            w_eff = float(q_weight) * rm
            vote_count[int(node_id)] = float(vote_count.get(int(node_id), 0.0) + w_eff)
            vote_score_sum[int(node_id)] = float(vote_score_sum.get(int(node_id), 0.0) + (w_eff * sn))
            if q_type == "target":
                target_score_map[oi] = s
        query_logs.append(rec)

    _accumulate_query("target", main_target, float(cfg.target_vote_weight))

    pick_text_boost = (str(refined_pick_text).strip() if refined_pick_text else "")
    refined_differs = bool(pick_text_boost and pick_text_boost != str(main_target).strip())
    if refined_differs:
        rk = max(1, int(cfg.refined_vote_top_k))
        sub_base = float(getattr(cfg, "refined_substitute_vote_weight", 1.0))
        if candidate_pick_scores is not None:
            ranked_pairs = [
                (int(oi), float(s))
                for oi, s in sorted(candidate_pick_scores.items(), key=lambda x: -float(x[1]))[:rk]
            ]
        else:
            need = max(int(rk) * 8, int(cfg.query_top_k))
            ranked_pairs = list(query_fn(pick_text_boost, int(need)))[:rk]
        raw_ref = [float(sc) for _, sc in ranked_pairs]
        norm_ref = _normalize_scores_within_query(raw_ref)
        refined_topk: List[Dict[str, Any]] = []
        for i, (oi_raw, sc_raw) in enumerate(ranked_pairs):
            if i >= rk:
                break
            oi = int(oi_raw)
            s = float(sc_raw)
            sn = float(norm_ref[i]) if i < len(norm_ref) else 0.0
            rm = _rank_decay_mult(i)
            w = sub_base * rm
            node_id = state.object_node.get(oi, None)
            allowed = _candidate_allowed(oi)
            refined_topk.append(
                {
                    "object_index": oi,
                    "score": s,
                    "score_norm": sn,
                    "node_id": None if node_id is None else int(node_id),
                    "vote_weight": float(w),
                    "rank_decay_mult": float(rm),
                    "candidate_allowed": bool(allowed),
                }
            )
            if node_id is None or not allowed:
                continue
            vote_count[int(node_id)] = float(vote_count.get(int(node_id), 0.0) + w)
            vote_score_sum[int(node_id)] = float(vote_score_sum.get(int(node_id), 0.0) + (w * sn))
        query_logs.append(
            {
                "query_type": "refined_substitute_topk",
                "query_text": pick_text_boost,
                "query_weight": None,
                "refined_substitute_vote_weight": float(sub_base),
                "query_rank_decay_gamma": float(rank_gamma),
                "topk": refined_topk,
            }
        )
    elif pick_text_boost:
        query_logs.append(
            {
                "query_type": "refined_global_topk_skipped",
                "reason": "same_as_main_target",
                "main_target": main_target,
                "query_text": pick_text_boost,
            }
        )

    for i, a in enumerate(anchor_list):
        base_w = float(weights_list[i])
        eff_w = base_w * anc_scale
        ranked = list(query_fn(a, int(cfg.query_top_k)))
        norm_scores = _normalize_scores_within_query([float(x[1]) for x in ranked])
        rec = {
            "query_type": "anchor",
            "query_text": a,
            "query_weight": base_w,
            "anchor_effective_vote_weight": float(eff_w),
            "query_rank_decay_gamma": float(rank_gamma),
            "topk": [],
        }
        for rank_i, (obj_idx, score) in enumerate(ranked):
            oi = int(obj_idx)
            s = float(score)
            sn = float(norm_scores[rank_i]) if rank_i < len(norm_scores) else 0.0
            rm = _rank_decay_mult(rank_i)
            node_id = state.object_node.get(oi, None)
            allowed = _candidate_allowed(oi)
            rec["topk"].append(
                {
                    "object_index": oi,
                    "score": s,
                    "score_norm": sn,
                    "rank_decay_mult": float(rm),
                    "node_id": None if node_id is None else int(node_id),
                    "candidate_allowed": bool(allowed),
                }
            )
            if node_id is None or not allowed:
                continue
            w_eff = float(eff_w) * rm
            vote_count[int(node_id)] = float(vote_count.get(int(node_id), 0.0) + w_eff)
            vote_score_sum[int(node_id)] = float(vote_score_sum.get(int(node_id), 0.0) + (w_eff * sn))
        query_logs.append(rec)

    if len(vote_count) == 0:
        return {"ok": False, "reason": "no_node_received_votes", "query_logs": query_logs}

    # pick voted node
    node_items = []
    for nid, c in vote_count.items():
        node_items.append((int(nid), float(c), float(vote_score_sum.get(int(nid), 0.0))))
    # vote_score_sum 优先，vote_count 仅作平局
    node_items.sort(key=lambda x: (-x[2], -x[1], x[0]))
    best_node = int(node_items[0][0])

    # collect objects bound to best node (candidate-filtered when provided)
    node_objects = [
        oi
        for oi, nid in state.object_node.items()
        if int(nid) == best_node and _candidate_allowed(int(oi))
    ]
    if len(node_objects) == 0:
        return {
            "ok": False,
            "reason": "best_node_has_no_candidate_objects",
            "best_node": best_node,
            "query_logs": query_logs,
            "candidate_object_indices": None if candidate_set is None else sorted(list(candidate_set)),
        }

    # Fill target_score_map (vote / main_target) for diagnostics and missing-object fill
    if candidate_target_scores is not None:
        for oi, s in candidate_target_scores.items():
            target_score_map[int(oi)] = float(s)
    missing = [oi for oi in node_objects if oi not in target_score_map]
    if len(missing) > 0:
        extra = list(query_fn(main_target, max(int(cfg.query_top_k), len(node_objects) * 8)))
        for oi, s in extra:
            target_score_map[int(oi)] = float(s)

    pick_text = (str(refined_pick_text).strip() if refined_pick_text else "") or main_target
    pick_score_map: Dict[int, float] = {}
    if candidate_pick_scores is not None:
        for oi, s in candidate_pick_scores.items():
            pick_score_map[int(oi)] = float(s)
        missing_p = [oi for oi in node_objects if oi not in pick_score_map]
        if len(missing_p) > 0:
            extra_p = list(query_fn(pick_text, max(int(cfg.query_top_k), len(node_objects) * 8)))
            for oi, s in extra_p:
                pick_score_map[int(oi)] = float(s)
    elif pick_text != main_target:
        extra_p = list(query_fn(pick_text, max(int(cfg.query_top_k), len(node_objects) * 8)))
        for oi, s in extra_p:
            pick_score_map[int(oi)] = float(s)
        for oi in node_objects:
            pick_score_map.setdefault(int(oi), float(target_score_map.get(int(oi), -1e6)))
    else:
        for oi in node_objects:
            pick_score_map[int(oi)] = float(target_score_map.get(int(oi), -1e6))

    _mode = str(getattr(cfg, "object_pick_score", "pick")).strip().lower()
    if _mode not in ("pick", "target", "max"):
        _mode = "pick"
    scored_eff: List[Tuple[int, float]] = []
    for oi in node_objects:
        pt = float(pick_score_map.get(int(oi), -1e6))
        tt = float(target_score_map.get(int(oi), -1e6))
        if _mode == "target":
            eff = tt
        elif _mode == "max":
            eff = max(pt, tt)
        else:
            eff = pt
        scored_eff.append((int(oi), eff))
    scored_eff.sort(key=lambda x: (-x[1], x[0]))

    chosen_obj = int(scored_eff[0][0])
    pick_pool = scored_eff[: max(1, min(int(cfg.node_pick_top_k), len(scored_eff)))]
    pool_eff = np.asarray([float(eff) for _, eff in pick_pool], dtype=float)
    probs = _softmax(pool_eff, cfg.softmax_temp)

    return {
        "ok": True,
        "main_target": main_target,
        "pick_query_text": pick_text,
        "anchors": list(anchors),
        "query_logs": query_logs,
        "node_votes": [
            {
                "node_id": int(nid),
                "vote_count": float(cnt),
                "vote_score_sum": float(score_sum),
                "node_center": state.nodes[int(nid)].tolist() if int(nid) < len(state.nodes) else None,
            }
            for nid, cnt, score_sum in node_items
        ],
        "best_node_id": best_node,
        "best_node_objects": [int(x) for x in node_objects],
        "candidate_object_indices": None if candidate_set is None else sorted(list(candidate_set)),
        "selected_from_candidates": True if candidate_set is not None else False,
        "object_pick_score_mode": _mode,
        "object_scores_in_best_node": [
            {
                "object_index": int(oi),
                "effective_score": float(eff),
                "pick_score": float(pick_score_map.get(int(oi), -1e6)),
                "target_score": float(target_score_map.get(int(oi), -1e6)),
            }
            for oi, eff in scored_eff
        ],
        "pick_pool": [
            {"object_index": int(oi), "effective_score": float(eff), "score": float(eff)} for oi, eff in pick_pool
        ],
        "pick_probs": probs.tolist(),
        "pick_stochastic": False,
        "chosen_object_index": chosen_obj,
        "node_pick_primary": "score_sum",
    }


def pq3d_stage2_object_logits(pq3d_model: Any, sentence: str) -> np.ndarray:
    """
    Run PQ3D **Stage2** (Query3DVLE) on the current ``representation_manager`` state with text ``sentence``,
    **without frontier queries** — same tensor layout as ``PQ3DModel.decision`` when ``frontier_list`` is empty.

    Returns ``og3d_logits`` for each real memory object, shape ``(N,)``, index ``i`` matches memory row ``i``.
    """
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
    real_logits = logits[mask]
    return real_logits.astype(np.float64)


def build_query_fn_from_pq3d_stage2(pq3d_model: Any) -> QueryFn:
    """
    Ranking callback using **PQ3D Stage2** text→object logits (same pathway as ``decision``, no CLIP).

    Use with ``run_position_vote(..., candidate_target_scores=None, candidate_pick_scores=None)`` so all
    vote queries (target / substitute top-k / anchors) use this backend; or call
    ``run_position_vote_with_pq3d_stage2``.
    """

    def _query_fn(text: str, top_k: int) -> Sequence[Tuple[int, float]]:
        scores = pq3d_stage2_object_logits(pq3d_model, text)
        if scores.size == 0:
            return []
        k = max(1, int(top_k))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    return _query_fn


def run_position_vote_with_pq3d_stage2(
    state: VoteState,
    *,
    pq3d_model: Any,
    main_target: str,
    anchors: Sequence[str],
    cfg: VoteConfig,
    anchor_weights: Optional[Sequence[float]] = None,
    candidate_object_indices: Optional[Sequence[int]] = None,
    refined_pick_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Spatial vote + node pick where **every** text query (target, anchors, refined global top-k, node object
    scores) is scored by PQ3D Stage2 — not open-vocab CLIP.
    """
    query_fn = build_query_fn_from_pq3d_stage2(pq3d_model)
    out = run_position_vote(
        state,
        query_fn=query_fn,
        main_target=main_target,
        anchors=anchors,
        anchor_weights=anchor_weights,
        cfg=cfg,
        candidate_object_indices=candidate_object_indices,
        candidate_target_scores=None,
        refined_pick_text=refined_pick_text,
        candidate_pick_scores=None,
    )
    out["query_backend"] = "pq3d_stage2_og3d"
    return out


def build_query_fn_from_open_vocab(
    *,
    rep: Any,
    text_encoder_fn: TextEncoderFn,
) -> QueryFn:
    """
    Build a lightweight query_fn on top of representation_manager.open_vocab_feat.
    This is a pure scoring path (no perception / no decision()).
    """
    ov = np.asarray(getattr(rep, "open_vocab_feat", np.zeros((0, 768))), dtype=float)
    if ov.ndim != 2:
        ov = ov.reshape(0, 768)
    ov_norm = np.linalg.norm(ov, axis=1) + 1e-12 if len(ov) > 0 else np.zeros((0,), dtype=float)

    def _query_fn(text: str, top_k: int) -> Sequence[Tuple[int, float]]:
        if len(ov) == 0:
            return []
        q = np.asarray(text_encoder_fn(text), dtype=float).reshape(-1)
        qn = float(np.linalg.norm(q) + 1e-12)
        sim = (ov @ q) / (ov_norm * qn)
        idx = np.argsort(-sim)[: max(1, int(top_k))]
        return [(int(i), float(sim[int(i)])) for i in idx]

    return _query_fn


def bind_from_object_first_positions(
    state: VoteState,
    *,
    object_first_positions: Dict[int, Any],
    cfg: VoteConfig,
) -> List[Dict[str, Any]]:
    """
    Bulk binding helper:
    Bind/rebind objects using pre-recorded first-seen positions.
    This matches the 'observation position' semantics and is framework-agnostic.
    """
    records: List[Dict[str, Any]] = []
    for obj_idx, pos in object_first_positions.items():
        records.append(
            bind_object_to_position(
                state,
                object_index=int(obj_idx),
                position_xyz=pos,
                cfg=cfg,
            )
        )
    return records


def run_position_vote_with_open_vocab(
    state: VoteState,
    *,
    rep: Any,
    text_encoder_fn: TextEncoderFn,
    main_target: str,
    anchors: Sequence[str],
    cfg: VoteConfig,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - build query_fn from open_vocab features in current memory
    - run position vote
    Pure scoring path, no decision() required.
    """
    query_fn = build_query_fn_from_open_vocab(rep=rep, text_encoder_fn=text_encoder_fn)
    out = run_position_vote(
        state,
        query_fn=query_fn,
        main_target=main_target,
        anchors=anchors,
        cfg=cfg,
    )
    out["query_backend"] = "open_vocab_cosine"
    return out
