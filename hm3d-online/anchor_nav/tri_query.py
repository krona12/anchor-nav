from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vote import pq3d_stage2_object_logits


QueryFn = Callable[[str, int], Sequence[Tuple[int, float]]]


@dataclass
class TriQueryConfig:
    top_k: int = 16
    sigma_anchor: float = 2.0
    sigma_full: float = 3.0
    anchor_distance_mode: str = "centroid"
    full_distance_mode: str = "centroid"
    centroid_temp: float = 0.07
    anchor_entropy_temp: float = 1.0
    anchor_entropy_scale: float = 2.0


def _min_max_normalize(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.ones_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _build_position_array(box: np.ndarray, ranked: Sequence[Tuple[int, float]]) -> np.ndarray:
    if len(ranked) == 0:
        return np.zeros((0, 3), dtype=float)
    out: List[np.ndarray] = []
    for obj_idx, _ in ranked:
        i = int(obj_idx)
        if i < 0 or i >= int(box.shape[0]):
            continue
        out.append(np.asarray(box[i, :3], dtype=float).reshape(3))
    if len(out) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.stack(out, axis=0)


def _min_nn_distance(p: np.ndarray, pts: np.ndarray) -> float:
    d = np.linalg.norm(pts - p[None, :], axis=1)
    return float(np.min(d))


def _weighted_centroid_from_ranked(
    box: np.ndarray,
    ranked: Sequence[Tuple[int, float]],
    *,
    temp: float,
) -> Optional[np.ndarray]:
    if len(ranked) == 0:
        return None
    pts: List[np.ndarray] = []
    raw_scores: List[float] = []
    for obj_idx, score in ranked:
        i = int(obj_idx)
        if i < 0 or i >= int(box.shape[0]):
            continue
        pts.append(np.asarray(box[i, :3], dtype=float).reshape(3))
        raw_scores.append(float(score))
    if len(pts) == 0:
        return None
    p = np.stack(pts, axis=0)
    s = np.asarray(raw_scores, dtype=float).reshape(-1)
    t = max(float(temp), 1e-6)
    z = (s - float(np.max(s))) / t
    w = np.exp(z)
    w = w / max(float(np.sum(w)), 1e-12)
    c = np.sum(p * w[:, None], axis=0)
    return np.asarray(c, dtype=float).reshape(3)


def _anchor_entropy_sigma(
    ranked: Sequence[Tuple[int, float]],
    *,
    sigma_base: float,
    entropy_temp: float,
    entropy_scale: float,
) -> Tuple[float, float]:
    """
    根据 anchor top-K 分数分布的 Shannon 熵动态调整 sigma。
    Returns:
        sigma_adaptive: 调整后的 sigma（米）
        H_norm: 归一化熵 [0, 1]
    """
    if len(ranked) == 0:
        return float(sigma_base) * float(np.exp(float(entropy_scale))), 1.0

    scores = np.asarray([float(s) for _, s in ranked], dtype=float).reshape(-1)
    k = int(scores.shape[0])
    t = max(float(entropy_temp), 1e-6)
    z = scores / t
    z = z - float(np.max(z))
    p = np.exp(z)
    p = p / max(float(np.sum(p)), 1e-12)

    h = float(-np.sum(p * np.log(p + 1e-12)))
    h_max = float(np.log(max(k, 1)))
    h_norm = float(h / h_max) if h_max > 1e-12 else 1.0
    sigma_adaptive = float(sigma_base) * float(np.exp(float(entropy_scale) * h_norm))
    return sigma_adaptive, h_norm


def run_tri_query(
    *,
    description: str,
    rep: Any,
    query_fn: QueryFn,
    main_target: str,
    nearest_anchor: str,
    cfg: TriQueryConfig,
) -> Dict[str, Any]:
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] < 3:
        return {"ok": False, "reason": "empty_memory"}

    q_target = str(main_target).strip()
    if not q_target:
        return {"ok": False, "reason": "empty_target_query"}

    top_k = int(cfg.top_k)
    sigma_anchor_base = float(cfg.sigma_anchor)
    sigma_full = float(cfg.sigma_full)
    anchor_mode = str(getattr(cfg, "anchor_distance_mode", "centroid")).strip().lower()
    full_mode = str(getattr(cfg, "full_distance_mode", "centroid")).strip().lower()
    if anchor_mode not in {"min", "centroid"}:
        anchor_mode = "centroid"
    if full_mode not in {"min", "centroid"}:
        full_mode = "centroid"
    centroid_temp = float(getattr(cfg, "centroid_temp", 0.07))
    entropy_temp = float(getattr(cfg, "anchor_entropy_temp", 1.0))
    entropy_scale = float(getattr(cfg, "anchor_entropy_scale", 2.0))

    res_full = list(query_fn(str(description), top_k))
    res_target = list(query_fn(q_target, top_k))
    if len(res_target) == 0:
        return {
            "ok": False,
            "reason": "empty_target_query",
            "query_logs": {
                "full_query": str(description),
                "target_query": q_target,
                "anchor_query": str(nearest_anchor).strip(),
                "full_topk": res_full,
                "target_topk": [],
                "anchor_topk": [],
            },
        }
    q_anchor = str(nearest_anchor).strip()
    res_anchor = list(query_fn(q_anchor, top_k)) if q_anchor else []
    sigma_anchor, anchor_h_norm = _anchor_entropy_sigma(
        res_anchor,
        sigma_base=sigma_anchor_base,
        entropy_temp=entropy_temp,
        entropy_scale=entropy_scale,
    )

    pos_full = _build_position_array(box, res_full)
    pos_anchor = _build_position_array(box, res_anchor)
    center_full = _weighted_centroid_from_ranked(box, res_full, temp=centroid_temp)
    center_anchor = _weighted_centroid_from_ranked(box, res_anchor, temp=centroid_temp)

    target_raw_scores = [float(s) for _, s in res_target]
    target_norm_scores = _min_max_normalize(target_raw_scores)

    rows: List[Dict[str, Any]] = []
    for rank_i, (obj_idx, raw_score) in enumerate(res_target):
        i = int(obj_idx)
        if i < 0 or i >= int(box.shape[0]):
            continue
        p_i = np.asarray(box[i, :3], dtype=float).reshape(3)

        if anchor_mode == "centroid" and center_anchor is not None:
            d_a = float(np.linalg.norm(p_i - center_anchor))
            prox_anchor = float(np.exp(-d_a / sigma_anchor))
        elif pos_anchor.shape[0] > 0:
            d_a = _min_nn_distance(p_i, pos_anchor)
            prox_anchor = float(np.exp(-d_a / sigma_anchor))
        else:
            d_a = None
            prox_anchor = 1.0

        if full_mode == "centroid" and center_full is not None:
            d_f = float(np.linalg.norm(p_i - center_full))
            prox_full = float(np.exp(-d_f / sigma_full))
        elif pos_full.shape[0] > 0:
            d_f = _min_nn_distance(p_i, pos_full)
            prox_full = float(np.exp(-d_f / sigma_full))
        else:
            d_f = None
            prox_full = 1.0

        semantic = float(target_norm_scores[rank_i]) if rank_i < len(target_norm_scores) else 0.0
        spatial = float(prox_anchor * prox_full)
        final_score = float(semantic * spatial)
        rows.append(
            {
                "rank_in_target": int(rank_i + 1),
                "object_index": int(i),
                "target_raw_score": float(raw_score),
                "semantic_score": float(semantic),
                "d_anchor": None if d_a is None else float(d_a),
                "prox_anchor": float(prox_anchor),
                "d_full": None if d_f is None else float(d_f),
                "prox_full": float(prox_full),
                "spatial_consensus": float(spatial),
                "final_score": float(final_score),
            }
        )

    if len(rows) == 0:
        return {"ok": False, "reason": "empty_target_candidates_after_filter"}

    rows.sort(key=lambda x: (-float(x["final_score"]), int(x["object_index"])))
    chosen_obj = int(rows[0]["object_index"])
    chosen_xyz = np.asarray(box[chosen_obj, :3], dtype=float).reshape(3).copy()
    chosen_xyz[[1, 2]] = chosen_xyz[[2, 1]]

    return {
        "ok": True,
        "chosen_object_index": int(chosen_obj),
        "target_xyz": chosen_xyz.tolist(),
        "final_ranking": rows,
        "query_logs": {
            "full_query": str(description),
            "target_query": q_target,
            "anchor_query": q_anchor,
            "full_topk": [{"object_index": int(i), "score": float(s)} for i, s in res_full],
            "target_topk": [{"object_index": int(i), "score": float(s)} for i, s in res_target],
            "anchor_topk": [{"object_index": int(i), "score": float(s)} for i, s in res_anchor],
        },
        "config": {
            "top_k": int(top_k),
            "sigma_anchor_base": float(sigma_anchor_base),
            "sigma_anchor_adaptive": float(sigma_anchor),
            "anchor_H_norm": float(anchor_h_norm),
            "sigma_full": float(sigma_full),
            "anchor_distance_mode": anchor_mode,
            "full_distance_mode": full_mode,
            "centroid_temp": float(centroid_temp),
            "anchor_entropy_temp": float(entropy_temp),
            "anchor_entropy_scale": float(entropy_scale),
        },
    }


def build_query_fn_from_pq3d_stage2(pq3d_model: Any) -> QueryFn:
    def _query_fn(text: str, top_k: int) -> Sequence[Tuple[int, float]]:
        logits = pq3d_stage2_object_logits(pq3d_model, text)
        if logits.size == 0:
            return []
        k = max(1, int(top_k))
        idx = np.argsort(-logits)[:k]
        return [(int(i), float(logits[i])) for i in idx]

    return _query_fn
