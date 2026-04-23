from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class WakeConfig:
    repeat_threshold: int = 2
    empty_path_threshold: int = 3
    force_final_decision_on_stuck: bool = False


@dataclass
class WakeState:
    last_frontier_key: Optional[Tuple[float, float, float]] = None
    repeated_count: int = 0


def _as_key(xyz: Any) -> Tuple[float, float, float]:
    arr = np.asarray(xyz, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError("xyz must have at least 3 dims")
    return tuple(np.round(arr[:3], 3).tolist())


def _pick_alternative_frontier(
    frontier_waypoints: Sequence[Any],
    avoid_key: Tuple[float, float, float],
    agent_position_xyz: Any,
    blocked_frontier_keys: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Optional[np.ndarray]:
    if len(frontier_waypoints) == 0:
        return None
    agent = np.asarray(agent_position_xyz, dtype=float).reshape(-1)[:3]
    cands: List[np.ndarray] = []
    blocked_set = set(blocked_frontier_keys or [])
    for fw in frontier_waypoints:
        p = np.asarray(fw, dtype=float).reshape(-1)[:3]
        pkey = tuple(np.round(p, 3).tolist())
        if pkey == avoid_key:
            continue
        if pkey in blocked_set:
            continue
        cands.append(p)
    if len(cands) == 0:
        return None
    d = [float(np.linalg.norm(p - agent)) for p in cands]
    return cands[int(np.argmin(np.asarray(d, dtype=float)))]


def apply_wake(
    state: WakeState,
    *,
    cfg: WakeConfig,
    is_final: bool,
    target_position_xyz: Any,
    frontier_waypoints: Sequence[Any],
    agent_position_xyz: Any,
    frontier_empty_path_counts: Optional[Dict[Tuple[float, float, float], int]] = None,
) -> Dict[str, Any]:
    """
    Wake strategy:
    - Track repeated non-final frontier target.
    - If same target repeats >= repeat_threshold, redirect to another frontier immediately.
    - If no alternative frontier and force_final_decision_on_stuck=True, trigger final decision.
    """
    if is_final:
        state.last_frontier_key = None
        state.repeated_count = 0
        return {
            "triggered": False,
            "reason": "already_final",
            "redirected_target": None,
            "force_final_decision": False,
            "repeat_count": 0,
        }

    cur_key = _as_key(target_position_xyz)
    if state.last_frontier_key is not None and state.last_frontier_key == cur_key:
        state.repeated_count += 1
    else:
        state.last_frontier_key = cur_key
        state.repeated_count = 1

    if state.repeated_count < int(cfg.repeat_threshold):
        return {
            "triggered": False,
            "reason": "repeat_not_reached",
            "redirected_target": None,
            "force_final_decision": False,
            "repeat_count": int(state.repeated_count),
        }

    blocked_frontier_keys: List[Tuple[float, float, float]] = []
    if frontier_empty_path_counts is not None:
        for k, v in frontier_empty_path_counts.items():
            if int(v) >= int(cfg.empty_path_threshold):
                blocked_frontier_keys.append(k)

    alt = _pick_alternative_frontier(
        frontier_waypoints=frontier_waypoints,
        avoid_key=cur_key,
        agent_position_xyz=agent_position_xyz,
        blocked_frontier_keys=blocked_frontier_keys,
    )
    if alt is not None:
        state.last_frontier_key = tuple(np.round(alt, 3).tolist())
        state.repeated_count = 1
        return {
            "triggered": True,
            "reason": "redirect_to_other_frontier",
            "redirected_target": alt.tolist(),
            "force_final_decision": False,
            "repeat_count": int(cfg.repeat_threshold),
            "blocked_frontier_count": int(len(blocked_frontier_keys)),
        }

    do_force_final = bool(cfg.force_final_decision_on_stuck)
    if do_force_final:
        state.last_frontier_key = None
        state.repeated_count = 0
    return {
        "triggered": bool(do_force_final),
        "reason": "force_final_decision" if do_force_final else "no_alternative_frontier",
        "redirected_target": None,
        "force_final_decision": bool(do_force_final),
        "repeat_count": int(cfg.repeat_threshold),
        "blocked_frontier_count": int(len(blocked_frontier_keys)),
    }
