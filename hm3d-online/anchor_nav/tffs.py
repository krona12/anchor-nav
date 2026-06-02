from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from vlm.client import DEFAULT_MODEL, chat as _vlm_chat

    _VLM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on deployment path.
    DEFAULT_MODEL = "gpt-4o-mini"
    _vlm_chat = None
    _VLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODULE_NAME = "tffs"
PROMPT_VERSION = "tffs_task_facing_visual_hypothesis_v1"


@dataclass
class TffsConfig:
    alpha_logit: float = 0.65
    beta_hypothesis: float = 0.35
    target_context_weight: float = 0.40
    anchor_context_weight: float = 0.25
    room_context_weight: float = 0.25
    negative_weight: float = 0.40
    min_candidates: int = 2
    min_task_chars: int = 3
    min_score_margin: float = 0.05
    min_confidence: float = 0.45
    max_vlm_candidates: int = 16
    vlm_call_interval: int = 1
    max_vlm_calls_per_decision: int = 4
    top_vlm_logit_candidates: int = 3
    min_logit_gap_for_vlm: float = 0.12
    enable_score_cache: bool = False
    cache_position_quantization_m: float = 0.25
    cache_heading_quantization_deg: float = 15.0
    min_frontier_distance_m: float = 1e-4
    require_image_file_exists: bool = True
    use_vlm: bool = True
    vlm_model: str = DEFAULT_MODEL
    vlm_max_tokens: int = 256
    vlm_max_retries: int = 2
    vlm_retry_sleep_sec: float = 0.5
    vlm_no_proxy: bool = True
    fallback_to_logits_only: bool = True
    prompt_version: str = PROMPT_VERSION
    non_oracle_inputs: Tuple[str, ...] = field(
        default_factory=lambda: (
            "task_text",
            "frontier_candidates",
            "frontier_logits",
            "current_panorama_views",
            "agent_pose",
        )
    )


class TffsDecisionError(RuntimeError):
    pass


class TffsInputError(TffsDecisionError):
    pass


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
    old = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS + ("NO_PROXY", "no_proxy")}
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


def _clamp01(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_jsonable(x) for x in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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
    raise RuntimeError("TFFS VLM response is not a JSON object: " + " | ".join(errors))


def _safe_float_list(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    out = arr[:3].astype(float)
    if not np.all(np.isfinite(out)):
        return None
    return out


def _mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Optional[Any], str]:
    lowered = {str(k).lower(): k for k in mapping.keys()}
    for key in keys:
        actual = lowered.get(str(key).lower())
        if actual is not None:
            return mapping[actual], str(actual)
    return None, ""


def _object_attr(value: Any, names: Sequence[str]) -> Tuple[Optional[Any], str]:
    for name in names:
        if hasattr(value, name):
            try:
                return getattr(value, name), str(name)
            except Exception:
                continue
    return None, ""


def _extract_xyz(value: Any, *, dict_keys: Sequence[str], name: str) -> Tuple[Optional[np.ndarray], str]:
    if isinstance(value, Mapping):
        direct = _safe_float_list(value)
        if direct is not None:
            return direct, name
        raw, key = _mapping_value(value, dict_keys)
        if raw is not None:
            arr = _safe_float_list(raw)
            if arr is not None:
                return arr, key
        for key in dict_keys:
            nested = value.get(key) if key in value else None
            if isinstance(nested, Mapping):
                arr = _safe_float_list(nested)
                if arr is not None:
                    return arr, key
        return None, ""
    arr = _safe_float_list(value)
    if arr is not None:
        return arr, name
    raw, attr = _object_attr(value, dict_keys)
    if raw is not None:
        arr = _safe_float_list(raw)
        if arr is not None:
            return arr, attr
    return None, ""


def _coerce_frontier_records(frontier_candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if frontier_candidates is None:
        return records
    if isinstance(frontier_candidates, np.ndarray):
        arr = np.asarray(frontier_candidates, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for idx in range(arr.shape[0]):
            xyz = _safe_float_list(arr[idx])
            records.append(
                {
                    "frontier_index": int(idx),
                    "point_xyz": xyz.tolist() if xyz is not None else None,
                    "point_source": "ndarray",
                    "valid_point": bool(xyz is not None),
                }
            )
        return records

    point_keys = (
        "point",
        "position",
        "xyz",
        "habitat_xyz",
        "frontier_xyz",
        "target_position",
        "loc",
        "location",
        "center",
    )
    for idx, item in enumerate(list(frontier_candidates)):
        xyz, source = _extract_xyz(item, dict_keys=point_keys, name="sequence")
        rec: Dict[str, Any] = {
            "frontier_index": int(idx),
            "point_xyz": xyz.tolist() if xyz is not None else None,
            "point_source": source,
            "valid_point": bool(xyz is not None),
        }
        if isinstance(item, Mapping):
            for key in ("id", "frontier_id", "rank", "visited", "source"):
                if key in item:
                    rec[str(key)] = _jsonable(item[key])
        records.append(rec)
    return records


def _coerce_logits(frontier_logits: Optional[Sequence[float]], n: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if n <= 0:
        return np.zeros((0,), dtype=float), {"frontier_logits_valid": False, "frontier_logits_reason": "no_candidates"}
    if frontier_logits is None:
        return np.zeros((n,), dtype=float), {
            "frontier_logits_valid": False,
            "frontier_logits_reason": "missing_frontier_logits",
        }
    try:
        arr = np.asarray(frontier_logits, dtype=float).reshape(-1)
    except Exception as exc:
        return np.zeros((n,), dtype=float), {
            "frontier_logits_valid": False,
            "frontier_logits_reason": f"logit_parse_error:{type(exc).__name__}",
        }
    valid = bool(arr.shape[0] == n and np.all(np.isfinite(arr)))
    reason = "ok" if valid else f"logit_shape_or_value_mismatch:got={arr.shape[0]} expected={n}"
    if arr.shape[0] < n:
        arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=0.0)
    arr = arr[:n].astype(float)
    if not np.all(np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        fill = float(np.min(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr, {"frontier_logits_valid": valid, "frontier_logits_reason": reason}


def _normalize_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=float)
    clean = np.where(finite, arr, np.min(arr[finite]))
    lo = float(np.min(clean))
    hi = float(np.max(clean))
    if hi <= lo + 1e-12:
        return np.zeros_like(clean, dtype=float)
    return (clean - lo) / (hi - lo)


def _coerce_view_record(view: Any, default_index: int = 0) -> Dict[str, Any]:
    path_keys = (
        "image_path",
        "panorama_path",
        "rgb_path",
        "path",
        "file",
        "filename",
        "image",
    )
    rec: Dict[str, Any] = {"view_index": int(default_index), "image_path": None, "view_source": type(view).__name__}
    if isinstance(view, (str, Path)):
        rec["image_path"] = str(view)
        rec["view_source"] = "path"
        return rec
    if isinstance(view, Mapping):
        if "view_index" in view:
            try:
                rec["view_index"] = int(view["view_index"])
            except Exception:
                rec["view_index"] = int(default_index)
        raw, key = _mapping_value(view, path_keys)
        if isinstance(raw, (str, Path)):
            rec["image_path"] = str(raw)
            rec["view_source"] = key
        for meta_key in ("yaw", "heading", "camera_heading", "timestamp"):
            if meta_key in view:
                rec[meta_key] = _jsonable(view[meta_key])
        return rec
    raw, attr = _object_attr(view, path_keys)
    if isinstance(raw, (str, Path)):
        rec["image_path"] = str(raw)
        rec["view_source"] = attr
    raw_index, _ = _object_attr(view, ("view_index", "index"))
    if raw_index is not None:
        try:
            rec["view_index"] = int(raw_index)
        except Exception:
            pass
    return rec


def _coerce_panorama_views(panorama_views: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    if panorama_views is None:
        return []
    if isinstance(panorama_views, (str, Path, Mapping)):
        return [_coerce_view_record(panorama_views, 0)]
    return [_coerce_view_record(view, idx) for idx, view in enumerate(list(panorama_views))]


def _heading_from_yaw(yaw: Any) -> Optional[np.ndarray]:
    try:
        value = float(yaw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return np.asarray([math.sin(value), math.cos(value)], dtype=float)


def _heading_xz(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] >= 3:
        out = arr[[0, 2]].astype(float)
    elif arr.shape[0] >= 2:
        out = arr[:2].astype(float)
    else:
        return None
    norm = float(np.linalg.norm(out))
    if norm <= 1e-9 or not np.all(np.isfinite(out)):
        return None
    return out / norm


def _coerce_agent_pose(agent_pose: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "agent_xyz": None,
        "heading_xz": None,
        "pose_ok": False,
        "pose_reason": "",
        "position_source": "",
        "heading_source": "",
    }
    if agent_pose is None:
        info["pose_reason"] = "missing_agent_pose"
        return None, None, info

    pos_keys = (
        "xyz",
        "position",
        "point",
        "habitat_xyz",
        "agent_xyz",
        "agent_position_xyz",
        "pos",
        "location",
    )
    heading_keys = (
        "heading",
        "heading_vector",
        "forward",
        "forward_vector",
        "direction",
        "dir",
        "look_vector",
    )
    yaw_keys = ("yaw", "heading_yaw", "heading_rad", "radian", "radians", "theta")

    xyz: Optional[np.ndarray]
    heading: Optional[np.ndarray] = None
    pos_source = ""
    heading_source = ""

    if isinstance(agent_pose, Mapping):
        xyz, pos_source = _extract_xyz(agent_pose, dict_keys=pos_keys, name="agent_pose")
        raw_heading, heading_source = _mapping_value(agent_pose, heading_keys)
        heading = _heading_xz(raw_heading)
        if heading is None:
            raw_yaw, yaw_source = _mapping_value(agent_pose, yaw_keys)
            heading = _heading_from_yaw(raw_yaw)
            heading_source = yaw_source if heading is not None else heading_source
    else:
        xyz, pos_source = _extract_xyz(agent_pose, dict_keys=pos_keys, name="agent_pose")
        if xyz is None:
            try:
                arr = np.asarray(agent_pose, dtype=float).reshape(-1)
            except Exception:
                arr = np.zeros((0,), dtype=float)
            if arr.shape[0] >= 3:
                xyz = arr[:3].astype(float)
                pos_source = "sequence"
                if arr.shape[0] >= 6:
                    heading = _heading_xz(arr[3:6])
                    heading_source = "sequence_heading"
                elif arr.shape[0] >= 4:
                    heading = _heading_from_yaw(arr[3])
                    heading_source = "sequence_yaw" if heading is not None else ""
        if heading is None:
            raw_heading, heading_source = _object_attr(agent_pose, heading_keys)
            heading = _heading_xz(raw_heading)
        if heading is None:
            raw_yaw, yaw_source = _object_attr(agent_pose, yaw_keys)
            heading = _heading_from_yaw(raw_yaw)
            heading_source = yaw_source if heading is not None else heading_source

    if xyz is not None:
        info["agent_xyz"] = xyz.tolist()
        info["position_source"] = pos_source
    if heading is not None:
        info["heading_xz"] = heading.tolist()
        info["heading_source"] = heading_source
    info["pose_ok"] = bool(xyz is not None and heading is not None)
    if not info["pose_ok"]:
        if xyz is None:
            info["pose_reason"] = "agent_xyz_unavailable"
        elif heading is None:
            info["pose_reason"] = "agent_heading_unavailable"
    else:
        info["pose_reason"] = "ok"
    return xyz, heading, info


def _frontier_view_assignment(
    *,
    frontier_xyz: np.ndarray,
    agent_xyz: np.ndarray,
    heading_xz: np.ndarray,
    n_views: int,
    cfg: TffsConfig,
) -> Dict[str, Any]:
    if n_views <= 0:
        return {"view_index": None, "relative_bearing_rad": None, "bearing_ok": False, "bearing_reason": "no_views"}
    frontier = np.asarray(frontier_xyz, dtype=float).reshape(3)
    agent = np.asarray(agent_xyz, dtype=float).reshape(3)
    delta = frontier[[0, 2]] - agent[[0, 2]]
    dist = float(np.linalg.norm(delta))
    if dist <= float(cfg.min_frontier_distance_m) or not math.isfinite(dist):
        return {
            "view_index": None,
            "relative_bearing_rad": None,
            "frontier_distance_m": dist,
            "bearing_ok": False,
            "bearing_reason": "frontier_too_close_or_invalid",
        }
    v = delta / dist
    h = np.asarray(heading_xz, dtype=float).reshape(2)
    h_norm = float(np.linalg.norm(h))
    if h_norm <= 1e-9 or not np.all(np.isfinite(h)):
        return {
            "view_index": None,
            "relative_bearing_rad": None,
            "frontier_distance_m": dist,
            "bearing_ok": False,
            "bearing_reason": "invalid_heading",
        }
    h = h / h_norm
    cross = float(h[0] * v[1] - h[1] * v[0])
    dot = float(np.clip(np.dot(h, v), -1.0, 1.0))
    phi = float(math.atan2(cross, dot))
    if n_views == 1:
        view_index = 0
    else:
        raw = (phi + math.pi) / (2.0 * math.pi) * float(n_views - 1)
        view_index = int(np.clip(math.floor(raw + 0.5), 0, n_views - 1))
    return {
        "view_index": int(view_index),
        "relative_bearing_rad": float(phi),
        "frontier_distance_m": dist,
        "bearing_ok": True,
        "bearing_reason": "ok",
    }


def _task_text_available(task_text: str, cfg: TffsConfig) -> bool:
    text = str(task_text or "").strip()
    if len(text) < int(cfg.min_task_chars):
        return False
    return any(not ch.isspace() for ch in text)


def _choose_baseline_index(
    *,
    baseline_frontier_index: Optional[int],
    logits: np.ndarray,
    n: int,
) -> Tuple[int, Dict[str, Any]]:
    if n <= 0:
        return -1, {"baseline_index_source": "no_candidates", "baseline_index_valid": False}
    if baseline_frontier_index is not None:
        try:
            idx = int(baseline_frontier_index)
        except Exception:
            idx = -1
        if 0 <= idx < n:
            return idx, {"baseline_index_source": "provided", "baseline_index_valid": True}
    if logits.size == n:
        return int(np.argmax(logits)), {"baseline_index_source": "argmax_frontier_logits", "baseline_index_valid": True}
    return 0, {"baseline_index_source": "default_zero", "baseline_index_valid": True}


def _vlm_decision_counter(context: Optional[Mapping[str, Any]]) -> Tuple[Optional[int], str]:
    if not isinstance(context, Mapping):
        return None, ""
    for key in ("decision_num", "decision_index", "decision_step", "step", "global_step", "t"):
        if key not in context:
            continue
        try:
            value = int(context[key])
        except Exception:
            continue
        if value >= 0:
            return value, key
    return None, ""


def _vlm_interval_gate(context: Optional[Mapping[str, Any]], cfg: TffsConfig) -> Tuple[bool, Dict[str, Any]]:
    try:
        interval = int(cfg.vlm_call_interval)
    except Exception:
        interval = 1
    interval = max(1, interval)
    counter, source = _vlm_decision_counter(context)
    if interval <= 1:
        return True, {
            "vlm_call_interval": int(interval),
            "vlm_interval_allowed": True,
            "vlm_interval_counter": counter,
            "vlm_interval_counter_source": source,
            "cost_control_reason": "vlm_interval_1_allow",
        }
    if counter is None:
        return True, {
            "vlm_call_interval": int(interval),
            "vlm_interval_allowed": True,
            "vlm_interval_counter": None,
            "vlm_interval_counter_source": "",
            "cost_control_reason": "vlm_interval_counter_missing_allow",
        }
    allowed = bool(counter <= 1 or counter % interval == 0)
    reason = "vlm_interval_allow" if allowed else "vlm_interval_skip_keep_baseline"
    return allowed, {
        "vlm_call_interval": int(interval),
        "vlm_interval_allowed": bool(allowed),
        "vlm_interval_counter": int(counter),
        "vlm_interval_counter_source": source,
        "cost_control_reason": reason,
    }


def _vlm_call_budget(cfg: TffsConfig, n: int) -> int:
    if n <= 0:
        return 0
    try:
        budget = int(cfg.max_vlm_calls_per_decision)
    except Exception:
        budget = 0
    budget = max(0, budget)
    try:
        legacy_limit = int(cfg.max_vlm_candidates)
    except Exception:
        legacy_limit = 0
    if legacy_limit > 0:
        budget = min(budget, legacy_limit)
    return int(min(budget, n))


def _candidate_indices_for_vlm(norm_logits: np.ndarray, baseline_idx: int, cfg: TffsConfig) -> List[int]:
    n = int(norm_logits.shape[0])
    limit = _vlm_call_budget(cfg, n)
    if limit <= 0:
        return []
    if n <= limit:
        return list(range(n))
    out: List[int] = []

    def add(idx: int) -> None:
        if 0 <= int(idx) < n and int(idx) not in out and len(out) < limit:
            out.append(int(idx))

    if 0 <= baseline_idx < n:
        add(int(baseline_idx))

    sorted_indices = [int(idx) for idx in np.argsort(-norm_logits)]
    top_k = max(0, int(cfg.top_vlm_logit_candidates))
    for idx in sorted_indices[:top_k]:
        add(idx)
        if len(out) >= limit:
            break

    if len(out) < limit and 0 <= baseline_idx < n:
        baseline_score = float(norm_logits[int(baseline_idx)])
        try:
            gap = float(cfg.min_logit_gap_for_vlm)
        except Exception:
            gap = 0.0
        gap = max(0.0, gap)
        competitive = [
            int(idx)
            for idx in sorted_indices
            if float(norm_logits[int(idx)]) >= baseline_score - gap
        ]
        for idx in competitive:
            add(idx)
            if len(out) >= limit:
                break

    for idx in sorted_indices:
        add(idx)
        if len(out) >= limit:
            break
    return out


def _quantize_scalar(value: Any, quantum: float) -> Optional[float]:
    try:
        out = float(value)
        step = float(quantum)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    if not math.isfinite(step) or step <= 0.0:
        return round(out, 6)
    return round(round(out / step) * step, 6)


def _quantize_xyz(value: Any, quantum: float) -> Optional[Tuple[float, float, float]]:
    arr = _safe_float_list(value)
    if arr is None:
        return None
    quantized = [_quantize_scalar(x, quantum) for x in arr[:3]]
    if any(x is None for x in quantized):
        return None
    return tuple(float(x) for x in quantized if x is not None)


def _heading_deg_from_xz(heading_xz: Optional[np.ndarray]) -> Optional[float]:
    if heading_xz is None:
        return None
    try:
        heading = np.asarray(heading_xz, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(heading)):
        return None
    norm = float(np.linalg.norm(heading))
    if norm <= 1e-9:
        return None
    heading = heading / norm
    return float((math.degrees(math.atan2(float(heading[0]), float(heading[1]))) + 360.0) % 360.0)


def _score_cache_from_context(context: Optional[Mapping[str, Any]]) -> MutableMapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    for key in ("tffs_score_cache", "score_cache", "vlm_score_cache"):
        cache = context.get(key)
        if isinstance(cache, MutableMapping):
            return cache
        if isinstance(cache, Mapping):
            return dict(cache)
    return {}


def _cache_key_for_score(
    *,
    task_text: str,
    frontier_record: Mapping[str, Any],
    agent_xyz: Optional[np.ndarray],
    heading_xz: Optional[np.ndarray],
    view_index: Optional[int],
    cfg: TffsConfig,
    context: Optional[Mapping[str, Any]],
) -> str:
    point_q = _quantize_xyz(frontier_record.get("point_xyz"), float(cfg.cache_position_quantization_m))
    agent_q = _quantize_xyz(agent_xyz, float(cfg.cache_position_quantization_m))
    heading_deg = _heading_deg_from_xz(heading_xz)
    heading_q = _quantize_scalar(heading_deg, float(cfg.cache_heading_quantization_deg))
    task_key = re.sub(r"\s+", " ", str(task_text or "").strip().lower())
    payload: Dict[str, Any] = {
        "prompt_version": str(cfg.prompt_version),
        "task_text": task_key,
        "frontier_xyz_q": point_q,
        "agent_xyz_q": agent_q,
        "agent_heading_deg_q": heading_q,
        "view_index": None if view_index is None else int(view_index),
    }
    if isinstance(context, Mapping):
        for key in ("task_id", "scene_name"):
            if key in context:
                payload[key] = _jsonable(context[key])
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))


def _score_info_from_cache(cache: Mapping[str, Any], key: str) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    if key not in cache:
        return None, None
    entry = cache.get(key)
    if isinstance(entry, Mapping):
        value = entry.get("hypothesis_score", entry.get("score"))
        confidence = _clamp01(entry.get("confidence", 0.0))
        parsed = entry.get("parsed")
    else:
        value = entry
        confidence = 0.0
        parsed = None
    try:
        score = float(value)
    except Exception:
        return None, None
    if not math.isfinite(score):
        return None, None
    info = {
        "ok": True,
        "module": MODULE_NAME,
        "prompt_version": PROMPT_VERSION,
        "hypothesis_score": _clamp01(score),
        "confidence": float(confidence),
        "parsed": _jsonable(parsed),
        "source": "cache",
        "cache_key": key,
        "parse_ok": True,
        "parse_attempts": 0,
        "error_type": "",
        "error_message": "",
        "errors": [],
    }
    return float(_clamp01(score)), info


def _store_score_cache(cache: MutableMapping[str, Any], key: str, score_info: Mapping[str, Any]) -> None:
    score = score_info.get("hypothesis_score")
    try:
        score_float = float(score)
    except Exception:
        return
    if not math.isfinite(score_float):
        return
    cache[key] = {
        "hypothesis_score": float(score_float),
        "confidence": _clamp01(score_info.get("confidence", 0.0)),
        "parsed": _jsonable(score_info.get("parsed")),
        "prompt_version": str(score_info.get("prompt_version", PROMPT_VERSION)),
    }


def build_task_facing_prompt(
    task_text: str,
    *,
    frontier_record: Optional[Mapping[str, Any]] = None,
    view_index: Optional[int] = None,
) -> str:
    frontier_bits: List[str] = []
    if frontier_record:
        if frontier_record.get("frontier_index") is not None:
            frontier_bits.append(f"frontier_index={int(frontier_record['frontier_index'])}")
        if frontier_record.get("point_xyz") is not None:
            frontier_bits.append(f"frontier_xyz={frontier_record['point_xyz']}")
        if frontier_record.get("relative_bearing_rad") is not None:
            frontier_bits.append(f"relative_bearing_rad={float(frontier_record['relative_bearing_rad']):.4f}")
    if view_index is not None:
        frontier_bits.append(f"current_panorama_view_index={int(view_index)}")
    frontier_context = ", ".join(frontier_bits) if frontier_bits else "not provided"
    return (
        "You evaluate one current robot panorama view for test-time frontier reranking.\n"
        "The image is already observed at the current step. It is not a rendered future view.\n"
        "Navigation task: "
        f"{str(task_text).strip()}\n"
        f"Frontier/view metadata: {frontier_context}\n\n"
        "Judge whether this direction is promising for continuing exploration toward the task.\n"
        "Do not require the target object to already be visible. Use room type, relevant furniture, "
        "landmarks, anchor objects, traversable direction, and clearly irrelevant or low-quality cues.\n\n"
        "Return strict JSON only with this exact schema:\n"
        "{"
        "\"target_context\": number between 0 and 1, "
        "\"anchor_context\": number between 0 and 1, "
        "\"room_context\": number between 0 and 1, "
        "\"negative\": number between 0 and 1, "
        "\"confidence\": number between 0 and 1, "
        "\"reason\": \"one short sentence\""
        "}"
    )


def _hypothesis_from_scores(parsed: Mapping[str, Any], cfg: TffsConfig) -> float:
    score = (
        float(cfg.target_context_weight) * _clamp01(parsed.get("target_context"))
        + float(cfg.anchor_context_weight) * _clamp01(parsed.get("anchor_context"))
        + float(cfg.room_context_weight) * _clamp01(parsed.get("room_context"))
        - float(cfg.negative_weight) * _clamp01(parsed.get("negative"))
    )
    return _clamp01(score)


def score_task_facing_view(
    *,
    task_text: str,
    view: Any,
    cfg: Optional[TffsConfig] = None,
    frontier_record: Optional[Mapping[str, Any]] = None,
    view_index: Optional[int] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    cfg = cfg or TffsConfig()
    view_rec = _coerce_view_record(view, default_index=0 if view_index is None else int(view_index))
    image_path = view_rec.get("image_path")
    prompt = build_task_facing_prompt(task_text, frontier_record=frontier_record, view_index=view_index)
    info: Dict[str, Any] = {
        "ok": False,
        "module": MODULE_NAME,
        "prompt_version": str(cfg.prompt_version),
        "view_index": view_index,
        "panorama_path": image_path,
        "hypothesis_score": None,
        "confidence": 0.0,
        "raw": "",
        "parsed": None,
        "source": "vlm",
        "parse_ok": False,
        "parse_attempts": 0,
        "error_type": "",
        "error_message": "",
        "errors": [],
    }
    if not bool(cfg.use_vlm):
        info.update({"source": "vlm_disabled", "error_type": "VLMDisabled", "error_message": "TFFS VLM disabled"})
        return None, info
    if _vlm_chat is None:
        info.update({"source": "vlm_import_failed", "error_type": "VLMImportError", "error_message": _VLM_IMPORT_ERROR})
        return None, info
    if not image_path:
        info.update({"source": "missing_image_path", "error_type": "MissingImagePath", "error_message": "view has no image path"})
        return None, info
    if bool(cfg.require_image_file_exists) and not Path(str(image_path)).exists():
        info.update(
            {
                "source": "missing_image_file",
                "error_type": "MissingImageFile",
                "error_message": f"image path does not exist: {image_path}",
            }
        )
        return None, info

    errors: List[Dict[str, str]] = []
    last_raw = ""
    for attempt in range(max(1, int(cfg.vlm_max_retries))):
        try:
            with no_proxy_env(bool(cfg.vlm_no_proxy)):
                raw = _vlm_chat(
                    text=prompt,
                    image_path=str(image_path),
                    model=str(cfg.vlm_model),
                    max_tokens=int(cfg.vlm_max_tokens),
                )
            last_raw = str(raw)
            parsed = parse_json_object(last_raw)
            hypothesis = _hypothesis_from_scores(parsed, cfg)
            confidence = _clamp01(parsed.get("confidence"))
            info.update(
                {
                    "ok": True,
                    "hypothesis_score": float(hypothesis),
                    "confidence": float(confidence),
                    "raw": last_raw,
                    "parsed": {
                        "target_context": _clamp01(parsed.get("target_context")),
                        "anchor_context": _clamp01(parsed.get("anchor_context")),
                        "room_context": _clamp01(parsed.get("room_context")),
                        "negative": _clamp01(parsed.get("negative")),
                        "confidence": float(confidence),
                        "reason": str(parsed.get("reason", ""))[:240],
                    },
                    "source": "vlm",
                    "parse_ok": True,
                    "parse_attempts": int(attempt + 1),
                    "error_type": "",
                    "error_message": "",
                    "errors": errors,
                }
            )
            return float(hypothesis), info
        except Exception as exc:
            errors.append({"error_type": type(exc).__name__, "error_message": str(exc)})
            if attempt + 1 < max(1, int(cfg.vlm_max_retries)):
                time.sleep(max(0.0, float(cfg.vlm_retry_sleep_sec)))

    info.update(
        {
            "raw": last_raw,
            "source": "vlm_error",
            "parse_ok": False,
            "parse_attempts": int(len(errors)),
            "error_type": errors[-1]["error_type"] if errors else "",
            "error_message": errors[-1]["error_message"] if errors else "",
            "errors": errors,
        }
    )
    return None, info


def _fallback_info_update(
    info: Dict[str, Any],
    *,
    selected_index: int,
    gate_reason: str,
    latency_start: float,
) -> Tuple[int, Dict[str, Any]]:
    paths = info.get("panorama_paths", [])
    views = info.get("view_index", [])
    selected_path = paths[selected_index] if 0 <= selected_index < len(paths) else None
    selected_view = views[selected_index] if 0 <= selected_index < len(views) else None
    info.update(
        {
            "tffs_applied": False,
            "selected_frontier_index": int(selected_index),
            "selected_view_index": selected_view,
            "selected_panorama_path": selected_path,
            "panorama_path": selected_path,
            "gate_reason": str(gate_reason),
            "latency_ms": float((time.time() - latency_start) * 1000.0),
        }
    )
    return int(selected_index), _jsonable(info)


def run_tffs_rerank(
    *,
    frontier_candidates: Sequence[Any],
    task_text: str = "",
    sentence: str = "",
    description: str = "",
    frontier_logits: Optional[Sequence[float]] = None,
    panorama_views: Optional[Sequence[Any]] = None,
    agent_pose: Any = None,
    branch_is_frontier: bool = True,
    baseline_frontier_index: Optional[int] = None,
    cfg: Optional[TffsConfig] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    cfg = cfg or TffsConfig()
    start = time.time()
    interval_allowed, interval_info = _vlm_interval_gate(context, cfg)
    score_cache = _score_cache_from_context(context) if bool(cfg.enable_score_cache) else {}
    resolved_task = str(task_text or sentence or description or "").strip()
    records = _coerce_frontier_records(frontier_candidates)
    n = int(len(records))
    vlm_budget = _vlm_call_budget(cfg, n)
    logits, logit_info = _coerce_logits(frontier_logits, n)
    norm_logits = _normalize_logits(logits)
    baseline_idx, baseline_info = _choose_baseline_index(
        baseline_frontier_index=baseline_frontier_index,
        logits=logits,
        n=n,
    )
    baseline_logit = float(logits[baseline_idx]) if 0 <= baseline_idx < n else None
    views = _coerce_panorama_views(panorama_views)
    agent_xyz, heading_xz, pose_info = _coerce_agent_pose(agent_pose)
    hypothesis_scores: List[Optional[float]] = [None for _ in range(n)]
    confidences = np.zeros((n,), dtype=float)
    fused = np.array(norm_logits, dtype=float)
    vlm_scores: List[Dict[str, Any]] = []
    view_indices: List[Optional[int]] = [None for _ in range(n)]
    panorama_paths: List[Optional[str]] = [None for _ in range(n)]

    info: Dict[str, Any] = {
        "ok": True,
        "module": MODULE_NAME,
        "prompt_version": str(cfg.prompt_version),
        "tffs_called": True,
        "tffs_applied": False,
        "branch_is_frontier": bool(branch_is_frontier),
        "baseline_frontier_index": int(baseline_idx),
        "selected_frontier_index": int(baseline_idx),
        "rerank_frontier_index": int(baseline_idx),
        "frontier_records": records,
        "frontier_count": int(n),
        "frontier_logits": logits.tolist(),
        "normalized_logits": norm_logits.tolist(),
        "baseline_logit": baseline_logit,
        "view_index": view_indices,
        "panorama_path": None,
        "panorama_paths": panorama_paths,
        "vlm_scores": vlm_scores,
        "vlm_call_interval": int(interval_info["vlm_call_interval"]),
        "vlm_interval_allowed": bool(interval_info["vlm_interval_allowed"]),
        "vlm_interval_counter": interval_info.get("vlm_interval_counter"),
        "vlm_interval_counter_source": str(interval_info.get("vlm_interval_counter_source", "")),
        "vlm_call_budget": int(vlm_budget),
        "vlm_candidate_indices": [],
        "vlm_call_count": 0,
        "cache_enabled": bool(cfg.enable_score_cache),
        "cache_hits": 0,
        "cache_misses": 0,
        "cost_control_reason": str(interval_info["cost_control_reason"]),
        "hypothesis_score": hypothesis_scores,
        "fused_score": fused.tolist(),
        "score_margin": 0.0,
        "confidence": 0.0,
        "gate_reason": "",
        "non_oracle_inputs": {
            "uses_only": list(cfg.non_oracle_inputs),
            "uses_current_panorama_only": True,
            "generates_new_frontier": False,
            "uses_future_or_rendered_frontier_view": False,
            "changes_object_frontier_gate": False,
            "changes_low_level_follower": False,
        },
        "latency_ms": 0.0,
        "task_text": resolved_task,
        "vlm_enabled": bool(cfg.use_vlm),
        "vlm_available": bool(_vlm_chat is not None),
        "vlm_import_error": _VLM_IMPORT_ERROR,
        "panorama_view_count": int(len(views)),
        "agent_pose": pose_info,
        "config": {
            "alpha_logit": float(cfg.alpha_logit),
            "beta_hypothesis": float(cfg.beta_hypothesis),
            "min_score_margin": float(cfg.min_score_margin),
            "min_confidence": float(cfg.min_confidence),
            "max_vlm_candidates": int(cfg.max_vlm_candidates),
            "vlm_call_interval": int(cfg.vlm_call_interval),
            "max_vlm_calls_per_decision": int(cfg.max_vlm_calls_per_decision),
            "top_vlm_logit_candidates": int(cfg.top_vlm_logit_candidates),
            "min_logit_gap_for_vlm": float(cfg.min_logit_gap_for_vlm),
            "enable_score_cache": bool(cfg.enable_score_cache),
            "cache_position_quantization_m": float(cfg.cache_position_quantization_m),
            "cache_heading_quantization_deg": float(cfg.cache_heading_quantization_deg),
        },
    }
    info.update(logit_info)
    info.update(baseline_info)
    if context:
        info.update({str(k): _jsonable(v) for k, v in context.items()})
    info.update(
        {
            "vlm_call_interval": int(interval_info["vlm_call_interval"]),
            "vlm_interval_allowed": bool(interval_info["vlm_interval_allowed"]),
            "vlm_interval_counter": interval_info.get("vlm_interval_counter"),
            "vlm_interval_counter_source": str(interval_info.get("vlm_interval_counter_source", "")),
            "vlm_call_budget": int(vlm_budget),
            "vlm_candidate_indices": [],
            "vlm_call_count": 0,
            "cache_enabled": bool(cfg.enable_score_cache),
            "cache_hits": 0,
            "cache_misses": 0,
            "cost_control_reason": str(interval_info["cost_control_reason"]),
        }
    )

    if n <= 0:
        return _fallback_info_update(info, selected_index=-1, gate_reason="no_frontier_candidates", latency_start=start)
    if not bool(branch_is_frontier):
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="not_frontier_branch_keep_baseline",
            latency_start=start,
        )
    if n < int(cfg.min_candidates):
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="insufficient_frontier_candidates_keep_baseline",
            latency_start=start,
        )
    if not _task_text_available(resolved_task, cfg):
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="task_text_unavailable_keep_baseline",
            latency_start=start,
        )
    if not bool(interval_allowed):
        info.update(
            {
                "rerank_frontier_index": int(baseline_idx),
                "fused_score": fused.tolist(),
                "score_margin": 0.0,
                "confidence": 0.0,
                "cost_control_reason": "vlm_interval_skip_keep_baseline",
            }
        )
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="vlm_interval_skip_keep_baseline",
            latency_start=start,
        )
    if int(vlm_budget) <= 0:
        info.update(
            {
                "rerank_frontier_index": int(baseline_idx),
                "fused_score": fused.tolist(),
                "score_margin": 0.0,
                "confidence": 0.0,
                "cost_control_reason": "vlm_budget_zero_keep_baseline",
            }
        )
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="vlm_budget_zero_keep_baseline",
            latency_start=start,
        )

    if not bool(cfg.use_vlm):
        proposed = int(np.argmax(fused)) if fused.size else int(baseline_idx)
        margin = float(fused[proposed] - fused[baseline_idx]) if 0 <= baseline_idx < n else 0.0
        info.update(
            {
                "rerank_frontier_index": int(proposed),
                "fused_score": fused.tolist(),
                "score_margin": margin,
                "confidence": 0.0,
            }
        )
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="vlm_disabled_logits_only_keep_baseline",
            latency_start=start,
        )
    if _vlm_chat is None:
        proposed = int(np.argmax(fused)) if fused.size else int(baseline_idx)
        margin = float(fused[proposed] - fused[baseline_idx]) if 0 <= baseline_idx < n else 0.0
        info.update(
            {
                "rerank_frontier_index": int(proposed),
                "fused_score": fused.tolist(),
                "score_margin": margin,
                "confidence": 0.0,
            }
        )
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="vlm_unavailable_logits_only_keep_baseline",
            latency_start=start,
        )
    if len(views) <= 0:
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason="panorama_unavailable_keep_baseline",
            latency_start=start,
        )
    if not bool(pose_info.get("pose_ok", False)) or agent_xyz is None or heading_xz is None:
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason=f"agent_pose_unavailable_keep_baseline:{pose_info.get('pose_reason', '')}",
            latency_start=start,
        )

    for rec in records:
        point = rec.get("point_xyz")
        if point is None:
            rec.update({"bearing_ok": False, "bearing_reason": "invalid_frontier_point"})
            continue
        assign = _frontier_view_assignment(
            frontier_xyz=np.asarray(point, dtype=float),
            agent_xyz=agent_xyz,
            heading_xz=heading_xz,
            n_views=len(views),
            cfg=cfg,
        )
        rec.update(assign)
        idx = rec.get("view_index")
        if idx is not None and 0 <= int(idx) < len(views):
            view_rec = views[int(idx)]
            rec["panorama_path"] = view_rec.get("image_path")
            rec["panorama_view_record"] = _jsonable(view_rec)
            view_indices[int(rec["frontier_index"])] = int(idx)
            panorama_paths[int(rec["frontier_index"])] = view_rec.get("image_path")

    info["view_index"] = view_indices
    info["panorama_paths"] = panorama_paths
    info["frontier_records"] = records

    score_indices = _candidate_indices_for_vlm(norm_logits, baseline_idx, cfg)
    info["vlm_candidate_indices"] = [int(i) for i in score_indices]
    info["cost_control_reason"] = "vlm_interval_allowed_candidate_subset"
    vlm_call_count = 0
    cache_hits = 0
    cache_misses = 0
    for idx in score_indices:
        if idx < 0 or idx >= n:
            continue
        rec = records[int(idx)]
        view_idx = rec.get("view_index")
        if view_idx is None or not bool(rec.get("bearing_ok", False)):
            score_info = {
                "ok": False,
                "frontier_index": int(idx),
                "view_index": view_idx,
                "panorama_path": rec.get("panorama_path"),
                "hypothesis_score": None,
                "confidence": 0.0,
                "source": "bearing_unavailable",
                "error_type": "BearingUnavailable",
                "error_message": str(rec.get("bearing_reason", "")),
            }
            vlm_scores.append(score_info)
            continue
        view_rec = views[int(view_idx)]
        cache_key = ""
        hypothesis: Optional[float]
        score_info: Dict[str, Any]
        if bool(cfg.enable_score_cache):
            cache_key = _cache_key_for_score(
                task_text=resolved_task,
                frontier_record=rec,
                agent_xyz=agent_xyz,
                heading_xz=heading_xz,
                view_index=int(view_idx),
                cfg=cfg,
                context=context,
            )
            hypothesis, cached_info = _score_info_from_cache(score_cache, cache_key)
            if cached_info is not None:
                cache_hits += 1
                score_info = cached_info
                score_info.update(
                    {
                        "view_index": int(view_idx),
                        "panorama_path": rec.get("panorama_path"),
                    }
                )
            else:
                cache_misses += 1
                hypothesis, score_info = score_task_facing_view(
                    task_text=resolved_task,
                    view=view_rec,
                    cfg=cfg,
                    frontier_record=rec,
                    view_index=int(view_idx),
                )
                if score_info.get("source") in ("vlm", "vlm_error"):
                    vlm_call_count += 1
                score_info["cache_key"] = cache_key
                if hypothesis is not None and bool(score_info.get("ok", False)):
                    _store_score_cache(score_cache, cache_key, score_info)
        else:
            hypothesis, score_info = score_task_facing_view(
                task_text=resolved_task,
                view=view_rec,
                cfg=cfg,
                frontier_record=rec,
                view_index=int(view_idx),
            )
            if score_info.get("source") in ("vlm", "vlm_error"):
                vlm_call_count += 1
        score_info["frontier_index"] = int(idx)
        vlm_scores.append(score_info)
        if hypothesis is None:
            continue
        hypothesis_scores[int(idx)] = float(hypothesis)
        confidences[int(idx)] = float(score_info.get("confidence", 0.0))
        rec["hypothesis_score"] = float(hypothesis)
        rec["vlm_confidence"] = float(confidences[int(idx)])
        rec["vlm_ok"] = bool(score_info.get("ok", False))
        rec["vlm_reason"] = (
            ((score_info.get("parsed") or {}) if isinstance(score_info.get("parsed"), Mapping) else {}).get("reason", "")
        )
    info["vlm_call_count"] = int(vlm_call_count)
    info["cache_hits"] = int(cache_hits)
    info["cache_misses"] = int(cache_misses)
    if bool(cfg.enable_score_cache):
        info["score_cache_size"] = int(len(score_cache))
        info["score_cache"] = _jsonable(score_cache)

    h_arr = np.zeros((n,), dtype=float)
    h_mask = np.zeros((n,), dtype=bool)
    for idx, value in enumerate(hypothesis_scores):
        if value is None:
            continue
        h_arr[int(idx)] = float(value)
        h_mask[int(idx)] = True
    fused = np.array(norm_logits, dtype=float)
    if np.any(h_mask):
        fused[h_mask] = float(cfg.alpha_logit) * norm_logits[h_mask] + float(cfg.beta_hypothesis) * h_arr[h_mask]
    for idx, rec in enumerate(records):
        rec["frontier_logit"] = float(logits[int(idx)])
        rec["normalized_logit"] = float(norm_logits[int(idx)])
        rec["hypothesis_score"] = hypothesis_scores[int(idx)]
        rec["vlm_confidence"] = float(confidences[int(idx)])
        rec["fused_score"] = float(fused[int(idx)])
        rec["is_baseline_frontier"] = bool(int(idx) == int(baseline_idx))

    proposed_idx = int(np.argmax(fused)) if fused.size else int(baseline_idx)
    margin = float(fused[proposed_idx] - fused[baseline_idx]) if 0 <= baseline_idx < n else 0.0
    proposed_confidence = float(confidences[proposed_idx]) if 0 <= proposed_idx < n else 0.0
    selected_path = panorama_paths[proposed_idx] if 0 <= proposed_idx < len(panorama_paths) else None
    selected_view = view_indices[proposed_idx] if 0 <= proposed_idx < len(view_indices) else None
    info.update(
        {
            "frontier_records": records,
            "vlm_scores": vlm_scores,
            "hypothesis_score": hypothesis_scores,
            "fused_score": fused.tolist(),
            "rerank_frontier_index": int(proposed_idx),
            "score_margin": float(margin),
            "confidence": float(proposed_confidence),
            "view_index": view_indices,
            "proposed_view_index": selected_view,
            "proposed_panorama_path": selected_path,
            "panorama_path": selected_path,
            "panorama_paths": panorama_paths,
        }
    )

    fail_reasons: List[str] = []
    if proposed_idx == baseline_idx:
        fail_reasons.append("selected_matches_baseline")
    if not bool(h_mask[proposed_idx]):
        fail_reasons.append("selected_missing_vlm_score")
    if margin < float(cfg.min_score_margin):
        fail_reasons.append("score_margin_below_threshold")
    if proposed_confidence < float(cfg.min_confidence):
        fail_reasons.append("confidence_below_threshold")
    if not any(bool(score.get("ok", False)) for score in vlm_scores):
        fail_reasons.append("vlm_scores_unavailable")

    if fail_reasons:
        return _fallback_info_update(
            info,
            selected_index=baseline_idx,
            gate_reason=",".join(fail_reasons) + "_keep_baseline",
            latency_start=start,
        )

    info.update(
        {
            "tffs_applied": True,
            "selected_frontier_index": int(proposed_idx),
            "selected_view_index": selected_view,
            "selected_panorama_path": selected_path,
            "gate_reason": "tffs_frontier_rerank_applied",
            "latency_ms": float((time.time() - start) * 1000.0),
        }
    )
    return int(proposed_idx), _jsonable(info)


__all__ = [
    "MODULE_NAME",
    "PROMPT_VERSION",
    "TffsConfig",
    "TffsDecisionError",
    "TffsInputError",
    "build_task_facing_prompt",
    "parse_json_object",
    "run_tffs_rerank",
    "score_task_facing_view",
]
