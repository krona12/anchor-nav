from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from vlm.client import DEFAULT_MODEL as VLM_DEFAULT_MODEL, chat_messages


@dataclass
class ValidateConfig:
    enabled: bool = True
    vlm_model: str = VLM_DEFAULT_MODEL
    num_views: int = 12
    group_size: int = 4
    max_tokens: int = 128
    turn_action: str = "turn_left"
    max_distance_cm: float = 50.0


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _grid_2x2(images: List[np.ndarray]) -> np.ndarray:
    if len(images) != 4:
        raise ValueError(f"_grid_2x2 expects 4 images, got {len(images)}")
    h = min(int(x.shape[0]) for x in images)
    w = min(int(x.shape[1]) for x in images)
    resized = [cv2.resize(np.ascontiguousarray(x[:, :, :3], dtype=np.uint8), (w, h)) for x in images]
    top = np.concatenate([resized[0], resized[1]], axis=1)
    bottom = np.concatenate([resized[2], resized[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _image_to_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    import base64

    b64 = base64.b64encode(data).decode()
    suffix = image_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/jpeg")
    return f"data:{mime};base64,{b64}"


def _contains_description_from_vlm(image_paths: List[Path], description: str, cfg: ValidateConfig) -> Dict[str, Any]:
    prompt = (
        "You are given 3 collage images. Each collage is a 2x2 view captured after arrival.\n"
        f"Original navigation description: {description}\n"
        f"Decide whether ANY image clearly contains the described target object instance, AND it is near enough "
        f"(estimated distance <= {float(cfg.max_distance_cm):.1f} cm).\n"
        "Return strict JSON only: "
        "{\"contains_target\": true/false, \"near_enough\": true/false, "
        "\"estimated_distance_cm\": number, \"confidence\": 0-1, \"reason\": \"...\"}."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
    raw = chat_messages(
        messages=[{"role": "user", "content": content}],
        model=cfg.vlm_model,
        max_tokens=int(cfg.max_tokens),
        temperature=0.0,
        timeout=60,
    )
    parsed = _parse_json(raw)
    return {
        "contains_target": bool(parsed.get("contains_target", False)),
        "near_enough": bool(parsed.get("near_enough", False)),
        "estimated_distance_cm": float(parsed.get("estimated_distance_cm", 999.0) or 999.0),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": str(parsed.get("reason", "")),
        "raw": raw,
    }


def validate_after_arrival(
    *,
    sim: Any,
    description: str,
    cfg: ValidateConfig,
    io_dir: Optional[Path] = None,
    jsonl_log_path: str = "",
) -> Dict[str, Any]:
    """
    到达目标后执行验证：
    1) 环视 num_views 次并采样 RGB；
    2) 每 4 张做一张 2x2 合图（12 视角默认得到 3 张）；
    3) 逐张请求 VLM 判断是否包含目标物体，任意一张命中即视为通过。
    """
    t0 = time.perf_counter()
    info: Dict[str, Any] = {
        "validate_enabled": bool(cfg.enabled),
        "num_views": int(cfg.num_views),
        "group_size": int(cfg.group_size),
        "contains_target": False,
        "near_enough": False,
        "num_steps": 0,
        "groups": [],
        "errors": [],
        "artifact_dir": None,
        "elapsed_ms": None,
    }
    if not cfg.enabled:
        info["skipped"] = "disabled"
        return info
    if not description:
        info["skipped"] = "empty_description"
        return info
    if int(cfg.num_views) <= 0:
        info["skipped"] = "invalid_num_views"
        return info
    if int(cfg.group_size) <= 0:
        info["skipped"] = "invalid_group_size"
        return info

    call_dir: Optional[Path] = None
    if io_dir is not None:
        call_dir = io_dir / f"validate_call_{int(time.time() * 1000)}"
        call_dir.mkdir(parents=True, exist_ok=True)
        info["artifact_dir"] = str(call_dir)

    views: List[np.ndarray] = []
    for i in range(int(cfg.num_views)):
        obs = sim.step(action=cfg.turn_action)
        rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8)
        views.append(rgb)
        info["num_steps"] += 1
        if call_dir is not None:
            cv2.imwrite(
                str(call_dir / f"view_{i:02d}.jpg"),
                cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
            )

    groups: List[List[np.ndarray]] = []
    gs = int(cfg.group_size)
    for i in range(0, len(views), gs):
        chunk = views[i : i + gs]
        if len(chunk) == gs:
            groups.append(chunk)

    collage_paths: List[Path] = []
    for gi, g in enumerate(groups):
        rec: Dict[str, Any] = {"group_index": gi}
        try:
            grid = _grid_2x2(g)
            if call_dir is not None:
                collage_path = call_dir / f"collage_{gi:02d}.jpg"
            else:
                collage_path = Path(f"/tmp/validate_collage_{int(time.time() * 1000)}_{gi}.jpg")
            cv2.imwrite(
                str(collage_path),
                cv2.cvtColor(np.ascontiguousarray(grid), cv2.COLOR_RGB2BGR),
            )
            collage_paths.append(collage_path)
            rec["collage_path"] = str(collage_path)
        except Exception as e:
            rec["error"] = repr(e)
            info["errors"].append(repr(e))
        info["groups"].append(rec)

    if len(collage_paths) > 0:
        try:
            vlm = _contains_description_from_vlm(collage_paths, description=description, cfg=cfg)
            info["near_enough"] = bool(vlm.get("near_enough", False))
            info["contains_target"] = bool(vlm["contains_target"]) and bool(info["near_enough"])
            info["vlm_result"] = vlm
        except Exception as e:
            info["errors"].append(repr(e))

    info["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    _append_jsonl(
        jsonl_log_path,
        {
            "event": "validate_after_arrival",
            "description": description,
            "contains_target": info["contains_target"],
            "near_enough": info["near_enough"],
            "num_steps": info["num_steps"],
            "elapsed_ms": info["elapsed_ms"],
            "groups": info["groups"],
            "vlm_result": info.get("vlm_result"),
            "artifact_dir": info["artifact_dir"],
            "errors": info["errors"],
        },
    )
    return info
