"""
物体目标 VLM 重排序：在 baseline 选择「走向某记忆物体」时，用 top-K 候选各自
「首次检测帧」完整 RGB + 原始描述，请求 VLM 输出最符合描述的下标，并替换导航目标。

默认仅对 instance 级别任务开启（由 RerankConfig.enabled_levels 控制）。

日志与落盘（便于核对 VLM 输入输出）：
- 设置环境变量 ``RERANK_LOG_JSONL=/path/to/rerank_vlm.jsonl`` 时，每次调用追加一行 JSON，
  含完整 ``description``、``user_prompt``、``system_prompt``、``raw_response``、``parsed`` 等。
- 同时会在 ``RERANK_LOG_JSONL`` 同目录下创建 ``rerank_vlm_io/call_<ms>_<pid>/``，
  写入 ``description.txt``、``user_prompt.txt``、``input_rank*_mem*.jpg``、``vlm_response_raw.txt``、``vlm_meta.json``。
- 若需自定义目录，可设 ``RERANK_IO_DIR=/path/to/dir``（每次调用仍建子目录 ``call_*``）。
- 设 ``RERANK_SKIP_IO_ARTIFACTS=1``（或 ``true``/``yes``）时不再创建上述 ``call_*`` 目录、不存图片与逐调用文本文件；
  ``RERANK_LOG_JSONL`` 仍可追加 JSON 行。设为 ``0``/``false`` 则恢复落盘。
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from vlm.client import BASE_URL as NEW_VLM_BASE_URL, DEFAULT_MODEL as NEW_DEFAULT_MODEL, chat_messages


SYSTEM_PROMPT = (
    "You compare candidate first-sighting RGB views against ONE embodied navigation description. "
    "You must always output exactly ONE chosen index (1..K). "
    "If some view clearly shows the MAIN target object described, pick that index. "
    "If NO view clearly shows the target, you must still pick the SINGLE most plausible index: "
    "infer likely room type, layout, and contextual objects (furniture, fixtures, decor) that best align with the text; "
    "use common sense about where that object category usually appears and what co-occurs with it in the description. "
    "Prefer the view that is most likely to contain the target or to be on the way to it. "
    "Explain briefly in 'reason' whether the match is direct or heuristic (room/context inference). "
    "Return strict JSON only."
)

USER_TEMPLATE = """Navigation goal (full description):
{description}

Below are {k} images in order. Image 1 is the first-sighting view for candidate 1, Image 2 for candidate 2, and so on.
Each full-frame RGB was captured when that memory object was first detected (it may or may not be the main target in the text).

Instructions:
1) Prefer the candidate where the MAIN target in the description is clearly visible or unambiguous.
2) If you cannot find a clear match, choose the candidate that is MOST LIKELY related to the goal anyway: infer likely room
   (e.g. bedroom vs kitchen), scene layout, and other visible objects that support the description; use contextual and
   commonsense cues (e.g. bed + nightstand, desk + chair, bathroom fixtures).
3) You must output exactly one index from 1 to {k}; do not refuse to choose.

Return JSON with EXACT keys:
{{"best_index": <integer from 1 to {k}>, "reason": "<English: state if match is direct or heuristic; mention room/context cues if no clear object match>"}}
"""

CONFIRM_SYSTEM_PROMPT = (
    "You verify whether current egocentric panoramic images already show the target object requested by ONE navigation description. "
    "Return strict JSON only."
)

CONFIRM_USER_TEMPLATE = """Navigation goal (full description):
{description}

Below are {k} panoramic downsampled views captured at the agent's current position.
Decide whether the target object is visible now in these views.

Return JSON with EXACT keys:
{{"has_target": <true_or_false>, "reason": "<English brief reason>"}}
"""


@dataclass
class RerankConfig:
    """可配置：在哪些 task_level 上启用重排序。"""

    enabled_levels: Set[str] = field(default_factory=lambda: {"instance"})
    top_k: int = 8
    min_candidates_with_rgb: int = 2
    model: str = NEW_DEFAULT_MODEL
    timeout: int = 120


def parse_levels_csv(s: str) -> Set[str]:
    """逗号分隔，如 'instance' 或 'region,instance'。"""
    parts = {p.strip() for p in (s or "").split(",") if p.strip()}
    return parts if parts else {"instance"}


def numpy_rgb_to_jpeg_data_url(rgb: np.ndarray) -> str:
    """RGB uint8 HxWx3 -> data:image/jpeg;base64,..."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _artifacts_root_dir() -> Optional[Path]:
    """落盘目录：优先环境变量 RERANK_IO_DIR；否则当设置了 RERANK_LOG_JSONL 时用其同目录下 rerank_vlm_io/。"""
    skip = os.environ.get("RERANK_SKIP_IO_ARTIFACTS", "").strip().lower()
    if skip in ("1", "true", "yes", "on"):
        return None
    env = os.environ.get("RERANK_IO_DIR", "").strip()
    if env:
        return Path(env)
    j = os.environ.get("RERANK_LOG_JSONL", "").strip()
    if j:
        return Path(j).resolve().parent / "rerank_vlm_io"
    return None


def _save_input_images_jpeg(call_dir: Path, cand: List[int], rgb_list: List[Any]) -> List[str]:
    """保存送入 VLM 的每张首检 RGB，返回相对文件名列表。"""
    rel_names: List[str] = []
    for rank, mid in enumerate(cand, start=1):
        if mid >= len(rgb_list) or rgb_list[mid] is None:
            continue
        img = rgb_list[mid]
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            continue
        name = f"input_rank{rank:02d}_mem{mid:03d}.jpg"
        path = call_dir / name
        bgr = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rel_names.append(name)
    return rel_names


def _parse_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _top_memory_indices_with_rgb(
    rep: Any,
    *,
    top_k: int,
) -> List[int]:
    """按 object_score 降序，取最多 top_k 个且 object_first_rgb[i] 非空的记忆下标。"""
    n = int(np.asarray(rep.object_score).reshape(-1).shape[0])
    if n == 0:
        return []
    order = np.argsort(-np.asarray(rep.object_score, dtype=float).reshape(-1))
    out: List[int] = []
    rgb_list = getattr(rep, "object_first_rgb", None) or []
    for i in order.tolist():
        if len(out) >= top_k:
            break
        if i >= len(rgb_list) or rgb_list[i] is None:
            continue
        img = rgb_list[i]
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] < 3:
            continue
        out.append(int(i))
    return out


def list_rerank_candidate_memory_ids(rep: Any, *, top_k: int = 8) -> List[int]:
    """与一次 `rerank_object_target` 所选候选顺序一致（调试用图保存）。"""
    return _top_memory_indices_with_rgb(rep, top_k=top_k)


def rerank_object_target(
    *,
    description: str,
    rep: Any,
    baseline_target_xyz: np.ndarray,
    decision_aux: Dict[str, Any],
    cfg: RerankConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    若本轮为物体决策且候选足够，则调用 VLM 重选记忆槽；否则返回 baseline 目标。

    Returns:
        target_xyz: shape (3,), 已与 decision() 一致地做 y/z 交换后的 Habitat 坐标
        info: 含 skipped / vlm / chosen_memory_index 等
    """
    info: Dict[str, Any] = {"rerank_applied": False}
    if not decision_aux.get("is_object_decision"):
        raise RuntimeError("rerank_object_target called on non-object decision (strict mode)")

    cand = _top_memory_indices_with_rgb(rep, top_k=cfg.top_k)
    if len(cand) < cfg.min_candidates_with_rgb:
        raise RuntimeError(
            f"insufficient_rgb_candidates: got {len(cand)}, require >= {cfg.min_candidates_with_rgb} (strict mode)"
        )

    jsonl_path = os.environ.get("RERANK_LOG_JSONL", "").strip()
    art_root = _artifacts_root_dir()
    call_dir: Optional[Path] = None
    if art_root is not None:
        call_dir = art_root / f"call_{int(time.time() * 1000)}_{os.getpid()}"
        try:
            call_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            call_dir = None

    urls: List[str] = []
    rgb_list = list(getattr(rep, "object_first_rgb", None) or [])
    for mid in cand:
        urls.append(numpy_rgb_to_jpeg_data_url(rgb_list[mid]))

    prompt = USER_TEMPLATE.format(description=description.strip(), k=len(cand))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    content: List[Dict[str, Any]] = []
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    input_image_files: List[str] = []
    if call_dir is not None:
        try:
            (call_dir / "description.txt").write_text(description, encoding="utf-8")
            (call_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
            (call_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
            (call_dir / "candidate_memory_ids.json").write_text(
                json.dumps({"candidate_memory_ids_in_order": cand}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            input_image_files = _save_input_images_jpeg(call_dir, cand, rgb_list)
        except OSError:
            pass

    t0 = time.perf_counter()
    raw: Optional[str] = None
    try:
        raw = chat_messages(
            messages=messages,
            model=cfg.model,
            max_tokens=384,
            temperature=0.0,
            timeout=cfg.timeout,
        )
    except Exception as e:
        err_rec: Dict[str, Any] = {
            "event": "http_err",
            "ts": time.time(),
            "error": repr(e),
            "description": description,
            "user_prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "candidate_memory_ids": cand,
            "model": cfg.model,
            "base_url": NEW_VLM_BASE_URL,
            "artifact_dir": str(call_dir) if call_dir else None,
            "input_image_files": input_image_files,
        }
        _append_jsonl(jsonl_path, err_rec)
        if call_dir is not None:
            try:
                (call_dir / "error.txt").write_text(repr(e), encoding="utf-8")
            except OSError:
                pass
        raise RuntimeError(f"rerank VLM request failed via client.py endpoint (strict mode): {e!r}") from e

    if call_dir is not None:
        try:
            (call_dir / "vlm_response_raw.txt").write_text(raw or "", encoding="utf-8")
        except OSError:
            pass

    try:
        parsed = _parse_json(raw or "")
        best = int(parsed.get("best_index"))
    except Exception as e:
        pe: Dict[str, Any] = {
            "event": "parse_err",
            "ts": time.time(),
            "error": repr(e),
            "description": description,
            "user_prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "raw_response": raw or "",
            "candidate_memory_ids": cand,
            "artifact_dir": str(call_dir) if call_dir else None,
            "input_image_files": input_image_files,
        }
        _append_jsonl(jsonl_path, pe)
        raise RuntimeError(f"rerank response parse failed (strict mode): {e!r}") from e

    if best < 1 or best > len(cand):
        oor: Dict[str, Any] = {
            "event": "best_index_out_of_range",
            "ts": time.time(),
            "best_index": best,
            "k": len(cand),
            "description": description,
            "user_prompt": prompt,
            "raw_response": raw or "",
            "parsed": parsed,
            "artifact_dir": str(call_dir) if call_dir else None,
        }
        _append_jsonl(jsonl_path, oor)
        raise RuntimeError(f"best_index_out_of_range: {best}, expected 1..{len(cand)} (strict mode)")

    chosen_mem = cand[best - 1]
    box = np.asarray(rep.object_box[chosen_mem], dtype=float).reshape(-1)
    if box.size < 3:
        raise RuntimeError(f"bad_box for chosen memory {chosen_mem} (strict mode)")

    target = box[:3].copy()
    target[[1, 2]] = target[[2, 1]]

    baseline_idx = int(decision_aux.get("real_object_decision_idx", -1))
    info["rerank_applied"] = True
    info["chosen_memory_index"] = chosen_mem
    info["baseline_memory_index"] = baseline_idx
    info["best_index_1based"] = best
    info["reason"] = str(parsed.get("reason", ""))[:1200]
    info["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    info["artifact_dir"] = str(call_dir) if call_dir else None

    ok_meta = {
        "event": "ok",
        "ts": time.time(),
        "chosen_memory_index": chosen_mem,
        "baseline_memory_index": baseline_idx,
        "best_index_1based": best,
        "k": len(cand),
        "candidate_memory_ids": cand,
        "description": description,
        "user_prompt": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "raw_response": raw or "",
        "parsed": parsed,
        "reason": str(parsed.get("reason", "")),
        "model": cfg.model,
        "base_url": NEW_VLM_BASE_URL,
        "elapsed_ms": info["elapsed_ms"],
        "artifact_dir": str(call_dir) if call_dir else None,
        "input_image_files": input_image_files,
    }
    _append_jsonl(jsonl_path, ok_meta)

    if call_dir is not None:
        try:
            (call_dir / "vlm_meta.json").write_text(
                json.dumps(
                    {
                        "parsed": parsed,
                        "chosen_memory_index": chosen_mem,
                        "baseline_memory_index": baseline_idx,
                        "elapsed_ms": info["elapsed_ms"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    return target, info


def rerank_memory_target(
    *,
    description: str,
    rep: Any,
    cfg: RerankConfig,
    baseline_memory_index: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    在 frontier 阶段也可调用：直接从 memory candidates 里让 VLM 选一个 target。
    """
    fake_aux = {
        "is_object_decision": True,
        "real_object_decision_idx": int(baseline_memory_index),
    }
    return rerank_object_target(
        description=description,
        rep=rep,
        baseline_target_xyz=np.zeros((3,), dtype=float),
        decision_aux=fake_aux,
        cfg=cfg,
    )


def should_run_rerank(task_level: str, cfg: RerankConfig) -> bool:
    return task_level in cfg.enabled_levels


def confirm_target_visible_from_rgb_views(
    *,
    description: str,
    rgb_views: List[np.ndarray],
    cfg: RerankConfig,
    max_images: int = 4,
) -> Dict[str, Any]:
    """VLM 二次确认：当前位置环视图是否已经看到目标物体。"""
    if len(rgb_views) == 0:
        raise RuntimeError("confirm_target_visible_from_rgb_views got empty rgb_views")
    # 优先策略：12 帧 -> 3 张 4 合 1 拼图。帧数不足时自动降级，避免临近 max_steps 时崩溃。
    tiles: List[np.ndarray] = []
    picked_indices: List[int] = []
    if len(rgb_views) >= 12:
        pano = rgb_views[:12]
        for t in range(3):
            base = t * 4
            imgs = pano[base : base + 4]
            h, w, _ = imgs[0].shape
            tile = np.zeros((h * 2, w * 2, 3), dtype=imgs[0].dtype)
            tile[0:h, 0:w] = imgs[0]
            tile[0:h, w : 2 * w] = imgs[1]
            tile[h : 2 * h, 0:w] = imgs[2]
            tile[h : 2 * h, w : 2 * w] = imgs[3]
            tiles.append(tile)
            picked_indices.append(t)
    else:
        # 降级：尽量按 4 帧拼图；若不足 4 帧则直接抽样原帧送 VLM。
        full_groups = len(rgb_views) // 4
        for g in range(full_groups):
            base = g * 4
            imgs = rgb_views[base : base + 4]
            h, w, _ = imgs[0].shape
            tile = np.zeros((h * 2, w * 2, 3), dtype=imgs[0].dtype)
            tile[0:h, 0:w] = imgs[0]
            tile[0:h, w : 2 * w] = imgs[1]
            tile[h : 2 * h, 0:w] = imgs[2]
            tile[h : 2 * h, w : 2 * w] = imgs[3]
            tiles.append(tile)
            picked_indices.append(g)
        if len(tiles) == 0:
            n = len(rgb_views)
            k = min(max_images, n)
            if n <= k:
                idxs = list(range(n))
            else:
                idxs = np.linspace(0, n - 1, num=k, dtype=int).tolist()
            for i in idxs:
                tiles.append(rgb_views[i])
            picked_indices = idxs

    prompt = CONFIRM_USER_TEMPLATE.format(description=description.strip(), k=len(tiles))
    messages = [{"role": "system", "content": CONFIRM_SYSTEM_PROMPT}]
    content: List[Dict[str, Any]] = []
    for tile in tiles:
        content.append({"type": "image_url", "image_url": {"url": numpy_rgb_to_jpeg_data_url(tile)}})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})

    t0 = time.perf_counter()
    raw = chat_messages(
        messages=messages,
        model=cfg.model,
        max_tokens=256,
        temperature=0.0,
        timeout=cfg.timeout,
    )
    parsed = _parse_json(raw or "")
    has_target = bool(parsed.get("has_target", False))
    return {
        "has_target": has_target,
        "reason": str(parsed.get("reason", ""))[:1200],
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "picked_indices": picked_indices,
        "raw_response": raw,
    }
