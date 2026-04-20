from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from vlm.client import DEFAULT_MODEL as VLM_DEFAULT_MODEL, chat
from .rerank import list_rerank_candidate_memory_ids


_TOKENIZER: Optional[CLIPTokenizer] = None
_TEXT_MODEL: Optional[CLIPTextModel] = None
_TEXT_MODEL_DEVICE: Optional[str] = None
_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROCESSOR: Optional[CLIPProcessor] = None
_CLIP_MODEL_DEVICE: Optional[str] = None


@dataclass
class SemanticEnhanceConfig:
    enabled_levels: Set[str] = field(default_factory=lambda: {"instance"})
    top_k: int = 8
    top_m: int = 5
    prob_temperature: float = 0.07
    clip_model_path: str = "openai/clip-vit-large-patch14"
    clip_device: str = "cuda"
    vlm_model: str = VLM_DEFAULT_MODEL
    override_final_on_apply: bool = False


def parse_levels_csv(s: str) -> Set[str]:
    parts = {p.strip() for p in (s or "").split(",") if p.strip()}
    return parts if parts else {"instance"}


def should_run_semantic_enhance(task_level: str, cfg: SemanticEnhanceConfig) -> bool:
    return task_level in cfg.enabled_levels


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _extract_main_target(description: str, vlm_model: str) -> str:
    prompt = (
        "Extract the main target object noun phrase from the navigation description. "
        "Return strict JSON only: {\"main_target\": \"...\"}\n\n"
        f"Description: {description}"
    )
    raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=128)
    parsed = _parse_json(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    if not main_target:
        raise RuntimeError(f"main_target extraction failed: raw={raw!r}")
    return main_target


def _extract_main_target_and_anchors(description: str, vlm_model: str) -> Tuple[str, List[str]]:
    prompt = (
        "Extract one main target and anchor objects from the navigation description.\n"
        "Rules:\n"
        "1) main_target must be one noun phrase only.\n"
        "2) anchors must be object noun phrases (no room names, no directions), deduplicated.\n"
        "3) anchors must NOT include main_target itself.\n"
        "Return strict JSON only: {\"main_target\": \"...\", \"anchors\": [\"...\", ...]}\n\n"
        f"Description: {description}"
    )
    raw = chat(text=prompt, image_path=None, model=vlm_model, max_tokens=256)
    parsed = _parse_json(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    anchors_raw = parsed.get("anchors", [])
    anchors: List[str] = []
    if isinstance(anchors_raw, list):
        for x in anchors_raw:
            s = str(x).strip()
            if s and s.lower() != main_target.lower() and s.lower() not in {a.lower() for a in anchors}:
                anchors.append(s)
    if not main_target:
        # 回退到旧接口，保持鲁棒
        main_target = _extract_main_target(description, vlm_model)
    return main_target, anchors


def _load_clip_text_model(model_path: str, device: str) -> Tuple[CLIPTokenizer, CLIPTextModel, str]:
    global _TOKENIZER, _TEXT_MODEL, _TEXT_MODEL_DEVICE
    want_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    if _TOKENIZER is None or _TEXT_MODEL is None or _TEXT_MODEL_DEVICE != want_device:
        _TOKENIZER = CLIPTokenizer.from_pretrained(model_path)
        _TEXT_MODEL = CLIPTextModel.from_pretrained(model_path).to(want_device).eval()
        _TEXT_MODEL_DEVICE = want_device
    return _TOKENIZER, _TEXT_MODEL, want_device


def _encode_text_clip(text: str, cfg: SemanticEnhanceConfig) -> np.ndarray:
    tok, model, dev = _load_clip_text_model(cfg.clip_model_path, cfg.clip_device)
    with torch.no_grad():
        t = tok([text], padding=True, truncation=True, return_tensors="pt")
        t = {k: v.to(dev) for k, v in t.items()}
        out = model(**t)
        feat = out.pooler_output[0]
        feat = torch.nn.functional.normalize(feat, p=2, dim=0)
    return feat.detach().cpu().numpy().astype(np.float32)


def _load_clip_model(model_path: str, device: str) -> Tuple[CLIPModel, CLIPProcessor, str]:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_MODEL_DEVICE
    want_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None or _CLIP_MODEL_DEVICE != want_device:
        _CLIP_MODEL = CLIPModel.from_pretrained(model_path).to(want_device).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_path)
        _CLIP_MODEL_DEVICE = want_device
    return _CLIP_MODEL, _CLIP_PROCESSOR, want_device


def _encode_image_clip(rgb: np.ndarray, cfg: SemanticEnhanceConfig) -> np.ndarray:
    model, processor, dev = _load_clip_model(cfg.clip_model_path, cfg.clip_device)
    img = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    pil = Image.fromarray(img)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        feat = model.get_image_features(**inputs)[0]
        feat = torch.nn.functional.normalize(feat, p=2, dim=0)
    return feat.detach().cpu().numpy().astype(np.float32)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    y = x / t
    y = y - np.max(y)
    e = np.exp(y)
    s = e / max(1e-12, float(np.sum(e)))
    return s


def _save_point_cloud_preview_jpg(path: Path, pts: np.ndarray, size: int = 640) -> None:
    """
    将点云导出为可直接查看的彩色俯视图（X-Z 平面投影），便于快速人工检查。
    pts: [N, >=3] 或 [N, >=6]；若有后 3 维则作为 RGB。
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 3:
        cv2.imwrite(str(path), canvas)
        return

    xyz = pts[:, :3].astype(np.float32)
    x = xyz[:, 0]
    z = xyz[:, 2]
    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_z, max_z = float(np.min(z)), float(np.max(z))
    span_x = max(1e-6, max_x - min_x)
    span_z = max(1e-6, max_z - min_z)
    pad = 10.0
    px = ((x - min_x) / span_x) * (size - 1 - 2 * pad) + pad
    py = ((z - min_z) / span_z) * (size - 1 - 2 * pad) + pad
    py = (size - 1) - py  # 让图看起来是“向上为前”
    pix = np.stack([px, py], axis=1).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, size - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, size - 1)

    if pts.shape[1] >= 6:
        rgb = pts[:, 3:6].astype(np.float32)
        if float(np.max(rgb)) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = np.tile(np.array([[255, 220, 0]], dtype=np.uint8), (len(pix), 1))

    for (u, v), c in zip(pix, rgb):
        canvas[v, u] = c[::-1]  # RGB -> BGR

    cv2.imwrite(str(path), canvas)


def _save_candidate_artifacts(call_dir: Path, rep: Any, rows: List[Dict[str, Any]]) -> None:
    rgb_list = getattr(rep, "object_first_rgb", None) or []
    point_cloud = np.asarray(getattr(rep, "point_cloud", np.zeros((0, 6))), dtype=float)
    object_mask = np.asarray(getattr(rep, "object_mask", np.zeros((0, 0))), dtype=bool)
    for rank, row in enumerate(rows, start=1):
        mid = int(row["memory_index"])
        if mid < len(rgb_list) and isinstance(rgb_list[mid], np.ndarray):
            rgb = rgb_list[mid][:, :, :3]
            cv2.imwrite(
                str(call_dir / f"top{rank:02d}_mem{mid:03d}_first_rgb.jpg"),
                cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
            )
        if object_mask.ndim == 2 and mid < object_mask.shape[1] and point_cloud.shape[0] == object_mask.shape[0]:
            pts = point_cloud[object_mask[:, mid]]
            np.save(call_dir / f"top{rank:02d}_mem{mid:03d}_point_cloud.npy", pts)
            _save_point_cloud_preview_jpg(call_dir / f"top{rank:02d}_mem{mid:03d}_point_cloud.jpg", pts)


def semantic_enhance_object_target(
    *,
    description: str,
    rep: Any,
    baseline_target_xyz: np.ndarray,
    decision_aux: Dict[str, Any],
    cfg: SemanticEnhanceConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"semantic_applied": False, "override_final_decision": False}
    if not decision_aux.get("is_object_decision"):
        info["skipped"] = "not_object_decision"
        return np.asarray(baseline_target_xyz, dtype=float).reshape(3).copy(), info

    cand = list_rerank_candidate_memory_ids(rep, top_k=cfg.top_k)
    if len(cand) == 0:
        info["skipped"] = "no_candidates"
        return np.asarray(baseline_target_xyz, dtype=float).reshape(3).copy(), info

    jsonl_path = os.environ.get("SEMANTIC_ENHANCE_LOG_JSONL", "").strip()
    io_root = os.environ.get("SEMANTIC_ENHANCE_IO_DIR", "").strip()
    call_dir: Optional[Path] = None
    if io_root:
        call_dir = Path(io_root) / f"call_{int(time.time() * 1000)}_{os.getpid()}"
        call_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        main_target, anchors = _extract_main_target_and_anchors(description, cfg.vlm_model)
    except Exception as e:
        info["error"] = f"main_target_or_clip:{e!r}"
        _append_jsonl(jsonl_path, {"event": "semantic_err", "error": info["error"], "description": description})
        return np.asarray(baseline_target_xyz, dtype=float).reshape(3).copy(), info

    rgb_list = getattr(rep, "object_first_rgb", None) or []
    ov = np.asarray(getattr(rep, "open_vocab_feat", np.zeros((0, 768))), dtype=float)
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    valid = [i for i in cand if i < len(box) and i < len(ov)]
    if len(valid) == 0:
        info["skipped"] = "no_valid_open_vocab_candidates"
        return np.asarray(baseline_target_xyz, dtype=float).reshape(3).copy(), info

    # 只在 top-k（通常设为 10）中做语义比对
    valid = valid[: int(cfg.top_k)]

    # target: 与 object 的 open_vocab_feature 相似度
    tfeat = _encode_text_clip(main_target, cfg)
    ov_valid = ov[valid]
    ov_norm = np.linalg.norm(ov_valid, axis=1, keepdims=True) + 1e-12
    tn = np.linalg.norm(tfeat) + 1e-12
    target_scores = (ov_valid @ tfeat) / (ov_norm[:, 0] * tn)  # [C]

    # anchors: 与 object 的 first_rgb 图像特征相似度（anchor 总权重 0.2）
    anchor_scores_agg = np.zeros((len(valid),), dtype=np.float32)
    anchor_similarity_records: List[Dict[str, Any]] = []
    if len(anchors) > 0:
        anchor_text_feats = np.stack([_encode_text_clip(a, cfg) for a in anchors], axis=0)  # [A, D]
        anchor_text_norm = np.linalg.norm(anchor_text_feats, axis=1, keepdims=True) + 1e-12
        for ci, mid in enumerate(valid):
            if mid < len(rgb_list) and isinstance(rgb_list[mid], np.ndarray):
                ifeat = _encode_image_clip(rgb_list[mid], cfg)
                inorm = np.linalg.norm(ifeat) + 1e-12
                per_anchor = (anchor_text_feats @ ifeat) / (anchor_text_norm[:, 0] * inorm)  # [A]
                anchor_scores_agg[ci] = float(np.mean(per_anchor))
            else:
                per_anchor = np.zeros((len(anchors),), dtype=np.float32)
                anchor_scores_agg[ci] = 0.0
            anchor_similarity_records.append(
                {
                    "memory_index": int(mid),
                    "anchors": [
                        {"anchor_text": anchors[ai], "cosine": float(per_anchor[ai])}
                        for ai in range(len(anchors))
                    ],
                    "anchor_aggregate_cosine": float(anchor_scores_agg[ci]),
                }
            )

    weighted_scores = 0.2 * target_scores + 0.8 * anchor_scores_agg
    best_local = int(np.argmax(weighted_scores))
    chosen_mid = int(valid[best_local])
    chosen_weighted = float(weighted_scores[best_local])

    # 日志：target 对每个 candidate 的相似度
    similarity_records: List[Dict[str, Any]] = []
    similarity_records.append(
        {
            "query_type": "target_open_vocab",
            "query_text": main_target,
            "candidates": [
                {"memory_index": int(mid), "cosine": float(target_scores[ci])}
                for ci, mid in enumerate(valid)
            ],
        }
    )

    # 记录各候选加权分并按降序排序（用于 top_rows 和可视化导出）
    rank_idx = np.argsort(-weighted_scores)
    top_n = min(int(cfg.top_m), len(rank_idx))
    top_idx = rank_idx[:top_n]
    top_rows: List[Dict[str, Any]] = []
    for ci in top_idx:
        mid = int(valid[int(ci)])
        top_rows.append(
            {
                "memory_index": mid,
                "weighted_score": float(weighted_scores[int(ci)]),
                "target_cosine": float(target_scores[int(ci)]),
                "anchor_aggregate_cosine": float(anchor_scores_agg[int(ci)]),
                "object_box_xyzwhd": box[mid].tolist(),
            }
        )

    probs = _softmax(np.asarray([x["weighted_score"] for x in top_rows], dtype=float), cfg.prob_temperature)
    target = np.asarray(box[chosen_mid, :3], dtype=float).reshape(3).copy()
    target[[1, 2]] = target[[2, 1]]

    if call_dir is not None:
        (call_dir / "semantic_top5.json").write_text(
            json.dumps(
                {
                    "description": description,
                    "main_target": main_target,
                    "anchors": anchors,
                    "topk_ranked": top_rows,
                    "probs": probs.tolist(),
                    "similarity_records": similarity_records,
                    "anchor_similarity_records": anchor_similarity_records,
                    "chosen_memory_index": chosen_mid,
                    "chosen_score_type": "0.2_target_open_vocab_plus_0.8_anchor_image_clip",
                    "chosen_weighted_score": chosen_weighted,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _save_candidate_artifacts(call_dir, rep, top_rows)

    info.update(
        {
            "semantic_applied": True,
            "override_final_decision": bool(cfg.override_final_on_apply),
            "main_target": main_target,
            "anchors": anchors,
            "top_candidates": top_rows,
            "top_probs": probs.tolist(),
            "similarity_records": similarity_records,
            "anchor_similarity_records": anchor_similarity_records,
            "chosen_memory_index": chosen_mid,
            "chosen_score_type": "0.2_target_open_vocab_plus_0.8_anchor_image_clip",
            "chosen_weighted_score": float(chosen_weighted),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "artifact_dir": str(call_dir) if call_dir is not None else None,
        }
    )
    _append_jsonl(
        jsonl_path,
        {
            "event": "semantic_ok",
            "description": description,
            "main_target": main_target,
            "anchors": anchors,
            "topk_ranked": top_rows,
            "probs": probs.tolist(),
            "similarity_records": similarity_records,
            "anchor_similarity_records": anchor_similarity_records,
            "chosen_memory_index": chosen_mid,
            "chosen_score_type": "0.2_target_open_vocab_plus_0.8_anchor_image_clip",
            "chosen_weighted_score": float(chosen_weighted),
            "elapsed_ms": info["elapsed_ms"],
            "artifact_dir": info["artifact_dir"],
        },
    )
    return target, info
