"""Co-select：任务级 main_target + anchor（VLM），以及在 final object decision 时对 top-K 记忆槽做 CLIP 重排。

- VLM 走 ``hm3d-online/vlm/client.py`` 的 ``chat``。
- 任务开始时与 ``anchor_nav.vfv.decompose_target_anchor`` 对齐做一次文本分解；
  final decision 时若有环视拼图则再调 VLM（带图）精炼 JSON。
- CLIP：同一 ``CLIPModel`` 的 ``get_text_features`` / ``get_image_features``；默认 ``CLIP_LOCAL_PATH`` 供 PQ3D 对齐，若目录缺 preprocessor 则自动改用 ``CLIP_HF_REPO_ID`` 走缓存。
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anchor_nav.semantic_enhance import SemanticEnhanceConfig, _load_clip_model
from anchor_nav.vfv import VfvDecompose, _try_parse_json_obj, decompose_target_anchor, dedupe_anchor_phrases
from vlm.client import chat

# PQ3D tokenizer 可用纯 snapshot 目录；CLIPProcessor 还需要 preprocessor_config.json，该目录常缺省。
CLIP_LOCAL_PATH = "/home/chenlin/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
CLIP_HF_REPO_ID = "openai/clip-vit-large-patch14"


@dataclass
class CoselectVlmDecompose:
    main_target: str
    anchor: str
    anchor_descs: List[str] = field(default_factory=list)
    raw: str = ""
    parse_ok: bool = True
    phase: str = ""  # task_start | final_decision


def vfv_to_coselect(d: VfvDecompose, *, phase: str) -> CoselectVlmDecompose:
    return CoselectVlmDecompose(
        main_target=str(d.target_desc or "").strip(),
        anchor=str(d.anchor_desc or "").strip(),
        anchor_descs=list(d.anchor_descs or []),
        raw=str(d.raw or ""),
        parse_ok=bool(d.parse_ok),
        phase=phase,
    )


def extract_main_target_anchor_at_task_start(task_sentence: str, vlm_model: str) -> CoselectVlmDecompose:
    """任务开始时：纯文本 VLM 分解（与 VFV 共用 ``decompose_target_anchor``）。"""
    d = decompose_target_anchor(task_sentence, vlm_model)
    return vfv_to_coselect(d, phase="task_start")


def extract_main_target_anchor_at_final_decision(
    task_sentence: str,
    vlm_model: str,
    *,
    panorama_path: Optional[Path],
    task_start: Optional[CoselectVlmDecompose] = None,
    max_tokens: int = 320,
) -> CoselectVlmDecompose:
    """Final object decision 时：优先带环视拼图再调 VLM；失败则回退为任务开始同款文本分解。"""
    hints = {}
    if task_start is not None:
        hints = {
            "main_target_hint": task_start.main_target,
            "anchor_hint": task_start.anchor,
            "anchor_descs_hint": list(task_start.anchor_descs or []),
        }
    p = Path(panorama_path) if panorama_path is not None else None
    if p is not None and p.is_file():
        prompt = (
            "You are at the FINAL object-navigation decision. A stitched 360 panorama is attached.\n"
            "Extract concise phrases for re-ranking object detections with CLIP.\n"
            "Return strict JSON only:\n"
            '{"main_target":"short English noun phrase for the PRIMARY object to reach",'
            '"anchor":"short phrase for ONE distinctive co-visible object for disambiguation (or empty)",'
            '"anchor_descs":["optional","up","to","three"]}\n\n'
            f"Full task: {task_sentence}\n"
            f"Earlier decomposition hints (refine if needed; JSON): {json.dumps(hints, ensure_ascii=False)}\n"
        )
        try:
            raw = chat(text=prompt, image_path=p, model=vlm_model, max_tokens=max_tokens)
        except Exception as ex:
            raw = f'{{"error":"{ex}"}}'
        parsed = _try_parse_json_obj(str(raw or ""))
        if isinstance(parsed, dict) and str(parsed.get("main_target", "")).strip():
            raw_list: List[str] = []
            ad = parsed.get("anchor_descs")
            if isinstance(ad, list):
                raw_list.extend(str(x) for x in ad if str(x).strip())
            if not raw_list:
                legacy = str(parsed.get("anchor", "") or "").strip()
                if legacy:
                    raw_list.append(legacy)
            anchor_descs = dedupe_anchor_phrases(raw_list)
            anchor_primary = anchor_descs[0] if anchor_descs else str(parsed.get("anchor", "") or "").strip()
            return CoselectVlmDecompose(
                main_target=str(parsed.get("main_target", "")).strip(),
                anchor=str(anchor_primary or parsed.get("anchor", "") or "").strip(),
                anchor_descs=list(anchor_descs),
                raw=str(raw or ""),
                parse_ok=True,
                phase="final_decision",
            )
    return extract_main_target_anchor_at_task_start(task_sentence, vlm_model)


def _encode_clip_text_vec(text: str, *, model: Any, processor: Any, dev: str) -> np.ndarray:
    """与 ``CLIPModel.get_image_features`` 同一对比空间，使用 ``get_text_features``（勿用裸 CLIPTextModel.pooler）。"""
    import torch

    with torch.no_grad():
        batch = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(dev) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        feat = model.get_text_features(**batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


def _encode_clip_image_vec(rgb: np.ndarray, *, model: Any, processor: Any, dev: str) -> np.ndarray:
    import torch
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8))
    with torch.no_grad():
        batch = processor(images=img, return_tensors="pt")
        batch = {k: v.to(dev) for k, v in batch.items() if k == "pixel_values"}
        feat = model.get_image_features(**batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].detach().cpu().numpy().astype(np.float32)


def _clip_pretrained_for_full_model(path_or_id: str) -> Tuple[str, Optional[str]]:
    """``CLIPModel``/``CLIPProcessor`` 需完整仓库文件；仅含权重的 ``snapshots/...`` 常缺 ``preprocessor_config.json``。

    缺省时退回 ``CLIP_HF_REPO_ID``（仍走本机 Hugging Face 缓存，与 snapshot 同一模型权重）。
    """
    s = str(path_or_id or "").strip()
    if not s:
        return CLIP_HF_REPO_ID, "empty path; using hub id"
    p = Path(s)
    if not p.is_dir():
        return s, None
    has_pre = (p / "preprocessor_config.json").is_file() or (p / "processor_config.json").is_file()
    if has_pre:
        return str(p.resolve()), None
    return CLIP_HF_REPO_ID, (
        f"snapshot missing preprocessor_config.json; load CLIP via hub id {CLIP_HF_REPO_ID!r} "
        f"(HF cache only, same weights family as snapshot)"
    )


def _semantic_enhance_clip_cfg(clip_model_path: str, clip_device: Optional[str]) -> SemanticEnhanceConfig:
    """与 ``semantic_enhance._load_clip_*`` 一致：默认 cuda，无 GPU 时在加载函数内落到 cpu。"""
    dev = (clip_device or "cuda").strip()
    return SemanticEnhanceConfig(clip_model_path=str(clip_model_path), clip_device=dev)


def clip_rerank_top_objects(
    *,
    slots_in_og3d_order: Sequence[int],
    object_first_rgbs: Sequence[Optional[np.ndarray]],
    main_target: str,
    anchor: str,
    clip_model_name: str = CLIP_LOCAL_PATH,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对 top-K 槽位（已与 og3d 排序对齐）用 CLIP 图像-文本相似度重排。

    使用 **同一** ``CLIPModel`` 的 ``get_text_features`` / ``get_image_features``（对比空间一致）；
    不复用 ``semantic_enhance._encode_text_clip``（CLIPTextModel.pooler 与图像支路不对齐会导致全失败）。
    返回 (per_slot 记录列表按 **clip 新序** 排序, meta)。
    """
    resolved, load_note = _clip_pretrained_for_full_model(clip_model_name)
    meta: Dict[str, Any] = {
        "clip_model_path_requested": clip_model_name,
        "clip_model_path": resolved,
        "clip_model": resolved,
        "clip_backend": "CLIPModel.get_text_features+get_image_features",
        "device_requested": device,
        "device": None,
        "error": None,
    }
    if load_note:
        meta["clip_load_note"] = load_note
    rows: List[Dict[str, Any]] = []
    if not slots_in_og3d_order:
        return rows, meta

    texts: List[str] = [str(main_target or "").strip() or "object"]
    if str(anchor or "").strip():
        texts.append(str(anchor).strip())

    print(f"[coselect] CLIP from_pretrained: requested={clip_model_name!r} resolved={resolved!r}", file=sys.stderr, flush=True)

    try:
        cfg = _semantic_enhance_clip_cfg(resolved, device)
        import torch

        meta["device"] = cfg.clip_device if (cfg.clip_device == "cpu" or torch.cuda.is_available()) else "cpu"
        model, processor, dev = _load_clip_model(cfg.clip_model_path, cfg.clip_device)
        text_feats = [_encode_clip_text_vec(t, model=model, processor=processor, dev=dev) for t in texts]
    except Exception as ex:
        meta["error"] = str(ex)
        for og_rank, slot in enumerate(slots_in_og3d_order, start=1):
            rows.append(
                {
                    "slot_index": int(slot),
                    "og3d_rank": int(og_rank),
                    "clip_score": None,
                    "rerank_after_clip": int(og_rank),
                }
            )
        return rows, meta

    scores: List[Tuple[int, int, float]] = []
    for og_rank, slot in enumerate(slots_in_og3d_order, start=1):
        slot = int(slot)
        rgb = object_first_rgbs[slot] if 0 <= slot < len(object_first_rgbs) else None
        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3:
            sc = float("-inf")
        else:
            try:
                ifeat = _encode_clip_image_vec(rgb, model=model, processor=processor, dev=dev)
                inorm = float(np.linalg.norm(ifeat)) + 1e-12
                cos_vals: List[float] = []
                for tf in text_feats:
                    tn = float(np.linalg.norm(tf)) + 1e-12
                    cos_vals.append(float(np.dot(tf, ifeat) / (tn * inorm)))
                sc = float(np.mean(cos_vals)) if cos_vals else float("-inf")
            except Exception:
                sc = float("-inf")
        scores.append((slot, og_rank, sc))

    order = sorted(range(len(scores)), key=lambda i: scores[i][2], reverse=True)
    ranked_rows: List[Dict[str, Any]] = []
    for new_r, idx in enumerate(order, start=1):
        slot, og_rank, sc = scores[idx]
        ranked_rows.append(
            {
                "slot_index": int(slot),
                "og3d_rank": int(og_rank),
                "clip_score": float(sc) if np.isfinite(sc) else None,
                "rerank_after_clip": int(new_r),
            }
        )
    return ranked_rows, meta


def build_topk_slots_from_stage2(stage2_path: Path, topk: int = 5) -> Tuple[List[int], List[Dict[str, Any]]]:
    """从 ``stage2_decision.json`` 取 og3d_logit 前 topk 的 ``slot_index`` 列表（及原始候选信息）。"""
    if not stage2_path.is_file():
        return [], []
    with open(stage2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objs = data.get("object_candidates") or []
    kk = max(1, int(topk))
    ranked = sorted(objs, key=lambda x: float(x.get("og3d_logit", -1e30)), reverse=True)[:kk]
    slots = [int(x["slot_index"]) for x in ranked if "slot_index" in x]
    return slots, ranked


def run_coselect_final_bundle(
    *,
    task_sentence: str,
    vlm_model: str,
    panorama_path: Path,
    task_start: Optional[CoselectVlmDecompose],
    stage2_path: Path,
    object_first_rgbs: Sequence[Optional[np.ndarray]],
    clip_model_name: str = CLIP_LOCAL_PATH,
    clip_device: Optional[str],
    topk: int = 5,
    enable_clip: bool = True,
) -> Dict[str, Any]:
    """一次 final decision 的完整落盘结构（由脚本写入 JSON）。"""
    final_vlm = extract_main_target_anchor_at_final_decision(
        task_sentence,
        vlm_model,
        panorama_path=panorama_path,
        task_start=task_start,
    )
    slots, og_objs = build_topk_slots_from_stage2(stage2_path, topk=topk)
    if enable_clip:
        clip_rows, clip_meta = clip_rerank_top_objects(
            slots_in_og3d_order=slots,
            object_first_rgbs=object_first_rgbs,
            main_target=final_vlm.main_target,
            anchor=final_vlm.anchor,
            clip_model_name=clip_model_name,
            device=clip_device,
        )
    else:
        clip_meta = {"skipped": True, "reason": "enable_clip=False"}
        clip_rows = [
            {
                "slot_index": int(s),
                "og3d_rank": i + 1,
                "clip_score": None,
                "rerank_after_clip": i + 1,
            }
            for i, s in enumerate(slots)
        ]
    return {
        "task_sentence": task_sentence,
        "task_start_vlm": asdict(task_start) if task_start is not None else None,
        "final_decision_vlm": asdict(final_vlm),
        "og3d_topk_object_candidates": og_objs,
        "clip_rerank_topk": clip_rows,
        "clip_meta": clip_meta,
    }
