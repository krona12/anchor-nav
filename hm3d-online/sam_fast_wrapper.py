import os
from pathlib import Path

import cv2
import numpy as np
import torch


SAM_DIR = Path(__file__).resolve().parent / "SAM"
DEFAULT_SAM_CHECKPOINT = SAM_DIR / "sam_vit_h_4b8939.pth"

# Per-task-level SAM parameter profiles.
# Rationale:
#   FAST-SAM uses conf=0.1 (very low) + iou=0.9 (high NMS), producing ~20-60 object-centric masks.
#   Current SAM defaults (pred_iou_thresh=0.88, stability_score_thresh=0.95, max_masks=0) produce
#   100-300 masks with no cap, overwhelming the downstream PQ3D super-point pipeline.
#
#   object  – simplest query, category-level only; fewer masks + faster inference is sufficient.
#   room    – must identify a room first, then the object; moderate coverage needed.
#   region  – multiple landmark objects required for localisation; wider mask recall helps.
#   instance– finest discrimination between similar objects; densest sampling + crops + lowest thresh.
#
# Select preset via SAM_LEVEL_PRESET env var: "normal" (default) or "loose" (faster).
LEVEL_PARAMS = {
    "object": dict(
        points_per_side=16,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        max_masks=32,
        min_mask_region_area=200,
        crop_n_layers=0,
    ),
    "room": dict(
        points_per_side=16,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.88,
        max_masks=40,
        min_mask_region_area=150,
        crop_n_layers=0,
    ),
    "region": dict(
        points_per_side=24,
        pred_iou_thresh=0.78,
        stability_score_thresh=0.86,
        max_masks=52,
        min_mask_region_area=100,
        crop_n_layers=0,
    ),
    "instance": dict(
        points_per_side=32,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.84,
        max_masks=64,
        min_mask_region_area=50,
        crop_n_layers=1,
    ),
}

# Loose preset: prioritises speed over density.
#   points_per_side halved vs normal → 4x fewer grid prompts.
#   crop_n_layers=0 for all levels (instance was 1, saved ~6x on that level alone).
#   Lower thresholds to compensate for sparser sampling and maintain recall.
#   Estimated forward-pass batches per frame (points_per_batch=64):
#     object/room: 8²/64=1 batch  (~50-100ms)
#     region:     12²/64=3 batches (~150-300ms)
#     instance:   16²/64=4 batches (~200-400ms)
LOOSE_LEVEL_PARAMS = {
    "object": dict(
        points_per_side=8,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.82,
        max_masks=24,
        min_mask_region_area=400,
        crop_n_layers=0,
    ),
    "room": dict(
        points_per_side=8,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.82,
        max_masks=28,
        min_mask_region_area=300,
        crop_n_layers=0,
    ),
    "region": dict(
        points_per_side=12,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.80,
        max_masks=36,
        min_mask_region_area=200,
        crop_n_layers=0,
    ),
    "instance": dict(
        points_per_side=16,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.78,
        max_masks=48,
        min_mask_region_area=100,
        crop_n_layers=0,
    ),
}

_LEVEL_PRESETS = {
    "normal": LEVEL_PARAMS,
    "loose": LOOSE_LEVEL_PARAMS,
}


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else float(default)


class _MaskData:
    def __init__(self, data):
        self.data = data


class _BoxData:
    def __init__(self, data, conf):
        self.data = data
        self.conf = conf


class _FastSAMLikeResult:
    def __init__(self, masks, boxes, conf):
        self.masks = _MaskData(masks)
        self.boxes = _BoxData(boxes, conf)


class SamFastSAM:
    """FastSAM-compatible adapter around Meta SAM automatic mask generation.

    Supports per-task-level parameter profiles (see LEVEL_PARAMS).  Pass
    ``task_level=`` to ``__call__`` to activate the matching profile; omit it
    to fall back to the legacy single-generator behaviour (env-var controlled).
    """

    def __init__(self, checkpoint=None):
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "segment-anything is not installed. Run: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        self.model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h").strip()
        checkpoint = os.environ.get("SAM_CHECKPOINT") or checkpoint or str(DEFAULT_SAM_CHECKPOINT)
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint.resolve()}. "
                "Put sam_vit_h_4b8939.pth under hm3d-online/SAM/ or set SAM_CHECKPOINT."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        points_per_batch = _env_int("SAM_POINTS_PER_BATCH", 64)

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        sam.to(device=self.device)
        sam.eval()

        # Legacy single generator (env-var driven, kept for backward compatibility).
        legacy_pts = _env_int("SAM_POINTS_PER_SIDE", 32)
        legacy_iou = _env_float("SAM_PRED_IOU_THRESH", 0.88)
        legacy_stab = _env_float("SAM_STABILITY_SCORE_THRESH", 0.95)
        legacy_max = _env_int("SAM_MAX_MASKS", 0)
        self._legacy_max_masks = legacy_max
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=legacy_pts,
            points_per_batch=points_per_batch,
            pred_iou_thresh=legacy_iou,
            stability_score_thresh=legacy_stab,
            crop_n_layers=_env_int("SAM_CROP_N_LAYERS", 0),
            crop_n_points_downscale_factor=_env_int("SAM_CROP_N_POINTS_DOWNSCALE_FACTOR", 1),
            min_mask_region_area=_env_int("SAM_MIN_MASK_REGION_AREA", 100),
            output_mode="binary_mask",
        )

        # Per-level generators: preset selected via SAM_LEVEL_PRESET ("normal" or "loose").
        level_preset_name = os.environ.get("SAM_LEVEL_PRESET", "normal").strip().lower()
        active_level_params = _LEVEL_PRESETS.get(level_preset_name, LEVEL_PARAMS)
        self._level_generators: dict = {}
        self._level_max_masks: dict = {}
        for level, p in active_level_params.items():
            self._level_max_masks[level] = p["max_masks"]
            self._level_generators[level] = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=p["points_per_side"],
                points_per_batch=points_per_batch,
                pred_iou_thresh=p["pred_iou_thresh"],
                stability_score_thresh=p["stability_score_thresh"],
                crop_n_layers=p["crop_n_layers"],
                crop_n_points_downscale_factor=1,
                min_mask_region_area=p["min_mask_region_area"],
                output_mode="binary_mask",
            )

        print(
            "[sam-baseline] "
            f"model_type={self.model_type} checkpoint={self.checkpoint} device={self.device} "
            f"level_preset={level_preset_name} "
            f"legacy(pts={legacy_pts},iou={legacy_iou},stab={legacy_stab},max={legacy_max}) "
            f"per-level profiles={list(active_level_params.keys())}",
            flush=True,
        )

    def __call__(self, images, *args, task_level: str = "", **kwargs):
        if isinstance(images, np.ndarray):
            image_list = [images]
        else:
            image_list = list(images)
        gen = self._level_generators.get(task_level, self.generator)
        max_masks = self._level_max_masks.get(task_level, self._legacy_max_masks)
        return [self._generate_one(image, gen, max_masks) for image in image_list]

    def _generate_one(self, image, generator=None, max_masks: int = 0):
        if generator is None:
            generator = self.generator
            max_masks = self._legacy_max_masks

        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        h, w = image.shape[:2]
        with torch.no_grad():
            annotations = generator.generate(image)
        annotations = sorted(annotations, key=lambda x: int(x.get("area", 0)), reverse=True)
        if max_masks > 0:
            annotations = annotations[:max_masks]

        mask_list = []
        box_list = []
        conf_list = []
        for ann in annotations:
            mask = np.asarray(ann["segmentation"])
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = mask.astype(bool)
            if not mask.any():
                continue

            x, y, bw, bh = [float(v) for v in ann.get("bbox", [0, 0, w, h])]
            box_list.append([x, y, x + bw, y + bh])
            conf_list.append(float(ann.get("predicted_iou", ann.get("stability_score", 0.0))))
            mask_list.append(mask)

        if mask_list:
            masks = torch.from_numpy(np.stack(mask_list, axis=0)).float()
            boxes = torch.tensor(box_list, dtype=torch.float32)
            conf = torch.tensor(conf_list, dtype=torch.float32)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            conf = torch.zeros((0,), dtype=torch.float32)
        return _FastSAMLikeResult(masks=masks, boxes=boxes, conf=conf)
