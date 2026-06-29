import inspect
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_DIR = Path(__file__).resolve().parent / "SAM2"
DEFAULT_SAM2_CHECKPOINT = SAM2_DIR / "sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

warnings.filterwarnings(
    "ignore",
    message=r"The default value of the antialias parameter.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated.*",
    category=UserWarning,
)


SAM2_LEVEL_PROFILES = {
    "sanity": {
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
    },
    "balanced": {
        "object": dict(
            points_per_side=12,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88,
            max_masks=32,
            min_mask_region_area=200,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "room": dict(
            points_per_side=16,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88,
            max_masks=40,
            min_mask_region_area=150,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "region": dict(
            points_per_side=20,
            pred_iou_thresh=0.78,
            stability_score_thresh=0.86,
            max_masks=52,
            min_mask_region_area=100,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "instance": dict(
            points_per_side=24,
            pred_iou_thresh=0.76,
            stability_score_thresh=0.84,
            max_masks=64,
            min_mask_region_area=50,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
    },
    "quality": {
        "object": dict(
            points_per_side=16,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88,
            max_masks=36,
            min_mask_region_area=200,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "room": dict(
            points_per_side=20,
            pred_iou_thresh=0.78,
            stability_score_thresh=0.86,
            max_masks=44,
            min_mask_region_area=150,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "region": dict(
            points_per_side=28,
            pred_iou_thresh=0.76,
            stability_score_thresh=0.84,
            max_masks=60,
            min_mask_region_area=100,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            use_m2m=False,
        ),
        "instance": dict(
            points_per_side=32,
            pred_iou_thresh=0.72,
            stability_score_thresh=0.82,
            max_masks=72,
            min_mask_region_area=50,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            use_m2m=True,
        ),
    },
}


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else float(default)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _maybe_add_sam2_repo_root():
    repo_root = os.environ.get("SAM2_REPO_ROOT", "").strip()
    if not repo_root:
        return
    repo_root = str(Path(repo_root).expanduser().resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _filter_kwargs(callable_obj, kwargs):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


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


class Sam2FastSAM:
    """FastSAM-compatible adapter around SAM2 automatic mask generation."""

    def __init__(self, checkpoint=None):
        _maybe_add_sam2_repo_root()
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "sam2 is not installed in the active environment. Install facebookresearch/sam2 "
                "inside mtu3d, or set SAM2_REPO_ROOT to an installed checkout."
            ) from e

        checkpoint = os.environ.get("SAM2_CHECKPOINT") or checkpoint or str(DEFAULT_SAM2_CHECKPOINT)
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint.resolve()}. "
                "Put sam2.1_hiera_large.pt under hm3d-online/SAM2/ or set SAM2_CHECKPOINT."
            )

        self.model_cfg = os.environ.get("SAM2_MODEL_CFG", DEFAULT_SAM2_MODEL_CFG).strip()
        self.device = os.environ.get("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip()
        self.use_bf16 = _env_bool("SAM2_USE_BF16", True)
        points_per_batch = _env_int("SAM2_POINTS_PER_BATCH", 64)
        preset_name = os.environ.get("SAM2_LEVEL_PRESET", "balanced").strip().lower()
        active_profiles = SAM2_LEVEL_PROFILES.get(preset_name, SAM2_LEVEL_PROFILES["balanced"])

        build_kwargs = _filter_kwargs(
            build_sam2,
            dict(
                config_file=self.model_cfg,
                ckpt_path=str(self.checkpoint),
                device=self.device,
                apply_postprocessing=_env_bool("SAM2_APPLY_POSTPROCESSING", True),
            ),
        )
        sam2 = build_sam2(**build_kwargs)
        sam2.eval()
        self.model = sam2

        self._legacy_max_masks = _env_int("SAM2_MAX_MASKS", 0)
        legacy_kwargs = dict(
            model=sam2,
            points_per_side=_env_int("SAM2_POINTS_PER_SIDE", 24),
            points_per_batch=points_per_batch,
            pred_iou_thresh=_env_float("SAM2_PRED_IOU_THRESH", 0.78),
            stability_score_thresh=_env_float("SAM2_STABILITY_SCORE_THRESH", 0.86),
            stability_score_offset=_env_float("SAM2_STABILITY_SCORE_OFFSET", 1.0),
            crop_n_layers=_env_int("SAM2_CROP_N_LAYERS", 0),
            crop_n_points_downscale_factor=_env_int("SAM2_CROP_N_POINTS_DOWNSCALE_FACTOR", 1),
            min_mask_region_area=_env_int("SAM2_MIN_MASK_REGION_AREA", 100),
            output_mode="binary_mask",
            use_m2m=_env_bool("SAM2_USE_M2M", False),
            multimask_output=_env_bool("SAM2_MULTIMASK_OUTPUT", True),
        )
        self.generator = SAM2AutomaticMaskGenerator(
            **_filter_kwargs(SAM2AutomaticMaskGenerator, legacy_kwargs)
        )

        self._level_generators = {}
        self._level_max_masks = {}
        for level, params in active_profiles.items():
            self._level_max_masks[level] = int(params["max_masks"])
            gen_kwargs = dict(
                model=sam2,
                points_per_side=params["points_per_side"],
                points_per_batch=points_per_batch,
                pred_iou_thresh=params["pred_iou_thresh"],
                stability_score_thresh=params["stability_score_thresh"],
                stability_score_offset=params.get("stability_score_offset", 1.0),
                crop_n_layers=params["crop_n_layers"],
                crop_n_points_downscale_factor=params.get("crop_n_points_downscale_factor", 1),
                min_mask_region_area=params["min_mask_region_area"],
                output_mode="binary_mask",
                use_m2m=params.get("use_m2m", False),
                multimask_output=params.get("multimask_output", True),
            )
            self._level_generators[level] = SAM2AutomaticMaskGenerator(
                **_filter_kwargs(SAM2AutomaticMaskGenerator, gen_kwargs)
            )

        print(
            "[sam2-baseline] "
            f"checkpoint={self.checkpoint} cfg={self.model_cfg} device={self.device} "
            f"level_preset={preset_name} points_per_batch={points_per_batch} "
            f"use_bf16={self.use_bf16} profiles={list(active_profiles.keys())}",
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

    def _generate_one(self, image, generator, max_masks: int = 0):
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        h, w = image.shape[:2]
        with torch.inference_mode():
            if self.device.startswith("cuda") and self.use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    annotations = generator.generate(image)
            else:
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
