# SAM2.1 Hiera-Large 替代 FastSAM 方案

调研日期：2026-06-25  
目标仓库：`facebookresearch/sam2`，远端 HEAD `2b90b9f5ceec907a1c18123530e92e794ad901a4`  
目标模型：`sam2.1_hiera_large.pt` + `configs/sam2.1/sam2.1_hiera_l.yaml`

## 0. 结论先行

在当前 MTU3D / HM3D online baseline 中，用 **SAM2.1 Hiera-Large** 替代 FastSAM 是可行的，而且是“质量优先”路线里最值得尝试的一档。原因有三点：

1. 官方 SAM2 支持 image automatic mask generation，返回字段可以稳定转换成当前 `FastSAM` result 结构。
2. 本仓库已有 `sam_fast_wrapper.py`，已经验证过“把 SAM 做成 FastSAM-like adapter”这条工程路径。
3. 本地 N=700 历史实验显示，SAM ViT-H 相比 FastSAM 在多数模块，尤其是 instance 任务上有 SR/SPL 正向迹象；但 VISTALS 总 SR 和部分 object 任务也出现过退化，因此需要用 profile 控制 mask 粒度。

但不能直接把 FastSAM 调用替换为默认 SAM2 generator。更好的方案是：

- 第一阶段用 `SAM2AutomaticMaskGenerator` 做 **逐图 automatic masks**，不先上 video predictor。
- 保持 FastSAM-like 返回接口，避免改动 PQ3D stage1/stage2。
- 必须使用 `object/room/region/instance` 分级参数、`max_masks` 截断、`points_per_batch` 控制显存。
- GOAT 的 `object/description/image` 需要映射到 SAM2 profile，否则会退回 legacy profile，质量和速度都不可控。
- 先跑 `balanced` profile 做 N=700 对齐，再把 `instance` 切到 `quality` profile 做质量优先 sweep。

推荐主路线：

```text
FastSAM -> Sam2FastSAM adapter -> data_utils.PQ3DModel 不改主逻辑
refhm3d baseline -> sam2 runner monkeypatch
goat-nav -> 显式传 task_level
```

## 1. 官方仓库调研要点

### 1.1 SAM2.1 Hiera-Large 的定位

官方 README 将 SAM2 定义为图像和视频的 promptable segmentation foundation model；静态图像被看作单帧视频。SAM2.1 checkpoint 是 2024-09-30 发布的新一组 improved checkpoints，使用 SAM2.1 需要最新仓库代码。

官方表格中 `sam2.1_hiera_large` 的指标：

| 模型 | 参数量 | 官方速度 | SA-V test J&F | MOSE val J&F | LVOS v2 J&F |
|---|---:|---:|---:|---:|---:|
| `sam2.1_hiera_large` | 224.4M | 39.5 FPS | 79.5 | 74.6 | 80.6 |

注意：这个 39.5 FPS 是官方 benchmark 条件下的模型速度，测量环境是 A100 + torch 2.5.1 + CUDA 12.4。它不能直接等价为本项目 automatic all-mask generation 的速度，因为在线导航每次 decision 通常要处理 `goto` 历史帧加 12 张环视图，且 automatic generator 会按网格点反复 prompt。

### 1.2 安装约束

官方要求：

- Linux
- Python >= 3.10
- PyTorch >= 2.5.1
- torchvision >= 0.20.1
- CUDA toolkit 与 PyTorch CUDA 版本匹配
- Windows 推荐 WSL + Ubuntu

官方 INSTALL 也说明可以用 `SAM2_BUILD_CUDA=0` 跳过 SAM2 CUDA extension。这样会跳过去除小洞和小碎片的后处理，通常不影响主体结果。对本项目建议先这样安装，先让替换实验跑通，再考虑重装 extension。

### 1.3 Image automatic mask generation API

SAM2 提供 `SAM2AutomaticMaskGenerator`，核心行为和 SAM 的 automatic generator 类似：

- 对图像采样网格点 prompt。
- 每个点输出多候选 mask。
- 用 predicted IoU、stability score、NMS、crop NMS 等过滤和去重。
- 返回 list of dict。

关键默认参数如下；这不是完整参数表，只列出替换 FastSAM 时最需要调的项：

| 参数 | 默认值 | 对本项目的意义 |
|---|---:|---|
| `points_per_side` | 32 | 每边采样点数，总 prompt 数约为平方增长 |
| `points_per_batch` | 64 | 单批 prompt 数，越大越快但越吃显存 |
| `pred_iou_thresh` | 0.8 | mask 质量阈值 |
| `stability_score_thresh` | 0.95 | 稳定性阈值 |
| `stability_score_offset` | 1.0 | 计算 stability 时的偏移 |
| `box_nms_thresh` | 0.7 | 同 crop 内 mask box NMS |
| `crop_nms_thresh` | 0.7 | 跨 crop mask box NMS |
| `crop_n_layers` | 0 | 是否多 crop 再预测，质量提高但很慢 |
| `min_mask_region_area` | 0 | 小连通区域/小洞后处理阈值 |
| `output_mode` | `binary_mask` | 当前 PQ3D 最容易接，但内存占用最高 |
| `use_m2m` | False | 额外 refinement，质量优先时可开 |
| `multimask_output` | True | 每点输出多候选，召回更高但候选更多 |

`generate(image)` 输入 HWC uint8 RGB，返回字段包括：

- `segmentation`: H,W mask 或 RLE
- `bbox`: XYWH
- `area`: mask area
- `predicted_iou`: 模型预测质量分
- `stability_score`: mask 稳定性
- `crop_box`: crop 范围

当前项目需要把 `bbox` 从 XYWH 转成 XYXY，把 `predicted_iou` 映射到 FastSAM-like `.boxes.conf`。

## 2. 当前 baseline 的接入点

### 2.1 `data_utils.PQ3DModel`

核心路径在 `hm3d-online/data_utils.py`：

- `PQ3DModel.__init__` 中构造 `FastSAM(str(fastsam_ckpt))`。
- `decision()` 中调用：

```python
everything_result = self.mask_generator(
    color_list,
    device="cuda",
    retina_masks=True,
    imgsz=640,
    conf=0.1,
    iou=0.9,
    task_level=task_level,
)
```

下游会对每帧执行：

```python
masks = format_result(everything_result[idx])
masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
```

因此 SAM2 adapter 必须返回一个 FastSAM-like result：

```text
result.masks.data -> torch.Tensor[N,H,W]
result.boxes.data -> torch.Tensor[N,4]  # XYXY
result.boxes.conf -> torch.Tensor[N]
```

这个接口非常适合复用 `sam_fast_wrapper.py` 的设计。

### 2.2 已有 SAM ViT-H wrapper 的价值

`hm3d-online/sam_fast_wrapper.py` 已经做了四件关键事情：

- 将 Meta SAM automatic generator 包装成 FastSAM-compatible class。
- 将 annotation dict 转成 `.masks.data / .boxes.data / .boxes.conf`。
- 按 `object/room/region/instance` 设置不同 prompt 密度和阈值。
- 对 mask 按 area 降序并用 `max_masks` 截断。

SAM2 adapter 不应该另起炉灶，应当做成 `sam2_fast_wrapper.py`，结构和 `sam_fast_wrapper.py` 尽量一致。

### 2.3 GOAT 入口缺少 task level

`hm3d-online/refhm3d-nav-sequence-baseline.py` 已经把 `task_type` 传给 `PQ3DModel.decision(..., task_level=task_type)`。

但 `hm3d-online/goat-nav.py` 当前没有传 `task_level`。GOAT 里的 `goal_type` 有：

- `object`
- `description`
- `image`

建议映射：

| GOAT `goal_type` | SAM2 profile |
|---|---|
| `object` | `object` |
| `description` | `instance` |
| `image` | `instance` |

原因：`description` 和 `image` 都是实例级定位，需要细粒度 mask；`object` 是类别级，mask 过多反而可能拖慢和扰动 PQ3D。

## 3. 速度分析

### 3.1 不能直接用官方 FPS 判断在线速度

官方 39.5 FPS 是模型级 benchmark，automatic mask generation 的实际成本更接近：

```text
每图 prompt 数 ~= points_per_side^2
每图 batch 数 ~= ceil(prompt 数 / points_per_batch)
如果 crop_n_layers=1:
  且 crop_n_points_downscale_factor=2，则总 prompt 约为 2x
  若 downscale_factor=1，则总 prompt 约为 5x
如果 use_m2m=True:
  还会额外做一轮 refinement
```

当前 online decision 不是单图，而是多图。`goat-nav.py` 每次会加入最多 6 张 goto 历史帧，再转 12 步采 12 张新图，所以一次 decision 可能接近 12-18 张图。

### 3.2 本地 SAM ViT-H 历史速度参照

已有 `key_logs/fastsam2sam/results_summary.md` 说明 SAM ViT-H 在质量上有收益，但速度代价明显。N=700 日志里的平均任务时间大致为：

| 模块 | FastSAM avg task time | SAM ViT-H avg task time | 放大倍数 |
|---|---:|---:|---:|
| MQSC-R1 | 162.855s | 819.455s | 约 5.0x |
| VISTA2MQSC | 208.924s | 995.095s | 约 4.8x |
| VISTALS | 232.887s | 755.434s | 约 3.2x |

Baseline 的总时长也显示 SAM ViT-H 显著慢于 FastSAM，并且历史日志出现 OOM。

### 3.3 SAM2.1 Hiera-Large 的预期速度

论文摘要称 SAM2 在图像分割上比 SAM 更准确且更快。结合官方 Hiera-Large 模型速度，可以提出以下实验假设，但必须用本项目 automatic mask generation 实测验证：

- SAM2.1 Hiera-Large 有机会快于 SAM ViT-H，但不能在未跑 AMG 实验前当作事实。
- 但 automatic all-mask generation 仍很可能慢于 FastSAM，尤其是 `points_per_side >= 32`、`crop_n_layers=1`、`use_m2m=True` 时。
- 对在线导航，真正决定速度的是 profile，而不是只看 backbone 名字。

合理预期：

- `sanity` profile：主要用于接口验证，速度应明显好于 SAM ViT-H。
- `balanced` profile：目标是比 SAM ViT-H 快，并把质量提升和速度控制同时做到可跑 N=700；是否达成以日志为准。
- `quality` profile：允许牺牲速度，重点看 instance/description/image 是否显著优于 FastSAM 和 SAM ViT-H。
- 官方 automatic mask notebook 里的 refined 示例 `points_per_side=64` 不适合直接作为在线默认，只适合小样本质量上限测试。

## 4. 质量分析

### 4.1 为什么 SAM2.1 可能提升

当前 pipeline 的第一步是从 2D mask proposal 生成 3D superpoints，再进入 PQ3D stage1/stage2。mask proposal 的质量会影响：

- 物体边界是否干净。
- 小物体是否被召回。
- 相邻实例是否被分开。
- DINO feature 与 superpoint 的对应是否稳定。
- 3D 表征合并时是否出现错误粘连。

SAM2.1 Hiera-Large 相比 FastSAM 更适合质量优先的原因：

- Hiera-Large 是 SAM2.1 中质量最高的一档。
- SAM2 官方支持 image automatic masks，和原 SAM 的 automatic generator 语义接近。
- 本地 SAM ViT-H 替换已经证明高质量 mask 对 Baseline/MQSC/VISTA2MQSC 有正向作用。

### 4.2 最可能受益的任务

优先看：

- `instance`
- GOAT `description`
- GOAT `image`

历史 SAM ViT-H 结果里，VISTA2MQSC 的 instance SR 从 FastSAM 0.311 到 SAM 0.377，提升很大。这说明实例级任务对 mask 质量更敏感。

### 4.3 可能变差的任务

需要警惕 `object` level：

- 历史 VISTA2MQSC 中 object SR 出现过 SAM 低于 FastSAM 的情况。
- 类别级 object query 不一定需要非常细碎的 mask。
- 过多 mask 会让后续 3D superpoint 和 proposal 排序变复杂，可能引入噪声。

因此 object profile 应该保守：少点、较高阈值、较小 `max_masks`，不要默认 crop。

## 5. 推荐参数 profile

以下 profile 是替换实验的初始建议，不是最终超参。最终应按第 9 节的 sweep 固化。

### 5.1 `sanity` profile

用于确认 wrapper、环境、返回结构、单 episode 能跑通。

| level | points_per_side | pred_iou | stability | crop | use_m2m | max_masks | min_area |
|---|---:|---:|---:|---:|---|---:|---:|
| object | 8 | 0.75 | 0.82 | 0 | False | 24 | 400 |
| room | 8 | 0.75 | 0.82 | 0 | False | 28 | 300 |
| region | 12 | 0.72 | 0.80 | 0 | False | 36 | 200 |
| instance | 16 | 0.70 | 0.78 | 0 | False | 48 | 100 |

### 5.2 `balanced` profile

建议作为 N=700 第一轮正式对比。目标是比 SAM ViT-H 更快，同时保留明显质量优势。

| level | points_per_side | pred_iou | stability | crop | crop_downscale | use_m2m | max_masks | min_area |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| object | 12 | 0.80 | 0.88 | 0 | 1 | False | 32 | 200 |
| room | 16 | 0.80 | 0.88 | 0 | 1 | False | 40 | 150 |
| region | 20 | 0.78 | 0.86 | 0 | 1 | False | 52 | 100 |
| instance | 24 | 0.76 | 0.84 | 0 | 1 | False | 64 | 50 |

### 5.3 `quality` profile

质量优先配置。建议先只在 `instance` 或 GOAT `description/image` 上打开，再扩展到全任务。

| level | points_per_side | pred_iou | stability | crop | crop_downscale | use_m2m | max_masks | min_area |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| object | 16 | 0.80 | 0.88 | 0 | 1 | False | 36 | 200 |
| room | 20 | 0.78 | 0.86 | 0 | 1 | False | 44 | 150 |
| region | 28 | 0.76 | 0.84 | 0 | 1 | False | 60 | 100 |
| instance | 32 | 0.72 | 0.82 | 1 | 2 | True | 72 | 50 |

### 5.4 `max_quality_offline` profile

只用于单图可视化、小样本上限测试，不建议在线 N=700 默认使用。

| 参数 | 值 |
|---|---:|
| `points_per_side` | 64 |
| `points_per_batch` | 128 |
| `pred_iou_thresh` | 0.70 |
| `stability_score_thresh` | 0.92 |
| `stability_score_offset` | 0.70 |
| `crop_n_layers` | 1 |
| `crop_n_points_downscale_factor` | 2 |
| `use_m2m` | True |
| `max_masks` | 96 |

## 6. 工程替换方案

### 6.1 文件规划

建议新增：

```text
hm3d-online/sam2_fast_wrapper.py
hm3d-online/refhm3d-nav-sequence-sam2-runner.py
hm3d-online/goat-nav-sam2-runner.py
```

建议最小修改：

```text
hm3d-online/goat-nav.py
```

不建议第一步修改：

```text
hm3d-online/data_utils.py 的 PQ3D 主流程
PQ3D stage1/stage2
RepresentationManager
```

### 6.2 `sam2_fast_wrapper.py` 结构

骨架如下：

```python
import os
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_DIR = Path(__file__).resolve().parent / "SAM2"
DEFAULT_SAM2_CHECKPOINT = SAM2_DIR / "sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

SAM2_LEVEL_PROFILES = {
    "sanity": {
        "object": dict(points_per_side=8, pred_iou_thresh=0.75, stability_score_thresh=0.82, max_masks=24, min_mask_region_area=400, crop_n_layers=0),
        "room": dict(points_per_side=8, pred_iou_thresh=0.75, stability_score_thresh=0.82, max_masks=28, min_mask_region_area=300, crop_n_layers=0),
        "region": dict(points_per_side=12, pred_iou_thresh=0.72, stability_score_thresh=0.80, max_masks=36, min_mask_region_area=200, crop_n_layers=0),
        "instance": dict(points_per_side=16, pred_iou_thresh=0.70, stability_score_thresh=0.78, max_masks=48, min_mask_region_area=100, crop_n_layers=0),
    },
    "balanced": {
        "object": dict(points_per_side=12, pred_iou_thresh=0.80, stability_score_thresh=0.88, max_masks=32, min_mask_region_area=200, crop_n_layers=0),
        "room": dict(points_per_side=16, pred_iou_thresh=0.80, stability_score_thresh=0.88, max_masks=40, min_mask_region_area=150, crop_n_layers=0),
        "region": dict(points_per_side=20, pred_iou_thresh=0.78, stability_score_thresh=0.86, max_masks=52, min_mask_region_area=100, crop_n_layers=0),
        "instance": dict(points_per_side=24, pred_iou_thresh=0.76, stability_score_thresh=0.84, max_masks=64, min_mask_region_area=50, crop_n_layers=0),
    },
    "quality": {
        "object": dict(points_per_side=16, pred_iou_thresh=0.80, stability_score_thresh=0.88, max_masks=36, min_mask_region_area=200, crop_n_layers=0),
        "room": dict(points_per_side=20, pred_iou_thresh=0.78, stability_score_thresh=0.86, max_masks=44, min_mask_region_area=150, crop_n_layers=0),
        "region": dict(points_per_side=28, pred_iou_thresh=0.76, stability_score_thresh=0.84, max_masks=60, min_mask_region_area=100, crop_n_layers=0),
        "instance": dict(points_per_side=32, pred_iou_thresh=0.72, stability_score_thresh=0.82, max_masks=72, min_mask_region_area=50, crop_n_layers=1, crop_n_points_downscale_factor=2, use_m2m=True),
    },
    "max_quality_offline": {
        level: dict(points_per_side=64, pred_iou_thresh=0.70, stability_score_thresh=0.92, stability_score_offset=0.70, max_masks=96, min_mask_region_area=25, crop_n_layers=1, crop_n_points_downscale_factor=2, use_m2m=True)
        for level in ("object", "room", "region", "instance")
    },
}


def _select_profiles(name):
    preset = (name or "balanced").strip().lower()
    return SAM2_LEVEL_PROFILES.get(preset, SAM2_LEVEL_PROFILES["balanced"])


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
    def __init__(self, checkpoint=None):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        checkpoint = os.environ.get("SAM2_CHECKPOINT") or checkpoint or str(DEFAULT_SAM2_CHECKPOINT)
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint.resolve()}")

        self.model_cfg = os.environ.get("SAM2_MODEL_CFG", DEFAULT_SAM2_MODEL_CFG)
        self.device = os.environ.get("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = os.environ.get("SAM2_USE_BF16", "1") not in ("0", "false", "False")
        points_per_batch = int(os.environ.get("SAM2_POINTS_PER_BATCH", "64"))

        sam2 = build_sam2(
            self.model_cfg,
            str(self.checkpoint),
            device=self.device,
            apply_postprocessing=os.environ.get("SAM2_APPLY_POSTPROCESSING", "0") == "1",
        )
        sam2.eval()
        self.model = sam2

        active_profiles = _select_profiles(os.environ.get("SAM2_LEVEL_PRESET", "balanced"))
        self.generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=int(os.environ.get("SAM2_POINTS_PER_SIDE", "24")),
            points_per_batch=points_per_batch,
            pred_iou_thresh=float(os.environ.get("SAM2_PRED_IOU_THRESH", "0.78")),
            stability_score_thresh=float(os.environ.get("SAM2_STABILITY_SCORE_THRESH", "0.86")),
            crop_n_layers=int(os.environ.get("SAM2_CROP_N_LAYERS", "0")),
            crop_n_points_downscale_factor=int(os.environ.get("SAM2_CROP_N_POINTS_DOWNSCALE_FACTOR", "1")),
            min_mask_region_area=int(os.environ.get("SAM2_MIN_MASK_REGION_AREA", "100")),
            output_mode="binary_mask",
            use_m2m=os.environ.get("SAM2_USE_M2M", "0") == "1",
        )
        self._legacy_max_masks = int(os.environ.get("SAM2_MAX_MASKS", "0"))

        self._level_generators = {}
        self._level_max_masks = {}
        for level, p in active_profiles.items():
            self._level_max_masks[level] = p["max_masks"]
            self._level_generators[level] = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=p["points_per_side"],
                points_per_batch=points_per_batch,
                pred_iou_thresh=p["pred_iou_thresh"],
                stability_score_thresh=p["stability_score_thresh"],
                stability_score_offset=p.get("stability_score_offset", 1.0),
                crop_n_layers=p["crop_n_layers"],
                crop_n_points_downscale_factor=p.get("crop_n_points_downscale_factor", 1),
                min_mask_region_area=p["min_mask_region_area"],
                output_mode="binary_mask",
                use_m2m=p.get("use_m2m", False),
                multimask_output=p.get("multimask_output", True),
            )

    def __call__(self, images, *args, task_level: str = "", **kwargs):
        image_list = [images] if isinstance(images, np.ndarray) else list(images)
        gen = self._level_generators.get(task_level, self.generator)
        max_masks = self._level_max_masks.get(task_level, self._legacy_max_masks)
        return [self._generate_one(image, gen, max_masks) for image in image_list]

    def _generate_one(self, image, generator, max_masks):
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

        mask_list, box_list, conf_list = [], [], []
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
```

实现时要注意三点：

1. BF16 建议默认开启，但如果 GPU 对 BF16 支持不好或结果异常，立刻设 `SAM2_USE_BF16=0` 回到 FP32。
2. 输出 tensor 建议先放 CPU，因为 `format_result()` 会转 CPU numpy；这能降低和 DINO/PQ3D 同进程时的 GPU 峰值压力。
3. `SAM2AutomaticMaskGenerator` 多个实例共享同一个 SAM2 model 是可行的，但不要并发调用同一个 wrapper 实例；当前导航流程是串行 decision，符合这个假设。

### 6.3 SAM2 runner

新增 `hm3d-online/refhm3d-nav-sequence-sam2-runner.py`，约定保持和现有 `refhm3d-nav-sequence-sam-runner.py` 一致：第一个参数是目标脚本，后面才是目标脚本参数。

```python
#!/usr/bin/env python3
import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sam2_fast_wrapper import DEFAULT_SAM2_CHECKPOINT, Sam2FastSAM
import data_utils as _base_data_utils


def _usage():
    print(
        "Usage: python hm3d-online/refhm3d-nav-sequence-sam2-runner.py "
        "<target_refhm3d_script.py> [target args...]",
        file=sys.stderr,
    )


if len(sys.argv) < 2:
    _usage()
    raise SystemExit(2)

target_script = Path(sys.argv[1])
if not target_script.is_absolute():
    target_script = PROJECT_ROOT / target_script
target_script = target_script.resolve()

if not target_script.is_file():
    print(f"SAM2 target script not found: {target_script}", file=sys.stderr)
    raise SystemExit(2)

sam2_ckpt = Path(os.environ.get("SAM2_CHECKPOINT", str(DEFAULT_SAM2_CHECKPOINT))).expanduser()
os.environ["SAM2_CHECKPOINT"] = str(sam2_ckpt)
os.environ["FASTSAM_WEIGHT"] = str(sam2_ckpt)
_base_data_utils.FastSAM = Sam2FastSAM

print(
    f"[sam2-runner] target={target_script} "
    f"checkpoint={sam2_ckpt} preset={os.environ.get('SAM2_LEVEL_PRESET', 'balanced')}",
    flush=True,
)

sys.argv = [str(target_script), *sys.argv[2:]]
runpy.run_path(str(target_script), run_name="__main__")
```

这样不动 `data_utils.py` 主逻辑。

### 6.4 GOAT 的 task_level 修改

`goat-nav.py` 里构造 goal 后加入：

```python
sam_task_level = "object" if goal_type == "object" else "instance"
```

decision 调用建议改成 keyword，避免后续接口变化时位置参数变脆弱，同时显式传入 `task_level`：

```python
if goal_type == "image":
    target_position, is_final_decision = pq3d_model.decision(
        color_list,
        depth_list,
        agent_state_list,
        frontier_waypoints,
        sentence,
        decision_num,
        image_feat=goal_image_feat,
        task_level=sam_task_level,
    )
else:
    target_position, is_final_decision = pq3d_model.decision(
        color_list,
        depth_list,
        agent_state_list,
        frontier_waypoints,
        sentence,
        decision_num,
        task_level=sam_task_level,
    )
```

再新增 `goat-nav-sam2-runner.py`，同样 monkeypatch `data_utils.FastSAM = Sam2FastSAM`。

## 7. 环境复现流程

### 7.1 先建 SAM2 API smoke test 环境

不要直接在旧 `mtu3d` 环境里 `pip install sam2`。本地/历史线索显示项目环境可能存在 Python、torch、torchvision 版本偏旧的问题，而 SAM2 官方要求 Python >= 3.10、torch >= 2.5.1。

下面这个环境只用于确认 SAM2 官方 API、checkpoint、CUDA/BF16 能跑；它还不足以直接跑 RefHM3D/GOAT，因为导航脚本还依赖 Habitat、habitat-sim、MinkowskiEngine、torch_scatter、Open3D、transformers、本项目 `common/`、`data/`、`model/`、`configs/` 等完整栈。

在 Linux/WSL/远程 GPU 机器上，可先这样做 smoke test：

```bash
conda create -n mtu3d-sam2 python=3.10 -y
conda activate mtu3d-sam2

# cu124 只是示例；按 nvidia-smi、驱动和 PyTorch 官网选择 cu121/cu124 等匹配 wheel。
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/facebookresearch/sam2.git third_party/sam2
cd third_party/sam2
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

完整导航实验有两条可选路线：

1. **同环境路线**：在 `mtu3d-sam2` 里补齐 MTU3D/Habitat/PQ3D 全部依赖，并确认 Habitat 与 torch 2.5.1 兼容。这样 adapter 最简单，直接在进程内调用 SAM2。
2. **两进程路线**：如果原导航环境无法升级 Python/torch，则保留原 `mtu3d` 环境跑导航，另用 `mtu3d-sam2` 启动本地 SAM2 mask server。导航进程把 RGB 帧发给 server，server 返回 FastSAM-like masks 或序列化后的 mask/box/conf。这个方案工程量稍大，但依赖隔离最稳。

第一轮建议优先尝试同环境路线；一旦 Habitat 或 torch extension 冲突，就切两进程路线，不要硬升级旧环境。

### 7.2 下载 checkpoint

建议放在项目内：

```bash
mkdir -p hm3d-online/SAM2
wget -O hm3d-online/SAM2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

或者使用官方脚本：

```bash
cd third_party/sam2/checkpoints
./download_ckpts.sh
```

### 7.3 官方 API smoke test

不要从 `third_party` 的父目录直接运行 Python，SAM2 的 `build_sam.py` 会检查 package shadowing。建议在项目根目录或任意非 `sam2` 父目录运行：

```bash
python - <<'PY'
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

ckpt = "hm3d-online/SAM2/sam2.1_hiera_large.pt"
cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
gen = SAM2AutomaticMaskGenerator(model, points_per_side=8, points_per_batch=32)
image = np.zeros((480, 640, 3), dtype=np.uint8)

with torch.inference_mode():
    anns = gen.generate(image)
print("device=", device, "num_masks=", len(anns))
PY
```

### 7.4 wrapper smoke test

实现 `sam2_fast_wrapper.py` 后，直接把 `hm3d-online` 加到 `PYTHONPATH`：

```bash
export SAM2_CHECKPOINT=hm3d-online/SAM2/sam2.1_hiera_large.pt
export FASTSAM_WEIGHT=$SAM2_CHECKPOINT
export SAM2_LEVEL_PRESET=sanity
export SAM2_POINTS_PER_BATCH=32
PYTHONPATH=hm3d-online:$PYTHONPATH python - <<'PY'
import numpy as np
from sam2_fast_wrapper import Sam2FastSAM

model = Sam2FastSAM()
images = [np.zeros((480, 640, 3), dtype=np.uint8)]
out = model(images, task_level="object")
r = out[0]
print(r.masks.data.shape, r.boxes.data.shape, r.boxes.conf.shape)
PY
```

验收：

- 不报 import / hydra config / checkpoint 错。
- 输出 shape 合法。
- 空图可以返回 0 mask，但真实 RGB 帧应返回非空 masks。

## 8. RefHM3D 复现实验流程

### 8.1 阶段 A：单 episode 单任务

先跑最小切片：

```bash
export SAM2_CHECKPOINT=hm3d-online/SAM2/sam2.1_hiera_large.pt
export FASTSAM_WEIGHT=$SAM2_CHECKPOINT
export SAM2_LEVEL_PRESET=sanity
export SAM2_POINTS_PER_BATCH=32
export SAM2_USE_BF16=1

python hm3d-online/refhm3d-nav-sequence-sam2-runner.py \
  hm3d-online/refhm3d-nav-sequence-baseline.py \
  --start_ratio 0 \
  --end_ratio 0.001 \
  --task_levels object
```

如果已有 single-analysis 脚本，优先打开 `analysis_output_dir`，检查：

- `stage1` 每帧 proposal 数量。
- mask overlay 是否覆盖主体物体。
- `group_ids` 是否出现过度碎片化。
- `last_decision_aux` 是否为空或异常。

### 8.2 阶段 B：sanity profile 小切片

```bash
export SAM2_LEVEL_PRESET=sanity
export SAM2_POINTS_PER_BATCH=32

python hm3d-online/refhm3d-nav-sequence-sam2-runner.py \
  hm3d-online/refhm3d-nav-sequence-baseline.py \
  --start_ratio 0 \
  --end_ratio 0.02 \
  --task_levels object,room,region,instance
```

目标：

- 确认四类 task 都能跑。
- 记录 `avg_task_time_sec`。
- 记录 GPU peak memory。
- 无 OOM、无 `RuntimeError: decision failed`。

### 8.3 阶段 C：balanced N=700 对齐

对齐已有 FastSAM/SAM ViT-H N=700 结果：

```bash
export SAM2_LEVEL_PRESET=balanced
export SAM2_POINTS_PER_BATCH=64
export SAM2_USE_BF16=1

python hm3d-online/refhm3d-nav-sequence-sam2-runner.py \
  hm3d-online/refhm3d-nav-sequence-baseline.py \
  --start_ratio 0 \
  --end_ratio 0.2 \
  --task_levels object,room,region,instance
```

记录：

- SR / SPL 总体
- SR / SPL by task level
- avg_task_time_sec
- follower_error_count
- planner/filter adjusted count
- OOM 次数
- decision failed 次数

### 8.4 阶段 D：quality instance sweep

质量优先的重点 sweep：

```bash
export SAM2_LEVEL_PRESET=quality
export SAM2_POINTS_PER_BATCH=64

python hm3d-online/refhm3d-nav-sequence-sam2-runner.py \
  hm3d-online/refhm3d-nav-sequence-baseline.py \
  --start_ratio 0 \
  --end_ratio 0.2 \
  --task_levels instance
```

如果显存足够再试：

```bash
export SAM2_POINTS_PER_BATCH=96
export SAM2_LEVEL_PRESET=quality
```

如果 OOM：

```bash
export SAM2_POINTS_PER_BATCH=32
```

或者将 quality profile 中 instance 的 `crop_n_layers` 从 1 改回 0。

### 8.5 阶段 E：GOAT 替换实验

当前 `goat-nav.py` 的数据路径、split、输出路径是硬编码的，而且没有 `argparse/start_ratio/max_episodes`。因此不能直接声称“少量 episode”可控。建议先补三个环境变量入口：

```python
output_path = os.environ.get("GOAT_OUTPUT_PATH", output_path)
split_env = os.environ.get("GOAT_SPLITS")
if split_env:
    split_list = [x.strip() for x in split_env.split(",") if x.strip()]
max_episodes = int(os.environ.get("GOAT_MAX_EPISODES", "0"))
```

然后在 episode loop 中维护计数：

```python
episode_counter = 0
for split in split_list:
    for cur_data in data_set[split]:
        if max_episodes and episode_counter >= max_episodes:
            break
        episode_counter += 1
        ...
```

在 `goat-nav.py` 传入 `sam_task_level`，并新增 `goat-nav-sam2-runner.py` 后，再跑可控小样本：

```bash
export SAM2_CHECKPOINT=hm3d-online/SAM2/sam2.1_hiera_large.pt
export FASTSAM_WEIGHT=$SAM2_CHECKPOINT
export SAM2_LEVEL_PRESET=balanced
export GOAT_SPLITS=val_seen
export GOAT_MAX_EPISODES=5
export GOAT_OUTPUT_PATH=./output_dirs/goat-sam2-smoke.json

python hm3d-online/goat-nav-sam2-runner.py
```

重点比较：

- `object` 是否因更细 mask 反而变差。
- `description/image` 是否比 FastSAM 更稳定。
- 每次 decision 的 mask 数和 stage1 proposals 是否暴涨。

## 9. 消融矩阵

最小必要矩阵：

| 实验 | 模型 | profile | task level | 目的 |
|---|---|---|---|---|
| A0 | FastSAM | 原配置 | all | 原 baseline |
| A1 | SAM ViT-H | 现有 wrapper | all | 历史参照 |
| B0 | SAM2.1-L | sanity | all small slice | 接口验证 |
| B1 | SAM2.1-L | balanced | all N=700 | 主对比 |
| B2 | SAM2.1-L | quality | instance N=700 | 质量优先 |
| B3 | SAM2.1-L | balanced | object only | 检查 object 退化 |
| B4 | SAM2.1-L | quality no crop | instance | 分离 crop 收益 |
| B5 | SAM2.1-L | quality m2m off | instance | 分离 m2m 收益 |

`points_per_batch` sweep：

| points_per_batch | 预期 |
|---:|---|
| 32 | 最稳，慢一点，适合 OOM 排查 |
| 64 | 推荐默认 |
| 96 | 中高显存尝试 |
| 128 | 只在显存充足时尝试 |

## 10. 验收标准

### 10.1 工程验收

必须满足：

- `sam2_fast_wrapper.py` 输出结构完全兼容 `format_result()`。
- `refhm3d` runner 不修改 PQ3D 主逻辑即可运行。
- `goat-nav.py` 显式传 `task_level`。
- 单 episode 不出现 import、config、checkpoint、dtype、bbox shape 错误。
- 小切片四类 task 都能跑完。

### 10.2 质量验收

`balanced` profile 的建议通过线：

- 总体 SR 不低于 FastSAM。
- SPL 不低于 FastSAM。
- instance SR 明显高于 FastSAM。
- 如果 object SR 下降，下降幅度应被 instance/region/room 的收益抵消。

质量优先通过线：

- `quality` instance SR 高于 SAM ViT-H 或至少持平。
- N=700 不 OOM。
- avg task time 低于 SAM ViT-H 同类实验，或在质量明显更好时可接受。

### 10.3 速度验收

建议目标：

- `balanced` profile：目标是快于 SAM ViT-H；如果超过 FastSAM 2-3 倍，需要结合 SR/SPL 增益判断是否值得继续。
- `quality` profile：允许慢，但不应比 SAM ViT-H 更慢且无质量收益。
- `max_quality_offline` 不参与在线速度验收。

## 11. 常见风险与处理

### 11.1 依赖冲突

风险：旧 Habitat/MTU3D 环境可能是 Python 3.8 或 torch 2.2，而 SAM2 官方推荐 Python >= 3.10 和 torch >= 2.5.1。

处理：

- 首选新建 `mtu3d-sam2` 环境。
- 如果 Habitat 依赖无法迁移，再考虑两进程方案：导航主进程保留旧环境，SAM2 mask server 独立运行在新环境，通过本地 HTTP/ZeroMQ/文件 IPC 传 RGB 和 masks。
- 不要直接在旧环境 pip 装 SAM2，避免 torch/torchvision 被升级后破坏 Habitat。

### 11.2 Hydra config 找不到

错误类似：

```text
MissingConfigException: Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'
```

处理：

- 确认执行过 `pip install -e .`。
- 或设置：

```bash
export SAM2_REPO_ROOT=/path/to/sam2
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```

### 11.3 checkpoint 版本不匹配

错误类似：

```text
RuntimeError: Error(s) in loading state_dict for SAM2Base
```

处理：

- 卸载旧包：

```bash
pip uninstall -y SAM-2
```

- 拉最新 `facebookresearch/sam2` main。
- 重新 `pip install -e ".[notebooks]"`。

### 11.4 OOM

优先降级顺序：

1. `SAM2_POINTS_PER_BATCH=32`
2. `crop_n_layers=0`
3. `use_m2m=False`
4. `points_per_side` 降一档
5. `max_masks` 降一档
6. `output_mode` 保持 `binary_mask`，不要第一步改 RLE，因为当前 pipeline 直接吃 dense mask

### 11.5 mask 过多导致 PQ3D 退化

症状：

- `stage1` proposal 数量暴涨。
- object level 下降。
- 运行时间和显存异常。

处理：

- 对 object 降低 `max_masks`。
- 提高 object 的 `pred_iou_thresh/stability_score_thresh`。
- 保持 area 降序排序，让下游小 mask 后覆盖大 mask 的逻辑尽量稳定。
- 如果 instance 小物体召回明显变差，说明“按 area 降序后直接截断”可能丢掉小 mask；可增加一个 instance-only 策略：先按 area 降序维持下游覆盖顺序，但在截断前保留一部分高 `predicted_iou` 的小面积 mask。

## 12. 回滚方案

如果 SAM2 跑崩或收益不稳定：

1. unset SAM2 环境变量。
2. 直接使用原 FastSAM 入口脚本。
3. 保留 `sam2_fast_wrapper.py` 和 runner，不影响原 pipeline。
4. 如修改了 `goat-nav.py` 传 `task_level`，这对 FastSAM 是无害的，因为 FastSAM 忽略额外 keyword。

回滚命令示例：

```bash
unset SAM2_CHECKPOINT
unset SAM2_LEVEL_PRESET
unset SAM2_POINTS_PER_BATCH
unset FASTSAM_WEIGHT
python hm3d-online/refhm3d-nav-sequence-baseline.py
```

## 13. 推荐落地顺序

1. 新建 `sam2_fast_wrapper.py`。
2. 新增 `refhm3d-nav-sequence-sam2-runner.py`。
3. 官方 API smoke test。
4. 确认完整导航环境路线：同环境补齐依赖，或两进程 mask server。
5. wrapper smoke test。
6. `sanity` profile 单 episode。
7. `sanity` profile 小切片。
8. `balanced` profile N=700。
9. 修改 `goat-nav.py` 传 `task_level`，并加入 `GOAT_SPLITS/GOAT_MAX_EPISODES/GOAT_OUTPUT_PATH`。
10. 新增 `goat-nav-sam2-runner.py`。
11. `quality` profile 只跑 instance / description / image。
12. 根据 N=700 结果固化最佳 profile。

## 14. 最终建议

如果目标是“效果和速度最优，质量优先可牺牲速度”，我的建议是：

- **模型选 SAM2.1 Hiera-Large。**
- **接口选 image automatic mask generation，不先上 video predictor。**
- **第一轮正式结果用 `balanced` profile。**
- **第二轮质量冲刺只对 `instance/description/image` 开 `quality` profile。**
- **不要使用官方 automatic mask notebook 的 refined 64 点配置作为在线默认。**

原因是当前导航 pipeline 对 mask 数和运行时很敏感。SAM2.1-L 的质量上限值得用，但必须让 profile 跟任务粒度绑定。否则最可能出现的情况是：mask 更细、更漂亮，但 PQ3D 后端被过量 proposal 拖慢甚至 OOM，最后 SR/SPL 不稳定。

## 15. Agent 自检记录

已调用独立 explorer agent `Fermat` 对本文档进行严格自检，审查范围包括官方 SAM2 API、checkpoint/config、automatic mask generator 参数、本地 FastSAM-like 接口、runner 约定、GOAT 入口和速度/质量表述。

自检指出并已修正：

- 补齐 `_select_profiles()` 与 `SAM2_LEVEL_PROFILES`，避免 wrapper 草案不可运行。
- SAM2 runner 改为沿用现有 SAM runner 约定：`runner.py <target_script.py> [target args...]`，并补上 `sys.path` 与 `sys.argv` 处理。
- 环境流程拆成 “SAM2 API smoke test 环境” 与 “完整导航环境”，明确只安装 SAM2 不足以运行 Habitat/PQ3D 导航。
- GOAT 实验改为先补 `GOAT_SPLITS/GOAT_MAX_EPISODES/GOAT_OUTPUT_PATH` 控制入口，再跑小样本。
- `goal_image_feat` 的说明改为“位置参数脆弱”，不再误称会抢占 `analysis_output_dir`。
- 速度表述收窄为实验假设和验收目标，不把官方 39.5 FPS 当成本项目 AMG 实测。
- 补充 automatic generator 默认参数说明，注明表格不是完整参数表。

自检保留结论：

- SAM2.1 Hiera-Large checkpoint/config、官方指标和仓库 HEAD 核对无误。
- `SAM2AutomaticMaskGenerator.generate(image)` 返回字段可转换为当前 `.masks.data/.boxes.data/.boxes.conf`。
- RefHM3D baseline 已传 `task_level`，GOAT 需要补传 `task_level` 的判断正确。
- “官方 FPS 不能直接等价为在线 automatic all-mask generation 速度”的警告必须保留。

## 16. 参考来源

- Official SAM2 GitHub README: https://github.com/facebookresearch/sam2
- Official SAM2 INSTALL.md: https://raw.githubusercontent.com/facebookresearch/sam2/main/INSTALL.md
- Official `SAM2AutomaticMaskGenerator`: https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/automatic_mask_generator.py
- Official `build_sam.py`: https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/build_sam.py
- Official checkpoint download script: https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
- SAM2 paper: https://arxiv.org/abs/2408.00714
- 本地实验摘要：`key_logs/fastsam2sam/results_summary.md`
- 本地接入代码：`hm3d-online/data_utils.py`
- 本地 SAM wrapper：`hm3d-online/sam_fast_wrapper.py`
- GOAT 入口：`hm3d-online/goat-nav.py`
