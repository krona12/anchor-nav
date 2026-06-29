# SAM2.1 Vista2MQSC 替换规划

日期：2026-06-25

范围只包含 ours 方法 `vista2mqsc`，目标底层脚本为：

```text
hm3d-online/refhm3d-nav-sequence-analyze-anchor-vista2mqsc-sequence-refine1.py
```

`prompt-md/sam2-substitute.md` 中的 GOAT 内容本轮忽略，不改 GOAT。

## 目标

用 SAM2.1 Hiera-Large 替换当前 FAST-SAM/SAM 分割组件，保持 PQ3D、MQSC-R1、VISTA-LS 和导航主流程不变，跑出 `0.0_0.2` 区间两套阈值结果，并与原 SAM 结果比较。

## 替换原则

- 不改 `data_utils.PQ3DModel` 主流程。
- 不改 PQ3D stage1/stage2、`RepresentationManager`、MQSC-R1、VISTA-LS。
- 采用与 `hm3d-online/sam_fast_wrapper.py` 相同的 adapter/runner 方式：
  - `Sam2FastSAM` 伪装成 FastSAM-like callable。
  - runner monkeypatch `data_utils.FastSAM = Sam2FastSAM`。
  - 下游继续使用现有 `format_result()`。
- SAM2 输出转换：
  - `segmentation` -> `result.masks.data`，shape 为 `[N,H,W]`。
  - `bbox` 从 XYWH 转为 XYXY -> `result.boxes.data`。
  - `predicted_iou` -> `result.boxes.conf`，缺失时 fallback 到 `stability_score`。

## 文件改动计划

新增：

```text
hm3d-online/sam2_fast_wrapper.py
hm3d-online/refhm3d-nav-sequence-sam2-runner.py
scripts/sam2/00_02/run_vista2mqsc_sam2_all_0.0_0.2.sh
scripts/sam2/00_02/tmux_run_vista2mqsc_sam2_balanced_quality_0.0_0.2.sh
prompt-md/sam2-vista2mqsc-plan.md
```

最小修改：

```text
hm3d-online/refhm3d-nav-sequence-analyze-anchor-vista2mqsc-sequence-refine1.py
```

修改点只有一处：`pq3d_model.decision(...)` 显式传入 `task_level=task_type`，使 `object/room/region/instance` 对应的 SAM2 阈值 profile 生效。

保留：

```text
hm3d-online/data_utils.py
hm3d-online/sam_fast_wrapper.py
hm3d-online/refhm3d-nav-sequence-sam-runner.py
hm3d-online/goat-nav.py
```

## 阈值体系

两套正式阈值来自 `prompt-md/sam2-substitute.md`。

### balanced

主对比配置，目标是控制速度和显存，同时保留 SAM2.1 质量收益。

| level | points_per_side | pred_iou | stability | crop | crop_downscale | use_m2m | max_masks | min_area |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| object | 12 | 0.80 | 0.88 | 0 | 1 | False | 32 | 200 |
| room | 16 | 0.80 | 0.88 | 0 | 1 | False | 40 | 150 |
| region | 20 | 0.78 | 0.86 | 0 | 1 | False | 52 | 100 |
| instance | 24 | 0.76 | 0.84 | 0 | 1 | False | 64 | 50 |

依据：

- `object` 保守，避免过多细碎 mask 干扰类别级导航。
- `instance` 更高召回，但不打开 crop/m2m，先保证 `0.0_0.2` 全量可跑。
- 对标历史 SAM ViT-H 全任务实验。

### quality

质量优先配置，重点观察 `instance` 收益和整体 SR/SPL 是否值得速度代价。

| level | points_per_side | pred_iou | stability | crop | crop_downscale | use_m2m | max_masks | min_area |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| object | 16 | 0.80 | 0.88 | 0 | 1 | False | 36 | 200 |
| room | 20 | 0.78 | 0.86 | 0 | 1 | False | 44 | 150 |
| region | 28 | 0.76 | 0.84 | 0 | 1 | False | 60 | 100 |
| instance | 32 | 0.72 | 0.82 | 1 | 2 | True | 72 | 50 |

依据：

- `instance` 使用更密采样、crop 和 m2m refinement，追求小物体和相邻实例质量。
- OOM 降级顺序：`SAM2_POINTS_PER_BATCH=32`，然后关闭 `instance.crop_n_layers`，再关闭 `use_m2m`。

## 运行方式

单配置：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/sam2/00_02/run_vista2mqsc_sam2_all_0.0_0.2.sh balanced detailed
CUDA_VISIBLE_DEVICES=2 bash scripts/sam2/00_02/run_vista2mqsc_sam2_all_0.0_0.2.sh quality detailed
```

双配置 tmux：

```bash
bash scripts/sam2/00_02/tmux_run_vista2mqsc_sam2_balanced_quality_0.0_0.2.sh
```

输出：

```text
output_logs/anchor/vista2mqsc_sam2_balanced_all_0.0_0.2/
output_logs/anchor/vista2mqsc_sam2_quality_all_0.0_0.2/
key_logs/vista2mqsc_sam2_0.0_0.2_<timestamp>/
```

## 环境现状与 blocker

当前 `mtu3d` 实测：

```text
Python 3.8.20
torch 2.1.2+cu118
sam2: not installed
segment_anything: installed
```

本机当前命令环境还显示 `nvidia-smi` 无法连接 NVIDIA driver。

`prompt-md/sam2-substitute.md` 和 agent 核查均显示官方 SAM2.1 当前要求：

```text
Python >= 3.10
torch >= 2.5.1
torchvision >= 0.20.1
```

因此在“不升级 Python、不跨环境、不破坏 mtu3d”的硬约束同时成立时，不能合规安装官方 SAM2.1。不得使用 `--ignore-requires-python`、手改 setup 或强行升级 torch/numpy 的方式绕过。

需要解除 blocker 后才能执行：

1. `sam2` import smoke test。
2. `Sam2FastSAM` 单图推理。
3. `sanity` 小切片。
4. `balanced` 与 `quality` 两套 `0.0_0.2` tmux 实验。
5. 与原 SAM 结果对比并选参。
