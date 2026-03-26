# MTU3D 在 HM3D 上推理复现（仅推理 / evaluation）

本文档面向“**你已经有 conda 环境 `envname`，并且 `checkpoint/` 下已下载 MTU3D 相关权重**”的情况，目标是在 **HM3D 数据集上跑推理**（仓库默认提供三套基准：HM3D-OVON / GOAT-Bench / SG3D；它们都在 HM3D 场景上运行）。

---

## 0. 你当前已有的前提

- conda 环境：`envname`（Python 3.8，torch/torchvision 等已装好）
- 权重：MTU3D stage1 / stage2（fine-tune）已放在 `MTU3D/checkpoint/`（或你自己的路径）
- FastSAM 权重：需要 `hm3d-online/FastSAM/FastSAM-x.pt`（顶层 `README.md` 要求）

---

## 1. 一键下载清单（推理最小集合，明确直链与放置位置）

你只做推理（`hm3d-online/*-nav.py`），**最小必须下载 2 个东西**：

- **HM3D v0.2 的 `val`（glb）**：提供 `*.basis.glb` 场景文件
- **MTU3D 的 `embodied_bench_data.tar.gz`**：提供 OVON/SG3D/GOAT 推理用的 episode 与 `*_full_set.json`

> 我已经把 `hm3d-online/ovon-nav.py / sg3d-nav.py / goat-nav.py` 改成：**直接从 HM3D 的目录枚举 scene id**，因此不再需要下载巨大的 stage1 数据（`embodied_base.tar.gz.*`）来做 scene id 映射。

### 1.1 推荐目录布局（强烈建议照抄）

```text
/home/zhaochaoyang/datasets/
  hm3d/
    val/                           # 解压 hm3d-val-glb-v0.2.tar 到这里
      <scene_id>/
        <scene_short_id>.basis.glb
  mtu3d/
    embodied_bench/                # 解压 embodied_bench_data.tar.gz 到这里
      our-set/
        ovon_full_set.json
        goat_full_set.json
        sg3d_full_set.json
      ovon/
      goat/
      sg3d/
```

### 1.2 一键下载 + 解压命令

在任意目录执行（建议用 `tmux`/后台，因为 HM3D 约 4GB）：

```bash
set -e

# 1) HM3D val glb（必须）
mkdir -p /home/zhaochaoyang/datasets/hm3d
cd /home/zhaochaoyang/datasets/hm3d
wget -c "https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar" -O hm3d-val-glb-v0.2.tar
mkdir -p val
tar -xf hm3d-val-glb-v0.2.tar -C val

# 2) MTU3D embodied_bench_data（必须）
mkdir -p /home/zhaochaoyang/datasets/mtu3d
cd /home/zhaochaoyang/datasets/mtu3d
wget -c "https://huggingface.co/datasets/bigai/MTU3D/resolve/main/embodied_bench_data.tar.gz" -O embodied_bench_data.tar.gz
mkdir -p embodied_bench
tar -xzf embodied_bench_data.tar.gz -C embodied_bench
```

### 1.3 你已说“权重已下载”，这里只给你权重来源链接（不强制下载）

- **MTU3D checkpoints**：`https://huggingface.co/bigai/MTU3D`
- **FastSAM-x.pt**：按项目 README 放到 `hm3d-online/FastSAM/FastSAM-x.pt`（代码里写死这个路径）

---

## 2. 推理前你只需要改哪些路径（最少 3 行）

### 2.1 修改 `hm3d-online/ovon-nav.py`（或你要跑的脚本）

以 `hm3d-online/ovon-nav.py` 顶部超参为例，你需要把下面这些路径改成你自己的：

- `data_set_path`
- `navigation_data_path`
- `hm3d_data_base_path`（指向 HM3D `val` 根目录，例如 `/home/zhaochaoyang/datasets/hm3d/val`）
- `pq3d_stage1_path`、`pq3d_stage2_path`（指向你下载的 checkpoint 目录，目录内需存在 `pytorch_model.bin`）

同理：
- OVON：改 `hm3d-online/ovon-nav.py`
- SG3D：改 `hm3d-online/sg3d-nav.py`
- GOAT：改 `hm3d-online/goat-nav.py`（外加 `image_feat_dir`）

> 备注：推理脚本使用的是 `common/embodied_utils/simulator.py` 里直接创建的 Habitat-Sim 配置，不依赖 `scene_dataset_config_file`，因此这里不再要求你改 `configs/habitat/goat_sim_config.yaml`。

---

## 3. 权重文件应该放哪里

### 3.1 FastSAM

`hm3d-online/data_utils.py` 中写死了：

- `FastSAM('./hm3d-online/FastSAM/FastSAM-x.pt')`

所以你必须保证这个文件存在：

- `hm3d-online/FastSAM/FastSAM-x.pt`

### 3.2 MTU3D stage1 / stage2

`PQ3DModel(stage1_dir, stage2_dir)` 会加载：

- `${stage1_dir}/pytorch_model.bin`
- `${stage2_dir}/pytorch_model.bin`

因此你在 `*-nav.py` 里填的 `pq3d_stage1_path / pq3d_stage2_path` 必须是**目录**，且目录内有对应的 `pytorch_model.bin`。

---

## 4. 运行（推理 / evaluation）

在 `MTU3D` 根目录执行：

```bash
conda activate envname
cd /home/zhaochaoyang/yuantingyu/3DShape2vecset/data/out/MTU3D

mkdir -p output_dirs
export PYTHONPATH=./:./hm3d-online:./hm3d-online/FastSAM
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
```

### 4.1 在 HM3D-OVON 上推理

```bash
python3 hm3d-online/ovon-nav.py
```

默认会把结果写到 `ovon-nav.py` 里设置的 `output_path`（例如 `./output_dirs/ovon-full-finetune-num-1.json`）。

### 4.2 在 SG3D 上推理

```bash
python3 hm3d-online/sg3d-nav.py
```

### 4.3 在 GOAT-Bench 上推理（可选）

```bash
python3 hm3d-online/goat-nav.py
```

若你未准备 `goat-clip-feat`，这里会在读取 `image_feat_dir` 时失败；只跑 OVON/SG3D 可跳过。

---

## 5. 常见报错快速定位

- **找不到 `FastSAM-x.pt`**：确认文件在 `hm3d-online/FastSAM/FastSAM-x.pt`（路径在代码里写死）。
- **找不到 `pytorch_model.bin`**：确认 `pq3d_stage1_path` / `pq3d_stage2_path` 指向的是“目录”，且目录内有 `pytorch_model.bin`。
- **scene 路径拼不出来 / 找不到 `*.basis.glb`**：确认 `hm3d_data_base_path` 指向 HM3D `val` 根目录，且目录结构满足脚本拼接规则。

