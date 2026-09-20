# MTU3D · LangMap 复现与交接

本分支将原 anchor-nav 项目精简为 [MTU3D](https://github.com/MTU3D/MTU3D) 的 LangMap / RefHM3D 序列导航评测，并保留两个分析模板。`refhm3d-nav-sequence-baseline.py` 是本项目的 LangMap 适配入口；官方 MTU3D 仓库本身没有这个 benchmark 入口。

**交接状态（2026-09-20）：环境、模型、36 个场景及短序列推理已实测；完整 3,600 子任务评测尚未完成。** 第一次完整运行在共享 GPU 上因 CUDA OOM 退出，已完成 60 个子任务。不要将短测试通过或文件下载完成解释成完整指标已复现。

按下面的 1 → 7 顺序执行即可重建。公开权重和示例场景提供一条下载命令；完整 HM3D 必须先由接手者取得 Matterport 授权，无法通过脚本绕过。仓库不携带模型、HM3D、Conda 环境、第三方克隆或历史日志。

## 1. 机器与仓库

实测机器：Ubuntu 22.04.5 LTS、Linux x86_64、RTX 4090 24 GB、NVIDIA 驱动 595.58.03。此版本使用 Python 3.8.20、PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118、Habitat-Sim/Lab 0.2.3、MinkowskiEngine 0.5.4。其他 GPU/系统需要自行验证；尤其不能把这套 CUDA 11.8 配方直接用于需要更新 CUDA 的 GPU。

准备 Conda、Git、curl、GCC/G++ 11 和可用的 NVIDIA 驱动/EGL。建议预留至少 40 GB 磁盘给环境、编译缓存和数据；这是容量建议，并非测得的最低要求。正式评测使用空闲 GPU，不要与其他训练任务共享显存。

```bash
# Ubuntu 22.04；已有依赖可以跳过这一条。
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential gcc-11 g++-11 \
  libegl1 libopengl0 libgl1 libglib2.0-0 libgomp1
nvidia-smi
command -v conda

# 本分支发布到 GitHub 后，只克隆当前快照，避免下载原分支的大量历史日志。
git clone --depth 1 --single-branch \
  --branch repro/mtu3d-langmap-handoff-20260920 \
  https://github.com/krona12/anchor-nav.git
cd anchor-nav
export MTU3D_ROOT="$PWD"
```

没有 Conda 时，先按 [Miniconda 官方安装说明](https://www.anaconda.com/docs/getting-started/miniconda/install) 安装并初始化 shell。`libegl1` 不等同于 NVIDIA EGL 驱动；主机/容器必须能访问 GPU，并具有与驱动匹配的 NVIDIA 用户态图形库。本分支不自动升级驱动。

之后所有命令均在仓库根目录、同一个 Bash 会话执行；新开终端时重新 `conda activate envname`、`cd` 仓库，并设置 `MTU3D_ROOT="$PWD"`。不需要使用原机器的 `/home/ubuntu/krona/anchor-nav` 路径。

## 2. 创建固定版本运行环境

下面清单是实测环境的精确 Conda 包 URL + MD5；只适用于 Linux x86_64。Conda 如果要求确认渠道条款，请按所用渠道的要求处理。

```bash
# 这是新建环境命令；已有同名环境时换名字，不要直接删除正在使用的环境。
conda create -n envname --file environment/conda-runtime-linux64.txt -y
conda activate envname
python -m pip install pip==24.2 setuptools==75.1.0 wheel==0.44.0
python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch_scatter==2.1.2+pt20cu118 \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

mkdir -p third_party
git clone https://github.com/NVIDIA/MinkowskiEngine.git third_party/MinkowskiEngine
git -C third_party/MinkowskiEngine checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
git clone https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
git -C third_party/habitat-lab checkout 2c7519b1baead8d5bc9557dd53d2a01e4aa9c6e5
git clone https://github.com/openai/CLIP.git third_party/CLIP
git -C third_party/CLIP checkout a1d071733d7111c9c014f024669f959182114e33
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git hm3d-online/FastSAM
git -C hm3d-online/FastSAM checkout b4ed20c2fed75eadc5aa7d8b09fedd137b873b52

python -m pip install -r requirements.txt -c environment/constraints-py38.txt
python -m pip install --no-deps \
  -e third_party/habitat-lab/habitat-lab \
  -e third_party/habitat-lab/habitat-baselines
```

`environment/constraints-py38.txt` 固定 189 个包版本，`requirements.txt` 指定实际直接依赖。必须先安装 Torch，再安装这些依赖；CLIP 和 Habitat 使用 editable 安装，因此运行时也要保留对应源码目录。重试某一步时，从失败的命令继续；已存在的第三方目录不要再次 `git clone`，先检查 `git -C <目录> rev-parse HEAD` 是否等于上面的提交。

不要另装最新版 `ultralytics`，模板会使用固定 FastSAM 克隆内的版本。不要再次执行上游未经固定版本的全部 requirements，它可能升级或冲掉兼容组合。Torch cu118 wheel 已带 CUDA runtime / cuDNN，不需要额外执行上游 README 中的 `nvidia-cudnn-cu11` 安装。

## 3. 编译 MinkowskiEngine

系统 `nvcc` 在原机器上是 CUDA 13，与此 PyTorch 不匹配。单独创建 **CUDA 11.8 编译环境**，继续使用运行环境的 Python，不要激活编译环境。

```bash
MTU3D_CUDA_PREFIX="$(dirname "$CONDA_PREFIX")/mtu3d-build"
conda create -p "$MTU3D_CUDA_PREFIX" \
  --file environment/conda-cuda118-linux64.txt -y

# 子 shell 避免编译变量污染后续运行。
(
  export CUDA_HOME="$MTU3D_CUDA_PREFIX"
  export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
  export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
  # 8.9 是 RTX 4090；其他 GPU 应改为其受 CUDA 11.8 支持的计算能力。
  export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=4
  export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export LDFLAGS="-Wl,-rpath,$CUDA_HOME/lib -L$CUDA_HOME/lib"
  "$CUDA_HOME/bin/nvcc" --version
  cd third_party/MinkowskiEngine
  python setup.py bdist_wheel --force_cuda --cuda_home="$CUDA_HOME" \
    --blas=openblas --blas_include_dirs="$CUDA_HOME/include"
  python -m pip install --no-deps dist/MinkowskiEngine-*.whl
)
python -m pip check
```

编译清单同时带有 OpenBLAS 开发库。`MAX_JOBS=4` 用于控制编译内存；CPU 内存不足时降为 1。**不要在装好后删除 `mtu3d-build`**：编译产物通过 RPATH 引用它的动态库。运行环境和编译环境改变安装路径后，应在新位置重新编译，不能直接拷贝原机器的 `.so` / wheel。重跑 ME 的 `setup.py` 会清理 build 并卸载已有 ME，诊断时不要随手重复安装。

验证实际 CUDA 运算：

```bash
OMP_NUM_THREADS=4 python - <<'PY'
import torch, torch_scatter, MinkowskiEngine as ME
assert torch.cuda.is_available()
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
v = torch.tensor([1., 2., 3.], device='cuda')
i = torch.tensor([0, 0, 1], device='cuda')
assert torch.equal(torch_scatter.scatter_add(v, i), torch.tensor([3., 3.], device='cuda'))
c = ME.utils.batched_coordinates([torch.tensor([[0,0,0],[1,0,0],[2,0,0]], dtype=torch.int32)])
x = ME.SparseTensor(torch.ones(3, 4, device='cuda'), coordinates=c, device='cuda')
conv = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).cuda().eval()
with torch.no_grad():
    y = conv(x)
assert y.F.shape == (3, 8) and torch.isfinite(y.F).all()
print('CUDA / torch_scatter / MinkowskiEngine PASS')
PY
```

## 4. 一条命令下载公开模型与示例场景

```bash
python environment/download_assets.py --example-scene
```

该命令使用 [assets-manifest.json](environment/assets-manifest.json) 中固定 revision 的官方 URL，只下载所需的 **14 个模型文件（4,140,465,091 bytes）**，逐个校验长度和 SHA256；同时准备无需完整数据授权的官方示例场景 `00861-GLAQ4DNUx5U`。有效文件会跳过，未完成下载保留 `.part` 供重试。发现已有文件损坏时会报出路径，不会悄悄覆盖；按报错说明处理该文件后再执行同一命令。

| 资源 | 相对仓库根目录的位置 |
| --- | --- |
| LangMap annotations，36 场景 / 720 序列 / 3,600 子任务 | `LangMap_Annotations/`，Git 已携带 |
| MTU3D stage1-pretrain-all | `checkpoint/stage1-pretrain-all/pytorch_model.bin` |
| MTU3D stage2-fine-tune-goat | `checkpoint/stage2-fine-tune-goat/pytorch_model.bin` |
| DINOv2 large | `checkpoint/dinov2-large/` |
| CLIP ViT-L/14 | `checkpoint/clip-vit-large-patch14/` |
| FastSAM-x | `hm3d-online/FastSAM/FastSAM-x.pt` |
| HM3D 场景 | `datascene/<完整场景 ID>/<短 ID>.basis.glb` |

不带 `--example-scene` 时只处理模型和标注校验；`--verify-only` 禁止下载/写入，适合搬迁后检查。所有路径都是可移植的相对路径，模型目录不依赖另一台机器的 Hugging Face 缓存。原开发机恰好使用缓存符号链接，不需要仿照它建立绝对路径链接。

下载器的 22 项本地集成测试也随分支保留，覆盖断点续传、损坏标注/权重、只读校验和安全提取；不需要 GPU 或外网：

```bash
python -m unittest discover -s environment -p 'test_*.py'
```

需要访问 GitHub release、Hugging Face 及其文件 CDN；若网络返回 401/403、超时或 TLS 错误，先检查授权、代理和出口。不要把 HTML 错误页改名当权重，也不要取消校验。脚本不会下载重复的 `.bin`/TensorFlow/Flax 格式。CLIP checkpoint 包含视觉塔，但本项目只调用文本编码器；保留原始 checkpoint 格式以兼容官方加载方式。

**本推理链不需要** SceneVerse、`embodied_base`、`embodied_scan_stage2_feat`、`embodied_scan_vle_data`、其他 benchmark episode 包、GOAT 图像目标特征、HM3D train、OBJ/MTL 或 semantic annotations。当前模拟器读取 RGB/深度并重建 navmesh。

来源：[MTU3D 权重](https://huggingface.co/bigai/MTU3D)、[DINOv2](https://huggingface.co/facebook/dinov2-large)、[CLIP](https://huggingface.co/openai/clip-vit-large-patch14)、[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)。

## 5. 完整 HM3D 授权、下载和提取

只有运行全部 36 场景时需要本节。尚无权限也可以先执行第 6 节的公开场景测试。

1. 按 [Matterport 官方数据说明](https://github.com/matterport/habitat-matterport-3dresearch) 注册/登录账号，并从 [Matterport / Meta 页面](https://matterport.com/partners/meta) 申请 HM3D 研究数据权限。
2. 审核通过后，在账号 Settings → Developer Tools 的 Habitat-Matterport 数据区域取得 token ID 和 secret；入口为 [Developer Tools](https://my.matterport.com/settings/account/devtools)。界面可能调整，以官方页面为准。
3. 只下载 `hm3d-val-habitat-v0.2.tar`，不是另一个 `glb` 包，也不是训练集。Habitat 下载器的最小 UID 是 `hm3d_val_habitat_v0.2`，不要选会额外拉取无关数据的整个资源组。

在自己的终端执行下列命令。curl 会询问 secret，不要将 secret 写进命令、README、Git 或聊天。

```bash
mkdir -p "$HOME/Downloads/hm3d"
read -r -p "Matterport token ID: " HM3D_TOKEN_ID
curl --fail --location --retry 5 --continue-at - --user "$HM3D_TOKEN_ID" \
  --output "$HOME/Downloads/hm3d/hm3d-val-habitat-v0.2.tar" \
  https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
unset HM3D_TOKEN_ID

# curl/传输进程退出成功之后再执行；不要解压仍在传输的文件。
python environment/download_assets.py \
  --hm3d-archive "$HOME/Downloads/hm3d/hm3d-val-habitat-v0.2.tar" \
  --require-scenes
python environment/download_assets.py --verify-only --require-scenes
```

已有授权数据包时直接执行最后两条命令；路径可替换为传输到本机的实际路径。脚本只提取 LangMap 的 36 个 `.basis.glb`，按清单逐文件校验，保留输入 tar。若直接使用完整解压目录，运行入口的 `--hm3d_data_base_path` 应指向**直接包含 36 个场景子目录**的目录。

原开发机收到的 tar 为 3,530,618,880 bytes，SHA256 `04c97761cb16ed8bd6f6600d4211ab10b9d3649d981401b527f0c0264a60371b`，含 100 个 GLB，所需 36 个 mesh 合计 1,335,637,424 bytes。这个 tar 哈希和 mesh 哈希来自本机实测，不应说成官方发布的整包签名；公开示例包另有官方校验值。仅看文件名、大小一段时间不变化，不能证明跨机器传输完成：应检查发送端/接收端退出码，并做发送端与接收端 SHA256 比对。

## 6. 离线启动与短序列验证

模型下载到项目目录后，明确设置本地加载路径。每次新开终端运行都执行以下环境设置：

```bash
conda activate envname
export MTU3D_ROOT="$PWD"
export MTU3D_PYTHON="$CONDA_PREFIX/bin/python"
export DINOV2_MODEL="$MTU3D_ROOT/checkpoint/dinov2-large"
export CLIP_MODEL="$MTU3D_ROOT/checkpoint/clip-vit-large-patch14"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/run_template.sh --help
```

模板已设置项目/FastSAM 导入路径、线程数和 NVIDIA EGL vendor。若指定了不同环境，可用 `MTU3D_ENV=环境名`；`MTU3D_PYTHON` 优先。`FASTSAM_WEIGHT` 可覆盖 FastSAM 路径。baseline 不需要任何 VLM API key。

下面从公开示例的原始标注复制**第一条完整序列的 5 个子任务**，不修改仓库标注。`SMOKE_DIR` 每次新建，避免旧结果让测试直接跳过。

```bash
export SMOKE_DIR="$(mktemp -d "$MTU3D_ROOT/.cache-smoke-XXXXXXXX")"
python - <<'PY'
import gzip, json, os
from pathlib import Path
root = Path(os.environ['MTU3D_ROOT'])
scene = '00861-GLAQ4DNUx5U.json.gz'
with gzip.open(root / 'LangMap_Annotations' / scene, 'rt') as f:
    data = json.load(f)
data['episode_by_sequence'] = data['episode_by_sequence'][:1]
assert len(data['episode_by_sequence'][0]['task_sequence']) == 5
out = Path(os.environ['SMOKE_DIR']) / 'annotations'
out.mkdir()
with gzip.open(out / scene, 'wt') as f:
    json.dump(data, f)
PY
bash scripts/run_template.sh \
  --navigation_data_path "$SMOKE_DIR/annotations" \
  --start_ratio 0.0 --end_ratio 1.0 --seed 1234 \
  --output_log_dir "$SMOKE_DIR/results"
python - <<'PY'
import json, math, os
from pathlib import Path
p = Path(os.environ['SMOKE_DIR']) / 'results/refhm3d_seq_0.0_1.0.json'
rows = json.loads(p.read_text())['sequence']
assert len(rows) == 5
assert len({(r['scene_name'], r['episode_id'], r['task_id']) for r in rows}) == 5
assert all(math.isfinite(float(r[k])) for r in rows for k in ['sr', 'spl'])
assert all(r.get('end_reason') != 'follower_error' for r in rows)
print('PASS: 5-task navigation sequence', p)
PY
```

此测试实际加载全部模型、渲染场景、重建 navmesh 并执行导航，不能只用 `--help` 代替。SR 不要求等于 1：模型导航失败与脚本异常是不同问题。不同 GPU、稀疏 CUDA 算子和库实现可能使轨迹/指标变化；固定 seed 不保证跨机器逐位相同。

## 7. 完整评测、监控和验收

先确认第 5 节的 `--require-scenes` 校验通过，并用 `nvidia-smi` 确认所选 GPU 没有其他训练任务。以下在 Bash 中运行，`RUN_DIR` 每次使用新目录；前台运行需要保留终端，也可放入自行管理的 tmux 会话。

```bash
export RUN_DIR="$MTU3D_ROOT/output_logs/baseline/langmap_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
# 单独进程记录整卡及每个计算进程显存，不从推理线程调用监控。
(
  while true; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
    sleep 5
  done
) > "$RUN_DIR/gpu-memory.log" 2>&1 &
MTU3D_MONITOR_PID=$!
trap 'kill "$MTU3D_MONITOR_PID" 2>/dev/null || true' EXIT

# tee 不能掩盖 Python 失败；记录真实退出码。
set +e
bash scripts/run_template.sh \
  --navigation_data_path "$MTU3D_ROOT/LangMap_Annotations" \
  --hm3d_data_base_path "$MTU3D_ROOT/datascene" \
  --task_levels object,room,region,instance \
  --start_ratio 0.0 --end_ratio 1.0 --seed 1234 \
  --output_log_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"
MTU3D_EXIT_CODE=${PIPESTATUS[0]}
printf '%s\n' "$MTU3D_EXIT_CODE" > "$RUN_DIR/exit_code.txt"
kill "$MTU3D_MONITOR_PID" 2>/dev/null || true
wait "$MTU3D_MONITOR_PID" 2>/dev/null || true
trap - EXIT
test "$MTU3D_EXIT_CODE" -eq 0
```

默认使用 detailed descriptions。`--concise_description` 是另一个实验设置，不能混合汇总。场景比例对按文件名排序后的标注文件生效；任务级别必须包含全部四种，否则被跳过的任务也会改变后续任务起点。第一次历史运行用过的 `/tmp` 监督脚本不属于本仓库交接依赖；按本节命令即可记录运行和退出码。

检查退出码为 0 后，再检查完整性；日志出现 “run finished” 本身不保证成功：

```bash
python - <<'PY'
import collections, gzip, json, math, os
from pathlib import Path
root, out = Path(os.environ['MTU3D_ROOT']), Path(os.environ['RUN_DIR'])
assert (out / 'exit_code.txt').read_text().strip() == '0', 'Process failed'
expected = set()
for p in sorted((root / 'LangMap_Annotations').glob('*.json.gz')):
    with gzip.open(p, 'rt') as f:
        data = json.load(f)
    scene = p.name[:-len('.json.gz')]
    for e in data['episode_by_sequence']:
        for i, task in enumerate(e['task_sequence']):
            expected.add((scene, e['episode_id'], i))
rows = json.loads((out / 'refhm3d_seq_0.0_1.0.json').read_text())['sequence']
actual = [(r['scene_name'], r['episode_id'], r['task_id']) for r in rows]
assert len(rows) == len(set(actual)) == len(expected) == 3600
assert set(actual) == expected, 'Missing or unexpected tasks'
assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for r in rows for k in ['sr', 'spl'])
levels = collections.Counter(r['task_level'] for r in rows)
assert dict(levels) == {'object': 841, 'room': 917, 'region': 1040, 'instance': 802}
print('Complete:', len(rows), 'SR:', sum(float(r['sr']) for r in rows)/len(rows),
      'SPL:', sum(float(r['spl']) for r in rows)/len(rows))
print('Follower errors:', sum(r.get('end_reason') == 'follower_error' for r in rows))
PY
```

**续跑限制：** 当前入口按“序列 ID”跳过已有结果，但每个子任务都会保存。若一条序列只完成部分任务，复用旧输出目录会把这条序列余下任务也跳过。为了正确复现，失败后保留旧目录并使用全新输出目录重新执行；不要直接拼接部分结果、删除失败记录或将 60 条结果冒充 3,600 条。序列中的起点、记忆及 RNG 状态有关联，单独重跑某个失败任务只用于诊断，不等于原完整轨迹的精确回放。

## 常见问题与处理顺序

| 症状 / 坑 | 已验证的原因与处理 |
| --- | --- |
| `No module named ...` 或 clone 后缺依赖目录 | Git 忽略第三方克隆；完整执行第 2 节，保留 editable 源码，使用模板启动。 |
| ME 报 CUDA version mismatch / nvcc 13 | Torch 是 cu118；第 3 节单独的 CUDA 11.8 工具链、GCC 11、`CUDA_HOME` 必须一致。`nvidia-smi` 显示的 CUDA 版本是驱动能力，不是选中的编译器版本。 |
| ME 编译 `Killed` | 检查 CPU RAM，将 `MAX_JOBS` 降为 1；不代表 GPU OOM。 |
| `libcusparse.so.11` / OpenBLAS 找不到 | 检查 `mtu3d-build` 是否还在、前缀是否移动、RPATH 是否正确；在新位置重新编译。 |
| `unable to find EGL device for CUDA device 0` | 原机 Conda base 把 EGL 限制到 Mesa。模板在没有显式覆盖时选择 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`；检查此文件、匹配的 NVIDIA 图形库及容器 GPU/EGL 映射。不要靠重装模型处理。 |
| 无显示服务器 / DISPLAY 为空 | 使用清单中的 headless Habitat-Sim；仍需要 NVIDIA EGL，无须桌面 X server。 |
| 下载包正确但场景找不到 | 检查 v0.2 **val habitat** 包、`basis.glb` 后缀和场景层级；`--hm3d_data_base_path` 不能多一层 `val` 或少一层。 |
| Hugging Face 离线模式仍缺文件 | 先完成下载校验，再设置 `DINOV2_MODEL` / `CLIP_MODEL` 为本地绝对目录。仅有 safetensors，没有 config/tokenizer/preprocessor 也不够。 |
| Python 3.8 依赖解不出来 | 不升级到各包最新版；使用 constraints。重点是 NumPy 1.23.5 + Numba 0.56.4、HF hub 0.23.5 + sentence-transformers 2.2.2、`hydra-core`、`volumentations==0.1.9`、`lmdb==1.4.1`。 |
| CUDA OOM | 先保存错误和同期整卡/各进程显存，再检查 GPU 共享。短测试余量不能代表长评测峰值；不要凭 OOM 通用提示就认定碎片或泄漏，也不要静默降低点数/分辨率改变评测。 |
| 重跑瞬间结束 / 结果少于 3,600 | 检查是否复用了有结果的目录和上述序列级续跑限制。 |
| 分析模板要求 API key | 两个模板使用 VLM，须单独配置 `ZZZ_API_KEY`；baseline 无须 VLM，不能把 API key 写进源码。 |

## 保留范围与验证证据

`hm3d-online/` 下 RefHM3D 入口只有这四个：

- `refhm3d-nav-sequence-baseline.py`：交接默认入口。
- `refhm3d-nav-sequence.py`：原始 LangMap 适配，保留原有任务黑名单。
- `refhm3d-nav-sequence-analyze-anchor-vista2mqsc-refine1.py`：分析模板。
- `refhm3d-nav-sequence-analyze-anchor-vista2mqsc-sequence-refine1.py`：序列分析模板。

模板依赖保留的 `anchor_nav/mqsc_r1.py`、`anchor_nav/vista_ls.py` 和 `vlm/client.py`；通过 `ENTRYPOINT=文件名 bash scripts/run_template.sh ...` 切换。`scripts/` 只留一个运行模板，环境与资源辅助文件放在 `environment/`。`run_nav.sh` 转发到模板；`launch.py` / `run.py` 是保留的官方训练入口，不用于本次 LangMap baseline 推理。

官方核心目录 `common/configs/data/evaluator/model/modules/optim/trainer` 对应 MTU3D 提交 `c10335ce8effd881e481a5f669add7d040aec947`；OVON / GOAT / SG3D 入口保留参考。点云处理、对象合并和目标选择已与官方核对，共享工具保留了分析模板所需的诊断字段与 RGB 快照，并非所有文件逐字回滚。stage2 使用 `stage2-fine-tune-goat`，不是重新训练的 LangMap checkpoint。

已完成的运行证据：

- `pip check`，Torch / torch_scatter / ME CUDA 运算、FastSAM 推理，以及项目配置的 Habitat RGB/深度渲染和 navmesh 检查通过。
- 四个入口的 `--help` 和保留 Python 文件语法检查通过。
- 公开场景第 0 序列完整 5 个子任务通过，原验证 SR 0.6、平均 SPL 0.3640955748；这只说明运行链路正常，不是全量 benchmark 指标。
- 交接前再次从独立源码副本按第 6 节运行这条五任务序列，退出码 0，得到相同 SR/SPL，无模型或寻路错误；此检查复用了本机已安装环境与资产，不等于在空白机器重装。
- 2026-09-20 12:50 北京时间启动的全量实验于 13:28 OOM 退出，保存 12 条完整序列 / 60 个子任务。位置为 scene `00800-TEEsavR23oF`、episode 12、task 0、decision 5：请求 94 MiB 时仅剩 49.06 MiB。检查时另一训练任务占 11,744 MiB，但错误发生时未同步记录各进程峰值，尚不能确认泄漏或完整显存归属。
- 该失败任务在新进程单独重跑成功；8 次 stage1 推理，采样到的进程显存最高 6,496 MiB。16 次 Habitat 创建/渲染/关闭和 100 次 ME 池化测试未见逐次增长。这些短测试不能排除更长运行的问题。

原机实验目录是 `output_logs/baseline/langmap_detailed_20260920T045026Z/`，其中有 `console.log`、`run_status.json`、`run_manifest.json`、结果 JSON 和 `oom_diagnostics/`。这些是**原机本地证据，不随 Git 分支分发**。新接手者应运行上述命令生成自己的记录；不能依赖原机 `/tmp` 文件。本次交接也不声称已在另一台空白机器完成全套重新安装或完整 benchmark。

## 来源与许可

MTU3D: *Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation*, ICCV 2025，[论文](https://arxiv.org/abs/2507.04047)。项目许可见 [LICENSE](LICENSE)；HM3D 和各模型遵循各自许可与访问条件。
