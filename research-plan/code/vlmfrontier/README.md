# 方案 C：VLMFrontier

方法名：Directional Semantic Frontier Prior。

核心：只在 MTU3D 做 frontier decision 时调用。将当前决策步最后 12 帧环视图按 `reversed(color_list[-12:])` 展开并打标签，VLM 给 12 个方向输出 target/anchor/room potential；程序根据 agent pose 和 frontier 坐标把方向分数映射到 frontier，再与 stage2 frontier logit 融合。

注意：

- VLM 不直接输出 3D 坐标，坐标仍来自 frontier detector。
- 当前包包含可编译的 `vlmfrontier.py` 与最小测试脚本草案；批量 refine1 文件是从现有结构派生的草案，建议先只落地最小测试，跑通后再整理批量版本。
- 该方向风险高于 VLMEvidence/VLMPanoBank，因为它会改变非最终探索轨迹。

服务器目标路径：

- `hm3d-online/anchor_nav/vlmfrontier.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmfrontier.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmfrontier-refine1.py`（草案）
- `scripts/run_vlmfrontier_instance_0.05_0.1.sh`（草案）

建议验证：

```bash
python -m py_compile hm3d-online/anchor_nav/vlmfrontier.py \
  hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmfrontier.py
```

