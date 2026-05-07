# 最终可落地科研 Methods 与代码对齐

本文件是对 MTU3D、你当前尝试、以及本地/远端 `anchor-nav` 代码接口的精读后结论。目标是形成可以落地到服务器、可以跑最小测试、可以写成 method 的改法。

## 1. MTU3D 在当前代码中的真实决策路径

关键文件：`E:\Deep_Learning\anchor-nav\hm3d-online\data_utils.py`

`PQ3DModel.decision()` 的执行顺序：

1. 对本轮输入的 `color_list/depth_list/agent_state_list` 批量提取 DINOv2 feature。
2. 用 FastSAM 对 RGB 做 instance segmentation。
3. 对每帧用 depth + pose 回投点云，按 segment pooling 2D feature。
4. stage1 `EmbodiedPQ3DInstSegModel` 输出局部 object query/mask/box/open-vocab feature。
5. `RepresentationManager.merge(pred_dict_list, frame_rgbs=color_list)` 把新观测合并到全局 object memory。
6. stage2 构造 `query_locs/query_scores/mv_seg_fts/vocab_seg_fts`。
7. 把 frontier waypoint 追加到 object query 后面：
   - frontier center = `[x, z, y]`
   - frontier box = center + zeros
   - frontier score = 1
   - `real_obj_pad_masks=False`
8. tokenizer 编码 sentence，`Query3DVLE` 输出 object/frontier logits 与 `goto_frontier_probability`。
9. 若 `goto_frontier_probability <= 0.5 and decision_num > min_decision_num`，进入 object decision；否则去 frontier。
10. 如果传入 `analysis_output_dir`，写出 `stage2_decision.json`，这是所有后处理模块最稳定的接口。

这说明：**最安全的研究改法是 post-stage2 decision auditor**。它不动 FastSAM/DINO/PQ3D/stage2，只在 decision target 已经生成后做候选校准。

## 2. 你当前 `vlmtop5` 的定位

关键文件：

- `hm3d-online/anchor_nav/vlmtop5.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmtop5.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmtop5-refine1.py`

当前流程：

1. `register_new_object_panorama_frames()` 记录新增 object slot 对应的当前决策步 360 panorama frames。
2. 最终 object decision 时从 `stage2_decision.json` 读取 top-k object candidates。
3. 给 VLM 输入 top-k first RGB，让 VLM 选 best_index 和 safe gate。
4. 若 VLM 选择非第一张，拼接该候选 slot 所属 panorama，做二次 verify。
5. verify 通过才把目标坐标改成 selected candidate center。
6. 写 10/01/11/00 有效性日志。

主要问题：

- first RGB 是候选第一次被检测到的单帧，不一定能体现任务中的锚点/关系。
- panorama verify 只验证 selected，不用于所有候选比较。
- selection prompt 的输出是选择 + gate，不是 evidence score，因此实验故事容易被质疑为“prompt trick”。
- hard override 太冒险；更适合改成 calibrated reranking。

## 3. 最终 Methods：Evidence-Calibrated Snapshot Verification

建议主方法名：**Evidence-Calibrated Snapshot Verification (ECSV)**  
一句话：在 MTU3D 的最终 object decision 处，为 top-k memory object 构建视觉证据卡片，用 VLM 做约束级证据审计，再以校准分数而非硬选择修正目标。

### 3.1 Evidence Card

每个 object candidate 的 evidence card 包含：

- stage2 rank；
- memory slot index；
- stage2 `og3d_logit`；
- memory object center；
- first RGB；
- registered 360 panorama snapshot；
- task-decomposed constraints：target phrase、attributes、nearby anchors、scene anchors、spatial relation。

其中 first RGB 用于看候选局部，panorama snapshot 用于看锚点和上下文。

### 3.2 VLM Constraint Audit

VLM 输出 strict JSON：

```json
{
  "task_constraints": {
    "target": "...",
    "attributes": ["..."],
    "anchors": ["..."],
    "relations": ["..."]
  },
  "candidates": [
    {
      "index": 1,
      "target_match": "exact|generic|absent",
      "attribute_score": 0.0,
      "anchor_score": 0.0,
      "relation_score": 0.0,
      "ambiguity": "low|medium|high",
      "evidence_score": 0.0,
      "decisive_visible_constraints": ["..."],
      "missing_constraints": ["..."]
    }
  ],
  "best_index": 1,
  "safe_to_override_image1": false,
  "reason": "..."
}
```

论文里可以把它表述成从 free-form VLM judgement 到 structured evidence vector 的转换。

### 3.3 Calibration Gate

不直接用 `best_index`，而是：

```text
normalized_stage2 = minmax_or_rank_norm(stage2 logits)
final_score = normalized_stage2 + lambda * evidence_score - mu * ambiguity
```

修正规则：

- 非 top1 的 `evidence_score - top1_evidence_score >= delta_evidence`；
- 非 top1 至少多满足一个 decisive constraint；
- 非 top1 的 ambiguity 不是 high；
- 若 `object_top1_top2_logit_gap` 很大，需要更高 evidence delta；
- 若 task 是 instance-level 或 description 含 anchor/relation，降低触发门槛；
- 否则保留 baseline，但记录 VLM 的 disagreement。

这样可以讲成 “VLM as conservative calibrator”，比“VLM 做最终裁判”更稳。

### 3.4 与当前代码对齐

需要新增：

- `hm3d-online/anchor_nav/vlmevidence.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence-refine1.py`
- `scripts/run_vlmevidence_instance_0.05_0.1.sh`

不需要改 baseline、不需要改 `data_utils.py`、不需要改 `posnode.py`。

最小测试直接复用 `vlmtop5.py` 的接口：

- `decision_aux`
- `stage2_json_path`
- `baseline_target_xyz`
- `panorama_frames_by_slot`
- `output_dir`
- `goal_positions_xyz`

## 4. 第二 Methods：Directional Semantic Frontier Prior

建议方法名：**Directional Semantic Frontier Prior (DSFP)**  
一句话：给 MTU3D 的 frontier query 添加来自当前 360 panorama 的方向语义先验，但 frontier 坐标仍由原 frontier detector 提供。

### 4.1 为什么不是“VLM 选 frontier 坐标”

VLM 对坐标不可靠。更稳的是让 VLM 看环视图，输出语义方向：

```json
{
  "directions": [
    {
      "view_index": 1,
      "target_potential": 0.0,
      "anchor_potential": 0.0,
      "room_potential": 0.0,
      "avoid": false,
      "reason": "..."
    }
  ],
  "best_view_indices": [3, 4]
}
```

然后程序用 agent pose 和 frontier 坐标算 bearing，把 direction prior 投到 frontier 上。

### 4.2 与 MTU3D 的差异点

MTU3D stage2 已经对 frontier 打分，但 frontier query 缺少视觉语义。DSFP 只在 `is_final=False` 时运行：

- 读取 `stage2_decision.json.frontier_candidates`。
- 取当前决策步最后 12 帧，按 `reversed(color_list[-12:])` 展开。
- 让 VLM 给 12 个方向打分。
- 对每个 frontier，根据 bearing 映射到最近 view index。
- 融合 stage2 frontier logit 与 VLM direction score。

日志输出：

- baseline frontier target；
- semantic frontier target；
- baseline/corrected 到 goal 的距离；
- `case=10/01/11/00`；
- VLM 选择的方向和原因；
- 若不改，则 `correction_rejected_reason`。

## 5. 第三 Methods：Snapshot Bank Comparison

建议方法名：**Co-visibility Snapshot Bank (CSB)**  
一句话：把现有 `panorama_frames_by_slot` 显式提升为 snapshot memory bank，让 VLM 比较 top-k candidate 的 full-context snapshot，而不是 first RGB。

这条线和 ECSV 的差别：

- ECSV 是 first RGB + snapshot 的 evidence audit。
- CSB 是只比较 snapshot，强调“上下文/锚点/关系比候选物体局部更重要”。

可作为 ablation：

- baseline MTU3D；
- first RGB VLM top5；
- panorama verify only；
- snapshot bank top-k；
- ECSV full。

如果 CSB 比 first RGB 更好，就能直接支撑论文叙事：MTU3D 的 memory object center 可靠，但 instance-level language alignment 需要 co-visible context。

## 6. 批判性自检

可能失败点：

- VLM 对 panorama 中小物体漏检：需要降低 panorama 分辨率损失，或保留 first RGB 辅助。
- VLM 过度相信语义先验：必须保留 stage2 logit 和 strict advantage gate。
- top-k 没有正确目标：任何 rerank 都救不了，需要 DSFP 改善探索采样。
- slot 对应 panorama 可能不是最佳视角：这是当前可用接口的限制；远端已有 `vlmdepthbox` 可进一步做 last-mile bbox/depth projection，但不要混入主线。
- 任务描述本身可能缺 anchor：此时 ECSV 应退化为 conservative generic target check，不应强行切换。

必须验证：

- `stage2_decision.json` 在最小/批量脚本均稳定生成。
- `panorama_frames_by_slot` 对 top-k selected slot 都存在，否则 strict raise，不能 silent fallback。
- VLM JSON parse 失败要显式报错，符合你“不做兜底掩盖 bug”的要求。
- 日志必须能区分：VLM called、VLM selected、gate passed、correction applied、helpful/harmful。

## 7. 代码落地顺序

1. 先落地 `vlmevidence`：风险最低，直接替换当前 `vlmtop5`。
2. 再落地 `vlmpanobank`：作为 snapshot-only ablation。
3. 最后落地 `vlmfrontier`：涉及非最终 frontier target，实验风险更高，但最有潜力解决 memory bank 质量。

远端注意：

- 当前 `/home/chenlin/krona/anchor-nav` 有未跟踪 `vlmdepthbox`，不要覆盖。
- 所有新增模块使用独立文件名。
- 每个方案的最小测试日志写 `output_process`，批量日志写 `output_logs/`。
- baseline 文件不修改。

