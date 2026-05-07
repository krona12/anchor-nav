# 最可能借鉴的零样本方案蒸馏

本文件把调研压缩成可落地的研究路线。目标不是工程 patch，而是在 MTU3D 的 memory/query/decision 框架上形成能讲得通、能实验验证、能写成 method 的改动。

## 总体判断

MTU3D 已经有强 perception 与 object/frontier unified query。我们不应该重复训练一个导航模型，也不应该让 VLM 每步接管决策。最有价值的切入点是：**把 MTU3D 的黑盒 query score 变成可被 VLM 审计的 evidence-aware decision**。

核心公式可以写成：

```text
score(candidate) =
  MTU3D_stage2_logit
  + lambda_v * VLM_constraint_match(candidate_snapshot, task)
  + lambda_c * co_visibility_context(candidate_snapshot, anchors)
  - lambda_a * ambiguity_penalty(candidate, top_k)
```

对于 frontier：

```text
score(frontier) =
  MTU3D_frontier_logit
  + lambda_s * semantic_direction_prior(bearing(frontier), current_panorama, task)
  - lambda_r * revisit_penalty(frontier)
```

这保留 MTU3D 的 spatial memory 与 planner，只让 VLM在两个弱点上提供低频语义校准：细粒度目标对齐、frontier 语义潜力。

## 方向 A：Evidence Trace Reranking

灵感来源：UniGoal 的 graph matching、MSGNav 的 visual edge evidence、ReMemNav 的 rethinking。

基本做法：

- 输入：stage2 top-k object candidates、每个 candidate 的 first RGB、每个 slot 注册时的 panorama snapshot、原始 task description、stage2 logit。
- VLM 输出结构化 evidence trace：
  - target category 是否可见；
  - attribute/material/color 是否可见；
  - nearby anchor 是否可见；
  - spatial relation 是否可见；
  - candidate 是否只是 generic category；
  - 与 baseline top-1 相比是否有严格优势；
  - `calibrated_score`，范围 0-1。
- 最终不是简单“VLM 选几号就切几号”，而是只在满足以下条件时修正：
  - VLM 认为非 top1 的 evidence_score 高；
  - 非 top1 至少满足一个 top1 缺失的 decisive constraint；
  - 非 top1 的 panorama snapshot 支持 instance-level context；
  - 与 top1 的分数差超过阈值，或者 stage2 top1/top2 gap 较小。

为什么比当前 `vlmtop5` 更能讲故事：

- 当前方案是 “VLM top-5 chooser + panorama veto”；方向 A 是 “constraint-level evidence audit”。
- 它能输出论文图表：约束矩阵、baseline 错因、VLM 修正原因、10/01/11/00。
- 它并不要求 VLM 产生 3D 位置，只利用 MTU3D 已经可靠的 slot center，因此落地风险低。

最小改动接口：

- 模块：`hm3d-online/anchor_nav/vlmevidence.py`
- 最小测试：`hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence.py`
- 批量测试：`hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence-refine1.py`
- 运行脚本：`scripts/run_vlmevidence_instance_0.05_0.1.sh`

## 方向 B：Panorama Snapshot Bank

灵感来源：3D-Mem 的 Memory Snapshot 与 MSGNav 的多模态 scene graph。

基本做法：

- 不在每个环视都调用 VLM 建 memory。
- 仍然让 MTU3D/FastSAM/PQ3D 负责 object memory。
- 每当 `RepresentationManager.merge()` 新增 object slot，使用已有 `register_new_object_panorama_frames()` 把该 slot 绑定到当前 360 panorama。
- 最终决策时，对 top-k 候选取各自 panorama，而不是只取 first RGB。
- VLM 不需要“识别所有物体”，只回答每个 snapshot 是否包含 target/anchor/relation。

为什么它解决当前痛点：

- 细粒度 instance 往往靠上下文区分，例如“床边的白色小桌”“带粉色椅子的桌子”“电视上方/壁炉旁的物体”。first RGB 只显示局部物体，snapshot 才包含 anchor。
- 这能把 `panorama_frames_by_slot` 从日志工具升级为研究方法：**Co-visibility Snapshot Memory for Query Verification**。

可落地注意：

- top-k panorama 都发给 VLM 会大图开销高，可先只对 stage2 top-3 或 logit gap 小的 top-k 触发。
- 也可以先发低分辨率 stitched panorama，再保存原图日志。
- 如果 VLM 输出非 top1，但 `stage2 logit_gap` 很大，应作为 “soft correction denied” 记录，不强切。

## 方向 C：Semantic Frontier Direction Prior

灵感来源：FOM-Nav 的 Frontier-Object Map、3D-Mem 的 Frontier Snapshot、BeliefMapNav 的 semantic belief distribution。

基本做法：

- 在 `is_final == False` 的 frontier decision 处触发。
- 将当前决策步最后 12 帧环视图拼成 labeled panorama，或者直接以 12 张顺序图输入 VLM。
- VLM 输出每个方向/扇区对目标的语义潜力：`target_likelihood`、`anchor_likelihood`、`room_prior`、`avoid_reason`。
- 根据 agent yaw 和 frontier 坐标计算 frontier bearing，把 VLM 的方向潜力映射到每个 frontier。
- 与 `stage2_decision.json.frontier_candidates[*].og3d_logit` 融合，选择新的 frontier。

重要约束：

- 不让 VLM 直接输出坐标，这会引入空间幻觉。
- VLM 只判断“朝哪个可见方向/门洞/房间继续更合理”，坐标仍来自 frontier detector 和 planner。
- 日志要输出 baseline frontier 是否接近最终 goal、semantic frontier 是否接近 goal，以及 frontier correction 的 10/01/11/00。

为什么它能补 MTU3D 的弱项：

- MTU3D 的 frontier query 对语言任务没有视觉语义 grounding，尤其 early exploration 决定后续 memory bank 质量。
- 方向 C 给 frontier 加的是轻量语义 prior，不改 stage2 模型也不训练。

## 最推荐组合

短期实验主线：

1. 先做方向 A，作为你当前 `vlmtop5` 的强版本。它最稳定，实验成本最低，最容易跟 baseline 对齐。
2. 同时做方向 B 的 panorama-only variant，用来证明“共视上下文比 first RGB 更能区分细粒度 instance”。
3. 方向 C 作为第二篇/后续章节：如果 object correction 有提升但受限于 memory bank 采集，frontier semantic prior 就能解释为什么要影响探索。

论文方法名建议：

- Evidence-Grounded Query Verification (EGQV)
- Snapshot-Conditioned Object Decision (SCOD)
- Directional Semantic Frontier Prior (DSFP)
- Fast-MTU / Slow-VLM Dual-Process Navigation

## 实验指标

必须记录：

- SR/SPL 原始指标。
- 最终 object correction 的 `case=10/01/11/00`。
- baseline distance 与 corrected distance 到 GT object 最近位置。
- VLM 调用次数、平均耗时、平均 token/image 数。
- correction acceptance/rejection reason。
- 按 task 类型分组：instance/object/room/region，优先 instance。

建议新增：

- `constraint_matrix`: 每个 candidate 满足 target/anchor/relation 的布尔/分数。
- `strict_advantage_count`: 非 top1 相对 top1 多满足多少关键约束。
- `frontier_semantic_gain`: semantic frontier 与 baseline frontier 到 GT 的距离差。
- `vlm_trigger_reason`: final, low_gap, repeated_frontier, long_explore, etc.

## 不推荐路线

- 每个环视都让 VLM 列出物体再建独立 memory bank：调用成本太高，且 VLM 输出没有 3D mask/深度绑定，融合难。
- 让 VLM 直接选择 3D 坐标：容易空间幻觉，且无法利用 MTU3D 已有 planner/point cloud 优势。
- 大范围改 `PQ3DModel.decision()` 或 stage2 网络：需要训练，违反当前“无需训练”的研究目标。

