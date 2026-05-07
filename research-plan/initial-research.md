# MTU3D 与零样本 VLM 导航初始调研

日期：2026-05-06  
目标：围绕 MTU3D 的 online memory/query/navigation 接口，寻找可借鉴的 2025/2026 零样本 VLM 导航策略，并把后续改法约束到 `anchor-nav` 现有代码可落地的位置。

## 1. MTU3D 本体

论文与代码：

- Paper: [Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Move_to_Understand_a_3D_Scene_Bridging_Visual_Grounding_and_ICCV_2025_paper.html)
- GitHub: [MTU3D/MTU3D](https://github.com/MTU3D/MTU3D)
- Project: [mtu3d.github.io](https://mtu3d.github.io/)

核心观点：

MTU3D 把「已经观测到的物体」和「尚未探索区域的 frontier」统一成 query，并用同一个 vision-language-exploration 决策层输出目标 query。论文摘要给出的三个关键点是：在线 query-based representation learning、把 frontier 也作为 query 的统一 grounding/exploration objective、以及大规模 trajectory pretraining/fine-tuning。官方代码在 `hm3d-online/data_utils.py::PQ3DModel.decision()` 中也能直接看到对应实现：FastSAM 分割 RGB，DINOv2 提取 2D feature，stage1 产生局部 3D instance query，`RepresentationManager.merge()` 写入 spatial memory bank，stage2 把 memory object query 与 frontier query 拼接后统一打分。

与当前问题相关的机制：

- 物体候选：`stage2_decision.json.object_candidates` 记录 memory slot、`merged_object_score`、`center_habitat_xyz`、`og3d_logit`。你当前 `vlmtop5` 就是读取这里的 top-k。
- frontier 候选：`stage2_decision.json.frontier_candidates` 只含 frontier 坐标与 logit；输入给模型时 frontier 的 feature 主要来自坐标盒和固定 tag，不含语义图像证据。
- 最终分支：当 `goto_frontier_probability <= 0.5` 且 `decision_num > min_decision_num`，或没有 frontier 时，进入 object decision；否则去 frontier。
- 细粒度错误来源：MTU3D 的 object memory 更像「视觉/几何可检索实例」，但 task 往往描述的是 instance-level language constraint，例如颜色、材质、相邻锚点、相对位置、局部用途。FastSAM + DINO/PQ3D 可以把同类物体聚起来，却不一定保留「为什么这个实例是目标」的可审计证据。

## 2. 近期引用/相邻工作

由于 MTU3D 是 ICCV 2025 新论文，正式引用链仍在快速变化。当前能检索到的近期相邻/引用线索如下：

- [Hydra-Nav: Object Navigation via Adaptive Dual-Process Reasoning, arXiv 2026](https://arxiv.org/abs/2602.09972)：引用 MTU3D 作为近期 VLM/ObjectNav 强基线；核心思想是 slow/fast dual-process，只有在关键点触发重推理，以减少 VLM 频繁调用成本。它本身有训练，不适合直接作为“无需训练”方案，但“关键点触发 VLM”非常适合改造 `vlmtop5`。
- [MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation, CVPR 2026](https://arxiv.org/abs/2511.10376)：accepted by CVPR 2026。它认为文本化 scene graph 会丢失视觉证据，因此把 graph edge 替换成动态分配的图像证据，并提出 last-mile visibility-based viewpoint decision。对 MTU3D 的启发很直接：不要只让 VLM 看候选物体 crop/first RGB，要让它看“候选物体所在的共视上下文”。
- [FOM-Nav: Frontier-Object Maps for Object Goal Navigation, arXiv 2025](https://arxiv.org/abs/2512.01009)：把 frontier 和 object 一起放进 Frontier-Object Map，再由 VLM 做 high-level goal prediction。它有数据构造/训练成分，不是纯无需训练，但对 MTU3D 的 frontier query 弱语义问题很有价值：frontier 不能只有坐标，至少应绑定“frontier 方向上看见什么/可能通向什么”的语义证据。
- [ReMemNav: A Rethinking and Memory-Augmented Framework for Zero-Shot Object Navigation, arXiv 2026](https://arxiv.org/abs/2603.26788)：明确训练-free/zero-shot；用 panoramic semantic priors、episodic semantic buffer 和 adaptive dual-modal rethinking 主动验证目标可见性、避免 deadlock。对当前项目的启发是：不要每一步都用 VLM 建 memory，而是在最终决策/低置信度/重复 frontier 等关键状态做 rethinking。

## 3. 2025/2026 顶会零样本 VLM 导航论文

### UniGoal, CVPR 2025

来源：[CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.html)

策略：

- 把 object category、instance image、text description 三类 goal 统一成 goal graph。
- 在线维护 scene graph，并在每一步进行 scene graph 与 goal graph matching。
- 按匹配状态分三阶段：zero match 时探索子图，partial match 时使用 coordinate projection 与 anchor pair alignment 推断目标位置，perfect match 时做 scene graph correction 与 goal verification。
- 使用 blacklist 防止在失败匹配处反复探索。

对 MTU3D 的可借鉴点：

- `vlmtop5` 现在只做“top-k 图片选择”，缺少“目标约束被分解成 target/anchor/relation，再对候选做约束匹配”的中间结构。
- 可在最终 object decision 前，将 task 分解为 target、nearby anchor、scene anchor、spatial relation，并要求 VLM 对每个候选输出可见约束矩阵。
- 对 frontier，可维护“失败 frontier blacklist + 语义方向记录”，避免重复探索视觉上不可能的方向。

### 3D-Mem, CVPR 2025

来源：[CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.html)

策略：

- 用 Memory Snapshot 表示已探索区域：一张多视角图像/观测帧绑定一组共视物体，保留 foreground objects、空间关系、背景 context。
- 用 Frontier Snapshot 表示未探索区域：frontier 绑定它首次被观测到的视觉 glimpse。
- VLM 不是直接看所有 object nodes，而是检索少量相关 snapshot 后做决策。

对 MTU3D 的可借鉴点：

- 当前 `panorama_frames_by_slot` 已经近似一个 snapshot 索引：每个新 object slot 绑定其被纳入 memory 时的 360 scan。
- 与其让 VLM 每次环视都产物体，不如复用 `register_new_object_panorama_frames()` 的“新物体 slot -> 当时环视帧”映射，在最终 top-k 决策时取候选对应 panorama 作为共视证据。
- 这比当前 first RGB top-5 更能解决“语义正确但细粒度/锚点不对齐”。

### BeliefMapNav, NeurIPS 2025

来源：[arXiv](https://arxiv.org/abs/2506.06487)；作者主页列为 NeurIPS 2025 接收。

策略：

- 维护 3D voxel belief map，估计目标存在的先验/后验分布。
- 把 LLM 语义先验、视觉 embedding 与在线观测融合到全局 3D posterior belief。
- 不让 VLM 贪心选下一点，而是把语义推理落到空间分布上，再做 sequential path planning。

对 MTU3D 的可借鉴点：

- MTU3D 的 stage2 已经给每个 object/frontier 一个 logit，但不是 belief distribution。
- 可以把 VLM 输出变成校准项，而不是硬切换：例如候选得分 = stage2 logit + constraint score + context score - ambiguity penalty。
- 对 frontier，VLM 不必直接输出坐标，可输出「12 个环视方向的语义潜力」，再投影到 frontier bearing 上形成 frontier semantic prior。

### MSGNav, CVPR 2026

来源：[arXiv](https://arxiv.org/abs/2511.10376)

策略：

- Multi-modal 3D Scene Graph 不把关系压成文本，而是把动态图像作为 edge evidence。
- Key Subgraph Selection 降低 VLM 推理规模。
- Adaptive Vocabulary Update 支持开放词汇。
- Closed-Loop Reasoning 和 Visibility-based Viewpoint Decision 解决 last-mile：不仅选对对象，还要选一个可达且可见的最终视角。

对 MTU3D 的可借鉴点：

- 把候选 object slot 的“first RGB + associated panorama + stage2 score”组织成 evidence card，VLM 输出 target/anchor/relation 是否被满足。
- 最后不只修正 object center，还可以选择“VLM 指定 bbox/depth 投影位置”或“候选附近可见 viewpoint”。远端已有未跟踪的 `vlmdepthbox`，说明你也在走这条线；后续新方案应避开同名文件。

### ReMemNav, arXiv 2026, training-free

来源：[arXiv](https://arxiv.org/abs/2603.26788)

策略：

- 使用 panoramic semantic priors 与 episodic semantic buffer queue。
- 通过 adaptive dual-modal rethinking 主动验证目标是否可见，并用历史 memory 修正错误决策。
- 强调 VLM 的空间幻觉、局部探索死循环、高层语义与低层控制脱节。

对 MTU3D 的可借鉴点：

- 当前最重要的不是每一帧都建立 VLM memory，而是构建“触发条件”：final object decision、top1/top2 logit gap 小、VLM 与 stage2 冲突、重复 frontier、已探索很久仍无 target。
- 低频 rethinking 的故事比“每次环视问询 VLM”更强：用 VLM 做关键决策审计器，而不是替换 MTU3D 的 perception/memory。

### EmergeNav, arXiv 2026, zero-shot VLN-CE

来源：[arXiv](https://arxiv.org/abs/2603.16947)

策略：

- 把 zero-shot VLN-CE 表述为 structured embodied inference。
- Plan--Solve--Transition hierarchy：先分阶段计划，再局部求解，并显式验证阶段转换。
- Contrastive dual-memory reasoning 用于进度 grounding。

对 MTU3D 的可借鉴点：

- 对 instance navigation，可以把“寻找目标”拆成：语义区域探索、候选发现、候选验证、last-mile 可见性验证。当前 `vlmtop5` 只覆盖第三步，缺少第二步与第四步的闭环。

## 4. 对当前 `anchor-nav` 的初步判断

本地路径实际是 `E:\Deep_Learning\anchor-nav`，不是 `E:\DeepLearning\anchor-nav`。远端仓库在 `/home/chenlin/krona/anchor-nav`，当前分支 `mtu3d-anchor`。远端已有未跟踪文件：

- `hm3d-online/anchor_nav/vlmdepthbox.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmdepthbox.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmdepthbox-refine1.py`
- `scripts/run_vlmdepthbox_instance_0.05_0.1.sh`

我不会覆盖这些文件。后续方案应使用全新模块名。

现有 `vlmtop5` 的优点：

- 只在最终 object decision 调 VLM，成本可控。
- 使用 `stage2_decision.json` 与 `last_decision_aux`，接口干净。
- 有 10/01/11/00 有效性日志，便于做实验故事。
- 已正确使用 `color_list[-12:]` + `reversed` + `stitch_panorama()` 获得候选 slot 的环视上下文。

现有 `vlmtop5` 的不足：

- 对 VLM 的输入主要是 candidate first RGB；first RGB 不一定包含 anchor/relation，且可能是局部/远景/遮挡视角。
- 二次 panorama verify 只作为 veto，不参与 top-k 比较；如果 first RGB 选错，panorama verify 可能无法恢复。
- VLM 输出是 hard gate，缺少与 stage2 logit 的校准融合，可能“少数高置信 VLM 错误”直接伤害 baseline。
- frontier 路径仍没有语义指导，导致 memory bank 的采集质量依赖无语义探索。

## 5. 初步结论

最可能有论文故事且能落地的方向不是“每次环视都让 VLM 重新建 memory”，而是：

1. **Snapshot-grounded final reranking**：把 top-k object 候选各自绑定的 panorama snapshot 作为共视上下文，让 VLM 做约束矩阵与校准打分。
2. **Semantic frontier prior**：让 VLM 只在 frontier decision 时看当前 360 panorama，输出 12 个方向的 target/anchor potential，再映射到 frontier bearing，与 stage2 frontier logit 融合。
3. **Dual-process trigger**：只在不确定、冲突、重复探索、最终决策时触发 VLM，形成“MTU3D fast memory + VLM slow audit”的结构，而不是让 VLM 替换 perception/memory。

