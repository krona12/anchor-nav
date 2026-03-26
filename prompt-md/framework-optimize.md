# AnchorNav：基于上下文锚链的长时域多粒度具身导航框架

> 以 MTU3D 为基础，通过 plugin_hooks.py 钩子机制叠加三个轻量模块，解决多粒度目标识别与长时域顺序导航两个核心问题。MTU3D 核心模型（Stage 1/Stage 2）**不做任何修改**。

---

## 1. 动机与核心洞察

### 1.1 锚点视角

LangMap 的实验揭示了一个统一观察：多粒度导航指令中，真正区分目标实例的是描述中的**上下文对象**，而非目标本身——

> "带几何图案地毯的卧室中的扶手椅" → 锁定关键：**几何地毯**（视觉区域特征）
> "床边靠近阳台的扶手椅" → 锁定关键：**床** + **阳台门**（空间参照物）

这些**上下文锚点（Context Anchors）**不是导航目标，但唯一确定了目标位置。**先找锚点，再找目标**，正是人类处理精细定位任务的自然策略。

### 1.2 统一的粒度抽象

将四个粒度级别统一为 **(target, anchor\_set)** 二元组：

| 粒度级别 | 目标 | 锚集合 | 锚类型 |
|---|---|---|---|
| 场景级 | teapot | ∅ | — |
| 房间级 | kettle | {kitchen} | `room` |
| 区域级 | armchair | {"geometric carpet floor"} | `area_cue` |
| 实例级 | armchair | {bed, balcony door} | `spatial_obj` |

**粒度即锚集合的特异性**：无需显式分类粒度级别。**锚点跨目标复用**：执行 g₂ 时发现的"bed"，直接成为 g₄ 的先验。

### 1.3 核心设计：锚链驱动的前沿导航

MTU3D 的 Stage 2 已经能在物体与 frontier 之间做二元选择（`decision_logits`），并通过 `frontier_selection_mode` 选取目标 frontier。AnchorNav 的扩展是：

- **当探索时**（Stage 2 决定去 frontier）：用锚点语义对 frontier 重排序——在 spin 阶段的 RGB 帧上叠加 Set-of-Marks 标记各 frontier 方向，一次 VLM 调用对所有 frontier 联合打分（借鉴 OpenFrontier 的图像空间推理思想），选出最可能找到当前缺失锚点的方向。

- **当定位时**（Stage 2 决定去物体）：用锚点约束验证候选目标——在当前帧上标记 top-K 候选物体，一次 VLM 调用确认哪个候选与锚集合最一致。

两处 VLM 干预均在**图像空间**进行，而非抽象 3D 特征空间，与 VLM 的视觉推理能力最匹配。

---

## 2. 问题定义

**输入**：含 N 个有序子目标的 episode；每个子目标有 `lang_desc`（自然语言描述）；流式 RGB-D 观测。

**输出**：按顺序导航完成 N 个子目标，每个子目标在对应粒度下被满足。

---

## 3. AnchorNav 框架

### 3.1 总体架构

```
Episode 开始（pq3d_model.reset()）
       │
       ▼
┌─────────────────────────────────┐
│  模块1: 锚链解析器 (GAP)        │ ← VLM × 1（批量处理所有子目标描述）
│  goal_anchor_list[0..N-1]       │   每个 GoalAnchor = (target, [{type,name}])
└─────────────┬───────────────────┘
              │ 持久化到 episode
              ▼
┌─────────────────────────────────┐
│  模块2: 锚状态注册表 (Registry) │ ← 零 VLM，episode 内不重置
│  每次 merge() 后自动更新        │   CLIP 相似度匹配 open_vocab_feat
└─────────────┬───────────────────┘
              │
    ┌─────────┴──────────┐
    │ plugin_hooks.py 钩子 │  ← 集成层（已存在，需填充逻辑）
    └─────────┬──────────┘
              │
    ┌─────────┴──────────────────────────────────────┐
    │  模块3: 锚导引决策覆写 (ADO)                   │
    │                                                 │
    │  Phase 1 [Stage2 决定去 frontier]               │ ← VLM × 1/决策步
    │    spin 帧 SoM 标记 frontier → 锚语义打分       │   （仅 A_missing ≠ ∅ 时）
    │    → 覆写 target_position 为最优 frontier       │
    │                                                 │
    │  Phase 2 [Stage2 决定去物体]                    │ ← VLM × 0 or 1/子目标
    │    CLIP 预筛选候选 → SoM 锚约束验证             │   （候选不唯一时触发）
    │    → 覆写 target_position 为已验证候选          │
    └─────────┬───────────────────────────────────────┘
              │ target_position (3D world coords)
              ▼
      MTU3D GreedyGeodesicFollower → 动作执行
```

### 3.2 plugin_hooks.py 钩子调用时序

```
Episode start
  → on_episode_start()  →  GAP.parse_all(), Registry.reset()

每次 decision() 前的 merge() 后
  → on_after_merge()    →  Registry.update(representation_manager)

decision() 返回后（after decision()）
  → on_after_decision() →
      if not is_final_decision (Stage2→frontier):
          ADO Phase 1: SoM frontier scoring → override target_position
      if is_final_decision (Stage2→object):
          ADO Phase 2: CLIP filter + optional SoM verify → override target_position

子目标完成后（is_final_decision=True 且到达位置）
  → on_goal_reached()   →  子目标推进 i→i+1，更新当前 GoalAnchor
```

---

### 3.3 模块一：锚链解析器（Goal-Anchor Parser, GAP）

**触发**：episode 开始，一次性处理所有子目标描述，**VLM 调用 1 次**。

**输入**：N 个子目标的 `lang_desc` 列表（从 `cur_episode['tasks']` 预读）

**输出**：`GoalAnchorList`，长度 N：
```python
@dataclass
class GoalAnchor:
    target: str                        # 要找的物体
    anchors: List[Dict[str,str]]       # [{"type": "spatial_obj"|"room"|"area_cue", "name": "..."}]
    raw_desc: str                      # 原始描述（备用）
```

**VLM Prompt 设计**（批量，一次调用）：
```
你是导航助手。以下是一个室内导航任务的 N 个子目标描述（按顺序执行）：
[1] {desc_1}
[2] {desc_2}
...
[N] {desc_N}

对每个子目标，识别：
- target: 最终要找的物体（类别名，英文）
- anchors: 帮助定位 target 的参照对象（非 target 本身），每个锚包含：
  - type: "spatial_obj"（具体物体）/ "room"（房间类型）/ "area_cue"（视觉纹理/区域特征）
  - name: 锚的英文名称

返回 JSON 数组，长度为 N。若无锚则 anchors 为空数组。

示例：
输入: "the armchair next to the bed near the balcony door"
输出: {"target": "armchair", "anchors": [{"type":"spatial_obj","name":"bed"}, {"type":"spatial_obj","name":"balcony door"}]}
```

**注意**：若当前 benchmark 每个子目标为单独 `goal_category`（object 类型），GAP 直接构造 `GoalAnchor(target=goal_category, anchors=[])` 无需调用 VLM，仅在 description 类型时触发 VLM。

---

### 3.4 模块二：锚状态注册表（Anchor Registry）

**认知类比**：只记录"哪些参照物在哪里找到了"的最小列表，而非完整场景图。

**数据结构**：
```python
@dataclass
class AnchorEntry:
    name: str          # 锚名（来自 GoalAnchor.anchors）
    mtuid: int         # RepresentationManager 中的对象索引
    position: np.ndarray  # shape (3,) 世界坐标，来自 object_box[:3]
    status: str        # "found" | "missing"
```

**更新逻辑（`on_after_merge` 触发，零 VLM）**：

每次 `representation_manager.merge()` 后执行：
1. 遍历当前子目标 `GoalAnchor.anchors` 中 status 为 `missing` 的锚
2. 对每个缺失锚，计算其名称的 CLIP 文本嵌入与 `representation_manager.open_vocab_feat`（M×768）的余弦相似度
3. 取相似度最高的对象索引 `j`，若 `sim[j] > θ_anchor`（建议 0.25）：
   - 标记该锚为 `found`，记录 `mtuid=j`，`position=object_box[j][:3]`
4. 对 `room` 类型锚：统计当前帧中所有对象的类别名称（用 CLIP 相似度推断），与预定义房间-物体先验表做软投票，推断 `room_zone`

**关键约束**：
- `representation_manager.object_class` 全为 0（`set_class_to_zero=True`），**不可用于分类**
- 必须通过 `open_vocab_feat`（768-dim CLIP 视觉嵌入）与 CLIP 文本嵌入做相似度匹配
- Registry 在整个 episode 内不重置（`pq3d_model.reset()` 时 Registry 也应 reset，但子目标切换时不 reset）

**需要添加 CLIP 文本编码器**：在 `PQ3DModel.__init__` 中加载一个轻量 CLIP 文本编码器（如 `openai/clip-vit-large-patch14` 的文本端），用于 Registry 匹配和 Phase 2 预筛选。此模型 `PQ3DModel` 中已有 `self.tokenizer`（CLIP tokenizer），只需额外加载 CLIP text encoder。

---

### 3.5 模块三：锚导引决策覆写（Anchor Decision Override, ADO）

#### Phase 1：锚导向 Frontier 选择（Stage 2 决定探索时）

**触发条件**：`not is_final_decision`（Stage 2 决定去 frontier）且 `A_missing ≠ ∅`

**marks 来源**：`frontier_waypoints`（3D 世界坐标列表）投影到 spin 阶段的 RGB 帧。

**实现流程**：

```
输入: frontier_waypoints（K 个，世界坐标）
     color_list（12 帧 spin 帧，各有对应 agent_state_list）
     A_missing（缺失锚描述列表）

① 对每个 frontier waypoint f_k（世界坐标）：
   对 12 个 spin 帧，用相机内参+位姿计算 f_k 的投影像素坐标 (px, py)
   选取 f_k 投影最居中（离图像中心最近且在 FOV 内）的帧 frame_idx[k]
   记录投影坐标 (px_k, py_k)

② 选取"最具代表性"的帧（包含最多 frontier 投影的帧，或所有帧取子集）
   对选定帧，在 (px_k, py_k) 处叠加数字标记（用 cv2.circle + cv2.putText 画圆圈数字）

③ VLM 调用（单次，对所有 frontier 联合打分）：
   输入: 标注了 frontier 标记的 RGB 帧（可多帧）+ A_missing 描述
   Prompt: "I need to find: [A_missing 描述]. The image shows marked directions (①②③...).
            Score each mark (0.0-1.0) for likelihood of finding the target objects.
            Return JSON: {"scores": [p1, p2, ..., pK]}"
   输出: {f_k: p_k}

④ 效用排序：
   u(f_k) = p_k / (||agent_pos - f_k|| + ε)
   选 argmax u(f_k)，其 3D 世界坐标覆写 target_position

⑤ 返回覆写后的 target_position（frontier 世界坐标）
```

**边界情况**：
- 若所有 frontier 均不在任何帧的 FOV 内（极少发生，frontiers 来自当前可见边界）：降级为 Stage 2 原始选择
- 若 VLM 调用失败：降级为 Stage 2 原始选择
- 若 A_missing 为空（所有锚已找到）：跳过 Phase 1，由 Stage 2 正常决策

**相机投影说明**：
- 相机内参：`make_intrinsic_hfov(42, 640/360)`，42° 水平 FOV，640×360 分辨率
- 相机外参：`agent_state.sensor_states['color_sensor']` 提供位置和旋转四元数
- 像素坐标转换需与 `data_utils.py` 中的 `convert_from_uvd` 保持坐标系一致

---

#### Phase 2：锚约束目标验证（Stage 2 决定去物体时）

**触发条件**：`is_final_decision=True`（Stage 2 决定去物体）且 `A_found` 充分

**marks 来源**：`representation_manager.object_box`（M×6）中 target 类别候选的中心点，投影到当前帧。

**实现流程**：

```
输入: representation_manager（含 open_vocab_feat, object_box）
     current_goal_anchor（当前子目标：target + A_found）
     最新 spin 帧 + agent_state

① CLIP 预筛选（快通道，零 VLM）：
   计算 target 名称的文本嵌入与所有 open_vocab_feat 的余弦相似度
   取 top-K（K=5）候选，索引为 candidate_ids

② 候选唯一性判断：
   若 top-1 sim 与 top-2 sim 差距 > θ_gap（建议 0.15）：
     → 直接用 top-1 候选，target_position = object_box[candidate_ids[0]][:3]
     → 跳过 VLM，返回

③ 候选模糊时触发 SoM VLM（慢通道）：
   对 candidate_ids 中各候选，投影 object_box 中心到当前帧，叠加数字标记
   同时标注 A_found 中锚点的投影位置（用不同形状区分，如方框）
   VLM 调用：
   Prompt: "Target: [target 描述].
            Reference objects (shown as squares): [A_found 各锚描述 + 相对关系].
            Candidate objects (shown as circles, numbered): [候选标记列表].
            Which numbered circle best matches the target description?
            [若含 area_cue 锚]: Also confirm the floor/wall in the area matches [area_cue 描述].
            Return JSON: {"best_match": <number>, "confidence": <0-1>}"
   若 confidence > θ_conf（建议 0.6）：覆写 target_position
   否则：保持 Stage 2 原始选择

④ 返回最终 target_position
```

---

## 4. 与 MTU3D 的集成

**三类修改，核心模型不动：**

| 修改位置 | 内容 | 侵入性 |
|---|---|---|
| `goat-nav.py` | 在固定位置插入 5 个钩子调用 | 低（仅插入调用，逻辑在 hooks 里） |
| `plugin_hooks.py` | 填充已有 stubs 的实现逻辑 | 已有接口，填充内容 |
| `PQ3DModel.__init__` | 新增 CLIP text encoder 加载 | 低（新增一个模型属性） |
| 新增 `anchor_nav/` | GAP, Registry, ADO 模块 | 新文件，不影响现有代码 |
| 新增 `vlm/` | Qwen VLM 客户端封装 | 新文件 |

**数据流接口**（只读，无需修改 MTU3D 内部）：

| 数据 | 来源 | 用途 |
|---|---|---|
| `frontier_waypoints` | `goat-nav.py` 中 `detect_frontier_waypoints()` 输出 | Phase 1 SoM 投影 |
| `representation_manager.open_vocab_feat` | `PQ3DModel.representation_manager` | Registry CLIP 匹配 |
| `representation_manager.object_box` | `PQ3DModel.representation_manager` | Phase 2 候选投影 |
| `color_list`, `agent_state_list` | `goat-nav.py` spin 阶段收集 | SoM 帧选择与投影 |
| `is_final_decision` | `pq3d_model.decision()` 返回值 | Phase 1/2 路由 |
| `target_position` | `pq3d_model.decision()` 返回值 | 覆写入口 |

---

## 5. VLM 调用汇总

| 位置 | 触发条件 | 估计次数 |
|---|---|---|
| GAP | episode 开始，description 类型任务 | **1**（批量） |
| ADO Phase 1 | 每次 frontier 决策步，A_missing ≠ ∅ | **~2–4 / 子目标** |
| ADO Phase 2 | CLIP top-1/top-2 差距 < θ_gap | **~0–1 / 子目标** |
| **合计（N=4）** | | **~9–21** |

**效率来源**：
- Phase 1 每次对**所有** frontier 联合打分（单次 VLM，不逐 frontier 查询）
- Phase 2 CLIP 快通道处理大多数清晰候选（VLM 仅在模糊时触发）
- Registry 零 VLM，锚复用减少后期 Phase 1 步数

---

## 6. 贡献总结

| 贡献 | 机制 | 创新点 |
|---|---|---|
| **锚链抽象**（GAP） | 批量解析 (target, anchor_set) | 统一四粒度；无需显式粒度分类 |
| **任务驱动注册表**（Registry） | CLIP 匹配 open_vocab_feat；跨子目标不重置 | 利用 MTU3D 已有嵌入，零额外标注 |
| **锚导引决策覆写**（ADO） | Phase 1 以锚为目标做 frontier SoM 打分；Phase 2 用锚约束验证候选 | 将 OpenFrontier SoM 范式从单目标扩展到锚链多目标 |

---

## 7. 与相关工作的关键区别

| 方法 | 差距 |
|---|---|
| **MTU3D** | 单目标，frontier 选择无锚语义，无粒度推理 |
| **LH-VLN**（CVPR 2025） | 多目标，但记忆非结构化，无锚点概念 |
| **OpenFrontier** | SoM frontier 导航，但单目标，无锚链分解 |
| **SayNav / MLFM+RGraph** | 场景图或关系图，但单目标或无多粒度统一 |
| **本文** | 锚链统一多粒度 + CLIP 驱动轻量注册表 + SoM 扩展至锚链导航 |

---

## 参考文献

- Zhu et al. (2025). MTU3D. *ICCV 2025*.
- Song et al. (2025). LH-VLN. *CVPR 2025*.
- Miao et al. (2026). LangMap. arXiv:2602.02220.
- Padilla et al. (2026). OpenFrontier. arXiv:2603.05377.
- Li et al. (2024). MLFM+RGraph.
- Chen et al. (2024). SayNav.
