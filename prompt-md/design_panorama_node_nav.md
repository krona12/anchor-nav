# Panorama Node Navigation 设计文档

**思路来源**：TQSI 的根本缺陷是用 PQ3D logit 分布熵估计锚点方向，是间接、噪声大的信号。
新方案改为记录**真实空间共现**：某个位置扫描到了 target + anchor → 该位置的 3D 对象就是候选目标。

---

## 1. 核心思路

```
每 2 次 decision 时：
  拼接 360° 全景图 → VLM 识别当前可见物体名词列表
  同时记录当前帧可见的 3D object_indices（来自 RepresentationManager）
  → 存入 pos_node_registry: pos → {vlm_names, object_indices}

final decision 时：
  description → VLM 分解为 target_desc + anchor_desc
  → 在 pos_node_registry 里查询：哪个 pos 同时见过 target + anchor
  → 在 PQ3D top-K 里找属于该 pos 的 object → 作为最终目标

  级联 fallback：
    1. target+anchor 共现 pos → 找 pos 内的 top-K 对象
    2. 只有唯一 anchor → 找 anchor 所在 pos 内的 top-K 对象
    3. 只有唯一 target → 找 target 所在 pos 内的 top-K 对象
    4. 多个 target / 多个 anchor → 用 PQ3D top-1
    5. 都没找到 → 用 PQ3D top-1
```

---

## 2. 数据结构

### 2.1 PosNodeRegistry

```python
@dataclass
class PosNode:
    pos: np.ndarray              # 3D 位置 (x, y, z)，已 snap 到导航网格
    step_index: int              # 记录时的全局步数
    vlm_names: List[str]         # VLM 识别出的物体名词列表（去重、小写）
    object_indices: List[int]    # 当时可见的 3D object_indices
    panorama_path: Optional[str] # 可选：保存全景图路径用于 debug

class PosNodeRegistry:
    nodes: List[PosNode]

    def add(self, node: PosNode) -> None: ...

    def query_co_occurrence(
        self, target_desc: str, anchor_desc: str
    ) -> List[PosNode]:
        """返回同时出现 target 和 anchor 的节点列表"""

    def query_single(self, desc: str) -> List[PosNode]:
        """返回出现 desc 所描述对象的节点列表"""

    def resolve_object_indices(
        self, node: PosNode, merge_tracker: MergeTracker
    ) -> List[int]:
        """将节点的 object_indices 通过 merge 历史展开为当前有效 indices"""
```

### 2.2 MergeTracker

RepresentationManager 在合并 object 时需要记录合并历史：

```python
class MergeTracker:
    # old_idx -> new_idx（合并链）
    merge_map: Dict[int, int]

    def record_merge(self, old_idx: int, new_idx: int) -> None: ...

    def resolve(self, idx: int) -> int:
        """返回 idx 当前对应的有效 object_index（跟踪合并链）"""

    def resolve_all(self, indices: List[int]) -> List[int]:
        """批量 resolve，去重，过滤已删除的"""
```

**说明**：合并发生时，old_idx 的所有 pos_node 记录的 object_indices 通过 resolve 后仍然有效。
查询时调用 `resolve_object_indices(node, merge_tracker)` 得到当前可用的 object_indices。

---

## 3. 算法流程

### 3.1 全景扫描节点更新（每 2 次 decision 触发）

```python
def update_panorama_node(
    agent_pos: np.ndarray,
    color_list: List[np.ndarray],   # 12 帧 360° 扫描图
    rep: RepresentationManager,
    registry: PosNodeRegistry,
    merge_tracker: MergeTracker,
    vlm_model: str,
    step_index: int,
) -> None:
    # 1. 拼接全景图
    panorama = stitch_panorama(color_list)  # 12 帧水平拼接

    # 2. VLM 识别物体
    prompt = (
        "List all distinct objects visible in this 360° panorama image. "
        "Return a JSON array of object noun phrases (with key modifiers like "
        "color/material/style if visible). Example: "
        "[\"round wooden table\", \"gray couch\", \"potted plant\"]"
    )
    raw = vlm_chat(panorama, prompt, max_tokens=128)
    vlm_names = parse_json_list(raw)  # List[str]

    # 3. 确定可见 object_indices
    visible_indices = get_visible_object_indices(agent_pos, rep)

    # 4. 存入 registry
    node = PosNode(
        pos=agent_pos.copy(),
        step_index=step_index,
        vlm_names=vlm_names,
        object_indices=visible_indices,
    )
    registry.add(node)
```

### 3.2 可见对象确定（get_visible_object_indices）

```python
def get_visible_object_indices(
    agent_pos: np.ndarray,
    rep: RepresentationManager,
    max_dist: float = 6.0,
) -> List[int]:
    box = np.asarray(rep.object_box, dtype=float)  # (M, 6)
    if box.shape[0] == 0:
        return []
    # 对象中心（nav 坐标）
    centers = box[:, :3].copy()
    centers[:, [1, 2]] = centers[:, [2, 1]]
    dists = np.linalg.norm(centers - agent_pos[None, :], axis=1)
    return [int(i) for i in np.where(dists < max_dist)[0]]
```

> **设计决策**：`max_dist=6.0m` 是初始值，需要根据场景大小调整。
> 也可以加上 FOV 锥体裁剪（只保留当前 heading ±90° 内的对象）来更精准，但实现更复杂。

### 3.3 VLM 节点查询（final decision 时）

```python
def query_registry_with_vlm(
    description: str,
    registry: PosNodeRegistry,
    vlm_model: str,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "mode": "co_occur" | "anchor_only" | "target_only" | "fallback",
          "matched_nodes": List[PosNode],
          "target_desc": str,
          "anchor_desc": str,
        }
    """
    # 1. 分解描述
    decomp = decompose_description(description, vlm_model)
    target_desc = decomp["target_desc"]
    anchor_desc = decomp["anchor_desc"]

    if not registry.nodes:
        return {"mode": "fallback", "matched_nodes": [], ...}

    # 2. 构建节点摘要传给 VLM
    node_summary = build_node_summary(registry)
    # 格式：
    # [node_0] pos=(3.1, 0.0, 2.4)  objects: round wooden table, decorative plant, couch
    # [node_1] pos=(1.2, 0.0, 5.6)  objects: mirror, white vanity, toilet
    # ...

    prompt = (
        f"Navigation task: find '{target_desc}' near '{anchor_desc}'.\n"
        f"Observation history:\n{node_summary}\n\n"
        "Q1: Which node indices show BOTH the target and the anchor together? "
        "(Return [] if none)\n"
        "Q2: Which node indices show the anchor alone (unique)? "
        "(Return [] if not found or multiple)\n"
        "Q3: Which node indices show the target alone (unique)? "
        "(Return [] if not found or multiple)\n"
        "Return strict JSON: "
        "{\"co_occur\": [...], \"anchor_only\": [...], \"target_only\": [...]}"
    )
    raw = vlm_chat_text(prompt, vlm_model, max_tokens=128)
    result = parse_json_obj(raw)

    co_occur = result.get("co_occur", [])
    anchor_only = result.get("anchor_only", [])
    target_only = result.get("target_only", [])

    if co_occur:
        return {"mode": "co_occur",
                "matched_nodes": [registry.nodes[i] for i in co_occur], ...}
    if len(anchor_only) == 1:
        return {"mode": "anchor_only",
                "matched_nodes": [registry.nodes[anchor_only[0]]], ...}
    if len(target_only) == 1:
        return {"mode": "target_only",
                "matched_nodes": [registry.nodes[target_only[0]]], ...}
    return {"mode": "fallback", "matched_nodes": [], ...}
```

### 3.4 从 top-K 中筛选（核心选择逻辑）

```python
def select_from_topk(
    topk: List[Tuple[int, float]],   # PQ3D top-K: [(obj_idx, score), ...]
    query_result: Dict[str, Any],
    merge_tracker: MergeTracker,
    rep: RepresentationManager,
) -> int:
    """返回最终选择的 object_index"""
    mode = query_result["mode"]
    matched_nodes = query_result["matched_nodes"]

    if mode == "fallback" or not matched_nodes:
        return topk[0][0]  # PQ3D top-1

    # 收集所有匹配节点的有效 object_indices
    candidate_set = set()
    for node in matched_nodes:
        resolved = merge_tracker.resolve_all(node.object_indices)
        candidate_set.update(resolved)

    # 在 top-K 中按原始排名找第一个属于候选集的
    for obj_idx, score in topk:
        if int(obj_idx) in candidate_set:
            return int(obj_idx)

    # top-K 里没有匹配的 → fallback
    return topk[0][0]
```

---

## 4. 描述分解（description decomposition）

```python
def decompose_description(description: str, vlm_model: str) -> Dict[str, str]:
    prompt = (
        "Decompose this navigation description into target and anchor.\n"
        "Rules:\n"
        "1. target: the object to navigate TO (full noun phrase with all modifiers)\n"
        "2. anchor: the most spatially distinctive nearby object "
        "(one noun phrase, empty string if none)\n"
        "Return strict JSON: {\"target_desc\": \"...\", \"anchor_desc\": \"...\"}\n\n"
        f"Description: {description}"
    )
    raw = vlm_chat_text(prompt, vlm_model, max_tokens=64)
    parsed = parse_json_obj(raw)
    return {
        "target_desc": str(parsed.get("target_desc", "")).strip(),
        "anchor_desc": str(parsed.get("anchor_desc", "")).strip(),
    }
```

---

## 5. 集成到主导航循环

```python
# 初始化
registry = PosNodeRegistry()
merge_tracker = MergeTracker()
panorama_decision_counter = 0

# 在主 while 循环中，每次 decision 后：
panorama_decision_counter += 1
if panorama_decision_counter % 2 == 0:
    update_panorama_node(
        agent_pos=agent_state.position,
        color_list=color_list,   # 当前 360° 扫描帧
        rep=pq3d_model.representation_manager,
        registry=registry,
        merge_tracker=merge_tracker,
        vlm_model=args.triquery_vlm_model,
        step_index=total_steps,
    )

# is_final 分支：
if is_final:
    baseline_final_target_pos = used_target.copy()

    query_result = query_registry_with_vlm(
        description=original_sentence,
        registry=registry,
        vlm_model=args.triquery_vlm_model,
    )

    topk = list(query_fn(main_target, top_k=16))  # PQ3D top-K
    chosen_idx = select_from_topk(topk, query_result, merge_tracker, rep)

    if chosen_idx != topk[0][0]:  # 实际发生了改变
        tri_used += 1
        chosen_xyz = rep.object_box[chosen_idx, :3].copy()
        chosen_xyz[[1, 2]] = chosen_xyz[[2, 1]]  # nav coord swap
        used_target = chosen_xyz
        final_selected_object_pos = used_target.copy()
    else:
        final_selected_object_pos = baseline_final_target_pos.copy()
        # 同 P9：同对象时用 PQ3D 原始坐标
        used_target = baseline_final_target_pos.copy()
```

---

## 6. MergeTracker 与 RepresentationManager 的集成

需要在 RepresentationManager 的合并逻辑中注入回调：

```python
# 在 RepresentationManager.update() 或 merge_objects() 中：
if merge_happens:
    merge_tracker.record_merge(old_idx=absorbed_obj, new_idx=survivor_obj)
```

如果 RepresentationManager 不方便直接修改，可以在每次 `update_panorama_node` 时，
通过比对前后 object_box shape 变化来推断合并，但这不够准确。
**建议**：在 RepresentationManager 里暴露一个 `on_merge` 回调接口。

---

## 7. 关键设计决策（待确认）

| 问题 | 选项A | 选项B | 建议 |
|------|-------|-------|------|
| 全景图拼接方式 | 12 帧水平直接拼接（简单） | 球形投影拼接（准确） | 先用A，验证效果 |
| 可见对象判定 | 距离阈值（6m）| 距离+FOV 锥体 | 先用距离阈值 |
| VLM 节点查询输入 | 纯文本节点摘要 | 纯文本 + 全景图（多模态） | 先用纯文本，降低 API 成本 |
| 节点更新频率 | 每 2 次 decision | 每次 final decision | 每 2 次，保证覆盖率 |
| 节点数量上限 | 无上限 | 保留最近 N 个 | 先无上限（episode 较短） |
| 描述分解时机 | final decision 时实时调用 | 任务开始时预计算 | 任务开始时预计算，避免 final 时延迟 |

---

## 8. 与 TQSI 的对比

| 维度 | TQSI | Panorama Node |
|------|------|---------------|
| 锚点信号来源 | PQ3D logit 分布（间接） | 真实视觉观测（直接） |
| 空间约束机制 | 高斯距离软权重 | 集合过滤（硬约束） |
| 语义匹配方式 | 向量相似度 | VLM 自然语言理解 |
| 对象合并处理 | 无 | 显式 merge tracking |
| 在 top-K 以外发现目标 | 不能 | 不能（仍依赖 PQ3D top-K） |
| 计算开销 | 每次 final 1 次 VLM | 每 2 decision 1 次 VLM + final 1 次 |
| 实现复杂度 | 低 | 中（需 MergeTracker + PosNodeRegistry） |

---

## 9. 开放问题

1. **RepresentationManager 的合并机制**：当前代码里合并逻辑在哪？是否可以加 on_merge 回调？
2. **panorama_decision_counter 重置**：每个 task 开始时 registry 和 tracker 是否清空？（建议：每个 episode 清空，每个 task 保留——同一 episode 内的观测对后续 task 有参考价值）
3. **节点摘要长度**：如果 episode 很长（20+ 节点），文本摘要可能超过 VLM context 限制，需要截断策略（优先保留最近节点）
4. **VLM 识别精度**：全景图中远处小物体 VLM 可能漏识别或错误命名，需要在实验中验证覆盖率



