# RefHM3D Baseline 导航机制详解

本文档总结 `hm3d-online/refhm3d-nav-sequence.py`（及其依赖 `hm3d-online/data_utils.py`, `hm3d-online/merge_utils.py`）的导航机制，聚焦以下问题：

- 一个 `scene/episode/task` 的执行流程是什么
- 何时触发 decision，何时结束一个 task
- 环视与输入帧如何组织（为什么常见 12 或 18 帧）
- object 与 frontier 候选如何构建与选择
- memory（对象库）如何 merge
- logits 表征什么含义
- 为什么会出现“接近目标却未结束并再次跑远”的失败现象

---

## 1. 数据层级与任务单位

在 RefHM3D LangMap 标注中，常用层级如下：

- **scene**：一个房屋场景（一个 `.json.gz` 标注文件 + 一个 3D 场景资产）。
- **episode**：某个 scene 里的一条 sequence 回合（包含起点位姿与 5 个子任务）。
- **task（sub-task）**：episode 内的单个子目标（object / room / region / instance）。

baseline 的外层评测按 scene->episode->task 迭代；每个 task 内部再进行多轮“观测-决策-移动”循环。

---

## 2. 单个 task 的主循环

在 `refhm3d-nav-sequence.py` 中，task 的核心循环是：

1. 组装本轮输入帧（上一段 goto 帧 + 本轮环视帧）
2. 更新探索图（fog of war）
3. 计算 frontier 候选点
4. 调用 `pq3d_model.decision(...)` 得到 `target_position` 与 `is_final_decision`
5. 用 `GreedyGeodesicFollower` 走向 `target_position`
6. 若 `is_final_decision == True` 则结束该 task；否则继续下一轮

循环上限通常是 `total_steps < 400`。

---

## 3. 什么时候触发 decision

**每一轮“环视完成并算完 frontier 后”触发一次 decision**，不是每一步动作都决策。

因此这是一个周期性决策结构：

- 观察一轮（含环视）
- 规划一次（decision）
- 执行动作一段（goto）
- 再进入下一轮

---

## 4. 环视与输入帧组织（12 / 18 的来源）

每轮 decision 的输入帧由两部分组成：

1. **goto 历史帧**：来自“上一轮决策后执行动作”的缓存帧，最多采样 6 帧  
2. **本轮环视帧**：固定 `turn_left` 12 次，得到 12 帧（360 度）

所以每轮输入帧数通常是：

- 仅环视：12 帧
- goto + 环视：最多 18 帧

### 关于“最多采样 6 帧”

- 只对**上一段 goto**缓存进行采样，不跨多轮累计。
- 若 `len(goto_frames) > 6`，按固定步长近似均匀采样后截取 6 帧。

---

## 5. frontier 是什么，如何得到

frontier 不是物体，而是“已探索区域与未探索区域的边界探索点”。

流程：

1. 环视/移动时调用 `reveal_fog_of_war` 更新探索掩码；
2. 调 `detect_frontier_waypoints(top_down_map, fog_of_war_mask, ...)` 得到边界像素候选；
3. 用 `pixel_to_map_coors` 转为世界坐标；
4. 过滤 `visited_frontier_set`（已访问过的 frontier 不再重复）。

这些 frontier 点是“探索候选”。

---

## 6. object 候选如何得到（感知到对象）

`PQ3DModel.decision` 中 object 流程（高层）：

1. 对多帧 RGB 提取 DINO 特征；
2. 对多帧运行 FastSAM 生成大量 mask proposals；
3. 利用 depth + pose 将像素投影到 3D 点云；
4. 进入 PQ3D stage1（实例/框/特征预测）；
5. 结果送入 `RepresentationManager.merge(...)` 融合到全局对象库（memory）。

因此 FastSAM 日志里的“几十/上百 objects”是 proposal 数，不等于最终 memory 中对象数。

---

## 7. memory（对象库）merge 机制

对象库在 `RepresentationManager` 中维护（如 `object_box/object_feat/object_score/open_vocab_feat`）。

每批 stage1 结果 merge 时大致包括：

1. 单帧候选筛选（TopK、NMS、score 阈值、最小点数阈值）；
2. 与历史对象做匹配（核心是 3D box IoU，匈牙利匹配 + 阈值）；
3. 匹配到的对象做增量更新（mask 合并、score/box/feat 运行平均）；
4. 未匹配对象作为新对象加入库；
5. 超过全局上限时按 score 截断（例如 topk_objects）。

> 注意：frontier 不进入这个对象 merge；frontier 每轮重算，属于临时候选。

---

## 8. decision 阶段如何“在 object 与 frontier 中选”

在 stage2（VLE）里，会把两类候选拼在一起：

- 真实 object 候选（来自 memory）
- frontier 候选（当前轮边界点）

模型输出包含两类关键分数：

1. **`decision_logits`**：对候选点的未归一化打分（用于选具体哪个点）
2. **`obj_frontier_decision_logits` -> `goto_frontier_probability`**：object vs frontier 的门控倾向

随后按规则决定：

- 这轮更偏 object 还是 frontier
- 在对应集合内取分数最高候选
- 返回 `target_position`

---

## 9. logits 的含义

logits 是模型内部“偏好分数”，不是概率本身：

- 候选间可比较大小，值越大代表相对更偏好
- 经过 softmax 才得到概率解释
- 不能直接等价为“真实成功率”或“距离误差”

在该 baseline 里，logits 决定“这轮去哪里”。

---

## 10. task 结束与成功判定

### 循环结束条件

- decision 返回 `is_final_decision == True`（语义上更接近“当前选择 object”）
- 或达到步数上限（如 400）

### 评测成功（SR/SPL）

任务结束后，脚本计算：

- 起点到目标视点集合的测地距离
- 终点到目标视点集合的测地距离
- `SR = (end_goal_geo <= success_threshold)`（常见阈值 0.25m）
- `SPL` 按标准公式计算

---

## 11. 常见失败模式（已在分析中观测到）

典型现象：曾非常接近目标（例如 <0.1m）但未停止，随后又被带离目标，最终超步数失败。

可能机制原因：

1. “是否 final”与“是否已到达目标”没有显式几何停机约束；
2. 文本描述过长、包含多锚点，导致 object/frontier 门控振荡；
3. frontier 与 object 的竞争在临界状态下不稳定，易回到探索路径。

---

## 12. 你可直接观测的诊断信号

在分析脚本输出（`output_process/...`）中重点看：

- `logs/trace.jsonl`：每轮 `cur_goal_geo`, `is_final_decision`, `target_position`, `memory_objects`, `num_frontiers`
- `decisions/dec_xxx/decision_stats.json`：模型侧统计（若可用）
- `decisions/dec_xxx/topdown_pre/post.png`：候选与目标点在地图上的变化

---

## 13. 一句话总结

这个 baseline 本质是：

> 用多帧 RGB-D 构建并维护对象 memory，同时每轮重算 frontier；再由 VLE 在“object 候选 vs frontier 候选”中做联合决策，输出目标点并执行导航。

