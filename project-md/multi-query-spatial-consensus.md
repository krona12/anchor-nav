# Multi-Query Spatial Consensus for MTU3D Decision

日期: 2026-05-13  
方法名: **MQSC-MTU3D: Multi-Query Spatial Consensus for Fine-Grained Decision Refinement**  
目标: 审查“完整 task 与分解 target/anchor 分别做 decision 查询 top8，再做 2D 空间聚类，寻找最可能地域并选择该地域内最高置信目标”的想法。

## 0. 一句话结论

这个 idea 是可实现的，而且比完整 RSCI 图优化更轻量。它的本质是:

> 用 MTU3D 自身的多次 text-query grounding 结果，形成多个 noisy views；如果 full task、target、anchors 的 top candidates 在同一个 2D 空间区域形成共识，则该区域比单个最高语义分 object 更可信。

但需要一个关键修正:

> 不能简单把所有 top8 object center 丢进 DBSCAN，然后取 cluster 里最高分 object。必须做 **role-aware region consensus**：不同 query 代表不同角色，anchor 与 target 不应被看作同一位置，而应被看作同一局部关系区域内的互相支持证据。

因此推荐方法是:

```text
Multi-query decision logits
  -> role-wise top-k candidates
  -> 2D footprint-aware clustering
  -> region posterior / coverage / compactness / relation feasibility
  -> choose best region
  -> choose best target-role object inside that region
  -> if uncertainty high, explore frontier near that region
```

## 1. 原始想法复述

给定长指令:

```text
找到卧室里白色床旁边的床头柜上的台灯。
```

构造多组 query:

```text
q_full   = "找到卧室里白色床旁边的床头柜上的台灯"
q_target = "台灯 / lamp"
q_anchor1 = "白色床 / white bed"
q_anchor2 = "床头柜 / nightstand"
q_room = "卧室 / bedroom"
```

每个 query 单独调用 MTU3D decision / stage2 grounding，取 real object top8:

```text
Top8(q_full), Top8(q_target), Top8(q_anchor1), Top8(q_anchor2), Top8(q_room)
```

将这些候选投影到 2D top-down 平面，做聚类，寻找多类候选共同聚集的区域。最终在最高置信区域中选择最可能的 target object。

这个思想的优点:

- 不需要训练；
- 不需要重写 MTU3D；
- 可以复用 stage2 的语言 grounding 能力；
- 比只用完整长指令更稳，因为 anchor/target 被显式激活；
- 比完整 factor graph 更容易实现。

## 2. 这个问题的本质

### 2.1 不是 object-level top-k，而是 region-level posterior

单 query 选 object 是:

```text
o* = argmax_o P(o | q_full)
```

MQSC 想做的是:

```text
R* = argmax_R P(R | q_full, q_target, q_anchor1, q_anchor2, q_room)
```

然后:

```text
o_target* = argmax_{o in R*} P(o is target | R*, queries)
```

也就是先找最可信区域，再在该区域内找最终目标。

### 2.2 为什么区域比单点更可信

如果一个 `lamp` 语义分很高，但它附近没有 bed/nightstand/bedroom evidence，它可能是错误候选。反过来，一个 `lamp` 分数中等，但它与 `nightstand` 和 `white bed` 处于同一个 bedside region，它更符合完整任务。

这就是 spatial consensus:

```text
可信目标 = 语义像目标 + 附近存在 anchor + anchor 间关系合理 + 区域上下文合理
```

## 3. 直接 DBSCAN top8 center 的问题

### 3.1 Anchor 与 target 不一定同中心

`lamp on nightstand next to bed` 中:

- lamp 和 nightstand 很近；
- bed 的中心可能距离 nightstand 1-2 米；
- 如果直接用 object center 聚类，bed 可能被分到另一个 cluster；
- 但从任务语义上，它们属于同一个 bedside region。

因此聚类应该使用 **2D footprint distance** 或 **expanded anchor region**，而不是裸 center distance。

### 3.2 大物体会扭曲聚类

bed、table、sofa 这类大物体中心不代表关系发生位置。床头柜在床边，不在床中心。应使用:

```text
distance_to_footprint(anchor_box)
```

而不是:

```text
distance_to_center(anchor_box)
```

### 3.3 不同 query 的 top8 置信度不可直接比较

`q_target` 的 softmax 分布可能很尖，`q_room` 可能很平。不同 query 的 logits 需要 query-wise normalize:

```text
p_q(o) = softmax(logit_q(o) / T_q)
```

再进入 region posterior。

### 3.4 如果只取 cluster 内最高 object，可能选到 anchor

最高分 object 可能是 bed，而不是 lamp。最终选择必须 role-conditioned:

```text
choose highest target-role object inside best region
```

如果 best region 没有足够 target evidence，不应该 stop，而应该探索该 region。

## 4. MQSC-MTU3D 算法

### 4.1 Stage A: 多 query 生成

从 instruction graph 或轻量 parser 得到:

```python
QuerySpec = {
    "full": sentence,
    "target": "lamp, table lamp, bedside lamp",
    "anchor_primary": "white bed",
    "anchor_support": "nightstand, bedside table, side table",
    "room_context": "bedroom, bed, pillow, wardrobe, dresser"
}
```

推荐 query 数量:

- 必选: full, target。
- 细粒度任务: anchor_primary, anchor_support。
- room 约束明显时: room_context。

不要无限扩展 query，否则会增加噪声。

### 4.2 Stage B: 重复调用 MTU3D Stage2

当前 `PQ3DModel.decision()` 在 `data_utils.py:421-496` 构建一次 stage2 batch 并 forward。MQSC 第一版可以把 stage2 forward 包成函数:

```python
def run_stage2_for_text(query_text, obj_tensors, frontier_tensors):
    encoded_input = self.tokenizer([query_text], add_special_tokens=True, truncation=True)
    data_dict["prompt"] = torch.FloatTensor(encoded_input.input_ids[0])
    data_dict["prompt_type"] = PromptType.TXT
    with torch.no_grad():
        return self.pq3d_stage2(batch)
```

对每个 query 得到:

```python
og3d_logits_q = output["og3d_logits"][0]
real_logits_q = og3d_logits_q[real_obj_pad_masks]
frontier_logits_q = og3d_logits_q[~real_obj_pad_masks]
```

对 real objects 取 top8:

```python
probs_q = softmax(real_logits_q / temperature_q)
top8_q = topk(probs_q, k=8)
```

注意:

- stage1 / DINO / SAM / merge 不需要重复跑；
- 只重复 stage2，开销相对可控；
- 第一版只聚类 real objects，不聚类 frontier。

### 4.3 Stage C: 2D footprint-aware candidate points

MTU3D 内部 object box 是模型坐标 `[x, z, y, dx, dz, dy]`。因此 top-down 2D 应使用:

```python
xy = object_box[:, [0, 1]]
```

不要用 `[0, 2]`，因为第 2 维是垂直方向。

每个候选:

```python
CandidateHit = {
    "object_id": int,
    "role": "full | target | anchor_primary | anchor_support | room_context",
    "query": str,
    "prob": float,
    "logit": float,
    "xy": [x, z],
    "footprint_radius": 0.5 * max(dx, dz),
    "box": [x, z, y, dx, dz, dy]
}
```

### 4.4 Stage D: 构建 2D 区域聚类

推荐第一版用 radius graph / DBSCAN:

```python
distance(i, j) =
    max(0, ||xy_i - xy_j|| - radius_i - radius_j)
```

也就是考虑物体 footprint。对于大物体 anchor，允许附近对象被吸入同一 relation region。

DBSCAN 参数建议:

```text
eps = 0.8m ~ 1.5m
min_samples = 2
sample_weight = candidate_prob
```

更轻实现:

```text
如果 distance(i,j) < eps，则连边；
cluster = connected components。
```

这种 connected-components 比 DBSCAN 更容易控制，也更适合几十个候选的小规模场景。

### 4.5 Stage E: Region posterior

对每个 region `R`，定义每个 query/role 在该 region 内的证据:

```text
p_q(R) = max_{o in R} p_q(o)
```

也可以用 noisy-or:

```text
p_q(R) = 1 - Π_{o in R} (1 - p_q(o))
```

推荐 noisy-or，因为一个 region 内多个中等候选也应形成证据。

Region posterior:

```text
Score(R) =
  Σ_q w_q log(ε + p_q(R))
  + λ_cov Coverage(R)
  + λ_comp Compactness(R)
  + λ_rel RelationFeasibility(R)
  + λ_stab Stability(R)
  - λ_noise NoisePenalty(R)
```

其中:

```text
Coverage(R) = number of required roles represented in R / number of required roles
```

```text
Compactness(R) = exp(-trace(cov_R) / σ_cluster^2)
```

```text
Stability(R) = mean object_count or repeated-view support
```

```text
NoisePenalty(R) = many hits from same role but missing other roles
```

这里的核心不是“线性加权 object 分数”，而是把 region 当作 latent variable，计算多 query 对同一区域的共识。

### 4.6 Stage F: Relation feasibility within region

如果只做聚类，仍然可能把“床、椅子、台灯”混在一起。需要在 region 内做轻量关系检查:

```text
RelationFeasibility(R) =
  max_assignment over objects in R:
    P(nightstand next_to bed)
    * P(lamp on nightstand)
```

第一版关系检查可只用几何:

```text
P(next_to(a,b)) = exp(-d_footprint(a,b)^2 / 2σ_next^2)
```

```text
P(on(a,b)) =
  exp(-d_xy(center(a), footprint(b))^2 / 2σ_xy^2)
  * exp(-(bottom_z(a)-top_z(b))^2 / 2σ_z^2)
  * support_overlap(a,b)
```

如果 relation feasibility 低，即使 region coverage 高，也不应 final stop。

### 4.7 Stage G: 选择 region 内 target object

选出:

```text
R* = argmax_R Score(R)
```

然后只在该 region 内选择 target-role object:

```text
o* = argmax_{o in R*} TargetScore(o | R*)
```

TargetScore:

```text
TargetScore(o | R*) =
  p_target(o)^α
  * p_full(o)^β
  * relation_support(o, R*)^γ
  * object_score(o)^η
```

如果 `o*` 不存在或置信度不足:

```text
status = explore_region
```

不应该把 region 内最高分 anchor 当成最终 target。

## 5. 与 MTU3D decision 的接口

### 5.1 插入位置

推荐插在:

- `data_utils.py:498` 获取 `decision_logits` 后；
- `data_utils.py:516` baseline final decision 前。

因为此时已经有:

```python
decision_logits
real_obj_pad_masks
query_locs
real_object_decision_logits
frontier_decision_logits
real_object_locs
frontier_locs
```

但 MQSC 还需要每个 decomposed query 的 logits，因此需要复用 stage2 forward。

### 5.2 推荐 refactor

把 stage2 输入构建拆成:

```python
stage2_base = self._build_stage2_base(
    query_feat,
    query_box,
    query_scores,
    obj_openvocab_feat,
    frontier_list,
)
```

再写:

```python
stage2_output = self._run_stage2_with_prompt(stage2_base, query_text, image_feat=None)
```

MQSC 调用:

```python
mqsc_result = self.mqsc_refiner.refine(
    query_specs=query_specs,
    run_stage2_fn=lambda text: self._run_stage2_with_prompt(stage2_base, text),
    object_boxes=query_box,
    object_scores=query_scores,
    object_counts=self.representation_manager.object_count,
    open_vocab_feats=obj_openvocab_feat,
    frontier_locs=frontier_locs,
    baseline_logits=decision_logits,
    baseline_real_logits=real_object_decision_logits,
    baseline_frontier_logits=frontier_decision_logits,
    agent_position=agent_position,
)
```

返回:

```python
MQSCResult = {
    "status": "stop | explore | fallback",
    "target_object_idx": int | None,
    "target_position_model_coord": np.ndarray | None,
    "frontier_idx": int | None,
    "frontier_position_model_coord": np.ndarray | None,
    "best_region": dict,
    "region_scores": list,
    "target_confidence": float,
    "region_margin": float,
    "reason": str
}
```

### 5.3 最终覆盖规则

MQSC 不应总是覆盖 baseline。

覆盖 baseline object stop:

```text
if baseline wants object
and MQSC best region lacks target/anchor relation:
    reject baseline stop, explore region/frontier
```

覆盖 baseline frontier:

```text
if baseline wants frontier
and MQSC region has high confidence target object:
    stop at target
```

fallback:

```text
if MQSC region_margin low or required roles missing:
    use baseline or explore highest information frontier
```

