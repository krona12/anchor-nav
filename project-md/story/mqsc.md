# MQSC 方法故事

MQSC = Multi-Query Spatial Consensus

一句话版本：

> MQSC 不把一句复杂导航指令当成一个孤立 query 去猜目标，而是像人一样把它拆成“我要找什么、它在哪里、旁边有什么、属于哪个房间”，再看这些线索是否在同一片空间里互相印证；只有当 target 与 anchors 在局部区域形成共识时，才把该区域里的目标物体交给导航器。

## 1. 人找东西时不是只问一次

一个人在房间里找东西，很少只在脑中重复完整句子：

```text
找到浴室里，白色边框镜子下方浅色洗手台旁、厕所上有蓝色画、白色簇绒浴帘、灰白毛巾旁边的 shower curtain。
```

真实的认知过程更像这样：

```text
我要找的主角是什么？shower curtain。
它大概在哪个场景？bathroom。
有哪些附近线索？vanity、mirror、toilet、blue picture、towels。
这些东西是否在同一个角落？如果是，这一片区域就可信。
在这片区域里，哪个东西最像 shower curtain？选它。
```

也就是说，人类不会把长描述压成一个单一语义向量，然后直接选最高分物体。人类会把描述拆成多个注意力探针，再把探针结果放回空间里对齐。

MQSC 想模拟的正是这件事：

```text
language attention is noisy,
but spatial agreement across multiple attentions is reliable.
```

## 2. Baseline 的隐含假设

传统 PQ3D / MTU3D 导航流程可以概括为：

```text
完整语言描述 -> 单次 grounding -> top-1 object / frontier -> follower
```

这个流程背后有一个强假设：

```text
完整句子的 top-1 语义分最高对象，就是最终目标。
```

但 RefHM3D 这类任务里的描述经常是细粒度指称。它不是只说“找桌子”，而是说：

```text
找卧室里，床旁边、白色床头柜附近、灯下面的那张桌子。
```

此时完整句子的 grounding 可能会被某个显著 anchor 吸走。模型可能选中 bed、nightstand、mirror、picture、toilet，而不是 target。问题不一定是模型完全不懂语言，而是长指令里有多个对象角色，单次 query 把这些角色混在了一起。

MQSC 的判断是：

```text
单个 top-1 object 不稳定；
多个角色 query 在同一空间区域的共识更稳定。
```

## 3. 核心认知假设：先确定局部场景，再确定目标

人在找东西时通常不是先锁定一个点，而是先锁定一片局部场景。

例如：

```text
lamp on the nightstand next to the bed
```

人不会要求 lamp、nightstand、bed 的几何中心落在同一个点上。人会形成一个更松的空间概念：

```text
这是床边那一片 bedside region。
```

然后再在 bedside region 里找 lamp。

MQSC 因此把决策顺序从 object-first 改成 region-first：

```text
baseline:
  object* = argmax_object P(object | full_instruction)

MQSC:
  region* = argmax_region P(region | full, target, anchors, room)
  object* = argmax_target_object P(object is target | region*)
```

这个变化很关键。MQSC 不问“哪个 object 分最高”，而是先问：

```text
哪一片区域同时解释了目标、房间和上下文线索？
```

再问：

```text
这片区域里，哪个 object 是 target 角色？
```

## 4. 从一句话到多个角色 query

MQSC 的第一步是把自然语言描述分解成角色。

在实现中，对应 `hm3d-online/anchor_nav/mqsc.py` 里的：

```python
decompose_navigation_text(...)
build_role_queries(...)
```

分解结果大致是：

```text
full:
  原始完整描述

target:
  真正要找的目标物体

anchor_primary:
  与 target 关系最强的主锚点

anchor_support:
  支撑局部语境的辅助锚点

room_context:
  房间或区域上下文
```

例如：

```text
shower curtain in the bathroom that has light-colored bath vanity below white framed mirror,
blue picture on toilet, white tufted shower curtain, gray towel beside white towel
```

可以拆成：

```text
full:
  整句

target:
  shower curtain

anchor_primary:
  light-colored bath vanity below white framed mirror
  blue picture on toilet

anchor_support:
  gray towel beside white towel

room_context:
  bathroom
```

这里的重点不是让 VLM 生成一个完美 scene graph。MQSC 只需要一个足够稳定的角色分解，让后续多 query grounding 能分别激活不同线索。

当 VLM 可用时，MQSC 使用 prompt 让模型输出 JSON 结构；当 VLM 不可用时，使用 heuristic fallback 保证模块不会中断导航。

## 5. 多 query 是多次“注意力扫视”

同一个空间记忆中，MQSC 会用不同 query 重新调用 MTU3D Stage2 object grounding。

实现中对应：

```python
pq3d_stage2_object_logits(pq3d_model, query_text)
```

注意这里没有重新跑 stage1、SAM、DINO 或点云融合。MQSC 复用当前 `PQ3DModel.representation_manager` 中已经建立好的 object memory，只是对不同文本重复跑轻量的 Stage2 grounding。

每个 query 得到一组 object logits：

```text
logits_full(object_i)
logits_target(object_i)
logits_anchor(object_i)
logits_room(object_i)
```

然后按 query 内 softmax 归一化，取 top-k：

```text
TopK(full), TopK(target), TopK(anchor_primary), TopK(anchor_support), TopK(room_context)
```

这相当于让模型做多次注意力扫视：

```text
“如果我只想看 shower curtain，哪里像？”
“如果我只想看 bathroom，哪里像？”
“如果我只想看 vanity and mirror，哪里像？”
“如果我只想看 towel，哪里像？”
```

单次扫视可能错，但多次扫视如果在同一区域相遇，这个区域就开始可信。

## 6. 为什么不是简单投票

MQSC 不能简单地把所有 top-k object 放在一起投票。原因是不同 query 的角色不一样。

target query 应该帮助选择最终物体：

```text
target -> 哪个 object 是主角
```

anchor query 应该帮助选择区域：

```text
anchor -> 哪片空间上下文可信
```

room query 更像低频背景约束：

```text
room_context -> 这片空间是否属于正确房间
```

所以 MQSC 采用 role-aware consensus，而不是 naive majority vote。

一个 bed 可能因为 bedroom query 得分很高，但它不能替代 lamp。一个 mirror 可能因为 bathroom query 得分很高，但它不能替代 shower curtain。MQSC 允许 anchor 支持区域，但最终必须回到 target-role object。

## 7. 从 object center 到 footprint-aware region

人判断“床旁边的床头柜”时，不会只看床的几何中心。床很大，关系发生在床边，而不是床中心。

所以 MQSC 做空间聚类时不用裸 object center 距离，而是使用 footprint-aware distance。

实现中对应：

```python
footprint_xy_and_radius(...)
_footprint_distance(...)
_connected_components(...)
```

MTU3D 的 object box 在模型内部是：

```text
[x, z, y, dx, dz, dy]
```

MQSC 投影到 top-down 平面时使用：

```text
xy = [x, z]
```

每个物体还有一个 footprint radius，用来近似它在平面上的占地范围。两个物体的空间距离不是中心到中心，而是：

```text
distance(center_i, center_j) - radius_i - radius_j
```

这样 bed、table、sofa 这类大物体不会因为中心偏远而被错误排除。它们的边界可以和 target/anchor 形成局部关系。

## 8. 区域共识：找一片能解释所有线索的地方

MQSC 把所有 query 的 top-k 候选投影到 2D 平面，然后用 footprint-aware 连接规则形成 connected components。

每个 component 就是一片候选区域：

```text
region = {objects that are spatially connected by footprint distance}
```

对每个 region，MQSC 计算：

```text
role_evidence:
  full evidence
  target evidence
  anchor evidence
  room evidence

coverage:
  required roles 中有多少被该 region 覆盖

compactness:
  区域是否紧凑
```

在代码里，region score 大致来自：

```text
score(region)
  = weighted log role_evidence
  + coverage bonus
  + compactness bonus
```

其中 role evidence 使用 noisy-or 聚合：

```text
P(role appears in region) = 1 - Π(1 - P(role appears on object_i))
```

这和人的直觉很像：如果一个区域里有多个物体都支持“bathroom”或“bedroom”，那么这片区域属于该房间的可信度会上升。

## 9. 目标选择：区域里的主角，而不是区域里的最高分物体

确定 best region 后，MQSC 不直接选 region 里总分最高的 object。它只在 target-role pool 里选：

```text
target_pool = objects in best_region that appeared in target query top-k
```

然后计算 target score：

```text
target_score(object)
  = target_prob
  + full_prob
  + relation_support
  + merged_object_score
  + object_count
```

这里 `relation_support` 表示 target 与 anchor/full/room 角色物体在 footprint 空间上的相互支持。一个 target 如果附近有合理 anchor，它会比孤零零的高分 target 更可信。

这对应人类判断里的最后一步：

```text
我已经确认这是正确的一片局部场景；
现在我只在这片场景里找“我要找的那个东西”。
```

## 10. 为什么 MQSC 是一种纠偏，而不是重写导航器

MQSC 插在 baseline 的 final decision 后面。

在批跑脚本中，对应：

```text
hm3d-online/refhm3d-nav-sequence-analyze-anchor-mqsc-refine1.py
```

主流程仍然是：

```text
frontier exploration
PQ3D decision
if final:
  MQSC refine hook
follower
metrics
```

MQSC 的入口是：

```python
run_mqsc_refine(
    sentence=sentence,
    task_type=task_type,
    pq3d_model=pq3d_model,
    target_position=baseline_target,
    cfg=MQSC_CFG,
    output_dir=task_dir,
    decision_num=decision_num,
    context={...},
)
```

它返回：

```text
new_target_xyz, module_info
```

如果 MQSC 没有足够证据，它会保持 baseline target 不变：

```text
applied = false
reason = selected_matches_baseline / threshold_fail / no_regions / ...
```

如果 MQSC 找到更可信的 target，它才替换：

```text
applied = true
reason = mqsc_selected_higher_consensus_target
target_after = selected_target_position
```

所以 MQSC 是保守 refinement，而不是完全替代 baseline。

## 11. 接口与日志：让每一次纠偏可审计

MQSC 的一个设计原则是：不能黑箱纠偏。

每次 final decision 都会写出模块日志，核心字段包括：

```text
decomposition
queries
query_summaries
candidate_hits
regions
best_region
selected_candidates
baseline_object_index
selected_object_index
selected_gain
target_before
target_after
applied
reason
```

这些字段可以回答三类问题：

```text
1. 语言是否被正确拆分？
2. 每个 query 的 top-k 候选是否合理？
3. 最终纠偏是否真的比 baseline 更接近目标？
```

在当前批跑脚本里，summary 还记录：

```text
baseline_target_to_goal_l2
selected_target_to_goal_l2
module_hook_called
module_hook_applied
module_helpful
```

这让 MQSC 的每一次修正都能被量化，而不只是看最终 SR/SPL。

## 12. 一个认知视角的例子

指令：

```text
bottom black-framed line drawing picture on gray wall between two others
in bedroom with light gray bedding and red bookshelf
```

如果只问完整句子，模型可能被 `bedroom`、`bedding`、`bookshelf`、`picture` 中任意一个显著线索影响。

MQSC 会拆成：

```text
target:
  bottom black-framed line drawing picture

anchor_primary:
  gray wall
  two other pictures

anchor_support:
  light gray bedding
  red bookshelf

room_context:
  bedroom
```

然后它期待看到这样的空间形态：

```text
picture candidates 在一面墙附近；
其他 picture / gray wall 线索也在附近；
bedroom 线索在同一局部房间区域；
bookshelf 和 bedding 提供房间级验证，但不抢 target 身份。
```

如果一个 picture 分数很高，但它远离这些 anchors，MQSC 会降低这片区域的可信度。

如果另一个 picture 分数稍低，但它与 gray wall、two other pictures、bedroom context 一起出现，MQSC 会认为它更像人类要找的那个指称对象。

## 13. MQSC 与 VISTA-LS 的边界

MQSC 和 VISTA-LS 都是在修 baseline，但修的是不同层次。

MQSC 修的是：

```text
语义选择问题：到底哪个 object 是目标？
```

VISTA-LS 修的是：

```text
可执行站位问题：选中目标后，agent 应该站在哪里？
```

所以二者是互补关系：

```text
MQSC:
  choose the right semantic target by multi-query spatial consensus

VISTA-LS:
  choose the right executable viewpoint around that target
```

从人类认知角度看：

```text
MQSC 像“我确认这就是那件东西”；
VISTA-LS 像“我应该走到哪里才能看清它”。
```

## 14. 方法的简短伪代码

```python
def mqsc_refine(sentence, task_type, pq3d_model, baseline_target):
    query_spec = decompose_navigation_text(sentence, task_type)
    queries = build_role_queries(sentence, query_spec)

    hits = []
    for query in queries:
        logits = pq3d_stage2_object_logits(pq3d_model, query.text)
        probs = softmax(logits)
        hits.extend(top_k_objects(probs, role=query.role))

    xy, radius = footprint_xy_and_radius(memory_object_boxes)
    regions = connected_components(hits.object_ids, xy, radius)

    scored_regions = []
    for region in regions:
        role_evidence = aggregate_role_evidence(region, hits)
        coverage = count_covered_roles(role_evidence)
        compactness = spatial_compactness(region)
        scored_regions.append(score(region, role_evidence, coverage, compactness))

    best_region = max(scored_regions)
    target_pool = objects_with_target_role(best_region)

    selected = max(target_pool, key=target_score_with_relation_support)

    if confidence_thresholds_pass(selected, best_region, baseline):
        return selected.center_habitat_xyz, applied=True

    return baseline_target, applied=False
```

## 15. 这套方法想靠近的人类能力

MQSC 不是试图让模型“更聪明地读懂一句话”。它更具体：

```text
它让模型像人一样，把一句话中的多个线索分别拿出来看，
再在空间中寻找它们是否指向同一片地方。
```

人类在真实环境里找东西时依赖三种稳定性：

```text
1. 角色稳定性：
   target 和 anchor 不混淆。

2. 空间稳定性：
   多个线索必须能在同一局部区域里解释。

3. 决策稳定性：
   只有当区域和目标都足够可信时，才替换原始目标。
```

MQSC 的方法贡献可以概括为：

> 把细粒度语言指称从单次 object grounding，改写成多角色 query 的空间共识问题；先找到能解释多条语言线索的局部区域，再在该区域内选择 target-role object，从而减少长描述中 target/anchor 混淆导致的导航错误。

## 16. 画方法图的 Prompt

下面这段可以直接丢给 GPT / 图像生成模型 / 绘图助手，用来生成 MQSC 的方法示意图。目标是画成论文方法图，而不是宣传海报。

```text
请绘制一张论文风格的 method figure，主题为:

MQSC: Multi-Query Spatial Consensus for Fine-Grained 3D Navigation Decision Refinement

整体画布:
- 横向 4-panel pipeline，从左到右。
- 风格是清晰的学术论文方法图，白底，细线框，少量颜色编码，不要卡通化，不要复杂背景。
- 图中必须呈现“语言角色分解 -> 多 query grounding -> footprint-aware spatial consensus -> target-role object selection”的流程。
- 使用 top-down 2D map 作为核心视觉元素，表现 objects、anchors、regions、selected target。

Panel A: Language Role Decomposition
- 左侧输入一条长指令 I:
  I = "shower curtain in the bathroom with vanity, mirror, toilet, towels"
- 从 I 分解出多个 role-aware queries:
  q_full = full instruction
  q_target = target phrase
  q_anchor_primary = main relational anchors
  q_anchor_support = supporting anchors
  q_room = room context
- 用不同颜色表示角色:
  full: black / gray
  target: red
  anchor_primary: blue
  anchor_support: orange
  room_context: green
- 画出从 I 到这些 query cards 的箭头。

Panel B: Multi-Query Object Grounding
- 中间显示一个 3D object memory / object bank:
  O = {o_i}_{i=1}^{N}
  b_i = [x_i, z_i, y_i, dx_i, dz_i, dy_i]
- 对每个 query q_r，调用 Stage2 grounding，得到:
  l_{r,i} = logits(q_r, o_i)
  p_{r,i} = softmax(l_{r,i} / T)
  H_r^K = TopK_i p_{r,i}
- 画成多条 query arrows 指向同一个 object memory，每条 query 输出一组 top-k colored object dots。
- 强调“Stage1 / segmentation / object memory are reused; only text grounding is repeated”。

Panel C: Footprint-Aware Spatial Consensus
- 画一个 top-down 2D map。
- 每个 object o_i 显示为一个圆盘或椭圆 footprint:
  xz_i = [x_i, z_i]
  rho_i = footprint radius
- 不同角色命中的 object 用对应颜色描边或小标记显示。
- 在图上写出 footprint-aware distance:
  d_ij = max(0, ||xz_i - xz_j||_2 - rho_i - rho_j)
- 根据 d_ij <= epsilon 建图:
  G = (V, E)
  E_ij = 1[d_ij <= epsilon]
- 将 connected components 画成半透明区域:
  R_1, R_2, ..., R_M
- 对每个 region 展示 region scoring:
  e_r(R_m) = 1 - product_{o_i in R_m}(1 - p_{r,i})
  coverage(R_m) = covered_roles / required_roles
  compactness(R_m) = exp(-mean_distance_to_region_center^2 / sigma^2)
  S(R_m) = sum_r w_r log(e_r(R_m)) + alpha coverage(R_m) + beta compactness(R_m)
- 用高亮边框标出 best region:
  R* = argmax_m S(R_m)
- 图形语义要表达: target and anchors do not need to be at the same point; they only need to form a coherent local region.

Panel D: Target-Role Selection and Refinement
- 在 best region R* 内，只从 target-role objects 中选择最终目标:
  P_target = {o_i in R* | o_i appeared in H_target^K}
- 展示 target score:
  s_target(o_i) =
    lambda_t log p_target,i
    + lambda_f log p_full,i
    + lambda_rel relation_support(o_i, R*)
    + lambda_m log merged_score_i
    + lambda_c log count_i
- 最终选择:
  o* = argmax_{o_i in P_target} s_target(o_i)
  tau* = center_xyz(o*)
- 画出 baseline target tau_0 和 MQSC selected target tau*:
  tau_0 用灰色叉号表示
  tau* 用红色星标表示
- 右侧输出:
  if thresholds pass:
    target_after = tau*
    applied = true
  else:
    target_after = tau_0
    applied = false
- 从 tau* 画箭头到 navigation follower。

图中请突出三个核心思想:
1. Role-aware decomposition:
   target、anchor、room context 是不同角色，不应混为同一个 object vote。
2. Region-first reasoning:
   先找能同时解释多条语言线索的局部区域 R*，再在 R* 内选 target object。
3. Conservative refinement:
   MQSC 只在 region coverage、target confidence、region margin、selected gain 通过阈值时替换 baseline target。

请在图底部加一条简洁 caption:
"MQSC turns a fine-grained referring expression into multiple role-aware grounding queries, aggregates their top-k object evidence into footprint-aware spatial regions, and selects the target-role object inside the most consistent region."

图中文字尽量少，但关键变量必须出现:
I, q_r, O={o_i}, b_i, p_{r,i}, H_r^K, d_ij, G=(V,E), R_m, S(R_m), R*, P_target, o*, tau_0, tau*, applied.
```

如果希望画得更像“人类认知故事图”，可以把 prompt 的第一句改成：

```text
请绘制一张结合 cognitive intuition 与 algorithm pipeline 的论文方法图，表现人类找东西时会先拆分目标、锚点和房间线索，再在空间中寻找多线索共识区域。
```

