# MQSC-R1: 从“名词竞争”到“角色协作”的局部线索共识

## 1. Introduction

在长指令室内导航中，智能体经常被要求寻找一个并不显眼的目标:

```text
the bottom black-framed line drawing picture on gray wall between two others in bedroom with light gray bedding and red bookshelf
```

这类指令并不是一个简单的类别标签。它包含目标、房间、墙面、床品、书架、相对位置和一串视觉上下文。传统 final grounding 常把整句压成一次 object-level matching:

```text
instruction -> one score distribution over memory objects -> top-1 target
```

这个过程高效，但也脆弱。长句里每个名词都可能变成竞争者: picture、wall、bed、bookshelf。模型可能确实看懂了某些词，却把注意力落到了最显著的 anchor，而不是真正的 target。

MQSC-R1 的出发点来自 landmark-based referring expression resolution 的一个简单启发:

> 目标线索应该决定最终物体，地标线索应该定位目标附近的局部邻域。

当人听到“床旁边床头柜上的台灯”时，一个实用的搜索策略是先确定目标身份是“台灯”，再利用“床”和“床头柜”缩小搜索区域。MQSC-R1 不试图完整模拟人类认知，而是抽取这个可实现的归纳偏置: target cue 与 landmark cue 不应彼此竞争同一个 top-1，而应在局部空间中协作。

MQSC-R1 将这种认知过程转化为一个轻量的 test-time module:

```text
decompose instruction into target and object-level anchors
  -> query the 3D memory from multiple semantic roles
  -> find a local region where these roles agree
  -> choose the target-role object inside that region
```

它不是替代主导航器，而是在主导航器已经决定 stop at object 时，重新审视这个 final target 是否真的处在一个有多角色语义共识的局部场景中。更准确地说，MQSC-R1 是:

```text
a test-time, object-centric, role-aware consistency prior
for final object grounding
```

## 2. Motivation: 为什么 R1 要剔除 room context

MQSC 的初始想法很自然: 如果指令里出现 bedroom、bathroom、kitchen，那么 room context 也应该成为一个 query。人类确实会利用房间语义。然而，在当前 object-centric 3D grounding 模型里，room words 并不总能落到“房间区域”。

例如 `bedroom` 可能激活 bed、wardrobe、pillow；`bathroom` 可能激活 toilet、sink、mirror。这些对象当然和房间相关，但它们不等价于房间本身。当 room query 被当作共识证据时，它可能把模型推向最典型的房间物体，而不是目标附近的真实局部结构。

因此 MQSC-R1 做了一个重要的认知折中:

> 房间词可以帮助人理解任务背景，但在当前模型里不应作为 object-level voting evidence。

换句话说，MQSC-R1 并不是否认 room context 的价值，而是承认当前 VLE/Stage2 的感知单位是 object memory。既然感知器以 object 为最小单位，R1 就只让 object-like 线索作为独立角色进入空间共识:

```text
full instruction
target object
nearby object / fixture anchors
```

而将:

```text
room_context
```

保留为 diagnostic metadata。需要特别说明的是，完整句子 `q_full = x` 仍可能包含 room words；R1 的设计不是把房间词从语言中删除，而是不允许 room context 作为单独 role 投出额外 object-level votes。这让方法更贴近模块的实际能力，也使叙事更清晰: MQSC-R1 是 object-centric memory 上的角色化空间共识，而不是完整房间识别器。

在论文实验中，这个设计最好由一个 ablation 支撑:

```text
full + target + anchors + room role
vs.
full + target + anchors only
```

如果 room role 常把 `bedroom` 推向 bed/wardrobe、把 `bathroom` 推向 toilet/sink/mirror，而不是目标附近区域，那么 R1 的 room 剔除就不再像经验补丁，而是 representation mismatch 下的设计选择。

## 3. Cognitive View: 人类如何把语言变成局部场景

作为一个可用的认知类比，人类找东西时常把问题拆成三类线索。

第一，分离角色:

```text
我要找什么? target
它靠近什么? anchors
它大概处在哪种环境? context
```

第二，压缩搜索空间:

```text
不是在整个房间里找一个孤立目标，
而是先找一片由目标和 anchor 共同定义的局部场景。
```

第三，在局部场景中重新选择主角:

```text
床和床头柜帮助找到 bedside region，
但最终要选的是 lamp，不是 bed 或 nightstand。
```

MQSC-R1 对应这三个过程:

```text
role decomposition
  -> role-aware spatial region consensus
  -> target-role object selection
```

它的核心不是“多问几次模型然后投票”，而是“不同问题承担不同认知角色”。target query 负责识别主角，anchor query 负责定位局部场景，full query 负责维持完整语言约束。只有当这些角色在同一片空间附近相遇时，智能体才获得比单次 top-1 更稳定的证据。

## 4. Problem Formulation

给定导航指令 `x`，主导航器已经构建了一个 object memory:

```text
M = {o_i}_{i=1}^{N}
```

每个 object `o_i` 包含:

```text
box_i, feature_i, open_vocab_feature_i, object_score_i, object_count_i
```

baseline final decision 给出:

```text
o_base = argmax_i P(o_i | x)
```

MQSC-R1 不直接否定这个结果，而是把它看作一个候选假设。模块目标是判断是否存在另一个 object `o*`，它在多角色语言线索和局部空间结构上更可信:

```text
o* = Refine(o_base, x, M)
```

如果证据不足，MQSC-R1 fallback 到 baseline。

## 5. Role Decomposition

MQSC-R1 首先将指令分解为角色集合:

```text
z = {
  target_desc,
  target_aliases,
  anchor_primary,
  anchor_support,
  room_context,
  relations
}
```

随后构造实际参与 grounding 的 role queries:

```text
Q = {q_full, q_target, q_anchor_primary, q_anchor_support}
```

其中:

```text
q_full = x
q_target = concat(target_desc, target_aliases)
q_anchor_primary = concat(anchor_primary)
q_anchor_support = concat(anchor_support)
```

`room_context` 被保留为解释信息:

```text
room_context -> diagnostic only
```

这一步可以写成一个认知假设:

> 在 object-centric memory 上，房间词更像背景先验，而非可靠的定位观测。MQSC-R1 将背景先验从投票机制中剥离，只让可被 object memory 直接承载的 target 与 landmark cues 参与共识。

## 6. Multi-Query Grounding as Multiple Glimpses

对每个 role query `q_r`，MQSC-R1 复用当前 PQ3D Stage2 object grounding，得到每个 memory object 的 logit:

```text
l_r(i) = f_stage2(o_i, q_r)
```

再做 query-wise softmax:

```text
p_r(i) =
  exp(l_r(i) / tau)
  / sum_j exp(l_r(j) / tau)
```

其中 `tau` 是 temperature。

每个 role 只保留 top-k objects:

```text
K_r = TopK_i p_r(i)
```

从认知角度看，这类似多次扫视:

```text
full glimpse:    整句话指向哪里?
target glimpse:  目标物本身在哪里?
anchor glimpse:  关键地标在哪里?
support glimpse: 辅助线索在哪里?
```

单次扫视可能被遮挡、显著性或词义偏差误导；但如果多次扫视落在同一片局部空间，智能体获得了更可靠的证据。

## 7. Footprint-Aware Spatial Graph

每个 object 的 3D box 被投影到 top-down footprint:

```text
xy_i = (x_i, z_i)
rho_i = 0.5 * max(dx_i, dz_i)
```

两个 object 的 footprint distance:

```text
d_fp(i,j) =
  max(0, ||xy_i - xy_j||_2 - rho_i - rho_j)
```

当:

```text
d_fp(i,j) <= epsilon
```

就在候选图中连边。所有连通分量构成候选局部区域:

```text
R = {R_1, R_2, ..., R_m}
```

这个设计比裸 center clustering 更符合人类空间直觉。床、桌子、沙发这类大物体的中心未必靠近目标，但它们的边界定义了人在空间中理解“旁边”“附近”的区域。

## 8. Role-Aware Region Consensus

对一个 region `R`，role `r` 的证据由 noisy-or 聚合:

```text
tilde_p_r(i) =
  p_r(i), if o_i in K_r
  0,      otherwise

E_r(R) = 1 - product_{o_i in R} (1 - tilde_p_r(i))
```

这里的 `E_r(R)` 不应被理解成严格校准概率，而是一个 bounded soft-union aggregator over role-normalized scores。top-k 截断是必要的: 非 top-k object 即使有 softmax 残余概率，也不会给该 role 的区域证据投票。

Coverage 衡量多少必要角色被该区域解释:

```text
C_cov(R) =
  |{r in Q_required : E_r(R) > delta}|
  / |Q_required|
delta = 1e-9
```

Compactness 衡量该区域是不是一个局部场景，而不是松散拼凑:

```text
C_cmp(R) =
  exp(
    - mean_{o_i in R} ||xy_i - mean(xy_R)||_2^2 / sigma^2
  )
```

Region score:

```text
S(R) =
  sum_{r in Q_required} w_r log(epsilon_0 + E_r(R))
  + lambda_cov C_cov(R)
  + lambda_cmp C_cmp(R)
  - lambda_noise N(R)
```

在当前实现中:

```text
lambda_cov = 1.25
lambda_cmp = 0.25
epsilon_0 = 1e-6
```

`N(R)` 是轻量噪声惩罚。令 `n_r(R)` 表示 region `R` 中属于 role `r` 的 top-k hits 数量:

```text
N(R) =
  0.2, if C_cov(R) < 0.5 and max_r n_r(R) >= 3
  0,   otherwise
```

如果一个区域被同一 role 反复命中但缺少跨角色覆盖，则它更像单一语义偏置，不像真实的指称场景。

最佳区域:

```text
R* = argmax_R S(R)
```

Region margin:

```text
Delta_R = S(R*) - max_{R != R*} S(R)
```

## 9. Target Selection Inside the Region

MQSC-R1 的一个关键原则是:

> anchor 可以帮助确定区域，但不能成为最终目标。

因此在 `R*` 内只考虑被 target role 命中的 objects:

```text
P_target(R*) =
  {o_i in R* : o_i in K_target}
```

对每个 target candidate，计算它与 region 中 anchor/full evidence 的空间支持:

```text
Rel(i, R*) =
  max_{j in R*, j != i}
  exp(- d_fp(i,j)^2 / (2 sigma_rel^2))
```

其中 `j` 必须来自:

```text
full, anchor_primary, anchor_support
```

最终 target score:

```text
T(i | R*) =
  alpha log(epsilon_0 + p_target(i))
  + beta log(epsilon_0 + p_full(i))
  + gamma Rel(i, R*)
  + eta log(max(object_score_i, 1e-6))
  + mu log(max(object_count_i, 1))
```

当前实现使用:

```text
alpha = 1.45
beta  = 0.75
gamma = 0.45
eta   = 0.05
mu    = 0.03
```

选择:

```text
o* = argmax_{o_i in P_target(R*)} T(i | R*)
```

这个设计体现了“区域先于目标，但目标不能被区域吞没”的原则。人会用 bed 找 bedside region，但不会把 bed 当作 lamp。

## 10. Conservative Replacement

MQSC-R1 不是强行覆盖 baseline。它只在证据足够时替换 final target。

令 baseline score 为:

```text
T_base = T(o_base | R*)
```

selected gain:

```text
G = T(o*) - T_base
```

论文主文中可以使用统一的 `T` 来描述 replacement gate；当前实现中，baseline gate score 使用 semantic + relation terms，而 selected target ranking 额外带有 `object_score/object_count` 的稳定性项。因此在正式写作中建议把它称作 heuristic conservative gate，而不是 calibrated likelihood ratio。

MQSC-R1 替换 baseline 需要同时满足:

```text
C_cov(R*) >= theta_cov
p_target(o*) >= theta_tgt
Delta_R >= theta_margin
G >= theta_gain
o* != o_base
```

默认阈值:

```text
theta_cov = 0.5
theta_tgt = 0.05
theta_margin = 0.05
theta_gain = -0.02
```

如果任何条件不满足，模块显式 fallback:

```text
target_after = target_before
```

这使 MQSC-R1 更像一个“认知校验器”而不是激进策略: 它只在多线索共识比 baseline 更有说服力时介入。

## 11. Why This Matters in the Larger Paper

如果论文主线是“长语言描述下的具身导航需要把语言线索与空间记忆进行动态对齐”，那么 MQSC-R1 可以作为其中一个模块承担以下角色:

```text
主导航器负责探索、建图和产生候选 stop decision；
MQSC-R1 负责在 final object decision 时进行角色一致性目标审查；
后续可视可达模块负责把目标点转成更好的可执行 viewpoint。
```

在故事层面，它回答的是:

> 当语言描述中含有多个物体线索时，智能体如何避免被最显著的单个名词误导?

MQSC-R1 的答案是:

> 把长指令里的不同名词从竞争关系改成角色协作关系。

这句话比“模拟人类认知”更适合作为论文模块的主 claim。认知启发是背景，硬贡献是 role-aware consistency prior。

这个模块可以和 vista-ls / vista2mqsc 形成自然叙事:

```text
MQSC-R1: 选对哪个 object / 哪片 semantic region
VISTA-LS: 站到哪里才能看见并接近它
```

前者解决 semantic disambiguation，后者解决 executable viewpoint selection。

## 12. Algorithm

```text
Input:
  instruction x
  object memory M = {o_i}
  baseline final target o_base

1. z <- Decompose(x)
2. Q <- BuildQueries(z)
     Q = {full, target, anchor_primary, anchor_support}
     room_context is diagnostic-only

3. For each role r in Q:
     logits l_r <- Stage2(M, q_r)
     p_r <- softmax(l_r / tau)
     K_r <- TopK(p_r)

4. Build candidate graph G:
     nodes = union_r K_r
     edge(i,j) if d_fp(i,j) <= epsilon
     regions = connected_components(G)

5. For each region R:
     E_r(R) <- noisy_or role evidence
     C_cov(R) <- role coverage
     C_cmp(R) <- spatial compactness
     S(R) <- region consensus score

6. R* <- argmax_R S(R)

7. target_pool <- target-role objects in R*
   if empty:
       return baseline

8. For each target candidate o_i:
     T(i | R*) <- target score with relation support

9. o* <- argmax_i T(i | R*)

10. If gates pass:
      return center(o*)
    else:
      return baseline
```

## 13. Paper-style Summary Paragraph

We introduce Multi-Query Spatial Consensus R1 (MQSC-R1), a test-time refinement module for final object decisions in long-instruction embodied navigation. Instead of grounding the entire instruction into a single object prediction, MQSC-R1 decomposes the instruction into role-specific cues, queries the current 3D object memory with each cue, and searches for a local spatial region where the full instruction, target cue, and object-level landmarks agree. To match the object-centric nature of the underlying grounding model, room context is retained only as diagnostic metadata and is not instantiated as an independent voting role, although room words may still appear in the full-instruction query. The module first selects a consensus region using footprint-aware connected components and role-aware evidence aggregation, then selects a target-role object within that region under conservative replacement gates. This yields a lightweight role-consistency check: a final target is accepted not merely because it is individually likely, but because it is supported by a coherent local scene.

## 14. 中文摘要段落

我们提出 MQSC-R1，一个用于长指令具身导航 final object decision 的测试时修正模块。与直接将完整指令匹配到单个最高分物体不同，MQSC-R1 将指令分解为目标、完整语义和物体级地标线索，并分别查询当前 3D object memory。模块首先寻找一片由多种角色线索共同支持的局部区域，再只在该区域内选择 target-role object。由于底层 grounding 模型以 object 为主要感知单位，R1 将 room context 降级为诊断信息，避免房间词变成噪声 object prior。最终，MQSC-R1 通过保守门控决定是否覆盖 baseline，使智能体在停止前进行一次受人类线索检验启发的角色一致性校验。

## 15. 需要诚实呈现的边界

为了避免论文叙事过度扩张，建议明确:

- MQSC-R1 是 final decision refinement，不是完整导航策略；
- MQSC-R1 不改变 non-final frontier exploration；
- room_context 在 R1 中不参与 consensus；
- relation reasoning 是 footprint proximity，不是完整 scene graph relation parser；
- 模块不训练新参数，依赖现有 PQ3D object memory 和 Stage2 logits；
- applied 不等于一定 helpful，效果需要通过 SR/SPL 和 per-level 指标验证。

这些边界不会削弱故事，反而让论文显得更可信: MQSC-R1 不是声称“人类认知全部被模拟”，而是抽取了其中一个最关键、最可实现的原则:

> 在长语言导航中，目标的可信度来自局部场景中的多线索共识，而不是孤立的单次语义匹配。

## 16. Reviewer 视角下的防守叙事

如果将该模块写入 NeurIPS / CVPR / AAAI 风格论文，建议主动防守以下问题。

第一，MQSC-R1 不是 learned module，而是 test-time prior。不要把它写成新模型，应写成对现有 object memory 的 final grounding refiner。

第二，room context exclusion 需要实验支撑。建议报告:

```text
with independent room role
without independent room role
```

并展示 room role 是否更容易命中典型房间物体而非目标邻域。

第三，VLM decomposition 需要可复现。建议报告:

```text
VLM model
parse success rate
heuristic fallback rate
without-VLM ablation
```

第四，follower repair 与 MQSC-R1 应分开消融。否则 reviewer 可能认为指标提升来自 navmesh repair，而不是 semantic target refinement。

第五，footprint proximity 只能称为 relational support approximation，不能称为完整 relation reasoning。它的优点是轻量、可解释、无需训练；边界是不能显式处理所有 `on/under/between` 关系。

最终建议主文中的一句话写法:

```text
MQSC-R1 converts nouns in a long referring expression from competing object labels into cooperative roles: target cues select the object, while landmark cues localize the neighborhood.
```
