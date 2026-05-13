# VISTA-LS 方法故事

VISTA-LS = Visibility-Informed Target Adjustment via Level-Set Search

一句话版本：

> 传统 VLN / embodied navigation baseline 往往把语义模块预测出的目标点直接交给导航器；VISTA-LS 认为“目标在哪里”和“agent 应该站在哪里”是两个不同问题，因此在执行导航前，将预测目标点转换成一个可达、可见、距离合适且不贴障碍的稳定观察位置。

## 1. 传统 baseline 的隐含假设

在很多视觉语言导航或 embodied reference navigation 系统中，整体流程可以简化为三步：

```text
理解语言指令 -> 预测目标位置 -> 导航到该目标位置
```

这个流程背后有一个很强的隐含假设：

```text
只要目标位置预测正确，直接导航到这个点就是合理的。
```

但在真实 3D 环境中，这个假设并不总成立。语言和视觉模块预测的是“目标物体在哪里”，而低层导航器需要的是“agent 应该站在哪里”。这两个点经常不是同一个点。

例如，一个模型可能准确定位到了墙上的画、桌上的杯子、床上的毯子或柜子里的物体。预测点在语义上是对的，但它可能位于：

- 物体内部；
- 墙面或家具表面；
- navmesh 外；
- 目标背后；
- 障碍物边缘；
- 离目标太近，导致相机看不全；
- 离目标太远，导致成功距离或可见性不足。

因此，baseline 的失败不一定来自“找错了目标”。很多失败来自一个更细的问题：

```text
找到了目标，但把导航器送到了一个不适合执行和观察的位置。
```

VISTA-LS 解决的正是这个位置转换问题。

## 2. 目标点不是导航点

传统 baseline 通常把目标预测点当成最终导航点。这个做法简单直接，但它混淆了两个空间概念。

第一个概念是 semantic target point：

```text
目标物体或目标区域的几何中心、点云中心、检测框中心，或模型预测的目标坐标。
```

第二个概念是 executable viewpoint：

```text
agent 可以站立、可以到达、可以观察目标、并且对局部扰动稳定的位置。
```

在导航任务中，真正应该交给 follower 的不是目标物体本身的位置，而是目标附近的一个可执行 viewpoint。对于桌子、床、柜子、画、灯等常见目标，这一点尤其明显：agent 不应该导航到物体内部或表面，而应该导航到一个合适的观察位置。

VISTA-LS 的核心主张可以写成：

```text
Navigation should not directly consume a semantic target point.
It should consume a visibility-informed, reachable viewpoint around the target.
```

## 3. 从“点预测”到“可行区域”

如果只是在目标周围随机或固定半径采几个点，再挑一个分数最高的点，仍然会保留 baseline 的一部分脆弱性：最终决策还是落在单个离散点上。

VISTA-LS 更进一步，把导航目标选择从“点级选择”改写为“区域级搜索”。

它不问：

```text
哪个候选点分数最高？
```

而是先问：

```text
目标周围哪一片区域是导航可行的？
```

这就是 VISTA-LS 名字中 level-set search 的含义。我们不是直接寻找一个最优点，而是先定义一组几何约束，找出满足这些约束的局部可行域：

```text
F = Reachable ∩ Visible ∩ Target-Distance-Shell ∩ Clearance-Safe
```

然后在这片可行域中选择一个稳定代表点。

这个转变对应论文中的方法动机：

```text
传统 baseline 直接导航到预测目标点；
VISTA-LS 先把预测目标点转化为目标周围的稳定可执行 viewpoint。
```

## 4. 四个约束分别在修正什么 baseline 问题

### 4.1 Reachable：修正不可执行目标点

baseline 预测出的点可能落在不可导航区域。即使预测点语义正确，follower 也可能无法到达。

VISTA-LS 首先要求候选 viewpoint 在 navmesh 上可导航，并且和当前 agent 位于同一个 navigable island：

```text
R(v) = 1
```

这一步把“语义正确但不可执行”的目标点过滤掉。

### 4.2 Visible：修正到达但看不见的问题

传统导航指标通常关心 agent 是否接近目标，但实际 embodied task 中，agent 还需要能够观察目标。一个点离目标很近，却可能在目标背后、墙后或被家具遮挡。

VISTA-LS 对目标点云做 ray casting，估计从候选 viewpoint 到目标点云的可见比例：

```text
V(v) = visible_target_points / target_sample_count
```

这一步把“到达附近但看不到目标”的位置过滤掉。

### 4.3 Target-Distance Shell：修正过近或过远的问题

baseline 的目标点往往是物体中心或点云中心。直接导航过去容易太近，甚至进入物体内部；但如果修正点太远，又可能不满足成功距离或观察质量。

VISTA-LS 定义一个距离壳层：

```text
rho_min <= d_T(v) <= rho_max
```

其中 `d_T(v)` 是候选 viewpoint 到目标表面的距离。这个壳层表达的是：

```text
agent 应该站在目标附近，但不是站到目标里面。
```

### 4.4 Clearance：修正贴墙和贴障碍问题

传统 baseline 可能给出一个几何上接近目标、但贴着墙或障碍物边缘的位置。这样的点容易导致 navmesh snap 后偏移，也容易让 follower 失败。

VISTA-LS 要求候选 viewpoint 与非目标点云保持最小 clearance：

```text
C(v) >= c_min
```

这一步让最终目标点更像一个可站立区域，而不是一个贴着障碍物的边缘点。

## 5. 为什么要选连通区域中心，而不是最高分点

即使经过四类约束过滤，仍然可能得到多个可行点。一个自然做法是选择 visibility 最高的点。但 VISTA-LS 不这么做。

原因是：最高 visibility 的单点可能只是偶然成立。点云稀疏、遮挡边界、ray casting 采样、navmesh snap 都可能制造局部尖峰。这样的点在离散评估中看起来很好，但执行时稍微偏移就可能失效。

VISTA-LS 因此把所有可行点组织成图：

```text
G = (F, E)
```

相邻 radius bin、相邻 angle bin，或空间距离足够近的可行点之间连边。然后提取 connected components：

```text
F = C1 ∪ C2 ∪ ... ∪ Ck
```

一个大的连通分量意味着：目标周围存在一片连续区域，里面很多相邻站位都满足可达、可见、距离和 clearance 约束。这样的区域比孤立高分点更稳定。

最终，VISTA-LS 在选中的连通分量内部选择 medial center：

```text
v* = argmax_v dist_G(v, boundary(C*))
```

这个点离可行域边界最远，因此对局部扰动更鲁棒。换句话说，VISTA-LS 不是选择“刚好可行”的点，而是选择“可行区域内部最稳”的点。

## 6. 方法流程

给定 baseline 或 PQ3D 的 predicted target，VISTA-LS 执行以下流程。

1. 提取目标点云和非目标点云。

目标点云用于估计目标中心、目标表面距离和可见性；非目标点云用于遮挡检测和 clearance 估计。

2. 在目标周围生成局部 polar grid。

```text
v(r, theta) = mu_ground + (r cos theta, 0, r sin theta)
```

这一步不是为了固定采几个圆环点，而是为了离散化目标周围的局部空间。

3. 对每个候选 viewpoint 计算四类属性。

```text
reachable(v)
visibility(v)
target_surface_distance(v)
clearance(v)
```

4. 根据约束构造可行域。

```text
F = { v | reachable(v),
          visibility(v) >= tau_v,
          rho_min <= target_surface_distance(v) <= rho_max,
          clearance(v) >= c_min }
```

5. 如果严格可行域为空，则逐步放松约束。

放松顺序是 visibility、distance shell、clearance。每次放松都会写入日志，因此模块不会静默回退。

6. 对可行域做 connected component selection。

优先选择更大的连通分量；若多个分量大小接近，再比较 visibility 分位数和路径代价。

7. 在选中分量内选择 medial center。

这个点作为 refined navigation target。

8. 通过 planner filter 做最终可执行性检查。

如果 selected target 对 follower 不稳定，则在局部范围内寻找最近的可 follow repair target。

## 7. 算法伪代码

```text
Input:
  predicted target point p
  target point cloud T
  blocker point cloud B
  current agent position a
  path finder P

1. Estimate target center mu from T.

2. Generate local search grid Omega around mu.

3. For each viewpoint v in Omega:
     R(v) <- navigable(v) and same_island(v, a)
     V(v) <- raycast_visibility(v, T, B)
     D(v) <- distance_to_target_surface(v, T)
     C(v) <- clearance_to_blockers(v, B)

4. Build feasible set:
     F = { v | R(v)=1,
               V(v) >= tau_v,
               rho_min <= D(v) <= rho_max,
               C(v) >= c_min }

5. If F is empty:
     relax constraints with explicit logging.

6. Build graph G over F.

7. Extract connected components.

8. Select the most stable component C*.

9. Select medial center:
     v* = argmax_v dist_G(v, boundary(C*))

10. Run planner precheck / local repair.

Output:
  executable viewpoint v*
```

## 8. 和传统 VLN / baseline 的区别

核心区别可以概括成一张表。

| 维度 | 传统 baseline | VISTA-LS |
|---|---|---|
| 输入语义模块输出 | 预测目标点 | 预测目标点 |
| 对目标点的处理 | 直接导航到该点 | 转换为可执行 viewpoint |
| 是否考虑可见性 | 通常不显式考虑 | 显式 ray casting |
| 是否考虑 navmesh 可达性 | 主要依赖 follower 后处理 | 在候选选择阶段显式约束 |
| 是否考虑目标表面距离 | 通常不显式考虑 | 使用 target-distance shell |
| 是否考虑 clearance | 通常不显式考虑 | 显式避免贴障碍 |
| 最终选择对象 | 一个预测点或最近可导航点 | 一个稳定连通可行域的 medial center |

因此，VISTA-LS 不是一个新的语言理解模块，也不是一个新的全局规划器。它是连接语义预测和低层导航执行的几何适配层：

```text
semantic target -> executable viewpoint
```

这个适配层是传统 baseline 中经常被忽略的关键环节。

## 9. 论文贡献表述

可以把 VISTA-LS 的贡献写成三点。

第一，指出传统 embodied navigation baseline 中存在 semantic target 与 executable viewpoint 的不匹配问题。预测目标点语义上正确，并不意味着该点适合作为导航目标。

第二，提出一种 VLM-free 的 visibility-informed target adjustment 方法。该方法只依赖目标点云、非目标点云、navmesh 和几何 ray casting，不需要额外调用视觉语言模型。

第三，将直接导航到预测目标点的问题改写为局部 level-set feasible region search。通过 reachability、visibility、target-distance shell 和 clearance 的联合约束，VISTA-LS 找到目标周围稳定可行的区域，并选择其 medial center 作为最终导航点。

## 10. 可以直接写进论文的方法段落

Traditional embodied navigation pipelines often pass the predicted target location directly to the low-level navigator. However, a semantic target point is not necessarily an executable viewpoint: it may lie inside the object, outside the navigable mesh, behind an occluder, or too close to surrounding obstacles. We propose VISTA-LS, a visibility-informed target adjustment module that converts a predicted target into a stable navigable viewpoint. Given the target point cloud, VISTA-LS constructs a local search space around the target and defines a feasible region as the intersection of reachability, visibility, target-distance-shell, and clearance constraints. Instead of selecting an isolated high-scoring point, it extracts connected components in this feasible region and chooses the medial center of the most stable component. This transforms target following from direct point navigation into executable viewpoint selection, improving robustness to occlusion, navmesh snapping, and local geometric perturbations without requiring additional VLM calls.

## 11. 图示讲法

论文图可以画成一个从 baseline 到 VISTA-LS 的对比。

左侧画传统 baseline：

```text
language instruction -> predicted target point -> direct navigation
```

并标出问题：目标点可能在物体内部、墙边或不可见位置。

右侧画 VISTA-LS：

```text
predicted target point -> local feasible region -> medial viewpoint -> planner-safe target
```

重点突出四层约束：

```text
Reachable
Visible
Target-Distance-Shell
Clearance-Safe
```

图的核心视觉信息应该是：

```text
baseline 导航到一个语义目标点；
VISTA-LS 导航到目标附近的一片稳定可行区域的中心。
```

这才是模块故事的主线。

## 12. 绘图 Prompt

下面这段可以直接丢给 GPT / 图像生成模型 / 论文插图助手，用来生成 VISTA-LS 方法图。

```text
Create a clean academic method figure for a robotics / embodied navigation paper. The figure should compare a traditional VLN baseline with our method VISTA-LS.

Overall layout:
- Use a two-column comparison layout.
- Left column title: "Traditional Baseline: Direct Target Navigation".
- Right column title: "VISTA-LS: Executable Viewpoint Selection".
- Use a top-down 2D floor-map style with simple 3D cues.
- Use clear arrows, labeled variables, and compact equations.
- Keep the style minimal, paper-ready, vector-diagram-like, with white background and restrained colors.

Scene elements:
- Draw a room floor plan with walls and obstacles.
- Show an object target as a small object point cloud, labeled "Target point cloud T".
- Show non-target / obstacle points around the object, labeled "Blocker points B".
- Show the current agent position as a robot icon or triangle, labeled "agent a".
- Show the predicted semantic target point as a red dot inside or very near the object, labeled "predicted target p".

Left baseline column:
- Draw an arrow from the agent a directly to the predicted target p.
- Label this path "direct navigation to predicted target".
- Visually indicate failure/risk: the red target point p lies inside the object, near a wall, or behind an obstacle.
- Add a small warning label: "semantic target ≠ executable viewpoint".
- Show that the baseline ignores or does not explicitly model visibility, clearance, and target-distance shell.

Right VISTA-LS column:
- Around the target point cloud T, draw a local polar search grid Ω with rings and angle rays.
- Label a candidate viewpoint as v = (x, y, z).
- Label the camera point as c(v) = v + (0, h, 0).
- Draw several candidate viewpoints around the target.
- Use four visual filters or badges:
  1. Reachable: R(v)=1
  2. Visible: V(v) ≥ τ_v
  3. Target-Distance Shell: ρ_min ≤ d_T(v) ≤ ρ_max
  4. Clearance-Safe: C(v) ≥ c_min
- Show these constraints forming the feasible set:
  F = Reachable ∩ Visible ∩ Target-Distance-Shell ∩ Clearance-Safe
- Color invalid candidates in gray.
- Color feasible candidates in green.
- Group feasible candidates into connected components C1, C2, ... .
- Highlight the selected stable component C* with a translucent green region.
- Mark the selected medial center as a blue star or blue dot, labeled:
  v* = argmax_v dist_G(v, boundary(C*))
- Draw a final arrow from the agent a to v*, labeled "planner-safe refined target".

Method pipeline strip:
- Add a small horizontal pipeline at the bottom:
  predicted target p
  → local grid Ω
  → feasible set F
  → connected component C*
  → medial center v*
  → planner filter
  → executable viewpoint

Important variables to include in the figure:
- p: predicted semantic target
- T: target point cloud
- B: blocker / non-target point cloud
- a: current agent position
- v: candidate viewpoint
- c(v): camera point
- Ω: local search region
- F: feasible set
- C*: selected connected component
- v*: final refined viewpoint
- R(v), V(v), d_T(v), C(v)

Core message:
- The figure should communicate that the baseline directly navigates to a semantic target point, while VISTA-LS converts that target into a stable executable viewpoint by searching a local feasible region and selecting the medial center of the best connected component.

Avoid:
- Do not make it look like a marketing slide.
- Do not use photorealistic humans.
- Do not over-emphasize neural network blocks.
- Do not compare against last-mile methods; the comparison should be against direct target navigation baseline.
```

