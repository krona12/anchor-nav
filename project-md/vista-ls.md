# VISTA-LS 模块方案

VISTA-LS = Visibility-Informed Target Adjustment via Level-Set Search

目标：在现有 MSGNav/VISTA mile 代码基础上，不调用 VLM，把 last-mile viewpoint selection 从固定半径候选点枚举，升级为局部几何可行域搜索：

```text
F = Reachable ∩ Visible ∩ Target-Distance-Shell ∩ Clearance-Safe
```

然后在最大稳定连通可行区域中选择 medial center，而不是用手工加权 score 选单点。

## 1. 核心思想

当前最好 mile 家族的本质是纯几何 last-mile 视点重选。它的问题在于候选点来自固定半径和固定角度枚举，容易错过真正好的可行区域。VISTA-LS 把这个问题改写成 level-set / constraint boundary search：

```text
先找满足约束的一片区域，再选区域内部最稳定的代表点。
```

好点通常不是 visibility 最高的孤立尖点，而是满足这些条件的一片连通区域中心：

- 能到达
- 能看见目标
- 离目标表面处于合适壳层
- 不贴墙、不贴障碍
- 周围小扰动仍然可行

## 2. 数学定义

给定目标点云：

```text
T = {t_i}_{i=1}^N, t_i ∈ R^3
```

目标中心：

```text
μ = mean(T)
```

当前 agent 位置：

```text
a ∈ R^3
```

候选地面视点：

```text
v = (x, y, z)
```

相机点：

```text
c(v) = v + (0, h, 0)
```

其中推荐：

```text
h = 1.50m
```

## 3. 搜索域

在目标周围构造局部 polar / navmesh grid：

```text
Ω = { v(r, θ) | r_min_global ≤ r ≤ r_max_global, θ ∈ [0, 2π) }
```

其中：

```text
v(r, θ) = μ_ground + (r cos θ, 0, r sin θ)
```

推荐初始参数：

```text
r_min_global = 0.30m
r_max_global = 1.30m
radial_step = 0.05m 或 0.10m
angle_step = 5° 或 10°
```

第一版可以先用 polar grid，后续再加 navmesh-local grid 或边界细化。

## 4. 四类约束

### 4.1 Reachable

定义：

```text
R(v) = 1
```

当且仅当：

```text
path_finder.is_navigable(v) = true
path_finder.get_island(v) = path_finder.get_island(a)
GreedyGeodesicFollower precheck succeeds
```

否则：

```text
R(v) = 0
```

实现上可分两阶段：

1. cheap reachable：navmesh navigable + same island
2. expensive reachable：GreedyGeodesicFollower precheck

这可以避免对所有点都跑 follower。

### 4.2 Visible

对 target sample 点做 ray casting：

```text
V(v) = (1 / N) Σ_i I[segment(c(v), t_i) is not occluded]
```

遮挡检测沿用当前 mile.py 的 KDTree ray sampling / occlusion_radius_m 逻辑：

```text
ray = t_i - c(v)
q_j = c(v) + α_j ray
visible(t_i | v) = true iff min_dist(q_j, blocker_points) >= τ_occ for all q_j
```

推荐：

```text
τ_occ = occlusion_radius_m = 0.05m
max_ray_sample_count = 1000
target_sample_count = 1000
```

注意：blocker_points 不应包含 target points，否则会自遮挡。

### 4.3 Target-Distance Shell

定义候选视点到目标表面的距离：

```text
d_T(v) = min_i ||v - t_i_ground||
```

可行 shell：

```text
ρ_min ≤ d_T(v) ≤ ρ_max
```

推荐：

```text
ρ_min = 0.35m
ρ_max = 1.20m
```

这比固定 `0.5,0.75` 两圈更自然，因为它搜索一个厚壳，而不是两个离散圆。

### 4.4 Clearance

定义候选点到非目标点云 / 障碍点云的最近距离：

```text
C(v) = min_j ||v - b_j||
```

其中 `b_j` 来自 blocker_points 或 scene blocker KDTree。

安全约束：

```text
C(v) ≥ c_min
```

推荐：

```text
c_min = 0.10m 或 0.15m
```

注意：第一版不要让 clearance 太硬，否则可能像后续 confidence guard 一样压低 SR。可以先记录 `clearance_ok`，只在特别危险时过滤。

## 5. 联合可行域

联合可行域定义为：

```text
F = { v ∈ Ω |
      R(v)=1,
      V(v) ≥ τ_v,
      ρ_min ≤ d_T(v) ≤ ρ_max,
      C(v) ≥ c_min }
```

推荐初始：

```text
τ_v = 0.0 或 0.02
```

SR 是主指标，SPL 仅参考，所以 `τ_v` 不要设高。

## 6. Level-Set / Connected Component

将 `F` 离散成局部图：

```text
G = (F, E)
```

如果两个候选点满足以下任一条件，则连边：

```text
相邻 radius bin 且相邻 angle bin
或 ||v_i - v_j|| < 1.5 * grid_step
```

提取连通分量：

```text
F = ⋃_k C_k
```

不要把所有指标揉成一个总分。采用 lexicographic selection：

1. 丢弃 `|C_k| < min_component_size` 的分量
2. 优先选择 `|C_k|` 最大的分量
3. 若多个分量 size 接近，选 `P75(V)` 更高者
4. 若仍接近，选平均 follower path action count 更低者
5. 在选中分量内取 medial center

推荐：

```text
min_component_size = 3 或 5
size_tie_ratio = 0.85
```

含义：若一个分量大小达到最大分量的 85%，就可以用 visibility 分位数继续比较。

## 7. Medial Center

选中最佳连通分量：

```text
C*
```

定义分量边界：

```text
∂C* = { v ∈ C* | exists neighbor u not in C* }
```

定义每个点到分量边界的图距离：

```text
m(v) = dist_G(v, ∂C*)
```

medial center：

```text
v* = argmax_{v ∈ C*} m(v)
```

如果多个点并列：

```text
v* = argmax visibility_score(v)
```

直觉：这个点不是贴边的偶然可行点，而是在可行区域内部最稳定的位置。它对点云噪声、navmesh snap、遮挡小扰动更稳。

## 8. Fallback

如果 `F` 为空，按以下顺序放松：

1. 放宽 visibility：

```text
τ_v → 0.0
```

2. 放宽 shell：

```text
[ρ_min, ρ_max] → [0.30, 1.35]
```

3. 暂时关闭 clearance hard constraint，仅记录：

```text
clearance_ok = false
```

4. fallback 到现有 first-max followable visibility 策略。

5. 如果仍失败，keep baseline，并记录：

```text
rejected_reason = no_vista_ls_feasible_component
```

重要：fallback 必须写清楚原因，不能静默回退。

## 9. 和当前代码集成

主要改：

- `hm3d-online/anchor_nav/mile.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py`
- `scripts/mile-all-0.05-0.5.sh`

最小侵入原则：

- 不改 `correct_final_decision_with_mile()` 的签名
- 在 `MileConfig` 新增 VISTA-LS 参数
- 在 `visibility_based_viewpoint_decision()` 内按开关使用 VISTA-LS candidate generator / selector
- 保持 `visibility_records` 中仍有 `visibility_score`
- 保持旧分析脚本兼容

建议新增 `MileConfig` 字段：

```python
vista_ls_enabled: bool = False
vista_ls_r_min_m: float = 0.30
vista_ls_r_max_m: float = 1.30
vista_ls_radial_step_m: float = 0.05
vista_ls_angle_step_deg: float = 5.0
vista_ls_shell_min_m: float = 0.35
vista_ls_shell_max_m: float = 1.20
vista_ls_min_visibility: float = 0.02
vista_ls_min_clearance_m: float = 0.10
vista_ls_min_component_size: int = 3
vista_ls_size_tie_ratio: float = 0.85
```

建议新增 CLI：

```bash
--mile_enable_vista_ls
--mile_vista_ls_r_min_m
--mile_vista_ls_r_max_m
--mile_vista_ls_radial_step_m
--mile_vista_ls_angle_step_deg
--mile_vista_ls_shell_min_m
--mile_vista_ls_shell_max_m
--mile_vista_ls_min_visibility
--mile_vista_ls_min_clearance_m
--mile_vista_ls_min_component_size
```

## 10. 新增日志字段

顶层 visibility：

```json
{
  "selection_policy": "vista_ls_level_set_medial_center",
  "vista_ls_enabled": true,
  "vista_ls_component_count": 3,
  "vista_ls_selected_component_id": 1,
  "vista_ls_selected_component_size": 18,
  "vista_ls_selected_medial_radius": 2,
  "vista_ls_fallback_used": false,
  "vista_ls_rejected_reason": null
}
```

每条 `visibility_records[]`：

```json
{
  "candidate_index": 12,
  "viewpoint_xyz": [0.0, 0.0, 0.0],
  "visibility_score": 0.41,
  "reachable_ok": true,
  "shell_ok": true,
  "clearance_ok": true,
  "clearance_m": 0.18,
  "target_surface_distance_m": 0.72,
  "component_id": 1,
  "component_size": 18,
  "medial_distance_to_boundary": 2,
  "selected_by_vista_ls": true
}
```

聚合计数：

```json
{
  "vista_ls_used": 123,
  "vista_ls_fallback": 12,
  "vista_ls_error": 0
}
```

## 11. 可行性验证例子

### 例 1：目标贴墙

固定 `0.5/0.75m` 环形采样可能采到墙内、不可达点，或者 visibility 高但贴边的尖点。

VISTA-LS 的预期行为：

```text
可行域 F 形成一段半月形 connected component。
模块选择半月形区域的 medial center。
```

好处：不会选墙边缘的偶然可行点，follower 成功率更稳。

### 例 2：目标被局部遮挡

单点 highest visibility 可能来自一个很窄的 ray gap，对点云噪声敏感。

VISTA-LS 的预期行为：

```text
选更大的 visible ∩ reachable component。
允许牺牲一点 visibility，换取区域稳定性。
```

这更符合 SR 目标，因为真正执行时稳定可达比单次 visibility 尖峰更重要。

### 例 3：baseline 近但 follower 不稳定

baseline 已在目标附近，但 GreedyGeodesicFollower precheck 失败或路径不稳定。

VISTA-LS 的预期行为：

```text
在 target-distance shell 内找 follower-precheck 可达区域。
如果找到，替换 baseline。
如果找不到，fallback 到 first-max 或 keep baseline，并记录原因。
```

## 12. 最小测试

1. 静态检查：

```bash
python -m py_compile hm3d-online/anchor_nav/mile.py
python -m py_compile hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py
```

2. 纯函数 fake test：

- fake `rep`
- fake `path_finder`
- 构造一个目标点云和障碍点云
- 验证 `visibility_based_viewpoint_decision()` 开启 VISTA-LS 后能输出：
  - `selection_policy=vista_ls_level_set_medial_center`
  - `vista_ls_component_count`
  - `component_id`
  - `medial_distance_to_boundary`
  - `selected_by_vista_ls`

3. 远端最小跑：

```bash
python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py \
  --start_ratio 0.05 \
  --end_ratio 0.5 \
  --max_eval_tasks 10 \
  --mile_enable_vvd_replacement \
  --mile_disable_visible_baseline_guard \
  --mile_enable_vista_ls \
  --mile_candidate_radii_m "0.5,0.75" \
  --mile_camera_height_m 1.50 \
  --mile_apply_task_levels "object,room,region,instance" \
  --mile_apply_non_final_object_decisions \
  --mile_enable_navigation_target_repair
```

4. 指标验收：

至少跑 110 个 same-sample，和 completed baseline exact count 对齐。

目标：

```text
count=110: delta SR 接近或超过 +0.09
count=145: delta SR 接近或超过 +0.06
count=170: delta SR 接近或超过 +0.05
```

如果达不到当前 best rerun2 / firstmax_sr110 家族，不要开全量。

## 13. 推荐实现策略

第一版不要做太多 hard guard。VISTA-LS 的卖点是：

```text
约束可行域 + 连通区域稳定性 + medial center
```

不是：

```text
手工加权打分器
```

因此实现优先级：

1. 先实现 dense local candidate field
2. 再实现 feasible mask
3. 再实现 connected components
4. 再实现 medial center
5. 最后再考虑局部细化和 adaptive threshold

模块描述：

> VISTA-LS models last-mile viewpoint selection as a local feasible-region search. It constructs the intersection of reachable, visible, safe, and target-distance-shell constraints around the target, extracts connected feasible components, and selects a robust medial center rather than a hand-scored discrete candidate.

中文描述：

> VISTA-LS 将 last-mile 目标点修正建模为局部几何可行域搜索：在目标周围求可达、可见、安全距离、目标距离壳层的交集，提取最大稳定连通区域，并选择区域中心作为最终观察点。

