# MQSC-R1 实现说明

日期: 2026-05-14  
模块名: `mqsc-r1`  
核心文件:

- `hm3d-online/anchor_nav/mqsc_r1.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-mqsc-r1-refine1.py`
- `scripts/run_mqsc_r1_all_0.05_0.1.sh`
- `scripts/run_mqsc_r1_all_0.0_0.2.sh`

## 1. 模块定位

MQSC-R1 是 Multi-Query Spatial Consensus 的第一轮优化版本。它不是一个独立导航器，也不是重新训练后的模型，而是一个 test-time final target refinement module。

它插在 RefHM3D sequence navigation 的 final decision 后面:

```text
PQ3D / MTU3D normal loop
  -> scan frames
  -> frontier detection
  -> PQ3DModel.decision(...)
  -> if is_final:
       MQSC-R1 refine target object
     else:
       keep frontier target
  -> GreedyGeodesicFollower / repaired follower
```

R1 的关键优化是:

```text
room_context 只保留为诊断信息，不被实例化为独立 query，也不作为独立 consensus role。
```

注意: `q_full` 仍然是原始完整指令，因此完整句子里可能仍含有 `bedroom`、`bathroom`、`kitchen` 等 room words。R1 的设计不是从语言中删除房间词，而是不再让 `room_context` 单独投出一组 object-level votes。

原因是当前 Stage2 / VLE grounding 明显偏 object-centric。`room`、`bedroom`、`bathroom` 这类词并不会稳定落到房间区域，反而容易被转译成床、马桶、柜子等 object prior，破坏 room-level 和 region-level 任务的最终目标选择。因此 R1 实现为:

```text
full instruction + target + object/fixture anchors
  -> object-only role-aware region consensus
```

## 2. 配置项

配置类在 `hm3d-online/anchor_nav/mqsc_r1.py`:

```python
@dataclass
class MqscR1Config:
    top_k: int = 8
    temperature: float = 1.0
    cluster_eps: float = 1.2
    min_region_coverage: float = 0.5
    min_target_prob: float = 0.05
    min_region_margin: float = 0.05
    min_selected_gain: float = -0.02
    use_vlm: bool = True
    vlm_model: str = DEFAULT_MODEL
    vlm_max_retries: int = 3
    vlm_retry_sleep_sec: float = 1.0
    vlm_no_proxy: bool = True
    allow_heuristic_decompose: bool = True
    write_debug_json: bool = True
    exclude_room_context_query: bool = True
    excluded_consensus_roles: Tuple[str, ...] = ("room_context",)
```

默认 role 权重:

```text
full           0.80
target         1.35
anchor_primary 0.90
anchor_support 0.70
```

这些权重只用于 region score 和 target score 的启发式重排，不是训练学习得到的参数。

## 3. 批量入口与导航接口

批量评估脚本是:

```text
hm3d-online/refhm3d-nav-sequence-analyze-anchor-mqsc-r1-refine1.py
```

它保留 refine1 scaffold:

- scene slicing: `--start_ratio`, `--end_ratio`
- episode resume: 若主 output JSON 已存在，则跳过已完成 episode
- sequence task loop: object / room / region / instance
- per-decision process JSON
- live metrics log
- task summary / effectiveness JSON

MQSC-R1 的唯一插入点是:

```python
if bool(is_final):
    used_target, module_info = mqsc_r1_refine_hook(...)
```

hook 输入:

```text
sentence: 当前任务文本
task_type: object / room / region / instance
scene_name
episode_id
task_id
decision_num
is_final
pq3d_model
target_position: PQ3D baseline final target
output_dir
```

hook 输出:

```text
new_target_xyz: shape=(3,)
module_info: JSON-serializable diagnostic dict
```

非 final decision 不会触发 MQSC-R1。frontier exploration 阶段保持 baseline 行为。

## 4. 任务文本构造

脚本中的 `build_sentence(...)` 把 RefHM3D annotations 统一成导航文本:

```text
object:
  object_category

room:
  object_category in the room_name

region:
  object_category in the region_category that has region_description

instance:
  annot_unique_detailed_description / annot_unique_concise_description
```

输出:

```text
sentence
goal_category
```

`sentence` 是 MQSC-R1 的语言输入，也是 baseline PQ3D decision 的语言输入。

## 5. 语言分解

函数:

```python
decompose_navigation_text(description, task_type, cfg)
```

优先调用 VLM，prompt 版本:

```text
mqsc_r1_decompose_v1_object_only_region_consensus
```

实现注意: 当前 prompt 文本仍让 VLM 输出 `room_context`，这与 R1 需要保留 room diagnostic 一致；但论文或后续代码注释中最好明确写成:

```text
The downstream algorithm queries full task, target, and object-level anchors;
room context is parsed for diagnostics only.
```

这样可以避免复现者误以为 R1 会单独调用 room query。

VLM 必须输出严格 JSON:

```json
{
  "target_desc": "string",
  "target_aliases": ["string"],
  "anchor_primary": ["string"],
  "anchor_support": ["string"],
  "room_context": ["string"],
  "relations": ["string"]
}
```

语义约束:

- `target_desc` 是真正要导航到的 object，不是房间或 anchor。
- `target_aliases` 是同一目标的短别名。
- `anchor_primary` 是最强局部参照物，最多 2 个。
- `anchor_support` 是辅助局部参照物，最多 4 个。
- `room_context` 是房间或区域词，但在 R1 中只做诊断。
- 不允许把 target 本身放入 anchor 或 room_context。

如果 VLM 失败，模块使用 `_heuristic_decompose(...)` fallback。fallback 通过正则处理常见模式:

```text
target in the room
target in the region that has anchors
target near / beside / next to / on / under anchors
```

分解结果再经过:

```python
sanitize_query_spec(...)
```

它会:

- 小写规范化；
- 去除标点与重复短语；
- 限制 alias / anchor 数量；
- 过滤与 target 重合的 anchor；
- 保留 VLM raw output、parse attempt 和错误信息。

## 6. Role Query 构造

函数:

```python
build_role_queries(description, query_spec)
```

R1 实际构造的 query roles:

```text
full:
  原始完整任务文本

target:
  target_desc + target_aliases

anchor_primary:
  最多 2 个主 anchor

anchor_support:
  最多 4 个辅助 anchor
```

虽然分解结果里有 `room_context`，但 R1 不创建独立 room query。换句话说，room words may still appear in `q_full = sentence`，但它们不能作为单独的 consensus role 去投票。

对应日志:

```json
"r1_policy": {
  "exclude_room_context_query": true,
  "excluded_consensus_roles": ["room_context"],
  "room_context_diagnostic_only": [...],
  "reason": "stage2_object_grounding_is_object_centric_room_words_are_noisy_priors"
}
```

这也是 MQSC-R1 相比初版 MQSC 的核心差异。

## 7. 重复调用 Stage2 Object Grounding

函数:

```python
pq3d_stage2_object_logits(pq3d_model, sentence)
```

输入来自当前 `PQ3DModel` 的 object memory:

```text
representation_manager.object_box
representation_manager.object_feat
representation_manager.object_score
representation_manager.open_vocab_feat
tokenizer
pq3d_stage2
```

它不重新执行:

- FastSAM / DINO / detection
- stage1 pretrain encoder
- point cloud merge
- map update

它只复用当前 memory object tensors，针对每个 role query 重新构造 Stage2 batch:

```python
encoded_input = tokenizer([query_text], add_special_tokens=True, truncation=True)
stage2_output = pq3d_stage2(batch)
logits = stage2_output["og3d_logits"]
```

最终只保留 real object logits:

```python
mask = stage2_output["real_obj_pad_masks"].bool()
return logits[mask]
```

R1 当前不对 frontier logits 做 MQSC。

每个 query 内部做 temperature softmax:

```text
p_r(o_i) = exp(l_r(o_i) / T) / Σ_j exp(l_r(o_j) / T)
```

然后取 top-k:

```text
K_r = TopK_o p_r(o), k = 8
```

日志字段:

```text
query_summaries:
  role, text, ok, top_indices, top_probs, top_logits

candidate_hits:
  object_id, rank, role, query, prob, logit,
  xy, footprint_radius,
  box_model_xzydxdzdy,
  center_habitat_xyz,
  merged_object_score,
  object_count
```

## 8. 坐标与 Footprint 建模

MTU3D object box 使用模型坐标:

```text
[x, z, y, dx, dz, dy]
```

因此 top-down 平面使用:

```python
xy = object_box[:, [0, 1]]
```

而 Habitat 坐标中心输出为:

```python
center_habitat_xyz = [x, y, z]
```

也就是:

```python
model_box_to_habitat_xyz(box) = [box[0], box[2], box[1]]
```

Footprint radius:

```text
rho_i = clip(0.5 * max(dx_i, dz_i), 0.05, 2.5)
```

这样大物体不会只用中心点参与聚类，而会以 footprint 影响局部区域连接。

## 9. Footprint-aware Connected Components

候选 object 来自所有 role query 的 top-k union。

两个 object 的 footprint distance 定义为:

```text
d_fp(i,j) = max(0, ||xy_i - xy_j||_2 - rho_i - rho_j)
```

若:

```text
d_fp(i,j) <= epsilon_cluster
```

则连边。R1 默认:

```text
epsilon_cluster = 1.2m
```

聚类由 `_connected_components(...)` 实现，不使用 sklearn DBSCAN。

输出 region:

```text
R_1, R_2, ..., R_m
```

每个 region 是一组 object ids。

## 10. Role-aware Region Evidence

对每个 role `r` 和 region `R`，R1 用 top-k 截断后的 bounded soft-union 汇聚同一 region 内的候选分数。先定义:

```text
tilde_p_r(i) =
  p_r(i), if o_i in K_r
  0,      otherwise
```

其中 `K_r` 是 role `r` 的 top-k 命中集合。然后:

```text
E_r(R) = 1 - Π_{o_i in R} (1 - tilde_p_r(i))
```

这不是校准概率模型，而是 role-normalized score 的有界 soft-union aggregator。显式 top-k 截断非常重要: softmax 后所有 object 理论上都有非零概率，但实现中只有 top-k hits 能对 region evidence 做贡献。

Required roles:

```text
required = {full, target}
if anchor_primary exists: add anchor_primary
if anchor_support exists: add anchor_support
remove excluded_consensus_roles
remove roles without query
```

其中 `room_context` 默认被排除。

Coverage:

```text
Coverage(R) = |{r in required : E_r(R) > delta}| / |required|
delta = 1e-9
```

Compactness:

```text
C(R) = exp(- mean_i ||xy_i - mean(xy_R)||_2^2 / sigma^2)
```

其中:

```text
sigma = max(cluster_eps, 0.5)
```

Region score:

```text
S(R) =
  Σ_{r in required} w_r log(1e-6 + E_r(R))
  + 1.25 * Coverage(R)
  + 0.25 * C(R)
  - NoisePenalty(R)
```

当前 noise penalty 是轻量规则。定义:

```text
n_r(R) = number of top-k hits in R assigned to role r
```

则:

```text
if Coverage(R) < 0.5 and max_r n_r(R) >= 3:
    NoisePenalty(R) = 0.2
else:
    NoisePenalty(R) = 0
```

直觉是: 如果一个区域只被单一 role 重复命中，却缺少 target/full/anchor 的跨角色覆盖，则它更像语义噪声，而不是空间共识。

所有 regions 按 `S(R)` 降序排序。

Region margin:

```text
Delta_R = S(R_best) - S(R_second)
```

若只有一个 region，则 margin 为 `inf`。

## 11. Best Region 内的 Target Selection

MQSC-R1 只在 best region 内选择 target-role object，避免把 anchor 选成最终目标。

Target pool:

```text
P_target(R*) = {o_i in R* : o_i 被 target role 命中}
```

如果 best region 没有 target role object:

```text
reason = best_region_has_no_target_role_object
fallback baseline target
```

对每个 target candidate:

```text
Rel(o_i, R*) =
  max_{o_j in R*, j != i, role_j in {full, anchor_primary, anchor_support}}
    exp(- d_fp(i,j)^2 / (2 * 1.2^2))
```

Target score:

```text
T(o_i | R*) =
  1.45 * log(1e-6 + p_target(o_i))
  + 0.75 * log(1e-6 + p_full(o_i))
  + 0.45 * Rel(o_i, R*)
  + 0.05 * log(max(object_score_i, 1e-6))
  + 0.03 * log(max(object_count_i, 1))
```

最终:

```text
o* = argmax_{o_i in P_target(R*)} T(o_i | R*)
```

输出:

```text
selected_object_index
selected_target_position
target_confidence
selected_candidates
```

## 12. Baseline 对比与覆盖门控

baseline object index 来自:

```python
pq3d_model.last_decision_aux["real_object_decision_idx"]
```

若 baseline index 有效，计算 baseline target score:

```text
T_baseline_gate =
  1.45 * log(1e-6 + p_target(o_base))
  + 0.75 * log(1e-6 + p_full(o_base))
  + 0.45 * Rel(o_base, R*)
```

Selected gain:

```text
G = T(o*) - T_baseline_gate
```

实现细节: 当前 `T(o*)` 包含 `object_score` 和 `object_count` 的稳定性项，而 `T_baseline_gate` 只包含 semantic + relation 项。这是一个启发式 gate，而不是严格可校准 likelihood ratio。写论文主文时建议将其表述为 conservative replacement gate，并在 appendix 中说明实现；若要做更严格的 camera-ready 公式，可以统一 baseline 与 selected 的打分口径，或报告该 gate 的 ablation。

如果 baseline index 无效，则:

```text
G = inf
```

覆盖 baseline 的门控条件:

```text
Coverage(R*) >= min_region_coverage
p_target(o*) >= min_target_prob
Delta_R >= min_region_margin
G >= min_selected_gain
o* != o_baseline
```

默认阈值:

```text
min_region_coverage = 0.5
min_target_prob = 0.05
min_region_margin = 0.05
min_selected_gain = -0.02
```

如果任何门控失败:

```text
applied = false
target_after = target_before
reason = comma-joined fail reasons
```

常见失败原因:

```text
coverage_below_threshold
target_prob_below_threshold
region_margin_below_threshold
selected_gain_below_threshold
selected_matches_baseline
```

如果通过:

```text
applied = true
reason = mqsc_selected_higher_consensus_target
target_after = selected_target_position
```

如果执行异常:

```text
ok = false
applied = false
reason = mqsc_exception_fallback_baseline
target_after = target_before
```

## 13. Follower 修复

MQSC-R1 batch 脚本保留了导航 target repair，避免 final target 或 frontier target 不在可行 navmesh 上时直接失败。

函数:

```python
_candidate_follow_targets(...)
_find_follow_actions_with_repair(...)
_follow_target(...)
```

候选包括:

- direct snap target
- 以 raw target 为中心的 repair rings

半径:

```text
0.35, 0.50, 0.75, 1.00, 1.25, 1.50 m
```

每个半径采样 16 个方向。候选 target 会按:

```text
path found first
small target_l2_from_raw
short geodesic distance
```

排序尝试。`follow_log` 会记录:

```text
ok
raw_target
snapped_target
candidate_source
candidate_radius
repair_applied
target_l2_from_raw
shortest_path_found
shortest_path_geodesic_distance
candidate_attempt_count
candidate_attempts_head
```

## 14. 日志结构

每个 final decision 的 MQSC debug JSON:

```text
process/scene=<scene>/episode=<episode>/task=<task>/mqsc-r1/dec_XXX_mqsc_r1.json
```

主 decision JSON:

```text
process/scene=<scene>/episode=<episode>/task=<task>/dec_XXX_mqsc_r1.json
```

主 decision JSON 包含:

```text
task_id
decision_num
is_final
target_used
pq3d_aux
module
module_info
follow
steps_total_after_follow
```

Task summary:

```text
summary.json
effectiveness.json
```

主结果 JSON row:

```text
scene_name
episode_id
task_id
task_level
navigation_type
sr
spl
object_category
sentence
task_time_sec
steps_total
decisions_total_episode_counter
task_decision_start
task_decision_count
end_reason
start_goal_geo
end_goal_geo
episode_cum_distance
module_name
module_hook_called
module_hook_applied
module_helpful
goal_positions
baseline_target_position
selected_target_position
baseline_target_to_goal_l2
selected_target_to_goal_l2
baseline_target_to_goal_l2_valid
selected_target_to_goal_l2_valid
pq3d_object_top1_top2_logit_gap
```

`module_helpful` 是事后诊断:

```text
module_helpful =
  selected_target_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6
```

只有 `hook_applied > 0` 且两个距离有效时才会计算。

## 15. 批量脚本参数

0.05-0.1:

```text
scripts/run_mqsc_r1_all_0.05_0.1.sh
```

0.0-0.2:

```text
scripts/run_mqsc_r1_all_0.0_0.2.sh
```

常用环境变量:

```text
MQSC_R1_SEED
MQSC_R1_TOP_K
MQSC_R1_TEMPERATURE
MQSC_R1_CLUSTER_EPS
MQSC_R1_MIN_REGION_COVERAGE
MQSC_R1_MIN_TARGET_PROB
MQSC_R1_MIN_REGION_MARGIN
MQSC_R1_MIN_SELECTED_GAIN
MQSC_R1_VLM_MODEL
MQSC_R1_VLM_MAX_RETRIES
MQSC_R1_VLM_RETRY_SLEEP_SEC
MQSC_R1_DISABLE_VLM
MQSC_R1_DISABLE_NO_PROXY
MQSC_R1_DISABLE_HEURISTIC_DECOMPOSE
MQSC_R1_DISABLE_DEBUG_JSON
```

脚本会写:

```text
run_args.txt
run_args.json
run_command.sh
script snapshot
```

注意: 当前模板要求 tmux 长跑不要把主 stdout/stderr 重定向到文件；脚本内部的 Python 运行命令保持直接输出，便于 attach tmux 时查看实时日志。

## 16. 实现边界

MQSC-R1 当前没有实现以下内容:

- 不使用 room_context 参与 consensus；
- 不做 frontier-level MQSC override；
- 不做显式 3D relation assignment，例如 `on`, `under`, `between`；
- 不重新训练模型；
- 不替代 PQ3D memory 或 Stage2，只复用它们；
- 不保证每次 applied 都提升，提升需要靠日志和指标验证。

更准确的论文表述是:

```text
MQSC-R1 is a test-time, object-centric, role-aware spatial consensus refiner
for final object decisions.
```

而不是:

```text
MQSC-R1 is a complete navigation policy or learned spatial reasoning model.
```

## 17. 论文写作与 Reviewer 风险

如果把 MQSC-R1 写进论文主方法，建议主文采用更克制的定位:

```text
MQSC-R1 is a test-time, object-centric, role-aware consistency prior
for final object grounding.
```

不要把它写成完整的人类认知模型。更稳的叙事是:

```text
long referring expressions contain target cues and landmark cues;
the target cue should determine the final object,
while landmark cues should localize a coherent neighborhood.
```

最可能的 reviewer attack 与对应写法:

1. 手工阈值与权重太多。

   应对: 报告固定参数表，并做 `top_k`, `cluster_eps`, `min_target_prob`, `min_region_coverage` 的小范围 sensitivity。

2. 排除 room role 是否 cherry-pick。

   应对: 做 ablation:

   ```text
   full + target + anchors + room role
   vs.
   full + target + anchors only
   ```

   同时统计 room query 常命中的典型物体，例如 bedroom -> bed/wardrobe，bathroom -> toilet/sink/mirror。

3. Softmax + noisy-or 不是真概率。

   应对: 写成 bounded soft-union over role-normalized scores，不声称 calibrated posterior。

4. VLM decomposition 引入额外模型。

   应对: 报告 VLM model、解析失败率、fallback 占比，并补 without-VLM / heuristic-only ablation。

5. 只在 final decision 生效，不能解决探索错误。

   应对: 主动承认它是 final object refiner。探索阶段留给主导航器。

6. Footprint proximity 不是显式 relation reasoning。

   应对: 写成:

   ```text
   We approximate relational support by footprint proximity
   rather than explicit scene-graph relation parsing.
   ```

7. 提升可能来自 follower repair，而不是 MQSC-R1。

   应对: 实验中分开比较:

   ```text
   baseline
   baseline + follower repair
   baseline + follower repair + MQSC-R1
   ```

主文保留方法骨架即可，工程日志、tmux、输出路径和 JSON 字段更适合 appendix 或 repo documentation。
