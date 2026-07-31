# 同类别但实例错误率可行性分析

## 结论

当前已有结果可以计算一个 **target-selection proxy**，但不能可靠计算“最终导航到同类别但非目标实例”的精确指标。

建议：不要把当前 proxy 当成论文/主表里的最终 baseline-vs-ours 指标；可以作为诊断项使用。若要做精确指标，需要重新跑或至少补充保存每条任务的最终 agent 位置和最终 snapped target。

## 为什么精确指标当前不可算

精确口径需要判断最终 agent 是否在 `0.25m` 内到达了同类非目标实例。这至少需要：

- 每条任务的目标实例集合 `target_object_ids`：`LangMap_Annotations` 里有，可以取。
- 每个场景所有同类实例的位置和 view points：`LangMap_Annotations` 里有，可以取。
- 每个方法每条任务的最终 agent position，或至少最终 snapped target position：当前结果缺。

现有 artifact 检查结果：

- baseline detailed JSON 只有 `sr/spl/end_reason/steps_total/object_category` 等字段，没有 `final_agent_position`，也没有成功样本的最终 `target_position`。
- baseline stdout/live metrics log 只打印累计指标、任务开始、SR/SPL 和少量 follower error；成功样本没有 final target 或 final agent position。
- SAM2.1 detailed JSON 有 `baseline_target_position` 和 `selected_target_position`，但也没有最终 agent position。

因此，用现有日志无法批量恢复 baseline 的真实最终导航终点；除非重新运行或 replay simulator 并补充记录。

## 已计算的 proxy

我新增了脚本：

- `stastics_scripts/compare_same_category_instance_error_proxy.py`

proxy 定义：

- 只在 baseline 与 SAM2.1 共集样本上算。
- baseline proxy 使用 SAM2.1 结果里的 `baseline_target_position`，即 refinement 前的 final PQ3D target。
- ours proxy 使用 `selected_target_position`。
- 若方法 `SR=0`，且 final target point 在 `0.25m` 内命中同类别非目标实例 anchor，同时没有命中目标实例 anchor，则记为 proxy 同类错实例。

默认输出：

- `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy.md`
- `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy.json`
- `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy_classified_rows.jsonl`

默认 `hybrid` anchor 是 object center + annotated view points：

| method | classifiable | alt same-category available | failures with alt | proxy wrong count | rate / failures+alt |
| --- | --- | --- | --- | --- | --- |
| baseline proxy | 1451 | 710 | 459 | 65 | 14.16% |
| ours proxy | 1451 | 710 | 447 | 112 | 25.06% |

paired switch:

| case | count |
| --- | --- |
| baseline-only wrong | 29 |
| ours-only wrong | 76 |
| both wrong | 36 |
| net ours-baseline | +47 |

## 敏感性检查

这个 proxy 对 anchor 口径非常敏感：

| anchor set | baseline wrong | ours wrong | net ours-baseline | baseline rate / failures+alt | ours rate / failures+alt |
| --- | --- | --- | --- | --- | --- |
| center only | 63 | 7 | -56 | 13.73% | 1.57% |
| viewpoint only | 2 | 105 | +103 | 0.44% | 23.49% |
| hybrid | 65 | 112 | +47 | 14.16% | 25.06% |

这个现象说明：baseline 的 final target 更像 object center，而 ours 经 VISTA-LS 后的 selected target 更像 navigable viewpoint。用 raw target point 去和 center/viewpoint 做 `0.25m` 邻近判断，会把方法内部 target 表示差异混进“实例错误率”。

## 最终建议

当前可以保留 proxy 作为诊断：

- 找出 ours 是否更容易把目标修到同类别非目标实例附近。
- 找出具体 case，辅助 qualitative/debug。

但不建议把它作为最终“导航到同类别但非目标实例错误率”。精确指标应在未来结果行中保存：

- `target_object_ids`
- `final_agent_position`
- `final_raw_target_position`
- `final_snapped_target_position`
- `nearest_target_instance_id/dist`
- `nearest_same_category_non_target_instance_id/dist`
- 对 target/non-target view points 的 geodesic 或 Euclidean distance，最好和 SR 的 `0.25m` 成功判定保持同一距离定义。
