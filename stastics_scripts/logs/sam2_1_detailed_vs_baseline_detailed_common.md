# SAM2.1 detailed vs baseline detailed

## Technical summary

- 本报告只比较 baseline detailed 与 SAM2.1 detailed 的共集样本；共集 n=1520，baseline-only n=4911，SAM2.1-only n=0。
- 共集总体：baseline SR=34.61%, SPL=0.1822；SAM2.1 SR=36.25%, SPL=0.1981；差值 SR=+1.64 pp, SPL=0.0160。
- 结果层面：SAM2.1 救回 baseline 错误 199 条，把 baseline 正确样本变错 174 条，净成功数变化 25。
- 目标距离层面：有效 L2 对 n=1445；selected target 更接近目标 578 条 (40.00%)，更远 793 条 (54.88%)。

## Scope and definitions

- Generated at: `2026-06-30T20:33:05`
- Baseline files: 14 files
- SAM2.1 files: 6 files
- Sample key: `(task_level, navigation_type, scene_name, episode_id, task_id)`.
- Head categories: top 73 / 361 object categories by full LangMap task frequency (77.03% of full tasks).
- Long-tail categories: remaining 288 categories (22.97% of full tasks).
- Outcome cases use `baseline_sr -> ours_sr`: `00` both wrong, `01` rescue, `10` regression, `11` both correct.
- Target correction uses `baseline_target_to_goal_l2 - selected_target_to_goal_l2`; positive means the SAM2.1 selected target is closer to the goal.

## Overall SR/SPL

| segment | n | baseline SR | baseline SPL | SAM2.1 SR | SAM2.1 SPL | SR delta | SPL delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1520 | 34.61% (526/1520) | 0.1822 | 36.25% (551/1520) | 0.1981 | +1.64 pp | 0.0160 |
| object | 344 | 32.27% (111/344) | 0.1648 | 31.69% (109/344) | 0.1712 | -0.58 pp | 0.0065 |
| room | 480 | 33.96% (163/480) | 0.1754 | 30.63% (147/480) | 0.1695 | -3.33 pp | -0.0059 |
| region | 373 | 37.27% (139/373) | 0.1936 | 45.04% (168/373) | 0.2387 | +7.77 pp | 0.0451 |
| instance | 323 | 34.98% (113/323) | 0.1975 | 39.32% (127/323) | 0.2225 | +4.33 pp | 0.0249 |

## Head vs long-tail SR/SPL

| bucket | n | baseline SR | baseline SPL | SAM2.1 SR | SAM2.1 SPL | SR delta | SPL delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1088 | 38.33% (417/1088) | 0.2010 | 39.34% (428/1088) | 0.2150 | +1.01 pp | 0.0139 |
| long_tail | 432 | 25.23% (109/432) | 0.1347 | 28.47% (123/432) | 0.1557 | +3.24 pp | 0.0210 |

## Outcome correction by level

| segment | n | rescue 01 | correction rate | regression 10 | regression rate | net success | net SR delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1520 | 199 | 20.02% | 174 | 33.08% | 25 | +1.64 pp |
| object | 344 | 35 | 15.02% | 37 | 33.33% | -2 | -0.58 pp |
| room | 480 | 37 | 11.67% | 53 | 32.52% | -16 | -3.33 pp |
| region | 373 | 70 | 29.91% | 41 | 29.50% | 29 | +7.77 pp |
| instance | 323 | 57 | 27.14% | 43 | 38.05% | 14 | +4.33 pp |

## Target-distance correction by level

| segment | valid L2 n | closer | farther | mean delta m | median delta m | outside->inside @1.0m | inside->outside @1.0m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1445 | 578 (40.00%) | 793 (54.88%) | -0.116 | -0.241 | 38 | 402 |
| object | 335 | 127 (37.91%) | 190 (56.72%) | -0.191 | -0.400 | 3 | 92 |
| room | 458 | 184 (40.17%) | 245 (53.49%) | -0.138 | -0.180 | 8 | 131 |
| region | 343 | 130 (37.90%) | 196 (57.14%) | -0.053 | -0.366 | 14 | 105 |
| instance | 309 | 137 (44.34%) | 162 (52.43%) | -0.070 | -0.153 | 13 | 74 |

## Limitations and checks

- 当前 SAM2.1 balanced detailed 只覆盖已完成 shard；因此所有方法对比都是共集样本上的描述性统计，不是全量 9420 条任务估计。
- head/long-tail 的类别集合来自 `LangMap_Annotations` 全量任务频率，但 SR/SPL 只在当前共集里投影计算。
- 未发现重复 sample key。
- `module_helpful`/L2 纠偏是事后诊断，说明目标点更近或更远；它不等同于最终导航 SR。

## Metrics to confirm before adding

- runtime/cost: compare task_time_sec and module elapsed_ms by level and category bucket
- module routing: split by mqsc_r1_applied, vistals_applied, module_reason, and vistals_input_slot_source
- category-level gain/loss: top object categories by rescue/regression and SR delta, with minimum-n filtering
- failure-mode shift: end_reason transition matrix from baseline to SAM2.1
- SPL conditional on success: efficiency among successful tasks only, separated from SR changes

## Output artifacts

- Summary JSON: `stastics_scripts/logs/sam2_1_detailed_vs_baseline_detailed_common.json`
- Paired samples JSONL: `stastics_scripts/logs/sam2_1_detailed_vs_baseline_detailed_common_paired_samples.jsonl`
- Analysis Markdown: `stastics_scripts/logs/sam2_1_detailed_vs_baseline_detailed_common.md`
