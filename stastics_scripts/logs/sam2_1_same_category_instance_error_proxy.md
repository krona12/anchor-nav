# Same-category wrong-instance proxy

## Feasibility decision

- Exact final-navigation wrong-instance rate is not recoverable from the existing baseline result JSON/logs because they do not store final agent position or final baseline target for successful rows.
- A target-selection proxy is computable on the SAM2.1/common rows because SAM2.1 detailed results store both the unmodified `baseline_target_position` and the refined `selected_target_position`.
- This report computes that proxy only; use it to compare target choice confusion, not as a replacement for exact endpoint-based navigation analysis.

## Proxy definition

- Threshold: `0.25m`; anchor set: `hybrid`.
- A method counts as proxy same-category wrong-instance when it failed SR, its final target point is within threshold of a same-category non-target instance anchor, and it is not within threshold of any target instance anchor.
- Target instances and non-target same-category instances are read from `LangMap_Annotations` by `(scene_name, task_level, episode_id)`.

## Overall comparison

| method | classifiable | alt category inst. available | failures with alt | wrong-inst count | rate / classifiable | rate / alt | rate / failures+alt | ambiguous hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline proxy | 1451 | 710 | 459 | 65 | 4.48% | 9.15% | 14.16% | 3 |
| ours proxy | 1451 | 710 | 447 | 112 | 7.72% | 15.77% | 25.06% | 19 |

Paired switch counts: baseline-only wrong=29, ours-only wrong=76, both=36, net ours-baseline=47.

## By level

| level | rows | baseline wrong | ours wrong | baseline-only | ours-only | both | net ours-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| object | 346 | 1 | 0 | 1 | 0 | 0 | -1 |
| room | 485 | 31 | 37 | 14 | 20 | 17 | 6 |
| region | 376 | 14 | 31 | 2 | 19 | 12 | 17 |
| instance | 323 | 19 | 44 | 12 | 37 | 7 | 25 |

## Notes

- For exact endpoint measurement, future runs should save `final_agent_position`, `final_snapped_target_position`, `target_object_ids`, and nearest target/non-target same-category object ids/distances in every result row.
- If you want a stricter interpretation, rerun this script with `--point-set center` or `--point-set viewpoint`; the default `hybrid` treats either object center or annotated view point as an instance anchor.

## Artifacts

- Summary JSON: `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy.json`
- Classified rows JSONL: `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy_classified_rows.jsonl`
- Analysis Markdown: `stastics_scripts/logs/sam2_1_same_category_instance_error_proxy.md`
