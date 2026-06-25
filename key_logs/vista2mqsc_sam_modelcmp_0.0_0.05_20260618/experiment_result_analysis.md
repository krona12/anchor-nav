# Vista2MQSC SAM Model Comparison 0.0-0.05

## Experiment Scope

- Method: Vista2MQSC, MQSC-R1 followed by VISTA-LS
- Segmenter: SAM
- Slice: `start_ratio=0.0`, `end_ratio=0.05`
- Task levels: `object,room,region,instance`
- Records per model: 100 sequence rows
- Source output root: `output_logs/anchor/vista2mqsc_sam_all_0.0_0.05`
- Key log archive: `key_logs/vista2mqsc_sam_modelcmp_0.0_0.05_20260618`

The files in each model subdirectory are copied from the original experiment output without editing the log contents:

- `refhm3d-nav-sequence-analyze-vista2mqsc-refine1-*.log`
- `tmux_*.log`
- `run_args.txt`
- `refhm3d_seq_vista2mqsc_refine1_0.0_0.05.json`
- `refhm3d_seq_vista2mqsc_refine1_effectiveness_0.0_0.05.json`

## Overall Results

| Rank | Model | CUDA | Count | SR | SPL | Avg Task Time Sec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `qwen-vl-plus` | 3 | 100 | **0.6200** | **0.4373** | 1092.041 |
| 2 | `gpt-4o-mini` | 0 | 100 | 0.6000 | 0.4202 | 1086.473 |
| 3 | `qwen3-vl-8b-instruct` | 2 | 100 | 0.5900 | 0.4012 | 1003.802 |
| 4 | `gemini-2.5-flash` | 0 | 100 | 0.5600 | 0.3991 | 1110.712 |

`qwen-vl-plus` is the best model on this shard by both SR and SPL. Its advantage over `gpt-4o-mini` is +0.0200 SR and +0.0171 SPL. `gpt-4o-mini` is second overall and is strongest on object and region levels. `qwen3-vl-8b-instruct` is close to `gpt-4o-mini` on object and region SR, but loses on instance and room. `gemini-2.5-flash` is the weakest overall in this run.

## Level-Wise Results

Values are `SR / SPL`.

| Model | Instance, n=20 | Object, n=15 | Region, n=39 | Room, n=26 |
|---|---:|---:|---:|---:|
| `qwen-vl-plus` | **0.5500 / 0.3101** | 0.6667 / 0.5506 | 0.5641 / 0.3936 | **0.7308 / 0.5353** |
| `gpt-4o-mini` | 0.4000 / 0.1719 | **0.7333 / 0.5781** | **0.6154 / 0.4503** | 0.6538 / 0.4749 |
| `qwen3-vl-8b-instruct` | 0.4000 / 0.2250 | **0.7333 / 0.5614** | **0.6154 / 0.4297** | 0.6154 / 0.4014 |
| `gemini-2.5-flash` | 0.4000 / 0.1972 | 0.5333 / 0.4162 | 0.5897 / 0.4296 | 0.6538 / 0.4987 |

## Hook and Module Behavior

| Model | Hook Called | Hook Applied | Helpful True | Helpful False | MQSC Called | MQSC Applied | VISTA-LS Called | VISTA-LS Applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen-vl-plus` | 100 | 91 | 29 | 62 | 100 | 33 | 100 | 86 |
| `gpt-4o-mini` | 100 | 91 | 33 | 58 | 100 | 33 | 100 | 89 |
| `qwen3-vl-8b-instruct` | 100 | 91 | 27 | 64 | 100 | 30 | 100 | 88 |
| `gemini-2.5-flash` | 100 | 89 | 29 | 60 | 100 | 29 | 100 | 86 |

All four runs exercised the VLM path: MQSC-R1 was called for every record, and VISTA-LS was also called for every record. Differences in final performance therefore reflect model-dependent decomposition behavior plus downstream VISTA-LS correction outcomes, not a missing VLM invocation.

## Interpretation

`qwen-vl-plus` wins this shard because it performs much better on the two levels that are hardest for this setup to stabilize: instance and room. Its instance SPL is 0.3101, clearly above the other three models, and its room SPL is also the best.

`gpt-4o-mini` remains a strong baseline. It is best on object-level tasks and tied for best SR on region-level tasks, but its instance performance is low, which pulls down the overall result.

`qwen3-vl-8b-instruct` is competitive on object and region SR, but it does not translate that into the best overall SPL. It trails `gpt-4o-mini` overall and is clearly behind `qwen-vl-plus`.

`gemini-2.5-flash` has the lowest overall SR and SPL in this specific shard. Its room SPL is solid, but object and instance results are weaker.

## Caveats

This is a single `0.0-0.05` shard with 100 task rows. It is useful as a controlled model comparison on the same script and slice, but the ordering should be verified on additional shards before treating it as a final global ranking.

