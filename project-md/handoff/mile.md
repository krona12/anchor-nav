# Mile / MSGNav VVD handoff

更新时间：2026-05-12

## 结论先行

当前 0.05-0.5 mile 实验里，需要区分两件事：

1. **指标表现最好的实验版本**是
   `output_logs/anchor/mile_all_0.05_0.5/20260511-021921-detailed-msgnav-firstmax-base-softpath-exp015-rerun2`。
   这一版在 exact-aligned baseline 上的 SR 提升最稳，尤其在 110/145/170/205 个样本处明显优于后续 rerun。
2. **完整源码仍可直接恢复的最佳代码快照**是
   `saved_versions/mile_firstmax_sr110/`。远端当前 live 代码已经和这个快照完全一致：
   `hm3d-online/anchor_nav/mile.py`、`hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py`、
   `scripts/mile-all-0.05-0.5.sh` 三个文件与快照 `diff -q` 无差异。

重要 caveat：`rerun2` 的完整 Python 源码没有保存在 `saved_versions/`，也没有在 git history 或当前源码树中找到
`path_efficiency_exponent` / `msgnav_visibility_soft_path_efficiency` 的实现。`rerun2` 目前可恢复的信息包括：
`run_args.txt`、`mile-all-0.05-0.5.sh.snapshot`、日志、输出 JSON 和 effectiveness JSON。若要精确复现 rerun2，需要基于
`saved_versions/mile_firstmax_sr110` 重新实现 soft-path efficiency policy。

## Best observed metrics

Baseline 固定为：

`output_logs/baseline_all_0.05_0.5/20260507-170755-detailed-0p5-baseline-rerun1`

指标按相同 sequence count exact-aligned 对齐，不能使用 baseline latest average 或小于等于 count 的近似值。

### Metric-best run: rerun2

目录：

`output_logs/anchor/mile_all_0.05_0.5/20260511-021921-detailed-msgnav-firstmax-base-softpath-exp015-rerun2`

关键设置来自 `run_args.txt` / log cfg：

- `uses_vlm=false`
- `mile_module=lastmile_vvd_only`
- `mile_camera_height_m=1.50`
- `mile_candidate_radii_m=0.5,0.75`
- `mile_visibility_tie_epsilon=0.0`
- `mile_path_efficiency_exponent=0.15`
- `mile_selection_policy=msgnav_visibility_soft_path_efficiency`
- `mile_enable_vvd_replacement=true`
- `mile_prefer_visible_baseline=false`
- `mile_apply_task_levels=object,room,region,instance`
- `mile_apply_non_final_object_decisions=true`
- `mile_enable_navigation_target_repair=true`

Exact-aligned checkpoints:

| count | mile SR | baseline SR | delta SR | mile SPL | baseline SPL | delta SPL |
|---:|---:|---:|---:|---:|---:|---:|
| 35 | 0.771429 | 0.571429 | +0.200000 | 0.473581 | 0.342657 | +0.130924 |
| 40 | 0.775000 | 0.575000 | +0.200000 | 0.460318 | 0.332904 | +0.127414 |
| 50 | 0.720000 | 0.540000 | +0.180000 | 0.425469 | 0.305859 | +0.119610 |
| 70 | 0.642857 | 0.500000 | +0.142857 | 0.406469 | 0.314984 | +0.091485 |
| 100 | 0.610000 | 0.480000 | +0.130000 | 0.413266 | 0.325328 | +0.087938 |
| 110 | 0.581818 | 0.481818 | +0.100000 | 0.396046 | 0.322833 | +0.073213 |
| 145 | 0.537931 | 0.468966 | +0.068965 | 0.337742 | 0.292291 | +0.045451 |
| 170 | 0.523529 | 0.464706 | +0.058823 | 0.333361 | 0.278743 | +0.054618 |
| 205 | 0.517073 | 0.478049 | +0.039024 | 0.318881 | 0.276320 | +0.042561 |

Interpretation:

- 这是所有已有 full-ish 运行里中段最强的一版；到 count=205 时 SR 只差一点点没有达到 +0.04，但 SPL 仍 +0.0426。
- 因为用户后续明确说 “只需要关注 SR，SPL 掉了也没太大关系”，后续优化应优先尝试把 rerun2 的 SR 曲线保住，而不是继续用强 guard 换 SPL。
- rerun2 的 `module-status-counts` 末尾是：
  `{'mile_applied': 176, 'mile_kept_baseline': 26, 'mile_rejected': 0, 'mile_error': 0, 'follower_error': 1}`。

## Best fully preserved code

目录：

`saved_versions/mile_firstmax_sr110/`

文件：

- `saved_versions/mile_firstmax_sr110/mile.py`
- `saved_versions/mile_firstmax_sr110/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py`
- `saved_versions/mile_firstmax_sr110/mile-all-0.05-0.5.sh`
- `saved_versions/mile_firstmax_sr110/README.md`

当前 live 文件与它一致：

- `hm3d-online/anchor_nav/mile.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py`
- `scripts/mile-all-0.05-0.5.sh`

快照关键设置：

- `uses_vlm=false`
- `camera_height=1.50`
- `candidate_radii=[0.5,0.75]`
- `enable_vvd_replacement=true`
- `prefer_visible_baseline=false`
- `selection_policy=msgnav_first_max_followable_visibility_no_baseline_threshold`
- `visibility_tie_epsilon=0.0`
- all task levels enabled
- non-final object decisions enabled
- navigation target repair enabled

这个快照对应的高 SR 表现：

| run | count | mile SR | baseline SR | delta SR | mile SPL | baseline SPL | delta SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure-vdd-height15-rerun1 / sr110 family | 110 | 0.572727 | 0.481818 | +0.090909 | 0.352425 | 0.322833 | +0.029592 |
| pure-vdd-height15-rerun1 / sr110 family | 145 | 0.510345 | 0.468966 | +0.041379 | 0.292203 | 0.292291 | -0.000088 |
| firstmax-sr110-rerun5 | 170 | 0.500000 | 0.464706 | +0.035294 | 0.285122 | 0.278743 | +0.006379 |
| firstmax-restore-rerun10 | 280 | 0.460714 | 0.442857 | +0.017857 | 0.270797 | 0.261594 | +0.009203 |

Interpretation:

- 这是可靠保存的“高 SR first-max”代码，可作为安全恢复点。
- 但它不是 metric-best：rerun2 的 soft-path efficiency policy 在 110/145/170/205 count 都更强。

## Why later versions got worse

后续版本大多是在 `mile_firstmax_sr110` 基础上加 guard，但 SR 曲线变差：

| run | key idea | last exact count | delta SR | delta SPL | note |
|---|---|---:|---:|---:|---|
| rerun6 | tie/path guard | 10 | +0.000000 | -0.046294 | 很早就失败 |
| rerun7 | SR focus tiepath | 15 | +0.000000 | -0.030978 | 很早就失败 |
| rerun8 | restored firstmax SR focus | 205 | +0.029268 | +0.003302 | 不如 sr110 早段，不如 rerun2 中段 |
| rerun9 | confidence minvis 0.15 | 230 | +0.013043 | +0.023650 | keep_baseline 太多，SR 被压低 |
| rerun10 | restored firstmax | 280 | +0.017857 | +0.009203 | 长跑后回落 |

经验判断：

- 强 guard / confidence guard 会减少错误替换，但也会挡掉能提升 SR 的替换，最终 SR 不够。
- `candidate_radii=[0.5,0.75]`、`camera_height=1.5`、`prefer_visible_baseline=false` 是高 SR 家族共同基础。
- `msgnav_visibility_soft_path_efficiency` 的 `path_efficiency_exponent=0.15` 是目前最值得恢复的 missing piece。

## How to restore / continue

### Restore known-good preserved code

如果当前代码被改坏，先回到保存快照：

```bash
cd /home/chenlin/krona/anchor-nav
cp saved_versions/mile_firstmax_sr110/mile.py hm3d-online/anchor_nav/mile.py
cp saved_versions/mile_firstmax_sr110/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py hm3d-online/refhm3d-nav-sequence-analyze-anchor-mile-refine1.py
cp saved_versions/mile_firstmax_sr110/mile-all-0.05-0.5.sh scripts/mile-all-0.05-0.5.sh
chmod +x scripts/mile-all-0.05-0.5.sh
```

### Reconstruct metric-best softpath policy

Start from `saved_versions/mile_firstmax_sr110` and re-add the rerun2 policy:

- CLI/config: add `--mile_path_efficiency_exponent`, default/target `0.15`.
- Run args: log `mile_path_efficiency_exponent=0.15`,
  `mile_selection_policy=msgnav_visibility_soft_path_efficiency`,
  `mile_replacement_policy=msgnav_visibility_soft_path_efficiency`.
- Selection behavior: among followable VVD candidates, prefer the candidate maximizing the recorded
  `followability_selected_efficiency_score`, with visibility still acting as the base signal.
- Effectiveness JSON from rerun2 records:
  `followability_selection_policy=msgnav_visibility_soft_path_efficiency`,
  `followability_path_efficiency_exponent=0.15`,
  `followability_max_visibility_score`,
  `followability_max_efficiency_score`,
  `followability_selected_efficiency_score`.

Do not claim exact source-level reproducibility until the Python implementation is re-created and validated, because the original rerun2 Python source is not preserved.

### Minimal validation before full rerun

Use exact same-sample comparison against the completed baseline. For a quick gate:

- Run 0.05-0.5 with `max_eval_tasks` / minimal subset at least 110 tasks if time allows.
- Compare only at exact same count.
- SR is primary; SPL is reference only.
- A useful target is:
  - count 110: SR delta near or above +0.09
  - count 145: SR delta near or above +0.06
  - count 170: SR delta near or above +0.05

If the minimal run cannot beat `saved_versions/mile_firstmax_sr110` on SR, do not start a full batch.

### Full run template

```bash
cd /home/chenlin/krona/anchor-nav
tmux new-session -d -s mile-all-0_05-0_5-softpath-exp015-next \
  'bash -lc "cd /home/chenlin/krona/anchor-nav && bash scripts/mile-all-0.05-0.5.sh detailed msgnav-firstmax-base-softpath-exp015-next; echo; echo [tmux-preserved] script exited, shell kept for log inspection; exec bash"'
```

Then monitor with exact-aligned baseline metrics only.

## Files and logs to preserve

Do not overwrite these without making a new copy:

- `saved_versions/mile_firstmax_sr110/`
- `output_logs/anchor/mile_all_0.05_0.5/20260511-021921-detailed-msgnav-firstmax-base-softpath-exp015-rerun2/`
- `output_logs/anchor/mile_all_0.05_0.5/20260510-224053-detailed-msgnav-pure-vdd-height15-rerun1/`
- `output_logs/anchor/mile_all_0.05_0.5/20260512-095738-detailed-msgnav-firstmax-restore-rerun10/`
- `output_logs/baseline_all_0.05_0.5/20260507-170755-detailed-0p5-baseline-rerun1/`

## Current running state

As of this handoff, no useful mile tmux is running. `rerun10` had been stopped because the SR-only rule failed:

- count 280
- mile SR 0.460714 vs baseline SR 0.442857
- delta SR +0.017857
- mile SPL 0.270797 vs baseline SPL 0.261594
- delta SPL +0.009203

Server cleanup also stopped an unrelated residual `run_nav.sh` / `goat-nav.py` process under `tmux-data`; no mile logs were deleted.

## Recommended next action

1. Treat `saved_versions/mile_firstmax_sr110` as the safe code base.
2. Reconstruct `msgnav_visibility_soft_path_efficiency` with `path_efficiency_exponent=0.15`.
3. Validate on exact-aligned same-sample metrics before full rerun.
4. Prefer SR over SPL for gating, per latest user instruction.

