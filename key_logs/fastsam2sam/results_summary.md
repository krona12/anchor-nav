# FastSAM vs SAM 对比实验汇总

**数据集**: HM3D, range 0–0.2, 700 sequences (7 scenes × 100 sequences)  
**场景顺序**: 00800 → 00802 → 00803 → 00808 → 00810 → 00813 → 00814  
**Task levels**: instance / object / region / room  
**FastSAM**: FastSAM 后端  
**SAM**: segment_anything vit_h (`sam_vit_h_4b8939.pth`, points_per_batch=64)

---

## 最终完整对比（全部 N=700）

### 总览

| 模块 | FastSAM SR | SAM SR | ΔSR | FastSAM SPL | SAM SPL | ΔSPL |
|------|-----------|--------|-----|------------|---------|------|
| Baseline | 0.4200 | 0.4400 | **+0.020** | 0.2530 | 0.2753 | +0.022 |
| MQSC-R1 | 0.4457 | 0.4757 | **+0.030** | 0.2742 | 0.2962 | +0.022 |
| VISTALS | 0.4343 | 0.4314 | **−0.003** | 0.2813 | 0.2832 | +0.002 |
| VISTA2MQSC | 0.4457 | 0.4600 | **+0.014** | 0.2864 | 0.3016 | +0.015 |

### Baseline（N=700 完整）

| Metric | FastSAM | SAM (rerun) | Δ |
|--------|---------|------------|---|
| SR | 0.4200 | 0.4400 | +0.020 |
| SPL | 0.2530 | 0.2753 | +0.022 |
| instance SR (n=151) | 0.278 | 0.318 | +0.040 |
| object SR (n=175) | 0.526 | 0.520 | −0.006 |
| region SR (n=202) | 0.381 | 0.396 | +0.015 |
| room SR (n=172) | 0.483 | 0.517 | +0.034 |

SAM log: `sam_complete/baseline_sam_rerun_n700.log`  
FastSAM log: `fastsam/baseline_fastsam_n700.log`

### MQSC-R1（N=700 完整）

| Metric | FastSAM | SAM (original) | Δ |
|--------|---------|---------------|---|
| SR | 0.4457 | 0.4757 | +0.030 |
| SPL | 0.2742 | 0.2962 | +0.022 |
| instance SR (n=151) | 0.331 | 0.371 | +0.040 |
| object SR (n=175) | 0.520 | 0.549 | +0.029 |
| region SR (n=202) | 0.441 | 0.455 | +0.014 |
| room SR (n=172) | 0.477 | 0.517 | +0.040 |

SAM log: `sam_complete/mqsc_r1_sam_n700.log`  
FastSAM log: `fastsam/mqsc_r1_fastsam_n700.log`

### VISTALS（N=700 完整）

| Metric | FastSAM | SAM (original) | Δ |
|--------|---------|---------------|---|
| SR | 0.4343 | 0.4314 | −0.003 |
| SPL | 0.2813 | 0.2832 | +0.002 |
| instance SR (n=151) | 0.272 | 0.285 | +0.013 |
| object SR (n=175) | 0.554 | 0.554 | ±0.000 |
| region SR (n=202) | 0.416 | 0.411 | −0.005 |
| room SR (n=172) | 0.477 | 0.459 | −0.018 |

SAM log: `sam_complete/vistals_sam_n700.log`  
FastSAM log: `fastsam/vistals_fastsam_n700.log`

### VISTA2MQSC（N=700 完整，SAM new threshold）

| Metric | FastSAM | SAM (new threshold) | Δ |
|--------|---------|---------------------|---|
| SR | 0.4457 | 0.4600 | +0.014 |
| SPL | 0.2864 | 0.3016 | +0.015 |
| instance SR (n=151) | 0.311 | 0.377 | +0.066 |
| object SR (n=175) | 0.554 | 0.526 | −0.028 |
| region SR (n=202) | 0.406 | 0.416 | +0.010 |
| room SR (n=172) | 0.500 | 0.517 | +0.017 |

SAM log: `sam_complete/vista2mqsc_sam_new_threshold_n700.log`  
FastSAM log: `fastsam/vista2mqsc_fastsam_n700.log`

> 注：VISTA2MQSC 使用 SAM new threshold 配置；原始阈值版本因OOM在N=184崩溃。

---

## 关键结论

1. **SAM 整体优于 FastSAM**：3/4 模块 ΔSR 正向（+0.014～+0.030）
2. **MQSC-R1 + SAM 提升最显著**：+0.030 SR，各 level 全部正向
3. **VISTALS 对分割后端不敏感**：ΔSR = −0.003，近似持平
4. **VISTA2MQSC object level 异常**：SAM 的 object SR 比 FastSAM 低 −0.028，但 instance 大幅提升 +0.066

---

## 实验日志索引

### fastsam/（FastSAM，完整 N=700）

| 文件 | 模块 | 最终 SR | 最终 SPL |
|------|------|--------|---------|
| baseline_fastsam_n700.log | Baseline | 0.4200 | 0.2530 |
| mqsc_r1_fastsam_n700.log | MQSC-R1 | 0.4457 | 0.2742 |
| vistals_fastsam_n700.log | VISTALS | 0.4343 | 0.2813 |
| vista2mqsc_fastsam_n700.log | VISTA2MQSC | 0.4457 | 0.2864 |

### sam_complete/（SAM，完整 N=700）

| 文件 | 模块 | 配置 | 最终 SR | 最终 SPL |
|------|------|------|--------|---------|
| baseline_sam_rerun_n700.log | Baseline | 原始阈值，cuda3 rerun | 0.4400 | 0.2753 |
| mqsc_r1_sam_n700.log | MQSC-R1 | 原始阈值，cuda2 | 0.4757 | 0.2962 |
| vistals_sam_n700.log | VISTALS | 原始阈值，cuda2 | 0.4314 | 0.2832 |
| vista2mqsc_sam_new_threshold_n700.log | VISTA2MQSC | new threshold | 0.4600 | 0.3016 |

### sam_crashed/（SAM，崩溃/未完成）

| 文件 | 模块 | 崩溃原因 | 崩溃时 N |
|------|------|---------|---------|
| baseline_sam_original_oom_n100.log | Baseline | OOM，cuda3 GPU占用过高 | 100 |
| baseline_sam_new_threshold_runtime_error_n314.log | Baseline | RuntimeError: decision failed (scene=00808 ep2 task4) | 314 |
| vista2mqsc_sam_original_oom_n184.log | VISTA2MQSC | OOM，cuda2 四进程竞争 | 184 |
| vista2mqsc_sam_loose_oom_n159.log | VISTA2MQSC | OOM，sam_level_preset=loose | 159 |
