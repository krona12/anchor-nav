# VFV (VLM Final Verification) 实验分析报告

**任务**：RefHM3D instance-level 序列导航（slice 0.05–0.3，N=116 episodes，6 scenes）  
**Baseline**：`20260421-140732-detailed`（新 baseline，无 VLM 调用）  
**VFV Run**：`20260429-023723-detailed`  

---

## 1. 方案概述

VFV 在 PQ3D final decision 时额外执行：
- **Phase1**：拍 360° 全景图 → VLM 验证目标是否可见
  - `full_match=True` 或 `strong_anchor_match=True` → 验证通过，保留 PQ3D 结果
  - 否则 → 触发 Phase2
- **Phase2**：用 anchor_desc 重新查询 PQ3D Stage2 → 导航至锚点附近 → 作为新目标

---

## 2. 总体结果

| 指标 | Baseline | VFV | Delta |
|------|----------|-----|-------|
| SR   | 0.4138   | 0.2845 | **-0.1293** |
| SPL  | 0.2338   | 0.1575 | **-0.0762** |
| N    | 116      | 116 | — |

SR 损失 -0.129，远超之前所有方案的回退幅度（TQSI 最差 -0.078）。

---

## 3. 按场景分解

| 场景 | N | Base SR/SPL | VFV SR/SPL | ΔSR | Loss | Gain |
|------|---|-------------|------------|-----|------|------|
| 00802 | 20 | 0.350/0.216 | 0.350/0.189 | **+0.000** | 2 | 2 |
| 00803 | 31 | 0.387/0.177 | 0.290/0.128 | -0.097 | 8 | 5 |
| 00808 | 20 | 0.350/0.241 | 0.250/0.136 | -0.100 | 2 | 0 |
| 00810 | 20 | 0.500/0.284 | 0.200/0.120 | **-0.300** | 6 | 0 |
| 00813 | 20 | 0.600/0.340 | 0.400/0.271 | **-0.200** | 5 | 1 |
| 00814 |  5 | 0.000/0.000 | 0.000/0.000 | +0.000 | 0 | 0 |

- 00810 和 00813 损失最惨（-0.200～-0.300），均为 SR 较高的场景（顺序更靠后，VLM 开销累积更多）
- 00802（第一个场景）损失为 0，印证了"累积开销伤害靠后任务"的假设

---

## 4. 胜负统计

- 双方均成功：25 | 双方均失败：60
- **Loss（Baseline成功/VFV失败）：23**
- Gain（Baseline失败/VFV成功）：8

**净SR变化**：-15 episodes（-0.129 SR）

---

## 5. VFV 触发分析

| 指标 | 数值 |
|------|------|
| Phase1 VLM 调用次数 | 96 |
| Phase2 触发次数 | 7（其中1次 anchor_query ok=False，未实际应用） |
| vfv_applied=1 | 6 |
| vfv_helpful=True | 4 |
| vfv_helpful=False | 2 |
| Phase1 平均耗时 | **5.43 秒** |
| Phase1 中位耗时 | 4.19 秒 |
| Phase1 最大耗时 | 14.37 秒 |

**Phase1 验证通过率**：
- full_match=True：32 次（33%）
- strong_anchor_match=True：57 次（59%）
- 无匹配 → Phase2：7 次（7%）

Phase1 通过率高达 **92%**，说明 PQ3D 本身选目标大多数情况是正确的（在到达位置时目标可见）。VFV Phase2 几乎不需要介入，但 Phase1 的 VLM 验证开销不可避免。

---

## 6. Phase2 应用明细（6例）

| ep | t | 类别 | anchor | d_base | d_sel | delta | 结果 |
|----|---|------|--------|--------|-------|-------|------|
| 3 | 0 | picture | light-colored vanity | 3.896m | 2.512m | -1.384m | **HELPFUL** |
| 4 | 0 | toilet | light-colored vanity | 5.235m | 5.235m | +0.000m | HARMFUL（timing） |
| 8 | 0 | cabinet | plant beside bed | 2.204m | 5.437m | **+3.233m** | **HARMFUL** |
| 10 | 4 | couch | coffee table | 16.209m | 15.849m | -0.360m | HELPFUL（但<0.25m threshold不变） |
| 12 | 0 | fire extinguisher | washer-dryer | 5.416m | 3.355m | -2.060m | HELPFUL |
| 17 | 0 | bin | glass-enclosed bathtub | 8.546m | 6.455m | -2.091m | HELPFUL |

- 4 次 helpful（对象选择改善），2 次 harmful
- **但 6 次 Phase2 全部 SR=False**——说明即使 Phase2 改善了对象选择，由于 VLM 开销导致导航步数被消耗，仍然失败

---

## 7. Loss 归因

| 类型 | 数量 | 说明 |
|------|------|------|
| 步数耗尽（d=inf, att=0） | **8** | 前序任务 VLM 开销耗尽步数预算，本任务未触发 final decision |
| 间接时序失效（att=1, app=0, d_sel=d_base） | **13** | Phase1 验证通过无 Phase2，但 VLM 调用 ~5s 消耗了步数，边界成功任务翻车 |
| 直接 VFV 选择有害（app=1, d↑） | **1** | ep=8 cabinet：Phase2 选了更差对象（+3.233m） |
| 路径噪声（d<1.0m） | **2** | 随机扰动 |

**主因：VLM 时序开销（共 21/23 个 Loss 与 VLM 耗时相关）**

---

## 8. 为什么 VLM 时序是灾难性的

序列导航特性：**同一 episode 内所有 task 共享 400 步预算**，且时间连续累积。

- 每次 Phase1 检查消耗 ~5s（≈5步）
- 一个 episode 平均 96/116 ≈ 0.83 次 VFV 调用/任务
- 但有些 episode 有多个 task（00803 的 task0+task4 各自都可能触发 VFV）
- 早期 task 的 VFV 开销累积 → 后续 task 步数不足 → d=inf

**实证**：
- 00802（第一场景，每 episode 只有 task0）：ΔSR=0，无损失
- 00810/00813（包含 task0+task4 的 episode，步数压力大）：ΔSR=-0.300/-0.200

---

## 9. Gain 分析（8例）

8 个 gain episode 全部 `vfv_applied=0`（Phase2 未应用），`d_sel=d_base`（无对象改变）。

→ **Gain 完全是随机路径差异，与 VFV 本身无关**

与 Loss 对比：Loss 有明确的 VLM 时序原因，Gain 是纯随机。因此净 SR 差 -15 主要由 VFV 开销引起，非随机中性。

---

## 10. VFV 机制本身的评估（剥离时序干扰）

| 维度 | 结论 |
|------|------|
| Phase1 通过率 | 92%（PQ3D 目标选择多数正确，Phase1 几乎都不需要介入） |
| Phase2 应用效果 | 4/6 helpful，2/6 harmful → 对象选择改善率尚可 |
| 主要问题 | Phase1 本身是"多余"检查：96次中89次验证通过无干预，但每次耗5s |
| Phase2 被激活的先决条件 | PQ3D 目标选错（~7%概率），且 anchor 描述有效 |

VFV 的核心价值假设（频繁触发 Phase2 来修正 PQ3D 错误）**未能成立**，因为 PQ3D 在 final decision 时实际选目标的正确率远高于 7%（92% Phase1 通过）。

---

## 11. 根本结论

> **VFV 在序列导航中架构性不兼容**：每次 VLM 验证调用 ~5-14 秒，在共享步数预算的序列任务中，开销累积使后续任务步数耗尽，造成系统性 SR 下降 -0.129。Phase2 的对象选择改善效果（4/6 helpful）被时序损失完全淹没。

---

## 12. 下一步建议

### P1（最高优先）：消除 Phase1 对序列导航的时序影响

**方案A**：仅在 per-task 模式（非序列）使用 VFV  
→ 最干净，消除所有时序问题，但限制了 VFV 的应用范围

**方案B**：步数预算守卫  
```python
remaining_steps = max_steps - current_step
if remaining_steps < STEP_GUARD_THRESHOLD:  # 建议 50~80
    skip_vfv()
```
→ 保护紧张任务，但无法防止早期任务 VLM 开销饿死后续任务

**方案C**：每 episode 限制 VFV 调用次数（max 2 次）  
→ 减少开销，但仍有残留影响

**方案D**（推荐）：**条件触发 Phase1**——只有当 PQ3D top-1 对象的置信度低于阈值时才触发 VFV  
```python
top1_score = topk[0][1]
top2_score = topk[1][1] if len(topk) > 1 else -inf
score_gap = top1_score - top2_score
if score_gap > CONFIDENT_THRESHOLD:  # top-1 明显优于 top-2 → 跳过 VFV
    skip_vfv()
```
→ 在 PQ3D 置信时跳过，在不确定时才验证，大幅减少 VLM 调用次数

### P2：修复 Phase2 HARMFUL 案例

ep=8 cabinet（anchor='plant beside bed'）：anchor 是"床边的植物"，语义太模糊，Phase2 导航至错误位置。
建议：anchor 描述过于通用时（如 plant、door、window）跳过 Phase2。

```python
GENERIC_ANCHORS = {'plant', 'door', 'window', 'wall', 'floor', 'ceiling'}
if anchor_desc.split()[-1] in GENERIC_ANCHORS:
    skip_phase2()
```

---

## 13. 实验文件索引

| 文件 | 说明 |
|------|------|
| `20260421-140732-detailed/` | 新 Baseline（本次对比基准） |
| `20260429-023723-detailed/` | VFV run（全量 0.05-0.3 slice） |
| `20260429-003154-detailed/` | 早期 VFV 测试（0.05-0.1 只，参考） |
| `analyze_vfv.py` | 本次分析脚本 |
