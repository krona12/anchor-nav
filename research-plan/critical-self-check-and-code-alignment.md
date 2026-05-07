# 批判性自我检查与代码对齐

## 1. 已确认接口

- `PQ3DModel.decision()` 支持 `analysis_output_dir`，会写 `stage2_decision.json`。
- `last_decision_aux` 至少包含：
  - `goto_frontier_probability`
  - `is_object_decision`
  - `real_object_decision_idx`
  - `n_real_objects`
  - `object_top1_top2_logit_gap`
- `stage2_decision.json.object_candidates` 包含 top-k rerank 所需的 slot、logit、center。
- `stage2_decision.json.frontier_candidates` 包含 frontier index、logit、center。
- `RepresentationManager.object_first_rgb` 可取 object first RGB。
- `panorama_frames_by_slot` 可以用当前 `color_list[-12:]` + `reversed` 绑定 slot 的 360 snapshot。
- VLM 调用应走 `hm3d-online/vlm/client.py::chat_messages`，现有 `vlmtop5` 已遵守这一点。

## 2. 主要风险

- **VLM 小物体漏检**：panorama 分辨率被压缩后，小目标可能不可见。VLMEvidence 仍保留 first RGB，VLMPanoBank 用作 ablation，不应单独替代主方法。
- **VLM 误判上下文**：看到“厨房”不等于看到“目标物体”。Prompt 与 gate 中已要求 exact/generic/insufficient 区分，但仍需看日志人工抽查。
- **top-k 不含真目标**：最终 rerank 无法补救，需要 VLMFrontier 或更好的探索策略补 memory bank。
- **stage2 强置信但 VLM 反向选择**：不能硬切；已引入 `evidence_delta` 和 strict advantage gate，但后续可加入 logit gap adaptive threshold。
- **frontier bearing 对齐误差**：VLMFrontier 使用 agent forward + panorama view index 映射，可能受 Habitat 坐标/环视展开方向影响；必须先用可视化日志人工核对。

## 3. 已落地/未落地边界

已落地到远端实际 repo 路径并通过语法检查：

- `vlmevidence`
- `vlmpanobank`

只作为方案包同步到远端 `research-plan/code/`，未安装到实际 repo：

- `vlmfrontier`

原因：frontier 会改变探索轨迹，且 bearing-view 对齐需要人工核对，直接加入批量测试风险较高。

## 4. 对用户当前 `vlmtop5` 的直接建议

如果你现在正在跑 `refhm3d-nav-sequence-analyze-anchor-vlmtop5.py`，建议先不要中断。跑完后对比：

- `vlmtop5`：first RGB top-k + selected panorama veto。
- `vlmevidence`：first RGB top-k + evidence delta + selected panorama veto。
- `vlmpanobank`：top-k panorama snapshot 直接比较。

这三者形成一个很干净的 ablation：

```text
VLM sees local object only
VLM sees local object with evidence calibration
VLM sees co-visible scene snapshot
```

如果 `vlmpanobank` 在 01 case 上更强，说明细粒度 instance 主要依赖上下文；如果 `vlmevidence` 更稳，说明 first RGB 仍有局部辨别优势，但需要 conservative calibration。

## 5. 需要进一步问询/人工确认

- `panorama_frames_by_slot` 是否确实对应“该 object 首次进入 memory 的同一决策步”，而不是 merge 后多个 slot 共用同一 scan 造成歧义？
- 当前 VLM API 的图像输入大小/数量限制是多少？VLMPanoBank 一次送 5 张 panorama，可能比 top5 first RGB 显著更慢。
- `object_top1_top2_logit_gap` 的典型分布是什么？需要统计后设置 adaptive evidence threshold。
- `vlmdepthbox` 是否已经在测试 last-mile bbox/depth projection？如果它有效，可以作为 ECSV 后的最后一步，而不是和 rerank 主方法混在一起。
- 是否只评 instance-level？如果 object-level 也跑，VLM 的 exact-instance gate 要放宽，否则会过度保守。

## 6. 下一步实验顺序

1. 跑 `vlmevidence` 单样本，检查 VLM JSON、候选图、selected panorama、10/01/11/00。
2. 跑 `vlmpanobank` 同一单样本，确认 panorama 顺序、VLM 是否能看清目标。
3. 若两者日志正常，再跑 `[0.05, 0.1]` instance slice。
4. 统计 correction_applied、helpful/harmful、VLM 耗时。
5. 只在确认 object-side ablation 有价值后，再安装 `vlmfrontier` 到 actual repo 路径做极小样本。

