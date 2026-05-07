# 方案 A：VLMEvidence

方法名：Evidence-Calibrated Snapshot Verification。

核心：在 MTU3D 最终 object decision 后，对 `stage2_decision.json` 的 top-5 object candidates 调用 VLM。VLM 不只输出 best index，还输出 `evidence_score_image1/evidence_score_best`、decisive constraints、missing constraints。只有当非 top1 的 evidence delta 足够大、且有严格优势约束、且 panorama verify 通过时，才替换目标坐标。

服务器目标路径：

- `hm3d-online/anchor_nav/vlmevidence.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence-refine1.py`
- `scripts/run_vlmevidence_instance_0.05_0.1.sh`

验证：

```bash
python -m py_compile hm3d-online/anchor_nav/vlmevidence.py \
  hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence.py \
  hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence-refine1.py
```

最小运行：

```bash
python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmevidence.py \
  --scene_name 00802-wcojb4TFT35 \
  --episode_id 17 \
  --task_id 0 \
  --num_tasks 1 \
  --output_root ./output_process
```

批量运行：

```bash
bash scripts/run_vlmevidence_instance_0.05_0.1.sh detailed
```

