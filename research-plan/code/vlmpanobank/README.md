# 方案 B：VLMPanoBank

方法名：Co-visibility Snapshot Bank。

核心：把当前已有的 `panorama_frames_by_slot` 显式当成 snapshot memory。最终 object decision 时，对 top-5 候选分别拼接该 slot 被注册时的 360 panorama，让 VLM 比较候选的共视上下文，而不是只看 first RGB。

区别于 VLMEvidence：

- VLMEvidence 的 VLM 输入是 top-k first RGB，再对 selected 做 panorama verify。
- VLMPanoBank 的 VLM 输入直接是 top-k panorama snapshots，目标是验证“共视上下文是否比局部 first RGB 更适合细粒度 instance 匹配”。

服务器目标路径：

- `hm3d-online/anchor_nav/vlmpanobank.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmpanobank.py`
- `hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmpanobank-refine1.py`
- `scripts/run_vlmpanobank_instance_0.05_0.1.sh`

验证：

```bash
python -m py_compile hm3d-online/anchor_nav/vlmpanobank.py \
  hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmpanobank.py \
  hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmpanobank-refine1.py
```

最小运行：

```bash
python3 hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmpanobank.py \
  --scene_name 00802-wcojb4TFT35 \
  --episode_id 17 \
  --task_id 0 \
  --num_tasks 1 \
  --output_root ./output_process
```

批量运行：

```bash
bash scripts/run_vlmpanobank_instance_0.05_0.1.sh detailed
```

