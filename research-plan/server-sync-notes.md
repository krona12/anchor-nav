# 服务器同步说明

远端仓库：

```text
/home/chenlin/krona/anchor-nav
```

当前远端已有未跟踪文件：

```text
hm3d-online/anchor_nav/vlmdepthbox.py
hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmdepthbox.py
hm3d-online/refhm3d-nav-sequence-analyze-anchor-vlmdepthbox-refine1.py
scripts/run_vlmdepthbox_instance_0.05_0.1.sh
```

本次方案使用独立名称，不覆盖这些文件：

- `vlmevidence`
- `vlmpanobank`
- `vlmfrontier`

已同步状态：

- 本地 `research-plan/` 已同步到远端 `/home/chenlin/krona/anchor-nav/research-plan/`。
- 已实际落地到远端仓库路径并通过 `python3 -m py_compile` 的方案：
  - `vlmevidence`
  - `vlmpanobank`
- `vlmfrontier` 当前保留在远端 `research-plan/code/vlmfrontier/`，作为高风险 frontier 轨迹改动草案；建议先读方案再决定是否安装到 `hm3d-online/`。

建议落地优先级：

1. `vlmevidence`：最接近当前 `vlmtop5`，风险最低。
2. `vlmpanobank`：作为 snapshot-only ablation。
3. `vlmfrontier`：改变探索轨迹，先跑极小样本。
