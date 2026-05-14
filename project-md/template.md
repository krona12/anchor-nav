# RefHM3D 模块开发与脚本模板约束

更新时间：2026-05-14

## 目标

本文档用于规范后续新模块开发、单场景测试脚本、批量 refine1 脚本、以及 shell launcher 的写法。

核心原则：

1. 新模块只接入模板预留的 hook，不破坏 PQ3D、frontier、follow、metric 的主流程。
2. 单场景调试先用 `hm3d-online/template-minimal-test.py` 的结构。
3. 批量实验先用 `hm3d-online/refhm3d-nav-sequence-analyze-template-xxx-refine1.py` 的结构。
4. launcher 对齐 `scripts/baseline-all-0.05-0.1.sh` 和 `scripts/run_vfv_instance_0.05_0.1.sh`：固定 run tag、输出目录、run args、脚本快照、run command。
5. 所有实验必须可 resume、可追溯、可聚合 metrics。
6. 开 tmux 长跑时必须显式指定当前使用的 CUDA，例如 `CUDA_VISIBLE_DEVICES=2`，并在启动输出与 `run_args.txt` 中记录。

## 参考文件

### 单场景最小模板

`hm3d-online/template-minimal-test.py`

用途：

- 单个 scene / episode / task range 调试。
- 保留 PQ3D + frontier + navigation + SR/SPL。
- 剔除具体模块逻辑。
- 适合验证模块 hook 的输入输出、日志、目标点是否合理。

### 批量 refine1 模板

`hm3d-online/refhm3d-nav-sequence-analyze-template-xxx-refine1.py`

用途：

- 按 `--start_ratio` / `--end_ratio` 批量跑 scene shard。
- 支持已有 output JSON resume。
- 默认对齐 `vfv-refine1` 的 instance-only 行为，也可通过 `--task_levels object,room,region,instance` 跑全层级。
- 预留 `template_refine_hook(...)` 作为模块插入点。

### launcher 模板

已有参考：

- `scripts/baseline-all-0.05-0.1.sh`
- `scripts/run_vfv_instance_0.05_0.1.sh`
- `scripts/run_template_instance_0.05_0.1.sh`
- `scripts/run_template_all_0.05_0.1.sh`

launcher 必须负责：

- conda 环境激活。
- `PYTHONPATH` / Habitat / YOLO 静默环境变量。
- CUDA 选择必须明确打印并写入 `run_args.txt`；不要让长跑依赖用户猜测当前 GPU。
- `RUN_TAG` / `OUT_DIR` 创建。
- 写 `run_args.txt`。
- 保存当前脚本快照。
- 写 `run_command.sh`。
- 删除空 JSON 后继续 resume。

## 标准模块开发流程

### 1. 确定模块类型

先明确模块属于哪类：

- `target-refine`：只在 PQ3D final decision 后改写目标点。
- `decision-rerank`：基于 PQ3D top-k 或 representation 重新选择对象。
- `visibility-verify`：需要 VLM / panorama / image evidence 验证。
- `navigation-repair`：对不可达点做 navmesh 修复。
- `logging-only`：不改变目标点，只记录诊断信息。

默认从 `target-refine` 的最小 hook 开始。不要一开始就改 PQ3D 主循环。

### 2. 新模块文件位置

模块代码优先放在：

`hm3d-online/anchor_nav/<module_name>.py`

要求：

- 模块函数只接收明确输入，不从全局读取运行状态。
- 输出必须 JSON 可序列化，或由调用侧用 `_jsonable` 转换。
- 不直接写主结果 JSON，最多写自己的 debug 文件到传入的 `output_dir`。
- 不直接改 `result_dict` / `effectiveness_dict`。
- 不吞异常：可返回 `{"ok": false, "error_type": ..., "error_message": ...}`，但不要静默失败。

### 3. Hook 约定

批量模板当前 hook：

```python
def template_refine_hook(
    *,
    sentence: str,
    task_type: str,
    scene_name: str,
    episode_id: int,
    task_id: int,
    decision_num: int,
    is_final: bool,
    pq3d_model: Any,
    target_position: np.ndarray,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
```

返回：

```python
new_target_xyz, module_info
```

`module_info` 必须至少包含：

```python
{
  "ok": true,
  "module": "<module_name>",
  "applied": false,
  "reason": "...",
  "target_before": [...],
  "target_after": [...]
}
```

若模块真正改变了目标点：

- `applied=true`
- `target_before` 是原始 PQ3D final target
- `target_after` 是模块选择后的 target
- 记录选择依据，如候选数量、分数、rank、阈值、失败原因

### 4. 插入点约束

默认只在 `is_final == True` 时调用模块。

原因：

- 与 `vfv-refine1` / `mile-refine1` 的 refine 类实验对齐。
- 避免模块影响 frontier exploration 阶段，降低变量数量。
- 指标更容易和 baseline 对齐。

如果模块需要作用于 non-final decision，必须显式写入：

- CLI 参数，例如 `--xxx_apply_non_final_decisions`
- `run_args.txt`
- `module_info`
- final summary

不要把 non-final 行为偷偷写死。

## 批量脚本结构要求

批量 Python 脚本建议从：

`hm3d-online/refhm3d-nav-sequence-analyze-template-xxx-refine1.py`

复制并改名：

`hm3d-online/refhm3d-nav-sequence-analyze-anchor-<module>-refine1.py`

必须保留的 CLI：

```text
--start_ratio
--end_ratio
--concise_description
--navigation_data_path
--hm3d_data_base_path
--pq3d_stage1_path
--pq3d_stage2_path
--output_log_dir
--task_levels
--max_steps
--decision_num_min
--success_distance
--seed
--quiet_nav_steps
--decision_log_interval
```

若模块需要额外参数，统一使用模块名前缀：

```text
--xxx_threshold
--xxx_top_k
--xxx_enable_repair
--xxx_apply_task_levels
```

不要使用模糊参数名如 `--threshold`、`--top_k`，避免和其他模块冲突。

### 输出文件约定

批量脚本输出目录由 launcher 传入：

`--output_log_dir "${OUT_DIR}"`

目录内必须包含：

- `run_args.json`
- `refhm3d_seq_<module>_refine1_<start>_<end>.json`
- `refhm3d_seq_<module>_refine1_effectiveness_<start>_<end>.json`
- `<module>_live_metrics_<start>_<end>.log`
- `process/scene=<scene>/episode=<episode>/task=<task>/dec_*.json`
- `process/scene=<scene>/episode=<episode>/task=<task>/summary.json`
- `process/scene=<scene>/episode=<episode>/task=<task>/effectiveness.json`
- stdout/stderr tee log：`refhm3d-nav-sequence-analyze-<module>-refine1-*.log`

### Metrics 约定

每条 result row 必须包含：

```text
scene_name
episode_id
task_id
task_level
navigation_type
sr
spl
object_category
task_time_sec
steps_total
end_reason
start_goal_geo
end_goal_geo
episode_cum_distance
module_name
module_hook_called
module_hook_applied
```

live metrics 必须输出：

- overall count / SR / SPL
- `object` / `room` / `region` / `instance` 四层级 count / SR / SPL

即使某个 level 暂时没有样本，也要显示 count=0。

### Resume 约定

resume key 默认：

```text
scene_name + navigation_type + episode_id
```

含义：某 episode 一旦完成并写入结果，重跑时跳过整个 episode。

不要用 task-level resume，除非明确实现了 task 内状态恢复；否则容易造成同一 episode 的导航状态不连续。

## Launcher 编写要求

### instance launcher

命名：

`scripts/run_<module>_instance_0.05_0.1.sh`

输出目录：

`output_logs/anchor/<module>_instance_0.05_0.1/${RUN_TAG}`

必须传：

```bash
--task_levels "instance"
```

### all launcher

命名：

`scripts/run_<module>_all_0.05_0.1.sh`

输出目录：

`output_logs/anchor/<module>_all_0.05_0.1/${RUN_TAG}`

必须传：

```bash
--task_levels "object,room,region,instance"
```

### launcher 共同约束

必须支持：

```bash
bash scripts/run_<module>_instance_0.05_0.1.sh [detailed|concise] [optional_tag]
bash scripts/run_<module>_all_0.05_0.1.sh [detailed|concise] [optional_tag]
```

必须写入 `run_args.txt`：

```text
timestamp
run_tag
desc_mode
user_tag
script
command
python
module
task_levels
slice_range
slice_step
num_shards_total
schedule
cuda_visible_devices
seed
模块专属参数
```

必须保存：

```bash
cp "$0" "${OUT_DIR}/<script_name>.snapshot"
```

必须写：

```bash
run_command.sh
```

必须删除空 JSON：

```bash
if [ -f "${OUT_JSON}" ] && [ ! -s "${OUT_JSON}" ]; then
  rm -f "${OUT_JSON}"
fi
```

如果有 effectiveness JSON，也要同样处理。

### tmux 长跑启动约束

正式长跑开 tmux 时，必须显式指定 CUDA。

硬性要求：

- tmux 命令里必须出现 `CUDA_VISIBLE_DEVICES=<gpu_id>`，例如第三张卡写 `CUDA_VISIBLE_DEVICES=2`。
- tmux session 名建议带上 cuda 信息，例如 `xxx-cuda2`。
- launcher 启动后必须在 pane 里打印 `CUDA_VISIBLE_DEVICES: <gpu_id>`。
- `run_args.txt` 必须写 `cuda_visible_devices=<gpu_id>`。
- 不要只依赖脚本里的默认值（如 `${CUDA_VISIBLE_DEVICES:-0}`）来开长跑；除非用户明确要求把 CUDA 写死进脚本，否则 tmux 启动命令也要显式传入。

正式长跑需要开 tmux 时，默认不要使用 shell 输出重定向。

原因：

- 用户需要 `tmux attach` 后直接看到实时导航日志。
- 如果用 `> xxx.log 2>&1`，tmux pane 会变成黑屏，难以及时检查进度、错误和 live metrics。
- 批量 Python 脚本本身已经写 stdout/stderr tee log、live metrics log、process/decision JSON、summary JSON；shell 层不需要再把整个 tmux 输出吞掉。

推荐写法：

```bash
tmux new-session -d -s <session_name> \
  "cd /home/chenlin/krona/anchor-nav && CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/run_<module>_all_<slice>.sh detailed <tag>"
```

实际示例：

```bash
tmux new-session -d -s mqsc-r1-all-0_0-0_2-cuda2 \
  "cd /home/chenlin/krona/anchor-nav && CUDA_VISIBLE_DEVICES=2 bash scripts/run_mqsc_r1_all_0.0_0.2.sh detailed"
```

启动后必须立刻确认：

```bash
tmux capture-pane -pt <session_name> -S -80
```

确认输出里有：

```text
CUDA_VISIBLE_DEVICES: <gpu_id>
```

禁止写法：

```bash
tmux new-session -d -s <session_name> \
  "cd /home/chenlin/krona/anchor-nav && CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/run_<module>_all_<slice>.sh detailed <tag> > output_logs/.../xxx_tmux.log 2>&1"
```

同样禁止没有 CUDA 的写法：

```bash
tmux new-session -d -s <session_name> \
  "cd /home/chenlin/krona/anchor-nav && bash scripts/run_<module>_all_<slice>.sh detailed <tag>"
```

如果确实需要额外保存 shell 层完整输出，应优先使用不遮挡 tmux pane 的方式，例如：

```bash
tmux new-session -d -s <session_name> \
  "cd /home/chenlin/krona/anchor-nav && CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/run_<module>_all_<slice>.sh detailed <tag> | tee -a output_logs/.../xxx_tmux.log"
```

但默认仍以“tmux pane 可直接看实时日志”为准。

## Prompt 约束：让 Codex 写新模块

推荐 prompt：

```text
请基于 project-md/template.md 和 hm3d-online/refhm3d-nav-sequence-analyze-template-xxx-refine1.py
开发一个新模块 <module_name>。

约束：
1. 模块代码放到 hm3d-online/anchor_nav/<module_name>.py。
2. 不改 PQ3D 主流程，不改 baseline 脚本。
3. 只在 refine1 final decision hook 接入，除非我明确要求 non-final。
4. hook 输入输出对齐 template_refine_hook，返回 (target_xyz, module_info)。
5. module_info 必须 JSON 可序列化，包含 ok/module/applied/reason/target_before/target_after。
6. 批量脚本从 hm3d-online/refhm3d-nav-sequence-analyze-template-xxx-refine1.py 派生，
   命名为 hm3d-online/refhm3d-nav-sequence-analyze-anchor-<module_name>-refine1.py。
7. 保留 start_ratio/end_ratio、resume、process/、effectiveness、live metrics、四层级 by-level metrics。
8. shell launcher 写 instance 和 all 两个：
   scripts/run_<module_name>_instance_0.05_0.1.sh
   scripts/run_<module_name>_all_0.05_0.1.sh
9. launcher 对齐 scripts/run_template_instance_0.05_0.1.sh 和 scripts/run_template_all_0.05_0.1.sh。
10. 完成后运行 py_compile 和 bash -n，不要启动长跑。
```

如果模块使用 VLM，补充：

```text
模块使用 VLM。请：
1. 所有 VLM 参数用 <module_name>_ 前缀，如 --<module_name>_vlm_model。
2. launcher 支持 VLM_MODEL 环境变量，但 run_args.txt 必须记录实际模型。
3. VLM raw response、parse_ok、parse_attempts、image_path 必须写入 module_info。
4. JSON parse 失败不能静默，必须记录 error_type/error_message，并 fallback 到 baseline target。
5. 图片输出只能写到 task_dir 下的模块子目录。
```

如果模块会改目标点，补充：

```text
模块会改变 final target。请：
1. 记录 baseline_target_position 和 selected_target_position。
2. 记录 baseline_target_to_goal_l2 和 selected_target_to_goal_l2。
3. 记录 module_helpful：仅当 module_hook_applied=true 且两者距离有限时计算。
4. follower_error 必须进入分母，SR/SPL 记 0。
```

## Prompt 约束：让 Codex 写 launcher

推荐 prompt：

```text
请根据 scripts/run_template_instance_0.05_0.1.sh 和 scripts/run_template_all_0.05_0.1.sh
为模块 <module_name> 写两个 launcher：

1. scripts/run_<module_name>_instance_0.05_0.1.sh
2. scripts/run_<module_name>_all_0.05_0.1.sh

约束：
- 保留 conda activate mtu3d 和 PYTHONPATH。
- 支持 [detailed|concise] [optional_tag]。
- instance 输出到 output_logs/anchor/<module_name>_instance_0.05_0.1/${RUN_TAG}。
- all 输出到 output_logs/anchor/<module_name>_all_0.05_0.1/${RUN_TAG}。
- instance 传 --task_levels instance。
- all 传 --task_levels object,room,region,instance。
- launcher 必须 echo 当前 `CUDA_VISIBLE_DEVICES`，并在 `run_args.txt` 写 `cuda_visible_devices=${CUDA_VISIBLE_DEVICES}`。
- run_args.txt 必须记录 module、python、task_levels、slice_range、cuda_visible_devices、seed 和模块参数。
- 保存脚本 snapshot 和 run_command.sh。
- 删除空 output/effectiveness JSON。
- 完成后 chmod +x，并运行 bash -n。
- 若需要开 tmux 长跑，tmux 命令必须显式包含 `CUDA_VISIBLE_DEVICES=<gpu_id>`；不要使用 `> xxx.log 2>&1` 输出重定向；必须保证 attach 后 pane 内能直接看到实时日志和 CUDA 信息。
```

## Prompt 约束：让 Codex 改已有模块

推荐 prompt：

```text
请修改模块 <module_name>，但必须遵守 project-md/template.md：

1. 不改 result_dict schema，除非先说明新增字段。
2. 不改 launcher 输出目录命名。
3. 不删除已有日志字段。
4. 不改变默认 task_levels 行为。
5. 不改变 resume key。
6. 不引入无法通过 json.dump 的对象。
7. 修改后运行 py_compile；若改了 shell，运行 bash -n。
8. 最终说明改了哪些文件、验证了什么、没有跑长实验。
```

## 日志字段最低要求

每个 decision JSON 至少包含：

```json
{
  "task_id": 0,
  "decision_num": 0,
  "is_final": false,
  "target_used": [0.0, 0.0, 0.0],
  "pq3d_aux": {},
  "module": {},
  "follow": {},
  "steps_total_after_follow": 0
}
```

每个 task summary 至少包含：

```json
{
  "scene_name": "...",
  "episode_id": 0,
  "task_id": 0,
  "task_level": "instance",
  "navigation_type": "...",
  "sr": 0.0,
  "spl": 0.0,
  "task_time_sec": 0.0,
  "steps_total": 0,
  "end_reason": "final_decision",
  "module_name": "...",
  "module_hook_called": 0,
  "module_hook_applied": 0
}
```

## 验证清单

代码修改后必须至少做：

```bash
python3 -m py_compile hm3d-online/refhm3d-nav-sequence-analyze-anchor-<module>-refine1.py
bash -n scripts/run_<module>_instance_0.05_0.1.sh
bash -n scripts/run_<module>_all_0.05_0.1.sh
```

若当前 shell 没有 `habitat_sim`，不要用普通 `python script.py --help` 作为失败依据；launcher 会先进入 `mtu3d` 环境。

正式长跑前建议只做小范围 smoke：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/run_<module>_instance_0.05_0.1.sh detailed smoke
```

如果 smoke 过程中发现：

- output JSON 不增长
- live metrics 没输出
- process/task 下没有 decision JSON
- `module_info` 不是 JSON 可序列化
- `follower_error` 没被记为 SR/SPL 0

则先修模板或模块，不要启动 0.05-0.5 长跑。

## 禁止事项

- 不要为了新模块修改 baseline 结果计算。
- 不要在 launcher 里写复杂 Python 逻辑。
- 不要把模块参数写成通用名，必须有模块名前缀。
- 不要覆盖已有 `saved_versions/` 或重要 output logs。
- 不要把 VLM API key 写死进代码或 run_args。
- 不要删除旧字段来“清理”JSON；新增字段可以，破坏字段不可以。
- 不要让异常静默 fallback，必须记录原因。
- 不要用 `tmux ... "command > log 2>&1"` 启动正式长跑；这会让用户 attach 后看到黑屏。
- 不要用没有 `CUDA_VISIBLE_DEVICES=<gpu_id>` 的 tmux 命令启动正式长跑。

## 推荐命名

模块：

`hm3d-online/anchor_nav/<module>.py`

批量脚本：

`hm3d-online/refhm3d-nav-sequence-analyze-anchor-<module>-refine1.py`

instance launcher：

`scripts/run_<module>_instance_0.05_0.1.sh`

all launcher：

`scripts/run_<module>_all_0.05_0.1.sh`

输出目录：

```text
output_logs/anchor/<module>_instance_0.05_0.1/${RUN_TAG}
output_logs/anchor/<module>_all_0.05_0.1/${RUN_TAG}
```

主结果：

```text
refhm3d_seq_<module>_refine1_<start>_<end>.json
refhm3d_seq_<module>_refine1_effectiveness_<start>_<end>.json
```

concise 结果：

```text
refhm3d_seq_<module>_refine1_concisedesc_<start>_<end>.json
refhm3d_seq_<module>_refine1_effectiveness_concisedesc_<start>_<end>.json
```
