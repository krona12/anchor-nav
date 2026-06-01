# Key Logs Manifest

This directory stores the key run logs used for the RefHM3D sequence comparison
and module-effectiveness analysis generated on 2026-05-29.

Generated analysis files:
- `test_scripts/refhm3d_sequence_experiment_comparison.md`
- `test_scripts/refhm3d_sequence_experiment_comparison.json`
- `test_scripts/module_effectiveness_comparison.md`
- `test_scripts/module_effectiveness_comparison.json`

Analysis scripts:
- `test_scripts/compare_refhm3d_sequence_experiments.py`
- `test_scripts/analyze_module_effectiveness.py`

Copied log mapping:

| Local copy | Original source |
| --- | --- |
| `baseline_0.0_0.2_20260513-112413.log` | `output_logs/baseline_all_0_0.2/20260513-112352-detailed/refhm3d-nav-sequence-baseline-20260513-112413-119890-pid1086119.log` |
| `baseline_0.2_1.0_20260513-101924.log` | `output_logs/baseline_all_0.2_1.0/20260513-101905-detailed/refhm3d-nav-sequence-baseline-20260513-101924-534262-pid862786.log` |
| `vistals_0.0_0.2_20260513-112519.log` | `output_logs/anchor/vistals_all_0_0.2/20260513-112500-detailed-fast-grid-v2-cuda2/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-112519-263530-pid1088698.log` |
| `vistals_0.2_1.0_partial_oom_20260513-102111.log` | `output_logs/anchor/vistals_all_0.2_1.0/20260513-102052-detailed/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-102111-463001-pid868899.log` |
| `mqsc_r1_0.0_0.2_20260514-144811.log` | `output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260514-144811-731948-pid1525210.log` |
| `mqsc_r1_0.2_1.0_20260516-113222.log` | `output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260516-113222-410036-pid56740.log` |
| `vista2mqsc_0.0_0.2_20260514-150659.log` | `output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260514-150659-226187-pid1684292.log` |
| `vista2mqsc_0.2_1.0_20260516-113222.log` | `output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260516-113222-409168-pid56747.log` |

Important note:
- `+ VISTA-LS` 0.2-1.0 is a partial run: the run hit OOM after 1736 rows in
  that shard, so its combined analysis covers 2436/3600 tasks.

