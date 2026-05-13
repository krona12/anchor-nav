#!/usr/bin/env bash
set -euo pipefail

cd /home/chenlin/krona/anchor-nav
source /opt/conda/bin/activate
conda activate mtu3d
export PYTHONPATH=/home/chenlin/krona/MTU3D:./:./hm3d-online:./hm3d-online/FastSAM:${PYTHONPATH:-}

python3 - <<'PY'
import json
from pathlib import Path
from types import SimpleNamespace

from anchor_nav.acsd import ACSDConfig, AnchorConditionedSoftDecomposition

legacy_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260509-164950-detailed-vlm-anchors-object-only-margin005-min080-tmux-all-20260509-164947/process")
topk8_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260510-153436-detailed-vlm-anchors-object-only-adaptive-level-policy-v5-topk8-safe-context-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260510-153434/process")
v8_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260510-172556-detailed-vlm-anchors-object-only-adaptive-level-policy-v8-topk8-room-target-positive-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260510-172554/process")
v13_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260510-215755-detailed-vlm-anchors-object-only-adaptive-level-policy-v13-topk12-region-positive-target-rescue-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260510-215752/process")
v14_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260510-232430-detailed-vlm-anchors-object-only-adaptive-level-policy-v14-topk12-region-target-adv018-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260510-232428/process")
v15_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-015056-detailed-vlm-anchors-object-only-adaptive-level-policy-v15-topk12-safe-room-instance-region-score-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-015053/process")
v16_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-021701-detailed-vlm-anchors-object-only-adaptive-level-policy-v16-topk12-exact-room-region-anchor-rescue-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-021658/process")
v18_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-025849-detailed-vlm-anchors-object-only-adaptive-level-policy-v18-topk12-v13like-roomlocal070-badcase-blocks-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-025847/process")
v20_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-051643-detailed-vlm-anchors-object-only-adaptive-level-policy-v20-topk12-region-exact-target-room-instance-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-051641/process")
v23_root = Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-085727-detailed-vlm-anchors-object-only-adaptive-level-policy-v23-topk12-no-room-no-small-instance-identity-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-085725/process")
v24_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260511-171722-detailed-vlm-anchors-object-only-adaptive-level-policy-v24-0102-topk12-region-midconf-context-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260511-171720/process")
v25_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260511-183649-detailed-vlm-anchors-object-only-adaptive-level-policy-v25-0102-lowerconf-region-context-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260511-183647/process")
v26_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260511-201749-detailed-vlm-anchors-object-only-adaptive-level-policy-v26-0102-lowconf-region-anchor-instance-safe-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260511-201747/process")
v27_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260512-004359-detailed-vlm-anchors-object-only-adaptive-level-policy-v27-0102-region-local-anchor-safety-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260512-004356/process")
v28_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260512-020402-detailed-vlm-anchors-object-only-adaptive-level-policy-v28-0102-region-high-context-rescue-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260512-020400/process")
v29_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260512-095955-detailed-vlm-anchors-object-only-adaptive-level-policy-v29-0102-strong-baseline-region-guard-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260512-095953/process")
v30_0102_root = Path("output_logs/anchor/acsd_all_0.1_0.2/20260512-124658-detailed-vlm-anchors-object-only-adaptive-level-policy-v30-0102-region-anchor-target-rescue-margin005-min080-tmux-all-acsd-all-0102-adaptive-level-20260512-124656/process")
cases = [
    (v15_root, "room_local_target_case11_now_disabled", "scene=00802-wcojb4TFT35/episode=0/task=1/dec_005/acsd_decision.json", False),
    (v15_root, "room_moderate_target_local_now_disabled", "scene=00802-wcojb4TFT35/episode=3/task=1/dec_007/acsd_decision.json", False),
    (v15_root, "region_anchor_substitute_no_gain_blocked", "scene=00802-wcojb4TFT35/episode=2/task=4/dec_015/acsd_decision.json", False),
    (v16_root, "region_context_probe_no_gain_blocked", "scene=00802-wcojb4TFT35/episode=0/task=3/dec_010/acsd_decision.json", False),
    (v16_root, "region_context_probe_neighbor_block", "scene=00802-wcojb4TFT35/episode=0/task=4/dec_011/acsd_decision.json", False),
    (v20_root, "region_low_raw_exact_target_now_blocked", "scene=00802-wcojb4TFT35/episode=2/task=3/dec_013/acsd_decision.json", False),
    (legacy_root, "region_low_raw_context_block", "scene=00802-wcojb4TFT35/episode=1/task=1/dec_010/acsd_decision.json", False),
    (legacy_root, "region_positive", "scene=00802-wcojb4TFT35/episode=8/task=4/dec_011/acsd_decision.json", True),
    (legacy_root, "region_weak_harm_block", "scene=00802-wcojb4TFT35/episode=18/task=2/dec_010/acsd_decision.json", False),
    (legacy_root, "instance_room_positive", "scene=00802-wcojb4TFT35/episode=16/task=0/dec_004/acsd_decision.json", True),
    (legacy_root, "instance_no_room_block", "scene=00803-k1cupFYWXJ6/episode=2/task=0/dec_009/acsd_decision.json", False),
    (topk8_root, "region_context_override_high_baseline_block", "scene=00802-wcojb4TFT35/episode=1/task=2/dec_011/acsd_decision.json", False),
    (v8_root, "region_mild_target_anchor_now_blocked", "scene=00802-wcojb4TFT35/episode=2/task=3/dec_011/acsd_decision.json", False),
    (v13_root, "region_small_adv_downstream_harm_block", "scene=00802-wcojb4TFT35/episode=10/task=2/dec_012/acsd_decision.json", False),
    (v14_root, "room_low_raw_target_now_blocked", "scene=00802-wcojb4TFT35/episode=14/task=1/dec_005/acsd_decision.json", False),
    (v14_root, "instance_low_raw_identity_now_blocked", "scene=00802-wcojb4TFT35/episode=16/task=0/dec_004/acsd_decision.json", False),
    (v14_root, "region_weak_anchor_low_raw_now_blocked", "scene=00802-wcojb4TFT35/episode=18/task=2/dec_010/acsd_decision.json", False),
    (v18_root, "region_exact_target_current_rescue", "scene=00802-wcojb4TFT35/episode=6/task=1/dec_005/acsd_decision.json", True),
    (v23_root, "region_mid_conf_target_context_rescue", "scene=00802-wcojb4TFT35/episode=13/task=1/dec_014/acsd_decision.json", True),
    (v24_0102_root, "region_0102_lower_conf_context_rescue", "scene=00808-y9hTuugGdiq/episode=7/task=3/dec_012/acsd_decision.json", True),
    (v24_0102_root, "region_0102_weak_anchor_block", "scene=00808-y9hTuugGdiq/episode=8/task=4/dec_013/acsd_decision.json", False),
    (v25_0102_root, "instance_0102_low_raw_anchor_regression_block", "scene=00808-y9hTuugGdiq/episode=10/task=0/dec_005/acsd_decision.json", False),
    (v25_0102_root, "region_0102_low_conf_target_anchor_rescue", "scene=00808-y9hTuugGdiq/episode=13/task=4/dec_016/acsd_decision.json", True),
    (v26_0102_root, "region_0102_exact_target_anchor_regression_block", "scene=00810-CrMo8WxCyVb/episode=3/task=1/dec_005/acsd_decision.json", False),
    (v27_0102_root, "region_0102_high_context_target_rescue", "scene=00808-y9hTuugGdiq/episode=8/task=4/dec_013/acsd_decision.json", True),
    (v28_0102_root, "region_0102_strong_baseline_low_target_block", "scene=00808-y9hTuugGdiq/episode=19/task=1/dec_009/acsd_decision.json", False),
    (v28_0102_root, "region_0102_strong_baseline_small_target_adv_block", "scene=00810-CrMo8WxCyVb/episode=4/task=4/dec_019/acsd_decision.json", False),
    (v29_0102_root, "region_0102_strong_anchor_moderate_target_now_blocked", "scene=00808-y9hTuugGdiq/episode=13/task=3/dec_016/acsd_decision.json", False),
    (v30_0102_root, "instance_0102_low_raw_exact_target_rescue", "scene=00808-y9hTuugGdiq/episode=6/task=0/dec_006/acsd_decision.json", True),
    (Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-025849-detailed-vlm-anchors-object-only-adaptive-level-policy-v18-topk12-v13like-roomlocal070-badcase-blocks-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-025847/process"), "region_weak_context_current_block", "scene=00802-wcojb4TFT35/episode=6/task=4/dec_008/acsd_decision.json", False),
    (Path("output_logs/anchor/acsd_all_0.05_0.1/20260511-025849-detailed-vlm-anchors-object-only-adaptive-level-policy-v18-topk12-v13like-roomlocal070-badcase-blocks-margin005-min080-tmux-all-acsd-all-00501-adaptive-level-20260511-025847/process"), "instance_small_target_identity_now_blocked", "scene=00802-wcojb4TFT35/episode=7/task=0/dec_004/acsd_decision.json", False),
]
acsd = AnchorConditionedSoftDecomposition(
    ACSDConfig(
        vlm_model="gpt-4o-mini",
        object_top_k=12,
        correction_margin=0.005,
        object_correction_min_baseline_score=0.80,
    )
)
failures = []
for root, name, rel, expected in cases:
    path = root / rel
    d = json.load(open(path, "r", encoding="utf-8"))
    stage2 = json.load(open(path.parent / "stage2_decision.json", "r", encoding="utf-8"))
    task_summary = json.load(open(path.parent.parent / "task_summary.json", "r", encoding="utf-8"))
    process = acsd.process_decision(
        instruction=d["instruction"],
        baseline_decision=d["baseline"],
        observation_context={"representation_manager": SimpleNamespace(object_first_rgb=None)},
        candidate_context={"stage2": stage2, "decomposition": d["decomposition"], "output_dir": str(path.parent / "adaptive_replay")},
        episode_context={
            "scene_name": d["scene_name"],
            "episode_id": d["episode_id"],
            "task_id": d["task_id"],
            "task_level": task_summary.get("task_level", ""),
        },
        gt_context={},
    )
    rer = process["object_rerank"]
    print(
        f"{name} expected={expected} got={process['correction_applied']} "
        f"reason={process['correction_reason']} policy={rer.get('level_policy')} "
        f"target_adv={rer.get('target_advantage'):.6f} anchor_adv={rer.get('anchor_advantage'):.6f} "
        f"semantic_gate={rer.get('semantic_gate')} room_gate={rer.get('room_or_identity_gate')} "
        f"candidate_gate={rer.get('candidate_confidence_gate')} margin_gate={rer.get('margin_gate')} "
        f"best={(rer.get('ranked_candidates') or [{}])[0].get('slot_index')} base={rer.get('baseline_slot_index')} "
        f"best_base={(rer.get('ranked_candidates') or [{}])[0].get('baseline_object_score')} "
        f"best_acsd={(rer.get('ranked_candidates') or [{}])[0].get('acsd_score')} "
    )
    if bool(process["correction_applied"]) != bool(expected):
        failures.append(name)
if failures:
    raise SystemExit(f"unexpected replay outcomes: {failures}")
PY
