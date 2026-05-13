from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HM3D_ONLINE = ROOT / "hm3d-online"
for path in (ROOT, HM3D_ONLINE):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import anchor_nav.mqsc as mqsc
from anchor_nav.mqsc import MqscConfig, run_mqsc_refine


class FakeRep:
    def __init__(self) -> None:
        # model box layout: [x, z, y, dx, dz, dy]
        self.object_box = np.asarray(
            [
                [0.0, 0.0, 0.8, 0.25, 0.25, 0.6],  # target lamp
                [0.4, 0.1, 0.4, 0.7, 0.7, 0.8],   # nightstand
                [1.8, 0.2, 0.4, 2.1, 1.2, 0.8],   # bed, baseline wrong object
            ],
            dtype=float,
        )
        self.object_score = np.asarray([0.75, 0.85, 0.90], dtype=float)
        self.object_count = np.asarray([2.0, 4.0, 5.0], dtype=float)
        self.object_feat = np.zeros((3, 768), dtype=np.float32)
        self.open_vocab_feat = np.zeros((3, 768), dtype=np.float32)


class FakePQ3D:
    def __init__(self) -> None:
        self.representation_manager = FakeRep()
        self.last_decision_aux = {"real_object_decision_idx": 2}


def main() -> None:
    old_fn = mqsc.pq3d_stage2_object_logits

    def fake_logits(_pq3d, text: str) -> np.ndarray:
        t = text.lower()
        if "lamp" in t and "bed" not in t:
            return np.asarray([5.0, 0.2, 0.1], dtype=float)
        if "bed" in t or "nightstand" in t:
            return np.asarray([0.1, 5.0, 4.5], dtype=float)
        return np.asarray([4.5, 3.5, 2.5], dtype=float)

    try:
        mqsc.pq3d_stage2_object_logits = fake_logits
        new_target, info = run_mqsc_refine(
            sentence="lamp near bed and nightstand",
            task_type="instance",
            pq3d_model=FakePQ3D(),
            target_position=np.asarray([1.8, 0.4, 0.2], dtype=float),
            cfg=MqscConfig(
                use_vlm=False,
                cluster_eps=0.8,
                min_region_coverage=0.5,
                min_target_prob=0.05,
                min_region_margin=-999.0,
                min_selected_gain=-999.0,
                write_debug_json=False,
            ),
        )
    finally:
        mqsc.pq3d_stage2_object_logits = old_fn

    assert info["ok"] is True, info
    assert info["applied"] is True, info
    assert info["selected_object_index"] == 0, info
    assert np.allclose(new_target, [0.0, 0.8, 0.0]), new_target
    assert info["best_region"]["coverage"] >= 0.5, info["best_region"]
    print(
        "mqsc_fake_ok "
        f"selected={info['selected_object_index']} "
        f"baseline={info['baseline_object_index']} "
        f"coverage={info['best_region']['coverage']:.3f} "
        f"reason={info['reason']}"
    )


if __name__ == "__main__":
    main()
