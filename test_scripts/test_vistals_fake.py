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

from anchor_nav.vista_ls import VistaLsConfig, visibility_based_viewpoint_decision


class FakeRep:
    def __init__(self) -> None:
        target_hab = []
        for x in np.linspace(-0.08, 0.08, 5):
            for y in np.linspace(0.35, 1.05, 5):
                target_hab.append([x, y, 0.0])
        target_hab_arr = np.asarray(target_hab, dtype=float)
        blocker_hab = np.asarray([[1.5, 0.0, 1.5], [-1.5, 0.0, -1.5]], dtype=float)
        all_hab = np.vstack([target_hab_arr, blocker_hab])
        # The production representation stores model xyz, converted by [x, z, y].
        self.point_cloud = all_hab[:, [0, 2, 1]]
        self.object_mask = np.zeros((all_hab.shape[0], 1), dtype=bool)
        self.object_mask[: target_hab_arr.shape[0], 0] = True


class FakePathFinder:
    def is_navigable(self, point) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.all(np.isfinite(p)) and abs(p[0]) <= 2.0 and abs(p[2]) <= 2.0)

    def get_island(self, point) -> int:
        return 0 if self.is_navigable(point) else -1

    def snap_point(self, point=None, island_index=0):
        del island_index
        p = np.asarray(point, dtype=float).reshape(3).copy()
        p[1] = 0.0
        p[0] = float(np.clip(p[0], -2.0, 2.0))
        p[2] = float(np.clip(p[2], -2.0, 2.0))
        return p

    def distance(self, start, end) -> float:
        return float(np.linalg.norm(np.asarray(start, dtype=float) - np.asarray(end, dtype=float)))


def main() -> None:
    cfg = VistaLsConfig(
        enable_vvd_replacement=True,
        prefer_visible_baseline=False,
        target_sample_count=25,
        max_ray_sample_count=40,
        occlusion_radius_m=0.05,
        min_visibility_score=0.0,
        r_min_m=0.30,
        r_max_m=0.75,
        radial_step_m=0.15,
        angle_step_deg=30.0,
        shell_min_m=0.25,
        shell_max_m=0.85,
        relaxed_shell_min_m=0.20,
        relaxed_shell_max_m=0.90,
        min_clearance_m=0.05,
        min_component_size=3,
    )
    info = visibility_based_viewpoint_decision(
        rep=FakeRep(),
        selected_slot_index=0,
        path_finder=FakePathFinder(),
        agent_position_xyz=np.asarray([0.0, 0.0, -1.5], dtype=float),
        cfg=cfg,
        baseline_target_xyz=np.asarray([0.0, 0.0, -0.5], dtype=float),
    )
    records = info["visibility_records"]
    selected = [rec for rec in records if rec.get("selected_by_vista_ls")]
    component_records = [rec for rec in records if rec.get("component_id") is not None]

    assert info["selection_policy"] == "vista_ls_level_set_medial_center", info["selection_policy"]
    assert int(info["vista_ls_component_count"]) >= 1, info["vista_ls_component_count"]
    assert selected, "expected one selected_by_vista_ls record"
    assert component_records, "expected component_id on feasible records"
    assert selected[0].get("medial_distance_to_boundary") is not None
    assert info["best_viewpoint_xyz"] is not None
    print(
        "vistals_fake_ok "
        f"components={info['vista_ls_component_count']} "
        f"selected_component={info['vista_ls_selected_component_id']} "
        f"medial={info['vista_ls_selected_medial_radius']} "
        f"best_visibility={info['best_visibility_score']:.6f}"
    )


if __name__ == "__main__":
    main()
