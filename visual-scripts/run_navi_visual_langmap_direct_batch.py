#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_BATCH = SCRIPT_DIR / "run_navi_visual_langmap_batch_guided.py"


def _load_guided_batch():
    spec = importlib.util.spec_from_file_location("guided_batch_for_direct", str(GUIDED_BATCH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import guided batch runner: {GUIDED_BATCH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["guided_batch_for_direct"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    batch = _load_guided_batch()
    batch.CHECK_SH = SCRIPT_DIR / "run_navi_visual_direct_check.sh"
    batch.DEFAULT_BATCH_ROOT = SCRIPT_DIR / "navi-visual" / "logs" / "direct-batch-5"

    def quick_artifact_check(run_dir: Path) -> List[str]:
        issues: List[str] = []
        required = [
            run_dir / "direct_navigation_summary.json",
            run_dir / "module_visual_check_report.json",
            run_dir / "direct_visual_check_report.json",
            run_dir / "log_streams" / "manifest.json",
            run_dir / "baseline_decisions" / "dec_000" / "decision.json",
            run_dir / "baseline_decisions" / "dec_000" / "panorama" / "current_decision_panorama_vfv_order.jpg",
            run_dir / "baseline_decisions" / "dec_000" / "topdown_scene_rgb.png",
            run_dir / "baseline_decisions" / "dec_000" / "topdown_scene_rgb_annotated.png",
            run_dir / "final" / "panorama" / "current_decision_panorama_vfv_order.jpg",
            run_dir / "final" / "topdown_scene_rgb.png",
            run_dir / "final" / "topdown_scene_rgb_annotated.png",
            run_dir / "final" / "target_facing_rgb.png",
            run_dir / "final" / "target_facing_rgb_info.json",
            run_dir / "trajectory" / "route_start_to_goal.png",
        ]
        for path in required:
            if not path.exists() or path.stat().st_size <= 0:
                issues.append(f"missing/empty required direct artifact: {path}")
        report = batch._read_json(run_dir / "direct_visual_check_report.json")
        if report and not bool(report.get("ok", False)):
            issues.append(f"direct_visual_check_report not ok: {run_dir / 'direct_visual_check_report.json'}")
        return issues

    batch.quick_artifact_check = quick_artifact_check
    return int(batch.main())


if __name__ == "__main__":
    raise SystemExit(main())
