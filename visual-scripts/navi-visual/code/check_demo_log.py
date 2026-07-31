#!/usr/bin/env python3
"""Validate a navi-visual demo log directory.

This checker is intentionally strict about control-flow evidence:
- TFFS may only appear as an active module on non-final frontier decisions.
- Vista2MQSC/MQSC may only appear as an active module on final decisions.
- Robot-eye frames, local overhead RGB, global RGB/top-down maps, and the
  final extra panorama must exist and be readable images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

FORBIDDEN_IMAGE_NAMES = {
    "topdown_fog.png",
    "topdown_full.png",
    "frontiers_on_topdown.png",
    "explored_map.png",
    "unexplored_map.png",
    "explored_unexplored_map.png",
    "topdown_map_fog.png",
    "topdown_map_rgb.png",
    "topdown_navmesh_rgb.png",
    "topdown_scene_rgb_raw.png",
    "topdown_scene_rgb_mosaic.png",
}
FORBIDDEN_IMAGE_SUFFIXES = ("_legend.png", "_rgb_raw.png")
FORBIDDEN_DIR_NAMES = {"topdown_maps", "topdown_rgb", "frontiers", "exploration_maps"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"_value": data}


def _find_latest_run(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "demo_visual_summary.json").exists():
        return root

    candidate_dirs = set()
    for p in root.rglob("demo_visual_summary.json"):
        if p.is_file():
            candidate_dirs.add(p.parent)
    for p in root.rglob("robot_eye_frames"):
        if p.is_dir():
            candidate_dirs.add(p.parent)
    for p in root.rglob("decisions"):
        if p.is_dir():
            candidate_dirs.add(p.parent)
    if not candidate_dirs:
        return root

    def newest_mtime(path: Path) -> float:
        mt = path.stat().st_mtime
        for child in path.rglob("*"):
            try:
                mt = max(mt, child.stat().st_mtime)
            except OSError:
                continue
        return mt

    return sorted(candidate_dirs, key=newest_mtime, reverse=True)[0]


def _image_stats(path: Path) -> Dict[str, Any]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"ok": False, "reason": "unreadable_image", "path": str(path)}
    arr = np.asarray(img)
    return {
        "ok": True,
        "path": str(path),
        "shape": list(arr.shape),
        "std": float(np.std(arr.astype(np.float32))),
        "mean": float(np.mean(arr.astype(np.float32))),
        "nonzero_ratio": float(np.count_nonzero(arr) / max(1, arr.size)),
    }


def _edge_ratios(path: Path, border_frac: float = 0.04) -> Dict[str, float]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"black": 1.0, "white": 1.0}
    h, w = img.shape[:2]
    bw = max(1, int(round(min(h, w) * float(border_frac))))
    edge = np.concatenate(
        [
            img[:bw, :, :].reshape(-1, 3),
            img[-bw:, :, :].reshape(-1, 3),
            img[:, :bw, :].reshape(-1, 3),
            img[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return {
        "black": float((edge.max(axis=1) <= 8).mean()),
        "white": float((edge.min(axis=1) >= 247).mean()),
    }


def _check_edge_limit(
    errors: List[str],
    path: Path,
    label: str,
    *,
    black_limit: float,
    white_limit: float,
) -> None:
    if not path.exists():
        return
    edge = _edge_ratios(path)
    if edge["black"] > float(black_limit) or edge["white"] > float(white_limit):
        errors.append(f"{label}: black/white edge ratio too high {edge}: {path}")


def _append_forbidden_artifact_errors(errors: List[str], run_dir: Path) -> None:
    for path in run_dir.rglob("*"):
        if any(part in FORBIDDEN_DIR_NAMES for part in path.parts):
            errors.append(f"forbidden old visual artifact directory still exists: {path}")
            continue
        if not path.is_file():
            continue
        name = path.name
        if name in FORBIDDEN_IMAGE_NAMES or any(name.endswith(suffix) for suffix in FORBIDDEN_IMAGE_SUFFIXES):
            errors.append(f"forbidden old/duplicate visual artifact still exists: {path}")


def _glob_images(root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        out.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(out))


def _check_images(
    *,
    label: str,
    paths: List[Path],
    min_count: int,
    errors: List[str],
    warnings: List[str],
    sample_count: int = 8,
    min_std: float = 1.0,
) -> Dict[str, Any]:
    report = {"label": label, "count": int(len(paths)), "samples": []}
    if len(paths) < int(min_count):
        errors.append(f"{label}: expected >= {min_count} image(s), found {len(paths)}")
    for path in paths[: int(sample_count)]:
        stats = _image_stats(path)
        report["samples"].append(stats)
        if not stats.get("ok", False):
            errors.append(f"{label}: unreadable image {path}")
            continue
        if float(stats.get("std", 0.0)) < float(min_std):
            warnings.append(f"{label}: low-variance image {path} std={stats.get('std'):.3f}")
        if float(stats.get("nonzero_ratio", 0.0)) < 0.05:
            warnings.append(f"{label}: mostly blank image {path}")
    return report


def _decision_dirs(run_dir: Path) -> List[Path]:
    dec_root = run_dir / "decisions"
    if not dec_root.exists():
        return []
    return sorted(p for p in dec_root.glob("dec_*") if p.is_dir())


def _check_decision_flow(dec_dir: Path, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    decision_json = dec_dir / "decision.json"
    if not decision_json.exists():
        errors.append(f"{dec_dir}: missing decision.json")
        return {"decision_dir": str(dec_dir), "ok": False, "reason": "missing_decision_json"}
    payload = _read_json(decision_json)
    is_final = bool(payload.get("is_final", False))
    tffs = payload.get("tffs", {}) if isinstance(payload.get("tffs", {}), dict) else {}
    mqsc = payload.get("vista2mqsc", {}) if isinstance(payload.get("vista2mqsc", {}), dict) else {}
    tffs_called = bool(tffs.get("called", tffs.get("tffs_called", False)))
    mqsc_called = bool(mqsc.get("called", False))

    if is_final and tffs_called:
        errors.append(f"{dec_dir}: final decision incorrectly called TFFS")
    if not is_final and mqsc_called:
        errors.append(f"{dec_dir}: non-final decision incorrectly called Vista2MQSC/MQSC")
    if not is_final and int(payload.get("frontier_count", 0)) <= 1:
        if tffs_called:
            errors.append(f"{dec_dir}: <=1 frontier should not call TFFS")
        reason = str(tffs.get("reason", tffs.get("gate_reason", "")))
        if reason and "single_frontier" not in reason and "no_tffs_attempted" not in reason and "no_frontier" not in reason:
            warnings.append(f"{dec_dir}: <=1 frontier has unexpected TFFS reason: {reason}")

    module_dir = dec_dir / "modules"
    if is_final:
        nonfinal_tffs_jsons = list((module_dir / "tffs").rglob("*.json")) if (module_dir / "tffs").exists() else []
        if nonfinal_tffs_jsons:
            errors.append(f"{dec_dir}: final decision has TFFS module files")
    else:
        mqsc_jsons = list((module_dir / "vista2mqsc").rglob("*.json")) if (module_dir / "vista2mqsc").exists() else []
        if mqsc_jsons:
            errors.append(f"{dec_dir}: non-final decision has Vista2MQSC module files")

    images_report = {
        "scan": _check_images(
            label=f"{dec_dir.name}/scan",
            paths=_glob_images(dec_dir / "scan", ["*.png", "**/*.png"]),
            min_count=1,
            errors=errors,
            warnings=warnings,
            sample_count=4,
        ),
        "local_topdown_rgb": _check_images(
            label=f"{dec_dir.name}/local_topdown_rgb",
            paths=_glob_images(dec_dir / "local_topdown_rgb", ["*.png", "**/*.png"]),
            min_count=2,
            errors=errors,
            warnings=warnings,
            sample_count=4,
        ),
        "global_topdown": _check_images(
            label=f"{dec_dir.name}/global_topdown",
            paths=_glob_images(dec_dir / "global_topdown", ["topdown_map.png", "topdown_scene_rgb.png", "topdown_scene_rgb_annotated.png"]),
            min_count=3,
            errors=errors,
            warnings=warnings,
            sample_count=6,
        ),
        "decision_topdown_map": _check_images(
            label=f"{dec_dir.name}/topdown_map",
            paths=_glob_images(dec_dir, ["topdown_map.png"]),
            min_count=1,
            errors=errors,
            warnings=warnings,
            sample_count=1,
        ),
    }
    map_info_path = dec_dir / "topdown_map_info.json"
    if map_info_path.exists():
        map_info = _read_json(map_info_path)
        if map_info.get("source") != "single_gray_topdown_theme":
            errors.append(f"{dec_dir}: topdown_map should use single gray theme: {map_info}")
    scene_info_path = dec_dir / "global_topdown" / "topdown_scene_rgb_info.json"
    if scene_info_path.exists():
        scene_info = _read_json(scene_info_path)
        if scene_info.get("source") != "global_topdown_rgb_tile_mosaic":
            errors.append(f"{dec_dir}: scene RGB source should be RGBD tile mosaic: {scene_info}")
        for key in ("fallback_filled_px", "dilation_filled_px", "inpainted_uncovered_px"):
            if int(scene_info.get(key, 0) or 0) != 0:
                errors.append(f"{dec_dir}: scene RGB should not use legacy blur/ghost filling: {key}={scene_info.get(key)}")
        if "no_patch_fallback" not in str(scene_info.get("paste_mode", "")):
            errors.append(f"{dec_dir}: scene RGB paste_mode should disable patch fallback: {scene_info.get('paste_mode')}")
    annotated_info_path = dec_dir / "global_topdown" / "topdown_scene_rgb_annotated_info.json"
    if annotated_info_path.exists():
        annotated_info = _read_json(annotated_info_path)
        if annotated_info.get("source") != "global_topdown_rgb_annotation":
            errors.append(f"{dec_dir}: annotated scene RGB source should be annotation metadata: {annotated_info}")
        if annotated_info.get("base_image") != "topdown_scene_rgb.png":
            errors.append(f"{dec_dir}: annotated scene RGB should reference clean base image: {annotated_info}")
    _check_edge_limit(
        errors,
        dec_dir / "global_topdown" / "topdown_scene_rgb.png",
        f"{dec_dir.name}/global_topdown_scene_rgb",
        black_limit=0.03,
        white_limit=0.03,
    )
    _check_edge_limit(
        errors,
        dec_dir / "local_topdown_rgb" / "robot_center_rgb.png",
        f"{dec_dir.name}/local_topdown_rgb",
        black_limit=0.05,
        white_limit=0.05,
    )
    return {
        "decision_dir": str(dec_dir),
        "ok": True,
        "is_final": bool(is_final),
        "frontier_count": int(payload.get("frontier_count", 0)),
        "tffs_called": bool(tffs_called),
        "tffs_applied": bool(tffs.get("tffs_applied", False)),
        "vista2mqsc_called": bool(mqsc_called),
        "images": images_report,
    }


def check_run(run_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    run_dir = _find_latest_run(run_dir)
    errors: List[str] = []
    warnings: List[str] = []
    summary_path = run_dir / "demo_visual_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    _append_forbidden_artifact_errors(errors, run_dir)
    if not summary_path.exists():
        errors.append(f"missing demo_visual_summary.json under {run_dir}")
    elif summary.get("ok") is False:
        errors.append(
            f"demo run failed: {summary.get('error_type', 'UnknownError')}: "
            f"{summary.get('error_message', '')}"
        )

    decision_dirs = _decision_dirs(run_dir)
    if not decision_dirs:
        errors.append(f"missing decisions/dec_* under {run_dir}")

    image_reports = {
        "robot_eye_frames": _check_images(
            label="robot_eye_frames",
            paths=_glob_images(run_dir / "robot_eye_frames", ["*.png"]),
            min_count=1,
            errors=errors,
            warnings=warnings,
            sample_count=10,
        ),
        "trajectory": _check_images(
            label="trajectory",
            paths=_glob_images(run_dir / "trajectory", ["*.png"]),
            min_count=1,
            errors=errors,
            warnings=warnings,
            sample_count=4,
        ),
        "final_panorama": _check_images(
            label="final_panorama",
            paths=_glob_images(run_dir / "modules" / "final_panorama", ["*.png"]),
            min_count=12,
            errors=errors,
            warnings=warnings,
            sample_count=6,
        ),
    }

    final_pano_json = run_dir / "modules" / "final_panorama" / "extra_stop_panorama.json"
    if not final_pano_json.exists():
        errors.append(f"missing final extra_stop_panorama.json under {run_dir}")
    else:
        pano = _read_json(final_pano_json)
        if bool(pano.get("navigation_steps_counted", True)):
            errors.append("final panorama must not be counted as navigation steps")
        if int(pano.get("view_count", 0)) < 12:
            errors.append("final panorama should contain 12 views")

    decisions = [_check_decision_flow(d, errors, warnings) for d in decision_dirs]
    final_count = sum(1 for d in decisions if bool(d.get("is_final", False)))
    nonfinal_count = sum(1 for d in decisions if d.get("ok") and not bool(d.get("is_final", False)))
    if final_count <= 0:
        warnings.append("no final decision found; demo may have stopped before object navigation")

    report: Dict[str, Any] = {
        "ok": len(errors) == 0,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "decision_count": int(len(decision_dirs)),
        "final_decision_count": int(final_count),
        "nonfinal_decision_count": int(nonfinal_count),
        "image_reports": image_reports,
        "decisions": decisions,
        "errors": errors,
        "warnings": warnings,
    }
    return len(errors) == 0, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a navi-visual demo log directory.")
    ap.add_argument("run_dir", nargs="?", default="visual-scripts/navi-visual/logs/sim")
    ap.add_argument("--report", default="", help="Optional JSON report path.")
    args = ap.parse_args()

    ok, report = check_run(Path(os.path.expanduser(args.run_dir)))
    report_path = Path(args.report).expanduser() if args.report else Path(report["run_dir"]) / "demo_visual_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    status = "OK" if ok else "FAIL"
    print(
        f"[demo-check] {status} run={report['run_dir']} decisions={report['decision_count']} "
        f"final={report['final_decision_count']} errors={len(report['errors'])} warnings={len(report['warnings'])}"
    )
    for err in report["errors"]:
        print(f"[demo-check][error] {err}")
    for warn in report["warnings"][:20]:
        print(f"[demo-check][warn] {warn}")
    print(f"[demo-check] report={report_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
