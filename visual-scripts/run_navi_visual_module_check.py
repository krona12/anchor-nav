#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODULE_SIM = SCRIPT_DIR / "navi-visual" / "code" / "module_sim.py"

DISALLOWED_MOCK_ENV_NAMES = {
    "MOCK",
    "USE_MOCK",
    "VLM_MOCK",
    "MOCK_VLM",
    "FAKE_VLM",
    "USE_FAKE_VLM",
    "ANCHOR_NAV_MOCK",
    "MODULE_SIM_MOCK",
    "DISABLE_VLM",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _require_single_cuda_device() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to one device")
    if "," in visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must name exactly one device, got {visible!r}")


def _require_real_run_env(logs_dir: Path) -> None:
    if any("smoke" in part.lower() for part in logs_dir.parts):
        raise RuntimeError(f"smoke output paths are not allowed for a real module check: {logs_dir}")
    bad_env = []
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name in DISALLOWED_MOCK_ENV_NAMES and str(value).strip() not in ("", "0", "false", "False"):
            bad_env.append(f"{name}={value}")
    if bad_env:
        raise RuntimeError("mock/fake VLM or simulator environment is not allowed: " + ", ".join(sorted(bad_env)))


def _image_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return False
    return bool(np.asarray(img).std() > 0.5)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image: {path}")
    return img


def _image_stats(path: Path) -> Dict[str, Any]:
    img = _load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "path": str(path),
        "shape": list(img.shape),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_ratio": float((gray < 8).mean()),
        "bright_ratio": float((gray > 247).mean()),
        "unique_sample": int(len(np.unique(img.reshape(-1, 3)[:: max(1, img.reshape(-1, 3).shape[0] // 4096)], axis=0))),
    }


def _dark_rgb_ratio(path: Path, threshold: int = 12) -> float:
    img = _load_image(path)
    dark = np.all(img <= int(threshold), axis=2)
    return float(dark.mean())


def _append_image_quality_errors(errors: List[str], path: Path, label: str, *, allow_discrete: bool = False) -> None:
    try:
        stats = _image_stats(path)
    except Exception as exc:
        errors.append(f"bad image {label}: {path} ({type(exc).__name__}: {exc})")
        return
    if stats["std"] <= 1.0:
        errors.append(f"low-information image {label}: {path} stats={stats}")
    if not allow_discrete and stats["unique_sample"] < 8:
        errors.append(f"low-color image {label}: {path} stats={stats}")
    if stats["dark_ratio"] > 0.98 or stats["bright_ratio"] > 0.98:
        errors.append(f"nearly solid image {label}: {path} stats={stats}")


def _audit_exploration_maps(errors: List[str], mod_dir: Path, module_name: str) -> Dict[str, Any]:
    exp_dir = mod_dir / "exploration_maps"
    explored_p = exp_dir / "explored_map.png"
    unexplored_p = exp_dir / "unexplored_map.png"
    merged_p = exp_dir / "explored_unexplored_map.png"
    if not explored_p.exists() or not unexplored_p.exists() or not merged_p.exists():
        errors.append(f"{module_name} missing exploration_maps trio under {exp_dir}")
        return {}
    explored = cv2.imread(str(explored_p), cv2.IMREAD_GRAYSCALE)
    unexplored = cv2.imread(str(unexplored_p), cv2.IMREAD_GRAYSCALE)
    merged = cv2.imread(str(merged_p), cv2.IMREAD_COLOR)
    if explored is None or unexplored is None or merged is None:
        errors.append(f"{module_name} cannot read exploration maps under {exp_dir}")
        return {}
    if explored.shape != unexplored.shape or explored.shape[:2] != merged.shape[:2]:
        errors.append(f"{module_name} exploration map shape mismatch: explored={None if explored is None else explored.shape}, unexplored={None if unexplored is None else unexplored.shape}, merged={None if merged is None else merged.shape}")
        return {}
    exp = explored > 0
    unexp = unexplored > 0
    nav = np.logical_or(exp, unexp)
    overlap = np.logical_and(exp, unexp)
    if int(nav.sum()) <= 64:
        errors.append(f"{module_name} navigable mask too small in {exp_dir}: {int(nav.sum())} px")
    if int(overlap.sum()) > 0:
        errors.append(f"{module_name} explored/unexplored overlap in {exp_dir}: {int(overlap.sum())} px")
    n_labels, labels = cv2.connectedComponents(nav.astype(np.uint8), connectivity=8)
    comp_areas = [int((labels == i).sum()) for i in range(1, n_labels)]
    largest = max(comp_areas) if comp_areas else 0
    largest_ratio = float(largest / max(int(nav.sum()), 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(nav.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    thin_ratio = float((int(nav.sum()) - int(opened.sum())) / max(int(nav.sum()), 1))
    if thin_ratio > 0.35:
        errors.append(f"{module_name} navigable mask has too many thin/sliver pixels: thin_ratio={thin_ratio:.3f} in {exp_dir}")
    return {
        "shape": list(explored.shape),
        "navigable_px": int(nav.sum()),
        "explored_px": int(exp.sum()),
        "unexplored_px": int(unexp.sum()),
        "component_count": int(n_labels - 1),
        "largest_component_ratio": largest_ratio,
        "thin_ratio_open5": thin_ratio,
    }


def _read_exploration_masks(mod_dir: Path) -> Dict[str, np.ndarray]:
    exp_dir = mod_dir / "exploration_maps"
    explored = cv2.imread(str(exp_dir / "explored_map.png"), cv2.IMREAD_GRAYSCALE)
    unexplored = cv2.imread(str(exp_dir / "unexplored_map.png"), cv2.IMREAD_GRAYSCALE)
    if explored is None or unexplored is None:
        return {}
    exp = explored > 0
    unexp = unexplored > 0
    return {"explored": exp, "unexplored": unexp, "navigable": np.logical_or(exp, unexp)}


def _audit_topdown_pair(errors: List[str], mod_dir: Path, module_name: str) -> Dict[str, Any]:
    paths = {
        "fog": mod_dir / "topdown_map_fog.png",
        "rgb": mod_dir / "topdown_map_rgb.png",
        "navmesh_rgb": mod_dir / "topdown_navmesh_rgb.png",
        "scene_raw": mod_dir / "topdown_scene_rgb_raw.png",
        "scene_mosaic": mod_dir / "topdown_scene_rgb_mosaic.png",
    }
    for key, path in paths.items():
        _require_image(errors, path, f"{module_name} {key}")
        if path.exists():
            _append_image_quality_errors(
                errors,
                path,
                f"{module_name} {key}",
                allow_discrete=key in ("fog", "rgb", "navmesh_rgb"),
            )
    shapes: Dict[str, List[int]] = {}
    for key, path in paths.items():
        if path.exists():
            try:
                shapes[key] = list(_load_image(path).shape[:2])
            except Exception:
                pass
    if len({tuple(v) for v in shapes.values()}) > 1:
        errors.append(f"{module_name} topdown shape mismatch in {mod_dir}: {shapes}")
    info_path = mod_dir / "topdown_rgb_mosaic_info.json"
    _require_file(errors, info_path, f"{module_name} rgb mosaic info")
    info = _read_json(info_path) if info_path.exists() else {}
    if info.get("ok"):
        if info.get("source") != "global_topdown_rgb_tile_mosaic":
            errors.append(f"{module_name} rgb mosaic source is not scene RGB mosaic: {info.get('source')} in {mod_dir}")
        fill = float(info.get("filled_navigable_ratio", 0.0) or 0.0)
        if fill < 0.35:
            errors.append(f"{module_name} rgb mosaic has low navigable fill: {fill:.3f} in {mod_dir}")
        tile_count = int(info.get("tile_count", 0) or 0)
        ok_tiles = int(info.get("successful_tile_count", 0) or 0)
        if tile_count <= 0 or ok_tiles / max(tile_count, 1) < 0.80:
            errors.append(f"{module_name} rgb mosaic tile success too low: {ok_tiles}/{tile_count} in {mod_dir}")
        if int(info.get("successful_tile_count", 0) or 0) <= 0:
            errors.append(f"{module_name} rgb mosaic has no successful tiles in {mod_dir}")
        paste_mode = str(info.get("paste_mode", ""))
        if "rgbd_projected" not in paste_mode:
            bad_roundtrip = [
                r
                for r in list(info.get("records", []))
                if r.get("ok") and float(r.get("roundtrip_error_px", 0.0) or 0.0) > 8.0
            ]
            if bad_roundtrip:
                errors.append(f"{module_name} rgb mosaic has tile/map roundtrip errors, first={bad_roundtrip[0]}")
    else:
        errors.append(f"{module_name} rgb mosaic did not run ok in {mod_dir}: {info}")
    scene_mosaic_path = mod_dir / "topdown_scene_rgb_mosaic.png"
    if scene_mosaic_path.exists():
        dark_ratio = _dark_rgb_ratio(scene_mosaic_path)
        if dark_ratio > 0.10:
            errors.append(
                f"{module_name} scene RGB mosaic still has a large dark/black background ratio: "
                f"{dark_ratio:.3f} in {scene_mosaic_path}"
            )
    exp_info = _audit_exploration_maps(errors, mod_dir, module_name)
    if exp_info and shapes:
        expected = tuple(exp_info["shape"])
        for key, shape in shapes.items():
            if tuple(shape) != expected:
                errors.append(f"{module_name} {key} shape {shape} != exploration shape {expected} in {mod_dir}")
    return {"shapes": shapes, "mosaic": info, "exploration": exp_info}


def _require_file(errors: List[str], path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        errors.append(f"missing/empty {label}: {path}")


def _require_image(errors: List[str], path: Path, label: str) -> None:
    if not _image_ok(path):
        errors.append(f"bad image {label}: {path}")


def _find_run_dir(output: str) -> Path:
    matches = re.findall(r"outputs under:\s*(.+)", output)
    if matches:
        return Path(matches[-1].strip()).expanduser().resolve()
    candidates = sorted((SCRIPT_DIR / "navi-visual" / "logs" / "module_sim").glob("run=*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("could not find module_sim output directory")
    return candidates[-1].resolve()


def validate_run(run_dir: Path) -> List[str]:
    errors: List[str] = []
    audit: Dict[str, Any] = {"run_dir": str(run_dir), "modules": {}}
    summary_path = run_dir / "module_sim_summary.json"
    _require_file(errors, summary_path, "summary json")
    summary = _read_json(summary_path) if summary_path.exists() else {}
    run_info_path = run_dir / "run_info.json"
    run_info = _read_json(run_info_path) if run_info_path.exists() else {}
    topdown_info = summary.get("topdown_map") or run_info.get("topdown_map") or {}
    audit["topdown_map"] = topdown_info
    if topdown_info.get("strategy") == "fallback_standard_topdown":
        errors.append(f"topdown map fell back to standard unfiltered slice: {topdown_info}")
    rounds = int(summary.get("decision_rounds", 0) or 0)
    if rounds <= 0:
        errors.append(f"expected at least one decision round, got {rounds}")

    _require_image(errors, run_dir / "trajectory" / "route_start_to_goal.png", "trajectory")
    _require_image(errors, run_dir / "trajectory" / "route_start_to_goal_legend.png", "trajectory legend")

    prev_explored: Any = None
    for dec in range(max(1, rounds)):
        dtag = f"dec_{dec:03d}"
        tdir = run_dir / "modules" / "tffs" / dtag
        mdir = run_dir / "modules" / "mqsc_r1" / dtag
        for module_name, mod_dir in (("tffs", tdir), ("mqsc_r1", mdir)):
            audit["modules"][f"{module_name}/{dtag}"] = _audit_topdown_pair(errors, mod_dir, f"{module_name}/{dtag}")
            _require_image(errors, mod_dir / "topdown_cam_rgb_raw.png", f"{module_name}/{dtag} local topdown rgb")
            _require_image(errors, mod_dir / "topdown_cam_depth.png", f"{module_name}/{dtag} local topdown depth")

        t_masks = _read_exploration_masks(tdir)
        m_masks = _read_exploration_masks(mdir)
        if t_masks and m_masks:
            if t_masks["navigable"].shape != m_masks["navigable"].shape:
                errors.append(f"{dtag} TFFS/MQSC navigable shape mismatch")
            else:
                diff = np.logical_xor(t_masks["navigable"], m_masks["navigable"])
                if float(diff.mean()) > 0.005:
                    errors.append(f"{dtag} TFFS/MQSC navigable masks differ: diff_ratio={float(diff.mean()):.4f}")
                exp_diff = np.logical_xor(t_masks["explored"], m_masks["explored"])
                if float(exp_diff.mean()) > 0.005:
                    errors.append(f"{dtag} TFFS/MQSC explored masks differ: diff_ratio={float(exp_diff.mean()):.4f}")
        if t_masks:
            if prev_explored is not None and prev_explored.shape == t_masks["explored"].shape:
                lost = np.logical_and(prev_explored, np.logical_not(t_masks["explored"]))
                if float(lost.sum() / max(prev_explored.sum(), 1)) > 0.005:
                    errors.append(f"{dtag} explored mask is not monotonic: lost_px={int(lost.sum())}")
            prev_explored = t_masks["explored"].copy()

        tffs_json = tdir / "tffs_decision.json"
        if tffs_json.exists():
            tj = _read_json(tffs_json)
            if str(tj.get("real_tffs_file", "")).find("hm3d-online/anchor_nav/tffs.py") < 0:
                errors.append(f"tffs_decision.json does not point at real tffs.py: {tffs_json}")
            if not bool(tj.get("skipped", False)):
                if bool(tj.get("vlm_interval_allowed", True)) and int(tj.get("vlm_call_count", 0) or 0) <= 0:
                    errors.append(f"TFFS interval allowed but no VLM calls recorded: {tffs_json}")
                for score in list(tj.get("vlm_scores", [])):
                    fi = int(score.get("frontier_index", -1))
                    if fi >= 0:
                        _require_file(errors, tdir / "prompts" / f"frontier_{fi:02d}_prompt.txt", "tffs prompt")
                        response_path = tdir / "prompts" / f"frontier_{fi:02d}_response.txt"
                        _require_file(errors, response_path, "tffs response")
                        if response_path.exists() and response_path.read_text(encoding="utf-8", errors="replace").strip() == "":
                            errors.append(f"empty TFFS response log: {response_path}")
        else:
            _require_file(errors, tffs_json, "tffs decision json")

        _require_file(errors, mdir / "mqsc_r1_decision.json", "mqsc decision json")
        _require_file(errors, mdir / "decomposition_prompt.txt", "mqsc decomposition prompt")
        _require_file(errors, mdir / "decomposition_response.txt", "mqsc decomposition response")
        _require_image(errors, tdir / "panorama" / "current_decision_panorama_vfv_order.jpg", "tffs VFV-order panorama")

    vdir = run_dir / "modules" / "vista_ls" / "final"
    final_pano = run_dir / "modules" / "final_panorama"
    _require_image(errors, final_pano / "current_decision_panorama_vfv_order.jpg", "final stop panorama")
    for i in range(12):
        _require_image(errors, final_pano / f"view_{i:02d}.png", f"final stop panorama view {i:02d}")
    _require_file(errors, vdir / "vista_ls_decision.json", "vista_ls decision json")
    audit["modules"]["vista_ls/final"] = _audit_topdown_pair(errors, vdir, "vista_ls/final")
    _require_image(errors, vdir / "topdown_cam_rgb_raw.png", "vista local topdown rgb")
    _require_image(errors, vdir / "topdown_cam_depth.png", "vista local topdown depth")
    _require_image(errors, vdir / "vista_ls_target_rgb_raw.png", "vista target rgb")
    vj = _read_json(vdir / "vista_ls_decision.json") if (vdir / "vista_ls_decision.json").exists() else {}
    target_rgb = vj.get("target_rgb", {}) if isinstance(vj.get("target_rgb", {}), dict) else {}
    selected = vj.get("selected_viewpoint")
    if isinstance(selected, dict):
        if not bool(selected.get("reachable", False)) or not bool(selected.get("same_island", False)):
            errors.append(f"vista_ls selected viewpoint is not reachable/same island: {selected}")
        if str(selected.get("category", "")) != "feasible":
            errors.append(f"vista_ls selected viewpoint category is not feasible: {selected}")
    with open(run_dir / "module_visual_audit_details.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Run module_sim once and validate visual/log artifacts.")
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--logs_dir", default=str(SCRIPT_DIR / "navi-visual" / "logs" / "module_sim_check"))
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()

    if args.skip_run:
        if not args.run_dir:
            raise SystemExit("--skip_run requires --run_dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        _require_single_cuda_device()
        _require_real_run_env(Path(args.logs_dir).expanduser())
        cmd = [
            sys.executable,
            str(MODULE_SIM),
            "--scene_name",
            str(args.scene_name),
            "--episode_id",
            str(args.episode_id),
            "--navigation_type",
            str(args.navigation_type),
            "--instance_id",
            str(args.instance_id),
            "--task_id",
            str(args.task_id),
            "--max_rounds",
            str(args.max_rounds),
            "--segment_advance_m",
            str(args.segment_advance_m),
            "--logs_dir",
            str(Path(args.logs_dir).expanduser()),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")
        env.setdefault("MAGNUM_LOG", "quiet")
        env.setdefault("HABITAT_SIM_LOG", "quiet")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout)
        if proc.returncode != 0:
            if "no EGL devices found" in proc.stdout or "Unable to create windowless context" in proc.stdout:
                print(
                    "[check] Habitat-Sim could not create an EGL context. "
                    "The code compiled, but this machine/session has no working NVIDIA/EGL device; "
                    "rerun in the Habitat GPU environment or use --skip_run --run_dir to audit existing logs.",
                    file=sys.stderr,
                )
            print(f"[check] module_sim failed with exit code {proc.returncode}", file=sys.stderr)
            return proc.returncode
        run_dir = _find_run_dir(proc.stdout)

    errors = validate_run(run_dir)
    report = {"run_dir": str(run_dir), "ok": len(errors) == 0, "errors": errors}
    report_path = run_dir / "module_visual_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if errors:
        print(f"[check] FAILED: {len(errors)} issue(s). Report: {report_path}")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"[check] OK: {run_dir}")
    print(f"[check] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
