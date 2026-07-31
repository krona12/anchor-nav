#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIRECT_SH = SCRIPT_DIR / "navi-visual" / "command" / "run_navi_visual_direct.sh"

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
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _require_single_cuda_device() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to one device")
    if "," in visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must name exactly one device, got {visible!r}")


def _require_real_run_env(logs_dir: Path) -> None:
    if any("smoke" in part.lower() for part in logs_dir.parts):
        raise RuntimeError(f"smoke output paths are not allowed for a real direct check: {logs_dir}")
    bad_env: List[str] = []
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name in DISALLOWED_MOCK_ENV_NAMES and str(value).strip() not in ("", "0", "false", "False"):
            bad_env.append(f"{name}={value}")
    if bad_env:
        raise RuntimeError("mock/fake simulator environment is not allowed: " + ", ".join(sorted(bad_env)))


def _latest_run_dir(logs_dir: Path) -> Path | None:
    runs = sorted([p for p in logs_dir.glob("run=*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _find_run_dir(output: str, logs_dir: Path) -> Path:
    matches = re.findall(r"outputs under:\s*(.+)", output)
    if matches:
        return Path(matches[-1].strip()).expanduser().resolve()
    latest = _latest_run_dir(logs_dir)
    if latest is None:
        raise RuntimeError(f"could not find direct output run directory under {logs_dir}")
    return latest.resolve()


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image: {path}")
    return img


def _image_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        img = _load_image(path)
    except Exception:
        return False
    return bool(np.asarray(img).std() > 0.5)


def _require_file(errors: List[str], path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        errors.append(f"missing/empty {label}: {path}")


def _require_image(errors: List[str], path: Path, label: str) -> None:
    if not _image_ok(path):
        errors.append(f"bad image {label}: {path}")


def _image_stats(path: Path) -> Dict[str, Any]:
    img = _load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "shape": list(img.shape),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_ratio": float((gray < 8).mean()),
        "bright_ratio": float((gray > 247).mean()),
        "unique_sample": int(len(np.unique(img.reshape(-1, 3)[:: max(1, img.reshape(-1, 3).shape[0] // 4096)], axis=0))),
    }


def _edge_ratios(path: Path, border_frac: float = 0.04) -> Dict[str, float]:
    img = _load_image(path)
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


def _audit_topdown_set(errors: List[str], directory: Path, label: str) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    info = _read_json(directory / "topdown_scene_rgb_info.json")
    for name, allow_discrete in (
        ("topdown_map.png", True),
        ("topdown_scene_rgb.png", False),
        ("topdown_scene_rgb_annotated.png", False),
        ("topdown_cam_rgb.png", False),
        ("topdown_cam_depth.png", True),
    ):
        path = directory / name
        _require_image(errors, path, f"{label} {name}")
        if path.exists():
            _append_image_quality_errors(errors, path, f"{label} {name}", allow_discrete=allow_discrete)
            audit[name] = _image_stats(path)
    scene_path = directory / "topdown_scene_rgb.png"
    if scene_path.exists():
        edge = _edge_ratios(scene_path)
        audit["topdown_scene_rgb_edge"] = edge
        if edge["black"] > 0.03 or edge["white"] > 0.03:
            errors.append(f"{label} scene RGB has black/white border ratio {edge}: {scene_path}")
    for info_name in (
        "topdown_map_info.json",
        "topdown_scene_rgb_info.json",
        "topdown_scene_rgb_annotated_info.json",
        "topdown_cam_info.json",
    ):
        _require_file(errors, directory / info_name, f"{label} {info_name}")
    if info:
        if info.get("source") != "global_topdown_rgb_tile_mosaic":
            errors.append(f"{label} topdown_scene_rgb must be real global RGB mosaic, got source={info.get('source')}")
        if info.get("semantic_type") != "global_scene_rgb_overhead_photo_mosaic":
            errors.append(f"{label} topdown_scene_rgb semantic_type is wrong: {info.get('semantic_type')}")
        if int(info.get("fallback_filled_px", 0) or 0) != 0 or int(info.get("dilation_filled_px", 0) or 0) != 0:
            errors.append(f"{label} topdown_scene_rgb used forbidden blur/fallback filling: {info}")
        try:
            filled = float(info.get("filled_navigable_ratio", 0.0) or 0.0)
        except Exception:
            filled = 0.0
        if filled <= 0.01:
            errors.append(f"{label} topdown_scene_rgb has too little RGB coverage: filled_navigable_ratio={filled}")
    ann_info = _read_json(directory / "topdown_scene_rgb_annotated_info.json")
    if ann_info:
        if ann_info.get("source") != "global_topdown_rgb_annotation":
            errors.append(f"{label} annotated scene RGB source is wrong: {ann_info.get('source')}")
        if ann_info.get("semantic_type") != "global_scene_rgb_overhead_photo_mosaic_with_navigation_overlays":
            errors.append(f"{label} annotated scene RGB semantic_type is wrong: {ann_info.get('semantic_type')}")
        if int(ann_info.get("goal_marker_count", 0) or 0) != 1:
            errors.append(f"{label} annotated scene RGB should have exactly one goal star, got {ann_info.get('goal_marker_count')}")
    map_info = _read_json(directory / "topdown_map_info.json")
    if map_info and int(map_info.get("goal_marker_count", 0) or 0) != 1:
        errors.append(f"{label} topdown_map should have exactly one goal star, got {map_info.get('goal_marker_count')}")
    cam_info = _read_json(directory / "topdown_cam_info.json")
    if cam_info and cam_info.get("source") != "robot_center_downward_rgbd_camera":
        errors.append(f"{label} topdown_cam_rgb source is wrong: {cam_info.get('source')}")
    return audit


def _audit_target_facing_rgb(errors: List[str], directory: Path, label: str) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    path = directory / "target_facing_rgb.png"
    info_path = directory / "target_facing_rgb_info.json"
    _require_image(errors, path, f"{label} target_facing_rgb.png")
    if path.exists():
        _append_image_quality_errors(errors, path, f"{label} target_facing_rgb.png", allow_discrete=False)
        audit["target_facing_rgb.png"] = _image_stats(path)
    _require_file(errors, info_path, f"{label} target_facing_rgb_info.json")
    info = _read_json(info_path) if info_path.exists() else {}
    if info and info.get("semantic_type") != "final_target_facing_rgb_photo":
        errors.append(f"{label} target_facing_rgb semantic_type is wrong: {info.get('semantic_type')}")
    if info and info.get("source") != "final_agent_color_sensor_look_at_goal_object":
        errors.append(f"{label} target_facing_rgb source is wrong: {info.get('source')}")
    audit["target_facing_rgb_info"] = info
    return audit


def _audit_guided_frames(errors: List[str], run_dir: Path) -> Dict[str, Any]:
    checked = 0
    max_tilt = 0.0
    missing = 0
    for idx_path in sorted((run_dir / "guided_frames").glob("dec_*/frames_index.json")):
        idx = _read_json(idx_path)
        frames = idx.get("frames", []) if isinstance(idx.get("frames", []), list) else []
        if not frames:
            errors.append(f"empty guided frame index: {idx_path}")
        for row in frames:
            checked += 1
            rgb = idx_path.parent / str(row.get("rgb", ""))
            top = idx_path.parent / str(row.get("topdown", ""))
            _require_image(errors, rgb, f"guided frame rgb {idx_path}")
            _require_image(errors, top, f"guided frame topdown {idx_path}")
            value = row.get("color_sensor_down_tilt_deg")
            if value is None:
                missing += 1
            else:
                try:
                    max_tilt = max(max_tilt, float(value))
                except Exception:
                    missing += 1
    if checked <= 0:
        errors.append(f"no guided navigation frames under {run_dir / 'guided_frames'}")
    if missing:
        errors.append(f"guided frame camera audit missing tilt metric on {missing}/{checked} frames")
    if max_tilt > 8.0:
        errors.append(f"guided navigation camera looks down too much: max={max_tilt:.2f}deg")
    return {"checked_frames": checked, "missing_tilt_metric": missing, "max_down_tilt_deg": max_tilt}


def _append_forbidden_module_errors(errors: List[str], run_dir: Path) -> None:
    forbidden_parts = {"EvidenceGrounding", "EntityGrounding", "EndpointGrounding", "tffs", "mqsc", "mqsc_r1", "vista_ls"}
    forbidden_names = {
        "evidence_decision.json",
        "entity_decision.json",
        "endpoint_decision.json",
        "decomposition_prompt.txt",
        "decomposition_response.txt",
        "tffs_decision.json",
        "mqsc_r1_decision.json",
        "vista_ls_decision.json",
    }
    for path in run_dir.rglob("*"):
        if any(part in forbidden_parts for part in path.parts):
            errors.append(f"forbidden module artifact path exists in direct run: {path}")
            continue
        if path.is_file() and path.name in forbidden_names:
            errors.append(f"forbidden module/VLM file exists in direct run: {path}")


def _planar_dist_from_json(a: Any, b: Any) -> float | None:
    try:
        aa = np.asarray(a, dtype=float).reshape(3)
        bb = np.asarray(b, dtype=float).reshape(3)
        return float(np.linalg.norm((aa - bb)[[0, 2]]))
    except Exception:
        return None


def validate_run(run_dir: Path) -> List[str]:
    errors: List[str] = []
    audit: Dict[str, Any] = {"run_dir": str(run_dir), "topdown_sets": {}}
    _append_forbidden_module_errors(errors, run_dir)

    summary_path = run_dir / "direct_navigation_summary.json"
    _require_file(errors, summary_path, "direct navigation summary")
    summary = _read_json(summary_path) if summary_path.exists() else {}
    if summary.get("direct_navigation_mode") is not True:
        errors.append(f"direct_navigation_mode marker missing/false: {summary_path}")
    if summary.get("no_vlm") is not True:
        errors.append(f"no_vlm marker missing/false: {summary_path}")
    if summary.get("module_artifacts_enabled") is not False:
        errors.append(f"module_artifacts_enabled should be false: {summary_path}")
    if summary.get("stop_reason") == "max_rounds_reached":
        errors.append(f"direct run stopped by max_rounds instead of reaching target navigation goal: {summary_path}")
    if summary.get("stop_reason") == "navigation_goal_unreached":
        errors.append(f"direct run did not reach target navigation goal: {summary_path}")

    route = summary.get("guided_route", {}) if isinstance(summary.get("guided_route", {}), dict) else {}
    if route.get("shortest_path_ok") is not True:
        errors.append(f"direct route shortest_path_ok should be true: {run_dir}")
    goal_selection = summary.get("goal_selection", {}) if isinstance(summary.get("goal_selection", {}), dict) else {}
    if goal_selection and int(goal_selection.get("reachable_candidate_count", 0) or 0) <= 0:
        errors.append(f"goal selection found no reachable candidate: {run_dir}")
    nav_goal = summary.get("goal_navigation_position") or route.get("direct_goal_nav_xyz") or route.get("snapped_goal_xyz")
    end_pos = summary.get("end_position")
    final_nav_dist = summary.get("final_planar_distance_to_navigation_goal_m")
    if final_nav_dist is None:
        final_nav_dist = _planar_dist_from_json(end_pos, nav_goal)
    try:
        final_nav_dist_f = float(final_nav_dist)
    except Exception:
        final_nav_dist_f = float("inf")
    arrive_thresh = summary.get("direct_arrive_thresh_m", 0.35)
    try:
        arrive_thresh_f = float(arrive_thresh)
    except Exception:
        arrive_thresh_f = 0.35
    max_allowed_nav_dist = max(0.45, arrive_thresh_f + 0.10)
    audit["final_planar_distance_to_navigation_goal_m"] = final_nav_dist_f
    audit["max_allowed_navigation_goal_distance_m"] = max_allowed_nav_dist
    if not np.isfinite(final_nav_dist_f) or final_nav_dist_f > max_allowed_nav_dist:
        errors.append(
            f"final position is too far from target navigation goal: dist={final_nav_dist_f:.3f}m "
            f"allowed={max_allowed_nav_dist:.3f}m run={run_dir}"
        )

    rounds = int(summary.get("decision_rounds", 0) or 0)
    if rounds <= 0:
        errors.append(f"expected at least one baseline decision round, got {rounds}")
    _require_file(errors, run_dir / "guided_route.json", "direct route json")
    _require_file(errors, run_dir / "guided_moves.json", "direct moves json")
    moves = []
    if (run_dir / "guided_moves.json").exists():
        try:
            with (run_dir / "guided_moves.json").open("r", encoding="utf-8") as f:
                moves = json.load(f)
        except Exception as exc:
            errors.append(f"cannot parse guided_moves.json: {type(exc).__name__}: {exc}")
    if not isinstance(moves, list) or len(moves) < rounds:
        errors.append(f"guided_moves should contain at least decision_rounds moves: moves={len(moves) if isinstance(moves, list) else 'bad'} rounds={rounds}")
    if isinstance(moves, list):
        teleports = [m for m in moves if isinstance(m, dict) and bool(m.get("teleported_to_waypoint"))]
        if teleports:
            errors.append(f"direct navigation should not teleport to waypoints; teleported moves={len(teleports)} run={run_dir}")

    manifest_path = run_dir / "log_streams" / "manifest.json"
    _require_file(errors, manifest_path, "log stream manifest")
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    cats = manifest.get("categories", {}) if isinstance(manifest.get("categories", {}), dict) else {}
    for cat in ("baseline_decision_pause_scan", "topdown_map_process", "rgb_map_process", "direct_navigation"):
        if cat not in cats:
            errors.append(f"log stream manifest missing direct category: {cat}")
        else:
            idx = run_dir / str(cats[cat].get("index", ""))
            _require_file(errors, idx, f"log stream {cat} index")

    for dec in range(max(1, rounds)):
        dtag = f"dec_{dec:03d}"
        dec_dir = run_dir / "baseline_decisions" / dtag
        _require_file(errors, dec_dir / "decision.json", f"{dtag} decision json")
        decision = _read_json(dec_dir / "decision.json") if (dec_dir / "decision.json").exists() else {}
        if decision and (decision.get("no_vlm") is not True or decision.get("no_module_grounding") is not True):
            errors.append(f"{dtag} decision should be no_vlm/no_module_grounding: {dec_dir / 'decision.json'}")
        if dec == rounds - 1 and decision:
            last_nav_dist = decision.get("distance_to_navigation_goal_at_decision_m")
            if last_nav_dist is None:
                last_nav_dist = _planar_dist_from_json(decision.get("decision_position_xyz"), nav_goal)
            try:
                last_nav_dist_f = float(last_nav_dist)
            except Exception:
                last_nav_dist_f = float("inf")
            audit["last_decision_planar_distance_to_navigation_goal_m"] = last_nav_dist_f
            if not np.isfinite(last_nav_dist_f) or last_nav_dist_f > max_allowed_nav_dist:
                errors.append(
                    f"last decision viewpoint is too far from target navigation goal: "
                    f"{dtag} dist={last_nav_dist_f:.3f}m allowed={max_allowed_nav_dist:.3f}m"
                )
        pano = dec_dir / "panorama"
        for i in range(12):
            _require_image(errors, pano / f"view_{i:02d}.png", f"{dtag} panorama view {i:02d}")
        _require_image(errors, pano / "current_decision_panorama_vfv_order.jpg", f"{dtag} stitched panorama")
        audit["topdown_sets"][dtag] = _audit_topdown_set(errors, dec_dir, dtag)

    final_dir = run_dir / "final"
    final_pano = final_dir / "panorama"
    for i in range(12):
        _require_image(errors, final_pano / f"view_{i:02d}.png", f"final panorama view {i:02d}")
    _require_image(errors, final_pano / "current_decision_panorama_vfv_order.jpg", "final stitched panorama")
    audit["topdown_sets"]["final"] = _audit_topdown_set(errors, final_dir, "final")
    audit["target_facing_rgb"] = _audit_target_facing_rgb(errors, final_dir, "final")
    _require_image(errors, run_dir / "trajectory" / "route_start_to_goal.png", "trajectory route image")
    _require_file(errors, run_dir / "trajectory" / "route_start_to_goal_info.json", "trajectory route info")
    audit["navigation_camera_level"] = _audit_guided_frames(errors, run_dir)

    _write_json(run_dir / "direct_visual_audit_details.json", audit)
    return errors


def _write_reports(run_dir: Path, errors: List[str]) -> None:
    report = {
        "run_dir": str(run_dir),
        "ok": len(errors) == 0,
        "mode": "direct_baseline_no_vlm_no_modules",
        "errors": errors,
    }
    _write_json(run_dir / "direct_visual_check_report.json", report)
    _write_json(run_dir / "module_visual_check_report.json", report)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run direct baseline navigation once and validate visual/log artifacts.")
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--arrive_thresh_m", type=float, default=0.7)
    ap.add_argument("--sequence_task_count", type=int, default=1)
    ap.add_argument("--logs_dir", default=str(SCRIPT_DIR / "navi-visual" / "logs" / "direct_check"))
    ap.add_argument("--concise_description", action="store_true")
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()

    if args.skip_run:
        if not args.run_dir:
            raise SystemExit("--skip_run requires --run_dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        logs_dir = Path(args.logs_dir).expanduser().resolve()
        _require_single_cuda_device()
        _require_real_run_env(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "SCENE_NAME": str(args.scene_name),
                "EPISODE_ID": str(args.episode_id),
                "NAVIGATION_TYPE": str(args.navigation_type),
                "INSTANCE_ID": str(args.instance_id),
                "TASK_ID": str(args.task_id),
                "MAX_ROUNDS": str(args.max_rounds),
                "SEGMENT_ADVANCE_M": str(args.segment_advance_m),
                "ARRIVE_THRESH_M": str(args.arrive_thresh_m),
                "SEQUENCE_TASK_COUNT": str(args.sequence_task_count),
                "OUT_DIR": str(logs_dir),
                "USE_LOCAL_NVIDIA_580": os.environ.get("USE_LOCAL_NVIDIA_580", "0"),
            }
        )
        cmd = [str(DIRECT_SH), "concise" if bool(args.concise_description) else "detailed", "direct_check"]
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"[direct-check] direct run failed with exit code {proc.returncode}")
            return int(proc.returncode)
        run_dir = _find_run_dir(proc.stdout, logs_dir)

    errors = validate_run(run_dir)
    _write_reports(run_dir, errors)
    if errors:
        print(f"[direct-check] FAILED: {len(errors)} issue(s). Report: {run_dir / 'direct_visual_check_report.json'}")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"[direct-check] OK: {run_dir}")
    print(f"[direct-check] report: {run_dir / 'direct_visual_check_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
