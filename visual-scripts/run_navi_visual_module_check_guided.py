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
MODULE_SIM = SCRIPT_DIR / "navi-visual" / "code" / "module_sim_guided.py"
EVIDENCE_MODULE = "Evidence Grounding"
ENTITY_MODULE = "Entity Grounding"
ENDPOINT_MODULE = "Endpoint Grounding"
ARTIFACT_ROOT_NAME = "artifacts"
EVIDENCE_ARTIFACT_DIR = "EvidenceGrounding"
ENTITY_ARTIFACT_DIR = "EntityGrounding"
ENDPOINT_ARTIFACT_DIR = "EndpointGrounding"
EVIDENCE_DECISION_JSON = "evidence_decision.json"
EVIDENCE_FRONTIERS_TOPDOWN = "evidence_frontiers_topdown.png"
ENTITY_DECISION_JSON = "entity_decision.json"
ENTITY_CLUSTERS_TOPDOWN = "entity_clusters_topdown.png"
ENDPOINT_DECISION_JSON = "endpoint_decision.json"
ENDPOINT_TARGET_RGB_RAW = "endpoint_target_rgb_raw.png"
ENDPOINT_CANDIDATES_TOPDOWN = "endpoint_candidates_topdown.png"
ENDPOINT_CANDIDATES_ZOOM = "endpoint_candidates_zoom.png"
ENDPOINT_TARGET_RGB_POINTS = "endpoint_target_rgb_points.png"

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
    "topdown_cam_rgb_raw.png",
    "route_start_to_goal_legend.png",
}
FORBIDDEN_IMAGE_SUFFIXES = ("_legend.png",)
FORBIDDEN_DIR_NAMES = {"topdown_maps", "topdown_rgb", "frontiers", "exploration_maps"}


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


def _audit_topdown_pair(errors: List[str], mod_dir: Path, module_name: str) -> Dict[str, Any]:
    paths = {
        "topdown_map": mod_dir / "topdown_map.png",
        "topdown_scene_rgb": mod_dir / "topdown_scene_rgb.png",
        "topdown_scene_rgb_annotated": mod_dir / "topdown_scene_rgb_annotated.png",
    }
    for key, path in paths.items():
        _require_image(errors, path, f"{module_name} {key}")
        if path.exists():
            _append_image_quality_errors(
                errors,
                path,
                f"{module_name} {key}",
                allow_discrete=key == "topdown_map",
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
    map_info_path = mod_dir / "topdown_map_info.json"
    _require_file(errors, map_info_path, f"{module_name} topdown map info")
    map_info = _read_json(map_info_path) if map_info_path.exists() else {}
    if map_info and map_info.get("source") != "single_gray_topdown_theme":
        errors.append(f"{module_name} topdown_map source should be the single gray theme: {map_info}")

    info_path = mod_dir / "topdown_scene_rgb_info.json"
    _require_file(errors, info_path, f"{module_name} scene RGB info")
    info = _read_json(info_path) if info_path.exists() else {}
    if info.get("ok"):
        if info.get("source") != "global_topdown_rgb_tile_mosaic":
            errors.append(f"{module_name} scene RGB source is not RGBD tile mosaic: {info.get('source')} in {mod_dir}")
        fill = float(info.get("filled_navigable_ratio", 0.0) or 0.0)
        if fill < 0.35:
            errors.append(f"{module_name} scene RGB has low navigable fill: {fill:.3f} in {mod_dir}")
        tile_count = int(info.get("tile_count", 0) or 0)
        ok_tiles = int(info.get("successful_tile_count", 0) or 0)
        if tile_count <= 0 or ok_tiles / max(tile_count, 1) < 0.80:
            errors.append(f"{module_name} scene RGB tile success too low: {ok_tiles}/{tile_count} in {mod_dir}")
        for fill_key in ("fallback_filled_px", "dilation_filled_px", "inpainted_uncovered_px"):
            if int(info.get(fill_key, 0) or 0) != 0:
                errors.append(f"{module_name} scene RGB should not use blur/ghost filling: {fill_key}={info.get(fill_key)}")
        paste_mode = str(info.get("paste_mode", ""))
        if "rgbd_projected" not in paste_mode or "no_patch_fallback" not in paste_mode:
            errors.append(f"{module_name} scene RGB paste_mode is not the clean projected mode: {paste_mode}")
    else:
        errors.append(f"{module_name} scene RGB did not run ok in {mod_dir}: {info}")
    scene_path = mod_dir / "topdown_scene_rgb.png"
    if scene_path.exists():
        edge = _edge_ratios(scene_path)
        if edge["black"] > 0.03 or edge["white"] > 0.03:
            errors.append(f"{module_name} scene RGB has black/white border ratio {edge}: {scene_path}")
        dark_ratio = _dark_rgb_ratio(scene_path)
        if dark_ratio > 0.10:
            errors.append(f"{module_name} scene RGB still has large dark background ratio: {dark_ratio:.3f} in {scene_path}")
    annotated_info_path = mod_dir / "topdown_scene_rgb_annotated_info.json"
    _require_file(errors, annotated_info_path, f"{module_name} annotated scene RGB info")
    annotated_info = _read_json(annotated_info_path) if annotated_info_path.exists() else {}
    if annotated_info and annotated_info.get("source") != "global_topdown_rgb_annotation":
        errors.append(f"{module_name} annotated scene RGB source is wrong: {annotated_info}")
    if annotated_info and annotated_info.get("base_image") != "topdown_scene_rgb.png":
        errors.append(f"{module_name} annotated scene RGB should preserve clean base image: {annotated_info}")
    return {"shapes": shapes, "topdown_map": map_info, "scene_rgb": info}


def _audit_local_topdown_cam(errors: List[str], mod_dir: Path, module_name: str) -> Dict[str, Any]:
    rgb_path = mod_dir / "topdown_cam_rgb.png"
    depth_path = mod_dir / "topdown_cam_depth.png"
    info_path = mod_dir / "topdown_cam_info.json"
    _require_image(errors, rgb_path, f"{module_name} local topdown rgb")
    _require_image(errors, depth_path, f"{module_name} local topdown depth")
    _require_file(errors, info_path, f"{module_name} local topdown info")
    info = _read_json(info_path) if info_path.exists() else {}
    edge = {}
    if rgb_path.exists():
        edge = _edge_ratios(rgb_path)
        if edge["black"] > 0.05 or edge["white"] > 0.05:
            errors.append(f"{module_name} local topdown RGB has black/white edge ratio {edge}: {rgb_path}")
    return {"info": info, "edge": edge}


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


def _audit_log_streams(errors: List[str], run_dir: Path, rounds: int) -> Dict[str, Any]:
    required = {
        "evidence_grounding",
        "entity_grounding",
        "endpoint_grounding",
    }
    manifest_path = run_dir / "log_streams" / "manifest.json"
    _require_file(errors, manifest_path, "log stream manifest")
    audit: Dict[str, Any] = {}
    if not manifest_path.exists():
        return audit
    manifest = _read_json(manifest_path)
    cats = manifest.get("categories", {}) if isinstance(manifest.get("categories", {}), dict) else {}
    missing = sorted(required.difference(cats.keys()))
    if missing:
        errors.append(f"log stream manifest missing categories: {missing}")
    for name in sorted(required):
        rel_index = str((cats.get(name, {}) or {}).get("index", f"log_streams/{name}/index.json"))
        index_path = run_dir / rel_index
        _require_file(errors, index_path, f"log stream {name} index")
        idx = _read_json(index_path) if index_path.exists() else {}
        entries = idx.get("entries", []) if isinstance(idx.get("entries", []), list) else []
        audit[name] = {"entry_count": len(entries), "index": rel_index}
        if len(entries) == 0:
            errors.append(f"log stream {name} has no entries: {index_path}")
        if name == "evidence_grounding" and int(rounds) > 0:
            for entry in entries:
                if not entry.get("decision_json") or not entry.get("panorama_inputs"):
                    errors.append(f"Evidence Grounding entry missing decision/panorama inputs: {entry}")
                if not entry.get("prompts") or not entry.get("responses"):
                    errors.append(f"Evidence Grounding entry missing prompt/response logs: {entry}")
                if not entry.get("clean_rgb") or not entry.get("annotated_rgb"):
                    errors.append(f"Evidence Grounding entry must keep clean and annotated RGB: {entry}")
        if name == "entity_grounding" and int(rounds) > 0:
            for entry in entries:
                if not entry.get("images") or not entry.get("decision_json"):
                    errors.append(f"Entity Grounding entry missing image/decision json: {entry}")
                if not entry.get("prompt") or not entry.get("response"):
                    errors.append(f"Entity Grounding entry missing linked prompt/response: {entry}")
                if not entry.get("clean_rgb") or not entry.get("annotated_rgb"):
                    errors.append(f"Entity Grounding entry must keep clean and annotated RGB: {entry}")
        if name == "endpoint_grounding":
            for entry in entries:
                if not entry.get("decision_json") or not entry.get("panorama_inputs"):
                    errors.append(f"Endpoint Grounding entry missing decision/panorama inputs: {entry}")
                if not entry.get("images") or not entry.get("target_rgb"):
                    errors.append(f"Endpoint Grounding entry missing candidate/target images: {entry}")
                if not entry.get("clean_rgb") or not entry.get("annotated_rgb"):
                    errors.append(f"Endpoint Grounding entry must keep clean and annotated RGB: {entry}")
    return audit


def _audit_guided_frame_camera(errors: List[str], run_dir: Path, max_down_tilt_deg: float = 8.0) -> Dict[str, Any]:
    checked = 0
    missing_metric = 0
    max_seen = 0.0
    bad: List[Dict[str, Any]] = []
    guided_root = run_dir / "guided_frames"
    for idx_path in sorted(guided_root.glob("dec_*/frames_index.json")):
        idx = _read_json(idx_path)
        frames = idx.get("frames", []) if isinstance(idx.get("frames", []), list) else []
        for row in frames:
            checked += 1
            value = row.get("color_sensor_down_tilt_deg")
            if value is None:
                missing_metric += 1
                continue
            try:
                tilt = float(value)
            except Exception:
                missing_metric += 1
                continue
            max_seen = max(max_seen, tilt)
            if tilt > float(max_down_tilt_deg):
                bad.append(
                    {
                        "index": str(idx_path),
                        "frame": row.get("frame"),
                        "goal_index": row.get("goal_index"),
                        "round": row.get("round"),
                        "step_count": row.get("step_count"),
                        "rgb": row.get("rgb"),
                        "color_sensor_down_tilt_deg": tilt,
                    }
                )
    if checked <= 0:
        errors.append(f"guided frame camera audit found no navigation frames under {guided_root}")
    if missing_metric > 0:
        errors.append(f"guided frame camera audit missing color_sensor_down_tilt_deg on {missing_metric}/{checked} frames")
    if bad:
        errors.append(
            f"guided navigation camera looks down in {len(bad)} frame(s); "
            f"max_seen={max_seen:.2f}deg allowed={float(max_down_tilt_deg):.2f}deg"
        )
    return {
        "checked_frames": int(checked),
        "missing_metric_frames": int(missing_metric),
        "max_allowed_down_tilt_deg": float(max_down_tilt_deg),
        "max_seen_down_tilt_deg": float(max_seen),
        "bad_frame_count": len(bad),
        "bad_frames": bad[:20],
    }


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
    _append_forbidden_artifact_errors(errors, run_dir)
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
    if not bool(summary.get("guided_visualization_mode", False)):
        errors.append(f"guided summary marker missing/false: {summary_path}")
    guided_route_path = run_dir / "guided_route.json"
    guided_moves_path = run_dir / "guided_moves.json"
    _require_file(errors, guided_route_path, "guided route json")
    _require_file(errors, guided_moves_path, "guided moves json")
    guided_route = _read_json(guided_route_path) if guided_route_path.exists() else {}
    if guided_route and str(guided_route.get("mode", "")) != "guided_visualization_oracle_route":
        errors.append(f"guided route mode is wrong: {guided_route.get('mode')}")
    waypoints = guided_route.get("decision_move_waypoints", []) if isinstance(guided_route, dict) else []
    if bool(summary.get("multi_goal_sequence", False)):
        moves = summary.get("guided_moves", []) if isinstance(summary.get("guided_moves", []), list) else []
        if rounds > 0 and len(moves) < rounds:
            errors.append(f"multi-goal guided moves fewer than decision rounds: moves={len(moves)} rounds={rounds}")
    elif rounds > 0 and len(waypoints) < rounds:
        errors.append(f"guided route has fewer waypoints than decision rounds: waypoints={len(waypoints)} rounds={rounds}")
    if guided_moves_path.exists():
        try:
            with open(guided_moves_path, "r", encoding="utf-8") as f:
                guided_moves = json.load(f)
            if not isinstance(guided_moves, list):
                errors.append(f"guided moves json should be a list: {guided_moves_path}")
        except Exception as exc:
            errors.append(f"cannot parse guided moves json: {guided_moves_path} ({type(exc).__name__}: {exc})")
    audit["log_streams"] = _audit_log_streams(errors, run_dir, rounds)
    audit["navigation_camera_level"] = _audit_guided_frame_camera(errors, run_dir)

    _require_image(errors, run_dir / "trajectory" / "route_start_to_goal.png", "trajectory")

    for dec in range(max(1, rounds)):
        dtag = f"dec_{dec:03d}"
        evidence_dir = run_dir / ARTIFACT_ROOT_NAME / EVIDENCE_ARTIFACT_DIR / dtag
        entity_dir = run_dir / ARTIFACT_ROOT_NAME / ENTITY_ARTIFACT_DIR / dtag
        for module_name, mod_dir in ((EVIDENCE_MODULE, evidence_dir), (ENTITY_MODULE, entity_dir)):
            audit_key = f"{module_name}/{dtag}"
            audit["modules"][audit_key] = _audit_topdown_pair(errors, mod_dir, audit_key)
            audit["modules"][audit_key]["local_topdown_cam"] = _audit_local_topdown_cam(
                errors, mod_dir, audit_key
            )

        evidence_json = evidence_dir / EVIDENCE_DECISION_JSON
        if evidence_json.exists():
            tj = _read_json(evidence_json)
            if str(tj.get("source_code_verified", "")) != "anchor_nav Evidence Grounding implementation":
                errors.append(f"Evidence Grounding decision missing implementation verification marker: {evidence_json}")
            if not bool(tj.get("guided_visualization_mode", False)) or not bool(tj.get("fake_frontier_selection", False)):
                errors.append(f"Evidence Grounding decision is not marked as guided fake frontier selection: {evidence_json}")
            if not bool(tj.get("skipped", False)):
                if bool(tj.get("vlm_interval_allowed", True)) and int(tj.get("vlm_call_count", 0) or 0) <= 0:
                    errors.append(f"Evidence Grounding interval allowed but no VLM calls recorded: {evidence_json}")
                for score in list(tj.get("vlm_scores", [])):
                    fi = int(score.get("frontier_index", -1))
                    if fi >= 0:
                        _require_file(errors, evidence_dir / "prompts" / f"frontier_{fi:02d}_prompt.txt", "Evidence Grounding prompt")
                        response_path = evidence_dir / "prompts" / f"frontier_{fi:02d}_response.txt"
                        _require_file(errors, response_path, "Evidence Grounding response")
                        if response_path.exists() and response_path.read_text(encoding="utf-8", errors="replace").strip() == "":
                            errors.append(f"empty Evidence Grounding response log: {response_path}")
                _require_image(errors, evidence_dir / EVIDENCE_FRONTIERS_TOPDOWN, "Evidence Grounding frontier topdown overlay")
        else:
            _require_file(errors, evidence_json, "Evidence Grounding decision json")

        _require_file(errors, entity_dir / ENTITY_DECISION_JSON, "Entity Grounding decision json")
        _require_file(errors, entity_dir / "decomposition_prompt.txt", "Entity Grounding decomposition prompt")
        _require_file(errors, entity_dir / "decomposition_response.txt", "Entity Grounding decomposition response")
        mj = _read_json(entity_dir / ENTITY_DECISION_JSON) if (entity_dir / ENTITY_DECISION_JSON).exists() else {}
        vlm_rec = (mj.get("decomposition", {}) or {}).get("vlm", {}) if isinstance(mj.get("decomposition", {}), dict) else {}
        if vlm_rec and vlm_rec.get("source") != "anchor_nav_vlm_real":
            errors.append(f"Entity Grounding decomposition did not use a real VLM response: source={vlm_rec.get('source')} file={entity_dir / ENTITY_DECISION_JSON}")
        if mj and (not bool(mj.get("guided_visualization_mode", False)) or not bool(mj.get("fake_object_selection", False))):
            errors.append(f"Entity Grounding decision is not marked as guided fake object selection: {entity_dir / ENTITY_DECISION_JSON}")
        _require_image(errors, entity_dir / ENTITY_CLUSTERS_TOPDOWN, "Entity Grounding cluster topdown overlay")
        _require_image(errors, evidence_dir / "panorama" / "current_decision_panorama_vfv_order.jpg", "Evidence Grounding VFV-order panorama")

    endpoint_dir = run_dir / ARTIFACT_ROOT_NAME / ENDPOINT_ARTIFACT_DIR / "final"
    final_pano = endpoint_dir / "panorama"
    _require_image(errors, final_pano / "current_decision_panorama_vfv_order.jpg", "Endpoint Grounding final panorama")
    for i in range(12):
        _require_image(errors, final_pano / f"view_{i:02d}.png", f"Endpoint Grounding final panorama view {i:02d}")
    _require_file(errors, endpoint_dir / ENDPOINT_DECISION_JSON, "Endpoint Grounding decision json")
    audit["modules"][f"{ENDPOINT_MODULE}/final"] = _audit_topdown_pair(errors, endpoint_dir, f"{ENDPOINT_MODULE}/final")
    audit["modules"][f"{ENDPOINT_MODULE}/final"]["local_topdown_cam"] = _audit_local_topdown_cam(errors, endpoint_dir, f"{ENDPOINT_MODULE}/final")
    _require_image(errors, endpoint_dir / ENDPOINT_TARGET_RGB_RAW, "Endpoint Grounding target rgb")
    _require_image(errors, endpoint_dir / ENDPOINT_CANDIDATES_TOPDOWN, "Endpoint Grounding candidates topdown overlay")
    _require_image(errors, endpoint_dir / ENDPOINT_CANDIDATES_ZOOM, "Endpoint Grounding candidates zoom overlay")
    _require_image(errors, endpoint_dir / ENDPOINT_TARGET_RGB_POINTS, "Endpoint Grounding target rgb point overlay")
    vj = _read_json(endpoint_dir / ENDPOINT_DECISION_JSON) if (endpoint_dir / ENDPOINT_DECISION_JSON).exists() else {}
    target_rgb = vj.get("target_rgb", {}) if isinstance(vj.get("target_rgb", {}), dict) else {}
    selected = vj.get("selected_viewpoint")
    if isinstance(selected, dict):
        if not bool(selected.get("reachable", False)) or not bool(selected.get("same_island", False)):
            errors.append(f"Endpoint Grounding selected viewpoint is not reachable/same island: {selected}")
        if str(selected.get("category", "")) != "feasible":
            errors.append(f"Endpoint Grounding selected viewpoint category is not feasible: {selected}")
    with open(run_dir / "module_visual_audit_details.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Run guided module_sim once and validate visual/log artifacts.")
    ap.add_argument("--scene_name", default="00844-q5QZSEeHe5g")
    ap.add_argument("--episode_id", type=int, default=122)
    ap.add_argument("--navigation_type", default="instance")
    ap.add_argument("--instance_id", default="armchair_906")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--logs_dir", default=str(SCRIPT_DIR / "navi-visual" / "logs" / "module_sim_check"))
    ap.add_argument("--concise_description", action="store_true")
    ap.add_argument("--sequence_task_count", type=int, default=1)
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
        if bool(args.concise_description):
            cmd.append("--concise_description")
        if int(args.sequence_task_count) > 1:
            cmd.extend(["--sequence_task_count", str(int(args.sequence_task_count))])
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
