#!/usr/bin/env python3
"""Generate fresh navigation logs, then render demo-style process videos.

This script deliberately does not consume historical visual output or old
navi-visual logs as video material.  By default it first runs the guided
navigation visual module to create a fresh multi-goal sequence log under
``visual-scripts/video-creator/nav-logs``.  The renderer then uses only that
fresh run directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CHECK_SH = PROJECT_ROOT / "visual-scripts" / "run_navi_visual_module_check_guided.sh"
OUTPUT_DIR = SCRIPT_DIR / "output"
NAV_LOG_DIR = SCRIPT_DIR / "nav-logs" / "video_creator"

CANVAS_W = 1920
CANVAS_H = 720
LEFT_W = 960
RIGHT_W = 960
FPS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
EVIDENCE_MODULE = "Evidence Grounding"
ENTITY_MODULE = "Entity Grounding"
ENDPOINT_MODULE = "Endpoint Grounding"
EVIDENCE_HOLD_FRAMES = 20
ENTITY_HOLD_FRAMES = 17
ENDPOINT_HOLD_FRAMES = 18

TASK_COLORS = [
    (45, 70, 255),     # red-ish BGR
    (70, 210, 90),     # green
    (255, 120, 40),    # blue/orange contrast on RGB map
    (0, 220, 255),     # yellow
    (230, 90, 230),
]

ARTIFACT_ROOT_NAME = "artifacts"
EVIDENCE_ARTIFACT_DIR = "EvidenceGrounding"
ENTITY_ARTIFACT_DIR = "EntityGrounding"
ENDPOINT_ARTIFACT_DIR = "EndpointGrounding"
EVIDENCE_OVERLAY_NAME = "evidence_frontiers_topdown.png"
EVIDENCE_DECISION_STEM = "evidence_decision"
ENTITY_OVERLAY_NAME = "entity_clusters_topdown.png"
ENTITY_DECISION_STEM = "entity_decision"
ENDPOINT_CANDIDATES_TOPDOWN_NAME = "endpoint_candidates_topdown.png"
ENDPOINT_CANDIDATES_ZOOM_NAME = "endpoint_candidates_zoom.png"
ENDPOINT_TARGET_RGB_POINTS_NAME = "endpoint_target_rgb_points.png"
ENDPOINT_TARGET_RGB_RAW_STEM = "endpoint_target_rgb_raw"

FORBIDDEN_SOURCE_MARKERS = (
    "visual" + "-output",
    "navi" + "-visual/logs",
    "key" + "_logs",
    "hm3d-online/output" + "_logs",
)

FORBIDDEN_PUBLIC_OUTPUT_MARKERS = FORBIDDEN_SOURCE_MARKERS
DRAFT_NAME_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"v[\W_]*i[\W_]*s[\W_]*t[\W_]*a(?:[\W_]*l[\W_]*s)?",
        r"m[\W_]*q[\W_]*s[\W_]*c(?:[\W_-]*r[\W_]*1)?",
        r"t[\W_]*f[\W_]*f[\W_]*s",
    )
)

EXPECTED_MODULE_STATUS = {
    EVIDENCE_MODULE: "[Evidence Grounding] Which frontier?",
    ENTITY_MODULE: "[Entity Grounding] Which object?",
    ENDPOINT_MODULE: "[Endpoint Grounding] Which viewpoint?",
}


@dataclass
class NavFrame:
    kind: str
    goal_index: int
    round_index: int
    step_count: int
    left_mode: str
    active_module: str
    status: str
    rgb_path: Optional[Path]
    topdown_path: Optional[Path]
    rgb_map_path: Optional[Path]
    module_image_paths: List[Path]
    frame_meta: Dict[str, Any]
    duration_frames: int = 1


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def rel_to_run(run_dir: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def public_source_name(run_dir: Path, path: Optional[Path]) -> str:
    return rel_to_run(run_dir, path)


def image_ok(path: Path) -> bool:
    if not path or not path.exists() or path.stat().st_size <= 0:
        return False
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return bool(img is not None and img.size > 0 and float(img.std()) > 1.0)


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image: {path}")
    return img


def fit_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = max(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return resized[y0:y0 + h, x0:x0 + w].copy()


def fit_contain(img: np.ndarray, w: int, h: int, bg: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.full((h, w, 3), bg, dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def edge_connected_mask(mask_in: np.ndarray) -> np.ndarray:
    mask_bin = (np.asarray(mask_in) > 0).astype(np.uint8)
    if mask_bin.size == 0 or int(mask_bin.sum()) == 0:
        return np.zeros_like(mask_bin, dtype=bool)
    h, w = mask_bin.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    work = mask_bin.copy()
    seeds: List[Tuple[int, int]] = []
    xs = np.where(work[0, :] > 0)[0]
    seeds.extend((int(x), 0) for x in xs)
    xs = np.where(work[h - 1, :] > 0)[0]
    seeds.extend((int(x), h - 1) for x in xs)
    ys = np.where(work[:, 0] > 0)[0]
    seeds.extend((0, int(y)) for y in ys)
    ys = np.where(work[:, w - 1] > 0)[0]
    seeds.extend((w - 1, int(y)) for y in ys)
    for seed in seeds:
        if work[seed[1], seed[0]] > 0:
            cv2.floodFill(work, flood_mask, seed, 2)
    return work == 2


def clean_camera_follow_bgr(img: np.ndarray) -> np.ndarray:
    out = np.asarray(img, dtype=np.uint8).copy()
    edge_black = edge_connected_mask(out.max(axis=2) <= 8)
    if not bool(np.any(edge_black)):
        return out
    visible = out[~edge_black]
    if visible.size >= 30:
        fill = np.median(visible.reshape(-1, 3), axis=0).astype(np.uint8)
        fill = np.maximum(fill, np.asarray([96, 96, 96], dtype=np.uint8))
    else:
        fill = np.asarray([184, 184, 184], dtype=np.uint8)
    out[edge_black] = fill.reshape(1, 3)
    return out


def draw_text(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    scale: float = 0.58,
    color: Tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel_label(img: np.ndarray, lines: Sequence[str]) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (LEFT_W, 76), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, dst=img)
    y = 28
    for line in lines[:3]:
        draw_text(img, line, (18, y), 0.62)
        y += 24


def color_for_goal(goal_index: int) -> Tuple[int, int, int]:
    return TASK_COLORS[int(goal_index) % len(TASK_COLORS)]


def draw_agent_marker(img: np.ndarray, rc: Sequence[int], heading_xz: Optional[Sequence[float]] = None) -> None:
    r, c = int(rc[0]), int(rc[1])
    cv2.circle(img, (c, r), 8, (0, 0, 255), -1, cv2.LINE_AA)
    if heading_xz and len(heading_xz) >= 2:
        v = np.asarray([float(heading_xz[0]), float(heading_xz[1])], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            v /= n
            tip = (int(round(c + v[0] * 18)), int(round(r + v[1] * 18)))
            cv2.line(img, (c, r), tip, (255, 255, 255), 3, cv2.LINE_AA)


def draw_rgb_map_overlay(base: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    out = base.copy()
    path = meta.get("path_pixels_rc") or []
    goal_index = int(meta.get("goal_index", 0) or 0)
    color = color_for_goal(goal_index)
    if len(path) >= 2:
        pts = np.asarray([[int(p[1]), int(p[0])] for p in path], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, color, 4, cv2.LINE_AA)
    if meta.get("agent_pixel_rc"):
        draw_agent_marker(out, meta["agent_pixel_rc"], meta.get("heading_xz"))
    return out


def make_contact_sheet(paths: Sequence[Path], selected: int = 0, title: str = "") -> np.ndarray:
    tile_w, tile_h = 300, 190
    cols, rows = 3, 3
    sheet = np.zeros((720, 960, 3), dtype=np.uint8)
    content_y = 104
    if title:
        draw_text(sheet, title, (22, 102), 0.68)
        content_y = 128
    for i, path in enumerate(paths[: cols * rows]):
        if not image_ok(path):
            continue
        usable_h = max(120, (720 - content_y - 12) // rows)
        img = fit_cover(load_bgr(path), tile_w - 14, usable_h - 16)
        row, col = divmod(i, cols)
        x = col * tile_w + 8
        y = content_y + row * usable_h
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        border = (70, 255, 70) if i == selected else (235, 235, 235)
        cv2.rectangle(sheet, (x, y), (x + img.shape[1], y + img.shape[0]), border, 3 if i == selected else 1)
        cv2.rectangle(sheet, (x, y), (x + 54, y + 28), (0, 0, 0), -1)
        draw_text(sheet, f"#{i}", (x + 8, y + 22), 0.58)
    return sheet


def make_module_page(run_dir: Path, frame: NavFrame) -> np.ndarray:
    module = frame.active_module
    if module == EVIDENCE_MODULE:
        return make_contact_sheet(frame.module_image_paths, selected=0)
    if module == ENTITY_MODULE:
        page = np.zeros((720, 960, 3), dtype=np.uint8)
        if frame.module_image_paths and image_ok(frame.module_image_paths[0]):
            img = fit_contain(load_bgr(frame.module_image_paths[0]), 620, 520, (0, 0, 0))
            page[104:624, 24:644] = img
        dec_path = (
            run_dir
            / ARTIFACT_ROOT_NAME
            / ENTITY_ARTIFACT_DIR
            / f"dec_{frame.round_index:03d}"
            / f"{ENTITY_DECISION_STEM}.json"
        )
        if dec_path.exists():
            data = read_json(dec_path)
            draw_text(page, f"selected object: {data.get('selected_object_index', '-')}", (680, 146), 0.56, (80, 255, 120))
            draw_text(page, f"baseline object: {data.get('baseline_object_index', '-')}", (680, 181), 0.48)
            draw_text(page, "landmark-anchored", (680, 226), 0.44)
            draw_text(page, "spatial voting", (680, 254), 0.44)
        return page
    if module == ENDPOINT_MODULE:
        page = np.zeros((720, 960, 3), dtype=np.uint8)
        imgs = [p for p in frame.module_image_paths if image_ok(p)]
        if imgs:
            page[104:624, 20:620] = fit_contain(load_bgr(imgs[0]), 600, 520, (0, 0, 0))
        if len(imgs) > 1:
            page[134:422, 650:938] = fit_contain(load_bgr(imgs[1]), 288, 288, (0, 0, 0))
        draw_text(page, "candidate ring", (650, 472), 0.54)
        draw_text(page, "reachable shell", (650, 506), 0.54)
        draw_text(page, "relation-verifiable stop", (650, 540), 0.48, (80, 255, 120))
        return page
    return np.zeros((720, 960, 3), dtype=np.uint8)


def find_latest_run(root: Path) -> Optional[Path]:
    runs = sorted([p for p in root.glob("run=*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def run_fresh_navigation(args: argparse.Namespace, logs_root: Path) -> Path:
    logs_root.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in logs_root.glob("run=*") if p.is_dir()}
    cmd = [
        str(CHECK_SH),
        "--scene_name", args.scene_name,
        "--episode_id", str(args.episode_id),
        "--navigation_type", "sequence",
        "--instance_id", f"sequence_ep{args.episode_id}_fresh_video",
        "--task_id", str(args.task_id),
        "--sequence_task_count", str(args.goal_count),
        "--concise_description",
        "--max_rounds", str(args.max_rounds_per_goal),
        "--segment_advance_m", str(args.segment_advance_m),
        "--logs_dir", str(logs_root),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (logs_root / "fresh_navigation_stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"fresh guided navigation failed with exit code {proc.returncode}")
    after = [p for p in logs_root.glob("run=*") if p.is_dir() and p.resolve() not in before]
    run_dir = sorted(after, key=lambda p: p.stat().st_mtime)[-1] if after else find_latest_run(logs_root)
    if run_dir is None:
        raise RuntimeError(f"fresh guided navigation did not create a run under {logs_root}")
    return run_dir.resolve()


def collect_timeline(run_dir: Path) -> Tuple[List[NavFrame], Dict[str, Any]]:
    summary = read_json(run_dir / "module_sim_summary.json")
    frames: List[NavFrame] = []
    goals = summary.get("goals", [])
    latest_meta: Dict[str, Any] = {}
    frame_index_paths = sorted((run_dir / "guided_frames").glob("dec_*/frames_index.json"))
    last_round_by_goal: Dict[int, int] = {}
    for idx_path in frame_index_paths:
        if not idx_path.exists():
            continue
        index = read_json(idx_path)
        frame_rows = index.get("frames", [])
        if not frame_rows:
            continue
        round_idx = int(index.get("round", 0) or 0)
        goal_idx = int(index.get("goal_index", frame_rows[0].get("goal_index", 0)) or 0)
        last_round_by_goal[goal_idx] = max(last_round_by_goal.get(goal_idx, -1), round_idx)

    for dec_dir in [p.parent for p in frame_index_paths]:
        idx_path = dec_dir / "frames_index.json"
        if not idx_path.exists():
            continue
        index = read_json(idx_path)
        frame_rows = index.get("frames", [])
        if not frame_rows:
            continue
        round_idx = int(index.get("round", len(frames)))
        goal_idx = int(index.get("goal_index", frame_rows[0].get("goal_index", 0)) or 0)
        goal_desc = goals[goal_idx]["sentence"] if goal_idx < len(goals) else f"goal {goal_idx}"

        evidence_dir = run_dir / ARTIFACT_ROOT_NAME / EVIDENCE_ARTIFACT_DIR / f"dec_{round_idx:03d}"
        entity_dir = run_dir / ARTIFACT_ROOT_NAME / ENTITY_ARTIFACT_DIR / f"dec_{round_idx:03d}"
        default_vtag = "final" if goal_idx == len(goals) - 1 else f"goal_{goal_idx:02d}"
        endpoint_dir = run_dir / ARTIFACT_ROOT_NAME / ENDPOINT_ARTIFACT_DIR / default_vtag

        def nav_frame_from_row(row: Dict[str, Any], duration_frames: int = 1) -> NavFrame:
            meta = dict(row)
            return NavFrame(
                kind="nav",
                goal_index=goal_idx,
                round_index=round_idx,
                step_count=int(row.get("step_count", 0)),
                left_mode="fpv",
                active_module="",
                status="[Policy] interval navigation",
                rgb_path=dec_dir / str(row["rgb"]),
                topdown_path=dec_dir / str(row["topdown"]),
                rgb_map_path=evidence_dir / "topdown_scene_rgb.png",
                module_image_paths=[],
                frame_meta=meta,
                duration_frames=max(1, int(duration_frames)),
            )

        prelude_count = min(8, max(1, len(frame_rows) // 4))
        for row in frame_rows[:prelude_count]:
            latest_meta = dict(row)
            frames.append(nav_frame_from_row(row))

        pano_imgs = sorted((evidence_dir / "panorama").glob("view_*.png"))
        frames.append(
            NavFrame(
                kind="module",
                goal_index=goal_idx,
                round_index=round_idx,
                step_count=int(frame_rows[0].get("step_count", 0)),
                left_mode="candidate_page",
                active_module=EVIDENCE_MODULE,
                status="[Evidence Grounding] Which frontier?",
                rgb_path=None,
                topdown_path=evidence_dir / EVIDENCE_OVERLAY_NAME,
                rgb_map_path=evidence_dir / "topdown_scene_rgb_annotated.png",
                module_image_paths=pano_imgs,
                frame_meta=latest_meta.copy(),
                duration_frames=EVIDENCE_HOLD_FRAMES,
            )
        )
        frames.append(
            NavFrame(
                kind="module",
                goal_index=goal_idx,
                round_index=round_idx,
                step_count=int(frame_rows[0].get("step_count", 0)),
                left_mode="candidate_page",
                active_module=ENTITY_MODULE,
                status="[Entity Grounding] Which object?",
                rgb_path=None,
                topdown_path=entity_dir / ENTITY_OVERLAY_NAME,
                rgb_map_path=entity_dir / "topdown_scene_rgb_annotated.png",
                module_image_paths=[entity_dir / ENTITY_OVERLAY_NAME],
                frame_meta=latest_meta.copy(),
                duration_frames=ENTITY_HOLD_FRAMES,
            )
        )
        for row in frame_rows[prelude_count:]:
            latest_meta = dict(row)
            frames.append(nav_frame_from_row(row))
        if round_idx == last_round_by_goal.get(goal_idx, round_idx):
            frames.append(
                NavFrame(
                    kind="module",
                    goal_index=goal_idx,
                    round_index=round_idx,
                    step_count=int(frame_rows[-1].get("step_count", 0)),
                    left_mode="candidate_page",
                    active_module=ENDPOINT_MODULE,
                    status="[Endpoint Grounding] Which viewpoint?",
                    rgb_path=None,
                    topdown_path=endpoint_dir / ENDPOINT_CANDIDATES_TOPDOWN_NAME,
                    rgb_map_path=endpoint_dir / "topdown_scene_rgb_annotated.png",
                    module_image_paths=[endpoint_dir / ENDPOINT_CANDIDATES_ZOOM_NAME, endpoint_dir / ENDPOINT_TARGET_RGB_POINTS_NAME],
                    frame_meta=latest_meta.copy(),
                    duration_frames=ENDPOINT_HOLD_FRAMES,
                )
            )
            if frame_rows:
                latest_meta = dict(frame_rows[-1])
                frames.append(nav_frame_from_row(frame_rows[-1], duration_frames=10 if goal_idx == len(goals) - 1 else 4))
    return frames, summary


def render_right_map(frame: NavFrame, variant: str) -> np.ndarray:
    if variant == "topdown":
        path = frame.topdown_path
        if path and image_ok(path):
            base = load_bgr(path)
            if frame.kind == "nav":
                base = draw_rgb_map_overlay(base, frame.frame_meta)
            return fit_cover(base, RIGHT_W, CANVAS_H)
    path = frame.rgb_map_path
    if path and image_ok(path):
        base = load_bgr(path)
        if frame.kind == "nav":
            base = draw_rgb_map_overlay(base, frame.frame_meta)
        return fit_cover(base, RIGHT_W, CANVAS_H)
    return np.full((CANVAS_H, RIGHT_W, 3), (32, 32, 32), dtype=np.uint8)


def render_left(run_dir: Path, frame: NavFrame) -> np.ndarray:
    if frame.left_mode == "candidate_page":
        return make_module_page(run_dir, frame)
    if frame.rgb_path and image_ok(frame.rgb_path):
        return fit_cover(clean_camera_follow_bgr(load_bgr(frame.rgb_path)), LEFT_W, CANVAS_H)
    return np.zeros((CANVAS_H, LEFT_W, 3), dtype=np.uint8)


def render_video_frame(run_dir: Path, frame: NavFrame, variant: str, out_frame_idx: int, summary: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    left = render_left(run_dir, frame)
    right = render_right_map(frame, variant)
    canvas[:, :LEFT_W] = left
    canvas[:, LEFT_W:] = right
    cv2.line(canvas, (LEFT_W, 0), (LEFT_W, CANVAS_H), (230, 230, 230), 2)
    goals = summary.get("goals", [])
    task_label = goals[frame.goal_index]["sentence"] if frame.goal_index < len(goals) else ""
    module_line = frame.status
    draw_panel_label(
        canvas,
        [
            f"[Task {frame.goal_index + 1}/{max(1, len(goals))}]: {task_label}",
            module_line,
            f"[Step]: {frame.step_count:04d}  [View]: {'TopDown Map' if variant == 'topdown' else 'RGB Map'}",
        ],
    )
    # Right-side task color legend, stable and readable without covering the map.
    legend_h = min(168, 28 * max(1, len(goals[:5])) + 20)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (LEFT_W + 10, 86), (LEFT_W + 178, 86 + legend_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, dst=canvas)
    y = 112
    for gi, goal in enumerate(goals[:5]):
        color = color_for_goal(gi)
        cv2.line(canvas, (LEFT_W + 26, y), (LEFT_W + 82, y), color, 7, cv2.LINE_AA)
        draw_text(canvas, f"Task {gi + 1}", (LEFT_W + 94, y + 7), 0.52, color, 1)
        y += 28
    actual_sources = [
        rel_to_run(run_dir, frame.rgb_path),
        rel_to_run(run_dir, frame.topdown_path),
        rel_to_run(run_dir, frame.rgb_map_path),
    ]
    actual_sources.extend(rel_to_run(run_dir, p) for p in frame.module_image_paths if p.exists())
    actual_sources = [src for src in actual_sources if src]
    meta = {
        "frame_idx": int(out_frame_idx),
        "variant": variant,
        "kind": frame.kind,
        "phase": "module_call" if frame.left_mode == "candidate_page" else "navigation",
        "goal_index": int(frame.goal_index),
        "task_id": int(frame.goal_index),
        "round_index": int(frame.round_index),
        "step": int(frame.step_count),
        "left_mode": frame.left_mode,
        "module_call": bool(frame.left_mode == "candidate_page"),
        "active_module": frame.active_module,
        "status": frame.status,
        "agent_pose_map_px": frame.frame_meta.get("agent_pixel_rc"),
        "camera_pitch_deg": frame.frame_meta.get("color_sensor_pitch_deg"),
        "camera_down_tilt_deg": frame.frame_meta.get("color_sensor_down_tilt_deg"),
        "source_rgb": public_source_name(run_dir, frame.rgb_path),
        "source_topdown": public_source_name(run_dir, frame.topdown_path),
        "source_rgb_map": public_source_name(run_dir, frame.rgb_map_path),
        "source_module_images": [public_source_name(run_dir, p) for p in frame.module_image_paths if p.exists()],
        "_actual_source_files": actual_sources,
    }
    return canvas, meta


def write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {raw}")
    count = 0
    for frame in frames:
        writer.write(frame)
        count += 1
    writer.release()
    if count == 0:
        raise RuntimeError("no frames written")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264",
             "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", str(path)],
            check=True,
        )
        raw.unlink(missing_ok=True)
    else:
        raw.replace(path)


def expanded_frames(run_dir: Path, timeline: Sequence[NavFrame], variant: str, summary: Dict[str, Any]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    imgs: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    out_idx = 0
    for item in timeline:
        for _ in range(max(1, int(item.duration_frames))):
            img, meta = render_video_frame(run_dir, item, variant, out_idx, summary)
            imgs.append(img)
            metas.append(meta)
            out_idx += 1
    return imgs, metas


def video_stats(path: Path, *, check_rgb_edge_black: bool) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "cannot open"}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    low_var = 0
    black_edge_bad = 0
    samples: List[np.ndarray] = []
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if float(frame.std()) < 2.0:
            low_var += 1
        if i in {0, frames // 4, frames // 2, frames * 3 // 4, max(0, frames - 1)}:
            samples.append(frame.copy())
        if check_rgb_edge_black:
            edge_parts = [
                frame[:5, LEFT_W:].reshape(-1, 3),
                frame[-5:, LEFT_W:].reshape(-1, 3),
                frame[:, LEFT_W:LEFT_W + 5].reshape(-1, 3),
                frame[:, -5:].reshape(-1, 3),
            ]
            edge = np.concatenate(edge_parts, axis=0).reshape(-1, 1, 3)
            if float((cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY) < 8).mean()) > 0.03:
                black_edge_bad += 1
    cap.release()
    sheet_path = path.with_name(path.stem + "_review_sheet.jpg")
    if samples:
        cv2.imwrite(str(sheet_path), np.concatenate([fit_contain(x, 384, 144) for x in samples], axis=1))
    return {
        "ok": bool(frames > 0 and w == CANVAS_W and h == CANVAS_H and low_var == 0 and black_edge_bad == 0),
        "frames": frames,
        "fps": fps,
        "duration_s": frames / fps if fps else 0,
        "width": w,
        "height": h,
        "low_variance_frames": low_var,
        "right_edge_black_bad_frames": black_edge_bad,
        "right_edge_black_check": "enabled_for_rgb_map" if check_rgb_edge_black else "skipped_for_semantic_topdown",
        "review_sheet": str(sheet_path),
    }


def build_timeline_summary(timeline: Sequence[NavFrame], frame_metadata: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    variants = {
        name: {
            "frames": len(rows),
            "navigation_frames": sum(1 for row in rows if row.get("phase") == "navigation"),
            "module_call_frames": sum(1 for row in rows if row.get("phase") == "module_call"),
        }
        for name, rows in frame_metadata.items()
    }
    return {
        "timeline_items": len(timeline),
        "navigation_items": sum(1 for item in timeline if item.kind == "nav"),
        "module_items": sum(1 for item in timeline if item.kind == "module"),
        "module_sequence": [
            {
                "goal_index": item.goal_index,
                "round_index": item.round_index,
                "active_module": item.active_module,
                "step": item.step_count,
                "duration_frames": item.duration_frames,
            }
            for item in timeline
            if item.kind == "module"
        ],
        "variants": variants,
    }


def check_fresh_sources(run_dir: Path, frame_metadata: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    external_sources: List[str] = []
    missing_sources: List[str] = []
    old_source_hits: List[str] = []
    seen_sources = set()
    for rows in frame_metadata.values():
        for row in rows:
            values: List[str] = []
            if row.get("_actual_source_files"):
                values.extend(str(x) for x in row.get("_actual_source_files", []) if x)
            else:
                for key in ("source_rgb", "source_topdown", "source_rgb_map"):
                    if row.get(key):
                        values.append(str(row[key]))
                values.extend(str(x) for x in row.get("source_module_images", []) if x)
            for src in values:
                if src in seen_sources:
                    continue
                seen_sources.add(src)
                if any(token in src for token in FORBIDDEN_SOURCE_MARKERS):
                    old_source_hits.append(src)
                p = Path(src)
                if p.is_absolute():
                    try:
                        p.relative_to(run_dir)
                    except ValueError:
                        external_sources.append(src)
                    real_path = p
                else:
                    real_path = run_dir / p
                if not real_path.exists():
                    missing_sources.append(src)
    return {
        "ok": not external_sources and not missing_sources and not old_source_hits,
        "source_root": str(run_dir),
        "unique_source_files": len(seen_sources),
        "external_sources": external_sources,
        "missing_sources": missing_sources,
        "old_source_hits": old_source_hits,
        "forbidden_source_rule_count": len(FORBIDDEN_SOURCE_MARKERS),
    }


def public_frame_metadata(frame_metadata: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    public: Dict[str, List[Dict[str, Any]]] = {}
    for variant, rows in frame_metadata.items():
        public[variant] = []
        for row in rows:
            public[variant].append({k: v for k, v in row.items() if not str(k).startswith("_")})
    return public


def public_goals(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    goals = summary.get("goals", [])
    out: List[Dict[str, Any]] = []
    if not isinstance(goals, list):
        return out
    allowed = {
        "goal_index",
        "task_id",
        "sentence",
        "stop_reason",
        "round_start",
        "round_end",
        "step_start",
        "step_end",
        "final_planar_distance_to_goal_m",
    }
    for goal in goals:
        if isinstance(goal, dict):
            out.append({k: v for k, v in goal.items() if k in allowed})
    return out


def check_navigation_camera_level(frame_metadata: Dict[str, List[Dict[str, Any]]], max_down_tilt_deg: float = 8.0) -> Dict[str, Any]:
    bad: List[Dict[str, Any]] = []
    missing = 0
    checked = 0
    max_seen = 0.0
    for variant, rows in frame_metadata.items():
        for row in rows:
            if row.get("phase") != "navigation":
                continue
            checked += 1
            value = row.get("camera_down_tilt_deg")
            if value is None:
                missing += 1
                continue
            tilt = float(value)
            max_seen = max(max_seen, tilt)
            if tilt > float(max_down_tilt_deg):
                bad.append(
                    {
                        "variant": variant,
                        "frame_idx": row.get("frame_idx"),
                        "goal_index": row.get("goal_index"),
                        "step": row.get("step"),
                        "camera_down_tilt_deg": tilt,
                        "source_rgb": row.get("source_rgb"),
                    }
                )
    sample_limit = 20
    return {
        "ok": checked > 0 and missing == 0 and not bad,
        "checked_navigation_frames": checked,
        "missing_pose_metric_frames": missing,
        "max_allowed_down_tilt_deg": float(max_down_tilt_deg),
        "max_seen_down_tilt_deg": float(max_seen),
        "bad_frames": bad[:sample_limit],
        "bad_frame_count": len(bad),
    }


def check_module_timeline(frame_metadata: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    variants = sorted(frame_metadata.keys())
    errors: List[str] = []
    if set(variants) != {"rgbmap", "topdown"}:
        errors.append("expected exactly topdown and rgbmap variants")
    baseline_rows = frame_metadata.get("topdown", [])
    first_last_errors = 0
    unbounded_module_windows = 0
    mismatch_count = 0
    nav_with_module = 0
    bad_module_names = 0
    bad_module_status = 0
    for variant, rows in frame_metadata.items():
        if len(rows) != len(baseline_rows):
            mismatch_count += abs(len(rows) - len(baseline_rows)) or 1
        for idx, row in enumerate(rows):
            if idx < len(baseline_rows):
                base = baseline_rows[idx]
                keys = ("phase", "goal_index", "round_index", "step", "active_module", "status")
                if any(row.get(k) != base.get(k) for k in keys):
                    mismatch_count += 1
            phase = row.get("phase")
            module = str(row.get("active_module", ""))
            if phase == "navigation" and module:
                nav_with_module += 1
            if phase == "module_call":
                expected_status = EXPECTED_MODULE_STATUS.get(module)
                if expected_status is None:
                    bad_module_names += 1
                elif str(row.get("status", "")) != expected_status:
                    bad_module_status += 1
        if rows:
            if rows[0].get("phase") != "navigation" or rows[-1].get("phase") != "navigation":
                first_last_errors += 1
            for i, row in enumerate(rows):
                if row.get("phase") != "module_call":
                    continue
                prev_phase = rows[i - 1].get("phase") if i > 0 else ""
                next_phase = rows[i + 1].get("phase") if i + 1 < len(rows) else ""
                if prev_phase != "module_call" and prev_phase != "navigation":
                    unbounded_module_windows += 1
                if next_phase != "module_call" and next_phase != "navigation":
                    unbounded_module_windows += 1
    module_counts = {
        name: sum(
            1
            for rows in frame_metadata.values()
            for row in rows
            if row.get("phase") == "module_call" and row.get("active_module") == name
        )
        for name in sorted(EXPECTED_MODULE_STATUS)
    }
    if nav_with_module:
        errors.append("navigation frames must not carry an active module")
    if bad_module_names:
        errors.append("module-call frames use unexpected module names")
    if bad_module_status:
        errors.append("module-call frames use unexpected question/status labels")
    if mismatch_count:
        errors.append("topdown and rgbmap frame metadata are not synchronized")
    if first_last_errors:
        errors.append("each video must start and end with navigation frames")
    if unbounded_module_windows:
        errors.append("module-call windows must be adjacent to navigation, not isolated as the whole video")
    if any(count <= 0 for count in module_counts.values()):
        errors.append("each grounding module must appear at least once")
    return {
        "ok": not errors,
        "errors": errors,
        "variant_count": len(variants),
        "timeline_mismatch_count": int(mismatch_count),
        "navigation_frames_with_active_module": int(nav_with_module),
        "bad_module_name_frames": int(bad_module_names),
        "bad_module_status_frames": int(bad_module_status),
        "first_last_navigation_errors": int(first_last_errors),
        "unbounded_module_window_errors": int(unbounded_module_windows),
        "module_call_frame_counts": module_counts,
    }


def check_track_visibility(
    videos: Dict[str, Path],
    frame_metadata: Dict[str, List[Dict[str, Any]]],
    *,
    min_color_hits: int = 20,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "ok": True,
        "min_required_task_color_pixels": int(min_color_hits),
        "variants": {},
    }
    for variant, path in videos.items():
        rows_by_frame = {
            int(row.get("frame_idx", -1)): row
            for row in frame_metadata.get(variant, [])
            if row.get("phase") == "navigation"
        }
        cap = cv2.VideoCapture(str(path))
        bad: List[Dict[str, Any]] = []
        checked = 0
        min_hits: Optional[int] = None
        frame_idx = 0
        if cap.isOpened():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                row = rows_by_frame.get(frame_idx)
                if row is not None:
                    goal_index = int(row.get("goal_index", 0) or 0)
                    color = np.asarray(color_for_goal(goal_index), dtype=np.int16)
                    # Exclude the status strip and the task legend, so the check
                    # proves the path itself is visible on the map body.
                    map_body = frame[86:, LEFT_W + 190:].astype(np.int16)
                    color_distance = np.abs(map_body - color).sum(axis=2)
                    hits = int((color_distance < 70).sum())
                    checked += 1
                    min_hits = hits if min_hits is None else min(min_hits, hits)
                    if hits < int(min_color_hits):
                        bad.append(
                            {
                                "frame_idx": int(frame_idx),
                                "goal_index": goal_index,
                                "task_color_pixel_hits": hits,
                            }
                        )
                frame_idx += 1
        cap.release()
        variant_report = {
            "ok": checked > 0 and not bad,
            "checked_navigation_frames": int(checked),
            "min_task_color_pixel_hits": int(min_hits or 0),
            "bad_frame_count": len(bad),
            "bad_frames": bad[:20],
        }
        if not variant_report["ok"]:
            report["ok"] = False
        report["variants"][variant] = variant_report
    return report


def check_camera_follow_clean(
    videos: Dict[str, Path],
    frame_metadata: Dict[str, List[Dict[str, Any]]],
    *,
    max_edge_black_ratio: float = 0.08,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "ok": True,
        "max_allowed_edge_connected_black_ratio": float(max_edge_black_ratio),
        "variants": {},
    }
    for variant, path in videos.items():
        rows_by_frame = {
            int(row.get("frame_idx", -1)): row
            for row in frame_metadata.get(variant, [])
            if row.get("phase") == "navigation"
        }
        cap = cv2.VideoCapture(str(path))
        bad: List[Dict[str, Any]] = []
        checked = 0
        max_seen = 0.0
        frame_idx = 0
        if cap.isOpened():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in rows_by_frame:
                    left = frame[:, :LEFT_W]
                    edge_black = edge_connected_mask(left.max(axis=2) <= 8)
                    ratio = float(edge_black.mean())
                    max_seen = max(max_seen, ratio)
                    checked += 1
                    if ratio > float(max_edge_black_ratio):
                        bad.append(
                            {
                                "frame_idx": int(frame_idx),
                                "goal_index": rows_by_frame[frame_idx].get("goal_index"),
                                "step": rows_by_frame[frame_idx].get("step"),
                                "edge_connected_black_ratio": ratio,
                            }
                        )
                frame_idx += 1
        cap.release()
        variant_report = {
            "ok": checked > 0 and not bad,
            "checked_navigation_frames": int(checked),
            "max_edge_connected_black_ratio": float(max_seen),
            "bad_frame_count": len(bad),
            "bad_frames": bad[:20],
        }
        if not variant_report["ok"]:
            report["ok"] = False
        report["variants"][variant] = variant_report
    return report


def check_public_outputs_clean(out_run: Path) -> Dict[str, Any]:
    text_suffixes = {".json", ".txt", ".log", ".md", ".csv"}
    checked_paths = 0
    checked_text_files = 0
    path_hits = 0
    text_hits = 0

    def has_discontinued_draft_name(text: str) -> bool:
        return any(pattern.search(text) for pattern in DRAFT_NAME_PATTERNS)

    for path in out_run.rglob("*"):
        checked_paths += 1
        path_text = str(path.relative_to(out_run))
        if any(token in path_text for token in FORBIDDEN_PUBLIC_OUTPUT_MARKERS) or has_discontinued_draft_name(path_text):
            path_hits += 1
        if path.is_file() and path.suffix.lower() in text_suffixes:
            checked_text_files += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if any(token in text for token in FORBIDDEN_PUBLIC_OUTPUT_MARKERS) or has_discontinued_draft_name(text):
                text_hits += 1
    return {
        "ok": path_hits == 0 and text_hits == 0,
        "forbidden_rule_count": len(FORBIDDEN_PUBLIC_OUTPUT_MARKERS) + len(DRAFT_NAME_PATTERNS),
        "checked_paths": int(checked_paths),
        "checked_text_files": int(checked_text_files),
        "path_hit_count": int(path_hits),
        "text_hit_count": int(text_hits),
    }


def retention(root: Path, keep: int) -> int:
    runs = sorted([p for p in root.glob("run=*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    deleted_count = 0
    for p in runs[int(keep):]:
        deleted_count += 1
        shutil.rmtree(p, ignore_errors=True)
    return deleted_count


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate fresh navigation process videos.")
    ap.add_argument("--scene_name", default="00800-TEEsavR23oF")
    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--goal_count", type=int, default=5)
    ap.add_argument("--max_rounds_per_goal", type=int, default=1)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--run-dir", default="", help="Render an existing fresh run; must be under video-creator/nav-logs unless --allow-any-run-dir is set.")
    ap.add_argument("--allow-any-run-dir", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--nav-log-dir", default=str(NAV_LOG_DIR))
    ap.add_argument("--keep-runs", type=int, default=2)
    ap.add_argument("--skip-log-generation", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    nav_log_root = Path(args.nav_log_dir).expanduser().resolve()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not args.allow_any_run_dir and nav_log_root not in [run_dir, *run_dir.parents]:
            raise SystemExit(f"--run-dir must be under {nav_log_root} unless --allow-any-run-dir is set")
    else:
        if bool(args.skip_log_generation):
            latest = find_latest_run(nav_log_root)
            if latest is None:
                raise SystemExit(f"no fresh run found under {nav_log_root}")
            run_dir = latest.resolve()
        else:
            run_dir = run_fresh_navigation(args, nav_log_root)

    timeline, summary = collect_timeline(run_dir)
    if not timeline:
        raise RuntimeError(f"no renderable guided timeline found in {run_dir}")

    out_run = output_root / f"run={dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_run.mkdir(parents=True, exist_ok=False)
    videos = {
        "topdown": out_run / "navi_process_fresh_topdown.mp4",
        "rgbmap": out_run / "navi_process_fresh_rgbmap.mp4",
    }
    frame_metadata: Dict[str, Any] = {}
    for variant, path in videos.items():
        imgs, metas = expanded_frames(run_dir, timeline, variant, summary)
        write_video(imgs, path, FPS)
        frame_metadata[variant] = metas

    stats = {name: video_stats(path, check_rgb_edge_black=(name == "rgbmap")) for name, path in videos.items()}
    timeline_report = build_timeline_summary(timeline, frame_metadata)
    source_report = check_fresh_sources(run_dir, frame_metadata)
    camera_report = check_navigation_camera_level(frame_metadata)
    module_report = check_module_timeline(frame_metadata)
    track_report = check_track_visibility(videos, frame_metadata)
    fpv_report = check_camera_follow_clean(videos, frame_metadata)
    public_metadata = public_frame_metadata(frame_metadata)
    write_json(out_run / "frame_metadata.json", public_metadata)
    errors = [f"{name}: {s}" for name, s in stats.items() if not s.get("ok")]
    if not source_report.get("ok"):
        errors.append(f"fresh source check failed: {source_report}")
    if not camera_report.get("ok"):
        errors.append(f"navigation camera level check failed: {camera_report}")
    if not module_report.get("ok"):
        errors.append("module timeline check failed")
    if not track_report.get("ok"):
        errors.append("track visibility check failed")
    if not fpv_report.get("ok"):
        errors.append("camera-follow FPV cleanliness check failed")
    audit = {
        "ok": len(errors) == 0,
        "errors": errors,
        "fresh_run_dir": str(run_dir),
        "videos": {k: str(v) for k, v in videos.items()},
        "video_stats": stats,
        "source_policy": "fresh_navigation_logs_only",
        "fresh_source_check": source_report,
        "navigation_camera_level_check": camera_report,
        "module_timeline_check": module_report,
        "track_visibility_check": track_report,
        "camera_follow_clean_check": fpv_report,
        "public_output_clean_check": {"ok": False, "pending_until_files_are_written": True},
        "timeline": timeline_report,
        "topdown_black_note": "Top-down map uses black/gray semantic occupancy; RGB edge-black is checked separately.",
        "goal_count": int(summary.get("goal_count", 0) or 0),
        "decision_rounds": int(summary.get("decision_rounds", 0) or 0),
        "fps": FPS,
        "resolution": [CANVAS_W, CANVAS_H],
    }
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_policy": audit["source_policy"],
        "fresh_run_dir": str(run_dir),
        "scene": summary.get("scene_name"),
        "episode_id": summary.get("episode_id"),
        "multi_goal_sequence": summary.get("multi_goal_sequence"),
        "description_style": "concise",
        "goal_count": audit["goal_count"],
        "decision_rounds": audit["decision_rounds"],
        "frame_count": {name: stat.get("frames") for name, stat in stats.items()},
        "fps": FPS,
        "resolution": [CANVAS_W, CANVAS_H],
        "goals": public_goals(summary),
        "videos": {k: str(v) for k, v in videos.items()},
        "timeline": timeline_report,
        "fresh_source_check": source_report,
        "navigation_camera_level_check": camera_report,
        "module_timeline_check": module_report,
        "track_visibility_check": track_report,
        "camera_follow_clean_check": fpv_report,
        "public_output_clean_check": audit["public_output_clean_check"],
        "video_stats": stats,
        "audit_ok": audit["ok"],
    }
    write_json(out_run / "video_creator_audit.json", audit)
    write_json(out_run / "manifest.json", manifest)
    deleted_output = retention(output_root, int(args.keep_runs))
    deleted_logs = retention(nav_log_root, int(args.keep_runs))
    retention_report = {
        "policy": "keep_latest_two_dynamic_windows",
        "kept_output_run_count": min(int(args.keep_runs), len([p for p in output_root.glob("run=*") if p.is_dir()])),
        "kept_nav_log_run_count": min(int(args.keep_runs), len([p for p in nav_log_root.glob("run=*") if p.is_dir()])),
        "deleted_output_run_count": int(deleted_output),
        "deleted_nav_log_run_count": int(deleted_logs),
    }
    write_json(out_run / "retention_report.json", retention_report)
    public_report = check_public_outputs_clean(out_run)
    if not public_report.get("ok"):
        errors.append("public output clean check failed")
    audit["errors"] = errors
    audit["ok"] = len(errors) == 0
    audit["public_output_clean_check"] = public_report
    audit["retention"] = retention_report
    manifest["public_output_clean_check"] = public_report
    manifest["retention"] = retention_report
    manifest["audit_ok"] = audit["ok"]
    write_json(out_run / "video_creator_audit.json", audit)
    write_json(out_run / "manifest.json", manifest)
    print(json.dumps({**audit, "retention": retention_report}, ensure_ascii=False, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
