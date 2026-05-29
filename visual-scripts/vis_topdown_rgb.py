"""vis_topdown_rgb.py - Render a photorealistic top-down RGB view of a HM3D scene.

Places a high-altitude camera pointing straight down, auto-computes height
from the scene bounding box so the whole floor plan is captured.

Outputs (under --output_dir):
  <scene_name>/
    topdown_rgb.png          raw render (高分辨率 RGB)
    topdown_rgb_anno.png     带标注版：场景名、比例尺、方向标

Usage (run from repo root with mtu3d env active):
  python visual-scripts/vis_topdown_rgb.py \\
      --scene_name 00802-wcojb4TFT35 \\
      [--output_dir  visual-output/topdown_rgb] \\
      [--resolution  2048] \\
      [--hfov        60] \\
      [--height_margin_m  2.0]   # extra altitude above computed minimum

Run via shell wrapper:
  bash visual-scripts/run_vis_topdown_rgb.sh --scene_name 00802-wcojb4TFT35
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import habitat_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"
if str(HM3D_ONLINE) not in sys.path:
    sys.path.insert(0, str(HM3D_ONLINE))


# ──────────────────────────────────────────────────────────────────────────────
# Scene path resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    for candidate in [scene_dir / f"{short}.basis.glb", scene_dir / f"{short}.glb"]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name} under {hm3d_root}")


# ──────────────────────────────────────────────────────────────────────────────
# Camera geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scene_bounds(sim: habitat_sim.Simulator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of the scene bounding box."""
    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    return np.array(bb.min, dtype=float), np.array(bb.max, dtype=float)


def compute_camera_params(
    sim: habitat_sim.Simulator,
    hfov_deg: float,
    height_margin_m: float,
    aspect: float = 1.0,
) -> Tuple[float, float, float, float, float, float]:
    """
    Return (agent_x, agent_y_floor, agent_z, sensor_height,
            scene_width_m, scene_depth_m).

    The camera is placed at (agent_x, agent_y_floor + sensor_height, agent_z)
    and points straight down.  sensor_height is calculated so that the whole
    horizontal extent of the scene fits inside the FOV.
    """
    min_pt, max_pt = _scene_bounds(sim)

    center_x = float((min_pt[0] + max_pt[0]) / 2)
    center_z = float((min_pt[2] + max_pt[2]) / 2)
    floor_y  = float(min_pt[1])          # approximate ground level

    scene_w = float(max_pt[0] - min_pt[0])   # x extent
    scene_d = float(max_pt[2] - min_pt[2])   # z extent

    # Half-diagonal of the horizontal footprint (worst-case corner distance)
    half_diag = math.hypot(scene_w / 2, scene_d / 2)

    # For a square render (aspect=1), the "radius" we must cover is half_diag
    hfov_rad  = math.radians(hfov_deg)
    min_h = half_diag / math.tan(hfov_rad / 2)

    sensor_height = min_h + height_margin_m

    return center_x, floor_y, center_z, sensor_height, scene_w, scene_d


# ──────────────────────────────────────────────────────────────────────────────
# Simulator builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_sim_for_bounds(scene_path: str) -> habitat_sim.Simulator:
    """Minimal sim (no sensors) just to query scene bounds."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = False
    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()])
    )


def _build_sim_with_overhead_cam(
    scene_path: str,
    sensor_height: float,
    resolution: int,
    hfov_deg: float,
) -> habitat_sim.Simulator:
    """Sim with a single overhead COLOR camera sensor."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.load_semantic_mesh = False

    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "topdown_rgb"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    cam.resolution = [resolution, resolution]
    # Sensor position is relative to the agent: place it 'sensor_height' above
    cam.position = [0.0, float(sensor_height), 0.0]
    # Pitch -90° → camera looks straight down (-Y world axis)
    cam.orientation = [-math.pi / 2, 0.0, 0.0]
    cam.hfov = float(hfov_deg)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]

    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )


# ──────────────────────────────────────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────────────────────────────────────

def render_topdown(
    scene_path: str,
    output_dir: Path,
    scene_name: str,
    resolution: int = 2048,
    hfov_deg: float = 60.0,
    height_margin_m: float = 2.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: query scene geometry ─────────────────────────────────────────
    print(f"[topdown] Loading scene for bounds: {scene_path}")
    tmp_sim = _build_sim_for_bounds(scene_path)
    min_pt, max_pt = _scene_bounds(tmp_sim)
    center_x, floor_y, center_z, sensor_h, scene_w, scene_d = compute_camera_params(
        tmp_sim, hfov_deg, height_margin_m
    )
    tmp_sim.close()

    print(
        f"[topdown] Scene extent  x={scene_w:.1f}m  z={scene_d:.1f}m  "
        f"y=[{min_pt[1]:.2f}, {max_pt[1]:.2f}]"
    )
    print(
        f"[topdown] Camera → agent at ({center_x:.2f}, {floor_y:.2f}, {center_z:.2f})  "
        f"sensor height={sensor_h:.2f}m  hfov={hfov_deg}°"
    )

    # ── Step 2: build sim with overhead camera ───────────────────────────────
    sim = _build_sim_with_overhead_cam(scene_path, sensor_h, resolution, hfov_deg)

    # Place agent at scene centre, floor level
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()

    target_pos = np.array([center_x, floor_y, center_z], dtype=float)
    # If navmesh is available, snap to navigable point
    if sim.pathfinder.is_loaded:
        snapped = np.array(sim.pathfinder.snap_point(target_pos), dtype=float)
        if np.isfinite(snapped).all() and not np.allclose(snapped, 0.0):
            target_pos = snapped
            target_pos[0] = center_x   # keep horizontal center
            target_pos[2] = center_z
    state.position = target_pos.tolist()
    agent.set_state(state)

    # ── Step 3: render ───────────────────────────────────────────────────────
    obs = sim.get_sensor_observations()
    img_rgba = obs["topdown_rgb"]                  # H×W×4  RGBA
    img_rgb  = img_rgba[:, :, :3].copy()           # H×W×3  RGB
    sim.close()

    print(f"[topdown] Rendered  shape={img_rgb.shape}  dtype={img_rgb.dtype}")

    # ── Step 4: save raw PNG ─────────────────────────────────────────────────
    raw_path = output_dir / "topdown_rgb.png"
    cv2.imwrite(str(raw_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"[topdown] Raw → {raw_path}")

    # ── Step 5: annotated figure ─────────────────────────────────────────────
    anno_path = _save_annotated(
        img_rgb, output_dir, scene_name,
        scene_w=scene_w, scene_d=scene_d,
        sensor_h=sensor_h, hfov_deg=hfov_deg,
    )
    print(f"[topdown] Annotated → {anno_path}")

    return raw_path


def _scale_bar_pixels(img_w: int, scene_w_m: float, bar_m: Optional[float] = None) -> int:
    """Return number of pixels for a 'bar_m' metre scale bar."""
    if bar_m is None:
        # Auto: round to nice number
        options = [1, 2, 5, 10, 20]
        for b in options:
            frac = b / scene_w_m
            if 0.1 <= frac <= 0.35:
                bar_m = b
                break
        else:
            bar_m = options[-1]
    px = int(img_w * bar_m / scene_w_m)
    return px, float(bar_m)


def _save_annotated(
    img_rgb: np.ndarray,
    output_dir: Path,
    scene_name: str,
    scene_w: float,
    scene_d: float,
    sensor_h: float,
    hfov_deg: float,
) -> Path:
    H, W = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 12 * H / W + 0.8))
    ax.imshow(img_rgb)
    ax.axis("off")

    ax.set_title(
        f"{scene_name}   "
        f"场景 {scene_w:.1f}m × {scene_d:.1f}m   "
        f"相机高度 {sensor_h:.1f}m   hfov={hfov_deg:.0f}°",
        fontsize=11, pad=8,
    )

    # ── Scale bar ────────────────────────────────────────────────────────────
    bar_px, bar_m = _scale_bar_pixels(W, scene_w)
    bar_x0 = int(W * 0.05)
    bar_y  = int(H * 0.96)
    bar_h  = max(6, int(H * 0.008))
    ax.add_patch(plt.Rectangle(
        (bar_x0 / W * W, bar_y / H * H),   # imshow coords = pixel coords
        bar_px, bar_h,
        linewidth=0, facecolor="white", transform=ax.transData,
    ))
    ax.text(
        bar_x0 + bar_px / 2, bar_y + bar_h + H * 0.01,
        f"{bar_m:.0f} m",
        color="white", fontsize=10, ha="center", va="top",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0),
    )

    # ── North arrow (top-right corner) ───────────────────────────────────────
    ax.annotate(
        "N", xy=(W * 0.93, H * 0.08), xytext=(W * 0.93, H * 0.05),
        fontsize=12, color="white", ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
    )

    anno_path = output_dir / "topdown_rgb_anno.png"
    fig.savefig(str(anno_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return anno_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Render top-down RGB view of a HM3D scene.")
    ap.add_argument("--scene_name",      required=True,
                    help="e.g. 00802-wcojb4TFT35")
    ap.add_argument("--hm3d_data_base_path", default=str(PROJECT_ROOT / "datascene"))
    ap.add_argument("--output_dir",      default=str(PROJECT_ROOT / "visual-output" / "topdown_rgb"))
    ap.add_argument("--resolution",      type=int,   default=2048,
                    help="Render resolution (square, pixels)")
    ap.add_argument("--hfov",            type=float, default=60.0,
                    help="Horizontal FOV in degrees (smaller = less distortion)")
    ap.add_argument("--height_margin_m", type=float, default=2.0,
                    help="Extra metres added above minimum camera height")
    args = ap.parse_args()

    scene_path = resolve_scene_path(
        os.path.expanduser(args.hm3d_data_base_path), args.scene_name
    )
    out_dir = Path(os.path.expanduser(args.output_dir)) / args.scene_name

    render_topdown(
        scene_path=scene_path,
        output_dir=out_dir,
        scene_name=args.scene_name,
        resolution=int(args.resolution),
        hfov_deg=float(args.hfov),
        height_margin_m=float(args.height_margin_m),
    )
    print(f"\n[topdown] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
