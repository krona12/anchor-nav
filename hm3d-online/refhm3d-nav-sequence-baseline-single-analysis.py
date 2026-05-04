"""单 episode / 单 task 的 baseline（PQ3D）导航过程探析：环视检测与分割、候选与决策、俯视图可视化。

运行示例（在仓库根目录或 hm3d-online 下，需能加载 configs 与数据）::

    cd hm3d-online && python refhm3d-nav-sequence-baseline-single-analysis.py \\
        --scene_name 00802-wcojb4TFT35 --episode_id 17 --task_id 0 --task_level instance

输出目录默认：``output_logs/baseline_analysis/<时间戳>/scene=.../episode=.../task=.../``，
每次 decision 子目录 ``dec_XXX/`` 内含 ``panorama/``、``stage1_per_frame/``、``stage2_decision.json``，
根目录含 ``summary.json``。俯视图按 ``og3d_logit`` 标 top-k 物体（默认 5，彩色圆点 + 序号）；``object_candidates_topk/`` 下输出各候选 **首次检测 RGB** 与 **3D 实例 mask 的俯视图 (BEV)**。环视条带与 ``refhm3d-nav-sequence-analyze-anchor-vfv.py`` 相同：最近 12 帧 → ``list(reversed(...))`` → 可选 ``--panorama_subsample_frames`` 均匀抽帧 → ``stitch_panorama(verify_rgb)``（**不传** fov 等关键字，与 vfv 一致，用 posnode 默认）。
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf

_HM3D_ONLINE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HM3D_ONLINE.parent
for _p in (_PROJECT_ROOT, _HM3D_ONLINE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from common.embodied_utils.simulator import HabitatSimulator
from data_utils import PQ3DModel
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)

try:
    from anchor_nav.posnode import _save_rgb_jpg, stitch_panorama
except Exception:  # pragma: no cover
    _save_rgb_jpg = None
    stitch_panorama = None


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [
        scene_dir / f"{short_scene_name}.basis.glb",
        scene_dir / f"{short_scene_name}.glb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Scene asset not found for {scene_name}. Checked: {[str(x) for x in candidates]}"
    )


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _render_topdown(
    *,
    top_down_map: np.ndarray,
    fog_mask: np.ndarray,
    agent_rc: np.ndarray,
    frontier_rc_list: Sequence[Tuple[int, int]],
    chosen_rc: Optional[Tuple[int, int]] = None,
    visited_rc_list: Optional[Sequence[Tuple[int, int]]] = None,
    object_rank_markers_rc: Optional[Sequence[Tuple[int, int, int]]] = None,
    agent_radius: int = 6,
    frontier_radius: int = 4,
) -> np.ndarray:
    try:
        base = maps.colorize_topdown_map(top_down_map, fog_mask)
        rgb = np.asarray(base[:, :, :3], dtype=np.uint8).copy()
    except Exception:
        m = np.asarray(top_down_map, dtype=float)
        mn = float(np.min(m)) if m.size > 0 else 0.0
        mx = float(np.max(m)) if m.size > 0 else 1.0
        g = ((m - mn) / max(mx - mn, 1e-6) * 180.0).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=2)
        explored = np.asarray(fog_mask) > 0
        rgb[~explored] = np.array([18, 18, 18], dtype=np.uint8)

    h, w = rgb.shape[:2]

    def _disk(rr: int, cc: int, rad: int, color: Tuple[int, int, int]) -> None:
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc > rad * rad:
                    continue
                r, c = rr + dr, cc + dc
                if 0 <= r < h and 0 <= c < w:
                    rgb[r, c, :] = np.array(color, dtype=np.uint8)

    if visited_rc_list:
        for (rr, cc) in visited_rc_list:
            rr = int(np.clip(rr, 0, h - 1))
            cc = int(np.clip(cc, 0, w - 1))
            _disk(rr, cc, max(2, frontier_radius - 1), (90, 90, 90))

    for (rr, cc) in frontier_rc_list:
        rr = int(np.clip(rr, 0, h - 1))
        cc = int(np.clip(cc, 0, w - 1))
        _disk(rr, cc, frontier_radius, (40, 220, 40))

    # PQ3D stage2 物体候选按 og3d_logit 的 top-k（默认 5 色）
    if object_rank_markers_rc:
        rank_colors = {
            1: (255, 215, 70),
            2: (255, 130, 50),
            3: (200, 90, 255),
            4: (80, 220, 220),
            5: (120, 255, 120),
        }
        for rr, cc, rk in object_rank_markers_rc:
            rk = int(rk)
            col = rank_colors.get(rk, (240, 240, 60))
            rr = int(np.clip(int(rr), 0, h - 1))
            cc = int(np.clip(int(cc), 0, w - 1))
            _disk(rr, cc, 6, (20, 20, 20))
            _disk(rr, cc, 5, col)
            _disk(rr, cc, 2, (255, 255, 255))

    if chosen_rc is not None:
        rr, cc = chosen_rc
        rr = int(np.clip(int(rr), 0, h - 1))
        cc = int(np.clip(int(cc), 0, w - 1))
        _disk(rr, cc, frontier_radius + 2, (40, 128, 255))

    rr = int(np.clip(int(agent_rc[0]), 0, h - 1))
    cc = int(np.clip(int(agent_rc[1]), 0, w - 1))
    _disk(rr, cc, agent_radius, (255, 36, 36))
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r, c = rr + dr, cc + dc
            if 0 <= r < h and 0 <= c < w:
                rgb[r, c, :] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _habitat_xyz_to_map_rc(pos: Sequence[float], top_down_map: np.ndarray, sim: Any) -> Tuple[int, int]:
    p = np.asarray(pos, dtype=float).reshape(3)
    rc = map_coors_to_pixel(p, top_down_map, sim)
    return int(rc[0]), int(rc[1])


def _object_topk_rc_from_stage2_json(
    stage2_path: Path,
    *,
    top_down_map: np.ndarray,
    sim: Any,
    k: int = 5,
) -> List[Tuple[int, int, int]]:
    """读取 stage2_decision.json，按 og3d_logit 取物体候选 top-k，返回 (row, col, rank)。"""
    if not stage2_path.is_file():
        return []
    try:
        with open(stage2_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    objs = data.get("object_candidates") or []
    if not objs:
        return []
    kk = max(1, int(k))
    sorted_objs = sorted(objs, key=lambda x: float(x.get("og3d_logit", -1e30)), reverse=True)[:kk]
    out: List[Tuple[int, int, int]] = []
    for rank, o in enumerate(sorted_objs, start=1):
        xyz = o.get("center_habitat_xyz")
        if not xyz or len(xyz) < 3:
            continue
        try:
            rr, cc = _habitat_xyz_to_map_rc(xyz, top_down_map, sim)
        except Exception:
            continue
        out.append((rr, cc, rank))
    return out


def _save_topdown_rgb_with_rank_labels(path: Path, rgb: np.ndarray, markers_rc: Sequence[Tuple[int, int, int]]) -> None:
    """保存俯视图；在物体 top-k 圆点旁标注序号（BGR 下绘制）。"""
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    for rr, cc, rk in markers_rc:
        label = str(int(rk))
        x = int(cc) + 6
        y = int(rr) + 6
        cv2.putText(bgr, label, (x, y), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, label, (x, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), bgr)


def _write_bev_instance_mask_png(
    pc: np.ndarray,
    mask_col: np.ndarray,
    out_path: Path,
    *,
    img_size: int = 640,
    max_draw_pts: int = 120000,
) -> None:
    """合并点云 ``pc`` (N×6, xyz+rgb) 上物体列 ``mask_col`` 的 3D 实例 mask → XZ 俯视图 PNG（白=前景点）。"""
    h = int(img_size)
    canvas = np.zeros((h, h, 3), dtype=np.uint8)
    if pc.ndim != 2 or pc.shape[1] < 3:
        cv2.putText(canvas, "invalid pc", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return
    pts = np.asarray(pc[:, :3], dtype=np.float64)
    m = (np.asarray(mask_col).reshape(-1) > 0.5).astype(bool)
    if pts.shape[0] != m.shape[0]:
        cv2.putText(canvas, "pc/mask len mismatch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return
    if not np.any(m):
        cv2.putText(canvas, "empty mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return
    xs = pts[m, 0]
    zs = pts[m, 2]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())
    span_x = max(max_x - min_x, 0.05)
    span_z = max(max_z - min_z, 0.05)
    pad_x = span_x * 0.06
    pad_z = span_z * 0.06
    min_x -= pad_x
    max_x += pad_x
    min_z -= pad_z
    max_z += pad_z
    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    idx = np.flatnonzero(m)
    if idx.size > int(max_draw_pts):
        rng = np.random.RandomState(12345)
        idx = rng.choice(idx, size=int(max_draw_pts), replace=False)
    for i in idx:
        x, _, z = pts[int(i)]
        u = int((x - min_x) / span_x * (h - 1))
        v = int((z - min_z) / span_z * (h - 1))
        u = int(np.clip(u, 0, h - 1))
        v = int(np.clip(v, 0, h - 1))
        row = h - 1 - v
        cv2.circle(canvas, (u, row), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _export_object_candidates_topk(
    dec_dir: Path,
    rep: Any,
    stage2_path: Path,
    topk: int = 5,
) -> List[Dict[str, Any]]:
    """按 og3d_logit 导出 top-k：首次检测图 + BEV mask；返回写入 ``object_candidates_topk/index.json`` 的列表。"""
    out: List[Dict[str, Any]] = []
    if not stage2_path.is_file():
        return out
    try:
        with open(stage2_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out
    objs = data.get("object_candidates") or []
    if not objs:
        return out
    kk = max(1, int(topk))
    ranked = sorted(objs, key=lambda x: float(x.get("og3d_logit", -1e30)), reverse=True)[:kk]
    top_dir = _ensure_dir(dec_dir / "object_candidates_topk")
    n_m = int(rep.object_mask.shape[1]) if getattr(rep, "object_mask", None) is not None else 0
    n_rgb = len(getattr(rep, "object_first_rgb", []) or [])
    for rank, o in enumerate(ranked, start=1):
        slot = int(o.get("slot_index", -1))
        stem = f"rank{rank:02d}_slot{slot:03d}"
        rec: Dict[str, Any] = {
            "rank_by_og3d_logit": rank,
            "slot_index": slot,
            "og3d_logit": o.get("og3d_logit"),
            "merged_object_score": o.get("merged_object_score"),
            "center_habitat_xyz": o.get("center_habitat_xyz"),
            "files": {},
        }
        det_path = top_dir / f"{stem}_first_detection_rgb.jpg"
        mask_path = top_dir / f"{stem}_mask_bev_xz.png"
        if 0 <= slot < n_rgb and rep.object_first_rgb[slot] is not None:
            try:
                cv2.imwrite(str(det_path), cv2.cvtColor(np.ascontiguousarray(rep.object_first_rgb[slot]), cv2.COLOR_RGB2BGR))
                rec["files"]["first_detection_rgb"] = str(det_path)
            except Exception:
                rec["files"]["first_detection_rgb"] = None
        else:
            rec["files"]["first_detection_rgb"] = None
        if 0 <= slot < n_m and rep.object_mask.shape[0] > 0:
            try:
                _write_bev_instance_mask_png(rep.point_cloud, rep.object_mask[:, slot], mask_path)
                rec["files"]["mask_bev_xz"] = str(mask_path)
            except Exception:
                rec["files"]["mask_bev_xz"] = None
        else:
            rec["files"]["mask_bev_xz"] = None
        out.append(rec)
    with open(top_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline PQ3D 单案例全过程分析")
    parser.add_argument("--scene_name", type=str, default="00802-wcojb4TFT35")
    parser.add_argument("--episode_id", type=int, default=17)
    parser.add_argument("--task_id", type=int, default=0, help="episode task_sequence 中的下标")
    parser.add_argument("--task_level", type=str, default="instance", choices=["object", "room", "region", "instance"])
    parser.add_argument("--concise_description", action="store_true")
    parser.add_argument(
        "--navigation_data_path",
        type=str,
        default="LangMap_Annotations",
        help="RefHM3D 序列标注根目录（其下含 <scene>.json.gz）",
    )
    parser.add_argument("--hm3d_data_base_path", type=str, default="/home/chenlin/krona/MTU3D/datascene")
    parser.add_argument(
        "--pq3d_stage1_path",
        type=str,
        default="/home/chenlin/krona/MTU3D/checkpoint/stage1-pretrain-all",
    )
    parser.add_argument(
        "--pq3d_stage2_path",
        type=str,
        default="/home/chenlin/krona/MTU3D/checkpoint/stage2-fine-tune-goat",
    )
    parser.add_argument("--output_log_dir", type=str, default="./output_logs/baseline_analysis")
    parser.add_argument("--sim_config", type=str, default="configs/habitat/goat_sim_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/habitat/goat_agent_config.yaml")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--decision_num_min", type=int, default=3)
    parser.add_argument("--visible_radius", type=float, default=3.0)
    parser.add_argument(
        "--panorama_subsample_frames",
        type=int,
        default=12,
        help="与 vfv 脚本一致：对环视帧均匀抽帧后再 stitch；默认 12 即不抽帧",
    )
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    project_root = _PROJECT_ROOT
    run_tag = args.run_tag or _now_tag()
    out_base = _ensure_dir(Path(args.output_log_dir).expanduser().resolve() / run_tag)

    scene_file = (project_root / args.navigation_data_path / f"{args.scene_name}.json.gz").resolve()
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing scene annotation: {scene_file}")

    with gzip.open(scene_file, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    region_to_annot_dict = scene_data["region_annotation"]
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    eps_list = [e for e in scene_data["episode_by_sequence"] if int(e["episode_id"]) == int(args.episode_id)]
    if not eps_list:
        raise ValueError(f"episode_id={args.episode_id} not found in {scene_file}")
    cur_episode = eps_list[0]
    navigation_type = cur_episode["navigation_type"]
    episode_id = cur_episode["episode_id"]
    task_sequence = cur_episode["task_sequence"]
    tid = int(args.task_id)
    if tid < 0 or tid >= len(task_sequence):
        raise ValueError(f"task_id={tid} out of range [0, {len(task_sequence)})")
    task_type, task_idx = task_sequence[tid]
    if task_type != args.task_level:
        raise ValueError(f"task_sequence[{tid}] has level={task_type}, expected --task_level={args.task_level}")

    cur_task = episode_mapping[task_type][task_idx]
    goals_ids = cur_task["target_object_ids"]
    goals = [all_navigation_goals_dict[x] for x in goals_ids]
    if task_type == "object":
        sentence = cur_task["object_category"]
        goal_category = cur_task["object_category"]
    elif task_type == "room":
        sentence = f"{cur_task['object_category']} in the {cur_task['room_name'].lower()}"
        goal_category = cur_task["object_category"]
    elif task_type == "region":
        region_desc = (
            region_to_annot_dict[cur_task["region_id"]]["concise_description"]
            if args.concise_description
            else region_to_annot_dict[cur_task["region_id"]]["detailed_description"]
        )
        sentence = (
            f"{cur_task['object_category']} in the {region_to_annot_dict[cur_task['region_id']]['region_category'].lower()} "
            f"that has {region_desc}"
        )
        goal_category = cur_task["object_category"]
    else:
        sentence = (
            all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"]
            if args.concise_description
            else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
        )
        goal_category = goals[0]["object_category"]

    out_task = _ensure_dir(
        out_base / f"scene={args.scene_name}" / f"episode={episode_id}" / f"task={tid}"
    )
    print(
        f"[baseline-analysis] scene={args.scene_name} episode={episode_id} task={tid} "
        f"level={task_type} -> {out_task}"
    )

    sim_settings = OmegaConf.load(str((project_root / args.sim_config).resolve()))
    goat_agent_setting = OmegaConf.load(str((project_root / args.agent_config).resolve()))
    sim_settings["scene"] = resolve_scene_path(
        str((project_root / Path(args.hm3d_data_base_path).expanduser()).resolve()),
        args.scene_name,
    )
    abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    agent_state = habitat_sim.AgentState()
    agent_state.position = cur_episode["start_position"]
    agent_state.rotation = cur_episode["start_rotation"]
    agent.set_state(agent_state)
    path_finder = sim.pathfinder

    map_resolution = 512
    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
    fog_of_war_mask = np.zeros_like(top_down_map)
    area_thres_in_pixels = convert_meters_to_pixel(9, map_resolution, sim)
    visibility_dist_in_pixels = convert_meters_to_pixel(args.visible_radius, map_resolution, sim)

    pq3d_model = PQ3DModel(
        str(Path(args.pq3d_stage1_path).expanduser().resolve()),
        str(Path(args.pq3d_stage2_path).expanduser().resolve()),
        min_decision_num=args.decision_num_min,
    )
    pq3d_model.reset()

    visited_frontier_set: set = set()
    decision_records: List[Dict[str, Any]] = []
    total_steps = 0
    rotation_steps = 0
    prev_agent_state = agent.get_state()
    sub_episode_start_position = prev_agent_state.position
    episode_cum_distance = 0.0
    goto_color_list: List[np.ndarray] = []
    goto_depth_list: List[np.ndarray] = []
    goto_agent_state_list: List[Any] = []

    while total_steps < args.max_steps:
        color_list: List[np.ndarray] = []
        depth_list: List[np.ndarray] = []
        agent_state_list: List[Any] = []
        if len(goto_color_list) > 6:
            goto_color_list = [
                goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)
            ][:6]
            goto_depth_list = [
                goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)
            ][:6]
            goto_agent_state_list = [
                goto_agent_state_list[i]
                for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)
            ][:6]
        color_list.extend(goto_color_list)
        depth_list.extend(goto_depth_list)
        agent_state_list.extend(goto_agent_state_list)

        for _ in range(12):
            obervations = sim.step(action="turn_left")
            color = obervations["color_sensor"][:, :, :3]
            color_list.append(color)
            depth = obervations["depth_sensor"][:, :]
            depth_list.append(depth)
            agent_state = agent.get_state()
            agent_state_list.append(agent_state)
            fog_of_war_mask = reveal_fog_of_war(
                top_down_map=top_down_map,
                current_fog_of_war_mask=fog_of_war_mask,
                current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                current_angle=get_polar_angle(agent_state),
                fov=42,
                max_line_len=visibility_dist_in_pixels,
                enable_debug_visualization=False,
            )
            total_steps += 1
            rotation_steps += 1

        agent_state = agent.get_state()
        frontier_pixels = detect_frontier_waypoints(
            top_down_map,
            fog_of_war_mask,
            area_thres_in_pixels,
            xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
            enable_visualization=False,
        )
        if len(frontier_pixels) == 0:
            frontier_waypoints = []
        else:
            fw = frontier_pixels[:, ::-1]
            frontier_waypoints = pixel_to_map_coors(fw, agent_state.position, top_down_map, sim)
        frontier_waypoints = [
            waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set
        ]

        dec_dir = out_task / f"dec_{len(decision_records):03d}"
        _ensure_dir(dec_dir)
        analysis_dir = str(dec_dir.resolve())

        agent_rc = map_coors_to_pixel(agent_state.position, top_down_map, sim)
        frontier_rc: List[Tuple[int, int]] = []
        for wp in frontier_waypoints:
            frontier_rc.append(_habitat_xyz_to_map_rc(wp, top_down_map, sim))
        visited_rc_list = []
        for t in visited_frontier_set:
            arr = np.asarray(t, dtype=float).reshape(3)
            visited_rc_list.append(_habitat_xyz_to_map_rc(arr, top_down_map, sim))

        # 与 refhm3d-nav-sequence-analyze-anchor-vfv.py 一致：reverse → 可选抽帧 → stitch_panorama（默认参数）
        if stitch_panorama is not None and _save_rgb_jpg is not None and len(color_list) >= 12:
            try:
                verify_rgb = list(color_list[-12:])
                verify_rgb = list(reversed(verify_rgb))
                if int(args.panorama_subsample_frames) < len(verify_rgb):
                    step = max(1, len(verify_rgb) // int(args.panorama_subsample_frames))
                    verify_rgb = [verify_rgb[i] for i in range(0, len(verify_rgb), step)][: int(args.panorama_subsample_frames)]
                pano = stitch_panorama(verify_rgb)
                _save_rgb_jpg(pano, dec_dir / "panorama_stitched_last12.jpg")
            except Exception as ex:
                print(f"[baseline-analysis] stitch panorama skipped: {ex}")

        target_position, is_final_decision = pq3d_model.decision(
            color_list,
            depth_list,
            agent_state_list,
            frontier_waypoints,
            sentence,
            len(decision_records),
            analysis_output_dir=analysis_dir,
        )

        stage2_json = dec_dir / "stage2_decision.json"
        object_rank_rc = _object_topk_rc_from_stage2_json(
            stage2_json, top_down_map=top_down_map, sim=sim, k=int(args.object_topk)
        )

        td_before = _render_topdown(
            top_down_map=top_down_map,
            fog_mask=fog_of_war_mask,
            agent_rc=agent_rc,
            frontier_rc_list=frontier_rc,
            chosen_rc=None,
            visited_rc_list=visited_rc_list,
            object_rank_markers_rc=object_rank_rc,
        )
        _save_topdown_rgb_with_rank_labels(dec_dir / "topdown_before_decision.jpg", td_before, object_rank_rc)

        aux = dict(pq3d_model.last_decision_aux)
        chosen_rc = _habitat_xyz_to_map_rc(target_position, top_down_map, sim)
        td_after = _render_topdown(
            top_down_map=top_down_map,
            fog_mask=fog_of_war_mask,
            agent_rc=agent_rc,
            frontier_rc_list=frontier_rc,
            chosen_rc=chosen_rc,
            visited_rc_list=visited_rc_list,
            object_rank_markers_rc=object_rank_rc,
        )
        _save_topdown_rgb_with_rank_labels(dec_dir / "topdown_with_chosen_target.jpg", td_after, object_rank_rc)

        mem_dir = dec_dir / "memory_object_first_rgb"
        _ensure_dir(mem_dir)
        for si, snap in enumerate(pq3d_model.representation_manager.object_first_rgb[:64]):
            if snap is None:
                continue
            try:
                cv2.imwrite(str(mem_dir / f"slot_{si:03d}.jpg"), cv2.cvtColor(snap, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

        object_topk_export = _export_object_candidates_topk(
            dec_dir,
            pq3d_model.representation_manager,
            stage2_json,
            topk=int(args.object_topk),
        )

        rec = {
            "decision_index": len(decision_records),
            "pq3d_last_decision_aux": aux,
            "is_final_decision": bool(is_final_decision),
            "target_habitat_xyz": [float(x) for x in np.asarray(target_position).reshape(3)],
            "n_scan_views": len(color_list),
            "n_frontiers_input": len(frontier_waypoints),
            "object_topk": int(args.object_topk),
            "object_topk_candidates": object_topk_export,
            "topdown_legend": {
                "object_topk": (
                    f"按 stage2 og3d_logit 前 {int(args.object_topk)}；"
                    "1=金 2=橙 3=紫 4=青 5=绿；旁注数字"
                ),
                "agent": "红白圆点",
                "frontier": "绿点",
                "chosen_target": "蓝点",
                "visited_frontier": "灰点",
            },
            "files": {
                "topdown_before": str(dec_dir / "topdown_before_decision.jpg"),
                "topdown_chosen": str(dec_dir / "topdown_with_chosen_target.jpg"),
                "stage2_json": aux.get("analysis_stage2_json"),
                "memory_thumbnails_dir": str(mem_dir),
                "object_candidates_topk_index": str(dec_dir / "object_candidates_topk" / "index.json"),
            },
        }
        decision_records.append(rec)
        with open(dec_dir / "decision_step_summary.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        if not is_final_decision:
            visited_frontier_set.add(tuple(np.round(target_position, 1)))

        agent_island = path_finder.get_island(agent_state.position)
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
        )
        try:
            action_list = follower.find_path(target_on_navmesh)
        except Exception:
            action_list = []
        if not action_list:
            print("[baseline-analysis] path planning failed, stopping.")
            break

        goto_color_list = []
        goto_depth_list = []
        goto_agent_state_list = []
        for action in action_list:
            if action:
                obervations = sim.step(action=action)
                agent_state = agent.get_state()
                color = obervations["color_sensor"][:, :, :3]
                depth = obervations["depth_sensor"][:, :]
                goto_color_list.append(color)
                goto_depth_list.append(depth)
                goto_agent_state_list.append(agent_state)
                fog_of_war_mask = reveal_fog_of_war(
                    top_down_map=top_down_map,
                    current_fog_of_war_mask=fog_of_war_mask,
                    current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim),
                    current_angle=get_polar_angle(agent_state),
                    fov=42,
                    max_line_len=visibility_dist_in_pixels,
                    enable_debug_visualization=False,
                )
                total_steps += 1
                if action in ["turn_left", "turn_right"]:
                    rotation_steps += 1
                episode_cum_distance += float(np.linalg.norm(agent_state.position - prev_agent_state.position))
                prev_agent_state = agent_state
        if is_final_decision:
            break

    agent_state = agent.get_state()
    view_points = [view_point["agent_state"]["position"] for goal in goals for view_point in goal["view_points"]]
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = sub_episode_start_position
    path.requested_ends = view_points
    if path_finder.find_path(path):
        start_end_geo_distance = path.geodesic_distance
    else:
        start_end_geo_distance = np.inf
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = agent_state.position
    path.requested_ends = view_points
    if path_finder.find_path(path):
        agent_end_geo_distance = path.geodesic_distance
    else:
        agent_end_geo_distance = np.inf

    if start_end_geo_distance == np.inf:
        sr, spl = 0, 0
    elif agent_end_geo_distance == np.inf:
        sr, spl = 0, 0
    else:
        sr = bool(agent_end_geo_distance <= 0.25)
        spl = float(sr) * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

    summary = {
        "scene_name": args.scene_name,
        "episode_id": episode_id,
        "task_id": tid,
        "task_level": task_type,
        "navigation_type": navigation_type,
        "sentence": sentence,
        "goal_category": goal_category,
        "sr": int(sr),
        "spl": spl,
        "total_steps": total_steps,
        "rotation_steps": rotation_steps,
        "episode_cum_distance": episode_cum_distance,
        "num_decisions": len(decision_records),
        "decisions": decision_records,
    }
    with open(out_task / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sim.close()
    print(
        f"[baseline-analysis] done SR={sr} SPL={spl:.4f} decisions={len(decision_records)} "
        f"summary={out_task / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
