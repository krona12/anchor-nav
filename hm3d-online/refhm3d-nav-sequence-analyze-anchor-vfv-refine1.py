"""RefHM3D：VFV refine1 批跑（全景验证 + 可选 PQ3D 锚点重选）。

与 ``anchor_nav.vfv`` 一致：可用作 PQ3D 锚点的约束写在
``decompose_target_anchor`` 的 VLM prompt 中；本脚本仅串联
``decompose_target_anchor`` / ``verify_description_visible`` /
``select_best_anchor_object``，不再对锚点做额外汇编语义过滤
（库内仅规范化去重，见 ``dedupe_anchor_phrases``）。
"""
import argparse
import atexit
import datetime
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat.utils.visualizations import maps
from omegaconf import OmegaConf
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)


def _tqdm_print(msg: str) -> None:
    """经 tqdm 写出，避免与 progress bar 同一行拼接（见 tqdm.write 文档）。"""
    tqdm.write(msg, file=sys.stdout)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anchor_nav.posnode import _save_rgb_jpg, stitch_panorama
from anchor_nav.vfv import (
    build_query_fn_from_pq3d_stage2,
    decompose_target_anchor,
    select_best_anchor_object,
    verify_description_visible,
)
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
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL


def resolve_scene_path(hm3d_root: str, scene_name: str) -> str:
    short_scene_name = scene_name.split("-")[-1]
    scene_dir = Path(hm3d_root) / scene_name
    candidates = [scene_dir / f"{short_scene_name}.basis.glb", scene_dir / f"{short_scene_name}.glb"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Scene asset not found for {scene_name}")


def sequence_compute_metric_results(result_dict: dict) -> None:
    sequence_results = result_dict.get("sequence", [])
    total_count = len(sequence_results)
    if total_count == 0:
        _tqdm_print("[Metrics] sequence count: 0")
        return
    total_sr = sum(float(item.get("sr", 0)) for item in sequence_results)
    total_spl = sum(float(item.get("spl", 0)) for item in sequence_results)
    total_task_time = sum(float(item.get("task_time_sec", 0.0)) for item in sequence_results)
    _tqdm_print(
        f"[Metrics] sequence count={total_count}, avg_sr={total_sr/total_count:.6f}, "
        f"avg_spl={total_spl/total_count:.6f}, avg_task_time_sec={total_task_time/total_count:.3f}"
    )


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-vfv-refine1-{ts}-pid{os.getpid()}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[VFVRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[VFVRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def _capture_scan_frames(
    *,
    sim: Any,
    agent: Any,
    top_down_map: np.ndarray,
    fog_of_war_mask: np.ndarray,
    visibility_dist_in_pixels: int,
    total_steps: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], np.ndarray, int]:
    scan_rgb, scan_depth, scan_state = [], [], []
    for _ in range(12):
        obs = sim.step(action="turn_left")
        agent_state = agent.get_state()
        rgb = obs["color_sensor"][:, :, :3]
        dep = obs["depth_sensor"][:, :]
        scan_rgb.append(rgb)
        scan_depth.append(dep)
        scan_state.append(agent_state)
        fog_of_war_mask = reveal_fog_of_war(
            top_down_map,
            fog_of_war_mask,
            map_coors_to_pixel(agent_state.position, top_down_map, sim),
            get_polar_angle(agent_state),
            42,
            visibility_dist_in_pixels,
            False,
        )
        total_steps += 1
        if total_steps >= int(max_steps):
            break
    return scan_rgb, scan_depth, scan_state, fog_of_war_mask, total_steps


def _follow_target(
    *,
    path_finder: Any,
    agent: Any,
    sim: Any,
    target: np.ndarray,
    prev_agent_state: Any,
    total_steps: int,
    max_steps: int,
    episode_cum_distance: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], Any, int, float]:
    agent_island = path_finder.get_island(agent.get_state().position)
    target_on_navmesh = path_finder.snap_point(point=target, island_index=agent_island)
    follower = habitat_sim.GreedyGeodesicFollower(
        path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right"
    )
    try:
        action_list = follower.find_path(target_on_navmesh)
    except Exception:
        action_list = []
    goto_color_list, goto_depth_list, goto_state_list = [], [], []
    for action in action_list:
        if not action:
            continue
        obs = sim.step(action=action)
        state = agent.get_state()
        goto_color_list.append(obs["color_sensor"][:, :, :3])
        goto_depth_list.append(obs["depth_sensor"][:, :])
        goto_state_list.append(state)
        total_steps += 1
        episode_cum_distance += np.linalg.norm(state.position - prev_agent_state.position)
        prev_agent_state = state
        if total_steps >= int(max_steps):
            break
    return goto_color_list, goto_depth_list, goto_state_list, prev_agent_state, total_steps, float(episode_cum_distance)


parser = argparse.ArgumentParser(
    description=(
        "Run RefHM3D anchor VFV refine1 batch evaluation. "
        "Anchor eligibility is enforced by the VLM prompt in anchor_nav.vfv.decompose_target_anchor."
    )
)
parser.add_argument("--start_ratio", type=float, default=0.0)
parser.add_argument("--end_ratio", type=float, default=0.2)
parser.add_argument("--concise_description", action="store_true")
parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/vfv"))
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--vfv_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
parser.add_argument("--vfv_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
parser.add_argument("--anchor_top_k", type=int, default=16)
parser.add_argument("--panorama_subsample_frames", type=int, default=12)
parser.add_argument(
    "--vfv_skip_verify_if_object_logit_gap",
    type=float,
    default=-1.0,
    help=">=0 时：若 PQ3D 对物体 top1/top2 的 logit 差 >= 该值则跳过全景+VLM（省步数）。默认 -1 关闭；建议从 0.2~0.5 试起（见 exp_vfv_analysis.md）",
)
parser.add_argument(
    "--vfv_min_remaining_steps",
    type=int,
    default=0,
    help="Phase1 跟目标点后若剩余步数 < 该值则跳过全景验证与 Phase2；建议序列任务设 50~80，0 关闭",
)
parser.add_argument(
    "--vfv_verify_parse_attempts",
    type=int,
    default=2,
    help="全景验证 JSON 解析失败时的最大重试次数",
)
parser.add_argument(
    "--quiet_nav_steps",
    action="store_true",
    help="不打印每次 decision 的 scan/frontier/pq3d 耗时行（日志更短；卡住时勿开此项）。",
)
parser.add_argument(
    "--decision_log_interval",
    type=int,
    default=0,
    help="每隔 N 次 decision 打印一次明细；<=0 表示关闭明细",
)
args = parser.parse_args()

if args.vfv_api_key:
    os.environ["ZZZ_API_KEY"] = args.vfv_api_key

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)
print(
    f"[VFVRefine1] cfg anchor_top_k={args.anchor_top_k} pano_frames={args.panorama_subsample_frames} "
    f"skip_verify_if_gap>={args.vfv_skip_verify_if_object_logit_gap} "
    f"min_remaining_steps={args.vfv_min_remaining_steps} verify_parse_attempts={args.vfv_verify_parse_attempts} "
    f"quiet_nav_steps={bool(args.quiet_nav_steps)}"
)

enabled_task_levels = {"instance"}
success_distance = 0.25
decision_num_min = 3
visible_radius = 3

navigation_data_root = Path(os.path.expanduser(args.navigation_data_path))
scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]
os.makedirs(output_log_dir, exist_ok=True)
out_name = f"refhm3d_seq_vfv_refine1_{args.start_ratio}_{args.end_ratio}.json"
eff_name = f"refhm3d_seq_vfv_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
if args.concise_description:
    out_name = f"refhm3d_seq_vfv_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_vfv_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
output_path = os.path.join(output_log_dir, out_name)
effectiveness_path = os.path.join(output_log_dir, eff_name)

if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    existing_episodes = {"_".join([r["scene_name"], r["navigation_type"], str(r["episode_id"])]) for k in result_dict for r in result_dict[k]}
else:
    result_dict = {"sequence": []}
    existing_episodes = set()

if os.path.exists(effectiveness_path):
    effectiveness_dict = json.load(open(effectiveness_path, "r"))
else:
    effectiveness_dict = {"records": []}

pq3d_model = PQ3DModel(
    os.path.expanduser(args.pq3d_stage1_path),
    os.path.expanduser(args.pq3d_stage2_path),
    min_decision_num=decision_num_min,
)

for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
    scene_name = scene_data_path.name.split(".")[0]
    with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    episode_mapping = {
        "object": scene_data["episodes_by_object_level"],
        "room": scene_data["episodes_by_room_level"],
        "region": scene_data["episodes_by_region_level"],
        "instance": scene_data["episodes_by_instance_level"],
    }
    all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
        pq3d_model.reset()
        query_fn = build_query_fn_from_pq3d_stage2(pq3d_model)
        decision_num = 0
        visited_frontier_set = set()
        start_position = cur_episode["start_position"]
        start_rotation = cur_episode["start_rotation"]
        episode_id, navigation_type = cur_episode["episode_id"], cur_episode["navigation_type"]
        episode_key = "_".join([scene_name, navigation_type, str(episode_id)])
        if episode_key in existing_episodes:
            continue

        sim_settings = OmegaConf.load("configs/habitat/goat_sim_config.yaml")
        goat_agent_setting = OmegaConf.load("configs/habitat/goat_agent_config.yaml")
        sim_settings["scene"] = resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), scene_name)
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=512, draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels = convert_meters_to_pixel(9, 512, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, 512, sim)
        out_episode_dir = Path(output_log_dir) / "process" / f"scene={scene_name}" / f"episode={episode_id}"
        out_episode_dir.mkdir(parents=True, exist_ok=True)

        for idx, cur_task in enumerate(cur_episode["task_sequence"]):
            task_t0 = time.perf_counter()
            task_type, task_idx = cur_task
            if task_type not in enabled_task_levels:
                continue
            cur_task = episode_mapping[task_type][task_idx]
            goals = [all_navigation_goals_dict[x] for x in cur_task["target_object_ids"]]
            goal_positions = [
                np.asarray(g.get("position", []), dtype=float).reshape(3)
                for g in goals
                if isinstance(g, dict) and len(g.get("position", [])) >= 3
            ]
            original_sentence = (
                all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"]
                if args.concise_description
                else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
            )
            decomp_t0 = time.perf_counter()
            decomp = decompose_target_anchor(original_sentence, args.vfv_vlm_model)
            decomp_ms = (time.perf_counter() - decomp_t0) * 1000.0
            goal_category = goals[0]["object_category"]
            _tqdm_print(
                f"[vfv-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} level={task_type} "
                f"target_desc={decomp.target_desc!r} anchor_desc={decomp.anchor_desc!r} "
                f"anchor_descs={decomp.anchor_descs!r} "
                f"decomp_parse_ok={decomp.parse_ok} decomp_ms={decomp_ms:.1f}"
            )
            if not decomp.parse_ok:
                _tqdm_print(
                    f"[vfv-refine1][task-start][warn] decompose VLM parse failed, using description fallback as target_desc"
                )
            _tqdm_print(f"[vfv-refine1][task-desc] {original_sentence}")
            _tqdm_print(
                f"[vfv-refine1][nav] task={idx} entering navigation loop; each step = 12×turn scan + "
                f"frontier + PQ3D (first GPU forward can take several minutes, not a hang)."
            )

            total_steps = 0
            episode_cum_distance = 0.0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            vfv_attempts = 0
            vfv_applied = 0
            vfv_vlm_elapsed_ms_total = float(decomp_ms)
            baseline_final_target_pos: Optional[np.ndarray] = None
            final_selected_object_pos: Optional[np.ndarray] = None
            task_effective_logs: List[Dict[str, Any]] = []
            task_pq3d_object_gap: Optional[float] = None
            task_vfv_verify_skipped = False

            task_dir = out_episode_dir / f"task={idx}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pano_dir = task_dir / "panorama"
            pano_dir.mkdir(parents=True, exist_ok=True)

            while total_steps < int(args.max_steps):
                color_list, depth_list, agent_state_list = [], [], []
                if len(goto_color_list) > 6:
                    step = max(1, len(goto_color_list) // 6)
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
                color_list.extend(goto_color_list)
                depth_list.extend(goto_depth_list)
                agent_state_list.extend(goto_agent_state_list)

                t_scan = time.perf_counter()
                scan_rgb, scan_depth, scan_states, fog_of_war_mask, total_steps = _capture_scan_frames(
                    sim=sim,
                    agent=agent,
                    top_down_map=top_down_map,
                    fog_of_war_mask=fog_of_war_mask,
                    visibility_dist_in_pixels=visibility_dist_in_pixels,
                    total_steps=total_steps,
                    max_steps=int(args.max_steps),
                )
                scan_ms = (time.perf_counter() - t_scan) * 1000.0
                color_list.extend(scan_rgb)
                depth_list.extend(scan_depth)
                agent_state_list.extend(scan_states)
                if total_steps >= int(args.max_steps):
                    break

                t_frontier = time.perf_counter()
                agent_state = agent.get_state()
                frontier_waypoints = detect_frontier_waypoints(
                    top_down_map,
                    fog_of_war_mask,
                    area_thres_in_pixels,
                    xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1],
                    enable_visualization=False,
                )
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints[:, ::-1], agent_state.position, top_down_map, sim)
                frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]
                frontier_ms = (time.perf_counter() - t_frontier) * 1000.0

                t_pq = time.perf_counter()
                target_position, is_final = pq3d_model.decision(
                    color_list, depth_list, agent_state_list, frontier_waypoints, original_sentence, decision_num
                )
                pq_ms = (time.perf_counter() - t_pq) * 1000.0
                if not bool(args.quiet_nav_steps):
                    _tqdm_print(
                        f"[vfv-refine1][step] task={idx} dec={decision_num} "
                        f"scan_ms={scan_ms:.0f} frontier_ms={frontier_ms:.0f} pq3d_ms={pq_ms:.0f} "
                        f"frames={len(color_list)} frontiers={len(frontier_waypoints)} final={bool(is_final)}"
                    )
                if int(args.decision_log_interval) > 0 and (decision_num % int(args.decision_log_interval) == 0):
                    _tqdm_print(
                        f"[vfv-refine1][decision] task={idx} dec={decision_num} frontiers={len(frontier_waypoints)} "
                        f"baseline_target={np.asarray(target_position, dtype=float).reshape(-1)[:3].tolist()} final={bool(is_final)}"
                    )

                used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                vfv_info: Dict[str, Any] = {"ok": False, "phase": "phase1", "triggered": False}
                if is_final:
                    baseline_final_target_pos = used_target.copy()
                    aux = getattr(pq3d_model, "last_decision_aux", {}) or {}
                    gap = float(aux.get("object_top1_top2_logit_gap", 0.0))
                    task_pq3d_object_gap = gap

                    goto_color_list, goto_depth_list, goto_agent_state_list, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                        path_finder=path_finder,
                        agent=agent,
                        sim=sim,
                        target=used_target,
                        prev_agent_state=prev_agent_state,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                        episode_cum_distance=float(episode_cum_distance),
                    )

                    remaining_after_p1 = int(args.max_steps) - int(total_steps)
                    skip_gap = (
                        float(args.vfv_skip_verify_if_object_logit_gap) >= 0.0
                        and gap >= float(args.vfv_skip_verify_if_object_logit_gap)
                    )
                    skip_steps = (
                        int(args.vfv_min_remaining_steps) > 0
                        and remaining_after_p1 < int(args.vfv_min_remaining_steps)
                    )
                    skip_verify = skip_gap or skip_steps

                    if skip_verify:
                        task_vfv_verify_skipped = True
                        verify_ms = 0.0
                        verify = {
                            "visible": True,
                            "skipped": True,
                            "skip_reason": "object_logit_gap" if skip_gap else "remaining_steps",
                            "object_top1_top2_logit_gap": gap,
                            "remaining_steps_after_phase1_follow": remaining_after_p1,
                            "full_match": True,
                            "strong_anchor_match": False,
                            "confidence": "high",
                            "reason": "vfv_verify_skipped_deployable",
                            "raw": "",
                            "parse_attempts": 0,
                        }
                        vfv_info = {
                            "ok": True,
                            "phase": "phase1",
                            "triggered": False,
                            "verify_skipped": True,
                            "phase1_target": used_target.tolist(),
                            "verify": verify,
                            "verify_elapsed_ms": 0.0,
                            "panorama_path": None,
                            "decompose": {
                                "target_desc": decomp.target_desc,
                                "anchor_desc": decomp.anchor_desc,
                                "anchor_descs": list(decomp.anchor_descs),
                                "parse_ok": bool(decomp.parse_ok),
                            },
                        }
                        used_target = baseline_final_target_pos.copy()
                    else:
                        vfv_attempts += 1
                        verify_rgb, _, _, fog_of_war_mask, total_steps = _capture_scan_frames(
                            sim=sim,
                            agent=agent,
                            top_down_map=top_down_map,
                            fog_of_war_mask=fog_of_war_mask,
                            visibility_dist_in_pixels=visibility_dist_in_pixels,
                            total_steps=total_steps,
                            max_steps=int(args.max_steps),
                        )
                        verify_rgb = list(reversed(verify_rgb))
                        if int(args.panorama_subsample_frames) < len(verify_rgb):
                            step = max(1, len(verify_rgb) // int(args.panorama_subsample_frames))
                            verify_rgb = [verify_rgb[i] for i in range(0, len(verify_rgb), step)][: int(args.panorama_subsample_frames)]
                        pano = stitch_panorama(verify_rgb)
                        pano_path = pano_dir / f"vfv_dec_{int(decision_num):03d}.jpg"
                        _save_rgb_jpg(pano, pano_path)

                        t_verify = time.perf_counter()
                        verify = verify_description_visible(
                            description=original_sentence,
                            image_path=str(pano_path),
                            vlm_model=args.vfv_vlm_model,
                            target_desc=decomp.target_desc,
                            anchor_hints=decomp.anchor_descs,
                            max_parse_attempts=int(args.vfv_verify_parse_attempts),
                        )
                        verify_ms = (time.perf_counter() - t_verify) * 1000.0
                        vfv_vlm_elapsed_ms_total += verify_ms
                        vfv_info = {
                            "ok": True,
                            "phase": "phase1",
                            "triggered": bool(not verify.get("visible", False)),
                            "verify_skipped": False,
                            "phase1_target": used_target.tolist(),
                            "verify": verify,
                            "verify_elapsed_ms": float(verify_ms),
                            "panorama_path": str(pano_path),
                            "decompose": {
                                "target_desc": decomp.target_desc,
                                "anchor_desc": decomp.anchor_desc,
                                "anchor_descs": list(decomp.anchor_descs),
                                "parse_ok": bool(decomp.parse_ok),
                            },
                        }

                        if not bool(verify.get("visible", False)):
                            anchor_info = select_best_anchor_object(
                                anchor_descs=decomp.anchor_descs,
                                query_fn=query_fn,
                                rep=pq3d_model.representation_manager,
                                top_k=int(args.anchor_top_k),
                            )
                            vfv_info["phase"] = "phase2"
                            vfv_info["anchor_query"] = anchor_info
                            if bool(anchor_info.get("ok", False)):
                                used_target = np.asarray(anchor_info["anchor_position"], dtype=float).reshape(3).copy()
                                vfv_applied += 1
                                goto_color_list, goto_depth_list, goto_agent_state_list, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                                    path_finder=path_finder,
                                    agent=agent,
                                    sim=sim,
                                    target=used_target,
                                    prev_agent_state=prev_agent_state,
                                    total_steps=total_steps,
                                    max_steps=int(args.max_steps),
                                    episode_cum_distance=float(episode_cum_distance),
                                )
                            else:
                                used_target = baseline_final_target_pos.copy()
                        else:
                            used_target = baseline_final_target_pos.copy()

                    final_selected_object_pos = used_target.copy()
                    _tqdm_print(
                        f"[vfv-refine1][final] task={idx} dec={decision_num} "
                        f"triggered={vfv_info.get('triggered')} phase={vfv_info.get('phase')} "
                        f"skipped={vfv_info.get('verify_skipped')} applied={bool(vfv_applied>0)}"
                    )
                    task_effective_logs.append(
                        {
                            "scene_name": scene_name,
                            "episode_id": int(episode_id),
                            "task_id": int(idx),
                            "decision_num": int(decision_num),
                            "original_sentence": original_sentence,
                            "decompose": {
                                "target_desc": decomp.target_desc,
                                "anchor_desc": decomp.anchor_desc,
                                "anchor_descs": list(decomp.anchor_descs),
                                "parse_ok": bool(decomp.parse_ok),
                            },
                            "vfv_ok": True,
                            "vfv_info": vfv_info,
                        }
                    )
                else:
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))
                    goto_color_list, goto_depth_list, goto_agent_state_list, prev_agent_state, total_steps, episode_cum_distance = _follow_target(
                        path_finder=path_finder,
                        agent=agent,
                        sim=sim,
                        target=used_target,
                        prev_agent_state=prev_agent_state,
                        total_steps=total_steps,
                        max_steps=int(args.max_steps),
                        episode_cum_distance=float(episode_cum_distance),
                    )

                with open(task_dir / f"dec_{decision_num:03d}_vfv.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": int(idx),
                            "decision_num": int(decision_num),
                            "is_final": bool(is_final),
                            "target_used": used_target.tolist(),
                            "vfv": vfv_info,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                decision_num += 1
                if is_final:
                    break

            task_time = float(time.perf_counter() - task_t0)
            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
            sp = habitat_sim.MultiGoalShortestPath()
            sp.requested_start = sub_episode_start_position
            sp.requested_ends = view_points
            start_end_geo_distance = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
            ep = habitat_sim.MultiGoalShortestPath()
            ep.requested_start = agent_state.position
            ep.requested_ends = view_points
            agent_end_geo_distance = float(ep.geodesic_distance) if path_finder.find_path(ep) else float("inf")
            if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance):
                sr, spl = 0, 0
            else:
                sr = agent_end_geo_distance <= success_distance
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            baseline_target_to_goal_l2 = float("inf")
            selected_object_to_goal_l2 = float("inf")
            vfv_helpful = None
            if len(goal_positions) > 0:
                if baseline_final_target_pos is not None:
                    d0 = [float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions]
                    baseline_target_to_goal_l2 = float(min(d0))
                if final_selected_object_pos is not None:
                    d1 = [float(np.linalg.norm(final_selected_object_pos - gp)) for gp in goal_positions]
                    selected_object_to_goal_l2 = float(min(d1))
                has_vfv_applied = bool(int(vfv_applied) > 0)
                if has_vfv_applied and np.isfinite(baseline_target_to_goal_l2) and np.isfinite(selected_object_to_goal_l2):
                    vfv_helpful = bool(selected_object_to_goal_l2 < baseline_target_to_goal_l2 - 1e-6)
                else:
                    vfv_helpful = None

            result_dict.setdefault(navigation_type, []).append(
                {
                    "scene_name": scene_name,
                    "episode_id": episode_id,
                    "task_id": idx,
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "sr": sr,
                    "spl": spl,
                    "object_category": goal_category,
                    "task_time_sec": task_time,
                    "steps_total": int(total_steps),
                    "decisions": int(decision_num),
                    "start_goal_geo": float(start_end_geo_distance),
                    "end_goal_geo": float(agent_end_geo_distance),
                    "episode_cum_distance": float(episode_cum_distance),
                    "vfv_attempts": int(vfv_attempts),
                    "vfv_applied": int(vfv_applied),
                    "vfv_vlm_elapsed_ms_total": float(vfv_vlm_elapsed_ms_total),
                    "goal_positions": [gp.tolist() for gp in goal_positions],
                    "baseline_target_position": None if baseline_final_target_pos is None else baseline_final_target_pos.tolist(),
                    "selected_object_position": None if final_selected_object_pos is None else final_selected_object_pos.tolist(),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
                    "vfv_helpful": vfv_helpful,
                    "decompose_parse_ok": bool(decomp.parse_ok),
                    "pq3d_object_top1_top2_logit_gap": task_pq3d_object_gap,
                    "vfv_verify_skipped": bool(task_vfv_verify_skipped),
                }
            )
            effectiveness_dict["records"].append(
                {
                    "scene_name": scene_name,
                    "episode_id": int(episode_id),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "vfv_attempts": int(vfv_attempts),
                    "vfv_applied": int(vfv_applied),
                    "vfv_vlm_elapsed_ms_total": float(vfv_vlm_elapsed_ms_total),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
                    "vfv_helpful": vfv_helpful,
                    "task_effective_logs": task_effective_logs,
                }
            )
            if len(goal_positions) == 0:
                dist_bl_txt = "n/a(no_goal_xyz)"
            elif baseline_final_target_pos is None:
                dist_bl_txt = "n/a(no_Phase1_is_final)"
            else:
                dist_bl_txt = f"{baseline_target_to_goal_l2:.3f}"
            if len(goal_positions) == 0:
                dist_vfv_txt = "n/a(no_goal_xyz)"
            elif int(vfv_applied) <= 0:
                dist_vfv_txt = "n/a(vfv_unused)"
            elif np.isfinite(selected_object_to_goal_l2):
                dist_vfv_txt = f"{selected_object_to_goal_l2:.3f}"
            else:
                dist_vfv_txt = "n/a"

            _tqdm_print(
                f"[vfv-refine1] scene={scene_name} ep={episode_id} task={idx} SR={sr} SPL={spl:.4f} "
                f"time={task_time:.3f}s steps={total_steps} decisions={decision_num} "
                f"vfv_attempts={vfv_attempts} vfv_applied={vfv_applied} "
                f"dist(baseline_target,goal)={dist_bl_txt} "
                f"dist(vfv_object,goal)={dist_vfv_txt} helpful={vfv_helpful}"
            )
            if int(vfv_attempts) == 0 and baseline_final_target_pos is None and int(decision_num) > 0:
                _tqdm_print(
                    "[vfv-refine1][hint] 从未出现 PQ3D is_final → 未跑全景验证/VFV；"
                    "上式 dist 为 n/a 属预期。可调 min_decision_num、PQ3D 或 max_steps。"
                )

        sim.close()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f)
        with open(effectiveness_path, "w", encoding="utf-8") as f:
            json.dump(effectiveness_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)

