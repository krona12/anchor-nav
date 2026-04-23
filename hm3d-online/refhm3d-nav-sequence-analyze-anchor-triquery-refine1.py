import argparse
import atexit
import datetime
import gzip
import json
import os
import re
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anchor_nav.tri_query import TriQueryConfig, build_query_fn_from_pq3d_stage2, run_tri_query
from anchor_nav.wake import WakeConfig, WakeState, apply_wake
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
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL, chat


def build_refined_query_prompt(description: str) -> str:
    return (
        "Refine the navigation description for robust object query.\n"
        "Keep ONLY:\n"
        "1) main_target: exact target object phrase\n"
        "2) key_anchor: at most one spatially tight anchor object that is directly linked to main_target.\n"
        "Drop broad scene/global context and weak anchors.\n"
        "Return strict JSON only: "
        "{\"main_target\": \"...\", \"key_anchor\": \"... or empty\", \"refined_query\": \"...\"}.\n"
        "If no reliable anchor, set key_anchor to empty and refined_query=main_target.\n\n"
        f"Description: {description}"
    )


def build_target_anchor_prompt(description: str) -> str:
    return (
        "Extract navigation target and anchors from the description.\n"
        "Return strict JSON only: "
        "{\"main_target\":\"...\", \"anchors\":[...], \"anchor_types\":[...], \"spatial_relation\":\"... or null\"}.\n\n"
        "Hard rules:\n"
        "1) main_target MUST be a single object noun phrase (with material/color/style modifiers if present).\n"
        "2) main_target MUST NOT contain relational or scene clauses.\n"
        "3) Nearby relation object(s) go to anchors, not main_target.\n"
        "4) anchor_types aligned to anchors, each is either \"nearby\" or \"scene\".\n"
        "5) Keep ONLY one nearby anchor and at most two secondary(scene) anchors.\n\n"
        f"Description: {description}"
    )


def _parse_json_obj(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _chat_with_retry_once(*, text: str, model: str, max_tokens: int = 128) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            return chat(text=text, image_path=None, model=model, max_tokens=max_tokens)
        except Exception as e:
            last_err = e
            if attempt == 0:
                print(f"[triquery-refine1][vlm-retry] first attempt failed, retry once, err={e!r}")
                continue
            raise
    raise RuntimeError(f"vlm request failed after retry: {last_err!r}")


def _extract_refined_navigation_query(description: str, model: str) -> Dict[str, str]:
    raw = _chat_with_retry_once(text=build_refined_query_prompt(description), model=model, max_tokens=128)
    parsed = _parse_json_obj(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    key_anchor = str(parsed.get("key_anchor", "")).strip()
    refined_query = str(parsed.get("refined_query", "")).strip()
    if not main_target:
        raise RuntimeError(f"empty main_target in refined query, raw={raw!r}")
    if not refined_query:
        refined_query = main_target if not key_anchor else f"{main_target} near {key_anchor}"
    return {"main_target": main_target, "key_anchor": key_anchor, "refined_query": refined_query, "raw": raw}


def _extract_target_anchor(description: str, model: str) -> Dict[str, Any]:
    raw = _chat_with_retry_once(text=build_target_anchor_prompt(description), model=model, max_tokens=128)
    parsed = _parse_json_obj(raw)
    main_target = str(parsed.get("main_target", "")).strip()
    anchors = [str(x).strip() for x in parsed.get("anchors", []) if str(x).strip()]
    anchor_types = [str(x).strip().lower() for x in parsed.get("anchor_types", [])]
    relation = str(parsed.get("spatial_relation", "") or "").strip()
    if not main_target:
        raise RuntimeError(f"empty main_target, raw={raw!r}")
    nearest_anchor = ""
    for i, a in enumerate(anchors):
        t = anchor_types[i] if i < len(anchor_types) else "nearby"
        if t == "nearby":
            nearest_anchor = a
            break
    if not nearest_anchor and len(anchors) > 0:
        nearest_anchor = anchors[0]
    return {
        "main_target": main_target,
        "nearest_anchor": nearest_anchor,
        "anchors": anchors,
        "anchor_types": anchor_types,
        "spatial_relation": relation,
        "raw": raw,
    }


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
        print("[Metrics] sequence count: 0")
        return
    total_sr = sum(float(item.get("sr", 0)) for item in sequence_results)
    total_spl = sum(float(item.get("spl", 0)) for item in sequence_results)
    total_task_time = sum(float(item.get("task_time_sec", 0.0)) for item in sequence_results)
    print(
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
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-triquery-refine1-{ts}-pid{os.getpid()}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[TriQueryRefine1] logging enabled -> {os.path.abspath(log_path)}")

    def _cleanup():
        try:
            print(f"[TriQueryRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()

    atexit.register(_cleanup)


def _query_top_hit_rank(topk: List[Dict[str, Any]], obj_idx: int) -> Optional[int]:
    for i, rec in enumerate(topk, start=1):
        if int(rec.get("object_index", -1)) == int(obj_idx):
            return int(i)
    return None


def _find_oracle_object_index(rep: Any, goal_positions: List[np.ndarray]) -> Optional[int]:
    if len(goal_positions) == 0:
        return None
    box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
    if box.ndim != 2 or box.shape[0] == 0 or box.shape[1] < 3:
        return None
    box_nav = np.asarray(box[:, :3], dtype=float).copy()
    box_nav[:, [1, 2]] = box_nav[:, [2, 1]]
    d_all = [np.linalg.norm(box_nav - gp[None, :], axis=1) for gp in goal_positions]
    d_min = np.min(np.stack(d_all, axis=0), axis=0)
    return int(np.argmin(d_min))


parser = argparse.ArgumentParser(description="Run RefHM3D anchor triquery refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0)
parser.add_argument("--end_ratio", type=float, default=0.2)
parser.add_argument("--concise_description", action="store_true")
parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/triquery"))
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--triquery_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
parser.add_argument("--triquery_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
parser.add_argument("--tri_top_k", type=int, default=16)
parser.add_argument("--tri_sigma_anchor", type=float, default=2.0)
parser.add_argument("--tri_sigma_full", type=float, default=3.0)
parser.add_argument("--tri_anchor_distance_mode", type=str, default="centroid", choices=("min", "centroid"))
parser.add_argument("--tri_full_distance_mode", type=str, default="centroid", choices=("min", "centroid"))
parser.add_argument("--tri_centroid_temp", type=float, default=0.07)
parser.add_argument("--tri_final_topn_log", type=int, default=5)
parser.add_argument(
    "--decision_log_interval",
    type=int,
    default=0,
    help="每隔 N 次 decision 打印一次明细；<=0 表示关闭明细",
)
parser.add_argument(
    "--stuck_repeat_threshold",
    type=int,
    default=6,
    help="连续相同 non-final target 达到阈值后触发一次 stuck 处理",
)
parser.add_argument(
    "--wake_repeat_threshold",
    type=int,
    default=2,
    help="连续命中同一 frontier 达到阈值后触发 wake",
)
parser.add_argument(
    "--wake_empty_path_threshold",
    type=int,
    default=3,
    help="同一 frontier 连续空路径达到阈值后标记为 blocked",
)
parser.add_argument(
    "--wake_force_final_decision_on_stuck",
    action="store_true",
    help="wake 无可选 frontier 时，立即触发 final decision",
)
args = parser.parse_args()

if args.triquery_api_key:
    os.environ["ZZZ_API_KEY"] = args.triquery_api_key

output_log_dir = os.path.expanduser(args.output_log_dir)
_setup_run_logging(output_log_dir)
tri_cfg = TriQueryConfig(
    top_k=int(args.tri_top_k),
    sigma_anchor=float(args.tri_sigma_anchor),
    sigma_full=float(args.tri_sigma_full),
    anchor_distance_mode=str(args.tri_anchor_distance_mode),
    full_distance_mode=str(args.tri_full_distance_mode),
    centroid_temp=float(args.tri_centroid_temp),
)
print(f"[TriQueryRefine1] tri_cfg={tri_cfg}")
wake_cfg = WakeConfig(
    repeat_threshold=int(args.wake_repeat_threshold),
    empty_path_threshold=int(args.wake_empty_path_threshold),
    force_final_decision_on_stuck=bool(args.wake_force_final_decision_on_stuck),
)
print(f"[TriQueryRefine1] wake_cfg={wake_cfg}")
print("[TriQueryRefine1] tri_query_backend=pq3d_stage2_og3d")

enabled_task_levels = {"instance"}
success_distance = 0.25
decision_num_min = 3
visible_radius = 3

navigation_data_root = Path(os.path.expanduser(args.navigation_data_path))
scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]
os.makedirs(output_log_dir, exist_ok=True)
out_name = f"refhm3d_seq_triquery_refine1_{args.start_ratio}_{args.end_ratio}.json"
eff_name = f"refhm3d_seq_triquery_refine1_effectiveness_{args.start_ratio}_{args.end_ratio}.json"
if args.concise_description:
    out_name = f"refhm3d_seq_triquery_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
    eff_name = f"refhm3d_seq_triquery_refine1_effectiveness_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
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
            refined_pack: Optional[Dict[str, str]] = None
            sentence_nav = original_sentence
            try:
                refined_pack = _extract_refined_navigation_query(original_sentence, args.triquery_vlm_model)
                sentence_nav = str(refined_pack["refined_query"])
                print(
                    f"[triquery-refine1][refined] task={idx} refined_query={sentence_nav!r} "
                    f"main_target={refined_pack['main_target']!r} key_anchor={refined_pack['key_anchor']!r}"
                )
            except Exception as e:
                print(f"[triquery-refine1][refined-fail] task={idx} use original for PQ3D, err={e!r}")
            goal_category = goals[0]["object_category"]
            print(f"[triquery-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} level={task_type}")
            print(f"[triquery-refine1][task-desc] {original_sentence}")

            total_steps = 0
            episode_cum_distance = 0.0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            tri_attempts = 0
            tri_used = 0
            tri_vlm_elapsed_ms_total = 0.0
            baseline_final_target_pos: Optional[np.ndarray] = None
            final_selected_object_pos: Optional[np.ndarray] = None
            task_effective_logs: List[Dict[str, Any]] = []
            wake_state = WakeState()
            prev_nonfinal_target_key: Optional[Tuple[float, float, float]] = None
            repeated_nonfinal_target = 0
            empty_path_count = 0
            frontier_empty_counts: Dict[Tuple[float, float, float], int] = {}
            blocked_frontier_count = 0

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
                for _ in range(12):
                    obs = sim.step(action="turn_left")
                    agent_state = agent.get_state()
                    color_list.append(obs["color_sensor"][:, :, :3])
                    depth_list.append(obs["depth_sensor"][:, :])
                    agent_state_list.append(agent_state)
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
                    if total_steps >= int(args.max_steps):
                        break
                if total_steps >= int(args.max_steps):
                    break
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
                target_position, is_final = pq3d_model.decision(
                    color_list, depth_list, agent_state_list, frontier_waypoints, sentence_nav, decision_num
                )
                target_xyz = np.asarray(target_position, dtype=float).reshape(-1)[:3].tolist()
                if int(args.decision_log_interval) > 0 and (decision_num % int(args.decision_log_interval) == 0):
                    print(
                        f"[triquery-refine1][decision] task={idx} dec={decision_num} frontiers={len(frontier_waypoints)} "
                        f"baseline_target={target_xyz} final={bool(is_final)}"
                    )
                decision_num += 1

                used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                wake_info = apply_wake(
                    wake_state,
                    cfg=wake_cfg,
                    is_final=bool(is_final),
                    target_position_xyz=used_target,
                    frontier_waypoints=frontier_waypoints,
                    agent_position_xyz=agent_state.position,
                    frontier_empty_path_counts=frontier_empty_counts,
                )
                if wake_info.get("triggered"):
                    if wake_info.get("redirected_target") is not None:
                        used_target = np.asarray(wake_info["redirected_target"], dtype=float).reshape(3).copy()
                        print(
                            f"[triquery-refine1][wake] task={idx} dec={decision_num-1} "
                            f"action=redirect_frontier target={used_target.tolist()}"
                        )
                    elif wake_info.get("force_final_decision"):
                        is_final = True
                        baseline_final_target_pos = used_target.copy()
                        print(
                            f"[triquery-refine1][wake] task={idx} dec={decision_num-1} action=force_final_decision"
                        )
                if is_final:
                    prev_nonfinal_target_key = None
                    repeated_nonfinal_target = 0
                    baseline_final_target_pos = used_target.copy()
                    tri_attempts += 1
                    t0 = time.perf_counter()
                    ta = _extract_target_anchor(original_sentence, args.triquery_vlm_model)
                    tri_vlm_elapsed_ms_total += (time.perf_counter() - t0) * 1000.0
                    print(
                        f"[triquery-refine1][vlm] task={idx} decision={decision_num-1} "
                        f"main_target={ta['main_target']!r} nearest_anchor={ta['nearest_anchor']!r} "
                        f"anchors={ta['anchors']} anchor_types={ta['anchor_types']}"
                    )
                    tri_info = run_tri_query(
                        description=original_sentence,
                        rep=pq3d_model.representation_manager,
                        query_fn=query_fn,
                        main_target=ta["main_target"],
                        nearest_anchor=ta["nearest_anchor"],
                        cfg=tri_cfg,
                    )
                    if tri_info.get("ok"):
                        tri_used += 1
                        used_target = np.asarray(tri_info["target_xyz"], dtype=float).reshape(3).copy()
                        final_selected_object_pos = used_target.copy()
                    final_topn = list(tri_info.get("final_ranking", []))[: max(1, int(args.tri_final_topn_log))]
                    topn_str = ", ".join(
                        f"rank={i+1}/obj={int(x.get('object_index'))}/final={float(x.get('final_score', 0.0)):.4f}"
                        for i, x in enumerate(final_topn)
                    )
                    print(f"[triquery-refine1][final-top] task={idx} dec={decision_num-1} top{len(final_topn)}=[{topn_str}]")

                    oracle_idx = _find_oracle_object_index(pq3d_model.representation_manager, goal_positions)
                    gt_hit = None
                    if oracle_idx is not None and tri_info.get("ok"):
                        qlogs = tri_info.get("query_logs", {})
                        full_rank = _query_top_hit_rank(list(qlogs.get("full_topk", [])), oracle_idx)
                        target_rank = _query_top_hit_rank(list(qlogs.get("target_topk", [])), oracle_idx)
                        anchor_rank = _query_top_hit_rank(list(qlogs.get("anchor_topk", [])), oracle_idx)
                        where = []
                        if full_rank is not None:
                            where.append(f"full@{full_rank}")
                        if target_rank is not None:
                            where.append(f"target@{target_rank}")
                        if anchor_rank is not None:
                            where.append(f"anchor@{anchor_rank}")
                        gt_hit = {
                            "oracle_object_index": int(oracle_idx),
                            "in_any_query_topk": bool(len(where) > 0),
                            "in_full_topk_rank": full_rank,
                            "in_target_topk_rank": target_rank,
                            "in_anchor_topk_rank": anchor_rank,
                            "where": where,
                        }
                        print(
                            f"[triquery-refine1][gt-hit] task={idx} dec={decision_num-1} oracle_obj={oracle_idx} "
                            f"in_queries={gt_hit['in_any_query_topk']} where={where}"
                        )
                    task_effective_logs.append(
                        {
                            "scene_name": scene_name,
                            "episode_id": int(episode_id),
                            "task_id": int(idx),
                            "decision_num": int(decision_num - 1),
                            "original_sentence": original_sentence,
                            "refined_query_for_pq3d": sentence_nav,
                            "triquery_extract": ta,
                            "triquery_ok": bool(tri_info.get("ok", False)),
                            "triquery_reason": tri_info.get("reason"),
                            "wake_info": wake_info,
                            "final_topn": final_topn,
                            "gt_hit": gt_hit,
                            "tri_info": tri_info,
                        }
                    )
                else:
                    cur_key = tuple(np.round(used_target, 3).tolist())
                    if prev_nonfinal_target_key is not None and cur_key == prev_nonfinal_target_key:
                        repeated_nonfinal_target += 1
                    else:
                        repeated_nonfinal_target = 1
                        prev_nonfinal_target_key = cur_key
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=used_target, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(
                    path_finder,
                    agent,
                    forward_key="move_forward",
                    left_key="turn_left",
                    right_key="turn_right",
                )
                try:
                    action_list = follower.find_path(target_on_navmesh)
                except Exception:
                    action_list = []
                if (not is_final) and len(action_list) == 0:
                    empty_path_count += 1
                    cur_target_key = tuple(np.round(used_target, 3).tolist())
                    frontier_empty_counts[cur_target_key] = int(frontier_empty_counts.get(cur_target_key, 0) + 1)
                    if frontier_empty_counts[cur_target_key] >= int(args.wake_empty_path_threshold):
                        blocked_frontier_count = len(
                            [k for k, v in frontier_empty_counts.items() if int(v) >= int(args.wake_empty_path_threshold)]
                        )
                        print(
                            f"[triquery-refine1][frontier-blocked] task={idx} dec={decision_num-1} "
                            f"frontier={list(cur_target_key)} empty_path_hits={frontier_empty_counts[cur_target_key]} "
                            f"blocked_frontiers={blocked_frontier_count}"
                        )
                    if int(args.stuck_repeat_threshold) > 0 and repeated_nonfinal_target >= int(args.stuck_repeat_threshold):
                        for fw in frontier_waypoints:
                            visited_frontier_set.add(tuple(np.round(np.asarray(fw, dtype=float).reshape(-1)[:3], 1)))
                        print(
                            f"[triquery-refine1][stuck] task={idx} dec={decision_num-1} "
                            f"repeat_target={list(cur_key)} repeat_count={repeated_nonfinal_target} "
                            f"empty_path_count={empty_path_count} action=mark_current_frontiers_visited"
                        )
                        repeated_nonfinal_target = 0
                        prev_nonfinal_target_key = None
                goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
                for action in action_list:
                    if not action:
                        continue
                    obs = sim.step(action=action)
                    agent_state = agent.get_state()
                    goto_color_list.append(obs["color_sensor"][:, :, :3])
                    goto_depth_list.append(obs["depth_sensor"][:, :])
                    goto_agent_state_list.append(agent_state)
                    total_steps += 1
                    episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                    prev_agent_state = agent_state
                    if total_steps >= int(args.max_steps):
                        break
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
            if len(goal_positions) > 0:
                if baseline_final_target_pos is not None:
                    d0 = [float(np.linalg.norm(baseline_final_target_pos - gp)) for gp in goal_positions]
                    baseline_target_to_goal_l2 = float(min(d0))
                if final_selected_object_pos is not None:
                    d1 = [float(np.linalg.norm(final_selected_object_pos - gp)) for gp in goal_positions]
                    selected_object_to_goal_l2 = float(min(d1))
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
                    "tri_attempts": int(tri_attempts),
                    "tri_used": int(tri_used),
                    "tri_vlm_elapsed_ms_total": float(tri_vlm_elapsed_ms_total),
                    "empty_path_count": int(empty_path_count),
                    "blocked_frontier_count": int(
                        len([k for k, v in frontier_empty_counts.items() if int(v) >= int(args.wake_empty_path_threshold)])
                    ),
                    "tri_query_backend": "pq3d_stage2_og3d",
                    "wake_repeat_threshold": int(args.wake_repeat_threshold),
                    "wake_empty_path_threshold": int(args.wake_empty_path_threshold),
                    "wake_force_final_decision_on_stuck": bool(args.wake_force_final_decision_on_stuck),
                    "goal_positions": [gp.tolist() for gp in goal_positions],
                    "baseline_target_position": None
                    if baseline_final_target_pos is None
                    else baseline_final_target_pos.tolist(),
                    "selected_object_position": None
                    if final_selected_object_pos is None
                    else final_selected_object_pos.tolist(),
                    "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                    "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
                }
            )
            effectiveness_dict["records"].append(
                {
                    "scene_name": scene_name,
                    "episode_id": int(episode_id),
                    "task_id": int(idx),
                    "task_level": task_type,
                    "navigation_type": navigation_type,
                    "tri_attempts": int(tri_attempts),
                    "tri_used": int(tri_used),
                    "empty_path_count": int(empty_path_count),
                    "blocked_frontier_count": int(
                        len([k for k, v in frontier_empty_counts.items() if int(v) >= int(args.wake_empty_path_threshold)])
                    ),
                    "tri_query_backend": "pq3d_stage2_og3d",
                    "wake_repeat_threshold": int(args.wake_repeat_threshold),
                    "wake_empty_path_threshold": int(args.wake_empty_path_threshold),
                    "wake_force_final_decision_on_stuck": bool(args.wake_force_final_decision_on_stuck),
                    "task_effective_logs": task_effective_logs,
                }
            )
            print(
                f"[triquery-refine1] scene={scene_name} ep={episode_id} task={idx} SR={sr} SPL={spl:.4f} "
                f"time={task_time:.3f}s steps={total_steps} decisions={decision_num} "
                f"tri_attempts={tri_attempts} tri_used={tri_used} "
                f"empty_path_count={empty_path_count} blocked_frontier_count={len([k for k, v in frontier_empty_counts.items() if int(v) >= int(args.wake_empty_path_threshold)])} "
                f"dist(baseline_target,goal)={baseline_target_to_goal_l2:.3f} "
                f"dist(tri_object,goal)={selected_object_to_goal_l2:.3f}"
            )

        sim.close()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f)
        with open(effectiveness_path, "w", encoding="utf-8") as f:
            json.dump(effectiveness_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
