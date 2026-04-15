import gzip
import os
import sys
import atexit
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from habitat.utils.visualizations import maps
import json
import habitat_sim
import numpy as np
from omegaconf import OmegaConf
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import (
    convert_meters_to_pixel,
    detect_frontier_waypoints,
    get_polar_angle,
    map_coors_to_pixel,
    pixel_to_map_coors,
    reveal_fog_of_war,
)
from data_utils import PQ3DModel
from tqdm import tqdm
import time
import argparse

from anchor_nav.vote import (
    VoteConfig,
    VoteState,
    build_refined_query_prompt,
    build_target_anchor_prompt,
    parse_refined_query_from_vlm_raw,
    parse_target_anchors_from_vlm_raw,
    pq3d_stage2_object_logits,
    run_position_vote_with_pq3d_stage2,
    update_bindings_for_new_objects,
)
from vlm.client import DEFAULT_MODEL as CLIENT_DEFAULT_MODEL, chat


def _extract_target_anchors(
    description: str, model: str, vote_cfg: VoteConfig
) -> Tuple[str, List[str], List[float], str, List[str]]:
    prompt = build_target_anchor_prompt(description)
    raw = chat(text=prompt, image_path=None, model=model, max_tokens=128)
    return parse_target_anchors_from_vlm_raw(
        raw,
        nearby_anchor_vote_weight=float(vote_cfg.primary_anchor_vote_weight),
        secondary_anchor_vote_weight=float(vote_cfg.other_anchor_vote_weight),
    )


def _extract_refined_navigation_query(description: str, model: str) -> Dict[str, str]:
    raw = chat(text=build_refined_query_prompt(description), image_path=None, model=model, max_tokens=128)
    mt, ka, rq = parse_refined_query_from_vlm_raw(raw)
    return {"main_target": mt, "key_anchor": ka, "refined_query": rq, "raw": raw}


def _pq3d_logit_at(logits: np.ndarray, idx: int) -> float:
    if logits.size == 0 or idx < 0 or idx >= int(logits.shape[0]):
        return float("nan")
    return float(logits[int(idx)])


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
    print(f"[Metrics] sequence count={total_count}, avg_sr={total_sr/total_count:.6f}, avg_spl={total_spl/total_count:.6f}, avg_task_time_sec={total_task_time/total_count:.3f}")


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data); stream.flush()
        return len(data)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def _setup_run_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = os.path.join(log_dir, f"refhm3d-nav-sequence-analyze-anchor-vote-refine1-{ts}-pid{os.getpid()}.log")
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_fp)
    sys.stderr = _TeeStream(old_err, log_fp)
    print(f"[VoteRefine1] logging enabled -> {os.path.abspath(log_path)}")
    def _cleanup():
        try:
            print(f"[VoteRefine1] run finished, log saved -> {os.path.abspath(log_path)}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            log_fp.close()
    atexit.register(_cleanup)


parser = argparse.ArgumentParser(description="Run RefHM3D anchor vote refine1 batch evaluation")
parser.add_argument("--start_ratio", type=float, default=0.0)
parser.add_argument("--end_ratio", type=float, default=0.2)
parser.add_argument("--concise_description", action="store_true")
parser.add_argument("--navigation_data_path", type=str, default=str(PROJECT_ROOT / "LangMap_Annotations"))
parser.add_argument("--hm3d_data_base_path", type=str, default=str(PROJECT_ROOT / "datascene"))
parser.add_argument("--pq3d_stage1_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage1-pretrain-all"))
parser.add_argument("--pq3d_stage2_path", type=str, default=str(PROJECT_ROOT / "checkpoint/stage2-fine-tune-goat"))
parser.add_argument("--output_log_dir", type=str, default=str(PROJECT_ROOT / "output_logs/anchor/vote"))
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--vote_vlm_model", type=str, default=CLIENT_DEFAULT_MODEL)
parser.add_argument("--vote_api_key", type=str, default=os.environ.get("ZZZ_API_KEY", ""))
parser.add_argument("--vote_query_top_k", type=int, default=5)
parser.add_argument("--vote_node_top_k", type=int, default=5)
parser.add_argument("--vote_node_min_dist_m", type=float, default=1.0)
parser.add_argument("--vote_target_weight", type=float, default=0.05, help="main_target 查询基础增益")
parser.add_argument("--vote_primary_anchor_weight", type=float, default=0.05)
parser.add_argument("--vote_other_anchor_weight", type=float, default=0.05)
parser.add_argument(
    "--vote_anchor_aggregate_scale",
    type=float,
    default=1.0,
    help="在 per-anchor 权重之上再乘一遍（默认 1；与 0.9/0.05/0.05 增益一致时可保持 1）",
)
parser.add_argument(
    "--vote_rank_decay_gamma",
    type=float,
    default=0.75,
    help="排名衰减：第 k 名贡献 × γ^k；1.0 为每名同等权重（旧行为）",
)
parser.add_argument(
    "--vote_substitute_weight",
    type=float,
    default=0.9,
    help="refined≠main 时 substitute top-k 基础增益（再乘 γ^rank）",
)
parser.add_argument(
    "--vote_object_score",
    type=str,
    default="pick",
    choices=("pick", "target", "max"),
    help="胜出节点内选物体：pick=Stage2+refined_query；target=Stage2+main_target；max=二者 logits 取大",
)
args = parser.parse_args()

if args.vote_api_key:
    os.environ["ZZZ_API_KEY"] = args.vote_api_key

_setup_run_logging(os.path.expanduser(args.output_log_dir))
vote_cfg = VoteConfig(
    node_min_dist_m=float(args.vote_node_min_dist_m),
    query_top_k=int(args.vote_query_top_k),
    node_pick_top_k=int(args.vote_node_top_k),
    softmax_temp=0.07,
    target_vote_weight=float(args.vote_target_weight),
    primary_anchor_vote_weight=float(args.vote_primary_anchor_weight),
    other_anchor_vote_weight=float(args.vote_other_anchor_weight),
    anchor_vote_aggregate_scale=float(args.vote_anchor_aggregate_scale),
    query_rank_decay_gamma=float(args.vote_rank_decay_gamma),
    refined_substitute_vote_weight=float(args.vote_substitute_weight),
    object_pick_score=str(args.vote_object_score),
)
print(f"[VoteRefine1] vote_cfg={vote_cfg}")

enabled_task_levels = {"instance"}
success_distance = 0.25
decision_num_min = 3
visible_radius = 3

navigation_data_root = Path(os.path.expanduser(args.navigation_data_path))
scene_data_paths = sorted(navigation_data_root.rglob("*.json.gz"))
scene_data_paths = scene_data_paths[int(args.start_ratio * len(scene_data_paths)): int(args.end_ratio * len(scene_data_paths))]
output_log_dir = os.path.expanduser(args.output_log_dir)
os.makedirs(output_log_dir, exist_ok=True)
out_name = f"refhm3d_seq_vote_refine1_{args.start_ratio}_{args.end_ratio}.json"
if args.concise_description:
    out_name = f"refhm3d_seq_vote_refine1_concisedesc_{args.start_ratio}_{args.end_ratio}.json"
output_path = os.path.join(output_log_dir, out_name)

if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
    existing_episodes = {"_".join([r["scene_name"], r["navigation_type"], str(r["episode_id"])]) for k in result_dict for r in result_dict[k]}
else:
    result_dict = {"sequence": []}
    existing_episodes = set()

pq3d_model = PQ3DModel(os.path.expanduser(args.pq3d_stage1_path), os.path.expanduser(args.pq3d_stage2_path), min_decision_num=decision_num_min)

for scene_data_path in tqdm(scene_data_paths, desc="*** Scene ***"):
    scene_name = scene_data_path.name.split(".")[0]
    with gzip.open(scene_data_path, "rt", encoding="utf-8") as f:
        scene_data = json.load(f)
    region_to_annot_dict = scene_data["region_annotation"]
    episode_mapping = {"object": scene_data["episodes_by_object_level"], "room": scene_data["episodes_by_room_level"], "region": scene_data["episodes_by_region_level"], "instance": scene_data["episodes_by_instance_level"]}
    all_navigation_goals_dict = {x["object_id"]: x for x in scene_data["goals"]}

    for _, cur_episode in tqdm(enumerate(scene_data["episode_by_sequence"]), desc="=== Episode ==="):
        pq3d_model.reset()
        vote_state = VoteState()
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
        prev_count = int(len(getattr(pq3d_model.representation_manager, "object_count", [])))

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
            original_sentence = all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_concise_description"] if args.concise_description else all_navigation_goals_dict[cur_task["instance_id"]]["annot_unique_detailed_description"]
            refined_pack: Optional[Dict[str, str]] = None
            sentence_nav = original_sentence
            try:
                refined_pack = _extract_refined_navigation_query(original_sentence, args.vote_vlm_model)
                sentence_nav = str(refined_pack["refined_query"])
                print(
                    f"[vote-refine1][refined] task={idx} refined_query={sentence_nav!r} "
                    f"main_target={refined_pack['main_target']!r} key_anchor={refined_pack['key_anchor']!r}"
                )
            except Exception as e:
                print(f"[vote-refine1][refined-fail] task={idx} use original for PQ3D, err={e!r}")
            goal_category = goals[0]["object_category"]
            print(f"[vote-refine1][task-start] scene={scene_name} ep={episode_id} task={idx} level={task_type}")
            print(f"[vote-refine1][task-desc] {original_sentence}")

            total_steps = 0
            episode_cum_distance = 0.0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
            vote_attempts = 0
            vote_applied = 0
            vote_vlm_elapsed_ms_total = 0.0
            baseline_final_target_pos: Optional[np.ndarray] = None
            final_selected_object_pos: Optional[np.ndarray] = None

            while total_steps < int(args.max_steps):
                color_list, depth_list, agent_state_list = [], [], []
                if len(goto_color_list) > 6:
                    step = max(1, len(goto_color_list) // 6)
                    goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), step)][:6]
                    goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), step)][:6]
                    goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), step)][:6]
                color_list.extend(goto_color_list); depth_list.extend(goto_depth_list); agent_state_list.extend(goto_agent_state_list)
                for _ in range(12):
                    obs = sim.step(action="turn_left")
                    agent_state = agent.get_state()
                    color_list.append(obs["color_sensor"][:, :, :3]); depth_list.append(obs["depth_sensor"][:, :]); agent_state_list.append(agent_state)
                    fog_of_war_mask = reveal_fog_of_war(top_down_map, fog_of_war_mask, map_coors_to_pixel(agent_state.position, top_down_map, sim), get_polar_angle(agent_state), 42, visibility_dist_in_pixels, False)
                    total_steps += 1
                    if total_steps >= int(args.max_steps): break
                if total_steps >= int(args.max_steps): break
                agent_state = agent.get_state()
                frontier_waypoints = detect_frontier_waypoints(top_down_map, fog_of_war_mask, area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=False)
                if len(frontier_waypoints) == 0: frontier_waypoints = []
                else:
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints[:, ::-1], agent_state.position, top_down_map, sim)
                frontier_waypoints = [w for w in frontier_waypoints if tuple(np.round(w, 1)) not in visited_frontier_set]
                target_position, is_final = pq3d_model.decision(color_list, depth_list, agent_state_list, frontier_waypoints, sentence_nav, decision_num)
                decision_num += 1

                cur_count = int(len(np.asarray(getattr(pq3d_model.representation_manager, "object_count", np.zeros((0,))), dtype=float)))
                rep_bind = pq3d_model.representation_manager
                box_bind = np.asarray(getattr(rep_bind, "object_box", np.zeros((0, 6))), dtype=float)
                object_positions_xyz = {}
                if box_bind.ndim == 2 and box_bind.shape[0] >= cur_count and box_bind.shape[1] >= 3:
                    for oi in range(0, cur_count):
                        obj_xyz = np.asarray(box_bind[int(oi), :3], dtype=float).reshape(3).copy()
                        obj_xyz[[1, 2]] = obj_xyz[[2, 1]]
                        object_positions_xyz[int(oi)] = obj_xyz
                bind_records = update_bindings_for_new_objects(vote_state, prev_object_count=prev_count, cur_object_count=cur_count, agent_position_xyz=agent_state.position, object_positions_xyz=object_positions_xyz, cfg=vote_cfg)
                prev_count = cur_count

                used_target = np.asarray(target_position, dtype=float).reshape(3).copy()
                if is_final:
                    baseline_final_target_pos = used_target.copy()
                    vote_attempts += 1
                    t0 = time.perf_counter()
                    main_target, anchors, anchor_weights, _, anchor_types = _extract_target_anchors(
                        original_sentence, args.vote_vlm_model, vote_cfg
                    )
                    refined_query = str(refined_pack["refined_query"]).strip() if refined_pack else main_target
                    vote_vlm_elapsed_ms_total += (time.perf_counter() - t0) * 1000.0
                    print(
                        f"[vote-refine1][vlm] task={idx} decision={decision_num-1} "
                        f"main_target={main_target!r} anchors={anchors} anchor_types={anchor_types} "
                        f"anchor_weights={anchor_weights} refined_query={refined_query!r}"
                    )
                    rep = pq3d_model.representation_manager
                    vote_info = run_position_vote_with_pq3d_stage2(
                        vote_state,
                        pq3d_model=pq3d_model,
                        main_target=main_target,
                        anchors=anchors,
                        anchor_weights=anchor_weights,
                        cfg=vote_cfg,
                        candidate_object_indices=[int(x.get("object_index")) for x in bind_records],
                        refined_pick_text=refined_query,
                    )
                    if vote_info.get("ok"):
                        chosen = int(vote_info["chosen_object_index"])
                        box = np.asarray(getattr(rep, "object_box", np.zeros((0, 6))), dtype=float)
                        if box.ndim == 2 and box.shape[0] > 0 and box.shape[1] >= 3 and len(goal_positions) > 0:
                            box_nav = np.asarray(box[:, :3], dtype=float).copy()
                            box_nav[:, [1, 2]] = box_nav[:, [2, 1]]
                            d_all = [np.linalg.norm(box_nav - gp[None, :], axis=1) for gp in goal_positions]
                            d_min = np.min(np.stack(d_all, axis=0), axis=0)
                            oracle_idx = int(np.argmin(d_min))
                            oracle_dist = float(d_min[oracle_idx])
                            oracle_node = vote_state.object_node.get(int(oracle_idx), None)
                            best_node = vote_info.get("best_node_id", None)
                            oracle_in_best = (
                                oracle_node is not None and best_node is not None and int(oracle_node) == int(best_node)
                            )
                            bn_list = list(vote_info.get("best_node_objects") or [])
                            pick_pool_n = len(vote_info.get("pick_pool") or [])
                            pqtxt = vote_info.get("pick_query_text", "")
                            ch0 = int(vote_info["chosen_object_index"])
                            log_pick = pq3d_stage2_object_logits(pq3d_model, refined_query)
                            log_tgt = pq3d_stage2_object_logits(pq3d_model, main_target)
                            op = _pq3d_logit_at(log_pick, ch0)
                            ot = _pq3d_logit_at(log_tgt, ch0)
                            oop = _pq3d_logit_at(log_pick, oracle_idx)
                            oot = _pq3d_logit_at(log_tgt, oracle_idx)
                            print(
                                f"[vote-refine1][diagnose] task={idx} decision={decision_num-1} "
                                f"oracle_obj={oracle_idx} oracle_dist_to_goal={oracle_dist:.3f} "
                                f"oracle_s2_pick={oop:.4f} oracle_s2_target={oot:.4f} oracle_node={oracle_node} "
                                f"best_node={best_node} oracle_in_best_node={oracle_in_best} "
                                f"best_node_object_count={len(bn_list)} node_pick_top_k={vote_cfg.node_pick_top_k} "
                                f"pick_pool_len={pick_pool_n} pick_query_text={pqtxt!r}"
                            )
                            print(
                                f"[vote-refine1][stage2-compare] mode={vote_info.get('object_pick_score_mode')} "
                                f"chosen={ch0} s2_pick={op:.4f} s2_target={ot:.4f} | "
                                f"oracle={oracle_idx} s2_pick={oop:.4f} s2_target={oot:.4f}"
                            )
                        if chosen < len(box):
                            used_target = np.asarray(box[chosen, :3], dtype=float).reshape(3).copy()
                            used_target[[1, 2]] = used_target[[2, 1]]
                            final_selected_object_pos = used_target.copy()
                            vote_applied += 1
                    print(f"[vote-refine1][vote] task={idx} decision={decision_num-1} ok={vote_info.get('ok')} best_node={vote_info.get('best_node_id')} chosen={vote_info.get('chosen_object_index')}")
                    if vote_info.get("ok"):
                        qlogs = vote_info.get("query_logs", [])
                        for qi, qrec in enumerate(qlogs):
                            tk = qrec.get("topk", [])
                            if qrec.get("query_type") in ("refined_global_topk", "refined_substitute_topk"):
                                topk = tk[:5]
                                topk_str = ", ".join(
                                    f"obj={x.get('object_index')} score={float(x.get('score', 0.0)):.4f} "
                                    f"node={x.get('node_id')} vw={x.get('vote_weight')}"
                                    for x in topk
                                )
                                print(
                                    f"[vote-refine1][vote-query] task={idx} dec={decision_num-1} q{qi} "
                                    f"type={qrec.get('query_type')} text={qrec.get('query_text')!r} top5=[{topk_str}]"
                                )
                            elif qrec.get("query_type") == "refined_global_topk_skipped":
                                print(
                                    f"[vote-refine1][vote-query] task={idx} dec={decision_num-1} q{qi} "
                                    f"type={qrec.get('query_type')} reason={qrec.get('reason')} "
                                    f"main_target={qrec.get('main_target')!r} text={qrec.get('query_text')!r}"
                                )
                            else:
                                topk = tk[:3]
                                topk_str = ", ".join(
                                    f"obj={x.get('object_index')} score={float(x.get('score', 0.0)):.4f} node={x.get('node_id')}"
                                    for x in topk
                                )
                                eff = qrec.get("anchor_effective_vote_weight")
                                eff_s = "" if eff is None else f" eff_w={float(eff):.4f}"
                                print(
                                    f"[vote-refine1][vote-query] task={idx} dec={decision_num-1} q{qi} "
                                    f"type={qrec.get('query_type')} weight={qrec.get('query_weight')}{eff_s} "
                                    f"text={qrec.get('query_text')!r} top3=[{topk_str}]"
                                )
                        node_votes = vote_info.get("node_votes", [])[:3]
                        node_str = ", ".join(
                            f"node={x.get('node_id')} votes={x.get('vote_count')} score_sum={float(x.get('vote_score_sum', 0.0)):.4f}"
                            for x in node_votes
                        )
                        print(
                            f"[vote-refine1][vote-node] task={idx} dec={decision_num-1} top_nodes=[{node_str}] "
                            f"pick_probs={vote_info.get('pick_probs', [])} "
                            f"pick_query_text={vote_info.get('pick_query_text')!r}"
                        )
                else:
                    visited_frontier_set.add(tuple(np.round(used_target, 1)))

                agent_island = path_finder.get_island(agent_state.position)
                target_on_navmesh = path_finder.snap_point(point=used_target, island_index=agent_island)
                follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                try: action_list = follower.find_path(target_on_navmesh)
                except Exception: action_list = []
                goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []
                for action in action_list:
                    if not action: continue
                    obs = sim.step(action=action)
                    agent_state = agent.get_state()
                    goto_color_list.append(obs["color_sensor"][:, :, :3]); goto_depth_list.append(obs["depth_sensor"][:, :]); goto_agent_state_list.append(agent_state)
                    total_steps += 1
                    episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                    prev_agent_state = agent_state
                    if total_steps >= int(args.max_steps): break
                if is_final:
                    break

            task_time = float(time.perf_counter() - task_t0)
            agent_state = agent.get_state()
            view_points = [vp["agent_state"]["position"] for goal in goals for vp in goal.get("view_points", [])]
            sp = habitat_sim.MultiGoalShortestPath(); sp.requested_start = sub_episode_start_position; sp.requested_ends = view_points
            start_end_geo_distance = float(sp.geodesic_distance) if path_finder.find_path(sp) else float("inf")
            ep = habitat_sim.MultiGoalShortestPath(); ep.requested_start = agent_state.position; ep.requested_ends = view_points
            agent_end_geo_distance = float(ep.geodesic_distance) if path_finder.find_path(ep) else float("inf")
            if np.isinf(start_end_geo_distance) or np.isinf(agent_end_geo_distance): sr, spl = 0, 0
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
            result_dict.setdefault(navigation_type, []).append({
                "scene_name": scene_name, "episode_id": episode_id, "task_id": idx, "task_level": task_type,
                "navigation_type": navigation_type, "sr": sr, "spl": spl, "object_category": goal_category,
                "task_time_sec": task_time, "steps_total": int(total_steps), "decisions": int(decision_num),
                "start_goal_geo": float(start_end_geo_distance), "end_goal_geo": float(agent_end_geo_distance),
                "episode_cum_distance": float(episode_cum_distance), "vote_attempts": int(vote_attempts),
                "vote_applied": int(vote_applied), "vote_vlm_elapsed_ms_total": float(vote_vlm_elapsed_ms_total),
                "goal_positions": [gp.tolist() for gp in goal_positions],
                "baseline_target_position": None if baseline_final_target_pos is None else baseline_final_target_pos.tolist(),
                "selected_object_position": None if final_selected_object_pos is None else final_selected_object_pos.tolist(),
                "baseline_target_to_goal_l2": float(baseline_target_to_goal_l2),
                "selected_object_to_goal_l2": float(selected_object_to_goal_l2),
            })
            print(
                f"[vote-refine1] scene={scene_name} ep={episode_id} task={idx} SR={sr} SPL={spl:.4f} "
                f"time={task_time:.3f}s steps={total_steps} decisions={decision_num} "
                f"vote_attempts={vote_attempts} vote_applied={vote_applied} "
                f"dist(baseline_target,goal)={baseline_target_to_goal_l2:.3f} "
                f"dist(vote_object,goal)={selected_object_to_goal_l2:.3f}"
            )

        sim.close()
        with open(output_path, "w") as f:
            json.dump(result_dict, f)
        sequence_compute_metric_results(result_dict)

sequence_compute_metric_results(result_dict)
