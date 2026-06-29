#!/usr/bin/env python3
"""Demo visual simulator for the real TFFS + Vista2MQSC sequence flow.

This script follows the standard sequence pipeline:
scan -> frontier detection -> PQ3D decision -> TFFS only for non-final frontier
overrides -> Vista2MQSC only for final object decisions -> follower.

It intentionally does not use smoke/mock fallbacks. PQ3D failures are fatal.
TFFS failures are non-fatal by design: TFFS is only a frontier override, so an
error keeps the PQ3D baseline frontier.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[2]
HM3D_ONLINE = PROJECT_ROOT / "hm3d-online"
REF_FLOW = HM3D_ONLINE / "refhm3d-nav-sequence-analyze-anchor-tffs-vista2mqsc-sequence-refine1.py"

for _p in (PROJECT_ROOT, HM3D_ONLINE, HM3D_ONLINE / "anchor_nav", CODE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import interactive_vista2mqsc_teleop as teleop  # noqa: E402
import module_sim as vismod  # noqa: E402
from frontier_utils import get_polar_angle  # noqa: E402


def _load_real_pq3d(args: argparse.Namespace) -> Any:
    print("[demo-sim] Loading PQ3DModel before Habitat simulator ...", flush=True)
    from data_utils import PQ3DModel

    pq3d = PQ3DModel(
        os.path.expanduser(args.pq3d_stage1_path),
        os.path.expanduser(args.pq3d_stage2_path),
        min_decision_num=int(args.decision_num_min),
    )
    pq3d.reset()
    if hasattr(pq3d, "mask_generator"):
        pq3d.mask_generator = teleop.VIS_NAV._DropTaskLevelMaskGenerator(pq3d.mask_generator)
    print("[demo-sim] PQ3DModel ready before Habitat simulator.", flush=True)
    return pq3d


def _load_ref_flow() -> Any:
    spec = importlib.util.spec_from_file_location("tffs_vista2mqsc_ref_flow", str(REF_FLOW))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference flow: {REF_FLOW}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tffs_vista2mqsc_ref_flow"] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_rgb = teleop.VIS_NAV._depth_to_rgb(np.asarray(depth, dtype=np.float32))
    cv2.imwrite(str(path), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))


class DemoNavigator(teleop.InteractiveNavigator):
    def __init__(self, *args: Any, ref_flow: Any, demo_log: List[str], **kwargs: Any) -> None:
        self.ref_flow = ref_flow
        self.demo_log = demo_log
        self.eye_frame_index = 0
        self.tffs_call_count = 0
        self.tffs_apply_count = 0
        self.vista2mqsc_call_count = 0
        self.vista2mqsc_apply_count = 0
        super().__init__(*args, **kwargs)

    def render_topdown(self) -> np.ndarray:
        return vismod.render_gray_topdown_rgb(
            self,
            self.sim,
            fog=self.fog.copy(),
            agent_state=self.agent.get_state(),
            target=self.current_target,
            is_final=bool(self.current_target_is_final),
            frontiers=list(self.current_frontiers),
            selected_frontier_idx=self.selected_frontier_idx,
        )

    def save_trajectory_snapshot(self, name: str = "trajectory_latest.png") -> None:
        rgb = vismod.render_gray_topdown_rgb(
            self,
            self.sim,
            fog=np.ones_like(self.fog),
            agent_state=self.agent.get_state(),
            target=self.current_target,
            is_final=bool(self.current_target_is_final),
            frontiers=[],
            selected_frontier_idx=None,
        )
        _save_rgb(self.out_dir / "trajectory" / name, rgb)

    def _demo_max_steps(self) -> int:
        return int(getattr(self.args, "demo_max_steps", 0) or 0)

    def _demo_step_budget_exhausted(self) -> bool:
        limit = self._demo_max_steps()
        return bool(limit > 0 and int(self.step_count) >= int(limit))

    def runtime_mode_evidence(self) -> Dict[str, Any]:
        return {
            "module_mode": "real",
            "smoke": False,
            "mock": False,
            "pq3d_required": True,
            "disable_pq3d": bool(getattr(self.args, "disable_pq3d", False)),
            "pq3d_preloaded": self.pq3d_model is not None,
            "tffs": {
                "use_vlm": bool(self.ref_flow.TFFS_CFG.use_vlm),
                "vlm_model": str(self.ref_flow.TFFS_CFG.vlm_model),
                "vlm_call_interval": int(self.ref_flow.TFFS_CFG.vlm_call_interval),
                "timeout_sec": float(getattr(self.ref_flow, "TFFS_TIMEOUT_SEC", 0.0)),
            },
            "mqsc": {
                "use_vlm": bool(self.ref_flow.MQSC_R1_CFG.use_vlm),
                "vlm_model": str(self.ref_flow.MQSC_R1_CFG.vlm_model),
            },
            "vista_ls": {
                "enable_vvd_replacement": bool(self.ref_flow.VISTALS_CFG.enable_vvd_replacement),
                "prefer_visible_baseline": bool(self.ref_flow.VISTALS_CFG.prefer_visible_baseline),
                "apply_task_levels": sorted(str(x) for x in self.ref_flow.VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS),
            },
        }

    def _record_current_observation(self, event: str, obs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        rec = super()._record_current_observation(event, obs)
        if obs is None:
            obs = self.sim.get_sensor_observations()
        idx = int(self.eye_frame_index)
        self.eye_frame_index += 1
        frame_dir = self.out_dir / "robot_eye_frames"
        _save_rgb(frame_dir / f"frame_{idx:06d}_{event}.png", np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8))
        _save_depth(frame_dir / f"frame_{idx:06d}_{event}_depth.png", np.asarray(obs["depth_sensor"][:, :], dtype=np.float32))
        rec["eye_frame_index"] = idx
        _write_json(frame_dir / f"frame_{idx:06d}_{event}.json", rec)
        return rec

    def _capture_scan_frames(self, dec_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        scan_rgb: List[np.ndarray] = []
        scan_depth: List[np.ndarray] = []
        scan_states: List[Any] = []
        for view_idx in range(12):
            if self._demo_step_budget_exhausted():
                self.demo_log.append(
                    f"[scan] stopped_by_max_steps step={self.step_count} limit={self._demo_max_steps()}"
                )
                break
            obs = self.sim.step(action="turn_left")
            self.step_count += 1
            state = self.agent.get_state()
            rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy()
            dep = np.asarray(obs["depth_sensor"][:, :], dtype=np.float32).copy()
            scan_rgb.append(rgb)
            scan_depth.append(dep)
            scan_states.append(teleop._state_copy(state))
            self._record_current_observation(f"scan_view_{view_idx:02d}", obs)
            self.display(obs, f"decision scan {view_idx + 1}/12")
            if not bool(self.args.headless):
                cv2.waitKey(max(1, int(self.args.scan_wait_ms)))

        teleop.VIS_NAV.save_panorama_frames(dec_dir, scan_rgb, scan_depth)
        return scan_rgb, scan_depth, scan_states

    def follow_latest_target(self) -> None:
        if self.current_target is None:
            print("[demo-sim] No decision target yet.", flush=True)
            return
        actions, follow_log = self._plan_follow_actions(self.current_target)
        if not actions:
            print(f"[demo-sim] No follow actions: {follow_log}", flush=True)
            if self.latest_decision_payload is not None and self.latest_decision_path is not None:
                self.latest_decision_payload["follow"] = follow_log
                _write_json(self.latest_decision_path, self.latest_decision_payload)
            return

        follow_rgb: List[np.ndarray] = []
        executed: List[Any] = []
        for action in actions:
            if not action:
                continue
            if self._demo_step_budget_exhausted():
                follow_log["truncated_by_demo_max_steps"] = True
                follow_log["demo_max_steps"] = int(self._demo_max_steps())
                break
            obs = self.sim.step(action=action)
            executed.append(action)
            follow_rgb.append(np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy())
            state_now = self.agent.get_state()
            self.episode_cum_distance += float(np.linalg.norm(state_now.position - self.prev_state.position))
            self.prev_state = teleop._state_copy(state_now)
            self.step_count += 1
            self._record_current_observation(f"follow_{action}", obs)
            self.display(obs, f"follow target: {action}")
            if not bool(self.args.headless):
                cv2.waitKey(max(1, int(self.args.follow_wait_ms)))
            if int(self.args.max_follow_actions) > 0 and len(executed) >= int(self.args.max_follow_actions):
                follow_log["truncated_by_max_follow_actions"] = True
                break

        follow_log.update(
            {
                "ok": bool(follow_log.get("ok", True)),
                "executed_action_count": int(len(executed)),
                "executed_actions": [str(x) for x in executed],
                "end_position": np.asarray(self.agent.get_state().position, dtype=float).reshape(3).tolist(),
                "step_count_after_follow": int(self.step_count),
                "episode_cum_distance": float(self.episode_cum_distance),
            }
        )
        if self.latest_decision_dir is not None:
            saved = teleop.VIS_NAV.save_follow_frames(
                self.latest_decision_dir,
                follow_rgb,
                max_saved=int(self.args.max_saved_follow_frames),
            )
            follow_log["saved_rgb_indices"] = [int(x) for x in saved]
        if self.latest_decision_payload is not None and self.latest_decision_path is not None:
            self.latest_decision_payload["follow"] = follow_log
            _write_json(self.latest_decision_path, self.latest_decision_payload)
        self.save_trajectory_snapshot("trajectory_latest.png")
        print(f"[demo-sim] Follow done actions={len(executed)}", flush=True)

    def save_extra_stop_panorama(self, pano_dir: Path, *, stop_reason: str) -> List[Dict[str, Any]]:
        pano_dir.mkdir(parents=True, exist_ok=True)
        views: List[Dict[str, Any]] = []
        step_start = int(self.step_count)
        for view_idx in range(12):
            obs = self.sim.step(action="turn_left")
            state = self.agent.get_state()
            rgb = np.asarray(obs["color_sensor"][:, :, :3], dtype=np.uint8).copy()
            dep = np.asarray(obs["depth_sensor"][:, :], dtype=np.float32).copy()
            rgb_path = pano_dir / f"view_{view_idx:02d}.png"
            depth_path = pano_dir / f"view_{view_idx:02d}_depth.png"
            _save_rgb(rgb_path, rgb)
            _save_depth(depth_path, dep)
            views.append(
                {
                    "view_index": int(view_idx),
                    "image_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
                    "yaw": float(get_polar_angle(state)),
                    "heading_xz": self.ref_flow._forward_xz(state).tolist(),
                }
            )
        _write_json(
            pano_dir / "extra_stop_panorama.json",
            {
                "stop_reason": str(stop_reason),
                "view_count": int(len(views)),
                "navigation_step_start": int(step_start),
                "navigation_step_after": int(self.step_count),
                "navigation_steps_counted": False,
                "views": views,
            },
        )
        return views

    def run_decision_round(self) -> None:
        dec_num = int(self.decision_num)
        dec_dir = self.out_dir / "decisions" / f"dec_{dec_num:03d}"
        dec_dir.mkdir(parents=True, exist_ok=True)
        print(f"[demo-sim] decision {dec_num:03d}: scan/frontier/PQ3D start", flush=True)

        context_rgb, context_depth, context_states = teleop._sample_context(
            self.context_buffer, int(self.args.max_context_frames)
        )
        scan_rgb, scan_depth, scan_states = self._capture_scan_frames(dec_dir / "scan")
        color_list = list(context_rgb) + scan_rgb
        depth_list = list(context_depth) + scan_depth
        state_list = list(context_states) + scan_states

        frontiers = self.detect_frontiers()
        t_pq = time.perf_counter()
        pq3d = self.ensure_pq3d()
        if pq3d is None:
            raise RuntimeError("PQ3D is required for demo sim; --disable_pq3d is not allowed")
        target, is_final = pq3d.decision(
            color_list,
            depth_list,
            state_list,
            frontiers,
            self.ctx.sentence,
            dec_num,
            task_level=self.ctx.task_level,
        )
        pq_ms = (time.perf_counter() - t_pq) * 1000.0
        pq3d_aux = getattr(pq3d, "last_decision_aux", {}) or {}

        used_target = np.asarray(target, dtype=float).reshape(3).copy()
        tffs_info: Dict[str, Any] = {
            "module": "tffs",
            "called": False,
            "tffs_applied": False,
            "reason": "final_decision_skip_tffs" if bool(is_final) else "no_tffs_attempted",
            "target_before": used_target.tolist(),
            "target_after": used_target.tolist(),
        }
        module_info: Dict[str, Any] = {
            "module": "vista2mqsc",
            "called": False,
            "applied": False,
            "reason": "non_final_decision",
            "target_before": used_target.tolist(),
            "target_after": used_target.tolist(),
        }

        if bool(is_final):
            used_target, module_info = self.ref_flow.vista2mqsc_refine_hook(
                sentence=self.ctx.sentence,
                task_type=self.ctx.task_level,
                scene_name=self.ctx.scene_name,
                episode_id=int(self.ctx.episode_id),
                task_id=int(self.ctx.task_id),
                decision_num=dec_num,
                is_final=True,
                pq3d_model=self.pq3d_model,
                target_position=used_target,
                decision_aux=pq3d_aux,
                output_dir=dec_dir / "modules" / "vista2mqsc",
                path_finder=self.path_finder,
                agent_position_xyz=np.asarray(self.agent.get_state().position, dtype=float).reshape(3),
            )
            self.vista2mqsc_call_count += int(bool(module_info.get("called", True)))
            self.vista2mqsc_apply_count += int(bool(module_info.get("applied", False)))
        else:
            used_target, tffs_info = self.ref_flow.tffs_frontier_hook(
                sentence=self.ctx.sentence,
                scene_name=self.ctx.scene_name,
                episode_id=int(self.ctx.episode_id),
                task_id=int(self.ctx.task_id),
                decision_num=dec_num,
                pq3d_model=pq3d,
                baseline_target=used_target,
                frontier_waypoints=frontiers,
                scan_rgb=scan_rgb,
                scan_states=scan_states,
                output_dir=dec_dir / "modules" / "tffs",
                disabled=False,
                is_final=False,
            )
            self.tffs_call_count += int(bool(tffs_info.get("called", tffs_info.get("tffs_called", False))))
            self.tffs_apply_count += int(bool(tffs_info.get("tffs_applied", False)))

        self.current_target = np.asarray(used_target, dtype=float).reshape(3)
        self.current_target_is_final = bool(is_final)
        self.selected_frontier_idx = teleop.VIS_NAV._nearest_frontier_index(self.current_target, frontiers)
        target_rc = teleop._pos_to_pixel(self.current_target, self.top_down_map, self.sim)
        if bool(is_final):
            self.final_decision_pixels.append(target_rc)
        else:
            self.decision_pixels.append(target_rc)
            self.visited_frontiers.add(teleop._frontier_visit_key(self.current_target))

        state = self.agent.get_state()
        vismod.save_decision_topdown_map(
            nav=self,
            sim=self.sim,
            out_dir=dec_dir,
            agent_state=state,
            target=self.current_target,
            is_final=bool(is_final),
            frontiers=list(frontiers),
            selected_frontier_idx=self.selected_frontier_idx if not bool(is_final) else None,
        )
        vismod.render_topdown_cam(self, self.sim, [dec_dir / "local_topdown_rgb" / "robot_center"], self.demo_log)
        vismod.save_global_topdown_maps(
            nav=self,
            sim=self.sim,
            out_dir=dec_dir / "global_topdown",
            title=f"decision {dec_num:03d}",
            frontiers=list(frontiers),
            selected_frontier_idx=self.selected_frontier_idx if not bool(is_final) else None,
            agent_state=state,
            log=self.demo_log,
        )

        payload = {
            "scene_name": self.ctx.scene_name,
            "episode_id": int(self.ctx.episode_id),
            "navigation_type": self.ctx.navigation_type,
            "task_id": int(self.ctx.task_id),
            "task_level": self.ctx.task_level,
            "sentence": self.ctx.sentence,
            "decision_num": int(dec_num),
            "is_final": bool(is_final),
            "pq3d_ms": float(pq_ms),
            "frontier_count": int(len(frontiers)),
            "selected_frontier_idx": self.selected_frontier_idx,
            "target_before_modules": np.asarray(target, dtype=float).reshape(3).tolist(),
            "target_used": self.current_target.tolist(),
            "pq3d_aux": pq3d_aux,
            "tffs": tffs_info,
            "vista2mqsc": module_info,
            "agent_position": np.asarray(state.position, dtype=float).reshape(3).tolist(),
            "agent_heading": float(get_polar_angle(state)),
            "outputs": {
                "robot_eye_frames": "robot_eye_frames/",
                "scan": str((dec_dir / "scan").relative_to(self.out_dir)),
                "topdown_map": str((dec_dir / "topdown_map.png").relative_to(self.out_dir)),
                "local_topdown_rgb": str((dec_dir / "local_topdown_rgb").relative_to(self.out_dir)),
                "global_topdown": str((dec_dir / "global_topdown").relative_to(self.out_dir)),
                "topdown_scene_rgb": str((dec_dir / "global_topdown" / "topdown_scene_rgb.png").relative_to(self.out_dir)),
            },
        }
        self.latest_decision_payload = payload
        self.latest_decision_dir = dec_dir
        self.latest_decision_path = dec_dir / "decision.json"
        _write_json(self.latest_decision_path, payload)
        self.save_trajectory_snapshot("trajectory_latest.png")

        print(
            f"[demo-sim] decision {dec_num:03d}: final={bool(is_final)} "
            f"frontiers={len(frontiers)} tffs_applied={bool(tffs_info.get('tffs_applied', False))} "
            f"vista2mqsc_called={bool(module_info.get('called', False))} target={self.current_target.tolist()}",
            flush=True,
        )
        self.decision_num += 1

    def write_demo_summary(self, stop_reason: str) -> None:
        summary = {
            "ok": True,
            "scene_name": self.ctx.scene_name,
            "episode_id": int(self.ctx.episode_id),
            "navigation_type": self.ctx.navigation_type,
            "task_id": int(self.ctx.task_id),
            "task_level": self.ctx.task_level,
            "sentence": self.ctx.sentence,
            "stop_reason": stop_reason,
            "steps": int(self.step_count),
            "decisions": int(self.decision_num),
            "robot_eye_frame_count": int(self.eye_frame_index),
            "tffs_called": int(self.tffs_call_count),
            "tffs_applied": int(self.tffs_apply_count),
            "vista2mqsc_called": int(self.vista2mqsc_call_count),
            "vista2mqsc_applied": int(self.vista2mqsc_apply_count),
            "runtime_mode": self.runtime_mode_evidence(),
            "demo_log": list(self.demo_log),
        }
        _write_json(self.out_dir / "demo_visual_summary.json", summary)

    def write_error_summary(self, stop_reason: str, exc: BaseException) -> None:
        error = {
            "ok": False,
            "stop_reason": str(stop_reason),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "scene_name": self.ctx.scene_name,
            "episode_id": int(self.ctx.episode_id),
            "navigation_type": self.ctx.navigation_type,
            "task_id": int(self.ctx.task_id),
            "task_level": self.ctx.task_level,
            "sentence": self.ctx.sentence,
            "steps": int(self.step_count),
            "decisions": int(self.decision_num),
            "robot_eye_frame_count": int(self.eye_frame_index),
            "tffs_called": int(self.tffs_call_count),
            "tffs_applied": int(self.tffs_apply_count),
            "vista2mqsc_called": int(self.vista2mqsc_call_count),
            "vista2mqsc_applied": int(self.vista2mqsc_apply_count),
            "runtime_mode": self.runtime_mode_evidence(),
            "demo_log": list(self.demo_log),
        }
        _write_json(self.out_dir / "demo_visual_error.json", error)
        _write_json(self.out_dir / "demo_visual_summary.json", error)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one real visual demo for TFFS + Vista2MQSC.")
    ap.add_argument("--scene_name", default="00800-TEEsavR23oF")
    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--navigation_type", default="sequence")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--instance_id", default="sequence_ep0_task0")
    ap.add_argument("--logs_dir", default=str(CODE_DIR.parent / "logs" / "sim"))
    ap.add_argument("--live_dir", default="")
    ap.add_argument("--max_decisions", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--topdown_cam_height", type=float, default=2.0)
    ap.add_argument("--topdown_cam_hfov", type=float, default=90.0)
    ap.add_argument("--topdown_cam_res", type=int, default=384)
    ap.add_argument("--tffs_vlm_model", default=os.environ.get("TFFS_VLM_MODEL", os.environ.get("VLM_MODEL", "gpt-4o-mini")))
    ap.add_argument("--tffs_vlm_call_interval", type=int, default=5)
    ap.add_argument("--tffs_timeout_sec", type=float, default=float(os.environ.get("TFFS_TIMEOUT_SEC", "120")))
    ap.add_argument("--tffs_max_vlm_calls_per_decision", type=int, default=4)
    ap.add_argument("--tffs_min_score_margin", type=float, default=0.05)
    ap.add_argument("--tffs_min_confidence", type=float, default=0.45)
    ap.add_argument("--mqsc_r1_vlm_model", default=os.environ.get("MQSC_R1_VLM_MODEL", os.environ.get("VLM_MODEL", "gpt-4o-mini")))
    ap.add_argument("--vistals_enable_vvd_replacement", dest="vistals_enable_vvd_replacement", action="store_true", default=True)
    ap.add_argument("--vistals_disable_vvd_replacement", dest="vistals_enable_vvd_replacement", action="store_false")
    ap.add_argument("--vistals_prefer_visible_baseline", dest="vistals_prefer_visible_baseline", action="store_true", default=False)
    ap.add_argument("--vistals_disable_visible_baseline_guard", dest="vistals_prefer_visible_baseline", action="store_false")
    ap.add_argument("--vistals_apply_task_levels", default="object,room,region,instance")
    ap.add_argument("--cuda_note", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return ap.parse_args()


def main() -> None:
    cli = parse_args()
    if "," in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly one device for demo sim")

    ref_flow = _load_ref_flow()
    ref_flow.TFFS_TIMEOUT_SEC = float(cli.tffs_timeout_sec)
    ref_flow.TFFS_CFG = ref_flow.TffsConfig(
        vlm_call_interval=int(cli.tffs_vlm_call_interval),
        max_vlm_calls_per_decision=int(cli.tffs_max_vlm_calls_per_decision),
        min_score_margin=float(cli.tffs_min_score_margin),
        min_confidence=float(cli.tffs_min_confidence),
        use_vlm=True,
        vlm_model=str(cli.tffs_vlm_model),
        vlm_no_proxy=True,
    )
    ref_flow.MQSC_R1_CFG = ref_flow.MqscR1Config(
        use_vlm=True,
        vlm_model=str(cli.mqsc_r1_vlm_model),
        vlm_no_proxy=True,
    )
    ref_flow.VISTALS_CFG = ref_flow.VistaLsConfig(
        enable_vvd_replacement=bool(cli.vistals_enable_vvd_replacement),
        prefer_visible_baseline=bool(cli.vistals_prefer_visible_baseline),
    )
    ref_flow.VISTA2MQSC_VISTALS_APPLY_TASK_LEVELS = {
        str(x).strip() for x in str(cli.vistals_apply_task_levels).split(",") if str(x).strip()
    }
    vismod._M = teleop
    vismod._VIS = teleop.VIS_NAV

    teleop_argv = [
        "sim",
        "--scene_name",
        str(cli.scene_name),
        "--episode_id",
        str(cli.episode_id),
        "--navigation_type",
        str(cli.navigation_type),
        "--instance_id",
        str(cli.instance_id),
        "--task_id",
        str(cli.task_id),
        "--headless",
        "--enable_topdown_cam",
        "--topdown_cam_height",
        str(cli.topdown_cam_height),
        "--topdown_cam_hfov",
        str(cli.topdown_cam_hfov),
        "--topdown_cam_res",
        str(cli.topdown_cam_res),
        "--logs_dir",
        str(cli.logs_dir),
    ]
    if cli.live_dir:
        teleop_argv.extend(["--live_dir", str(cli.live_dir)])

    old_argv = sys.argv
    try:
        sys.argv = teleop_argv
        args = teleop.parse_args()
    finally:
        sys.argv = old_argv
    args.demo_max_steps = int(cli.max_steps)

    ctx = teleop.load_task_context(args)
    scene_path = teleop._resolve_scene_path(os.path.expanduser(args.hm3d_data_base_path), ctx.scene_name)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(os.path.expanduser(args.logs_dir))
        / f"run={run_id}"
        / f"scene={ctx.scene_name}"
        / f"navigation_type={ctx.navigation_type}"
        / f"episode={ctx.episode_id}"
        / f"task={ctx.task_id:02d}_{ctx.task_level}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo-sim] scene_path={scene_path}", flush=True)
    print(f"[demo-sim] task={ctx.task_level} sentence={ctx.sentence}", flush=True)
    print(f"[demo-sim] logs={out_dir}", flush=True)
    print(
        "[demo-sim] real modules: "
        f"TFFS interval={ref_flow.TFFS_CFG.vlm_call_interval} model={ref_flow.TFFS_CFG.vlm_model}; "
        f"MQSC model={ref_flow.MQSC_R1_CFG.vlm_model}; "
        f"Vista-LS vvd={ref_flow.VISTALS_CFG.enable_vvd_replacement} "
        f"prefer_visible_baseline={ref_flow.VISTALS_CFG.prefer_visible_baseline}; no smoke/mock",
        flush=True,
    )
    print(
        f"[demo-sim] visual sensors: topdown_height={cli.topdown_cam_height} "
        f"topdown_hfov={cli.topdown_cam_hfov} topdown_res={cli.topdown_cam_res}",
        flush=True,
    )

    try:
        preloaded_pq3d = _load_real_pq3d(args)
    except Exception as exc:
        early_error = {
            "ok": False,
            "stop_reason": f"error:{type(exc).__name__}",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "scene_name": ctx.scene_name,
            "episode_id": int(ctx.episode_id),
            "navigation_type": ctx.navigation_type,
            "task_id": int(ctx.task_id),
            "task_level": ctx.task_level,
            "sentence": ctx.sentence,
            "stage": "preload_pq3d_before_habitat",
            "steps": 0,
            "decisions": 0,
        }
        _write_json(out_dir / "demo_visual_error.json", early_error)
        _write_json(out_dir / "demo_visual_summary.json", early_error)
        raise

    sim, agent = teleop.build_interactive_simulator(args, scene_path)
    demo_log: List[str] = []
    nav = DemoNavigator(args, ctx, sim, agent, scene_path, out_dir, ref_flow=ref_flow, demo_log=demo_log)
    nav.pq3d_model = preloaded_pq3d
    nav.VIS_NAV = teleop.VIS_NAV
    nav.VIS_meters_per_px = float(teleop.maps.calculate_meters_per_pixel(int(args.map_resolution), sim=sim))
    _write_json(out_dir / "runtime_mode.json", nav.runtime_mode_evidence())

    stop_reason = "max_decisions"
    try:
        try:
            while nav.decision_num < int(cli.max_decisions) and nav.step_count < int(cli.max_steps):
                nav.run_decision_round()
                nav.follow_latest_target()
                if bool(nav.current_target_is_final):
                    stop_reason = "final_decision"
                    break
            final_pano_dir = out_dir / "modules" / "final_panorama"
            views = nav.save_extra_stop_panorama(final_pano_dir, stop_reason=stop_reason)
            demo_log.append(f"[final_panorama] stop_reason={stop_reason} views={len(views)} dir={final_pano_dir}")
            nav.finalize()
            nav.write_demo_summary(stop_reason)
        except Exception as exc:
            stop_reason = f"error:{type(exc).__name__}"
            try:
                nav.save_trajectory_snapshot("trajectory_error.png")
            except Exception as vis_exc:
                demo_log.append(f"[error_snapshot] failed: {type(vis_exc).__name__}: {vis_exc}")
            try:
                final_pano_dir = out_dir / "modules" / "final_panorama"
                views = nav.save_extra_stop_panorama(final_pano_dir, stop_reason=stop_reason)
                demo_log.append(
                    f"[final_panorama] stop_reason={stop_reason} views={len(views)} dir={final_pano_dir}"
                )
            except Exception as pano_exc:
                demo_log.append(f"[error_final_panorama] failed: {type(pano_exc).__name__}: {pano_exc}")
            nav.write_error_summary(stop_reason, exc)
            raise
    finally:
        sim.close()

    print(f"[demo-sim] complete stop_reason={stop_reason} logs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
