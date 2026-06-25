#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHECK_SH = SCRIPT_DIR / "run_navi_visual_module_check.sh"
DEFAULT_NAV_ROOT = PROJECT_ROOT / "LangMap_Annotations"
DEFAULT_BATCH_ROOT = SCRIPT_DIR / "navi-visual" / "logs" / "batch"

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


@dataclass
class BatchTask:
    index: int
    scene_name: str
    episode_id: int
    task_id: int
    task_pair: Tuple[str, int]

    @property
    def scene_dir_name(self) -> str:
        return self.scene_name.replace("/", "_")

    @property
    def task_label(self) -> str:
        level, idx = self.task_pair
        return f"task={self.task_id:02d}_{level}={idx}"


def _read_scene(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"scene json is not an object: {path}")
    return data


def build_manifest(nav_root: Path, scene_limit: int = 10) -> List[BatchTask]:
    tasks: List[BatchTask] = []
    scene_files = sorted(nav_root.glob("*.json.gz"))[: int(scene_limit)]
    for scene_file in scene_files:
        scene_name = scene_file.name[: -len(".json.gz")]
        scene = _read_scene(scene_file)
        chosen = None
        for episode in list(scene.get("episode_by_sequence", [])):
            seq = list(episode.get("task_sequence", []))
            if len(seq) == 5:
                chosen = episode
                break
        if chosen is None:
            raise RuntimeError(f"no 5-subtask sequence episode found in {scene_file}")
        episode_id = int(chosen["episode_id"])
        for task_id, pair in enumerate(chosen["task_sequence"][:5]):
            level, idx = pair
            tasks.append(
                BatchTask(
                    index=len(tasks),
                    scene_name=scene_name,
                    episode_id=episode_id,
                    task_id=int(task_id),
                    task_pair=(str(level), int(idx)),
                )
            )
    return tasks


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _require_single_cuda_device() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to one device for the serial batch run")
    if "," in visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must name exactly one device, got {visible!r}")


def _require_real_batch_mode(args: argparse.Namespace, batch_root: Path) -> None:
    if bool(args.dry_run):
        raise RuntimeError("--dry_run is not allowed for the real LangMap batch")
    root_parts = {part.lower() for part in batch_root.parts}
    if any("smoke" in part for part in root_parts):
        raise RuntimeError(f"smoke output paths are not allowed for the real LangMap batch: {batch_root}")
    bad_env = []
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name in DISALLOWED_MOCK_ENV_NAMES and str(value).strip() not in ("", "0", "false", "False"):
            bad_env.append(f"{name}={value}")
    if bad_env:
        raise RuntimeError("mock/fake VLM or simulator environment is not allowed: " + ", ".join(sorted(bad_env)))


def task_dir(batch_root: Path, task: BatchTask) -> Path:
    return batch_root / task.scene_dir_name / f"episode={task.episode_id:03d}" / task.task_label


def latest_run_dir(tdir: Path) -> Optional[Path]:
    runs = sorted([p for p in tdir.glob("run=*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def task_status_path(tdir: Path) -> Path:
    return tdir / "batch_task_status.json"


def task_is_done(tdir: Path) -> bool:
    status = _read_json(task_status_path(tdir))
    return bool(status.get("ok")) and str(status.get("state")) == "done"


def validate_run(run_dir: Path, log_path: Path) -> Tuple[bool, str]:
    cmd = [str(CHECK_SH), "--skip_run", "--run_dir", str(run_dir)]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n[batch-validate]\n")
        f.write(proc.stdout)
    return proc.returncode == 0, proc.stdout


def quick_artifact_check(run_dir: Path) -> List[str]:
    issues: List[str] = []
    required = [
        run_dir / "module_sim_summary.json",
        run_dir / "module_visual_check_report.json",
        run_dir / "modules" / "final_panorama" / "current_decision_panorama_vfv_order.jpg",
        run_dir / "modules" / "vista_ls" / "final" / "topdown_scene_rgb_mosaic.png",
        run_dir / "trajectory" / "route_start_to_goal.png",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size <= 0:
            issues.append(f"missing/empty required artifact: {path}")
    report = _read_json(run_dir / "module_visual_check_report.json")
    if report and not bool(report.get("ok", False)):
        issues.append(f"module_visual_check_report not ok: {run_dir / 'module_visual_check_report.json'}")
    return issues


def health_check(batch_root: Path, tasks: List[BatchTask], running_task: Optional[BatchTask]) -> List[str]:
    issues: List[str] = []
    completed = 0
    for task in tasks:
        tdir = task_dir(batch_root, task)
        status = _read_json(task_status_path(tdir))
        state = str(status.get("state", "pending"))
        if state == "done" and bool(status.get("ok")):
            completed += 1
            run_dir = Path(status.get("run_dir", ""))
            if run_dir.exists():
                issues.extend(quick_artifact_check(run_dir))
            else:
                issues.append(f"done task has missing run_dir: {task.scene_name} task={task.task_id} {run_dir}")
        elif running_task is not None and task.index == running_task.index:
            log_path = tdir / "module_check_stdout.log"
            if log_path.exists() and time.time() - log_path.stat().st_mtime > 900:
                issues.append(f"running task log has not changed for >15min: {log_path}")
        elif state == "failed":
            issues.append(f"task failed: scene={task.scene_name} episode={task.episode_id} task={task.task_id}")
    summary = {"completed": completed, "total": len(tasks), "issues": issues}
    _write_json(batch_root / "batch_health_latest.json", summary)
    return issues


def run_one_task(
    *,
    task: BatchTask,
    all_tasks: List[BatchTask],
    batch_root: Path,
    max_rounds: int,
    segment_advance_m: float,
    retry_limit: int,
    check_interval_sec: int,
    dry_run: bool,
) -> bool:
    tdir = task_dir(batch_root, task)
    tdir.mkdir(parents=True, exist_ok=True)
    if task_is_done(tdir):
        return True

    status_path = task_status_path(tdir)
    attempts = int(_read_json(status_path).get("attempts", 0) or 0)
    if attempts > retry_limit:
        return False
    attempts += 1
    _write_json(
        status_path,
        {
            "state": "running",
            "ok": False,
            "attempts": attempts,
            "scene_name": task.scene_name,
            "episode_id": task.episode_id,
            "task_id": task.task_id,
            "task_pair": list(task.task_pair),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    cmd = [
        str(CHECK_SH),
        "--scene_name",
        task.scene_name,
        "--episode_id",
        str(task.episode_id),
        "--navigation_type",
        "sequence",
        "--instance_id",
        f"sequence_ep{task.episode_id}_task{task.task_id}",
        "--task_id",
        str(task.task_id),
        "--max_rounds",
        str(max_rounds),
        "--segment_advance_m",
        str(segment_advance_m),
        "--logs_dir",
        str(tdir),
    ]
    _write_json(
        tdir / "batch_command.json",
        {
            "cmd": cmd,
            "cwd": str(PROJECT_ROOT),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "mode": "real_serial_vlm",
        },
    )
    if dry_run:
        _write_json(status_path, {**_read_json(status_path), "state": "dry_run", "cmd": cmd})
        return True

    log_path = tdir / "module_check_stdout.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[batch-run] attempt={attempts} task={task}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        last_check = time.time()
        while proc.poll() is None:
            now = time.time()
            if now - last_check >= max(1, int(check_interval_sec)):
                issues = health_check(batch_root, all_tasks, running_task=task)
                _write_json(
                    batch_root / "batch_progress.json",
                    {
                        "state": "running",
                        "running_task": {
                            "index": task.index,
                            "scene_name": task.scene_name,
                            "episode_id": task.episode_id,
                            "task_id": task.task_id,
                            "task_pair": list(task.task_pair),
                        },
                        "last_issues": issues,
                        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                last_check = now
            time.sleep(1.0)
        returncode = int(proc.returncode)

    run_dir = latest_run_dir(tdir)
    ok = returncode == 0 and run_dir is not None
    validation_output = ""
    if ok and run_dir is not None:
        ok, validation_output = validate_run(run_dir, log_path)
    status = {
        "state": "done" if ok else "failed",
        "ok": bool(ok),
        "attempts": attempts,
        "returncode": returncode,
        "scene_name": task.scene_name,
        "episode_id": task.episode_id,
        "task_id": task.task_id,
        "task_pair": list(task.task_pair),
        "run_dir": "" if run_dir is None else str(run_dir.resolve()),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_tail": validation_output[-2000:],
    }
    _write_json(status_path, status)
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the first LangMap multi-goal sequence for the first 10 scenes.")
    ap.add_argument("--navigation_data_path", default=str(DEFAULT_NAV_ROOT))
    ap.add_argument("--batch_root", default=str(DEFAULT_BATCH_ROOT))
    ap.add_argument("--scene_limit", type=int, default=10)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--segment_advance_m", type=float, default=1.0)
    ap.add_argument("--check_interval_sec", type=int, default=300)
    ap.add_argument("--clean_checks_to_exit", type=int, default=5)
    ap.add_argument("--retry_limit", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    nav_root = Path(args.navigation_data_path).expanduser().resolve()
    batch_root = Path(args.batch_root).expanduser().resolve()
    _require_single_cuda_device()
    _require_real_batch_mode(args, batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    tasks = build_manifest(nav_root, scene_limit=int(args.scene_limit))
    _write_json(
        batch_root / "batch_manifest.json",
        {
            "navigation_data_path": str(nav_root),
            "scene_limit": int(args.scene_limit),
            "total_tasks": len(tasks),
            "tasks": [
                {
                    "index": t.index,
                    "scene_name": t.scene_name,
                    "episode_id": t.episode_id,
                    "task_id": t.task_id,
                    "task_pair": list(t.task_pair),
                    "task_dir": str(task_dir(batch_root, t)),
                }
                for t in tasks
            ],
        },
    )

    clean_streak = 0
    last_check = 0.0
    for task in tasks:
        while True:
            ok = run_one_task(
                task=task,
                all_tasks=tasks,
                batch_root=batch_root,
                max_rounds=int(args.max_rounds),
                segment_advance_m=float(args.segment_advance_m),
                retry_limit=int(args.retry_limit),
                check_interval_sec=int(args.check_interval_sec),
                dry_run=bool(args.dry_run),
            )
            if ok:
                break
            status = _read_json(task_status_path(task_dir(batch_root, task)))
            if int(status.get("attempts", 0) or 0) > int(args.retry_limit):
                _write_json(batch_root / "batch_failed.json", {"failed_task": status})
                return 1
        now = time.time()
        if now - last_check >= max(1, int(args.check_interval_sec)):
            issues = health_check(batch_root, tasks, running_task=None)
            clean_streak = clean_streak + 1 if not issues else 0
            last_check = now
            _write_json(batch_root / "batch_progress.json", {"clean_streak": clean_streak, "last_issues": issues})

    while clean_streak < int(args.clean_checks_to_exit):
        issues = health_check(batch_root, tasks, running_task=None)
        clean_streak = clean_streak + 1 if not issues else 0
        _write_json(batch_root / "batch_progress.json", {"clean_streak": clean_streak, "last_issues": issues})
        if clean_streak >= int(args.clean_checks_to_exit):
            break
        time.sleep(max(1, int(args.check_interval_sec)))

    _write_json(
        batch_root / "batch_done.json",
        {
            "ok": True,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clean_streak": clean_streak,
            "total_tasks": len(tasks),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
