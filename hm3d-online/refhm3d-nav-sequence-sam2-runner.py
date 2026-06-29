#!/usr/bin/env python3
import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

sam2_repo_root = os.environ.get("SAM2_REPO_ROOT", "").strip()
if sam2_repo_root:
    sam2_repo_root = str(Path(sam2_repo_root).expanduser().resolve())
    if sam2_repo_root not in sys.path:
        sys.path.insert(0, sam2_repo_root)

from sam2_fast_wrapper import DEFAULT_SAM2_CHECKPOINT, Sam2FastSAM
import data_utils as _base_data_utils


def _usage():
    print(
        "Usage: python hm3d-online/refhm3d-nav-sequence-sam2-runner.py "
        "<target_refhm3d_script.py> [target args...]",
        file=sys.stderr,
    )


if len(sys.argv) < 2:
    _usage()
    raise SystemExit(2)

target_script = Path(sys.argv[1])
if not target_script.is_absolute():
    target_script = PROJECT_ROOT / target_script
target_script = target_script.resolve()

if not target_script.is_file():
    print(f"SAM2 target script not found: {target_script}", file=sys.stderr)
    raise SystemExit(2)

sam2_checkpoint = Path(os.environ.get("SAM2_CHECKPOINT", str(DEFAULT_SAM2_CHECKPOINT))).expanduser()
os.environ["SAM2_CHECKPOINT"] = str(sam2_checkpoint)
os.environ["FASTSAM_WEIGHT"] = str(sam2_checkpoint)
_base_data_utils.FastSAM = Sam2FastSAM

print(
    f"[sam2-runner] target={target_script} checkpoint={sam2_checkpoint} "
    f"preset={os.environ.get('SAM2_LEVEL_PRESET', 'balanced')}",
    flush=True,
)

sys.argv = [str(target_script), *sys.argv[2:]]
runpy.run_path(str(target_script), run_name="__main__")
