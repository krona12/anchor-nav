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

from sam_fast_wrapper import DEFAULT_SAM_CHECKPOINT, SamFastSAM
import data_utils as _base_data_utils


def _usage():
    print(
        "Usage: python hm3d-online/refhm3d-nav-sequence-sam-runner.py "
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
    print(f"SAM target script not found: {target_script}", file=sys.stderr)
    raise SystemExit(2)

sam_checkpoint = Path(os.environ.get("SAM_CHECKPOINT", str(DEFAULT_SAM_CHECKPOINT))).expanduser()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SAM_MODEL_TYPE", "vit_h")
os.environ["SAM_CHECKPOINT"] = str(sam_checkpoint)
os.environ["FASTSAM_WEIGHT"] = str(sam_checkpoint)
_base_data_utils.FastSAM = SamFastSAM

print(
    f"[sam-runner] target={target_script} "
    f"model_type={os.environ.get('SAM_MODEL_TYPE')} checkpoint={sam_checkpoint}",
    flush=True,
)

sys.argv = [str(target_script), *sys.argv[2:]]
runpy.run_path(str(target_script), run_name="__main__")
