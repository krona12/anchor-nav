#!/usr/bin/env python3
import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASELINE_SCRIPT = SCRIPT_DIR / "refhm3d-nav-sequence-baseline.py"

for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sam_fast_wrapper import DEFAULT_SAM_CHECKPOINT, SamFastSAM
import data_utils as _base_data_utils


def _has_cli_arg(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


sam_checkpoint = Path(os.environ.get("SAM_CHECKPOINT", str(DEFAULT_SAM_CHECKPOINT))).expanduser()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("SAM_MODEL_TYPE", "vit_h")
os.environ["SAM_CHECKPOINT"] = str(sam_checkpoint)
os.environ["FASTSAM_WEIGHT"] = str(sam_checkpoint)
_base_data_utils.FastSAM = SamFastSAM

if not _has_cli_arg("--output_log_dir"):
    sys.argv.extend(["--output_log_dir", "./output_logs/baseline_sam"])

sys.argv[0] = str(BASELINE_SCRIPT)
runpy.run_path(str(BASELINE_SCRIPT), run_name="__main__")
