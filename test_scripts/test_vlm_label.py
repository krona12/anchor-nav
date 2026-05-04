#!/usr/bin/env python3
"""
Smoke test for hm3d-online/vlm client: one image + prompt, log to test_scripts/vlm_label_logs/.

Usage (from repo root):
  python test_scripts/test_vlm_label.py

Or with overrides:
  python test_scripts/test_vlm_label.py --image path/to.jpg --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "hm3d-online") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "hm3d-online"))

from vlm.client import DEFAULT_MODEL, chat  # noqa: E402

DEFAULT_IMAGE = (
    _REPO_ROOT
    / "output_logs/anchor/vfv_instance_0.05_0.1/20260503-163402-detailed/process/scene=00802-wcojb4TFT35/episode=17/task=0/panorama/vfv_dec_004.jpg"
)

DEFAULT_PROMPT = (
    "检查当前全景图是不是在一个房间内，还是在房间外，并且输出所有物体。"
    "如果是在房间内，请判断房间类型（例如卧室、客厅、厨房、浴室、走廊等），并简要说明依据。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call VLM on a panorama image and write logs.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to test image")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt text")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="VLM model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="max_tokens for completion")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "vlm_label_logs",
        help="Directory for log files",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"vlm_label_{stamp}.json"

    t0 = time.perf_counter()
    reply = chat(
        text=args.prompt,
        image_path=str(image_path),
        model=args.model,
        max_tokens=args.max_tokens,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "image": str(image_path),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "reply": reply,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote log: {log_path}")
    print("--- reply ---")
    print(reply)


if __name__ == "__main__":
    main()
