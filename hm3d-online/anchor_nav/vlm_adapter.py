import base64
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any, List, Optional

import cv2
import numpy as np

VLM_BASE_URL = os.environ.get("ANCHORNAV_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLM_API_KEY = os.environ.get("ANCHORNAV_VLM_API_KEY", "")
VLM_MODEL_FAST = os.environ.get("ANCHORNAV_VLM_MODEL_FAST", "qwen3.5-flash")
VLM_MODEL_BEST = os.environ.get("ANCHORNAV_VLM_MODEL_BEST", "qwen3.5-flash")
VLM_VERBOSE = os.environ.get("ANCHORNAV_VLM_VERBOSE", "1").strip().lower() not in {"0", "false", "off", "no"}

_client = None
_init_attempted = False
_warned_disabled = False
_call_seq = 0


def _next_call_id() -> int:
    global _call_seq
    _call_seq += 1
    return _call_seq


def _encode_image_to_data_url(img: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode image for VLM request")
    payload = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def _init_openai_client():
    global _client, _init_attempted, _warned_disabled
    if _client is not None:
        return _client
    if _init_attempted:
        return None
    _init_attempted = True

    api_key = VLM_API_KEY
    base_url = VLM_BASE_URL
    if not api_key:
        # Fallback 1: import config module directly from file path (avoid package __init__ side effects).
        try:
            cfg_file = Path(__file__).resolve().parents[1] / "vlm" / "config.py"
            if cfg_file.exists():
                spec = importlib.util.spec_from_file_location("anchornav_vlm_config", str(cfg_file))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    api_key = getattr(module, "API_KEY", "") or api_key
                    base_url = getattr(module, "BASE_URL", "") or base_url
        except Exception:
            pass
    if not api_key:
        raise RuntimeError(
            "AnchorNav VLM is required but no API key was found. "
            "Set ANCHORNAV_VLM_API_KEY or provide hm3d-online/vlm/config.py with API_KEY."
        )

    try:
        from openai import OpenAI
    except Exception as err:
        raise RuntimeError(f"AnchorNav VLM is required but OpenAI client import failed: {err}")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    if VLM_VERBOSE:
        print(f"[AnchorNav/VLM] enabled: base_url={base_url}, fast={VLM_MODEL_FAST}, best={VLM_MODEL_BEST}")
    return _client


def ensure_vlm_ready() -> None:
    _init_openai_client()


def probe_vlm_or_raise() -> None:
    client = _init_openai_client()
    call_id = _next_call_id()
    t0 = time.perf_counter()
    if VLM_VERBOSE:
        print(f"[AnchorNav/VLM] call#{call_id} probe start model={VLM_MODEL_FAST}")
    resp = client.chat.completions.create(
        model=VLM_MODEL_FAST,
        messages=[{"role": "user", "content": 'Reply ONLY JSON: {"ok": true}'}],
    )
    text = resp.choices[0].message.content
    parsed = extract_json(text)
    if not (isinstance(parsed, dict) and parsed.get("ok") is True):
        raise RuntimeError(f"AnchorNav VLM probe failed: unexpected response: {text}")
    if VLM_VERBOSE:
        print(f"[AnchorNav/VLM] call#{call_id} probe passed elapsed={time.perf_counter() - t0:.3f}s")


def extract_json(response: str) -> Any:
    text = response.strip()
    if "```" in text:
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = []
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start : obj_end + 1])

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(text[arr_start : arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise json.JSONDecodeError("cannot parse JSON from model response", text, 0)


def call_vlm_text(prompt: str, model: str = VLM_MODEL_BEST) -> Optional[str]:
    client = _init_openai_client()
    call_id = _next_call_id()
    t0 = time.perf_counter()
    if VLM_VERBOSE:
        print(
            f"[AnchorNav/VLM] call#{call_id} text request model={model}, prompt_chars={len(prompt)}"
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if VLM_VERBOSE:
        print(f"[AnchorNav/VLM] call#{call_id} text response elapsed={time.perf_counter() - t0:.3f}s")
    return resp.choices[0].message.content


def call_vlm_image(images: List[np.ndarray], prompt: str, model: str = VLM_MODEL_FAST) -> Optional[str]:
    client = _init_openai_client()
    call_id = _next_call_id()
    t0 = time.perf_counter()
    if VLM_VERBOSE:
        print(
            f"[AnchorNav/VLM] call#{call_id} image request model={model}, "
            f"images={len(images)}, prompt_chars={len(prompt)}"
        )
    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(img)}})
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    if VLM_VERBOSE:
        print(f"[AnchorNav/VLM] call#{call_id} image response elapsed={time.perf_counter() - t0:.3f}s")
    return resp.choices[0].message.content

