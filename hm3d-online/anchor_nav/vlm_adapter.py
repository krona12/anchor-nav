import base64
import json
import os
from typing import Any, List, Optional

import cv2
import numpy as np


VLM_BASE_URL = os.environ.get("ANCHORNAV_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLM_API_KEY = os.environ.get("ANCHORNAV_VLM_API_KEY", "")
VLM_MODEL_FAST = os.environ.get("ANCHORNAV_VLM_MODEL_FAST", "qwen3.5-flash")
VLM_MODEL_BEST = os.environ.get("ANCHORNAV_VLM_MODEL_BEST", "qwen3.5-plus")


def _encode_image_to_data_url(img: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode image for VLM request")
    payload = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def _init_openai_client():
    if not VLM_API_KEY:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)


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
    if client is None:
        return None
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_vlm_image(images: List[np.ndarray], prompt: str, model: str = VLM_MODEL_FAST) -> Optional[str]:
    client = _init_openai_client()
    if client is None:
        return None
    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_url(img)}})
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content

