import base64
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ.get("ZZZ_API_KEY", "sk-zk28106fd788bebc27a554683dd2777e6b668a40d4167fa9")
BASE_URL = "https://api.zhizengzeng.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def _current_api_key() -> str:
    return os.environ.get("ZZZ_API_KEY", API_KEY)


def _post_chat(payload: Dict[str, Any], timeout: int = 60) -> str:
    resp = requests.post(
        BASE_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_current_api_key()}", "Content-Type": "application/json"},
        timeout=timeout,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def chat(
    text: str,
    image_path: Optional[Union[str, Path]] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
) -> str:
    """
    发送请求到 VLM，返回回复文本。
    image_path: 本地图片路径（可选）
    """
    content = []

    if image_path is not None:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        suffix = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    content.append({"type": "text", "text": text})

    return _post_chat(
        {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens},
        timeout=60,
    )


def chat_messages(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """多模态 messages 直传接口，供 rerank 等模块使用。"""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _post_chat(payload, timeout=timeout)
