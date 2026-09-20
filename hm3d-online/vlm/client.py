import base64
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

# NOTE for future VLM module development:
# Some servers export HTTP(S)/ALL proxy variables globally, and this endpoint may
# fail or hang when requests goes through that proxy. If a module needs direct
# no-proxy VLM calls, clear HTTP_PROXY/HTTPS_PROXY/ALL_PROXY (upper and lower
# case) around the call and set NO_PROXY=no_proxy="*"; restore the original
# environment afterwards. This client intentionally preserves requests' default
# environment behavior so existing modules keep their current network semantics.

BASE_URL = "https://api.zhizengzeng.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def _current_api_key():
    key = os.environ.get("ZZZ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ZZZ_API_KEY before running a VLM analysis template.")
    return key


def _post_chat(payload: Dict[str, Any], timeout: int = 60) -> str:
    resp = requests.post(
        BASE_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_current_api_key()}", "Content-Type": "application/json"},
        timeout=timeout,
        verify=True,
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

    Proxy note:
    - 本函数默认遵循 requests / 环境变量里的 proxy 设置。
    - 若服务器 proxy 会导致 VLM API 失败，调用侧应使用 no-proxy 包装：
      临时清除 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/http_proxy/https_proxy/all_proxy，
      并设置 NO_PROXY/no_proxy="*"，调用结束后恢复原环境。
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
    """多模态 messages 直传接口，供 rerank 等模块使用。

    Proxy note: 默认使用当前环境 proxy；需要直连时请在调用侧临时清除
    HTTP(S)/ALL proxy 并设置 NO_PROXY/no_proxy="*"。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _post_chat(payload, timeout=timeout)
