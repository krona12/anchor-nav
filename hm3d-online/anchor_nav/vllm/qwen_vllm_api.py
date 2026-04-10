"""
Qwen2.5-VL 本地 vLLM OpenAI 兼容接口客户端。

用途：
1) 统一封装对 `http://<host>:<port>/v1/*` 的调用，避免业务代码重复拼装 JSON。
2) 同时支持纯文本、多模态（图片 URL / 本地图片文件）请求。
3) 提供统一错误处理，便于在上层模块做重试或告警。

注意：
- 本文件放在仓库 `vllm/` 目录下，避免与 pip 包 `vllm` 的 import 冲突，
  推荐通过“把本目录加入 sys.path”后 `from qwen_vllm_api import VLLMOpenAIClient` 使用。
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from urllib import error
from urllib import request


class VLLMAPIError(RuntimeError):
    """vLLM API 调用异常。

    Attributes:
        status_code: HTTP 状态码。若是网络层错误且拿不到状态码，为 None。
        response_text: 服务端返回体（通常包含错误详情）。
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


@dataclass
class VLLMClientConfig:
    """vLLM 客户端配置。

    对应你的服务启动参数：
    - host: 0.0.0.0
    - port: 8000
    - served model name: Qwen2.5-VL-32B-Instruct
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen2.5-VL-32B-Instruct"
    timeout: int = 120
    api_key: str = "EMPTY"


class VLLMOpenAIClient:
    """Qwen vLLM OpenAI 兼容接口客户端。

    API 设计目标：
    - 让调用方只关心“我要问什么、是否带图”，而不关心底层 JSON 结构。
    - 保留 `chat(messages=...)` 通用入口，满足复杂上层流程拼装。
    - 提供 `chat_text` / `chat_with_image_*` 快捷方法，减少样板代码。
    """

    def __init__(self, config: Optional[VLLMClientConfig] = None) -> None:
        self.config = config or VLLMClientConfig()

    # ----------------------------
    # 公共查询接口
    # ----------------------------
    def health(self) -> dict[str, Any]:
        """调用 `/health`，用于服务健康检查。"""
        return self._get_json("/health")

    def list_models(self) -> dict[str, Any]:
        """调用 `/v1/models`，返回当前服务可用模型列表。"""
        return self._get_json("/models")

    # ----------------------------
    # 核心对话接口
    # ----------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """发送 OpenAI 兼容 `chat/completions` 请求并返回完整 JSON。

        Args:
            messages:
                OpenAI 风格消息列表。常见两种：
                1) 纯文本：{"role":"user","content":"你好"}
                2) 图文：{"role":"user","content":[{"type":"image_url",...},{"type":"text",...}]}
            max_tokens:
                本次最大生成 token 数。
            temperature:
                采样温度。设为 0 附近可提升稳定性。
            top_p:
                nucleus sampling 参数。
            extra_body:
                传递 vLLM 额外参数（如视频采样参数）时使用。

        Returns:
            vLLM 返回的完整字典对象（与 OpenAI SDK 响应结构兼容）。
        """

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_body:
            payload["extra_body"] = extra_body

        return self._post_json("/chat/completions", payload)

    def chat_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """纯文本快捷调用，直接返回模型文本内容。"""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self.extract_text(resp)

    def chat_with_image_url(
        self,
        image_url: str,
        prompt: str = "请描述这张图。",
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """图片 URL 图文调用，直接返回模型文本内容。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resp = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self.extract_text(resp)

    def chat_with_image_file(
        self,
        image_path: str | Path,
        prompt: str = "请描述这张图。",
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """本地图片图文调用。

        说明：
        - OpenAI 兼容接口通常不直接接收本地文件路径；
        - 因此这里自动把本地图片转为 `data:image/...;base64,...` 再发送。
        """
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        data_url = self.image_file_to_data_url(image_path)
        return self.chat_with_image_url(
            image_url=data_url,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ----------------------------
    # 响应解析工具
    # ----------------------------
    @staticmethod
    def extract_text(response_json: dict[str, Any]) -> str:
        """从标准 chat/completions 响应中提取文本。

        若响应结构异常，抛出 ValueError，便于上游统一捕获。
        """
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected response format: {response_json}") from exc

    @staticmethod
    def image_file_to_data_url(image_path: Path) -> str:
        """把本地图片转成 data URL。"""
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{image_b64}"

    @staticmethod
    def build_messages_from_images(
        image_urls: Iterable[str],
        prompt: str,
    ) -> list[dict[str, Any]]:
        """多图辅助函数：把多张图与文本提示拼成标准 messages。"""
        content: list[dict[str, Any]] = []
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ----------------------------
    # 底层 HTTP
    # ----------------------------
    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _get_json(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        req = request.Request(url=url, headers=self._build_headers(), method="GET")
        return self._send(req)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, headers=self._build_headers(), method="POST")
        return self._send(req)

    def _send(self, req: request.Request) -> dict[str, Any]:
        try:
            with request.urlopen(req, timeout=self.config.timeout) as resp:
                text = resp.read().decode("utf-8")
                return json.loads(text) if text else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise VLLMAPIError(
                message=f"HTTPError when requesting {req.full_url}: {exc.code}",
                status_code=exc.code,
                response_text=body,
            ) from exc
        except error.URLError as exc:
            raise VLLMAPIError(
                message=f"URLError when requesting {req.full_url}: {exc.reason}",
                status_code=None,
                response_text="",
            ) from exc


if __name__ == "__main__":
    # 便于直接 `python vllm/qwen_vllm_api.py` 快速验证。
    client = VLLMOpenAIClient(
        VLLMClientConfig(
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen2.5-VL-32B-Instruct",
        )
    )
    print("models:", client.list_models())
    print("chat:", client.chat_text("你好，请用一句话介绍你自己。"))
