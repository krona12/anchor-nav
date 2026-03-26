"""
VLM 调用模块 - 基于阿里云百炼 Qwen3-VL-Flash
支持：单图、多图、URL图片、本地图片、纯文本
"""

import base64
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

from openai import OpenAI


def _safe_print(*args, **kwargs):
    """Print with fallback encoding to avoid UnicodeEncodeError on Windows GBK consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        end = kwargs.get("end", "\n")
        flush = kwargs.get("flush", False)
        text = " ".join(str(a) for a in args)
        sys.stdout.flush()  # flush buffered output first to preserve ordering
        sys.stdout.buffer.write((text + end).encode("utf-8", errors="replace"))
        if flush:
            sys.stdout.buffer.flush()

from .config import API_KEY, BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE


@dataclass
class VLMResult:
    """VLM 调用结果，包含回复文本与计时信息"""
    text: str                          # 模型回复文本
    elapsed: float                     # 总耗时（秒）
    time_to_first_token: float = None  # 首 token 延迟（仅流式模式有值）
    prompt: str = ""                   # 原始问题（verbose 时记录）
    images: list = field(default_factory=list)  # 输入图片列表

    def __str__(self):
        return self.text


def _load_image_as_base64(image_path: str) -> str:
    """将本地图片转为 base64 data URL"""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _build_image_content(image: str) -> dict:
    """根据输入自动判断是 URL 还是本地路径，构建 image_url content 块"""
    if image.startswith("http://") or image.startswith("https://"):
        url = image
    else:
        url = _load_image_as_base64(image)
    return {"type": "image_url", "image_url": {"url": url}}


class VLMClient:
    """
    Qwen VLM 客户端

    快速使用：
        from vlm import VLMClient
        vlm = VLMClient()
        result = vlm.chat("Describe this image.", images="path/to/image.jpg")
        print(result)           # 直接打印文本
        print(result.elapsed)   # 查看耗时
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        on_timing: Callable[[VLMResult], None] = None,
    ):
        """
        Args:
            api_key:    阿里云 API Key
            base_url:   API 端点
            model:      模型名称
            verbose:    全局默认是否打印 [Q] / [A] / 耗时信息
            on_timing:  计时回调，每次调用完成后触发，接收 VLMResult
        """
        self.model = model
        self.verbose = verbose
        self.on_timing = on_timing
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        prompt: str,
        images: Union[str, list[str], None] = None,
        system: str = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stream: bool = False,
        verbose: bool = None,
    ) -> VLMResult:
        """
        发送 VLM 请求。

        Args:
            prompt:      用户问题或指令
            images:      图片路径/URL（单张传字符串，多张传列表）
            system:      可选的系统提示词
            max_tokens:  最大输出 token 数
            temperature: 采样温度
            stream:      是否流式输出
            verbose:     是否打印 [Q]/[A]/耗时（覆盖实例级 self.verbose）

        Returns:
            VLMResult  包含 .text（回复文本）、.elapsed（总耗时）、
                       .time_to_first_token（首 token 延迟，仅流式有值）
        """
        show = self.verbose if verbose is None else verbose

        # 归一化图片列表
        img_list = []
        if images:
            img_list = [images] if isinstance(images, str) else list(images)

        if show:
            img_hint = f"  [{len(img_list)} image(s)]" if img_list else ""
            _safe_print(f"[Q]{img_hint} {prompt}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        content = [_build_image_content(img) for img in img_list]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        t_start = time.perf_counter()

        if stream:
            text, ttft = self._stream_response(kwargs, show=show)
        else:
            completion = self.client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content
            ttft = None
            if show:
                _safe_print(f"[A] {text}")

        elapsed = time.perf_counter() - t_start

        if show:
            ttft_str = f", TTFT {ttft:.2f}s" if ttft is not None else ""
            _safe_print(f"[Time] {elapsed:.2f}s{ttft_str}")

        result = VLMResult(
            text=text,
            elapsed=elapsed,
            time_to_first_token=ttft,
            prompt=prompt,
            images=img_list,
        )

        if self.on_timing:
            self.on_timing(result)

        return result

    def _stream_response(self, kwargs: dict, show: bool) -> tuple[str, float]:
        """流式输出，返回 (完整文本, 首token延迟)"""
        full_text = ""
        ttft = None
        t_start = time.perf_counter()

        if show:
            _safe_print("[A] ", end="", flush=True)

        with self.client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    if show:
                        _safe_print(delta, end="", flush=True)
                    full_text += delta

        if show:
            _safe_print()  # 换行
        return full_text, ttft

    def describe_image(self, image: str, **kwargs) -> VLMResult:
        """描述单张图片内容（快捷方法）"""
        return self.chat("Describe the content of this image in detail.", images=image, **kwargs)

    def compare_images(self, images: list[str], question: str = "Compare these images and describe the differences.", **kwargs) -> VLMResult:
        """对比多张图片（快捷方法）"""
        return self.chat(question, images=images, **kwargs)

    def analyze_document(self, image: str, question: str = "Parse and extract all content from this document.", **kwargs) -> VLMResult:
        """文档/表格分析（快捷方法）"""
        return self.chat(question, images=image, **kwargs)
