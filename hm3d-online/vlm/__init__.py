"""
vlm - Qwen3-VL-Flash 调用模块
"""

from .vlm_client import VLMClient, VLMResult
from .config import DEFAULT_MODEL, AVAILABLE_MODELS

__all__ = ["VLMClient", "VLMResult", "DEFAULT_MODEL", "AVAILABLE_MODELS"]

# 模块级快捷实例（开箱即用）
_default_client = None


def _get_client() -> VLMClient:
    global _default_client
    if _default_client is None:
        _default_client = VLMClient()
    return _default_client


def chat(prompt: str, images=None, verbose: bool = False, **kwargs) -> VLMResult:
    """模块级快捷调用，无需实例化"""
    return _get_client().chat(prompt, images=images, verbose=verbose, **kwargs)
