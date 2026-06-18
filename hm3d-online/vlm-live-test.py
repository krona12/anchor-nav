from __future__ import annotations

import argparse
import base64
import os
import re
import struct
import zlib
from typing import Any, Dict, Iterable, List, Mapping

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GOOGLE_BASE_URL = os.environ.get("ZZZ_GOOGLE_BASE_URL", "https://api.zhizengzeng.com/google").rstrip("/")
ALIBABA_BASE_URL = os.environ.get("ZZZ_ALIBABA_BASE_URL", "https://api.zhizengzeng.com/alibaba").rstrip("/")
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
QWEN3_VL_8B_MODEL = "qwen3-vl-8b-instruct"


def _alias_key(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", str(model or "").lower()).strip()


def normalize_model_name(model: str) -> str:
    raw = str(model or "").strip()
    key = _alias_key(raw)
    if key in {
        "gemini 2.5 flash",
        "gemini 2 5 flash",
        "gemini flash 2.5",
    }:
        return GEMINI_FLASH_MODEL
    if "qwen" in key and "3" in key and "vl" in key and "8b" in key:
        return QWEN3_VL_8B_MODEL
    if raw.lower().startswith(("gemini", "models/gemini", "qwen", "qvq")):
        return raw.lower()
    return raw


def provider_for_model(model: str) -> str:
    normalized = normalize_model_name(model).lower()
    if "gemini" in normalized:
        return "google"
    if "qwen" in normalized or "qvq" in normalized:
        return "alibaba"
    raise ValueError(f"unsupported live-test model provider for {model!r}")


def base_url_for_model(model: str) -> str:
    provider = provider_for_model(model)
    if provider == "google":
        return GOOGLE_BASE_URL
    return f"{ALIBABA_BASE_URL}/compatible-mode/v1/chat/completions"


def _api_key(provider: str) -> str:
    env_names = {
        "google": ("ZZZ_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ZZZ_API_KEY"),
        "alibaba": ("ZZZ_ALIBABA_API_KEY", "DASHSCOPE_API_KEY", "ALIBABA_API_KEY", "ZZZ_API_KEY"),
    }[provider]
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"missing API key for {provider}; set ZZZ_API_KEY first")


def _google_generate_url(model: str) -> str:
    model_name = normalize_model_name(model)
    model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
    return f"{GOOGLE_BASE_URL}/v1beta/{model_path}:generateContent"


def _extract_gemini_text(data: Mapping[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini response has no candidates: {data!r}")
    content = candidates[0].get("content", {}) if isinstance(candidates[0], Mapping) else {}
    parts = content.get("parts", []) if isinstance(content, Mapping) else []
    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, Mapping) and p.get("text")).strip()
    if not text:
        raise RuntimeError(f"Gemini response has no text part: {data!r}")
    return text


def _extract_openai_text(data: Mapping[str, Any]) -> str:
    if "error" in data:
        raise RuntimeError(f"OpenAI-compatible response error: {data.get('error')!r}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response has no choices: {data!r}")
    return str(choices[0]["message"]["content"]).strip()


def _png_solid_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw)),
            chunk(b"IEND", b""),
        ]
    )


def _parse_models(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _preview(text: str, limit: int = 240) -> str:
    return str(text).replace("\n", " ")[:limit]


def _call_text(model: str, prompt: str, *, max_tokens: int, timeout: int) -> str:
    provider = provider_for_model(model)
    normalized = normalize_model_name(model)
    if provider == "google":
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
        }
        resp = requests.post(
            _google_generate_url(normalized),
            json=payload,
            headers={"x-goog-api-key": _api_key("google"), "Content-Type": "application/json"},
            timeout=timeout,
            verify=False,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:500]}")
        return _extract_gemini_text(resp.json())

    payload = {
        "model": normalized,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(
        base_url_for_model(normalized),
        json=payload,
        headers={"Authorization": f"Bearer {_api_key('alibaba')}", "Content-Type": "application/json"},
        timeout=timeout,
        verify=False,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Alibaba HTTP {resp.status_code}: {resp.text[:500]}")
    return _extract_openai_text(resp.json())


def _call_image(model: str, png_b64: str, *, max_tokens: int, timeout: int) -> str:
    provider = provider_for_model(model)
    normalized = normalize_model_name(model)
    prompt = "What is the dominant color in this image? Answer with one English color word only."
    if provider == "google":
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": png_b64}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
        }
        resp = requests.post(
            _google_generate_url(normalized),
            json=payload,
            headers={"x-goog-api-key": _api_key("google"), "Content-Type": "application/json"},
            timeout=timeout,
            verify=False,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:500]}")
        return _extract_gemini_text(resp.json())

    payload = {
        "model": normalized,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(
        base_url_for_model(normalized),
        json=payload,
        headers={"Authorization": f"Bearer {_api_key('alibaba')}", "Content-Type": "application/json"},
        timeout=timeout,
        verify=False,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Alibaba HTTP {resp.status_code}: {resp.text[:500]}")
    return _extract_openai_text(resp.json())


def main() -> int:
    parser = argparse.ArgumentParser("Live-test Zhizengzeng Gemini/Qwen VLM calls")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gemini-2.5-flash", "qwen vl 3 vl 8b", "qwen-vl-plus"],
        help="Model names or comma-separated model names. Calls the real API.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not any(os.environ.get(k, "").strip() for k in ("ZZZ_API_KEY", "ZZZ_GOOGLE_API_KEY", "ZZZ_ALIBABA_API_KEY")):
        raise SystemExit("Set ZZZ_API_KEY first. This script intentionally has no mock mode.")

    red_png = base64.b64encode(_png_solid_rgb(16, 16, (255, 0, 0))).decode("ascii")
    failures = 0
    for model in _parse_models(args.models):
        normalized = normalize_model_name(model)
        provider = provider_for_model(normalized)
        print(f"=== {model} -> {normalized} ({provider}) ===")
        print(f"base_url={base_url_for_model(normalized)}")

        try:
            text = _call_text(
                model,
                "Return strict JSON only with exactly these keys: ok, model_family. ok must be true.",
                max_tokens=256,
                timeout=int(args.timeout),
            )
            print("text_ok", _preview(text))
        except Exception as exc:
            failures += 1
            print("text_error", type(exc).__name__, _preview(str(exc), 500))

        try:
            image_text = _call_image(
                model,
                red_png,
                max_tokens=128,
                timeout=int(args.timeout),
            )
            print("image_ok", _preview(image_text, 120))
        except Exception as exc:
            failures += 1
            print("image_error", type(exc).__name__, _preview(str(exc), 500))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
