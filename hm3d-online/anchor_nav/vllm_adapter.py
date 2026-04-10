import json
import re
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
VLLM_DIR = CURRENT_DIR / "vllm"
if str(VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_DIR))

from qwen_vllm_api import VLLMClientConfig, VLLMOpenAIClient  # noqa: E402


SYSTEM_PROMPT = (
    "You are an information extraction assistant for embodied navigation. "
    "Always return strict JSON only."
)


USER_PROMPT_TEMPLATE = """Given a navigation description, extract:
1) target: final target object category in English
2) anchor: a nearby anchor object phrase in English (empty string if missing)
3) room: room/category phrase in English (empty string if missing)

Return JSON object with exactly these keys:
{{"target":"...","anchor":"...","room":"..."}}

Description:
{description}
"""


class VLLMDescriptionAdapter:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen2.5-VL-32B-Instruct",
        timeout: int = 120,
        api_key: str = "EMPTY",
    ) -> None:
        self.client = VLLMOpenAIClient(
            VLLMClientConfig(
                base_url=base_url,
                model=model,
                timeout=timeout,
                api_key=api_key,
            )
        )

    def parse_description(self, description: str) -> dict:
        prompt = USER_PROMPT_TEMPLATE.format(description=description)
        raw_text = self.client.chat_text(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.0,
        )
        parsed = self._parse_json(raw_text)
        self._validate(parsed, raw_text)
        return parsed

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        text = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        return json.loads(text)

    @staticmethod
    def _validate(parsed: dict, raw_text: str) -> None:
        if not isinstance(parsed, dict):
            raise RuntimeError(f"VLLM output is not a JSON object: {raw_text}")
        required_keys = ("target", "anchor", "room")
        for key in required_keys:
            if key not in parsed:
                raise RuntimeError(f"VLLM output missing key '{key}': {raw_text}")
            if not isinstance(parsed[key], str):
                raise RuntimeError(f"VLLM output key '{key}' must be string: {raw_text}")
