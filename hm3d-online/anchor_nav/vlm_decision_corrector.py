import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .vllm_adapter import VLLMClientConfig, VLLMOpenAIClient


SYSTEM_PROMPT = (
    "You are a conservative visual referee for embodied navigation. "
    "Given panorama tiles and a target description, decide whether the target object is clearly present NOW. "
    "Return strict JSON only."
)


USER_PROMPT_TEMPLATE = """Task description:
{description}

Current baseline decision:
- decision_num: {decision_num}
- baseline_target_type: {baseline_target_type}
- num_frontiers: {num_frontiers}
- memory_objects: {memory_objects}

Candidate object summaries from CURRENT round (new/updated preferred):
{candidate_text}

Question:
Should we force object-target query now (instead of frontier exploration)?
Return JSON with EXACT keys:
{{"found_target": true/false, "confidence": 0.0-1.0, "force_object_query": true/false, "reason": "short reason"}}
"""


@dataclass
class CorrectorConfig:
    enabled: bool = True
    stride: int = 2
    min_decision_num: int = 2
    confidence_threshold: float = 0.8
    max_image_tiles: int = 3
    max_candidate_objects: int = 10
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen2.5-VL-32B-Instruct"
    timeout: int = 120
    api_key: str = "EMPTY"


class VLMDecisionCorrector:
    def __init__(self, cfg: Optional[CorrectorConfig] = None) -> None:
        self.cfg = cfg or CorrectorConfig()
        self.client = VLLMOpenAIClient(
            VLLMClientConfig(
                base_url=self.cfg.base_url,
                model=self.cfg.model,
                timeout=self.cfg.timeout,
                api_key=self.cfg.api_key,
            )
        )

    def should_call(self, decision_num: int) -> bool:
        if not self.cfg.enabled:
            return False
        if decision_num < self.cfg.min_decision_num:
            return False
        if self.cfg.stride <= 1:
            return True
        return (decision_num % self.cfg.stride) == 0

    @staticmethod
    def _parse_json(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        return json.loads(text)

    def _build_candidate_text(self, objects: List[Dict[str, Any]]) -> str:
        if not objects:
            return "[]"
        rows = []
        for i, obj in enumerate(objects[: self.cfg.max_candidate_objects]):
            rows.append(
                {
                    "rank": i + 1,
                    "object_id": obj.get("object_id_in_memory"),
                    "score": round(float(obj.get("score", 0.0)), 4),
                    "count": obj.get("count"),
                    "center_xyz": obj.get("center_xyz"),
                }
            )
        return json.dumps(rows, ensure_ascii=False)

    def evaluate(
        self,
        *,
        description: str,
        decision_num: int,
        baseline_target_type: str,
        num_frontiers: int,
        memory_objects: int,
        candidate_objects: List[Dict[str, Any]],
        image_tile_paths: List[Path],
    ) -> Dict[str, Any]:
        image_tile_paths = image_tile_paths[: self.cfg.max_image_tiles]
        data_urls = [self.client.image_file_to_data_url(Path(p)) for p in image_tile_paths]
        candidate_text = self._build_candidate_text(candidate_objects)
        prompt = USER_PROMPT_TEMPLATE.format(
            description=description,
            decision_num=decision_num,
            baseline_target_type=baseline_target_type,
            num_frontiers=num_frontiers,
            memory_objects=memory_objects,
            candidate_text=candidate_text,
        )

        messages = self.client.build_messages_from_images(data_urls, prompt)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        resp = self.client.chat(messages=messages, max_tokens=256, temperature=0.0)
        raw = self.client.extract_text(resp)
        parsed = self._parse_json(raw)

        found = bool(parsed.get("found_target", False))
        conf = float(parsed.get("confidence", 0.0))
        force_object_query = bool(parsed.get("force_object_query", False))
        reason = str(parsed.get("reason", "")).strip()

        # conservative thresholding
        final_force = bool(found and force_object_query and conf >= self.cfg.confidence_threshold)
        return {
            "vlm_called": True,
            "raw_text": raw,
            "parsed": parsed,
            "found_target": found,
            "confidence": conf,
            "force_object_query": final_force,
            "reason": reason,
            "image_tiles": [str(p) for p in image_tile_paths],
        }

