import json
import re
import time
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .vllm_adapter import VLLMClientConfig, VLLMOpenAIClient


SYSTEM_PROMPT = (
    "You are a conservative visual referee for embodied navigation. "
    "Focus on the MAIN target object category in the description. "
    "Do NOT be distracted by contextual/nearby objects. "
    "Given panorama tiles and a target description, decide whether the main target object is clearly present NOW. "
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
Decision criterion:
- If main target object is clearly visible in current panorama: found_target=true and force_object_query=true.
- If uncertain / ambiguous / only context objects are visible: return false.
Return JSON with EXACT keys:
{{"found_target": true/false, "confidence": 0.0-1.0, "force_object_query": true/false, "reason": "short reason"}}
"""


@dataclass
class CorrectorConfig:
    enabled: bool = True
    stride: int = 2
    min_decision_num: int = 2
    confidence_threshold: float = 0.6
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


class AsyncVLMDecisionCorrector:
    """Async wrapper that owns scheduling, pending state, and cleanup."""

    def __init__(self, corrector: VLMDecisionCorrector) -> None:
        self.corrector = corrector
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending_future: Optional[concurrent.futures.Future] = None
        self.pending_meta: Optional[Dict[str, Any]] = None
        self.calls_total: int = 0

    def should_call(self, decision_num: int) -> bool:
        return self.corrector.should_call(decision_num)

    def _run_vlm_async(
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
        t0 = time.perf_counter()
        try:
            out = self.corrector.evaluate(
                description=description,
                decision_num=decision_num,
                baseline_target_type=baseline_target_type,
                num_frontiers=num_frontiers,
                memory_objects=memory_objects,
                candidate_objects=candidate_objects,
                image_tile_paths=image_tile_paths,
            )
        except Exception as e:
            out = {"vlm_called": True, "error": str(e), "force_object_query": False}
        out["elapsed_ms"] = float((time.perf_counter() - t0) * 1000.0)
        return out

    def submit_if_needed(
        self,
        *,
        decision_num: int,
        description: str,
        baseline_target_type: str,
        num_frontiers: int,
        memory_objects: int,
        candidate_objects: List[Dict[str, Any]],
        image_tile_paths: List[Path],
        source_agent_position: Optional[List[float]],
        cleanup_tile_paths: Optional[List[Path]] = None,
        cleanup_dir: Optional[Path] = None,
    ) -> bool:
        if self.pending_future is not None:
            return False
        if not self.should_call(decision_num):
            return False
        if len(image_tile_paths) == 0:
            return False
        self.pending_future = self.executor.submit(
            self._run_vlm_async,
            description=description,
            decision_num=int(decision_num),
            baseline_target_type=baseline_target_type,
            num_frontiers=int(num_frontiers),
            memory_objects=int(memory_objects),
            candidate_objects=candidate_objects,
            image_tile_paths=image_tile_paths,
        )
        self.pending_meta = {
            "source_decision_num": int(decision_num),
            "source_agent_position": source_agent_position,
            "cleanup_tile_paths": cleanup_tile_paths or [],
            "cleanup_dir": cleanup_dir,
        }
        self.calls_total += 1
        return True

    def poll_ready(self) -> Dict[str, Any]:
        out = {
            "ready": False,
            "vlm_info": {"vlm_called": False},
            "source_decision_num": None,
            "source_agent_position": None,
        }
        if self.pending_future is None or self.pending_meta is None:
            return out
        if not self.pending_future.done():
            return out

        out["ready"] = True
        out["source_decision_num"] = int(self.pending_meta.get("source_decision_num", -1))
        out["source_agent_position"] = self.pending_meta.get("source_agent_position", None)
        try:
            out["vlm_info"] = self.pending_future.result(timeout=0.0)
        except Exception as e:
            out["vlm_info"] = {"vlm_called": True, "error": str(e), "force_object_query": False}

        for p in self.pending_meta.get("cleanup_tile_paths", []):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        d = self.pending_meta.get("cleanup_dir", None)
        if d is not None:
            try:
                d.rmdir()
            except Exception:
                pass

        self.pending_future = None
        self.pending_meta = None
        return out

    def close(self) -> None:
        if self.pending_future is not None:
            self.pending_future.cancel()
        self.executor.shutdown(wait=False)

