import json
import os
import re
import sys
import time
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_VLLM_DIR = Path(__file__).resolve().parent / "vllm"
if str(_VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(_VLLM_DIR))
from qwen_vllm_api import VLLMAPIError  # noqa: E402

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
In the CURRENT panorama tiles, is the MAIN target object clearly visible right now?
Decision criterion:
- If clearly visible: found_target=true with high confidence.
- If uncertain, ambiguous, occluded, or only similar/context objects: found_target=false (lower confidence is OK).
Return JSON with EXACT keys:
{{"found_target": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}
"""


@dataclass
class CorrectorConfig:
    enabled: bool = True
    stride: int = 2
    min_decision_num: int = 2
    confidence_threshold: float = 0.8
    # 压制 baseline 错误 object-commit（found=false→frontier）的 conf 下限；None 则与 confidence_threshold 相同
    suppress_commit_conf_threshold: Optional[float] = None
    max_image_tiles: int = 3
    max_candidate_objects: int = 10
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen2.5-VL-32B-Instruct"
    timeout: int = 120
    api_key: str = "EMPTY"


def _as_xyz(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    v = np.asarray(arr, dtype=float).reshape(-1)
    if v.size < 3:
        return None
    return v[:3].copy()


def _nearest_frontier_xyz(frontiers: List[Any], agent_position: Any) -> Optional[np.ndarray]:
    if not frontiers:
        return None
    ap = _as_xyz(agent_position)
    if ap is None:
        return None
    best: Optional[np.ndarray] = None
    best_d = float("inf")
    for w in frontiers:
        p = _as_xyz(w)
        if p is None:
            continue
        d = float(np.linalg.norm(p - ap))
        if d < best_d:
            best_d = d
            best = p
    return best


def resolve_navigation_after_vlm(
    *,
    baseline_is_final: bool,
    baseline_target: Any,
    frontiers: List[Any],
    agent_position: Any,
    vlm_info: Dict[str, Any],
    cfg: CorrectorConfig,
    memory_top_xyz: Optional[Any] = None,
    async_agent_xyz: Optional[Any] = None,
    sync_vlm_round: bool = True,
) -> Dict[str, Any]:
    """
    在 baseline 导航决策之后，根据 VLM「是否看到主目标」做纯决策纠偏（无几何定位）。

   仅当 sync_vlm_round=True（通常即同步 VLM）时合并，避免异步结果与当前轮次错配。

    - baseline=frontier（非 final）+ found + conf≥阈值 → 强制 commit：目标为记忆 top → 异步源位姿 → 当前智能体位姿。
    - baseline=object（final）+ not found + conf≥压制阈值 → 压制 commit：改最近 frontier 继续探索。
    - 其余 → 不干预。
    """
    t0 = _as_xyz(baseline_target)
    if t0 is None:
        t0 = np.zeros(3, dtype=float)
    target = t0.copy()
    final = bool(baseline_is_final)
    navigation_corrected = False
    vlm_force_applied = False
    vlm_bidir_object_final: Optional[str] = None

    if not isinstance(vlm_info, dict) or not vlm_info.get("vlm_called") or vlm_info.get("error"):
        return {
            "corrected_target": target,
            "corrected_final": final,
            "navigation_corrected": navigation_corrected,
            "vlm_force_applied": vlm_force_applied,
            "vlm_bidir_object_final": vlm_bidir_object_final,
        }

    if not sync_vlm_round:
        return {
            "corrected_target": target,
            "corrected_final": final,
            "navigation_corrected": navigation_corrected,
            "vlm_force_applied": vlm_force_applied,
            "vlm_bidir_object_final": vlm_bidir_object_final,
        }

    commit_th = float(cfg.confidence_threshold)
    suppress_th = (
        float(cfg.suppress_commit_conf_threshold)
        if cfg.suppress_commit_conf_threshold is not None
        else commit_th
    )
    conf_v = float(vlm_info.get("confidence", 0.0))
    found_v = bool(vlm_info.get("found_target", False))

    # frontier + 高置信看到目标 → 强制 commit（原地停 / 记忆 top）
    if (not baseline_is_final) and found_v and conf_v >= commit_th:
        src = _as_xyz(memory_top_xyz)
        if src is None:
            src = _as_xyz(async_agent_xyz)
        if src is None:
            src = _as_xyz(agent_position)
        if src is not None:
            target = src
            final = True
            navigation_corrected = True
            vlm_force_applied = True
            vlm_bidir_object_final = "force_commit"
            return {
                "corrected_target": target,
                "corrected_final": final,
                "navigation_corrected": navigation_corrected,
                "vlm_force_applied": vlm_force_applied,
                "vlm_bidir_object_final": vlm_bidir_object_final,
            }

    # object-commit + 高置信未看到 → 压制 commit
    if baseline_is_final and (not found_v) and conf_v >= suppress_th:
        nf = _nearest_frontier_xyz(frontiers, agent_position)
        if nf is not None:
            target = nf
            final = False
            navigation_corrected = True
            vlm_bidir_object_final = "suppress"

    return {
        "corrected_target": target,
        "corrected_final": final,
        "navigation_corrected": navigation_corrected,
        "vlm_force_applied": vlm_force_applied,
        "vlm_bidir_object_final": vlm_bidir_object_final,
    }


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

    @staticmethod
    def _vlm_log(msg: str) -> None:
        print(msg, flush=True)

    @staticmethod
    def _append_vlm_jsonl(path: str, record: Dict[str, Any]) -> None:
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[VLM] jsonl_append_failed path={path} err={e!r}", flush=True)

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
        pid = os.getpid()
        jsonl_path = os.environ.get("VLM_CORRECTOR_LOG_JSONL", "").strip()
        raw_log_max = int(os.environ.get("VLM_CORRECTOR_LOG_RAW_MAX", "1200"))

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

        desc_preview = description.replace("\n", " ")[:160]
        self._vlm_log(
            f"[VLM] begin pid={pid} dec={decision_num} url={self.cfg.base_url} "
            f"model={self.cfg.model} conf_thresh={self.cfg.confidence_threshold} "
            f"baseline={baseline_target_type} frontiers={num_frontiers} mem={memory_objects} "
            f"tiles={len(image_tile_paths)} desc_preview={desc_preview!r}"
        )

        messages = self.client.build_messages_from_images(data_urls, prompt)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        t0 = time.perf_counter()
        try:
            resp = self.client.chat(messages=messages, max_tokens=256, temperature=0.0)
        except VLLMAPIError as e:
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            body = str(getattr(e, "response_text", "") or "")
            self._vlm_log(
                f"[VLM] http_err pid={pid} dec={decision_num} ms={elapsed_ms:.1f} "
                f"status={getattr(e, 'status_code', None)!r} err={e!r} body_preview={body[:800]!r}"
            )
            rec = {
                "event": "http_err",
                "ts": time.time(),
                "pid": pid,
                "decision_num": decision_num,
                "elapsed_ms": elapsed_ms,
                "status_code": getattr(e, "status_code", None),
                "error": repr(e),
                "response_text_preview": body[:2000],
            }
            self._append_vlm_jsonl(jsonl_path, rec)
            raise
        except Exception as e:
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            self._vlm_log(f"[VLM] chat_err pid={pid} dec={decision_num} ms={elapsed_ms:.1f} err={e!r}")
            self._append_vlm_jsonl(
                jsonl_path,
                {
                    "event": "chat_err",
                    "ts": time.time(),
                    "pid": pid,
                    "decision_num": decision_num,
                    "elapsed_ms": elapsed_ms,
                    "error": repr(e),
                },
            )
            raise

        try:
            raw = self.client.extract_text(resp)
        except Exception as e:
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            keys = list(resp.keys()) if isinstance(resp, dict) else None
            self._vlm_log(
                f"[VLM] extract_err pid={pid} dec={decision_num} ms={elapsed_ms:.1f} err={e!r} resp_keys={keys}"
            )
            self._append_vlm_jsonl(
                jsonl_path,
                {
                    "event": "extract_err",
                    "ts": time.time(),
                    "pid": pid,
                    "decision_num": decision_num,
                    "elapsed_ms": elapsed_ms,
                    "error": repr(e),
                    "response_keys": keys,
                },
            )
            raise

        if not (raw or "").strip():
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            self._vlm_log(f"[VLM] empty_reply pid={pid} dec={decision_num} ms={elapsed_ms:.1f}")
            self._append_vlm_jsonl(
                jsonl_path,
                {
                    "event": "empty_reply",
                    "ts": time.time(),
                    "pid": pid,
                    "decision_num": decision_num,
                    "elapsed_ms": elapsed_ms,
                },
            )

        try:
            parsed = self._parse_json(raw)
        except Exception as e:
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            preview = (raw or "")[:800].replace("\n", "\\n")
            self._vlm_log(
                f"[VLM] parse_err pid={pid} dec={decision_num} ms={elapsed_ms:.1f} err={e!r} raw_preview={preview!r}"
            )
            self._append_vlm_jsonl(
                jsonl_path,
                {
                    "event": "parse_err",
                    "ts": time.time(),
                    "pid": pid,
                    "decision_num": decision_num,
                    "elapsed_ms": elapsed_ms,
                    "error": repr(e),
                    "raw_preview": (raw or "")[:4000],
                },
            )
            raise

        found = bool(parsed.get("found_target", False))
        conf = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "")).strip()

        high_conf_visible = bool(found and conf >= self.cfg.confidence_threshold)
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
        self._vlm_log(
            f"[VLM] reply pid={pid} dec={decision_num} ms={elapsed_ms:.1f} raw_len={len(raw)} "
            f"found={found} conf={conf} high_conf_visible={high_conf_visible} "
            f"reason={reason[:240]!r}"
        )
        if raw_log_max > 0 and raw:
            snippet = raw[:raw_log_max] + ("..." if len(raw) > raw_log_max else "")
            self._vlm_log(f"[VLM] raw_text pid={pid} dec={decision_num} {snippet!r}")

        self._append_vlm_jsonl(
            jsonl_path,
            {
                "event": "ok",
                "ts": time.time(),
                "pid": pid,
                "decision_num": decision_num,
                "elapsed_ms": elapsed_ms,
                "found_target": found,
                "confidence": conf,
                "high_conf_visible": high_conf_visible,
                "reason": reason,
                "raw_text": raw if raw_log_max <= 0 else ((raw[:raw_log_max] + "...") if len(raw) > raw_log_max else raw),
                "parsed": parsed,
            },
        )

        return {
            "vlm_called": True,
            "raw_text": raw,
            "parsed": parsed,
            "found_target": found,
            "confidence": conf,
            "high_conf_visible": high_conf_visible,
            "reason": reason,
            "image_tiles": [str(p) for p in image_tile_paths],
            "elapsed_ms": elapsed_ms,
            "vlm_world_point": None,
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
            out = {"vlm_called": True, "error": str(e)}
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
            out["vlm_info"] = {"vlm_called": True, "error": str(e)}

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

