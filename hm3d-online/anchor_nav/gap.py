from typing import List

from .types import AnchorItem, GoalAnchor
from .vlm_adapter import VLM_MODEL_BEST, call_vlm_text, extract_json


PROMPT_TEMPLATE = """You are a navigation assistant. I provide {n} ordered sub-goal descriptions:
{numbered}

For each sub-goal, return:
- target: final object category in English (single noun phrase)
- anchors: zero or more reference anchors (NOT target itself), each with:
  - type: one of ["spatial_obj", "room", "area_cue"]
  - name: English anchor name

Return only a JSON array with exactly {n} items.
"""


class GoalAnchorParser:
    def parse_all(self, descriptions: List[str], force_vlm: bool = False) -> List[GoalAnchor]:
        if not descriptions:
            return []

        # Simple object-only setting: skip VLM for stability.
        if (not force_vlm) and all(len(desc.split()) <= 2 for desc in descriptions):
            print("[AnchorNav/GAP] fast-path: object-like goals detected, skip VLM parsing")
            return [
                GoalAnchor(target=desc.strip(), anchors=[], raw_desc=desc, sub_idx=idx)
                for idx, desc in enumerate(descriptions)
            ]

        numbered = "\n".join(f"[{i + 1}] {desc}" for i, desc in enumerate(descriptions))
        prompt = PROMPT_TEMPLATE.format(n=len(descriptions), numbered=numbered)
        print(f"[AnchorNav/GAP] calling VLM parser for {len(descriptions)} sub-goals")
        raw = call_vlm_text(prompt, model=VLM_MODEL_BEST)
        if not raw:
            raise RuntimeError("[AnchorNav/GAP] VLM returned empty response")
        parsed = extract_json(raw)
        if not isinstance(parsed, list) or len(parsed) != len(descriptions):
            raise RuntimeError(
                f"[AnchorNav/GAP] invalid response format, expect list len={len(descriptions)}, got={type(parsed)}"
            )

        results: List[GoalAnchor] = []
        for idx, (item, desc) in enumerate(zip(parsed, descriptions)):
            anchor_items: List[AnchorItem] = []
            for anchor in item.get("anchors", []):
                name = str(anchor.get("name", "")).strip()
                if not name:
                    continue
                kind = str(anchor.get("type", "spatial_obj")).strip()
                if kind not in {"spatial_obj", "room", "area_cue"}:
                    kind = "spatial_obj"
                anchor_items.append(AnchorItem(kind=kind, name=name))
            target = str(item.get("target", "")).strip() or desc.split()[0]
            results.append(GoalAnchor(target=target, anchors=anchor_items, raw_desc=desc, sub_idx=idx))
        return results

