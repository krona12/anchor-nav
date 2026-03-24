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
    def parse_all(self, descriptions: List[str]) -> List[GoalAnchor]:
        if not descriptions:
            return []

        # Simple object-only setting: skip VLM for stability.
        if all(len(desc.split()) <= 2 for desc in descriptions):
            return [
                GoalAnchor(target=desc.strip(), anchors=[], raw_desc=desc, sub_idx=idx)
                for idx, desc in enumerate(descriptions)
            ]

        numbered = "\n".join(f"[{i + 1}] {desc}" for i, desc in enumerate(descriptions))
        prompt = PROMPT_TEMPLATE.format(n=len(descriptions), numbered=numbered)
        parsed = None
        try:
            raw = call_vlm_text(prompt, model=VLM_MODEL_BEST)
            if raw:
                parsed = extract_json(raw)
        except Exception as err:
            print(f"[AnchorNav/GAP] parse failed, fallback to no-anchor mode: {err}")

        if not isinstance(parsed, list) or len(parsed) != len(descriptions):
            parsed = [{"target": desc.split()[0], "anchors": []} for desc in descriptions]

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

