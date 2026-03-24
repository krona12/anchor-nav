from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .types import AnchorEntry, GoalAnchor


ANCHOR_SIM_THRESHOLD = 0.25
ROOM_SIM_THRESHOLD = 0.2

ROOM_PRIOR = {
    "bedroom": ["bed", "wardrobe", "nightstand", "pillow", "lamp"],
    "kitchen": ["stove", "fridge", "refrigerator", "sink", "oven", "microwave"],
    "living_room": ["sofa", "couch", "tv", "television", "coffee table"],
    "bathroom": ["toilet", "bathtub", "shower", "sink", "mirror"],
    "study": ["desk", "bookshelf", "computer", "chair"],
}


class AnchorRegistry:
    def __init__(self, clip_text_model, clip_tokenizer):
        self.clip_text_model = clip_text_model
        self.clip_tokenizer = clip_tokenizer
        self.entries: Dict[str, AnchorEntry] = {}
        self._text_cache: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.entries.clear()
        self._text_cache.clear()

    def register_goal(self, goal_anchor: GoalAnchor) -> None:
        for anchor in goal_anchor.anchors:
            if anchor.name not in self.entries:
                self.entries[anchor.name] = AnchorEntry(
                    name=anchor.name,
                    kind=anchor.kind,
                    mtuid=None,
                    position=None,
                    status="missing",
                )

    def get_status(self, anchor_name: str) -> str:
        entry = self.entries.get(anchor_name)
        return entry.status if entry else "missing"

    def get_position(self, anchor_name: str) -> Optional[np.ndarray]:
        entry = self.entries.get(anchor_name)
        if not entry or entry.status != "found":
            return None
        return entry.position

    def _encode_text(self, text: str) -> np.ndarray:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        inputs = self.clip_tokenizer([text], truncation=True, padding=True, return_tensors="pt")
        device = next(self.clip_text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.clip_text_model(**inputs)
            pooled = out.pooler_output
            if pooled is None:
                pooled = out.last_hidden_state[:, 0, :]
            pooled = F.normalize(pooled, dim=-1)
        vec = pooled.squeeze(0).detach().cpu().numpy()
        self._text_cache[text] = vec
        return vec

    @staticmethod
    def _to_tensor(feat) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().float()
        else:
            feat = torch.from_numpy(np.asarray(feat)).float()
        return F.normalize(feat, dim=-1)

    def update(self, representation_manager) -> None:
        open_vocab_feat = getattr(representation_manager, "open_vocab_feat", None)
        object_box = getattr(representation_manager, "object_box", None)
        if open_vocab_feat is None or object_box is None:
            return
        if len(open_vocab_feat) == 0:
            return

        feat_norm = self._to_tensor(open_vocab_feat)
        object_box_np = np.asarray(object_box)

        for anchor_name, entry in self.entries.items():
            if entry.status == "found":
                continue
            if entry.kind == "room":
                priors: Iterable[str] = ROOM_PRIOR.get(anchor_name, [anchor_name])
                scores = []
                for prior in priors:
                    text_vec = torch.from_numpy(self._encode_text(prior)).float()
                    sims = (feat_norm @ text_vec).numpy()
                    scores.append(float(np.max(sims)))
                if scores and float(np.mean(scores)) > ROOM_SIM_THRESHOLD:
                    entry.status = "found"
                    entry.mtuid = None
                    entry.position = object_box_np[:, :3].mean(axis=0)
                continue

            text_vec = torch.from_numpy(self._encode_text(anchor_name)).float()
            sims = (feat_norm @ text_vec).numpy()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim > ANCHOR_SIM_THRESHOLD:
                entry.status = "found"
                entry.mtuid = best_idx
                entry.position = object_box_np[best_idx][:3].copy()

