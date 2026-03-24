from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AnchorItem:
    kind: str
    name: str


@dataclass
class GoalAnchor:
    target: str
    anchors: List[AnchorItem]
    raw_desc: str
    sub_idx: int

    def missing_anchors(self, registry: "AnchorRegistry") -> List[AnchorItem]:
        return [a for a in self.anchors if registry.get_status(a.name) != "found"]

    def found_anchors(self, registry: "AnchorRegistry") -> List[AnchorItem]:
        return [a for a in self.anchors if registry.get_status(a.name) == "found"]


@dataclass
class AnchorEntry:
    name: str
    kind: str
    mtuid: Optional[int]
    position: Optional[np.ndarray]
    status: str

