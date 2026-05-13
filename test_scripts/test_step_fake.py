from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HM3D_ONLINE = ROOT / "hm3d-online"
if str(HM3D_ONLINE) not in sys.path:
    sys.path.insert(0, str(HM3D_ONLINE))

from anchor_nav.step import StepConfig, decompose_navigation_description, no_proxy_env, sanitize_decomposition


def test_room_only_anchor_query_runs_prestep() -> None:
    out = sanitize_decomposition(
        {
            "target_desc": "wardrobe",
            "object_anchors": [],
            "room_anchors": ["Bedroom"],
            "attribute_anchors": [],
            "anchor_query": "bedroom",
            "source": "fake",
        },
        description="wardrobe in the bedroom",
        task_level="room",
        target_category="wardrobe",
        cfg=StepConfig(use_vlm=False),
    )
    assert out["should_prestep"] is True
    assert out["anchor_query"] == "bedroom"


def test_target_head_terms_removed_from_anchors() -> None:
    out = sanitize_decomposition(
        {
            "target_desc": "laundry basket",
            "object_anchors": ["shoe rack", "white baskets", "bed"],
            "room_anchors": ["walk-in closet"],
            "attribute_anchors": ["tightly packed clothes", "basket corner"],
            "anchor_query": "walk-in closet, white baskets",
            "source": "fake",
        },
        description="laundry basket in the walk-in closet that has tightly packed clothes, shoe rack, white baskets",
        task_level="region",
        target_category="laundry basket",
        cfg=StepConfig(use_vlm=False),
    )
    assert "white baskets" not in out["object_anchors"]
    assert "basket corner" not in out["attribute_anchors"]
    assert "basket" not in out["anchor_query"]
    assert out["anchor_query"] == "walk-in closet, shoe rack, bed, tightly packed clothes"


def test_no_proxy_env_restores_proxy_vars() -> None:
    old_http = os.environ.get("HTTP_PROXY")
    old_no_proxy = os.environ.get("NO_PROXY")
    os.environ["HTTP_PROXY"] = "http://proxy.invalid:8080"
    os.environ["NO_PROXY"] = "localhost"
    try:
        with no_proxy_env(True):
            assert "HTTP_PROXY" not in os.environ
            assert os.environ["NO_PROXY"] == "*"
        assert os.environ["HTTP_PROXY"] == "http://proxy.invalid:8080"
        assert os.environ["NO_PROXY"] == "localhost"
    finally:
        if old_http is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old_http
        if old_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = old_no_proxy


def test_vlm_disabled_does_not_apply_heuristic_prestep() -> None:
    out = decompose_navigation_description(
        description="wardrobe in the bedroom",
        task_level="room",
        target_category="wardrobe",
        cfg=StepConfig(use_vlm=False),
    )
    assert out["ok"] is False
    assert out["parse_ok"] is False
    assert out["error_type"] == "VLMDisabled"
    assert out["fallback_anchor_query"] == "bedroom"
    assert out["anchor_query"] == ""
    assert out["should_prestep"] is False
