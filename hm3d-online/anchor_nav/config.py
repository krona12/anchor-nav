import os
from typing import Dict

import yaml


def load_anchor_plugin_config(config_path: str) -> Dict[str, bool]:
    if not os.path.exists(config_path):
        return {"use_gap": False, "use_registry": False, "use_ado": False}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        plugins = cfg.get("plugins", {})
        return {
            "use_gap": bool(plugins.get("use_gap", False)),
            "use_registry": bool(plugins.get("use_registry", False)),
            "use_ado": bool(plugins.get("use_ado", False)),
        }
    except Exception:
        return {"use_gap": False, "use_registry": False, "use_ado": False}

