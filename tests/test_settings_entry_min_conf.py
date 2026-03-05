from __future__ import annotations

import os
import importlib


def test_entry_min_conf_env_overrides_default() -> None:
    old = os.environ.get("ENTRY_MIN_CONFIDENCE")
    try:
        os.environ["ENTRY_MIN_CONFIDENCE"] = "0.00"
        mod = importlib.import_module("config.settings")
        mod = importlib.reload(mod)
        cfg = mod.Config()
        assert float(cfg.ENTRY_MIN_CONFIDENCE) == 0.0
        assert float(cfg.MIN_CONFIDENCE) == 0.0
    finally:
        if old is None:
            os.environ.pop("ENTRY_MIN_CONFIDENCE", None)
        else:
            os.environ["ENTRY_MIN_CONFIDENCE"] = old
