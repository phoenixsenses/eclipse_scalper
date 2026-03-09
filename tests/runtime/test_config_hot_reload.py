"""Tests for config hot-reload from JSON override file."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from unittest.mock import patch

from config.hot_reload import (
    HOT_RELOAD_ALLOWED,
    HOT_RELOAD_BLOCKED,
    check_and_apply,
    _coerce,
)


@dataclass
class MockConfig:
    MAX_RISK_PER_TRADE: float = 0.10
    MAX_PORTFOLIO_HEAT: float = 0.45
    MIN_CONFIDENCE: float = 0.72
    NOTIFY_ON_ENTRY: bool = True
    MAX_ORDER_RETRIES: int = 2
    ACTIVE_SYMBOLS: list = None
    LEVERAGE: int = 20

    def __post_init__(self):
        if self.ACTIVE_SYMBOLS is None:
            self.ACTIVE_SYMBOLS = ["BTCUSDT"]


class TestCoerce:
    def test_float_coerce(self):
        assert _coerce(0.15, "float", 0.10) == 0.15
        assert _coerce("0.15", "float", 0.10) == 0.15

    def test_int_coerce(self):
        assert _coerce(5, "int", 2) == 5
        assert _coerce("5", "int", 2) == 5

    def test_bool_coerce(self):
        assert _coerce(True, "bool", False) is True
        assert _coerce("true", "bool", False) is True
        assert _coerce("0", "bool", True) is False
        assert _coerce(False, "bool", True) is False

    def test_str_coerce(self):
        assert _coerce("hello", "str", "world") == "hello"


class TestCheckAndApply:
    def test_applies_allowed_fields(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps({
            "MAX_RISK_PER_TRADE": 0.15,
            "MIN_CONFIDENCE": 0.65,
        }), encoding="utf-8")

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert cfg.MAX_RISK_PER_TRADE == 0.15
        assert cfg.MIN_CONFIDENCE == 0.65
        assert "MAX_RISK_PER_TRADE" in changes
        assert changes["MAX_RISK_PER_TRADE"]["old"] == 0.10
        assert changes["MAX_RISK_PER_TRADE"]["new"] == 0.15

    def test_blocks_forbidden_fields(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps({
            "LEVERAGE": 50,
            "ACTIVE_SYMBOLS": ["ETHUSDT"],
        }), encoding="utf-8")

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert cfg.LEVERAGE == 20  # unchanged
        assert cfg.ACTIVE_SYMBOLS == ["BTCUSDT"]  # unchanged
        assert len(changes) == 0

    def test_skips_unchanged_values(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps({
            "MAX_RISK_PER_TRADE": 0.10,  # same as default
        }), encoding="utf-8")

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert len(changes) == 0

    def test_no_file_no_error(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "nonexistent.json"

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert len(changes) == 0

    def test_invalid_json_no_crash(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text("{broken json", encoding="utf-8")

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert len(changes) == 0

    def test_mtime_cache_prevents_reread(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps({
            "MAX_RISK_PER_TRADE": 0.15,
        }), encoding="utf-8")

        cfg = MockConfig()
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes1 = check_and_apply(cfg)
            assert len(changes1) == 1
            # Second call with same mtime should return no changes
            hr._last_check_ts = 0.0  # bypass interval check
            changes2 = check_and_apply(cfg)
            assert len(changes2) == 0

    def test_bool_field_reload(self, tmp_path):
        import config.hot_reload as hr
        hr._last_mtime = 0.0
        hr._last_check_ts = 0.0

        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps({
            "NOTIFY_ON_ENTRY": False,
        }), encoding="utf-8")

        cfg = MockConfig()
        assert cfg.NOTIFY_ON_ENTRY is True
        with patch.object(hr, "_OVERRIDE_PATH", override_file):
            changes = check_and_apply(cfg)

        assert cfg.NOTIFY_ON_ENTRY is False
        assert len(changes) == 1


class TestAllowedBlockedDisjoint:
    def test_no_overlap(self):
        overlap = HOT_RELOAD_ALLOWED & HOT_RELOAD_BLOCKED
        assert len(overlap) == 0, f"Fields in both ALLOWED and BLOCKED: {overlap}"
