from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.runtime_helpers import (
    cfg_env_bool,
    cfg_env_float,
    cfg_value,
    parse_group_kv,
    parse_symbol_kv,
    safe_float,
    symkey,
    truthy,
)


def test_symkey_normalizes_variants() -> None:
    assert symkey("eth/usdt:usdt") == "ETHUSDT"
    assert symkey("ETH:USDT") == "ETHUSDT"
    assert symkey("ETHUSDTUSDT") == "ETHUSDT"
    assert symkey("ETH-USDT") == "ETHUSDT"
    assert symkey("ETH_USDT") == "ETHUSDT"
    assert symkey("  eth usdt  ") == "ETHUSDT"


def test_safe_float_handles_invalid_and_nan() -> None:
    assert safe_float("12.5", 0.0) == 12.5
    assert safe_float("bad", 7.0) == 7.0
    assert safe_float(float("nan"), 3.0) == 3.0


def test_truthy_matches_runtime_expectations() -> None:
    assert truthy(True) is True
    assert truthy(1) is True
    assert truthy("on") is True
    assert truthy("false") is False
    assert truthy(0) is False


def test_parse_group_and_symbol_kv() -> None:
    assert parse_group_kv("MEME=1, MAJOR=2.5") == {"MEME": 1.0, "MAJOR": 2.5}
    assert parse_symbol_kv("eth/usdt=2;BTCUSDT=3") == {"ETHUSDT": 2.0, "BTCUSDT": 3.0}


def test_cfg_helpers_prefer_env(monkeypatch) -> None:
    bot = SimpleNamespace(cfg=SimpleNamespace(TEST_FLOAT=2.0, TEST_BOOL=False))
    monkeypatch.setenv("TEST_FLOAT", "4.5")
    monkeypatch.setenv("TEST_BOOL", "1")
    assert cfg_env_float(bot, "TEST_FLOAT", 1.0) == 4.5
    assert cfg_env_bool(bot, "TEST_BOOL", False) is True


def test_cfg_value_falls_back_cleanly() -> None:
    bot = SimpleNamespace(cfg=SimpleNamespace(ALPHA=9))
    assert cfg_value(bot, "ALPHA", 1) == 9
    assert cfg_value(bot, "BETA", 2) == 2
