from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:
    from execution import bootstrap
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import bootstrap


def test_resolve_active_symbols_prefers_bot_state() -> None:
    bot = SimpleNamespace(
        active_symbols={"btcusdt"},
        cfg=SimpleNamespace(ACTIVE_SYMBOLS=["ETHUSDT"]),
    )
    out = bootstrap._resolve_active_symbols_for_boot(bot)
    assert out == ["BTCUSDT"]


def test_resolve_active_symbols_cfg_fallback() -> None:
    bot = SimpleNamespace(
        active_symbols=None,
        cfg=SimpleNamespace(ACTIVE_SYMBOLS=["BTC/USDT", "ETH-USDT"]),
    )
    out = bootstrap._resolve_active_symbols_for_boot(bot)
    assert out == ["BTCUSDT", "ETHUSDT"]

