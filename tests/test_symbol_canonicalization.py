from __future__ import annotations

from pathlib import Path

import pytest

try:
    from utils.symbols import canonical_symbol
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.symbols import canonical_symbol


@pytest.mark.parametrize(
    "raw,exp",
    [
        ("BTCUSDT", "BTCUSDT"),
        ("btcUSDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC-USDT", "BTCUSDT"),
        ("BTC_USDT", "BTCUSDT"),
        ("BTC:USDT", "BTCUSDT"),
    ],
)
def test_canonical_symbol(raw: str, exp: str) -> None:
    assert canonical_symbol(raw) == exp


def test_canonical_symbol_empty_raises() -> None:
    with pytest.raises(ValueError):
        canonical_symbol("")
    with pytest.raises(ValueError):
        canonical_symbol("   ")

