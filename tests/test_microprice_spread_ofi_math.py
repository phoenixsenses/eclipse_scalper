from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_micro_features import compute_micro_bars_from_frames


def test_microprice_spread_ofi_math_exact() -> None:
    trades = pd.DataFrame(
        [
            {"ts": 0.10, "symbol": "ETHUSDT", "price": 100.0, "qty": 2.0, "side": "buy"},
            {"ts": 0.20, "symbol": "ETHUSDT", "price": 101.0, "qty": 1.0, "side": "sell"},
        ]
    )
    book = pd.DataFrame(
        [
            {"ts": 0.15, "symbol": "ETHUSDT", "bid_px": 99.0, "bid_qty": 6.0, "ask_px": 101.0, "ask_qty": 4.0},
        ]
    )
    liq = pd.DataFrame([], columns=["ts", "symbol", "side", "qty", "price"])

    bars = compute_micro_bars_from_frames(
        trades,
        book,
        liq,
        symbol="ETHUSDT",
        start_ts=0.0,
        end_ts=1.0,
        interval_ms=1000,
        rv_window_sec=1.0,
    )
    assert len(bars) >= 1
    row = bars.iloc[0]

    # mid=(99+101)/2=100, spread=(101-99)/100=0.02
    assert abs(float(row["mid"]) - 100.0) < 1e-12
    assert abs(float(row["spread"]) - 0.02) < 1e-12

    # microprice=(ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty)
    expected_micro = (101.0 * 6.0 + 99.0 * 4.0) / (6.0 + 4.0)
    assert abs(float(row["microprice"]) - expected_micro) < 1e-10

    # OFI=buy_qty-sell_qty = 2-1 = 1 ; normalized = 1/(2+1)
    assert abs(float(row["ofi"]) - 1.0) < 1e-12
    assert abs(float(row["ofi_norm"]) - (1.0 / 3.0)) < 1e-12
