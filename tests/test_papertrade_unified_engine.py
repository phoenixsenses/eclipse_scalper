from __future__ import annotations

import pandas as pd

from src.microphys.sim.papertrade import PaperTradeConfig, generate_papertrades


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ts_ms": 1000, "ts_utc": "2026-03-05T10:00:00Z", "symbol": "ETHUSDT", "mid": 100.0, "spread": 0.02, "ensemble_side": 1.0, "signal_count": 1},
            {"ts_ms": 2000, "ts_utc": "2026-03-05T10:00:01Z", "symbol": "ETHUSDT", "mid": 100.1, "spread": 0.02, "ensemble_side": 0.0, "signal_count": 0},
            {"ts_ms": 3000, "ts_utc": "2026-03-05T10:00:02Z", "symbol": "ETHUSDT", "mid": 100.2, "spread": 0.02, "ensemble_side": 0.0, "signal_count": 0},
        ]
    )


def test_papertrade_unified_engine_path_emits_order_id() -> None:
    out = generate_papertrades(
        _frame(),
        horizon_bars=1,
        cfg=PaperTradeConfig(mode="taker", fee_bps=0.5, execution_model="simple", use_unified_engine=True),
    )
    assert not out.empty
    assert "order_id" in out.columns
    assert str(out.iloc[0].get("order_id", "")).startswith("paper_simple_")

