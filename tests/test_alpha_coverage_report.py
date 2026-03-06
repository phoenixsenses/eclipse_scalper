from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.diagnostics import candidate_coverage
from src.microphys.alpha.spec import SignalSpec


def test_candidate_coverage_counts() -> None:
    df = pd.DataFrame(
        {
            "ts_ms": list(range(20)),
            "F_ofi_z": [2.0 if i % 2 == 0 else 0.0 for i in range(20)],
            "mid": [100.0 + (0.01 * i) for i in range(20)],
            "regime_id": [0] * 20,
        }
    )
    spec = SignalSpec(
        name="x",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        horizon_bars=1,
        cooldown_bars=2,
    )
    out = candidate_coverage(df, [spec], splits=2)
    assert not out.empty
    assert int(out["triggered_events"].sum()) == 10
    assert int(out["after_cooldown"].sum()) <= 10
    assert int(out["effective_trades"].sum()) <= int(out["after_cooldown"].sum())
