from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.overlap import dedupe_specs, pairwise_overlap
from src.microphys.alpha.spec import SignalSpec


def _frame() -> pd.DataFrame:
    n = 200
    return pd.DataFrame(
        {
            "ts_ms": list(range(n)),
            "ts_utc": [f"2024-03-01T00:00:{(i % 60):02d}Z" for i in range(n)],
            "x": [i / n for i in range(n)],
            "regime_id": [0] * n,
        }
    )


def test_overlap_and_dedup_identical_candidates() -> None:
    df = _frame()
    s1 = SignalSpec(
        name="a",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "x", "right": 0.5},
        cooldown_bars=0,
    )
    s2 = SignalSpec(
        name="b",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "x", "right": 0.5},
        cooldown_bars=0,
    )
    s3 = SignalSpec(
        name="c",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "x", "right": 0.9},
        cooldown_bars=0,
    )
    pairs = pairwise_overlap(df, [s1, s2, s3])
    ab = pairs[(pairs["a"] == "a") & (pairs["b"] == "b")]
    assert not ab.empty
    assert float(ab.iloc[0]["jaccard"]) >= 0.99

    res = dedupe_specs([s1, s2, s3], pairs, jaccard_thr=0.90, target_triggers_per_day=10.0)
    names = sorted(s.name for s in res.selected)
    assert len(names) == 2
    assert "c" in names
