from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.eval import apply_signal_entries
from src.microphys.alpha.spec import SignalSpec


def test_cooldown_enforced() -> None:
    df = pd.DataFrame({"F_ofi_z": [2.0] * 10, "regime_id": [0] * 10})
    spec = SignalSpec(
        name="cd",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": 1.0},
        cooldown_bars=3,
    )
    mask = apply_signal_entries(df, spec)
    idx = [i for i, v in enumerate(mask.tolist()) if v]
    assert idx == [0, 3, 6, 9]
