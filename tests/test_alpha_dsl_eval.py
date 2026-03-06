from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.dsl import evaluate_expr


def test_alpha_dsl_boolean_and_functions() -> None:
    df = pd.DataFrame(
        {
            "F_ofi_z": [0.5, 1.2, 2.1],
            "compression_flag": [0, 1, 1],
            "spread_z": [0.2, -0.5, -1.2],
        }
    )
    expr = {
        "type": "and",
        "args": [
            {"type": "fn", "fn": "z_gt", "col": "F_ofi_z", "thr": 1.0},
            {"type": "gte", "op": "gte", "left": "compression_flag", "right": 1},
            {"type": "fn", "fn": "pct_lt", "col": "spread_z", "p": 0.6},
        ],
    }
    mask = evaluate_expr(df, expr)
    assert mask.tolist() == [False, True, True]
