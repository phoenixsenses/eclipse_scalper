from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.generalization import compute_family_generalization


def test_generalization_score_stable_on_synthetic_data() -> None:
    per_symbol = {
        "ETHUSDT": {
            "candidates": pd.DataFrame(
                [
                    {"signal": "ofi_a", "family": "ofi"},
                    {"signal": "ofi_b", "family": "ofi"},
                    {"signal": "compression_a", "family": "compression"},
                ]
            ),
            "selected": pd.DataFrame([{"signal": "ofi_a", "family": "ofi"}]),
            "summary": pd.DataFrame(
                [
                    {"signal": "ofi_a", "family": "ofi", "test_net_mean": 0.10, "regime_concentration": 0.40},
                    {"signal": "ofi_b", "family": "ofi", "test_net_mean": 0.08, "regime_concentration": 0.45},
                    {"signal": "compression_a", "family": "compression", "test_net_mean": 0.03, "regime_concentration": 0.60},
                ]
            ),
        },
        "BTCUSDT": {
            "candidates": pd.DataFrame(
                [
                    {"signal": "ofi_a", "family": "ofi"},
                    {"signal": "ofi_b", "family": "ofi"},
                    {"signal": "compression_a", "family": "compression"},
                ]
            ),
            "selected": pd.DataFrame([{"signal": "ofi_b", "family": "ofi"}]),
            "summary": pd.DataFrame(
                [
                    {"signal": "ofi_a", "family": "ofi", "test_net_mean": 0.09, "regime_concentration": 0.42},
                    {"signal": "ofi_b", "family": "ofi", "test_net_mean": 0.07, "regime_concentration": 0.44},
                    {"signal": "compression_a", "family": "compression", "test_net_mean": 0.01, "regime_concentration": 0.66},
                ]
            ),
        },
    }
    g1 = compute_family_generalization(per_symbol=per_symbol)
    g2 = compute_family_generalization(per_symbol=per_symbol)
    assert g1.to_dict(orient="records") == g2.to_dict(orient="records")
    assert not g1.empty
    assert ((g1["generalization_score"] >= 0.0) & (g1["generalization_score"] <= 1.0)).all()

