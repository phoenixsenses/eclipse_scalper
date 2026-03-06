from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.ensemble_experts import ExpertBuildConfig, build_regime_experts, compute_transfer_penalties
from src.microphys.alpha.gating import build_gating_decisions


def _eval_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"signal": "ofi_sig", "split_id": 1, "stability_score": 0.6},
            {"signal": "ofi_sig", "split_id": 2, "stability_score": 0.5},
            {"signal": "vac_sig", "split_id": 1, "stability_score": 0.4},
            {"signal": "vac_sig", "split_id": 2, "stability_score": 0.4},
        ]
    )


def _trades_df() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append({"ts_ms": i, "signal": "ofi_sig", "net_ret": 0.01, "filled": 1})
    for i in range(20, 40):
        rows.append({"ts_ms": i, "signal": "vac_sig", "net_ret": -0.005, "filled": 1})
    return pd.DataFrame(rows)


def _aligned_df() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append({"ts_ms": i, "symbol": "ETHUSDT", "aligned_regime_id": 0 if i < 20 else 1})
    return pd.DataFrame(rows)


def test_expert_builder_deterministic_and_penalty() -> None:
    transfer = pd.DataFrame(
        [
            {"aligned_regime_id": 0, "mean_net_ret": 0.001},
            {"aligned_regime_id": 1, "mean_net_ret": -0.002},
        ]
    )
    a = build_regime_experts(
        eval_df=_eval_df(),
        trades_df=_trades_df(),
        aligned_regimes_df=_aligned_df(),
        symbol="ETHUSDT",
        transfer_by_regime_df=transfer,
        cfg=ExpertBuildConfig(top_k_per_regime=2, min_trades_per_signal=5, min_regime_rows=5),
    )
    b = build_regime_experts(
        eval_df=_eval_df(),
        trades_df=_trades_df(),
        aligned_regimes_df=_aligned_df(),
        symbol="ETHUSDT",
        transfer_by_regime_df=transfer,
        cfg=ExpertBuildConfig(top_k_per_regime=2, min_trades_per_signal=5, min_regime_rows=5),
    )
    assert a.to_dict(orient="records") == b.to_dict(orient="records")
    p = dict(zip(a["aligned_regime_id"].astype(int), a["penalty"].astype(float)))
    assert p[0] >= p[1]


def test_gating_selects_expert_by_regime() -> None:
    frame = pd.DataFrame({"ts_ms": [1, 2, 3], "aligned_regime_id": [0, 1, 9]})
    experts = pd.DataFrame(
        [
            {"aligned_regime_id": 0, "expert_quality": 0.01},
            {"aligned_regime_id": 1, "expert_quality": 0.02},
        ]
    )
    gd = build_gating_decisions(frame, experts, regime_col="aligned_regime_id", data_quality_ok=True)
    assert gd["active_expert_id"].tolist() == [0, 1, -1]
    assert gd["fallback_used"].tolist() == [False, False, True]


def test_transfer_penalty_deterministic() -> None:
    transfer = pd.DataFrame(
        [
            {"aligned_regime_id": 0, "mean_net_ret": 0.001},
            {"aligned_regime_id": 1, "mean_net_ret": -0.0002},
            {"aligned_regime_id": 2, "mean_net_ret": -0.002},
        ]
    )
    _, reg_pen = compute_transfer_penalties(transfer)
    assert reg_pen[0] == 1.0
    assert reg_pen[1] == 0.5
    assert reg_pen[2] == 0.25

