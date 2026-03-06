from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.execution.cost_models import CostConfig
from src.microphys.execution.eval import evaluate_conditions


def test_regime_edge_revealed_by_slice() -> None:
    n = 1000
    reg = np.where(np.arange(n) < 500, 0, 1)
    ofi_z = np.where(np.arange(n) % 5 == 0, 2.5, -2.5)
    # edge only in regime 0 for top OFI
    r1 = np.where((reg == 0) & (ofi_z > 0), 0.004, -0.001)

    mid = 100.0 * np.exp(np.cumsum(r1))
    df = pd.DataFrame(
        {
            "ts_ms": np.arange(n),
            "ts_utc": pd.date_range("2024-03-01", periods=n, freq="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "mid": mid,
            "spread": np.full(n, 0.0005),
            "F_ofi_z": ofi_z,
            "rv_z": np.where(reg == 0, 0.5, 1.5),
            "spread_z": np.where(reg == 0, -0.2, 0.5),
            "compression_flag": np.where(reg == 0, ofi_z > 0, False),
            "vacuum_flag": np.zeros(n, dtype=bool),
            "liq_burst_flag": np.zeros(n, dtype=bool),
            "F_intensity_z": np.full(n, 0.0),
        }
    )

    cfg = CostConfig(fee_bps=0.5, latency_bars=1, mode="taker", fill_prob=0.3)
    res_all = evaluate_conditions(df, horizon=3, cfg=cfg)
    res_r0 = evaluate_conditions(df[reg == 0].reset_index(drop=True), horizon=3, cfg=cfg)
    res_r1 = evaluate_conditions(df[reg == 1].reset_index(drop=True), horizon=3, cfg=cfg)

    c = "ofi_top_decile_buy"
    m_all = float(res_all[res_all["condition"] == c]["net_mean"].iloc[0])
    m_r0 = float(res_r0[res_r0["condition"] == c]["net_mean"].iloc[0])
    m_r1 = float(res_r1[res_r1["condition"] == c]["net_mean"].iloc[0])

    assert m_r0 > m_r1
    # global value should sit between regime slices
    assert min(m_r0, m_r1) <= m_all <= max(m_r0, m_r1)
