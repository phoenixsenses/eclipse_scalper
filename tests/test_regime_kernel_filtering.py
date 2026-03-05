from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.analysis.regime_metrics import compute_regime_metrics


def test_regime_kernel_filtering_uses_regime_partition() -> None:
    n = 500
    regime = np.where(np.arange(n) < 250, 0, 1)
    ofi = np.ones(n)
    # regime 0 positive drift, regime 1 negative drift
    step = np.where(regime == 0, 1e-4, -1e-4)
    mid = 100.0 * np.exp(np.cumsum(step))

    df = pd.DataFrame(
        {
            "ts_ms": [1000 + i for i in range(n)],
            "regime_id": regime,
            "mid": mid,
            "ofi": ofi,
            "F_ofi_z": ofi,
            "r_1": np.r_[np.diff(np.log(mid)), np.nan],
            "r_5": np.r_[np.log(mid[5:] / mid[:-5]), [np.nan] * 5],
            "volume_proxy": np.ones(n),
            "compression_flag": [False] * n,
            "vacuum_flag": [False] * n,
            "liq_burst_flag": [False] * n,
        }
    )

    metrics, kernels = compute_regime_metrics(df, tau_max=5)
    m0 = metrics[metrics["regime_id"] == 0].iloc[0]
    m1 = metrics[metrics["regime_id"] == 1].iloc[0]

    assert float(m0["kernel_lag1"]) > 0
    assert float(m1["kernel_lag1"]) < 0
    assert set(kernels.keys()) == {0, 1}
