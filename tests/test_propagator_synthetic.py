from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.propagator.kernel import compute_response_kernel


def test_propagator_recovers_positive_short_lag_response() -> None:
    n = 3000
    ofi = np.ones(n, dtype=float)
    # create price increments positively aligned with OFI sign
    step = 1e-4 * ofi
    price = 100.0 * np.exp(np.cumsum(step))

    kernel = compute_response_kernel(pd.Series(price), pd.Series(ofi), max_lag=20)
    assert len(kernel) == 20
    # first-lag response should be positive
    assert float(kernel.loc[kernel["tau"] == 1, "response"].iloc[0]) > 0
    # cumulative over first few lags should stay positive
    assert float(kernel.loc[kernel["tau"] == 5, "cumulative_response"].iloc[0]) > 0
