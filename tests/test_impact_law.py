from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.impact.models import fit_impact_models


def test_impact_estimator_prefers_sqrt_when_data_is_sqrt() -> None:
    vol = pd.Series(np.linspace(1.0, 100.0, 500))
    # synthetic sqrt law with tiny deterministic perturbation
    abs_ret = 0.001 + 0.01 * np.sqrt(vol) + 1e-6 * (vol % 7)

    fits = fit_impact_models(vol, abs_ret)
    assert fits["sqrt"].r2 > fits["linear"].r2
    assert fits["sqrt"].beta > 0
    assert fits["sqrt"].n == 500
