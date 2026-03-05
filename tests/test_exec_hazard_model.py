from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.execution.fill_models import HazardParams, hazard_fill_prob


def test_hazard_prob_increases_with_intensity() -> None:
    p = HazardParams(a=1.0, b=-0.5, c=0.5, d=-0.2)
    low = hazard_fill_prob(intensity_z=-1.0, spread_z=0.0, imbalance=0.0, params=p)
    high = hazard_fill_prob(intensity_z=2.0, spread_z=0.0, imbalance=0.0, params=p)
    assert high > low
