"""
Regime-aware position sizer.

Applies a multiplicative scale to the base position amount based on
the current market regime (UP / DOWN / unknown) and trade side (buy / sell).

Scale table (conservative defaults, configurable via ENV):
  UP   + buy  -> 1.25  (momentum favors longs)
  UP   + sell -> 1.10  (mean-rev shorts also benefit but less)
  DOWN + buy  -> 0.80  (reduce longs against trend)
  DOWN + sell -> 1.00  (neutral -- shorts fine in DOWN)
  ""   + *    -> 1.00  (no regime label -> neutral)

ENV overrides
-------------
  REGIME_SIZER_ENABLED=1      (1=on, 0=off; default 1)
  REGIME_SIZER_MAX_SCALE=1.5  (hard cap, default 1.5)
  REGIME_SIZER_UP_BUY=1.25
  REGIME_SIZER_UP_SELL=1.10
  REGIME_SIZER_DOWN_BUY=0.80
  REGIME_SIZER_DOWN_SELL=1.00
"""

from __future__ import annotations

import os
from typing import Dict

# ---------------------------------------------------------------------------
# Default scale table
# ---------------------------------------------------------------------------
_DEFAULT_SCALES: Dict[str, Dict[str, float]] = {
    "UP":   {"buy": 1.25, "sell": 1.10},
    "DOWN": {"buy": 0.80, "sell": 1.00},
    "":     {"buy": 1.00, "sell": 1.00},
}


def _env_float(key: str, default: float) -> float:
    try:
        v = os.getenv(key)
        if v is not None:
            return float(v)
    except Exception:
        pass
    return default


def _build_scale_table() -> Dict[str, Dict[str, float]]:
    return {
        "UP": {
            "buy":  _env_float("REGIME_SIZER_UP_BUY",   _DEFAULT_SCALES["UP"]["buy"]),
            "sell": _env_float("REGIME_SIZER_UP_SELL",  _DEFAULT_SCALES["UP"]["sell"]),
        },
        "DOWN": {
            "buy":  _env_float("REGIME_SIZER_DOWN_BUY",  _DEFAULT_SCALES["DOWN"]["buy"]),
            "sell": _env_float("REGIME_SIZER_DOWN_SELL", _DEFAULT_SCALES["DOWN"]["sell"]),
        },
        "": {"buy": 1.0, "sell": 1.0},
    }


def regime_size_scale(regime: str, side: str) -> float:
    """
    Return the multiplicative scale factor for (regime, side).

    Parameters
    ----------
    regime : str
        "_regime_label" value from the signal dict: "UP", "DOWN", or "".
        Case-insensitive; unknown labels map to neutral (1.0).
    side : str
        "buy" or "sell" (case-insensitive).

    Returns
    -------
    float
        Scale factor in (0, REGIME_SIZER_MAX_SCALE].
    """
    enabled_raw = str(os.getenv("REGIME_SIZER_ENABLED", "1") or "1").strip().lower()
    if enabled_raw in ("0", "false", "no", "off"):
        return 1.0

    max_scale = _env_float("REGIME_SIZER_MAX_SCALE", 1.5)
    table = _build_scale_table()

    r = (regime or "").upper()
    s = (side or "").lower().strip()
    if s not in ("buy", "sell"):
        return 1.0

    row = table.get(r, table[""])
    raw = row.get(s, 1.0)
    return float(min(max(raw, 0.01), max_scale))
