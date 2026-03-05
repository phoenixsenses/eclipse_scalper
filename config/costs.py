from __future__ import annotations

import os


def parse_maker_fee_bps(raw: str | None, fallback: float = 1.0) -> float:
    """Parse maker fee bps from string; supports negative rebate values."""
    if raw is None:
        return float(fallback)
    s = str(raw).strip()
    if not s:
        return float(fallback)
    try:
        return float(s)
    except Exception:
        return float(fallback)


def default_maker_fee_bps_from_env(env: dict[str, str] | None = None) -> float:
    source = env if env is not None else os.environ
    return parse_maker_fee_bps(source.get("MAKER_FEE_BPS"), fallback=1.0)


DEFAULT_MAKER_FEE_BPS: float = default_maker_fee_bps_from_env()

