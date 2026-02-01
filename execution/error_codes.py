"""
Standardized error codes for telemetry and safety gates.
"""

ERR_STALE_DATA = "ERR_STALE_DATA"
ERR_DATA_QUALITY = "ERR_DATA_QUALITY"
ERR_ROUTER_BLOCK = "ERR_ROUTER_BLOCK"
ERR_MARGIN = "ERR_MARGIN"
ERR_SLIPPAGE = "ERR_SLIPPAGE"
ERR_SPREAD = "ERR_SPREAD"
ERR_MIN_NOTIONAL = "ERR_MIN_NOTIONAL"
ERR_RISK = "ERR_RISK"
ERR_COOLDOWN = "ERR_COOLDOWN"
ERR_SESSION = "ERR_SESSION"
ERR_UNKNOWN = "ERR_UNKNOWN"
ERR_PARTIAL_FILL = "ERR_PARTIAL_FILL"
EXIT_MOM = "EXIT_MOM"
EXIT_VWAP = "EXIT_VWAP"
EXIT_TIME = "EXIT_TIME"
EXIT_STAGNATION = "EXIT_STAGNATION"


def map_reason(reason: str) -> str:
    r = str(reason or "").lower()
    if not r:
        return ERR_UNKNOWN
    if "stale" in r:
        return ERR_STALE_DATA
    if "quality" in r:
        return ERR_DATA_QUALITY
    if "margin" in r or "insufficient" in r:
        return ERR_MARGIN
    if "slippage" in r:
        return ERR_SLIPPAGE
    if "spread" in r:
        return ERR_SPREAD
    if "min_notional" in r or "min notional" in r:
        return ERR_MIN_NOTIONAL
    if "risk" in r or "heat" in r:
        return ERR_RISK
    if "cooldown" in r:
        return ERR_COOLDOWN
    if "session" in r:
        return ERR_SESSION
    if "partial fill" in r or "partial" in r:
        return ERR_PARTIAL_FILL
    if "router" in r or "blocked" in r:
        return ERR_ROUTER_BLOCK
    return ERR_UNKNOWN
