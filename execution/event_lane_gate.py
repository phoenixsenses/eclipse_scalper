"""
execution/event_lane_gate.py — Event lane gate for the h=60 imb>=0.85 ETHUSDT pocket.

Shadow mode: computes and logs gate decisions but does NOT block orders until
    ENTRY_EVENT_LANE_GATE_ENABLED=1  AND  ENTRY_EVENT_LANE_GATE_SHADOW=0

Lane detection logic copied verbatim from tools/check_event_lanes.py.
Do NOT import from research tools — this module is self-contained.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------

def applies_to_live_event_gate(symbol: str, rule_name: str, horizon_sec: int) -> bool:
    """Returns True only for ETHUSDT + micro_edge_v3_passive_alpha + h=60."""
    return (
        symbol.upper() == "ETHUSDT"
        and rule_name == "micro_edge_v3_passive_alpha"
        and int(horizon_sec) == 60
    )


# ---------------------------------------------------------------------------
# Data loading (self-contained, mirrors check_event_lanes._load_buckets)
# ---------------------------------------------------------------------------

def _load_buckets_for_gate(
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(lookback_min * 60 * 1000)
    con = sqlite3.connect(str(db))
    try:
        trade_rows = con.execute(
            """
            SELECT
                (ts_ms / (?*1000)) * (?*1000) AS bucket_ms,
                SUM(CASE WHEN is_buyer_maker=0 THEN quantity ELSE 0 END) AS buy_qty,
                SUM(CASE WHEN is_buyer_maker=1 THEN quantity ELSE 0 END) AS sell_qty,
                SUM(price*quantity)/SUM(quantity) AS vwap,
                COUNT(*) AS cnt
            FROM agg_trades
            WHERE symbol=? AND ts_ms BETWEEN ? AND ?
            GROUP BY bucket_ms
            ORDER BY bucket_ms
            """,
            (bucket_sec, bucket_sec, symbol, start_ms, now_ms),
        ).fetchall()

        mark_rows = con.execute(
            """
            SELECT (ts_ms / (?*1000)) * (?*1000) AS bucket_ms,
                   AVG(mark_price) AS mark_price
            FROM mark_prices
            WHERE symbol=? AND ts_ms BETWEEN ? AND ?
            GROUP BY bucket_ms
            ORDER BY bucket_ms
            """,
            (bucket_sec, bucket_sec, symbol, start_ms, now_ms),
        ).fetchall()
    finally:
        con.close()

    mark_map: Dict[int, float] = {int(r[0]): float(r[1]) for r in mark_rows}

    buckets: List[Dict[str, Any]] = []
    prev_vwap = None
    for r in trade_rows:
        bms = int(r[0])
        bv = float(r[1] or 0)
        sv = float(r[2] or 0)
        vwap = float(r[3])
        cnt = int(r[4])
        tot = bv + sv
        imb = (bv - sv) / tot if tot > 0 else 0.0
        mark = mark_map.get(bms, vwap)
        spread = abs(vwap - mark) / vwap if vwap > 0 else 0.0
        intensity = cnt * (60.0 / bucket_sec)
        ret_1 = (vwap - prev_vwap) / prev_vwap if prev_vwap and prev_vwap > 0 else 0.0
        buckets.append({
            "ts_ms": bms,
            "imbalance": imb,
            "trade_intensity": intensity,
            "spread": spread,
            "ret_1": ret_1,
            "vwap": vwap,
        })
        prev_vwap = vwap
    return buckets


# ---------------------------------------------------------------------------
# Lane detection — copied verbatim from tools/check_event_lanes.py
# ---------------------------------------------------------------------------

def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * max(0.0, min(1.0, q))
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    return xs[lo] * (1.0 - (pos - lo)) + xs[hi] * (pos - lo)


def _detect_book_proxy_pressure(buckets: List[Dict[str, Any]]) -> List[bool]:
    spreads = [float(r["spread"]) for r in buckets]
    intensities = [float(r["trade_intensity"]) for r in buckets]
    imbalances = [abs(float(r["imbalance"])) for r in buckets]
    rets = [abs(float(r["ret_1"])) for r in buckets]

    spread_q50 = _quantile(spreads, 0.50)
    intensity_q50 = _quantile(intensities, 0.50)
    intensity_q75 = _quantile(intensities, 0.75)
    imbalance_q75 = _quantile(imbalances, 0.75)
    imbalance_q90 = _quantile(imbalances, 0.90)
    ret_q50 = _quantile(rets, 0.50)

    fired = []
    for r in buckets:
        spr = float(r["spread"])
        inten = float(r["trade_intensity"])
        abs_imb = abs(float(r["imbalance"]))
        abs_ret = abs(float(r["ret_1"]))
        high = abs_imb >= imbalance_q90 and inten >= intensity_q75 and spr >= spread_q50
        med = abs_imb >= imbalance_q75 and inten >= intensity_q50 and abs_ret <= ret_q50 and spr >= spread_q50
        fired.append(high or med)
    return fired


def _detect_volatility_burst(buckets: List[Dict[str, Any]]) -> List[bool]:
    abs_rets = [abs(float(r["ret_1"])) for r in buckets]
    intensities = [float(r["trade_intensity"]) for r in buckets]
    spreads = [float(r["spread"]) for r in buckets]

    abs_ret_q75 = _quantile(abs_rets, 0.75)
    abs_ret_q90 = _quantile(abs_rets, 0.90)
    intensity_q40 = _quantile(intensities, 0.40)
    intensity_q60 = _quantile(intensities, 0.60)
    spread_q75 = _quantile(spreads, 0.75)

    fired = []
    for r in buckets:
        abs_move = abs(float(r["ret_1"]))
        inten = float(r["trade_intensity"])
        spr = float(r["spread"])
        high = abs_move >= abs_ret_q90 and inten >= intensity_q60
        med = abs_move >= abs_ret_q75 and inten >= intensity_q40 and spr <= max(spread_q75, 0.0)
        fired.append(high or med)
    return fired


# ---------------------------------------------------------------------------
# Gate payload helpers
# ---------------------------------------------------------------------------

def _inactive_payload(symbol: str, reason: str = "gate_disabled") -> dict:
    return {
        "symbol": symbol,
        "gate": "inactive",
        "allow_trade": True,
        "blocked_lanes": [],
        "reason": reason,
        "latest_ts_ms": None,
        "latest_abs_imbalance": None,
        "lanes": {
            "book_proxy_pressure": {"rule_fired": False, "severity": "none"},
            "volatility_burst": {"rule_fired": False, "severity": "none"},
        },
    }


def _no_data_payload(symbol: str, reason: str = "no_buckets") -> dict:
    return {
        "symbol": symbol,
        "gate": "no_data",
        "allow_trade": True,
        "blocked_lanes": [],
        "reason": reason,
        "latest_ts_ms": None,
        "latest_abs_imbalance": None,
        "lanes": {
            "book_proxy_pressure": {"rule_fired": False, "severity": "none"},
            "volatility_burst": {"rule_fired": False, "severity": "none"},
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_current_event_gate(
    *,
    db: str,
    symbol: str,
    lookback_min: int = 60,
    bucket_sec: int = 5,
    pocket_horizon_sec: int = 60,
    pocket_min_abs_imbalance: float = 0.85,
) -> dict:
    """
    Load recent buckets from microstructure DB and compute current lane state.
    Returns a gate payload dict. Never raises.
    Returns {"gate": "no_data", "allow_trade": True} on any error.

    Gate semantics:
      inactive  — ENTRY_EVENT_LANE_GATE_ENABLED=0 (default)
      no_data   — DB unavailable or empty
      blocked   — last bucket has at least one active lane
      allowed   — last bucket is clean
    """
    gate_enabled = os.getenv("ENTRY_EVENT_LANE_GATE_ENABLED", "0") == "1"
    if not gate_enabled:
        return _inactive_payload(symbol)

    try:
        buckets = _load_buckets_for_gate(db, symbol, lookback_min, bucket_sec)
    except Exception as e:
        return _no_data_payload(symbol, reason=f"error:{e}")

    if not buckets:
        return _no_data_payload(symbol)

    try:
        bpp_fired = _detect_book_proxy_pressure(buckets)
        vb_fired = _detect_volatility_burst(buckets)

        # Current-bucket gate: inspect ONLY the last bucket
        last_bpp = bpp_fired[-1]
        last_vb = vb_fired[-1]
        last_bucket = buckets[-1]

        blocked_lanes = []
        if last_bpp:
            blocked_lanes.append("book_proxy_pressure")
        if last_vb:
            blocked_lanes.append("volatility_burst")

        latest_ts_ms = int(last_bucket["ts_ms"])
        latest_abs_imbalance = abs(float(last_bucket.get("imbalance", 0.0)))

        if blocked_lanes:
            gate = "blocked"
            allow_trade = False
            reason = "lane_blocked"
        else:
            gate = "allowed"
            allow_trade = True
            reason = "no_blocking_lanes"

        return {
            "symbol": symbol,
            "gate": gate,
            "allow_trade": allow_trade,
            "blocked_lanes": blocked_lanes,
            "reason": reason,
            "latest_ts_ms": latest_ts_ms,
            "latest_abs_imbalance": latest_abs_imbalance,
            "lanes": {
                "book_proxy_pressure": {
                    "rule_fired": last_bpp,
                    "severity": "high" if last_bpp else "none",
                },
                "volatility_burst": {
                    "rule_fired": last_vb,
                    "severity": "high" if last_vb else "none",
                },
            },
        }
    except Exception as e:
        return _no_data_payload(symbol, reason=f"error:{e}")


def should_block_event_gate(
    gate_payload: dict,
    *,
    symbol: str,
    rule_name: str,
    horizon_sec: int,
) -> Tuple[bool, str, dict]:
    """
    Given a gate payload, decide if this entry should be blocked.
    Returns (blocked: bool, reason: str, details: dict).

    Only applies to ETHUSDT + micro_edge_v3_passive_alpha + h=60.
    Always returns (False, "gate_not_applicable", {}) for other pockets.
    Never blocks on inactive or no_data gates.
    """
    if not applies_to_live_event_gate(symbol, rule_name, horizon_sec):
        return False, "gate_not_applicable", {}

    gate = gate_payload.get("gate", "no_data")

    if gate in ("inactive", "no_data"):
        return False, gate, {}

    allow_trade = bool(gate_payload.get("allow_trade", True))
    if not allow_trade:
        details = {
            "blocking_lanes": list(gate_payload.get("blocked_lanes", [])),
            "latest_ts_ms": gate_payload.get("latest_ts_ms"),
            "latest_abs_imbalance": gate_payload.get("latest_abs_imbalance"),
            "lane_details": dict(gate_payload.get("lanes", {})),
        }
        return True, "event_lane_gate_blocked", details

    return False, "allowed", {}
