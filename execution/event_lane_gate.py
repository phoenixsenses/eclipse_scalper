"""
Live event lane gate for the ETHUSDT h=60 / min_imbalance>=0.85 micro pocket.

This module is intentionally self-contained. It reads the live microstructure DB,
derives current-bucket lane state, and exposes a narrow helper for the live entry
loop to decide whether the current pocket would be blocked.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


def applies_to_live_event_gate(
    symbol: str,
    rule_name: str,
    horizon_sec: int,
    signal: Optional[Dict[str, Any]] = None,
) -> bool:
    """Restrict the gate to the live ETH micro pocket that research validated."""
    signal = signal or {}
    source = str(signal.get("source") or "")
    try:
        min_imbalance = abs(float(signal.get("min_imbalance", 0.0) or 0.0))
    except Exception:
        min_imbalance = 0.0
    return (
        symbol.upper() == "ETHUSDT"
        and rule_name == "micro_edge_v3_passive_alpha"
        and int(horizon_sec) == 60
        and source == "micro_signal"
        and min_imbalance >= 0.85
    )


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
            SELECT
                (ts_ms / (?*1000)) * (?*1000) AS bucket_ms,
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

    mark_map: Dict[int, float] = {int(row[0]): float(row[1]) for row in mark_rows}
    buckets: List[Dict[str, Any]] = []
    prev_vwap = None
    for row in trade_rows:
        bucket_ms = int(row[0])
        buy_qty = float(row[1] or 0.0)
        sell_qty = float(row[2] or 0.0)
        vwap = float(row[3] or 0.0)
        count = int(row[4] or 0)
        total_qty = buy_qty + sell_qty
        imbalance = (buy_qty - sell_qty) / total_qty if total_qty > 0 else 0.0
        mark_price = mark_map.get(bucket_ms, vwap)
        spread = abs(vwap - mark_price) / vwap if vwap > 0 else 0.0
        trade_intensity = count * (60.0 / max(1, int(bucket_sec)))
        ret_1 = (vwap - prev_vwap) / prev_vwap if prev_vwap and prev_vwap > 0 else 0.0
        buckets.append(
            {
                "ts_ms": bucket_ms,
                "imbalance": imbalance,
                "trade_intensity": trade_intensity,
                "spread": spread,
                "ret_1": ret_1,
                "vwap": vwap,
            }
        )
        prev_vwap = vwap
    return buckets


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * max(0.0, min(1.0, q))
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _detect_book_proxy_pressure(buckets: List[Dict[str, Any]]) -> List[bool]:
    spreads = [float(row["spread"]) for row in buckets]
    intensities = [float(row["trade_intensity"]) for row in buckets]
    imbalances = [abs(float(row["imbalance"])) for row in buckets]
    rets = [abs(float(row["ret_1"])) for row in buckets]

    spread_q50 = _quantile(spreads, 0.50)
    intensity_q50 = _quantile(intensities, 0.50)
    intensity_q75 = _quantile(intensities, 0.75)
    imbalance_q75 = _quantile(imbalances, 0.75)
    imbalance_q90 = _quantile(imbalances, 0.90)
    ret_q50 = _quantile(rets, 0.50)

    fired: List[bool] = []
    for row in buckets:
        spread = float(row["spread"])
        intensity = float(row["trade_intensity"])
        abs_imbalance = abs(float(row["imbalance"]))
        abs_ret = abs(float(row["ret_1"]))
        high = abs_imbalance >= imbalance_q90 and intensity >= intensity_q75 and spread >= spread_q50
        medium = (
            abs_imbalance >= imbalance_q75
            and intensity >= intensity_q50
            and abs_ret <= ret_q50
            and spread >= spread_q50
        )
        fired.append(high or medium)
    return fired


def _detect_volatility_burst(buckets: List[Dict[str, Any]]) -> List[bool]:
    abs_rets = [abs(float(row["ret_1"])) for row in buckets]
    intensities = [float(row["trade_intensity"]) for row in buckets]
    spreads = [float(row["spread"]) for row in buckets]

    abs_ret_q75 = _quantile(abs_rets, 0.75)
    abs_ret_q90 = _quantile(abs_rets, 0.90)
    intensity_q40 = _quantile(intensities, 0.40)
    intensity_q60 = _quantile(intensities, 0.60)
    spread_q75 = _quantile(spreads, 0.75)

    fired: List[bool] = []
    for row in buckets:
        abs_move = abs(float(row["ret_1"]))
        intensity = float(row["trade_intensity"])
        spread = float(row["spread"])
        high = abs_move >= abs_ret_q90 and intensity >= intensity_q60
        medium = abs_move >= abs_ret_q75 and intensity >= intensity_q40 and spread <= max(spread_q75, 0.0)
        fired.append(high or medium)
    return fired


def _base_payload(symbol: str, gate: str, reason: str) -> Dict[str, Any]:
    return {
        "symbol": str(symbol).upper(),
        "gate": gate,
        "pocket_active": False,
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


def load_current_event_gate(
    *,
    db: str,
    symbol: str,
    lookback_min: int = 60,
    bucket_sec: int = 5,
    pocket_horizon_sec: int = 60,
    pocket_min_abs_imbalance: float = 0.85,
) -> Dict[str, Any]:
    """
    Derive gate state from the latest bucket only.

    Gate semantics:
      inactive       - gate disabled
      no_data        - DB missing/unreadable or empty
      inactive_pocket- latest bucket does not satisfy abs(imbalance)>=threshold
      blocked        - latest bucket is active and at least one blocking lane fired
      allowed        - latest bucket is active and clean
    """
    del pocket_horizon_sec  # Horizon is currently enforced by the caller scope.

    if os.getenv("ENTRY_EVENT_LANE_GATE_ENABLED", "0") != "1":
        return _base_payload(symbol, "inactive", "gate_disabled")

    try:
        buckets = _load_buckets_for_gate(db, symbol, lookback_min, bucket_sec)
    except Exception as exc:
        return _base_payload(symbol, "no_data", f"error:{exc}")

    if not buckets:
        return _base_payload(symbol, "no_data", "no_buckets")

    try:
        bpp_fired = _detect_book_proxy_pressure(buckets)
        vb_fired = _detect_volatility_burst(buckets)
        latest_bucket = buckets[-1]
        latest_abs_imbalance = abs(float(latest_bucket.get("imbalance", 0.0) or 0.0))
        pocket_active = latest_abs_imbalance >= float(pocket_min_abs_imbalance)
        blocked_lanes: List[str] = []
        if bpp_fired[-1]:
            blocked_lanes.append("book_proxy_pressure")
        if vb_fired[-1]:
            blocked_lanes.append("volatility_burst")

        if not pocket_active:
            gate = "inactive_pocket"
            allow_trade = True
            reason = "imbalance_below_threshold"
        elif blocked_lanes:
            gate = "blocked"
            allow_trade = False
            reason = "lane_blocked"
        else:
            gate = "allowed"
            allow_trade = True
            reason = "no_blocking_lanes"

        return {
            "symbol": str(symbol).upper(),
            "gate": gate,
            "pocket_active": pocket_active,
            "allow_trade": allow_trade,
            "blocked_lanes": blocked_lanes,
            "reason": reason,
            "latest_ts_ms": int(latest_bucket.get("ts_ms", 0) or 0),
            "latest_abs_imbalance": latest_abs_imbalance,
            "lanes": {
                "book_proxy_pressure": {
                    "rule_fired": bool(bpp_fired[-1]),
                    "severity": "high" if bpp_fired[-1] else "none",
                },
                "volatility_burst": {
                    "rule_fired": bool(vb_fired[-1]),
                    "severity": "high" if vb_fired[-1] else "none",
                },
            },
        }
    except Exception as exc:
        return _base_payload(symbol, "no_data", f"error:{exc}")


def should_block_event_gate(
    gate_payload: Dict[str, Any],
    *,
    symbol: str,
    rule_name: str,
    horizon_sec: int,
    signal: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Decide whether this live entry should be blocked.

    The gate is narrow by design. Anything outside the current ETH micro pocket
    returns `gate_not_applicable`.
    """
    if not applies_to_live_event_gate(symbol, rule_name, horizon_sec, signal=signal):
        return False, "gate_not_applicable", {}

    gate = str(gate_payload.get("gate") or "no_data")
    if gate in ("inactive", "no_data", "inactive_pocket"):
        return False, gate, {}

    if not bool(gate_payload.get("allow_trade", True)):
        details = {
            "blocking_lanes": list(gate_payload.get("blocked_lanes", [])),
            "latest_ts_ms": gate_payload.get("latest_ts_ms"),
            "latest_abs_imbalance": gate_payload.get("latest_abs_imbalance"),
            "lane_details": dict(gate_payload.get("lanes", {})),
        }
        return True, "event_lane_gate_blocked", details

    return False, "allowed", {}
