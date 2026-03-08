"""
check_event_lanes.py — Live event lane gate check for h=60 imb>=0.85 pocket.

Queries the microstructure DB, computes book_proxy_pressure and volatility_burst
lane states for the most recent window, and outputs whether the pocket is
currently ALLOWED or BLOCKED.

Self-contained: loads from SQLite directly, no tools/ imports beyond _load_rows.

Usage:
    python -m tools.check_event_lanes --db ../eclipse_scalper/data/microstructure.db
    python -m tools.check_event_lanes --db ../eclipse_scalper/data/microstructure.db --symbol BTCUSDT
    python -m tools.check_event_lanes --db ../eclipse_scalper/data/microstructure.db --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from typing import Any, Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# How many recent buckets must be alert-free for the lane to be considered clear.
# At bucket_sec=5, stale_after_sec=60 => 12 buckets.
_STALE_AFTER_SEC = 60
_LOOKBACK_MIN = 60  # 1h of data for quantile calibration


# ---------------------------------------------------------------------------
# Data loading (self-contained, mirrors liquidation_regime_tagger._load_rows)
# ---------------------------------------------------------------------------

def _load_buckets(db: str, symbol: str, lookback_min: int, bucket_sec: int) -> List[Dict[str, Any]]:
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
        bms, bv, sv, vwap, cnt = int(r[0]), float(r[1] or 0), float(r[2] or 0), float(r[3]), int(r[4])
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
# Lane detection (mirrors book_proxy_pressure_alerts and volatility_burst_alerts)
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
# Lane state: is the lane active RIGHT NOW?
# ---------------------------------------------------------------------------

def _lane_state(
    buckets: List[Dict[str, Any]],
    fired: List[bool],
    stale_after_sec: int,
    bucket_sec: int,
) -> Dict[str, Any]:
    """Return state dict: active, last_alert_ts_ms, age_sec, stale."""
    now_ms = int(time.time() * 1000)
    last_alert_ts_ms: int | None = None
    for b, f in zip(buckets, fired):
        if f:
            ts = int(b["ts_ms"])
            if last_alert_ts_ms is None or ts > last_alert_ts_ms:
                last_alert_ts_ms = ts

    if last_alert_ts_ms is None:
        return {"active": False, "last_alert_ts_ms": None, "age_sec": None, "stale": True}

    age_sec = (now_ms - last_alert_ts_ms) / 1000.0
    stale = age_sec > stale_after_sec
    # Count alerts in the last stale window
    cutoff_ms = now_ms - stale_after_sec * 1000
    recent_count = sum(1 for b, f in zip(buckets, fired) if f and int(b["ts_ms"]) >= cutoff_ms)
    return {
        "active": not stale,
        "last_alert_ts_ms": last_alert_ts_ms,
        "age_sec": round(age_sec, 1),
        "stale": stale,
        "recent_alert_count": recent_count,
    }


# ---------------------------------------------------------------------------
# Main gate check
# ---------------------------------------------------------------------------

def check_gate(
    db: str,
    symbol: str = "ETHUSDT",
    lookback_min: int = _LOOKBACK_MIN,
    bucket_sec: int = 5,
    stale_after_sec: int = _STALE_AFTER_SEC,
) -> Dict[str, Any]:
    buckets = _load_buckets(db, symbol, lookback_min, bucket_sec)
    if not buckets:
        return {
            "symbol": symbol,
            "gate": "UNKNOWN",
            "reason": "no_data",
            "lanes": {},
        }

    bpp_fired = _detect_book_proxy_pressure(buckets)
    vb_fired = _detect_volatility_burst(buckets)

    bpp_state = _lane_state(buckets, bpp_fired, stale_after_sec, bucket_sec)
    vb_state = _lane_state(buckets, vb_fired, stale_after_sec, bucket_sec)

    blocked_lanes = []
    if bpp_state["active"]:
        blocked_lanes.append("book_proxy_pressure")
    if vb_state["active"]:
        blocked_lanes.append("volatility_burst")

    gate = "BLOCKED" if blocked_lanes else "ALLOWED"
    reason = f"active_lanes={','.join(blocked_lanes)}" if blocked_lanes else "no_active_block_lanes"

    return {
        "symbol": symbol,
        "gate": gate,
        "reason": reason,
        "blocked_lanes": blocked_lanes,
        "pocket": "h=60 imb>=0.85 int>=4000 spr<=0.000150",
        "profile": "event_block_eth_micro_imb085_v1",
        "buckets_loaded": len(buckets),
        "lookback_min": lookback_min,
        "stale_after_sec": stale_after_sec,
        "lanes": {
            "book_proxy_pressure": bpp_state,
            "volatility_burst": vb_state,
        },
    }


def _run(args: argparse.Namespace) -> None:
    result = check_gate(
        db=args.db,
        symbol=args.symbol,
        lookback_min=args.lookback_min,
        bucket_sec=args.bucket_sec,
        stale_after_sec=args.stale_after_sec,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    gate = result["gate"]
    gate_display = f"[{gate}]"
    print(f"\n{gate_display} {result['symbol']} pocket={result['pocket']}")
    print(f"  reason        : {result['reason']}")
    print(f"  buckets_loaded: {result['buckets_loaded']} x {args.bucket_sec}s "
          f"(lookback={args.lookback_min}min)")
    print()
    for lane, state in result["lanes"].items():
        active = state.get("active", False)
        age = state.get("age_sec")
        recent = state.get("recent_alert_count", 0)
        age_str = f"{age:.0f}s ago" if age is not None else "never"
        status = "ACTIVE (BLOCKING)" if active else f"clear  (last alert {age_str})"
        print(f"  {lane:<28}: {status}  recent_alerts={recent}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=_LOOKBACK_MIN,
                   help="Lookback window for quantile calibration (default 60)")
    p.add_argument("--bucket-sec", type=int, default=5,
                   help="Bucket size in seconds (default 5)")
    p.add_argument("--stale-after-sec", type=int, default=_STALE_AFTER_SEC,
                   help="Seconds after which a lane alert is considered stale (default 60)")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    args = p.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
