from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


FULL_COND_TOUCH_PROXY = 0.5
DEFAULT_MIN_INTENSITY = 4000.0
DEFAULT_MAX_SPREAD = 0.000150
DEFAULT_MIN_ABS_IMBALANCE = 0.85
DEFAULT_HORIZON_SEC = 120


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _quantile(values: Sequence[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _min_max_id(conn: sqlite3.Connection, table: str) -> tuple[int, int]:
    row = conn.execute(f"SELECT MIN(id), MAX(id) FROM {table}").fetchone()
    if not row or row[0] is None or row[1] is None:
        return (0, 0)
    return (int(row[0]), int(row[1]))


def _ts_at_id(conn: sqlite3.Connection, table: str, row_id: int) -> Optional[int]:
    row = conn.execute(f"SELECT ts_ms FROM {table} WHERE id = ?", (int(row_id),)).fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _find_start_id(conn: sqlite3.Connection, table: str, start_ts_ms: int) -> int:
    lo, hi = _min_max_id(conn, table)
    if hi <= 0:
        return 0
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        ts_ms = _ts_at_id(conn, table, mid)
        if ts_ms is None:
            lo = mid + 1
            continue
        if ts_ms >= int(start_ts_ms):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)


def _tag_book_proxy_pressure(rows: Sequence[Dict[str, Any]]) -> List[bool]:
    spreads = [_safe_float(r.get("spread")) for r in rows if r.get("spread") is not None]
    intensities = [_safe_float(r.get("trade_intensity")) for r in rows if r.get("trade_intensity") is not None]
    imbalances = [abs(_safe_float(r.get("imbalance"))) for r in rows if r.get("imbalance") is not None]
    abs_rets = [abs(_safe_float(r.get("ret_1"))) for r in rows if r.get("ret_1") is not None]
    spread_q50 = _quantile(spreads, 0.50)
    intensity_q75 = _quantile(intensities, 0.75)
    intensity_q50 = _quantile(intensities, 0.50)
    imbalance_q90 = _quantile(imbalances, 0.90)
    imbalance_q75 = _quantile(imbalances, 0.75)
    ret_q50 = _quantile(abs_rets, 0.50)
    tagged: List[bool] = []
    for row in rows:
        spread = _safe_float(row.get("spread"))
        intensity = _safe_float(row.get("trade_intensity"))
        abs_imbalance = abs(_safe_float(row.get("imbalance")))
        abs_ret = abs(_safe_float(row.get("ret_1")))
        high = abs_imbalance >= imbalance_q90 and intensity >= intensity_q75 and spread >= spread_q50
        medium = (
            abs_imbalance >= imbalance_q75
            and intensity >= intensity_q50
            and abs_ret <= ret_q50
            and spread >= spread_q50
        )
        tagged.append(bool(high or medium))
    return tagged


def _tag_volatility_burst(rows: Sequence[Dict[str, Any]]) -> List[bool]:
    abs_rets = [abs(_safe_float(r.get("ret_1"))) for r in rows if r.get("ret_1") is not None]
    intensities = [_safe_float(r.get("trade_intensity")) for r in rows if r.get("trade_intensity") is not None]
    spreads = [_safe_float(r.get("spread")) for r in rows if r.get("spread") is not None]
    ret_q90 = _quantile(abs_rets, 0.90)
    ret_q75 = _quantile(abs_rets, 0.75)
    int_q60 = _quantile(intensities, 0.60)
    int_q40 = _quantile(intensities, 0.40)
    spread_q75 = _quantile(spreads, 0.75)
    tagged: List[bool] = []
    for row in rows:
        abs_ret = abs(_safe_float(row.get("ret_1")))
        intensity = _safe_float(row.get("trade_intensity"))
        spread = _safe_float(row.get("spread"))
        high = abs_ret >= ret_q90 and intensity >= int_q60
        medium = abs_ret >= ret_q75 and intensity >= int_q40 and spread <= spread_q75
        tagged.append(bool(high or medium))
    return tagged


def analyze_features(
    features: Sequence[Dict[str, Any]],
    *,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    min_fills: int,
    min_abs_imbalance: float = DEFAULT_MIN_ABS_IMBALANCE,
    min_trade_intensity: float = DEFAULT_MIN_INTENSITY,
    max_spread: float = DEFAULT_MAX_SPREAD,
    horizon_sec: int = DEFAULT_HORIZON_SEC,
) -> Dict[str, Any]:
    horizon_steps = max(1, int(round(float(horizon_sec) / max(1, bucket_sec))))
    rows = [dict(row) for row in features if row.get("mid") is not None]
    bp_flags = _tag_book_proxy_pressure(rows)
    vb_flags = _tag_volatility_burst(rows)

    signal_indices: List[int] = []
    filtered_indices: List[int] = []
    touch_count = 0
    blocked_book = 0
    blocked_vol = 0
    blocked_both = 0

    for i, row in enumerate(rows):
        if i + horizon_steps >= len(rows):
            break
        imbalance = _safe_float(row.get("imbalance"))
        intensity = _safe_float(row.get("trade_intensity"))
        spread = _safe_float(row.get("spread"))
        mid = _safe_float(row.get("mid"))
        if mid <= 0.0:
            continue
        if abs(imbalance) < float(min_abs_imbalance):
            continue
        if intensity < float(min_trade_intensity):
            continue
        if spread <= 0.0 or spread > float(max_spread):
            continue
        signal_indices.append(i)
        blocked_now = bool(bp_flags[i] or vb_flags[i])
        if bp_flags[i] and vb_flags[i]:
            blocked_both += 1
        elif bp_flags[i]:
            blocked_book += 1
        elif vb_flags[i]:
            blocked_vol += 1
        if blocked_now:
            continue
        filtered_indices.append(i)
        future_mids = [_safe_float(rows[j].get("mid")) for j in range(i + 1, i + horizon_steps + 1)]
        if imbalance > 0:
            limit_px = mid * (1.0 - 0.5 * spread)
            touched = any(px <= limit_px for px in future_mids if px > 0.0)
        else:
            limit_px = mid * (1.0 + 0.5 * spread)
            touched = any(px >= limit_px for px in future_mids if px > 0.0)
        if touched:
            touch_count += 1

    total_signals = len(signal_indices)
    filtered_n = len(filtered_indices)
    touch_rate = (float(touch_count) / float(filtered_n)) if filtered_n else 0.0
    fill_rate = touch_rate * float(FULL_COND_TOUCH_PROXY)
    estimated_fills = float(filtered_n) * fill_rate
    kept_ratio = (float(filtered_n) / float(total_signals)) if total_signals else 0.0
    status = "READY_TO_RANK" if estimated_fills >= float(min_fills) else "INSUFFICIENT"
    fills_needed = max(0.0, float(min_fills) - estimated_fills)

    return {
        "symbol": str(symbol).upper(),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "horizon_sec": int(horizon_sec),
        "signal_definition": {
            "abs_imbalance_gte": float(min_abs_imbalance),
            "trade_intensity_gte": float(min_trade_intensity),
            "spread_lte": float(max_spread),
        },
        "signals_total": int(total_signals),
        "signals_filtered": int(filtered_n),
        "kept_ratio": kept_ratio,
        "touch_rate": touch_rate,
        "fill_rate": fill_rate,
        "full_cond_touch_proxy": float(FULL_COND_TOUCH_PROXY),
        "estimated_fills": estimated_fills,
        "min_fills_needed": int(min_fills),
        "additional_fills_needed": int(math.ceil(fills_needed)),
        "status": status,
        "blocked_counts": {
            "book_proxy_pressure_only": int(blocked_book),
            "volatility_burst_only": int(blocked_vol),
            "both": int(blocked_both),
        },
    }


def _load_features(db: str, symbol: str, lookback_min: int, bucket_sec: int) -> List[Dict[str, Any]]:
    bucket_ms = max(1, int(bucket_sec)) * 1000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
    conn = sqlite3.connect(str(db))
    try:
        trade_start_id = _find_start_id(conn, "agg_trades", start_ms)
        mark_start_id = _find_start_id(conn, "mark_prices", start_ms)
        rows = conn.execute(
            """
            WITH mark_buckets AS (
                SELECT
                    CAST((ts_ms / ?) AS INTEGER) * ? AS bucket_ms,
                    AVG(mark_price) AS mid
                FROM mark_prices
                WHERE id >= ? AND symbol = ? AND ts_ms >= ? AND ts_ms <= ?
                GROUP BY 1
            ),
            trade_buckets AS (
                SELECT
                    CAST((t.ts_ms / ?) AS INTEGER) * ? AS bucket_ms,
                    COUNT(*) AS trade_count,
                    SUM(ABS(t.quantity)) AS sum_abs_qty,
                    SUM(CASE WHEN t.is_buyer_maker = 1 THEN -t.quantity ELSE t.quantity END) AS signed_qty,
                    AVG(CASE WHEN mb.mid > 0 THEN ABS(t.price - mb.mid) / mb.mid ELSE NULL END) AS spread
                FROM agg_trades t
                LEFT JOIN mark_buckets mb
                    ON mb.bucket_ms = CAST((t.ts_ms / ?) AS INTEGER) * ?
                WHERE t.id >= ? AND t.symbol = ? AND t.ts_ms >= ? AND t.ts_ms <= ?
                GROUP BY 1
            ),
            bucket_union AS (
                SELECT bucket_ms FROM mark_buckets
                UNION
                SELECT bucket_ms FROM trade_buckets
            )
            SELECT
                u.bucket_ms,
                mb.mid,
                tb.trade_count,
                tb.sum_abs_qty,
                tb.signed_qty,
                tb.spread
            FROM bucket_union u
            LEFT JOIN mark_buckets mb ON mb.bucket_ms = u.bucket_ms
            LEFT JOIN trade_buckets tb ON tb.bucket_ms = u.bucket_ms
            ORDER BY u.bucket_ms ASC
            """,
            (
                bucket_ms,
                bucket_ms,
                mark_start_id,
                str(symbol).upper(),
                start_ms,
                now_ms,
                bucket_ms,
                bucket_ms,
                bucket_ms,
                bucket_ms,
                trade_start_id,
                str(symbol).upper(),
                start_ms,
                now_ms,
            ),
        ).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    prev_mid: Optional[float] = None
    for row in rows:
        ts_ms = int(row[0])
        mid = None if row[1] is None else float(row[1])
        trade_count = 0.0 if row[2] is None else float(row[2])
        sum_abs_qty = 0.0 if row[3] is None else float(row[3])
        signed_qty = 0.0 if row[4] is None else float(row[4])
        spread = None if row[5] is None else float(row[5])
        imbalance = None
        if sum_abs_qty > 0.0:
            imbalance = signed_qty / sum_abs_qty
        trade_intensity = trade_count * (60.0 / max(1.0, float(bucket_sec)))
        ret_1 = None
        if mid is not None and prev_mid is not None and mid > 0.0 and prev_mid > 0.0:
            ret_1 = math.log(mid / prev_mid)
        if mid is not None and mid > 0.0:
            prev_mid = mid
        out.append(
            {
                "ts_ms": float(ts_ms),
                "mid": mid,
                "spread": spread,
                "imbalance": imbalance,
                "trade_intensity": trade_intensity,
                "ret_1": ret_1,
            }
        )
    return out


def _format_summary(payload: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "h=120 imb>=0.85 Fill Density Monitor",
            f"  symbol           : {payload['symbol']}",
            f"  lookback_min     : {payload['lookback_min']}",
            f"  signals_total    : {payload['signals_total']}",
            f"  signals_filtered : {payload['signals_filtered']}  (kept {payload['kept_ratio']:.1%} after event block)",
            f"  touch_rate       : {payload['touch_rate']:.1%}",
            f"  fill_rate        : {payload['fill_rate']:.1%}",
            f"  estimated_fills  : {payload['estimated_fills']:.1f}",
            f"  min_fills_needed : {payload['min_fills_needed']}",
            f"  status           : {payload['status']}",
            f"  more_fills_needed: {payload['additional_fills_needed']}",
        ]
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor h=120 imb>=0.85 fill density after event blocking.")
    p.add_argument("--db", required=True)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=20160)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--min-fills", type=int, default=30)
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    features = _load_features(str(args.db), str(args.symbol), int(args.lookback_min), int(args.bucket_sec))
    payload = analyze_features(
        features,
        symbol=str(args.symbol),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        min_fills=int(args.min_fills),
    )
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(_format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
