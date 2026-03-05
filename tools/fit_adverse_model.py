from __future__ import annotations

"""Calibrate adverse selection model from microstructure DB and forward outcomes.

Estimates conditional expected adverse_bps for signal-firing buckets by
feature quartiles (spread, trade_intensity, imbalance).

The output JSON can be loaded by execution/passive_execution_simulator.py
(see load_adverse_model / get_conditional_adverse_bps).

Usage:
  python -m tools.fit_adverse_model \\
    --db data/microstructure.db \\
    --symbol ETHUSDT \\
    --lookback-min 1440 \\
    --bucket-sec 1 \\
    --rule micro_edge_v3_passive_alpha \\
    --h 120 \\
    --out-json reports/ADVERSE_MODEL_ETH_120.json \\
    --out-md   reports/ADVERSE_MODEL_ETH_120.md
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.micro_edge_lib import build_bucket_features, compute_rule_thresholds, rule_fires

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _quartile_label(v: float, q1: float, q2: float, q3: float) -> str:
    if v <= q1:
        return "Q1"
    if v <= q2:
        return "Q2"
    if v <= q3:
        return "Q3"
    return "Q4"


def _conditional_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return {"mean": round(m, 6), "std": round(s, 6), "n": len(values)}


def _quartile_groups(
    paired: List[Tuple[float, float]],  # (feature_value, adverse_bps)
) -> Dict[str, Dict[str, Any]]:
    """Bin (feature, adverse) pairs into Q1–Q4 by feature quartile."""
    if len(paired) < 4:
        return {}
    feature_vals = [x for x, _ in paired]
    q1 = _percentile(feature_vals, 0.25)
    q2 = _percentile(feature_vals, 0.50)
    q3 = _percentile(feature_vals, 0.75)
    bins: Dict[str, List[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for fv, av in paired:
        bins[_quartile_label(fv, q1, q2, q3)].append(av)
    return {k: _conditional_stats(v) for k, v in bins.items()}


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent.parent),
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# DB loading
# ─────────────────────────────────────────────

def _load_db_data(
    db: str, symbol: str, lookback_min: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load agg_trades and mark_prices for symbol from the last lookback_min minutes."""
    import sqlite3
    import time

    cutoff_ms = int((time.time() - lookback_min * 60) * 1000)
    sym = str(symbol).upper()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        trades_raw = conn.execute(
            "SELECT ts_ms, price, quantity, is_buyer_maker FROM agg_trades "
            "WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC",
            (sym, cutoff_ms),
        ).fetchall()
        marks_raw = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices "
            "WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC",
            (sym, cutoff_ms),
        ).fetchall()
    finally:
        conn.close()

    trades = [
        {
            "ts_ms": int(r["ts_ms"]),
            "price": float(r["price"]),
            "quantity": float(r["quantity"]),
            "is_buyer_maker": int(r["is_buyer_maker"]),
        }
        for r in trades_raw
    ]
    marks = [
        {"ts_ms": int(r["ts_ms"]), "mark_price": float(r["mark_price"])}
        for r in marks_raw
    ]
    return trades, marks


# ─────────────────────────────────────────────
# Forward return & adverse selection
# ─────────────────────────────────────────────

def _nearest_mark_price(
    ts_ms: int, mark_ts: List[int], mark_px: List[float]
) -> Optional[float]:
    """Binary search for the mark price at or just before ts_ms."""
    if not mark_ts:
        return None
    lo, hi, best = 0, len(mark_ts) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mark_ts[mid] <= ts_ms:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return mark_px[best]


def _compute_forward_returns(
    rows: List[Dict[str, Any]],
    marks: List[Dict[str, Any]],
    horizon_sec: int,
) -> List[Optional[float]]:
    """For each bucket row, compute signed forward return at horizon_sec.

    Uses mark_price timestamps directly (not bucket indices), so the result
    is accurate regardless of bucket_sec or sparse marks.
    """
    mark_ts = [m["ts_ms"] for m in marks]
    mark_px = [m["mark_price"] for m in marks]
    horizon_ms = horizon_sec * 1000
    results: List[Optional[float]] = []

    for row in rows:
        ts0 = int(row.get("ts_ms", 0))
        p0 = _nearest_mark_price(ts0, mark_ts, mark_px)
        p1 = _nearest_mark_price(ts0 + horizon_ms, mark_ts, mark_px)
        if p0 is None or p1 is None or p0 == 0.0:
            results.append(None)
        else:
            results.append((p1 - p0) / p0)
    return results


def _adverse_bps(forward_ret: float, side: str) -> float:
    """Adverse selection in bps: how much price moved against us after fill.

    LONG entry: adverse = max(0, -ret) * 10000  (price fell)
    SHORT entry: adverse = max(0, +ret) * 10000  (price rose)
    """
    if side == "SHORT":
        return max(0.0, forward_ret) * 10_000.0
    return max(0.0, -forward_ret) * 10_000.0


# ─────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────

def analyze_symbol(
    *,
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    rule: str,
    horizon_sec: int,
    side: str = "LONG",
) -> Dict[str, Any]:
    log.info(
        "Loading %s (lookback=%dmin, bucket_sec=%d, rule=%s, h=%ds)",
        symbol, lookback_min, bucket_sec, rule, horizon_sec,
    )
    try:
        trades, marks = _load_db_data(db, symbol, lookback_min)
    except Exception as exc:
        log.warning("DB load failed for %s: %s", symbol, exc)
        return {"error": str(exc)}

    log.info("  trades=%d  marks=%d", len(trades), len(marks))
    if not trades or not marks:
        return {"error": "insufficient_data"}

    rows = build_bucket_features(trades=trades, marks=marks, bucket_sec=bucket_sec)
    log.info("  bucket_rows=%d", len(rows))
    if not rows:
        return {"error": "no_bucket_rows"}

    thresholds = compute_rule_thresholds(rows)
    fwd_rets = _compute_forward_returns(rows, marks, horizon_sec)

    # Collect per-signal samples: (adverse_bps, spread, intensity, imbalance)
    samples: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not rule_fires(rule, row, thresholds):
            continue
        fret = fwd_rets[i]
        if fret is None:
            continue
        adv = _adverse_bps(fret, side)
        samples.append({
            "adverse_bps": adv,
            "spread": _safe_float(row.get("spread")),
            "trade_intensity": _safe_float(row.get("trade_intensity")),
            "imbalance": _safe_float(row.get("imbalance")),
        })

    n_signal = len(samples)
    log.info("  rule fires on %d / %d buckets", n_signal, len(rows))

    if n_signal < 5:
        return {
            "n_total_buckets": len(rows),
            "n_signal_buckets": n_signal,
            "error": "insufficient_signal_fires (need>=5)",
        }

    adverse_vals = [s["adverse_bps"] for s in samples]
    global_mean_val = mean(adverse_vals)
    global_std_val = stdev(adverse_vals) if len(adverse_vals) > 1 else 0.0

    pcts = {
        "p25": _percentile(adverse_vals, 0.25),
        "p50": _percentile(adverse_vals, 0.50),
        "p75": _percentile(adverse_vals, 0.75),
        "p90": _percentile(adverse_vals, 0.90),
        "p95": _percentile(adverse_vals, 0.95),
    }

    # Build paired (feature, adverse) for each feature
    def _paired(key: str) -> List[Tuple[float, float]]:
        return [
            (s[key], s["adverse_bps"])
            for s in samples
            if s[key] is not None
        ]

    cond_spread    = _quartile_groups(_paired("spread"))
    cond_intensity = _quartile_groups(_paired("trade_intensity"))
    cond_imbalance = _quartile_groups(_paired("imbalance"))

    # Store quantile edges so simulator can look up buckets
    spread_vals = [s["spread"] for s in samples if s["spread"] is not None]
    int_vals    = [s["trade_intensity"] for s in samples if s["trade_intensity"] is not None]
    imb_vals    = [s["imbalance"] for s in samples if s["imbalance"] is not None]

    def _edges(vals: List[float]) -> List[Optional[float]]:
        if len(vals) < 4:
            return [None, None, None]
        return [
            round(_percentile(vals, 0.25), 8),
            round(_percentile(vals, 0.50), 8),
            round(_percentile(vals, 0.75), 8),
        ]

    return {
        "n_total_buckets": len(rows),
        "n_signal_buckets": n_signal,
        "statistics": {
            "global_mean_adverse_bps":   round(global_mean_val, 6),
            "global_std_adverse_bps":    round(global_std_val, 6),
            "percentiles": {k: round(v, 6) for k, v in pcts.items()},
        },
        "conditional": {
            "by_spread_quartile":    cond_spread,
            "by_intensity_quartile": cond_intensity,
            "by_imbalance_quartile": cond_imbalance,
        },
        "quantile_edges": {
            "spread":    _edges(spread_vals),
            "trade_intensity": _edges(int_vals),
            "imbalance": _edges(imb_vals),
        },
        # Implied adverse_mult relative to a 1 bps baseline — useful for
        # calibrating the simulator's passive_adverse_mult parameter.
        "implied_adverse_mult_vs_1bps": round(global_mean_val, 4),
    }


def run(
    *,
    db: str,
    symbols: List[str],
    lookback_min: int,
    bucket_sec: int,
    rule: str,
    horizon_sec: int,
    side: str = "LONG",
) -> Dict[str, Any]:
    per_symbol = {}
    for sym in symbols:
        per_symbol[sym] = analyze_symbol(
            db=db,
            symbol=sym,
            lookback_min=lookback_min,
            bucket_sec=bucket_sec,
            rule=rule,
            horizon_sec=horizon_sec,
            side=side,
        )

    return {
        "version": "1.0",
        "tool": "fit_adverse_model",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "inputs": {
            "db": str(db),
            "symbols": symbols,
            "lookback_min": lookback_min,
            "bucket_sec": bucket_sec,
            "rule": rule,
            "horizon_sec": horizon_sec,
            "side": side,
        },
        "per_symbol": per_symbol,
    }


# ─────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────

def _render_md(result: Dict[str, Any]) -> str:
    inp = result.get("inputs", {})
    lines = [
        "# ADVERSE SELECTION MODEL — DIAGNOSTICS",
        "",
        f"generated : {result.get('generated_utc')}",
        f"git_hash  : {result.get('git_hash')}",
        f"rule      : {inp.get('rule')}",
        f"horizon   : {inp.get('horizon_sec')}s",
        f"lookback  : {inp.get('lookback_min')}min",
        f"bucket_sec: {inp.get('bucket_sec')}",
        "",
    ]
    for sym, data in result.get("per_symbol", {}).items():
        lines += [f"## {sym}", ""]
        if "error" in data:
            lines += [f"**Error:** `{data['error']}`", ""]
            continue

        stats = data.get("statistics", {})
        lines += [
            f"- Total buckets  : {data.get('n_total_buckets')}",
            f"- Signal-firing  : {data.get('n_signal_buckets')}",
            f"- Global mean adverse_bps : `{stats.get('global_mean_adverse_bps'):.4f}`",
            f"- Global std              : `{stats.get('global_std_adverse_bps'):.4f}`",
            f"- Implied passive_adverse_mult vs 1bps baseline: "
            f"`{data.get('implied_adverse_mult_vs_1bps')}`",
            "",
            "**Percentiles (adverse_bps):**",
        ]
        for k, v in stats.get("percentiles", {}).items():
            lines.append(f"  - {k}: `{v:.4f}`")

        for label, cond_key in [
            ("Spread",    "by_spread_quartile"),
            ("Intensity", "by_intensity_quartile"),
            ("Imbalance", "by_imbalance_quartile"),
        ]:
            cond = data.get("conditional", {}).get(cond_key, {})
            if not cond:
                continue
            lines += [
                "",
                f"**Conditional by {label} quartile:**",
                "",
                "| Quartile | mean_adverse_bps | std | n |",
                "|---|---:|---:|---:|",
            ]
            for q in ("Q1", "Q2", "Q3", "Q4"):
                s = cond.get(q, {})
                m = s.get("mean")
                lines.append(
                    f"| {q} | "
                    f"{'n/a' if m is None else f'{m:.4f}'} | "
                    f"{s.get('std', 0.0):.4f} | {s.get('n', 0)} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate adverse selection model from microstructure DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="data/microstructure.db",
                   help="SQLite microstructure DB path")
    p.add_argument("--symbol", default="ETHUSDT",
                   help="Comma-separated symbols (e.g. ETHUSDT,BTCUSDT)")
    p.add_argument("--lookback-min", type=int, default=1440,
                   help="Lookback window in minutes")
    p.add_argument("--bucket-sec", type=int, default=1,
                   help="Bucket size in seconds (must match ranking runs)")
    p.add_argument("--rule", default="micro_edge_v3_passive_alpha",
                   help="Signal rule name")
    p.add_argument("--h", type=int, default=120, dest="horizon_sec",
                   help="Forward return horizon in seconds")
    p.add_argument("--side", default="LONG", choices=["LONG", "SHORT"],
                   help="Assumed entry side for adverse calculation")
    p.add_argument("--out-json", default="reports/ADVERSE_MODEL.json",
                   help="Output model JSON path")
    p.add_argument("--out-md", default="reports/ADVERSE_MODEL.md",
                   help="Output diagnostics markdown path")
    return p.parse_args()


def main() -> int:
    args = _args()
    symbols = [s.strip().upper() for s in str(args.symbol).split(",") if s.strip()]
    result = run(
        db=str(args.db),
        symbols=symbols,
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        rule=str(args.rule),
        horizon_sec=int(args.horizon_sec),
        side=str(args.side),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s", out_json)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(result), encoding="utf-8")
    log.info("wrote %s", out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
