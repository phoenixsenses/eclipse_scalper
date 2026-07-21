"""
TP/SL sweep for top cross-symbol candidates:
  BTC 1000K, SOL 100K, SOL 200K  (+ ETH 500K as control baseline)

Sweeps TP × SL with real bookTicker fills, same engine as cross-symbol scan.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import (
    RiskConfig,
    S34Rule,
    _bucket_events,
    _evaluate_trade,
    _paper_trade_from_signal,
)

SOURCE_DB    = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON     = ROOT / "reports" / "research" / "s34" / "S34_TP_SL_SWEEP.json"
OUT_MD       = ROOT / "reports" / "research" / "s34" / "S34_TP_SL_SWEEP.md"

# Candidates from cross-symbol scan
CANDIDATES = [
    ("BTCUSDT", 1_000_000.0),
    ("SOLUSDT",   100_000.0),
    ("SOLUSDT",   200_000.0),
    ("ETHUSDT",   500_000.0),   # control — pre-reg variant
]

TP_GRID      = [40.0, 60.0, 80.0, 100.0]
SL_GRID      = [30.0, 40.0, 50.0]
BE_BPS       = 30.0             # fixed
BUCKET_SEC   = 300
MIN_GAP_SEC  = 900
MAX_HORIZON  = 3600
LOOKBACK_DAYS = 120
SIGNAL_LIMIT  = 100_000


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _summarize(rows: list[dict]) -> dict:
    vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0,
                "wr": None, "top3_removed": 0.0, "positive_days": 0, "exit_counts": {}}
    import statistics
    day_cums: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.get("net_bps") is not None:
            day_cums[_day(int(r["signal_ts_ms"]))] += float(r["net_bps"])
    exit_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        exit_counts[r.get("exit_reason") or "?"] += 1
    return {
        "n":             len(vals),
        "median":        statistics.median(vals),
        "mean":          statistics.mean(vals),
        "cum":           sum(vals),
        "wr":            sum(v > 0 for v in vals) / len(vals),
        "top3_removed":  sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "total_days":    len(day_cums),
        "exit_counts":   dict(exit_counts),
    }


def _run(conn: sqlite3.Connection, rule: S34Rule,
         start_ms: int, end_ms: int) -> dict:
    signals  = _bucket_events(conn, rule, start_ms, end_ms, SIGNAL_LIMIT)
    rows: list[dict] = []
    no_fill = 0
    for sig in signals:
        if sig.get("fill_error"):
            no_fill += 1
            continue
        trade = _paper_trade_from_signal(rule, sig, RiskConfig())
        try:
            ev = _evaluate_trade(
                conn, trade,
                min(end_ms, int(sig["entry_ts_ms"]) + MAX_HORIZON * 1000))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                no_fill += 1
                continue
            raise
        if ev.get("status") == "CLOSED":
            rows.append(ev)

    split_ts = signals[len(signals) // 2]["ts_ms"] if len(signals) >= 2 else None
    second   = [r for r in rows
                if split_ts and int(r["signal_ts_ms"]) >= int(split_ts)]
    return {
        "total_signals": len(signals),
        "real_closed":   len(rows),
        "no_fill":       no_fill,
        "no_fill_pct":   no_fill / len(signals) if signals else None,
        "all":           _summarize(rows),
        "second_half":   _summarize(second),
    }


def _fmt(v: Any, d: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.{d}f}" if v != 0 else f"{v:.{d}f}"
    return str(v)


def _pct(v: Any) -> str:
    return f"{float(v)*100:.0f}%" if v is not None else "—"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.execute("PRAGMA query_only=1")

    max_ts   = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms   = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    results: list[dict] = []
    total = len(CANDIDATES) * len(TP_GRID) * len(SL_GRID)
    done  = 0

    for symbol, threshold in CANDIDATES:
        for tp in TP_GRID:
            for sl in SL_GRID:
                done += 1
                label = f"{symbol} {int(threshold//1000)}K  TP{int(tp)}/SL{int(sl)}"
                print(f"  [{done:2d}/{total}]  {label}...", flush=True)
                rule = S34Rule(
                    name=f"{symbol}_BUY_LIQ_LONG_{int(threshold)}_"
                         f"TP{int(tp)}_SL{int(sl)}_BE{int(BE_BPS)}_SWEEP",
                    symbol=symbol,
                    liq_side="BUY",
                    direction="LONG",
                    threshold_usd=threshold,
                    bucket_sec=BUCKET_SEC,
                    min_gap_sec=MIN_GAP_SEC,
                    tp_bps=tp,
                    sl_bps=sl,
                    be_trigger_bps=BE_BPS,
                    max_horizon_sec=MAX_HORIZON,
                    use_global_regime=False,
                )
                out = _run(conn, rule, start_ms, end_ms)
                results.append({
                    "symbol": symbol, "threshold": threshold,
                    "tp_bps": tp, "sl_bps": sl, "be_bps": BE_BPS,
                    **out,
                })

    conn.close()

    # ── write JSON ────────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "candidates": [{"symbol": s, "threshold": t} for s, t in CANDIDATES],
        "tp_grid": TP_GRID, "sl_grid": SL_GRID, "be_bps": BE_BPS,
        "results": results,
    }, indent=2), encoding="utf-8")

    # ── write MD ──────────────────────────────────────────────────────────────
    lines = [
        "# S34 TP/SL Sweep — BTC/SOL/ETH Top Candidates",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`  ",
        f"Lookback: {LOOKBACK_DAYS}d | BE fixed at {BE_BPS} bps | real bookTicker fills",
        "",
    ]

    for symbol, threshold in CANDIDATES:
        label = f"{symbol} {int(threshold//1000)}K"
        lines += [f"## {label}", ""]
        subset = [r for r in results
                  if r["symbol"] == symbol and r["threshold"] == threshold]

        hdr = ["TP", "SL", "N", "Median", "Mean", "WR",
               "2nd Median", "2nd N", "Top3-Rmv", "No Fill%", "Exit mix"]
        rows_md = [hdr]
        for r in sorted(subset, key=lambda x: -(x["all"]["median"] or -999)):
            a  = r["all"]
            s2 = r["second_half"]
            exits = a.get("exit_counts", {})
            exit_str = " ".join(f"{k}={v}" for k, v in
                                sorted(exits.items(), key=lambda x: -x[1]))
            rows_md.append([
                f"TP{int(r['tp_bps'])}",
                f"SL{int(r['sl_bps'])}",
                a["n"],
                _fmt(a["median"]),
                _fmt(a["mean"]),
                _pct(a["wr"]),
                _fmt(s2["median"]),
                s2["n"],
                _fmt(a["top3_removed"]),
                _pct(r["no_fill_pct"]),
                exit_str,
            ])

        # markdown table
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("| " + " | ".join("---" for _ in hdr) + " |")
        for row in rows_md[1:]:
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

        # highlight best
        best = max(subset, key=lambda x: (x["all"]["median"] or -999))
        ba = best["all"]
        lines.append(
            f"> **Best:** TP{int(best['tp_bps'])}/SL{int(best['sl_bps'])}  "
            f"N={ba['n']}  median={_fmt(ba['median'])} bps  "
            f"WR={_pct(ba['wr'])}  2nd-half median={_fmt(best['second_half']['median'])} bps"
        )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
