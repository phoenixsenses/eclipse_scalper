"""
SELL liquidation cluster → SHORT entry scan.

Symmetric counterpart to the BUY-liq scan.
Hypothesis: large SELL liq cluster (forced short-covering = price spike) → exhaustion → SHORT.

Uses identical fill model, same 120-day lookback, TP60/SL40/BE30 baseline.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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

SOURCE_DB     = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON      = ROOT / "reports" / "research" / "s34" / "S34_SELL_LIQ_SCAN.json"
OUT_MD        = ROOT / "reports" / "research" / "s34" / "S34_SELL_LIQ_SCAN.md"

SYMBOLS       = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
THRESHOLDS    = [50_000.0, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0]
TP_BPS        = 60.0
SL_BPS        = 40.0
BE_BPS        = 30.0
LOOKBACK_DAYS = 120
SIGNAL_LIMIT  = 100_000
BUCKET_SEC    = 300
MIN_GAP_SEC   = 900
MAX_HORIZON   = 3600


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _summarize(rows: list[dict]) -> dict:
    import statistics
    vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0,
                "wr": None, "top3_removed": 0.0}
    day_cums: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.get("net_bps") is not None:
            day_cums[_day(int(r["signal_ts_ms"]))] += float(r["net_bps"])
    return {
        "n":             len(vals),
        "median":        statistics.median(vals),
        "mean":          statistics.mean(vals),
        "cum":           sum(vals),
        "wr":            sum(v > 0 for v in vals) / len(vals),
        "top3_removed":  sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "total_days":    len(day_cums),
    }


def _run(conn: sqlite3.Connection, rule: S34Rule,
         start_ms: int, end_ms: int) -> dict:
    signals = _bucket_events(conn, rule, start_ms, end_ms, SIGNAL_LIMIT)
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


def _fmt(v, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v):+.{d}f}" if float(v) != 0 else f"{float(v):.{d}f}"


def _pct(v) -> str:
    return f"{float(v)*100:.0f}%" if v is not None else "—"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.execute("PRAGMA query_only=1")

    max_ts   = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms   = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    results: list[dict] = []
    total = len(SYMBOLS) * len(THRESHOLDS)
    done  = 0

    for symbol in SYMBOLS:
        for threshold in THRESHOLDS:
            done += 1
            label = f"{symbol} {int(threshold//1000)}K"
            print(f"  [{done:2d}/{total}]  {label}...", flush=True)
            rule = S34Rule(
                name=f"{symbol}_SELL_LIQ_SHORT_{int(threshold)}_TP{int(TP_BPS)}_SL{int(SL_BPS)}_BE{int(BE_BPS)}_RESEARCH",
                symbol=symbol,
                liq_side="SELL",
                direction="SHORT",
                threshold_usd=threshold,
                bucket_sec=BUCKET_SEC,
                min_gap_sec=MIN_GAP_SEC,
                tp_bps=TP_BPS,
                sl_bps=SL_BPS,
                be_trigger_bps=BE_BPS,
                max_horizon_sec=MAX_HORIZON,
                use_global_regime=False,
            )
            out = _run(conn, rule, start_ms, end_ms)
            results.append({
                "symbol": symbol,
                "threshold": threshold,
                **out,
            })

    conn.close()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "liq_side": "SELL",
        "direction": "SHORT",
        "tp_bps": TP_BPS, "sl_bps": SL_BPS, "be_bps": BE_BPS,
        "results": results,
    }, indent=2), encoding="utf-8")

    lines = [
        "# S34 SELL Liquidation → SHORT Scan",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`  ",
        f"Lookback: {LOOKBACK_DAYS}d | TP{int(TP_BPS)}/SL{int(SL_BPS)}/BE{int(BE_BPS)} | real bookTicker fills",
        "",
        "Hypothesis: SELL liq cluster (short squeeze) → exhaustion → SHORT entry.",
        "",
        "## Full Grid",
        "",
    ]

    hdr = ["Symbol", "Threshold", "Signals", "Real", "No Fill", "No Fill%",
           "Median", "Mean", "Cum", "WR", "Days", "2nd Median", "2nd Cum", "Top3-Rmv"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join("---" for _ in hdr) + " |")
    for r in results:
        a  = r["all"]
        s2 = r["second_half"]
        lines.append("| " + " | ".join(str(x) for x in [
            r["symbol"],
            f"{int(r['threshold']//1000)}K",
            r["total_signals"],
            a["n"],
            r["no_fill"],
            _pct(r["no_fill_pct"]),
            _fmt(a["median"]),
            _fmt(a["mean"]),
            _fmt(a["cum"], 0),
            _pct(a["wr"]),
            a.get("total_days", "—"),
            _fmt(s2["median"]),
            _fmt(s2["cum"], 0),
            _fmt(a["top3_removed"], 0),
        ]) + " |")

    lines.append("")

    # candidate screen — same criteria as BUY scan
    screen = [r for r in results if (
        (r["all"]["n"] or 0) >= 25
        and (r["second_half"]["n"] or 0) >= 10
        and (r["all"]["median"] or -999) > 0
        and (r["second_half"]["median"] or -999) > 0
        and (r["all"]["top3_removed"] or -999) > 0
    )]
    screen.sort(key=lambda x: -(x["all"]["median"] or -999))

    lines += ["## Candidate Screen", "",
              "Screen: N>=25, second-half N>=10, full median >0, second-half median >0, top3-removed >0.",
              ""]
    if screen:
        hdr2 = ["Rank", "Symbol", "Threshold", "N", "Median", "2nd Median", "2nd Cum", "No Fill", "Note"]
        lines.append("| " + " | ".join(hdr2) + " |")
        lines.append("| " + " | ".join("---" for _ in hdr2) + " |")
        for i, r in enumerate(screen, 1):
            a = r["all"]
            s2 = r["second_half"]
            lines.append("| " + " | ".join(str(x) for x in [
                i, r["symbol"], f"{int(r['threshold']//1000)}K",
                a["n"], _fmt(a["median"]), _fmt(s2["median"]),
                _fmt(s2["cum"], 0),
                _pct(r["no_fill_pct"]),
                "passes screen",
            ]) + " |")
    else:
        lines.append("*No candidates passed the screen.*")

    lines += [
        "",
        "## Notes",
        "",
        "- SELL liq = liquidation of SHORT positions (forced short-covering → price spike).",
        "- SHORT entry = fade the squeeze: hypothesis is price reverts after forced covering exhausts.",
        "- Fill model: entry at bid (short), exits at ask (cover). Same taker fee as BUY side.",
        "- This is research-only. No runner or config changes.",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
