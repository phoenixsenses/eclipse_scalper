"""BookTicker impulse scan around S34 liquidation events.

Research only. Uses liquidation events as alarms, then requires short-horizon
bookTicker impulse/imbalance before entering. No runner/config/live changes.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import RiskConfig, S34Rule, _bucket_events, _close_trade, _fill_quote, _paper_trade_from_signal

SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_BOOK_IMPULSE_SCALP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_BOOK_IMPULSE_SCALP.md"

LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000
BUCKET_SEC = 300
MIN_GAP_SEC = 900

BUCKETS = [
    ("SOL_SELL_200K", "SOLUSDT", "SELL", "SHORT", 200_000.0),
    ("SOL_SELL_100K", "SOLUSDT", "SELL", "SHORT", 100_000.0),
    ("ETH_SELL_500K", "ETHUSDT", "SELL", "SHORT", 500_000.0),
    ("ETH_SELL_1M", "ETHUSDT", "SELL", "SHORT", 1_000_000.0),
    ("BTC_SELL_1M", "BTCUSDT", "SELL", "SHORT", 1_000_000.0),
]

IMPULSE_GRID = [
    {"name": "book1s_down2_imbNeg", "wait_sec": 1, "move_bps": 2.0, "imbalance_max": -0.2, "max_spread_bps": 1.0},
    {"name": "book3s_down3_imbNeg", "wait_sec": 3, "move_bps": 3.0, "imbalance_max": -0.2, "max_spread_bps": 1.0},
    {"name": "book5s_down5_imbNeg", "wait_sec": 5, "move_bps": 5.0, "imbalance_max": -0.2, "max_spread_bps": 1.0},
    {"name": "book10s_down5_imbNeg", "wait_sec": 10, "move_bps": 5.0, "imbalance_max": -0.2, "max_spread_bps": 1.0},
    {"name": "book5s_down3_imbAny", "wait_sec": 5, "move_bps": 3.0, "imbalance_max": None, "max_spread_bps": 1.0},
    {"name": "book10s_down8_imbAny", "wait_sec": 10, "move_bps": 8.0, "imbalance_max": None, "max_spread_bps": 1.5},
]

TP_GRID = [10.0, 15.0, 20.0, 25.0]
SL_GRID = [10.0, 15.0, 20.0]
BE_GRID = [6.0, 10.0, 15.0]
HOLD_GRID = [30, 60, 120]


def _rule(name: str, symbol: str, side: str, direction: str, threshold: float, tp: float, sl: float, be: float, hold: int) -> S34Rule:
    return S34Rule(
        name=name,
        symbol=symbol,
        liq_side=side,
        direction=direction,
        threshold_usd=threshold,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        tp_bps=tp,
        sl_bps=sl,
        be_trigger_bps=be,
        max_horizon_sec=hold,
        use_global_regime=False,
        require_book_ticker_fill=True,
    )


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    return f"{float(v):+.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.0f}%"


def _book_at_or_before(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def _book_series(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance
        FROM book_ticker
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()


def _choose_book_entry(
    conn: sqlite3.Connection,
    rule: S34Rule,
    signal: dict[str, Any],
    impulse: dict[str, Any],
) -> dict[str, Any] | None:
    signal_ts = int(signal["ts_ms"])
    start = _book_at_or_before(conn, rule.symbol, signal_ts)
    if start is None:
        return None
    target_ts = signal_ts + int(impulse["wait_sec"]) * 1000
    end = _book_at_or_before(conn, rule.symbol, target_ts)
    if end is None:
        return None
    if int(target_ts) - int(end["ts_ms"]) > 1000:
        return None

    start_mid = float(start["mid_price"])
    end_mid = float(end["mid_price"])
    move_bps = (start_mid - end_mid) / start_mid * 10_000.0
    spread_bps = float(end["spread_pct"]) * 10_000.0
    imb = float(end["book_imbalance"])
    if move_bps < float(impulse["move_bps"]):
        return None
    if impulse.get("imbalance_max") is not None and imb > float(impulse["imbalance_max"]):
        return None
    if spread_bps > float(impulse["max_spread_bps"]):
        return None

    try:
        fill = _fill_quote(conn, rule, rule.symbol, int(end["ts_ms"]), end_mid, rule.direction, "ENTRY", "taker")
    except RuntimeError as exc:
        clone = dict(signal)
        clone["fill_error"] = str(exc)
        return clone

    clone = dict(signal)
    clone.update(
        {
            "entry_ts_ms": int(fill["ts_ms"]),
            "entry_ts_utc": _iso(int(fill["ts_ms"])),
            "mark_ts_ms": int(end["ts_ms"]),
            "mark_ts_utc": _iso(int(end["ts_ms"])),
            "entry_reference_price": end_mid,
            "entry_price": float(fill["fill_price"]),
            "entry_fill": fill,
            "fill_error": "",
            "book_impulse_name": impulse["name"],
            "book_move_bps": move_bps,
            "book_spread_bps": spread_bps,
            "book_imbalance": imb,
            "signal_to_entry_sec": (int(fill["ts_ms"]) - signal_ts) / 1000.0,
        }
    )
    return clone


def _eval_trade(conn: sqlite3.Connection, trade: dict[str, Any], hold: int) -> dict[str, Any]:
    entry_ms = int(trade["entry_ts_ms"])
    rows = _book_series(conn, str(trade["symbol"]), entry_ms + 1, entry_ms + int(hold) * 1000)
    if not rows:
        return trade
    for row in rows:
        ts = int(row["ts_ms"])
        mid = float(row["mid_price"])
        if not trade["be_active"] and mid <= float(trade["be_trigger_price"]):
            trade["be_active"] = True
            trade["be_activated_ts_ms"] = ts
            trade["be_activated_ts_utc"] = _iso(ts)
        if mid <= float(trade["tp_price"]):
            return _close_trade(conn, trade, ts, mid, "TP")
        stop = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
        if mid >= stop:
            return _close_trade(conn, trade, ts, mid, "BE" if trade["be_active"] else "SL")
    row = rows[-1]
    return _close_trade(conn, trade, int(row["ts_ms"]), float(row["mid_price"]), "TIME")


def _simulate(
    conn: sqlite3.Connection,
    rule: S34Rule,
    raw_signals: list[dict[str, Any]],
    impulse: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = {"raw": len(raw_signals), "filtered": 0, "no_fill": 0, "closed": 0}
    rows: list[dict[str, Any]] = []
    for raw in raw_signals:
        sig = _choose_book_entry(conn, rule, raw, impulse)
        if sig is None:
            counts["filtered"] += 1
            continue
        if sig.get("fill_error"):
            counts["no_fill"] += 1
            continue
        trade = _paper_trade_from_signal(rule, sig, RiskConfig())
        try:
            evaluated = _eval_trade(conn, trade, int(rule.max_horizon_sec))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                counts["no_fill"] += 1
                continue
            raise
        if evaluated.get("status") == "CLOSED":
            rows.append(evaluated)
            counts["closed"] += 1
    return rows, counts


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "wr": None}
    days: dict[str, float] = defaultdict(float)
    exits: dict[str, int] = defaultdict(int)
    holds: list[float] = []
    impulses: list[float] = []
    adverse: list[float] = []
    for row in rows:
        days[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
        exits[str(row.get("exit_reason") or "?")] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        if row.get("book_move_bps") is not None:
            impulses.append(float(row["book_move_bps"]))
        if row.get("mark_to_fill_cost_bps") is not None:
            adverse.append(float(row["mark_to_fill_cost_bps"]))
    sorted_vals = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted_vals[3:]) if len(sorted_vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in days.values()),
        "total_days": len(days),
        "avg_hold_sec": statistics.mean(holds) if holds else None,
        "median_hold_sec": statistics.median(holds) if holds else None,
        "mean_book_move_bps": statistics.mean(impulses) if impulses else None,
        "mean_mark_to_fill_cost_bps": statistics.mean(adverse) if adverse else None,
        "exit_counts": dict(exits),
    }


def _verdict(s: dict[str, Any], counts: dict[str, int]) -> str:
    n = int(s.get("n") or 0)
    if n < 25:
        return "thin"
    if (s.get("median") or 0) <= 0:
        return "reject_negative_median"
    if (s.get("top3_removed_cum") or 0) <= 0:
        return "reject_outlier_dependent"
    if s.get("total_days") and s.get("positive_days") / s.get("total_days") < 0.6:
        return "watch_day_consistency"
    if counts["raw"] and counts["closed"] / counts["raw"] < 0.2:
        return "watch_too_selective"
    return "candidate"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=1")
    end_ms = int(conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0])
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000
    results: list[dict[str, Any]] = []

    for bucket, symbol, side, direction, threshold in BUCKETS:
        base = _rule(bucket, symbol, side, direction, threshold, 10.0, 10.0, 6.0, 30)
        raw_signals = _bucket_events(conn, base, start_ms, end_ms, SIGNAL_LIMIT)
        print(f"[bucket] {bucket} raw={len(raw_signals)}", flush=True)
        for impulse in IMPULSE_GRID:
            for tp in TP_GRID:
                for sl in SL_GRID:
                    for be in BE_GRID:
                        for hold in HOLD_GRID:
                            rule = _rule(bucket, symbol, side, direction, threshold, tp, sl, be, hold)
                            rows, counts = _simulate(conn, rule, raw_signals, impulse)
                            s = _summary(rows)
                            results.append(
                                {
                                    "bucket": bucket,
                                    "symbol": symbol,
                                    "threshold": threshold,
                                    "impulse": impulse,
                                    "tp_bps": tp,
                                    "sl_bps": sl,
                                    "be_bps": be,
                                    "max_hold_sec": hold,
                                    "counts": counts,
                                    "summary": s,
                                    "verdict": _verdict(s, counts),
                                }
                            )
    conn.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "scope": "liquidation alarm + bookTicker impulse fast scalp scan, real fills",
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = sorted(
        results,
        key=lambda r: (
            r["verdict"] == "candidate",
            float(r["summary"].get("median") or -9999),
            float(r["summary"].get("top3_removed_cum") or -999999),
        ),
        reverse=True,
    )
    lines = [
        "# S34 Book Impulse Fast Scalp Scan",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research only. Liquidation event is used as an alarm; bookTicker impulse decides entry. No runner/config/live changes.",
        "",
        "## Top 30 Rows",
        "",
        "| Bucket | Impulse | TP/SL/BE/H | Raw | Closed | Filtered | No-fill | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Book move | Adverse | Exits | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in best[:30]:
        s = r["summary"]
        c = r["counts"]
        exits = " ".join(f"{k}={v}" for k, v in sorted((s.get("exit_counts") or {}).items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    r["bucket"],
                    r["impulse"]["name"],
                    f"TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])}",
                    str(c["raw"]),
                    str(c["closed"]),
                    str(c["filtered"]),
                    str(c["no_fill"]),
                    _fmt(s.get("median")),
                    _fmt(s.get("mean")),
                    _pct(s.get("wr")),
                    _fmt(s.get("cum"), 0),
                    _fmt(s.get("top3_removed_cum"), 0),
                    f"{s.get('positive_days')}/{s.get('total_days')}",
                    "-" if s.get("avg_hold_sec") is None else f"{s['avg_hold_sec']:.0f}s",
                    _fmt(s.get("mean_book_move_bps")),
                    _fmt(s.get("mean_mark_to_fill_cost_bps")),
                    exits,
                    r["verdict"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Best Per Bucket",
        "",
        "| Bucket | Impulse | TP/SL/BE/H | Closed | Median | WR | T3R | Pos days | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bucket, *_ in BUCKETS:
        rows = [r for r in best if r["bucket"] == bucket]
        if not rows:
            continue
        r = rows[0]
        s = r["summary"]
        lines.append(
            f"| {bucket} | {r['impulse']['name']} | TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])} | {s.get('n')} | {_fmt(s.get('median'))} | {_pct(s.get('wr'))} | {_fmt(s.get('top3_removed_cum'),0)} | {s.get('positive_days')}/{s.get('total_days')} | {r['verdict']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `candidate` means this deserves a separate paper/shadow validation, not live trading.",
        "- Thin/highly filtered rows are hypothesis seeds only.",
        "- If no row survives top3-removed and day spread, liquidation is not sufficient as a fast-scalp alarm for this book impulse definition.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
