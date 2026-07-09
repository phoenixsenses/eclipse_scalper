"""Micro-entry fast scalp scan for S34 SELL-liq -> SHORT candidates.

Research only. Tests delayed/confirmed/pullback entries with real bookTicker
fills and short time stops. No runner/config/live state changes.
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

import tools.s34_shadow_paper_runner as runner
from tools.s34_shadow_paper_runner import (
    RiskConfig,
    S34Rule,
    _bucket_events,
    _close_trade,
    _fill_quote,
    _mark_at,
    _paper_trade_from_signal,
)

from ami.storage import production as PR
from ami.storage import research_reader as RR

SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_MICRO_ENTRY_SCALP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_MICRO_ENTRY_SCALP.md"

LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000
BUCKET_SEC = 300
MIN_GAP_SEC = 900

BASE_BUCKETS = [
    ("SOL_SELL_200K", "SOLUSDT", "SELL", "SHORT", 200_000.0),
    ("SOL_SELL_100K", "SOLUSDT", "SELL", "SHORT", 100_000.0),
    ("ETH_SELL_500K", "ETHUSDT", "SELL", "SHORT", 500_000.0),
    ("ETH_SELL_1M", "ETHUSDT", "SELL", "SHORT", 1_000_000.0),
    ("BTC_SELL_1M", "BTCUSDT", "SELL", "SHORT", 1_000_000.0),
]

ENTRY_MODES = [
    {"mode": "immediate", "wait_sec": 0, "confirm_bps": None, "pullback_bps": None},
    {"mode": "wait5", "wait_sec": 5, "confirm_bps": None, "pullback_bps": None},
    {"mode": "wait10", "wait_sec": 10, "confirm_bps": None, "pullback_bps": None},
    {"mode": "wait20", "wait_sec": 20, "confirm_bps": None, "pullback_bps": None},
    {"mode": "confirm10_down5", "wait_sec": 10, "confirm_bps": 5.0, "pullback_bps": None},
    {"mode": "confirm10_down10", "wait_sec": 10, "confirm_bps": 10.0, "pullback_bps": None},
    {"mode": "confirm20_down5", "wait_sec": 20, "confirm_bps": 5.0, "pullback_bps": None},
    {"mode": "pullback30_retrace5", "wait_sec": 30, "confirm_bps": None, "pullback_bps": 5.0},
]

TP_GRID = [20.0, 25.0, 30.0, 40.0]
SL_GRID = [20.0, 30.0]
BE_GRID = [10.0, 15.0, 20.0]
HOLD_GRID = [60, 120, 180]


def _base_rule(name: str, symbol: str, liq_side: str, direction: str, threshold: float) -> S34Rule:
    return S34Rule(
        name=name,
        symbol=symbol,
        liq_side=liq_side,
        direction=direction,
        threshold_usd=threshold,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        tp_bps=20.0,
        sl_bps=20.0,
        be_trigger_bps=10.0,
        max_horizon_sec=60,
        use_global_regime=False,
        require_book_ticker_fill=True,
    )


def _variant_rule(base: S34Rule, tp: float, sl: float, be: float, hold: int) -> S34Rule:
    return S34Rule(
        name=f"{base.name}_TP{int(tp)}_SL{int(sl)}_BE{int(be)}_H{int(hold)}",
        symbol=base.symbol,
        liq_side=base.liq_side,
        direction=base.direction,
        threshold_usd=base.threshold_usd,
        bucket_sec=base.bucket_sec,
        min_gap_sec=base.min_gap_sec,
        tp_bps=tp,
        sl_bps=sl,
        be_trigger_bps=be,
        max_horizon_sec=hold,
        taker_fee_bps=base.taker_fee_bps,
        maker_fee_bps=base.maker_fee_bps,
        tp_fill_mode=base.tp_fill_mode,
        max_book_staleness_sec=base.max_book_staleness_sec,
        require_book_ticker_fill=base.require_book_ticker_fill,
        use_global_regime=False,
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


def _mark_series(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mark_series_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V7). No longer called by main()'s live path."""
    return [
        (int(ts), float(px))
        for ts, px in conn.execute(
            """
            SELECT ts_ms, mark_price FROM mark_prices
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
            ORDER BY ts_ms ASC
            """,
            (symbol, int(start_ms), int(end_ms)),
        ).fetchall()
    ]


def _mark_series_v2(root: str, symbol: str, start_ms: int, end_ms: int,
                     source_db_path: str | None = None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `_mark_series`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter
    (SOLUSDT/ETHUSDT/BTCUSDT depending on the calling bucket).

    Range semantics: the oracle's SQL uses an INCLUSIVE upper bound
    (`ts_ms<=end_ms`); `plan_read`/`execute_read` use the reader's
    half-open `[start_ms, end_ms)` convention. Since `ts_ms` is always an
    integer column, `ts_ms<=end_ms` is exactly equivalent to
    `ts_ms<end_ms+1` -- so `end_ms+1` is passed here to reproduce the
    original inclusive boundary bit-for-bit, not approximately.

    Ordering: the reader's canonical `(ts_ms ASC, id ASC)` is a strict
    refinement of the oracle's `ORDER BY ts_ms ASC` (which has no
    tie-break at all) -- for the `(ts_ms, mark_price)` pairs collected
    here, any tie-break ordering among rows sharing the same `ts_ms`
    produces the same output sequence only if those rows also share
    `mark_price`; this is proven by direct parity test against real data
    rather than assumed."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(px)) for ts, px in result.iter_rows()]


def _first_mark_at_or_after(series: list[tuple[int, float]], ts_ms: int) -> tuple[int, float] | None:
    for ts, px in series:
        if ts >= ts_ms:
            return ts, px
    return None


def _choose_entry(
    conn: sqlite3.Connection,
    base: S34Rule,
    signal: dict[str, Any],
    mode: dict[str, Any],
    root: str,
    source_db_path: str,
) -> dict[str, Any] | None:
    signal_ts = int(signal["ts_ms"])
    ref_price = float(signal.get("entry_reference_price") or signal.get("mark_price") or signal.get("entry_price"))
    wait_sec = int(mode["wait_sec"])
    horizon_start = signal_ts
    horizon_end = signal_ts + max(wait_sec, 30) * 1000
    series = _mark_series_v2(root, base.symbol, horizon_start, horizon_end, source_db_path=source_db_path)
    if not series:
        return None

    chosen: tuple[int, float] | None = None
    if mode["mode"] == "immediate":
        chosen = _first_mark_at_or_after(series, signal_ts)
    elif mode.get("confirm_bps") is not None:
        chosen = _first_mark_at_or_after(series, signal_ts + wait_sec * 1000)
        if chosen is None:
            return None
        move_bps = (ref_price - float(chosen[1])) / ref_price * 10_000.0
        if move_bps < float(mode["confirm_bps"]):
            return None
    elif mode.get("pullback_bps") is not None:
        # For SHORT: require an initial favorable down move, then a small retrace
        # toward the reference price within 30s. Entry at the first retrace.
        min_px = ref_price
        pullback = float(mode["pullback_bps"])
        for ts, px in series:
            min_px = min(min_px, px)
            favorable_bps = (ref_price - min_px) / ref_price * 10_000.0
            retrace_bps = (px - min_px) / ref_price * 10_000.0
            if favorable_bps >= 5.0 and retrace_bps >= pullback:
                chosen = (ts, px)
                break
        if chosen is None:
            return None
    else:
        chosen = _first_mark_at_or_after(series, signal_ts + wait_sec * 1000)

    if chosen is None:
        return None

    entry_ts, entry_ref = int(chosen[0]), float(chosen[1])
    try:
        entry_fill = _fill_quote(conn, base, base.symbol, entry_ts, entry_ref, base.direction, "ENTRY", "taker")
    except RuntimeError as exc:
        clone = dict(signal)
        clone["fill_error"] = str(exc)
        return clone
    clone = dict(signal)
    clone.update(
        {
            "entry_ts_ms": int(entry_fill["ts_ms"]),
            "entry_ts_utc": _iso(int(entry_fill["ts_ms"])),
            "mark_ts_ms": int(entry_ts),
            "mark_ts_utc": _iso(int(entry_ts)),
            "entry_reference_price": float(entry_ref),
            "entry_price": float(entry_fill["fill_price"]),
            "entry_fill": entry_fill,
            "entry_mode": mode["mode"],
            "entry_wait_sec": wait_sec,
            "signal_to_entry_sec": (int(entry_fill["ts_ms"]) - signal_ts) / 1000.0,
            "fill_error": "",
        }
    )
    return clone


def _evaluate_from_marks(
    conn: sqlite3.Connection,
    trade: dict[str, Any],
    marks: list[tuple[int, float]],
    hold: int,
    end_ms: int,
) -> dict[str, Any]:
    entry_ms = int(trade["entry_ts_ms"])
    cutoff = min(int(end_ms), entry_ms + int(hold) * 1000)
    rows = [(ts, px) for ts, px in marks if entry_ms < ts <= cutoff]
    if not rows:
        return trade
    direction = str(trade["direction"]).upper()
    for ts, price in rows:
        if direction == "SHORT":
            if not trade["be_active"] and price <= float(trade["be_trigger_price"]):
                trade["be_active"] = True
                trade["be_activated_ts_ms"] = int(ts)
                trade["be_activated_ts_utc"] = _iso(int(ts))
            if price <= float(trade["tp_price"]):
                return _close_trade(conn, trade, int(ts), float(price), "TP")
            stop_price = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
            if price >= stop_price:
                return _close_trade(conn, trade, int(ts), float(price), "BE" if trade["be_active"] else "SL")
        else:
            raise ValueError("micro scalp scan currently expects SHORT")
    if cutoff >= entry_ms + int(hold) * 1000:
        ts, px = rows[-1]
        return _close_trade(conn, trade, int(ts), float(px), "TIME")
    return trade


def _path_stats(conn: sqlite3.Connection, row: dict[str, Any], root: str, source_db_path: str) -> dict[str, Any]:
    entry_ms, exit_ms = int(row["entry_ts_ms"]), int(row["exit_ts_ms"])
    basis = float(row.get("entry_reference_price") or row["entry_price"])
    marks = _mark_series_v2(root, str(row["symbol"]), entry_ms, exit_ms, source_db_path=source_db_path)
    if not marks:
        return {"mfe_bps": None, "mae_bps": None, "time_to_mfe_sec": None}
    prices = [px for _, px in marks]
    mfe_px = min(prices)
    mae_px = max(prices)
    idx = prices.index(mfe_px)
    return {
        "mfe_bps": (basis - mfe_px) / basis * 10_000.0,
        "mae_bps": (basis - mae_px) / basis * 10_000.0,
        "time_to_mfe_sec": (marks[idx][0] - entry_ms) / 1000.0,
    }


def _simulate_variant(
    conn: sqlite3.Connection,
    base: S34Rule,
    rule: S34Rule,
    raw_signals: list[dict[str, Any]],
    mode: dict[str, Any],
    end_ms: int,
    root: str,
    source_db_path: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"raw_signals": len(raw_signals), "entry_filtered": 0, "no_fill": 0, "closed": 0}
    for raw in raw_signals:
        sig = _choose_entry(conn, base, raw, mode, root, source_db_path)
        if sig is None:
            counts["entry_filtered"] += 1
            continue
        if sig.get("fill_error"):
            counts["no_fill"] += 1
            continue
        trade = _paper_trade_from_signal(rule, sig, RiskConfig())
        marks = _mark_series_v2(root, rule.symbol, int(trade["entry_ts_ms"]), int(trade["entry_ts_ms"]) + int(rule.max_horizon_sec) * 1000, source_db_path=source_db_path)
        try:
            evaluated = _evaluate_from_marks(conn, trade, marks, int(rule.max_horizon_sec), end_ms)
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                counts["no_fill"] += 1
                continue
            raise
        if evaluated.get("status") == "CLOSED":
            evaluated.update(_path_stats(conn, evaluated, root, source_db_path))
            evaluated["_signal"] = sig
            rows.append(evaluated)
            counts["closed"] += 1
    return rows, counts


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows if row.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "wr": None}
    days: dict[str, float] = defaultdict(float)
    exits: dict[str, int] = defaultdict(int)
    holds: list[float] = []
    ttms: list[float] = []
    adverse: list[float] = []
    for row in rows:
        days[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
        exits[str(row.get("exit_reason") or "?")] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        if row.get("time_to_mfe_sec") is not None:
            ttms.append(float(row["time_to_mfe_sec"]))
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
        "median_time_to_mfe_sec": statistics.median(ttms) if ttms else None,
        "mean_mark_to_fill_cost_bps": statistics.mean(adverse) if adverse else None,
        "exit_counts": dict(exits),
    }


def _verdict(summary: dict[str, Any], counts: dict[str, int]) -> str:
    n = int(summary.get("n") or 0)
    if n < 30:
        return "thin"
    if (summary.get("median") or 0.0) <= 0:
        return "reject_negative_median"
    if (summary.get("top3_removed_cum") or 0.0) <= 0:
        return "reject_outlier_dependent"
    if summary.get("total_days") and summary.get("positive_days") / summary.get("total_days") < 0.6:
        return "watch_day_consistency"
    if counts["raw_signals"] and counts["closed"] / counts["raw_signals"] < 0.25:
        return "watch_too_selective"
    return "candidate"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.execute("PRAGMA query_only=1")
    end_ms = int(conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0])
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    root, _ = PR.resolve_production_root()
    source_db_path = str(ROOT / "data" / "microstructure.db")

    results: list[dict[str, Any]] = []
    for name, symbol, liq_side, direction, threshold in BASE_BUCKETS:
        base = _base_rule(name, symbol, liq_side, direction, threshold)
        raw_signals = _bucket_events(conn, base, start_ms, end_ms, SIGNAL_LIMIT)
        print(f"[bucket] {name} raw_signals={len(raw_signals)}", flush=True)
        for mode in ENTRY_MODES:
            for tp in TP_GRID:
                for sl in SL_GRID:
                    for be in BE_GRID:
                        for hold in HOLD_GRID:
                            rule = _variant_rule(base, tp, sl, be, hold)
                            rows, counts = _simulate_variant(conn, base, rule, raw_signals, mode, end_ms, root, source_db_path)
                            s = _summary(rows)
                            results.append(
                                {
                                    "bucket": name,
                                    "symbol": symbol,
                                    "liq_side": liq_side,
                                    "direction": direction,
                                    "threshold": threshold,
                                    "entry_mode": mode,
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
        "scope": "SELL-liq SHORT micro-entry fast scalp scan, real bookTicker fills",
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = sorted(
        results,
        key=lambda r: (
            r["verdict"] == "candidate",
            float(r["summary"].get("median") or -9999.0),
            float(r["summary"].get("top3_removed_cum") or -999999.0),
        ),
        reverse=True,
    )

    lines = [
        "# S34 Micro-Entry Fast Scalp Scan",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Lookback: `{LOOKBACK_DAYS}d`",
        "",
        "Research only. Tests wait/confirmation/pullback entries with real bookTicker fills. No runner/config/live changes.",
        "",
        "## Top 30 Rows",
        "",
        "| Bucket | Entry | TP/SL/BE/H | Raw | Closed | Filtered | No-fill | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Adverse | Exits | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in best[:30]:
        s = r["summary"]
        c = r["counts"]
        exits = " ".join(f"{k}={v}" for k, v in sorted((s.get("exit_counts") or {}).items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["bucket"]),
                    str(r["entry_mode"]["mode"]),
                    f"TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])}",
                    str(c["raw_signals"]),
                    str(c["closed"]),
                    str(c["entry_filtered"]),
                    str(c["no_fill"]),
                    _fmt(s.get("median")),
                    _fmt(s.get("mean")),
                    _pct(s.get("wr")),
                    _fmt(s.get("cum"), 0),
                    _fmt(s.get("top3_removed_cum"), 0),
                    f"{s.get('positive_days')}/{s.get('total_days')}",
                    "-" if s.get("avg_hold_sec") is None else f"{s['avg_hold_sec']:.0f}s",
                    _fmt(s.get("mean_mark_to_fill_cost_bps")),
                    exits,
                    str(r["verdict"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Best Candidate Per Bucket",
        "",
        "| Bucket | Entry | TP/SL/BE/H | Closed | Median | WR | T3R | Pos days | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bucket in [b[0] for b in BASE_BUCKETS]:
        rows = [r for r in best if r["bucket"] == bucket]
        if not rows:
            continue
        r = rows[0]
        s = r["summary"]
        lines.append(
            f"| {bucket} | {r['entry_mode']['mode']} | TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])} | {s.get('n')} | {_fmt(s.get('median'))} | {_pct(s.get('wr'))} | {_fmt(s.get('top3_removed_cum'),0)} | {s.get('positive_days')}/{s.get('total_days')} | {r['verdict']} |"
        )

    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- `candidate` is research-only: it may justify a separately pre-registered paper/shadow route.",
        "- Confirmation/pullback modes can be over-selective; closed/raw coverage is shown explicitly.",
        "- If all candidates fail top3-removed or day consistency, do not add a fast scalp route.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
