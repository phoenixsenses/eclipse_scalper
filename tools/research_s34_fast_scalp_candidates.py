"""Focused fast-scalp validation for S34 SELL-liq continuation candidates.

Research only. This script does not modify runner/config/live state. It uses
the same threshold-cross event extraction and real bookTicker fill helpers as
the shadow runner, but only for the shortlisted fast-scalp candidates.
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
    _evaluate_trade,
    _paper_trade_from_signal,
)

SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_FAST_SCALP_CANDIDATES.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_FAST_SCALP_CANDIDATES.md"

LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000
BUCKET_SEC = 300
MIN_GAP_SEC = 900


CANDIDATES = [
    {
        "name": "SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40",
        "symbol": "SOLUSDT",
        "liq_side": "SELL",
        "direction": "SHORT",
        "threshold_usd": 200_000.0,
        "tp_bps": 40.0,
        "sl_bps": 30.0,
        "be_bps": 40.0,
        "max_horizon_sec": 300,
    },
    {
        "name": "SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40",
        "symbol": "SOLUSDT",
        "liq_side": "SELL",
        "direction": "SHORT",
        "threshold_usd": 100_000.0,
        "tp_bps": 40.0,
        "sl_bps": 30.0,
        "be_bps": 40.0,
        "max_horizon_sec": 300,
    },
    {
        "name": "BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40",
        "symbol": "BTCUSDT",
        "liq_side": "SELL",
        "direction": "SHORT",
        "threshold_usd": 1_000_000.0,
        "tp_bps": 40.0,
        "sl_bps": 40.0,
        "be_bps": 40.0,
        "max_horizon_sec": 300,
    },
]


def _rule(spec: dict[str, Any]) -> S34Rule:
    return S34Rule(
        name=str(spec["name"]),
        symbol=str(spec["symbol"]),
        liq_side=str(spec["liq_side"]),
        direction=str(spec["direction"]),
        threshold_usd=float(spec["threshold_usd"]),
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        tp_bps=float(spec["tp_bps"]),
        sl_bps=float(spec["sl_bps"]),
        be_trigger_bps=float(spec["be_bps"]),
        max_horizon_sec=int(spec["max_horizon_sec"]),
        use_global_regime=False,
        require_book_ticker_fill=True,
    )


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _fmt(v: Any, digits: int = 1, signed: bool = True) -> str:
    if v is None:
        return "-"
    sign = "+" if signed else ""
    return f"{float(v):{sign}.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.0f}%"


def _marks_for_signal(
    conn: sqlite3.Connection,
    symbol: str,
    entry_ts_ms: int,
    max_horizon_sec: int,
    end_ms: int,
) -> list[tuple[int, float]]:
    cutoff = min(int(end_ms), int(entry_ts_ms) + int(max_horizon_sec) * 1000)
    return [
        (int(ts), float(px))
        for ts, px in conn.execute(
            """
            SELECT ts_ms, mark_price FROM mark_prices
            WHERE symbol=? AND ts_ms>? AND ts_ms<=?
            ORDER BY ts_ms ASC
            """,
            (symbol, int(entry_ts_ms), cutoff),
        ).fetchall()
    ]


def _evaluate_trade_from_marks(
    conn: sqlite3.Connection,
    trade: dict[str, Any],
    marks: list[tuple[int, float]],
    max_horizon_sec: int,
    end_ms: int,
) -> dict[str, Any]:
    if trade.get("status") != "OPEN":
        return trade
    entry_ms = int(trade.get("entry_ts_ms") or trade.get("signal_ts_ms") or 0)
    cutoff_ms = min(int(end_ms), entry_ms + int(max_horizon_sec) * 1000)
    rows = [(ts, px) for ts, px in marks if ts <= cutoff_ms]
    if not rows:
        return trade

    direction = str(trade["direction"]).upper()
    for ts, price in rows:
        trade["last_evaluated_mark_ts_ms"] = int(ts)
        trade["last_evaluated_mark_ts_utc"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        if direction == "LONG":
            if not trade["be_active"] and price >= float(trade["be_trigger_price"]):
                trade["be_active"] = True
                trade["be_activated_ts_ms"] = int(ts)
                trade["be_activated_ts_utc"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            if price >= float(trade["tp_price"]):
                return _close_trade(conn, trade, int(ts), float(price), "TP")
            stop_price = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
            if price <= stop_price:
                return _close_trade(conn, trade, int(ts), float(price), "BE" if trade["be_active"] else "SL")
        else:
            if not trade["be_active"] and price <= float(trade["be_trigger_price"]):
                trade["be_active"] = True
                trade["be_activated_ts_ms"] = int(ts)
                trade["be_activated_ts_utc"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            if price <= float(trade["tp_price"]):
                return _close_trade(conn, trade, int(ts), float(price), "TP")
            stop_price = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
            if price >= stop_price:
                return _close_trade(conn, trade, int(ts), float(price), "BE" if trade["be_active"] else "SL")

    if cutoff_ms >= entry_ms + int(max_horizon_sec) * 1000:
        last_ts, last_px = rows[-1]
        return _close_trade(conn, trade, int(last_ts), float(last_px), "TIME")
    return trade


def _mark_path_stats(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, float | None]:
    entry_ms = int(row["entry_ts_ms"])
    exit_ms = int(row["exit_ts_ms"])
    basis = float(row.get("entry_reference_price") or row["entry_price"])
    marks = conn.execute(
        """
        SELECT ts_ms, mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (str(row["symbol"]), entry_ms, exit_ms),
    ).fetchall()
    if not marks:
        return {"mfe_bps": None, "mae_bps": None, "time_to_mfe_sec": None}
    prices = [float(item[1]) for item in marks]
    direction = str(row["direction"]).upper()
    if direction == "LONG":
        mfe_price = max(prices)
        mae_price = min(prices)
        mfe = (mfe_price - basis) / basis * 10_000.0
        mae = (mae_price - basis) / basis * 10_000.0
    else:
        mfe_price = min(prices)
        mae_price = max(prices)
        mfe = (basis - mfe_price) / basis * 10_000.0
        mae = (basis - mae_price) / basis * 10_000.0
    mfe_idx = prices.index(mfe_price)
    return {
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_sec": (int(marks[mfe_idx][0]) - entry_ms) / 1000.0,
    }


def _simulate(
    conn: sqlite3.Connection,
    rule: S34Rule,
    signals: list[dict[str, Any]],
    end_ms: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    no_fill = 0
    mark_cache = {
        int(signal["entry_ts_ms"]): _marks_for_signal(
            conn,
            rule.symbol,
            int(signal["entry_ts_ms"]),
            int(rule.max_horizon_sec),
            end_ms,
        )
        for signal in signals
    }
    for signal in signals:
        if signal.get("fill_error"):
            no_fill += 1
            continue
        trade = _paper_trade_from_signal(rule, signal, RiskConfig())
        try:
            evaluated = _evaluate_trade_from_marks(
                conn,
                trade,
                mark_cache.get(int(signal["entry_ts_ms"]), []),
                int(rule.max_horizon_sec),
                int(end_ms),
            )
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                no_fill += 1
                continue
            raise
        if evaluated.get("status") == "CLOSED":
            evaluated.update(_mark_path_stats(conn, evaluated))
            evaluated["_signal"] = signal
            rows.append(evaluated)
    return rows, no_fill


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows if row.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "wr": None}
    by_day: dict[str, float] = defaultdict(float)
    exits: dict[str, int] = defaultdict(int)
    holds: list[float] = []
    mfes: list[float] = []
    maes: list[float] = []
    ttms: list[float] = []
    for row in rows:
        by_day[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
        exits[str(row.get("exit_reason") or "?")] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        if row.get("mfe_bps") is not None:
            mfes.append(float(row["mfe_bps"]))
        if row.get("mae_bps") is not None:
            maes.append(float(row["mae_bps"]))
        if row.get("time_to_mfe_sec") is not None:
            ttms.append(float(row["time_to_mfe_sec"]))
    sorted_vals = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted_vals[3:]) if len(sorted_vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in by_day.values()),
        "total_days": len(by_day),
        "avg_hold_sec": statistics.mean(holds) if holds else None,
        "median_hold_sec": statistics.median(holds) if holds else None,
        "median_mfe_bps": statistics.median(mfes) if mfes else None,
        "median_mae_bps": statistics.median(maes) if maes else None,
        "median_time_to_mfe_sec": statistics.median(ttms) if ttms else None,
        "fast3_pct": sum(t <= 180 for t in ttms) / len(ttms) if ttms else None,
        "fast5_pct": sum(t <= 300 for t in ttms) / len(ttms) if ttms else None,
        "exit_counts": dict(exits),
    }


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("signal_ts_ms") or 0))
    mid = len(ordered) // 2
    return {"first_half": _summary(ordered[:mid]), "second_half": _summary(ordered[mid:])}


def _bucketize(value: float | None, cuts: list[tuple[str, float | None, float | None]]) -> str:
    if value is None:
        return "unknown"
    for label, lo, hi in cuts:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return label
    return "unknown"


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sig = row.get("_signal") or {}
        groups[f"trend:{_bucketize(sig.get('day_trend_bps'), [('bear', None, 0), ('bull', 0, None)])}"].append(row)
        groups[f"range:{_bucketize(sig.get('day_range_bps'), [('<250', None, 250), ('250-500', 250, 500), ('>=500', 500, None)])}"].append(row)
        groups[f"notional:{_bucketize(sig.get('liq_total_notional'), [('<500K', None, 500000), ('500K-1M', 500000, 1000000), ('>=1M', 1000000, None)])}"].append(row)
        groups[f"liq_count:{_bucketize(sig.get('cluster_liq_count') or sig.get('liq_count'), [('<5', None, 5), ('5-15', 5, 15), ('>=15', 15, None)])}"].append(row)
        share = sig.get("max_single_liq_share")
        if share is None:
            share = sig.get("max_single_liq_share_pct")
        groups[f"single_share:{_bucketize(share, [('<50', None, 50), ('50-80', 50, 80), ('>=80', 80, None)])}"].append(row)
    return {key: _summary(val) for key, val in sorted(groups.items()) if val}


def _verdict(s: dict[str, Any], no_fill_pct: float | None) -> str:
    if int(s.get("n") or 0) < 30:
        return "thin"
    if (s.get("median") or 0.0) <= 0:
        return "reject_negative_median"
    if (s.get("top3_removed_cum") or 0.0) <= 0:
        return "reject_outlier_dependent"
    if s.get("total_days") and s.get("positive_days") / s.get("total_days") < 0.6:
        return "watch_day_consistency"
    if no_fill_pct is not None and no_fill_pct > 0.35:
        return "watch_no_fill_high"
    return "candidate_for_paper_shadow"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.execute("PRAGMA query_only=1")
    max_ts = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    original_fill_quote = runner._fill_quote
    fill_cache: dict[tuple[Any, ...], tuple[bool, Any]] = {}

    def cached_fill_quote(
        fill_conn: sqlite3.Connection,
        rule: S34Rule | dict[str, Any],
        symbol: str,
        ts_ms: int,
        reference_price: float,
        direction: str,
        leg: str,
        mode: str,
        *,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        key = (
            str(symbol),
            int(ts_ms),
            str(direction).upper(),
            str(leg).upper(),
            str(mode).lower(),
            None if limit_price is None else round(float(limit_price), 8),
        )
        cached = fill_cache.get(key)
        if cached is not None:
            ok, value = cached
            if ok:
                return dict(value)
            raise RuntimeError(str(value))
        try:
            value = original_fill_quote(
                fill_conn,
                rule,
                symbol,
                ts_ms,
                reference_price,
                direction,
                leg,
                mode,
                limit_price=limit_price,
            )
        except RuntimeError as exc:
            fill_cache[key] = (False, str(exc))
            raise
        fill_cache[key] = (True, dict(value))
        return value

    runner._fill_quote = cached_fill_quote
    results: list[dict[str, Any]] = []
    try:
        for spec in CANDIDATES:
            rule = _rule(spec)
            print(f"[candidate] {rule.name}", flush=True)
            signals = _bucket_events(conn, rule, start_ms, end_ms, SIGNAL_LIMIT)
            rows, no_fill = _simulate(conn, rule, signals, end_ms)
            s = _summary(rows)
            no_fill_pct = no_fill / len(signals) if signals else None
            result = {
                "candidate": spec,
                "signals": len(signals),
                "closed": len(rows),
                "no_fill": no_fill,
                "no_fill_pct": no_fill_pct,
                "summary": s,
                "halves": _split_summary(rows),
                "groups": _group_summaries(rows),
                "verdict": _verdict(s, no_fill_pct),
            }
            results.append(result)
            print(
                f"  signals={len(signals)} closed={len(rows)} no_fill={no_fill} "
                f"median={_fmt(s.get('median'))} wr={_pct(s.get('wr'))} verdict={result['verdict']}",
                flush=True,
            )
    finally:
        runner._fill_quote = original_fill_quote
        conn.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "source_db": "data/microstructure.db",
        "scope": "focused fast-scalp candidates, real bookTicker fills, research only",
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Fast Scalp Candidate Validation",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Lookback: `{LOOKBACK_DAYS}d`",
        "",
        "Focused validation of SELL-liq -> SHORT fast scalp candidates. Real bookTicker fills via shadow-runner helpers. No runner/config changes.",
        "",
        "## Summary",
        "",
        "| Candidate | Signals | Closed | No-fill | Median | Mean | WR | Cum | Top3 removed | Pos days | Med TTM | Fast5 | Exits | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in results:
        s = item["summary"]
        exits = " ".join(f"{k}={v}" for k, v in sorted((s.get("exit_counts") or {}).items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    item["candidate"]["name"],
                    str(item["signals"]),
                    str(item["closed"]),
                    _pct(item["no_fill_pct"]),
                    _fmt(s.get("median")),
                    _fmt(s.get("mean")),
                    _pct(s.get("wr")),
                    _fmt(s.get("cum"), 0),
                    _fmt(s.get("top3_removed_cum"), 0),
                    f"{s.get('positive_days')}/{s.get('total_days')}",
                    "-" if s.get("median_time_to_mfe_sec") is None else f"{s['median_time_to_mfe_sec']/60:.1f}m",
                    _pct(s.get("fast5_pct")),
                    exits,
                    item["verdict"],
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Half Split",
        "",
        "| Candidate | Half | N | Median | WR | Cum | Top3 removed | Pos days |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        for half, s in item["halves"].items():
            lines.append(
                f"| {item['candidate']['name']} | {half} | {s.get('n')} | {_fmt(s.get('median'))} | {_pct(s.get('wr'))} | {_fmt(s.get('cum'),0)} | {_fmt(s.get('top3_removed_cum'),0)} | {s.get('positive_days')}/{s.get('total_days')} |"
            )

    lines += [
        "",
        "## Regime / Geometry Splits",
        "",
        "| Candidate | Slice | N | Median | WR | Cum | Pos days |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        for key, s in item["groups"].items():
            if int(s.get("n") or 0) < 5:
                continue
            lines.append(
                f"| {item['candidate']['name']} | {key} | {s.get('n')} | {_fmt(s.get('median'))} | {_pct(s.get('wr'))} | {_fmt(s.get('cum'),0)} | {s.get('positive_days')}/{s.get('total_days')} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `candidate_for_paper_shadow` means the route is strong enough for a separately pre-registered paper/shadow bucket, not live capital.",
        "- `watch_no_fill_high` means performance is positive but bookTicker coverage bias is too large to ignore.",
        "- `thin` means N is too small for a route decision.",
        "- This report does not change any live executor allow-list.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
