"""Fast scalp sweep for S34 liquidation buckets.

Research only. Uses the corrected threshold-cross signal timestamp from
``_bucket_events`` and real bookTicker fills through the shadow runner helpers.
No runner/config files are modified by this script.
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
    DEFAULT_RULES,
    RiskConfig,
    S34Rule,
    _bucket_events,
    _close_trade,
    _deprecated_paper_rule_reason,
    _evaluate_trade,
    _paper_trade_from_signal,
)

SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_FAST_SCALP_SWEEP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_FAST_SCALP_SWEEP.md"

LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000
BUCKET_SEC = 300
MIN_GAP_SEC = 900
PROGRESS_EVERY = 50

# Stage-1 grid: broad enough to test whether a bucket has a fast scalp shape,
# small enough to run across every active bucket in one pass. Deep per-bucket
# sweeps should only be run after this screen identifies candidates.
TP_GRID = [20.0, 30.0, 40.0]
SL_GRID = [20.0, 30.0, 40.0]
BE_GRID = [15.0, 20.0]
HOLD_GRID = [180, 300]

def _active_bucket_combos() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for rule in DEFAULT_RULES:
        if _deprecated_paper_rule_reason(rule):
            continue
        combos.append(
            {
                "combo": rule.name,
                "symbol": rule.symbol,
                "liq_side": rule.liq_side,
                "direction": rule.direction,
                "threshold": rule.threshold_usd,
                "base_rule": rule,
                "current": {
                    "tp": rule.tp_bps,
                    "sl": rule.sl_bps,
                    "be": rule.be_trigger_bps,
                    "hold": rule.max_horizon_sec,
                },
            }
        )
    return combos


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _fmt(v: Any, digits: int = 1, signed: bool = True) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        sign = "+" if signed else ""
        return f"{v:{sign}.{digits}f}"
    return str(v)


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


def _simulate(
    conn: sqlite3.Connection,
    rule: S34Rule,
    signals: list[dict[str, Any]],
    end_ms: int,
    mark_cache: dict[int, list[tuple[int, float]]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    no_fill = 0
    for signal in signals:
        if signal.get("fill_error"):
            no_fill += 1
            continue
        trade = _paper_trade_from_signal(rule, signal, RiskConfig())
        try:
            if mark_cache is None:
                evaluated = _evaluate_trade(
                    conn,
                    trade,
                    min(int(end_ms), int(signal["entry_ts_ms"]) + int(rule.max_horizon_sec) * 1000),
                )
            else:
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
            rows.append(evaluated)
    return rows, no_fill


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows if row.get("net_bps") is not None]
    if not vals:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "cum": 0.0,
            "wr": None,
            "top3_removed_cum": 0.0,
            "positive_days": 0,
            "total_days": 0,
            "avg_hold_sec": None,
            "mean_fee": None,
            "mean_mfe_bps": None,
            "mean_mae_bps": None,
            "exit_counts": {},
        }

    day_cums: dict[str, float] = defaultdict(float)
    exit_counts: dict[str, int] = defaultdict(int)
    hold_secs: list[float] = []
    fees: list[float] = []
    mfes: list[float] = []
    maes: list[float] = []
    for row in rows:
        signal_ms = int(row.get("signal_ts_ms") or row.get("entry_ts_ms") or 0)
        day_cums[_day(signal_ms)] += float(row["net_bps"])
        exit_counts[str(row.get("exit_reason") or "?")] += 1
        if row.get("entry_ts_ms") and row.get("exit_ts_ms"):
            hold_secs.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        if row.get("fee_cost_bps") is not None:
            fees.append(float(row["fee_cost_bps"]))
        if row.get("mfe_bps") is not None:
            mfes.append(float(row["mfe_bps"]))
        if row.get("mae_bps") is not None:
            maes.append(float(row["mae_bps"]))

    sorted_vals = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted_vals[3:]) if len(sorted_vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "total_days": len(day_cums),
        "avg_hold_sec": statistics.mean(hold_secs) if hold_secs else None,
        "mean_fee": statistics.mean(fees) if fees else None,
        "mean_mfe_bps": statistics.mean(mfes) if mfes else None,
        "mean_mae_bps": statistics.mean(maes) if maes else None,
        "exit_counts": dict(exit_counts),
    }


def _mark_path_stats(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, float | None]:
    entry_ms = int(row["entry_ts_ms"])
    exit_ms = int(row["exit_ts_ms"])
    direction = str(row["direction"]).upper()
    basis = float(row.get("entry_reference_price") or row["entry_price"])
    marks = conn.execute(
        """
        SELECT mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (str(row["symbol"]), entry_ms, exit_ms),
    ).fetchall()
    if not marks:
        return {"mfe_bps": None, "mae_bps": None}
    prices = [float(item[0]) for item in marks]
    if direction == "LONG":
        mfe = (max(prices) - basis) / basis * 10_000.0
        mae = (min(prices) - basis) / basis * 10_000.0
    else:
        mfe = (basis - min(prices)) / basis * 10_000.0
        mae = (basis - max(prices)) / basis * 10_000.0
    return {"mfe_bps": mfe, "mae_bps": mae}


def _annotate_paths(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.update(_mark_path_stats(conn, row))


def _verdict(summary: dict[str, Any]) -> str:
    if int(summary.get("n") or 0) < 30:
        return "preliminary_N_lt_30"
    if (summary.get("median") or 0.0) <= 0:
        return "too_noisy_or_fee_dominated"
    if (summary.get("top3_removed_cum") or 0.0) <= 0:
        return "outlier_dependent"
    if summary.get("total_days") and summary.get("positive_days") / summary.get("total_days") < 0.55:
        return "day_consistency_weak"
    if (summary.get("avg_hold_sec") or 9999.0) > 600.0:
        return "not_fast_enough"
    return "viable_fast_scalp_candidate"


def _rule_for(combo: dict[str, Any], tp: float, sl: float, be: float, hold: int) -> S34Rule:
    base = combo.get("base_rule")
    if not isinstance(base, S34Rule):
        base = S34Rule(
            name=str(combo["combo"]),
            symbol=str(combo["symbol"]),
            liq_side=str(combo["liq_side"]),
            direction=str(combo["direction"]),
            threshold_usd=float(combo["threshold"]),
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            use_global_regime=False,
        )
    return S34Rule(
        name=f"{base.name}_FAST_TP{int(tp)}_SL{int(sl)}_BE{int(be)}_H{int(hold)}",
        symbol=base.symbol,
        liq_side=base.liq_side,
        direction=base.direction,
        threshold_usd=base.threshold_usd,
        bucket_sec=base.bucket_sec,
        min_gap_sec=base.min_gap_sec,
        tp_bps=float(tp),
        sl_bps=float(sl),
        be_trigger_bps=float(be),
        max_horizon_sec=int(hold),
        taker_fee_bps=base.taker_fee_bps,
        maker_fee_bps=base.maker_fee_bps,
        tp_fill_mode=base.tp_fill_mode,
        modeled_spread_bps=base.modeled_spread_bps,
        max_book_staleness_sec=base.max_book_staleness_sec,
        require_book_ticker_fill=base.require_book_ticker_fill,
        max_open_trades=base.max_open_trades,
        daily_max_sl=base.daily_max_sl,
        entry_delay_sec=base.entry_delay_sec,
        btc_confirm_symbol=base.btc_confirm_symbol,
        btc_pre_window_sec=base.btc_pre_window_sec,
        btc_pre_min_return_bps=base.btc_pre_min_return_bps,
        use_global_regime=base.use_global_regime,
        min_day_trend_bps=base.min_day_trend_bps,
        max_day_trend_bps=base.max_day_trend_bps,
        min_cluster_liq_count=base.min_cluster_liq_count,
        required_shape_label=base.required_shape_label,
        max_single_liq_share_pct=base.max_single_liq_share_pct,
        priority=base.priority,
    )


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=60)
    conn.execute("PRAGMA query_only=1")
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
        if isinstance(rule, dict):
            max_stale_sec = int(rule.get("max_book_staleness_sec") or 5)
            require_book = bool(rule.get("require_book_ticker_fill", True))
            modeled_spread_bps = float(rule.get("modeled_spread_bps") or 0.0)
            taker_fee_bps = float(rule.get("taker_fee_bps") or 4.0)
            maker_fee_bps = float(rule.get("maker_fee_bps") or 2.0)
        else:
            max_stale_sec = int(rule.max_book_staleness_sec)
            require_book = bool(rule.require_book_ticker_fill)
            modeled_spread_bps = float(rule.modeled_spread_bps)
            taker_fee_bps = float(rule.taker_fee_bps)
            maker_fee_bps = float(rule.maker_fee_bps)
        key = (
            str(symbol),
            int(ts_ms),
            str(direction).upper(),
            str(leg).upper(),
            str(mode).lower(),
            None if limit_price is None else round(float(limit_price), 8),
            max_stale_sec,
            require_book,
            modeled_spread_bps,
            taker_fee_bps,
            maker_fee_bps,
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
    max_ts = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000
    combos = _active_bucket_combos()

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    p678_like: dict[str, Any] | None = None
    total = len(combos) * len(TP_GRID) * len(SL_GRID) * len(BE_GRID) * len(HOLD_GRID)
    done = 0

    for combo in combos:
        print(f"[combo] {combo['combo']}", flush=True)
        base_rule = _rule_for(combo, 20.0, 20.0, 10.0, 180)
        signals = _bucket_events(conn, base_rule, start_ms, end_ms, SIGNAL_LIMIT)
        signal_count = len(signals)
        print(f"  signals={signal_count}", flush=True)
        fast_mark_cache = {
            int(signal["entry_ts_ms"]): _marks_for_signal(
                conn,
                str(combo["symbol"]),
                int(signal["entry_ts_ms"]),
                max(HOLD_GRID),
                end_ms,
            )
            for signal in signals
        }
        for tp in TP_GRID:
            for sl in SL_GRID:
                for be in BE_GRID:
                    for hold in HOLD_GRID:
                        done += 1
                        if done == 1 or done % PROGRESS_EVERY == 0 or done == total:
                            print(f"  progress {done}/{total}", flush=True)
                        rule = _rule_for(combo, tp, sl, be, hold)
                        rows, no_fill = _simulate(conn, rule, signals, end_ms, fast_mark_cache)
                        summary = _summarize(rows)
                        result = {
                            "combo": combo["combo"],
                            "symbol": combo["symbol"],
                            "liq_side": combo["liq_side"],
                            "direction": combo["direction"],
                            "threshold": combo["threshold"],
                            "tp_bps": tp,
                            "sl_bps": sl,
                            "be_bps": be,
                            "max_hold_sec": hold,
                            "total_signals": signal_count,
                            "closed": len(rows),
                            "no_fill": no_fill,
                            "no_fill_pct": no_fill / signal_count if signal_count else None,
                            "summary": summary,
                            "verdict": _verdict(summary),
                        }
                        results.append(result)

        current = combo["current"]
        current_rule = _rule_for(combo, current["tp"], current["sl"], current["be"], current["hold"])
        current_mark_cache = {
            int(signal["entry_ts_ms"]): _marks_for_signal(
                conn,
                str(combo["symbol"]),
                int(signal["entry_ts_ms"]),
                int(current["hold"]),
                end_ms,
            )
            for signal in signals
        }
        current_rows, current_no_fill = _simulate(conn, current_rule, signals, end_ms, current_mark_cache)
        current_summary = _summarize(current_rows)
        best = max(
            [r for r in results if r["combo"] == combo["combo"]],
            key=lambda r: (
                r["verdict"] == "viable_fast_scalp_candidate",
                r["summary"].get("median") if r["summary"].get("median") is not None else -9999.0,
                r["summary"].get("top3_removed_cum") if r["summary"].get("top3_removed_cum") is not None else -999999.0,
            ),
        )
        comparisons.append(
            {
                "combo": combo["combo"],
                "current": {
                    **current,
                    "closed": len(current_rows),
                    "no_fill": current_no_fill,
                    "summary": current_summary,
                    "verdict": _verdict(current_summary),
                },
                "best_fast": best,
            }
        )

        if combo["combo"] == "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40":
            p678_signal = next(
                (s for s in signals if 1782489071327 <= int(s["cluster_start_ts_ms"]) <= 1782489085000),
                None,
            )
            if p678_signal is not None:
                p678_rows = []
                for tp in TP_GRID:
                    for sl in SL_GRID:
                        for be in BE_GRID:
                            for hold in HOLD_GRID:
                                rule = _rule_for(combo, tp, sl, be, hold)
                                rows, _ = _simulate(conn, rule, [p678_signal], end_ms, {
                                    int(p678_signal["entry_ts_ms"]): fast_mark_cache.get(int(p678_signal["entry_ts_ms"]), [])
                                })
                                if rows:
                                    row = rows[0]
                                    p678_rows.append(
                                        {
                                            "tp_bps": tp,
                                            "sl_bps": sl,
                                            "be_bps": be,
                                            "max_hold_sec": hold,
                                            "exit_reason": row.get("exit_reason"),
                                            "net_bps": row.get("net_bps"),
                                            "exit_ts_utc": row.get("exit_ts_utc"),
                                        }
                                    )
                p678_like = {
                    "corrected_signal_ts_utc": p678_signal.get("ts_utc"),
                    "corrected_entry_ts_utc": p678_signal.get("entry_ts_utc"),
                    "cluster_notional_at_cross": p678_signal.get("liq_total_notional"),
                    "rows": sorted(
                        p678_rows,
                        key=lambda r: (float(r["net_bps"] or -9999.0)),
                        reverse=True,
                    )[:20],
                }

    runner._fill_quote = original_fill_quote
    conn.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "tp_grid": TP_GRID,
        "sl_grid": SL_GRID,
        "be_grid": BE_GRID,
        "hold_grid": HOLD_GRID,
        "combos": [
            {
                "combo": combo["combo"],
                "symbol": combo["symbol"],
                "liq_side": combo["liq_side"],
                "direction": combo["direction"],
                "threshold": combo["threshold"],
                "current": combo["current"],
            }
            for combo in combos
        ],
        "results": results,
        "comparisons": comparisons,
        "p678_like": p678_like,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Fast Scalp Sweep",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Lookback: `{LOOKBACK_DAYS}d` | active buckets: `{len(combos)}` | signal timing: `threshold-cross` | fills: `real bookTicker`",
        "",
        "Research only. No runner/config/rule changes.",
        "",
        "## Best Fast Route Per Combo",
        "",
        "| Combo | Best fast route | N | Median | Mean | WR | Cum | Top3 removed | Pos days | Avg hold | No-fill | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for comp in comparisons:
        best = comp["best_fast"]
        s = best["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    comp["combo"],
                    f"TP{int(best['tp_bps'])}/SL{int(best['sl_bps'])}/BE{int(best['be_bps'])}/H{int(best['max_hold_sec'])}",
                    str(s["n"]),
                    _fmt(s["median"]),
                    _fmt(s["mean"]),
                    _pct(s["wr"]),
                    _fmt(s["cum"]),
                    _fmt(s["top3_removed_cum"]),
                    f"{s['positive_days']}/{s['total_days']}",
                    _fmt(s["avg_hold_sec"], 0, signed=False),
                    _pct(best["no_fill_pct"]),
                    best["verdict"],
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Best Fast vs Current Route",
        "",
        "| Combo | Current route | Current N | Current median | Current WR | Best fast median | Best fast WR | Delta median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comp in comparisons:
        cur = comp["current"]
        cur_s = cur["summary"]
        best = comp["best_fast"]
        best_s = best["summary"]
        cur_med = cur_s["median"]
        best_med = best_s["median"]
        delta = None if cur_med is None or best_med is None else float(best_med) - float(cur_med)
        lines.append(
            "| "
            + " | ".join(
                [
                    comp["combo"],
                    f"TP{int(cur['tp'])}/SL{int(cur['sl'])}/BE{int(cur['be'])}/H{int(cur['hold'])}",
                    str(cur_s["n"]),
                    _fmt(cur_med),
                    _pct(cur_s["wr"]),
                    _fmt(best_med),
                    _pct(best_s["wr"]),
                    _fmt(delta),
                ]
            )
            + " |"
        )

    if p678_like:
        lines += [
            "",
            "## P678-Like Corrected Threshold-Cross Replay",
            "",
            f"Corrected signal: `{p678_like['corrected_signal_ts_utc']}` | entry: `{p678_like['corrected_entry_ts_utc']}` | notional-at-cross: `{p678_like['cluster_notional_at_cross']:.2f}`",
            "",
            "| Route | Exit | Net bps | Exit time |",
            "| --- | --- | ---: | --- |",
        ]
        for row in p678_like["rows"][:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"TP{int(row['tp_bps'])}/SL{int(row['sl_bps'])}/BE{int(row['be_bps'])}/H{int(row['max_hold_sec'])}",
                        str(row["exit_reason"]),
                        _fmt(float(row["net_bps"])),
                        str(row["exit_ts_utc"]),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- `viable_fast_scalp_candidate` requires N>=30, positive median, positive top3-removed cumulative net, >=55% positive days, and average hold <=600s.",
        "- This is an in-sample route sweep. Any candidate needs temporal OOS and forward paper validation before use.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
