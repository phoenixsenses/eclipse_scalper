from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from ami.storage import production as PR
from ami.storage import research_reader as RR

ROOT = Path(__file__).resolve().parents[1]
DB_URI = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_REGIME_FILTER_SHADOW_EVAL.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_REGIME_FILTER_SHADOW_EVAL.md"


def utc_iso(ts_ms: int | float | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def signed_bps(entry: float, price: float, direction: str) -> float:
    if direction.upper() == "SHORT":
        return (entry - price) / entry * 10000.0
    return (price - entry) / entry * 10000.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows]
    if not vals:
        return {
            "n": 0,
            "cum_net_bps": 0.0,
            "mean_net_bps": None,
            "median_net_bps": None,
            "win_rate": None,
            "exit_counts": {},
        }
    return {
        "n": len(vals),
        "cum_net_bps": sum(vals),
        "mean_net_bps": statistics.mean(vals),
        "median_net_bps": statistics.median(vals),
        "win_rate": sum(v > 0 for v in vals) / len(vals),
        "exit_counts": dict(sorted(Counter(row["exit_reason"] for row in rows).items())),
    }


def book_at(con: sqlite3.Connection, symbol: str, ts_ms: int, max_staleness_ms: int) -> dict[str, Any] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-
    MIGRATION-V7). No longer called by `simulate_counterfactual`; the
    reader-backed path is used instead."""
    row = con.execute(
        """
        SELECT ts_ms, bid_price, ask_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    age_ms = int(ts_ms) - int(row[0])
    if age_ms > int(max_staleness_ms):
        return None
    bid = float(row[1])
    ask = float(row[2])
    return {"ts_ms": int(row[0]), "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "age_ms": age_ms}


_BOOK_COLS = ("ts_ms", "bid_price", "ask_price")


def book_at_v2(root, symbol: str, ts_ms: int, max_staleness_ms: int, source_db_path=None) -> dict[str, Any] | None:
    """Reader-backed replacement for `book_at`, via
    lookup_latest_at_or_before. `symbol` is a genuine runtime parameter
    here, sourced from each shadow trade's own `symbol`/`rule.symbol`
    field (defaulting to "ETHUSDT" only if absent) -- this file's real
    trades snapshot spans ETHUSDT/SOLUSDT/BTCUSDT, though every real
    REGIME_FILTER-skip trade observed to date happens to be ETHUSDT
    (confirmed by direct inspection of the trades JSON, not assumed).
    The helper is always called with whichever symbol the trade actually
    names; it never assumes SOLUSDT's archive coverage applies to a
    different symbol."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask = result.row
    age_ms = int(ts_ms) - int(row_ts)
    if age_ms > int(max_staleness_ms):
        return None
    bid = float(bid)
    ask = float(ask)
    return {"ts_ms": int(row_ts), "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "age_ms": age_ms}


def mark_rows(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `mark_rows_v2` (the mark_prices range-read leg of this file's
    migration; the book_ticker ASOF leg was migrated separately in Batch
    4/b40441f2, see `book_at`/`book_at_v2` above). No longer called by
    `simulate_counterfactual`; the reader-backed path is used instead.
    Both bounds are INCLUSIVE (`ts_ms>=start_ms AND ts_ms<=end_ms`) --
    unlike some other migrated files' oracles, this one has no
    exclusive-lower variant to worry about."""
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            SELECT ts_ms, mark_price
            FROM mark_prices
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
            ORDER BY ts_ms ASC
            """,
            (symbol, int(start_ms), int(end_ms)),
        ).fetchall()
    ]


def mark_rows_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `mark_rows`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter, same as
    `book_at_v2` above.

    Boundary mapping: `mark_rows`'s oracle is INCLUSIVE on BOTH bounds
    (`ts_ms>=start_ms AND ts_ms<=end_ms`); `research_reader.plan_read`/
    `execute_read` use a half-open `[start_ms, end_ms)` window. Only the
    UPPER bound needs adjusting to reproduce the oracle exactly for
    integer ts_ms: `end_ms+1` turns the reader's exclusive upper bound
    into an inclusive one. The LOWER bound is passed through UNCHANGED --
    `start_ms` is already the reader's own inclusive lower bound, so no
    `+1` shift is applied here. (This mirrors the identical
    `_mark_prices_range`/`_mark_prices_range_v2` pattern in
    `tools/micro_edge_smoke.py`, Range-Read V4, which has the exact same
    inclusive/inclusive oracle shape. It is NOT the different `+1`-on-the-
    lower-bound mapping used for a different, EXCLUSIVE-lower oracle
    elsewhere in this migration series -- that mapping does not apply to
    this file's inclusive-lower oracle.)"""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(price)) for ts, price in result.iter_rows()]


def simulate_counterfactual(
    con: sqlite3.Connection,
    trade: dict[str, Any],
    now_ms: int,
    max_book_staleness_sec: float,
    *,
    root,
    source_db_path=None,
) -> dict[str, Any] | None:
    # `con` is no longer read inside this function body -- both DB reads
    # (book_at_v2 for the ASOF quote, mark_rows_v2 for the mark_prices
    # range) are now reader-backed via `root`/`source_db_path`. The
    # parameter is kept, unused, to avoid a call-site signature change in
    # `main()` (which still opens `con` for its own lifecycle) -- out of
    # scope for this range-read migration gate.
    rule = trade.get("rule") or {}
    symbol = str(trade.get("symbol") or rule.get("symbol") or "ETHUSDT")
    direction = str(trade.get("direction") or rule.get("direction") or "LONG").upper()
    entry_ts_ms = int(trade.get("entry_ts_ms") or trade.get("signal_ts_ms") or 0)
    entry_price = safe_float(trade.get("entry_price") or trade.get("entry_reference_price"))
    if not entry_ts_ms or entry_price <= 0:
        return None

    tp_price = safe_float(trade.get("tp_price"))
    sl_price = safe_float(trade.get("sl_price"))
    be_trigger_price = safe_float(trade.get("be_trigger_price"))
    if tp_price <= 0:
        tp_price = entry_price * (1.0 + safe_float(rule.get("tp_bps")) / 10000.0)
    if sl_price <= 0:
        sl_price = entry_price * (1.0 - safe_float(rule.get("sl_bps")) / 10000.0)
    if be_trigger_price <= 0:
        be_trigger_price = entry_price * (1.0 + safe_float(rule.get("be_trigger_bps")) / 10000.0)

    horizon_ms = int(safe_float(rule.get("max_horizon_sec"), 3600.0) * 1000)
    planned_end_ms = entry_ts_ms + horizon_ms
    end_ms = min(planned_end_ms, now_ms)
    rows = mark_rows_v2(root, symbol, entry_ts_ms, end_ms, source_db_path=source_db_path)
    if not rows:
        return None

    be_active = False
    exit_reason = "TIME"
    exit_ts_ms = rows[-1][0]
    exit_mark = rows[-1][1]
    mfe_bps = -1e18
    mae_bps = 1e18

    for ts_ms, mark in rows:
        move_bps = signed_bps(entry_price, mark, direction)
        mfe_bps = max(mfe_bps, move_bps)
        mae_bps = min(mae_bps, move_bps)

        if direction == "SHORT":
            if not be_active and mark <= be_trigger_price:
                be_active = True
            if mark <= tp_price:
                exit_reason = "TP"
                exit_ts_ms = ts_ms
                exit_mark = mark
                break
            active_stop = entry_price if be_active else sl_price
            if mark >= active_stop:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = ts_ms
                exit_mark = mark
                break
        else:
            if not be_active and mark >= be_trigger_price:
                be_active = True
            if mark >= tp_price:
                exit_reason = "TP"
                exit_ts_ms = ts_ms
                exit_mark = mark
                break
            active_stop = entry_price if be_active else sl_price
            if mark <= active_stop:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = ts_ms
                exit_mark = mark
                break

    quote = book_at_v2(root, symbol, exit_ts_ms, int(max_book_staleness_sec * 1000), source_db_path=source_db_path)
    if quote:
        exit_fill = quote["ask"] if direction == "SHORT" else quote["bid"]
        fill_source = "BOOK_TICKER"
    else:
        exit_fill = exit_mark
        fill_source = "MARK_FALLBACK"

    gross_bps = signed_bps(entry_price, exit_mark, direction)
    executable_bps = signed_bps(entry_price, exit_fill, direction)
    fee_bps = safe_float(rule.get("taker_fee_bps"), 4.0) * 2.0
    net_bps = executable_bps - fee_bps

    return {
        "trade_id": trade.get("trade_id"),
        "rule_name": rule.get("name"),
        "symbol": symbol,
        "direction": direction,
        "signal_ts_ms": trade.get("signal_ts_ms"),
        "signal_ts_utc": trade.get("signal_ts_utc") or utc_iso(trade.get("signal_ts_ms")),
        "entry_ts_ms": entry_ts_ms,
        "entry_ts_utc": trade.get("entry_ts_utc") or utc_iso(entry_ts_ms),
        "entry_price": entry_price,
        "exit_ts_ms": exit_ts_ms,
        "exit_ts_utc": utc_iso(exit_ts_ms),
        "exit_reason": exit_reason,
        "exit_mark": exit_mark,
        "exit_fill": exit_fill,
        "exit_fill_source": fill_source,
        "gross_bps": gross_bps,
        "executable_bps": executable_bps,
        "fee_bps": fee_bps,
        "net_bps": net_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "be_active": be_active,
        "complete_horizon": planned_end_ms <= now_ms or exit_reason != "TIME",
        "liq_total_notional": safe_float((trade.get("signal") or {}).get("liq_total_notional")),
        "liq_count": int((trade.get("signal") or {}).get("liq_count") or 0),
        "regime": trade.get("regime") or {},
    }


def render_table(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    out = [
        "| " + " | ".join(str(x) for x in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate S34 REGIME_FILTER skipped signals as shadow counterfactuals.")
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--trades-json", type=Path, default=TRADES_PATH)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--max-book-staleness-sec", type=float, default=5.0)
    args = parser.parse_args()

    now_ms = int(time.time() * 1000)
    cutoff_ms = int(now_ms - float(args.hours) * 3600 * 1000)
    payload = json.loads(args.trades_json.read_text(encoding="utf-8"))
    trades = payload.get("trades", [])
    recent = [t for t in trades if int(t.get("signal_ts_ms") or 0) >= cutoff_ms]
    regime_skips = [
        t
        for t in recent
        if t.get("status") == "SKIPPED"
        and (t.get("risk_gate_reason") or t.get("exit_reason")) == "REGIME_FILTER"
    ]

    root, _ = PR.resolve_production_root()
    con = sqlite3.connect(DB_URI, uri=True)
    simulations = [
        sim
        for trade in regime_skips
        if (sim := simulate_counterfactual(con, trade, now_ms, args.max_book_staleness_sec,
                                            root=root, source_db_path=SOURCE_DB_PATH)) is not None
    ]
    con.close()

    complete = [row for row in simulations if row["complete_horizon"]]
    incomplete = [row for row in simulations if not row["complete_horizon"]]
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_rule[str(row["rule_name"])].append(row)

    fail_combos = Counter()
    for trade in regime_skips:
        checks = ((trade.get("regime") or {}).get("checks") or {})
        failed = tuple(key for key, ok in checks.items() if not ok)
        fail_combos[failed] += 1

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_hours": float(args.hours),
        "cutoff_ts_ms": cutoff_ms,
        "cutoff_utc": utc_iso(cutoff_ms),
        "recent_trials": len(recent),
        "recent_closed": sum(1 for t in recent if t.get("status") == "CLOSED"),
        "regime_filter_skips": len(regime_skips),
        "simulated": len(simulations),
        "complete": len(complete),
        "incomplete": len(incomplete),
        "summary_complete": summarize(complete),
        "summary_all": summarize(simulations),
        "by_rule_complete": {rule: summarize(rows) for rule, rows in sorted(by_rule.items())},
        "regime_fail_combos": [
            {"count": count, "failed_checks": list(combo)} for combo, count in fail_combos.most_common()
        ],
        "top_counterfactuals_complete": sorted(complete, key=lambda r: float(r["net_bps"]), reverse=True)[:15],
        "worst_counterfactuals_complete": sorted(complete, key=lambda r: float(r["net_bps"]))[:15],
        "simulations": simulations,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md: list[str] = []
    md.append("# S34 REGIME_FILTER Shadow Evaluation")
    md.append("")
    md.append(f"- generated_at_utc: `{result['generated_at_utc']}`")
    md.append(f"- window_hours: `{args.hours}`")
    md.append(f"- cutoff_utc: `{result['cutoff_utc']}`")
    md.append(f"- recent_trials: `{len(recent)}`")
    md.append(f"- recent_closed: `{result['recent_closed']}`")
    md.append(f"- regime_filter_skips: `{len(regime_skips)}`")
    md.append(f"- simulated: `{len(simulations)}`")
    md.append(f"- complete_horizon: `{len(complete)}`")
    md.append(f"- incomplete_horizon: `{len(incomplete)}`")
    md.append("")
    md.append("## Complete-Horizon Summary")
    s = result["summary_complete"]
    md += render_table(
        [
            ["N", "cum net bps", "mean", "median", "win rate", "exits"],
            [
                s["n"],
                fmt(s["cum_net_bps"]),
                fmt(s["mean_net_bps"]),
                fmt(s["median_net_bps"]),
                fmt(None if s["win_rate"] is None else s["win_rate"] * 100.0) + "%",
                json.dumps(s["exit_counts"], sort_keys=True),
            ],
        ]
    )
    md.append("")
    md.append("## By Rule")
    rows = [["rule", "N", "cum net bps", "mean", "median", "win rate", "exits"]]
    for rule, summary in result["by_rule_complete"].items():
        rows.append(
            [
                rule,
                summary["n"],
                fmt(summary["cum_net_bps"]),
                fmt(summary["mean_net_bps"]),
                fmt(summary["median_net_bps"]),
                fmt(None if summary["win_rate"] is None else summary["win_rate"] * 100.0) + "%",
                json.dumps(summary["exit_counts"], sort_keys=True),
            ]
        )
    md += render_table(rows)
    md.append("")
    md.append("## Regime Fail Combos")
    rows = [["count", "failed checks"]]
    for item in result["regime_fail_combos"]:
        rows.append([item["count"], ", ".join(item["failed_checks"])])
    md += render_table(rows)
    md.append("")
    md.append("## Top Counterfactuals")
    rows = [["trade", "rule", "signal utc", "net bps", "exit", "MFE", "MAE", "liq notional"]]
    for row in result["top_counterfactuals_complete"][:10]:
        rows.append(
            [
                row["trade_id"],
                row["rule_name"],
                row["signal_ts_utc"],
                fmt(row["net_bps"]),
                row["exit_reason"],
                fmt(row["mfe_bps"]),
                fmt(row["mae_bps"]),
                fmt(row["liq_total_notional"], 0),
            ]
        )
    md += render_table(rows)
    md.append("")
    md.append("## Worst Counterfactuals")
    rows = [["trade", "rule", "signal utc", "net bps", "exit", "MFE", "MAE", "liq notional"]]
    for row in result["worst_counterfactuals_complete"][:10]:
        rows.append(
            [
                row["trade_id"],
                row["rule_name"],
                row["signal_ts_utc"],
                fmt(row["net_bps"]),
                row["exit_reason"],
                fmt(row["mfe_bps"]),
                fmt(row["mae_bps"]),
                fmt(row["liq_total_notional"], 0),
            ]
        )
    md += render_table(rows)
    md.append("")
    md.append("## Interpretation Guardrail")
    md.append("")
    md.append(
        "This is a shadow/counterfactual diagnostic only. It does not alter the live runner, "
        "does not count toward the pre-registered sample, and should not be used to retune gates without a separate OOS/real-fill validation."
    )
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(json.dumps({k: result[k] for k in ["recent_trials", "recent_closed", "regime_filter_skips", "complete", "incomplete", "summary_complete"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
