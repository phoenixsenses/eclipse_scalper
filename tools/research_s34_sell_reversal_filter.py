import datetime as dt
import itertools
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
DETAILS_JSON = Path("reports/research/s34/S34_SELL_LIQ_REVERSAL_LONG_2026-06-07_15.json")
OUT_JSON = Path("reports/research/s34/S34_SELL_REVERSAL_FILTER_SWEEP_2026-06-07_15.json")
OUT_MD = Path("reports/research/s34/S34_SELL_REVERSAL_FILTER_SWEEP_2026-06-07_15.md")
BASE_NAME = "SELL_REVERSAL_LONG 200000 TP40 DELAY300s"


def day_start_ms(ts_ms: int) -> int:
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).date()
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    # OUT-OF-SCOPE for RANGE-READ V2: this is an ASOF-style point lookup
    # (ORDER BY ts_ms ASC/DESC LIMIT 1) -- it belongs to the ASOF track's
    # `lookup_latest_at_or_before` primitive, not the range-read helper this
    # gate migrates. (The static inventory scanner reported desc_limit1=0
    # only because `ORDER BY ts_ms {order}` is parameterized, not a literal
    # `DESC`.) Left on direct SQL deliberately.
    op = "<=" if before else ">="
    order = "desc" if before else "asc"
    return con.execute(
        f"""
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms {op} ?
        order by ts_ms {order}
        limit 1
        """,
        (symbol, ts_ms),
    ).fetchone()


def ret_bps(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int):
    p0 = mark_at(con, symbol, start_ms, before=False)
    p1 = mark_at(con, symbol, end_ms, before=False)
    if not p0 or not p1 or not p0[1]:
        return None
    return (p1[1] - p0[1]) / p0[1] * 10000.0


def _day_high_low(con: sqlite3.Connection, start_ms: int, end_ms: int):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_day_high_low_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V2). Returns (max, min) mark_price over the inclusive
    window; (None, None) for an empty window (SQL MAX/MIN over no rows)."""
    return con.execute(
        """
        select max(mark_price), min(mark_price)
        from mark_prices
        where symbol='ETHUSDT' and ts_ms>=? and ts_ms<=?
        """,
        (start_ms, end_ms),
    ).fetchone()


def _day_high_low_v2(root, start_ms: int, end_ms: int, source_db_path=None):
    """Reader-backed replacement for `_day_high_low`, via `plan_read`/
    `execute_read`. Symbol hardcoded 'ETHUSDT' (as in the oracle SQL).
    Inclusive upper bound reproduced with `end_ms+1` (exact for integer
    ts_ms). `mark_price` is a non-nullable column, but SQL MAX/MIN ignore
    NULLs anyway, so None values are skipped here for exact SQL parity; an
    empty window yields (None, None), matching SQL's MAX/MIN over no rows."""
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=source_db_path)
    hi = lo = None
    for (mp,) in result.iter_rows():
        if mp is None:
            continue
        v = float(mp)
        hi = v if hi is None or v > hi else hi
        lo = v if lo is None or v < lo else lo
    return (hi, lo)


def _day_agg_count(con: sqlite3.Connection, start_ms: int, end_ms: int) -> int:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_day_agg_count_v2` (same gate). COUNT(*) of ETHUSDT agg_trades over the
    inclusive window."""
    return con.execute(
        """
        select count(*)
        from agg_trades
        where symbol='ETHUSDT' and ts_ms>=? and ts_ms<=?
        """,
        (start_ms, end_ms),
    ).fetchone()[0]


def _day_agg_count_v2(root, start_ms: int, end_ms: int, source_db_path=None) -> int:
    """Reader-backed replacement for `_day_agg_count`, via `plan_read`/
    `execute_read`. Symbol hardcoded 'ETHUSDT'. Inclusive upper bound
    reproduced with `end_ms+1`; the streamed row count equals COUNT(*)."""
    plan = RR.plan_read(root, table="agg_trades", symbol="ETHUSDT", start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms",), source_db_path=source_db_path)
    return sum(1 for _ in result.iter_rows())


def day_so_far(con: sqlite3.Connection, ts_ms: int, root, source_db_path=None):
    start = day_start_ms(ts_ms)
    cur = mark_at(con, "ETHUSDT", ts_ms, before=True)
    open_row = mark_at(con, "ETHUSDT", start, before=False)
    if not cur or not open_row or not open_row[1]:
        return {}
    # Migrated to the reader (allowlisted mark_prices / agg_trades):
    high, low = _day_high_low_v2(root, start, ts_ms, source_db_path=source_db_path)
    # OUT-OF-ALLOWLIST (liquidations has no archive partition / reader
    # support): left on direct SQL deliberately.
    sell_liq = con.execute(
        """
        select coalesce(sum(notional), 0)
        from liquidations
        where symbol='ETHUSDT' and side='SELL' and ts_ms>=? and ts_ms<=?
        """,
        (start, ts_ms),
    ).fetchone()[0]
    buy_liq = con.execute(
        """
        select coalesce(sum(notional), 0)
        from liquidations
        where symbol='ETHUSDT' and side='BUY' and ts_ms>=? and ts_ms<=?
        """,
        (start, ts_ms),
    ).fetchone()[0]
    agg_count = _day_agg_count_v2(root, start, ts_ms, source_db_path=source_db_path)
    return {
        "day_trend_bps": (cur[1] - open_row[1]) / open_row[1] * 10000.0,
        "day_range_bps": (high - low) / low * 10000.0 if high and low else None,
        "day_sell_liq_m": sell_liq / 1_000_000.0,
        "day_buy_liq_m": buy_liq / 1_000_000.0,
        "day_agg_count": agg_count,
    }


def enrich(con: sqlite3.Connection, trades: list[dict], root, source_db_path=None):
    out = []
    for trade in trades:
        ts = trade["signal_ts_ms"]
        row = dict(trade)
        row.update(day_so_far(con, ts, root, source_db_path=source_db_path))
        row["eth_wait5_bps"] = ret_bps(con, "ETHUSDT", ts, ts + 300_000)
        row["btc_wait5_bps"] = ret_bps(con, "BTCUSDT", ts, ts + 300_000)
        out.append(row)
    return out


def avg(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def med(values):
    values = [v for v in values if v is not None]
    return None if not values else statistics.median(values)


def summarize(name: str, rows: list[dict]):
    vals = [r["net_bps"] for r in rows]
    if not vals:
        return None
    days = sorted({r["day"] for r in rows})
    by_day = {}
    for day in days:
        day_vals = [r["net_bps"] for r in rows if r["day"] == day]
        by_day[day] = {
            "n": len(day_vals),
            "cum": sum(day_vals),
            "mean": avg(day_vals),
            "median": med(day_vals),
        }
    return {
        "filter": name,
        "n": len(rows),
        "days": len(days),
        "mean": avg(vals),
        "median": med(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "tp": sum(r["exit_reason"] == "TP" for r in rows),
        "sl": sum(r["exit_reason"] == "SL" for r in rows),
        "be": sum(r["exit_reason"] == "BE" for r in rows),
        "time": sum(r["exit_reason"] == "TIME" for r in rows),
        "by_day": by_day,
    }


def passes(row: dict, feature: str, op: str, threshold: float):
    value = row.get(feature)
    if value is None:
        return False
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(op)


def main():
    payload = json.loads(DETAILS_JSON.read_text(encoding="utf-8"))
    base = payload["details"][BASE_NAME]
    root, _ = PR.resolve_production_root()
    # `con` stays for the still-direct ASOF (mark_at) + out-of-allowlist
    # (liquidations) queries; the mark_prices/agg_trades range aggregates
    # moved to the reader (via SOURCE_DB_PATH).
    con = sqlite3.connect(DB, uri=True, timeout=3)
    rows = enrich(con, base, root, source_db_path=SOURCE_DB_PATH)

    predicates = []
    feature_grid = {
        "btc_pre15_bps": [-80, -60, -40, -20, 0],
        "eth_pre5_bps": [-100, -80, -60, -40, -20, 0],
        "eth_wait5_bps": [-80, -60, -40, -20, 0, 20],
        "btc_wait5_bps": [-40, -20, 0, 20],
        "day_trend_bps": [-200, 0, 100, 200, 300, 400],
        "day_range_bps": [200, 300, 400, 500, 600],
        "day_sell_liq_m": [2, 5, 10, 20],
        "day_buy_liq_m": [2, 5, 10, 20],
        "day_agg_count": [100_000, 250_000, 500_000, 750_000],
    }
    for feature, thresholds in feature_grid.items():
        for threshold in thresholds:
            predicates.append((feature, ">=", threshold))
            predicates.append((feature, "<=", threshold))

    results = []
    base_summary = summarize("BASE_NO_FILTER", rows)
    results.append(base_summary)

    for pred in predicates:
        feature, op, threshold = pred
        selected = [row for row in rows if passes(row, feature, op, threshold)]
        summary = summarize(f"{feature} {op} {threshold}", selected)
        if summary and summary["n"] >= 8 and summary["days"] >= 3:
            results.append(summary)

    # Simple two-predicate combinations, constrained to avoid overfitting tiny pockets.
    for left, right in itertools.combinations(predicates, 2):
        selected = [
            row
            for row in rows
            if passes(row, left[0], left[1], left[2]) and passes(row, right[0], right[1], right[2])
        ]
        summary = summarize(
            f"{left[0]} {left[1]} {left[2]} AND {right[0]} {right[1]} {right[2]}",
            selected,
        )
        if summary and summary["n"] >= 8 and summary["days"] >= 3:
            results.append(summary)

    results = sorted(
        results,
        key=lambda r: (r["median"], r["mean"], r["cum"], r["n"]),
        reverse=True,
    )

    OUT_JSON.write_text(json.dumps({"base": rows, "results": results}, indent=2), encoding="utf-8")

    def fnum(v):
        return "n/a" if v is None else f"{v:+.2f}"

    lines = [
        "# S34 SELL Reversal LONG Regime Filter Sweep",
        "",
        "Date: 2026-06-16",
        "",
        f"Base candidate: `{BASE_NAME}`",
        "",
        "Goal: find a no-lookahead discriminator that keeps the delayed SELL-liquidation bounce behavior while reducing the 2026-06-07 failure mode. Features use only data available at signal time or during the 300s wait before entry.",
        "",
        "## Top Filters",
        "",
        "| Rank | Filter | N | Days | Mean | Median | Cum | WR | TP/SL/BE/TIME | 06-07 Cum | 06-11 Cum | 06-14 Cum | 06-15 Cum |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(results[:20], 1):
        by_day = row["by_day"]
        lines.append(
            f"| {idx} | {row['filter']} | {row['n']} | {row['days']} | {fnum(row['mean'])} | "
            f"{fnum(row['median'])} | {fnum(row['cum'])} | {row['wr']*100:.1f}% | "
            f"{row['tp']}/{row['sl']}/{row['be']}/{row['time']} | "
            f"{fnum(by_day.get('2026-06-07', {}).get('cum'))} | "
            f"{fnum(by_day.get('2026-06-11', {}).get('cum'))} | "
            f"{fnum(by_day.get('2026-06-14', {}).get('cum'))} | "
            f"{fnum(by_day.get('2026-06-15', {}).get('cum'))} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This sweep is exploratory. A usable filter must reduce the 2026-06-07 loss without collapsing N or relying on post-entry information. If no broad, interpretable filter clears that bar, the reversal-long idea stays research-only.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(results[:8], indent=2))
    con.close()


if __name__ == "__main__":
    main()
