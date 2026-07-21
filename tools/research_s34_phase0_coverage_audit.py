import datetime as dt
import json
import sqlite3
from pathlib import Path


DB = "file:data/microstructure.db?mode=ro"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
OUT_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE0_COVERAGE_2026-06-16.json")
OUT_MD = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE0_COVERAGE_2026-06-16.md")


def iso(ts_ms):
    if ts_ms is None:
        return None
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).isoformat()


def days_between(min_ts, max_ts):
    if min_ts is None or max_ts is None:
        return 0.0
    return (max_ts - min_ts) / 86_400_000.0


def table_exists(con, table):
    return (
        con.execute(
            "select 1 from sqlite_master where type='table' and name=?",
            (table,),
        ).fetchone()
        is not None
    )


def coverage(con, table, symbol, side=None, exact_count=True):
    if side is None:
        if exact_count:
            row = con.execute(
                f"select count(*), min(ts_ms), max(ts_ms) from {table} where symbol=?",
                (symbol,),
            ).fetchone()
        else:
            first = con.execute(
                f"select ts_ms from {table} where symbol=? order by ts_ms asc limit 1",
                (symbol,),
            ).fetchone()
            last = con.execute(
                f"select ts_ms from {table} where symbol=? order by ts_ms desc limit 1",
                (symbol,),
            ).fetchone()
            row = (None, first[0] if first else None, last[0] if last else None)
    else:
        row = con.execute(
            f"select count(*), min(ts_ms), max(ts_ms) from {table} where symbol=? and side=?",
            (symbol, side),
        ).fetchone()
    count, min_ts, max_ts = row
    return {
        "count": count,
        "count_exact": exact_count,
        "min_ts_ms": min_ts,
        "max_ts_ms": max_ts,
        "first_utc": iso(min_ts),
        "last_utc": iso(max_ts),
        "days": days_between(min_ts, max_ts),
    }


def daily_liq(con, symbol):
    rows = con.execute(
        """
        select date(ts_ms/1000, 'unixepoch') as day,
               side,
               count(*) as rows,
               sum(notional) as notional
        from liquidations
        where symbol=?
        group by day, side
        order by day, side
        """,
        (symbol,),
    ).fetchall()
    return [
        {"day": day, "side": side, "rows": rows, "notional": notional or 0.0}
        for day, side, rows, notional in rows
    ]


def latest_gap_minutes(max_ts):
    if max_ts is None:
        return None
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    return (now_ms - max_ts) / 60000.0


def main():
    con = sqlite3.connect(DB, uri=True, timeout=10)
    con.execute("pragma query_only=1")

    tables = [name for (name,) in con.execute("select name from sqlite_master where type='table' order by name")]
    result = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "symbols": {}, "tables": tables}

    for symbol in SYMBOLS:
        item = {}
        if table_exists(con, "liquidations"):
            item["liquidations_total"] = coverage(con, "liquidations", symbol)
            item["liquidations_buy"] = coverage(con, "liquidations", symbol, "BUY")
            item["liquidations_sell"] = coverage(con, "liquidations", symbol, "SELL")
            item["liquidations_daily"] = daily_liq(con, symbol)
            item["liq_minutes_since_last"] = latest_gap_minutes(item["liquidations_total"]["max_ts_ms"])
        if table_exists(con, "mark_prices"):
            item["mark_prices"] = coverage(con, "mark_prices", symbol, exact_count=False)
            item["mark_minutes_since_last"] = latest_gap_minutes(item["mark_prices"]["max_ts_ms"])
        if table_exists(con, "agg_trades"):
            item["agg_trades"] = coverage(con, "agg_trades", symbol, exact_count=False)
            item["agg_minutes_since_last"] = latest_gap_minutes(item["agg_trades"]["max_ts_ms"])
        if table_exists(con, "book_ticker"):
            item["book_ticker"] = coverage(con, "book_ticker", symbol, exact_count=False)
            item["book_minutes_since_last"] = latest_gap_minutes(item["book_ticker"]["max_ts_ms"])
        result["symbols"][symbol] = item

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Phase 0 Coverage Audit",
        "",
        f"Generated: {result['generated_at_utc']}",
        "",
        "Scope: read-only coverage audit over `data/microstructure.db`. No runner/config changes.",
        "",
        "## Symbol Coverage",
        "",
        "| Symbol | Liq Rows | BUY Liq | SELL Liq | Liq First | Liq Last | Liq Days | Min Since Last Liq | Mark First/Last | Agg First/Last | Book First/Last |",
        "|---|---:|---:|---:|---|---|---:|---:|---|---|---|",
    ]
    for symbol, item in result["symbols"].items():
        liq = item.get("liquidations_total", {})
        buy = item.get("liquidations_buy", {})
        sell = item.get("liquidations_sell", {})
        mark = item.get("mark_prices", {})
        agg = item.get("agg_trades", {})
        book = item.get("book_ticker", {})
        lines.append(
            f"| {symbol} | {liq.get('count', 0):,} | {buy.get('count', 0):,} | {sell.get('count', 0):,} | "
            f"{liq.get('first_utc') or 'n/a'} | {liq.get('last_utc') or 'n/a'} | "
            f"{liq.get('days', 0):.2f} | "
            f"{item.get('liq_minutes_since_last') if item.get('liq_minutes_since_last') is not None else 'n/a'} | "
            f"{mark.get('first_utc') or 'n/a'} / {mark.get('last_utc') or 'n/a'} | "
            f"{agg.get('first_utc') or 'n/a'} / {agg.get('last_utc') or 'n/a'} | "
            f"{book.get('first_utc') or 'n/a'} / {book.get('last_utc') or 'n/a'} |"
        )

    lines.extend(["", "## Daily Liquidation Rows", ""])
    for symbol, item in result["symbols"].items():
        lines.extend([f"### {symbol}", "", "| Day | BUY Rows | BUY Notional | SELL Rows | SELL Notional |", "|---|---:|---:|---:|---:|"])
        by_day = {}
        for row in item.get("liquidations_daily", []):
            by_day.setdefault(row["day"], {})[row["side"]] = row
        for day in sorted(by_day):
            buy = by_day[day].get("BUY", {"rows": 0, "notional": 0})
            sell = by_day[day].get("SELL", {"rows": 0, "notional": 0})
            lines.append(
                f"| {day} | {buy['rows']:,} | {buy['notional']/1_000_000:.2f}M | "
                f"{sell['rows']:,} | {sell['notional']/1_000_000:.2f}M |"
            )
        lines.append("")

    eth_liq = result["symbols"].get("ETHUSDT", {}).get("liquidations_total", {}).get("count", 0)
    sol_liq = result["symbols"].get("SOLUSDT", {}).get("liquidations_total", {}).get("count", 0)
    sol_ratio = (sol_liq / eth_liq) if eth_liq else 0.0
    lines.extend(
        [
            "## Phase 1 Scope Recommendation",
            "",
            f"SOL/ETH liquidation row ratio: {sol_ratio:.2%}.",
            "",
            "Recommendation:",
            "",
            "- Build the Feature Factory schema as symbol-parametric from the start.",
            "- Run Phase 1 extraction on ETH first, because ETH is the active S34 target and has the clearest existing evidence.",
            "- Include BTC and SOL in the Phase 0 coverage table and keep them as eligible symbols.",
            "- Promote SOL into Phase 1 extraction only if its liquidation coverage is not materially thinner than ETH and latest timestamps are fresh.",
            "",
            "Acceptance for Phase 0: coverage measured for BTC/ETH/SOL, latest timestamps known, and SOL inclusion decision made from measured rows rather than assumption.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({s: result["symbols"][s].get("liquidations_total", {}) for s in SYMBOLS}, indent=2))
    con.close()


if __name__ == "__main__":
    main()
