import datetime as dt
import json
import sqlite3
import statistics
from pathlib import Path

from ami.storage import production as PR
from ami.storage import research_reader as RR

DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
DAYS = ["2026-06-07", "2026-06-11", "2026-06-14", "2026-06-15"]
OUT_JSON = Path("reports/research/s34/S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15.json")
OUT_MD = Path("reports/research/s34/S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15.md")


def day_ms(day: str) -> int:
    return int(dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp() * 1000)


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    # OUT-OF-SCOPE for RANGE-READ V3: ASOF-style point lookup (ORDER BY
    # ts_ms ASC/DESC LIMIT 1) -- belongs to the ASOF track's
    # lookup_latest_at_or_before, not the range-read helper this gate
    # migrates. Left on direct SQL deliberately.
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


def events(con: sqlite3.Connection, day: str, threshold: int):
    # OUT-OF-SCOPE for RANGE-READ V3: `liquidations` is an out-of-allowlist
    # table (no archive partition / reader support). Left on direct SQL.
    start_ms = day_ms(day)
    end_ms = start_ms + 86_400_000
    return con.execute(
        """
        select ts_ms, notional
        from liquidations
        where symbol='ETHUSDT'
          and side='BUY'
          and notional>=?
          and ts_ms>=?
          and ts_ms<?
        order by ts_ms
        """,
        (threshold, start_ms, end_ms),
    ).fetchall()


def _horizon_marks(con: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_horizon_marks_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V3). No longer called by `sim_short`; the reader-
    backed path is used instead."""
    return con.execute(
        """
        select ts_ms, mark_price
        from mark_prices
        where symbol='ETHUSDT' and ts_ms>=? and ts_ms<=?
        order by ts_ms
        """,
        (start_ms, end_ms),
    ).fetchall()


def _horizon_marks_v2(root, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `_horizon_marks`, via `plan_read`/
    `execute_read`. Symbol hardcoded 'ETHUSDT' (as in the oracle SQL).
    Inclusive upper bound reproduced with `end_ms+1` (exact for integer
    ts_ms). Streams in canonical (ts_ms ASC, id ASC) order -- a refinement
    of the oracle's `ORDER BY ts_ms` that yields an identical ts_ms
    sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(mp)) for ts, mp in result.iter_rows()]


def sim_short(
    con: sqlite3.Connection,
    signal_ts_ms: int,
    threshold: int,
    tp_bps: int,
    *,
    root,
    source_db_path=None,
    entry_delay_sec: int = 0,
    sl_bps: int = 40,
    be_bps: int = 30,
    max_horizon_sec: int = 3600,
):
    entry_row = mark_at(con, "ETHUSDT", signal_ts_ms + entry_delay_sec * 1000, before=False)
    if not entry_row:
        return None
    entry_ts_ms, entry_price = entry_row
    rows = _horizon_marks_v2(root, entry_ts_ms, entry_ts_ms + max_horizon_sec * 1000, source_db_path=source_db_path)
    if not rows:
        return None

    be_active = False
    exit_reason = "TIME"
    exit_ts_ms, exit_price = rows[-1]
    mfe_bps = -1e9
    mae_bps = 1e9
    time_to_mfe_sec = 0.0
    tp_touch = False
    sl_touch = False
    be_hit = False

    for ts_ms, price in rows:
        ret = (entry_price - price) / entry_price * 10000.0
        if ret > mfe_bps:
            mfe_bps = ret
            time_to_mfe_sec = (ts_ms - entry_ts_ms) / 1000.0
        if ret < mae_bps:
            mae_bps = ret
        if ret >= tp_bps:
            tp_touch = True
        if ret <= -sl_bps:
            sl_touch = True
        if not be_active and ret >= be_bps:
            be_active = True
            be_hit = True
        if ret >= tp_bps:
            exit_reason = "TP"
            exit_ts_ms = ts_ms
            exit_price = price
            break
        if ret <= -sl_bps:
            exit_reason = "SL"
            exit_ts_ms = ts_ms
            exit_price = price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts_ms = ts_ms
            exit_price = price
            break

    gross_bps = (entry_price - exit_price) / entry_price * 10000.0
    return {
        "signal_ts_ms": signal_ts_ms,
        "day": dt.datetime.fromtimestamp(signal_ts_ms / 1000, dt.timezone.utc).date().isoformat(),
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "exit_ts_ms": exit_ts_ms,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - 8.0,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "time_to_mfe_sec": time_to_mfe_sec,
        "tp_touch": tp_touch,
        "sl_touch": sl_touch,
        "be_hit": be_hit,
        "threshold": threshold,
        "tp_bps": tp_bps,
        "entry_delay_sec": entry_delay_sec,
        "btc_pre15_bps": ret_bps(con, "BTCUSDT", signal_ts_ms - 900_000, signal_ts_ms),
        "eth_pre5_bps": ret_bps(con, "ETHUSDT", signal_ts_ms - 300_000, signal_ts_ms),
        "eth_post1_bps": ret_bps(con, "ETHUSDT", signal_ts_ms, signal_ts_ms + 60_000),
        "eth_post2_bps": ret_bps(con, "ETHUSDT", signal_ts_ms, signal_ts_ms + 120_000),
    }


def avg(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def med(values):
    values = [v for v in values if v is not None]
    return None if not values else statistics.median(values)


def summarize(name: str, trades: list[dict]):
    values = [t["net_bps"] for t in trades]
    if not values:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(trades),
        "days": len({t["day"] for t in trades}),
        "mean_net_bps": avg(values),
        "median_net_bps": med(values),
        "cum_net_bps": sum(values),
        "wr": sum(v > 0 for v in values) / len(values),
        "mean_mfe_bps": avg([t["mfe_bps"] for t in trades]),
        "median_mfe_bps": med([t["mfe_bps"] for t in trades]),
        "mean_mae_bps": avg([t["mae_bps"] for t in trades]),
        "median_mae_bps": med([t["mae_bps"] for t in trades]),
        "tp_touch_rate": sum(t["tp_touch"] for t in trades) / len(trades),
        "sl_touch_rate": sum(t["sl_touch"] for t in trades) / len(trades),
        "be_hit_rate": sum(t["be_hit"] for t in trades) / len(trades),
        "median_btc_pre15_bps": med([t["btc_pre15_bps"] for t in trades]),
        "median_eth_pre5_bps": med([t["eth_pre5_bps"] for t in trades]),
        "median_eth_post1_bps": med([t["eth_post1_bps"] for t in trades]),
        "median_eth_post2_bps": med([t["eth_post2_bps"] for t in trades]),
        "exits": {k: sum(t["exit_reason"] == k for t in trades) for k in ["TP", "BE", "SL", "TIME"]},
    }


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.2f}"
    return str(value)


def pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main():
    # `con` stays for the still-direct ASOF (mark_at/ret_bps) and
    # out-of-allowlist (events/liquidations) reads; the mark_prices horizon
    # window moved to the reader (via `root`/SOURCE_DB_PATH).
    con = sqlite3.connect(DB, uri=True, timeout=3)
    root, _ = PR.resolve_production_root()
    rows = []
    details = {}

    for threshold in [50_000, 100_000, 200_000, 300_000, 500_000]:
        for tp_bps in [40, 60, 80, 120]:
            for delay_sec in [0, 60, 120, 300, 600]:
                trades = []
                for day in DAYS:
                    for ts_ms, notional in events(con, day, threshold):
                        trade = sim_short(con, ts_ms, threshold, tp_bps, root=root, source_db_path=SOURCE_DB_PATH,
                                           entry_delay_sec=delay_sec)
                        if trade:
                            trade["notional"] = notional
                            trades.append(trade)
                name = f"BUY_REVERSAL_SHORT {threshold} TP{tp_bps} DELAY{delay_sec}s"
                summary = summarize(name, trades)
                rows.append(summary)
                details[name] = trades

    rows = sorted(rows, key=lambda r: (r.get("median_net_bps", -999), r.get("mean_net_bps", -999)), reverse=True)
    OUT_JSON.write_text(json.dumps({"summaries": rows, "details": details}, indent=2), encoding="utf-8")

    lines = [
        "# S34 BUY Liquidation Reversal Short Replay",
        "",
        "Date: 2026-06-16",
        "",
        "Question: besides the active BUY-liq momentum LONG, is there an exhaustion/reversal SHORT after large BUY liquidations?",
        "",
        "Model: simplified mark-price replay, flat 8 bps round trip, no real bid/ask live fill parity.",
        "",
        "## Top Results",
        "",
        "| Rank | Candidate | N | Days | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | TP Touch | SL Touch | BE Hit | Exits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows[:25], 1):
        lines.append(
            f"| {idx} | {row['name']} | {row.get('n', 0)} | {row.get('days', 0)} | "
            f"{fmt(row.get('mean_net_bps'))} | {fmt(row.get('median_net_bps'))} | "
            f"{fmt(row.get('cum_net_bps'))} | {pct(row.get('wr'))} | "
            f"{fmt(row.get('mean_mfe_bps'))} | {fmt(row.get('mean_mae_bps'))} | "
            f"{pct(row.get('tp_touch_rate'))} | {pct(row.get('sl_touch_rate'))} | "
            f"{pct(row.get('be_hit_rate'))} | {row.get('exits', {})} |"
        )

    best = rows[0]
    best_details = details[best["name"]]
    lines.extend(["", "## Day Split For Top Candidate", ""])
    lines.append(f"Candidate: `{best['name']}`")
    lines.extend(["", "| Day | N | Cum Net | Mean Net | Median Net |", "|---|---:|---:|---:|---:|"])
    for day in DAYS:
        vals = [t["net_bps"] for t in best_details if t["day"] == day]
        if vals:
            lines.append(f"| {day} | {len(vals)} | {sum(vals):+.2f} | {avg(vals):+.2f} | {med(vals):+.2f} |")
        else:
            lines.append(f"| {day} | 0 | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This is research-only. A BUY-liq reversal short must beat the active BUY-liq momentum long on stability and must not be a delayed lookback artifact. Do not add it to live paper unless the result is broad, interpretable, and survives no-lookahead filtering.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(rows[:10], indent=2))
    con.close()


if __name__ == "__main__":
    main()
