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
OUT_JSON = Path("reports/research/s34/S34_SELL_LIQ_PATH_QUALITY_2026-06-07_15.json")
OUT_MD = Path("reports/research/s34/S34_SELL_LIQ_PATH_QUALITY_2026-06-07_15.md")
RAW_REPLAY_JSON = Path("reports/research/s34/S34_SELL_LIQ_REPLAY_2026-06-07_15.json")


def day_ms(day: str) -> int:
    return int(dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp() * 1000)


def ts_iso(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).isoformat()


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    # OUT-OF-SCOPE for RANGE-READ V5: ASOF-style point lookup (ORDER BY
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


def _horizon_marks(con: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_horizon_marks_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V5). No longer called by `sim_short`; the reader-
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


def _horizon_marks_v2(root, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_horizon_marks`, via `plan_read`/
    `execute_read`. Symbol hardcoded 'ETHUSDT' (as in the oracle SQL).
    Inclusive upper bound reproduced with `end_ms+1` (exact for integer
    ts_ms). Streams in canonical (ts_ms ASC, id ASC) order -- a refinement
    of the oracle's `ORDER BY ts_ms` that yields an identical ts_ms
    sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol="ETHUSDT", start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return list(result.iter_rows())


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
    be_ts_ms = None
    tp_touch_ts_ms = None
    sl_touch_ts_ms = None
    first_positive_ts_ms = None
    first_neg20_ts_ms = None
    mfe_bps = -1e9
    mae_bps = 1e9
    time_to_mfe_sec = 0.0
    exit_reason = "TIME"
    exit_ts_ms, exit_price = rows[-1]

    for ts_ms, price in rows:
        ret = (entry_price - price) / entry_price * 10000.0
        if ret > mfe_bps:
            mfe_bps = ret
            time_to_mfe_sec = (ts_ms - entry_ts_ms) / 1000.0
        if ret < mae_bps:
            mae_bps = ret
        if first_positive_ts_ms is None and ret > 0:
            first_positive_ts_ms = ts_ms
        if first_neg20_ts_ms is None and ret <= -20:
            first_neg20_ts_ms = ts_ms
        if tp_touch_ts_ms is None and ret >= tp_bps:
            tp_touch_ts_ms = ts_ms
        if sl_touch_ts_ms is None and ret <= -sl_bps:
            sl_touch_ts_ms = ts_ms
        if not be_active and ret >= be_bps:
            be_active = True
            be_ts_ms = ts_ms
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
        "signal_utc": ts_iso(signal_ts_ms),
        "entry_ts_ms": entry_ts_ms,
        "entry_utc": ts_iso(entry_ts_ms),
        "entry_price": entry_price,
        "exit_ts_ms": exit_ts_ms,
        "exit_utc": ts_iso(exit_ts_ms),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - 8.0,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "time_to_mfe_sec": time_to_mfe_sec,
        "be_hit": be_ts_ms is not None,
        "be_ts_ms": be_ts_ms,
        "tp_touch": tp_touch_ts_ms is not None,
        "sl_touch": sl_touch_ts_ms is not None,
        "first_positive_sec": None
        if first_positive_ts_ms is None
        else (first_positive_ts_ms - entry_ts_ms) / 1000.0,
        "first_neg20_sec": None
        if first_neg20_ts_ms is None
        else (first_neg20_ts_ms - entry_ts_ms) / 1000.0,
        "btc_pre15_bps": ret_bps(con, "BTCUSDT", signal_ts_ms - 900_000, signal_ts_ms),
        "eth_pre5_bps": ret_bps(con, "ETHUSDT", signal_ts_ms - 300_000, signal_ts_ms),
        "eth_post1_bps": ret_bps(con, "ETHUSDT", signal_ts_ms, signal_ts_ms + 60_000),
        "eth_post2_bps": ret_bps(con, "ETHUSDT", signal_ts_ms, signal_ts_ms + 120_000),
        "threshold": threshold,
        "tp_bps": tp_bps,
        "entry_delay_sec": entry_delay_sec,
    }


def events(con: sqlite3.Connection, day: str, threshold: int):
    # OUT-OF-SCOPE for RANGE-READ V5: `liquidations` is an out-of-allowlist
    # table (no archive partition / reader support). Left on direct SQL.
    start_ms = day_ms(day)
    end_ms = start_ms + 86_400_000
    return con.execute(
        """
        select ts_ms, notional
        from liquidations
        where symbol='ETHUSDT'
          and side='SELL'
          and notional>=?
          and ts_ms>=?
          and ts_ms<?
        order by ts_ms
        """,
        (threshold, start_ms, end_ms),
    ).fetchall()


def avg(values):
    values = [x for x in values if x is not None]
    return None if not values else sum(values) / len(values)


def med(values):
    values = [x for x in values if x is not None]
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
        "be_hit_rate": sum(t["be_hit"] for t in trades) / len(trades),
        "tp_touch_rate": sum(t["tp_touch"] for t in trades) / len(trades),
        "sl_touch_rate": sum(t["sl_touch"] for t in trades) / len(trades),
        "avg_time_to_mfe_sec": avg([t["time_to_mfe_sec"] for t in trades]),
        "median_eth_post1_bps": med([t["eth_post1_bps"] for t in trades]),
        "median_eth_post2_bps": med([t["eth_post2_bps"] for t in trades]),
        "median_btc_pre15_bps": med([t["btc_pre15_bps"] for t in trades]),
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
    candidates = []

    raw_replay = json.loads(RAW_REPLAY_JSON.read_text(encoding="utf-8"))
    replay_trade_candidates = []
    for side, direction, threshold, tp_bps in [
        ("SELL", "SHORT", 200_000, 80),
        ("SELL", "SHORT", 200_000, 60),
        ("SELL", "SHORT", 200_000, 120),
        ("SELL", "SHORT", 100_000, 80),
        ("SELL", "SHORT", 50_000, 80),
    ]:
        trades = []
        for row in raw_replay["trades"]:
            if (
                row["side"] == side
                and row["direction"] == direction
                and row["threshold"] == threshold
                and row["tp"] == tp_bps
            ):
                signal_ts_ms = int(dt.datetime.fromisoformat(row["ts_utc"]).timestamp() * 1000)
                trade = sim_short(con, signal_ts_ms, threshold, tp_bps, root=root, source_db_path=SOURCE_DB_PATH, entry_delay_sec=0)
                if trade:
                    trade["net_bps"] = row["net_bps"]
                    trade["exit_reason"] = row["exit_reason"]
                    trade.update(
                        day=row["day"],
                        notional=row["liq_notional"],
                        variant=f"REPLAY_CLUSTER_{threshold}_TP{tp_bps}",
                        replay_net_bps=row["net_bps"],
                        replay_exit_reason=row["exit_reason"],
                    )
                    trades.append(trade)
        replay_trade_candidates.append(
            {"summary": summarize(f"REPLAY_CLUSTER {threshold} TP{tp_bps}", trades), "trades": trades}
        )

    for threshold, tp_bps in [(200_000, 80), (200_000, 60), (100_000, 40), (100_000, 80)]:
        raw = []
        confirm1 = []
        confirm2 = []
        for day in DAYS:
            for ts_ms, notional in events(con, day, threshold):
                trade = sim_short(con, ts_ms, threshold, tp_bps, root=root, source_db_path=SOURCE_DB_PATH, entry_delay_sec=0)
                if trade:
                    trade.update(day=day, notional=notional, variant=f"RAW_{threshold}_TP{tp_bps}")
                    raw.append(trade)

                post1 = ret_bps(con, "ETHUSDT", ts_ms, ts_ms + 60_000)
                if post1 is not None and post1 <= -5:
                    trade = sim_short(con, ts_ms, threshold, tp_bps, root=root, source_db_path=SOURCE_DB_PATH, entry_delay_sec=60)
                    if trade:
                        trade.update(
                            day=day,
                            notional=notional,
                            variant=f"CONFIRM1M_{threshold}_TP{tp_bps}",
                            confirm_bps=post1,
                        )
                        confirm1.append(trade)

                post2 = ret_bps(con, "ETHUSDT", ts_ms, ts_ms + 120_000)
                if post2 is not None and post2 <= -8:
                    trade = sim_short(con, ts_ms, threshold, tp_bps, root=root, source_db_path=SOURCE_DB_PATH, entry_delay_sec=120)
                    if trade:
                        trade.update(
                            day=day,
                            notional=notional,
                            variant=f"CONFIRM2M_{threshold}_TP{tp_bps}",
                            confirm_bps=post2,
                        )
                        confirm2.append(trade)

        candidates.append({"summary": summarize(f"RAW {threshold} TP{tp_bps}", raw), "trades": raw})
        candidates.append(
            {
                "summary": summarize(f"CONFIRM1M_ENTRY_AFTER_60S {threshold} TP{tp_bps}", confirm1),
                "trades": confirm1,
            }
        )
        candidates.append(
            {
                "summary": summarize(f"CONFIRM2M_ENTRY_AFTER_120S {threshold} TP{tp_bps}", confirm2),
                "trades": confirm2,
            }
        )

    summaries = sorted(
        [candidate["summary"] for candidate in replay_trade_candidates + candidates],
        key=lambda row: row.get("mean_net_bps", -999),
        reverse=True,
    )

    interesting = []
    for label in [
        "REPLAY_CLUSTER 200000 TP80",
        "REPLAY_CLUSTER 200000 TP60",
        "RAW 200000 TP80",
        "RAW 200000 TP60",
        "CONFIRM1M_ENTRY_AFTER_60S 200000 TP60",
        "CONFIRM2M_ENTRY_AFTER_120S 100000 TP40",
    ]:
        trades = []
        for candidate in replay_trade_candidates + candidates:
            if candidate["summary"]["name"] == label:
                trades = candidate["trades"]
                break
        interesting.append(
            {
                "name": label,
                "day_rows": [summarize(day, [t for t in trades if t["day"] == day]) for day in DAYS],
            }
        )

    OUT_JSON.write_text(
        json.dumps({"summaries": summaries, "interesting_day_splits": interesting}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# S34 SELL Liquidation Path Quality",
        "",
        "Date: 2026-06-16",
        "",
        "This is a read-only replay on historical `liquidations` and `mark_prices`. It uses simplified mark-price fills and a flat 8 bps round trip. It is not live paper parity.",
        "",
        "Important: `REPLAY_CLUSTER` rows use the exact signal/trade timestamps from the prior SELL replay JSON. `RAW` rows in this report count individual liquidation events directly and are included only as a diagnostic; they are not comparable to the clustered replay sample.",
        "",
        "## Candidate Quality Table",
        "",
        "| Rank | Candidate | N | Days | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | TP Touch | SL Touch | BE Hit | Exits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, summary in enumerate(summaries[:12], 1):
        lines.append(
            f"| {idx} | {summary['name']} | {summary.get('n', 0)} | {summary.get('days', 0)} | "
            f"{fmt(summary.get('mean_net_bps'))} | {fmt(summary.get('median_net_bps'))} | "
            f"{fmt(summary.get('cum_net_bps'))} | {pct(summary.get('wr'))} | "
            f"{fmt(summary.get('mean_mfe_bps'))} | {fmt(summary.get('mean_mae_bps'))} | "
            f"{pct(summary.get('tp_touch_rate'))} | {pct(summary.get('sl_touch_rate'))} | "
            f"{pct(summary.get('be_hit_rate'))} | {summary.get('exits', {})} |"
        )

    lines.extend(
        [
            "",
            "## Deployability Check",
            "",
            "The earlier filter sweep showed positive SELL pockets when a 1m/2m short confirmation was observed while entry remained at the original signal timestamp. This path-quality pass re-tested confirmation in deployable form: wait for confirmation first, then enter after 60s or 120s. Under that deployable timing, the apparent edge did not survive. The best deployable confirmation candidates are still negative on mean net bps.",
            "",
            "## Day Splits",
        ]
    )
    for block in interesting:
        lines.extend(
            [
                "",
                f"### {block['name']}",
                "",
                "| Day | N | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | Exits |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in block["day_rows"]:
            lines.append(
                f"| {row['name']} | {row.get('n', 0)} | {fmt(row.get('mean_net_bps'))} | "
                f"{fmt(row.get('median_net_bps'))} | {fmt(row.get('cum_net_bps'))} | "
                f"{pct(row.get('wr'))} | {fmt(row.get('mean_mfe_bps'))} | "
                f"{fmt(row.get('mean_mae_bps'))} | {row.get('exits', {})} |"
            )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "SELL liquidation -> SHORT is not ready for an exploratory live-paper rule. Raw 200K SELL shorts are only mildly positive and median-negative, while deployable confirmation filters turn negative once the entry is delayed until after confirmation. Keep SELL as research-only for now; do not mix it into the active S34 paper validation.",
        ]
    )
    OUT_MD.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    print(OUT_MD)
    print(json.dumps(summaries[:5], indent=2))
    con.close()


if __name__ == "__main__":
    main()
