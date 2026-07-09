import datetime as dt
import itertools
import json
import sqlite3
import statistics
from pathlib import Path

from ami.storage import production as PR
from ami.storage import research_reader as RR

FEATURE_DB = "data/s34_feature_factory.db"
SOURCE_DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
OUT_JSON = Path("reports/research/s34/S34_500K_DAYTREND_ROUTE_SWEEP.json")
OUT_MD = Path("reports/research/s34/S34_500K_DAYTREND_ROUTE_SWEEP.md")

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
FILTER_SQL = "cluster_notional >= 500000 and day_trend_bps >= 0"
ENTRY_DELAY_SEC = 0
MAX_HORIZON_SEC = 3600
TP_GRID = [40.0, 60.0, 80.0, 100.0, 120.0]
SL_GRID = [30.0, 40.0, 50.0]
BE_GRID = [20.0, 25.0, 30.0, 35.0, 40.0]
CURRENT = {"tp": 60.0, "sl": 40.0, "be": 30.0}
ROUND_TRIP_FEE_BPS = 8.0
TAKER_FEE_BPS = 4.0
MAX_BOOK_STALENESS_SEC = 5


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def iso_day(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).date().isoformat()


def signed_ret(entry_price: float, price: float) -> float:
    return (float(price) - float(entry_price)) / float(entry_price) * 10000.0


def price_from_ret(entry_price: float, bps: float) -> float:
    return float(entry_price) * (1.0 + float(bps) / 10000.0)


def summarize(rows: list[dict], key: str = "net_bps") -> dict:
    vals = [float(row[key]) for row in rows]
    days = sorted({row["day"] for row in rows})
    day_cums = {day: sum(float(row[key]) for row in rows if row["day"] == day) for day in days}
    top3_removed = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0
    return {
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals) if vals else None,
        "median": median(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals) if vals else None,
        "top3_removed_cum": top3_removed,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "worst_day_cum": min(day_cums.values()) if day_cums else None,
        "exit_counts": count_by(rows, "exit_reason"),
    }


def count_by(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def mark_at(con: sqlite3.Connection, ts_ms: int):
    # OUT-OF-SCOPE for ASOF V7: this is a forward "at-or-after" lookup
    # (ORDER BY ts_ms ASC LIMIT 1 with ts_ms>=?), the opposite direction of
    # the `ORDER BY ts_ms DESC LIMIT 1` as-of semantics that
    # lookup_latest_at_or_before implements. Left on direct SQL.
    return con.execute(
        """
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms>=?
        order by ts_ms asc
        limit 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()


def path_marks(con: sqlite3.Connection, entry_ts_ms: int) -> list[tuple[int, float]]:
    # OUT-OF-SCOPE for ASOF V7: bounded range read (ts_ms BETWEEN), not the
    # as-of point lookup this gate migrates. Left on direct SQL.
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            select ts_ms, mark_price
            from mark_prices
            where symbol=? and ts_ms>=? and ts_ms<=?
            order by ts_ms asc
            """,
            (SYMBOL, int(entry_ts_ms), int(entry_ts_ms) + MAX_HORIZON_SEC * 1000),
        )
    ]


def book_ticker_at(con: sqlite3.Connection, ts_ms: int):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_ticker_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V7). No longer called by `real_fill_net`; the
    reader-backed path is used instead."""
    row = con.execute(
        """
        select ts_ms, bid_price, ask_price, mid_price
        from book_ticker
        where symbol=? and ts_ms<=?
        order by ts_ms desc
        limit 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    if int(ts_ms) - int(row[0]) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


_BOOK_COLS = ("ts_ms", "bid_price", "ask_price", "mid_price")


def book_ticker_at_v2(root, ts_ms: int, source_db_path=None):
    """Reader-backed replacement for `book_ticker_at`, via
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT, exactly as in
    the oracle's SQL (this file never varies it). book_ticker has no
    ETHUSDT archive partition (only SOLUSDT/2026-04 is archived), so real
    production use of this file resolves SQLITE_ONLY -- confirmed, not
    assumed."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=SYMBOL, ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask, mid = result.row
    if int(ts_ms) - int(row_ts) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row_ts), "bid": float(bid), "ask": float(ask), "mid": float(mid)}


def real_fill_net(root, row: dict, source_db_path=None) -> dict | None:
    entry_book = book_ticker_at_v2(root, int(row["entry_ts_ms"]), source_db_path=source_db_path)
    exit_book = book_ticker_at_v2(root, int(row["exit_ts_ms"]), source_db_path=source_db_path)
    if not entry_book or not exit_book:
        return None
    basis = float(row["entry_price"])
    exit_ref = float(row["exit_price"])
    entry_fill = float(entry_book["ask"])
    exit_fill = float(exit_book["bid"])
    entry_mid = float(entry_book["mid"])
    exit_mid = float(exit_book["mid"])
    gross_bps = signed_ret(basis, exit_ref)
    executable_bps = signed_ret(entry_fill, exit_fill) * (entry_fill / basis)
    entry_adverse_bps = (entry_mid - basis) / basis * 10000.0
    exit_adverse_bps = (exit_ref - exit_mid) / basis * 10000.0
    spread_cost_bps = ((entry_fill - entry_mid) + (exit_mid - exit_fill)) / basis * 10000.0
    fee_cost_bps = TAKER_FEE_BPS * 2.0
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net = executable_bps - fee_cost_bps
    if abs(net_bps - executable_net) > 1e-6:
        raise RuntimeError(f"identity mismatch {net_bps} != {executable_net}")
    return {
        **row,
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "real_net_bps": net_bps,
    }


def simulate_path(event: dict, marks: list[tuple[int, float]], tp: float, sl: float, be: float) -> dict | None:
    if not marks:
        return None
    entry_ts_ms, entry_price = marks[0]
    be_active = False
    mfe = -1e9
    mae = 1e9
    exit_reason = "TIME"
    exit_ts_ms, exit_price = marks[-1]
    for ts_ms, price in marks:
        ret = signed_ret(entry_price, price)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if not be_active and ret >= be:
            be_active = True
        if ret >= tp:
            exit_reason = "TP"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if ret <= -sl:
            exit_reason = "SL"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts_ms = ts_ms
            exit_price = price_from_ret(entry_price, 0.0)
            break
    gross = signed_ret(entry_price, exit_price)
    return {
        "event_id": event["event_id"],
        "event_ts_ms": int(event["event_ts_ms"]),
        "day": iso_day(int(event["event_ts_ms"])),
        "entry_ts_ms": int(entry_ts_ms),
        "entry_price": float(entry_price),
        "exit_ts_ms": int(exit_ts_ms),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def load_events(feature_con: sqlite3.Connection) -> list[dict]:
    feature_con.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in feature_con.execute(
            f"""
            select *
            from liq_event_features
            where symbol=? and liq_side=? and ({FILTER_SQL})
            order by event_ts_ms
            """,
            (SYMBOL, LIQ_SIDE),
        )
    ]


def split_rows(rows: list[dict], split_ts_ms: int) -> dict[str, list[dict]]:
    return {
        "train": [row for row in rows if int(row["event_ts_ms"]) <= split_ts_ms],
        "test": [row for row in rows if int(row["event_ts_ms"]) > split_ts_ms],
        "all": rows,
    }


def route_id(tp: float, sl: float, be: float) -> str:
    return f"TP{int(tp)}_SL{int(sl)}_BE{int(be)}"


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    source_con.execute("pragma query_only=1")
    root, _ = PR.resolve_production_root()

    events = load_events(feature_con)
    min_ts, max_ts = feature_con.execute("select min(event_ts_ms), max(event_ts_ms) from liq_event_features").fetchone()
    split_ts_ms = int((int(min_ts) + int(max_ts)) / 2)

    event_paths = {}
    for event in events:
        entry = mark_at(source_con, int(event["event_ts_ms"]) + ENTRY_DELAY_SEC * 1000)
        if not entry:
            continue
        event_paths[event["event_id"]] = path_marks(source_con, int(entry[0]))

    combinations = list(itertools.product(TP_GRID, SL_GRID, BE_GRID))
    route_rows: dict[str, list[dict]] = {}
    route_summary: dict[str, dict] = {}
    for tp, sl, be in combinations:
        rid = route_id(tp, sl, be)
        rows = []
        for event in events:
            path = event_paths.get(event["event_id"], [])
            row = simulate_path(event, path, tp, sl, be)
            if row:
                row.update({"route": rid, "tp": tp, "sl": sl, "be": be})
                rows.append(row)
        route_rows[rid] = rows
        periods = split_rows(rows, split_ts_ms)
        route_summary[rid] = {
            "tp": tp,
            "sl": sl,
            "be": be,
            "train": summarize(periods["train"]),
            "test": summarize(periods["test"]),
            "all": summarize(periods["all"]),
        }

    ranked_train = sorted(
        route_summary.values(),
        key=lambda r: (
            r["train"]["median"] is not None and r["train"]["median"] > 0,
            r["train"]["top3_removed_cum"],
            r["train"]["median"] if r["train"]["median"] is not None else -1e9,
            r["train"]["mean"] if r["train"]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )
    top5 = ranked_train[:5]
    current_rid = route_id(CURRENT["tp"], CURRENT["sl"], CURRENT["be"])
    top_ids = {route_id(r["tp"], r["sl"], r["be"]) for r in top5}
    top_ids.add(current_rid)

    real_fill = {}
    for rid in sorted(top_ids):
        rows = route_rows[rid]
        filled = []
        no_fill = 0
        for row in rows:
            rf = real_fill_net(root, row, source_db_path=SOURCE_DB_PATH)
            if not rf:
                no_fill += 1
                continue
            filled.append(rf)
        periods = split_rows(filled, split_ts_ms)
        real_fill[rid] = {
            "total_rows": len(rows),
            "real_fill_rows": len(filled),
            "no_fill_rows": no_fill,
            "no_fill_rate": no_fill / len(rows) if rows else None,
            "train": summarize(periods["train"], key="real_net_bps"),
            "test": summarize(periods["test"], key="real_net_bps"),
            "all": summarize(periods["all"], key="real_net_bps"),
            "mean_entry_adverse": sum(r["entry_adverse_bps"] for r in filled) / len(filled) if filled else None,
            "mean_exit_adverse": sum(r["exit_adverse_bps"] for r in filled) / len(filled) if filled else None,
            "mean_spread": sum(r["spread_cost_bps"] for r in filled) / len(filled) if filled else None,
            "mean_fee": sum(r["fee_cost_bps"] for r in filled) / len(filled) if filled else None,
        }

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "filter": FILTER_SQL,
            "entry_delay_sec": ENTRY_DELAY_SEC,
            "events": len(events),
            "simulated_events": len(event_paths),
            "combination_count": len(combinations),
            "split_ts_ms": split_ts_ms,
            "split_utc": dt.datetime.fromtimestamp(split_ts_ms / 1000, tz=dt.timezone.utc).isoformat(),
            "current_route": current_rid,
            "cost_note": "OOS grid uses mark path + flat 8 bps. Real-fill parity uses historical book_ticker ask entry / bid exit where available.",
        },
        "top5_train": top5,
        "current": route_summary[current_rid],
        "all_routes": route_summary,
        "real_fill": real_fill,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 500K/daytrend Route Sweep",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: `ETH BUY`, `cluster_notional >= 500K`, `day_trend_bps >= 0`, `delay0`; live runner/config unchanged.",
        "",
        f"Combinations evaluated: `{len(combinations)}`",
        f"Events: `{len(events)}`; simulated events: `{len(event_paths)}`",
        f"Temporal split: `{payload['scope']['split_utc']}`",
        "",
        "## 1. Train-Selected Top 5, With Test Performance",
        "",
        "| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(top5, 1):
        rid = route_id(row["tp"], row["sl"], row["be"])
        tr = row["train"]
        te = row["test"]
        lines.append(
            f"| {idx} | {rid} | {tr['n']} | {fmt(tr['median'])} | {fmt(tr['cum'])} | {fmt(tr['top3_removed_cum'])} | "
            f"{te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | {fmt(te['cum'])} | "
            f"{te['wr']*100:.1f}% | {fmt(te['top3_removed_cum'])} | {te['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Current Live Route Comparator",
            "",
            "| Route | Period | N | Median | Mean | Cum | WR | Top3 Removed | Exits |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    cur = route_summary[current_rid]
    for period in ["train", "test", "all"]:
        row = cur[period]
        lines.append(
            f"| {current_rid} | {period} | {row['n']} | {fmt(row['median'])} | {fmt(row['mean'])} | "
            f"{fmt(row['cum'])} | {row['wr']*100:.1f}% | {fmt(row['top3_removed_cum'])} | {row['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Real-Fill Parity: Top 5 + Current",
            "",
            "| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_real = [route_id(r["tp"], r["sl"], r["be"]) for r in top5]
    if current_rid not in ordered_real:
        ordered_real.append(current_rid)
    for rid in ordered_real:
        rf = real_fill[rid]
        te = rf["test"]
        lines.append(
            f"| {rid} | {rf['total_rows']} | {rf['real_fill_rows']} | {rf['no_fill_rows']} | "
            f"{rf['no_fill_rate']*100:.1f}% | {te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | "
            f"{fmt(te['cum'])} | {fmt(te['top3_removed_cum'])} | {te['positive_days']}/{te['days']} | "
            f"{fmt(rf['mean_entry_adverse'])} | {fmt(rf['mean_exit_adverse'])} | {fmt(rf['mean_spread'])} | {fmt(rf['mean_fee'])} |"
        )
    best_real = sorted(
        [(rid, real_fill[rid]["test"]) for rid in ordered_real],
        key=lambda x: (
            x[1]["n"] >= 20,
            x[1]["median"] if x[1]["median"] is not None else -1e9,
            x[1]["top3_removed_cum"],
            x[1]["mean"] if x[1]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )[0]
    lines.extend(
        [
            "",
            "## Read",
            "",
            f"Best real-fill test median among reported routes: `{best_real[0]}`. This remains a retrospective route sweep over 75 combinations; a stronger route should be pre-registered as a separate exploratory variant before any live paper promotion.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({"top5": top5, "current": route_summary[current_rid], "real_fill": real_fill}, indent=2)[:6000])
    source_con.close()
    feature_con.close()


if __name__ == "__main__":
    main()
