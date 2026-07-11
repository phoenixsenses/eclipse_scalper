import datetime as dt
import json
import sqlite3
import statistics
from pathlib import Path

from ami.storage import production as PR
from ami.storage import research_reader as RR

FEATURE_DB = "data/s34_feature_factory.db"
SOURCE_DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
OUT_JSON = Path("reports/research/s34/S34_TRAILING_HALF_MFE_OOS_REALFILL.json")
OUT_MD = Path("reports/research/s34/S34_TRAILING_HALF_MFE_OOS_REALFILL.md")

FILTER_SQL = "f.cluster_notional >= 500000 and f.day_trend_bps >= 0"
ROUTE_ID = "LONG_DELAY0_TP60"
SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"

TP_BPS = 60.0
SL_BPS = 40.0
BE_BPS = 30.0
TRAIL_ARM_BPS = 30.0
TRAIL_KEEP_FRACTION = 0.5
ENTRY_DELAY_SEC = 0
MAX_HORIZON_SEC = 3600
TAKER_FEE_BPS = 4.0
ROUND_TRIP_FEE_BPS = 8.0
MAX_BOOK_STALENESS_SEC = 5


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


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


def iso_day(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).date().isoformat()


def signed_ret(entry_price: float, price: float) -> float:
    return (float(price) - float(entry_price)) / float(entry_price) * 10000.0


def price_from_signed_ret(entry_price: float, signed_bps: float) -> float:
    return float(entry_price) * (1.0 + float(signed_bps) / 10000.0)


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    # OUT-OF-SCOPE for ASOF V8: dual-direction (before=True is an as-of DESC
    # LIMIT 1; before=False is a forward "at-or-after" ASC LIMIT 1), but this
    # file's only two call sites always pass before=False (confirmed by
    # inspection) -- the as-of branch is never actually exercised, and the
    # exercised branch is the opposite direction from
    # lookup_latest_at_or_before's semantics regardless. Left on direct SQL.
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
        (symbol, int(ts_ms)),
    ).fetchone()


def path_marks(con: sqlite3.Connection, symbol: str, entry_ts_ms: int) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `path_marks_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-CONSUMER-
    MIGRATION-V9). No longer called by `simulate_current_be30`/
    `simulate_trail_half_mfe`; the reader-backed path is used instead.
    Bounded range read (ts_ms BETWEEN) on `mark_prices`; the as-of `mark_at`
    forward lookup is out of scope for this range-read gate."""
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            select ts_ms, mark_price
            from mark_prices
            where symbol=? and ts_ms>=? and ts_ms<=?
            order by ts_ms
            """,
            (symbol, int(entry_ts_ms), int(entry_ts_ms) + MAX_HORIZON_SEC * 1000),
        )
    ]


def path_marks_v2(root, symbol: str, entry_ts_ms: int, source_db_path=None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `path_marks`, via `plan_read`/
    `execute_read`. `symbol` is a genuine function parameter, but this file's
    two call sites always pass the module constant `SYMBOL="ETHUSDT"`
    (confirmed by inspection); the helper stays symbol-generic regardless.

    Range semantics: the oracle uses `ts_ms>=?` (INCLUSIVE lower) and
    `ts_ms<=?` (INCLUSIVE upper). `plan_read`/`execute_read` use the reader's
    half-open `[start_ms, end_ms)` convention. Since `ts_ms` is always an
    integer column, `ts_ms<=hi` is exactly equivalent to `ts_ms<hi+1` -- so
    `start_ms=lo, end_ms=hi+1` reproduces BOTH boundaries bit-for-bit, proven
    by a dedicated parity test (ARCHIVE_ONLY / HYBRID / SQLITE_ONLY)."""
    lo = int(entry_ts_ms)
    hi = int(entry_ts_ms) + MAX_HORIZON_SEC * 1000
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=lo, end_ms=hi + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(price)) for ts, price in result.iter_rows()]


def book_ticker_at(con: sqlite3.Connection, symbol: str, ts_ms: int):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_ticker_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V8). No longer called by `real_fill_cost`; the
    reader-backed path is used instead."""
    row = con.execute(
        """
        select ts_ms, bid_price, ask_price, mid_price
        from book_ticker
        where symbol=? and ts_ms<=?
        order by ts_ms desc
        limit 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    if int(ts_ms) - int(row[0]) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


_BOOK_COLS = ("ts_ms", "bid_price", "ask_price", "mid_price")


def book_ticker_at_v2(root, symbol: str, ts_ms: int, source_db_path=None):
    """Reader-backed replacement for `book_ticker_at`, via
    lookup_latest_at_or_before. `symbol` is a genuine function parameter,
    but this file's sole call site (`real_fill_cost`, called twice per
    event) always passes the hardcoded module constant `SYMBOL="ETHUSDT"`
    -- confirmed by direct source inspection, never overridden. book_ticker
    has no ETHUSDT archive partition (only SOLUSDT/2026-04 is archived),
    so real production use of this file resolves SQLITE_ONLY -- confirmed,
    not assumed. The helper itself stays symbol-generic and never assumes
    SOLUSDT's archive coverage applies to a different symbol."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask, mid = result.row
    if int(ts_ms) - int(row_ts) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row_ts), "bid": float(bid), "ask": float(ask), "mid": float(mid)}


def real_fill_cost(entry_ref: float, exit_ref: float, entry_ts_ms: int, exit_ts_ms: int, root, source_db_path=None):
    entry_book = book_ticker_at_v2(root, SYMBOL, entry_ts_ms, source_db_path=source_db_path)
    exit_book = book_ticker_at_v2(root, SYMBOL, exit_ts_ms, source_db_path=source_db_path)
    if not entry_book or not exit_book:
        return None
    basis = float(entry_ref)
    entry_fill = float(entry_book["ask"])
    exit_fill = float(exit_book["bid"])
    entry_mid = float(entry_book["mid"])
    exit_mid = float(exit_book["mid"])
    gross_bps = signed_ret(basis, exit_ref)
    executable_bps = signed_ret(entry_fill, exit_fill) * (entry_fill / basis)
    entry_adverse_bps = (entry_mid - basis) / basis * 10000.0
    exit_adverse_bps = (float(exit_ref) - exit_mid) / basis * 10000.0
    spread_cost_bps = ((entry_fill - entry_mid) + (exit_mid - exit_fill)) / basis * 10000.0
    fee_cost_bps = TAKER_FEE_BPS * 2
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net_bps = executable_bps - fee_cost_bps
    if abs(net_bps - executable_net_bps) > 1e-6:
        raise RuntimeError(f"real-fill identity mismatch net={net_bps} executable={executable_net_bps}")
    return {
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "real_net_bps": net_bps,
        "entry_book_ts_ms": entry_book["ts_ms"],
        "exit_book_ts_ms": exit_book["ts_ms"],
    }


def simulate_current_be30(source_con: sqlite3.Connection, event: dict, root, source_db_path=None) -> dict | None:
    entry_target = int(event["event_ts_ms"]) + ENTRY_DELAY_SEC * 1000
    entry = mark_at(source_con, SYMBOL, entry_target, before=False)
    if not entry:
        return None
    entry_ts_ms, entry_price = int(entry[0]), float(entry[1])
    marks = path_marks_v2(root, SYMBOL, entry_ts_ms, source_db_path=source_db_path)
    if not marks:
        return None
    be_active = False
    mfe = -1e9
    exit_reason = "TIME"
    exit_ts_ms, exit_price = marks[-1]
    for ts_ms, price in marks:
        ret = signed_ret(entry_price, price)
        mfe = max(mfe, ret)
        if not be_active and ret >= BE_BPS:
            be_active = True
        if ret >= TP_BPS:
            exit_reason = "TP"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if ret <= -SL_BPS:
            exit_reason = "SL"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts_ms = ts_ms
            exit_price = price_from_signed_ret(entry_price, 0.0)
            break
    gross_bps = signed_ret(entry_price, exit_price)
    return {
        "event_id": event["event_id"],
        "event_ts_ms": int(event["event_ts_ms"]),
        "day": iso_day(int(event["event_ts_ms"])),
        "variant": "CURRENT_BE30",
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "exit_ts_ms": int(exit_ts_ms),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - ROUND_TRIP_FEE_BPS,
        "mfe_bps": mfe,
    }


def simulate_trail_half_mfe(source_con: sqlite3.Connection, event: dict, root, source_db_path=None) -> dict | None:
    entry_target = int(event["event_ts_ms"]) + ENTRY_DELAY_SEC * 1000
    entry = mark_at(source_con, SYMBOL, entry_target, before=False)
    if not entry:
        return None
    entry_ts_ms, entry_price = int(entry[0]), float(entry[1])
    marks = path_marks_v2(root, SYMBOL, entry_ts_ms, source_db_path=source_db_path)
    if not marks:
        return None
    mfe = -1e9
    stop_bps = -SL_BPS
    exit_reason = "TIME"
    exit_ts_ms, exit_price = marks[-1]
    for ts_ms, price in marks:
        ret = signed_ret(entry_price, price)
        if ret > mfe:
            mfe = ret
            if mfe >= TRAIL_ARM_BPS:
                stop_bps = max(stop_bps, mfe * TRAIL_KEEP_FRACTION)
        if ret >= TP_BPS:
            exit_reason = "TP"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if ret <= stop_bps:
            exit_reason = "TRAIL" if stop_bps > 0 else "SL"
            exit_ts_ms = ts_ms
            exit_price = price_from_signed_ret(entry_price, stop_bps)
            break
    gross_bps = signed_ret(entry_price, exit_price)
    return {
        "event_id": event["event_id"],
        "event_ts_ms": int(event["event_ts_ms"]),
        "day": iso_day(int(event["event_ts_ms"])),
        "variant": "TRAIL_HALF_MFE_ARM30",
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "exit_ts_ms": int(exit_ts_ms),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - ROUND_TRIP_FEE_BPS,
        "mfe_bps": mfe,
    }


def load_events(feature_con: sqlite3.Connection) -> list[dict]:
    feature_con.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in feature_con.execute(
            f"""
            select f.*
            from liq_event_features f
            where f.symbol=? and f.liq_side=? and ({FILTER_SQL})
            order by f.event_ts_ms
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


def add_real_fill(rows: list[dict], root, source_db_path=None) -> tuple[list[dict], int]:
    out = []
    no_fill = 0
    for row in rows:
        real = real_fill_cost(
            float(row["entry_price"]),
            float(row["exit_price"]),
            int(row["entry_ts_ms"]),
            int(row["exit_ts_ms"]),
            root,
            source_db_path=source_db_path,
        )
        if not real:
            no_fill += 1
            continue
        out.append({**row, **real})
    return out, no_fill


def format_num(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    # `source_con` stays open for the still-direct mark_prices ASOF reads
    # (simulate_current_be30 / simulate_trail_half_mfe via `mark_at`); the
    # bounded mark_prices range read (`path_marks` -> `path_marks_v2`) and the
    # book_ticker point-lookup path (real_fill_cost) both moved to the reader,
    # via `root`/SOURCE_DB_PATH.
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    source_con.execute("pragma query_only=1")
    root, _ = PR.resolve_production_root()

    min_ts, max_ts = feature_con.execute("select min(event_ts_ms), max(event_ts_ms) from liq_event_features").fetchone()
    split_ts_ms = int((int(min_ts) + int(max_ts)) / 2)
    events = load_events(feature_con)

    current_rows = [row for event in events if (row := simulate_current_be30(source_con, event, root, source_db_path=SOURCE_DB_PATH))]
    trail_rows = [row for event in events if (row := simulate_trail_half_mfe(source_con, event, root, source_db_path=SOURCE_DB_PATH))]

    splits = {
        "CURRENT_BE30": split_rows(current_rows, split_ts_ms),
        "TRAIL_HALF_MFE_ARM30": split_rows(trail_rows, split_ts_ms),
    }

    summary = {}
    for variant, periods in splits.items():
        summary[variant] = {period: summarize(rows) for period, rows in periods.items()}

    real_fill = {}
    for variant, rows in {"CURRENT_BE30": current_rows, "TRAIL_HALF_MFE_ARM30": trail_rows}.items():
        filled_rows, no_fill = add_real_fill(rows, root, source_db_path=SOURCE_DB_PATH)
        periods = split_rows(filled_rows, split_ts_ms)
        real_fill[variant] = {
            "total_rows": len(rows),
            "real_fill_rows": len(filled_rows),
            "no_fill_rows": no_fill,
            "no_fill_rate": no_fill / len(rows) if rows else None,
            "summary": {period: summarize(period_rows, key="real_net_bps") for period, period_rows in periods.items()},
            "mean_entry_adverse": sum(r["entry_adverse_bps"] for r in filled_rows) / len(filled_rows) if filled_rows else None,
            "mean_exit_adverse": sum(r["exit_adverse_bps"] for r in filled_rows) / len(filled_rows) if filled_rows else None,
            "mean_spread": sum(r["spread_cost_bps"] for r in filled_rows) / len(filled_rows) if filled_rows else None,
            "mean_fee": sum(r["fee_cost_bps"] for r in filled_rows) / len(filled_rows) if filled_rows else None,
        }

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "feature_db": FEATURE_DB,
            "source_db": SOURCE_DB,
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "filter": FILTER_SQL,
            "route": ROUTE_ID,
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "current_be_bps": BE_BPS,
            "trail_arm_bps": TRAIL_ARM_BPS,
            "trail_keep_fraction": TRAIL_KEEP_FRACTION,
            "event_count": len(events),
            "split_ts_ms": split_ts_ms,
            "split_utc": dt.datetime.fromtimestamp(split_ts_ms / 1000, tz=dt.timezone.utc).isoformat(),
            "cost_note": "OOS section uses simplified mark path with flat 8 bps. Real-fill section uses historical book_ticker ask entry / bid exit where available.",
        },
        "simplified_oos": summary,
        "real_fill": real_fill,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Trailing Half-MFE OOS + Real-Fill Check",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: ETH BUY feature-factory events with `cluster_notional >= 500K AND day_trend_bps >= 0`, route `LONG_DELAY0_TP60`.",
        "",
        "No runner/config changes. This is research-only.",
        "",
        f"Events: `{len(events)}`",
        f"Temporal split: `{payload['scope']['split_utc']}`",
        "",
        "## 1. Temporal OOS, Simplified Mark-Fill",
        "",
        "| Variant | Period | N | Days | Cum | Mean | Median | WR | Top3 Removed | Positive Days | Exits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in ["CURRENT_BE30", "TRAIL_HALF_MFE_ARM30"]:
        for period in ["train", "test", "all"]:
            row = summary[variant][period]
            lines.append(
                f"| {variant} | {period} | {row['n']} | {row['days']} | {format_num(row['cum'])} | "
                f"{format_num(row['mean'])} | {format_num(row['median'])} | "
                f"{row['wr']*100:.1f}% | {format_num(row['top3_removed_cum'])} | "
                f"{row['positive_days']}/{row['days']} | {row['exit_counts']} |"
            )
    lines.extend(
        [
            "",
            "## 2. Real-Fill Parity",
            "",
            "| Variant | Total | Real Fill | No Fill | No Fill Rate | Period | Real Cum | Real Mean | Real Median | WR | Top3 Removed | Positive Days |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ["CURRENT_BE30", "TRAIL_HALF_MFE_ARM30"]:
        rf = real_fill[variant]
        for period in ["train", "test", "all"]:
            row = rf["summary"][period]
            lines.append(
                f"| {variant} | {rf['total_rows']} | {rf['real_fill_rows']} | {rf['no_fill_rows']} | "
                f"{rf['no_fill_rate']*100:.1f}% | {period} | {format_num(row['cum'])} | "
                f"{format_num(row['mean'])} | {format_num(row['median'])} | "
                f"{row['wr']*100:.1f}% | {format_num(row['top3_removed_cum'])} | "
                f"{row['positive_days']}/{row['days']} |"
            )
    lines.extend(
        [
            "",
            "## 3. Cost Components, Real-Fill Rows",
            "",
            "| Variant | Entry Adverse | Exit Adverse | Spread | Fee |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in ["CURRENT_BE30", "TRAIL_HALF_MFE_ARM30"]:
        rf = real_fill[variant]
        lines.append(
            f"| {variant} | {format_num(rf['mean_entry_adverse'])} | {format_num(rf['mean_exit_adverse'])} | "
            f"{format_num(rf['mean_spread'])} | {format_num(rf['mean_fee'])} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "This checks whether the trailing idea survives a temporal split and whether the result remains positive under real historical bid/ask fills where available. It is still a retrospective sweep on a discovered exit idea, not authorization to change the live runner.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({"simplified_oos": summary, "real_fill": real_fill}, indent=2)[:6000])
    source_con.close()
    feature_con.close()


if __name__ == "__main__":
    main()
