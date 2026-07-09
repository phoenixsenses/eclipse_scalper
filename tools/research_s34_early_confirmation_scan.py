from __future__ import annotations

import datetime as dt
import itertools
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


from ami.storage import production as PR
from ami.storage import research_reader as RR

FEATURE_DB = "data/s34_feature_factory.db"
SOURCE_DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
OUT_JSON = Path("reports/research/s34/S34_EARLY_CONFIRMATION_SCAN.json")
OUT_MD = Path("reports/research/s34/S34_EARLY_CONFIRMATION_SCAN.md")

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
TP_BPS = 60.0
SL_BPS = 40.0
BE_BPS = 30.0
MAX_HORIZON_SEC = 3600
TAKER_FEE_BPS = 4.0
ROUND_TRIP_FEE_BPS = 8.0
MAX_BOOK_STALENESS_SEC = 5

SCOPES = {
    "ALL_200K": "cluster_notional >= 200000",
    "500K_DAYTREND": "cluster_notional >= 500000 and day_trend_bps >= 0",
}

WAIT_GRID_SEC = [30, 60, 120]
WAIT_RET_MIN_GRID = [-5.0, 0.0, 5.0, 10.0, 15.0]
EARLY_MFE_MIN_GRID = [0.0, 10.0, 15.0, 20.0]
EARLY_MAE_MIN_GRID = [-40.0, -25.0, -15.0]


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iso_day(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).date().isoformat()


def signed_ret(entry: float, exit_: float) -> float:
    return (float(exit_) - float(entry)) / float(entry) * 10000.0


def price_from_ret(entry: float, bps: float) -> float:
    return float(entry) * (1.0 + float(bps) / 10000.0)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def summarize(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    days = sorted({str(row["day"]) for row in rows})
    day_cums = {day: sum(float(row[key]) for row in rows if row["day"] == day and row.get(key) is not None) for day in days}
    if not vals:
        return {
            "n": 0,
            "days": 0,
            "mean": None,
            "median": None,
            "cum": 0.0,
            "wr": None,
            "top3_removed_cum": 0.0,
            "positive_days": 0,
            "worst_day_cum": None,
            "exit_counts": {},
        }
    return {
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "worst_day_cum": min(day_cums.values()) if day_cums else None,
        "exit_counts": count_by(rows, "exit_reason"),
    }


def mark_at(con: sqlite3.Connection, ts_ms: int, *, before: bool = False) -> tuple[int, float] | None:
    # OUT-OF-SCOPE for ASOF V9: dual-direction (before=True is an as-of DESC
    # LIMIT 1; before=False is a forward "at-or-after" ASC LIMIT 1), but
    # this file's only two call sites (in early_window) always pass
    # before=False -- the as-of branch is never actually exercised, and the
    # exercised branch is the opposite direction from
    # lookup_latest_at_or_before's semantics regardless. Left on direct SQL.
    op = "<=" if before else ">="
    order = "desc" if before else "asc"
    row = con.execute(
        f"""
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms {op} ?
        order by ts_ms {order}
        limit 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()
    return None if not row else (int(row[0]), float(row[1]))


def path_marks(con: sqlite3.Connection, start_ts_ms: int, horizon_sec: int) -> list[tuple[int, float]]:
    # OUT-OF-SCOPE for ASOF V9: bounded range read (ts_ms BETWEEN), not the
    # as-of point lookup this gate migrates. Left on direct SQL.
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            select ts_ms, mark_price
            from mark_prices
            where symbol=? and ts_ms>=? and ts_ms<=?
            order by ts_ms
            """,
            (SYMBOL, int(start_ts_ms), int(start_ts_ms) + int(horizon_sec) * 1000),
        )
    ]


def book_ticker_at(con: sqlite3.Connection, ts_ms: int) -> dict[str, float] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_ticker_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V9). No longer called by `real_fill_cost`; the
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


def real_fill_cost(entry_ref: float, exit_ref: float, entry_ts_ms: int, exit_ts_ms: int, root, source_db_path=None):
    entry_book = book_ticker_at_v2(root, entry_ts_ms, source_db_path=source_db_path)
    exit_book = book_ticker_at_v2(root, exit_ts_ms, source_db_path=source_db_path)
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
    fee_cost_bps = TAKER_FEE_BPS * 2.0
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net = executable_bps - fee_cost_bps
    if abs(net_bps - executable_net) > 1e-6:
        raise RuntimeError(f"identity mismatch {net_bps} != {executable_net}")
    return {
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "real_net_bps": net_bps,
    }


def load_events(feature_con: sqlite3.Connection) -> list[dict[str, Any]]:
    feature_con.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in feature_con.execute(
            """
            select *
            from liq_event_features
            where symbol=? and liq_side=?
            order by event_ts_ms
            """,
            (SYMBOL, LIQ_SIDE),
        )
    ]


def early_window(source_con: sqlite3.Connection, event: dict[str, Any], wait_sec: int) -> dict[str, Any] | None:
    signal_mark = mark_at(source_con, int(event["event_ts_ms"]), before=False)
    wait_mark = mark_at(source_con, int(event["event_ts_ms"]) + int(wait_sec) * 1000, before=False)
    if not signal_mark or not wait_mark:
        return None
    signal_ts, signal_price = signal_mark
    wait_ts, wait_price = wait_mark
    marks = path_marks(source_con, signal_ts, wait_sec)
    if not marks:
        return None
    rets = [signed_ret(signal_price, price) for _, price in marks if _ <= wait_ts]
    return {
        "signal_ts_ms": signal_ts,
        "signal_price": signal_price,
        "wait_ts_ms": wait_ts,
        "wait_price": wait_price,
        "wait_ret_bps": signed_ret(signal_price, wait_price),
        "early_mfe_bps": max(rets),
        "early_mae_bps": min(rets),
    }


def simulate_after_wait(source_con: sqlite3.Connection, event: dict[str, Any], wait: dict[str, Any]) -> dict[str, Any] | None:
    entry_ts_ms = int(wait["wait_ts_ms"])
    entry_price = float(wait["wait_price"])
    marks = path_marks(source_con, entry_ts_ms, MAX_HORIZON_SEC)
    if not marks:
        return None
    be_active = False
    mfe = -1e9
    mae = 1e9
    exit_reason = "TIME"
    exit_ts_ms, exit_price = marks[-1]
    for ts_ms, price in marks:
        ret = signed_ret(entry_price, price)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
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
            exit_price = price_from_ret(entry_price, 0.0)
            break
    gross = signed_ret(entry_price, exit_price)
    return {
        "event_id": event["event_id"],
        "event_ts_ms": int(event["event_ts_ms"]),
        "event_utc": event["event_utc"],
        "day": iso_day(int(event["event_ts_ms"])),
        "cluster_notional": float(event.get("cluster_notional") or 0.0),
        "day_trend_bps": event.get("day_trend_bps"),
        "wait_ts_ms": entry_ts_ms,
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "exit_ts_ms": int(exit_ts_ms),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
        **wait,
    }


def row_in_scope(row: dict[str, Any], scope_expr: str) -> bool:
    if " and " in scope_expr:
        return all(row_in_scope(row, part.strip()) for part in scope_expr.split(" and "))
    for op in (">=", "<=", "<", ">"):
        if op in scope_expr:
            col, val = scope_expr.split(op, 1)
            actual = row.get(col.strip())
            return actual is not None and eval_compare(float(actual), op, float(val.strip()))
    raise ValueError(f"unsupported scope {scope_expr}")


def eval_compare(actual: float, op: str, value: float) -> bool:
    if op == ">=":
        return actual >= value
    if op == "<=":
        return actual <= value
    if op == "<":
        return actual < value
    if op == ">":
        return actual > value
    raise ValueError(op)


def split_rows(rows: list[dict[str, Any]], split_ts_ms: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return [r for r in rows if int(r["event_ts_ms"]) <= split_ts_ms], [r for r in rows if int(r["event_ts_ms"]) > split_ts_ms]


def build_rows(source_con: sqlite3.Connection, events: list[dict[str, Any]], wait_sec: int) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        wait = early_window(source_con, event, wait_sec)
        if not wait:
            continue
        sim = simulate_after_wait(source_con, event, wait)
        if sim:
            rows.append(sim)
    return rows


def scan_scope(scope_name: str, scope_expr: str, events: list[dict[str, Any]], source_con: sqlite3.Connection, split_ts_ms: int) -> dict[str, Any]:
    scoped_events = [e for e in events if row_in_scope(e, scope_expr)]
    by_wait = {wait_sec: build_rows(source_con, scoped_events, wait_sec) for wait_sec in WAIT_GRID_SEC}
    candidates = []
    for wait_sec, rows in by_wait.items():
        train, test = split_rows(rows, split_ts_ms)
        for wait_ret_min, early_mfe_min, early_mae_min in itertools.product(
            WAIT_RET_MIN_GRID, EARLY_MFE_MIN_GRID, EARLY_MAE_MIN_GRID
        ):
            label = f"wait{wait_sec}_ret>={wait_ret_min:g}_mfe>={early_mfe_min:g}_mae>={early_mae_min:g}"
            train_rows = [
                row
                for row in train
                if float(row["wait_ret_bps"]) >= wait_ret_min
                and float(row["early_mfe_bps"]) >= early_mfe_min
                and float(row["early_mae_bps"]) >= early_mae_min
            ]
            if len(train_rows) < 10:
                continue
            train_summary = summarize(train_rows)
            candidates.append(
                {
                    "label": label,
                    "wait_sec": wait_sec,
                    "wait_ret_min": wait_ret_min,
                    "early_mfe_min": early_mfe_min,
                    "early_mae_min": early_mae_min,
                    "train": train_summary,
                }
            )
    candidates.sort(key=lambda c: (c["train"]["median"] or -999, c["train"]["top3_removed_cum"], c["train"]["cum"]), reverse=True)
    top = []
    for rank, cand in enumerate(candidates[:8], start=1):
        rows = by_wait[cand["wait_sec"]]
        train, test = split_rows(rows, split_ts_ms)
        filt = lambda row: (
            float(row["wait_ret_bps"]) >= cand["wait_ret_min"]
            and float(row["early_mfe_bps"]) >= cand["early_mfe_min"]
            and float(row["early_mae_bps"]) >= cand["early_mae_min"]
        )
        top.append(
            {
                **cand,
                "rank": rank,
                "test": summarize([r for r in test if filt(r)]),
                "all": summarize([r for r in rows if filt(r)]),
            }
        )
    return {
        "scope": scope_name,
        "scope_expr": scope_expr,
        "scoped_events": len(scoped_events),
        "wait_baselines": {str(wait): summarize(rows) for wait, rows in by_wait.items()},
        "candidates_scanned": len(candidates),
        "top_candidates": top,
        "rows_by_wait": by_wait,
    }


def add_real_fill(scan: dict[str, Any], root, split_ts_ms: int, source_db_path=None) -> dict[str, Any]:
    for cand in scan["top_candidates"]:
        rows = scan["rows_by_wait"][cand["wait_sec"]]
        _, test = split_rows(rows, split_ts_ms)
        selected = [
            r
            for r in rows
            if float(r["wait_ret_bps"]) >= cand["wait_ret_min"]
            and float(r["early_mfe_bps"]) >= cand["early_mfe_min"]
            and float(r["early_mae_bps"]) >= cand["early_mae_min"]
        ]
        selected_test = [
            r
            for r in test
            if float(r["wait_ret_bps"]) >= cand["wait_ret_min"]
            and float(r["early_mfe_bps"]) >= cand["early_mfe_min"]
            and float(r["early_mae_bps"]) >= cand["early_mae_min"]
        ]
        filled = [{**r, **rf} for r in selected if (rf := real_fill_cost(r["entry_price"], r["exit_price"], r["entry_ts_ms"], r["exit_ts_ms"], root, source_db_path=source_db_path))]
        filled_test = [
            {**r, **rf}
            for r in selected_test
            if (rf := real_fill_cost(r["entry_price"], r["exit_price"], r["entry_ts_ms"], r["exit_ts_ms"], root, source_db_path=source_db_path))
        ]
        cand["real_fill"] = {
            "total_rows": len(selected),
            "real_fill_rows": len(filled),
            "no_fill_rows": len(selected) - len(filled),
            "no_fill_rate": (len(selected) - len(filled)) / len(selected) if selected else None,
            "test": summarize(filled_test, "real_net_bps"),
        }
    scan.pop("rows_by_wait", None)
    return scan


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.2f}"


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def write_report(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# S34 Early Confirmation Scan",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Scope: ETH BUY events. This models delayed entry after observing the first 30/60/120 seconds.",
        "",
        "No live runner/config changes. First-window data is not used to pretend signal-time entry; entry is moved to the wait timestamp.",
        "",
    ]
    for scan in payload["scans"]:
        lines.extend(
            [
                f"## {scan['scope']}",
                "",
                f"- Scope SQL: `{scan['scope_expr']}`",
                f"- Events: `{scan['scoped_events']}`",
                f"- Candidate count after train min-N filter: `{scan['candidates_scanned']}`",
                "",
                "### Wait Baselines",
                "",
                "| Wait | N | Median | Mean | Cum | WR | Top3 Removed | Positive Days | Exits |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for wait, summary in scan["wait_baselines"].items():
            lines.append(
                f"| {wait}s | {summary['n']} | {fmt(summary['median'])} | {fmt(summary['mean'])} | {fmt(summary['cum'])} | "
                f"{pct(summary['wr'])} | {fmt(summary['top3_removed_cum'])} | {summary['positive_days']}/{summary['days']} | {summary['exit_counts']} |"
            )
        lines.extend(
            [
                "",
                "### Top OOS Candidates",
                "",
                "| Rank | Candidate | Train N | Train Median | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cand in scan["top_candidates"]:
            test = cand["test"]
            train = cand["train"]
            lines.append(
                f"| {cand['rank']} | {cand['label']} | {train['n']} | {fmt(train['median'])} | {test['n']} | "
                f"{fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | {fmt(test['top3_removed_cum'])} | "
                f"{test['positive_days']}/{test['days']} |"
            )
        lines.extend(
            [
                "",
                "### Real-Fill Parity",
                "",
                "| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cand in scan["top_candidates"]:
            rf = cand["real_fill"]
            test = rf["test"]
            lines.append(
                f"| {cand['label']} | {rf['total_rows']} | {rf['real_fill_rows']} | {rf['no_fill_rows']} ({pct(rf['no_fill_rate'])}) | "
                f"{test['n']} | {fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | "
                f"{fmt(test['top3_removed_cum'])} | {test['positive_days']}/{test['days']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Read",
            "",
            "This is a confirmation-delay research scan. Positives are not immediately live-tradeable because the same surface was swept; they require a separate pre-registered forward rule.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    # `source_con` stays open for the still-direct mark_prices reads
    # (scan_scope -> build_rows -> early_window/simulate_after_wait via
    # mark_at/path_marks); only the book_ticker point-lookup path
    # (add_real_fill -> real_fill_cost) moved to the reader, via `root`/
    # SOURCE_DB_PATH.
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    source_con.execute("pragma query_only=1")
    root, _ = PR.resolve_production_root()
    events = load_events(feature_con)
    min_ts, max_ts = feature_con.execute("select min(event_ts_ms), max(event_ts_ms) from liq_event_features").fetchone()
    split_ts_ms = int((int(min_ts) + int(max_ts)) / 2)
    scans = []
    for scope_name, scope_expr in SCOPES.items():
        scan = scan_scope(scope_name, scope_expr, events, source_con, split_ts_ms)
        scans.append(add_real_fill(scan, root, split_ts_ms, source_db_path=SOURCE_DB_PATH))
    payload = {
        "generated_at": iso_now(),
        "symbol": SYMBOL,
        "liq_side": LIQ_SIDE,
        "route": {"tp_bps": TP_BPS, "sl_bps": SL_BPS, "be_bps": BE_BPS, "max_horizon_sec": MAX_HORIZON_SEC},
        "split_ts_ms": split_ts_ms,
        "scans": scans,
    }
    write_report(payload)
    print(json.dumps({"out_md": str(OUT_MD), "scopes": [(s["scope"], s["scoped_events"]) for s in scans]}, indent=2))
    feature_con.close()
    source_con.close()


if __name__ == "__main__":
    main()
