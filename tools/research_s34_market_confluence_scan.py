from __future__ import annotations

import datetime as dt
import itertools
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


FEATURE_DB = "data/s34_feature_factory.db"
SOURCE_DB = "file:data/microstructure.db?mode=ro"
OUT_JSON = Path("reports/research/s34/S34_MARKET_CONFLUENCE_SCAN.json")
OUT_MD = Path("reports/research/s34/S34_MARKET_CONFLUENCE_SCAN.md")

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
ROUTE_ID = "LONG_DELAY0_TP60"
MAX_BOOK_STALENESS_SEC = 5
TAKER_FEE_BPS = 4.0


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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
        return {"n": 0, "days": 0, "mean": None, "median": None, "cum": 0.0, "wr": None, "top3_removed_cum": 0.0, "positive_days": 0, "worst_day_cum": None, "exit_counts": {}}
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


def signed_ret(entry: float, exit_: float) -> float:
    return (float(exit_) - float(entry)) / float(entry) * 10000.0


def book_ticker_at(con: sqlite3.Connection, ts_ms: int) -> dict[str, float] | None:
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


def real_fill_net(source_con: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any] | None:
    entry_book = book_ticker_at(source_con, int(row["entry_ts_ms"]))
    exit_book = book_ticker_at(source_con, int(row["exit_ts_ms"]))
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
    return {**row, "real_net_bps": net_bps}


def load_base_rows(feature_con: sqlite3.Connection) -> list[dict[str, Any]]:
    feature_con.row_factory = sqlite3.Row
    rows = feature_con.execute(
        """
        select
          f.event_id, f.event_ts_ms, f.event_utc, date(f.event_ts_ms/1000, 'unixepoch') as day,
          f.cluster_notional, f.day_trend_bps, f.btc_pre_15m_bps, f.sol_pre_15m_bps,
          l.route_id, l.entry_ts_ms, l.entry_price, l.exit_ts_ms, l.exit_price,
          l.exit_reason, l.net_bps, l.mfe_bps, l.mae_bps
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where f.symbol=? and f.liq_side=? and l.route_id=?
        order by f.event_ts_ms
        """,
        (SYMBOL, LIQ_SIDE, ROUTE_ID),
    ).fetchall()
    return [dict(row) for row in rows]


def liq_notional(source_con: sqlite3.Connection, symbol: str, side: str, ts_ms: int, window_sec: int) -> float:
    row = source_con.execute(
        """
        select coalesce(sum(notional), 0.0)
        from liquidations
        where symbol=? and side=? and ts_ms>=? and ts_ms<?
        """,
        (symbol, side, int(ts_ms) - int(window_sec) * 1000, int(ts_ms)),
    ).fetchone()
    return float(row[0] or 0.0)


def enrich(rows: list[dict[str, Any]], source_con: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ts = int(row["event_ts_ms"])
        btc5 = liq_notional(source_con, "BTCUSDT", "BUY", ts, 300)
        btc15 = liq_notional(source_con, "BTCUSDT", "BUY", ts, 900)
        btc30 = liq_notional(source_con, "BTCUSDT", "BUY", ts, 1800)
        sol15 = liq_notional(source_con, "SOLUSDT", "BUY", ts, 900)
        eth15 = liq_notional(source_con, "ETHUSDT", "BUY", ts, 900)
        out.append(
            {
                **row,
                "btc_buy_liq_5m": btc5,
                "btc_buy_liq_15m": btc15,
                "btc_buy_liq_30m": btc30,
                "sol_buy_liq_15m": sol15,
                "eth_prior_buy_liq_15m": eth15,
                "market_buy_liq_15m": btc15 + sol15 + eth15,
                "btc_eth_liq_ratio_15m": btc15 / max(eth15, 1.0),
            }
        )
    return out


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    ts = sorted(int(row["event_ts_ms"]) for row in rows)
    split_ts = ts[len(ts) // 2]
    return [r for r in rows if int(r["event_ts_ms"]) <= split_ts], [r for r in rows if int(r["event_ts_ms"]) > split_ts], split_ts


def row_matches(row: dict[str, Any], expr: str) -> bool:
    if " AND " in expr:
        return all(row_matches(row, part.strip()) for part in expr.split(" AND "))
    for op in (">=", "<=", "<", ">"):
        if op in expr:
            col, val = expr.split(op, 1)
            actual = row.get(col.strip())
            if actual is None:
                return False
            actual_f = float(actual)
            val_f = float(val.strip())
            if op == ">=":
                return actual_f >= val_f
            if op == "<=":
                return actual_f <= val_f
            if op == "<":
                return actual_f < val_f
            if op == ">":
                return actual_f > val_f
    raise ValueError(expr)


def build_predicates(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    preds = [
        {"label": "eth_500k_daytrend", "expr": "cluster_notional >= 500000 AND day_trend_bps >= 0"},
        {"label": "btc5_ge_500k", "expr": "btc_buy_liq_5m >= 500000"},
        {"label": "btc15_ge_1m", "expr": "btc_buy_liq_15m >= 1000000"},
        {"label": "btc15_ge_2m", "expr": "btc_buy_liq_15m >= 2000000"},
        {"label": "btc30_ge_3m", "expr": "btc_buy_liq_30m >= 3000000"},
        {"label": "sol15_ge_200k", "expr": "sol_buy_liq_15m >= 200000"},
        {"label": "market15_ge_2m", "expr": "market_buy_liq_15m >= 2000000"},
        {"label": "market15_ge_5m", "expr": "market_buy_liq_15m >= 5000000"},
        {"label": "btc_pre15_ge_0", "expr": "btc_pre_15m_bps >= 0"},
        {"label": "btc_pre15_ge_20", "expr": "btc_pre_15m_bps >= 20"},
    ]
    for feature in ("btc_buy_liq_15m", "market_buy_liq_15m", "btc_eth_liq_ratio_15m"):
        vals = [float(r[feature]) for r in rows if r.get(feature) is not None]
        for q in (0.5, 0.75):
            threshold = statistics.quantiles(vals, n=4)[1 if q == 0.5 else 2] if vals else 0.0
            preds.append({"label": f"{feature}_ge_p{int(q*100)}_{threshold:.0f}", "expr": f"{feature} >= {threshold}"})
    return preds


def scan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train, test, _ = split_rows(rows)
    preds = build_predicates(rows)
    all_preds = preds[:]
    for a, b in itertools.combinations(preds, 2):
        all_preds.append({"label": f"{a['label']} AND {b['label']}", "expr": f"{a['expr']} AND {b['expr']}"})
    candidates = []
    for pred in all_preds:
        train_rows = [r for r in train if row_matches(r, pred["expr"])]
        if len(train_rows) < 10:
            continue
        candidates.append({"label": pred["label"], "expr": pred["expr"], "train": summarize(train_rows)})
    candidates.sort(key=lambda c: (c["train"]["median"] or -999.0, c["train"]["top3_removed_cum"], c["train"]["cum"]), reverse=True)
    out = []
    for rank, cand in enumerate(candidates[:10], start=1):
        out.append(
            {
                **cand,
                "rank": rank,
                "test": summarize([r for r in test if row_matches(r, cand["expr"])]),
                "all": summarize([r for r in rows if row_matches(r, cand["expr"])]),
            }
        )
    return out


def add_real_fill(candidates: list[dict[str, Any]], rows: list[dict[str, Any]], source_con: sqlite3.Connection) -> list[dict[str, Any]]:
    _, test, _ = split_rows(rows)
    for cand in candidates:
        selected = [r for r in rows if row_matches(r, cand["expr"])]
        selected_test = [r for r in test if row_matches(r, cand["expr"])]
        filled = [rf for r in selected if (rf := real_fill_net(source_con, r)) is not None]
        filled_test = [rf for r in selected_test if (rf := real_fill_net(source_con, r)) is not None]
        cand["real_fill"] = {
            "total_rows": len(selected),
            "real_fill_rows": len(filled),
            "no_fill_rows": len(selected) - len(filled),
            "no_fill_rate": (len(selected) - len(filled)) / len(selected) if selected else None,
            "test": summarize(filled_test, "real_net_bps"),
        }
    return candidates


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
        "# S34 Market Confluence Scan",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Scope: ETH BUY feature-factory events, route `LONG_DELAY0_TP60`.",
        "",
        "Confluence features use only BTC/SOL/ETH liquidation flow before the ETH event timestamp.",
        "",
        f"- Rows: `{payload['rows']}`",
        f"- Predicate count: `{payload['predicate_count']}`",
        "",
        "## OOS Candidates",
        "",
        "| Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        train = row["train"]
        test = row["test"]
        lines.append(
            f"| {row['rank']} | {row['label']} | {train['n']} | {fmt(train['median'])} | {fmt(train['cum'])} | "
            f"{test['n']} | {fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | "
            f"{fmt(test['top3_removed_cum'])} | {test['positive_days']}/{test['days']} |"
        )
    lines.extend(
        [
            "",
            "## Real-Fill Parity",
            "",
            "| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["candidates"]:
        rf = row["real_fill"]
        test = rf["test"]
        lines.append(
            f"| {row['label']} | {rf['total_rows']} | {rf['real_fill_rows']} | "
            f"{rf['no_fill_rows']} ({pct(rf['no_fill_rate'])}) | {test['n']} | {fmt(test['median'])} | "
            f"{fmt(test['mean'])} | {fmt(test['cum'])} | {fmt(test['top3_removed_cum'])} | "
            f"{test['positive_days']}/{test['days']} |"
        )
    lines.extend(["", "## Read", "", "This is a confluence research scan. Do not promote a candidate without separate forward pre-registration."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    source_con.execute("pragma query_only=1")
    rows = enrich(load_base_rows(feature_con), source_con)
    candidates = add_real_fill(scan(rows), rows, source_con)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(rows),
        "predicate_count": len(build_predicates(rows)) + len(list(itertools.combinations(build_predicates(rows), 2))),
        "candidates": candidates,
        "sample_rows": rows[:8],
    }
    write_report(payload)
    print(json.dumps({"rows": len(rows), "predicate_count": payload["predicate_count"], "out_md": str(OUT_MD)}, indent=2))
    feature_con.close()
    source_con.close()


if __name__ == "__main__":
    main()
