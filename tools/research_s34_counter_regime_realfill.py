from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from ami.storage import production as PR
from ami.storage import research_reader as RR

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_COUNTER_REGIME_REAL_FILL.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_COUNTER_REGIME_REAL_FILL.md"

ROUTE_ID = "LONG_DELAY0_TP60"
SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
MAX_BOOK_STALENESS_SEC = 5
TAKER_FEE_BPS = 4.0

CANDIDATES = {
    "500K_daytrend_negative": "cluster_notional>=500000 AND day_trend_bps<0",
    "500K_neg_count_ge10": "cluster_notional>=500000 AND day_trend_bps<0 AND cluster_liq_count>=10",
    "500K_neg_count_ge15": "cluster_notional>=500000 AND day_trend_bps<0 AND cluster_liq_count>=15",
    "500K_neg_count_ge20": "cluster_notional>=500000 AND day_trend_bps<0 AND cluster_liq_count>=20",
    "500K_neg_count_ge22": "cluster_notional>=500000 AND day_trend_bps<0 AND cluster_liq_count>=22",
    "500K_neg_stretched": "cluster_notional>=500000 AND day_trend_bps<0 AND shape_label='stretched_120s'",
    "500K_neg_stretched_count_ge15": "cluster_notional>=500000 AND day_trend_bps<0 AND shape_label='stretched_120s' AND cluster_liq_count>=15",
    "500K_neg_stretched_count_ge22": "cluster_notional>=500000 AND day_trend_bps<0 AND shape_label='stretched_120s' AND cluster_liq_count>=22",
}


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def signed_ret(entry: float, exit_: float) -> float:
    return (float(exit_) - float(entry)) / float(entry) * 10000.0


def day(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).date().isoformat()


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    day_cums: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.get(key) is not None:
            day_cums[day(int(row["event_ts_ms"]))] += float(row[key])
    if not vals:
        return {
            "n": 0,
            "days": 0,
            "cum": 0.0,
            "mean": None,
            "median": None,
            "wr": None,
            "top3_removed": 0.0,
            "positive_days": 0,
            "exit_counts": {},
        }
    return {
        "n": len(vals),
        "days": len(day_cums),
        "cum": sum(vals),
        "mean": statistics.mean(vals),
        "median": median(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "exit_counts": dict(sorted((row["exit_reason"], sum(1 for r in rows if r["exit_reason"] == row["exit_reason"])) for row in rows)),
    }


def book_at(con: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, float] | None:
    """Direct-SQL oracle -- kept as the parity reference for `book_at_v2`
    (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V5).
    No longer called by main(); the reader-backed path is used instead."""
    row = con.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
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
    if age_ms > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def book_at_v2(root, symbol: str, ts_ms: int, source_db_path=None) -> dict[str, float] | None:
    """Reader-backed replacement for `book_at`, via
    lookup_latest_at_or_before. `symbol` stays a genuine parameter (not
    hardcoded to the module's `SYMBOL` constant here) so the helper is
    never accidentally called with the wrong symbol, even though every
    real call site in this file passes `row["symbol"]`, which -- because
    `load_candidate`'s query always filters `f.symbol=?` bound to the
    module constant `SYMBOL="ETHUSDT"` -- is always "ETHUSDT" in
    practice (book_ticker has no ETHUSDT archive partition; only
    SOLUSDT is archived, so real production use of this file resolves
    SQLITE_ONLY -- confirmed, not assumed)."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=("ts_ms", "bid_price", "ask_price", "mid_price"), source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask, mid = result.row
    if int(ts_ms) - int(row_ts) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row_ts), "bid": float(bid), "ask": float(ask), "mid": float(mid)}


def real_fill_row(root, row: dict[str, Any], source_db_path=None) -> dict[str, Any] | None:
    entry_book = book_at_v2(root, row["symbol"], int(row["entry_ts_ms"]), source_db_path=source_db_path)
    exit_book = book_at_v2(root, row["symbol"], int(row["exit_ts_ms"]), source_db_path=source_db_path)
    if not entry_book or not exit_book:
        return None
    basis = float(row["entry_price"])
    exit_ref = float(row["exit_price"])
    entry_fill = float(entry_book["ask"])
    exit_fill = float(exit_book["bid"])
    entry_mid = float(entry_book["mid"])
    exit_mid = float(exit_book["mid"])
    gross_bps = signed_ret(basis, exit_ref)
    entry_adverse_bps = (entry_mid - basis) / basis * 10000.0
    exit_adverse_bps = (exit_ref - exit_mid) / basis * 10000.0
    spread_cost_bps = ((entry_fill - entry_mid) + (exit_mid - exit_fill)) / basis * 10000.0
    fee_cost_bps = TAKER_FEE_BPS * 2.0
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net = signed_ret(entry_fill, exit_fill) * (entry_fill / basis) - fee_cost_bps
    if abs(net_bps - executable_net) > 1e-6:
        raise RuntimeError(f"cost identity mismatch: {net_bps} != {executable_net}")
    return {
        **row,
        "real_net_bps": net_bps,
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "fill_penalty_bps": float(row["simplified_net_bps"]) - net_bps,
    }


def load_candidate(feature_con: sqlite3.Connection, where_sql: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in feature_con.execute(
            f"""
            SELECT
                f.event_id, f.symbol, f.event_ts_ms, f.event_utc,
                f.cluster_notional, f.cluster_liq_count, f.shape_label,
                f.day_trend_bps, f.day_range_bps, f.day_buy_liq_notional,
                o.route_id, o.direction, o.entry_ts_ms, o.entry_price,
                o.exit_ts_ms, o.exit_price, o.exit_reason,
                o.net_bps AS simplified_net_bps,
                o.mfe_bps, o.mae_bps, o.time_to_mfe_sec
            FROM liq_event_features f
            JOIN liq_event_outcome_labels o ON o.event_id=f.event_id
            WHERE f.symbol=? AND f.liq_side=? AND o.route_id=? AND ({where_sql})
            ORDER BY f.event_ts_ms
            """,
            (SYMBOL, LIQ_SIDE, ROUTE_ID),
        )
    ]


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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


def main() -> int:
    feature_con = sqlite3.connect(FEATURE_DB)
    feature_con.row_factory = sqlite3.Row
    root, _root_source = PR.resolve_production_root()

    results = []
    details: dict[str, list[dict[str, Any]]] = {}
    for name, where_sql in CANDIDATES.items():
        raw = load_candidate(feature_con, where_sql)
        real = []
        no_fill = 0
        for row in raw:
            filled = real_fill_row(root, row, source_db_path=SOURCE_DB_PATH)
            if filled is None:
                no_fill += 1
            else:
                real.append(filled)
        split_ts = raw[len(raw) // 2]["event_ts_ms"] if raw else None
        real_test = [row for row in real if split_ts is not None and int(row["event_ts_ms"]) >= int(split_ts)]
        result = {
            "name": name,
            "where": where_sql,
            "total": len(raw),
            "real_fill": len(real),
            "no_fill": no_fill,
            "no_fill_rate": no_fill / len(raw) if raw else None,
            "simplified": summarize(raw, "simplified_net_bps"),
            "real": summarize(real, "real_net_bps"),
            "real_test_half": summarize(real_test, "real_net_bps"),
            "mean_fill_penalty": statistics.mean([row["fill_penalty_bps"] for row in real]) if real else None,
            "median_fill_penalty": median([row["fill_penalty_bps"] for row in real]),
            "mean_entry_adverse": statistics.mean([row["entry_adverse_bps"] for row in real]) if real else None,
            "median_entry_adverse": median([row["entry_adverse_bps"] for row in real]),
        }
        results.append(result)
        details[name] = real

    feature_con.close()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route_id": ROUTE_ID,
        "symbol": SYMBOL,
        "liq_side": LIQ_SIDE,
        "candidates": results,
        "details": details,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# S34 Counter-Regime Real-Fill Test", ""]
    md.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    md.append(f"- route_id: `{ROUTE_ID}`")
    md.append("- scope: ETH BUY liquidation, LONG delay0 TP60/SL40/BE30, day_trend_bps < 0 counter-regime candidates")
    md.append("- live runner/config changes: `none`")
    md.append("")
    md.append("## Candidate Results")
    rows = [[
        "candidate", "total", "real", "no-fill", "real median", "real mean", "real cum",
        "real WR", "test-half N", "test-half median", "test-half cum", "fill penalty med",
    ]]
    for item in results:
        real = item["real"]
        test = item["real_test_half"]
        rows.append([
            item["name"],
            item["total"],
            item["real_fill"],
            f"{item['no_fill']} ({fmt((item['no_fill_rate'] or 0) * 100)}%)",
            fmt(real["median"]),
            fmt(real["mean"]),
            fmt(real["cum"]),
            fmt(None if real["wr"] is None else real["wr"] * 100.0) + "%",
            test["n"],
            fmt(test["median"]),
            fmt(test["cum"]),
            fmt(item["median_fill_penalty"]),
        ])
    md += render_table(rows)
    md.append("")
    md.append("## Read")
    md.append("")
    md.append(
        "This is still research. A candidate surviving real-fill here is eligible for a separate exploratory paper rule only after explicit pre-registration; it does not change the current pre-reg S34 sample."
    )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    for item in results:
        real = item["real"]
        test = item["real_test_half"]
        print(
            item["name"],
            "total", item["total"],
            "real", item["real_fill"],
            "nofill", item["no_fill"],
            "median", fmt(real["median"]),
            "cum", fmt(real["cum"]),
            "testN", test["n"],
            "testMed", fmt(test["median"]),
            "testCum", fmt(test["cum"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
