import json
import sqlite3
from pathlib import Path

from ami.storage import production as PR
from ami.storage import research_reader as RR

SOURCE_DB = "file:data/microstructure.db?mode=ro"
SOURCE_DB_PATH = "data/microstructure.db"
FEATURE_DB = "data/s34_feature_factory.db"
OOS_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_OOS_VALIDATION.json")
OUT_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_REAL_FILL_PARITY_TOP5.json")
OUT_MD = Path("reports/research/s34/S34_FEATURE_FACTORY_REAL_FILL_PARITY_TOP5.md")
MAX_BOOK_STALENESS_SEC = 5
TAKER_FEE_BPS = 4.0


def book_ticker_at(con: sqlite3.Connection, symbol: str, ts_ms: int):
    """Direct-SQL oracle -- kept as the parity reference for
    `book_ticker_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V5). No longer called by main(); the reader-
    backed path is used instead."""
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


def book_ticker_at_v2(root, symbol: str, ts_ms: int, source_db_path=None):
    """Reader-backed replacement for `book_ticker_at`, via
    lookup_latest_at_or_before. `symbol` is a genuine runtime parameter
    here -- this file's rows span ETHUSDT/BTCUSDT/SOLUSDT (confirmed via
    the feature-factory DB), so the helper is always called with
    whatever symbol the row actually names; it never assumes SOLUSDT's
    archive coverage applies to a different symbol. The staleness check
    stays a Python-side post-lookup guard (compares the found row's own
    ts_ms to the query ts, not a fixed filter value)."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=("ts_ms", "bid_price", "ask_price", "mid_price"), source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask, mid = result.row
    if int(ts_ms) - int(row_ts) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row_ts), "bid": float(bid), "ask": float(ask), "mid": float(mid)}


def pnl_bps(direction: str, entry: float, exit_: float):
    if direction.upper() == "LONG":
        return (exit_ - entry) / entry * 10000.0
    return (entry - exit_) / entry * 10000.0


def fill_quote(book: dict, direction: str, leg: str, reference_price: float):
    direction = direction.upper()
    leg = leg.upper()
    if direction == "LONG":
        fill_price = book["ask"] if leg == "ENTRY" else book["bid"]
    else:
        fill_price = book["bid"] if leg == "ENTRY" else book["ask"]
    return {
        "reference_price": float(reference_price),
        "bid": book["bid"],
        "ask": book["ask"],
        "mid": book["mid"],
        "fill_price": float(fill_price),
        "fee_bps": TAKER_FEE_BPS,
        "book_ts_ms": book["ts_ms"],
    }


def cost_decomposition(direction: str, entry_reference: float, exit_reference: float, entry_fill: dict, exit_fill: dict):
    basis = float(entry_reference)
    gross_bps = pnl_bps(direction, basis, float(exit_reference))
    entry_fill_px = float(entry_fill["fill_price"])
    exit_fill_px = float(exit_fill["fill_price"])
    entry_mid = float(entry_fill["mid"])
    exit_mid = float(exit_fill["mid"])
    executable_bps = pnl_bps(direction, entry_fill_px, exit_fill_px) * (entry_fill_px / basis)
    mid_to_mid_bps = pnl_bps(direction, entry_mid, exit_mid) * (entry_mid / basis)
    if direction.upper() == "LONG":
        entry_adverse_bps = (entry_mid - basis) / basis * 10000.0
        exit_adverse_bps = (float(exit_reference) - exit_mid) / basis * 10000.0
        entry_spread_bps = (entry_fill_px - entry_mid) / basis * 10000.0
        exit_spread_bps = (exit_mid - exit_fill_px) / basis * 10000.0
    else:
        entry_adverse_bps = (basis - entry_mid) / basis * 10000.0
        exit_adverse_bps = (exit_mid - float(exit_reference)) / basis * 10000.0
        entry_spread_bps = (entry_mid - entry_fill_px) / basis * 10000.0
        exit_spread_bps = (exit_fill_px - exit_mid) / basis * 10000.0
    spread_cost_bps = entry_spread_bps + exit_spread_bps
    fee_cost_bps = entry_fill["fee_bps"] + exit_fill["fee_bps"]
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net_bps = executable_bps - fee_cost_bps
    if abs(net_bps - executable_net_bps) > 1e-6:
        raise RuntimeError(f"identity mismatch net={net_bps} executable_net={executable_net_bps}")
    return {
        "gross_bps": gross_bps,
        "mid_to_mid_bps": mid_to_mid_bps,
        "executable_gross_bps": executable_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "net_bps": net_bps,
    }


def median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def summarize(rows):
    vals = [row["real_net_bps"] for row in rows]
    if not vals:
        return None
    simplified = [row["simplified_net_bps"] for row in rows]
    penalty = [row["fill_penalty_bps"] for row in rows]
    return {
        "n": len(vals),
        "mean_real_net": sum(vals) / len(vals),
        "median_real_net": median(vals),
        "cum_real_net": sum(vals),
        "wr_real": sum(v > 0 for v in vals) / len(vals),
        "mean_simplified_net": sum(simplified) / len(simplified),
        "median_simplified_net": median(simplified),
        "mean_fill_penalty": sum(penalty) / len(penalty),
        "median_fill_penalty": median(penalty),
        "mean_entry_adverse": sum(row["entry_adverse_bps"] for row in rows) / len(rows),
        "mean_exit_adverse": sum(row["exit_adverse_bps"] for row in rows) / len(rows),
        "mean_spread": sum(row["spread_cost_bps"] for row in rows) / len(rows),
        "mean_fee": sum(row["fee_cost_bps"] for row in rows) / len(rows),
        "tp": sum(row["exit_reason"] == "TP" for row in rows),
        "be": sum(row["exit_reason"] == "BE" for row in rows),
        "sl": sum(row["exit_reason"] == "SL" for row in rows),
        "time": sum(row["exit_reason"] == "TIME" for row in rows),
    }


def load_candidate_rows(feature_con: sqlite3.Connection, route_id: str, filter_label: str):
    # Filter labels were generated as SQL fragments without the leading f. alias in JSON.
    where = filter_label
    if where == "BASE":
        where_sql = "1=1"
    else:
        # Re-apply known safe column names by prefixing f. to identifiers via simple replacements.
        # Labels originate from our own query script, not user input.
        for col in [
            "cluster_notional",
            "cluster_count",
            "cluster_intensity_notional_per_sec",
            "btc_pre_15m_bps",
            "symbol_pre_5m_bps",
            "symbol_pre_15m_bps",
            "day_trend_bps",
            "day_range_bps",
            "day_buy_liq_notional",
            "day_agg_count",
        ]:
            where = where.replace(col, f"f.{col}")
        where_sql = where
    return feature_con.execute(
        f"""
        select
          f.event_id, f.symbol, f.event_ts_ms,
          l.route_id, l.direction, l.entry_ts_ms, l.entry_price,
          l.exit_ts_ms, l.exit_price, l.exit_reason, l.net_bps
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where l.route_id=? and ({where_sql})
        order by f.event_ts_ms
        """,
        (route_id,),
    ).fetchall()


def main():
    oos = json.loads(OOS_JSON.read_text(encoding="utf-8"))
    top = oos["surviving_candidates"][:5]
    root, _root_source = PR.resolve_production_root()
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=10)
    source_con.execute("pragma query_only=1")
    feature_con = sqlite3.connect(FEATURE_DB)

    results = []
    details = {}
    for candidate in top:
        route_id = candidate["route_id"]
        filter_label = candidate["filter"]
        raw_rows = load_candidate_rows(feature_con, route_id, filter_label)
        real_rows = []
        no_fill = 0
        for (
            event_id,
            symbol,
            event_ts_ms,
            _route_id,
            direction,
            entry_ts_ms,
            entry_price,
            exit_ts_ms,
            exit_price,
            exit_reason,
            simplified_net_bps,
        ) in raw_rows:
            entry_book = book_ticker_at_v2(root, symbol, entry_ts_ms, source_db_path=SOURCE_DB_PATH)
            exit_book = book_ticker_at_v2(root, symbol, exit_ts_ms, source_db_path=SOURCE_DB_PATH)
            if not entry_book or not exit_book:
                no_fill += 1
                continue
            entry_fill = fill_quote(entry_book, direction, "ENTRY", entry_price)
            exit_fill = fill_quote(exit_book, direction, "EXIT", exit_price)
            cost = cost_decomposition(direction, entry_price, exit_price, entry_fill, exit_fill)
            row = {
                "event_id": event_id,
                "symbol": symbol,
                "event_ts_ms": event_ts_ms,
                "route_id": route_id,
                "filter": filter_label,
                "exit_reason": exit_reason,
                "simplified_net_bps": float(simplified_net_bps),
                "real_net_bps": cost["net_bps"],
                "fill_penalty_bps": float(simplified_net_bps) - cost["net_bps"],
                **cost,
            }
            real_rows.append(row)
        summary = summarize(real_rows)
        if summary:
            summary.update(
                {
                    "route_id": route_id,
                    "filter": filter_label,
                    "total_rows": len(raw_rows),
                    "real_fill_rows": len(real_rows),
                    "no_fill_rows": no_fill,
                    "no_fill_rate": no_fill / len(raw_rows) if raw_rows else 0.0,
                    "oos_test": candidate["test"],
                }
            )
        else:
            summary = {
                "route_id": route_id,
                "filter": filter_label,
                "total_rows": len(raw_rows),
                "real_fill_rows": 0,
                "no_fill_rows": no_fill,
                "no_fill_rate": no_fill / len(raw_rows) if raw_rows else 0.0,
                "oos_test": candidate["test"],
            }
        results.append(summary)
        details[f"{route_id}::{filter_label}"] = real_rows

    payload = {"results": results, "details": details, "max_book_staleness_sec": MAX_BOOK_STALENESS_SEC}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Real-Fill Parity - Top 5 OOS Filters",
        "",
        "Scope: recompute top 5 OOS candidates with real historical `book_ticker` bid/ask fills where available.",
        "",
        "Caveat: `book_ticker` starts later than the full feature set, so older events are counted as `NO_FILL_DATA` and excluded from real-fill metrics. No modeled spread fallback is used.",
        "",
        "| Rank | Route | Filter | Total | Real Fill | No Fill | OOS Test Median | Real Median | Real Mean | Real Cum | Fill Penalty Mean | Entry Adv | Exit Adv | Spread | Fee |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(results, 1):
        lines.append(
            f"| {idx} | {row['route_id']} | {row['filter']} | {row['total_rows']} | "
            f"{row['real_fill_rows']} | {row['no_fill_rows']} | "
            f"{row['oos_test']['median']:+.2f} | "
            f"{row.get('median_real_net', 0):+.2f} | {row.get('mean_real_net', 0):+.2f} | "
            f"{row.get('cum_real_net', 0):+.2f} | {row.get('mean_fill_penalty', 0):+.2f} | "
            f"{row.get('mean_entry_adverse', 0):+.2f} | {row.get('mean_exit_adverse', 0):+.2f} | "
            f"{row.get('mean_spread', 0):+.2f} | {row.get('mean_fee', 0):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "A candidate passes this gate only if real-fill median remains positive, mean remains positive, and no-fill coverage is not the dominant sample. This is still historical replay, not live paper validation.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(results, indent=2))
    source_con.close()
    feature_con.close()


if __name__ == "__main__":
    main()
