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
OUT_JSON = Path("reports/research/s34/S34_CLUSTER_GEOMETRY_FEATURES.json")
OUT_MD = Path("reports/research/s34/S34_CLUSTER_GEOMETRY_FEATURES.md")

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
ROUTE_ID = "LONG_DELAY0_TP60"
MAX_BOOK_STALENESS_SEC = 5
TAKER_FEE_BPS = 4.0


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


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


def signed_ret(entry: float, exit_: float) -> float:
    return (float(exit_) - float(entry)) / float(entry) * 10000.0


def book_ticker_at(con: sqlite3.Connection, ts_ms: int):
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_ticker_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V8). No longer called by `real_fill_net`; the
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
        "real_net_bps": net_bps,
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
    }


def ensure_columns(con: sqlite3.Connection) -> dict[str, str]:
    existing = {row[1] for row in con.execute("pragma table_info(liq_event_features)").fetchall()}
    desired = {
        "cluster_liq_count": "INTEGER",
        "max_single_liq_share": "REAL",
        "intensity_per_sec": "REAL",
        "inter_cluster_gap_sec": "REAL",
        "shape_label": "TEXT",
    }
    added = {}
    for col, typ in desired.items():
        if col not in existing:
            con.execute(f"alter table liq_event_features add column {col} {typ}")
            added[col] = typ
    return added


def shape_label(duration_sec: float, max_share: float) -> str:
    if max_share >= 80.0:
        return "single_dominant_80pct"
    if duration_sec >= 120.0:
        return "stretched_120s"
    return "distributed_mid_duration"


def populate_geometry(con: sqlite3.Connection) -> int:
    rows = con.execute(
        """
        select event_id, cluster_duration_sec, cluster_count, cluster_notional,
               cluster_max_notional, cluster_intensity_notional_per_sec,
               inter_kept_gap_sec, inter_candidate_gap_sec
        from liq_event_features
        where symbol=? and liq_side=?
        """,
        (SYMBOL, LIQ_SIDE),
    ).fetchall()
    updates = []
    for row in rows:
        event_id, duration, count, notional, max_notional, existing_intensity, kept_gap, candidate_gap = row
        duration = float(duration or 0.0)
        notional = float(notional or 0.0)
        max_notional = float(max_notional or 0.0)
        max_share = (max_notional / notional * 100.0) if notional > 0 else None
        intensity = float(existing_intensity) if existing_intensity is not None else (notional / max(duration, 1.0))
        gap = kept_gap if kept_gap is not None else candidate_gap
        label = shape_label(duration, float(max_share or 0.0))
        updates.append((int(count or 0), max_share, intensity, gap, label, event_id))
    con.executemany(
        """
        update liq_event_features
        set cluster_liq_count=?,
            max_single_liq_share=?,
            intensity_per_sec=?,
            inter_cluster_gap_sec=?,
            shape_label=?
        where event_id=?
        """,
        updates,
    )
    con.commit()
    return len(updates)


def load_joined_rows(con: sqlite3.Connection, base_where: str) -> list[dict]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"""
        select
          f.event_id, f.event_ts_ms, date(f.event_ts_ms/1000, 'unixepoch') as day,
          f.cluster_notional, f.day_trend_bps, f.cluster_duration_sec,
          f.cluster_liq_count, f.max_single_liq_share, f.intensity_per_sec,
          f.inter_cluster_gap_sec, f.shape_label,
          l.route_id, l.entry_ts_ms, l.entry_price, l.exit_ts_ms, l.exit_price,
          l.exit_reason, l.net_bps, l.mfe_bps, l.mae_bps
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where f.symbol=? and f.liq_side=? and l.route_id=? and ({base_where})
        order by f.event_ts_ms
        """,
        (SYMBOL, LIQ_SIDE, ROUTE_ID),
    ).fetchall()
    return [dict(row) for row in rows]


def split_rows(rows: list[dict], split_ts_ms: int) -> dict[str, list[dict]]:
    return {
        "train": [row for row in rows if int(row["event_ts_ms"]) <= split_ts_ms],
        "test": [row for row in rows if int(row["event_ts_ms"]) > split_ts_ms],
        "all": rows,
    }


def build_predicates(all_rows: list[dict]) -> list[dict]:
    intensities = [float(r["intensity_per_sec"]) for r in all_rows if r.get("intensity_per_sec") is not None]
    counts = [float(r["cluster_liq_count"]) for r in all_rows if r.get("cluster_liq_count") is not None]
    durations = [float(r["cluster_duration_sec"]) for r in all_rows if r.get("cluster_duration_sec") is not None]
    p25_intensity = quantile(intensities, 0.25)
    p75_intensity = quantile(intensities, 0.75)
    p75_count = quantile(counts, 0.75)
    p25_count = quantile(counts, 0.25)
    p75_duration = quantile(durations, 0.75)
    preds = [
        {"label": "max_share_ge_80", "feature": "max_single_liq_share", "sql": "max_single_liq_share >= 80"},
        {"label": "max_share_lt_50", "feature": "max_single_liq_share", "sql": "max_single_liq_share < 50"},
        {"label": f"intensity_ge_p75_{p75_intensity:.0f}", "feature": "intensity_per_sec", "sql": f"intensity_per_sec >= {p75_intensity}"},
        {"label": f"intensity_le_p25_{p25_intensity:.0f}", "feature": "intensity_per_sec", "sql": f"intensity_per_sec <= {p25_intensity}"},
        {"label": "gap_lt_30m", "feature": "inter_cluster_gap_sec", "sql": "inter_cluster_gap_sec < 1800"},
        {"label": "gap_ge_2h", "feature": "inter_cluster_gap_sec", "sql": "inter_cluster_gap_sec >= 7200"},
        {"label": f"liq_count_ge_p75_{p75_count:.0f}", "feature": "cluster_liq_count", "sql": f"cluster_liq_count >= {p75_count}"},
        {"label": f"liq_count_le_p25_{p25_count:.0f}", "feature": "cluster_liq_count", "sql": f"cluster_liq_count <= {p25_count}"},
        {"label": f"duration_ge_p75_{p75_duration:.0f}s", "feature": "cluster_duration_sec", "sql": f"cluster_duration_sec >= {p75_duration}"},
    ]
    for label in ["single_dominant_80pct", "stretched_120s", "distributed_mid_duration"]:
        preds.append({"label": f"shape_{label}", "feature": "shape_label", "sql": f"shape_label = '{label}'"})
    return preds


def row_matches(row: dict, sql: str) -> bool:
    # Predicate strings are generated by this script; keep evaluator intentionally narrow.
    if " and " in sql:
        return all(row_matches(row, part.strip()) for part in sql.split(" and "))
    if " = '" in sql:
        col, val = sql.split(" = '", 1)
        return str(row.get(col.strip())) == val.rstrip("'")
    for op in [">=", "<=", "<", ">"]:
        if op in sql:
            col, val = sql.split(op, 1)
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
    raise ValueError(f"unsupported predicate {sql}")


def apply_predicate(rows: list[dict], sql: str) -> list[dict]:
    return [row for row in rows if row_matches(row, sql)]


def scan_scope(name: str, rows: list[dict], predicates: list[dict], split_ts_ms: int) -> dict:
    candidates = [{"label": pred["label"], "sql": pred["sql"], "features": [pred["feature"]]} for pred in predicates]
    for left, right in itertools.combinations(predicates, 2):
        if left["feature"] == right["feature"]:
            continue
        candidates.append(
            {
                "label": f"{left['label']} AND {right['label']}",
                "sql": f"{left['sql']} and {right['sql']}",
                "features": sorted([left["feature"], right["feature"]]),
            }
        )
    out = []
    for cand in candidates:
        filtered = apply_predicate(rows, cand["sql"])
        periods = split_rows(filtered, split_ts_ms)
        train = summarize(periods["train"])
        test = summarize(periods["test"])
        out.append({**cand, "train": train, "test": test, "all": summarize(periods["all"])})
    ranked = sorted(
        [
            row
            for row in out
            if row["train"]["n"] >= 15
            and row["train"]["days"] >= 4
            and row["train"]["median"] is not None
            and row["train"]["median"] > 0
            and row["train"]["top3_removed_cum"] > 0
        ],
        key=lambda row: (
            row["test"]["n"] >= 10,
            row["test"]["median"] if row["test"]["median"] is not None else -1e9,
            row["test"]["top3_removed_cum"],
            row["test"]["mean"] if row["test"]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )
    return {"scope": name, "evaluated": len(out), "candidates": out, "top": ranked[:5]}


def real_fill_for_candidates(root, scans: list[dict], source_db_path=None) -> dict:
    result = {}
    for scan in scans:
        for cand in scan["top"]:
            key = f"{scan['scope']}::{cand['label']}"
            rows = apply_predicate(scan["rows"], cand["sql"])
            filled = []
            no_fill = 0
            for row in rows:
                real = real_fill_net(root, row, source_db_path=source_db_path)
                if not real:
                    no_fill += 1
                    continue
                filled.append(real)
            periods = split_rows(filled, scan["split_ts_ms"])
            result[key] = {
                "scope": scan["scope"],
                "label": cand["label"],
                "sql": cand["sql"],
                "total_rows": len(rows),
                "real_fill_rows": len(filled),
                "no_fill_rows": no_fill,
                "no_fill_rate": no_fill / len(rows) if rows else None,
                "train": summarize(periods["train"], key="real_net_bps"),
                "test": summarize(periods["test"], key="real_net_bps"),
                "all": summarize(periods["all"], key="real_net_bps"),
            }
    return result


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    added = ensure_columns(feature_con)
    updated = populate_geometry(feature_con)

    min_ts, max_ts = feature_con.execute("select min(event_ts_ms), max(event_ts_ms) from liq_event_features").fetchone()
    split_ts_ms = int((int(min_ts) + int(max_ts)) / 2)

    all_rows = load_joined_rows(feature_con, "1=1")
    daytrend_rows = load_joined_rows(feature_con, "cluster_notional >= 500000 and day_trend_bps >= 0")
    predicates = build_predicates(all_rows)
    scans = [
        scan_scope("LONG_DELAY0_TP60_ALL_200K", all_rows, predicates, split_ts_ms),
        scan_scope("LONG_DELAY0_TP60_500K_DAYTREND", daytrend_rows, predicates, split_ts_ms),
    ]
    for scan, rows in zip(scans, [all_rows, daytrend_rows]):
        scan["rows"] = rows
        scan["split_ts_ms"] = split_ts_ms
    # Every book_ticker point read now goes through the reader (via
    # SOURCE_DB_PATH), so the old direct `source_con` connection to the
    # 778GB source DB is fully dead here and is not opened (mirrors the
    # ASOF Batch 3 preliq_detector dead-connection drop).
    root, _ = PR.resolve_production_root()
    real_fill = real_fill_for_candidates(root, scans, source_db_path=SOURCE_DB_PATH)

    examples = [
        dict(row)
        for row in feature_con.execute(
            """
            select event_id, event_utc, cluster_notional, cluster_duration_sec,
                   cluster_liq_count, max_single_liq_share, intensity_per_sec,
                   inter_cluster_gap_sec, shape_label
            from liq_event_features
            where symbol=? and liq_side=?
            order by event_ts_ms
            limit 8
            """,
            (SYMBOL, LIQ_SIDE),
        )
    ]
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema_update": {"added_columns": added, "updated_rows": updated},
        "scope": {
            "feature_db": FEATURE_DB,
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "route_id": ROUTE_ID,
            "split_ts_ms": split_ts_ms,
            "split_utc": dt.datetime.fromtimestamp(split_ts_ms / 1000, tz=dt.timezone.utc).isoformat(),
            "lookahead_note": "Geometry features are computed from cluster-internal data available at/after cluster completion; outcome labels remain separate.",
        },
        "examples": examples,
        "predicate_count": len(predicates),
        "scans": [{k: v for k, v in scan.items() if k not in {"rows", "split_ts_ms"}} for scan in scans],
        "real_fill": real_fill,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Cluster Geometry Feature Scan",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: add no-lookahead cluster geometry fields to `liq_event_features`, then scan relationship with `LONG_DELAY0_TP60` outcomes.",
        "",
        "No live runner/config changes. `liq_event_outcome_labels` was not modified.",
        "",
        "## 1. Schema / Fill Check",
        "",
        f"- Added columns: `{added}`",
        f"- Updated ETH BUY rows: `{updated}`",
        f"- Predicate count: `{len(predicates)}`",
        "",
        "| event_id | utc | notional | duration | count | max_share | intensity/sec | gap_sec | shape |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in examples:
        lines.append(
            f"| {row['event_id']} | {row['event_utc']} | {row['cluster_notional']:.0f} | "
            f"{row['cluster_duration_sec']:.1f} | {row['cluster_liq_count']} | "
            f"{row['max_single_liq_share']:.1f}% | {row['intensity_per_sec']:.0f} | "
            f"{'NA' if row['inter_cluster_gap_sec'] is None else f'{row['inter_cluster_gap_sec']:.1f}'} | {row['shape_label']} |"
        )
    lines.extend(
        [
            "",
            "## 2. OOS Geometry Candidates",
            "",
            "| Scope | Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scan in scans:
        for idx, row in enumerate(scan["top"], 1):
            tr = row["train"]
            te = row["test"]
            lines.append(
                f"| {scan['scope']} | {idx} | {row['label']} | {tr['n']} | {fmt(tr['median'])} | {fmt(tr['cum'])} | "
                f"{te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | {fmt(te['cum'])} | "
                f"{fmt(te['top3_removed_cum'])} | {te['positive_days']}/{te['days']} |"
            )
    lines.extend(
        [
            "",
            "## 3. Real-Fill Parity For Top Candidates",
            "",
            "| Scope | Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in real_fill.values():
        te = row["test"]
        lines.append(
            f"| {row['scope']} | {row['label']} | {row['total_rows']} | {row['real_fill_rows']} | "
            f"{row['no_fill_rows']} ({row['no_fill_rate']*100:.1f}%) | {te['n']} | {fmt(te['median'])} | "
            f"{fmt(te['mean'])} | {fmt(te['cum'])} | {fmt(te['top3_removed_cum'])} | {te['positive_days']}/{te['days']} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "These are geometry-only retrospective filters selected from the same feature surface. Treat them as hypothesis seeds unless they survive a separately pre-registered forward test.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({"schema_update": payload["schema_update"], "top": [{s["scope"]: s["top"][:3]} for s in scans], "real_fill": real_fill}, indent=2)[:8000])
    feature_con.close()


if __name__ == "__main__":
    main()
