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
OUT_JSON = Path("reports/research/s34/S34_CASCADE_PHASE_SCAN.json")
OUT_MD = Path("reports/research/s34/S34_CASCADE_PHASE_SCAN.md")

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
ROUTE_ID = "LONG_DELAY0_TP60"
TAKER_FEE_BPS = 4.0
MAX_BOOK_STALENESS_SEC = 5


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


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iso_day(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).date().isoformat()


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


def signed_ret(entry: float, exit_: float) -> float:
    return (float(exit_) - float(entry)) / float(entry) * 10000.0


def book_ticker_at(con: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, float] | None:
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


def real_fill_net(source_con: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any] | None:
    entry_book = book_ticker_at(source_con, SYMBOL, int(row["entry_ts_ms"]))
    exit_book = book_ticker_at(source_con, SYMBOL, int(row["exit_ts_ms"]))
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


def load_base_rows(feature_con: sqlite3.Connection) -> list[dict[str, Any]]:
    feature_con.row_factory = sqlite3.Row
    rows = feature_con.execute(
        """
        select
          f.event_id, f.event_ts_ms, f.event_utc, date(f.event_ts_ms/1000, 'unixepoch') as day,
          f.cluster_notional, f.cluster_duration_sec, f.cluster_count, f.cluster_max_notional,
          f.cluster_intensity_notional_per_sec, f.day_trend_bps, f.day_range_bps,
          f.max_single_liq_share, f.shape_label,
          l.route_id, l.entry_ts_ms, l.entry_price, l.exit_ts_ms, l.exit_price,
          l.exit_reason, l.net_bps, l.mfe_bps, l.mae_bps, l.time_to_mfe_sec
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where f.symbol=? and f.liq_side=? and l.route_id=?
        order by f.event_ts_ms
        """,
        (SYMBOL, LIQ_SIDE, ROUTE_ID),
    ).fetchall()
    return [dict(row) for row in rows]


def raw_liq_stats(source_con: sqlite3.Connection, event_ts_ms: int, window_sec: int) -> dict[str, float]:
    row = source_con.execute(
        """
        select count(*), coalesce(sum(notional), 0.0), coalesce(max(notional), 0.0)
        from liquidations
        where symbol=? and side=? and ts_ms>=? and ts_ms<?
        """,
        (SYMBOL, LIQ_SIDE, int(event_ts_ms) - int(window_sec) * 1000, int(event_ts_ms)),
    ).fetchone()
    count = int(row[0] or 0)
    notional = float(row[1] or 0.0)
    max_notional = float(row[2] or 0.0)
    return {
        "count": count,
        "notional": notional,
        "max_notional": max_notional,
        "max_share": (max_notional / notional * 100.0) if notional > 0 else 0.0,
    }


def enrich_phase(rows: list[dict[str, Any]], source_con: sqlite3.Connection) -> list[dict[str, Any]]:
    event_ts = [int(row["event_ts_ms"]) for row in rows]
    out = []
    for idx, row in enumerate(rows):
        ts = int(row["event_ts_ms"])
        prior15 = raw_liq_stats(source_con, ts, 900)
        prior30 = raw_liq_stats(source_con, ts, 1800)
        prior60 = raw_liq_stats(source_con, ts, 3600)
        cluster_15 = sum(1 for prev_ts in event_ts[:idx] if ts - prev_ts <= 900_000)
        cluster_30 = sum(1 for prev_ts in event_ts[:idx] if ts - prev_ts <= 1_800_000)
        prev_gap_sec = None if idx == 0 else (ts - event_ts[idx - 1]) / 1000.0
        current = float(row.get("cluster_notional") or 0.0)
        phase_pressure_15m = prior15["notional"] / max(current, 1.0)
        phase_pressure_30m = prior30["notional"] / max(current, 1.0)
        current_share_15m = current / max(current + prior15["notional"], 1.0) * 100.0
        if prior15["notional"] < 500_000 and cluster_15 == 0:
            phase_label = "fresh_start"
        elif current_share_15m >= 50.0:
            phase_label = "early_impulse"
        elif prior15["notional"] >= 3.0 * max(current, 1.0) or cluster_15 >= 3:
            phase_label = "late_saturated"
        else:
            phase_label = "mid_cascade"
        out.append(
            {
                **row,
                "prior15_buy_liq_notional": prior15["notional"],
                "prior30_buy_liq_notional": prior30["notional"],
                "prior60_buy_liq_notional": prior60["notional"],
                "prior15_buy_liq_count": prior15["count"],
                "prior30_buy_liq_count": prior30["count"],
                "prior15_cluster_count": cluster_15,
                "prior30_cluster_count": cluster_30,
                "prev_cluster_gap_sec": prev_gap_sec,
                "phase_pressure_15m": phase_pressure_15m,
                "phase_pressure_30m": phase_pressure_30m,
                "current_share_15m": current_share_15m,
                "phase_label": phase_label,
            }
        )
    return out


def row_matches(row: dict[str, Any], expr: str) -> bool:
    if " AND " in expr:
        return all(row_matches(row, part.strip()) for part in expr.split(" AND "))
    if " = '" in expr:
        col, val = expr.split(" = '", 1)
        return str(row.get(col.strip())) == val.rstrip("'")
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
    raise ValueError(f"unsupported predicate {expr}")


def build_predicates(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    p75_pressure15 = quantile([float(r["phase_pressure_15m"]) for r in rows], 0.75)
    p25_pressure15 = quantile([float(r["phase_pressure_15m"]) for r in rows], 0.25)
    p75_prior15 = quantile([float(r["prior15_buy_liq_notional"]) for r in rows], 0.75)
    p25_prior15 = quantile([float(r["prior15_buy_liq_notional"]) for r in rows], 0.25)
    preds = [
        {"label": "phase_fresh_start", "expr": "phase_label = 'fresh_start'"},
        {"label": "phase_early_impulse", "expr": "phase_label = 'early_impulse'"},
        {"label": "phase_mid_cascade", "expr": "phase_label = 'mid_cascade'"},
        {"label": "phase_late_saturated", "expr": "phase_label = 'late_saturated'"},
        {"label": "current_share_15m_ge_50", "expr": "current_share_15m >= 50"},
        {"label": "current_share_15m_ge_70", "expr": "current_share_15m >= 70"},
        {"label": "prior15_notional_lt_500k", "expr": "prior15_buy_liq_notional < 500000"},
        {"label": "prior15_notional_lt_1m", "expr": "prior15_buy_liq_notional < 1000000"},
        {"label": "prior15_notional_ge_3m", "expr": "prior15_buy_liq_notional >= 3000000"},
        {"label": "prior15_clusters_eq_0", "expr": "prior15_cluster_count <= 0"},
        {"label": "prior15_clusters_ge_3", "expr": "prior15_cluster_count >= 3"},
        {"label": f"pressure15_ge_p75_{p75_pressure15:.2f}", "expr": f"phase_pressure_15m >= {p75_pressure15}"},
        {"label": f"pressure15_le_p25_{p25_pressure15:.2f}", "expr": f"phase_pressure_15m <= {p25_pressure15}"},
        {"label": f"prior15_ge_p75_{p75_prior15:.0f}", "expr": f"prior15_buy_liq_notional >= {p75_prior15}"},
        {"label": f"prior15_le_p25_{p25_prior15:.0f}", "expr": f"prior15_buy_liq_notional <= {p25_prior15}"},
    ]
    return preds


def split_by_time(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    ts = sorted(int(r["event_ts_ms"]) for r in rows)
    split_ts = ts[len(ts) // 2]
    return [r for r in rows if int(r["event_ts_ms"]) <= split_ts], [r for r in rows if int(r["event_ts_ms"]) > split_ts], split_ts


def evaluate_predicates(rows: list[dict[str, Any]], predicates: list[dict[str, str]]) -> list[dict[str, Any]]:
    train, test, split_ts = split_by_time(rows)
    candidates = []
    scans = []
    all_preds = predicates[:]
    for a, b in itertools.combinations(predicates, 2):
        all_preds.append({"label": f"{a['label']} AND {b['label']}", "expr": f"{a['expr']} AND {b['expr']}"})
    for pred in all_preds:
        train_rows = [r for r in train if row_matches(r, pred["expr"])]
        if len(train_rows) < 10:
            continue
        train_summary = summarize(train_rows)
        scans.append({"predicate": pred, "train": train_summary})
    scans.sort(key=lambda r: (r["train"]["median"] or -999.0, r["train"]["top3_removed_cum"], r["train"]["cum"]), reverse=True)
    for rank, item in enumerate(scans[:8], start=1):
        pred = item["predicate"]
        test_rows = [r for r in test if row_matches(r, pred["expr"])]
        all_rows = [r for r in rows if row_matches(r, pred["expr"])]
        candidates.append(
            {
                "rank": rank,
                "label": pred["label"],
                "expr": pred["expr"],
                "train": item["train"],
                "test": summarize(test_rows),
                "all": summarize(all_rows),
            }
        )
    return candidates


def add_real_fill(candidates: list[dict[str, Any]], rows: list[dict[str, Any]], source_con: sqlite3.Connection) -> list[dict[str, Any]]:
    _, test, _ = split_by_time(rows)
    out = []
    for candidate in candidates:
        selected = [r for r in rows if row_matches(r, candidate["expr"])]
        selected_test = [r for r in test if row_matches(r, candidate["expr"])]
        filled = [rf for r in selected if (rf := real_fill_net(source_con, r)) is not None]
        filled_test = [rf for r in selected_test if (rf := real_fill_net(source_con, r)) is not None]
        out.append(
            {
                **candidate,
                "real_fill": {
                    "total_rows": len(selected),
                    "real_fill_rows": len(filled),
                    "no_fill_rows": len(selected) - len(filled),
                    "no_fill_rate": (len(selected) - len(filled)) / len(selected) if selected else None,
                    "test": summarize(filled_test, "real_net_bps"),
                },
            }
        )
    return out


def write_report(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# S34 Cascade Phase Scan",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Scope: ETH BUY feature-factory events, route `LONG_DELAY0_TP60`. No live runner/config changes.",
        "",
        "Phase features are no-lookahead: only liquidation flow before the cluster timestamp is used.",
        "",
        "## Phase Label Distribution",
        "",
        "| Phase | N | Median | Mean | Cum | WR | Top3 Removed | Days | Exits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, summary in payload["phase_distribution"].items():
        lines.append(
            f"| {label} | {summary['n']} | {fmt(summary['median'])} | {fmt(summary['mean'])} | "
            f"{fmt(summary['cum'])} | {pct(summary['wr'])} | {fmt(summary['top3_removed_cum'])} | "
            f"{summary['days']} | {summary['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## OOS Phase Candidates",
            "",
            "| Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["candidates"]:
        test = row["test"]
        train = row["train"]
        lines.append(
            f"| {row['rank']} | {row['label']} | {train['n']} | {fmt(train['median'])} | {fmt(train['cum'])} | "
            f"{test['n']} | {fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | "
            f"{fmt(test['top3_removed_cum'])} | {test['positive_days']}/{test['days']} |"
        )
    lines.extend(
        [
            "",
            "## Real-Fill Parity For Top Candidates",
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
    lines.extend(
        [
            "",
            "## Read",
            "",
            "This is a research scan over phase predicates and predicate pairs. Treat positives as hypothesis seeds, not live-rule proof.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.2f}"


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def main() -> None:
    feature_con = sqlite3.connect(FEATURE_DB)
    source_con = sqlite3.connect(SOURCE_DB, uri=True)
    rows = enrich_phase(load_base_rows(feature_con), source_con)
    predicates = build_predicates(rows)
    candidates = add_real_fill(evaluate_predicates(rows, predicates), rows, source_con)
    phase_distribution = {label: summarize([r for r in rows if r["phase_label"] == label]) for label in sorted({r["phase_label"] for r in rows})}
    payload = {
        "generated_at": iso_now(),
        "scope": {"symbol": SYMBOL, "liq_side": LIQ_SIDE, "route_id": ROUTE_ID, "rows": len(rows)},
        "predicate_count": len(predicates) + len(list(itertools.combinations(predicates, 2))),
        "phase_distribution": phase_distribution,
        "candidates": candidates,
        "sample_rows": rows[:8],
    }
    write_report(payload)
    print(json.dumps({"rows": len(rows), "predicate_count": payload["predicate_count"], "out_md": str(OUT_MD)}, indent=2))
    feature_con.close()
    source_con.close()


if __name__ == "__main__":
    main()
