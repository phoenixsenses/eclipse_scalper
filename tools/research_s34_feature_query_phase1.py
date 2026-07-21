import itertools
import json
import sqlite3
from pathlib import Path


DB = "data/s34_feature_factory.db"
OUT_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_QUERY_RESULTS.json")
OUT_MD = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_QUERY_RESULTS.md")

ROUTES = ["LONG_DELAY0_TP60", "LONG_DELAY60_TP120", "SHORT_DELAY0_TP40_CONTROL"]
MIN_N = 30
MIN_DAYS = 5


PREDICATES = [
    ("cluster_notional", ">=", 200_000),
    ("cluster_notional", ">=", 500_000),
    ("cluster_notional", ">=", 1_000_000),
    ("cluster_count", ">=", 2),
    ("cluster_count", ">=", 3),
    ("cluster_intensity_notional_per_sec", ">=", 5_000),
    ("cluster_intensity_notional_per_sec", ">=", 10_000),
    ("btc_pre_15m_bps", ">=", 0),
    ("btc_pre_15m_bps", ">=", 25),
    ("btc_pre_15m_bps", "<=", 0),
    ("symbol_pre_5m_bps", ">=", 0),
    ("symbol_pre_5m_bps", ">=", 25),
    ("symbol_pre_5m_bps", "<=", 0),
    ("symbol_pre_15m_bps", ">=", 0),
    ("symbol_pre_15m_bps", ">=", 50),
    ("day_trend_bps", ">=", 0),
    ("day_trend_bps", ">=", 100),
    ("day_range_bps", ">=", 250),
    ("day_range_bps", ">=", 500),
    ("day_buy_liq_notional", ">=", 5_000_000),
    ("day_buy_liq_notional", ">=", 20_000_000),
    ("day_agg_count", ">=", 250_000),
    ("day_agg_count", ">=", 750_000),
]


def predicate_sql(predicate):
    col, op, value = predicate
    return f"f.{col} {op} {float(value)}"


def summarize(con: sqlite3.Connection, route_id: str, where_sql: str, label: str):
    rows = con.execute(
        f"""
        select
          l.net_bps,
          l.exit_reason,
          date(f.event_ts_ms/1000, 'unixepoch') as day
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where l.route_id=? and ({where_sql})
        """,
        (route_id,),
    ).fetchall()
    if len(rows) < MIN_N:
        return None
    vals = [float(row[0]) for row in rows]
    days = sorted({row[2] for row in rows})
    if len(days) < MIN_DAYS:
        return None
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    median = vals_sorted[mid] if len(vals_sorted) % 2 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
    day_cums = {}
    for day in days:
        day_vals = [float(row[0]) for row in rows if row[2] == day]
        day_cums[day] = sum(day_vals)
    top3_removed = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0
    positive_days = sum(1 for value in day_cums.values() if value > 0)
    return {
        "route_id": route_id,
        "filter": label,
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals),
        "median": median,
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "tp": sum(row[1] == "TP" for row in rows),
        "be": sum(row[1] == "BE" for row in rows),
        "sl": sum(row[1] == "SL" for row in rows),
        "time": sum(row[1] == "TIME" for row in rows),
        "top3_removed_cum": top3_removed,
        "positive_days": positive_days,
        "positive_day_rate": positive_days / len(days),
        "worst_day_cum": min(day_cums.values()),
        "best_day_cum": max(day_cums.values()),
    }


def main():
    con = sqlite3.connect(DB)
    results = []
    for route_id in ROUTES:
        base = summarize(con, route_id, "1=1", "BASE")
        if base:
            results.append(base)

        for pred in PREDICATES:
            results.append(
                summarize(
                    con,
                    route_id,
                    predicate_sql(pred),
                    f"{pred[0]} {pred[1]} {pred[2]}",
                )
            )

        for left, right in itertools.combinations(PREDICATES, 2):
            # Keep pair search small and interpretable: avoid same-column contradictions.
            if left[0] == right[0]:
                continue
            where = f"{predicate_sql(left)} and {predicate_sql(right)}"
            label = f"{left[0]} {left[1]} {left[2]} AND {right[0]} {right[1]} {right[2]}"
            results.append(summarize(con, route_id, where, label))

    results = [row for row in results if row is not None]
    results = sorted(
        results,
        key=lambda row: (
            row["median"] > 0,
            row["top3_removed_cum"] > 0,
            row["positive_day_rate"],
            row["mean"],
            row["cum"],
        ),
        reverse=True,
    )

    OUT_JSON.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Phase 1 Query Results",
        "",
        "Scope: query layer over `data/s34_feature_factory.db`.",
        "",
        "Only `liq_event_features` no-lookahead columns are used as filters. Outcome columns are joined only after filtering to evaluate route labels.",
        "",
        f"Eligibility: N >= {MIN_N}, days >= {MIN_DAYS}.",
        "",
        "## Top Results",
        "",
        "| Rank | Route | Filter | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME | Top3 Removed Cum | Positive Days | Worst Day |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(results[:30], 1):
        lines.append(
            f"| {idx} | {row['route_id']} | {row['filter']} | {row['n']} | {row['days']} | "
            f"{row['mean']:+.2f} | {row['median']:+.2f} | {row['cum']:+.2f} | "
            f"{row['wr']*100:.1f}% | {row['tp']}/{row['be']}/{row['sl']}/{row['time']} | "
            f"{row['top3_removed_cum']:+.2f} | {row['positive_days']}/{row['days']} | {row['worst_day_cum']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "This is still research infrastructure. A filter is interesting only if it has positive median, survives top-3 removal, and is spread across days. It still needs live bid/ask forward validation before becoming a runner rule.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(results[:10], indent=2))
    con.close()


if __name__ == "__main__":
    main()
