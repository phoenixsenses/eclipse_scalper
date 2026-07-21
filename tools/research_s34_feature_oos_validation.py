import ast
import itertools
import json
import sqlite3
from pathlib import Path


DB = "data/s34_feature_factory.db"
QUERY_SCRIPT = Path("tools/research_s34_feature_query_phase1.py")
OUT_JSON = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_OOS_VALIDATION.json")
OUT_MD = Path("reports/research/s34/S34_FEATURE_FACTORY_PHASE1_OOS_VALIDATION.md")

MIN_TRAIN_N = 20
MIN_TEST_N = 10
MIN_TRAIN_DAYS = 4
MIN_TEST_DAYS = 4


def load_constants():
    tree = ast.parse(QUERY_SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PREDICATES", "ROUTES"}:
                    out[target.id] = ast.literal_eval(node.value)
    return out["ROUTES"], out["PREDICATES"]


def pred_sql(pred):
    col, op, val = pred
    return f"f.{col} {op} {float(val)}"


def median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def summarize(con, route_id, where_sql, period_sql, label):
    rows = con.execute(
        f"""
        select l.net_bps, l.exit_reason, date(f.event_ts_ms/1000, 'unixepoch') as day
        from liq_event_features f
        join liq_event_outcome_labels l on l.event_id=f.event_id
        where l.route_id=? and ({where_sql}) and ({period_sql})
        """,
        (route_id,),
    ).fetchall()
    if not rows:
        return {"label": label, "n": 0, "days": 0}
    vals = [float(r[0]) for r in rows]
    days = sorted({r[2] for r in rows})
    top3_removed = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0
    day_cums = {day: sum(float(r[0]) for r in rows if r[2] == day) for day in days}
    return {
        "label": label,
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": top3_removed,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "worst_day_cum": min(day_cums.values()),
        "tp": sum(r[1] == "TP" for r in rows),
        "be": sum(r[1] == "BE" for r in rows),
        "sl": sum(r[1] == "SL" for r in rows),
        "time": sum(r[1] == "TIME" for r in rows),
    }


def main():
    routes, predicates = load_constants()
    con = sqlite3.connect(DB)
    min_ts, max_ts = con.execute("select min(event_ts_ms), max(event_ts_ms) from liq_event_features").fetchone()
    split_ts = int((min_ts + max_ts) / 2)

    filters = [("BASE", "1=1")]
    for pred in predicates:
        filters.append((f"{pred[0]} {pred[1]} {pred[2]}", pred_sql(pred)))
    for left, right in itertools.combinations(predicates, 2):
        if left[0] == right[0]:
            continue
        filters.append(
            (
                f"{left[0]} {left[1]} {left[2]} AND {right[0]} {right[1]} {right[2]}",
                f"{pred_sql(left)} and {pred_sql(right)}",
            )
        )

    candidates = []
    evaluated = 0
    for route in routes:
        for label, sql in filters:
            evaluated += 1
            train = summarize(con, route, sql, f"f.event_ts_ms <= {split_ts}", label)
            if train["n"] < MIN_TRAIN_N or train["days"] < MIN_TRAIN_DAYS:
                continue
            if train["median"] is None or train["median"] <= 0 or train["top3_removed_cum"] <= 0:
                continue
            test = summarize(con, route, sql, f"f.event_ts_ms > {split_ts}", label)
            if test["n"] < MIN_TEST_N or test["days"] < MIN_TEST_DAYS:
                continue
            candidates.append({"route_id": route, "filter": label, "train": train, "test": test})

    candidates = sorted(
        candidates,
        key=lambda row: (
            row["test"]["median"] > 0,
            row["test"]["top3_removed_cum"] > 0,
            row["test"]["positive_days"] / row["test"]["days"] if row["test"]["days"] else 0,
            row["test"]["mean"],
        ),
        reverse=True,
    )

    payload = {
        "evaluated_candidates": evaluated,
        "split_ts_ms": split_ts,
        "train_period": {"min_ts_ms": min_ts, "max_ts_ms": split_ts},
        "test_period": {"min_ts_ms": split_ts + 1, "max_ts_ms": max_ts},
        "surviving_candidates": candidates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Feature Factory Phase 1 OOS Validation",
        "",
        "Purpose: control the Phase 1 query-layer multiple-testing risk by selecting filters on the first half of the event timeline and validating them on the second half.",
        "",
        f"Evaluated candidates: `{evaluated}`",
        f"Surviving train-selected candidates with test support: `{len(candidates)}`",
        f"Split timestamp ms: `{split_ts}`",
        "",
        "Selection rule on train: N >= 20, days >= 4, median > 0, top3-removed cum > 0.",
        "Validation reporting on test: N >= 10, days >= 4.",
        "",
        "## Top OOS Candidates",
        "",
        "| Rank | Route | Filter | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Positive Days |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(candidates[:30], 1):
        tr = row["train"]
        te = row["test"]
        lines.append(
            f"| {idx} | {row['route_id']} | {row['filter']} | {tr['n']} | {tr['median']:+.2f} | "
            f"{tr['cum']:+.2f} | {te['n']} | {te['median']:+.2f} | {te['mean']:+.2f} | "
            f"{te['cum']:+.2f} | {te['wr']*100:.1f}% | {te['top3_removed_cum']:+.2f} | "
            f"{te['positive_days']}/{te['days']} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "A candidate is not accepted just because it appears in this table. It still needs real bid/ask fill parity and forward paper validation. This table only checks whether a train-selected no-lookahead filter survives a simple temporal holdout.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({"evaluated": evaluated, "survivors": len(candidates), "top": candidates[:5]}, indent=2))
    con.close()


if __name__ == "__main__":
    main()
