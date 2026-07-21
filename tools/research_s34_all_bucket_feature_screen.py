"""All-bucket S34 feature-factory screen.

Research only. Reads data/s34_feature_factory.db and evaluates every active
non-deprecated S34 bucket against the route labels already present in the
feature factory. No runner, config, journal, or microstructure DB writes.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import DEFAULT_RULES, S34Rule, _deprecated_paper_rule_reason

FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_ALL_BUCKET_FEATURE_SCREEN.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_ALL_BUCKET_FEATURE_SCREEN.md"


def _active_rules() -> list[S34Rule]:
    return [rule for rule in DEFAULT_RULES if not _deprecated_paper_rule_reason(rule)]


def _route_candidates(symbol: str, liq_side: str, direction: str) -> list[str]:
    if direction == "LONG":
        routes = ["LONG_DELAY0_TP60"]
        if symbol == "ETHUSDT" and liq_side == "BUY":
            routes.append("LONG_DELAY60_TP120")
        return routes
    if symbol == "ETHUSDT" and liq_side == "SELL":
        return ["SHORT_DELAY0_TP60", "SHORT_DELAY0_TP80"]
    if symbol == "BTCUSDT" and liq_side == "SELL":
        return ["SHORT_DELAY0_TP40", "SHORT_DELAY0_TP60"]
    if symbol == "SOLUSDT" and liq_side == "SELL":
        return ["SHORT_DELAY0_TP40", "SHORT_DELAY0_TP60"]
    return ["SHORT_DELAY0_TP40_CONTROL"]


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def _mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    return f"{float(v):+.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.0f}%"


def _rule_where(rule: S34Rule) -> tuple[str, list[Any], list[str]]:
    clauses = ["f.symbol=?", "f.liq_side=?", "f.cluster_notional>=?"]
    params: list[Any] = [rule.symbol, rule.liq_side, float(rule.threshold_usd)]
    notes: list[str] = []

    if rule.min_day_trend_bps is not None:
        clauses.append("f.day_trend_bps>=?")
        params.append(float(rule.min_day_trend_bps))
        notes.append(f"day_trend>={rule.min_day_trend_bps:g}")
    if rule.max_day_trend_bps is not None:
        clauses.append("f.day_trend_bps<=?")
        params.append(float(rule.max_day_trend_bps))
        notes.append(f"day_trend<={rule.max_day_trend_bps:g}")
    if rule.min_cluster_liq_count is not None:
        clauses.append("COALESCE(f.cluster_liq_count, f.cluster_count)>=?")
        params.append(int(rule.min_cluster_liq_count))
        notes.append(f"liq_count>={rule.min_cluster_liq_count}")
    if rule.required_shape_label is not None:
        clauses.append("f.shape_label=?")
        params.append(str(rule.required_shape_label))
        notes.append(f"shape={rule.required_shape_label}")
    if rule.max_single_liq_share_pct is not None:
        clauses.append("f.max_single_liq_share<=?")
        params.append(float(rule.max_single_liq_share_pct))
        notes.append(f"max_share<={rule.max_single_liq_share_pct:g}")
    if rule.btc_pre_min_return_bps is not None:
        clauses.append("f.btc_pre_15m_bps>=?")
        params.append(float(rule.btc_pre_min_return_bps))
        notes.append(f"btc_pre15>={rule.btc_pre_min_return_bps:g}")

    return " AND ".join(clauses), params, notes


def _summarize(rows: list[sqlite3.Row]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows]
    if not vals:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "cum": 0.0,
            "wr": None,
            "top3_removed_cum": 0.0,
            "positive_days": 0,
            "total_days": 0,
            "avg_hold_sec": None,
            "exit_counts": {},
            "giveback_n": 0,
            "giveback_pct": None,
        }

    day_cums: dict[str, float] = defaultdict(float)
    exits: dict[str, int] = defaultdict(int)
    holds: list[float] = []
    giveback = 0
    for row in rows:
        day = datetime.fromtimestamp(int(row["event_ts_ms"]) / 1000, tz=timezone.utc).date().isoformat()
        day_cums[day] += float(row["net_bps"])
        exits[str(row["exit_reason"])] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        # "Gave it back": reached at least half TP but still closed negative.
        if float(row["mfe_bps"]) >= float(row["tp_bps"]) * 0.5 and float(row["net_bps"]) < 0:
            giveback += 1

    sorted_vals = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "median": _median(vals),
        "mean": _mean(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted_vals[3:]) if len(sorted_vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "total_days": len(day_cums),
        "avg_hold_sec": _mean(holds),
        "exit_counts": dict(exits),
        "giveback_n": giveback,
        "giveback_pct": giveback / len(vals),
    }


def _verdict(summary: dict[str, Any]) -> str:
    n = int(summary.get("n") or 0)
    if n < 30:
        return "thin"
    if (summary.get("median") or 0.0) <= 0:
        return "negative_median"
    if (summary.get("top3_removed_cum") or 0.0) <= 0:
        return "outlier_dependent"
    total_days = int(summary.get("total_days") or 0)
    if total_days and int(summary.get("positive_days") or 0) / total_days < 0.55:
        return "weak_day_consistency"
    return "candidate"


def _screen_rule(con: sqlite3.Connection, rule: S34Rule) -> dict[str, Any]:
    where, params, notes = _rule_where(rule)
    event_ids = [
        row["event_id"]
        for row in con.execute(f"SELECT f.event_id FROM liq_event_features f WHERE {where}", params).fetchall()
    ]
    route_rows: list[dict[str, Any]] = []
    for route_id in _route_candidates(rule.symbol, rule.liq_side, rule.direction):
        if not event_ids:
            rows: list[sqlite3.Row] = []
        else:
            ph = ",".join("?" for _ in event_ids)
            rows = con.execute(
                f"""
                SELECT f.event_ts_ms, l.*
                FROM liq_event_outcome_labels l
                JOIN liq_event_features f ON f.event_id=l.event_id
                WHERE l.event_id IN ({ph}) AND l.route_id=?
                ORDER BY f.event_ts_ms ASC
                """,
                [*event_ids, route_id],
            ).fetchall()
        summary = _summarize(rows)
        route_rows.append(
            {
                "route_id": route_id,
                "summary": summary,
                "verdict": _verdict(summary),
                "label_matches_runner_exactly": _label_matches_rule(rule, route_id),
            }
        )

    best = max(
        route_rows,
        key=lambda r: (
            r["verdict"] == "candidate",
            r["summary"]["median"] if r["summary"]["median"] is not None else -9999.0,
            r["summary"]["top3_removed_cum"],
        ),
    )
    return {
        "rule_id": rule.name,
        "symbol": rule.symbol,
        "liq_side": rule.liq_side,
        "direction": rule.direction,
        "threshold": rule.threshold_usd,
        "filters": notes,
        "event_n": len(event_ids),
        "routes": route_rows,
        "best": best,
    }


def _label_matches_rule(rule: S34Rule, route_id: str) -> bool:
    if rule.direction == "LONG" and route_id == "LONG_DELAY0_TP60":
        return rule.entry_delay_sec == 0 and rule.tp_bps == 60 and rule.sl_bps == 40 and rule.be_trigger_bps == 30
    if rule.direction == "LONG" and route_id == "LONG_DELAY60_TP120":
        return rule.entry_delay_sec == 60 and rule.tp_bps == 120 and rule.sl_bps == 40 and rule.be_trigger_bps == 30
    if rule.direction == "SHORT" and route_id == "SHORT_DELAY0_TP60":
        return rule.entry_delay_sec == 0 and rule.tp_bps == 60 and rule.sl_bps == 40 and rule.be_trigger_bps == 40
    if rule.direction == "SHORT" and route_id == "SHORT_DELAY0_TP80":
        return rule.entry_delay_sec == 0 and rule.tp_bps == 80 and rule.sl_bps == 40 and rule.be_trigger_bps == 40
    if rule.direction == "SHORT" and route_id == "SHORT_DELAY0_TP40":
        return rule.entry_delay_sec == 0 and rule.tp_bps == 40 and rule.sl_bps == 40 and rule.be_trigger_bps == 40
    return False


def main() -> None:
    if not FEATURE_DB.exists():
        raise SystemExit(f"missing feature DB: {FEATURE_DB}")
    con = sqlite3.connect(FEATURE_DB)
    con.row_factory = sqlite3.Row

    screens = [_screen_rule(con, rule) for rule in _active_rules()]
    coverage = [
        dict(row)
        for row in con.execute(
            """
            SELECT symbol, liq_side, COUNT(*) AS events,
                   MIN(event_utc) AS first_utc, MAX(event_utc) AS last_utc
            FROM liq_event_features
            GROUP BY symbol, liq_side
            ORDER BY symbol, liq_side
            """
        )
    ]
    con.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(FEATURE_DB),
        "scope": "feature-factory all active non-deprecated S34 buckets",
        "coverage": coverage,
        "screens": screens,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 All-Bucket Feature Screen",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research-only screen over `data/s34_feature_factory.db`. No runner/config changes.",
        "",
        "Important limitation: this uses the route labels already present in the feature factory. "
        "Some live runner rules have exact label parity; others are screened with the nearest available route label.",
        "",
        "## Feature DB Coverage",
        "",
        "| Symbol | Side | Events | First | Last |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in coverage:
        lines.append(f"| {row['symbol']} | {row['liq_side']} | {row['events']} | {row['first_utc']} | {row['last_utc']} |")

    lines += [
        "",
        "## Best Available Route Per Active Bucket",
        "",
        "| Rule | Events | Filters | Best route | Exact? | N | Median | Mean | WR | Cum | Top3 removed | Pos days | Avg hold | Giveback | Verdict |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in screens:
        best = item["best"]
        s = best["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["rule_id"],
                    str(item["event_n"]),
                    ", ".join(item["filters"]) if item["filters"] else "-",
                    best["route_id"],
                    "yes" if best["label_matches_runner_exactly"] else "nearest",
                    str(s["n"]),
                    _fmt(s["median"]),
                    _fmt(s["mean"]),
                    _pct(s["wr"]),
                    _fmt(s["cum"]),
                    _fmt(s["top3_removed_cum"]),
                    f"{s['positive_days']}/{s['total_days']}",
                    _fmt(s["avg_hold_sec"], 0),
                    _pct(s["giveback_pct"]),
                    best["verdict"],
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## All Route Labels By Bucket",
        "",
    ]
    for item in screens:
        lines += [
            f"### {item['rule_id']}",
            "",
            "| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        for route in item["routes"]:
            s = route["summary"]
            exits = ", ".join(f"{k}={v}" for k, v in sorted(s["exit_counts"].items())) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        route["route_id"],
                        "yes" if route["label_matches_runner_exactly"] else "nearest",
                        str(s["n"]),
                        _fmt(s["median"]),
                        _pct(s["wr"]),
                        exits,
                        _pct(s["giveback_pct"]),
                        route["verdict"],
                    ]
                )
                + " |"
            )
        lines.append("")

    lines += [
        "## Next-Step Interpretation Rules",
        "",
        "- `candidate` means the bucket deserves forward collection or a deeper exact-route sweep.",
        "- `nearest` means do not promote directly; first generate exact labels or run a runner-helper parity check.",
        "- High `giveback` flags the fast-exit question: the route often sees MFE but closes negative.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
