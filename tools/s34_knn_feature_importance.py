"""Permutation/drop-one audit for S34 KNN features.

This is diagnostic research only. It does not change the runner or any live
rules. Each prediction uses only outcomes closed before the signal timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data" / "s34_intelligence.db"
DEFAULT_OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_KNN_FEATURE_IMPORTANCE.md"
DEFAULT_OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_KNN_FEATURE_IMPORTANCE.json"

FEATURES = [
    "log_cluster_notional",
    "cluster_liq_count_ratio",
    "shape_match",
    "cluster_duration_sec",
    "max_single_liq_share",
    "btc_pre_return_bps",
]


def _safe_json(text: str | None) -> dict[str, Any]:
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}


def _scaled_abs(left: Any, right: Any, scale: float) -> float:
    if left is None or right is None or left == "" or right == "":
        return 0.0
    try:
        return abs(float(left) - float(right)) / max(1.0, float(scale))
    except (TypeError, ValueError):
        return 0.0


def _distance(target: dict[str, Any], candidate: dict[str, Any], features: set[str]) -> float:
    distance = 0.0
    if "log_cluster_notional" in features:
        distance += abs(math.log1p(float(target.get("cluster_notional") or 0.0)) - math.log1p(float(candidate.get("cluster_notional") or 0.0)))
    if "cluster_liq_count_ratio" in features:
        left = int(target.get("cluster_liq_count") or 0)
        right = int(candidate.get("cluster_liq_count") or 0)
        distance += abs(left - right) / max(1.0, float(max(left, right, 1)))
    if "shape_match" in features:
        distance += 0.0 if str(target.get("cluster_shape_label") or "") == str(candidate.get("cluster_shape_label") or "") else 0.75
    if "cluster_duration_sec" in features:
        distance += _scaled_abs(target.get("cluster_duration_sec"), candidate.get("cluster_duration_sec"), 180.0)
    if "max_single_liq_share" in features:
        distance += _scaled_abs(target.get("cluster_max_single_liq_share"), candidate.get("cluster_max_single_liq_share"), 100.0)
    if "btc_pre_return_bps" in features:
        distance += _scaled_abs(target.get("btc_pre_return_bps"), candidate.get("btc_pre_return_bps"), 100.0)
    return distance


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "mae_bps": None, "bias_bps": None, "hit_direction_rate": None, "coverage": 0}
    errors = [float(row["actual_net_bps"]) - float(row["expected_net_bps"]) for row in pairs]
    dir_hits = [
        row
        for row in pairs
        if (float(row["expected_net_bps"]) >= 0 and float(row["actual_net_bps"]) >= 0)
        or (float(row["expected_net_bps"]) < 0 and float(row["actual_net_bps"]) < 0)
    ]
    return {
        "n": len(pairs),
        "mae_bps": sum(abs(err) for err in errors) / len(errors),
        "bias_bps": sum(errors) / len(errors),
        "hit_direction_rate": len(dir_hits) / len(pairs),
    }


def _prediction_pairs(events: list[dict[str, Any]], features: set[str], k: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i, event in enumerate(events):
        history = [
            row
            for row in events[:i]
            if row["rule_name"] == event["rule_name"] and row["exit_ts_ms"] < event["signal_ts_ms"]
        ]
        if not history:
            continue
        neighbors = sorted(history, key=lambda row: _distance(event, row, features))[:k]
        expected = _median([float(row["net_bps"]) for row in neighbors])
        if expected is None:
            continue
        pairs.append(
            {
                "trade_id": event["trade_id"],
                "rule_name": event["rule_name"],
                "expected_net_bps": expected,
                "actual_net_bps": float(event["net_bps"]),
                "neighbor_count": len(neighbors),
            }
        )
    return pairs


def load_events(ledger_db: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT o.trade_id, o.rule_name, o.exit_ts_ms, o.exit_reason, o.net_bps,
                   s.signal_ts_ms, s.cluster_notional, s.cluster_liq_count,
                   s.cluster_shape_label, s.features_json
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            WHERE o.net_bps IS NOT NULL
            ORDER BY s.signal_ts_ms ASC
            """
        ).fetchall()
    finally:
        con.close()
    events: list[dict[str, Any]] = []
    for row in rows:
        features = _safe_json(row["features_json"])
        events.append(
            {
                "trade_id": row["trade_id"],
                "rule_name": row["rule_name"],
                "exit_ts_ms": int(row["exit_ts_ms"] or 0),
                "exit_reason": row["exit_reason"],
                "net_bps": float(row["net_bps"]),
                "signal_ts_ms": int(row["signal_ts_ms"] or 0),
                "cluster_notional": float(row["cluster_notional"] or 0.0),
                "cluster_liq_count": int(row["cluster_liq_count"] or 0),
                "cluster_shape_label": row["cluster_shape_label"] or "",
                "cluster_duration_sec": features.get("cluster_duration_sec"),
                "cluster_max_single_liq_share": features.get("cluster_max_single_liq_share"),
                "btc_pre_return_bps": features.get("btc_pre_return_bps"),
            }
        )
    return events


def build_report(ledger_db: Path, k: int) -> dict[str, Any]:
    events = load_events(ledger_db)
    full_features = set(FEATURES)
    baseline_pairs = _prediction_pairs(events, full_features, k)
    baseline = _summary(baseline_pairs)
    drops = []
    for feature in FEATURES:
        reduced = set(FEATURES)
        reduced.remove(feature)
        pairs = _prediction_pairs(events, reduced, k)
        summary = _summary(pairs)
        drops.append(
            {
                "removed_feature": feature,
                **summary,
                "mae_delta_vs_full": None
                if baseline.get("mae_bps") is None or summary.get("mae_bps") is None
                else float(summary["mae_bps"]) - float(baseline["mae_bps"]),
                "direction_delta_vs_full": None
                if baseline.get("hit_direction_rate") is None or summary.get("hit_direction_rate") is None
                else float(summary["hit_direction_rate"]) - float(baseline["hit_direction_rate"]),
            }
        )
    return {
        "ledger_db": str(ledger_db),
        "k": k,
        "features": FEATURES,
        "events": len(events),
        "baseline": baseline,
        "drop_one": drops,
        "note": "Temporal-safe diagnostic: each event only sees same-rule outcomes closed before its signal timestamp.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S34 KNN Feature Importance",
        "",
        f"- Events: {report['events']}",
        f"- K: {report['k']}",
        f"- Note: {report['note']}",
        "",
        "## Baseline",
        "",
        "| N | MAE bps | Bias bps | Direction hit |",
        "|---:|---:|---:|---:|",
    ]
    base = report["baseline"]
    lines.append(
        f"| {base.get('n', 0)} | {_fmt(base.get('mae_bps'))} | {_fmt(base.get('bias_bps'))} | {_pct(base.get('hit_direction_rate'))} |"
    )
    lines.extend(
        [
            "",
            "## Drop-One",
            "",
            "| Removed feature | N | MAE bps | MAE delta | Direction hit | Direction delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["drop_one"]:
        lines.append(
            f"| {row['removed_feature']} | {row.get('n', 0)} | {_fmt(row.get('mae_bps'))} | {_fmt(row.get('mae_delta_vs_full'))} | {_pct(row.get('hit_direction_rate'))} | {_pct(row.get('direction_delta_vs_full'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal-safe KNN feature importance audit.")
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()
    report = build_report(args.ledger_db, args.k)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json), "baseline": report["baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
