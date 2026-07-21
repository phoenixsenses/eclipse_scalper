"""Report calibration diagnostics for S34 prediction snapshots."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data" / "s34_intelligence.db"


def build_report(ledger_db: Path, model_name: str) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT p.prediction_json, o.trade_id, o.rule_name, o.exit_reason, o.net_bps
            FROM s34_predictions p
            JOIN s34_outcomes o ON o.signal_id=p.signal_id
            WHERE p.model_name=?
            ORDER BY o.exit_ts_ms ASC
            """,
            (model_name,),
        ).fetchall()
    finally:
        con.close()
    pairs: list[dict[str, Any]] = []
    knn_pairs: list[dict[str, Any]] = []
    for row in rows:
        try:
            prediction = json.loads(str(row["prediction_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        expected = prediction.get("expected_net_bps")
        actual = row["net_bps"]
        if expected is None or actual is None:
            continue
        knn_expected = prediction.get("knn_v0_expected_net_bps")
        pairs.append(
            {
                "trade_id": row["trade_id"],
                "rule_name": row["rule_name"],
                "exit_reason": row["exit_reason"],
                "expected_net_bps": float(expected),
                "actual_net_bps": float(actual),
                "error_bps": float(actual) - float(expected),
            }
        )
        if knn_expected is not None:
            knn_pairs.append(
                {
                    "trade_id": row["trade_id"],
                    "rule_name": row["rule_name"],
                    "exit_reason": row["exit_reason"],
                    "expected_net_bps": float(knn_expected),
                    "actual_net_bps": float(actual),
                    "error_bps": float(actual) - float(knn_expected),
                }
            )
    if not pairs:
        return {"model_name": model_name, "n": 0, "pairs": []}
    base_summary = _summary(pairs)
    knn_summary = _summary(knn_pairs)
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_rule.setdefault(str(row["rule_name"]), []).append(row)
    return {
        "model_name": model_name,
        "n": len(pairs),
        "bias_bps": base_summary["bias_bps"],
        "mae_bps": base_summary["mae_bps"],
        "hit_direction_rate": base_summary["hit_direction_rate"],
        "optimism_rate": base_summary["optimism_rate"],
        "base_rate": base_summary,
        "knn_v0": knn_summary,
        "by_rule": {
            rule: {
                "n": len(items),
                "bias_bps": sum(item["error_bps"] for item in items) / len(items),
                "mae_bps": sum(abs(item["error_bps"]) for item in items) / len(items),
            }
            for rule, items in sorted(by_rule.items())
        },
        "latest": pairs[-10:],
        "note": "Diagnostic only. Backfilled predictions are not a clean holdout.",
    }


def build_model_comparison(ledger_db: Path, models: list[str]) -> dict[str, Any]:
    reports = {model: build_report(ledger_db, model) for model in models}
    return {
        "models": {
            model: {
                "n": report.get("n", 0),
                "bias_bps": report.get("bias_bps"),
                "mae_bps": report.get("mae_bps"),
                "hit_direction_rate": report.get("hit_direction_rate"),
                "optimism_rate": report.get("optimism_rate"),
            }
            for model, report in reports.items()
        },
        "note": "Diagnostic only. Backfilled predictions are not a clean holdout.",
    }


def _summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "bias_bps": None, "mae_bps": None, "hit_direction_rate": None, "optimism_rate": None}
    errors = [row["error_bps"] for row in pairs]
    abs_errors = [abs(value) for value in errors]
    direction_hits = [
        row
        for row in pairs
        if (row["expected_net_bps"] >= 0 and row["actual_net_bps"] >= 0)
        or (row["expected_net_bps"] < 0 and row["actual_net_bps"] < 0)
    ]
    optimistic = [row for row in pairs if row["expected_net_bps"] > row["actual_net_bps"]]
    return {
        "n": len(pairs),
        "bias_bps": sum(errors) / len(errors),
        "mae_bps": sum(abs_errors) / len(abs_errors),
        "hit_direction_rate": len(direction_hits) / len(pairs),
        "optimism_rate": len(optimistic) / len(pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S34 prediction calibration diagnostics.")
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--model-name", default="base_rate_v1")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = (
        build_model_comparison(args.ledger_db, ["base_rate_v1", "knn_v0", "knn_v1", "knn_v2"])
        if args.compare
        else build_report(args.ledger_db, args.model_name)
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"model={report['model_name']} n={report['n']}")
        if report["n"]:
            print(
                f"bias={report['bias_bps']:.2f}bps mae={report['mae_bps']:.2f}bps "
                f"dir_hit={report['hit_direction_rate']:.1%} optimism={report['optimism_rate']:.1%}"
            )
        print(report.get("note", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
