from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_intelligence_ledger as ledger


DEFAULT_DB = Path("data/s34_intelligence.db")


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"intelligence db not found: {path}")
    con = ledger.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def populate_prediction_errors(db_path: Path = DEFAULT_DB) -> int:
    with _connect(db_path) as con:
        rows = con.execute(
            """
            SELECT
              o.trade_id,
              o.signal_id,
              o.rule_name,
              o.net_bps AS realized_net_bps,
              o.outcome_ts_utc,
              p.prediction_id,
              p.model_name,
              p.model_version,
              p.predicted_at_utc,
              p.prediction_json
            FROM s34_outcomes o
            JOIN s34_predictions p ON p.signal_id=o.signal_id
            WHERE o.net_bps IS NOT NULL
            """
        ).fetchall()
        written = 0
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            prediction = _json_loads(row["prediction_json"])
            predicted = _safe_float(prediction.get("expected_net_bps"))
            realized = _safe_float(row["realized_net_bps"])
            if predicted is None or realized is None:
                continue
            error = realized - predicted
            direction_correct = 1 if _sign(predicted) == _sign(realized) else 0
            error_id = f"{row['trade_id']}:{row['prediction_id']}"
            con.execute(
                """
                INSERT OR REPLACE INTO s34_prediction_errors (
                    error_id, trade_id, signal_id, prediction_id, model_id, model_version,
                    rule_id, predicted_net_bps, realized_net_bps, forecast_error_bps,
                    abs_error_bps, direction_correct, predicted_at_utc, outcome_at_utc, scored_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    error_id,
                    row["trade_id"],
                    row["signal_id"],
                    row["prediction_id"],
                    row["model_name"],
                    row["model_version"],
                    row["rule_name"],
                    predicted,
                    realized,
                    error,
                    abs(error),
                    direction_correct,
                    row["predicted_at_utc"],
                    row["outcome_ts_utc"],
                    now,
                ),
            )
            written += 1
        con.commit()
    return written


def model_error_summary(db_path: Path = DEFAULT_DB, min_n: int = 0) -> list[dict[str, Any]]:
    with _connect(db_path) as con:
        rows = con.execute("SELECT * FROM s34_prediction_errors").fetchall()
    by_model: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_id"]), []).append(row)
    summary = []
    for model_id, items in sorted(by_model.items()):
        if len(items) < min_n:
            continue
        abs_errors = [float(row["abs_error_bps"]) for row in items if row["abs_error_bps"] is not None]
        errors = [float(row["forecast_error_bps"]) for row in items if row["forecast_error_bps"] is not None]
        dirs = [int(row["direction_correct"]) for row in items if row["direction_correct"] is not None]
        summary.append(
            {
                "model_id": model_id,
                "n": len(items),
                "mae_bps": mean(abs_errors) if abs_errors else None,
                "direction_accuracy": mean(dirs) if dirs else None,
                "mean_error_bps": mean(errors) if errors else None,
            }
        )
    summary.sort(key=lambda row: (row["mae_bps"] is None, row["mae_bps"] or 999999))
    return summary


def format_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "=== MODEL PREDICTION ERROR REPORT ===",
        f"{'Model':<14} {'N':>5} {'MAE(bps)':>10} {'Dir.Acc':>9} {'Mean Err':>10}",
    ]
    for row in rows:
        dir_acc = "-" if row["direction_accuracy"] is None else f"{row['direction_accuracy'] * 100:.1f}%"
        mae = "-" if row["mae_bps"] is None else f"{row['mae_bps']:.2f}"
        mean_err = "-" if row["mean_error_bps"] is None else f"{row['mean_error_bps']:+.2f}"
        lines.append(f"{row['model_id']:<14} {row['n']:>5} {mae:>10} {dir_acc:>9} {mean_err:>10}")
    lines.extend(
        [
            "=====================================",
            "Note: N < 30 models should be treated as preliminary.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate and summarize S34 prediction error rows.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min_n", "--min-n", dest="min_n", type=int, default=0)
    args = parser.parse_args()
    try:
        written = populate_prediction_errors(args.db)
        rows = model_error_summary(args.db, args.min_n)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    print(format_summary(rows))
    print(f"Rows upserted: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
