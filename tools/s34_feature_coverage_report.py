from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_FEATURE_COVERAGE.md")
DEFAULT_JSON = Path("reports/research/s34/S34_FEATURE_COVERAGE.json")

FEATURES = [
    "day_trend_bps",
    "day_range_bps",
    "day_buy_liq_notional",
    "day_agg_trade_count",
    "cluster_duration_sec",
    "max_single_liq_share",
    "intensity_per_sec",
    "inter_cluster_gap_sec",
    "prev_liq_gap_sec",
    "btc_pre_return_bps",
]


def _load_rows(db_path: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT signal_id, signal_ts_ms, signal_ts_utc, rule_name, features_json
            FROM s34_signals
            ORDER BY signal_ts_ms ASC
            """
        ).fetchall()
    finally:
        con.close()
    out = []
    for row in rows:
        try:
            features = json.loads(str(row["features_json"] or "{}"))
        except json.JSONDecodeError:
            features = {}
        out.append(
            {
                "signal_id": row["signal_id"],
                "signal_ts_ms": int(row["signal_ts_ms"] or 0),
                "signal_ts_utc": row["signal_ts_utc"],
                "rule_name": row["rule_name"],
                "features": features if isinstance(features, dict) else {},
            }
        )
    return out


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    features = {}
    for key in FEATURES:
        count = sum(1 for row in rows if _has_value(row["features"].get(key)))
        features[key] = {
            "count": count,
            "coverage_pct": round(100.0 * count / n, 2) if n else 0.0,
        }
    return {"n": n, "features": features}


def _latest_examples(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for row in list(reversed(rows))[:limit]:
        f = row["features"]
        out.append(
            {
                "signal_ts_utc": row["signal_ts_utc"],
                "rule_name": row["rule_name"],
                "liq_total_notional": f.get("liq_total_notional"),
                "day_trend_bps": f.get("day_trend_bps"),
                "day_range_bps": f.get("day_range_bps"),
                "intensity_per_sec": f.get("intensity_per_sec"),
                "inter_cluster_gap_sec": f.get("inter_cluster_gap_sec"),
                "max_single_liq_share": f.get("max_single_liq_share"),
            }
        )
    return out


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_payload(db_path: Path) -> dict[str, Any]:
    rows = _load_rows(db_path)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    day_rows = [row for row in rows if row["signal_ts_ms"] >= now_ms - 24 * 3600 * 1000]
    latest_50 = rows[-50:]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_time": _coverage(rows),
        "last_24h": _coverage(day_rows),
        "latest_50": _coverage(latest_50),
        "latest_examples": _latest_examples(rows),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# S34 Feature Coverage",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "Coverage means the feature exists in `s34_signals.features_json` with a non-null value. Historical rows before feature-completeness work are expected to be sparse.",
        "",
        "## Coverage",
        "",
    ]
    rows = []
    for feature in FEATURES:
        rows.append(
            [
                feature,
                payload["all_time"]["features"][feature]["coverage_pct"],
                payload["last_24h"]["features"][feature]["coverage_pct"],
                payload["latest_50"]["features"][feature]["coverage_pct"],
            ]
        )
    lines.append(_table(["Feature", "All Time %", "Last 24h %", "Latest 50 %"], rows))
    lines.extend(["", "## Latest Signals", ""])
    example_rows = []
    for item in payload["latest_examples"]:
        example_rows.append(
            [
                item["signal_ts_utc"],
                item["rule_name"],
                _fmt(item["liq_total_notional"]),
                _fmt(item["day_trend_bps"]),
                _fmt(item["day_range_bps"]),
                _fmt(item["intensity_per_sec"]),
                _fmt(item["inter_cluster_gap_sec"]),
                _fmt(item["max_single_liq_share"]),
            ]
        )
    lines.append(
        _table(
            ["Signal UTC", "Rule", "Notional", "Day Trend", "Day Range", "Intensity/s", "Gap/s", "Max Share"],
            example_rows,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report S34 signal feature coverage in the intelligence ledger.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    payload = build_payload(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.out_md, payload)
    print(
        json.dumps(
            {
                "all_time_n": payload["all_time"]["n"],
                "last_24h_n": payload["last_24h"]["n"],
                "latest_50_day_trend_pct": payload["latest_50"]["features"]["day_trend_bps"]["coverage_pct"],
                "latest_50_intensity_pct": payload["latest_50"]["features"]["intensity_per_sec"]["coverage_pct"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
