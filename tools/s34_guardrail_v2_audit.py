from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_GUARDRAIL_V2_AUDIT.md")
DEFAULT_JSON = Path("reports/research/s34/S34_GUARDRAIL_V2_AUDIT.json")


@dataclass
class Row:
    trade_id: str
    signal_id: str
    rule_name: str
    signal_ts_utc: str
    exit_reason: str
    net_bps: float
    gross_bps: float | None
    entry_adverse_bps: float | None
    exit_adverse_bps: float | None
    cluster_notional: float | None
    cluster_liq_count: int | None
    cluster_shape_label: str | None
    features: dict[str, Any]
    guardrail_level: str
    guardrail_headline: str
    guardrail: dict[str, Any]
    expected_by_model: dict[str, float]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _expected_from_prediction(payload: dict[str, Any]) -> float | None:
    for key in ("expected_net_bps", "knn_v0_expected_net_bps", "knn_v1_expected_net_bps", "knn_v2_expected_net_bps"):
        value = _float_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _load_rows(db_path: Path) -> list[Row]:
    with _connect(db_path) as conn:
        raw = conn.execute(
            """
            SELECT
              o.trade_id, o.signal_id, o.rule_name, o.exit_reason, o.net_bps,
              o.gross_bps, o.entry_adverse_bps, o.exit_adverse_bps,
              s.signal_ts_utc, s.cluster_notional, s.cluster_liq_count,
              s.cluster_shape_label, s.features_json,
              g.level AS guardrail_level, g.headline AS guardrail_headline,
              g.guardrail_json
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            LEFT JOIN s34_model_guardrails g ON g.signal_id=o.signal_id
            ORDER BY o.exit_ts_ms ASC, o.trade_id ASC
            """
        ).fetchall()
        pred_rows = conn.execute(
            """
            SELECT signal_id, model_name, prediction_json
            FROM s34_predictions
            ORDER BY predicted_at_utc ASC
            """
        ).fetchall()

    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    for pred in pred_rows:
        payload = _json_loads(pred["prediction_json"])
        expected = _expected_from_prediction(payload)
        if expected is not None:
            predictions[str(pred["signal_id"])][str(pred["model_name"])] = expected

    rows: list[Row] = []
    for item in raw:
        features = _json_loads(item["features_json"])
        guardrail = _json_loads(item["guardrail_json"])
        rows.append(
            Row(
                trade_id=str(item["trade_id"]),
                signal_id=str(item["signal_id"]),
                rule_name=str(item["rule_name"]),
                signal_ts_utc=str(item["signal_ts_utc"]),
                exit_reason=str(item["exit_reason"]),
                net_bps=float(item["net_bps"] or 0.0),
                gross_bps=_float_or_none(item["gross_bps"]),
                entry_adverse_bps=_float_or_none(item["entry_adverse_bps"]),
                exit_adverse_bps=_float_or_none(item["exit_adverse_bps"]),
                cluster_notional=_float_or_none(item["cluster_notional"]),
                cluster_liq_count=int(item["cluster_liq_count"]) if item["cluster_liq_count"] is not None else None,
                cluster_shape_label=str(item["cluster_shape_label"]) if item["cluster_shape_label"] else None,
                features=features,
                guardrail_level=str(item["guardrail_level"] or "missing"),
                guardrail_headline=str(item["guardrail_headline"] or "missing"),
                guardrail=guardrail,
                expected_by_model=dict(predictions.get(str(item["signal_id"]), {})),
            )
        )
    return rows


def _metrics(rows: Iterable[Row]) -> dict[str, Any]:
    items = list(rows)
    nets = [r.net_bps for r in items]
    wins = [x for x in nets if x > 0]
    return {
        "n": len(items),
        "cum_net_bps": round(sum(nets), 2) if nets else 0.0,
        "mean_net_bps": round(mean(nets), 2) if nets else 0.0,
        "median_net_bps": round(median(nets), 2) if nets else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / len(items), 2) if items else 0.0,
    }


def _prediction_features(row: Row) -> dict[str, Any]:
    vals = list(row.expected_by_model.values())
    neg = [v for v in vals if v < 0]
    pos = [v for v in vals if v > 0]
    strong_neg = [v for v in vals if v <= -30.0]
    if vals:
        dispersion = max(vals) - min(vals)
        avg_expected = mean(vals)
        min_expected = min(vals)
        max_expected = max(vals)
    else:
        dispersion = None
        avg_expected = None
        min_expected = None
        max_expected = None
    return {
        "model_count": len(vals),
        "negative_model_count": len(neg),
        "positive_model_count": len(pos),
        "strong_negative_count": len(strong_neg),
        "avg_expected_bps": avg_expected,
        "min_expected_bps": min_expected,
        "max_expected_bps": max_expected,
        "prediction_dispersion_bps": dispersion,
    }


def _bin_cluster_notional(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 100_000:
        return "<100K"
    if value < 200_000:
        return "100K-200K"
    if value < 500_000:
        return "200K-500K"
    if value < 1_000_000:
        return "500K-1M"
    return ">=1M"


def _bin_liq_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return ">10"


def _bin_expected(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= -50:
        return "<=-50"
    if value <= -30:
        return "-50..-30"
    if value < 0:
        return "-30..0"
    if value < 30:
        return "0..30"
    return ">=30"


def _group(rows: list[Row], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    return {
        key: _metrics(value)
        for key, value in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _warning_split(rows: list[Row]) -> dict[str, Any]:
    warnings = [r for r in rows if r.guardrail_level == "warning"]
    winners = [r for r in warnings if r.net_bps > 0]
    losers = [r for r in warnings if r.net_bps <= 0]

    def summarize(items: list[Row]) -> dict[str, Any]:
        base = _metrics(items)
        enriched = []
        for row in items:
            pf = _prediction_features(row)
            enriched.append(
                {
                    "trade_id": row.trade_id,
                    "rule_name": row.rule_name,
                    "exit_reason": row.exit_reason,
                    "net_bps": round(row.net_bps, 2),
                    "cluster_notional": row.cluster_notional,
                    "cluster_liq_count": row.cluster_liq_count,
                    "shape": row.cluster_shape_label,
                    **pf,
                }
            )
        if items:
            prediction_rows = [_prediction_features(r) for r in items]
            for key in (
                "negative_model_count",
                "strong_negative_count",
                "avg_expected_bps",
                "min_expected_bps",
                "prediction_dispersion_bps",
            ):
                vals = [p[key] for p in prediction_rows if p[key] is not None]
                base[f"median_{key}"] = round(median(vals), 2) if vals else None
        base["examples"] = enriched
        return base

    return {
        "warning_all": summarize(warnings),
        "warning_winners": summarize(winners),
        "warning_losers": summarize(losers),
        "by_rule": _group(warnings, lambda r: r.rule_name),
        "by_cluster_notional": _group(warnings, lambda r: _bin_cluster_notional(r.cluster_notional)),
        "by_liq_count": _group(warnings, lambda r: _bin_liq_count(r.cluster_liq_count)),
        "by_min_expected": _group(warnings, lambda r: _bin_expected(_prediction_features(r)["min_expected_bps"])),
        "by_avg_expected": _group(warnings, lambda r: _bin_expected(_prediction_features(r)["avg_expected_bps"])),
    }


def _candidate_v2(rows: list[Row]) -> dict[str, Any]:
    def hard_block(row: Row) -> bool:
        # Research label only. This is the narrowest split found in the current
        # warning bucket: 100K-200K warning clusters are 7/7 losers in-sample.
        return (
            row.guardrail_level == "warning"
            and row.cluster_notional is not None
            and 100_000 <= row.cluster_notional < 200_000
        )

    hard = [r for r in rows if hard_block(r)]
    kept = [r for r in rows if not hard_block(r)]
    baseline = _metrics(rows)
    kept_metrics = _metrics(kept)
    hard_metrics = _metrics(hard)
    return {
        "definition": "warning AND 100K <= cluster_notional < 200K",
        "baseline": baseline,
        "hard_block_candidate": hard_metrics,
        "kept_after_candidate": {
            **kept_metrics,
            "delta_cum_vs_baseline_bps": round(kept_metrics["cum_net_bps"] - baseline["cum_net_bps"], 2),
            "blocked_n": len(hard),
        },
        "blocked_examples": [
            {
                "trade_id": r.trade_id,
                "rule_name": r.rule_name,
                "exit_reason": r.exit_reason,
                "net_bps": round(r.net_bps, 2),
                "cluster_notional": r.cluster_notional,
                **_prediction_features(r),
            }
            for r in sorted(hard, key=lambda x: x.net_bps)
        ],
    }


def _feature_inventory(rows: list[Row]) -> dict[str, Any]:
    keys: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.features.items():
            if value is not None:
                keys[key] += 1
    missing_desired = [
        "day_trend_bps",
        "day_range_bps",
        "btc_pre_15m_bps",
        "cluster_duration_sec",
        "max_single_liq_share",
        "intensity_per_sec",
        "inter_cluster_gap_sec",
    ]
    aliases = {
        "btc_pre_15m_bps": "btc_pre_return_bps",
    }
    return {
        "available_feature_counts": dict(sorted(keys.items())),
        "desired_but_not_in_signal_feature_json": [
            key for key in missing_desired if keys.get(key, 0) == 0 and keys.get(aliases.get(key, ""), 0) == 0
        ],
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _metrics_rows(grouped: dict[str, dict[str, Any]]) -> list[list[Any]]:
    return [
        [key, m["n"], m["cum_net_bps"], m["mean_net_bps"], m["median_net_bps"], m["win_rate_pct"]]
        for key, m in grouped.items()
    ]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    split = payload["warning_split"]
    candidate = payload["candidate_v2"]
    inv = payload["feature_inventory"]
    rows = [
        ["warning_all", split["warning_all"]["n"], split["warning_all"]["cum_net_bps"], split["warning_all"]["median_net_bps"], split["warning_all"]["win_rate_pct"], split["warning_all"].get("median_min_expected_bps"), split["warning_all"].get("median_prediction_dispersion_bps")],
        ["warning_winners", split["warning_winners"]["n"], split["warning_winners"]["cum_net_bps"], split["warning_winners"]["median_net_bps"], split["warning_winners"]["win_rate_pct"], split["warning_winners"].get("median_min_expected_bps"), split["warning_winners"].get("median_prediction_dispersion_bps")],
        ["warning_losers", split["warning_losers"]["n"], split["warning_losers"]["cum_net_bps"], split["warning_losers"]["median_net_bps"], split["warning_losers"]["win_rate_pct"], split["warning_losers"].get("median_min_expected_bps"), split["warning_losers"].get("median_prediction_dispersion_bps")],
    ]
    blocked_rows = [
        [x["trade_id"], x["rule_name"], x["exit_reason"], x["net_bps"], round(float(x["cluster_notional"] or 0.0), 2), x["strong_negative_count"], x["min_expected_bps"]]
        for x in candidate["blocked_examples"][:20]
    ]
    lines = [
        "# S34 Guardrail V2 Audit",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "Scope: closed ledger trades only. This is research. No runner/config/live rule changed.",
        "",
        "## Feature Inventory",
        "",
        f"Closed rows: `{payload['closed_n']}`",
        "",
        "Available signal feature counts:",
        "",
        _table(["Feature", "Rows"], [[k, v] for k, v in inv["available_feature_counts"].items()]),
        "",
        "Desired fields missing from `features_json`: " + ", ".join(inv["desired_but_not_in_signal_feature_json"]),
        "",
        "## Warning Winner vs Warning Loser",
        "",
        _table(["Bucket", "N", "Cum", "Median", "WR %", "Median Min Exp", "Median Dispersion"], rows),
        "",
        "### Warning By Rule",
        "",
        _table(["Rule", "N", "Cum", "Mean", "Median", "WR %"], _metrics_rows(split["by_rule"])),
        "",
        "### Warning By Cluster Notional",
        "",
        _table(["Cluster", "N", "Cum", "Mean", "Median", "WR %"], _metrics_rows(split["by_cluster_notional"])),
        "",
        "### Warning By Min Expected",
        "",
        _table(["Min Expected", "N", "Cum", "Mean", "Median", "WR %"], _metrics_rows(split["by_min_expected"])),
        "",
        "## Candidate V2 Hard Block (Research Only)",
        "",
        f"Definition: `{candidate['definition']}`",
        "",
        _table(
            ["Scenario", "N", "Cum", "Mean", "Median", "WR %", "Extra"],
            [
                ["baseline", candidate["baseline"]["n"], candidate["baseline"]["cum_net_bps"], candidate["baseline"]["mean_net_bps"], candidate["baseline"]["median_net_bps"], candidate["baseline"]["win_rate_pct"], ""],
                ["blocked_bucket", candidate["hard_block_candidate"]["n"], candidate["hard_block_candidate"]["cum_net_bps"], candidate["hard_block_candidate"]["mean_net_bps"], candidate["hard_block_candidate"]["median_net_bps"], candidate["hard_block_candidate"]["win_rate_pct"], ""],
                ["kept_after_block", candidate["kept_after_candidate"]["n"], candidate["kept_after_candidate"]["cum_net_bps"], candidate["kept_after_candidate"]["mean_net_bps"], candidate["kept_after_candidate"]["median_net_bps"], candidate["kept_after_candidate"]["win_rate_pct"], f"delta {candidate['kept_after_candidate']['delta_cum_vs_baseline_bps']}"],
            ],
        ),
        "",
        "Blocked examples:",
        "",
        _table(["Trade", "Rule", "Exit", "Net", "Cluster", "StrongNeg", "MinExp"], blocked_rows) if blocked_rows else "None.",
        "",
        "## Read",
        "",
        "V2 is not promoted. The audit identifies whether warning can be split into a narrower hard-block candidate. Any useful candidate must be forward-tested as a shadow rule before becoming a live filter.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(db_path: Path) -> dict[str, Any]:
    rows = _load_rows(db_path)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "closed_n": len(rows),
        "feature_inventory": _feature_inventory(rows),
        "guardrail_level_breakdown": {level: _metrics([r for r in rows if r.guardrail_level == level]) for level in sorted({r.guardrail_level for r in rows})},
        "warning_split": _warning_split(rows),
        "candidate_v2": _candidate_v2(rows),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 guardrail v2 warning split audit.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    payload = build_payload(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.out_md, payload)
    print(
        json.dumps(
            {
                "closed_n": payload["closed_n"],
                "warning_n": payload["warning_split"]["warning_all"]["n"],
                "warning_winners": payload["warning_split"]["warning_winners"]["n"],
                "warning_losers": payload["warning_split"]["warning_losers"]["n"],
                "candidate_block_n": payload["candidate_v2"]["hard_block_candidate"]["n"],
                "candidate_block_cum_bps": payload["candidate_v2"]["hard_block_candidate"]["cum_net_bps"],
                "candidate_kept_delta_bps": payload["candidate_v2"]["kept_after_candidate"]["delta_cum_vs_baseline_bps"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
