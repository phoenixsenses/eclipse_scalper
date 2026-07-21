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
from typing import Any, Callable, Iterable


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_GUARDRAIL_V3_AUDIT.md")
DEFAULT_JSON = Path("reports/research/s34/S34_GUARDRAIL_V3_AUDIT.json")
MIN_CANDIDATE_N = 10
MIN_FEATURE_COVERAGE_PCT = 20.0


@dataclass(frozen=True)
class Row:
    trade_id: str
    signal_id: str
    rule_name: str
    signal_ts_utc: str
    exit_reason: str
    net_bps: float
    cluster_notional: float | None
    cluster_liq_count: int | None
    features: dict[str, Any]
    guardrail_level: str
    guardrail_headline: str


@dataclass(frozen=True)
class Candidate:
    name: str
    definition: str
    required_features: tuple[str, ...]
    predicate: Callable[[Row], bool]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _feature(row: Row, name: str) -> float | None:
    aliases = {
        "max_single_liq_share": ("max_single_liq_share", "cluster_max_single_liq_share"),
        "inter_cluster_gap_sec": ("inter_cluster_gap_sec", "prev_liq_gap_sec"),
        "prev_liq_gap_sec": ("prev_liq_gap_sec", "inter_cluster_gap_sec"),
        "btc_pre_15m_bps": ("btc_pre_15m_bps", "btc_pre_return_bps"),
    }
    keys = aliases.get(name, (name,))
    for key in keys:
        value = _float_or_none(row.features.get(key))
        if value is not None:
            return value
    return None


def _has_features(row: Row, names: Iterable[str]) -> bool:
    return all(_feature(row, name) is not None for name in names)


def _load_rows(db_path: Path) -> list[Row]:
    with _connect(db_path) as conn:
        raw = conn.execute(
            """
            SELECT
              o.trade_id, o.signal_id, o.rule_name, o.exit_reason, o.net_bps,
              s.signal_ts_utc, s.cluster_notional, s.cluster_liq_count,
              s.features_json,
              g.level AS guardrail_level, g.headline AS guardrail_headline
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            LEFT JOIN s34_model_guardrails g ON g.signal_id=o.signal_id
            ORDER BY o.exit_ts_ms ASC, o.trade_id ASC
            """
        ).fetchall()

    rows: list[Row] = []
    for item in raw:
        rows.append(
            Row(
                trade_id=str(item["trade_id"]),
                signal_id=str(item["signal_id"]),
                rule_name=str(item["rule_name"]),
                signal_ts_utc=str(item["signal_ts_utc"]),
                exit_reason=str(item["exit_reason"]),
                net_bps=float(item["net_bps"] or 0.0),
                cluster_notional=_float_or_none(item["cluster_notional"]),
                cluster_liq_count=int(item["cluster_liq_count"]) if item["cluster_liq_count"] is not None else None,
                features=_json_loads(item["features_json"]),
                guardrail_level=str(item["guardrail_level"] or "missing"),
                guardrail_headline=str(item["guardrail_headline"] or "missing"),
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


def _candidate_status(candidate_n: int, feature_coverage_pct: float) -> str:
    if candidate_n < MIN_CANDIDATE_N:
        return "too_early_n"
    if feature_coverage_pct < MIN_FEATURE_COVERAGE_PCT:
        return "too_early_feature_coverage"
    return "auditable_shadow_only"


def _candidate_rows() -> list[Candidate]:
    return [
        Candidate(
            "v2_baseline_warning_100k_200k",
            "warning AND 100K <= cluster_notional < 200K",
            (),
            lambda r: r.guardrail_level == "warning"
            and r.cluster_notional is not None
            and 100_000 <= r.cluster_notional < 200_000,
        ),
        Candidate(
            "warning_day_trend_negative",
            "warning AND day_trend_bps < 0",
            ("day_trend_bps",),
            lambda r: r.guardrail_level == "warning" and (_feature(r, "day_trend_bps") or 0.0) < 0,
        ),
        Candidate(
            "warning_100k_200k_day_trend_negative",
            "warning AND 100K <= cluster_notional < 200K AND day_trend_bps < 0",
            ("day_trend_bps",),
            lambda r: r.guardrail_level == "warning"
            and r.cluster_notional is not None
            and 100_000 <= r.cluster_notional < 200_000
            and (_feature(r, "day_trend_bps") or 0.0) < 0,
        ),
        Candidate(
            "warning_max_single_share_ge80",
            "warning AND max_single_liq_share >= 80",
            ("max_single_liq_share",),
            lambda r: r.guardrail_level == "warning" and (_feature(r, "max_single_liq_share") or -1.0) >= 80.0,
        ),
        Candidate(
            "warning_intensity_ge10k",
            "warning AND intensity_per_sec >= 10K",
            ("intensity_per_sec",),
            lambda r: r.guardrail_level == "warning" and (_feature(r, "intensity_per_sec") or -1.0) >= 10_000.0,
        ),
        Candidate(
            "warning_intensity_ge50k",
            "warning AND intensity_per_sec >= 50K",
            ("intensity_per_sec",),
            lambda r: r.guardrail_level == "warning" and (_feature(r, "intensity_per_sec") or -1.0) >= 50_000.0,
        ),
        Candidate(
            "warning_intensity_lt5k",
            "warning AND intensity_per_sec < 5K",
            ("intensity_per_sec",),
            lambda r: r.guardrail_level == "warning"
            and _feature(r, "intensity_per_sec") is not None
            and (_feature(r, "intensity_per_sec") or 0.0) < 5_000.0,
        ),
        Candidate(
            "warning_gap_le5s",
            "warning AND inter_cluster_gap_sec <= 5",
            ("inter_cluster_gap_sec",),
            lambda r: r.guardrail_level == "warning"
            and _feature(r, "inter_cluster_gap_sec") is not None
            and (_feature(r, "inter_cluster_gap_sec") or 999999.0) <= 5.0,
        ),
        Candidate(
            "warning_gap_le60s",
            "warning AND inter_cluster_gap_sec <= 60",
            ("inter_cluster_gap_sec",),
            lambda r: r.guardrail_level == "warning"
            and _feature(r, "inter_cluster_gap_sec") is not None
            and (_feature(r, "inter_cluster_gap_sec") or 999999.0) <= 60.0,
        ),
        Candidate(
            "warning_50k_rule_only",
            "warning AND rule_name = 50K/TP120",
            (),
            lambda r: r.guardrail_level == "warning" and r.rule_name == "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
        ),
        Candidate(
            "warning_50k_rule_day_trend_negative",
            "warning AND rule_name = 50K/TP120 AND day_trend_bps < 0",
            ("day_trend_bps",),
            lambda r: r.guardrail_level == "warning"
            and r.rule_name == "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"
            and (_feature(r, "day_trend_bps") or 0.0) < 0,
        ),
    ]


def _feature_inventory(rows: list[Row]) -> dict[str, Any]:
    desired = [
        "day_trend_bps",
        "day_range_bps",
        "day_buy_liq_notional",
        "day_agg_trade_count",
        "cluster_duration_sec",
        "max_single_liq_share",
        "intensity_per_sec",
        "inter_cluster_gap_sec",
        "prev_liq_gap_sec",
        "btc_pre_15m_bps",
    ]
    inventory = {}
    for name in desired:
        available = [r for r in rows if _feature(r, name) is not None]
        inventory[name] = {
            "closed_rows_with_feature": len(available),
            "coverage_pct": round(100.0 * len(available) / len(rows), 2) if rows else 0.0,
            "warning_rows_with_feature": sum(1 for r in available if r.guardrail_level == "warning"),
        }
    return inventory


def _evaluate_candidate(rows: list[Row], candidate: Candidate) -> dict[str, Any]:
    feature_pool = [r for r in rows if _has_features(r, candidate.required_features)]
    hits = [r for r in feature_pool if candidate.predicate(r)]
    hit_keys = {(r.trade_id, r.signal_id) for r in hits}
    kept = [r for r in rows if (r.trade_id, r.signal_id) not in hit_keys]
    baseline = _metrics(rows)
    kept_metrics = _metrics(kept)
    coverage_pct = round(100.0 * len(feature_pool) / len(rows), 2) if rows else 0.0
    hit_metrics = _metrics(hits)
    return {
        "name": candidate.name,
        "definition": candidate.definition,
        "required_features": list(candidate.required_features),
        "feature_available_n": len(feature_pool),
        "feature_coverage_pct": coverage_pct,
        "status": _candidate_status(len(hits), coverage_pct if candidate.required_features else 100.0),
        "candidate": hit_metrics,
        "kept_after_shadow_block": {
            **kept_metrics,
            "blocked_n": len(hits),
            "delta_cum_vs_baseline_bps": round(kept_metrics["cum_net_bps"] - baseline["cum_net_bps"], 2),
        },
        "examples": [
            {
                "trade_id": r.trade_id,
                "rule_name": r.rule_name,
                "exit_reason": r.exit_reason,
                "net_bps": round(r.net_bps, 2),
                "cluster_notional": r.cluster_notional,
                "day_trend_bps": _feature(r, "day_trend_bps"),
                "max_single_liq_share": _feature(r, "max_single_liq_share"),
                "intensity_per_sec": _feature(r, "intensity_per_sec"),
                "inter_cluster_gap_sec": _feature(r, "inter_cluster_gap_sec"),
            }
            for r in sorted(hits, key=lambda row: row.net_bps)[:20]
        ],
    }


def _by_rule_warning(rows: list[Row]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.guardrail_level == "warning":
            buckets[row.rule_name].append(row)
    return {key: _metrics(value) for key, value in sorted(buckets.items())}


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _candidate_table_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for item in candidates:
        m = item["candidate"]
        kept = item["kept_after_shadow_block"]
        rows.append(
            [
                item["name"],
                item["status"],
                item["feature_available_n"],
                item["feature_coverage_pct"],
                m["n"],
                m["cum_net_bps"],
                m["median_net_bps"],
                m["win_rate_pct"],
                kept["delta_cum_vs_baseline_bps"],
            ]
        )
    return rows


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    candidate_rows = _candidate_table_rows(payload["candidates"])
    feature_rows = [
        [name, data["closed_rows_with_feature"], data["coverage_pct"], data["warning_rows_with_feature"]]
        for name, data in payload["feature_inventory"].items()
    ]
    rule_rows = [
        [rule, m["n"], m["cum_net_bps"], m["mean_net_bps"], m["median_net_bps"], m["win_rate_pct"]]
        for rule, m in payload["warning_by_rule"].items()
    ]
    lines = [
        "# S34 Guardrail V3 Audit",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "Scope: closed intelligence-ledger trades only. This is shadow/audit work. No runner, config, or live filter changed.",
        "",
        "## Read",
        "",
        "V3 tests whether the newer no-lookahead features can refine the V2 warning bucket. A row marked `too_early_*` is not a failed idea; it means the closed sample or feature coverage is not large enough to promote or reject it.",
        "",
        "## Baseline",
        "",
        _table(
            ["Bucket", "N", "Cum", "Mean", "Median", "WR %"],
            [
                [
                    "all_closed",
                    payload["baseline"]["n"],
                    payload["baseline"]["cum_net_bps"],
                    payload["baseline"]["mean_net_bps"],
                    payload["baseline"]["median_net_bps"],
                    payload["baseline"]["win_rate_pct"],
                ],
                [
                    "warning_closed",
                    payload["warning_all"]["n"],
                    payload["warning_all"]["cum_net_bps"],
                    payload["warning_all"]["mean_net_bps"],
                    payload["warning_all"]["median_net_bps"],
                    payload["warning_all"]["win_rate_pct"],
                ],
            ],
        ),
        "",
        "## Feature Coverage On Closed Trades",
        "",
        _table(["Feature", "Closed Rows", "Coverage %", "Warning Rows"], feature_rows),
        "",
        "## Warning By Rule",
        "",
        _table(["Rule", "N", "Cum", "Mean", "Median", "WR %"], rule_rows) if rule_rows else "No warning rows.",
        "",
        "## Candidate Shadow Blocks",
        "",
        _table(
            ["Candidate", "Status", "Feature N", "Coverage %", "Block N", "Block Cum", "Block Median", "Block WR %", "Kept Delta"],
            candidate_rows,
        ),
        "",
        "## Candidate Examples",
        "",
    ]
    for candidate in payload["candidates"]:
        lines.extend(
            [
                f"### {candidate['name']}",
                "",
                f"Definition: `{candidate['definition']}`",
                "",
                f"Status: `{candidate['status']}`",
                "",
            ]
        )
        examples = candidate["examples"]
        if examples:
            lines.append(
                _table(
                    ["Trade", "Rule", "Exit", "Net", "Cluster", "Trend", "Share", "Intensity", "Gap"],
                    [
                        [
                            ex["trade_id"],
                            ex["rule_name"],
                            ex["exit_reason"],
                            ex["net_bps"],
                            "" if ex["cluster_notional"] is None else round(float(ex["cluster_notional"]), 2),
                            "" if ex["day_trend_bps"] is None else round(float(ex["day_trend_bps"]), 2),
                            "" if ex["max_single_liq_share"] is None else round(float(ex["max_single_liq_share"]), 2),
                            "" if ex["intensity_per_sec"] is None else round(float(ex["intensity_per_sec"]), 2),
                            "" if ex["inter_cluster_gap_sec"] is None else round(float(ex["inter_cluster_gap_sec"]), 2),
                        ]
                        for ex in examples
                    ],
                )
            )
        else:
            lines.append("No closed examples yet.")
        lines.append("")
    lines.extend(
        [
            "## Verdict",
            "",
            payload["verdict"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(db_path: Path) -> dict[str, Any]:
    rows = _load_rows(db_path)
    warnings = [r for r in rows if r.guardrail_level == "warning"]
    candidates = [_evaluate_candidate(rows, candidate) for candidate in _candidate_rows()]
    strong = [
        item
        for item in candidates
        if item["status"] == "auditable_shadow_only"
        and item["candidate"]["n"] >= MIN_CANDIDATE_N
        and item["kept_after_shadow_block"]["delta_cum_vs_baseline_bps"] > 0
    ]
    if strong:
        verdict = (
            "At least one V3 candidate has enough closed sample to audit as a shadow block. "
            "Do not promote automatically; next step is forward shadow tracking."
        )
    else:
        verdict = (
            "V3 engine is ready, but the newer feature sample is still too sparse for a new hard-block rule. "
            "Keep V2 shadow running and rerun this audit after more feature-complete closed trades."
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": _metrics(rows),
        "warning_all": _metrics(warnings),
        "feature_inventory": _feature_inventory(rows),
        "warning_by_rule": _by_rule_warning(rows),
        "candidates": candidates,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 guardrail v3 feature-aware shadow audit.")
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
                "closed_n": payload["baseline"]["n"],
                "warning_n": payload["warning_all"]["n"],
                "candidate_count": len(payload["candidates"]),
                "auditable_candidates": [
                    item["name"] for item in payload["candidates"] if item["status"] == "auditable_shadow_only"
                ],
                "verdict": payload["verdict"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
