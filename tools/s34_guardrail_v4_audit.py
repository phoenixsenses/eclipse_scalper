from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_GUARDRAIL_V4_AUDIT.md")
DEFAULT_JSON = Path("reports/research/s34/S34_GUARDRAIL_V4_AUDIT.json")
GUARDRAIL_NAME = "guardrail_v4_50k_warning_lt200k"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _metrics(values: Iterable[float]) -> dict[str, Any]:
    vals = list(values)
    wins = [x for x in vals if x > 0]
    return {
        "n": len(vals),
        "cum_net_bps": round(sum(vals), 2) if vals else 0.0,
        "mean_net_bps": round(mean(vals), 2) if vals else 0.0,
        "median_net_bps": round(median(vals), 2) if vals else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / len(vals), 2) if vals else 0.0,
    }


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def build_payload(db_path: Path) -> dict[str, Any]:
    with _connect(db_path) as conn:
        baseline_rows = conn.execute(
            """
            SELECT o.trade_id, o.rule_name, o.exit_reason, o.net_bps
            FROM s34_outcomes o
            ORDER BY o.exit_ts_ms ASC
            """
        ).fetchall()
        rows = conn.execute(
            """
            SELECT
              o.trade_id, o.rule_name, o.exit_reason, o.net_bps,
              s.signal_ts_utc, s.cluster_notional, s.cluster_liq_count,
              sg.action, sg.level, sg.headline, sg.shadow_json
            FROM s34_shadow_guardrails sg
            JOIN s34_signals s ON s.signal_id=sg.signal_id
            LEFT JOIN s34_outcomes o ON o.signal_id=sg.signal_id
            WHERE sg.guardrail_name=?
            ORDER BY s.signal_ts_ms ASC
            """,
            (GUARDRAIL_NAME,),
        ).fetchall()

    baseline = _metrics(float(row["net_bps"] or 0.0) for row in baseline_rows if row["net_bps"] is not None)
    would_block = [row for row in rows if str(row["action"] or "") == "would_block"]
    would_block_closed = [row for row in would_block if row["net_bps"] is not None]
    block_ids = {str(row["trade_id"]) for row in would_block_closed if row["trade_id"]}
    kept = [row for row in baseline_rows if row["net_bps"] is not None and str(row["trade_id"]) not in block_ids]
    kept_metrics = _metrics(float(row["net_bps"] or 0.0) for row in kept)
    blocked_metrics = _metrics(float(row["net_bps"] or 0.0) for row in would_block_closed)
    by_rule: dict[str, list[float]] = {}
    for row in would_block_closed:
        by_rule.setdefault(str(row["rule_name"] or "unknown"), []).append(float(row["net_bps"] or 0.0))
    examples = []
    for row in sorted(would_block_closed, key=lambda item: float(item["net_bps"] or 0.0)):
        shadow = _json_loads(row["shadow_json"])
        examples.append(
            {
                "trade_id": row["trade_id"],
                "rule_name": row["rule_name"],
                "signal_ts_utc": row["signal_ts_utc"],
                "exit_reason": row["exit_reason"],
                "net_bps": round(float(row["net_bps"] or 0.0), 2),
                "cluster_notional": row["cluster_notional"],
                "cluster_liq_count": row["cluster_liq_count"],
                "definition": shadow.get("definition"),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "guardrail_name": GUARDRAIL_NAME,
        "definition": "rule=50K/TP120 AND model_guardrail=warning AND cluster_notional < 200K",
        "baseline": baseline,
        "would_block_signals": len(would_block),
        "would_block_closed": blocked_metrics,
        "kept_after_shadow_block": {
            **kept_metrics,
            "blocked_closed_n": len(would_block_closed),
            "delta_cum_vs_baseline_bps": round(kept_metrics["cum_net_bps"] - baseline["cum_net_bps"], 2),
        },
        "blocked_by_rule": {rule: _metrics(vals) for rule, vals in sorted(by_rule.items())},
        "blocked_examples": examples,
        "verdict": (
            "Strong in-sample shadow reject candidate for 50K weak clusters. "
            "Still shadow-only; promotion requires forward confirmation."
        ),
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    examples = payload["blocked_examples"]
    lines = [
        "# S34 Guardrail V4 Audit",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        f"Guardrail: `{payload['guardrail_name']}`",
        "",
        f"Definition: `{payload['definition']}`",
        "",
        "Scope: shadow-only. No runner/config/live reject was changed by this audit.",
        "",
        "## Result",
        "",
        _table(
            ["Scenario", "N", "Cum", "Mean", "Median", "WR %", "Extra"],
            [
                ["baseline_all_closed", payload["baseline"]["n"], payload["baseline"]["cum_net_bps"], payload["baseline"]["mean_net_bps"], payload["baseline"]["median_net_bps"], payload["baseline"]["win_rate_pct"], ""],
                ["would_block_closed", payload["would_block_closed"]["n"], payload["would_block_closed"]["cum_net_bps"], payload["would_block_closed"]["mean_net_bps"], payload["would_block_closed"]["median_net_bps"], payload["would_block_closed"]["win_rate_pct"], f"signals {payload['would_block_signals']}"],
                ["kept_after_block", payload["kept_after_shadow_block"]["n"], payload["kept_after_shadow_block"]["cum_net_bps"], payload["kept_after_shadow_block"]["mean_net_bps"], payload["kept_after_shadow_block"]["median_net_bps"], payload["kept_after_shadow_block"]["win_rate_pct"], f"delta {payload['kept_after_shadow_block']['delta_cum_vs_baseline_bps']}"],
            ],
        ),
        "",
        "## Blocked By Rule",
        "",
        _table(
            ["Rule", "N", "Cum", "Mean", "Median", "WR %"],
            [[rule, m["n"], m["cum_net_bps"], m["mean_net_bps"], m["median_net_bps"], m["win_rate_pct"]] for rule, m in payload["blocked_by_rule"].items()],
        ),
        "",
        "## Blocked Examples",
        "",
        _table(
            ["Trade", "Exit", "Net", "Cluster", "Count", "Signal UTC"],
            [
                [
                    ex["trade_id"],
                    ex["exit_reason"],
                    ex["net_bps"],
                    "" if ex["cluster_notional"] is None else round(float(ex["cluster_notional"]), 2),
                    ex["cluster_liq_count"],
                    ex["signal_ts_utc"],
                ]
                for ex in examples
            ],
        )
        if examples
        else "No closed blocked examples.",
        "",
        "## Verdict",
        "",
        payload["verdict"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 guardrail V4 shadow audit.")
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
                "would_block_signals": payload["would_block_signals"],
                "would_block_closed": payload["would_block_closed"],
                "kept_delta_bps": payload["kept_after_shadow_block"]["delta_cum_vs_baseline_bps"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
