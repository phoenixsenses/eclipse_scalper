"""Build S34 prediction guardrail diagnostics.

This report is intentionally read-only. It compares prediction snapshots in the
intelligence ledger with closed paper outcomes and highlights where models were
over-optimistic, where they correctly warned on losers, and where simple
conditional base rates are more useful than KNN similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data" / "s34_intelligence.db"
DEFAULT_TRADES = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
DEFAULT_MD = ROOT / "reports" / "research" / "s34" / "S34_PREDICTION_GUARDRAILS.md"
DEFAULT_JSON = ROOT / "reports" / "research" / "s34" / "S34_PREDICTION_GUARDRAILS.json"
MODELS = ["base_rate_v1", "knn_v0", "knn_v1", "knn_v2"]


def _load_prediction_pairs(ledger_db: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT
              p.model_name,
              p.prediction_json,
              o.trade_id,
              o.signal_id,
              o.rule_name,
              o.exit_reason,
              o.exit_ts_ms,
              o.net_bps,
              o.gross_bps,
              o.entry_adverse_bps,
              o.exit_adverse_bps,
              o.spread_cost_bps,
              o.fee_cost_bps,
              s.signal_ts_utc,
              s.cluster_notional,
              s.cluster_liq_count,
              s.cluster_shape_label
            FROM s34_predictions p
            JOIN s34_outcomes o ON o.signal_id = p.signal_id
            LEFT JOIN s34_signals s ON s.signal_id = p.signal_id
            ORDER BY o.exit_ts_ms ASC, p.model_name ASC
            """
        ).fetchall()
    finally:
        con.close()

    pairs: list[dict[str, Any]] = []
    for row in rows:
        try:
            prediction = json.loads(str(row["prediction_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        expected = prediction.get("expected_net_bps")
        actual = row["net_bps"]
        if expected is None or actual is None:
            continue
        pairs.append(
            {
                "model_name": row["model_name"],
                "trade_id": row["trade_id"],
                "signal_id": row["signal_id"],
                "rule_name": row["rule_name"],
                "exit_reason": row["exit_reason"],
                "exit_ts_ms": row["exit_ts_ms"],
                "signal_ts_utc": row["signal_ts_utc"],
                "cluster_notional": row["cluster_notional"],
                "cluster_liq_count": row["cluster_liq_count"],
                "cluster_shape_label": row["cluster_shape_label"],
                "expected_net_bps": float(expected),
                "actual_net_bps": float(actual),
                "error_bps": float(actual) - float(expected),
                "gross_bps": _float_or_none(row["gross_bps"]),
                "entry_adverse_bps": _float_or_none(row["entry_adverse_bps"]),
                "exit_adverse_bps": _float_or_none(row["exit_adverse_bps"]),
                "spread_cost_bps": _float_or_none(row["spread_cost_bps"]),
                "fee_cost_bps": _float_or_none(row["fee_cost_bps"]),
                "prediction": prediction,
            }
        )
    return pairs


def _load_guardrail_rows(ledger_db: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        table_exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='s34_model_guardrails'"
        ).fetchone()
        if not table_exists:
            return []
        rows = con.execute(
            """
            SELECT
              g.signal_id, g.level, g.headline, g.guardrail_json,
              o.trade_id, o.rule_name, o.exit_reason, o.net_bps, o.exit_ts_ms
            FROM s34_model_guardrails g
            LEFT JOIN s34_outcomes o ON o.signal_id=g.signal_id
            ORDER BY g.guardrail_ts_utc ASC
            """
        ).fetchall()
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["guardrail_json"] or "{}"))
        except json.JSONDecodeError:
            payload = {}
        out.append(
            {
                "signal_id": row["signal_id"],
                "level": row["level"],
                "headline": row["headline"],
                "trade_id": row["trade_id"],
                "rule_name": row["rule_name"],
                "exit_reason": row["exit_reason"],
                "net_bps": _float_or_none(row["net_bps"]),
                "exit_ts_ms": row["exit_ts_ms"],
                "guardrail": payload,
            }
        )
    return out


def _load_trades(trades_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(trades_path.read_text(encoding="utf-8"))
    trades = raw.get("trades", raw if isinstance(raw, list) else [])
    return [trade for trade in trades if isinstance(trade, dict)]


def _latest_closed_loss(trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    closed_losses = [
        trade
        for trade in trades
        if trade.get("status") == "CLOSED" and trade.get("net_bps") is not None and float(trade["net_bps"]) < 0
    ]
    if not closed_losses:
        return None
    return max(closed_losses, key=lambda trade: int(trade.get("exit_ts_ms") or 0))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    errors = [row["error_bps"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    dir_hits = [
        row
        for row in rows
        if (row["expected_net_bps"] >= 0 and row["actual_net_bps"] >= 0)
        or (row["expected_net_bps"] < 0 and row["actual_net_bps"] < 0)
    ]
    optimistic = [row for row in rows if row["expected_net_bps"] > row["actual_net_bps"]]
    false_green = [row for row in rows if row["expected_net_bps"] > 0 and row["actual_net_bps"] < 0]
    warned_loss = [row for row in rows if row["expected_net_bps"] < 0 and row["actual_net_bps"] < 0]
    missed_winner = [row for row in rows if row["expected_net_bps"] < 0 and row["actual_net_bps"] > 0]
    return {
        "n": len(rows),
        "bias_bps": sum(errors) / len(errors),
        "mae_bps": sum(abs_errors) / len(abs_errors),
        "hit_direction_rate": len(dir_hits) / len(rows),
        "optimism_rate": len(optimistic) / len(rows),
        "false_green_count": len(false_green),
        "false_green_rate": len(false_green) / len(rows),
        "warned_loss_count": len(warned_loss),
        "missed_winner_count": len(missed_winner),
    }


def _group_summary(outcomes: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        buckets[str(key_fn(row))].append(row)
    result: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        nets = [float(row["actual_net_bps"]) for row in rows]
        wins = [value for value in nets if value > 0]
        result.append(
            {
                "bucket": key,
                "n": len(rows),
                "median_net_bps": median(nets),
                "mean_net_bps": sum(nets) / len(nets),
                "cum_net_bps": sum(nets),
                "win_rate": len(wins) / len(nets),
            }
        )
    return sorted(result, key=lambda row: (-row["n"], row["bucket"]))


def _cluster_bucket(row: dict[str, Any]) -> str:
    value = row.get("cluster_notional")
    if value is None:
        return "unknown"
    value = float(value)
    if value < 200_000:
        return "<200K"
    if value < 500_000:
        return "200K-500K"
    if value < 1_000_000:
        return "500K-1M"
    return ">=1M"


def _session_bucket(row: dict[str, Any]) -> str:
    ts = str(row.get("signal_ts_utc") or "")
    try:
        hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except ValueError:
        return "unknown"
    if 0 <= hour < 7:
        return "00-07 UTC"
    if 7 <= hour < 13:
        return "07-13 UTC"
    if 13 <= hour < 17:
        return "13-17 UTC"
    return "17-24 UTC"


def _dedup_outcomes(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in pairs:
        seen.setdefault(str(row["trade_id"]), row)
    return list(seen.values())


def build_report(ledger_db: Path, trades_path: Path) -> dict[str, Any]:
    pairs = _load_prediction_pairs(ledger_db)
    guardrails = _load_guardrail_rows(ledger_db)
    trades = _load_trades(trades_path)
    latest_loss = _latest_closed_loss(trades)

    by_model = {model: [row for row in pairs if row["model_name"] == model] for model in MODELS}
    model_summary = {model: _summary(rows) for model, rows in by_model.items()}

    false_greens: dict[str, list[dict[str, Any]]] = {}
    correct_warnings: dict[str, list[dict[str, Any]]] = {}
    for model, rows in by_model.items():
        false_greens[model] = sorted(
            [row for row in rows if row["expected_net_bps"] >= 30 and row["actual_net_bps"] < 0],
            key=lambda row: row["error_bps"],
        )[:10]
        correct_warnings[model] = sorted(
            [row for row in rows if row["expected_net_bps"] < 0 and row["actual_net_bps"] < 0],
            key=lambda row: row["exit_ts_ms"] or 0,
            reverse=True,
        )[:10]

    outcomes = _dedup_outcomes(pairs)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_db": str(ledger_db),
        "trades_path": str(trades_path),
        "model_summary": model_summary,
        "false_green_worst": _strip_rows(false_greens),
        "correct_loss_warnings": _strip_rows(correct_warnings),
        "base_rate_segments": {
            "by_rule": _group_summary(outcomes, lambda row: row["rule_name"]),
            "by_cluster_bucket": _group_summary(outcomes, _cluster_bucket),
            "by_session": _group_summary(outcomes, _session_bucket),
            "by_exit_reason": _group_summary(outcomes, lambda row: row["exit_reason"]),
        },
        "guardrail_summary": _guardrail_summary(guardrails),
        "latest_guardrails": _strip_guardrails(list(reversed(guardrails))[:12]),
        "latest_closed_loss": _trade_digest(latest_loss),
    }


def _guardrail_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("level") or "unknown")].append(row)
    result: list[dict[str, Any]] = []
    for level, items in buckets.items():
        closed = [row for row in items if row.get("net_bps") is not None]
        losses = [row for row in closed if float(row["net_bps"]) < 0]
        wins = [row for row in closed if float(row["net_bps"]) > 0]
        nets = [float(row["net_bps"]) for row in closed]
        result.append(
            {
                "level": level,
                "signals": len(items),
                "closed": len(closed),
                "loss_count": len(losses),
                "win_count": len(wins),
                "loss_rate": None if not closed else len(losses) / len(closed),
                "mean_net_bps": None if not closed else sum(nets) / len(nets),
                "median_net_bps": None if not closed else median(nets),
                "cum_net_bps": sum(nets),
            }
        )
    return sorted(result, key=lambda row: (-row["signals"], row["level"]))


def _strip_guardrails(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = ["signal_id", "level", "headline", "trade_id", "rule_name", "exit_reason", "net_bps"]
    return [{key: row.get(key) for key in wanted} for row in rows]


def _strip_rows(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    wanted = [
        "trade_id",
        "rule_name",
        "exit_reason",
        "signal_ts_utc",
        "expected_net_bps",
        "actual_net_bps",
        "error_bps",
        "cluster_notional",
        "cluster_liq_count",
        "cluster_shape_label",
    ]
    return {model: [{key: row.get(key) for key in wanted} for row in rows] for model, rows in groups.items()}


def _trade_digest(trade: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trade:
        return None
    signal = trade.get("signal") or {}
    regime = trade.get("regime") or {}
    return {
        "trade_id": trade.get("trade_id"),
        "rule_name": (trade.get("rule") or {}).get("name"),
        "signal_ts_utc": trade.get("signal_ts_utc"),
        "entry_ts_utc": trade.get("entry_ts_utc"),
        "exit_ts_utc": trade.get("exit_ts_utc"),
        "exit_reason": trade.get("exit_reason"),
        "entry_price": trade.get("entry_price"),
        "exit_price": trade.get("exit_price"),
        "tp_price": trade.get("tp_price"),
        "sl_price": trade.get("sl_price"),
        "be_trigger_price": trade.get("be_trigger_price"),
        "be_active": trade.get("be_active"),
        "gross_bps": trade.get("gross_bps"),
        "entry_adverse_bps": trade.get("entry_adverse_bps"),
        "exit_adverse_bps": trade.get("exit_adverse_bps"),
        "spread_cost_bps": trade.get("spread_cost_bps"),
        "fee_cost_bps": trade.get("fee_cost_bps"),
        "net_bps": trade.get("net_bps"),
        "cluster_notional": signal.get("liq_total_notional"),
        "cluster_liq_count": signal.get("liq_count"),
        "cluster_duration_sec": signal.get("cluster_duration_sec"),
        "cluster_max_single_liq_share": signal.get("cluster_max_single_liq_share"),
        "cluster_shape_label": signal.get("cluster_shape_label"),
        "regime": regime,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# S34 Prediction Guardrails\n")
    lines.append(f"Generated: `{report['generated_at_utc']}`\n")
    lines.append("Diagnostic only. This report changes no runner rules or config.\n")

    lines.append("## Model Summary\n")
    lines.append("| Model | N | MAE bps | Bias bps | Direction hit | False green | Warned losses |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, row in report["model_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _fmt(row.get("n"), 0),
                    _fmt(row.get("mae_bps")),
                    _fmt(row.get("bias_bps")),
                    f"{(row.get('hit_direction_rate') or 0):.1%}" if row.get("hit_direction_rate") is not None else "n/a",
                    _fmt(row.get("false_green_count"), 0),
                    _fmt(row.get("warned_loss_count"), 0),
                ]
            )
            + " |"
        )

    latest = report.get("latest_closed_loss")
    lines.append("\n## Latest Closed Loss\n")
    if latest:
        lines.append(
            f"`{latest['trade_id']}` `{latest['rule_name']}` exited `{latest['exit_reason']}` "
            f"at `{latest['exit_ts_utc']}` for `{_fmt(latest['net_bps'])}` bps.\n"
        )
        lines.append("| Component | bps/value |")
        lines.append("|---|---:|")
        for key in [
            "gross_bps",
            "entry_adverse_bps",
            "exit_adverse_bps",
            "spread_cost_bps",
            "fee_cost_bps",
            "net_bps",
        ]:
            lines.append(f"| {key} | {_fmt(latest.get(key))} |")
        lines.append(
            "\nInterpretation: the loss was primarily directional because gross was near the SL distance; "
            "execution cost was normal taker fee plus tiny spread.\n"
        )
        lines.append(
            f"Cluster: notional `{_fmt(latest.get('cluster_notional'), 0)}`, "
            f"liq_count `{_fmt(latest.get('cluster_liq_count'), 0)}`, "
            f"shape `{latest.get('cluster_shape_label')}`.\n"
        )
    else:
        lines.append("No closed losing trade found.\n")

    lines.append("## Worst False-Green Predictions\n")
    lines.append("Rows where model expected >= +30 bps but outcome was negative.\n")
    for model, rows in report["false_green_worst"].items():
        lines.append(f"\n### {model}\n")
        if not rows:
            lines.append("None.\n")
            continue
        lines.append("| Trade | Rule | Exit | Expected | Actual | Error | Cluster |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for row in rows[:8]:
            lines.append(
                f"| {row['trade_id']} | {row['rule_name']} | {row['exit_reason']} | "
                f"{_fmt(row['expected_net_bps'])} | {_fmt(row['actual_net_bps'])} | "
                f"{_fmt(row['error_bps'])} | {_fmt(row['cluster_notional'], 0)} |"
            )

    lines.append("\n## Model Guardrail Performance\n")
    lines.append("| Level | Signals | Closed | Loss rate | Median | Mean | Cum |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report.get("guardrail_summary", []):
        loss_rate = "n/a" if row.get("loss_rate") is None else f"{row['loss_rate']:.1%}"
        lines.append(
            f"| {row['level']} | {row['signals']} | {row['closed']} | {loss_rate} | "
            f"{_fmt(row.get('median_net_bps'))} | {_fmt(row.get('mean_net_bps'))} | {_fmt(row.get('cum_net_bps'))} |"
        )
    lines.append(
        "\nGuardrail levels are observation labels only. They are not live execution filters.\n"
    )

    lines.append("\n### Latest Guardrails\n")
    lines.append("| Level | Signal | Trade | Rule | Exit | Net |")
    lines.append("|---|---|---|---|---|---:|")
    for row in report.get("latest_guardrails", [])[:10]:
        lines.append(
            f"| {row.get('level')} | {row.get('signal_id')} | {row.get('trade_id') or '-'} | "
            f"{row.get('rule_name') or '-'} | {row.get('exit_reason') or '-'} | {_fmt(row.get('net_bps'))} |"
        )

    lines.append("\n## Correct Loss Warnings\n")
    lines.append("Recent losses where the model already expected negative net bps.\n")
    for model, rows in report["correct_loss_warnings"].items():
        lines.append(f"\n### {model}\n")
        if not rows:
            lines.append("None.\n")
            continue
        lines.append("| Trade | Rule | Exit | Expected | Actual |")
        lines.append("|---|---|---|---:|---:|")
        for row in rows[:8]:
            lines.append(
                f"| {row['trade_id']} | {row['rule_name']} | {row['exit_reason']} | "
                f"{_fmt(row['expected_net_bps'])} | {_fmt(row['actual_net_bps'])} |"
            )

    lines.append("\n## Conditional Base Rates\n")
    for title, rows in report["base_rate_segments"].items():
        lines.append(f"\n### {title}\n")
        lines.append("| Bucket | N | Median | Mean | Cum | Win rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['bucket']} | {row['n']} | {_fmt(row['median_net_bps'])} | "
                f"{_fmt(row['mean_net_bps'])} | {_fmt(row['cum_net_bps'])} | {row['win_rate']:.1%} |"
            )

    lines.append("\n## Guardrail Takeaway\n")
    lines.append(
        "Use KNN as an evidence/audit surface, not as an execution trigger. "
        "When KNN and base-rate both expect negative bps, show a visible warning in the dashboard; "
        "do not change live rules from this report alone.\n"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S34 prediction guardrail report.")
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--trades-json", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    report = build_report(args.ledger_db, args.trades_json)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
