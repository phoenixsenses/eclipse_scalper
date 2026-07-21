from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_DB = Path("data/s34_intelligence.db")


def _connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"intelligence db not found: {db_path}")
    con = sqlite3.connect(str(db_path))
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


def _notional_bucket(value: float | None) -> tuple[str, float | None, float | None]:
    if value is None:
        return "unknown", None, None
    if value < 100_000:
        return "<100K", None, 100_000
    if value < 200_000:
        return "100K-200K", 100_000, 200_000
    if value < 500_000:
        return "200K-500K", 200_000, 500_000
    return ">=500K", 500_000, None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _maybe_mae(row: sqlite3.Row) -> float | None:
    payload = _json_loads(row["outcome_json"])
    for key in ("mae_bps", "min_path_bps", "max_adverse_bps"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    trade = payload.get("trade") if isinstance(payload.get("trade"), dict) else {}
    for key in ("mae_bps", "min_path_bps", "max_adverse_bps"):
        value = _safe_float(trade.get(key))
        if value is not None:
            return value
    return None


def _stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    nets = [_safe_float(row["net_bps"]) for row in rows]
    nets = [v for v in nets if v is not None]
    maes = [_maybe_mae(row) for row in rows]
    maes = [v for v in maes if v is not None]
    top3_removed = sorted(nets, reverse=True)[3:] if len(nets) > 3 else []
    return {
        "closed_n": len(nets),
        "median_net_bps": round(median(nets), 4) if nets else None,
        "mean_net_bps": round(mean(nets), 4) if nets else None,
        "win_rate": round(sum(1 for v in nets if v > 0) / len(nets), 6) if nets else None,
        "cum_net_bps": round(sum(nets), 4) if nets else None,
        "p95_mae_bps": round(_percentile(maes, 0.95), 4) if maes else None,
        "top3_removed_median_net_bps": round(median(top3_removed), 4) if top3_removed else None,
    }


def _route_rows(con: sqlite3.Connection, rule_name: str) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT o.*, s.cluster_notional
        FROM s34_outcomes o
        JOIN s34_signals s ON s.signal_id=o.signal_id
        WHERE o.rule_name=?
          AND o.net_bps IS NOT NULL
        ORDER BY o.exit_ts_ms ASC
        """,
        (rule_name,),
    ).fetchall()


def _bucket_rows(rows: list[sqlite3.Row], lo: float | None, hi: float | None) -> list[sqlite3.Row]:
    out = []
    for row in rows:
        value = _safe_float(row["cluster_notional"])
        if value is None:
            continue
        if lo is not None and value < lo:
            continue
        if hi is not None and value >= hi:
            continue
        out.append(row)
    return out


def _model_predictions(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        payload = _json_loads(row["prediction_json"])
        out.append(
            {
                "prediction_id": row["prediction_id"],
                "model_id": row["model_name"],
                "model_version": row["model_version"],
                "expected_net_bps": _safe_float(payload.get("expected_net_bps")),
                "confidence": payload.get("confidence_note"),
                "neighbor_n": payload.get("k"),
                "win_rate": payload.get("win_rate"),
            }
        )
    return out


def _shadow_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        payload = _json_loads(row["shadow_json"])
        action = str(row["action"] or "")
        out.append(
            {
                "guardrail_id": row["guardrail_name"],
                "action": action,
                "level": row["level"],
                "headline": row["headline"],
                "would_block": action == "would_block" or "block" in action.lower() or bool(payload.get("would_block")),
                "triggered": action != "observe",
            }
        )
    return out


def _verdict(
    route_stats: dict[str, Any],
    predictions: list[dict[str, Any]],
    model_guardrail: dict[str, Any],
    shadows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    level = str(model_guardrail.get("level") or "unknown")
    shadow_block = any(row.get("would_block") for row in shadows)
    closed_n = int(route_stats.get("closed_n") or 0)
    top3_median = route_stats.get("top3_removed_median_net_bps")
    positive_models = sum(1 for row in predictions if (row.get("expected_net_bps") is not None and float(row["expected_net_bps"]) > 0))

    if level == "warning":
        return "BLOCKED", ["model_guardrail=warning"]
    if shadow_block:
        return "BLOCKED", ["shadow_guardrail_would_block"]
    if closed_n < 10:
        return "BLOCKED", [f"route_closed_n={closed_n} < 10"]
    if top3_median is None:
        return "BLOCKED", ["top3_removed_median unavailable"]
    if float(top3_median) < 0:
        return "BLOCKED", [f"top3_removed_median={top3_median} < 0"]

    if level in {"ok", "caution"} and not shadow_block and closed_n >= 10 and float(top3_median) >= 0 and positive_models >= 2:
        reasons.extend(
            [
                f"route_closed_n={closed_n} >= 10",
                f"top3_removed_median={top3_median} >= 0",
                f"model_guardrail={level}",
                "no_shadow_block",
                f"{positive_models}/{len(predictions)} models positive",
            ]
        )
        return "PAPER_CANDIDATE", reasons

    reasons.extend(
        [
            f"model_guardrail={level}",
            "no_shadow_block" if not shadow_block else "shadow_block",
            f"route_closed_n={closed_n}",
            f"top3_removed_median={top3_median}",
            f"{positive_models}/{len(predictions)} models positive",
        ]
    )
    if not predictions:
        reasons.append("no_predictions")
    return "RESEARCH_ONLY", reasons


def generate_prediction_card(db_path: str | Path = DEFAULT_DB, signal_id: str | None = None) -> dict[str, Any]:
    with _connect(db_path) as con:
        if signal_id:
            signal = con.execute("SELECT * FROM s34_signals WHERE signal_id=?", (signal_id,)).fetchone()
        else:
            signal = con.execute("SELECT * FROM s34_signals ORDER BY signal_ts_ms DESC LIMIT 1").fetchone()
        if signal is None:
            raise RuntimeError("no signals found")
        predictions = _model_predictions(
            con.execute("SELECT * FROM s34_predictions WHERE signal_id=? ORDER BY model_name", (signal["signal_id"],)).fetchall()
        )
        guard_row = con.execute(
            "SELECT * FROM s34_model_guardrails WHERE signal_id=? ORDER BY guardrail_ts_utc DESC LIMIT 1",
            (signal["signal_id"],),
        ).fetchone()
        if guard_row:
            guard_payload = _json_loads(guard_row["guardrail_json"])
            model_guardrail = {
                "level": guard_row["level"],
                "reason": guard_row["headline"],
                "headline": guard_row["headline"],
                "details": guard_payload,
            }
        else:
            model_guardrail = {"level": "unknown", "reason": "no_model_guardrail", "headline": "No model guardrail."}
        shadows = _shadow_rows(
            con.execute("SELECT * FROM s34_shadow_guardrails WHERE signal_id=? ORDER BY guardrail_name", (signal["signal_id"],)).fetchall()
        )
        rows = _route_rows(con, signal["rule_name"])
        route_stats = _stats(rows)
        bucket, lo, hi = _notional_bucket(_safe_float(signal["cluster_notional"]))
        bucket_stats = _stats(_bucket_rows(rows, lo, hi))
        verdict, reasons = _verdict(route_stats, predictions, model_guardrail, shadows)
        warnings = []
        if route_stats.get("p95_mae_bps") is None:
            warnings.append("mae_unavailable")
        if not predictions:
            warnings.append("no_predictions")
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "signal_id": signal["signal_id"],
            "signal_ts_utc": signal["signal_ts_utc"],
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "rule_id": signal["rule_name"],
            "cluster_notional": signal["cluster_notional"],
            "notional_bucket": bucket,
            "route_base_rate": route_stats,
            "notional_bucket_base_rate": {"bucket": bucket, **bucket_stats},
            "model_predictions": predictions,
            "model_guardrail": model_guardrail,
            "shadow_guardrails": shadows,
            "verdict": verdict,
            "verdict_reasons": reasons,
            "data_warnings": warnings,
        }


def format_card(card: dict[str, Any]) -> str:
    stats = card["route_base_rate"]
    prefix = "[+]" if card["verdict"] == "PAPER_CANDIDATE" else "[!]" if card["verdict"] == "BLOCKED" else "[-]"
    lines = [
        "=== S34 PREDICTION CARD ===",
        f"Generated : {card['generated_at_utc']}",
        f"Signal    : {card['signal_id']} @ {card['signal_ts_utc']}",
        f"Symbol    : {card['symbol']} {card['direction']} | {card['notional_bucket']} cluster",
        f"Rule      : {card['rule_id']}",
        "",
        f"ROUTE BASE RATE (N={stats['closed_n']})",
        f"  Median net   : {stats['median_net_bps']} bps",
        f"  Top3-removed : {stats['top3_removed_median_net_bps']} bps",
        f"  Win rate     : {None if stats['win_rate'] is None else round(float(stats['win_rate']) * 100, 2)}%",
        f"  P95 MAE      : {stats['p95_mae_bps']} bps",
        "",
        "MODEL PREDICTIONS",
    ]
    if card["model_predictions"]:
        for row in card["model_predictions"]:
            k = "" if row.get("neighbor_n") is None else f", K={row['neighbor_n']}"
            lines.append(f"  {row['model_id']:<12}: {row['expected_net_bps']} bps [{row.get('confidence') or '-'}{k}]")
    else:
        lines.append("  no predictions")
    shadows = card["shadow_guardrails"]
    shadow_state = "triggered" if any(row.get("would_block") for row in shadows) else "not triggered"
    lines.extend(
        [
            "",
            f"MODEL GUARDRAIL : {str(card['model_guardrail'].get('level') or 'unknown').upper()}",
            f"SHADOW GUARDRAIL: {shadow_state}",
            "",
            f"VERDICT : {card['verdict']}",
            "Reasons :",
        ]
    )
    lines.extend(f"  {prefix} {reason}" for reason in card["verdict_reasons"])
    if card["data_warnings"]:
        lines.extend(["", "Warnings :"])
        lines.extend(f"  [-] {warning}" for warning in card["data_warnings"])
    lines.append("===========================")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the current S34 prediction card from the intelligence ledger.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--signal_id", "--signal-id", dest="signal_id", default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    try:
        card = generate_prediction_card(args.db, args.signal_id)
    except Exception as exc:  # noqa: BLE001 - CLI should print a clear error.
        print(f"error: {exc}")
        return 1
    print(format_card(card))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
