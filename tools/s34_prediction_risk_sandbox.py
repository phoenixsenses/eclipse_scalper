from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_PREDICTION_RISK_SANDBOX.md")
DEFAULT_JSON = Path("reports/research/s34/S34_PREDICTION_RISK_SANDBOX.json")
DEFAULT_BALANCE_USDT = 40.0
DEFAULT_LEVERAGES = (10, 20, 40, 70)
DEFAULT_RISK_BUDGET_PCT = 2.0


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


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "cum_net_bps": 0.0, "mean_net_bps": None, "median_net_bps": None, "win_rate": None}
    return {
        "n": len(values),
        "cum_net_bps": round(sum(values), 2),
        "mean_net_bps": round(mean(values), 2),
        "median_net_bps": round(median(values), 2),
        "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4),
    }


def _top3_removed(values: list[float]) -> dict[str, Any]:
    if len(values) <= 3:
        return {"n": 0, "cum_net_bps": None, "median_net_bps": None}
    trimmed = sorted(values, reverse=True)[3:]
    return {
        "n": len(trimmed),
        "cum_net_bps": round(sum(trimmed), 2),
        "median_net_bps": round(median(trimmed), 2),
    }


def _route_params(rule_name: str) -> dict[str, float | None]:
    def grab(name: str) -> float | None:
        m = re.search(name + r"(\d+)", rule_name)
        return float(m.group(1)) if m else None

    return {
        "tp_bps": grab("TP"),
        "sl_bps": grab("SL"),
        "be_bps": grab("BE"),
    }


def _money_from_bps(notional: float, bps: float) -> float:
    return notional * bps / 10_000.0


def _scenario(
    *,
    account_equity: float,
    margin_usdt: float,
    leverage: float,
    tp_bps: float,
    sl_bps: float,
    fee_bps: float,
    risk_budget_pct: float,
) -> dict[str, Any]:
    notional = margin_usdt * leverage
    tp_net_bps = tp_bps - fee_bps
    sl_net_bps = -(sl_bps + fee_bps)
    be_net_bps = -fee_bps
    sl_loss = abs(_money_from_bps(notional, sl_net_bps))
    sl_pct_equity = sl_loss / account_equity * 100.0 if account_equity else None
    risk_budget_usdt = account_equity * risk_budget_pct / 100.0
    max_margin_for_budget = (
        risk_budget_usdt / (leverage * (abs(sl_net_bps) / 10_000.0))
        if leverage > 0 and sl_net_bps
        else 0.0
    )
    margin_pct_equity = margin_usdt / account_equity * 100.0 if account_equity else None
    if sl_pct_equity is None:
        risk_label = "unknown"
    elif sl_pct_equity <= risk_budget_pct:
        risk_label = "low"
    elif sl_pct_equity <= risk_budget_pct * 2.5:
        risk_label = "high"
    else:
        risk_label = "too_high"
    return {
        "leverage": leverage,
        "notional_usdt": round(notional, 2),
        "account_equity_usdt": round(account_equity, 2),
        "margin_usdt": round(margin_usdt, 2),
        "margin_pct_equity": None if margin_pct_equity is None else round(margin_pct_equity, 2),
        "tp_net_bps_est": round(tp_net_bps, 2),
        "tp_usdt_est": round(_money_from_bps(notional, tp_net_bps), 4),
        "be_usdt_est": round(_money_from_bps(notional, be_net_bps), 4),
        "sl_net_bps_est": round(sl_net_bps, 2),
        "sl_usdt_est": round(_money_from_bps(notional, sl_net_bps), 4),
        "sl_pct_equity": None if sl_pct_equity is None else round(sl_pct_equity, 2),
        "risk_budget_usdt": round(risk_budget_usdt, 4),
        "max_margin_for_budget_usdt": round(max_margin_for_budget, 4),
        "budget_ok": sl_loss <= risk_budget_usdt,
        "risk_label": risk_label,
    }


def _confidence(route_stats: dict[str, Any], top3: dict[str, Any], guardrail_level: str, shadow_blocks: list[str]) -> dict[str, Any]:
    reasons: list[str] = []
    n = int(route_stats.get("n") or 0)
    median_net = route_stats.get("median_net_bps")
    top3_cum = top3.get("cum_net_bps")
    if n < 30:
        reasons.append(f"route closed N={n} < 30")
    if median_net is None or float(median_net) <= 0:
        reasons.append("route median net <= 0")
    if top3_cum is None or float(top3_cum) <= 0:
        reasons.append("top3-removed cum net <= 0 or unavailable")
    if guardrail_level == "warning":
        reasons.append("model guardrail is warning")
    if shadow_blocks:
        reasons.append("shadow guardrail would block: " + ", ".join(shadow_blocks))
    if not reasons:
        return {"level": "paper_candidate", "live_gate": "not_approved_without_manual_review", "reasons": []}
    return {"level": "blocked_or_immature", "live_gate": "blocked", "reasons": reasons}


def _route_rank_score(card: dict[str, Any]) -> float:
    stats = card["route_stats"]
    top3 = card["top3_removed"]
    n = float(stats.get("n") or 0)
    median_net = float(stats.get("median_net_bps") or -999.0)
    top3_cum = float(top3.get("cum_net_bps") or -999.0)
    expected = float(card.get("model_expected_median_bps") or median_net)
    confidence_penalty = 0.0 if card["confidence"]["live_gate"] != "blocked" else 40.0
    n_penalty = max(0.0, 30.0 - n) * 1.5
    guard_penalty = 25.0 if str(card["model_guardrail"].get("level") or "") == "warning" else 0.0
    shadow_penalty = 30.0 if card["shadow_blocks"] else 0.0
    return median_net * 0.45 + expected * 0.25 + min(top3_cum, 500.0) * 0.03 - n_penalty - guard_penalty - shadow_penalty - confidence_penalty


def _recommendation_for_card(card: dict[str, Any], risk_budget_pct: float) -> dict[str, Any]:
    scenarios = card["leverage_scenarios"]
    eligible = [row for row in scenarios if row.get("budget_ok") and row.get("risk_label") == "low"]
    if card["confidence"]["live_gate"] == "blocked":
        return {
            "decision": "blocked",
            "headline": "Do not size this route for live use yet.",
            "reason": "; ".join(card["confidence"]["reasons"]) or "route gate blocked",
            "best_leverage": None,
            "max_margin_usdt": None,
        }
    if not eligible:
        best_cap = max(scenarios, key=lambda row: float(row.get("max_margin_for_budget_usdt") or 0.0), default=None)
        return {
            "decision": "paper_only_too_risky_at_requested_margin",
            "headline": f"No tested leverage stays within the {risk_budget_pct:.2f}% SL risk budget at requested margin.",
            "reason": "Reduce margin allocation or use lower leverage.",
            "best_leverage": None,
            "max_margin_usdt": None if best_cap is None else best_cap.get("max_margin_for_budget_usdt"),
        }
    best = max(eligible, key=lambda row: float(row["leverage"]))
    return {
        "decision": "paper_candidate",
        "headline": "Eligible for paper sizing only; live still requires manual review.",
        "reason": "sample, guardrail, and risk budget checks passed for this sandbox rule.",
        "best_leverage": best["leverage"],
        "max_margin_usdt": best["max_margin_for_budget_usdt"],
    }


def build_payload(
    db_path: Path,
    balance: float,
    leverages: tuple[float, ...],
    fee_bps: float,
    risk_budget_pct: float,
    margin_usdt: float | None = None,
) -> dict[str, Any]:
    account_equity = balance
    trade_margin = balance if margin_usdt is None else margin_usdt
    with _connect(db_path) as conn:
        signals = conn.execute(
            """
            SELECT s.*
            FROM s34_signals s
            JOIN (
              SELECT rule_name, MAX(signal_ts_ms) AS max_ts
              FROM s34_signals
              GROUP BY rule_name
            ) latest ON latest.rule_name=s.rule_name AND latest.max_ts=s.signal_ts_ms
            ORDER BY s.signal_ts_ms DESC
            """
        ).fetchall()
        outcome_rows = conn.execute(
            "SELECT rule_name, exit_reason, net_bps FROM s34_outcomes WHERE net_bps IS NOT NULL"
        ).fetchall()
        pred_rows = conn.execute(
            """
            SELECT signal_id, model_name, model_version, predicted_at_utc, prediction_json
            FROM s34_predictions
            ORDER BY predicted_at_utc DESC
            """
        ).fetchall()
        guard_rows = conn.execute(
            "SELECT signal_id, level, headline, guardrail_json FROM s34_model_guardrails"
        ).fetchall()
        shadow_rows = conn.execute(
            "SELECT signal_id, guardrail_name, action, level, headline, shadow_json FROM s34_shadow_guardrails"
        ).fetchall()

    outcomes_by_rule: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in outcome_rows:
        outcomes_by_rule[str(row["rule_name"])].append(row)

    predictions_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        payload = _json_loads(row["prediction_json"])
        predictions_by_signal[str(row["signal_id"])].append(
            {
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "predicted_at_utc": row["predicted_at_utc"],
                "expected_net_bps": payload.get("expected_net_bps"),
                "win_rate": payload.get("win_rate"),
                "k": payload.get("k"),
                "confidence_note": payload.get("confidence_note"),
            }
        )

    guard_by_signal: dict[str, dict[str, Any]] = {}
    for row in guard_rows:
        guard_by_signal[str(row["signal_id"])] = {
            "level": row["level"],
            "headline": row["headline"],
            "payload": _json_loads(row["guardrail_json"]),
        }

    shadow_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shadow_rows:
        shadow_by_signal[str(row["signal_id"])].append(
            {
                "name": row["guardrail_name"],
                "action": row["action"],
                "level": row["level"],
                "headline": row["headline"],
                "payload": _json_loads(row["shadow_json"]),
            }
        )

    cards = []
    for signal in signals:
        rule_name = str(signal["rule_name"])
        params = _route_params(rule_name)
        tp_bps = float(params["tp_bps"] or 60.0)
        sl_bps = float(params["sl_bps"] or 40.0)
        nets = [float(row["net_bps"] or 0.0) for row in outcomes_by_rule.get(rule_name, [])]
        route_stats = _stats(nets)
        top3 = _top3_removed(nets)
        signal_id = str(signal["signal_id"])
        guard = guard_by_signal.get(signal_id, {"level": "unknown", "headline": "No guardrail snapshot."})
        shadows = shadow_by_signal.get(signal_id, [])
        shadow_blocks = [s["name"] for s in shadows if s.get("action") == "would_block"]
        confidence = _confidence(route_stats, top3, str(guard.get("level") or "unknown"), shadow_blocks)
        predictions = predictions_by_signal.get(signal_id, [])[:4]
        expected_values = [
            float(p["expected_net_bps"])
            for p in predictions
            if p.get("expected_net_bps") is not None
        ]
        expected_median = round(median(expected_values), 2) if expected_values else None
        scenarios = [
            _scenario(
                account_equity=account_equity,
                margin_usdt=trade_margin,
                leverage=lev,
                tp_bps=tp_bps,
                sl_bps=sl_bps,
                fee_bps=fee_bps,
                risk_budget_pct=risk_budget_pct,
            )
            for lev in leverages
        ]
        cards.append(
            {
                "signal_id": signal_id,
                "rule_name": rule_name,
                "signal_ts_utc": signal["signal_ts_utc"],
                "cluster_notional": signal["cluster_notional"],
                "cluster_liq_count": signal["cluster_liq_count"],
                "route_params": {"tp_bps": tp_bps, "sl_bps": sl_bps, "be_bps": params["be_bps"]},
                "route_stats": route_stats,
                "top3_removed": top3,
                "model_expected_median_bps": expected_median,
                "predictions": predictions,
                "model_guardrail": guard,
                "shadow_blocks": shadow_blocks,
                "confidence": confidence,
                "leverage_scenarios": scenarios,
            }
        )
        cards[-1]["route_rank_score"] = round(_route_rank_score(cards[-1]), 4)
        cards[-1]["sizing_recommendation"] = _recommendation_for_card(cards[-1], risk_budget_pct)

    cards.sort(key=lambda card: float(card.get("route_rank_score") or -999999.0), reverse=True)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "not_live_trade_advice": True,
        "account_equity_usdt": account_equity,
        "trade_margin_usdt": trade_margin,
        "risk_budget_pct": risk_budget_pct,
        "risk_budget_usdt": round(account_equity * risk_budget_pct / 100.0, 4),
        "fee_bps_round_trip_assumption": fee_bps,
        "leverage_scenarios": list(leverages),
        "cards": cards,
        "read": (
            "This is a paper-only prediction and risk sizing sandbox. It does not approve live orders. "
            "Live gate remains blocked unless sample, guardrail, and manual review criteria pass."
        ),
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# S34 Prediction Risk Sandbox",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        (
            f"Account equity: `${payload['account_equity_usdt']}` | requested trade margin: "
            f"`${payload['trade_margin_usdt']}` | SL risk budget: `{payload['risk_budget_pct']}%` "
            f"(`${payload['risk_budget_usdt']}`). This is paper-only and not a live order recommendation."
        ),
        "",
    ]
    for card in payload["cards"]:
        conf = card["confidence"]
        lines.extend(
            [
                f"## {card['rule_name']}",
                "",
                f"Signal: `{card['signal_ts_utc']}` | cluster `{card['cluster_notional']}` | liq count `{card['cluster_liq_count']}`",
                "",
                _table(
                    ["Metric", "Value"],
                    [
                        ["live_gate", conf["live_gate"]],
                        ["sandbox_decision", card["sizing_recommendation"]["decision"]],
                        ["rank_score", card["route_rank_score"]],
                        ["best_leverage", card["sizing_recommendation"]["best_leverage"]],
                        ["max_margin_at_best_or_cap", card["sizing_recommendation"]["max_margin_usdt"]],
                        ["confidence_level", conf["level"]],
                        ["reasons", "; ".join(conf["reasons"]) or "none"],
                        ["route_N", card["route_stats"]["n"]],
                        ["route_median_bps", card["route_stats"]["median_net_bps"]],
                        ["route_cum_bps", card["route_stats"]["cum_net_bps"]],
                        ["top3_removed_cum", card["top3_removed"]["cum_net_bps"]],
                        ["model_expected_median_bps", card["model_expected_median_bps"]],
                        ["model_guardrail", card["model_guardrail"].get("level")],
                        ["shadow_blocks", ", ".join(card["shadow_blocks"]) or "none"],
                    ],
                ),
                "",
                _table(
                    ["Lev", "Margin", "Notional", "TP est $", "BE est $", "SL est $", "SL % equity", "Max margin for budget", "Risk"],
                    [
                        [
                            row["leverage"],
                            row["margin_usdt"],
                            row["notional_usdt"],
                            row["tp_usdt_est"],
                            row["be_usdt_est"],
                            row["sl_usdt_est"],
                            row["sl_pct_equity"],
                            row["max_margin_for_budget_usdt"],
                            row["risk_label"],
                        ]
                        for row in card["leverage_scenarios"]
                    ],
                ),
                "",
            ]
        )
    lines.append(payload["read"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 paper-only prediction risk sizing sandbox.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--balance-usdt", type=float, default=DEFAULT_BALANCE_USDT)
    parser.add_argument(
        "--margin-usdt",
        type=float,
        default=None,
        help="Per-trade margin allocation. Defaults to using the full balance as margin.",
    )
    parser.add_argument("--leverages", default=",".join(str(x) for x in DEFAULT_LEVERAGES))
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--risk-budget-pct", type=float, default=DEFAULT_RISK_BUDGET_PCT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    leverages = tuple(float(x.strip()) for x in str(args.leverages).split(",") if x.strip())
    payload = build_payload(args.db, args.balance_usdt, leverages, args.fee_bps, args.risk_budget_pct, args.margin_usdt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.out_md, payload)
    print(
        json.dumps(
            {
                "cards": len(payload["cards"]),
                "account_equity_usdt": payload["account_equity_usdt"],
                "trade_margin_usdt": payload["trade_margin_usdt"],
                "risk_budget_pct": payload["risk_budget_pct"],
                "top_routes": [
                    {
                        "rule_name": card["rule_name"],
                        "decision": card["sizing_recommendation"]["decision"],
                        "best_leverage": card["sizing_recommendation"]["best_leverage"],
                        "rank_score": card["route_rank_score"],
                    }
                    for card in payload["cards"][:5]
                ],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
