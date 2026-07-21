"""Backfill the S34 intelligence ledger from the existing paper trade journal."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_intelligence_ledger as ledger


DEFAULT_TRADES = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
DEFAULT_LEDGER = ROOT / "data" / "s34_intelligence.db"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def backfill(trades_json: Path, ledger_db: Path) -> dict[str, int]:
    payload = _read_json(trades_json)
    trades = payload.get("trades", []) if isinstance(payload, dict) else []
    counts = {"signals": 0, "predictions": 0, "audits": 0, "accepted": 0, "rejected": 0, "closed": 0}
    conn = ledger.connect(ledger_db)
    try:
        for trade in trades:
            rule = trade.get("rule") or {}
            signal = trade.get("signal") or {}
            if not rule or not signal:
                continue
            signal_id = ledger.record_signal(conn, rule, signal)
            counts["signals"] += 1
            rule_name = str(rule.get("name") or "UNKNOWN")
            signal_ts_ms = int(signal.get("ts_ms") or 0)
            net_values = [
                float(row[0])
                for row in conn.execute(
                    "SELECT net_bps FROM s34_outcomes WHERE rule_name=? AND exit_ts_ms<? AND net_bps IS NOT NULL",
                    (rule_name, signal_ts_ms),
                ).fetchall()
            ]
            prediction = {
                "model": "base_rate_v1",
                "model_version": "2026-06-22",
                "rule_name": rule_name,
                "symbol": rule.get("symbol"),
                "direction": rule.get("direction"),
                "signal_ts_ms": signal.get("ts_ms"),
                "cluster_notional": signal.get("liq_total_notional"),
                "cluster_liq_count": signal.get("liq_count"),
                "cluster_shape_label": signal.get("cluster_shape_label"),
                "base_rates": {"same_rule": _stats(net_values)},
                "expected_net_bps": _stats(net_values).get("median_net_bps"),
                "confidence_note": "usable" if len(net_values) >= 20 else "thin",
                "backfilled": True,
            }
            audit = _neighbor_audit(conn, rule, signal)
            knn = audit.get("knn_v0") or {}
            prediction["knn_v0_expected_net_bps"] = knn.get("median_net_bps")
            prediction["knn_v0_win_rate"] = knn.get("win_rate")
            prediction["knn_v0_k"] = knn.get("k")
            prediction["knn_v0_avg_similarity"] = knn.get("avg_similarity")
            ledger.record_prediction(conn, signal_id, "base_rate_v1", "2026-06-22", prediction)
            counts["predictions"] += 1
            knn_v0_prediction = {
                "model": "knn_v0",
                "model_version": "2026-06-22",
                "rule_name": rule_name,
                "symbol": rule.get("symbol"),
                "direction": rule.get("direction"),
                "signal_ts_ms": signal.get("ts_ms"),
                "cluster_notional": signal.get("liq_total_notional"),
                "cluster_liq_count": signal.get("liq_count"),
                "cluster_shape_label": signal.get("cluster_shape_label"),
                "expected_net_bps": knn.get("median_net_bps"),
                "win_rate": knn.get("win_rate"),
                "k": knn.get("k"),
                "avg_similarity": knn.get("avg_similarity"),
                "feature_set": knn.get("feature_set"),
                "confidence_note": "usable"
                if int(knn.get("k") or 0) >= 5 and (knn.get("avg_similarity") or 0) >= 0.4
                else "thin",
                "backfilled": True,
            }
            ledger.record_prediction(
                conn,
                signal_id,
                "knn_v0",
                "2026-06-22",
                knn_v0_prediction,
            )
            counts["predictions"] += 1
            knn_v1 = audit.get("knn_v1") or {}
            knn_v1_prediction = {
                "model": "knn_v1",
                "model_version": "2026-06-22",
                "rule_name": rule_name,
                "symbol": rule.get("symbol"),
                "direction": rule.get("direction"),
                "signal_ts_ms": signal.get("ts_ms"),
                "cluster_notional": signal.get("liq_total_notional"),
                "cluster_liq_count": signal.get("liq_count"),
                "cluster_shape_label": signal.get("cluster_shape_label"),
                "expected_net_bps": knn_v1.get("median_net_bps"),
                "win_rate": knn_v1.get("win_rate"),
                "k": knn_v1.get("k"),
                "avg_similarity": knn_v1.get("avg_similarity"),
                "feature_set": knn_v1.get("feature_set"),
                "confidence_note": "usable"
                if int(knn_v1.get("k") or 0) >= 5 and (knn_v1.get("avg_similarity") or 0) >= 0.4
                else "thin",
                "backfilled": True,
            }
            ledger.record_prediction(
                conn,
                signal_id,
                "knn_v1",
                "2026-06-22",
                knn_v1_prediction,
            )
            counts["predictions"] += 1
            knn_v2 = audit.get("knn_v2") or {}
            knn_v2_prediction = {
                "model": "knn_v2",
                "model_version": "2026-06-22",
                "rule_name": rule_name,
                "symbol": rule.get("symbol"),
                "direction": rule.get("direction"),
                "signal_ts_ms": signal.get("ts_ms"),
                "cluster_notional": signal.get("liq_total_notional"),
                "cluster_liq_count": signal.get("liq_count"),
                "cluster_shape_label": signal.get("cluster_shape_label"),
                "expected_net_bps": knn_v2.get("median_net_bps"),
                "win_rate": knn_v2.get("win_rate"),
                "k": knn_v2.get("k"),
                "avg_similarity": knn_v2.get("avg_similarity"),
                "feature_set": knn_v2.get("feature_set"),
                "confidence_note": "usable"
                if int(knn_v2.get("k") or 0) >= 5 and (knn_v2.get("avg_similarity") or 0) >= 0.4
                else "thin",
                "backfilled": True,
            }
            ledger.record_prediction(
                conn,
                signal_id,
                "knn_v2",
                "2026-06-22",
                knn_v2_prediction,
            )
            counts["predictions"] += 1
            ledger.record_model_audit(conn, signal_id, "base_rate_v1", audit)
            counts["audits"] += 1
            model_guardrail = _model_guardrail([prediction, knn_v0_prediction, knn_v1_prediction, knn_v2_prediction])
            ledger.record_model_guardrail(
                conn,
                signal_id,
                model_guardrail,
            )
            ledger.record_shadow_guardrail(
                conn,
                signal_id,
                _shadow_hard_block_v2(signal, model_guardrail),
            )
            ledger.record_shadow_guardrail(
                conn,
                signal_id,
                _shadow_hard_block_v4_50k_weak_cluster(rule_name, signal, model_guardrail),
            )
            status = str(trade.get("status") or "")
            if status == "SKIPPED":
                reason = str(trade.get("risk_gate_reason") or trade.get("exit_reason") or "UNKNOWN")
                ledger.record_trade_lifecycle(conn, trade, "REJECT", reason)
                counts["rejected"] += 1
            elif status in {"OPEN", "CLOSED"}:
                ledger.record_trade_lifecycle(conn, trade, "ACCEPT", "")
                counts["accepted"] += 1
                if status == "CLOSED":
                    ledger.record_trade_lifecycle(conn, trade, "CLOSE", str(trade.get("exit_reason") or ""))
                    counts["closed"] += 1
        conn.commit()
    finally:
        conn.close()
    return counts


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median_net_bps": None, "mean_net_bps": None, "win_rate": None, "cum_net_bps": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "n": n,
        "median_net_bps": median,
        "mean_net_bps": sum(values) / n,
        "win_rate": sum(1 for value in values if value > 0) / n,
        "cum_net_bps": sum(values),
    }


def _model_guardrail(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for prediction in predictions:
        expected = prediction.get("expected_net_bps")
        if expected is None:
            continue
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            continue
        values.append(
            {
                "model_name": prediction.get("model"),
                "expected_net_bps": expected_value,
                "confidence_note": prediction.get("confidence_note"),
                "k": prediction.get("k"),
                "win_rate": prediction.get("win_rate"),
            }
        )
    if not values:
        return {
            "version": "2026-06-22",
            "level": "unknown",
            "headline": "No usable model prediction yet.",
            "reasons": [],
            "models": [],
            "backfilled": True,
        }
    negative = [row for row in values if row["expected_net_bps"] < 0]
    strongly_negative = [row for row in values if row["expected_net_bps"] <= -30]
    positive = [row for row in values if row["expected_net_bps"] > 0]
    reasons: list[str] = []
    if len(negative) >= 3:
        reasons.append(f"{len(negative)}/{len(values)} models expect negative net bps")
    if strongly_negative:
        names = ", ".join(str(row["model_name"]) for row in strongly_negative[:3])
        reasons.append(f"strong negative warning from {names}")
    if positive and negative:
        reasons.append("models disagree; treat confidence as low")
    if len(negative) >= 3 or len(strongly_negative) >= 2:
        level = "warning"
        headline = "MODEL WARNING: similar signals have negative expectancy."
    elif len(negative) >= 1 and len(positive) >= 1:
        level = "caution"
        headline = "MODEL CAUTION: predictions disagree."
    else:
        level = "ok"
        headline = "MODEL OK: no negative consensus."
    return {"version": "2026-06-22", "level": level, "headline": headline, "reasons": reasons, "models": values, "backfilled": True}


def _shadow_hard_block_v2(signal: dict[str, Any], model_guardrail: dict[str, Any]) -> dict[str, Any]:
    cluster_notional = float(signal.get("liq_total_notional") or 0.0)
    would_block = (
        str(model_guardrail.get("level") or "") == "warning"
        and 100_000.0 <= cluster_notional < 200_000.0
    )
    if would_block:
        level = "hard_block_candidate"
        action = "would_block"
        headline = "SHADOW V2: warning 100K-200K cluster would be blocked."
    else:
        level = "observe"
        action = "observe"
        headline = "SHADOW V2: no hard-block candidate."
    return {
        "name": "guardrail_v2_warning_100k_200k",
        "version": "2026-06-23",
        "action": action,
        "level": level,
        "headline": headline,
        "cluster_notional": cluster_notional,
        "model_guardrail_level": model_guardrail.get("level"),
        "definition": "model_guardrail=warning AND 100K <= cluster_notional < 200K",
        "live_effect": "none_shadow_only",
        "backfilled": True,
    }


def _shadow_hard_block_v4_50k_weak_cluster(
    rule_name: str, signal: dict[str, Any], model_guardrail: dict[str, Any]
) -> dict[str, Any]:
    cluster_notional = float(signal.get("liq_total_notional") or 0.0)
    would_block = (
        rule_name == "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"
        and str(model_guardrail.get("level") or "") == "warning"
        and cluster_notional < 200_000.0
    )
    if would_block:
        level = "hard_block_candidate"
        action = "would_block"
        headline = "SHADOW V4: 50K warning cluster below 200K would be blocked."
    else:
        level = "observe"
        action = "observe"
        headline = "SHADOW V4: no 50K weak-cluster candidate."
    return {
        "name": "guardrail_v4_50k_warning_lt200k",
        "version": "2026-06-24",
        "action": action,
        "level": level,
        "headline": headline,
        "cluster_notional": cluster_notional,
        "rule_name": rule_name,
        "model_guardrail_level": model_guardrail.get("level"),
        "definition": "rule=50K/TP120 AND model_guardrail=warning AND cluster_notional < 200K",
        "live_effect": "none_shadow_only",
        "source_report": "S34_50K_LOSS_POSTMORTEM",
        "backfilled": True,
    }


def _neighbor_audit(conn, rule: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    rule_name = str(rule.get("name") or "UNKNOWN")
    target_notional = max(0.0, float(signal.get("liq_total_notional") or 0.0))
    signal_ts_ms = int(signal.get("ts_ms") or 0)
    target_count = max(0, int(signal.get("liq_count") or 0))
    target_shape = str(signal.get("cluster_shape_label") or "")
    rows = conn.execute(
        """
        SELECT o.trade_id, o.exit_reason, o.net_bps, s.signal_ts_ms,
               s.cluster_notional, s.cluster_liq_count, s.cluster_shape_label, s.features_json
        FROM s34_outcomes o
        JOIN s34_signals s ON s.signal_id=o.signal_id
        WHERE o.rule_name=? AND o.exit_ts_ms<? AND o.net_bps IS NOT NULL
        """,
        (rule_name, signal_ts_ms),
    ).fetchall()
    scored = []
    for row in rows:
        row_notional = max(0.0, float(row[4] or 0.0))
        row_count = max(0, int(row[5] or 0))
        row_shape = str(row[6] or "")
        try:
            row_features = json.loads(str(row[7] or "{}"))
        except json.JSONDecodeError:
            row_features = {}
        notional_distance = abs(math.log1p(target_notional) - math.log1p(row_notional))
        count_distance = abs(target_count - row_count) / max(1.0, float(max(target_count, row_count, 1)))
        shape_penalty = 0.0 if row_shape == target_shape else 0.75
        distance = notional_distance + count_distance + shape_penalty
        duration_distance = _scaled_abs_distance(signal.get("cluster_duration_sec"), row_features.get("cluster_duration_sec"), 180.0)
        max_share_distance = _scaled_abs_distance(
            signal.get("cluster_max_single_liq_share"), row_features.get("cluster_max_single_liq_share"), 100.0
        )
        btc_distance = _scaled_abs_distance(signal.get("btc_pre_return_bps"), row_features.get("btc_pre_return_bps"), 100.0)
        v1_distance = distance + duration_distance + max_share_distance + btc_distance
        v2_distance = count_distance + duration_distance + max_share_distance
        scored.append(
            {
                "trade_id": row[0],
                "exit_reason": row[1],
                "net_bps": float(row[2]),
                "signal_ts_ms": int(row[3] or 0),
                "cluster_notional": row_notional,
                "cluster_liq_count": row_count,
                "cluster_shape_label": row_shape,
                "distance": distance,
                "similarity": 1.0 / (1.0 + distance),
                "v1_distance": v1_distance,
                "v1_similarity": 1.0 / (1.0 + v1_distance),
                "v2_distance": v2_distance,
                "v2_similarity": 1.0 / (1.0 + v2_distance),
                "cluster_duration_sec": row_features.get("cluster_duration_sec"),
                "cluster_max_single_liq_share": row_features.get("cluster_max_single_liq_share"),
                "btc_pre_return_bps": row_features.get("btc_pre_return_bps"),
            }
        )
    neighbors = sorted(scored, key=lambda item: (float(item["distance"]), -int(item["signal_ts_ms"])))[:5]
    v1_neighbors = sorted(scored, key=lambda item: (float(item["v1_distance"]), -int(item["signal_ts_ms"])))[:5]
    v2_neighbors = sorted(scored, key=lambda item: (float(item["v2_distance"]), -int(item["signal_ts_ms"])))[:5]
    knn = {
        "k": len(neighbors),
        "feature_set": ["log_cluster_notional", "cluster_liq_count_ratio", "shape_match"],
        "distance": "abs(log1p(notional_diff))+normalized_count_diff+shape_penalty",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
    }
    knn_v1 = {
        "k": len(v1_neighbors),
        "feature_set": [
            "log_cluster_notional",
            "cluster_liq_count_ratio",
            "shape_match",
            "cluster_duration_sec",
            "max_single_liq_share",
            "btc_pre_return_bps_if_available",
        ],
        "distance": "knn_v0_distance+duration_distance+max_single_share_distance+btc_pre_return_distance",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
        "neighbors": v1_neighbors,
    }
    knn_v2 = {
        "k": len(v2_neighbors),
        "feature_set": ["cluster_liq_count_ratio", "cluster_duration_sec", "max_single_liq_share"],
        "distance": "normalized_count_diff+duration_distance+max_single_share_distance",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
        "neighbors": v2_neighbors,
    }
    explanation = "No same-rule closed outcomes available for neighbor audit."
    if neighbors:
        values = sorted(float(n["net_bps"]) for n in neighbors)
        median_neighbor = values[len(values) // 2]
        knn.update(
            {
                "median_net_bps": median_neighbor,
                "mean_net_bps": sum(values) / len(values),
                "win_rate": sum(1 for value in values if value > 0) / len(values),
                "avg_similarity": sum(float(n["similarity"]) for n in neighbors) / len(neighbors),
            }
        )
    if v1_neighbors:
        v1_values = sorted(float(n["net_bps"]) for n in v1_neighbors)
        knn_v1.update(
            {
                "median_net_bps": v1_values[len(v1_values) // 2],
                "mean_net_bps": sum(v1_values) / len(v1_values),
                "win_rate": sum(1 for value in v1_values if value > 0) / len(v1_values),
                "avg_similarity": sum(float(n["v1_similarity"]) for n in v1_neighbors) / len(v1_neighbors),
            }
        )
    if v2_neighbors:
        v2_values = sorted(float(n["net_bps"]) for n in v2_neighbors)
        knn_v2.update(
            {
                "median_net_bps": v2_values[len(v2_values) // 2],
                "mean_net_bps": sum(v2_values) / len(v2_values),
                "win_rate": sum(1 for value in v2_values if value > 0) / len(v2_values),
                "avg_similarity": sum(float(n["v2_similarity"]) for n in v2_neighbors) / len(v2_neighbors),
            }
        )
        explanation = (
            f"Selected {len(neighbors)} same-rule neighbors by cluster notional, liq_count, and shape. "
            f"Neighbor median net {median_neighbor:.2f} bps."
        )
    return {
        "audit_version": "neighbor_audit_v1",
        "model": "base_rate_v1",
        "rule_name": rule_name,
        "signal_ts_ms": signal.get("ts_ms"),
        "cluster_notional": target_notional,
        "cluster_liq_count": target_count,
        "cluster_shape_label": target_shape,
        "neighbors": neighbors,
        "knn_v0": knn,
        "knn_v1": knn_v1,
        "knn_v2": knn_v2,
        "explanation": explanation,
        "backfilled": True,
    }


def _scaled_abs_distance(left: Any, right: Any, scale: float) -> float:
    if left is None or right is None or left == "" or right == "":
        return 0.0
    try:
        return abs(float(left) - float(right)) / max(1.0, float(scale))
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill S34 intelligence ledger from shadow paper trades JSON.")
    parser.add_argument("--trades-json", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    counts = backfill(args.trades_json, args.ledger_db)
    print(json.dumps({"ledger_db": str(args.ledger_db), **counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
