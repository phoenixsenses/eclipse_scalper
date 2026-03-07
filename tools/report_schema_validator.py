from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _validate_run_summary(run_summary: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(run_summary, dict):
        return ["bad_type:run_summary"]
    required_top = {
        "version": str,
        "run_type": str,
        "inputs": dict,
        "metrics": dict,
        "artifacts": dict,
    }
    for key, expected in required_top.items():
        if key not in run_summary:
            errors.append(f"missing:run_summary.{key}")
            continue
        if not isinstance(run_summary[key], expected):
            errors.append(f"bad_type:run_summary.{key}")
    return errors


def _validate_micro_edge_record(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ts_utc": str,
        "symbol": (str, type(None)),
        "lookback_min": int,
        "bucket_sec": int,
        "horizon_sec": int,
        "raw_rows": int,
        "bucket_rows": int,
        "label_counts": dict,
        "baseline_hit_rate": (int, float, type(None)),
        "min_rule_n": int,
        "correlations": dict,
        "naive_rules": dict,
        "label_definition": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")

    label_counts = payload.get("label_counts")
    if isinstance(label_counts, dict):
        for key in ("up", "down", "flat"):
            if key not in label_counts:
                errors.append(f"missing:label_counts.{key}")
            elif not isinstance(label_counts[key], int):
                errors.append(f"bad_type:label_counts.{key}")

    label_definition = payload.get("label_definition")
    if isinstance(label_definition, dict):
        for key in (
            "timing",
            "label",
            "horizon_steps",
            "threshold",
            "label_values",
            "hit_definition",
            "baseline_definition",
        ):
            if key not in label_definition:
                errors.append(f"missing:label_definition.{key}")
        if "horizon_steps" in label_definition and not isinstance(label_definition["horizon_steps"], int):
            errors.append("bad_type:label_definition.horizon_steps")
        if "threshold" in label_definition and not _is_number(label_definition["threshold"]):
            errors.append("bad_type:label_definition.threshold")

    naive_rules = payload.get("naive_rules")
    if isinstance(naive_rules, dict):
        for name, rule in naive_rules.items():
            if not isinstance(name, str):
                errors.append("bad_type:naive_rules.key")
                continue
            if not isinstance(rule, dict):
                errors.append(f"bad_type:naive_rules.{name}")
                continue
            for key in ("hit_rate", "n", "delta_vs_baseline"):
                if key not in rule:
                    errors.append(f"missing:naive_rules.{name}.{key}")
            if "n" in rule and not isinstance(rule["n"], int):
                errors.append(f"bad_type:naive_rules.{name}.n")
            if "hit_rate" in rule and rule["hit_rate"] is not None and not _is_number(rule["hit_rate"]):
                errors.append(f"bad_type:naive_rules.{name}.hit_rate")
            if "delta_vs_baseline" in rule and rule["delta_vs_baseline"] is not None and not _is_number(rule["delta_vs_baseline"]):
                errors.append(f"bad_type:naive_rules.{name}.delta_vs_baseline")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_validate_canonical(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "status": str,
        "run_id": str,
        "source": str,
        "violations": list,
        "column_stats": dict,
        "invariant_summary": dict,
        "notes": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    if "status" in payload and payload["status"] not in {"pass", "fail", "skip"}:
        errors.append("bad_value:status")
    violations = payload.get("violations")
    if isinstance(violations, list):
        for idx, item in enumerate(violations):
            if not isinstance(item, dict):
                errors.append(f"bad_type:violations[{idx}]")
                continue
            for key in ("type", "code", "severity"):
                if key not in item:
                    errors.append(f"missing:violations[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_validate_passive_pocket_forward(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "horizon_sec": int,
        "rows_total": int,
        "pass_count": int,
        "pass_rate": (int, float),
        "per_combo": list,
        "per_split": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    per_combo = payload.get("per_combo")
    if isinstance(per_combo, list):
        for idx, item in enumerate(per_combo):
            if not isinstance(item, dict):
                errors.append(f"bad_type:per_combo[{idx}]")
                continue
            for key in ("seed", "split", "filled_n", "attempt_fill_rate", "fail_reason", "pass"):
                if key not in item:
                    errors.append(f"missing:per_combo[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_summarize_rank_attribution(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source": str,
        "rows_total": int,
        "top_n": int,
        "reason_share": dict,
        "gate_high_share": (int, float),
        "next_action": str,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_summarize_liq_regime_tag_impact(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source": str,
        "discovery": dict,
        "validation": dict,
        "recommendation": str,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    for section in ("discovery", "validation"):
        block = payload.get(section)
        if not isinstance(block, dict):
            continue
        for key in ("available", "tagged", "normal", "delta_avg_net", "delta_p90_net", "sample_warning"):
            if key not in block:
                errors.append(f"missing:{section}.{key}")
        if "available" in block and not isinstance(block["available"], bool):
            errors.append(f"bad_type:{section}.available")
        if "sample_warning" in block and not isinstance(block["sample_warning"], bool):
            errors.append(f"bad_type:{section}.sample_warning")
        for sub in ("tagged", "normal"):
            sub_block = block.get(sub)
            if not isinstance(sub_block, dict):
                errors.append(f"bad_type:{section}.{sub}")
                continue
            for key in ("n", "avg_net", "p90_net"):
                if key not in sub_block:
                    errors.append(f"missing:{section}.{sub}.{key}")
            if "n" in sub_block and not isinstance(sub_block["n"], int):
                errors.append(f"bad_type:{section}.{sub}.n")
            for key in ("avg_net", "p90_net"):
                if key in sub_block and not _is_number(sub_block[key]):
                    errors.append(f"bad_type:{section}.{sub}.{key}")
        for key in ("delta_avg_net", "delta_p90_net"):
            if key in block and not _is_number(block[key]):
                errors.append(f"bad_type:{section}.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_summarize_liq_tag_signal_behavior(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "debug": str,
        "rule": str,
        "overall": dict,
        "recommendation": str,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    overall = payload.get("overall")
    if isinstance(overall, dict):
        for key in ("rows_total", "tagged", "normal", "delta_avg_net", "delta_p90_net"):
            if key not in overall:
                errors.append(f"missing:overall.{key}")
        if "rows_total" in overall and not isinstance(overall["rows_total"], int):
            errors.append("bad_type:overall.rows_total")
        for group in ("tagged", "normal"):
            block = overall.get(group)
            if not isinstance(block, dict):
                errors.append(f"bad_type:overall.{group}")
                continue
            for key in ("n", "avg_net", "p90_net", "break_even_bps_total"):
                if key not in block:
                    errors.append(f"missing:overall.{group}.{key}")
            if "n" in block and not isinstance(block["n"], int):
                errors.append(f"bad_type:overall.{group}.n")
        for key in ("delta_avg_net", "delta_p90_net"):
            if key in overall and not _is_number(overall[key]):
                errors.append(f"bad_type:overall.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_liquidation_regime_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "rule": str,
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "min_liq_rate": (int, float),
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("rows_total", "tagged_count", "tagged_rate", "recent_alert_count", "max_consecutive_tagged", "max_liq_rate_recent", "side_bias_counts", "severity_counts"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, item in enumerate(alerts):
            if not isinstance(item, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "side_bias", "severity", "liq_rate_per_sec", "liq_imbalance", "spread", "trade_intensity", "ret_1"):
                if key not in item:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_liquidation_alert_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source_json": str,
        "symbol": str,
        "rule": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "primary_side_bias", "dominant_severity", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
        if "reasons" in state and not isinstance(state["reasons"], list):
            errors.append("bad_type:state.reasons")
        freshness = state.get("freshness")
        if not isinstance(freshness, dict):
            errors.append("bad_type:state.freshness")
        else:
            for key in ("status", "age_sec", "stale_after_sec"):
                if key not in freshness:
                    errors.append(f"missing:state.freshness.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "recent_alert_count",
            "tagged_rate",
            "max_consecutive_tagged",
            "max_liq_rate_recent",
            "primary_side_bias",
            "dominant_severity",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "max_consecutive_tagged",
            "max_liq_rate_recent",
            "side_bias_counts",
            "severity_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_liquidation_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "rule": str,
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "min_liq_rate": (int, float),
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "primary_side_bias",
                "dominant_severity",
                "recent_alert_count",
                "max_liq_rate_recent",
                "tagged_rate",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_spread_stress_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_spread_tagged",
            "avg_trade_intensity_tagged",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, row in enumerate(alerts):
            if not isinstance(row, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "severity", "spread", "trade_intensity", "ret_1"):
                if key not in row:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_spread_stress_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source_json": str,
        "symbol": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "recent_alert_count",
            "tagged_rate",
            "high_count",
            "medium_count",
            "avg_spread_tagged",
            "avg_trade_intensity_tagged",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_return_shock_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "direction_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, row in enumerate(alerts):
            if not isinstance(row, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "severity", "direction", "ret_1", "abs_ret_1", "spread", "trade_intensity"):
                if key not in row:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_return_shock_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source_json": str,
        "symbol": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "dominant_direction", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "recent_alert_count",
            "tagged_rate",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "dominant_direction",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "direction_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_return_shock_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "dominant_direction",
                "recent_alert_count",
                "high_count",
                "medium_count",
                "avg_abs_ret_1_tagged",
                "avg_trade_intensity_tagged",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volume_vacuum_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "symbol": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, row in enumerate(alerts):
            if not isinstance(row, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "severity", "trade_intensity", "spread", "ret_1"):
                if key not in row:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volume_vacuum_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "source_json": str,
        "symbol": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "recent_alert_count",
            "tagged_rate",
            "high_count",
            "medium_count",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volume_vacuum_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "recent_alert_count",
                "high_count",
                "medium_count",
                "avg_trade_intensity_tagged",
                "avg_spread_tagged",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volatility_burst_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "symbol": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "direction_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, row in enumerate(alerts):
            if not isinstance(row, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "severity", "direction", "ret_1", "abs_ret_1", "trade_intensity", "spread"):
                if key not in row:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volatility_burst_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "source_json": str,
        "symbol": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "dominant_direction", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "recent_alert_count",
            "tagged_rate",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "dominant_direction",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_ret_1_tagged",
            "avg_trade_intensity_tagged",
            "direction_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_volatility_burst_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "dominant_direction",
                "recent_alert_count",
                "high_count",
                "medium_count",
                "avg_abs_ret_1_tagged",
                "avg_trade_intensity_tagged",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_book_proxy_pressure_alerts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "symbol": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "alerts": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_imbalance_tagged",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
            "side_bias_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        for idx, row in enumerate(alerts):
            if not isinstance(row, dict):
                errors.append(f"bad_type:alerts[{idx}]")
                continue
            for key in ("ts_ms", "severity", "side_bias", "imbalance", "abs_imbalance", "trade_intensity", "spread", "ret_1"):
                if key not in row:
                    errors.append(f"missing:alerts[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_book_proxy_pressure_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "source_json": str,
        "symbol": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons", "primary_side_bias", "freshness"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "recent_alert_count",
            "tagged_rate",
            "high_count",
            "medium_count",
            "avg_abs_imbalance_tagged",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
            "primary_side_bias",
            "latest_alert_ts_ms",
            "freshness_status",
            "age_sec",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows_total",
            "tagged_count",
            "tagged_rate",
            "recent_alert_count",
            "high_count",
            "medium_count",
            "avg_abs_imbalance_tagged",
            "avg_trade_intensity_tagged",
            "avg_spread_tagged",
            "side_bias_counts",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_book_proxy_pressure_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lane": str,
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "primary_side_bias",
                "recent_alert_count",
                "high_count",
                "medium_count",
                "avg_abs_imbalance_tagged",
                "avg_trade_intensity_tagged",
                "avg_spread_tagged",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_spread_stress_watchlist(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "lookback_min": int,
        "bucket_sec": int,
        "recent_limit": int,
        "summary": dict,
        "top_summary": dict,
        "banner": dict,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("symbol_count", "top_n", "state_counts", "top_symbol"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_summary = payload.get("top_summary")
    if isinstance(top_summary, dict):
        for key in ("symbol", "state_level", "freshness_status", "recommended_action", "dashboard_summary"):
            if key not in top_summary:
                errors.append(f"missing:top_summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in (
            "headline",
            "recommended_action",
            "top_symbol",
            "top_state_level",
            "top_freshness_status",
            "severe_count",
            "elevated_count",
            "quiet_count",
        ):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rows[{idx}]")
                continue
            for key in (
                "symbol",
                "state_level",
                "freshness_status",
                "recommended_action",
                "recent_alert_count",
                "high_count",
                "medium_count",
                "avg_spread_tagged",
                "avg_trade_intensity_tagged",
                "age_sec",
                "dashboard_summary",
                "priority_score",
            ):
                if key not in row:
                    errors.append(f"missing:rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_fill_toxicity_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source": str,
        "rows": int,
        "top_side": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in ("headline", "operator_note", "top_side", "rows", "toxicity_score", "adverse_bps_mean", "pnl_bps_mean"):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in ("rows", "sides"):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_latency_stress_state(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source": str,
        "state": dict,
        "dashboard_summary": str,
        "notification_text": str,
        "recommended_action": str,
        "card": dict,
        "summary_snapshot": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    state = payload.get("state")
    if isinstance(state, dict):
        for key in ("level", "reasons"):
            if key not in state:
                errors.append(f"missing:state.{key}")
    card = payload.get("card")
    if isinstance(card, dict):
        for key in (
            "headline",
            "operator_note",
            "rows",
            "fill_rate",
            "latency_fill_delay_sec_p50",
            "latency_fill_delay_sec_p95",
            "latency_impact_vs_net_corr",
        ):
            if key not in card:
                errors.append(f"missing:card.{key}")
    summary = payload.get("summary_snapshot")
    if isinstance(summary, dict):
        for key in (
            "rows",
            "fill_rate",
            "queue_competition_score",
            "toxicity_score",
            "adverse_selection_bps_mean",
            "latency_fill_delay_sec_p50",
            "latency_fill_delay_sec_p95",
            "latency_impact_vs_net_corr",
        ):
            if key not in summary:
                errors.append(f"missing:summary_snapshot.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_research_event_watchboard(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "summary": dict,
        "top_event": dict,
        "banner": dict,
        "lanes": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("lane_count", "state_counts", "top_lane"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    top_event = payload.get("top_event")
    if isinstance(top_event, dict):
        for key in ("lane", "level", "recommended_action", "headline", "detail"):
            if key not in top_event:
                errors.append(f"missing:top_event.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key in ("headline", "recommended_action", "top_lane", "top_level"):
            if key not in banner:
                errors.append(f"missing:banner.{key}")
    lanes = payload.get("lanes")
    if isinstance(lanes, list):
        for idx, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                errors.append(f"bad_type:lanes[{idx}]")
                continue
            for key in ("lane", "level", "freshness_status", "recommended_action", "headline", "detail", "priority_score"):
                if key not in lane:
                    errors.append(f"missing:lanes[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_watchboard_trend(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "summary": dict,
        "latest": dict,
        "points": list,
        "lane_deltas": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("snapshot_count", "start_top_lane", "end_top_lane", "delta_priority_score", "trend"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    latest = payload.get("latest")
    if isinstance(latest, dict):
        for key in ("index", "top_lane", "top_level", "top_recommended_action", "priority_score"):
            if key not in latest:
                errors.append(f"missing:latest.{key}")
    points = payload.get("points")
    if isinstance(points, list):
        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                errors.append(f"bad_type:points[{idx}]")
                continue
            for key in ("index", "source", "top_lane", "top_level", "top_recommended_action", "priority_score"):
                if key not in point:
                    errors.append(f"missing:points[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_watchboard_snapshot_append(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "history_path": str,
        "appended": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    appended = payload.get("appended")
    if isinstance(appended, dict):
        for key in ("source", "top_lane", "state_counts", "lanes", "top_event", "banner", "upstream_run_type"):
            if key not in appended:
                errors.append(f"missing:appended.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_watchboard_trend_from_history(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "summary": dict,
        "latest": dict,
        "points": list,
        "history": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    history = payload.get("history")
    if isinstance(history, dict):
        for key in ("history_path", "last_n", "available_rows", "used_rows"):
            if key not in history:
                errors.append(f"missing:history.{key}")
    if "lane_deltas" not in payload:
        errors.append("missing:lane_deltas")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_run_research_event_watchboard_cycle(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "watchboard_json": str,
        "append_json": str,
        "overlap_json": str,
        "consolidation_json": str,
        "suppression_json": str,
        "persistence_json": str,
        "merged_banner_json": str,
        "trend_json": str,
        "brief_json": str,
        "history_jsonl": str,
        "summary": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "top_lane",
            "top_action",
            "history_rows",
            "trend",
            "trimmed_rows",
            "top_overlap_pair",
            "suppression_candidate_count",
            "suppression_rule_count",
            "noisy_lane_count",
            "merged_banner_mode",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
        if "history_rows" in summary and not isinstance(summary["history_rows"], int):
            errors.append("bad_type:summary.history_rows")
        if "trimmed_rows" in summary and not isinstance(summary["trimmed_rows"], int):
            errors.append("bad_type:summary.trimmed_rows")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_research_event_operator_brief(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "watchboard_json": str,
        "trend_json": str,
        "overlap_json": str,
        "consolidation_json": str,
        "persistence_json": str,
        "merged_banner_json": str,
        "summary": dict,
        "brief": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "top_lane",
            "top_action",
            "trend",
            "severe_lane_count",
            "stale_lane_count",
            "strongest_delta_lane",
            "strongest_delta_trend",
            "strongest_overlap_pair",
            "suppression_candidate_count",
            "primary_suppression_lane",
            "noisy_lane_count",
            "primary_noisy_lane",
            "merged_banner_mode",
            "merged_focus_lane_count",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    brief = payload.get("brief")
    if isinstance(brief, dict):
        for key in (
            "headline",
            "operator_note",
            "top_event",
            "strongest_delta",
            "strongest_overlap",
            "primary_suppression",
            "primary_persistence",
            "merged_banner",
            "severe_lanes",
            "stale_lanes",
        ):
            if key not in brief:
                errors.append(f"missing:brief.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_validate_artifacts(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ok": bool,
        "calibration": dict,
        "execution": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_report_check(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "results": list,
        "summary": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("checked", "ok_count", "fail_count"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
            elif not isinstance(summary[key], int):
                errors.append(f"bad_type:summary.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_validate_micro_edge_forward(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "debug": str,
        "group_by": list,
        "discover_frac": (int, float),
        "counts": dict,
        "thresholds": dict,
        "discovery": dict,
        "validation": dict,
        "collapse": dict,
        "liquidation_impact": dict,
        "liquidation_regime_tag_impact": dict,
        "event_lane_context_impact": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    counts = payload.get("counts")
    if isinstance(counts, dict):
        for key in ("total", "discovery", "validation", "selected_discovery", "selected_validation", "top_groups"):
            if key not in counts:
                errors.append(f"missing:counts.{key}")
            elif not isinstance(counts[key], int):
                errors.append(f"bad_type:counts.{key}")
    collapse = payload.get("collapse")
    if isinstance(collapse, dict):
        if "detected" not in collapse:
            errors.append("missing:collapse.detected")
        elif not isinstance(collapse["detected"], bool):
            errors.append("bad_type:collapse.detected")
        if "flags" not in collapse:
            errors.append("missing:collapse.flags")
        elif not isinstance(collapse["flags"], dict):
            errors.append("bad_type:collapse.flags")
    event_ctx = payload.get("event_lane_context_impact")
    if isinstance(event_ctx, dict):
        for section_name in ("discovery", "validation"):
            section = event_ctx.get(section_name)
            if not isinstance(section, dict):
                errors.append(f"bad_type:event_lane_context_impact.{section_name}")
                continue
            for key in ("available", "rows_total", "lane_count", "by_lane"):
                if key not in section:
                    errors.append(f"missing:event_lane_context_impact.{section_name}.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_liquidation_rule_coverage(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "rule": str,
        "bucket_sec": int,
        "results": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    if isinstance(payload.get("results"), list):
        for idx, row in enumerate(payload["results"]):
            if not isinstance(row, dict):
                errors.append(f"bad_type:results[{idx}]")
                continue
            for key in ("lookback_min", "bucket_rows", "liq_rows", "rule_fire_count"):
                if key not in row:
                    errors.append(f"missing:results[{idx}].{key}")
                elif not isinstance(row[key], int):
                    errors.append(f"bad_type:results[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_analyze_cost_breakdown(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "tool": str,
        "source_json": str,
        "n_pockets": int,
        "pockets": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_analyze_fill_timing(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "status": str,
        "live_parquet": str,
        "trade_db": str,
        "bar_sec": (int, float),
        "timeout_candidates": list,
        "live_summary": dict,
        "trade_db_summary": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_daily_execution_calibration(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ts_utc": str,
        "symbol": str,
        "interval_ms": int,
        "days": int,
        "ok": bool,
        "steps": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_execution_diagnostics(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "rows": int,
        "fill_rate": (int, float),
        "queue_competition_score": (int, float),
        "toxicity_score": (int, float),
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_preflight_check(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ok": bool,
        "failures": list,
        "warnings": list,
        "checks": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_paper_trade_summary(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "total_trades": int,
        "win_rate": (int, float),
        "mean_pnl_bps": (int, float),
        "total_pnl_bps": (int, float),
        "scratch_rate": (int, float),
        "max_drawdown_bps": (int, float),
        "exit_types": dict,
        "daily": list,
        "rolling7": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_post_rollout_audit(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ts_utc": str,
        "flags": dict,
        "checks": dict,
        "overall_ok": bool,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_toxicity_report(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "rows": int,
        "sides": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_replay_parity_report(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "sim_count": int,
        "live_count": int,
        "matched_count": int,
        "match_rate_vs_sim": (int, float),
        "matches": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_live_fill_drift_root_cause(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "ts_utc": str,
        "overall_status": str,
        "parity_json": str,
        "diagnostics_json": str,
        "toxicity_json": str,
        "audit_json": str,
        "causes": list,
        "pipeline": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_execution_e2e_pipeline(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"ok": bool, "steps": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_execution_quality_audit(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"status": str, "input": str, "timestamp_utc": str, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_optimize_fill_timeout(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "analysis_json": str,
        "env_file": str,
        "recommended": int,
        "raw_recommended": (int, float),
        "source": str,
        "reason": str,
        "applied": bool,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_fit_adverse_model(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"tool": str, "generated_utc": str, "git_hash": str, "inputs": dict, "per_symbol": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_triage_capacity(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"inputs": dict, "gate_config": dict, "rows": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_rank_passive_pockets_forward(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"count": int, "mitigation_profile": str, "gate_config": dict, "statistical": dict, "decomposition": list, "ranking": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_calibrate_capacity_thresholds(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"inputs": dict, "rows": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_evaluate_canary_expansion_gate(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"ts_utc": str, "passed": bool, "gate": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_run_full_sweep(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"generated_ts": int, "jobs": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_run_scratch_calibration(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"symbol": str, "sell": dict, "buy": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_backtest_scratch(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"symbol": str, "side": str, "baseline": dict, "adverse_sweep": list, "trailing_sweep": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_compare_scratch_live_vs_backtest(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"status": str, "live": dict, "backtest_sell": dict, "backtest_buy": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_db_introspect(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"db": str, "tables": list, "likely_core_tables": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_funding_rate_analysis(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"summary": dict, "sample": list, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_prototype_ws_vs_db_latency(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"status": str, "symbol": str, "db_path": str, "collector_heartbeat_path": str, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_freeze_runtime_profile(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {"ts_utc": str, "hash": str, "profile": dict, "run_summary": dict}
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_microstructure_contract(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "db": str,
        "symbols": list,
        "required_tables": list,
        "status": str,
        "table_contracts": dict,
        "symbol_coverage": dict,
        "feature_capability": dict,
        "warnings": list,
        "failures": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    if "status" in payload and payload["status"] not in {"pass", "warn", "fail"}:
        errors.append("bad_value:status")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_generate_liq_reversal_candidates(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "rule": str,
        "regime": str,
        "symbols": list,
        "grid": dict,
        "count": int,
        "rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_run_liq_reversal_e2e(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "rule": str,
        "lookback_min": int,
        "bucket_sec": int,
        "coverage_json": str,
        "candidates_json": str,
        "rank_baseline_json": str,
        "rank_v5_json": str,
        "summary": dict,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_liquidation_regime_tagger(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "symbol": str,
        "rule": str,
        "lookback_min": int,
        "bucket_sec": int,
        "summary": dict,
        "tags": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_lane_overlap(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "history_jsonl": str,
        "summary": dict,
        "lane_stats": list,
        "pairwise": list,
        "strongest_overlaps": list,
        "redundancy_notes": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("available_rows", "used_rows", "lane_count", "active_lane_count", "active_snapshot_count", "min_level", "top_overlap_pair"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    lane_stats = payload.get("lane_stats")
    if isinstance(lane_stats, list):
        for idx, row in enumerate(lane_stats):
            if not isinstance(row, dict):
                errors.append(f"bad_type:lane_stats[{idx}]")
                continue
            for key in ("lane", "active_count", "active_rate", "fresh_active_count", "top_count"):
                if key not in row:
                    errors.append(f"missing:lane_stats[{idx}].{key}")
    pairwise = payload.get("pairwise")
    if isinstance(pairwise, list):
        for idx, row in enumerate(pairwise):
            if not isinstance(row, dict):
                errors.append(f"bad_type:pairwise[{idx}]")
                continue
            for key in ("lane_a", "lane_b", "coactive_count", "coactive_rate", "jaccard"):
                if key not in row:
                    errors.append(f"missing:pairwise[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_lane_consolidation(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "watchboard_json": str,
        "overlap_json": str,
        "summary": dict,
        "decisions": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("top_lane", "top_overlap_pair", "decision_count", "recommendation_counts"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        for idx, row in enumerate(decisions):
            if not isinstance(row, dict):
                errors.append(f"bad_type:decisions[{idx}]")
                continue
            for key in ("lane_a", "lane_b", "jaccard", "coactive_count", "secondary_lane", "recommendation", "reason"):
                if key not in row:
                    errors.append(f"missing:decisions[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_lane_suppression_policy(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "watchboard_json": str,
        "consolidation_json": str,
        "summary": dict,
        "rules": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("top_lane", "rule_count", "suppressed_lanes"):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    rules = payload.get("rules")
    if isinstance(rules, list):
        for idx, row in enumerate(rules):
            if not isinstance(row, dict):
                errors.append(f"bad_type:rules[{idx}]")
                continue
            for key in ("secondary_lane", "when_lane_a", "when_lane_b", "display_mode", "reason", "secondary_level", "secondary_action"):
                if key not in row:
                    errors.append(f"missing:rules[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_watchboard_effective(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "watchboard_json": str,
        "suppression_json": str,
        "persistence_json": str,
        "summary": dict,
        "effective_top_event": dict,
        "lanes": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in (
            "raw_top_lane",
            "effective_top_lane",
            "hidden_lane_count",
            "degraded_lane_count",
            "collapsed_lane_count",
            "noisy_lane_count",
            "primary_noisy_lane",
        ):
            if key not in summary:
                errors.append(f"missing:summary.{key}")
    effective_top_event = payload.get("effective_top_event")
    if isinstance(effective_top_event, dict):
        for key in (
            "lane",
            "level",
            "recommended_action",
            "effective_display_mode",
            "persistence_recommendation",
            "recommended_min_persist_snapshots",
            "recommended_cooldown_snapshots",
        ):
            if key not in effective_top_event:
                errors.append(f"missing:effective_top_event.{key}")
    lanes = payload.get("lanes")
    if isinstance(lanes, list):
        for idx, row in enumerate(lanes):
            if not isinstance(row, dict):
                errors.append(f"bad_type:lanes[{idx}]")
                continue
            for key in (
                "lane",
                "level",
                "recommended_action",
                "effective_display_mode",
                "effective_priority_score",
                "persistence_recommendation",
                "recommended_min_persist_snapshots",
                "recommended_cooldown_snapshots",
                "is_noisy",
            ):
                if key not in row:
                    errors.append(f"missing:lanes[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_lane_persistence_policy(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "history_path": str,
        "last_n": int,
        "summary": dict,
        "lanes": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, expected in {
            "history_path": str,
            "available_rows": int,
            "used_rows": int,
            "sequence_length": int,
            "latest_top_lane": str,
            "flip_count": int,
            "noisy_lane_count": int,
            "primary_noisy_lane": str,
        }.items():
            if key not in summary:
                errors.append(f"missing:summary.{key}")
            elif not isinstance(summary[key], expected):
                errors.append(f"bad_type:summary.{key}")
    lanes = payload.get("lanes")
    if isinstance(lanes, list):
        for idx, row in enumerate(lanes):
            if not isinstance(row, dict):
                errors.append(f"bad_type:lanes[{idx}]")
                continue
            for key, expected in {
                "lane": str,
                "top_hits": int,
                "hit_rate": (int, float),
                "longest_streak": int,
                "transitions_involved": int,
                "is_noisy": bool,
                "recommended_min_persist_snapshots": int,
                "recommended_cooldown_snapshots": int,
                "recommendation": str,
            }.items():
                if key not in row:
                    errors.append(f"missing:lanes[{idx}].{key}")
                elif not isinstance(row[key], expected):
                    errors.append(f"bad_type:lanes[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_event_merged_banner_policy(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "effective_json": str,
        "summary": dict,
        "banner": dict,
        "focus_rows": list,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, expected in {
            "banner_mode": str,
            "focus_lane_count": int,
            "focus_lanes": list,
            "top_lane": str,
            "top_action": str,
        }.items():
            if key not in summary:
                errors.append(f"missing:summary.{key}")
            elif not isinstance(summary[key], expected):
                errors.append(f"bad_type:summary.{key}")
    banner = payload.get("banner")
    if isinstance(banner, dict):
        for key, expected in {
            "headline": str,
            "recommended_action": str,
            "top_lane": str,
            "banner_mode": str,
            "focus_lanes": list,
            "reasons": list,
            "operator_note": str,
        }.items():
            if key not in banner:
                errors.append(f"missing:banner.{key}")
            elif not isinstance(banner[key], expected):
                errors.append(f"bad_type:banner.{key}")
    focus_rows = payload.get("focus_rows")
    if isinstance(focus_rows, list):
        for idx, row in enumerate(focus_rows):
            if not isinstance(row, dict):
                errors.append(f"bad_type:focus_rows[{idx}]")
                continue
            for key, expected in {
                "lane": str,
                "level": str,
                "freshness_status": str,
                "recommended_action": str,
                "effective_display_mode": str,
                "effective_priority_score": (int, float),
                "headline": str,
            }.items():
                if key not in row:
                    errors.append(f"missing:focus_rows[{idx}].{key}")
                elif not isinstance(row[key], expected):
                    errors.append(f"bad_type:focus_rows[{idx}].{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


def _validate_summarize_event_signal_bridge(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "source_forward_json": str,
        "discovery": dict,
        "validation": dict,
        "recommendation": str,
        "run_summary": dict,
    }
    for key, expected in required_top.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"bad_type:{key}")
    for section_name in ("discovery", "validation"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            for key, expected in {
                "available": bool,
                "rows_total": int,
                "best_positive_lane": dict,
                "worst_negative_lane": dict,
                "positive_lane_count": int,
                "negative_lane_count": int,
                "ranked": list,
            }.items():
                if key not in section:
                    errors.append(f"missing:{section_name}.{key}")
                elif not isinstance(section[key], expected):
                    errors.append(f"bad_type:{section_name}.{key}")
    errors.extend(_validate_run_summary(payload.get("run_summary")))
    return errors


SCHEMAS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "micro_edge_smoke": _validate_micro_edge_record,
    "validate_canonical": _validate_validate_canonical,
    "validate_passive_pocket_forward": _validate_validate_passive_pocket_forward,
    "summarize_rank_attribution": _validate_summarize_rank_attribution,
    "summarize_liq_regime_tag_impact": _validate_summarize_liq_regime_tag_impact,
    "summarize_liq_tag_signal_behavior": _validate_summarize_liq_tag_signal_behavior,
    "liquidation_regime_alerts": _validate_liquidation_regime_alerts,
    "liquidation_alert_state": _validate_liquidation_alert_state,
    "liquidation_watchlist": _validate_liquidation_watchlist,
    "spread_stress_alerts": _validate_spread_stress_alerts,
    "spread_stress_state": _validate_spread_stress_state,
    "return_shock_alerts": _validate_return_shock_alerts,
    "return_shock_state": _validate_return_shock_state,
    "return_shock_watchlist": _validate_return_shock_watchlist,
    "volume_vacuum_alerts": _validate_volume_vacuum_alerts,
    "volume_vacuum_state": _validate_volume_vacuum_state,
    "volume_vacuum_watchlist": _validate_volume_vacuum_watchlist,
    "volatility_burst_alerts": _validate_volatility_burst_alerts,
    "volatility_burst_state": _validate_volatility_burst_state,
    "volatility_burst_watchlist": _validate_volatility_burst_watchlist,
    "book_proxy_pressure_alerts": _validate_book_proxy_pressure_alerts,
    "book_proxy_pressure_state": _validate_book_proxy_pressure_state,
    "book_proxy_pressure_watchlist": _validate_book_proxy_pressure_watchlist,
    "spread_stress_watchlist": _validate_spread_stress_watchlist,
    "fill_toxicity_state": _validate_fill_toxicity_state,
    "latency_stress_state": _validate_latency_stress_state,
    "research_event_watchboard": _validate_research_event_watchboard,
    "event_watchboard_trend": _validate_event_watchboard_trend,
    "event_watchboard_snapshot_append": _validate_event_watchboard_snapshot_append,
    "event_watchboard_trend_from_history": _validate_event_watchboard_trend_from_history,
    "run_research_event_watchboard_cycle": _validate_run_research_event_watchboard_cycle,
    "research_event_operator_brief": _validate_research_event_operator_brief,
    "validate_artifacts": _validate_validate_artifacts,
    "report_check": _validate_report_check,
    "validate_micro_edge_forward": _validate_validate_micro_edge_forward,
    "liquidation_rule_coverage": _validate_liquidation_rule_coverage,
    "analyze_cost_breakdown": _validate_analyze_cost_breakdown,
    "analyze_fill_timing": _validate_analyze_fill_timing,
    "daily_execution_calibration": _validate_daily_execution_calibration,
    "execution_diagnostics": _validate_execution_diagnostics,
    "preflight_check": _validate_preflight_check,
    "paper_trade_summary": _validate_paper_trade_summary,
    "post_rollout_audit": _validate_post_rollout_audit,
    "toxicity_report": _validate_toxicity_report,
    "replay_parity_report": _validate_replay_parity_report,
    "live_fill_drift_root_cause": _validate_live_fill_drift_root_cause,
    "execution_e2e_pipeline": _validate_execution_e2e_pipeline,
    "execution_quality_audit": _validate_execution_quality_audit,
    "optimize_fill_timeout": _validate_optimize_fill_timeout,
    "fit_adverse_model": _validate_fit_adverse_model,
    "triage_capacity": _validate_triage_capacity,
    "rank_passive_pockets_forward": _validate_rank_passive_pockets_forward,
    "calibrate_capacity_thresholds": _validate_calibrate_capacity_thresholds,
    "evaluate_canary_expansion_gate": _validate_evaluate_canary_expansion_gate,
    "run_full_sweep": _validate_run_full_sweep,
    "run_scratch_calibration": _validate_run_scratch_calibration,
    "backtest_scratch": _validate_backtest_scratch,
    "compare_scratch_live_vs_backtest": _validate_compare_scratch_live_vs_backtest,
    "db_introspect": _validate_db_introspect,
    "funding_rate_analysis": _validate_funding_rate_analysis,
    "prototype_ws_vs_db_latency": _validate_prototype_ws_vs_db_latency,
    "freeze_runtime_profile": _validate_freeze_runtime_profile,
    "validate_microstructure_contract": _validate_microstructure_contract,
    "generate_liq_reversal_candidates": _validate_generate_liq_reversal_candidates,
    "run_liq_reversal_e2e": _validate_run_liq_reversal_e2e,
    "liquidation_regime_tagger": _validate_liquidation_regime_tagger,
    "event_lane_overlap": _validate_event_lane_overlap,
    "event_lane_consolidation": _validate_event_lane_consolidation,
    "event_lane_suppression_policy": _validate_event_lane_suppression_policy,
    "event_watchboard_effective": _validate_event_watchboard_effective,
    "event_lane_persistence_policy": _validate_event_lane_persistence_policy,
    "event_merged_banner_policy": _validate_event_merged_banner_policy,
    "summarize_event_signal_bridge": _validate_summarize_event_signal_bridge,
}


def infer_schema_name(payload: Dict[str, Any]) -> Optional[str]:
    keys = set(payload)
    lane = payload.get("lane")
    if lane == "volume_vacuum" and {"symbol", "lookback_min", "bucket_sec", "summary", "alerts"}.issubset(keys):
        return "volume_vacuum_alerts"
    if lane == "volume_vacuum" and {"source_json", "state", "card", "summary_snapshot"}.issubset(keys):
        return "volume_vacuum_state"
    if lane == "volume_vacuum" and {"summary", "top_summary", "banner", "rows"}.issubset(keys):
        return "volume_vacuum_watchlist"
    if lane == "volatility_burst" and {"symbol", "lookback_min", "bucket_sec", "summary", "alerts"}.issubset(keys):
        return "volatility_burst_alerts"
    if lane == "volatility_burst" and {"source_json", "state", "card", "summary_snapshot"}.issubset(keys):
        return "volatility_burst_state"
    if lane == "volatility_burst" and {"summary", "top_summary", "banner", "rows"}.issubset(keys):
        return "volatility_burst_watchlist"
    if lane == "book_proxy_pressure" and {"symbol", "lookback_min", "bucket_sec", "summary", "alerts"}.issubset(keys):
        return "book_proxy_pressure_alerts"
    if lane == "book_proxy_pressure" and {"source_json", "state", "card", "summary_snapshot"}.issubset(keys):
        return "book_proxy_pressure_state"
    if lane == "book_proxy_pressure" and {"summary", "top_summary", "banner", "rows"}.issubset(keys):
        return "book_proxy_pressure_watchlist"
    if {"naive_rules", "label_definition", "label_counts", "baseline_hit_rate"}.issubset(keys):
        return "micro_edge_smoke"
    if {"status", "run_id", "violations", "column_stats", "invariant_summary"}.issubset(keys):
        return "validate_canonical"
    if {"rows_total", "pass_count", "per_combo", "per_split", "failure_attribution_median"}.issubset(keys):
        return "validate_passive_pocket_forward"
    if {"reason_share", "gate_high_share", "next_action", "source"}.issubset(keys):
        return "summarize_rank_attribution"
    if {"source", "discovery", "validation", "recommendation"}.issubset(keys):
        return "summarize_liq_regime_tag_impact"
    if {"debug", "rule", "overall", "recommendation"}.issubset(keys):
        return "summarize_liq_tag_signal_behavior"
    if {"symbol", "rule", "recent_limit", "min_liq_rate", "alerts"}.issubset(keys):
        return "liquidation_regime_alerts"
    if {"rule", "summary", "rows"}.issubset(keys) and "source_json" not in keys:
        return "liquidation_watchlist"
    if {"symbol", "lookback_min", "bucket_sec", "alerts"}.issubset(keys) and "rule" not in keys and "summary" in keys:
        alerts = payload.get("alerts") or []
        if alerts and isinstance(alerts, list) and isinstance(alerts[0], dict) and "direction" in alerts[0]:
            return "return_shock_alerts"
    if {"symbol", "lookback_min", "bucket_sec", "alerts"}.issubset(keys) and "rule" not in keys:
        return "spread_stress_alerts"
    if {"lookback_min", "bucket_sec", "top_summary", "banner", "rows"}.issubset(keys) and "rule" not in keys and "source_json" not in keys:
        rows = payload.get("rows") or []
        if rows and isinstance(rows, list) and isinstance(rows[0], dict) and "dominant_direction" in rows[0]:
            return "return_shock_watchlist"
        return "spread_stress_watchlist"
    if {"source", "top_side", "notification_text", "summary_snapshot"}.issubset(keys):
        return "fill_toxicity_state"
    if {"source", "notification_text", "summary_snapshot"}.issubset(keys) and "top_side" not in keys:
        return "latency_stress_state"
    if {"summary", "top_event", "banner", "lanes"}.issubset(keys) and "top_summary" not in keys:
        return "research_event_watchboard"
    if {"history_path", "appended"}.issubset(keys):
        return "event_watchboard_snapshot_append"
    if {"history_jsonl", "summary", "lane_stats", "pairwise", "strongest_overlaps"}.issubset(keys):
        return "event_lane_overlap"
    if {"watchboard_json", "overlap_json", "summary", "decisions"}.issubset(keys):
        return "event_lane_consolidation"
    if {"watchboard_json", "consolidation_json", "summary", "rules"}.issubset(keys):
        return "event_lane_suppression_policy"
    if {"watchboard_json", "suppression_json", "summary", "effective_top_event", "lanes"}.issubset(keys):
        return "event_watchboard_effective"
    if {"history_path", "last_n", "summary", "lanes"}.issubset(keys) and "effective_top_event" not in keys:
        summary = payload.get("summary") or {}
        if "flip_count" in summary and "noisy_lane_count" in summary:
            return "event_lane_persistence_policy"
    if {"effective_json", "summary", "banner", "focus_rows"}.issubset(keys):
        return "event_merged_banner_policy"
    if {"watchboard_json", "append_json", "trend_json", "brief_json", "history_jsonl", "summary"}.issubset(keys):
        return "run_research_event_watchboard_cycle"
    if {"watchboard_json", "trend_json", "summary", "brief"}.issubset(keys):
        return "research_event_operator_brief"
    if {"summary", "latest", "points", "history"}.issubset(keys):
        return "event_watchboard_trend_from_history"
    if {"summary", "latest", "points"}.issubset(keys) and "top_event" not in keys:
        return "event_watchboard_trend"
    if {"source_json", "state", "card", "summary_snapshot"}.issubset(keys) and "rule" not in keys:
        summary_snapshot = payload.get("summary_snapshot") or {}
        if "avg_abs_ret_1_tagged" in summary_snapshot:
            return "return_shock_state"
        return "spread_stress_state"
    if {"source_json", "state", "card", "summary_snapshot"}.issubset(keys):
        return "liquidation_alert_state"
    if {"ok", "calibration", "execution"}.issubset(keys):
        return "validate_artifacts"
    if {"results", "summary"}.issubset(keys):
        return "report_check"
    if {"debug", "group_by", "counts", "collapse", "liquidation_impact"}.issubset(keys):
        return "validate_micro_edge_forward"
    if {"symbol", "rule", "bucket_sec", "results"}.issubset(keys):
        return "liquidation_rule_coverage"
    if {"tool", "source_json", "n_pockets", "pockets"}.issubset(keys):
        return "analyze_cost_breakdown"
    if {"live_parquet", "trade_db", "timeout_candidates", "live_summary"}.issubset(keys):
        return "analyze_fill_timing"
    if {"ts_utc", "symbol", "interval_ms", "days", "steps"}.issubset(keys):
        return "daily_execution_calibration"
    if {"rows", "fill_rate", "queue_competition_score", "toxicity_score"}.issubset(keys):
        return "execution_diagnostics"
    if {"failures", "warnings", "checks"}.issubset(keys):
        return "preflight_check"
    if {"total_trades", "win_rate", "exit_types", "rolling7"}.issubset(keys):
        return "paper_trade_summary"
    if {"flags", "checks", "overall_ok"}.issubset(keys):
        return "post_rollout_audit"
    if {"rows", "sides"}.issubset(keys):
        return "toxicity_report"
    if {"sim_count", "live_count", "matched_count", "matches"}.issubset(keys):
        return "replay_parity_report"
    if {"overall_status", "parity_json", "diagnostics_json", "causes", "pipeline"}.issubset(keys):
        return "live_fill_drift_root_cause"
    if {"ok", "steps"}.issubset(keys):
        return "execution_e2e_pipeline"
    if {"status", "input", "timestamp_utc", "overall", "by_side"}.issubset(keys):
        return "execution_quality_audit"
    if {"analysis_json", "env_file", "recommended", "raw_recommended", "applied"}.issubset(keys):
        return "optimize_fill_timeout"
    if {"tool", "generated_utc", "git_hash", "inputs", "per_symbol"}.issubset(keys):
        return "fit_adverse_model"
    if {"inputs", "gate_config", "rows"}.issubset(keys):
        return "triage_capacity"
    if {"count", "mitigation_profile", "gate_config", "statistical", "decomposition", "ranking"}.issubset(keys):
        return "rank_passive_pockets_forward"
    if {"inputs", "rows"}.issubset(keys) and "gate_config" not in keys:
        return "calibrate_capacity_thresholds"
    if {"passed", "gate"}.issubset(keys):
        return "evaluate_canary_expansion_gate"
    if {"generated_ts", "jobs"}.issubset(keys):
        return "run_full_sweep"
    if {"symbol", "sell", "buy"}.issubset(keys):
        return "run_scratch_calibration"
    if {"symbol", "side", "baseline", "adverse_sweep", "trailing_sweep"}.issubset(keys):
        return "backtest_scratch"
    if {"live", "backtest_sell", "backtest_buy"}.issubset(keys):
        return "compare_scratch_live_vs_backtest"
    if {"db", "tables", "likely_core_tables"}.issubset(keys):
        return "db_introspect"
    if {"summary", "sample"}.issubset(keys):
        return "funding_rate_analysis"
    if {"db_path", "collector_heartbeat_path", "estimated_ws_bypass_gain_sec"}.issubset(keys):
        return "prototype_ws_vs_db_latency"
    if {"hash", "profile"}.issubset(keys):
        return "freeze_runtime_profile"
    if {"table_contracts", "symbol_coverage", "feature_capability", "required_tables"}.issubset(keys):
        return "validate_microstructure_contract"
    if {"rule", "regime", "symbols", "grid", "rows"}.issubset(keys):
        return "generate_liq_reversal_candidates"
    if {"coverage_json", "candidates_json", "rank_baseline_json", "rank_v5_json", "summary"}.issubset(keys):
        return "run_liq_reversal_e2e"
    if {"symbol", "rule", "lookback_min", "bucket_sec", "summary", "tags"}.issubset(keys):
        return "liquidation_regime_tagger"
    if {"source_forward_json", "discovery", "validation", "recommendation"}.issubset(keys):
        return "summarize_event_signal_bridge"
    return None


def validate_payload(payload: Any, schema_name: str) -> List[str]:
    if schema_name not in SCHEMAS:
        return [f"unknown_schema:{schema_name}"]
    if isinstance(payload, list):
        errors: List[str] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append(f"bad_type:payload[{idx}]")
                continue
            errors.extend(f"item[{idx}]:{err}" for err in SCHEMAS[schema_name](item))
        return errors
    if not isinstance(payload, dict):
        return ["bad_type:payload"]
    return SCHEMAS[schema_name](payload)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate report JSON/JSONL schema.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--schema", default="auto")
    return p.parse_args()


def main() -> int:
    args = _args()
    path = Path(str(args.in_path))
    payload = _load_payload(path)
    schema_name = str(args.schema)
    if schema_name == "auto":
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            schema_name = infer_schema_name(payload[0]) or ""
        elif isinstance(payload, dict):
            schema_name = infer_schema_name(payload) or ""
        else:
            schema_name = ""
        if not schema_name:
            print("report_schema_validator status=fail reason=unknown_schema")
            return 2
    errors = validate_payload(payload, schema_name=schema_name)
    if errors:
        print(f"report_schema_validator status=fail schema={schema_name} errors={len(errors)}")
        for err in errors:
            print(f"- {err}")
        return 2
    print(f"report_schema_validator status=pass schema={schema_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
