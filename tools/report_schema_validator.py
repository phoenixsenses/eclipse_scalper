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


SCHEMAS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "micro_edge_smoke": _validate_micro_edge_record,
    "validate_canonical": _validate_validate_canonical,
    "validate_passive_pocket_forward": _validate_validate_passive_pocket_forward,
    "summarize_rank_attribution": _validate_summarize_rank_attribution,
    "summarize_liq_regime_tag_impact": _validate_summarize_liq_regime_tag_impact,
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
}


def infer_schema_name(payload: Dict[str, Any]) -> Optional[str]:
    keys = set(payload)
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
