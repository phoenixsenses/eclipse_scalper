"""S34 V Engine execution-management audit.

Read-only research/risk report. It combines:
- protective-stop sweep results;
- tail-budget sizing monitor;
- live executor source/env audit for stop atomicity and kill-switch wiring;
- gap-through / stop-realization estimates.

It does not alter live state, executor config, size, or order logic.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STOP_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_PROTECTIVE_STOP.json"
MGMT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
V6_JSON = ROOT / "reports" / "research" / "s34" / "S34_V6_MANAGEMENT_SYSTEM.json"
ENV_PATH = ROOT / ".env"
EXECUTOR = ROOT / "tools" / "s34_v_engine_live_executor.py"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.md"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def fenv(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def find_stop_variant(stop: dict[str, Any], variant: str) -> dict[str, Any] | None:
    for row in stop.get("summaries") or []:
        if row.get("variant") == variant:
            return row
    return None


def bps_loss_usdt(notional: float, bps: float) -> float:
    return float(notional) * abs(float(bps)) / 10_000.0


def source_finding(source: str, pattern: str) -> bool:
    return bool(re.search(pattern, source, flags=re.MULTILINE))


def build_report() -> dict[str, Any]:
    env = parse_env(ENV_PATH)
    stop = load_json(STOP_JSON, {})
    mgmt = load_json(MGMT_JSON, {})
    v6 = load_json(V6_JSON, {})
    src = EXECUTOR.read_text(encoding="utf-8", errors="ignore") if EXECUTOR.exists() else ""

    sizing = mgmt.get("tail_aware_sizing_monitor") or {}
    live = sizing.get("planned_live_size_from_env") or {}
    planned_notional = float(live.get("planned_notional_usdt") or 0.0)
    planned_margin = float(live.get("planned_margin_usdt") or 0.0)
    equity = float(live.get("equity_usdt_assumption") or 35.0)
    max_budget_notional = float(sizing.get("max_tail_budget_notional_usdt") or 0.0)
    max_budget_margin = float(sizing.get("max_tail_budget_margin_usdt") or 0.0)
    live_stop_bps = fenv(env, "S34_V_ENGINE_LIVE_STOP_BPS", 150.0)
    leverage = fenv(env, "S34_LIVE_MAX_LEVERAGE", 40.0)
    poll_sec = fenv(env, "S34_LIVE_POLL_SEC", 2.0)

    current_stop = find_stop_variant(stop, f"fixed_sl_{live_stop_bps:g}")
    best = (stop.get("summaries") or [{}])[0] if stop.get("summaries") else {}
    baseline = stop.get("baseline") or {}
    stop_summary = current_stop.get("summary") if current_stop else {}
    realized_stop_max_loss_bps = float((stop_summary or {}).get("max_loss_bps") or -live_stop_bps)

    v6_failure = (v6.get("failure_mode_classifier") or {})
    tail_n = float(v6_failure.get("large_loss_n") or 0.0)
    total_n = float(((v6.get("dissipation_defensive_observer") or {}).get("primary_reference") or {}).get("coverage", {}).get("all") or 539.0)
    tail_rate = tail_n / total_n if total_n > 0 else None
    tail_probs = {}
    if tail_rate is not None:
        for n in (1, 3, 5, 10, 20):
            tail_probs[f"at_least_one_tail_in_{n}_trades"] = r1(100.0 * (1.0 - (1.0 - tail_rate) ** n))

    atomicity = {
        "exchange_native_stop_market": source_finding(src, r'create_order\(\s*SYMBOL,\s*"STOP_MARKET"'),
        "reduce_only_stop": source_finding(src, r'"reduceOnly":\s*True'),
        "mark_price_trigger": source_finding(src, r'"workingType":\s*"MARK_PRICE"'),
        "stop_repair_path": source_finding(src, r'repair_missing_protective_stop'),
        "orphan_emergency_stop_path": source_finding(src, r'place_orphan_emergency_stop'),
        "kill_switch_file_path": env.get("S34_LIVE_KILL_SWITCH_FILE"),
        "kill_switch_blocks_new_entries": source_finding(src, r'KILL SWITCH active.*new entries blocked'),
        "poll_sec": poll_sec,
        "finding": "NOT_ATOMIC_ENTRY_THEN_STOP_AFTER_FILL_DETECTION",
        "read": "entry limit is placed first; protective stop is submitted only after position detection in a later manage_active cycle",
    }

    planned_stop_loss = bps_loss_usdt(planned_notional, realized_stop_max_loss_bps)
    budget_stop_loss = bps_loss_usdt(max_budget_notional, realized_stop_max_loss_bps)
    planned_raw_stop_loss = bps_loss_usdt(planned_notional, live_stop_bps)
    budget_raw_stop_loss = bps_loss_usdt(max_budget_notional, live_stop_bps)
    liquidation_adverse_bps = 10_000.0 / leverage if leverage > 0 else None

    return {
        "mode": "READ_ONLY_RESEARCH_NO_LIVE_CHANGE",
        "live_env": {
            "margin_usdt": r1(planned_margin),
            "notional_usdt": r1(planned_notional),
            "max_budget_margin_usdt": r1(max_budget_margin),
            "max_budget_notional_usdt": r1(max_budget_notional),
            "leverage": r1(leverage),
            "live_stop_bps": r1(live_stop_bps),
            "poll_sec": r1(poll_sec),
            "kill_switch_file": env.get("S34_LIVE_KILL_SWITCH_FILE"),
            "trading_enabled": env.get("S34_LIVE_TRADING_ENABLED"),
            "dry_run": env.get("S34_LIVE_DRY_RUN"),
        },
        "protective_stop_research": {
            "baseline": baseline,
            "current_stop_variant": current_stop,
            "best_t3r_variant": best,
            "read": "150 bps is least destructive among tested hard stops, but it does not replace sizing",
        },
        "stop_budget_math": {
            "realized_stop_max_loss_bps_from_research": r1(realized_stop_max_loss_bps),
            "planned_notional_loss_at_research_stop_usdt": r1(planned_stop_loss),
            "tail_budget_notional_loss_at_research_stop_usdt": r1(budget_stop_loss),
            "planned_notional_loss_at_nominal_stop_usdt": r1(planned_raw_stop_loss),
            "tail_budget_notional_loss_at_nominal_stop_usdt": r1(budget_raw_stop_loss),
            "planned_stop_loss_pct_equity_research": r1(100.0 * planned_stop_loss / equity) if equity else None,
            "tail_budget_stop_loss_pct_equity_research": r1(100.0 * budget_stop_loss / equity) if equity else None,
            "liquidation_adverse_move_bps_at_leverage": r1(liquidation_adverse_bps),
            "stress_tail_bps": 634.0,
        },
        "gap_through": {
            "current_stop_nominal_bps": r1(live_stop_bps),
            "current_stop_research_max_loss_bps": r1(realized_stop_max_loss_bps),
            "gap_plus_fee_bps": r1(abs(realized_stop_max_loss_bps) - live_stop_bps),
            "read": "stop is not guaranteed at nominal bps; book/taker exit can realize worse than trigger",
        },
        "atomicity_audit": atomicity,
        "tail_frequency": {
            "large_loss_n": int(tail_n),
            "sample_n": int(total_n),
            "large_loss_rate": r1(100.0 * tail_rate) if tail_rate is not None else None,
            "probabilities": tail_probs,
        },
        "recommendations": [
            "Do not change live logic automatically.",
            "Operator should reduce size to tail-budget or disarm before relying on any stop.",
            "Keep exchange-native STOP_MARKET; process-only exits are not outage-safe.",
            "Treat the entry-fill-to-stop-placement window as real atomicity risk.",
            "Kill switch should be operator-actionable: create runtime/KILL_SWITCH to block new entries.",
        ],
    }


def fmt_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "N=0"
    return (
        f"N={summary.get('n')} sum={summary.get('sum_bps')} "
        f"med={summary.get('median_bps')} T3R={summary.get('top3_winner_removed_sum_bps')} "
        f"maxL={summary.get('max_loss_bps')}"
    )


def render_md(report: dict[str, Any]) -> str:
    env = report["live_env"]
    stop = report["protective_stop_research"]
    math_row = report["stop_budget_math"]
    gap = report["gap_through"]
    atom = report["atomicity_audit"]
    tail = report["tail_frequency"]
    current = stop.get("current_stop_variant") or {}
    best = stop.get("best_t3r_variant") or {}
    lines = [
        "# S34 V Engine Execution Management Audit",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Live Env Snapshot (Read-Only)",
        "",
        f"- env planned notional/margin: `${env['notional_usdt']}` / `${env['margin_usdt']}`",
        f"- tail-budget notional/margin: `${env['max_budget_notional_usdt']}` / `${env['max_budget_margin_usdt']}`",
        f"- leverage: `{env['leverage']}x`",
        f"- configured stop: `{env['live_stop_bps']} bps`",
        f"- poll: `{env['poll_sec']}s`",
        f"- kill switch file: `{env['kill_switch_file']}`",
        "",
        "## Stop Sweep",
        "",
        f"- baseline: {fmt_summary(stop.get('baseline') or {})}",
        f"- current configured stop: `{current.get('variant')}` -> {fmt_summary(current.get('summary') or {})}, exit rate `{current.get('exit_rate')}`",
        f"- best T3R stop: `{best.get('variant')}` -> {fmt_summary(best.get('summary') or {})}",
        f"- read: {stop['read']}",
        "",
        "## Stop Budget Math",
        "",
        f"- research realized max loss at current stop: `{math_row['realized_stop_max_loss_bps_from_research']} bps`",
        f"- current env notional loss at that stop: `${math_row['planned_notional_loss_at_research_stop_usdt']}` = `{math_row['planned_stop_loss_pct_equity_research']}%` equity",
        f"- tail-budget notional loss at that stop: `${math_row['tail_budget_notional_loss_at_research_stop_usdt']}` = `{math_row['tail_budget_stop_loss_pct_equity_research']}%` equity",
        f"- 40x liquidation adverse move approx: `{math_row['liquidation_adverse_move_bps_at_leverage']} bps`",
        f"- stress tail: `{math_row['stress_tail_bps']} bps`",
        "",
        "## Gap-Through / Realization",
        "",
        f"- nominal stop: `{gap['current_stop_nominal_bps']} bps`",
        f"- observed worst realized stop loss: `{gap['current_stop_research_max_loss_bps']} bps`",
        f"- gap+fee beyond nominal: `{gap['gap_plus_fee_bps']} bps`",
        f"- read: {gap['read']}",
        "",
        "## Atomicity / Kill Switch",
        "",
        f"- exchange-native stop-market: `{atom['exchange_native_stop_market']}`",
        f"- reduce-only: `{atom['reduce_only_stop']}`",
        f"- mark-price trigger: `{atom['mark_price_trigger']}`",
        f"- stop repair path: `{atom['stop_repair_path']}`",
        f"- orphan emergency stop path: `{atom['orphan_emergency_stop_path']}`",
        f"- kill switch blocks new entries: `{atom['kill_switch_blocks_new_entries']}`",
        f"- finding: `{atom['finding']}`",
        f"- read: {atom['read']}",
        "",
        "## Tail Frequency",
        "",
        f"- large-loss rate: `{tail['large_loss_rate']}%` ({tail['large_loss_n']}/{tail['sample_n']})",
        f"- at least one tail probabilities: `{tail['probabilities']}`",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
