"""S34 v8 risk refinement.

Observation/risk only. This answers the next management questions:

1. Reconcile tail-budget vs stop-budget with stop-reliability-weighted sizing.
2. Scan whether catastrophic adverse moves started inside the fill->stop
   atomicity window.
3. Simulate simple kill criteria on the filled lifecycle sequence.
4. Summarize ledger completeness for forward validation.

No live executor, .env, size, leverage, or order logic changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms

DEFAULT_DB = ROOT / "data" / "microstructure.db"
STOP_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_PROTECTIVE_STOP.json"
MGMT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
MGMT_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_LEDGER.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V8_STOP_RELIABILITY_SIZING.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V8_STOP_RELIABILITY_SIZING.md"

TAIL_STRESS_BPS = 634.0
STOP_REALIZED_BPS = 175.7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def r3(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "mean_bps": None, "median_bps": None, "max_loss_bps": None, "win_rate": None}
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "max_loss_bps": r1(min(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
    }


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return min(1.0, (center + margin) / denom)


def empirical_rate(k: int, n: int) -> float:
    return float(k) / float(n) if n > 0 else 1.0


def combined_fail_probability(p_atomic: float, p_gap: float) -> float:
    return 1.0 - (1.0 - float(p_atomic)) * (1.0 - float(p_gap))


def weighted_bps(p_fail: float, *, stop_bps: float, tail_bps: float) -> float:
    return (1.0 - float(p_fail)) * float(stop_bps) + float(p_fail) * float(tail_bps)


def notional_for_budget(equity: float, risk_pct: float, loss_bps: float) -> float:
    return (float(equity) * float(risk_pct) / 100.0) / (float(loss_bps) / 10_000.0)


def stop_reliability_sizing(mgmt: dict[str, Any], *, equity: float, leverage: float, risk_pct: float) -> dict[str, Any]:
    sizing = mgmt.get("tail_aware_sizing_monitor") or {}
    atomicity = mgmt.get("atomicity_gap_monitor") or {}
    observed_n = int(atomicity.get("observed_n") or 0)
    alert_n = int(atomicity.get("alert_n") or 0)
    # For stop gap-through beyond the observed worst (-175.7), there are zero
    # exceedances by definition in the 23-row stop sweep. Use rule-of-three via
    # Wilson upper as the conservative uncertainty penalty.
    stop_n = 23
    gap_exceed_n = 0
    p_atomic_emp = empirical_rate(alert_n, observed_n)
    p_atomic_upper = wilson_upper(alert_n, observed_n)
    p_gap_emp = empirical_rate(gap_exceed_n, stop_n)
    p_gap_upper = wilson_upper(gap_exceed_n, stop_n)
    p_fail_emp = combined_fail_probability(p_atomic_emp, p_gap_emp)
    p_fail_conservative = combined_fail_probability(p_atomic_upper, p_gap_upper)
    emp_bps = weighted_bps(p_fail_emp, stop_bps=STOP_REALIZED_BPS, tail_bps=TAIL_STRESS_BPS)
    cons_bps = weighted_bps(p_fail_conservative, stop_bps=STOP_REALIZED_BPS, tail_bps=TAIL_STRESS_BPS)
    tail_bps = TAIL_STRESS_BPS
    stop_bps = STOP_REALIZED_BPS
    planned = sizing.get("planned_live_size_from_env") or {}
    planned_notional = float(planned.get("planned_notional_usdt") or 0.0)
    rows = []
    for label, bps in (
        ("stop_only_unreliable_floor", stop_bps),
        ("empirical_weighted", emp_bps),
        ("conservative_weighted", cons_bps),
        ("tail_only_hard_floor", tail_bps),
    ):
        notional = notional_for_budget(equity, risk_pct, bps)
        margin = notional / float(leverage)
        rows.append(
            {
                "basis": label,
                "loss_bps": r1(bps),
                "max_notional_usdt": r1(notional),
                "max_margin_usdt_at_40x": r1(margin),
                "oversize_multiple_vs_env": r1(planned_notional / notional) if notional > 0 else None,
            }
        )
    return {
        "status": "STOP_RELIABILITY_WEIGHTED_RECOMMENDATION",
        "risk_pct_equity": float(risk_pct),
        "equity_usdt": float(equity),
        "leverage_kept": float(leverage),
        "p_atomic_empirical": r3(p_atomic_emp),
        "p_atomic_wilson95_upper": r3(p_atomic_upper),
        "p_gap_exceed_empirical": r3(p_gap_emp),
        "p_gap_exceed_wilson95_upper": r3(p_gap_upper),
        "p_stop_fail_empirical": r3(p_fail_emp),
        "p_stop_fail_conservative": r3(p_fail_conservative),
        "recommendation": "use conservative_weighted as single operational recommendation unless operator explicitly chooses tail_only",
        "sizing_rows": rows,
    }


def parse_iso_ms(text: str) -> int:
    t = str(text).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return int(datetime.fromisoformat(t).timestamp() * 1000)


def mark_min(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    row = conn.execute(
        "SELECT MIN(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def fixed_stop_rows(stop: dict[str, Any], variant: str = "fixed_sl_150") -> list[dict[str, Any]]:
    rows = [r for r in stop.get("rows") or [] if r.get("variant") == variant]
    rows.sort(key=lambda r: parse_iso_ms(str(r["signal_utc"])))
    return rows


def catastrophic_atomicity_scan(db: Path, stop: dict[str, Any], *, gap_sec: float) -> dict[str, Any]:
    rows = fixed_stop_rows(stop)
    cards = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for row in rows:
            signal_ms = parse_iso_ms(str(row["signal_utc"]))
            fill_delay = float(row.get("fill_delay_sec") or 0.0)
            fill_ms = int(signal_ms + fill_delay * 1000)
            end_ms = fill_ms + int(float(gap_sec) * 1000)
            entry = float(row["entry_price"])
            min_px = mark_min(conn, "ETHUSDT", fill_ms, end_ms)
            adverse = None if min_px is None else (float(min_px) - entry) / entry * 10_000.0
            exit_ts = row.get("exit_ts_ms")
            sl_in_gap = bool(row.get("exit_reason") == "SL" and exit_ts is not None and int(exit_ts) <= end_ms)
            baseline_tail = float(row.get("baseline_net_bps") or 0.0) <= -100.0
            cards.append(
                {
                    "event_id": row.get("event_id"),
                    "signal_utc": row.get("signal_utc"),
                    "fill_utc": iso_ms(fill_ms),
                    "gap_end_utc": iso_ms(end_ms),
                    "baseline_net_bps": row.get("baseline_net_bps"),
                    "stop_net_bps": row.get("net_bps"),
                    "exit_reason": row.get("exit_reason"),
                    "adverse_in_gap_bps": r1(adverse),
                    "sl_trigger_inside_gap": sl_in_gap,
                    "baseline_tail": baseline_tail,
                }
            )
    vals = [float(c["adverse_in_gap_bps"]) for c in cards if c.get("adverse_in_gap_bps") is not None]
    catastrophic = [
        c for c in cards
        if c.get("sl_trigger_inside_gap")
        or float(c.get("adverse_in_gap_bps") or 0.0) <= -150.0
        or (c.get("baseline_tail") and float(c.get("adverse_in_gap_bps") or 0.0) <= -25.0)
    ]
    return {
        "status": "NO_CATASTROPHIC_GAP_FOUND" if not catastrophic else "CATASTROPHIC_GAP_FOUND",
        "gap_sec": float(gap_sec),
        "filled_n": len(cards),
        "worst_adverse_gap_bps": r1(min(vals)) if vals else None,
        "gap_adverse_lte_5bps_n": sum(1 for v in vals if v <= -5.0),
        "gap_adverse_lte_25bps_n": sum(1 for v in vals if v <= -25.0),
        "sl_trigger_inside_gap_n": sum(1 for c in cards if c.get("sl_trigger_inside_gap")),
        "baseline_tail_inside_gap_start_n": sum(1 for c in catastrophic if c.get("baseline_tail")),
        "catastrophic_cards": catastrophic[:20],
        "read": "No hit does not prove zero risk; it bounds observed history only.",
    }


def kill_drawdown_sim(stop: dict[str, Any], *, notional: float, equity: float) -> dict[str, Any]:
    rows = fixed_stop_rows(stop)
    vals = [float(r.get("baseline_net_bps") or 0.0) for r in rows]
    equity_curve = []
    cum_usdt = 0.0
    max_equity = 0.0
    max_dd = 0.0
    first_trigger = None
    for i, bps in enumerate(vals, start=1):
        pnl = float(notional) * bps / 10_000.0
        cum_usdt += pnl
        max_equity = max(max_equity, cum_usdt)
        max_dd = min(max_dd, cum_usdt - max_equity)
        rolling5 = vals[max(0, i - 5):i]
        trigger = None
        if i >= 5 and sum(vals[:i]) < 0:
            trigger = "KILL_CUM_SUM_NEGATIVE_AFTER_5"
        if len(rolling5) == 5 and sum(rolling5) < 0:
            trigger = trigger or "PAUSE_ROLLING_5_NEGATIVE"
        if trigger and first_trigger is None:
            first_trigger = {"trade_index": i, "signal_utc": rows[i - 1].get("signal_utc"), "trigger": trigger, "cum_bps": r1(sum(vals[:i]))}
        equity_curve.append({"i": i, "bps": r1(bps), "cum_usdt": r1(cum_usdt), "drawdown_usdt": r1(cum_usdt - max_equity), "trigger": trigger})
    tail_rate = 101.0 / 539.0
    return {
        "sample_n": len(vals),
        "sequence_summary_bps": summary(vals),
        "notional_usdt": r1(notional),
        "max_drawdown_usdt": r1(max_dd),
        "max_drawdown_pct_equity": r1(100.0 * abs(max_dd) / equity) if equity else None,
        "first_trigger": first_trigger,
        "kill_read": "Historical 23-row filled sequence does not trigger early enough to be a primary defense; first-tail risk is before kill.",
        "tail_probability_before_5_trade_kill_window": r1(100.0 * (1.0 - (1.0 - tail_rate) ** 5)),
        "equity_curve": equity_curve,
    }


def alert_dedup_read(alert_state_path: Path) -> dict[str, Any]:
    state = load_json(alert_state_path, {})
    return {
        "state_path": str(alert_state_path),
        "state_exists": alert_state_path.exists(),
        "last_signature": state.get("last_signature"),
        "last_emit_utc": state.get("last_emit_utc"),
        "read": "alert tool should emit on state change and suppress unchanged repeats unless explicitly requested",
    }


def ledger_completeness(mgmt: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = int(mgmt.get("source_shadow_rows") or 0)
    management = int(mgmt.get("management_ledger_rows") or len(ledger_rows))
    ids = [r.get("source_observation_id") for r in ledger_rows if r.get("source_observation_id")]
    return {
        "source_shadow_rows": source,
        "management_rows": management,
        "unique_source_observation_ids": len(set(ids)),
        "duplicate_source_observation_ids": len(ids) - len(set(ids)),
        "forward_rows": int(mgmt.get("forward_rows") or 0),
        "reference_rows": int(mgmt.get("reference_rows") or 0),
        "complete_against_shadow_ledger": bool(source == management == len(set(ids))),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    stop = load_json(args.stop_json, {})
    mgmt = load_json(args.management_json, {})
    ledger = load_jsonl(args.management_ledger)
    sizing = stop_reliability_sizing(mgmt, equity=float(args.equity_usdt), leverage=float(args.leverage), risk_pct=float(args.risk_pct))
    catastrophic = catastrophic_atomicity_scan(args.db, stop, gap_sec=float(args.atomicity_gap_sec))
    planned_notional = float(((mgmt.get("tail_aware_sizing_monitor") or {}).get("planned_live_size_from_env") or {}).get("planned_notional_usdt") or 1190.0)
    kill = {
        "current_env_notional": kill_drawdown_sim(stop, notional=planned_notional, equity=float(args.equity_usdt)),
        "conservative_weighted_notional": kill_drawdown_sim(
            stop,
            notional=float(next(r["max_notional_usdt"] for r in sizing["sizing_rows"] if r["basis"] == "conservative_weighted")),
            equity=float(args.equity_usdt),
        ),
    }
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_RISK_ONLY_NO_LIVE_CHANGE",
        "stop_reliability_weighted_sizing": sizing,
        "catastrophic_atomicity_scan": catastrophic,
        "kill_drawdown_simulation": kill,
        "alert_dedup_plan": alert_dedup_read(args.alert_state),
        "ledger_completeness": ledger_completeness(mgmt, ledger),
        "final_read": "The single honest sizing recommendation is conservative_weighted; tail-only remains the hard floor if operator wants maximum survival.",
    }


def render_md(report: dict[str, Any]) -> str:
    sizing = report["stop_reliability_weighted_sizing"]
    cat = report["catastrophic_atomicity_scan"]
    kill = report["kill_drawdown_simulation"]
    complete = report["ledger_completeness"]
    lines = [
        "# S34 v8 Stop-Reliability Sizing",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## 1. Unified Sizing",
        "",
        f"- p_atomic empirical / upper95: `{sizing['p_atomic_empirical']}` / `{sizing['p_atomic_wilson95_upper']}`",
        f"- p_gap-exceed empirical / upper95: `{sizing['p_gap_exceed_empirical']}` / `{sizing['p_gap_exceed_wilson95_upper']}`",
        f"- p_stop_fail empirical / conservative: `{sizing['p_stop_fail_empirical']}` / `{sizing['p_stop_fail_conservative']}`",
        f"- recommendation: `{sizing['recommendation']}`",
        "",
        "| Basis | Loss bps | Max notional | Max margin @40x | Oversize vs env |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sizing["sizing_rows"]:
        lines.append(
            f"| `{row['basis']}` | {row['loss_bps']} | ${row['max_notional_usdt']} | ${row['max_margin_usdt_at_40x']} | {row['oversize_multiple_vs_env']}x |"
        )
    lines.extend(
        [
            "",
            "## 2. Catastrophic Atomicity Scan",
            "",
            f"- status: `{cat['status']}`",
            f"- filled N: `{cat['filled_n']}`",
            f"- worst adverse gap: `{cat['worst_adverse_gap_bps']} bps`",
            f"- adverse <= -5bps N: `{cat['gap_adverse_lte_5bps_n']}`",
            f"- adverse <= -25bps N: `{cat['gap_adverse_lte_25bps_n']}`",
            f"- SL trigger inside gap N: `{cat['sl_trigger_inside_gap_n']}`",
            f"- baseline-tail gap-start N: `{cat['baseline_tail_inside_gap_start_n']}`",
            f"- read: {cat['read']}",
            "",
            "## 3. Kill / Drawdown Simulation",
            "",
            f"- current env notional max DD: `${kill['current_env_notional']['max_drawdown_usdt']}` = `{kill['current_env_notional']['max_drawdown_pct_equity']}%` equity",
            f"- current env first trigger: `{kill['current_env_notional']['first_trigger']}`",
            f"- conservative-weighted notional max DD: `${kill['conservative_weighted_notional']['max_drawdown_usdt']}` = `{kill['conservative_weighted_notional']['max_drawdown_pct_equity']}%` equity",
            f"- tail probability before 5-trade kill window: `{kill['current_env_notional']['tail_probability_before_5_trade_kill_window']}%`",
            f"- read: {kill['current_env_notional']['kill_read']}",
            "",
            "## 4. Ledger Completeness",
            "",
            f"- source shadow rows: `{complete['source_shadow_rows']}`",
            f"- management rows: `{complete['management_rows']}`",
            f"- unique source ids: `{complete['unique_source_observation_ids']}`",
            f"- duplicates: `{complete['duplicate_source_observation_ids']}`",
            f"- complete against shadow ledger: `{complete['complete_against_shadow_ledger']}`",
            "",
            "## Final Read",
            "",
            report["final_read"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 v8 stop-reliability-weighted sizing and atomicity scan.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--stop-json", type=Path, default=STOP_JSON)
    p.add_argument("--management-json", type=Path, default=MGMT_JSON)
    p.add_argument("--management-ledger", type=Path, default=MGMT_LEDGER)
    p.add_argument("--alert-state", type=Path, default=ROOT / "runtime" / "s34_v_engine_management_alert_state.json")
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--equity-usdt", type=float, default=35.0)
    p.add_argument("--leverage", type=float, default=40.0)
    p.add_argument("--risk-pct", type=float, default=2.0)
    p.add_argument("--atomicity-gap-sec", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
