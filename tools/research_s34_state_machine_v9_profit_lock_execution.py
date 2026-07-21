"""S34 state-machine v9 profit-lock execution realism.

Research-only. No live executor, env, runtime state, orders, buckets, or
dashboard changes.

Tests whether the V8 best management lead (profit-lock trigger 100 / lock 50)
survives live-like implementation frictions:
- polling cadence;
- exit delay after lock trigger;
- taker/slippage cost;
- partial lock exit;
- stop-market versus stop-limit style execution;
- side-specific robustness;
- shadow observer payload requirements.
"""

from __future__ import annotations

import bisect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    FEE_BPS,
    apply_conflict_policy,
    build_signals,
    fold_summaries,
    mark_at_or_after,
    summary_with_dd,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import build_base_rows  # noqa: E402
from tools.research_s34_state_machine_v6_development_ideas import FINAL_CFG, split  # noqa: E402


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V9_PROFIT_LOCK_EXECUTION.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V9_PROFIT_LOCK_EXECUTION.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat_line(s: dict[str, Any]) -> str:
    wr = s.get("wr")
    wrs = "NA" if wr is None else f"{float(wr) * 100:.1f}%"
    return (
        f"N={s.get('n')} WR={wrs} sum={s.get('sum')} mean={s.get('mean')} "
        f"med={s.get('median')} T3R={s.get('t3r')} maxL={s.get('max_loss')} DD={s.get('max_dd_bps')}"
    )


def hold_ms(side: str) -> int:
    return (4 if side.upper() == "LONG" else 2) * 3600_000


def pnl_bps(side: str, entry: float, px: float) -> float:
    raw = (float(px) - float(entry)) / float(entry) * 10_000.0
    return -raw if side.upper() == "SHORT" else raw


def observed_px(mk_ts: list[int], mk_px: list[float], ts_ms: int) -> tuple[int, float] | None:
    idx = bisect.bisect_left(mk_ts, int(ts_ms))
    if idx >= len(mk_ts):
        return None
    return int(mk_ts[idx]), float(mk_px[idx])


def baseline_exit(s: dict[str, Any], mk_ts: list[int], mk_px: list[float]) -> float | None:
    side = str(s["side"]).upper()
    entry_ts = int(s["entry_ts_ms"])
    entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
    exit_px = mark_at_or_after(mk_ts, mk_px, entry_ts + hold_ms(side))
    if not entry or not exit_px:
        return None
    return pnl_bps(side, float(entry), float(exit_px)) - FEE_BPS


def simulate_profit_lock(
    signals: list[dict[str, Any]],
    mk_ts: list[int],
    mk_px: list[float],
    *,
    trigger_bps: float = 100.0,
    lock_bps: float = 50.0,
    poll_sec: int = 2,
    exit_delay_sec: int = 0,
    extra_exit_cost_bps: float = 0.0,
    lock_exit_fraction: float = 1.0,
    stop_limit: bool = False,
    side_filter: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    poll_ms = max(1, int(poll_sec)) * 1000
    for s in signals:
        side = str(s["side"]).upper()
        base_net = baseline_exit(s, mk_ts, mk_px)
        if base_net is None:
            continue
        if side_filter and side != side_filter:
            rows.append({**s, "net_bps": round(base_net, 1), "pl_triggered": False, "pl_exit": False})
            continue
        entry_ts = int(s["entry_ts_ms"])
        entry_px = mark_at_or_after(mk_ts, mk_px, entry_ts)
        if not entry_px:
            continue
        deadline = entry_ts + hold_ms(side)
        armed = False
        trigger_ts = None
        trigger_pnl = None
        lock_ts = None
        lock_pnl = None
        t = entry_ts
        while t <= deadline:
            obs = observed_px(mk_ts, mk_px, t)
            if not obs:
                break
            obs_ts, px = obs
            if obs_ts > deadline:
                break
            p = pnl_bps(side, float(entry_px), px)
            if not armed and p >= trigger_bps:
                armed = True
                trigger_ts = obs_ts
                trigger_pnl = p
            elif armed and p <= lock_bps:
                lock_ts = obs_ts
                lock_pnl = p
                break
            t += poll_ms
        net = base_net
        exit_mode = "TIME"
        exit_pnl = None
        missed_limit = False
        if lock_ts is not None:
            exec_obs = observed_px(mk_ts, mk_px, lock_ts + int(exit_delay_sec) * 1000)
            if exec_obs:
                exec_ts, exec_px = exec_obs
                actual_pnl = pnl_bps(side, float(entry_px), exec_px)
                if stop_limit and actual_pnl < lock_bps:
                    missed_limit = True
                    net = base_net
                    exit_mode = "STOP_LIMIT_MISSED_THEN_TIME"
                else:
                    locked_net = actual_pnl - FEE_BPS - float(extra_exit_cost_bps)
                    net = lock_exit_fraction * locked_net + (1.0 - lock_exit_fraction) * base_net
                    exit_mode = "LOCK_EXIT"
                    exit_pnl = actual_pnl
        rows.append(
            {
                **s,
                "net_bps": round(net, 1),
                "pl_triggered": armed,
                "pl_exit": lock_ts is not None and not missed_limit,
                "pl_trigger_ts_ms": trigger_ts,
                "pl_trigger_pnl_bps": round(trigger_pnl, 1) if trigger_pnl is not None else None,
                "pl_lock_ts_ms": lock_ts,
                "pl_lock_observed_pnl_bps": round(lock_pnl, 1) if lock_pnl is not None else None,
                "pl_exit_pnl_bps": round(exit_pnl, 1) if exit_pnl is not None else None,
                "pl_poll_sec": poll_sec,
                "pl_exit_delay_sec": exit_delay_sec,
                "pl_extra_exit_cost_bps": extra_exit_cost_bps,
                "pl_lock_exit_fraction": lock_exit_fraction,
                "pl_stop_limit": stop_limit,
                "pl_exit_mode": exit_mode,
                "pl_missed_limit": missed_limit,
            }
        )
    return rows


def enrich_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = split(rows)
    triggered = [r for r in rows if r.get("pl_triggered")]
    exited = [r for r in rows if r.get("pl_exit")]
    missed = [r for r in rows if r.get("pl_missed_limit")]
    slippage = [
        float(r["pl_exit_pnl_bps"]) - float(r["pl_lock_observed_pnl_bps"])
        for r in exited
        if r.get("pl_exit_pnl_bps") is not None and r.get("pl_lock_observed_pnl_bps") is not None
    ]
    out["triggered_n"] = len(triggered)
    out["exit_n"] = len(exited)
    out["missed_limit_n"] = len(missed)
    out["exit_rate"] = round(len(exited) / len(rows), 3) if rows else 0.0
    out["avg_exit_slip_from_lock_bps"] = round(sum(slippage) / len(slippage), 2) if slippage else None
    out["folds"] = fold_summaries(rows, folds=5)
    out["hold_top3_removed"] = summary_with_dd(sorted([r for r in rows if r["row"]["is_hold"]], key=lambda x: float(x["net_bps"]), reverse=True)[3:])
    return out


def scenario_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "baseline": enrich_stats([{**s, "net_bps": round(float(s["net_bps"]), 1)} for s in signals]),
    }
    for poll in [1, 2, 5, 10, 30, 60]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=poll)
        out[f"poll_{poll}s"] = enrich_stats(rows)
    for delay in [0, 1, 2, 5, 10, 30]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, exit_delay_sec=delay)
        out[f"exit_delay_{delay}s"] = enrich_stats(rows)
    for cost in [0, 2, 5, 10, 20]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, extra_exit_cost_bps=cost)
        out[f"extra_exit_cost_{cost}bps"] = enrich_stats(rows)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, lock_exit_fraction=frac)
        out[f"lock_exit_fraction_{frac:g}"] = enrich_stats(rows)
    for side in ["LONG", "SHORT"]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, side_filter=side)
        out[f"{side.lower()}_only_lock"] = enrich_stats(rows)
    rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, stop_limit=True)
    out["stop_limit_style"] = enrich_stats(rows)
    rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=2, exit_delay_sec=2, extra_exit_cost_bps=5)
    out["live_like_conservative_delay2_cost5"] = enrich_stats(rows)
    rows = simulate_profit_lock(signals, mk_ts, mk_px, poll_sec=5, exit_delay_sec=5, extra_exit_cost_bps=10)
    out["stress_poll5_delay5_cost10"] = enrich_stats(rows)
    return out


def readiness_verdict(scenarios: dict[str, Any]) -> dict[str, Any]:
    base = scenarios["baseline"]["hold"]
    best = scenarios["poll_2s"]["hold"]
    conservative = scenarios["live_like_conservative_delay2_cost5"]["hold"]
    stop_limit = scenarios["stop_limit_style"]["hold"]
    return {
        "baseline_hold": base,
        "profit_lock_poll2_hold": best,
        "conservative_hold": conservative,
        "stop_limit_hold": stop_limit,
        "pass_shadow": bool(
            (best.get("t3r") or -1e9) > (base.get("t3r") or 0)
            and (conservative.get("t3r") or -1e9) > (base.get("t3r") or 0)
            and (best.get("max_loss") or -1e9) >= (base.get("max_loss") or -1e9)
        ),
        "pass_live_logic": False,
        "reason": "Shadow observer passes if conservative poll/delay/cost remains above baseline; live order-logic still requires forward shadow and operator sign-off.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 State Machine V9 Profit-Lock Execution Realism",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        "- research_only: `true`",
        "- live_changes: `none`",
        "",
        "## Questions Tested",
        "",
        "1. Does profit-lock survive 1/2/5/10/30/60s polling?",
        "2. Does exit delay after lock trigger kill it?",
        "3. Does extra taker/slippage cost kill it?",
        "4. Is partial lock exit better than full lock exit?",
        "5. Is the improvement LONG-side or SHORT-side?",
        "6. Would stop-limit style miss too many exits?",
        "7. Does conservative live-like execution still beat baseline?",
        "8. Does top-3-winner removal still pass?",
        "9. Do folds stay positive?",
        "10. Is this ready for shadow or live?",
        "",
        "## Core Results",
        "",
    ]
    for key in [
        "baseline",
        "poll_2s",
        "live_like_conservative_delay2_cost5",
        "stress_poll5_delay5_cost10",
        "stop_limit_style",
        "long_only_lock",
        "short_only_lock",
    ]:
        val = report["scenarios"][key]
        lines.append(
            f"- {key}: hold `{stat_line(val['hold'])}`, folds={val['folds']['positive_folds']}/5, "
            f"hold_top3_removed `{stat_line(val['hold_top3_removed'])}`, exit_rate={val['exit_rate']}, "
            f"avg_slip={val['avg_exit_slip_from_lock_bps']}"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- pass_shadow: `{report['verdict']['pass_shadow']}`",
        f"- pass_live_logic: `{report['verdict']['pass_live_logic']}`",
        f"- reason: {report['verdict']['reason']}",
        "",
        "## Full JSON",
        "",
        f"- `{OUT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows, *_unused, mk_ts, mk_px = build_base_rows()
    raw = build_signals(rows, FINAL_CFG, mk_ts=mk_ts, mk_px=mk_px)
    signals, blocked = apply_conflict_policy(raw, "short_replace")
    scenarios = scenario_grid(signals, mk_ts, mk_px)
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_changes": "none",
        "data": {
            "raw_signals": len(raw),
            "taken_signals": len(signals),
            "blocked_signals": len(blocked),
        },
        "scenarios": scenarios,
        "observer_payload_fields": [
            "rule",
            "entry_ts_ms",
            "side",
            "entry_price",
            "trigger_bps",
            "lock_bps",
            "pl_triggered",
            "pl_trigger_ts_ms",
            "pl_lock_ts_ms",
            "pl_exit_mode",
            "would_exit",
            "would_exit_price",
            "would_exit_net_bps",
        ],
        "verdict": readiness_verdict(scenarios),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "baseline": report["scenarios"]["baseline"]["hold"],
        "poll_2s": report["scenarios"]["poll_2s"]["hold"],
        "conservative": report["scenarios"]["live_like_conservative_delay2_cost5"]["hold"],
        "stress": report["scenarios"]["stress_poll5_delay5_cost10"]["hold"],
        "verdict": report["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
