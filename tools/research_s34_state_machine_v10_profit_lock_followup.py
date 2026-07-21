"""S34 state-machine v10 profit-lock follow-up suite.

Research-only. No live executor, env, runtime state, orders, buckets, or
dashboard changes.

This suite answers the next 10 profit-lock observer questions after V9:
- forward observer sample count/delta;
- SHORT-only confirmation;
- trigger/lock sensitivity;
- lock execution delay reality;
- false-lock / missed-upside anatomy;
- confidence, session, and BTC-divergence conditioning;
- adverse no-trigger case;
- forward promotion rule.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    DEFAULT_DB,
    apply_conflict_policy,
    build_signals,
    fold_summaries,
    mark_at_or_after,
    summary_with_dd,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import build_base_rows  # noqa: E402
from tools.research_s34_state_machine_v6_development_ideas import (  # noqa: E402
    FINAL_CFG,
    mfe_mae_for_signal,
    split,
)
from tools.research_s34_state_machine_v9_profit_lock_execution import (  # noqa: E402
    enrich_stats,
    simulate_profit_lock,
    stat_line,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V10_PROFIT_LOCK_FOLLOWUP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V10_PROFIT_LOCK_FOLLOWUP.md"
SHADOW_LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def top_removed(rows: list[dict[str, Any]], n: int = 3) -> dict[str, Any]:
    return summary_with_dd(sorted(rows, key=lambda r: float(r["net_bps"]), reverse=True)[n:])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "split": split(rows),
        "folds": fold_summaries(rows, folds=5),
        "hold_top3_removed": top_removed([r for r in rows if r["row"]["is_hold"]], 3),
    }


def read_forward_observer() -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if SHADOW_LEDGER.exists():
        with SHADOW_LEDGER.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    by_type: dict[str, int] = {}
    deltas = []
    for ev in events:
        typ = str(ev.get("event") or ev.get("type") or ev.get("kind") or "UNKNOWN")
        by_type[typ] = by_type.get(typ, 0) + 1
        if ev.get("delta_vs_baseline_bps") is not None:
            deltas.append(float(ev["delta_vs_baseline_bps"]))
    exits = [ev for ev in events if str(ev.get("event") or ev.get("type") or "").upper() == "PROFIT_LOCK_SHADOW_EXIT"]
    return {
        "ledger": str(SHADOW_LEDGER),
        "exists": SHADOW_LEDGER.exists(),
        "events_n": len(events),
        "events_by_type": by_type,
        "profit_lock_shadow_exit_n": len(exits),
        "delta_sum_bps": round(sum(deltas), 1) if deltas else 0.0,
        "delta_avg_bps": round(sum(deltas) / len(deltas), 2) if deltas else None,
        "status": "WAITING_FOR_FORWARD_SAMPLES" if len(exits) == 0 else "HAS_FORWARD_SAMPLES",
    }


def baseline_rows(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**s, "net_bps": round(float(s["net_bps"]), 1)} for s in signals]


def trigger_sensitivity(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for trig, lock in [(75, 25), (100, 50), (125, 50), (125, 75), (150, 75)]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=trig, lock_bps=lock, poll_sec=2)
        out[f"trig{trig}_lock{lock}"] = enrich_stats(rows)
    return out


def delay_reality(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for delay in [0, 2, 5, 10, 30]:
        rows = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2, exit_delay_sec=delay)
        out[f"delay_{delay}s"] = enrich_stats(rows)
    return out


def false_lock_anatomy(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    managed = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    by_key = {(int(r["entry_ts_ms"]), str(r["side"]), str(r["arm"])): r for r in managed}
    cards = []
    for base in baseline_rows(signals):
        m = by_key.get((int(base["entry_ts_ms"]), str(base["side"]), str(base["arm"])))
        if not m or not m.get("pl_exit"):
            continue
        delta = float(m["net_bps"]) - float(base["net_bps"])
        missed = float(base["net_bps"]) - float(m["net_bps"])
        cards.append({**m, "baseline_net_bps": round(float(base["net_bps"]), 1), "delta_vs_baseline_bps": round(delta, 1), "missed_upside_bps": round(missed, 1)})
    helped = [r for r in cards if float(r["delta_vs_baseline_bps"]) > 0]
    hurt = [r for r in cards if float(r["delta_vs_baseline_bps"]) < 0]
    return {
        "exit_n": len(cards),
        "helped_n": len(helped),
        "hurt_n": len(hurt),
        "helped_delta_sum_bps": round(sum(float(r["delta_vs_baseline_bps"]) for r in helped), 1),
        "hurt_delta_sum_bps": round(sum(float(r["delta_vs_baseline_bps"]) for r in hurt), 1),
        "missed_upside_gt_50_n": sum(1 for r in cards if float(r["missed_upside_bps"]) > 50),
        "missed_upside_gt_100_n": sum(1 for r in cards if float(r["missed_upside_bps"]) > 100),
        "exited_summary": split(cards),
        "helped_summary": split(helped),
        "hurt_summary": split(hurt),
        "worst_false_locks": sorted(cards, key=lambda r: float(r["delta_vs_baseline_bps"]))[:10],
        "best_saved_trades": sorted(cards, key=lambda r: float(r["delta_vs_baseline_bps"]), reverse=True)[:10],
    }


def confidence_rows(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        mm = mfe_mae_for_signal(s, mk_ts, mk_px, 5 * 60_000)
        early_ok = bool(mm and mm["mfe_bps"] >= 20)
        score = int(s.get("score") or 0)
        conf = 0
        conf += int(score >= 4)
        conf += int(early_ok)
        conf += int(s["arm"] == "SILENCE_LONG" and s["row"].get("vd", 0) >= 30)
        conf += int(s["arm"] == "NEITHER_SHORT" and (int(s["entry_ts_ms"]) - int(s["anchor_ts_ms"])) <= 15 * 60_000)
        conf += int(abs(float(s["row"].get("b4h") or 0)) >= 50)
        out.append({**s, "confidence": conf})
    return out


def profit_lock_confidence(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    enriched = confidence_rows(signals, mk_ts, mk_px)
    managed = simulate_profit_lock(enriched, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    out = {}
    for label, pred in {
        "conf_0_1": lambda r: int(r.get("confidence") or 0) <= 1,
        "conf_2": lambda r: int(r.get("confidence") or 0) == 2,
        "conf_ge3": lambda r: int(r.get("confidence") or 0) >= 3,
        "conf_ge4": lambda r: int(r.get("confidence") or 0) >= 4,
    }.items():
        base_sub = [r for r in enriched if pred(r)]
        managed_sub = [r for r in managed if pred(r)]
        out[label] = {
            "baseline": summarize(base_sub),
            "profit_lock": enrich_stats(managed_sub),
        }
    return out


def profit_lock_session(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    managed = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    sessions = sorted({str(s["row"].get("session")) for s in signals})
    out = {}
    for session in sessions:
        out[session] = {
            "baseline": summarize([s for s in signals if str(s["row"].get("session")) == session]),
            "profit_lock": enrich_stats([s for s in managed if str(s["row"].get("session")) == session]),
        }
    return out


def add_btc_divergence(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not signals:
        return []
    ts_list = [int(s["anchor_ts_ms"]) for s in signals]
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
            (min(ts_list) - 3600_000, max(ts_list) + 60_000),
        ).fetchall()
    btc_ts = [int(r[0]) for r in rows]
    btc_px = [float(r[1]) for r in rows]
    enriched = []
    for s in signals:
        ts = int(s["anchor_ts_ms"])
        b0 = mark_at_or_after(btc_ts, btc_px, ts - 30 * 60_000)
        b1 = mark_at_or_after(btc_ts, btc_px, ts)
        btc30 = ((b1 - b0) / b0 * 10_000.0) if b0 and b1 and b0 > 0 else 0.0
        eth30 = float(s["row"].get("eth_shift_30_bps") or 0.0)
        div = eth30 - btc30
        enriched.append({**s, "btc30_bps": round(btc30, 1), "eth_minus_btc_30m": round(div, 1)})
    return enriched


def profit_lock_btc_divergence(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    enriched = add_btc_divergence(signals)
    managed = simulate_profit_lock(enriched, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    buckets = {
        "eth_weaker_than_btc_lt_-20": lambda r: float(r.get("eth_minus_btc_30m") or 0) < -20,
        "eth_inline_-20_20": lambda r: -20 <= float(r.get("eth_minus_btc_30m") or 0) <= 20,
        "eth_stronger_than_btc_gt_20": lambda r: float(r.get("eth_minus_btc_30m") or 0) > 20,
    }
    out = {}
    for label, pred in buckets.items():
        out[label] = {
            "baseline": summarize([r for r in enriched if pred(r)]),
            "profit_lock": enrich_stats([r for r in managed if pred(r)]),
        }
    return out


def adverse_no_trigger(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    managed = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    no_trigger = [r for r in managed if not r.get("pl_triggered")]
    triggered = [r for r in managed if r.get("pl_triggered")]
    # Historical check only: if no-trigger rows are the danger pocket, do not turn
    # this into an entry filter without a separate frozen forward test.
    return {
        "triggered": enrich_stats(triggered),
        "no_trigger": enrich_stats(no_trigger),
        "no_trigger_by_side": {
            "LONG": enrich_stats([r for r in no_trigger if r["side"] == "LONG"]),
            "SHORT": enrich_stats([r for r in no_trigger if r["side"] == "SHORT"]),
        },
        "read": "No-trigger rows are adverse diagnostics after entry; this is not a pre-entry gate.",
    }


def promotion_rule(forward: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    base_hold = historical["baseline"]["hold"]
    pl_hold = historical["profit_lock_100_50"]["hold"]
    conservative_hold = historical["delay_10s"]["hold"]
    criteria = {
        "forward_shadow_exit_n_ge_20": forward["profit_lock_shadow_exit_n"] >= 20,
        "forward_delta_positive": (forward["delta_sum_bps"] or 0.0) > 0,
        "historical_hold_sum_improves": (pl_hold.get("sum") or -1e9) > (base_hold.get("sum") or 0),
        "historical_hold_t3r_improves": (pl_hold.get("t3r") or -1e9) > (base_hold.get("t3r") or 0),
        "historical_max_loss_not_worse": (pl_hold.get("max_loss") or -1e9) >= (base_hold.get("max_loss") or -1e9),
        "delay_10s_still_improves_t3r": (conservative_hold.get("t3r") or -1e9) > (base_hold.get("t3r") or 0),
        "folds_positive_5_of_5": historical["profit_lock_100_50"]["folds"]["positive_folds"] == 5,
    }
    return {
        "criteria": criteria,
        "status": "PROMOTABLE_TO_LIVE_LOGIC" if all(criteria.values()) else "SHADOW_RUNNING_NOT_PROMOTABLE",
        "reason": "Forward sample gate blocks promotion until >=20 shadow exits with positive delta; live logic still needs operator sign-off.",
    }


def render_md(report: dict[str, Any]) -> str:
    hist = report["historical"]
    lines = [
        "# S34 State Machine V10 Profit-Lock Follow-Up",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        "- research_only: `true`",
        "- live_changes: `none`",
        f"- rule: `{report['rule']}`",
        "",
        "## Core Read",
        "",
        f"- forward observer: `{report['forward_observer']['status']}`, exits={report['forward_observer']['profit_lock_shadow_exit_n']}, delta_sum={report['forward_observer']['delta_sum_bps']} bps",
        f"- baseline hold: `{stat_line(hist['baseline']['hold'])}`",
        f"- profit_lock_100_50 hold: `{stat_line(hist['profit_lock_100_50']['hold'])}`",
        f"- short_only_lock hold: `{stat_line(hist['short_only_lock']['hold'])}`",
        f"- long_only_lock hold: `{stat_line(hist['long_only_lock']['hold'])}`",
        "",
        "## Trigger Sensitivity",
        "",
    ]
    for key, val in report["trigger_sensitivity"].items():
        lines.append(f"- {key}: hold `{stat_line(val['hold'])}`, exit_rate={val['exit_rate']}")
    lines += ["", "## Delay Reality", ""]
    for key, val in report["delay_reality"].items():
        lines.append(f"- {key}: hold `{stat_line(val['hold'])}`, avg_slip={val['avg_exit_slip_from_lock_bps']}")
    lines += [
        "",
        "## False Lock",
        "",
        f"- exits={report['false_lock']['exit_n']}, helped={report['false_lock']['helped_n']}, hurt={report['false_lock']['hurt_n']}",
        f"- helped_delta_sum={report['false_lock']['helped_delta_sum_bps']} bps, hurt_delta_sum={report['false_lock']['hurt_delta_sum_bps']} bps",
        f"- missed_upside_gt_50={report['false_lock']['missed_upside_gt_50_n']}, missed_upside_gt_100={report['false_lock']['missed_upside_gt_100_n']}",
        "",
        "## Adverse No-Trigger Case",
        "",
        f"- triggered hold: `{stat_line(report['adverse_no_trigger']['triggered']['hold'])}`",
        f"- no_trigger hold: `{stat_line(report['adverse_no_trigger']['no_trigger']['hold'])}`",
        "",
        "## Promotion Rule",
        "",
        f"- status: `{report['promotion_rule']['status']}`",
        f"- reason: {report['promotion_rule']['reason']}",
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
    base = baseline_rows(signals)
    pl_100_50 = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2)
    short_only = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2, side_filter="SHORT")
    long_only = simulate_profit_lock(signals, mk_ts, mk_px, trigger_bps=100, lock_bps=50, poll_sec=2, side_filter="LONG")
    delays = delay_reality(signals, mk_ts, mk_px)
    historical = {
        "baseline": enrich_stats(base),
        "profit_lock_100_50": enrich_stats(pl_100_50),
        "short_only_lock": enrich_stats(short_only),
        "long_only_lock": enrich_stats(long_only),
        "delay_10s": delays["delay_10s"],
    }
    forward = read_forward_observer()
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_changes": "none",
        "rule": FINAL_CFG.name,
        "data": {
            "raw_signals": len(raw),
            "taken_signals": len(signals),
            "blocked_signals": len(blocked),
        },
        "forward_observer": forward,
        "historical": historical,
        "trigger_sensitivity": trigger_sensitivity(signals, mk_ts, mk_px),
        "delay_reality": delays,
        "false_lock": false_lock_anatomy(signals, mk_ts, mk_px),
        "profit_lock_confidence": profit_lock_confidence(signals, mk_ts, mk_px),
        "profit_lock_session": profit_lock_session(signals, mk_ts, mk_px),
        "profit_lock_btc_divergence": profit_lock_btc_divergence(signals, mk_ts, mk_px),
        "adverse_no_trigger": adverse_no_trigger(signals, mk_ts, mk_px),
    }
    report["promotion_rule"] = promotion_rule(forward, historical)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "forward": forward,
        "baseline_hold": historical["baseline"]["hold"],
        "profit_lock_hold": historical["profit_lock_100_50"]["hold"],
        "short_only_hold": historical["short_only_lock"]["hold"],
        "promotion": report["promotion_rule"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
