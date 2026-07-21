"""S34 v9 kill-criteria redesign and forward-review policy.

Research/risk only. No live executor, .env, leverage, size, or order-logic
changes. This tool answers:

1. Which kill rules actually bound drawdown?
2. Did any catastrophic move begin inside the unprotected fill->stop window at
   tick/book resolution?
3. What fixed forward-review decision gate should be used after 30/60 days?
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
V8_JSON = ROOT / "reports" / "research" / "s34" / "S34_V8_STOP_RELIABILITY_SIZING.json"
MGMT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V9_KILL_FORWARD_REVIEW.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V9_KILL_FORWARD_REVIEW.md"

TAIL_BPS = -100.0
STRESS_TAIL_BPS = 634.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def r3(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


def parse_iso_ms(text: str) -> int:
    t = str(text).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return int(datetime.fromisoformat(t).timestamp() * 1000)


def fixed_rows(stop: dict[str, Any], variant: str = "fixed_sl_150") -> list[dict[str, Any]]:
    rows = [r for r in stop.get("rows") or [] if r.get("variant") == variant]
    rows.sort(key=lambda r: parse_iso_ms(str(r["signal_utc"])))
    return rows


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "mean_bps": None, "median_bps": None, "max_loss_bps": None, "win_rate": None, "t3r_bps": 0.0}
    ordered = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "max_loss_bps": r1(min(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "t3r_bps": r1(sum(ordered[3:])) if len(ordered) > 3 else 0.0,
    }


def equity_curve(vals: list[float], *, notional: float, equity: float) -> dict[str, Any]:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    curve = []
    for i, bps in enumerate(vals, 1):
        pnl = notional * float(bps) / 10_000.0
        cum += pnl
        peak = max(peak, cum)
        dd = cum - peak
        max_dd = min(max_dd, dd)
        curve.append({"i": i, "bps": r1(bps), "cum_usdt": r1(cum), "drawdown_usdt": r1(dd)})
    return {
        "final_pnl_usdt": r1(cum),
        "max_drawdown_usdt": r1(max_dd),
        "max_drawdown_pct_equity": r1(abs(max_dd) / equity * 100.0) if equity else None,
        "curve": curve,
    }


def rule_state(rule: str, vals_so_far: list[float], equity_dd_pct: float, current_bps: float) -> bool:
    """Return True if rule trips after observing current trade."""
    if rule == "FIRST_TAIL_PAUSE":
        return current_bps <= TAIL_BPS
    if rule == "ANY_LOSS_PAUSE":
        return current_bps < 0
    if rule == "ROLLING_2_NEGATIVE":
        return len(vals_so_far) >= 2 and sum(vals_so_far[-2:]) < 0
    if rule == "ROLLING_3_NEGATIVE":
        return len(vals_so_far) >= 3 and sum(vals_so_far[-3:]) < 0
    if rule == "ROLLING_5_NEGATIVE":
        return len(vals_so_far) >= 5 and sum(vals_so_far[-5:]) < 0
    if rule == "DRAWDOWN_10PCT":
        return equity_dd_pct >= 10.0
    if rule == "DRAWDOWN_20PCT":
        return equity_dd_pct >= 20.0
    if rule == "DRAWDOWN_40PCT":
        return equity_dd_pct >= 40.0
    return False


def simulate_kill_rules(vals: list[float], rows: list[dict[str, Any]], *, notional: float, equity: float) -> list[dict[str, Any]]:
    rules = [
        "FIRST_TAIL_PAUSE",
        "ANY_LOSS_PAUSE",
        "ROLLING_2_NEGATIVE",
        "ROLLING_3_NEGATIVE",
        "ROLLING_5_NEGATIVE",
        "DRAWDOWN_10PCT",
        "DRAWDOWN_20PCT",
        "DRAWDOWN_40PCT",
    ]
    out = []
    for rule in rules:
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        observed: list[float] = []
        trigger = None
        kept = []
        for idx, bps in enumerate(vals, 1):
            pnl = notional * float(bps) / 10_000.0
            cum += pnl
            peak = max(peak, cum)
            dd = cum - peak
            max_dd = min(max_dd, dd)
            observed.append(float(bps))
            kept.append(float(bps))
            dd_pct = abs(dd) / equity * 100.0 if equity else 0.0
            if trigger is None and rule_state(rule, observed, dd_pct, float(bps)):
                trigger = {
                    "trade_index": idx,
                    "signal_utc": rows[idx - 1].get("signal_utc") if idx - 1 < len(rows) else None,
                    "trigger_after_bps": r1(bps),
                    "drawdown_pct_equity_at_trigger": r1(dd_pct),
                }
                break
        out.append(
            {
                "rule": rule,
                "triggered": trigger is not None,
                "trigger": trigger,
                "trades_before_pause_including_trigger": len(kept),
                "pnl_until_pause_usdt": r1(cum),
                "max_dd_until_pause_usdt": r1(max_dd),
                "max_dd_until_pause_pct_equity": r1(abs(max_dd) / equity * 100.0) if equity else None,
                "unnecessary_cut_score": sum(1 for v in kept if v > 0 and trigger is not None),
            }
        )
    out.sort(key=lambda r: (float(r["max_dd_until_pause_pct_equity"] or 1e9), int(r["trades_before_pause_including_trigger"])))
    return out


def kill_design(rows: list[dict[str, Any]], *, equity: float, current_notional: float, weighted_notional: float, tail_notional: float) -> dict[str, Any]:
    vals = [float(r.get("baseline_net_bps") or 0.0) for r in rows]
    return {
        "hard_truth": "No realized-PnL kill rule can fire before the first tail; only sizing/stop can bound first-tail loss.",
        "sequence_summary_bps": summary(vals),
        "current_env": {
            "notional_usdt": r1(current_notional),
            "curve": equity_curve(vals, notional=current_notional, equity=equity),
            "rules": simulate_kill_rules(vals, rows, notional=current_notional, equity=equity),
        },
        "weighted_size": {
            "notional_usdt": r1(weighted_notional),
            "curve": equity_curve(vals, notional=weighted_notional, equity=equity),
            "rules": simulate_kill_rules(vals, rows, notional=weighted_notional, equity=equity),
        },
        "tail_size": {
            "notional_usdt": r1(tail_notional),
            "curve": equity_curve(vals, notional=tail_notional, equity=equity),
            "rules": simulate_kill_rules(vals, rows, notional=tail_notional, equity=equity),
        },
        "recommended_kill_rule": {
            "name": "FIRST_TAIL_OR_10PCT_DD_PAUSE",
            "logic": "pause after any closed trade <= -100 bps OR realized equity drawdown >= 10%; operator review required to resume",
            "reason": "FIRST_TAIL stops repeat-tail exposure; DD threshold catches non-tail loss clusters. It still cannot protect the first tail.",
        },
    }


def min_trade_price(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    row = conn.execute(
        "SELECT MIN(price) FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def min_book_bid(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    row = conn.execute(
        "SELECT MIN(bid_price) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def tick_atomicity_scan(db: Path, rows: list[dict[str, Any]], *, gap_sec: float) -> dict[str, Any]:
    cards = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for row in rows:
            signal_ms = parse_iso_ms(str(row["signal_utc"]))
            fill_ms = int(signal_ms + float(row.get("fill_delay_sec") or 0.0) * 1000)
            end_ms = fill_ms + int(float(gap_sec) * 1000)
            entry = float(row["entry_price"])
            min_trade = min_trade_price(conn, "ETHUSDT", fill_ms, end_ms)
            min_bid = min_book_bid(conn, "ETHUSDT", fill_ms, end_ms)
            trade_adv = None if min_trade is None else (min_trade - entry) / entry * 10_000.0
            bid_adv = None if min_bid is None else (min_bid - entry) / entry * 10_000.0
            worst = min([v for v in (trade_adv, bid_adv) if v is not None], default=None)
            cards.append(
                {
                    "event_id": row.get("event_id"),
                    "signal_utc": row.get("signal_utc"),
                    "fill_utc": iso_ms(fill_ms),
                    "gap_end_utc": iso_ms(end_ms),
                    "baseline_net_bps": row.get("baseline_net_bps"),
                    "entry_price": row.get("entry_price"),
                    "trade_adverse_bps": r1(trade_adv),
                    "book_bid_adverse_bps": r1(bid_adv),
                    "worst_tick_adverse_bps": r1(worst),
                    "catastrophic": bool(worst is not None and worst <= -150.0),
                }
            )
    vals = [float(c["worst_tick_adverse_bps"]) for c in cards if c.get("worst_tick_adverse_bps") is not None]
    return {
        "status": "NO_TICK_CATASTROPHIC_GAP_FOUND" if not any(c["catastrophic"] for c in cards) else "TICK_CATASTROPHIC_GAP_FOUND",
        "gap_sec": float(gap_sec),
        "filled_n": len(cards),
        "covered_n": len(vals),
        "worst_tick_adverse_bps": r1(min(vals)) if vals else None,
        "lte_5bps_n": sum(1 for v in vals if v <= -5.0),
        "lte_25bps_n": sum(1 for v in vals if v <= -25.0),
        "lte_150bps_n": sum(1 for v in vals if v <= -150.0),
        "worst_cards": sorted(cards, key=lambda c: float(c.get("worst_tick_adverse_bps") or 0.0))[:10],
        "read": "Bounded by filled lifecycle rows; uses agg_trades and book_ticker inside the unprotected 2s window.",
    }


def forward_review_policy() -> dict[str, Any]:
    return {
        "status": "FROZEN_DECISION_GATE",
        "checkpoints": [
            {
                "name": "30D_INTERIM",
                "minimum": ">=30 calendar days AND >=10 independent closed forward fills",
                "actions": [
                    "if sum_bps < 0 or any tail-budget breach -> recommend disarm",
                    "if N < 10 -> continue observation, no promotion",
                    "if positive -> no resize yet; wait for 60D",
                ],
            },
            {
                "name": "60D_DECISION",
                "minimum": ">=60 calendar days AND >=20 independent closed forward fills across >=2 UTC weeks",
                "promote_to_operator_review_only_if": [
                    "sum_bps > 0",
                    "T3R > 0",
                    "median_bps > 0",
                    "win_rate >= 0.55",
                    "max realized loss does not breach accepted budget",
                    "no permutation/artifact gate failure on forward sample",
                ],
                "disarm_recommendation_if": [
                    "sum_bps < 0",
                    "T3R < 0",
                    "tail-budget breach",
                    "operator keeps size > weighted budget",
                ],
            },
        ],
        "non_negotiable": "Passing this gate authorizes only operator review, not automatic live scaling.",
    }


def extract_v8_notional(v8: dict[str, Any], basis: str, default: float) -> float:
    rows = ((v8.get("stop_reliability_weighted_sizing") or {}).get("sizing_rows") or [])
    for row in rows:
        if row.get("basis") == basis:
            return float(row.get("max_notional_usdt") or default)
    return float(default)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    stop = load_json(args.stop_json, {})
    v8 = load_json(args.v8_json, {})
    mgmt = load_json(args.management_json, {})
    rows = fixed_rows(stop)
    current_notional = float((((mgmt.get("tail_aware_sizing_monitor") or {}).get("planned_live_size_from_env") or {}).get("planned_notional_usdt")) or 1190.0)
    weighted_notional = extract_v8_notional(v8, "conservative_weighted", 16.3)
    tail_notional = extract_v8_notional(v8, "tail_only_hard_floor", 11.0)
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_RISK_ONLY_NO_LIVE_CHANGE",
        "kill_criteria_redesign": kill_design(rows, equity=float(args.equity_usdt), current_notional=current_notional, weighted_notional=weighted_notional, tail_notional=tail_notional),
        "tick_level_atomicity_scan": tick_atomicity_scan(args.db, rows, gap_sec=float(args.atomicity_gap_sec)),
        "forward_review_decision_gate": forward_review_policy(),
        "final_read": "Kill rules cannot prevent the first tail; use weighted/tail sizing for first-loss survival, then FIRST_TAIL_OR_10PCT_DD_PAUSE for repeat-risk control.",
    }


def fmt_rule(row: dict[str, Any]) -> str:
    trig = row.get("trigger") or {}
    return (
        f"trigger={row.get('triggered')} idx={trig.get('trade_index')} "
        f"dd%={row.get('max_dd_until_pause_pct_equity')} pnl=${row.get('pnl_until_pause_usdt')}"
    )


def render_md(report: dict[str, Any]) -> str:
    kill = report["kill_criteria_redesign"]
    tick = report["tick_level_atomicity_scan"]
    policy = report["forward_review_decision_gate"]
    current_rules = kill["current_env"]["rules"]
    weighted_rules = kill["weighted_size"]["rules"]
    lines = [
        "# S34 v9 Kill Criteria & Forward Review",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## 1. Kill Criteria",
        "",
        f"- hard truth: {kill['hard_truth']}",
        f"- recommended rule: `{kill['recommended_kill_rule']['name']}`",
        f"- logic: {kill['recommended_kill_rule']['logic']}",
        f"- reason: {kill['recommended_kill_rule']['reason']}",
        "",
        "### Current Env Notional",
        "",
        f"- notional: `${kill['current_env']['notional_usdt']}`",
        f"- max DD: `${kill['current_env']['curve']['max_drawdown_usdt']}` = `{kill['current_env']['curve']['max_drawdown_pct_equity']}%` equity",
        "",
        "| Rule | Result |",
        "| --- | --- |",
    ]
    for row in current_rules:
        lines.append(f"| `{row['rule']}` | {fmt_rule(row)} |")
    lines.extend(
        [
            "",
            "### Conservative Weighted Size",
            "",
            f"- notional: `${kill['weighted_size']['notional_usdt']}`",
            f"- max DD: `${kill['weighted_size']['curve']['max_drawdown_usdt']}` = `{kill['weighted_size']['curve']['max_drawdown_pct_equity']}%` equity",
            "",
            "| Rule | Result |",
            "| --- | --- |",
        ]
    )
    for row in weighted_rules:
        lines.append(f"| `{row['rule']}` | {fmt_rule(row)} |")
    lines.extend(
        [
            "",
            "## 2. Tick-Level Atomicity Scan",
            "",
            f"- status: `{tick['status']}`",
            f"- filled N: `{tick['filled_n']}` covered N: `{tick['covered_n']}`",
            f"- worst tick adverse: `{tick['worst_tick_adverse_bps']} bps`",
            f"- <= -5bps N: `{tick['lte_5bps_n']}`",
            f"- <= -25bps N: `{tick['lte_25bps_n']}`",
            f"- <= -150bps N: `{tick['lte_150bps_n']}`",
            f"- read: {tick['read']}",
            "",
            "## 3. Forward Review Gate",
            "",
            f"- status: `{policy['status']}`",
            f"- non-negotiable: {policy['non_negotiable']}",
        ]
    )
    for cp in policy["checkpoints"]:
        lines.append(f"- `{cp['name']}` minimum: {cp['minimum']}")
    lines.extend(["", "## Final Read", "", report["final_read"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 v9 kill criteria and forward review.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--stop-json", type=Path, default=STOP_JSON)
    p.add_argument("--v8-json", type=Path, default=V8_JSON)
    p.add_argument("--management-json", type=Path, default=MGMT_JSON)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--equity-usdt", type=float, default=35.0)
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
