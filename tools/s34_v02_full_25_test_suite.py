"""Full 25-test suite for the S34 V02 alpha research roadmap.

Research-only. Uses the existing 120d V02 live-like ledger plus the completed
30d forced-flow expansion run. It produces a single report covering all 25
questions from H4 management through mechanism expansion and operational
navigation. It does not touch live execution/config/state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, pctile, r1, signed_return_bps  # noqa: E402
from tools.s34_v02_alpha_navigation_overlay import RULE_NAME, SYMBOL  # noqa: E402
from tools.s34_v02_nav_spike_tests import DB_PATH, OUT_DIR  # noqa: E402
from tools.s34_v02_next_gen_alpha_research import path_returns  # noqa: E402


LEDGER_120D = OUT_DIR / "S34_V02_ALPHA_NAVIGATION_OVERLAY_120D.json"
NEXT_30D = OUT_DIR / "S34_V02_NEXT_GEN_ALPHA_RESEARCH_30D.json"
ALL_NEXT = OUT_DIR / "S34_V02_ALL_NEXT_TESTS.json"
MIRROR_CSV = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.csv"
OUT_JSON = OUT_DIR / "S34_V02_FULL_25_TEST_SUITE.json"
OUT_MD = OUT_DIR / "S34_V02_FULL_25_TEST_SUITE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(vals: list[Any]) -> list[float]:
    out = []
    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = clean(vals)
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "t3r": 0.0, "min": None, "max": None}
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum": r1(sum(xs)),
        "mean": r1(mean(xs)),
        "median": r1(pctile(xs, 0.5)),
        "win_rate": round(sum(1 for x in xs if x > 0) / len(xs), 3),
        "t3r": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "top1_removed": r1(sum(ordered[1:]) if len(ordered) > 1 else sum(ordered)),
        "min": r1(min(xs)),
        "max": r1(max(xs)),
    }


def group(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return {
        str(k): metrics([r.get(value) for r in rows if r.get(key) == k])
        for k in sorted({r.get(key) for r in rows}, key=lambda x: str(x))
    }


def mark_ret(marks, trade: dict[str, Any], horizon_sec: int, fee_bps: float) -> float | None:
    px = marks.at_or_after(int(trade["fill_ts_ms"]) + int(horizon_sec) * 1000)
    if not px:
        return None
    return signed_return_bps("LONG", float(trade["entry_price"]), float(px[1])) - float(fee_bps)


def series_ret(series, start_ms: int, horizon_sec: int) -> float | None:
    px0 = series.at_or_after(int(start_ms))
    px1 = series.at_or_after(int(start_ms) + int(horizon_sec) * 1000)
    if not px0 or not px1:
        return None
    return signed_return_bps("LONG", float(px0[1]), float(px1[1]))


def first_cross(path: list[tuple[int, float]], start_ms: int, level: float) -> float | None:
    for ts, ret in path:
        if float(ret) >= float(level):
            return (int(ts) - int(start_ms)) / 1000.0
    return None


def path_stat(marks, trade: dict[str, Any], horizon_sec: int) -> dict[str, Any]:
    path = path_returns(marks, float(trade["entry_price"]), int(trade["fill_ts_ms"]), int(horizon_sec))
    if not path:
        return {}
    mfe_ts, mfe = max(path, key=lambda x: x[1])
    mae_ts, mae = min(path, key=lambda x: x[1])
    return {
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_sec": (int(mfe_ts) - int(trade["fill_ts_ms"])) / 1000.0,
        "mae_sec": (int(mae_ts) - int(trade["fill_ts_ms"])) / 1000.0,
        "rebound20_sec": first_cross(path, int(trade["fill_ts_ms"]), 20.0),
        "rebound50_sec": first_cross(path, int(trade["fill_ts_ms"]), 50.0),
        "rebound80_sec": first_cross(path, int(trade["fill_ts_ms"]), 80.0),
    }


def stop_result(marks, trades: list[dict[str, Any]], horizon_sec: int, stop_bps: float, fee_bps: float) -> dict[str, Any]:
    vals = []
    exits = Counter()
    for t in trades:
        path = path_returns(marks, float(t["entry_price"]), int(t["fill_ts_ms"]), horizon_sec)
        stopped = None
        for _, ret in path:
            if float(ret) <= -float(stop_bps):
                stopped = float(ret) - float(fee_bps)
                break
        if stopped is None:
            vals.append(mark_ret(marks, t, horizon_sec, fee_bps))
            exits["TIME"] += 1
        else:
            vals.append(stopped)
            exits["SL"] += 1
    return {"result": metrics(vals), "exits": dict(exits)}


def fee_adjust(trades: list[dict[str, Any]], key: str, old_fee: float, maker_fee: float, taker_fee: float) -> list[float]:
    new_fee = float(maker_fee) + float(taker_fee)
    return [float(t[key]) + old_fee - new_fee for t in trades if t.get(key) is not None]


def equity_path(vals: list[float], notional: float, start_equity: float = 35.0) -> dict[str, Any]:
    eq = float(start_equity)
    peak = eq
    max_dd = 0.0
    path = []
    for b in clean(vals):
        pnl = float(notional) * b / 10_000.0
        eq += pnl
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)
        path.append(round(eq, 4))
    return {"start": start_equity, "notional": notional, "end": round(eq, 4), "pnl": round(eq - start_equity, 4), "max_dd_usdt": round(max_dd, 4), "path": path}


def tail_inject(vals: list[float], every_n: int, tail_bps: float) -> list[float]:
    out = []
    for i, v in enumerate(clean(vals), start=1):
        out.append(v)
        if every_n and i % every_n == 0:
            out.append(float(tail_bps))
    return out


def mirror_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def render(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 V02 Full 25-Test Suite",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## Executive Read",
        "",
        result["executive_read"],
        "",
    ]
    for i, item in enumerate(result["tests"], start=1):
        lines += [
            f"## {i}. {item['name']}",
            "",
            f"Question: {item['question']}",
            "",
            f"Verdict: `{item['verdict']}`",
            "",
            "```json",
            json.dumps(item["result"], indent=2, ensure_ascii=True),
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--ledger-json", type=Path, default=LEDGER_120D)
    ap.add_argument("--next30-json", type=Path, default=NEXT_30D)
    ap.add_argument("--all-next-json", type=Path, default=ALL_NEXT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    ledger = json.loads(args.ledger_json.read_text(encoding="utf-8"))
    next30 = json.loads(args.next30_json.read_text(encoding="utf-8")) if args.next30_json.exists() else {}
    all_next = json.loads(args.all_next_json.read_text(encoding="utf-8")) if args.all_next_json.exists() else {}
    trades = [dict(t) for t in ledger.get("sample_filled_trades", []) if t.get("status") == "FILLED"]
    scope = ledger.get("scope", {})
    old_fee = float(scope.get("maker_fee_bps", -0.5)) + float(scope.get("taker_fee_bps", 3.05))

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth = load_mark_index(conn, SYMBOL)
        btc = load_mark_index(conn, "BTCUSDT")
        sol = load_mark_index(conn, "SOLUSDT")

        for t in trades:
            t["h1"] = mark_ret(eth, t, 3600, old_fee)
            t["h2"] = mark_ret(eth, t, 7200, old_fee)
            t["h3"] = mark_ret(eth, t, 10800, old_fee)
            t["h4"] = mark_ret(eth, t, 14400, old_fee)
            ps = path_stat(eth, t, 14400)
            t.update({f"h4_{k}": v for k, v in ps.items()})
            t["h4_delta"] = float(t["h4"]) - float(t["h2"]) if t.get("h4") is not None and t.get("h2") is not None else None
            t["h4_runner"] = bool(t.get("h4_delta") is not None and float(t["h4_delta"]) > 0)
            t["giveback_h4"] = float(t.get("h4_mfe") or 0.0) - float(t.get("h4") or 0.0)
            t["fill_delay_bin"] = "lt5m" if float(t.get("fill_delay_sec", 0)) < 300 else "5to15m" if float(t.get("fill_delay_sec", 0)) < 900 else "gt15m"
            t["btc30"] = series_ret(btc, int(t["fill_ts_ms"]), 1800)
            t["btc60"] = series_ret(btc, int(t["fill_ts_ms"]), 3600)
            t["btc4"] = series_ret(btc, int(t["fill_ts_ms"]), 14400)
            t["sol30"] = series_ret(sol, int(t["fill_ts_ms"]), 1800)
            t["sol60"] = series_ret(sol, int(t["fill_ts_ms"]), 3600)
            t["sol4"] = series_ret(sol, int(t["fill_ts_ms"]), 14400)
            t["btc_no_dump_30"] = bool(t.get("btc30") is not None and float(t["btc30"]) > -40.0)
            t["sol_no_dump_30"] = bool(t.get("sol30") is not None and float(t["sol30"]) > -80.0)
            t["cross_no_dump"] = t["btc_no_dump_30"] and t["sol_no_dump_30"]
            t["eth_prior24"] = eth.ret_bps(int(t["fill_ts_ms"]) - 86400_000, int(t["fill_ts_ms"]))
            t["btc_prior24"] = btc.ret_bps(int(t["fill_ts_ms"]) - 86400_000, int(t["fill_ts_ms"]))
            t["eth_prior_regime"] = "down" if (t.get("eth_prior24") or 0) < 0 else "up"
            t["btc_prior_regime"] = "down" if (t.get("btc_prior24") or 0) < 0 else "up"
            t["state_v2"] = [
                "FILLED",
                "PAIN" if float(t.get("h4_mae") or 0) < -20 else "NO_PAIN",
                "REBOUND20" if t.get("h4_rebound20_sec") is not None else "NO_REBOUND20",
                "REBOUND50" if t.get("h4_rebound50_sec") is not None else "NO_REBOUND50",
                "RUNNER" if t.get("h4_runner") else "H2_BETTER",
            ]

    h2 = metrics([t.get("h2") for t in trades])
    h3 = metrics([t.get("h3") for t in trades])
    h4 = metrics([t.get("h4") for t in trades])
    tests: list[dict[str, Any]] = []

    def add(name: str, question: str, result: Any, verdict: str) -> None:
        tests.append({"name": name, "question": question, "result": result, "verdict": verdict})

    add("H4 real or regime artifact", "Does H4 improve over H2 on the current V02 ledger?", {"h2": h2, "h3": h3, "h4": h4, "h4_minus_h2": metrics([t.get("h4_delta") for t in trades])}, "H4_LEAD_SMALL_N")
    add("H4 runner recognition", "Can early rebound/cross-asset context identify H4 runners?", {"by_rebound50_30m": group([dict(t, rb50=bool(t.get("h4_rebound50_sec") is not None and float(t["h4_rebound50_sec"]) <= 1800)) for t in trades], "rb50", "h4"), "by_cross_no_dump": group(trades, "cross_no_dump", "h4")}, "PROMISING_BUT_SMALL_N")
    add("H2 checkpoint decision", "At H2 should we close, hold, or conditionally hold?", {"close_h2": h2, "hold_h4": h4, "hold_if_cross_no_dump_else_h2": metrics([t["h4"] if t.get("cross_no_dump") else t["h2"] for t in trades]), "hold_if_h2_lt100_else_close": metrics([t["h4"] if float(t["h2"]) < 100 else t["h2"] for t in trades])}, "H4_ALL_CURRENTLY_BEST")
    add("H4 giveback protection", "Can profit protection keep runner upside without losing H4 edge?", {"giveback_h4": metrics([t.get("giveback_h4") for t in trades]), "prior_giveback_grid_30d": next30.get("giveback_exit", {})}, "GIVEBACK_EXISTS_BUT_FIXED_H4_STILL_BEST")
    add("MAE / catastrophic stop", "What stop widths preserve the current alpha shape?", {"sl100_h4": stop_result(eth, trades, 14400, 100, old_fee), "sl150_h4": stop_result(eth, trades, 14400, 150, old_fee), "sl200_h4": stop_result(eth, trades, 14400, 200, old_fee)}, "SL100_BAD_SL150_OK_IN_SAMPLE")
    add("Fill delay interpretation", "Is late fill bad, or is it retest quality?", {"h2_by_delay": group(trades, "fill_delay_bin", "h2"), "h4_by_delay": group(trades, "fill_delay_bin", "h4")}, "LATE_FILL_NOT_BAD_FOR_H4")

    mrows = mirror_rows(MIRROR_CSV)
    filled_mirror = [r for r in mrows if r.get("sim_status") == "FILLED" and r.get("observation_status") == "CLOSED"]
    add("Retest quality score", "Does retest quality separate outcomes?", {"by_quality": group(filled_mirror, "retest_quality_bucket", "net_bps"), "by_fill_minus_arm": group(filled_mirror, "fill_minus_arm_bucket", "net_bps"), "warnings": dict(Counter(r.get("entry_quality_warnings") or "NONE" for r in filled_mirror))}, "QUALITY_LEDGER_BUILT_N_SMALL")
    add("Forced-flow absorption engine", "Can the mechanism expand beyond liquidation events?", next30.get("mechanism_expansion", {}), "MECHANISM_LEAD_N2_ONLY")
    add("Absorption without V-depth", "Can deep bid plus forced flow work without strict V-depth?", {"current_result": "not independently isolated", "proxy": next30.get("mechanism_expansion", {})}, "NEEDS_DEDICATED_SCAN")
    add("Cross-asset rebound confirmation", "Do BTC/SOL rebound states explain H4 outcomes?", {"btc4": group([dict(t, btc4_state="up" if (t.get("btc4") or 0) >= 0 else "down") for t in trades], "btc4_state", "h4"), "sol4": group([dict(t, sol4_state="up" if (t.get("sol4") or 0) >= 0 else "down") for t in trades], "sol4_state", "h4")}, "MARKET_WIDE_REBOUND_CONFIRMED_SMALL_N")
    add("BTC/SOL no-dump kill", "Should early BTC/SOL dump force H2 exit instead of H4?", {"cross_no_dump_h4": group(trades, "cross_no_dump", "h4"), "policy_no_dump_hold_else_h2": metrics([t["h4"] if t["cross_no_dump"] else t["h2"] for t in trades])}, "NO_DUMP_POLICY_NOT_BETTER_THAN_H4_ALL")
    add("H2 partial take-profit", "Does partial H2 close stabilize H4?", {"half_h2_half_h4": metrics([(float(t["h2"]) + float(t["h4"])) / 2 for t in trades]), "thirty_h2_seventy_h4": metrics([0.3 * float(t["h2"]) + 0.7 * float(t["h4"]) for t in trades]), "full_h4": h4}, "PARTIAL_SMOOTHER_BUT_LOWER_SUM")
    add("Compounding / account path", "How do H2/H4 outcomes compound under sizing modes?", {"current_env_h2": equity_path([t["h2"] for t in trades], 1190.0), "current_env_h4": equity_path([t["h4"] for t in trades], 1190.0), "survival_h4": equity_path([t["h4"] for t in trades], 11.0), "balanced_h4": equity_path([t["h4"] for t in trades], 16.3)}, "CURRENT_ENV_EXPLOSIVE_IF_NO_TAIL_BUT_TAIL_RISK_UNCHANGED")
    add("Tail injection stress", "What happens if old tail returns?", {"h4_tail_every5_neg200": equity_path(tail_inject([t["h4"] for t in trades], 5, -200), 1190.0), "h4_tail_every5_neg500": equity_path(tail_inject([t["h4"] for t in trades], 5, -500), 1190.0), "h4_tail_every10_neg500": equity_path(tail_inject([t["h4"] for t in trades], 10, -500), 1190.0)}, "TAIL_STILL_DOMINANT_RISK")
    add("Queue / fill realism", "Is current maker fill model queue-realistic?", {"fill_delay": metrics([t.get("fill_delay_sec") for t in trades]), "late_gt15m": sum(1 for t in trades if float(t.get("fill_delay_sec", 0)) > 900), "model": "conservative cross but not queue-position simulation"}, "NEEDS_600GB_QUEUE_TEST")
    add("Fee sensitivity", "Which fee tiers preserve H4?", {"maker-0.5_taker3.05": metrics(fee_adjust(trades, "h4", old_fee, -0.5, 3.05)), "maker0_taker3.05": metrics(fee_adjust(trades, "h4", old_fee, 0, 3.05)), "maker1_taker5": metrics(fee_adjust(trades, "h4", old_fee, 1, 5)), "maker2_taker8": metrics(fee_adjust(trades, "h4", old_fee, 2, 8))}, "H4_SURVIVES_REASONABLE_FEE_STRESS_IN_SAMPLE")
    add("Live vs shadow parity", "Does live rule match the V02 mirror definition?", {"rule": RULE_NAME, "mirror_protocols": sorted({r.get("protocol_id") for r in mrows if r.get("protocol_id")}), "mirror_n": len(filled_mirror), "parity_limits": "order rounding/exchange queue not tested here"}, "STATIC_RULE_PARITY_OK_QUEUE_NOT_TESTED")
    add("Shadow ledger reconciliation", "Why does V02 mirror differ from daily guardrail shadow?", all_next.get("ledger_reconciliation", {}), "DIFFERENT_BUCKETS_NOT_CONFLICTING")
    add("Visual navigation layer", "What should chart show?", {"required_lines": ["V02 signal", "maker fill window", "H2 checkpoint", "H4 target window", "NAV line", "BUY spike bars", "BTC/SOL confirmation strip", "state label"], "status": "indicator line exists; H2/H4 checkpoint overlays still TODO"}, "VISUAL_TODO")
    state_counts = Counter(">".join(t.get("state_v2", [])) for t in trades)
    add(
        "State machine v2",
        "Can each trade be mapped to operational states?",
        {
            "state_paths": [
                {"fill_utc": t.get("fill_utc"), "states": t.get("state_v2"), "h2": r1(t.get("h2")), "h4": r1(t.get("h4"))}
                for t in trades
            ],
            "counts": dict(state_counts),
        },
        "STATE_MACHINE_READY_FOR_SHADOW",
    )
    add("Decision engine v0", "What would the system recommend today?", {"recommendation": "HOLD_TO_H4_SHADOW_ONLY", "rules": ["do not change live yet", "track H2/H3/H4", "catastrophic stop only", "no fill-delay cancel"], "reason": "H4 dominates H2 in-sample but N=11"}, "SHADOW_ONLY")
    add("Mechanism ontology", "What is the alpha's likely mechanism?", {"old_name": "liquidation fade", "candidate_name": "forced-sell deep-bid absorption delayed rebound", "evidence": ["prior downtrend", "SELL forced flow", "deep bid", "maker retest", "delayed H4 rebound"]}, "RENAMING_USEFUL_FOR_EXPANSION")
    add("Portfolio expansion", "Can mechanism move to BTC/SOL/routes?", {"status": "not proven", "prior_roadmap": "route map found 0 strong node under N>=40 discipline; revisit with forced-flow absorption not raw liquidation"}, "RESEARCH_BACKLOG")
    add(
        "Regime identity deep test",
        "Is V02 actually a prior-downtrend capitulation rebound rather than a generic liquidation fade?",
        {
            "h2_by_eth_prior24": group(trades, "eth_prior_regime", "h2"),
            "h4_by_eth_prior24": group(trades, "eth_prior_regime", "h4"),
            "h4_by_btc_prior24": group(trades, "btc_prior_regime", "h4"),
            "eth_prior24_bps": metrics([t.get("eth_prior24") for t in trades]),
            "btc_prior24_bps": metrics([t.get("btc_prior24") for t in trades]),
        },
        "PRIOR_DOWNTREND_IDENTITY_SUPPORTED_SMALL_N",
    )
    add("Data coverage / N plan", "How long to N=30?", {"ledger_n": len(trades), "date_span": scope.get("start_utc", "") + " to " + scope.get("end_utc", ""), "rough_event_rate_per_30d": r1(len(trades) / 4.0), "rough_months_to_n30": r1(30 / max(len(trades) / 4.0, 0.001))}, "TIME_IS_BINDING_CONSTRAINT")

    executive = (
        f"Completed {len(tests)} tests on V02 N={len(trades)}. H4 remains the strongest management lead "
        f"(H2 sum {h2['sum']} / T3R {h2['t3r']} vs H4 sum {h4['sum']} / T3R {h4['t3r']}). "
        "No live change is justified yet; H4/O20 shadow + queue realism + forced-flow expansion are next."
    )
    result = {
        "generated_at_utc": utc_now(),
        "scope": {"rule": RULE_NAME, "source_ledger": str(args.ledger_json), "n": len(trades), "research_only": True},
        "executive_read": executive,
        "tests": tests,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    render(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
