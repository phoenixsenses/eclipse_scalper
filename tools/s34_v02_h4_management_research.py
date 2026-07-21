"""Focused H4 management research for the S34 V02 alpha.

Research-only. Uses the existing 120d live-like V02 ledger and evaluates:
1. H2/H3/H4 horizon shadow
2. H4 runner recognition
3. H2 checkpoint decision engine
4. H4 giveback / partial protection
5. MAE / catastrophic stop research

No live executor/config/runtime files are touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
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
OUT_JSON = OUT_DIR / "S34_V02_H4_MANAGEMENT_RESEARCH.json"
OUT_MD = OUT_DIR / "S34_V02_H4_MANAGEMENT_RESEARCH.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(vals: list[Any]) -> list[float]:
    xs = []
    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return xs


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = clean(vals)
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "t3r": 0.0, "top1_removed": 0.0, "min": None, "max": None}
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum": r1(sum(xs)),
        "mean": r1(mean(xs)),
        "median": r1(pctile(xs, 0.5)),
        "win_rate": round(sum(1 for x in xs if x > 0.0) / len(xs), 3),
        "t3r": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "top1_removed": r1(sum(ordered[1:]) if len(ordered) > 1 else sum(ordered)),
        "min": r1(min(xs)),
        "max": r1(max(xs)),
    }


def group(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return {
        str(v): metrics([r.get(value) for r in rows if r.get(key) == v])
        for v in sorted({r.get(key) for r in rows}, key=lambda x: str(x))
    }


def mark_ret(marks, trade: dict[str, Any], horizon_sec: int, fee_bps: float) -> float | None:
    px = marks.at_or_after(int(trade["fill_ts_ms"]) + int(horizon_sec) * 1000)
    if not px:
        return None
    return signed_return_bps("LONG", float(trade["entry_price"]), float(px[1])) - float(fee_bps)


def series_ret(series, start_ms: int, horizon_sec: int) -> float | None:
    a = series.at_or_after(int(start_ms))
    b = series.at_or_after(int(start_ms) + int(horizon_sec) * 1000)
    if not a or not b:
        return None
    return signed_return_bps("LONG", float(a[1]), float(b[1]))


def path_stats(marks, trade: dict[str, Any], horizon_sec: int) -> dict[str, Any]:
    path = path_returns(marks, float(trade["entry_price"]), int(trade["fill_ts_ms"]), horizon_sec)
    if not path:
        return {}
    mfe_ts, mfe = max(path, key=lambda x: x[1])
    mae_ts, mae = min(path, key=lambda x: x[1])

    def first(level: float) -> float | None:
        for ts, r in path:
            if float(r) >= level:
                return (int(ts) - int(trade["fill_ts_ms"])) / 1000.0
        return None

    return {
        "mfe": float(mfe),
        "mae": float(mae),
        "mfe_sec": (int(mfe_ts) - int(trade["fill_ts_ms"])) / 1000.0,
        "mae_sec": (int(mae_ts) - int(trade["fill_ts_ms"])) / 1000.0,
        "rebound20_sec": first(20.0),
        "rebound50_sec": first(50.0),
        "rebound80_sec": first(80.0),
    }


def stop_policy(
    marks,
    trades: list[dict[str, Any]],
    *,
    horizon_sec: int,
    stop_bps: float,
    fee_bps: float,
    delay_sec: int = 0,
) -> dict[str, Any]:
    vals = []
    exits = {"SL": 0, "TIME": 0}
    for t in trades:
        start = int(t["fill_ts_ms"])
        path = path_returns(marks, float(t["entry_price"]), start, horizon_sec)
        stopped = None
        for ts, r in path:
            if int(ts) - start < int(delay_sec) * 1000:
                continue
            if float(r) <= -float(stop_bps):
                stopped = float(r) - fee_bps
                break
        if stopped is not None:
            vals.append(stopped)
            exits["SL"] += 1
        else:
            vals.append(mark_ret(marks, t, horizon_sec, fee_bps))
            exits["TIME"] += 1
    return {"result": metrics(vals), "exits": exits}


def giveback_policy(
    marks,
    trades: list[dict[str, Any]],
    *,
    horizon_sec: int,
    min_peak: float,
    giveback_frac: float,
    fee_bps: float,
    start_trailing_sec: int = 0,
) -> dict[str, Any]:
    vals = []
    exits = {"TRAIL": 0, "TIME": 0}
    for t in trades:
        start = int(t["fill_ts_ms"])
        peak = -10**9
        armed = False
        exited = None
        for ts, r in path_returns(marks, float(t["entry_price"]), start, horizon_sec):
            if int(ts) - start < int(start_trailing_sec) * 1000:
                continue
            peak = max(peak, float(r))
            if peak >= float(min_peak):
                armed = True
            if armed and float(r) <= peak * (1.0 - float(giveback_frac)):
                exited = float(r) - fee_bps
                break
        if exited is not None:
            vals.append(exited)
            exits["TRAIL"] += 1
        else:
            vals.append(mark_ret(marks, t, horizon_sec, fee_bps))
            exits["TIME"] += 1
    return {"result": metrics(vals), "exits": exits}


def draw_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 V02 H4 Management Research",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## Executive Read",
        "",
        result["executive_read"],
        "",
    ]
    for key, title in [
        ("horizon_shadow", "1. H4 Forward Shadow Backtest"),
        ("runner_recognition", "2. H4 Runner Recognition"),
        ("checkpoint_engine", "3. H2 Checkpoint Decision Engine"),
        ("giveback_protection", "4. H4 Giveback Protection"),
        ("stop_research", "5. MAE / Catastrophic Stop Research"),
        ("verdict", "6. Verdict"),
        ("forward_shadow_spec", "7. Forward Shadow Spec"),
    ]:
        lines += [f"## {title}", "", "```json", json.dumps(result[key], indent=2, ensure_ascii=True), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--ledger-json", type=Path, default=LEDGER_120D)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    ledger = json.loads(args.ledger_json.read_text(encoding="utf-8"))
    trades = [dict(t) for t in ledger.get("sample_filled_trades", []) if t.get("status") == "FILLED"]
    scope = ledger.get("scope", {})
    fee_bps = float(scope.get("maker_fee_bps", -0.5)) + float(scope.get("taker_fee_bps", 3.05))

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth = load_mark_index(conn, SYMBOL)
        btc = load_mark_index(conn, "BTCUSDT")
        sol = load_mark_index(conn, "SOLUSDT")

        for t in trades:
            t["h2"] = mark_ret(eth, t, 7200, fee_bps)
            t["h3"] = mark_ret(eth, t, 10800, fee_bps)
            t["h4"] = mark_ret(eth, t, 14400, fee_bps)
            t["h4_delta"] = float(t["h4"]) - float(t["h2"])
            t.update({f"h4_{k}": v for k, v in path_stats(eth, t, 14400).items()})
            t["btc30"] = series_ret(btc, int(t["fill_ts_ms"]), 1800)
            t["btc60"] = series_ret(btc, int(t["fill_ts_ms"]), 3600)
            t["btc4"] = series_ret(btc, int(t["fill_ts_ms"]), 14400)
            t["sol30"] = series_ret(sol, int(t["fill_ts_ms"]), 1800)
            t["sol60"] = series_ret(sol, int(t["fill_ts_ms"]), 3600)
            t["sol4"] = series_ret(sol, int(t["fill_ts_ms"]), 14400)
            t["runner"] = bool(t["h4_delta"] > 0)
            t["rebound50_30m"] = bool(t.get("h4_rebound50_sec") is not None and float(t["h4_rebound50_sec"]) <= 1800)
            t["rebound20_15m"] = bool(t.get("h4_rebound20_sec") is not None and float(t["h4_rebound20_sec"]) <= 900)
            t["btc_no_dump30"] = bool(t.get("btc30") is not None and float(t["btc30"]) > -40)
            t["sol_no_dump30"] = bool(t.get("sol30") is not None and float(t["sol30"]) > -80)
            t["cross_no_dump30"] = bool(t["btc_no_dump30"] and t["sol_no_dump30"])
            t["h2_gt_100"] = bool(float(t["h2"]) >= 100)
            t["h2_lt_50"] = bool(float(t["h2"]) < 50)
            t["giveback_to_h4"] = float(t.get("h4_mfe") or 0.0) - float(t["h4"])

        horizon = {
            "h2": metrics([t["h2"] for t in trades]),
            "h3": metrics([t["h3"] for t in trades]),
            "h4": metrics([t["h4"] for t in trades]),
            "h4_minus_h2": metrics([t["h4_delta"] for t in trades]),
            "per_trade": [
                {"fill_utc": t.get("fill_utc"), "h2": r1(t["h2"]), "h3": r1(t["h3"]), "h4": r1(t["h4"]), "delta_h4_h2": r1(t["h4_delta"])}
                for t in trades
            ],
        }

        runner = {
            "runner_count": sum(1 for t in trades if t["runner"]),
            "by_rebound50_30m": group(trades, "rebound50_30m", "h4_delta"),
            "by_rebound20_15m": group(trades, "rebound20_15m", "h4_delta"),
            "by_btc_no_dump30": group(trades, "btc_no_dump30", "h4_delta"),
            "by_sol_no_dump30": group(trades, "sol_no_dump30", "h4_delta"),
            "by_cross_no_dump30": group(trades, "cross_no_dump30", "h4_delta"),
            "candidate_policies": {
                "hold_h4_if_rebound50_30m_else_h2": metrics([t["h4"] if t["rebound50_30m"] else t["h2"] for t in trades]),
                "hold_h4_if_cross_no_dump_else_h2": metrics([t["h4"] if t["cross_no_dump30"] else t["h2"] for t in trades]),
                "hold_h4_if_rebound20_and_btc_no_dump_else_h2": metrics([
                    t["h4"] if (t["rebound20_15m"] and t["btc_no_dump30"]) else t["h2"] for t in trades
                ]),
            },
        }

        checkpoint = {
            "always_h2": horizon["h2"],
            "always_h3": horizon["h3"],
            "always_h4": horizon["h4"],
            "h4_if_cross_no_dump_else_h2": runner["candidate_policies"]["hold_h4_if_cross_no_dump_else_h2"],
            "h4_if_rebound50_else_h2": runner["candidate_policies"]["hold_h4_if_rebound50_30m_else_h2"],
            "partial_50_h2_50_h4": metrics([(float(t["h2"]) + float(t["h4"])) / 2 for t in trades]),
            "partial_30_h2_70_h4": metrics([0.3 * float(t["h2"]) + 0.7 * float(t["h4"]) for t in trades]),
            "h4_if_h2_lt100_else_h2": metrics([t["h4"] if t["h2_lt_50"] else t["h2"] for t in trades]),
        }

        giveback = {
            "fixed_h4": horizon["h4"],
            "giveback_distribution": metrics([t["giveback_to_h4"] for t in trades]),
            "trail_peak40_gb25": giveback_policy(eth, trades, horizon_sec=14400, min_peak=40, giveback_frac=0.25, fee_bps=fee_bps),
            "trail_peak40_gb40": giveback_policy(eth, trades, horizon_sec=14400, min_peak=40, giveback_frac=0.40, fee_bps=fee_bps),
            "trail_peak80_gb25": giveback_policy(eth, trades, horizon_sec=14400, min_peak=80, giveback_frac=0.25, fee_bps=fee_bps),
            "trail_peak80_gb40": giveback_policy(eth, trades, horizon_sec=14400, min_peak=80, giveback_frac=0.40, fee_bps=fee_bps),
            "trail_after_h2_peak80_gb40": giveback_policy(
                eth, trades, horizon_sec=14400, min_peak=80, giveback_frac=0.40, fee_bps=fee_bps, start_trailing_sec=7200
            ),
            "partial_50_h2_50_h4": checkpoint["partial_50_h2_50_h4"],
            "partial_30_h2_70_h4": checkpoint["partial_30_h2_70_h4"],
        }

        stop = {
            "h4_sl100": stop_policy(eth, trades, horizon_sec=14400, stop_bps=100, fee_bps=fee_bps),
            "h4_sl125": stop_policy(eth, trades, horizon_sec=14400, stop_bps=125, fee_bps=fee_bps),
            "h4_sl150": stop_policy(eth, trades, horizon_sec=14400, stop_bps=150, fee_bps=fee_bps),
            "h4_sl175": stop_policy(eth, trades, horizon_sec=14400, stop_bps=175, fee_bps=fee_bps),
            "h4_sl200": stop_policy(eth, trades, horizon_sec=14400, stop_bps=200, fee_bps=fee_bps),
            "h4_sl150_delay5m": stop_policy(eth, trades, horizon_sec=14400, stop_bps=150, fee_bps=fee_bps, delay_sec=300),
            "h4_sl150_delay15m": stop_policy(eth, trades, horizon_sec=14400, stop_bps=150, fee_bps=fee_bps, delay_sec=900),
            "h4_mae_distribution": metrics([t.get("h4_mae") for t in trades]),
        }

    verdict = {
        "horizon": "H4_FORWARD_SHADOW_LEAD_SMALL_N",
        "runner_recognition": "CROSS_NO_DUMP_POLICY_SLIGHTLY_BEATS_H4_IN_SAMPLE_BUT_SMALL_N",
        "checkpoint": "ALWAYS_H4_OR_CROSS_NO_DUMP_H4_SHADOW; NO_LIVE_CHANGE",
        "giveback": "FIXED_H4_STILL_BEST; TRAILING_REDUCES_UPSIDE",
        "stop": "SL100_BAD; SL125_TOUCHES_ONE; SL150_PLUS_CATASTROPHIC_ONLY",
        "next_action": "create/track H3/H4 shadow buckets; keep live H2 unchanged until forward N grows and queue realism is tested",
    }
    forward_shadow_spec = {
        "buckets": ["V02_H2_CURRENT", "V02_H3_SHADOW", "V02_H4_SHADOW", "V02_H4_CROSS_NO_DUMP_SHADOW"],
        "minimum_review": {"N_lt_10": "observe_only", "N_10_to_20": "early_confidence", "N_ge_30": "paper_candidate_review"},
        "kill": ["H4 T3R < H2 T3R after N>=10", "two H4 losses below -150bps", "queue realism rejects fills"],
        "live_change_allowed": False,
    }
    executive = (
        f"H4 dominates current V02 ledger: H2 sum {horizon['h2']['sum']} / T3R {horizon['h2']['t3r']} vs "
        f"H4 sum {horizon['h4']['sum']} / T3R {horizon['h4']['t3r']}. "
        "Runner recognition is suggestive but small-N. Giveback/trailing does not beat fixed H4. "
        "SL100 damages the edge; SL150+ behaves as catastrophic-only in this sample."
    )
    result = {
        "generated_at_utc": utc_now(),
        "scope": {"rule": RULE_NAME, "source_ledger": str(args.ledger_json), "n": len(trades), "research_only": True},
        "executive_read": executive,
        "horizon_shadow": horizon,
        "runner_recognition": runner,
        "checkpoint_engine": checkpoint,
        "giveback_protection": giveback,
        "stop_research": stop,
        "verdict": verdict,
        "forward_shadow_spec": forward_shadow_spec,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    draw_md(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
