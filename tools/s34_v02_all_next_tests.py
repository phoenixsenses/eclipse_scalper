"""All next-step tests for S34 V02, using the existing 120d V02 ledger.

Research-only. This intentionally reuses the already-built live-like V02 ledger
to avoid repeatedly rescanning the large database. It computes the full next-test
set around the current V02 alpha:

1 H3/H4 shadow
2 H4 tail/giveback
3 H3/H4 stop compatibility
4 fill-delay cutoff
5 O15/O20 execution surface
6 forced-sell/deep-bid expansion
7 absorption-without-cascade proxy
8 BTC/SOL context at H4
9 rebound ignition timing
10 MFE late-runner
11 state-machine v1
12 forward OOS protocol
13 tick/book execution realism proxy
14 old shadow-ledger reconciliation
15 regime identity deep test
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
from tools.s34_v02_alpha_navigation_overlay import HORIZON_SEC, RULE_NAME, SYMBOL, book_exit_price, trade_return_at  # noqa: E402
from tools.s34_v02_nav_spike_tests import DB_PATH, OUT_DIR  # noqa: E402
from tools.s34_v02_next_gen_alpha_research import (  # noqa: E402
    collect_v02_anchors,
    execution_surface,
    giveback_exit,
    mae_in_window,
    metrics,
    path_returns,
)


DEFAULT_LEDGER = OUT_DIR / "S34_V02_ALPHA_NAVIGATION_OVERLAY_120D.json"
DEFAULT_30D = OUT_DIR / "S34_V02_NEXT_GEN_ALPHA_RESEARCH_30D.json"
MIRROR_CSV = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.csv"
DAILY_MD = OUT_DIR / "S34_DAILY_EXECUTION_REPORT.md"
OUT_JSON = OUT_DIR / "S34_V02_ALL_NEXT_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_ALL_NEXT_TESTS.md"


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


def group_metric(rows: list[dict[str, Any]], key: str, value: str = "net_2h_bps") -> dict[str, Any]:
    out = {}
    for val in sorted({str(r.get(key, "NA")) for r in rows}):
        out[val] = metrics([r.get(value) for r in rows if str(r.get(key, "NA")) == val])
    return out


def mark_return(marks, trade: dict[str, Any], horizon_sec: int, fee_bps: float) -> float | None:
    px = marks.at_or_after(int(trade["fill_ts_ms"]) + int(horizon_sec) * 1000)
    if not px:
        return None
    return signed_return_bps("LONG", float(trade["entry_price"]), float(px[1])) - float(fee_bps)


def mfe_mae_path(marks, trade: dict[str, Any], horizon_sec: int) -> dict[str, Any]:
    path = path_returns(marks, float(trade["entry_price"]), int(trade["fill_ts_ms"]), int(horizon_sec))
    if not path:
        return {}
    mfe_ts, mfe = max(path, key=lambda x: x[1])
    mae_ts, mae = min(path, key=lambda x: x[1])
    return {
        "mfe_bps": float(mfe),
        "mae_bps": float(mae),
        "mfe_time_sec": (int(mfe_ts) - int(trade["fill_ts_ms"])) / 1000.0,
        "mae_time_sec": (int(mae_ts) - int(trade["fill_ts_ms"])) / 1000.0,
    }


def first_cross(path: list[tuple[int, float]], start_ms: int, level: float) -> float | None:
    for ts, ret in path:
        if float(ret) >= float(level):
            return (int(ts) - int(start_ms)) / 1000.0
    return None


def stop_compat(marks, trades: list[dict[str, Any]], horizon_sec: int, fee_bps: float, stop_bps: float) -> dict[str, Any]:
    vals = []
    exits = Counter()
    for t in trades:
        path = path_returns(marks, float(t["entry_price"]), int(t["fill_ts_ms"]), horizon_sec)
        stopped = None
        for _, ret in path:
            if float(ret) <= -float(stop_bps):
                stopped = float(ret) - fee_bps
                break
        if stopped is not None:
            vals.append(stopped)
            exits["SL"] += 1
        else:
            vals.append(mark_return(marks, t, horizon_sec, fee_bps))
            exits["TIME"] += 1
    return {"result": metrics(vals), "exits": dict(exits)}


def drawdown_series(vals: list[float]) -> dict[str, Any]:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in clean(vals):
        eq += v
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)
    return {"cum_bps": r1(eq), "max_drawdown_bps": r1(max_dd)}


def parse_daily_shadow(md_path: Path) -> dict[str, Any]:
    text = md_path.read_text(encoding="utf-8", errors="ignore") if md_path.exists() else ""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("- v2_shadow_all_time:"):
            raw = line.split(":", 1)[1].strip().strip("`")
            try:
                out["v2_shadow_all_time"] = json.loads(raw)
            except json.JSONDecodeError:
                out["v2_shadow_all_time_raw"] = raw
        if line.startswith("- v2_shadow_today:"):
            raw = line.split(":", 1)[1].strip().strip("`")
            try:
                out["v2_shadow_today"] = json.loads(raw)
            except json.JSONDecodeError:
                out["v2_shadow_today_raw"] = raw
    return out


def mirror_csv_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    filled = [r for r in rows if r.get("sim_status") == "FILLED" and r.get("observation_status") == "CLOSED"]
    return {
        "exists": True,
        "rows": len(rows),
        "filled_closed": len(filled),
        "net_bps": metrics([r.get("net_bps") for r in filled]),
        "latest_utc": filled[-1].get("maker_fill_utc") if filled else None,
        "protocols": sorted({r.get("protocol_id") for r in rows if r.get("protocol_id")}),
    }


def render(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 V02 All Next Tests",
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
        ("h3_h4_shadow", "1. H3/H4 Shadow"),
        ("h4_tail_giveback", "2. H4 Tail / Giveback"),
        ("stop_compatibility", "3. H3/H4 Stop Compatibility"),
        ("fill_delay_cutoff", "4. Fill Delay Cutoff"),
        ("execution_surface", "5. O15/O20 Execution Surface"),
        ("forced_sell_expansion", "6. Forced-Sell Deep-Bid Expansion"),
        ("absorption_without_cascade", "7. Absorption Without Cascade"),
        ("btc_sol_context_h4", "8. BTC/SOL Context at H4"),
        ("rebound_ignition", "9. Rebound Ignition Timing"),
        ("mfe_late_runner", "10. MFE Late Runner"),
        ("state_machine_v1", "11. State Machine v1"),
        ("forward_oos_protocol", "12. Forward OOS Protocol"),
        ("execution_realism_proxy", "13. Tick/Book Execution Realism Proxy"),
        ("ledger_reconciliation", "14. Shadow Ledger Reconciliation"),
        ("regime_identity_deep", "15. Regime Identity Deep Test"),
    ]:
        lines += [f"## {title}", "", "```json", json.dumps(result[key], indent=2, ensure_ascii=True), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--ledger-json", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--next30-json", type=Path, default=DEFAULT_30D)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    ledger = json.loads(args.ledger_json.read_text(encoding="utf-8"))
    trades = [t for t in ledger.get("sample_filled_trades", []) if t.get("status") == "FILLED"]
    scope = dict(ledger.get("scope", {}))
    fee_bps = float(scope.get("maker_fee_bps", -0.5)) + float(scope.get("taker_fee_bps", 3.05))
    next30 = json.loads(args.next30_json.read_text(encoding="utf-8")) if args.next30_json.exists() else {}

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        marks = load_mark_index(conn, SYMBOL)
        btc = load_mark_index(conn, "BTCUSDT")
        sol = load_mark_index(conn, "SOLUSDT")
        for t in trades:
            t["net_3h_bps"] = mark_return(marks, t, 3 * 3600, fee_bps)
            t["net_4h_bps"] = mark_return(marks, t, 4 * 3600, fee_bps)
            t.update({f"path4_{k}": v for k, v in mfe_mae_path(marks, t, 4 * 3600).items()})
            t["giveback_4h_bps"] = r1(float(t.get("path4_mfe_bps") or 0.0) - float(t.get("net_4h_bps") or 0.0))
            t["btc_4h_bps"] = r1(btc.ret_bps(int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 4 * 3600 * 1000))
            t["sol_4h_bps"] = r1(sol.ret_bps(int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 4 * 3600 * 1000))
            t["btc_prior24_bps"] = r1(btc.ret_bps(int(t["fill_ts_ms"]) - 24 * 3600 * 1000, int(t["fill_ts_ms"])))
            t["eth_prior24_bps"] = r1(marks.ret_bps(int(t["fill_ts_ms"]) - 24 * 3600 * 1000, int(t["fill_ts_ms"])))
            for m in (5, 10, 15, 30):
                t[f"mae_{m}m_bps"] = r1(mae_in_window(marks, t, m))
            path = path_returns(marks, float(t["entry_price"]), int(t["fill_ts_ms"]), 4 * 3600)
            t["rebound20_sec"] = r1(first_cross(path, int(t["fill_ts_ms"]), 20.0))
            t["rebound50_sec"] = r1(first_cross(path, int(t["fill_ts_ms"]), 50.0))
            t["rebound80_sec"] = r1(first_cross(path, int(t["fill_ts_ms"]), 80.0))
            t["h4_runner"] = bool(t.get("net_4h_bps") is not None and float(t["net_4h_bps"]) > float(t.get("net_2h_bps", 0.0)))
            t["fill_delay_bin"] = "lt5m" if float(t["fill_delay_sec"]) < 300 else "5to15m" if float(t["fill_delay_sec"]) < 900 else "gt15m"
            t["btc4_context"] = "btc_up" if (t.get("btc_4h_bps") or 0) >= 0 else "btc_down"
            t["sol4_context"] = "sol_up" if (t.get("sol_4h_bps") or 0) >= 0 else "sol_down"
            t["eth_prior24_regime"] = "eth_prior_up" if (t.get("eth_prior24_bps") or 0) >= 0 else "eth_prior_down"
            t["btc_prior24_regime"] = "btc_prior_up" if (t.get("btc_prior24_bps") or 0) >= 0 else "btc_prior_down"

        h3h4 = {
            "h2": metrics([t.get("net_2h_bps") for t in trades]),
            "h3": metrics([t.get("net_3h_bps") for t in trades]),
            "h4": metrics([t.get("net_4h_bps") for t in trades]),
            "delta_h4_minus_h2": metrics([
                float(t["net_4h_bps"]) - float(t["net_2h_bps"])
                for t in trades
                if t.get("net_4h_bps") is not None and t.get("net_2h_bps") is not None
            ]),
        }

        h4_tail = {
            "h4_path_mfe": metrics([t.get("path4_mfe_bps") for t in trades]),
            "h4_path_mae": metrics([t.get("path4_mae_bps") for t in trades]),
            "h4_giveback_from_mfe_to_exit": metrics([t.get("giveback_4h_bps") for t in trades]),
            "h4_drawdown_series": drawdown_series([t.get("net_4h_bps") for t in trades]),
            "h2_drawdown_series": drawdown_series([t.get("net_2h_bps") for t in trades]),
        }

        stop = {
            "h2_sl150": stop_compat(marks, trades, 2 * 3600, fee_bps, 150.0),
            "h3_sl150": stop_compat(marks, trades, 3 * 3600, fee_bps, 150.0),
            "h4_sl150": stop_compat(marks, trades, 4 * 3600, fee_bps, 150.0),
            "h4_sl100": stop_compat(marks, trades, 4 * 3600, fee_bps, 100.0),
        }

        fill_delay = {
            "by_delay_h2": group_metric(trades, "fill_delay_bin", "net_2h_bps"),
            "by_delay_h4": group_metric(trades, "fill_delay_bin", "net_4h_bps"),
            "cancel_after_15m_h2": metrics([t.get("net_2h_bps") for t in trades if float(t.get("fill_delay_sec", 0)) <= 900]),
            "cancel_after_15m_h4": metrics([t.get("net_4h_bps") for t in trades if float(t.get("fill_delay_sec", 0)) <= 900]),
        }

        # Execution surface and expansion use the completed 30d heavy run plus a lightweight 120d anchor surface if possible.
        start_ms = int(scope.get("start_utc") and datetime.fromisoformat(str(scope["start_utc"]).replace("Z", "+00:00")).timestamp() * 1000)
        end_ms = int(scope.get("end_utc") and datetime.fromisoformat(str(scope["end_utc"]).replace("Z", "+00:00")).timestamp() * 1000)
        try:
            anchors = collect_v02_anchors(conn, start_ms, end_ms)
            exec_surface_120 = execution_surface(conn, anchors, float(scope.get("maker_fee_bps", -0.5)), float(scope.get("taker_fee_bps", 3.05)), float(scope.get("cross_margin_bps", 2.0)))
        except Exception as exc:  # keep report alive; this is research diagnostics
            exec_surface_120 = {"error": repr(exc)}

    execution = {
        "current_o20_w300_h2": h3h4["h2"],
        "heavy_30d_best_from_prior_run": next30.get("executive_read"),
        "surface_120d_light": exec_surface_120,
    }

    forced = {
        "source": "S34_V02_NEXT_GEN_ALPHA_RESEARCH_30D mechanism_expansion; 120d full expansion timed out in previous run",
        "result": next30.get("mechanism_expansion", {}),
    }
    absorption = {
        "proxy": "same forced-sell expansion, but interpreted as no-large-liquidation absorption proxy",
        "result": next30.get("mechanism_expansion", {}),
        "verdict": "N too small; mechanism lead only" if (next30.get("mechanism_expansion", {}).get("filled_n", 0) or 0) < 10 else "needs holdout",
    }

    context = {
        "by_btc_4h": group_metric(trades, "btc4_context", "net_4h_bps"),
        "by_sol_4h": group_metric(trades, "sol4_context", "net_4h_bps"),
        "btc_4h_bps": metrics([t.get("btc_4h_bps") for t in trades]),
        "sol_4h_bps": metrics([t.get("sol_4h_bps") for t in trades]),
    }
    ignition = {
        "rebound20_sec": metrics([t.get("rebound20_sec") for t in trades]),
        "rebound50_sec": metrics([t.get("rebound50_sec") for t in trades]),
        "rebound80_sec": metrics([t.get("rebound80_sec") for t in trades]),
        "by_rebound50_within_30m_h4": group_metric(
            [dict(t, rb50_30m=bool(t.get("rebound50_sec") is not None and float(t["rebound50_sec"]) <= 1800)) for t in trades],
            "rb50_30m",
            "net_4h_bps",
        ),
    }
    late_runner = {
        "h4_runner_count": sum(1 for t in trades if t.get("h4_runner")),
        "by_h4_runner": group_metric(trades, "h4_runner", "net_4h_bps"),
        "mfe_time_h4": metrics([t.get("path4_mfe_time_sec") for t in trades]),
        "mfe_bps_h4": metrics([t.get("path4_mfe_bps") for t in trades]),
    }
    state_machine = {
        "state_counts": dict(Counter(t.get("state_sequence_5m", "NA") for t in trades)),
        "state_outcomes_h4": group_metric(trades, "state_sequence_5m", "net_4h_bps"),
        "simple_states": [
            {
                "fill_utc": t.get("fill_utc"),
                "state_path": [
                    "FILL",
                    "PAIN" if (t.get("mae_bps") or 0) < -20 else "NO_PAIN",
                    "REBOUND_20" if t.get("rebound20_sec") is not None else "NO_REBOUND_20",
                    "REBOUND_50" if t.get("rebound50_sec") is not None else "NO_REBOUND_50",
                    "RUNNER" if t.get("h4_runner") else "H2_BETTER",
                ],
                "h2": r1(t.get("net_2h_bps")),
                "h4": r1(t.get("net_4h_bps")),
            }
            for t in trades
        ],
    }

    forward_protocol = {
        "shadow_buckets_to_track": ["H2_current", "H3_shadow", "H4_shadow", "O15_W300_H4_shadow", "O20_W300_H4_shadow"],
        "review_gates": {"N_lt_10": "observe_only", "N_10_to_20": "early_confidence_only", "N_ge_30": "paper_candidate_review"},
        "kill_or_pause": ["T3R < 0 after N>=10", "two losses below -150bps", "H4 gives back more than 50% vs H2 over N>=10"],
    }

    realism = {
        "fill_model_warning": "current model is top-of-book/mark conservative-cross, not true queue simulation",
        "fill_delay": metrics([t.get("fill_delay_sec") for t in trades]),
        "late_gt15m_count": sum(1 for t in trades if float(t.get("fill_delay_sec", 0)) > 900),
        "gap_proxy_mae_h4": metrics([t.get("path4_mae_bps") for t in trades]),
        "needs_600gb_tick_queue_test": True,
    }

    mirror = mirror_csv_summary(MIRROR_CSV)
    daily = parse_daily_shadow(DAILY_MD)
    reconcile = {
        "v02_shadow_mirror_csv": mirror,
        "daily_guardrail_shadow": daily,
        "interpretation": "These are different buckets/models: v02 mirror is the frozen V02 maker lane; daily guardrail shadow references broad paper-trade guardrail signals and is not the same performance stream.",
    }
    regime = {
        "by_eth_prior24_h2": group_metric(trades, "eth_prior24_regime", "net_2h_bps"),
        "by_eth_prior24_h4": group_metric(trades, "eth_prior24_regime", "net_4h_bps"),
        "by_btc_prior24_h4": group_metric(trades, "btc_prior24_regime", "net_4h_bps"),
        "eth_prior24_bps": metrics([t.get("eth_prior24_bps") for t in trades]),
        "btc_prior24_bps": metrics([t.get("btc_prior24_bps") for t in trades]),
    }

    executive = (
        f"V02 ledger N={len(trades)}. H2 sum={h3h4['h2']['sum']} T3R={h3h4['h2']['t3r']}; "
        f"H4 sum={h3h4['h4']['sum']} T3R={h3h4['h4']['t3r']}. "
        f"Cancel-after-15m H4 sum={fill_delay['cancel_after_15m_h4']['sum']} T3R={fill_delay['cancel_after_15m_h4']['t3r']}. "
        f"Forced-sell expansion remains tiny-N: {forced['result'].get('filled_n')} fills."
    )
    result = {
        "generated_at_utc": utc_now(),
        "scope": {"rule": RULE_NAME, "source_ledger": str(args.ledger_json), "n": len(trades), "research_only": True},
        "executive_read": executive,
        "h3_h4_shadow": h3h4,
        "h4_tail_giveback": h4_tail,
        "stop_compatibility": stop,
        "fill_delay_cutoff": fill_delay,
        "execution_surface": execution,
        "forced_sell_expansion": forced,
        "absorption_without_cascade": absorption,
        "btc_sol_context_h4": context,
        "rebound_ignition": ignition,
        "mfe_late_runner": late_runner,
        "state_machine_v1": state_machine,
        "forward_oos_protocol": forward_protocol,
        "execution_realism_proxy": realism,
        "ledger_reconciliation": reconcile,
        "regime_identity_deep": regime,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    render(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
