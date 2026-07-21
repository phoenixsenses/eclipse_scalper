"""Repair research for the rejected ETH BUY-spike scalp.

Research-only. This script tests the next plausible fixes after the live
gauntlet rejected the simple BUY-spike LONG:
1. pre-spike build-up features
2. acceleration anchors
3. passive pullback entries
4. second-leg confirmation
5. execution/filter gates
6. swing marker use
7. extreme spike fade
8. cross-asset lead
9. NAV transitions
10. state labels as navigation
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_buy_spike_live_gauntlet import (
    MarkSeries,
    executable_ret,
    fee_net,
    load_buy_liqs,
    load_mark_raw,
    running_cross_events,
    rolling_sums_at_liqs,
)
from tools.s34_v02_nav_spike_tests import (
    DB_PATH,
    MINUTE,
    OUT_DIR,
    bps,
    build_nav,
    latest_ts,
    load_book_for_buckets,
    load_flow_1m,
    load_liq_1m,
    load_mark_1m,
    nonoverlap,
    pct,
    spike_thresholds,
    summary,
)

OUT_JSON = OUT_DIR / "S34_BUY_SPIKE_REPAIR_TESTS.json"
OUT_MD = OUT_DIR / "S34_BUY_SPIKE_REPAIR_TESTS.md"


def utc_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def load_liqs(conn: sqlite3.Connection, start_ms: int, end_ms: int, symbol: str, side: str) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms",
        (symbol, side.upper(), int(start_ms), int(end_ms)),
    ).fetchall()
    return [(int(r[0]), float(r[1] or 0.0)) for r in rows]


def sum_liq(liqs: list[tuple[int, float]], start: int, end: int) -> float:
    # Small raw lists per symbol/side; linear scan is acceptable for the event counts here.
    return sum(n for ts, n in liqs if start <= ts < end)


def book_at(conn: sqlite3.Connection, ts_ms: int) -> dict[str, float] | None:
    row = conn.execute(
        """
        SELECT bid_price, bid_qty, ask_price, ask_qty, spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker
        WHERE symbol='ETHUSDT' AND ts_ms<=? AND ts_ms>=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (int(ts_ms), int(ts_ms) - 10_000),
    ).fetchone()
    if not row:
        return None
    bid = float(row[0] or 0.0)
    bid_qty = float(row[1] or 0.0)
    ask = float(row[2] or 0.0)
    ask_qty = float(row[3] or 0.0)
    return {
        "bid": bid,
        "ask": ask,
        "bid_depth_usd": float(row[6]) if row[6] is not None else bid * bid_qty,
        "ask_depth_usd": ask * ask_qty,
        "spread_bps": float(row[4] or 0.0) * 10_000.0,
        "book_imbalance": float(row[5] or 0.0),
    }


def taker_flow(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> float:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),0.0),
          COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),0.0)
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?
        """,
        (int(start_ms), int(end_ms)),
    ).fetchone()
    buy = float(row[0] or 0.0)
    sell = float(row[1] or 0.0)
    total = buy + sell
    return ((buy - sell) / total) if total > 0 else 0.0


def mark_minmax(mark: MarkSeries, start_ms: int, end_ms: int) -> tuple[float | None, float | None]:
    a = bisect.bisect_left(mark.ts, int(start_ms))
    b = bisect.bisect_left(mark.ts, int(end_ms))
    vals = mark.px[a:b]
    if not vals:
        return None, None
    return min(vals), max(vals)


def ret_mark(mark: MarkSeries, ts_ms: int, horizon_min: int) -> float | None:
    return mark.ret(ts_ms, horizon_min * MINUTE)


def ret_short_mark(mark: MarkSeries, ts_ms: int, horizon_min: int) -> float | None:
    r = ret_mark(mark, ts_ms, horizon_min)
    return -r if r is not None else None


def rolling_accel_events(
    liqs: list[tuple[int, float]],
    *,
    threshold_ratio: float,
    min_5s_notional: float,
    cooldown_ms: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last = -10**30
    q5: deque[tuple[int, float]] = deque()
    q30: deque[tuple[int, float]] = deque()
    acc5 = 0.0
    acc30 = 0.0
    prev_above = False
    for ts, n in liqs:
        q5.append((ts, n))
        q30.append((ts, n))
        acc5 += n
        acc30 += n
        while q5 and q5[0][0] < ts - 5_000:
            acc5 -= q5.popleft()[1]
        while q30 and q30[0][0] < ts - 30_000:
            acc30 -= q30.popleft()[1]
        base_rate = max(1.0, (acc30 - acc5) / 25.0)
        ratio = (acc5 / 5.0) / base_rate
        above = ratio >= threshold_ratio and acc5 >= min_5s_notional
        if above and not prev_above and ts - last >= cooldown_ms:
            out.append({"ts": ts, "utc": utc_ms(ts), "ratio": round(ratio, 2), "notional_5s": round(acc5, 1)})
            last = ts
        prev_above = above
    return out


def event_summary(mark: MarkSeries, events: list[dict[str, Any]], horizon_min: int, fee_bps: float = 6.1) -> dict[str, Any]:
    vals = [ret_mark(mark, int(e["ts"]), horizon_min) for e in events]
    return summary(fee_net([v for v in vals if v is not None], fee_bps))


def executable_summary(conn: sqlite3.Connection, events: list[dict[str, Any]], horizon_sec: int, fee_bps: float = 6.1) -> dict[str, Any]:
    vals = [executable_ret(conn, int(e["ts"]), delay_sec=0, horizon_sec=horizon_sec) for e in events]
    return summary(fee_net([v for v in vals if v is not None], fee_bps))


def passive_pullback(mark: MarkSeries, events: list[dict[str, Any]], offset_bps: float, wait_min: int, hold_min: int, fee_bps: float) -> dict[str, Any]:
    vals = []
    nofill_cf = []
    fill_count = 0
    for e in events:
        t = int(e["ts"])
        entry_ref = mark.at_or_after(t)
        if not entry_ref:
            continue
        limit_px = entry_ref * (1 - offset_bps / 10_000.0)
        lo, _ = mark_minmax(mark, t, t + wait_min * MINUTE)
        if lo is not None and lo <= limit_px:
            fill_count += 1
            exit_px = mark.at_or_after(t + wait_min * MINUTE + hold_min * MINUTE)
            if exit_px:
                vals.append(bps(limit_px, exit_px))
        else:
            r = ret_mark(mark, t, hold_min)
            if r is not None:
                nofill_cf.append(r)
    return {
        "fills": fill_count,
        "fill_rate": round(fill_count / len(events), 3) if events else None,
        "filled_fee_net": summary(fee_net(vals, fee_bps)),
        "nofill_counterfactual": summary(fee_net(nofill_cf, fee_bps)),
    }


def split_folds(events: list[dict[str, Any]], folds: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda e: int(e["ts"]))
    return [ordered[int(i * len(ordered) / folds) : int((i + 1) * len(ordered) / folds)] for i in range(folds)]


def render(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 BUY Spike Repair Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## 1. Pre-Spike Build-Up",
        "",
        json.dumps(result["pre_spike_build_up"], indent=2),
        "",
        "## 2. Acceleration Anchor",
        "",
        json.dumps(result["acceleration_anchor"], indent=2),
        "",
        "## 3. Passive Pullback Entry",
        "",
        json.dumps(result["passive_pullback"], indent=2),
        "",
        "## 4. Second Leg Confirmation",
        "",
        json.dumps(result["second_leg_confirmation"], indent=2),
        "",
        "## 5. Execution / Fill Filters",
        "",
        json.dumps(result["execution_filters"], indent=2),
        "",
        "## 6. Swing Marker",
        "",
        json.dumps(result["swing_marker"], indent=2),
        "",
        "## 7. Extreme Spike Fade",
        "",
        json.dumps(result["extreme_fade"], indent=2),
        "",
        "## 8. Cross-Asset Lead",
        "",
        json.dumps(result["cross_asset_lead"], indent=2),
        "",
        "## 9. NAV Transitions",
        "",
        json.dumps(result["nav_transitions"], indent=2),
        "",
        "## 10. State Labels",
        "",
        json.dumps(result["state_labels"], indent=2),
        "",
        "## Verdict",
        "",
        result["verdict"],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    random.seed(3402)
    with sqlite3.connect(args.db) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE
        eth_mark = load_mark_raw(conn, start_ms, end_ms + 8 * 60 * MINUTE, "ETHUSDT")
        btc_mark = load_mark_raw(conn, start_ms, end_ms + 8 * 60 * MINUTE, "BTCUSDT")
        eth_mark_1m = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")
        btc_mark_1m = load_mark_1m(conn, start_ms, end_ms, "BTCUSDT")
        sol_mark_1m = load_mark_1m(conn, start_ms, end_ms, "SOLUSDT")
        liq_1m = load_liq_1m(conn, start_ms, end_ms)
        thresholds = spike_thresholds(liq_1m)
        buckets = sorted(eth_mark_1m)
        book_1m = load_book_for_buckets(conn, buckets)
        flow_1m = load_flow_1m(conn, start_ms, end_ms)
        nav = build_nav(sorted(set(buckets) & set(book_1m)), book_1m, liq_1m, flow_1m, btc_mark_1m)
        nav_by_ts = {int(n["ts"]): n for n in nav}

        eth_buy_liqs = load_buy_liqs(conn, start_ms, end_ms)
        eth_sell_liqs = load_liqs(conn, start_ms, end_ms, "ETHUSDT", "SELL")
        btc_buy_liqs = load_liqs(conn, start_ms, end_ms, "BTCUSDT", "BUY")
        sol_buy_liqs = load_liqs(conn, start_ms, end_ms, "SOLUSDT", "BUY")

        # Closed 1m BUY spike set: the visually strong but rejected anchor.
        closed_spikes = []
        for ts in buckets:
            n = float(liq_1m.get(ts, {}).get("BUY", 0.0))
            if n >= thresholds["BUY"]["primary_threshold"]:
                closed_spikes.append({"ts": ts, "utc": utc_ms(ts), "notional": round(n, 1)})
        closed_spikes = nonoverlap(closed_spikes, 5 * MINUTE)

        sums60 = rolling_sums_at_liqs(eth_buy_liqs, 60_000)
        run60_threshold = max(50_000.0, pct(sums60, 0.95))
        running_events = running_cross_events(eth_buy_liqs, window_ms=60_000, threshold=run60_threshold, cooldown_ms=5 * MINUTE)

        # 1. Pre-spike build-up vs sampled controls.
        control_pool = [ts for ts in buckets if all(abs(ts - int(e["ts"])) > 15 * MINUTE for e in closed_spikes)]
        controls = [{"ts": ts} for ts in random.sample(control_pool, min(len(control_pool), len(closed_spikes) * 3))]

        def feat_rows(events: list[dict[str, Any]]) -> list[dict[str, float]]:
            rows = []
            for e in events:
                t = int(e["ts"])
                bk = book_at(conn, t - 5_000) or {}
                rows.append(
                    {
                        "buy_liq_pre60": sum_liq(eth_buy_liqs, t - 60_000, t),
                        "buy_liq_pre30": sum_liq(eth_buy_liqs, t - 30_000, t),
                        "buy_liq_pre10": sum_liq(eth_buy_liqs, t - 10_000, t),
                        "sell_liq_pre60": sum_liq(eth_sell_liqs, t - 60_000, t),
                        "flow_pre60": taker_flow(conn, t - 60_000, t),
                        "spread_bps": float(bk.get("spread_bps", 0.0)),
                        "bid_depth_usd": float(bk.get("bid_depth_usd", 0.0)),
                        "book_imbalance": float(bk.get("book_imbalance", 0.0)),
                    }
                )
            return rows

        spike_feats = feat_rows(closed_spikes)
        ctrl_feats = feat_rows(controls)

        def avg(rows: list[dict[str, float]], key: str) -> float | None:
            vals = [r[key] for r in rows if math.isfinite(r[key])]
            return round(sum(vals) / len(vals), 4) if vals else None

        pre_build = {
            k: {"spike_avg": avg(spike_feats, k), "control_avg": avg(ctrl_feats, k)}
            for k in ["buy_liq_pre60", "buy_liq_pre30", "buy_liq_pre10", "sell_liq_pre60", "flow_pre60", "spread_bps", "bid_depth_usd", "book_imbalance"]
        }

        # 2. Acceleration anchors.
        accel_results = {}
        for ratio in [2.0, 3.0, 5.0]:
            evs = rolling_accel_events(eth_buy_liqs, threshold_ratio=ratio, min_5s_notional=50_000, cooldown_ms=5 * MINUTE)
            accel_results[f"ratio_{ratio}"] = {
                "n": len(evs),
                "15m_fee_net": event_summary(eth_mark, evs, 15, 6.1),
                "executable_15m_fee_net": executable_summary(conn, evs, 15 * 60, 6.1),
            }

        # 3. Passive pullback entry after running event.
        passive = {
            f"offset_{off}bps_wait5m_hold15m": passive_pullback(eth_mark, running_events, off, 5, 15, 6.1)
            for off in [5, 10, 20, 30]
        }

        # 4. Second leg: spike -> pullback in next 5m -> NAV remains high at +5m.
        second_leg = []
        for e in closed_spikes:
            t = int(e["ts"])
            entry = eth_mark.at_or_after(t)
            lo, _ = mark_minmax(eth_mark, t, t + 5 * MINUTE)
            nav5 = nav_by_ts.get(((t + 5 * MINUTE) // MINUTE) * MINUTE)
            if entry and lo and bps(entry, lo) is not None and bps(entry, lo) <= -5 and nav5 and int(nav5["score"]) >= 7:
                second_leg.append({"ts": t + 5 * MINUTE})
        second_leg_res = {
            "n": len(second_leg),
            "15m_fee_net": event_summary(eth_mark, second_leg, 15, 6.1),
            "60m_fee_net": event_summary(eth_mark, second_leg, 60, 6.1),
        }

        # 5. Execution filters on running events.
        filter_groups: dict[str, list[dict[str, Any]]] = {"all": running_events, "spread_le_0p10": [], "spread_le_0p15": [], "ask_depth_ge_100k": [], "bid_imbalance_ge_0": []}
        for e in running_events:
            bk = book_at(conn, int(e["ts"])) or {}
            if float(bk.get("spread_bps", 99)) <= 0.10:
                filter_groups["spread_le_0p10"].append(e)
            if float(bk.get("spread_bps", 99)) <= 0.15:
                filter_groups["spread_le_0p15"].append(e)
            if float(bk.get("ask_depth_usd", 0)) >= 100_000:
                filter_groups["ask_depth_ge_100k"].append(e)
            if float(bk.get("book_imbalance", -1)) >= 0:
                filter_groups["bid_imbalance_ge_0"].append(e)
        exec_filters = {
            k: {"n": len(v), "executable_15m_fee_net": executable_summary(conn, v, 15 * 60, 6.1)}
            for k, v in filter_groups.items()
        }

        # 6. Swing marker use, not entry.
        swing = {f"{h}m_fee_net": event_summary(eth_mark, closed_spikes, h, 6.1) for h in [15, 60, 120, 240, 480]}

        # 7. Extreme p99 spike fade.
        extreme_threshold = thresholds["BUY"]["p99_nonzero"]
        extreme = [e for e in closed_spikes if float(e["notional"]) >= extreme_threshold]
        extreme_fade = {
            "threshold": round(extreme_threshold, 1),
            "n": len(extreme),
            "short_15m_fee_net": summary(fee_net([ret_short_mark(eth_mark, int(e["ts"]), 15) for e in extreme if ret_short_mark(eth_mark, int(e["ts"]), 15) is not None], 6.1)),
            "short_60m_fee_net": summary(fee_net([ret_short_mark(eth_mark, int(e["ts"]), 60) for e in extreme if ret_short_mark(eth_mark, int(e["ts"]), 60) is not None], 6.1)),
            "long_15m_fee_net": event_summary(eth_mark, extreme, 15, 6.1),
        }

        # 8. Cross-asset lead: BTC/SOL BUY spike within prior 60s before ETH running event.
        def lead_events(sym_liqs: list[tuple[int, float]], symbol: str) -> dict[str, Any]:
            vals = rolling_sums_at_liqs(sym_liqs, 30_000)
            th = max(50_000.0, pct(vals, 0.95))
            evs = running_cross_events(sym_liqs, window_ms=30_000, threshold=th, cooldown_ms=5 * MINUTE)
            ts_list = sorted(int(e["ts"]) for e in evs)
            subset = []
            for e in running_events:
                t = int(e["ts"])
                i = bisect.bisect_left(ts_list, t)
                ok = (i > 0 and 0 <= t - ts_list[i - 1] <= 60_000)
                if ok:
                    subset.append(e)
            return {
                "symbol": symbol,
                "lead_threshold": round(th, 1),
                "lead_events": len(evs),
                "eth_subset_n": len(subset),
                "eth_subset_15m_fee_net": event_summary(eth_mark, subset, 15, 6.1),
                "eth_without_lead_15m_fee_net": event_summary(eth_mark, [e for e in running_events if e not in subset], 15, 6.1),
            }

        cross_asset = {"BTC": lead_events(btc_buy_liqs, "BTCUSDT"), "SOL": lead_events(sol_buy_liqs, "SOLUSDT")}

        # 9. NAV transitions.
        nav_transitions = {"low_to_high": [], "high_to_low": []}
        prev = None
        for n in nav:
            if prev:
                if int(prev["score"]) < 5 and int(n["score"]) >= 7:
                    nav_transitions["low_to_high"].append({"ts": int(n["ts"])})
                if int(prev["score"]) >= 7 and int(n["score"]) < 5:
                    nav_transitions["high_to_low"].append({"ts": int(n["ts"])})
            prev = n
        nav_transition_res = {
            k: {"n": len(v), "15m_fee_net": event_summary(eth_mark, v, 15, 6.1), "60m_fee_net": event_summary(eth_mark, v, 60, 6.1)}
            for k, v in nav_transitions.items()
        }

        # 10. State labels on running events.
        labels: dict[str, list[dict[str, Any]]] = {
            "MOMENTUM_BUILDING": [],
            "SQUEEZE_ACTIVE": [],
            "EXHAUSTION_RISK": [],
            "PULLBACK_ENTRY_OK": [],
            "NO_TRADE_LIQUIDITY_THIN": [],
            "SECOND_LEG_PROBABLE": second_leg,
        }
        for e in running_events:
            t = int(e["ts"])
            navrow = nav_by_ts.get((t // MINUTE) * MINUTE) or {}
            score = int(navrow.get("score", 0))
            bk = book_at(conn, t) or {}
            r1 = eth_mark.ret(t - MINUTE, MINUTE)
            rn = float(e.get("running_notional", 0))
            if rn >= run60_threshold and score >= 5:
                labels["MOMENTUM_BUILDING"].append(e)
            if rn >= run60_threshold and r1 is not None and r1 > 0:
                labels["SQUEEZE_ACTIVE"].append(e)
            if rn >= run60_threshold * 1.5 or (r1 is not None and r1 > 40):
                labels["EXHAUSTION_RISK"].append(e)
            if float(bk.get("spread_bps", 99)) <= 0.15 and float(bk.get("bid_depth_usd", 0)) >= 135_423:
                labels["PULLBACK_ENTRY_OK"].append(e)
            if float(bk.get("bid_depth_usd", 999999)) < 75_000 or float(bk.get("spread_bps", 0)) > 0.35:
                labels["NO_TRADE_LIQUIDITY_THIN"].append(e)
        state_label_res = {
            k: {"n": len(v), "15m_fee_net": event_summary(eth_mark, v, 15, 6.1), "60m_fee_net": event_summary(eth_mark, v, 60, 6.1)}
            for k, v in labels.items()
        }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "symbol": "ETHUSDT",
            "days": int(args.days),
            "start_utc": utc_ms(start_ms),
            "end_utc": utc_ms(end_ms),
            "closed_buy_spikes": len(closed_spikes),
            "running_60s_events": len(running_events),
            "run60_threshold": round(run60_threshold, 1),
            "note": "Research-only. No live executor/config/order logic touched.",
        },
        "pre_spike_build_up": pre_build,
        "acceleration_anchor": accel_results,
        "passive_pullback": passive,
        "second_leg_confirmation": second_leg_res,
        "execution_filters": exec_filters,
        "swing_marker": swing,
        "extreme_fade": extreme_fade,
        "cross_asset_lead": cross_asset,
        "nav_transitions": nav_transition_res,
        "state_labels": state_label_res,
    }

    good = []
    bad = []
    if accel_results["ratio_3.0"]["executable_15m_fee_net"]["t3r"] and accel_results["ratio_3.0"]["executable_15m_fee_net"]["t3r"] > 0:
        good.append("acceleration ratio_3 executable survives")
    else:
        bad.append("acceleration anchors do not survive executable/T3R")
    if passive["offset_10bps_wait5m_hold15m"]["filled_fee_net"]["t3r"] and passive["offset_10bps_wait5m_hold15m"]["filled_fee_net"]["t3r"] > 0:
        good.append("passive 10bps pullback has positive T3R")
    else:
        bad.append("passive pullback not robust enough")
    if second_leg_res["15m_fee_net"]["t3r"] and second_leg_res["15m_fee_net"]["t3r"] > 0:
        good.append("second-leg state has positive T3R")
    else:
        bad.append("second-leg confirmation does not produce robust sample")
    if extreme_fade["short_60m_fee_net"]["t3r"] and extreme_fade["short_60m_fee_net"]["t3r"] > 0:
        good.append("extreme fade short survives")
    else:
        bad.append("extreme fade short does not survive")
    result["verdict"] = f"RESEARCH_ONLY: possible_leads={good}; failed_or_weak={bad}"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    render(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
