"""Next-generation research map for S34 V02.

Research-only. This script tests the 12 larger questions around the current
live V02 lane without touching live execution/config/state:

1. event graph anatomy
2. phase timing
3. fixed horizon surface
4. MFE/giveback exits
5. MAE survival
6. NAV/BUY spike as phase sensor
7. tail/weak-trade early detection
8. execution surface
9. fill-delay quality
10. regime identity
11. similarity-memory/KNN
12. mechanism expansion: forced sell into deep bid without liquidation
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    MarkIndex,
    iso_ms,
    load_liquidations,
    load_mark_index,
    pctile,
    r1,
    reconstruct_anchors,
    signed_return_bps,
)
from tools.research_s34_maker_fade import anchor_vdepth_bps, maker_limit_price  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_v02_alpha_navigation_overlay import (  # noqa: E402
    ACCEL_WINDOW_SEC,
    BUCKET_SEC,
    DEFAULT_CROSS_MARGIN_BPS,
    HORIZON_SEC,
    INITIAL_OFFSET_BPS,
    LIQ_SIDE,
    MIN_BID_DEPTH_USD,
    MIN_GAP_SEC,
    PRIOR4H_LT_BPS,
    REPLACE_OFFSET_BPS,
    REPLACE_WAIT_SEC,
    RULE_NAME,
    SYMBOL,
    THRESHOLD_USD,
    VDEPTH_MAX_BPS,
    VDEPTH_MIN_BPS,
    book_exit_price,
    build_live_like_trades,
    find_v02_fill,
    has_event_between,
    load_buy_spike_minutes,
    make_trade_tags,
    nav_score,
    state_sequence,
    trade_path_stats,
    trade_return_at,
)
from tools.s34_v02_nav_spike_tests import (  # noqa: E402
    DB_PATH,
    MINUTE,
    OUT_DIR,
    build_nav,
    bps,
    latest_ts,
    load_book_for_buckets,
    load_flow_1m,
    load_liq_1m,
    load_mark_1m,
    make_series,
    spike_thresholds,
)


OUT_JSON = OUT_DIR / "S34_V02_NEXT_GEN_ALPHA_RESEARCH.json"
OUT_MD = OUT_DIR / "S34_V02_NEXT_GEN_ALPHA_RESEARCH.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(vals: list[float | None]) -> list[float]:
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def metrics(vals: list[float | None]) -> dict[str, Any]:
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
        "min": r1(min(xs)),
        "max": r1(max(xs)),
    }


def summarize_group(rows: list[dict[str, Any]], key: str, value_key: str = "net_2h_bps") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for val in sorted({str(r.get(key, "NA")) for r in rows}):
        out[val] = metrics([r.get(value_key) for r in rows if str(r.get(key, "NA")) == val])
    return out


def mark_return_from_entry(marks: MarkIndex, entry_px: float, start_ms: int, horizon_sec: int) -> float | None:
    px = marks.at_or_after(int(start_ms) + int(horizon_sec) * 1000)
    return signed_return_bps("LONG", float(entry_px), float(px[1])) if px else None


def path_returns(marks: MarkIndex, entry_px: float, start_ms: int, horizon_sec: int) -> list[tuple[int, float]]:
    return [
        (int(ts), signed_return_bps("LONG", float(entry_px), float(px)))
        for ts, px in marks.slice_range(int(start_ms), int(start_ms) + int(horizon_sec) * 1000)
        if int(ts) >= int(start_ms)
    ]


def first_cross_time(path: list[tuple[int, float]], start_ms: int, level_bps: float) -> float | None:
    for ts, r in path:
        if float(r) >= float(level_bps):
            return round((int(ts) - int(start_ms)) / 1000.0, 1)
    return None


def first_nav_high(nav_by_ts: dict[int, dict[str, Any]], start_ms: int, horizon_min: int) -> float | None:
    for i in range(horizon_min + 1):
        t = int(start_ms) + i * MINUTE
        if nav_score(nav_by_ts, t) >= 7:
            return float(i * 60)
    return None


def first_buy_spike_after(ts_list: list[int], start_ms: int, horizon_min: int) -> float | None:
    i = bisect.bisect_left(ts_list, int(start_ms))
    if i < len(ts_list) and ts_list[i] <= int(start_ms) + horizon_min * MINUTE:
        return round((ts_list[i] - int(start_ms)) / 1000.0, 1)
    return None


def giveback_exit(
    marks: MarkIndex,
    trade: dict[str, Any],
    *,
    horizon_sec: int,
    min_peak_bps: float,
    giveback_frac: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> float | None:
    if trade.get("status") != "FILLED":
        return None
    start = int(trade["fill_ts_ms"])
    entry = float(trade["entry_price"])
    peak = -10**9
    armed = False
    for _, r in path_returns(marks, entry, start, horizon_sec):
        peak = max(peak, float(r))
        if peak >= float(min_peak_bps):
            armed = True
        if armed and float(r) <= peak * (1.0 - float(giveback_frac)):
            return float(r) - float(maker_fee_bps) - float(taker_fee_bps)
    return mark_return_from_entry(marks, entry, start, horizon_sec) - float(maker_fee_bps) - float(taker_fee_bps)


def mae_in_window(marks: MarkIndex, trade: dict[str, Any], minutes: int) -> float | None:
    if trade.get("status") != "FILLED":
        return None
    rets = path_returns(marks, float(trade["entry_price"]), int(trade["fill_ts_ms"]), int(minutes) * 60)
    return min((r for _, r in rets), default=None)


def book_replenishment(conn: sqlite3.Connection, ts_ms: int, base_bid_depth: float, sec: int) -> float | None:
    b = book_features_at(conn, SYMBOL, int(ts_ms) + int(sec) * 1000, 10)
    if not b or float(base_bid_depth) <= 0:
        return None
    return (float(b["bid_depth_usd"]) - float(base_bid_depth)) / float(base_bid_depth)


def liq_sum(rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    return sum(float(r["notional"]) for r in rows if int(start_ms) <= int(r["ts_ms"]) < int(end_ms))


def custom_v02_fill(marks: MarkIndex, anchor_ts: int, anchor_px: float, initial_offset: float, replace_wait: int, replace_offset: float, cross_margin: float) -> dict[str, Any] | None:
    first_limit = maker_limit_price(float(anchor_px), "LONG", float(initial_offset))
    first_required = first_limit * (1.0 - float(cross_margin) / 10_000.0)
    repl_limit = maker_limit_price(float(anchor_px), "LONG", float(replace_offset))
    repl_required = repl_limit * (1.0 - float(cross_margin) / 10_000.0)
    replace_ts = int(anchor_ts) + int(replace_wait) * 1000
    end = int(anchor_ts) + HORIZON_SEC * 1000
    for ts, px in marks.slice_range(int(anchor_ts), end):
        if int(ts) <= int(anchor_ts):
            continue
        if int(ts) <= replace_ts:
            if float(px) <= first_required:
                return {"fill_ts_ms": int(ts), "entry_price": float(first_limit), "leg": "initial", "offset": float(initial_offset)}
        elif float(px) <= repl_required:
            return {"fill_ts_ms": int(ts), "entry_price": float(repl_limit), "leg": "replacement", "offset": float(replace_offset)}
    return None


def collect_v02_anchors(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    liqs = load_liquidations(conn, SYMBOL, LIQ_SIDE, start_ms, end_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    out = []
    for anchor in anchors:
        depth = anchor_vdepth_bps(marks, anchor, LIQ_SIDE)
        if depth is None or not (VDEPTH_MIN_BPS <= float(depth) < VDEPTH_MAX_BPS):
            continue
        prior4h = marks.ret_bps(int(anchor.anchor_ts_ms) - 4 * 3600 * 1000, int(anchor.anchor_ts_ms))
        if prior4h is None or not (float(prior4h) < PRIOR4H_LT_BPS):
            continue
        book = book_features_at(conn, SYMBOL, int(anchor.anchor_ts_ms), 10)
        if not book or float(book["bid_depth_usd"]) < MIN_BID_DEPTH_USD:
            continue
        mark = marks.at_or_after(int(anchor.anchor_ts_ms))
        if not mark:
            continue
        out.append(
            {
                "anchor_ts_ms": int(anchor.anchor_ts_ms),
                "anchor_mark_ts_ms": int(mark[0]),
                "anchor_mark_price": float(mark[1]),
                "vdepth_bps": float(depth),
                "prior4h_bps": float(prior4h),
                "bid_depth_usd": float(book["bid_depth_usd"]),
                "spread_bps": float(book["spread_bps"]),
                "book_imbalance": float(book["book_imbalance"]),
            }
        )
    return out


def execution_surface(conn: sqlite3.Connection, anchors: list[dict[str, Any]], maker_fee: float, taker_fee: float, cross_margin: float) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    out: dict[str, Any] = {}
    for offset in (10.0, 15.0, 20.0, 25.0, 30.0):
        for wait in (120, 180, 300, 600):
            vals = []
            fills = 0
            legs = Counter()
            for a in anchors:
                fill = custom_v02_fill(marks, int(a["anchor_mark_ts_ms"]), float(a["anchor_mark_price"]), offset, wait, REPLACE_OFFSET_BPS, cross_margin)
                if not fill:
                    continue
                fills += 1
                legs[str(fill["leg"])] += 1
                px = book_exit_price(conn, int(fill["fill_ts_ms"]) + HORIZON_SEC * 1000)
                if px is None:
                    continue
                vals.append(signed_return_bps("LONG", float(fill["entry_price"]), float(px)) - maker_fee - taker_fee)
            out[f"O{int(offset)}_W{wait}"] = {"fills": fills, "legs": dict(legs), "result": metrics(vals)}
    return out


def normalize_features(rows: list[dict[str, Any]], feature_keys: list[str]) -> dict[str, tuple[float, float]]:
    params = {}
    for k in feature_keys:
        xs = clean([r.get(k) for r in rows])
        mu = mean(xs) if xs else 0.0
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) if xs else 1.0
        params[k] = (mu, sd if sd > 1e-9 else 1.0)
    return params


def knn_memory(rows: list[dict[str, Any]], feature_keys: list[str], k: int = 3) -> dict[str, Any]:
    filled = [r for r in rows if r.get("status") == "FILLED" and r.get("net_2h_bps") is not None]
    if len(filled) < k + 2:
        return {"n": len(filled), "status": "DATA_INSUFFICIENT"}
    params = normalize_features(filled, feature_keys)
    preds = []
    actuals = []
    details = []
    for i, row in enumerate(filled):
        dists = []
        for j, other in enumerate(filled):
            if i == j:
                continue
            dist = 0.0
            ok = True
            for key in feature_keys:
                if row.get(key) is None or other.get(key) is None:
                    ok = False
                    break
                mu, sd = params[key]
                dist += ((float(row[key]) - mu) / sd - (float(other[key]) - mu) / sd) ** 2
            if ok:
                dists.append((math.sqrt(dist), other))
        nbrs = [x[1] for x in sorted(dists, key=lambda z: z[0])[:k]]
        if len(nbrs) < k:
            continue
        pred = mean([float(n["net_2h_bps"]) for n in nbrs])
        preds.append(pred)
        actuals.append(float(row["net_2h_bps"]))
        details.append({"ts": row.get("fill_utc"), "actual": r1(float(row["net_2h_bps"])), "pred": r1(pred)})
    if len(preds) < 3:
        return {"n": len(preds), "status": "DATA_INSUFFICIENT"}
    mp = mean(preds)
    ma = mean(actuals)
    cov = sum((p - mp) * (a - ma) for p, a in zip(preds, actuals))
    vp = sum((p - mp) ** 2 for p in preds)
    va = sum((a - ma) ** 2 for a in actuals)
    corr = cov / math.sqrt(vp * va) if vp > 0 and va > 0 else None
    mae = mean([abs(p - a) for p, a in zip(preds, actuals)])
    return {"n": len(preds), "k": k, "corr": r1(corr), "mae_bps": r1(mae), "details": details[:20]}


def forced_sell_expansion(conn: sqlite3.Connection, start_ms: int, end_ms: int, maker_fee: float, taker_fee: float, cross_margin: float) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    rows = conn.execute(
        """
        SELECT CAST(ts_ms / 300000 AS INTEGER) * 300000 AS bucket,
               SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) AS sell_notional,
               SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END) AS buy_notional
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?
        GROUP BY bucket
        ORDER BY bucket
        """,
        (int(start_ms), int(end_ms)),
    ).fetchall()
    buckets = [(int(r[0]), float(r[1] or 0.0), float(r[2] or 0.0)) for r in rows]
    if not buckets:
        return {"status": "NO_DATA"}
    sell_vals = sorted([b[1] for b in buckets if b[1] > 0])
    threshold = pctile(sell_vals, 0.95) or 0.0
    liqs = load_liquidations(conn, SYMBOL, "SELL", start_ms, end_ms)
    candidates = []
    for bucket, sell_n, buy_n in buckets:
        if sell_n < threshold:
            continue
        liq_n = liq_sum(liqs, bucket, bucket + 300_000)
        if liq_n >= 50_000.0:
            continue
        start_px = marks.at_or_after(bucket)
        end_px = marks.at_or_after(bucket + 300_000)
        if not start_px or not end_px:
            continue
        depth = (float(start_px[1]) - float(end_px[1])) / float(start_px[1]) * 10_000.0
        if not (VDEPTH_MIN_BPS <= depth < VDEPTH_MAX_BPS):
            continue
        prior4h = marks.ret_bps(bucket - 4 * 3600 * 1000, bucket)
        if prior4h is None or not (float(prior4h) < PRIOR4H_LT_BPS):
            continue
        book = book_features_at(conn, SYMBOL, bucket + 300_000, 10)
        if not book or float(book["bid_depth_usd"]) < MIN_BID_DEPTH_USD:
            continue
        fill = custom_v02_fill(marks, int(end_px[0]), float(end_px[1]), INITIAL_OFFSET_BPS, REPLACE_WAIT_SEC, REPLACE_OFFSET_BPS, cross_margin)
        if not fill:
            candidates.append({"status": "NO_FILL", "bucket": bucket, "sell_notional": sell_n, "depth_bps": depth})
            continue
        exit_px = book_exit_price(conn, int(fill["fill_ts_ms"]) + HORIZON_SEC * 1000)
        if exit_px is None:
            continue
        net = signed_return_bps("LONG", float(fill["entry_price"]), float(exit_px)) - maker_fee - taker_fee
        candidates.append(
            {
                "status": "FILLED",
                "bucket": bucket,
                "utc": iso_ms(bucket),
                "sell_notional": r1(sell_n),
                "buy_notional": r1(buy_n),
                "depth_bps": r1(depth),
                "prior4h_bps": r1(prior4h),
                "bid_depth_usd": r1(float(book["bid_depth_usd"])),
                "fill_delay_sec": r1((int(fill["fill_ts_ms"]) - int(end_px[0])) / 1000.0),
                "net_2h_bps": float(net),
            }
        )
    filled = [c for c in candidates if c.get("status") == "FILLED"]
    return {
        "sell_notional_p95": r1(threshold),
        "candidates_total": len(candidates),
        "filled_n": len(filled),
        "result": metrics([c.get("net_2h_bps") for c in filled]),
        "sample": filled[:20],
    }


def render(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 V02 Next-Gen Alpha Research",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## Executive Read",
        "",
        result["executive_read"],
        "",
    ]
    sections = [
        ("1. Event Graph Anatomy", "event_graph_anatomy"),
        ("2. Phase Timing", "phase_timing"),
        ("3. Fixed Horizon Surface", "fixed_horizon_surface"),
        ("4. MFE/Giveback Exit", "giveback_exit"),
        ("5. MAE Survival", "mae_survival"),
        ("6. NAV/BUY Spike Phase Sensor", "phase_sensor"),
        ("7. Tail / Weak-Trade Detection", "tail_detection"),
        ("8. Execution Surface", "execution_surface"),
        ("9. Fill Delay Quality", "fill_delay_quality"),
        ("10. Regime Identity", "regime_identity"),
        ("11. Similarity Memory / KNN", "similarity_memory"),
        ("12. Mechanism Expansion", "mechanism_expansion"),
    ]
    for title, key in sections:
        lines += [f"## {title}", "", "```json", json.dumps(result[key], indent=2, ensure_ascii=True), "```", ""]
    lines += ["## Decision Tags", "", "```json", json.dumps(result["decision_tags"], indent=2, ensure_ascii=True), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--maker-fee-bps", type=float, default=-0.5)
    ap.add_argument("--taker-fee-bps", type=float, default=3.05)
    ap.add_argument("--cross-margin-bps", type=float, default=DEFAULT_CROSS_MARGIN_BPS)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE
        maker_fee = float(args.maker_fee_bps)
        taker_fee = float(args.taker_fee_bps)
        fee = maker_fee + taker_fee

        eth_mark_1m = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")
        btc_mark_1m = load_mark_1m(conn, start_ms, end_ms, "BTCUSDT")
        buckets = sorted(eth_mark_1m)
        liq_1m = load_liq_1m(conn, start_ms, end_ms)
        flow_1m = load_flow_1m(conn, start_ms, end_ms)
        book_1m = load_book_for_buckets(conn, buckets)
        buckets = sorted(set(eth_mark_1m) & set(book_1m))
        nav = build_nav(buckets, book_1m, liq_1m, flow_1m, btc_mark_1m)
        nav_by_ts = {int(n["ts"]): n for n in nav}
        buy_th = spike_thresholds(liq_1m)["BUY"]
        buy_spikes = load_buy_spike_minutes(liq_1m, float(buy_th["primary_threshold"]))
        buy_spike_ts = [int(e["ts"]) for e in buy_spikes]
        buy_spike_notional_by_ts = {int(e["ts"]): float(e["notional"]) for e in buy_spikes}

        trades_all = build_live_like_trades(
            conn,
            start_ms=start_ms,
            end_ms=end_ms,
            maker_fee_bps=maker_fee,
            taker_fee_bps=taker_fee,
            cross_margin_bps=float(args.cross_margin_bps),
        )
        filled = [t for t in trades_all if t.get("status") == "FILLED"]
        marks = load_mark_index(conn, SYMBOL)
        btc_marks = load_mark_index(conn, "BTCUSDT")
        sol_marks = load_mark_index(conn, "SOLUSDT")
        sell_liqs = load_liquidations(conn, SYMBOL, "SELL", start_ms, end_ms)
        btc_sell_liqs = load_liquidations(conn, "BTCUSDT", "SELL", start_ms, end_ms)
        sol_sell_liqs = load_liquidations(conn, "SOLUSDT", "SELL", start_ms, end_ms)

        for t in filled:
            t.update(
                make_trade_tags(
                    t,
                    nav_by_ts=nav_by_ts,
                    buy_spike_ts=buy_spike_ts,
                    buy_spike_notional_by_ts=buy_spike_notional_by_ts,
                    extreme_buy_threshold=float(buy_th["p99_nonzero"]),
                )
            )
            t.update(trade_path_stats(marks, t, HORIZON_SEC))
            path = path_returns(marks, float(t["entry_price"]), int(t["fill_ts_ms"]), HORIZON_SEC)
            t["rebound_20bps_time_sec"] = first_cross_time(path, int(t["fill_ts_ms"]), 20.0)
            t["rebound_50bps_time_sec"] = first_cross_time(path, int(t["fill_ts_ms"]), 50.0)
            t["first_nav_high_sec"] = first_nav_high(nav_by_ts, int(t["fill_ts_ms"]), 120)
            t["first_buy_spike_sec"] = first_buy_spike_after(buy_spike_ts, int(t["fill_ts_ms"]), 120)
            t["giveback_to_exit_bps"] = r1(float(t.get("mfe_bps") or 0.0) - float(t.get("net_2h_bps") or 0.0))
            for m in (5, 10, 15, 30):
                t[f"mae_{m}m_bps"] = r1(mae_in_window(marks, t, m))
            t["replenish_120s_pct"] = r1(100.0 * book_replenishment(conn, int(t["fill_ts_ms"]), float(t.get("bid_depth_usd", 0.0)), 120)) if t.get("bid_depth_usd") else None
            t["sell_liq_5m_after"] = r1(liq_sum(sell_liqs, int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 5 * MINUTE))
            t["sell_liq_15m_after"] = r1(liq_sum(sell_liqs, int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 15 * MINUTE))
            t["btc_sell_liq_10m_after"] = r1(liq_sum(btc_sell_liqs, int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 10 * MINUTE))
            t["sol_sell_liq_10m_after"] = r1(liq_sum(sol_sell_liqs, int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + 10 * MINUTE))
            t["eth_daily_prior_bps"] = r1(marks.ret_bps(int(t["fill_ts_ms"]) - 24 * 3600 * 1000, int(t["fill_ts_ms"])))
            t["btc_daily_prior_bps"] = r1(btc_marks.ret_bps(int(t["fill_ts_ms"]) - 24 * 3600 * 1000, int(t["fill_ts_ms"])))
            t["btc_2h_after_bps"] = r1(btc_marks.ret_bps(int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + HORIZON_SEC * 1000))
            t["sol_2h_after_bps"] = r1(sol_marks.ret_bps(int(t["fill_ts_ms"]), int(t["fill_ts_ms"]) + HORIZON_SEC * 1000))
            t["fill_delay_bin"] = "lt5m" if float(t["fill_delay_sec"]) < 300 else "5to15m" if float(t["fill_delay_sec"]) < 900 else "gt15m"
            t["leg"] = "initial" if float(t["entry_offset_bps"]) == INITIAL_OFFSET_BPS else "replacement"
            t["regime_eth_daily"] = "eth_daily_up" if (t.get("eth_daily_prior_bps") or 0.0) >= 0.0 else "eth_daily_down"
            t["regime_btc_daily"] = "btc_daily_up" if (t.get("btc_daily_prior_bps") or 0.0) >= 0.0 else "btc_daily_down"
            t["weak_trade"] = float(t.get("net_2h_bps") or 0.0) < 50.0

        event_cards = [
            {
                "fill_utc": t.get("fill_utc"),
                "net_2h_bps": r1(t.get("net_2h_bps")),
                "fill_delay_sec": t.get("fill_delay_sec"),
                "mae_bps": t.get("mae_bps"),
                "mae_time_sec": t.get("mae_time_sec"),
                "mfe_bps": t.get("mfe_bps"),
                "mfe_time_sec": t.get("mfe_time_sec"),
                "first_nav_high_sec": t.get("first_nav_high_sec"),
                "first_buy_spike_sec": t.get("first_buy_spike_sec"),
                "state_sequence_5m": t.get("state_sequence_5m"),
                "nav_recommendation": t.get("nav_recommendation"),
            }
            for t in filled
        ]

        phase_timing = {
            "mae_time_sec": metrics([t.get("mae_time_sec") for t in filled]),
            "mfe_time_sec": metrics([t.get("mfe_time_sec") for t in filled]),
            "rebound_20bps_time_sec": metrics([t.get("rebound_20bps_time_sec") for t in filled]),
            "rebound_50bps_time_sec": metrics([t.get("rebound_50bps_time_sec") for t in filled]),
            "first_nav_high_sec": metrics([t.get("first_nav_high_sec") for t in filled]),
            "first_buy_spike_sec": metrics([t.get("first_buy_spike_sec") for t in filled]),
        }

        fixed_horizons = {
            f"{m}m": metrics([trade_return_at(conn, t, m * 60, maker_fee, taker_fee) for t in filled])
            for m in (15, 30, 60, 90, 120, 180, 240)
        }

        giveback = {
            f"peak{int(min_peak)}_gb{int(gb * 100)}": metrics(
                [
                    giveback_exit(
                        marks,
                        t,
                        horizon_sec=240 * 60,
                        min_peak_bps=min_peak,
                        giveback_frac=gb,
                        maker_fee_bps=maker_fee,
                        taker_fee_bps=taker_fee,
                    )
                    for t in filled
                ]
            )
            for min_peak in (40.0, 80.0)
            for gb in (0.25, 0.40, 0.60)
        }

        mae_survival = {}
        for m in (5, 10, 15, 30):
            key = f"mae_{m}m_bps"
            for cut in (-20.0, -50.0, -100.0):
                label = f"{key}_le_{int(cut)}"
                rows = []
                for t in filled:
                    v = t.get(key)
                    if v is not None:
                        tt = dict(t)
                        tt[label] = float(v) <= cut
                        rows.append(tt)
                mae_survival[label] = summarize_group(rows, label)

        phase_sensor = {
            "buy_spike_post_5m": summarize_group(filled, "buy_spike_post_5m"),
            "buy_spike_post_15m": summarize_group(filled, "buy_spike_post_15m"),
            "nav_high_fill": summarize_group(filled, "nav_high_fill"),
            "nav_high_holds_5m": summarize_group(filled, "nav_high_holds_5m"),
            "rebound_confirmed_5m": summarize_group(filled, "rebound_confirmed_5m"),
        }

        tail_detection = {
            "actual_negative_tail_n": sum(1 for t in filled if float(t.get("net_2h_bps") or 0.0) < 0.0),
            "weak_trade_n_lt_50bps": sum(1 for t in filled if t.get("weak_trade")),
            "weak_by_replenish_120s": summarize_group(
                [
                    dict(t, replenish_bucket="high" if (t.get("replenish_120s_pct") is not None and float(t["replenish_120s_pct"]) >= 0.0) else "low")
                    for t in filled
                ],
                "replenish_bucket",
            ),
            "weak_by_sell_liq_5m": summarize_group(
                [
                    dict(t, sell_liq_5m_bucket="high" if float(t.get("sell_liq_5m_after") or 0.0) >= 50_000.0 else "low")
                    for t in filled
                ],
                "sell_liq_5m_bucket",
            ),
            "weak_by_btc_after": summarize_group(
                [dict(t, btc_after_bucket="btc_down" if (t.get("btc_2h_after_bps") or 0.0) < 0.0 else "btc_up") for t in filled],
                "btc_after_bucket",
            ),
        }

        anchors = collect_v02_anchors(conn, start_ms, end_ms)
        exec_surface = execution_surface(conn, anchors, maker_fee, taker_fee, float(args.cross_margin_bps))

        fill_delay_quality = {
            "by_leg": summarize_group(filled, "leg"),
            "by_delay_bin": summarize_group(filled, "fill_delay_bin"),
            "fill_delay_sec": metrics([t.get("fill_delay_sec") for t in filled]),
        }

        regime_identity = {
            "by_eth_daily": summarize_group(filled, "regime_eth_daily"),
            "by_btc_daily": summarize_group(filled, "regime_btc_daily"),
            "prior_eth_daily_bps": metrics([t.get("eth_daily_prior_bps") for t in filled]),
            "prior_btc_daily_bps": metrics([t.get("btc_daily_prior_bps") for t in filled]),
        }

        feature_keys = [
            "vdepth_bps",
            "prior4h_bps",
            "bid_depth_usd",
            "spread_bps",
            "book_imbalance",
            "fill_delay_sec",
            "nav_score_fill",
            "mae_15m_bps",
        ]
        similarity = knn_memory(filled, feature_keys, k=3)
        expansion = forced_sell_expansion(conn, start_ms, end_ms, maker_fee, taker_fee, float(args.cross_margin_bps))

    baseline = metrics([t.get("net_2h_bps") for t in filled])
    best_horizon = max(fixed_horizons.items(), key=lambda kv: float(kv[1].get("t3r") or -10**9))
    best_exec = max(exec_surface.items(), key=lambda kv: float(kv[1]["result"].get("t3r") or -10**9)) if exec_surface else None
    best_giveback = max(giveback.items(), key=lambda kv: float(kv[1].get("t3r") or -10**9))

    tags = []
    if best_horizon[0] != "120m" and float(best_horizon[1].get("t3r") or 0.0) > float(baseline.get("t3r") or 0.0):
        tags.append("MANAGEMENT_LEAD_FIXED_HORIZON")
    else:
        tags.append("KEEP_BASELINE_2H")
    if best_giveback[1].get("t3r", 0.0) > baseline.get("t3r", 0.0):
        tags.append("MANAGEMENT_LEAD_GIVEBACK")
    if best_exec and best_exec[0] != "O20_W300" and best_exec[1]["result"].get("t3r", 0.0) > baseline.get("t3r", 0.0):
        tags.append("EXECUTION_SURFACE_LEAD")
    if expansion.get("result", {}).get("t3r", 0.0) > 0 and expansion.get("filled_n", 0) >= 5:
        tags.append("MECHANISM_EXPANSION_LEAD")
    if similarity.get("corr") is not None and float(similarity["corr"]) > 0:
        tags.append("MEMORY_LEAD")

    executive = (
        f"V02 baseline filled N={baseline['n']} sum={baseline['sum']} median={baseline['median']} "
        f"T3R={baseline['t3r']}. Best fixed horizon is {best_horizon[0]} with T3R={best_horizon[1]['t3r']}. "
        f"Best giveback is {best_giveback[0]} with T3R={best_giveback[1]['t3r']}. "
        f"Best execution cell is {best_exec[0] if best_exec else None}. "
        f"Mechanism expansion filled N={expansion.get('filled_n')} T3R={expansion.get('result', {}).get('t3r')}."
    )

    result = {
        "generated_at_utc": utc_now(),
        "scope": {
            "rule": RULE_NAME,
            "days": int(args.days),
            "start_utc": iso_ms(start_ms),
            "end_utc": iso_ms(end_ms),
            "maker_fee_bps": maker_fee,
            "taker_fee_bps": taker_fee,
            "cross_margin_bps": float(args.cross_margin_bps),
            "anchors_total": len(trades_all),
            "filled_n": len(filled),
            "note": "Research-only. No live executor/config/order logic touched.",
        },
        "executive_read": executive,
        "event_graph_anatomy": {"cards": event_cards, "baseline": baseline},
        "phase_timing": phase_timing,
        "fixed_horizon_surface": fixed_horizons,
        "giveback_exit": giveback,
        "mae_survival": mae_survival,
        "phase_sensor": phase_sensor,
        "tail_detection": tail_detection,
        "execution_surface": exec_surface,
        "fill_delay_quality": fill_delay_quality,
        "regime_identity": regime_identity,
        "similarity_memory": similarity,
        "mechanism_expansion": expansion,
        "decision_tags": tags,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    render(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
