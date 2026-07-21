"""Navigation overlay tests for the live S34 V02 deepbid alpha.

Research-only. This does not touch the live executor, env, runtime state, or
order logic. It replays historical live-like V02 maker fills, then tests whether
the BUY-spike/NAV indicator is useful as a permission/management layer around
the existing alpha:

1. BUY spike before/after the V02 trade
2. scalp horizon decomposition
3. danger/navigation tags versus tail
4. first-5m state sequence anatomy
5. shadow management policies
"""

from __future__ import annotations

import argparse
import bisect
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
from tools.s34_buy_spike_repair_tests import mark_minmax  # noqa: E402
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
    pct,
    spike_thresholds,
    summary,
)


OUT_JSON = OUT_DIR / "S34_V02_ALPHA_NAVIGATION_OVERLAY.json"
OUT_MD = OUT_DIR / "S34_V02_ALPHA_NAVIGATION_OVERLAY.md"

RULE_NAME = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
THRESHOLD_USD = 200_000.0
VDEPTH_MIN_BPS = 28.0
VDEPTH_MAX_BPS = 40.0
PRIOR4H_LT_BPS = -50.0
MIN_BID_DEPTH_USD = 135_423.8
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30
INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
REPLACE_WAIT_SEC = 300
HORIZON_SEC = 2 * 3600
DEFAULT_CROSS_MARGIN_BPS = 2.0


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
        "win_rate": round(sum(1 for x in xs if x > 0.0) / len(xs), 3),
        "t3r": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "min": r1(min(xs)),
        "max": r1(max(xs)),
    }


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def book_exit_price(conn: sqlite3.Connection, ts_ms: int) -> float | None:
    row = conn.execute(
        """
        SELECT bid_price
        FROM book_ticker
        WHERE symbol='ETHUSDT' AND ts_ms>=?
        ORDER BY ts_ms ASC
        LIMIT 1
        """,
        (int(ts_ms),),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def find_v02_fill(
    marks: MarkIndex,
    anchor_mark_ts_ms: int,
    anchor_mark_price: float,
    *,
    cross_margin_bps: float,
    max_horizon_sec: int,
) -> dict[str, Any] | None:
    first_limit = maker_limit_price(anchor_mark_price, "LONG", INITIAL_OFFSET_BPS)
    first_required = first_limit * (1.0 - float(cross_margin_bps) / 10_000.0)
    repl_limit = maker_limit_price(anchor_mark_price, "LONG", REPLACE_OFFSET_BPS)
    repl_required = repl_limit * (1.0 - float(cross_margin_bps) / 10_000.0)
    deadline = int(anchor_mark_ts_ms) + int(max_horizon_sec) * 1000
    replace_ts = int(anchor_mark_ts_ms) + REPLACE_WAIT_SEC * 1000
    for ts_ms, px in marks.slice_range(int(anchor_mark_ts_ms), deadline):
        if int(ts_ms) <= int(anchor_mark_ts_ms):
            continue
        if int(ts_ms) <= replace_ts:
            if float(px) <= first_required:
                return {"fill_ts_ms": int(ts_ms), "entry_price": float(first_limit), "offset_bps": INITIAL_OFFSET_BPS}
        elif float(px) <= repl_required:
            return {"fill_ts_ms": int(ts_ms), "entry_price": float(repl_limit), "offset_bps": REPLACE_OFFSET_BPS}
    return None


def load_buy_spike_minutes(
    liq_1m: dict[int, dict[str, float]],
    threshold: float,
) -> list[dict[str, Any]]:
    events = []
    for ts in sorted(liq_1m):
        n = float(liq_1m.get(ts, {}).get("BUY", 0.0))
        if n >= threshold:
            events.append({"ts": int(ts), "notional": n})
    return events


def has_event_between(ts_list: list[int], start_ms: int, end_ms: int) -> bool:
    i = bisect.bisect_left(ts_list, int(start_ms))
    return i < len(ts_list) and ts_list[i] <= int(end_ms)


def event_count_between(ts_list: list[int], start_ms: int, end_ms: int) -> int:
    a = bisect.bisect_left(ts_list, int(start_ms))
    b = bisect.bisect_right(ts_list, int(end_ms))
    return max(0, b - a)


def nav_at(nav_by_ts: dict[int, dict[str, Any]], ts_ms: int) -> dict[str, Any]:
    return nav_by_ts.get((int(ts_ms) // MINUTE) * MINUTE) or {}


def nav_score(nav_by_ts: dict[int, dict[str, Any]], ts_ms: int) -> int:
    return int(nav_at(nav_by_ts, ts_ms).get("score", 0))


def nav_bucket(nav_by_ts: dict[int, dict[str, Any]], ts_ms: int) -> str:
    return str(nav_at(nav_by_ts, ts_ms).get("bucket", "NA"))


def build_live_like_trades(
    conn: sqlite3.Connection,
    *,
    start_ms: int,
    end_ms: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
    cross_margin_bps: float,
) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    liqs = load_liquidations(conn, SYMBOL, LIQ_SIDE, start_ms, end_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    trades: list[dict[str, Any]] = []
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
        anchor_mark = marks.at_or_after(int(anchor.anchor_ts_ms))
        if not anchor_mark:
            continue
        fill = find_v02_fill(
            marks,
            int(anchor_mark[0]),
            float(anchor_mark[1]),
            cross_margin_bps=float(cross_margin_bps),
            max_horizon_sec=HORIZON_SEC,
        )
        if not fill:
            trades.append(
                {
                    "status": "NO_MAKER_FILL",
                    "anchor_ts_ms": int(anchor.anchor_ts_ms),
                    "anchor_utc": iso_ms(anchor.anchor_ts_ms),
                    "vdepth_bps": r1(depth),
                    "prior4h_bps": r1(prior4h),
                    "bid_depth_usd": r1(float(book["bid_depth_usd"])),
                    "spread_bps": r1(float(book["spread_bps"])),
                    "book_imbalance": round(float(book["book_imbalance"]), 4),
                }
            )
            continue
        fill_ts = int(fill["fill_ts_ms"])
        entry = float(fill["entry_price"])
        exit_ts = fill_ts + HORIZON_SEC * 1000
        exit_px = book_exit_price(conn, exit_ts)
        if exit_px is None:
            continue
        gross_2h = signed_return_bps("LONG", entry, float(exit_px))
        trades.append(
            {
                "status": "FILLED",
                "rule": RULE_NAME,
                "anchor_ts_ms": int(anchor.anchor_ts_ms),
                "anchor_utc": iso_ms(anchor.anchor_ts_ms),
                "fill_ts_ms": fill_ts,
                "fill_utc": iso_ms(fill_ts),
                "fill_delay_sec": round((fill_ts - int(anchor.anchor_ts_ms)) / 1000.0, 1),
                "entry_price": entry,
                "entry_offset_bps": float(fill["offset_bps"]),
                "exit_2h_ts_ms": int(exit_ts),
                "exit_2h_price": float(exit_px),
                "net_2h_bps": float(gross_2h) - float(maker_fee_bps) - float(taker_fee_bps),
                "vdepth_bps": r1(depth),
                "prior4h_bps": r1(prior4h),
                "bid_depth_usd": r1(float(book["bid_depth_usd"])),
                "ask_depth_usd": r1(float(book["ask_depth_usd"])),
                "spread_bps": r1(float(book["spread_bps"])),
                "book_imbalance": round(float(book["book_imbalance"]), 4),
                "month": month_of(fill_ts),
            }
        )
    return sorted(trades, key=lambda r: int(r.get("fill_ts_ms", r.get("anchor_ts_ms", 0))))


def trade_return_at(conn: sqlite3.Connection, trade: dict[str, Any], horizon_sec: int, maker_fee_bps: float, taker_fee_bps: float) -> float | None:
    if trade.get("status") != "FILLED":
        return None
    px = book_exit_price(conn, int(trade["fill_ts_ms"]) + int(horizon_sec) * 1000)
    if px is None:
        return None
    return signed_return_bps("LONG", float(trade["entry_price"]), float(px)) - float(maker_fee_bps) - float(taker_fee_bps)


def trade_path_stats(marks: MarkIndex, trade: dict[str, Any], horizon_sec: int) -> dict[str, Any]:
    if trade.get("status") != "FILLED":
        return {}
    start = int(trade["fill_ts_ms"])
    end = start + int(horizon_sec) * 1000
    entry = float(trade["entry_price"])
    path = [(ts, px) for ts, px in marks.slice_range(start, end) if int(ts) >= start]
    if not path:
        return {"mfe_bps": None, "mae_bps": None, "mfe_time_sec": None, "mae_time_sec": None}
    rets = [(int(ts), signed_return_bps("LONG", entry, float(px))) for ts, px in path]
    mfe_ts, mfe = max(rets, key=lambda x: x[1])
    mae_ts, mae = min(rets, key=lambda x: x[1])
    return {
        "mfe_bps": r1(float(mfe)),
        "mae_bps": r1(float(mae)),
        "mfe_time_sec": round((int(mfe_ts) - start) / 1000.0, 1),
        "mae_time_sec": round((int(mae_ts) - start) / 1000.0, 1),
    }


def group_by(trades: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for val in sorted({str(t.get(key, False)) for t in trades}):
        subset = [t for t in trades if str(t.get(key, False)) == val]
        out[val] = metrics([t.get("net_2h_bps") for t in subset])
    return out


def state_sequence(nav_by_ts: dict[int, dict[str, Any]], fill_ts: int, minutes: int = 5) -> str:
    buckets = []
    for i in range(minutes + 1):
        score = nav_score(nav_by_ts, int(fill_ts) + i * MINUTE)
        if score >= 8:
            buckets.append("H")
        elif score >= 5:
            buckets.append("M")
        else:
            buckets.append("L")
    return "".join(buckets)


def make_trade_tags(
    trade: dict[str, Any],
    *,
    nav_by_ts: dict[int, dict[str, Any]],
    buy_spike_ts: list[int],
    buy_spike_notional_by_ts: dict[int, float],
    extreme_buy_threshold: float,
) -> dict[str, Any]:
    fill_ts = int(trade["fill_ts_ms"])
    anchor_ts = int(trade["anchor_ts_ms"])
    tags: dict[str, Any] = {}
    for m in (5, 15, 30):
        tags[f"buy_spike_pre_{m}m"] = has_event_between(buy_spike_ts, anchor_ts - m * MINUTE, anchor_ts)
    for m in (1, 5, 15):
        tags[f"buy_spike_post_{m}m"] = has_event_between(buy_spike_ts, fill_ts, fill_ts + m * MINUTE)
        tags[f"buy_spike_count_post_{m}m"] = event_count_between(buy_spike_ts, fill_ts, fill_ts + m * MINUTE)
    tags["nav_score_fill"] = nav_score(nav_by_ts, fill_ts)
    tags["nav_high_fill"] = tags["nav_score_fill"] >= 7
    tags["nav_score_5m"] = nav_score(nav_by_ts, fill_ts + 5 * MINUTE)
    tags["nav_high_holds_5m"] = sum(1 for i in range(6) if nav_score(nav_by_ts, fill_ts + i * MINUTE) >= 7) >= 4
    tags["state_sequence_5m"] = state_sequence(nav_by_ts, fill_ts, 5)
    tags["liquidity_thin"] = float(trade.get("bid_depth_usd", 0.0)) < 100_000 or float(trade.get("spread_bps", 0.0)) > 0.20
    tags["book_support"] = float(trade.get("bid_depth_usd", 0.0)) >= MIN_BID_DEPTH_USD and float(trade.get("book_imbalance", 0.0)) >= 0.0
    extreme = False
    i = bisect.bisect_left(buy_spike_ts, fill_ts)
    while i < len(buy_spike_ts) and buy_spike_ts[i] <= fill_ts + 15 * MINUTE:
        if buy_spike_notional_by_ts.get(buy_spike_ts[i], 0.0) >= extreme_buy_threshold:
            extreme = True
            break
        i += 1
    tags["exhaustion_risk"] = extreme or tags["buy_spike_count_post_15m"] >= 2 or tags["nav_score_fill"] <= 4
    tags["squeeze_active"] = tags["buy_spike_post_5m"] or tags["nav_high_holds_5m"]
    tags["rebound_confirmed_5m"] = tags["nav_high_holds_5m"] and tags["buy_spike_post_15m"]
    if tags["liquidity_thin"] or tags["exhaustion_risk"]:
        tags["nav_recommendation"] = "SCALP_OR_REDUCE"
    elif tags["rebound_confirmed_5m"]:
        tags["nav_recommendation"] = "HOLD_ALLOWED"
    elif tags["squeeze_active"]:
        tags["nav_recommendation"] = "SCALP_ONLY"
    else:
        tags["nav_recommendation"] = "BASELINE"
    return tags


def shadow_policy_return(
    conn: sqlite3.Connection,
    trade: dict[str, Any],
    policy: str,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> float | None:
    if trade.get("status") != "FILLED":
        return None
    rec = str(trade.get("nav_recommendation"))
    if policy == "baseline_2h":
        return trade_return_at(conn, trade, HORIZON_SEC, maker_fee_bps, taker_fee_bps)
    if policy == "scalp_or_reduce_5m":
        h = 300 if rec in {"SCALP_OR_REDUCE", "SCALP_ONLY"} else HORIZON_SEC
        return trade_return_at(conn, trade, h, maker_fee_bps, taker_fee_bps)
    if policy == "confirmed_hold_else_15m":
        h = HORIZON_SEC if rec == "HOLD_ALLOWED" else 900
        return trade_return_at(conn, trade, h, maker_fee_bps, taker_fee_bps)
    if policy == "danger_exit_1m_else_2h":
        h = 60 if bool(trade.get("exhaustion_risk") or trade.get("liquidity_thin")) else HORIZON_SEC
        return trade_return_at(conn, trade, h, maker_fee_bps, taker_fee_bps)
    return None


def render(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 V02 Alpha Navigation Overlay",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## 1. Live-Like V02 Fill Set",
        "",
        f"- all anchors: `{result['fill_set']['anchors_total']}`",
        f"- filled: `{result['fill_set']['filled_n']}`",
        f"- no maker fill: `{result['fill_set']['nofill_n']}`",
        f"- baseline 2h: `{result['fill_set']['baseline_2h']}`",
        "",
        "## 2. BUY Spike Overlay",
        "",
    ]
    for k, v in result["buy_spike_overlay"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 3. Scalp Horizon Decomposition", ""]
    for k, v in result["scalp_horizons"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 4. MFE/MAE Path", "", f"`{result['mfe_mae']}`", "", "## 5. Danger / Navigation Tags", ""]
    for k, v in result["navigation_tags"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 6. State Sequence Anatomy", ""]
    for k, v in result["state_sequences_top"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 7. Shadow Management Policies", ""]
    for k, v in result["shadow_management"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 8. Interpretation", "", result["interpretation"]]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--maker-fee-bps", type=float, default=-0.5)
    ap.add_argument("--taker-fee-bps", type=float, default=3.05)
    ap.add_argument("--cross-margin-bps", type=float, default=DEFAULT_CROSS_MARGIN_BPS)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE

        eth_mark_1m = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")
        btc_mark_1m = load_mark_1m(conn, start_ms, end_ms, "BTCUSDT")
        buckets = sorted(eth_mark_1m)
        liq_1m = load_liq_1m(conn, start_ms, end_ms)
        flow_1m = load_flow_1m(conn, start_ms, end_ms)
        book_1m = load_book_for_buckets(conn, buckets)
        buckets = sorted(set(eth_mark_1m) & set(book_1m))
        nav = build_nav(buckets, book_1m, liq_1m, flow_1m, btc_mark_1m)
        nav_by_ts = {int(n["ts"]): n for n in nav}
        th = spike_thresholds(liq_1m)
        buy_spike_threshold = float(th["BUY"]["primary_threshold"])
        buy_extreme_threshold = float(th["BUY"]["p99_nonzero"])
        buy_spikes = load_buy_spike_minutes(liq_1m, buy_spike_threshold)
        buy_spike_ts = [int(e["ts"]) for e in buy_spikes]
        buy_spike_notional_by_ts = {int(e["ts"]): float(e["notional"]) for e in buy_spikes}

        trades_all = build_live_like_trades(
            conn,
            start_ms=start_ms,
            end_ms=end_ms,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            cross_margin_bps=float(args.cross_margin_bps),
        )
        filled = [t for t in trades_all if t.get("status") == "FILLED"]
        marks = load_mark_index(conn, SYMBOL)

        for t in filled:
            t.update(
                make_trade_tags(
                    t,
                    nav_by_ts=nav_by_ts,
                    buy_spike_ts=buy_spike_ts,
                    buy_spike_notional_by_ts=buy_spike_notional_by_ts,
                    extreme_buy_threshold=buy_extreme_threshold,
                )
            )
            t.update(trade_path_stats(marks, t, HORIZON_SEC))

        baseline = metrics([t.get("net_2h_bps") for t in filled])

        overlay: dict[str, Any] = {}
        for key in (
            "buy_spike_pre_5m",
            "buy_spike_pre_15m",
            "buy_spike_pre_30m",
            "buy_spike_post_1m",
            "buy_spike_post_5m",
            "buy_spike_post_15m",
        ):
            overlay[key] = group_by(filled, key)

        horizons = {
            "15s": 15,
            "30s": 30,
            "60s": 60,
            "2m": 120,
            "5m": 300,
            "15m": 900,
            "60m": 3600,
            "2h": HORIZON_SEC,
        }
        scalp_horizons = {
            k: metrics([trade_return_at(conn, t, sec, float(args.maker_fee_bps), float(args.taker_fee_bps)) for t in filled])
            for k, sec in horizons.items()
        }

        mfe_mae = {
            "mfe": metrics([t.get("mfe_bps") for t in filled]),
            "mae": metrics([t.get("mae_bps") for t in filled]),
            "mfe_time_sec_median": r1(pctile(clean([t.get("mfe_time_sec") for t in filled]), 0.5)) if filled else None,
            "mae_time_sec_median": r1(pctile(clean([t.get("mae_time_sec") for t in filled]), 0.5)) if filled else None,
        }

        nav_tags: dict[str, Any] = {}
        for key in (
            "nav_high_fill",
            "nav_high_holds_5m",
            "liquidity_thin",
            "book_support",
            "exhaustion_risk",
            "squeeze_active",
            "rebound_confirmed_5m",
            "nav_recommendation",
        ):
            nav_tags[key] = group_by(filled, key)

        seq_counter = Counter(str(t.get("state_sequence_5m")) for t in filled)
        state_sequences_top: dict[str, Any] = {}
        for seq, _ in seq_counter.most_common(10):
            subset = [t for t in filled if str(t.get("state_sequence_5m")) == seq]
            state_sequences_top[seq] = metrics([t.get("net_2h_bps") for t in subset])

        shadow_management = {
            policy: metrics([shadow_policy_return(conn, t, policy, float(args.maker_fee_bps), float(args.taker_fee_bps)) for t in filled])
            for policy in (
                "baseline_2h",
                "scalp_or_reduce_5m",
                "confirmed_hold_else_15m",
                "danger_exit_1m_else_2h",
            )
        }

    interp_bits = []
    if scalp_horizons.get("5m", {}).get("t3r", 0) > baseline.get("t3r", 0):
        interp_bits.append("5m scalp improves T3R versus baseline 2h.")
    else:
        interp_bits.append("Scalp horizons do not improve robust T3R versus baseline 2h.")
    if nav_tags.get("exhaustion_risk", {}).get("True", {}).get("t3r", 0) < nav_tags.get("exhaustion_risk", {}).get("False", {}).get("t3r", 0):
        interp_bits.append("EXHAUSTION_RISK behaves like a risk/veto label, not an entry signal.")
    if shadow_management["scalp_or_reduce_5m"]["t3r"] > baseline["t3r"]:
        interp_bits.append("Navigation-based scalp/reduce policy is a management lead.")
    else:
        interp_bits.append("Navigation management policies are not yet better than baseline.")

    result = {
        "generated_at_utc": utc_now(),
        "scope": {
            "rule": RULE_NAME,
            "days": int(args.days),
            "start_utc": iso_ms(start_ms),
            "end_utc": iso_ms(end_ms),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "cross_margin_bps": float(args.cross_margin_bps),
            "buy_spike_threshold": r1(buy_spike_threshold),
            "buy_extreme_threshold": r1(buy_extreme_threshold),
            "note": "Research-only. No live executor/config/order logic touched.",
        },
        "fill_set": {
            "anchors_total": len(trades_all),
            "filled_n": len(filled),
            "nofill_n": sum(1 for t in trades_all if t.get("status") == "NO_MAKER_FILL"),
            "baseline_2h": baseline,
        },
        "buy_spike_overlay": overlay,
        "scalp_horizons": scalp_horizons,
        "mfe_mae": mfe_mae,
        "navigation_tags": nav_tags,
        "state_sequences_top": state_sequences_top,
        "shadow_management": shadow_management,
        "sample_filled_trades": filled[:20],
        "interpretation": " ".join(interp_bits),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    render(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
