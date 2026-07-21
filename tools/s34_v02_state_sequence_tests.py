"""S34 V02 state-sequence scalp/swing tests.

Research-only. Tests whether the observed NAV + liquidation spike behavior is a
tradeable state sequence rather than a single event.
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

from tools.s34_v02_nav_spike_tests import (
    DB_PATH,
    MINUTE,
    OUT_DIR,
    Series,
    bps,
    build_nav,
    latest_ts,
    load_book_for_buckets,
    load_flow_1m,
    load_liq_1m,
    load_mark_1m,
    make_series,
    nonoverlap,
    pct,
    spike_thresholds,
    summary,
    utc_ms,
)

OUT_JSON = OUT_DIR / "S34_V02_STATE_SEQUENCE_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_STATE_SEQUENCE_TESTS.md"


def corr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    mx = mean([p[0] for p in pairs])
    my = mean([p[1] for p in pairs])
    vx = sum((p[0] - mx) ** 2 for p in pairs)
    vy = sum((p[1] - my) ** 2 for p in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return round(cov / math.sqrt(vx * vy), 3)


def top3_removed(vals: list[float]) -> float | None:
    clean = sorted([float(v) for v in vals if v is not None and math.isfinite(v)], reverse=True)
    if len(clean) <= 3:
        return None
    return round(sum(clean[3:]), 1)


def fee_net(vals: list[float], round_trip_fee_bps: float) -> list[float]:
    return [float(v) - round_trip_fee_bps for v in vals if v is not None and math.isfinite(v)]


def ret(series: Series, t: int, h_min: int) -> float | None:
    return series.ret_after(t, h_min * MINUTE)


def price(series: Series, t: int) -> float | None:
    return series.price_at_or_after(t)


def build_events(
    buckets: list[int],
    liq: dict[int, dict[str, float]],
    nav_by_ts: dict[int, dict[str, Any]],
    eth_series: Series,
    btc_series: Series,
    thresholds: dict[str, dict[str, float]],
    cooldown_min: int,
) -> list[dict[str, Any]]:
    raw = []
    for ts in buckets:
        buy_n = float(liq.get(ts, {}).get("BUY", 0.0))
        if buy_n < thresholds["BUY"]["primary_threshold"]:
            continue
        nav = nav_by_ts.get(ts)
        prev5 = nav_by_ts.get(ts - 5 * MINUTE)
        prev1_px = price(eth_series, ts - MINUTE)
        now_px = price(eth_series, ts)
        pre15 = eth_series.ret_after(ts - 15 * MINUTE, 15 * MINUTE)
        btc1 = btc_series.ret_after(ts - MINUTE, MINUTE)
        raw.append(
            {
                "ts": ts,
                "utc": utc_ms(ts),
                "side": "BUY",
                "notional": round(buy_n, 1),
                "nav_score": int((nav or {}).get("score", 0)),
                "nav_bucket": (nav or {}).get("bucket"),
                "tags": (nav or {}).get("tags", []),
                "warnings": (nav or {}).get("warnings", []),
                "nav_delta_5m": int((nav or {}).get("score", 0)) - int((prev5 or {}).get("score", 0)),
                "nav_high_prev5": sum(1 for j in range(5) if int((nav_by_ts.get(ts - j * MINUTE) or {}).get("score", 0)) >= 7),
                "price_impulse_1m_bps": bps(prev1_px, now_px) if prev1_px and now_px else None,
                "pre15_bps": pre15,
                "btc1_bps": btc1,
                "bid_thin": "BID_THIN" in ((nav or {}).get("warnings", []) or []),
                "nav_falling": (int((nav or {}).get("score", 0)) - int((prev5 or {}).get("score", 0))) < 0,
            }
        )
    return nonoverlap(raw, cooldown_min * MINUTE)


def group_summary(events: list[dict[str, Any]], eth_series: Series, *, delay_min: int, horizon_min: int, fee_bps: float) -> dict[str, Any]:
    vals = []
    for e in events:
        r = ret(eth_series, int(e["ts"]) + delay_min * MINUTE, horizon_min)
        if r is not None:
            vals.append(r)
    return {"gross": summary(vals), "fee_net": summary(fee_net(vals, fee_bps))}


def invalidation_exit(
    events: list[dict[str, Any]],
    eth_series: Series,
    nav_by_ts: dict[int, dict[str, Any]],
    *,
    horizon_min: int,
    fee_bps: float,
) -> dict[str, Any]:
    vals = []
    exits = {"TIME": 0, "NAV_LOW": 0, "BID_THIN": 0, "BTC_DUMPING": 0}
    for e in events:
        entry_t = int(e["ts"])
        entry_px = price(eth_series, entry_t)
        if not entry_px:
            continue
        exit_t = entry_t + horizon_min * MINUTE
        reason = "TIME"
        for j in range(1, horizon_min + 1):
            t = entry_t + j * MINUTE
            n = nav_by_ts.get(t) or {}
            warnings = n.get("warnings", []) or []
            if int(n.get("score", 10)) < 5:
                exit_t = t
                reason = "NAV_LOW"
                break
            if "BID_THIN" in warnings:
                exit_t = t
                reason = "BID_THIN"
                break
            if "BTC_DUMPING" in warnings:
                exit_t = t
                reason = "BTC_DUMPING"
                break
        exit_px = price(eth_series, exit_t)
        r = bps(entry_px, exit_px) if exit_px else None
        if r is not None:
            vals.append(r)
            exits[reason] = exits.get(reason, 0) + 1
    return {"gross": summary(vals), "fee_net": summary(fee_net(vals, fee_bps)), "exit_mix": exits}


def split_holdout(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda e: int(e["ts"]))
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cooldown-min", type=int, default=5)
    ap.add_argument("--fee-bps", type=float, default=6.1, help="Round-trip taker-like cost for scalp tests.")
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    with sqlite3.connect(args.db) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE
        eth_mark = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")
        btc_mark = load_mark_1m(conn, start_ms, end_ms, "BTCUSDT")
        buckets = sorted(eth_mark)
        liq = load_liq_1m(conn, start_ms, end_ms)
        flow = load_flow_1m(conn, start_ms, end_ms)
        book = load_book_for_buckets(conn, buckets)

    buckets = sorted(set(eth_mark) & set(book))
    nav = build_nav(buckets, book, liq, flow, btc_mark)
    nav_by_ts = {int(n["ts"]): n for n in nav}
    eth_series = make_series(eth_mark)
    btc_series = make_series(btc_mark)
    thresholds = spike_thresholds(liq)
    events = build_events(buckets, liq, nav_by_ts, eth_series, btc_series, thresholds, int(args.cooldown_min))

    groups = {
        "ALL_BUY_SPIKE": events,
        "NAV_HIGH_BUY_SPIKE": [e for e in events if int(e["nav_score"]) >= 7],
        "NAV_HIGH_PERSIST_3OF5": [e for e in events if int(e["nav_score"]) >= 7 and int(e["nav_high_prev5"]) >= 3],
        "NAV_RISING_BUY_SPIKE": [e for e in events if int(e["nav_delta_5m"]) > 0],
        "NAV_HIGH_RISING": [e for e in events if int(e["nav_score"]) >= 7 and int(e["nav_delta_5m"]) > 0],
        "PRICE_IMPULSE_POSITIVE": [e for e in events if (e.get("price_impulse_1m_bps") is not None and float(e["price_impulse_1m_bps"]) > 0)],
        "NAV_HIGH_AND_IMPULSE": [e for e in events if int(e["nav_score"]) >= 7 and (e.get("price_impulse_1m_bps") is not None and float(e["price_impulse_1m_bps"]) > 0)],
        "EXHAUSTION_RISK": [
            e
            for e in events
            if float(e["notional"]) >= thresholds["BUY"]["p99_nonzero"]
            or (e.get("pre15_bps") is not None and float(e["pre15_bps"]) > 50.0)
            or bool(e.get("nav_falling"))
            or (e.get("btc1_bps") is not None and float(e["btc1_bps"]) < 0.0)
            or bool(e.get("bid_thin"))
        ],
    }
    groups["CLEAN_NOT_EXHAUSTION"] = [e for e in events if e not in groups["EXHAUSTION_RISK"]]

    horizons = [1, 5, 15, 60, 120]
    delays = [0, 1, 2, 5]
    group_results: dict[str, Any] = {}
    for name, evs in groups.items():
        group_results[name] = {
            "n": len(evs),
            "horizons_delay0": {f"{h}m": group_summary(evs, eth_series, delay_min=0, horizon_min=h, fee_bps=float(args.fee_bps)) for h in horizons},
            "delays_15m": {f"{d}m": group_summary(evs, eth_series, delay_min=d, horizon_min=15, fee_bps=float(args.fee_bps)) for d in delays},
            "invalidation_15m": invalidation_exit(evs, eth_series, nav_by_ts, horizon_min=15, fee_bps=float(args.fee_bps)),
        }
        cal, hold = split_holdout(evs)
        group_results[name]["chronological_split_15m_fee_net"] = {
            "cal": group_summary(cal, eth_series, delay_min=0, horizon_min=15, fee_bps=float(args.fee_bps))["fee_net"],
            "hold": group_summary(hold, eth_series, delay_min=0, horizon_min=15, fee_bps=float(args.fee_bps))["fee_net"],
        }

    # Scalp/swing relationship.
    rel_rows = []
    for e in events:
        t = int(e["ts"])
        r5 = ret(eth_series, t, 5)
        r15 = ret(eth_series, t, 15)
        r120 = ret(eth_series, t, 120)
        if r5 is not None and r15 is not None and r120 is not None:
            rel_rows.append((r5, r15, r120))
    scalp_swing = {
        "n": len(rel_rows),
        "corr_5m_120m": corr([r[0] for r in rel_rows], [r[2] for r in rel_rows]),
        "corr_15m_120m": corr([r[1] for r in rel_rows], [r[2] for r in rel_rows]),
        "sign_matrix_15m_vs_120m": {
            "both_pos": sum(1 for _, r15, r120 in rel_rows if r15 > 0 and r120 > 0),
            "scalp_pos_swing_neg": sum(1 for _, r15, r120 in rel_rows if r15 > 0 and r120 <= 0),
            "scalp_neg_swing_pos": sum(1 for _, r15, r120 in rel_rows if r15 <= 0 and r120 > 0),
            "both_neg": sum(1 for _, r15, r120 in rel_rows if r15 <= 0 and r120 <= 0),
        },
    }

    # Promotion candidates: strict screen, research-only.
    promotion = []
    for name, res in group_results.items():
        hold = res["chronological_split_15m_fee_net"]["hold"]
        full = res["horizons_delay0"]["15m"]["fee_net"]
        if res["n"] >= 40 and full["sum"] > 0 and (full["t3r"] or -1) > 0 and hold["sum"] > 0 and (hold["t3r"] or -1) > 0:
            promotion.append(
                {
                    "name": name,
                    "n": res["n"],
                    "full_15m_fee_net": full,
                    "hold_15m_fee_net": hold,
                }
            )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "symbol": "ETHUSDT",
            "days": int(args.days),
            "start_utc": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
            "nav_points": len(nav),
            "buy_spike_events": len(events),
            "round_trip_fee_bps": float(args.fee_bps),
            "spike_threshold": thresholds["BUY"],
            "note": "Research-only. One-minute state-sequence proxy, not live order logic.",
        },
        "groups": group_results,
        "scalp_vs_swing": scalp_swing,
        "promotion_candidates": promotion,
        "sample_events": events[-10:],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# S34 V02 State Sequence Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: ETHUSDT last `{args.days}` days, BUY spike events `{len(events)}`, fee `{args.fee_bps}` bps.",
        f"BUY spike threshold: `{thresholds['BUY']['primary_threshold']}` notional/min.",
        "",
        "## Candidate Groups",
        "",
        "| group | N | 15m fee-net sum | median | WR | T3R | hold sum | hold T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, res in group_results.items():
        full = res["horizons_delay0"]["15m"]["fee_net"]
        hold = res["chronological_split_15m_fee_net"]["hold"]
        lines.append(
            f"| {name} | {res['n']} | {full['sum']} | {full['median']} | {full['win_rate']} | {full['t3r']} | {hold['sum']} | {hold['t3r']} |"
        )
    lines += ["", "## Scalp vs Swing", "", f"- {scalp_swing}", "", "## Entry Delay, Exits, Promotion", ""]
    for name in ("ALL_BUY_SPIKE", "NAV_HIGH_BUY_SPIKE", "NAV_HIGH_AND_IMPULSE", "CLEAN_NOT_EXHAUSTION"):
        res = group_results[name]
        lines.append(f"### {name}")
        lines.append(f"- delays 15m fee-net: `{ {k:v['fee_net'] for k,v in res['delays_15m'].items()} }`")
        lines.append(f"- invalidation 15m fee-net: `{res['invalidation_15m']['fee_net']}`, exits `{res['invalidation_15m']['exit_mix']}`")
        lines.append("")
    lines += [
        "## Promotion Screen",
        "",
        f"- candidates passing N>=40, full sum/T3R>0, hold sum/T3R>0: `{promotion}`",
        "",
        "## Notes",
        "",
        "- Research-only. Live executor/config/order logic untouched.",
        "- This is still a one-minute proxy. A passing candidate would need tick-level execution and forward paper before live.",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
