"""Live-readiness gauntlet for ETH BUY-liq spike LONG scalp.

Research-only. This tries to falsify the state-sequence candidate before any
paper/live promotion:
- running-notional threshold-cross instead of closed 1m buckets
- executable ask/bid entry and exit
- second-level delay sensitivity
- sub-minute window robustness
- fee/slippage stress
- walk-forward, permutation-null, stops, and regime splits
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
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_v02_nav_spike_tests import DB_PATH, MINUTE, OUT_DIR, bps, latest_ts, load_liq_1m, load_mark_1m, pct, summary

OUT_JSON = OUT_DIR / "S34_BUY_SPIKE_LIVE_GAUNTLET.json"
OUT_MD = OUT_DIR / "S34_BUY_SPIKE_LIVE_GAUNTLET.md"


def utc_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def fee_net(vals: list[float], fee_bps: float) -> list[float]:
    return [float(v) - float(fee_bps) for v in vals if v is not None and math.isfinite(v)]


def t3r(vals: list[float]) -> float | None:
    clean = sorted([float(v) for v in vals if v is not None and math.isfinite(v)], reverse=True)
    if len(clean) <= 3:
        return None
    return round(sum(clean[3:]), 1)


class MarkSeries:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self.ts = [int(r[0]) for r in rows]
        self.px = [float(r[1]) for r in rows]

    def at_or_after(self, ts_ms: int) -> float | None:
        i = bisect.bisect_left(self.ts, int(ts_ms))
        if i >= len(self.ts):
            return None
        return self.px[i]

    def ret(self, ts_ms: int, horizon_ms: int) -> float | None:
        a = self.at_or_after(ts_ms)
        b = self.at_or_after(ts_ms + horizon_ms)
        return bps(a, b) if a is not None and b is not None else None


def load_mark_raw(conn: sqlite3.Connection, start_ms: int, end_ms: int, symbol: str) -> MarkSeries:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    return MarkSeries([(int(r[0]), float(r[1])) for r in rows if r[1] is not None])


def load_buy_liqs(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY' AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms",
        (int(start_ms), int(end_ms)),
    ).fetchall()
    return [(int(r[0]), float(r[1] or 0.0)) for r in rows]


def rolling_sums_at_liqs(liqs: list[tuple[int, float]], window_ms: int) -> list[float]:
    q: deque[tuple[int, float]] = deque()
    acc = 0.0
    vals: list[float] = []
    for ts, n in liqs:
        while q and q[0][0] < ts - window_ms:
            acc -= q.popleft()[1]
        q.append((ts, n))
        acc += n
        vals.append(acc)
    return vals


def running_cross_events(
    liqs: list[tuple[int, float]],
    *,
    window_ms: int,
    threshold: float,
    cooldown_ms: int,
) -> list[dict[str, Any]]:
    q: deque[tuple[int, float]] = deque()
    acc = 0.0
    out: list[dict[str, Any]] = []
    last = -10**30
    was_above = False
    for ts, n in liqs:
        while q and q[0][0] < ts - window_ms:
            acc -= q.popleft()[1]
        before = acc
        q.append((ts, n))
        acc += n
        crossed = before < threshold <= acc
        if acc < threshold:
            was_above = False
        if crossed and not was_above and ts - last >= cooldown_ms:
            out.append({"ts": ts, "utc": utc_ms(ts), "running_notional": round(acc, 1), "liq_count": len(q)})
            last = ts
            was_above = True
    return out


def book_price(conn: sqlite3.Connection, ts_ms: int, side: str) -> float | None:
    col = "ask_price" if side == "LONG_ENTRY" else "bid_price"
    row = conn.execute(
        f"SELECT {col} FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (int(ts_ms),),
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return None


def executable_ret(conn: sqlite3.Connection, event_ts: int, *, delay_sec: int, horizon_sec: int) -> float | None:
    entry_t = int(event_ts) + int(delay_sec) * 1000
    exit_t = entry_t + int(horizon_sec) * 1000
    entry = book_price(conn, entry_t, "LONG_ENTRY")
    exit_px = book_price(conn, exit_t, "LONG_EXIT")
    if entry is None or exit_px is None:
        return None
    return bps(entry, exit_px)


def mark_path_stop(mark: MarkSeries, event_ts: int, *, horizon_min: int, sl_bps: float) -> float | None:
    entry = mark.at_or_after(event_ts)
    if entry is None:
        return None
    for m in range(1, horizon_min + 1):
        px = mark.at_or_after(event_ts + m * MINUTE)
        if px is None:
            return None
        r = bps(entry, px)
        if r is not None and r <= -float(sl_bps):
            return r
    final = mark.at_or_after(event_ts + horizon_min * MINUTE)
    return bps(entry, final) if final is not None else None


def split_folds(events: list[dict[str, Any]], folds: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda e: int(e["ts"]))
    if not ordered:
        return []
    out = []
    for i in range(folds):
        a = int(len(ordered) * i / folds)
        b = int(len(ordered) * (i + 1) / folds)
        out.append(ordered[a:b])
    return out


def ret_list_mark(mark: MarkSeries, events: list[dict[str, Any]], horizon_min: int, delay_sec: int = 0) -> list[float]:
    vals = []
    for e in events:
        r = mark.ret(int(e["ts"]) + delay_sec * 1000, horizon_min * MINUTE)
        if r is not None:
            vals.append(r)
    return vals


def draw_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# S34 BUY Spike Live-Readiness Gauntlet",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: `{result['scope']}`",
        "",
        "## 1. Knowable Running Threshold-Cross",
        "",
        "| window | threshold | N | 15m fee-net sum | median | WR | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, v in result["window_robustness"].items():
        s = v["delay0_15m_fee_net"]
        lines.append(f"| {k} | {v['threshold']} | {v['n']} | {s['sum']} | {s['median']} | {s['win_rate']} | {s['t3r']} |")
    lines += ["", "## 2. Second-Level Entry Delay, 60s Running Anchor", ""]
    for k, s in result["delay_sweep_60s"].items():
        lines.append(f"- {k}: N `{s['n']}`, sum `{s['sum']}`, median `{s['median']}`, WR `{s['win_rate']}`, T3R `{s['t3r']}`")
    lines += ["", "## 3. Fee / Slippage Stress, 60s Running Anchor", ""]
    for k, s in result["fee_stress_60s"].items():
        lines.append(f"- fee {k}: sum `{s['sum']}`, median `{s['median']}`, WR `{s['win_rate']}`, T3R `{s['t3r']}`")
    lines += ["", "## 4. Exit / Stop Robustness", ""]
    for k, s in result["exit_robustness_60s"].items():
        lines.append(f"- {k}: sum `{s['sum']}`, median `{s['median']}`, WR `{s['win_rate']}`, T3R `{s['t3r']}`, min `{s.get('min')}`")
    lines += ["", "## 5. Walk-Forward", ""]
    for row in result["walk_forward_60s"]:
        lines.append(f"- fold {row['fold']}: N `{row['n']}`, sum `{row['sum']}`, median `{row['median']}`, T3R `{row['t3r']}`")
    lines += ["", "## 6. Permutation Null", ""]
    lines.append(f"- `{result['permutation_null']}`")
    lines += ["", "## 7. Regime Splits", ""]
    for k, s in result["regime_splits_60s"].items():
        lines.append(f"- {k}: N `{s['n']}`, sum `{s['sum']}`, median `{s['median']}`, WR `{s['win_rate']}`, T3R `{s['t3r']}`")
    lines += ["", "## Verdict", "", result["verdict"]]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--permutations", type=int, default=300)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    random.seed(34)
    with sqlite3.connect(args.db) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE
        liqs = load_buy_liqs(conn, start_ms, end_ms)
        mark = load_mark_raw(conn, start_ms, end_ms + 3 * 60 * MINUTE, "ETHUSDT")
        btc_mark = load_mark_raw(conn, start_ms, end_ms + 3 * 60 * MINUTE, "BTCUSDT")
        liq_1m = load_liq_1m(conn, start_ms, end_ms)
        eth_mark_1m = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")

        windows_sec = [10, 15, 30, 60]
        window_robustness: dict[str, Any] = {}
        events_by_window: dict[int, list[dict[str, Any]]] = {}
        for w in windows_sec:
            sums = rolling_sums_at_liqs(liqs, w * 1000)
            threshold = max(50_000.0, pct(sums, 0.95))
            events = running_cross_events(liqs, window_ms=w * 1000, threshold=threshold, cooldown_ms=5 * MINUTE)
            events_by_window[w] = events
            gross = [executable_ret(conn, int(e["ts"]), delay_sec=0, horizon_sec=15 * 60) for e in events]
            gross = [v for v in gross if v is not None]
            net = fee_net(gross, 6.1)
            window_robustness[f"{w}s"] = {
                "threshold": round(threshold, 1),
                "n": len(events),
                "delay0_15m_fee_net": summary(net),
            }

        base_events = events_by_window[60]
        delay_sweep = {}
        for d in [0, 2, 5, 10, 30, 60]:
            gross = [executable_ret(conn, int(e["ts"]), delay_sec=d, horizon_sec=15 * 60) for e in base_events]
            delay_sweep[f"{d}s"] = summary(fee_net([v for v in gross if v is not None], 6.1))

        fees = {}
        base_gross = [executable_ret(conn, int(e["ts"]), delay_sec=0, horizon_sec=15 * 60) for e in base_events]
        base_gross = [v for v in base_gross if v is not None]
        for f in [6.1, 10.0, 15.0, 20.0, 30.0]:
            fees[f"{f}bps"] = summary(fee_net(base_gross, f))

    # Everything below uses loaded mark series, no DB required.
    exit_robustness = {
        "fixed_5m_fee6.1": summary(fee_net(ret_list_mark(mark, base_events, 5), 6.1)),
        "fixed_10m_fee6.1": summary(fee_net(ret_list_mark(mark, base_events, 10), 6.1)),
        "fixed_15m_fee6.1": summary(fee_net(ret_list_mark(mark, base_events, 15), 6.1)),
        "fixed_20m_fee6.1": summary(fee_net(ret_list_mark(mark, base_events, 20), 6.1)),
        "fixed_30m_fee6.1": summary(fee_net(ret_list_mark(mark, base_events, 30), 6.1)),
    }
    for sl in [30, 50, 75, 100]:
        vals = [mark_path_stop(mark, int(e["ts"]), horizon_min=15, sl_bps=sl) for e in base_events]
        exit_robustness[f"SL{sl}_15m_fee6.1"] = summary(fee_net([v for v in vals if v is not None], 6.1))

    wf = []
    for i, fold in enumerate(split_folds(base_events, 5), start=1):
        vals = fee_net(ret_list_mark(mark, fold, 15), 6.1)
        s = summary(vals)
        wf.append({"fold": i, **s})

    # Multiple-comparison-ish null: sample random non-overlapping minute anchors,
    # then compare max 15m fee-net sum across candidate Ns.
    minute_buckets = sorted(eth_mark_1m)
    candidate_ns = [max(1, len(events_by_window[w])) for w in windows_sec]
    real_max = max(window_robustness[f"{w}s"]["delay0_15m_fee_net"]["sum"] for w in windows_sec)
    null_maxes = []
    for _ in range(int(args.permutations)):
        max_sum = -10**9
        for n in candidate_ns:
            sample = sorted(random.sample(minute_buckets[:-20], min(n, len(minute_buckets[:-20]))))
            vals = [mark.ret(ts, 15 * MINUTE) for ts in sample]
            s = sum(fee_net([v for v in vals if v is not None], 6.1))
            max_sum = max(max_sum, s)
        null_maxes.append(max_sum)
    p_right = sum(1 for x in null_maxes if x >= real_max) / len(null_maxes) if null_maxes else None
    perm = {
        "permutations": int(args.permutations),
        "real_max_sum": round(real_max, 1),
        "null_p95": round(pct(null_maxes, 0.95), 1) if null_maxes else None,
        "null_p99": round(pct(null_maxes, 0.99), 1) if null_maxes else None,
        "p_right": round(p_right, 4) if p_right is not None else None,
    }

    regimes: dict[str, list[dict[str, Any]]] = {
        "btc_1h_up": [],
        "btc_1h_down": [],
        "eth_pre15_up": [],
        "eth_pre15_down": [],
        "p99_running_notional": [],
        "non_p99_running_notional": [],
    }
    rn_vals = [float(e.get("running_notional", 0.0)) for e in base_events]
    rn_p99 = pct(rn_vals, 0.99)
    for e in base_events:
        t = int(e["ts"])
        btc1h = btc_mark.ret(t - 60 * MINUTE, 60 * MINUTE)
        eth15 = mark.ret(t - 15 * MINUTE, 15 * MINUTE)
        if btc1h is not None and btc1h >= 0:
            regimes["btc_1h_up"].append(e)
        if btc1h is not None and btc1h < 0:
            regimes["btc_1h_down"].append(e)
        if eth15 is not None and eth15 >= 0:
            regimes["eth_pre15_up"].append(e)
        if eth15 is not None and eth15 < 0:
            regimes["eth_pre15_down"].append(e)
        if float(e.get("running_notional", 0.0)) >= rn_p99:
            regimes["p99_running_notional"].append(e)
        else:
            regimes["non_p99_running_notional"].append(e)
    regime_splits = {k: summary(fee_net(ret_list_mark(mark, v, 15), 6.1)) for k, v in regimes.items()}

    pass_reasons = []
    fail_reasons = []
    base = window_robustness["60s"]["delay0_15m_fee_net"]
    if base["sum"] > 0 and (base["t3r"] or -1) > 0:
        pass_reasons.append("60s running threshold has positive 15m fee-net sum and T3R")
    else:
        fail_reasons.append("60s running threshold fails fee-net/T3R")
    if all((row["sum"] or 0) > 0 for row in wf) and all((row["t3r"] or -1) > 0 for row in wf):
        pass_reasons.append("all walk-forward folds positive")
    else:
        fail_reasons.append("walk-forward is not uniformly positive")
    if perm["p_right"] is not None and perm["p_right"] <= 0.05:
        pass_reasons.append("permutation-null p_right<=0.05")
    else:
        fail_reasons.append("permutation-null does not clear p<=0.05")
    if delay_sweep["10s"]["sum"] > 0 and (delay_sweep["10s"]["t3r"] or -1) > 0:
        pass_reasons.append("10s delay survives")
    else:
        fail_reasons.append("10s delay does not robustly survive")
    verdict = "PAPER_CANDIDATE" if not fail_reasons else "RESEARCH_ONLY"

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "symbol": "ETHUSDT",
            "days": int(args.days),
            "start_utc": utc_ms(start_ms),
            "end_utc": utc_ms(end_ms),
            "liq_rows": len(liqs),
            "note": "Research-only. No live executor/config/order logic touched.",
        },
        "window_robustness": window_robustness,
        "delay_sweep_60s": delay_sweep,
        "fee_stress_60s": fees,
        "exit_robustness_60s": exit_robustness,
        "walk_forward_60s": wf,
        "permutation_null": perm,
        "regime_splits_60s": regime_splits,
        "verdict": f"{verdict}: pass={pass_reasons}; fail={fail_reasons}",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    draw_md(result, args.out_md)
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
