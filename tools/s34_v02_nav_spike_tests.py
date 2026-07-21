"""S34 V02 navigation indicator x liquidation spike diagnostics.

Research-only. This script does not touch live execution. It builds a 1-minute
proxy of the V02 navigation indicator and tests whether high navigation states
lead liquidation spikes, whether spikes are continuation/exhaustion points, and
how the live V02 alpha lines up with those spikes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
LEDGER_PATH = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
OUT_JSON = OUT_DIR / "S34_V02_NAV_SPIKE_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_NAV_SPIKE_TESTS.md"

MINUTE = 60_000


def utc_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def bps(a: float, b: float) -> float | None:
    if not a or a <= 0:
        return None
    return (b - a) / a * 10_000.0


def pct(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


def summary(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "t3r": None}
    sorted_vals = sorted(vals, reverse=True)
    t3r_vals = sorted_vals[3:] if len(sorted_vals) > 3 else []
    return {
        "n": len(vals),
        "sum": round(sum(vals), 1),
        "mean": round(mean(vals), 2),
        "median": round(median(vals), 2),
        "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "t3r": round(sum(t3r_vals), 1) if t3r_vals else None,
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


def nonoverlap(events: list[dict[str, Any]], cooldown_ms: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last = -10**30
    for e in sorted(events, key=lambda x: int(x["ts"])):
        ts = int(e["ts"])
        if ts - last >= cooldown_ms:
            out.append(e)
            last = ts
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_ts(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("No ETHUSDT mark_prices rows")
    return int(row[0])


def load_mark_1m(conn: sqlite3.Connection, start_ms: int, end_ms: int, symbol: str) -> dict[int, float]:
    rows = conn.execute(
        """
        SELECT (ts_ms / 60000) * 60000 AS bucket_ms, AVG(mark_price) AS px
        FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        GROUP BY bucket_ms
        ORDER BY bucket_ms
        """,
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}


def load_liq_1m(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[int, dict[str, float]]:
    rows = conn.execute(
        """
        SELECT (ts_ms / 60000) * 60000 AS bucket_ms, side, SUM(notional) AS n
        FROM liquidations
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?
        GROUP BY bucket_ms, side
        ORDER BY bucket_ms
        """,
        (int(start_ms), int(end_ms)),
    ).fetchall()
    out: dict[int, dict[str, float]] = {}
    for bucket, side, notional in rows:
        out.setdefault(int(bucket), {"BUY": 0.0, "SELL": 0.0})[str(side).upper()] = float(notional or 0.0)
    return out


def load_flow_1m(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[int, float]:
    rows = conn.execute(
        """
        SELECT (ts_ms / 60000) * 60000 AS bucket_ms,
               SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END) AS taker_buy,
               SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) AS taker_sell
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?
        GROUP BY bucket_ms
        ORDER BY bucket_ms
        """,
        (int(start_ms), int(end_ms)),
    ).fetchall()
    out: dict[int, float] = {}
    for bucket, buy, sell in rows:
        b = float(buy or 0.0)
        s = float(sell or 0.0)
        total = b + s
        out[int(bucket)] = ((b - s) / total) if total > 0 else 0.0
    return out


def load_book_for_buckets(conn: sqlite3.Connection, buckets: list[int]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    stmt = """
        SELECT bid_price, bid_qty, ask_price, ask_qty, spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker
        WHERE symbol='ETHUSDT' AND ts_ms<=? AND ts_ms>=?
        ORDER BY ts_ms DESC
        LIMIT 1
    """
    for bucket in buckets:
        r = conn.execute(stmt, (int(bucket), int(bucket) - 120_000)).fetchone()
        if not r:
            continue
        bid = float(r[0] or 0.0)
        bid_qty = float(r[1] or 0.0)
        ask = float(r[2] or 0.0)
        ask_qty = float(r[3] or 0.0)
        out[bucket] = {
            "bid_depth_usd": float(r[6]) if r[6] is not None else bid * bid_qty,
            "ask_depth_usd": ask * ask_qty,
            "spread_bps": float(r[4] or 0.0) * 10_000.0,
            "book_imbalance": float(r[5] or 0.0),
        }
    return out


def rolling_sum(src: dict[int, float], buckets: list[int], window: int) -> dict[int, float]:
    out: dict[int, float] = {}
    acc = 0.0
    q: list[tuple[int, float]] = []
    for ts in buckets:
        val = float(src.get(ts, 0.0))
        q.append((ts, val))
        acc += val
        cutoff = ts - window * MINUTE
        while q and q[0][0] <= cutoff:
            _, old = q.pop(0)
            acc -= old
        out[ts] = acc
    return out


def build_nav(
    buckets: list[int],
    book: dict[int, dict[str, float]],
    liq: dict[int, dict[str, float]],
    flow: dict[int, float],
    btc_mark: dict[int, float],
) -> list[dict[str, Any]]:
    sell_1m = {ts: liq.get(ts, {}).get("SELL", 0.0) for ts in buckets}
    sell_5m = rolling_sum(sell_1m, buckets, 5)
    nav: list[dict[str, Any]] = []
    for ts in buckets:
        bk = book.get(ts)
        if not bk:
            continue
        btc_now = btc_mark.get(ts)
        btc_prev = btc_mark.get(ts - MINUTE)
        btc1 = bps(btc_prev, btc_now) if btc_prev and btc_now else None
        score = 0
        tags: list[str] = []
        warnings: list[str] = []

        def add(cond: bool, tag: str, pts: int = 1) -> None:
            nonlocal score
            if cond:
                score += pts
                tags.append(tag)

        def warn(cond: bool, tag: str, pts: int = 1) -> None:
            nonlocal score
            if cond:
                score -= pts
                warnings.append(tag)

        add(float(bk["bid_depth_usd"]) >= 135_423.8, "BID_OK", 2)
        add(float(bk["spread_bps"]) <= 0.15, "SPREAD_CLEAN", 2)
        add(float(bk["book_imbalance"]) >= 0.0, "BID_IMBALANCE")
        add(float(sell_1m.get(ts, 0.0)) <= 250_000.0, "NO_LARGE_SELL_LIQ_1M")
        add(float(flow.get(ts, 0.0)) > -0.25, "FLOW_NOT_HEAVY_SELL")
        add(btc1 is not None and float(btc1) > -10.0, "BTC_NOT_CRASHING")
        add(200_000.0 <= float(sell_5m.get(ts, 0.0)) <= 2_000_000.0, "SELL_CASCADE_CONTEXT", 2)

        warn(float(bk["bid_depth_usd"]) < 75_000.0, "BID_THIN", 2)
        warn(float(bk["spread_bps"]) > 0.35, "SPREAD_WIDE", 2)
        warn(float(sell_1m.get(ts, 0.0)) > 1_000_000.0, "SELL_LIQ_RESTART_HEAVY", 2)
        warn(btc1 is not None and float(btc1) < -25.0, "BTC_DUMPING", 2)

        bounded = max(0, min(10, int(score)))
        nav.append(
            {
                "ts": ts,
                "score": bounded,
                "bucket": "NAV_HIGH" if bounded >= 7 else ("NAV_MID" if bounded >= 5 else "NAV_LOW"),
                "tags": tags,
                "warnings": warnings,
                "bid_depth_usd": round(float(bk["bid_depth_usd"]), 1),
                "spread_bps": round(float(bk["spread_bps"]), 3),
                "sell_liq_5m": round(float(sell_5m.get(ts, 0.0)), 1),
            }
        )
    return nav


@dataclass
class Series:
    ts: list[int]
    px: list[float]

    def price_at_or_after(self, t: int) -> float | None:
        i = bisect.bisect_left(self.ts, int(t))
        if i >= len(self.ts):
            return None
        return self.px[i]

    def ret_after(self, t: int, horizon_ms: int) -> float | None:
        a = self.price_at_or_after(t)
        b = self.price_at_or_after(t + horizon_ms)
        if a is None or b is None:
            return None
        return bps(a, b)


def make_series(mark: dict[int, float]) -> Series:
    items = sorted(mark.items())
    return Series([int(k) for k, _ in items], [float(v) for _, v in items])


def nearest_nav(nav_by_ts: dict[int, dict[str, Any]], ts: int) -> dict[str, Any] | None:
    bucket = (int(ts) // MINUTE) * MINUTE
    return nav_by_ts.get(bucket)


def spike_thresholds(liq: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for side in ("BUY", "SELL"):
        vals = [float(v.get(side, 0.0)) for v in liq.values() if float(v.get(side, 0.0)) > 0]
        out[side] = {
            "p95_nonzero": round(pct(vals, 0.95), 1),
            "p99_nonzero": round(pct(vals, 0.99), 1),
            "primary_threshold": round(max(200_000.0, pct(vals, 0.95)), 1),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--spike-cooldown-min", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    end_ms: int
    with sqlite3.connect(args.db) as conn:
        end_ms = latest_ts(conn)
        start_ms = end_ms - int(args.days) * 24 * 60 * MINUTE
        eth_mark = load_mark_1m(conn, start_ms, end_ms, "ETHUSDT")
        btc_mark = load_mark_1m(conn, start_ms, end_ms, "BTCUSDT")
        mark_buckets = sorted(eth_mark)
        liq = load_liq_1m(conn, start_ms, end_ms)
        flow = load_flow_1m(conn, start_ms, end_ms)
        book = load_book_for_buckets(conn, mark_buckets)

    buckets = sorted(set(eth_mark) & set(book))
    nav = build_nav(buckets, book, liq, flow, btc_mark)
    nav_by_ts = {int(x["ts"]): x for x in nav}
    series = make_series(eth_mark)
    thresholds = spike_thresholds(liq)

    spikes_all: list[dict[str, Any]] = []
    for ts in buckets:
        for side in ("BUY", "SELL"):
            notional = float(liq.get(ts, {}).get(side, 0.0))
            if notional >= thresholds[side]["primary_threshold"]:
                n = nav_by_ts.get(ts)
                spikes_all.append(
                    {
                        "ts": ts,
                        "utc": utc_ms(ts),
                        "side": side,
                        "notional": round(notional, 1),
                        "nav_score": n.get("score") if n else None,
                        "nav_bucket": n.get("bucket") if n else None,
                    }
                )
    spikes = nonoverlap(spikes_all, int(args.spike_cooldown_min) * MINUTE)
    spikes_by_side = {s: [e for e in spikes if e["side"] == s] for s in ("BUY", "SELL")}
    spike_ts_by_side = {s: sorted(int(e["ts"]) for e in events) for s, events in spikes_by_side.items()}

    # 1. NAV_HIGH -> spike lead.
    nav_high_transitions = []
    prev = None
    for n in nav:
        if int(n["score"]) >= 7 and (prev is None or int(prev["score"]) < 7):
            nav_high_transitions.append(n)
        prev = n
    lead_windows = [1, 3, 5, 15]
    nav_lead: dict[str, Any] = {"events": len(nav_high_transitions), "windows_min": {}}
    for w in lead_windows:
        row: dict[str, Any] = {}
        for side in ("BUY", "SELL"):
            hits = 0
            delays: list[float] = []
            for n in nav_high_transitions:
                t = int(n["ts"])
                arr = spike_ts_by_side[side]
                i = bisect.bisect_left(arr, t)
                if i < len(arr) and arr[i] <= t + w * MINUTE:
                    hits += 1
                    delays.append((arr[i] - t) / 60_000)
            row[side] = {
                "hit_rate": round(hits / len(nav_high_transitions), 3) if nav_high_transitions else None,
                "hits": hits,
                "median_delay_min": round(median(delays), 2) if delays else None,
            }
        nav_lead["windows_min"][str(w)] = row

    # 2. Spike -> move exhaustion/continuation and 5. side symmetry.
    horizons = {"1m": 1 * MINUTE, "5m": 5 * MINUTE, "15m": 15 * MINUTE, "60m": 60 * MINUTE, "120m": 120 * MINUTE}
    spike_forward: dict[str, Any] = {}
    for side, events in spikes_by_side.items():
        side_res: dict[str, Any] = {"n": len(events), "horizons": {}, "nav_interaction": {}}
        for label, h in horizons.items():
            vals = [series.ret_after(int(e["ts"]), h) for e in events]
            vals = [v for v in vals if v is not None]
            side_res["horizons"][label] = summary(vals)
            high_vals = [series.ret_after(int(e["ts"]), h) for e in events if (e.get("nav_score") or 0) >= 7]
            low_vals = [series.ret_after(int(e["ts"]), h) for e in events if (e.get("nav_score") or 0) < 7]
            side_res["nav_interaction"][label] = {"NAV_HIGH": summary([v for v in high_vals if v is not None]), "not_high": summary([v for v in low_vals if v is not None])}
        spike_forward[side] = side_res

    # 3. Pre-spike indicator shape.
    pre_shape: dict[str, Any] = {}
    non_spike_set = {int(e["ts"]) for e in spikes}
    sampled_non_spikes = [ts for i, ts in enumerate(buckets) if ts not in non_spike_set and i % 30 == 0]
    for side, events in spikes_by_side.items():
        rows = []
        for e in events:
            t = int(e["ts"])
            n0 = nav_by_ts.get(t)
            n5 = nav_by_ts.get(t - 5 * MINUTE)
            if n0 and n5:
                rows.append(
                    {
                        "score_t": int(n0["score"]),
                        "score_t_minus_5": int(n5["score"]),
                        "delta_5m": int(n0["score"]) - int(n5["score"]),
                        "high_minutes_prev_5": sum(1 for j in range(5) if (nav_by_ts.get(t - j * MINUTE) or {}).get("score", 0) >= 7),
                    }
                )
        ctrl_rows = []
        for t in sampled_non_spikes:
            n0 = nav_by_ts.get(t)
            n5 = nav_by_ts.get(t - 5 * MINUTE)
            if n0 and n5:
                ctrl_rows.append(
                    {
                        "score_t": int(n0["score"]),
                        "score_t_minus_5": int(n5["score"]),
                        "delta_5m": int(n0["score"]) - int(n5["score"]),
                        "high_minutes_prev_5": sum(1 for j in range(5) if (nav_by_ts.get(t - j * MINUTE) or {}).get("score", 0) >= 7),
                    }
                )
        pre_shape[side] = {
            "spike_n": len(rows),
            "spike_avg_score_t": round(mean([r["score_t"] for r in rows]), 2) if rows else None,
            "spike_avg_delta_5m": round(mean([r["delta_5m"] for r in rows]), 2) if rows else None,
            "spike_avg_high_minutes_prev_5": round(mean([r["high_minutes_prev_5"] for r in rows]), 2) if rows else None,
            "control_n": len(ctrl_rows),
            "control_avg_score_t": round(mean([r["score_t"] for r in ctrl_rows]), 2) if ctrl_rows else None,
            "control_avg_delta_5m": round(mean([r["delta_5m"] for r in ctrl_rows]), 2) if ctrl_rows else None,
            "control_avg_high_minutes_prev_5": round(mean([r["high_minutes_prev_5"] for r in ctrl_rows]), 2) if ctrl_rows else None,
        }

    # 4. Alpha + spike timing.
    ledger = [r for r in read_jsonl(LEDGER_PATH) if int(r.get("signal_ts_ms") or 0) >= start_ms]
    alpha_rows = []
    all_spike_ts = sorted((int(e["ts"]), e["side"]) for e in spikes)
    all_ts_only = [x[0] for x in all_spike_ts]
    for r in ledger:
        t = int(r.get("signal_ts_ms") or 0)
        if not t:
            continue
        bucket = (t // MINUTE) * MINUTE
        navrow = nav_by_ts.get(bucket)
        nearest = None
        if all_ts_only:
            i = bisect.bisect_left(all_ts_only, bucket)
            candidates = []
            if i < len(all_ts_only):
                candidates.append(all_spike_ts[i])
            if i > 0:
                candidates.append(all_spike_ts[i - 1])
            nearest = min(candidates, key=lambda x: abs(x[0] - bucket)) if candidates else None
        alpha_rows.append(
            {
                "signal_utc": utc_ms(t),
                "net_bps": r.get("net_bps"),
                "sim_status": r.get("sim_status"),
                "nav_score_at_signal": navrow.get("score") if navrow else None,
                "nav_bucket_at_signal": navrow.get("bucket") if navrow else None,
                "nearest_spike_side": nearest[1] if nearest else None,
                "nearest_spike_delta_min": round((nearest[0] - bucket) / 60_000, 2) if nearest else None,
                "nearest_spike_after": bool(nearest and nearest[0] >= bucket),
            }
        )
    alpha_timing = {
        "n": len(alpha_rows),
        "rows": alpha_rows,
        "summary_by_nav_high": {
            "NAV_HIGH": summary([float(r["net_bps"]) for r in alpha_rows if r.get("net_bps") is not None and (r.get("nav_score_at_signal") or 0) >= 7]),
            "not_high": summary([float(r["net_bps"]) for r in alpha_rows if r.get("net_bps") is not None and (r.get("nav_score_at_signal") or 0) < 7]),
        },
    }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "symbol": "ETHUSDT",
            "days": int(args.days),
            "start_utc": utc_ms(start_ms),
            "end_utc": utc_ms(end_ms),
            "nav_points": len(nav),
            "spike_cooldown_min": int(args.spike_cooldown_min),
            "note": "Navigation indicator is a 1-minute research proxy of the chart indicator, not live order logic.",
        },
        "spike_thresholds": thresholds,
        "spike_counts": {side: len(events) for side, events in spikes_by_side.items()},
        "test_1_nav_high_leads_spikes": nav_lead,
        "test_2_spike_forward_returns": spike_forward,
        "test_3_pre_spike_indicator_shape": pre_shape,
        "test_4_alpha_spike_timing": alpha_timing,
        "test_5_side_symmetry": {
            "BUY_interpretation": "Positive forward return means BUY-liq spike continuation; negative means BUY-liq spike exhaustion/fade.",
            "SELL_interpretation": "Positive forward return means SELL-liq spike rebound; negative means sell-side continuation.",
            "by_side": spike_forward,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# S34 V02 Navigation x Liquidation Spike Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        f"Scope: ETHUSDT, last `{args.days}` days, `{len(nav)}` 1m nav points.",
        "",
        "## Spike Thresholds",
        "",
        f"- BUY primary threshold: `{thresholds['BUY']['primary_threshold']}` notional/min",
        f"- SELL primary threshold: `{thresholds['SELL']['primary_threshold']}` notional/min",
        f"- Non-overlap cooldown: `{args.spike_cooldown_min}` minutes",
        f"- Spike counts: BUY `{len(spikes_by_side['BUY'])}`, SELL `{len(spikes_by_side['SELL'])}`",
        "",
        "## 1. NAV_HIGH -> Liq Spike Lead",
        "",
    ]
    for w, row in nav_lead["windows_min"].items():
        lines.append(f"- within {w}m: BUY hit `{row['BUY']['hit_rate']}` ({row['BUY']['hits']}), SELL hit `{row['SELL']['hit_rate']}` ({row['SELL']['hits']})")
    lines += ["", "## 2 + 5. Spike Forward Returns / Side Symmetry", ""]
    for side in ("BUY", "SELL"):
        lines.append(f"### {side} spikes")
        for h, s in spike_forward[side]["horizons"].items():
            lines.append(f"- {h}: N `{s['n']}`, sum `{s['sum']}`, median `{s['median']}`, WR `{s['win_rate']}`, T3R `{s['t3r']}`")
        lines.append("")
    lines += ["## 3. Pre-Spike Indicator Shape", ""]
    for side, s in pre_shape.items():
        lines.append(
            f"- {side}: spike avg score `{s['spike_avg_score_t']}` vs control `{s['control_avg_score_t']}`; "
            f"delta5m `{s['spike_avg_delta_5m']}` vs control `{s['control_avg_delta_5m']}`; "
            f"prev5 high-min `{s['spike_avg_high_minutes_prev_5']}` vs control `{s['control_avg_high_minutes_prev_5']}`"
        )
    lines += ["", "## 4. Live Alpha + Spike Timing", ""]
    lines.append(f"- alpha rows in scope: `{alpha_timing['n']}`")
    lines.append(f"- alpha NAV_HIGH: `{alpha_timing['summary_by_nav_high']['NAV_HIGH']}`")
    lines.append(f"- alpha not-high: `{alpha_timing['summary_by_nav_high']['not_high']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Research-only. No live executor/config/order logic touched.")
    lines.append("- The indicator here is a 1-minute proxy; use the chart line for visual monitoring, not as a live trigger.")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
