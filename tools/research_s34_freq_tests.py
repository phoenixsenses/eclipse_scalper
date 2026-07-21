"""S34 state-machine frequency expansion tests.

Research-only. This script reads historical DB/ledger data and writes reports.
It does not touch live executor state, env, orders, sizing, or config.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations,
    reconstruct_anchors,
)
from tools.research_s34_state_machine_v2_gauntlet import load_nav_events  # noqa: E402


DB_PATH = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_FREQ_EXPANSION_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_FREQ_EXPANSION_TESTS.md"

FEE_BPS = 5.0
PROP_THRESH = 50_000.0
SYNC_WIN_MS = 10 * 60_000
SIL_LO_MS = 60_000
SIL_HI_MS = 30 * 60_000
LONG_HOLD_MS = 4 * 3600_000
SHORT_HOLD_MS = 2 * 3600_000
BASELINE_N = 34
BASELINE_WR = 0.706
BASELINE_AVG = 76.2
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Series:
    ts: list[int]
    vals: list[float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def r1(x: float | None) -> float | None:
    return None if x is None else round(float(x), 1)


def load_liq_series(conn: sqlite3.Connection, symbol: str, side: str) -> Series:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])


def load_mark_series(conn: sqlite3.Connection, symbol: str) -> Series:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        (symbol,),
    ).fetchall()
    return Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])


def load_mark_series_range(conn: sqlite3.Connection, symbol: str, lo: int, hi: int) -> Series:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, int(lo), int(hi)),
    ).fetchall()
    return Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])


def mark_at_or_after(series: Series, t: int) -> float | None:
    i = bisect.bisect_left(series.ts, int(t))
    if 0 <= i < len(series.vals):
        return float(series.vals[i])
    return None


def mark_at_or_before(series: Series, t: int) -> float | None:
    i = bisect.bisect_right(series.ts, int(t)) - 1
    if 0 <= i < len(series.vals):
        return float(series.vals[i])
    return None


def liq_sum(series: Series, lo: int, hi: int) -> float:
    a = bisect.bisect_left(series.ts, int(lo))
    b = bisect.bisect_left(series.ts, int(hi))
    return float(sum(series.vals[a:b]))


def liq_count(series: Series, lo: int, hi: int, thr: float) -> int:
    a = bisect.bisect_left(series.ts, int(lo))
    b = bisect.bisect_left(series.ts, int(hi))
    return sum(1 for v in series.vals[a:b] if float(v) >= float(thr))


def first_liq_above(series: Series, lo: int, hi: int, thr: float) -> tuple[int, float] | None:
    a = bisect.bisect_left(series.ts, int(lo))
    b = bisect.bisect_left(series.ts, int(hi))
    for i in range(a, b):
        if float(series.vals[i]) >= float(thr):
            return int(series.ts[i]), float(series.vals[i])
    return None


def ret_bps(series: Series, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(series, int(start_ms))
    b = mark_at_or_after(series, int(end_ms))
    if a is None or b is None or a <= 0:
        return None
    return (b - a) / a * 10_000.0


def signed_net(series: Series, side: str, entry_ts: int, exit_ts: int) -> float | None:
    raw = ret_bps(series, entry_ts, exit_ts)
    if raw is None:
        return None
    if side.upper() == "SHORT":
        raw = -raw
    return float(raw) - FEE_BPS


def session_for(hour: int) -> str:
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "EUROPE"
    if hour < 21:
        return "US"
    return "OFF"


def stat(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "sum": 0.0, "median": None, "t3r": 0.0, "per_month": 0.0}
    ts_vals = [int(r["anchor_ts_ms"]) for r in rows if r.get(key) is not None]
    months = months_span(ts_vals)
    sv = sorted(vals)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "avg": round(mean(vals), 1),
        "sum": round(sum(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(sum(sv[:-3]) if len(sv) > 3 else sum(sv), 1),
        "per_month": round(len(vals) / months, 1) if months > 0 else 0.0,
    }


def months_span(ts_vals: list[int]) -> float:
    if len(ts_vals) < 2:
        return 1.0
    days = (max(ts_vals) - min(ts_vals)) / 86_400_000.0
    return max(days / 30.4375, 1.0)


def verdict(s: dict[str, Any], *, min_wr: float = 0.70) -> str:
    if not s.get("n"):
        return "WORSE"
    wr = float(s.get("wr") or 0.0)
    avg = float(s.get("avg") or 0.0)
    if wr >= min_wr and avg >= BASELINE_AVG:
        return "PROMISING"
    if wr >= min_wr and avg > 0:
        return "NEUTRAL"
    return "WORSE"


def fmt_stat(s: dict[str, Any]) -> str:
    wr = s.get("wr")
    wrs = "NA" if wr is None else f"{float(wr) * 100:.1f}%"
    avg = s.get("avg")
    avgs = "NA" if avg is None else f"{float(avg):+.1f}"
    return f"N={s.get('n', 0)} | WR={wrs} | avg={avgs} bps | /mo={s.get('per_month', 0)}"


def group_stats(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, Any]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(key_fn(row), []).append(row)
    return {k: stat(v) for k, v in sorted(out.items())}


def build_dataset(db_path: Path = DB_PATH) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        nav = [
            r
            for r in load_nav_events()
            if r.get("symbol") == "ETHUSDT" and r.get("liq_side") == "SELL" and float(r.get("threshold_usd") or 0.0) >= 150_000.0
        ]
        nav_ts = [int(r["signal_ts_ms"]) for r in nav]
        if not nav_ts:
            return [], []
        lo_ms = min(nav_ts) - 8 * 24 * 3600_000
        hi_ms = max(nav_ts) + 5 * 3600_000
        nav_by_key = {
            (int(float(r.get("threshold_usd") or 0.0)), int(r["signal_ts_ms"])): r
            for r in nav
        }
        eth_sell = load_liq_series(conn, "ETHUSDT", "SELL")
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
        sol_sell = load_liq_series(conn, "SOLUSDT", "SELL")
        eth_marks = load_mark_series_range(conn, "ETHUSDT", lo_ms, hi_ms)
        btc_marks = load_mark_series_range(conn, "BTCUSDT", lo_ms, hi_ms)
        funding_cols = [r[1] for r in conn.execute("PRAGMA table_info(mark_prices)").fetchall()]
        has_funding = "funding_rate" in funding_cols
        funding_table_exists = bool(
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='funding_rates'").fetchone()
        )
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", min(nav_ts) - 3600_000, max(nav_ts) + 3600_000)
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=300,
            min_gap_sec=900,
            thresholds=(150_000.0, 200_000.0),
            accel_window_sec=30,
        )
        funding_by_ts: dict[int, float | None] = {}
        if has_funding:
            rows = conn.execute(
                "SELECT ts_ms, funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND funding_rate IS NOT NULL "
                "AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
                (lo_ms, hi_ms),
            ).fetchall()
            fseries = Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])
        else:
            fseries = Series([], [])

    all_rows: list[dict[str, Any]] = []
    for a in anchors:
        ts = int(a.anchor_ts_ms)
        nav_row = nav_by_key.get((int(float(a.threshold_usd)), ts), {})
        if float(a.threshold_usd) >= 200_000.0 and not nav_row:
            continue
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        hour = dt.hour
        dow = dt.weekday()
        session = session_for(hour)
        long_net = signed_net(eth_marks, "LONG", ts, ts + LONG_HOLD_MS)
        short_anchor_net = signed_net(eth_marks, "SHORT", ts, ts + SHORT_HOLD_MS)
        if long_net is None:
            continue
        btc4h = ret_bps(btc_marks, ts - 4 * 3600_000, ts)
        btc7d = ret_bps(btc_marks, ts - 7 * 24 * 3600_000, ts)
        btc3d = ret_bps(btc_marks, ts - 3 * 24 * 3600_000, ts)
        btc1h = ret_bps(btc_marks, ts - 3600_000, ts)
        eth4h = ret_bps(eth_marks, ts - 4 * 3600_000, ts)
        eth1h = ret_bps(eth_marks, ts - 3600_000, ts)
        if btc4h is None or eth1h is None:
            continue
        vdepth = nav_row.get("vdepth_bps")
        vdepth = 0.0 if vdepth is None or not math.isfinite(float(vdepth)) else float(vdepth)
        sync_k = liq_sum(btc_sell, ts - SYNC_WIN_MS, ts) + liq_sum(sol_sell, ts - SYNC_WIN_MS, ts)
        n2h = liq_count(eth_sell, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
        eth_follow = first_liq_above(eth_sell, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
        close_reason = "TIME_EXIT" if eth_follow is None else "NOISY_EARLY_EXIT"
        noisy_exit_net = None
        noisy_hold4_net = None
        noisy_cost_bps = None
        if eth_follow is not None:
            noisy_exit_net = signed_net(eth_marks, "LONG", ts, int(eth_follow[0]))
            noisy_hold4_net = long_net
            if noisy_exit_net is not None and noisy_hold4_net is not None:
                noisy_cost_bps = noisy_exit_net - noisy_hold4_net
        funding = None
        if fseries.ts:
            funding = mark_at_or_before(fseries, ts)
            funding_by_ts[ts] = funding
        bull = bool((eth1h or 0.0) > 20.0 and (btc4h or 0.0) > 50.0)
        base_score = sum(
            [
                int(n2h >= 3),
                int(float(btc4h) < 0.0),
                int(vdepth >= 30.0),
                int(session == "US"),
                int(sync_k >= 200_000.0),
            ]
        )
        row = {
            "event_id": str(a.event_id),
            "bucket": int(a.bucket),
            "threshold_usd": float(a.threshold_usd),
            "anchor_ts_ms": ts,
            "anchor_utc": iso_ms(ts),
            "hour": hour,
            "dow": dow,
            "dow_name": DOW[dow],
            "session": session,
            "running_notional": float(a.running_notional),
            "running_count": int(a.running_liq_count),
            "elapsed_since_first_sec": float(a.elapsed_since_first_sec),
            "running_rate": float(a.running_rate),
            "sync_k": sync_k,
            "n2h": n2h,
            "btc4h_bps": btc4h,
            "btc7d_bps": btc7d,
            "btc3d_bps": btc3d,
            "btc1h_bps": btc1h,
            "eth4h_bps": eth4h,
            "eth1h_bps": eth1h,
            "vdepth_bps": vdepth,
            "base_score": base_score,
            "long_score": base_score + 1,
            "bull": bull,
            "close_reason": close_reason,
            "net_bps": float(long_net),
            "long_4h_net_bps": float(long_net),
            "short_anchor_2h_net_bps": short_anchor_net,
            "eth_follow_ts_ms": None if eth_follow is None else int(eth_follow[0]),
            "eth_follow_notional": None if eth_follow is None else float(eth_follow[1]),
            "noisy_exit_net_bps": noisy_exit_net,
            "noisy_hold4_net_bps": noisy_hold4_net,
            "noisy_exit_cost_bps": noisy_cost_bps,
            "funding_rate": funding,
            "funding_source": "mark_prices" if fseries.ts else ("funding_rates" if funding_table_exists else "missing"),
        }
        all_rows.append(row)
    all_rows.sort(key=lambda r: (int(r["anchor_ts_ms"]), float(r["threshold_usd"])))
    rows200 = [r for r in all_rows if float(r["threshold_usd"]) == 200_000.0]
    return rows200, all_rows


def time_exit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["close_reason"] == "TIME_EXIT" and not r["bull"]]


def baseline_sync200(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in time_exit(rows) if r["sync_k"] < 200_000.0]


def current_long_gate(rows: list[dict[str, Any]], *, btc7d_limit: float = 0.0) -> list[dict[str, Any]]:
    return [
        r
        for r in time_exit(rows)
        if r["sync_k"] < 200_000.0
        and r["session"] != "EUROPE"
        and not (r["session"] == "US" and r["hour"] in {13, 14})
        and r["dow"] not in {0, 2}
        and r["long_score"] >= 3
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < btc7d_limit
    ]


def short_rows(
    rows: list[dict[str, Any]],
    *,
    btc_thr: float,
    delay_min: int,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
        eth_marks = load_mark_series(conn, "ETHUSDT")
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["bull"] or r["session"] == "EUROPE" or r["dow"] == 6 or int(r["base_score"]) < 4:
            continue
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_sell, ts + delay_min * 60_000, ts + SIL_HI_MS, btc_thr)
        if hit is None:
            continue
        net = signed_net(eth_marks, "SHORT", int(hit[0]), int(hit[0]) + SHORT_HOLD_MS)
        if net is None:
            continue
        out.append({**r, "entry_ts_ms": int(hit[0]), "btc_confirm_notional": float(hit[1]), "net_bps": float(net)})
    return out


def test_b_frequency(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = baseline_sync200(rows)

    # B1
    b1 = [
        r for r in time_exit(rows)
        if r["sync_k"] < 200_000.0
        and r["session"] != "EUROPE"
        and not (r["session"] == "US" and r["hour"] in {13, 14})
        and r["dow"] not in {0, 2}
        and r["long_score"] >= 3
        and r.get("btc3d_bps") is not None
        and float(r["btc3d_bps"]) < 0
    ]
    out["B1_btc3d_lt0"] = {"summary": stat(b1), "decision": verdict(stat(b1))}

    # B2
    b2 = [
        r for r in time_exit(rows)
        if r["sync_k"] < 200_000.0
        and r["session"] != "EUROPE"
        and not (r["session"] == "US" and r["hour"] in {13, 14})
        and r["dow"] not in {0, 2}
        and r["long_score"] >= 3
        and float(r["btc4h_bps"]) < 0
    ]
    out["B2_btc4h_only_no_btc7d"] = {"summary": stat(b2), "decision": verdict(stat(b2))}

    # B3
    b3 = [
        r for r in time_exit(rows)
        if r["session"] == "ASIA"
        and r["sync_k"] < 200_000.0
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 0
    ]
    out["B3_asia_sync_btc7d"] = {"summary": stat(b3), "decision": verdict(stat(b3))}

    # B4
    b4 = [
        r for r in time_exit(rows)
        if r["base_score"] == 5
        and r["sync_k"] < 200_000.0
        and r["session"] != "EUROPE"
        and r["dow"] not in {0, 2}
    ]
    out["B4_score5_no_btc7d"] = {"summary": stat(b4), "decision": verdict(stat(b4))}

    # B5
    rows150 = [r for r in all_rows if r["threshold_usd"] == 150_000.0]
    buckets200 = {int(r["bucket"]) for r in rows if r["threshold_usd"] == 200_000.0}
    added150 = [r for r in rows150 if int(r["bucket"]) not in buckets200 and r["close_reason"] == "TIME_EXIT" and not r["bull"]]
    out["B5_eth_sell_150k_added_no_200k_bucket"] = {"summary": stat(added150), "decision": verdict(stat(added150))}

    # B6
    b6_groups = {
        "n2h_0_1": [r for r in base if r["n2h"] <= 1],
        "n2h_2_4": [r for r in base if 2 <= r["n2h"] <= 4],
        "n2h_5_plus": [r for r in base if r["n2h"] >= 5],
    }
    out["B6_multiple_anchor_n2h"] = {k: stat(v) for k, v in b6_groups.items()}

    # B7
    out["B7_btc1h_split"] = {
        "btc1h_gt0": stat([r for r in base if (r.get("btc1h_bps") or 0) > 0]),
        "btc1h_lt0": stat([r for r in base if (r.get("btc1h_bps") or 0) < 0]),
    }

    # B8
    out["B8_short_btc1m_longer_delay"] = {
        "btc1m_delay10": stat(short_rows(rows, btc_thr=1_000_000.0, delay_min=10)),
        "btc1m_delay15": stat(short_rows(rows, btc_thr=1_000_000.0, delay_min=15)),
        "current_btc2m_delay5": stat(short_rows(rows, btc_thr=2_000_000.0, delay_min=5)),
    }

    # B9
    funding_rows = [r for r in base if r.get("funding_rate") is not None]
    out["B9_funding_rate"] = {
        "source": funding_rows[0]["funding_source"] if funding_rows else "missing",
        "negative": stat([r for r in funding_rows if float(r["funding_rate"]) < 0]),
        "non_negative": stat([r for r in funding_rows if float(r["funding_rate"]) >= 0]),
    }

    # B10
    b10 = [
        r for r in time_exit(rows)
        if r["sync_k"] < 200_000.0
        and r["session"] != "EUROPE"
        and not (r["session"] == "US" and r["hour"] in {13, 14})
        and r["dow"] not in {0, 2}
        and r["long_score"] >= 3
        and ((r.get("btc7d_bps") is not None and float(r["btc7d_bps"]) < 0.0) or int(r["base_score"]) == 5)
    ]
    out["B10_btc7d_or_score5"] = {"summary": stat(b10), "decision": verdict(stat(b10))}

    return out


def render_results(results: dict[str, Any]) -> str:
    lines = ["# S34 Frequency Expansion Tests", "", f"Generated: `{results['generated_at_utc']}`", ""]
    baseline = results["baseline"]
    lines.append(f"Baseline sync<200K: {fmt_stat(baseline)}")
    lines.append("")
    for key, val in results["tests"].items():
        lines.append(f"## [{key}]")
        if "summary" in val:
            lines.append(f"{fmt_stat(val['summary'])}")
            lines.append(f"vs baseline (N={BASELINE_N}, WR={BASELINE_WR*100:.1f}%, {BASELINE_AVG:+.1f} bps)")
            lines.append(f"Karar: {val.get('decision', verdict(val['summary']))}")
        else:
            for sub, s in val.items():
                if isinstance(s, dict) and "n" in s:
                    lines.append(f"- {sub}: {fmt_stat(s)}")
                elif isinstance(s, dict):
                    lines.append(f"- {sub}:")
                    for sub2, s2 in s.items():
                        if isinstance(s2, dict) and "n" in s2:
                            lines.append(f"  - {sub2}: {fmt_stat(s2)}")
                        else:
                            lines.append(f"  - {sub2}: `{s2}`")
                else:
                    lines.append(f"- {sub}: `{s}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows, all_rows = build_dataset()
    results = {
        "generated_at_utc": utc_now(),
        "dataset": {
            "anchors_200k": len(rows),
            "anchors_all_thresholds": len(all_rows),
            "time_exit_200k": len(time_exit(rows)),
        },
        "baseline": stat(baseline_sync200(rows)),
        "tests": test_b_frequency(rows, all_rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_results(results), encoding="utf-8")
    print(render_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
