"""S34 BUY-side state-machine symmetry gauntlet.

Research-only. Tests the mirror family of the current ETH SELL state machine:

    ETH BUY liquidation = short liquidation / forced buy
    BUY cascade -> SHORT mean-reversion/fade
    BUY cascade -> LONG continuation

No live executor, env, order logic, leverage, sizing, or dashboard files are
modified.
"""

from __future__ import annotations

import bisect
import json
import math
import random
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

from tools.research_s34_knowable_anchor_continuation import load_liquidations, reconstruct_anchors  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402


DB_PATH = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_BUY_SIDE_STATE_MACHINE_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_BUY_SIDE_STATE_MACHINE_GAUNTLET.md"

FEE_BPS = 5.0
ETH_THRESH = 200_000.0
PROP_THRESH = 50_000.0
SYNC_WIN_MS = 10 * 60_000
SIL_LO_MS = 60_000
SIL_HI_MS = 30 * 60_000
ECHO_LO_MS = 45 * 60_000
ECHO_HI_MS = 120 * 60_000
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WIN_SEC = 30
BTC_THRESHOLDS = (500_000.0, 1_000_000.0, 2_000_000.0)
DELAYS_MIN = (5, 10)
HOLDS_MIN = (60, 120, 180, 240)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_SERIES_CACHE: dict[str, Series] = {}


@dataclass(frozen=True)
class Series:
    ts: list[int]
    vals: list[float]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    direction: str
    rows: list[dict[str, Any]]
    note: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def session_for(hour: int) -> str:
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "EUROPE"
    if hour < 21:
        return "US"
    return "OFF"


def load_liq_series(conn: sqlite3.Connection, symbol: str, side: str) -> Series:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])


def load_mark_series(conn: sqlite3.Connection, symbol: str, lo: int | None = None, hi: int | None = None) -> Series:
    clauses = ["symbol=?"]
    params: list[Any] = [symbol]
    if lo is not None:
        clauses.append("ts_ms>=?")
        params.append(int(lo))
    if hi is not None:
        clauses.append("ts_ms<=?")
        params.append(int(hi))
    rows = conn.execute(
        f"SELECT ts_ms, mark_price FROM mark_prices WHERE {' AND '.join(clauses)} ORDER BY ts_ms",
        tuple(params),
    ).fetchall()
    return Series([int(r[0]) for r in rows], [float(r[1]) for r in rows])


def cached_series(db_path: Path = DB_PATH) -> dict[str, Series]:
    if _SERIES_CACHE:
        return _SERIES_CACHE
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        _SERIES_CACHE["btc_buy"] = load_liq_series(conn, "BTCUSDT", "BUY")
        _SERIES_CACHE["eth_marks"] = load_mark_series(conn, "ETHUSDT")
    return _SERIES_CACHE


def mark_at_or_after(series: Series, t: int) -> float | None:
    i = bisect.bisect_left(series.ts, int(t))
    if 0 <= i < len(series.vals):
        return float(series.vals[i])
    return None


def ret_bps(series: Series, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(series, int(start_ms))
    b = mark_at_or_after(series, int(end_ms))
    if a is None or b is None or a <= 0:
        return None
    return (b - a) / a * 10_000.0


def signed_net(series: Series, direction: str, entry_ts: int, exit_ts: int) -> float | None:
    raw = ret_bps(series, entry_ts, exit_ts)
    if raw is None or not math.isfinite(float(raw)):
        return None
    if direction.upper() == "SHORT":
        raw = -raw
    return float(raw) - FEE_BPS


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


def has_bucketed_cascade(series: Series, lo: int, hi: int, thr: float, bucket_ms: int = 5 * 60_000) -> bool:
    a = bisect.bisect_left(series.ts, int(lo))
    b = bisect.bisect_left(series.ts, int(hi))
    buckets: dict[int, float] = {}
    for i in range(a, b):
        bucket = int((series.ts[i] - lo) // bucket_ms)
        buckets[bucket] = buckets.get(bucket, 0.0) + float(series.vals[i])
    return any(v >= float(thr) for v in buckets.values())


def months_span(rows: list[dict[str, Any]]) -> float:
    ts = [int(r.get("entry_ts_ms") or r["anchor_ts_ms"]) for r in rows]
    if len(ts) < 2:
        return 1.0
    return max((max(ts) - min(ts)) / 86_400_000.0 / 30.4375, 1.0)


def finite_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("net_bps") is not None and math.isfinite(float(r["net_bps"]))]


def stat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in finite_rows(rows)]
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "sum": 0.0, "median": None, "t3r": 0.0, "per_month": 0.0}
    desc = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "avg": round(mean(vals), 1),
        "sum": round(sum(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(sum(desc[3:]) if len(desc) > 3 else sum(desc), 1),
        "per_month": round(len(vals) / months_span(rows), 1),
        "worst": round(min(vals), 1),
        "tail100_n": sum(1 for v in vals if v <= -100.0),
    }


def chronological_folds(rows: list[dict[str, Any]], k: int = 5) -> list[list[dict[str, Any]]]:
    rs = sorted(finite_rows(rows), key=lambda r: int(r.get("entry_ts_ms") or r["anchor_ts_ms"]))
    if not rs:
        return [[] for _ in range(k)]
    n = len(rs)
    return [rs[round(i * n / k): round((i + 1) * n / k)] for i in range(k)]


def fold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    folds = chronological_folds(rows, 5)
    fs = [stat(f) for f in folds]
    return {
        "folds": fs,
        "positive_sum_folds": sum(1 for s in fs if float(s.get("sum") or 0.0) > 0.0),
        "positive_t3r_folds": sum(1 for s in fs if float(s.get("t3r") or 0.0) > 0.0),
        "worst_fold_sum": round(min((float(s.get("sum") or 0.0) for s in fs), default=0.0), 1),
        "worst_fold_t3r": round(min((float(s.get("t3r") or 0.0) for s in fs), default=0.0), 1),
    }


def split_70_30(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rs = sorted(finite_rows(rows), key=lambda r: int(r.get("entry_ts_ms") or r["anchor_ts_ms"]))
    cut = int(round(len(rs) * 0.70))
    return rs[:cut], rs[cut:]


def no_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    next_free = -10**18
    for r in sorted(finite_rows(rows), key=lambda x: int(x.get("entry_ts_ms") or x["anchor_ts_ms"])):
        entry = int(r.get("entry_ts_ms") or r["anchor_ts_ms"])
        exit_ts = int(r.get("exit_ts_ms") or entry)
        if entry < next_free:
            continue
        out.append(r)
        next_free = exit_ts
    return out


def readiness(summary: dict[str, Any], holdout: dict[str, Any], folds: dict[str, Any], noov: dict[str, Any]) -> str:
    if int(summary.get("n") or 0) < 10:
        return "LOW_N_RESEARCH_ONLY"
    if float(summary.get("t3r") or 0.0) <= 0.0:
        return "REJECT_T3R"
    if float(holdout.get("sum") or 0.0) <= 0.0 or float(holdout.get("t3r") or 0.0) <= 0.0:
        return "RESEARCH_ONLY_HOLDOUT_WEAK"
    if int(folds.get("positive_sum_folds") or 0) < 3 or int(folds.get("positive_t3r_folds") or 0) < 3:
        return "RESEARCH_ONLY_FOLD_WEAK"
    if float(noov.get("sum") or 0.0) <= 0.0 or float(noov.get("t3r") or 0.0) <= 0.0:
        return "RESEARCH_ONLY_OVERLAP_DEPENDENT"
    if int(summary.get("tail100_n") or 0) > 0:
        return "SHADOW_ONLY_TAIL"
    return "PAPER_CANDIDATE"


def build_buy_dataset(db_path: Path = DB_PATH) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        bounds = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY'").fetchone()
        if not bounds or bounds[0] is None:
            return [], {"error": "no ETH BUY liquidations"}
        lo_ms = int(bounds[0]) - 8 * 24 * 3600_000
        hi_ms = int(bounds[1]) + 5 * 3600_000
        liqs = load_liquidations(conn, "ETHUSDT", "BUY", int(bounds[0]) - 3600_000, int(bounds[1]) + 3600_000)
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            thresholds=(ETH_THRESH,),
            accel_window_sec=ACCEL_WIN_SEC,
        )
        eth_buy = load_liq_series(conn, "ETHUSDT", "BUY")
        btc_buy = load_liq_series(conn, "BTCUSDT", "BUY")
        sol_buy = load_liq_series(conn, "SOLUSDT", "BUY")
        eth_marks = load_mark_series(conn, "ETHUSDT", lo_ms, hi_ms)
        btc_marks = load_mark_series(conn, "BTCUSDT", lo_ms, hi_ms)

        rows: list[dict[str, Any]] = []
        for a in anchors:
            ts = int(a.anchor_ts_ms)
            p = mark_at_or_after(eth_marks, ts)
            if p is None:
                continue
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            hour = dt.hour
            dow = dt.weekday()
            session = session_for(hour)
            btc4h = ret_bps(btc_marks, ts - 4 * 3600_000, ts)
            btc7d = ret_bps(btc_marks, ts - 7 * 24 * 3600_000, ts)
            btc3d = ret_bps(btc_marks, ts - 3 * 24 * 3600_000, ts)
            btc1h = ret_bps(btc_marks, ts - 3600_000, ts)
            eth1h = ret_bps(eth_marks, ts - 3600_000, ts)
            eth4h = ret_bps(eth_marks, ts - 4 * 3600_000, ts)
            if btc4h is None or btc7d is None or eth1h is None:
                continue
            book = book_features_at(conn, "ETHUSDT", ts, 30)
            ask_depth = None if not book else float(book.get("ask_depth_usd") or 0.0)
            bid_depth = None if not book else float(book.get("bid_depth_usd") or 0.0)
            imbalance = None if not book else float(book.get("book_imbalance") or 0.0)
            spread_bps = None if not book else float(book.get("spread_bps") or 0.0)
            sync_k = liq_sum(btc_buy, ts - SYNC_WIN_MS, ts) + liq_sum(sol_buy, ts - SYNC_WIN_MS, ts)
            n2h = liq_count(eth_buy, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
            follow = first_liq_above(eth_buy, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
            echo_45_120 = has_bucketed_cascade(eth_buy, ts - ECHO_HI_MS, ts - ECHO_LO_MS, ETH_THRESH)
            prebuildup = liq_count(eth_buy, ts - 30 * 60_000, ts - 1000, PROP_THRESH)
            # Mirror score: upside cascade regime. Keep it simple and point-in-time.
            bear_squeeze = bool((eth1h or 0.0) < -20.0 and (btc4h or 0.0) < -50.0)
            base_score = sum(
                [
                    int(n2h >= 3),
                    int(float(btc4h) > 0.0),
                    int(ask_depth is not None and ask_depth >= 50_000.0),
                    int(session == "US"),
                    int(sync_k >= 200_000.0),
                ]
            )
            short_net_by_hold = {str(h): signed_net(eth_marks, "SHORT", ts, ts + h * 60_000) for h in HOLDS_MIN}
            long_net_by_hold = {str(h): signed_net(eth_marks, "LONG", ts, ts + h * 60_000) for h in HOLDS_MIN}
            rows.append(
                {
                    "event_id": f"ETH_BUY:{a.bucket}:{int(a.threshold_usd)}",
                    "bucket": int(a.bucket),
                    "anchor_ts_ms": ts,
                    "anchor_utc": iso_ms(ts),
                    "liq_side": "BUY",
                    "threshold_usd": float(a.threshold_usd),
                    "running_notional": float(a.running_notional),
                    "running_count": int(a.running_liq_count),
                    "running_rate": float(a.running_rate),
                    "running_accel": float(a.running_accel),
                    "dominance": float(a.running_single_liq_dominance),
                    "hour": hour,
                    "dow": dow,
                    "dow_name": DOW[dow],
                    "session": session,
                    "btc4h_bps": btc4h,
                    "btc7d_bps": btc7d,
                    "btc3d_bps": btc3d,
                    "btc1h_bps": btc1h,
                    "eth1h_bps": eth1h,
                    "eth4h_bps": eth4h,
                    "sync_k": sync_k,
                    "n2h": n2h,
                    "follow_ts_ms": None if follow is None else int(follow[0]),
                    "follow_notional": None if follow is None else float(follow[1]),
                    "state": "SILENCE" if follow is None else "NOISY",
                    "echo_45_120": bool(echo_45_120),
                    "prebuildup": int(prebuildup),
                    "base_score": int(base_score),
                    "score_if_silence": int(base_score + 1),
                    "bear_squeeze": bear_squeeze,
                    "ask_depth_usd": ask_depth,
                    "bid_depth_usd": bid_depth,
                    "book_imbalance": imbalance,
                    "spread_bps": spread_bps,
                    "short_net_by_hold": short_net_by_hold,
                    "long_net_by_hold": long_net_by_hold,
                }
            )
    rows.sort(key=lambda r: int(r["anchor_ts_ms"]))
    meta = {"anchors_200k": len(rows), "start_utc": iso_ms(rows[0]["anchor_ts_ms"]) if rows else None, "end_utc": iso_ms(rows[-1]["anchor_ts_ms"]) if rows else None}
    return rows, meta


def base_prefilter(r: dict[str, Any]) -> bool:
    return not bool(r["bear_squeeze"]) and str(r["session"]) != "EUROPE"


def short_fade_rows(rows: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool], hold_min: int, name: str) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not pred(r):
            continue
        net = r["short_net_by_hold"].get(str(hold_min))
        if net is None:
            continue
        out.append({**r, "candidate": name, "family": "BUY_TO_SHORT_FADE", "direction": "SHORT", "entry_ts_ms": int(r["anchor_ts_ms"]), "exit_ts_ms": int(r["anchor_ts_ms"]) + hold_min * 60_000, "net_bps": float(net)})
    return out


def long_cont_rows(
    rows: list[dict[str, Any]],
    *,
    btc_thr: float,
    delay_min: int,
    hold_min: int,
    score_min: int,
    db_path: Path = DB_PATH,
    name: str,
) -> list[dict[str, Any]]:
    cache = cached_series(db_path)
    btc_buy = cache["btc_buy"]
    eth_marks = cache["eth_marks"]
    out = []
    for r in rows:
        if not base_prefilter(r) or int(r["dow"]) == 6 or int(r["base_score"]) < score_min:
            continue
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_buy, ts + delay_min * 60_000, ts + SIL_HI_MS, btc_thr)
        if hit is None:
            continue
        net = signed_net(eth_marks, "LONG", int(hit[0]), int(hit[0]) + hold_min * 60_000)
        if net is None:
            continue
        out.append({**r, "candidate": name, "family": "BUY_TO_LONG_CONT", "direction": "LONG", "entry_ts_ms": int(hit[0]), "exit_ts_ms": int(hit[0]) + hold_min * 60_000, "btc_confirm_notional": float(hit[1]), "net_bps": float(net)})
    return out


def same_side_prop_long_rows(rows: list[dict[str, Any]], hold_min: int, name: str) -> list[dict[str, Any]]:
    eth_marks = cached_series(DB_PATH)["eth_marks"]
    out = []
    for r in rows:
        if not base_prefilter(r) or int(r["base_score"]) < 3 or r.get("follow_ts_ms") is None:
            continue
        entry = int(r["follow_ts_ms"])
        net = signed_net(eth_marks, "LONG", entry, entry + hold_min * 60_000)
        if net is None:
            continue
        out.append({**r, "candidate": name, "family": "BUY_TO_LONG_SAME_PROP", "direction": "LONG", "entry_ts_ms": entry, "exit_ts_ms": entry + hold_min * 60_000, "net_bps": float(net)})
    return out


def build_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    cands: list[Candidate] = []

    def add_short(name: str, pred: Callable[[dict[str, Any]], bool], hold: int, note: str) -> None:
        cands.append(Candidate(name, "BUY_TO_SHORT_FADE", "SHORT", short_fade_rows(rows, pred, hold, name), note))

    # Mean-reversion / fade variants.
    for hold in HOLDS_MIN:
        add_short(
            f"F_all_short_h{hold}",
            lambda r: base_prefilter(r),
            hold,
            "BUY cascade -> immediate SHORT fade, broad point-in-time universe.",
        )
        add_short(
            f"F_silence_short_h{hold}",
            lambda r: base_prefilter(r) and r["state"] == "SILENCE",
            hold,
            "BUY cascade -> SHORT fade only if no same-side BUY follow-on in 30m.",
        )
        add_short(
            f"F_echo45_120_silence_short_h{hold}",
            lambda r: base_prefilter(r) and r["state"] == "SILENCE" and bool(r["echo_45_120"]),
            hold,
            "Prior BUY echo 45-120m + silence -> SHORT fade.",
        )
        add_short(
            f"F_prebuild2_silence_short_h{hold}",
            lambda r: base_prefilter(r) and r["state"] == "SILENCE" and int(r["prebuildup"]) >= 2,
            hold,
            "Prebuild>=2 / double cascade + silence -> SHORT fade.",
        )
        add_short(
            f"F_score3_silence_short_h{hold}",
            lambda r: base_prefilter(r) and r["state"] == "SILENCE" and int(r["score_if_silence"]) >= 3,
            hold,
            "Mirror score>=3 + silence -> SHORT fade.",
        )
        add_short(
            f"F_sync_lt200_regime_short_h{hold}",
            lambda r: base_prefilter(r) and r["state"] == "SILENCE" and float(r["sync_k"]) < 200_000.0 and (float(r["btc4h_bps"]) > 0.0 or float(r["btc7d_bps"]) > 0.0),
            hold,
            "SELL-side current gate mirrored: low prior sync + positive BTC regime -> SHORT fade.",
        )

    # Continuation variants: BTC BUY confirm, delays, holds.
    for score_min in (3, 4):
        for btc_thr in BTC_THRESHOLDS:
            for delay in DELAYS_MIN:
                for hold in HOLDS_MIN:
                    name = f"C_score{score_min}_btc{int(btc_thr/1000)}k_delay{delay}_long_h{hold}"
                    cands.append(
                        Candidate(
                            name,
                            "BUY_TO_LONG_CONT",
                            "LONG",
                            long_cont_rows(rows, btc_thr=btc_thr, delay_min=delay, hold_min=hold, score_min=score_min, name=name),
                            "BUY cascade -> LONG continuation after BTC BUY confirmation.",
                        )
                    )

    for hold in HOLDS_MIN:
        name = f"C_same_side_follow_long_h{hold}"
        cands.append(Candidate(name, "BUY_TO_LONG_SAME_PROP", "LONG", same_side_prop_long_rows(rows, hold, name), "BUY cascade -> LONG continuation at same-side ETH BUY follow-on."))

    return cands


def evaluate(c: Candidate) -> dict[str, Any]:
    cal, ho = split_70_30(c.rows)
    s = stat(c.rows)
    h = stat(ho)
    folds = fold_summary(c.rows)
    noov = stat(no_overlap(c.rows))
    return {
        "name": c.name,
        "family": c.family,
        "direction": c.direction,
        "note": c.note,
        "summary": s,
        "calibration": stat(cal),
        "holdout": h,
        "folds": folds,
        "no_overlap": noov,
        "readiness": readiness(s, h, folds, noov),
    }


def permutation_maxstat(results: list[dict[str, Any]], *, iterations: int = 500, seed: int = 3405) -> dict[str, Any]:
    rng = random.Random(seed)
    observed = max((float(r["summary"].get("t3r") or 0.0) for r in results), default=0.0)
    candidate_values = []
    for r in results:
        vals = [float(x["net_bps"]) for x in finite_rows(r.get("_rows", []))]
        if len(vals) >= 10:
            candidate_values.append(vals)
    if not candidate_values:
        return {"iterations": iterations, "observed_max_t3r": observed, "p_right": None, "note": "no candidates with N>=10"}
    pool = [v for vals in candidate_values for v in vals]
    maxes = []
    for _ in range(iterations):
        rng.shuffle(pool)
        pos = 0
        m = -10**18
        for vals in candidate_values:
            n = len(vals)
            sample = pool[pos: pos + n]
            pos += n
            desc = sorted(sample, reverse=True)
            t3r = sum(desc[3:]) if len(desc) > 3 else sum(desc)
            m = max(m, t3r)
        maxes.append(m)
    p_right = (sum(1 for x in maxes if x >= observed) + 1) / (len(maxes) + 1)
    return {
        "iterations": iterations,
        "observed_max_t3r": round(observed, 1),
        "null_p95_max_t3r": round(sorted(maxes)[int(0.95 * (len(maxes) - 1))], 1),
        "p_right": round(p_right, 4),
        "note": "max-stat permutation across searched candidate cells; conservative artifact check",
    }


def group_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        net = r["short_net_by_hold"].get("240")
        if net is None or not math.isfinite(float(net)):
            continue
        out.setdefault(str(r.get(key)), []).append({"net_bps": float(net), "anchor_ts_ms": r["anchor_ts_ms"]})
    return {k: stat(v) for k, v in sorted(out.items())}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 BUY-Side State-Machine Symmetry Gauntlet",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor, `.env`, order logic, leverage, or sizing was changed.",
        "",
        "## Side Semantics",
        "- `ETH SELL liquidation` = long liquidation / forced sell.",
        "- `ETH BUY liquidation` = short liquidation / forced buy.",
        "- BUY-side tested directions:",
        "  - `ETH BUY -> SHORT` = mean-reversion / fade after short squeeze.",
        "  - `ETH BUY -> LONG` = continuation after short squeeze.",
        "",
        "## Dataset",
        f"- ETH BUY 200K knowable anchors: `{report['dataset']['anchors_200k']}`",
        f"- Date range: `{report['dataset']['start_utc']}` -> `{report['dataset']['end_utc']}`",
        f"- Candidate cells searched: `{report['dataset']['candidate_count']}`",
        "",
        "## Top Candidates",
        "| Candidate | Family | Dir | N | WR | Avg | Sum | T3R | Holdout N | Holdout Avg | Holdout T3R | No-overlap N | No-overlap T3R | Folds +sum/+t3r | Worst | TailN | Readiness |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["ranked"][:30]:
        s, h, noov, f = r["summary"], r["holdout"], r["no_overlap"], r["folds"]
        wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        havg = "" if h["avg"] is None else f"{float(h['avg']):+.1f}"
        lines.append(
            f"| {r['name']} | {r['family']} | {r['direction']} | {s['n']} | {wr} | {avg} | {float(s['sum']):+.1f} | {float(s['t3r']):+.1f} | "
            f"{h['n']} | {havg} | {float(h['t3r']):+.1f} | {noov['n']} | {float(noov['t3r']):+.1f} | "
            f"{f['positive_sum_folds']}/{f['positive_t3r_folds']} | {s['worst']} | {s['tail100_n']} | {r['readiness']} |"
        )
    lines.extend(
        [
            "",
            "## Family Summary",
            "| Family | Best candidate | N | WR | Avg | T3R | Readiness |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for fam, row in report["family_best"].items():
        s = row["summary"]
        wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        lines.append(f"| {fam} | {row['name']} | {s['n']} | {wr} | {avg} | {float(s['t3r']):+.1f} | {row['readiness']} |")
    lines.extend(
        [
            "",
            "## Multiple-Comparison Permutation",
            "```json",
            json.dumps(report["permutation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Direct Answers",
        ]
    )
    for item in report["answers"]:
        lines.append(f"- {item}")
    lines.extend(["", "## State Diagnostics"])
    for key, block in report["diagnostics"].items():
        lines.append(f"### {key}")
        for sub, s in block.items():
            wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
            avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
            lines.append(f"- `{sub}`: N={s['n']} WR={wr} avg={avg} T3R={float(s['t3r']):+.1f}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows, meta = build_buy_dataset()
    candidates = build_candidates(rows)
    evaluated = []
    for c in candidates:
        row = evaluate(c)
        row["_rows"] = c.rows
        evaluated.append(row)
    ranked = sorted(
        evaluated,
        key=lambda r: (
            r["readiness"] == "PAPER_CANDIDATE",
            r["readiness"] == "SHADOW_ONLY_TAIL",
            float(r["summary"].get("t3r") or 0.0),
            float(r["summary"].get("sum") or 0.0),
            int(r["summary"].get("n") or 0),
        ),
        reverse=True,
    )
    family_best = {}
    for fam in sorted({r["family"] for r in ranked}):
        fam_rows = [r for r in ranked if r["family"] == fam]
        family_best[fam] = fam_rows[0] if fam_rows else None
    perm_input = [{k: v for k, v in r.items()} for r in ranked]
    perm = permutation_maxstat(perm_input, iterations=500)
    for r in ranked:
        r.pop("_rows", None)
    for r in family_best.values():
        if r:
            r.pop("_rows", None)
    best_fade = family_best.get("BUY_TO_SHORT_FADE")
    best_cont = family_best.get("BUY_TO_LONG_CONT")
    best_same = family_best.get("BUY_TO_LONG_SAME_PROP")
    answers = []
    if best_cont:
        answers.append(f"ETH BUY -> LONG continuation best: `{best_cont['name']}` {best_cont['readiness']} N={best_cont['summary']['n']} avg={best_cont['summary']['avg']} T3R={best_cont['summary']['t3r']}.")
    if best_same:
        answers.append(f"ETH BUY -> LONG same-side propagation best: `{best_same['name']}` {best_same['readiness']} N={best_same['summary']['n']} avg={best_same['summary']['avg']} T3R={best_same['summary']['t3r']}.")
    if best_fade:
        answers.append(f"ETH BUY -> SHORT mean-reversion best: `{best_fade['name']}` {best_fade['readiness']} N={best_fade['summary']['n']} avg={best_fade['summary']['avg']} T3R={best_fade['summary']['t3r']}.")
    passers = [r for r in ranked if r["readiness"] == "PAPER_CANDIDATE"]
    if passers:
        answers.append(f"Potential new frequency exists as shadow/paper candidates: {', '.join(r['name'] for r in passers[:5])}.")
    else:
        answers.append("No BUY-side cell reached PAPER_CANDIDATE under holdout + folds + no-overlap gates.")
    if perm.get("p_right") is not None:
        if float(perm["p_right"]) <= 0.05:
            answers.append("Max-stat permutation says at least one searched BUY-side cell exceeds the 95% null; still needs forward shadow before live.")
        else:
            answers.append("Max-stat permutation does not clear 95% after multiple-comparison correction; treat apparent winners as research-only.")
    diagnostics = {
        "BUY_anchor_SHORT_4h_by_state": group_table(rows, "state"),
        "BUY_anchor_SHORT_4h_by_session": group_table(rows, "session"),
        "BUY_anchor_SHORT_4h_by_dow": group_table(rows, "dow_name"),
    }
    report = {
        "generated_at_utc": utc_now(),
        "dataset": {**meta, "candidate_count": len(candidates)},
        "ranked": ranked,
        "family_best": family_best,
        "permutation": perm,
        "answers": answers,
        "diagnostics": diagnostics,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
