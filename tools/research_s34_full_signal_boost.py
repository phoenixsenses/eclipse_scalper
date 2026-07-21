"""S34 full signal boost gauntlet.

Research-only. This script does not touch live execution, shadow state, .env,
leverage, or sizing. It consolidates the current next questions:

1. hour17 confidence / tail / entry / exit tests.
2. SHORT_NOISY BTC-confirmed portfolio leg.
3. BUY-side fade as a separate diversification leg.
4. Cross-asset, funding, book, and OFI separators.

Outputs:
  reports/research/s34/S34_FULL_SIGNAL_BOOST.json
  reports/research/s34/S34_FULL_SIGNAL_BOOST.md
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    MarkIndex,
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_FULL_SIGNAL_BOOST.json"
OUT_MD = OUT_DIR / "S34_FULL_SIGNAL_BOOST.md"

FEE_BPS = 5.0
MC_ITER = 1000
LOOKBACK_MS = 400 * 24 * 3600_000
ETH_THRESHOLDS = (150_000.0, 200_000.0)
PROP_THRESH = 50_000.0
H17_MIN_HOUR = 17
REGIME_LOOKBACK_7D_MS = 7 * 24 * 3600_000
SYNC_WIN_MS = 10 * 60_000
NOISY_LO_MS = 60_000
NOISY_HI_MS = 30 * 60_000
HOUR17_HOLD_MS = 6 * 3600_000
BUY_FADE_HOLD_MS = 45 * 60_000
BUY_FADE_SL_BPS = 75.0


def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def hour_of(ts_ms: int) -> int:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).hour


def dow_of(ts_ms: int) -> int:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).weekday()


def session_of(ts_ms: int) -> str:
    hour = hour_of(ts_ms)
    if 7 <= hour < 13:
        return "EUROPE"
    if 13 <= hour < 21:
        return "US"
    return "OFF"


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> float:
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def liq_sum(conn: sqlite3.Connection, symbol: str, side: str, lo: int, hi: int, min_notional: float = 0.0) -> float:
    return scalar(
        conn,
        """
        SELECT COALESCE(SUM(notional),0)
        FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?
        """,
        (symbol, side, int(lo), int(hi), float(min_notional)),
    )


def liq_count(conn: sqlite3.Connection, symbol: str, side: str, lo: int, hi: int, min_notional: float) -> int:
    return int(
        scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM liquidations
            WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?
            """,
            (symbol, side, int(lo), int(hi), float(min_notional)),
        )
    )


def liq_max(conn: sqlite3.Connection, symbol: str, side: str, lo: int, hi: int, min_notional: float = 0.0) -> float:
    return scalar(
        conn,
        """
        SELECT COALESCE(MAX(notional),0)
        FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?
        """,
        (symbol, side, int(lo), int(hi), float(min_notional)),
    )


def liq_first_ts(
    conn: sqlite3.Connection,
    symbol: str,
    side: str,
    lo: int,
    hi: int,
    min_notional: float,
) -> int | None:
    row = conn.execute(
        """
        SELECT ts_ms
        FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?
        ORDER BY ts_ms ASC
        LIMIT 1
        """,
        (symbol, side, int(lo), int(hi), float(min_notional)),
    ).fetchone()
    return int(row[0]) if row else None


def mark_bps(marks: MarkIndex, ts_ms: int, lookback_ms: int) -> float:
    a = marks.at_or_before(int(ts_ms) - int(lookback_ms))
    b = marks.at_or_before(int(ts_ms))
    if not a or not b or float(a[1]) <= 0:
        return 0.0
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def mark_price_at(marks: MarkIndex, ts_ms: int) -> float | None:
    row = marks.at_or_after(int(ts_ms))
    return float(row[1]) if row else None


def signed_gross(direction: str, entry: float, exit_: float) -> float:
    raw = (float(exit_) - float(entry)) / float(entry) * 10_000.0
    return raw if direction == "LONG" else -raw


def hold_gross(
    marks: MarkIndex,
    direction: str,
    entry_ts: int,
    hold_ms: int,
    *,
    stop_bps: float | None = None,
) -> float | None:
    entry_row = marks.at_or_after(int(entry_ts))
    exit_row = marks.at_or_before(int(entry_ts) + int(hold_ms))
    if not entry_row or not exit_row or float(entry_row[1]) <= 0:
        return None
    entry = float(entry_row[1])
    if stop_bps is not None:
        if direction == "LONG":
            stop_px = entry * (1.0 - float(stop_bps) / 10_000.0)
            for _, px in marks.slice_range(int(entry_row[0]), int(entry_ts) + int(hold_ms)):
                if float(px) <= stop_px:
                    return -float(stop_bps)
        else:
            stop_px = entry * (1.0 + float(stop_bps) / 10_000.0)
            for _, px in marks.slice_range(int(entry_row[0]), int(entry_ts) + int(hold_ms)):
                if float(px) >= stop_px:
                    return -float(stop_bps)
    return signed_gross(direction, entry, float(exit_row[1]))


def profit_lock_gross(
    marks: MarkIndex,
    direction: str,
    entry_ts: int,
    hold_ms: int,
    trigger_bps: float,
    lock_bps: float,
) -> float | None:
    entry_row = marks.at_or_after(int(entry_ts))
    if not entry_row or float(entry_row[1]) <= 0:
        return None
    entry = float(entry_row[1])
    armed = False
    target_end = int(entry_ts) + int(hold_ms)
    for _, px_raw in marks.slice_range(int(entry_row[0]), target_end):
        px = float(px_raw)
        gross = signed_gross(direction, entry, px)
        if gross >= float(trigger_bps):
            armed = True
        if armed and gross <= float(lock_bps):
            return float(lock_bps)
    exit_row = marks.at_or_before(target_end)
    if not exit_row:
        return None
    return signed_gross(direction, entry, float(exit_row[1]))


def time_damage_gross(marks: MarkIndex, entry_ts: int, hold_ms: int, damage_check_ms: int) -> float | None:
    entry_row = marks.at_or_after(int(entry_ts))
    check_row = marks.at_or_before(int(entry_ts) + int(damage_check_ms))
    exit_row = marks.at_or_before(int(entry_ts) + int(hold_ms))
    if not entry_row or not check_row or not exit_row or float(entry_row[1]) <= 0:
        return None
    check_gross = signed_gross("LONG", float(entry_row[1]), float(check_row[1]))
    if check_gross < 0:
        return check_gross
    return signed_gross("LONG", float(entry_row[1]), float(exit_row[1]))


def ofi(conn: sqlite3.Connection, symbol: str, lo: int, hi: int) -> dict[str, float]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),0),
          COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),0)
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        """,
        (symbol, int(lo), int(hi)),
    ).fetchone()
    buy = float(row[0] or 0.0) if row else 0.0
    sell = float(row[1] or 0.0) if row else 0.0
    total = buy + sell
    return {
        "ofi_notional": buy - sell,
        "ofi_ratio": (buy - sell) / total if total > 0 else 0.0,
        "buy_notional": buy,
        "sell_notional": sell,
    }


def funding_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, float | None]:
    row = conn.execute(
        """
        SELECT funding_rate, next_funding_time_ms
        FROM mark_prices
        WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return {"funding_rate": None, "minutes_to_funding": None, "minutes_since_8h_slot": None}
    next_funding = int(row[1]) if row[1] is not None else None
    slot_ms = 8 * 3600_000
    return {
        "funding_rate": float(row[0]) if row[0] is not None else None,
        "minutes_to_funding": ((next_funding - int(ts_ms)) / 60_000.0) if next_funding else None,
        "minutes_since_8h_slot": (int(ts_ms) % slot_ms) / 60_000.0,
    }


def pctile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def terciles(values: list[float]) -> tuple[float, float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if len(vals) < 3:
        return (vals[0] if vals else 0.0, vals[-1] if vals else 0.0)
    return vals[len(vals) // 3], vals[(2 * len(vals)) // 3]


def no_overlap(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    busy_until = -1
    kept = []
    for trade in sorted(trades, key=lambda r: int(r["entry_ts_ms"])):
        if int(trade["entry_ts_ms"]) >= busy_until:
            kept.append(trade)
            busy_until = int(trade["exit_ts_ms"])
    return kept


def mc_p(vals: list[float], avg: float) -> float | None:
    if len(vals) < 8:
        return None
    rng = random.Random(7)
    abs_vals = [abs(float(v)) for v in vals]
    hits = 0
    for _ in range(MC_ITER):
        sample = sum(rng.choice((-1.0, 1.0)) * v for v in abs_vals) / len(abs_vals)
        if sample >= avg:
            hits += 1
    return round(hits / MC_ITER, 3)


def wf(vals: list[float], folds: int = 5) -> str | None:
    if len(vals) < folds:
        return None
    n = len(vals)
    pos = 0
    for i in range(folds):
        part = vals[i * n // folds : (i + 1) * n // folds]
        if sum(part) > 0:
            pos += 1
    return f"{pos}/{folds}"


def stats_from_gross(gross_vals: list[float], months: float, *, fee_bps: float = FEE_BPS) -> dict[str, Any]:
    vals = [float(v) - float(fee_bps) for v in gross_vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    avg = sum(vals) / len(vals)
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for value in vals:
        cum += value
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    cut = max(1, int(len(vals) * 0.70))
    hold = vals[cut:]
    return {
        "n": len(vals),
        "per_month": round(len(vals) / months, 1),
        "wr": round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1),
        "avg": round(avg, 1),
        "total": round(sum(vals), 1),
        "pnl_per_month": round(sum(vals) / months, 1),
        "worst": round(min(vals), 1),
        "best": round(max(vals), 1),
        "tail100": sum(1 for v in vals if v < -100.0),
        "tail200": sum(1 for v in vals if v < -200.0),
        "mdd": round(mdd, 1),
        "t3r": round(sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals), 1),
        "mc_p": mc_p(vals, avg),
        "wf": wf(vals),
        "holdout_wr": round(100.0 * sum(1 for v in hold if v > 0) / len(hold), 1) if hold else None,
        "holdout_avg": round(sum(hold) / len(hold), 1) if hold else None,
    }


def stats_from_trades(trades: list[dict[str, Any]], months: float) -> dict[str, Any]:
    return stats_from_gross([float(t["gross_bps"]) for t in trades], months)


def month_span(events: list[dict[str, Any]]) -> float:
    if len(events) < 2:
        return 1.0
    first = min(int(e["ts_ms"]) for e in events)
    last = max(int(e["ts_ms"]) for e in events)
    return max(1.0, (last - first) / 86_400_000.0 / 30.0)


def split_train_test(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cut = int(len(events) * 0.70)
    return events[:cut], events[cut:]


def bin_gate(feature: str, bin_name: str, q1: float, q2: float) -> Callable[[dict[str, Any]], bool]:
    if bin_name == "lo":
        return lambda row: float(row.get(feature) or 0.0) < q1
    if bin_name == "hi":
        return lambda row: float(row.get(feature) or 0.0) >= q2
    return lambda row: q1 <= float(row.get(feature) or 0.0) < q2


@dataclass
class DataPack:
    conn: sqlite3.Connection
    eth_marks: MarkIndex
    btc_marks: MarkIndex
    sol_marks: MarkIndex
    now_ms: int
    start_ms: int


def build_sell_events(pack: DataPack, threshold: float) -> list[dict[str, Any]]:
    liqs = load_liquidations(pack.conn, "ETHUSDT", "SELL", pack.start_ms, pack.now_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=300,
        min_gap_sec=900,
        thresholds=(float(threshold),),
        accel_window_sec=30,
    )
    events: list[dict[str, Any]] = []
    for anchor in anchors:
        ts = int(anchor.anchor_ts_ms)
        entry = mark_price_at(pack.eth_marks, ts)
        if entry is None:
            continue
        eth1h = mark_bps(pack.eth_marks, ts, 3600_000)
        btc4h = mark_bps(pack.btc_marks, ts, 4 * 3600_000)
        btc7d = mark_bps(pack.btc_marks, ts, REGIME_LOOKBACK_7D_MS)
        bull = eth1h > 20.0 and btc4h > 50.0
        session = session_of(ts)
        regime = btc4h < 0.0 or btc7d < 0.0
        if bull or session == "EUROPE" or not regime:
            continue
        rn = float(anchor.running_notional)
        sync_sell_pre = liq_sum(pack.conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts) + liq_sum(
            pack.conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts
        )
        btc_conc = liq_max(pack.conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts)
        sol_conc = liq_max(pack.conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts)
        book = book_features_at(pack.conn, "ETHUSDT", ts, 30) or {}
        pre_ofi = ofi(pack.conn, "ETHUSDT", ts - 5 * 60_000, ts)
        post_ofi = ofi(pack.conn, "ETHUSDT", ts, ts + 60_000)
        fund = funding_at(pack.conn, "ETHUSDT", ts)
        events.append(
            {
                "ts_ms": ts,
                "utc": iso(ts),
                "threshold": float(threshold),
                "entry_price": entry,
                "running_notional": rn,
                "dominance": float(anchor.running_single_liq_dominance),
                "accel": float(anchor.running_accel),
                "liq_count": float(anchor.running_liq_count),
                "prebuild": float(liq_count(pack.conn, "ETHUSDT", "SELL", ts - 30 * 60_000, ts - 1000, PROP_THRESH)),
                "n2h": float(liq_count(pack.conn, "ETHUSDT", "SELL", ts - 2 * 3600_000, ts - 1000, PROP_THRESH)),
                "density24": float(liq_count(pack.conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, threshold)),
                "hour": float(hour_of(ts)),
                "dow": float(dow_of(ts)),
                "session": session,
                "eth1h": eth1h,
                "btc5m": mark_bps(pack.btc_marks, ts, 5 * 60_000),
                "btc4h": btc4h,
                "btc7d": btc7d,
                "btc3d": mark_bps(pack.btc_marks, ts, 3 * 24 * 3600_000),
                "sol4h": mark_bps(pack.sol_marks, ts, 4 * 3600_000),
                "sync_sell_pre": sync_sell_pre,
                "sync_ratio": sync_sell_pre / rn if rn > 0 else 0.0,
                "btc_conc_pre": btc_conc,
                "sol_conc_pre": sol_conc,
                "be_ratio_pre": btc_conc / rn if rn > 0 else 0.0,
                "spread_bps": float(book.get("spread_bps") or 0.0),
                "book_imbalance": float(book.get("book_imbalance") or 0.0),
                "bid_depth_usd": float(book.get("bid_depth_usd") or 0.0),
                "ask_depth_usd": float(book.get("ask_depth_usd") or 0.0),
                "vdepth_bps": float(book.get("vdepth_bps") or 0.0),
                "ofi_pre_ratio": pre_ofi["ofi_ratio"],
                "ofi_pre_notional": pre_ofi["ofi_notional"],
                "ofi_0_60_ratio": post_ofi["ofi_ratio"],
                "ofi_0_60_notional": post_ofi["ofi_notional"],
                "funding_rate": fund["funding_rate"] or 0.0,
                "minutes_to_funding": fund["minutes_to_funding"] if fund["minutes_to_funding"] is not None else 999.0,
                "minutes_since_8h_slot": fund["minutes_since_8h_slot"] if fund["minutes_since_8h_slot"] is not None else 999.0,
                "gross_h6": hold_gross(pack.eth_marks, "LONG", ts, HOUR17_HOLD_MS),
            }
        )
    events.sort(key=lambda r: int(r["ts_ms"]))
    return [e for e in events if e.get("gross_h6") is not None]


def base_hour17(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if float(e["hour"]) >= H17_MIN_HOUR]


def make_trade(name: str, direction: str, entry_ts: int, hold_ms: int, gross: float, **extra: Any) -> dict[str, Any]:
    out = {
        "name": name,
        "direction": direction,
        "entry_ts_ms": int(entry_ts),
        "exit_ts_ms": int(entry_ts) + int(hold_ms),
        "gross_bps": float(gross),
    }
    out.update(extra)
    return out


def hour17_trades(events: list[dict[str, Any]], marks: MarkIndex, hold_ms: int = HOUR17_HOLD_MS) -> list[dict[str, Any]]:
    trades = []
    for event in base_hour17(events):
        gross = hold_gross(marks, "LONG", int(event["ts_ms"]), hold_ms)
        if gross is not None:
            trades.append(make_trade("H17_LONG", "LONG", int(event["ts_ms"]), hold_ms, gross, event=event))
    return trades


def train_best_single_features(events: list[dict[str, Any]], features: list[str], months: float) -> list[dict[str, Any]]:
    train, test = split_train_test(events)
    all_test = stats_from_gross([float(e["gross_h6"]) for e in test], max(1.0, months * 0.30))
    rows = []
    for feature in features:
        vals = [float(e.get(feature) or 0.0) for e in train]
        if len(set(vals)) < 3:
            continue
        q1, q2 = terciles(vals)
        bins = {
            "lo": [float(e["gross_h6"]) for e in train if float(e.get(feature) or 0.0) < q1],
            "mid": [float(e["gross_h6"]) for e in train if q1 <= float(e.get(feature) or 0.0) < q2],
            "hi": [float(e["gross_h6"]) for e in train if float(e.get(feature) or 0.0) >= q2],
        }
        best = max(bins, key=lambda b: stats_from_gross(bins[b], max(1.0, months * 0.70)).get("avg", -1e9))
        gate = bin_gate(feature, best, q1, q2)
        test_stats = stats_from_gross([float(e["gross_h6"]) for e in test if gate(e)], max(1.0, months * 0.30))
        full_stats = stats_from_gross([float(e["gross_h6"]) for e in events if gate(e)], months)
        noov_stats = stats_from_trades(no_overlap([make_trade(feature, "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, float(e["gross_h6"])) for e in events if gate(e)]), months)
        rows.append(
            {
                "feature": feature,
                "best_bin": best,
                "q1": round(q1, 4),
                "q2": round(q2, 4),
                "test_lift_avg": round((test_stats.get("avg") or 0.0) - (all_test.get("avg") or 0.0), 1),
                "test": test_stats,
                "full": full_stats,
                "no_overlap": noov_stats,
            }
        )
    rows.sort(key=lambda r: float(r["test_lift_avg"]), reverse=True)
    return rows


def hour17_confidence_suite(events: list[dict[str, Any]], marks: MarkIndex, months: float) -> dict[str, Any]:
    h17 = base_hour17(events)
    features = [
        "sync_sell_pre",
        "btc4h",
        "btc7d",
        "btc3d",
        "running_notional",
        "n2h",
        "prebuild",
        "density24",
        "be_ratio_pre",
        "sync_ratio",
        "spread_bps",
        "book_imbalance",
        "bid_depth_usd",
        "ask_depth_usd",
        "ofi_pre_ratio",
        "ofi_0_60_ratio",
        "minutes_to_funding",
        "funding_rate",
    ]
    ranking = train_best_single_features(h17, features, months)
    top = ranking[:5]
    combos = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a = top[i]
            b = top[j]
            ga = bin_gate(a["feature"], a["best_bin"], float(a["q1"]), float(a["q2"]))
            gb = bin_gate(b["feature"], b["best_bin"], float(b["q1"]), float(b["q2"]))
            sub = [e for e in h17 if ga(e) and gb(e)]
            trades = [make_trade("combo", "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, float(e["gross_h6"])) for e in sub]
            combos.append(
                {
                    "combo": f"{a['feature']}={a['best_bin']} & {b['feature']}={b['best_bin']}",
                    "full": stats_from_gross([float(e["gross_h6"]) for e in sub], months),
                    "no_overlap": stats_from_trades(no_overlap(trades), months),
                }
            )
    combos.sort(key=lambda r: (r["no_overlap"].get("avg", -1e9), r["no_overlap"].get("n", 0)), reverse=True)

    by_hour = {}
    for hour in range(17, 24):
        sub = [e for e in h17 if int(e["hour"]) == hour]
        by_hour[str(hour)] = stats_from_gross([float(e["gross_h6"]) for e in sub], months)
    slices = {
        "17_19": stats_from_gross([float(e["gross_h6"]) for e in h17 if 17 <= int(e["hour"]) <= 19], months),
        "20_21": stats_from_gross([float(e["gross_h6"]) for e in h17 if 20 <= int(e["hour"]) <= 21], months),
        "22_23": stats_from_gross([float(e["gross_h6"]) for e in h17 if 22 <= int(e["hour"]) <= 23], months),
    }

    return {
        "base": stats_from_trades(hour17_trades(events, marks), months),
        "base_no_overlap": stats_from_trades(no_overlap(hour17_trades(events, marks)), months),
        "feature_ranking": ranking[:12],
        "top_combos": combos[:12],
        "by_hour": by_hour,
        "hour_slices": slices,
    }


def hour17_tail_suite(events: list[dict[str, Any]], marks: MarkIndex, months: float) -> dict[str, Any]:
    h17 = base_hour17(events)
    base_trades = hour17_trades(events, marks)
    base_noov = no_overlap(base_trades)
    vetoes: dict[str, Callable[[dict[str, Any]], bool]] = {
        "exclude_be_ratio_ge2": lambda e: float(e["be_ratio_pre"]) < 2.0,
        "exclude_btc_conc_ge1m": lambda e: float(e["btc_conc_pre"]) < 1_000_000.0,
        "exclude_sync_100_200k": lambda e: not (100_000.0 <= float(e["sync_sell_pre"]) < 200_000.0),
        "exclude_spread_gt_0p35": lambda e: float(e["spread_bps"]) <= 0.35,
        "only_bid_depth_ge100k": lambda e: float(e["bid_depth_usd"]) >= 100_000.0,
        "only_book_bid_support": lambda e: float(e["book_imbalance"]) >= 0.0,
        "exclude_sat_sun": lambda e: int(e["dow"]) not in (5, 6),
        "exclude_btc5m_lt_minus50": lambda e: float(e["btc5m"]) >= -50.0,
        "exclude_near_funding_30m": lambda e: float(e["minutes_to_funding"]) > 30.0,
    }
    veto_rows = {}
    for name, gate in vetoes.items():
        trades = [
            make_trade(name, "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, float(e["gross_h6"]), event=e)
            for e in h17
            if gate(e)
        ]
        dropped = [e for e in h17 if not gate(e)]
        veto_rows[name] = {
            "kept": stats_from_trades(trades, months),
            "kept_no_overlap": stats_from_trades(no_overlap(trades), months),
            "dropped": stats_from_gross([float(e["gross_h6"]) for e in dropped], months),
        }
    worst = sorted(
        [
            {
                "utc": iso(int(t["entry_ts_ms"])),
                "net_bps": round(float(t["gross_bps"]) - FEE_BPS, 1),
                "hour": int(t["event"]["hour"]),
                "dow": int(t["event"]["dow"]),
                "sync_sell_pre": round(float(t["event"]["sync_sell_pre"]), 1),
                "be_ratio_pre": round(float(t["event"]["be_ratio_pre"]), 2),
                "btc5m": round(float(t["event"]["btc5m"]), 1),
                "btc4h": round(float(t["event"]["btc4h"]), 1),
                "btc7d": round(float(t["event"]["btc7d"]), 1),
                "spread_bps": round(float(t["event"]["spread_bps"]), 3),
                "bid_depth_usd": round(float(t["event"]["bid_depth_usd"]), 1),
                "book_imbalance": round(float(t["event"]["book_imbalance"]), 3),
                "ofi_0_60_ratio": round(float(t["event"]["ofi_0_60_ratio"]), 3),
                "minutes_to_funding": round(float(t["event"]["minutes_to_funding"]), 1),
            }
            for t in base_trades
        ],
        key=lambda r: float(r["net_bps"]),
    )[:12]
    return {
        "base_no_overlap": stats_from_trades(base_noov, months),
        "vetoes": veto_rows,
        "worst_cards": worst,
    }


def hour17_entry_exit_suite(events: list[dict[str, Any]], marks: MarkIndex, months: float) -> dict[str, Any]:
    h17 = base_hour17(events)
    entry = {}
    for delay_min in (0, 1, 5, 15, 30, 60):
        trades = []
        delay_ms = delay_min * 60_000
        for e in h17:
            gross = hold_gross(marks, "LONG", int(e["ts_ms"]) + delay_ms, HOUR17_HOLD_MS)
            if gross is not None:
                trades.append(make_trade(f"d{delay_min}", "LONG", int(e["ts_ms"]) + delay_ms, HOUR17_HOLD_MS, gross))
        entry[f"delay_{delay_min}m"] = {
            "full": stats_from_trades(trades, months),
            "no_overlap": stats_from_trades(no_overlap(trades), months),
        }

    gates = {
        "d1_ofi_pos": lambda e: float(e["ofi_0_60_notional"]) > 0.0,
        "d1_bid_support": lambda e: float(e["book_imbalance"]) >= 0.0,
        "d1_spread_clean": lambda e: float(e["spread_bps"]) <= 0.35,
        "d1_bid100k": lambda e: float(e["bid_depth_usd"]) >= 100_000.0,
        "d1_ofi_pos_bid_support": lambda e: float(e["ofi_0_60_notional"]) > 0.0 and float(e["book_imbalance"]) >= 0.0,
    }
    for name, gate in gates.items():
        trades = []
        for e in h17:
            if not gate(e):
                continue
            gross = hold_gross(marks, "LONG", int(e["ts_ms"]) + 60_000, HOUR17_HOLD_MS)
            if gross is not None:
                trades.append(make_trade(name, "LONG", int(e["ts_ms"]) + 60_000, HOUR17_HOLD_MS, gross))
        entry[name] = {
            "full": stats_from_trades(trades, months),
            "no_overlap": stats_from_trades(no_overlap(trades), months),
        }

    exit_rows = {}
    for hold_hr in (4, 6, 8, 10):
        hold_ms = hold_hr * 3600_000
        trades = []
        for e in h17:
            gross = hold_gross(marks, "LONG", int(e["ts_ms"]), hold_ms)
            if gross is not None:
                trades.append(make_trade(f"h{hold_hr}", "LONG", int(e["ts_ms"]), hold_ms, gross))
        exit_rows[f"hold_{hold_hr}h"] = {
            "full": stats_from_trades(trades, months),
            "no_overlap": stats_from_trades(no_overlap(trades), months),
        }
    for trigger, lock in ((100, 50), (150, 75), (200, 100)):
        trades = []
        for e in h17:
            gross = profit_lock_gross(marks, "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, trigger, lock)
            if gross is not None:
                trades.append(make_trade(f"pl{trigger}_{lock}", "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, gross))
        exit_rows[f"profit_lock_{trigger}_{lock}"] = {
            "full": stats_from_trades(trades, months),
            "no_overlap": stats_from_trades(no_overlap(trades), months),
        }
    trades = []
    for e in h17:
        gross = time_damage_gross(marks, int(e["ts_ms"]), HOUR17_HOLD_MS, 3 * 3600_000)
        if gross is not None:
            trades.append(make_trade("time_damage_3h", "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, gross))
    exit_rows["time_damage_exit_if_neg_3h"] = {
        "full": stats_from_trades(trades, months),
        "no_overlap": stats_from_trades(no_overlap(trades), months),
    }
    for stop in (150, 200, 300):
        trades = []
        for e in h17:
            gross = hold_gross(marks, "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, stop_bps=stop)
            if gross is not None:
                trades.append(make_trade(f"stop{stop}", "LONG", int(e["ts_ms"]), HOUR17_HOLD_MS, gross))
        exit_rows[f"stop_{stop}"] = {
            "full": stats_from_trades(trades, months),
            "no_overlap": stats_from_trades(no_overlap(trades), months),
        }
    return {"entry": entry, "exit": exit_rows}


def short_noisy_suite(events: list[dict[str, Any]], pack: DataPack, months: float) -> dict[str, Any]:
    rows = {}
    for btc_thr in (500_000.0, 1_000_000.0, 2_000_000.0):
        for delay_min in (5, 10, 15):
            for hold_min in (90, 120, 180, 240):
                trades = []
                for e in events:
                    ts = int(e["ts_ms"])
                    prop_ts = liq_first_ts(pack.conn, "ETHUSDT", "SELL", ts + NOISY_LO_MS, ts + NOISY_HI_MS, PROP_THRESH)
                    btc_ts = liq_first_ts(
                        pack.conn,
                        "BTCUSDT",
                        "SELL",
                        ts + delay_min * 60_000,
                        ts + NOISY_HI_MS,
                        btc_thr,
                    )
                    if prop_ts is None or btc_ts is None:
                        continue
                    entry_ts = max(prop_ts, btc_ts)
                    gross = hold_gross(pack.eth_marks, "SHORT", entry_ts, hold_min * 60_000)
                    if gross is not None:
                        trades.append(
                            make_trade(
                                f"SHORT_NOISY_BTC{int(btc_thr/1000)}K_D{delay_min}_H{hold_min}",
                                "SHORT",
                                entry_ts,
                                hold_min * 60_000,
                                gross,
                                anchor_ts_ms=ts,
                                prop_delay_ms=prop_ts - ts,
                                btc_delay_ms=btc_ts - ts,
                            )
                        )
                key = f"btc{int(btc_thr/1000)}k_d{delay_min}_h{hold_min}"
                rows[key] = {
                    "full": stats_from_trades(trades, months),
                    "no_overlap": stats_from_trades(no_overlap(trades), months),
                }
    best = sorted(
        rows.items(),
        key=lambda kv: (kv[1]["no_overlap"].get("total", -1e9), kv[1]["no_overlap"].get("avg", -1e9)),
        reverse=True,
    )[:10]
    return {"grid": rows, "best_no_overlap_by_total": [{"name": k, **v} for k, v in best]}


def build_buy_events(pack: DataPack, threshold: float = 200_000.0) -> list[dict[str, Any]]:
    liqs = load_liquidations(pack.conn, "ETHUSDT", "BUY", pack.start_ms, pack.now_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=300,
        min_gap_sec=900,
        thresholds=(float(threshold),),
        accel_window_sec=30,
    )
    events = []
    for anchor in anchors:
        ts = int(anchor.anchor_ts_ms)
        entry = mark_price_at(pack.eth_marks, ts)
        if entry is None:
            continue
        session = session_of(ts)
        if session == "EUROPE":
            continue
        eth1h = mark_bps(pack.eth_marks, ts, 3600_000)
        btc4h = mark_bps(pack.btc_marks, ts, 4 * 3600_000)
        bear_squeeze_continuation_risk = eth1h < -20.0 and btc4h < -50.0
        if bear_squeeze_continuation_risk:
            continue
        book = book_features_at(pack.conn, "ETHUSDT", ts, 30) or {}
        silent30 = liq_first_ts(pack.conn, "ETHUSDT", "BUY", ts + 60_000, ts + 30 * 60_000, PROP_THRESH) is None
        events.append(
            {
                "ts_ms": ts,
                "utc": iso(ts),
                "session": session,
                "dow": float(dow_of(ts)),
                "running_notional": float(anchor.running_notional),
                "silent30": silent30,
                "sync_buy_pre": liq_sum(pack.conn, "BTCUSDT", "BUY", ts - SYNC_WIN_MS, ts)
                + liq_sum(pack.conn, "SOLUSDT", "BUY", ts - SYNC_WIN_MS, ts),
                "btc4h": btc4h,
                "btc7d": mark_bps(pack.btc_marks, ts, REGIME_LOOKBACK_7D_MS),
                "ask_depth_usd": float(book.get("ask_depth_usd") or 0.0),
                "book_imbalance": float(book.get("book_imbalance") or 0.0),
                "spread_bps": float(book.get("spread_bps") or 0.0),
            }
        )
    return events


def buy_fade_suite(pack: DataPack, months: float) -> dict[str, Any]:
    events = build_buy_events(pack, 200_000.0)
    def short_trade(e: dict[str, Any], entry_ts: int) -> dict[str, Any] | None:
        gross = hold_gross(pack.eth_marks, "SHORT", entry_ts, BUY_FADE_HOLD_MS, stop_bps=BUY_FADE_SL_BPS)
        if gross is None:
            return None
        return make_trade("BUY_FADE_H45_SL75", "SHORT", entry_ts, BUY_FADE_HOLD_MS, gross, event=e)

    all_t0 = [t for e in events if (t := short_trade(e, int(e["ts_ms"]))) is not None]
    silent_t0 = [t for e in events if e["silent30"] and (t := short_trade(e, int(e["ts_ms"]))) is not None]
    confirmed_t30 = [
        t for e in events if e["silent30"] and (t := short_trade(e, int(e["ts_ms"]) + 30 * 60_000)) is not None
    ]
    ask_depth = [float(e["ask_depth_usd"]) for e in events if float(e["ask_depth_usd"]) > 0.0]
    ask_med = pctile(ask_depth, 0.50) or 0.0
    ask_hi_t0 = [
        t
        for e in events
        if e["silent30"] and float(e["ask_depth_usd"]) >= ask_med and (t := short_trade(e, int(e["ts_ms"]))) is not None
    ]
    return {
        "events": len(events),
        "ask_depth_median": round(ask_med, 1),
        "all_t0_h45_sl75": {
            "full": stats_from_trades(all_t0, months),
            "no_overlap": stats_from_trades(no_overlap(all_t0), months),
        },
        "silent30_t0_h45_sl75_lookahead_label": {
            "full": stats_from_trades(silent_t0, months),
            "no_overlap": stats_from_trades(no_overlap(silent_t0), months),
        },
        "silent30_confirm_t30_h45_sl75_tradeable": {
            "full": stats_from_trades(confirmed_t30, months),
            "no_overlap": stats_from_trades(no_overlap(confirmed_t30), months),
        },
        "silent30_ask_depth_hi_t0": {
            "full": stats_from_trades(ask_hi_t0, months),
            "no_overlap": stats_from_trades(no_overlap(ask_hi_t0), months),
        },
    }


def portfolio_suite(
    hour_events: list[dict[str, Any]],
    h17_result: dict[str, Any],
    short_result: dict[str, Any],
    buy_result: dict[str, Any],
    pack: DataPack,
    months: float,
) -> dict[str, Any]:
    h17 = hour17_trades(hour_events, pack.eth_marks)
    # Rebuild best short route trades from the highest total no-overlap config.
    best_short_name = None
    if short_result["best_no_overlap_by_total"]:
        best_short_name = short_result["best_no_overlap_by_total"][0]["name"]
    short_trades: list[dict[str, Any]] = []
    if best_short_name:
        parts = best_short_name.replace("btc", "").replace("k", "").split("_")
        btc_thr = float(parts[0]) * 1000.0
        delay_min = int(parts[1].replace("d", ""))
        hold_min = int(parts[2].replace("h", ""))
        for e in hour_events:
            ts = int(e["ts_ms"])
            prop_ts = liq_first_ts(pack.conn, "ETHUSDT", "SELL", ts + NOISY_LO_MS, ts + NOISY_HI_MS, PROP_THRESH)
            btc_ts = liq_first_ts(pack.conn, "BTCUSDT", "SELL", ts + delay_min * 60_000, ts + NOISY_HI_MS, btc_thr)
            if prop_ts is None or btc_ts is None:
                continue
            entry_ts = max(prop_ts, btc_ts)
            gross = hold_gross(pack.eth_marks, "SHORT", entry_ts, hold_min * 60_000)
            if gross is not None:
                short_trades.append(make_trade("SHORT_NOISY_BEST", "SHORT", entry_ts, hold_min * 60_000, gross))

    buy_events = build_buy_events(pack, 200_000.0)
    buy_trades = []
    for e in buy_events:
        if not e["silent30"]:
            continue
        gross = hold_gross(pack.eth_marks, "SHORT", int(e["ts_ms"]), BUY_FADE_HOLD_MS, stop_bps=BUY_FADE_SL_BPS)
        if gross is not None:
            buy_trades.append(make_trade("BUY_FADE_T0_LABEL", "SHORT", int(e["ts_ms"]), BUY_FADE_HOLD_MS, gross))

    portfolios = {
        "h17_only": no_overlap(h17),
        "short_noisy_only": no_overlap(short_trades),
        "buy_fade_only": no_overlap(buy_trades),
        "h17_plus_short": no_overlap(h17 + short_trades),
        "h17_plus_buy": no_overlap(h17 + buy_trades),
        "all_three": no_overlap(h17 + short_trades + buy_trades),
    }
    return {
        "best_short_config": best_short_name,
        "portfolio_stats": {name: stats_from_trades(trades, months) for name, trades in portfolios.items()},
        "route_counts_in_all_three": {
            name: sum(1 for t in portfolios["all_three"] if t["name"] == name)
            for name in sorted({t["name"] for t in portfolios["all_three"]})
        },
    }


def render_stat(s: dict[str, Any]) -> str:
    if not s or int(s.get("n") or 0) == 0:
        return "N=0"
    return (
        f"N={s['n']} /mo={s.get('per_month')} WR={s.get('wr')}% avg={s.get('avg')} "
        f"total={s.get('total')} tail100={s.get('tail100')} mc={s.get('mc_p')} wf={s.get('wf')}"
    )


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# S34 Full Signal Boost Gauntlet",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Research-only. No live executor, shadow runner, .env, leverage, or sizing changes.",
        "",
        "## Scope",
        "",
        f"- months: `{payload['meta']['months']}`",
        f"- thresholds: `{payload['meta']['thresholds']}`",
        "",
    ]
    for thr, section in payload["hour17"].items():
        lines += [
            f"## Hour17 Threshold {thr}",
            "",
            f"- base: {render_stat(section['confidence']['base'])}",
            f"- base no-overlap: {render_stat(section['confidence']['base_no_overlap'])}",
            "",
            "### Feature Ranking",
            "",
            "| Feature | Bin | TEST lift | TEST | FULL | NOOV |",
            "|---|---:|---:|---|---|---|",
        ]
        for row in section["confidence"]["feature_ranking"][:8]:
            lines.append(
                f"| `{row['feature']}` | `{row['best_bin']}` | {row['test_lift_avg']} | "
                f"{render_stat(row['test'])} | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |"
            )
        lines += ["", "### Top Combos", "", "| Combo | FULL | NOOV |", "|---|---|---|"]
        for row in section["confidence"]["top_combos"][:8]:
            lines.append(f"| `{row['combo']}` | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |")
        lines += ["", "### Hour Slices", "", "| Slice | Stats |", "|---|---|"]
        for name, stat in section["confidence"]["hour_slices"].items():
            lines.append(f"| `{name}` | {render_stat(stat)} |")
        lines += ["", "### Tail Vetoes", "", "| Veto | Kept NOOV | Dropped |", "|---|---|---|"]
        for name, row in section["tail"]["vetoes"].items():
            lines.append(f"| `{name}` | {render_stat(row['kept_no_overlap'])} | {render_stat(row['dropped'])} |")
        lines += ["", "### Entry Tests", "", "| Test | FULL | NOOV |", "|---|---|---|"]
        for name, row in section["entry_exit"]["entry"].items():
            lines.append(f"| `{name}` | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |")
        lines += ["", "### Exit Tests", "", "| Test | FULL | NOOV |", "|---|---|---|"]
        for name, row in section["entry_exit"]["exit"].items():
            lines.append(f"| `{name}` | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |")
        lines += ["", "### Worst Hour17 Cards", "", "| UTC | Net | Hour | DOW | Sync | BE | BTC5m | Spread | BidDepth | Imb | OFI60 | ToFund |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for card in section["tail"]["worst_cards"][:8]:
            lines.append(
                f"| {card['utc']} | {card['net_bps']} | {card['hour']} | {card['dow']} | "
                f"{card['sync_sell_pre']} | {card['be_ratio_pre']} | {card['btc5m']} | {card['spread_bps']} | "
                f"{card['bid_depth_usd']} | {card['book_imbalance']} | {card['ofi_0_60_ratio']} | {card['minutes_to_funding']} |"
            )
        lines.append("")

    lines += ["## SHORT_NOISY BTC-Confirmed", "", "| Rank | Config | FULL | NOOV |", "|---:|---|---|---|"]
    for idx, row in enumerate(payload["short_noisy"]["best_no_overlap_by_total"][:10], start=1):
        lines.append(f"| {idx} | `{row['name']}` | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |")

    lines += ["", "## BUY-Side Fade", "", "| Variant | FULL | NOOV |", "|---|---|---|"]
    for name, row in payload["buy_fade"].items():
        if isinstance(row, dict) and "full" in row:
            lines.append(f"| `{name}` | {render_stat(row['full'])} | {render_stat(row['no_overlap'])} |")

    lines += ["", "## Portfolio", "", f"- best SHORT config: `{payload['portfolio']['best_short_config']}`", "", "| Portfolio | Stats |", "|---|---|"]
    for name, stat in payload["portfolio"]["portfolio_stats"].items():
        lines.append(f"| `{name}` | {render_stat(stat)} |")
    lines += ["", "## Route Counts In All-Three", "", "```json", json.dumps(payload["portfolio"]["route_counts_in_all_three"], indent=2), "```", ""]
    lines += [
        "## Read",
        "",
        "- Treat T0 silence-labelled BUY fade variants as research labels unless explicitly shown as confirmed/tradeable.",
        "- Promotion needs forward paper/shadow accumulation and operator sign-off.",
        "- Sizing/tail-budget remains separate and urgent; this report only ranks signals.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    random.seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - LOOKBACK_MS
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-200000")
        pack = DataPack(
            conn=conn,
            eth_marks=load_mark_index(conn, "ETHUSDT"),
            btc_marks=load_mark_index(conn, "BTCUSDT"),
            sol_marks=load_mark_index(conn, "SOLUSDT"),
            now_ms=now_ms,
            start_ms=start_ms,
        )
        events_by_thr = {}
        for threshold in ETH_THRESHOLDS:
            print(f"[build] SELL threshold={threshold:.0f}")
            events_by_thr[str(int(threshold / 1000)) + "K"] = build_sell_events(pack, threshold)
            print(f"[build] usable={len(events_by_thr[str(int(threshold / 1000)) + 'K'])}")
        primary_events = events_by_thr["200K"]
        months = month_span(primary_events)
        payload: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "meta": {
                "months": round(months, 2),
                "thresholds": list(events_by_thr.keys()),
                "events": {k: len(v) for k, v in events_by_thr.items()},
            },
            "hour17": {},
        }
        for label, events in events_by_thr.items():
            print(f"[hour17] {label}")
            m = month_span(events)
            payload["hour17"][label] = {
                "confidence": hour17_confidence_suite(events, pack.eth_marks, m),
                "tail": hour17_tail_suite(events, pack.eth_marks, m),
                "entry_exit": hour17_entry_exit_suite(events, pack.eth_marks, m),
            }
        print("[short_noisy]")
        payload["short_noisy"] = short_noisy_suite(primary_events, pack, months)
        print("[buy_fade]")
        payload["buy_fade"] = buy_fade_suite(pack, months)
        print("[portfolio]")
        payload["portfolio"] = portfolio_suite(
            primary_events,
            payload["hour17"]["200K"],
            payload["short_noisy"],
            payload["buy_fade"],
            pack,
            months,
        )

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(f"[done] {OUT_JSON}")
    print(f"[done] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
