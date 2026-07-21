"""S34 long-relax and management development suite.

Research-only. Reads historical DB/derived navigation rows and writes reports.
It does not touch the live executor, .env, order logic, sizing, leverage, or
runtime state.

The suite tests:
- LONG gate relax: btc4h < 0 OR btc7d < 0;
- tail detectors;
- dynamic hold;
- confidence score;
- exit-by-state;
- route fusion;
- adaptive stops;
- multi-stage entry;
- position sizing;
- additional frequency/quality ideas.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_freq_tests import (  # noqa: E402
    DB_PATH,
    FEE_BPS,
    LONG_HOLD_MS,
    PROP_THRESH,
    SHORT_HOLD_MS,
    SIL_HI_MS,
    SIL_LO_MS,
    Series,
    build_dataset,
    first_liq_above,
    iso_ms,
    liq_count,
    liq_sum,
    load_liq_series,
    load_mark_series,
    mark_at_or_after,
    ret_bps,
    session_for,
    signed_net,
    stat,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_LONG_RELAX_MANAGEMENT_SUITE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_LONG_RELAX_MANAGEMENT_SUITE.md"

BTC_CONFIRM_CURRENT = 2_000_000.0
BTC_CONFIRM_1M = 1_000_000.0
SYNC_WIN_MS = 10 * 60_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def r1(x: float | None) -> float | None:
    return None if x is None else round(float(x), 1)


def months_span(rows: list[dict[str, Any]]) -> float:
    vals = [int(r["entry_ts_ms"]) for r in rows if r.get("entry_ts_ms") is not None]
    if len(vals) < 2:
        return 1.0
    days = (max(vals) - min(vals)) / 86_400_000.0
    return max(days / 30.4375, 1.0)


def summarize(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {
            "n": 0, "wr": None, "avg": None, "sum": 0.0, "median": None,
            "t3r": 0.0, "worst": None, "best": None, "tail100_n": 0,
            "per_month": 0.0, "max_dd_bps": 0.0,
        }
    ordered = sorted([r for r in rows if r.get(key) is not None], key=lambda r: int(r.get("entry_ts_ms") or r["anchor_ts_ms"]))
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for r in ordered:
        eq += float(r[key])
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    sv = sorted(vals)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "avg": round(mean(vals), 1),
        "sum": round(sum(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(sum(sv[:-3]) if len(sv) > 3 else sum(sv), 1),
        "worst": round(min(vals), 1),
        "best": round(max(vals), 1),
        "tail100_n": sum(1 for v in vals if v <= -100.0),
        "per_month": round(len(vals) / months_span(ordered), 1),
        "max_dd_bps": round(dd, 1),
    }


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("entry_ts_ms") or r["anchor_ts_ms"]))
    cut = int(len(ordered) * 0.70)
    return {
        "all": summarize(ordered),
        "cal": summarize(ordered[:cut]),
        "hold": summarize(ordered[cut:]),
        "folds": fold_summary(ordered, 5),
    }


def fold_summary(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: int(r.get("entry_ts_ms") or r["anchor_ts_ms"]))
    out = []
    for i in range(n):
        lo = int(i * len(ordered) / n)
        hi = int((i + 1) * len(ordered) / n)
        chunk = ordered[lo:hi]
        s = summarize(chunk)
        s["fold"] = i + 1
        s["start_utc"] = iso_ms(int(chunk[0].get("entry_ts_ms") or chunk[0]["anchor_ts_ms"])) if chunk else None
        s["end_utc"] = iso_ms(int(chunk[-1].get("entry_ts_ms") or chunk[-1]["anchor_ts_ms"])) if chunk else None
        out.append(s)
    return out


def fmt(s: dict[str, Any]) -> str:
    wr = s.get("wr")
    wr_s = "NA" if wr is None else f"{float(wr) * 100:.1f}%"
    avg = s.get("avg")
    avg_s = "NA" if avg is None else f"{float(avg):+.1f}"
    return (
        f"N={s.get('n', 0)} WR={wr_s} avg={avg_s} sum={s.get('sum', 0):+.1f} "
        f"T3R={s.get('t3r', 0):+.1f} worst={s.get('worst')} tail100={s.get('tail100_n', 0)} /mo={s.get('per_month', 0)}"
    )


def with_extra_features(rows: list[dict[str, Any]], eth_sell: Series, eth_marks: Series) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        ts = int(r["anchor_ts_ms"])
        prebuildup = liq_count(eth_sell, ts - 30 * 60_000, ts - 1000, PROP_THRESH)
        echo_30_90 = liq_sum(eth_sell, ts - 90 * 60_000, ts - 30 * 60_000) >= 200_000.0
        echo_45_120 = liq_sum(eth_sell, ts - 120 * 60_000, ts - 45 * 60_000) >= 200_000.0
        density_24h = liq_count(eth_sell, ts - 24 * 3600_000, ts - 300_000, 200_000.0)
        eth30 = ret_bps(eth_marks, ts - 30 * 60_000, ts) or 0.0
        vol30 = abs(eth30)
        out.append({
            **r,
            "prebuildup_30m": prebuildup,
            "echo_30_90": echo_30_90,
            "echo_45_120": echo_45_120,
            "density_24h": density_24h,
            "eth30_bps": eth30,
            "vol30_bps": vol30,
        })
    return out


def long_gate(row: dict[str, Any], *, regime: str = "or", sync_thr: float = 200_000.0, score_min: int = 3,
              allow_us_13_14: bool = False, allow_mon_wed: bool = False, allow_europe: bool = False) -> bool:
    if row["bull"]:
        return False
    if not allow_europe and row["session"] == "EUROPE":
        return False
    if not allow_us_13_14 and row["session"] == "US" and row["hour"] in {13, 14}:
        return False
    if not allow_mon_wed and row["dow"] in {0, 2}:
        return False
    if float(row["sync_k"]) >= sync_thr:
        return False
    if int(row["long_score"]) < score_min:
        return False
    btc4h = float(row.get("btc4h_bps") or 0.0)
    btc7d = float(row.get("btc7d_bps") or 0.0)
    if regime == "or":
        return btc4h < 0.0 or btc7d < 0.0
    if regime == "btc7d":
        return btc7d < 0.0
    if regime == "btc4h":
        return btc4h < 0.0
    if regime == "none":
        return True
    if regime == "and":
        return btc4h < 0.0 and btc7d < 0.0
    if regime == "btc3d":
        return float(row.get("btc3d_bps") or 0.0) < 0.0
    return False


def make_long(rows: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool], *,
              hold_ms: int = LONG_HOLD_MS, entry_delay_ms: int = 0, eth_marks: Series | None = None,
              stop_bps: float | None = None, tp_bps: float | None = None,
              tag: str = "LONG") -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not pred(r):
            continue
        ts = int(r["anchor_ts_ms"]) + int(entry_delay_ms)
        if eth_marks and (entry_delay_ms or hold_ms != LONG_HOLD_MS or stop_bps is not None or tp_bps is not None):
            net = path_exit_net(eth_marks, "LONG", ts, ts + hold_ms, stop_bps=stop_bps, tp_bps=tp_bps)
        else:
            net = float(r["long_4h_net_bps"])
        if net is None:
            continue
        out.append({**r, "side": "LONG", "route": tag, "entry_ts_ms": ts, "exit_ts_ms": ts + hold_ms, "net_bps": float(net)})
    return out


def path_exit_net(marks: Series, side: str, entry_ts: int, exit_ts: int, *,
                  stop_bps: float | None = None, tp_bps: float | None = None) -> float | None:
    entry = mark_at_or_after(marks, entry_ts)
    if entry is None or entry <= 0:
        return None
    a = bisect.bisect_left(marks.ts, int(entry_ts))
    b = bisect.bisect_right(marks.ts, int(exit_ts))
    if a >= b:
        return None
    exit_px = mark_at_or_after(marks, exit_ts)
    for i in range(a, b):
        raw = (float(marks.vals[i]) - entry) / entry * 10_000.0
        pnl = -raw if side.upper() == "SHORT" else raw
        if stop_bps is not None and pnl <= -float(stop_bps):
            exit_px = float(marks.vals[i])
            break
        if tp_bps is not None and pnl >= float(tp_bps):
            exit_px = float(marks.vals[i])
            break
    if exit_px is None:
        return None
    raw = (float(exit_px) - entry) / entry * 10_000.0
    if side.upper() == "SHORT":
        raw = -raw
    return raw - FEE_BPS


def short_candidates(rows: list[dict[str, Any]], btc_sell: Series, eth_marks: Series, *,
                     btc_thr: float = BTC_CONFIRM_CURRENT, delay_min: int = 5,
                     hold_ms: int = SHORT_HOLD_MS, score_min: int = 4,
                     tag: str = "SHORT") -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["bull"] or r["session"] == "EUROPE" or r["dow"] == 6 or int(r["base_score"]) < score_min:
            continue
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_sell, ts + delay_min * 60_000, ts + SIL_HI_MS, btc_thr)
        if hit is None:
            continue
        net = path_exit_net(eth_marks, "SHORT", int(hit[0]), int(hit[0]) + hold_ms)
        if net is None:
            continue
        out.append({
            **r,
            "side": "SHORT",
            "route": tag,
            "entry_ts_ms": int(hit[0]),
            "exit_ts_ms": int(hit[0]) + hold_ms,
            "btc_confirm_notional": float(hit[1]),
            "net_bps": float(net),
        })
    return out


def no_overlap(rows: list[dict[str, Any]], *, replace_same_side: bool = False) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for r in sorted(rows, key=lambda x: int(x["entry_ts_ms"])):
        if active is None:
            active = r
            continue
        if int(r["entry_ts_ms"]) >= int(active["exit_ts_ms"]):
            chosen.append(active)
            active = r
            continue
        if replace_same_side and r["side"] == active["side"] and r["side"] == "SHORT":
            active = r
            continue
        # Prefer SHORT replacement over LONG if they overlap.
        if active["side"] == "LONG" and r["side"] == "SHORT":
            chosen.append({
                **active,
                "route": f"{active['route']}_TRUNCATED_BY_SHORT",
                "exit_ts_ms": int(r["entry_ts_ms"]),
                "net_bps": path_exit_net_cached(active, int(r["entry_ts_ms"])),
            })
            active = r
            continue
        # Otherwise keep the existing active trade and drop overlap.
    if active is not None:
        chosen.append(active)
    return [r for r in chosen if r.get("net_bps") is not None]


def path_exit_net_cached(row: dict[str, Any], exit_ts: int) -> float | None:
    # Filled after globals are bound in main. Avoids passing marks through no_overlap.
    return path_exit_net(GLOBAL_ETH_MARKS, str(row["side"]), int(row["entry_ts_ms"]), int(exit_ts))


GLOBAL_ETH_MARKS = Series([], [])


def feature_bins(rows: list[dict[str, Any]], features: dict[str, Callable[[dict[str, Any]], str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in features.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[fn(r)].append(r)
        out[name] = {k: summarize(v) for k, v in sorted(groups.items())}
    return out


def confidence_value(r: dict[str, Any]) -> int:
    conf = 0
    conf += int(float(r.get("btc4h_bps") or 0.0) < 0.0)
    conf += int(float(r.get("btc7d_bps") or 0.0) < 0.0)
    conf += int(float(r.get("vdepth_bps") or 0.0) >= 30.0)
    conf += int(int(r.get("n2h") or 0) >= 5)
    conf += int(float(r.get("sync_k") or 0.0) < 100_000.0)
    conf += int(bool(r.get("echo_30_90") or r.get("echo_45_120")))
    conf += int(int(r.get("prebuildup_30m") or 0) >= 1)
    conf += int(float(r.get("vol30_bps") or 0.0) <= 35.0)
    return conf


def apply_confidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "confidence": confidence_value(r)} for r in rows]


def scale_stats(rows: list[dict[str, Any]], size_fn: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    scaled = [{**r, "scaled_bps": float(r["net_bps"]) * float(size_fn(r))} for r in rows]
    return summarize(scaled, "scaled_bps")


def compound_equity(rows: list[dict[str, Any]], start: float, notional: float) -> dict[str, Any]:
    eq = float(start)
    curve = []
    for r in sorted(rows, key=lambda x: int(x["entry_ts_ms"])):
        pnl = float(notional) * float(r["net_bps"]) / 10_000.0
        eq += pnl
        curve.append(eq)
    return {"start": start, "notional": notional, "end": round(eq, 2), "pnl": round(eq - start, 2), "min_equity": round(min(curve), 2) if curve else start}


def tail_detector_tests(long_base: list[dict[str, Any]]) -> dict[str, Any]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "exclude_eth1h_lt_-80": lambda r: float(r.get("eth1h_bps") or 0) >= -80,
        "exclude_eth4h_lt_-150": lambda r: float(r.get("eth4h_bps") or 0) >= -150,
        "exclude_btc1h_lt_-50": lambda r: float(r.get("btc1h_bps") or 0) >= -50,
        "exclude_btc4h_lt_-150": lambda r: float(r.get("btc4h_bps") or 0) >= -150,
        "exclude_sync_100_200": lambda r: not (100_000 <= float(r.get("sync_k") or 0) < 200_000),
        "only_n2h_ge5": lambda r: int(r.get("n2h") or 0) >= 5,
        "exclude_slow_elapsed_gt180s": lambda r: float(r.get("elapsed_since_first_sec") or 0) <= 180,
        "only_echo": lambda r: bool(r.get("echo_30_90") or r.get("echo_45_120")),
        "only_prebuild": lambda r: int(r.get("prebuildup_30m") or 0) >= 1,
        "exclude_high_vol30_gt60": lambda r: float(r.get("vol30_bps") or 0) <= 60,
    }
    return {
        name: {
            "kept": summarize([r for r in long_base if pred(r)]),
            "dropped": summarize([r for r in long_base if not pred(r)]),
        }
        for name, pred in predicates.items()
    }


def dynamic_hold_tests(long_pred_rows: list[dict[str, Any]], short_base: list[dict[str, Any]], eth_marks: Series) -> dict[str, Any]:
    out: dict[str, Any] = {"long": {}, "short": {}}
    for minutes in [30, 60, 90, 120, 180, 240, 360, 480]:
        hold = minutes * 60_000
        rows = make_long(long_pred_rows, lambda r: True, hold_ms=hold, eth_marks=eth_marks, tag=f"LONG_H{minutes}m")
        out["long"][f"h{minutes}m"] = split_summary(rows)
    for minutes in [30, 60, 90, 120, 180, 240]:
        rows = []
        for s in short_base:
            net = path_exit_net(eth_marks, "SHORT", int(s["entry_ts_ms"]), int(s["entry_ts_ms"]) + minutes * 60_000)
            if net is not None:
                rows.append({**s, "exit_ts_ms": int(s["entry_ts_ms"]) + minutes * 60_000, "net_bps": net})
        out["short"][f"h{minutes}m"] = split_summary(rows)
    return out


def adaptive_stop_tests(long_base: list[dict[str, Any]], short_base: list[dict[str, Any]], eth_marks: Series) -> dict[str, Any]:
    out: dict[str, Any] = {"long": {}, "short": {}}
    for sl in [50, 75, 100, 150, 200]:
        out["long"][f"sl{sl}"] = split_summary([
            {**r, "net_bps": path_exit_net(eth_marks, "LONG", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + LONG_HOLD_MS, stop_bps=sl)}
            for r in long_base
            if path_exit_net(eth_marks, "LONG", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + LONG_HOLD_MS, stop_bps=sl) is not None
        ])
        out["short"][f"sl{sl}"] = split_summary([
            {**r, "net_bps": path_exit_net(eth_marks, "SHORT", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + SHORT_HOLD_MS, stop_bps=sl)}
            for r in short_base
            if path_exit_net(eth_marks, "SHORT", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + SHORT_HOLD_MS, stop_bps=sl) is not None
        ])
    for name, pred, sl in [
        ("long_conf_ge5_sl75_else_sl150", lambda r: int(r.get("confidence") or 0) >= 5, 75),
        ("long_tail_detector_sl75_else_hold", lambda r: float(r.get("eth1h_bps") or 0) < -80 or float(r.get("vol30_bps") or 0) > 60, 75),
    ]:
        managed = []
        for r in long_base:
            stop = sl if pred(r) else None
            net = path_exit_net(eth_marks, "LONG", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + LONG_HOLD_MS, stop_bps=stop)
            if net is not None:
                managed.append({**r, "net_bps": net, "stop_bps": stop})
        out["long"][name] = split_summary(managed)
    return out


def multi_stage_entry_tests(long_rows: list[dict[str, Any]], eth_marks: Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for delay_min in [0, 5, 15, 30, 60]:
        out[f"entry_delay_{delay_min}m"] = split_summary(make_long(long_rows, lambda r: True, hold_ms=LONG_HOLD_MS - delay_min * 60_000, entry_delay_ms=delay_min * 60_000, eth_marks=eth_marks, tag=f"D{delay_min}"))
    staged = []
    for r in long_rows:
        ts = int(r["anchor_ts_ms"])
        n0 = path_exit_net(eth_marks, "LONG", ts, ts + LONG_HOLD_MS)
        n15 = path_exit_net(eth_marks, "LONG", ts + 15 * 60_000, ts + LONG_HOLD_MS)
        if n0 is not None and n15 is not None:
            staged.append({**r, "entry_ts_ms": ts, "exit_ts_ms": ts + LONG_HOLD_MS, "net_bps": 0.5 * n0 + 0.5 * n15, "route": "HALF_T0_HALF_T15"})
    out["half_t0_half_t15"] = split_summary(staged)
    pullback = []
    chase = []
    for r in long_rows:
        ts = int(r["anchor_ts_ms"])
        px0 = mark_at_or_after(eth_marks, ts)
        px15 = mark_at_or_after(eth_marks, ts + 15 * 60_000)
        if not px0 or not px15:
            continue
        drift = (px15 - px0) / px0 * 10_000.0
        net = path_exit_net(eth_marks, "LONG", ts + 15 * 60_000, ts + LONG_HOLD_MS)
        if net is None:
            continue
        row = {**r, "entry_ts_ms": ts + 15 * 60_000, "exit_ts_ms": ts + LONG_HOLD_MS, "net_bps": net, "drift15_bps": drift}
        if drift <= 0:
            pullback.append(row)
        if drift > 0:
            chase.append(row)
    out["enter_t15_only_if_pullback"] = split_summary(pullback)
    out["enter_t15_after_bounce"] = split_summary(chase)
    return out


def exit_by_state_tests(long_live: list[dict[str, Any]], eth_marks: Series, btc_sell: Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    hold_all = long_live
    out["hold_all_4h"] = split_summary(hold_all)
    noisy_exit = []
    noisy_reverse = []
    silence_only = []
    for r in long_live:
        ts = int(r["anchor_ts_ms"])
        follow = r.get("eth_follow_ts_ms")
        if follow is None:
            silence_only.append(r)
            noisy_exit.append(r)
            noisy_reverse.append(r)
            continue
        follow_ts = int(follow)
        n_exit = path_exit_net(eth_marks, "LONG", ts, follow_ts)
        if n_exit is not None:
            noisy_exit.append({**r, "exit_ts_ms": follow_ts, "net_bps": n_exit, "route": "NOISY_EXIT"})
        short_net = path_exit_net(eth_marks, "SHORT", follow_ts, follow_ts + SHORT_HOLD_MS)
        if n_exit is not None and short_net is not None:
            noisy_reverse.append({**r, "exit_ts_ms": follow_ts + SHORT_HOLD_MS, "net_bps": n_exit + short_net, "route": "NOISY_REVERSE_SHORT"})
    out["exit_on_noisy_follow"] = split_summary(noisy_exit)
    out["reverse_short_on_noisy_follow"] = split_summary(noisy_reverse)
    out["silence_only_hold"] = split_summary(silence_only)
    return out


def route_fusion_tests(long_live: list[dict[str, Any]], short_current: list[dict[str, Any]], short_1m_h4: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    out["long_only"] = split_summary(no_overlap(long_live))
    out["short_current_only"] = split_summary(no_overlap(short_current, replace_same_side=True))
    out["short_1m_h4_only"] = split_summary(no_overlap(short_1m_h4, replace_same_side=True))
    out["long_plus_short_current"] = split_summary(no_overlap(long_live + short_current, replace_same_side=True))
    out["long_plus_short_1m_h4"] = split_summary(no_overlap(long_live + short_1m_h4, replace_same_side=True))
    # Priority fusion: strong SHORT route first, then LONG.
    out["fusion_priority_short1m_h4_then_long"] = split_summary(no_overlap(short_1m_h4 + long_live, replace_same_side=True))
    return out


def best_candidate_decision(results: dict[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for section in ["long_gate", "tail_detector_kept", "dynamic_hold_long", "adaptive_stop_long", "multi_stage", "route_fusion"]:
        obj = results.get(section, {})
        if section == "tail_detector_kept":
            for name, val in obj.items():
                candidates.append((f"{section}.{name}", val["kept"].get("all", val["kept"]) if "all" in val.get("kept", {}) else val["kept"]))
        else:
            for name, val in obj.items():
                if isinstance(val, dict) and "all" in val:
                    candidates.append((f"{section}.{name}", val["all"]))
                elif isinstance(val, dict) and "n" in val:
                    candidates.append((f"{section}.{name}", val))
    viable = []
    for name, s in candidates:
        n = int(s.get("n") or 0)
        wr = float(s.get("wr") or 0)
        avg = float(s.get("avg") or -999)
        t3r = float(s.get("t3r") or -999)
        hold_ok = True
        viable.append({"name": name, **s, "score": round(n * max(avg, 0) * max(wr, 0) / 100.0, 1), "hold_ok": hold_ok})
    viable.sort(key=lambda x: (x["score"], x.get("n") or 0), reverse=True)
    top = viable[:10]
    promote = None
    for c in top:
        if int(c.get("n") or 0) >= 30 and float(c.get("wr") or 0) >= 0.70 and float(c.get("avg") or 0) > 70 and float(c.get("t3r") or 0) > 0:
            promote = c
            break
    return {
        "top_10_by_score": top,
        "promotion_candidate": promote,
        "decision": "PROMOTE_REVIEW" if promote else "NO_LIVE_CHANGE",
        "reason": "Requires N>=30, WR>=70%, avg>70bps, T3R>0. If no candidate passes, keep current live unchanged.",
    }


def render(results: dict[str, Any]) -> str:
    lines = ["# S34 Long Relax + Management Suite", "", f"Generated: `{results['generated_at_utc']}`", ""]
    lines.append("## Baselines")
    for k, v in results["baselines"].items():
        lines.append(f"- `{k}`: {fmt(v['all'] if 'all' in v else v)}")
    lines.append("")
    for section in [
        "long_gate",
        "tail_detector_kept",
        "dynamic_hold_long",
        "dynamic_hold_short",
        "confidence",
        "exit_by_state",
        "route_fusion",
        "adaptive_stop_long",
        "adaptive_stop_short",
        "multi_stage",
        "position_sizing",
        "feature_bins",
    ]:
        lines.append(f"## {section}")
        obj = results.get(section, {})
        for name, val in obj.items():
            if isinstance(val, dict) and "all" in val:
                lines.append(f"- `{name}`: {fmt(val['all'])} | hold {fmt(val.get('hold', {}))}")
            elif isinstance(val, dict) and "n" in val:
                lines.append(f"- `{name}`: {fmt(val)}")
            elif isinstance(val, dict) and "kept" in val:
                lines.append(f"- `{name}` kept: {fmt(val['kept'])} | dropped: {fmt(val['dropped'])}")
            elif isinstance(val, dict):
                lines.append(f"- `{name}`:")
                for sub, s in val.items():
                    if isinstance(s, dict) and "n" in s:
                        lines.append(f"  - `{sub}`: {fmt(s)}")
                    elif isinstance(s, dict) and "all" in s:
                        lines.append(f"  - `{sub}`: {fmt(s['all'])}")
                    else:
                        lines.append(f"  - `{sub}`: `{s}`")
            else:
                lines.append(f"- `{name}`: `{val}`")
        lines.append("")
    lines.append("## Decision")
    dec = results["decision"]
    lines.append(f"- Decision: `{dec['decision']}`")
    lines.append(f"- Reason: {dec['reason']}")
    if dec.get("promotion_candidate"):
        lines.append(f"- Promotion candidate: `{dec['promotion_candidate']['name']}` -> {fmt(dec['promotion_candidate'])}")
    else:
        lines.append("- Promotion candidate: none.")
    return "\n".join(lines)


def main() -> int:
    global GLOBAL_ETH_MARKS
    rows, _all_rows = build_dataset()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_series(conn, "ETHUSDT")
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
        eth_sell = load_liq_series(conn, "ETHUSDT", "SELL")
    GLOBAL_ETH_MARKS = eth_marks
    rows = with_extra_features(rows, eth_sell, eth_marks)

    long_current = apply_confidence(make_long(rows, lambda r: long_gate(r, regime="or"), eth_marks=eth_marks, tag="LONG_RELAX_OR"))
    long_strict = apply_confidence(make_long(rows, lambda r: long_gate(r, regime="btc7d"), eth_marks=eth_marks, tag="LONG_STRICT_BTC7D"))
    short_current = short_candidates(rows, btc_sell, eth_marks, btc_thr=BTC_CONFIRM_CURRENT, delay_min=5, hold_ms=SHORT_HOLD_MS, score_min=4, tag="SHORT_BTC2M_D5_H2")
    short_1m_h4 = short_candidates(rows, btc_sell, eth_marks, btc_thr=BTC_CONFIRM_1M, delay_min=5, hold_ms=4 * 3600_000, score_min=4, tag="SHORT_BTC1M_D5_H4")

    long_gate_tests = {
        "current_relax_or": split_summary(long_current),
        "strict_btc7d_only": split_summary(long_strict),
        "btc4h_only": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="btc4h"), eth_marks=eth_marks, tag="LONG_BTC4H"))),
        "btc4h_and_btc7d": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="and"), eth_marks=eth_marks, tag="LONG_AND"))),
        "no_btc_regime": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="none"), eth_marks=eth_marks, tag="LONG_NOBTC"))),
        "sync300_or": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="or", sync_thr=300_000), eth_marks=eth_marks, tag="LONG_SYNC300"))),
        "sync500_or": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="or", sync_thr=500_000), eth_marks=eth_marks, tag="LONG_SYNC500"))),
        "score4_or": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="or", score_min=4), eth_marks=eth_marks, tag="LONG_SCORE4"))),
        "allow_mon_wed_or": split_summary(apply_confidence(make_long(rows, lambda r: long_gate(r, regime="or", allow_mon_wed=True), eth_marks=eth_marks, tag="LONG_ALLOW_MW"))),
    }

    tail = tail_detector_tests(long_current)
    dyn = dynamic_hold_tests(long_current, short_current, eth_marks)
    conf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in long_current:
        c = int(r.get("confidence") or 0)
        key = "conf_0_2" if c <= 2 else "conf_3_4" if c <= 4 else "conf_5_plus"
        conf_groups[key].append(r)
    confidence = {k: split_summary(v) for k, v in sorted(conf_groups.items())}
    confidence["conf_ge4"] = split_summary([r for r in long_current if int(r.get("confidence") or 0) >= 4])
    confidence["conf_ge5"] = split_summary([r for r in long_current if int(r.get("confidence") or 0) >= 5])

    exit_state = exit_by_state_tests(long_current, eth_marks, btc_sell)
    route_fusion = route_fusion_tests(long_current, short_current, short_1m_h4)
    stops = adaptive_stop_tests(long_current, short_current, eth_marks)
    staged = multi_stage_entry_tests(long_current, eth_marks)

    sizing = {
        "flat_long_current": summarize(long_current),
        "confidence_0p5_to_1p5": scale_stats(long_current, lambda r: max(0.5, min(1.5, 0.5 + 0.2 * int(r.get("confidence") or 0)))),
        "confidence_ge5_only": summarize([r for r in long_current if int(r.get("confidence") or 0) >= 5]),
        "half_size_tail_risk": scale_stats(long_current, lambda r: 0.5 if (float(r.get("eth1h_bps") or 0) < -80 or float(r.get("vol30_bps") or 0) > 60) else 1.0),
        "compound_35_current_env_1190": compound_equity(long_current, 35.0, 1190.0),
        "compound_35_balanced_16p3": compound_equity(long_current, 35.0, 16.3),
    }

    features = feature_bins(long_current, {
        "state": lambda r: str(r["close_reason"]),
        "session": lambda r: str(r["session"]),
        "dow": lambda r: str(r["dow_name"]),
        "sync_bucket": lambda r: "sync_0_100" if float(r["sync_k"]) < 100_000 else "sync_100_200",
        "n2h_bucket": lambda r: "n2h_0_2" if int(r["n2h"]) <= 2 else "n2h_3_4" if int(r["n2h"]) <= 4 else "n2h_5p",
        "vol30_bucket": lambda r: "vol_le35" if float(r["vol30_bps"]) <= 35 else "vol_35_60" if float(r["vol30_bps"]) <= 60 else "vol_gt60",
        "echo": lambda r: "echo" if (r.get("echo_30_90") or r.get("echo_45_120")) else "no_echo",
        "prebuild": lambda r: "prebuild" if int(r.get("prebuildup_30m") or 0) >= 1 else "no_prebuild",
    })

    results: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "dataset": {"rows_200k": len(rows)},
        "baselines": {
            "long_current_relax_or": split_summary(long_current),
            "long_strict_btc7d": split_summary(long_strict),
            "short_current_btc2m_d5_h2": split_summary(short_current),
            "short_btc1m_d5_h4": split_summary(short_1m_h4),
        },
        "long_gate": long_gate_tests,
        "tail_detector_kept": tail,
        "dynamic_hold_long": dyn["long"],
        "dynamic_hold_short": dyn["short"],
        "confidence": confidence,
        "exit_by_state": exit_state,
        "route_fusion": route_fusion,
        "adaptive_stop_long": stops["long"],
        "adaptive_stop_short": stops["short"],
        "multi_stage": staged,
        "position_sizing": sizing,
        "feature_bins": features,
    }
    results["decision"] = best_candidate_decision(results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
