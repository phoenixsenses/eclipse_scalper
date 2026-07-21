"""S34 state-machine v2 gauntlet.

Research-only. Does not touch live executor, env, runtime state, or orders.

This script consolidates the state-machine checks needed before any live work:
- live-like mark-price entry outcomes
- chronological holdout and walk-forward
- BTC/DOW/score/sync sensitivity
- corrected feature-label permutation with max-stat multiple-comparison control
- provisional entry vs confirmed entry
- conflict policies
- state transition map
- feature availability and shadow/backfill parity notes
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V2_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V2_GAUNTLET.md"
SHADOW_STATE = ROOT / "reports" / "shadow" / "s34_realtime_shadow_state.json"

LIVE_THRESH = 200_000.0
PROP_THRESH = 50_000.0
SIL_LO_MS = 60_000
SIL_HI_MS = 30 * 60_000
SYNC_WIN_MS = 10 * 60_000
FEE_BPS = 5.0
CAL_FRAC = 0.70
PERM_N = 1000

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def load_nav_events() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with NAV_EVENTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def load_liq(conn: sqlite3.Connection, symbol: str, side: str) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]


def load_marks(conn: sqlite3.Connection, symbol: str, lo: int, hi: int) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, int(lo), int(hi)),
    ).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]


def win_cnt(ts: list[int], vals: list[float], lo: int, hi: int, thr: float) -> int:
    a = bisect.bisect_left(ts, int(lo))
    b = bisect.bisect_right(ts, int(hi))
    return sum(1 for i in range(a, b) if vals[i] >= float(thr))


def win_sum(ts: list[int], vals: list[float], lo: int, hi: int) -> float:
    a = bisect.bisect_left(ts, int(lo))
    b = bisect.bisect_right(ts, int(hi))
    return float(sum(vals[i] for i in range(a, b)))


def first_above(ts: list[int], vals: list[float], lo: int, hi: int, thr: float) -> int | None:
    a = bisect.bisect_left(ts, int(lo))
    b = bisect.bisect_right(ts, int(hi))
    for i in range(a, b):
        if vals[i] >= float(thr):
            return int(ts[i])
    return None


def mark_at_or_after(ts: list[int], px: list[float], t: int) -> float | None:
    i = bisect.bisect_left(ts, int(t))
    if 0 <= i < len(px):
        return float(px[i])
    return None


def signed_net(direction: str, entry: float | None, exit_: float | None, fee_bps: float = FEE_BPS) -> float | None:
    if entry is None or exit_ is None or entry <= 0:
        return None
    raw = (float(exit_) - float(entry)) / float(entry) * 10_000.0
    if direction.upper() == "SHORT":
        raw = -raw
    return raw - float(fee_bps)


def finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "wr": None, "sum": 0.0, "mean": None, "median": None, "t3r": 0.0, "max_loss": None, "max_win": None}
    sv = sorted(vals)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "sum": round(sum(vals), 1),
        "mean": round(mean(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(sum(sv[:-3]) if len(sv) > 3 else sum(sv), 1),
        "max_loss": round(min(vals), 1),
        "max_win": round(max(vals), 1),
    }


def max_drawdown(vals: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        equity += float(v)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 1)


def summary_with_dd(signals: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(s["net_bps"]) for s in sorted(signals, key=lambda r: int(r["entry_ts_ms"])) if finite(s.get("net_bps")) is not None]
    out = summarize(vals)
    out["max_dd_bps"] = max_drawdown(vals)
    return out


@dataclass(frozen=True)
class Config:
    name: str
    btc_thr: float = 500_000.0
    long_score_min: int = 0
    short_score_min: int = 3
    exclude_long_dow: tuple[int, ...] = ()
    exclude_short_dow: tuple[int, ...] = ()
    exclude_europe_long: bool = True
    sync_thr: float = 200_000.0
    n2h_thr: int = 3
    include_noisy_short: bool = False


def recompute_score(row: dict[str, Any], *, sync_thr: float, n2h_thr: int) -> int:
    return sum(
        [
            int(row["sil_eth"]),
            int(row["n2h"] >= n2h_thr),
            int(row["b4h"] < 0),
            int(row["vd"] >= 30),
            int(row["sess_us"]),
            int(row["sync_k"] >= sync_thr),
        ]
    )


def classify_rows(
    nav: list[dict[str, Any]],
    *,
    eth_ts: list[int],
    eth_not: list[float],
    btc_ts: list[int],
    btc_not: list[float],
    sol_ts: list[int],
    sol_not: list[float],
    mk_ts: list[int],
    mk_px: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in nav:
        ts = int(raw["signal_ts_ms"])
        thr = float(raw.get("threshold_usd") or 0)
        net2 = finite(raw.get("net_2h_bps"))
        if thr < LIVE_THRESH or net2 is None:
            continue
        net4 = finite(raw.get("net_4h_bps"))
        if net4 is None:
            net4 = net2
        tags = raw.get("tags") or []
        bull = "BULL_PULLBACK" in tags
        n_prop = win_cnt(eth_ts, eth_not, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
        sil_eth = n_prop == 0
        b4h = float(raw.get("btc4h_bps") or 0.0)
        vd = float(raw.get("vdepth_bps") or 0.0)
        bid = float(raw.get("bid_depth_usd") or 0.0)
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        hour = dt.hour
        dow = dt.weekday()
        session = "ASIA" if hour < 7 else "EUROPE" if hour < 13 else "US" if hour < 21 else "OFF"
        sess_us = 13 <= hour < 21
        sync_k = win_sum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts) + win_sum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts)
        n2h = win_cnt(eth_ts, eth_not, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
        first_btc_by_thr = {
            str(int(thr_)): first_above(btc_ts, btc_not, ts + SIL_LO_MS, ts + SIL_HI_MS, thr_)
            for thr_ in (200_000.0, 300_000.0, 500_000.0, 750_000.0, 1_000_000.0, 1_500_000.0)
        }
        entry_t0 = mark_at_or_after(mk_ts, mk_px, ts)
        px_15 = mark_at_or_after(mk_ts, mk_px, ts + 15 * 60_000)
        px_30 = mark_at_or_after(mk_ts, mk_px, ts + 30 * 60_000)
        px_2h = mark_at_or_after(mk_ts, mk_px, ts + 2 * 3600_000)
        px_4h = mark_at_or_after(mk_ts, mk_px, ts + 4 * 3600_000)
        score_default = sum([int(sil_eth), int(n2h >= 3), int(b4h < 0), int(vd >= 30), int(sess_us), int(sync_k >= 200_000)])
        out.append(
            {
                "ts": ts,
                "utc": iso_ms(ts),
                "thr": thr,
                "net2_nav": net2,
                "net4_nav": net4,
                "bull": bull,
                "tags": tags,
                "sil_eth": sil_eth,
                "n_prop": n_prop,
                "first_btc_by_thr": first_btc_by_thr,
                "b4h": b4h,
                "vd": vd,
                "bid": bid,
                "hour": hour,
                "dow": dow,
                "session": session,
                "sess_us": sess_us,
                "sync_k": sync_k,
                "n2h": n2h,
                "score_default": score_default,
                "long_t0_4h": signed_net("LONG", entry_t0, px_4h),
                "long_t15_to_4h": signed_net("LONG", px_15, px_4h),
                "long_t30_to_4h": signed_net("LONG", px_30, px_4h),
                "short_anchor_2h": signed_net("SHORT", entry_t0, px_2h),
                "eth_shift_15_bps": signed_net("LONG", entry_t0, px_15, 0.0),
                "eth_shift_30_bps": signed_net("LONG", entry_t0, px_30, 0.0),
            }
        )
    out.sort(key=lambda r: int(r["ts"]))
    n_cal = int(len(out) * CAL_FRAC)
    cutoff = int(out[n_cal]["ts"]) if out else 0
    for i, row in enumerate(out):
        row["idx"] = i
        row["split"] = "hold" if i >= n_cal else "cal"
        row["is_hold"] = i >= n_cal
        row["holdout_cutoff_utc"] = iso_ms(cutoff)
    return out


def short_btc_outcome(row: dict[str, Any], btc_thr: float, mk_ts: list[int], mk_px: list[float]) -> tuple[int | None, float | None]:
    entry_ts = row["first_btc_by_thr"].get(str(int(btc_thr)))
    if entry_ts is None:
        return None, None
    entry = mark_at_or_after(mk_ts, mk_px, int(entry_ts))
    exit_ = mark_at_or_after(mk_ts, mk_px, int(entry_ts) + 2 * 3600_000)
    return int(entry_ts), signed_net("SHORT", entry, exit_)


def state_for(row: dict[str, Any], btc_thr: float = 500_000.0) -> str:
    if row["bull"]:
        suffix = "_BULL"
    else:
        suffix = ""
    if row["sil_eth"]:
        return "SILENCE" + suffix
    if row["first_btc_by_thr"].get(str(int(btc_thr))) is not None:
        return "NEITHER" + suffix
    return "NOISY" + suffix


def build_signals(
    rows: list[dict[str, Any]],
    cfg: Config,
    *,
    mk_ts: list[int],
    mk_px: list[float],
    use_permuted_features: bool = False,
) -> list[dict[str, Any]]:
    sigs: list[dict[str, Any]] = []
    for row in rows:
        feat = row.get("_perm_feat", row) if use_permuted_features else row
        if feat.get("bull"):
            continue
        score = recompute_score(feat, sync_thr=cfg.sync_thr, n2h_thr=cfg.n2h_thr)
        if feat["sil_eth"]:
            if cfg.exclude_europe_long and feat["session"] == "EUROPE":
                continue
            if int(feat["dow"]) in cfg.exclude_long_dow:
                continue
            if score < cfg.long_score_min:
                continue
            net = row.get("long_t0_4h")
            if net is not None:
                sigs.append(
                    {
                        "entry_ts_ms": int(row["ts"]),
                        "anchor_ts_ms": int(row["ts"]),
                        "side": "LONG",
                        "arm": "SILENCE_LONG",
                        "net_bps": float(net),
                        "row": row,
                        "score": score,
                    }
                )
        else:
            btc_entry, short_net = short_btc_outcome(row, cfg.btc_thr, mk_ts, mk_px)
            if btc_entry is not None:
                if int(feat["dow"]) in cfg.exclude_short_dow:
                    continue
                if score < cfg.short_score_min:
                    continue
                if short_net is not None:
                    sigs.append(
                        {
                            "entry_ts_ms": int(btc_entry),
                            "anchor_ts_ms": int(row["ts"]),
                            "side": "SHORT",
                            "arm": "NEITHER_SHORT",
                            "net_bps": float(short_net),
                            "row": row,
                            "score": score,
                        }
                    )
            elif cfg.include_noisy_short:
                if int(feat["dow"]) in cfg.exclude_short_dow:
                    continue
                if score < cfg.short_score_min:
                    continue
                net = row.get("short_anchor_2h")
                if net is not None:
                    sigs.append(
                        {
                            "entry_ts_ms": int(row["ts"]),
                            "anchor_ts_ms": int(row["ts"]),
                            "side": "SHORT",
                            "arm": "NOISY_SHORT",
                            "net_bps": float(net),
                            "row": row,
                            "score": score,
                        }
                    )
    sigs.sort(key=lambda s: int(s["entry_ts_ms"]))
    return sigs


def apply_conflict_policy(signals: list[dict[str, Any]], policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if policy == "all_independent":
        return list(signals), []
    taken: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    active_end: int | None = None
    active_side: str | None = None
    for sig in sorted(signals, key=lambda s: int(s["entry_ts_ms"])):
        side = str(sig["side"])
        hold_ms = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
        entry = int(sig["entry_ts_ms"])
        if active_end is None or entry >= active_end:
            taken.append(sig)
            active_side = side
            active_end = entry + hold_ms
            continue
        if policy == "one_pos_ignore":
            if side == "SHORT" and active_side == "LONG":
                taken.append({**sig, "conflict_action": "flip_long_to_short"})
                active_side = "SHORT"
                active_end = entry + hold_ms
            else:
                blocked.append({**sig, "blocked_reason": f"{side}_on_{active_side}"})
        elif policy == "short_replace":
            if side == "SHORT" and active_side == "LONG":
                taken.append({**sig, "conflict_action": "flip_long_to_short"})
                active_side = "SHORT"
                active_end = entry + hold_ms
            elif side == "SHORT" and active_side == "SHORT":
                taken.append({**sig, "conflict_action": "replace_short"})
                active_side = "SHORT"
                active_end = entry + hold_ms
            else:
                blocked.append({**sig, "blocked_reason": f"{side}_on_{active_side}"})
        else:
            raise ValueError(policy)
    return taken, blocked


def split_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    cal = [s for s in signals if not s["row"]["is_hold"]]
    hold = [s for s in signals if s["row"]["is_hold"]]
    return {
        "all": summary_with_dd(signals),
        "cal": summary_with_dd(cal),
        "hold": summary_with_dd(hold),
        "by_arm": {arm: summary_with_dd([s for s in signals if s["arm"] == arm]) for arm in sorted({s["arm"] for s in signals})},
    }


def fold_summaries(signals: list[dict[str, Any]], folds: int = 5) -> dict[str, Any]:
    if not signals:
        return {"folds": [], "positive_folds": 0, "t3r_sum": 0.0}
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    out = []
    n = len(ordered)
    for i in range(folds):
        lo = int(i * n / folds)
        hi = int((i + 1) * n / folds)
        chunk = ordered[lo:hi]
        sm = summary_with_dd(chunk)
        sm["start_utc"] = iso_ms(chunk[0]["entry_ts_ms"]) if chunk else None
        sm["end_utc"] = iso_ms(chunk[-1]["entry_ts_ms"]) if chunk else None
        out.append(sm)
    return {
        "folds": out,
        "positive_folds": sum(1 for f in out if float(f.get("sum") or 0) > 0),
        "t3r_sum": round(sum(float(f.get("t3r") or 0.0) for f in out), 1),
    }


def run_permutation(
    hold_rows: list[dict[str, Any]],
    configs: list[Config],
    *,
    mk_ts: list[int],
    mk_px: list[float],
    n_perm: int = PERM_N,
    seed: int = 3402,
) -> dict[str, Any]:
    rng = random.Random(seed)
    feature_keys = ["bull", "sil_eth", "session", "dow", "n2h", "b4h", "vd", "sess_us", "sync_k"]
    base_features = [{k: r[k] for k in feature_keys} for r in hold_rows]
    real: dict[str, float] = {}
    for cfg in configs:
        real_sigs = build_signals(hold_rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        real[cfg.name] = float(summary_with_dd(real_sigs).get("t3r") or 0.0)

    per_cfg_null: dict[str, list[float]] = {cfg.name: [] for cfg in configs}
    max_null: list[float] = []
    rows_mut = [dict(r) for r in hold_rows]
    for _ in range(n_perm):
        shuffled = list(base_features)
        rng.shuffle(shuffled)
        for row, feat in zip(rows_mut, shuffled, strict=False):
            row["_perm_feat"] = feat
        vals = []
        for cfg in configs:
            sigs = build_signals(rows_mut, cfg, mk_ts=mk_ts, mk_px=mk_px, use_permuted_features=True)
            t3r = float(summary_with_dd(sigs).get("t3r") or 0.0)
            per_cfg_null[cfg.name].append(t3r)
            vals.append(t3r)
        max_null.append(max(vals) if vals else 0.0)

    results = {}
    for cfg in configs:
        r = real[cfg.name]
        null = per_cfg_null[cfg.name]
        raw_p = (sum(1 for v in null if v >= r) + 1) / (len(null) + 1)
        mc_p = (sum(1 for v in max_null if v >= r) + 1) / (len(max_null) + 1)
        results[cfg.name] = {
            "real_hold_t3r": round(r, 1),
            "raw_perm_p": round(raw_p, 4),
            "mc_perm_p": round(mc_p, 4),
            "null_p95_t3r": round(sorted(null)[int(0.95 * (len(null) - 1))], 1),
            "max_null_p95_t3r": round(sorted(max_null)[int(0.95 * (len(max_null) - 1))], 1),
            "verdict": "PASS_MC_5PCT" if mc_p <= 0.05 and r > 0 else "FAIL",
        }
    return {
        "n_perm": n_perm,
        "seed": seed,
        "method": "shuffle feature/state labels across holdout events; preserve timestamp/outcome path; max-stat across tested configs",
        "configs": results,
    }


def sensitivity(rows: list[dict[str, Any]], *, mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for btc_thr in [300_000, 500_000, 750_000, 1_000_000, 1_500_000]:
        cfg = Config(name=f"btc{int(btc_thr/1000)}k", btc_thr=float(btc_thr), long_score_min=3, short_score_min=3)
        sigs = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        out[cfg.name] = split_summary(sigs)
    for score in [0, 2, 3, 4, 5]:
        cfg = Config(name=f"score_ge{score}", btc_thr=750_000, long_score_min=score, short_score_min=score)
        sigs = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        out[cfg.name] = split_summary(sigs)
    for sync in [0, 100_000, 200_000, 500_000]:
        cfg = Config(name=f"sync_score_{int(sync/1000)}k", btc_thr=750_000, long_score_min=3, short_score_min=3, sync_thr=float(sync))
        sigs = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        out[cfg.name] = split_summary(sigs)
    for name, ldow, sdow in [
        ("no_dow", (), ()),
        ("excl_monwed_long", (0, 2), ()),
        ("excl_sun_short", (), (6,)),
        ("excl_monwed_long_sun_short", (0, 2), (6,)),
    ]:
        cfg = Config(name=name, btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=ldow, exclude_short_dow=sdow)
        sigs = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        out[name] = split_summary(sigs)
    return out


def entry_timing(rows: list[dict[str, Any]], *, mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    sil = [r for r in rows if r["sil_eth"] and not r["bull"] and r["session"] != "EUROPE"]
    long_t0 = [r["long_t0_4h"] for r in sil if r.get("long_t0_4h") is not None]
    long_t15 = [r["long_t15_to_4h"] for r in sil if r.get("long_t15_to_4h") is not None]
    long_t30 = [r["long_t30_to_4h"] for r in sil if r.get("long_t30_to_4h") is not None]
    shift15 = [r["eth_shift_15_bps"] for r in sil if r.get("eth_shift_15_bps") is not None]
    shift30 = [r["eth_shift_30_bps"] for r in sil if r.get("eth_shift_30_bps") is not None]
    noisy = [r for r in rows if (not r["sil_eth"]) and not r["bull"]]
    short_anchor = [r["short_anchor_2h"] for r in noisy if r.get("short_anchor_2h") is not None]
    out = {
        "silence_long_t0": summarize(long_t0),
        "silence_long_t15": summarize(long_t15),
        "silence_long_t30": summarize(long_t30),
        "silence_eth_shift_15_bps": summarize(shift15),
        "silence_eth_shift_30_bps": summarize(shift30),
        "noisy_short_anchor_provisional": summarize(short_anchor),
    }
    for thr in [500_000, 750_000, 1_000_000]:
        vals = []
        shifts = []
        for r in noisy:
            entry_ts, net = short_btc_outcome(r, thr, mk_ts, mk_px)
            if net is not None:
                vals.append(net)
                px0 = mark_at_or_after(mk_ts, mk_px, int(r["ts"]))
                px1 = mark_at_or_after(mk_ts, mk_px, int(entry_ts)) if entry_ts else None
                sh = signed_net("LONG", px0, px1, 0.0)
                if sh is not None:
                    shifts.append(sh)
        out[f"neither_short_btc{int(thr/1000)}k_confirmed"] = summarize(vals)
        out[f"neither_eth_shift_to_btc{int(thr/1000)}k_bps"] = summarize(shifts)
    return out


def conflict_tests(signals: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for policy in ["all_independent", "one_pos_ignore", "short_replace"]:
        taken, blocked = apply_conflict_policy(signals, policy)
        out[policy] = {
            "taken": summary_with_dd(taken),
            "blocked": summary_with_dd(blocked),
            "blocked_by_reason": {},
        }
        reasons = sorted({b.get("blocked_reason", "none") for b in blocked})
        for reason in reasons:
            out[policy]["blocked_by_reason"][reason] = summary_with_dd([b for b in blocked if b.get("blocked_reason", "none") == reason])
    return out


def transition_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted([r for r in rows if not r["bull"]], key=lambda r: int(r["ts"]))
    out: dict[str, Any] = {}
    for max_gap_h in [2, 4, 8, 24]:
        pairs: dict[str, list[float]] = {}
        for a, b in zip(ordered, ordered[1:], strict=False):
            gap_ms = int(b["ts"]) - int(a["ts"])
            if gap_ms < 0 or gap_ms > max_gap_h * 3600_000:
                continue
            key = f"{state_for(a)}->{state_for(b)}"
            val = b["long_t0_4h"] if b["sil_eth"] else b["short_anchor_2h"]
            if val is not None:
                pairs.setdefault(key, []).append(float(val))
        out[f"next_within_{max_gap_h}h"] = {k: summarize(v) for k, v in sorted(pairs.items())}
    return out


def feature_availability() -> dict[str, Any]:
    return {
        "dow": {"class": "POINT_IN_TIME", "knowable": True, "source": "timestamp"},
        "session": {"class": "POINT_IN_TIME", "knowable": True, "source": "timestamp UTC"},
        "n2h": {"class": "RUNNING_CLUSTER/HISTORY", "knowable": True, "source": "liquidations before T"},
        "sync_k": {"class": "RUNNING_CLUSTER/HISTORY", "knowable": True, "source": "BTC+SOL liquidations before T"},
        "btc4h_bps": {"class": "POINT_IN_TIME_HISTORY", "knowable": True, "source": "mark_prices before T"},
        "vdepth_bps": {"class": "POINT_IN_TIME_BOOK", "knowable": "conditional", "source": "book ticker at T; stale book must reject"},
        "sil_eth": {"class": "FORWARD_STATE_RESOLUTION", "knowable": False, "source": "requires 30m future; live must use provisional entry + resolve later"},
        "btc_cascade_in_30m": {"class": "FORWARD_STATE_RESOLUTION", "knowable": False, "source": "only knowable when BTC threshold actually crosses"},
        "net_2h/net_4h": {"class": "FORWARD/OUTCOME", "knowable": False, "source": "label only"},
        "live_blocker": "SILENCE cannot be an entry filter at T=0; it is a management/resolution state.",
    }


def shadow_parity(rows: list[dict[str, Any]], *, mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    state = {}
    if SHADOW_STATE.exists():
        try:
            state = json.loads(SHADOW_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    shadow_pnl = state.get("pnl", {}) if isinstance(state, dict) else {}
    default_cfg = Config(name="shadow_equivalent_500k", btc_thr=500_000, long_score_min=0, short_score_min=0, include_noisy_short=True)
    sigs = build_signals(rows, default_cfg, mk_ts=mk_ts, mk_px=mk_px)
    return {
        "shadow_state_path": str(SHADOW_STATE),
        "shadow_state_exists": SHADOW_STATE.exists(),
        "shadow_pnl": shadow_pnl,
        "recomputed_mark_equivalent": split_summary(sigs),
        "note": "Backfill ledger uses NAV net labels for historical rows; gauntlet recomputes mark-price outcomes, so exact bps parity is not expected. For live promotion, a separate timestamp/decision parity test is required.",
    }


def render_table_stats(title: str, rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [f"## {title}", "", "| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in rows:
        wr = "" if s.get("wr") is None else f"{100*float(s['wr']):.1f}%"
        lines.append(
            f"| {name} | {s.get('n',0)} | {wr} | {s.get('sum')} | {s.get('mean')} | "
            f"{s.get('median')} | {s.get('t3r')} | {s.get('max_loss')} | {s.get('max_dd_bps','')} |"
        )
    lines.append("")
    return lines


def render_md(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# S34 State Machine V2 Gauntlet",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- events: `{report['counts']['events']}` | cal: `{report['counts']['cal']}` | hold: `{report['counts']['hold']}`",
        f"- holdout_cutoff_utc: `{report['counts']['holdout_cutoff_utc']}`",
        "- research_only: `true`",
        "",
        "## Verdict",
        "",
        f"- primary_config: `{report['primary_config']}`",
        f"- primary_hold: `{report['primary']['hold']}`",
        f"- primary_walk_forward: positive_folds `{report['primary_walk_forward']['positive_folds']}/5`, fold_t3r_sum `{report['primary_walk_forward']['t3r_sum']}`",
        f"- corrected_permutation: `{report['permutation']['configs'][report['primary_config']]['verdict']}` "
        f"mc_p=`{report['permutation']['configs'][report['primary_config']]['mc_perm_p']}`",
        f"- live_blocker: `{report['feature_availability']['live_blocker']}`",
        "",
    ]
    lines += render_table_stats("Config Holdout Summary", [(k, v["hold"]) for k, v in report["config_results"].items()])
    lines += [
        "## Corrected Permutation",
        "",
        "| Config | Real hold T3R | Raw p | MC p | Null p95 | Max-null p95 | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, p in report["permutation"]["configs"].items():
        lines.append(
            f"| {name} | {p['real_hold_t3r']} | {p['raw_perm_p']} | {p['mc_perm_p']} | "
            f"{p['null_p95_t3r']} | {p['max_null_p95_t3r']} | {p['verdict']} |"
        )
    lines.append("")
    lines += render_table_stats(
        "Entry Timing",
        [(k, v) for k, v in report["entry_timing"].items() if isinstance(v, dict) and "n" in v],
    )
    lines += render_table_stats(
        "Conflict Policies",
        [(k, v["taken"]) for k, v in report["conflict_policies"].items()],
    )
    lines += [
        "## State Transitions",
        "",
    ]
    for horizon, data in report["transition_graph"].items():
        best = sorted(data.items(), key=lambda kv: float(kv[1].get("t3r") or -1e18), reverse=True)[:8]
        lines += render_table_stats(horizon, best)
    lines += [
        "## Feature Availability",
        "",
        "| Feature | Class | Knowable at entry? | Source / note |",
        "| --- | --- | --- | --- |",
    ]
    for k, v in report["feature_availability"].items():
        if not isinstance(v, dict):
            continue
        lines.append(f"| {k} | {v.get('class')} | {v.get('knowable')} | {v.get('source')} |")
    lines += [
        "",
        "## Shadow Parity",
        "",
        f"- shadow_state_exists: `{report['shadow_parity']['shadow_state_exists']}`",
        f"- note: {report['shadow_parity']['note']}",
        "",
        "## Next Required Work",
        "",
        "1. Do not live-promote from this report alone.",
        "2. Build a timestamp-level realtime/backfill parity test before executor work.",
        "3. If proceeding, model SILENCE as provisional-entry management, not as an entry-known filter.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    nav = load_nav_events()
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")
        raw_valid_ts = [int(r["signal_ts_ms"]) for r in nav if float(r.get("threshold_usd") or 0) >= LIVE_THRESH]
        if not raw_valid_ts:
            raise RuntimeError("No NAV events found")
        mk_ts, mk_px = load_marks(conn, "ETHUSDT", min(raw_valid_ts) - 60_000, max(raw_valid_ts) + 5 * 3600_000)

    rows = classify_rows(nav, eth_ts=eth_ts, eth_not=eth_not, btc_ts=btc_ts, btc_not=btc_not, sol_ts=sol_ts, sol_not=sol_not, mk_ts=mk_ts, mk_px=mk_px)
    hold = [r for r in rows if r["is_hold"]]
    configs = [
        Config("baseline_500k_score3", btc_thr=500_000, long_score_min=3, short_score_min=3),
        Config("btc750_score3", btc_thr=750_000, long_score_min=3, short_score_min=3),
        Config("btc750_dow_score3", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc750_dow_score4", btc_thr=750_000, long_score_min=4, short_score_min=4, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc1000_dow_score3", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc750_dow_score3_noisy", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,), include_noisy_short=True),
    ]
    config_results: dict[str, Any] = {}
    all_signals: dict[str, list[dict[str, Any]]] = {}
    for cfg in configs:
        sigs = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        taken, blocked = apply_conflict_policy(sigs, "short_replace")
        all_signals[cfg.name] = taken
        config_results[cfg.name] = split_summary(taken)
        config_results[cfg.name]["blocked_short_replace"] = summary_with_dd(blocked)
        config_results[cfg.name]["walk_forward"] = fold_summaries(taken, 5)

    primary_name = "btc750_dow_score3"
    primary_signals = all_signals[primary_name]
    sens = sensitivity(rows, mk_ts=mk_ts, mk_px=mk_px)
    entry = entry_timing(rows, mk_ts=mk_ts, mk_px=mk_px)
    conflict_base = build_signals(rows, Config("conflict_base", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)), mk_ts=mk_ts, mk_px=mk_px)
    conflicts = conflict_tests(conflict_base)
    transitions = transition_graph(rows)
    perm = run_permutation(hold, configs, mk_ts=mk_ts, mk_px=mk_px)

    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "source": {
            "nav_events": str(NAV_EVENTS),
            "db": str(DEFAULT_DB),
        },
        "counts": {
            "nav_events": len(nav),
            "events": len(rows),
            "cal": sum(1 for r in rows if not r["is_hold"]),
            "hold": sum(1 for r in rows if r["is_hold"]),
            "holdout_cutoff_utc": rows[sum(1 for r in rows if not r["is_hold"])]["holdout_cutoff_utc"] if rows else None,
        },
        "primary_config": primary_name,
        "primary": config_results[primary_name],
        "primary_walk_forward": config_results[primary_name]["walk_forward"],
        "config_results": config_results,
        "sensitivity": sens,
        "entry_timing": entry,
        "conflict_policies": conflicts,
        "transition_graph": transitions,
        "permutation": perm,
        "feature_availability": feature_availability(),
        "shadow_parity": shadow_parity(rows, mk_ts=mk_ts, mk_px=mk_px),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "primary": primary_name,
        "hold": report["primary"]["hold"],
        "wf": report["primary_walk_forward"],
        "perm": report["permutation"]["configs"][primary_name],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
