#!/usr/bin/env python3
"""
audit_fill_calibration.py -- Self-contained fill probability calibration audit.

Investigates WHY passive execution model produces 9:1 sell/buy NPA gap
when path excursions are symmetric (confirmed in NPA_DECOMPOSITION_REGIME.md).

Top pocket: abs(imbalance)>=0.5, trade_intensity>=3500, spread<=0.0003, h=120s
  SELL signals: imbalance <= -0.5  (go SHORT)
  BUY  signals: imbalance >= +0.5  (go LONG)

Passive fill mechanics:
  SHORT limit = ep*(1+0.5*spread) -- fills when price goes UP to limit
  LONG  limit = ep*(1-0.5*spread) -- fills when price goes DOWN to limit

Output: reports/FILL_CALIBRATION_AUDIT.md + reports/plots/09..12_*.png

Usage:
  python -m tools.audit_fill_calibration [--db data/microstructure.db]
"""

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_DEFAULT = "data/microstructure.db"
SYMBOL = "ETHUSDT"
HORIZON = 120       # seconds hold
MIN_IMB = 0.5       # abs(imbalance) threshold
MIN_INT = 3500      # trade_intensity threshold
MAX_SPR = 0.0003    # spread_proxy threshold
MAKER_FEE_BPS = 2.0 # one-way maker fee bps
REPORT = Path("reports/FILL_CALIBRATION_AUDIT.md")
PLOT_DIR = Path("reports/plots")

# ── Inline calibration helpers (replicated from execution/passive_execution_simulator.py) ──

def _quantile(vals, q):
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    if q <= 0.0:
        return xs[0]
    if q >= 1.0:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _bin_idx(x, edges):
    if x is None:
        return None
    e = [v for v in edges if v is not None]
    if len(e) != len(edges):
        return None
    xv = float(x)
    if len(e) == 2:
        return 0 if xv <= e[0] else (1 if xv <= e[1] else 2)
    if len(e) == 3:
        return 0 if xv <= e[0] else (1 if xv <= e[1] else (2 if xv <= e[2] else 3))
    return None


def _fmean(vals, default=0.0):
    xs = [float(v) for v in vals]
    return sum(xs) / len(xs) if xs else float(default)


def calibrate_inline(samples, maker_fee_bps=MAKER_FEE_BPS):
    """Replicate calibrate_passive_model from execution/passive_execution_simulator.py."""
    if not samples:
        return {
            "n": 0,
            "base_touch": 0.0,
            "base_full_cond_touch": 0.0,
            "base_adverse_bps": 0.0,
            "edges": {},
            "touch_rates": {},
            "full_rates": {},
            "adverse_means": {},
        }
    feature_names = ["spread", "trade_intensity", "vol_proxy", "imbalance_for_fill"]
    touched = [1.0 if s.get("touched") else 0.0 for s in samples]
    touched_s = [s for s in samples if s.get("touched")]
    full_ct = [1.0 if s.get("full_proxy") else 0.0 for s in touched_s]
    adverse = [float(s["adverse_bps"]) for s in samples if s.get("adverse_bps") is not None]

    edges = {}
    touch_rates = {}
    full_rates = {}
    adverse_means = {}
    for fn in feature_names:
        vals = [float(s[fn]) for s in samples if s.get(fn) is not None]
        e = [_quantile(vals, 1.0 / 3.0), _quantile(vals, 2.0 / 3.0)]
        edges[fn] = e
        tb = [[], [], []]
        fb = [[], [], []]
        ab = [[], [], []]
        for s in samples:
            bi = _bin_idx(s.get(fn), e)
            if bi is None:
                continue
            tb[bi].append(1.0 if s.get("touched") else 0.0)
            ab[bi].append(float(s.get("adverse_bps") or 0.0))
            if s.get("touched"):
                fb[bi].append(1.0 if s.get("full_proxy") else 0.0)
        touch_rates[fn] = [_fmean(x, _fmean(touched, 0.0)) for x in tb]
        full_rates[fn] = [_fmean(x, _fmean(full_ct, 0.0)) for x in fb]
        adverse_means[fn] = [_fmean(x, _fmean(adverse, 0.0)) for x in ab]

    depth_vals = [float(s.get("depth", 0.0)) for s in touched_s if s.get("depth") is not None]
    depth_thr = _quantile(depth_vals, 0.50) or 0.5

    return {
        "n": len(samples),
        "n_touched": len(touched_s),
        "maker_fee_bps": float(maker_fee_bps),
        "base_touch": _fmean(touched, 0.0),
        "base_full_cond_touch": _fmean(full_ct, 0.0),
        "base_adverse_bps": _fmean(adverse, 0.0),
        "depth_full_threshold": float(depth_thr),
        "edges": edges,
        "touch_rates": touch_rates,
        "full_rates": full_rates,
        "adverse_means": adverse_means,
    }

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(db_path):
    """Load mark price dense array + 1-sec trade bucket features."""
    print(f"[load] connecting to {db_path}")
    conn = sqlite3.connect(db_path)
    # -- Mark prices: 1-second dense array
    t0 = time.time()
    mp = pd.read_sql_query(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        conn, params=(SYMBOL,),
    )
    print(f"[load] mark_prices: {len(mp):,} rows in {time.time()-t0:.1f}s")
    mp["ts_sec"] = (mp["ts_ms"] // 1000).astype(np.int64)
    mp_s = mp.groupby("ts_sec")["mark_price"].last()
    min_sec = int(mp_s.index.min())
    max_sec = int(mp_s.index.max())
    n_arr = max_sec - min_sec + 1
    mark_arr = np.full(n_arr, np.nan, dtype=np.float64)
    idxs = (mp_s.index.values - min_sec).astype(np.int64)
    mark_arr[idxs] = mp_s.values
    # forward-fill NaNs
    for i in range(1, n_arr):
        if np.isnan(mark_arr[i]):
            mark_arr[i] = mark_arr[i - 1]
    fill_pct = np.sum(~np.isnan(mark_arr)) / n_arr * 100
    print(f"[load] mark_arr: span={n_arr:,}s, fill={fill_pct:.1f}%")
    # -- 1-second trade buckets (aggregate without cross-join for speed)
    t0 = time.time()
    sql = """
        SELECT
            CAST(ts_ms / 1000 AS INTEGER)                       AS ts_sec,
            SUM(CASE WHEN is_buyer_maker=0 THEN  quantity ELSE 0 END)  AS buy_qty,
            SUM(CASE WHEN is_buyer_maker=1 THEN  quantity ELSE 0 END)  AS sell_qty,
            SUM(quantity)                                         AS tot_qty,
            SUM(price * quantity)                                 AS notional,
            COUNT(*)                                              AS n_trades
        FROM agg_trades
        WHERE symbol=?
        GROUP BY ts_sec
    """
    df = pd.read_sql_query(sql, conn, params=(SYMBOL,))
    conn.close()
    print(f"[load] trade buckets: {len(df):,} rows in {time.time()-t0:.1f}s")
    df["imbalance"] = (df["buy_qty"] - df["sell_qty"]) / (df["tot_qty"] + 1e-12)
    df["trade_intensity"] = df["n_trades"] * 60.0
    df["vwap"] = df["notional"] / (df["tot_qty"] + 1e-12)
    # match VWAP to mark price for spread proxy
    bucket_mark_idx = np.clip(df["ts_sec"].values - min_sec, 0, n_arr - 1)
    df["mark_px"] = mark_arr[bucket_mark_idx]
    df["spread_proxy"] = np.where(
        df["mark_px"] > 0,
        np.abs(df["vwap"] - df["mark_px"]) / df["mark_px"],
        np.nan,
    )
    df = df.dropna(subset=["spread_proxy", "mark_px"]).copy()
    print(f"[load] after dropna: {len(df):,} bucket rows")
    return df, mark_arr, min_sec

# ── Signal extraction ─────────────────────────────────────────────────────────

def extract_signals(df, mark_arr, min_sec):
    """Extract SELL and BUY signals with all features needed for calibration."""
    n_arr = len(mark_arr)
    # Top-pocket filter
    mask = (
        (df["spread_proxy"] <= MAX_SPR)
        & (df["trade_intensity"] >= MIN_INT)
        & (df["imbalance"].abs() >= MIN_IMB)
    )
    sig = df[mask].copy()
    sig["ts_sec"] = sig["ts_sec"].astype(np.int64)
    # Split by direction
    sell_mask = sig["imbalance"] <= -MIN_IMB
    buy_mask = sig["imbalance"] >= MIN_IMB
    sell_df = sig[sell_mask].copy()
    buy_df = sig[buy_mask].copy()
    print(f"[signals] SELL={len(sell_df):,}  BUY={len(buy_df):,}")

    def build_signal_array(sdf, side):
        """For each signal, compute touch stats and endpoint returns."""
        ts = sdf["ts_sec"].values
        imb = sdf["imbalance"].values
        spr = sdf["spread_proxy"].values
        itns = sdf["trade_intensity"].values
        vol_proxy = np.abs(sdf["imbalance"].values)  # simplified vol proxy
        imb_for_fill = np.abs(imb)
        n = len(ts)
        base_idx = (ts - min_sec).astype(np.int64)
        entry_idx = base_idx + 1
        exit_idx = base_idx + 1 + HORIZON
        # Clip to valid range
        ok = (entry_idx >= 0) & (exit_idx < n_arr - 1)
        if not np.any(ok):
            return []
        ts = ts[ok]; base_idx = base_idx[ok]; entry_idx = entry_idx[ok]
        exit_idx = exit_idx[ok]; spr = spr[ok]; itns = itns[ok]
        vol_proxy = vol_proxy[ok]; imb_for_fill = imb_for_fill[ok]
        n = len(ts)

        entry_px = mark_arr[entry_idx]
        exit_px = mark_arr[exit_idx]

        # Build path matrix: (N, HORIZON) -- prices at t+1..t+120 from signal
        offsets = np.arange(1, HORIZON + 1, dtype=np.int64)
        path_idx = base_idx[:, None] + offsets[None, :]  # (N, 120)
        path_idx_clipped = np.clip(path_idx, 0, n_arr - 1)
        paths = mark_arr[path_idx_clipped]  # (N, 120)
        # Zero out out-of-bounds (shouldn't happen due to ok mask but be safe)
        paths[path_idx >= n_arr] = np.nan

        # Passive limit computation
        if side == "SHORT":
            limits = entry_px * (1.0 + 0.5 * spr)
            touched_mat = paths >= limits[:, None]
        else:  # LONG
            limits = entry_px * (1.0 - 0.5 * spr)
            touched_mat = paths <= limits[:, None]

        touched_vec = np.any(touched_mat, axis=1)  # (N,) bool
        touch_idx_raw = np.where(touched_vec, np.argmax(touched_mat, axis=1), -1)  # (N,)

        # Depth at touch
        safe_ti = np.where(touch_idx_raw >= 0, touch_idx_raw, 0)
        row_idx = np.arange(n)
        p_touch = paths[row_idx, safe_ti]  # (N,)
        if side == "SHORT":
            depth_vec = np.where(
                touched_vec,
                (p_touch - limits) / (entry_px * spr + 1e-12),
                0.0,
            )
        else:
            depth_vec = np.where(
                touched_vec,
                (limits - p_touch) / (entry_px * spr + 1e-12),
                0.0,
            )
        depth_vec = np.maximum(0.0, depth_vec)
        full_proxy_vec = touched_vec & (depth_vec >= 0.5)

        # adverse_bps (1-step reversal after touch, as defined in build_passive_calibration_samples)
        next_idx = np.minimum(safe_ti + 1, HORIZON - 1)
        p_next = paths[row_idx, next_idx]
        if side == "SHORT":
            adv = np.where(
                touched_vec & (p_touch > 0),
                np.maximum(0.0, (p_touch - p_next) / p_touch * 1e4),
                0.0,
            )
        else:
            adv = np.where(
                touched_vec & (p_touch > 0),
                np.maximum(0.0, (p_next - p_touch) / p_touch * 1e4),
                0.0,
            )

        # Endpoint return at horizon (always from entry, regardless of fill timing)
        # Gross return in bps for the FILLED trade
        if side == "SHORT":
            endpoint_ret = np.where(
                (entry_px > 0),
                (entry_px - exit_px) / entry_px * 1e4,
                np.nan,
            )
            # Conditional return given filled at p_touch (what the SHORT trade would earn)
            cond_ret = np.where(
                touched_vec & (p_touch > 0),
                (p_touch - exit_px) / p_touch * 1e4,
                np.nan,
            )
        else:
            endpoint_ret = np.where(
                (entry_px > 0),
                (exit_px - entry_px) / entry_px * 1e4,
                np.nan,
            )
            cond_ret = np.where(
                touched_vec & (p_touch > 0),
                (exit_px - p_touch) / p_touch * 1e4,
                np.nan,
            )

        # Build calibration sample dicts
        samples = []
        for i in range(n):
            samples.append({
                "ts_sec": int(ts[i]),
                "spread": float(spr[i]),
                "trade_intensity": float(itns[i]),
                "vol_proxy": float(vol_proxy[i]),
                "imbalance_for_fill": float(imb_for_fill[i]),
                "entry_px": float(entry_px[i]),
                "exit_px": float(exit_px[i]),
                "touched": bool(touched_vec[i]),
                "depth": float(depth_vec[i]),
                "full_proxy": bool(full_proxy_vec[i]),
                "adverse_bps": float(adv[i]),
                "p_touch": float(p_touch[i]) if touched_vec[i] else None,
                "touch_idx": int(touch_idx_raw[i]),
                "cond_ret_bps": float(cond_ret[i]) if (touched_vec[i] and not np.isnan(cond_ret[i])) else None,
                "endpoint_ret_bps": float(endpoint_ret[i]) if not np.isnan(endpoint_ret[i]) else None,
            })
        return samples

    t0 = time.time()
    print("[signals] computing SELL calibration samples ...")
    sell_samples = build_signal_array(sell_df, "SHORT")
    print(f"[signals] SELL done: {len(sell_samples)} samples in {time.time()-t0:.1f}s")
    t0 = time.time()
    print("[signals] computing BUY calibration samples ...")
    buy_samples = build_signal_array(buy_df, "LONG")
    print(f"[signals] BUY done: {len(buy_samples)} samples in {time.time()-t0:.1f}s")
    return sell_samples, buy_samples

# ── Statistics helpers ────────────────────────────────────────────────────────

def describe_side(samples, side_label):
    """Compute summary statistics for one side's samples."""
    n = len(samples)
    if n == 0:
        return {"n": 0}
    touched = [s for s in samples if s["touched"]]
    full = [s for s in samples if s["full_proxy"]]
    touch_rate = len(touched) / n
    full_rate_cond = len(full) / len(touched) if touched else 0.0
    full_rate_joint = len(full) / n

    adv_all = [s["adverse_bps"] for s in samples]
    adv_touched = [s["adverse_bps"] for s in touched]

    cond_ret = [s["cond_ret_bps"] for s in touched if s["cond_ret_bps"] is not None]
    ep_ret_all = [s["endpoint_ret_bps"] for s in samples if s["endpoint_ret_bps"] is not None]

    touch_idxs = [s["touch_idx"] for s in touched]

    depths = [s["depth"] for s in touched]
    spreads = [s["spread"] for s in samples]

    def _pct(arr, q):
        if not arr:
            return float("nan")
        a = np.array(arr, dtype=float)
        return float(np.nanpercentile(a, q * 100))

    stats = {
        "side": side_label,
        "n": n,
        "n_touched": len(touched),
        "n_full": len(full),
        "touch_rate": touch_rate,
        "full_rate_cond_touch": full_rate_cond,
        "fill_rate_joint": full_rate_joint,
        # adverse_bps (over all signals, not just touched)
        "adv_mean_all": _fmean(adv_all),
        "adv_mean_touched": _fmean(adv_touched),
        "adv_p50_touched": _pct(adv_touched, 0.5),
        "adv_p90_touched": _pct(adv_touched, 0.9),
        # conditional return given fill
        "cond_ret_mean": _fmean(cond_ret, float("nan")),
        "cond_ret_p25": _pct(cond_ret, 0.25),
        "cond_ret_p50": _pct(cond_ret, 0.5),
        "cond_ret_p75": _pct(cond_ret, 0.75),
        # endpoint return unconditional (all signals)
        "ep_ret_mean": _fmean(ep_ret_all, float("nan")),
        # touch timing
        "touch_idx_mean": _fmean(touch_idxs) if touch_idxs else float("nan"),
        "touch_idx_p50": _pct(touch_idxs, 0.5),
        "touch_idx_p25": _pct(touch_idxs, 0.25),
        "touch_idx_p75": _pct(touch_idxs, 0.75),
        # depth
        "depth_mean": _fmean(depths),
        "depth_p50": _pct(depths, 0.5),
        # spreads
        "spread_mean": _fmean(spreads),
        "spread_p50": _pct(spreads, 0.5),
        # raw arrays for plotting
        "_adv_touched": adv_touched,
        "_touch_idxs": touch_idxs,
        "_cond_ret": cond_ret,
        "_spreads": spreads,
    }
    # NPA estimate = fill_rate * (mean_cond_ret - adverse_bps_mean_touched - 2*fee)
    cost_per_fill = _fmean(adv_touched, 0.0) + 2.0 * MAKER_FEE_BPS
    stats["npa_estimate_bps"] = full_rate_joint * (stats["cond_ret_mean"] - cost_per_fill)
    return stats


def describe_by_spread_tertile(samples):
    """Compute touch_rate by spread tertile."""
    spreads = [s["spread"] for s in samples]
    if not spreads:
        return {}
    q33 = np.percentile(spreads, 33.3)
    q67 = np.percentile(spreads, 66.7)
    bins = {"T1_tight": [], "T2_mid": [], "T3_wide": []}
    for s in samples:
        sp = s["spread"]
        if sp <= q33:
            bins["T1_tight"].append(s)
        elif sp <= q67:
            bins["T2_mid"].append(s)
        else:
            bins["T3_wide"].append(s)
    out = {}
    for k, ss in bins.items():
        n = len(ss)
        touched = sum(1 for s in ss if s["touched"])
        full = sum(1 for s in ss if s["full_proxy"])
        out[k] = {
            "n": n,
            "touch_rate": touched / n if n else 0.0,
            "full_rate_joint": full / n if n else 0.0,
            "spread_mean": _fmean([s["spread"] for s in ss]),
        }
    out["_edges"] = (q33, q67)
    return out

# ── Counterfactual ────────────────────────────────────────────────────────────

def counterfactual(sell_stats, buy_stats, sell_calib, buy_calib):
    """
    Estimate SELL NPA if SELL had BUY's fill rates (touch_rate, full_rate).
    Isolates whether NPA gap comes from fill rates vs conditional returns.
    """
    sell_cond_ret = sell_stats["cond_ret_mean"]
    sell_adv = sell_stats["adv_mean_touched"]
    buy_cond_ret = buy_stats["cond_ret_mean"]
    buy_adv = buy_stats["adv_mean_touched"]
    cost = 2.0 * MAKER_FEE_BPS

    actual_sell_npa = sell_stats["npa_estimate_bps"]
    actual_buy_npa = buy_stats["npa_estimate_bps"]

    # CF1: swap SELL fill rate to BUY fill rate, keep SELL returns
    sell_fill_joint = sell_stats["fill_rate_joint"]
    buy_fill_joint = buy_stats["fill_rate_joint"]
    cf1_npa = buy_fill_joint * (sell_cond_ret - sell_adv - cost)

    # CF2: swap SELL returns to BUY returns, keep SELL fill rate
    cf2_npa = sell_fill_joint * (buy_cond_ret - sell_adv - cost)

    # CF3: both fill rate AND returns equal to BUY
    cf3_npa = buy_fill_joint * (buy_cond_ret - buy_adv - cost)

    # Decompose the NPA gap
    gap = actual_sell_npa - actual_buy_npa
    gap_from_fillrate = cf1_npa - actual_buy_npa
    gap_from_returns = cf2_npa - actual_buy_npa

    return {
        "actual_sell_npa": actual_sell_npa,
        "actual_buy_npa": actual_buy_npa,
        "npa_ratio": actual_sell_npa / actual_buy_npa if actual_buy_npa != 0 else float("inf"),
        "gap_bps": gap,
        "cf1_npa_sell_fillrate_buy": cf1_npa,
        "cf2_npa_sell_returns_buy": cf2_npa,
        "cf3_npa_both_buy": cf3_npa,
        "gap_explained_by_fillrate": gap_from_fillrate,
        "gap_explained_by_returns": gap_from_returns,
        "sell_fill_joint": sell_fill_joint,
        "buy_fill_joint": buy_fill_joint,
        "fill_rate_ratio": sell_fill_joint / buy_fill_joint if buy_fill_joint > 0 else float("inf"),
        "sell_cond_ret": sell_cond_ret,
        "buy_cond_ret": buy_cond_ret,
        "cond_ret_ratio": sell_cond_ret / buy_cond_ret if buy_cond_ret != 0 else float("inf"),
    }

# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(sell_stats, buy_stats, sell_tert, buy_tert):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 09: Touch rate and fill rate bar chart (overall + spread tertile) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fill Probability: SELL (SHORT) vs BUY (LONG)", fontsize=13)

    ax = axes[0]
    cats = ["SELL", "BUY"]
    tr_vals = [sell_stats["touch_rate"], buy_stats["touch_rate"]]
    fr_vals = [sell_stats["fill_rate_joint"], buy_stats["fill_rate_joint"]]
    x = np.arange(2)
    w = 0.3
    b1 = ax.bar(x - w / 2, tr_vals, w, label="touch_rate", color=["#e8503a", "#4f8fc0"])
    b2 = ax.bar(x + w / 2, fr_vals, w, label="fill_rate (joint)", color=["#e8503a", "#4f8fc0"], alpha=0.5, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Rate")
    ax.set_title("Overall Fill Rates")
    ax.legend()
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    tert_labels = ["T1_tight", "T2_mid", "T3_wide"]
    sell_tr = [sell_tert.get(k, {}).get("touch_rate", 0.0) for k in tert_labels]
    buy_tr = [buy_tert.get(k, {}).get("touch_rate", 0.0) for k in tert_labels]
    x = np.arange(3)
    ax.bar(x - w / 2, sell_tr, w, label="SELL", color="#e8503a")
    ax.bar(x + w / 2, buy_tr, w, label="BUY", color="#4f8fc0")
    ax.set_xticks(x); ax.set_xticklabels(["Tight spr", "Mid spr", "Wide spr"])
    ax.set_ylabel("Touch rate")
    ax.set_title("Touch Rate by Spread Tertile")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "09_fill_rates.png", dpi=120)
    plt.close(fig)
    print("[plot] saved 09_fill_rates.png")

    # ── Plot 10: adverse_bps distribution (given touched) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("adverse_bps Distribution (conditional on touch)", fontsize=13)

    adv_s = np.array(sell_stats["_adv_touched"], dtype=float)
    adv_b = np.array(buy_stats["_adv_touched"], dtype=float)

    ax = axes[0]
    bins = np.linspace(0, 10, 40)
    ax.hist(adv_s, bins=bins, density=True, alpha=0.6, label="SELL", color="#e8503a")
    ax.hist(adv_b, bins=bins, density=True, alpha=0.6, label="BUY", color="#4f8fc0")
    ax.axvline(np.mean(adv_s), color="#e8503a", lw=2, ls="--", label=f"SELL mean {np.mean(adv_s):.3f}")
    ax.axvline(np.mean(adv_b), color="#4f8fc0", lw=2, ls="--", label=f"BUY mean {np.mean(adv_b):.3f}")
    ax.set_xlabel("adverse_bps"); ax.set_ylabel("Density")
    ax.set_title("adverse_bps Histogram (0-10 bps)")
    ax.legend(fontsize=8)

    ax = axes[1]
    pcts = np.arange(0, 101, 5)
    if len(adv_s) > 0 and len(adv_b) > 0:
        ax.plot(pcts, np.percentile(adv_s, pcts), color="#e8503a", label="SELL")
        ax.plot(pcts, np.percentile(adv_b, pcts), color="#4f8fc0", label="BUY")
    ax.set_xlabel("Percentile"); ax.set_ylabel("adverse_bps")
    ax.set_title("adverse_bps CDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "10_adverse_bps_dist.png", dpi=120)
    plt.close(fig)
    print("[plot] saved 10_adverse_bps_dist.png")

    # ── Plot 11: Touch timing distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Touch Timing: When in 120s does the limit get hit?", fontsize=13)

    ti_s = np.array(sell_stats["_touch_idxs"], dtype=float)
    ti_b = np.array(buy_stats["_touch_idxs"], dtype=float)

    ax = axes[0]
    bins = np.arange(0, HORIZON + 1, 5)
    ax.hist(ti_s, bins=bins, density=True, alpha=0.6, label="SELL", color="#e8503a")
    ax.hist(ti_b, bins=bins, density=True, alpha=0.6, label="BUY", color="#4f8fc0")
    ax.set_xlabel("Touch index (seconds after entry)"); ax.set_ylabel("Density")
    ax.set_title("Touch Timing Histogram")
    ax.legend()

    ax = axes[1]
    pcts = np.arange(0, 101, 5)
    if len(ti_s) > 0 and len(ti_b) > 0:
        ax.plot(pcts, np.percentile(ti_s, pcts), color="#e8503a", label="SELL")
        ax.plot(pcts, np.percentile(ti_b, pcts), color="#4f8fc0", label="BUY")
    ax.set_xlabel("Percentile"); ax.set_ylabel("Touch index (s)")
    ax.set_title("Touch Timing CDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "11_touch_timing.png", dpi=120)
    plt.close(fig)
    print("[plot] saved 11_touch_timing.png")

    # ── Plot 12: Conditional return distribution (given touched) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Trade Return (given touched, at 120s horizon)", fontsize=13)

    cr_s = np.array(sell_stats["_cond_ret"], dtype=float)
    cr_b = np.array(buy_stats["_cond_ret"], dtype=float)

    ax = axes[0]
    bins = np.linspace(-40, 40, 50)
    ax.hist(cr_s, bins=bins, density=True, alpha=0.6, label="SELL", color="#e8503a")
    ax.hist(cr_b, bins=bins, density=True, alpha=0.6, label="BUY", color="#4f8fc0")
    sell_cm = float(np.mean(cr_s)) if len(cr_s) > 0 else float("nan")
    buy_cm = float(np.mean(cr_b)) if len(cr_b) > 0 else float("nan")
    ax.axvline(sell_cm, color="#e8503a", lw=2, ls="--", label=f"SELL mean {sell_cm:.2f}")
    ax.axvline(buy_cm, color="#4f8fc0", lw=2, ls="--", label=f"BUY mean {buy_cm:.2f}")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Return (bps)"); ax.set_ylabel("Density")
    ax.set_title("Conditional Return (given touched)")
    ax.legend(fontsize=8)

    ax = axes[1]
    pcts = np.arange(0, 101, 5)
    if len(cr_s) > 0 and len(cr_b) > 0:
        ax.plot(pcts, np.percentile(cr_s, pcts), color="#e8503a", label="SELL")
        ax.plot(pcts, np.percentile(cr_b, pcts), color="#4f8fc0", label="BUY")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Percentile"); ax.set_ylabel("Return (bps)")
    ax.set_title("Conditional Return CDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "12_conditional_returns.png", dpi=120)
    plt.close(fig)
    print("[plot] saved 12_conditional_returns.png")

# ── Report writer ─────────────────────────────────────────────────────────────

def fmt(v, digits=4):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_report(sell_stats, buy_stats, sell_tert, buy_tert, sell_calib, buy_calib, cf):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    def h1(s): lines.append(f"# {s}\n")
    def h2(s): lines.append(f"\n## {s}\n")
    def h3(s): lines.append(f"\n### {s}\n")
    def row(*cols): lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(*widths): lines.append("|" + "|".join(f"{'---':>{w}}" for w in widths) + "|")
    def blank(): lines.append("")

    h1("FILL CALIBRATION AUDIT -- ETHUSDT")
    lines.append(f"\n**Generated:** {now}\n")
    lines.append("""
## Context

Prior analysis (NPA_DECOMPOSITION_REGIME.md) showed:
- Sell NPA = 6.3e-05 vs Buy NPA = 7.0e-06  (9:1 ratio)
- Path excursions are SYMMETRIC (sell_adverse=10.21 bps vs buy_adverse=10.15 bps)
- NPA gap is NOT from directional hit rates (both ~55-56%)

This report isolates the gap by auditing the passive fill probability model:
  - touch_rate: does price reach the passive limit within the 120s hold?
  - full_proxy_rate: is depth >= 0.5 when touched (proxy for queue priority)?
  - adverse_bps: 1-step reversal immediately after fill
  - conditional return: trade P&L at horizon given fill

Passive fill mechanics:
  SHORT limit = ep*(1+0.5*spread)  -- fills when price goes UP
  LONG  limit = ep*(1-0.5*spread)  -- fills when price goes DOWN

SELL signals: imbalance <= -0.5  |  BUY signals: imbalance >= +0.5
Top pocket: trade_intensity>=3500, spread<=0.0003, horizon=120s
""")

    h2("1. Basic Fill Statistics")

    row("Metric", "SELL (SHORT)", "BUY (LONG)", "Sell / Buy ratio")
    sep(25, 14, 14, 16)
    def ratio(a, b):
        if b and b != 0:
            return fmt(a / b, 2)
        return "inf"
    row("n signals", sell_stats["n"], buy_stats["n"], "—")
    row("n touched", sell_stats["n_touched"], buy_stats["n_touched"], "—")
    row("n full_proxy", sell_stats["n_full"], buy_stats["n_full"], "—")
    row("touch_rate", fmt(sell_stats["touch_rate"]), fmt(buy_stats["touch_rate"]), ratio(sell_stats["touch_rate"], buy_stats["touch_rate"]))
    row("full_rate | touch", fmt(sell_stats["full_rate_cond_touch"]), fmt(buy_stats["full_rate_cond_touch"]), ratio(sell_stats["full_rate_cond_touch"], buy_stats["full_rate_cond_touch"]))
    row("fill_rate (joint)", fmt(sell_stats["fill_rate_joint"]), fmt(buy_stats["fill_rate_joint"]), ratio(sell_stats["fill_rate_joint"], buy_stats["fill_rate_joint"]))
    row("depth_mean | touch", fmt(sell_stats["depth_mean"]), fmt(buy_stats["depth_mean"]), ratio(sell_stats["depth_mean"], buy_stats["depth_mean"]))
    row("depth_p50 | touch", fmt(sell_stats["depth_p50"]), fmt(buy_stats["depth_p50"]), ratio(sell_stats["depth_p50"], buy_stats["depth_p50"]))
    row("spread_mean", fmt(sell_stats["spread_mean"], 6), fmt(buy_stats["spread_mean"], 6), "—")
    row("spread_p50", fmt(sell_stats["spread_p50"], 6), fmt(buy_stats["spread_p50"], 6), "—")
    blank()

    h2("2. adverse_bps Distribution (1-step reversal after touch)")
    lines.append("""
> adverse_bps is computed at the moment of fill (touch), not over the full hold.
> Formula:
>   SHORT: max(0, (p_touch - p_next) / p_touch * 10000)
>   LONG:  max(0, (p_next - p_touch) / p_touch * 10000)
> This measures immediate 1-second reversal AFTER the passive limit is touched.
""")
    row("Stat", "SELL (SHORT)", "BUY (LONG)", "Sell - Buy")
    sep(25, 14, 14, 12)
    def diff(a, b):
        try:
            return fmt(a - b)
        except Exception:
            return "nan"
    row("adverse_bps mean (all)", fmt(sell_stats["adv_mean_all"]), fmt(buy_stats["adv_mean_all"]), diff(sell_stats["adv_mean_all"], buy_stats["adv_mean_all"]))
    row("adverse_bps mean (touched)", fmt(sell_stats["adv_mean_touched"]), fmt(buy_stats["adv_mean_touched"]), diff(sell_stats["adv_mean_touched"], buy_stats["adv_mean_touched"]))
    row("adverse_bps p50 (touched)", fmt(sell_stats["adv_p50_touched"]), fmt(buy_stats["adv_p50_touched"]), diff(sell_stats["adv_p50_touched"], buy_stats["adv_p50_touched"]))
    row("adverse_bps p90 (touched)", fmt(sell_stats["adv_p90_touched"]), fmt(buy_stats["adv_p90_touched"]), diff(sell_stats["adv_p90_touched"], buy_stats["adv_p90_touched"]))
    blank()

    h2("3. Touch Timing Distribution")
    lines.append("Mean seconds to first touch (within 120s hold), conditional on touching.\n")
    row("Stat", "SELL (SHORT)", "BUY (LONG)")
    sep(25, 14, 14)
    row("touch_idx mean (s)", fmt(sell_stats["touch_idx_mean"]), fmt(buy_stats["touch_idx_mean"]))
    row("touch_idx p25 (s)", fmt(sell_stats["touch_idx_p25"]), fmt(buy_stats["touch_idx_p25"]))
    row("touch_idx p50 (s)", fmt(sell_stats["touch_idx_p50"]), fmt(buy_stats["touch_idx_p50"]))
    row("touch_idx p75 (s)", fmt(sell_stats["touch_idx_p75"]), fmt(buy_stats["touch_idx_p75"]))
    blank()

    h2("4. Conditional Trade Returns (given touched at limit)")
    lines.append("""
Return computed from the FILL PRICE to the 120s horizon exit price.
  SHORT: (p_touch - p_exit) / p_touch * 10000 bps
  LONG:  (p_exit - p_touch) / p_touch * 10000 bps
""")
    row("Stat", "SELL (SHORT)", "BUY (LONG)", "Sell - Buy")
    sep(25, 14, 14, 12)
    row("cond_ret mean (bps)", fmt(sell_stats["cond_ret_mean"]), fmt(buy_stats["cond_ret_mean"]), diff(sell_stats["cond_ret_mean"], buy_stats["cond_ret_mean"]))
    row("cond_ret p25 (bps)", fmt(sell_stats["cond_ret_p25"]), fmt(buy_stats["cond_ret_p25"]), diff(sell_stats["cond_ret_p25"], buy_stats["cond_ret_p25"]))
    row("cond_ret p50 (bps)", fmt(sell_stats["cond_ret_p50"]), fmt(buy_stats["cond_ret_p50"]), diff(sell_stats["cond_ret_p50"], buy_stats["cond_ret_p50"]))
    row("cond_ret p75 (bps)", fmt(sell_stats["cond_ret_p75"]), fmt(buy_stats["cond_ret_p75"]), diff(sell_stats["cond_ret_p75"], buy_stats["cond_ret_p75"]))
    blank()

    h2("5. Touch Rate by Spread Tertile")
    lines.append("Touch rate within each spread tercile (T1=tight, T2=mid, T3=wide).\n")
    row("Tertile", "SELL n", "SELL touch_rate", "SELL fill_rate", "BUY n", "BUY touch_rate", "BUY fill_rate", "spread_mean")
    sep(12, 9, 15, 14, 9, 15, 14, 12)
    for k in ["T1_tight", "T2_mid", "T3_wide"]:
        ss = sell_tert.get(k, {}); bs = buy_tert.get(k, {})
        row(
            k,
            ss.get("n", 0), fmt(ss.get("touch_rate", 0.0)), fmt(ss.get("full_rate_joint", 0.0)),
            bs.get("n", 0), fmt(bs.get("touch_rate", 0.0)), fmt(bs.get("full_rate_joint", 0.0)),
            fmt(ss.get("spread_mean", float("nan")), 5),
        )
    blank()

    h2("6. Calibrated Model Parameters")
    lines.append("""
Parameters from inline replication of calibrate_passive_model (same algorithm as
execution/passive_execution_simulator.py). These are the base rates the simulator
blends against per-feature tertile corrections.
""")
    row("Parameter", "SELL", "BUY")
    sep(30, 12, 12)
    row("n samples", sell_calib["n"], buy_calib["n"])
    row("n touched", sell_calib["n_touched"], buy_calib["n_touched"])
    row("base_touch", fmt(sell_calib["base_touch"]), fmt(buy_calib["base_touch"]))
    row("base_full_cond_touch", fmt(sell_calib["base_full_cond_touch"]), fmt(buy_calib["base_full_cond_touch"]))
    row("implied_fill_rate = base_touch x base_full", fmt(sell_calib["base_touch"] * sell_calib["base_full_cond_touch"]), fmt(buy_calib["base_touch"] * buy_calib["base_full_cond_touch"]))
    row("base_adverse_bps", fmt(sell_calib["base_adverse_bps"]), fmt(buy_calib["base_adverse_bps"]))
    row("depth_full_threshold", fmt(sell_calib["depth_full_threshold"]), fmt(buy_calib["depth_full_threshold"]))
    blank()

    h2("7. NPA Decomposition and Counterfactual")
    lines.append(f"""
NPA estimate (bps per signal):
  NPA = fill_rate_joint * (cond_ret_mean - adverse_bps_touched_mean - 2*maker_fee)
  maker_fee = {MAKER_FEE_BPS} bps one-way

Note: This is an approximation. The actual validate_pocket_forward NPA uses the
full simulation (including queue competition penalty and partial fill logic).
""")
    row("Component", "SELL", "BUY", "Sell / Buy")
    sep(35, 12, 12, 12)
    row("fill_rate_joint", fmt(cf["sell_fill_joint"]), fmt(cf["buy_fill_joint"]), fmt(cf["fill_rate_ratio"], 2))
    row("cond_ret_mean (bps)", fmt(cf["sell_cond_ret"]), fmt(cf["buy_cond_ret"]), fmt(cf["cond_ret_ratio"], 2))
    row("NPA estimate (bps)", fmt(cf["actual_sell_npa"]), fmt(cf["actual_buy_npa"]), fmt(cf["npa_ratio"], 2))
    blank()
    lines.append("**Counterfactual analysis** (swapping SELL parameters to BUY values):\n")
    row("Scenario", "SELL NPA (bps)", "vs actual SELL", "Interpretation")
    sep(40, 15, 15, 35)
    row("Actual SELL", fmt(cf["actual_sell_npa"]), "—", "Baseline")
    row("CF1: use BUY fill_rate, keep SELL cond_ret", fmt(cf["cf1_npa_sell_fillrate_buy"]), fmt(cf["cf1_npa_sell_fillrate_buy"] - cf["actual_sell_npa"]), "Isolate fill_rate effect")
    row("CF2: keep SELL fill_rate, use BUY cond_ret", fmt(cf["cf2_npa_sell_returns_buy"]), fmt(cf["cf2_npa_sell_returns_buy"] - cf["actual_sell_npa"]), "Isolate return effect")
    row("CF3: use BUY fill_rate + BUY cond_ret", fmt(cf["cf3_npa_both_buy"]), fmt(cf["cf3_npa_both_buy"] - cf["actual_sell_npa"]), "Full convergence = actual BUY")
    row("Actual BUY", fmt(cf["actual_buy_npa"]), "—", "Reference")
    blank()

    h2("8. Verdict: Source of 9:1 NPA Gap")
    fill_ratio = cf["fill_rate_ratio"]
    ret_ratio = cf["cond_ret_ratio"]
    lines.append(f"""
**Fill rate ratio (SELL/BUY):** {fill_ratio:.2f}x
**Conditional return ratio (SELL/BUY):** {ret_ratio:.2f}x

| Factor | Value | Contribution to gap |
|---|---:|---|
""")
    # Interpret
    fill_dominates = abs(fill_ratio - 1.0) > abs(ret_ratio - 1.0)
    if fill_dominates:
        interp = ("The NPA gap is primarily driven by ASYMMETRIC FILL RATES. "
                  "SELL signals achieve higher touch and fill rates, likely because "
                  "SELL signals fire in UP-momentum conditions where the passive SELL limit "
                  "(placed above market) is more readily touched by the upward price movement. "
                  "After touch, the momentum reverses (mean-reversion alpha) and the SHORT trade profits.")
    else:
        interp = ("The NPA gap is primarily driven by ASYMMETRIC CONDITIONAL RETURNS. "
                  "SELL signals, once filled, achieve better 120s returns than BUY signals. "
                  "This suggests the sell-side alpha (mean-reversion after momentum) has "
                  "stronger post-fill directional persistence than the buy-side alpha.")
    lines.append(f"| fill_rate | {fill_ratio:.2f}x | {'PRIMARY' if fill_dominates else 'secondary'} |\n")
    lines.append(f"| cond_ret  | {ret_ratio:.2f}x | {'PRIMARY' if not fill_dominates else 'secondary'} |\n")
    lines.append(f"\n{interp}\n")

    lines.append("\n---\n*Generated by `tools/audit_fill_calibration.py` -- self-contained.*\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] written to {REPORT}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fill calibration audit for sell vs buy passive pockets")
    ap.add_argument("--db", default=DB_DEFAULT, help="SQLite DB path")
    args = ap.parse_args()

    t_start = time.time()
    print(f"=== Fill Calibration Audit === db={args.db}")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, mark_arr, min_sec = load_data(args.db)

    # Extract calibration samples for SELL and BUY
    sell_samples, buy_samples = extract_signals(df, mark_arr, min_sec)

    # Summary statistics
    print("[stats] computing summary statistics ...")
    sell_stats = describe_side(sell_samples, "SELL")
    buy_stats = describe_side(buy_samples, "BUY")

    sell_tert = describe_by_spread_tertile(sell_samples)
    buy_tert = describe_by_spread_tertile(buy_samples)

    # Calibrate model (inline replication)
    print("[calib] calibrating passive model for each side ...")
    sell_calib = calibrate_inline(sell_samples)
    buy_calib = calibrate_inline(buy_samples)

    # Counterfactual
    cf = counterfactual(sell_stats, buy_stats, sell_calib, buy_calib)

    # Print key numbers
    print()
    print("  ---- KEY RESULTS ----")
    print(f"  SELL: n={sell_stats['n']}, touch={sell_stats['touch_rate']:.3f}, "
          f"fill_joint={sell_stats['fill_rate_joint']:.3f}, "
          f"cond_ret={sell_stats['cond_ret_mean']:.3f} bps, "
          f"NPA_est={sell_stats['npa_estimate_bps']:.5f} bps")
    print(f"  BUY:  n={buy_stats['n']}, touch={buy_stats['touch_rate']:.3f}, "
          f"fill_joint={buy_stats['fill_rate_joint']:.3f}, "
          f"cond_ret={buy_stats['cond_ret_mean']:.3f} bps, "
          f"NPA_est={buy_stats['npa_estimate_bps']:.5f} bps")
    print(f"  Fill_rate ratio (SELL/BUY): {cf['fill_rate_ratio']:.2f}x")
    print(f"  Cond_ret ratio  (SELL/BUY): {cf['cond_ret_ratio']:.2f}x")
    print(f"  NPA ratio       (SELL/BUY): {cf['npa_ratio']:.2f}x")
    print()

    # Generate plots
    print("[plots] generating ...")
    make_plots(sell_stats, buy_stats, sell_tert, buy_tert)

    # Write markdown report
    write_report(sell_stats, buy_stats, sell_tert, buy_tert, sell_calib, buy_calib, cf)

    print(f"\n=== done in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
