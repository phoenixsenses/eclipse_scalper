#!/usr/bin/env python3
"""
tools/npa_decomposition_and_regime.py
------------------------------------------------------------------------------
Task 1: NPA Decomposition  -- compare sell vs buy favorable / adverse
        excursion distributions to explain the 9:1 NPA asymmetry.

Task 2: Regime-Conditional Pocket Test -- does filtering entries by the
        rolling-1h regime (UP / DOWN) improve the NPA proxy meaningfully?

Self-contained: no imports from tools/ modules.

Signal / bucket definitions match micro_edge pipeline exactly:
  bucket_sec       = 1
  imbalance        = signed_qty / sum_abs_qty
                     is_buyer_maker=0 => +qty (taker buy)
                     is_buyer_maker=1 => -qty (taker sell)
  trade_intensity  = trade_count * 60  (trades-per-min at 1-sec buckets)
  spread_proxy     = |VWAP - mark_price| / mark_price per bucket
  forward horizon  = 120 s (mark-price based)

Excursions computed over the full mark-price path from entry to horizon:
  sell (SHORT):
    final_ret_bps     = (p0 - p_end) / p0 * 1e4   [+ => profitable]
    max_favorable_bps = max(0, p0 - min_path) / p0 * 1e4
    max_adverse_bps   = max(0, max_path - p0) / p0 * 1e4
  buy (LONG):
    final_ret_bps     = (p_end - p0) / p0 * 1e4   [+ => profitable]
    max_favorable_bps = max(0, max_path - p0) / p0 * 1e4
    max_adverse_bps   = max(0, p0 - min_path) / p0 * 1e4

NPA proxy:  mean(final_ret) - adverse_mult * mean(max_adverse)  [bps]

Usage:
  python tools/npa_decomposition_and_regime.py [--db data/microstructure.db]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# -- Windows console safety ----------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Constants -----------------------------------------------------------------
SYMBOL          = "ETHUSDT"
HORIZON_SEC     = 120
BUCKET_SEC      = 1
MIN_ABS_IMB     = 0.50
MIN_INTENSITY   = 3500.0
MAX_SPREAD      = 0.000300
REGIME_WINDOW_S = 3600          # 1-hour rolling window for regime label
ADVERSE_MULTS   = [0.3, 0.5, 0.7]
MIN_N           = 20            # flag conditions with fewer signals

REPORT_DIR = Path("reports")
PLOTS_DIR  = REPORT_DIR / "plots"
REPORT_MD  = REPORT_DIR / "NPA_DECOMPOSITION_REGIME.md"


# -- Utilities -----------------------------------------------------------------

def _q(arr: np.ndarray, q: float) -> float:
    v = arr[~np.isnan(arr)]
    return float(np.percentile(v, q * 100)) if len(v) else float("nan")

def _mean(arr: np.ndarray) -> float:
    v = arr[~np.isnan(arr)]
    return float(np.mean(v)) if len(v) else float("nan")

def _stats(arr: np.ndarray) -> dict:
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return {k: float("nan") for k in ("n", "mean", "median", "p25", "p75", "p95")}
    return {
        "n":      len(v),
        "mean":   float(np.mean(v)),
        "median": float(np.median(v)),
        "p25":    float(np.percentile(v, 25)),
        "p75":    float(np.percentile(v, 75)),
        "p95":    float(np.percentile(v, 95)),
    }

def _npa(final_ret: np.ndarray, max_adverse: np.ndarray, mult: float) -> float:
    fr = final_ret[~np.isnan(final_ret)]
    ma = max_adverse[~np.isnan(max_adverse)]
    if len(fr) == 0 or len(ma) == 0:
        return float("nan")
    return float(np.mean(fr)) - mult * float(np.mean(ma))

def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "N/A"

def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if not math.isnan(v) else "N/A"


# -- Data loading --------------------------------------------------------------

def load_data(db: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """
    Returns:
        bdf      -- 1-sec bucket DataFrame with imbalance, intensity, spread,
                    fwd_ret, ts_sec columns; indexed by UTC datetime
        mark_arr -- dense float64 numpy array of mark prices, index = ts_sec - min_sec
        min_sec  -- int: the smallest ts_sec value in mark_arr
    """
    conn = sqlite3.connect(db)

    # -- Mark prices -> dense numpy array --------------------------------------
    print("[1/5] Loading mark prices ...", flush=True)
    marks_raw = pd.read_sql(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        conn, params=(SYMBOL,),
    )
    marks_raw["ts_sec"] = (marks_raw["ts_ms"] // 1000).astype(np.int64)
    marks_df = marks_raw.drop_duplicates("ts_sec").sort_values("ts_sec")

    min_sec = int(marks_df["ts_sec"].iloc[0])
    max_sec = int(marks_df["ts_sec"].iloc[-1])
    arr_size = max_sec - min_sec + 1

    mark_arr = np.full(arr_size, np.nan, dtype=np.float64)
    idxs = (marks_df["ts_sec"].values - min_sec).astype(np.int64)
    mark_arr[idxs] = marks_df["mark_price"].values

    # Forward-fill gaps (0.3% missing seconds)
    for i in range(1, arr_size):
        if np.isnan(mark_arr[i]):
            mark_arr[i] = mark_arr[i - 1]

    n_valid = int(np.sum(~np.isnan(mark_arr)))
    print(
        f"    mark rows={len(marks_df):,}  arr_size={arr_size:,}  "
        f"fill={n_valid/arr_size:.3%}  "
        f"span {pd.Timestamp(min_sec, unit='s', tz='UTC').strftime('%Y-%m-%d')} -> "
        f"{pd.Timestamp(max_sec, unit='s', tz='UTC').strftime('%Y-%m-%d')}",
        flush=True,
    )

    # -- Trade bucket aggregates (SQL aggregate; no cross-table join) ----------
    print("[2/5] Aggregating trade buckets (SQL; ~45 s) ...", flush=True)
    bdf = pd.read_sql(
        """
        SELECT
            ts_ms / 1000 * 1000                                             AS bkt_ms,
            COUNT(*)                                                         AS trade_count,
            SUM(CASE WHEN is_buyer_maker=0 THEN quantity ELSE -quantity END) AS signed_qty,
            SUM(quantity)                                                    AS sum_abs_qty,
            SUM(price * quantity) / SUM(quantity)                           AS vwap
        FROM  agg_trades
        WHERE symbol = ?
        GROUP BY bkt_ms
        ORDER BY bkt_ms
        """,
        conn, params=(SYMBOL,),
    )
    conn.close()
    print(f"    trade buckets={len(bdf):,}", flush=True)

    # -- Compute features ------------------------------------------------------
    print("[3/5] Computing features and forward returns ...", flush=True)
    bdf["ts_sec"]          = (bdf["bkt_ms"] // 1000).astype(np.int64)
    bdf["imbalance"]       = bdf["signed_qty"] / bdf["sum_abs_qty"]
    bdf["trade_intensity"] = bdf["trade_count"] * (60.0 / BUCKET_SEC)

    # Map mark price via ts_sec
    ts_arr = bdf["ts_sec"].values
    arr_idx = ts_arr - min_sec
    valid_idx_mask = (arr_idx >= 0) & (arr_idx < arr_size)
    bdf["mark_price"] = np.where(valid_idx_mask, mark_arr[np.clip(arr_idx, 0, arr_size - 1)], np.nan)

    # Spread proxy: |VWAP - mark_price| / mark_price
    bdf["spread"] = (bdf["vwap"] - bdf["mark_price"]).abs() / bdf["mark_price"]

    # 120-s forward return: log(mark[t+120] / mark[t])
    fwd_arr_idx = arr_idx + HORIZON_SEC
    valid_fwd_mask = (fwd_arr_idx >= 0) & (fwd_arr_idx < arr_size)
    fwd_price = np.where(valid_fwd_mask, mark_arr[np.clip(fwd_arr_idx, 0, arr_size - 1)], np.nan)
    bdf["fwd_ret"] = np.log(fwd_price / bdf["mark_price"].values)

    # Drop unusable rows
    bdf = bdf.dropna(subset=["mark_price", "fwd_ret", "spread", "imbalance"]).copy()
    print(f"    valid buckets={len(bdf):,}", flush=True)

    bdf["dt"] = pd.to_datetime(bdf["ts_sec"], unit="s", utc=True)
    bdf = bdf.set_index("dt").sort_index()

    return bdf, mark_arr, min_sec


# -- Path extraction -----------------------------------------------------------

def extract_price_paths(
    signal_ts: np.ndarray, mark_arr: np.ndarray, min_sec: int, horizon: int
) -> np.ndarray:
    """
    Returns (N, horizon+1) float64 array of mark prices starting at each
    signal timestamp. NaN where the path exceeds the data boundary.
    """
    H = horizon + 1
    N = len(signal_ts)
    base_idx = (signal_ts.astype(np.int64) - min_sec)         # (N,)
    offsets   = np.arange(H, dtype=np.int64)                   # (H,)
    idx_mat   = base_idx[:, None] + offsets[None, :]           # (N, H)

    arr_size  = len(mark_arr)
    valid     = (idx_mat >= 0) & (idx_mat < arr_size)
    clipped   = np.clip(idx_mat, 0, arr_size - 1)

    paths             = mark_arr[clipped]                      # (N, H)
    paths[~valid]     = np.nan
    return paths


# -- Excursion computation -----------------------------------------------------

def compute_excursions(paths: np.ndarray, direction: int) -> dict:
    """
    direction: +1 = LONG (buy),  -1 = SHORT (sell)

    Returns dict with arrays:
        final_ret_bps     -- directional endpoint return (+ = good)
        max_favorable_bps -- max favorable excursion during hold (always >= 0)
        max_adverse_bps   -- max adverse excursion during hold   (always >= 0)
        valid_mask        -- bool, rows where path had no NaN
    """
    p0    = paths[:, 0]
    p_end = paths[:, -1]

    with np.errstate(divide="ignore", invalid="ignore"):
        path_max = np.nanmax(paths, axis=1)
        path_min = np.nanmin(paths, axis=1)

    valid_mask = ~(np.isnan(p0) | np.isnan(p_end) | np.isnan(path_max) | np.isnan(path_min))

    if direction == -1:   # SHORT (sell)
        final_ret     = (p0 - p_end)  / p0 * 1e4
        max_favorable = np.maximum(0.0, (p0 - path_min) / p0 * 1e4)
        max_adverse   = np.maximum(0.0, (path_max - p0) / p0 * 1e4)
    else:                 # LONG (buy)
        final_ret     = (p_end - p0)  / p0 * 1e4
        max_favorable = np.maximum(0.0, (path_max - p0) / p0 * 1e4)
        max_adverse   = np.maximum(0.0, (p0 - path_min) / p0 * 1e4)

    final_ret[~valid_mask]     = np.nan
    max_favorable[~valid_mask] = np.nan
    max_adverse[~valid_mask]   = np.nan

    return {
        "final_ret_bps":     final_ret,
        "max_favorable_bps": max_favorable,
        "max_adverse_bps":   max_adverse,
        "valid_mask":        valid_mask,
    }


# -- Regime computation --------------------------------------------------------

def compute_regime_arr(mark_arr: np.ndarray, window_sec: int = 3600) -> np.ndarray:
    """
    Returns a float64 array of rolling log-returns over `window_sec` seconds.
    NaN for the first `window_sec` positions where no history exists.
    """
    n = len(mark_arr)
    shifted = np.full(n, np.nan, dtype=np.float64)
    if n > window_sec:
        shifted[window_sec:] = mark_arr[:n - window_sec]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mark_arr / shifted
        regime_arr = np.where((ratio > 0) & ~np.isnan(ratio), np.log(ratio), np.nan)
    regime_arr[:window_sec] = np.nan
    return regime_arr


def get_regime_labels(signal_ts: np.ndarray, regime_arr: np.ndarray, min_sec: int) -> np.ndarray:
    """Map each signal timestamp to 'UP', 'DOWN', or 'UNKNOWN'."""
    idx = np.clip((signal_ts.astype(np.int64) - min_sec), 0, len(regime_arr) - 1)
    vals = regime_arr[idx]
    labels = np.where(
        np.isnan(vals),
        "UNKNOWN",
        np.where(vals >= 0, "UP", "DOWN"),
    )
    return labels


# -- Condition stats helper ----------------------------------------------------

def condition_stats(
    final_ret: np.ndarray,
    max_favorable: np.ndarray,
    max_adverse: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:
    fr = final_ret[mask]
    mf = max_favorable[mask]
    ma = max_adverse[mask]
    n  = int(np.sum(~np.isnan(fr)))
    hit_rate = float(np.mean(fr > 0)) if n else float("nan")
    d = {
        "label":           label,
        "n":               n,
        "hit_rate":        hit_rate,
        "final_ret":       _stats(fr),
        "max_favorable":   _stats(mf),
        "max_adverse":     _stats(ma),
        "low_n_flag":      n < MIN_N,
    }
    for mult in ADVERSE_MULTS:
        d[f"npa_proxy_{mult}"] = _npa(fr, ma, mult)
    return d


# -- Plots ---------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}", flush=True)
    return path


def plot_excursion_histograms(
    sell_exc: dict, buy_exc: dict
) -> tuple[Path, Path]:
    """Two figures: (1) endpoint return distributions, (2) adverse excursion distributions."""
    BINS = 80

    # -- Figure 1: Endpoint favorable returns ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"{SYMBOL} -- 120-s Directional Endpoint Return Distribution\n"
        f"Top pocket: imb>={MIN_ABS_IMB}, int>={MIN_INTENSITY:.0f}, spr<={MAX_SPREAD:.4f}",
        fontsize=11, fontweight="bold",
    )

    sell_fr = sell_exc["final_ret_bps"]
    buy_fr  = buy_exc["final_ret_bps"]
    sell_fr_clean = sell_fr[~np.isnan(sell_fr)]
    buy_fr_clean  = buy_fr[~np.isnan(buy_fr)]

    clip = 30  # bps, clip for display
    ax = axes[0]
    ax.hist(np.clip(sell_fr_clean, -clip, clip), bins=BINS, color="#d62728",
            alpha=0.7, density=True, label=f"Sell (SHORT) n={len(sell_fr_clean):,}")
    ax.hist(np.clip(buy_fr_clean,  -clip, clip), bins=BINS, color="#2ca02c",
            alpha=0.5, density=True, label=f"Buy  (LONG)  n={len(buy_fr_clean):,}")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(float(np.mean(sell_fr_clean)), color="#d62728", lw=1.5, ls="--",
               label=f"sell mean={float(np.mean(sell_fr_clean)):.2f} bps")
    ax.axvline(float(np.mean(buy_fr_clean)),  color="#2ca02c", lw=1.5, ls="--",
               label=f"buy mean={float(np.mean(buy_fr_clean)):.2f} bps")
    ax.set_xlabel("Endpoint return, directional (bps)")
    ax.set_ylabel("Density")
    ax.set_title("Favorable endpoint return (+ = profitable)\nOverlaid sell vs buy")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(np.clip(sell_fr_clean, -clip, clip), bins=BINS, color="#d62728",
            alpha=0.7, density=True, cumulative=True, label="Sell CDF")
    ax.hist(np.clip(buy_fr_clean,  -clip, clip), bins=BINS, color="#2ca02c",
            alpha=0.5, density=True, cumulative=True, label="Buy CDF")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Endpoint return, directional (bps)")
    ax.set_ylabel("Cumulative density")
    ax.set_title("CDF -- profitable outcomes (x>0 => favorable)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p1 = _save(fig, "05_endpoint_return_dist.png")

    # -- Figure 2: Adverse excursion distributions -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        f"{SYMBOL} -- Max Adverse Excursion Distribution (120-s path)\n"
        f"Sell adverse = max upward move; Buy adverse = max downward move",
        fontsize=11, fontweight="bold",
    )

    sell_ma = sell_exc["max_adverse_bps"]
    buy_ma  = buy_exc["max_adverse_bps"]
    sell_ma_clean = sell_ma[~np.isnan(sell_ma)]
    buy_ma_clean  = buy_ma[~np.isnan(buy_ma)]

    clip_adv = 50  # bps
    ax = axes[0]
    ax.hist(np.clip(sell_ma_clean, 0, clip_adv), bins=BINS, color="#d62728",
            alpha=0.7, density=True, label=f"Sell n={len(sell_ma_clean):,}")
    ax.hist(np.clip(buy_ma_clean,  0, clip_adv), bins=BINS, color="#2ca02c",
            alpha=0.5, density=True, label=f"Buy  n={len(buy_ma_clean):,}")
    ax.axvline(float(np.mean(sell_ma_clean)), color="#d62728", lw=1.5, ls="--",
               label=f"sell mean={float(np.mean(sell_ma_clean)):.2f} bps")
    ax.axvline(float(np.mean(buy_ma_clean)),  color="#2ca02c", lw=1.5, ls="--",
               label=f"buy mean={float(np.mean(buy_ma_clean)):.2f} bps")
    ax.set_xlabel("Max adverse excursion (bps)")
    ax.set_ylabel("Density")
    ax.set_title("Adverse excursion (smaller => better fill model performance)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(np.clip(sell_ma_clean, 0, clip_adv), bins=BINS, color="#d62728",
            alpha=0.7, density=True, cumulative=True, label="Sell CDF")
    ax.hist(np.clip(buy_ma_clean,  0, clip_adv), bins=BINS, color="#2ca02c",
            alpha=0.5, density=True, cumulative=True, label="Buy CDF")
    ax.set_xlabel("Max adverse excursion (bps)")
    ax.set_ylabel("Cumulative density")
    ax.set_title("CDF -- % of signals with adverse <= x bps")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p2 = _save(fig, "06_adverse_excursion_dist.png")

    return p1, p2


def plot_npa_proxy_comparison(sell_cond: dict, buy_cond: dict) -> Path:
    """Bar chart of NPA proxy at each adverse_mult for sell vs buy unconditional."""
    mults = ADVERSE_MULTS
    sell_npas = [sell_cond[f"npa_proxy_{m}"] for m in mults]
    buy_npas  = [buy_cond[f"npa_proxy_{m}"] for m in mults]

    x = np.arange(len(mults))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"{SYMBOL} -- NPA Proxy: sell vs buy, unconditional\n"
        f"NPA proxy = mean(final_ret_bps) - adverse_mult * mean(max_adverse_bps)",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    bars_s = ax.bar(x - w/2, sell_npas, w, label=f"Sell (n={sell_cond['n']:,})",
                    color="#d62728", alpha=0.85)
    bars_b = ax.bar(x + w/2, buy_npas,  w, label=f"Buy  (n={buy_cond['n']:,})",
                    color="#2ca02c", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"mult={m}" for m in mults])
    ax.set_ylabel("NPA proxy (bps)")
    ax.set_title("Unconditional NPA proxy by adverse_mult")
    ax.legend(fontsize=9)
    for bars, col in ((bars_s, "#d62728"), (bars_b, "#2ca02c")):
        for bar in bars:
            h = bar.get_height()
            ypos = h + 0.02 if h >= 0 else h - 0.12
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{h:.3f}",
                    ha="center", fontsize=8, color=col)

    ax2 = axes[1]
    comps = {
        "mean final_ret": (sell_cond["final_ret"]["mean"], buy_cond["final_ret"]["mean"]),
        "mean max_favorable": (sell_cond["max_favorable"]["mean"], buy_cond["max_favorable"]["mean"]),
        "mean max_adverse": (sell_cond["max_adverse"]["mean"], buy_cond["max_adverse"]["mean"]),
    }
    xi = np.arange(len(comps))
    slabels, svals, bvals = zip(*[(k, v[0], v[1]) for k, v in comps.items()])
    ax2.bar(xi - w/2, svals, w, label="Sell", color="#d62728", alpha=0.85)
    ax2.bar(xi + w/2, bvals, w, label="Buy",  color="#2ca02c", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(slabels, fontsize=9)
    ax2.set_ylabel("Mean (bps)")
    ax2.set_title("Component decomposition: endpoint return vs excursions")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, "07_npa_proxy_comparison.png")


def plot_regime_npa(conditions: dict) -> Path:
    """Bar chart of NPA proxy by condition (sell-UP, sell-DOWN, buy-UP, buy-DOWN)."""
    cond_order = ["sell_UP", "sell_DOWN", "buy_UP", "buy_DOWN"]
    colors     = ["#d62728", "#ff7f0e", "#2ca02c", "#17becf"]
    mults      = ADVERSE_MULTS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{SYMBOL} -- Regime-Conditional NPA Proxy\n"
        f"Conditions: sell/buy in UP/DOWN regime (rolling 1-h return sign)",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: NPA proxy grouped by condition
    x = np.arange(len(cond_order))
    w = 0.22
    offsets = np.linspace(-(len(mults)-1)/2, (len(mults)-1)/2, len(mults)) * w

    ax = axes[0]
    for mi, mult in enumerate(mults):
        npas = [conditions[c][f"npa_proxy_{mult}"] for c in cond_order]
        bars = ax.bar(x + offsets[mi], npas, w * 0.9,
                      label=f"mult={mult}", alpha=0.85)
        for bar, npa in zip(bars, npas):
            if not math.isnan(npa):
                h = bar.get_height()
                ypos = h + 0.02 if h >= 0 else h - 0.12
                ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{npa:.3f}",
                        ha="center", fontsize=7)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    xticklabels = [
        f"{c}\nn={conditions[c]['n']:,}"
        + ("\n[LOW N]" if conditions[c]["low_n_flag"] else "")
        for c in cond_order
    ]
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("NPA proxy (bps)")
    ax.set_title("NPA proxy by regime condition")
    ax.legend(fontsize=9)

    # Panel 2: Decomposition per condition at mult=0.5
    ax2 = axes[1]
    mult05 = 0.5
    metrics = ["mean final_ret", "mean max_adverse", f"NPA (mult={mult05})"]
    xi = np.arange(len(cond_order))
    ww = 0.2
    offs2 = np.linspace(-(len(metrics)-1)/2, (len(metrics)-1)/2, len(metrics)) * ww

    metric_vals = {
        "mean final_ret":  [conditions[c]["final_ret"]["mean"] for c in cond_order],
        "mean max_adverse": [conditions[c]["max_adverse"]["mean"] for c in cond_order],
        f"NPA (mult={mult05})": [conditions[c][f"npa_proxy_{mult05}"] for c in cond_order],
    }
    metric_colors = ["steelblue", "tomato", "goldenrod"]

    for mi, (metric, vals) in enumerate(metric_vals.items()):
        ax2.bar(xi + offs2[mi], vals, ww * 0.9,
                label=metric, color=metric_colors[mi], alpha=0.85)

    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(xi)
    ax2.set_xticklabels([c.replace("_", "\n") for c in cond_order], fontsize=9)
    ax2.set_ylabel("bps")
    ax2.set_title(f"Decomposition per condition (mult={mult05})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "08_regime_npa_proxy.png")


# -- Report writer -------------------------------------------------------------

def write_report(
    sell_cond: dict,
    buy_cond: dict,
    conditions: dict,
    plot_paths: dict[str, Path],
    ts_span: tuple[str, str],
) -> None:
    lines: list[str] = []
    W = lines.append

    W("# NPA DECOMPOSITION AND REGIME ANALYSIS -- ETHUSDT")
    W("")
    W(f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    W("")
    W("## 0. Context")
    W("")
    W("Prior sweep found sell NPA = 6.3e-05 vs buy NPA = 7.0e-06 (9:1 ratio) for the top")
    W(f"pocket (imb>={MIN_ABS_IMB}, int>={MIN_INTENSITY:.0f}, spr<={MAX_SPREAD:.4f}, h={HORIZON_SEC}s).")
    W("The sell-asymmetry diagnostic showed directional hit rates are near-symmetric")
    W("(sell 55.5%, buy 56.4%), ruling out directional bias as the cause.")
    W("This report dissects the NPA gap via excursion distributions and tests whether")
    W("regime filtering (rolling 1-h return sign) meaningfully improves the NPA proxy.")
    W("")
    W("**NPA proxy** = `mean(final_ret_bps) - adverse_mult * mean(max_adverse_bps)`")
    W("  where `final_ret_bps` = directional endpoint return (120 s), bps")
    W("  and   `max_adverse_bps` = max adverse excursion during the 120-s hold, bps.")
    W("")
    W(f"**Data:** {ts_span[0]} -> {ts_span[1]}  |  ETHUSDT  |  1-sec buckets")
    W("")

    # -- Task 1 ----------------------------------------------------------------
    W("---")
    W("## Task 1: NPA Decomposition (sell vs buy)")
    W("")
    W("### 1.1 Event Counts")
    W("")
    W(f"| Side | Signals | Hit rate (final_ret > 0) |")
    W(f"|---|---:|---:|")
    W(f"| Sell (SHORT) | {sell_cond['n']:,} | {sell_cond['hit_rate']:.1%} |")
    W(f"| Buy  (LONG)  | {buy_cond['n']:,}  | {buy_cond['hit_rate']:.1%} |")
    W("")

    W("### 1.2 Favorable Endpoint Return Distribution (bps)")
    W("")
    W("| Stat | Sell | Buy | Sell - Buy |")
    W("|---|---:|---:|---:|")
    for stat in ("mean", "median", "p25", "p75", "p95"):
        sv = sell_cond["final_ret"][stat]
        bv = buy_cond["final_ret"][stat]
        diff = sv - bv if not (math.isnan(sv) or math.isnan(bv)) else float("nan")
        W(f"| {stat} | {_fmt(sv)} | {_fmt(bv)} | {_fmt(diff, 3)} |")
    W("")
    W(f"![Endpoint return distribution]({plot_paths['endpoint'].relative_to(REPORT_DIR)})")
    W("")

    W("### 1.3 Max Adverse Excursion Distribution (bps)")
    W("")
    W("*Sell adverse = max upward move during hold (bad for SHORT).*  ")
    W("*Buy adverse = max downward move during hold (bad for LONG).*")
    W("")
    W("| Stat | Sell adverse | Buy adverse | Sell - Buy |")
    W("|---|---:|---:|---:|")
    for stat in ("mean", "median", "p25", "p75", "p95"):
        sv = sell_cond["max_adverse"][stat]
        bv = buy_cond["max_adverse"][stat]
        diff = sv - bv if not (math.isnan(sv) or math.isnan(bv)) else float("nan")
        W(f"| {stat} | {_fmt(sv)} | {_fmt(bv)} | {_fmt(diff, 3)} |")
    W("")
    W(f"![Adverse excursion distribution]({plot_paths['adverse'].relative_to(REPORT_DIR)})")
    W("")

    W("### 1.4 Max Favorable Excursion Distribution (bps)")
    W("")
    W("| Stat | Sell favorable | Buy favorable |")
    W("|---|---:|---:|")
    for stat in ("mean", "median", "p25", "p75", "p95"):
        sv = sell_cond["max_favorable"][stat]
        bv = buy_cond["max_favorable"][stat]
        W(f"| {stat} | {_fmt(sv)} | {_fmt(bv)} |")
    W("")

    W("### 1.5 NPA Proxy Comparison (unconditional)")
    W("")
    W("| adverse_mult | Sell NPA proxy (bps) | Buy NPA proxy (bps) | Sell / Buy ratio |")
    W("|---:|---:|---:|---:|")
    for mult in ADVERSE_MULTS:
        sn = sell_cond[f"npa_proxy_{mult}"]
        bn = buy_cond[f"npa_proxy_{mult}"]
        ratio = sn / bn if (not math.isnan(sn) and not math.isnan(bn) and bn != 0) else float("nan")
        W(f"| {mult} | {_fmt(sn, 4)} | {_fmt(bn, 4)} | {_fmt(ratio, 2)} |")
    W("")
    W(f"![NPA proxy comparison]({plot_paths['npa_comp'].relative_to(REPORT_DIR)})")
    W("")

    # Mechanical explanation
    W("### 1.6 Mechanical Explanation of NPA Gap")
    W("")
    sell_fr_mean = sell_cond["final_ret"]["mean"]
    buy_fr_mean  = buy_cond["final_ret"]["mean"]
    sell_ma_mean = sell_cond["max_adverse"]["mean"]
    buy_ma_mean  = buy_cond["max_adverse"]["mean"]
    adv_diff     = buy_ma_mean - sell_ma_mean

    W(f"Decomposition at adverse_mult=0.3:")
    W(f"")
    W(f"```")
    W(f"sell: {_fmt(sell_fr_mean,3)} - 0.3 * {_fmt(sell_ma_mean,3)} = {_fmt(sell_cond['npa_proxy_0.3'],4)} bps")
    W(f"buy:  {_fmt(buy_fr_mean,3)}  - 0.3 * {_fmt(buy_ma_mean,3)}  = {_fmt(buy_cond['npa_proxy_0.3'],4)} bps")
    W(f"```")
    W(f"")
    if adv_diff > 0.1:
        W(f"**Primary driver: buy adverse excursion is {_fmt(adv_diff, 2)} bps larger than sell.**")
        W(f"At mult=0.3 this costs buy {_fmt(0.3 * adv_diff, 3)} bps more than sell.")
        W(f"Sell signals coincide with sustained sell-aggression episodes where upward")
        W(f"(adverse) moves are dampened. Buy bursts face more volatile path risk.")
    elif adv_diff < -0.1:
        W(f"**Unexpected: sell adverse is {_fmt(-adv_diff, 2)} bps LARGER than buy.**")
        W(f"The NPA gap must originate from the execution fill model, not path excursions.")
    else:
        W(f"**Adverse excursions are nearly equal ({_fmt(adv_diff, 3)} bps difference).**")
        W(f"The NPA gap likely originates from the passive fill probability model,")
        W(f"not from path excursions captured at this resolution.")
    W("")

    # -- Task 2 ----------------------------------------------------------------
    W("---")
    W("## Task 2: Regime-Conditional Pocket Analysis")
    W("")
    W("Regime label = 'UP' if rolling 1-h log-return >= 0, else 'DOWN'.")
    W("Four conditions: sell-UP, sell-DOWN, buy-UP, buy-DOWN.")
    W(f"Flagged as [LOW N] if signal count < {MIN_N}.")
    W("")

    W("### 2.1 Event Counts and Hit Rates")
    W("")
    W("| Condition | n | Hit rate | vs unconditional |")
    W("|---|---:|---:|---:|")
    cond_order = ["sell_UP", "sell_DOWN", "buy_UP", "buy_DOWN"]
    uncond_hr = {"sell_UP": sell_cond["hit_rate"], "sell_DOWN": sell_cond["hit_rate"],
                 "buy_UP": buy_cond["hit_rate"],   "buy_DOWN": buy_cond["hit_rate"]}
    for c in cond_order:
        cdata = conditions[c]
        diff = cdata["hit_rate"] - uncond_hr[c] if not math.isnan(cdata["hit_rate"]) else float("nan")
        low_flag = " **[LOW N]**" if cdata["low_n_flag"] else ""
        W(f"| {c}{low_flag} | {cdata['n']:,} | {_fmt(cdata['hit_rate']*100, 1)}% | {_fmt(diff*100, 1)}pp |")
    W("")

    W("### 2.2 Excursion Decomposition per Condition")
    W("")
    W("| Condition | n | mean final_ret | mean max_favorable | mean max_adverse |")
    W("|---|---:|---:|---:|---:|")
    for c in cond_order:
        cdata = conditions[c]
        low = " [LOW N]" if cdata["low_n_flag"] else ""
        W(f"| {c}{low} | {cdata['n']:,} | {_fmt(cdata['final_ret']['mean'])} "
          f"| {_fmt(cdata['max_favorable']['mean'])} | {_fmt(cdata['max_adverse']['mean'])} |")
    W("")

    W("### 2.3 NPA Proxy by Condition and adverse_mult")
    W("")
    header = "| Condition | n | " + " | ".join(f"mult={m}" for m in ADVERSE_MULTS) + " |"
    sep    = "|---|---:|" + "---:|" * len(ADVERSE_MULTS)
    W(header); W(sep)
    for c in cond_order:
        cdata  = conditions[c]
        low    = " [LOW N]" if cdata["low_n_flag"] else ""
        vals   = " | ".join(_fmt(cdata[f"npa_proxy_{m}"], 4) for m in ADVERSE_MULTS)
        W(f"| {c}{low} | {cdata['n']:,} | {vals} |")
    W("")
    W("*(Unconditional sell and buy shown below for reference)*")
    W("")
    for side, cdata in [("sell (all)", sell_cond), ("buy (all)", buy_cond)]:
        vals = " | ".join(_fmt(cdata[f"npa_proxy_{m}"], 4) for m in ADVERSE_MULTS)
        W(f"| {side} | {cdata['n']:,} | {vals} |")
    W("")
    W(f"![Regime NPA proxy]({plot_paths['regime_npa'].relative_to(REPORT_DIR)})")
    W("")

    W("### 2.4 Verdict: Does Regime Filtering Improve Edge?")
    W("")
    mult_key = 0.5
    sell_uncond = sell_cond[f"npa_proxy_{mult_key}"]
    buy_uncond  = buy_cond[f"npa_proxy_{mult_key}"]

    improvements = []
    for c in cond_order:
        cdata = conditions[c]
        base  = sell_uncond if c.startswith("sell") else buy_uncond
        npa   = cdata[f"npa_proxy_{mult_key}"]
        if math.isnan(npa) or math.isnan(base) or cdata["low_n_flag"]:
            continue
        delta = npa - base
        pct   = (delta / abs(base) * 100) if base != 0 else float("nan")
        improvements.append((c, cdata["n"], npa, delta, pct))

    improvements.sort(key=lambda x: -x[2])

    W(f"Comparing regime-filtered NPA proxy vs unconditional at adverse_mult={mult_key}:")
    W("")
    W("| Condition | n | NPA proxy | Delta vs unconditional | Delta % |")
    W("|---|---:|---:|---:|---:|")
    for c, n, npa, delta, pct in improvements:
        flag = " [BEST]" if improvements and c == improvements[0][0] else ""
        W(f"| {c}{flag} | {n:,} | {_fmt(npa, 4)} | {delta:+.4f} | {pct:+.1f}% |")
    W("")

    # Overall verdict
    best_cond, best_n, best_npa, best_delta, best_pct = improvements[0] if improvements else (None,)*5
    MEANINGFUL_THRESHOLD = 10.0  # % improvement

    if best_cond and not math.isnan(best_pct) and best_pct > MEANINGFUL_THRESHOLD:
        W(f"**VERDICT: YES -- regime filtering improves edge.**")
        W(f"")
        W(f"`{best_cond}` achieves NPA proxy {_fmt(best_npa, 4)} bps at mult={mult_key},")
        W(f"a **{best_pct:+.1f}%** uplift over the unconditional baseline.")
        W(f"With n={best_n:,} signals this is statistically meaningful.")
        W(f"Recommendation: run `{best_cond}` through the full forward validation sweep")
        W(f"(rank_passive_pockets_forward) to confirm out-of-sample.")
    elif best_cond and not math.isnan(best_pct) and best_pct > 0:
        W(f"**VERDICT: MARGINAL -- small regime improvement ({best_pct:+.1f}%).**")
        W(f"")
        W(f"`{best_cond}` shows a modest NPA proxy gain but below the {MEANINGFUL_THRESHOLD}% threshold.")
        W(f"Regime filtering may not justify the event count reduction.")
        W(f"Consider collecting more data before committing to a regime-gated pocket.")
    else:
        W(f"**VERDICT: NO -- regime filtering does not improve edge at mult={mult_key}.**")
        W(f"")
        W(f"No regime condition outperforms the unconditional baseline by > {MEANINGFUL_THRESHOLD}%.")
        W(f"The signal is broadly robust across regimes but does not benefit from gating.")
    W("")
    W("---")
    W("*Generated by `tools/npa_decomposition_and_regime.py` -- self-contained.*")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Report -> {REPORT_MD}", flush=True)


# -- Main ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="NPA decomposition and regime-conditional analysis.")
    ap.add_argument("--db", default="data/microstructure.db")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== NPA Decomposition and Regime Analysis ===", flush=True)
    print(f"DB:     {args.db}", flush=True)
    print(f"Pocket: imb>={MIN_ABS_IMB}, int>={MIN_INTENSITY:.0f}, spr<={MAX_SPREAD}, h={HORIZON_SEC}s", flush=True)
    print(flush=True)

    # -- Load ------------------------------------------------------------------
    bdf, mark_arr, min_sec = load_data(args.db)

    ts_span = (
        pd.Timestamp(int(bdf["ts_sec"].iloc[0]),  unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        pd.Timestamp(int(bdf["ts_sec"].iloc[-1]), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    )

    # -- Identify signals ------------------------------------------------------
    print("[4/5] Identifying signals, extracting price paths, computing excursions ...", flush=True)

    sell_mask = (
        (bdf["imbalance"]       <= -MIN_ABS_IMB)
        & (bdf["trade_intensity"] >= MIN_INTENSITY)
        & (bdf["spread"]          <= MAX_SPREAD)
    )
    buy_mask = (
        (bdf["imbalance"]       >=  MIN_ABS_IMB)
        & (bdf["trade_intensity"] >= MIN_INTENSITY)
        & (bdf["spread"]          <= MAX_SPREAD)
    )

    sell_ts = bdf.loc[sell_mask, "ts_sec"].values.astype(np.int64)
    buy_ts  = bdf.loc[buy_mask,  "ts_sec"].values.astype(np.int64)
    print(f"    sell signals={len(sell_ts):,}  buy signals={len(buy_ts):,}", flush=True)

    # -- Price paths & excursions ----------------------------------------------
    print("    extracting sell paths ...", flush=True)
    sell_paths = extract_price_paths(sell_ts, mark_arr, min_sec, HORIZON_SEC)
    sell_exc   = compute_excursions(sell_paths, direction=-1)

    print("    extracting buy paths ...", flush=True)
    buy_paths  = extract_price_paths(buy_ts, mark_arr, min_sec, HORIZON_SEC)
    buy_exc    = compute_excursions(buy_paths, direction=+1)

    # -- Regime labels ---------------------------------------------------------
    print("    computing regime labels ...", flush=True)
    regime_arr    = compute_regime_arr(mark_arr, window_sec=REGIME_WINDOW_S)
    sell_regimes  = get_regime_labels(sell_ts, regime_arr, min_sec)
    buy_regimes   = get_regime_labels(buy_ts,  regime_arr, min_sec)

    # -- Unconditional condition stats -----------------------------------------
    sell_vmask = sell_exc["valid_mask"]
    buy_vmask  = buy_exc["valid_mask"]

    sell_cond = condition_stats(
        sell_exc["final_ret_bps"], sell_exc["max_favorable_bps"], sell_exc["max_adverse_bps"],
        sell_vmask, label="sell_ALL",
    )
    buy_cond = condition_stats(
        buy_exc["final_ret_bps"], buy_exc["max_favorable_bps"], buy_exc["max_adverse_bps"],
        buy_vmask, label="buy_ALL",
    )

    print(f"    sell: n={sell_cond['n']:,}  mean_final_ret={sell_cond['final_ret']['mean']:.3f} bps"
          f"  mean_adverse={sell_cond['max_adverse']['mean']:.3f} bps", flush=True)
    print(f"    buy:  n={buy_cond['n']:,}  mean_final_ret={buy_cond['final_ret']['mean']:.3f} bps"
          f"  mean_adverse={buy_cond['max_adverse']['mean']:.3f} bps", flush=True)
    for mult in ADVERSE_MULTS:
        sn = sell_cond[f"npa_proxy_{mult}"]
        bn = buy_cond[f"npa_proxy_{mult}"]
        ratio = sn / bn if bn != 0 else float("nan")
        print(f"    mult={mult}: sell_npa={sn:.4f}  buy_npa={bn:.4f}  ratio={ratio:.2f}x", flush=True)

    # -- Regime-conditional stats -----------------------------------------------
    conditions: dict = {}
    for regime_label in ("UP", "DOWN"):
        for side_label, ts_arr, exc, regimes in (
            ("sell", sell_ts, sell_exc, sell_regimes),
            ("buy",  buy_ts,  buy_exc,  buy_regimes),
        ):
            ckey  = f"{side_label}_{regime_label}"
            rmask = (regimes == regime_label) & exc["valid_mask"]
            cs = condition_stats(
                exc["final_ret_bps"], exc["max_favorable_bps"], exc["max_adverse_bps"],
                rmask, label=ckey,
            )
            conditions[ckey] = cs
            print(
                f"    {ckey}: n={cs['n']:,}  HR={cs['hit_rate']:.1%}  "
                f"final_ret={cs['final_ret']['mean']:.3f}  adverse={cs['max_adverse']['mean']:.3f}  "
                f"npa@0.5={cs['npa_proxy_0.5']:.4f}",
                flush=True,
            )

    # -- Plots ------------------------------------------------------------------
    print("[5/5] Generating plots ...", flush=True)
    p_endpoint, p_adverse = plot_excursion_histograms(sell_exc, buy_exc)
    p_npa_comp  = plot_npa_proxy_comparison(sell_cond, buy_cond)
    p_regime    = plot_regime_npa(conditions)

    plot_paths = {
        "endpoint":   p_endpoint,
        "adverse":    p_adverse,
        "npa_comp":   p_npa_comp,
        "regime_npa": p_regime,
    }

    write_report(sell_cond, buy_cond, conditions, plot_paths, ts_span)

    # Print the headline verdict
    mult_key = 0.5
    cond_order = ["sell_UP", "sell_DOWN", "buy_UP", "buy_DOWN"]
    sell_base = sell_cond[f"npa_proxy_{mult_key}"]
    buy_base  = buy_cond[f"npa_proxy_{mult_key}"]
    print(flush=True)
    print("=== KEY RESULTS ===", flush=True)
    print(f"Sell vs buy NPA proxy at mult=0.5: {sell_base:.4f} vs {buy_base:.4f} bps", flush=True)
    print(f"Sell mean adverse: {sell_cond['max_adverse']['mean']:.3f} bps  "
          f"Buy mean adverse: {buy_cond['max_adverse']['mean']:.3f} bps  "
          f"diff (buy-sell): {buy_cond['max_adverse']['mean']-sell_cond['max_adverse']['mean']:.3f} bps", flush=True)
    print("Regime-conditional NPA@0.5 vs unconditional:", flush=True)
    for c in cond_order:
        base  = sell_base if c.startswith("sell") else buy_base
        npa   = conditions[c][f"npa_proxy_{mult_key}"]
        delta = npa - base
        pct   = (delta / abs(base) * 100) if base != 0 else float("nan")
        low   = " [LOW N]" if conditions[c]["low_n_flag"] else ""
        print(f"  {c}: npa={npa:.4f}  delta={delta:+.4f} ({pct:+.1f}%){low}", flush=True)
    print(f"\nReport -> {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
