#!/usr/bin/env python3
"""
tools/diagnose_sell_asymmetry.py
-----------------------------------------------------------------------------
Diagnostic: Is the sell-side passive pocket edge on ETHUSDT structural alpha
or a regime artifact from a sustained downtrend in the data window?

Self-contained -- no imports from tools/ modules.

Signal definitions mirror the existing micro_edge pipeline:
  * bucket_sec = 1  (1-second aggregation)
  * imbalance  = signed_qty / sum_abs_qty
      - is_buyer_maker=0 (taker buy)  -> +qty
      - is_buyer_maker=1 (taker sell) -> -qty
      - imbalance <  0 -> net sell aggression -> SHORT pocket signal
      - imbalance >  0 -> net buy aggression  -> LONG  pocket signal
  * trade_intensity = trade_count x 60  [trades-per-minute equivalent at 1-sec]
  * spread_proxy = |vwap - mark_price| / mark_price per bucket
      (approximation of the per-trade average used in build_bucket_features;
       slightly conservative but equivalent for filter purposes)
  * forward_return(120 s) = log(mark_price[t+120] / mark_price[t])

Top pocket filter (matches the sweep's best sell pocket):
  * abs(imbalance)   >= 0.50
  * trade_intensity  >= 3500
  * spread_proxy     <= 0.000300

Usage:
  python tools/diagnose_sell_asymmetry.py [--db data/microstructure.db]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Constants -----------------------------------------------------------------
SYMBOL          = "ETHUSDT"
HORIZON_SEC     = 120
BUCKET_SEC      = 1
MIN_ABS_IMB     = 0.50
MIN_INTENSITY   = 3500.0   # trades-per-minute equivalent at 1-sec buckets
MAX_SPREAD      = 0.000300
N_SUBPERIODS    = 4
REGIME_WINDOW_H = 1        # rolling window (hours) for up/down regime classification

REPORT_DIR = Path("reports")
PLOTS_DIR  = REPORT_DIR / "plots"
REPORT_MD  = REPORT_DIR / "SELL_ASYMMETRY_DIAGNOSTIC.md"


# -- Helpers -------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "N/A"

def _log_ret(future: pd.Series, current: pd.Series) -> pd.Series:
    return np.log(future / current)

def _hit_rate(signals: pd.Series, favorable: pd.Series) -> tuple[int, int, float]:
    """Return (n_signals, n_hits, hit_rate) where signals and favorable are bool series."""
    n = int(signals.sum())
    h = int((signals & favorable).sum())
    return n, h, h / n if n else float("nan")

def _edge_bps(signals: pd.Series, fwd_ret: pd.Series, direction: int) -> float:
    """Mean forward return (in bps, signed by direction) for signalled rows."""
    vals = fwd_ret[signals] * direction * 10_000.0
    return float(vals.mean()) if signals.any() else float("nan")


# -- Load & prepare data -------------------------------------------------------

def load_data(db: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        buckets_df  -- 1-second bucket DataFrame with features and forward returns
        marks_s     -- mark_price Series indexed by Unix second (int)
    """
    conn = sqlite3.connect(db)

    # -- 1. Mark prices --------------------------------------------------------
    print("[1/5] Loading mark prices ...", flush=True)
    marks_raw = pd.read_sql(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        conn, params=(SYMBOL,),
    )
    marks_raw["ts_sec"] = (marks_raw["ts_ms"] // 1000).astype(np.int64)
    marks_s = (marks_raw
               .drop_duplicates("ts_sec")
               .set_index("ts_sec")["mark_price"]
               .astype(np.float64))
    print(f"    mark rows={len(marks_s):,}  "
          f"span {pd.to_datetime(marks_s.index[0], unit='s', utc=True).strftime('%Y-%m-%d')} -> "
          f"{pd.to_datetime(marks_s.index[-1], unit='s', utc=True).strftime('%Y-%m-%d')}")

    # -- 2. Trade bucket aggregates (pure SQL -- no per-row join) ---------------
    print("[2/5] Aggregating trade buckets (SQL scan ~45 s) ...", flush=True)
    bucket_sql = """
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
    """
    bdf = pd.read_sql(bucket_sql, conn, params=(SYMBOL,))
    conn.close()
    print(f"    trade buckets={len(bdf):,}")

    # -- 3. Compute features ---------------------------------------------------
    print("[3/5] Computing features and forward returns ...", flush=True)

    bdf["ts_sec"]        = (bdf["bkt_ms"] // 1000).astype(np.int64)
    bdf["imbalance"]     = bdf["signed_qty"] / bdf["sum_abs_qty"]
    bdf["trade_intensity"] = bdf["trade_count"] * (60.0 / BUCKET_SEC)

    # Spread proxy: |VWAP - mark_price| / mark_price
    bdf["mark_price"]   = bdf["ts_sec"].map(marks_s)
    bdf["spread"]       = (bdf["vwap"] - bdf["mark_price"]).abs() / bdf["mark_price"]

    # Forward return: log(mark[t + HORIZON_SEC] / mark[t])
    bdf["fwd_ts_sec"]   = bdf["ts_sec"] + HORIZON_SEC
    bdf["mark_future"]  = bdf["fwd_ts_sec"].map(marks_s)
    bdf["fwd_ret"]      = _log_ret(bdf["mark_future"], bdf["mark_price"])

    # Drop rows missing mark or forward price
    bdf = bdf.dropna(subset=["mark_price", "fwd_ret", "spread"]).copy()
    print(f"    valid buckets after dropna={len(bdf):,}")

    # Datetime index (UTC-aware)
    bdf["dt"] = pd.to_datetime(bdf["ts_sec"], unit="s", utc=True)
    bdf = bdf.set_index("dt").sort_index()

    return bdf, marks_s


# -- Section 1: Regime characterisation ---------------------------------------

def characterise_regime(bdf: pd.DataFrame, marks_s: pd.Series) -> dict:
    print("[4/5] Characterising price regime ...", flush=True)

    # Build a 1-minute resampled mark price series for rolling stats
    marks_dt = (pd.to_datetime(marks_s.index, unit="s", utc=True)
                .to_series()
                .reset_index(drop=True))
    marks_series = pd.Series(
        marks_s.values,
        index=pd.to_datetime(marks_s.index, unit="s", utc=True),
        name="mark_price",
    )
    # Resample to 1-min, build a proper DataFrame for multi-column assignment
    mark_px_1min = marks_series.resample("1min").last().ffill()
    mark_1min = pd.DataFrame({"mark_price": mark_px_1min})

    t_start = mark_1min.index[0]
    t_end   = mark_1min.index[-1]
    duration_days = (t_end - t_start).total_seconds() / 86400

    p_start = float(mark_1min["mark_price"].iloc[0])
    p_end   = float(mark_1min["mark_price"].iloc[-1])
    total_drift_pct = (p_end - p_start) / p_start * 100.0
    total_drift_bps = (p_end - p_start) / p_start * 10_000.0

    # Rolling returns at 1h, 4h, 24h windows (in minutes)
    px = mark_1min["mark_price"]
    mark_1min["ret_1h"]  = np.log(px / px.shift(60))
    mark_1min["ret_4h"]  = np.log(px / px.shift(240))
    mark_1min["ret_24h"] = np.log(px / px.shift(1440))

    # Up/down regime on 1-h rolling return — map to each 1-second bucket
    # bdf index is UTC-aware datetimes; floor to 1-min to align with mark_1min
    regime_1h = mark_1min["ret_1h"].reindex(
        bdf.index.floor("1min"), method="ffill"
    )
    regime_1h.index = bdf.index
    bdf["regime_1h_ret"] = regime_1h.values
    bdf["regime"] = np.where(bdf["regime_1h_ret"] >= 0, "UP", "DOWN")

    return {
        "t_start": t_start,
        "t_end":   t_end,
        "duration_days": duration_days,
        "p_start": p_start,
        "p_end":   p_end,
        "total_drift_pct": total_drift_pct,
        "total_drift_bps": total_drift_bps,
        "mark_1min": mark_1min,
    }


# -- Section 2: Sub-period stability ------------------------------------------

def subperiod_analysis(bdf: pd.DataFrame) -> list[dict]:
    idx = bdf.index
    n   = len(bdf)
    step = n // N_SUBPERIODS
    results = []

    for i in range(N_SUBPERIODS):
        lo = i * step
        hi = (i + 1) * step if i < N_SUBPERIODS - 1 else n
        sub = bdf.iloc[lo:hi]

        t0 = sub.index[0].strftime("%m-%d %H:%M")
        t1 = sub.index[-1].strftime("%m-%d %H:%M")

        sell_sig = (sub["imbalance"] <= -MIN_ABS_IMB) & (sub["trade_intensity"] >= MIN_INTENSITY) & (sub["spread"] <= MAX_SPREAD)
        buy_sig  = (sub["imbalance"] >=  MIN_ABS_IMB) & (sub["trade_intensity"] >= MIN_INTENSITY) & (sub["spread"] <= MAX_SPREAD)

        sell_n, sell_h, sell_hr = _hit_rate(sell_sig, sub["fwd_ret"] < 0)
        buy_n,  buy_h,  buy_hr  = _hit_rate(buy_sig,  sub["fwd_ret"] > 0)

        sell_edge = _edge_bps(sell_sig, sub["fwd_ret"], direction=-1)   # SHORT direction
        buy_edge  = _edge_bps(buy_sig,  sub["fwd_ret"], direction=+1)   # LONG  direction

        # Baseline (unconditional direction frequency)
        down_frac = float((sub["fwd_ret"] < 0).mean())
        up_frac   = float((sub["fwd_ret"] > 0).mean())
        drift_bps = float(sub["fwd_ret"].mean() * 10_000)

        results.append({
            "period":     f"P{i+1}",
            "t_range":    f"{t0} -> {t1}",
            "n_buckets":  len(sub),
            "drift_bps":  drift_bps,
            "down_frac":  down_frac,
            "up_frac":    up_frac,
            "sell_n":     sell_n,
            "sell_h":     sell_h,
            "sell_hr":    sell_hr,
            "sell_edge":  sell_edge,
            "buy_n":      buy_n,
            "buy_h":      buy_h,
            "buy_hr":     buy_hr,
            "buy_edge":   buy_edge,
        })
        print(f"    {results[-1]['period']} {t0}->{t1}: "
              f"sell_n={sell_n} sell_hr={sell_hr:.1%} | "
              f"buy_n={buy_n}  buy_hr={buy_hr:.1%} | "
              f"drift={drift_bps:+.2f}bps/bucket", flush=True)

    return results


# -- Section 3: Regime-conditioned analysis ------------------------------------

def regime_analysis(bdf: pd.DataFrame) -> dict:
    results = {}
    for regime_label in ["UP", "DOWN", "ALL"]:
        sub = bdf if regime_label == "ALL" else bdf[bdf["regime"] == regime_label]
        if len(sub) == 0:
            continue

        sell_sig = (sub["imbalance"] <= -MIN_ABS_IMB) & (sub["trade_intensity"] >= MIN_INTENSITY) & (sub["spread"] <= MAX_SPREAD)
        buy_sig  = (sub["imbalance"] >=  MIN_ABS_IMB) & (sub["trade_intensity"] >= MIN_INTENSITY) & (sub["spread"] <= MAX_SPREAD)

        sell_n, sell_h, sell_hr = _hit_rate(sell_sig, sub["fwd_ret"] < 0)
        buy_n,  buy_h,  buy_hr  = _hit_rate(buy_sig,  sub["fwd_ret"] > 0)
        sell_edge = _edge_bps(sell_sig, sub["fwd_ret"], direction=-1)
        buy_edge  = _edge_bps(buy_sig,  sub["fwd_ret"], direction=+1)

        regime_frac = len(sub) / len(bdf)
        down_frac   = float((sub["fwd_ret"] < 0).mean())
        drift_bps   = float(sub["fwd_ret"].mean() * 10_000)

        results[regime_label] = {
            "regime":       regime_label,
            "n_buckets":    len(sub),
            "regime_frac":  regime_frac,
            "drift_bps":    drift_bps,
            "down_frac":    down_frac,
            "sell_n":       sell_n,
            "sell_h":       sell_h,
            "sell_hr":      sell_hr,
            "sell_edge":    sell_edge,
            "buy_n":        buy_n,
            "buy_h":        buy_h,
            "buy_hr":       buy_hr,
            "buy_edge":     buy_edge,
        }
        print(f"    regime={regime_label:4s} n={len(sub):,}  drift={drift_bps:+.3f}bps  "
              f"sell_hr={sell_hr:.1%}(n={sell_n})  buy_hr={buy_hr:.1%}(n={buy_n})", flush=True)

    return results


# -- Plots ---------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}", flush=True)
    return path


def plot_price_trajectory(mark_1min: pd.DataFrame, subperiod_rows: list[dict]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{SYMBOL} -- Price Trajectory & Rolling Returns", fontsize=13, fontweight="bold")

    price = mark_1min["mark_price"] if "mark_price" in mark_1min.columns else mark_1min.iloc[:, 0]

    # Panel 1: Price
    ax = axes[0]
    ax.plot(mark_1min.index, price, color="#1f77b4", lw=0.8, label="Mark price")
    for r in subperiod_rows:
        tlo, thi = r["t_range"].split(" -> ")
        ax.axvline(pd.Timestamp(f"2026-{tlo}", tz="UTC"), color="grey", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: 1-h and 4-h rolling log-returns
    ax = axes[1]
    ax.plot(mark_1min.index, mark_1min["ret_1h"]  * 100, color="steelblue",  lw=0.7, label="1h log-ret (%)", alpha=0.9)
    ax.plot(mark_1min.index, mark_1min["ret_4h"]  * 100, color="darkorange", lw=1.0, label="4h log-ret (%)", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.fill_between(mark_1min.index, mark_1min["ret_1h"] * 100, 0,
                    where=(mark_1min["ret_1h"] < 0), color="red",   alpha=0.15)
    ax.fill_between(mark_1min.index, mark_1min["ret_1h"] * 100, 0,
                    where=(mark_1min["ret_1h"] >= 0), color="green", alpha=0.15)
    ax.set_ylabel("Log-return (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: 24-h rolling log-return
    ax = axes[2]
    ax.plot(mark_1min.index, mark_1min["ret_24h"] * 100, color="purple", lw=1.0, label="24h log-ret (%)")
    ax.axhline(0, color="black", lw=0.5)
    ax.fill_between(mark_1min.index, mark_1min["ret_24h"] * 100, 0,
                    where=(mark_1min["ret_24h"] < 0), color="red",   alpha=0.18)
    ax.fill_between(mark_1min.index, mark_1min["ret_24h"] * 100, 0,
                    where=(mark_1min["ret_24h"] >= 0), color="green", alpha=0.18)
    ax.set_ylabel("24h Log-return (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return _save(fig, "01_price_trajectory.png")


def plot_subperiod_hit_rates(subperiod_rows: list[dict]) -> Path:
    labels = [r["period"] for r in subperiod_rows]
    sell_hr = [r["sell_hr"] if not math.isnan(r["sell_hr"]) else 0 for r in subperiod_rows]
    buy_hr  = [r["buy_hr"]  if not math.isnan(r["buy_hr"])  else 0 for r in subperiod_rows]
    down_fr = [r["down_frac"] for r in subperiod_rows]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_s = ax.bar(x - w,   sell_hr, w, label="Sell hit-rate (SHORT)", color="#d62728", alpha=0.85)
    bars_b = ax.bar(x,       buy_hr,  w, label="Buy hit-rate (LONG)",   color="#2ca02c", alpha=0.85)
    bars_d = ax.bar(x + w,   down_fr, w, label="Unconditional down-frac", color="#7f7f7f", alpha=0.60)

    ax.set_xticks(x)
    xticklabels = [f"{r['period']}\n{r['t_range']}" for r in subperiod_rows]
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="black", lw=0.6, ls="--", label="50% baseline")
    ax.set_title(f"{SYMBOL} -- Sub-period Signal Hit Rates vs Unconditional Down Fraction\n"
                 f"(sell: imb<=-{MIN_ABS_IMB}, int>={MIN_INTENSITY:.0f}, spr<={MAX_SPREAD:.4f}, h={HORIZON_SEC}s)", fontsize=10)
    ax.legend(fontsize=8)

    for bars in (bars_s, bars_b, bars_d):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    fig.tight_layout()
    return _save(fig, "02_subperiod_hit_rates.png")


def plot_regime_breakdown(regime_rows: dict) -> Path:
    order  = ["UP", "DOWN", "ALL"]
    labels = [r for r in order if r in regime_rows]
    sell_hr = [regime_rows[r]["sell_hr"] if not math.isnan(regime_rows[r]["sell_hr"]) else 0 for r in labels]
    buy_hr  = [regime_rows[r]["buy_hr"]  if not math.isnan(regime_rows[r]["buy_hr"])  else 0 for r in labels]
    down_fr = [regime_rows[r]["down_frac"] for r in labels]

    x = np.arange(len(labels))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{SYMBOL} -- Regime-Conditioned Hit Rates (1h rolling return sign)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.bar(x - w,   sell_hr, w, label="Sell hit-rate (SHORT)", color="#d62728", alpha=0.85)
    ax.bar(x,       buy_hr,  w, label="Buy hit-rate (LONG)",   color="#2ca02c", alpha=0.85)
    ax.bar(x + w,   down_fr, w, label="Unconditional down-frac", color="#7f7f7f", alpha=0.60)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.axhline(0.5, color="black", lw=0.6, ls="--")
    ax.set_ylabel("Rate"); ax.set_title("Hit rates by regime"); ax.legend(fontsize=8)
    for i, (s, b, d) in enumerate(zip(sell_hr, buy_hr, down_fr)):
        ax.text(i - w, s + 0.02, f"{s:.1%}", ha="center", fontsize=7, color="#d62728")
        ax.text(i,     b + 0.02, f"{b:.1%}", ha="center", fontsize=7, color="#2ca02c")
        ax.text(i + w, d + 0.02, f"{d:.1%}", ha="center", fontsize=7, color="#7f7f7f")

    # Panel 2: mean edge in bps
    ax2 = axes[1]
    sell_edge = [regime_rows[r]["sell_edge"] if not math.isnan(regime_rows[r]["sell_edge"]) else 0 for r in labels]
    buy_edge  = [regime_rows[r]["buy_edge"]  if not math.isnan(regime_rows[r]["buy_edge"])  else 0 for r in labels]
    ax2.bar(x - w / 2, sell_edge, w, label="Sell mean fwd-edge (bps)", color="#d62728", alpha=0.85)
    ax2.bar(x + w / 2, buy_edge,  w, label="Buy  mean fwd-edge (bps)", color="#2ca02c", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean forward edge (bps, post-direction)")
    ax2.set_title("Mean directional forward return by regime"); ax2.legend(fontsize=8)
    for i, (s, b) in enumerate(zip(sell_edge, buy_edge)):
        ax2.text(i - w / 2, s + 0.002, f"{s:.2f}", ha="center", fontsize=7, color="#d62728")
        ax2.text(i + w / 2, b + 0.002, f"{b:.2f}", ha="center", fontsize=7, color="#2ca02c")

    fig.tight_layout()
    return _save(fig, "03_regime_breakdown.png")


def plot_signal_distribution(bdf: pd.DataFrame) -> Path:
    """Scatter: imbalance vs log(trade_intensity), coloured by forward return sign."""
    # Sample up to 20k points for readability
    sample = bdf.sample(min(20_000, len(bdf)), random_state=42)

    colors = np.where(sample["fwd_ret"] < 0, "red", "green")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{SYMBOL} -- Signal Feature Space (sample 20k buckets)", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.scatter(sample["imbalance"], np.log10(sample["trade_intensity"] + 1),
               c=colors, alpha=0.15, s=4, linewidths=0)
    ax.axvline(-MIN_ABS_IMB, color="red",   lw=1.2, ls="--", label=f"sell threshold (imb<=-{MIN_ABS_IMB})")
    ax.axvline(+MIN_ABS_IMB, color="green", lw=1.2, ls="--", label=f"buy  threshold (imb>=+{MIN_ABS_IMB})")
    ax.axhline(math.log10(MIN_INTENSITY + 1), color="grey", lw=1.0, ls=":", label=f"intensity>={MIN_INTENSITY:.0f}")
    ax.set_xlabel("Imbalance"); ax.set_ylabel("log10(trade_intensity + 1)")
    ax.set_title("Imbalance vs Intensity\n(red=pricev, green=price^ in 120 s)")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    # Imbalance distribution: sell-pocket vs buy-pocket
    sell_mask = (bdf["imbalance"] <= -MIN_ABS_IMB) & (bdf["trade_intensity"] >= MIN_INTENSITY) & (bdf["spread"] <= MAX_SPREAD)
    buy_mask  = (bdf["imbalance"] >=  MIN_ABS_IMB) & (bdf["trade_intensity"] >= MIN_INTENSITY) & (bdf["spread"] <= MAX_SPREAD)

    ax2.hist(bdf.loc[sell_mask, "fwd_ret"] * 10_000, bins=60, color="#d62728",
             alpha=0.6, label=f"Sell pocket fwd-ret (n={sell_mask.sum():,})", density=True)
    ax2.hist(bdf.loc[buy_mask,  "fwd_ret"] * 10_000, bins=60, color="#2ca02c",
             alpha=0.6, label=f"Buy  pocket fwd-ret (n={buy_mask.sum():,})", density=True)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("120-s forward return (bps)"); ax2.set_ylabel("Density")
    ax2.set_title("Forward-return distribution -- top pocket signals\n(sell pocket = SHORT direction, buy pocket = LONG direction)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "04_signal_feature_space.png")


# -- Verdict -------------------------------------------------------------------

def compute_verdict(regime_analysis_rows: dict, subperiod_rows: list[dict],
                    total_drift_pct: float) -> dict:
    """
    Derive a verdict from the evidence:
    1. If sell hit-rate in UP regime >= sell hit-rate in DOWN regime - 0.03:
       sell edge is robust to regime -> structural
    2. If sell hit-rate in DOWN regime >> sell hit-rate in UP regime (diff > 0.05)
       AND baseline down-frac in DOWN regime explains most of the hit-rate:
       -> trend-following artifact
    3. If sell edge is consistent across sub-periods (std of sell_hr < 0.05):
       -> suggests structural rather than one-period artifact
    """
    up    = regime_analysis_rows.get("UP",   {})
    down  = regime_analysis_rows.get("DOWN", {})
    all_  = regime_analysis_rows.get("ALL",  {})

    sell_hr_up   = up.get("sell_hr",   float("nan"))
    sell_hr_down = down.get("sell_hr", float("nan"))
    sell_hr_all  = all_.get("sell_hr", float("nan"))
    buy_hr_all   = all_.get("buy_hr",  float("nan"))

    sell_n_up   = up.get("sell_n",   0)
    sell_n_down = down.get("sell_n", 0)

    down_frac_up   = up.get("down_frac",   0.5)
    down_frac_down = down.get("down_frac", 0.5)

    # Sub-period consistency
    valid_hrs = [r["sell_hr"] for r in subperiod_rows if not math.isnan(r["sell_hr"])]
    sell_hr_std   = float(np.std(valid_hrs)) if len(valid_hrs) > 1 else float("nan")
    sell_hr_range = max(valid_hrs) - min(valid_hrs) if valid_hrs else float("nan")

    # Trend contamination: how much of sell hit-rate is explained by unconditional down-bias?
    # If sell_hr ~= down_frac (unconditional) -> signal has no extra value
    edge_over_baseline_all  = sell_hr_all  - all_.get("down_frac",  0.5)
    edge_over_baseline_down = sell_hr_down - down_frac_down
    edge_over_baseline_up   = sell_hr_up   - down_frac_up

    # Regime uplift: does the sell edge appear even in UP regimes?
    regime_diff = (sell_hr_down - sell_hr_up) if not (math.isnan(sell_hr_down) or math.isnan(sell_hr_up)) else float("nan")

    flags = []

    # Flag 1: Is the data window even bearish?
    if total_drift_pct < -2.0:
        flags.append(("BEARISH_WINDOW", f"Total price drift = {total_drift_pct:+.1f}% -- data window has downward bias."))
    elif total_drift_pct > 2.0:
        flags.append(("BULLISH_WINDOW", f"Total price drift = {total_drift_pct:+.1f}% -- data window has upward bias."))
    else:
        flags.append(("NEUTRAL_WINDOW", f"Total price drift = {total_drift_pct:+.1f}% -- roughly balanced window."))

    # Flag 2: Regime uplift on sell side
    if not math.isnan(regime_diff):
        if regime_diff > 0.06:
            flags.append(("REGIME_DEPENDENT_SELL",
                          f"Sell hit-rate is {regime_diff:.1%} higher in DOWN regime vs UP regime. "
                          "Strong regime dependency -- possible trend artifact."))
        elif regime_diff < 0.02:
            flags.append(("REGIME_ROBUST_SELL",
                          f"Sell hit-rate gap between DOWN vs UP regime = {regime_diff:.1%}. "
                          "Signal is relatively regime-robust."))
        else:
            flags.append(("WEAK_REGIME_DEPENDENCY",
                          f"Sell hit-rate gap DOWN vs UP = {regime_diff:.1%}. Moderate dependency."))

    # Flag 3: Edge over baseline
    if not math.isnan(edge_over_baseline_all):
        if edge_over_baseline_all < 0.01:
            flags.append(("NO_EDGE_OVER_BASELINE",
                          f"Sell hit-rate ({sell_hr_all:.1%}) barely exceeds unconditional down-fraction "
                          f"({all_.get('down_frac', 0):.1%}). Net alpha ~= {edge_over_baseline_all:.2%}."))
        else:
            flags.append(("EDGE_OVER_BASELINE",
                          f"Sell signal adds {edge_over_baseline_all:.2%} above the unconditional down-fraction "
                          f"({all_.get('down_frac', 0):.1%} -> {sell_hr_all:.1%})."))

    # Flag 4: Sub-period consistency
    if not math.isnan(sell_hr_std):
        if sell_hr_std > 0.05:
            flags.append(("INCONSISTENT_SUBPERIODS",
                          f"Sell hit-rate std across sub-periods = {sell_hr_std:.3f} -- "
                          "high variance, signal may be period-specific."))
        else:
            flags.append(("CONSISTENT_SUBPERIODS",
                          f"Sell hit-rate std across sub-periods = {sell_hr_std:.3f} -- "
                          "reasonably stable across time."))

    # Final verdict
    has_regime_dep  = any(f[0] == "REGIME_DEPENDENT_SELL"   for f in flags)
    no_edge_base    = any(f[0] == "NO_EDGE_OVER_BASELINE"    for f in flags)
    inconsistent    = any(f[0] == "INCONSISTENT_SUBPERIODS"  for f in flags)
    has_edge_base   = any(f[0] == "EDGE_OVER_BASELINE"       for f in flags)
    regime_robust   = any(f[0] == "REGIME_ROBUST_SELL"       for f in flags)
    consistent      = any(f[0] == "CONSISTENT_SUBPERIODS"    for f in flags)

    if no_edge_base or has_regime_dep:
        verdict = "LIKELY_ARTIFACT"
        summary = (
            "The sell-side edge is likely a **trend-following artifact**. "
            "The hit-rate advantage is largely explained by the regime "
            "direction (down-biased data window), not by genuine microstructure "
            "alpha. Treat with caution before allocating capital."
        )
    elif regime_robust and consistent and has_edge_base:
        verdict = "LIKELY_STRUCTURAL"
        summary = (
            "The sell-side edge appears **structurally robust**. "
            "It persists across sub-periods and is not confined to DOWN regimes, "
            "suggesting genuine microstructure alpha above the directional baseline. "
            "However, always validate on fresh data before going live."
        )
    else:
        verdict = "UNCERTAIN"
        summary = (
            "The evidence is **mixed**. Some sub-periods show consistent sell edge "
            "but regime dependency is non-trivial. Collect more data across diverse "
            "regimes before drawing conclusions."
        )

    return {
        "verdict":                   verdict,
        "summary":                   summary,
        "flags":                     flags,
        "sell_hr_all":               sell_hr_all,
        "buy_hr_all":                buy_hr_all,
        "sell_hr_up":                sell_hr_up,
        "sell_hr_down":              sell_hr_down,
        "regime_diff":               regime_diff,
        "edge_over_baseline_all":    edge_over_baseline_all,
        "edge_over_baseline_down":   edge_over_baseline_down,
        "edge_over_baseline_up":     edge_over_baseline_up,
        "sell_hr_std":               sell_hr_std,
        "sell_hr_range":             sell_hr_range,
        "sell_n_up":                 sell_n_up,
        "sell_n_down":               sell_n_down,
    }


# -- Report writer -------------------------------------------------------------

def write_report(
    regime_info: dict,
    subperiod_rows: list[dict],
    regime_rows: dict,
    verdict: dict,
    plot_paths: dict[str, Path],
) -> None:
    ri = regime_info
    vd = verdict

    lines: list[str] = []
    W = lines.append  # shorthand

    W("# SELL ASYMMETRY DIAGNOSTIC -- ETHUSDT")
    W("")
    W(f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    W("")
    W("## Overview")
    W("")
    W("Investigates whether the sell-side passive pocket edge on ETHUSDT is genuine "
      "microstructure alpha or a directional artifact from a bearish data window.")
    W("")
    W("**Pocket filter (top pocket from sweep):**")
    W(f"- `abs(imbalance) >= {MIN_ABS_IMB}`")
    W(f"- `trade_intensity >= {MIN_INTENSITY:.0f}`  (trades/min equivalent at 1-sec buckets)")
    W(f"- `spread_proxy <= {MAX_SPREAD:.6f}`  (VWAP deviation from mark price)")
    W(f"- Horizon: {HORIZON_SEC} s")
    W(f"- Rule: `micro_edge_v3_passive_alpha` direction logic -- sell when imbalance <= -{MIN_ABS_IMB}")
    W("")
    W("> **Spread note:** The original `build_bucket_features` computes spread as the per-trade "
      "average of `|price - mark_price| / mark_price`. This diagnostic uses "
      "`|VWAP - mark_price| / mark_price` as a bucket-level approximation. "
      "For high-liquidity ETH futures this is a close proxy; the threshold is loose enough "
      "that conclusions are not sensitive to the approximation method.")
    W("")

    # -- 1. Data coverage ------------------------------------------------------
    W("---")
    W("## 1. Data Coverage & Price Regime")
    W("")
    W(f"| Item | Value |")
    W(f"|---|---|")
    W(f"| Symbol | {SYMBOL} |")
    W(f"| Data window | {ri['t_start'].strftime('%Y-%m-%d %H:%M UTC')} -> {ri['t_end'].strftime('%Y-%m-%d %H:%M UTC')} |")
    W(f"| Duration | {ri['duration_days']:.1f} days |")
    W(f"| Start price | ${ri['p_start']:,.2f} |")
    W(f"| End price | ${ri['p_end']:,.2f} |")
    W(f"| **Total drift** | **{ri['total_drift_pct']:+.2f}% ({ri['total_drift_bps']:+.0f} bps)** |")
    W("")
    W(f"![Price trajectory]({plot_paths['price'].relative_to(REPORT_DIR)})")
    W("")

    # -- 2. Sub-period stability -----------------------------------------------
    W("---")
    W("## 2. Sub-period Stability Test")
    W("")
    W("The data window is split into 4 equal sub-periods. For each, the raw signal "
      "hit rates are computed independently to detect whether the asymmetry is "
      "concentrated in one regime slice or consistent throughout.")
    W("")
    W("| Period | Date Range | Drift (bps/bkt) | v Baseline | Sell n | Sell HR | Buy n | Buy HR | Sell - Buy |")
    W("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in subperiod_rows:
        sell_hr_str = f"{r['sell_hr']:.1%}" if not math.isnan(r["sell_hr"]) else "N/A"
        buy_hr_str  = f"{r['buy_hr']:.1%}"  if not math.isnan(r["buy_hr"])  else "N/A"
        diff = (r["sell_hr"] - r["buy_hr"]) if not (math.isnan(r["sell_hr"]) or math.isnan(r["buy_hr"])) else float("nan")
        diff_str = f"{diff:+.1%}" if not math.isnan(diff) else "N/A"
        W(f"| {r['period']} | {r['t_range']} | {r['drift_bps']:+.3f} | {r['down_frac']:.1%} "
          f"| {r['sell_n']} | {sell_hr_str} | {r['buy_n']} | {buy_hr_str} | {diff_str} |")
    W("")
    W(f"![Sub-period hit rates]({plot_paths['subperiod'].relative_to(REPORT_DIR)})")
    W("")
    W("**Sell HR** = fraction of sell-pocket signals where price falls within 120 s (favorable for SHORT).  ")
    W("**v Baseline** = unconditional fraction of 120-s windows that are down in that sub-period.  ")
    W("A sell HR close to the baseline suggests no microstructure alpha beyond trend following.")
    W("")

    # -- 3. Regime conditioning ------------------------------------------------
    W("---")
    W("## 3. Regime-Conditioned Analysis")
    W("")
    W("Each 1-second bucket is labelled **UP** or **DOWN** based on the rolling 1-hour "
      f"log-return of the mark price (UP if ret >= 0, DOWN otherwise). "
      "Signal performance is then measured separately within each regime label.")
    W("")
    W("| Regime | Buckets | % of time | Drift (bps/bkt) | v Baseline | Sell n | Sell HR | Sell edge (bps) | Buy n | Buy HR | Buy edge (bps) |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lbl in ["UP", "DOWN", "ALL"]:
        r = regime_rows.get(lbl, {})
        if not r:
            continue
        sell_hr_s = f"{r['sell_hr']:.1%}" if not math.isnan(r["sell_hr"]) else "N/A"
        buy_hr_s  = f"{r['buy_hr']:.1%}"  if not math.isnan(r["buy_hr"])  else "N/A"
        se = f"{r['sell_edge']:.3f}" if not math.isnan(r["sell_edge"]) else "N/A"
        be = f"{r['buy_edge']:.3f}"  if not math.isnan(r["buy_edge"])  else "N/A"
        W(f"| **{lbl}** | {r['n_buckets']:,} | {r['regime_frac']:.1%} | {r['drift_bps']:+.4f} "
          f"| {r['down_frac']:.1%} | {r['sell_n']} | {sell_hr_s} | {se} | {r['buy_n']} | {buy_hr_s} | {be} |")
    W("")
    W(f"![Regime breakdown]({plot_paths['regime'].relative_to(REPORT_DIR)})")
    W("")
    W(f"![Signal feature space]({plot_paths['features'].relative_to(REPORT_DIR)})")
    W("")

    # -- Key statistics --------------------------------------------------------
    W("---")
    W("## 4. Key Diagnostic Statistics")
    W("")
    W("| Metric | Value |")
    W("|---|---|")
    fmt_pct = lambda v: f"{v:.2%}" if not math.isnan(v) else "N/A"
    fmt_f   = lambda v: f"{v:.4f}" if not math.isnan(v) else "N/A"
    W(f"| Sell hit-rate (all regimes) | {fmt_pct(vd['sell_hr_all'])} |")
    W(f"| Buy  hit-rate (all regimes) | {fmt_pct(vd['buy_hr_all'])} |")
    W(f"| Sell HR in UP regime        | {fmt_pct(vd['sell_hr_up'])} |")
    W(f"| Sell HR in DOWN regime      | {fmt_pct(vd['sell_hr_down'])} |")
    W(f"| DOWN - UP sell HR gap       | {fmt_pct(vd['regime_diff'])} |")
    W(f"| Sell edge above baseline (all) | {fmt_pct(vd['edge_over_baseline_all'])} |")
    W(f"| Sell edge above baseline (UP)  | {fmt_pct(vd['edge_over_baseline_up'])} |")
    W(f"| Sell edge above baseline (DOWN) | {fmt_pct(vd['edge_over_baseline_down'])} |")
    W(f"| Sell HR std (across sub-periods) | {fmt_f(vd['sell_hr_std'])} |")
    W(f"| Sell HR range (max-min)          | {fmt_pct(vd['sell_hr_range'])} |")
    W(f"| Sell signals in UP regime   | {vd['sell_n_up']} |")
    W(f"| Sell signals in DOWN regime | {vd['sell_n_down']} |")
    W("")

    # -- Verdict ---------------------------------------------------------------
    W("---")
    W("## 5. Verdict")
    W("")
    verdict_emoji = {
        "LIKELY_ARTIFACT":   "!",
        "LIKELY_STRUCTURAL": "[OK]",
        "UNCERTAIN":         "[?]",
    }.get(vd["verdict"], "")
    W(f"### {verdict_emoji} {vd['verdict']}")
    W("")
    W(vd["summary"])
    W("")
    W("### Evidence flags")
    W("")
    for flag, desc in vd["flags"]:
        W(f"- **{flag}**: {desc}")
    W("")
    W("### Interpretation guide")
    W("")
    W("- If sell HR >> buy HR across **all regimes** (UP and DOWN) and consistently across "
      "sub-periods -> the asymmetry is likely structural microstructure alpha.")
    W("- If sell HR is elevated **only in DOWN regime** and approximately equals the "
      "unconditional down-fraction -> the signal is merely a proxy for trend, not edge.")
    W("- If the sell HR gap (DOWN - UP) > 5 pp the signal has non-trivial regime sensitivity. "
      "Consider regime-filtering or collecting data across a bull run to confirm robustness.")
    W("")
    W("---")
    W("*Generated by `tools/diagnose_sell_asymmetry.py` -- self-contained, "
      "no dependency on tools/ modules.*")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Report written to {REPORT_MD}")


# -- Main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose sell-vs-buy asymmetry in passive pocket signals.")
    ap.add_argument("--db", default="data/microstructure.db", help="Path to microstructure SQLite DB")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Sell Asymmetry Diagnostic -- {SYMBOL} ===", flush=True)
    print(f"DB:     {args.db}")
    print(f"Pocket: imb>={MIN_ABS_IMB}, int>={MIN_INTENSITY:.0f}, spr<={MAX_SPREAD}, h={HORIZON_SEC}s")
    print()

    # Load + compute
    bdf, marks_s = load_data(args.db)

    # Regime characterisation
    print("[4/5] Characterising regime and conditioning ...", flush=True)
    regime_info = characterise_regime(bdf, marks_s)
    print(f"    {regime_info['t_start'].strftime('%Y-%m-%d')} -> {regime_info['t_end'].strftime('%Y-%m-%d')}  "
          f"duration={regime_info['duration_days']:.1f}d  "
          f"drift={regime_info['total_drift_pct']:+.2f}% ({regime_info['total_drift_bps']:+.0f} bps)")

    # Sub-period and regime analyses
    subperiod_rows = subperiod_analysis(bdf)
    regime_rows    = regime_analysis(bdf)

    # Plots
    print("[5/5] Generating plots ...", flush=True)
    plot_paths = {
        "price":     plot_price_trajectory(regime_info["mark_1min"], subperiod_rows),
        "subperiod": plot_subperiod_hit_rates(subperiod_rows),
        "regime":    plot_regime_breakdown(regime_rows),
        "features":  plot_signal_distribution(bdf),
    }

    # Verdict + report
    verdict = compute_verdict(regime_rows, subperiod_rows, regime_info["total_drift_pct"])
    print(f"\n>>> VERDICT: {verdict['verdict']}")
    print(f"    {verdict['summary'][:120]}..." if len(verdict["summary"]) > 120 else f"    {verdict['summary']}")
    for flag, desc in verdict["flags"]:
        print(f"    [{flag}] {desc[:90]}{'...' if len(desc) > 90 else ''}")

    write_report(regime_info, subperiod_rows, regime_rows, verdict, plot_paths)
    print(f"\nDone. Report -> {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
