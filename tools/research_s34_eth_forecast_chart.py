# encoding: utf-8
"""
S34 ETH 1-Month "Prediction" Chart
For entertainment purposes only.

Shows:
  - Last 30 days ETH real price
  - Our live BUY/SELL trade entries overlaid
  - 30-day forward Monte Carlo fan based on our alpha statistics
  - Key stats from live trades
"""
from __future__ import annotations
import sqlite3
import json
import random
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path("D:/eclipse_scalper")
MICRO_DB = ROOT / "data" / "microstructure.db"
LEDGER_DB = ROOT / "data" / "s34_intelligence.db"
OUT = ROOT / "reports" / "research" / "s34" / "S34_ETH_FORECAST_2026-06.png"

random.seed(42)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    micro  = sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True)
    ledger = sqlite3.connect(f"file:{LEDGER_DB}?mode=ro", uri=True)

    # ── 1. ETH price last 30 days ─────────────────────────────────────────────
    now_ms    = int(micro.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0])
    start_ms  = now_ms - 30 * 24 * 3600 * 1000

    price_rows = micro.execute(
        """SELECT CAST(ts_ms/300000 AS INTEGER)*300000 AS bucket, AVG(mark_price)
           FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
           GROUP BY bucket ORDER BY bucket""",
        (start_ms, now_ms)
    ).fetchall()

    hist_ts    = [r[0] / 1000 for r in price_rows]
    hist_price = [r[1] for r in price_rows]
    current    = hist_price[-1]

    # ── 2. Live trade entries (ETH rules) ─────────────────────────────────────
    trades = ledger.execute(
        """SELECT rule_name, entry_ts_ms, exit_ts_ms, net_bps, exit_reason
           FROM s34_trades
           WHERE rule_name LIKE '%ETH%' AND entry_ts_ms >= ?
           ORDER BY entry_ts_ms""",
        (start_ms,)
    ).fetchall()

    buy_entries  = [(r[1]/1000, r[3], r[4]) for r in trades if 'BUY' in r[0]]
    sell_entries = [(r[1]/1000, r[3], r[4]) for r in trades if 'SELL' in r[0]]

    # Mark prices at trade entries
    def price_at(ts_ms):
        r = micro.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (int(ts_ms),)
        ).fetchone()
        return r[0] if r else None

    # ── 3. Alpha statistics for projection ────────────────────────────────────
    all_eth = ledger.execute(
        "SELECT net_bps FROM s34_trades WHERE rule_name LIKE '%ETH%BUY%' AND status='CLOSED'"
    ).fetchall()
    nets = [r[0] for r in all_eth if r[0] is not None]

    wr          = sum(1 for n in nets if n > 0) / len(nets) if nets else 0.7
    mean_bps    = sum(nets) / len(nets) if nets else 35.0
    trades_day  = 1.5  # avg ETH trades per day
    daily_edge  = mean_bps * trades_day / 10000  # bps → fraction

    # Price vol from last 30 days
    log_rets = [math.log(hist_price[i]/hist_price[i-1]) for i in range(1, len(hist_price))]
    vol_5m   = (sum(r**2 for r in log_rets) / len(log_rets)) ** 0.5
    vol_day  = vol_5m * math.sqrt(288)  # 5-min bars per day

    # ── 4. Monte Carlo forward paths ──────────────────────────────────────────
    N_PATHS    = 300
    N_DAYS     = 30
    N_STEPS    = N_DAYS * 288   # 5-min steps

    dt         = 5 / (60 * 24)  # 5 min in days
    drift_step = daily_edge * dt
    vol_step   = vol_day * math.sqrt(dt)

    paths = []
    for _ in range(N_PATHS):
        p = current
        path = [p]
        for _ in range(N_STEPS):
            p *= math.exp(drift_step - 0.5 * vol_step**2 + vol_step * random.gauss(0, 1))
            path.append(p)
        paths.append(path)

    # Time axis for forward
    step_sec    = 5 * 60
    fwd_ts      = [now_ms/1000 + i * step_sec for i in range(N_STEPS + 1)]

    paths_arr   = np.array(paths)
    p5  = np.percentile(paths_arr, 5,  axis=0)
    p25 = np.percentile(paths_arr, 25, axis=0)
    p50 = np.percentile(paths_arr, 50, axis=0)
    p75 = np.percentile(paths_arr, 75, axis=0)
    p95 = np.percentile(paths_arr, 95, axis=0)

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   facecolor="#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    def ts2dt(ts): return datetime.fromtimestamp(ts, tz=timezone.utc)

    hist_dt = [ts2dt(t) for t in hist_ts]
    fwd_dt  = [ts2dt(t) for t in fwd_ts]

    # Historical line
    ax1.plot(hist_dt, hist_price, color="#58a6ff", linewidth=1.5, zorder=5, label="ETH/USDT (actual)")

    # Forward fan
    ax1.fill_between(fwd_dt, p5,  p95, alpha=0.12, color="#f78166", label="5-95% range")
    ax1.fill_between(fwd_dt, p25, p75, alpha=0.20, color="#f78166", label="25-75% range")
    ax1.plot(fwd_dt, p50, color="#f78166", linewidth=1.5, linestyle="--", alpha=0.8, label="Median path")

    # Vertical "now" line
    now_dt = ts2dt(now_ms / 1000)
    ax1.axvline(now_dt, color="#8b949e", linewidth=1, linestyle=":", alpha=0.7)
    ax1.text(now_dt, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else current * 1.05,
             "  NOW", color="#8b949e", fontsize=8, va="top")

    # Trade markers
    for ts, net, reason in buy_entries:
        p = price_at(ts * 1000)
        if p:
            color = "#3fb950" if (net and net > 0) else "#f85149"
            marker = "^"
            ax1.scatter(ts2dt(ts), p, color=color, marker=marker, s=60, zorder=10, alpha=0.85)

    for ts, net, reason in sell_entries:
        p = price_at(ts * 1000)
        if p:
            color = "#3fb950" if (net and net > 0) else "#f85149"
            ax1.scatter(ts2dt(ts), p, color=color, marker="v", s=60, zorder=10, alpha=0.85)

    # Stats box
    end_p50 = float(p50[-1])
    ret_pct  = (end_p50 / current - 1) * 100
    stats_txt = (
        f"Alpha stats (ETH BUY N={len(nets)})\n"
        f"WR: {wr*100:.0f}%   Mean: {mean_bps:+.1f} bps\n"
        f"Daily vol: {vol_day*100:.1f}%\n"
        f"Median path: {end_p50:.0f} ({ret_pct:+.1f}%)\n"
        f"FOR ENTERTAINMENT ONLY"
    )
    ax1.text(0.01, 0.97, stats_txt, transform=ax1.transAxes,
             color="#8b949e", fontsize=8, va="top", family="monospace",
             bbox=dict(facecolor="#161b22", edgecolor="#30363d", alpha=0.9, pad=5))

    ax1.set_title("ETH/USDT — Last 30 Days + 30-Day Alpha-Drift Projection",
                  color="#e6edf3", fontsize=13, pad=12)
    ax1.set_ylabel("Price (USDT)", color="#8b949e")
    ax1.yaxis.label.set_color("#8b949e")

    legend_elements = [
        mpatches.Patch(color="#58a6ff", label="ETH actual"),
        mpatches.Patch(color="#f78166", alpha=0.5, label="MC fan (30d)"),
        plt.Line2D([0], [0], color="#f78166", linestyle="--", label="Median path"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#3fb950", markersize=8, label="BUY win"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#f85149", markersize=8, label="BUY loss"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#3fb950", markersize=8, label="SELL win"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#f85149", markersize=8, label="SELL loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8,
               facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")

    # ── Bottom panel: trade P&L cumulative ────────────────────────────────────
    if nets:
        cum = []
        s = 0.0
        all_trade_rows = ledger.execute(
            "SELECT entry_ts_ms, net_bps FROM s34_trades WHERE rule_name LIKE '%ETH%BUY%' AND status='CLOSED' AND entry_ts_ms>=? ORDER BY entry_ts_ms",
            (start_ms,)
        ).fetchall()
        pnl_ts, pnl_cum = [], []
        s = 0.0
        for ts, net in all_trade_rows:
            if net is not None:
                s += net
                pnl_ts.append(ts2dt(ts/1000))
                pnl_cum.append(s)

        if pnl_ts:
            ax2.plot(pnl_ts, pnl_cum, color="#3fb950", linewidth=1.5)
            ax2.fill_between(pnl_ts, 0, pnl_cum,
                             where=[v >= 0 for v in pnl_cum], color="#3fb950", alpha=0.15)
            ax2.fill_between(pnl_ts, 0, pnl_cum,
                             where=[v < 0 for v in pnl_cum], color="#f85149", alpha=0.15)
            ax2.axhline(0, color="#8b949e", linewidth=0.5, linestyle=":")

    ax2.set_ylabel("Cum bps\n(ETH BUY)", color="#8b949e", fontsize=8)
    ax2.set_xlabel("Date (UTC)", color="#8b949e")

    plt.tight_layout(h_pad=0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved: {OUT}")

    micro.close()
    ledger.close()


if __name__ == "__main__":
    main()
