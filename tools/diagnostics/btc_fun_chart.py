# encoding: utf-8
"""BTC 1-Month Fun Prediction — S34 Alpha Based. For entertainment only."""
import sqlite3, json, random, math, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT     = Path("D:/eclipse_scalper")
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
MICRO_DB = ROOT / "data" / "microstructure.db"
OUT_PATH = ROOT / "reports" / "btc_fun_prediction.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── 1. BTC hourly prices (last 30 days) ────────────────────────────────────
cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
conn_m = sqlite3.connect("file:" + str(MICRO_DB) + "?mode=ro", uri=True)
rows = conn_m.execute(
    "SELECT (ts_ms/3600000)*3600000 AS hr, AVG(mid_price) AS px "
    "FROM book_ticker WHERE symbol='BTCUSDT' AND ts_ms>=? GROUP BY hr ORDER BY hr",
    (cutoff_ms,)
).fetchall()
conn_m.close()

hist_ts = [datetime.fromtimestamp(r[0]/1000, tz=timezone.utc) for r in rows]
hist_px = [float(r[1]) for r in rows]
print(f"BTC hourly candles: {len(hist_px)}")

# ── 2. BTC signals from intelligence DB ────────────────────────────────────
conn_i = sqlite3.connect("file:" + str(INTEL_DB) + "?mode=ro", uri=True)
btc_trades = conn_i.execute(
    "SELECT entry_ts_ms, net_bps, trade_json FROM s34_trades "
    "WHERE status='CLOSED' AND symbol='BTCUSDT' AND net_bps IS NOT NULL ORDER BY entry_ts_ms"
).fetchall()
conn_i.close()

bt_ts, bt_px, bt_net = [], [], []
for ts_ms, net_bps, tj in btc_trades:
    dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
    if not hist_ts or dt < hist_ts[0]:
        continue
    try:
        ep = float(json.loads(tj).get("entry_price") or 0)
    except Exception:
        ep = 0
    if ep > 0:
        bt_ts.append(dt); bt_px.append(ep); bt_net.append(float(net_bps))
print(f"BTC trade signals in window: {len(bt_ts)}")

# ── 3. Forward projection ──────────────────────────────────────────────────
last_px = hist_px[-1] if hist_px else 107_000.0
last_dt = hist_ts[-1] if hist_ts else datetime.now(timezone.utc)

if len(hist_px) > 24:
    rets = [math.log(hist_px[i]/hist_px[i-1]) for i in range(1, len(hist_px)) if hist_px[i-1]>0]
    vol = float(np.std(rets))
else:
    vol = 0.0028

HOURS = 30 * 24
# Alpha drift: 0.25 BTC signals/day * 35.7 bps mean = 8.9 bps/day
alpha_drift_hr = 35.7 * 0.25 / 24 / 10000

BULL  = alpha_drift_hr + 0.00035   # bull run extra momentum
BASE  = alpha_drift_hr
BEAR  = alpha_drift_hr - 0.00020

N_PATHS = 80

def gen_paths(drift, seed):
    paths = []
    for i in range(N_PATHS):
        random.seed(seed + i)
        p = [last_px]
        for _ in range(HOURS):
            p.append(p[-1] * math.exp(drift + random.gauss(0, vol)))
        paths.append(p)
    return paths

bull_p = gen_paths(BULL, 100)
base_p = gen_paths(BASE, 200)
bear_p = gen_paths(BEAR, 300)
fwd_dt = [last_dt + timedelta(hours=h) for h in range(HOURS+1)]

def pct(paths, q):
    return [np.percentile([p[h] for p in paths], q) for h in range(HOURS+1)]

# ── 4. Draw ────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# History
ax.plot(hist_ts, hist_px, color="#e6edf3", lw=2.0, zorder=5, label="BTC/USDT actual")

# Now line
ax.axvline(last_dt, color="#58a6ff", lw=1.2, ls="--", alpha=0.8)
y_top = max(hist_px) * 1.005 if hist_px else last_px * 1.01
ax.text(last_dt, y_top, "  NOW", color="#58a6ff", fontsize=9, va="top")

# Fans
ax.fill_between(fwd_dt, pct(bear_p,20), pct(bear_p,80), color="#f85149", alpha=0.10)
ax.plot(fwd_dt, pct(bear_p,50), color="#f85149", lw=1.5, ls=":", alpha=0.75,
        label=f"Bear  ${pct(bear_p,50)[-1]:,.0f}")

ax.fill_between(fwd_dt, pct(base_p,20), pct(base_p,80), color="#58a6ff", alpha=0.12)
ax.plot(fwd_dt, pct(base_p,50), color="#58a6ff", lw=2.0, alpha=0.90,
        label=f"Base  ${pct(base_p,50)[-1]:,.0f}")

ax.fill_between(fwd_dt, pct(bull_p,20), pct(bull_p,80), color="#3fb950", alpha=0.14)
ax.plot(fwd_dt, pct(bull_p,50), color="#3fb950", lw=2.0, alpha=0.90,
        label=f"Bull  ${pct(bull_p,50)[-1]:,.0f}")

# BTC signal markers
if bt_ts:
    w = [(t,p) for t,p,n in zip(bt_ts,bt_px,bt_net) if n>0]
    l = [(t,p) for t,p,n in zip(bt_ts,bt_px,bt_net) if n<=0]
    if w: ax.scatter([x[0] for x in w],[x[1] for x in w], marker="^", s=90, c="#3fb950", zorder=10, label=f"BTC signal WIN x{len(w)}")
    if l: ax.scatter([x[0] for x in l],[x[1] for x in l], marker="v", s=90, c="#f85149", zorder=10, label=f"BTC signal LOSS x{len(l)}")

# End labels
for paths, clr in [(bull_p,"#3fb950"),(base_p,"#58a6ff"),(bear_p,"#f85149")]:
    med = np.median([p[-1] for p in paths])
    ax.annotate(f"${med:,.0f}", xy=(fwd_dt[-1], med),
                xytext=(6,0), textcoords="offset points", color=clr, fontsize=8.5,
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0d1117", ec=clr, alpha=0.85))

# Alpha stats box
info = (
    "S34 Alpha Inputs\n"
    "ETH 500K  WR=79%  +39 bps\n"
    "SOL 200K  WR=80%  +55 bps\n"
    "BTC 1M    WR=67%  +36 bps\n"
    f"BTC vol   {vol*100:.2f}%/hr (30d actual)\n"
    "Drift     +8.9 bps/day (alpha)"
)
ax.text(0.01, 0.97, info, transform=ax.transAxes, fontsize=7.5, color="#8b949e",
        va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#161b22", ec="#30363d", alpha=0.9))

ax.text(0.99, 0.01, "For entertainment only — not a price forecast.",
        transform=ax.transAxes, fontsize=7, color="#484f58", ha="right", va="bottom",
        style="italic")

ax.set_title("BTC/USDT — 30-Day S34 Alpha Projection", color="#e6edf3",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Price (USDT)", color="#8b949e", fontsize=10)
ax.tick_params(colors="#8b949e", labelsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.grid(color="#21262d", lw=0.5, alpha=0.6)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.65,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
          bbox_to_anchor=(0.01, 0.75))

plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight",
            facecolor="#0d1117", edgecolor="none")
plt.close()
print(f"Saved: {OUT_PATH}")
print(f"Last px: ${last_px:,.2f}")
bull_end = np.median([p[-1] for p in bull_p])
base_end = np.median([p[-1] for p in base_p])
bear_end = np.median([p[-1] for p in bear_p])
print(f"Bull 30d: ${bull_end:,.0f}  Base: ${base_end:,.0f}  Bear: ${bear_end:,.0f}")
