# encoding: utf-8
"""
S34 ETH Live Forecast Chart — standalone server on port 8055
FOR ENTERTAINMENT ONLY

Math:
  - Exponentially-weighted realized volatility (lambda=0.94, GARCH-inspired)
  - GBM with Poisson jump component (Merton model) calibrated from cascade data
  - Vol-regime scaling: 3d vs 30d vol ratio
  - Alpha drift: (trades/day * mean_bps) / 10000 added to mu
  - 500 Monte Carlo paths, N_DAYS=30 forward
  - Percentile fan: 5/15/25/50/75/85/95

Run:  python tools/s34_eth_forecast_live.py
Open: http://localhost:8055
"""
from __future__ import annotations
import json
import math
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT        = Path(__file__).resolve().parents[1]
MICRO_DB    = ROOT / "data" / "microstructure.db"
LEDGER_DB   = ROOT / "data" / "s34_intelligence.db"
PORT        = 8055
REFRESH_SEC = 60
N_PATHS     = 500
N_DAYS      = 30
EWMA_LAMBDA = 0.94   # GARCH-style vol decay
JUMP_WINDOW = 30     # days to estimate jump frequency
HIST_DAYS   = 45     # days of price history to show

_cache: dict[str, Any] = {}
_lock  = threading.Lock()


# ─── Data helpers ────────────────────────────────────────────────────────────

def micro_conn():
    return sqlite3.connect(f"file:{MICRO_DB}?mode=ro", uri=True, timeout=5)

def ledger_conn():
    return sqlite3.connect(f"file:{LEDGER_DB}?mode=ro", uri=True, timeout=5)


def eth_prices_5m(start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """5-min OHLC close prices from mark_prices."""
    with micro_conn() as c:
        rows = c.execute(
            """SELECT CAST(ts_ms/300000 AS INTEGER)*300000 AS b, AVG(mark_price)
               FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
               GROUP BY b ORDER BY b""",
            (start_ms, end_ms)
        ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def live_trades(start_ms: int) -> list[dict]:
    with ledger_conn() as c:
        rows = c.execute(
            """SELECT rule_name, entry_ts_ms, exit_ts_ms, net_bps, exit_reason
               FROM s34_trades WHERE status='CLOSED' AND entry_ts_ms >= ?
               ORDER BY entry_ts_ms""",
            (start_ms,)
        ).fetchall()
    return [
        {"rule": r[0], "entry_ms": r[1], "exit_ms": r[2],
         "net": r[3], "exit": r[4]}
        for r in rows
    ]


def alpha_stats() -> dict:
    """Per-rule stats for drift estimation."""
    with ledger_conn() as c:
        rows = c.execute(
            """SELECT rule_name, net_bps FROM s34_trades
               WHERE status='CLOSED' AND rule_name LIKE '%ETH%BUY%500K%'"""
        ).fetchall()
    nets = [float(r[1]) for r in rows if r[1] is not None]
    if not nets:
        return {"n": 0, "mean_bps": 0.0, "wr": 0.5, "trades_per_day": 0.0}
    # Estimate trades/day from date span
    with ledger_conn() as c:
        span = c.execute(
            """SELECT MIN(entry_ts_ms), MAX(entry_ts_ms) FROM s34_trades
               WHERE status='CLOSED' AND rule_name LIKE '%ETH%BUY%500K%'"""
        ).fetchone()
    days = max(1, (span[1] - span[0]) / (86400 * 1000)) if span[0] and span[1] else 1
    return {
        "n": len(nets),
        "mean_bps": sum(nets) / len(nets),
        "wr": sum(1 for n in nets if n > 0) / len(nets),
        "trades_per_day": len(nets) / days,
    }


def cascade_jump_params(prices: list[tuple[int, float]]) -> dict:
    """Estimate jump intensity and size from LARGE cascade events only (>=2M).
    Regular cascades are absorbed into diffusion vol.
    Jump = tail event, expected ~0.2-1/day for $2M+ threshold.
    """
    with micro_conn() as c:
        now_ms = prices[-1][0] if prices else int(time.time() * 1000)
        start  = now_ms - JUMP_WINDOW * 86400 * 1000
        rows = c.execute(
            """SELECT ts_ms, notional FROM liquidations
               WHERE symbol='ETHUSDT' AND side='BUY' AND ts_ms >= ?
               ORDER BY ts_ms""",
            (start,)
        ).fetchall()
    # Bucket to 30-min for large events (captures full cascade cluster)
    buckets: dict[int, float] = {}
    for ts, n in rows:
        bk = (ts // 1_800_000) * 1_800_000   # 30-min buckets
        buckets[bk] = buckets.get(bk, 0) + n

    JUMP_THRESHOLD = 2_000_000  # $2M+ = true jump event
    jump_buckets = {bk: v for bk, v in buckets.items() if v >= JUMP_THRESHOLD}
    lam = len(jump_buckets) / JUMP_WINDOW   # jumps per day

    # Price impact: 1h return centered on jump
    price_map  = {t: p for t, p in prices}
    sorted_ts  = sorted(price_map)
    jump_rets  = []
    for bk in jump_buckets:
        before = next((price_map[t] for t in sorted_ts if t <= bk), None)
        # 10-min later = immediate price impact only, not next hour
        after  = next((price_map[t] for t in sorted_ts if t >= bk + 600_000), None)
        if before and after and before > 0:
            r = (after - before) / before
            jump_rets.append(r)

    jump_rets.sort()
    # Median is robust to April-crash outliers
    k = jump_rets[len(jump_rets)//2] if jump_rets else 0.001
    # Clamp: realistic 10-min price impact of large cascade = 0.1%-1.5%
    k = max(-0.015, min(0.015, k))
    var_j   = sum((r - k)**2 for r in jump_rets) / len(jump_rets) if jump_rets else 0.0003
    sigma_j = max(0.005, min(0.02, var_j ** 0.5))   # clamp 0.5%-2% per jump
    lam     = max(0.1, min(1.0, lam))                # clamp 0.1-1 jumps/day

    return {"lambda": lam, "k": k, "sigma_j": sigma_j, "n_events": len(jump_buckets)}


# ─── Volatility ──────────────────────────────────────────────────────────────

def ewma_vol(log_rets: list[float], lam: float = EWMA_LAMBDA) -> float:
    """Exponentially-weighted moving-average variance → daily vol."""
    if len(log_rets) < 2:
        return 0.02
    var = log_rets[0] ** 2
    for r in log_rets[1:]:
        var = lam * var + (1 - lam) * r ** 2
    return var ** 0.5 * math.sqrt(288)   # 5-min bars → daily


def regime_vol_scale(prices: list[tuple[int, float]]) -> float:
    """3d/30d realized vol ratio — scale forecast vol by current regime."""
    if len(prices) < 50:
        return 1.0
    def rvol(rows):
        lr = [math.log(rows[i][1] / rows[i-1][1]) for i in range(1, len(rows))]
        return (sum(r**2 for r in lr) / len(lr)) ** 0.5 * math.sqrt(288) if lr else 0.02

    bars_3d  = max(2, 3 * 288)
    vol_3d   = rvol(prices[-bars_3d:])
    vol_30d  = rvol(prices)
    return vol_3d / vol_30d if vol_30d > 0 else 1.0


# ─── Monte Carlo ─────────────────────────────────────────────────────────────

def monte_carlo(
    current: float,
    mu_annual: float,          # annual drift
    sigma_daily: float,        # daily vol
    lam_jump: float,           # jumps per day
    k_jump: float,             # mean jump return
    sigma_jump: float,         # jump vol
    n_paths: int = N_PATHS,
    n_days: int = N_DAYS,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Merton jump-diffusion MC."""
    rng = random.Random(seed)
    dt  = 1 / 288              # 5-min step in days
    n_steps = n_days * 288
    mu_step = (mu_annual / 365 - lam_jump * k_jump - 0.5 * sigma_daily**2) * dt
    sig_step = sigma_daily * math.sqrt(dt)

    all_paths = []
    for _ in range(n_paths):
        p = current
        path = [p]
        for _ in range(n_steps):
            # Diffusion
            z   = rng.gauss(0, 1)
            ret = mu_step + sig_step * z
            # Poisson jumps
            n_j = rng.random() < (1 - math.exp(-lam_jump * dt))  # 0 or 1 jump
            if n_j:
                ret += k_jump + sigma_jump * rng.gauss(0, 1)
            p *= math.exp(ret)
            path.append(p)
        all_paths.append(path)

    # Percentile fan — use 10/90 as extremes (avoid pathological tail blow-ups)
    pcts = [10, 20, 30, 50, 70, 80, 90]
    result: dict[str, list[float]] = {str(p): [] for p in pcts}
    for step in range(n_steps + 1):
        vals = sorted(path[step] for path in all_paths)
        n    = len(vals)
        for p in pcts:
            idx = min(int(p / 100 * n), n - 1)
            result[str(p)].append(vals[idx])
    return result


# ─── Payload builder ─────────────────────────────────────────────────────────

def build_payload() -> dict:
    now_ms   = int(time.time() * 1000)
    hist_ms  = now_ms - HIST_DAYS * 86400 * 1000
    prices   = eth_prices_5m(hist_ms, now_ms)
    if not prices:
        return {"error": "no price data"}

    current  = prices[-1][1]

    # Log returns (5-min)
    log_rets = [math.log(prices[i][1] / prices[i-1][1])
                for i in range(1, len(prices))
                if prices[i-1][1] > 0]

    # Vol params
    vol_daily  = ewma_vol(log_rets)
    vol_scale  = regime_vol_scale(prices)
    vol_scaled = vol_daily * max(0.7, min(2.0, vol_scale))  # cap scaling

    # Drift: use 90d log-return annualised (more stable than 30d)
    # 90d in 5-min bars = 90*288 = 25920 bars
    long_rets = [math.log(prices[i][1] / prices[i-1][1])
                 for i in range(max(1, len(prices)-25920), len(prices))
                 if prices[i-1][1] > 0]
    raw_drift_90d = (sum(long_rets) / len(long_rets)) * 288 * 365 if long_rets else 0.0
    # Blend with 0 (neutral prior) so extreme recent trends are dampened
    raw_drift = raw_drift_90d * 0.4   # 40% weight on trend, 60% prior = 0

    alpha     = alpha_stats()
    # Alpha edge: conservative — only count 50% of historical mean (out-of-sample haircut)
    alpha_drift_annual = alpha["mean_bps"] * alpha["trades_per_day"] * 365 / 10000 * 0.5
    mu_annual = raw_drift + alpha_drift_annual

    # Jump params — parametric (data-calibrated but bounded for realism)
    jp = cascade_jump_params(prices)

    # Monte Carlo (re-seed each refresh with current minute for reproducibility)
    seed = int(now_ms // 60000)
    fan  = monte_carlo(current, mu_annual, vol_scaled, jp["lambda"], jp["k"], jp["sigma_j"],
                       n_paths=N_PATHS, n_days=N_DAYS, seed=seed)

    # Forward timestamps (5-min steps)
    step_ms  = 300_000
    fwd_ts   = [now_ms + i * step_ms for i in range(N_DAYS * 288 + 1)]

    # Trades for overlay
    trades = live_trades(hist_ms)

    # Price for chart
    price_series = [{"x": t, "y": p} for t, p in prices]

    # 30d summary
    p50_end   = fan["50"][-1]
    p10_end   = fan["10"][-1]
    p90_end   = fan["90"][-1]
    ret_med   = (p50_end / current - 1) * 100
    ret_bear  = (p10_end / current - 1) * 100
    ret_bull  = (p90_end / current - 1) * 100

    return {
        "ts":         now_ms,
        "updated":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "current":    current,
        "price":      price_series,
        "fwd_ts":     fwd_ts,
        "fan":        fan,
        "trades":     trades,
        "alpha": {
            "n":             alpha["n"],
            "mean_bps":      round(alpha["mean_bps"], 1),
            "wr_pct":        round(alpha["wr"] * 100, 0),
            "trades_per_day":round(alpha["trades_per_day"], 2),
            "alpha_drift_annual_bps": round(alpha_drift_annual * 10000, 0),
        },
        "vol": {
            "ewma_daily_pct":  round(vol_daily * 100, 2),
            "regime_scale":    round(vol_scale, 2),
            "scaled_daily_pct":round(vol_scaled * 100, 2),
        },
        "jumps": {
            "lambda_per_day": round(jp["lambda"], 2),
            "mean_k_pct":     round(jp["k"] * 100, 2),
            "sigma_j_pct":    round(jp["sigma_j"] * 100, 2),
        },
        "drift": {
            "raw_annual_pct":   round(raw_drift * 100, 1),
            "alpha_annual_pct": round(alpha_drift_annual * 100, 2),
            "total_annual_pct": round(mu_annual * 100, 1),
        },
        "summary": {
            "p50_30d":  round(p50_end, 2),
            "ret_med":  round(ret_med, 1),
            "ret_bear": round(ret_bear, 1),
            "ret_bull": round(ret_bull, 1),
        },
        "model": f"GBM+Jumps  vol={vol_scaled*100:.1f}%/d  lambda={jp['lambda']:.2f}j/d  N={N_PATHS} paths",
        "disclaimer": "FOR ENTERTAINMENT ONLY — Not financial advice.",
    }


# ─── HTTP server ─────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>S34 ETH Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Courier New',monospace;font-size:13px}
header{background:#161b22;border-bottom:1px solid #30363d;padding:10px 18px;display:flex;align-items:center;gap:16px}
header h1{font-size:15px;color:#58a6ff;letter-spacing:1px}
#updated{color:#8b949e;font-size:11px;margin-left:auto}
#disclaimer{color:#f85149;font-size:10px;font-weight:bold}
#main{display:grid;grid-template-columns:1fr 260px;gap:12px;padding:12px;height:calc(100vh - 48px)}
#chartWrap{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;position:relative}
canvas{display:block;width:100%!important}
#sidebar{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px}
.card h2{font-size:11px;color:#8b949e;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase}
.stat{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}
.stat:last-child{border:none}
.stat .k{color:#8b949e}
.stat .v{color:#e6edf3;font-weight:bold}
.pos{color:#3fb950!important}.neg{color:#f85149!important}.neu{color:#d29922!important}
.big{font-size:22px;color:#58a6ff;font-weight:bold;text-align:center;padding:6px 0}
.sub{font-size:11px;color:#8b949e;text-align:center}
#model{color:#8b949e;font-size:10px;text-align:center;padding:6px;background:#0d1117;border-radius:4px;margin-top:4px;word-break:break-all}
</style>
</head>
<body>
<header>
  <h1>S34 ETH Forecast</h1>
  <span id="disclaimer">FOR ENTERTAINMENT ONLY</span>
  <span id="updated">loading...</span>
</header>
<div id="main">
  <div id="chartWrap"><canvas id="fc"></canvas></div>
  <div id="sidebar">
    <div class="card">
      <h2>30-Day Outlook</h2>
      <div class="big" id="p50">—</div>
      <div class="sub" id="retMed">median path</div>
      <div style="margin-top:8px">
        <div class="stat"><span class="k">Bull (90th)</span><span class="v pos" id="bull">—</span></div>
        <div class="stat"><span class="k">Bear (10th)</span><span class="v neg" id="bear">—</span></div>
      </div>
    </div>
    <div class="card">
      <h2>Volatility</h2>
      <div class="stat"><span class="k">EWMA daily</span><span class="v" id="volEwma">—</span></div>
      <div class="stat"><span class="k">Regime scale</span><span class="v" id="volScale">—</span></div>
      <div class="stat"><span class="k">Used (scaled)</span><span class="v" id="volScaled">—</span></div>
    </div>
    <div class="card">
      <h2>Drift (annual)</h2>
      <div class="stat"><span class="k">ETH trend</span><span class="v" id="driftRaw">—</span></div>
      <div class="stat"><span class="k">Alpha edge</span><span class="v pos" id="driftAlpha">—</span></div>
      <div class="stat"><span class="k">Total mu</span><span class="v" id="driftTotal">—</span></div>
    </div>
    <div class="card">
      <h2>Jump Model</h2>
      <div class="stat"><span class="k">lambda (j/day)</span><span class="v" id="jLam">—</span></div>
      <div class="stat"><span class="k">Mean jump</span><span class="v" id="jK">—</span></div>
      <div class="stat"><span class="k">Jump sigma</span><span class="v" id="jSig">—</span></div>
    </div>
    <div class="card">
      <h2>Alpha (ETH BUY)</h2>
      <div class="stat"><span class="k">N trades</span><span class="v" id="aN">—</span></div>
      <div class="stat"><span class="k">WR</span><span class="v" id="aWR">—</span></div>
      <div class="stat"><span class="k">Mean bps</span><span class="v" id="aMean">—</span></div>
      <div class="stat"><span class="k">Trades/day</span><span class="v" id="aTpd">—</span></div>
    </div>
    <div id="model"></div>
  </div>
</div>
<script>
let chart = null;

function cls(v){ return v > 0 ? "pos" : v < 0 ? "neg" : "neu"; }
function pct(v){ return (v > 0 ? "+" : "") + v.toFixed(1) + "%"; }
function usd(v){ return "$" + Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2}); }

function tsLabel(ms){
  const d = new Date(ms);
  return d.toISOString().slice(0,10);
}

async function refresh(){
  const resp = await fetch("/data");
  if(!resp.ok) return;
  const d = await resp.json();
  if(d.error){ console.warn(d.error); return; }

  document.getElementById("updated").textContent = d.updated;

  // Sidebar
  document.getElementById("p50").textContent   = usd(d.summary.p50_30d);
  const medEl = document.getElementById("retMed");
  medEl.textContent = pct(d.summary.ret_med) + " median path (30d)";
  medEl.className   = "sub " + cls(d.summary.ret_med);

  document.getElementById("bull").textContent  = pct(d.summary.ret_bull);
  document.getElementById("bear").textContent  = pct(d.summary.ret_bear);
  document.getElementById("volEwma").textContent  = d.vol.ewma_daily_pct + "%";
  document.getElementById("volScale").textContent = "×" + d.vol.regime_scale;
  document.getElementById("volScaled").textContent = d.vol.scaled_daily_pct + "%";
  const dRaw = document.getElementById("driftRaw");
  dRaw.textContent = pct(d.drift.raw_annual_pct);
  dRaw.className = "v " + cls(d.drift.raw_annual_pct);
  document.getElementById("driftAlpha").textContent = "+" + d.drift.alpha_annual_pct + "%";
  const dTot = document.getElementById("driftTotal");
  dTot.textContent = pct(d.drift.total_annual_pct);
  dTot.className = "v " + cls(d.drift.total_annual_pct);
  document.getElementById("jLam").textContent  = d.jumps.lambda_per_day;
  document.getElementById("jK").textContent    = pct(d.jumps.mean_k_pct);
  document.getElementById("jSig").textContent  = pct(d.jumps.sigma_j_pct);
  document.getElementById("aN").textContent    = d.alpha.n;
  document.getElementById("aWR").textContent   = d.alpha.wr_pct + "%";
  document.getElementById("aMean").textContent = (d.alpha.mean_bps > 0 ? "+" : "") + d.alpha.mean_bps + " bps";
  document.getElementById("aTpd").textContent  = d.alpha.trades_per_day + "/day";
  document.getElementById("model").textContent = d.model;

  // Chart
  buildChart(d);
}

function buildChart(d){
  const now_ms  = d.ts;
  const fwdTs   = d.fwd_ts;
  const fan     = d.fan;
  const prices  = d.price;

  // Thinned forecast points (every 6 = 30 min resolution)
  const thin = 6;
  const fTs  = fwdTs.filter((_,i) => i % thin === 0);
  const fp   = pct => fan[pct].filter((_,i) => i % thin === 0);

  const datasets = [
    // Price history
    {
      label: "ETH/USDT",
      data: prices,
      parsing: false,
      borderColor: "#58a6ff",
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0,
      order: 1,
      s34type: "price",
    },
    // Fan bands (10/20/30/50/70/80/90)
    { label:"_p90", data: fTs.map((x,i)=>({x,y:fp("90")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"rgba(248,113,113,0.07)",
      fill:"+1", pointRadius:0, order:10 },
    { label:"_p80", data: fTs.map((x,i)=>({x,y:fp("80")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"rgba(248,113,113,0.10)",
      fill:"+1", pointRadius:0, order:10 },
    { label:"_p70", data: fTs.map((x,i)=>({x,y:fp("70")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"rgba(248,113,113,0.13)",
      fill:"+1", pointRadius:0, order:10 },
    { label:"_p50 (median)", data: fTs.map((x,i)=>({x,y:fp("50")[i]})), parsing:false,
      borderColor:"rgba(248,113,113,0.9)", borderWidth:1.8,
      fill:"+1", backgroundColor:"rgba(248,113,113,0.13)",
      pointRadius:0, order:5 },
    { label:"_p30", data: fTs.map((x,i)=>({x,y:fp("30")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"rgba(248,113,113,0.13)",
      fill:"+1", pointRadius:0, order:10 },
    { label:"_p20", data: fTs.map((x,i)=>({x,y:fp("20")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"rgba(248,113,113,0.10)",
      fill:"+1", pointRadius:0, order:10 },
    { label:"_p10",  data: fTs.map((x,i)=>({x,y:fp("10")[i]})), parsing:false,
      borderColor:"transparent", backgroundColor:"transparent",
      fill:false, pointRadius:0, order:10 },
  ];

  // Trade markers (BUY = triangle up, SELL = triangle down)
  const buyWin  = [], buyLoss = [], sellWin = [], sellLoss = [];
  for(const t of d.trades){
    const isBuy  = t.rule.includes("BUY");
    const isWin  = t.net > 0;
    // find nearest price
    const nearest = prices.reduce((a,b) => Math.abs(b.x - t.entry_ms) < Math.abs(a.x - t.entry_ms) ? b : a, prices[0]);
    const pt = { x: t.entry_ms, y: nearest ? nearest.y : null };
    if(pt.y === null) continue;
    if(isBuy  && isWin)  buyWin.push(pt);
    if(isBuy  && !isWin) buyLoss.push(pt);
    if(!isBuy && isWin)  sellWin.push(pt);
    if(!isBuy && !isWin) sellLoss.push(pt);
  }

  function scatter(data, color, ptStyle, label){
    return { label, data, parsing:false, type:"scatter",
             pointStyle:ptStyle, pointRadius:7, borderColor:color,
             backgroundColor:color+"cc", order:2 };
  }
  datasets.push(scatter(buyWin,   "#3fb950", "triangle",       "BUY win"));
  datasets.push(scatter(buyLoss,  "#f85149", "triangle",       "BUY loss"));
  datasets.push(scatter(sellWin,  "#3fb950", "triangleDown",   "SELL win"));
  datasets.push(scatter(sellLoss, "#f85149", "triangleDown",   "SELL loss"));

  // "NOW" vertical line via annotation-free approach: add a vertical dataset
  datasets.push({
    label:"_now",
    data: [{x: now_ms, y: d.current * 0.85}, {x: now_ms, y: d.current * 1.15}],
    parsing:false, borderColor:"rgba(139,148,158,0.4)", borderWidth:1,
    borderDash:[4,4], pointRadius:0, order:3,
  });

  const ctx = document.getElementById("fc");
  if(chart){ chart.destroy(); chart = null; }

  // x range
  const xMin = prices[0].x;
  const xMax = fwdTs[fwdTs.length - 1];

  chart = new Chart(ctx, {
    type:"line",
    data:{ datasets },
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      parsing:false, normalized:true,
      scales:{
        x:{
          type:"linear", min:xMin, max:xMax,
          grid:{color:"rgba(220,228,236,0.06)"},
          ticks:{ color:"#8b949e", maxTicksLimit:12,
            callback: v => { const d = new Date(v); return d.toISOString().slice(5,10); }
          }
        },
        y:{
          position:"left",
          grid:{color:"rgba(220,228,236,0.06)"},
          ticks:{ color:"#dce4ec",
            callback: v => "$" + Number(v).toLocaleString("en-US",{maximumFractionDigits:0})
          }
        }
      },
      plugins:{
        legend:{
          labels:{ color:"#8b949e", font:{size:11},
            filter: item => !String(item.text||"").startsWith("_")
          }
        },
        tooltip:{
          mode:"nearest", intersect:false,
          callbacks:{
            title: items => { const x = items[0]?.parsed.x; return x ? new Date(x).toISOString().slice(0,16).replace("T"," ")+" UTC" : ""; },
            label: ctx => {
              const ds = ctx.dataset;
              if(ds.s34type==="price") return "ETH: $" + ctx.parsed.y.toFixed(2);
              return ds.label + ": $" + ctx.parsed.y.toFixed(2);
            }
          }
        }
      }
    }
  });
}

refresh();
setInterval(refresh, """ + str(REFRESH_SEC * 1000) + r""");
</script>
</body>
</html>"""


PAYLOAD_CACHE: dict[str, Any] = {}
PAYLOAD_LOCK = threading.Lock()
PAYLOAD_TS   = 0.0


def get_payload() -> dict:
    global PAYLOAD_TS
    now = time.time()
    with PAYLOAD_LOCK:
        if now - PAYLOAD_TS > REFRESH_SEC or not PAYLOAD_CACHE:
            try:
                data = build_payload()
                PAYLOAD_CACHE.clear()
                PAYLOAD_CACHE.update(data)
                PAYLOAD_TS = now
            except Exception as e:  # noqa
                PAYLOAD_CACHE["error"] = str(e)
        return dict(PAYLOAD_CACHE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): return  # silence

    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(get_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)


def main():
    import webbrowser
    print(f"S34 ETH Forecast  ->  http://localhost:{PORT}")
    print(f"Refresh: every {REFRESH_SEC}s  |  Ctrl+C to stop")
    # Pre-warm cache in background
    threading.Thread(target=get_payload, daemon=True).start()
    server = ThreadingHTTPServer(("", PORT), Handler)
    webbrowser.open(f"http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
