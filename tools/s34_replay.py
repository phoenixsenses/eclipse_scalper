"""
S34 Trade Replay Viewer  —  port 5052
Browse all historical S34 trades with price action, liquidation flow,
CVD, and entry/exit markers rebuilt from microstructure.db.

Usage:
    python tools/s34_replay.py
    python tools/s34_replay.py --host 127.0.0.1 --port 5052
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

ROOT     = Path(__file__).resolve().parents[1]
MICRO_DB = ROOT / "data" / "microstructure.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
VARIANT  = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
WINDOW_MS = 30 * 60 * 1000  # ±30 min around signal


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _intel() -> sqlite3.Connection:
    c = sqlite3.connect(str(INTEL_DB), timeout=5)
    c.row_factory = sqlite3.Row
    return c


def _micro() -> sqlite3.Connection:
    c = sqlite3.connect(str(MICRO_DB), timeout=5)
    c.row_factory = sqlite3.Row
    return c


def trade_list() -> list[dict]:
    conn = _intel()
    rows = conn.execute("""
        SELECT trade_id, rule_name, entry_ts_ms, exit_ts_ms,
               entry_price, exit_price, exit_reason, net_bps,
               opened_at_utc, trade_json
        FROM s34_trades
        WHERE rule_name = ? AND status = 'CLOSED'
        ORDER BY entry_ts_ms DESC
    """, (VARIANT,)).fetchall()
    conn.close()

    result = []
    for r in rows:
        try:
            tj = json.loads(r["trade_json"])
        except Exception:
            tj = {}
        result.append({
            "id":          r["trade_id"],
            "entry_ts":    r["entry_ts_ms"],
            "exit_ts":     r["exit_ts_ms"],
            "entry_price": r["entry_price"],
            "exit_price":  r["exit_price"],
            "exit_reason": r["exit_reason"],
            "net_bps":     r["net_bps"],
            "gross_bps":   tj.get("gross_bps"),
            "tp_price":    tj.get("tp_price"),
            "sl_price":    tj.get("sl_price"),
            "be_price":    tj.get("be_trigger_price"),
            "signal_ts":   tj.get("signal_ts_ms"),
            "liq_notional":tj.get("signal", {}).get("liq_total_notional"),
            "liq_count":   tj.get("signal", {}).get("liq_count"),
            "opened_at":   r["opened_at_utc"],
            "fee_bps":     tj.get("fee_cost_bps"),
            "entry_adv":   tj.get("entry_adverse_bps"),
        })
    return result


def trade_context(trade_id: str) -> dict | None:
    conn_i = _intel()
    row = conn_i.execute(
        "SELECT * FROM s34_trades WHERE trade_id=?", (trade_id,)).fetchone()
    if not row:
        conn_i.close()
        return None

    tj = json.loads(row["trade_json"])
    signal_ts  = tj.get("signal_ts_ms") or row["entry_ts_ms"]
    entry_ts   = row["entry_ts_ms"]
    exit_ts    = row["exit_ts_ms"]
    t_start    = signal_ts - WINDOW_MS
    t_end      = (exit_ts  or signal_ts) + WINDOW_MS
    conn_i.close()

    conn_m = _micro()

    # 15-second price candles from agg_trades
    candles = conn_m.execute("""
        SELECT (ts_ms / 15000) * 15000 AS t,
               MIN(price) AS lo, MAX(price) AS hi,
               FIRST_VALUE(price) OVER (
                   PARTITION BY (ts_ms/15000) ORDER BY ts_ms
               ) AS op,
               LAST_VALUE(price) OVER (
                   PARTITION BY (ts_ms/15000) ORDER BY ts_ms
                   ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
               ) AS cl,
               SUM(notional) AS vol
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
        GROUP BY t ORDER BY t ASC
    """, (t_start, t_end)).fetchall()

    # simpler: just use minute OHLC via plain GROUP BY
    prices = conn_m.execute("""
        SELECT (ts_ms / 15000)*15000 AS t,
               AVG(price) AS mid,
               SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END) AS buy_n,
               SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) AS sell_n
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
        GROUP BY t ORDER BY t ASC
    """, (t_start, t_end)).fetchall()

    # CVD in 1-minute buckets
    cvd_rows = conn_m.execute("""
        SELECT (ts_ms/60000)*60000 AS t,
               SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END) AS buy_n,
               SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) AS sell_n
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
        GROUP BY t ORDER BY t ASC
    """, (t_start, t_end)).fetchall()

    # Liquidations
    liqs = conn_m.execute("""
        SELECT ts_ms, price, notional, side
        FROM liquidations
        WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
        ORDER BY ts_ms ASC
    """, (t_start, t_end)).fetchall()

    conn_m.close()

    # Build CVD cumulative
    cum = 0.0
    cvd_series = []
    for r in cvd_rows:
        buy = float(r["buy_n"] or 0)
        sell = float(r["sell_n"] or 0)
        cum += buy - sell
        cvd_series.append({"t": int(r["t"]), "buy": buy, "sell": sell, "cum": cum})

    return {
        "trade_id":    trade_id,
        "entry_ts":    entry_ts,
        "exit_ts":     exit_ts,
        "signal_ts":   signal_ts,
        "entry_price": row["entry_price"],
        "exit_price":  row["exit_price"],
        "exit_reason": row["exit_reason"],
        "net_bps":     row["net_bps"],
        "gross_bps":   tj.get("gross_bps"),
        "tp_price":    tj.get("tp_price"),
        "sl_price":    tj.get("sl_price"),
        "be_price":    tj.get("be_trigger_price"),
        "fee_bps":     tj.get("fee_cost_bps"),
        "entry_adv":   tj.get("entry_adverse_bps"),
        "liq_notional":tj.get("signal", {}).get("liq_total_notional"),
        "window_start":t_start,
        "window_end":  t_end,
        "prices": [{"t": int(r["t"]), "mid": float(r["mid"]),
                    "buy": float(r["buy_n"] or 0),
                    "sell": float(r["sell_n"] or 0)} for r in prices],
        "cvd":    cvd_series,
        "liqs":   [{"t": int(r["ts_ms"]), "price": float(r["price"]),
                    "notional": float(r["notional"]), "side": str(r["side"])}
                   for r in liqs],
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>S34 Replay — ETH</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #060a0f; --panel: #0c1420; --border: #1a2535;
  --ink: #d0dce8; --muted: #5a7080;
  --green: #1eff8e; --red: #ff3355; --gold: #ffd060;
  --blue: #2ec4ff; --orange: #ff8c00;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--ink);
  font: 12px/1.4 Consolas, monospace; display: flex; flex-direction: column; overflow: hidden; }

/* ── header ── */
#header { display:flex; align-items:center; gap:16px; padding:6px 14px;
  background:var(--panel); border-bottom:1px solid var(--border); flex-shrink:0; }
#header .sym { font-size:14px; font-weight:bold; color:var(--blue); }
#header .stat { color:var(--muted); }
#header .stat span { color:var(--ink); margin-left:4px; }
#hdr-outcome { font-size:15px; font-weight:bold; }
#hdr-nav { margin-left:auto; display:flex; gap:8px; }
#hdr-nav button { background:var(--panel); border:1px solid var(--border);
  color:var(--ink); padding:3px 10px; cursor:pointer; font:12px monospace; }
#hdr-nav button:hover { border-color:var(--blue); color:var(--blue); }

/* ── body ── */
#body { display:flex; flex:1; overflow:hidden; min-height:0; }

/* ── trade list ── */
#trade-list { width:200px; flex-shrink:0; border-right:1px solid var(--border);
  overflow-y:auto; scrollbar-width:thin; }
.tl-row { display:grid; grid-template-columns:40px 1fr 58px;
  padding:4px 8px; cursor:pointer; border-bottom:1px solid rgba(26,37,53,0.4);
  font-size:11px; align-items:center; }
.tl-row:hover { background:rgba(46,196,255,0.06); }
.tl-row.active { background:rgba(46,196,255,0.12); border-left:2px solid var(--blue); }
.tl-id { color:var(--muted); font-size:10px; }
.tl-date { color:var(--muted); font-size:10px; }
.tl-net { text-align:right; font-weight:bold; }
.win  { color:var(--green); }
.loss { color:var(--red); }

/* ── main chart area ── */
#main { flex:1; display:flex; flex-direction:column; overflow:hidden; min-height:0; min-width:0; }
#chart-wrap { flex:1; position:relative; min-height:0; }
#price-canvas { display:block; width:100%; height:100%; }
#cvd-wrap { height:90px; flex-shrink:0; border-top:1px solid var(--border); position:relative; }
#cvd-label { position:absolute; top:3px; left:8px; font-size:10px; color:var(--muted); z-index:1; pointer-events:none; }
#cvd-canvas { width:100%; height:100%; display:block; }

/* ── right info panel ── */
#info-panel { width:200px; flex-shrink:0; border-left:1px solid var(--border);
  padding:12px 10px; font-size:11px; overflow-y:auto; }
.info-row { display:flex; justify-content:space-between; padding:3px 0;
  border-bottom:1px solid rgba(26,37,53,0.3); }
.info-row .label { color:var(--muted); }
.info-row .val { font-weight:bold; }
.info-section { color:var(--muted); font-size:10px; letter-spacing:.05em;
  margin:10px 0 4px; text-transform:uppercase; }
</style>
</head>
<body>

<div id="header">
  <div class="sym">S34 REPLAY</div>
  <div id="hdr-outcome">—</div>
  <div class="stat">EXIT <span id="hdr-exit">—</span></div>
  <div class="stat">NET <span id="hdr-net">—</span></div>
  <div class="stat">GROSS <span id="hdr-gross">—</span></div>
  <div id="hdr-nav">
    <button id="btn-prev">◀ PREV</button>
    <span id="hdr-idx" style="color:var(--muted);font-size:11px;padding:3px 6px;">—</span>
    <button id="btn-next">NEXT ▶</button>
  </div>
</div>

<div id="body">

  <div id="trade-list"><div id="tl-rows"></div></div>

  <div id="main">
    <div id="chart-wrap">
      <canvas id="price-canvas"></canvas>
    </div>
    <div id="cvd-wrap">
      <div id="cvd-label">CVD</div>
      <canvas id="cvd-canvas"></canvas>
    </div>
  </div>

  <div id="info-panel">
    <div class="info-section">Trade</div>
    <div class="info-row"><span class="label">ID</span>       <span class="val" id="inf-id">—</span></div>
    <div class="info-row"><span class="label">Entry</span>    <span class="val" id="inf-entry">—</span></div>
    <div class="info-row"><span class="label">Exit</span>     <span class="val" id="inf-exit">—</span></div>
    <div class="info-row"><span class="label">TP</span>       <span class="val" id="inf-tp">—</span></div>
    <div class="info-row"><span class="label">SL</span>       <span class="val" id="inf-sl">—</span></div>
    <div class="info-row"><span class="label">BE</span>       <span class="val" id="inf-be">—</span></div>
    <div class="info-section">Costs</div>
    <div class="info-row"><span class="label">Fee</span>      <span class="val" id="inf-fee">—</span></div>
    <div class="info-row"><span class="label">Entry adv</span><span class="val" id="inf-adv">—</span></div>
    <div class="info-section">Signal</div>
    <div class="info-row"><span class="label">Liq $</span>   <span class="val" id="inf-liq">—</span></div>
    <div class="info-row"><span class="label">Liq count</span><span class="val" id="inf-liqn">—</span></div>
  </div>

</div>

<script>
let trades = [];
let cvdChart = null;
let currentIdx = 0;

// ── helpers ─────────────────────────────────────────────────────────────────
function fmtTime(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.getUTCFullYear() + '-' +
    String(d.getUTCMonth()+1).padStart(2,'0') + '-' +
    String(d.getUTCDate()).padStart(2,'0') + ' ' +
    String(d.getUTCHours()).padStart(2,'0') + ':' +
    String(d.getUTCMinutes()).padStart(2,'0') + ':' +
    String(d.getUTCSeconds()).padStart(2,'0');
}
function fmtShort(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return String(d.getUTCHours()).padStart(2,'0') + ':' +
         String(d.getUTCMinutes()).padStart(2,'0') + ':' +
         String(d.getUTCSeconds()).padStart(2,'0');
}

// ── trade list ───────────────────────────────────────────────────────────────
function renderList() {
  const rows = trades.map((t, i) => {
    const win = t.net_bps > 0;
    const dt = t.opened_at ? t.opened_at.slice(0,16).replace('T',' ') : '—';
    const active = i === currentIdx ? ' active' : '';
    const cls = win ? 'win' : 'loss';
    const sign = t.net_bps >= 0 ? '+' : '';
    return `<div class="tl-row${active}" data-idx="${i}" onclick="loadTrade(${i})">
      <div class="tl-id">${t.id}</div>
      <div class="tl-date">${dt.slice(5)}</div>
      <div class="tl-net ${cls}">${sign}${(t.net_bps||0).toFixed(1)}</div>
    </div>`;
  });
  document.getElementById('tl-rows').innerHTML = rows.join('');
}

// ── price chart ─────────────────────────────────────────────────────────────
function renderPrice(ctx) {
  const canvas = document.getElementById('price-canvas');
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  if (!W || !H) return;
  canvas.width = W; canvas.height = H;
  const c = canvas.getContext('2d');

  const prices = ctx.prices || [];
  if (!prices.length) {
    c.fillStyle = '#060a0f'; c.fillRect(0,0,W,H);
    c.fillStyle = '#5a7080'; c.font='13px monospace';
    c.fillText('No price data', W/2-60, H/2); return;
  }

  // Price range
  const mids = prices.map(p=>p.mid);
  let pMin = Math.min(...mids), pMax = Math.max(...mids);
  // Include entry, exit, tp, sl in range
  [ctx.entry_price, ctx.exit_price, ctx.tp_price, ctx.sl_price].forEach(v => {
    if (v) { pMin = Math.min(pMin, v); pMax = Math.max(pMax, v); }
  });
  const buf = (pMax - pMin) * 0.15 + 0.1;
  pMin -= buf; pMax += buf;
  const pRange = pMax - pMin;

  // Time range
  const tMin = prices[0].t, tMax = prices[prices.length-1].t;
  const tRange = Math.max(tMax - tMin, 1);

  const toX = t => ((t - tMin) / tRange) * W;
  const toY = p => ((pMax - p) / pRange) * H;

  c.fillStyle = '#060a0f'; c.fillRect(0,0,W,H);

  // Grid lines (price)
  const step = Math.pow(10, Math.floor(Math.log10(pRange))) / 4;
  c.strokeStyle = 'rgba(26,37,53,0.6)'; c.lineWidth = 1;
  for (let p = Math.ceil(pMin/step)*step; p <= pMax; p += step) {
    const y = toY(p);
    c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke();
    c.fillStyle = 'rgba(90,112,128,0.6)'; c.font='9px monospace';
    c.fillText(p.toFixed(2), 2, y - 2);
  }

  // ── Liquidation dots ──────────────────────────────────────────────────────
  for (const liq of (ctx.liqs||[])) {
    const lx = toX(liq.t), ly = toY(liq.price);
    const r = Math.max(3, Math.min(14, liq.notional / 60000));
    c.beginPath(); c.arc(lx, ly, r, 0, Math.PI*2);
    c.fillStyle = liq.side === 'BUY' ? 'rgba(255,140,0,0.7)' : 'rgba(180,0,255,0.7)';
    c.fill();
    c.strokeStyle='rgba(255,255,255,0.4)'; c.lineWidth=1; c.stroke();
  }

  // ── Volume bars (faint, at bottom) ──────────────────────────────────────
  const maxVol = Math.max(...prices.map(p=>p.buy+p.sell), 1);
  const barH = H * 0.12;
  for (const p of prices) {
    const x = toX(p.t);
    const bh = (p.buy+p.sell)/maxVol * barH;
    const isBuy = p.buy >= p.sell;
    c.fillStyle = isBuy ? 'rgba(30,255,142,0.18)' : 'rgba(255,51,85,0.18)';
    c.fillRect(x-1, H-bh, 3, bh);
  }

  // ── TP / SL / BE lines ────────────────────────────────────────────────────
  const drawLevel = (price, color, label) => {
    if (!price) return;
    const y = toY(price);
    c.save(); c.setLineDash([4,3]);
    c.strokeStyle = color; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke();
    c.setLineDash([]);
    c.fillStyle = color; c.font = 'bold 10px monospace';
    c.fillText(label + ' ' + price.toFixed(2), W-120, y-3);
    c.restore();
  };
  drawLevel(ctx.tp_price, 'rgba(30,255,142,0.65)', 'TP');
  drawLevel(ctx.sl_price, 'rgba(255,51,85,0.65)',  'SL');
  drawLevel(ctx.be_price, 'rgba(255,208,96,0.45)', 'BE');

  // ── Signal vertical line ──────────────────────────────────────────────────
  if (ctx.signal_ts) {
    const sx = toX(ctx.signal_ts);
    c.save(); c.setLineDash([2,2]);
    c.strokeStyle = 'rgba(255,140,0,0.7)'; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(sx,0); c.lineTo(sx,H); c.stroke();
    c.setLineDash([]);
    c.fillStyle = 'rgba(255,140,0,0.8)'; c.font = '10px monospace';
    c.fillText('SIG', sx+3, 14);
    c.restore();
  }

  // ── Entry marker ──────────────────────────────────────────────────────────
  if (ctx.entry_ts && ctx.entry_price) {
    const ex = toX(ctx.entry_ts), ey = toY(ctx.entry_price);
    c.beginPath(); c.arc(ex, ey, 7, 0, Math.PI*2);
    c.fillStyle = 'rgba(46,196,255,0.85)'; c.fill();
    c.strokeStyle='#ffffff'; c.lineWidth=1.5; c.stroke();
    c.fillStyle='#2ec4ff'; c.font='bold 10px monospace';
    c.fillText('ENTRY', ex+10, ey+4);
  }

  // ── Exit marker ───────────────────────────────────────────────────────────
  if (ctx.exit_ts && ctx.exit_price) {
    const xx = toX(ctx.exit_ts), xy = toY(ctx.exit_price);
    const win = (ctx.net_bps||0) > 0;
    const col = win ? '#1eff8e' : '#ff3355';
    c.beginPath(); c.arc(xx, xy, 7, 0, Math.PI*2);
    c.fillStyle = col + 'cc'; c.fill();
    c.strokeStyle='#ffffff'; c.lineWidth=1.5; c.stroke();
    c.fillStyle=col; c.font='bold 10px monospace';
    c.fillText(ctx.exit_reason||'EXIT', xx+10, xy+4);
    const sign = (ctx.net_bps||0) >= 0 ? '+' : '';
    c.fillText(sign+(ctx.net_bps||0).toFixed(1)+'bp', xx+10, xy+16);
  }

  // ── Price line ────────────────────────────────────────────────────────────
  c.beginPath();
  prices.forEach((p, i) => {
    const x = toX(p.t + 7500), y = toY(p.mid);
    i === 0 ? c.moveTo(x,y) : c.lineTo(x,y);
  });
  c.strokeStyle='rgba(255,255,255,0.15)'; c.lineWidth=5; c.stroke();
  c.beginPath();
  prices.forEach((p, i) => {
    const x = toX(p.t + 7500), y = toY(p.mid);
    i === 0 ? c.moveTo(x,y) : c.lineTo(x,y);
  });
  c.strokeStyle='#ffffff'; c.lineWidth=1.5; c.stroke();

  // ── Time axis ─────────────────────────────────────────────────────────────
  c.font='9px monospace'; c.fillStyle='rgba(90,112,128,0.85)';
  const steps = 8;
  for (let i=0; i<=steps; i++) {
    const t = tMin + (tMax-tMin)*i/steps;
    const x = toX(t);
    c.fillText(fmtShort(t), x-14, H-3);
  }
}

// ── CVD chart ────────────────────────────────────────────────────────────────
function renderCVD(ctx) {
  const cvdData = ctx.cvd || [];
  const labels = cvdData.map(r => fmtShort(r.t));
  const deltas = cvdData.map(r => (r.buy-r.sell)/1e6);
  const cums   = cvdData.map(r => r.cum/1e6);

  if (cvdChart) {
    cvdChart.data.labels = labels;
    cvdChart.data.datasets[0].data = deltas;
    cvdChart.data.datasets[0].backgroundColor =
      deltas.map(v => v>=0 ? 'rgba(30,255,142,0.5)' : 'rgba(255,51,85,0.5)');
    cvdChart.data.datasets[1].data = cums;
    cvdChart.update('none'); return;
  }
  cvdChart = new Chart(document.getElementById('cvd-canvas'), {
    data: { labels,
      datasets: [
        { type:'bar',  label:'Delta', data:deltas,
          backgroundColor: deltas.map(v=>v>=0?'rgba(30,255,142,0.5)':'rgba(255,51,85,0.5)'),
          borderWidth:0, yAxisID:'y' },
        { type:'line', label:'Cum', data:cums,
          borderColor:'#ff8c00', borderWidth:1.5, pointRadius:0, tension:0.3, yAxisID:'y2' },
      ]},
    options: { animation:false, responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{
        x:{ticks:{color:'#5a7080',font:{size:8},maxTicksLimit:10,maxRotation:0},
           grid:{color:'rgba(26,37,53,0.5)'}},
        y:{position:'left',ticks:{color:'#5a7080',font:{size:8}},
           grid:{color:'rgba(26,37,53,0.3)'}},
        y2:{position:'right',ticks:{color:'#ff8c00',font:{size:8}},
            grid:{drawOnChartArea:false}},
      }}});
}

// ── info panel ───────────────────────────────────────────────────────────────
function renderInfo(ctx) {
  const set = (id, val, col) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = val ?? '—';
    if (col) el.style.color = col;
  };
  const net = ctx.net_bps || 0;
  const col = net > 0 ? '#1eff8e' : '#ff3355';
  set('inf-id',    ctx.trade_id);
  set('inf-entry', ctx.entry_price?.toFixed(2));
  set('inf-exit',  ctx.exit_price?.toFixed(2));
  set('inf-tp',    ctx.tp_price?.toFixed(2), '#1eff8e');
  set('inf-sl',    ctx.sl_price?.toFixed(2), '#ff3355');
  set('inf-be',    ctx.be_price?.toFixed(2), '#ffd060');
  set('inf-fee',   ctx.fee_bps?.toFixed(1) + ' bps');
  set('inf-adv',   ctx.entry_adv?.toFixed(2) + ' bps');
  const liq = ctx.liq_notional;
  set('inf-liq',   liq ? (liq >= 1e6 ? (liq/1e6).toFixed(2)+'M' : Math.round(liq/1000)+'K') : '—');
  set('inf-liqn',  ctx.liq_count ?? '—');

  // Header
  const sign = net >= 0 ? '+' : '';
  document.getElementById('hdr-outcome').textContent =
    (net > 0 ? '▲ WIN' : '▼ LOSS');
  document.getElementById('hdr-outcome').style.color = col;
  document.getElementById('hdr-exit').textContent  = ctx.exit_reason || '—';
  document.getElementById('hdr-net').textContent   = sign+net.toFixed(1)+' bps';
  document.getElementById('hdr-net').style.color   = col;
  const gross = ctx.gross_bps;
  document.getElementById('hdr-gross').textContent =
    gross != null ? (gross>=0?'+':'')+gross.toFixed(1)+' bps' : '—';
  document.getElementById('hdr-gross').style.color =
    gross != null ? (gross>=0?'#1eff8e':'#ff3355') : '#5a7080';
}

// ── load trade ────────────────────────────────────────────────────────────────
async function loadTrade(idx) {
  if (idx < 0 || idx >= trades.length) return;
  currentIdx = idx;
  renderList();
  document.getElementById('hdr-idx').textContent =
    (idx+1) + ' / ' + trades.length;

  const t = trades[idx];
  const res = await fetch('/api/context?id=' + encodeURIComponent(t.id));
  if (!res.ok) return;
  const ctx = await res.json();

  renderPrice(ctx);
  renderCVD(ctx);
  renderInfo(ctx);
}

// ── navigation ────────────────────────────────────────────────────────────────
document.getElementById('btn-prev').onclick = () => loadTrade(currentIdx + 1);
document.getElementById('btn-next').onclick = () => loadTrade(currentIdx - 1);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowDown')  loadTrade(currentIdx + 1);
  if (e.key === 'ArrowRight' || e.key === 'ArrowUp')    loadTrade(currentIdx - 1);
});

window.addEventListener('resize', () => {
  if (trades[currentIdx]) {
    fetch('/api/context?id=' + encodeURIComponent(trades[currentIdx].id))
      .then(r=>r.json()).then(ctx => { renderPrice(ctx); });
  }
});

// ── init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
  const res = await fetch('/api/trades');
  trades = await res.json();
  renderList();
  document.getElementById('hdr-idx').textContent = '1 / ' + trades.length;
  if (trades.length) loadTrade(0);
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path

        if path in ("/", "/index.html"):
            body = HTML.encode()
            self._send(200, "text/html; charset=utf-8", body)

        elif path == "/api/trades":
            data = json.dumps(trade_list(), separators=(",", ":")).encode()
            self._send(200, "application/json", data)

        elif path == "/api/context":
            qs  = parse_qs(parsed.query)
            tid = (qs.get("id") or [""])[0]
            ctx = trade_context(tid)
            if ctx is None:
                self.send_error(404)
                return
            data = json.dumps(ctx, separators=(",", ":")).encode()
            self._send(200, "application/json", data)

        else:
            self.send_error(404)

    def _send(self, code: int, ct: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",       default="127.0.0.1")
    ap.add_argument("--port",       type=int, default=5052)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not INTEL_DB.exists():
        print(f"ERROR: {INTEL_DB} not found")
        return
    if not MICRO_DB.exists():
        print(f"ERROR: {MICRO_DB} not found")
        return

    url = f"http://{args.host}:{args.port}"
    print(f"S34 Replay  ->  {url}")
    if not args.no_browser:
        webbrowser.open(url)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
