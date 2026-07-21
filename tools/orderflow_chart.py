"""
Bookmap-style order flow chart for ETH.
Runs on port 5051. Reads from microstructure.db + Binance REST depth.

Data:
  - Depth heatmap : Binance /fapi/v1/depth polled every 1s, memory only (no disk)
  - Volume bubbles: agg_trades table
  - CVD panel     : agg_trades table
  - Big trades    : agg_trades table (>$100K notional)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MICRO_DB = ROOT / "data" / "microstructure.db"
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth?symbol=ETHUSDT&limit=20"

# Rolling depth buffer: 1200 snapshots @ 1s = 20 minutes history
DEPTH_HISTORY: deque[dict[str, Any]] = deque(maxlen=1200)
DEPTH_LOCK = threading.Lock()
_LAST_PRICE: float = 0.0
_LAST_SPREAD: float = 0.0
_POLL_FAILURES: int = 0
_LAST_POLL_SUCCESS_MS: int = 0


# ---------------------------------------------------------------------------
# Depth polling thread
# ---------------------------------------------------------------------------

def _is_valid_depth(bids: list, asks: list, prev_price: float) -> bool:
    """Reject malformed or implausible snapshots before writing to history."""
    if not bids or not asks:
        return False
    bid0, ask0 = bids[0][0], asks[0][0]
    if bid0 <= 0 or ask0 <= 0:
        return False
    if ask0 <= bid0:                          # inverted book
        return False
    if (ask0 - bid0) > 10.0:                 # spread > $10 = bad data
        return False
    if any(q <= 0 for _, q in bids[:5]):     # zero-qty top-of-book
        return False
    if prev_price > 0:
        mid = (bid0 + ask0) / 2.0
        if abs(mid - prev_price) / prev_price > 0.03:  # >3% jump = stale/bad
            return False
    return True


def _poll_depth() -> None:
    global _LAST_PRICE, _LAST_SPREAD, _POLL_FAILURES, _LAST_POLL_SUCCESS_MS
    backoff = 1.0
    while True:
        try:
            with urllib.request.urlopen(DEPTH_URL, timeout=4) as resp:
                data = json.loads(resp.read())
            bids = [[float(p), float(q)] for p, q in (data.get("bids") or [])]
            asks = [[float(p), float(q)] for p, q in (data.get("asks") or [])]
            if _is_valid_depth(bids, asks, _LAST_PRICE):
                ts = int(time.time() * 1000)
                _LAST_PRICE = (bids[0][0] + asks[0][0]) / 2.0
                _LAST_SPREAD = asks[0][0] - bids[0][0]
                with DEPTH_LOCK:
                    DEPTH_HISTORY.append({"ts": ts, "bids": bids, "asks": asks})
                _POLL_FAILURES = 0
                _LAST_POLL_SUCCESS_MS = ts
                backoff = 1.0
            else:
                _POLL_FAILURES += 1
        except Exception:
            _POLL_FAILURES += 1
            backoff = min(backoff * 1.5, 10.0)  # cap at 10s between retries
        time.sleep(backoff)


def start_depth_poll() -> None:
    t = threading.Thread(target=_poll_depth, daemon=True, name="depth-poll")
    t.start()


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MICRO_DB), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def cvd_series(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT
            (ts_ms / 60000) * 60000 AS bucket_ms,
            SUM(CASE WHEN is_buyer_maker = 0 THEN notional ELSE 0 END) AS buy_n,
            SUM(CASE WHEN is_buyer_maker = 1 THEN notional ELSE 0 END) AS sell_n
        FROM agg_trades
        WHERE symbol = 'ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?
        GROUP BY bucket_ms ORDER BY bucket_ms ASC
    """, (start_ms, end_ms)).fetchall()
    result: list[dict[str, Any]] = []
    cum = 0.0
    for r in rows:
        buy = float(r["buy_n"] or 0)
        sell = float(r["sell_n"] or 0)
        delta = buy - sell
        cum += delta
        result.append({"ts": int(r["bucket_ms"]), "buy": buy, "sell": sell,
                        "delta": delta, "cum": cum})
    return result


def big_trades(conn: sqlite3.Connection, start_ms: int, limit: int = 60) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT ts_ms, price, notional, is_buyer_maker
        FROM agg_trades
        WHERE symbol = 'ETHUSDT' AND notional >= 100000 AND ts_ms >= ?
        ORDER BY ts_ms DESC LIMIT ?
    """, (start_ms, limit)).fetchall()
    return [{"ts": int(r["ts_ms"]), "price": float(r["price"]),
             "notional": float(r["notional"]),
             "side": "BUY" if not r["is_buyer_maker"] else "SELL"} for r in rows]


def vol_bubbles(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT
            (ts_ms / 60000) * 60000 AS bucket_ms,
            AVG(price) AS avg_price,
            SUM(CASE WHEN is_buyer_maker = 0 THEN notional ELSE 0 END) AS buy_n,
            SUM(CASE WHEN is_buyer_maker = 1 THEN notional ELSE 0 END) AS sell_n
        FROM agg_trades
        WHERE symbol = 'ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?
        GROUP BY bucket_ms ORDER BY bucket_ms ASC
    """, (start_ms, end_ms)).fetchall()
    return [{"ts": int(r["bucket_ms"]), "price": float(r["avg_price"]),
             "buy": float(r["buy_n"] or 0), "sell": float(r["sell_n"] or 0)} for r in rows]


def liquidation_events(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    try:
        rows = conn.execute("""
            SELECT ts_ms, price, notional, side
            FROM liquidations
            WHERE symbol = 'ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?
            ORDER BY ts_ms ASC
        """, (start_ms, end_ms)).fetchall()
        return [{"ts": int(r["ts_ms"]), "price": float(r["price"]),
                 "notional": float(r["notional"]), "side": str(r["side"])} for r in rows]
    except Exception:
        return []


def s34_signals(start_ms: int) -> list[dict[str, Any]]:
    try:
        conn = sqlite3.connect(str(INTEL_DB), timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT d.signal_ts_ms, d.rule_name,
                   t.entry_price, t.net_bps, t.exit_reason, t.exit_ts_ms
            FROM s34_decisions d
            LEFT JOIN s34_trades t ON t.signal_id = d.signal_id
            WHERE d.decision = 'ACCEPT' AND d.signal_ts_ms >= ?
            ORDER BY d.signal_ts_ms ASC LIMIT 100
        """, (start_ms,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def build_payload() -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    start_1h = now_ms - 3_600_000
    start_20m = now_ms - 1_200_000

    cvd: list[dict] = []
    trades: list[dict] = []
    bubbles: list[dict] = []
    liqs: list[dict] = []

    try:
        conn = _connect()
        cvd = cvd_series(conn, start_1h, now_ms)
        trades = big_trades(conn, start_20m)
        bubbles = vol_bubbles(conn, start_20m, now_ms)
        liqs = liquidation_events(conn, start_20m, now_ms)
        conn.close()
    except Exception:
        pass

    with DEPTH_LOCK:
        history = list(DEPTH_HISTORY)
    # Keep last 500 snapshots raw (no thinning — ~8 min at 1s)
    thinned = history[-500:] if len(history) > 500 else history

    return {
        "ts": now_ms,
        "price": _LAST_PRICE,
        "spread": _LAST_SPREAD,
        "depth_history": thinned,
        "cvd": cvd,
        "big_trades": trades,
        "vol_bubbles": bubbles,
        "liquidations": liqs,
        "signals": s34_signals(start_20m),
        "data_age_ms": now_ms - _LAST_POLL_SUCCESS_MS if _LAST_POLL_SUCCESS_MS else 99999,
        "poll_errors": _POLL_FAILURES,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Order Flow — ETH</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #060a0f;
  --panel: #0c1420;
  --border: #1a2535;
  --ink: #d0dce8;
  --muted: #5a7080;
  --green: #1eff8e;
  --red: #ff3355;
  --gold: #ffd060;
  --blue: #2ec4ff;
  --orange: #ff8c00;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  height: 100%;
  max-height: 100vh;
}
body {
  background: var(--bg);
  color: var(--ink);
  font: 12px/1.4 Consolas, "SFMono-Regular", monospace;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ---- header ---- */
#header {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 6px 14px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  font-size: 12px;
}
#header .sym { font-size: 14px; font-weight: bold; color: var(--blue); }
#header .price { font-size: 18px; font-weight: bold; color: var(--gold); }
#header .stat { color: var(--muted); }
#header .stat span { color: var(--ink); margin-left: 4px; }
#ts { margin-left: auto; color: var(--muted); font-size: 11px; }
#hdr-status { font-size: 16px; cursor: default; }
.st-ok    { color: #1eff8e; }
.st-stale { color: #ffd060; }
.st-dead  { color: #ff3355; animation: blink 0.9s step-start infinite; }
@keyframes blink { 50% { opacity: 0.2; } }

/* dead-feed banner */
#conn-banner {
  display: none;
  position: fixed;
  top: 34px; left: 0; right: 0;
  background: rgba(255,51,85,0.12);
  border-bottom: 1px solid rgba(255,51,85,0.5);
  padding: 3px 0;
  text-align: center;
  font-size: 11px;
  color: #ff3355;
  letter-spacing: 0.06em;
  z-index: 200;
  pointer-events: none;
}

/* ---- main layout ---- */
#body {
  display: flex;
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

/* ---- heatmap area ---- */
#heatmap-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  min-height: 0;
  min-width: 0;
}
#heatmap {
  flex: 1;
  width: 100%;
  display: block;
  cursor: crosshair;
  min-height: 0;
}
#heatmap-labels {
  position: absolute;
  top: 0; right: 0;
  width: 52px;
  height: 100%;
  pointer-events: none;
}
#heatmap-tooltip {
  position: absolute;
  display: none;
  background: rgba(10,14,20,0.92);
  border: 1px solid var(--border);
  padding: 5px 8px;
  font-size: 11px;
  color: var(--ink);
  pointer-events: none;
  white-space: nowrap;
  z-index: 10;
}
#heatmap-xhair {
  position: absolute;
  top: 0; left: 0;
  pointer-events: none;
  display: block;
  z-index: 5;
}

/* ---- CVD panel ---- */
#cvd-wrap {
  height: 100px;
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  position: relative;
  min-height: 80px;
}
#cvd-label {
  position: absolute;
  top: 4px; left: 8px;
  font-size: 10px;
  color: var(--muted);
  pointer-events: none;
  z-index: 1;
}
#cvd { width: 100%; height: 100%; display: block; }

/* ---- right sidebar ---- */
#sidebar {
  width: 220px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  border-left: 1px solid var(--border);
  overflow: hidden;
  min-height: 0;
}

/* DOM */
#dom-wrap {
  flex: 1;
  overflow-y: auto;
  scrollbar-width: none;
  cursor: grab;
  user-select: none;
  min-height: 0;
}
#dom-wrap::-webkit-scrollbar { display: none; }
#dom-wrap.grabbing { cursor: grabbing; }
.dom-header {
  display: grid;
  grid-template-columns: 1fr 80px 1fr;
  padding: 4px 8px;
  font-size: 10px;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  position: sticky; top: 0; z-index: 1;
  text-align: center;
}
.dom-row {
  display: grid;
  grid-template-columns: 1fr 80px 1fr;
  padding: 1px 0;
  font-size: 11px;
  position: relative;
  height: 18px;
  align-items: center;
}
.dom-row .bar-bid {
  position: absolute;
  right: calc(80px);
  top: 1px; bottom: 1px;
  background: rgba(30, 255, 142, 0.12);
  transition: width 0.3s;
}
.dom-row .bar-ask {
  position: absolute;
  left: calc(80px);
  top: 1px; bottom: 1px;
  background: rgba(255, 51, 85, 0.12);
  transition: width 0.3s;
}
.dom-row .bid-qty { text-align: right; padding-right: 6px; color: var(--green); position: relative; z-index: 1; }
.dom-row .ask-qty { padding-left: 6px; color: var(--red); position: relative; z-index: 1; }
.dom-row .price-cell {
  text-align: center;
  font-weight: bold;
  position: relative;
  z-index: 1;
  color: var(--ink);
  font-size: 11px;
}
.dom-row.mid .price-cell { color: var(--gold); background: rgba(255,208,96,0.08); }

/* Big trades tape */
#tape-wrap {
  height: 200px;
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  overflow-y: auto;
  scrollbar-width: none;
}
#tape-wrap::-webkit-scrollbar { display: none; }
.tape-header {
  padding: 4px 8px;
  font-size: 10px;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  position: sticky; top: 0;
  text-align: center;
}
.tape-row {
  display: grid;
  grid-template-columns: 42px 1fr 60px;
  padding: 2px 8px;
  font-size: 11px;
  border-bottom: 1px solid rgba(26,37,53,0.5);
}
.tape-row .t-side { font-weight: bold; }
.tape-row .t-side.buy { color: var(--green); }
.tape-row .t-side.sell { color: var(--red); }
.tape-row .t-price { text-align: center; }
.tape-row .t-size { text-align: right; color: var(--gold); }
.tape-row.liq { background: rgba(255,140,0,0.08); }

/* Liq flash */
@keyframes liqFlash {
  0% { background: rgba(255,140,0,0.35); }
  100% { background: transparent; }
}
.tape-row.liq-new { animation: liqFlash 1.5s ease-out forwards; }
</style>
</head>
<body>

<div id="conn-banner">⚠ DEPTH FEED DEAD</div>
<div id="header">
  <div id="hdr-status" class="st-ok" title="Feed status">●</div>
  <div class="sym">ETHUSDT</div>
  <div class="price" id="hdr-price">—</div>
  <div class="stat">SPREAD <span id="hdr-spread">—</span></div>
  <div class="stat">CVD 1H <span id="hdr-cvd">—</span></div>
  <div class="stat">BIG TRADES 20M <span id="hdr-bigtrades">—</span></div>
  <div class="stat">IMBAL <span id="hdr-imbalance">—</span></div>
  <div id="ts">—</div>
</div>

<div id="body">

  <div id="heatmap-wrap">
    <canvas id="heatmap"></canvas>
    <canvas id="heatmap-labels"></canvas>
    <canvas id="heatmap-xhair"></canvas>
    <div id="heatmap-tooltip"></div>
    <div id="cvd-wrap">
      <div id="cvd-label">CUMULATIVE DELTA (1H)</div>
      <canvas id="cvd"></canvas>
    </div>
  </div>

  <div id="sidebar">
    <div id="dom-wrap">
      <div class="dom-header">
        <span>BID</span><span>PRICE</span><span>ASK</span>
      </div>
      <div id="dom-rows"></div>
    </div>
    <div id="tape-wrap">
      <div class="tape-header">BIG TRADES &amp; LIQS (&gt;$100K)</div>
      <div id="tape-rows"></div>
    </div>
  </div>

</div>

<script>
// ── constants ──────────────────────────────────────────────────────────────
const PRICE_BUCKET = 0.01;      // $0.01 per row — matches order book tick
const POLL_MS = 1000;

// ── state ──────────────────────────────────────────────────────────────────
let cvdChart = null;
let lastPayload = null;
let prevLiqTs = 0;
let priceZoom = 1.0;     // >1 zoomed in, <1 zoomed out
let _hm = null;          // heatmap metadata (set each render, used by crosshair)
let xhairX = null, xhairY = null;
let _pollInFlight = false;
let _zoomTimer = null;
const MAX_ROWS = 4000;   // crash guard: cap numRows so canvas doesn't OOM

// ── heat color (rgba) ──────────────────────────────────────────────────────
function heatRGBA(intensity, ageFactor) {
  // ageFactor: 1.0 = newest, 0.2 = oldest
  let r, g, b, a;
  if (intensity <= 0.001) { return null; } // transparent — no order here
  if (intensity < 0.06)   { r=13;  g=30;  b=60;  a=0.55; }
  else if (intensity < 0.15) { r=10;  g=70;  b=130; a=0.70; }
  else if (intensity < 0.28) { r=8;   g=110; b=140; a=0.80; }
  else if (intensity < 0.45) { r=10;  g=150; b=80;  a=0.85; }
  else if (intensity < 0.62) { r=200; g=160; b=0;   a=0.90; }
  else if (intensity < 0.80) { r=210; g=80;  b=0;   a=0.95; }
  else                        { r=200; g=15;  b=15;  a=1.00; }
  return `rgba(${r},${g},${b},${(a * ageFactor).toFixed(2)})`;
}

// ── heatmap rendering ──────────────────────────────────────────────────────
const PROFILE_W = 48; // width of volume profile strip on right

function renderHeatmap(payload) {
  const canvas = document.getElementById('heatmap');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  if (W === 0 || H === 0) return;
  canvas.width = W;
  canvas.height = H;

  const price = payload.price || 0;
  if (!price) return;

  const history = payload.depth_history || [];
  if (!history.length) {
    ctx.fillStyle = '#060a0f';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#5a7080';
    ctx.font = '13px monospace';
    ctx.fillText('Waiting for depth data…', W/2 - 90, H/2);
    return;
  }

  const CHART_W = W - PROFILE_W;

  // Dynamic price range from actual depth data
  let dMin = Infinity, dMax = -Infinity;
  for (const snap of history) {
    for (const [p] of (snap.bids||[])) { if (p < dMin) dMin = p; if (p > dMax) dMax = p; }
    for (const [p] of (snap.asks||[])) { if (p < dMin) dMin = p; if (p > dMax) dMax = p; }
  }
  if (!isFinite(dMin)) { dMin = price - 1; dMax = price + 1; }
  const buf = Math.max((dMax - dMin) * 0.15, 0.2);
  const pRange = (dMax + buf) - (dMin - buf);

  // Apply zoom — centered on current price
  let zoomedRange = pRange / priceZoom;
  const zPriceMin = price - zoomedRange * 0.5;
  const zPriceMax = price + zoomedRange * 0.5;
  let numRows = Math.round(zoomedRange / PRICE_BUCKET) + 1;
  // Crash guard: if zoom-out creates too many rows, silently clamp
  if (numRows > MAX_ROWS) {
    numRows = MAX_ROWS;
    zoomedRange = numRows * PRICE_BUCKET;
    priceZoom = pRange / zoomedRange;
  }
  const rowH = H / numRows;

  // Fixed column width: 3px.
  const COL_W = 3;
  const cols = history.length;
  const visibleCols = Math.min(cols, Math.floor(CHART_W / COL_W));
  const startIdx = cols - visibleCols;
  const colW = COL_W;

  // 95th-percentile max for smooth heat scale
  const allQtys = history.flatMap(snap =>
    [...(snap.bids||[]), ...(snap.asks||[])].map(([,q])=>q)
  ).sort((a,b)=>a-b);
  const p95idx = Math.floor(allQtys.length * 0.95);
  const maxQty = allQtys[p95idx] || 1;

  // Draw background
  ctx.fillStyle = '#060a0f';
  ctx.fillRect(0, 0, W, H);

  if (visibleCols < 30) {
    ctx.fillStyle = 'rgba(90,112,128,0.7)';
    ctx.font = '13px monospace';
    ctx.fillText(`Building history… (${cols}/500 snapshots)`, W/2 - 130, H/2);
  }

  // Draw each snapshot column
  history.slice(startIdx).forEach((snap, ci) => {
    const x = ci * colW;
    const ageFactor = 0.45 + 0.55 * (ci / Math.max(visibleCols - 1, 1));

    const qMap = new Map();
    for (const [p, q] of (snap.bids || [])) {
      const b = Math.round(p / PRICE_BUCKET) * PRICE_BUCKET;
      qMap.set(b.toFixed(2), (qMap.get(b.toFixed(2))||0) + q);
    }
    for (const [p, q] of (snap.asks || [])) {
      const b = Math.round(p / PRICE_BUCKET) * PRICE_BUCKET;
      qMap.set(b.toFixed(2), (qMap.get(b.toFixed(2))||0) + q);
    }

    for (let row = 0; row < numRows; row++) {
      const rowPrice = zPriceMax - row * PRICE_BUCKET;
      const key = (Math.round(rowPrice / PRICE_BUCKET) * PRICE_BUCKET).toFixed(2);
      const qty = qMap.get(key) || 0;
      if (qty === 0) continue;
      const intensity = Math.min(qty / maxQty, 1);
      const color = heatRGBA(intensity, ageFactor);
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(
        Math.floor(x) - 0.5,
        Math.floor(row * rowH) - 0.5,
        Math.ceil(colW) + 1.5,
        Math.ceil(rowH) + 1.5
      );
    }
  });

  // Time→pixel helper
  const visSnaps = history.slice(startIdx);
  const tsMin = visSnaps[0]?.ts || 0;
  const tsMax = visSnaps[visSnaps.length-1]?.ts || (tsMin + 1);
  const tRange = Math.max(tsMax - tsMin, 1);
  const tsToX = ts => ((ts - tsMin) / tRange) * (visibleCols * colW);

  // Store metadata for crosshair rendering (called separately on mousemove)
  _hm = { zPriceMax, zoomedRange, H, CHART_W, W, visSnaps, colW };

  // ── Volume profile strip ─────────────────────────────────────────────────
  const PROF_BUCKET = 0.5;  // $0.5 coarse buckets so bars are visible
  const profBuckets = Math.round(zoomedRange / PROF_BUCKET) + 1;
  const profRowH = H / profBuckets;
  const profMap = new Map();
  for (const snap of visSnaps) {
    for (const [p, q] of [...(snap.bids||[]), ...(snap.asks||[])]) {
      const row = Math.round((zPriceMax - p) / PROF_BUCKET);
      if (row < 0 || row >= profBuckets) continue;
      profMap.set(row, (profMap.get(row)||0) + q);
    }
  }
  const maxProf = profMap.size ? Math.max(...profMap.values()) : 1;
  ctx.fillStyle = 'rgba(6,10,15,0.85)';
  ctx.fillRect(CHART_W, 0, PROFILE_W, H);
  ctx.strokeStyle = 'rgba(26,37,53,0.9)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(CHART_W, 0); ctx.lineTo(CHART_W, H); ctx.stroke();
  for (const [row, qty] of profMap) {
    const ry = row * profRowH;
    const rowPrice = zPriceMax - row * PROF_BUCKET;
    const isBid = rowPrice <= price;
    const barW = (qty / maxProf) * (PROFILE_W - 4);
    ctx.fillStyle = isBid ? 'rgba(30,255,142,0.45)' : 'rgba(255,51,85,0.45)';
    ctx.fillRect(CHART_W + 2, ry, barW, Math.max(profRowH - 0.5, 1));
  }
  ctx.fillStyle = 'rgba(90,112,128,0.55)';
  ctx.font = '8px monospace';
  ctx.fillText('VOL', CHART_W + 16, 9);

  // ── Volume bubbles ───────────────────────────────────────────────────────
  const bubbles = payload.vol_bubbles || [];
  let maxBubble = 1;
  for (const b of bubbles) maxBubble = Math.max(maxBubble, b.buy + b.sell);
  for (const b of bubbles) {
    const total = b.buy + b.sell;
    if (total < 50000) continue;
    const bx = tsToX(b.ts);
    const by = ((zPriceMax - b.price) / zoomedRange) * H;
    if (by < -20 || by > H + 20) continue;
    const r = Math.max(3, Math.min(16, (total / maxBubble) * 24));
    const isBuy = b.buy >= b.sell;
    ctx.beginPath();
    ctx.arc(bx, by, r, 0, Math.PI * 2);
    ctx.fillStyle = isBuy ? 'rgba(30,255,142,0.30)' : 'rgba(255,51,85,0.30)';
    ctx.fill();
    ctx.strokeStyle = isBuy ? 'rgba(30,255,142,0.85)' : 'rgba(255,51,85,0.85)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ── Liquidation markers ──────────────────────────────────────────────────
  for (const liq of (payload.liquidations || [])) {
    const lx = tsToX(liq.ts);
    const ly = ((zPriceMax - liq.price) / zoomedRange) * H;
    if (ly < -20 || ly > H + 20) continue;
    const liqR = Math.max(5, Math.min(18, liq.notional / 40000));
    ctx.beginPath();
    ctx.arc(lx, ly, liqR, 0, Math.PI * 2);
    ctx.fillStyle = liq.side === 'BUY' ? 'rgba(255,140,0,0.65)' : 'rgba(180,0,255,0.65)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // ── S34 signal overlay ───────────────────────────────────────────────────
  for (const sig of (payload.signals || [])) {
    const sx = tsToX(sig.signal_ts_ms);
    if (sx < 0 || sx > CHART_W) continue;
    const ep = sig.entry_price;
    if (!ep) continue;
    const sy = ((zPriceMax - ep) / zoomedRange) * H;
    if (sy < 0 || sy > H) continue;
    const hasResult = sig.net_bps !== null && sig.net_bps !== undefined;
    const col = !hasResult ? '#ffd060' : (sig.net_bps > 0 ? '#1eff8e' : '#ff3355');
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(sx, sy - 2);
    ctx.lineTo(sx - 7, sy + 13);
    ctx.lineTo(sx + 7, sy + 13);
    ctx.closePath();
    ctx.fillStyle = col + 'bb';
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
    if (hasResult) {
      ctx.save();
      ctx.fillStyle = col;
      ctx.font = 'bold 9px monospace';
      ctx.fillText((sig.net_bps >= 0 ? '+' : '') + sig.net_bps.toFixed(1), sx + 9, sy + 9);
      ctx.restore();
    }
  }

  // ── Price line (smooth bezier) ───────────────────────────────────────────
  const midpoints = visSnaps.map((snap, ci) => {
    const bid0 = snap.bids?.[0]?.[0];
    const ask0 = snap.asks?.[0]?.[0];
    if (!bid0 || !ask0) return null;
    return [ci * colW + colW / 2, ((zPriceMax - (bid0 + ask0) / 2) / zoomedRange) * H];
  }).filter(Boolean);

  if (midpoints.length > 1) {
    const drawLine = (width, color) => {
      ctx.beginPath();
      ctx.moveTo(midpoints[0][0], midpoints[0][1]);
      for (let i = 1; i < midpoints.length - 1; i++) {
        const cpx = (midpoints[i][0] + midpoints[i+1][0]) / 2;
        const cpy = (midpoints[i][1] + midpoints[i+1][1]) / 2;
        ctx.quadraticCurveTo(midpoints[i][0], midpoints[i][1], cpx, cpy);
      }
      ctx.lineTo(midpoints[midpoints.length-1][0], midpoints[midpoints.length-1][1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    };
    drawLine(6, 'rgba(255,255,255,0.15)');
    drawLine(2.5, 'rgba(255,255,255,0.5)');
    drawLine(1.5, '#ffffff');
  }

  // ── Current price horizontal line ────────────────────────────────────────
  const priceY = ((zPriceMax - price) / zoomedRange) * H;
  if (priceY >= 0 && priceY <= H) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(255,208,96,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, priceY);
    ctx.lineTo(CHART_W - 2, priceY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ── Price axis labels canvas ─────────────────────────────────────────────
  const lblCanvas = document.getElementById('heatmap-labels');
  lblCanvas.width = 52;
  lblCanvas.height = H;
  lblCanvas.style.cssText = `position:absolute;top:0;right:${PROFILE_W}px;pointer-events:none;`;
  const lctx = lblCanvas.getContext('2d');
  lctx.clearRect(0, 0, 52, H);
  lctx.font = '10px monospace';
  const labelEvery = Math.max(1, Math.round(0.5 / PRICE_BUCKET));
  for (let row = 0; row < numRows; row += labelEvery) {
    const rowPrice = zPriceMax - row * PRICE_BUCKET;
    const ry = row * rowH;
    const isMid = Math.abs(rowPrice - price) < PRICE_BUCKET * 1.5;
    lctx.fillStyle = isMid ? '#ffd060' : 'rgba(90,112,128,0.85)';
    lctx.fillText(rowPrice.toFixed(2), 2, ry + 9);
  }

  // ── Time axis labels ─────────────────────────────────────────────────────
  ctx.font = '9px monospace';
  ctx.fillStyle = 'rgba(90,112,128,0.85)';
  const timeSteps = 7;
  for (let i = 0; i <= timeSteps; i++) {
    const ci = Math.round((i / timeSteps) * (visibleCols - 1));
    const snap = visSnaps[ci];
    if (!snap) continue;
    const dt = new Date(snap.ts);
    const label = dt.getUTCHours().toString().padStart(2,'0') + ':' +
                  dt.getUTCMinutes().toString().padStart(2,'0');
    ctx.fillText(label, ci * colW, H - 3);
  }

  // ── Zoom indicator (top-left when zoomed) ────────────────────────────────
  if (priceZoom !== 1.0) {
    ctx.fillStyle = 'rgba(90,112,128,0.65)';
    ctx.font = '10px monospace';
    ctx.fillText(`×${priceZoom.toFixed(1)}`, 6, 14);
  }
}

// ── CVD chart ──────────────────────────────────────────────────────────────
function renderCVD(payload) {
  const cvdData = payload.cvd || [];
  if (!cvdData.length) return;

  const labels = cvdData.map(r => {
    const d = new Date(r.ts);
    return d.getUTCHours().toString().padStart(2,'0') + ':' + d.getUTCMinutes().toString().padStart(2,'0');
  });
  const deltas = cvdData.map(r => r.delta / 1e6);
  const cums = cvdData.map(r => r.cum / 1e6);

  const canvas = document.getElementById('cvd');
  if (cvdChart) {
    cvdChart.data.labels = labels;
    cvdChart.data.datasets[0].data = deltas;
    cvdChart.data.datasets[1].data = cums;
    cvdChart.update('none');
    return;
  }

  cvdChart = new Chart(canvas, {
    data: {
      labels,
      datasets: [
        {
          type: 'bar',
          label: 'Delta',
          data: deltas,
          backgroundColor: deltas.map(v => v >= 0 ? 'rgba(30,255,142,0.5)' : 'rgba(255,51,85,0.5)'),
          borderWidth: 0,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Cumulative',
          data: cums,
          borderColor: '#ff8c00',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          yAxisID: 'y2',
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: {
          ticks: { color: '#5a7080', font: { size: 9 }, maxTicksLimit: 12,
                   maxRotation: 0 },
          grid: { color: 'rgba(26,37,53,0.6)' },
        },
        y: {
          position: 'left',
          ticks: { color: '#5a7080', font: { size: 9 } },
          grid: { color: 'rgba(26,37,53,0.4)' },
        },
        y2: {
          position: 'right',
          ticks: { color: '#ff8c00', font: { size: 9 } },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

// ── DOM ────────────────────────────────────────────────────────────────────
function renderDOM(payload) {
  const history = payload.depth_history || [];
  if (!history.length) return;

  const latest = history[history.length - 1];
  const bids = latest.bids || [];
  const asks = latest.asks || [];
  const mid = payload.price || 0;

  const allQtys = [...bids.map(b=>b[1]), ...asks.map(a=>a[1])];
  const maxQ = Math.max(...allQtys, 1);

  const rows = [];
  // asks (reversed so highest ask is at top, lowest ask closest to mid)
  const asksDisplay = [...asks].reverse();
  for (const [p, q] of asksDisplay) {
    const pct = (q / maxQ * 100).toFixed(0);
    rows.push(`<div class="dom-row">
      <div class="bar-bid"></div>
      <div class="bar-ask" style="width:${pct}%"></div>
      <div class="bid-qty"></div>
      <div class="price-cell" style="color:var(--red)">${p.toFixed(2)}</div>
      <div class="ask-qty">${q.toFixed(2)}</div>
    </div>`);
  }
  // mid line
  rows.push(`<div class="dom-row mid">
    <div class="bid-qty"></div>
    <div class="price-cell">${mid.toFixed(2)}</div>
    <div class="ask-qty"></div>
  </div>`);
  // bids
  for (const [p, q] of bids) {
    const pct = (q / maxQ * 100).toFixed(0);
    rows.push(`<div class="dom-row">
      <div class="bar-bid" style="width:${pct}%"></div>
      <div class="bar-ask"></div>
      <div class="bid-qty">${q.toFixed(2)}</div>
      <div class="price-cell" style="color:var(--green)">${p.toFixed(2)}</div>
      <div class="ask-qty"></div>
    </div>`);
  }

  document.getElementById('dom-rows').innerHTML = rows.join('');
}

// ── tape ───────────────────────────────────────────────────────────────────
function renderTape(payload) {
  const trades = payload.big_trades || [];
  const liqs = payload.liquidations || [];

  // merge & sort by ts desc
  const all = [
    ...trades.map(t => ({...t, kind: 'trade'})),
    ...liqs.map(l => ({...l, side: l.side === 'BUY' ? 'BUY_LIQ' : 'SELL_LIQ', kind: 'liq'})),
  ].sort((a, b) => b.ts - a.ts).slice(0, 60);

  const rows = all.map(item => {
    const dt = new Date(item.ts);
    const hm = dt.getUTCHours().toString().padStart(2,'0') + ':' +
               dt.getUTCMinutes().toString().padStart(2,'0') + ':' +
               dt.getUTCSeconds().toString().padStart(2,'0');
    const isBuy = item.side === 'BUY' || item.side === 'BUY_LIQ';
    const isLiq = item.kind === 'liq';
    const sideLabel = isLiq ? (isBuy ? '⚡BUY' : '⚡SEL') : (isBuy ? 'BUY' : 'SEL');
    const cls = isLiq ? 'tape-row liq' : 'tape-row';
    const m = item.notional >= 1e6 ? (item.notional/1e6).toFixed(1)+'M' : Math.round(item.notional/1000)+'K';
    return `<div class="${cls}">
      <div class="t-side ${isBuy?'buy':'sell'}">${sideLabel}</div>
      <div class="t-price">${item.price.toFixed(1)}</div>
      <div class="t-size">$${m}</div>
    </div>`;
  });

  document.getElementById('tape-rows').innerHTML = rows.join('');
}

// ── header ─────────────────────────────────────────────────────────────────
function renderHeader(payload) {
  const p = payload.price || 0;
  document.getElementById('hdr-price').textContent = p ? '$' + p.toFixed(2) : '—';
  const sp = payload.spread;
  document.getElementById('hdr-spread').textContent = sp ? sp.toFixed(2) : '—';

  const cvd = payload.cvd || [];
  if (cvd.length) {
    const cum = cvd[cvd.length-1].cum;
    const sign = cum >= 0 ? '+' : '';
    const col = cum >= 0 ? '#1eff8e' : '#ff3355';
    document.getElementById('hdr-cvd').innerHTML =
      `<span style="color:${col}">${sign}${(cum/1e6).toFixed(1)}M</span>`;
  }

  const bt = payload.big_trades || [];
  document.getElementById('hdr-bigtrades').textContent = bt.length + ' trades';

  const latest = (payload.depth_history||[]).slice(-1)[0];
  if (latest) {
    const bidQ = (latest.bids||[]).reduce((s,[,q])=>s+q, 0);
    const askQ = (latest.asks||[]).reduce((s,[,q])=>s+q, 0);
    const total = bidQ + askQ || 1;
    const bidPct = Math.round(bidQ / total * 100);
    const col = bidPct > 52 ? '#1eff8e' : bidPct < 48 ? '#ff3355' : '#d0dce8';
    document.getElementById('hdr-imbalance').innerHTML =
      `<span style="color:${col}">B${bidPct}% / A${100-bidPct}%</span>`;
  }

  // Feed freshness indicator
  const age = payload.data_age_ms ?? 99999;
  const errs = payload.poll_errors ?? 0;
  const statusEl = document.getElementById('hdr-status');
  const banner   = document.getElementById('conn-banner');
  if (age < 4000) {
    statusEl.className = 'st-ok';
    statusEl.title = `Live  (err=${errs})`;
  } else if (age < 12000) {
    statusEl.className = 'st-stale';
    statusEl.title = `Stale ${(age/1000).toFixed(0)}s  (err=${errs})`;
  } else {
    statusEl.className = 'st-dead';
    statusEl.title = `DEAD ${(age/1000).toFixed(0)}s`;
  }
  if (banner) banner.style.display = age > 12000 ? 'block' : 'none';

  document.getElementById('ts').textContent =
    'updated ' + new Date(payload.ts||Date.now()).toUTCString().slice(17,25) + ' UTC';
}

// ── poll loop ──────────────────────────────────────────────────────────────
async function poll() {
  if (!_pollInFlight) {
    _pollInFlight = true;
    try {
      const res = await fetch('/api/orderflow', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      // Sanity: ignore payloads with clearly bad price
      if (payload.price > 0) {
        lastPayload = payload;
        renderHeader(payload);
        renderHeatmap(payload);
        renderCVD(payload);
        renderDOM(payload);
        renderTape(payload);
        if (cvdChart) {
          const deltas = cvdChart.data.datasets[0].data;
          cvdChart.data.datasets[0].backgroundColor =
            deltas.map(v => v >= 0 ? 'rgba(30,255,142,0.5)' : 'rgba(255,51,85,0.5)');
          cvdChart.update('none');
        }
      }
    } catch (e) {
      console.warn('poll error', e);
    } finally {
      _pollInFlight = false;
    }
  }
  setTimeout(poll, POLL_MS);
}

// ── DOM drag scroll ────────────────────────────────────────────────────────
(function() {
  const el = document.getElementById('dom-wrap');
  let drag = false, startY = 0, scrollStart = 0;

  el.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    drag = true;
    startY = e.clientY;
    scrollStart = el.scrollTop;
    el.classList.add('grabbing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!drag) return;
    el.scrollTop = scrollStart - (e.clientY - startY);
  });
  document.addEventListener('mouseup', () => {
    drag = false;
    el.classList.remove('grabbing');
  });

  // also drag-scroll the tape
  const tape = document.getElementById('tape-wrap');
  let tapeDrag = false, tapeStartY = 0, tapeScrollStart = 0;
  tape.style.cursor = 'grab';
  tape.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    tapeDrag = true;
    tapeStartY = e.clientY;
    tapeScrollStart = tape.scrollTop;
    tape.style.cursor = 'grabbing';
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!tapeDrag) return;
    tape.scrollTop = tapeScrollStart - (e.clientY - tapeStartY);
  });
  document.addEventListener('mouseup', () => {
    tapeDrag = false;
    tape.style.cursor = 'grab';
  });
})();

// ── crosshair overlay ─────────────────────────────────────────────────────
function renderCrosshair() {
  const hCanvas = document.getElementById('heatmap');
  const xCanvas = document.getElementById('heatmap-xhair');
  if (!xCanvas || !_hm) return;
  const W = hCanvas.width;
  const H = hCanvas.height;
  if (!W || !H) return;
  xCanvas.width = W;
  xCanvas.height = H;
  xCanvas.style.width = W + 'px';
  xCanvas.style.height = H + 'px';
  const ctx = xCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (xhairX === null) return;

  const { zPriceMax, zoomedRange, CHART_W } = _hm;
  const px = xhairX, py = xhairY;
  const hoverPrice = zPriceMax - (py / H) * zoomedRange;

  // Compute UTC time at this x column
  let timeStr = '';
  const { visSnaps, colW } = _hm;
  if (visSnaps && colW && px < CHART_W) {
    const ci = Math.max(0, Math.min(Math.floor(px / colW), visSnaps.length - 1));
    const snap = visSnaps[ci];
    if (snap) {
      const dt = new Date(snap.ts);
      timeStr = dt.getUTCHours().toString().padStart(2,'0') + ':' +
                dt.getUTCMinutes().toString().padStart(2,'0') + ':' +
                dt.getUTCSeconds().toString().padStart(2,'0');
    }
  }

  // Crosshair lines (stop before price label on right)
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(px, 0); ctx.lineTo(px, H);
  ctx.moveTo(0, py); ctx.lineTo(W - 56, py);
  ctx.stroke();
  ctx.setLineDash([]);

  // Price label (right edge)
  const priceLabel = hoverPrice.toFixed(2);
  ctx.fillStyle = 'rgba(6,10,15,0.92)';
  ctx.fillRect(W - 54, py - 9, 52, 16);
  ctx.strokeStyle = '#ffd060';
  ctx.lineWidth = 1;
  ctx.strokeRect(W - 54, py - 9, 52, 16);
  ctx.fillStyle = '#ffd060';
  ctx.font = 'bold 10px monospace';
  ctx.fillText(priceLabel, W - 52, py + 4);

  // Time label (top of vertical line)
  if (timeStr && px < CHART_W) {
    const tw = timeStr.length * 6.5 + 10;
    const tx = Math.max(2, Math.min(px - tw / 2, W - tw - 2));
    ctx.fillStyle = 'rgba(6,10,15,0.92)';
    ctx.fillRect(tx, 2, tw, 14);
    ctx.strokeStyle = 'rgba(90,112,128,0.7)';
    ctx.lineWidth = 1;
    ctx.strokeRect(tx, 2, tw, 14);
    ctx.fillStyle = 'rgba(90,112,128,0.9)';
    ctx.font = '9px monospace';
    ctx.fillText(timeStr, tx + 4, 13);
  }
}

// ── mouse wheel zoom & crosshair ─────────────────────────────────────────
(function() {
  const heatmap = document.getElementById('heatmap');
  const wrap    = document.getElementById('heatmap-wrap');
  const cvdWrap = document.getElementById('cvd-wrap');

  // Wheel on the whole wrap div (canvas overlays can swallow events)
  wrap.addEventListener('wheel', e => {
    // If cursor is over CVD panel, don't zoom
    const cvdTop = cvdWrap.getBoundingClientRect().top;
    if (e.clientY >= cvdTop) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.2 : (1 / 1.2);
    priceZoom = Math.max(0.3, Math.min(30, priceZoom * factor));
    clearTimeout(_zoomTimer);
    _zoomTimer = setTimeout(() => {
      if (lastPayload) renderHeatmap(lastPayload);
    }, 35);
  }, { passive: false });

  // Crosshair tracks mouse on the canvas itself
  heatmap.addEventListener('mousemove', e => {
    const rect = heatmap.getBoundingClientRect();
    xhairX = (e.clientX - rect.left) * (heatmap.width / rect.width);
    xhairY = (e.clientY - rect.top)  * (heatmap.height / rect.height);
    renderCrosshair();
  });

  heatmap.addEventListener('mouseleave', () => {
    xhairX = null;
    renderCrosshair();
  });

  // Re-render on window resize so canvas dimensions stay correct
  window.addEventListener('resize', () => {
    if (lastPayload) {
      renderHeatmap(lastPayload);
      renderCrosshair();
    }
  });
})();

window.addEventListener('load', () => {
  // Auto-scroll DOM to mid on first load
  setTimeout(() => {
    const dw = document.getElementById('dom-wrap');
    dw.scrollTop = dw.scrollHeight / 2 - dw.clientHeight / 2;
  }, 2000);
  poll();
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/orderflow":
            payload = build_payload()
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5051)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    start_depth_poll()
    print(f"[orderflow] depth polling started (ETHUSDT, 1s interval)")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[orderflow] http://{args.host}:{args.port}")

    if not args.no_browser:
        import webbrowser, threading as _t
        _t.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
