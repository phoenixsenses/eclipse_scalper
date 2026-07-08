from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import socket
import shutil
import sqlite3
import statistics
import subprocess
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import s34_current_prediction_card
import s34_preliq_shadow_detector
import s34_prediction_error_tracker
import s34_prediction_risk_sandbox


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "microstructure.db"
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
STATUS_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_STATUS.json"
INTELLIGENCE_DB_PATH = ROOT / "data" / "s34_intelligence.db"
GUARDRAIL_V3_AUDIT_PATH = ROOT / "reports" / "research" / "s34" / "S34_GUARDRAIL_V3_AUDIT.json"
BUCKET_INDEPENDENCE_AUDIT_PATH = ROOT / "reports" / "research" / "s34" / "S34_BUCKET_INDEPENDENCE_AUDIT.json"
CALCULATOR_LATEST_PATH = ROOT / "reports" / "research" / "s34" / "S34_LIQ_OUTCOME_CALCULATOR_LATEST.json"
LIVE_EXECUTOR_STATE_PATH = ROOT / "runtime" / "s34_v_engine_live_state.json"
V_ENGINE_V02_MIRROR_BRIEF_PATH = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
V_ENGINE_V02_MIRROR_LEDGER_PATH = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
V_ENGINE_V02_MIRROR_STATE_PATH = ROOT / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json"
V_ENGINE_V02_H4_SHADOW_PATH = ROOT / "reports" / "research" / "s34" / "S34_V02_H4_FORWARD_SHADOW.json"
V_ENGINE_V02_H4_SHADOW_LEDGER_PATH = ROOT / "reports" / "research" / "s34" / "S34_V02_H4_FORWARD_SHADOW_LEDGER.jsonl"
V_ENGINE_SIZING_SHADOW_PATH = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_SIZING_SHADOW_PAPER.json"
STATE_MACHINE_SHADOW_STATE_PATH = ROOT / "reports" / "shadow" / "s34_state_machine_shadow_state.json"
STATE_MACHINE_SHADOW_LEDGER_PATH = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
PID_PATH = ROOT / "logs" / "pids" / "s34_shadow_paper_runner.pid"
V_ENGINE_PID_PATH = ROOT / "logs" / "pids" / "s34_v_engine_live_executor.pid"
V_ENGINE_V02_MIRROR_PID_PATH = ROOT / "logs" / "pids" / "s34_v_engine_v02_shadow_mirror.pid"
STATE_MACHINE_SHADOW_PID_PATH = ROOT / "logs" / "pids" / "s34_state_machine_shadow_runner.pid"
LIVE_CHART_PID_PATH = ROOT / "logs" / "pids" / "s34_live_chart.pid"
STDERR_PATH = ROOT / "logs" / "s34_shadow_paper_runner.stderr.log"
PID_DIR = ROOT / "logs" / "pids"

RULE_NAME = "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"
HIDDEN_DASHBOARD_RULES = frozenset({"ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"})
EXCLUDED_TRADE_IDS = {"P013", "P056"}
FORWARD_RULES = [
    ("ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30", "200K/TP60 exploratory", None),
    ("ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30", "500K/daytrend exploratory", None),
    ("ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30", "500K/negtrend stretched", None),
    ("ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40", "ETH SELL 500K/TP60 exploratory", 30),
    ("ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40", "ETH SELL 1M/TP80 exploratory", 30),
    ("SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30", "SOL 100K/TP60 main", 100),
    ("SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30", "SOL 200K/TP60 exploratory", None),
    ("SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40", "SOL SELL 100K/TP60 exploratory", 30),
    ("SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30", "SOL SELL 200K/TP60 exploratory", 30),
    ("BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30", "BTC 1M distributed exploratory", None),
]
CHART_SYMBOL = "ETHUSDT"
PROCESS_PID_FILES = {
    "collector_supervisor": "collector_supervisor.pid",
    "heartbeat_watchdog": "heartbeat_watchdog.pid",
    "bookticker_collector": "bookticker_collector.pid",
    "microstructure_collector": "microstructure_collector.pid",
    "event_diary": "event_diary.pid",
    "s34_shadow_paper_runner": "s34_shadow_paper_runner.pid",
    "s34_state_machine_shadow_runner": "s34_state_machine_shadow_runner.pid",
    "s34_state_machine_live_executor": "s34_state_machine_live_executor.pid",
    "s34_v_engine_v02_shadow_mirror": "s34_v_engine_v02_shadow_mirror.pid",
    "s34_live_chart": "s34_live_chart.pid",
}
PROCESS_COMMAND_HINTS = {
    "collector_supervisor": "scripts\\collector_supervisor.py",
    "heartbeat_watchdog": "tools.heartbeat_watchdog",
    "bookticker_collector": "data.bookticker_collector",
    "microstructure_collector": "data.microstructure_collector",
    "event_diary": "data.event_diary",
    "s34_shadow_paper_runner": "tools\\s34_shadow_paper_runner.py",
    "s34_state_machine_shadow_runner": "tools.s34_realtime_shadow_runner",
    "s34_state_machine_live_executor": "tools.s34_state_machine_live_executor",
    "s34_v_engine_v02_shadow_mirror": "tools.s34_v_engine_v02_shadow_mirror",
    "s34_live_chart": "tools\\s34_live_chart.py",
}
REGIME_THRESHOLDS = {
    "trend_pct": 1.0,
    "range_pct": 2.5,
    "buy_liq_notional": 5_000_000.0,
    "agg_count": 250_000,
}
RISK_SANDBOX_ACCOUNT_USDT = 40.0
RISK_SANDBOX_MARGIN_USDT = 40.0
RISK_SANDBOX_RISK_BUDGET_PCT = 2.0
RISK_SANDBOX_LEVERAGES = (10.0, 20.0, 40.0, 70.0)

LAST_GOOD: dict[str, Any] | None = None
LAST_ERROR = ""


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>S34 State Machine · ETHUSDT</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0a0a09;--surface:#11110f;--panel:#161613;--panel-2:#1c1b17;--line:#2d2b24;
    --ink:#eee9dc;--muted:#9b927f;--soft:#6f6859;--green:#57d68d;--red:#f06565;
    --gold:#e9b84d;--blue:#5eb7d6;--orange:#e18f46;--violet:#a78bfa;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font:13px/1.45 Inter,"Segoe UI",system-ui,-apple-system,sans-serif;min-height:100vh}
  header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--surface);position:sticky;top:0;z-index:5}
  .brand{display:flex;align-items:center;gap:12px;min-width:0}
  .brand-mark{width:34px;height:34px;border:1px solid var(--gold);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--gold);font-weight:800;font-size:13px;background:#181611}
  header > .h-title{display:none}
  .h-title{font-size:14px;font-weight:800;letter-spacing:0;color:#fff7e6;white-space:nowrap}
  .h-sub{font-size:11px;color:var(--muted);margin-top:1px;white-space:nowrap}
  .h-right{display:flex;gap:10px;align-items:center;font-size:11px;color:var(--muted);flex-wrap:wrap;justify-content:flex-end}
  .h-right > span:not(#state-badge){height:28px;display:inline-flex;align-items:center;gap:7px;padding:0 10px;border-radius:7px;border:1px solid var(--line);font-weight:800;font-size:11px;background:var(--panel-2);white-space:nowrap}
  .status-pill,#state-badge{height:28px;display:inline-flex;align-items:center;gap:7px;padding:0 10px;border-radius:7px;border:1px solid var(--line);font-weight:800;font-size:11px;letter-spacing:0;background:var(--panel-2);white-space:nowrap}
  .badge-wait{color:var(--muted);border-color:var(--line)}
  .badge-long{color:var(--blue);border-color:rgba(94,183,214,.5);background:rgba(94,183,214,.08)}
  .badge-short{color:var(--orange);border-color:rgba(225,143,70,.5);background:rgba(225,143,70,.08)}
  .badge-pending{color:var(--gold);border-color:rgba(233,184,77,.5);background:rgba(233,184,77,.08)}
  .badge-dead{color:var(--red);border-color:rgba(240,101,101,.5);background:rgba(240,101,101,.08)}
  .badge-dev{color:var(--gold);border-color:rgba(233,184,77,.55);background:rgba(233,184,77,.10)}
  #exec-dot{width:8px;height:8px;border-radius:50%;background:var(--soft);display:inline-block;transition:background .3s}
  .dot-alive{background:var(--green)!important;box-shadow:0 0 0 3px rgba(87,214,141,.12)}
  .dot-dead{background:var(--red)!important;box-shadow:0 0 0 3px rgba(240,101,101,.12)}
  .layout{display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:14px;padding:14px;min-height:calc(100vh - 63px)}
  .left-col{display:flex;flex-direction:column;gap:14px;min-width:0}
  .right-col{display:flex;flex-direction:column;gap:14px;min-width:0}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;overflow:hidden;box-shadow:0 10px 24px rgba(0,0,0,.20)}
  .panel-title{font-size:11px;font-weight:800;letter-spacing:0;color:#d8cfba;text-transform:none;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;gap:8px}
  .panel-kicker{font-size:10px;font-weight:700;color:var(--muted)}
  .section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}
  .section-title{font-size:12px;font-weight:850;color:#fff7e6}
  .section-note{font-size:11px;color:var(--muted);margin-top:2px}
  .metric-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:12px}
  .metric{background:var(--panel-2);border:1px solid var(--line);border-radius:7px;padding:9px;min-width:0}
  .metric-label{font-size:10px;color:var(--muted);font-weight:750;text-transform:uppercase}
  .metric-value{font-size:16px;font-weight:850;margin-top:2px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #chart-wrap{position:relative;width:100%;height:350px}
  #chart-wrap canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
  .sm-state{display:flex;flex-direction:column;gap:10px}
  .state-row{display:flex;align-items:center;justify-content:space-between}
  .state-row > div:last-child{display:none}
  .state-action{font-size:22px;font-weight:700;letter-spacing:.04em}
  .state-detail{font-size:11px;color:var(--muted);margin-top:2px}
  .pos-card{border-radius:8px;padding:12px;margin-top:2px}
  .pos-long{border:1px solid rgba(94,183,214,.45);background:rgba(94,183,214,.06)}
  .pos-short{border:1px solid rgba(225,143,70,.45);background:rgba(225,143,70,.06)}
  .pos-pending{border:1px solid rgba(233,184,77,.45);background:rgba(233,184,77,.06)}
  .pos-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
  .pos-dir{font-size:16px;font-weight:700;letter-spacing:.05em}
  .pos-prices{font-size:11px;color:var(--muted);margin-top:2px}
  .pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
  .pos-stat{background:rgba(0,0,0,.18);border-radius:7px;padding:7px 8px;border:1px solid rgba(45,43,36,.8)}
  .ps-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
  .ps-value{font-size:13px;font-weight:700;margin-top:1px}
  .pos-features{font-size:11px;color:var(--muted);margin-top:8px;display:flex;flex-wrap:wrap;gap:8px}
  .feat{display:flex;gap:4px;align-items:center}
  .feat b{color:var(--ink)}
  .score-dots{display:flex;gap:3px;align-items:center}
  .dot-f{color:var(--blue);font-size:10px}
  .dot-e{color:#2a3d52;font-size:10px}
  .silence-tag{display:inline-block;background:rgba(87,214,141,.12);border:1px solid rgba(87,214,141,.35);color:var(--green);border-radius:6px;padding:3px 8px;font-size:10px;font-weight:800;margin-top:6px}
  .noisy-tag{display:inline-block;background:rgba(240,101,101,.1);border:1px solid rgba(240,101,101,.3);color:var(--red);border-radius:6px;padding:3px 8px;font-size:10px;font-weight:800;margin-top:6px}
  .pending-tag{display:inline-block;background:rgba(233,184,77,.1);border:1px solid rgba(233,184,77,.3);color:var(--gold);border-radius:6px;padding:3px 8px;font-size:10px;font-weight:800;margin-top:6px}
  .col-pos{color:var(--green)}
  .col-neg{color:var(--red)}
  .col-warn{color:var(--gold)}
  .col-blue{color:var(--blue)}
  .col-orange{color:var(--orange)}
  .col-muted{color:var(--muted)}
  .shadow-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
  .sig-card{background:var(--panel-2);border-radius:8px;padding:11px;border:1px solid var(--line)}
  .sig-name{font-size:10px;font-weight:850;letter-spacing:0;color:var(--muted);text-transform:uppercase;margin-bottom:6px}
  .sig-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px}
  .sig-stat-item{font-size:11px}
  .sig-stat-label{color:var(--muted)}
  .sig-stat-val{font-weight:700}
  .op-card{border-radius:8px;padding:10px 12px;border:1px solid var(--line);background:rgba(0,0,0,.18);margin-top:6px}
  .op-card.op-long{border-left:3px solid var(--blue)}
  .op-card.op-short{border-left:3px solid var(--orange)}
  .op-card.op-pending2{border-left:3px dashed var(--gold);background:rgba(233,184,77,.05)}
  .op-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
  .op-sig{font-size:11px;font-weight:800;letter-spacing:.02em}
  .op-status{display:inline-block;font-size:9px;font-weight:800;padding:1px 6px;border-radius:4px;background:rgba(233,184,77,.12);border:1px solid rgba(233,184,77,.3);color:var(--gold);margin-left:6px;vertical-align:1px}
  .op-pnl{font-size:17px;font-weight:800;line-height:1;text-align:right;white-space:nowrap}
  .op-pnl-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;text-align:right;margin-top:2px}
  .op-meta{display:flex;flex-wrap:wrap;gap:5px 14px;font-size:10px;color:var(--muted);margin-top:7px}
  .op-meta b{color:var(--ink);font-weight:700}
  .op-prog{height:3px;border-radius:2px;background:rgba(0,0,0,.4);overflow:hidden;margin-top:8px}
  .op-prog>div{height:100%;background:linear-gradient(90deg,var(--gold),var(--orange));border-radius:2px}
  .op-times{display:flex;justify-content:space-between;font-size:9px;color:var(--soft);margin-top:3px}
  .orders-list{display:flex;flex-direction:column;gap:6px}
  .order-row{background:var(--panel-2);border-radius:7px;padding:8px 10px;border:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
  .order-row .or-left{display:flex;flex-direction:column;gap:2px}
  .order-row .or-dir{font-size:11px;font-weight:700}
  .order-row .or-meta{font-size:10px;color:var(--muted)}
  .order-row .or-right{text-align:right}
  .order-row .or-bps{font-size:13px;font-weight:700}
  .order-row .or-price{font-size:10px;color:var(--muted)}
  .proc-list{display:flex;flex-direction:column;gap:5px}
  .proc-row{display:flex;justify-content:space-between;align-items:center;font-size:11px;padding:5px 0;border-bottom:1px solid rgba(45,43,36,.75)}
  .proc-name{color:var(--muted);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .proc-status{display:flex;align-items:center;gap:5px}
  .proc-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
  .proc-alive{background:var(--green);box-shadow:0 0 5px var(--green)}
  .proc-dead{background:var(--red)}
  .scan-info{font-size:11px;color:var(--muted);margin-top:8px;padding-top:8px;border-top:1px solid var(--line)}
  .recon-info{font-size:11px;margin-top:6px;display:flex;gap:12px}
  .empty-state{text-align:center;padding:24px;color:var(--muted);font-size:12px}
  .empty-icon{font-size:28px;margin-bottom:8px;opacity:.4}
  @media(max-width:1100px){.layout{grid-template-columns:1fr}.right-col{order:-1}.metric-row{grid-template-columns:repeat(2,1fr)}}
  @media(max-width:640px){header{align-items:flex-start;flex-direction:column}.h-right{justify-content:flex-start}.metric-row,.shadow-grid{grid-template-columns:1fr}#chart-wrap{height:300px}}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-mark">S34</div>
    <div>
      <div class="h-title">Eclipse S34 Control</div>
      <div class="h-sub">ETHUSDT state machine, shadow research, process health</div>
    </div>
  </div>
  <div class="h-right">
    <span id="state-badge" class="badge-wait">READY_WAIT</span>
    <span><span id="exec-dot"></span><span id="exec-label">executor</span></span>
    <span id="updated" style="opacity:.5">-</span>
  </div>
</header>

<div class="layout">
  <!-- LEFT COLUMN -->
  <div class="left-col">
    <!-- Price chart -->
    <div class="panel" style="padding:14px 14px 10px">
      <div class="panel-title"><span>ETH mark price</span><span class="panel-kicker"><span id="eth-price" style="color:var(--blue);font-weight:850">-</span> / <span id="price-change">-</span></span></div>
      <div id="chart-wrap"><canvas id="chart"></canvas><div id="no-chart" style="display:none;position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px">chart loading...</div></div>
    </div>

    <!-- Shadow paper results -->
    <div class="panel">
      <div class="panel-title"><span>Shadow state machine</span><span id="shadow-n" class="panel-kicker"></span></div>
      <div id="shadow-body">
        <div class="empty-state"><div class="empty-icon">◎</div>waiting for shadow data...</div>
      </div>
    </div>

    <!-- Recent liq events -->
    <div class="panel">
      <div class="panel-title"><span>Recent ETH sell cascades</span><span class="panel-kicker">liquidation tape</span></div>
      <div id="liq-body">
        <div class="empty-state"><div class="empty-icon">○</div>no recent cascades</div>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN -->
  <div class="right-col">
    <!-- State machine panel -->
    <div class="panel">
      <div class="panel-title"><span>Runtime mode</span><span class="panel-kicker">live off by default</span></div>
      <div id="sm-body" class="sm-state">
        <div class="empty-state"><div class="empty-icon">⊙</div>loading...</div>
      </div>
    </div>

    <!-- Order history -->
    <div class="panel">
      <div class="panel-title"><span>Live order history</span><span class="panel-kicker">real orders only</span></div>
      <div id="orders-body">
        <div class="empty-state" style="padding:16px"><div style="color:var(--muted);font-size:11px">No live orders yet — first trade will appear here</div></div>
      </div>
    </div>

    <!-- Process health -->
    <div class="panel">
      <div class="panel-title"><span>Process health</span><span class="panel-kicker">current PIDs</span></div>
      <div id="proc-body" class="proc-list"></div>
    </div>

    <!-- PC / Host health -->
    <div class="panel">
      <div class="panel-title"><span>PC / Host health</span><span class="panel-kicker">restart readiness, read-only</span></div>
      <div id="host-health-body" style="padding:10px 12px;font-size:11px"></div>
    </div>
  </div>
</div>

<script>
  let _chart = null;
  let _chartReady = typeof Chart !== "undefined";

  function esc(s) {
    return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }
  function fmt(v, d) {
    if (v == null || isNaN(Number(v))) return "-";
    return Number(v).toFixed(d ?? 0);
  }
  function fmtPx(v) {
    if (v == null) return "-";
    return Number(v).toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
  }
  function fmtSec(s) {
    if (s == null || s < 0) return "-";
    const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=Math.floor(s%60);
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${ss}s` : `${ss}s`;
  }
  function clsBps(v) {
    if (v == null) return "";
    return Number(v) >= 0 ? "col-pos" : "col-neg";
  }
  function hydrateStaticLabels() {
    const titles = document.querySelectorAll(".panel-title");
    if (titles[0]) titles[0].innerHTML = `<span>ETH mark price</span><span class="panel-kicker"><span id="eth-price" style="color:var(--blue);font-weight:850">-</span> / <span id="price-change">-</span></span>`;
    if (titles[1]) titles[1].innerHTML = `<span>Shadow state machine</span><span id="shadow-n" class="panel-kicker"></span>`;
    if (titles[2]) titles[2].innerHTML = `<span>Recent ETH sell cascades</span><span class="panel-kicker">liquidation tape</span>`;
    if (titles[3]) titles[3].innerHTML = `<span>Runtime mode</span><span class="panel-kicker">live off by default</span>`;
    if (titles[4]) titles[4].innerHTML = `<span>Live order history</span><span class="panel-kicker">real orders only</span>`;
    if (titles[5]) titles[5].innerHTML = `<span>Process health</span><span class="panel-kicker">current PIDs</span>`;
  }
  function scoreDots(base, full) {
    const b = Number(base ?? 0);
    const f = Number(full ?? b);
    let html = '<span class="score-dots">';
    for (let i=0;i<5;i++) html += `<span class="${i<b?"dot-f":"dot-e"}">●</span>`;
    html += `</span> <span style="font-size:11px">${b}/5`;
    if (full != null) html += ` <span class="col-muted">(sil:${f}/6)</span>`;
    html += "</span>";
    return html;
  }

  function renderStateMachine(payload) {
    const live = payload.live_execution || {};
    const alpha = live.alpha_decision || {};
    const proc = live.process || {};
    const env = live.env || {};
    const ap = live.active_position;
    const pe = live.pending_events || [];
    const alive = !!proc.alive;
    const action = alive ? (alpha.action || "READY_WAIT") : "LIVE_OFF";
    const actionDetail = alive
      ? (alpha.detail || alpha.blocked_by || "-")
      : "Live executor is disabled. Shadow runners, chart, and collectors remain online.";

    // Header badge
    const badge = document.getElementById("state-badge");
    const dot = document.getElementById("exec-dot");
    const execLabel = document.getElementById("exec-label");
    badge.textContent = action;
    badge.className = "";
    if (!alive) badge.className = "badge-dev";
    else if (action === "HOLD_LONG") badge.className = "badge-long";
    else if (action === "HOLD_SHORT") badge.className = "badge-short";
    else if (action === "PENDING_STATE") badge.className = "badge-pending";
    else badge.className = "badge-wait";
    dot.className = alive ? "dot-alive" : "dot-dead";
    execLabel.textContent = alive ? `live PID ${proc.pid || "-"}` : "live off";

    const scan = alpha.last_signal_scan || {};
    const recon = alpha.reconciliation || {};
    const pos = recon.position_amounts || {};
    const armed = alive && !!env.live_armed;
    const actionClass = !alive ? "col-warn" : action==="HOLD_LONG" ? "col-blue" : action==="HOLD_SHORT" ? "col-orange" : action==="PENDING_STATE" ? "col-warn" : "col-muted";
    const modeLabel = !alive ? "SHADOW DEVELOPMENT" : (armed ? "LIVE ARMED" : "DRY RUN");
    const modeMeta = !alive ? "orders disabled" : `$${env.margin_usdt||"-"} x ${env.max_leverage||"-"}x`;

    let html = "";

    // Mode row
    html += `<div class="state-row">
      <div>
        <div class="state-action ${actionClass}">${esc(action)}</div>
        <div class="state-detail">${esc(actionDetail)}</div>
        <div class="pending-tag">${esc(modeLabel)} / ${esc(modeMeta)}</div>
      </div>
      <div style="text-align:right;font-size:11px">
        <div style="color:${armed?"var(--green)":"var(--muted)"}">● ${armed?"LIVE ARMED":"DRY RUN"}</div>
        <div class="col-muted" style="margin-top:2px">$${env.margin_usdt||"-"} × ${env.max_leverage||"-"}x</div>
      </div>
    </div>`;

    // Active position card
    if (ap) {
      const dir = String(ap.direction || "").toUpperCase();
      const isLong = dir !== "SHORT";
      const cardCls = isLong ? "pos-card pos-long" : "pos-card pos-short";
      const dirCol = isLong ? "col-blue" : "col-orange";
      const unreal = ap.unrealized_bps;
      const silConfirmed = ap.state_resolution === "SILENCE_CONFIRMED";
      html += `<div class="${cardCls}">
        <div class="pos-top">
          <div>
            <div class="pos-dir ${dirCol}">POSITION OPEN — ${esc(dir)}</div>
            <div class="pos-prices">entry $${fmtPx(ap.entry_price)} &nbsp;→&nbsp; current $${fmtPx(ap.current_price)}</div>
          </div>
          <div style="text-align:right">
            <div class="${clsBps(unreal)}" style="font-size:18px;font-weight:700">${unreal!=null?(Number(unreal)>=0?"+":"")+fmt(unreal,1)+" bps":"-"}</div>
            <div class="col-muted" style="font-size:10px">unrealized</div>
          </div>
        </div>
        <div class="pos-grid">
          <div class="pos-stat"><div class="ps-label">time left</div><div class="ps-value ${Number(ap.time_left_sec)<600?"col-warn":""}">${fmtSec(ap.time_left_sec)}</div></div>
          <div class="pos-stat"><div class="ps-label">stop</div><div class="ps-value">${ap.stop_bps?fmt(ap.stop_bps,0)+" bps":"-"}</div></div>
          <div class="pos-stat"><div class="ps-label">notional</div><div class="ps-value">$${ap.notional_usdt?fmt(ap.notional_usdt,0):"-"}</div></div>
          <div class="pos-stat"><div class="ps-label">margin×lev</div><div class="ps-value">$${ap.margin_usdt?fmt(ap.margin_usdt,1):"-"} × ${ap.leverage||"-"}</div></div>
        </div>
        <div class="pos-features">
          <div class="feat">score: ${scoreDots(ap.base_score, ap.score_if_silence)}</div>
          <div class="feat"><span class="col-muted">session</span> <b>${esc(ap.session||"-")}</b></div>
          <div class="feat"><span class="col-muted">n2h</span> <b>${ap.n2h??"-"}</b></div>
          <div class="feat"><span class="col-muted">b4h</span> <b class="${clsBps(ap.btc4h_bps)}">${ap.btc4h_bps!=null?(Number(ap.btc4h_bps)>=0?"+":"")+fmt(ap.btc4h_bps,1)+" bps":"-"}</b></div>
          <div class="feat"><span class="col-muted">sync</span> <b>${ap.sync_k!=null?fmt(ap.sync_k/1000,0)+"K":"-"}</b></div>
          <div class="feat"><span class="col-muted">vd</span> <b>${ap.vdepth_bps!=null?fmt(ap.vdepth_bps,1)+" bps":"-"}</b></div>
        </div>
        ${silConfirmed ? '<div class="silence-tag">SILENCE CONFIRMED ✓ — hold 4h</div>' : (action==="PENDING_STATE"?'<div class="pending-tag">MONITORING 30min window</div>':"")}
        ${ap.event_id ? `<div style="font-size:10px;color:var(--muted);margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">id: ${esc(ap.event_id)}</div>` : ""}
      </div>`;
    } else if (pe.length > 0) {
      const pev = pe[0];
      const expiresIn = pev.expires_ts_ms ? Math.max(0,(pev.expires_ts_ms - Date.now())/1000) : null;
      html += `<div class="pos-card pos-pending">
        <div class="pos-top">
          <div>
            <div class="pos-dir col-warn">CASCADE DETECTED — CLASSIFYING</div>
            <div class="pos-prices">${esc(pev.status||"-")} · elapsed ${fmtSec(pev.elapsed_sec)}</div>
          </div>
          <div style="text-align:right;font-size:11px">
            <div class="col-warn">expires in</div>
            <div style="font-size:16px;font-weight:700;color:var(--gold)">${fmtSec(expiresIn)}</div>
          </div>
        </div>
        <div class="pos-features">
          <div class="feat">score: ${scoreDots(pev.base_score, pev.score_if_silence)}</div>
          <div class="feat"><span class="col-muted">LONG</span> <b class="${pev.long_eligible?"col-pos":"col-neg"}">${pev.long_eligible?"✓":"✗"}</b></div>
          <div class="feat"><span class="col-muted">SHORT</span> <b class="${pev.short_eligible?"col-pos":"col-neg"}">${pev.short_eligible?"✓":"✗"}</b></div>
          <div class="feat"><span class="col-muted">session</span> <b>${esc(pev.session||"-")}</b></div>
          <div class="feat"><span class="col-muted">sync</span> <b>${pev.sync_k!=null?fmt(pev.sync_k/1000,0)+"K":"-"}</b></div>
          <div class="feat"><span class="col-muted">n2h</span> <b>${pev.n2h??"-"}</b></div>
        </div>
        ${pev.long_opened ? '<div class="pending-tag">LONG OPEN — watching for noisy/BTC/silence</div>' : ""}
      </div>`;
    } else {
      const idleText = alive
        ? "Waiting for ETH SELL >= 200K cascade within 120s freshness window"
        : "Live execution is off. Use shadow buckets and candidate tracker for development.";
      html += `<div style="padding:14px 0">
        <div class="pending-tag">${esc(idleText)}</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Waiting for ETH SELL ≥ 200K cascade within 120s freshness window</div>
        <div class="scan-info">
          last scan: ${scan.fresh_candidates??0} fresh / ${scan.anchors_reconstructed??0} anchors &nbsp;·&nbsp; blocked: ${esc(alpha.blocked_by||"none")}
        </div>
        <div class="recon-info">
          <span><span class="col-muted">LONG</span> <b>${fmt(pos.LONG,4)}</b></span>
          <span><span class="col-muted">SHORT</span> <b>${fmt(pos.SHORT,4)}</b></span>
          <span><span class="col-muted">orders</span> <b>${recon.state_machine_open_order_count??0}</b></span>
          <span class="col-muted">${esc((recon.updated_at_utc||"").slice(11,19))} UTC</span>
        </div>
      </div>`;
    }

    document.getElementById("sm-body").innerHTML = html;
  }

  function renderShadow(payload) {
    const sm = (payload.shadow_paper_buckets||{}).state_machine_shadow || {};
    const stats = sm.signal_stats || {};
    const statsF = sm.signal_stats_filtered || {};
    const closed = sm.closed_trades ?? sm.rows_total ?? 0;
    const smProc = sm.process || {};
    const smAlive = !!smProc.alive;
    const mode = sm.mode || "OBSERVE_ONLY_NO_ORDER";
    const rule = sm.rule_name || "S34_STATE_MACHINE_V1";
    const recent = Array.isArray(sm.recent_events) ? sm.recent_events : [];
    const observer = sm.profit_lock_observer || {};
    const liveProc = (payload.live_execution || {}).process || {};
    const liveOn = !!liveProc.alive;

    document.getElementById("shadow-n").textContent = closed > 0 ? `${closed} closed trades` : "";

    if (sm.available === false) {
      document.getElementById("shadow-body").innerHTML = `<div class="empty-state"><div class="empty-icon">◎</div>shadow accumulating — ${sm.rows_total||0} events so far</div>`;
      return;
    }

    function buildBreakdownHtml(s) {
      const bySession = s.by_session || {};
      const sesOrder = ["US","ASIA","EUROPE","OFF"];
      const sesExtra = Object.keys(bySession).filter(k=>!sesOrder.includes(k));
      const sessionHtml = [...sesOrder,...sesExtra].filter(k=>bySession[k]).map(k=>{
        const b = bySession[k];
        const bwr = b.wr!=null?(Number(b.wr)*100).toFixed(0)+"%":"-";
        const bavg = b.avg_bps!=null?(Number(b.avg_bps)>=0?"+":"")+fmt(b.avg_bps,1):"-";
        return `<div style="display:flex;justify-content:space-between;font-size:10px;padding:2px 0;border-bottom:1px solid rgba(26,37,53,.4)">
          <span style="color:var(--muted);min-width:52px">${esc(k)}</span>
          <span>N=${b.n}</span>
          <span class="${Number(b.wr)>=0.5?"col-pos":"col-neg"}">${bwr}</span>
          <span class="${clsBps(b.avg_bps)}">${bavg} bps</span>
        </div>`;
      }).join("");
      const byMonth = s.by_month || {};
      const monthKeys = Object.keys(byMonth).sort().slice(-4);
      const monthHtml = monthKeys.map(m=>{
        const b = byMonth[m];
        const bwr = b.wr!=null?(Number(b.wr)*100).toFixed(0)+"%":"-";
        const bavg = b.avg_bps!=null?(Number(b.avg_bps)>=0?"+":"")+fmt(b.avg_bps,1):"-";
        return `<div style="display:flex;justify-content:space-between;font-size:10px;padding:2px 0;border-bottom:1px solid rgba(26,37,53,.4)">
          <span style="color:var(--muted);min-width:52px">${esc(m)}</span>
          <span>N=${b.n}</span>
          <span class="${Number(b.wr)>=0.5?"col-pos":"col-neg"}">${bwr}</span>
          <span class="${clsBps(b.avg_bps)}">${bavg} bps</span>
        </div>`;
      }).join("");
      return {sessionHtml, monthHtml};
    }

    function sigCard(key, label, col) {
      const sf = statsF[key];
      const sa = stats[key];
      if (!sa) return "";

      // Primary: new-gates-filtered view (forward-relevant population)
      const primary = sf || sa;
      const wr = primary.wr != null ? (Number(primary.wr)*100).toFixed(0)+"%" : "-";
      const avg = primary.avg_bps != null ? (Number(primary.avg_bps)>=0?"+":"")+fmt(primary.avg_bps,1)+" bps" : "-";
      const total = primary.sum_bps != null ? (Number(primary.sum_bps)>=0?"+":"")+fmt(primary.sum_bps,0)+" bps" : "-";
      const gateLabel = sf
        ? (key==="LONG_SILENCE" ? "TIME_EXIT · sync&lt;200K" : "score≥4")
        : "all records";
      const {sessionHtml, monthHtml} = buildBreakdownHtml(primary);

      // Secondary: full historical context (greyed out)
      let histLine = "";
      if (sf && sa && sa.n !== sf.n) {
        const hwr = sa.wr!=null?(Number(sa.wr)*100).toFixed(0)+"%":"-";
        const havg = sa.avg_bps!=null?(Number(sa.avg_bps)>=0?"+":"")+fmt(sa.avg_bps,1)+" bps":"-";
        histLine = `<div style="margin-top:6px;padding:4px 6px;background:rgba(0,0,0,.18);border-radius:4px;font-size:10px;color:var(--muted)">
          full ledger N=${sa.n} · WR ${hwr} · avg ${havg} (includes pre-gate history)
        </div>`;
      }

      return `<div class="sig-card">
        <div class="sig-name ${col}">${esc(key.replace(/_/g," "))}</div>
        <div style="font-size:10px;color:var(--muted);margin-bottom:4px">${label}</div>
        <div style="font-size:9px;font-weight:700;letter-spacing:.06em;color:#7ca4d4;text-transform:uppercase;margin-bottom:6px">▶ ${gateLabel}</div>
        <div class="sig-stats">
          <div class="sig-stat-item"><span class="sig-stat-label">N </span><span class="sig-stat-val">${primary.n??"-"}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">WR </span><span class="sig-stat-val ${Number(primary.wr)>=0.5?"col-pos":"col-neg"}">${wr}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">avg </span><span class="sig-stat-val ${clsBps(primary.avg_bps)}">${avg}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">total </span><span class="sig-stat-val ${clsBps(primary.sum_bps)}">${total}</span></div>
        </div>
        ${sessionHtml ? `<div style="margin-top:8px"><div style="font-size:10px;font-weight:700;letter-spacing:.05em;color:var(--muted);text-transform:uppercase;margin-bottom:3px">by session</div>${sessionHtml}</div>` : ""}
        ${monthHtml ? `<div style="margin-top:8px"><div style="font-size:10px;font-weight:700;letter-spacing:.05em;color:var(--muted);text-transform:uppercase;margin-bottom:3px">by month</div>${monthHtml}</div>` : ""}
        ${histLine}
      </div>`;
    }

    const sigRows = [
      ["LONG_SILENCE", "LONG hold 4h · sync&lt;200K · btc7d&lt;0", "col-blue"],
      ["SHORT_NEITHER", "SHORT BTC≥2M +5min delay 2h · score≥4", "col-orange"],
    ].map(([key,label,col]) => sigCard(key,label,col)).join("");

    const _pr = payload.price || [];
    const _lastPxRow = _pr.length ? _pr[_pr.length-1] : null;
    const lastPx = _lastPxRow && _lastPxRow.close != null ? Number(_lastPxRow.close) : null;
    const openRows = (sm.open_positions || []).map(p => {
      const dir = String(p.direction||"").toUpperCase();
      const isShort = dir === "SHORT";
      const entry = p.entry_price != null ? Number(p.entry_price) : null;
      const pending = entry == null;
      let pnl = null;
      if (entry && lastPx) pnl = (isShort ? (entry - lastPx) / entry : (lastPx - entry) / entry) * 1e4;
      else if (p.observer_pnl_bps != null) pnl = Number(p.observer_pnl_bps);
      const pnlTxt = pnl != null ? (pnl>=0?"+":"")+fmt(pnl,1)+" bps" : "—";
      const now = Date.now();
      const inTradeSec = p.entry_ts_ms ? Math.max(0,(now - p.entry_ts_ms)/1000) : null;
      const exitInSec = p.exit_due_ms ? Math.max(0,(p.exit_due_ms - now)/1000) : null;
      const totalSec = (p.entry_ts_ms && p.exit_due_ms && p.exit_due_ms > p.entry_ts_ms) ? (p.exit_due_ms - p.entry_ts_ms)/1000 : null;
      const progPct = (totalSec && inTradeSec != null) ? Math.min(100, inTradeSec/totalSec*100) : null;
      const triggerInSec = pending && p.sil_check_ms ? Math.max(0,(p.sil_check_ms - now)/1000) : null;
      const waitingSec = pending && p.anchor_ts_ms ? Math.max(0,(now - p.anchor_ts_ms)/1000) : null;
      const sk = p.sync_k != null ? fmt(p.sync_k/1000,0)+"K" : null;
      const metaBits = [
        entry != null ? `entry <b>$${fmtPx(entry)}</b>` : null,
        entry != null && lastPx != null ? `now <b>$${fmtPx(lastPx)}</b>` : null,
        p.sl_bps != null ? `stop <b class="col-neg">-${fmt(p.sl_bps,0)} bps</b>` : null,
        p.score != null ? `score <b>${p.score}</b>` : null,
        sk ? `sync <b>${esc(sk)}</b>` : null,
        p.n2h != null ? `n2h <b>${p.n2h}</b>` : null,
        p.session ? `session <b>${esc(p.session)}</b>` : null,
        p.buy_state ? `state <b>${esc(p.buy_state)}</b>` : null,
        p.observer_triggered ? `<b class="col-warn">lock armed</b>` : null,
      ].filter(Boolean).map(x=>`<span>${x}</span>`).join("");
      return `<div class="op-card ${pending?"op-pending2":(isShort?"op-short":"op-long")}">
        <div class="op-head">
          <div>
            <span class="op-sig ${pending?"col-warn":(isShort?"col-orange":"col-blue")}">${esc(p.signal||"-")} · ${esc(dir||"-")}</span>
            <span class="op-status">${esc(p.status||"-")}</span>
          </div>
          <div>
            ${pending
              ? `<div class="op-pnl col-warn" style="font-size:13px">AWAITING TRIGGER</div><div class="op-pnl-label">no position yet</div>`
              : `<div class="op-pnl ${pnl!=null?clsBps(pnl):"col-muted"}">${pnlTxt}</div><div class="op-pnl-label">${entry && lastPx ? "live pnl (mark)" : "pnl unavailable"}</div>`}
          </div>
        </div>
        <div class="op-meta">${metaBits}</div>
        ${progPct != null ? `<div class="op-prog"><div style="width:${progPct.toFixed(1)}%"></div></div>
        <div class="op-times"><span>in trade ${fmtSec(inTradeSec)}</span><span>time exit in ${fmtSec(exitInSec)}</span></div>` : ""}
        ${pending ? `<div class="op-prog"><div style="width:${triggerInSec!=null && waitingSec!=null && (waitingSec+triggerInSec)>0 ? Math.min(100, waitingSec/(waitingSec+triggerInSec)*100).toFixed(1) : 0}%;background:linear-gradient(90deg,var(--gold),var(--gold))"></div></div>
        <div class="op-times"><span>anchor age ${fmtSec(waitingSec)}</span><span>trigger window closes in ${fmtSec(triggerInSec)}</span></div>` : ""}
      </div>`;
    }).join("");

    const recentClosed = Array.isArray(sm.recent_closed) ? sm.recent_closed : recent.filter(r=>r.event==="CLOSE");
    function gatePass(r) {
      const sig = String(r.signal||"");
      const cr = String(r.close_reason||"");
      const sk = Number(r.sync_k||0);
      const sc = Number(r.score||0);
      const btc7d = r.btc7d_bps != null ? Number(r.btc7d_bps) : null;
      if (sig==="LONG_SILENCE") return cr==="TIME_EXIT" && sk<200000 && btc7d!==null && btc7d<0;
      if (sig==="SHORT_NEITHER") return sc>=4;
      return false;
    }
    const recentSorted = [...recentClosed].sort((a,b) => (b.anchor_ts_ms||b.entry_ts_ms||0) - (a.anchor_ts_ms||a.entry_ts_ms||0));
    const recentGated = recentSorted.filter(r => gatePass(r));
    const recentRows = recentGated.slice(0,10).map(r => {
      const dir = String(r.direction || "").toUpperCase();
      const sig = String(r.signal||"-");
      const nb = r.net_bps ?? r.outcome_bps;
      const ts = String(r.closed_utc || r.opened_utc || "").replace("T"," ").slice(0,16);
      const cr = String(r.close_reason||"-");
      const sk = r.sync_k!=null ? fmt(r.sync_k/1000,0)+"K" : "-";
      const sc = r.score != null ? r.score : "-";
      const btc7d = r.btc7d_bps != null ? (Number(r.btc7d_bps)>=0?"+":"")+fmt(r.btc7d_bps,0) : "-";
      const pass = gatePass(r);
      const gateTag = `<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:${pass?"rgba(40,242,124,.12)":"rgba(255,79,95,.08)"};color:${pass?"var(--green)":"var(--muted)"};">${pass?"✓ gate":"✗ gate"}</span>`;
      return `<div style="display:grid;grid-template-columns:100px 80px 1fr 50px 40px 60px 70px;gap:4px;align-items:center;font-size:10px;padding:4px 0;border-bottom:1px solid rgba(26,37,53,.4)">
        <span class="${dir==="SHORT"?"col-orange":"col-blue"}" style="font-weight:700">${esc(sig)}</span>
        <span class="col-muted" style="font-size:9px">${esc(cr)}</span>
        <span class="col-muted" style="font-size:9px">sc:${sc} sk:${esc(sk)} b7d:${esc(btc7d)}</span>
        <span>${gateTag}</span>
        <span class="${clsBps(nb)}" style="font-weight:700">${nb!=null?(Number(nb)>=0?"+":"")+fmt(nb,1):"-"}</span>
        <span class="col-muted" style="font-size:9px">bps</span>
        <span class="col-muted" style="font-size:9px">${esc(ts)}</span>
      </div>`;
    }).join("");

    const observerLine = observer && observer.protocol ? `<div style="margin-top:8px;font-size:10px;color:var(--muted)">
      observer: <b style="color:var(--ink)">${esc(observer.protocol)}</b>
      &nbsp; trigger <b style="color:var(--ink)">${fmt(observer.trigger_bps,0)}</b>
      &nbsp; lock <b style="color:var(--ink)">${fmt(observer.lock_bps,0)}</b>
      &nbsp; exits <b style="color:var(--ink)">${observer.shadow_exit_n??0}</b>
    </div>` : "";

    const statusCard = `<div class="sig-card" style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">
        <div>
          <div class="sig-name col-blue">V1 STATE MACHINE PAPER BUCKET</div>
          <div style="font-size:10px;color:var(--muted);word-break:break-all">${esc(rule)}</div>
          <div style="font-size:10px;color:var(--muted);margin-top:3px">records paper/shadow trades only; no exchange orders</div>
        </div>
        <div style="text-align:right;font-size:10px;min-width:118px">
          <div class="${smAlive?"col-pos":"col-neg"}">${smAlive?"alive":"dead"} PID ${esc(smProc.pid||"-")}</div>
          <div class="col-muted">${esc(mode)}</div>
          <div class="col-muted">state age ${fmtSec(sm.state_age_sec)} / ledger ${fmtSec(sm.ledger_age_sec)}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px">
        <div class="sig-stat-item"><span class="sig-stat-label">rows </span><span class="sig-stat-val">${sm.rows_total??0}</span></div>
        <div class="sig-stat-item"><span class="sig-stat-label">closed </span><span class="sig-stat-val">${sm.closed_trades??0}</span></div>
        <div class="sig-stat-item"><span class="sig-stat-label">open </span><span class="sig-stat-val">${sm.open_trades??0}</span></div>
        <div class="sig-stat-item"><span class="sig-stat-label">updated </span><span class="sig-stat-val">${esc(String(sm.state_updated_utc||"").slice(11,19) || "-")}</span></div>
      </div>
      ${observerLine}
      <div style="margin-top:8px;padding:6px 8px;background:rgba(0,0,0,.2);border-radius:5px;font-size:10px;border:1px solid rgba(26,37,53,.8)">
        <span style="font-size:9px;font-weight:700;letter-spacing:.06em;color:var(--muted);text-transform:uppercase">live gates (active)</span>
        <div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:6px">
          <span style="color:var(--blue)">LONG:</span>
          <span class="col-muted">sync&lt;200K · btc7d&lt;0 · excl US13-14 · excl Mon/Wed · score+1≥3</span>
        </div>
        <div style="margin-top:2px;display:flex;flex-wrap:wrap;gap:6px">
          <span style="color:var(--orange)">SHORT:</span>
          <span class="col-muted">score≥4 · BTC≥2M · delay≥5min · excl EUROPE/Sun</span>
        </div>
      </div>
      ${openRows ? `<div style="margin-top:8px"><div class="sig-name" style="margin-bottom:3px">open shadow positions · ${(sm.open_positions||[]).length}</div>${openRows}</div>` : ""}
    </div>`;

    const recentHeader = `<div style="display:grid;grid-template-columns:100px 80px 1fr 50px 40px 60px 70px;gap:4px;font-size:9px;color:var(--muted);font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding-bottom:4px;border-bottom:1px solid rgba(26,37,53,.6);margin-bottom:2px">
      <span>signal</span><span>exit</span><span>sc / sk / b7d</span><span>gate</span><span>bps</span><span></span><span>time UTC</span>
    </div>`;
    const recentCard = recentRows ? `<div class="sig-card" style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <div class="sig-name" style="margin-bottom:0">recent paper ledger — gate-pass ${recentGated.length} / ${recentClosed.length} total</div>
        <div style="font-size:9px;color:var(--muted)">✓=passes live gates</div>
      </div>
      ${recentHeader}
      ${recentRows}
    </div>` : "";

    const cands = Array.isArray(sm.candidate_buckets) ? sm.candidate_buckets : [];
    const candRows = cands.map(c => {
      const s = c.stats || {};
      const wr = s.wr != null ? (Number(s.wr)*100).toFixed(0)+"%" : "-";
      const avg = s.avg_bps != null ? (Number(s.avg_bps)>=0?"+":"")+fmt(s.avg_bps,1) : "-";
      const mcp = c.mc_p != null ? `p=${Number(c.mc_p).toFixed(3)}` : "";
      const lastTs = s.last_utc ? String(s.last_utc).replace("T"," ").slice(0,16) : "-";
      const ok = s.n > 0 && Number(s.wr) >= 0.7;
      const hot = c.name === "C_short_noisy_btc1m_d5_h180";
      return `<div style="display:grid;grid-template-columns:1fr 30px 50px 60px 55px 120px;gap:4px;align-items:center;font-size:10px;padding:5px 0;border-bottom:1px solid rgba(26,37,53,.4);${hot?"border-left:3px solid var(--gold);padding-left:6px;background:rgba(255,184,77,.06)":""}">
        <span>
          <span style="display:block;font-size:9px;font-weight:700;color:${hot?"var(--gold)":"var(--muted)"}">${esc(c.name||"")}</span>
          <span class="col-muted" style="display:block;font-size:9px">${esc(c.label)}</span>
        </span>
        <span style="font-weight:700">${s.n??"-"}</span>
        <span class="${ok?"col-pos":"col-muted"}">${wr}</span>
        <span class="${clsBps(s.avg_bps)}">${avg} bps</span>
        <span style="font-size:9px;color:var(--muted)">${esc(mcp)}</span>
        <span style="font-size:9px;color:var(--muted)">${esc(lastTs)}</span>
      </div>`;
    }).join("");
    // Forward alpha spotlight (hour17 route; live state is shown from process truth).
    const aa = sm.active_alpha || null;
    let alphaCard = "";
    if (aa) {
      const alphaTitle = liveOn ? "ACTIVE ALPHA" : "FORWARD ALPHA";
      const liveLabel = liveOn ? `LIVE ON / opened=${((aa.live_summary || {}).orders_opened ?? 0)}` : "LIVE OFF / orders disabled";
      const st = aa.shadow_stats || {};
      const ls = aa.live_summary || {};
      const rs = aa.research || {};
      const n = st.n ?? 0;
      const wr = st.wr != null ? (Number(st.wr)*100).toFixed(0)+"%" : "-";
      const tot = st.sum_bps != null ? (Number(st.sum_bps)>=0?"+":"")+fmt(st.sum_bps,0)+" bps" : "-";
      const avg = st.avg_bps != null ? (Number(st.avg_bps)>=0?"+":"")+fmt(st.avg_bps,1)+" bps" : "-";
      const liveOpened = ls.orders_opened ?? 0;
      const liveActive = ls.active ? "● ACTIVE "+esc(ls.active_direction||"LONG") : "flat";
      const ops = Array.isArray(aa.open_positions) ? aa.open_positions : [];
      const opRows = ops.length ? ops.map(p => {
        const exitTs = p.exit_due_ms ? new Date(p.exit_due_ms).toISOString().replace("T"," ").slice(0,16) : "-";
        const nk = p.running_notional ? "$"+fmt(p.running_notional/1000,0)+"K" : "-";
        return `<div style="display:grid;grid-template-columns:44px 1fr 1fr 1fr 1fr;gap:4px;font-size:9px;padding:2px 0;border-top:1px solid rgba(26,37,53,.4)">
          <span class="col-blue" style="font-weight:700">${esc(p.direction||"LONG")}</span>
          <span class="col-muted">h=${p.hour??"-"} ${esc(p.status||"OPEN")}</span>
          <span class="col-muted">btc4h ${fmt(p.btc4h_bps,0)}</span>
          <span class="col-muted">casc ${nk}</span>
          <span class="col-muted">exit ${esc(exitTs)}</span>
        </div>`;
      }).join("") : `<div style="font-size:9px;color:var(--muted);padding:3px 0">No open ${esc(aa.name)} position right now — fires on next hour>=17 UTC cascade in a down regime.</div>`;
      alphaCard = `<div class="sig-card" style="margin-top:10px;border:1px solid var(--gold)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <div class="sig-name" style="margin-bottom:0;color:var(--gold)">${esc(alphaTitle)} / ${esc(aa.name)}</div>
          <div style="font-size:9px;color:${liveOn?"var(--green)":"var(--gold)"};font-weight:800">${esc(liveLabel)}</div>
        </div>
        <div style="font-size:9px;color:var(--muted);margin-bottom:6px">${esc(aa.definition||"")}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">
          <span style="font-size:9px;font-weight:800;color:${liveOn?"var(--green)":"var(--gold)"}">${liveOn?"LIVE (real orders)":"LIVE OFF"}</span>
          <span style="font-size:9px;color:var(--muted)">${esc(liveLabel)}</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">
          <span style="font-size:9px;font-weight:700;color:var(--gold)">SHADOW (paper mirror)</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:4px">
          <div class="sig-stat-item"><span class="sig-stat-label">trades </span><span class="sig-stat-val" style="font-weight:700">${n}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">WR </span><span class="sig-stat-val ${n>0&&Number(st.wr)>=0.55?"col-pos":"col-muted"}">${wr}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">total </span><span class="sig-stat-val ${clsBps(st.sum_bps)}">${tot}</span></div>
          <div class="sig-stat-item"><span class="sig-stat-label">avg </span><span class="sig-stat-val ${clsBps(st.avg_bps)}">${avg}</span></div>
        </div>
        <div style="font-size:8px;color:var(--muted);margin-bottom:3px">${esc(ls.note||"")}</div>
        <div style="font-size:9px;color:var(--muted);margin-bottom:4px">research OOS: ${rs.per_month??"?"}/mo · WR ${rs.wr!=null?(rs.wr*100).toFixed(0)+"%":"?"} · mc_p ${rs.oos_mc_p??"?"} · WF ${esc(rs.wf||"?")} · mdd ${rs.mdd_bps??"?"} bps · dir ${esc(aa.direction||"LONG")}-only</div>
        <div style="font-size:9px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-top:4px">open positions</div>
        ${opRows}
      </div>`;
    }
    const candCard = candRows ? `<div class="sig-card" style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <div class="sig-name" style="margin-bottom:0;color:var(--gold)">CANDIDATE TRACKER — shadow/live routes</div>
        <div style="font-size:9px;color:var(--muted)">${liveOn?"hour17 live enabled":"live disabled"} / others shadow-only</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 30px 50px 60px 55px 120px;gap:4px;font-size:9px;color:var(--muted);font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding-bottom:4px;border-bottom:1px solid rgba(26,37,53,.6);margin-bottom:2px">
        <span>gate</span><span>N</span><span>WR</span><span>avg</span><span>MC p</span><span>last event</span>
      </div>
      ${candRows}
    </div>` : "";
    let shadowHtml = `${statusCard}${alphaCard}${candCard}<div class="shadow-grid">${sigRows}</div>${recentCard}`;
    if (!liveOn) {
      shadowHtml = shadowHtml
        .replace("ACTIVE ALPHA", "FORWARD ALPHA")
        .replace("LIVE (real orders)", "LIVE OFF")
        .replace("hour17 = LIVE", "live disabled")
        .replace(/opened=.*?<\/span>/, "orders disabled</span>");
    }
    document.getElementById("shadow-body").innerHTML = shadowHtml;
  }

  function renderLiq(payload) {
    const liqs = (payload.liq || []).filter(l => l.side === "SELL" && l.symbol === "ETHUSDT" && (l.notional||l.running_notional||0) >= 50000);
    if (!liqs.length) {
      document.getElementById("liq-body").innerHTML = `<div class="empty-state" style="padding:12px"><span style="color:var(--muted);font-size:11px">No ETH SELL cascades ≥50K in data window</span></div>`;
      return;
    }
    const sorted = liqs.slice().sort((a,b) => (b.ts||0)-(a.ts||0)).slice(0,8);
    const rows = sorted.map(l => {
      const n = l.notional || l.running_notional || 0;
      const big = n >= 200000;
      const ts = l.ts ? new Date(l.ts).toISOString().slice(11,19) : "-";
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(26,37,53,.5);font-size:11px">
        <span class="col-muted">${ts} UTC</span>
        <span class="${big?"col-orange":"col-muted"}">${big?"▶ ":""}$${fmt(n/1000,0)}K SELL${big?" ← cascade trigger":""}</span>
      </div>`;
    }).join("");
    document.getElementById("liq-body").innerHTML = rows;
  }

  function renderOrders(payload) {
    const orders = (payload.live_execution||{}).last_orders || [];
    const liveOn = !!(((payload.live_execution||{}).process||{}).alive);
    if (!orders.length) {
      document.getElementById("orders-body").innerHTML = `<div class="empty-state" style="padding:14px"><div style="color:${liveOn?"var(--muted)":"var(--gold)"};font-size:11px">${liveOn?"No live orders in the current window":"Live execution disabled. Shadow research stays online."}</div></div>`;
      return;
    }
    const rows = orders.slice(0,8).map(o => {
      const dir = (o.direction||"?").toUpperCase();
      const isLong = dir !== "SHORT";
      const nb = o.net_bps ?? o.outcome_bps;
      return `<div class="order-row">
        <div class="or-left">
          <span class="or-dir ${isLong?"col-blue":"col-orange"}">${esc(dir)}</span>
          <span class="or-meta">${esc((o.closed_utc||o.created_at_utc||o.opened_utc||"").slice(0,19))} UTC</span>
          <span class="or-meta">${esc(o.close_reason||o.signal||"-")}</span>
        </div>
        <div class="or-right">
          <div class="or-bps ${clsBps(nb)}">${nb!=null?(Number(nb)>=0?"+":"")+fmt(nb,1)+" bps":"-"}</div>
          <div class="or-price">entry $${fmtPx(o.entry_price||o.entry_price_ref)}</div>
        </div>
      </div>`;
    }).join("");
    document.getElementById("orders-body").innerHTML = `<div class="orders-list">${rows}</div>`;
  }

  function renderProcesses(payload) {
    const procs = payload.process_health || [];
    if (!procs.length) { document.getElementById("proc-body").innerHTML = ""; return; }
    const SM_FIRST = ["s34_state_machine_live_executor","s34_state_machine_shadow_runner","collector_supervisor","microstructure_collector","bookticker_collector","heartbeat_watchdog","s34_live_chart"];
    const sorted = [...procs].sort((a,b) => {
      const ai = SM_FIRST.indexOf(a.name), bi = SM_FIRST.indexOf(b.name);
      if (ai>=0 && bi<0) return -1; if (bi>=0 && ai<0) return 1;
      if (ai>=0 && bi>=0) return ai-bi; return (a.name||"").localeCompare(b.name||"");
    });
    const rows = sorted.map(p => {
      const alive = !!p.alive;
      const disabled = !alive && /live_executor/.test(String(p.name||"")) && (!p.pid || Number(p.pid) === 0);
      const statusText = disabled ? "disabled" : (alive ? `PID ${p.pid||"-"}` : `PID ${p.pid||"-"}`);
      const statusColor = disabled ? "var(--gold)" : (alive ? "var(--green)" : "var(--red)");
      const dotClass = disabled ? "" : (alive ? "proc-alive" : "proc-dead");
      return `<div class="proc-row">
        <span class="proc-name">${esc(p.name)}</span>
        <span class="proc-status"><span class="proc-dot ${dotClass}" style="${disabled?"background:var(--gold)":""}"></span><span style="color:${statusColor}">${esc(statusText)}</span></span>
      </div>`;
    }).join("");
    document.getElementById("proc-body").innerHTML = rows;
  }

  function hostStateColor(state) {
    if (state === "HOST_RESTART_RED") return "var(--red)";
    if (state === "HOST_RESTART_YELLOW") return "var(--gold)";
    if (state === "HOST_RESTART_GREEN") return "var(--green)";
    return "var(--muted)";
  }
  function renderHostHealth(payload) {
    const hh = payload.host_health || {};
    const el = document.getElementById("host-health-body");
    if (!el) return;
    if (!hh.available) {
      el.innerHTML = `<div style="color:var(--muted)">host health unavailable${hh.error ? ": "+esc(hh.error) : ""}</div>`;
      return;
    }
    const obs = hh.observations || {};
    const stateLabel = String(hh.state||"HOST_RESTART_UNKNOWN").replace("HOST_RESTART_","");
    const gb = (b) => (b==null ? "-" : (b/(1024**3)).toFixed(2));
    el.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="color:${hostStateColor(hh.state)};font-weight:800">${esc(stateLabel)}</span>
        ${hh.deferred ? '<span style="color:var(--gold)">DEFER_UNTIL_SAFE_CHECKPOINT</span>' : ""}
      </div>
      <div style="color:var(--muted);margin-bottom:6px">${esc(hh.recommended_action||"-")}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;color:var(--muted)">
        <span>uptime: ${esc(obs.uptime_human||"-")}</span>
        <span>pending reboot: ${esc(obs.pending_reboot||"-")}</span>
        <span>RAM: ${obs.ram_used_pct!=null?obs.ram_used_pct+"%":"-"}</span>
        <span>commit: ${obs.commit_used_pct!=null?obs.commit_used_pct+"%":"-"}</span>
        <span>D: free: ${gb(obs.d_drive_free_bytes)} GB</span>
        <span>dist to ${fmt(hh.d_drive_intervention_free_gb,0)}GB thr: ${obs.d_drive_distance_to_threshold_gb ?? "-"} GB</span>
        <span>collector: ${esc(obs.collector_status||"-")}</span>
        <span>SSD temp: ${obs.ssd_temp_c!=null?obs.ssd_temp_c+"°C":"unavailable"}</span>
      </div>
      <div style="margin-top:6px;color:var(--muted)">why: ${esc((hh.reasons||[]).join(", "))}</div>
    `;
  }

  function renderChart(payload) {
    if (!_chartReady) { document.getElementById("no-chart").style.display="flex"; return; }
    const prices = payload.price || [];
    if (!prices.length) return;
    const last = prices[prices.length-1];
    if (last) {
      document.getElementById("eth-price").textContent = "$" + fmtPx(last.close);
      const first = prices[0];
      if (first && first.close) {
        const chg = (last.close - first.close) / first.close * 100;
        const el = document.getElementById("price-change");
        el.textContent = (chg>=0?"+":"")+chg.toFixed(2)+"% (1h)";
        el.className = chg>=0?"col-pos":"col-neg";
      }
    }
    // Cascade markers from liq
    const cascades = (payload.liq||[]).filter(l=>l.side==="SELL"&&l.symbol==="ETHUSDT"&&(l.notional||0)>=200000);
    const casc_ts = new Set(cascades.map(c=>Math.round(c.ts/60000)*60000));

    const pData = prices.map(p=>({x:p.ts,y:p.close}));
    const markerData = prices.map(p=>{
      const bucket = Math.round(p.ts/60000)*60000;
      return casc_ts.has(bucket) ? p.close : null;
    });

    if (!_chart) {
      const ctx = document.getElementById("chart").getContext("2d");
      _chart = new Chart(ctx, {
        data: {datasets: [
          {label:"ETH mark",type:"line",data:pData,borderColor:"#2ec4ff",borderWidth:1.8,pointRadius:0,tension:.2,fill:true,backgroundColor:"rgba(46,196,255,.04)",yAxisID:"y"},
          {label:"cascade",type:"scatter",data:markerData.map((y,i)=>y!=null?{x:pData[i].x,y}:null).filter(Boolean),pointRadius:7,pointStyle:"triangle",pointBackgroundColor:"rgba(255,159,28,.8)",pointBorderColor:"#ff9f1c",yAxisID:"y"},
        ]},
        options:{
          responsive:true,maintainAspectRatio:false,animation:false,
          interaction:{mode:"index",intersect:false},
          plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.dataset.label==="cascade"?"CASCADE $"+fmtPx(c.raw.y):"$"+fmtPx(c.raw.y)}}},
          scales:{
            x:{type:"linear",display:true,ticks:{color:"#3a5068",font:{size:9},callback:v=>{const d=new Date(v);return d.getUTCHours()+":"+String(d.getUTCMinutes()).padStart(2,"0")}},grid:{color:"rgba(26,37,53,.6)"}},
            y:{display:true,position:"right",ticks:{color:"#3a5068",font:{size:9},callback:v=>"$"+fmtPx(v)},grid:{color:"rgba(26,37,53,.6)"}}
          }
        }
      });
    } else {
      _chart.data.datasets[0].data = pData;
      _chart.data.datasets[1].data = markerData.map((y,i)=>y!=null?{x:pData[i].x,y}:null).filter(Boolean);
      _chart.update("none");
    }

    // Active position line overlay
    const ap = (payload.live_execution||{}).active_position;
    if (ap && ap.entry_price && _chart.data.datasets.length < 3) {
      _chart.data.datasets.push({label:"entry",type:"line",data:pData.map(p=>({x:p.x,y:ap.entry_price})),borderColor:"rgba(40,242,124,.5)",borderWidth:1,borderDash:[4,4],pointRadius:0,yAxisID:"y"});
      _chart.update("none");
    } else if (!ap && _chart.data.datasets.length > 2) {
      _chart.data.datasets.splice(2);
      _chart.update("none");
    }
  }

  async function refresh() {
    try {
      const res = await fetch("/api/data", {cache:"no-store"});
      if (!res.ok) throw new Error("HTTP "+res.status);
      const payload = await res.json();
      document.getElementById("updated").textContent = (payload.updated_utc||"-").slice(11,19)+" UTC";
      renderStateMachine(payload);
      renderShadow(payload);
      renderLiq(payload);
      renderOrders(payload);
      renderProcesses(payload);
      renderHostHealth(payload);
      renderChart(payload);
    } catch(e) {
      document.getElementById("updated").textContent = "error: "+e.message;
    }
  }

  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded — price chart disabled");
  } else {
    _chartReady = true;
  }

  hydrateStaticLabels();
  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>
"""


def utc_iso_ms(ts_ms: int | float | None) -> str:
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    if limit is not None and len(lines) > int(limit):
        lines = lines[-int(limit):]
    out: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def read_env_file(path: Path = ROOT / ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def read_text_tail(path: Path, lines: int = 40) -> list[str]:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return data[-lines:]
    except Exception:
        return []


def pid_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    try:
        kernel32 = ctypes.windll.kernel32
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(ok) and int(exit_code.value) == 259
    except Exception:
        return False


def read_pid_file(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


_PROCESS_SCAN_CACHE: tuple[float, dict[str, int]] = (0.0, {})


def commandline_process_pids() -> dict[str, int]:
    global _PROCESS_SCAN_CACHE
    now = time.time()
    if now - _PROCESS_SCAN_CACHE[0] < 5:
        return dict(_PROCESS_SCAN_CACHE[1])
    found: dict[str, int] = {}
    if os.name != "nt":
        _PROCESS_SCAN_CACHE = (now, found)
        return found
    try:
        raw = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
        data = json.loads(raw) if raw else []
        if isinstance(data, dict):
            data = [data]
        for row in data if isinstance(data, list) else []:
            cmd = str(row.get("CommandLine") or "")
            pid = int(row.get("ProcessId") or 0)
            if pid <= 0:
                continue
            low = cmd.lower()
            for name, hint in PROCESS_COMMAND_HINTS.items():
                if hint.lower() in low:
                    found.setdefault(name, pid)
    except Exception:
        found = {}
    _PROCESS_SCAN_CACHE = (now, found)
    return dict(found)


def process_health() -> list[dict[str, Any]]:
    rows = []
    cmd_pids = commandline_process_pids()
    for name, pid_file in PROCESS_PID_FILES.items():
        pid = read_pid_file(PID_DIR / pid_file)
        if name == "s34_live_chart" and (not pid or not pid_is_alive(pid)):
            pid = os.getpid()
        alive = pid_is_alive(pid)
        source = "pid_file"
        if not alive and cmd_pids.get(name):
            pid = cmd_pids[name]
            alive = pid_is_alive(pid)
            source = "command_line"
        rows.append({"name": name, "pid": pid, "alive": alive, "source": source})
    return rows


def runner_stderr_status() -> dict[str, Any]:
    tail = read_text_tail(STDERR_PATH, 40)
    nonempty = [line for line in tail if line.strip()]
    return {
        "has_error": bool(nonempty),
        "tail": tail,
        "last_error": nonempty[-1] if nonempty else "",
    }


def timed_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=3, check_same_thread=False)
    deadline = time.monotonic() + 3.0

    def progress() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(progress, 50_000)
    return conn


def fetchall(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(sql, params).fetchall())


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return bool(row)


def latest_mark_ms(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "mark_prices"):
        return int(time.time() * 1000)
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    return int(row[0] or time.time() * 1000)


def stream_health(conn: sqlite3.Connection, now_ms: int) -> list[dict[str, Any]]:
    specs = [
        ("liquidations", "ts_ms"),
        ("book_ticker", "ts_ms"),
        ("mark_prices", "ts_ms"),
        ("agg_trades", "ts_ms"),
    ]
    rows = []
    for table, ts_col in specs:
        if not table_exists(conn, table):
            rows.append({"name": table, "last_ts_ms": None, "last_utc": "", "minutes_since": None, "rows_last_hour": 0, "status": "dead"})
            continue
        try:
            last_ts = conn.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()[0]
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {ts_col}>=?",
                (int(now_ms) - 3_600_000,),
            ).fetchone()[0]
            minutes = None if last_ts is None else (int(now_ms) - int(last_ts)) / 60_000.0
            status = "dead"
            if minutes is not None:
                status = "green" if minutes < 1 else ("yellow" if minutes <= 5 else "red")
            rows.append(
                {
                    "name": table,
                    "last_ts_ms": int(last_ts) if last_ts is not None else None,
                    "last_utc": utc_iso_ms(last_ts) if last_ts is not None else "",
                    "minutes_since": minutes,
                    "rows_last_hour": int(count or 0),
                    "status": status,
                }
            )
        except Exception as exc:
            rows.append({"name": table, "last_ts_ms": None, "last_utc": "", "minutes_since": None, "rows_last_hour": 0, "status": "dead", "error": repr(exc)})
    return rows


def price_series(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, float | int]]:
    if not table_exists(conn, "mark_prices"):
        return []
    rows = fetchall(
        conn,
        """
        WITH bucketed AS (
          SELECT (ts_ms / 60000) * 60000 AS bucket_ms, ts_ms, mark_price
          FROM mark_prices
          WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?
        ),
        closes AS (
          SELECT bucket_ms, MAX(ts_ms) AS close_ts
          FROM bucketed
          GROUP BY bucket_ms
        )
        SELECT c.bucket_ms, b.mark_price
        FROM closes c
        JOIN bucketed b ON b.bucket_ms=c.bucket_ms AND b.ts_ms=c.close_ts
        ORDER BY c.bucket_ms ASC
        """,
        (start_ms, end_ms),
    )
    return [{"ts": int(r["bucket_ms"]), "close": float(r["mark_price"])} for r in rows]


def simple_trade_flow(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[str, float]:
    rows = fetchall(
        conn,
        """
        SELECT is_buyer_maker, COALESCE(SUM(notional),0.0) AS notion
        FROM agg_trades
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?
        GROUP BY is_buyer_maker
        """,
        (int(start_ms), int(end_ms)),
    )
    taker_buy = 0.0
    taker_sell = 0.0
    for row in rows:
        if int(row["is_buyer_maker"] or 0) == 0:
            taker_buy += float(row["notion"] or 0.0)
        else:
            taker_sell += float(row["notion"] or 0.0)
    total = taker_buy + taker_sell
    return {
        "taker_buy": taker_buy,
        "taker_sell": taker_sell,
        "imbalance": ((taker_buy - taker_sell) / total) if total > 0 else 0.0,
    }


def simple_liq_notional(conn: sqlite3.Connection, start_ms: int, end_ms: int, side: str = "SELL") -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(notional),0.0) FROM liquidations WHERE symbol='ETHUSDT' AND side=? AND ts_ms>=? AND ts_ms<?",
        (str(side).upper(), int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0] or 0.0) if row else 0.0


def simple_mark_ret(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, int(start_ms)),
    ).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, int(end_ms)),
    ).fetchone()
    if not a or not b or float(a[0] or 0.0) <= 0:
        return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0


def book_snapshot(conn: sqlite3.Connection, ts_ms: int) -> dict[str, float] | None:
    row = conn.execute(
        """
        SELECT bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker
        WHERE symbol='ETHUSDT' AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (int(ts_ms),),
    ).fetchone()
    if not row:
        return None
    bid = float(row[0] or 0.0)
    bid_qty = float(row[1] or 0.0)
    ask = float(row[2] or 0.0)
    ask_qty = float(row[3] or 0.0)
    spread_bps = float(row[5] or 0.0) * 10_000.0
    return {
        "bid_depth_usd": float(row[7]) if row[7] is not None else bid * bid_qty,
        "ask_depth_usd": ask * ask_qty,
        "spread_bps": spread_bps,
        "book_imbalance": float(row[6] or 0.0),
    }


def v02_navigation_quality_series(conn: sqlite3.Connection, prices: list[dict[str, float | int]]) -> dict[str, Any]:
    # Continuous point-in-time indicator. It is not a trade trigger; it gives a
    # 0-10 "environment quality" readout for the v0.2 maker LONG setup.
    points: list[dict[str, Any]] = []
    ema: float | None = None
    alpha = 0.25
    for p in prices:
        ts = int(p["ts"])
        book = book_snapshot(conn, ts)
        if not book:
            continue
        flow15 = simple_trade_flow(conn, ts - 15_000, ts)
        sell15 = simple_liq_notional(conn, ts - 15_000, ts, "SELL")
        sell5m = simple_liq_notional(conn, ts - 300_000, ts, "SELL")
        btc15 = simple_mark_ret(conn, "BTCUSDT", ts - 15_000, ts)
        score = 0
        tags: list[str] = []
        warnings: list[str] = []

        def add(cond: bool, tag: str, pts: int = 1) -> None:
            nonlocal score
            if cond:
                score += pts
                tags.append(tag)

        def warn(cond: bool, tag: str, pts: int = 1) -> None:
            nonlocal score
            if cond:
                score -= pts
                warnings.append(tag)

        add(float(book["bid_depth_usd"]) >= 135_423.8, "BID_OK", 2)
        add(float(book["spread_bps"]) <= 0.15, "SPREAD_CLEAN", 2)
        add(float(book["book_imbalance"]) >= 0.0, "BID_IMBALANCE")
        add(float(sell15) <= 250_000.0, "NO_LARGE_SELL_LIQ_15S")
        add(float(flow15["imbalance"]) > -0.25, "FLOW_NOT_HEAVY_SELL")
        add(btc15 is not None and float(btc15) > -10.0, "BTC_NOT_CRASHING")
        add(200_000.0 <= float(sell5m) <= 2_000_000.0, "SELL_CASCADE_CONTEXT", 2)

        warn(float(book["bid_depth_usd"]) < 75_000.0, "BID_THIN", 2)
        warn(float(book["spread_bps"]) > 0.35, "SPREAD_WIDE", 2)
        warn(float(sell15) > 1_000_000.0, "SELL_LIQ_RESTART_HEAVY", 2)
        warn(btc15 is not None and float(btc15) < -25.0, "BTC_DUMPING", 2)

        bounded = max(0, min(10, int(score)))
        ema = float(bounded) if ema is None else alpha * float(bounded) + (1.0 - alpha) * ema
        bucket = "NAV_HIGH" if bounded >= 7 else ("NAV_MID" if bounded >= 5 else "NAV_LOW")
        points.append(
            {
                "ts": ts,
                "utc": utc_iso_ms(ts),
                "score": bounded,
                "ema_score": round(float(ema), 3),
                "bucket": bucket,
                "tags": ",".join(tags),
                "warnings": ",".join(warnings),
                "bid_depth_usd": round(float(book["bid_depth_usd"]), 1),
                "spread_bps": round(float(book["spread_bps"]), 3),
                "sell_liq_5m": round(float(sell5m), 1),
            }
        )
    return {"available": bool(points), "series": points, "latest": points[-1] if points else None, "ema_alpha": alpha}


def liq_buckets(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[dict[str, float | int]]:
    if not table_exists(conn, "liquidations"):
        return []
    rows = fetchall(
        conn,
        """
        SELECT (ts_ms / 300000) * 300000 AS bucket_ms, COALESCE(SUM(notional), 0.0) AS notional
        FROM liquidations
        WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<=?
        GROUP BY bucket_ms
        ORDER BY bucket_ms ASC
        """,
        (start_ms, end_ms),
    )
    return [{"ts": int(r["bucket_ms"]), "notional": float(r["notional"])} for r in rows]


def day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)


def regime_status(conn: sqlite3.Connection, now_ms: int) -> dict[str, Any]:
    start = day_start_ms(now_ms)
    thresholds = REGIME_THRESHOLDS.copy()
    empty = {
        "trend_pct": 0.0,
        "range_pct": 0.0,
        "buy_liq_notional": 0.0,
        "agg_count": 0,
        "thresholds": thresholds,
        "gates": {"trend_pct": False, "range_pct": False, "buy_liq_notional": False, "agg_count": False},
        "regime_on": False,
    }
    if not all(table_exists(conn, t) for t in ("mark_prices", "liquidations", "agg_trades")):
        return empty
    first = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC LIMIT 1",
        (start, now_ms),
    ).fetchone()
    current = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (now_ms,),
    ).fetchone()
    low_high = conn.execute(
        "SELECT MIN(mark_price), MAX(mark_price) FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
        (start, now_ms),
    ).fetchone()
    buy_liq = conn.execute(
        "SELECT COALESCE(SUM(notional), 0.0) FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY' AND ts_ms>=? AND ts_ms<=?",
        (start, now_ms),
    ).fetchone()[0]
    agg_count = conn.execute(
        "SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?",
        (start, now_ms),
    ).fetchone()[0]
    if not first or not current or not low_high or not first[0]:
        return empty
    first_px = float(first[0])
    current_px = float(current[0])
    low = float(low_high[0] or first_px)
    high = float(low_high[1] or first_px)
    trend_pct = (current_px - first_px) / first_px * 100.0
    range_pct = (high - low) / low * 100.0 if low else 0.0
    gates = {
        "trend_pct": trend_pct >= thresholds["trend_pct"],
        "range_pct": range_pct >= thresholds["range_pct"],
        "buy_liq_notional": float(buy_liq or 0.0) >= thresholds["buy_liq_notional"],
        "agg_count": int(agg_count or 0) >= thresholds["agg_count"],
    }
    return {
        "trend_pct": trend_pct,
        "range_pct": range_pct,
        "buy_liq_notional": float(buy_liq or 0.0),
        "agg_count": int(agg_count or 0),
        "thresholds": thresholds,
        "gates": gates,
        "regime_on": all(gates.values()),
        "day_start_utc": utc_iso_ms(start),
    }


def valid_trade(trade: dict[str, Any]) -> bool:
    if str(trade.get("trade_id") or "") in EXCLUDED_TRADE_IDS:
        return False
    if str((trade.get("rule") or {}).get("name") or "") != RULE_NAME:
        return False
    if trade.get("status") != "CLOSED":
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    needed = ("gross_bps", "entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps")
    return all(trade.get(k) is not None for k in needed)


def dashboard_valid_trade(trade: dict[str, Any]) -> bool:
    if str(trade.get("trade_id") or "") in EXCLUDED_TRADE_IDS:
        return False
    if not dashboard_trade_visible(trade):
        return False
    if trade.get("status") != "CLOSED":
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    needed = ("gross_bps", "entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps")
    return all(trade.get(k) is not None for k in needed)


def rule_label(rule_name: str) -> str:
    if "S34_STATE_MACHINE_V1" in rule_name:
        return "State Machine v1 BTC1000/DOW/score3"
    if "S34_V_ENGINE_V0_2" in rule_name:
        return "V Engine v0.2 deep bid"
    if "S34_V_ENGINE_V0_1" in rule_name:
        return "V Engine v0.1"
    if "ETH_SELL_LIQ_SHORT_1M" in rule_name:
        return "ETH SELL 1M"
    if "ETH_SELL_LIQ_SHORT_500K" in rule_name:
        return "ETH SELL 500K"
    if "SOL_SELL_LIQ_SHORT_100K" in rule_name:
        return "SOL SELL 100K"
    if "SOL_SELL_LIQ_SHORT_200K" in rule_name:
        return "SOL SELL 200K"
    if "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED" in rule_name:
        return "BTC 1M/dist"
    if "SOL_BUY_LIQ_LONG_100K_TP60" in rule_name:
        return "SOL 100K/TP60"
    if "SOL_BUY_LIQ_LONG_200K_TP60" in rule_name:
        return "SOL 200K/TP60"
    if "500K_NEGTREND_STRETCHED" in rule_name:
        return "500K/neg-stretched"
    if "500K_DAYTREND0_TP60" in rule_name:
        return "500K/DT0/TP60"
    if "200K_BTC_PRE15_TP120" in rule_name:
        return "200K/BTC15/TP120"
    if "200K_TP60" in rule_name:
        return "200K/TP60"
    if "50K_TP120" in rule_name:
        return "50K/TP120"
    return rule_name


def unrealized_bps(trade: dict[str, Any], current_price: float | None) -> float | None:
    if current_price is None:
        return None
    entry = float(trade.get("entry_price") or 0.0)
    if entry <= 0:
        return None
    raw = (float(current_price) - entry) / entry * 1e4
    return raw if str(trade.get("direction") or "LONG").upper() == "LONG" else -raw


def latest_mark_price(conn: sqlite3.Connection, symbol: str) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return None if not row else float(row[0])


def current_prices(conn: sqlite3.Connection, trades: list[dict[str, Any]]) -> dict[str, float]:
    symbols = {CHART_SYMBOL, "BTCUSDT", "SOLUSDT"}
    for trade in trades:
        symbol = str(trade.get("symbol") or "")
        if symbol:
            symbols.add(symbol)
    out: dict[str, float] = {}
    for symbol in symbols:
        price = latest_mark_price(conn, symbol)
        if price is not None:
            out[symbol] = price
    return out


def trade_rule_name(trade: dict[str, Any]) -> str:
    return str((trade.get("rule") or {}).get("name") or "")


def dashboard_rule_visible(rule_name: str) -> bool:
    return str(rule_name or "") not in HIDDEN_DASHBOARD_RULES


def dashboard_trade_visible(trade: dict[str, Any]) -> bool:
    return dashboard_rule_visible(trade_rule_name(trade))


def num_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def constellation_trade_valid(trade: dict[str, Any]) -> bool:
    if str(trade.get("trade_id") or "") in EXCLUDED_TRADE_IDS:
        return False
    if trade.get("status") != "CLOSED":
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    return trade.get("net_bps") is not None


def constellation_routes(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        if not constellation_trade_valid(trade):
            continue
        name = trade_rule_name(trade)
        if not name or not dashboard_rule_visible(name):
            continue
        groups.setdefault(name, []).append(trade)

    route_order = [
        "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
        "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
        "ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
        "ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60",
        "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
        "ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40",
        "SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30",
        "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
        "SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40",
        "SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30",
        "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
    ]
    for name in sorted(groups):
        if name not in route_order:
            route_order.append(name)

    rows: list[dict[str, Any]] = []
    for name in route_order:
        route_trades = sorted(groups.get(name, []), key=lambda t: (int(t.get("entry_ts_ms") or 0), str(t.get("trade_id") or "")))
        if not route_trades:
            continue
        nets = [float(t.get("net_bps") or 0.0) for t in route_trades]
        exported = []
        for trade in route_trades:
            signal = trade.get("signal") or {}
            exported.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "entry_ts": trade.get("entry_ts_utc") or trade.get("signal_ts_utc"),
                    "exit_ts": trade.get("exit_ts_utc"),
                    "exit_type": trade.get("exit_reason"),
                    "entry_price": num_or_none(trade.get("entry_price")),
                    "exit_price": num_or_none(trade.get("exit_price")),
                    "net_bps": num_or_none(trade.get("net_bps")),
                    "mfe_bps": num_or_none(trade.get("mfe_bps")),
                    "mae_bps": num_or_none(trade.get("mae_bps")),
                    "cluster_notional": num_or_none(signal.get("liq_total_notional")),
                    "cluster_liq_count": signal.get("liq_count"),
                }
            )
        rows.append(
            {
                "route_id": name,
                "rule_name": name,
                "label": rule_label(name),
                "category": "pre_reg" if name == RULE_NAME else "exploratory_live",
                "n_closed": len(route_trades),
                "n_valid": len(route_trades),
                "median_net_bps": statistics.median(nets) if nets else None,
                "mean_net_bps": statistics.fmean(nets) if nets else None,
                "win_rate": (sum(1 for value in nets if value > 0.0) / len(nets)) if nets else None,
                "cum_net_bps": sum(nets),
                "trades": exported,
            }
        )
    return rows


def trade_status_payload(trades: list[dict[str, Any]], prices_by_symbol: dict[str, float], end_ms: int) -> dict[str, Any]:
    return trade_status_payload_with_guardrails(trades, prices_by_symbol, end_ms, {})


_SOL_200K_RULE = "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30"
_SOL_WEAK_TAG  = "SOL_WEAK_GEOMETRY_SHADOW"


def _sol_geometry_tag(trade: dict[str, Any]) -> dict[str, Any] | None:
    """Returns geometry tag dict if SOL 200K trade matches weak geometry criteria, else None."""
    if trade_rule_name(trade) != _SOL_200K_RULE:
        return None
    sig   = trade.get("signal") or {}
    cas   = sig.get("liq_total_notional")
    cnt   = sig.get("liq_count")
    mx    = sig.get("liq_max_notional")
    share = (mx / cas * 100) if (cas and mx and cas > 0) else None
    reasons: list[str] = []
    if cas is not None and 500_000 <= cas < 1_000_000:
        reasons.append("cascade_500K_1M")
    if share is not None and share >= 80.0:
        reasons.append(f"single_share_{share:.0f}pct")
    if cnt is not None and cnt <= 2:
        reasons.append(f"liq_count_{int(cnt)}")
    if not reasons:
        return None
    return {
        "tag": _SOL_WEAK_TAG,
        "reasons": reasons,
        "cascade_usd": cas,
        "liq_count": cnt,
        "single_share": round(share, 1) if share is not None else None,
    }


def _geometry_summary_payload() -> dict[str, Any]:
    """Read SOL 200K geometry tag summary from s34_shadow_geometry_tags."""
    if not INTELLIGENCE_DB_PATH.exists():
        return {"available": False}
    try:
        con = sqlite3.connect(f"file:{INTELLIGENCE_DB_PATH}?mode=ro", uri=True, timeout=2.0)
        tag_rows = con.execute(
            "SELECT trade_id, net_bps FROM s34_shadow_geometry_tags WHERE tag=?",
            (_SOL_WEAK_TAG,),
        ).fetchall()
        sol_rows = con.execute(
            "SELECT trade_id, net_bps FROM s34_trades "
            "WHERE status='CLOSED' AND rule_name=? AND net_bps IS NOT NULL",
            (_SOL_200K_RULE,),
        ).fetchall()
        con.close()
        tagged_ids  = {r[0] for r in tag_rows}
        tagged_nets = [float(r[1]) for r in tag_rows if r[1] is not None]
        all_sol     = {r[0]: float(r[1]) for r in sol_rows}
        clean_nets  = [v for k, v in all_sol.items() if k not in tagged_ids]

        def _st(nets: list[float]) -> dict[str, Any] | None:
            if not nets:
                return None
            return {
                "n":      len(nets),
                "median": round(statistics.median(nets), 1),
                "wr":     round(sum(1 for n in nets if n > 0) / len(nets), 2),
                "cum":    round(sum(nets), 1),
            }

        return {
            "available": True,
            "tagged_ids": list(tagged_ids),
            "tagged": _st(tagged_nets),
            "clean":  _st(clean_nets),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "tagged_ids": []}


def trade_status_payload_with_guardrails(
    trades: list[dict[str, Any]],
    prices_by_symbol: dict[str, float],
    end_ms: int,
    guardrails_by_rule: dict[str, dict[str, Any]],
    geometry_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cards = []
    open_rows = []
    closed_rows = [t for t in trades if t.get("status") == "CLOSED" and dashboard_trade_visible(t)]
    latest_closed = sorted(closed_rows, key=lambda t: str(t.get("closed_at_utc") or t.get("exit_ts_utc") or ""))[-10:]
    for rule_name, label, target_n in FORWARD_RULES:
        rows = [t for t in trades if trade_rule_name(t) == rule_name]
        closed = [t for t in rows if t.get("status") == "CLOSED"]
        skipped = [t for t in rows if t.get("status") == "SKIPPED"]
        opened = [t for t in rows if t.get("status") == "OPEN"]
        skips: dict[str, int] = {}
        for trade in skipped:
            reason = str(trade.get("risk_gate_reason") or trade.get("exit_reason") or "UNKNOWN")
            skips[reason] = skips.get(reason, 0) + 1
        last_closed = sorted(closed, key=lambda t: str(t.get("closed_at_utc") or t.get("exit_ts_utc") or ""))[-1:] or [None]
        valid_n = None
        progress = None
        if rule_name == RULE_NAME:
            valid_n = len([t for t in rows if valid_trade(t)])
            progress = min(1.0, valid_n / float(target_n or 1))
        for trade in opened:
            symbol = str(trade.get("symbol") or "ETHUSDT")
            open_rows.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "rule_id": rule_name,
                    "rule_label": rule_label(rule_name),
                    "symbol": symbol,
                    "entry_ts": trade.get("entry_ts_ms") or trade.get("signal_ts_ms"),
                    "entry_utc": trade.get("entry_ts_utc") or trade.get("signal_ts_utc"),
                    "entry_price": trade.get("entry_price"),
                    "unrealized_bps": unrealized_bps(trade, prices_by_symbol.get(symbol)),
                    "geometry_tag": _sol_geometry_tag(trade),
                }
            )
        card_geometry = None
        if rule_name == _SOL_200K_RULE and geometry_summary:
            card_geometry = geometry_summary
        cards.append(
            {
                "rule_id": rule_name,
                "label": label,
                "trials": len(rows),
                "closed": len(closed),
                "open": len(opened),
                "skipped": len(skipped),
                "valid_n": valid_n,
                "target_n": target_n,
                "progress": progress,
                "skips": skips,
                "last_closed": None
                if last_closed[0] is None
                else {
                    "trade_id": last_closed[0].get("trade_id"),
                    "exit_reason": last_closed[0].get("exit_reason"),
                    "net_bps": last_closed[0].get("net_bps"),
                    "closed_at_utc": last_closed[0].get("closed_at_utc") or last_closed[0].get("exit_ts_utc"),
                },
                "guardrails": guardrails_by_rule.get(rule_name, {}),
                "geometry_summary": card_geometry,
            }
        )
    return {
        "cards": cards,
        "open_trades": open_rows,
        "latest_closed": [
            {
                "trade_id": t.get("trade_id"),
                "rule_id": trade_rule_name(t),
                "rule_label": rule_label(trade_rule_name(t)),
                "exit_reason": t.get("exit_reason"),
                "net_bps": t.get("net_bps"),
                "closed_at_utc": t.get("closed_at_utc") or t.get("exit_ts_utc"),
                "geometry_tag": _sol_geometry_tag(t),
            }
            for t in latest_closed
        ],
        "asof_ts_ms": end_ms,
    }


def disk_status() -> dict[str, Any]:
    usage = shutil.disk_usage(str(ROOT.drive + "\\") if ROOT.drive else str(ROOT))
    micro = DB_PATH.stat().st_size if DB_PATH.exists() else None
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_pct": usage.used / usage.total if usage.total else None,
        "microstructure_bytes": micro,
        "smoke_microstructure_bytes": (ROOT / "logs" / "smoke_microstructure.db").stat().st_size
        if (ROOT / "logs" / "smoke_microstructure.db").exists()
        else None,
        "lead_lag_work_bytes": (ROOT / "data" / "lead_lag_work.db").stat().st_size
        if (ROOT / "data" / "lead_lag_work.db").exists()
        else None,
    }


_HOST_HEALTH_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_HOST_HEALTH_CACHE_TTL_SEC = 20.0
_HOST_HEALTH_RAM_HISTORY: list[tuple[float, float]] = []
_HOST_HEALTH_COMMIT_HISTORY: list[tuple[float, float]] = []


def host_health_payload() -> dict[str, Any]:
    """Read-only PC/host restart-readiness snapshot -- see ami/host_health/.
    Never restarts, shuts down, or modifies anything. Cached so the 3s
    client poll doesn't spawn a PowerShell subprocess every tick."""
    global _HOST_HEALTH_CACHE
    now = time.monotonic()
    cached_ts, cached_payload = _HOST_HEALTH_CACHE
    if now - cached_ts < _HOST_HEALTH_CACHE_TTL_SEC and cached_payload:
        return cached_payload
    try:
        import dataclasses

        from ami.host_health.evaluator import D_DRIVE_INTERVENTION_FREE_GB, SUSTAINED_WINDOW_MINUTES, evaluate_restart_readiness
        from ami.host_health.observation import build_health_inputs, collect_host_observation, sustained_value

        obs = collect_host_observation(repo_root=ROOT)
        wall_now = time.time()
        if obs.ram_used_pct is not None:
            _HOST_HEALTH_RAM_HISTORY.append((wall_now, obs.ram_used_pct))
            del _HOST_HEALTH_RAM_HISTORY[:-200]
        if obs.commit_used_pct is not None:
            _HOST_HEALTH_COMMIT_HISTORY.append((wall_now, obs.commit_used_pct))
            del _HOST_HEALTH_COMMIT_HISTORY[:-200]
        ram_sustained = sustained_value(_HOST_HEALTH_RAM_HISTORY, SUSTAINED_WINDOW_MINUTES, wall_now)
        commit_sustained = sustained_value(_HOST_HEALTH_COMMIT_HISTORY, SUSTAINED_WINDOW_MINUTES, wall_now)
        inputs = build_health_inputs(obs, ram_pct_sustained=ram_sustained, commit_pct_sustained=commit_sustained)
        evaluation = evaluate_restart_readiness(inputs)
        payload = {
            "available": True,
            "state": evaluation.state,
            "recommended_action": evaluation.recommended_action,
            "deferred": evaluation.deferred,
            "primary_reason": evaluation.primary_reason,
            "reasons": list(evaluation.reason_codes),
            "unknown_fields": list(dict.fromkeys(list(evaluation.unknown_fields) + list(obs.unknown_fields))),
            "observation_timestamp": obs.observation_ts_utc,
            "no_automatic_action": True,
            "d_drive_intervention_free_gb": D_DRIVE_INTERVENTION_FREE_GB,
            "observations": dataclasses.asdict(obs),
        }
    except Exception as exc:  # noqa: BLE001 - never let host-health break the main chart
        payload = {"available": False, "state": "HOST_RESTART_UNKNOWN", "error": repr(exc)}
    _HOST_HEALTH_CACHE = (now, payload)
    return payload


def guardrail_v3_dashboard_payload() -> dict[str, Any]:
    raw = read_json(GUARDRAIL_V3_AUDIT_PATH, {})
    if not isinstance(raw, dict) or not raw:
        return {"available": False}
    candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
    feature_inventory = raw.get("feature_inventory") if isinstance(raw.get("feature_inventory"), dict) else {}
    feature_focus = {}
    for key in ("day_trend_bps", "intensity_per_sec", "inter_cluster_gap_sec", "max_single_liq_share"):
        data = feature_inventory.get(key) if isinstance(feature_inventory.get(key), dict) else {}
        feature_focus[key] = {
            "closed_rows": data.get("closed_rows_with_feature"),
            "coverage_pct": data.get("coverage_pct"),
            "warning_rows": data.get("warning_rows_with_feature"),
        }
    relevant = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        metric = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        kept = item.get("kept_after_shadow_block") if isinstance(item.get("kept_after_shadow_block"), dict) else {}
        relevant.append(
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "definition": item.get("definition"),
                "feature_coverage_pct": item.get("feature_coverage_pct"),
                "block_n": metric.get("n"),
                "block_cum_net_bps": metric.get("cum_net_bps"),
                "block_median_net_bps": metric.get("median_net_bps"),
                "kept_delta_bps": kept.get("delta_cum_vs_baseline_bps"),
            }
        )
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        examples = item.get("examples") if isinstance(item.get("examples"), list) else []
        rules = {
            str(example.get("rule_name") or "")
            for example in examples
            if isinstance(example, dict) and example.get("rule_name")
        }
        definition = str(item.get("definition") or "")
        if "50K/TP120" in definition:
            rules.add("ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30")
        for rule_name in rules:
            by_rule.setdefault(rule_name, []).append(
                {
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "block_n": (item.get("candidate") or {}).get("n") if isinstance(item.get("candidate"), dict) else None,
                    "block_cum_net_bps": (item.get("candidate") or {}).get("cum_net_bps") if isinstance(item.get("candidate"), dict) else None,
                    "kept_delta_bps": (item.get("kept_after_shadow_block") or {}).get("delta_cum_vs_baseline_bps")
                    if isinstance(item.get("kept_after_shadow_block"), dict)
                    else None,
                }
            )
    auditable = [item for item in relevant if item.get("status") == "auditable_shadow_only"]
    too_early = [item for item in relevant if str(item.get("status") or "").startswith("too_early")]
    return {
        "available": True,
        "generated_at_utc": raw.get("generated_at_utc"),
        "verdict": raw.get("verdict"),
        "candidate_count": len(relevant),
        "auditable_count": len(auditable),
        "too_early_count": len(too_early),
        "feature_focus": feature_focus,
        "top_candidates": relevant[:4],
        "by_rule": by_rule,
    }


def route_guardrail_payload(limit_per_rule: int = 500) -> dict[str, dict[str, Any]]:
    if not INTELLIGENCE_DB_PATH.exists():
        return {}
    v3_payload = guardrail_v3_dashboard_payload()
    con = sqlite3.connect(f"file:{INTELLIGENCE_DB_PATH}?mode=ro", uri=True, timeout=3.0)
    con.row_factory = sqlite3.Row
    try:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='s34_model_guardrails'"
        ).fetchone()
        if not exists:
            return {}
        rows = con.execute(
            """
            SELECT
              s.rule_name,
              s.signal_ts_ms,
              g.level,
              g.headline,
              g.guardrail_json,
              o.trade_id,
              o.exit_reason,
              o.net_bps
            FROM s34_model_guardrails g
            JOIN s34_signals s ON s.signal_id=g.signal_id
            LEFT JOIN s34_outcomes o ON o.signal_id=g.signal_id
            ORDER BY s.signal_ts_ms DESC
            LIMIT 2000
            """
        ).fetchall()
        shadow_exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='s34_shadow_guardrails'"
        ).fetchone()
        shadow_rows = []
        if shadow_exists:
            shadow_rows = con.execute(
                """
                SELECT
                  s.rule_name,
                  s.signal_ts_ms,
                  sg.guardrail_name,
                  sg.action,
                  sg.level,
                  sg.headline,
                  sg.shadow_json,
                  o.trade_id,
                  o.exit_reason,
                  o.net_bps
                FROM s34_shadow_guardrails sg
                JOIN s34_signals s ON s.signal_id=sg.signal_id
                LEFT JOIN s34_outcomes o ON o.signal_id=sg.signal_id
                WHERE sg.guardrail_name IN ('guardrail_v2_warning_100k_200k', 'guardrail_v4_50k_warning_lt200k')
                ORDER BY s.signal_ts_ms DESC
                LIMIT 2000
                """
            ).fetchall()
    finally:
        con.close()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rule_name = str(row["rule_name"] or "")
        if not rule_name:
            continue
        bucket = grouped.setdefault(rule_name, [])
        if len(bucket) >= limit_per_rule:
            continue
        bucket.append(dict(row))

    shadow_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in shadow_rows:
        rule_name = str(row["rule_name"] or "")
        guardrail_name = str(row["guardrail_name"] or "")
        if not rule_name:
            continue
        bucket = shadow_grouped.setdefault((rule_name, guardrail_name), [])
        if len(bucket) >= limit_per_rule:
            continue
        bucket.append(dict(row))

    out: dict[str, dict[str, Any]] = {}
    for rule_name, items in grouped.items():
        level_counts: dict[str, int] = {}
        closed = []
        for item in items:
            level = str(item.get("level") or "unknown")
            level_counts[level] = level_counts.get(level, 0) + 1
            if item.get("net_bps") is not None:
                closed.append(item)
        losses = [item for item in closed if float(item.get("net_bps") or 0.0) < 0.0]
        wins = [item for item in closed if float(item.get("net_bps") or 0.0) > 0.0]
        nets = [float(item.get("net_bps") or 0.0) for item in closed]
        latest = items[0] if items else {}
        def summarize_shadow(name: str) -> dict[str, Any]:
            shadow_items = shadow_grouped.get((rule_name, name), [])
            would_block = [item for item in shadow_items if str(item.get("action") or "") == "would_block"]
            would_block_closed = [item for item in would_block if item.get("net_bps") is not None]
            would_block_nets = [float(item.get("net_bps") or 0.0) for item in would_block_closed]
            shadow_latest = shadow_items[0] if shadow_items else {}
            return {
                "name": name,
                "sample_n": len(shadow_items),
                "would_block_n": len(would_block),
                "would_block_closed_n": len(would_block_closed),
                "would_block_cum_net_bps": None if not would_block_nets else sum(would_block_nets),
                "would_block_median_net_bps": None if not would_block_nets else statistics.median(would_block_nets),
                "latest_action": shadow_latest.get("action"),
                "latest_level": shadow_latest.get("level"),
                "latest_headline": shadow_latest.get("headline"),
            }
        out[rule_name] = {
            "sample_n": len(items),
            "level_counts": level_counts,
            "latest_level": latest.get("level"),
            "latest_headline": latest.get("headline"),
            "latest_trade_id": latest.get("trade_id"),
            "latest_exit_reason": latest.get("exit_reason"),
            "latest_net_bps": latest.get("net_bps"),
            "closed_n": len(closed),
            "loss_rate": None if not closed else len(losses) / len(closed),
            "win_rate": None if not closed else len(wins) / len(closed),
            "median_net_bps": None if not nets else statistics.median(nets),
            "mean_net_bps": None if not nets else statistics.fmean(nets),
            "shadow_v2": summarize_shadow("guardrail_v2_warning_100k_200k"),
            "shadow_v4": summarize_shadow("guardrail_v4_50k_warning_lt200k"),
            "v3_audit": {
                "available": v3_payload.get("available", False),
                "generated_at_utc": v3_payload.get("generated_at_utc"),
                "auditable_count": v3_payload.get("auditable_count"),
                "too_early_count": v3_payload.get("too_early_count"),
                "candidate_count": v3_payload.get("candidate_count"),
                "feature_focus": v3_payload.get("feature_focus", {}),
                "rule_candidates": (v3_payload.get("by_rule") or {}).get(rule_name, []),
                "verdict": v3_payload.get("verdict"),
            },
        }
    return out


def _short_money(value: Any) -> str:
    try:
        n = float(value or 0.0)
    except (TypeError, ValueError):
        return "-"
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def _build_intelligence_explanation(
    latest_signal: dict[str, Any] | None, cluster_decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not latest_signal:
        return {
            "headline": "Ledger is waiting for the next signal.",
            "reasoning": [],
            "accepted": [],
            "rejected": [],
        }
    accepted = [row for row in cluster_decisions if row.get("decision") == "ACCEPT"]
    rejected = [row for row in cluster_decisions if row.get("decision") == "REJECT"]
    closed = [row for row in cluster_decisions if row.get("decision") == "CLOSE"]
    cluster_text = (
        f"{latest_signal.get('symbol')} {latest_signal.get('direction')} candidate from "
        f"{_short_money(latest_signal.get('cluster_notional'))} {latest_signal.get('cluster_shape_label') or 'unknown-shape'} "
        f"cluster ({latest_signal.get('cluster_liq_count') or 0} liqs)."
    )
    if accepted:
        action = "Accepted " + ", ".join(rule_label(str(row.get("rule_name") or "")) for row in accepted[:3])
    elif rejected:
        action = "Rejected all candidate rules"
    else:
        action = "No decision rows found for this signal"
    reasons: list[str] = [cluster_text, action + "."]
    if rejected:
        grouped: dict[str, int] = {}
        for row in rejected:
            reason = str(row.get("reason") or "UNKNOWN")
            grouped[reason] = grouped.get(reason, 0) + 1
        reasons.append(
            "Reject map: "
            + ", ".join(f"{reason} x{count}" for reason, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0])))
            + "."
        )
    if closed:
        reasons.append(
            "A linked trade closed: "
            + ", ".join(f"{row.get('trade_id')} {row.get('reason') or ''}" for row in closed[:3])
            + "."
        )
    return {
        "headline": f"{cluster_text} {action}.",
        "reasoning": reasons,
        "accepted": accepted,
        "rejected": rejected,
    }


def _stats_from_rows(rows: list[sqlite3.Row]) -> dict[str, Any]:
    values = [float(row["net_bps"]) for row in rows if row["net_bps"] is not None]
    if not values:
        return {"n": 0, "median_net_bps": None, "mean_net_bps": None, "win_rate": None, "cum_net_bps": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        median = ordered[n // 2]
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "n": n,
        "median_net_bps": median,
        "mean_net_bps": sum(values) / n,
        "win_rate": sum(1 for value in values if value > 0) / n,
        "cum_net_bps": sum(values),
    }


def _base_rate_payload(con: sqlite3.Connection, latest_signal: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not latest_signal:
        return []
    rule_name = str(latest_signal.get("rule_name") or "")
    symbol = str(latest_signal.get("symbol") or "")
    direction = str(latest_signal.get("direction") or "")
    shape = str(latest_signal.get("cluster_shape_label") or "")
    buckets: list[tuple[str, str, tuple[Any, ...]]] = [
        (
            "same_rule",
            f"Same rule: {rule_label(rule_name)}",
            (rule_name,),
        ),
        (
            "same_shape",
            f"Same shape: {shape or 'unknown'}",
            (shape,),
        ),
        (
            "same_symbol_direction",
            f"Same symbol/direction: {symbol} {direction}",
            (symbol, direction),
        ),
    ]
    out: list[dict[str, Any]] = []
    queries = {
        "same_rule": """
            SELECT o.net_bps
            FROM s34_outcomes o
            WHERE o.rule_name=?
        """,
        "same_shape": """
            SELECT o.net_bps
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            WHERE s.cluster_shape_label=?
        """,
        "same_symbol_direction": """
            SELECT o.net_bps
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            WHERE s.symbol=? AND s.direction=?
        """,
    }
    for key, label, params in buckets:
        rows = con.execute(queries[key], params).fetchall()
        item = {"key": key, "label": label, **_stats_from_rows(rows)}
        item["confidence_note"] = "thin" if int(item["n"] or 0) < 20 else "usable"
        out.append(item)
    return out


def _model_calibration(con: sqlite3.Connection, model_name: str) -> dict[str, Any]:
    rows = con.execute(
        """
        SELECT p.prediction_json, o.trade_id, o.exit_reason, o.net_bps
        FROM s34_predictions p
        JOIN s34_outcomes o ON o.signal_id=p.signal_id
        WHERE p.model_name=?
        ORDER BY o.exit_ts_ms ASC
        """,
        (model_name,),
    ).fetchall()
    pairs: list[dict[str, Any]] = []
    for row in rows:
        try:
            prediction = json.loads(str(row["prediction_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        expected = prediction.get("expected_net_bps")
        actual = row["net_bps"]
        if expected is None or actual is None:
            continue
        error = float(actual) - float(expected)
        pairs.append(
            {
                "trade_id": row["trade_id"],
                "exit_reason": row["exit_reason"],
                "expected_net_bps": float(expected),
                "actual_net_bps": float(actual),
                "error_bps": error,
            }
        )
    summary = _calibration_summary(pairs)
    summary["latest"] = pairs[-8:]
    return summary


def _calibration_payload(con: sqlite3.Connection) -> dict[str, Any]:
    base_summary = _model_calibration(con, "base_rate_v1")
    knn_summary = _model_calibration(con, "knn_v0")
    knn_v1_summary = _model_calibration(con, "knn_v1")
    knn_v2_summary = _model_calibration(con, "knn_v2")
    return {
        "n": base_summary.get("n", 0),
        "bias_bps": base_summary["bias_bps"],
        "mae_bps": base_summary["mae_bps"],
        "hit_direction_rate": base_summary["hit_direction_rate"],
        "optimism_rate": base_summary["optimism_rate"],
        "base_rate": base_summary,
        "knn_v0": knn_summary,
        "knn_v1": knn_v1_summary,
        "knn_v2": knn_v2_summary,
        "latest": base_summary.get("latest", []),
        "note": "diagnostic; backfilled predictions are not holdout",
    }


def _calibration_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "bias_bps": None, "mae_bps": None, "hit_direction_rate": None, "optimism_rate": None}
    errors = [row["error_bps"] for row in pairs]
    abs_errors = [abs(value) for value in errors]
    direction_hits = [
        row
        for row in pairs
        if (row["expected_net_bps"] >= 0 and row["actual_net_bps"] >= 0)
        or (row["expected_net_bps"] < 0 and row["actual_net_bps"] < 0)
    ]
    optimistic = [row for row in pairs if row["expected_net_bps"] > row["actual_net_bps"]]
    return {
        "n": len(pairs),
        "bias_bps": sum(errors) / len(errors),
        "mae_bps": sum(abs_errors) / len(abs_errors),
        "hit_direction_rate": len(direction_hits) / len(pairs),
        "optimism_rate": len(optimistic) / len(pairs),
    }


def _model_guardrail(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for row in predictions:
        prediction = row.get("prediction") or {}
        expected = prediction.get("expected_net_bps")
        if expected is None:
            continue
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            continue
        values.append(
            {
                "model_name": row.get("model_name"),
                "expected_net_bps": expected_value,
                "confidence_note": prediction.get("confidence_note"),
                "k": prediction.get("k"),
                "win_rate": prediction.get("win_rate"),
            }
        )
    if not values:
        return {"level": "unknown", "headline": "No usable model prediction yet.", "reasons": [], "models": []}

    negative = [row for row in values if row["expected_net_bps"] < 0]
    strongly_negative = [row for row in values if row["expected_net_bps"] <= -30]
    positive = [row for row in values if row["expected_net_bps"] > 0]
    reasons: list[str] = []
    if len(negative) >= 3:
        reasons.append(f"{len(negative)}/{len(values)} models expect negative net bps")
    if strongly_negative:
        names = ", ".join(str(row["model_name"]) for row in strongly_negative[:3])
        reasons.append(f"strong negative warning from {names}")
    if positive and negative:
        reasons.append("models disagree; treat confidence as low")

    if len(negative) >= 3 or len(strongly_negative) >= 2:
        level = "warning"
        headline = "MODEL WARNING: similar signals have negative expectancy."
    elif len(negative) >= 1 and len(positive) >= 1:
        level = "caution"
        headline = "MODEL CAUTION: predictions disagree."
    else:
        level = "ok"
        headline = "MODEL OK: no negative consensus."
    return {"level": level, "headline": headline, "reasons": reasons, "models": values}


def intelligence_payload() -> dict[str, Any]:
    empty = {
        "available": False,
        "db_age_sec": None,
        "counts": {"signals": 0, "accept": 0, "reject": 0, "close": 0, "outcomes": 0},
        "latest_signal": None,
        "latest_decisions": [],
        "cluster_decisions": [],
        "explanation": {"headline": "Ledger unavailable.", "reasoning": [], "accepted": [], "rejected": []},
        "base_rates": [],
        "latest_prediction": None,
        "latest_model_audit": None,
        "prediction_vs_outcome": None,
        "calibration": {"n": 0, "note": "ledger unavailable"},
        "latest_outcomes": [],
        "reject_reasons": [],
        "error": "",
    }
    if not INTELLIGENCE_DB_PATH.exists():
        return empty
    try:
        con = sqlite3.connect(f"file:{INTELLIGENCE_DB_PATH}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        try:
            counts = dict(empty["counts"])
            counts["signals"] = int(con.execute("SELECT COUNT(*) FROM s34_signals").fetchone()[0])
            counts["accept"] = int(con.execute("SELECT COUNT(*) FROM s34_decisions WHERE decision='ACCEPT'").fetchone()[0])
            counts["reject"] = int(con.execute("SELECT COUNT(*) FROM s34_decisions WHERE decision='REJECT'").fetchone()[0])
            counts["close"] = int(con.execute("SELECT COUNT(*) FROM s34_decisions WHERE decision='CLOSE'").fetchone()[0])
            counts["outcomes"] = int(con.execute("SELECT COUNT(*) FROM s34_outcomes").fetchone()[0])
            latest_signal_row = con.execute(
                """
                SELECT signal_id, signal_ts_ms, signal_ts_utc, rule_name, symbol, direction,
                       cluster_notional, cluster_liq_count, cluster_shape_label
                FROM s34_signals
                ORDER BY signal_ts_ms DESC
                LIMIT 1
                """
            ).fetchone()
            latest_decisions = [
                dict(row)
                for row in con.execute(
                    """
                    SELECT decision_id, signal_id, trade_id, signal_ts_ms, rule_name, decision, reason, decision_ts_utc
                    FROM s34_decisions
                    ORDER BY signal_ts_ms DESC, decision_ts_utc DESC
                    LIMIT 8
                    """
                ).fetchall()
            ]
            cluster_decisions: list[dict[str, Any]] = []
            if latest_signal_row:
                cluster_decisions = [
                    dict(row)
                    for row in con.execute(
                        """
                        SELECT decision_id, signal_id, trade_id, signal_ts_ms, rule_name, decision, reason, decision_ts_utc
                        FROM s34_decisions
                        WHERE signal_ts_ms BETWEEN ? AND ?
                        ORDER BY signal_ts_ms DESC, decision_ts_utc DESC
                        LIMIT 16
                        """,
                        (
                            int(latest_signal_row["signal_ts_ms"]) - 5 * 60 * 1000,
                            int(latest_signal_row["signal_ts_ms"]) + 5 * 60 * 1000,
                        ),
                    ).fetchall()
                ]
            latest_outcomes = [
                dict(row)
                for row in con.execute(
                    """
                    SELECT trade_id, signal_id, rule_name, exit_ts_ms, exit_reason, gross_bps,
                           entry_adverse_bps, exit_adverse_bps, fee_cost_bps, net_bps
                    FROM s34_outcomes
                    ORDER BY exit_ts_ms DESC
                    LIMIT 8
                    """
                ).fetchall()
            ]
            latest_prediction_row = con.execute(
                """
                SELECT prediction_id, signal_id, model_name, model_version, predicted_at_utc, prediction_json
                FROM s34_predictions
                WHERE model_name='base_rate_v1'
                ORDER BY predicted_at_utc DESC
                LIMIT 1
                """
            ).fetchone()
            latest_prediction_rows = []
            if latest_signal_row:
                latest_prediction_rows = con.execute(
                    """
                    SELECT prediction_id, signal_id, model_name, model_version, predicted_at_utc, prediction_json
                    FROM s34_predictions
                    WHERE signal_id=?
                    ORDER BY model_name ASC
                    """,
                    (latest_signal_row["signal_id"],),
                ).fetchall()
            latest_audit_row = con.execute(
                """
                SELECT a.audit_id, a.signal_id, a.model_name, a.audit_ts_utc, a.audit_json
                FROM s34_model_audit a
                JOIN s34_signals s ON s.signal_id=a.signal_id
                ORDER BY s.signal_ts_ms DESC, a.audit_ts_utc DESC
                LIMIT 1
                """
            ).fetchone()
            latest_prediction = None
            latest_predictions = []
            latest_model_audit = None
            prediction_vs_outcome = None
            model_guardrail = {"level": "unknown", "headline": "No model prediction yet.", "reasons": []}
            for row in latest_prediction_rows:
                item = dict(row)
                try:
                    item["prediction"] = json.loads(str(item.get("prediction_json") or "{}"))
                except json.JSONDecodeError:
                    item["prediction"] = {}
                item.pop("prediction_json", None)
                latest_predictions.append(item)
            model_guardrail = _model_guardrail(latest_predictions)
            if latest_audit_row:
                latest_model_audit = dict(latest_audit_row)
                try:
                    latest_model_audit["audit"] = json.loads(str(latest_model_audit.get("audit_json") or "{}"))
                except json.JSONDecodeError:
                    latest_model_audit["audit"] = {}
                latest_model_audit.pop("audit_json", None)
            if latest_prediction_row:
                latest_prediction = dict(latest_prediction_row)
                try:
                    latest_prediction["prediction"] = json.loads(str(latest_prediction.get("prediction_json") or "{}"))
                except json.JSONDecodeError:
                    latest_prediction["prediction"] = {}
                latest_prediction.pop("prediction_json", None)
                outcome_row = con.execute(
                    """
                    SELECT trade_id, exit_reason, net_bps
                    FROM s34_outcomes
                    WHERE signal_id=?
                    ORDER BY exit_ts_ms DESC
                    LIMIT 1
                    """,
                    (latest_prediction["signal_id"],),
                ).fetchone()
                if outcome_row:
                    expected = latest_prediction.get("prediction", {}).get("expected_net_bps")
                    actual = float(outcome_row["net_bps"]) if outcome_row["net_bps"] is not None else None
                    prediction_vs_outcome = {
                        "trade_id": outcome_row["trade_id"],
                        "exit_reason": outcome_row["exit_reason"],
                        "expected_net_bps": expected,
                        "actual_net_bps": actual,
                        "error_bps": None if expected is None or actual is None else actual - float(expected),
                    }
            if prediction_vs_outcome is None:
                matched_row = con.execute(
                    """
                    SELECT o.trade_id, o.exit_reason, o.net_bps, p.prediction_json
                    FROM s34_outcomes o
                    JOIN s34_predictions p ON p.signal_id=o.signal_id
                    ORDER BY o.exit_ts_ms DESC
                    LIMIT 1
                    """
                ).fetchone()
                if matched_row:
                    try:
                        matched_prediction = json.loads(str(matched_row["prediction_json"] or "{}"))
                    except json.JSONDecodeError:
                        matched_prediction = {}
                    expected = matched_prediction.get("expected_net_bps")
                    actual = float(matched_row["net_bps"]) if matched_row["net_bps"] is not None else None
                    prediction_vs_outcome = {
                        "trade_id": matched_row["trade_id"],
                        "exit_reason": matched_row["exit_reason"],
                        "expected_net_bps": expected,
                        "actual_net_bps": actual,
                        "error_bps": None if expected is None or actual is None else actual - float(expected),
                    }
            reject_reasons = [
                dict(row)
                for row in con.execute(
                    """
                    SELECT reason, COUNT(*) AS count
                    FROM s34_rejected_signals
                    GROUP BY reason
                    ORDER BY count DESC, reason ASC
                    LIMIT 8
                    """
                ).fetchall()
            ]
            latest_signal = dict(latest_signal_row) if latest_signal_row else None
            base_rates = _base_rate_payload(con, latest_signal)
            calibration = _calibration_payload(con)
        finally:
            con.close()
        return {
            "available": True,
            "db_age_sec": round(time.time() - INTELLIGENCE_DB_PATH.stat().st_mtime, 1),
            "counts": counts,
            "latest_signal": latest_signal,
            "latest_decisions": latest_decisions,
            "cluster_decisions": cluster_decisions,
            "explanation": _build_intelligence_explanation(latest_signal, cluster_decisions),
            "base_rates": base_rates,
            "latest_prediction": latest_prediction,
            "latest_predictions": latest_predictions,
            "model_guardrail": model_guardrail,
            "latest_model_audit": latest_model_audit,
            "prediction_vs_outcome": prediction_vs_outcome,
            "calibration": calibration,
            "latest_outcomes": latest_outcomes,
            "reject_reasons": reject_reasons,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - chart should remain alive.
        payload = dict(empty)
        payload["error"] = repr(exc)
        return payload


def trade_payload(
    start_ms: int, end_ms: int, prices_by_symbol: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    payload = read_json(TRADES_PATH, {"trades": []})
    trades = payload.get("trades", []) if isinstance(payload, dict) else []
    signals = []
    open_trade = None
    valid = [t for t in trades if dashboard_valid_trade(t)]
    today = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).date().isoformat()
    today_valid = [t for t in valid if utc_iso_ms(t.get("signal_ts_ms"))[:10] == today]
    for t in trades:
        rule_id = str((t.get("rule") or {}).get("name") or "")
        if not dashboard_rule_visible(rule_id):
            continue
        signal_ts = int(t.get("signal_ts_ms") or 0)
        if signal_ts < start_ms or signal_ts > end_ms:
            continue
        if t.get("status") not in {"OPEN", "CLOSED"}:
            continue
        if str(t.get("symbol") or CHART_SYMBOL) != CHART_SYMBOL:
            continue
        signals.append(
            {
                "trade_id": t.get("trade_id"),
                "symbol": t.get("symbol") or CHART_SYMBOL,
                "rule_id": rule_id,
                "rule_label": rule_label(rule_id),
                "entry_ts": signal_ts,
                "entry_time": signal_ts,
                "entry_price": float(t.get("entry_price") or 0.0),
                "tp_price": float(t.get("tp_price") or 0.0),
                "sl_price": float(t.get("sl_price") or 0.0),
                "be_price": float(t.get("be_trigger_price") or 0.0),
                "be_trigger_price": float(t.get("be_trigger_price") or 0.0),
                "exit_ts": t.get("exit_ts_ms"),
                "exit_time": t.get("exit_ts_ms"),
                "exit_price": t.get("exit_price"),
                "exit_reason": t.get("exit_reason"),
                "exit_type": t.get("exit_reason"),
                "net_bps": t.get("net_bps"),
            }
        )
        if t.get("status") == "OPEN":
            open_trade = t
    levels = []
    for t in trades:
        if t.get("status") == "OPEN":
            if str(t.get("symbol") or CHART_SYMBOL) != CHART_SYMBOL:
                continue
            rule_id = str((t.get("rule") or {}).get("name") or "")
            if not dashboard_rule_visible(rule_id):
                continue
            label = f"{t.get('trade_id')} {rule_label(rule_id)}"
            levels.extend(
                [
                    {"label": f"{label} TP", "price": float(t.get("tp_price") or 0.0), "color": "#ffd166", "dash": False},
                    {"label": f"{label} SL", "price": float(t.get("sl_price") or 0.0), "color": "#ff4f5f", "dash": False},
                    {"label": f"{label} BE", "price": float(t.get("be_trigger_price") or 0.0), "color": "#ff9f1c", "dash": True},
                ]
            )
            open_trade = {
                "trade_id": t.get("trade_id"),
                "rule_id": rule_id,
                "rule_label": rule_label(rule_id),
                "entry_price": t.get("entry_price"),
                "tp_price": t.get("tp_price"),
                "sl_price": t.get("sl_price"),
                "be_price": t.get("be_trigger_price"),
                "be_trigger_price": t.get("be_trigger_price"),
                "unrealized_bps": unrealized_bps(t, prices_by_symbol.get(CHART_SYMBOL)),
            }
    valid.sort(key=lambda x: int(x.get("signal_ts_ms") or 0))
    last_signal = valid[-1] if valid else None
    status_mtime = STATUS_PATH.stat().st_mtime if STATUS_PATH.exists() else 0.0
    pid_alive = False
    pid = None
    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text(encoding="utf-8").strip())
            # os.kill(pid, 0) is unreliable on Windows permissions; mtime freshness is the primary signal.
            pid_alive = True
        except Exception:
            pid_alive = False
    runner_alive = (time.time() - status_mtime) <= 180 and pid_alive
    summary = {
        "today_valid_trades": len(today_valid),
        "closed_trades": len(valid),
        "net_bps_today": sum(float(t.get("net_bps") or 0.0) for t in today_valid),
        "cum_net_bps": sum(float(t.get("net_bps") or 0.0) for t in valid),
        "last_signal_utc": utc_iso_ms(last_signal.get("signal_ts_ms")) if last_signal else "",
        "last_signal_ts": int(last_signal.get("signal_ts_ms") or 0) if last_signal else None,
        "runner_alive": runner_alive,
        "runner_pid": pid,
        "status_age_sec": round(time.time() - status_mtime, 1) if status_mtime else None,
    }
    return signals, open_trade, {"levels": levels, "summary": summary}


def risk_sandbox_payload() -> dict[str, Any]:
    if not INTELLIGENCE_DB_PATH.exists():
        return {"available": False, "error": "s34_intelligence.db missing", "cards": []}
    try:
        payload = s34_prediction_risk_sandbox.build_payload(
            INTELLIGENCE_DB_PATH,
            RISK_SANDBOX_ACCOUNT_USDT,
            RISK_SANDBOX_LEVERAGES,
            8.0,
            RISK_SANDBOX_RISK_BUDGET_PCT,
            RISK_SANDBOX_MARGIN_USDT,
        )
        payload["available"] = True
        payload["cards"] = payload.get("cards", [])[:7]
        return payload
    except Exception as exc:  # noqa: BLE001 - dashboard should not fail if sandbox has an issue.
        return {"available": False, "error": repr(exc), "cards": []}


def live_execution_payload(trades: list[dict[str, Any]], prices_by_symbol: dict[str, float] | None = None) -> dict[str, Any]:
    env = read_env_file()
    allowed = [x.strip() for x in str(env.get("S34_LIVE_ALLOWED_RULES") or "").split(",") if x.strip()]
    allowed_set = set(allowed)
    state = read_json(LIVE_EXECUTOR_STATE_PATH, {})
    status = state.get("status") if isinstance(state, dict) else {}
    mirrored = state.get("mirrored_trade_ids") if isinstance(state, dict) else {}
    orders = state.get("orders") if isinstance(state, dict) else []
    reconciliation = state.get("reconciliation") if isinstance(state, dict) else {}
    state_mtime = LIVE_EXECUTOR_STATE_PATH.stat().st_mtime if LIVE_EXECUTOR_STATE_PATH.exists() else None
    state_age_sec = None if state_mtime is None else round(time.time() - state_mtime, 1)
    proc_rows = process_health()
    proc = next((row for row in proc_rows if row["name"] == "s34_state_machine_live_executor"), None)
    if proc is None:
        # fallback: check shared compat PID file (state machine writes both PID files)
        pid = read_pid_file(V_ENGINE_PID_PATH)
        proc = {"name": "s34_state_machine_live_executor", "pid": pid, "alive": pid_is_alive(pid)}
    if not proc.get("alive") and state_age_sec is not None and state_age_sec <= 30:
        proc = dict(proc)
        proc["alive"] = True
    candidates = []
    for trade in trades:
        rule_id = str((trade.get("rule") or {}).get("name") or trade.get("rule_name") or "")
        if trade.get("status") == "OPEN" and rule_id in allowed_set:
            candidates.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "rule_id": rule_id,
                    "rule_label": rule_label(rule_id),
                    "symbol": trade.get("symbol"),
                    "direction": trade.get("direction"),
                    "entry_price": trade.get("entry_price"),
                    "tp_price": trade.get("tp_price"),
                    "sl_price": trade.get("sl_price"),
                    "be_trigger_price": trade.get("be_trigger_price"),
                    "signal_ts_utc": trade.get("signal_ts_utc"),
                }
            )
    s34_live_enabled = env_truthy(env.get("S34_LIVE_TRADING_ENABLED"))
    s34_dry = env_truthy(env.get("S34_LIVE_DRY_RUN"))
    scalper_dry = env_truthy(env.get("SCALPER_DRY_RUN"))
    live_armed = bool(s34_live_enabled and not s34_dry and not scalper_dry)
    open_trade_ids = {str(t.get("trade_id")) for t in trades if t.get("status") == "OPEN"}
    last_orders = []
    open_positions = []
    active = state.get("active") if isinstance(state, dict) else None
    if isinstance(active, dict):
        open_positions.append(
            {
                "trade_id": active.get("event_id"),
                "symbol": active.get("symbol") or CHART_SYMBOL,
                "rule_id": active.get("rule") or (status or {}).get("rule"),
                "rule_label": rule_label(str(active.get("rule") or (status or {}).get("rule") or "")),
                "direction": str(active.get("direction") or "").upper() or "LONG",
                "mode": (status or {}).get("mode"),
                "amount": active.get("raw_amount") or active.get("position_amount"),
                "notional_usdt": active.get("notional_usdt"),
                "margin_usdt": active.get("margin_usdt"),
                "leverage": active.get("leverage"),
                "entry_price_ref": active.get("anchor_mark_price"),
                "tp_price": None,
                "sl_price": active.get("stop_price"),
                "be_trigger_price": None,
                "tp_bps": None,
                "sl_bps": env.get("S34_V_ENGINE_LIVE_STOP_BPS"),
                "be_bps": None,
                "created_at_utc": active.get("opened_at_utc"),
                "mirrored_at_utc": None,
                "current_price": (prices_by_symbol or {}).get(active.get("symbol") or CHART_SYMBOL),
                "is_open": True,
                "entry_order_id": active.get("replace_order_id") or active.get("initial_order_id"),
                "tp_order_id": None,
                "sl_order_id": active.get("stop_order_id"),
            }
        )
    if isinstance(orders, list):
        for item in orders[-30:]:
            if not isinstance(item, dict):
                continue
            plan = item.get("plan") if isinstance(item.get("plan"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if not plan:
                order_rec = {
                    "trade_id": item.get("event"),
                    "symbol": result.get("symbol") or CHART_SYMBOL,
                    "rule_id": (status or {}).get("rule"),
                    "rule_label": rule_label(str((status or {}).get("rule") or "")),
                    "direction": str(result.get("side") or active.get("direction") if isinstance(active, dict) else "LONG").upper(),
                    "mode": (status or {}).get("mode"),
                    "amount": result.get("amount"),
                    "notional_usdt": None,
                    "margin_usdt": None,
                    "leverage": None,
                    "entry_price_ref": result.get("price") or result.get("stopPrice"),
                    "tp_price": None,
                    "sl_price": result.get("stopPrice") if str(item.get("action") or "").endswith("stop") else None,
                    "be_trigger_price": None,
                    "tp_bps": None,
                    "sl_bps": env.get("S34_V_ENGINE_LIVE_STOP_BPS"),
                    "be_bps": None,
                    "created_at_utc": result.get("timestamp") or item.get("created_at_utc"),
                    "mirrored_at_utc": None,
                    "current_price": (prices_by_symbol or {}).get(result.get("symbol") or CHART_SYMBOL),
                    "is_open": bool(isinstance(active, dict) and str(active.get("event_id") or "") == str(item.get("event") or "")),
                    "entry_order_id": result.get("id") if "limit" in str(item.get("action") or "") else None,
                    "tp_order_id": None,
                    "sl_order_id": result.get("id") if "stop" in str(item.get("action") or "") else None,
                    "action": item.get("action"),
                }
                last_orders.append(order_rec)
                continue
            tid = str(plan.get("trade_id") or "")
            sym = str(plan.get("symbol") or "")
            cur_price = (prices_by_symbol or {}).get(sym)
            mir_info = (mirrored or {}).get(tid, {}) if isinstance(mirrored, dict) else {}
            entry_price_ref = plan.get("entry_price_ref")
            direction = str(plan.get("direction") or "").upper()
            is_open = tid in open_trade_ids
            entry_id = (result.get("entry") or {}).get("id") if isinstance(result.get("entry"), dict) else None
            tp_id = (result.get("tp") or {}).get("id") if isinstance(result.get("tp"), dict) else None
            sl_id = (result.get("sl") or {}).get("id") if isinstance(result.get("sl"), dict) else None
            order_rec = {
                "trade_id": plan.get("trade_id"),
                "symbol": sym,
                "rule_id": plan.get("rule"),
                "rule_label": rule_label(str(plan.get("rule") or "")),
                "direction": direction,
                "mode": mir_info.get("mode"),
                "amount": plan.get("amount"),
                "notional_usdt": plan.get("notional_usdt"),
                "margin_usdt": plan.get("margin_usdt"),
                "leverage": plan.get("leverage"),
                "entry_price_ref": entry_price_ref,
                "tp_price": plan.get("tp_price"),
                "sl_price": plan.get("sl_price"),
                "be_trigger_price": plan.get("be_trigger_price"),
                "tp_bps": plan.get("tp_bps"),
                "sl_bps": plan.get("sl_bps"),
                "be_bps": plan.get("be_bps"),
                "created_at_utc": plan.get("created_at_utc"),
                "mirrored_at_utc": mir_info.get("mirrored_at_utc"),
                "current_price": cur_price,
                "is_open": is_open,
                "entry_order_id": entry_id,
                "tp_order_id": tp_id,
                "sl_order_id": sl_id,
            }
            last_orders.append(order_rec)
            if is_open:
                open_positions.append(order_rec)
    last_orders = list(reversed(last_orders))
    active_direction = None
    if isinstance(active, dict):
        active_direction = str(active.get("direction") or "").upper() or None
    pending_count = len(state.get("pending") or {}) if isinstance(state, dict) and isinstance(state.get("pending"), dict) else int((status or {}).get("pending_count") or 0)
    blocked_by = str((status or {}).get("new_entry_blocked_by") or "")
    if active_direction in {"LONG", "SHORT"}:
        alpha_action = f"HOLD_{active_direction}"
        alpha_color = "pos" if active_direction == "LONG" else "warn"
        alpha_detail = "live position is open; executor manages time exit, stop and state transitions"
    elif pending_count > 0:
        alpha_action = "PENDING_STATE"
        alpha_color = "warn"
        alpha_detail = "anchor accepted; waiting for silence/follow-on/BTC confirmation window"
    elif live_armed and proc.get("alive") and blocked_by == "no_fresh_eligible_anchor":
        alpha_action = "READY_WAIT"
        alpha_color = "pos"
        alpha_detail = "armed and waiting for the next eligible ETH SELL anchor"
    elif live_armed and proc.get("alive"):
        alpha_action = "BLOCKED"
        alpha_color = "warn"
        alpha_detail = blocked_by or "waiting"
    else:
        alpha_action = "NOT_READY"
        alpha_color = "neg"
        alpha_detail = "executor is not both alive and armed"
    now_ms = int(time.time() * 1000)
    # active position detail for live monitor
    active_position: dict[str, Any] | None = None
    if isinstance(active, dict):
        cur_px = (prices_by_symbol or {}).get(active.get("symbol") or CHART_SYMBOL)
        entry_px = active.get("entry_ref_price") or active.get("anchor_mark_price")
        direction_up = str(active.get("direction") or "LONG").upper()
        unrealized_bps: float | None = None
        if cur_px and entry_px and float(entry_px) > 0:
            raw = (float(cur_px) - float(entry_px)) / float(entry_px) * 10_000.0
            unrealized_bps = round(raw if direction_up == "LONG" else -raw, 1)
        exit_due = int(active.get("exit_due_ts_ms") or 0)
        time_left_sec = max(0, (exit_due - now_ms) / 1000) if exit_due else None
        active_position = {
            "direction": direction_up,
            "event_id": active.get("event_id"),
            "entry_price": entry_px,
            "current_price": cur_px,
            "unrealized_bps": unrealized_bps,
            "exit_due_ts_ms": exit_due,
            "time_left_sec": round(time_left_sec) if time_left_sec is not None else None,
            "stop_bps": active.get("stop_bps"),
            "stop_order_id": active.get("stop_order_id"),
            "opened_at_utc": active.get("opened_at_utc"),
            "notional_usdt": active.get("notional_usdt"),
            "margin_usdt": active.get("margin_usdt"),
            "base_score": active.get("base_score"),
            "score_if_silence": active.get("score_if_silence"),
            "session": active.get("session"),
            "vdepth_bps": active.get("vdepth_bps"),
            "n2h": active.get("n2h"),
            "btc4h_bps": active.get("btc4h_bps"),
            "sync_k": active.get("sync_k"),
            "anchor_ts_ms": active.get("anchor_ts_ms"),
            "running_notional": active.get("running_notional"),
            "state_resolution": active.get("state_resolution"),
            "silence_confirmed_at_utc": active.get("silence_confirmed_at_utc"),
        }
    # pending events detail
    pending_dict = state.get("pending") if isinstance(state, dict) else {}
    pending_events: list[dict[str, Any]] = []
    if isinstance(pending_dict, dict):
        for eid, pev in list(pending_dict.items())[:5]:
            if not isinstance(pev, dict):
                continue
            ts_anchor = int(pev.get("anchor_ts_ms") or 0)
            elapsed_sec = round((now_ms - ts_anchor) / 1000) if ts_anchor else None
            expires = int(pev.get("expires_ts_ms") or 0)
            pending_events.append({
                "event_id": eid,
                "status": pev.get("status"),
                "long_opened": pev.get("long_opened"),
                "short_opened": pev.get("short_opened"),
                "anchor_ts_ms": ts_anchor,
                "elapsed_sec": elapsed_sec,
                "expires_ts_ms": expires,
                "base_score": pev.get("base_score"),
                "score_if_silence": pev.get("score_if_silence"),
                "long_eligible": pev.get("long_eligible"),
                "short_eligible": pev.get("short_eligible"),
                "session": pev.get("session"),
                "vdepth_bps": pev.get("vdepth_bps"),
                "n2h": pev.get("n2h"),
                "sync_k": pev.get("sync_k"),
                "btc4h_bps": pev.get("btc4h_bps"),
                "running_notional": pev.get("running_notional"),
            })
    return {
        "available": True,
        "process": proc or {"name": "s34_state_machine_live_executor", "pid": None, "alive": False},
        "alpha_decision": {
            "rule": (status or {}).get("rule") or (allowed[0] if allowed else ""),
            "engine": (status or {}).get("engine"),
            "action": alpha_action,
            "color": alpha_color,
            "detail": alpha_detail,
            "blocked_by": blocked_by,
            "pending_count": pending_count,
            "active_direction": active_direction,
            "last_signal_scan": (status or {}).get("last_signal_scan") if isinstance(status, dict) else {},
            "reconciliation": reconciliation if isinstance(reconciliation, dict) else {},
        },
        "env": {
            "live_enabled": s34_live_enabled,
            "s34_dry_run": s34_dry,
            "scalper_dry_run": scalper_dry,
            "live_armed": live_armed,
            "margin_usdt": env.get("S34_LIVE_MARGIN_USDT"),
            "max_leverage": env.get("S34_LIVE_MAX_LEVERAGE"),
            "max_open_positions": env.get("S34_LIVE_MAX_OPEN_POSITIONS"),
        },
        "state": {
            "exists": LIVE_EXECUTOR_STATE_PATH.exists(),
            "age_sec": state_age_sec,
            "status": status if isinstance(status, dict) else {},
            "mirrored_count": len(mirrored) if isinstance(mirrored, dict) else 0,
            "orders_count": len(orders) if isinstance(orders, list) else 0,
        },
        "allowed_rules": [{"rule_id": rule, "label": rule_label(rule)} for rule in allowed],
        "open_candidates": candidates,
        "open_positions": open_positions,
        "last_orders": last_orders,
        "active_position": active_position,
        "pending_events": pending_events,
    }


def v02_entry_quality_series(start_ms: int | None = None, end_ms: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(V_ENGINE_V02_MIRROR_LEDGER_PATH, limit=2000)
    points: list[dict[str, Any]] = []
    ema: float | None = None
    alpha = 0.35
    for row in sorted(rows, key=lambda r: int(r.get("maker_fill_ts_ms") or r.get("signal_ts_ms") or 0)):
        if row.get("sim_status") != "FILLED" or row.get("maker_fill_ts_ms") is None:
            continue
        ts = int(row.get("maker_fill_ts_ms") or 0)
        if start_ms is not None and ts < int(start_ms):
            continue
        if end_ms is not None and ts > int(end_ms):
            continue
        score = num_or_none(row.get("retest_quality_score"))
        if score is None:
            continue
        ema = score if ema is None else (alpha * score + (1.0 - alpha) * ema)
        points.append(
            {
                "ts": ts,
                "utc": utc_iso_ms(ts),
                "score": round(float(score), 3),
                "ema_score": round(float(ema), 3),
                "bucket": row.get("retest_quality_bucket"),
                "depth": row.get("retest_depth_bucket"),
                "delay": row.get("fill_minus_arm_bucket"),
                "tags": row.get("entry_quality_tags"),
                "warnings": row.get("entry_quality_warnings"),
                "net_bps": num_or_none(row.get("net_bps")),
                "fill_leg": row.get("fill_leg"),
            }
        )
    latest = points[-1] if points else None
    return {
        "available": bool(points),
        "series": points,
        "latest": latest,
        "ema_alpha": alpha,
    }


def v_engine_v02_shadow_mirror_payload(start_ms: int | None = None, end_ms: int | None = None) -> dict[str, Any]:
    brief = read_json(V_ENGINE_V02_MIRROR_BRIEF_PATH, {})
    state = read_json(V_ENGINE_V02_MIRROR_STATE_PATH, {})
    pid = read_pid_file(V_ENGINE_V02_MIRROR_PID_PATH)
    alive = pid_is_alive(pid)
    brief_mtime = V_ENGINE_V02_MIRROR_BRIEF_PATH.stat().st_mtime if V_ENGINE_V02_MIRROR_BRIEF_PATH.exists() else None
    state_mtime = V_ENGINE_V02_MIRROR_STATE_PATH.stat().st_mtime if V_ENGINE_V02_MIRROR_STATE_PATH.exists() else None
    brief_age_sec = None if brief_mtime is None else round(time.time() - brief_mtime, 1)
    state_age_sec = None if state_mtime is None else round(time.time() - state_mtime, 1)
    if not isinstance(brief, dict) or not brief:
        return {
            "available": False,
            "error": "v0.2 shadow mirror brief missing",
            "process": {"name": "s34_v_engine_v02_shadow_mirror", "pid": pid, "alive": alive},
            "brief_age_sec": brief_age_sec,
            "state_age_sec": state_age_sec,
        }
    protocol = brief.get("protocol") if isinstance(brief.get("protocol"), dict) else {}
    ledger = brief.get("ledger") if isinstance(brief.get("ledger"), dict) else {}
    overall = brief.get("overall") if isinstance(brief.get("overall"), dict) else {}
    recent = brief.get("recent") if isinstance(brief.get("recent"), dict) else {}
    latest = brief.get("latest_observations") if isinstance(brief.get("latest_observations"), list) else []
    return {
        "available": True,
        "generated_at_utc": brief.get("generated_at_utc"),
        "brief_age_sec": brief_age_sec,
        "state_age_sec": state_age_sec,
        "process": {"name": "s34_v_engine_v02_shadow_mirror", "pid": pid, "alive": alive},
        "state": state if isinstance(state, dict) else {},
        "protocol": protocol,
        "config": brief.get("config") if isinstance(brief.get("config"), dict) else {},
        "ledger": ledger,
        "overall": overall,
        "recent": recent,
        "latest_observations": latest[-5:],
        "entry_quality": v02_entry_quality_series(start_ms, end_ms),
    }


def v02_h4_shadow_payload(start_ms: int | None = None, end_ms: int | None = None) -> dict[str, Any]:
    data = read_json(V_ENGINE_V02_H4_SHADOW_PATH, {})
    mtime = V_ENGINE_V02_H4_SHADOW_PATH.stat().st_mtime if V_ENGINE_V02_H4_SHADOW_PATH.exists() else None
    age_sec = None if mtime is None else round(time.time() - mtime, 1)
    if not isinstance(data, dict) or not data:
        return {
            "available": False,
            "error": "v0.2 H4 shadow report missing",
            "age_sec": age_sec,
        }
    ledger_rows = read_jsonl(V_ENGINE_V02_H4_SHADOW_LEDGER_PATH, limit=4000)
    series = []
    for row in ledger_rows:
        if row.get("bucket") != "H4_SHADOW":
            continue
        ts = int(row.get("maker_fill_ts_ms") or 0)
        if ts <= 0:
            continue
        if start_ms is not None and ts < int(start_ms):
            continue
        if end_ms is not None and ts > int(end_ms):
            continue
        entry = num_or_none(row.get("entry_price"))
        if entry is None:
            continue
        series.append(
            {
                "ts": ts,
                "utc": utc_iso_ms(ts),
                "entry_price": entry,
                "net_bps": num_or_none(row.get("net_bps")),
                "h2_net_bps": num_or_none(row.get("h2_net_bps")),
                "h4_net_bps": num_or_none(row.get("h4_net_bps")),
                "state_path_v2": row.get("state_path_v2"),
                "cross_no_dump": bool(row.get("cross_no_dump") in (True, "True", "true", "1", 1)),
            }
        )
    return {
        "available": True,
        "generated_at_utc": data.get("generated_at_utc"),
        "age_sec": age_sec,
        "scope": data.get("scope") if isinstance(data.get("scope"), dict) else {},
        "buckets": data.get("buckets") if isinstance(data.get("buckets"), dict) else {},
        "cross_no_dump_observer": data.get("cross_no_dump_observer") if isinstance(data.get("cross_no_dump_observer"), dict) else {},
        "catastrophic_stop_observer": data.get("catastrophic_stop_observer") if isinstance(data.get("catastrophic_stop_observer"), dict) else {},
        "queue_fill_realism": data.get("queue_fill_realism") if isinstance(data.get("queue_fill_realism"), dict) else {},
        "state_machine_v2": data.get("state_machine_v2") if isinstance(data.get("state_machine_v2"), dict) else {},
        "latest_rows": data.get("latest_rows") if isinstance(data.get("latest_rows"), list) else [],
        "series": series,
        "decision": "H4_SHADOW_OBSERVATION_ONLY",
    }


def sizing_shadow_payload() -> dict[str, Any]:
    data = read_json(V_ENGINE_SIZING_SHADOW_PATH, {})
    mtime = V_ENGINE_SIZING_SHADOW_PATH.stat().st_mtime if V_ENGINE_SIZING_SHADOW_PATH.exists() else None
    age_sec = None if mtime is None else round(time.time() - mtime, 1)
    if not isinstance(data, dict) or not data:
        return {"available": False, "error": "sizing shadow report missing", "age_sec": age_sec}
    return {
        "available": True,
        "generated_at_utc": data.get("generated_at_utc"),
        "age_sec": age_sec,
        "status": data.get("status"),
        "rule_id": data.get("rule_id"),
        "source_shadow_rows": data.get("source_shadow_rows"),
        "equity_assumption_usdt": data.get("equity_assumption_usdt"),
        "modes": data.get("modes") if isinstance(data.get("modes"), dict) else {},
        "read": data.get("read"),
    }


def state_machine_shadow_payload(process_rows: list[dict[str, Any]]) -> dict[str, Any]:
    state = read_json(STATE_MACHINE_SHADOW_STATE_PATH, {})
    rows = read_jsonl(STATE_MACHINE_SHADOW_LEDGER_PATH, limit=5000)
    pid = read_pid_file(STATE_MACHINE_SHADOW_PID_PATH)
    proc = next((row for row in process_rows if row.get("name") == "s34_state_machine_shadow_runner"), {})
    alive = bool(proc.get("alive")) or pid_is_alive(pid)
    state_mtime = STATE_MACHINE_SHADOW_STATE_PATH.stat().st_mtime if STATE_MACHINE_SHADOW_STATE_PATH.exists() else None
    ledger_mtime = STATE_MACHINE_SHADOW_LEDGER_PATH.stat().st_mtime if STATE_MACHINE_SHADOW_LEDGER_PATH.exists() else None
    by_sig: dict[str, list[float]] = {}
    by_sig_session: dict[str, dict[str, list[float]]] = {}
    by_sig_month: dict[str, dict[str, list[float]]] = {}
    by_sig_filt: dict[str, list[float]] = {}
    by_sig_filt_session: dict[str, dict[str, list[float]]] = {}
    by_sig_filt_month: dict[str, dict[str, list[float]]] = {}
    closed = 0
    for row in rows:
        if row.get("event") != "CLOSE":
            continue
        closed += 1
        net = num_or_none(row.get("net_bps"))
        sig = str(row.get("signal") or "?")
        sess = str(row.get("session") or "?")
        ts_ms = row.get("anchor_ts_ms") or row.get("entry_ts_ms") or 0
        try:
            dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
            month_key = dt.strftime("%Y-%m")
        except Exception:
            month_key = "?"
        if net is not None:
            by_sig.setdefault(sig, []).append(float(net))
            by_sig_session.setdefault(sig, {}).setdefault(sess, []).append(float(net))
            by_sig_month.setdefault(sig, {}).setdefault(month_key, []).append(float(net))
            # New-gate-filtered: LONG=TIME_EXIT+sync<200K, SHORT=score>=4
            sync_k_val = float(row.get("sync_k") or 0)
            score_val = float(row.get("score") or 0)
            close_reason = str(row.get("close_reason") or "")
            gate_pass = (
                (sig in {"LONG_T15_BOUNCE", "LONG_SILENCE"} and close_reason == "TIME_EXIT" and sync_k_val < 200_000) or
                (sig == "SHORT_NEITHER" and score_val >= 4)
            )
            if gate_pass:
                by_sig_filt.setdefault(sig, []).append(float(net))
                by_sig_filt_session.setdefault(sig, {}).setdefault(sess, []).append(float(net))
                by_sig_filt_month.setdefault(sig, {}).setdefault(month_key, []).append(float(net))

    def _stat_block(vals: list[float]) -> dict[str, Any]:
        wins = sum(1 for v in vals if v > 0)
        return {
            "n": len(vals),
            "wr": round(wins / len(vals), 3) if vals else None,
            "sum_bps": round(sum(vals), 1),
            "avg_bps": round(sum(vals) / len(vals), 1) if vals else None,
        }

    signal_stats = {}
    for sig, vals in by_sig.items():
        stat = _stat_block(vals)
        session_breakdown = {
            s: _stat_block(vs)
            for s, vs in sorted((by_sig_session.get(sig) or {}).items())
        }
        month_breakdown = {
            m: _stat_block(vs)
            for m, vs in sorted((by_sig_month.get(sig) or {}).items())
        }
        signal_stats[sig] = {**stat, "by_session": session_breakdown, "by_month": month_breakdown}
    # Candidate research buckets — LONG side only, tracked from ledger
    rows_closed_all = [r for r in rows if r.get("event") == "CLOSE"]

    def _cand_stat(gate_fn: Any) -> dict[str, Any]:
        vals: list[float] = []
        last_utc: str | None = None
        for r in rows_closed_all:
            if not gate_fn(r):
                continue
            net = num_or_none(r.get("net_bps"))
            if net is not None:
                vals.append(float(net))
                ts = str(r.get("closed_utc") or r.get("opened_utc") or "")
                if ts:
                    last_utc = ts
        wins = sum(1 for v in vals if v > 0)
        return {
            "n": len(vals),
            "wr": round(wins / len(vals), 3) if vals else None,
            "avg_bps": round(sum(vals) / len(vals), 1) if vals else None,
            "sum_bps": round(sum(vals), 1) if vals else None,
            "last_utc": last_utc,
        }

    def _long_base(r: dict[str, Any]) -> bool:
        return (
            str(r.get("signal") or "") in {"LONG_T15_BOUNCE", "LONG_SILENCE"}
            and str(r.get("close_reason") or "") == "TIME_EXIT"
            and float(r.get("sync_k") or 0) < 200_000
        )

    candidate_buckets = [
        {
            "name": "LIVE_long_t15_bounce",
            "label": "LIVE LONG T+15 bounce confirm - anchor+4h exit",
            "readiness": "LIVE_FINAL",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_T15_BOUNCE"),
        },
        {
            "name": "C_score_relax",
            "label": "LONG score≥2 · sync<200K (btc7d gate removed)",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.018,
            "stats": _cand_stat(lambda r: _long_base(r) and float(r.get("score") or 0) >= 2),
        },
        {
            "name": "C_btc7d500",
            "label": "LONG btc7d∈(-500,0) · sync<200K [live only]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.240,
            "stats": _cand_stat(lambda r: _long_base(r) and r.get("btc7d_bps") is not None and float(r.get("btc7d_bps")) < 500 and float(r.get("btc7d_bps")) > -500),
        },
        {
            "name": "C_freq_btc4h",
            "label": "LONG btc4h<0 · sync<200K · no btc7d [live only]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.375,
            "stats": _cand_stat(lambda r: _long_base(r) and r.get("btc4h_bps") is not None and float(r.get("btc4h_bps")) < 0),
        },
        {
            "name": "C_double_cascade",
            "label": "LONG DOUBLE_CASCADE density_24h>=1 + prebuildup>=2 [live only]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.001,
            "stats": _cand_stat(lambda r: _long_base(r) and r.get("density_24h") is not None and r.get("prebuildup") is not None and int(r.get("density_24h") or 0) >= 1 and int(r.get("prebuildup") or 0) >= 2),
        },
        {
            "name": "C_btc_falling_fast",
            "label": "LONG BTC_FALLING_FAST btc5m<-20bps [live only]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.001,
            "stats": _cand_stat(lambda r: _long_base(r) and r.get("btc5m_bps") is not None and float(r.get("btc5m_bps")) < -20),
        },
        {
            "name": "C_failed_cascade",
            "label": "LONG FAILED_CASCADE price UP at anchor+5m [live only]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.006,
            "stats": _cand_stat(lambda r: _long_base(r) and r.get("failed_cascade") is True),
        },
        {
            "name": "C_echo_45_120_silence",
            "label": "LONG ECHO 45-120m + silence [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_ECHO_45_120_SILENCE"),
        },
        {
            "name": "C_hour17_hold6h",
            "label": "LONG hour>=17 UTC + regime · hold 6h no early-exit [LIVE + shadow]",
            "readiness": "LIVE_FORWARD_VALIDATION",
            "mc_p": 0.003,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_HOLD6H"),
        },
        {
            "name": "C_hour17_composite",
            "label": "LONG hour17 + conviction>=3/8 (sync/rv/density/ofi/be/ask/shelf/whale) +funding-veto [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_COMPOSITE"),
        },
        {
            "name": "C_hour17_composite_s4",
            "label": "LONG hour17 conviction>=4/8 (high-WR sleeve, OOS ~85%) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_COMPOSITE" and int(r.get("conviction_score") or 0) >= 4),
        },
        {
            "name": "C_hour17_100k_composite",
            "label": "LONG 100K mini + hour17 + conviction>=3 (freq-expansion, OOS noov 11.8/mo) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_100K_COMPOSITE"),
        },
        {
            "name": "C_hour17_comp_notmon",
            "label": "LONG hour17 composite · NOT Monday (2 evren OOS: Mon WR 18-37%) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_COMPOSITE"
                                and r.get("entry_ts_ms")
                                and datetime.fromtimestamp(int(r["entry_ts_ms"]) / 1000, tz=timezone.utc).weekday() != 0),
        },
        {
            "name": "C_hour17_triple_rsw",
            "label": "LONG hour17 rv+shelf+whale triple interaction (RA 59.6, proxy-rv TEST WR100) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_COMPOSITE"
                                and (r.get("conviction") or {}).get("rv_hit") is True
                                and (r.get("conviction") or {}).get("shelf_hit") is True
                                and (r.get("conviction") or {}).get("whale_hit") is True),
        },
        {
            "name": "C_hour17_score10",
            "label": "LONG hour17 score10>=6 (composite + tsl<115m + two_sided BUY-liq) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_HOUR17_COMPOSITE"
                                and (r.get("conviction") or {}).get("score10") is not None
                                and int((r.get("conviction") or {}).get("score10") or 0) >= 6),
        },
        {
            "name": "C_hour17_scalein100",
            "label": "LONG hour17 SCALE-IN@-100 dip subset (M7 observer: perU 86.3, mdd iyilesir) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") in ("LONG_HOUR17_HOLD6H", "LONG_HOUR17_COMPOSITE", "LONG_HOUR17_100K_COMPOSITE")
                                and (r.get("scalein_observer") or {}).get("added") is True),
        },
        {
            "name": "C_short_noisy_all",
            "label": "SHORT_NOISY base - first ETH SELL prop >=50K in 1-30min window, 2h hold [shadow route, live sizing]",
            "readiness": "SHADOW_ONLY_TRACKING",
            "mc_p": None,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "SHORT_NOISY"),
        },
        {
            "name": "C_short_noisy_1317",
            "label": "SHORT_NOISY BTC1M entry 13-17 UTC (time-machine sleeve, OOS WR82) [shadow]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.006,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "SHORT_NOISY_BTC1M_D5_H180"
                                and r.get("entry_ts_ms")
                                and 13 <= datetime.fromtimestamp(int(r["entry_ts_ms"]) / 1000, tz=timezone.utc).hour < 17),
        },
        {
            "name": "C_short_btc1m_h4",
            "label": "SHORT BTC>=1M delay5 · 4h hold [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.001,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "SHORT_BTC1M_H4"),
        },
        {
            "name": "C_short_btc1m_d10_h3",
            "label": "SHORT score4 BTC>=1M delay10 - 3h hold [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.001,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "SHORT_BTC1M_D10_H3"),
        },
        {
            "name": "C_short_noisy_btc1m_d5_h180",
            "label": "SHORT NOISY + BTC>=1M delay5 - 180m hold [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.003,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "SHORT_NOISY_BTC1M_D5_H180"),
        },
        {
            "name": "C_double_cascade_prebuild2",
            "label": "LONG prebuild>=2 / DOUBLE_CASCADE + silence [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.0,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE"),
        },
        {
            "name": "C_ofi_silence_buyers",
            "label": "LONG OFI silence + buyers [shadow route]",
            "readiness": "PAPER_CANDIDATE",
            "mc_p": 0.001,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "LONG_OFI_SILENCE_BUYERS"),
        },
        {
            "name": "C_buy_fade_h45_sl75_all",
            "label": "BUY-side fade SHORT T0 - 45m hold / 75bps SL [shadow route]",
            "readiness": "SHADOW_ONLY_TAIL_REPAIR",
            "mc_p": 0.01,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "BUY_FADE_SHORT_H45_SL75"),
        },
        {
            "name": "C_buy_fade_h45_sl75_silence",
            "label": "BUY-side fade SHORT T0 - silence subset [shadow route]",
            "readiness": "SHADOW_ONLY_TAIL_REPAIR",
            "mc_p": 0.01,
            "stats": _cand_stat(lambda r: str(r.get("signal") or "") == "BUY_FADE_SHORT_H45_SL75" and str(r.get("buy_state") or "") == "SILENCE"),
        },
    ]

    signal_stats_filtered = {}
    for sig, vals in by_sig_filt.items():
        stat = _stat_block(vals)
        session_breakdown = {
            s: _stat_block(vs)
            for s, vs in sorted((by_sig_filt_session.get(sig) or {}).items())
        }
        month_breakdown = {
            m: _stat_block(vs)
            for m, vs in sorted((by_sig_filt_month.get(sig) or {}).items())
        }
        signal_stats_filtered[sig] = {**stat, "by_session": session_breakdown, "by_month": month_breakdown}
    positions = state.get("positions") if isinstance(state, dict) else {}
    open_positions = []
    if isinstance(positions, dict):
        for pid_key, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            if str(pos.get("status") or "").startswith("CLOSED") or pos.get("status") in {"EXPIRED_NO_BTC", "EXPIRED_SILENCE"}:
                continue
            observer = pos.get("profit_lock_observer") if isinstance(pos.get("profit_lock_observer"), dict) else {}
            open_positions.append({
                "id": pid_key,
                "signal": pos.get("signal"),
                "direction": pos.get("direction"),
                "status": pos.get("status"),
                "score": pos.get("score"),
                "sync_k": pos.get("sync_k"),
                "hour": pos.get("hour"),
                "btc4h_bps": pos.get("btc4h_bps"),
                "btc7d_bps": pos.get("btc7d_bps"),
                "entry_price": pos.get("entry_price"),
                "running_notional": pos.get("running_notional"),
                "anchor_ts_ms": pos.get("anchor_ts_ms"),
                "entry_ts_ms": pos.get("entry_ts_ms"),
                "exit_due_ms": pos.get("exit_due_ms"),
                "sil_check_ms": pos.get("sil_check_ms"),
                "opened_utc": pos.get("opened_utc"),
                "sl_bps": pos.get("sl_bps"),
                "session": pos.get("session"),
                "buy_state": pos.get("buy_state"),
                "n2h": pos.get("n2h"),
                "observer_pnl_bps": observer.get("last_seen_pnl_bps"),
                "observer_triggered": observer.get("triggered"),
            })
    # Active LIVE alpha spotlight: hour17 hold-6h predictor route (LONG only).
    _h17_sig = "LONG_HOUR17_HOLD6H"
    _shadow_stats = signal_stats.get(_h17_sig, {"n": 0, "wr": None, "sum_bps": None, "avg_bps": None})
    # LIVE side: read the live executor state (real orders), separate from shadow paper.
    _live_state = read_json(LIVE_EXECUTOR_STATE_PATH, {})
    _live_orders = _live_state.get("orders") if isinstance(_live_state, dict) else []
    _live_orders = _live_orders if isinstance(_live_orders, list) else []
    _h17_opens = [o for o in _live_orders if isinstance(o, dict) and o.get("action") == "open_long_hour17_hold6h"]
    _live_active = _live_state.get("active") if isinstance(_live_state, dict) else None
    _live_active_h17 = bool(
        isinstance(_live_active, dict)
        and _live_active.get("status") == "POSITION_OPEN"
        and (_live_active.get("reason") == _h17_sig or _live_active.get("entry_trigger") == "HOUR17_T0")
    )
    live_summary = {
        "orders_opened": len(_h17_opens),
        "active": _live_active_h17,
        "active_direction": (_live_active or {}).get("direction") if _live_active_h17 else None,
        "note": "live closed-PnL: live_execution / orders panelinde; buradaki WR/total SHADOW (paper) mirror'idir",
    }
    active_alpha = {
        "name": _h17_sig,
        "label": "hour>=17 UTC + regime · hold 6h (no early-exit, 300bps safety stop) · LONG only",
        "definition": "ETH SELL cascade >=200K, not bull, not EUROPE, regime(btc4h<0 OR btc7d<0), hour>=17 UTC -> LONG at T0, hold 6h, no early-exit. (LONG only; SHORT ayrı route.)",
        "status": "LIVE armed + shadow paper (forward validation)",
        "direction": "LONG",
        "research": {"per_month": 16.2, "wr": 0.60, "oos_mc_p": 0.003, "wf": "5/5", "mdd_bps": -391},
        "shadow_stats": _shadow_stats,
        "live_summary": live_summary,
        "open_positions": [p for p in open_positions if p.get("signal") == _h17_sig],
    }
    return {
        "available": STATE_MACHINE_SHADOW_STATE_PATH.exists() or STATE_MACHINE_SHADOW_LEDGER_PATH.exists(),
        "process": {"name": "s34_state_machine_shadow_runner", "pid": proc.get("pid") or pid, "alive": alive},
        "active_alpha": active_alpha,
        "rule_name": state.get("rule_name") if isinstance(state, dict) else None,
        "mode": state.get("mode") if isinstance(state, dict) else "OBSERVE_ONLY_NO_ORDER",
        "profit_lock_observer": state.get("profit_lock_observer") if isinstance(state, dict) else None,
        "state_pnl": state.get("pnl") if isinstance(state, dict) else None,
        "state_updated_utc": state.get("updated_utc") if isinstance(state, dict) else None,
        "state_age_sec": None if state_mtime is None else round(time.time() - state_mtime, 1),
        "ledger_age_sec": None if ledger_mtime is None else round(time.time() - ledger_mtime, 1),
        "rows_total": len(rows),
        "closed_trades": closed,
        "open_trades": len(open_positions),
        "open_positions": open_positions[:5],
        "recent_events": rows[-12:],
        "recent_closed": sorted([r for r in rows if r.get("event") == "CLOSE"], key=lambda r: int(r.get("anchor_ts_ms") or r.get("entry_ts_ms") or 0))[-100:],
        "signal_stats": signal_stats,
        "signal_stats_filtered": signal_stats_filtered,
        "candidate_buckets": candidate_buckets,
    }


def shadow_paper_buckets_payload(
    *,
    mirror: dict[str, Any],
    h4_shadow: dict[str, Any],
    sizing: dict[str, Any],
    process_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    status = read_json(STATUS_PATH, {})
    runner_proc = next((row for row in process_rows if row.get("name") == "s34_shadow_paper_runner"), {})
    state_machine = state_machine_shadow_payload(process_rows)
    return {
        "available": True,
        "mode": "OBSERVATION_ONLY_NO_ORDER",
        "paper_runner": {
            "alive": bool(runner_proc.get("alive")),
            "pid": runner_proc.get("pid"),
            "updated_at_utc": status.get("updated_at_utc") if isinstance(status, dict) else None,
            "total_trades": status.get("total_trades") if isinstance(status, dict) else None,
            "open_trades": status.get("open_trades") if isinstance(status, dict) else None,
            "closed_trades": status.get("closed_trades") if isinstance(status, dict) else None,
            "risk_skipped_trades": status.get("risk_skipped_trades") if isinstance(status, dict) else None,
        },
        "v02_mirror": mirror,
        "state_machine_shadow": state_machine,
        "h4_shadow": h4_shadow,
        "sizing_shadow": sizing,
    }


def bucket_independence_payload() -> dict[str, Any]:
    data = read_json(BUCKET_INDEPENDENCE_AUDIT_PATH, {})
    if not isinstance(data, dict) or not data:
        return {
            "available": False,
            "error": "bucket independence audit missing",
            "generated_at_utc": None,
            "counts": {},
            "routes": {},
            "pairs": [],
        }

    pairs = data.get("pairs") if isinstance(data.get("pairs"), list) else []
    route_stats = data.get("routes") if isinstance(data.get("routes"), dict) else {}
    counts: dict[str, int] = {}
    route_info: dict[str, dict[str, Any]] = {}
    same_edges: list[tuple[str, str]] = []

    for route_id in route_stats:
        route_info[str(route_id)] = {
            "family_label": "Likely independent",
            "family_type": "LIKELY_INDEPENDENT",
            "same_family_routes": [],
            "related_routes": [],
            "pair_counts": {},
        }

    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        verdict = str(pair.get("verdict") or "UNKNOWN")
        counts[verdict] = counts.get(verdict, 0) + 1
        a = str(pair.get("route_a") or "")
        b = str(pair.get("route_b") or "")
        if not a or not b:
            continue
        for route_id, other in ((a, b), (b, a)):
            info = route_info.setdefault(
                route_id,
                {
                    "family_label": "Likely independent",
                    "family_type": "LIKELY_INDEPENDENT",
                    "same_family_routes": [],
                    "related_routes": [],
                    "pair_counts": {},
                },
            )
            info["pair_counts"][verdict] = int(info["pair_counts"].get(verdict, 0)) + 1
            if verdict == "SAME_FAMILY":
                info["same_family_routes"].append(other)
            elif verdict == "RELATED":
                info["related_routes"].append(other)
        if verdict == "SAME_FAMILY":
            same_edges.append((a, b))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for route_id in route_info:
        find(route_id)
    for a, b in same_edges:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for route_id in route_info:
        groups.setdefault(find(route_id), []).append(route_id)

    family_index = 1
    for members in groups.values():
        members = sorted(set(members))
        if len(members) <= 1:
            route_id = members[0]
            info = route_info[route_id]
            if info["related_routes"]:
                info["family_label"] = "Related cascade"
                info["family_type"] = "RELATED"
            continue
        label = f"Same alpha family {family_index}"
        family_index += 1
        for route_id in members:
            info = route_info[route_id]
            info["family_label"] = label
            info["family_type"] = "SAME_FAMILY"
            info["same_family_routes"] = sorted(set(x for x in members if x != route_id))

    for info in route_info.values():
        info["same_family_routes"] = sorted(set(info.get("same_family_routes") or []))
        info["related_routes"] = sorted(set(info.get("related_routes") or []))

    return {
        "available": True,
        "generated_at_utc": data.get("generated_at_utc"),
        "closed_clean_n": data.get("closed_clean_n"),
        "counts": counts,
        "routes": route_info,
        "pairs": pairs,
    }


_MODEL_USEFULNESS: dict[tuple[str, str], str] = {
    ("ETHUSDT", "SELL"): "KNN_USEFUL",
    ("BTCUSDT", "SELL"): "BASE_RATE_ONLY",
    ("ETHUSDT", "BUY"):  "REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY",
    ("SOLUSDT", "BUY"):  "BASE_RATE_ONLY",
    ("SOLUSDT", "SELL"): "BASE_RATE_ONLY",
    ("BTCUSDT", "BUY"):  "BASE_RATE_ONLY",
}


def similarity_payload() -> dict[str, Any]:
    if not CALCULATOR_LATEST_PATH.exists():
        return {"available": False, "reason": "No calculator run yet"}
    try:
        data = json.loads(CALCULATOR_LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "reason": "Parse error"}
    gen_at = data.get("generated_at_utc", "")
    stale = False
    if gen_at:
        try:
            gen_dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - gen_dt).total_seconds() > 172_800  # 48 h
        except Exception:
            pass
    filters = data.get("filters") or []
    sym = next((f.split("=", 1)[1] for f in filters if f.startswith("symbol=")), None)
    side = next((f.split("=", 1)[1] for f in filters if f.startswith("side=")), None)
    model_tag = data.get("model_tag") or (_MODEL_USEFULNESS.get((sym, side)) if sym and side else None)
    _route_label = {
        "LONG_DELAY0_TP60":       "LONG / TP60",
        "LONG_DELAY60_TP120":     "LONG / TP120 d60",
        "LONG_DELAY0_TP40_CONTROL": "LONG / TP40 (ctrl)",
        "SHORT_DELAY0_TP40":      "SHORT / TP40",
        "SHORT_DELAY0_TP60":      "SHORT / TP60",
        "SHORT_DELAY0_TP40_CONTROL": "SHORT / TP40 (ctrl)",
    }
    routes_out = []
    for route_id, rs in (data.get("route_simulation") or {}).items():
        if not rs:
            continue
        net = rs.get("net_bps") or {}
        routes_out.append({
            "route_id": route_id,
            "label": _route_label.get(route_id, route_id),
            "n": rs.get("n"),
            "median": net.get("median"),
            "wr": net.get("positive_rate"),
            "exit_mix": rs.get("exit_mix") or {},
        })
    analogs = []
    for ev in (data.get("nearest_analogs") or [])[:3]:
        analogs.append({
            "event_utc": (ev.get("event_utc") or "")[:10],
            "cluster_notional": ev.get("cluster_notional"),
            "day_trend_bps": ev.get("day_trend_bps"),
            "distance": ev.get("distance"),
        })
    oos = data.get("oos_validation") or {}
    oos_verdict = oos.get("verdict") or oos.get("summary") if oos.get("enabled") else None
    avail = (data.get("coverage") or {}).get("available_symbol_sides") or []
    scope_warning = None
    if len(avail) == 1 and avail[0].get("symbol") == "ETHUSDT" and avail[0].get("liq_side") == "BUY":
        scope_warning = "ETH BUY only — other symbol/sides not yet in feature DB"
    mc = data.get("model_config") or {}
    return {
        "available": True,
        "stale": stale,
        "generated_at": gen_at,
        "symbol": sym,
        "liq_side": side,
        "preset": data.get("preset"),
        "source_signal": data.get("source_signal"),
        "source_trade": data.get("source_trade"),
        "run_command": data.get("run_command"),
        "model_tag": model_tag,
        "prediction_mode": data.get("prediction_mode", "filter_scan"),
        "decision_grade": data.get("decision_grade", False),
        "warning": data.get("warning"),
        "threshold_source": data.get("threshold_source"),
        "min_notional_used": data.get("min_notional_used"),
        "model_config": {
            "k":               mc.get("k"),
            "metric":          mc.get("metric"),
            "feature_set":     mc.get("feature_set"),
            "excluded_features": mc.get("excluded_features"),
            "config_source":   mc.get("config_source"),
        },
        "decision_card": data.get("decision_card"),
        "selection_mode": data.get("selection_mode"),
        "candidate_n": data.get("candidate_n"),
        "matched_n": data.get("matched_n"),
        "confidence": data.get("confidence"),
        "filters": filters,
        "routes": routes_out,
        "nearest_analogs": analogs,
        "oos_verdict": oos_verdict,
        "scope_warning": scope_warning,
    }


def preliq_shadow_payload() -> dict[str, Any]:
    try:
        return s34_preliq_shadow_detector.current_preliq_shadow_payload(DB_PATH)
    except Exception as exc:  # noqa: BLE001 - panel should degrade without breaking /api/data.
        return {
            "available": False,
            "error": repr(exc),
            "mode": "SHADOW_ONLY",
            "decision_grade": False,
        }


def build_payload() -> dict[str, Any]:
    global LAST_GOOD, LAST_ERROR
    try:
        with timed_conn() as conn:
            end_ms = latest_mark_ms(conn)
            start_ms = end_ms - 4 * 60 * 60 * 1000
            prices = price_series(conn, start_ms, end_ms)
            current_price = float(prices[-1]["close"]) if prices else None
            trades_payload = read_json(TRADES_PATH, {"trades": []})
            trades = trades_payload.get("trades", []) if isinstance(trades_payload, dict) else []
            prices_by_symbol = current_prices(conn, trades)
            signals, open_trade, trade_data = trade_payload(start_ms, end_ms, prices_by_symbol)
            guardrails_by_rule = route_guardrail_payload()
            proc = process_health()
            runner_proc = next((row for row in proc if row["name"] == "s34_shadow_paper_runner"), None)
            if runner_proc:
                trade_data["summary"]["runner_alive"] = bool(runner_proc.get("alive"))
                trade_data["summary"]["runner_pid"] = runner_proc.get("pid")
            v02_mirror = v_engine_v02_shadow_mirror_payload(start_ms, end_ms)
            v02_h4_shadow = v02_h4_shadow_payload(start_ms, end_ms)
            sizing_shadow = sizing_shadow_payload()
            payload = {
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "window": {"start_utc": utc_iso_ms(start_ms), "end_utc": utc_iso_ms(end_ms)},
                "price": prices,
                "liq": liq_buckets(conn, start_ms, end_ms),
                "regime": regime_status(conn, end_ms),
                "signals": signals,
                "open_trade": open_trade,
                "levels": trade_data["levels"],
                "summary": trade_data["summary"],
                "stream_health": stream_health(conn, end_ms),
                "process_health": proc,
                "runner_stderr": runner_stderr_status(),
                "forward": trade_status_payload_with_guardrails(trades, prices_by_symbol, end_ms, guardrails_by_rule, _geometry_summary_payload()),
                "live_execution": live_execution_payload(trades, prices_by_symbol),
                "v_engine_v02_shadow_mirror": v02_mirror,
                "v02_h4_shadow": v02_h4_shadow,
                "sizing_shadow_paper": sizing_shadow,
                "shadow_paper_buckets": shadow_paper_buckets_payload(
                    mirror=v02_mirror,
                    h4_shadow=v02_h4_shadow,
                    sizing=sizing_shadow,
                    process_rows=proc,
                ),
                "v02_navigation_quality": v02_navigation_quality_series(conn, prices),
                "intelligence": intelligence_payload(),
                "risk_sandbox": risk_sandbox_payload(),
                "bucket_independence": bucket_independence_payload(),
                "constellation_routes": constellation_routes(trades),
                "disk": disk_status(),
                "host_health": host_health_payload(),
                "preliq_shadow": preliq_shadow_payload(),
                "similarity": similarity_payload(),
                "error": "",
            }
        LAST_GOOD = payload
        LAST_ERROR = ""
        return payload
    except Exception as exc:  # noqa: BLE001 - server should return last known chart.
        LAST_ERROR = repr(exc)
        if LAST_GOOD is not None:
            fallback = dict(LAST_GOOD)
            fallback["error"] = LAST_ERROR
            return fallback
        return {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window": {},
            "price": [],
            "liq": [],
            "regime": {"thresholds": REGIME_THRESHOLDS, "gates": {}, "regime_on": False},
            "signals": [],
            "open_trade": None,
            "levels": [],
            "summary": {"runner_alive": False},
            "stream_health": [],
            "process_health": process_health(),
            "runner_stderr": runner_stderr_status(),
            "forward": {"cards": [], "open_trades": [], "latest_closed": []},
            "live_execution": {"available": False, "error": "build error"},
            "v_engine_v02_shadow_mirror": {"available": False, "error": "build error"},
            "v02_h4_shadow": {"available": False, "error": "build error"},
            "sizing_shadow_paper": {"available": False, "error": "build error"},
            "shadow_paper_buckets": {"available": False, "error": "build error"},
            "v02_navigation_quality": {"available": False, "series": [], "latest": None},
            "intelligence": intelligence_payload(),
            "risk_sandbox": risk_sandbox_payload(),
            "bucket_independence": bucket_independence_payload(),
            "constellation_routes": [],
            "disk": disk_status(),
            "host_health": host_health_payload(),
            "preliq_shadow": preliq_shadow_payload(),
            "similarity": {"available": False, "reason": "build error"},
            "error": LAST_ERROR,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, socket.timeout, OSError) as exc:
            winerr = getattr(exc, "winerror", None)
            err_no = getattr(exc, "errno", None)
            if winerr in {10053, 10054} or err_no in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}:
                return
            raise

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/data":
            body = json.dumps(build_payload(), ensure_ascii=False).encode("utf-8")
            self.send_bytes(200, body, "application/json; charset=utf-8")
            return
        self.send_bytes(404, b"not found", "text/plain; charset=utf-8")


def serve(host: str, port: int, open_browser: bool) -> None:
    url = f"http://{host}:{port}" if host not in {"0.0.0.0", ""} else f"http://localhost:{port}"
    LIVE_CHART_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_CHART_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        print(f"S34 Live Chart running at {url}")
    except Exception:
        pass
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an S34 live chart from read-only microstructure.db.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--once", action="store_true", help="Print one /api/data payload and exit.")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(build_payload(), ensure_ascii=False, indent=2))
        return 0
    serve(str(args.host), int(args.port), not bool(args.no_browser))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            import traceback

            fatal_path = ROOT / "logs" / "s34_live_chart.fatal.log"
            fatal_path.parent.mkdir(parents=True, exist_ok=True)
            fatal_path.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise
