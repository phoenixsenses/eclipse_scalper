"""S34 Real-Time Shadow Runner — paper trades all validated signals, no live orders.

Polls every 5s. For each new ETH SELL cascade ≥200K:
  1. LONG_T15_BOUNCE : enter LONG at T+15m only if ETH mark reclaimed above anchor, exit anchor+4h
  2. SHORT_NOISY   : wait for first ETH SELL propagation ≥50K in 1-30min, enter SHORT 2h
  3. SHORT_NEITHER : if BTC SELL ≥2M detected in 30min window, enter SHORT 2h
  4. LONG_ECHO_45_120_SILENCE : shadow-only echo route, prior cascade 45-120m + silence
  5. SHORT_BTC1M_H4 : shadow-only BTC≥1M confirm route, 4h hold
  6. SHORT_BTC1M_D10_H3 : shadow-only BTC≥1M confirm after 10m, 3h hold
  7. LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE : shadow-only prebuild>=2 + silence
  8. LONG_OFI_SILENCE_BUYERS : shadow-only silence + post-cascade buyer OFI
  9. BUY_FADE_SHORT_H45_SL75 : shadow-only ETH BUY cascade -> SHORT, 45m hold, 75bps SL
 10. SHORT_NOISY_BTC1M_D5_H180 : shadow-only noisy ETH + BTC>=1M after 5m, 180m hold

All positions are paper-only (no exchange calls). Outcomes tracked vs mark price history.

Ledger : reports/shadow/s34_realtime_shadow.jsonl
State  : reports/shadow/s34_realtime_shadow_state.json

Usage:
    python tools/s34_realtime_shadow_runner.py [--once] [--db data/microstructure.db]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DEFAULT_DB    = ROOT / "data" / "microstructure.db"
LEDGER_PATH   = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
STATE_PATH    = ROOT / "reports" / "shadow" / "s34_state_machine_shadow_state.json"
PID_PATH      = ROOT / "logs" / "pids" / "s34_state_machine_shadow_runner.pid"
RULE_NAME     = "S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3"

# Signal parameters — must match research findings exactly
ETH_THRESH        = 200_000.0   # anchor cascade threshold
PROP_THRESH       = 50_000.0    # follow-on / propagation threshold
BTC_THRESH        = 2_000_000.0 # BTC cascade threshold for neither_silence
BTC_SHADOW_1M_THRESH = 1_000_000.0
SYNC_WIN_MS       = 10 * 60_000 # sync_k window (prior 10min)
SIL_LO_MS         = 60_000      # silence window low bound (skip ultra-early)
BTC_CONFIRM_MIN_DELAY_MS = 5 * 60_000
BTC_CONFIRM_D10_MS = 10 * 60_000
LONG_ENTRY_DELAY_MS = 15 * 60_000
SIL_HI_MS         = 30 * 60_000 # silence confirmed after 30min
NOISY_WIN_HI_MS   = 30 * 60_000 # noisy detection window top
HORIZON_LONG_MS   = 4 * 3600_000
HORIZON_LONG_H6_MS = 6 * 3600_000  # hour17 hold-through route (no early exit)
HOUR17_MIN_HOUR   = 17            # US afternoon/evening predictor (hour >= 17 UTC)
HORIZON_SHORT_MS  = 2 * 3600_000
HORIZON_SHORT_H3_MS = 3 * 3600_000
HORIZON_SHORT_H4_MS = 4 * 3600_000
FEE_BPS           = 5.0
BTC_7D_LOOKBACK_MS = 7 * 24 * 3600_000
ECHO_LO_MS        = 45 * 60_000
ECHO_HI_MS        = 120 * 60_000
OFI_WIN_MS        = 15 * 60_000
BUY_FADE_HOLD_MS  = 45 * 60_000
BUY_FADE_SL_BPS   = 75.0
PROFIT_LOCK_TRIGGER_BPS = 200.0   # TRADE_MGMT M5: 200/100 tek baseline-ustu varyant (100/50 avg 47.7 = KOTU)
PROFIT_LOCK_LOCK_BPS    = 100.0
SCALEIN_DIP_BPS         = -100.0  # TRADE_MGMT M7: ilk 2h'de -100bps dip -> +1 birim (perU 86.3, mdd iyilesir)
SCALEIN_WINDOW_MS       = 2 * 3600_000
# OD-004 (operator onayi 2026-07-10): bd_first_buy50 observation-only exit gozlemcisi.
# EV-BUYFADE-SILEXIT-001 en iyi adayi (8/9 kontrol, val econ +1.37<3bps REJECTED) —
# forward delta birikimi icin gozlemde; SIPARIS YOK, baseline cikisi DEGISMEZ.
BD_FIRST_BUY50_ACTIVATION_UTC = "2026-07-10T18:00:00+00:00"  # frozen activation-ts (Phase 8 disiplini)
# OD-004-FOLLOWUP (independent-review corrective): activation_utc is an
# ENFORCED event-time lower boundary, not a label. The observer's search
# window never starts before max(entry_ts_ms + 30min, this ms value) —
# restart timing itself is never the economic boundary.
BD_FIRST_BUY50_ACTIVATION_MS  = int(datetime.fromisoformat(BD_FIRST_BUY50_ACTIVATION_UTC).timestamp() * 1000)
BD_FIRST_BUY50_THRESH_USD     = 50_000.0
BD_FIRST_BUY50_START_MS       = 30 * 60_000  # T+30m sonrasi ilk yeni BUY>=50K
BD_FIRST_BUY50_PROTOCOL_VERSION = "v2"
# Process-instance boundary: set once at import (= once per process
# lifetime/restart). Used only to LABEL whether a found event predates this
# process (reconstructed_after_restart) -- never gates selection itself.
_PROCESS_START_MS = int(time.time() * 1000)
SCALEIN_ROUTES = {"LONG_HOUR17_HOLD6H", "LONG_HOUR17_COMPOSITE", "LONG_HOUR17_100K_COMPOSITE"}

BUCKET_SEC        = 300
MIN_GAP_SEC       = 900
ACCEL_WIN_SEC     = 30
LOOKBACK_SEC      = 3 * 3600    # how far back to look for fresh anchors
SIGNAL_FRESH_MS   = 120_000     # max age for new anchor to trigger
POLL_SEC          = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ts_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _scalar(conn: sqlite3.Connection, sql: str, params: tuple) -> float:
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row else 0.0

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def has_bucketed_cascade(conn, sym, side, lo, hi, threshold, bucket_ms=5 * 60_000):
    row = conn.execute(
        """
        SELECT 1
        FROM (
            SELECT CAST((ts_ms - ?) / ? AS INTEGER) AS bucket_id,
                   SUM(notional) AS bucket_notional
            FROM liquidations
            WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?
            GROUP BY bucket_id
        )
        WHERE bucket_notional >= ?
        LIMIT 1
        """,
        (int(lo), int(bucket_ms), sym, side, int(lo), int(hi), float(threshold)),
    ).fetchone()
    return bool(row)

def agg_trade_ofi(conn, sym, lo, hi):
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END), 0)
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        """,
        (sym, int(lo), int(hi)),
    ).fetchone()
    buy_notional = float(row[0] or 0.0) if row else 0.0
    sell_notional = float(row[1] or 0.0) if row else 0.0
    return {
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "ofi_notional": buy_notional - sell_notional,
    }

def mark_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    return float(row[0]) if row else None

def prior_bps(conn, sym, ts_ms, lookback_ms):
    p0 = mark_at(conn, sym, ts_ms - lookback_ms)
    p1 = mark_at(conn, sym, ts_ms)
    if p0 and p1 and p0 > 0:
        return (p1 - p0) / p0 * 10_000.0
    return 0.0


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(conn, ts_ms: int, sync_k: float, n2h: int) -> int:
    """Score 0-6 at cascade time (no silence gate — added separately when confirmed)."""
    hour   = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    b4h    = prior_bps(conn, "BTCUSDT", ts_ms, 4 * 3600_000)
    book   = book_features_at(conn, "ETHUSDT", ts_ms, 30)
    vdep   = float(book.get("vdepth_bps") or 0) if book else 0.0
    sess_us = 13 <= hour < 21
    return sum([
        int(n2h >= 3),
        int(b4h < 0),
        int(vdep >= 30),
        int(sess_us),
        int(sync_k >= 200_000),
    ])  # silence adds +1 later = max 6


# Composite conviction thresholds (research S34_CONVICTION_COMPOSITE + S34_DEEP_QUESTIONS)
COMP_SYNC_RATIO = 0.5421
COMP_RV5M       = 0.0304
COMP_DENSITY24  = 5.0
COMP_BE_LO      = 0.2195
COMP_BE_HI      = 2.0
COMP_IMB        = 0.2633
COMP_SHELF      = 2_775_000.0   # Q6: 24h ETH SELL notional in [price*0.98, price] (hour17 median)
COMP_FUNDING_VETO_MIN = 60.0    # Q8: cascade <60m to funding = DEAD -> veto
COMP_WHALE      = 6440.0        # HORIZON: pre-5m avg agg-trade size < this = retail (holdout WR 94%)
COMP_RV_PROXY   = 0.0026337     # RV_STALE_FIX: mark-price 1m log-return RMS(5m) TRAIN medyani
RV_STALE_MAX_MS = 10 * 60_000   # vol_state bundan eskiyse mark-proxy kullan
COMP_TSL_MIN    = 114.65        # SIGNAL_DISCOVERY_V2: onceki 200K anchor'dan dk (TRAIN medyan); < bu = clustering iyi
COMP_TWO_SIDED  = 68_425.0      # SIGNAL_DISCOVERY_V2: pre-1h ETH BUY liq notional (TRAIN medyan); >= bu = iki-yonlu stres iyi

# ── MECH SCORE (S34_MECHANISM_TAXONOMY esikleri; FORWARD-ONLY loglama) ────────
# Gecmis trade'lere GERIYE DONUK mech_score EKLENMEZ; feature bu tarihten itibaren
# loglanir ve oncesindeki hicbir trade E-MECHCOMP forward kaniti sayilamaz.
MECH_SCHEMA_VERSION  = "mech_v1"
MECH_FEATURE_VERSION = "2026-07-02"
MECH_CODE_REF        = "tools/s34_realtime_shadow_runner.py@mech_v1"
MECH_THR = {"fund_rate": 1.39e-06, "px_rv": 0.0021055590, "whale": 6941.57,
            "ofi": -0.1503316, "two_sided": 72_113.61, "pull": 7.9746e-06,
            "refill": 1.0319052}
# missing-value policy: eksik sensor -> ilgili bilesken None, skora katilmaz,
# mech dict'te missing listesine yazilir (imputation YOK).


def compute_mech_features(conn, ts_ms: int, rn: float) -> dict:
    """PRE-only mech bileskenleri (T0'da bilinen). refill CLOSE'da eklenir."""
    missing: list[str] = []
    fr = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    fund = float(fr[0]) if fr and fr[0] is not None else None
    if fund is None: missing.append("fund_rate")
    rv, _thr = rv5m_robust(conn, ts_ms)
    rv_hit = None
    if rv is None:
        missing.append("px_rv")
    else:
        rv_hit = rv >= MECH_THR["px_rv"]
    o = agg_trade_ofi(conn, "ETHUSDT", ts_ms - 10 * 60_000, ts_ms)
    tot = o["buy_notional"] + o["sell_notional"]
    ofi10 = ((o["ofi_notional"] / tot) if tot > 0 else None)
    wc = conn.execute("SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (ts_ms - 10 * 60_000, ts_ms)).fetchone()
    whale = (tot / int(wc[0])) if wc and wc[0] else None
    if ofi10 is None: missing.append("ofi10")
    if whale is None: missing.append("whale")
    two = liq_sum(conn, "ETHUSDT", "BUY", ts_ms - 3600_000, ts_ms)
    pre10 = conn.execute("SELECT AVG(bid_qty) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (ts_ms - 600_000, ts_ms)).fetchone()
    pre1m = conn.execute("SELECT MIN(bid_qty) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (ts_ms - 60_000, ts_ms)).fetchone()
    pull = None
    if pre10 and pre10[0] and pre1m and pre1m[0] is not None and float(pre10[0]) > 0:
        pull = float(pre1m[0]) / float(pre10[0])
    else:
        missing.append("bk_pull")
    hits = {"fund_lo": (fund < MECH_THR["fund_rate"]) if fund is not None else None,
            "rv_hi": rv_hit,
            "retail": (whale < MECH_THR["whale"]) if whale is not None else None,
            "ofi_hi": (ofi10 >= MECH_THR["ofi"]) if ofi10 is not None else None,
            "two_sided_hi": two >= MECH_THR["two_sided"],
            "pull_hi": (pull >= MECH_THR["pull"]) if pull is not None else None}
    score_pre = sum(1 for v in hits.values() if v is True)
    return {"schema_version": MECH_SCHEMA_VERSION, "feature_version": MECH_FEATURE_VERSION,
            "code_ref": MECH_CODE_REF, "calc_ts_ms": int(time.time() * 1000),
            "provenance": {"tables": ["mark_prices", "agg_trades", "liquidations", "book_ticker"],
                           "thresholds": "S34_MECHANISM_TAXONOMY.json (ungated-TRAIN medyan)"},
            "missing": missing, "missing_policy": "component=None, skora katilmaz, imputation yok",
            "values": {"fund_rate": fund, "rv": rv, "whale": whale, "ofi10": ofi10,
                       "two_sided_1h": round(two, 0), "pull": pull},
            "hits": hits, "mech_score_pre": score_pre, "mech_score_pre_max": 6,
            "refill": None, "refill_hi": None, "mech_score_full": None}


def rv5m_robust(conn, ts_ms: int) -> tuple[float | None, float]:
    """(rv, esik) doner. vol_state taze ise onu kullanir; degilse mark_prices'tan
    1m log-return RMS(5m) proxy hesaplar (vol_state producer'i 2026-06-05'te oldu)."""
    vs = conn.execute("SELECT rv_5m, ts_ms FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    if vs and vs[0] is not None and (ts_ms - int(vs[1])) <= RV_STALE_MAX_MS:
        return float(vs[0]), COMP_RV5M
    px = []
    for k in range(5, -1, -1):
        r = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms - k * 60_000,)).fetchone()
        if not r or r[0] is None or float(r[0]) <= 0:
            return None, COMP_RV_PROXY
        px.append(float(r[0]))
    rets = [math.log(px[i + 1] / px[i]) for i in range(5)]
    return math.sqrt(sum(x * x for x in rets)), COMP_RV_PROXY

def compute_composite_score(conn, ts_ms: int, sync_k: float, rn: float, density24: int) -> tuple[int, dict]:
    """T0-knowable conviction score 0-8 (robust mining signals) + funding veto flag.
    Higher = stronger bounce. comp['funding_veto']=True -> skip (Q8)."""
    sync_ratio = (sync_k / rn) if rn > 0 else 0.0
    rv5m, rv_thr = rv5m_robust(conn, ts_ms)
    o = agg_trade_ofi(conn, "ETHUSDT", ts_ms - 5 * 60_000, ts_ms)
    tot = o["buy_notional"] + o["sell_notional"]
    ofi_ratio = (o["ofi_notional"] / tot) if tot > 0 else None
    wc = conn.execute("SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (ts_ms - 5 * 60_000, ts_ms)).fetchone()
    whale = (tot / int(wc[0])) if wc and wc[0] else None   # avg agg-trade size (retail if small)
    btc_conc = liq_max(conn, "BTCUSDT", "SELL", ts_ms - 10 * 60_000, ts_ms)
    be_ratio = (btc_conc / rn) if rn > 0 else 0.0
    bt = conn.execute("SELECT book_imbalance, ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    imb = float(bt[0]) if bt and bt[0] is not None and (ts_ms - int(bt[1])) <= 5 * 60_000 else None
    # Q6 liq-shelf: clustered ETH SELL fuel just below current price (prior 24h)
    pr = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    px = float(pr[0]) if pr and pr[0] is not None else None
    shelf = 0.0
    if px and px > 0:
        shelf = _scalar(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts_ms - 24 * 3600_000, ts_ms, px * 0.98, px))
    # Q8 funding veto
    fr = conn.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    min_to_fund = ((int(fr[0]) - ts_ms) / 60_000) if fr and fr[0] else None
    funding_veto = bool(min_to_fund is not None and min_to_fund < COMP_FUNDING_VETO_MIN)
    # SIGNAL_DISCOVERY_V2 (2026-07-02) yeni T0 alanlari — forward gozlem + score10/would_units
    prev_ancs = reconstruct_anchors(
        load_liquidations(conn, "ETHUSDT", "SELL", ts_ms - 48 * 3600_000, ts_ms - MIN_GAP_SEC * 1000),
        bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_WIN_SEC)
    tsl_min = ((ts_ms - int(prev_ancs[-1].anchor_ts_ms)) / 60_000.0) if prev_ancs else None
    two_sided = liq_sum(conn, "ETHUSDT", "BUY", ts_ms - 3600_000, ts_ms)
    frr = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    funding_rate = float(frr[0]) if frr and frr[0] is not None else None
    lt = conn.execute("SELECT price FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    basis_bps = ((px - float(lt[0])) / float(lt[0]) * 1e4) if (px and lt and lt[0] is not None and float(lt[0]) > 0) else None
    rv_hit = bool(rv5m is not None and rv5m >= rv_thr)
    shelf_hit = bool(shelf >= COMP_SHELF)
    whale_hit = bool(whale is not None and whale < COMP_WHALE)
    tsl_hit = bool(tsl_min is not None and tsl_min < COMP_TSL_MIN)
    two_sided_hit = bool(two_sided >= COMP_TWO_SIDED)
    # DIVERSIFICATION double-trigger sizing gozlemcisi (order yok, sadece kayit)
    would_units = 3.0 if (rv_hit and shelf_hit and whale_hit) else (2.0 if (rv_hit and shelf_hit) else 1.0)
    comp = {
        "sync_ratio": round(sync_ratio, 4), "rv5m": rv5m, "density24": density24,
        "ofi_pre": None if ofi_ratio is None else round(ofi_ratio, 4),
        "be_ratio": round(be_ratio, 4), "book_imb": imb, "shelf": round(shelf, 0),
        "whale": None if whale is None else round(whale, 0),
        "min_to_fund": None if min_to_fund is None else round(min_to_fund, 1),
        "funding_veto": funding_veto,
        "rv_thr": rv_thr, "rv_hit": rv_hit, "shelf_hit": shelf_hit, "whale_hit": whale_hit,
        "tsl_min": None if tsl_min is None else round(tsl_min, 1), "tsl_hit": tsl_hit,
        "two_sided_usd": round(two_sided, 0), "two_sided_hit": two_sided_hit,
        "funding_rate": funding_rate,
        "basis_bps": None if basis_bps is None else round(basis_bps, 2),
        "would_units": would_units,
    }
    score = 0
    if sync_ratio >= COMP_SYNC_RATIO: score += 1
    if rv_hit: score += 1
    if density24 >= COMP_DENSITY24: score += 1
    if ofi_ratio is not None and ofi_ratio >= 0.0: score += 1        # buyers
    if COMP_BE_LO <= be_ratio < COMP_BE_HI: score += 1               # moderate (extreme veto)
    if imb is not None and imb <= COMP_IMB: score += 1               # ask-heavy
    if shelf_hit: score += 1                                         # Q6 liq-shelf fuel
    if whale_hit: score += 1                                         # HORIZON retail (whale_lo, holdout WR94)
    comp["score10"] = score + int(tsl_hit) + int(two_sided_hit)      # SIGNAL_DISCOVERY_V2 genisletilmis skor
    return score, comp


def session_name(ts_ms: int) -> str:
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= hour < 13:
        return "EUROPE"
    if 13 <= hour < 21:
        return "US"
    return "OFF"


def is_bull_pullback(conn, ts_ms: int) -> bool:
    eth1h = prior_bps(conn, "ETHUSDT", ts_ms, 3600_000)
    btc4h = prior_bps(conn, "BTCUSDT", ts_ms, 4 * 3600_000)
    return eth1h > 20.0 and btc4h > 50.0


# ── State / ledger ────────────────────────────────────────────────────────────

def load_state() -> dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return {"positions": {}, "processed": {}, "pnl": {}}
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d.setdefault("positions", {})
    d.setdefault("processed", {})
    d.setdefault("pnl", {})
    d.setdefault("rule_name", RULE_NAME)
    d.setdefault("mode", "OBSERVE_ONLY_NO_ORDER")
    return d

def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

def log_event(rec: dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def init_profit_lock_observer(pos: dict) -> None:
    """Attach a shadow-only profit-lock observer to a paper position."""
    pos.setdefault("profit_lock_observer", {
        "protocol": "PROFIT_LOCK_200_100_OBSERVER",
        "trigger_bps": PROFIT_LOCK_TRIGGER_BPS,
        "lock_bps": PROFIT_LOCK_LOCK_BPS,
        "triggered": False,
        "shadow_exit": False,
        "mode": "OBSERVE_ONLY_NO_ORDER",
    })


def update_scalein_observer(pos: dict, conn, now_ms: int) -> None:
    """Shadow-only scale-in gozlemcisi (TRADE_MGMT M7): hour17 LONG'da ilk 2h icinde
    -100bps dip gelirse paper +1 birim eklendi say; kapanista birlesik per-unit hesaplanir.
    Order gondermez, pozisyonu degistirmez."""
    if pos.get("signal") not in SCALEIN_ROUTES:
        return
    if pos.get("status") not in {"OPEN", "OPEN_SILENCE", "OPEN_NOISY"}:
        return
    ets = pos.get("entry_ts_ms")
    if not ets or not pos.get("entry_price"):
        return
    obs = pos.setdefault("scalein_observer", {
        "protocol": "SCALEIN_DIP100_2H_OBSERVER", "dip_bps": SCALEIN_DIP_BPS,
        "window_ms": SCALEIN_WINDOW_MS, "added": False, "mode": "OBSERVE_ONLY_NO_ORDER",
    })
    if obs.get("added") or now_ms > int(ets) + SCALEIN_WINDOW_MS:
        return
    pnl = _paper_pnl_bps(pos, conn, now_ms)
    if pnl <= SCALEIN_DIP_BPS:
        obs["added"] = True
        obs["add_ts_ms"] = now_ms
        obs["add_utc"] = ts_to_utc(now_ms)
        obs["add_pnl_bps"] = round(pnl, 2)
        log_event({**pos, "event": "SCALEIN_ADD", "scalein_observer": obs})
        print(f"{utc_now()} [SHD] SCALEIN_ADD pnl={pnl:.1f}bps id={pos.get('id')}")


def _paper_pnl_bps(pos: dict, conn, now_ms: int) -> float:
    entry_px = float(pos.get("entry_price") or 0.0)
    if entry_px <= 0:
        return 0.0
    px = mark_at(conn, "ETHUSDT", now_ms) or entry_px
    raw = (px - entry_px) / entry_px * 10_000.0
    return -raw if pos.get("direction") == "SHORT" else raw


def update_profit_lock_observer(pos: dict, conn, now_ms: int) -> None:
    """Shadow-only 100/50 profit-lock. It logs would-exit events, never closes."""
    if pos.get("status") not in {"OPEN", "OPEN_SILENCE", "OPEN_NOISY"}:
        return
    if not pos.get("entry_ts_ms") or not pos.get("entry_price"):
        return
    init_profit_lock_observer(pos)
    obs = pos["profit_lock_observer"]
    pnl = _paper_pnl_bps(pos, conn, now_ms)
    obs["last_seen_ts_ms"] = now_ms
    obs["last_seen_pnl_bps"] = round(pnl, 2)
    if not obs.get("triggered") and pnl >= PROFIT_LOCK_TRIGGER_BPS:
        obs["triggered"] = True
        obs["trigger_ts_ms"] = now_ms
        obs["trigger_utc"] = ts_to_utc(now_ms)
        obs["trigger_pnl_bps"] = round(pnl, 2)
        log_event({**pos, "event": "PROFIT_LOCK_TRIGGER", "profit_lock_observer": obs})
        print(f"{utc_now()} [SHD] PROFIT_LOCK_TRIGGER pnl={pnl:.1f}bps id={pos.get('id')}")
    if obs.get("triggered") and not obs.get("shadow_exit") and pnl <= PROFIT_LOCK_LOCK_BPS:
        obs["shadow_exit"] = True
        obs["shadow_exit_ts_ms"] = now_ms
        obs["shadow_exit_utc"] = ts_to_utc(now_ms)
        obs["shadow_exit_pnl_bps"] = round(pnl, 2)
        obs["shadow_exit_net_bps"] = round(pnl - FEE_BPS, 2)
        log_event({**pos, "event": "PROFIT_LOCK_SHADOW_EXIT", "profit_lock_observer": obs})
        print(f"{utc_now()} [SHD] PROFIT_LOCK_SHADOW_EXIT net={obs['shadow_exit_net_bps']:.1f}bps id={pos.get('id')}")


# ── Per-anchor signal evaluation ──────────────────────────────────────────────

def open_signals_for_anchor(conn, anchor_ts_ms: int, anchor_price: float,
                             running_notional: float, now_ms: int, state: dict) -> list[dict]:
    ts = anchor_ts_ms
    base_id = f"SHD:{ts}"
    new_pos = []

    # Features knowable at cascade time
    sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts)
             + liq_sum(conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts))
    n2h    = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
    book   = book_features_at(conn, "ETHUSDT", ts, 30)
    bid_dep = float(book.get("bid_depth_usd") or 0) if book else 0.0
    score  = compute_score(conn, ts, sync_k, n2h)
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    hour = dt.hour
    dow = dt.weekday()
    session = session_name(ts)
    bull = is_bull_pullback(conn, ts)
    btc7d = prior_bps(conn, "BTCUSDT", ts, BTC_7D_LOOKBACK_MS)
    btc4h = prior_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    btc5m = prior_bps(conn, "BTCUSDT", ts, 5 * 60_000)
    density_24h = liq_cnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, ETH_THRESH)
    prebuildup   = liq_cnt(conn, "ETHUSDT", "SELL", ts - 30 * 60_000, ts - 1000, PROP_THRESH)
    echo_45_120 = has_bucketed_cascade(conn, "ETHUSDT", "SELL", ts - ECHO_HI_MS, ts - ECHO_LO_MS, ETH_THRESH)
    long_score = score + 1
    blocked_us_13_14 = bool(session == "US" and hour in {13, 14})
    long_eligible = (
        (not bull)
        and session != "EUROPE"
        and not blocked_us_13_14
        and dow not in {0, 2}
        and sync_k < 200_000
        and (btc4h < 0.0 or btc7d < 0.0)
        and long_score >= 3
    )
    short_eligible = (not bull) and session != "EUROPE" and dow != 6 and score >= 4

    # 1 — LONG_T15_BOUNCE: live mirror, delayed knowable entry.
    lid = f"{base_id}:LS"
    if lid not in state["processed"] and long_eligible:
        pos = {
            "id": lid, "signal": "LONG_T15_BOUNCE", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
            "exit_due_ms": None,
            "status": "MONITORING_T15_BOUNCE",
            "entry_due_ms": ts + LONG_ENTRY_DELAY_MS,
            "sil_check_ms": ts + SIL_HI_MS,
            "anchor_price": anchor_price,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
        }
        new_pos.append(pos)
        state["processed"][lid] = utc_now()

    # 2 — SHORT_NEITHER: ETH cascade + BTC cascade → SHORT immediately
    nid = f"{base_id}:SN"
    if nid not in state["processed"] and short_eligible:
        btc_now = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_MIN_DELAY_MS, now_ms)
        if btc_now >= BTC_THRESH:
            pos = {
                "id": nid, "signal": "SHORT_NEITHER", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": now_ms, "entry_price": anchor_price,
                "exit_due_ms": now_ms + HORIZON_SHORT_MS,
                "status": "OPEN",
                "btc_max": btc_now,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
            }
            init_profit_lock_observer(pos)
            new_pos.append(pos)
            state["processed"][nid] = utc_now()
        elif now_ms < ts + SIL_HI_MS:
            # Queue for BTC monitoring
            pos = {
                "id": nid, "signal": "SHORT_NEITHER", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_BTC",
                "sil_check_ms": ts + SIL_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][nid] = utc_now()

    # 3 — LONG_ECHO_45_120_SILENCE: observe prior-cascade echo; count only if silence confirms.
    eid = f"{base_id}:ECHO45"
    if eid not in state["processed"] and long_eligible and echo_45_120:
        pos = {
            "id": eid, "signal": "LONG_ECHO_45_120_SILENCE", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_MS,
            "status": "MONITORING_ECHO_SILENCE",
            "sil_check_ms": ts + SIL_HI_MS,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
            "observer_note": "shadow_only_echo_route_counts_after_silence_confirmation",
        }
        init_profit_lock_observer(pos)
        new_pos.append(pos)
        state["processed"][eid] = utc_now()

    # 3b — LONG_HOUR17_HOLD6H: T0-knowable hold predictor (validated deploy candidate).
    #   Gate: not bull, not EUROPE, regime(btc4h<0 OR btc7d<0), hour>=17 UTC.
    #   Mechanics: enter at anchor T0, HOLD 6h, NO early-exit on noise, NO stop.
    #   Research: S34_SILENCE_PREDICTOR — ~14-16/mo, WR ~60-62%, OOS mc_p 0.003-0.021, mdd small.
    hour17_eligible = (
        (not bull)
        and session != "EUROPE"
        and (btc4h < 0.0 or btc7d < 0.0)
        and hour >= HOUR17_MIN_HOUR
    )
    hid = f"{base_id}:H17"
    if hid not in state["processed"] and hour17_eligible:
        pos = {
            "id": hid, "signal": "LONG_HOUR17_HOLD6H", "direction": "LONG",
            "mech": compute_mech_features(conn, ts, running_notional),
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_H6_MS,
            "status": "OPEN",
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "hour": hour,
            "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
            "observer_note": "hold_predictor_hour17_no_early_exit_no_stop",
        }
        init_profit_lock_observer(pos)
        log_event({**pos, "event": "OPEN"})
        new_pos.append(pos)
        state["processed"][hid] = utc_now()

    # 3c — LONG_HOUR17_COMPOSITE: hour17 + conviction score>=3 (research S34_CONVICTION_COMPOSITE).
    #   Score 0-6 from robust T0 signals (sync_ratio, rv5m, density24, ofi_pre, be_ratio-mod, ask-heavy).
    #   Monotonic (OOS: score>=3 WR 82%, score>=4 WR 90%). Entry T0, hold 6h, no early-exit.
    #   Shadow-only; forward-validates the conviction score. Score stored for per-score dashboard stats.
    comp_score, comp = compute_composite_score(conn, ts, sync_k, running_notional, density_24h)
    cid = f"{base_id}:H17C"
    if cid not in state["processed"] and hour17_eligible and comp_score >= 3 and not comp.get("funding_veto"):
        pos = {
            "id": cid, "signal": "LONG_HOUR17_COMPOSITE", "direction": "LONG",
            "mech": compute_mech_features(conn, ts, running_notional),
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_H6_MS,
            "status": "OPEN",
            "conviction_score": comp_score, "conviction": comp,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "hour": hour, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup, "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14, "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional, "opened_utc": utc_now(),
            "observer_note": "conviction_composite_score_ge3_no_early_exit_no_stop",
        }
        init_profit_lock_observer(pos)
        log_event({**pos, "event": "OPEN"})
        new_pos.append(pos)
        state["processed"][cid] = utc_now()

    # 4 — SHORT_BTC1M_H4: observe BTC>=1M confirm with 4h hold, separate from live 2M/2h route.
    sid4 = f"{base_id}:SB1M4H"
    if sid4 not in state["processed"] and short_eligible:
        btc_now_1m = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_MIN_DELAY_MS, now_ms)
        if btc_now_1m >= BTC_SHADOW_1M_THRESH:
            pos = {
                "id": sid4, "signal": "SHORT_BTC1M_H4", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": now_ms, "entry_price": anchor_price,
                "exit_due_ms": now_ms + HORIZON_SHORT_H4_MS,
                "status": "OPEN",
                "btc_max": btc_now_1m,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_btc1m_confirm_4h_hold",
            }
            init_profit_lock_observer(pos)
            new_pos.append(pos)
            state["processed"][sid4] = utc_now()
        elif now_ms < ts + SIL_HI_MS:
            pos = {
                "id": sid4, "signal": "SHORT_BTC1M_H4", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_BTC_1M_H4",
                "sil_check_ms": ts + SIL_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_btc1m_confirm_4h_hold",
            }
            new_pos.append(pos)
            state["processed"][sid4] = utc_now()

    # 5 — SHORT_BTC1M_D10_H3: clean SHORT candidate, BTC>=1M after 10m, 3h hold.
    sid3 = f"{base_id}:SB1MD10H3"
    if sid3 not in state["processed"] and short_eligible:
        btc_now_1m_d10 = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_D10_MS, now_ms)
        if btc_now_1m_d10 >= BTC_SHADOW_1M_THRESH:
            pos = {
                "id": sid3, "signal": "SHORT_BTC1M_D10_H3", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": now_ms, "entry_price": anchor_price,
                "exit_due_ms": now_ms + HORIZON_SHORT_H3_MS,
                "status": "OPEN",
                "btc_max": btc_now_1m_d10,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_btc1m_delay10_3h_hold",
            }
            init_profit_lock_observer(pos)
            new_pos.append(pos)
            state["processed"][sid3] = utc_now()
        elif now_ms < ts + SIL_HI_MS:
            pos = {
                "id": sid3, "signal": "SHORT_BTC1M_D10_H3", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_BTC_1M_D10_H3",
                "sil_check_ms": ts + SIL_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_btc1m_delay10_3h_hold",
            }
            new_pos.append(pos)
            state["processed"][sid3] = utc_now()

    # 5b — SHORT_NOISY_BTC1M_D5_H180: clean diversifier from full signal boost gauntlet.
    #   Gate: not bull, not EUROPE, regime(btc4h<0 OR btc7d<0).
    #   Sequence: ETH SELL propagation >=50K in 1-30m AND BTC SELL >=1M after T+5m within 30m.
    #   Entry: max(prop_ts, btc_ts), hold 180m.
    #   Research: S34_FULL_SIGNAL_BOOST — no-overlap N=14, WR=92.9%, avg +110.6bps, mc_p=0.003.
    snd5 = f"{base_id}:SND5H180"
    short_noisy_btc1m_eligible = (
        (not bull)
        and session != "EUROPE"
        and (btc4h < 0.0 or btc7d < 0.0)
    )
    if snd5 not in state["processed"] and short_noisy_btc1m_eligible and now_ms < ts + NOISY_WIN_HI_MS + 5_000:
        prop_ts = liq_first_ts(
            conn, "ETHUSDT", "SELL",
            ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH,
        )
        btc_ts = liq_first_ts(
            conn, "BTCUSDT", "SELL",
            ts + BTC_CONFIRM_MIN_DELAY_MS, min(now_ms, ts + NOISY_WIN_HI_MS), BTC_SHADOW_1M_THRESH,
        )
        if prop_ts is not None and btc_ts is not None:
            entry_ts = max(int(prop_ts), int(btc_ts))
            px = mark_at(conn, "ETHUSDT", entry_ts) or anchor_price
            pos = {
                "id": snd5, "signal": "SHORT_NOISY_BTC1M_D5_H180", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": entry_ts, "entry_price": px,
                "exit_due_ms": entry_ts + HORIZON_SHORT_H3_MS,
                "status": "OPEN",
                "prop_ts_ms": prop_ts,
                "btc_confirm_ts_ms": btc_ts,
                "prop_delay_ms": prop_ts - ts,
                "btc_delay_ms": btc_ts - ts,
                "btc_threshold_usd": BTC_SHADOW_1M_THRESH,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "running_notional": running_notional,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_short_noisy_btc1m_delay5_hold180",
            }
            init_profit_lock_observer(pos)
            new_pos.append(pos)
            state["processed"][snd5] = utc_now()
        else:
            pos = {
                "id": snd5, "signal": "SHORT_NOISY_BTC1M_D5_H180", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_SHORT_NOISY_BTC1M_D5_H180",
                "sil_check_ms": ts + NOISY_WIN_HI_MS,
                "prop_ts_ms": prop_ts,
                "btc_confirm_ts_ms": btc_ts,
                "btc_threshold_usd": BTC_SHADOW_1M_THRESH,
                "sync_k": sync_k, "n2h": n2h, "score": score, "dow": dow,
                "session": session, "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
                "density_24h": density_24h, "prebuildup": prebuildup,
                "echo_45_120": echo_45_120,
                "bull_pullback": bull, "rule_name": RULE_NAME,
                "running_notional": running_notional,
                "opened_utc": utc_now(),
                "observer_note": "shadow_only_short_noisy_btc1m_delay5_hold180",
            }
            new_pos.append(pos)
            state["processed"][snd5] = utc_now()

    # 6 — LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE: prebuild>=2, count only after silence.
    did = f"{base_id}:PRE2"
    if did not in state["processed"] and long_eligible and prebuildup >= 2:
        pos = {
            "id": did, "signal": "LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_MS,
            "status": "MONITORING_PREBUILD_SILENCE",
            "sil_check_ms": ts + SIL_HI_MS,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
            "observer_note": "shadow_only_prebuild2_counts_after_silence_confirmation",
        }
        init_profit_lock_observer(pos)
        new_pos.append(pos)
        state["processed"][did] = utc_now()

    # 7 — LONG_OFI_SILENCE_BUYERS: count only after silence and post-cascade OFI>0.
    oid = f"{base_id}:OFIBUY"
    if oid not in state["processed"] and long_eligible:
        pos = {
            "id": oid, "signal": "LONG_OFI_SILENCE_BUYERS", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_MS,
            "status": "MONITORING_OFI_SILENCE",
            "sil_check_ms": ts + SIL_HI_MS,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "long_score": long_score, "dow": dow, "session": session,
            "btc7d_bps": btc7d, "btc4h_bps": btc4h, "btc5m_bps": btc5m,
            "density_24h": density_24h, "prebuildup": prebuildup,
            "echo_45_120": echo_45_120,
            "blocked_us_13_14": blocked_us_13_14,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
            "observer_note": "shadow_only_ofi_buyers_counts_after_silence_confirmation",
        }
        init_profit_lock_observer(pos)
        new_pos.append(pos)
        state["processed"][oid] = utc_now()

    # 8 — SHORT_NOISY: first ETH SELL propagation ≥50K in 1-30min window
    snid = f"{base_id}:SP"
    if snid not in state["processed"] and now_ms < ts + NOISY_WIN_HI_MS + 5_000:
        prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                               ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH)
        if prop_ts is not None:
            px = mark_at(conn, "ETHUSDT", prop_ts) or anchor_price
            pos = {
                "id": snid, "signal": "SHORT_NOISY", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": prop_ts, "entry_price": px,
                "exit_due_ms": prop_ts + HORIZON_SHORT_MS,
                "status": "OPEN",
                "prop_delay_ms": prop_ts - ts,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            init_profit_lock_observer(pos)
            new_pos.append(pos)
            state["processed"][snid] = utc_now()
        elif now_ms < ts + NOISY_WIN_HI_MS:
            pos = {
                "id": snid, "signal": "SHORT_NOISY", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_PROP",
                "sil_check_ms": ts + NOISY_WIN_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][snid] = utc_now()

    return new_pos


def is_bear_squeeze_continuation_risk(conn, ts_ms: int) -> bool:
    eth1h = prior_bps(conn, "ETHUSDT", ts_ms, 3600_000)
    btc4h = prior_bps(conn, "BTCUSDT", ts_ms, 4 * 3600_000)
    return eth1h < -20.0 and btc4h < -50.0


def open_buy_fade_for_anchor(conn, anchor_ts_ms: int, anchor_price: float,
                             running_notional: float, now_ms: int, state: dict) -> list[dict]:
    ts = int(anchor_ts_ms)
    base_id = f"BUYF:{ts}"
    bid = f"{base_id}:H45SL75"
    if bid in state["processed"]:
        return []

    session = session_name(ts)
    if session == "EUROPE" or is_bear_squeeze_continuation_risk(conn, ts):
        state["processed"][bid] = utc_now()
        return []

    sync_k = (
        liq_sum(conn, "BTCUSDT", "BUY", ts - SYNC_WIN_MS, ts)
        + liq_sum(conn, "SOLUSDT", "BUY", ts - SYNC_WIN_MS, ts)
    )
    n2h = liq_cnt(conn, "ETHUSDT", "BUY", ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
    prebuildup = liq_cnt(conn, "ETHUSDT", "BUY", ts - 30 * 60_000, ts - 1000, PROP_THRESH)
    echo_45_120 = has_bucketed_cascade(conn, "ETHUSDT", "BUY", ts - ECHO_HI_MS, ts - ECHO_LO_MS, ETH_THRESH)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    ask_depth = float(book.get("ask_depth_usd") or 0.0) if book else None
    bid_depth = float(book.get("bid_depth_usd") or 0.0) if book else None
    imbalance = float(book.get("book_imbalance") or 0.0) if book else None
    spread_bps = float(book.get("spread_bps") or 0.0) if book else None
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    pos = {
        "id": bid,
        "signal": "BUY_FADE_SHORT_H45_SL75",
        "direction": "SHORT",
        "anchor_ts_ms": ts,
        "entry_ts_ms": ts,
        "entry_price": anchor_price,
        "exit_due_ms": ts + BUY_FADE_HOLD_MS,
        "status": "OPEN_BUY_FADE",
        "buy_state": "PENDING",
        "sil_check_ms": ts + SIL_HI_MS,
        "sl_bps": BUY_FADE_SL_BPS,
        "sync_k": sync_k,
        "n2h": n2h,
        "prebuildup": prebuildup,
        "echo_45_120": echo_45_120,
        "ask_depth_usd": ask_depth,
        "bid_depth_usd": bid_depth,
        "book_imbalance": imbalance,
        "spread_bps": spread_bps,
        "btc4h_bps": prior_bps(conn, "BTCUSDT", ts, 4 * 3600_000),
        "btc7d_bps": prior_bps(conn, "BTCUSDT", ts, BTC_7D_LOOKBACK_MS),
        "btc5m_bps": prior_bps(conn, "BTCUSDT", ts, 5 * 60_000),
        "eth1h_bps": prior_bps(conn, "ETHUSDT", ts, 3600_000),
        "eth4h_bps": prior_bps(conn, "ETHUSDT", ts, 4 * 3600_000),
        "dow": dt.weekday(),
        "session": session,
        "running_notional": running_notional,
        "rule_name": RULE_NAME,
        "opened_utc": utc_now(),
        "observer_note": "shadow_only_buy_liq_fade_short_h45_sl75",
    }
    init_profit_lock_observer(pos)
    state["processed"][bid] = utc_now()
    return [pos]


# ── Position management ────────────────────────────────────────────────────────

def advance_positions(conn, state: dict, now_ms: int) -> None:
    closed = []
    for pid, pos in list(state["positions"].items()):
        sig    = pos["signal"]
        status = pos["status"]
        ts     = int(pos["anchor_ts_ms"])

        # ── LONG_T15_BOUNCE monitoring ───────────────────────────────────────
        if sig == "LONG_T15_BOUNCE" and status == "MONITORING_T15_BOUNCE":
            due_ms = int(pos.get("entry_due_ms") or (ts + LONG_ENTRY_DELAY_MS))
            if now_ms >= due_ms and not pos.get("entry_ts_ms"):
                anchor_px = float(pos.get("anchor_price") or pos.get("entry_price") or 0.0)
                due_px = mark_at(conn, "ETHUSDT", due_ms)
                if anchor_px > 0 and due_px:
                    drift_bps = (float(due_px) - anchor_px) / anchor_px * 10_000.0
                    pos["t15_drift_bps"] = round(drift_bps, 2)
                    if drift_bps > 0:
                        pos["status"] = "OPEN"
                        pos["entry_ts_ms"] = due_ms
                        pos["entry_price"] = float(due_px)
                        pos["exit_due_ms"] = ts + HORIZON_LONG_MS
                        init_profit_lock_observer(pos)
                        log_event({**pos, "event": "OPEN"})
                        print(f"{utc_now()} [SHD] LONG_T15_BOUNCE OPEN drift={drift_bps:.1f}bps id={pid}")
                    else:
                        pos["status"] = "EXPIRED_T15_NO_BOUNCE"
                        pos["expired_utc"] = utc_now()
                        log_event({**pos, "event": "EXPIRE"})
                        closed.append(pid)
                        continue
            if now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
                pos["state_resolution"] = "SILENCE_CONFIRMED" if prop_ts is None else "NOISY_FOLLOW"
                if prop_ts is not None:
                    pos["eth_follow_ts_ms"] = prop_ts
                if pos.get("status") == "OPEN":
                    pos["status"] = "OPEN_SILENCE" if prop_ts is None else "OPEN_NOISY"
                    log_event({**pos, "event": pos["state_resolution"]})
                elif pos.get("status") == "MONITORING_T15_BOUNCE":
                    pos["status"] = "EXPIRED_T15_NO_DECISION"
                    closed.append(pid)

        # ── LONG_ECHO_45_120_SILENCE monitoring ─────────────────────────────
        elif sig == "LONG_ECHO_45_120_SILENCE" and status == "MONITORING_ECHO_SILENCE":
            prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + SIL_LO_MS, min(now_ms, ts + SIL_HI_MS), PROP_THRESH)
            if prop_ts is not None:
                pos["status"] = "EXPIRED_NOISY"
                pos["expired_utc"] = utc_now()
                pos["eth_follow_ts_ms"] = prop_ts
                log_event({**pos, "event": "EXPIRE"})
                closed.append(pid)
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "OPEN_SILENCE"
                pos["silence_confirmed_utc"] = utc_now()
                pos["score"] = pos.get("score", 0) + 1
                pos["exit_due_ms"] = ts + HORIZON_LONG_MS
                log_event({**pos, "event": "SILENCE_CONFIRMED"})
                print(f"{utc_now()} [SHD] LONG_ECHO_45_120_SILENCE confirmed score={pos['score']} id={pid}")

        # ── SHORT_NEITHER BTC monitoring ──────────────────────────────────────
        elif sig == "SHORT_NEITHER" and status == "MONITORING_BTC":
            btc_max = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_MIN_DELAY_MS, now_ms)
            if btc_max >= BTC_THRESH:
                px = mark_at(conn, "ETHUSDT", now_ms) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = now_ms
                pos["entry_price"] = px
                pos["exit_due_ms"] = now_ms + HORIZON_SHORT_MS
                pos["btc_max"] = btc_max
                init_profit_lock_observer(pos)
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_NEITHER OPEN id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "EXPIRED_NO_BTC"
                closed.append(pid)

        # ── SHORT_BTC1M_H4 BTC monitoring ────────────────────────────────────
        elif sig == "SHORT_BTC1M_H4" and status == "MONITORING_BTC_1M_H4":
            btc_max = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_MIN_DELAY_MS, now_ms)
            if btc_max >= BTC_SHADOW_1M_THRESH:
                px = mark_at(conn, "ETHUSDT", now_ms) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = now_ms
                pos["entry_price"] = px
                pos["exit_due_ms"] = now_ms + HORIZON_SHORT_H4_MS
                pos["btc_max"] = btc_max
                init_profit_lock_observer(pos)
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_BTC1M_H4 OPEN id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "EXPIRED_NO_BTC"
                closed.append(pid)

        # ── SHORT_NOISY propagation monitoring ────────────────────────────────
        # shadow-only: BTC>=1M after 10m, then 3h SHORT hold
        elif sig == "SHORT_BTC1M_D10_H3" and status == "MONITORING_BTC_1M_D10_H3":
            btc_max = liq_max(conn, "BTCUSDT", "SELL", ts + BTC_CONFIRM_D10_MS, now_ms)
            if btc_max >= BTC_SHADOW_1M_THRESH:
                px = mark_at(conn, "ETHUSDT", now_ms) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = now_ms
                pos["entry_price"] = px
                pos["exit_due_ms"] = now_ms + HORIZON_SHORT_H3_MS
                pos["btc_max"] = btc_max
                init_profit_lock_observer(pos)
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_BTC1M_D10_H3 OPEN id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "EXPIRED_NO_BTC"
                closed.append(pid)

        # shadow-only: ETH noisy propagation + BTC>=1M after 5m, then 180m SHORT hold
        elif sig == "SHORT_NOISY_BTC1M_D5_H180" and status == "MONITORING_SHORT_NOISY_BTC1M_D5_H180":
            prop_ts = liq_first_ts(
                conn, "ETHUSDT", "SELL",
                ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH,
            )
            btc_ts = liq_first_ts(
                conn, "BTCUSDT", "SELL",
                ts + BTC_CONFIRM_MIN_DELAY_MS, min(now_ms, ts + NOISY_WIN_HI_MS), BTC_SHADOW_1M_THRESH,
            )
            if prop_ts is not None:
                pos["prop_ts_ms"] = prop_ts
                pos["prop_delay_ms"] = prop_ts - ts
            if btc_ts is not None:
                pos["btc_confirm_ts_ms"] = btc_ts
                pos["btc_delay_ms"] = btc_ts - ts
            if prop_ts is not None and btc_ts is not None:
                entry_ts = max(int(prop_ts), int(btc_ts))
                px = mark_at(conn, "ETHUSDT", entry_ts) or mark_at(conn, "ETHUSDT", now_ms) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = entry_ts
                pos["entry_price"] = px
                pos["exit_due_ms"] = entry_ts + HORIZON_SHORT_H3_MS
                init_profit_lock_observer(pos)
                log_event({**pos, "event": "OPEN"})
                print(
                    f"{utc_now()} [SHD] SHORT_NOISY_BTC1M_D5_H180 OPEN "
                    f"prop_delay={prop_ts-ts:.0f}ms btc_delay={btc_ts-ts:.0f}ms id={pid}"
                )
            elif now_ms >= int(pos.get("sil_check_ms", ts + NOISY_WIN_HI_MS)):
                if prop_ts is None and btc_ts is None:
                    pos["status"] = "EXPIRED_NO_PROP_NO_BTC"
                elif prop_ts is None:
                    pos["status"] = "EXPIRED_NO_PROP"
                else:
                    pos["status"] = "EXPIRED_NO_BTC"
                pos["expired_utc"] = utc_now()
                log_event({**pos, "event": "EXPIRE"})
                closed.append(pid)

        # shadow-only: prebuild>=2 / double-cascade route, counted only after silence
        elif sig == "LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE" and status == "MONITORING_PREBUILD_SILENCE":
            prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + SIL_LO_MS, min(now_ms, ts + SIL_HI_MS), PROP_THRESH)
            if prop_ts is not None:
                pos["status"] = "EXPIRED_NOISY"
                pos["expired_utc"] = utc_now()
                pos["eth_follow_ts_ms"] = prop_ts
                log_event({**pos, "event": "EXPIRE"})
                closed.append(pid)
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "OPEN_SILENCE"
                pos["silence_confirmed_utc"] = utc_now()
                pos["score"] = pos.get("score", 0) + 1
                pos["exit_due_ms"] = ts + HORIZON_LONG_MS
                log_event({**pos, "event": "SILENCE_CONFIRMED"})
                print(f"{utc_now()} [SHD] LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE confirmed score={pos['score']} id={pid}")

        # shadow-only: silence + first 15m taker-buy OFI
        elif sig == "LONG_OFI_SILENCE_BUYERS" and status == "MONITORING_OFI_SILENCE":
            prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + SIL_LO_MS, min(now_ms, ts + SIL_HI_MS), PROP_THRESH)
            if prop_ts is not None:
                pos["status"] = "EXPIRED_NOISY"
                pos["expired_utc"] = utc_now()
                pos["eth_follow_ts_ms"] = prop_ts
                log_event({**pos, "event": "EXPIRE"})
                closed.append(pid)
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                ofi = agg_trade_ofi(conn, "ETHUSDT", ts, ts + OFI_WIN_MS)
                pos["ofi_buy_notional"] = ofi["buy_notional"]
                pos["ofi_sell_notional"] = ofi["sell_notional"]
                pos["ofi_notional"] = ofi["ofi_notional"]
                if ofi["ofi_notional"] > 0:
                    pos["status"] = "OPEN_SILENCE"
                    pos["silence_confirmed_utc"] = utc_now()
                    pos["score"] = pos.get("score", 0) + 1
                    pos["exit_due_ms"] = ts + HORIZON_LONG_MS
                    log_event({**pos, "event": "SILENCE_CONFIRMED"})
                    print(f"{utc_now()} [SHD] LONG_OFI_SILENCE_BUYERS confirmed ofi={ofi['ofi_notional']:.0f} id={pid}")
                else:
                    pos["status"] = "EXPIRED_OFI_NOT_BUYERS"
                    pos["expired_utc"] = utc_now()
                    log_event({**pos, "event": "EXPIRE"})
                    closed.append(pid)

        elif sig == "SHORT_NOISY" and status == "MONITORING_PROP":
            prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                                   ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH)
            if prop_ts is not None:
                px = mark_at(conn, "ETHUSDT", prop_ts) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = prop_ts
                pos["entry_price"] = px
                pos["exit_due_ms"] = prop_ts + HORIZON_SHORT_MS
                pos["prop_delay_ms"] = prop_ts - ts
                init_profit_lock_observer(pos)
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_NOISY OPEN delay={prop_ts-ts:.0f}ms id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + NOISY_WIN_HI_MS)):
                # Window passed, no propagation → it's a silence event, not noisy
                pos["status"] = "EXPIRED_SILENCE"
                closed.append(pid)

        # ── Time exit for all OPEN positions ──────────────────────────────────
        if sig == "BUY_FADE_SHORT_H45_SL75" and status == "OPEN_BUY_FADE":
            if pos.get("buy_state") == "PENDING":
                prop_ts = liq_first_ts(conn, "ETHUSDT", "BUY", ts + SIL_LO_MS, min(now_ms, ts + SIL_HI_MS), PROP_THRESH)
                if prop_ts is not None:
                    pos["buy_state"] = "NOISY"
                    pos["eth_buy_follow_ts_ms"] = prop_ts
                    log_event({**pos, "event": "BUY_FADE_STATE_NOISY"})
                elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                    pos["buy_state"] = "SILENCE"
                    pos["silence_confirmed_utc"] = utc_now()
                    log_event({**pos, "event": "BUY_FADE_STATE_SILENCE"})
                    print(f"{utc_now()} [SHD] BUY_FADE_SHORT_H45_SL75 SILENCE id={pid}")

            # OD-004: bd_first_buy50 observation-only exit gozlemcisi (delta loglama, siparis yok)
            bobs = pos.get("bd_first_buy50_observer")
            if bobs is None:
                bobs = {
                    "protocol": "BD_FIRST_BUY50_EXIT_OBSERVER",
                    "protocol_version": BD_FIRST_BUY50_PROTOCOL_VERSION,
                    "route": sig,
                    "mode": "OBSERVE_ONLY_NO_ORDER",
                    "activation_utc": BD_FIRST_BUY50_ACTIVATION_UTC,
                    "start_after_ms": BD_FIRST_BUY50_START_MS,
                    "thresh_usd": BD_FIRST_BUY50_THRESH_USD,
                    "triggered": False,
                    "hypothetical_exit_ts_ms": None,
                    "hypothetical_exit_price": None,
                    "hypothetical_exit_net_bps": None,
                    "detected_at_ts_ms": None,
                    "reconstructed_after_restart": None,
                    "baseline_exit_ts_ms": None,
                    "baseline_net_bps": None,
                    "delta_vs_baseline_bps": None,
                }
                pos["bd_first_buy50_observer"] = bobs
            _bets = int(pos.get("entry_ts_ms") or 0)
            if _bets and not bobs.get("triggered"):
                # OD-004-FOLLOWUP: enforced lower bound = later of (entry+30m,
                # activation). Never selects an event before either boundary,
                # regardless of when this tick happens to run (restart-safe).
                _search_lower_ms = max(_bets + BD_FIRST_BUY50_START_MS, BD_FIRST_BUY50_ACTIVATION_MS)
                if now_ms >= _search_lower_ms:
                    _bt = liq_first_ts(conn, "ETHUSDT", "BUY", _search_lower_ms, now_ms, BD_FIRST_BUY50_THRESH_USD)
                    if _bt is not None:
                        _bpx = mark_at(conn, "ETHUSDT", int(_bt))
                        _bentry = float(pos.get("entry_price") or 0.0)
                        if _bpx and _bentry > 0:
                            _bmove = (float(_bpx) - _bentry) / _bentry * 10_000.0
                            _bnet = (-_bmove if pos.get("direction") == "SHORT" else _bmove) - FEE_BPS
                            bobs["triggered"] = True
                            bobs["hypothetical_exit_ts_ms"] = int(_bt)
                            bobs["hypothetical_exit_price"] = float(_bpx)
                            bobs["hypothetical_exit_net_bps"] = round(_bnet, 2)
                            bobs["detected_at_ts_ms"] = int(now_ms)
                            bobs["reconstructed_after_restart"] = bool(int(_bt) < _PROCESS_START_MS)
                            log_event({**pos, "event": "BD_FIRST_BUY50_OBS_EXIT"})

            entry_px = float(pos.get("entry_price") or 0.0)
            cur_px = mark_at(conn, "ETHUSDT", now_ms)
            if entry_px > 0 and cur_px:
                adverse_bps = (float(cur_px) - entry_px) / entry_px * 10_000.0
                pos["last_adverse_bps"] = round(adverse_bps, 2)
                if adverse_bps >= float(pos.get("sl_bps") or BUY_FADE_SL_BPS):
                    _close_pos(pos, conn, now_ms, "BUY_FADE_SL75")
                    log_event({**pos, "event": "CLOSE"})
                    closed.append(pid)
                    print(f"{utc_now()} [SHD] BUY_FADE_SHORT_H45_SL75 SL75 net={pos.get('net_bps','?'):.1f}bps id={pid}")

        if pos.get("status") in {"OPEN", "OPEN_SILENCE", "OPEN_NOISY", "OPEN_BUY_FADE"}:
            update_profit_lock_observer(pos, conn, now_ms)
            update_scalein_observer(pos, conn, now_ms)
            exit_due = pos.get("exit_due_ms")
            if exit_due and now_ms >= int(exit_due):
                _close_pos(pos, conn, now_ms, "TIME_EXIT")
                log_event({**pos, "event": "CLOSE"})
                closed.append(pid)
                print(f"{utc_now()} [SHD] {sig} TIME_EXIT net={pos.get('net_bps','?'):.1f}bps id={pid}")

    for pid in closed:
        state["positions"].pop(pid, None)


def _close_pos(pos: dict, conn, now_ms: int, reason: str) -> None:
    entry_px = float(pos.get("entry_price") or 0)
    exit_px  = mark_at(conn, "ETHUSDT", now_ms) or entry_px
    if entry_px > 0:
        raw_move = (exit_px - entry_px) / entry_px * 10_000.0
        outcome  = -raw_move if pos["direction"] == "SHORT" else raw_move
    else:
        outcome = 0.0
    anch_ts = pos.get("anchor_ts_ms")
    if anch_ts:
        px_anch = mark_at(conn, "ETHUSDT", anch_ts)
        px_5m   = mark_at(conn, "ETHUSDT", anch_ts + 5 * 60_000)
        pos["failed_cascade"] = bool(px_anch and px_5m and px_5m > px_anch)
    else:
        pos["failed_cascade"] = None
    pos["status"]      = f"CLOSED_{reason}"
    pos["close_reason"]= reason
    pos["exit_price"]  = exit_px
    pos["exit_ts_ms"]  = now_ms
    pos["closed_utc"]  = utc_now()
    pos["outcome_bps"] = round(outcome, 2)
    pos["net_bps"]     = round(outcome - FEE_BPS, 2)
    # MECH refill: CLOSE aninda entry+5m penceresi artik gecmiste — full skor tamamlanir
    mech = pos.get("mech")
    ets = pos.get("entry_ts_ms")
    if mech and ets and mech.get("refill") is None:
        pre = conn.execute("SELECT AVG(bid_qty) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (int(ets) - 600_000, int(ets))).fetchone()
        post = conn.execute("SELECT AVG(bid_qty) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (int(ets), int(ets) + 300_000)).fetchone()
        if pre and pre[0] and post and post[0] is not None and float(pre[0]) > 0:
            mech["refill"] = float(post[0]) / float(pre[0])
            mech["refill_hi"] = mech["refill"] >= MECH_THR["refill"]
            mech["mech_score_full"] = mech["mech_score_pre"] + (1 if mech["refill_hi"] else 0)
        else:
            mech["missing"] = list(set(mech.get("missing", []) + ["bk_refill"]))
    # OD-004 bd_first_buy50 gozlemcisi: kapaniste baseline'a gore delta
    bobs = pos.get("bd_first_buy50_observer")
    if bobs and bobs.get("triggered") and bobs.get("hypothetical_exit_net_bps") is not None:
        bobs["baseline_exit_ts_ms"] = pos.get("exit_ts_ms")
        bobs["baseline_net_bps"] = pos.get("net_bps")
        bobs["delta_vs_baseline_bps"] = round(float(bobs["hypothetical_exit_net_bps"]) - float(pos["net_bps"]), 2)
    # SCALEIN gozlemcisi: eklenmisse birlesik per-unit (2 birim, her biri fee oder)
    sobs = pos.get("scalein_observer")
    if sobs and sobs.get("added") and sobs.get("add_pnl_bps") is not None:
        add_pnl = float(sobs["add_pnl_bps"])
        unit2_net = (outcome - add_pnl) - FEE_BPS
        sobs["combined_per_unit_net_bps"] = round((pos["net_bps"] + unit2_net) / 2.0, 2)
        sobs["delta_vs_baseline_bps"] = round(sobs["combined_per_unit_net_bps"] - pos["net_bps"], 2)


# ── Stats ─────────────────────────────────────────────────────────────────────

def refresh_pnl(state: dict) -> None:
    by_sig: dict[str, list[float]] = {}
    profit_lock_shadow: list[float] = []
    profit_lock_delta: list[float] = []
    scalein_combined: list[float] = []
    scalein_delta: list[float] = []
    scalein_noadd: list[float] = []
    if not LEDGER_PATH.exists():
        state["profit_lock_observer"] = {
            "protocol": "PROFIT_LOCK_100_50_OBSERVER",
            "mode": "OBSERVE_ONLY_NO_ORDER",
            "trigger_bps": PROFIT_LOCK_TRIGGER_BPS,
            "lock_bps": PROFIT_LOCK_LOCK_BPS,
            "shadow_exit_n": 0,
            "shadow_exit_sum_bps": 0.0,
            "shadow_exit_avg_bps": None,
            "delta_vs_baseline_n": 0,
            "delta_vs_baseline_sum_bps": 0.0,
            "delta_vs_baseline_avg_bps": None,
        }
        return
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "PROFIT_LOCK_SHADOW_EXIT":
                obs = r.get("profit_lock_observer") or {}
                net = obs.get("shadow_exit_net_bps")
                if net is not None:
                    profit_lock_shadow.append(float(net))
            if r.get("event") == "CLOSE":
                sig = r.get("signal", "?")
                net = r.get("net_bps")
                if net is not None:
                    by_sig.setdefault(sig, []).append(float(net))
                obs = r.get("profit_lock_observer") or {}
                shadow_net = obs.get("shadow_exit_net_bps")
                if net is not None and shadow_net is not None:
                    profit_lock_delta.append(float(shadow_net) - float(net))
                sobs = r.get("scalein_observer") or {}
                if sobs.get("combined_per_unit_net_bps") is not None:
                    scalein_combined.append(float(sobs["combined_per_unit_net_bps"]))
                    if net is not None:
                        scalein_delta.append(float(sobs["combined_per_unit_net_bps"]) - float(net))
                elif net is not None and r.get("signal") in SCALEIN_ROUTES:
                    scalein_noadd.append(float(net))
    pnl: dict[str, Any] = {}
    for sig, vals in by_sig.items():
        wins = sum(1 for v in vals if v > 0)
        pnl[sig] = {
            "n": len(vals),
            "wins": wins,
            "wr": round(wins / len(vals), 3) if vals else 0,
            "total_net": round(sum(vals), 1),
            "avg_net": round(sum(vals) / len(vals), 1) if vals else 0,
        }
    pnl["updated_utc"] = utc_now()
    state["pnl"] = pnl
    state["profit_lock_observer"] = {
        "protocol": "PROFIT_LOCK_200_100_OBSERVER",
        "mode": "OBSERVE_ONLY_NO_ORDER",
        "trigger_bps": PROFIT_LOCK_TRIGGER_BPS,
        "lock_bps": PROFIT_LOCK_LOCK_BPS,
        "shadow_exit_n": len(profit_lock_shadow),
        "shadow_exit_sum_bps": round(sum(profit_lock_shadow), 1),
        "shadow_exit_avg_bps": round(sum(profit_lock_shadow) / len(profit_lock_shadow), 1) if profit_lock_shadow else None,
        "delta_vs_baseline_n": len(profit_lock_delta),
        "delta_vs_baseline_sum_bps": round(sum(profit_lock_delta), 1),
        "delta_vs_baseline_avg_bps": round(sum(profit_lock_delta) / len(profit_lock_delta), 1) if profit_lock_delta else None,
    }
    state["scalein_observer"] = {
        "protocol": "SCALEIN_DIP100_2H_OBSERVER",
        "mode": "OBSERVE_ONLY_NO_ORDER",
        "dip_bps": SCALEIN_DIP_BPS,
        "added_n": len(scalein_combined),
        "no_add_n": len(scalein_noadd),
        "combined_per_unit_avg_bps": round(sum(scalein_combined) / len(scalein_combined), 1) if scalein_combined else None,
        "delta_vs_baseline_avg_bps": round(sum(scalein_delta) / len(scalein_delta), 1) if scalein_delta else None,
        "delta_vs_baseline_sum_bps": round(sum(scalein_delta), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def open_100k_composite_for_anchor(conn, ts: int, anchor_price: float, rn: float, now_ms: int, state: dict) -> list[dict]:
    """100K frequency-expansion composite route (HORIZON holdout-validated, no-overlap 11.8/mo WR66).
    Mini cascade 100-200K + hour17 + regime + composite score>=3 (no funding veto). LONG T0, 6h hold."""
    if not (100_000.0 <= rn < ETH_THRESH):
        return []
    id100 = f"SHD:{ts}:C100K"
    if id100 in state["processed"]:
        return []
    hour = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
    if hour < HOUR17_MIN_HOUR or session_name(ts) == "EUROPE" or is_bull_pullback(conn, ts):
        return []
    btc4h = prior_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    btc7d = prior_bps(conn, "BTCUSDT", ts, BTC_7D_LOOKBACK_MS)
    if not (btc4h < 0.0 or btc7d < 0.0):
        return []
    sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts)
              + liq_sum(conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts))
    density24 = liq_cnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, ETH_THRESH)
    comp_score, comp = compute_composite_score(conn, ts, sync_k, rn, density24)
    if comp_score < 3 or comp.get("funding_veto"):
        return []
    pos = {
        "id": id100, "signal": "LONG_HOUR17_100K_COMPOSITE", "direction": "LONG",
        "mech": compute_mech_features(conn, ts, rn),
        "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_price,
        "exit_due_ms": ts + HORIZON_LONG_H6_MS, "status": "OPEN",
        "conviction_score": comp_score, "conviction": comp,
        "sync_k": sync_k, "density_24h": density24, "hour": hour,
        "btc7d_bps": btc7d, "btc4h_bps": btc4h, "running_notional": rn,
        "rule_name": RULE_NAME, "opened_utc": utc_now(),
        "observer_note": "freq_expansion_100k_composite_ge3_no_early_exit_no_stop",
    }
    init_profit_lock_observer(pos)
    log_event({**pos, "event": "OPEN"})
    state["processed"][id100] = utc_now()
    return [pos]


def run_once(db_path: Path, state: dict) -> None:
    now_ms = int(time.time() * 1000)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        # 1. Advance existing positions
        advance_positions(conn, state, now_ms)

        # 2. Detect fresh anchors
        start_ms = now_ms - LOOKBACK_SEC * 1000
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        anchors = reconstruct_anchors(
            liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
            thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_WIN_SEC,
        )
        marks = load_mark_index(conn, "ETHUSDT")
        for anchor in anchors:
            age_ms = now_ms - int(anchor.anchor_ts_ms)
            if not (0 <= age_ms <= SIGNAL_FRESH_MS):
                continue
            mark = marks.at_or_after(int(anchor.anchor_ts_ms))
            if not mark:
                continue
            anchor_price = float(mark[1])
            new_pos = open_signals_for_anchor(
                conn, int(anchor.anchor_ts_ms), anchor_price,
                float(anchor.running_notional), now_ms, state,
            )
            for pos in new_pos:
                state["positions"][pos["id"]] = pos
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] OPENED {pos['signal']} status={pos['status']} id={pos['id']}")

        # 2b. 100K frequency-expansion composite route (separate mini-anchor stream)
        anchors_100k = reconstruct_anchors(
            liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
            thresholds=(100_000.0,), accel_window_sec=ACCEL_WIN_SEC,
        )
        for anchor in anchors_100k:
            age_ms = now_ms - int(anchor.anchor_ts_ms)
            if not (0 <= age_ms <= SIGNAL_FRESH_MS):
                continue
            mark = marks.at_or_after(int(anchor.anchor_ts_ms))
            if not mark:
                continue
            new_pos = open_100k_composite_for_anchor(
                conn, int(anchor.anchor_ts_ms), float(mark[1]),
                float(anchor.running_notional), now_ms, state,
            )
            for pos in new_pos:
                state["positions"][pos["id"]] = pos
                print(f"{utc_now()} [SHD] OPENED {pos['signal']} status={pos['status']} id={pos['id']}")

        buy_liqs = load_liquidations(conn, "ETHUSDT", "BUY", start_ms, now_ms)
        buy_anchors = reconstruct_anchors(
            buy_liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
            thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_WIN_SEC,
        )
        for anchor in buy_anchors:
            age_ms = now_ms - int(anchor.anchor_ts_ms)
            if not (0 <= age_ms <= SIGNAL_FRESH_MS):
                continue
            mark = marks.at_or_after(int(anchor.anchor_ts_ms))
            if not mark:
                continue
            anchor_price = float(mark[1])
            new_pos = open_buy_fade_for_anchor(
                conn, int(anchor.anchor_ts_ms), anchor_price,
                float(anchor.running_notional), now_ms, state,
            )
            for pos in new_pos:
                state["positions"][pos["id"]] = pos
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] OPENED {pos['signal']} status={pos['status']} id={pos['id']}")

    state["rule_name"] = RULE_NAME
    state["mode"] = "OBSERVE_ONLY_NO_ORDER"
    state["updated_utc"] = utc_now()

    # 3. Refresh PnL stats
    refresh_pnl(state)
    save_state(state)


def print_status(state: dict) -> None:
    open_pos = {k: v for k, v in state["positions"].items()
                if v.get("status") not in {"EXPIRED_NO_BTC", "EXPIRED_SILENCE"}}
    print(f"\n=== Shadow Paper Status  {utc_now()} ===")
    print(f"Open positions: {len(open_pos)}")
    for pid, pos in open_pos.items():
        print(f"  {pos['signal']:20s} {pos['status']:20s}  score={pos.get('score',0)}"
              f"  sync_k={pos.get('sync_k',0):.0f}")
    pnl = state.get("pnl", {})
    if pnl:
        print("\nSignal P&L:")
        for sig, d in pnl.items():
            if sig == "updated_utc":
                continue
            print(f"  {sig:20s}  N={d['n']:4d}  WR={d['wr']:.1%}  total={d['total_net']:+.0f}bps"
                  f"  avg={d['avg_net']:+.1f}bps")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 Real-Time Shadow Runner")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=POLL_SEC)
    args = parser.parse_args()

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    state = load_state()
    print(f"{utc_now()} S34 State Machine Shadow Runner started  db={args.db}")
    print(f"  Rule   : {RULE_NAME}")
    print(f"  Signals: LONG_T15_BOUNCE, SHORT_NEITHER")
    print(f"  Ledger : {LEDGER_PATH}")
    print(f"  State  : {STATE_PATH}")

    if args.once:
        run_once(args.db, state)
        print_status(state)
        return 0

    while True:
        try:
            run_once(args.db, state)
        except Exception as exc:  # noqa: BLE001
            print(f"{utc_now()} ERROR: {type(exc).__name__}: {exc}")
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
