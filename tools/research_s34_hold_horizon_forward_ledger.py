"""research_s34_hold_horizon_forward_ledger.py — FORWARD paper ledger, per hold horizon (READ-ONLY).

Records, for every strictly-forward ETH SELL anchor (after 2026-07-20) that qualifies under the
hour17 and/or echo(causal) T0 gates, the paper outcome at each hold horizon 2/4/6/12/24/48h — an
OPEN when the anchor fires, then a RESOLVE per (signal, horizon) as each hold matures (up to 48h).

WHY: the in-sample sweep (research_s34_hold_horizon_sweep) is on the BURNED ~5mo sample — its rising
avg is necessary-not-sufficient. THIS ledger is the honest forward test of the hold-response curve,
never re-mined. NOT a trader, no orders. DB read-only (mode=ro, query_only=1); FEE=5bps baked in.

Ledger: reports/shadow/hold_horizon_forward_ledger.jsonl   State: ...hold_horizon_forward_ledger_state.json
Loop: `--once` or persistent via start_eclipse.ps1 (role hold_horizon_forward_ledger).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the echo forward-ledger primitives verbatim so anchor reconstruction / marks / gates
# reconcile exactly with the echo ledger and the sweep.
from tools.research_s34_echo_forward_ledger import (  # noqa: E402
    _mark_at, _mark_bps, _echo_check, _session, _detect_fresh_anchors, _min_mark,
)
from ami.storage.union_reader import open_union_ro, open_live_ro, RotationStateError  # noqa: E402

DB_URI = f"file:{ROOT / 'data' / 'microstructure.db'}?mode=ro"
LEDGER = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger.jsonl"
STATE = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger_state.json"

HORIZONS_H = [2, 4, 6, 12, 24, 48]
# Swing OBSERVATION-ONLY horizons (mark-checkpoint drift; NOT the <=48h envelope / exit-sweep). Each is
# a single cheap _mark_at/_book_at point-lookup at anchor+h; the anchor persists in a lightweight
# swing_pending (state, reset-survive) until 365d matures. EXCLUDED from the exit-sweep FWER family
# (observation only; no verdict — N=20 would take years). Dense envelope over these is infeasible
# (1yr ~ 8.7B book_ticker rows/anchor) and DB-rotation prunes old book anyway.
SWING_HORIZONS_H = [168, 720, 4320, 8760]   # 7d / 30d / 180d / 365d (hours)
FEE_BPS = 5.0                  # legacy mark-based fee assumption (kept only for reconciliation net)
CUTOFF_MS = 1784505600000  # 2026-07-20T00:00:00Z — strictly forward only (discovery burned before)

# --- Measured-cost forward (V2, operator-confirmed 2026-07-20; design S34_ECHO_FORWARD_EVALUATOR_DESIGN_V1) ---
COMMISSION_BPS = 5.0           # round-trip taker commission ONLY; spread is now measured separately (no double count)
QUOTE_STALE_MS = 60_000        # entry/exit book_ticker older than this => stale quote (outage guard, §166)
WINDOW_GAP_MS = 300_000        # any book_ticker gap > 5min inside the hold => path/stop untrustworthy
RESOLVE_BUDGET_SEC = 40.0      # per-pass wall-clock cap on RESOLVE work: OPENs are time-critical
                               #   (FRESH_MS window) but matured RESOLVEs are not — defer the rest a pass
                               #   so a slow long-horizon scan can never delay fresh-anchor detection
NOTIONAL_USD = 25.0            # read-only mirror of config/settings.py FIXED_NOTIONAL_USDT (live sizing UNCHANGED);
                               #   used ONLY for the top-of-book depth honesty flag, never for orders

# ── FROZEN give-back / trailing-exit FORWARD-LOGGING spec (S34_HH_GIVEBACK_FORWARD_LOG_V1) ──────
# Purpose: persist the CAUSAL running-max-bid envelope so a give-back / trailing exit can later be
# OFF-POLICY replayed on FORWARD data — this LOGS the data, it does NOT select or apply a rule now.
# Frozen BEFORE data accumulates so the eventual Z-sweep can't be fit to the first noisy paths.
#   FORM     : LONG exit when (run_max_bid - bid)/run_max_bid*1e4 >= Z   (trailing from running peak)
#   Z        : a-priori only (below); swept later off-policy on QUALIFIED paths, family-corrected
#   FILL     : taker sell at the bid AT the trigger ts, +commission, NO gap-through => OPTIMISTIC
#              (mirrors _stop_meas; every downstream report must carry the "optimistic-fill" caveat)
#   BASIS    : give-back = bps-of-running-max-bid ; realized-net = bps-of-entry(ask). RAW prices are
#              logged so either convention is re-derivable; only the TRIGGER convention is frozen.
#   METRIC   : menu {baseline horizon, bracket TP/SL, trailing give-back} => deflated/Bonferroni; frozen
#   REPLAY   : only QUALIFIED paths (q_h17 / q_echo); control kept separate, never swept.
#   ENVELOPE : THREE monotone candidate-exit families make the frozen menu off-policy replayable for
#              ANY threshold — kind="high" (run_max_bid up-crossings => TP first-cross; 1bps step-floor,
#              MFE curve), kind="dd" (max drawdown-from-ENTRY => SL first-cross; 1bps-quantized), kind="gb"
#              (max give-back-from-PEAK => trailing-Z first-cross; 1bps-quantized). high/gb reference the
#              true running-max rmb (advances every tick, only the high RECORD is throttled); dd references
#              the fixed ask_entry. Together: baseline horizon (RESOLVE) + bracket TP/SL (high & dd) +
#              trailing Z (gb). All 1bps-quantized => crossing-location error <=1bps, finer than any Z/TP/SL.
#              SL replay is OPTIMISTIC (fill AT the dd level, no gap-through — same caveat as _stop_meas).
#              (operator sign-off 2026-07-22 — recorded frozen-spec change, not a silent sparsification.)
#   DIRECTION: SIX envelopes = LONG (high/dd/gb on bid, entry@ask) + SHORT MIRROR (short_low/short_dd/
#              short_gb on ask, entry@bid, favorable=ask-falling). SHORT = a pre-registered SYMMETRIC arm
#              (operator sign-off 2026-07-22), measured-cost honest (NOT a sign-flip). Both directions are
#              counted in the FWER family => 2x the multiple-testing burden (CONFIRM harder — honest).
# The envelope is BID-authoritative (give-back must reference the exitable side; mark peaks above any
# reachable bid — e.g. hr17 saw +131 mark vs +31 bid). run_max_mark is kept only for reconciliation +
# the mark<->bid peak-gap metric (itself a microstructure signal: how far mark overshoots the bid on a
# liquidation spike). CAUSAL: every envelope record's ts is the REAL book_ticker ts of the extremum,
# NOT the scan-pass ts — the give-back trigger must be knowable when the bid actually took that value,
# not "noticed 30s later" (knowable-anchor discipline extended to the exit path).
GIVEBACK_Z_BPS_A_PRIORI = 60.0
ENV_SCAN_CATCHUP_CAP_MS = 30 * 60_000   # per-pass envelope scan window cap (defer rest -> stays in budget)
ENV_MIN_GB_STEP_BPS = 1.0               # emit a new give-back record only when it advances >= this (compact;
                                        #   1bps-quantized => Z-sweep crossing-location error <=1bps, finer
                                        #   than any plausible Z)
ENV_MIN_HIGH_STEP_BPS = 1.0             # throttle ONLY the high RECORD to 1bps steps (MFE curve/dashboard);
                                        #   rmb (true running max) still updates every tick, gb envelope 1bps-quantized
ENV_MIN_DD_STEP_BPS = 1.0               # drawdown-from-ENTRY (SL) envelope: emit a new record only when the
                                        #   max drawdown advances >= this; 1bps-quantized (SL-cross error <=1bps)
ENV_BUDGET_SEC = 20.0                   # wall-clock cap on envelope work per pass (forward-logging is NOT
                                        #   time-critical; OPEN/RESOLVE run first, envelope gets what's left)


def _book_at(cur, ts, sym="ETHUSDT"):
    """Last top-of-book quote at-or-before ts (causal, no lookahead). None if none exists."""
    r = cur.execute(
        "SELECT ts_ms, bid_price, ask_price, mid_price, bid_depth_usd "
        "FROM book_ticker WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts)).fetchone()
    if not r or r[1] is None or r[2] is None:
        return None
    qts, bid, ask = int(r[0]), float(r[1]), float(r[2])
    mid = float(r[3]) if r[3] is not None else (bid + ask) / 2.0
    depth = float(r[4]) if r[4] is not None else None
    spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 1e4 if (ask + bid) > 0 else None
    return {"quote_ts": qts, "age_ms": ts - qts, "bid": bid, "ask": ask,
            "mid": mid, "spread_bps": spread_bps, "bid_depth_usd": depth}


def _min_bid(cur, lo, hi, sym="ETHUSDT"):
    """Lowest bid over [lo,hi] — measured-cost stop path (LONG stop fills at bid)."""
    r = cur.execute("SELECT MIN(bid_price) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
                    (sym, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None


def _book_max(cur, sym="ETHUSDT"):
    """Latest book_ticker ts (the book-feed clock). Indexed MAX, cheap."""
    r = cur.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol=?", (sym,)).fetchone()
    return int(r[0]) if r and r[0] else None


def _window_has_gap(cur, lo, hi, thr_ms, sym="ETHUSDT"):
    """True if any book_ticker gap > thr_ms exists in [lo,hi]. CHEAP probe (indexed point lookups
    every thr_ms) instead of a full-range LAG scan over up to 48h of a 114M-row table: any gap wider
    than thr_ms necessarily contains a probe point, so it is detected; short-circuits on first gap."""
    p = lo
    while p <= hi:
        last = cur.execute("SELECT ts_ms FROM book_ticker WHERE symbol=? AND ts_ms<=? "
                           "ORDER BY ts_ms DESC LIMIT 1", (sym, p)).fetchone()
        nxt = cur.execute("SELECT ts_ms FROM book_ticker WHERE symbol=? AND ts_ms>=? "
                          "ORDER BY ts_ms ASC LIMIT 1", (sym, p)).fetchone()
        lo_ts = int(last[0]) if last and last[0] is not None else None
        hi_ts = int(nxt[0]) if nxt and nxt[0] is not None else None
        if lo_ts is not None and hi_ts is not None and (hi_ts - lo_ts) > thr_ms:
            return True
        p += thr_ms
    return False


def _max_bid(cur, lo, hi, sym="ETHUSDT"):
    """Highest bid over [lo,hi] — measured-cost MFE path (LONG exit fills at bid)."""
    r = cur.execute("SELECT MAX(bid_price) FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
                    (sym, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None


def _max_mark(cur, lo, hi, sym="ETHUSDT"):
    """Highest mark over [lo,hi] — reconciliation MFE (mark overshoots the reachable bid on spikes)."""
    r = cur.execute("SELECT MAX(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
                    (sym, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None


def _scan_envelope(cur, ats, info, book_max, sym="ETHUSDT"):
    """Incrementally extend the CAUSAL exit-candidate envelopes for one open anchor, BOTH directions.

    Walks ONLY the new book_ticker slice (env_scan_ts, scan_end] each pass (scan_end capped by
    ENV_SCAN_CATCHUP_CAP_MS so a long catch-up never blows the budget), reading bid AND ask, updating
    running state IN PLACE on `info` and returning new envelope records. SIX monotone envelopes are
    emitted (3 per direction) so the whole frozen exit menu is off-policy replayable for ANY threshold,
    per direction:
      LONG (entry@ask / exit@bid, favorable = bid rising):
      * kind="high" — new running-max bid (MFE up-envelope; TP first-cross T)
      * kind="dd"   — new max drawdown-from-ENTRY on bid (SL first-cross S; refs ask_entry)
      * kind="gb"   — new max give-back from running peak (trailing-Z down-leg)
      SHORT mirror (entry@bid / exit@ask, favorable = ask FALLING):
      * kind="short_low" — new running-MIN ask (short-MFE; short-TP first-cross T)
      * kind="short_dd"  — new max ask-RISE from ENTRY (short-SL first-cross S; refs bid_entry)
      * kind="short_gb"  — new max give-back (ask rising) from the running low (short-trailing-Z)
    high+dd = long bracket TP/SL; short_low+short_dd = short bracket; gb/short_gb = trailing stops.
    Each record's ts is the REAL book_ticker ts of the extremum (causal). A book gap > WINDOW_GAP_MS
    inside the scanned span sets a STICKY env_quar flag: the whole anchor's envelope is then
    contaminated (unreliable peak from a broken feed) and every record carries quarantined=True so a
    consumer drops the anchor's envelope from the sweep — mirroring how a quarantined RESOLVE zeroes
    net_bps. Returns [] when there is nothing new to scan.
    """
    pos_end = ats + max(HORIZONS_H) * 3600_000                     # envelope lives over the full 48h hold
    scan_from = int(info.get("env_scan_ts", ats))
    scan_end = min(int(book_max), pos_end, scan_from + ENV_SCAN_CATCHUP_CAP_MS)
    if scan_end <= scan_from:
        return []
    rows = cur.execute("SELECT ts_ms, bid_price, ask_price FROM book_ticker WHERE symbol=? AND ts_ms>? "
                       "AND ts_ms<=? ORDER BY ts_ms ASC", (sym, scan_from, scan_end)).fetchall()
    # ---- LONG bid-side state ----
    rmb = info.get("run_max_bid")
    rmb_ts = info.get("run_max_bid_ts")
    rmg = float(info.get("run_max_gb_bps", 0.0))
    rmdd = float(info.get("run_max_dd_bps", 0.0))       # max drawdown-from-ENTRY so far (long SL envelope)
    ask_entry = info.get("ask_entry")                   # fixed long-SL reference (the long entry fill)
    dd_from = int(info.get("env_dd_from_ts", info.get("env_scan_ts", ats)))  # dd-coverage start; a value
    #   > ats means dd tracking began mid-hold (deploy-straddling anchor) => PARTIAL dd; stamped on every
    #   dd record so a bracket-SL replay can self-exclude it without out-of-band deploy-time knowledge.
    last_high_emit = info.get("run_max_bid_emitted")   # last EMITTED high bid (high-record throttle ref)
    # ---- SHORT ask-side MIRROR state (entry@bid / exit@ask, favorable = ask falling) ----
    rmin = info.get("run_min_ask")                      # running MIN ask = short-MFE peak
    rmin_ts = info.get("run_min_ask_ts")
    rmin_emit = info.get("run_min_ask_emitted")        # last EMITTED short_low ask (throttle ref)
    s_gb = float(info.get("run_max_short_gb_bps", 0.0))  # max give-back from run_min_ask (short-trailing)
    s_dd = float(info.get("run_max_short_dd_bps", 0.0))  # max ask-rise from entry (short-SL)
    bid_entry = info.get("bid_entry")                   # fixed short reference (short sells into the bid)
    s_dd_from = int(info.get("short_dd_from_ts", info.get("env_scan_ts", ats)))
    quar = bool(info.get("env_quar", False))
    # Gap detection references the last REAL scanned book point — NOT env_scan_ts, which is an
    # artificial CAP/book_max boundary (line: scan_end) and can sit past the last real tick. Measuring
    # from the real last point makes a >WINDOW_GAP_MS outage that straddles a pass/CAP boundary — the
    # backfill-on-deploy / post-downtime resume path — quarantine correctly (blocker #1 fix).
    prev_ts = int(info.get("env_last_pt_ts", info.get("env_scan_ts", ats)))
    recs = []
    for ts, bid, ask in rows:
        ts = int(ts)
        if ts - prev_ts > WINDOW_GAP_MS:        # broken feed inside the hold -> envelope untrustworthy
            quar = True
        prev_ts = ts                            # feed-continuity point (a row exists => feed alive)
        # ---- LONG bid-side (high / gb / dd) ----
        if bid is not None:
            bid = float(bid)
            if rmb is None or bid > rmb:
                rmb, rmb_ts = bid, ts            # TRUE running max advances every tick (gb references it)
                # high RECORD throttled to 1bps steps (MFE curve only); gb envelope stays 1bps-quantized
                if last_high_emit is None or (rmb - last_high_emit) / last_high_emit * 1e4 >= ENV_MIN_HIGH_STEP_BPS:
                    last_high_emit = rmb
                    recs.append({"kind": "high", "ts_ms": ts, "bid": bid, "run_max_bid": rmb,
                                 "giveback_bps": 0.0, "quarantined": quar})
            else:
                gb = (rmb - bid) / rmb * 1e4
                if gb >= rmg + ENV_MIN_GB_STEP_BPS:
                    rmg = gb
                    recs.append({"kind": "gb", "ts_ms": ts, "bid": bid, "run_max_bid": rmb,
                                 "giveback_bps": round(gb, 2), "quarantined": quar})
            # long-SL: max drawdown-from-ENTRY (references fixed ask_entry). first dd_bps>=S = SL first-cross,
            # fill AT that bid, OPTIMISTIC / no gap-through.
            if ask_entry:
                dd = (ask_entry - bid) / ask_entry * 1e4
                if dd >= rmdd + ENV_MIN_DD_STEP_BPS:
                    rmdd = dd
                    recs.append({"kind": "dd", "ts_ms": ts, "bid": bid, "dd_bps": round(dd, 2),
                                 "dd_from_ts": dd_from, "quarantined": quar})
        # ---- SHORT ask-side MIRROR (short_low / short_gb / short_dd) ----
        if ask is not None:
            ask = float(ask)
            if rmin is None or ask < rmin:
                rmin, rmin_ts = ask, ts          # TRUE running min ask advances every tick (short_gb refs it)
                # short_low RECORD throttled 1bps (short-MFE curve); short_gb stays 1bps-quantized
                if rmin_emit is None or (rmin_emit - rmin) / rmin_emit * 1e4 >= ENV_MIN_HIGH_STEP_BPS:
                    rmin_emit = rmin
                    recs.append({"kind": "short_low", "ts_ms": ts, "ask": ask, "run_min_ask": rmin,
                                 "giveback_bps": 0.0, "quarantined": quar})
            else:
                sgb = (ask - rmin) / rmin * 1e4   # short give-back: ask risen back from the low
                if sgb >= s_gb + ENV_MIN_GB_STEP_BPS:
                    s_gb = sgb
                    recs.append({"kind": "short_gb", "ts_ms": ts, "ask": ask, "run_min_ask": rmin,
                                 "giveback_bps": round(sgb, 2), "quarantined": quar})
            # short-SL: max ask-RISE from ENTRY (references fixed bid_entry). first dd_bps>=S = short-SL cross.
            if bid_entry:
                sdd = (ask - bid_entry) / bid_entry * 1e4
                if sdd >= s_dd + ENV_MIN_DD_STEP_BPS:
                    s_dd = sdd
                    recs.append({"kind": "short_dd", "ts_ms": ts, "ask": ask, "dd_bps": round(sdd, 2),
                                 "dd_from_ts": s_dd_from, "quarantined": quar})
    info["run_max_bid"] = rmb
    info["run_max_bid_ts"] = rmb_ts
    info["run_max_gb_bps"] = rmg
    info["run_max_dd_bps"] = rmdd
    info["env_dd_from_ts"] = dd_from
    info["run_max_bid_emitted"] = last_high_emit
    # SHORT ask-side mirror state
    info["run_min_ask"] = rmin
    info["run_min_ask_ts"] = rmin_ts
    info["run_min_ask_emitted"] = rmin_emit
    info["run_max_short_gb_bps"] = s_gb
    info["run_max_short_dd_bps"] = s_dd
    info["short_dd_from_ts"] = s_dd_from
    # last REAL book point seen this pass (prev_ts stays at the carried-in value if the slice was empty,
    # so an all-outage slice keeps the pre-outage reference and the next pass measures the full gap).
    info["env_last_pt_ts"] = prev_ts
    info["env_scan_ts"] = scan_end   # SQL window bound only — never used for gap distance (see prev_ts)
    info["env_quar"] = quar
    if scan_end >= pos_end:
        info["env_done"] = True
    return recs


def _load_state():
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                d.setdefault("processed", [])
                d.setdefault("pending", {})
                d.setdefault("swing_pending", {})
                return d
        except json.JSONDecodeError:
            pass
    return {"processed": [], "pending": {}, "swing_pending": {}}


def _save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE)


def _log(rec):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _latest_le(cur, table, col, symbol, ts, max_stale_ms):
    """Latest {col} for symbol with ts_ms<=ts (CAUSAL), None if absent or staler than max_stale_ms.
    table/col are code-literals (not external input). Indexed point lookup, cheap."""
    r = cur.execute(
        f"SELECT ts_ms, {col} FROM {table} WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts)).fetchone()
    if not r or r[1] is None:
        return None
    if ts - int(r[0]) > max_stale_ms:
        return None
    return float(r[1])


def _reversal_features(cur, ts):
    """S34_REVERSAL_SELECTOR_V1 (FROZEN prereg 2026-07-23): T0-causal features to test whether a selector
    separates REVERTER (bounce) from TRENDER (-100 tail) anchors — FORWARD ONLY, evidence not deploy.
    ALL strictly causal (ts<=anchor_ts). Absent/stale source -> None (evaluator excludes; no fabrication)."""
    W = 30 * 60 * 1000          # frozen pre-window = 30 min
    STALE = 5 * 60 * 1000       # OI/spot sample must be <=5min stale to count (else None)
    btc_ret_30m = _mark_bps(cur, "BTCUSDT", ts, W)     # F1 PRIMARY: concurrent BTC move (systematic vs local)
    eth_ret_30m = _mark_bps(cur, "ETHUSDT", ts, W)
    div_30m = (round(eth_ret_30m - btc_ret_30m, 1)     # F1b: ETH-BTC decoupling (idiosyncratic dislocation)
               if (eth_ret_30m is not None and btc_ret_30m is not None) else None)
    oi_now = _latest_le(cur, "open_interest", "open_interest_usd", "ETHUSDT", ts, STALE)
    oi_prev = _latest_le(cur, "open_interest", "open_interest_usd", "ETHUSDT", ts - W, STALE)
    d_oi_pct = (round((oi_now - oi_prev) / oi_prev * 100.0, 3)   # F2: OI direction (deleverage vs pile-in)
                if (oi_now and oi_prev) else None)
    spot_now = _latest_le(cur, "spot_prices", "spot_price", "ETHUSDT", ts, STALE)
    fut_now = _mark_at(cur, ts)                          # ETHUSDT futures mark @ ts (causal)
    basis_bps = (round((fut_now - spot_now) / spot_now * 1e4, 1)  # F3: futures-spot basis (dislocation)
                 if (spot_now and fut_now) else None)
    return {"rs_btc_ret_30m": (round(btc_ret_30m, 1) if btc_ret_30m is not None else None),
            "rs_eth_ret_30m": (round(eth_ret_30m, 1) if eth_ret_30m is not None else None),
            "rs_div_30m": div_30m, "rs_d_oi_pct": d_oi_pct, "rs_basis_bps": basis_bps}


def _gates(cur, ts, ts_list):
    """Return (signals, snapshot) for T0-knowable hour17 / echo-causal gates."""
    d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
    btc4h = _mark_bps(cur, "BTCUSDT", ts, 4 * 3600_000) or 0.0
    btc7d = _mark_bps(cur, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0.0
    eth1h = _mark_bps(cur, "ETHUSDT", ts, 3600_000) or 0.0
    echo_30_90 = _echo_check(ts_list, ts, 30, 90)
    regime = (btc4h < 0) or (btc7d < 0)
    bull = (eth1h > 20.0) and (btc4h > 50.0)
    sess = _session(d.hour)
    dow = d.weekday()
    sigs = []
    if (not bull) and sess != "EUROPE" and regime and d.hour >= 17:
        sigs.append("hour17")
    if (not bull) and sess != "EUROPE" and dow not in {0, 2} and echo_30_90 and regime:
        sigs.append("echo_causal")
    snap = {"hour_utc": d.hour, "session": sess, "dow": dow,
            "btc4h_bps": round(btc4h, 1), "btc7d_bps": round(btc7d, 1), "echo_30_90": echo_30_90,
            **_reversal_features(cur, ts)}
    return sigs, snap


def run_once(conn, st):
    cur = conn.cursor()
    # data clocks (latest ts) come from the LIVE file ONLY: over a union view MAX(ts_ms) would
    # full-scan BOTH files (biggest = book_ticker, ~114M rows), losing the index min/max fast-path.
    # The global max is always in the live file (ts_ms >= cutoff). History reads use `cur` (union).
    _live = open_live_ro()
    try:
        _lcur = _live.cursor()
        mx = _lcur.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
        now_ms = int(mx[0]) if mx and mx[0] else int(time.time() * 1000)
        book_max = _book_max(_lcur)   # book-feed clock, on the live file (a horizon resolves only once BOTH clocks pass exit_ts)
    finally:
        _live.close()
    processed = set(int(x) for x in st.get("processed", []))
    pending = {int(k): v for k, v in st.get("pending", {}).items()}
    swing_pending = {int(k): v for k, v in st.get("swing_pending", {}).items()}
    opened = resolved = quarantined_ct = 0

    fresh, ts_list = _detect_fresh_anchors(cur, now_ms)
    for ts, rn in fresh:
        if ts in processed:
            continue
        if ts <= CUTOFF_MS:          # burned pre-forward anchor
            processed.add(ts)
            continue
        sigs, snap = _gates(cur, ts, ts_list)
        entry = _mark_at(cur, ts)
        book_e = _book_at(cur, ts)                       # measured-cost entry: a buy lifts the ask
        if entry:
            # Open ALL forward anchors (not just qualifying ones) so each signal has a NON-qualified
            # CONTROL group — if the gate adds value, qualified must beat control forward (operator Q).
            q_h17 = "hour17" in sigs
            q_echo = "echo_causal" in sigs
            d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
            ask_entry = book_e["ask"] if book_e else None
            bid_entry = book_e["bid"] if book_e else None   # SHORT entry (short taker sells into the bid)
            e_age = book_e["age_ms"] if book_e else None
            e_spread = round(book_e["spread_bps"], 2) if book_e and book_e["spread_bps"] is not None else None
            e_depth = book_e["bid_depth_usd"] if book_e else None
            _log({"event": "OPEN", "anchor_ts_ms": ts, "utc": d.isoformat(),
                  "qualified_hour17": q_h17, "qualified_echo": q_echo, "signals": sigs,
                  "entry_mark": entry, "ask_entry": ask_entry, "bid_entry": bid_entry,
                  "entry_quote_age_ms": e_age,
                  "spread_bps_entry": e_spread, "bid_depth_usd_entry": e_depth, "cost_model": "measured_v2",
                  "running_notional": rn, "horizons_h": HORIZONS_H, **snap})
            entry_stale = (ask_entry is None or e_age is None or e_age > QUOTE_STALE_MS)
            pending[ts] = {"entry": entry, "ask_entry": ask_entry, "bid_entry": bid_entry,
                           "entry_quote_age_ms": e_age,
                           "spread_bps_entry": e_spread, "bid_depth_usd_entry": e_depth,
                           "q_h17": q_h17, "q_echo": q_echo,
                           "resolved": [], "hour_utc": snap["hour_utc"],
                           # LONG bid-side envelope state (S34_HH_GIVEBACK_FORWARD_LOG_V1)
                           "env_scan_ts": ts, "env_last_pt_ts": ts,
                           "run_max_bid": None, "run_max_bid_ts": None,
                           "run_max_bid_emitted": None, "run_max_gb_bps": 0.0,
                           "run_max_dd_bps": 0.0,   # SL envelope: max drawdown-from-entry so far
                           "env_dd_from_ts": ts,    # dd-coverage start (=ats => COMPLETE; > ats => PARTIAL,
                                                    #   so a bracket-SL replay self-excludes it)
                           # SHORT ask-side MIRROR envelope state (direction dimension; entry@bid/exit@ask,
                           #   favorable = ask falling). short_low=run_min_ask (short-MFE/short-TP),
                           #   short_gb=give-back from that low (short-trailing), short_dd=ask rise from
                           #   entry (short-SL). Same throttle/quantize/quarantine/gap-detection as long.
                           "run_min_ask": None, "run_min_ask_ts": None, "run_min_ask_emitted": None,
                           "run_max_short_gb_bps": 0.0, "run_max_short_dd_bps": 0.0, "short_dd_from_ts": ts,
                           "env_quar": bool(entry_stale), "env_done": False}
            # lightweight swing-observation tracking (mark-checkpoint drift; separate from the envelope
            # pending so the main pending drops at 48h while swing persists to 365d via state)
            swing_pending[ts] = {"entry_mark": entry, "ask_entry": ask_entry, "bid_entry": bid_entry,
                                 "q_h17": q_h17, "q_echo": q_echo, "hour_utc": snap["hour_utc"],
                                 "utc": d.isoformat(), "swing_resolved": []}
            opened += 1
            processed.add(ts)
        # N5: if the mark feed hasn't caught up to this anchor yet (collector race), DO NOT mark it
        # processed — leave it for the next pass to retry. _detect_fresh_anchors only surfaces anchors
        # <= FRESH_MS old, so retries are naturally bounded and a transient race no longer drops the OPEN.

    still = {}
    t_budget0 = time.monotonic()
    budget_hit = False
    for ats, info in pending.items():
        entry = info.get("entry")
        done_h = set(info.get("resolved", []))
        for h in HORIZONS_H:
            if h in done_h:
                continue
            exit_ts = ats + h * 3600_000
            if now_ms < exit_ts:
                continue                                   # mark clock not yet at the horizon
            if book_max is None or book_max < exit_ts:
                continue                                   # B3: book feed hasn't reached exit_ts -> WAIT, never drop
            if not budget_hit and time.monotonic() - t_budget0 > RESOLVE_BUDGET_SEC:
                budget_hit = True                          # B2: stop starting new (possibly slow) scans this pass
            if budget_hit:
                break                                      # remaining horizons/anchors stay pending, retried next pass
            exit_mark = _mark_at(cur, exit_ts)
            if not exit_mark or not entry:
                continue
            # --- mark-based net (LEGACY, kept ONLY for reconciliation + slippage attribution) ---
            net_mark = round((exit_mark - entry) / entry * 1e4 - FEE_BPS, 2)
            pm_mark = _min_mark(cur, "ETHUSDT", ats, exit_ts)

            # --- measured-cost net: buy@ask (entry) / sell@bid (exit) => spread PAID + commission ---
            ask_entry = info.get("ask_entry")
            book_x = _book_at(cur, exit_ts)
            bid_exit = book_x["bid"] if book_x else None
            min_bid = _min_bid(cur, ats, exit_ts)
            # --- peak side (MFE), symmetric with path_min_bid/path_min_mark; single aggregates ---
            max_bid = _max_bid(cur, ats, exit_ts)
            px_mark = _max_mark(cur, "ETHUSDT", ats, exit_ts)
            mfe_bid = round((max_bid - ask_entry) / ask_entry * 1e4 - COMMISSION_BPS, 2) \
                if (ask_entry and max_bid) else None
            mfe_mark = round((px_mark - entry) / entry * 1e4 - FEE_BPS, 2) if (px_mark and entry) else None

            # --- source outage quarantine (in the LEDGER, per horizon) — the §166 guard ---
            # (book_max >= exit_ts is already guaranteed above, so exit_quote_stale now flags only a GENUINE
            #  gap at exit time, not a transient book-feed lag — that path defers, it does not quarantine.)
            reasons = []
            e_age = info.get("entry_quote_age_ms")
            if ask_entry is None or e_age is None or e_age > QUOTE_STALE_MS:
                reasons.append("entry_quote_stale")
            if book_x is None or book_x["age_ms"] > QUOTE_STALE_MS:
                reasons.append("exit_quote_stale")          # genuine stale exit quote = the §166 fake-+900 mechanism
            if _window_has_gap(cur, ats, exit_ts, WINDOW_GAP_MS):
                reasons.append("in_window_gap")
            quarantined = bool(reasons)
            # DEFENSE-IN-DEPTH backstop (operator-required, on TOP of the env_last_pt_ts primary fix):
            # _window_has_gap is a FULL-SPAN check that does NOT trust the incremental envelope scan. If
            # it finds a real gap in [ats, exit_ts], force the anchor's envelope quarantine sticky too —
            # so a contaminated envelope is caught even if incremental gap-detection ever missed it. As
            # more horizons resolve, this independently re-checks progressively more of the 48h span.
            if "in_window_gap" in reasons:
                info["env_quar"] = True

            net_meas = net_s150_meas = net_s300_meas = slip = None
            if ask_entry and bid_exit:
                net_meas = round((bid_exit - ask_entry) / ask_entry * 1e4 - COMMISSION_BPS, 2)
                slip = round(net_mark - net_meas, 2)        # per-trade measured cost of (spread+commission) vs old fee

                def _stop_meas(bps):
                    # LONG stop fills at bid; measured realized loss = -(stop_bps + commission). OPTIMISTIC:
                    # assumes fill AT the stop level, no gap-through (real stops slip worse — §162/§163).
                    if min_bid is not None and min_bid <= ask_entry * (1.0 - bps / 1e4):
                        return round(-(bps + COMMISSION_BPS), 2)
                    return net_meas
                net_s150_meas = _stop_meas(150.0)
                net_s300_meas = _stop_meas(300.0)

            # B1: net_bps* are the AUTHORITATIVE, consumer-facing interface (dashboard :8771 + evaluator read
            # these). They carry the measured value ONLY when the row is clean (not quarantined & measurable),
            # else None so any consumer auto-excludes contaminated rows. Raw measured + mark kept for forensics.
            clean = (not quarantined) and net_meas is not None
            net_bps = net_meas if clean else None
            net_bps_s150 = net_s150_meas if clean else None
            net_bps_s300 = net_s300_meas if clean else None

            depth_e = info.get("bid_depth_usd_entry")
            tob_insufficient = (depth_e < NOTIONAL_USD) if depth_e is not None else None

            # ONE RESOLVE per (anchor, horizon) carrying every gate flag. net_bps* = authoritative measured
            # (None when quarantined); net_bps_measured/net_bps_mark kept for reconciliation. Flags let the
            # reader form qualified vs CONTROL vs echo∩hour>=17 slices. Captured NOW (forward = un-minable).
            _log({"event": "RESOLVE", "anchor_ts_ms": ats, "hold_h": h, "cost_model": "measured_v2",
                  # AUTHORITATIVE consumer-facing net (measured; None when quarantined/unmeasurable)
                  "net_bps": net_bps, "net_bps_s150": net_bps_s150, "net_bps_s300": net_bps_s300,
                  # measured detail + slippage attribution
                  "ask_entry": ask_entry, "bid_exit": bid_exit, "net_bps_measured": net_meas,
                  "path_min_bid": min_bid, "slip_bps": slip, "spread_bps_entry": info.get("spread_bps_entry"),
                  "bid_depth_usd_entry": depth_e, "top_of_book_insufficient": tob_insufficient,
                  # integrity
                  "quarantined": quarantined, "quarantine_reasons": reasons,
                  "exit_quote_age_ms": (book_x["age_ms"] if book_x else None),
                  # peak side (MFE, symmetric with path_min) — magnitude at this horizon; the fine
                  # give-back TIMING for a trailing-Z sweep lives in the separate ENV records.
                  "path_max_bid": max_bid, "path_max_mark": px_mark,
                  "mfe_bid_bps": mfe_bid, "mfe_mark_bps": mfe_mark,
                  # mark-based (RECONCILIATION only)
                  "entry_mark": entry, "exit_mark": exit_mark, "net_bps_mark": net_mark, "path_min_mark": pm_mark,
                  # tags
                  "hour_utc": info.get("hour_utc"),
                  "qualified_hour17": info.get("q_h17"), "qualified_echo": info.get("q_echo"),
                  "commission_bps": COMMISSION_BPS, "fee_bps": FEE_BPS})
            if quarantined:
                quarantined_ct += 1
            resolved += 1
            done_h.add(h)
        info["resolved"] = sorted(done_h)
        still[ats] = info   # keep-decision deferred to after the envelope pass below

    # --- envelope pass: extend the CAUSAL running-max-bid give-back envelope for each open anchor.
    # Forward-logging, NOT time-critical -> runs AFTER OPEN/RESOLVE under its own wall-clock budget and
    # DEFERS (never drops) on overrun; state persists so the next pass resumes exactly where it stopped.
    env_recs = 0
    if book_max is not None:
        t_env0 = time.monotonic()
        for ats, info in list(still.items()):
            if info.get("env_done"):
                continue
            if time.monotonic() - t_env0 > ENV_BUDGET_SEC:
                break
            for r in _scan_envelope(cur, ats, info, book_max):
                # record already carries kind-specific fields (high/gb: run_max_bid+giveback_bps;
                # dd: dd_bps) — spread it so each envelope kind logs exactly its own schema.
                _log({"event": "ENV", "anchor_ts_ms": ats, "cost_model": "measured_v2",
                      "qualified_hour17": info.get("q_h17"), "qualified_echo": info.get("q_echo"), **r})
                env_recs += 1

    # drop an anchor ONLY when it has BOTH resolved every horizon AND finished the 48h envelope
    still = {a: i for a, i in still.items()
             if len(set(i.get("resolved", []))) < len(HORIZONS_H) or not i.get("env_done")}

    # --- SWING observation checkpoints (mark-checkpoint drift; OBSERVATION-ONLY, cheap point-lookups,
    # NOT in the exit-sweep verdict family). Each anchor persists here until 365d matures (state-backed,
    # reset-survive). Mark clock = now_ms; measured taker via a single _book_at lookup. ---
    swing_still = {}
    for ats, si in swing_pending.items():
        em = si.get("entry_mark"); ae = si.get("ask_entry"); be = si.get("bid_entry")
        done_s = set(si.get("swing_resolved", []))
        for h in SWING_HORIZONS_H:
            if h in done_s:
                continue
            exit_ts = ats + h * 3600_000
            if now_ms < exit_ts:
                continue                                   # mark clock not yet at the swing horizon
            xm = _mark_at(cur, exit_ts)
            if not xm or not em:
                continue
            long_mark = round((xm - em) / em * 1e4 - FEE_BPS, 2)
            short_mark = round(-((xm - em) / em * 1e4) - FEE_BPS, 2)
            bx = _book_at(cur, exit_ts)                    # single measured lookup (cheap)
            long_meas = round((bx["bid"] - ae) / ae * 1e4 - COMMISSION_BPS, 2) if (bx and ae) else None
            short_meas = round((be - bx["ask"]) / be * 1e4 - COMMISSION_BPS, 2) if (bx and be) else None
            _log({"event": "SWING", "observation": True, "anchor_ts_ms": ats, "swing_h": h,
                  "kind": "swing_observation", "entry_mark": em, "exit_mark": xm,
                  "long_net_mark": long_mark, "short_net_mark": short_mark,
                  "long_net_meas": long_meas, "short_net_meas": short_meas,
                  "hour_utc": si.get("hour_utc"),
                  "qualified_hour17": si.get("q_h17"), "qualified_echo": si.get("q_echo"),
                  "fee_bps": FEE_BPS, "commission_bps": COMMISSION_BPS,
                  "NOTE": "swing OBSERVATION-only drift checkpoint; NOT in the exit-sweep verdict family"})
            done_s.add(h)
        si["swing_resolved"] = sorted(done_s)
        if len(done_s) < len(SWING_HORIZONS_H):
            swing_still[ats] = si                          # keep until 365d matures

    st["processed"] = sorted(processed)[-5000:]
    st["pending"] = still
    st["swing_pending"] = swing_still
    _save_state(st)
    return opened, resolved, len(still), quarantined_ct


def main():
    global LEDGER, STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    ap.add_argument("--ledger", help="override ledger JSONL path (isolated dry-run)")
    ap.add_argument("--state", help="override state JSON path (isolated dry-run)")
    args = ap.parse_args()
    if args.ledger:
        LEDGER = Path(args.ledger)
    if args.state:
        STATE = Path(args.state)
    st = _load_state()
    while True:
        # Opened INSIDE the try, matching liq_tip_forward: a Phase-4 frozen-segment
        # delete (or a malformed rotation_state.json) makes open_union_ro raise, and
        # an uncaught raise here would end main() permanently. Nothing supervises or
        # restarts this role, and the forward days it would miss CANNOT be re-mined.
        conn = None
        try:
            conn = open_union_ro()  # live+frozen union post-rotation (sets query_only)
            o, r, p, q = run_once(conn, st)
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} opened={o} resolved={r} "
                  f"pending={p} quarantined={q}")
        except RotationStateError as exc:
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} ROTATION_STATE_ERROR {exc} (retrying next cycle)")
        finally:
            if conn is not None:
                conn.close()
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
