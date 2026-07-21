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

DB_URI = f"file:{ROOT / 'data' / 'microstructure.db'}?mode=ro"
LEDGER = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger.jsonl"
STATE = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger_state.json"

HORIZONS_H = [2, 4, 6, 12, 24, 48]
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


def _load_state():
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                d.setdefault("processed", [])
                d.setdefault("pending", {})
                return d
        except json.JSONDecodeError:
            pass
    return {"processed": [], "pending": {}}


def _save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE)


def _log(rec):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


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
            "btc4h_bps": round(btc4h, 1), "btc7d_bps": round(btc7d, 1), "echo_30_90": echo_30_90}
    return sigs, snap


def run_once(conn, st):
    cur = conn.cursor()
    mx = cur.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    now_ms = int(mx[0]) if mx and mx[0] else int(time.time() * 1000)
    book_max = _book_max(cur)     # book-feed clock; a horizon resolves only once BOTH clocks pass exit_ts
    processed = set(int(x) for x in st.get("processed", []))
    pending = {int(k): v for k, v in st.get("pending", {}).items()}
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
            e_age = book_e["age_ms"] if book_e else None
            e_spread = round(book_e["spread_bps"], 2) if book_e and book_e["spread_bps"] is not None else None
            e_depth = book_e["bid_depth_usd"] if book_e else None
            _log({"event": "OPEN", "anchor_ts_ms": ts, "utc": d.isoformat(),
                  "qualified_hour17": q_h17, "qualified_echo": q_echo, "signals": sigs,
                  "entry_mark": entry, "ask_entry": ask_entry, "entry_quote_age_ms": e_age,
                  "spread_bps_entry": e_spread, "bid_depth_usd_entry": e_depth, "cost_model": "measured_v2",
                  "running_notional": rn, "horizons_h": HORIZONS_H, **snap})
            pending[ts] = {"entry": entry, "ask_entry": ask_entry, "entry_quote_age_ms": e_age,
                           "spread_bps_entry": e_spread, "bid_depth_usd_entry": e_depth,
                           "q_h17": q_h17, "q_echo": q_echo,
                           "resolved": [], "hour_utc": snap["hour_utc"]}
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
        if len(done_h) < len(HORIZONS_H):
            still[ats] = info   # keep until every horizon matures (B2/B3 defers never drop)

    st["processed"] = sorted(processed)[-5000:]
    st["pending"] = still
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
        conn = sqlite3.connect(DB_URI, uri=True)
        conn.execute("PRAGMA query_only=1")
        try:
            o, r, p, q = run_once(conn, st)
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} opened={o} resolved={r} "
                  f"pending={p} quarantined={q}")
        finally:
            conn.close()
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
