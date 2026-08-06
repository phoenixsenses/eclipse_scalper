"""
research_s34_echo_forward_ledger.py — FROZEN-RULE forward ledger for echo_30_90+regime (read-only).

Governs the last un-burned lead (prereg S34_ECHO_FRESH_GATED_PREREGISTRATION_V1.md, SYSTEM_STATE §164).
There is NO un-burned historical holdout (discovery burned Feb-Jul 2026 entirely), so the ONLY honest
test is FORWARD accumulation of anchors AFTER 2026-07-20, recorded here and never re-mined.

CAUSALITY NOTE (important): the discovery rule `cand_9090` includes a `not noisy` gate that inspects
(T+60s, T+30m) — i.e. it uses information 30 min AHEAD of the T0 entry. That is a LOOKAHEAD contaminant
(it removes, with hindsight, exactly the events whose selling continues = the tails), and likely explains
the pristine "+92.5bps / tail 0". This ledger therefore records TWO qualification levels honestly:
  - qualified_t0   : only T0-knowable gates (regime, echo_30_90, not-bull, session, dow) — CAUSAL, tradeable
  - qualified_full : qualified_t0 AND not-noisy — the discovery rule, but noisy resolves only at T+30m
The forward data will show whether the causal candidate survives without the lookahead gate.

Also captures a rich per-event INDICATOR SNAPSHOT (see docs/ECHO_SIGNAL_DEV_INDICATORS.md) so the signal
can later be IMPROVED on forward-clean data WITHOUT re-mining the burned sample (OD-029).

NOT a trader, no orders. DB read-only (mode=ro, query_only=1); writes only its own append-only JSONL +
state. FEE=5bps net baked into recorded net_bps. Loop: `--once` or persistent via start_eclipse.ps1.

Ledger: reports/shadow/echo_forward_ledger.jsonl   State: reports/shadow/echo_forward_ledger_state.json
"""
from __future__ import annotations
import sqlite3, json, sys, time, argparse, bisect, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors, load_liquidations  # noqa: E402
from ami.storage.union_reader import (  # noqa: E402
    open_union_ro, open_live_ro, RotationStateError, InsufficientHistoryError)

# What a Phase-4 delete can actually throw at the connection factory. RotationStateError
# alone is too narrow: open_union_ro checks os.path.exists and only THEN attaches, and a
# segment that is truncated, mid-replace, or a leftover directory fails inside the ATTACH
# as a plain sqlite3 error. InsufficientHistoryError is included ahead of the coverage
# assertions being wired in, so landing those cannot silently start killing this role.
SURVIVABLE_ESTATE_ERRORS = (RotationStateError, InsufficientHistoryError, sqlite3.Error)
# Retrying a broken estate forever, printing a line each time, IS the silent-standstill
# failure -- the role looks alive while accumulating nothing. Surface it instead.
MAX_CONSECUTIVE_ESTATE_FAILURES = 20  # ~10 min at the 30s default interval

DB_URI = f"file:{ROOT / 'data' / 'microstructure.db'}?mode=ro"
LEDGER = ROOT / "reports" / "shadow" / "echo_forward_ledger.jsonl"
STATE = ROOT / "reports" / "shadow" / "echo_forward_ledger_state.json"

# ---- FROZEN parameters (must match prereg §1; changing any = new prereg) --------------------
ETH_THRESH = 200_000.0
PROP_THRESH = 50_000.0
BUCKET_SEC, MIN_GAP_SEC, ACCEL_SEC = 300, 900, 30
FRESH_MS = 120_000          # fire on anchors <= 2min old (no history backfill flood)
LOOKBACK_ANCHOR_MS = 3_600_000
ECHO_LOOKBACK_MS = 2 * 3_600_000   # window to reconstruct prior anchors for echo_check
FEE_BPS = 5.0
HOLD_MS = 4 * 3_600_000
NOISY_RESOLVE_MS = 30 * 60_000     # not-noisy gate resolves here (T+30m) — post-hoc, lookahead marker
# --------------------------------------------------------------------------------------------


def _load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": [], "pending": {}}


def _save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATE)


def _log(rec):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _processed_from_ledger():
    seen = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("event") == "OPEN" and r.get("anchor_ts_ms") is not None:
                    seen.add(int(r["anchor_ts_ms"]))
            except Exception:
                continue
    return seen


def _mark_at(cur, ts, sym="ETHUSDT"):
    r = cur.execute("SELECT mark_price, ts_ms FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts)).fetchone()
    return float(r[0]) if r and r[0] else None


def _mark_bps(cur, sym, ts, lookback_ms):
    a = cur.execute("SELECT mark_price, ts_ms FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts - lookback_ms)).fetchone()
    b = cur.execute("SELECT mark_price, ts_ms FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts)).fetchone()
    if a and b and a[0] and float(a[0]) > 0:
        return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4
    return None


def _liq_sum(cur, sym, side, lo, hi):
    r = cur.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
                    (sym, side, lo, hi)).fetchone()
    return float(r[0]) if r else 0.0


def _liq_cnt(cur, sym, side, lo, hi, thr):
    r = cur.execute("SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
                    (sym, side, lo, hi, thr)).fetchone()
    return int(r[0]) if r else 0


def _first_sell_after(cur, sym, lo, hi, thr):
    r = cur.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side='SELL' AND ts_ms>=? AND ts_ms<? AND notional>=? "
                    "ORDER BY ts_ms ASC LIMIT 1", (sym, lo, hi, thr)).fetchone()
    return int(r[0]) if r else None


def _liq_max(cur, sym, side, lo, hi):
    r = cur.execute("SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<=?",
                    (sym, side, lo, hi)).fetchone()
    return float(r[0]) if r else 0.0


def _min_mark(cur, sym, lo, hi):
    r = cur.execute("SELECT MIN(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
                    (sym, lo, hi)).fetchone()
    return float(r[0]) if r and r[0] else None


def _be_ratio(cur, ts, rn, ahead_ms=0):
    """BTC-flush concentration = max BTC SELL liq in [ts-10m, ts+ahead] / ETH anchor notional.
    ahead=0 => causal (T0). PASSIVE context only: reactive/stop arms built on this are DEAD in-sample
    (S34_ECHO_REACTIVE_ARM_SPEC); captured for FORWARD re-confirmation, NOT a gate."""
    if not rn or rn <= 0:
        return None
    bc = _liq_max(cur, "BTCUSDT", "SELL", ts - 10 * 60_000, ts + ahead_ms)
    return round(bc / rn, 4)


def _detect_fresh_anchors(cur, now_ms):
    liqs = load_liquidations(cur.connection, "ETHUSDT", "SELL", now_ms - LOOKBACK_ANCHOR_MS - ECHO_LOOKBACK_MS, now_ms)
    if not liqs:
        return [], []
    anchors = reconstruct_anchors(liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
                                  thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_SEC)
    ts_list = sorted(int(a.anchor_ts_ms) for a in anchors)
    fresh = [(int(a.anchor_ts_ms), float(a.running_notional)) for a in anchors
             if 0 <= now_ms - int(a.anchor_ts_ms) <= FRESH_MS]
    return fresh, ts_list


def _echo_parents(ts_list, ts, lo_min, hi_min):
    """The prior anchors that make the echo gate fire — i.e. the PARENT event(s) this
    candidate is an echo of. Returns every match, not one, because the window can contain
    several and there is no non-arbitrary tie-break; addendum §A rule 2 only needs the
    identity ("same parent => same observation"), which set overlap answers exactly.

    Recorded so counted-independent N can be computed at all: with only the boolean gate,
    two candidates echoing ONE flush are indistinguishable from two independent events, and
    N inflates silently — the exact failure §A exists to prevent."""
    lo_ms, hi_ms = ts - hi_min * 60_000, ts - lo_min * 60_000
    lo_i, hi_i = bisect.bisect_left(ts_list, lo_ms), bisect.bisect_left(ts_list, hi_ms)
    return [ts_list[i] for i in range(lo_i, hi_i) if ts_list[i] != ts]


def _echo_check(ts_list, ts, lo_min, hi_min):
    return bool(_echo_parents(ts_list, ts, lo_min, hi_min))


def _session(hour):
    return "EUROPE" if 7 <= hour < 13 else ("US" if 13 <= hour < 21 else "OFF")


def indicator_snapshot(cur, ts, rn=None):
    """Rich T0-causal indicator snapshot for forward-clean signal development (OD-029 safe).
    See docs/ECHO_SIGNAL_DEV_INDICATORS.md for the rationale of each family."""
    snap = {}
    # --- regime / trend (BTC + ETH multi-horizon) ---
    snap["btc4h_bps"] = _mark_bps(cur, "BTCUSDT", ts, 4 * 3600_000)
    snap["btc7d_bps"] = _mark_bps(cur, "BTCUSDT", ts, 7 * 24 * 3600_000)
    snap["btc3d_bps"] = _mark_bps(cur, "BTCUSDT", ts, 3 * 24 * 3600_000)
    snap["eth1h_bps"] = _mark_bps(cur, "ETHUSDT", ts, 3600_000)
    snap["eth4h_bps"] = _mark_bps(cur, "ETHUSDT", ts, 4 * 3600_000)
    snap["eth15m_bps"] = _mark_bps(cur, "ETHUSDT", ts, 15 * 60_000)
    # --- echo structure (the prior-cluster geometry that DEFINES the signal) ---
    snap["prebuildup_30m_cnt"] = _liq_cnt(cur, "ETHUSDT", "SELL", ts - 30 * 60_000, ts - 1000, PROP_THRESH)
    snap["sell_liq_2h_cnt"] = _liq_cnt(cur, "ETHUSDT", "SELL", ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
    snap["sell_liq_5m_$M"] = round(_liq_sum(cur, "ETHUSDT", "SELL", ts - 5 * 60_000, ts) / 1e6, 3)
    # --- cross-asset sync (BTC/SOL forced flow) ---
    snap["btc_sell_10m_$M"] = round(_liq_sum(cur, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / 1e6, 3)
    snap["sol_sell_10m_$M"] = round(_liq_sum(cur, "SOLUSDT", "SELL", ts - 10 * 60_000, ts) / 1e6, 3)
    # --- positioning / leverage (funding + OI + basis) ---
    # PRE-EXISTING BUG FIX: mark_prices has NO open_interest_usd / basis_bps columns, so the original
    # 3-column SELECT threw OperationalError on EVERY real anchor => ledger crashed on first OPEN,
    # silently defeating forward capture. Reduced to funding_rate (exists); OI/basis left None until a
    # real source is wired. (Found during be_ratio_causal implementation, 2026-07-20.)
    r = cur.execute("SELECT funding_rate FROM mark_prices "
                    "WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    if r:
        snap["funding_rate"] = r[0]
    snap["open_interest_$B"] = None
    snap["basis_bps"] = None
    # --- flow / microstructure (aggressor imbalance, CVD) ---
    r = cur.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
                    "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades "
                    "WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?", (ts - 60_000, ts)).fetchone()
    # Binance: is_buyer_maker=0 => aggressive BUY (taker buys); =1 => aggressive SELL. (Review Finding 1
    # correction 2026-07-20: labels were swapped, inverting flow_sell_imb_60s/cvd sign vs microstructure_indicators.py.)
    abuy, asell = (r[0] or 0.0), (r[1] or 0.0)
    tot = asell + abuy
    snap["flow_sell_imb_60s"] = round((asell - abuy) / tot, 3) if tot > 0 else None
    snap["cvd_60s_$M"] = round((abuy - asell) / 1e6, 3)
    # --- volatility (own RV; vol_state unreliable) ---
    vr = cur.execute("SELECT rv_5m, vol_decile FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? "
                     "ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    if vr:
        snap["rv_5m"] = vr[0]
        snap["vol_decile"] = vr[1]
    # --- BTC-flush concentration @T0 (be_ratio_causal): PASSIVE context, NOT a gate (arm dead in-sample) ---
    snap["be_ratio_causal"] = _be_ratio(cur, ts, rn, 0)
    return snap


def _live_now_ms():
    """Data clock = MAX(ts_ms) on mark_prices, from the LIVE file ONLY. The global
    max always lives in the fresh file (ts_ms >= cutoff); over a union view MAX would
    full-scan BOTH files, losing SQLite's index min/max fast-path."""
    lc = open_live_ro()
    try:
        mx = lc.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    finally:
        lc.close()
    return int(mx[0]) if mx and mx[0] else int(time.time() * 1000)


def run_once(conn, st):
    cur = conn.cursor()
    now_ms = _live_now_ms()  # live-file clock (fast-path); `conn` (union) is for history range scans
    processed = set(int(x) for x in st.get("processed", [])) | _processed_from_ledger()
    pending = {int(k): v for k, v in st.get("pending", {}).items()}
    opened = closed = 0

    fresh, ts_list = _detect_fresh_anchors(cur, now_ms)
    for ts, rn in fresh:
        if ts in processed:
            continue
        # anchors born on/before the prereg date are still burned; only strictly-forward count
        if ts <= 1784505600000:  # 2026-07-20T00:00:00Z
            processed.add(ts)
            continue
        d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        btc4h = _mark_bps(cur, "BTCUSDT", ts, 4 * 3600_000) or 0.0
        btc7d = _mark_bps(cur, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0.0
        eth1h = _mark_bps(cur, "ETHUSDT", ts, 3600_000) or 0.0
        echo_parents_30_90 = _echo_parents(ts_list, ts, 30, 90)
        echo_30_90 = bool(echo_parents_30_90)
        echo_30_120 = _echo_check(ts_list, ts, 30, 120)
        regime = (btc4h < 0) or (btc7d < 0)
        bull = (eth1h > 20.0) and (btc4h > 50.0)
        sess = _session(d.hour)
        dow = d.weekday()
        # T0-KNOWABLE gates only (causal). noisy resolves later (T+30m) — recorded separately.
        gates_t0 = {"regime": regime, "echo_30_90": echo_30_90, "not_bull": not bull,
                    "sess_not_europe": sess != "EUROPE", "dow_ok": dow not in {0, 2}}
        qualified_t0 = all(gates_t0.values())
        entry = _mark_at(cur, ts)
        _log({"event": "OPEN", "anchor_ts_ms": ts, "utc": d.isoformat(),
              "session": sess, "hour_utc": d.hour, "dow": dow, "month": d.strftime("%Y-%m"),
              "running_notional": rn, "entry_mark": entry,
              "echo_30_90": echo_30_90, "echo_30_120": echo_30_120,
              # ADDITIVE (2026-07-25, SYSTEM_STATE §192): parent identity for addendum §A
              # rule 2. No gate, threshold or qualification semantics changed -- `echo_30_90`
              # is still exactly `bool(parents)`. Forward-only: rows written before this
              # field existed cannot be deduped and are handled fail-closed downstream.
              "echo_parents_30_90": echo_parents_30_90,
              "parent_anchor_ts_ms": (max(echo_parents_30_90) if echo_parents_30_90 else None),
              "gates_t0": gates_t0, "qualified_t0": qualified_t0,
              "indicators": indicator_snapshot(cur, ts, rn),
              "NOTE": "qualified_full (incl not-noisy) resolved at CLOSE; noisy gate is T+30m lookahead."})
        processed.add(ts)
        pending[ts] = {"entry": entry, "qualified_t0": qualified_t0, "rn": rn}
        opened += 1

    # backfill: resolve noisy(T+30m) + outcome(T+4h) then CLOSE
    still = {}
    for ats, info in pending.items():
        entry = info.get("entry")
        if now_ms - ats < HOLD_MS:
            still[ats] = info
            continue
        noisy = _first_sell_after(cur, "ETHUSDT", ats + 60_000, ats + NOISY_RESOLVE_MS, PROP_THRESH) is not None
        exit_mark = _mark_at(cur, ats + HOLD_MS)
        gross = ((exit_mark - entry) / entry * 1e4) if (exit_mark and entry) else None
        net = round(gross - FEE_BPS, 2) if gross is not None else None
        q_full = bool(info.get("qualified_t0")) and (not noisy)
        # --- PASSIVE forward re-confirmation of the (in-sample DEAD) reactive/stop arms ---
        rn = info.get("rn")
        m6 = _mark_at(cur, ats + 6 * 60_000)
        net_6m = round((m6 - entry) / entry * 1e4 - FEE_BPS, 2) if (m6 and entry) else None
        pm = _min_mark(cur, "ETHUSDT", ats, ats + HOLD_MS)
        path_min_bps = round((pm - entry) / entry * 1e4, 2) if (pm and entry) else None
        _log({"event": "CLOSE", "anchor_ts_ms": ats, "entry_mark": entry, "exit_mark": exit_mark,
              "net_bps": net, "noisy_T30m": noisy, "qualified_t0": info.get("qualified_t0"),
              "qualified_full": q_full, "fee_bps": FEE_BPS,
              "be_ratio_resolved_6m": _be_ratio(cur, ats, rn, 6 * 60_000),
              "be_ratio_resolved_10m": _be_ratio(cur, ats, rn, 10 * 60_000),
              "net_6m": net_6m, "path_min_bps": path_min_bps,
              "context_note": "be_ratio/net_6m/path_min are PASSIVE (reactive & stop arms dead in-sample, "
                              "S34_ECHO_REACTIVE_ARM_SPEC); recorded for forward re-confirmation only, NOT gates"})
        closed += 1

    st["processed"] = sorted(processed)[-5000:]
    st["pending"] = still
    _save_state(st)
    return opened, closed, len(still)


def main():
    global LEDGER, STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    ap.add_argument("--ledger", help="override ledger JSONL path (for isolated dry-run)")
    ap.add_argument("--state", help="override state JSON path (for isolated dry-run)")
    args = ap.parse_args()
    if args.ledger:
        LEDGER = Path(args.ledger)
    if args.state:
        STATE = Path(args.state)
    st = _load_state()
    consecutive_failures = 0
    while True:
        # Only the CONNECTION is retried. A Phase-4 frozen-segment delete or a
        # malformed rotation_state.json makes open_union_ro raise, and an uncaught
        # raise would end main() permanently -- nothing supervises this role and the
        # forward days it would miss CANNOT be re-mined.
        #
        # run_once is deliberately OUTSIDE that catch. Wrapping it too meant any
        # persistent sqlite error inside the cycle was swallowed and retried forever:
        # _save_state() is the last statement, so the ledger accumulated NOTHING while
        # printing a line every 30s and staying up. A visible death beats a silent
        # standstill on the un-burned sample.
        conn = None
        try:
            conn = open_union_ro()  # live+frozen union post-rotation (sets query_only)
        except SURVIVABLE_ESTATE_ERRORS as exc:
            consecutive_failures += 1
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} "
                  f"ESTATE_UNAVAILABLE {type(exc).__name__}: {exc} "
                  f"(attempt {consecutive_failures}/{MAX_CONSECUTIVE_ESTATE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_ESTATE_FAILURES:
                # retrying forever IS the silent-standstill failure; surface it
                raise
            if args.once:
                break
            time.sleep(args.interval_sec)
            continue
        consecutive_failures = 0
        try:
            o, cl, p = run_once(conn, st)
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} opened={o} closed={cl} pending={p}")
        finally:
            conn.close()
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
