"""
liq_anomaly_monitor.py — live multi-stream ANOMALY-MONITORING engine (read-only tool).

Part of the liquidation "tip system" (SYSTEM_STATE §151-158+): this session established there is NO
fee-surviving directional edge (CV logistic AUC≈0.55). So this is a TOOL — it flags "what is UNUSUAL in
the market right now" (situational awareness), NOT a predictor. Every flag is an a-priori anomaly
definition (percentile / z-score / threshold), never tuned to any outcome. Anything claiming to PREDICT
must go to forward validation, never backtest-mining.

Writes ONE bounded JSON artifact `reports/research/s34/LIQ_ANOMALY_MONITOR.json`:
    {as_of_ms, as_of_utc, symbol, overall_severity, flags:[{name,value,severity,fresh,note}]}

Read-only DB (mode=ro, query_only=1), causal, freshness-flagged (never silently trusts stale OI/spot).
Loop: `--once` or `while True: run_once(); sleep(interval)`.
"""
from __future__ import annotations
import sqlite3, json, sys, math, time, argparse, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.liq_indicator_library import compute_indicators, SYMBOL, FRESH_SPOT_MS  # noqa: E402
from ami.storage.union_reader import (  # noqa: E402
    open_union_ro, open_live_ro, RotationStateError, InsufficientHistoryError)

SURVIVABLE_ESTATE_ERRORS = (RotationStateError, InsufficientHistoryError, sqlite3.Error)
MAX_CONSECUTIVE_ESTATE_FAILURES = 20  # retrying a dead estate forever is itself the failure

OUT = ROOT / "reports" / "research" / "s34" / "LIQ_ANOMALY_MONITOR.json"
SEV_RANK = {"ok": 0, "info": 1, "low": 2, "medium": 3, "high": 4}
FRESH_OI_MS = 5 * 60_000
FRESH_FUND_MS = 20 * 60_000
# Bounded lookback for the funding probes below. funding_rate is SPARSE in mark_prices
# (populated ~every 8h; the rest NULL), so an unbounded `... IS NOT NULL ORDER BY ts_ms
# DESC LIMIT 1` walks backwards over every NULL row. On a single file the index makes that
# walk exit early; across the rotation UNION view the compound query cannot push LIMIT 1
# into each branch and the walk explodes (measured: 19.3s of a 20.4s cycle, on a 30s loop).
# 24h >> the 8h funding cadence, so a live feed is always found; if nothing is found the
# feed is broken and emitting NO funding flag is the correct outcome (never a stale one).
FUNDING_LOOKBACK_MS = 24 * 3_600_000
LIQ_FLIP_LOOKBACK_MS = 7 * 24 * 3_600_000   # same class of bound for the sparse big-liq probe


def _one(cur, sql, args=()):
    r = cur.execute(sql, args).fetchone()
    return r if r else None


def _mark(cur, sym, ts):
    r = _one(cur, "SELECT ts_ms,mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (sym, ts))
    return (r[0], r[1]) if r else (None, None)


def _ret(cur, sym, ts, back):
    _, a = _mark(cur, sym, ts - back); _, b = _mark(cur, sym, ts)
    return (b - a) / a * 1e4 if (a and b) else None


def _flag(name, value, severity, fresh=True, note=""):
    return {"name": name, "value": value, "severity": severity, "fresh": bool(fresh), "note": note}


def compute_flags(conn, as_of_ms):
    cur = conn.cursor()
    ind = compute_indicators(conn, as_of_ms).values
    flags = []

    # ---- FUNDING extremity (14d pctile) + acceleration ----
    fp = ind.get("funding_pctile_14d")
    if fp is not None and ind.get("funding_rate") is not None:
        sev = "high" if (fp >= 0.98 or fp <= 0.02) else ("medium" if (fp >= 0.9 or fp <= 0.1) else "ok")
        flags.append(_flag("funding_extremity", round(fp, 3), sev, note="14d percentile of funding"))
    # `ts_ms` is in the select list because it is the ORDER BY term: without it the union view
    # plans as CO-ROUTINE + COMPOUND and sorts the whole estate (§258). It is LAST so `f0[0]`
    # still means the funding rate. My first scan missed this site because the query is built
    # from two concatenated literals -- a single-line-literal scan cannot see it, which is why
    # the guard in tests/test_union_asof_query_shape.py checks the assembled shapes it can see
    # AND asserts the scan is finding a plausible number of sites.
    _FUND_SQL = ("SELECT funding_rate, ts_ms FROM mark_prices WHERE symbol=? AND ts_ms<=? "
                 "AND ts_ms>=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1")
    f0 = _one(cur, _FUND_SQL, (SYMBOL, as_of_ms, as_of_ms - FUNDING_LOOKBACK_MS))
    f1 = _one(cur, _FUND_SQL, (SYMBOL, as_of_ms - 3_600_000, as_of_ms - 3_600_000 - FUNDING_LOOKBACK_MS))
    f2 = _one(cur, _FUND_SQL, (SYMBOL, as_of_ms - 2 * 3_600_000, as_of_ms - 2 * 3_600_000 - FUNDING_LOOKBACK_MS))
    if f0 and f1 and f2:
        accel = (f0[0] - f1[0]) - (f1[0] - f2[0])
        flags.append(_flag("funding_acceleration", round(accel * 1e6, 2), "info", note="2nd derivative (x1e6)"))

    # ---- OI change + funding<->OI divergence ----
    oi = _one(cur, "SELECT ts_ms,open_interest_usd FROM open_interest WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (SYMBOL, as_of_ms))
    oi1 = _one(cur, "SELECT open_interest_usd, ts_ms FROM open_interest WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (SYMBOL, as_of_ms - 3_600_000))
    oi_fresh = bool(oi) and (as_of_ms - oi[0] <= FRESH_OI_MS)
    if oi and oi1 and oi1[0]:
        chg = (oi[1] - oi1[0]) / oi1[0] * 100
        sev = "medium" if abs(chg) >= 3 else "ok"
        flags.append(_flag("oi_change_1h_pct", round(chg, 2), sev, fresh=oi_fresh, note="stale!" if not oi_fresh else "leverage build/unwind"))
        if f0 and f1:
            fchg = f0[0] - f1[0]
            diverge = (fchg > 0 and abs(chg) < 0.5) or (fchg < 0 and abs(chg) < 0.5)
            if diverge:
                flags.append(_flag("funding_oi_divergence", True, "info", fresh=oi_fresh, note="funding moving, OI flat"))

    # ---- BASIS dislocation (fresh-gated per §156) ----
    b = ind.get("basis_bps")
    if b is not None and ind.get("spot_age_ms") is not None and ind["spot_age_ms"] <= FRESH_SPOT_MS:
        sev = "medium" if abs(b) >= 8 else "ok"
        flags.append(_flag("basis_dislocation_bps", round(b, 2), sev, note="fresh spot"))
    else:
        flags.append(_flag("basis_dislocation_bps", None, "info", fresh=False, note=f"spot stale (age_ms={ind.get('spot_age_ms')}) — basis invalid per §156"))

    # ---- Realized-vol regime + jump (own; vol_state is broken) ----
    rows = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms", (SYMBOL, as_of_ms - 300_000, as_of_ms)).fetchall()
    px = [r[0] for r in rows if r[0]]
    if len(px) >= 5:
        rets = [(px[i] - px[i - 1]) / px[i - 1] for i in range(1, len(px)) if px[i - 1]]
        rv = math.sqrt(sum(x * x for x in rets)) * 1e4
        # jump proxy: max single 1-tick move vs rv
        mx = max(abs(x) for x in rets) * 1e4 if rets else 0
        jumpy = mx > 2.5 * (rv / max(len(rets), 1) ** 0.5) if rv else False
        sev = "medium" if rv > 15 else ("low" if rv > 8 else "ok")
        flags.append(_flag("realized_vol_5m_bps", round(rv, 1), sev, note="own RV (vol_state unreliable)"))
        if jumpy:
            flags.append(_flag("price_jump", round(mx, 1), "medium", note="single-tick jump >> avg"))

    # ---- LIQ intensity + cascade velocity + side-flip ----
    lr = _one(cur, "SELECT SUM(CASE WHEN side='SELL' THEN notional ELSE 0 END),SUM(CASE WHEN side='BUY' THEN notional ELSE 0 END),COUNT(*) FROM liquidations WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (SYMBOL, as_of_ms - 300_000, as_of_ms))
    lsell, lbuy, lc = (lr[0] or 0, lr[1] or 0, lr[2] or 0) if lr else (0, 0, 0)
    if lc > 0:
        kind = "LONG-FLUSH" if lsell >= lbuy else "SHORT-SQUEEZE"
        dom = max(lsell, lbuy)
        sev = "high" if dom >= 5e6 else ("medium" if dom >= 1e6 else "low")
        flags.append(_flag("liq_intensity", {"kind": kind, "dom_$M": round(dom / 1e6, 2), "count_5m": lc}, sev, note="active liquidation cluster"))
        # cascade velocity: liqs in last 60s vs prior 60-300s (accelerating cascade)
        recent = _one(cur, "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (SYMBOL, as_of_ms - 60_000, as_of_ms))[0]
        if recent and recent >= 5:
            flags.append(_flag("cascade_accelerating", recent, "medium", note="liqs firing fast (last 60s)"))
    # side-flip: was the last cluster the opposite side?
    # Bounded like the funding probes above and for the same reason: a sparse predicate
    # (notional>=200K) + ORDER BY ts_ms DESC LIMIT loses its index early-exit on the union
    # view (measured 1.35s of a 2.41s cycle). The flag only describes a RECENT side flip,
    # so a window bound is the honest semantics anyway -- no big liq in the window => no flag.
    lastflip = cur.execute(
        "SELECT side FROM liquidations WHERE symbol=? AND notional>=200000 "
        "AND ts_ms<=? AND ts_ms>=? ORDER BY ts_ms DESC LIMIT 6",
        (SYMBOL, as_of_ms, as_of_ms - LIQ_FLIP_LOOKBACK_MS)).fetchall()
    sides = [s[0] for s in lastflip]
    if len(sides) >= 4 and len(set(sides[:4])) == 2:
        flags.append(_flag("liq_side_flip", True, "info", note="recent SELL<->BUY liq transition"))

    # ---- CROSS-ASSET dispersion (BTC/ETH/SOL 5m returns) ----
    e5 = _ret(cur, "ETHUSDT", as_of_ms, 300_000); b5 = _ret(cur, "BTCUSDT", as_of_ms, 300_000); s5 = _ret(cur, "SOLUSDT", as_of_ms, 300_000)
    rr = [x for x in (e5, b5, s5) if x is not None]
    if len(rr) >= 2:
        signs = [1 if x > 0 else -1 for x in rr]
        codisperse = len(set(signs)) > 1  # coins diverging in direction
        spread = max(rr) - min(rr)
        flags.append(_flag("cross_asset", {"eth5m": round(e5, 1) if e5 is not None else None, "btc5m": round(b5, 1) if b5 is not None else None, "sol5m": round(s5, 1) if s5 is not None else None, "diverging": codisperse, "spread_bps": round(spread, 1)}, "medium" if codisperse and spread > 30 else "ok", note="idiosyncratic move" if codisperse else "co-moving"))

    # ---- BOOK stress (L1) ----
    spr = ind.get("spread_pct"); dep = ind.get("bid_depth_usd"); imb = ind.get("book_imbalance")
    book_fresh = ind.get("spread_pct") is not None
    if spr is not None:
        sev = "medium" if spr > 2e-5 else "ok"  # ~0.2bps normal; blowout if wide
        flags.append(_flag("book_state", {"spread_pct": spr, "bid_depth_$K": round((dep or 0) / 1e3), "imbalance": round(imb, 3) if imb is not None else None}, sev, fresh=book_fresh, note="book stress" if sev != "ok" else "normal book"))

    return flags


def run_once(conn, clock_conn=None):
    # Phase-1b SPLIT rule: a latest-clock probe (MAX(ts_ms)) must hit the LIVE file's index
    # fast path, never the union view -- on the union it degrades to a CO-ROUTINE COMPOUND
    # scan (measured 0.00s -> 2.16s). History-spanning reads keep using `conn` (the union).
    mx = _one((clock_conn or conn).cursor(),
              "SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (SYMBOL,))
    as_of = int(mx[0]) if mx and mx[0] else int(time.time() * 1000)
    flags = compute_flags(conn, as_of)
    overall = "ok"
    for f in flags:
        if SEV_RANK.get(f["severity"], 0) > SEV_RANK.get(overall, 0):
            overall = f["severity"]
    doc = {
        "as_of_ms": as_of,
        "as_of_utc": dt.datetime.fromtimestamp(as_of / 1000, dt.timezone.utc).isoformat(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "overall_severity": overall,
        "kind": "situational anomaly flags (TOOL, not a predictor; no edge — AUC~0.55)",
        "flags": flags,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    args = ap.parse_args()
    consecutive_failures = 0
    while True:
        # Union reader (live + frozen segments) for history-spanning reads, never a
        # hard-coded single file: post-rotation data/microstructure.db is the FROZEN file,
        # which published cutover-old market state under a fresh generated_utc. The
        # latest-clock probe gets a separate LIVE-only connection (Phase-1b split rule).
        # Pre-rotation both are the same file (no-op).
        # Opening is INSIDE the try: after a Phase-4 frozen-segment delete (or a malformed
        # rotation_state.json) open_union_ro raises, and an uncaught raise here would kill
        # this loop permanently -- nothing supervises/restarts this role.
        # RotationStateError alone is too narrow: open_union_ro checks os.path.exists and
        # only THEN attaches, so a truncated or mid-replace segment fails inside the
        # ATTACH as a plain sqlite3 error. This role also reaches 14 days back through
        # liq_indicator_library, so it is the one that can raise InsufficientHistoryError
        # once coverage assertions are wired -- it was hardened for neither.
        # run_once stays OUTSIDE the catch: swallowing a persistent error inside the cycle
        # would republish nothing while the process looks healthy.
        conn = clock_conn = None
        try:
            conn = open_union_ro()
            clock_conn = open_live_ro()
        except SURVIVABLE_ESTATE_ERRORS as exc:
            for c in (conn, clock_conn):
                if c is not None:
                    c.close()
            consecutive_failures += 1
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} ESTATE_UNAVAILABLE "
                  f"{type(exc).__name__}: {exc} "
                  f"(attempt {consecutive_failures}/{MAX_CONSECUTIVE_ESTATE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_ESTATE_FAILURES:
                raise
            if args.once:
                break
            time.sleep(args.interval_sec)
            continue
        consecutive_failures = 0
        try:
            doc = run_once(conn, clock_conn=clock_conn)
            print(f"{doc['generated_utc']} overall={doc['overall_severity']} flags={len(doc['flags'])} -> {OUT.name}")
        finally:
            for c in (conn, clock_conn):
                if c is not None:
                    c.close()
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
