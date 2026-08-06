"""
liq_tip_forward.py — FORWARD-LEDGER runner for the liquidation tip system (read-only).

Accumulates clean, UN-BURNED forward data: on each new liquidation anchor it records the full tip
fingerprint + anomaly flags (T0, causal), then backfills the realized mark-price outcomes as horizons
mature. This is the ONLY honest way to ever test the weak (AUC~0.55) signal — never backtest-mining the
burned data again. It also doubles as the compact per-event context capture that lets us prune raw
book_ticker later (800GB storage sustainability).

NOT an edge, NOT a trader — it observes and records. DB read-only (mode=ro, query_only=1); writes ONLY
its own append-only JSONL ledger + a small state file. No orders, no execution.

Ledger: reports/shadow/liq_tip_forward.jsonl   State: reports/shadow/liq_tip_forward_state.json
Loop: `--once` or `while True: run_once(); sleep(interval)`.
"""
from __future__ import annotations
import sqlite3, json, sys, time, argparse, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.liq_signal_system import signal_readout  # noqa: E402
from tools.liq_anomaly_monitor import compute_flags  # noqa: E402
from ami.storage.union_reader import (  # noqa: E402
    open_union_ro, RotationStateError, InsufficientHistoryError)

SURVIVABLE_ESTATE_ERRORS = (RotationStateError, InsufficientHistoryError, sqlite3.Error)
MAX_CONSECUTIVE_ESTATE_FAILURES = 20  # retrying a dead estate forever is itself the failure

LEDGER = ROOT / "reports" / "shadow" / "liq_tip_forward.jsonl"
STATE = ROOT / "reports" / "shadow" / "liq_tip_forward_state.json"

MIN_NOTIONAL = 200_000
MIN_GAP_MS = 900_000            # one anchor per 15 min (matches research dedup)
FRESH_MS = 120_000             # only fire on anchors ≤ 2 min old (live cadence, no backfill-flood)
LOOKBACK_MS = 3_600_000        # scan window for new anchors
HORIZONS = [("+5m", 300_000), ("+30m", 1_800_000), ("+1h", 3_600_000), ("+6h", 6 * 3_600_000)]
CLOSE_HORIZON_MS = 6 * 3_600_000


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
    """Restart-safety: re-derive already-OPENed anchor ids from the durable ledger."""
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


def _mark(cur, ts):
    r = cur.execute("SELECT mark_price, ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return r[0] if r else None


def _detect_anchors(cur, now_ms):
    """Fresh ETHUSDT liq anchors (either side) ≥ MIN_NOTIONAL, min-gap deduped, age ≤ FRESH_MS."""
    rows = cur.execute(
        "SELECT ts_ms,side FROM liquidations WHERE symbol='ETHUSDT' AND notional>=? "
        "AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms", (MIN_NOTIONAL, now_ms - LOOKBACK_MS, now_ms)).fetchall()
    anchors = []
    last = -10 ** 15
    for ts, side in rows:
        if ts - last >= MIN_GAP_MS:
            anchors.append((ts, side))
            last = ts
    # only the fresh ones (avoid re-processing history on first run)
    return [(ts, side) for ts, side in anchors if 0 <= now_ms - ts <= FRESH_MS]


def run_once(conn, st):
    cur = conn.cursor()
    mx = cur.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    now_ms = int(mx[0]) if mx and mx[0] else int(time.time() * 1000)
    processed = set(int(x) for x in st.get("processed", [])) | _processed_from_ledger()
    pending = {int(k): v for k, v in st.get("pending", {}).items()}
    opened = closed = 0

    # 1) detect + OPEN new anchors
    for ts, side in _detect_anchors(cur, now_ms):
        if ts in processed:
            continue
        entry = _mark(cur, ts)
        tip = signal_readout(conn, ts)
        anomaly = compute_flags(conn, ts)
        d_utc = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        # explicit regime tags for forward stratification (§161 session finding must be
        # confirmable forward, un-burned, per-session — never re-mined from burned history).
        session = "ASIA" if d_utc.hour < 7 else ("EU" if d_utc.hour < 13 else "US")
        _log({"event": "OPEN", "anchor_ts_ms": ts,
              "utc": d_utc.isoformat(),
              "session": session, "hour_utc": d_utc.hour, "month": d_utc.strftime("%Y-%m"),
              "side": side, "entry_mark": entry, "tip": tip, "anomaly": anomaly})
        processed.add(ts)
        pending[ts] = {"entry": entry, "outcomes": {}}
        opened += 1

    # 2) backfill matured outcomes; CLOSE when the longest horizon is done
    still = {}
    for ats, info in pending.items():
        entry = info.get("entry")
        oc = dict(info.get("outcomes", {}))
        for lbl, h in HORIZONS:
            if lbl not in oc and now_ms - ats >= h:
                m = _mark(cur, ats + h)
                oc[lbl] = round((m - entry) / entry * 1e4, 2) if (m and entry) else None
        if now_ms - ats >= CLOSE_HORIZON_MS:
            _log({"event": "CLOSE", "anchor_ts_ms": ats, "entry_mark": entry, "outcomes_bps": oc})
            closed += 1
        else:
            still[ats] = {"entry": entry, "outcomes": oc}

    st["processed"] = sorted(processed)[-5000:]  # bound
    st["pending"] = still
    _save_state(st)
    return opened, closed, len(still)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    args = ap.parse_args()
    st = _load_state()
    consecutive_failures = 0
    while True:
        # Union reader (live + frozen segments), never a hard-coded single file: the tip
        # fingerprint carries history-spanning context (14d funding percentile, prior 4h
        # trend) and outcome backfill must find marks on BOTH sides of a rotation cutover.
        # Post-rotation the old hard-coded data/microstructure.db was the FROZEN file, so
        # this ledger silently stopped accumulating at the cutover (no new anchor is ever
        # "<=2 min old" in a frozen file). Pre-rotation this is the same file (no-op).
        # Opened INSIDE the try: a Phase-4 frozen-segment delete (or malformed
        # rotation_state.json) makes open_union_ro raise, and an uncaught raise would kill
        # this ledger permanently -- nothing supervises/restarts this role.
        # Only the CONNECTION is retried, and the catch is wider than RotationStateError:
        # a truncated or mid-replace segment fails inside the ATTACH as a plain sqlite3
        # error, and this role reaches 14 days back through liq_indicator_library so it is
        # one of the two that can see InsufficientHistoryError. run_once stays outside --
        # swallowing a persistent error there would accumulate nothing while looking alive.
        conn = None
        try:
            conn = open_union_ro()
        except SURVIVABLE_ESTATE_ERRORS as exc:
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
            o, cl, p = run_once(conn, st)
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} opened={o} closed={cl} pending={p}")
        finally:
            conn.close()
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
