"""s34_leads_monitor_dashboard.py — two-lead live monitor (READ-ONLY, diagnostic).

Dedicated operator monitor for the two standing leads:
  1. echo_30_90 + regime (CAUSAL)  — forward paper via reports/shadow/echo_forward_ledger.jsonl
  2. LONG_HOUR17 (hold 6h)          — forward/backfill paper via reports/shadow/s34_state_machine_shadow.jsonl

Shows, per lead: open positions (while trading), closed trades, aggregates (N / WR / avg net /
total / tail), per-event detail, and the historical CAUSAL baseline card for honest comparison.

GUARDRAILS (this is a SECONDARY / diagnostic surface — does NOT supersede the canonical
tools/s34_cascade_navigation_dashboard.py :8770):
  * strictly READ-ONLY — HTTP GET/HEAD only; POST/PUT/PATCH/DELETE -> 405.
  * no trade/order/cancel/executor/scheduler/PROCESS control; no subprocess; no mutation.
  * binds loopback only.
  * any DB read is mode=ro + PRAGMA query_only=ON.
  * reads RAW ledgers directly — deliberately AVOIDS the SYSTEM_STATE §141 tainted adapters
    (dashboard/backend/adapters/shadow_paper_activity.py population-less pnl; freshness.py:101
    fabricated-GREEN). CLOSE rows are deduped by id; `backfill` (BACKFILL_SIMULATED) trades are
    labelled separately and NEVER summed into forward paper.

Not a trader. Numbers are PAPER unless a real live executor position is present (labelled REAL).

Usage:
  python -m tools.s34_leads_monitor_dashboard --once                 # self-test, prints payload
  python -m tools.s34_leads_monitor_dashboard --serve --serve-port 8771
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ECHO_LEDGER = ROOT / "reports" / "shadow" / "echo_forward_ledger.jsonl"
ECHO_STATE = ROOT / "reports" / "shadow" / "echo_forward_ledger_state.json"
SM_LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
CAUSAL_JSON = ROOT / "reports" / "research" / "s34" / "S34_ECHO_CAUSAL_VS_LOOKAHEAD.json"
HOLD_SWEEP_JSON = ROOT / "reports" / "research" / "s34" / "S34_HOLD_HORIZON_SWEEP.json"
HOLD_FWD_LEDGER = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"

TAIL_BPS = -100.0
HOLD_HORIZONS = [2, 4, 6, 12, 24, 48]
EVENT_PRIMARY_H = {"echo": 4, "hour17": 6}   # scoreboard reference horizon per signal
BASE_TAIL_RATE_REF = 12.0                    # ~echo base-rate tail % (§168 refutation reference)
HOUR17_SIGNAL_PREFIX = "LONG_HOUR17"
HOUR17_INTENDED_HOLD_H = 6.0
# A hold materially longer than the intended 6h is a force-close / ledger-gap OUTAGE ARTIFACT
# (SYSTEM_STATE §141 / DIRECT_SHORT_AUDIT: a 46.4h gap force-closed 3 positions → fake +900bps).
# These are quarantined out of the forward aggregate, never summed into the headline.
HOUR17_GAP_HOLD_H = 7.0

# Static historical context for hour17 (source reports; parsing markdown is brittle, so the
# reconciled headlines are pinned here WITH their source paths — display only, never an edge claim).
HOUR17_CONTEXT = {
    "historical_cycle": {
        "cycles": 93, "wr": 62.4, "mean_bps": 32.47, "cum_bps": 3019.4,
        "verdict": "POSITIVE_BUT_FRAGILE",
        "source": "reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md",
    },
    "forward_audit": {
        "hold": "6h", "alpha_bps": -41.0, "ci": [-96.0, 15.0], "wr": 25.0, "n_clusters": 7,
        "verdict": "FORWARD_DOES_NOT_SUPPORT_HISTORICAL",
        "source": "reports/research/s34/S34_HOUR17_DIRECT_SHORT_AUDIT_2026-07-17.md",
    },
    "note": "Microstructure/indicator filtering already tried (2026-07-17) and FAILED "
            "(MICROSTRUCTURE_FEATURES_NOT_OOS_STABLE, permutation p=0.76); root cause = "
            "INSUFFICIENT_INDEPENDENT_EVENTS. Only more FORWARD data helps.",
}


# ── utilities ─────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return out
    return out


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_eth_mark() -> dict[str, Any] | None:
    """Latest ETHUSDT mark for unrealized bps on open positions. Read-only, mode=ro, fail-soft."""
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only=1")
            row = conn.execute(
                "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' "
                "ORDER BY ts_ms DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return {"ts_ms": int(row[0]), "price": float(row[1])}


def _agg(nets: list[float], months: float | None = None) -> dict[str, Any]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "wr": None, "avg": None, "total": None, "worst": None,
                "tail_n": 0, "per_month": None}
    wins = sum(1 for x in nets if x > 0)
    tail = sum(1 for x in nets if x <= TAIL_BPS)
    total = sum(nets)
    return {
        "n": n,
        "wr": round(100.0 * wins / n, 1),
        "avg": round(total / n, 1),
        "total": round(total, 1),
        "worst": round(min(nets), 1),
        "tail_n": tail,
        "per_month": (round(n / months, 1) if months else None),
    }


# ── echo lead ─────────────────────────────────────────────────────────────────

def load_echo(mark: dict[str, Any] | None = None) -> dict[str, Any]:
    recs = _read_jsonl(ECHO_LEDGER)
    opens: dict[int, dict[str, Any]] = {}
    closes: dict[int, dict[str, Any]] = {}
    for r in recs:
        ats = r.get("anchor_ts_ms")
        if ats is None:
            continue
        ats = int(ats)
        if r.get("event") == "OPEN":
            opens[ats] = r
        elif r.get("event") == "CLOSE":
            closes[ats] = r  # dedup: last CLOSE per anchor wins

    closed_trades = []
    for ats, c in sorted(closes.items()):
        o = opens.get(ats, {})
        closed_trades.append({
            "id": str(ats),
            "utc": o.get("utc") or _iso(ats),
            "net_bps": c.get("net_bps"),
            "qualified_t0": c.get("qualified_t0"),
            "qualified_full": c.get("qualified_full"),
            "noisy_T30m": c.get("noisy_T30m"),
            "entry_mark": c.get("entry_mark"),
            "exit_mark": c.get("exit_mark"),
            "session": o.get("session"),
            "hour_utc": o.get("hour_utc"),
            "echo_30_90": o.get("echo_30_90"),
            "path_min_bps": c.get("path_min_bps"),
            "indicators": o.get("indicators"),
        })

    open_positions = []
    for ats, o in sorted(opens.items()):
        if ats in closes:
            continue
        entry = o.get("entry_mark")
        unreal = None
        if mark and entry:
            try:
                unreal = round((mark["price"] - float(entry)) / float(entry) * 1e4, 1)
            except (TypeError, ValueError, ZeroDivisionError):
                unreal = None
        open_positions.append({
            "id": str(ats),
            "utc": o.get("utc") or _iso(ats),
            "direction": "LONG",
            "entry_mark": entry,
            "elapsed_min": round((_now_ms() - ats) / 60000.0, 1),
            "target_hold_h": 4,
            "unrealized_bps": unreal,
            "qualified_t0": o.get("qualified_t0"),
            "session": o.get("session"),
            "hour_utc": o.get("hour_utc"),
            "echo_30_90": o.get("echo_30_90"),
            "indicators": o.get("indicators"),
        })

    # qualified_t0 (causal echo, tradeable) vs the NON-qualified CONTROL — if the gate adds value
    # the qualified arm must beat the control forward; if the control wins just as much, the "edge"
    # is the bare cascade-bounce, not the echo/regime selection. (operator Q 2026-07-20)
    nets_qual = [t["net_bps"] for t in closed_trades
                 if t.get("net_bps") is not None and t.get("qualified_t0")]
    nets_ctrl = [t["net_bps"] for t in closed_trades
                 if t.get("net_bps") is not None and not t.get("qualified_t0")]
    nets_full = [t["net_bps"] for t in closed_trades
                 if t.get("net_bps") is not None and t.get("qualified_full")]

    return {
        "name": "echo_30_90 + regime (CAUSAL)",
        "kind": "FORWARD_PAPER",
        "hold_label": "T0 · hold 4h · FEE 5bps",
        "open_positions": open_positions,
        "closed_trades": list(reversed(closed_trades)),
        "agg_forward_t0": _agg(nets_qual),          # qualified_t0 ONLY (was: pooled all — fixed)
        "agg_control_nonqual": _agg(nets_ctrl),     # NON-qualified control group
        "agg_forward_full": _agg(nets_full),
        "context": _echo_context(),
        "empty_note": (None if closed_trades or open_positions else
                       "0 forward event — 2026-07-20 sonrası anchor bekleniyor "
                       "(discovery Feb–Jul burned; forward-only)."),
    }


def _echo_context() -> dict[str, Any]:
    data = _read_json(CAUSAL_JSON)
    if not isinstance(data, dict):
        return {"available": False}
    arms = (data.get("arms") or {}).get("T0 hold 4h") or {}
    causal = arms.get("causal_no_lookahead") or {}
    full = arms.get("full_with_lookahead") or {}
    return {
        "available": True,
        "n_anchors": data.get("n_anchors"),
        "months": data.get("months"),
        "frame": data.get("frame"),
        "causal": {k: causal.get(k) for k in
                   ("n", "wr", "avg", "worst", "tail_n", "per_month", "mc_p", "wf",
                    "noov_n", "noov_wr", "noov_per_month")},
        "full": {k: full.get(k) for k in
                 ("n", "wr", "avg", "worst", "tail_n", "per_month", "mc_p", "wf")},
        "source": "reports/research/s34/S34_ECHO_CAUSAL_VS_LOOKAHEAD.json",
    }


# ── hour17 lead ───────────────────────────────────────────────────────────────

def load_hour17(mark: dict[str, Any] | None = None) -> dict[str, Any]:
    recs = _read_jsonl(SM_LEDGER)
    opens: dict[str, dict[str, Any]] = {}
    closed_by_id: dict[str, dict[str, Any]] = {}
    terminal_ids: set[str] = set()
    for r in recs:
        sig = str(r.get("signal") or "")
        if not sig.startswith(HOUR17_SIGNAL_PREFIX):
            continue
        rid = str(r.get("id") or r.get("anchor_ts_ms") or "")
        if not rid:
            continue
        ev = r.get("event")
        if ev == "OPEN" or (ev is None and r.get("status") == "OPEN"):
            opens[rid] = r
        elif ev == "CLOSE":
            closed_by_id[rid] = r  # dedup by id
            terminal_ids.add(rid)
        elif ev in ("EXPIRE",):
            terminal_ids.add(rid)

    closed_trades = []
    for rid, c in closed_by_id.items():
        net = c.get("net_bps")
        if net is None:
            continue
        entry_ts = c.get("entry_ts_ms")
        exit_ts = c.get("exit_ts_ms")
        hold_h = None
        if entry_ts and exit_ts:
            hold_h = round((int(exit_ts) - int(entry_ts)) / 3600000.0, 2)
        gap_artifact = hold_h is not None and hold_h > HOUR17_GAP_HOLD_H
        variant = str(c.get("signal") or "").replace("LONG_HOUR17_", "") or "?"
        closed_trades.append({
            "id": rid,
            "signal": c.get("signal"),
            "variant": variant,
            "utc": c.get("opened_utc") or _iso(c.get("entry_ts_ms")),
            "closed_utc": c.get("closed_utc") or _iso(c.get("exit_ts_ms")),
            "net_bps": round(float(net), 1),
            "close_reason": c.get("close_reason") or c.get("status"),
            "entry_price": c.get("entry_price"),
            "exit_price": c.get("exit_price"),
            "hold_h": hold_h,
            "backfill": bool(c.get("backfill")),
            "gap_artifact": gap_artifact,
            "session": c.get("session"),
            "hour": c.get("hour"),
            "btc4h_bps": c.get("btc4h_bps"),
            "running_notional": c.get("running_notional"),
        })
    closed_trades.sort(key=lambda t: t.get("closed_utc") or "")

    open_positions = []
    for rid, o in opens.items():
        if rid in terminal_ids:
            continue
        entry = o.get("entry_price")
        unreal = None
        if mark and entry:
            try:
                unreal = round((mark["price"] - float(entry)) / float(entry) * 1e4, 1)
            except (TypeError, ValueError, ZeroDivisionError):
                unreal = None
        entry_ts = o.get("entry_ts_ms") or o.get("anchor_ts_ms")
        exit_due = o.get("exit_due_ms")
        open_positions.append({
            "id": rid,
            "signal": o.get("signal"),
            "direction": o.get("direction") or "LONG",
            "utc": o.get("opened_utc") or _iso(entry_ts),
            "entry_price": entry,
            "elapsed_min": (round((_now_ms() - int(entry_ts)) / 60000.0, 1) if entry_ts else None),
            "exit_due_utc": _iso(exit_due),
            "unrealized_bps": unreal,
            "session": o.get("session"),
            "hour": o.get("hour"),
            "btc4h_bps": o.get("btc4h_bps"),
            "sync_k": o.get("sync_k"),
            "long_score": o.get("long_score"),
        })

    clean = [t for t in closed_trades if not t["backfill"] and not t["gap_artifact"]]
    hold6 = [t["net_bps"] for t in clean if t["variant"] == "HOLD6H"]
    comp = [t["net_bps"] for t in clean if t["variant"] == "COMPOSITE"]
    gap = [t["net_bps"] for t in closed_trades if t["gap_artifact"]]
    bfl = [t["net_bps"] for t in closed_trades if t["backfill"]]
    n_gap = len(gap)
    return {
        "name": "LONG_HOUR17 (hold 6h)",
        "kind": "FORWARD_PAPER",
        "hold_label": "T0 · hold 6h · stop 300bps · FEE 5bps",
        "open_positions": open_positions,
        "closed_trades": list(reversed(closed_trades)),
        "agg_forward": _agg(hold6),            # headline = clean HOLD6H forward only
        "agg_composite": _agg(comp),           # COMPOSITE overlay variant, shown separately
        "agg_gap": _agg(gap),                  # outage/force-close artifacts, QUARANTINED
        "agg_backfill": _agg(bfl),
        "context": HOUR17_CONTEXT,
        "quarantine_note": (f"{n_gap} outage/gap artifact (hold>{HOUR17_GAP_HOLD_H:g}h, sahte +900bps) "
                            f"forward toplamından ÇIKARILDI (§141)." if n_gap else None),
        "empty_note": (None if closed_trades or open_positions else
                       "hour17 route için henüz shadow trade yok."),
    }


# ── live (real) executor ─────────────────────────────────────────────────────

def load_live() -> dict[str, Any]:
    st = _read_json(LIVE_STATE)
    if not isinstance(st, dict):
        return {"available": False}
    active = st.get("active")
    status = st.get("status") if isinstance(st.get("status"), dict) else {}
    return {
        "available": True,
        "mode": status.get("mode"),
        "active": active if isinstance(active, dict) else None,
        "has_real_position": isinstance(active, dict) and active.get("status") == "POSITION_OPEN",
        "new_entry_blocked_by": status.get("new_entry_blocked_by"),
        "updated_at_utc": status.get("updated_at_utc"),
        "orders_n": len(st.get("orders") or []),
        "note": "REAL trades appear here only when start_eclipse.ps1 -EnableLive is on. Today: paper only.",
    }


# ── hold-horizon sweep (historical) + forward ledger ──────────────────────────

def load_hold_horizons() -> dict[str, Any]:
    """Historical causal hold-horizon sweep + forward paper accumulation per horizon."""
    hist = _read_json(HOLD_SWEEP_JSON)
    fwd = _hold_forward_agg()
    return {
        "historical": hist if isinstance(hist, dict) else {"available": False},
        "forward": fwd,
        "note": "avg uzun hold'da yükselir AMA worst/tail patlar + bağımsız-N çöker; "
                "'CAN describe CANNOT bless'. +900@48h = outage artifact (§166), edge değil.",
    }


def _hold_forward_agg() -> dict[str, Any]:
    """Aggregate the forward paper hold-horizon ledger (per signal × horizon), if present."""
    recs = _read_jsonl(HOLD_FWD_LEDGER)
    if not recs:
        return {"available": False, "note": "forward ledger boş — ufuklar ileriye doğru birikecek."}
    buckets: dict[str, list[float]] = {}
    for r in recs:
        if r.get("event") != "RESOLVE":
            continue
        net = r.get("net_bps")
        h = r.get("hold_h")
        if net is None or h is None:
            continue
        net = float(net)
        hr = r.get("hour_utc")
        # hour17: qualified vs NON-qualified CONTROL
        key = f"hour17|h{h}" if r.get("qualified_hour17") else f"hour17_ctrl|h{h}"
        buckets.setdefault(key, []).append(net)
        # echo_causal: qualified vs control, plus the echo∩hour>=17 tail-gate slice (§168)
        if r.get("qualified_echo"):
            buckets.setdefault(f"echo_causal|h{h}", []).append(net)
            if hr is not None and hr >= 17:
                buckets.setdefault(f"echo_hi|h{h}", []).append(net)
        else:
            buckets.setdefault(f"echo_ctrl|h{h}", []).append(net)
    return {"available": True,
            "arms": {k: _agg(v) for k, v in sorted(buckets.items())}}


# ── event feed + tail-rate scoreboard (per-event monitor) ─────────────────────

def load_event_feed() -> dict[str, Any]:
    """Per-event tape from the hold-horizon forward ledger + a tail-RATE scoreboard.

    §169 lesson: the correct forward metric is TAIL-RATE (not avg). Tail-rate is measured on the
    NOSTOP net (intrinsic downtrend-continuation), per group, at each signal's primary horizon.
    The stop only redistributes tails into a −305 wall, so it does not answer the survival question.
    """
    recs = _read_jsonl(HOLD_FWD_LEDGER)
    if not recs:
        return {"available": False, "events": [], "scoreboard": {},
                "note": "forward ledger boş — olaylar ileriye doğru birikecek."}
    opens: dict[int, dict[str, Any]] = {}
    resolves: dict[tuple[int, int], dict[str, Any]] = {}
    for r in recs:
        ats = r.get("anchor_ts_ms")
        if ats is None:
            continue
        ats = int(ats)
        if r.get("event") == "OPEN":
            opens[ats] = r
        elif r.get("event") == "RESOLVE" and r.get("hold_h") is not None:
            resolves[(ats, int(r["hold_h"]))] = r

    events = []
    for ats, o in opens.items():
        res: dict[int, Any] = {}
        for h in HOLD_HORIZONS:
            rr = resolves.get((ats, h))
            if rr and rr.get("net_bps") is not None:
                net = float(rr["net_bps"])
                res[h] = {"net": round(net, 1), "s300": rr.get("net_bps_s300"),
                          "tail": net <= TAIL_BPS}
            else:
                res[h] = None
        n_res = sum(1 for h in HOLD_HORIZONS if res[h] is not None)
        hr = o.get("hour_utc")
        events.append({
            "id": str(ats), "utc": o.get("utc") or _iso(ats), "hour_utc": hr,
            "qualified_hour17": bool(o.get("qualified_hour17")),
            "qualified_echo": bool(o.get("qualified_echo")),
            "echo_hi": bool(o.get("qualified_echo")) and hr is not None and hr >= 17,
            "running_notional": o.get("running_notional"), "entry_mark": o.get("entry_mark"),
            "res": {str(h): res[h] for h in HOLD_HORIZONS},
            "n_resolved": n_res, "open": n_res < len(HOLD_HORIZONS),
            "any_tail": any(res[h] and res[h]["tail"] for h in HOLD_HORIZONS),
        })
    events.sort(key=lambda e: int(e["id"]), reverse=True)  # newest first

    def rate(pred, h: int) -> dict[str, Any]:
        rows = [e for e in events if pred(e) and e["res"].get(str(h))]
        n = len(rows)
        t = sum(1 for e in rows if e["res"][str(h)]["tail"])
        return {"n": n, "tail": t, "rate": (round(100 * t / n, 1) if n else None)}

    eh, hh = EVENT_PRIMARY_H["echo"], EVENT_PRIMARY_H["hour17"]
    scoreboard = {
        "primary_h": EVENT_PRIMARY_H, "base_rate_ref": BASE_TAIL_RATE_REF,
        "echo_qual": rate(lambda e: e["qualified_echo"], eh),
        "echo_ctrl": rate(lambda e: not e["qualified_echo"], eh),
        "echo_hi": rate(lambda e: e["echo_hi"], eh),
        "hour17_qual": rate(lambda e: e["qualified_hour17"], hh),
        "hour17_ctrl": rate(lambda e: not e["qualified_hour17"], hh),
    }
    return {"available": True, "n_events": len(events), "horizons": HOLD_HORIZONS,
            "events": events[:120], "scoreboard": scoreboard}


# ── payload ───────────────────────────────────────────────────────────────────

def build_payload() -> dict[str, Any]:
    mark = _latest_eth_mark()
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "eth_mark": mark,
        "leads": {"echo": load_echo(mark), "hour17": load_hour17(mark)},
        "hold_horizons": load_hold_horizons(),
        "event_feed": load_event_feed(),
        "live": load_live(),
        "contract": {"read_only": True, "control_actions_available": False,
                     "db_mode": "ro", "kind": "SECONDARY_DIAGNOSTIC_NOT_CANONICAL"},
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

def render_html() -> str:
    return """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Eclipse · İki-Lead Monitör</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#111725; --panel2:#0e1420; --line:#1e2838;
    --ink:#e6edf3; --dim:#8b98a9; --dim2:#5c6b7e;
    --pos:#3fb950; --neg:#f85149; --warn:#d29922; --acc:#58a6ff; --acc2:#bc8cff;
    --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box} html,body{margin:0}
  body{background:radial-gradient(1200px 600px at 20% -10%,#131c2b 0%,var(--bg) 55%);
    color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased}
  a{color:var(--acc)}
  header{display:flex;align-items:baseline;gap:14px;padding:18px 22px 10px;flex-wrap:wrap}
  header h1{font-size:16px;font-weight:650;margin:0;letter-spacing:.3px}
  .sub{color:var(--dim);font-size:12px}
  .mark{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--dim)}
  .mark b{color:var(--ink)}
  .contract{font-size:10.5px;color:var(--dim2);border:1px solid var(--line);border-radius:999px;
    padding:2px 9px;font-family:var(--mono)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:8px 18px 30px}
  @media(max-width:940px){.grid{grid-template-columns:1fr}}
  .lead{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
    border:1px solid var(--line);border-radius:14px;overflow:hidden}
  .lead>.hd{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line)}
  .lead>.hd .nm{font-weight:650;font-size:14.5px}
  .badge{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:6px;letter-spacing:.4px}
  .b-paper{background:#1b2436;color:var(--acc);border:1px solid #23324b}
  .b-real{background:#12261a;color:var(--pos);border:1px solid #1d3a26}
  .b-bf{background:#2a2417;color:var(--warn);border:1px solid #3d3520}
  .hold{margin-left:auto;color:var(--dim);font-size:11px;font-family:var(--mono)}
  .sec{padding:12px 16px;border-bottom:1px solid var(--line)}
  .sec:last-child{border-bottom:0}
  .sec h3{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--dim)}
  .tiles{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
  @media(max-width:520px){.tiles{grid-template-columns:repeat(3,1fr)}}
  .tile{background:#0c121d;border:1px solid var(--line);border-radius:9px;padding:8px 9px}
  .tile .k{font-size:9.5px;color:var(--dim2);text-transform:uppercase;letter-spacing:.6px}
  .tile .v{font-family:var(--mono);font-size:16px;font-weight:600;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}
  th{text-align:right;color:var(--dim2);font-weight:500;padding:4px 6px;border-bottom:1px solid var(--line);
    font-size:10px;text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0;background:var(--panel2)}
  th:first-child,td:first-child{text-align:left}
  td{padding:5px 6px;border-bottom:1px solid #141b28;white-space:nowrap}
  tr:hover td{background:#0e1622}
  .scroll{max-height:260px;overflow:auto}
  .pos{color:var(--pos)} .neg{color:var(--neg)} .dim{color:var(--dim)}
  .open-row{display:flex;gap:14px;flex-wrap:wrap;background:#0c1420;border:1px solid #17324a;
    border-radius:9px;padding:10px 12px;margin-bottom:8px}
  .open-row .f{display:flex;flex-direction:column} .open-row .f .k{font-size:9px;color:var(--dim2);text-transform:uppercase}
  .open-row .f .v{font-family:var(--mono);font-size:13px}
  .pulse{width:7px;height:7px;border-radius:50%;background:var(--pos);box-shadow:0 0 0 0 rgba(63,185,80,.6);
    animation:p 1.8s infinite;align-self:center}
  @keyframes p{0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}70%{box-shadow:0 0 0 7px rgba(63,185,80,0)}100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
  .ctx{font-size:11.5px;color:var(--dim)}
  .ctx .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed #172030;font-family:var(--mono)}
  .ctx .row:last-child{border-bottom:0}
  .ctx b{color:var(--ink)} .warnc{color:var(--warn)}
  .empty{color:var(--dim2);font-style:italic;font-size:12px;padding:6px 0}
  #holds{margin:2px 18px 8px;background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
    border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  #holds h2{margin:0 0 4px;font-size:13px;font-weight:650}
  #holds .hint{color:var(--dim);font-size:11px;margin-bottom:10px}
  .hcols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:860px){.hcols{grid-template-columns:1fr}}
  .hcols h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--acc2)}
  #feed{margin:2px 18px 8px;background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
    border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  #feed h2{margin:0 0 4px;font-size:13px;font-weight:650}
  #feed .hint{color:var(--dim);font-size:11px;margin-bottom:10px}
  .score{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px}
  @media(max-width:760px){.score{grid-template-columns:repeat(2,1fr)}}
  .scard{background:#0c121d;border:1px solid var(--line);border-radius:9px;padding:8px 10px}
  .scard .k{font-size:9.5px;color:var(--dim2);text-transform:uppercase;letter-spacing:.5px}
  .scard .r{font-family:var(--mono);font-size:17px;font-weight:600;margin-top:2px}
  .scard .sub{font-family:var(--mono);font-size:10px;color:var(--dim)}
  .scard.ctrl{border-style:dashed}
  .grid6{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;min-width:210px}
  .cell{font-family:var(--mono);font-size:10px;text-align:center;border-radius:4px;padding:3px 2px;background:#0c121d;border:1px solid #141b28}
  .cell .hh{font-size:8px;color:var(--dim2)}
  .cell.tail{background:#2a1518;border-color:#4d2126;color:var(--neg)}
  .cell.win{color:var(--pos)} .cell.lose{color:var(--neg)} .cell.pend{color:var(--dim2)}
  .evrow{display:flex;gap:10px;align-items:center;padding:6px 4px;border-bottom:1px solid #141b28;flex-wrap:wrap}
  .evrow.tailrow{background:#180f12}
  .chip{font-family:var(--mono);font-size:9.5px;padding:1px 6px;border-radius:5px;border:1px solid}
  .c-q{background:#12261a;color:var(--pos);border-color:#1d3a26}
  .c-ctrl{background:#1a1d24;color:var(--dim);border-color:#2a3140}
  .c-hi{background:#1b1630;color:var(--acc2);border-color:#2f2650}
  .foot{padding:6px 22px 20px;color:var(--dim2);font-size:11px;font-family:var(--mono)}
  .sig{color:var(--acc2)}
  #banner{margin:6px 18px 2px;padding:8px 14px;border-radius:10px;font-family:var(--mono);
    font-size:12px;font-weight:600;letter-spacing:.3px;border:1px solid}
  .ban-paper{background:#2a2417;color:var(--warn);border-color:#3d3520}
  .ban-real{background:#2a1518;color:var(--neg);border-color:#4d2126}
</style></head>
<body>
<header>
  <h1>ECLIPSE · İki-Lead Monitör</h1>
  <span class="sub" id="ts">yükleniyor…</span>
  <span class="contract">READ-ONLY · no control · db=ro</span>
  <span class="mark" id="mark"></span>
</header>
<div id="banner"></div>
<div class="grid" id="grid"></div>
<div id="holds"></div>
<div id="feed"></div>
<div class="foot" id="foot"></div>
<script>
const f=(x,d=1)=>x==null?'—':Number(x).toFixed(d);
const cls=x=>x==null?'dim':(x>0?'pos':(x<0?'neg':''));
const sgn=x=>x==null?'—':(x>0?'+':'')+f(x);
function tiles(a){const t=[['N',a.n],['WR%',a.wr],['avg',a.avg==null?null:sgn(a.avg)],
  ['total',a.total==null?null:sgn(a.total,0)],['tail',a.tail_n],['/ay',a.per_month]];
  return '<div class="tiles">'+t.map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v ${k=='avg'||k=='total'?cls(a.avg):(k=='tail'&&v>0?'neg':'')}">${v==null?'—':v}</div></div>`).join('')+'</div>';}
function openBlk(p,isEcho){
  if(!p.length) return '<div class="empty">açık pozisyon yok</div>';
  return p.map(o=>{
    const fields=[['yön',o.direction],['giriş',isEcho?f(o.entry_mark,2):f(o.entry_price,2)],
      ['geçen',(o.elapsed_min==null?'—':f(o.elapsed_min,0)+'m')],
      ['unreal',`<span class="${cls(o.unrealized_bps)}">${sgn(o.unrealized_bps)}bps</span>`],
      ['hedef',isEcho?'4h':(o.exit_due_utc?o.exit_due_utc.slice(11,16)+'Z':'6h')],
      [isEcho?'q_t0':'score',isEcho?String(o.qualified_t0):String(o.long_score??'—')]];
    return '<div class="open-row"><span class="pulse"></span>'+
      fields.map(([k,v])=>`<div class="f"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('')+
      `<div class="f"><span class="k">sess</span><span class="v">${o.session||'—'} ${(o.hour_utc??o.hour)??''}</span></div></div>`;
  }).join('');
}
function echoRows(t){return t.map(r=>`<tr>
  <td>${(r.utc||'').slice(5,16).replace('T',' ')}</td>
  <td class="${cls(r.net_bps)}">${sgn(r.net_bps)}</td>
  <td class="${r.qualified_full?'pos':'dim'}">${r.qualified_full?'full':'t0'}</td>
  <td class="${r.noisy_T30m?'warnc':'dim'}">${r.noisy_T30m?'noisy':'—'}</td>
  <td>${r.session||'—'}</td><td class="dim">${r.echo_30_90?'echo':'—'}</td>
  <td class="${cls(r.path_min_bps)}">${sgn(r.path_min_bps)}</td></tr>`).join('');}
function h17tag(r){return r.gap_artifact?'<span class="badge b-bf">GAP</span>':
  (r.backfill?'<span class="badge b-bf">BF</span>':'<span class="badge b-paper">'+(r.variant==='COMPOSITE'?'comp':'fwd')+'</span>');}
function h17Rows(t){return t.map(r=>`<tr style="${r.gap_artifact?'opacity:.5':''}">
  <td>${(r.closed_utc||r.utc||'').slice(5,16).replace('T',' ')}</td>
  <td class="${r.gap_artifact?'dim':cls(r.net_bps)}">${sgn(r.net_bps)}</td>
  <td class="dim">${(r.close_reason||'').slice(0,14)}</td>
  <td class="${r.gap_artifact?'neg':''}">${f(r.hold_h,1)}h</td>
  <td>${h17tag(r)}</td>
  <td>${r.session||'—'} ${r.hour??''}</td>
  <td class="${cls(r.btc4h_bps)}">${sgn(r.btc4h_bps,0)}</td></tr>`).join('');}
function echoCtx(c){if(!c||!c.available) return '<div class="empty">bağlam yok</div>';
  const q=c.causal,fu=c.full;
  return `<div class="ctx"><div class="dim" style="margin-bottom:6px">geçmiş ${c.n_anchors} anchor · ${c.months} ay · <b>CAN kill / CANNOT bless</b></div>
  <div class="row"><span>CAUSAL (lookahead'sız) 4h</span><span><b class="${q.avg>0?'pos':'neg'}">${sgn(q.avg)}</b> · N${q.n} · WR${q.wr} · tail<b class="neg">${q.tail_n}</b> · mc${q.mc_p} · ${q.wf}</span></div>
  <div class="row"><span>FULL (lookahead'lı) 4h</span><span><b class="pos">${sgn(fu.avg)}</b> · N${fu.n} · WR${fu.wr} · tail${fu.tail_n}</span></div>
  <div class="row"><span>no-overlap CAUSAL</span><span>N${q.noov_n} · WR${q.noov_wr} · ${q.noov_per_month}/ay</span></div>
  <div class="dim" style="margin-top:6px;font-size:10.5px">forward, causal echo'nun noisy-gate olmadan yaşayıp yaşamadığını gösterecek</div></div>`;}
function h17Ctx(c){if(!c) return '';const h=c.historical_cycle,a=c.forward_audit;
  return `<div class="ctx">
  <div class="row"><span>geçmiş cycle-adj</span><span><b class="pos">+${h.mean_bps}</b> · ${h.cycles}cyc · WR${h.wr} · <b class="warnc">${h.verdict}</b></span></div>
  <div class="row"><span>forward audit ${a.hold}</span><span><b class="neg">${a.alpha_bps}</b> alpha · CI[${a.ci[0]},${a.ci[1]}] · WR${a.wr} · n${a.n_clusters}</span></div>
  <div class="dim" style="margin-top:6px;font-size:10.5px">${c.note}</div></div>`;}
function leadCard(L,key){
  const isEcho=key==='echo';
  const fwdAgg=isEcho?L.agg_forward_t0:L.agg_forward;
  const rows=isEcho?echoRows(L.closed_trades):h17Rows(L.closed_trades);
  const th=isEcho?'<tr><th>zaman</th><th>net</th><th>qual</th><th>noisy</th><th>sess</th><th>echo</th><th>pathMin</th></tr>'
                 :'<tr><th>kapanış</th><th>net</th><th>neden</th><th>hold</th><th>tür</th><th>sess</th><th>btc4h</th></tr>';
  const ctx=isEcho?echoCtx(L.context):h17Ctx(L.context);
  const secondAgg=isEcho
    ?`<div class="dim" style="font-size:10.5px;margin-top:6px">qualified_full (lookahead) alt-küme: N${L.agg_forward_full.n} · ${sgn(L.agg_forward_full.avg)} avg</div>
      <div class="dim" style="font-size:10.5px;margin-top:3px">KONTROL (q_t0=false, nitelenmedi): <b class="${cls((L.agg_control_nonqual||{}).avg)}">N${(L.agg_control_nonqual||{}).n||0} · ${sgn((L.agg_control_nonqual||{}).avg)} avg</b> — gate değer katıyorsa <b>qualified &gt; kontrol</b> olmalı</div>`
    :(L.agg_backfill.n?`<div class="dim" style="font-size:10.5px;margin-top:6px">BACKFILL_SIMULATED (forward'a toplanmaz): N${L.agg_backfill.n} · ${sgn(L.agg_backfill.avg)} avg</div>`:'');
  return `<div class="lead">
    <div class="hd"><span class="nm">${L.name}</span>
      <span class="badge b-paper">${L.kind}</span>
      <span class="hold">${L.hold_label}</span></div>
    <div class="sec"><h3>Açık pozisyon · PAPER (simülasyon — borsaya emir YOK)</h3>${openBlk(L.open_positions,isEcho)}</div>
    <div class="sec"><h3>Forward paper · toplamlar</h3>${tiles(fwdAgg)}${secondAgg}
      ${L.empty_note?`<div class="empty">${L.empty_note}</div>`:''}</div>
    <div class="sec"><h3>Kapanmış trade'ler (${L.closed_trades.length})</h3>
      <div class="scroll"><table>${th}${rows||''}</table></div></div>
    <div class="sec"><h3>Tarihsel bağlam (edge iddiası DEĞİL)</h3>${ctx}</div>
  </div>`;
}
function holdTable(sig,H){
  const hist=(H&&H.historical)||{};
  if(!hist.signals||!hist.signals[sig]) return '<div class="empty">sweep verisi yok — research_s34_hold_horizon_sweep çalıştır</div>';
  const S=hist.signals[sig]; const hs=hist.horizons_h||[2,4,6,12,24,48];
  const fwd=((H.forward&&H.forward.arms)||{});
  let rows=hs.map(h=>{const v=S['h'+h]||{};const ns=v.nostop||{};const s3=v.s300||{};const fa=fwd[sig+'|h'+h];
    const best=(h===6);
    return `<tr style="${best?'background:#0e1a12':''}">
     <td>${h}h${best?' ★':''}</td>
     <td class="dim">${ns.noov_n??'—'}</td>
     <td class="${cls(ns.avg)}">${sgn(ns.avg,0)}</td>
     <td class="${cls(s3.avg)}">${sgn(s3.avg,0)}</td>
     <td class="${ns.worst<-400?'neg':'dim'}">${sgn(ns.worst,0)}</td>
     <td class="dim">${sgn(s3.worst,0)}</td>
     <td class="${s3.tail_n>25?'neg':'dim'}">${s3.tail_n??'—'}</td>
     <td class="dim">${ns.mc_p??'—'}</td>
     <td class="${fa?cls(fa.avg):'dim'}">${fa?sgn(fa.avg)+'·n'+fa.n:'—'}</td></tr>`;}).join('');
  return `<table><tr><th>hold</th><th>bağN</th><th>avg°</th><th>avg₃₀₀</th><th>worst°</th><th>w₃₀₀</th><th>tail₃₀₀</th><th>mc</th><th>FWD</th></tr>${rows}</table>
   <div class="dim" style="font-size:10px;margin-top:4px">°=stop yok · ₃₀₀=−300bps stop · ★=6h (en az kötü; −150 whipsaw §163)</div>`;
}
function holdsPanel(d){
  const H=d.hold_horizons; const hist=(H&&H.historical)||{};
  const meta=hist.available===false?'':`geçmiş ${hist.n_anchors||'?'} anchor · ${hist.months||'?'} ay · funding≈${hist.funding_bps_per_8h??'?'}bps/8h`;
  document.getElementById('holds').innerHTML=
    `<h2>Hold-Ufku Süpürmesi · 2/4/6/12/24/48h <span class="badge b-paper">CAUSAL · tarihsel</span></h2>
     <div class="hint">${meta} — <b>avg uzun hold'da yükselir ama worst/tail patlar + bağımsız-N (bağN) çöker.</b> CAN describe / CANNOT bless. +900@48h = outage artifact (§166). FWD paper = ileriye doğru biriken kâğıt kol.</div>
     <div class="hcols">
       <div><h4>hour17</h4>${holdTable('hour17',H)}</div>
       <div><h4>echo (causal)</h4>${holdTable('echo_causal',H)}</div>
     </div>`;
}
function scoreCard(label,s,isCtrl,refute){
  const rate=(s&&s.rate!=null)?s.rate:null;
  const c=rate==null?'dim':((refute&&rate>=8)?'neg':(rate<=3?'pos':''));
  return `<div class="scard ${isCtrl?'ctrl':''}"><div class="k">${label}</div>
    <div class="r ${c}">${rate==null?'—':rate+'%'}</div>
    <div class="sub">${s?('tail '+s.tail+'/'+s.n):'n0'}</div></div>`;
}
function evCell(hh,c){
  if(!c) return `<div class="cell pend"><div class="hh">${hh}h</div>⏳</div>`;
  const k=c.tail?'tail':(c.net>0?'win':'lose');
  return `<div class="cell ${k}"><div class="hh">${hh}h</div>${sgn(c.net,0)}</div>`;
}
function feedPanel(d){
  const el=document.getElementById('feed'); const F=d.event_feed;
  if(!F||F.available===false){el.innerHTML=`<h2>Olay Akışı + Tail-Rate</h2><div class="hint">${(F&&F.note)||'forward ledger boş — olaylar ileriye doğru birikecek.'}</div>`;return;}
  const sb=F.scoreboard; const hs=F.horizons||[2,4,6,12,24,48];
  const score=`<div class="score">
    ${scoreCard('echo qualified',sb.echo_qual,false,false)}
    ${scoreCard('echo KONTROL',sb.echo_ctrl,true,false)}
    ${scoreCard('echo∩h≥17',sb.echo_hi,false,true)}
    ${scoreCard('hour17 qualified',sb.hour17_qual,false,false)}
    ${scoreCard('hour17 KONTROL',sb.hour17_ctrl,true,false)}</div>`;
  const rows=F.events.map(e=>{
    const chips=`${e.qualified_hour17?'<span class="chip c-q">H17</span>':'<span class="chip c-ctrl">h17·kontrol</span>'} `+
      `${e.qualified_echo?'<span class="chip c-q">ECHO</span>':'<span class="chip c-ctrl">echo·kontrol</span>'} `+
      `${e.echo_hi?'<span class="chip c-hi">∩h≥17</span>':''}`;
    const grid=`<div class="grid6">${hs.map(h=>evCell(h,e.res[String(h)])).join('')}</div>`;
    return `<div class="evrow ${e.any_tail?'tailrow':''}">
      <span class="dim" style="font-family:var(--mono);font-size:11px;min-width:94px">${(e.utc||'').slice(5,16).replace('T',' ')}</span>
      <span class="dim" style="font-family:var(--mono);font-size:10px">h${e.hour_utc??'?'}</span>
      ${chips}
      <span class="dim" style="font-family:var(--mono);font-size:10px">${e.open?('açık '+e.n_resolved+'/6'):'çözüldü'}</span>
      ${grid}</div>`;
  }).join('');
  el.innerHTML=`<h2>Olay Akışı + Tail-Rate <span class="badge b-paper">CANLI · her olay</span></h2>
     <div class="hint">Metrik = <b>tail-rate</b> (nostop net, §169) · primer ufuk echo=${sb.primary_h.echo}h hour17=${sb.primary_h.hour17}h. <b>qualified &lt; kontrol</b> olmalı; echo∩h≥17 base-rate ~${sb.base_rate_ref}%'de kuyruk gösterirse iddia ÖLÜR. Grid: ufuk başına nostop net · ⏳=bekliyor · kuyruk(&lt;−100)=kırmızı. Toplam ${F.n_events} olay.</div>
     ${score}${rows||'<div class="empty">henüz olay yok — ilk forward anchor bekleniyor</div>'}`;
}
async function tick(){
  try{
    const r=await fetch('/api/leads',{cache:'no-store'}); const d=await r.json();
    document.getElementById('ts').textContent=d.generated_utc.slice(0,19).replace('T',' ')+'Z';
    document.getElementById('mark').innerHTML=d.eth_mark?`ETH <b>${f(d.eth_mark.price,2)}</b>`:'ETH mark yok';
    document.getElementById('grid').innerHTML=leadCard(d.leads.echo,'echo')+leadCard(d.leads.hour17,'hour17');
    holdsPanel(d);
    feedPanel(d);
    const lv=d.live; const real=lv&&lv.available&&lv.has_real_position;
    const ban=document.getElementById('banner');
    if(real){ban.className='ban-real';ban.innerHTML='● GERÇEK POZİSYON AÇIK — live executor emir yerleştirmiş (gerçek para).';}
    else{ban.className='ban-paper';ban.innerHTML='◆ TÜM VERİ PAPER / SHADOW SİMÜLASYON — gerçek executor OFF, 0 gerçek pozisyon, 0 emir. Buradaki tüm pozisyon ve sonuçlar KÂĞIT (borsaya emir gitmiyor).';}
    document.getElementById('foot').innerHTML=
      `live executor: ${lv&&lv.available?('<span class="'+(real?'pos':'dim')+'">'+(lv.mode||'?')+(real?' · REAL POSITION':' · paper only')+'</span> · blocked_by='+(lv.new_entry_blocked_by||'—')):'state yok'} `+
      `&nbsp;·&nbsp; secondary/diagnostic surface — canonical :8770 yüzeyin YERİNE GEÇMEZ`;
  }catch(e){document.getElementById('ts').textContent='fetch hatası: '+e;}
}
tick(); setInterval(tick,7000);
</script></body></html>"""


# ── HTTP server (read-only) ───────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    server_version = "EclipseLeadsMonitor/1.0"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = render_html().encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
            self.wfile.write(body)
            return
        if path == "/api/leads":
            try:
                body = json.dumps(build_payload(), default=str).encode("utf-8")
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, body, "application/json")
                self.wfile.write(body)
                return
            self._send(200, body, "application/json")
            self.wfile.write(body)
            return
        self._send(404, b'{"error":"not found"}', "application/json")
        self.wfile.write(b'{"error":"not found"}')

    def do_HEAD(self) -> None:  # noqa: N802
        self._send(200, b"", "text/html; charset=utf-8")

    def _reject(self) -> None:
        body = b'{"error":"read-only surface: GET/HEAD only"}'
        self._send(405, body, "application/json")
        self.wfile.write(body)

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _reject  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
        return


def serve(host: str, port: int) -> int:
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"refusing non-loopback bind host={host!r}", file=sys.stderr)
        return 2
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"{datetime.now(timezone.utc).isoformat()} leads monitor (READ-ONLY) "
          f"http://{host}:{port}/  — echo + hour17")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        httpd.server_close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-lead read-only monitor dashboard.")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--serve-port", type=int, default=8771)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--once", action="store_true", help="print payload JSON and exit (self-test)")
    args = ap.parse_args()
    if args.once or not args.serve:
        print(json.dumps(build_payload(), indent=2, default=str))
        return 0
    return serve(args.host, args.serve_port)


if __name__ == "__main__":
    raise SystemExit(main())
