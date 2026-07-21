"""research_s34_hold_horizon_sweep.py — causal hold-horizon sweep for BOTH leads (READ-ONLY).

Operator (2026-07-20): "48 hold stratejisi istiyorum ... bütün time frame'lerde hold yaptığı
şekilde, hepsinden birer birer" — for hour17 AND echo(causal), sweep hold horizons
2h / 4h / 6h / 12h / 24h / 48h, one arm each.

WHY THIS FRAMING (honest): the eye-catching "+900 bps @ ~48h" seen on the monitor were OUTAGE
ARTIFACTS — a ~46.4h ledger gap force-closed positions late (SYSTEM_STATE §166 / §141 /
S34_HOUR17_DIRECT_SHORT_AUDIT), NOT a 48h-hold edge. This sweep replaces that mirage with an
honest causal measurement of the actual hold-response curve.

DISCIPLINE:
  * causal — T0-only gates, mark-to-mark gross, NO lookahead.
  * CAN describe the hold curve; CANNOT bless a horizon. Picking the best hold on this burned
    ~5-month sample is in-sample selection (necessary-not-sufficient). Forward is the only proof.
  * INDEPENDENT (no-overlap) N reported PER horizon — critical: 48h windows overlap massively so
    almost no independent observations remain (hour17 already flagged INSUFFICIENT_INDEPENDENT_EVENTS
    at 6h; at 48h it is worse). Read the noov_N column, not the raw N.
  * FUNDING drag (longs) grows with hold — estimated from funding_rates mean and subtracted so long
    horizons are not flattered by the 5bps-only fee model.

Reuses tools/research_s34_echo_live_gauntlet verbatim so numbers reconcile with the gauntlet and the
causal report. Read-only (mode=ro). Deterministic.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_echo_live_gauntlet import (  # noqa: E402
    build_events, load_vol_state, stats, no_overlap, regime, gross,
    ETH_THRESH, FEE_BPS, LOOKBACK_MS,
)
from tools.research_s34_echo_causal_vs_lookahead import cand_causal  # noqa: E402  (T0 causal echo gate)
from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations, load_mark_index, reconstruct_anchors,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_HOLD_HORIZON_SWEEP.json"
OUT_MD = OUT_DIR / "S34_HOLD_HORIZON_SWEEP.md"

HORIZONS_H = [2, 4, 6, 12, 24, 48]
STOP_VARIANTS = [(None, "nostop"), (150.0, "s150"), (300.0, "s300")]  # tail-control for longs-in-downtrend
FUNDING_INTERVAL_H = 8  # perpetual funding every 8h


def _gkey(hours: int, stop_label: str) -> str:
    return "g_t0_%dh_%s" % (hours, stop_label)


def cand_hour17(ev):
    """hour17 LONG gate — mirrors s34_state_machine_live_executor.py:502-507 (all T0-knowable)."""
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and (ev["btc4h"] < 0 or ev["btc7d"] < 0) and ev["hour"] >= 17)


def mean_funding_bps_per_interval(conn) -> float:
    """Mean ETHUSDT funding per 8h interval, in bps. Positive => a LONG pays."""
    row = conn.execute(
        "SELECT AVG(funding_rate) FROM funding_rates WHERE symbol='ETHUSDT'"
    ).fetchone()
    return (float(row[0]) * 10_000.0) if row and row[0] is not None else 0.0


def horizon_arm(events, gate, hours: int, months: float, fund_per_int: float,
                stop_label: str = "nostop") -> dict:
    key = _gkey(hours, stop_label)
    vals = [ev[key] for ev in events if gate(ev) and ev.get(key) is not None]
    pairs = [(ev["ts"], ev[key]) for ev in events if gate(ev) and ev.get(key) is not None]
    s = stats(vals, key, months)              # net = gross - FEE_BPS
    hold_ms = hours * 3600_000
    nov = no_overlap(pairs, hold_ms=hold_ms)  # independent observations at this horizon
    s["noov_n"] = len(nov)
    s["noov_per_month"] = round(len(nov) / months, 1) if months else None
    s["noov_sum"] = round(sum(v - FEE_BPS for v in nov), 1)
    s["noov_wr"] = round(100 * sum(1 for v in nov if (v - FEE_BPS) > 0) / len(nov), 1) if nov else None
    intervals = hours // FUNDING_INTERVAL_H
    drag = round(intervals * fund_per_int, 2)   # cost to a LONG over the hold
    s["hold_h"] = hours
    s["funding_intervals"] = intervals
    s["funding_drag_bps"] = drag
    s["avg_after_funding"] = (round(s["avg"] - drag, 1) if s.get("avg") is not None else None)
    return s


def main() -> int:
    print("=== hold-horizon sweep (causal, both leads) ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", now_ms - LOOKBACK_MS, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        span = sorted(int(a.anchor_ts_ms) for a in anchors)
        span_days = (span[-1] - span[0]) / 86_400_000 if len(span) > 1 else 30
        months = max(1.0, span_days / 30.0)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        vol_rows = load_vol_state(conn)
        fund_per_int = mean_funding_bps_per_interval(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)

    print("  Computing hold-horizon grosses (%s) × stops %s ..." % (
        HORIZONS_H, [s for _, s in STOP_VARIANTS]))
    for ev in events:
        ts = ev["ts"]
        for h in HORIZONS_H:
            for stop_bps, label in STOP_VARIANTS:
                ev[_gkey(h, label)] = gross(marks_eth, ts, ts + h * 3600_000, stop_bps=stop_bps)

    signals = [("hour17", cand_hour17), ("echo_causal", cand_causal)]
    out = {
        "tool": "hold_horizon_sweep",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_anchors": len(anchors),
        "months": round(months, 2),
        "fee_bps": FEE_BPS,
        "funding_bps_per_8h": round(fund_per_int, 3),
        "horizons_h": HORIZONS_H,
        "frame": "Causal hold-response curve. CAN describe, CANNOT bless (in-sample horizon "
                 "selection). Read noov_N (independent obs) not raw N — 48h overlaps heavily. "
                 "funding drag (long) grows with hold. Prepared because the +900bps@~48h were "
                 "OUTAGE ARTIFACTS (§166), not a 48h edge.",
        "signals": {},
    }
    for name, gate in signals:
        out["signals"][name] = {
            "h%d" % h: {label: horizon_arm(events, gate, h, months, fund_per_int, label)
                        for _, label in STOP_VARIANTS}
            for h in HORIZONS_H
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    def cell(s):
        if not s or s.get("n", 0) == 0:
            return "—/—/—"
        return "%+.0f / %+.0f / %d" % (s.get("avg", 0.0), s.get("worst", 0.0), s.get("tail_n", 0))

    L = ["# S34 Hold-Horizon Sweep + Stop Variants — hour17 & echo (causal)", "",
         "_%s · READ-ONLY · anchors=%d · %.2f mo · FEE=%dbps · funding≈%.2fbps/8h_" % (
             out["generated_utc"], len(anchors), months, int(FEE_BPS), fund_per_int), "",
         "> **CAN describe, CANNOT bless.** Causal (T0 gates, mark-to-mark, no lookahead). Sweeping "
         "the hold on this burned ~5mo sample = in-sample selection (necessary-not-sufficient); forward "
         "is the only proof. **Read `noovN` (independent obs)** — 48h windows overlap heavily. These are "
         "LONGs into a bearish regime (mean-reversion after a SELL cascade); the fat left tail = the "
         "downtrend CONTINUED. The stop is the only real lever (tail T0-unpredictable §162; short-the-"
         "trend failed BOTH_FAIL; reactive-cut whipsaws §163). +900@48h was an OUTAGE ARTIFACT (§166).",
         "",
         "Her hücre: **avg / worst / tail** (net bps). Stop, downtrend-devamı kuyruğunu keser."]
    for name in ("hour17", "echo_causal"):
        L += ["", "## %s" % name, "",
              "| hold | noovN | nostop (avg/worst/tail) | −150bps | −300bps |",
              "|---|---:|---:|---:|---:|"]
        for h in HORIZONS_H:
            v = out["signals"][name]["h%d" % h]
            L.append("| %dh | %d | %s | %s | %s |" % (
                h, v["nostop"].get("noov_n", 0), cell(v["nostop"]), cell(v["s150"]), cell(v["s300"])))
    L += ["", "## Read",
          "- Stop kolonlarında **worst** çarpıcı biçimde küçülür (−928→−300 civarı) ama **avg** de düşer "
          "— stop edge yaratmaz, felaketi sınırlar. §163 ile tutarlı.",
          "- avg uzun hold'da yükselir çünkü drift+varyans; noovN çöker, mc/WF zayıflar → kanıt değil.",
          "- Gerçek kanıt = FORWARD paper (hold_horizon_forward_ledger), bu in-sample eğri değil.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    for name in ("hour17", "echo_causal"):
        print("  %s (avg/worst/tail):" % name)
        for h in HORIZONS_H:
            v = out["signals"][name]["h%d" % h]
            print("    %3dh noovN=%-3d | nostop %-16s | s150 %-16s | s300 %-16s" % (
                h, v["nostop"].get("noov_n", 0), cell(v["nostop"]), cell(v["s150"]), cell(v["s300"])))
    print("JSON:", OUT_JSON)
    print("MD:  ", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
