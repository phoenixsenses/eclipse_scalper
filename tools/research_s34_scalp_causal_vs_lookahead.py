"""research_s34_scalp_causal_vs_lookahead.py — 45m scalp + all horizons, CAUSAL vs LOOKAHEAD (READ-ONLY).

Operator (2026-07-20): "45 dakikalık scalp testi yaptır lookahead ve lookaheadsiz tüm timeframe'lerde."

Adds a 45-minute scalp horizon to the full hold set (45m/2h/4h/6h/12h/24h/48h) and, for the ECHO
signal, splits each into two arms:
  - CAUSAL     : cand_causal  = T0-only gates (regime, echo_30_90, not-bull, session, dow). No lookahead.
  - LOOKAHEAD  : cand_full    = cand_causal AND `not noisy`, where noisy inspects (T0+60s, T0+30m) —
                 i.e. it uses ≤30 min of FUTURE data (a T0-entry decision can't know it). This is the
                 frozen discovery gate whose hindsight tail-removal likely inflated echo (§165).

So the delta between the two arms at each horizon = the lookahead's contribution. NOTE: for holds
<=30m the not-noisy window even extends past the exit, so it is lookahead in the strongest sense.

hour17 has NO lookahead gate (all T0-knowable), so this causal/lookahead split is echo-only.
Reuses research_s34_echo_live_gauntlet verbatim so numbers reconcile. Read-only (mode=ro). Deterministic.
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
    build_events, load_vol_state, stats, no_overlap, gross,
    ETH_THRESH, FEE_BPS, LOOKBACK_MS,
)
from tools.research_s34_echo_causal_vs_lookahead import (  # noqa: E402
    cand_causal, cand_full, removed_by_noisy,
)
from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations, load_mark_index, reconstruct_anchors,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_SCALP_CAUSAL_VS_LOOKAHEAD.json"
OUT_MD = OUT_DIR / "S34_SCALP_CAUSAL_VS_LOOKAHEAD.md"

# horizon minutes → label (30m + 45m scalp first, then the full hold set)
HORIZONS_MIN = [30, 45, 120, 240, 360, 720, 1440, 2880]
LABEL = {30: "30m", 45: "45m", 120: "2h", 240: "4h", 360: "6h", 720: "12h", 1440: "24h", 2880: "48h"}


def _arm(events, gate, minutes: int, months: float) -> dict:
    key = "g_%d" % minutes
    vals = [ev[key] for ev in events if gate(ev) and ev.get(key) is not None]      # gross bps
    pairs = [(ev["ts"], ev[key]) for ev in events if gate(ev) and ev.get(key) is not None]
    s = stats(vals, key, months)                       # net = gross - FEE_BPS
    nov = no_overlap(pairs, hold_ms=minutes * 60_000)  # dedup at THIS horizon
    s["noov_n"] = len(nov)
    s["noov_sum"] = round(sum(v - FEE_BPS for v in nov), 1)
    s["noov_wr"] = round(100 * sum(1 for v in nov if (v - FEE_BPS) > 0) / len(nov), 1) if nov else None
    return s


def main() -> int:
    print("=== 45m scalp + all horizons: causal vs lookahead ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", now_ms - LOOKBACK_MS, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        span = sorted(int(a.anchor_ts_ms) for a in anchors)
        months = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0) if len(span) > 1 else 1.0
        marks_eth = load_mark_index(conn, "ETHUSDT")
        vol_rows = load_vol_state(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)

    print("  Computing scalp/hold grosses %s ..." % [LABEL[m] for m in HORIZONS_MIN])
    for ev in events:
        ts = ev["ts"]
        for m in HORIZONS_MIN:
            ev["g_%d" % m] = gross(marks_eth, ts, ts + m * 60_000)

    arms = {"CAUSAL": cand_causal, "LOOKAHEAD": cand_full, "REMOVED_by_noisy": removed_by_noisy}
    out = {
        "tool": "scalp_causal_vs_lookahead",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_anchors": len(anchors), "months": round(months, 2), "fee_bps": FEE_BPS,
        "horizons_min": HORIZONS_MIN,
        "frame": "45m scalp + full hold set. CAUSAL (no-lookahead) vs LOOKAHEAD (not-noisy, T+30m "
                 "future). Delta = lookahead contribution. echo-only (hour17 has no lookahead gate). "
                 "CAN kill, CANNOT bless — all gates in-sample; forward is the proof.",
        "arms": {name: {LABEL[m]: _arm(events, gate, m, months) for m in HORIZONS_MIN}
                 for name, gate in arms.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    def rowvals(s):
        if not s or s.get("n", 0) == 0:
            return ("0", "—", "—", "—", "—", "—", "—")
        return (str(s["n"]), "%.1f%%" % s["wr"], "%+.1f" % s["avg"], "%+.1f" % s.get("worst", 0),
                str(s.get("tail_n", 0)), str(s.get("mc_p")), str(s.get("noov_n", 0)))

    L = ["# S34 — 45m Scalp + All Horizons · CAUSAL vs LOOKAHEAD (echo)", "",
         "_%s · READ-ONLY · anchors=%d · %.2f mo · FEE=%dbps_" % (
             out["generated_utc"], len(anchors), months, int(FEE_BPS)), "",
         "> **LOOKAHEAD** arm keeps the frozen `not noisy` gate (inspects T0+60s→T0+30m = future); "
         "**CAUSAL** drops it. The gap = the lookahead's contribution (hindsight tail removal, §165). "
         "For holds ≤30m the not-noisy window even extends past exit. echo-only. CAN kill, CANNOT bless.",
         "", "Kolon: N, WR, avg(net), worst, tail(<-100), mc_p, noovN."]
    for name in ("CAUSAL", "LOOKAHEAD", "REMOVED_by_noisy"):
        L += ["", "## %s" % name, "",
              "| hold | N | WR | avg | worst | tail | mc_p | noovN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in HORIZONS_MIN:
            n, wr, avg, worst, tail, mc, noov = rowvals(out["arms"][name][LABEL[m]])
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (LABEL[m], n, wr, avg, worst, tail, mc, noov))
    L += ["", "## Read",
          "- 45m scalp: if CAUSAL avg≈0/neg or tail-heavy → the quick exit doesn't harvest the bounce "
          "(cascade rebound is slower than 45m) — scalp dead. If LOOKAHEAD >> CAUSAL at 45m, the pretty "
          "number is hindsight, not a scalp edge.",
          "- Compare LOOKAHEAD−CAUSAL per horizon: a large positive gap = lookahead doing the work "
          "(esp. tail/worst). Forward is the only proof; no tuning here.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    for name in ("CAUSAL", "LOOKAHEAD"):
        print("  %s:" % name)
        for m in HORIZONS_MIN:
            s = out["arms"][name][LABEL[m]]
            print("    %4s  N=%-3d WR=%-6s avg=%+7.1f worst=%+8.1f tail=%-2d mc=%s noovN=%d" % (
                LABEL[m], s.get("n", 0), str(s.get("wr")), s.get("avg", 0.0), s.get("worst", 0.0),
                s.get("tail_n", 0), str(s.get("mc_p")), s.get("noov_n", 0)))
    print("JSON:", OUT_JSON)
    print("MD:  ", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
