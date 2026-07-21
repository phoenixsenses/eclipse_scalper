"""
research_s34_echo_causal_vs_lookahead.py — CONTAMINATION DIAGNOSTIC (read-only).

Question (user, 2026-07-20): does echo_30_90+regime survive WITHOUT the lookahead?

The frozen candidate cand_9090 bakes in `not noisy`, where
    noisy = a >=50K ETH SELL liq exists in (T0+60s, T0+30m)
i.e. it inspects 30 minutes of FUTURE data and, with hindsight, drops exactly the
events whose selling continues (the tails). This diagnostic re-runs the SAME frozen
population and gross returns, toggling ONLY that one gate, so the delta is a clean
measure of the lookahead's contribution (all other gates held fixed).

WHAT THIS CAN DO: KILL. If the causal arm (no-noisy) is negative / tail-heavy even
in-sample, the headline "+92.5bps/tail0" was the lookahead. Stop.
WHAT THIS CANNOT DO: BLESS. The other gates (echo window, regime, session, dow) were
ALSO selected on this burned sample, so a positive causal arm is necessary-not-sufficient;
real proof is FORWARD only (echo_forward_ledger, post-2026-07-20). No threshold is tuned here.

Reuses tools/research_s34_echo_live_gauntlet.py verbatim (build_events, stats, no_overlap)
so numbers reconcile exactly with the gauntlet. Read-only (mode=ro). Deterministic (seed 42
inherited from the gauntlet module).
"""
from __future__ import annotations
import json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations, load_mark_index, reconstruct_anchors,
)
from tools.research_s34_echo_live_gauntlet import (  # noqa: E402
    build_events, load_vol_state, stats, no_overlap, regime,
    ETH_THRESH, FEE_BPS, LOOKBACK_MS,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ECHO_CAUSAL_VS_LOOKAHEAD.json"
OUT_MD = OUT_DIR / "S34_ECHO_CAUSAL_VS_LOOKAHEAD.md"


# --- the frozen echo_30_90 gate, split into causal core vs the lookahead add-on ---
def cand_causal(ev):   # echo_30_90 + regime, T0-KNOWABLE ONLY (drops `not noisy`)
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] not in {0, 2} and ev["echo_30_90"] and regime(ev))

def cand_full(ev):     # == gauntlet cand_9090 (causal core AND not-noisy lookahead)
    return cand_causal(ev) and (not ev["noisy"])

def removed_by_noisy(ev):  # events the lookahead gate DROPS from the causal set
    return cand_causal(ev) and ev["noisy"]


def arm(events, gate, key, months):
    vals = [ev[key] for ev in events if gate(ev) and ev.get(key) is not None]
    pairs = [(ev["ts"], ev[key]) for ev in events if gate(ev) and ev.get(key) is not None]
    s = stats(vals, key, months)
    nov = no_overlap(pairs)
    s["noov_n"] = len(nov)
    s["noov_per_month"] = round(len(nov) / months, 1) if months else None
    s["noov_sum"] = round(sum(v - FEE_BPS for v in nov), 1)
    s["noov_wr"] = round(100 * sum(1 for v in nov if (v - FEE_BPS) > 0) / len(nov), 1) if nov else None
    return s


def main():
    print("=== Echo causal vs lookahead diagnostic ===")
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
        events = build_events(conn, anchors, marks_eth, vol_rows)

    out = {"tool": "echo_causal_vs_lookahead", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_anchors": len(anchors), "months": round(months, 2), "fee_bps": FEE_BPS,
           "frame": "Contamination diagnostic. CAN kill (negative causal arm), CANNOT bless "
                    "(other gates also in-sample). Proof is FORWARD only. No tuning here.",
           "arms": {}}
    for key, hl in [("g_t0_4h", "T0 hold 4h"), ("g_t0_6h", "T0 hold 6h")]:
        out["arms"][hl] = {
            "full_with_lookahead": arm(events, cand_full, key, months),
            "causal_no_lookahead": arm(events, cand_causal, key, months),
            "removed_by_noisy_gate": arm(events, removed_by_noisy, key, months),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    def row(tag, s):
        if not s or s.get("n", 0) == 0:
            return "| %s | 0 | — | — | — | — | — | — | — |" % tag
        return "| %s | %d | %.1f | %.1f%% | %+.1f | %+.1f | %d | %s | %s |" % (
            tag, s["n"], s.get("per_month", 0), s["wr"], s["avg"], s.get("worst", 0),
            s.get("tail_n", 0), str(s.get("mc_p")), str(s.get("wf")))

    L = ["# Echo — Causal (no-lookahead) vs Frozen (with-lookahead) Diagnostic", "",
         "_%s · READ-ONLY · anchors=%d · %.2f mo · FEE=%dbps_" % (
             out["generated_utc"], len(anchors), months, int(FEE_BPS)), "",
         "> **CAN kill, CANNOT bless.** Only the `not noisy` (T+30m lookahead) gate is toggled; "
         "all other gates (echo_30_90, regime, session, dow, bull) held fixed. A positive causal "
         "arm is necessary-not-sufficient — the other gates are also in-sample. Proof is FORWARD only.",
         "", "Kolon: N, /ay, WR, Avg(net bps), Worst, Tail(<-100), mc_p, WF."]
    for hl in ("T0 hold 4h", "T0 hold 6h"):
        a = out["arms"][hl]
        L += ["", "## %s" % hl, "",
              "| arm | N | /ay | WR | Avg | Worst | Tail | mc_p | WF |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
              row("FULL (with lookahead)", a["full_with_lookahead"]),
              row("CAUSAL (no lookahead)", a["causal_no_lookahead"]),
              row("REMOVED by noisy gate", a["removed_by_noisy_gate"]), "",
              "No-overlap (realistic, hold=4h dedup):", "",
              "| arm | noov N | noov /ay | noov WR | noov sum |", "|---|---:|---:|---:|---:|"]
        for tag, kk in [("FULL", "full_with_lookahead"), ("CAUSAL", "causal_no_lookahead"),
                        ("REMOVED", "removed_by_noisy_gate")]:
            s = a[kk]
            L.append("| %s | %s | %s | %s | %s |" % (
                tag, s.get("noov_n"), s.get("noov_per_month"), str(s.get("noov_wr")) + "%",
                s.get("noov_sum")))
    L += ["", "## Read", "- If CAUSAL avg/sum collapses vs FULL **and** REMOVED-by-noisy is disproportionately",
          "  negative/tail-heavy → the lookahead was doing the work (hindsight tail removal). KILL.",
          "- If CAUSAL holds up → necessary-not-sufficient; proceed to FORWARD accumulation + snapshot",
          "  enrichment (dev-list #15/#19). Still no deploy, no tuning on this burned sample.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({k: {kk: {"n": vv.get("n"), "avg": vv.get("avg"), "wr": vv.get("wr"),
                               "tail": vv.get("tail_n"), "sum": vv.get("sum"),
                               "noov_sum": vv.get("noov_sum")}
                          for kk, vv in v.items()} for k, v in out["arms"].items()}, indent=2))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
