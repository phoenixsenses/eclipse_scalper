"""LIQUIDATION SOURCE-QUALITY COVERAGE CONTRACT RECONCILIATION -- READ-ONLY.

Reconstructs METHOD_A (original audit: 83/1/136) and METHOD_B (rehearsal:
125/1/94) exactly, produces the per-signal disagreement table, gathers
independent collector-health evidence per window, and evaluates a
semantics-based candidate contract (STANDARD-2). No canonical write, no
outcome read, microstructure.db mode=ro only.

Import-safe (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-CONSUMER-PREP-
SOURCE-QUALITY-MAIN-GUARD-V1): all DB connections, asserts, the heavy
reconciliation, and CSV/JSON output happen ONLY inside main() at run time,
never at import. This is a prep/hygiene change only -- no query,
computation, assert, or output semantics were altered, and no reader
migration was done: the window_health mark_prices/agg_trades range reads
remain direct SQL, deferred to a later range-read gate. The old hardcoded
session-scratchpad output path is replaced by a run-time --out-dir default.
"""
import argparse
import bisect
import csv
import datetime as dt
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, r"D:\eclipse_scalper")

from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import birth_truncated_geometry_rehearsal as rehearsal
from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors

CANON = r"D:\eclipse_scalper\data\ami\canonical.sqlite"
MICRO = r"D:\eclipse_scalper\data\microstructure.db"
OUT_SUBDIR = "s34_source_quality_reconciliation"


def iso(ms):
    if ms is None:
        return None
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- methods
def overlaps(ws, we, gs, ge):
    return ws <= ge and gs <= we


def method_a(ws, we, birth):
    """Original-audit hypothesis: resolved-gap overlap first; then everything
    at/after the FIRST open-ended (never-resolved) liquidations gap start is
    UNRESOLVED; else COMPLETE."""
    for gs, ge in resolved_gaps:
        if overlaps(ws, birth, gs, ge):
            return "SOURCE_GAPPED"
    if birth >= first_open_gap_start:
        return "SOURCE_COVERAGE_UNRESOLVED"
    return "SOURCE_COMPLETE"


def method_b(ws, we, birth):
    return rehearsal.classify_window_quality(ws, birth, all_gaps_b, cutoff_b)


# per-window collector-health evidence (STANDARD-2 inputs)
def window_health(ws, birth):
    rows = conn_m.execute(
        "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms",
        (ws - 15_000, birth + 15_000)).fetchall()
    ts = [r[0] for r in rows]
    inside = [t for t in ts if ws <= t <= birth]
    # max internal inter-arrival gap across the window, including edges
    seq = [ws] + inside + [birth]
    max_gap_ms = max(b - a for a, b in zip(seq, seq[1:])) if len(seq) > 1 else None
    lead_ok = any(ws - 15_000 <= t <= ws for t in ts)
    trail_ok = any(birth <= t <= birth + 15_000 for t in ts)
    at_n = conn_m.execute(
        "SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?",
        (ws, birth)).fetchone()[0]
    return {"mark_rows_n": len(inside), "mark_max_gap_ms": max_gap_ms,
            "mark_lead_ok": lead_ok, "mark_trail_ok": trail_ok, "agg_rows_n": at_n}


def standard2(ws, we, birth, health):
    """Candidate semantics-based contract: positive per-window process-health
    evidence + no liquidations-gap evidence anywhere near the window."""
    for gs, ge in resolved_gaps:
        if overlaps(ws, birth, gs, ge):
            return "SOURCE_GAPPED"
    for sh in open_gap_shadows:
        ge = sh["empirical_end"] if sh["empirical_end"] is not None else birth
        if overlaps(ws, birth, sh["start"], ge):
            return "SOURCE_COVERAGE_UNRESOLVED"
    for g in gap_rows:  # ANY liq-gap flag (resolved or open) starting near/inside window
        if ws - 600_000 <= g[1] <= birth:
            return "SOURCE_COVERAGE_UNRESOLVED"
    ok = (health["mark_max_gap_ms"] is not None and health["mark_max_gap_ms"] <= 10_000
          and health["mark_lead_ok"] and health["mark_trail_ok"] and health["agg_rows_n"] > 0)
    return "SOURCE_COMPLETE" if ok else "SOURCE_COVERAGE_UNRESOLVED"


def counts(key):
    c = {}
    for r in table:
        c[r[key]] = c.get(r[key], 0) + 1
    return dict(sorted(c.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Liquidation source-quality coverage contract reconciliation (read-only).")
    parser.add_argument(
        "--out-dir", default=os.path.join(tempfile.gettempdir(), OUT_SUBDIR),
        help="Directory for the per-signal CSV/JSON dump, created at run time. Defaults to a "
             "deterministic OS-temp subdirectory (the old hardcoded, foreign session-scratchpad "
             "path is gone).")
    args = parser.parse_args(argv)

    # These are populated here (run time) and read by the module-level
    # helper functions above; keeping them global preserves the original
    # top-level-execution semantics exactly, just deferred out of import.
    global conn_m, gap_rows, resolved_gaps, open_gap_shadows, all_gaps_b, cutoff_b, first_open_gap_start, table

    # ---------------------------------------------------------------- load inputs
    conn_c = sqlite3.connect(f"file:{CANON}?mode=ro", uri=True)
    # direct read-only query (feature_gateway.fetch_lifecycle_signals writes an
    # exposure-ledger row, which a mode=ro connection correctly refuses -- this
    # reconciliation must not write ANYTHING to the canonical DB)
    _cols = ["signal_id", "setup_id", "source_event_id", "independent_cycle_id",
             "symbol", "direction", "signal_birth_ts"]
    signals = [dict(zip(_cols, r)) for r in conn_c.execute(
        f"SELECT {', '.join(_cols)} FROM ami_signal_lifecycle WHERE symbol='ETHUSDT' AND direction='LONG'"
    ).fetchall()]
    event_ids = {s["source_event_id"] for s in signals}
    events_by_id = rehearsal.fetch_events_by_id(conn_c, event_ids)
    conn_c.close()
    assert len(signals) == 220, len(signals)

    conn_m = sqlite3.connect(f"file:{MICRO}?mode=ro", uri=True)
    all_sell_liqs = rehearsal.fetch_all_sell_liqs(conn_m)
    gap_rows = conn_m.execute(
        "SELECT id, start_ts_ms, end_ts_ms, duration_sec, resolved_bool FROM gaps "
        "WHERE stream='liquidations' ORDER BY start_ts_ms"
    ).fetchall()

    resolved_gaps = [(g[1], g[2]) for g in gap_rows if g[4] == 1 and g[2] is not None]
    open_gaps = [(g[0], g[1], g[3]) for g in gap_rows if g[4] == 0]

    # empirical bound for each open-ended gap: first liquidation row (ANY symbol,
    # ANY side -- !forceOrder@arr is all-market) strictly after the gap start.
    open_gap_shadows = []
    for gid, gs, dur in open_gaps:
        nxt = conn_m.execute(
            "SELECT MIN(ts_ms) FROM liquidations WHERE ts_ms > ?", (gs,)
        ).fetchone()[0]
        open_gap_shadows.append({"gap_id": gid, "start": gs, "empirical_end": nxt,
                                 "duration_at_detect_sec": dur,
                                 "empirical_silence_sec": (nxt - gs) / 1000.0 if nxt else None})

    all_gaps_b = [(g[1], g[2]) for g in gap_rows]  # METHOD_B input shape (start, end-or-None)
    cutoff_b = rehearsal.gap_registry_cutoff_ts_ms(all_gaps_b)
    first_open_gap_start = min(gs for _gid, gs, _d in open_gaps)

    # ---------------------------------------------------------------- windows
    windows = {}
    for s in signals:
        ev = events_by_id[s["source_event_id"]]
        g = geo.reconstruct_signal_geometry(
            all_sell_liqs, int(ev["anchor_ts_ms"]), int(s["signal_birth_ts"]),
            reconstruct_anchors_fn=reconstruct_anchors)
        assert g is not None, s["signal_id"]
        windows[s["signal_id"]] = (g["source_window_start_ts_ms"], g["source_window_end_ts_ms"])

    # ---------------------------------------------------------------- classify all
    table = []
    for s in signals:
        ws, we = windows[s["signal_id"]]
        birth = int(s["signal_birth_ts"])
        h = window_health(ws, birth)
        a = method_a(ws, we, birth)
        b = method_b(ws, we, birth)
        s2 = standard2(ws, we, birth, h)
        gap_evidence = [g for g in gap_rows if overlaps(ws, birth, g[1], g[2] if g[2] else birth)]
        table.append({
            "signal_id": s["signal_id"], "signal_birth_ts": birth, "birth_utc": iso(birth),
            "window_start_ts": ws, "window_end_ts": we,
            "month": iso(birth)[:7], "independent_cycle_id": s["independent_cycle_id"],
            "setup_id": s["setup_id"],
            "method_a": a, "method_b": b, "standard2": s2,
            "disagree_ab": a != b,
            "gap_rows_touching": [{"id": g[0], "start": iso(g[1]), "end": iso(g[2]),
                                   "resolved": g[4]} for g in gap_evidence],
            **h,
        })

    conn_m.close()

    print("first_open_gap_start (METHOD_A cutoff) =", iso(first_open_gap_start))
    print("registry last liq-gap start (METHOD_B cutoff) =", iso(cutoff_b))
    print()
    print("METHOD_A counts   =", counts("method_a"))
    print("METHOD_B counts   =", counts("method_b"))
    print("STANDARD2 counts  =", counts("standard2"))
    print()

    dis = [r for r in table if r["disagree_ab"]]
    print(f"A-vs-B disagreement set: {len(dis)} signals")
    months = {}
    for r in dis:
        months[r["month"]] = months.get(r["month"], 0) + 1
    print("disagreement by month:", dict(sorted(months.items())))
    print("disagreement A-status:", sorted({r['method_a'] for r in dis}))
    print("disagreement B-status:", sorted({r['method_b'] for r in dis}))
    print("disagreement distinct cycles:", len({r['independent_cycle_id'] for r in dis}))
    print()

    gapped = [r for r in table if r["method_a"] == "SOURCE_GAPPED" or r["method_b"] == "SOURCE_GAPPED"]
    print("GAPPED signal(s):")
    for r in gapped:
        print(" ", r["signal_id"], r["birth_utc"], "A=", r["method_a"], "B=", r["method_b"],
              "gap rows:", r["gap_rows_touching"])
    print()

    # open-gap shadow summary
    print("open-ended gap empirical shadows (worst 6 by silence):")
    for sh in sorted(open_gap_shadows, key=lambda x: -(x["empirical_silence_sec"] or 0))[:6]:
        print(f"  gap id={sh['gap_id']} start={iso(sh['start'])} empirical_end={iso(sh['empirical_end'])} "
              f"silence={sh['empirical_silence_sec']:.0f}s (duration_at_detect={sh['duration_at_detect_sec']:.0f}s)")
    print()

    # health summary for the disputed set
    bad_health = [r for r in dis if r["mark_max_gap_ms"] is None or r["mark_max_gap_ms"] > 10_000]
    print(f"disputed-42 with mark_prices hole >10s inside window: {len(bad_health)}")
    s2_dis = {}
    for r in dis:
        s2_dis[r["standard2"]] = s2_dis.get(r["standard2"], 0) + 1
    print("disputed-42 under STANDARD2:", s2_dis)
    print()

    # research-readiness populations
    sig_by_id = {s["signal_id"]: s for s in signals}
    for name, key in (("METHOD_A", "method_a"), ("METHOD_B", "method_b"), ("STANDARD2", "standard2")):
        comp = [sig_by_id[r["signal_id"]] for r in table if r[key] == "SOURCE_COMPLETE"]
        if comp:
            rep = rehearsal.compute_population_report(comp)
            print(f"{name} SOURCE_COMPLETE_ONLY: signals={rep['signal_n']} events={rep['source_event_n']} "
                  f"cycles={rep['independent_cycle_n']} train={rep['train_cycle_n']} test={rep['test_cycle_n']} "
                  f"min_bucket_n={rep['min_bucket_n_verdict']} monthly={rep['monthly_distribution']}")
        else:
            print(f"{name} SOURCE_COMPLETE_ONLY: EMPTY")

    # full per-signal dump
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "source_quality_per_signal.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in table[0] if k != "gap_rows_touching"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    out_json = os.path.join(out_dir, "source_quality_per_signal.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"table": table, "open_gap_shadows": open_gap_shadows,
                   "method_a_counts": counts("method_a"), "method_b_counts": counts("method_b"),
                   "standard2_counts": counts("standard2")}, f, indent=1, default=str)
    print()
    print("WROTE", out_csv)
    print("WROTE", out_json)


if __name__ == "__main__":
    main()
