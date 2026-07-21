"""research_s34_echo_forward_evaluator.py — Part 2 of the echo forward-evaluator (READ-ONLY).

Reads the measured-cost forward ledger (Part 1, SYSTEM_STATE §175/§177) + the frozen burned baseline,
and scores echo_causal across every (signal × horizon × stop) cell + the §169 hour17 tail-gate against
PRE-REGISTERED FROZEN decision rules (design reports/research/s34/S34_ECHO_FORWARD_EVALUATOR_DESIGN_V1).

Produces EVIDENCE, never blesses/deploys. No DB, no orders, no network — reads two files, writes two.
The rules below were frozen BEFORE any forward data existed (N=0): applying them now or later is a clean
out-of-sample test, NOT lookahead. The one discipline: the verdict is whatever the frozen rule says.

Inputs : reports/shadow/hold_horizon_forward_ledger.jsonl   (forward = the only evidence)
         reports/research/s34/S34_HOLD_HORIZON_SWEEP.json    (burned baseline = comparison bar only)
Outputs: reports/research/s34/ECHO_FORWARD_SCORECARD.{json,md}
Run    : python -m tools.research_s34_echo_forward_evaluator [--once]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "hold_horizon_forward_ledger.jsonl"
BASELINE = ROOT / "reports" / "research" / "s34" / "S34_HOLD_HORIZON_SWEEP.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "ECHO_FORWARD_SCORECARD.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "ECHO_FORWARD_SCORECARD.md"

# ── FROZEN decision rules (design §2/§3; ratified 2026-07-20 while forward N=0) ────────────────────
HORIZONS = [2, 4, 6, 12, 24, 48]
STOP_FIELD = {"nostop": "net_bps", "s150": "net_bps_s150", "s300": "net_bps_s300"}
N_MIN = 20                 # min no-overlap obs before any confirm/refute verdict (power gate)
CONFIRM_FRACTION = 0.40    # CONFIRM needs point-avg >= 0.40 x in-sample noov-avg
TAIL_BPS = -100.0          # a "tail" = net < -100 bps
TAIL_CONFIRM = 0.05        # gate CONFIRM: hour>=17 forward tail-rate <= 5%
TAIL_REFUTE = 0.10         # gate REFUTE : hour>=17 forward tail-rate >= 10%
BOOTSTRAP_N = 10_000
SEED = 1234
PRIMARY = {("echo", 6, "nostop")}        # P1; P2 = the hour17 tail-gate overlay at h6 (handled separately)

# signal groups over RESOLVE rows (measured_v2 only)
SIGNALS = {
    "echo":        lambda r: bool(r.get("qualified_echo")),
    "echo_hi":     lambda r: bool(r.get("qualified_echo")) and (r.get("hour_utc") is not None and r["hour_utc"] >= 17),
    "echo_lo":     lambda r: bool(r.get("qualified_echo")) and (r.get("hour_utc") is not None and r["hour_utc"] < 17),
    "echo_ctrl":   lambda r: not r.get("qualified_echo"),
    "hour17":      lambda r: bool(r.get("qualified_hour17")),
    "hour17_ctrl": lambda r: not r.get("qualified_hour17"),
}


def _load_ledger():
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # forward evidence = measured_v2 RESOLVE rows only; a clean (countable) obs has net_bps != None
        if r.get("event") == "RESOLVE" and r.get("cost_model") == "measured_v2":
            rows.append(r)
    return rows


def _baseline_noov_avg(sweep, horizon, stop):
    """In-sample no-overlap avg for echo_causal cell, from the frozen sweep. None if unavailable."""
    try:
        cell = sweep["signals"]["echo_causal"][f"h{horizon}"][stop]
        n = cell.get("noov_n") or 0
        return (cell["noov_sum"] / n) if n else None
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def _no_overlap(rows, horizon):
    """Greedy no-overlap: keep clean rows for this horizon >= horizon apart (anchor spacing)."""
    win = horizon * 3600_000
    rs = sorted((r for r in rows if r.get("hold_h") == horizon and r.get("net_bps") is not None),
                key=lambda r: r["anchor_ts_ms"])
    kept, last = [], None
    for r in rs:
        if last is None or r["anchor_ts_ms"] - last >= win:
            kept.append(r)
            last = r["anchor_ts_ms"]
    return kept


def _boot_ci(vals):
    """Seeded bootstrap 95% CI of the mean. (None,None) if < 2 obs."""
    k = len(vals)
    if k < 2:
        return (None, None)
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP_N):
        s = 0.0
        for _ in range(k):
            s += vals[rng.randrange(k)]
        means.append(s / k)
    means.sort()
    return (means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N)])


def _tail_rate(vals):
    return (sum(1 for v in vals if v < TAIL_BPS) / len(vals)) if vals else None


def _cell_verdict(noov_n, avg, ci_lo, ci_hi, insample_avg):
    if noov_n < N_MIN:
        return "ACCUMULATING"
    if ci_hi is not None and ci_hi <= 0:
        return "REFUTED"
    bar = CONFIRM_FRACTION * insample_avg if (insample_avg is not None and insample_avg > 0) else 0.0
    if ci_lo is not None and ci_lo > 0 and avg is not None and avg >= bar:
        return "CONFIRMED"
    return "DEGRADED"


def _gate_verdict(hi_n, hi_tail, lo_tail):
    if hi_n < N_MIN or hi_tail is None:
        return "GATE_ACCUMULATING"
    if hi_tail >= TAIL_REFUTE:
        return "GATE_REFUTED"
    if hi_tail <= TAIL_CONFIRM and lo_tail is not None and lo_tail >= TAIL_REFUTE:
        return "GATE_CONFIRMED"
    return "GATE_ACCUMULATING"


def _stats(kept, stop, insample_avg):
    field = STOP_FIELD[stop]
    vals = [r[field] for r in kept if r.get(field) is not None]
    n = len(vals)
    if n == 0:
        return {"noov_n": 0, "wr": None, "avg": None, "worst": None, "tail_rate": None,
                "ci_lo": None, "ci_hi": None,
                "insample_noov_avg": (round(insample_avg, 2) if insample_avg is not None else None),
                "verdict": "ACCUMULATING"}
    avg = sum(vals) / n
    ci_lo, ci_hi = _boot_ci(vals)
    return {"noov_n": n, "wr": round(100 * sum(1 for v in vals if v > 0) / n, 1),
            "avg": round(avg, 2), "worst": round(min(vals), 2), "tail_rate": round(_tail_rate(vals), 3),
            "ci_lo": (round(ci_lo, 2) if ci_lo is not None else None),
            "ci_hi": (round(ci_hi, 2) if ci_hi is not None else None),
            "insample_noov_avg": (round(insample_avg, 2) if insample_avg is not None else None),
            "verdict": _cell_verdict(n, avg, ci_lo, ci_hi, insample_avg)}


def evaluate():
    rows = _load_ledger()
    sweep = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {"signals": {}}

    resolves = len(rows)
    quarantined = sum(1 for r in rows if r.get("quarantined"))
    quarantine_rate = round(quarantined / resolves, 3) if resolves else None

    cells = {}
    for sig_name, pred in SIGNALS.items():
        sig_rows = [r for r in rows if pred(r)]
        for h in HORIZONS:
            kept = _no_overlap(sig_rows, h)
            for stop in STOP_FIELD:
                ins = _baseline_noov_avg(sweep, h, stop) if sig_name in ("echo", "echo_hi", "echo_lo") else None
                st = _stats(kept, stop, ins)
                key = f"{sig_name}|h{h}|{stop}"
                st["primary"] = (sig_name, h, stop) in PRIMARY
                st["label"] = "PRIMARY" if st["primary"] else "SECONDARY_DESCRIPTIVE"
                cells[key] = st

    # P2 — hour17 tail-gate overlay (§169), per horizon; primary at h6
    gate = {}
    for h in HORIZONS:
        hi = _no_overlap([r for r in rows if SIGNALS["echo_hi"](r)], h)
        lo = _no_overlap([r for r in rows if SIGNALS["echo_lo"](r)], h)
        hi_vals = [r["net_bps"] for r in hi if r.get("net_bps") is not None]
        lo_vals = [r["net_bps"] for r in lo if r.get("net_bps") is not None]
        hi_tail, lo_tail = _tail_rate(hi_vals), _tail_rate(lo_vals)
        gate[f"h{h}"] = {
            "hi_noov_n": len(hi_vals), "hi_tail_rate": (round(hi_tail, 3) if hi_tail is not None else None),
            "lo_noov_n": len(lo_vals), "lo_tail_rate": (round(lo_tail, 3) if lo_tail is not None else None),
            "primary": h == 6, "verdict": _gate_verdict(len(hi_vals), hi_tail, lo_tail)}

    status = "NO_DATA" if resolves == 0 else "SCORING"
    return {
        "tool": "echo_forward_evaluator", "status": status,
        "frozen_rules": {"N_MIN": N_MIN, "CONFIRM_FRACTION": CONFIRM_FRACTION,
                         "TAIL_BPS": TAIL_BPS, "TAIL_CONFIRM": TAIL_CONFIRM, "TAIL_REFUTE": TAIL_REFUTE,
                         "bootstrap_n": BOOTSTRAP_N, "seed": SEED},
        "forward": {"resolve_rows": resolves, "quarantined": quarantined,
                    "quarantine_rate": quarantine_rate},
        "primary_hypotheses": {
            "P1_echo_6h_nostop": cells.get("echo|h6|nostop", {}).get("verdict"),
            "P2_hour17_tailgate_6h": gate.get("h6", {}).get("verdict")},
        "cells": cells, "gate": gate,
        "note": ("Rules frozen while forward N=0 (design V1). Verdict = frozen rule output; wanting "
                 "different rules => NEW prereg on FRESH forward data, never re-judge the same data. "
                 "Quarantined rows excluded from all stats (net_bps=None); tail-rate is the §169 primary "
                 "metric. Read noov_n, not raw. Produces evidence only — no bless/deploy."),
    }


def _render_md(sc):
    L = ["# ECHO FORWARD SCORECARD", "",
         f"**Status:** `{sc['status']}` · resolve rows: {sc['forward']['resolve_rows']} · "
         f"quarantined: {sc['forward']['quarantined']} "
         f"(rate {sc['forward']['quarantine_rate']})", "",
         "**Frozen rules:** N_MIN=20 · CONFIRM avg>=0.40x in-sample & CI_LB>0 · "
         "tail-gate CONFIRM<=5% / REFUTE>=10% · seed 1234 (pre-registered while forward N=0 → not lookahead).",
         "",
         "## Primary hypotheses",
         f"- **P1 echo 6h nostop:** `{sc['primary_hypotheses']['P1_echo_6h_nostop']}`",
         f"- **P2 hour17 tail-gate 6h:** `{sc['primary_hypotheses']['P2_hour17_tailgate_6h']}`",
         "",
         "## Cells (noov_n · WR · avg · tail-rate · CI · verdict)",
         "| cell | label | noovN | WR | avg | tail% | CI | in-samp | verdict |",
         "|---|---|--:|--:|--:|--:|--|--:|---|"]
    for key, c in sc["cells"].items():
        ci = f"[{c['ci_lo']}, {c['ci_hi']}]" if c["ci_lo"] is not None else "—"
        tail = f"{100 * c['tail_rate']:.0f}" if c["tail_rate"] is not None else "—"
        star = "★" if c["primary"] else ""
        disp = key.replace("|", "/")   # '|' in a cell label would break the markdown table
        L.append(f"| {star}{disp} | {c['label']} | {c['noov_n']} | {c['wr'] if c['wr'] is not None else '—'} | "
                 f"{c['avg'] if c['avg'] is not None else '—'} | {tail} | {ci} | "
                 f"{c['insample_noov_avg'] if c['insample_noov_avg'] is not None else '—'} | {c['verdict']} |")
    L += ["", "## hour17 tail-gate (§169; primary metric = tail-rate)",
          "| horizon | hi noovN | hi tail% | lo tail% | verdict |", "|---|--:|--:|--:|---|"]
    for hk, g in sc["gate"].items():
        htr = f"{100 * g['hi_tail_rate']:.0f}" if g["hi_tail_rate"] is not None else "—"
        ltr = f"{100 * g['lo_tail_rate']:.0f}" if g["lo_tail_rate"] is not None else "—"
        star = "★" if g["primary"] else ""
        L.append(f"| {star}h{hk[1:]} | {g['hi_noov_n']} | {htr} | {ltr} | {g['verdict']} |")
    L += ["", f"> {sc['note']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="(default) single pass; no daemon")
    ap.add_argument("--ledger", help="override ledger path (isolated dry-run)")
    ap.add_argument("--out-json", help="override json output path")
    ap.add_argument("--out-md", help="override md output path")
    args = ap.parse_args()
    global LEDGER, OUT_JSON, OUT_MD
    if args.ledger:
        LEDGER = Path(args.ledger)
    if args.out_json:
        OUT_JSON = Path(args.out_json)
    if args.out_md:
        OUT_MD = Path(args.out_md)

    sc = evaluate()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(sc, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_md(sc), encoding="utf-8")
    print(f"scorecard: status={sc['status']} rows={sc['forward']['resolve_rows']} "
          f"P1={sc['primary_hypotheses']['P1_echo_6h_nostop']} "
          f"P2={sc['primary_hypotheses']['P2_hour17_tailgate_6h']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
