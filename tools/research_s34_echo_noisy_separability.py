"""
research_s34_echo_noisy_separability.py — can the LOOKAHEAD be replaced by a CAUSAL signal?
(read-only, DESCRIPTIVE / hypothesis-generation only, OD-029 safe)

Context: research_s34_echo_causal_vs_lookahead showed the frozen `not noisy` (T+30m) gate is
doing ~half the per-trade edge AND removing 100% of the 4h tails with hindsight. The causal core
survives in-sample but tail-heavy. Question (user): is there a T0-KNOWABLE pattern that separates
the noisy / tail events, so the lookahead could be replaced causally?

METHOD: over the CAUSAL echo set (cand_causal), univariate rank-AUC of every T0 feature (event
features + the 4 impact/liquidity proxies joined from ECHO_LIQ_IMPACT_PROXIES_v1_records.jsonl)
against two labels:
    - noisy       : the frozen T+30m lookahead label (what the gate uses)
    - tail_4h     : (net_4h < -100) the actual disaster outcome the gate removes
AUC ~ 0.5 => not separable at T0 => lookahead irreplaceable, tail irreducible (matches §162).
AUC >> 0.5 => a FORWARD hypothesis (candidate causal gate) — NOT adopted or thresholded here.

HARD LINE: this generates hypotheses, it does NOT select a gate or claim return improvement. Any
"use feature X as a gate" decision is FORWARD-ONLY (fresh prereg, OD-028/029). No threshold picked.
Small-n: ~118 events, ~14 tails — AUCs are fragile; treated as descriptive, not inferential.

Reuses build_events from the gauntlet (numbers reconcile). Read-only, deterministic.
"""
from __future__ import annotations
import json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations, load_mark_index, reconstruct_anchors,
)
from tools.research_s34_echo_live_gauntlet import (  # noqa: E402
    build_events, load_vol_state, regime, ETH_THRESH, FEE_BPS, LOOKBACK_MS,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
PROXY_RECORDS = OUT_DIR / "ECHO_LIQ_IMPACT_PROXIES_v1_records.jsonl"
OUT_JSON = OUT_DIR / "S34_ECHO_NOISY_SEPARABILITY.json"
OUT_MD = OUT_DIR / "S34_ECHO_NOISY_SEPARABILITY.md"

PROXY_KEYS = ["kyle_lambda", "amihud", "rv_bps", "bv_bps", "jump_frac", "liq_impact_bps_per_M"]
EVENT_KEYS = ["btc4h", "btc7d", "btc3d", "rn", "sync_k", "score", "prebuildup", "be_ratio",
              "vd_now", "hour", "dow"]


def cand_causal(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] not in {0, 2} and ev["echo_30_90"] and regime(ev))


def rank_auc(values, labels):
    """AUC = P(feature ranks higher | label=1). Tie-corrected mid-ranks. Deterministic."""
    pairs = [(v, y) for v, y in zip(values, labels) if v is not None]
    n1 = sum(1 for _, y in pairs if y)
    n0 = sum(1 for _, y in pairs if not y)
    if n1 == 0 or n0 == 0:
        return None, n1, n0
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    sum_r1 = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y)
    auc = (sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return auc, n1, n0


def med(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def main():
    proxy = {}
    if PROXY_RECORDS.exists():
        for line in PROXY_RECORDS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            proxy[int(r["anchor_ts_ms"])] = r
    else:
        print("WARN: proxy records missing; proxy features will be null")

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", now_ms - LOOKBACK_MS, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        vol_rows = load_vol_state(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)

    # causal echo population + attach proxies + labels
    rows = []
    for ev in events:
        if not cand_causal(ev):
            continue
        g4 = ev.get("g_t0_4h")
        if g4 is None:
            continue
        p = proxy.get(int(ev["ts"]), {})
        rec = {k: ev.get(k) for k in EVENT_KEYS}
        for k in PROXY_KEYS:
            rec[k] = p.get(k)
        rec["_noisy"] = 1 if ev["noisy"] else 0
        rec["_tail_4h"] = 1 if (g4 - FEE_BPS) < -100 else 0
        rec["_net_4h"] = round(g4 - FEE_BPS, 1)
        rows.append(rec)

    feats = EVENT_KEYS + PROXY_KEYS
    result = {"tool": "echo_noisy_separability", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "n_causal": len(rows), "n_noisy": sum(r["_noisy"] for r in rows),
              "n_tail_4h": sum(r["_tail_4h"] for r in rows),
              "frame": "DESCRIPTIVE separability / hypothesis-generation. AUC~0.5 => lookahead "
                       "irreplaceable causally (tail irreducible, matches §162). AUC>>0.5 => FORWARD "
                       "hypothesis only. NO gate selected, NO threshold, NO return claim. Small-n fragile.",
              "auc": {}}
    for label in ("_noisy", "_tail_4h"):
        ys = [r[label] for r in rows]
        tbl = []
        for f in feats:
            vs = [r[f] for r in rows]
            auc, n1, n0 = rank_auc(vs, ys)
            if auc is None:
                continue
            m1 = med([r[f] for r in rows if r[label] == 1])
            m0 = med([r[f] for r in rows if r[label] == 0])
            tbl.append({"feature": f, "auc": round(auc, 3), "sep": round(abs(auc - 0.5), 3),
                        "n1": n1, "n0": n0, "median_label1": m1, "median_label0": m0})
        tbl.sort(key=lambda d: -d["sep"])
        result["auc"][label] = tbl

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    L = ["# Echo — Noisy / Tail Causal Separability (can the lookahead be replaced?)", "",
         "_%s · READ-ONLY · causal set N=%d · noisy=%d · tail_4h=%d_" % (
             result["generated_utc"], result["n_causal"], result["n_noisy"], result["n_tail_4h"]),
         "", "> " + result["frame"], ""]
    labelname = {"_noisy": "Predict `noisy` (the T+30m lookahead label)",
                 "_tail_4h": "Predict `tail_4h` (net_4h < -100, the actual disaster)"}
    for label in ("_noisy", "_tail_4h"):
        L += ["## %s" % labelname[label], "",
              "| feature | AUC | |AUC-.5| | med(label=1) | med(label=0) | n1 | n0 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for d in result["auc"][label]:
            def f(x): return "—" if x is None else ("%.4g" % x)
            L.append("| %s | %.3f | %.3f | %s | %s | %d | %d |" % (
                d["feature"], d["auc"], d["sep"], f(d["median_label1"]), f(d["median_label0"]),
                d["n1"], d["n0"]))
        L.append("")
    L += ["## Read",
          "- Max |AUC-.5| near 0 (~<=0.10) across features => noisy/tail is T0-UNSEPARABLE => the",
          "  lookahead cannot be replaced by any causal parameter here; the frozen edge is not causally",
          "  reproducible and its pristine tail-0 was pure hindsight. Consistent with §162 (tail AUC~0.5).",
          "- A feature with |AUC-.5| clearly elevated is a FORWARD HYPOTHESIS ONLY — candidate causal gate",
          "  to record in the forward ledger and validate post-2026-07-20. It is NOT adopted or thresholded",
          "  on this burned sample. (n is small; a single-sample AUC is not evidence of a real gate.)", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # console: top separators per label
    top = {lab: [(d["feature"], d["auc"], d["sep"]) for d in result["auc"][lab][:5]]
           for lab in ("_noisy", "_tail_4h")}
    print(json.dumps({"n_causal": result["n_causal"], "n_noisy": result["n_noisy"],
                      "n_tail_4h": result["n_tail_4h"], "top_separators": top}, indent=2))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
