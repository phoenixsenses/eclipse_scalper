"""
research_s34_echo_separability_stats.py — full statistical panel for noisy/tail separability
(read-only, DESCRIPTIVE / hypothesis-generation, OD-029 safe).

Extends research_s34_echo_noisy_separability with what AUC alone cannot say:
  - Cliff's delta / rank-biserial (= 2*AUC - 1)   effect size, robust to small-n AUC inflation
  - Mann-Whitney U + two-sided p (tie-corrected normal approx, continuity correction)
  - median + IQR (p25/p75) per class
  - Benjamini-Hochberg FDR across the feature panel (17 feats x 2 labels multiplicity)
  - be_ratio computed BOTH tainted (gauntlet ±10m, uses future) AND causal ([ts-10m, ts]) to
    expose exactly how much of its separation is lookahead.
  - boxplot + violin(KDE) PNGs for the clean-causal features.

HARD LINE (unchanged): these numbers PRIORITIZE FORWARD hypotheses; they do NOT select a gate or a
threshold on the burned sample. p<0.05 in-sample is a nomination, not a decision. N_tail=14 => any
"significant" tail separator is fragile; forward is the only judge (OD-028/029). No return claim.

Reuses build_events from the gauntlet. Read-only, deterministic.
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
    build_events, load_vol_state, regime, liq_max, ETH_THRESH, FEE_BPS, LOOKBACK_MS,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
PROXY_RECORDS = OUT_DIR / "ECHO_LIQ_IMPACT_PROXIES_v1_records.jsonl"
OUT_JSON = OUT_DIR / "S34_ECHO_SEPARABILITY_STATS.json"
OUT_MD = OUT_DIR / "S34_ECHO_SEPARABILITY_STATS.md"
REC_OUT = OUT_DIR / "S34_ECHO_SEPARABILITY_STATS_records.jsonl"

PROXY_KEYS = ["kyle_lambda", "amihud", "rv_bps", "bv_bps", "jump_frac", "liq_impact_bps_per_M"]
EVENT_KEYS = ["btc4h", "btc7d", "btc3d", "rn", "sync_k", "score", "prebuildup", "vd_now", "hour", "dow"]
# be_ratio handled separately (tainted + causal)


def cand_causal(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] not in {0, 2} and ev["echo_30_90"] and regime(ev))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rank_stats(values, labels):
    """Returns AUC, Cliff's delta, Mann-Whitney U, tie-corrected two-sided p, n1, n0."""
    pairs = [(v, y) for v, y in zip(values, labels) if v is not None]
    n1 = sum(1 for _, y in pairs if y); n0 = sum(1 for _, y in pairs if not y)
    if n1 == 0 or n0 == 0:
        return None
    n = n1 + n0
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs); tie_term = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        t = j - i + 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        tie_term += t ** 3 - t
        i = j + 1
    R1 = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    auc = U1 / (n1 * n0)
    delta = 2.0 * auc - 1.0
    mu = n1 * n0 / 2.0
    var = (n1 * n0 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        p = 1.0
    else:
        z = (abs(U1 - mu) - 0.5) / math.sqrt(var)   # continuity correction
        z = max(z, 0.0)
        p = 2.0 * (1.0 - _norm_cdf(z))
    return {"auc": auc, "cliffs_delta": delta, "U": U1, "p": p, "n1": n1, "n0": n0}


def _pctile(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return vals[int(k)] if f == c else vals[f] * (c - k) + vals[c] * (k - f)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0] * m; prev = 1.0
    for rank, i in enumerate(reversed(idx)):
        k = m - rank
        val = min(prev, pvals[i] * m / k)
        adj[i] = val; prev = val
    return adj


def main():
    proxy = {}
    if PROXY_RECORDS.exists():
        for line in PROXY_RECORDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); proxy[int(r["anchor_ts_ms"])] = r

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-128000"); conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", now_ms - LOOKBACK_MS, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        vol_rows = load_vol_state(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)
        # causal be_ratio for the causal echo set (btc concentration [ts-10m, ts] only)
        causal_bt = {}
        for ev in events:
            if cand_causal(ev):
                bc = liq_max(conn, "BTCUSDT", "SELL", ev["ts"] - 10 * 60_000, ev["ts"])
                causal_bt[ev["ts"]] = (bc / ev["rn"]) if ev["rn"] > 0 else 0.0

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
        rec["be_ratio_tainted"] = ev.get("be_ratio")       # gauntlet ±10m (USES FUTURE)
        rec["be_ratio_causal"] = causal_bt.get(ev["ts"])    # [ts-10m, ts] only
        rec["_noisy"] = 1 if ev["noisy"] else 0
        rec["_tail_4h"] = 1 if (g4 - FEE_BPS) < -100 else 0
        rec["_net_4h"] = round(g4 - FEE_BPS, 1)
        rec["ts"] = ev["ts"]
        rows.append(rec)

    feats = ["be_ratio_causal", "be_ratio_tainted"] + EVENT_KEYS + PROXY_KEYS
    out = {"tool": "echo_separability_stats", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_causal": len(rows), "n_noisy": sum(r["_noisy"] for r in rows),
           "n_tail_4h": sum(r["_tail_4h"] for r in rows),
           "frame": "Effect sizes + significance PRIORITIZE FORWARD hypotheses; NO gate/threshold "
                    "selected on burned data. N_tail=14 => fragile. be_ratio_tainted uses FUTURE data "
                    "(reference only). BH-FDR across the panel. Proof is FORWARD (OD-028/029).",
           "targets": {"auc_good": 0.60, "auc_strong": 0.65, "cliffs_medium": 0.25,
                       "cliffs_strong": 0.40, "p": 0.05},
           "panel": {}}
    for label in ("_noisy", "_tail_4h"):
        ys = [r[label] for r in rows]
        entries = []
        for f in feats:
            vs = [r[f] for r in rows]
            rs = rank_stats(vs, ys)
            if rs is None:
                continue
            m1 = _pctile([r[f] for r in rows if r[label] == 1], .50)
            m0 = _pctile([r[f] for r in rows if r[label] == 0], .50)
            iqr1 = [_pctile([r[f] for r in rows if r[label] == 1], .25),
                    _pctile([r[f] for r in rows if r[label] == 1], .75)]
            iqr0 = [_pctile([r[f] for r in rows if r[label] == 0], .25),
                    _pctile([r[f] for r in rows if r[label] == 0], .75)]
            entries.append({"feature": f, **rs, "median1": m1, "median0": m0,
                            "iqr1": iqr1, "iqr0": iqr0})
        padj = bh_fdr([e["p"] for e in entries])
        for e, pa in zip(entries, padj):
            e["p_bh"] = pa
        entries.sort(key=lambda e: -abs(e["cliffs_delta"]))
        out["panel"][label] = entries

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    REC_OUT.write_text("\n".join(json.dumps(r, default=str) for r in rows), encoding="utf-8")

    def sig(e):
        s = ""
        if e["p"] < 0.05: s += "*"
        if e["p_bh"] < 0.05: s += "†"
        if e["feature"] == "be_ratio_tainted": s += " ⚠LOOKAHEAD"
        return s
    L = ["# Echo Separability — Effect Sizes + Significance (forward prioritization)", "",
         "_%s · READ-ONLY · causal N=%d · noisy=%d · tail_4h=%d_" % (
             out["generated_utc"], out["n_causal"], out["n_noisy"], out["n_tail_4h"]),
         "", "> " + out["frame"], "",
         "Targets: AUC>0.60 good / >0.65 strong · |Cliff δ|>0.25 med / >0.40 strong · p<0.05 · "
         "`*`=raw p<.05 `†`=BH-FDR<.05 `⚠`=uses future.", ""]
    name = {"_noisy": "Predict `noisy` (removed=80 vs kept=38 — the lookahead's own filter)",
            "_tail_4h": "Predict `tail_4h` (net<-100; N=14 — fragile)"}
    for label in ("_noisy", "_tail_4h"):
        L += ["## %s" % name[label], "",
              "| feature | AUC | Cliff δ | p | p(BH) | med1 [IQR] | med0 [IQR] | n1/n0 | sig |",
              "|---|---:|---:|---:|---:|---|---|---:|---|"]
        for e in out["panel"][label]:
            def g(x): return "—" if x is None else ("%.4g" % x)
            L.append("| %s | %.3f | %+.3f | %.3f | %.3f | %s [%s,%s] | %s [%s,%s] | %d/%d | %s |" % (
                e["feature"], e["auc"], e["cliffs_delta"], e["p"], e["p_bh"],
                g(e["median1"]), g(e["iqr1"][0]), g(e["iqr1"][1]),
                g(e["median0"]), g(e["iqr0"][0]), g(e["iqr0"][1]), e["n1"], e["n0"], sig(e)))
        L.append("")
    L += ["## Read",
          "- Clean-causal features only (ignore ⚠ be_ratio_tainted). If the best clean |Cliff δ| stays",
          "  below ~0.25 and nothing survives BH-FDR → noisy/tail is NOT causally separable in-sample →",
          "  the lookahead is irreplaceable and the frozen tail-0 was hindsight (matches §162).",
          "- be_ratio_tainted vs be_ratio_causal delta = the lookahead inflation, measured directly.",
          "- Any elevated clean feature is a FORWARD hypothesis to pre-register + record, NOT a gate to",
          "  adopt now. Boxplots: S34_ECHO_SEP_*.png.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # ---- viz: boxplot + violin(KDE) for clean-causal features ----
    viz_note = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_feats = {"_noisy": ["be_ratio_causal", "be_ratio_tainted", "rv_bps", "amihud", "btc4h"],
                      "_tail_4h": ["be_ratio_causal", "hour", "jump_frac", "bv_bps"]}
        for label, fl in plot_feats.items():
            fig, axes = plt.subplots(1, len(fl), figsize=(3.2 * len(fl), 4.2))
            if len(fl) == 1:
                axes = [axes]
            for ax, f in zip(axes, fl):
                d0 = [r[f] for r in rows if r[label] == 0 and r[f] is not None]
                d1 = [r[f] for r in rows if r[label] == 1 and r[f] is not None]
                if d0 and d1:
                    parts = ax.violinplot([d0, d1], showextrema=False)
                    for b in parts['bodies']:
                        b.set_alpha(0.25)
                    ax.boxplot([d0, d1], widths=0.25, showfliers=True)
                ax.set_xticks([1, 2]); ax.set_xticklabels(["0", "1"])
                tainted = " (⚠future)" if f == "be_ratio_tainted" else ""
                ax.set_title("%s%s" % (f, tainted), fontsize=9)
            fig.suptitle("%s split (0 vs 1) — causal echo set N=%d" % (label.strip("_"), len(rows)),
                         fontsize=10)
            fig.tight_layout()
            png = OUT_DIR / ("S34_ECHO_SEP_%s.png" % label.strip("_"))
            fig.savefig(png, dpi=110); plt.close(fig)
        viz_note = "written"
    except Exception as e:
        viz_note = "SKIPPED: %s" % e

    top = {lab: [(e["feature"], round(e["auc"], 3), round(e["cliffs_delta"], 3),
                  round(e["p"], 3), round(e["p_bh"], 3))
                 for e in out["panel"][lab][:6]] for lab in ("_noisy", "_tail_4h")}
    print(json.dumps({"n_causal": out["n_causal"], "n_noisy": out["n_noisy"],
                      "n_tail_4h": out["n_tail_4h"], "viz": viz_note,
                      "top_by_effect_size": top}, indent=2))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
