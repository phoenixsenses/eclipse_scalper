"""
research_s34_tail_forensics.py — TAIL forensics for the faded liquidation flush.

Operator: "en son iyiydik, tail'i araştıralım." (§161 session regime was the last live thread.)

CONTEXT (SYSTEM_STATE §159-161): every AUC study so far measured the MEDIAN bounce
DIRECTION and hit a hard 0.55 ceiling. But the money in a fade is not lost on the median
event — it is lost on the fat LEFT tail: the minority of SELL-flushes that do NOT bounce and
keep falling (the "-447 bps disasters"). The tail is a RARE-EVENT classification problem,
distinct from median direction, and it is ASYMMETRIC: even with no harvestable bounce,
*avoiding* the tail has value. Known result: the tail is reactive-predictable (heavy selling
in the first 60s) but T0-unpredictable. This script pushes on that:

  Q1  WHERE does the tail live? Tail-rate by T0-observable bucket (session §161, trend4h,
      cluster size, forced-print size, funding crowding, pre-move magnitude, vol regime).
  Q2  Does §161's session structure extend to the TAIL? Hypothesis: US (big directional
      moves) is where the tails live -> "don't fade in US" would be a T0 tail-avoidance rule.
  Q3  How good is the reactive-60s CVD detector AS A TAIL CLASSIFIER (precision/recall/AUC on
      the tail label)? It is the one thing known to work; it was never quantified.
  Q4  Does any separation survive a temporal TRAIN/TEST split + bootstrap CI, or is it fragile
      like everything else?

Tail depth is measured as MAE (max adverse excursion over the hold path), not just endpoint —
the disaster is a path minimum. Read-only (DB mode=ro), causal, no outcome tuning of anchors.
NOT a claim of edge; a forensic characterization + candidate avoidance rule.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    MarkIndex,
    load_liquidations,
    load_mark_index_range,
    mean,
    pctile,
    reconstruct_anchors,
    signed_return_bps,
)
from tools.liq_indicator_library import compute_indicators  # noqa: E402

DEFAULT_DB = "file:data/microstructure.db?mode=ro"
OUT_MD = ROOT / "reports/research/s34/S34_TAIL_FORENSICS.md"
OUT_JSON = ROOT / "reports/research/s34/S34_TAIL_FORENSICS.json"


# ----------------------------- helpers -------------------------------------

def session_of(ts_ms: int) -> str:
    h = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).hour
    if h < 7:
        return "ASIA"
    if h < 13:
        return "EU"
    return "US"


def mae_bps(marks: MarkIndex, entry_px: float, t0: int, horizon_sec: int, direction: str = "LONG") -> float | None:
    """Max ADVERSE excursion (most negative signed return) over [t0, t0+horizon]. Tail depth."""
    path = marks.slice_range(t0, t0 + horizon_sec * 1000)
    if not path or entry_px <= 0:
        return None
    worst = 0.0
    for _, px in path:
        r = signed_return_bps(direction, entry_px, px)
        if r < worst:
            worst = r
    return worst


def cvd_first_60s_musd(conn: sqlite3.Connection, symbol: str, t0: int) -> float:
    """Reactive continuation signal (liq_signal_system.continuation_check): buy-sell notional
    in the first 60s after the anchor. Negative = selling continues (the disaster regime)."""
    r = conn.execute(
        "SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
        "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades "
        "WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (symbol, t0, t0 + 60_000)).fetchone()
    return ((r[0] or 0.0) - (r[1] or 0.0)) / 1e6


def auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC of `score` predicting label==1 (higher score -> more likely tail). Rank/Mann-Whitney."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    rank = {}
    i = 0
    r_accum = 0
    # average-rank tie handling
    n = len(paired)
    ranks = [0.0] * n
    j = 0
    while j < n:
        k = j
        while k + 1 < n and paired[k + 1][0] == paired[j][0]:
            k += 1
        avg = (j + k) / 2.0 + 1.0
        for m in range(j, k + 1):
            ranks[m] = avg
        j = k + 1
    sum_pos_rank = sum(rk for rk, (_, y) in zip(ranks, paired) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos_rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def rate_ci(k: int, n: int, iters: int = 2000, seed: int = 12345) -> tuple[float, float, float]:
    """Deterministic bootstrap CI for a rate k/n (no Math.random ban issue — LCG)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    state = seed
    rates = []
    for _ in range(iters):
        hits = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            if (state / 0x7FFFFFFF) < p:
                hits += 1
        rates.append(hits / n)
    rates.sort()
    return (p, rates[int(0.025 * iters)], rates[int(0.975 * iters)])


def bucket_tail_rates(rows: list[dict], feat: str, label_key: str, kind: str, edges=None) -> list[dict]:
    """Tail-rate per bucket of a feature. kind='session' categorical, else numeric tercile."""
    if kind == "session":
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["session"], []).append(r)
        order = ["ASIA", "EU", "US"]
        out = []
        for g in order:
            gr = groups.get(g, [])
            if not gr:
                continue
            k = sum(x[label_key] for x in gr)
            p, lo, hi = rate_ci(k, len(gr))
            out.append({"bucket": g, "n": len(gr), "tail_k": k, "tail_rate": round(p, 3),
                        "ci95": [round(lo, 3), round(hi, 3)]})
        return out
    vals = sorted(r[feat] for r in rows if r.get(feat) is not None and math.isfinite(r[feat]))
    if len(vals) < 6:
        return []
    q33 = pctile(vals, 1 / 3)
    q67 = pctile(vals, 2 / 3)
    labels = [("low", -math.inf, q33), ("mid", q33, q67), ("high", q67, math.inf)]
    out = []
    for name, lo_e, hi_e in labels:
        gr = [r for r in rows if r.get(feat) is not None and math.isfinite(r[feat])
              and (lo_e <= r[feat] < hi_e or (name == "high" and r[feat] >= q67))]
        if name == "high":
            gr = [r for r in rows if r.get(feat) is not None and math.isfinite(r[feat]) and r[feat] >= q67]
        elif name == "low":
            gr = [r for r in rows if r.get(feat) is not None and math.isfinite(r[feat]) and r[feat] < q33]
        else:
            gr = [r for r in rows if r.get(feat) is not None and math.isfinite(r[feat]) and q33 <= r[feat] < q67]
        if not gr:
            continue
        k = sum(x[label_key] for x in gr)
        p, clo, chi = rate_ci(k, len(gr))
        out.append({"bucket": f"{name}(<{round(q33,2)}|{round(q67,2)})" if name != "mid" else f"mid[{round(q33,2)},{round(q67,2)})",
                    "n": len(gr), "tail_k": k, "tail_rate": round(p, 3), "ci95": [round(clo, 3), round(chi, 3)]})
    return out


# ----------------------------- main ----------------------------------------

def build_population(conn, args) -> list[dict[str, Any]]:
    liqs = load_liquidations(conn, args.symbol, args.side, None, None)
    if not liqs:
        return []
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=args.bucket_sec,
        min_gap_sec=args.min_gap_sec,
        thresholds=(args.threshold,),
        accel_window_sec=args.accel_window_sec,
    )
    if not anchors:
        return []
    t_lo = min(a.anchor_ts_ms for a in anchors) - 5 * 3_600_000
    t_hi = max(a.anchor_ts_ms for a in anchors) + args.horizon_sec * 1000 + 3_600_000
    marks = load_mark_index_range(conn, args.symbol, t_lo, t_hi)

    direction = "LONG" if args.side == "SELL" else "SHORT"
    rows: list[dict[str, Any]] = []
    for a in anchors:
        em = marks.at_or_after(a.anchor_ts_ms)
        if not em:
            continue
        entry_px = float(em[1])
        xm = marks.at_or_after(a.anchor_ts_ms + args.horizon_sec * 1000)
        if not xm:
            continue
        endpoint = signed_return_bps(direction, entry_px, float(xm[1]))
        mae = mae_bps(marks, entry_px, a.anchor_ts_ms, args.horizon_sec, direction)
        if mae is None:
            continue
        ind = compute_indicators(conn, a.anchor_ts_ms).values
        cvd60 = cvd_first_60s_musd(conn, args.symbol, a.anchor_ts_ms)
        rows.append({
            "anchor_ts_ms": a.anchor_ts_ms,
            "utc": datetime.fromtimestamp(a.anchor_ts_ms / 1000, timezone.utc).isoformat(),
            "session": session_of(a.anchor_ts_ms),
            "endpoint_bps": round(endpoint, 1),
            "mae_bps": round(mae, 1),
            # T0-observable features
            "running_notional": a.running_notional,
            "max_single_notional": a.max_single_notional,
            "running_liq_count": a.running_liq_count,
            "running_accel": a.running_accel,
            "trend4h_bps": ind.get("ret_4h_bps"),
            "trend1h_bps": ind.get("ret_1h_bps"),
            "premove_15m_bps": ind.get("ret_15m_bps"),
            "funding_pctile_14d": ind.get("funding_pctile_14d"),
            "rv_5m": ind.get("rv_5m"),
            "vol_decile": ind.get("vol_decile"),
            "flow_sell_imb_60s": ind.get("flow_sell_imbalance_60s"),
            # reactive (NOT T0): the known detector
            "cvd_first60s_musd": round(cvd60, 3),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")  # SELL-flush -> LONG fade
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--horizon-sec", type=int, default=4 * 3600)  # 4h fade hold
    p.add_argument("--tail-mae-bps", type=float, default=-150.0)  # disaster threshold
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", default=str(OUT_JSON))
    p.add_argument("--md-out", default=str(OUT_MD))
    args = p.parse_args()

    conn = sqlite3.connect(args.db, uri=True)
    conn.execute("PRAGMA query_only=1")

    rows = build_population(conn, args)
    conn.close()
    n = len(rows)
    if n == 0:
        print(json.dumps({"error": "no anchors/population"}))
        return

    rows.sort(key=lambda r: r["anchor_ts_ms"])
    # labels
    endpoints = [r["endpoint_bps"] for r in rows]
    maes = [r["mae_bps"] for r in rows]
    p10_end = pctile(endpoints, 0.10)
    for r in rows:
        r["tail_mae"] = 1 if r["mae_bps"] <= args.tail_mae_bps else 0
        r["tail_end"] = 1 if r["endpoint_bps"] <= p10_end else 0

    tail_n = sum(r["tail_mae"] for r in rows)
    base_p, base_lo, base_hi = rate_ci(tail_n, n)

    # temporal split
    cut = int(n * (1 - args.holdout_frac))
    train, test = rows[:cut], rows[cut:]

    FEATS = [
        ("session", "session"),
        ("trend4h_bps", "num"), ("trend1h_bps", "num"), ("premove_15m_bps", "num"),
        ("running_notional", "num"), ("max_single_notional", "num"), ("running_liq_count", "num"),
        ("running_accel", "num"), ("funding_pctile_14d", "num"), ("rv_5m", "num"),
        ("flow_sell_imb_60s", "num"),
    ]
    label_key = "tail_mae"
    feat_report = {}
    for feat, kind in FEATS:
        overall = bucket_tail_rates(rows, feat, label_key, "session" if kind == "session" else "num")
        tr = bucket_tail_rates(train, feat, label_key, "session" if kind == "session" else "num")
        te = bucket_tail_rates(test, feat, label_key, "session" if kind == "session" else "num")
        a_all = None
        if kind == "num":
            sc = [r[feat] for r in rows if r.get(feat) is not None and math.isfinite(r[feat])]
            ly = [r[label_key] for r in rows if r.get(feat) is not None and math.isfinite(r[feat])]
            a_all = auc(sc, ly)
        feat_report[feat] = {"auc_tail": round(a_all, 3) if a_all is not None else None,
                             "overall": overall, "train": tr, "test": te}

    # reactive detector as a tail classifier (cvd<threshold => predict tail)
    def detector_stats(thr_musd: float) -> dict:
        tp = sum(1 for r in rows if r["cvd_first60s_musd"] < thr_musd and r["tail_mae"] == 1)
        fp = sum(1 for r in rows if r["cvd_first60s_musd"] < thr_musd and r["tail_mae"] == 0)
        fn = sum(1 for r in rows if r["cvd_first60s_musd"] >= thr_musd and r["tail_mae"] == 1)
        flagged = tp + fp
        prec = tp / flagged if flagged else None
        rec = tp / (tp + fn) if (tp + fn) else None
        return {"thr_musd": thr_musd, "flagged": flagged, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 3) if prec is not None else None,
                "recall": round(rec, 3) if rec is not None else None}
    cvd_scores = [-r["cvd_first60s_musd"] for r in rows]  # more-negative-cvd => higher tail score
    cvd_auc = auc(cvd_scores, [r["tail_mae"] for r in rows])
    detector = {"auc_reactive_cvd": round(cvd_auc, 3) if cvd_auc is not None else None,
                "at_thresholds": [detector_stats(t) for t in (0.0, -0.5, -1.0, -2.0)]}

    # avoidance value: mean net if we DROP the worst session/bucket (illustrative, fee-aware)
    def mean_net(subset, fee_side=3.05):
        vals = [r["endpoint_bps"] - 2 * fee_side for r in subset]
        return {"n": len(vals), "mean_net_bps": round(mean(vals), 1) if vals else None,
                "tail_rate": round(sum(x["tail_mae"] for x in subset) / len(subset), 3) if subset else None}

    summary = {
        "population": {"symbol": args.symbol, "side": args.side, "direction": "LONG" if args.side == "SELL" else "SHORT",
                       "n": n, "threshold": args.threshold, "horizon_sec": args.horizon_sec,
                       "date_range": [rows[0]["utc"], rows[-1]["utc"]]},
        "outcome_shape": {
            "endpoint_bps": {"mean": round(mean(endpoints), 1), "median": round(pctile(endpoints, 0.5), 1),
                             "p10": round(p10_end, 1), "min": round(min(endpoints), 1), "max": round(max(endpoints), 1)},
            "mae_bps": {"mean": round(mean(maes), 1), "median": round(pctile(maes, 0.5), 1),
                        "p10": round(pctile(maes, 0.10), 1), "min": round(min(maes), 1)},
            "tail_def": f"mae_bps <= {args.tail_mae_bps}",
            "tail_rate": {"k": tail_n, "n": n, "rate": round(base_p, 3), "ci95": [round(base_lo, 3), round(base_hi, 3)]},
        },
        "Q1_Q2_features_vs_tail": feat_report,
        "Q3_reactive_detector": detector,
        "Q4_avoidance_illustration": {
            "all": mean_net(rows),
            "drop_US_session": mean_net([r for r in rows if r["session"] != "US"]),
            "only_US_session": mean_net([r for r in rows if r["session"] == "US"]),
        },
    }

    Path(args.json_out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # markdown
    L = []
    L.append("# S34 Tail Forensics — Faded Liquidation Flush\n")
    L.append(f"_Read-only, causal. {args.symbol} {args.side}-flush -> "
             f"{'LONG' if args.side=='SELL' else 'SHORT'} fade, >= {int(args.threshold/1000)}K, "
             f"horizon {args.horizon_sec//3600}h. n={n}._\n")
    o = summary["outcome_shape"]
    L.append("## Outcome shape\n")
    L.append(f"- Endpoint bps: mean {o['endpoint_bps']['mean']}, median {o['endpoint_bps']['median']}, "
             f"p10 {o['endpoint_bps']['p10']}, min {o['endpoint_bps']['min']}, max {o['endpoint_bps']['max']}")
    L.append(f"- MAE bps (path worst): mean {o['mae_bps']['mean']}, median {o['mae_bps']['median']}, "
             f"p10 {o['mae_bps']['p10']}, min {o['mae_bps']['min']}")
    L.append(f"- **Tail** ({o['tail_def']}): {tail_n}/{n} = {round(base_p*100,1)}% "
             f"(95% CI {round(base_lo*100,1)}-{round(base_hi*100,1)}%)\n")
    L.append("## Q1/Q2 — where does the tail live (T0-observable)\n")
    for feat, kind in FEATS:
        fr = feat_report[feat]
        L.append(f"### {feat}  (tail-AUC={fr['auc_tail']})")
        L.append("| split | bucket | n | tail% | CI95 |")
        L.append("|---|---|---|---|---|")
        for split_name, key in (("all", "overall"), ("train", "train"), ("test", "test")):
            for b in fr[key]:
                L.append(f"| {split_name} | {b['bucket']} | {b['n']} | {round(b['tail_rate']*100,1)} | "
                         f"{round(b['ci95'][0]*100,1)}-{round(b['ci95'][1]*100,1)} |")
        L.append("")
    d = summary["Q3_reactive_detector"]
    L.append(f"## Q3 — reactive 60s-CVD as tail classifier (AUC={d['auc_reactive_cvd']})\n")
    L.append("| cvd<thr($M) | flagged | TP | FP | precision | recall |")
    L.append("|---|---|---|---|---|---|")
    for s in d["at_thresholds"]:
        L.append(f"| {s['thr_musd']} | {s['flagged']} | {s['tp']} | {s['fp']} | {s['precision']} | {s['recall']} |")
    av = summary["Q4_avoidance_illustration"]
    L.append("\n## Q4 — session avoidance (illustrative, fee 3.05/side)\n")
    L.append("| subset | n | mean_net_bps | tail_rate |")
    L.append("|---|---|---|---|")
    for k, v in av.items():
        L.append(f"| {k} | {v['n']} | {v['mean_net_bps']} | {v['tail_rate']} |")
    L.append("\n_Descriptive forensic. No edge claim; separation is trusted only if it holds "
             "on BOTH train and test with non-overlapping CIs._")
    Path(args.md_out).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({"n": n, "tail_rate": round(base_p, 3), "tail_k": tail_n,
                      "reactive_cvd_auc": detector["auc_reactive_cvd"],
                      "session_overall": feat_report["session"]["overall"],
                      "md": args.md_out, "json": args.json_out}, indent=2, default=str))


if __name__ == "__main__":
    main()
