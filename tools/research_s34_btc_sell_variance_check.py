"""
BTC SELL KNN Variance Check
Determines whether K=50 recency improvement is real or majority-class artifact.
"""
from __future__ import annotations
import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"

SYMBOL, SIDE, MIN_NOTIONAL, PRIMARY_ROUTE = "BTCUSDT", "SELL", 1_000_000.0, "SHORT_DELAY0_TP40"
TRAIN_RATIO = 0.70
RECENCY_HALFLIFE_DAYS = 90.0

ALL_FEATURES = [
    "cluster_notional", "cluster_duration_sec", "cluster_liq_count",
    "max_single_liq_share", "intensity_per_sec", "inter_cluster_gap_sec",
    "day_trend_bps", "day_range_bps", "symbol_pre_5m_bps", "symbol_pre_15m_bps",
    "btc_pre_15m_bps",
]
WEIGHTS = {"cluster_notional": 2.0, "cluster_duration_sec": 0.8, "cluster_liq_count": 0.8,
           "max_single_liq_share": 0.8, "intensity_per_sec": 1.0, "inter_cluster_gap_sec": 0.7,
           "day_trend_bps": 1.4, "day_range_bps": 1.0, "symbol_pre_5m_bps": 1.0,
           "symbol_pre_15m_bps": 1.0, "btc_pre_15m_bps": 0.8}


def _pctile(vals, q):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    pos = (len(c) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (pos - lo)

def _med(v): return _pctile(v, 0.5)
def _mean(v):
    c = [x for x in v if x is not None and math.isfinite(x)]
    return sum(c) / len(c) if c else None
def _std(v):
    c = [x for x in v if x is not None and math.isfinite(x)]
    if len(c) < 2: return None
    mu = sum(c) / len(c)
    return math.sqrt(sum((x - mu)**2 for x in c) / (len(c) - 1))
def _iqr(v):
    p75, p25 = _pctile(v, 0.75), _pctile(v, 0.25)
    iqr = abs((p75 or 0) - (p25 or 0))
    if iqr > 1e-9: return iqr
    s = _std(v)
    return s if s and s > 1e-9 else 1.0
def _tx(col, v): return math.log(max(v, 1.0)) if col == "cluster_notional" else v


def load_events():
    con = sqlite3.connect(f"file:{FEATURE_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM liq_event_features WHERE symbol=? AND liq_side=? AND cluster_notional>=? ORDER BY event_ts_ms ASC",
        (SYMBOL, SIDE, MIN_NOTIONAL),
    ).fetchall()
    events = [dict(r) for r in rows]
    eids = [str(e["event_id"]) for e in events]
    ph = ",".join("?" * len(eids))
    outs = con.execute(
        f"SELECT event_id, route_id, net_bps FROM liq_event_outcome_labels WHERE event_id IN ({ph})", eids
    ).fetchall()
    con.close()
    om = {}
    for r in outs:
        om.setdefault(str(r[0]), {})[str(r[1])] = float(r[2])
    for e in events:
        e["_out"] = om.get(str(e["event_id"]), {})
    return events


def knn_preds(train, test, k, metric):
    wlist = [WEIGHTS.get(f, 1.0) for f in ALL_FEATURES]

    def vec(e):
        return [_tx(f, float(e[f])) if e.get(f) is not None else None for f in ALL_FEATURES]

    tr_vecs = [vec(e) for e in train]
    te_vecs = [vec(e) for e in test]
    scales = [_iqr([v[i] for v in tr_vecs if v[i] is not None]) for i in range(len(ALL_FEATURES))]
    tr_out = [e["_out"].get(PRIMARY_ROUTE) for e in train]
    tr_ts  = [int(e.get("event_ts_ms") or 0) for e in train]

    preds, realized = [], []
    for tvec, ev in zip(te_vecs, test):
        real = ev["_out"].get(PRIMARY_ROUTE)
        if real is None: continue
        tts = int(ev.get("event_ts_ms") or 0)
        scored = []
        for j, (nvec, outcome) in enumerate(zip(tr_vecs, tr_out)):
            if outcome is None: continue
            dims = []
            for i, (t, n) in enumerate(zip(tvec, nvec)):
                if t is None or n is None: continue
                s = scales[i] if scales[i] > 1e-9 else 1.0
                w = wlist[i]
                z = (n - t) / s
                dims.append((w, z))
            if not dims: continue
            tw = sum(w for w, _ in dims)
            if tw <= 0: continue
            if metric == "euclidean":
                d = math.sqrt(sum(w * z * z for w, z in dims) / tw)
            elif metric == "recency":
                feat_d = math.sqrt(sum(w * z * z for w, z in dims) / tw)
                age_days = (tts - tr_ts[j]) / (1000 * 86400.0)
                d = feat_d * math.exp(0.693 * age_days / RECENCY_HALFLIFE_DAYS)
            scored.append((d, float(outcome)))
        if not scored: continue
        scored.sort(key=lambda x: x[0])
        nbr = [o for _, o in scored[:min(k, len(scored))]]
        pred = _med(nbr)
        if pred is not None:
            preds.append(pred)
            realized.append(float(real))
    return preds, realized


def evaluate(preds, realized, train_outcomes):
    base = _med(train_outcomes)
    da = sum(1 for p, r in zip(preds, realized) if (p > 0) == (r > 0)) / len(preds) if preds else None
    mae = _mean([abs(p - r) for p, r in zip(preds, realized)])
    pred_std = _std(preds)
    pred_range = (max(preds) - min(preds)) if preds else None
    n_unique = len(set(round(p, 1) for p in preds)) if preds else 0
    # Constant baseline: always predict sign(base)
    const_da = (sum(1 for r in realized if (base or 0) > 0 == r > 0) / len(realized)
                if realized and base is not None else None)
    const_da = (sum(1 for r in realized if r > 0) / len(realized)
                if realized and (base or 0) > 0 else
                sum(1 for r in realized if r <= 0) / len(realized)
                if realized else None)
    knn_gain = round(float(da or 0) - float(const_da or 0), 3) if da is not None and const_da is not None else None
    return {
        "n_test": len(preds),
        "dir_acc_knn": round(da, 3) if da is not None else None,
        "dir_acc_constant_baseline": round(const_da, 3) if const_da is not None else None,
        "knn_gain_over_baseline": knn_gain,
        "mae": round(mae, 1) if mae is not None else None,
        "pred_median": round(float(_med(preds)), 1) if preds else None,
        "realized_median": round(float(_med(realized)), 1) if realized else None,
        "base_rate": round(float(base), 1) if base is not None else None,
        "pred_std": round(float(pred_std), 1) if pred_std is not None else None,
        "pred_range": round(float(pred_range), 1) if pred_range is not None else None,
        "n_unique_preds": n_unique,
    }


def verdict(res_k50_recency, res_k20_eucl):
    knn_gain = res_k50_recency.get("knn_gain_over_baseline") or 0.0
    pred_std = res_k50_recency.get("pred_std") or 0.0
    n_unique = res_k50_recency.get("n_unique_preds") or 0
    n_test = res_k50_recency.get("n_test") or 1

    low_variance = pred_std < 5.0
    no_gain = knn_gain < 0.05  # less than 5pp over constant baseline
    low_diversity = n_unique < n_test * 0.3  # fewer than 30% unique prediction values

    if low_variance and no_gain:
        v = "NOT_USEFUL_BASE_RATE_ALIGNMENT"
        explain = (f"Prediction std={pred_std} bps (< 5 bps) and KNN gain={knn_gain*100:+.1f} pp over "
                   "constant baseline. K=50 predictions converge to train median — majority-class alignment confirmed. "
                   "Keep BTC SELL as BASE_RATE_ONLY.")
    elif not low_variance and knn_gain >= 0.05:
        v = "POTENTIALLY_USEFUL"
        explain = (f"Prediction std={pred_std} bps and KNN gain={knn_gain*100:+.1f} pp over constant baseline. "
                   "Some discrimination detected. Consider cautious upgrade after more data accumulates.")
    elif low_diversity:
        v = "NOT_USEFUL_LOW_DIVERSITY"
        explain = (f"Only {n_unique} unique prediction values for {n_test} test events ({n_unique/n_test*100:.0f}%). "
                   "KNN is effectively binning rather than discriminating. Keep BASE_RATE_ONLY.")
    else:
        v = "INCONCLUSIVE"
        explain = (f"Mixed signals: pred_std={pred_std}, knn_gain={knn_gain*100:+.1f} pp, "
                   f"n_unique={n_unique}/{n_test}. Insufficient evidence to upgrade from BASE_RATE_ONLY.")
    return v, explain


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    cut = max(1, min(len(events) - 1, int(len(events) * TRAIN_RATIO)))
    train, test = events[:cut], events[cut:]
    tr_out = [e["_out"].get(PRIMARY_ROUTE) for e in train if e["_out"].get(PRIMARY_ROUTE) is not None]
    print(f"BTC SELL: N={len(events)}  train={len(train)}  test={len(test)}")

    print("K=20 euclidean (baseline)...")
    p20, r20 = knn_preds(train, test, 20, "euclidean")
    res20 = evaluate(p20, r20, tr_out)
    print(f"  dir_acc={res20['dir_acc_knn']:.2f}  baseline={res20['dir_acc_constant_baseline']:.2f}  gain={res20['knn_gain_over_baseline']:+.3f}  pred_std={res20['pred_std']}")

    print("K=50 recency (suspect)...")
    p50, r50 = knn_preds(train, test, 50, "recency")
    res50 = evaluate(p50, r50, tr_out)
    print(f"  dir_acc={res50['dir_acc_knn']:.2f}  baseline={res50['dir_acc_constant_baseline']:.2f}  gain={res50['knn_gain_over_baseline']:+.3f}  pred_std={res50['pred_std']}  n_unique={res50['n_unique_preds']}/{res50['n_test']}")

    v, explain = verdict(res50, res20)
    print(f"\nVerdict: {v}")
    print(f"  {explain}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combo": f"{SYMBOL}_{SIDE}",
        "primary_route": PRIMARY_ROUTE,
        "n_events": len(events), "n_train": len(train), "n_test": len(test),
        "k20_euclidean": res20,
        "k50_recency": res50,
        "verdict": v,
        "verdict_explanation": explain,
        "recommendation": "keep BASE_RATE_ONLY" if "NOT_USEFUL" in v or "INCONCLUSIVE" in v else "consider upgrade",
    }

    def _f(v, d=1, s=True):
        if v is None: return "NA"
        return f"{float(v):+.{d}f}" if s else f"{float(v):.{d}f}"

    lines = [
        "# BTC SELL KNN Variance Check",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Combo: {SYMBOL} {SIDE}  |  Route: {PRIMARY_ROUTE}",
        f"N events: {len(events)}  |  Train: {len(train)} (70%)  |  Test: {len(test)} (30%)",
        "",
        "## Question",
        "",
        "Does K=50 recency KNN add real discriminative power, or is the 0.94 dir_acc",
        "just majority-class alignment (predicting always-positive and test is 94% positive)?",
        "",
        "## Results",
        "",
        "| Config | DirAcc KNN | DirAcc Constant | KNN Gain | MAE | PredStd | PredRange | UniquePreds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| K=20 euclidean | {res20['dir_acc_knn']*100:.0f}% | {res20['dir_acc_constant_baseline']*100:.0f}% | {res20['knn_gain_over_baseline']*100:+.1f} pp | {res20['mae']} | {res20['pred_std']} | {res20['pred_range']} | {res20['n_unique_preds']}/{res20['n_test']} |",
        f"| K=50 recency | {res50['dir_acc_knn']*100:.0f}% | {res50['dir_acc_constant_baseline']*100:.0f}% | {res50['knn_gain_over_baseline']*100:+.1f} pp | {res50['mae']} | {res50['pred_std']} | {res50['pred_range']} | {res50['n_unique_preds']}/{res50['n_test']} |",
        "",
        "**Constant baseline**: always predict sign(train_median). If train_median > 0, predict positive.",
        "**KNN Gain**: dir_acc(KNN) - dir_acc(constant) — how much the model adds over naive prediction.",
        "",
        f"## Verdict: `{v}`",
        "",
        explain,
        "",
        f"## Recommendation: {payload['recommendation']}",
        "",
        "- BTC SELL tag remains `BASE_RATE_ONLY_PENDING_VARIANCE_CHECK` until clearly supported.",
        "- Review when test N grows (currently N_test=" + str(len(test)) + ").",
    ]

    md_path = OUT_DIR / "S34_BTC_SELL_KNN_VARIANCE_CHECK.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = OUT_DIR / "S34_BTC_SELL_KNN_VARIANCE_CHECK.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMD  : {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
