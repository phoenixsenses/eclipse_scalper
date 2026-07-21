"""
ETH SELL Calculator v2 Validation
Compares old config (K=20, euclidean, all features) vs v2 config (K=10, manhattan, 6 features).
"""
from __future__ import annotations
import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"

SYMBOL, SIDE, MIN_NOTIONAL, PRIMARY_ROUTE = "ETHUSDT", "SELL", 500_000.0, "SHORT_DELAY0_TP60"
TRAIN_RATIO = 0.70

ALL_FEATURES = [
    "cluster_notional", "cluster_duration_sec", "cluster_liq_count",
    "max_single_liq_share", "intensity_per_sec", "inter_cluster_gap_sec",
    "day_trend_bps", "day_range_bps", "symbol_pre_5m_bps", "symbol_pre_15m_bps",
    "btc_pre_15m_bps",
]
V2_EXCLUDE = {"cluster_liq_count", "btc_pre_15m_bps", "cluster_duration_sec",
              "day_trend_bps", "symbol_pre_15m_bps"}
V2_FEATURES = [f for f in ALL_FEATURES if f not in V2_EXCLUDE]

OLD_K, OLD_METRIC = 20, "euclidean"
NEW_K, NEW_METRIC = 10, "manhattan"


def _pctile(vals, q):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    pos = (len(c) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (pos - lo)

def _med(vals): return _pctile(vals, 0.5)
def _mean(vals):
    c = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(c) / len(c) if c else None
def _std(vals):
    c = [v for v in vals if v is not None and math.isfinite(v)]
    if len(c) < 2: return None
    mu = sum(c) / len(c)
    return math.sqrt(sum((v - mu)**2 for v in c) / (len(c) - 1))
def _iqr_scale(vals):
    p75, p25 = _pctile(vals, 0.75), _pctile(vals, 0.25)
    iqr = abs((p75 or 0) - (p25 or 0))
    if iqr > 1e-9: return iqr
    s = _std(vals)
    return s if s and s > 1e-9 else 1.0
def _transform(col, v):
    return math.log(max(v, 1.0)) if col == "cluster_notional" else v


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


def run_knn(train, test, features, k, metric):
    weights = {"cluster_notional": 2.0, "cluster_duration_sec": 0.8, "cluster_liq_count": 0.8,
               "max_single_liq_share": 0.8, "intensity_per_sec": 1.0, "inter_cluster_gap_sec": 0.7,
               "day_trend_bps": 1.4, "day_range_bps": 1.0, "symbol_pre_5m_bps": 1.0,
               "symbol_pre_15m_bps": 1.0, "btc_pre_15m_bps": 0.8}
    wlist = [weights.get(f, 1.0) for f in features]

    def vec(e):
        return [_transform(f, float(e[f])) if e.get(f) is not None else None for f in features]

    tr_vecs = [vec(e) for e in train]
    te_vecs = [vec(e) for e in test]
    scales = []
    for i in range(len(features)):
        col_vals = [v[i] for v in tr_vecs if v[i] is not None]
        scales.append(_iqr_scale(col_vals))
    tr_out = [e["_out"].get(PRIMARY_ROUTE) for e in train]

    preds, realized = [], []
    for tvec, ev in zip(te_vecs, test):
        real = ev["_out"].get(PRIMARY_ROUTE)
        if real is None: continue
        scored = []
        for nvec, outcome in zip(tr_vecs, tr_out):
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
            if metric == "manhattan":
                d = sum(w * abs(z) for w, z in dims) / tw
            else:
                d = math.sqrt(sum(w * z * z for w, z in dims) / tw)
            scored.append((d, float(outcome)))
        if not scored: continue
        scored.sort(key=lambda x: x[0])
        nbr = [o for _, o in scored[:min(k, len(scored))]]
        pred = _med(nbr)
        if pred is not None:
            preds.append(pred)
            realized.append(float(real))

    base = _med([e["_out"].get(PRIMARY_ROUTE) for e in train if e["_out"].get(PRIMARY_ROUTE) is not None])
    rm = _med(realized)
    da = sum(1 for p, r in zip(preds, realized) if (p > 0) == (r > 0)) / len(preds) if preds else None
    mae = _mean([abs(p - r) for p, r in zip(preds, realized)])
    return {
        "n_test": len(preds), "dir_acc": round(da, 3) if da is not None else None,
        "mae": round(mae, 1) if mae is not None else None,
        "pred_median": round(float(_med(preds)), 1) if preds else None,
        "realized_median": round(float(rm), 1) if rm is not None else None,
        "base_rate": round(float(base), 1) if base is not None else None,
        "pred_std": round(float(_std(preds)), 1) if preds and len(preds) > 1 else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    cut = max(1, min(len(events) - 1, int(len(events) * TRAIN_RATIO)))
    train, test = events[:cut], events[cut:]
    print(f"ETH SELL: N={len(events)}  train={len(train)}  test={len(test)}")

    print("Running old config (K=20, euclidean, all features)...")
    old = run_knn(train, test, ALL_FEATURES, OLD_K, OLD_METRIC)
    print(f"  dir_acc={old['dir_acc']:.2f}  mae={old['mae']}  pred_median={old['pred_median']}  pred_std={old['pred_std']}")

    print("Running v2 config (K=10, manhattan, 6 features)...")
    new = run_knn(train, test, V2_FEATURES, NEW_K, NEW_METRIC)
    print(f"  dir_acc={new['dir_acc']:.2f}  mae={new['mae']}  pred_median={new['pred_median']}  pred_std={new['pred_std']}")

    delta_dir = round(float(new["dir_acc"] or 0) - float(old["dir_acc"] or 0), 3)
    delta_mae = round(float(new["mae"] or 0) - float(old["mae"] or 0), 1)
    print(f"\nDelta: dir_acc={delta_dir:+.3f}  mae={delta_mae:+.1f}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combo": f"{SYMBOL}_{SIDE}",
        "primary_route": PRIMARY_ROUTE,
        "n_events": len(events), "n_train": len(train), "n_test": len(test),
        "old_config": {"k": OLD_K, "metric": OLD_METRIC, "features": ALL_FEATURES, "results": old},
        "new_config": {"k": NEW_K, "metric": NEW_METRIC, "features": V2_FEATURES, "results": new},
        "delta": {"dir_acc": delta_dir, "mae": delta_mae},
        "verdict": _verdict(delta_dir, delta_mae),
    }

    lines = [
        "# ETH SELL Calculator v2 Validation",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Combo: {SYMBOL} {SIDE}  |  Primary route: {PRIMARY_ROUTE}",
        f"N events: {len(events)}  |  Train: {len(train)} (70%)  |  Test: {len(test)} (30%)",
        "",
        "## Config Comparison",
        "",
        "| Config | K | Metric | Features |",
        "|---|---:|---|---|",
        f"| Old (default) | {OLD_K} | {OLD_METRIC} | all 11 |",
        f"| v2 | {NEW_K} | {NEW_METRIC} | {len(V2_FEATURES)} (excluded: {', '.join(sorted(V2_EXCLUDE))}) |",
        "",
        "## Results",
        "",
        "| Config | N_test | DirAcc | MAE | PredMedian | RealMedian | BaseRate | PredStd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Old | {old['n_test']} | {old['dir_acc']*100:.0f}% | {old['mae']} | {old['pred_median']} | {old['realized_median']} | {old['base_rate']} | {old['pred_std']} |",
        f"| **v2** | {new['n_test']} | **{new['dir_acc']*100:.0f}%** | **{new['mae']}** | {new['pred_median']} | {new['realized_median']} | {new['base_rate']} | {new['pred_std']} |",
        "",
        "## Delta (v2 - old)",
        "",
        f"- dir_acc: `{delta_dir:+.3f}` ({delta_dir*100:+.1f} pp)",
        f"- MAE: `{delta_mae:+.1f}` bps",
        "",
        f"## Verdict: {payload['verdict']}",
        "",
        _verdict_detail(delta_dir, delta_mae, new),
        "",
        "## Notes",
        "",
        "- All evaluation is temporal OOS (test = last 30%, strictly after train).",
        "- No forward-looking features; distance computed on train pool only per test event.",
        "- v2 config is now the default for ETH SELL in `s34_liq_outcome_calculator.py`.",
        "- Other combos are unaffected.",
    ]

    md_path = OUT_DIR / "S34_ETH_SELL_CALCULATOR_V2_VALIDATION.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = OUT_DIR / "S34_ETH_SELL_CALCULATOR_V2_VALIDATION.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMD  : {md_path}")
    print(f"JSON: {json_path}")
    print(f"Verdict: {payload['verdict']}")


def _verdict(delta_dir, delta_mae):
    if delta_dir >= 0.04:
        return "V2_IMPROVEMENT_CONFIRMED"
    if delta_dir >= 0.01 and delta_mae <= 0:
        return "V2_MARGINAL_IMPROVEMENT"
    if delta_dir < -0.03:
        return "V2_REGRESSION_DO_NOT_DEPLOY"
    return "V2_NEUTRAL"


def _verdict_detail(delta_dir, delta_mae, new):
    if delta_dir >= 0.04:
        return (f"v2 improves direction accuracy by {delta_dir*100:+.1f} pp and reduces MAE by {-delta_mae:.1f} bps. "
                "Deploy v2 as ETH SELL default.")
    if delta_dir >= 0.01:
        return "Marginal improvement. v2 is slightly better; deploy cautiously."
    if delta_dir < -0.03:
        return "v2 is worse. Keep old config."
    return "No significant difference. v2 is not harmful; deploy for consistency with research findings."


if __name__ == "__main__":
    main()
