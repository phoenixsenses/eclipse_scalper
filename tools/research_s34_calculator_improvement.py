"""
S34 Calculator Improvement Research
Sections 1-8: architecture audit, feature ablation, K sweep, distance metrics,
calibration, regime drift, and v2 design proposal.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"

TRAIN_RATIO = 0.70
CALIB_STRONG_BPS = 15.0
MIN_TEST_N_NONPRELIM = 30

# (symbol, liq_side, min_notional_filter, primary_route)
COMBOS: list[tuple[str, str, float, str]] = [
    ("BTCUSDT", "BUY",  1_000_000.0, "LONG_DELAY0_TP60"),
    ("BTCUSDT", "SELL", 1_000_000.0, "SHORT_DELAY0_TP40"),
    ("ETHUSDT", "BUY",    200_000.0, "LONG_DELAY0_TP60"),
    ("ETHUSDT", "SELL",   500_000.0, "SHORT_DELAY0_TP60"),
    ("SOLUSDT", "BUY",    100_000.0, "LONG_DELAY0_TP60"),
    ("SOLUSDT", "SELL",   100_000.0, "SHORT_DELAY0_TP60"),
]

FEATURE_COLS: list[str] = [
    "cluster_notional",     # log-transformed in distance space
    "cluster_duration_sec",
    "cluster_liq_count",
    "max_single_liq_share",
    "intensity_per_sec",
    "inter_cluster_gap_sec",
    "day_trend_bps",
    "day_range_bps",
    "symbol_pre_5m_bps",
    "symbol_pre_15m_bps",
    "btc_pre_15m_bps",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "cluster_notional": 2.0,
    "cluster_duration_sec": 0.8,
    "cluster_liq_count": 0.8,
    "max_single_liq_share": 0.8,
    "intensity_per_sec": 1.0,
    "inter_cluster_gap_sec": 0.7,
    "day_trend_bps": 1.4,
    "day_range_bps": 1.0,
    "symbol_pre_5m_bps": 1.0,
    "symbol_pre_15m_bps": 1.0,
    "btc_pre_15m_bps": 0.8,
}

K_SWEEP: list[int] = [5, 10, 15, 20, 30, 50]
DEFAULT_K: int = 20
METRICS: list[str] = ["euclidean", "manhattan", "cosine", "recency"]
RECENCY_HALFLIFE_DAYS: float = 90.0  # older neighbors penalised: distance doubles at this age

CURRENT_TAGS: dict[str, str] = {
    "ETHUSDT_SELL": "KNN_USEFUL",
    "BTCUSDT_SELL": "BASE_RATE_ONLY",
    "ETHUSDT_BUY":  "REGIME_SHIFT_WARNING",
    "SOLUSDT_BUY":  "DRIFT_ARTIFACT_PRELIMINARY",
    "SOLUSDT_SELL": "DRIFT_ARTIFACT_PRELIMINARY",
    "BTCUSDT_BUY":  "DRIFT_ARTIFACT_PRELIMINARY",
}


# ─── math helpers ─────────────────────────────────────────────────────────────

def pctile(vals: list[float], q: float) -> float | None:
    clean = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not clean:
        return None
    pos = (len(clean) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return clean[lo] if lo == hi else clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def median(vals: list[float]) -> float | None:
    return pctile(vals, 0.5)


def _mean(vals: list[float]) -> float | None:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(clean) / len(clean) if clean else None


def _stdev(vals: list[float]) -> float | None:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def iqr_scale(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    if len(clean) < 3:
        return 1.0
    p75, p25 = pctile(clean, 0.75), pctile(clean, 0.25)
    iqr = abs((p75 or 0.0) - (p25 or 0.0))
    if iqr > 1e-9:
        return iqr
    sd = _stdev(clean)
    return sd if sd and sd > 1e-9 else 1.0


def dir_acc_fn(preds: list[float], realized: list[float]) -> float | None:
    pairs = [(p, r) for p, r in zip(preds, realized)]
    if not pairs:
        return None
    return sum(1 for p, r in pairs if (p > 0) == (r > 0)) / len(pairs)


def mae_fn(preds: list[float], realized: list[float]) -> float | None:
    vals = [abs(p - r) for p, r in zip(preds, realized)]
    return _mean(vals) if vals else None


def transform_feature(col: str, v: float) -> float:
    return math.log(max(v, 1.0)) if col == "cluster_notional" else v


# ─── data loading ─────────────────────────────────────────────────────────────

def load_combo(symbol: str, liq_side: str, min_notional: float) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{FEATURE_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    feat_rows = con.execute(
        "SELECT * FROM liq_event_features WHERE symbol=? AND liq_side=? AND cluster_notional>=?"
        " ORDER BY event_ts_ms ASC",
        (symbol, liq_side, min_notional),
    ).fetchall()
    events = [dict(r) for r in feat_rows]
    if not events:
        con.close()
        return []
    eids = [str(e["event_id"]) for e in events]
    ph = ",".join("?" * len(eids))
    out_rows = con.execute(
        f"SELECT event_id, route_id, net_bps FROM liq_event_outcome_labels WHERE event_id IN ({ph})",
        eids,
    ).fetchall()
    con.close()
    outcomes_map: dict[str, dict[str, float]] = {}
    for row in out_rows:
        eid = str(row[0])
        outcomes_map.setdefault(eid, {})[str(row[1])] = float(row[2])
    for e in events:
        e["_outcomes"] = outcomes_map.get(str(e["event_id"]), {})
    return events


def temporal_split(events: list[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    cut = max(1, min(len(events) - 1, int(len(events) * ratio)))
    return events[:cut], events[cut:]


# ─── feature vectors + scaling ────────────────────────────────────────────────

def extract_vec(event: dict, active_cols: list[str]) -> list[float | None]:
    vec: list[float | None] = []
    for col in active_cols:
        raw = event.get(col)
        if raw is None:
            vec.append(None)
        else:
            try:
                vec.append(transform_feature(col, float(raw)))
            except (TypeError, ValueError):
                vec.append(None)
    return vec


def build_scales(train_vecs: list[list[float | None]], n_cols: int) -> list[float]:
    scales = []
    for i in range(n_cols):
        col_vals = [v[i] for v in train_vecs if v[i] is not None]
        scales.append(iqr_scale(col_vals))
    return scales


# ─── distance ─────────────────────────────────────────────────────────────────

def compute_dist(
    target: list[float | None],
    neighbor: list[float | None],
    scales: list[float],
    weights: list[float],
    metric: str,
    age_days: float = 0.0,
) -> float | None:
    dims: list[tuple[float, float, float]] = []  # (weight, zt, zn)
    for i, (t, n) in enumerate(zip(target, neighbor)):
        if t is None or n is None:
            continue
        s = scales[i] if i < len(scales) and scales[i] > 1e-9 else 1.0
        w = weights[i] if i < len(weights) else 1.0
        dims.append((w, t / s, n / s))
    if not dims:
        return None
    total_w = sum(w for w, _, _ in dims)
    if total_w <= 0:
        return None
    if metric == "euclidean":
        return math.sqrt(sum(w * (zn - zt) ** 2 for w, zt, zn in dims) / total_w)
    elif metric == "manhattan":
        return sum(w * abs(zn - zt) for w, zt, zn in dims) / total_w
    elif metric == "cosine":
        dot = sum(w * zt * zn for w, zt, zn in dims)
        mag_t = math.sqrt(sum(w * zt ** 2 for w, zt, _ in dims))
        mag_n = math.sqrt(sum(w * zn ** 2 for w, _, zn in dims))
        if mag_t < 1e-9 or mag_n < 1e-9:
            return None
        return 1.0 - max(-1.0, min(1.0, dot / (mag_t * mag_n)))
    elif metric == "recency":
        feat_d = math.sqrt(sum(w * (zn - zt) ** 2 for w, zt, zn in dims) / total_w)
        # ln(2)/halflife → distance doubles when neighbor is halflife old
        recency_factor = math.exp(0.693 * age_days / RECENCY_HALFLIFE_DAYS)
        return feat_d * recency_factor
    return None


# ─── core KNN evaluation (per-event, train-pool only) ─────────────────────────

def run_eval(
    train: list[dict],
    test: list[dict],
    primary_route: str,
    k: int = DEFAULT_K,
    drop_feature: str | None = None,
    metric: str = "euclidean",
) -> dict[str, Any]:
    active_cols = [c for c in FEATURE_COLS if c != drop_feature] if drop_feature else FEATURE_COLS[:]
    weights = [DEFAULT_WEIGHTS.get(c, 1.0) for c in active_cols]

    train_vecs = [extract_vec(e, active_cols) for e in train]
    test_vecs  = [extract_vec(e, active_cols) for e in test]
    scales = build_scales(train_vecs, len(active_cols))

    train_outcomes = [e["_outcomes"].get(primary_route) for e in train]
    train_ts = [int(e.get("event_ts_ms") or 0) for e in train]

    preds: list[float] = []
    realized: list[float] = []

    for ev, tvec in zip(test, test_vecs):
        real = ev["_outcomes"].get(primary_route)
        if real is None:
            continue
        target_ts = int(ev.get("event_ts_ms") or 0)
        scored: list[tuple[float, float]] = []
        for j, (nvec, outcome) in enumerate(zip(train_vecs, train_outcomes)):
            if outcome is None:
                continue
            age_days = (target_ts - train_ts[j]) / (1000 * 86400.0) if metric == "recency" else 0.0
            d = compute_dist(tvec, nvec, scales, weights, metric, age_days=age_days)
            if d is not None:
                scored.append((d, float(outcome)))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        nbr = [o for _, o in scored[: min(k, len(scored))]]
        pred = median(nbr)
        if pred is not None:
            preds.append(pred)
            realized.append(float(real))

    train_real = [e["_outcomes"].get(primary_route) for e in train
                  if e["_outcomes"].get(primary_route) is not None]
    base = median(train_real)
    real_med = median(realized)
    uplift = round(float(real_med) - float(base), 1) if base is not None and real_med is not None else None

    return {
        "n_test":          len(preds),
        "dir_acc":         round(dir_acc_fn(preds, realized) or 0.0, 3),
        "mae":             round(mae_fn(preds, realized) or 0.0, 1),
        "pred_median":     round(float(median(preds)), 1) if preds else None,
        "realized_median": round(float(real_med), 1) if real_med is not None else None,
        "base_rate":       round(float(base), 1) if base is not None else None,
        "uplift":          uplift,
        "preds":           preds,
        "realized":        realized,
    }


# ─── Section 2: Feature ablation ──────────────────────────────────────────────

def run_ablation(train: list[dict], test: list[dict], primary_route: str) -> tuple[dict, list[dict]]:
    baseline = run_eval(train, test, primary_route, k=DEFAULT_K)
    rows = []
    for feat in FEATURE_COLS:
        res = run_eval(train, test, primary_route, k=DEFAULT_K, drop_feature=feat)
        dda = round((res["dir_acc"] or 0.0) - (baseline["dir_acc"] or 0.0), 3)
        dmae = round((res["mae"] or 0.0) - (baseline["mae"] or 0.0), 1)
        rows.append({
            "feature":        feat,
            "dir_acc_without": res["dir_acc"],
            "delta_dir_acc":  dda,   # negative = feature was useful
            "mae_without":    res["mae"],
            "delta_mae":      dmae,  # positive = feature was useful (dropping raised MAE)
            "verdict":        _ablation_verdict(dda, dmae),
        })
    # Sort: most negative delta_dir_acc first (most important)
    rows.sort(key=lambda r: r["delta_dir_acc"])
    return baseline, rows


def _ablation_verdict(delta_dir: float, delta_mae: float) -> str:
    if delta_dir < -0.03 or delta_mae > 3.0:
        return "useful"
    if delta_dir > 0.03 or delta_mae < -3.0:
        return "noise"
    return "marginal"


# ─── Section 3: K sweep ───────────────────────────────────────────────────────

def run_k_sweep(train: list[dict], test: list[dict], primary_route: str) -> list[dict]:
    rows = []
    for k in K_SWEEP:
        res = run_eval(train, test, primary_route, k=k)
        rows.append({"k": k, **{kk: vv for kk, vv in res.items() if kk not in ("preds", "realized")}})
    return rows


def _best_k(k_rows: list[dict]) -> int:
    if not k_rows:
        return DEFAULT_K
    return max(k_rows, key=lambda r: float(r["dir_acc"] or 0.0))["k"]


# ─── Section 4: Distance metric comparison ────────────────────────────────────

def run_metric_sweep(train: list[dict], test: list[dict], primary_route: str) -> list[dict]:
    rows = []
    for metric in METRICS:
        res = run_eval(train, test, primary_route, k=DEFAULT_K, metric=metric)
        rows.append({"metric": metric, **{kk: vv for kk, vv in res.items() if kk not in ("preds", "realized")}})
    return rows


def _best_metric(m_rows: list[dict]) -> str:
    if not m_rows:
        return "euclidean"
    return max(m_rows, key=lambda r: float(r["dir_acc"] or 0.0))["metric"]


# ─── Section 5: Calibration ───────────────────────────────────────────────────

def run_calibration(train: list[dict], test: list[dict], primary_route: str) -> dict[str, Any]:
    res = run_eval(train, test, primary_route, k=DEFAULT_K)
    preds, realized = res["preds"], res["realized"]
    buckets: dict[str, tuple[list[float], list[float]]] = {
        "strong_pos": ([], []),
        "neutral":    ([], []),
        "neg":        ([], []),
    }
    for p, r in zip(preds, realized):
        if p > CALIB_STRONG_BPS:
            key = "strong_pos"
        elif p < -CALIB_STRONG_BPS:
            key = "neg"
        else:
            key = "neutral"
        buckets[key][0].append(p)
        buckets[key][1].append(r)

    out: dict[str, Any] = {}
    for label, (bp, br) in buckets.items():
        pm = median(bp)
        rm = median(br)
        wr = round(sum(1 for v in br if v > 0) / len(br), 3) if br else None
        calibrated = _calib_verdict(pm, rm)
        out[label] = {
            "n":                len(bp),
            "pred_median":      round(float(pm), 1) if pm is not None else None,
            "realized_median":  round(float(rm), 1) if rm is not None else None,
            "win_rate":         wr,
            "calibrated":       calibrated,
        }
    return out


def _calib_verdict(pred_med: float | None, real_med: float | None) -> str:
    if pred_med is None or real_med is None:
        return "no_data"
    if pred_med > CALIB_STRONG_BPS:
        return "yes" if real_med > 0 else "no"
    if pred_med < -CALIB_STRONG_BPS:
        return "yes" if real_med < 0 else "no"
    return "neutral"


# ─── Section 6: Regime drift ──────────────────────────────────────────────────

def run_drift(train: list[dict], test: list[dict], primary_route: str) -> dict[str, Any]:
    feature_drifts = []
    for col in FEATURE_COLS:
        tr_vals = [transform_feature(col, float(e[col])) for e in train if e.get(col) is not None]
        te_vals = [transform_feature(col, float(e[col])) for e in test  if e.get(col) is not None]
        if not tr_vals or not te_vals:
            feature_drifts.append({"feature": col, "drift_score": None, "verdict": "missing"})
            continue
        tr_m = _mean(tr_vals) or 0.0
        tr_s = _stdev(tr_vals) or 1.0
        te_m = _mean(te_vals) or 0.0
        drift = round(abs((te_m - tr_m) / tr_s), 3)
        feature_drifts.append({
            "feature":    col,
            "train_mean": round(tr_m, 3),
            "test_mean":  round(te_m, 3),
            "train_std":  round(tr_s, 3),
            "drift_score": drift,
            "verdict":    _drift_verdict(drift),
        })

    tr_out = [e["_outcomes"].get(primary_route) for e in train if e["_outcomes"].get(primary_route) is not None]
    te_out = [e["_outcomes"].get(primary_route) for e in test  if e["_outcomes"].get(primary_route) is not None]
    tr_om = _mean(tr_out) or 0.0
    tr_os = _stdev(tr_out) or 1.0
    te_om = _mean(te_out) or 0.0
    out_drift = round(abs((te_om - tr_om) / tr_os), 3) if tr_os else None

    max_feat_drift = max((r["drift_score"] or 0.0) for r in feature_drifts)
    n_drifted = sum(1 for r in feature_drifts if (r["drift_score"] or 0.0) > 0.5)

    return {
        "feature_drifts":      feature_drifts,
        "outcome_drift": {
            "train_median":  round(float(median(tr_out)), 1) if tr_out else None,
            "test_median":   round(float(median(te_out)), 1) if te_out else None,
            "drift_score":   out_drift,
            "verdict":       _drift_verdict(out_drift),
        },
        "max_feature_drift":   round(max_feat_drift, 3),
        "n_drifted_features":  n_drifted,
        "overall_drift_level": _drift_verdict(max_feat_drift),
    }


def _drift_verdict(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score > 1.5:
        return "high_drift"
    if score > 0.7:
        return "moderate_drift"
    if score > 0.3:
        return "low_drift"
    return "stable"


# ─── Section 7: V2 design proposal ───────────────────────────────────────────

def generate_v2_proposal(combo_results: dict[str, Any]) -> dict[str, Any]:
    proposals: dict[str, Any] = {}
    summary_changes: list[str] = []

    for key, cr in combo_results.items():
        symbol, liq_side = key.split("|")
        baseline = cr["baseline"]
        da = float(baseline.get("dir_acc") or 0.0)
        n_test = baseline.get("n_test") or 0
        preliminary = n_test < MIN_TEST_N_NONPRELIM

        best_k = _best_k(cr["k_sweep"])
        best_metric = _best_metric(cr["metric_sweep"])
        best_k_da = max((r["dir_acc"] or 0.0) for r in cr["k_sweep"]) if cr["k_sweep"] else da
        best_metric_da = max((r["dir_acc"] or 0.0) for r in cr["metric_sweep"]) if cr["metric_sweep"] else da

        noise_feats = [r["feature"] for r in cr["ablation_rows"] if r["verdict"] == "noise"]
        useful_feats = [r["feature"] for r in cr["ablation_rows"] if r["verdict"] == "useful"]

        max_drift = cr["drift"].get("max_feature_drift") or 0.0
        out_drift = (cr["drift"].get("outcome_drift") or {}).get("drift_score") or 0.0
        n_drifted = cr["drift"].get("n_drifted_features") or 0

        calib_sp = cr["calibration"].get("strong_pos") or {}
        calib_calibrated = calib_sp.get("calibrated") == "yes"
        calib_sp_n = calib_sp.get("n") or 0

        # Determine proposed tag
        if preliminary:
            proposed_tag = "DRIFT_ARTIFACT_PRELIMINARY"
            proposed_k = min(best_k, 10)
            fallback = "mark_preliminary, show base-rate, hold until N_test >= 30"
        elif out_drift > 1.5 or max_drift > 1.5:
            proposed_tag = "REGIME_SHIFT_WARNING"
            proposed_k = best_k
            fallback = "warn_regime_drift, show train base-rate alongside prediction"
        elif best_k_da >= 0.68 and max_drift < 0.7:
            proposed_tag = "KNN_USEFUL"
            proposed_k = best_k
            fallback = "predict_with_optimised_K"
        elif best_k_da >= 0.58 and calib_calibrated and calib_sp_n >= 5:
            proposed_tag = "KNN_USEFUL"
            proposed_k = best_k
            fallback = "predict_with_strong-bucket_caveat"
        elif max_drift > 0.7:
            proposed_tag = "REGIME_SHIFT_WARNING"
            proposed_k = best_k
            fallback = "show_prediction_with_drift_warning"
        else:
            proposed_tag = "BASE_RATE_ONLY"
            proposed_k = max(K_SWEEP)  # large K → approaches base-rate
            fallback = "show_base-rate_median_only"

        current_tag = CURRENT_TAGS.get(f"{symbol}_{liq_side}", "unknown")
        tag_changed = proposed_tag != current_tag

        confidence_labels = ["broad", "usable", "thin", "too_thin"]
        if max_drift > 0.7:
            confidence_labels = ["drifted"] + confidence_labels
        if out_drift > 1.5:
            confidence_labels = ["regime_shift"] + confidence_labels

        proposals[key] = {
            "symbol":           symbol,
            "liq_side":         liq_side,
            "current_tag":      current_tag,
            "proposed_tag":     proposed_tag,
            "tag_changed":      tag_changed,
            "proposed_k":       proposed_k,
            "best_metric":      best_metric,
            "best_metric_beats_euclidean": round(best_metric_da - da, 3),
            "noise_features":   noise_feats,
            "useful_features":  useful_feats,
            "fallback_behavior": fallback,
            "confidence_labels": confidence_labels,
            "no_target_mode":   "population_scan_only — must label as such, not a prediction",
            "calibration_strong_pos": {
                "n": calib_sp_n,
                "realized_median": calib_sp.get("realized_median"),
                "calibrated": calib_sp.get("calibrated"),
            },
            "drift_summary": {
                "max_feature_drift": round(max_drift, 3),
                "outcome_drift":     round(out_drift, 3),
                "n_drifted_features": n_drifted,
            },
            "notes": _v2_notes(symbol, liq_side, proposed_tag, baseline, cr),
        }

        if tag_changed:
            summary_changes.append(
                f"  {symbol} {liq_side}: {current_tag} -> {proposed_tag} (K: {DEFAULT_K} -> {proposed_k})"
            )

    return {"proposals": proposals, "tag_changes": summary_changes}


def _v2_notes(sym: str, side: str, tag: str, baseline: dict, cr: dict) -> list[str]:
    notes: list[str] = []
    k_rows = cr["k_sweep"]
    if k_rows:
        best_k = _best_k(k_rows)
        if best_k != DEFAULT_K:
            best_da = max(r["dir_acc"] or 0 for r in k_rows)
            notes.append(f"K={best_k} outperforms default K={DEFAULT_K} (dir_acc {baseline['dir_acc']:.2f} -> {best_da:.2f})")
    m_rows = cr["metric_sweep"]
    if m_rows:
        best_m = _best_metric(m_rows)
        if best_m != "euclidean":
            eucl_da = next((r["dir_acc"] for r in m_rows if r["metric"] == "euclidean"), None)
            best_da = max(r["dir_acc"] or 0 for r in m_rows)
            notes.append(f"{best_m} metric outperforms euclidean ({eucl_da:.2f} -> {best_da:.2f})")
    abl_rows = cr["ablation_rows"]
    noise = [r["feature"] for r in abl_rows if r["verdict"] == "noise"]
    if noise:
        notes.append(f"Noise features (dropping helps): {', '.join(noise)}")
    useful = [r["feature"][:18] for r in abl_rows if r["verdict"] == "useful"]
    if useful:
        notes.append(f"Useful features (top by dir_acc delta): {', '.join(useful[:4])}")
    drift = cr["drift"]
    high_drift_feats = [r["feature"] for r in drift.get("feature_drifts", []) if (r.get("drift_score") or 0) > 1.0]
    if high_drift_feats:
        notes.append(f"High-drift features (>1.0 sigma shift train→test): {', '.join(high_drift_feats[:4])}")
    return notes


# ─── architecture audit (static, code-derived) ────────────────────────────────

ARCH_AUDIT: dict[str, Any] = {
    "features": {
        "count": 11,
        "list": FEATURE_COLS,
        "transforms": {"cluster_notional": "log(max(v, 1))"},
        "all_signal_time": True,
        "no_lookahead_at_signal_time": True,
        "note": "All features come from liq_event_features, populated at cluster-formation time from mark_prices history. No forward-looking fields.",
    },
    "normalization": {
        "method": "IQR (robust_scale) — p75-p25",
        "scope": "computed per-call on filtered event universe (symbol+liq_side+notional_threshold)",
        "not_global": True,
        "note": "Scales are recomputed each time generate() is called. In live mode the full historical DB is the candidate pool.",
    },
    "k_selection": {
        "default_k": 50,
        "note": "CLI default is 50 — high for combos with N=100-130. Validation script used adaptive min(20, train_n//5). No per-combo K tuning exists.",
        "adaptive_k_in_validation_script": True,
    },
    "missing_features": {
        "handling": "silently skipped per-dimension",
        "weight_normalization": "total_weight sum adjusts automatically",
        "risk": "if high-weight features (notional, day_trend) are missing, distance is computed on weaker features silently",
    },
    "temporal_no_lookahead": {
        "live_mode": True,
        "live_note": "In live mode the target is the current event; all DB events are historical. No leakage.",
        "oos_eval_in_calculator": False,
        "oos_eval_note": "run_oos_validation() calls knn_select(test_pool, args) where test_pool contains future events relative to some test events. This is NOT proper per-event temporal evaluation — it selects the K test events closest to the CLI query target, not K train events for each test event. The research scripts implement correct per-event evaluation.",
        "research_scripts_correct": True,
    },
    "target_less_mode": {
        "behavior": "returns first K events in temporal order",
        "metadata_flag": "error: 'no target features supplied; returned first K events from filtered universe'",
        "display_risk": "dashboard would show the first K historical events as 'predictions' with no signal. Must be explicitly labeled as population_scan_only.",
        "no_warning_in_output_json": True,
    },
    "default_min_notional": {
        "value": 200_000,
        "note": "Too permissive for BTC (threshold should be 1M) and SOL (100K). Running calculator for BTC without --min-notional 1000000 pulls all ETH-threshold events. ETH SELL DB was populated at 500K so 200K default passes all events.",
    },
    "oos_train_frac_default": {
        "value": 0.5,
        "note": "50/50 split vs 70/30 used in research scripts. With N=113-127 for BTC, 50/50 gives test_n=57-64 but train_n same size, limiting KNN quality.",
    },
}


# ─── markdown/JSON output ─────────────────────────────────────────────────────

def _fmt(v: Any, digits: int = 1, sign: bool = True) -> str:
    if v is None:
        return "NA"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    return fmt.format(fv)


def write_report(payload: dict[str, Any], md_path: Path, json_path: Path) -> None:
    lines: list[str] = [
        "# S34 Calculator Improvement Research",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "---",
        "",
        "## Section 1: Architecture Audit",
        "",
        "### Features",
        f"- Count: {ARCH_AUDIT['features']['count']}  ({', '.join(ARCH_AUDIT['features']['list'])})",
        "- `cluster_notional` is log-transformed before distance computation.",
        "- All features are signal-time (populated at cluster-formation from mark_prices history). No forward-looking fields.",
        "",
        "### Normalization",
        "- **IQR-based robust scale** (p75-p25) per feature.",
        "- Scope: computed per-call on the filtered event universe (symbol+liq_side+notional filter). Not global.",
        "",
        "### K Selection",
        "- CLI default K = **50**. Very large relative to small combos (BTC N=127, SOL N=104).",
        "- No per-combo K tuning. Validation scripts used adaptive `min(20, train_n//5)`.",
        "",
        "### Missing Features",
        "- Missing dimensions are silently skipped; `total_weight` normalises automatically.",
        "- Risk: if high-weight features (notional, day_trend) are missing, distance falls back to weaker features without warning.",
        "",
        "### Temporal No-Lookahead",
        "- **Live mode**: correct. Target = current event; all DB events are historical.",
        "- **`run_oos_validation()` in calculator**: NOT per-event evaluation. It calls `knn_select(test_pool, cli_target_args)` — selects K test-pool events closest to the CLI target query, not K train events per test event. Research scripts implement correct per-event train-pool prediction.",
        "",
        "### Target-Less Mode",
        "- When no `--target-*` args given: returns **first K events in temporal order**.",
        "- Metadata flags `error: 'no target features supplied'` — but this is not surfaced in dashboard output.",
        "- **Display risk**: dashboard would show the first K historical events as 'predictions' with no signal. Must be labelled `population_scan_only`.",
        "",
        "### Default min-notional = 200K",
        "- Correct for ETH BUY/SELL (DB populated at 200K/500K — all pass).",
        "- Too permissive for BTC (should be 1M) and SOL (should be 100K).",
        "- Running BTC calculator without `--min-notional 1000000` mixes ETH-threshold events into the candidate pool.",
        "",
        "---",
        "",
        "## Section 2: Feature Ablation",
        "",
        "> delta_dir_acc < 0 → dropping feature hurt accuracy (feature is **useful**)",
        "> delta_dir_acc > 0 → dropping feature improved accuracy (feature is **noise**)",
        "",
    ]

    for key, cr in payload["combo_results"].items():
        symbol, liq_side = key.split("|")
        bl = cr["baseline"]
        prelim = " *" if bl["n_test"] < MIN_TEST_N_NONPRELIM else ""
        lines.append(f"### {symbol} {liq_side}{prelim}  (baseline: dir_acc={_fmt(bl['dir_acc']*100, 0, False)}%  MAE={_fmt(bl['mae'], 1, False)}  N_test={bl['n_test']})")
        lines.append("")
        lines.append("| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in cr["ablation_rows"]:
            lines.append(
                f"| `{row['feature']}` | {_fmt(row['dir_acc_without']*100, 0, False)}% "
                f"| {_fmt(row['delta_dir_acc']*100, 1)} pp "
                f"| {_fmt(row['mae_without'], 1, False)} "
                f"| {_fmt(row['delta_mae'], 1)} "
                f"| {row['verdict']} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Section 3: K Selection Sweep",
        "",
    ]
    for key, cr in payload["combo_results"].items():
        symbol, liq_side = key.split("|")
        best_k = _best_k(cr["k_sweep"])
        lines.append(f"### {symbol} {liq_side}  (best K = {best_k})")
        lines.append("")
        lines.append("| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in cr["k_sweep"]:
            marker = " **" if row["k"] == best_k else ""
            lines.append(
                f"| {row['k']}{marker} | {row['n_test']} "
                f"| {_fmt(row['dir_acc']*100, 0, False)}% "
                f"| {_fmt(row['mae'], 1, False)} "
                f"| {_fmt(row['pred_median'])} "
                f"| {_fmt(row['realized_median'])} "
                f"| {_fmt(row['uplift'])} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Section 4: Distance Metric Comparison",
        "",
        f"All at K={DEFAULT_K}.",
        "",
    ]
    for key, cr in payload["combo_results"].items():
        symbol, liq_side = key.split("|")
        best_m = _best_metric(cr["metric_sweep"])
        lines.append(f"### {symbol} {liq_side}  (best metric = {best_m})")
        lines.append("")
        lines.append("| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in cr["metric_sweep"]:
            marker = " **" if row["metric"] == best_m else ""
            lines.append(
                f"| {row['metric']}{marker} | {row['n_test']} "
                f"| {_fmt(row['dir_acc']*100, 0, False)}% "
                f"| {_fmt(row['mae'], 1, False)} "
                f"| {_fmt(row['pred_median'])} "
                f"| {_fmt(row['realized_median'])} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        f"## Section 5: Calibration  (threshold = ±{CALIB_STRONG_BPS:.0f} bps)",
        "",
        "Does 'predicted strong positive' actually realize a positive outcome?",
        "",
        "| Combo | Bucket | N | PredMedian | RealMedian | WinRate | Calibrated |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for key, cr in payload["combo_results"].items():
        symbol, liq_side = key.split("|")
        for bucket_label in ("strong_pos", "neutral", "neg"):
            b = cr["calibration"].get(bucket_label) or {}
            lines.append(
                f"| {symbol} {liq_side} | {bucket_label} | {b.get('n', 0)} "
                f"| {_fmt(b.get('pred_median'))} "
                f"| {_fmt(b.get('realized_median'))} "
                f"| {_fmt(b.get('win_rate', 0) * 100 if b.get('win_rate') is not None else None, 0, False)}% "
                f"| {b.get('calibrated', 'NA')} |"
            )
    lines.append("")

    lines += [
        "---",
        "",
        "## Section 6: Regime Drift  (train vs test feature distributions)",
        "",
        "> drift_score = |test_mean - train_mean| / train_std",
        "> high_drift > 1.5 | moderate_drift 0.7-1.5 | low_drift 0.3-0.7 | stable < 0.3",
        "",
    ]
    for key, cr in payload["combo_results"].items():
        symbol, liq_side = key.split("|")
        drift = cr["drift"]
        out_d = drift.get("outcome_drift") or {}
        lines.append(
            f"### {symbol} {liq_side}  "
            f"(max_feat_drift={drift.get('max_feature_drift'):.3f}  "
            f"outcome_drift={out_d.get('drift_score', '?')}  "
            f"n_drifted={drift.get('n_drifted_features')})"
        )
        lines.append("")
        lines.append("| Feature | TrainMean | TestMean | DriftScore | Verdict |")
        lines.append("|---|---:|---:|---:|---|")
        for fd in drift.get("feature_drifts") or []:
            lines.append(
                f"| `{fd['feature']}` | {_fmt(fd.get('train_mean'), 2, False)} "
                f"| {_fmt(fd.get('test_mean'), 2, False)} "
                f"| {_fmt(fd.get('drift_score'), 3, False)} "
                f"| {fd.get('verdict', 'NA')} |"
            )
        lines.append(
            f"\n**Outcome (primary route)**: train_median={_fmt(out_d.get('train_median'))}  "
            f"test_median={_fmt(out_d.get('test_median'))}  "
            f"drift={_fmt(out_d.get('drift_score'), 3, False)}  "
            f"verdict={out_d.get('verdict', 'NA')}"
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## Section 7: Proposed Calculator v2 Design",
        "",
    ]
    v2 = payload.get("v2_proposal") or {}
    changes = v2.get("tag_changes") or []
    if changes:
        lines.append("**Tag changes from current:**")
        lines += [f"  {c}" for c in changes]
    else:
        lines.append("All current tags confirmed by findings.")
    lines.append("")
    lines.append("| Combo | CurrentTag | ProposedTag | K | Metric | NoiseFeat | FallbackBehavior |")
    lines.append("|---|---|---|---:|---|---|---|")
    for key, p in (v2.get("proposals") or {}).items():
        symbol, liq_side = key.split("|")
        noise_str = " ".join(p.get("noise_features") or []) or "—"
        lines.append(
            f"| {symbol} {liq_side} | {p['current_tag']} | **{p['proposed_tag']}** "
            f"| {p['proposed_k']} | {p['best_metric']} | `{noise_str}` | {p['fallback_behavior']} |"
        )
    lines.append("")

    lines.append("### Per-Combo Detail")
    lines.append("")
    for key, p in (v2.get("proposals") or {}).items():
        symbol, liq_side = key.split("|")
        lines.append(f"**{symbol} {liq_side}** → `{p['proposed_tag']}`")
        for note in p.get("notes") or []:
            lines.append(f"- {note}")
        calib = p.get("calibration_strong_pos") or {}
        if calib.get("n"):
            lines.append(
                f"- Calibration strong_pos bucket: N={calib['n']}, realized_median={_fmt(calib.get('realized_median'))}, calibrated={calib.get('calibrated')}"
            )
        drift = p.get("drift_summary") or {}
        lines.append(
            f"- Drift: max_feature={drift.get('max_feature_drift')}  outcome={drift.get('outcome_drift')}  n_drifted_features={drift.get('n_drifted_features')}"
        )
        lines.append(f"- no_target_mode: {p['no_target_mode']}")
        lines.append(f"- confidence_labels: {' / '.join(p.get('confidence_labels') or [])}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Overfitting Risk Notes",
        "",
        "- All evaluation is temporal OOS (test = last 30%). Features are pre-computed, no lookahead.",
        "- K sweep and ablation use the same 70/30 split — selecting best K/metric on test data is mild overfitting. Treat as directional, not certified.",
        "- Combos with N_test < 30 are marked `*` and tagged DRIFT_ARTIFACT_PRELIMINARY regardless of metrics.",
        "- ETH BUY N_test=135 is the only combo with sufficient test size for confident conclusions.",
        "- Large 'uplift' values for BTC BUY / SOL combos reflect regime drift (test period stronger than train), not genuine KNN signal.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combo_results: dict[str, Any] = {}

    for symbol, liq_side, min_notional, primary_route in COMBOS:
        key = f"{symbol}|{liq_side}"
        print(f"\n=== {symbol} {liq_side} ===")
        events = load_combo(symbol, liq_side, min_notional)
        if len(events) < 10:
            print(f"  Skipping — only {len(events)} events")
            continue
        train, test = temporal_split(events, TRAIN_RATIO)
        print(f"  N={len(events)}  train={len(train)}  test={len(test)}")

        print("  ablation...", end=" ", flush=True)
        baseline, ablation_rows = run_ablation(train, test, primary_route)
        print(f"baseline dir_acc={baseline['dir_acc']:.2f}")

        print("  K sweep...", end=" ", flush=True)
        k_rows = run_k_sweep(train, test, primary_route)
        best_k = _best_k(k_rows)
        best_k_da = max(r["dir_acc"] or 0 for r in k_rows)
        print(f"best K={best_k} ({best_k_da:.2f})")

        print("  metrics...", end=" ", flush=True)
        m_rows = run_metric_sweep(train, test, primary_route)
        best_m = _best_metric(m_rows)
        best_m_da = max(r["dir_acc"] or 0 for r in m_rows)
        print(f"best metric={best_m} ({best_m_da:.2f})")

        print("  calibration + drift...", end=" ", flush=True)
        calib = run_calibration(train, test, primary_route)
        drift = run_drift(train, test, primary_route)
        print(f"max_drift={drift['max_feature_drift']:.3f}")

        combo_results[key] = {
            "symbol":       symbol,
            "liq_side":     liq_side,
            "n_events":     len(events),
            "n_train":      len(train),
            "n_test":       len(test),
            "primary_route": primary_route,
            "baseline":     {k: v for k, v in baseline.items() if k not in ("preds", "realized")},
            "ablation_rows": ablation_rows,
            "k_sweep":      k_rows,
            "metric_sweep": m_rows,
            "calibration":  calib,
            "drift":        drift,
        }

    v2 = generate_v2_proposal(combo_results)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arch_audit":       ARCH_AUDIT,
        "combo_results":    combo_results,
        "v2_proposal":      v2,
    }

    md_path   = OUT_DIR / "S34_CALCULATOR_IMPROVEMENT_RESEARCH.md"
    json_path = OUT_DIR / "S34_CALCULATOR_IMPROVEMENT_RESEARCH.json"
    write_report(payload, md_path, json_path)

    print("\n=== SUMMARY ===")
    print(f"Combos evaluated: {len(combo_results)}")
    for key, cr in combo_results.items():
        bl = cr["baseline"]
        best_k = _best_k(cr["k_sweep"])
        best_m = _best_metric(cr["metric_sweep"])
        best_k_da = max(r["dir_acc"] or 0 for r in cr["k_sweep"])
        prop = (v2.get("proposals") or {}).get(key) or {}
        print(
            f"  {key.replace('|', ' '):<16}  "
            f"baseline={bl['dir_acc']:.2f}  bestK({best_k})={best_k_da:.2f}  "
            f"best_metric={best_m}  drift={cr['drift']['max_feature_drift']:.2f}  "
            f"tag: {prop.get('current_tag','?')} -> {prop.get('proposed_tag','?')}"
        )

    v2_changes = v2.get("tag_changes") or []
    if v2_changes:
        print("\nProposed tag changes:")
        for c in v2_changes:
            print(c)
    else:
        print("\nNo tag changes proposed — current tags confirmed.")

    print(f"\nMD  : {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
