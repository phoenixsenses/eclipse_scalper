"""S34 Phase 2 Calculator Validation — per-symbol/per-side predictive quality.

For each of the 6 combos (ETH/SOL/BTC × BUY/SELL), validates whether the KNN
similarity calculator adds predictive value over the base-rate (train-set median).

Temporal split: train = older 70%, test = newer 30%. No lookahead.
No runner, live rule, or DB changes.

Output:
  reports/research/s34/S34_PHASE2_CALCULATOR_VALIDATION.md
  reports/research/s34/S34_PHASE2_CALCULATOR_VALIDATION.json
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB_PATH = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_PHASE2_CALCULATOR_VALIDATION.json"
OUT_MD   = ROOT / "reports" / "research" / "s34" / "S34_PHASE2_CALCULATOR_VALIDATION.md"

TRAIN_RATIO = 0.70
PRELIM_THRESHOLD = 30   # test N below this → mark as preliminary
CALIB_STRONG_BPS = 15.0  # predicted > +15 → "positive", < -15 → "negative"

# KNN configuration
K_DEFAULT = 20
FEATURE_COLS: list[str] = [
    "cluster_notional",      # log-transformed
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
    "cluster_notional":      2.0,
    "cluster_duration_sec":  0.8,
    "cluster_liq_count":     0.8,
    "max_single_liq_share":  0.8,
    "intensity_per_sec":     1.0,
    "inter_cluster_gap_sec": 0.7,
    "day_trend_bps":         1.4,
    "day_range_bps":         1.0,
    "symbol_pre_5m_bps":     1.0,
    "symbol_pre_15m_bps":    1.0,
    "btc_pre_15m_bps":       0.8,
}

# Primary route per combo — highlighted in summary ranking
PRIMARY_ROUTE: dict[tuple[str, str], str] = {
    ("ETHUSDT", "BUY"):  "LONG_DELAY0_TP60",
    ("ETHUSDT", "SELL"): "SHORT_DELAY0_TP60",
    ("SOLUSDT", "BUY"):  "LONG_DELAY0_TP60",
    ("SOLUSDT", "SELL"): "SHORT_DELAY0_TP60",
    ("BTCUSDT", "BUY"):  "LONG_DELAY0_TP60",
    ("BTCUSDT", "SELL"): "SHORT_DELAY0_TP40",
}

COMBOS: list[tuple[str, str]] = [
    ("BTCUSDT", "BUY"),
    ("BTCUSDT", "SELL"),
    ("ETHUSDT", "BUY"),
    ("ETHUSDT", "SELL"),
    ("SOLUSDT", "BUY"),
    ("SOLUSDT", "SELL"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def median(vals: list[float]) -> float | None:
    v = sorted(v for v in vals if math.isfinite(v))
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def wr(vals: list[float]) -> float | None:
    v = [x for x in vals if math.isfinite(x)]
    return sum(1 for x in v if x > 0) / len(v) if v else None


def mae(pred: list[float], real: list[float]) -> float | None:
    pairs = [(p, r) for p, r in zip(pred, real) if math.isfinite(p) and math.isfinite(r)]
    if not pairs:
        return None
    return sum(abs(p - r) for p, r in pairs) / len(pairs)


def direction_acc(pred: list[float], real: list[float]) -> float | None:
    pairs = [(p, r) for p, r in zip(pred, real) if math.isfinite(p) and math.isfinite(r)]
    if not pairs:
        return None
    correct = sum(1 for p, r in pairs if (p >= 0) == (r >= 0))
    return correct / len(pairs)


def pctile(vals: list[float], q: float) -> float | None:
    v = sorted(x for x in vals if math.isfinite(x))
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (pos - lo)


def transform_feature(col: str, val: float) -> float:
    if col == "cluster_notional":
        return math.log(max(val, 1.0))
    return val


def robust_iqr(vals: list[float]) -> float:
    v = sorted(x for x in vals if math.isfinite(x))
    if len(v) < 3:
        return 1.0
    p75 = pctile(v, 0.75)
    p25 = pctile(v, 0.25)
    if p75 is None or p25 is None:
        return 1.0
    iqr = abs(p75 - p25)
    if iqr > 1e-9:
        return iqr
    # fallback: stddev
    mean_v = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean_v) ** 2 for x in v) / len(v))
    return sd if sd > 1e-9 else 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combo(fdb: sqlite3.Connection, symbol: str, liq_side: str) -> list[dict[str, Any]]:
    fdb.row_factory = sqlite3.Row
    rows = fdb.execute(
        """
        SELECT f.*, l.route_id, l.net_bps, l.exit_reason
        FROM liq_event_features f
        JOIN liq_event_outcome_labels l ON l.event_id = f.event_id
        WHERE f.symbol = ? AND f.liq_side = ?
        ORDER BY f.event_ts_ms ASC
        """,
        (symbol, liq_side),
    ).fetchall()
    # Group into event → {route_id: net_bps}
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        eid = row["event_id"]
        if eid not in events:
            d = dict(row)
            d.pop("route_id", None)
            d.pop("net_bps", None)
            d.pop("exit_reason", None)
            events[eid] = {**d, "_outcomes": {}}
        events[eid]["_outcomes"][row["route_id"]] = float(row["net_bps"])
    return sorted(events.values(), key=lambda e: int(e["event_ts_ms"]))


def temporal_split(events: list[dict], train_ratio: float = TRAIN_RATIO) -> tuple[list, list]:
    cut = max(1, int(len(events) * train_ratio))
    return events[:cut], events[cut:]


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def build_scales(train: list[dict], weights: dict[str, float]) -> dict[str, float]:
    scales = {}
    for col in FEATURE_COLS:
        if col not in weights:
            continue
        vals = [transform_feature(col, float(e[col])) for e in train
                if e.get(col) is not None and math.isfinite(float(e[col]))]
        scales[col] = robust_iqr(vals)
    return scales


def build_corr_weights(train: list[dict], route_id: str) -> dict[str, float]:
    """Correlation-based auto-weights: |corr(feature, outcome)| on train set."""
    outcomes = [e["_outcomes"].get(route_id) for e in train]
    weights = {}
    for col in FEATURE_COLS:
        xs = []
        ys = []
        for e, y in zip(train, outcomes):
            if y is None or e.get(col) is None:
                continue
            x = transform_feature(col, float(e[col]))
            if not math.isfinite(x) or not math.isfinite(float(y)):
                continue
            xs.append(x)
            ys.append(float(y))
        if len(xs) < 5:
            weights[col] = 0.1
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if sx < 1e-9 or sy < 1e-9:
            weights[col] = 0.1
        else:
            weights[col] = max(0.1, abs(cov / (sx * sy)))
    # Normalize so max weight = 2.0
    mx_w = max(weights.values()) if weights else 1.0
    return {k: v / mx_w * 2.0 for k, v in weights.items()}


def knn_dist(target: dict, neighbor: dict, scales: dict, weights: dict) -> float:
    total = 0.0
    total_w = 0.0
    for col in FEATURE_COLS:
        tv = target.get(col)
        nv = neighbor.get(col)
        if tv is None or nv is None:
            continue
        try:
            tf = transform_feature(col, float(tv))
            nf = transform_feature(col, float(nv))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(tf) or not math.isfinite(nf):
            continue
        scale = scales.get(col, 1.0)
        w = weights.get(col, 1.0)
        z = (tf - nf) / scale
        total += w * z * z
        total_w += w
    if total_w <= 0:
        return float("inf")
    return math.sqrt(total / total_w)


def knn_predict(
    target: dict,
    train: list[dict],
    scales: dict,
    weights: dict,
    route_id: str,
    k: int,
) -> float | None:
    scored = []
    for ev in train:
        outcome = ev["_outcomes"].get(route_id)
        if outcome is None:
            continue
        d = knn_dist(target, ev, scales, weights)
        if math.isfinite(d):
            scored.append((d, float(outcome)))
    scored.sort(key=lambda x: x[0])
    neighbors = [net for _, net in scored[:k]]
    return median(neighbors)


# ---------------------------------------------------------------------------
# Calibration bucketing
# ---------------------------------------------------------------------------

def calibration(
    predictions: list[float],
    realized: list[float],
    threshold: float = CALIB_STRONG_BPS,
) -> list[dict[str, Any]]:
    buckets = [
        ("positive",  lambda p: p >= threshold),
        ("neutral",   lambda p: -threshold < p < threshold),
        ("negative",  lambda p: p <= -threshold),
    ]
    out = []
    for label, pred_fn in buckets:
        idx = [i for i, p in enumerate(predictions) if math.isfinite(p) and pred_fn(p)]
        real_bucket = [realized[i] for i in idx if math.isfinite(realized[i])]
        out.append({
            "bucket": label,
            "n": len(real_bucket),
            "pred_median": median([predictions[i] for i in idx]),
            "realized_median": median(real_bucket),
            "realized_wr": wr(real_bucket),
        })
    return out


# ---------------------------------------------------------------------------
# Per-combo evaluation
# ---------------------------------------------------------------------------

def evaluate_combo(
    fdb: sqlite3.Connection,
    symbol: str,
    liq_side: str,
) -> dict[str, Any]:
    events = load_combo(fdb, symbol, liq_side)
    if not events:
        return {"symbol": symbol, "liq_side": liq_side, "error": "no events"}

    train, test = temporal_split(events)
    train_cut_utc = train[-1]["event_utc"][:10] if train else "-"
    test_start_utc = test[0]["event_utc"][:10] if test else "-"

    route_ids = sorted({r for e in events for r in e["_outcomes"]})
    primary = PRIMARY_ROUTE.get((symbol, liq_side))

    k = max(5, min(K_DEFAULT, len(train) // 5))

    route_results = []
    for route_id in route_ids:
        train_nets = [e["_outcomes"][route_id] for e in train if route_id in e["_outcomes"]]
        test_nets  = [e["_outcomes"][route_id] for e in test  if route_id in e["_outcomes"]]
        if not train_nets or not test_nets:
            continue
        base_med = median(train_nets)
        base_wr  = wr(train_nets)

        # --- default weights ---
        scales_def = build_scales(train, DEFAULT_WEIGHTS)
        preds_def = []
        for ev in test:
            if route_id not in ev["_outcomes"]:
                preds_def.append(float("nan"))
            else:
                p = knn_predict(ev, train, scales_def, DEFAULT_WEIGHTS, route_id, k)
                preds_def.append(p if p is not None else float("nan"))

        valid_pairs = [(p, r) for p, r in zip(preds_def, test_nets)
                       if math.isfinite(p) and math.isfinite(r)]
        preds_v = [p for p, _ in valid_pairs]
        real_v  = [r for _, r in valid_pairs]

        # --- auto-weights (correlation-based) ---
        auto_w = build_corr_weights(train, route_id)
        scales_auto = build_scales(train, auto_w)
        preds_auto = []
        for ev in test:
            if route_id not in ev["_outcomes"]:
                preds_auto.append(float("nan"))
            else:
                p = knn_predict(ev, train, scales_auto, auto_w, route_id, k)
                preds_auto.append(p if p is not None else float("nan"))
        preds_auto_v = [p for p, r in zip(preds_auto, test_nets)
                        if math.isfinite(p) and math.isfinite(r)]

        realized_med    = median(real_v)
        realized_wr_val = wr(real_v)
        pred_med        = median(preds_v)
        pred_auto_med   = median(preds_auto_v)
        mae_def         = mae(preds_v, real_v)
        mae_auto        = mae(preds_auto_v, real_v)
        dir_acc_def     = direction_acc(preds_v, real_v)
        dir_acc_auto    = direction_acc(preds_auto_v, real_v)
        calib_def       = calibration(preds_def[:len(test_nets)], test_nets)
        uplift_def      = (realized_med - base_med) if realized_med is not None and base_med is not None else None

        preliminary = len(test_nets) < PRELIM_THRESHOLD

        route_results.append({
            "route_id":          route_id,
            "is_primary":        route_id == primary,
            "preliminary":       preliminary,
            "n_train":           len(train_nets),
            "n_test":            len(valid_pairs),
            "k":                 k,
            "base_rate_median":  round(base_med, 2) if base_med is not None else None,
            "base_rate_wr":      round(base_wr, 3)  if base_wr  is not None else None,
            "realized_median":   round(realized_med, 2) if realized_med is not None else None,
            "realized_wr":       round(realized_wr_val, 3) if realized_wr_val is not None else None,
            "pred_median_def":   round(pred_med, 2) if pred_med is not None else None,
            "pred_median_auto":  round(pred_auto_med, 2) if pred_auto_med is not None else None,
            "mae_def":           round(mae_def, 2) if mae_def is not None else None,
            "mae_auto":          round(mae_auto, 2) if mae_auto is not None else None,
            "dir_acc_def":       round(dir_acc_def, 3) if dir_acc_def is not None else None,
            "dir_acc_auto":      round(dir_acc_auto, 3) if dir_acc_auto is not None else None,
            "uplift_vs_baserate": round(uplift_def, 2) if uplift_def is not None else None,
            "calibration":       calib_def,
            "auto_weights":      {k2: round(v, 3) for k2, v in sorted(auto_w.items(), key=lambda x: -x[1])},
        })

    return {
        "symbol":          symbol,
        "liq_side":        liq_side,
        "n_total":         len(events),
        "n_train":         len(train),
        "n_test":          len(test),
        "train_ends_utc":  train_cut_utc,
        "test_starts_utc": test_start_utc,
        "primary_route":   primary,
        "routes":          route_results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float | None, digits: int = 1, sign: bool = True) -> str:
    if v is None:
        return "NA"
    return (f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}")


def write_report(results: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_ratio": TRAIN_RATIO,
        "k_default": K_DEFAULT,
        "prelim_threshold": PRELIM_THRESHOLD,
        "calib_threshold_bps": CALIB_STRONG_BPS,
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# S34 Phase 2 Calculator Validation",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Train/test split: {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}% by event timestamp (strict temporal, no leakage).",
        f"KNN: K={K_DEFAULT} (adaptive: min(20, train_n//5)), default weights from research_s34_cluster_geometry_features.py.",
        f"Auto-weights: correlation-based reweighting on train set. Preliminary: test N < {PRELIM_THRESHOLD}.",
        "",
    ]

    # --- Summary ranking ---
    lines += ["## Summary — Primary Route KNN Uplift Ranking", ""]
    summary_rows = []
    for r in results:
        prim = r.get("primary_route")
        for rv in r.get("routes", []):
            if rv["route_id"] == prim:
                prelim = "*" if rv.get("preliminary") else ""
                uplift = rv.get("uplift_vs_baserate")
                dir_acc = rv.get("dir_acc_def")
                mae_val = rv.get("mae_def")
                summary_rows.append({
                    "label": f"{r['symbol']} {r['liq_side']}",
                    "route": rv["route_id"],
                    "n_test": rv["n_test"],
                    "prelim": prelim,
                    "base_med": rv.get("base_rate_median"),
                    "realized_med": rv.get("realized_median"),
                    "pred_med": rv.get("pred_median_def"),
                    "uplift": uplift,
                    "dir_acc": dir_acc,
                    "mae": mae_val,
                })
    # Sort by uplift desc (None last)
    summary_rows.sort(key=lambda x: x["uplift"] if x["uplift"] is not None else -999, reverse=True)
    lines += [
        "| Combo | Route | N_test | Base Median | Realized Median | Pred Median | Uplift | Dir Acc | MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary_rows:
        prelim = s["prelim"]
        lines.append(
            f"| {s['label']}{prelim} | {s['route']} | {s['n_test']} "
            f"| {_fmt(s['base_med'])} | {_fmt(s['realized_med'])} "
            f"| {_fmt(s['pred_med'])} | {_fmt(s['uplift'])} "
            f"| {_fmt(s['dir_acc']*100, 0, False) + '%' if s['dir_acc'] is not None else 'NA'} "
            f"| {_fmt(s['mae'], 1, False)} |"
        )
    lines += [
        "",
        "\\* = preliminary (test N < 30)",
        "",
        "> **Uplift** = realized_median(test) - base_rate_median(train). Positive = KNN selected a better-than-average subset.",
        "> **Dir Acc** = fraction of test events where sign(KNN prediction) == sign(realized outcome).",
        "> **MAE** = mean absolute error between per-event KNN prediction and realized outcome.",
        "",
    ]

    # --- Per-combo detail ---
    lines.append("---")
    lines.append("")
    for r in results:
        lines += [
            f"## {r['symbol']} {r['liq_side']}",
            "",
            f"- Total events: {r['n_total']}  Train: {r['n_train']}  Test: {r['n_test']}",
            f"- Train ends: {r['train_ends_utc']}  Test starts: {r['test_starts_utc']}",
            f"- Primary route: `{r.get('primary_route', '-')}`",
            "",
        ]
        for rv in r.get("routes", []):
            prim_tag = " **(primary)**" if rv.get("is_primary") else ""
            prelim_tag = " **[PRELIMINARY]**" if rv.get("preliminary") else ""
            lines += [
                f"### {rv['route_id']}{prim_tag}{prelim_tag}",
                "",
                f"Train N={rv['n_train']}  Test N={rv['n_test']}  K={rv['k']}",
                "",
                "| | Base-rate | KNN (default) | KNN (auto-w) |",
                "|---|---:|---:|---:|",
                f"| Predicted median | {_fmt(rv['base_rate_median'])} | {_fmt(rv['pred_median_def'])} | {_fmt(rv['pred_median_auto'])} |",
                f"| Realized median | — | {_fmt(rv['realized_median'])} | {_fmt(rv['realized_median'])} |",
                f"| MAE | — | {_fmt(rv['mae_def'], 1, False)} | {_fmt(rv['mae_auto'], 1, False)} |",
                f"| Direction accuracy | — | {_fmt(rv['dir_acc_def']*100 if rv['dir_acc_def'] else None, 0, False) + '%' if rv['dir_acc_def'] else 'NA'} | {_fmt(rv['dir_acc_auto']*100 if rv['dir_acc_auto'] else None, 0, False) + '%' if rv['dir_acc_auto'] else 'NA'} |",
                f"| Uplift vs base-rate | — | {_fmt(rv['uplift_vs_baserate'])} | — |",
                f"| Base-rate WR | {_fmt(rv['base_rate_wr']*100 if rv['base_rate_wr'] else None, 0, False) + '%' if rv['base_rate_wr'] else 'NA'} | — | — |",
                "",
            ]
            calib = rv.get("calibration", [])
            if calib:
                lines += [
                    "Calibration (KNN default, threshold " + str(CALIB_STRONG_BPS) + " bps):",
                    "",
                    "| Predicted bucket | N | Pred Median | Realized Median | WR |",
                    "|---|---:|---:|---:|---:|",
                ]
                for b in calib:
                    lines.append(
                        f"| {b['bucket']} | {b['n']} "
                        f"| {_fmt(b['pred_median'])} | {_fmt(b['realized_median'])} "
                        f"| {_fmt(b['realized_wr']*100 if b['realized_wr'] else None, 0, False) + '%' if b['realized_wr'] else 'NA'} |"
                    )
                lines.append("")

    # --- Verdict ---
    useful = [s for s in summary_rows if s["uplift"] is not None and s["uplift"] > 5 and not s["prelim"]]
    noisy  = [s for s in summary_rows if s["uplift"] is not None and s["uplift"] <= 0 and not s["prelim"]]
    lines += [
        "---",
        "",
        "## Verdict",
        "",
        "**Calculator adds value (uplift > +5 bps, confirmed):**",
    ]
    for s in useful:
        lines.append(f"- {s['label']} / {s['route']}: uplift={_fmt(s['uplift'])} bps, dir_acc={_fmt(s['dir_acc']*100 if s['dir_acc'] else None, 0, False) + '%' if s['dir_acc'] else 'NA'}")
    if not useful:
        lines.append("- None confirmed")
    lines += [
        "",
        "**Treat as base-rate only (uplift <= 0, confirmed):**",
    ]
    for s in noisy:
        lines.append(f"- {s['label']} / {s['route']}: uplift={_fmt(s['uplift'])} bps")
    if not noisy:
        lines.append("- None")
    lines += [
        "",
        "_Read-only validation. No runner, config, or pre-reg changes made._",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not FEATURE_DB_PATH.exists():
        raise SystemExit(f"Feature DB not found: {FEATURE_DB_PATH}")

    fdb = sqlite3.connect(FEATURE_DB_PATH)
    fdb.row_factory = sqlite3.Row

    results = []
    for symbol, liq_side in COMBOS:
        print(f"\n=== {symbol} {liq_side} ===", flush=True)
        result = evaluate_combo(fdb, symbol, liq_side)
        results.append(result)
        n_test = result.get("n_test", 0)
        for rv in result.get("routes", []):
            if rv.get("is_primary"):
                print(f"  {rv['route_id']}  base={_fmt(rv['base_rate_median'])}  "
                      f"realized={_fmt(rv['realized_median'])}  "
                      f"pred={_fmt(rv['pred_median_def'])}  "
                      f"uplift={_fmt(rv['uplift_vs_baserate'])}  "
                      f"dir_acc={_fmt(rv['dir_acc_def']*100 if rv['dir_acc_def'] else None, 0, False) + '%' if rv['dir_acc_def'] else 'NA'}  "
                      f"N_test={n_test}{'*' if rv.get('preliminary') else ''}",
                      flush=True)

    fdb.close()
    write_report(results)
    print(f"\nJSON: {OUT_JSON}")
    print(f"MD  : {OUT_MD}")


if __name__ == "__main__":
    main()
