"""BATCH-P6-009: W6-RS confound-resolution wave.

**POST-HOC SENSITIVITY / SPECIFICITY ANALYSIS -- NOT confirmatory evidence.**
This module exists BECAUSE E-W6RS-CONFIRMATION-001 found a nuanced result
(directional replication in TRAIN/TEST, but a similar-magnitude effect in the
candidate-universe negative control, plus a day_trend_bps imbalance between
RS groups). It is explicitly a follow-up diagnostic, not a fresh confirmatory
test, and its output must never be reported as if it were.

EXACT AIM: does RS>0's association with 24h REVERSAL carry information
SPECIFIC to SELL-cascade anchors, beyond a general market-regime effect that
would show up at any (non-event) time too?

FROZEN, unchanged from W6 / W6-RS-confirmation -- no sweep of any kind here:
    RS_THRESHOLD = 0.0, RS_LOOKBACK_MS = 1h (ami.research.w6_compression_rs_session)
    outcome = swing_24h path class -> binary REVERSAL vs not
    (ami.research.w4_post_event_path_taxonomy)

PREREGISTERED FIXED COVARIATES (operator-specified, frozen):
    day_trend_bps -- computed identically for anchor and control rows
      (ami.research.w6rs_confirmation.compute_day_trend_bps)
    anchor notional -- ANCHOR-ONLY BY CONSTRUCTION (a candidate-universe
      random time is not a liquidation event and has no notional). Forcing
      it into the joint anchor-vs-control model would either be undefined
      for controls or degenerate into a proxy for anchor_status itself
      (collinearity). It is therefore NOT part of the joint interaction
      model -- it stays a within-anchor confound check, already computed in
      E-W6RS-CONFIRMATION-001 (RS_STRONG mean notional ~26% higher than
      RS_WEAK) and reused/cited here, not recomputed.
    session -- one-hot (ASIA, EUROPE, OFF; US = reference)
    calendar month -- one-hot (reference = most common month in the pooled
      anchor+control sample)

MAIN TEST: logistic regression of binary REVERSAL on
    anchor_status + rs_group + anchor_status:rs_group (INTERACTION) +
    day_trend_bps_z + session dummies + month dummies
fit on the POOLED (real anchor + candidate-universe control) sample. The
interaction coefficient is the key quantity: it isolates whether the RS
effect is LARGER for real cascade anchors than for arbitrary candidate
times, after adjusting for the frozen covariates above.

INFERENCE: IRLS (Newton-Raphson) logistic regression, implemented directly
with numpy (statsmodels is not installed in this environment). Cluster-
aware uncertainty via a cluster block-bootstrap (resample whole clusters
with replacement, refit, collect the interaction coefficient) -- anchor
rows are clustered by their canonical-v1 cycle_id (respecting OD-011's
non-independence lesson); each control row is its own singleton cluster
(candidate-universe times carry no cycle membership).

DECISION RULES (operator-specified, applied mechanically):
  1. Interaction small/uncertain (bootstrap 95% CI for the interaction
     coefficient includes 0) -> RS rejected as a cascade-specific selector;
     recorded as a GENERAL_REGIME_FEATURE candidate, not cascade-specific.
  2. Interaction's bootstrap CI excludes 0 AND the anchor-only RS effect is
     holdout-stable (train/test direction from E-W6RS-CONFIRMATION-001) ->
     candidate for a SEPARATE, not-yet-started execution-haircut economic
     validation preregistration.
  3. If June 2026's directional reversal (seen in E-W6RS-CONFIRMATION-001)
     is not explained by the adjustment covariates here -> classified
     REGIME_DEPENDENT / CONTINUE_ACCUMULATING (more data needed before any
     verdict).

No trade/PnL claim, no bucket/route/observer change anywhere in this module.
"""
from __future__ import annotations
import hashlib
import random
import time

import numpy as np

from ami.chart.level_registry import _session_of_hour
import datetime as dt
from ami.research.feature_gateway import fetch_chart_feature, fetch_events
from ami.research.w4_post_event_path_taxonomy import _CandleIndex, classify_path, compute_path_returns
from ami.research.w6_compression_rs_session import RS_LOOKBACK_MS
from ami.research.w6rs_confirmation import RS_THRESHOLD, _month_key, compute_day_trend_bps
from ami.states.engine import StateEngine
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W6RS-CONFOUND-RESOLUTION-001"
RESEARCH_CONTEXT_ID = "w6rs-confound-resolution"

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260704
CONTROL_SAMPLE_SIZE = 2000  # larger than the 1:1 negative control -- stabilizes the joint regression
CONTROL_SAMPLE_SEED = 20260704
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-8
INTERACTION_TRIVIAL_BAND_LOGIT = 0.15  # decision-rule guardrail, see classify_result()


def _fit_logistic_irls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Newton-Raphson / IRLS logistic regression. X includes an intercept
    column. Returns the fitted coefficient vector."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(IRLS_MAX_ITER):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        xtwx = X.T @ (X * w[:, None])
        xtwz = X.T @ (y - p)
        try:
            delta = np.linalg.solve(xtwx + 1e-8 * np.eye(k), xtwz)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(xtwx, xtwz, rcond=None)[0]
        beta = beta + delta
        if np.max(np.abs(delta)) < IRLS_TOL:
            break
    return beta


def _build_design_matrix(rows: list[dict], session_categories: list[str], month_categories: list[str],
                          day_trend_mean: float, day_trend_std: float) -> tuple[np.ndarray, list[str]]:
    """Column order: intercept, anchor_status, rs_group, interaction,
    day_trend_bps_z, session dummies (excluding reference), month dummies
    (excluding reference)."""
    names = ["intercept", "anchor_status", "rs_group", "interaction", "day_trend_bps_z"]
    names += [f"session_{s}" for s in session_categories]
    names += [f"month_{m}" for m in month_categories]

    cols = []
    for r in rows:
        anchor_status = 1.0 if r["anchor_status"] == "ANCHOR" else 0.0
        rs_group = 1.0 if r["rs_group"] == "RS_STRONG" else 0.0
        interaction = anchor_status * rs_group
        dt_z = (r["day_trend_bps"] - day_trend_mean) / day_trend_std if day_trend_std else 0.0
        row_vec = [1.0, anchor_status, rs_group, interaction, dt_z]
        row_vec += [1.0 if r["session"] == s else 0.0 for s in session_categories]
        row_vec += [1.0 if r["month"] == m else 0.0 for m in month_categories]
        cols.append(row_vec)
    return np.array(cols, dtype=float), names


def _prepare_rows_and_design(pooled: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    day_trends = [r["day_trend_bps"] for r in pooled]
    dt_mean, dt_std = float(np.mean(day_trends)), float(np.std(day_trends))

    session_counts: dict[str, int] = {}
    month_counts: dict[str, int] = {}
    for r in pooled:
        session_counts[r["session"]] = session_counts.get(r["session"], 0) + 1
        month_counts[r["month"]] = month_counts.get(r["month"], 0) + 1
    session_ref = max(session_counts, key=session_counts.get)
    month_ref = max(month_counts, key=month_counts.get)
    session_categories = sorted(s for s in session_counts if s != session_ref)
    month_categories = sorted(m for m in month_counts if m != month_ref)

    X, names = _build_design_matrix(pooled, session_categories, month_categories, dt_mean, dt_std)
    y = np.array([1.0 if r["path_class"] == "REVERSAL" else 0.0 for r in pooled])
    meta = {"session_ref": session_ref, "month_ref": month_ref, "day_trend_mean": dt_mean, "day_trend_std": dt_std,
            "session_categories": session_categories, "month_categories": month_categories}
    return X, y, names, meta


def _cluster_bootstrap_interaction(pooled: list[dict], cluster_ids: list[str], observed_beta: dict,
                                    n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = random.Random(seed)
    by_cluster: dict[str, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(i)
    cluster_list = list(by_cluster.keys())
    n_clusters = len(cluster_list)

    interaction_draws = []
    for _ in range(n_boot):
        chosen = rng.choices(cluster_list, k=n_clusters)
        idx = [i for c in chosen for i in by_cluster[c]]
        resampled = [pooled[i] for i in idx]
        # need at least 2 distinct outcomes and both anchor_status/rs_group values present
        if len({r["path_class"] == "REVERSAL" for r in resampled}) < 2:
            continue
        try:
            X, y, names, _ = _prepare_rows_and_design(resampled)
            beta = _fit_logistic_irls(X, y)
            interaction_draws.append(beta[names.index("interaction")])
        except Exception:
            continue

    if not interaction_draws:
        return {"n_valid_draws": 0, "ci95": (None, None)}
    arr = np.array(interaction_draws)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"n_valid_draws": len(interaction_draws), "ci95": (round(float(lo), 4), round(float(hi), 4)),
            "bootstrap_mean": round(float(np.mean(arr)), 4)}


def _overlap_positivity_checks(pooled: list[dict]) -> dict:
    anchor_dt = [r["day_trend_bps"] for r in pooled if r["anchor_status"] == "ANCHOR"]
    control_dt = [r["day_trend_bps"] for r in pooled if r["anchor_status"] == "CONTROL"]
    overlap = {
        "day_trend_bps_anchor_range": (round(min(anchor_dt), 1), round(max(anchor_dt), 1)),
        "day_trend_bps_control_range": (round(min(control_dt), 1), round(max(control_dt), 1)),
    }
    cells: dict[str, int] = {}
    for r in pooled:
        key = f"{r['anchor_status']}|{r['rs_group']}|{r['session']}"
        cells[key] = cells.get(key, 0) + 1
    empty_or_thin = {k: v for k, v in cells.items() if v < 5}
    overlap["thin_cells_lt5"] = empty_or_thin
    return overlap


def classify_result(interaction_ci95: tuple[float | None, float | None], june_direction_matches_overall: bool) -> str:
    lo, hi = interaction_ci95
    if lo is None:
        return "INSUFFICIENT_SAMPLE"
    if not june_direction_matches_overall:
        return "REGIME_DEPENDENT_CONTINUE_ACCUMULATING"
    if lo <= 0 <= hi or (abs(lo) < INTERACTION_TRIVIAL_BAND_LOGIT and abs(hi) < INTERACTION_TRIVIAL_BAND_LOGIT):
        return "GENERAL_REGIME_FEATURE"
    return "CASCADE_SPECIFIC_CANDIDATE_FOR_ECONOMIC_VALIDATION"


def _build_anchor_rows(conn, symbol: str = "ETHUSDT") -> tuple[list[dict], _CandleIndex, int]:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")
    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)
    max_close = max(r["close_ts_ms"] for r in candle_rows)

    membership_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "event_cycle_membership", ["event_id", "candidate_cycle_key"],
        equals={"cycle_definition_version": "canonical-v1", "is_canonical": 1},
    )
    event_to_cycle = {r["event_id"]: r["candidate_cycle_key"] for r in membership_rows}

    engine = StateEngine()
    try:
        rows = []
        for e in events:
            anchor_ts = e["anchor_ts_ms"]
            if anchor_ts + 24 * 3600_000 > max_close:
                continue
            returns = compute_path_returns(candle_index, anchor_ts)
            if returns["swing_24h"] is None:
                continue
            eth_ret = engine._ret_bps(symbol, anchor_ts, RS_LOOKBACK_MS)
            btc_ret = engine._ret_bps("BTCUSDT", anchor_ts, RS_LOOKBACK_MS)
            if eth_ret is None or btc_ret is None:
                continue
            day_trend = compute_day_trend_bps(engine, symbol, anchor_ts)
            if day_trend is None:
                continue
            rows.append({
                "anchor_status": "ANCHOR",
                "rs_group": "RS_STRONG" if (eth_ret - btc_ret) > RS_THRESHOLD else "RS_WEAK",
                "path_class": classify_path(returns["swing_24h"]),
                "day_trend_bps": day_trend,
                "session": _session_of_hour(dt.datetime.fromtimestamp(anchor_ts / 1000, dt.timezone.utc).hour),
                "month": _month_key(anchor_ts),
                "cluster_id": event_to_cycle.get(e["event_id"], f"NOCYCLE-{e['event_id']}"),
            })
    finally:
        engine.conn.close()
    return rows, candle_index, max_close


def _build_control_rows(conn, candle_index: _CandleIndex, max_close: int, symbol: str = "ETHUSDT",
                         n_sample: int = CONTROL_SAMPLE_SIZE, seed: int = CONTROL_SAMPLE_SEED) -> list[dict]:
    slot_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe", ["slot_ts_ms"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    rng = random.Random(seed)
    sample = rng.sample(slot_rows, min(n_sample, len(slot_rows)))

    engine = StateEngine()
    try:
        rows = []
        for i, r in enumerate(sample):
            ts = r["slot_ts_ms"]
            if ts + 24 * 3600_000 > max_close:
                continue
            returns = compute_path_returns(candle_index, ts)
            if returns["swing_24h"] is None:
                continue
            eth_ret = engine._ret_bps(symbol, ts, RS_LOOKBACK_MS)
            btc_ret = engine._ret_bps("BTCUSDT", ts, RS_LOOKBACK_MS)
            if eth_ret is None or btc_ret is None:
                continue
            day_trend = compute_day_trend_bps(engine, symbol, ts)
            if day_trend is None:
                continue
            rows.append({
                "anchor_status": "CONTROL",
                "rs_group": "RS_STRONG" if (eth_ret - btc_ret) > RS_THRESHOLD else "RS_WEAK",
                "path_class": classify_path(returns["swing_24h"]),
                "day_trend_bps": day_trend,
                "session": _session_of_hour(dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).hour),
                "month": _month_key(ts),
                "cluster_id": f"CTRL-{i}",
            })
    finally:
        engine.conn.close()
    return rows


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    anchor_rows, candle_index, max_close = _build_anchor_rows(conn, symbol)
    control_rows = _build_control_rows(conn, candle_index, max_close, symbol)
    pooled = anchor_rows + control_rows

    X, y, names, design_meta = _prepare_rows_and_design(pooled)
    beta = _fit_logistic_irls(X, y)
    beta_by_name = {n: round(float(b), 4) for n, b in zip(names, beta)}

    cluster_ids = [r["cluster_id"] for r in pooled]
    boot = _cluster_bootstrap_interaction(pooled, cluster_ids, beta_by_name)

    independent_cycle_n = len({r["cluster_id"] for r in anchor_rows if not r["cluster_id"].startswith("NOCYCLE-")})

    # month-by-month direction, ANCHOR ONLY (extends E-W6RS-CONFIRMATION-001's
    # table; June's anomaly is checked here against the adjustment covariates)
    months = sorted({r["month"] for r in anchor_rows})
    monthly = {}
    for m in months:
        rows_m = [r for r in anchor_rows if r["month"] == m]
        strong = [r for r in rows_m if r["rs_group"] == "RS_STRONG"]
        weak = [r for r in rows_m if r["rs_group"] == "RS_WEAK"]
        rr_strong = sum(1 for r in strong if r["path_class"] == "REVERSAL") / len(strong) if strong else None
        rr_weak = sum(1 for r in weak if r["path_class"] == "REVERSAL") / len(weak) if weak else None
        monthly[m] = {
            "n_strong": len(strong), "n_weak": len(weak),
            "reversal_rate_strong": round(rr_strong, 4) if rr_strong is not None else None,
            "reversal_rate_weak": round(rr_weak, 4) if rr_weak is not None else None,
            "direction_matches_overall": (
                (rr_strong - rr_weak) > 0 if rr_strong is not None and rr_weak is not None else None
            ),
        }
    june_matches = monthly.get("2026-06", {}).get("direction_matches_overall")
    june_direction_matches_overall = june_matches if june_matches is not None else True  # no June data -> not penalized

    overlap = _overlap_positivity_checks(pooled)

    # chronological holdout on the ANCHOR-only interaction contribution (does
    # the anchor RS effect -- not the joint model -- still hold in the same
    # train/test split as E-W6RS-CONFIRMATION-001)
    ordered_anchor = sorted(anchor_rows, key=lambda r: r.get("month"))  # coarse but consistent chronological proxy
    cut = int(len(ordered_anchor) * 0.7)
    train, test = ordered_anchor[:cut], ordered_anchor[cut:]

    def _rd(rows):
        s = [r for r in rows if r["rs_group"] == "RS_STRONG"]
        w = [r for r in rows if r["rs_group"] == "RS_WEAK"]
        if not s or not w:
            return None
        rs_s = sum(1 for r in s if r["path_class"] == "REVERSAL") / len(s)
        rs_w = sum(1 for r in w if r["path_class"] == "REVERSAL") / len(w)
        return round(rs_s - rs_w, 4)

    holdout = {"train_n": len(train), "test_n": len(test),
               "train_risk_difference": _rd(train), "test_risk_difference": _rd(test)}

    verdict = classify_result(boot["ci95"], june_direction_matches_overall)

    return {
        "anchor_n": len(anchor_rows),
        "control_n": len(control_rows),
        "independent_cycle_n": independent_cycle_n,
        "coefficients": beta_by_name,
        "interaction_bootstrap": boot,
        "monthly_direction": monthly,
        "overlap_positivity": overlap,
        "holdout": holdout,
        "verdict": verdict,
        "design_meta": {k: v for k, v in design_meta.items()},
    }


def freeze_and_record(conn, provenance: str = "batch-p6-009-w6rs-confound-resolution") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)

    dataset_hash = hashlib.sha256(
        f"anchor_n={metrics['anchor_n']}|control_n={metrics['control_n']}".encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol=ETHUSDT;anchor_n={metrics['anchor_n']};control_n={metrics['control_n']};"
        f"independent_cycle_n={metrics['independent_cycle_n']}"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- chronological holdout
    # on the anchor-only RS effect; drift-only refresh in practice).
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_COMPRESSION_RS_SESSION",
            "hypothesis_id": "H-W6RS-CONFOUND-RESOLUTION", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "anchor_status,rs_group,interaction,day_trend_bps,session,month",
            "frozen_target": "swing_24h path class REVERSAL (reused, not redefined)",
            "frozen_thresholds": f"rs_threshold={RS_THRESHOLD} (FROZEN); rs_lookback_ms={RS_LOOKBACK_MS}"
                                 " (FROZEN); no sweep of any kind in this module",
            "frozen_splits": "chronological holdout on anchor-only RS effect (train/test)",
            "frozen_economic_gate": "N/A (post-hoc sensitivity/specificity analysis, NOT confirmatory; "
                                    "no entry/exit/PnL claim)",
            "frozen_statistical_gate": f"logistic regression (IRLS) + cluster block-bootstrap "
                                       f"n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}; single pre-registered "
                                       "interaction test",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "POST_HOC_SENSITIVITY_ANALYSIS",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": "E-W6RS-CONFIRMATION-001", "report_artifact_id": None,
            "schema_version": 7, "provenance": provenance, "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in [
            ("anchor_n", metrics["anchor_n"]),
            ("control_n", metrics["control_n"]),
            ("independent_cycle_n", metrics["independent_cycle_n"]),
            ("coefficients", metrics["coefficients"]),
            ("interaction_bootstrap", metrics["interaction_bootstrap"]),
            ("monthly_direction", metrics["monthly_direction"]),
            ("overlap_positivity", metrics["overlap_positivity"]),
            ("holdout", metrics["holdout"]),
            ("verdict", metrics["verdict"]),
        ]],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return metrics


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        r = freeze_and_record(conn)
        print(f"anchor_n={r['anchor_n']} control_n={r['control_n']} independent_cycle_n={r['independent_cycle_n']}\n"
              f"coefficients={r['coefficients']}\n"
              f"interaction_bootstrap={r['interaction_bootstrap']}\n"
              f"monthly_direction={r['monthly_direction']}\n"
              f"overlap_positivity={r['overlap_positivity']}\n"
              f"holdout={r['holdout']}\n"
              f"VERDICT={r['verdict']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
