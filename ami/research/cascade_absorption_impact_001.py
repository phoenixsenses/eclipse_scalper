"""BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1.

Executes the frozen preregistration `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-
PREREG-001` (reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_
PREREGISTRATION_V1.md, committed as fb002a75) exactly once. This module is
the ONLY code that reads the TEST population's `endpoint_return_bps`/
`mfe_bps` values for this experiment, and it may only do so after
`epistemic_gates.consume_test_evidence()` has marked the frozen TEST-
evidence nullifier consumed for this exact experiment_id.

Every identity constant below is copied verbatim from the committed
preregistration artifacts -- not re-derived from a `frozen_splits` free-text
description (same discipline as `ami/research/cvd_windowed_flow_001.py`,
whose exact lifecycle behavior this module mirrors for consistency:
`verify_pre_execution()` independently reproduces the population/split/
family/nullifier from the real database and raises `ProtocolInvalidation`
if any of it drifts from these frozen constants -- the fail-closed check
the frozen preregistration itself requires before any TEST access).

FROZEN DEVIATION FROM THE CVD PRECEDENT (deliberate, recorded in the
preregistration itself, not an oversight): the CVD execution kept an
always-zero `session_EUROPE` design column and used `np.linalg.pinv` to
handle the resulting singularity. THIS experiment's preregistration
explicitly froze the opposite policy (`pseudo_inverse_permitted: false`,
`rank_valid_encoding_permitted: true`): `session_EUROPE` is dropped from
the design matrix entirely before fitting, and a genuine `np.linalg.inv`
is used, with an explicit pre-fit matrix-rank check as a mandatory
validation (not merely assumed from the parameter count).

No code in this module may be changed after TEST outcomes have been read
for this experiment_id (frozen protocol). If a bug is found post-execution,
a corrected rerun is a NEW experiment_id, never a silent edit here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time

import numpy as np
from scipy import stats as _stats

from ami.chart.level_registry import _session_of_hour
from ami.governance import epistemic_gates as gates
from ami.research.feature_gateway import fetch_events, fetch_lifecycle_signals
from ami.research.w6rs_confirmation import compute_day_trend_bps
from ami.states.engine import StateEngine
from ami.warehouse.experiment_ledger import record_experiment_registry, record_experiment_results

# ---------------------------------------------------------------------------
# Frozen identity (copied verbatim from the committed preregistration fb002a75;
# never re-derived from prose -- see module docstring)
# ---------------------------------------------------------------------------
RESEARCH_CONTEXT_ID = "cascade-absorption-impact-001-execution"
SYMBOL = "ETHUSDT"
HORIZON = "swing_24h"

EXPERIMENT_ID = "E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001"
QUESTION_IDS = "FAM_CASCADE_ABSORPTION_IMPACT_LONG_REVERSAL"
HYPOTHESIS_ID = "H-CASCADE-ABSORPTION-IMPACT-LONG-W300-EXACT-PRICE-RESPONSE-PER-SIGNED-NOTIONAL-V1"
FAMILY_ID = "FAMv1:3e2dfe63f9e271bf"
SPLIT_VERSION = "SPLITv1:16ea98c239034593"
SPEC_HASH = "531b16232a88d5a6c692055bd00fa59bd508b7b69cd7fd45cf8e666772fb6608"
EXPECTED_TRAIN_HASH = "61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5"
EXPECTED_TEST_HASH = "98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04"
EXPECTED_NULLIFIER = "4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4"
PREREG_GATE_RECEIPT_HASH = "6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4"
PREREG_CODE_COMMIT = "fb002a75"
EXPECTED_SCHEMA_VERSION = 13

TRAIN_FRACTION = 0.7
MIN_BUCKET_N = 20
EVENT_SCALE = 100_000.0
RELEVANCE_FLOOR_BPS = 5.0
PREDICTOR_TRAIN_STDEV = 10.70108397867223  # frozen at preregistration time, re-verified below
CORRECTED_PATH_VERSION = "path-v2-candle-repair-r1"
COLLINEARITY_VIF_THRESHOLD = 10.0

# Frozen policy: EUROPE dropped entirely (0/91 TRAIN, 0/40 TEST) -- no
# always-zero column, no pseudo-inverse. 6 parameters, ASIA implicit reference.
DESIGN_NAMES = ("const", "price_response_w300", "event_notional_per_100k",
                "session_US", "session_OFF", "day_trend_bps")

VERDICT_SUPPORTS = "SUPPORTS_INCREMENTAL_ABSORPTION_IMPACT_ASSOCIATION"
VERDICT_NO_RELIABLE = "NO_RELIABLE_INCREMENTAL_ASSOCIATION"
VERDICT_UNDERPOWERED = "UNDERPOWERED_OR_INCONCLUSIVE"
VERDICT_INVALIDATED = "PROTOCOL_OR_DATA_QUALITY_INVALIDATED"


class ProtocolInvalidation(Exception):
    """Raised when the real database no longer reproduces the frozen
    preregistration's identity/population/split/nullifier/gate state, or
    when the design matrix fails its mandatory pre-fit rank check. Per the
    governed-execution protocol, this must stop BEFORE any TEST outcome
    access -- never repaired or improvised after seeing TEST data."""


def _cycle_key(signal: dict) -> str:
    return signal["independent_cycle_id"] or f"NOCYCLE-{signal['source_event_id']}"


# ---------------------------------------------------------------------------
# Step 1: population/split resolution (TEST-outcome-blind by construction --
# only signal_id/independent_cycle_id/signal_birth_ts/source_event_id and
# swing_24h observation_status are ever read here; endpoint_return_bps/
# mfe_bps columns are never selected by this function)
# ---------------------------------------------------------------------------

def resolve_population(conn) -> dict:
    signals = fetch_lifecycle_signals(conn, RESEARCH_CONTEXT_ID, symbol=SYMBOL)
    long_signals = {s["signal_id"]: s for s in signals if s["direction"] == "LONG"}

    obs_rows = conn.execute(
        "SELECT signal_id, path_definition_version, observation_status "
        "FROM ami_lifecycle_path_observations WHERE horizon_name=?", (HORIZON,)).fetchall()
    effective_status: dict[str, str] = {}
    version_by_sig: dict[str, str] = {}
    for sid, version, status in obs_rows:
        if sid not in version_by_sig or version == CORRECTED_PATH_VERSION:
            effective_status[sid] = status
            version_by_sig[sid] = version

    eligible_long = [s for sid, s in long_signals.items() if effective_status.get(sid) == "OK"]

    by_cycle: dict[str, dict] = {}
    for s in eligible_long:
        key = _cycle_key(s)
        if key not in by_cycle or s["signal_birth_ts"] < by_cycle[key]["signal_birth_ts"]:
            by_cycle[key] = s

    reps = sorted(by_cycle.items(), key=lambda kv: kv[1]["signal_birth_ts"])
    cut = int(len(reps) * TRAIN_FRACTION)
    train_reps = reps[:cut]
    test_reps = reps[cut:]

    train_hash = hashlib.sha256(",".join(sorted(k for k, _ in train_reps)).encode()).hexdigest()
    test_hash = hashlib.sha256(",".join(sorted(k for k, _ in test_reps)).encode()).hexdigest()

    return {
        "long_n": len(long_signals),
        "eligible_long_n": len(eligible_long),
        "representative_cycle_n": len(by_cycle),
        "train_reps": train_reps,
        "test_reps": test_reps,
        "train_hash": train_hash,
        "test_hash": test_hash,
    }


def verify_pre_execution(canonical_conn, knowledge_conn) -> dict:
    """Read-only. Reproduces every frozen identity element and reports any
    drift. Raises nothing itself -- callers must check `errors`."""
    pop = resolve_population(canonical_conn)
    errors: list[str] = []

    if pop["representative_cycle_n"] != 131:
        errors.append(f"representative_cycle_n={pop['representative_cycle_n']} != 131")
    if len(pop["train_reps"]) != 91 or len(pop["test_reps"]) != 40:
        errors.append(f"split count mismatch: train={len(pop['train_reps'])} test={len(pop['test_reps'])}")
    if pop["train_hash"] != EXPECTED_TRAIN_HASH:
        errors.append(f"TRAIN cycle-set hash mismatch: {pop['train_hash']}")
    if pop["test_hash"] != EXPECTED_TEST_HASH:
        errors.append(f"TEST cycle-set hash mismatch: {pop['test_hash']}")
    if pop["train_reps"] and pop["test_reps"]:
        if pop["train_reps"][-1][1]["signal_birth_ts"] >= pop["test_reps"][0][1]["signal_birth_ts"]:
            errors.append("TRAIN/TEST chronological straddling detected")

    train_keys = {k for k, _ in pop["train_reps"]}
    test_keys = {k for k, _ in pop["test_reps"]}
    if train_keys & test_keys:
        errors.append(f"TRAIN/TEST cycle-key overlap: {sorted(train_keys & test_keys)[:5]}")

    family_id = gates.resolve_canonical_family_id(QUESTION_IDS, HYPOTHESIS_ID)
    if family_id != FAMILY_ID:
        errors.append(f"family_id mismatch: {family_id} != {FAMILY_ID}")

    test_cycle_ids = [k for k, _ in pop["test_reps"]]
    nullifier = gates.derive_test_nullifier(FAMILY_ID, SPLIT_VERSION, test_cycle_ids)
    if nullifier != EXPECTED_NULLIFIER:
        errors.append(f"nullifier mismatch: {nullifier} != {EXPECTED_NULLIFIER}")

    prior = knowledge_conn.execute(
        "SELECT consumed_by_experiment_id FROM epistemic_test_nullifiers WHERE nullifier=?",
        (nullifier,)).fetchall()
    consumed_by_other = [p[0] for p in prior if p[0] != EXPERIMENT_ID]
    if consumed_by_other:
        errors.append(f"nullifier already consumed by a DIFFERENT experiment: {consumed_by_other}")
    is_rerun_of_self = any(p[0] == EXPERIMENT_ID for p in prior)

    receipt = knowledge_conn.execute(
        "SELECT registry_result, canonical_family_id, split_version, nullifier "
        "FROM experiment_gate_receipts WHERE experiment_id=?", (EXPERIMENT_ID,)).fetchone()
    if receipt is None:
        errors.append("gate receipt missing for experiment_id")
    elif receipt[1] != FAMILY_ID or receipt[2] != SPLIT_VERSION or receipt[3] != nullifier:
        errors.append(f"gate receipt identity mismatch: {receipt}")
    elif receipt[0] not in ("PREREGISTERED_NOT_EXECUTED", "EXECUTED"):
        errors.append(f"unexpected gate receipt state: {receipt[0]}")

    w300 = canonical_conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN quality_status='EXACT_RECONSTRUCTABLE' THEN 1 ELSE 0 END) "
        "FROM ami_absorption_impact_window_quality_v1 WHERE window_id='W300'").fetchone()
    if w300[0] != 324 or w300[1] != 324:
        errors.append(f"W300 quality-population drift: total={w300[0]} exact={w300[1]}")

    violations = canonical_conn.execute(
        "SELECT COUNT(*) FROM ami_absorption_impact_windowed_flow WHERE window_id='W300' AND "
        "(window_end_ts_ms > signal_birth_ts OR known_at_classification != 'KNOWN_AT_SAFE')"
    ).fetchone()[0]
    if violations != 0:
        errors.append(f"known-at violations for W300: {violations}")

    proxy_check = canonical_conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE '%absorption_impact%proxy%'").fetchall()
    if proxy_check:
        errors.append(f"unexpected proxy table present: {proxy_check}")

    schema_version = canonical_conn.execute(
        "SELECT version FROM schema_versions WHERE component='canonical_warehouse'").fetchone()[0]
    if schema_version != EXPECTED_SCHEMA_VERSION:
        errors.append(f"schema_version drift: {schema_version} != {EXPECTED_SCHEMA_VERSION}")

    reg_count = canonical_conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    res_count = canonical_conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
    already_has_results = canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)).fetchone()[0]

    return {
        "pop": pop, "family_id": family_id, "nullifier": nullifier,
        "is_rerun_of_self": is_rerun_of_self, "receipt": receipt, "errors": errors,
        "experiment_registry_count_before": reg_count, "experiment_results_count_before": res_count,
        "already_has_results_before": already_has_results, "schema_version": schema_version,
    }


# ---------------------------------------------------------------------------
# Step 2: predictor/control extraction (scoped strictly to the signal_id
# list supplied by the caller -- calling this with TRAIN signal_ids never
# touches a TEST row, and vice versa)
# ---------------------------------------------------------------------------

def _fetch_predictors_controls(conn, reps: list[tuple[str, dict]], engine: StateEngine,
                                event_notional_by_id: dict) -> list[dict]:
    signal_ids = [s["signal_id"] for _, s in reps]
    if not signal_ids:
        return []
    placeholders = ",".join("?" for _ in signal_ids)
    rows = conn.execute(
        f"SELECT signal_id, price_response_per_signed_notional, floor_usd_m_applied "
        f"FROM ami_absorption_impact_windowed_flow "
        f"WHERE window_id='W300' AND signal_id IN ({placeholders})", signal_ids).fetchall()
    predictor_by_signal = {r[0]: (r[1], r[2]) for r in rows}

    out = []
    for cyc, s in reps:
        sid = s["signal_id"]
        predictor_row = predictor_by_signal.get(sid)
        price_response = predictor_row[0] if predictor_row else None
        floor_applied = bool(predictor_row[1]) if predictor_row else None
        event_notional = event_notional_by_id.get(s["source_event_id"])
        hour = dt.datetime.fromtimestamp(s["signal_birth_ts"] / 1000, dt.timezone.utc).hour
        session = _session_of_hour(hour)
        day_trend = compute_day_trend_bps(engine, SYMBOL, s["signal_birth_ts"])
        missing = price_response is None or event_notional is None or day_trend is None
        out.append({
            "cycle_key": cyc, "signal_id": sid,
            "price_response_w300": price_response, "floor_applied": floor_applied,
            "event_notional": event_notional,
            "session": session, "day_trend_bps": day_trend, "missing_predictor": missing,
        })
    return out


def _fetch_effective_outcome_for_signals(conn, signal_ids: list[str], horizon: str = HORIZON) -> dict:
    """Scoped strictly to `signal_ids` -- this is the ONLY function in this
    module (or the whole codebase, for this experiment) permitted to read
    `endpoint_return_bps`/`mfe_bps` for the given signal set. Callers must
    never pass a mix of TRAIN and not-yet-authorized TEST signal_ids."""
    if not signal_ids:
        return {}
    placeholders = ",".join("?" for _ in signal_ids)
    rows = conn.execute(
        f"SELECT signal_id, path_definition_version, observation_status, "
        f"endpoint_return_bps, mfe_bps FROM ami_lifecycle_path_observations "
        f"WHERE horizon_name=? AND signal_id IN ({placeholders})",
        [horizon, *signal_ids]).fetchall()
    effective: dict[str, dict] = {}
    version_by_sig: dict[str, str] = {}
    for sid, version, status, ret_bps, mfe in rows:
        if sid not in version_by_sig or version == CORRECTED_PATH_VERSION:
            effective[sid] = {"observation_status": status, "endpoint_return_bps": ret_bps, "mfe_bps": mfe}
            version_by_sig[sid] = version
    return effective


# ---------------------------------------------------------------------------
# Step 3: statistics (pure functions, testable on synthetic data)
# ---------------------------------------------------------------------------

def build_design(rows: list[dict], outcome_key: str) -> dict:
    """rows: dicts with price_response_w300/event_notional/session/
    day_trend_bps/<outcome_key>/missing_predictor. Applies listwise
    deletion. Frozen policy: EUROPE is not a column at all (never observed
    in TRAIN or TEST for this population) -- ASIA is the implicit reference,
    US/OFF are the only encoded dummies."""
    total = len(rows)
    usable = [r for r in rows if not r.get("missing_predictor") and r.get(outcome_key) is not None]
    dropped = total - len(usable)
    n = len(usable)
    X = np.zeros((n, len(DESIGN_NAMES)))
    y = np.zeros(n)
    cluster_ids = []
    for i, r in enumerate(usable):
        if r["session"] not in ("ASIA", "US", "OFF"):
            raise ProtocolInvalidation(
                f"PROTOCOL_OR_DATA_QUALITY_INVALIDATED: unexpected session level "
                f"{r['session']!r} for signal_id={r['signal_id']!r} -- EUROPE was frozen as "
                "having zero TRAIN/TEST observations; any occurrence (including EUROPE itself) "
                "invalidates the frozen design matrix, per the preregistration's own policy.")
        X[i, 0] = 1.0
        X[i, 1] = r["price_response_w300"]
        X[i, 2] = r["event_notional"] / EVENT_SCALE
        X[i, 3] = 1.0 if r["session"] == "US" else 0.0
        X[i, 4] = 1.0 if r["session"] == "OFF" else 0.0
        X[i, 5] = r["day_trend_bps"]
        y[i] = r[outcome_key]
        cluster_ids.append(r["cycle_key"])
    return {"X": X, "y": y, "cluster_ids": cluster_ids, "n": n, "n_total": total, "n_dropped": dropped}


def check_design_rank(X: np.ndarray) -> dict:
    """Mandatory pre-fit validation (frozen policy: no pseudo-inverse
    permitted). Rank must equal column count -- any deficiency is
    PROTOCOL_OR_DATA_QUALITY_INVALIDATED, never silently patched."""
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))
    return {"n": n, "k": k, "rank": rank, "full_rank": rank == k}


def run_cluster_robust_ols(X: np.ndarray, y: np.ndarray, cluster_ids: list[str]) -> dict:
    """OLS with the CR1 cluster-robust sandwich variance estimator
    (Cameron-Miller convention: V = c * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1,
    c = (N-1)/(N-k) * G/(G-1)), inference via a t-distribution with G-1 df.

    Frozen policy deviation from the CVD precedent: uses a genuine
    `np.linalg.inv` (never `pinv`) -- the design matrix is expected to be
    full rank because the zero-variance EUROPE column was dropped entirely
    upstream (build_design), not because of any looser numerical tolerance
    here. Callers must run `check_design_rank` first and treat any rank
    deficiency as PROTOCOL_OR_DATA_QUALITY_INVALIDATED before ever calling
    this function."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    clusters: dict[str, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)
    G = len(clusters)

    meat = np.zeros((k, k))
    for idxs in clusters.values():
        Xg = X[idxs, :]
        ug = resid[idxs]
        score = Xg.T @ ug
        meat += np.outer(score, score)

    correction = ((n - 1) / (n - k)) * (G / (G - 1)) if G > 1 and n > k else float("nan")
    V = correction * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    df = max(G - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
        p_values = 2.0 * (1.0 - _stats.t.cdf(np.abs(t_stats), df))
    crit = _stats.t.ppf(0.975, df)
    ci_lo = beta - crit * se
    ci_hi = beta + crit * se

    return {"beta": beta, "se": se, "t_stats": t_stats, "p_values": p_values,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "df": df, "n": n, "k": k, "G": G, "resid": resid}


def compute_vif(X: np.ndarray) -> dict:
    """VIF_j = 1/(1-R_j^2). Only the three continuous columns (predictor,
    event_notional, day_trend_bps) are ever meaningfully computed here --
    session dummies are excluded from VIF-based dropping entirely (frozen
    policy: session is governed only by the zero-variance structural rule,
    never by VIF)."""
    n, k = X.shape
    vifs = {}
    for j in (1, 2, 5):  # price_response_w300, event_notional_per_100k, day_trend_bps
        y_j = X[:, j]
        other_cols = [c for c in range(k) if c != j]
        Xo = X[:, other_cols]
        ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
        if ss_tot <= 0:
            vifs[DESIGN_NAMES[j]] = None
            continue
        beta_j, *_ = np.linalg.lstsq(Xo, y_j, rcond=None)
        resid = y_j - Xo @ beta_j
        ss_res = float(np.sum(resid ** 2))
        r2 = 1.0 - ss_res / ss_tot
        vifs[DESIGN_NAMES[j]] = (1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
    return vifs


def apply_collinearity_policy(vifs: dict) -> list[str]:
    """Frozen drop order: day_trend_bps first, then event_notional_per_100k;
    session dummies are never dropped via VIF."""
    drops = []
    day_vif = vifs.get("day_trend_bps")
    if day_vif is not None and day_vif > COLLINEARITY_VIF_THRESHOLD:
        drops.append("day_trend_bps")
    event_vif = vifs.get("event_notional_per_100k")
    if event_vif is not None and event_vif > COLLINEARITY_VIF_THRESHOLD:
        drops.append("event_notional_per_100k")
    return drops


def apply_verdict_rule(n_test: int, coef: float, se: float, ci_lo: float, ci_hi: float,
                        p_value: float) -> tuple[str, str]:
    if n_test < MIN_BUCKET_N:
        return VERDICT_UNDERPOWERED, f"n_test={n_test} < MIN_BUCKET_N={MIN_BUCKET_N}"
    half_width = (ci_hi - ci_lo) / 2.0
    relevance_floor = RELEVANCE_FLOOR_BPS
    if half_width > 2.0 * relevance_floor:
        return VERDICT_UNDERPOWERED, (
            f"CI half-width {half_width:.4f} exceeds 2x relevance floor ({2 * relevance_floor})")
    ci_excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)
    magnitude = abs(coef * PREDICTOR_TRAIN_STDEV)
    if ci_excludes_zero and p_value < 0.05 and magnitude >= relevance_floor:
        return VERDICT_SUPPORTS, (
            f"CI excludes 0, p={p_value:.4g}<0.05, |coef*{PREDICTOR_TRAIN_STDEV:.4f}|="
            f"{magnitude:.4f}>={relevance_floor}")
    return VERDICT_NO_RELIABLE, (
        f"ci_excludes_zero={ci_excludes_zero} p={p_value:.4g} "
        f"|coef*{PREDICTOR_TRAIN_STDEV:.4f}|={magnitude:.4f}")


# ---------------------------------------------------------------------------
# Step 4: the single governed execution entry point
# ---------------------------------------------------------------------------

def execute_governed_run(canonical_conn, knowledge_conn) -> dict:
    verification = verify_pre_execution(canonical_conn, knowledge_conn)
    if verification["errors"]:
        raise ProtocolInvalidation(
            "CASCADE_ABSORPTION_IMPACT_GOVERNED_EXECUTION_V1_BLOCKED: " + "; ".join(verification["errors"]))

    pop = verification["pop"]
    nullifier = verification["nullifier"]
    engine = StateEngine()
    events = fetch_events(canonical_conn, RESEARCH_CONTEXT_ID, symbol=SYMBOL)
    event_notional_by_id = {e["event_id"]: e["notional"] for e in events}

    # ---- TRAIN: predictors + outcome (always permitted; used only for
    # scaling/VIF/descriptive secondary checks, never to select anything) ----
    train_features = _fetch_predictors_controls(canonical_conn, pop["train_reps"], engine, event_notional_by_id)
    train_signal_ids = [r["signal_id"] for r in train_features]
    train_outcome = _fetch_effective_outcome_for_signals(canonical_conn, train_signal_ids)
    for r in train_features:
        o = train_outcome.get(r["signal_id"], {})
        r["endpoint_return_bps"] = o.get("endpoint_return_bps")
        r["mfe_bps"] = o.get("mfe_bps")

    train_design = build_design(train_features, "endpoint_return_bps")
    train_rank = check_design_rank(train_design["X"])
    if not train_rank["full_rank"]:
        raise ProtocolInvalidation(
            f"PROTOCOL_OR_DATA_QUALITY_INVALIDATED: TRAIN design matrix rank {train_rank['rank']} "
            f"!= column count {train_rank['k']} -- unresolved rank deficiency, no ad hoc fix permitted.")

    # TRAIN-only predictor stdev re-verification (outcome-blind, drift check
    # against the value frozen at preregistration time)
    train_predictor_stdev = float(np.std(train_design["X"][:, 1], ddof=1))
    if abs(train_predictor_stdev - PREDICTOR_TRAIN_STDEV) > 1e-6:
        raise ProtocolInvalidation(
            f"PROTOCOL_OR_DATA_QUALITY_INVALIDATED: TRAIN predictor stdev drift "
            f"{train_predictor_stdev} != frozen {PREDICTOR_TRAIN_STDEV}")

    vif = compute_vif(train_design["X"]) if train_design["n"] > len(DESIGN_NAMES) else {}
    drops = apply_collinearity_policy(vif) if vif else []
    train_fit = run_cluster_robust_ols(train_design["X"], train_design["y"], train_design["cluster_ids"]) \
        if train_design["n"] > len(DESIGN_NAMES) else None

    predictor_scale_stats = {
        "n": train_design["n"],
        "min": float(np.min(train_design["X"][:, 1])) if train_design["n"] else None,
        "max": float(np.max(train_design["X"][:, 1])) if train_design["n"] else None,
        "mean": float(np.mean(train_design["X"][:, 1])) if train_design["n"] else None,
        "median": float(np.median(train_design["X"][:, 1])) if train_design["n"] else None,
        "stdev": train_predictor_stdev,
    }

    # ---- AUTHORIZATION: consume the TEST-evidence nullifier BEFORE any
    # TEST-row outcome is read. This is the point of no return. ----
    test_cycle_ids = [k for k, _ in pop["test_reps"]]
    consume_result = gates.consume_test_evidence(
        knowledge_conn, family_id=FAMILY_ID, split_version=SPLIT_VERSION,
        test_cycle_ids=test_cycle_ids, experiment_id=EXPERIMENT_ID)

    # ---- TEST ACCESS (first and only time TEST outcome is read for this
    # experiment_id) ----
    test_features = _fetch_predictors_controls(canonical_conn, pop["test_reps"], engine, event_notional_by_id)
    test_signal_ids = [r["signal_id"] for r in test_features]
    test_outcome = _fetch_effective_outcome_for_signals(canonical_conn, test_signal_ids)
    for r in test_features:
        o = test_outcome.get(r["signal_id"], {})
        r["endpoint_return_bps"] = o.get("endpoint_return_bps")
        r["mfe_bps"] = o.get("mfe_bps")

    test_design = build_design(test_features, "endpoint_return_bps")
    test_rank = check_design_rank(test_design["X"])
    if not test_rank["full_rank"]:
        raise ProtocolInvalidation(
            f"PROTOCOL_OR_DATA_QUALITY_INVALIDATED: TEST design matrix rank {test_rank['rank']} "
            f"!= column count {test_rank['k']} -- unresolved rank deficiency, no ad hoc fix permitted.")
    test_fit = run_cluster_robust_ols(test_design["X"], test_design["y"], test_design["cluster_ids"])

    coef = float(test_fit["beta"][1])
    se_coef = float(test_fit["se"][1])
    ci_lo = float(test_fit["ci_lo"][1])
    ci_hi = float(test_fit["ci_hi"][1])
    p_value = float(test_fit["p_values"][1])
    verdict, verdict_reason = apply_verdict_rule(test_design["n"], coef, se_coef, ci_lo, ci_hi, p_value)

    # secondary, non-promotable check: same model with mfe_bps as outcome
    test_design_mfe = build_design(test_features, "mfe_bps")
    secondary_rank = check_design_rank(test_design_mfe["X"]) if test_design_mfe["n"] > len(DESIGN_NAMES) else None
    secondary_fit = (run_cluster_robust_ols(test_design_mfe["X"], test_design_mfe["y"],
                                             test_design_mfe["cluster_ids"])
                      if secondary_rank and secondary_rank["full_rank"] else None)

    now_ms = int(time.time() * 1000)
    frozen_population = (
        f"ami_signal_lifecycle direction=LONG (long_n={pop['long_n']}), absorption/impact W300 "
        f"quality_status=EXACT_RECONSTRUCTABLE (324/324, 0 SOURCE_GAPPED), swing_24h "
        f"observation_status=OK (eligible_long_n={pop['eligible_long_n']}), one representative "
        f"per independent_cycle_id (earliest signal_birth_ts; representative_cycle_n="
        f"{pop['representative_cycle_n']}); test n_used={test_design['n']} "
        f"(n_dropped_missing={test_design['n_dropped']})"
    )
    frozen_features = (
        "primary=ami_absorption_impact_windowed_flow.price_response_per_signed_notional WHERE "
        "window_id='W300', continuous, no scaling; controls=event_notional(/100,000), "
        "session(dummy, ref=ASIA, EUROPE dropped -- 0 TRAIN/TEST observations), day_trend_bps(raw)"
    )
    frozen_target = "endpoint_return_bps@swing_24h (ami_lifecycle_path_observations, effective selection)"
    frozen_thresholds = (
        f"significance p<0.05 two-sided; relevance floor |coef*{PREDICTOR_TRAIN_STDEV:.4f}|>="
        f"{RELEVANCE_FLOOR_BPS}bps; MIN_BUCKET_N={MIN_BUCKET_N}; CI half-width invalidation > "
        f"{2 * RELEVANCE_FLOOR_BPS}bps"
    )
    frozen_splits = (
        f"FROZEN AT PREREGISTRATION (SPLIT_VERSION={SPLIT_VERSION}, reproduced not re-derived): "
        f"cycle-grouped chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split "
        f"by earliest signal_birth_ts per independent_cycle_id, cut by cycle count "
        f"(train={len(pop['train_reps'])}, test={len(pop['test_reps'])})"
    )
    provenance = (
        f"BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1; preregistration_commit={PREREG_CODE_COMMIT}; "
        f"spec_hash={SPEC_HASH}; test_nullifier={nullifier}; "
        f"gate_receipt_hash_at_preregistration={PREREG_GATE_RECEIPT_HASH}; collinearity_drops={drops}; "
        f"design_rank_check=full_rank_no_pinv"
    )

    registry_values = {
        "experiment_id": EXPERIMENT_ID, "question_ids": QUESTION_IDS, "hypothesis_id": HYPOTHESIS_ID,
        "preregistered_at": "2026-07-07T00:00:00Z (commit fb002a75)",
        "frozen_population": frozen_population, "frozen_features": frozen_features,
        "frozen_target": frozen_target, "frozen_thresholds": frozen_thresholds,
        "frozen_splits": frozen_splits, "frozen_economic_gate": "NONE (descriptive/inferential only)",
        "frozen_statistical_gate": (
            "OLS + cluster-robust(CR1) SE clustered by independent_cycle_id; two-sided p<0.05; "
            "95% CI; effect-size relevance floor; VIF collinearity policy; EUROPE session dropped "
            "structurally (no pseudo-inverse)"
        ),
        "code_commit": PREREG_CODE_COMMIT, "dataset_hash": nullifier,
        "started_at": now_ms, "completed_at": now_ms,
        "software_verdict": "COMPLETED", "scientific_verdict": verdict,
        "mutation_test_count": 0, "mutation_test_passed": 0,
        "supersedes_experiment_id": None, "report_artifact_id": None,
        "schema_version": EXPECTED_SCHEMA_VERSION, "provenance": provenance,
        "created_ms": now_ms, "updated_ms": now_ms,
    }

    results_rows = [
        ("primary_predictor_coefficient_bps_per_unit", str(coef)),
        ("primary_predictor_se_cluster_robust", str(se_coef)),
        ("primary_predictor_ci95_lo", str(ci_lo)),
        ("primary_predictor_ci95_hi", str(ci_hi)),
        ("primary_predictor_p_value", str(p_value)),
        ("primary_predictor_t_stat", str(float(test_fit["t_stats"][1]))),
        ("primary_predictor_df", str(test_fit["df"])),
        ("test_n_used", str(test_design["n"])),
        ("test_n_total_representative", str(len(pop["test_reps"]))),
        ("test_n_dropped_missing", str(test_design["n_dropped"])),
        ("test_n_clusters", str(test_fit["G"])),
        ("test_design_rank", str(test_rank["rank"])),
        ("train_n_used", str(train_design["n"])),
        ("train_n_dropped_missing", str(train_design["n_dropped"])),
        ("train_design_rank", str(train_rank["rank"])),
        ("train_predictor_stdev_reverified", str(train_predictor_stdev)),
        ("vif", str(vif)),
        ("collinearity_drops_applied", str(drops)),
        ("predictor_train_scale_stats", str(predictor_scale_stats)),
        ("design_columns", str(DESIGN_NAMES)),
        ("full_beta_vector", str(test_fit["beta"].tolist())),
        ("full_se_vector", str(test_fit["se"].tolist())),
        ("verdict_reason", verdict_reason),
        ("secondary_mfe_bps_coefficient", str(float(secondary_fit["beta"][1])) if secondary_fit else "N/A"),
        ("secondary_mfe_bps_p_value", str(float(secondary_fit["p_values"][1])) if secondary_fit else "N/A"),
        ("train_side_descriptive_coefficient", str(float(train_fit["beta"][1])) if train_fit else "N/A"),
        ("train_side_descriptive_p_value", str(float(train_fit["p_values"][1])) if train_fit else "N/A"),
        ("test_cycle_set_hash", pop["test_hash"]),
        ("train_cycle_set_hash", pop["train_hash"]),
        ("test_nullifier_sha256", nullifier),
        ("cross_family_test_cycle_reuse_disclosure",
         "identical TEST cycle set to E-CVD-PRIMARY-LONG-W300-PREREG-001 (different canonical family/"
         "nullifier, not an independent market-period replication -- see exposure ledger note"),
    ]

    registry_result = record_experiment_registry(canonical_conn, registry_values)
    results_result = record_experiment_results(
        canonical_conn, EXPERIMENT_ID, results_rows, schema_version=EXPECTED_SCHEMA_VERSION,
        provenance=provenance, created_ms=now_ms)
    canonical_conn.commit()

    receipt_hash = gates.issue_gate_receipt(
        knowledge_conn, experiment_id=EXPERIMENT_ID, canonical_family_id=FAMILY_ID,
        split_version=SPLIT_VERSION, nullifier=nullifier, registry_result="EXECUTED")
    knowledge_conn.commit()

    return {
        "verification": verification, "verdict": verdict, "verdict_reason": verdict_reason,
        "coef": coef, "se": se_coef, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p_value,
        "test_n": test_design["n"], "test_n_dropped": test_design["n_dropped"],
        "test_design_rank": test_rank, "train_design_rank": train_rank,
        "train_n": train_design["n"], "train_n_dropped": train_design["n_dropped"],
        "vif": vif, "collinearity_drops": drops,
        "consume_result": consume_result, "nullifier": nullifier,
        "registry_result": registry_result, "results_result": results_result,
        "receipt_hash": receipt_hash, "registry_values": registry_values,
        "secondary_fit": secondary_fit, "train_fit": train_fit, "test_fit": test_fit,
        "predictor_scale_stats": predictor_scale_stats,
    }
