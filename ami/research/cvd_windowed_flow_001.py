"""G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1.

Executes the frozen preregistration `E-CVD-PRIMARY-LONG-W300-PREREG-001`
(reports/research/s34/S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md, committed
as 749520b3) exactly once. This module is the ONLY code that reads the TEST
population's `endpoint_return_bps`/`mfe_bps` values for this experiment, and
it may only do so after `epistemic_gates.consume_test_evidence()` has marked
the frozen TEST-evidence nullifier consumed for this exact experiment_id.

Every identity constant below (FAMILY_ID/SPLIT_VERSION/EXPECTED_*_HASH/
EXPECTED_NULLIFIER) is copied verbatim from the committed preregistration
artifacts -- not re-derived from a `frozen_splits` free-text description
(the exact string hashed into SPLIT_VERSION at preregistration time was
never recorded verbatim in any committed file; re-deriving it from prose
would risk silently producing a DIFFERENT split_version/nullifier than the
one already issued). `verify_pre_execution()` independently reproduces the
population/split/family/nullifier from the real database and raises
`ProtocolInvalidation` if any of it drifts from these frozen constants --
this is the fail-closed check the frozen preregistration itself requires
before any TEST access is permitted.

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
# Frozen identity (copied verbatim from the committed preregistration; never
# re-derived from prose -- see module docstring)
# ---------------------------------------------------------------------------
RESEARCH_CONTEXT_ID = "cvd-windowed-flow-001-execution"
SYMBOL = "ETHUSDT"
HORIZON = "swing_24h"

EXPERIMENT_ID = "E-CVD-PRIMARY-LONG-W300-PREREG-001"
QUESTION_IDS = "FAM_CVD_PRIMARY_LONG_REVERSAL"
HYPOTHESIS_ID = "H-CVD-PRIMARY-LONG-W300-EXACT-NET-TAKER-FLOW-NOTIONAL-V1"
FAMILY_ID = "FAMv1:bec99d8d36f7d6a1"
SPLIT_VERSION = "SPLITv1:0a1b96fd74dd281e"
SPEC_HASH = "a2fd9e5b08ed2a716ac0c1cae0658740f24b48024d5b7524eb843e4441940b57"
EXPECTED_TRAIN_HASH = "61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5"
EXPECTED_TEST_HASH = "98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04"
EXPECTED_NULLIFIER = "085397f31c199c1d0c1d5ce647af4d1aa311166c63199f92872e089db8e72a7a"
PREREG_GATE_RECEIPT_HASH = "d46f7e2c6b3621215e13eed136f1e22aec2531549769c96da694a407855f7e5c"
PREREG_CODE_COMMIT = "09104298"

TRAIN_FRACTION = 0.7
MIN_BUCKET_N = 20
CVD_SCALE = 1_000_000.0
EVENT_SCALE = 100_000.0
RELEVANCE_FLOOR_PER_10M_BPS = 5.0
CORRECTED_PATH_VERSION = "path-v2-candle-repair-r1"
COLLINEARITY_VIF_THRESHOLD = 10.0

DESIGN_NAMES = ("const", "cvd_notional_w300_per_1m", "event_notional_per_100k",
                "session_EUROPE", "session_US", "session_OFF", "day_trend_bps")


class ProtocolInvalidation(Exception):
    """Raised when the real database no longer reproduces the frozen
    preregistration's identity/population/split/nullifier/gate state.
    Per the governed-execution protocol, this must stop BEFORE any TEST
    outcome access -- never repaired or improvised after seeing TEST data."""


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

    w300 = canonical_conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN quality_status='EXACT_RECONSTRUCTABLE' THEN 1 ELSE 0 END) "
        "FROM ami_cvd_window_quality_v1 WHERE window_id='W300'").fetchone()
    if w300[0] != 324 or w300[1] != 324:
        errors.append(f"W300 quality-population drift: total={w300[0]} exact={w300[1]}")

    violations = canonical_conn.execute(
        "SELECT COUNT(*) FROM ami_cvd_windowed_flow WHERE window_id='W300' AND "
        "(window_end_ts_ms > signal_birth_ts OR known_at_classification != 'KNOWN_AT_SAFE')"
    ).fetchone()[0]
    if violations != 0:
        errors.append(f"known-at violations for W300: {violations}")

    bucket = dict(canonical_conn.execute(
        "SELECT direction, COUNT(*) FROM ami_cvd_bucket_exclusions GROUP BY direction").fetchall())
    if bucket != {"SHORT": 104}:
        errors.append(f"bucket-exclusion population drift: {bucket}")

    reg_count = canonical_conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0]
    res_count = canonical_conn.execute("SELECT COUNT(*) FROM experiment_results").fetchone()[0]
    already_has_results = canonical_conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)).fetchone()[0]

    return {
        "pop": pop, "family_id": family_id, "nullifier": nullifier,
        "is_rerun_of_self": is_rerun_of_self, "receipt": receipt, "errors": errors,
        "experiment_registry_count_before": reg_count, "experiment_results_count_before": res_count,
        "already_has_results_before": already_has_results,
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
    cvd_rows = conn.execute(
        f"SELECT signal_id, cvd_notional FROM ami_cvd_windowed_flow "
        f"WHERE window_id='W300' AND signal_id IN ({placeholders})", signal_ids).fetchall()
    cvd_by_signal = dict(cvd_rows)

    out = []
    for cyc, s in reps:
        sid = s["signal_id"]
        cvd_notional = cvd_by_signal.get(sid)
        event_notional = event_notional_by_id.get(s["source_event_id"])
        hour = dt.datetime.fromtimestamp(s["signal_birth_ts"] / 1000, dt.timezone.utc).hour
        session = _session_of_hour(hour)
        day_trend = compute_day_trend_bps(engine, SYMBOL, s["signal_birth_ts"])
        missing = cvd_notional is None or event_notional is None or day_trend is None
        out.append({
            "cycle_key": cyc, "signal_id": sid,
            "cvd_notional": cvd_notional, "event_notional": event_notional,
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
    """rows: dicts with cvd_notional/event_notional/session/day_trend_bps/
    <outcome_key>/missing_predictor/missing_outcome (outcome missingness is
    the caller's responsibility to flag). Applies listwise deletion."""
    total = len(rows)
    usable = [r for r in rows if not r.get("missing_predictor") and r.get(outcome_key) is not None]
    dropped = total - len(usable)
    n = len(usable)
    X = np.zeros((n, len(DESIGN_NAMES)))
    y = np.zeros(n)
    cluster_ids = []
    for i, r in enumerate(usable):
        X[i, 0] = 1.0
        X[i, 1] = r["cvd_notional"] / CVD_SCALE
        X[i, 2] = r["event_notional"] / EVENT_SCALE
        X[i, 3] = 1.0 if r["session"] == "EUROPE" else 0.0
        X[i, 4] = 1.0 if r["session"] == "US" else 0.0
        X[i, 5] = 1.0 if r["session"] == "OFF" else 0.0
        X[i, 6] = r["day_trend_bps"]
        y[i] = r[outcome_key]
        cluster_ids.append(r["cycle_key"])
    return {"X": X, "y": y, "cluster_ids": cluster_ids, "n": n, "n_total": total, "n_dropped": dropped}


def run_cluster_robust_ols(X: np.ndarray, y: np.ndarray, cluster_ids: list[str]) -> dict:
    """OLS with the CR1 cluster-robust sandwich variance estimator
    (Cameron-Miller convention: V = c * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1,
    c = (N-1)/(N-k) * G/(G-1)), inference via a t-distribution with G-1 df.

    Uses the Moore-Penrose pseudo-inverse rather than a strict inverse:
    the real population has zero LONG signals in the EUROPE session in
    BOTH TRAIN (n=91) and TEST (n=40) -- discovered from TRAIN diagnostics
    before any TEST access, not a researcher choice -- making the
    session_EUROPE column identically zero and X'X exactly singular under
    a strict inverse. pinv leaves the frozen model formula's column set
    unchanged (no term is dropped or renamed) and gives the minimum-norm
    solution: session_EUROPE's own coefficient/SE come out as exactly 0
    (there is no information to estimate it), while the primary predictor
    (column 1, cvd_notional) and every other identified column are
    unaffected, since pinv reduces to the ordinary inverse whenever X'X is
    already full-rank."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
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
    # np.clip guards a column with zero variance in the population being fit
    # (e.g. session_EUROPE, entirely absent from both TRAIN and TEST here):
    # its own diagonal entry can come out as a tiny negative float from
    # floating-point noise under the pinv sandwich product, which is not a
    # real negative variance -- clip to 0 rather than emit NaN/a RuntimeWarning.
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    df = max(G - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # a zero-variance column (see note above) has se=0 for its own
        # coefficient -- division produces +/-inf or nan there by design,
        # never for the primary predictor column, which is always identified.
        t_stats = beta / se
        p_values = 2.0 * (1.0 - _stats.t.cdf(np.abs(t_stats), df))
    crit = _stats.t.ppf(0.975, df)
    ci_lo = beta - crit * se
    ci_hi = beta + crit * se

    return {"beta": beta, "se": se, "t_stats": t_stats, "p_values": p_values,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "df": df, "n": n, "k": k, "G": G, "resid": resid}


def compute_vif(X: np.ndarray) -> dict:
    """VIF_j = 1/(1-R_j^2), R_j^2 from regressing predictor/control column j
    (excluding the intercept) on every other column (including intercept).
    A column with zero variance (ss_tot=0 -- e.g. a session dummy for a
    level entirely absent from the population, a real, TRAIN-discovered
    data condition, not a collinearity problem) has no defined VIF; it is
    reported as `None` rather than raising a ZeroDivisionError, and the
    collinearity drop policy never acts on it (only on a genuinely
    high-VIF day_trend_bps/event_notional column)."""
    n, k = X.shape
    vifs = {}
    for j in range(1, k):
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
    session dummies are never dropped. Returns the list of column names
    (from DESIGN_NAMES) to drop, in the order checked."""
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
        return "INSUFFICIENT_SAMPLE_OR_INCONCLUSIVE", f"n_test={n_test} < MIN_BUCKET_N={MIN_BUCKET_N}"
    half_width = (ci_hi - ci_lo) / 2.0
    if half_width > 2.0 * RELEVANCE_FLOOR_PER_10M_BPS:
        return ("INSUFFICIENT_SAMPLE_OR_INCONCLUSIVE",
                f"CI half-width {half_width:.4f} exceeds 2x relevance floor "
                f"({2 * RELEVANCE_FLOOR_PER_10M_BPS})")
    ci_excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)
    magnitude_per_10m = abs(coef * 10.0)
    if ci_excludes_zero and p_value < 0.05 and magnitude_per_10m >= RELEVANCE_FLOOR_PER_10M_BPS:
        return "EVIDENCE_SUPPORTS_INCREMENTAL_ASSOCIATION", (
            f"CI excludes 0, p={p_value:.4g}<0.05, |coef*10|={magnitude_per_10m:.4f}"
            f">={RELEVANCE_FLOOR_PER_10M_BPS}"
        )
    return "NO_RELIABLE_ASSOCIATION", (
        f"ci_excludes_zero={ci_excludes_zero} p={p_value:.4g} |coef*10|={magnitude_per_10m:.4f}"
    )


# ---------------------------------------------------------------------------
# Step 4: the single governed execution entry point
# ---------------------------------------------------------------------------

def execute_governed_run(canonical_conn, knowledge_conn) -> dict:
    verification = verify_pre_execution(canonical_conn, knowledge_conn)
    if verification["errors"]:
        raise ProtocolInvalidation(
            "CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_BLOCKED: " + "; ".join(verification["errors"]))

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
    vif = compute_vif(train_design["X"]) if train_design["n"] > len(DESIGN_NAMES) else {}
    drops = apply_collinearity_policy(vif) if vif else []
    train_fit = run_cluster_robust_ols(train_design["X"], train_design["y"], train_design["cluster_ids"]) \
        if train_design["n"] > len(DESIGN_NAMES) else None

    predictor_scale_stats = {
        "n": train_design["n"],
        "min": float(np.min(train_design["X"][:, 1])) * CVD_SCALE if train_design["n"] else None,
        "max": float(np.max(train_design["X"][:, 1])) * CVD_SCALE if train_design["n"] else None,
        "mean": float(np.mean(train_design["X"][:, 1])) * CVD_SCALE if train_design["n"] else None,
        "median": float(np.median(train_design["X"][:, 1])) * CVD_SCALE if train_design["n"] else None,
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
    test_fit = run_cluster_robust_ols(test_design["X"], test_design["y"], test_design["cluster_ids"])

    coef = float(test_fit["beta"][1])
    se_coef = float(test_fit["se"][1])
    ci_lo = float(test_fit["ci_lo"][1])
    ci_hi = float(test_fit["ci_hi"][1])
    p_value = float(test_fit["p_values"][1])
    verdict, verdict_reason = apply_verdict_rule(test_design["n"], coef, se_coef, ci_lo, ci_hi, p_value)

    # secondary, non-promotable check: same model with mfe_bps as outcome
    test_design_mfe = build_design(test_features, "mfe_bps")
    secondary_fit = run_cluster_robust_ols(test_design_mfe["X"], test_design_mfe["y"],
                                            test_design_mfe["cluster_ids"]) if test_design_mfe["n"] > len(DESIGN_NAMES) else None

    now_ms = int(time.time() * 1000)
    frozen_population = (
        f"ami_signal_lifecycle direction=LONG (long_n={pop['long_n']}), CVD W300 "
        f"quality_status=EXACT_RECONSTRUCTABLE (324/324, 0 SOURCE_GAPPED), swing_24h "
        f"observation_status=OK (eligible_long_n={pop['eligible_long_n']}), one representative "
        f"per independent_cycle_id (earliest signal_birth_ts; representative_cycle_n="
        f"{pop['representative_cycle_n']}); test n_used={test_design['n']} "
        f"(n_dropped_missing={test_design['n_dropped']})"
    )
    frozen_features = (
        "primary=ami_cvd_windowed_flow.cvd_notional WHERE window_id='W300', continuous, "
        "scaled /1,000,000; controls=event_notional(/100,000), session(dummy, ref=ASIA), "
        "day_trend_bps(raw)"
    )
    frozen_target = "endpoint_return_bps@swing_24h (ami_lifecycle_path_observations, effective selection)"
    frozen_thresholds = (
        f"significance p<0.05 two-sided; relevance floor |coef*10|>={RELEVANCE_FLOOR_PER_10M_BPS}bps"
        f"/$10M; MIN_BUCKET_N={MIN_BUCKET_N}; CI half-width invalidation > "
        f"{2 * RELEVANCE_FLOOR_PER_10M_BPS}bps"
    )
    frozen_splits = (
        f"FROZEN AT PREREGISTRATION (SPLIT_VERSION={SPLIT_VERSION}, reproduced not re-derived): "
        f"cycle-grouped chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} split "
        f"by earliest signal_birth_ts per independent_cycle_id, cut by cycle count "
        f"(train={len(pop['train_reps'])}, test={len(pop['test_reps'])})"
    )
    # Content-stable across an idempotent rerun (record_experiment_registry's
    # immutability check compares this field byte-for-byte): must NOT embed
    # `consume_result`, which is legitimately "CONSUMED" on the first call
    # and "NOOP_IDENTICAL" on every rerun -- that is bookkeeping about THIS
    # CALL, not about the experiment's frozen scientific content, and is
    # already reported separately in `results_rows` (nullifier_consume_result).
    provenance = (
        f"G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1; preregistration_commit={PREREG_CODE_COMMIT}; "
        f"spec_hash={SPEC_HASH}; test_nullifier={nullifier}; "
        f"gate_receipt_hash_at_preregistration={PREREG_GATE_RECEIPT_HASH}; collinearity_drops={drops}"
    )

    registry_values = {
        "experiment_id": EXPERIMENT_ID, "question_ids": QUESTION_IDS, "hypothesis_id": HYPOTHESIS_ID,
        "preregistered_at": "2026-07-06T00:00:00Z (commit 749520b3)",
        "frozen_population": frozen_population, "frozen_features": frozen_features,
        "frozen_target": frozen_target, "frozen_thresholds": frozen_thresholds,
        "frozen_splits": frozen_splits, "frozen_economic_gate": "NONE (descriptive/inferential only)",
        "frozen_statistical_gate": (
            "OLS + cluster-robust(CR1) SE clustered by independent_cycle_id; two-sided p<0.05; "
            "95% CI; effect-size relevance floor; VIF collinearity policy"
        ),
        "code_commit": PREREG_CODE_COMMIT, "dataset_hash": nullifier,
        "started_at": now_ms, "completed_at": now_ms,
        "software_verdict": "COMPLETED", "scientific_verdict": verdict,
        "mutation_test_count": 0, "mutation_test_passed": 0,
        "supersedes_experiment_id": None, "report_artifact_id": None,
        "schema_version": 12, "provenance": provenance,
        "created_ms": now_ms, "updated_ms": now_ms,
    }

    results_rows = [
        ("primary_predictor_coefficient_bps_per_1m", str(coef)),
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
        ("train_n_used", str(train_design["n"])),
        ("train_n_dropped_missing", str(train_design["n_dropped"])),
        ("vif", str(vif)),
        ("collinearity_drops_applied", str(drops)),
        ("predictor_train_scale_stats_usd", str(predictor_scale_stats)),
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
    ]

    registry_result = record_experiment_registry(canonical_conn, registry_values)
    results_result = record_experiment_results(
        canonical_conn, EXPERIMENT_ID, results_rows, schema_version=12,
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
        "train_n": train_design["n"], "train_n_dropped": train_design["n_dropped"],
        "vif": vif, "collinearity_drops": drops,
        "consume_result": consume_result, "nullifier": nullifier,
        "registry_result": registry_result, "results_result": results_result,
        "receipt_hash": receipt_hash, "registry_values": registry_values,
        "secondary_fit": secondary_fit, "train_fit": train_fit, "test_fit": test_fit,
        "predictor_scale_stats": predictor_scale_stats,
    }
