"""BATCH-P6-010 (W7A): Historical State/Structure Aging & Market Clocks
(HISTORICAL_RESEARCH_WAVES.md W7, preconditions P3 identity + P5 shared
engines -- both already satisfied, no new motor required).

Preregistration frozen BEFORE any outcome is looked at (this docstring +
the module-level constants below ARE the freeze; freeze_and_record() writes
the same frozen values to experiment_registry so they cannot silently drift).
Operator approved the initial scope, then required 5 corrections (all
applied below, marked "[CORRECTION n]"); nothing else was changed after
that approval.

NOT A FULL SIGNAL-LIFECYCLE CLAIM (OD-016, backlog): whitepaper v0.3
Section 61's `signal_birth_ts`/`first_executable_ts`/`time_since_last_
confirmation` fields and `book_update_age` require a forward-observer
concept (Phase 8, not built) and/or an order-book-continuity stream that
does not exist for historical replay. Neither is computed here.

POPULATION (frozen, reused verbatim from W1/W4/W5a/W6 -- not redefined):
ETHUSDT REAL_LIQUIDATION anchors via feature_gateway.fetch_events,
raw anchor_n=252, independent_cycle_n via event_cycle_membership
(cycle_definition_version=canonical-v1, is_canonical=1).

OUTCOME (frozen, REUSED from W4, single representative horizon like
W4's C2-C4 and W6's E1-E3): swing_24h path class (CONTINUATION/REVERSAL/
CHOP, E-W4-POST-EVENT-PATH-TAXONOMY-001's fixed +-20bps band).

[CORRECTION 4] EXACT MATURITY CUTOFF (frozen, identical pattern to
E-W6RS-CONFIRMATION-001): MATURITY_CUTOFF_TS_MS = MAX(ami_candles.close_ts_ms)
at freeze time. Anchors where anchor_ts_ms + 24h > MATURITY_CUTOFF_TS_MS are
EXCLUDED from this experiment (their 24h outcome has not happened yet --
excluded_no_horizon_data, counted and reported, never fabricated). If more
anchors mature later (more wall-clock time passes), that is NOT retro-
computed into this experiment's frozen numbers -- a later re-run must be
recorded as a separate, explicitly-labeled append-only follow-up experiment
(e.g. E-W7A-...-FOLLOWUP-001), never an in-place mutation of this one.

===========================================================================
1. STATE AGE (frozen)
===========================================================================
Evaluation: ami.states.engine.StateEngine._structure("ETHUSDT", ts, "1h")
-- the SAME call W6 uses for compression_at_anchor, same TF, not redefined.
Cadence: STATE_AGE_CADENCE_MS = 3,600,000 (1h, matches StructurePhase's own
bar). Algorithm: current_phase = _structure(anchor_ts). Walk backward in
fixed 1h steps; the age is anchor_ts - (last probed ts whose phase still
equalled current_phase), i.e. resolution-bounded to +-1h by the cadence
(documented, not fabricated to sub-hour precision).
Maximum lookback: STATE_AGE_MAX_STEPS = 168 (7 days). Rationale: each
backward step evaluates _structure() (~20 indexed point-lookups against
mark_prices) plus one staleness probe; 168 steps x (252+100 anchors) is
already a multi-minute batch on this single-threaded, sequential-only
machine -- 30 days would be ~4x that. If the true dwell time exceeds 7
days the exact magnitude is not needed for a HIGH/LOW conditioning test,
only that it is "old" -- see LEFT_CENSORED_AT_CAP below.
Censoring taxonomy (never a fabricated 0 or estimate):
  OK                    -- transition found within the cap; age_ms known
                           to +-1h.
  LEFT_CENSORED          -- backward walk reached the true start of
                           mark_prices (ETHUSDT) before finding a
                           transition; true age is older than the data
                           horizon, exact value unknowable.
  LEFT_CENSORED_AT_CAP   -- backward walk reached STATE_AGE_MAX_STEPS
                           without finding a transition or the data start;
                           true age is >= 7 days, exact value not computed
                           (compute-bound cap, distinct from a genuine data
                           boundary).
  MISSING                -- a backward probe's nearest mark_prices row is
                           more than STATE_AGE_STALENESS_GAP_MS (2h) stale
                           (a genuine mid-series collector gap); the walk
                           stops there rather than silently evaluating
                           _structure() on a stale/fabricated price.

===========================================================================
2. STRUCTURAL OBJECT AGE -- 3 SEPARATE features (frozen, never merged)
===========================================================================
swing_age, level_age and push_age are three INDEPENDENT features/tests.
They are never combined via min/mean/nearest-selected-across-types -- each
is its own nearest-PRIOR-object-of-that-type age (a within-type nearest
selection is the only way to define "age" for a single type, but no
cross-type merge happens anywhere in this module).
Eligibility (frozen, identical to W4's near_swing_low/near_level/
recent_down_push point-in-time-safety rule): only ami_swings/ami_levels/
ami_pushes rows with known_at_ts <= anchor_ts_ms are eligible.
Object pool: symbol=ETHUSDT, timeframe='1m' (existing default build), ALL
swing_type/level_type/direction values pooled within each table (no further
sub-typing -- that would be a 4th/5th/6th feature, not asked for).
swing_age  = anchor_ts_ms - MAX(known_at_ts) over eligible ami_swings.
level_age  = anchor_ts_ms - MAX(known_at_ts) over eligible ami_levels
             (via feature_gateway.fetch_level_features's safe-column
             allowlist -- F-B2-blocked columns never requested).
push_age   = anchor_ts_ms - MAX(known_at_ts) over eligible ami_pushes.
If no eligible object of a given type exists -> MISSING for that feature
(never a fabricated 0/estimate).

===========================================================================
3. MARKET CLOCKS -- 3 SEPARATE features (frozen)
===========================================================================
Metrics (all computed from ami_candles, timeframe='1m', already-existing
trade_count/volume/close columns -- no new agg_trades scan):
  trade_count  = SUM(candle.trade_count) over a window
  volume       = SUM(candle.volume) over a window
  realized_vol = sqrt(sum(r^2)/59) over 59 consecutive close-to-close
                 log-returns from the window's 60 candles
Window definition: CLOCK_WINDOW_CANDLES = 60 (exactly 1h at 1m
granularity), the 60 candles at-or-before the reference ts (known-at-safe,
same "at-or-before" convention as W4's _CandleIndex).
[CORRECTION 2] Completeness gate (frozen, applied identically to the
anchor window and every baseline window -- single validity definition
shared by all 3 metrics, since all 3 are computed from the same 60-candle
block):
  - a window is VALID only if all 60/60 of its candles have
    data_quality == 'AVAILABLE' (never GAPPED);
  - the anchor's own measurement window must be 60/60 valid, else
    MISSING for all 3 clocks at that anchor (collector/data gap);
  - the baseline (below) needs >= 24/30 valid windows, else MISSING;
  - if a baseline median resolves to 0 -> MISSING (avoid divide-by-zero
    or a degenerate ratio).
Baseline/normalization (frozen, NOT swept): the CLOCK_BASELINE_WINDOWS=30
non-overlapping 60-candle blocks immediately preceding the anchor's
measurement window (known-at-safe: strictly before the anchor). For each
metric, baseline_median = MEDIAN over the valid baseline windows' values
(a single fixed statistic, not a search over candidate windows/percentiles
-- "median" is the one frozen quantile, never re-selected).
clock_ratio = anchor_value / baseline_median.
Bucket (frozen, natural zero-point in ratio-space, same discipline as W6's
RS sign-split -- NOT the TRAIN-median rule used for age features below):
  HIGH if clock_ratio > 1.0, else LOW. 1.0 is not searched or fit.
3 tests: trade_count_clock, volume_clock, realized_vol_clock.

===========================================================================
4. LIQUIDATION AGE -- 2 SEPARATE features (frozen)
===========================================================================
Direction: liquidations.side in {SELL, BUY} (existing column). Population
is 100% SELL-cascade anchors, so same_direction=SELL, opposite_direction=
BUY -- 2 independent tests, never pooled into one "generic age".
"Significant print" threshold: LIQ_NOTIONAL_THRESHOLD = 200,000 -- reused
VERBATIM from ami.states.engine.StateEngine._cascade()'s own ACCELERATION/
EARLY_CASCADE trigger; not a new fit.
[CORRECTION 3] Exclusion boundary -- PREFERRED canonical-membership method
IS available and used: ami_events.event_end_ts_ms (now exposed by
feature_gateway.fetch_events) marks where the anchor's OWN detected
cascade cluster ends. reference_ts = event_end_ts_ms when not NULL
(248/252 anchors); for the 4/252 anchors lacking event_end_ts_ms,
reference_ts falls back to anchor_ts_ms itself (still a real, recorded
boundary, not a fabricated one -- documented per-anchor via
liq_reference_source in the results). Because a true canonical boundary is
used, features are named generically (pre2h_* naming is NOT used --
that fallback only applies if canonical exclusion were unavailable, which
it is not here).
same_direction age      = reference_ts - MAX(ts_ms) over liquidations
                          WHERE side='SELL' AND notional>=200000 AND
                          ts_ms < reference_ts.
opposite_direction age  = same query with side='BUY'.
No qualifying prior print anywhere in the liquidations history before
reference_ts -> LEFT_CENSORED (never a fabricated 0/estimate). No
compute-bound cap is needed here (a single indexed MAX query, not a
stepwise scan) -- so there is no LEFT_CENSORED_AT_CAP for this feature.
For candidate-universe negative-control rows (not real cascades),
reference_ts = the slot's own ts_ms directly (no cascade to exclude).

===========================================================================
CONTINUOUS AGE -> BUCKET RULE (frozen; applies to the 5 duration features:
state_age, swing_age, level_age, push_age, same/opposite liquidation age)
===========================================================================
[CORRECTION 1] The HIGH/LOW cutpoint for each duration feature is the
MEDIAN of that feature's OK-status values within the chronological TRAIN
split ONLY (first TRAIN_FRACTION=0.7 of analyzed anchors by anchor_ts_ms --
same split as W4/W6/W6RS). This TRAIN median is frozen once computed and
then applied IDENTICALLY to bucket TEST anchors AND the candidate-universe
negative-control's values for that same feature. The full-population
median is NEVER used for bucketing (this was the operator's explicit
correction to the original proposal). HIGH = age_ms > train_median, LOW =
age_ms <= train_median. Anchors with a non-OK status for a given feature
are excluded from THAT feature's bucketed test (never imputed).

===========================================================================
MULTIPLE-TESTING FAMILY (frozen, 9 comparisons, FAM_SIGNAL_AGING_MARKET_
CLOCK -- an independent feature/contrast is what counts, not "4
components"; the operator's own framing):
    E1: state_age              vs swing_24h outcome
    E2: swing_age               vs swing_24h outcome
    E3: level_age                vs swing_24h outcome
    E4: push_age                 vs swing_24h outcome
    E5: trade_count_clock         vs swing_24h outcome
    E6: volume_clock               vs swing_24h outcome
    E7: realized_vol_clock          vs swing_24h outcome
    E8: liquidation_age_same_direction (SELL)   vs swing_24h outcome
    E9: liquidation_age_opposite_direction (BUY) vs swing_24h outcome
Chronological 70/30 stability and the candidate-universe negative control
apply to all 9 as robustness checks, NOT as additional family members
(same convention as W4 C1-C5 / W6 E1-E3).

===========================================================================
[CORRECTION 5] INFERENCE PLAN (frozen)
===========================================================================
PRIMARY, per test: independent-cycle-clustered effect size = risk
difference (REVERSAL rate in HIGH bucket - REVERSAL rate in LOW bucket),
with a cluster block-bootstrap 95% CI (resample whole canonical-v1 cycles
with replacement, n=2000, seed=20260704 -- same cluster-bootstrap
discipline as E-W6RS-CONFOUND-RESOLUTION-001; anchors with no resolved
cycle get their own singleton cluster, same "NOCYCLE-<event_id>"
convention).
SECONDARY, per test: a two-sided label-permutation p-value (n_perm=2000,
seed=20260704 -- two-sided because, unlike W6-RS's operator-specified
directional hypothesis, none of these 9 features carries a pre-committed
direction). The 9 raw p-values are then Holm-adjusted (step-down,
family-wise) -- reported alongside the raw values, never substituted for
them.
Negative control (candidate-universe, is_event_aligned=0): descriptive
only (n / reversal-rate per bucket using the SAME anchor-derived TRAIN
median), same lightweight treatment W4/W6/W6RS gave their own negative
controls -- not run through the full bootstrap/permutation machinery.
Sample capped at NEGATIVE_CONTROL_SAMPLE_SIZE=100 (state_age's per-row
cost dominates total runtime on this sequential-only machine; 100 is a
frozen, documented scope limit, not a silent shortcut).

STOP CONDITIONS: any HIGH/LOW bucket with N < MIN_BUCKET_N=20 is reported
INSUFFICIENT_SAMPLE for that specific test, never silently dropped/merged.

Descriptive/conditioning only: NO trade/PnL claim, NO bucket/route/
observer promotion or change anywhere in this module.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import random
import time
from collections import Counter

import numpy as np

from ami.research.feature_gateway import (
    fetch_chart_feature,
    fetch_events,
    fetch_level_features,
)
from ami.research.w4_post_event_path_taxonomy import (
    MIN_BUCKET_N,
    TRAIN_FRACTION,
    _CandleIndex,
    _split_chronological,
    classify_path,
    compute_path_returns,
)
from ami.states.engine import StateEngine
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W7A-STATE-STRUCTURE-AGING-MARKET-CLOCKS-001"
RESEARCH_CONTEXT_ID = "w7a-state-structure-aging-market-clocks"

STATE_AGE_CADENCE_MS = 3_600_000  # 1h, matches StructurePhase TF (W6 precedent)
STATE_AGE_MAX_STEPS = 168  # 7 days, compute-bounded cap (see docstring)
STATE_AGE_STALENESS_GAP_MS = 2 * STATE_AGE_CADENCE_MS  # 2h

LIQ_NOTIONAL_THRESHOLD = 200_000.0  # reused verbatim from StateEngine._cascade()

CLOCK_WINDOW_CANDLES = 60
CLOCK_BASELINE_WINDOWS = 30
CLOCK_BASELINE_MIN_VALID = 24

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260704
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 20260704
NEGATIVE_CONTROL_SEED = 20260704
NEGATIVE_CONTROL_SAMPLE_SIZE = 100

DURATION_FEATURES = ["state_age", "swing_age", "level_age", "push_age",
                      "liq_age_same_direction", "liq_age_opposite_direction"]
CLOCK_FEATURES = ["trade_count_clock", "volume_clock", "realized_vol_clock"]
ALL_FEATURES = DURATION_FEATURES + CLOCK_FEATURES  # 9, matches FAM_SIGNAL_AGING_MARKET_CLOCK


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _nearest_prior_ts(sorted_known_at: list[int], ts_ms: int) -> int | None:
    i = bisect.bisect_right(sorted_known_at, ts_ms)
    if i == 0:
        return None
    return sorted_known_at[i - 1]


class _CandleSeries:
    """Sorted 1m candle series (close_ts_ms/close/volume/trade_count/
    data_quality) for windowed market-clock aggregation via array index
    slicing (contiguous minute slots, one row per minute including
    GAPPED placeholders)."""

    def __init__(self, rows: list[dict]):
        ordered = sorted(rows, key=lambda c: c["close_ts_ms"])
        self._ts = [c["close_ts_ms"] for c in ordered]
        self._rows = ordered

    def index_at_or_before(self, ts_ms: int) -> int:
        return bisect.bisect_right(self._ts, ts_ms)

    def block(self, end_index: int, n: int) -> list[dict] | None:
        if end_index < n or end_index > len(self._rows):
            return None
        return self._rows[end_index - n:end_index]


def _window_metrics(window: list[dict]) -> dict | None:
    if any(c["data_quality"] != "AVAILABLE" for c in window):
        return None
    trade_count = sum(c["trade_count"] or 0 for c in window)
    volume = sum(c["volume"] or 0.0 for c in window)
    closes = [c["close"] for c in window]
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)
            if closes[i] > 0 and closes[i + 1] > 0]
    realized_vol = math.sqrt(sum(r * r for r in rets) / len(rets)) if rets else None
    return {"trade_count": trade_count, "volume": volume, "realized_vol": realized_vol}


def compute_state_age(engine: StateEngine, symbol: str, anchor_ts_ms: int, earliest_mark_ts_ms: int) -> dict:
    current_phase, _direction, _conf = engine._structure(symbol, anchor_ts_ms, "1h")
    last_same_ts = anchor_ts_ms
    for step in range(1, STATE_AGE_MAX_STEPS + 1):
        probe_ts = anchor_ts_ms - step * STATE_AGE_CADENCE_MS
        if probe_ts < earliest_mark_ts_ms:
            return {"status": "LEFT_CENSORED", "age_ms": None, "steps_scanned": step - 1}
        row = engine.conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (symbol, probe_ts),
        ).fetchone()
        found_ts = int(row[0]) if row else None
        if found_ts is None or (probe_ts - found_ts) > STATE_AGE_STALENESS_GAP_MS:
            return {"status": "MISSING", "age_ms": None, "steps_scanned": step - 1}
        probe_phase, _d, _c = engine._structure(symbol, probe_ts, "1h")
        if probe_phase != current_phase:
            return {"status": "OK", "age_ms": anchor_ts_ms - last_same_ts, "steps_scanned": step}
        last_same_ts = probe_ts
    return {"status": "LEFT_CENSORED_AT_CAP", "age_ms": None, "steps_scanned": STATE_AGE_MAX_STEPS}


def compute_structural_object_age(sorted_known_at: list[int], ts_ms: int) -> dict:
    prior = _nearest_prior_ts(sorted_known_at, ts_ms)
    if prior is None:
        return {"status": "MISSING", "age_ms": None}
    return {"status": "OK", "age_ms": ts_ms - prior}


def compute_market_clocks(series: _CandleSeries, ts_ms: int) -> dict:
    def _missing(n_baseline_valid: int = 0) -> dict:
        return {"status": "MISSING", "n_baseline_valid": n_baseline_valid,
                "trade_count_clock_ratio": None, "trade_count_clock_bucket": None,
                "volume_clock_ratio": None, "volume_clock_bucket": None,
                "realized_vol_clock_ratio": None, "realized_vol_clock_bucket": None}

    end_idx = series.index_at_or_before(ts_ms)
    anchor_window = series.block(end_idx, CLOCK_WINDOW_CANDLES)
    if anchor_window is None:
        return _missing()
    anchor_metrics = _window_metrics(anchor_window)
    if anchor_metrics is None or anchor_metrics["realized_vol"] is None:
        return _missing()

    anchor_start_idx = end_idx - CLOCK_WINDOW_CANDLES
    baseline_values: dict[str, list[float]] = {"trade_count": [], "volume": [], "realized_vol": []}
    for k in range(1, CLOCK_BASELINE_WINDOWS + 1):
        b_end = anchor_start_idx - CLOCK_WINDOW_CANDLES * (k - 1)
        b_block = series.block(b_end, CLOCK_WINDOW_CANDLES)
        if b_block is None:
            continue
        m = _window_metrics(b_block)
        if m is None or m["realized_vol"] is None:
            continue
        for key in baseline_values:
            baseline_values[key].append(m[key])

    n_valid = len(baseline_values["realized_vol"])
    if n_valid < CLOCK_BASELINE_MIN_VALID:
        return _missing(n_baseline_valid=n_valid)

    out = {"status": "OK", "n_baseline_valid": n_valid}
    for key, anchor_val in (("trade_count", anchor_metrics["trade_count"]),
                            ("volume", anchor_metrics["volume"]),
                            ("realized_vol", anchor_metrics["realized_vol"])):
        med = _median(baseline_values[key])
        if not med:
            out[f"{key}_clock_ratio"] = None
            out[f"{key}_clock_bucket"] = None
        else:
            ratio = anchor_val / med
            out[f"{key}_clock_ratio"] = round(ratio, 4)
            out[f"{key}_clock_bucket"] = "HIGH" if ratio > 1.0 else "LOW"
    return out


def compute_liquidation_age(conn_micro, side: str, reference_ts_ms: int) -> dict:
    row = conn_micro.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side=? AND notional>=? AND ts_ms<? "
        "ORDER BY ts_ms DESC LIMIT 1",
        (side, LIQ_NOTIONAL_THRESHOLD, reference_ts_ms),
    ).fetchone()
    if row is None:
        return {"status": "LEFT_CENSORED", "age_ms": None}
    return {"status": "OK", "age_ms": reference_ts_ms - int(row[0])}


def permutation_test_two_sided(high_n: int, high_succ: int, low_n: int, low_succ: int,
                                n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> dict:
    if high_n == 0 or low_n == 0:
        return {"observed_risk_difference": None, "p_value": None, "n_perm": n_perm}
    obs_diff = high_succ / high_n - low_succ / low_n
    pooled = [1] * (high_succ + low_succ) + [0] * ((high_n - high_succ) + (low_n - low_succ))
    rng = random.Random(seed)
    n_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        p_high = sum(pooled[:high_n]) / high_n
        p_low = sum(pooled[high_n:high_n + low_n]) / low_n
        if abs(p_high - p_low) >= abs(obs_diff):
            n_extreme += 1
    return {"observed_risk_difference": round(obs_diff, 4), "p_value": round(n_extreme / n_perm, 5), "n_perm": n_perm}


def cluster_bootstrap_risk_difference(rows: list[dict], bucket_field: str, cluster_field: str = "cluster_id",
                                       n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                                       label_high: str = "HIGH", label_low: str = "LOW") -> dict:
    """label_high/label_low let callers reuse this generic two-group cluster
    bootstrap with their own bucket vocabulary (e.g. W10a's CONFLICT/AGREEMENT)
    without duplicating the bootstrap loop -- default is W7A's own HIGH/LOW."""
    by_cluster: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_cluster.setdefault(r[cluster_field], []).append(i)
    clusters = list(by_cluster.keys())
    if not clusters:
        return {"n_valid_draws": 0, "ci95": (None, None)}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.choices(clusters, k=len(clusters))
        idx = [i for c in chosen for i in by_cluster[c]]
        sample = [rows[i] for i in idx]
        high = [r for r in sample if r[bucket_field] == label_high]
        low = [r for r in sample if r[bucket_field] == label_low]
        if not high or not low:
            continue
        rd = (sum(1 for r in high if r["path_class"] == "REVERSAL") / len(high)
              - sum(1 for r in low if r["path_class"] == "REVERSAL") / len(low))
        draws.append(rd)
    if not draws:
        return {"n_valid_draws": 0, "ci95": (None, None)}
    arr = np.array(draws)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"n_valid_draws": len(draws), "ci95": (round(float(lo), 4), round(float(hi), 4))}


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down, family-wise. None values (undefined tests) pass through untouched."""
    idx_valid = [i for i, p in enumerate(p_values) if p is not None]
    n = len(idx_valid)
    out: list[float | None] = list(p_values)
    if n == 0:
        return out
    ranked = sorted(idx_valid, key=lambda i: p_values[i])
    running_max = 0.0
    for rank, i in enumerate(ranked):
        val = min(1.0, (n - rank) * p_values[i])
        running_max = max(running_max, val)
        out[i] = round(running_max, 5)
    return out


def classify_closure(tests: dict) -> dict:
    """CHECKPOINT CLOSURE (operator instruction, 2026-07-04): the two tests
    with a nominal (pre-Holm) p<0.05 -- liq_age_same_direction (p=0.0435) and
    realized_vol_clock (p=0.0295) -- are NOT chased with a post-hoc
    threshold/subgroup search. Once Holm-adjusted they are not family-wise
    significant, so they are recorded as UNCONFIRMED_DESCRIPTIVE_LEAD (real,
    logged, but not pursued further in this checkpoint) rather than either
    promoted or silently discarded. Every other test is NULL. A future
    HOLM_SIGNIFICANT branch is defined for completeness/generality (not
    triggered by this run's actual numbers)."""
    out = {}
    for feat, t in tests.items():
        perm = t["permutation"]
        p_raw, p_holm = perm["p_value"], perm.get("p_value_holm_adjusted")
        if p_raw is None:
            out[feat] = "UNDEFINED"
        elif p_holm is not None and p_holm < 0.05:
            out[feat] = "HOLM_SIGNIFICANT"
        elif p_raw < 0.05:
            out[feat] = "UNCONFIRMED_DESCRIPTIVE_LEAD"
        else:
            out[feat] = "NULL"
    return out


def _bucket_test(rows: list[dict], bucket_field: str) -> dict:
    valid = [r for r in rows if r.get(bucket_field) in ("HIGH", "LOW") and r.get("path_class")]
    high = [r for r in valid if r[bucket_field] == "HIGH"]
    low = [r for r in valid if r[bucket_field] == "LOW"]
    high_n = len(high)
    high_succ = sum(1 for r in high if r["path_class"] == "REVERSAL")
    low_n = len(low)
    low_succ = sum(1 for r in low if r["path_class"] == "REVERSAL")
    perm = permutation_test_two_sided(high_n, high_succ, low_n, low_succ)
    boot = cluster_bootstrap_risk_difference(valid, bucket_field) if valid else {"n_valid_draws": 0, "ci95": (None, None)}
    return {
        "high": {"n": high_n, "n_reversal": high_succ,
                 "reversal_rate": round(high_succ / high_n, 4) if high_n else None,
                 "insufficient_sample": high_n < MIN_BUCKET_N},
        "low": {"n": low_n, "n_reversal": low_succ,
                "reversal_rate": round(low_succ / low_n, 4) if low_n else None,
                "insufficient_sample": low_n < MIN_BUCKET_N},
        "permutation": perm,
        "bootstrap_ci95": boot,
        "independent_cycle_n": len({r["cluster_id"] for r in valid}),
    }


def _compute_row_features(engine: StateEngine, ts_ms: int, earliest_mark_ts_ms: int,
                           swing_known_at: list[int], level_known_at: list[int], push_known_at: list[int],
                           candle_series: _CandleSeries, liq_reference_ts_ms: int) -> dict:
    out: dict = {}
    st = compute_state_age(engine, "ETHUSDT", ts_ms, earliest_mark_ts_ms)
    out["state_age_status"], out["state_age_age_ms"] = st["status"], st["age_ms"]

    sw = compute_structural_object_age(swing_known_at, ts_ms)
    out["swing_age_status"], out["swing_age_age_ms"] = sw["status"], sw["age_ms"]
    lv = compute_structural_object_age(level_known_at, ts_ms)
    out["level_age_status"], out["level_age_age_ms"] = lv["status"], lv["age_ms"]
    ph = compute_structural_object_age(push_known_at, ts_ms)
    out["push_age_status"], out["push_age_age_ms"] = ph["status"], ph["age_ms"]

    liq_s = compute_liquidation_age(engine.conn, "SELL", liq_reference_ts_ms)
    out["liq_age_same_direction_status"], out["liq_age_same_direction_age_ms"] = liq_s["status"], liq_s["age_ms"]
    liq_b = compute_liquidation_age(engine.conn, "BUY", liq_reference_ts_ms)
    out["liq_age_opposite_direction_status"], out["liq_age_opposite_direction_age_ms"] = liq_b["status"], liq_b["age_ms"]

    clocks = compute_market_clocks(candle_series, ts_ms)
    out["clocks_status"] = clocks["status"]
    for cf in CLOCK_FEATURES:
        base = cf.replace("_clock", "")
        out[f"{cf}_ratio"] = clocks.get(f"{base}_clock_ratio")
        out[f"{cf}_bucket"] = clocks.get(f"{base}_clock_bucket")
    return out


def _fetch_known_at_lists(conn, symbol: str) -> tuple[list[int], list[int], list[int]]:
    swing_rows = fetch_chart_feature(conn, RESEARCH_CONTEXT_ID, "ami_swings", ["known_at_ts"],
                                      symbol=symbol, equals={"timeframe": "1m"})
    push_rows = fetch_chart_feature(conn, RESEARCH_CONTEXT_ID, "ami_pushes", ["known_at_ts"],
                                     symbol=symbol, equals={"timeframe": "1m"})
    level_rows = fetch_level_features(conn, RESEARCH_CONTEXT_ID, ["known_at_ts", "timeframe"], symbol=symbol)
    level_known_at = sorted(r["known_at_ts"] for r in level_rows if r["timeframe"] == "1m")
    swing_known_at = sorted(r["known_at_ts"] for r in swing_rows)
    push_known_at = sorted(r["known_at_ts"] for r in push_rows)
    return swing_known_at, level_known_at, push_known_at


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles",
        ["close_ts_ms", "close", "volume", "trade_count", "data_quality"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex([{"close_ts_ms": r["close_ts_ms"], "close": r["close"]} for r in candle_rows])
    candle_series = _CandleSeries(candle_rows)
    maturity_cutoff_ts_ms = max(r["close_ts_ms"] for r in candle_rows)

    membership_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "event_cycle_membership", ["event_id", "candidate_cycle_key"],
        equals={"cycle_definition_version": "canonical-v1", "is_canonical": 1},
    )
    event_to_cycle = {r["event_id"]: r["candidate_cycle_key"] for r in membership_rows}

    swing_known_at, level_known_at, push_known_at = _fetch_known_at_lists(conn, symbol)

    engine = StateEngine()
    try:
        earliest_mark_ts_ms = engine.conn.execute(
            "SELECT MIN(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)
        ).fetchone()[0]

        per_anchor = []
        excluded_no_horizon_data = 0
        for e in events:
            anchor_ts = e["anchor_ts_ms"]
            if anchor_ts + 24 * 3600_000 > maturity_cutoff_ts_ms:
                excluded_no_horizon_data += 1
                continue
            returns = compute_path_returns(candle_index, anchor_ts)
            if returns["swing_24h"] is None:
                excluded_no_horizon_data += 1
                continue
            path_class = classify_path(returns["swing_24h"])

            liq_reference_ts = e["event_end_ts_ms"] if e["event_end_ts_ms"] is not None else anchor_ts
            liq_reference_source = "event_end_ts_ms" if e["event_end_ts_ms"] is not None else "anchor_ts_ms_fallback"

            row = {
                "event_id": e["event_id"],
                "anchor_ts_ms": anchor_ts,
                "path_class": path_class,
                "cluster_id": event_to_cycle.get(e["event_id"], f"NOCYCLE-{e['event_id']}"),
                "liq_reference_source": liq_reference_source,
            }
            row.update(_compute_row_features(
                engine, anchor_ts, earliest_mark_ts_ms, swing_known_at, level_known_at, push_known_at,
                candle_series, liq_reference_ts,
            ))
            per_anchor.append(row)
    finally:
        engine.conn.close()

    total_anchor_n = len(events)
    analyzed_n = len(per_anchor)
    independent_cycle_n = len({a["cluster_id"] for a in per_anchor if not a["cluster_id"].startswith("NOCYCLE-")})

    train_anchors, test_anchors = _split_chronological(per_anchor)

    # [CORRECTION 1] TRAIN-only median cutpoints for the 6 duration features.
    train_medians: dict[str, float | None] = {}
    for feat in DURATION_FEATURES:
        vals = [a[f"{feat}_age_ms"] for a in train_anchors if a[f"{feat}_status"] == "OK"]
        train_medians[feat] = _median(vals)

    def _apply_duration_buckets(rows: list[dict]) -> None:
        for a in rows:
            for feat in DURATION_FEATURES:
                med = train_medians[feat]
                if med is None or a[f"{feat}_status"] != "OK":
                    a[f"{feat}_bucket"] = None
                else:
                    a[f"{feat}_bucket"] = "HIGH" if a[f"{feat}_age_ms"] > med else "LOW"

    _apply_duration_buckets(per_anchor)

    tests = {}
    for feat in ALL_FEATURES:
        tests[feat] = _bucket_test(per_anchor, f"{feat}_bucket")

    p_values = [tests[f]["permutation"]["p_value"] for f in ALL_FEATURES]
    holm = holm_adjust(p_values)
    for f, adj in zip(ALL_FEATURES, holm):
        tests[f]["permutation"]["p_value_holm_adjusted"] = adj

    stability = {
        "train_n": len(train_anchors), "test_n": len(test_anchors),
        "train_medians": {k: (round(v, 1) if v is not None else None) for k, v in train_medians.items()},
    }

    return {
        "per_anchor": per_anchor,
        "candle_index": candle_index,
        "candle_series": candle_series,
        "swing_known_at": swing_known_at,
        "level_known_at": level_known_at,
        "push_known_at": push_known_at,
        "earliest_mark_ts_ms": earliest_mark_ts_ms,
        "maturity_cutoff_ts_ms": maturity_cutoff_ts_ms,
        "total_anchor_n": total_anchor_n,
        "analyzed_n": analyzed_n,
        "excluded_no_horizon_data": excluded_no_horizon_data,
        "independent_cycle_n": independent_cycle_n,
        "train_medians": train_medians,
        "stability": stability,
        "tests": tests,
    }


def compute_negative_control(conn, metrics: dict, symbol: str = "ETHUSDT",
                              n_target: int = NEGATIVE_CONTROL_SAMPLE_SIZE) -> dict:
    """Descriptive-only robustness check: same 9 features on a deterministic
    random sample of non-event-aligned candidate-universe slots, bucketed
    with the SAME anchor-derived TRAIN medians/ratio=1.0 rule (never a
    separately-fit control cutpoint)."""
    rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe", ["slot_ts_ms"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    rng = random.Random(NEGATIVE_CONTROL_SEED)
    sample = rng.sample(rows, min(n_target, len(rows)))

    engine = StateEngine()
    try:
        control_rows = []
        for r in sample:
            ts = r["slot_ts_ms"]
            if ts + 24 * 3600_000 > metrics["maturity_cutoff_ts_ms"]:
                continue
            returns = compute_path_returns(metrics["candle_index"], ts)
            if returns["swing_24h"] is None:
                continue
            path_class = classify_path(returns["swing_24h"])
            feats = _compute_row_features(
                engine, ts, metrics["earliest_mark_ts_ms"],
                metrics["swing_known_at"], metrics["level_known_at"], metrics["push_known_at"],
                metrics["candle_series"], ts,
            )
            row = {"path_class": path_class, **feats}
            for feat in DURATION_FEATURES:
                med = metrics["train_medians"][feat]
                if med is None or row[f"{feat}_status"] != "OK":
                    row[f"{feat}_bucket"] = None
                else:
                    row[f"{feat}_bucket"] = "HIGH" if row[f"{feat}_age_ms"] > med else "LOW"
            control_rows.append(row)
    finally:
        engine.conn.close()

    def _descriptive(rows_: list[dict], bucket_field: str) -> dict:
        high = [r for r in rows_ if r.get(bucket_field) == "HIGH" and r.get("path_class")]
        low = [r for r in rows_ if r.get(bucket_field) == "LOW" and r.get("path_class")]

        def _rate(rr):
            n = len(rr)
            nr = sum(1 for r in rr if r["path_class"] == "REVERSAL")
            return {"n": n, "n_reversal": nr, "reversal_rate": round(nr / n, 4) if n else None}

        return {"high": _rate(high), "low": _rate(low)}

    return {"n_sampled": len(control_rows), "by_feature": {f: _descriptive(control_rows, f"{f}_bucket") for f in ALL_FEATURES}}


def freeze_and_record(conn, provenance: str = "batch-p6-010-w7a-state-structure-aging-market-clocks") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    neg_control = compute_negative_control(conn, metrics)
    closure = classify_closure(metrics["tests"])

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;total_anchor_n={metrics['total_anchor_n']};"
        f"analyzed_n={metrics['analyzed_n']};independent_cycle_n={metrics['independent_cycle_n']};"
        f"maturity_cutoff_ts_ms={metrics['maturity_cutoff_ts_ms']}"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- TRAIN medians frozen,
    # applied to TEST; drift-only refresh in practice).
    rows_to_write = [
        ("total_anchor_n", metrics["total_anchor_n"]),
        ("analyzed_n", metrics["analyzed_n"]),
        ("excluded_no_horizon_data", metrics["excluded_no_horizon_data"]),
        ("independent_cycle_n", metrics["independent_cycle_n"]),
        ("maturity_cutoff_ts_ms", metrics["maturity_cutoff_ts_ms"]),
        ("stability", metrics["stability"]),
        ("negative_control", neg_control),
        ("closure_classification", closure),
    ]
    for feat in ALL_FEATURES:
        rows_to_write.append((f"test_{feat}", metrics["tests"][feat]))

    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_SIGNAL_AGING_MARKET_CLOCK",
            "hypothesis_id": "H-W7A-STATE-STRUCTURE-AGING-MARKET-CLOCKS", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "state_age,swing_age,level_age,push_age,liq_age_same_direction,"
                               "liq_age_opposite_direction,trade_count_clock,volume_clock,realized_vol_clock",
            "frozen_target": "swing_24h path class REVERSAL (reused verbatim from "
                             "E-W4-POST-EVENT-PATH-TAXONOMY-001)",
            "frozen_thresholds": f"state_age_cadence_ms={STATE_AGE_CADENCE_MS};"
                                 f"state_age_max_steps={STATE_AGE_MAX_STEPS};"
                                 f"liq_notional_threshold={LIQ_NOTIONAL_THRESHOLD} (reused from "
                                 f"StateEngine._cascade());clock_window_candles={CLOCK_WINDOW_CANDLES};"
                                 f"clock_baseline_windows={CLOCK_BASELINE_WINDOWS};"
                                 f"clock_baseline_min_valid={CLOCK_BASELINE_MIN_VALID};"
                                 "clock_bucket_split=ratio>1.0 (non-fit);duration_bucket_split=TRAIN-median "
                                 "(frozen on TRAIN, applied to TEST+negative-control, never "
                                 "full-population); all thresholds fixed pre-hoc, no sweep",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " (TRAIN medians frozen, applied to TEST)",
            "frozen_economic_gate": "N/A (descriptive conditioning only -- no entry/exit/economic claim, "
                                    "no bucket/route/observer change)",
            "frozen_statistical_gate": f"primary=independent-cycle cluster block-bootstrap "
                                       f"risk-difference CI (n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}); "
                                       f"secondary=two-sided label-permutation (n={N_PERMUTATIONS}, "
                                       f"seed={PERMUTATION_SEED}) + Holm step-down across the 9 "
                                       "FAM_SIGNAL_AGING_MARKET_CLOCK tests. CHECKPOINT CLOSED 2026-07-04 "
                                       "(operator instruction): nominal (pre-Holm) p<0.05 tests were NOT "
                                       "chased with a post-hoc threshold/subgroup search; see "
                                       "closure_classification result -- UNCONFIRMED_DESCRIPTIVE_LEAD, "
                                       "not promoted, not discarded",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
            "report_artifact_id": None, "schema_version": 7, "provenance": provenance,
            "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in rows_to_write],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return {
        "total_anchor_n": metrics["total_anchor_n"],
        "analyzed_n": metrics["analyzed_n"],
        "excluded_no_horizon_data": metrics["excluded_no_horizon_data"],
        "independent_cycle_n": metrics["independent_cycle_n"],
        "maturity_cutoff_ts_ms": metrics["maturity_cutoff_ts_ms"],
        "stability": metrics["stability"],
        "tests": metrics["tests"],
        "negative_control": neg_control,
        "closure_classification": closure,
    }


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        r = freeze_and_record(conn)
        print(f"total_anchor_n={r['total_anchor_n']} analyzed_n={r['analyzed_n']} "
              f"excluded_no_horizon_data={r['excluded_no_horizon_data']} "
              f"independent_cycle_n={r['independent_cycle_n']}")
        print(f"maturity_cutoff_ts_ms={r['maturity_cutoff_ts_ms']}")
        print(f"stability={r['stability']}")
        for feat in ALL_FEATURES:
            print(f"{feat}: {r['tests'][feat]}")
        print(f"negative_control={r['negative_control']}")
        print(f"closure_classification={r['closure_classification']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
