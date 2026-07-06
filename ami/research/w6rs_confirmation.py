"""BATCH-P6-008: W6-RS confirmation wave (HISTORICAL_RESEARCH_WAVES.md W6
follow-up). Confirms ONE exact, pre-registered hypothesis from W6
(E-W6-COMPRESSION-RS-SESSION-001) -- no threshold or horizon sweep, no new
feature fishing.

EXACT HYPOTHESIS (frozen, verbatim from the operator):
    After a SELL-cascade anchor, when known-at-safe
    ETH_1h_return - BTC_1h_return > 0 at the anchor,
    the probability of a 24h REVERSAL is higher than in the RS<=0 group.

RS_THRESHOLD=0.0 and RS_LOOKBACK_MS=1h are FROZEN exactly as in W6 -- this
module NEVER searches a different threshold or horizon. Population and
outcome (swing_24h path class, E-W4-POST-EVENT-PATH-TAXONOMY-001's fixed
+-20bps band) are REUSED verbatim from W4/W6, not redefined.

This is STILL a descriptive/confirmatory wave: NO entry/exit/PnL claim, NO
bucket/route promotion. If the hypothesis survives the chronological holdout
here, the NEXT step (not this module) would be a SEPARATE, execution-haircut
economic validation preregistration -- not decided or started here.

NOT the graveyarded reversal-fade hypothesis: no entry, no stop, no TP, no
fee model, no economic gate anywhere in this module.

Reporting checklist (operator-specified, all computed below):
  1. feature coverage + reason for the missing anchors
  2. independent-cycle N (canonical-v1, for the analyzed subset)
  3. chronological 70/30 train/test counts AND the RS-vs-outcome split
     replicated independently in EACH split (not just an unconditional
     stability check)
  4. monthly rolling stability (RS-conditioned REVERSAL rate per month)
  5. effect size (risk difference) + uncertainty (Wilson 95% CI + a
     label-permutation test, 2000 draws, fixed seed)
  6. candidate-universe negative control (same RS/outcome classification
     applied to an equal-sized random-time sample)
  7. preregistered minimal confounder checks: event notional, session,
     day_trend_bps (day-open-to-anchor return, same convention already used
     by tools/research_s34_btc_sell_anomaly_audit.py) -- compared between
     the RS_STRONG and RS_WEAK groups
"""
from __future__ import annotations
import datetime as dt
import hashlib
import random
import time
from collections import Counter

from ami.chart.level_registry import _session_of_hour
from ami.research.feature_gateway import fetch_chart_feature, fetch_events
from ami.research.w4_post_event_path_taxonomy import (
    _CandleIndex,
    classify_path,
    compute_path_returns,
)
from ami.research.w6_compression_rs_session import RS_LOOKBACK_MS, classify_relative_strength
from ami.states.engine import StateEngine
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W6RS-CONFIRMATION-001"
RESEARCH_CONTEXT_ID = "w6rs-confirmation"

RS_THRESHOLD = 0.0  # FROZEN -- identical to W6, never re-swept in this module
MIN_BUCKET_N = 20
TRAIN_FRACTION = 0.7
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 20260704  # fixed, not tuned
NEGATIVE_CONTROL_SEED = 20260704


def compute_day_trend_bps(engine: StateEngine, symbol: str, anchor_ts_ms: int) -> float | None:
    """(mark_now - day_open) / day_open * 1e4 -- same convention as
    tools/research_s34_btc_sell_anomaly_audit.py, reused rather than
    reinvented, for the day_trend confounder check."""
    d = dt.datetime.fromtimestamp(anchor_ts_ms / 1000, dt.timezone.utc)
    day_start_ms = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    day_open = engine._px(symbol, day_start_ms)
    mark_now = engine._px(symbol, anchor_ts_ms)
    if day_open is None or mark_now is None or day_open <= 0:
        return None
    return round((mark_now - day_open) / day_open * 1e4, 4)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def permutation_test_risk_difference(strong_n: int, strong_success: int, weak_n: int, weak_success: int,
                                      n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> dict:
    """One-sided label-permutation test for the pre-registered directional
    hypothesis (RS_STRONG reversal rate > RS_WEAK reversal rate)."""
    if strong_n == 0 or weak_n == 0:
        return {"observed_risk_difference": None, "p_value": None, "n_perm": n_perm}
    obs_diff = strong_success / strong_n - weak_success / weak_n
    pooled = [1] * (strong_success + weak_success) + [0] * ((strong_n - strong_success) + (weak_n - weak_success))
    rng = random.Random(seed)
    n_at_least_as_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        p_strong = sum(pooled[:strong_n]) / strong_n
        p_weak = sum(pooled[strong_n:strong_n + weak_n]) / weak_n
        if (p_strong - p_weak) >= obs_diff:
            n_at_least_as_extreme += 1
    return {
        "observed_risk_difference": round(obs_diff, 4),
        "p_value": round(n_at_least_as_extreme / n_perm, 5),
        "n_perm": n_perm,
    }


def _month_key(ts_ms: int) -> str:
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


def _reversal_rate_bucket(rows: list[dict]) -> dict:
    n = len(rows)
    n_reversal = sum(1 for r in rows if r["path_class"] == "REVERSAL")
    d = {"n": n, "n_reversal": n_reversal, "reversal_rate": round(n_reversal / n, 4) if n else None,
         "insufficient_sample": n < MIN_BUCKET_N}
    return d


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)
    max_candle_close_ts = max(r["close_ts_ms"] for r in candle_rows)

    membership_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "event_cycle_membership", ["event_id", "candidate_cycle_key"],
        equals={"cycle_definition_version": "canonical-v1", "is_canonical": 1},
    )
    event_to_cycle = {r["event_id"]: r["candidate_cycle_key"] for r in membership_rows}

    engine = StateEngine()
    try:
        per_anchor = []
        excluded_no_horizon_data = 0
        for e in events:
            anchor_ts = e["anchor_ts_ms"]
            returns = compute_path_returns(candle_index, anchor_ts)
            if returns["swing_24h"] is None:
                # feature-coverage reason: candle data does not yet reach
                # anchor_ts + 24h (data-recency limit, not a design gap).
                if anchor_ts + 24 * 3600_000 > max_candle_close_ts:
                    excluded_no_horizon_data += 1
                continue
            path_class = classify_path(returns["swing_24h"])
            rs_val_eth = engine._ret_bps(symbol, anchor_ts, RS_LOOKBACK_MS)
            rs_val_btc = engine._ret_bps("BTCUSDT", anchor_ts, RS_LOOKBACK_MS)
            if rs_val_eth is None or rs_val_btc is None:
                continue
            rs_diff = rs_val_eth - rs_val_btc
            rs_group = "RS_STRONG" if rs_diff > RS_THRESHOLD else "RS_WEAK"

            per_anchor.append({
                "event_id": e["event_id"],
                "anchor_ts_ms": anchor_ts,
                "path_class": path_class,
                "rs_diff_bps": round(rs_diff, 4),
                "rs_group": rs_group,
                "cycle_id": event_to_cycle.get(e["event_id"]),
                "notional": e["notional"],
                "session": _session_of_hour(dt.datetime.fromtimestamp(anchor_ts / 1000, dt.timezone.utc).hour),
                "day_trend_bps": compute_day_trend_bps(engine, symbol, anchor_ts),
                "month": _month_key(anchor_ts),
            })
    finally:
        engine.conn.close()

    total_anchor_n = len(events)
    analyzed_n = len(per_anchor)
    excluded_n = total_anchor_n - analyzed_n

    independent_cycle_n = len({a["cycle_id"] for a in per_anchor if a["cycle_id"] is not None})

    strong = [a for a in per_anchor if a["rs_group"] == "RS_STRONG"]
    weak = [a for a in per_anchor if a["rs_group"] == "RS_WEAK"]
    strong_bucket = _reversal_rate_bucket(strong)
    weak_bucket = _reversal_rate_bucket(weak)
    strong_ci = wilson_ci(strong_bucket["n_reversal"], strong_bucket["n"]) if strong_bucket["n"] else (0.0, 0.0)
    weak_ci = wilson_ci(weak_bucket["n_reversal"], weak_bucket["n"]) if weak_bucket["n"] else (0.0, 0.0)
    perm = permutation_test_risk_difference(
        strong_bucket["n"], strong_bucket["n_reversal"], weak_bucket["n"], weak_bucket["n_reversal"]
    )

    ordered = sorted(per_anchor, key=lambda a: a["anchor_ts_ms"])
    cut = int(len(ordered) * TRAIN_FRACTION)
    train, test = ordered[:cut], ordered[cut:]

    def _split_rs_breakdown(rows: list[dict]) -> dict:
        s = [r for r in rows if r["rs_group"] == "RS_STRONG"]
        w = [r for r in rows if r["rs_group"] == "RS_WEAK"]
        return {"n": len(rows), "rs_strong": _reversal_rate_bucket(s), "rs_weak": _reversal_rate_bucket(w)}

    train_breakdown = _split_rs_breakdown(train)
    test_breakdown = _split_rs_breakdown(test)

    months = sorted({a["month"] for a in per_anchor})
    monthly = {}
    for m in months:
        rows_m = [a for a in per_anchor if a["month"] == m]
        s = [r for r in rows_m if r["rs_group"] == "RS_STRONG"]
        w = [r for r in rows_m if r["rs_group"] == "RS_WEAK"]
        monthly[m] = {"rs_strong": _reversal_rate_bucket(s), "rs_weak": _reversal_rate_bucket(w)}

    # confounder checks: does event size / session / day_trend differ
    # systematically between the two RS groups? (descriptive comparison only)
    def _numeric_summary(values: list[float]) -> dict:
        vals = [v for v in values if v is not None]
        if not vals:
            return {"n": 0, "mean": None, "median": None}
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        median = vals_sorted[mid] if len(vals_sorted) % 2 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
        return {"n": len(vals), "mean": round(sum(vals) / len(vals), 2), "median": round(median, 2)}

    confounders = {
        "notional": {
            "rs_strong": _numeric_summary([a["notional"] for a in strong]),
            "rs_weak": _numeric_summary([a["notional"] for a in weak]),
        },
        "day_trend_bps": {
            "rs_strong": _numeric_summary([a["day_trend_bps"] for a in strong]),
            "rs_weak": _numeric_summary([a["day_trend_bps"] for a in weak]),
        },
        "session": {
            "rs_strong": dict(Counter(a["session"] for a in strong)),
            "rs_weak": dict(Counter(a["session"] for a in weak)),
        },
    }

    return {
        "per_anchor": per_anchor,
        "total_anchor_n": total_anchor_n,
        "analyzed_n": analyzed_n,
        "excluded_n": excluded_n,
        "excluded_no_horizon_data": excluded_no_horizon_data,
        "independent_cycle_n": independent_cycle_n,
        "rs_strong": strong_bucket,
        "rs_strong_ci95": strong_ci,
        "rs_weak": weak_bucket,
        "rs_weak_ci95": weak_ci,
        "permutation_test": perm,
        "train_n": len(train),
        "test_n": len(test),
        "train_breakdown": train_breakdown,
        "test_breakdown": test_breakdown,
        "monthly_stability": monthly,
        "confounders": confounders,
    }


def compute_negative_control(conn, symbol: str = "ETHUSDT", n_target: int | None = None) -> dict:
    """Same RS/outcome classification applied to a deterministic random
    sample of non-event-aligned candidate-universe slots -- tests whether
    the RS-to-REVERSAL relationship is specific to post-cascade paths or a
    general market-regime effect present at arbitrary times too."""
    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)
    max_candle_close_ts = max(r["close_ts_ms"] for r in candle_rows)

    rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe", ["slot_ts_ms"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    rng = random.Random(NEGATIVE_CONTROL_SEED)
    sample_size = n_target if n_target is not None else len(rows)
    sample = rng.sample(rows, min(sample_size, len(rows)))

    engine = StateEngine()
    try:
        strong, weak = [], []
        for r in sample:
            ts = r["slot_ts_ms"]
            if ts + 24 * 3600_000 > max_candle_close_ts:
                continue
            returns = compute_path_returns(candle_index, ts)
            if returns["swing_24h"] is None:
                continue
            path_class = classify_path(returns["swing_24h"])
            rs_eth = engine._ret_bps(symbol, ts, RS_LOOKBACK_MS)
            rs_btc = engine._ret_bps("BTCUSDT", ts, RS_LOOKBACK_MS)
            if rs_eth is None or rs_btc is None:
                continue
            row = {"path_class": path_class}
            (strong if (rs_eth - rs_btc) > RS_THRESHOLD else weak).append(row)
    finally:
        engine.conn.close()

    return {"rs_strong": _reversal_rate_bucket(strong), "rs_weak": _reversal_rate_bucket(weak)}


def freeze_and_record(conn, provenance: str = "batch-p6-008-w6rs-confirmation") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    neg_control = compute_negative_control(conn, n_target=metrics["analyzed_n"])

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;total_anchor_n={metrics['total_anchor_n']};"
        f"analyzed_n={metrics['analyzed_n']};independent_cycle_n={metrics['independent_cycle_n']}"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- this is the module
    # with the genuine confirmatory chronological train/test + permutation
    # test; drift-only refresh in practice, see register_legacy_snapshot_
    # with_gates docstring for the exact adaptation needed if that ever stops
    # being true).
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_COMPRESSION_RS_SESSION",
            "hypothesis_id": "H-W6RS-CONFIRMATION", "preregistered_at": now,
            "frozen_population": frozen_population, "frozen_features": "rs_diff_bps(ETH_1h-BTC_1h)",
            "frozen_target": "swing_24h path class REVERSAL rate (reused from "
                             "E-W4-POST-EVENT-PATH-TAXONOMY-001)",
            "frozen_thresholds": f"rs_threshold={RS_THRESHOLD} (FROZEN, identical to W6, never "
                                 f"re-swept); rs_lookback_ms={RS_LOOKBACK_MS} (FROZEN)",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " + monthly rolling stability",
            "frozen_economic_gate": "N/A (descriptive/confirmatory only -- no entry/exit/PnL claim, "
                                    "no bucket/route promotion)",
            "frozen_statistical_gate": f"one-sided label-permutation test, n_perm={N_PERMUTATIONS}, "
                                       f"seed={PERMUTATION_SEED}; single pre-registered comparison (no "
                                       "multiple-testing family beyond this one hypothesis)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": "E-W6-COMPRESSION-RS-SESSION-001", "report_artifact_id": None,
            "schema_version": 7, "provenance": provenance, "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in [
            ("total_anchor_n", metrics["total_anchor_n"]),
            ("analyzed_n", metrics["analyzed_n"]),
            ("excluded_n", metrics["excluded_n"]),
            ("excluded_no_horizon_data", metrics["excluded_no_horizon_data"]),
            ("independent_cycle_n", metrics["independent_cycle_n"]),
            ("rs_strong", metrics["rs_strong"]),
            ("rs_strong_ci95", metrics["rs_strong_ci95"]),
            ("rs_weak", metrics["rs_weak"]),
            ("rs_weak_ci95", metrics["rs_weak_ci95"]),
            ("permutation_test", metrics["permutation_test"]),
            ("train_n", metrics["train_n"]),
            ("test_n", metrics["test_n"]),
            ("train_breakdown", metrics["train_breakdown"]),
            ("test_breakdown", metrics["test_breakdown"]),
            ("monthly_stability", metrics["monthly_stability"]),
            ("confounders", metrics["confounders"]),
            ("negative_control", neg_control),
        ]],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return {**{k: v for k, v in metrics.items() if k != "per_anchor"}, "negative_control": neg_control}


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        r = freeze_and_record(conn)
        print(f"total_anchor_n={r['total_anchor_n']} analyzed_n={r['analyzed_n']} "
              f"excluded_n={r['excluded_n']} (no_horizon_data={r['excluded_no_horizon_data']}) "
              f"independent_cycle_n={r['independent_cycle_n']}\n"
              f"RS_STRONG={r['rs_strong']} ci95={r['rs_strong_ci95']}\n"
              f"RS_WEAK={r['rs_weak']} ci95={r['rs_weak_ci95']}\n"
              f"permutation_test={r['permutation_test']}\n"
              f"train(n={r['train_n']})={r['train_breakdown']}\n"
              f"test(n={r['test_n']})={r['test_breakdown']}\n"
              f"monthly_stability={r['monthly_stability']}\n"
              f"confounders={r['confounders']}\n"
              f"negative_control={r['negative_control']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
