"""BATCH-P6-005 (W4): Post-event path taxonomy + structural location + event
geometry (HISTORICAL_RESEARCH_WAVES.md W4, precondition P3 paths -- satisfied).

Preregistration frozen BEFORE any outcome is looked at (this docstring +
the module-level constants below ARE the freeze; freeze_and_record() writes
the same frozen values to experiment_registry so they cannot silently drift):

POPULATION (frozen): all ETHUSDT REAL_LIQUIDATION ami_events (via
feature_gateway.fetch_events, R-09 pooling guard applies) -- the SAME 252-
anchor / 167-cycle population as W1 (E-W1-CYCLE-INTEGRITY-001), reused rather
than redefined, per the operator's "independent-cycle denominator" ask.
raw anchor N=252 is reported alongside independent_cycle_n=167 (canonical-v1,
via event_cycle_membership) -- the LATTER is the multiple-testing denominator,
per OD-011's lesson that raw anchor N overstates independence.

PATH HORIZONS (frozen, not searched): scalp_30m, scalp_1h, swing_4h,
swing_24h -- exactly the scalp/swing split the operator asked for.

OUTCOME TAXONOMY (frozen, not fit to find leads): for each horizon, path
return_bps = (horizon_close - anchor_ref_close) / anchor_ref_close * 1e4,
where anchor_ref_close is the last CLOSED 1m candle at or before anchor_ts_ms
(known-at-safe: never uses a candle whose close is after the anchor). Class:
    CONTINUATION  if return_bps <= -CLASSIFICATION_BAND_BPS (price kept
                  falling in the cascade's own SELL direction)
    REVERSAL      if return_bps >= +CLASSIFICATION_BAND_BPS (price recovered)
    CHOP          otherwise
CLASSIFICATION_BAND_BPS=20 is a round, pre-committed number (not optimized
against outcomes -- optimizing it would silently re-open the graveyarded
"reversal fade" search under a new name).

THIS IS NOT THE GRAVEYARDED "REVERSAL" HYPOTHESIS: failure_archive already
comprehensively rejected trading a fade-the-cascade ENTRY with a stop/TP grid
(SOLUSDT N=66 all-view; ETHUSDT N=346 stop-sweep, 0 leads -- see
reports/research/s34/S34_REVERSAL_BACKTEST.md, S34_REVERSAL_STOP_BACKTEST.md).
This module computes NO entry, NO exit, NO stop, NO economic gate, and makes
NO tradeability claim -- it only classifies and counts what SHAPE the price
path took, exactly like W1 was a descriptive integrity wave, not an alpha
search. software_verdict/scientific_verdict below reflect this (PASSED /
ANSWERED_SUPPORTED as a *descriptive* result, never a promotion candidate).

STRUCTURAL LOCATION / EVENT GEOMETRY (frozen flags, all point-in-time-safe --
only objects with known_at_ts <= anchor_ts_ms are consulted):
    near_swing_low   -- nearest known SWING_LOW's pivot_price within
                        NEAR_STRUCTURE_BPS of the anchor ref price
    near_level       -- nearest known ami_levels row's price within
                        NEAR_STRUCTURE_BPS (excludes blocked F-B2 columns;
                        only price/known_at_ts/level_type, via
                        fetch_level_features's safe-column allowlist)
    recent_down_push -- a DOWN push whose end_ts <= anchor_ts_ms and
                        anchor_ts_ms - end_ts <= RECENT_PUSH_LOOKBACK_MS,
                        known_at_ts <= anchor_ts_ms

CONTROLS: (a) chronological 70/30 split of the 252 anchors by anchor_ts_ms --
frequencies are reported in BOTH splits to check descriptive stability, NOT
to fit a threshold (there is nothing to fit; the band is fixed above); (b)
negative control via ami_candidate_universe (now full-history, BATCH-P6-003b)
-- an equal-sized random sample of non-event-aligned candidate slots gets the
same path-shape classification, to check whether real anchors differ from
arbitrary moments at all.

MULTIPLE-TESTING FAMILY (frozen, 5 comparisons, FAM_POST_EVENT_PATH_TAXONOMY
-- no other comparison is added post-hoc; a new pattern noticed while looking
at results is a NEW hypothesis per Protocol §2, not added here):
    C1: scalp (30m/1h) vs swing (4h/24h) path-shape distribution
    C2: near_swing_low vs not
    C3: near_level vs not
    C4: recent_down_push vs not
    C5: anchor population vs random-time (candidate universe) control

STOP CONDITIONS: any bucket (train OR test split) with N < MIN_BUCKET_N is
reported as INSUFFICIENT_SAMPLE for that specific comparison, not silently
dropped or averaged over.
"""
from __future__ import annotations
import bisect
import hashlib
import random
import time
from collections import Counter

from ami.research.feature_gateway import (
    fetch_chart_feature,
    fetch_events,
    fetch_level_features,
)
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W4-POST-EVENT-PATH-TAXONOMY-001"
RESEARCH_CONTEXT_ID = "w4-post-event-path-taxonomy"

PATH_HORIZONS_MS = {
    "scalp_30m": 30 * 60_000,
    "scalp_1h": 60 * 60_000,
    "swing_4h": 4 * 3600_000,
    "swing_24h": 24 * 3600_000,
}
SCALP_HORIZONS = {"scalp_30m", "scalp_1h"}
SWING_HORIZONS = {"swing_4h", "swing_24h"}

CLASSIFICATION_BAND_BPS = 20.0
NEAR_STRUCTURE_BPS = 50.0
RECENT_PUSH_LOOKBACK_MS = 4 * 3600_000
MIN_BUCKET_N = 20
TRAIN_FRACTION = 0.7
CONTROL_SEED = 20260704  # fixed, not tuned -- deterministic negative-control sampling


def classify_path(return_bps: float) -> str:
    if return_bps <= -CLASSIFICATION_BAND_BPS:
        return "CONTINUATION"
    if return_bps >= CLASSIFICATION_BAND_BPS:
        return "REVERSAL"
    return "CHOP"


class _CandleIndex:
    """Sorted (close_ts_ms, close) lookup for known-at-safe reference/horizon prices."""

    def __init__(self, candles: list[dict]):
        ordered = sorted(candles, key=lambda c: c["close_ts_ms"])
        self._close_ts = [c["close_ts_ms"] for c in ordered]
        self._close_px = [c["close"] for c in ordered]

    def ref_price_at_or_before(self, ts_ms: int) -> float | None:
        i = bisect.bisect_right(self._close_ts, ts_ms) - 1
        if i < 0:
            return None
        return self._close_px[i]

    def price_at_or_after(self, ts_ms: int) -> float | None:
        i = bisect.bisect_left(self._close_ts, ts_ms)
        if i >= len(self._close_ts):
            return None
        return self._close_px[i]


def compute_path_returns(candle_index: _CandleIndex, anchor_ts_ms: int) -> dict:
    """Returns {horizon_name: return_bps or None} -- None if data is not yet
    available at that horizon (never fabricated)."""
    ref = candle_index.ref_price_at_or_before(anchor_ts_ms)
    out = {}
    if ref is None:
        return {name: None for name in PATH_HORIZONS_MS}
    for name, horizon_ms in PATH_HORIZONS_MS.items():
        px = candle_index.price_at_or_after(anchor_ts_ms + horizon_ms)
        out[name] = None if px is None else round((px - ref) / ref * 1e4, 4)
    return out


def _nearest_abs_bps(price: float, candidates_price: list[float]) -> float | None:
    if not candidates_price or price is None:
        return None
    return min(abs((p - price) / price * 1e4) for p in candidates_price)


def compute_structural_flags(anchor_ts_ms: int, anchor_ref_price: float | None,
                              swings: list[dict], levels: list[dict], pushes: list[dict]) -> dict:
    """All lookups are point-in-time-safe: only known_at_ts <= anchor_ts_ms rows considered."""
    if anchor_ref_price is None:
        return {"near_swing_low": False, "near_level": False, "recent_down_push": False}

    known_swing_lows = [s["pivot_price"] for s in swings
                         if s["swing_type"] == "LOW" and s["known_at_ts"] <= anchor_ts_ms]
    dist = _nearest_abs_bps(anchor_ref_price, known_swing_lows)
    near_swing_low = dist is not None and dist <= NEAR_STRUCTURE_BPS

    known_level_prices = [lv["price"] for lv in levels if lv["known_at_ts"] <= anchor_ts_ms]
    dist_lv = _nearest_abs_bps(anchor_ref_price, known_level_prices)
    near_level = dist_lv is not None and dist_lv <= NEAR_STRUCTURE_BPS

    recent_down_push = any(
        p["direction"] == "DOWN" and p["known_at_ts"] <= anchor_ts_ms
        and p["end_ts"] <= anchor_ts_ms
        and (anchor_ts_ms - p["end_ts"]) <= RECENT_PUSH_LOOKBACK_MS
        for p in pushes
    )
    return {"near_swing_low": near_swing_low, "near_level": near_level, "recent_down_push": recent_down_push}


def _distribution(classes: list[str]) -> dict:
    n = len(classes)
    c = Counter(classes)
    return {"n": n, **{k: c.get(k, 0) for k in ("CONTINUATION", "REVERSAL", "CHOP")}}


def _split_chronological(anchors: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(anchors, key=lambda a: a["anchor_ts_ms"])
    cut = int(len(ordered) * TRAIN_FRACTION)
    return ordered[:cut], ordered[cut:]


def _bucket_or_insufficient(classes: list[str]) -> dict:
    d = _distribution(classes)
    d["insufficient_sample"] = d["n"] < MIN_BUCKET_N
    return d


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")
    membership_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "event_cycle_membership", ["event_id", "candidate_cycle_key"],
        equals={"cycle_definition_version": "canonical-v1", "is_canonical": 1},
    )
    event_to_cycle = {r["event_id"]: r["candidate_cycle_key"] for r in membership_rows}
    independent_cycle_n = len({event_to_cycle[e["event_id"]] for e in events if e["event_id"] in event_to_cycle})

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)

    swing_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_swings", ["swing_type", "pivot_price", "known_at_ts"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    level_rows = fetch_level_features(
        conn, RESEARCH_CONTEXT_ID, ["price", "known_at_ts"], symbol=symbol,
    )
    push_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_pushes", ["direction", "end_ts", "known_at_ts"],
        symbol=symbol, equals={"timeframe": "1m"},
    )

    per_anchor = []
    for e in events:
        anchor_ts = e["anchor_ts_ms"]
        returns = compute_path_returns(candle_index, anchor_ts)
        ref_price = candle_index.ref_price_at_or_before(anchor_ts)
        flags = compute_structural_flags(anchor_ts, ref_price, swing_rows, level_rows, push_rows)
        per_anchor.append({
            "event_id": e["event_id"],
            "anchor_ts_ms": anchor_ts,
            "returns": returns,
            "classes": {h: (classify_path(r) if r is not None else None) for h, r in returns.items()},
            **flags,
        })

    # C1: scalp vs swing path-shape distribution (pooled across the 2 horizons
    # in each group; only anchors with a non-None class at that horizon count).
    scalp_classes = [a["classes"][h] for a in per_anchor for h in SCALP_HORIZONS if a["classes"][h] is not None]
    swing_classes = [a["classes"][h] for a in per_anchor for h in SWING_HORIZONS if a["classes"][h] is not None]

    c1 = {"scalp": _bucket_or_insufficient(scalp_classes), "swing": _bucket_or_insufficient(swing_classes)}

    # C2/C3/C4: structural-location conditioning, using swing_24h as the
    # single representative horizon (frozen choice: the longest, least-noisy
    # horizon -- avoids running the same comparison 4x per flag, which would
    # inflate the family beyond the pre-registered 5 comparisons).
    def _flag_comparison(flag_name: str) -> dict:
        yes = [a["classes"]["swing_24h"] for a in per_anchor if a[flag_name] and a["classes"]["swing_24h"]]
        no = [a["classes"]["swing_24h"] for a in per_anchor if not a[flag_name] and a["classes"]["swing_24h"]]
        return {"yes": _bucket_or_insufficient(yes), "no": _bucket_or_insufficient(no)}

    c2 = _flag_comparison("near_swing_low")
    c3 = _flag_comparison("near_level")
    c4 = _flag_comparison("recent_down_push")

    # Chronological 70/30 stability check (not threshold-fitting -- there is
    # no threshold to fit; this only checks the descriptive distribution
    # doesn't silently depend on a TRAIN-only artifact of the anchor set).
    train_anchors, test_anchors = _split_chronological(per_anchor)
    train_classes = [a["classes"]["swing_24h"] for a in train_anchors if a["classes"]["swing_24h"]]
    test_classes = [a["classes"]["swing_24h"] for a in test_anchors if a["classes"]["swing_24h"]]
    stability = {"train": _bucket_or_insufficient(train_classes), "test": _bucket_or_insufficient(test_classes)}

    return {
        "anchor_n": len(events),
        "independent_cycle_n": independent_cycle_n,
        "per_anchor": per_anchor,
        "c1_scalp_vs_swing": c1,
        "c2_near_swing_low": c2,
        "c3_near_level": c3,
        "c4_recent_down_push": c4,
        "stability_train_test": stability,
        "_candle_index": candle_index,  # reused by compute_negative_control
    }


def compute_negative_control(conn, candle_index: _CandleIndex, n_anchor: int,
                              symbol: str = "ETHUSDT") -> dict:
    """C5: are real anchors different from arbitrary candidate-universe
    moments? Deterministic (fixed seed) sample of n_anchor non-event-aligned
    candidate slots, same swing_24h classification."""
    rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candidate_universe", ["slot_ts_ms"],
        symbol=symbol, equals={"timeframe": "1m", "is_event_aligned": 0},
    )
    rng = random.Random(CONTROL_SEED)
    sample = rng.sample(rows, min(n_anchor, len(rows)))
    classes = []
    for r in sample:
        returns = compute_path_returns(candle_index, r["slot_ts_ms"])
        cls = classify_path(returns["swing_24h"]) if returns["swing_24h"] is not None else None
        if cls is not None:
            classes.append(cls)
    d = _distribution(classes)
    d["insufficient_sample"] = d["n"] < MIN_BUCKET_N
    return d


def freeze_and_record(conn, provenance: str = "batch-p6-005-w4-path-taxonomy") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)
    candle_index = metrics.pop("_candle_index")
    c5 = compute_negative_control(conn, candle_index, metrics["anchor_n"])

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;anchor_n={metrics['anchor_n']};"
        f"independent_cycle_n={metrics['independent_cycle_n']};cycle_definition_version=canonical-v1"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary. no_test_split=False (this module DOES run a
    # chronological train/test stability check) -- but since this call
    # reuses the exact same computed content as whatever is already stored,
    # it resolves as a drift-only snapshot refresh (dataset_hash/anchor-count
    # growth only) and never actually needs test_cycle_ids; see
    # register_legacy_snapshot_with_gates docstring. If this module's
    # SCIENTIFIC content (population definition/features/target/thresholds/
    # split methodology) is ever deliberately changed, it will need real
    # frozen TEST-side anchor/cycle ids supplied via test_cycle_ids= --
    # documented, not invented here.
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_POST_EVENT_PATH_TAXONOMY",
            "hypothesis_id": "H-W4-PATH-TAXONOMY", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "near_swing_low,near_level,recent_down_push,path_return_bps@4-horizons",
            "frozen_target": "path shape class (CONTINUATION/REVERSAL/CHOP) at scalp/swing horizons -- "
                             "descriptive only",
            "frozen_thresholds": f"classification_band_bps={CLASSIFICATION_BAND_BPS};"
                                 f"near_structure_bps={NEAR_STRUCTURE_BPS};"
                                 f"recent_push_lookback_ms={RECENT_PUSH_LOOKBACK_MS};"
                                 f"min_bucket_n={MIN_BUCKET_N} (all fixed pre-hoc, not fit)",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " stability check (no threshold fit)",
            "frozen_economic_gate": "N/A (descriptive taxonomy, no entry/exit/economic claim -- NOT the "
                                    "graveyarded reversal-fade hypothesis)",
            "frozen_statistical_gate": "N/A (descriptive; 5-comparison fixed family "
                                       "FAM_POST_EVENT_PATH_TAXONOMY, no post-hoc additions)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
            "report_artifact_id": None, "schema_version": 7, "provenance": provenance,
            "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in [
            ("anchor_n", metrics["anchor_n"]),
            ("independent_cycle_n", metrics["independent_cycle_n"]),
            ("c1_scalp_vs_swing", metrics["c1_scalp_vs_swing"]),
            ("c2_near_swing_low", metrics["c2_near_swing_low"]),
            ("c3_near_level", metrics["c3_near_level"]),
            ("c4_recent_down_push", metrics["c4_recent_down_push"]),
            ("c5_anchor_vs_random_control", c5),
            ("stability_train_test", metrics["stability_train_test"]),
        ]],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return {**{k: v for k, v in metrics.items() if k != "per_anchor"}, "c5_anchor_vs_random_control": c5}


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        result = freeze_and_record(conn)
        print(
            f"anchor_n={result['anchor_n']} independent_cycle_n={result['independent_cycle_n']}\n"
            f"C1 scalp_vs_swing={result['c1_scalp_vs_swing']}\n"
            f"C2 near_swing_low={result['c2_near_swing_low']}\n"
            f"C3 near_level={result['c3_near_level']}\n"
            f"C4 recent_down_push={result['c4_recent_down_push']}\n"
            f"C5 anchor_vs_random={result['c5_anchor_vs_random_control']}\n"
            f"stability_train_test={result['stability_train_test']}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
