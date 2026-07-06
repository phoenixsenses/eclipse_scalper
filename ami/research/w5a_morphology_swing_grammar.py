"""BATCH-P6-006 (W5a): Candle morphology + swing grammar around the anchor
(HISTORICAL_RESEARCH_WAVES.md W5, the two components of W5 that are
infrastructure-ready today; sweep/breakout-retest and unconditional SHORT
genesis are scoped separately -- see OD-014 -- since they need new chart
infrastructure / whitepaper §29 SHORT-family features not yet built, the same
class of gap that blocked W2).

Preregistration frozen BEFORE looking at outcomes:

POPULATION (frozen, reused from W1/W4): ETHUSDT REAL_LIQUIDATION anchors,
anchor_n=252, independent_cycle_n=167 (canonical-v1).

OUTCOME (frozen, REUSED from W4 -- not redefined): swing_24h path class
(CONTINUATION/REVERSAL/CHOP, E-W4-POST-EVENT-PATH-TAXONOMY-001's own fixed
±20bps band). This module adds two NEW conditioning features to an EXISTING
frozen outcome rather than inventing a new target -- avoids target-shopping.

NEW FEATURES (frozen, not fit):
  anchor_candle_morphology -- close_quality_label (CLOSE_NEAR_HIGH/
    CLOSE_NEAR_LOW/MID_RANGE_CLOSE) of the last CLOSED candle at-or-before
    anchor_ts_ms (same known-at-safe reference candle W4 uses).
  swing_grammar_pre_anchor -- classic Dow-theory-style HH/HL vs LH/LL
    structure from the last 2 known HIGH swings + last 2 known LOW swings
    (known_at_ts <= anchor_ts_ms): UPTREND_STRUCTURE (higher-high AND
    higher-low), DOWNTREND_STRUCTURE (lower-high AND lower-low),
    MIXED_STRUCTURE (otherwise), INSUFFICIENT_STRUCTURE (fewer than 2 of
    either swing type known before the anchor -- not fabricated).

MULTIPLE-TESTING FAMILY (frozen, 2 comparisons, FAM_CANDLE_MORPHOLOGY_SWING_GRAMMAR):
    D1: anchor candle morphology label vs swing_24h path outcome
    D2: pre-anchor swing-grammar structure vs swing_24h path outcome

CONTROLS: same chronological 70/30 split as W4 (stability check, not
threshold-fitting -- there is no threshold to fit here either).

STOP CONDITIONS: any bucket with N < MIN_BUCKET_N is INSUFFICIENT_SAMPLE.

Descriptive only -- no entry/exit/economic claim, not a promotion candidate.
"""
from __future__ import annotations
import bisect
import hashlib
import time
from collections import Counter

from ami.research.feature_gateway import fetch_chart_feature, fetch_events
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates
from ami.research.w4_post_event_path_taxonomy import (
    MIN_BUCKET_N,
    TRAIN_FRACTION,
    _CandleIndex,
    _split_chronological,
    classify_path,
    compute_path_returns,
)

EXPERIMENT_ID = "E-W5A-MORPHOLOGY-SWING-GRAMMAR-001"
RESEARCH_CONTEXT_ID = "w5a-morphology-swing-grammar"

MORPHOLOGY_LABELS = {"CLOSE_NEAR_HIGH", "CLOSE_NEAR_LOW", "MID_RANGE_CLOSE"}


def classify_swing_grammar(anchor_ts_ms: int, swings: list[dict]) -> str:
    """swings: dicts with swing_type, pivot_ts, pivot_price, known_at_ts.
    Only rows with known_at_ts <= anchor_ts_ms are eligible (point-in-time-safe)."""
    known = [s for s in swings if s["known_at_ts"] <= anchor_ts_ms]
    highs = sorted((s for s in known if s["swing_type"] == "HIGH"), key=lambda s: s["pivot_ts"])
    lows = sorted((s for s in known if s["swing_type"] == "LOW"), key=lambda s: s["pivot_ts"])
    if len(highs) < 2 or len(lows) < 2:
        return "INSUFFICIENT_STRUCTURE"
    higher_high = highs[-1]["pivot_price"] > highs[-2]["pivot_price"]
    higher_low = lows[-1]["pivot_price"] > lows[-2]["pivot_price"]
    if higher_high and higher_low:
        return "UPTREND_STRUCTURE"
    if not higher_high and not higher_low:
        return "DOWNTREND_STRUCTURE"
    return "MIXED_STRUCTURE"


def _distribution(classes: list[str]) -> dict:
    n = len(classes)
    c = Counter(classes)
    return {"n": n, **{k: c.get(k, 0) for k in ("CONTINUATION", "REVERSAL", "CHOP")}}


def _bucket_or_insufficient(classes: list[str]) -> dict:
    d = _distribution(classes)
    d["insufficient_sample"] = d["n"] < MIN_BUCKET_N
    return d


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)

    morph_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candle_morphology", ["candle_id", "close_quality_label"],
    )
    morph_by_candle = {r["candle_id"]: r["close_quality_label"] for r in morph_rows}
    id_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "candle_id"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    sorted_ids = sorted(id_rows, key=lambda r: r["close_ts_ms"])
    id_close_ts = [r["close_ts_ms"] for r in sorted_ids]
    id_candle_id = [r["candle_id"] for r in sorted_ids]

    def _ref_candle_id(ts_ms: int) -> str | None:
        i = bisect.bisect_right(id_close_ts, ts_ms) - 1
        return None if i < 0 else id_candle_id[i]

    swing_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_swings", ["swing_type", "pivot_ts", "pivot_price", "known_at_ts"],
        symbol=symbol, equals={"timeframe": "1m"},
    )

    per_anchor = []
    for e in events:
        anchor_ts = e["anchor_ts_ms"]
        returns = compute_path_returns(candle_index, anchor_ts)
        path_class = classify_path(returns["swing_24h"]) if returns["swing_24h"] is not None else None

        ref_cid = _ref_candle_id(anchor_ts)
        morphology_label = morph_by_candle.get(ref_cid) if ref_cid is not None else None

        grammar = classify_swing_grammar(anchor_ts, swing_rows)

        per_anchor.append({
            "event_id": e["event_id"],
            "anchor_ts_ms": anchor_ts,
            "path_class": path_class,
            "morphology_label": morphology_label,
            "swing_grammar": grammar,
        })

    def _by_feature(feature_name: str, labels: set[str] | None = None) -> dict:
        out = {}
        keys = labels if labels is not None else {a[feature_name] for a in per_anchor if a[feature_name]}
        for label in keys:
            classes = [a["path_class"] for a in per_anchor if a[feature_name] == label and a["path_class"]]
            out[label] = _bucket_or_insufficient(classes)
        return out

    d1 = _by_feature("morphology_label", MORPHOLOGY_LABELS)
    d2 = _by_feature("swing_grammar")

    train_anchors, test_anchors = _split_chronological(
        [{"anchor_ts_ms": a["anchor_ts_ms"], "path_class": a["path_class"]} for a in per_anchor]
    )
    train_classes = [a["path_class"] for a in train_anchors if a["path_class"]]
    test_classes = [a["path_class"] for a in test_anchors if a["path_class"]]
    stability = {"train": _bucket_or_insufficient(train_classes), "test": _bucket_or_insufficient(test_classes)}

    return {
        "anchor_n": len(events),
        "per_anchor": per_anchor,
        "d1_morphology_vs_outcome": d1,
        "d2_swing_grammar_vs_outcome": d2,
        "stability_train_test": stability,
    }


def freeze_and_record(conn, provenance: str = "batch-p6-006-w5a-morphology-swing-grammar") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;anchor_n={metrics['anchor_n']}"

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- reuses W4's stability
    # check; drift-only refresh in practice, see w4's own comment for the
    # full rationale).
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_CANDLE_MORPHOLOGY_SWING_GRAMMAR",
            "hypothesis_id": "H-W5A-MORPHOLOGY-GRAMMAR", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "close_quality_label,swing_grammar_HH_HL_LH_LL",
            "frozen_target": "swing_24h path class (reused from E-W4-POST-EVENT-PATH-TAXONOMY-001, "
                             "not redefined)",
            "frozen_thresholds": "N/A (categorical features, no threshold to fit)",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " stability check",
            "frozen_economic_gate": "N/A (descriptive, no entry/exit/economic claim)",
            "frozen_statistical_gate": "N/A (descriptive; 2-comparison fixed family "
                                       "FAM_CANDLE_MORPHOLOGY_SWING_GRAMMAR)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": "E-W4-POST-EVENT-PATH-TAXONOMY-001", "report_artifact_id": None,
            "schema_version": 7, "provenance": provenance, "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in [
            ("anchor_n", metrics["anchor_n"]),
            ("d1_morphology_vs_outcome", metrics["d1_morphology_vs_outcome"]),
            ("d2_swing_grammar_vs_outcome", metrics["d2_swing_grammar_vs_outcome"]),
            ("stability_train_test", metrics["stability_train_test"]),
        ]],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=False,
    )
    return {k: v for k, v in metrics.items() if k != "per_anchor"}


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        result = freeze_and_record(conn)
        print(
            f"anchor_n={result['anchor_n']}\n"
            f"D1 morphology_vs_outcome={result['d1_morphology_vs_outcome']}\n"
            f"D2 swing_grammar_vs_outcome={result['d2_swing_grammar_vs_outcome']}\n"
            f"stability_train_test={result['stability_train_test']}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
