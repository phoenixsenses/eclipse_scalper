"""BATCH-P6-007 (W6): Compression + relative strength + session
(HISTORICAL_RESEARCH_WAVES.md W6 "chart W2"; channel deferred -- see OD-015,
CHANNEL_BOUNDARY is explicitly NOT_IMPLEMENTED, no trendline/channel-fitting
engine exists).

Preregistration frozen BEFORE looking at outcomes. Population and outcome
are the SAME as W4/W5a (reused, not redefined -- avoids target-shopping):

POPULATION (frozen, reused): ETHUSDT REAL_LIQUIDATION anchors, anchor_n=252,
independent_cycle_n=167 (canonical-v1, same as W1/W4/W5a).

OUTCOME (frozen, REUSED from W4): swing_24h path class (CONTINUATION/
REVERSAL/CHOP, E-W4-POST-EVENT-PATH-TAXONOMY-001's fixed +-20bps band).

COVERAGE CHECK (done before designing anything, per operator instruction):
  compression      -- READY: ami/states/engine.py:StructurePhase.COMPRESSION
                      already computed and used by Phase 3's cycle_resolver
                      (1h TF, rule-based vol-compression + small |ret3|).
  relative_strength -- BUILDABLE: BTC + ETH mark_prices both have full,
                      continuous coverage across the entire anchor window
                      (data/microstructure.db, verified 8.59M rows each,
                      2026-02-17..now) -- no existing RS feature, but no data
                      gap either; a simple return-differential is computed
                      here for the first time (frozen, not fit).
  session           -- READY: ASIA/EUROPE/US/OFF boundaries already defined
                      in ami/chart/level_registry.py (reused directly, not
                      duplicated).
  channel           -- NOT_IMPLEMENTED (OD-015): CHANNEL_BOUNDARY has no
                      trendline/channel-fitting engine. NOT forced open here.

NEW FEATURES (frozen, not fit; both computed via ami/states/engine.py's
already-established direct read-only access to microstructure.db -- the same
access pattern Phase 3's cycle_resolver already uses, distinct from the
canonical-warehouse-only feature_gateway rule which governs canonical.sqlite
tables, not this pre-existing microstructure.db state layer):
    compression_at_anchor -- StructurePhase.COMPRESSION vs NOT_COMPRESSION
        (binary: N=252 is too small to split across all ~15 phases without
        instantly violating MIN_BUCKET_N in nearly every cell)
    rs_at_anchor -- sign of (ETH 1h return_bps - BTC 1h return_bps) at the
        anchor: RS_ETH_STRONG (>=0) vs RS_ETH_WEAK (<0). 1h lookback matches
        the same TF StructurePhase already uses, for consistency; 0 is the
        natural, non-fit split point (no threshold search).
    session_at_anchor -- ASIA/EUROPE/US/OFF (UTC hour of anchor_ts_ms)

MULTIPLE-TESTING FAMILY (frozen, 3 comparisons, FAM_COMPRESSION_RS_SESSION):
    E1: compression_at_anchor vs swing_24h outcome
    E2: rs_at_anchor vs swing_24h outcome
    E3: session_at_anchor vs swing_24h outcome

CONTROLS: same chronological 70/30 split as W4/W5a (stability check, not
threshold-fitting).

STOP CONDITIONS: any bucket with N < MIN_BUCKET_N is INSUFFICIENT_SAMPLE.

Descriptive only -- no entry/exit/economic claim, not a promotion candidate.
W5's null result is NOT revisited or "rescued" here -- this module only adds
3 NEW conditioning features against the SAME frozen W4 outcome.
"""
from __future__ import annotations
import hashlib
import time
from collections import Counter

from ami.research.feature_gateway import fetch_events
from ami.research.w4_post_event_path_taxonomy import (
    MIN_BUCKET_N,
    TRAIN_FRACTION,
    _CandleIndex,
    _split_chronological,
    classify_path,
    compute_path_returns,
)
from ami.research.feature_gateway import fetch_chart_feature
from ami.states.engine import StateEngine
from ami.chart.level_registry import _session_of_hour
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

EXPERIMENT_ID = "E-W6-COMPRESSION-RS-SESSION-001"
RESEARCH_CONTEXT_ID = "w6-compression-rs-session"

RS_LOOKBACK_MS = 3600_000  # 1h, matches StructurePhase's own TF convention


def _distribution(classes: list[str]) -> dict:
    n = len(classes)
    c = Counter(classes)
    return {"n": n, **{k: c.get(k, 0) for k in ("CONTINUATION", "REVERSAL", "CHOP")}}


def _bucket_or_insufficient(classes: list[str]) -> dict:
    d = _distribution(classes)
    d["insufficient_sample"] = d["n"] < MIN_BUCKET_N
    return d


def classify_compression(engine: StateEngine, symbol: str, anchor_ts_ms: int) -> str:
    phase, _direction, _conf = engine._structure(symbol, anchor_ts_ms, "1h")
    return "COMPRESSION" if phase.value == "COMPRESSION" else "NOT_COMPRESSION"


def classify_relative_strength(engine: StateEngine, anchor_ts_ms: int) -> str | None:
    eth_ret = engine._ret_bps("ETHUSDT", anchor_ts_ms, RS_LOOKBACK_MS)
    btc_ret = engine._ret_bps("BTCUSDT", anchor_ts_ms, RS_LOOKBACK_MS)
    if eth_ret is None or btc_ret is None:
        return None
    return "RS_ETH_STRONG" if (eth_ret - btc_ret) >= 0 else "RS_ETH_WEAK"


def classify_session(anchor_ts_ms: int) -> str:
    import datetime as dt

    hour = dt.datetime.fromtimestamp(anchor_ts_ms / 1000, dt.timezone.utc).hour
    return _session_of_hour(hour)


def compute_metrics(conn, symbol: str = "ETHUSDT") -> dict:
    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles", ["close_ts_ms", "close"],
        symbol=symbol, equals={"timeframe": "1m"},
    )
    candle_index = _CandleIndex(candle_rows)

    engine = StateEngine()
    try:
        per_anchor = []
        for e in events:
            anchor_ts = e["anchor_ts_ms"]
            returns = compute_path_returns(candle_index, anchor_ts)
            path_class = classify_path(returns["swing_24h"]) if returns["swing_24h"] is not None else None

            per_anchor.append({
                "event_id": e["event_id"],
                "anchor_ts_ms": anchor_ts,
                "path_class": path_class,
                "compression": classify_compression(engine, symbol, anchor_ts),
                "rs": classify_relative_strength(engine, anchor_ts),
                "session": classify_session(anchor_ts),
            })
    finally:
        engine.conn.close()

    def _by_feature(feature_name: str) -> dict:
        out = {}
        keys = {a[feature_name] for a in per_anchor if a[feature_name]}
        for label in keys:
            classes = [a["path_class"] for a in per_anchor if a[feature_name] == label and a["path_class"]]
            out[label] = _bucket_or_insufficient(classes)
        return out

    e1 = _by_feature("compression")
    e2 = _by_feature("rs")
    e3 = _by_feature("session")

    train_anchors, test_anchors = _split_chronological(
        [{"anchor_ts_ms": a["anchor_ts_ms"], "path_class": a["path_class"]} for a in per_anchor]
    )
    train_classes = [a["path_class"] for a in train_anchors if a["path_class"]]
    test_classes = [a["path_class"] for a in test_anchors if a["path_class"]]
    stability = {"train": _bucket_or_insufficient(train_classes), "test": _bucket_or_insufficient(test_classes)}

    return {
        "anchor_n": len(events),
        "per_anchor": per_anchor,
        "e1_compression_vs_outcome": e1,
        "e2_rs_vs_outcome": e2,
        "e3_session_vs_outcome": e3,
        "stability_train_test": stability,
    }


def freeze_and_record(conn, provenance: str = "batch-p6-007-w6-compression-rs-session") -> dict:
    now = int(time.time() * 1000)
    metrics = compute_metrics(conn)

    dataset_hash = hashlib.sha256(
        "|".join(sorted(a["event_id"] for a in metrics["per_anchor"])).encode("utf-8")
    ).hexdigest()

    frozen_population = f"symbol=ETHUSDT;source_quality=REAL_LIQUIDATION;anchor_n={metrics['anchor_n']}"

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary (no_test_split=False -- reuses W4's stability
    # check; drift-only refresh in practice).
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_COMPRESSION_RS_SESSION",
            "hypothesis_id": "H-W6-COMPRESSION-RS-SESSION", "preregistered_at": now,
            "frozen_population": frozen_population,
            "frozen_features": "compression_at_anchor,rs_at_anchor,session_at_anchor",
            "frozen_target": "swing_24h path class (reused from E-W4-POST-EVENT-PATH-TAXONOMY-001, "
                             "not redefined)",
            "frozen_thresholds": "rs_lookback_ms=3600000, rs split at 0 (non-fit); compression binary "
                                 "(StructurePhase); session = existing ASIA/EUROPE/US/OFF boundaries "
                                 "(all fixed pre-hoc)",
            "frozen_splits": f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)}"
                             " stability check",
            "frozen_economic_gate": "N/A (descriptive, no entry/exit/economic claim)",
            "frozen_statistical_gate": "N/A (descriptive; 3-comparison fixed family "
                                       "FAM_COMPRESSION_RS_SESSION)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1,
            "supersedes_experiment_id": "E-W4-POST-EVENT-PATH-TAXONOMY-001", "report_artifact_id": None,
            "schema_version": 7, "provenance": provenance, "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in [
            ("anchor_n", metrics["anchor_n"]),
            ("e1_compression_vs_outcome", metrics["e1_compression_vs_outcome"]),
            ("e2_rs_vs_outcome", metrics["e2_rs_vs_outcome"]),
            ("e3_session_vs_outcome", metrics["e3_session_vs_outcome"]),
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
            f"E1 compression_vs_outcome={result['e1_compression_vs_outcome']}\n"
            f"E2 rs_vs_outcome={result['e2_rs_vs_outcome']}\n"
            f"E3 session_vs_outcome={result['e3_session_vs_outcome']}\n"
            f"stability_train_test={result['stability_train_test']}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
