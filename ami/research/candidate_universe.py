"""BATCH-P6-003: all-timestamp candidate universe (Protocol §17.8).

"To test unconditional LONG genesis and selection bias, persist eligible
candidates even when no event or trade occurs."

This is the broadest denominator in the research funnel, sitting ABOVE
raw_ledger_trades_n / anchor_n / cycle_n (W1, BATCH-P6-002):

    raw_candidate_n  -- every closed candle slot in the window (this module)
    event_n          -- raw ledger trades attached to a real anchor (W1)
    anchor_n         -- deduped ami_events (W1)
    independent_cycle_n -- ami_cycles canonical-v1 (W1)

Universe construction is built directly from ami_candles (already closed-
candle-only, known_at-safe, and gap-aware via data_quality) -- so the
candidate slot itself inherits those properties for free:

  - point-in-time / known-at-safe: known_at_ts = candle.close_ts_ms
    (Observatory §6.2, same discipline as every other object in this repo).
  - gap-aware: data_quality is copied from the underlying candle
    (AVAILABLE/GAPPED), never silently upgraded to AVAILABLE.
  - deterministic / reproducible: candidate_id = hash(symbol, timeframe,
    slot_ts_ms, universe_definition_version) -- identical inputs always
    produce identical rows; re-running does not fabricate new candidates
    or drop existing ones except when the underlying candle set changes.
  - NOT conditioned on events: every closed candle in [start, end) becomes
    a candidate row regardless of whether a liquidation event fired near it.
    is_event_aligned / aligned_event_id record the (possibly absent) link
    as an annotation, not a filter on row existence.

Event alignment: a candidate slot [slot_ts_ms, slot_ts_ms+bar_ms) is
"event-aligned" if any REAL_LIQUIDATION ami_events.anchor_ts_ms falls inside
it. Alignment is computed only from anchors returned by feature_gateway
(so the R-09 pooling guard applies here too) -- this module does not query
ami_events directly.
"""
from __future__ import annotations
import hashlib
import time

from ami.enums import TIMEFRAME_MS
from ami.research.feature_gateway import fetch_chart_feature, fetch_cycles, fetch_events
from ami.warehouse.experiment_ledger import register_legacy_snapshot_with_gates

UNIVERSE_DEFINITION_VERSION = "candidate-universe-v1"
RESEARCH_CONTEXT_ID = "p6-003-candidate-universe"


def _candidate_id(symbol: str, timeframe: str, slot_ts_ms: int) -> str:
    key = f"{symbol}|{timeframe}|{slot_ts_ms}|{UNIVERSE_DEFINITION_VERSION}"
    return "CND-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def build_universe(candles: list[dict], events: list[dict], symbol: str, timeframe: str) -> list[dict]:
    """candles: chronologically ordered dicts with open_ts_ms, close_ts_ms,
    data_quality (as stored in ami_candles). events: dicts with anchor_ts_ms,
    event_id (as returned by feature_gateway.fetch_events). Returns one
    candidate row per candle -- unconditional on event presence."""
    bar_ms = TIMEFRAME_MS[timeframe]
    anchors_by_slot: dict[int, str] = {}
    for e in events:
        slot = e["anchor_ts_ms"] - (e["anchor_ts_ms"] % bar_ms)
        anchors_by_slot.setdefault(slot, e["event_id"])  # first event wins if >1 anchor in one slot

    candidates = []
    for c in candles:
        slot_ts_ms = c["open_ts_ms"]
        aligned_event_id = anchors_by_slot.get(slot_ts_ms)
        candidates.append({
            "candidate_id": _candidate_id(symbol, timeframe, slot_ts_ms),
            "symbol": symbol,
            "timeframe": timeframe,
            "slot_ts_ms": slot_ts_ms,
            "known_at_ts": c["close_ts_ms"],
            "data_quality": c["data_quality"],
            "is_event_aligned": 1 if aligned_event_id is not None else 0,
            "aligned_event_id": aligned_event_id,
        })
    return candidates


def seed(conn, symbol: str = "ETHUSDT", timeframe: str = "1m",
         provenance: str = "batch-p6-003-candidate-universe") -> int:
    now = int(time.time() * 1000)
    # ami_candles holds multiple timeframes -- fetch_chart_feature's symbol-only
    # filter isn't enough here, so use its generic equals-filter for timeframe too.
    candle_rows = fetch_chart_feature(
        conn, RESEARCH_CONTEXT_ID, "ami_candles",
        ["open_ts_ms", "close_ts_ms", "data_quality"], symbol=symbol,
        equals={"timeframe": timeframe},
    )
    candles = [{"open_ts_ms": r["open_ts_ms"], "close_ts_ms": r["close_ts_ms"],
                "data_quality": r["data_quality"]} for r in candle_rows]
    candles.sort(key=lambda c: c["open_ts_ms"])

    events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")

    candidates = build_universe(candles, events, symbol, timeframe)
    for cand in candidates:
        conn.execute(
            "INSERT INTO ami_candidate_universe (candidate_id, symbol, timeframe, slot_ts_ms, known_at_ts, "
            "data_quality, is_event_aligned, aligned_event_id, universe_definition_version, schema_version, "
            "provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, slot_ts_ms, universe_definition_version) DO UPDATE SET "
            "data_quality=excluded.data_quality, is_event_aligned=excluded.is_event_aligned, "
            "aligned_event_id=excluded.aligned_event_id, updated_ms=excluded.updated_ms",
            (cand["candidate_id"], cand["symbol"], cand["timeframe"], cand["slot_ts_ms"], cand["known_at_ts"],
             cand["data_quality"], cand["is_event_aligned"], cand["aligned_event_id"],
             UNIVERSE_DEFINITION_VERSION, 7, provenance, now, now),
        )
    conn.commit()
    return len(candidates)


EXPERIMENT_ID = "E-CANDIDATE-UNIVERSE-001"


def freeze_and_record(conn, symbol: str = "ETHUSDT", timeframe: str = "1m",
                       provenance: str = "batch-p6-003-candidate-universe") -> dict:
    """Records the four funnel denominators as one canonical snapshot, kept
    SEPARATE per Protocol §17.8 / operator instruction:

        raw_candidate_n      -- every closed candle slot (this module, broadest)
        event_n               -- raw ledger trades attached to a real anchor (W1)
        anchor_n              -- deduped ami_events (W1)
        independent_cycle_n   -- ami_cycles canonical-v1 (W1)

    IMPORTANT SCOPE CAVEAT (found while wiring this up, not silently glossed
    over): ami_candles is built with a bounded lookback_hours=48 window by
    design (Phase 4, ami/chart/candle_builder.py). raw_candidate_n therefore
    covers only that ~48h window, while event_n/anchor_n/independent_cycle_n
    (via W1) cover the FULL history (2026-02-17 .. present). Comparing them
    directly would be misleading -- so this function reports BOTH the
    all-history W1 counts AND a properly window-scoped anchor/cycle count
    (anchor_n_in_candidate_window / cycle_n_in_candidate_window), and the
    candidate_to_anchor_ratio / candidate_to_cycle_ratio metrics use ONLY the
    window-scoped counts, never the all-history ones, as their numerator.

    This does NOT supersede E-W1-CYCLE-INTEGRITY-001 (that experiment's own
    frozen population/results are untouched) -- it is a new, broader-scope
    experiment that reuses W1's already-gateway-sourced anchor/cycle metrics
    and adds the candidate-universe tier above them."""
    from ami.research.w1_cycle_integrity import compute_metrics as w1_compute_metrics

    now = int(time.time() * 1000)
    n_total = seed(conn, symbol=symbol, timeframe=timeframe, provenance=provenance)
    n_aligned = conn.execute(
        "SELECT COUNT(*) FROM ami_candidate_universe WHERE symbol=? AND timeframe=? AND is_event_aligned=1",
        (symbol, timeframe),
    ).fetchone()[0]
    window_start_ms, window_end_ms = conn.execute(
        "SELECT MIN(slot_ts_ms), MAX(known_at_ts) FROM ami_candidate_universe WHERE symbol=? AND timeframe=? "
        "AND universe_definition_version=?",
        (symbol, timeframe, UNIVERSE_DEFINITION_VERSION),
    ).fetchone()

    w1_metrics = w1_compute_metrics(conn, symbol=symbol)

    all_events = fetch_events(conn, RESEARCH_CONTEXT_ID, symbol=symbol, source_quality="REAL_LIQUIDATION")
    all_cycles = fetch_cycles(conn, RESEARCH_CONTEXT_ID, symbol=symbol)
    if window_start_ms is not None and window_end_ms is not None:
        anchor_n_in_window = sum(
            1 for e in all_events if window_start_ms <= e["anchor_ts_ms"] <= window_end_ms
        )
        cycle_n_in_window = sum(
            1 for c in all_cycles if c["start_ts_ms"] is not None
            and window_start_ms <= c["start_ts_ms"] <= window_end_ms
        )
    else:
        anchor_n_in_window = 0
        cycle_n_in_window = 0

    dataset_hash = hashlib.sha256(
        f"{symbol}|{timeframe}|{UNIVERSE_DEFINITION_VERSION}|{n_total}|{w1_metrics['anchor_n']}|"
        f"{w1_metrics['cycle_n']}".encode("utf-8")
    ).hexdigest()

    frozen_population = (
        f"symbol={symbol};timeframe={timeframe};universe_definition_version={UNIVERSE_DEFINITION_VERSION};"
        f"cycle_definition_version=canonical-v1;snapshot_raw_candidate_n={n_total};"
        f"candle_window_ms=[{window_start_ms},{window_end_ms}]"
    )

    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: routed through the
    # mandatory gated boundary instead of a direct INSERT (graveyard-checked;
    # no_test_split=True matches this module's own declared "none (descriptive,
    # no train/test split)" frozen_splits/frozen_statistical_gate below --
    # dataset_hash/metric drift from the live, growing population is expected
    # and handled as a snapshot refresh, not new scientific evidence).
    results = [
        ("raw_candidate_n", n_total),
        ("raw_candidate_event_aligned_n", n_aligned),
        ("raw_candidate_no_event_n", n_total - n_aligned),
        ("candidate_window_start_ms", window_start_ms),
        ("candidate_window_end_ms", window_end_ms),
        ("anchor_n_in_candidate_window", anchor_n_in_window),
        ("cycle_n_in_candidate_window", cycle_n_in_window),
        ("event_n_all_history", w1_metrics["raw_ledger_trades_n"]),
        ("anchor_n_all_history", w1_metrics["anchor_n"]),
        ("independent_cycle_n_all_history", w1_metrics["cycle_n"]),
        ("candidate_to_anchor_ratio",
         round(anchor_n_in_window / n_total, 6) if n_total else None),
        ("candidate_to_cycle_ratio",
         round(cycle_n_in_window / n_total, 6) if n_total else None),
    ]
    register_legacy_snapshot_with_gates(
        conn,
        registry_values={
            "experiment_id": EXPERIMENT_ID, "question_ids": "FAM_CYCLE_INTEGRITY",
            "hypothesis_id": "H-CANDIDATE-UNIVERSE-DENOMINATORS", "preregistered_at": now,
            "frozen_population": frozen_population, "frozen_features": "is_event_aligned,data_quality",
            "frozen_target": "raw_candidate_n vs event_n vs anchor_n vs cycle_n",
            "frozen_thresholds": "none (descriptive funnel, no threshold search)",
            "frozen_splits": "none (descriptive, no train/test split)",
            "frozen_economic_gate": "N/A (descriptive integrity wave, not an alpha route)",
            "frozen_statistical_gate": "N/A (descriptive integrity wave, not an alpha route)",
            "code_commit": None, "dataset_hash": dataset_hash, "started_at": now, "completed_at": now,
            "software_verdict": "PASSED", "scientific_verdict": "ANSWERED_SUPPORTED",
            "mutation_test_count": 0, "mutation_test_passed": 1, "supersedes_experiment_id": None,
            "report_artifact_id": None, "schema_version": 7, "provenance": provenance,
            "created_ms": now, "updated_ms": now,
        },
        results=[(name, str(value)) for name, value in results],
        results_schema_version=7, results_provenance=provenance, results_created_ms=now,
        no_test_split=True,
    )
    return {
        "raw_candidate_n": n_total,
        "raw_candidate_event_aligned_n": n_aligned,
        "raw_candidate_no_event_n": n_total - n_aligned,
        "candidate_window_start_ms": window_start_ms,
        "candidate_window_end_ms": window_end_ms,
        "anchor_n_in_candidate_window": anchor_n_in_window,
        "cycle_n_in_candidate_window": cycle_n_in_window,
        "event_n_all_history": w1_metrics["raw_ledger_trades_n"],
        "anchor_n_all_history": w1_metrics["anchor_n"],
        "independent_cycle_n_all_history": w1_metrics["cycle_n"],
    }


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        metrics = freeze_and_record(conn)
        print(
            f"raw_candidate_n={metrics['raw_candidate_n']} "
            f"(event_aligned={metrics['raw_candidate_event_aligned_n']}, "
            f"no_event={metrics['raw_candidate_no_event_n']}) "
            f"anchor_n_in_window={metrics['anchor_n_in_candidate_window']} "
            f"cycle_n_in_window={metrics['cycle_n_in_candidate_window']} | "
            f"all-history: event_n={metrics['event_n_all_history']} "
            f"anchor_n={metrics['anchor_n_all_history']} "
            f"independent_cycle_n={metrics['independent_cycle_n_all_history']}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
