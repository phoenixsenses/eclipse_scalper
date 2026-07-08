"""BATCH-P3-001: shadow ledger -> ami_events ingestion tests (read-only source).

Run: pytest tests/test_ami_identity_shadow_ledger_ingest.py --basetemp <scratchpad> -p no:cacheprovider
"""
import json

import pytest

from ami.identity.event_identity import SourceQuality, generate_event_id
from ami.identity.shadow_ledger_ingest import (
    DEFAULT_LEDGER_PATH,
    DuplicateOpenConflict,
    _load_logical_trades,
    parse_shadow_ledger,
    seed,
)
from ami.warehouse.schema import connect, init_schema


def test_parse_shadow_ledger_finds_real_events():
    events = parse_shadow_ledger()
    assert len(events) > 0
    assert all(e["source_quality"] == SourceQuality.REAL_LIQUIDATION.value for e in events)
    assert all(e["symbol"] == "ETHUSDT" for e in events)


def test_parse_shadow_ledger_no_fabricated_fields():
    events = parse_shadow_ledger()
    for e in events:
        assert e["venue"] is None
        assert e["liquidation_side"] is None
        assert e["feature_available_ts_ms"] is None


def test_parse_shadow_ledger_censor_status_is_honest():
    events = parse_shadow_ledger()
    statuses = {e["censor_status"] for e in events}
    assert statuses.issubset({"COMPLETED", "RIGHT_CENSORED"})
    # COMPLETED requires every logical trade sharing the anchor to have closed,
    # so an end timestamp must exist. RIGHT_CENSORED may still have a partial
    # end timestamp (some but not all attached trades closed) -- only the
    # COMPLETED -> event_end_ts_ms present direction is guaranteed.
    completed = [e for e in events if e["censor_status"] == "COMPLETED"]
    assert all(e["event_end_ts_ms"] is not None for e in completed)


def test_dedup_collapses_shared_anchors_by_event_count():
    events = parse_shadow_ledger()
    trades = _load_logical_trades(DEFAULT_LEDGER_PATH)
    assert sum(e["event_count"] for e in events) == len(trades)
    assert any(e["event_count"] > 1 for e in events)  # at least one anchor shared by multiple routes


def test_event_ids_are_deterministic_across_parses():
    ids1 = {e["event_id"] for e in parse_shadow_ledger()}
    ids2 = {e["event_id"] for e in parse_shadow_ledger()}
    assert ids1 == ids2
    assert len(ids1) == len(parse_shadow_ledger())  # no accidental collisions


def test_seed_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    n1 = seed(conn)
    count1 = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    n2 = seed(conn)
    count2 = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
    conn.close()
    assert n1 == n2 == count1 == count2


def test_seeded_events_all_real_liquidation(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    seed(conn)
    rows = conn.execute("SELECT DISTINCT source_quality FROM ami_events").fetchall()
    conn.close()
    assert rows == [("REAL_LIQUIDATION",)]


def test_duplicate_open_with_identical_fields_is_harmless(tmp_path):
    # FABLE-REVIEW-A F2: a duplicate OPEN with identical critical fields
    # (matches the 13 real cases in the current ledger) must not raise.
    ledger = tmp_path / "ledger.jsonl"
    row = {"id": "X:1:LS", "rule_name": "S34_STATE_MACHINE_V1_ETH_SELL", "anchor_ts_ms": 1000,
           "entry_ts_ms": 1000, "signal": "LONG_SILENCE", "running_notional": 1.0, "event": "OPEN"}
    ledger.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    trades = _load_logical_trades(ledger)
    assert len(trades) == 1


def test_duplicate_open_with_divergent_field_raises(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    row1 = {"id": "X:1:LS", "rule_name": "S34_STATE_MACHINE_V1_ETH_SELL", "anchor_ts_ms": 1000,
            "entry_ts_ms": 1000, "signal": "LONG_SILENCE", "event": "OPEN"}
    row2 = dict(row1, anchor_ts_ms=2000)  # diverges on a critical field
    ledger.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
    with pytest.raises(DuplicateOpenConflict, match="anchor_ts_ms"):
        _load_logical_trades(ledger)


def test_event_family_change_mints_new_id_never_mutates_in_place():
    # FABLE-REVIEW-A F1: event_id is a hash of (symbol, event_family, anchor_ts_ms,
    # source_artifact_id) -- there is no code path that could change event_family
    # for an existing row without also producing a different event_id.
    id_a = generate_event_id("ETHUSDT", "FAMILY_A", 1000, "src")
    id_b = generate_event_id("ETHUSDT", "FAMILY_B", 1000, "src")
    assert id_a != id_b
