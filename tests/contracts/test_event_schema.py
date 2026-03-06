from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.microphys.contracts.events import (
    EVENT_SCHEMA_VERSION,
    FillEvent,
    OrderIntent,
    event_from_dict,
    validate_event,
    validate_event_sequence,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def test_event_from_dict_normalizes_side_and_types() -> None:
    raw = {
        "event_type": "order_intent",
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": "evt1",
        "ts_ms": 1772000000000,
        "source": "test",
        "symbol": "ETHUSDT",
        "order_id": "ord1",
        "client_order_id": "cid1",
        "side": "long",
        "order_type": "limit",
        "tif": "gtc",
        "qty": "0.01",
        "limit_price": "2100.5",
    }
    ev = event_from_dict(raw)
    assert isinstance(ev, OrderIntent)
    assert ev.side == "BUY"
    assert ev.order_type == "LIMIT"
    assert ev.tif == "GTC"
    assert abs(float(ev.qty) - 0.01) < 1e-12
    assert abs(float(ev.limit_price or 0.0) - 2100.5) < 1e-12


def test_fill_event_validate_happy_path() -> None:
    ev = FillEvent(
        event_type="fill",
        schema_version=EVENT_SCHEMA_VERSION,
        event_id="fill1",
        ts_ms=1772000001000,
        source="test",
        symbol="ETHUSDT",
        order_id="ord1",
        client_order_id="cid1",
        side="BUY",
        fill_qty=0.01,
        fill_price=2100.0,
        cumulative_qty=0.01,
        remaining_qty=0.0,
        liquidity="maker",
        fee_bps=0.5,
        effective_cost_bps=1.1,
    )
    assert validate_event(ev) == []


def test_golden_fixture_valid_sequence() -> None:
    fixture = Path("eclipse_scalper/tests/fixtures/execution/golden_execution_events.jsonl")
    raws = _load_jsonl(fixture)
    events = [event_from_dict(r) for r in raws]
    all_errs = [e for ev in events for e in validate_event(ev)]
    seq_errs = validate_event_sequence(events)
    assert all_errs == []
    assert seq_errs == []


def test_golden_fixture_invalid_has_failures() -> None:
    fixture = Path("eclipse_scalper/tests/fixtures/execution/golden_execution_events_invalid.jsonl")
    raws = _load_jsonl(fixture)
    events = [event_from_dict(r) for r in raws]
    all_errs = [e for ev in events for e in validate_event(ev)]
    seq_errs = validate_event_sequence(events)
    assert len(all_errs) > 0
    assert len(seq_errs) > 0
    assert any("event_id_missing" in x for x in all_errs)
    assert any("intent_after_ACKED" in x for x in seq_errs)

