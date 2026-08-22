"""Acceptance tests for the Phase 03B E-DER V1 → Eclipse candidate adapter.

Written before the adapter, against the criteria in
`docs/PHASE_03B_V1_CONTRACT_MAPPING.md`.

The tests that matter most are the ones about the seal. `make_event` in
`tools/e_der_v1_forward_shadow.py` returns a dict that already carries
`gross_return_bps` / `net_return_bps` as None, and `mature()` later mutates that
*same object* in place with realised returns. An adapter that keeps a reference,
queues the event, or publishes lazily would publish a sealed arm's outcome. So
the adapter must snapshot, and must refuse anything that is not a fresh
DETECTED event.

Run:
    .venv\\Scripts\\python.exe -m pytest tests/test_eclipse_alpha_adapter.py -q \\
        -p no:cacheprovider --basetemp=scratchpad/pytest_eclipse_alpha
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from pydantic import ValidationError

from integrations.eclipse_alpha import manifest
from integrations.eclipse_alpha.candidate_adapter import (
    OutcomeLeak,
    NotEligible,
    to_trade_candidate,
)

MINUTE_MS = 60_000
ANCHOR = 1_790_000_000_000  # after the integration boundary used in tests


def detected_event(**overrides) -> dict:
    """A DETECTED event with exactly the shape `make_event` produces at T0.

    Including the outcome keys present-and-None, because that is the real shape
    and the adapter has to cope with it without ever letting them through.
    """
    base_ms = (ANCHOR // MINUTE_MS) * MINUTE_MS + MINUTE_MS
    event = {
        "event": "DETECTED",
        "protocol": manifest.ARM_VERSION,
        "classification": "PROSPECTIVE_FORWARD",
        "event_id": f"E:ETHUSDT:{ANCHOR}",
        "symbol": "ETHUSDT",
        "anchor_ts": ANCHOR,
        "base_ms": base_ms,
        "entry_ms": base_ms + 31 * MINUTE_MS,
        "boundary_ms": base_ms + 240 * MINUTE_MS,
        "parent_id": f"P:ETHUSDT:{ANCHOR - 3 * MINUTE_MS}",
        "parent_ts_ms": ANCHOR - 3 * MINUTE_MS,
        "echo_id": "ECHO:ETHUSDT:1",
        "q_parent": 1.5,
        "q_echo": 2.25,
        "prior_stress_count": 3,
        "multiscale_votes": {"i1_v30": -1, "i3_v30": -1, "i5_v30": -1, "i10_v30": -1},
        "multiscale_vote_sum": -4,
        "status": "AWAITING_ENTRY",
        "entry_open": None,
        "boundary_open": None,
        "gross_return_bps": None,
        "net_return_bps": None,
        "cost_bps": 10.0,
        "data_quality_status": "PENDING_EXACT_OPENS",
        "universe_version": "2026-08-21",
        "code_sha": "deadbeef",
        "contract_sha": "ABC123",
        "cascade_id": f"CASCADE:{ANCHOR}",
        "product_cohort": "NATIVE_CRYPTO",
        "listing_age_days": 900.0,
        "session_state": "ALWAYS_OPEN",
        "paper_only": True,
        "real_order_sent": False,
        "created_at_utc": "2026-08-22T00:00:00Z",
    }
    event.update(overrides)
    return event


# ==========================================================================
# Criterion 3 — a schema-valid TradeCandidate
# ==========================================================================
def test_maps_a_detected_event():
    c = to_trade_candidate(detected_event())
    assert c.candidate_id == f"E:ETHUSDT:{ANCHOR}"
    assert c.arm == manifest.ARM
    assert c.arm_version == manifest.ARM_VERSION
    assert c.anchor_id == str(ANCHOR)
    assert c.symbol == "ETHUSDT"
    assert c.direction.value == "long"
    # boundary(+240m) minus entry(+31m), from the event's own frozen timing
    assert c.horizon_minutes == 209.0


def test_context_carries_only_whitelisted_t0_keys():
    c = to_trade_candidate(detected_event())
    assert set(c.context) <= set(manifest.CONTEXT_WHITELIST)
    assert c.context["q_echo"] == "2.25"
    assert c.context["prior_stress_count"] == "3"
    assert c.context["integration_contract"] == manifest.INTEGRATION_CONTRACT
    # everything is a string, because the schema says dict[str, str]
    assert all(isinstance(v, str) for v in c.context.values())


def test_fields_deliberately_excluded_never_reach_the_context():
    c = to_trade_candidate(detected_event())
    for banned in ("code_sha", "contract_sha", "cost_bps", "data_quality_status",
                   "real_order_sent", "listing_age_days", "created_at_utc",
                   "entry_open", "boundary_open", "gross_return_bps", "net_return_bps"):
        assert banned not in c.context


def test_sizing_fields_are_impossible():
    """The frozen schema has no size. Confirm it cannot be smuggled via context."""
    c = to_trade_candidate(detected_event())
    for absent in ("size", "quantity", "notional", "leverage", "venue"):
        assert not hasattr(c, absent)
        assert absent not in c.context
    with pytest.raises(ValidationError):
        type(c)(**{**c.model_dump(), "size": 1})


# ==========================================================================
# Criterion 4 — the seal. No outcome may cross, ever.
# ==========================================================================
@pytest.mark.parametrize("stage", ["ENTRY", "CLOSE", "ENTRY_UNAVAILABLE", "BOUNDARY_UNAVAILABLE"])
def test_refuses_a_matured_event(stage):
    """Exactly what `mature()` turns the object into."""
    with pytest.raises(NotEligible):
        to_trade_candidate(detected_event(event=stage, status="CLOSED"))


@pytest.mark.parametrize("status", ["OPEN", "CLOSED", "UNAVAILABLE"])
def test_refuses_a_non_awaiting_status(status):
    with pytest.raises(NotEligible):
        to_trade_candidate(detected_event(status=status))


@pytest.mark.parametrize(
    "field,value",
    [("gross_return_bps", 41.2), ("net_return_bps", 31.2),
     ("entry_open", 2500.0), ("boundary_open", 2530.0)],
)
def test_refuses_when_outcome_fields_are_populated(field, value):
    """A DETECTED event whose outcome field is set is a mutated event.

    This is the aliasing hazard: same dict, mutated in place by mature().
    """
    with pytest.raises(OutcomeLeak, match=field):
        to_trade_candidate(detected_event(**{field: value}))


def test_a_present_but_none_outcome_field_is_fine():
    """`make_event` always emits these as None — that must stay publishable."""
    c = to_trade_candidate(detected_event())
    assert c.candidate_id


def test_mutating_the_source_event_afterwards_cannot_change_the_candidate():
    """Criterion 7 — snapshot isolation.

    Reproduces run_cycle: the caller keeps the dict, mature() mutates it.
    """
    event = detected_event()
    c = to_trade_candidate(event)
    before = dict(c.context)

    event.update({"event": "CLOSE", "status": "CLOSED",
                  "gross_return_bps": 41.2, "net_return_bps": 31.2,
                  "entry_open": 2500.0, "boundary_open": 2530.0})
    event["multiscale_votes"]["i1_v30"] = 999

    assert dict(c.context) == before
    assert "gross_return_bps" not in c.context
    # the candidate holds no reference into the source event
    assert c.context is not event
    for value in event.values():
        assert c.context is not value


def test_the_candidate_context_is_mutable_in_process_but_the_bus_closes_it():
    """A residual, recorded rather than hidden.

    `TradeCandidate.context` is `dict[str, str]`, so the model is frozen but the
    dict is not — an in-process caller could mutate it after the adapter
    returns. That is a property of the frozen Phase 01-02 contract, which this
    phase must not change (§295).

    It does not reach the bus: at publish time the payload becomes
    `Event.payload`, which `freeze()` makes deeply immutable (platform P3). This
    test pins both halves so the gap cannot widen unnoticed.
    """
    from eclipse_shared.immutable import FrozenDict, freeze

    c = to_trade_candidate(detected_event())

    # in-process: mutable, and we are not permitted to fix that here
    c.context["scratch"] = "1"
    assert c.context["scratch"] == "1"

    # at the bus boundary: closed
    frozen = freeze(c.model_dump())
    assert isinstance(frozen, FrozenDict)
    with pytest.raises(TypeError):
        frozen["context"]["scratch"] = "2"


def test_adapter_source_never_names_an_outcome_field():
    """Enforced by parsing the module, not by reading it.

    Criterion 4 says "no code path reads an outcome field". A grep-style test
    is weak, but combined with the runtime tests above it closes the case where
    someone adds a convenience read later.
    """
    src = pathlib.Path("integrations/eclipse_alpha/candidate_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"gross_return_bps", "net_return_bps", "entry_open", "boundary_open"}
    # string literals are allowed only inside the refusal list; subscripting is not
    reads = {
        n.slice.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
        and isinstance(n.slice.value, str)
    }
    assert not (reads & forbidden), f"adapter subscripts an outcome field: {reads & forbidden}"


# ==========================================================================
# Criteria 5 & 6 — no transport, no platform internals
# ==========================================================================
def test_adapter_imports_nothing_forbidden():
    src = pathlib.Path("integrations/eclipse_alpha/candidate_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"eclipse_master_center", "nats", "socket", "asyncio", "requests",
                 "httpx", "sqlite3", "aiohttp", "urllib", "websockets"}
    assert not (imported & forbidden), f"forbidden import: {imported & forbidden}"


def test_the_adapter_does_not_touch_the_ledger():
    src = pathlib.Path("integrations/eclipse_alpha/candidate_adapter.py").read_text(encoding="utf-8")
    for token in ("events.jsonl", "LEDGER", "append_jsonl", "open(", "Path("):
        assert token not in src, f"adapter references {token!r}"


# ==========================================================================
# Criterion 8 — P2, no backfill
# ==========================================================================
def test_refuses_an_anchor_before_the_integration_boundary():
    stale = manifest.INTEGRATION_BOUNDARY_MS - MINUTE_MS
    with pytest.raises(NotEligible, match="boundary"):
        to_trade_candidate(detected_event(anchor_ts=stale, event_id=f"E:ETHUSDT:{stale}"))


def test_accepts_an_anchor_on_the_boundary():
    at = manifest.INTEGRATION_BOUNDARY_MS
    c = to_trade_candidate(detected_event(anchor_ts=at, event_id=f"E:ETHUSDT:{at}"))
    assert c.anchor_id == str(at)


# ==========================================================================
# Manifest — producer-owned, and not derived from anything volatile
# ==========================================================================
def test_arm_version_is_the_runners_own_frozen_protocol_constant():
    """Read the runner's PROTOCOL without importing it.

    Importing `tools.e_der_v1_forward_shadow` pulls in the whole runtime tree
    (requests, the source-support module, config loading). Parsing keeps this
    test honest and free of side effects.
    """
    tree = ast.parse(pathlib.Path("tools/e_der_v1_forward_shadow.py").read_text(encoding="utf-8"))
    protocol = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "PROTOCOL" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    )
    assert manifest.ARM_VERSION == protocol


def test_manifest_is_not_derived_from_git_or_filenames():
    tree = ast.parse(pathlib.Path("integrations/eclipse_alpha/manifest.py").read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "tools" not in imported, (
        "the manifest must not import the runner; equality is checked by parsing")
    assert not (imported & {"subprocess", "os", "pathlib", "git"}), (
        f"manifest imports something environment-derived: {imported}")

    # Names actually referenced in code — docstrings may legitimately say
    # "not derived from a Git SHA or a branch", and a prose grep would fail on
    # its own explanation.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("git_sha", "__file__", "check_output", "rev_parse"):
        assert banned not in names, f"manifest derives identity from {banned!r}"


def test_manifest_constants_are_immutable():
    with pytest.raises((AttributeError, TypeError)):
        manifest.CONTEXT_WHITELIST.add("gross_return_bps")
