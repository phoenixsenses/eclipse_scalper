"""BATCH-P3-001: immutable event identity + real-vs-proxy pooling guard tests.

Run: pytest tests/test_ami_identity_event_identity.py --basetemp <scratchpad> -p no:cacheprovider
"""
import pytest

from ami.identity.event_identity import (
    PooledPopulationViolation,
    SourceQuality,
    assert_not_pooled,
    generate_event_id,
)


def test_generate_event_id_is_deterministic():
    id1 = generate_event_id("ETHUSDT", "S34_STATE_MACHINE_V1", 1780858928177, "reports/shadow/ledger.jsonl")
    id2 = generate_event_id("ETHUSDT", "S34_STATE_MACHINE_V1", 1780858928177, "reports/shadow/ledger.jsonl")
    assert id1 == id2
    assert id1.startswith("EVT-")


def test_generate_event_id_differs_on_different_inputs():
    base = generate_event_id("ETHUSDT", "ROUTE_A", 1000, "src")
    diff_symbol = generate_event_id("BTCUSDT", "ROUTE_A", 1000, "src")
    diff_ts = generate_event_id("ETHUSDT", "ROUTE_A", 1001, "src")
    diff_family = generate_event_id("ETHUSDT", "ROUTE_B", 1000, "src")
    assert len({base, diff_symbol, diff_ts, diff_family}) == 4


def test_assert_not_pooled_allows_pure_real_population():
    assert_not_pooled([SourceQuality.REAL_LIQUIDATION, SourceQuality.REAL_LIQUIDATION])


def test_assert_not_pooled_allows_pure_proxy_population():
    assert_not_pooled([SourceQuality.PROXY_CASCADE_6H_GAP, SourceQuality.PROXY_OTHER])


def test_assert_not_pooled_raises_on_real_and_proxy_mix():
    with pytest.raises(PooledPopulationViolation, match="R-09"):
        assert_not_pooled([SourceQuality.REAL_LIQUIDATION, SourceQuality.PROXY_CASCADE_6H_GAP])


def test_assert_not_pooled_allows_real_and_unknown_mix():
    # UNKNOWN is an unclassified data-quality gap, not an asserted proxy --
    # it must not trigger the pooling guard by itself.
    assert_not_pooled([SourceQuality.REAL_LIQUIDATION, SourceQuality.UNKNOWN])


def test_assert_not_pooled_accepts_string_values():
    assert_not_pooled(["REAL_LIQUIDATION", "REAL_LIQUIDATION"])
    with pytest.raises(PooledPopulationViolation):
        assert_not_pooled(["REAL_LIQUIDATION", "PROXY_OTHER"])
