"""BATCH-P3-003: cycle-grouped chronological split + purge/embargo tests.

Run: pytest tests/test_ami_identity_split_utils.py --basetemp <scratchpad> -p no:cacheprovider
"""
import pytest

from ami.identity.split_utils import chronological_group_split

DAY_MS = 86_400_000


def _mk_records(n_groups, per_group=3, spacing_days=1):
    """n_groups groups, each with `per_group` records at the same ts (simulating
    multiple ledger rows sharing one cycle), groups spaced `spacing_days` apart."""
    records = []
    for g in range(n_groups):
        ts = g * spacing_days * DAY_MS
        for i in range(per_group):
            records.append({"group": f"G{g}", "ts": ts, "idx": i})
    return records


def _group_key(r):
    return r["group"]


def _ts(r):
    return r["ts"]


def test_no_group_split_across_sets():
    records = _mk_records(20)
    result = chronological_group_split(records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2)
    train_groups = {r["group"] for r in result.train}
    val_groups = {r["group"] for r in result.val}
    test_groups = {r["group"] for r in result.test}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_chronological_order_preserved():
    records = _mk_records(20)
    result = chronological_group_split(records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2)
    assert max(r["ts"] for r in result.train) <= min(r["ts"] for r in result.val)
    assert max(r["ts"] for r in result.val) <= min(r["ts"] for r in result.test)


def test_every_record_accounted_for_without_purge_embargo():
    records = _mk_records(20)
    result = chronological_group_split(records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2)
    total = len(result.train) + len(result.val) + len(result.test) + len(result.purged) + len(result.embargoed)
    assert total == len(records)
    assert result.purged == []
    assert result.embargoed == []


def test_purge_removes_records_near_boundaries():
    records = _mk_records(20)
    no_purge = chronological_group_split(records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2)
    with_purge = chronological_group_split(
        records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2, purge_seconds=int(DAY_MS / 1000) * 2
    )
    assert len(with_purge.purged) > 0
    assert len(with_purge.train) < len(no_purge.train)
    total = (len(with_purge.train) + len(with_purge.val) + len(with_purge.test)
             + len(with_purge.purged) + len(with_purge.embargoed))
    assert total == len(records)


def test_embargo_creates_dead_zone_after_train():
    records = _mk_records(20)
    result = chronological_group_split(
        records, _group_key, _ts, val_ratio=0.2, test_ratio=0.2, embargo_seconds=int(DAY_MS / 1000) * 2
    )
    assert len(result.embargoed) > 0
    total = len(result.train) + len(result.val) + len(result.test) + len(result.purged) + len(result.embargoed)
    assert total == len(records)


def test_raises_on_invalid_ratios():
    records = _mk_records(10)
    with pytest.raises(ValueError):
        chronological_group_split(records, _group_key, _ts, val_ratio=0.6, test_ratio=0.6)


def test_raises_when_too_few_groups():
    records = _mk_records(1)
    with pytest.raises(ValueError):
        chronological_group_split(records, _group_key, _ts, val_ratio=0.4, test_ratio=0.4)


def test_empty_records_returns_empty_result():
    result = chronological_group_split([], _group_key, _ts)
    assert result.train == [] and result.val == [] and result.test == []
