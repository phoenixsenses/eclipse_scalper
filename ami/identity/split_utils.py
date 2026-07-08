"""BATCH-P3-003: cycle-grouped chronological split with purge/embargo.

Generic and independent of which cycle_definition_version eventually becomes
canonical (OD-003 is not resolved by this module). Callers supply
group_key_fn (a cooldown-sensitivity key today; a canonical cycle_id once
OD-003 is approved) so the same utility serves both. Guarantees whitepaper
§51.1: "the same cycle must never be split across train and validation" --
every record sharing a group key lands in exactly one of train/val/test.

Purge (boundary-distance proxy): removes records within `purge_seconds` of
any split boundary. This is a simplified symmetric-distance proxy, not a
per-record label-horizon-aware purge -- callers whose records carry a known
outcome/label horizon should pass purge_seconds >= their maximum horizon.

Embargo: a one-directional dead zone of `embargo_seconds` immediately AFTER
each boundary, at the START of val and test, blocking records too close to
the preceding split (protects against forward-leaking information from the
split that immediately precedes it).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class SplitResult:
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)
    purged: list = field(default_factory=list)
    embargoed: list = field(default_factory=list)


def chronological_group_split(
    records: list,
    group_key_fn: Callable,
    timestamp_fn: Callable[..., int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_seconds: int = 0,
    embargo_seconds: int = 0,
) -> SplitResult:
    if not records:
        return SplitResult()
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val_ratio/test_ratio must each be in (0,1) and sum to less than 1")

    groups: dict = {}
    for r in records:
        groups.setdefault(group_key_fn(r), []).append(r)

    group_order = sorted(groups.keys(), key=lambda k: min(timestamp_fn(r) for r in groups[k]))
    n = len(group_order)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough groups to satisfy val_ratio/test_ratio with at least one train group")

    train_keys = group_order[:n_train]
    val_keys = group_order[n_train:n_train + n_val]
    test_keys = group_order[n_train + n_val:]

    train = [r for k in train_keys for r in groups[k]]
    val = [r for k in val_keys for r in groups[k]]
    test = [r for k in test_keys for r in groups[k]]

    train_end_ts = max(timestamp_fn(r) for r in train)
    val_end_ts = max(timestamp_fn(r) for r in val) if val else train_end_ts

    embargoed: list = []
    if embargo_seconds > 0:
        val, e1 = _drop_dead_zone(val, timestamp_fn, train_end_ts, embargo_seconds)
        test, e2 = _drop_dead_zone(test, timestamp_fn, val_end_ts, embargo_seconds)
        embargoed = e1 + e2

    purged: list = []
    if purge_seconds > 0:
        boundaries = [train_end_ts]
        if val:
            boundaries.append(min(timestamp_fn(r) for r in val))
            boundaries.append(max(timestamp_fn(r) for r in val))
        if test:
            boundaries.append(min(timestamp_fn(r) for r in test))
        train, p1 = _purge_near(train, timestamp_fn, boundaries, purge_seconds)
        val, p2 = _purge_near(val, timestamp_fn, boundaries, purge_seconds)
        test, p3 = _purge_near(test, timestamp_fn, boundaries, purge_seconds)
        purged = p1 + p2 + p3

    return SplitResult(train=train, val=val, test=test, purged=purged, embargoed=embargoed)


def _drop_dead_zone(records: list, timestamp_fn, boundary_ts: int, embargo_seconds: int) -> tuple[list, list]:
    kept, dropped = [], []
    dead_end = boundary_ts + embargo_seconds * 1000
    for r in records:
        (dropped if boundary_ts < timestamp_fn(r) <= dead_end else kept).append(r)
    return kept, dropped


def _purge_near(records: list, timestamp_fn, boundaries: list[int], purge_seconds: int) -> tuple[list, list]:
    kept, dropped = [], []
    window_ms = purge_seconds * 1000
    for r in records:
        ts = timestamp_fn(r)
        if any(abs(ts - b) <= window_ms for b in boundaries):
            dropped.append(r)
        else:
            kept.append(r)
    return kept, dropped
