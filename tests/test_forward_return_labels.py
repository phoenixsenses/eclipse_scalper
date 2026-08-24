from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.labels.forward_return import direction_label, forward_return, make_labels


def test_forward_return_basic():
    mids = [100.0, 101.0, 99.0, 100.0]
    out = forward_return(mids, horizon_steps=1)
    assert out[0] == (101.0 / 100.0) - 1.0
    assert out[1] == (99.0 / 101.0) - 1.0
    assert out[2] == (100.0 / 99.0) - 1.0
    assert out[3] is None


def test_direction_label_threshold():
    assert direction_label(0.001, 0.0002) == 1
    assert direction_label(-0.001, 0.0002) == -1
    assert direction_label(0.0001, 0.0002) == 0


def test_make_labels_shapes():
    ts = [1, 2, 3, 4, 5]
    mids = [100, 101, 102, 103, 104]
    out = make_labels(ts, mids, horizons=[1, 2], threshold=0.0001)
    assert 1 in out["returns"]
    assert 2 in out["returns"]
    assert len(out["returns"][1]) == len(ts)
    assert len(out["labels"][2]) == len(ts)
