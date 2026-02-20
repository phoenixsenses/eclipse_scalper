from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.data_layer_probe import compute_probe_ok


def test_probe_ok_without_diary_when_not_required():
    ok, warn = compute_probe_ok(
        collector_process_live=True,
        collector_data_live=True,
        diary_process_live=False,
        require_diary=False,
    )
    assert ok is True
    assert warn is not None
    assert "not required" in warn


def test_probe_fail_without_diary_when_required():
    ok, warn = compute_probe_ok(
        collector_process_live=True,
        collector_data_live=True,
        diary_process_live=False,
        require_diary=True,
    )
    assert ok is False
    assert warn is None
