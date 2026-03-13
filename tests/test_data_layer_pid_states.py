from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.data_layer_probe import _pid_state


def test_pid_state_missing_with_no_process():
    assert _pid_state(None, False, 0, False) == "missing"


def test_pid_state_missing_with_live_process():
    assert _pid_state(None, False, 1, False) == "missing_process_live"


def test_pid_state_live():
    assert _pid_state(1234, True, 1, True) == "live"


def test_pid_state_stale_but_process_or_data_live():
    assert _pid_state(1234, False, 1, False) == "stale_process_live"
    assert _pid_state(1234, False, 0, True) == "stale_process_live"


def test_pid_state_stale():
    assert _pid_state(1234, False, 0, False) == "stale"
