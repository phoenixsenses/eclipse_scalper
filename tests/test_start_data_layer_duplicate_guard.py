from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.start_data_layer_guard import extract_matching_pids, should_start_instance


def test_extract_matching_pids_filters_module():
    rows = [
        {"ProcessId": 111, "CommandLine": "python -m data.microstructure_collector --symbols BTCUSDT"},
        {"ProcessId": 222, "CommandLine": "python -m data.event_diary --db-path data/microstructure.db"},
        {"ProcessId": 333, "CommandLine": "python -m something.else"},
        {"ProcessId": "111", "CommandLine": "python -m data.microstructure_collector --symbols ETHUSDT"},
    ]
    out = extract_matching_pids(rows, "data.microstructure_collector")
    assert out == [111]


def test_should_start_instance_guard():
    assert should_start_instance(force_restart=False, existing_pids=[]) is True
    assert should_start_instance(force_restart=False, existing_pids=[100]) is False
    assert should_start_instance(force_restart=True, existing_pids=[100]) is True
