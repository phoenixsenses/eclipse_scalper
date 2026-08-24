from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_data_ready import compute_age_sec, detect_ts_col, normalize_ts_to_seconds


def test_detect_ts_col_prefers_order():
    cols = ["symbol", "eventTime", "ts_ms", "price"]
    assert detect_ts_col(cols) == "ts_ms"


def test_detect_ts_col_case_insensitive():
    cols = ["SYMBOL", "Created_At", "qty"]
    assert detect_ts_col(cols) == "Created_At"


def test_detect_ts_col_none():
    cols = ["symbol", "price", "qty"]
    assert detect_ts_col(cols) is None


def test_normalize_ts_to_seconds():
    assert normalize_ts_to_seconds(1_700_000_000) == 1_700_000_000.0
    assert normalize_ts_to_seconds(1_700_000_000_000) == 1_700_000_000.0


def test_compute_age_sec_fixed_now():
    now = 1_700_000_100.0
    assert compute_age_sec(1_700_000_000, now) == 100
    assert compute_age_sec(1_700_000_000_000, now) == 100
