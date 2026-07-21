from tools.s34_v_engine_data_incomplete_audit import build_report, classify_incomplete, distribution


def test_classify_incomplete_missing_fill_ts():
    assert classify_incomplete({}, before=None, after=None, max_staleness_sec=10) == "missing_fill_ts"


def test_classify_incomplete_book_end_before_exit():
    row = {"maker_fill_ts_ms": 1000}
    before = {"staleness_sec": 3600.0}

    assert classify_incomplete(row, before=before, after=None, max_staleness_sec=10) == "book_history_ends_before_exit"


def test_classify_incomplete_unexpected_complete_book_available():
    row = {"maker_fill_ts_ms": 1000}
    before = {"staleness_sec": 5.0}

    assert classify_incomplete(row, before=before, after=None, max_staleness_sec=10) == "unexpected_complete_book_available"


def test_distribution_median_uses_sorted_values():
    assert distribution([9.0, 1.0, 5.0]) == {"n": 3, "min": 1.0, "median": 5.0, "max": 9.0}


def test_build_report_counts_reasons():
    audit = [
        {"reason": "stale_exit_book_gap", "sim_status": "NO_EXIT_BOOK", "signal_utc": "a", "book_staleness_sec": 20.0},
        {"reason": "stale_exit_book_gap", "sim_status": "NO_EXIT_BOOK", "signal_utc": "b", "book_staleness_sec": 30.0},
        {"reason": "missing_fill_ts", "sim_status": "NO_EXIT_BOOK", "signal_utc": "c"},
    ]

    report = build_report(
        ledger_rows=[{}] * 5,
        audit=audit,
        source_db={"path": "db"},
        source_ledger={"path": "ledger"},
        max_staleness_sec=10,
    )

    assert report["ledger_rows"] == 5
    assert report["data_incomplete_rows"] == 3
    assert report["reason_counts"] == {"missing_fill_ts": 1, "stale_exit_book_gap": 2}
