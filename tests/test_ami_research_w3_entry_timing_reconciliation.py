"""BATCH-P6-004: W3 entry-timing reconciliation tests.

Run: pytest tests/test_ami_research_w3_entry_timing_reconciliation.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.research.w3_entry_timing_reconciliation import (
    CONCLUSION,
    EXPERIMENT_ID,
    RECONCILIATION,
    freeze_and_record,
)
from ami.warehouse.schema import connect, init_schema

VALID_VERDICTS = {
    "EXACT_HYPOTHESIS_ALREADY_TESTED", "SCIENTIFICALLY_REJECTED", "ECONOMICALLY_REJECTED",
    "RETRY_CONDITION_NOT_MET", "RETRY_CONDITION_MET", "DISTINCT_MECHANISM",
    "UNANSWERED_RESEARCH_GAP", "SUPERSEDED", "INSUFFICIENT_SAMPLE",
}


def test_all_ten_reports_reconciled():
    assert len(RECONCILIATION) == 10


def test_every_verdict_uses_the_operator_specified_taxonomy():
    for item in RECONCILIATION:
        parts = item["verdict"].split(";")
        for p in parts:
            assert p in VALID_VERDICTS, f"{item['report']} has non-taxonomy verdict {p!r}"


def test_no_report_verdict_is_silently_forced_new_hypothesis():
    # None of the 10 verdicts may claim a fresh, standalone new hypothesis
    # (RETRY_CONDITION_MET alone would license reopening a graveyard family --
    # this reconciliation found none of the graveyard retry-conditions met).
    for item in RECONCILIATION:
        assert "RETRY_CONDITION_MET" not in item["verdict"].split(";")


def test_conclusion_does_not_force_open_w3():
    assert "NOT forced open" in CONCLUSION or "not forced open" in CONCLUSION or "is NOT forced" in CONCLUSION


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def test_freeze_and_record_writes_canonical_sql_not_only_markdown(tmp_path):
    conn = _db(tmp_path)
    result = freeze_and_record(conn)
    exp_row = conn.execute(
        "SELECT software_verdict, scientific_verdict FROM experiment_registry WHERE experiment_id=?",
        (EXPERIMENT_ID,),
    ).fetchone()
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    conn.close()
    assert exp_row == ("PASSED", "NO_NEW_TESTABLE_HYPOTHESIS")
    assert n_results == len(RECONCILIATION) + 1  # +1 for the CONCLUSION row
    assert result["n_reports_reconciled"] == 10


def test_freeze_and_record_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    freeze_and_record(conn)
    freeze_and_record(conn)
    n_registry = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    conn.close()
    assert n_registry == 1
    assert n_results == len(RECONCILIATION) + 1
