"""BATCH-P2-003: funding/OI data-coverage audit tests (OD-006, read-only).

Run: pytest tests/test_ami_warehouse_funding_oi_audit.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.warehouse.funding_oi_audit import audit, seed
from ami.warehouse.schema import connect, init_schema


def test_audit_finds_known_stale_and_orphaned_sources():
    findings = audit()
    ids = {f["event_id"] for f in findings}
    assert "DQE-FUNDING-HIST-STALE" in ids
    assert "DQE-MICRO-FUNDING-ORPHANED" in ids
    assert "DQE-OI-HIST-STALE" in ids
    assert "DQE-OI-HISTORY-DB-STALE" in ids
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert f"DQE-MICRO-OI-COVERAGE-{sym}" in ids


def test_orphaned_funding_table_flagged_high_severity():
    findings = {f["event_id"]: f for f in audit()}
    assert findings["DQE-MICRO-FUNDING-ORPHANED"]["severity"] == "HIGH"
    assert "no live funding-rate collector" in findings["DQE-MICRO-FUNDING-ORPHANED"]["description"].lower()


def test_audit_is_stable_across_runs():
    # audit() opens all sources with mode=ro; re-running must not error and
    # must return the same finding count (7: 4 stale/orphaned-source findings
    # + 3 per-symbol OI coverage notes).
    n1 = len(audit())
    n2 = len(audit())
    assert n1 == n2 == 7


def test_seed_is_idempotent_and_matches_audit_count(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    n1 = seed(conn)
    count1 = conn.execute("SELECT COUNT(*) FROM data_quality_events").fetchone()[0]
    n2 = seed(conn)
    count2 = conn.execute("SELECT COUNT(*) FROM data_quality_events").fetchone()[0]
    conn.close()
    assert n1 == n2 == len(audit())
    assert count1 == count2 == len(audit())
