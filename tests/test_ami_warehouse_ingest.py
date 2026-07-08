"""BATCH-P1-002: artifact discovery + ingest tests.

Run: pytest tests/test_ami_warehouse_ingest.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.warehouse.artifact_ingest import (
    detect_ami_collision,
    discover,
    ingest,
)
from ami.warehouse.schema import connect, init_schema


def test_discover_finds_known_canonical_docs():
    records = discover()
    paths = {r.path for r in records}
    assert "SYSTEM_STATE.md" in paths
    assert "CLAUDE.md" in paths
    assert "docs/protocols/AMI_S34_MASTER_EXECUTION_PROTOCOL_v1.1.md" in paths


def test_whitepaper_pair_marked_under_reconciliation():
    records = {r.path: r for r in discover()}
    v02 = records.get("AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md")
    v03 = records.get("AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md")
    assert v02 is not None and v03 is not None
    assert v02.canonical_status == "UNDER_RECONCILIATION"
    assert v03.canonical_status == "UNDER_RECONCILIATION"


def test_collision_gate_negative_on_repo_docs():
    # None of the discovered canonical docs should trip the Advanced-Metering-Infrastructure gate.
    records = discover()
    quarantined = [r for r in records if r.canonical_status == "NAMESPACE_QUARANTINED"]
    assert quarantined == []


def test_collision_gate_positive_on_synthetic_text():
    assert detect_ami_collision("The utility deployed an Advanced Metering Infrastructure rollout.")
    assert not detect_ami_collision("AMI is Artificial Market Intelligence, a research programme.")


def test_collision_gate_self_aware_discussion_without_spelled_out_phrase():
    # Operational logs (e.g. SYSTEM_STATE.md) discuss the metering-infrastructure
    # collision while referring to this programme only as "AMI"/"S34", never
    # spelling out "Artificial Market Intelligence" -- must still not quarantine.
    text = (
        "AMI S34 build log: Reconstruction Protocol discusses an 'Advanced "
        "Metering Infrastructure' docx that must be quarantined from the "
        "AMI x S34 research programme."
    )
    assert not detect_ami_collision(text)


def test_collision_gate_self_aware_discussion_not_quarantined():
    # Reconstruction Protocol §2.2/15.2/18 instructs quarantining a *separate*
    # external "Advanced Metering Infrastructure" docx -- discussing that
    # instruction must not cause the protocol document itself to be quarantined.
    text = (
        "Artificial Market Intelligence research must not import the "
        "Advanced Metering Infrastructure Word document into the canonical "
        "market whitepaper except in a clearly labeled external analogy section."
    )
    assert not detect_ami_collision(text)


def test_ingest_is_idempotent_and_hash_stable(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    records = discover()

    n1 = ingest(conn, records)
    count1 = conn.execute("SELECT COUNT(*) FROM artifact_registry").fetchone()[0]
    hash1 = conn.execute(
        "SELECT content_hash FROM artifact_registry WHERE artifact_id='SYSTEM_STATE.md'"
    ).fetchone()[0]

    n2 = ingest(conn, records)  # re-run: same file set, must not duplicate
    count2 = conn.execute("SELECT COUNT(*) FROM artifact_registry").fetchone()[0]

    conn.close()
    assert n1 == n2 == len(records)
    assert count1 == count2 == len(records)
    assert hash1 and len(hash1) == 64


def test_ingest_populates_namespace_registry(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    ingest(conn, discover())
    row = conn.execute(
        "SELECT is_ami_artificial_market_intelligence FROM namespace_registry WHERE namespace='ami_s34'"
    ).fetchone()
    conn.close()
    assert row == (1,)
