"""BATCH-P1-004: contradiction/operator-decision/lineage seed tests.

Run: pytest tests/test_ami_warehouse_registry_seed.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.warehouse.registry_seed import (
    derive_artifact_lineage,
    parse_conflict_register,
    parse_contradiction_register,
    parse_operator_decision_queue,
    seed,
)
from ami.warehouse.schema import connect, init_schema


def _stub_lineage_artifacts(conn):
    """seed()'s artifact_lineage rows FK-reference artifact_registry(artifact_id)
    (F4: now actually enforced). In the real pipeline artifact_ingest always
    runs before registry_seed; these tests exercise registry_seed in
    isolation, so they must stub the two referenced artifact_ids themselves."""
    now = 0
    for artifact_id in (
        "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md",
        "AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md",
    ):
        conn.execute(
            "INSERT INTO artifact_registry (artifact_id, path, role, canonical_status, schema_version, "
            "provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?)",
            (artifact_id, artifact_id, "canonical_spec", "UNDER_RECONCILIATION", 1, "test", now, now),
        )
    conn.commit()


def test_parse_conflict_register_finds_all_ten():
    rows = parse_conflict_register()
    ids = {r["contradiction_id"] for r in rows}
    assert ids == {f"CONFLICT-{i:03d}" for i in range(1, 11)}


def test_parse_contradiction_register_finds_five():
    # CT-005 added by the candidate-universe batch (per-KO AFFECTED/RECOMPUTE_REQUIRED
    # classification for OD-011's cycle-adjusted-N finding).
    rows = parse_contradiction_register()
    ids = {r["contradiction_id"] for r in rows}
    assert ids == {"CT-001", "CT-002", "CT-003", "CT-004", "CT-005"}


def test_parse_operator_decision_queue_finds_seventeen():
    # OD-017 added by BATCH-P6-011 (W10a LONG<->SHORT transitions half, backlog)
    rows = parse_operator_decision_queue()
    ids = {r["decision_id"] for r in rows}
    assert ids == {f"OD-{i:03d}" for i in range(1, 18)}


def test_lineage_records_reconciliation_not_fabricated_supersession():
    lineage = derive_artifact_lineage()
    assert len(lineage) == 1
    assert lineage[0]["relation"] == "UNDER_RECONCILIATION"
    assert "v0.2" in lineage[0]["predecessor_id"]
    assert "v0.3" in lineage[0]["artifact_id"]


def test_seed_populates_all_three_tables(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    _stub_lineage_artifacts(conn)
    n_c, n_d, n_l = seed(conn)
    conn_counts = (
        conn.execute("SELECT COUNT(*) FROM contradiction_registry").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM operator_decision_queue").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM artifact_lineage").fetchone()[0],
    )
    conn.close()
    assert n_c == 15  # 10 CONFLICT + 5 CT
    assert n_d == 17  # OD-001..017
    assert n_l == 1
    assert conn_counts == (15, 17, 1)


def test_seed_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    _stub_lineage_artifacts(conn)
    seed(conn)
    seed(conn)  # re-run: same sources, must not duplicate
    counts = (
        conn.execute("SELECT COUNT(*) FROM contradiction_registry").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM operator_decision_queue").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM artifact_lineage").fetchone()[0],
    )
    conn.close()
    assert counts == (15, 17, 1)


def test_no_operator_decisions_silently_marked_resolved():
    # OD-003 (BATCH-P3-005, A2+B2+C2) and OD-012 (BATCH-P6-003b, W2 deferred)
    # are the only explicitly-resolved decisions; every other OD entry is
    # still OPEN. The seed must reflect the source markdown verbatim, not
    # invent any other resolution.
    rows = parse_operator_decision_queue()
    by_id = {r["decision_id"]: r["status"] for r in rows}
    assert by_id.pop("OD-003") == "IMPLEMENTED"
    assert by_id.pop("OD-012") == "IMPLEMENTED"
    assert all(status == "OPEN" for status in by_id.values())
