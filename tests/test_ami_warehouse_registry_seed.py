"""BATCH-P1-004: contradiction/operator-decision/lineage seed tests.

Run: pytest tests/test_ami_warehouse_registry_seed.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.warehouse.registry_seed import (
    OPERATOR_DECISION_QUEUE_MD,
    derive_artifact_lineage,
    parse_conflict_register,
    parse_contradiction_register,
    parse_operator_decision_queue,
    seed,
)
from ami.warehouse.schema import connect, init_schema


def _raw_operator_decision_rows():
    """Independent, minimal re-extraction of OPERATOR_DECISION_QUEUE.md's
    own OD-* rows (id + verbatim status cell), read directly from the
    source markdown -- not via parse_operator_decision_queue() -- so tests
    below can cross-check the production parser's output against the
    source of truth instead of a hardcoded, inevitably-stale snapshot.
    OPERATOR_DECISION_QUEUE.md is actively operator-maintained (new OD-*
    rows and status transitions land there continuously); the count and
    status distribution are expected to keep growing/changing."""
    text = OPERATOR_DECISION_QUEUE_MD.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("| OD-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        rows[cells[0]] = cells[5]
    return rows


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


def test_parse_operator_decision_queue_matches_source_markdown():
    # OPERATOR_DECISION_QUEUE.md is actively operator-maintained -- the
    # OD-* row count keeps growing (17 -> 23 as of the 2026-07-13
    # broad-regression corrective). This asserts parity against the raw
    # markdown itself (no row silently dropped/duplicated by the parser)
    # rather than a hardcoded, inevitably-stale count.
    rows = parse_operator_decision_queue()
    ids = {r["decision_id"] for r in rows}
    raw_ids = set(_raw_operator_decision_rows().keys())
    assert raw_ids, "source markdown yielded zero OD-* rows -- parser regex likely broken"
    assert ids == raw_ids


def test_lineage_records_reconciliation_not_fabricated_supersession():
    lineage = derive_artifact_lineage()
    assert len(lineage) == 1
    assert lineage[0]["relation"] == "UNDER_RECONCILIATION"
    assert "v0.2" in lineage[0]["predecessor_id"]
    assert "v0.3" in lineage[0]["artifact_id"]


def test_seed_populates_all_three_tables(tmp_path):
    expected_od_count = len(_raw_operator_decision_rows())
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
    assert n_d == expected_od_count  # OPERATOR_DECISION_QUEUE.md row count (source of truth)
    assert n_l == 1
    assert conn_counts == (15, expected_od_count, 1)


def test_seed_is_idempotent(tmp_path):
    expected_od_count = len(_raw_operator_decision_rows())
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
    assert counts == (15, expected_od_count, 1)


def test_no_operator_decisions_silently_marked_resolved():
    # The seed must reflect OPERATOR_DECISION_QUEUE.md's own status column
    # verbatim -- never a fabricated/altered resolution. Cross-checked
    # against an independent re-extraction of the raw markdown (not a
    # hardcoded snapshot, which would go stale every time the operator
    # resolves or adds a decision -- as happened between the original
    # 17-row/2-resolved baseline and the current 23-row state).
    rows = parse_operator_decision_queue()
    by_id = {r["decision_id"]: r["status"] for r in rows}
    raw = _raw_operator_decision_rows()
    assert raw, "source markdown yielded zero OD-* rows -- parser regex likely broken"
    assert by_id == raw
