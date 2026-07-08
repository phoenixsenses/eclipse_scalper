"""BATCH-P1-003: question registry seed tests.

Run: pytest tests/test_ami_warehouse_question_seed.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.warehouse.question_seed import LEGACY_SLUG_FAMILY_ID, seed
from ami.warehouse.schema import connect, init_schema


def test_seed_loads_1058_numeric_plus_14_legacy_questions(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    n_fam, n_q = seed(conn)
    total = conn.execute("SELECT COUNT(*) FROM question_registry").fetchone()[0]
    conn.close()
    assert n_q == total
    assert total == 1058 + 14
    assert n_fam >= 1


def test_no_fabricated_text_for_missing_canonical_questions(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    seed(conn)
    row = conn.execute(
        "SELECT question_text, text_status FROM question_registry WHERE question_id='Q001'"
    ).fetchone()
    conn.close()
    assert row[1] == "MISSING_CANONICAL_TEXT"
    assert row[0] == ""


def test_chart_native_verbatim_text_present(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    seed(conn)
    row = conn.execute(
        "SELECT question_text, text_status FROM question_registry WHERE question_id='Q867'"
    ).fetchone()
    conn.close()
    assert row[1] == "CANONICAL_TEXT_PRESENT_PROPOSED"
    assert "upper-wick" in row[0].lower()


def test_legacy_slug_questions_present_and_distinct_namespace(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    seed(conn)
    row = conn.execute(
        "SELECT family_id, current_status FROM question_registry WHERE question_id='Q-MECHCOMP-FORWARD-001'"
    ).fetchone()
    fam = conn.execute(
        "SELECT title FROM question_families WHERE family_id=?", (LEGACY_SLUG_FAMILY_ID,)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == LEGACY_SLUG_FAMILY_ID
    assert row[1] == "OPEN"
    assert fam is not None


def test_seed_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    seed(conn)
    count1 = conn.execute("SELECT COUNT(*) FROM question_registry").fetchone()[0]
    seed(conn)  # re-run: same sources, must not duplicate
    count2 = conn.execute("SELECT COUNT(*) FROM question_registry").fetchone()[0]
    conn.close()
    assert count1 == count2 == 1058 + 14
