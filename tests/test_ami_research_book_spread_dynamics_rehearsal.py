"""BATCH-BOOK-SPREAD-DYNAMICS-DISPOSABLE-REHEARSAL-V1 -- focused, outcome-blind
validation of ami/research/book_spread_dynamics_rehearsal.py.

No outcome table is opened; no experiment/result/nullifier/gate-receipt is
created. Real-data checks use disposable output DBs under the pytest
tmp_path only; the real canonical.sqlite is never written.

Run: pytest tests/test_ami_research_book_spread_dynamics_rehearsal.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import ast
import inspect
import math
import sqlite3

import pytest

from ami.research import book_spread_dynamics_rehearsal as R
from ami.research import spread_dynamics_readiness_audit as A

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_MICROSTRUCTURE_PATH = "D:/eclipse_scalper/data/microstructure.db"


# ---------------------------------------------------------------------------
# 1. Formula + sign (pure)
# ---------------------------------------------------------------------------

def test_mid_and_spread_formula():
    r = A.spread_bps_and_mid(100.0, 101.0)
    assert r["mid_price"] == 100.5
    assert abs(r["spread_bps"] - (1e4 * 1.0 / 100.5)) < 1e-12


def test_w300_additive_difference_expansion_positive():
    cur = A.spread_bps_and_mid(100.0, 100.2)["spread_bps"]   # wider
    hist = A.spread_bps_and_mid(100.0, 100.1)["spread_bps"]  # narrower
    assert cur - hist > 0  # widening = expansion = positive


def test_w300_additive_difference_compression_negative():
    cur = A.spread_bps_and_mid(100.0, 100.1)["spread_bps"]   # narrower
    hist = A.spread_bps_and_mid(100.0, 100.2)["spread_bps"]  # wider
    assert cur - hist < 0  # narrowing = compression = negative


def test_w300_additive_difference_zero_when_unchanged():
    cur = A.spread_bps_and_mid(100.0, 100.15)["spread_bps"]
    hist = A.spread_bps_and_mid(100.0, 100.15)["spread_bps"]
    assert cur - hist == 0.0


def test_window_is_exactly_300_seconds():
    assert R.WINDOW_SEC == 300
    assert R.WINDOW_MS == 300_000


# ---------------------------------------------------------------------------
# 2. Quality precedence determinism
# ---------------------------------------------------------------------------

def test_row_quality_both_exact():
    cls, reason, ep = R._row_quality(A.QUALITY_EXACT, A.QUALITY_EXACT)
    assert cls == A.QUALITY_EXACT and reason is None and ep is None


def test_row_quality_unavailable_beats_stale():
    cls, reason, ep = R._row_quality(A.QUALITY_STALE, A.QUALITY_UNAVAILABLE)
    assert cls == A.QUALITY_UNAVAILABLE and ep == "historical"


def test_row_quality_crossed_beats_locked_beats_stale():
    assert R._row_quality(A.QUALITY_LOCKED, A.QUALITY_CROSSED)[0] == A.QUALITY_CROSSED
    assert R._row_quality(A.QUALITY_STALE, A.QUALITY_LOCKED)[0] == A.QUALITY_LOCKED


def test_row_quality_zero_neg_is_highest_invalid_precedence():
    assert R._row_quality(A.QUALITY_CROSSED, A.QUALITY_ZERO_NEG)[0] == A.QUALITY_ZERO_NEG


def test_row_quality_tie_marks_both():
    cls, reason, ep = R._row_quality(A.QUALITY_STALE, A.QUALITY_STALE)
    assert cls == A.QUALITY_STALE and ep == "both"


def test_row_quality_tie_current_before_historical_when_same_rank_different_endpoint():
    # only one endpoint non-exact -> that endpoint tagged
    assert R._row_quality(A.QUALITY_STALE, A.QUALITY_EXACT)[2] == "current"
    assert R._row_quality(A.QUALITY_EXACT, A.QUALITY_STALE)[2] == "historical"


# ---------------------------------------------------------------------------
# 3. Synthetic end-to-end build (deterministic disposable DBs)
# ---------------------------------------------------------------------------

def _mk_micro(rows):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, "
              "symbol TEXT NOT NULL, bid_price REAL, ask_price REAL, bid_qty REAL, ask_qty REAL)")
    c.executemany("INSERT INTO book_ticker (id, ts_ms, symbol, bid_price, ask_price, bid_qty, ask_qty) "
                  "VALUES (?,?,?,?,?,?,?)",
                  [(i, ts, "ETHUSDT", b, a, 1.0, 1.0) for ts, i, b, a in rows])
    c.commit()
    return c


def _mk_canon(signals):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE ami_signal_lifecycle (signal_id TEXT, direction TEXT, "
              "independent_cycle_id TEXT, signal_birth_ts INTEGER, source_event_id TEXT)")
    c.executemany("INSERT INTO ami_signal_lifecycle VALUES (?,?,?,?,?)", signals)
    c.commit()
    return c


def test_end_to_end_exact_row_computes_additive_change(tmp_path):
    t0 = 1_000_000_000
    micro = _mk_micro([
        (t0 - 300_000, 1, 100.0, 100.1),   # historical: spread ~1 unit
        (t0, 2, 100.0, 100.3),             # current: wider -> expansion
    ])
    canon = _mk_canon([("SIG-A", "LONG", "CYC-1", t0, "EVT-1")])
    disp = sqlite3.connect(str(tmp_path / "d.sqlite"))
    counts = R.run_rehearsal(disp, canon, micro, "test-manifest")
    assert counts["exact"] == 1
    row = disp.execute("SELECT source_quality_class, spread_change_bps_w300, is_cycle_representative "
                       "FROM book_spread_change_w300").fetchone()
    assert row[0] == "EXACT_RECONSTRUCTABLE"
    assert row[1] > 0  # expansion
    assert row[2] == 1  # sole cycle representative
    disp.close(); canon.close(); micro.close()


def test_end_to_end_unavailable_historical_endpoint(tmp_path):
    t0 = 1_000_000_000
    # only a current quote exists; historical target has no quote at-or-before
    micro = _mk_micro([(t0, 2, 100.0, 100.1)])
    canon = _mk_canon([("SIG-B", "SHORT", "CYC-2", t0, "EVT-2")])
    disp = sqlite3.connect(str(tmp_path / "d.sqlite"))
    R.run_rehearsal(disp, canon, micro, "test-manifest")
    row = disp.execute("SELECT source_quality_class, exclusion_reason, exclusion_endpoint, "
                       "spread_change_bps_w300 FROM book_spread_change_w300").fetchone()
    assert row[0] == "UNAVAILABLE_BEFORE_COLLECTION"
    assert row[2] == "historical"
    assert row[3] is None  # no feature value when not exact
    disp.close(); canon.close(); micro.close()


def test_end_to_end_crossed_book_rejected(tmp_path):
    t0 = 1_000_000_000
    micro = _mk_micro([(t0 - 300_000, 1, 100.0, 100.1), (t0, 2, 101.0, 100.0)])  # current crossed
    canon = _mk_canon([("SIG-C", "LONG", "CYC-3", t0, "EVT-3")])
    disp = sqlite3.connect(str(tmp_path / "d.sqlite"))
    R.run_rehearsal(disp, canon, micro, "test-manifest")
    row = disp.execute("SELECT source_quality_class, exclusion_endpoint FROM book_spread_change_w300").fetchone()
    assert row[0] == "INVALID_QUOTE_CROSSED"
    assert row[1] == "current"
    disp.close(); canon.close(); micro.close()


def test_one_row_per_anchor_and_reconciles(tmp_path):
    t0 = 1_000_000_000
    micro = _mk_micro([(t0 - 300_000, 1, 100.0, 100.1), (t0, 2, 100.0, 100.2)])
    canon = _mk_canon([
        ("SIG-A", "LONG", "CYC-1", t0, "EVT-1"),
        ("SIG-B", "LONG", "CYC-1", t0 + 5, "EVT-1"),   # same cycle -> dedup
        ("SIG-C", "SHORT", "CYC-2", t0, "EVT-2"),
    ])
    disp = sqlite3.connect(str(tmp_path / "d.sqlite"))
    R.run_rehearsal(disp, canon, micro, "test-manifest")
    acc = R.accounting(disp)
    assert acc["total_rows"] == acc["distinct_anchor_ids"] == 3
    assert acc["exact_rows"] == 3
    assert acc["exact_independent_cycles"] == 2      # CYC-1, CYC-2
    assert acc["cycle_representatives"] == 2
    assert acc["duplicate_cycle_representatives"] == 0
    disp.close(); canon.close(); micro.close()


def test_two_build_content_hash_identical(tmp_path):
    t0 = 1_000_000_000
    rows = [(t0 - 300_000, 1, 100.0, 100.1), (t0, 2, 100.0, 100.2)]
    sigs = [("SIG-A", "LONG", "CYC-1", t0, "EVT-1")]
    h = []
    for tag in ("a", "b"):
        micro = _mk_micro(rows); canon = _mk_canon(sigs)
        disp = sqlite3.connect(str(tmp_path / f"{tag}.sqlite"))
        R.run_rehearsal(disp, canon, micro, "test-manifest")
        h.append(R.content_hash(disp))
        disp.close(); canon.close(); micro.close()
    assert h[0] == h[1]


# ---------------------------------------------------------------------------
# 4. Real-data determinism + access denial
# ---------------------------------------------------------------------------

def test_real_data_two_build_deterministic_and_authorizer_clean(tmp_path):
    hashes = []
    for tag in ("r1", "r2"):
        disp = sqlite3.connect(str(tmp_path / f"{tag}.sqlite"))
        canon = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
        micro = sqlite3.connect(f"file:{REAL_MICROSTRUCTURE_PATH}?mode=ro", uri=True)
        viol = R.install_access_guard(canon)
        R.run_rehearsal(disp, canon, micro, "test-manifest")
        acc = R.accounting(disp)
        hashes.append((R.content_hash(disp), R.row_manifest_hash(disp)))
        assert viol == []  # zero outcome/experiment/nullifier/gate-receipt access
        assert acc["total_rows"] == 324 and acc["reconciles_to_324"] is True
        assert acc["by_source_quality_class"].get("EXACT_RECONSTRUCTABLE") == 196
        assert acc["exact_independent_cycles"] == 97
        assert acc["duplicate_cycle_representatives"] == 0
        assert acc["known_at_field_violations"] == 0
        disp.close(); canon.close(); micro.close()
    assert hashes[0] == hashes[1]  # content + manifest hash identical across builds


def test_real_data_numerical_validity(tmp_path):
    disp = sqlite3.connect(str(tmp_path / "n.sqlite"))
    canon = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    micro = sqlite3.connect(f"file:{REAL_MICROSTRUCTURE_PATH}?mode=ro", uri=True)
    R.run_rehearsal(disp, canon, micro, "test-manifest")
    rows = disp.execute("SELECT current_mid, current_spread_bps, historical_spread_bps, spread_change_bps_w300 "
                        "FROM book_spread_change_w300 WHERE source_quality_class='EXACT_RECONSTRUCTABLE'").fetchall()
    disp.close(); canon.close(); micro.close()
    assert len(rows) == 196
    for mid, cs, hs, ch in rows:
        assert math.isfinite(mid) and mid > 0
        assert math.isfinite(cs) and math.isfinite(hs) and math.isfinite(ch)
        assert abs(ch - (cs - hs)) < 1e-9  # additive difference holds exactly


# ---------------------------------------------------------------------------
# 5. Structural: single formula/window, no alt forms, access-denial guards
# ---------------------------------------------------------------------------

def test_single_frozen_formula_version_and_window():
    assert R.FORMULA_VERSION == "BOOK_SPREAD_CHANGE_BPS_W300_V1"
    src = inspect.getsource(R)
    # the module must not build any alternative-window feature column/constant
    for forbidden in ("W60", "W600", "W1800", "W3600", "z_score", "zscore", "log_ratio", "quantile"):
        assert forbidden not in src


def test_no_alternative_transform_columns_in_schema():
    disp = sqlite3.connect(":memory:")
    R.build_rehearsal_schema(disp)
    cols = [r[1] for r in disp.execute("PRAGMA table_info(book_spread_change_w300)")]
    disp.close()
    # exactly one feature column; the two endpoint spreads are provenance only
    assert "spread_change_bps_w300" in cols
    assert not any(("ratio" in c or "zscore" in c or "log" in c or "quantile" in c or "bin" in c) for c in cols)


def test_no_outcome_column_in_disposable_schema():
    disp = sqlite3.connect(":memory:")
    R.build_rehearsal_schema(disp)
    cols = [r[1] for r in disp.execute("PRAGMA table_info(book_spread_change_w300)")]
    disp.close()
    for bad in ("endpoint_return_bps", "mfe_bps", "mae_bps"):
        assert bad not in cols


def test_access_guard_denies_outcome_and_governance_tables():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ami_lifecycle_path_observations (endpoint_return_bps REAL)")
    conn.execute("CREATE TABLE experiment_registry (x INTEGER)")
    conn.execute("CREATE TABLE epistemic_test_nullifiers (nullifier TEXT)")
    conn.commit()
    R.install_access_guard(conn)
    for sql in ("SELECT endpoint_return_bps FROM ami_lifecycle_path_observations",
                "SELECT * FROM experiment_registry",
                "SELECT * FROM epistemic_test_nullifiers"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql).fetchall()
    conn.close()


def test_module_never_executes_sql_naming_outcome_or_governance_tables():
    """AST guard over .execute()-family string literals only (docstrings/deny-
    list constants excluded)."""
    tree = ast.parse(inspect.getsource(R))
    forbidden = ("ami_lifecycle_path_observations", "endpoint_return_bps", "mfe_bps", "mae_bps",
                 "experiment_registry", "experiment_results", "epistemic_test_nullifiers",
                 "experiment_gate_receipts")
    methods = {"execute", "executescript", "executemany"}
    n = 0
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods:
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    n += 1
                    if any(t in a.value for t in forbidden):
                        bad.append(a.value)
    assert n > 0
    assert bad == []


def test_specification_hash_is_stable():
    assert R.specification_hash() == R.specification_hash()
    assert len(R.specification_hash()) == 64
