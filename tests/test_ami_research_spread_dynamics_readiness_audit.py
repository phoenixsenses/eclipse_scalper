"""BATCH-SPREAD-EXPANSION-COMPRESSION-READINESS-AND-CONTRACT-V1 -- focused,
outcome-blind validation of ami/research/spread_dynamics_readiness_audit.py.

No outcome table is ever opened. No experiment, result, nullifier, or gate
receipt is created by this file or the module it tests.

Run: pytest tests/test_ami_research_spread_dynamics_readiness_audit.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import ast
import inspect
import sqlite3

import pytest

from ami.research import spread_dynamics_readiness_audit as audit

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_MICROSTRUCTURE_PATH = "D:/eclipse_scalper/data/microstructure.db"


def _ro_canonical():
    return sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)


def _ro_micro():
    return sqlite3.connect(f"file:{REAL_MICROSTRUCTURE_PATH}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# 1. Spread + mid formula correctness
# ---------------------------------------------------------------------------

def test_mid_price_formula():
    r = audit.spread_bps_and_mid(100.0, 102.0)
    assert r["mid_price"] == 101.0


def test_spread_bps_formula():
    r = audit.spread_bps_and_mid(100.0, 101.0)
    # spread = 1, mid = 100.5, bps = 1e4 * 1 / 100.5
    assert abs(r["spread_bps"] - (1e4 * 1.0 / 100.5)) < 1e-12


def test_spread_bps_zero_when_locked_handled_by_classifier_not_formula():
    # the raw formula gives 0 for bid==ask, but the classifier flags LOCKED
    r = audit.spread_bps_and_mid(100.0, 100.0)
    assert r["spread_bps"] == 0.0


def test_spread_formula_rejects_crossed():
    with pytest.raises(ValueError):
        audit.spread_bps_and_mid(101.0, 100.0)


def test_spread_formula_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        audit.spread_bps_and_mid(0.0, 100.0)
    with pytest.raises(ValueError):
        audit.spread_bps_and_mid(100.0, -1.0)


# ---------------------------------------------------------------------------
# 2. Quote selection: timestamp ordering / no post-birth / duplicate-ts
#    determinism (tie-break by id) -- synthetic in-memory book_ticker
# ---------------------------------------------------------------------------

def _mk_book(rows):
    """rows: list of (ts_ms, id, bid, ask). Builds an in-memory book_ticker
    with the same schema/index as the real table."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, "
              "symbol TEXT NOT NULL, bid_price REAL, ask_price REAL, bid_qty REAL, ask_qty REAL)")
    c.executemany("INSERT INTO book_ticker (id, ts_ms, symbol, bid_price, ask_price, bid_qty, ask_qty) "
                  "VALUES (?,?,?,?,?,?,?)",
                  [(i, ts, "ETHUSDT", b, a, 1.0, 1.0) for ts, i, b, a in rows])
    c.commit()
    return c


def test_quote_selection_latest_at_or_before():
    c = _mk_book([(100, 1, 10.0, 10.1), (200, 2, 11.0, 11.1), (300, 3, 12.0, 12.1)])
    q = audit.select_quote_at_or_before(c, 250)
    assert q["quote_ts_ms"] == 200 and q["bid_price"] == 11.0


def test_quote_selection_never_returns_future():
    c = _mk_book([(200, 1, 11.0, 11.1)])
    assert audit.select_quote_at_or_before(c, 100) is None  # only a future quote exists -> None


def test_quote_selection_exact_boundary_inclusive():
    c = _mk_book([(100, 1, 10.0, 10.1), (200, 2, 11.0, 11.1)])
    q = audit.select_quote_at_or_before(c, 200)
    assert q["quote_ts_ms"] == 200


def test_duplicate_timestamp_deterministic_tiebreak_by_id():
    """The real book_ticker has ~75% of rows sharing a ts_ms; the frozen rule
    tie-breaks by id DESC (latest insert). Two rows at the same ts_ms with
    different bid/ask must deterministically resolve to the higher-id one."""
    c = _mk_book([(100, 5, 10.0, 10.1), (100, 9, 20.0, 20.1), (100, 7, 30.0, 30.1)])
    q = audit.select_quote_at_or_before(c, 150)
    assert q["quote_id"] == 9 and q["bid_price"] == 20.0
    # deterministic across repeated calls
    for _ in range(5):
        assert audit.select_quote_at_or_before(c, 150)["quote_id"] == 9


# ---------------------------------------------------------------------------
# 3. Quality classification: crossed / locked / zero / stale / unavailable
# ---------------------------------------------------------------------------

def test_classify_unavailable_when_no_quote():
    r = audit.classify_quote(None, 1000)
    assert r["quality_status"] == audit.QUALITY_UNAVAILABLE


def test_classify_crossed_book_rejected():
    q = {"quote_ts_ms": 990, "quote_id": 1, "bid_price": 101.0, "ask_price": 100.0}
    r = audit.classify_quote(q, 1000)
    assert r["quality_status"] == audit.QUALITY_CROSSED
    assert r["spread_bps"] is None


def test_classify_zero_or_negative_rejected():
    q = {"quote_ts_ms": 990, "quote_id": 1, "bid_price": 0.0, "ask_price": 100.0}
    r = audit.classify_quote(q, 1000)
    assert r["quality_status"] == audit.QUALITY_ZERO_NEG


def test_classify_locked_book_flagged_not_silently_kept():
    q = {"quote_ts_ms": 990, "quote_id": 1, "bid_price": 100.0, "ask_price": 100.0}
    r = audit.classify_quote(q, 1000)
    assert r["quality_status"] == audit.QUALITY_LOCKED
    assert r["spread_bps"] == 0.0  # value present but under its own explicit code


def test_classify_stale_rejected_beyond_healthy_age():
    healthy = audit.BOOK_TICKER_HEALTHY_AGE_MS
    birth = 100_000_000
    q = {"quote_ts_ms": birth - healthy - 1, "quote_id": 1, "bid_price": 100.0, "ask_price": 100.1}
    r = audit.classify_quote(q, birth, healthy_age_ms=healthy)
    assert r["quality_status"] == audit.QUALITY_STALE


def test_classify_exact_at_boundary_of_healthy_age():
    healthy = audit.BOOK_TICKER_HEALTHY_AGE_MS
    birth = 100_000_000
    q = {"quote_ts_ms": birth - healthy, "quote_id": 1, "bid_price": 100.0, "ask_price": 100.1}
    r = audit.classify_quote(q, birth, healthy_age_ms=healthy)
    assert r["quality_status"] == audit.QUALITY_EXACT
    assert r["spread_bps"] is not None


# ---------------------------------------------------------------------------
# 4. Real-data accounting: reconciliation, idempotency, known-at, dedup
# ---------------------------------------------------------------------------

def test_anchor_accounting_reconciles_and_matches_expected():
    canon, micro = _ro_canonical(), _ro_micro()
    try:
        r = audit.anchor_accounting(canon, micro)
    finally:
        canon.close(); micro.close()
    assert r["reconciliation_ok"] is True
    assert r["total_anchors"] == 324
    assert r["quality_breakdown"].get(audit.QUALITY_EXACT) == 196
    assert r["quality_breakdown"].get(audit.QUALITY_UNAVAILABLE) == 106
    assert r["quality_breakdown"].get(audit.QUALITY_STALE) == 22
    assert r["exact_independent_cycles"] == 97


def test_anchor_accounting_idempotent_across_two_runs():
    c1, m1 = _ro_canonical(), _ro_micro()
    c2, m2 = _ro_canonical(), _ro_micro()
    try:
        r1 = audit.anchor_accounting(c1, m1)
        r2 = audit.anchor_accounting(c2, m2)
    finally:
        c1.close(); m1.close(); c2.close(); m2.close()
    assert r1["quality_breakdown"] == r2["quality_breakdown"]
    assert r1["exact_independent_cycles"] == r2["exact_independent_cycles"]
    assert [x["signal_id"] for x in r1["rows"]] == [x["signal_id"] for x in r2["rows"]]
    assert [x["quality_status"] for x in r1["rows"]] == [x["quality_status"] for x in r2["rows"]]


def test_no_lookahead_clean_on_real_accounting():
    canon, micro = _ro_canonical(), _ro_micro()
    try:
        r = audit.anchor_accounting(canon, micro)
    finally:
        canon.close(); micro.close()
    assert audit.verify_no_lookahead(r["rows"])["known_at_violations"] == 0


def test_duplicate_cycle_dedup_on_real_accounting():
    canon, micro = _ro_canonical(), _ro_micro()
    try:
        r = audit.anchor_accounting(canon, micro)
        dup = audit.verify_duplicate_cycle_free(r["rows"])
    finally:
        canon.close(); micro.close()
    assert dup["representative_cycle_count"] == r["exact_independent_cycles"] == 97
    assert dup["fresh_signal_count"] == r["exact_rows"] == 196


def test_windowed_pair_coverage_is_window_invariant_and_sufficient():
    canon, micro = _ro_canonical(), _ro_micro()
    try:
        for w in (60, 300, 3600):
            r = audit.windowed_pair_accounting(canon, micro, w)
            # book_ticker updates continuously -> windowed coverage ~= level coverage
            assert r["independent_cycles"] >= 90
            assert r["both_endpoints_exact"] >= 190
    finally:
        canon.close(); micro.close()


# ---------------------------------------------------------------------------
# 5. Symbol / venue / market-segment identity + exact/proxy separation
# ---------------------------------------------------------------------------

def test_symbol_scoped_to_ethusdt():
    assert audit.SYMBOL == "ETHUSDT"
    canon = _ro_canonical()
    try:
        sigs = audit.fetch_anchor_universe(canon)
    finally:
        canon.close()
    assert len(sigs) == 324
    assert {s["direction"] for s in sigs} == {"LONG", "SHORT"}


def test_book_ticker_coverage_index_backed_single_symbol():
    micro = _ro_micro()
    try:
        cov = audit.book_ticker_coverage(micro, "ETHUSDT")
    finally:
        micro.close()
    # coverage begins 2026-04-11T17:08:42.005Z (= 1775927322005), matching the
    # mechanism plan and the accepted absorption readiness audit
    assert cov["first_ts_ms"] == 1775927322005
    assert cov["last_ts_ms"] > cov["first_ts_ms"]


def test_no_proxy_tier_in_quality_taxonomy():
    """Spread from L1 best bid/ask is EXACT, not proxy -- there is no
    PROXY_ONLY class in this audit (the L1-is-proxy caveat applies to depth/
    absorption inference, not to the spread itself)."""
    codes = {audit.QUALITY_EXACT, audit.QUALITY_STALE, audit.QUALITY_UNAVAILABLE,
             audit.QUALITY_CROSSED, audit.QUALITY_ZERO_NEG, audit.QUALITY_LOCKED}
    assert not any("PROXY" in c for c in codes)


# ---------------------------------------------------------------------------
# 6. Outcome / experiment / nullifier / gate-receipt access denial (AST)
# ---------------------------------------------------------------------------

def test_module_never_executes_sql_naming_outcome_or_governance_tables():
    """AST guard: no string literal passed to any .execute()-family call may
    name the outcome table/columns or any experiment/nullifier/gate-receipt
    table. Narrower than a blunt substring scan (which would false-positive
    on docstrings), matching the absorption-impact rehearsal precedent."""
    tree = ast.parse(inspect.getsource(audit))
    forbidden = (
        "ami_lifecycle_path_observations", "endpoint_return_bps", "mfe_bps", "mae_bps",
        "experiment_registry", "experiment_results", "epistemic_test_nullifiers",
        "experiment_gate_receipts",
    )
    execute_methods = {"execute", "executescript", "executemany"}
    sql_literals = 0
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in execute_methods:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    sql_literals += 1
                    if any(t in arg.value for t in forbidden):
                        violations.append(arg.value)
    assert sql_literals > 0
    assert violations == []


def test_module_opens_no_outcome_or_governance_table_in_any_real_sql_literal():
    """Even outside the .execute() call sites, no *actual SQL statement*
    string literal in the module may name the outcome/governance tables. A
    string counts as real SQL only if, once stripped, it BEGINS with a SQL
    verb (SELECT/INSERT/UPDATE/DELETE/CREATE) -- so the module's own prose
    docstrings (which legitimately mention `ami_lifecycle_path_observations`
    to explain that it is never opened) are correctly excluded, unlike a
    naive substring scan."""
    sql_verbs = ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ")
    checked = 0
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            stripped = node.value.lstrip().upper()
            if stripped.startswith(sql_verbs):
                checked += 1
                v = node.value
                assert "ami_lifecycle_path_observations" not in v
                assert "experiment_" not in v
                assert "nullifier" not in v
    assert checked > 0  # sanity: the scan actually found real SQL
