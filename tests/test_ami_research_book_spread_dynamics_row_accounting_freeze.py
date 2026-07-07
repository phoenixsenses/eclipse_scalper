"""BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1 -- focused, outcome-blind
validation of ami/research/book_spread_dynamics_row_accounting_freeze.py.

No outcome table opened; no experiment/result/nullifier/gate-receipt created.
Real-data builds write to disposable DBs under pytest tmp_path only; the real
canonical.sqlite/knowledge.sqlite are never written.

Run: pytest tests/test_ami_research_book_spread_dynamics_row_accounting_freeze.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import ast
import inspect
import sqlite3

import pytest

from ami.research import book_spread_dynamics_rehearsal as REH
from ami.research import book_spread_dynamics_row_accounting_freeze as FR

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_MICROSTRUCTURE_PATH = "D:/eclipse_scalper/data/microstructure.db"
ACCEPTED_RUN1 = "D:/eclipse_scalper/.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite"

ACCEPTED_CONTENT = "5e9ee58cd9c260c2877b05ed803dbf51767ecedc579bdc90c37b5391a867bcbb"
ACCEPTED_ROWMAN = "8e8e23ff8af6dfd1c11199f963698d4a148583fd2b9c979dffa7f4e4fdec72f2"
ACCEPTED_SPEC = "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212"
FROZEN_ROOT = "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31"

COMPONENT_HASHES = {
    "ordered_anchor_manifest": "a77a8daf2a8d198d775436674a20a9bd5328dc071e2883938b7c331c17c534bb",
    "exact_feature_manifest": "b1eb902f5b3d1ea0f19b4b60d0ad999907a042b228adf506bbe09800a81e155b",
    "exclusion_manifest": "0694e43300710e1204c1b23643d9eacb9f10188c21aa0ceda572c28229cc8449",
    "cycle_membership_manifest": "e692ff1c8ce37b54a3349a501a38bd44f24865e75a51accc81c7e97399d29e18",
    "representative_manifest": "edadf5972cbbdddb0efa1db8234473ee089972f504d3bfbfafbae508238db246",
    "rehearsal_content_hash": ACCEPTED_CONTENT,
    "rehearsal_row_manifest_hash": ACCEPTED_ROWMAN,
    "specification_hash": ACCEPTED_SPEC,
}


def _replay(tmp_path, tag, manifest_id="spread-rehearsal-v1-input-manifest"):
    disp = sqlite3.connect(str(tmp_path / f"{tag}.sqlite"))
    canon = sqlite3.connect(f"file:{REAL_CANONICAL_PATH}?mode=ro", uri=True)
    micro = sqlite3.connect(f"file:{REAL_MICROSTRUCTURE_PATH}?mode=ro", uri=True)
    viol = REH.install_access_guard(canon)
    FR.replay_into(disp, canon, micro, manifest_id)
    canon.close(); micro.close()
    return disp, viol


# ---------------------------------------------------------------------------
# 1. Identity / definition enforcement (unchanged from accepted rehearsal)
# ---------------------------------------------------------------------------

def test_identity_constants_match_accepted():
    assert FR.FAMILY_ID == "FAM_BOOK_SPREAD_DYNAMICS"
    assert FR.CHILD_WORKING_ID == "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
    assert FR.FORMULA_VERSION == "BOOK_SPREAD_CHANGE_BPS_W300_V1"
    assert REH.WINDOW_SEC == 300
    assert REH.specification_hash() == ACCEPTED_SPEC


def test_expected_counts_frozen():
    assert (FR.EXPECTED_TOTAL, FR.EXPECTED_EXACT, FR.EXPECTED_EXCLUDED, FR.EXPECTED_CYCLES) == (324, 196, 128, 97)


# ---------------------------------------------------------------------------
# 2. Accounting identities + manifests on real replay
# ---------------------------------------------------------------------------

def test_accounting_identities_all_true(tmp_path):
    disp, viol = _replay(tmp_path, "acc")
    acc = FR.accounting_identities(disp)
    disp.close()
    assert viol == []
    assert acc["total"] == 324 and acc["exact"] == 196 and acc["stale"] == 22 and acc["unavailable"] == 106
    assert acc["excluded"] == 128
    assert acc["exact_independent_cycles"] == 97 and acc["cycle_representatives"] == 97
    assert acc["invalid_crossed"] == acc["invalid_zero_neg"] == acc["invalid_locked"] == 0
    assert acc["repaired_exact"] == acc["source_gapped"] == acc["proxy_only"] == 0
    assert acc["exact_with_feature_value"] == 196 and acc["excluded_with_feature_value"] == 0
    assert acc["cycles_with_multiple_representatives"] == 0
    assert acc["identity_anchor"] and acc["identity_nonexact"]
    assert acc["identity_exact_cycles"] and acc["identity_representatives"]


def test_manifests_reconcile_counts_and_hashes(tmp_path):
    disp, _ = _replay(tmp_path, "man")
    m = FR.build_all_manifests(disp)
    disp.close()
    assert m["ordered_anchor"]["count"] == 324
    assert m["exact_feature"]["count"] == 196
    assert m["exclusion"]["count"] == 128
    assert m["cycle_membership"]["count"] == 196   # all exact anchors, grouped by cycle
    assert m["representative"]["count"] == 97
    assert m["ordered_anchor"]["hash"] == COMPONENT_HASHES["ordered_anchor_manifest"]
    assert m["exact_feature"]["hash"] == COMPONENT_HASHES["exact_feature_manifest"]
    assert m["exclusion"]["hash"] == COMPONENT_HASHES["exclusion_manifest"]
    assert m["cycle_membership"]["hash"] == COMPONENT_HASHES["cycle_membership_manifest"]
    assert m["representative"]["hash"] == COMPONENT_HASHES["representative_manifest"]


def test_root_hash_deterministic_and_matches_frozen():
    assert FR.root_hash(COMPONENT_HASHES) == FROZEN_ROOT
    # order-independent over the dict
    reordered = dict(reversed(list(COMPONENT_HASHES.items())))
    assert FR.root_hash(reordered) == FROZEN_ROOT


# ---------------------------------------------------------------------------
# 3. Independent replay equality (A == B == accepted)
# ---------------------------------------------------------------------------

def test_two_replays_and_accepted_all_equal(tmp_path):
    dispA, _ = _replay(tmp_path, "A")
    dispB, _ = _replay(tmp_path, "B")
    mA, mB = FR.build_all_manifests(dispA), FR.build_all_manifests(dispB)
    a_content, b_content = REH.content_hash(dispA), REH.content_hash(dispB)
    a_rowman, b_rowman = REH.row_manifest_hash(dispA), REH.row_manifest_hash(dispB)
    dispA.close(); dispB.close()
    # A == B
    assert a_content == b_content
    assert a_rowman == b_rowman
    for k in mA:
        assert mA[k]["hash"] == mB[k]["hash"]
    # A/B == accepted retained evidence
    acc = sqlite3.connect(f"file:{ACCEPTED_RUN1}?mode=ro", uri=True)
    accepted_content = REH.content_hash(acc)
    accepted_rowman = REH.row_manifest_hash(acc)
    acc.close()
    assert a_content == accepted_content == ACCEPTED_CONTENT
    assert a_rowman == accepted_rowman == ACCEPTED_ROWMAN


# ---------------------------------------------------------------------------
# 4. Known-at revalidation + no-lookahead
# ---------------------------------------------------------------------------

def test_known_at_revalidation_all_zero(tmp_path):
    disp, _ = _replay(tmp_path, "ka")
    ka = FR.known_at_revalidation(disp)
    disp.close()
    assert ka["all_zero"] is True
    assert ka["current_endpoint_future_quote_selections"] == 0
    assert ka["historical_endpoint_future_quote_selections"] == 0
    assert ka["current_endpoint_staleness_violations"] == 0
    assert ka["historical_endpoint_staleness_violations"] == 0
    assert ka["identity_violations"] == 0
    assert ka["known_at_field_violations"] == 0


def test_no_selected_quote_postdates_its_target(tmp_path):
    disp, _ = _replay(tmp_path, "future")
    bad = disp.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE "
        "(current_quote_ts IS NOT NULL AND current_quote_ts > current_target_ts) OR "
        "(historical_quote_ts IS NOT NULL AND historical_quote_ts > historical_target_ts)").fetchone()[0]
    disp.close()
    assert bad == 0


def test_symbol_venue_segment_currency_identity(tmp_path):
    disp, _ = _replay(tmp_path, "ident")
    bad = disp.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE symbol!='ETHUSDT' OR venue!='BINANCE_USDM_PERP' "
        "OR market_segment!='PERPETUAL_FUTURES' OR quote_currency!='USDT'").fetchone()[0]
    disp.close()
    assert bad == 0


# ---------------------------------------------------------------------------
# 5. Exclusion / cycle integrity
# ---------------------------------------------------------------------------

def test_no_excluded_anchor_has_feature_value_and_no_dupes(tmp_path):
    disp, _ = _replay(tmp_path, "excl")
    excluded_with_val = disp.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE source_quality_class!='EXACT_RECONSTRUCTABLE' "
        "AND spread_change_bps_w300 IS NOT NULL").fetchone()[0]
    dup_anchor = disp.execute(
        "SELECT COUNT(*) FROM (SELECT anchor_id FROM book_spread_change_w300 GROUP BY anchor_id HAVING COUNT(*)>1)").fetchone()[0]
    dup_rep = disp.execute(
        "SELECT COUNT(*) FROM (SELECT cycle_id FROM book_spread_change_w300 WHERE is_cycle_representative=1 "
        "GROUP BY cycle_id HAVING COUNT(*)>1)").fetchone()[0]
    disp.close()
    assert excluded_with_val == 0 and dup_anchor == 0 and dup_rep == 0


def test_every_exact_cycle_has_exactly_one_representative(tmp_path):
    disp, _ = _replay(tmp_path, "rep")
    exact_cycles = disp.execute(
        "SELECT COUNT(DISTINCT cycle_id) FROM book_spread_change_w300 "
        "WHERE source_quality_class='EXACT_RECONSTRUCTABLE'").fetchone()[0]
    reps = disp.execute("SELECT COUNT(*) FROM book_spread_change_w300 WHERE is_cycle_representative=1").fetchone()[0]
    disp.close()
    assert exact_cycles == reps == 97


# ---------------------------------------------------------------------------
# 6. Structural: single formula/window, no alt forms, access denial
# ---------------------------------------------------------------------------

def test_freeze_module_no_alternative_window_or_transform():
    """Guard against alternative windows / transforms leaking into the freeze
    module. Uses specific transform identifiers (not the bare word 'ratio',
    which appears inside 'migration' in prose)."""
    src = inspect.getsource(FR)
    for forbidden in ("W60", "W600", "W1800", "W3600", "z_score", "zscore",
                      "log_ratio", "spread_ratio", "quantile", "winsor"):
        assert forbidden not in src


def test_ordering_and_serialization_policy_declared():
    assert FR.ORDERING_POLICY == "signal_birth_ts ASC, anchor_id ASC"
    assert "repr()" in FR.SERIALIZATION_POLICY and "sha256" in FR.SERIALIZATION_POLICY


def test_access_guard_denies_outcome_and_governance(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ami_lifecycle_path_observations (endpoint_return_bps REAL)")
    conn.execute("CREATE TABLE experiment_registry (x INTEGER)")
    conn.execute("CREATE TABLE experiment_gate_receipts (x INTEGER)")
    conn.commit()
    REH.install_access_guard(conn)
    for sql in ("SELECT endpoint_return_bps FROM ami_lifecycle_path_observations",
                "SELECT * FROM experiment_registry",
                "SELECT * FROM experiment_gate_receipts"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql).fetchall()
    conn.close()


def test_module_never_executes_sql_naming_outcome_or_governance():
    tree = ast.parse(inspect.getsource(FR))
    forbidden = ("ami_lifecycle_path_observations", "endpoint_return_bps", "mfe_bps", "mae_bps",
                 "experiment_registry", "experiment_results", "epistemic_test_nullifiers",
                 "experiment_gate_receipts")
    methods = {"execute", "executescript", "executemany"}
    n, bad = 0, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods:
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    n += 1
                    if any(t in a.value for t in forbidden):
                        bad.append(a.value)
    assert n > 0 and bad == []
