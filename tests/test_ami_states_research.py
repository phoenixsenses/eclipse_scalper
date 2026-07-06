"""AMI tests — states, lifecycle, research OS (synthetic data; no main-DB dependency)."""
from __future__ import annotations
import sqlite3, sys, time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.enums import DataQuality, StateFamily, TIMEFRAMES
from ami.governance import epistemic_gates as _gates
from ami.lifecycle.engine import classify_lifecycle_path
from ami.research.marketplace import priority_score, rank_backlog
from ami.research.registry import (EvidenceBundle, ExperimentSpec, ResearchQuestion,
                                   ResearchRegistry)
from ami.states.engine import StateEngine
from ami.states.objects import StateBundle, StateObject


def make_synth_db(path: Path, hours: int = 24 * 40) -> Path:
    """Synthetic microstructure DB: trending+noise mark path, some liqs, book rows."""
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL,
                              funding_rate REAL, next_funding_time_ms INTEGER);
    CREATE TABLE liquidations (ts_ms INTEGER, symbol TEXT, side TEXT, notional REAL, price REAL);
    CREATE TABLE book_ticker (ts_ms INTEGER, symbol TEXT, bid_qty REAL, ask_qty REAL,
                              spread_pct REAL, book_imbalance REAL);
    CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT, price REAL, notional REAL, is_buyer_maker INTEGER);
    CREATE TABLE open_interest (ts_ms INTEGER, symbol TEXT, open_interest_usd REAL);
    CREATE TABLE spot_prices (ts_ms INTEGER, symbol TEXT, spot_price REAL);
    CREATE TABLE vol_state (ts_ms INTEGER, symbol TEXT, rv_5m REAL, vol_decile INTEGER);
    CREATE INDEX ix_m ON mark_prices(symbol, ts_ms);
    CREATE INDEX ix_l ON liquidations(symbol, ts_ms);
    CREATE INDEX ix_b ON book_ticker(symbol, ts_ms);
    """)
    now = int(time.time() * 1000)
    start = now - hours * 3_600_000
    import math, random
    rng = random.Random(7)
    px = 2000.0
    rows = []
    ts = start
    while ts <= now:
        px *= (1 + rng.gauss(0, 0.0008) + 0.00001 * math.sin(ts / 8.64e7))
        rows.append((ts, "ETHUSDT", px, -1e-5 if (ts // 3_600_000) % 2 else 1e-5, ts + 3_600_000))
        ts += 60_000
    conn.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO mark_prices VALUES (?,?,?,?,?)",
                     [(t, "BTCUSDT", p * 30, None, None) for t, _, p, _, _ in rows[::5]])
    conn.execute("INSERT INTO liquidations VALUES (?,?,?,?,?)", (now - 120_000, "ETHUSDT", "SELL", 250_000, px))
    conn.executemany("INSERT INTO book_ticker VALUES (?,?,?,?,?,?)",
                     [(now - k * 30_000, "ETHUSDT", 10 + k % 3, 9.0, 1e-5, 0.05) for k in range(30)])
    conn.execute("INSERT INTO open_interest VALUES (?,?,?)", (now - 60_000, "ETHUSDT", 2.5e9))
    conn.commit(); conn.close()
    return path


@pytest.fixture(scope="module")
def synth_engine(tmp_path_factory):
    db = make_synth_db(tmp_path_factory.mktemp("db") / "synth.sqlite")
    eng = StateEngine(db)
    yield eng
    eng.close()


def test_bundle_all_timeframes(synth_engine):
    b = synth_engine.build_bundle("ETHUSDT")
    tfs = {s.timeframe for s in b.by_family(StateFamily.STRUCTURE_STATE)}
    assert tfs == set(TIMEFRAMES)
    for s in b.states:
        assert s.label and s.data_quality in DataQuality


def test_bundle_conflict_report(synth_engine):
    b = synth_engine.build_bundle("ETHUSDT")
    rep = b.conflict_report()
    assert 0.0 <= rep["alignment_score"] <= 1.0
    assert rep["dominant"] in ("UP", "DOWN", "FLAT")


def test_data_quality_grades(synth_engine):
    ages = {"mark_prices": 1.0, "book_ticker": 30.0, "vol_state": 99999.0, "open_interest": None}
    q = synth_engine.data_quality(ages)
    assert q["mark_prices"] == DataQuality.HEALTHY
    assert q["book_ticker"] == DataQuality.DEGRADED
    assert q["vol_state"] == DataQuality.STALE
    assert q["open_interest"] == DataQuality.UNAVAILABLE


def test_lifecycle_classifier_paths():
    up = [0.0] + [i * 2.0 for i in range(1, 200)]
    seq = classify_lifecycle_path(up)
    labels = [s for _, s in seq]
    assert labels[0] == "OPEN" and labels[-1] == "CLOSED"
    assert "ACCELERATING" in labels or "HEALTHY" in labels
    crash = [0.0] + [-i * 1.5 for i in range(1, 200)]
    labels2 = [s for _, s in classify_lifecycle_path(crash)]
    assert "INVALIDATED" in labels2
    giveback = [0.0] + [i * 3 for i in range(1, 60)] + [180 - i * 2.5 for i in range(1, 100)]
    labels3 = [s for _, s in classify_lifecycle_path(giveback)]
    assert "WEAKENING" in labels3 or "REVERSING" in labels3


def test_experiment_freeze_enforced(tmp_path):
    reg = ResearchRegistry(tmp_path / "r.sqlite", knowledge_path=tmp_path / "k.sqlite")
    spec = ExperimentSpec("E-1", "Q-1", population="events", target="net", features=["f"],
                          threshold_method="frozen", untouched_data="post-2026-07",
                          decision_criteria="avg>0", falsification_rule="avg<=0")
    with pytest.raises(ConstitutionViolation):
        reg.register_experiment(spec)          # not frozen yet
    spec.freeze()
    # BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: direct gate receipt
    # (this test exercises freeze/attach_evidence immutability, not the gate).
    kconn = sqlite3.connect(str(reg.knowledge_path))
    _gates.init_gates_schema(kconn)
    _gates.issue_gate_receipt(kconn, experiment_id=spec.experiment_id,
                              canonical_family_id="STATES_RESEARCH_TEST_HARNESS",
                              split_version=None, nullifier=None, registry_result="TEST_HARNESS_DIRECT_RECEIPT")
    kconn.commit(); kconn.close()
    reg.register_experiment(spec)
    # post-hoc tampering -> evidence rejected
    spec.decision_criteria = "avg>-50 (relaxed after seeing results)"
    with pytest.raises(ConstitutionViolation):
        reg.attach_evidence(EvidenceBundle("EV-1", "E-1", {"avg": -10}, "SUPPORTS"), spec)
    reg.close()


def test_incomplete_prereg_blocked():
    spec = ExperimentSpec("E-2", "Q-1", population="events", target="net", features=["f"],
                          threshold_method="frozen")  # missing untouched/criteria/falsification
    with pytest.raises(ConstitutionViolation):
        spec.freeze()


def test_marketplace_ranking(tmp_path):
    qs = [ResearchQuestion("Q-HI", "high value", scientific_value=0.9, economic_value=0.9,
                           falsifiability=0.9, data_readiness=0.9, estimated_cost=0.1,
                           multiple_testing_risk=0.1, novelty=0.3),
          ResearchQuestion("Q-LO", "low value", scientific_value=0.2, economic_value=0.1,
                           falsifiability=0.3, data_readiness=0.2, estimated_cost=0.9,
                           multiple_testing_risk=0.8, novelty=0.9)]
    assert priority_score(qs[0]) > priority_score(qs[1])
    out = rank_backlog(qs)
    assert out["ranked"][0][0] == "Q-HI"
    assert "Q-LO" in out["portfolio"]["curiosity_15pct"]
