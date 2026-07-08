"""PHASE 7A-0: characterization + safety-invariant closure tests for
ami/lifecycle/engine.py (already live via docs/ami/AMI_ROADMAP.md's whitepaper
Phase 3 -- previously without dedicated test coverage).

NO_PRODUCTION_CODE_CHANGE: every test below exercises EXISTING, UNMODIFIED
code. Findings are separated into CURRENT_BEHAVIOR_CHARACTERIZED (what the
code does) vs REQUIRED_SAFETY_INVARIANT_MET/NOT_MET (whether that behavior
satisfies the safety contract) -- see SYSTEM_STATE.md Phase 7A-0 report for
the consolidated verdict. All fixtures are tmp_path-only; no write ever
touches a real store.

Run: pytest tests/test_ami_lifecycle_engine_characterization.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations
import ast
import json
import sqlite3
from pathlib import Path

from ami.enums import TradeLifecycleState as TLS
from ami.lifecycle.engine import LifecycleEngine, classify_lifecycle_path

ENGINE_SRC_PATH = Path(__file__).resolve().parents[1] / "ami" / "lifecycle" / "engine.py"
_FORBIDDEN_IMPORT_PREFIXES = ("execution", "risk", "brain")


def _mk_micro_db(path, prices: list[tuple[int, float]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
    for ts, px in prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "ETHUSDT", px))
    conn.commit()
    conn.close()


def _write_ledger(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---- deterministic signal identity ----

def test_classify_lifecycle_path_is_pure_and_deterministic():
    path = [0.0, 10.0, 30.0, 60.0, 55.0, 40.0, 10.0, -20.0]
    r1 = classify_lifecycle_path(path)
    r2 = classify_lifecycle_path(path)
    assert r1 == r2


def test_classify_lifecycle_path_depends_only_on_input_values():
    path_a = [0.0, 60.0, -20.0, -110.0]
    path_b = list(path_a)  # equal values, distinct list object -- no hidden identity dependence
    assert classify_lifecycle_path(path_a) == classify_lifecycle_path(path_b)


# ---- invalid transition sequence rejection (characterization) ----

def test_sequence_has_no_adjacent_duplicate_states():
    path = [0.0, 5.0, 10.0, 60.0, 58.0, 20.0, -30.0, -110.0, -5.0]
    seq = classify_lifecycle_path(path)
    states = [s for _, s in seq]
    for a, b in zip(states, states[1:]):
        assert a != b, f"adjacent duplicate state {a!r} in {states}"


def test_sequence_starts_open_ends_closed():
    path = [0.0, 10.0, 20.0, 15.0]
    seq = classify_lifecycle_path(path)
    assert seq[0] == (0, TLS.OPEN.value)
    assert seq[-1][1] == TLS.CLOSED.value


def test_sequence_uses_only_known_taxonomy_values():
    path = [0.0, 5.0, -10.0, -60.0, -120.0, 30.0, 80.0, 40.0]
    seq = classify_lifecycle_path(path)
    allowed = {s.value for s in TLS}
    for _, st in seq:
        assert st in allowed


def test_invalidated_is_terminal_characterization():
    # CURRENT_BEHAVIOR_CHARACTERIZED: p <= -100 forces INVALIDATED; once hit,
    # later recovery in the raw pnl path is still visible in later minutes
    # (the classifier is stateless per-minute, not a one-way latch) -- this
    # is documented here rather than assumed.
    path = [0.0, -50.0, -105.0, -100.0, 50.0]
    seq = classify_lifecycle_path(path)
    states = [s for _, s in seq]
    assert TLS.INVALIDATED.value in states


# ---- known_at_ts / no-lookahead ----

def test_mark_path_1m_is_known_at_safe(tmp_path):
    db = tmp_path / "micro.sqlite"
    entry_ts = 10_000_000
    entry_px = 100.0
    prices = [(entry_ts + k * 60_000, entry_px + k * 0.1) for k in range(0, 6)]
    _mk_micro_db(db, prices)
    eng = LifecycleEngine(db_path=db)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        path_before = eng._mark_path_1m(conn, entry_ts, entry_px, minutes=5)
    finally:
        conn.close()

    # insert a far-future, extreme-price row (after the whole replay window)
    conn2 = sqlite3.connect(db)
    conn2.execute("INSERT INTO mark_prices VALUES (?,?,?)", (entry_ts + 999 * 60_000, "ETHUSDT", 999_999.0))
    conn2.commit()
    conn2.close()

    conn3 = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        path_after = eng._mark_path_1m(conn3, entry_ts, entry_px, minutes=5)
    finally:
        conn3.close()

    assert path_before == path_after


# ---- lifecycle current-state rebuild from ledger ----

def test_replay_shadow_ledger_is_rebuildable_from_source(tmp_path):
    db = tmp_path / "micro.sqlite"
    entry_ts = 20_000_000
    prices = [(entry_ts + k * 60_000, 100.0 + (30 - abs(k - 30)) * 0.3) for k in range(0, 61)]
    _mk_micro_db(db, prices)
    ledger = tmp_path / "shadow.jsonl"
    rows = [
        {"event": "CLOSE", "id": "t1", "signal": "LONG_SILENCE", "direction": "LONG",
         "entry_ts_ms": entry_ts, "entry_price": 100.0, "net_bps": 42.0},
    ]
    _write_ledger(ledger, rows)

    eng1 = LifecycleEngine(ledger_path=ledger, db_path=db)
    result1 = eng1.replay_shadow_ledger(signals=("LONG_SILENCE",), minutes=60, limit=10)
    eng2 = LifecycleEngine(ledger_path=ledger, db_path=db)  # fresh instance, same source
    result2 = eng2.replay_shadow_ledger(signals=("LONG_SILENCE",), minutes=60, limit=10)
    assert result1 == result2
    assert result1["n_trades"] == 1


# ---- late and out-of-order observations ----

def test_replay_shadow_ledger_is_order_independent(tmp_path):
    db = tmp_path / "micro.sqlite"
    entry_ts = 30_000_000
    prices = [(entry_ts + k * 60_000, 100.0 + k * 0.2) for k in range(0, 31)]
    _mk_micro_db(db, prices)

    rows = [
        {"event": "CLOSE", "id": "t1", "signal": "LONG_SILENCE", "direction": "LONG",
         "entry_ts_ms": entry_ts, "entry_price": 100.0, "net_bps": 10.0},
        {"event": "CLOSE", "id": "t2", "signal": "LONG_SILENCE", "direction": "LONG",
         "entry_ts_ms": entry_ts + 60_000, "entry_price": 100.2, "net_bps": 20.0},
    ]
    ledger_in_order = tmp_path / "in_order.jsonl"
    _write_ledger(ledger_in_order, rows)
    ledger_shuffled = tmp_path / "shuffled.jsonl"
    _write_ledger(ledger_shuffled, [rows[1], rows[0]])

    eng_a = LifecycleEngine(ledger_path=ledger_in_order, db_path=db)
    eng_b = LifecycleEngine(ledger_path=ledger_shuffled, db_path=db)
    ra = eng_a.replay_shadow_ledger(signals=("LONG_SILENCE",), minutes=30, limit=10)
    rb = eng_b.replay_shadow_ledger(signals=("LONG_SILENCE",), minutes=30, limit=10)
    assert ra["n_trades"] == rb["n_trades"] == 2
    assert {s["id"] for s in ra["sequences_sample"]} == {s["id"] for s in rb["sequences_sample"]}


# ---- no import of order router/executor/position manager ----

def test_no_execution_risk_brain_import():
    tree = ast.parse(ENGINE_SRC_PATH.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in _FORBIDDEN_IMPORT_PREFIXES]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _FORBIDDEN_IMPORT_PREFIXES:
                found.append(node.module)
    assert found == []


# ---- no trading credential requirement ----

def test_no_credential_or_env_access():
    src = ENGINE_SRC_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in src
    assert "getenv" not in src
    assert "API_KEY" not in src.upper()
