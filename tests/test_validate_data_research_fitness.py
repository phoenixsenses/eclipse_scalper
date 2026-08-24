from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import check_data_ready as cdr
from tools import validate_data_research_fitness as vdrf


FIXTURE_DB = Path("tests/fixtures/microstructure_sample.db")


def _mk_local_tmp() -> Path:
    root = Path("localtests") / "research_fitness" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_analyze_research_fitness_warns_on_degraded_but_usable_fixture() -> None:
    tmp = _mk_local_tmp()
    csv_path = tmp / "event_diary.csv"
    csv_path.write_text("ts_ms,symbol,note\n1700000000000,BTCUSDT,ok\n", encoding="utf-8")

    payload = vdrf.analyze_research_fitness(
        db_path=FIXTURE_DB,
        csv_path=csv_path,
        symbols=["BTCUSDT", "ETHUSDT"],
        fresh_sec=9_999_999_999,
        min_trade_rows_per_symbol=10,
        now=1_700_000_100.0,
    )

    assert payload["status"] == "warn"
    assert payload["db_ready"] is True
    assert payload["contract"]["status"] == "warn"
    assert payload["feature_stats"]["BTCUSDT"]["has_mid"] is True
    assert payload["feature_stats"]["BTCUSDT"]["has_trade_intensity"] is True
    assert "contract_warn" in payload["warnings"]
    assert "no_spread:BTCUSDT" in payload["warnings"]
    assert "no_spread:ETHUSDT" in payload["warnings"]


def test_analyze_research_fitness_fails_when_required_inputs_missing() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "broken.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
            conn.commit()
        finally:
            conn.close()

        payload = vdrf.analyze_research_fitness(
            db_path=db,
            csv_path=tmp / "missing.csv",
            symbols=["BTCUSDT"],
            fresh_sec=120,
            now=1_700_000_100.0,
        )
        assert payload["status"] == "fail"
        assert "missing_event_diary_csv" in payload["failures"]
        assert "db_not_ready" in payload["failures"]
        assert "contract_fail" in payload["failures"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _mk_bounded_db(db_path: Path, ts_ms: int, rows: int = 20) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, mark_price REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, price REAL NOT NULL, quantity REAL NOT NULL)"
        )
        for i in range(rows):
            conn.execute(
                "INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES(?,?,?)",
                (ts_ms - i * 1000, "ETHUSDT", 100.0 + i),
            )
            conn.execute(
                "INSERT INTO agg_trades(ts_ms, symbol, price, quantity) VALUES(?,?,?,?)",
                (ts_ms - i * 1000, "ETHUSDT", 100.0 + i, 1.0),
            )
        conn.commit()
    finally:
        conn.close()


def _add_large_unindexed_unrelated_table(db_path: Path, rows: int) -> None:
    """Mirrors the real production hazard: an unrelated, unindexed table
    (shaped like detector_heartbeat) large enough that a full scan is not
    free."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE detector_heartbeat (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, note TEXT)")
        conn.executemany(
            "INSERT INTO detector_heartbeat (ts_ms, note) VALUES (?, ?)",
            ((1_700_000_000_000 - i, "x" * 64) for i in range(rows)),
        )
        conn.commit()
    finally:
        conn.close()


def test_large_unrelated_table_excluded_from_allowlisted_scan() -> None:
    """Only the RESEARCH_FITNESS_TABLE_ALLOWLIST tables are inspected --
    detector_heartbeat never appears in the returned diagnostics at all,
    regardless of its size."""
    tmp = _mk_local_tmp()
    try:
        db = tmp / "prod_shaped.db"
        _mk_bounded_db(db, ts_ms=1_700_000_000_000)
        _add_large_unindexed_unrelated_table(db, rows=50_000)
        conn = sqlite3.connect(str(db))
        try:
            diags = cdr.inspect_tables(
                conn, now=1_700_000_100.0, table_allowlist=cdr.RESEARCH_FITNESS_TABLE_ALLOWLIST
            )
        finally:
            conn.close()
        names = {d.table for d in diags}
        assert names == {"mark_prices", "agg_trades"}
        assert "detector_heartbeat" not in names
        assert "liquidations" not in names  # allowlisted but absent from this fixture -- not fabricated
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unbounded_scan_still_available_for_other_callers() -> None:
    """table_allowlist=None (the default) preserves the pre-fix full-scan
    behavior for callers other than research fitness -- this fix must not
    silently change behavior for unrelated consumers of inspect_tables."""
    tmp = _mk_local_tmp()
    try:
        db = tmp / "prod_shaped.db"
        _mk_bounded_db(db, ts_ms=1_700_000_000_000)
        _add_large_unindexed_unrelated_table(db, rows=10)
        conn = sqlite3.connect(str(db))
        try:
            diags = cdr.inspect_tables(conn, now=1_700_000_100.0)
        finally:
            conn.close()
        names = {d.table for d in diags}
        assert "detector_heartbeat" in names
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_research_fitness_bounded_runtime_against_production_shaped_fixture() -> None:
    """Acceptance proof for Part B: a production-shaped database (small
    required tables + a large unindexed unrelated table) must not hang --
    the whole evaluation must complete quickly because detector_heartbeat is
    never scanned."""
    tmp = _mk_local_tmp()
    try:
        db = tmp / "prod_shaped.db"
        _mk_bounded_db(db, ts_ms=1_700_000_000_000, rows=50)
        _add_large_unindexed_unrelated_table(db, rows=200_000)
        csv_path = tmp / "event_diary.csv"
        csv_path.write_text("ts_ms,symbol,note\n1700000000000,ETHUSDT,ok\n", encoding="utf-8")

        start = time.monotonic()
        payload = vdrf.analyze_research_fitness(
            db_path=db,
            csv_path=csv_path,
            symbols=["ETHUSDT"],
            fresh_sec=9_999_999_999,
            now=1_700_000_100.0,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"research fitness evaluation took {elapsed:.2f}s against a production-shaped fixture"
        assert payload["status"] in ("pass", "warn", "fail")
        assert "detector_heartbeat" not in json.dumps(payload["db_ready_details"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_unrelated_tables_do_not_block_fitness() -> None:
    """A database with only the allowlisted tables (no detector_heartbeat,
    no any other operational table at all) must evaluate exactly the same
    as one that has them -- unrelated tables are irrelevant either way."""
    tmp = _mk_local_tmp()
    try:
        db = tmp / "minimal.db"
        _mk_bounded_db(db, ts_ms=1_700_000_000_000, rows=50)
        csv_path = tmp / "event_diary.csv"
        csv_path.write_text("ts_ms,symbol,note\n1700000000000,ETHUSDT,ok\n", encoding="utf-8")

        payload = vdrf.analyze_research_fitness(
            db_path=db, csv_path=csv_path, symbols=["ETHUSDT"],
            fresh_sec=9_999_999_999, now=1_700_000_100.0,
        )
        assert payload["db_ready"] is True
        assert "db_not_ready" not in payload["failures"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_required_table_is_deterministically_blocked() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "no_agg_trades.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute(
                "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, mark_price REAL NOT NULL)"
            )
            conn.execute(
                "INSERT INTO mark_prices(ts_ms, symbol, mark_price) VALUES(?,?,?)",
                (1_700_000_000_000, "ETHUSDT", 100.0),
            )
            conn.commit()
        finally:
            conn.close()
        csv_path = tmp / "event_diary.csv"
        csv_path.write_text("ts_ms,symbol,note\n1700000000000,ETHUSDT,ok\n", encoding="utf-8")

        from tools.research_fitness_report import build_report

        report = build_report(
            db_path=db, csv_path=csv_path, symbols=["ETHUSDT"],
            fresh_sec=9_999_999_999, stale_after_sec=3600, now=1_700_000_100.0,
        )
        assert report["status"] == "blocked"
        assert report["failures"]  # deterministically fails, not silently "ready"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluation_never_mutates_the_database() -> None:
    """analyze_research_fitness is read-only (SELECT/PRAGMA only, no
    INSERT/UPDATE/DDL) under every status path -- a failed evaluation
    cannot corrupt or alter the source database."""
    tmp = _mk_local_tmp()
    try:
        db = tmp / "readonly_check.db"
        _mk_bounded_db(db, ts_ms=1_700_000_000_000, rows=5)
        before = hashlib.sha256(db.read_bytes()).hexdigest()

        vdrf.analyze_research_fitness(
            db_path=db,
            csv_path=tmp / "missing_csv_forces_failure.csv",
            symbols=["ETHUSDT"],
            fresh_sec=120,
            now=1_700_000_100.0,
        )

        after = hashlib.sha256(db.read_bytes()).hexdigest()
        assert before == after
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_writes_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        csv_path = tmp / "event_diary.csv"
        out_json = tmp / "fitness.json"
        out_md = tmp / "fitness.md"
        csv_path.write_text("ts_ms,symbol,note\n1700000000000,ETHUSDT,ok\n", encoding="utf-8")

        # FIXTURE_DB's rows are all ~2023-era (ts_ms=1_700_000_000_000). The
        # recent-activity window in _symbol_sample_stats/_feature_fitness is
        # relative to real wall-clock "now" (the CLI never accepts a --now
        # override -- that would be a footgun in production), so this test
        # freezes time.time() to the fixture's own era, exactly like the
        # other fixture-based tests above pin now=1_700_000_100.0 directly.
        monkeypatch.setattr(vdrf.time, "time", lambda: 1_700_000_100.0)
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(FIXTURE_DB),
                "--csv",
                str(csv_path),
                "--symbols",
                "BTCUSDT,ETHUSDT",
                "--fresh-sec",
                "9999999999",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert vdrf.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "warn"
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
