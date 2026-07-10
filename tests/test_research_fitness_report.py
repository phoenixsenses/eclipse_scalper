from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.research_fitness_report as rf
from tools.research_fitness_report import (
    ProtectedOperationalOutputError,
    build_report,
    main as rf_main,
)


def _mk_db(path: Path, ts_ms: int, rows: int = 20) -> None:
    conn = sqlite3.connect(str(path))
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


def test_build_report_writes_only_its_own_schema_fields(tmp_path):
    db = tmp_path / "micro.db"
    csv = tmp_path / "event_diary.csv"
    csv.write_text("ts,event\n1,x\n", encoding="utf-8")
    _mk_db(db, ts_ms=1_700_000_000_000)

    report = build_report(
        db_path=db,
        csv_path=csv,
        symbols=["ETHUSDT"],
        fresh_sec=120,
        stale_after_sec=3600,
        now=1_700_000_000.5,
    )

    assert report["status"] in ("ready", "limited", "blocked")
    assert "evaluated_at_utc" in report
    assert report["symbols"] == ["ETHUSDT"]
    assert report["stale_after_sec"] == 3600


def test_report_degrades_safely_on_missing_db(tmp_path):
    report = build_report(
        db_path=tmp_path / "does_not_exist.db",
        csv_path=tmp_path / "does_not_exist.csv",
        symbols=["ETHUSDT"],
        fresh_sec=120,
        stale_after_sec=3600,
    )
    assert report["status"] == "blocked"
    assert report["error"] is None  # analyze_research_fitness itself handles missing-db gracefully


def test_report_degrades_safely_on_evaluation_exception(monkeypatch, tmp_path):
    import tools.research_fitness_report as rf

    def _boom(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(rf, "analyze_research_fitness", _boom)
    report = rf.build_report(
        db_path=tmp_path / "whatever.db",
        csv_path=tmp_path / "whatever.csv",
        symbols=["ETHUSDT"],
        fresh_sec=120,
        stale_after_sec=3600,
    )
    assert report["status"] == "blocked"
    assert "synthetic failure" in (report["error"] or "")


def test_cli_writes_only_the_dedicated_output_file(monkeypatch, tmp_path):
    """The production CLI has no --out; output-path injection for tests goes
    through the internal API only (here: DEFAULT_OUT_PATH monkeypatch)."""
    db = tmp_path / "micro.db"
    csv = tmp_path / "event_diary.csv"
    csv.write_text("ts,event\n1,x\n", encoding="utf-8")
    _mk_db(db, ts_ms=1_700_000_000_000)
    out_path = tmp_path / "research_fitness.json"

    before = set(tmp_path.iterdir())
    monkeypatch.setattr(rf, "DEFAULT_OUT_PATH", out_path)
    monkeypatch.setattr(
        sys, "argv",
        ["research_fitness_report.py", "--db", str(db), "--csv", str(csv), "--symbols", "ETHUSDT"],
    )
    rc = rf_main()
    after = set(tmp_path.iterdir())

    assert rc in (0, 1, 2)
    assert out_path.exists()
    new_files = after - before
    assert new_files == {out_path}, f"unexpected extra files written: {new_files}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] in ("ready", "limited", "blocked")


# --- protected operational-output enforcement (single-writer, Part A of the
# --- 2026-07-10 final corrective round) ------------------------------------


def test_default_output_path_is_exactly_research_fitness_json():
    assert rf.DEFAULT_OUT_PATH == Path("logs/health/research_fitness.json")


def test_cli_cannot_select_an_output_path_at_all(monkeypatch):
    """--out was removed from the production CLI: argparse must reject it,
    so no CLI invocation can ever target overall.json (or anything else)."""
    monkeypatch.setattr(
        sys, "argv",
        ["research_fitness_report.py", "--out", "logs/health/overall.json"],
    )
    with pytest.raises(SystemExit) as exc_info:
        rf_main()
    assert exc_info.value.code == 2  # argparse: unrecognized arguments


def test_internal_writer_rejects_every_protected_operational_output(tmp_path):
    """Rejection is by resolved basename, case-insensitively, regardless of
    directory -- and performs zero writes (no target file, no temp file)."""
    protected = [
        "overall.json",
        "watchdog.json",
        "WATCHDOG_STATUS.json",
        "collector.json",
        "bookticker.json",
        "paper_trader.json",
        "replay.json",
    ]
    for name in protected:
        target = tmp_path / name
        with pytest.raises(ProtectedOperationalOutputError):
            rf._atomic_write_json(target, {"status": "ready"})
        assert not target.exists(), name
    assert list(tmp_path.glob(".tmp_*")) == []
    assert list(tmp_path.iterdir()) == []  # zero writes of any kind


def test_internal_writer_rejects_relative_alias_resolving_to_protected(tmp_path):
    alias = tmp_path / "sub" / ".." / "overall.json"
    with pytest.raises(ProtectedOperationalOutputError):
        rf._atomic_write_json(alias, {"status": "ready"})
    assert not (tmp_path / "overall.json").exists()
    assert not (tmp_path / "sub").exists()  # not even the parent dir was created


def test_internal_writer_rejects_windows_case_insensitive_alias(tmp_path):
    for name in ("OVERALL.JSON", "Overall.Json", "PAPER_TRADER.JSON"):
        with pytest.raises(ProtectedOperationalOutputError):
            rf._atomic_write_json(tmp_path / name, {"status": "ready"})
    assert list(tmp_path.iterdir()) == []


def test_collection_watchdog_wrapper_cannot_bypass_protection(monkeypatch, tmp_path):
    """The deprecated wrapper delegates to this module's _atomic_write_json,
    so its own --out inherits the identical guard -- rejected before any
    write, with no bypass of its own."""
    import tools.collection_watchdog as cw

    target = tmp_path / "overall.json"
    monkeypatch.setattr(
        sys, "argv",
        ["collection_watchdog.py", "--db", str(tmp_path / "missing.db"),
         "--csv", str(tmp_path / "missing.csv"), "--out", str(target)],
    )
    with pytest.raises(ProtectedOperationalOutputError):
        cw.main()
    assert not target.exists()
    assert list(tmp_path.glob(".tmp_*")) == []


def test_module_never_imports_operational_health_writers():
    """Guard: this module must never gain a dependency on the operational
    overall.json/watchdog.json writers -- it is advisory-only by design.
    Checked structurally via ast (imports + non-docstring string literals),
    so docstrings/comments documenting the constraint by name never trip it.
    Protected filenames ARE allowed as literals in exactly one place: the
    PROTECTED_OPERATIONAL_OUTPUT_BASENAMES rejection set, whose whole job is
    to name what may never be written."""
    import ast
    import inspect

    source = inspect.getsource(rf)
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert "write_overall_health" not in imported
    assert "write_component_health" not in imported

    allowed_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PROTECTED_OPERATIONAL_OUTPUT_BASENAMES" in names:
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant):
                        allowed_ids.add(id(c))

    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstring_ids.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_ids:
            low = node.value.lower()
            if any(p in low for p in ("overall.json", "watchdog.json", "watchdog_status")):
                assert id(node) in allowed_ids, (
                    f"protected filename literal {node.value!r} outside the rejection set"
                )
