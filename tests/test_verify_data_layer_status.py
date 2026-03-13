from __future__ import annotations

import sys
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verify_data_layer as vdl


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"verify_data_layer_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_verify_running(monkeypatch):
    tmp_path = _mk_local_tmp()
    db = tmp_path / "micro.db"
    csv = tmp_path / "event_diary.csv"
    db.write_bytes(b"a" * 10)
    csv.write_text("x\n", encoding="utf-8")

    rows = [
        {"ProcessId": 1, "CommandLine": "python -m data.microstructure_collector"},
        {"ProcessId": 2, "CommandLine": "python -m data.event_diary"},
    ]
    monkeypatch.setattr(vdl, "_list_python_processes", lambda: (rows, None))
    stats = iter([(10, 10.0), (1, 10.0), (20, 11.0), (2, 11.0)])
    monkeypatch.setattr(vdl, "_file_stats", lambda path: next(stats))
    monkeypatch.setattr(vdl.time, "sleep", lambda _: None)
    monkeypatch.setattr(vdl.time, "time", lambda: 150.0)
    try:
        ok, details = vdl.verify(db, csv, wait_sec=1, min_db_growth_bytes=1)
        assert ok is True
        assert details["overall_status"] == "running"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_verify_degraded_when_collector_live_but_diary_not_progressing(monkeypatch):
    tmp_path = _mk_local_tmp()
    db = tmp_path / "micro.db"
    csv = tmp_path / "event_diary.csv"
    db.write_bytes(b"a" * 10)
    csv.write_text("x\n", encoding="utf-8")

    rows = [{"ProcessId": 1, "CommandLine": "python -m data.microstructure_collector"}]
    monkeypatch.setattr(vdl, "_list_python_processes", lambda: (rows, None))
    stats = iter([(10, 10.0), (1, 10.0), (20, 11.0), (1, 10.0)])
    monkeypatch.setattr(vdl, "_file_stats", lambda path: next(stats))
    monkeypatch.setattr(vdl.time, "sleep", lambda _: None)
    monkeypatch.setattr(vdl.time, "time", lambda: 150.0)
    try:
        ok, details = vdl.verify(db, csv, wait_sec=1, min_db_growth_bytes=1)
        assert ok is False
        assert details["overall_status"] == "degraded"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_verify_stopped_when_no_process_and_no_progress(monkeypatch):
    tmp_path = _mk_local_tmp()
    db = tmp_path / "micro.db"
    csv = tmp_path / "event_diary.csv"
    db.write_bytes(b"a" * 10)
    csv.write_text("x\n", encoding="utf-8")

    monkeypatch.setattr(vdl, "_list_python_processes", lambda: ([], None))
    stats = iter([(10, 10.0), (1, 10.0), (10, 10.0), (1, 10.0)])
    monkeypatch.setattr(vdl, "_file_stats", lambda path: next(stats))
    monkeypatch.setattr(vdl.time, "sleep", lambda _: None)
    monkeypatch.setattr(vdl.time, "time", lambda: 150.0)
    try:
        ok, details = vdl.verify(db, csv, wait_sec=1, min_db_growth_bytes=1)
        assert ok is False
        assert details["overall_status"] == "stopped"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
