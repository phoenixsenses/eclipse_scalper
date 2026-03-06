from __future__ import annotations

from pathlib import Path

try:
    from tools.pid_registry import build_watchdog_record, is_identity_match, evaluate_watchdog_identity
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.pid_registry import build_watchdog_record, is_identity_match, evaluate_watchdog_identity


def test_build_watchdog_record_schema() -> None:
    rec = build_watchdog_record(1234, "python -m tools.collection_watchdog", exe_path="python.exe", parent_pid=1, repo_root="x")
    assert rec["role"] == "paper_watchdog"
    assert rec["pid"] == 1234
    assert "start_ts_utc" in rec
    assert rec["cmdline_sig"] == "python -m tools.collection_watchdog"


def test_identity_match() -> None:
    assert is_identity_match("python -m tools.collection_watchdog", "C:\\Python\\python.exe -u -m tools.collection_watchdog --dry-run")
    assert not is_identity_match("python -m tools.collection_watchdog", "python -m execution.bootstrap")


def test_duplicate_start_refuses_when_live_identity_matches() -> None:
    rec = {
        "role": "paper_watchdog",
        "pid": 1234,
        "start_ts_utc": "2026-03-01T12:00:00Z",
        "cmdline_sig": "python -m tools.collection_watchdog",
    }
    observed = {
        "pid": 1234,
        "command_line": "python -u -m tools.collection_watchdog --check-interval-sec 30",
        "creation_ts_utc": "2026-03-01T12:00:01Z",
    }
    ok, reason = evaluate_watchdog_identity(rec, observed, "python -m tools.collection_watchdog")
    assert ok is True
    assert reason == "identity_match"


def test_pid_reuse_does_not_refuse_and_cleans_registry() -> None:
    rec = {
        "role": "paper_watchdog",
        "pid": 1234,
        "start_ts_utc": "2026-03-01T12:00:00Z",
        "cmdline_sig": "python -m tools.collection_watchdog",
    }
    observed = {
        "pid": 1234,
        "command_line": "python -m execution.bootstrap",
        "creation_ts_utc": "2026-03-01T12:00:01Z",
    }
    ok, reason = evaluate_watchdog_identity(rec, observed, "python -m tools.collection_watchdog")
    assert ok is False
    assert reason == "pid_reuse_signature_mismatch"


def test_missing_process_cleans_registry() -> None:
    rec = {
        "role": "paper_watchdog",
        "pid": 1234,
        "start_ts_utc": "2026-03-01T12:00:00Z",
        "cmdline_sig": "python -m tools.collection_watchdog",
    }
    ok, reason = evaluate_watchdog_identity(rec, None, "python -m tools.collection_watchdog")
    assert ok is False
    assert reason == "pid_not_running"
