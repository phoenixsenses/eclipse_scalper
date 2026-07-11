"""Lifecycle tests for tools/liquidation_silence_scheduler.py (the periodic-
scheduling corrective batch, see reports/research/s34/
LIQUIDATION_SILENCE_DETECTOR_PERIODIC_SCHEDULING_DESIGN.md and its
corrective-implementation follow-up).

Scope: wrapper LIFECYCLE only (cadence representation, single-instance/
overlap protection, stale-PID recovery, stop behavior, import-graph safety,
PowerShell opt-in wiring) -- NOT the detector's own read/policy correctness,
which is already covered by tests/test_liquidation_silence_detector.py and
tests/test_liquidation_silence_policy.py, unmodified and unaffected by this
module.

No test here touches a live exchange endpoint, mutates a production DB,
starts a real persistent background process, or depends on Windows Task
Scheduler or any of the 7 foreign-owned dirty files -- every run_once() call
this file exercises is monkeypatched to a canned in-memory payload; nothing
here reads or writes the real logs/health/liquidation_silence.json or the
real data/microstructure.db.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import liquidation_silence_scheduler as sched


@pytest.fixture(autouse=True)
def _reset_module_global_state():
    """_STOP_REQUESTED and _LOCK_STATE are both process-global module state
    -- reset/release them around every test in this file so one test's stop
    request or held OS lock handle can never leak into the next test (e.g.
    an assertion failure mid-test, before that test's own release call)."""
    sched.reset_stop_flag()
    yield
    sched.reset_stop_flag()
    handle = sched._LOCK_STATE.get("handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
        sched._LOCK_STATE["handle"] = None


def _healthy_payload(**overrides):
    payload = {
        "severity": "GREEN",
        "status": "HEALTHY",
        "error": None,
        "detector_runtime_sec": 0.002,
    }
    payload.update(overrides)
    return payload


# --- case 1: single-run invocation succeeds ----------------------------------

def test_single_run_invocation_succeeds(monkeypatch, tmp_path):
    log_path = tmp_path / "sched.log"
    monkeypatch.setattr(sched, "run_once", lambda **kw: _healthy_payload())

    record = sched.run_cycle(log_path=log_path, quiet=True)

    assert record["outcome"] == "SUCCESS"
    assert record["severity"] == "GREEN"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["outcome"] == "SUCCESS"


# --- case 2: default cadence represented correctly without sleeping ---------

def test_default_cadence_represented_without_sleeping(monkeypatch, tmp_path):
    assert sched.DEFAULT_CADENCE_SEC == 300  # grounded value, see module docstring
    pid_path = tmp_path / "liq.pid"
    log_path = tmp_path / "liq.log"
    sleep_calls = []

    def fake_sleep(total_sec, poll_sec=1.0):
        sleep_calls.append(total_sec)
        sched._STOP_REQUESTED = True  # stop after capturing the cadence value -- no real sleep occurs

    monkeypatch.setattr(sched, "interruptible_sleep", fake_sleep)
    monkeypatch.setattr(sched, "run_once", lambda **kw: _healthy_payload())
    monkeypatch.setattr(
        sys, "argv",
        ["liquidation_silence_scheduler.py", "--pid-path", str(pid_path), "--log-path", str(log_path),
         "--startup-delay-sec", "0"],
    )

    rc = sched.main()

    assert rc == 0
    assert sleep_calls == [300]  # the loop's cadence sleep, using the unmodified default


# --- case 3: existing live PID blocks a duplicate instance -------------------
#
# CORRECTIVE NOTE (2026-07-11, independent re-review finding F3): these
# tests previously monkeypatched is_pid_alive_and_matching (now removed --
# see the module's corrective note) to simulate "a live owner exists,"
# which meant the actual acquisition logic being exercised was the mock,
# not production code, and never proved genuine concurrent-acquisition
# safety. Rewritten below to hold a REAL OS-level lock (via an actual
# acquire_single_instance_lock() call that is deliberately not released)
# and assert a second, genuinely concurrent acquire attempt against the
# same real lock fails -- no mocking anywhere in these three tests.

def test_existing_live_pid_blocks_duplicate_instance(tmp_path):
    pid_path = tmp_path / "liq.pid"
    assert sched.acquire_single_instance_lock(pid_path) is True  # instance A: real lock held, not released

    acquired_b = sched.acquire_single_instance_lock(pid_path)  # instance B: genuinely concurrent attempt

    assert acquired_b is False
    # Instance A's PID (this process, since both "instances" run in the same
    # test process) remains the recorded owner -- instance B's failed
    # attempt must not have overwritten it.
    assert pid_path.read_text(encoding="utf-8").strip() == str(os.getpid())


# --- case 4: stale PID is recovered safely -----------------------------------

def test_stale_pid_is_recovered_safely(tmp_path):
    """A leftover informational PID file from a prior hard kill (arbitrary,
    possibly-dead-looking content, and critically: no live OS lock behind
    it, since a hard kill releases the lock automatically -- see the
    module's acquire_single_instance_lock docstring) must not block a fresh
    acquire. The new implementation doesn't need to classify the old value
    as "dead" at all: it simply attempts the real OS lock, which succeeds
    because nothing holds it, and overwrites the stale content."""
    pid_path = tmp_path / "liq.pid"
    pid_path.write_text("999999", encoding="utf-8")  # stale leftover value, no process backs it

    acquired = sched.acquire_single_instance_lock(pid_path)

    assert acquired is True
    assert pid_path.read_text(encoding="utf-8").strip() == str(os.getpid())


# --- case 5: overlapping detector runs are impossible ------------------------

def test_overlapping_runs_impossible_single_instance_enforced(tmp_path):
    """Cross-process overlap: a second, genuinely concurrent acquire attempt
    while the real OS lock is still held must fail (no mocking -- see the
    corrective note above). Within-process overlap is structurally
    impossible by construction -- main()'s while-loop calls run_cycle()
    synchronously every iteration, never concurrently (no threading/async
    anywhere in this module; verified below directly from source)."""
    pid_path = tmp_path / "liq.pid"

    assert sched.acquire_single_instance_lock(pid_path) is True
    assert sched.acquire_single_instance_lock(pid_path) is False
    sched.release_single_instance_lock(pid_path)
    assert sched.acquire_single_instance_lock(pid_path) is True  # released -> a fresh acquire succeeds

    # Also lock in the structural, in-process guarantee directly from source:
    # no threading/asyncio/multiprocessing primitive exists anywhere in this
    # module that could run two detector cycles concurrently.
    src = inspect.getsource(sched)
    for forbidden in ("threading.Thread", "asyncio", "multiprocessing", "concurrent.futures"):
        assert forbidden not in src


# --- case 6: detector timeout/failure is observable --------------------------

def test_detector_exception_is_observable(monkeypatch, tmp_path):
    log_path = tmp_path / "sched.log"

    def failing_run_once(**kwargs):
        raise RuntimeError("simulated hang/timeout")

    monkeypatch.setattr(sched, "run_once", failing_run_once)
    record = sched.run_cycle(log_path=log_path, quiet=True)

    assert record["outcome"] == "WRAPPER_EXCEPTION"
    assert "simulated hang/timeout" in record["exception"]
    logged = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert logged["outcome"] == "WRAPPER_EXCEPTION"


def test_detector_error_payload_is_observable(monkeypatch, tmp_path):
    """The detector's own typed error field (e.g. DB_LOCKED) surfaces as
    outcome=DETECTOR_ERROR without the wrapper raising."""
    log_path = tmp_path / "sched.log"
    monkeypatch.setattr(
        sched, "run_once",
        lambda **kw: _healthy_payload(severity="UNKNOWN", status="UNKNOWN", error=[{"code": "DB_LOCKED", "message": "x"}]),
    )

    record = sched.run_cycle(log_path=log_path, quiet=True)

    assert record["outcome"] == "DETECTOR_ERROR"
    assert record["detector_error"] == [{"code": "DB_LOCKED", "message": "x"}]


# --- case 7: stop request exits cleanly --------------------------------------

def test_stop_request_exits_cleanly(monkeypatch, tmp_path):
    pid_path = tmp_path / "liq.pid"
    log_path = tmp_path / "liq.log"
    monkeypatch.setattr(sched, "run_once", lambda **kw: _healthy_payload())

    def fake_sleep(total_sec, poll_sec=1.0):
        sched._STOP_REQUESTED = True  # simulates a signal handler having fired mid-cycle

    monkeypatch.setattr(sched, "interruptible_sleep", fake_sleep)
    monkeypatch.setattr(
        sys, "argv",
        ["liquidation_silence_scheduler.py", "--pid-path", str(pid_path), "--log-path", str(log_path),
         "--startup-delay-sec", "0"],
    )

    rc = sched.main()

    assert rc == 0
    # Clean exit releases its own PID file (release_single_instance_lock) --
    # distinct from a hard Stop-Process -Force kill, which cannot run this
    # cleanup and is exactly why acquire_single_instance_lock treats a
    # leftover PID file as recoverable stale state (case 4 above).
    assert not pid_path.exists()


# --- case 8: no trading/execution module is imported or invoked -------------

def test_no_trading_or_execution_module_imported():
    src = inspect.getsource(sched)
    forbidden_tokens = (
        "execution.", "s34_state_machine_live_executor", "s34_v_engine_live_executor",
        "s34_live_order_executor", "ccxt", "binance.client", "place_order", "create_order",
        "new_order", ".env",
    )
    for token in forbidden_tokens:
        assert token not in src, f"forbidden token {token!r} found in scheduler module source"

    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    stdlib = {
        "argparse", "json", "msvcrt", "os", "signal", "time", "datetime", "pathlib", "typing",
        "__future__",
    }
    allowed_non_stdlib = {"tools.liquidation_silence_detector"}
    non_stdlib = {name for name in imported if name.split(".")[0] not in stdlib}
    assert non_stdlib <= allowed_non_stdlib, f"unexpected import(s): {non_stdlib - allowed_non_stdlib}"


# --- case 9: default repository startup does not enable the wrapper ---------
#
# CORRECTIVE NOTE (2026-07-11, independent re-review finding F4, honestly
# documented rather than "fixed" -- flagged by the reviewer as LOW-MEDIUM,
# not blocking): the two tests below (this one and case 10) are static
# source-text assertions against the .ps1 files, not executed PowerShell
# behavior. This is a deliberate, not an oversight, trade-off: actually
# invoking start_eclipse.ps1/stop_eclipse.ps1 from a test -- with or
# without the opt-in flag -- would spawn real background python.exe
# processes (Start-RegisteredPythonProcess's whole purpose) as a side
# effect of "testing," directly violating this file's own stated
# invariant that no test here starts a real persistent background
# process. A safer middle ground (Pester-style mocking of Start-Process/
# Start-RegisteredPythonProcess, or PowerShell -WhatIf plumbing) was not
# implemented in this corrective batch -- it would require adding
# PowerShell-side test infrastructure this repository does not currently
# have, which is out of the minimum corrective scope authorized here.
# These two tests prove the source text contains the right tokens in the
# right structural region (which the parser-validated block-slicing below
# targets specifically, not a whole-file substring search) -- a real but
# bounded form of assurance, not a substitute for runtime verification.

def test_default_repo_startup_does_not_enable_wrapper():
    ps1_path = Path(__file__).resolve().parents[1] / "start_eclipse.ps1"
    text = ps1_path.read_text(encoding="utf-8")

    assert "[switch]$EnableLiquidationSilenceScheduler" in text
    # A bare [switch] parameter has no way to default to $true via this
    # declaration syntax in PowerShell -- it is $false unless the flag is
    # explicitly passed on the command line. This alone proves the role is
    # opt-in; the else-branch below additionally proves a prior opt-in run
    # never silently lingers across a plain (non-opted-in) invocation.
    assert "if ($EnableLiquidationSilenceScheduler) {" in text
    assert (
        'Stop-PythonProcessesByCommandLine -Role "liquidation_silence_scheduler" '
        '-Needle "tools.liquidation_silence_scheduler"'
    ) in text
    assert 'Set-Content -LiteralPath (Join-Path $pidDir "liquidation_silence_scheduler.pid") -Value "0"' in text
    # "Disabled = $true" must appear within the SAME if/else construct that
    # gates this role -- not merely somewhere in the file (e.g. the
    # unrelated -EnableLive else-branch above it, which also sets
    # Disabled = $true for its own role).
    block_start = text.index("if ($EnableLiquidationSilenceScheduler) {")
    block_end = text.index("\n}\n", block_start) + len("\n}\n")
    assert "Disabled = $true" in text[block_start:block_end]


# --- case 10: explicit opt-in wiring selects only this wrapper --------------

def test_explicit_opt_in_selects_only_this_wrapper():
    ps1_path = Path(__file__).resolve().parents[1] / "start_eclipse.ps1"
    text = ps1_path.read_text(encoding="utf-8")
    block_start = text.index("if ($EnableLiquidationSilenceScheduler) {")
    block_end = text.index("\n}\n", block_start) + len("\n}\n")
    block = text[block_start:block_end]

    assert '"-m", "tools.liquidation_silence_scheduler"' in block
    assert '-CommandNeedle "tools.liquidation_silence_scheduler"' in block

    other_role_needles = (
        "scripts\\collector_supervisor.py", "tools.heartbeat_watchdog", "data.bookticker_collector",
        "data.oi_spot_poller", "tools.s34_shadow_paper_runner.py", "tools.s34_live_chart.py",
        "tools.s34_v_engine_v02_shadow_mirror", "tools.s34_realtime_shadow_runner",
        "tools.s34_state_machine_live_executor", "tools.orderflow_chart.py", "tools.s34_replay.py",
    )
    for needle in other_role_needles:
        assert needle not in block, f"opt-in block unexpectedly references another role's needle: {needle!r}"

    # stop_eclipse.ps1's scoped stop switch is symmetric: it must also touch
    # only this wrapper's process/PID file, never another role's.
    stop_ps1 = (Path(__file__).resolve().parents[1] / "stop_eclipse.ps1").read_text(encoding="utf-8")
    scoped_start = stop_ps1.index("if ($LiquidationSilenceSchedulerOnly) {")
    scoped_end = stop_ps1.index("\n}\n", scoped_start) + len("\n}\n")
    scoped_block = stop_ps1[scoped_start:scoped_end]
    assert "tools.liquidation_silence_scheduler" in scoped_block
    for needle in other_role_needles:
        assert needle not in scoped_block
