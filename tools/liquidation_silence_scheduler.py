"""Minimal persistent-loop scheduler wrapper around the accepted, disabled-
by-default one-shot tools/liquidation_silence_detector.py.

Adds NO new detection logic: every cycle calls the already-accepted
tools.liquidation_silence_detector.run_once(evaluation_mode=LIVE) verbatim
-- same bounded read-only DB access (mode=ro), same atomic single-file
output (logs/health/liquidation_silence.json via
tools.health_state.write_component_health), same policy/thresholds. This
module only adds: a cadence loop, a PID file for single-instance
protection, graceful-stop handling, and a per-cycle success/failure log.
See reports/research/s34/
LIQUIDATION_SILENCE_DETECTOR_PERIODIC_SCHEDULING_DESIGN.md (recommended
Option B) and reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md
(the accepted detector this wraps) for full background.

Disabled by default: nothing in this repository imports or launches this
module automatically. It is not wired into tools/heartbeat_watchdog.py, not
started by start_eclipse.ps1 unless the explicit
-EnableLiquidationSilenceScheduler opt-in flag is passed (a separate,
explicitly operator-authorized change -- see that script). Running this
module, directly or via that flag, is always an explicit operator action.

No order/execution/exchange-write/trading path is reachable from this
module: it imports only tools.liquidation_silence_detector's already-
accepted run_once()/MODE_LIVE and stdlib. Every DB access remains strictly
mode=ro, owned entirely by the wrapped detector, unmodified.

CORRECTIVE NOTE (2026-07-11, independent re-review finding F3): the
original single-instance lock in this module was a plain read-PID ->
check-liveness -> write-PID sequence with no atomicity between the check
and the write -- two processes racing to start within the same narrow
window could both observe "no live owner" and both proceed, defeating the
single-instance guarantee. Replaced below with an OS-enforced exclusive
byte-range lock (msvcrt.locking, Windows stdlib -- this repository targets
Windows exclusively) held open for the process's entire lifetime: the OS
itself arbitrates any race, and automatically releases the lock if this
process dies for any reason (including a hard Stop-Process -Force kill),
which also makes stale-lock recovery automatic and correct with no PID-
liveness/command-line inspection needed at all. The psutil-optional /
Get-CimInstance-fallback cmdline lookup this replaced is removed as dead
code, not merely superseded.

Bounded-invocation note: tools/liquidation_silence_detector.py's own
_open_ro() already sets a 5s sqlite busy-timeout and every query is
indexed/LIMIT-bounded (see that module's docstring) -- there is no known
unbounded-blocking path for a single cycle to hang on indefinitely, so this
wrapper does not add a redundant outer kill-timeout (which would need
threading/subprocess machinery of its own, a real complexity/risk cost for
a call already provably bounded). Instead, cycle_duration_sec is recorded
on every cycle (see run_cycle) so an anomalously slow cycle is still
observable in the log without needing a preemptive kill.
"""
from __future__ import annotations

import argparse
import json
import msvcrt
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.liquidation_silence_detector import MODE_HISTORICAL_REPLAY, MODE_LIVE, run_once

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PID_PATH = ROOT / "logs" / "pids" / "liquidation_silence_scheduler.pid"
DEFAULT_LOG_PATH = ROOT / "logs" / "liquidation_silence_scheduler.log"

# Cadence is grounded in tools/liquidation_silence_policy.py's own
# CONTROL_STREAM_FRESH_AGE_SEC=300.0 (see the design report referenced in
# the module docstring above) -- not an invented number. This is also the
# exact value tools/heartbeat_watchdog.py's
# LIQUIDATION_SILENCE_SCHEDULER_CADENCE_SEC constant assumes when deriving
# LIQUIDATION_SILENCE_MAX_AGE_SEC (its staleness budget for this wrapper's
# output); if this default is ever changed, that constant should be
# reviewed too.
DEFAULT_CADENCE_SEC = 300
# Matches the established REST_FALLBACK_GRACE_SEC=60.0 precedent in
# tools/native_ws_health_policy.py -- the only existing "benign transient
# state" grace constant in this codebase, reused rather than inventing a
# new number.
DEFAULT_STARTUP_DELAY_SEC = 60

# Command-line needle every PowerShell lifecycle script (start_eclipse.ps1,
# status_eclipse.ps1, stop_eclipse.ps1) matches against, independently
# declared there too (PowerShell cannot import a Python module constant) --
# kept here only as the single documented cross-reference for that literal,
# not consumed by any Python code path in this module anymore (see the
# corrective note above the imports: PID single-instance ownership is now
# OS-lock-arbitrated, not cmdline-inspection-based).
PROCESS_NEEDLE = "tools.liquidation_silence_scheduler"

_STOP_REQUESTED = False
# Holds the open file handle backing the OS-level exclusive lock for the
# life of this process (see acquire_single_instance_lock/
# release_single_instance_lock below). Must stay open -- closing it (or
# process exit) is what releases the lock; garbage-collecting a stray
# reference and closing it prematurely would silently give up ownership.
_LOCK_STATE: Dict[str, Any] = {"handle": None}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lock_path_for(pid_path: Path) -> Path:
    """A SEPARATE file from pid_path, used only as the OS-lock target (see
    acquire_single_instance_lock). msvcrt.locking's exclusive byte-range
    lock blocks ANY other handle from reading the locked region -- including
    a handle opened by a completely different call in the SAME process, not
    only other processes (verified empirically: a plain Path.read_text() on
    the locked file raises PermissionError while the lock is held). Locking
    pid_path itself would therefore make it unreadable by
    status_eclipse.ps1's own PID-file reader (and by read_pid_file below)
    for the entire time the wrapper is healthy and running -- exactly
    backwards from what a "show me the current PID" status file is for.
    Keeping the lock on a dedicated sibling file leaves pid_path a plain,
    always-readable text file, written via an ordinary, independent
    Path.write_text() call after the lock is acquired."""
    return pid_path.with_name(pid_path.name + ".lock")


def read_pid_file(path: Path) -> Optional[int]:
    """Informational only -- reads the last PID this module (or a prior
    instance) wrote, for status/display purposes (e.g. status_eclipse.ps1's
    equivalent PowerShell-side reader). Always safely readable, independent
    of whether the OS lock (see _lock_path_for) is currently held, and not
    itself consulted to decide single-instance ownership -- see
    acquire_single_instance_lock."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def acquire_single_instance_lock(pid_path: Path) -> bool:
    """Returns True if this process may proceed as the sole instance,
    having acquired an OS-level exclusive byte-range lock (msvcrt.locking,
    non-blocking) on a dedicated sibling lock file (see _lock_path_for) --
    held open for the remainder of this process's life -- and, only upon
    success, having written its own PID to the plain, freely-readable
    pid_path via a separate, ordinary write.

    This is genuinely race-safe: msvcrt.locking(..., LK_NBLCK, ...) is an
    OS-arbitrated primitive, not a read-then-write check -- two processes
    racing to open+lock the same file can never both succeed, because the
    OS resolves the race, this code does not. Windows automatically
    releases the lock if this process dies for any reason, including a
    hard Stop-Process -Force kill (the mechanism stop_eclipse.ps1 uses for
    every role in this repository, which gives no chance to run
    release_single_instance_lock's own cleanup below) -- which is what
    makes stale-lock recovery automatic and correct with no PID-value or
    command-line inspection needed at all: "is a live owner already
    holding this lock" and "is a leftover PID file merely stale" collapse
    into the single OS-arbitrated fact of whether this call can acquire
    the lock, which is strictly more correct than best-effort external
    process inspection (subject to PID reuse and cross-platform
    assumptions -- see the corrective note above the imports)."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for(pid_path)
    try:
        handle = open(lock_path, "a+b")
    except OSError:
        return False
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return False
    _LOCK_STATE["handle"] = handle
    try:
        pid_path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        # Extremely unlikely (parent dir just created above), but the lock
        # is the actual ownership guarantee -- a failed informational
        # write must not be treated as a failed acquire.
        pass
    return True


def release_single_instance_lock(pid_path: Path) -> None:
    """Releases the OS-level lock (closing the handle releases it) and
    removes both the lock file and the informational PID file, on a clean
    exit (graceful stop or --once completion) only. Only acts if THIS
    process actually holds the lock (`_LOCK_STATE["handle"]` is not
    None) -- a process that never acquired the lock (e.g. the
    ALREADY_RUNNING early-return path in main()) must never touch or
    delete another process's lock/PID file. A hard kill skips this
    function entirely; Windows still releases the underlying OS lock
    automatically when the process dies, which is exactly why
    acquire_single_instance_lock above needs no separate stale-PID
    recovery step."""
    handle = _LOCK_STATE.get("handle")
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass
    _LOCK_STATE["handle"] = None
    try:
        _lock_path_for(pid_path).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass


def _request_stop(signum, frame) -> None:  # noqa: ARG001 - signal handler signature
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_signal_handlers() -> None:
    """Best-effort graceful-stop support. Windows' own Stop-Process -Force
    (the mechanism every role's stop path in this repository ultimately
    guarantees) sends no catchable signal at all -- these handlers exist
    for the cases that DO deliver one (interactive Ctrl+C/SIGINT, a
    supervisor sending CTRL_BREAK_EVENT/SIGBREAK, or SIGTERM where
    available), so a stop request is honored within one poll tick (see
    interruptible_sleep) instead of the process ignoring it and needing a
    hard kill regardless."""
    signal.signal(signal.SIGINT, _request_stop)
    sigbreak = getattr(signal, "SIGBREAK", None)  # Windows-only console Ctrl+Break event
    if sigbreak is not None:
        signal.signal(sigbreak, _request_stop)
    try:
        signal.signal(signal.SIGTERM, _request_stop)
    except (AttributeError, ValueError):
        pass


def reset_stop_flag() -> None:
    """Test-only helper: _STOP_REQUESTED is process-global module state: a
    test that sets it (directly or via a delivered signal) must reset it
    afterward so later tests in the same process are not silently
    short-circuited."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


def interruptible_sleep(total_sec: float, poll_sec: float = 1.0) -> None:
    """Sleeps in small increments so a stop request (see
    install_signal_handlers) is honored within poll_sec, not only at the
    next full cadence boundary."""
    remaining = max(0.0, float(total_sec))
    while remaining > 0 and not _STOP_REQUESTED:
        step = min(poll_sec, remaining)
        time.sleep(step)
        remaining -= step


def _append_log(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    except Exception:
        # Logging must never be able to take down the loop.
        pass


def run_cycle(
    *,
    evaluation_mode: str = MODE_LIVE,
    log_path: Path = DEFAULT_LOG_PATH,
    quiet: bool = False,
) -> Dict[str, Any]:
    """One bounded, read-only detector invocation + a structured per-run
    success/failure record, appended to log_path and (unless quiet)
    printed to stdout as a single JSON line -- matching
    tools/heartbeat_watchdog.py's own per-cycle print(json.dumps(...))
    convention.

    Never raises: a run_once() failure surfaces via the accepted detector's
    own typed payload["error"] field (outcome=DETECTOR_ERROR); a truly
    unexpected exception from run_once() itself (defensive only -- that
    function is exception-safe by its own accepted design) is caught here
    too (outcome=WRAPPER_EXCEPTION), so one bad cycle can never silently
    kill the surrounding loop, and never produces a record with no
    observable outcome field.

    outcome is always derived from THIS cycle's own return value, never
    from re-reading logs/health/liquidation_silence.json afterward -- an
    old successful artifact on disk is never treated as evidence that the
    current cycle succeeded."""
    started = time.time()
    record: Dict[str, Any] = {"ts_utc": utc_now_iso(), "cycle_started_epoch": started}
    try:
        payload = run_once(evaluation_mode=evaluation_mode)
        record.update(
            {
                "outcome": "DETECTOR_ERROR" if payload.get("error") else "SUCCESS",
                "severity": payload.get("severity"),
                "status": payload.get("status"),
                "detector_error": payload.get("error"),
                "detector_runtime_sec": payload.get("detector_runtime_sec"),
            }
        )
    except Exception as exc:  # defensive: run_once() is exception-safe by design
        record.update({"outcome": "WRAPPER_EXCEPTION", "exception": f"{type(exc).__name__}: {exc}"})
    record["cycle_duration_sec"] = time.time() - started
    _append_log(log_path, record)
    if not quiet:
        print(json.dumps(record, ensure_ascii=True, separators=(",", ":")), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent-loop scheduler wrapper for the accepted, disabled-by-default "
            "liquidation-silence detector (tools/liquidation_silence_detector.py). "
            "Read-only: calls run_once(evaluation_mode=LIVE) on a cadence; no order/"
            "execution/exchange-write path. Disabled by default -- nothing launches this "
            "automatically; running it is always an explicit operator action."
        )
    )
    parser.add_argument(
        "--cadence-sec",
        type=int,
        default=DEFAULT_CADENCE_SEC,
        help=f"Seconds between detector cycles (default: {DEFAULT_CADENCE_SEC}, grounded in "
        "tools.liquidation_silence_policy.CONTROL_STREAM_FRESH_AGE_SEC).",
    )
    parser.add_argument(
        "--startup-delay-sec",
        type=int,
        default=DEFAULT_STARTUP_DELAY_SEC,
        help=f"Delay before the first cycle (default: {DEFAULT_STARTUP_DELAY_SEC}).",
    )
    parser.add_argument("--pid-path", type=str, default=str(DEFAULT_PID_PATH))
    parser.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH))
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle (skipping the startup delay) and exit. Test/rehearsal "
        "mode only -- never the production default; the while-loop above is unconditionally "
        "what runs unless this flag is explicitly passed.",
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default=MODE_LIVE,
        choices=[MODE_LIVE, MODE_HISTORICAL_REPLAY],
        help="Passed through verbatim to the wrapped detector's run_once(); default LIVE.",
    )
    args = parser.parse_args()

    pid_path = Path(args.pid_path)
    log_path = Path(args.log_path)

    if not acquire_single_instance_lock(pid_path):
        print(json.dumps({"outcome": "ALREADY_RUNNING", "pid_path": str(pid_path)}), flush=True)
        return 1

    install_signal_handlers()
    try:
        if args.once:
            run_cycle(evaluation_mode=args.evaluation_mode, log_path=log_path)
            return 0
        if args.startup_delay_sec > 0:
            interruptible_sleep(args.startup_delay_sec)
        while not _STOP_REQUESTED:
            run_cycle(evaluation_mode=args.evaluation_mode, log_path=log_path)
            if _STOP_REQUESTED:
                break
            interruptible_sleep(max(1, int(args.cadence_sec)))
        return 0
    finally:
        release_single_instance_lock(pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
