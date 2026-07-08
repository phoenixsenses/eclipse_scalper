"""Parent-side guarded runner for `ami.storage.reverify_worker`
(BATCH-STORAGE-PRODUCTION-ARCHIVE-REVERIFY-MEMORY-HARDENING-V1).

Spawns the worker as a genuinely fresh OS subprocess and polls its
resident memory from OUTSIDE that process via `psutil` -- the guard
does not depend on the worker being able to check its own memory (a
worker starved enough to matter might not get scheduler time to run its
own check), and a subprocess exiting (whether normally, or terminated by
this guard) unconditionally returns all of its memory to the OS, which
is the actual fix for the allocator-fragmentation root cause (see
`ami.storage.reverify_worker`'s module docstring).

Never writes to the source database, shard files, manifest, catalog
entry, or root catalog index -- the worker it spawns is read-only (plus
one disposable, cleaned-up temp file for the bounded restore check).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


class ReverifyMemoryGuardAbort(Exception):
    """Raised when the worker subprocess's resident memory reaches
    `rss_limit_bytes` at any poll. The subprocess is terminated
    immediately. Nothing in catalog/manifest/shard state is touched --
    the worker never writes anything except one disposable temp file,
    which is removed by the worker's own `finally` block before this
    could ever fire mid-restore-check; if terminated before that cleanup
    ran, the temp file is simply an orphaned disposable artifact beneath
    the approved temp root, never a production path."""


class ReverifyWorkerCrashed(Exception):
    """Raised when the worker subprocess exits non-zero, dies from a
    signal, or exits 0 without producing a parseable result file."""


def run_guarded_reverify(
        final_dir: str, *, rss_limit_bytes: int = 2 * 1024 ** 3,
        poll_interval_seconds: float = 1.0, batch_size: int = 1_000_000,
        do_restore_check: bool = True, restore_temp_root: str = ".runtime_temp",
        python_executable: str | None = None, timeout_seconds: float | None = None,
        result_path: str | None = None) -> dict:
    """Spawns `python -m ami.storage.reverify_worker <final_dir> ...` as a
    fresh subprocess and polls its RSS every `poll_interval_seconds` via
    `psutil.Process(pid).memory_info().rss`. If RSS ever reaches
    `rss_limit_bytes`, terminates the subprocess and raises
    `ReverifyMemoryGuardAbort` (message includes the peak RSS observed).
    If the subprocess exits non-zero, or exits 0 but its result file is
    missing/unparseable, raises `ReverifyWorkerCrashed`. Otherwise
    returns the worker's parsed result dict, with `peak_rss_bytes` and
    `rss_limit_bytes` added for closure reporting.

    `timeout_seconds`, if given, aborts (like a guard trip, but reported
    distinctly) a worker that has neither finished nor tripped the RSS
    guard within that wall-clock budget."""
    import psutil

    python_executable = python_executable or sys.executable
    own_result_path = result_path is None
    if result_path is None:
        result_path = os.path.join(
            restore_temp_root, f"reverify_result_{os.getpid()}_{int(time.time()*1000)}.json")
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    if os.path.exists(result_path):
        os.remove(result_path)

    cmd = [python_executable, "-m", "ami.storage.reverify_worker", final_dir, result_path,
           "--batch-size", str(batch_size), "--restore-temp-root", restore_temp_root]
    if not do_restore_check:
        cmd.append("--no-restore-check")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.Popen(cmd, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_rss = 0
    start = time.monotonic()
    try:
        ps_proc = psutil.Process(proc.pid)
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            try:
                rss = ps_proc.memory_info().rss
                peak_rss = max(peak_rss, rss)
            except psutil.NoSuchProcess:
                break
            if rss >= rss_limit_bytes:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                raise ReverifyMemoryGuardAbort(
                    f"reverify worker resident memory {rss} bytes reached guard limit "
                    f"{rss_limit_bytes} bytes (peak observed {peak_rss} bytes) -- terminated; "
                    f"catalog/manifest/shard state untouched (worker is read-only)")
            if timeout_seconds is not None and (time.monotonic() - start) > timeout_seconds:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                raise ReverifyMemoryGuardAbort(
                    f"reverify worker exceeded timeout_seconds={timeout_seconds} "
                    f"(peak RSS observed {peak_rss} bytes, under the {rss_limit_bytes}-byte guard) "
                    f"-- terminated; catalog/manifest/shard state untouched")
            time.sleep(poll_interval_seconds)

        stdout, stderr = proc.communicate(timeout=30)
        if proc.returncode != 0:
            raise ReverifyWorkerCrashed(
                f"reverify worker exited with code {proc.returncode}; stderr: "
                f"{stderr.decode(errors='replace')[-4000:]}")
        if not os.path.exists(result_path):
            raise ReverifyWorkerCrashed(
                f"reverify worker exited 0 but produced no result file at {result_path!r}; "
                f"stderr: {stderr.decode(errors='replace')[-4000:]}")
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        if result.get("crashed"):
            raise ReverifyWorkerCrashed(
                f"reverify worker reported an internal error: {result.get('error')}")

        result["peak_rss_bytes"] = peak_rss
        result["rss_limit_bytes"] = rss_limit_bytes
        return result
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if own_result_path and os.path.exists(result_path):
            os.remove(result_path)
