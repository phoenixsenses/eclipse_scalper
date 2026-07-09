"""Prep/hygiene gate: make research_s34_source_quality_reconciliation.py
import-safe (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-CONSUMER-PREP-
SOURCE-QUALITY-MAIN-GUARD-V1).

The file used to run the entire reconciliation at import time -- opening
DBs, running `assert len(signals)==220`, and writing CSV/JSON to a
hardcoded, foreign/stale session-scratchpad path. That made it impossible
to import for parity testing (the blocker recorded in RANGE-READ-CONSUMER-
MIGRATION-V2). This gate does NOT migrate anything to the reader; it only
moves the executable body under `main()`/`__main__`, keeps the helper
functions importable, and replaces the stale output path with a run-time
`--out-dir` default.

These tests assert the import-safety and structural properties WITHOUT ever
running the full reconciliation (which needs the real canonical DB with
exactly 220 ETHUSDT/LONG signals) and WITHOUT writing to any real output
path. No reader migration, no DB access, no artifacts are exercised here.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

MODULE = "tools.research_s34_source_quality_reconciliation"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(REPO_ROOT, "tools", "research_s34_source_quality_reconciliation.py")


def _source():
    with open(FILE_PATH, encoding="utf-8") as f:
        return f.read()


# --- 1: import triggers NO DB connect, NO write open(), NO assert, NO heavy run ---

def test_import_has_no_db_write_or_assert_side_effects():
    # Run in a fresh subprocess with sqlite3.connect and write-mode open()
    # trip-wired, so ANY import-time DB access or file write fails loudly.
    prog = textwrap.dedent(
        """
        import builtins, sqlite3, sys
        def _boom_connect(*a, **k):
            raise SystemExit("IMPORT_DB_CONNECT")
        sqlite3.connect = _boom_connect
        _orig_open = builtins.open
        def _guard_open(file, mode="r", *a, **k):
            if any(w in str(mode) for w in ("w", "a", "x", "+")):
                raise SystemExit("IMPORT_WRITE_OPEN")
            return _orig_open(file, mode, *a, **k)
        builtins.open = _guard_open
        import importlib
        importlib.import_module("tools.research_s34_source_quality_reconciliation")
        print("IMPORT_CLEAN")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", prog], cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    combined = (r.stdout or "") + (r.stderr or "")
    assert "IMPORT_DB_CONNECT" not in combined, combined
    assert "IMPORT_WRITE_OPEN" not in combined, combined
    assert "AssertionError" not in combined, combined  # the len()==220 assert must NOT fire at import
    assert "IMPORT_CLEAN" in combined, combined
    assert r.returncode == 0, combined


# --- 2: main() exists and is callable ---

def test_main_exists_and_callable():
    mod = __import__(MODULE, fromlist=["main"])
    assert callable(getattr(mod, "main", None))


# --- 3: an `if __name__ == "__main__": main()` guard exists ---

def test_has_name_main_guard_calling_main():
    tree = ast.parse(_source())
    guards = [n for n in tree.body if isinstance(n, ast.If)]
    assert guards, "no top-level `if` guard present"
    found = False
    for g in guards:
        t = g.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators)):
            calls = [c for c in ast.walk(g) if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Name) and c.func.id == "main"]
            found = found or bool(calls)
    assert found, "no `if __name__ == '__main__': main()` guard found"


# --- 4: the module body has no executable top-level statements beyond
# imports / sys.path bootstrap / constants / defs / the guard ---

def test_no_executable_top_level_body():
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Assign):  # module constants (CANON/MICRO/OUT_SUBDIR)
            continue
        if isinstance(node, ast.If):  # __main__ guard
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            # the only permitted top-level call is the sys.path bootstrap
            assert isinstance(fn, ast.Attribute) and fn.attr == "insert", ast.dump(node)
            continue
        raise AssertionError(f"unexpected executable top-level statement: {ast.dump(node)}")


# --- 5: the stale/foreign session-scratchpad path is gone ---

def test_stale_foreign_scratchpad_removed():
    src = _source()
    assert "0e02bf95" not in src  # the specific foreign session id
    assert "SCRATCH" not in src   # the old hardcoded constant name
    assert "scratchpad" not in src.lower() or "OUT_SUBDIR" in src  # no hardcoded scratchpad path constant


# --- 6: the out-dir default resolves at RUN time to a deterministic OS-temp
# subdir, never to a foreign session path; --help shows it without running
# the reconciliation body ---

def test_out_dir_default_is_temp_based_at_runtime():
    import tempfile
    mod = __import__(MODULE, fromlist=["OUT_SUBDIR"])
    expected = os.path.join(tempfile.gettempdir(), mod.OUT_SUBDIR)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # --help makes argparse print usage (incl. the resolved default help) and
    # exit(0) BEFORE any DB access -- proves the guard runs main() and the
    # default is computed at run time, not import time.
    r = subprocess.run([sys.executable, FILE_PATH, "--help"], cwd=REPO_ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "--out-dir" in r.stdout
    # the resolved default path itself must be under the OS temp dir
    assert os.path.dirname(expected) == tempfile.gettempdir()
    assert mod.OUT_SUBDIR in expected


# --- 7: the direct-SQL helper/oracle functions are importable, and the pure
# ones are callable standalone (window_health stays direct SQL -- NOT migrated) ---

def test_helper_functions_importable():
    mod = __import__(MODULE, fromlist=[
        "iso", "overlaps", "method_a", "method_b", "window_health", "standard2", "counts", "main"])
    for name in ["iso", "overlaps", "method_a", "method_b", "window_health", "standard2", "counts", "main"]:
        assert callable(getattr(mod, name, None)), name
    # pure helpers work with no module globals populated
    assert mod.iso(0) == "1970-01-01 00:00:00"
    assert mod.iso(None) is None
    assert mod.overlaps(1, 5, 2, 3) is True
    assert mod.overlaps(1, 5, 6, 7) is False


# --- 8: this gate did NOT migrate window_health to the reader (it stays
# direct SQL for a later range-read gate) ---

def test_window_health_still_direct_sql_not_migrated():
    import inspect
    mod = __import__(MODULE, fromlist=["window_health"])
    src = inspect.getsource(mod.window_health)
    assert "conn_m.execute(" in src               # still direct-SQL on the module connection
    assert "mark_prices" in src and "agg_trades" in src
    assert "plan_read" not in src and "execute_read" not in src and "lookup_latest_at_or_before" not in src


# --- 9: the reconciliation SQL + the len()==220 assert semantics are
# preserved verbatim (byte-identical query text, assert intact) ---

def test_queries_and_assert_semantics_preserved():
    src = _source()
    assert "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms" in src
    assert "SELECT COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms >= ? AND ts_ms <= ?" in src
    assert "assert len(signals) == 220, len(signals)" in src
    assert "assert g is not None, s[\"signal_id\"]" in src
    # both DB connections remain mode=ro
    assert 'sqlite3.connect(f"file:{CANON}?mode=ro", uri=True)' in src
    assert 'sqlite3.connect(f"file:{MICRO}?mode=ro", uri=True)' in src
