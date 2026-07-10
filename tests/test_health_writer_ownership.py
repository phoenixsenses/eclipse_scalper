"""Writer-ownership guard for logs/health/overall.json.

Proves the single-writer architecture required by the CANONICAL_OPERATIONAL_
HEALTH corrective review: tools/heartbeat_watchdog.py is the only process
permitted to write logs/health/overall.json. Every other component (paper
trader, replay, collector, bookticker) writes only its own dedicated file.
See reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.health_state as health_state
from tools import heartbeat_watchdog as hw

ROOT = Path(__file__).resolve().parents[1]


# --- structural (ast-based) source analysis helpers -------------------------
# These inspect behaviorally relevant code only: comments never reach the AST
# at all, and docstrings are explicitly excluded, so prose wording or line
# wrapping in documentation can never change an ownership verdict (the exact
# failure mode of this file's earlier substring-matching version, which broke
# on a harmless docstring line wrap in tools/replay_slice.py).


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _code_string_constants(source: str) -> list[ast.Constant]:
    """Every string literal that is actual code (not a docstring)."""
    tree = ast.parse(source)
    doc_ids = _docstring_ids(tree)
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids
    ]


def _referenced_symbols(source: str) -> set[str]:
    """Every Name/Attribute/import symbol the module's code can reach."""
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.name for a in node.names}
    return out


def _overall_json_code_literals(source: str) -> list[ast.Constant]:
    return [n for n in _code_string_constants(source) if "overall.json" in n.value.lower()]


def test_write_overall_health_no_longer_exists():
    """The generic read-merge-write helper that made a second writer
    possible has been deleted outright, not merely deprecated -- there is no
    function left for a future caller to accidentally reach for."""
    assert not hasattr(health_state, "write_overall_health")


def test_write_component_health_rejects_overall_as_component_name():
    with pytest.raises(ValueError):
        health_state.write_component_health("overall", {"status": "ok"})


def test_known_former_writers_no_longer_reference_overall_json():
    """Structural guard over the two modules that used to read-merge-write
    logs/health/overall.json. Checked via ast, never via raw text:

    - neither module's code may reference the deleted write_overall_health
      helper (as a name, attribute, or import);
    - tools/replay_slice.py may not contain 'overall.json' in ANY code
      string literal (it only ever wrote that file; it has no read either);
    - execution/health_gate.py may contain 'overall.json' in exactly one
      place: the default argument of load_overall_health, its legitimate
      READ-ONLY loader (the live-safety gate's actual input). Any other
      occurrence -- e.g. a path handed to a generic JSON writer -- fails.
    """
    # utf-8-sig: strips a BOM if present (tools/heartbeat_watchdog.py has
    # one; Python's own importer strips it too), identical to utf-8 otherwise.
    hg_source = (ROOT / "execution" / "health_gate.py").read_text(encoding="utf-8-sig")
    rs_source = (ROOT / "tools" / "replay_slice.py").read_text(encoding="utf-8-sig")

    assert "write_overall_health" not in _referenced_symbols(hg_source)
    assert "write_overall_health" not in _referenced_symbols(rs_source)

    assert _overall_json_code_literals(rs_source) == [], (
        "tools/replay_slice.py must not name overall.json in code at all"
    )

    hg_tree = ast.parse(hg_source)
    allowed_ids: set[int] = set()
    for node in ast.walk(hg_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_overall_health":
            for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
                for c in ast.walk(default):
                    if isinstance(c, ast.Constant):
                        allowed_ids.add(id(c))
    doc_ids = _docstring_ids(hg_tree)
    for node in ast.walk(hg_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in doc_ids and "overall.json" in node.value.lower():
            assert id(node) in allowed_ids, (
                "execution/health_gate.py may reference overall.json only as "
                "load_overall_health's read-path default, never anywhere else"
            )


def test_ownership_scan_ignores_docstring_prose_and_line_wrapping():
    """Regression for the exact failure that broke this file's earlier
    substring-based version: documentation wording ('owned solely' wrapped
    across a line break) must have zero effect on the ownership verdict,
    while a real code literal must still be detected."""
    wrapped_docstring_only = (
        "def f():\n"
        '    """logs/health/overall.json is owned\n'
        '    solely by tools/heartbeat_watchdog.py -- reworded, re-wrapped,\n'
        '    or deleted entirely, this prose must never matter."""\n'
        "    return 1\n"
    )
    assert _overall_json_code_literals(wrapped_docstring_only) == []

    comment_only = "# writes logs/health/overall.json (comment, not code)\nx = 1\n"
    assert _overall_json_code_literals(comment_only) == []

    real_code_literal = 'TARGET = "logs/health/overall.json"\n'
    assert len(_overall_json_code_literals(real_code_literal)) == 1


def test_collector_and_bookticker_write_only_their_own_component_file():
    """Retirement guards for the other two known-eligible writers: neither
    module's code may reference the deleted write_overall_health helper or
    name overall.json in any code string literal (comments/docstrings that
    document the ownership rule are invisible to this ast-based check)."""
    suspects = [
        ROOT / "data" / "microstructure_collector.py",
        ROOT / "data" / "bookticker_collector.py",
    ]
    for path in suspects:
        source = path.read_text(encoding="utf-8-sig")
        assert "write_overall_health" not in _referenced_symbols(source), path
        assert _overall_json_code_literals(source) == [], path


def test_heartbeat_watchdog_is_the_sole_owner_of_the_canonical_write_path():
    """Positive control: the one authorized module genuinely does construct
    the canonical overall.json path in code (atomic_write's target). If this
    ever stops matching, the ownership map itself has changed and every
    other assertion in this file needs re-review."""
    source = (ROOT / "tools" / "heartbeat_watchdog.py").read_text(encoding="utf-8-sig")
    assert len(_overall_json_code_literals(source)) >= 1


def test_research_fitness_writer_structurally_rejects_operational_outputs(tmp_path):
    """Repository-wide coverage of the CLI/generic-writer gap found by the
    independent review: tools/research_fitness_report.py's own writer (the
    one tools/collection_watchdog.py also delegates to) must reject every
    protected operational-health output before writing anything."""
    from tools.research_fitness_report import ProtectedOperationalOutputError, _atomic_write_json

    for name in ("overall.json", "watchdog.json", "WATCHDOG_STATUS.json",
                 "collector.json", "paper_trader.json"):
        with pytest.raises(ProtectedOperationalOutputError):
            _atomic_write_json(tmp_path / name, {"status": "ready"})
    assert list(tmp_path.iterdir()) == []


def test_paper_trader_write_does_not_touch_overall_json_file(tmp_path):
    """End-to-end proof (not just source scan): calling the real
    write_paper_trader_health() against an isolated root never creates
    overall.json there."""
    from execution.health_gate import GateDecision, write_paper_trader_health

    decision = GateDecision(
        allow=True, reason="", state="ok", collector_connected=True,
        collector_lag_sec=1, reconnects_last_5m=0, errors_last_5m=0,
    )
    write_paper_trader_health(decision, "", root=tmp_path)
    assert (tmp_path / "paper_trader.json").exists()
    assert not (tmp_path / "overall.json").exists()


def test_concurrent_paper_trader_and_watchdog_updates_cannot_lose_each_other(monkeypatch, tmp_path):
    """Simulates the exact race the review describes: paper-trader writer
    and heartbeat watchdog both 'running' against the same root. Because
    each owns a distinct file, interleaving their writes in either order
    can never lose either verdict -- unlike the old read-merge-write
    architecture, there is no shared mutable file for them to race on."""
    from execution.health_gate import GateDecision, write_paper_trader_health

    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok", "ts_utc": now, "last_progress_ts_utc": now,
        "required_streams_progressing": True, "transport_connected": True, "connected": True,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready", "active_mode": "paper", "fallback_to_paper": False, "ts_utc": now,
    }), encoding="utf-8")
    (root / "logs" / "collector_heartbeat.json").write_text(json.dumps({
        "connected": True, "last_message_ts_utc": now, "last_data_progress_ts_utc": now,
        "rest_fallback_enabled": True, "rest_fallback_active": False,
        "rest_last_progress_ts_utc": now, "current_backoff_seconds": 1.0, "last_error": "",
    }), encoding="utf-8")
    import sqlite3
    import time as time_mod
    db = root / "data" / "microstructure.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    now_ms = int(time_mod.time() * 1000)
    for t in ("agg_trades", "mark_prices", "liquidations"):
        conn.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
        conn.execute(f"INSERT INTO {t} (ts_ms) VALUES (?)", (now_ms - 1000,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    # 1) watchdog cycle A (paper trader not written yet)
    hw.run_once(max_age_sec=180, expect_bookticker=False, expect_detector=False, expect_runtime=True)
    overall_a = json.loads((root / "logs" / "health" / "overall.json").read_text(encoding="utf-8"))
    assert "paper_trader" not in overall_a["components"]

    # 2) paper-trader writer runs "concurrently" (writes only its own file)
    decision = GateDecision(
        allow=True, reason="", state="ok", collector_connected=True,
        collector_lag_sec=1, reconnects_last_5m=0, errors_last_5m=0,
    )
    write_paper_trader_health(decision, "", root=root / "logs" / "health")

    # 3) watchdog cycle B: must pick up paper_trader fresh, and its own
    #    write must not have been affected by paper trader's write landing
    #    in between (or after) cycle A.
    hw.run_once(max_age_sec=180, expect_bookticker=False, expect_detector=False, expect_runtime=True)
    overall_b = json.loads((root / "logs" / "health" / "overall.json").read_text(encoding="utf-8"))
    assert overall_b["components"]["paper_trader"]["status"] == "ok"
    assert overall_b["components"]["collector"]["status"] == "ok"
    assert overall_b["state"] == "ok"


def test_corrupt_optional_component_file_is_omitted_not_fatal(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "logs" / "health" / "paper_trader.json").write_text("{not valid json", encoding="utf-8")
    result = hw.build_canonical_overall(
        overall="GREEN", issues=[],
        collector_component={"status": "ok", "connected": True},
        bookticker_component={"status": "ok", "connected": True},
        native_ws_policy={"status": "GREEN", "reasons": [], "native_websocket": True,
                           "rest_fallback": False, "source_freshness": {}, "thresholds": {}},
        runtime_mode="paper",
        now_iso=hw.utc_now_z(),
        log_health=root / "logs" / "health",
    )
    assert "paper_trader" not in result["components"]
