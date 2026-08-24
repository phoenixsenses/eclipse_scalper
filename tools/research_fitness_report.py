"""One-shot research-fitness readiness report.

This is advisory/offline research-readiness information (is the data suitable
for *research use* -- contract tier, feature coverage, event_diary freshness),
not operational process health. It must never write to
logs/health/overall.json, logs/health/watchdog.json, or
reports/WATCHDOG_STATUS.json, and a limited/blocked result here must never by
itself change operational GREEN/YELLOW/RED or ok/degraded/halted state. See
an internal research report

This replaces the always-on, competing-writer responsibility that used to
live in tools/collection_watchdog.py (retired to a deprecated compatibility
wrapper that delegates here). Run on demand or via a scheduled task -- never
add this to start_eclipse.ps1 as an always-on process.

BOUNDED TABLE SCAN (fixed 2026-07-10): analyze_research_fitness() ->
tools/validate_data_research_fitness.py passes
tools.check_data_ready.RESEARCH_FITNESS_TABLE_ALLOWLIST
(mark_prices/agg_trades/liquidations) into check_db_fresh()/inspect_tables(),
so unrelated operational tables in the target database -- e.g. the unindexed
`detector_heartbeat`, which previously made a bare `SELECT MAX()` hang past
60s against the real data/microstructure.db -- are never scanned. See
an internal research report Part B and
tests/test_validate_data_research_fitness.py for the bounded-runtime proof.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.validate_data_research_fitness import analyze_research_fitness

DEFAULT_OUT_PATH = Path("logs/health/research_fitness.json")

# Operational-health outputs owned by other processes. Research fitness is
# advisory-only: it may write research_fitness.json and nothing else. These
# are matched by RESOLVED basename, case-insensitively, so relative aliases
# ("logs/health/../health/overall.json"), traversal, symlinks (resolve()
# follows them to the real target), and Windows case aliases ("OVERALL.JSON")
# are all rejected -- and rejected regardless of directory, so a test can
# never accidentally exercise a protected-write path either. Owners:
# overall.json / watchdog.json / WATCHDOG_STATUS.json -> tools/heartbeat_watchdog.py
# collector.json -> data/microstructure_collector.py
# bookticker.json -> data/bookticker_collector.py
# paper_trader.json -> execution/health_gate.py
# replay.json -> tools/replay_slice.py
# See an internal research report
PROTECTED_OPERATIONAL_OUTPUT_BASENAMES = frozenset({
    "overall.json",
    "watchdog.json",
    "watchdog_status.json",
    "collector.json",
    "bookticker.json",
    "paper_trader.json",
    "replay.json",
})


class ProtectedOperationalOutputError(ValueError):
    """Raised BEFORE any byte is written (no mkdir, no temp file) when a
    write is attempted against an operational-health file owned by another
    process. tools/collection_watchdog.py delegates to the same
    _atomic_write_json below, so the deprecated wrapper inherits this
    restriction with no bypass of its own."""


def _resolve_unprotected_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.name.lower() in PROTECTED_OPERATIONAL_OUTPUT_BASENAMES:
        raise ProtectedOperationalOutputError(
            f"research fitness may only write its own advisory report; "
            f"'{resolved}' is an operational-health output owned by another process"
        )
    return resolved


# analyze_research_fitness's own vocabulary (pass/warn/fail) is mapped to a
# distinct research-readiness vocabulary so it can never be confused with the
# operational GREEN/YELLOW/RED or ok/degraded/halted enums used elsewhere.
_STATUS_MAP = {"pass": "ready", "warn": "limited", "fail": "blocked"}
_EXIT_CODE_MAP = {"ready": 0, "limited": 1, "blocked": 2}


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    # Guard runs first -- a protected target raises before mkdir/mkstemp,
    # so a rejected call performs zero filesystem writes of any kind.
    path = _resolve_unprotected_output(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_research_fitness_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_name, str(path))
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def build_report(
    *,
    db_path: Path,
    csv_path: Path,
    symbols: List[str],
    fresh_sec: int,
    stale_after_sec: int,
    now: float | None = None,
) -> Dict[str, Any]:
    """Pure-ish evaluation wrapper (only I/O is the analyze_research_fitness
    call itself) -- degrades safely on any exception rather than raising."""
    evaluated_at = _utc_now_iso()
    try:
        payload = analyze_research_fitness(
            db_path=db_path,
            csv_path=csv_path,
            symbols=symbols,
            fresh_sec=fresh_sec,
            now=now,
        )
        raw_status = str(payload.get("status") or "fail")
        status = _STATUS_MAP.get(raw_status, "blocked")
        return {
            "status": status,
            "raw_status": raw_status,
            "contract_tier": str((payload.get("contract") or {}).get("tier") or "unknown"),
            "db_ready": bool(payload.get("db_ready")),
            "warnings": list(payload.get("warnings") or []),
            "failures": list(payload.get("failures") or []),
            "symbols": list(symbols),
            "db_path": str(db_path),
            "csv_path": str(csv_path),
            "fresh_sec": int(fresh_sec),
            "evaluated_at_utc": evaluated_at,
            "stale_after_sec": int(stale_after_sec),
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "raw_status": "error",
            "contract_tier": "unknown",
            "db_ready": False,
            "warnings": [],
            "failures": [f"evaluation_error:{type(exc).__name__}:{exc}"],
            "symbols": list(symbols),
            "db_path": str(db_path),
            "csv_path": str(csv_path),
            "fresh_sec": int(fresh_sec),
            "evaluated_at_utc": evaluated_at,
            "stale_after_sec": int(stale_after_sec),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot research-fitness readiness report (advisory, offline; never affects operational health)."
    )
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--fresh-sec", type=int, default=120)
    p.add_argument("--stale-after-sec", type=int, default=3600,
                   help="How old this report itself may get before a reader should treat it as STALE (default 3600).")
    # No --out: the production CLI writes exactly DEFAULT_OUT_PATH
    # (logs/health/research_fitness.json). Output-path injection exists only
    # through the internal API (_atomic_write_json / monkeypatching
    # DEFAULT_OUT_PATH in tests), which itself rejects every protected
    # operational-health output -- see PROTECTED_OPERATIONAL_OUTPUT_BASENAMES.
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        db_path=Path(str(args.db)),
        csv_path=Path(str(args.csv)),
        symbols=_parse_symbols(args.symbols),
        fresh_sec=int(args.fresh_sec),
        stale_after_sec=int(args.stale_after_sec),
    )
    try:
        _atomic_write_json(Path(DEFAULT_OUT_PATH), report)
    except Exception as exc:
        print(f"ERROR research_fitness_report write_failed err={exc}")
        return 3
    print(
        f"research_fitness status={report['status']} raw_status={report['raw_status']} "
        f"contract_tier={report['contract_tier']} db_ready={report['db_ready']} "
        f"warnings={len(report['warnings'])} failures={len(report['failures'])}"
    )
    return _EXIT_CODE_MAP.get(report["status"], 3)


if __name__ == "__main__":
    sys.exit(main())
