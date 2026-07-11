"""One-shot liquidation-silence detector: bounded read-only snapshot
acquisition + tools.liquidation_silence_policy evaluation + component-only
output write.

Writes ONLY its own dedicated component file (default
logs/health/liquidation_silence.json, via
tools.health_state.write_component_health) -- it never writes
logs/health/overall.json or any other component's file. See
tools/liquidation_silence_policy.py's module docstring and
reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md for full
background.

Disabled by default: this module is not imported by
tools/heartbeat_watchdog.py, not launched by start_eclipse.ps1, and has no
scheduled/looping mode -- running it is always an explicit one-shot CLI
invocation or an explicit test/rehearsal call.

All data-layer reads open data/microstructure.db with `mode=ro` and use
only indexed, LIMIT-bounded queries (matching the established
`ORDER BY ts_ms DESC LIMIT 1` pattern already used elsewhere in this
repository for the same tables) -- never a full-table scan and never
`MAX(ts_ms)` (see tools/native_ws_health_policy.py's own note on why that
form defeats the rowid/btree fast path on this database).

--- v2 corrective changes (2026-07-11, independent-review corrective pass) --
- EVALUATION MODE: evaluate_once/run_once take an explicit evaluation_mode
  ("LIVE" | "HISTORICAL_REPLAY"). LIVE requires now_ts near wall-clock and
  may read the live default overall/collector/pid files. HISTORICAL_REPLAY
  forbids reading the live default health/pid files (they are present-day
  snapshots with no historical version) -- it requires explicit historical
  evidence paths (or explicit disabled/unknown upstream). This structurally
  prevents present-day health evidence silently contaminating a historical
  replay (review MEDIUM 3), rather than relying on a documentation warning.
- STRUCTURED ERROR PROPAGATION: DB/read failures (table missing, schema
  mismatch, DB locked, read error, permission failure) surface as
  deterministic error codes in the component payload's "error" field
  instead of being silently swallowed to None (review LOW). The error field
  stays null for a normal healthy evaluation.
- TEMPORAL VALIDATION: the collector component timestamp is validated
  against now_ts; a component timestamp in the future beyond tolerance is
  rejected as a temporal mismatch rather than counted as fresh evidence.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.health_state import write_component_health
from tools.liquidation_silence_policy import (
    FUTURE_TIMESTAMP_TOLERANCE_SEC,
    POLICY_FINGERPRINT,
    POLICY_VERSION,
    SEVERITY_UNKNOWN,
    STATUS_UNKNOWN,
    evaluate_liquidation_silence,
    normalize_tracked_symbols,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "microstructure.db"
DEFAULT_HEALTH_ROOT = ROOT / "logs" / "health"
DEFAULT_PID_META_PATH = ROOT / "logs" / "pids" / "collector_supervisor.json"
DEFAULT_OVERALL_PATH = ROOT / "logs" / "health" / "overall.json"
DEFAULT_COLLECTOR_COMPONENT_PATH = ROOT / "logs" / "health" / "collector.json"

COMPONENT_NAME = "liquidation_silence"
SCHEMA_VERSION = "liquidation_silence_component_v2"

# Evaluation modes.
MODE_LIVE = "LIVE"
MODE_HISTORICAL_REPLAY = "HISTORICAL_REPLAY"

# LIVE mode: now_ts must be within this tolerance of real wall-clock time,
# otherwise a caller has passed a historical/synthetic instant while still
# pointed (potentially) at the live default health files -- rejected as a
# contamination risk. Generous enough for a slow one-shot run to complete.
LIVE_MODE_WALLCLOCK_TOLERANCE_SEC = 900.0  # 15 min

# Deterministic error codes (see module docstring; also referenced by the
# governance report).
ERR_DB_TABLE_MISSING = "DB_TABLE_MISSING"
ERR_DB_SCHEMA_MISMATCH = "DB_SCHEMA_MISMATCH"
ERR_DB_LOCKED = "DB_LOCKED"
ERR_DB_READ_ERROR = "DB_READ_ERROR"
ERR_DB_CONNECT_ERROR = "DB_CONNECT_ERROR"
ERR_DB_PERMISSION = "DB_PERMISSION_DENIED"
ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH = "CONTROL_COMPONENT_TEMPORAL_MISMATCH"
ERR_HISTORICAL_EVIDENCE_REQUIRED = "HISTORICAL_EVIDENCE_REQUIRED"
ERR_LIVE_MODE_TIMESTAMP_MISMATCH = "LIVE_MODE_TIMESTAMP_MISMATCH"
ERR_INVALID_EVALUATION_MODE = "INVALID_EVALUATION_MODE"

# Used only if the canonical runtime config (collector_supervisor's own PID
# metadata, written by start_eclipse.ps1 at launch) cannot be read at all --
# never silently substituted for a successfully-discovered universe. See
# discover_tracked_symbols(); the output payload always records which path
# was taken via "symbol_source".
_FALLBACK_TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

_CONTROL_TABLES = ("mark_prices", "agg_trades")


class DetectorReadError(Exception):
    """Typed internal read failure carrying a deterministic error code, so
    evaluate_once can surface it centrally in the payload's error field
    instead of a helper silently degrading to None."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _classify_sqlite_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "no such table" in text:
        return ERR_DB_TABLE_MISSING
    if "no such column" in text or "has no column" in text or "schema" in text:
        return ERR_DB_SCHEMA_MISMATCH
    if "locked" in text or "busy" in text:
        return ERR_DB_LOCKED
    if "permission" in text or "denied" in text or "readonly" in text or "read-only" in text or "unable to open" in text:
        return ERR_DB_PERMISSION
    return ERR_DB_READ_ERROR


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        # utf-8-sig: some PID-metadata files (e.g. logs/pids/collector_supervisor.json,
        # written by start_eclipse.ps1) carry a UTF-8 BOM, which plain utf-8
        # json.loads rejects outright. utf-8-sig strips a BOM if present and is
        # otherwise identical to utf-8 for BOM-less files.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def discover_tracked_symbols(pid_meta_path: Path = DEFAULT_PID_META_PATH) -> Dict[str, Any]:
    """Reads the canonical runtime symbol universe from
    logs/pids/collector_supervisor.json's own "symbols" field (written by
    start_eclipse.ps1 at process launch, the same source both
    scripts/collector_supervisor.py and data/bookticker_collector.py are
    started with). Never hardcodes BTCUSDT/ETHUSDT/SOLUSDT except as a
    last-resort fallback when that file is entirely unreadable -- and the
    fallback is always reported explicitly via "source". Symbols are
    normalized/de-duplicated (see normalize_tracked_symbols)."""
    payload = _read_json(pid_meta_path)
    if isinstance(payload, dict):
        raw = payload.get("symbols")
        if isinstance(raw, str) and raw.strip():
            norm = normalize_tracked_symbols([s for s in raw.split(",")])
            if norm["symbols"]:
                return {"symbols": norm["symbols"], "source": "canonical_runtime_config", "path": str(pid_meta_path)}
    norm = normalize_tracked_symbols(list(_FALLBACK_TRACKED_SYMBOLS))
    return {"symbols": norm["symbols"], "source": "fallback_default", "path": str(pid_meta_path)}


def _open_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)


def read_last_liquidation_ts(
    db_path: Path, symbols: List[str], now_ts: Optional[float] = None
) -> Tuple[Dict[str, Optional[int]], List[Dict[str, str]]]:
    """Bounded per-symbol read: one indexed `ORDER BY ts_ms DESC LIMIT 1`
    query per tracked symbol (uses idx_liq_symbol_ts), never a scan of the
    whole liquidations table regardless of its size.

    Returns (per_symbol_last_ts_ms, errors). errors is a list of
    {code, message} dicts (empty on success); a connect/lock/permission
    failure or per-symbol DB error is recorded there deterministically
    instead of silently degrading to None with no explanation.

    now_ts, when given, adds a `ts_ms <= now_ts*1000` upper bound -- required
    for point-in-time historical replay against real production data (see
    reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md's replay
    section): without it this query returns the row nearest the real wall
    clock regardless of what past instant is being evaluated, which is
    exactly the lookahead this detector (and this repository generally,
    per CLAUDE.md) must never introduce. A live one-shot run passes its own
    real now_ts, so this bound is a no-op there -- the row it would find is
    already <= now."""
    out: Dict[str, Optional[int]] = {s: None for s in symbols}
    errors: List[Dict[str, str]] = []
    upper_ms = int(now_ts * 1000) if now_ts is not None else None
    if not Path(db_path).exists():
        return out, [{"code": ERR_DB_CONNECT_ERROR, "message": f"database file not present: {db_path}"}]
    try:
        conn = _open_ro(db_path)
    except sqlite3.Error as exc:
        return out, [{"code": _classify_sqlite_error(exc), "message": f"liquidations open failed: {exc}"}]
    except OSError as exc:
        return out, [{"code": ERR_DB_CONNECT_ERROR, "message": f"liquidations open failed: {exc}"}]
    try:
        cur = conn.cursor()
        for sym in symbols:
            try:
                if upper_ms is not None:
                    cur.execute(
                        "SELECT ts_ms FROM liquidations WHERE symbol = ? AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1",
                        (sym, upper_ms),
                    )
                else:
                    cur.execute(
                        "SELECT ts_ms FROM liquidations WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1", (sym,)
                    )
                row = cur.fetchone()
                out[sym] = int(row[0]) if row and row[0] is not None else None
            except sqlite3.Error as exc:
                code = _classify_sqlite_error(exc)
                errors.append({"code": code, "message": f"liquidations[{sym}] read failed: {exc}"})
                out[sym] = None
                if code in (ERR_DB_TABLE_MISSING, ERR_DB_SCHEMA_MISMATCH):
                    # A structural failure affects every symbol identically;
                    # no value in re-hitting it per-symbol.
                    break
    finally:
        conn.close()
    return out, errors


def read_control_freshness(
    db_path: Path, now_ts: float
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, str]]]:
    """Bounded read of the two REST-covered control tables this detector
    cross-validates liquidation silence against. Same indexed
    `ORDER BY ts_ms DESC LIMIT 1` pattern (idx_mark_ts / idx_trade_ts),
    with the same `ts_ms <= now_ts*1000` no-lookahead bound as
    read_last_liquidation_ts (see that function's docstring).

    Returns (per_table_age_sec, errors). Ages are SIGNED (now_ts -
    row_ts) -- not clamped to >= 0 here -- so the pure policy can detect a
    future control timestamp; a bounded `ts_ms <= now_ts*1000` query makes a
    future row structurally impossible in normal operation, but the signed
    form keeps the contract honest."""
    out: Dict[str, Optional[float]] = {t: None for t in _CONTROL_TABLES}
    errors: List[Dict[str, str]] = []
    upper_ms = int(now_ts * 1000)
    if not Path(db_path).exists():
        return out, [{"code": ERR_DB_CONNECT_ERROR, "message": f"database file not present: {db_path}"}]
    try:
        conn = _open_ro(db_path)
    except sqlite3.Error as exc:
        return out, [{"code": _classify_sqlite_error(exc), "message": f"control open failed: {exc}"}]
    except OSError as exc:
        return out, [{"code": ERR_DB_CONNECT_ERROR, "message": f"control open failed: {exc}"}]
    try:
        cur = conn.cursor()
        for table in _CONTROL_TABLES:
            try:
                cur.execute(f"SELECT ts_ms FROM {table} WHERE ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1", (upper_ms,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    out[table] = now_ts - int(row[0]) / 1000.0
            except sqlite3.Error as exc:
                errors.append({"code": _classify_sqlite_error(exc), "message": f"{table} read failed: {exc}"})
                out[table] = None
    finally:
        conn.close()
    return out, errors


def read_native_ws_status(overall_path: Path = DEFAULT_OVERALL_PATH) -> Optional[str]:
    """Read-only consumption of the existing canonical overall.json's
    additive native_ws_status field. Never writes to this path -- see
    module docstring."""
    payload = _read_json(overall_path)
    if not payload:
        return None
    status = payload.get("native_ws_status")
    return status if isinstance(status, str) else None


def read_collector_component_alive(
    component_path: Path = DEFAULT_COLLECTOR_COMPONENT_PATH,
    now_ts: Optional[float] = None,
    max_age_sec: float = 180.0,
) -> Tuple[Optional[bool], Optional[str]]:
    """Cheap proxy for collector aliveness from its own already-written
    component file (no new process-enumeration subprocess is spawned here
    -- that responsibility belongs to tools/heartbeat_watchdog.py).

    Returns (alive, error_code). alive is None (unknown, not a verdict) when
    the file is missing/corrupt, the timestamp cannot be parsed, or the
    timestamp is temporally inconsistent with now_ts (in the future beyond
    tolerance -- a present-day file leaking into a historical evaluation);
    the temporal-mismatch case also returns
    ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH so it is visible, not silently
    absorbed. Only returns (False, None) when the file is present, temporally
    consistent, but clearly stale."""
    payload = _read_json(component_path)
    if not payload:
        return None, None
    ts_raw = payload.get("ts_utc")
    if not ts_raw:
        return None, None
    try:
        import datetime as _dt

        text = str(ts_raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        ref = now_ts if now_ts is not None else time.time()
        age = ref - dt.timestamp()
    except Exception:
        return None, None
    if age < -FUTURE_TIMESTAMP_TOLERANCE_SEC:
        # Component timestamp is in the future relative to the evaluation
        # instant beyond benign clock skew -- e.g. a present-day collector.json
        # read during a historical replay. Refuse to treat as fresh evidence.
        return None, ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH
    return (age <= max_age_sec), None


def _uses_live_default_health_paths(
    pid_meta_path: Path, overall_path: Path, collector_component_path: Path
) -> List[str]:
    """Return the names of any present-day live default health/pid files
    among the given paths. Used by HISTORICAL_REPLAY mode to refuse silently
    reading live evidence into a historical evaluation."""
    offenders = []
    if Path(pid_meta_path) == DEFAULT_PID_META_PATH:
        offenders.append("pid_meta_path")
    if Path(overall_path) == DEFAULT_OVERALL_PATH:
        offenders.append("overall_path")
    if Path(collector_component_path) == DEFAULT_COLLECTOR_COMPONENT_PATH:
        offenders.append("collector_component_path")
    return offenders


def _unknown_payload(
    *,
    now_ts: float,
    symbols: List[str],
    symbol_source: str,
    evaluation_mode: str,
    reason_codes: List[str],
    errors: List[Dict[str, str]],
    db_path: Path,
    pid_meta_path: Path,
    overall_path: Path,
    collector_component_path: Path,
    runtime_sec: float,
) -> Dict[str, Any]:
    """Fail-visible UNKNOWN payload used when a mode/guard rejects the
    evaluation before (or instead of) reading live evidence."""
    return {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT_NAME,
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "evaluation_mode": evaluation_mode,
        "evaluated_at_utc": _iso(now_ts),
        "evaluated_at_ms": int(now_ts * 1000),
        "ts_utc": _iso(now_ts),
        "status": STATUS_UNKNOWN,
        "severity": SEVERITY_UNKNOWN,
        "reason_codes": sorted(set(reason_codes)),
        "tracked_symbols": symbols,
        "symbol_source": symbol_source,
        "last_liquidation_ts_ms": {s: None for s in symbols},
        "per_symbol_silence_age_sec": {s: None for s in symbols},
        "all_symbol_silence_age_sec": None,
        "tracked_symbol_count": len(symbols),
        "known_symbol_count": 0,
        "missing_symbols": list(symbols),
        "complete_symbol_evidence": False,
        "control_stream_ages_sec": {t: None for t in _CONTROL_TABLES},
        "mark_prices_age_sec": None,
        "agg_trades_age_sec": None,
        "native_ws_status": None,
        "collector_component_alive": None,
        "thresholds": None,
        "data_sources": {
            "db_path": str(db_path),
            "pid_meta_path": str(pid_meta_path),
            "overall_path": str(overall_path),
            "collector_component_path": str(collector_component_path),
        },
        "detector_runtime_sec": runtime_sec,
        "error": errors or None,
    }


def evaluate_once(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    pid_meta_path: Path = DEFAULT_PID_META_PATH,
    overall_path: Path = DEFAULT_OVERALL_PATH,
    collector_component_path: Path = DEFAULT_COLLECTOR_COMPONENT_PATH,
    now_ts: Optional[float] = None,
    evaluation_mode: str = MODE_LIVE,
) -> Dict[str, Any]:
    """Bounded read-only snapshot + pure policy evaluation. Does not write
    anything.

    evaluation_mode:
      - MODE_LIVE (default): now_ts must be near real wall-clock time
        (within LIVE_MODE_WALLCLOCK_TOLERANCE_SEC); the live default
        overall/collector/pid files may be read. Passing a historical
        now_ts in LIVE mode is rejected (fail-visible UNKNOWN) so a
        historical instant can never be silently graded against present-day
        live evidence.
      - MODE_HISTORICAL_REPLAY: now_ts is an arbitrary past instant; the
        live default overall/collector/pid files may NOT be read (they are
        present-day snapshots with no historical version) -- explicit
        historical evidence paths (or nonexistent/disabled paths) are
        required. Component timestamps are validated against now_ts.

    now_ts is always explicit for replay (never a bare time.time() call
    buried in the policy layer)."""
    start = time.time()
    wall_now = time.time()
    now_ts = now_ts if now_ts is not None else wall_now

    symbol_info = discover_tracked_symbols(pid_meta_path)
    symbols = symbol_info["symbols"]

    # --- evaluation-mode guards (contamination safety) -----------------------
    if evaluation_mode not in (MODE_LIVE, MODE_HISTORICAL_REPLAY):
        return _unknown_payload(
            now_ts=now_ts, symbols=symbols, symbol_source=symbol_info["source"],
            evaluation_mode=str(evaluation_mode),
            reason_codes=[ERR_INVALID_EVALUATION_MODE],
            errors=[{"code": ERR_INVALID_EVALUATION_MODE, "message": f"unknown evaluation_mode {evaluation_mode!r}"}],
            db_path=db_path, pid_meta_path=pid_meta_path, overall_path=overall_path,
            collector_component_path=collector_component_path, runtime_sec=time.time() - start,
        )

    if evaluation_mode == MODE_LIVE:
        if abs(now_ts - wall_now) > LIVE_MODE_WALLCLOCK_TOLERANCE_SEC:
            return _unknown_payload(
                now_ts=now_ts, symbols=symbols, symbol_source=symbol_info["source"],
                evaluation_mode=evaluation_mode,
                reason_codes=[ERR_LIVE_MODE_TIMESTAMP_MISMATCH],
                errors=[{
                    "code": ERR_LIVE_MODE_TIMESTAMP_MISMATCH,
                    "message": (
                        f"LIVE mode requires now_ts within {LIVE_MODE_WALLCLOCK_TOLERANCE_SEC:.0f}s of wall-clock; "
                        f"delta={now_ts - wall_now:.1f}s. Use evaluation_mode=HISTORICAL_REPLAY with explicit "
                        "historical evidence paths for point-in-time replay."
                    ),
                }],
                db_path=db_path, pid_meta_path=pid_meta_path, overall_path=overall_path,
                collector_component_path=collector_component_path, runtime_sec=time.time() - start,
            )
    else:  # MODE_HISTORICAL_REPLAY
        offenders = _uses_live_default_health_paths(pid_meta_path, overall_path, collector_component_path)
        if offenders:
            return _unknown_payload(
                now_ts=now_ts, symbols=symbols, symbol_source=symbol_info["source"],
                evaluation_mode=evaluation_mode,
                reason_codes=[ERR_HISTORICAL_EVIDENCE_REQUIRED],
                errors=[{
                    "code": ERR_HISTORICAL_EVIDENCE_REQUIRED,
                    "message": (
                        "HISTORICAL_REPLAY may not read live default health/pid files "
                        f"({', '.join(offenders)}); pass explicit historical evidence paths "
                        "(or nonexistent/disabled paths) so present-day evidence cannot contaminate the replay."
                    ),
                }],
                db_path=db_path, pid_meta_path=pid_meta_path, overall_path=overall_path,
                collector_component_path=collector_component_path, runtime_sec=time.time() - start,
            )

    errors: List[Dict[str, str]] = []
    last_liq_ts: Dict[str, Optional[int]] = {s: None for s in symbols}
    control_freshness: Dict[str, Optional[float]] = {t: None for t in _CONTROL_TABLES}
    last_liq_ts, liq_errors = read_last_liquidation_ts(db_path, symbols, now_ts=now_ts)
    control_freshness, control_errors = read_control_freshness(db_path, now_ts)
    errors.extend(liq_errors)
    errors.extend(control_errors)

    native_ws_status = read_native_ws_status(overall_path)
    collector_alive, collector_err = read_collector_component_alive(collector_component_path, now_ts=now_ts)
    if collector_err:
        errors.append({"code": collector_err, "message": "collector component timestamp temporally inconsistent with now_ts"})

    decision = evaluate_liquidation_silence(
        now_ts=now_ts,
        tracked_symbols=symbols,
        last_liquidation_ts_ms=last_liq_ts,
        mark_prices_age_sec=control_freshness.get("mark_prices"),
        agg_trades_age_sec=control_freshness.get("agg_trades"),
        native_ws_status=native_ws_status,
        collector_process_alive=collector_alive,
    )

    runtime_sec = time.time() - start
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT_NAME,
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "evaluation_mode": evaluation_mode,
        "evaluated_at_utc": _iso(now_ts),
        "evaluated_at_ms": int(now_ts * 1000),
        # Explicit "ts_utc" (not just evaluated_at_utc) so this component
        # file is gradeable by the same freshness convention every other
        # logs/health/*.json component already uses (see
        # tools/heartbeat_watchdog.py::component_fresh /
        # OPTIONAL_COMPONENT_FILES) -- setting it ourselves means
        # tools.health_state.write_component_health's own ts_utc
        # setdefault() is a deliberate no-op, not a second, slightly-later
        # timestamp.
        "ts_utc": _iso(now_ts),
        "status": decision["status"],
        "severity": decision["severity"],
        "reason_codes": decision["reasons"],
        "tracked_symbols": symbols,
        "symbol_source": symbol_info["source"],
        "last_liquidation_ts_ms": last_liq_ts,
        "per_symbol_silence_age_sec": decision["per_symbol_silence_age_sec"],
        "all_symbol_silence_age_sec": decision["all_symbol_silence_age_sec"],
        "tracked_symbol_count": decision["tracked_symbol_count"],
        "known_symbol_count": decision["known_symbol_count"],
        "missing_symbols": decision["missing_symbols"],
        "complete_symbol_evidence": decision["complete_symbol_evidence"],
        "control_stream_ages_sec": control_freshness,
        "mark_prices_age_sec": decision["mark_prices_age_sec"],
        "agg_trades_age_sec": decision["agg_trades_age_sec"],
        "native_ws_status": native_ws_status,
        "collector_component_alive": collector_alive,
        "thresholds": decision["thresholds"],
        "data_sources": {
            "db_path": str(db_path),
            "pid_meta_path": str(pid_meta_path),
            "overall_path": str(overall_path),
            "collector_component_path": str(collector_component_path),
        },
        "detector_runtime_sec": runtime_sec,
        "error": errors or None,
    }
    return payload


def _iso(ts: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run_once(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    health_root: Path = DEFAULT_HEALTH_ROOT,
    pid_meta_path: Path = DEFAULT_PID_META_PATH,
    overall_path: Path = DEFAULT_OVERALL_PATH,
    collector_component_path: Path = DEFAULT_COLLECTOR_COMPONENT_PATH,
    now_ts: Optional[float] = None,
    evaluation_mode: str = MODE_LIVE,
    dry_run: bool = False,
) -> Dict[str, Any]:
    payload = evaluate_once(
        db_path=db_path,
        pid_meta_path=pid_meta_path,
        overall_path=overall_path,
        collector_component_path=collector_component_path,
        now_ts=now_ts,
        evaluation_mode=evaluation_mode,
    )
    if not dry_run:
        write_component_health(COMPONENT_NAME, payload, root=health_root)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Eclipse liquidation-silence detector (one-shot, disabled-by-default)")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--health-root", type=str, default=str(DEFAULT_HEALTH_ROOT))
    parser.add_argument("--pid-meta-path", type=str, default=str(DEFAULT_PID_META_PATH))
    parser.add_argument("--overall-path", type=str, default=str(DEFAULT_OVERALL_PATH))
    parser.add_argument("--collector-component-path", type=str, default=str(DEFAULT_COLLECTOR_COMPONENT_PATH))
    parser.add_argument("--mode", type=str, default=MODE_LIVE, choices=[MODE_LIVE, MODE_HISTORICAL_REPLAY], help="Evaluation mode.")
    parser.add_argument("--now-ts", type=float, default=None, help="Epoch seconds to evaluate as-of (default: real now). Used for replay/rehearsal (requires --mode HISTORICAL_REPLAY with explicit evidence paths).")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and print but do not write the component file.")
    args = parser.parse_args()

    payload = run_once(
        db_path=Path(args.db_path),
        health_root=Path(args.health_root),
        pid_meta_path=Path(args.pid_meta_path),
        overall_path=Path(args.overall_path),
        collector_component_path=Path(args.collector_component_path),
        now_ts=args.now_ts,
        evaluation_mode=args.mode,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
