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
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.health_state import write_component_health
from tools.liquidation_silence_policy import (
    POLICY_FINGERPRINT,
    POLICY_VERSION,
    evaluate_liquidation_silence,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "microstructure.db"
DEFAULT_HEALTH_ROOT = ROOT / "logs" / "health"
DEFAULT_PID_META_PATH = ROOT / "logs" / "pids" / "collector_supervisor.json"
DEFAULT_OVERALL_PATH = ROOT / "logs" / "health" / "overall.json"
DEFAULT_COLLECTOR_COMPONENT_PATH = ROOT / "logs" / "health" / "collector.json"

COMPONENT_NAME = "liquidation_silence"
SCHEMA_VERSION = "liquidation_silence_component_v1"

# Used only if the canonical runtime config (collector_supervisor's own PID
# metadata, written by start_eclipse.ps1 at launch) cannot be read at all --
# never silently substituted for a successfully-discovered universe. See
# discover_tracked_symbols(); the output payload always records which path
# was taken via "symbol_source".
_FALLBACK_TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

_CONTROL_TABLES = ("mark_prices", "agg_trades")


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
    fallback is always reported explicitly via "source"."""
    payload = _read_json(pid_meta_path)
    if isinstance(payload, dict):
        raw = payload.get("symbols")
        if isinstance(raw, str) and raw.strip():
            symbols = [s.strip() for s in raw.split(",") if s.strip()]
            if symbols:
                return {"symbols": symbols, "source": "canonical_runtime_config", "path": str(pid_meta_path)}
    return {"symbols": list(_FALLBACK_TRACKED_SYMBOLS), "source": "fallback_default", "path": str(pid_meta_path)}


def _open_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)


def read_last_liquidation_ts(db_path: Path, symbols: List[str], now_ts: Optional[float] = None) -> Dict[str, Optional[int]]:
    """Bounded per-symbol read: one indexed `ORDER BY ts_ms DESC LIMIT 1`
    query per tracked symbol (uses idx_liq_symbol_ts), never a scan of the
    whole liquidations table regardless of its size.

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
    upper_ms = int(now_ts * 1000) if now_ts is not None else None
    try:
        conn = _open_ro(db_path)
    except Exception:
        return out
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
            except sqlite3.DatabaseError:
                out[sym] = None
    finally:
        conn.close()
    return out


def read_control_freshness(db_path: Path, now_ts: float) -> Dict[str, Optional[float]]:
    """Bounded read of the two REST-covered control tables this detector
    cross-validates liquidation silence against. Same indexed
    `ORDER BY ts_ms DESC LIMIT 1` pattern (idx_mark_ts / idx_trade_ts),
    with the same `ts_ms <= now_ts*1000` no-lookahead bound as
    read_last_liquidation_ts (see that function's docstring)."""
    out: Dict[str, Optional[float]] = {t: None for t in _CONTROL_TABLES}
    upper_ms = int(now_ts * 1000)
    try:
        conn = _open_ro(db_path)
    except Exception:
        return out
    try:
        cur = conn.cursor()
        for table in _CONTROL_TABLES:
            try:
                cur.execute(f"SELECT ts_ms FROM {table} WHERE ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1", (upper_ms,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    out[table] = max(0.0, now_ts - int(row[0]) / 1000.0)
            except sqlite3.DatabaseError:
                out[table] = None
    finally:
        conn.close()
    return out


def read_native_ws_status(overall_path: Path = DEFAULT_OVERALL_PATH) -> Optional[str]:
    """Read-only consumption of the existing canonical overall.json's
    additive native_ws_status field. Never writes to this path -- see
    module docstring."""
    payload = _read_json(overall_path)
    if not payload:
        return None
    status = payload.get("native_ws_status")
    return status if isinstance(status, str) else None


def read_collector_component_alive(component_path: Path = DEFAULT_COLLECTOR_COMPONENT_PATH, now_ts: Optional[float] = None, max_age_sec: float = 180.0) -> Optional[bool]:
    """Cheap proxy for collector aliveness from its own already-written
    component file (no new process-enumeration subprocess is spawned here
    -- that responsibility belongs to tools/heartbeat_watchdog.py). Returns
    None (unknown, not a verdict) when the file is missing/corrupt or the
    timestamp cannot be parsed; only returns False when the file is present
    but clearly stale, which is real (if weaker than direct process-list)
    evidence of the collector being down."""
    payload = _read_json(component_path)
    if not payload:
        return None
    ts_raw = payload.get("ts_utc")
    if not ts_raw:
        return None
    try:
        import datetime as _dt

        text = str(ts_raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = _dt.datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        age = (now_ts if now_ts is not None else time.time()) - dt.timestamp()
    except Exception:
        return None
    return age <= max_age_sec


def evaluate_once(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    pid_meta_path: Path = DEFAULT_PID_META_PATH,
    overall_path: Path = DEFAULT_OVERALL_PATH,
    collector_component_path: Path = DEFAULT_COLLECTOR_COMPONENT_PATH,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Bounded read-only snapshot + pure policy evaluation. Does not write
    anything. now_ts is always explicit (never a bare time.time() call
    buried in the policy layer) so this function is safe to call for
    historical replay against arbitrary past instants."""
    start = time.time()
    now_ts = now_ts if now_ts is not None else time.time()

    symbol_info = discover_tracked_symbols(pid_meta_path)
    symbols = symbol_info["symbols"]

    error: Optional[str] = None
    last_liq_ts: Dict[str, Optional[int]] = {s: None for s in symbols}
    control_freshness: Dict[str, Optional[float]] = {t: None for t in _CONTROL_TABLES}
    try:
        last_liq_ts = read_last_liquidation_ts(db_path, symbols, now_ts=now_ts)
        control_freshness = read_control_freshness(db_path, now_ts)
    except Exception as exc:  # pragma: no cover - defensive, evaluate_liquidation_silence handles missing data
        error = f"data_read_failed:{exc}"

    native_ws_status = read_native_ws_status(overall_path)
    collector_alive = read_collector_component_alive(collector_component_path, now_ts=now_ts)

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
        "error": error,
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
    dry_run: bool = False,
) -> Dict[str, Any]:
    payload = evaluate_once(
        db_path=db_path,
        pid_meta_path=pid_meta_path,
        overall_path=overall_path,
        collector_component_path=collector_component_path,
        now_ts=now_ts,
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
    parser.add_argument("--now-ts", type=float, default=None, help="Epoch seconds to evaluate as-of (default: real now). Used for replay/rehearsal.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and print but do not write the component file.")
    args = parser.parse_args()

    payload = run_once(
        db_path=Path(args.db_path),
        health_root=Path(args.health_root),
        pid_meta_path=Path(args.pid_meta_path),
        overall_path=Path(args.overall_path),
        collector_component_path=Path(args.collector_component_path),
        now_ts=args.now_ts,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
