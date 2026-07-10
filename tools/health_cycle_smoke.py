from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools import heartbeat_watchdog as hw

REQUIRED_TOP_LEVEL_KEYS = ("ts_utc", "mode", "state", "components")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline health-cycle smoke: assert collector health transitions ok -> degraded -> ok."
    )
    p.add_argument("--db-path", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--cycle-sec", type=float, default=10.0)
    p.add_argument("--down-sec", type=float, default=4.0)
    p.add_argument("--max-seconds", type=float, default=22.0)
    p.add_argument("--stats-interval", type=float, default=2.0)
    p.add_argument("--out-dir", default="logs/health/smoke")
    # --root scopes every health/report/heartbeat path this run touches:
    # logs/health/ (component files + canonical overall.json, produced
    # in-process here exactly like the real watchdog would), reports/, and
    # logs/collector_heartbeat.json. Production default "." preserves the
    # original real-path behavior for ad-hoc operator runs; isolated test
    # runs pass a temp directory so nothing here ever touches the live
    # runtime's actual logs/health/overall.json or depends on the real,
    # already-running heartbeat watchdog process being alive. See
    # reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md Part C.
    p.add_argument("--root", default=".")
    p.add_argument("--poll-interval-sec", type=float, default=0.5)
    # Only ever creates/seeds --db-path when it does not already exist --
    # never touches a real database. Isolated test runs pass a fresh temp
    # db path together with this flag; production/ops runs against the
    # real data/microstructure.db must never set it.
    p.add_argument("--seed-market-data", action="store_true")
    return p


def _read_health(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _collector_state(obj: Dict[str, Any]) -> tuple[str, Optional[bool]]:
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    collector = comps.get("collector") if isinstance(comps.get("collector"), dict) else {}
    status = str(collector.get("status") or "").lower()
    connected = collector.get("connected")
    if isinstance(connected, bool):
        return status, connected
    return status, None


def _validate_snapshot(obj: Dict[str, Any], expected_state: str) -> tuple[bool, str]:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in obj:
            return False, f"missing_key:{key}"
    state = str(obj.get("state") or "").lower()
    if state != expected_state:
        return False, f"state_mismatch:expected={expected_state}:actual={state}"
    c_status, c_connected = _collector_state(obj)
    if expected_state == "ok":
        if c_status != "ok" or c_connected is not True:
            return False, f"collector_mismatch_ok:status={c_status}:connected={c_connected}"
    elif expected_state == "degraded":
        # Only components.collector.connected is checked here, not .status.
        # Discovered 2026-07-10 while isolating this test (Part C): for a
        # brief disconnect shorter than data.microstructure_collector.py's
        # own stall_timeout_sec (default 45s), collector.json's *status*
        # field legitimately stays "ok" -- it is staleness-gated, not
        # connection-gated, by design (data.microstructure_collector.py::
        # _write_heartbeat: status only flips once progress has been stale
        # for stall_timeout_sec). The canonical top-level "degraded" state
        # observed here comes from the faster native_ws_policy connection
        # signal instead (native_ws_degraded:NATIVE_WS_DISCONNECTED), which
        # is the correct, intended fast-detection layer. Requiring
        # collector.status=="degraded" too was a false invariant that a
        # short simulated outage can never satisfy; it previously appeared
        # to pass only because this test read the real, live-mutating
        # production overall.json instead of its own isolated one (see
        # reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md
        # Part C) -- it was not actually being exercised.
        if c_connected is not False:
            return False, f"collector_mismatch_degraded:status={c_status}:connected={c_connected}"
    return True, "ok"


def _extract_reconnects(obj: Dict[str, Any]) -> int:
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    collector = comps.get("collector") if isinstance(comps.get("collector"), dict) else {}
    try:
        return int(collector.get("reconnects_last_5m", 0))
    except Exception:
        return 0


def _cleanup_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _seed_market_data(db_path: Path) -> None:
    """Only ever runs against a not-yet-existing db_path (checked by the
    caller) -- creates the three source-freshness tables native_ws_policy
    reads, with one fresh row each, so the smoke test's overall/native_ws
    state is driven purely by the simulated collector's own connect/
    disconnect cycle, not by an empty, permanently-stale synthetic database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        now_ms = int(time.time() * 1000)
        for table in ("agg_trades", "mark_prices", "liquidations"):
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT)")
            conn.execute(f"INSERT INTO {table} (ts_ms, symbol) VALUES (?, ?)", (now_ms, "ETHUSDT"))
        conn.commit()
    finally:
        conn.close()


def _refresh_canonical_overall(*, root: Path, db_path: Path, max_age_sec: int) -> None:
    """Runs exactly one heartbeat_watchdog evaluation cycle scoped to
    `root`, in-process -- no second Python process is spawned (only the
    simulated collector subprocess runs concurrently), and no real
    production heartbeat_watchdog process is depended on. Mirrors what the
    real watchdog does every --interval-sec in production."""
    prev_root, prev_log_health, prev_reports = hw.ROOT, hw.LOG_HEALTH, hw.REPORTS
    try:
        hw.ROOT = root
        hw.LOG_HEALTH = root / "logs" / "health"
        hw.REPORTS = root / "reports"
        hw.run_once(
            max_age_sec=max_age_sec,
            expect_bookticker=False,
            expect_detector=False,
            expect_runtime=False,
            db_path=db_path,
        )
    finally:
        hw.ROOT, hw.LOG_HEALTH, hw.REPORTS = prev_root, prev_log_health, prev_reports


def run_smoke(args: argparse.Namespace) -> int:
    out_dir = Path(str(args.out_dir))
    root = Path(str(args.root)).resolve()
    health_root = root / "logs" / "health"
    heartbeat_path = root / "logs" / "collector_heartbeat.json"
    overall_path = health_root / "overall.json"
    db_path = Path(str(args.db_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("overall_snapshot*.json"):
        f.unlink(missing_ok=True)

    if bool(args.seed_market_data) and not db_path.exists():
        _seed_market_data(db_path)

    pid_file = root / "logs" / "pids" / "paper_watchdog.pid"
    meta_file = root / "logs" / "pids" / "paper_watchdog.json"
    pid_exists_before = pid_file.exists()
    meta_exists_before = meta_file.exists()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "data.microstructure_collector",
        "--symbols",
        str(args.symbols),
        "--db-path",
        str(db_path),
        "--stats-interval",
        str(args.stats_interval),
        "--simulate-connection",
        "--simulate-cycle-sec",
        str(args.cycle_sec),
        "--simulate-down-sec",
        str(args.down_sec),
        "--simulate-max-seconds",
        str(args.max_seconds),
        "--health-root",
        str(health_root),
        "--heartbeat-path",
        str(heartbeat_path),
    ]

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        end_ts = time.time() + float(args.max_seconds) + 5.0
        last_obj: Optional[Dict[str, Any]] = None
        phase = 0
        reconnect_marks: list[int] = []
        snapshots: list[Path] = []
        file_names = [
            "overall_snapshot1_ok.json",
            "overall_snapshot2_degraded.json",
            "overall_snapshot3_ok.json",
        ]
        expected_states = ["ok", "degraded", "ok"]

        while time.time() < end_ts:
            # This process, not the collector subprocess or any real
            # background watchdog, is what turns the isolated collector.json
            # into canonical overall.json for this run.
            _refresh_canonical_overall(root=root, db_path=db_path, max_age_sec=180)
            obj = _read_health(overall_path)
            if isinstance(obj, dict):
                last_obj = obj
                want = expected_states[phase]
                ok, reason = _validate_snapshot(obj, want)
                if ok:
                    snap_path = out_dir / file_names[phase]
                    snap_path.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")
                    snapshots.append(snap_path)
                    reconnect_marks.append(_extract_reconnects(obj))
                    phase += 1
                    if phase >= 3:
                        break
            if proc.poll() is not None and phase < 3:
                break
            time.sleep(max(0.1, float(args.poll_interval_sec)))

        if phase < 3:
            last_state = None if last_obj is None else last_obj.get("state")
            print(f"health_cycle_smoke fail transition_timeout phase={phase} last_state={last_state}")
            return 1

        if not (reconnect_marks[0] <= reconnect_marks[1] <= reconnect_marks[2]):
            print(
                "health_cycle_smoke fail reconnects_non_monotonic "
                f"values={reconnect_marks}"
            )
            return 1

        if (not pid_exists_before and pid_file.exists()) or (not meta_exists_before and meta_file.exists()):
            print("health_cycle_smoke fail watchdog_registry_created_unexpectedly")
            return 1

        print("health_cycle_smoke ok")
        for p in snapshots:
            print(f"- snapshot={p}")
        return 0
    except AssertionError as e:
        print(f"health_cycle_smoke fail assertion={e}")
        return 1
    except Exception as e:
        print(f"health_cycle_smoke error runtime={type(e).__name__}:{e}")
        return 2
    finally:
        _cleanup_process(proc)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
