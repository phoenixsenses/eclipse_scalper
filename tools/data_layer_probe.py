from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TS_CANDIDATES = [
    "ts",
    "timestamp",
    "time",
    "ts_ms",
    "time_ms",
    "event_time",
    "event_ts",
    "T",
    "t",
    "created_at",
    "updated_at",
    "datetime",
]
KEY_TABLES = ["agg_trades", "mark_prices", "liquidations"]


@dataclass(frozen=True)
class TableSample:
    table: str
    ts_col: Optional[str]
    max_ts: Optional[float]
    age_sec: Optional[int]
    count_recent: Optional[int]
    count_total: int
    note: str = ""


def detect_ts_col(columns: List[str]) -> Optional[str]:
    lower_to_actual = {str(c).lower(): str(c) for c in columns}
    for cand in TS_CANDIDATES:
        found = lower_to_actual.get(cand.lower())
        if found:
            return found
    return None


def normalize_ts_to_seconds(mx: int | float) -> float:
    x = float(mx)
    if x > 1e12:
        return x / 1000.0
    return x


def parse_heartbeat_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(str(text or "").strip())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def serialize_heartbeat_json(hb: Dict[str, Any]) -> str:
    return json.dumps(hb, ensure_ascii=True, sort_keys=True)


def is_heartbeat_fresh(hb: Dict[str, Any], fresh_sec: int, now_ts: Optional[float] = None) -> bool:
    ts = hb.get("ts_utc")
    if not ts:
        return False
    try:
        ts_float = float(datetime_from_iso(str(ts)))
    except Exception:
        return False
    now = float(now_ts if now_ts is not None else time.time())
    return (now - ts_float) <= float(max(1, fresh_sec))


def datetime_from_iso(raw: str) -> float:
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def compute_row_count_deltas(
    start_counts: Dict[str, int],
    end_counts: Dict[str, int],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    keys = set(start_counts) | set(end_counts)
    for k in keys:
        out[str(k)] = int(end_counts.get(k, 0) - start_counts.get(k, 0))
    return out


def compute_probe_ok(
    collector_process_live: bool,
    collector_data_live: bool,
    diary_process_live: bool,
    require_diary: bool,
) -> Tuple[bool, Optional[str]]:
    if not collector_process_live or not collector_data_live:
        return False, None
    if require_diary and not diary_process_live:
        return False, None
    if (not require_diary) and (not diary_process_live):
        return True, "diary=down (not required)"
    return True, None


def _normalize_process_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _powershell_json(command: str) -> Any:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "powershell failure")
    payload = (proc.stdout or "").strip()
    if not payload:
        return []
    return json.loads(payload)


def _psutil_pid_cmdline(pid: int) -> Optional[str]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process(int(pid))
        return " ".join(proc.cmdline())
    except Exception:
        return None


def _cim_pid_cmdline(pid: int) -> Optional[str]:
    try:
        obj = _powershell_json(
            f"Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" "
            "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        rows = _normalize_process_rows(obj)
        if not rows:
            return None
        return str(rows[0].get("CommandLine") or "")
    except Exception:
        return None


def _scan_process_modules() -> Dict[str, int]:
    counts = {"data.microstructure_collector": 0, "data.event_diary": 0}
    # try psutil first
    try:
        import psutil  # type: ignore

        for p in psutil.process_iter(["name", "cmdline"]):
            name = str((p.info or {}).get("name") or "").lower()
            if name not in {"python.exe", "python"}:
                continue
            cl = " ".join((p.info or {}).get("cmdline") or [])
            for k in list(counts):
                if k in cl:
                    counts[k] += 1
        return counts
    except Exception:
        pass
    # fallback CIM scan
    try:
        obj = _powershell_json(
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
            "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        for row in _normalize_process_rows(obj):
            cl = str(row.get("CommandLine") or "")
            for k in list(counts):
                if k in cl:
                    counts[k] += 1
    except Exception:
        return counts
    return counts


def _pid_exists(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} | Out-Null"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(r[0]) for r in rows if r and str(r[0]) and not str(r[0]).startswith("sqlite_")]


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows if len(r) > 1]


def _safe_scalar(conn: sqlite3.Connection, sql: str) -> Optional[float]:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.DatabaseError:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


def _safe_count(conn: sqlite3.Connection, sql: str) -> Optional[int]:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.DatabaseError:
        return None
    if not row or row[0] is None:
        return None
    try:
        return int(row[0])
    except Exception:
        return None


def _table_sample(conn: sqlite3.Connection, table: str, now_sec: float, window_sec: int) -> TableSample:
    cols = _table_columns(conn, table)
    ts_col = detect_ts_col(cols)
    total = _safe_count(conn, f"SELECT COUNT(*) FROM {table}") or 0
    if ts_col is None:
        return TableSample(table=table, ts_col=None, max_ts=None, age_sec=None, count_recent=None, count_total=total, note="NO_TS_COL")
    max_ts = _safe_scalar(conn, f"SELECT MAX({ts_col}) FROM {table}")
    if max_ts is None:
        return TableSample(table=table, ts_col=ts_col, max_ts=None, age_sec=None, count_recent=None, count_total=total, note="NO_ROWS")
    max_sec = normalize_ts_to_seconds(max_ts)
    age_sec = int(max(0.0, now_sec - max_sec))
    # estimate if column is ms epoch by max value.
    if max_ts > 1e12:
        cutoff = (now_sec - float(window_sec)) * 1000.0
    else:
        cutoff = now_sec - float(window_sec)
    recent = _safe_count(conn, f"SELECT COUNT(*) FROM {table} WHERE {ts_col} >= {cutoff}")
    return TableSample(table=table, ts_col=ts_col, max_ts=max_ts, age_sec=age_sec, count_recent=recent, count_total=total, note="")


def _collect_table_samples(conn: sqlite3.Connection, window_sec: int) -> Dict[str, TableSample]:
    now_sec = time.time()
    available = set(_list_tables(conn))
    out: Dict[str, TableSample] = {}
    for t in KEY_TABLES:
        if t in available:
            out[t] = _table_sample(conn, t, now_sec=now_sec, window_sec=window_sec)
    return out


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            count += buf.count(b"\n")
    return count


def _append_probe_json(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        return


def _read_pid(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _read_last_run_record_map(path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        try:
            pid = int(rec.get("pid"))
            name = str(rec.get("name") or "")
        except Exception:
            continue
        out[(pid, name)] = rec
    return out


def _is_pid_alive_with_module(pid: int, module_substring: str, run_map: Dict[Tuple[int, str], Dict[str, Any]]) -> bool:
    if not _pid_exists(pid):
        return False
    cmd = _psutil_pid_cmdline(pid) or _cim_pid_cmdline(pid) or ""
    if cmd:
        return module_substring in cmd
    # fallback: if cmdline unavailable due permission, trust starter run map for this pid+module.
    expected_name = "microstructure_collector" if "microstructure_collector" in module_substring else "event_diary"
    rec = run_map.get((int(pid), expected_name))
    if not rec:
        return False
    rec_cmd = str(rec.get("cmd") or "")
    return module_substring in rec_cmd


def probe(
    db_path: Path,
    csv_path: Path,
    symbols: List[str],
    wait_sec: int,
    fresh_sec: int,
    log_path: Path,
    require_diary: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    _ = symbols  # kept for CLI parity and future symbol-level checks
    details: Dict[str, Any] = {}
    if not db_path.exists():
        return False, {"error": f"db_missing:{db_path}"}
    if not csv_path.exists():
        return False, {"error": f"csv_missing:{csv_path}"}

    pid_root = log_path.parent / "pids"
    run_map = _read_last_run_record_map(log_path.parent / "data_layer_runs.jsonl")
    micro_pid = _read_pid(pid_root / "microstructure_collector.pid")
    diary_pid = _read_pid(pid_root / "event_diary.pid")
    details["micro_pid"] = micro_pid
    details["event_diary_pid"] = diary_pid
    details["micro_pid_alive"] = bool(
        micro_pid and _is_pid_alive_with_module(micro_pid, "data.microstructure_collector", run_map)
    )
    details["event_diary_pid_alive"] = bool(diary_pid and _is_pid_alive_with_module(diary_pid, "data.event_diary", run_map))

    scan_counts = _scan_process_modules()
    details["scan_micro_count"] = scan_counts.get("data.microstructure_collector", 0)
    details["scan_diary_count"] = scan_counts.get("data.event_diary", 0)

    with sqlite3.connect(str(db_path)) as conn:
        start_samples = _collect_table_samples(conn, window_sec=max(1, int(wait_sec)))
    start_counts = {k: int(v.count_total) for k, v in start_samples.items()}
    csv_mtime_0 = csv_path.stat().st_mtime
    csv_lines_0 = _count_lines(csv_path)
    time.sleep(max(1, int(wait_sec)))
    with sqlite3.connect(str(db_path)) as conn:
        end_samples = _collect_table_samples(conn, window_sec=max(1, int(wait_sec)))
    end_counts = {k: int(v.count_total) for k, v in end_samples.items()}
    csv_mtime_1 = csv_path.stat().st_mtime
    csv_lines_1 = _count_lines(csv_path)

    table_details: Dict[str, Any] = {}
    deltas = compute_row_count_deltas(start_counts, end_counts)
    data_live = False
    for t, end in end_samples.items():
        start = start_samples.get(t)
        max_ts_delta = None
        count_delta = None
        if start and start.max_ts is not None and end.max_ts is not None:
            max_ts_delta = float(end.max_ts) - float(start.max_ts)
        if start:
            count_delta = int(deltas.get(t, int(end.count_total - start.count_total)))
        table_details[t] = {
            "ts_col": end.ts_col,
            "max_ts": end.max_ts,
            "age_sec": end.age_sec,
            "count_recent": end.count_recent,
            "count_total": end.count_total,
            "count_delta": count_delta,
            "max_ts_delta": max_ts_delta,
            "note": end.note,
        }
        if (end.age_sec is not None and end.age_sec <= int(max(1, fresh_sec))) or ((count_delta or 0) > 0):
            data_live = True

    csv_mtime_delta = float(csv_mtime_1 - csv_mtime_0)
    csv_line_delta = int(csv_lines_1 - csv_lines_0)
    details["csv_mtime_delta_sec"] = csv_mtime_delta
    details["csv_line_delta"] = csv_line_delta
    details["row_count_deltas"] = deltas
    details["tables"] = table_details
    heartbeat_path = log_path.parent / "collector_heartbeat.json"
    hb_fresh = False
    hb_data: Optional[Dict[str, Any]] = None
    if heartbeat_path.exists():
        hb_data = parse_heartbeat_json(heartbeat_path.read_text(encoding="utf-8", errors="ignore"))
        if hb_data:
            hb_fresh = is_heartbeat_fresh(hb_data, fresh_sec=fresh_sec)
    details["heartbeat_path"] = str(heartbeat_path)
    details["heartbeat_fresh"] = hb_fresh
    if hb_data:
        details["heartbeat_connected"] = bool(hb_data.get("connected"))
        details["heartbeat_last_flush_ts_utc"] = hb_data.get("last_flush_ts_utc")
        details["heartbeat_last_message_ts_utc"] = hb_data.get("last_message_ts_utc")
    collector_data_live = bool(data_live or hb_fresh)
    collector_process_live = bool(details["micro_pid_alive"] or details["scan_micro_count"] > 0)
    diary_process_live = bool(details["event_diary_pid_alive"] or details["scan_diary_count"] > 0)
    diary_data_live = bool(csv_mtime_delta > 0 or csv_line_delta > 0)
    details["collector_data_live"] = collector_data_live
    details["collector_process_live"] = collector_process_live
    details["diary_data_live"] = diary_data_live
    details["diary_process_live"] = diary_process_live
    details["require_diary"] = bool(require_diary)
    ok, warn_line = compute_probe_ok(
        collector_process_live=collector_process_live,
        collector_data_live=collector_data_live,
        diary_process_live=diary_process_live,
        require_diary=bool(require_diary),
    )
    if warn_line:
        details["warning"] = warn_line
    details["data_live"] = bool(collector_data_live or diary_data_live)
    details["process_live"] = bool(collector_process_live and (diary_process_live or (not require_diary)))

    record = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": ok,
        "details": details,
    }
    _append_probe_json(log_path, record)
    return ok, details


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data layer liveness probe (process + db/csv).")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--wait-sec", type=int, default=20)
    p.add_argument("--fresh-sec", type=int, default=120)
    p.add_argument("--log", default="logs/data_layer_probe.jsonl")
    p.add_argument("--require-diary", action="store_true", help="Require event_diary process to be live.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ok, details = probe(
            db_path=Path(str(args.db)),
            csv_path=Path(str(args.csv)),
            symbols=_parse_symbols(args.symbols),
            wait_sec=int(args.wait_sec),
            fresh_sec=int(args.fresh_sec),
            log_path=Path(str(args.log)),
            require_diary=bool(args.require_diary),
        )
    except Exception as exc:
        print(f"FAIL data_layer_not_running err={exc}")
        return 3

    print("OK live" if ok else "FAIL stale")
    print(f"process_live={details.get('process_live')}")
    print(f"data_live={details.get('data_live')}")
    print(f"micro_pid_alive={details.get('micro_pid_alive')} scan_micro_count={details.get('scan_micro_count')}")
    print(f"event_diary_pid_alive={details.get('event_diary_pid_alive')} scan_diary_count={details.get('scan_diary_count')}")
    print(f"csv_mtime_delta_sec={details.get('csv_mtime_delta_sec')} csv_line_delta={details.get('csv_line_delta')}")
    print(
        f"heartbeat_fresh={details.get('heartbeat_fresh')} "
        f"heartbeat_connected={details.get('heartbeat_connected')}"
    )
    if details.get("warning"):
        print(f"WARN {details.get('warning')}")
    tables = details.get("tables", {})
    for t in KEY_TABLES:
        d = tables.get(t)
        if not d:
            print(f"table={t} missing")
            continue
        print(
            f"table={t} ts_col={d.get('ts_col')} max_ts={d.get('max_ts')} age_sec={d.get('age_sec')} "
            f"count_delta={d.get('count_delta')} recent={d.get('count_recent')} note={d.get('note')}"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
