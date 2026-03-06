from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_watchdog_record(pid: int, cmdline_sig: str, exe_path: str = "", parent_pid: int | None = None, repo_root: str = "") -> dict:
    return {
        "role": "paper_watchdog",
        "pid": int(pid),
        "start_ts_utc": utc_iso(),
        "cmdline_sig": str(cmdline_sig or "").strip(),
        "exe_path": str(exe_path or ""),
        "parent_pid": (None if parent_pid is None else int(parent_pid)),
        "repo_root": str(repo_root or ""),
    }


def is_identity_match(expected_sig: str, observed_cmdline: str | None) -> bool:
    exp = str(expected_sig or "").strip().lower()
    obs = str(observed_cmdline or "").strip().lower()
    if not exp or not obs:
        return False
    if exp in obs:
        return True
    # relaxed token-based match to allow extra flags/order differences
    tokens = [t for t in exp.split() if t]
    return all(t in obs for t in tokens)


def parse_iso_utc(ts: str | None) -> Optional[datetime]:
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def evaluate_watchdog_identity(
    record: dict | None,
    observed: dict | None,
    expected_sig: str,
    creation_skew_sec: float = 2.0,
) -> tuple[bool, str]:
    """Evaluate whether observed process matches a watchdog registry record."""
    if not isinstance(record, dict):
        return False, "registry_missing"
    if str(record.get("role", "")).strip() != "paper_watchdog":
        return False, "role_mismatch"
    if not isinstance(observed, dict):
        return False, "pid_not_running"
    try:
        rpid = int(record.get("pid"))
        opid = int(observed.get("pid"))
    except Exception:
        return False, "invalid_pid"
    if rpid <= 0 or opid <= 0 or rpid != opid:
        return False, "pid_mismatch"
    cmd = observed.get("command_line")
    if not is_identity_match(expected_sig or str(record.get("cmdline_sig", "")), cmd):
        return False, "pid_reuse_signature_mismatch"
    rec_ts = parse_iso_utc(record.get("start_ts_utc"))
    obs_ts = parse_iso_utc(observed.get("creation_ts_utc"))
    if rec_ts and obs_ts:
        if obs_ts < (rec_ts - timedelta(seconds=max(0.0, float(creation_skew_sec)))):
            return False, "creation_time_mismatch"
    return True, "identity_match"
