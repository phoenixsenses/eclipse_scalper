# execution/env_sanity.py -- SCALPER ETERNAL -- ENV SANITY CHECKER -- 2026 v1.0
# Runtime environment validation: Python version, env vars, disk, memory, network.
# Guardian-safe: never raises, never fatal.

from __future__ import annotations

import os
import sys
import socket
import shutil
from typing import Dict

from utils.logging import log_core

# ---------------------------------------------------------------------------
# Result constants
# ---------------------------------------------------------------------------
PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_PYTHON = (3, 10)
MIN_DISK_MB = 500
DEFAULT_RSS_LIMIT_MB = 2048  # warn if RSS exceeds this

REQUIRED_ENV_VARS = ("BINANCE_API_KEY", "BINANCE_API_SECRET")
BINANCE_HOST = "api.binance.com"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python_version() -> Dict[str, str]:
    try:
        ver = sys.version_info[:2]
        if ver >= MIN_PYTHON:
            return {"check": "python_version", "status": PASS,
                    "detail": f"{ver[0]}.{ver[1]}"}
        return {"check": "python_version", "status": FAIL,
                "detail": f"{ver[0]}.{ver[1]} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"}
    except Exception as e:
        return {"check": "python_version", "status": SKIP, "detail": str(e)}


def _check_env_vars() -> Dict[str, str]:
    try:
        missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v, "").strip()]
        if not missing:
            return {"check": "env_vars", "status": PASS,
                    "detail": f"all {len(REQUIRED_ENV_VARS)} present"}
        return {"check": "env_vars", "status": FAIL,
                "detail": f"missing: {', '.join(missing)}"}
    except Exception as e:
        return {"check": "env_vars", "status": SKIP, "detail": str(e)}


def _check_disk_space() -> Dict[str, str]:
    try:
        usage = shutil.disk_usage(os.getcwd())
        free_mb = usage.free / (1024 * 1024)
        if free_mb >= MIN_DISK_MB:
            return {"check": "disk_space", "status": PASS,
                    "detail": f"{free_mb:.0f} MB free"}
        return {"check": "disk_space", "status": WARN,
                "detail": f"{free_mb:.0f} MB free (< {MIN_DISK_MB} MB)"}
    except Exception as e:
        return {"check": "disk_space", "status": SKIP, "detail": str(e)}


def _check_memory() -> Dict[str, str]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return {"check": "memory_rss", "status": SKIP, "detail": "psutil not installed"}
    try:
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        limit = float(os.getenv("SANITY_RSS_LIMIT_MB", str(DEFAULT_RSS_LIMIT_MB)))
        if rss_mb <= limit:
            return {"check": "memory_rss", "status": PASS,
                    "detail": f"{rss_mb:.0f} MB (limit {limit:.0f} MB)"}
        return {"check": "memory_rss", "status": WARN,
                "detail": f"{rss_mb:.0f} MB exceeds limit {limit:.0f} MB"}
    except Exception as e:
        return {"check": "memory_rss", "status": SKIP, "detail": str(e)}


def _check_network() -> Dict[str, str]:
    try:
        socket.getaddrinfo(BINANCE_HOST, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return {"check": "network_dns", "status": PASS,
                "detail": f"resolved {BINANCE_HOST}"}
    except socket.gaierror as e:
        return {"check": "network_dns", "status": FAIL,
                "detail": f"cannot resolve {BINANCE_HOST}: {e}"}
    except Exception as e:
        return {"check": "network_dns", "status": SKIP, "detail": str(e)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_env_sanity() -> Dict[str, str]:
    """Run all environment sanity checks.

    Returns {check_name: status} where status is pass/fail/warn/skip.
    Guardian-safe: never raises.
    """
    results: Dict[str, str] = {}
    checks = [
        _check_python_version,
        _check_env_vars,
        _check_disk_space,
        _check_memory,
        _check_network,
    ]
    for fn in checks:
        try:
            r = fn()
            results[r["check"]] = r["status"]
            _log_result(r)
        except Exception as e:
            log_core.warning(f"ENV SANITY: check {fn.__name__} error: {e}")
    passed = sum(1 for v in results.values() if v == PASS)
    total = len(results)
    log_core.info(f"ENV SANITY COMPLETE: {passed}/{total} passed")
    return results


def _log_result(r: Dict[str, str]) -> None:
    status = r.get("status", "?")
    name = r.get("check", "?")
    detail = r.get("detail", "")
    msg = f"ENV SANITY [{status.upper()}] {name}"
    if detail:
        msg += f" -- {detail}"
    if status == FAIL:
        log_core.critical(msg)
    elif status == WARN:
        log_core.warning(msg)
    else:
        log_core.info(msg)
