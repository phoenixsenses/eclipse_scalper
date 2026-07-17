"""Read-only runtime process topology for the canonical operator dashboard.

Enumerates Eclipse process roles by command-line identity match using a
read-only ``psutil`` scan. It NEVER starts, stops, signals, or otherwise
manages any process; it only observes. If ``psutil`` is unavailable the module
fails closed (returns an empty, explicitly-degraded topology).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoleSpec:
    key: str
    label: str
    # All regexes that identify this role in a process command line.
    patterns: tuple[str, ...]
    expected_count: int
    # A role we expect to be OFF by default (e.g. live executor). Seeing it
    # running is a safety-relevant event, not a health failure.
    expected_off: bool = False
    category: str = "core"


# Canonical Eclipse role registry. Ordering here is the deterministic display
# order for the topology panel.
ROLE_REGISTRY: tuple[RoleSpec, ...] = (
    RoleSpec("collector_supervisor", "Collector Supervisor", (r"collector_supervisor",), 1),
    RoleSpec("microstructure_collector", "Microstructure Collector", (r"data\.microstructure_collector", r"microstructure_collector"), 1),
    RoleSpec("bookticker_collector", "BookTicker Collector", (r"data\.bookticker_collector", r"bookticker_collector"), 1),
    RoleSpec("oi_spot_poller", "OI/Spot Poller", (r"data\.oi_spot_poller", r"oi_spot_poller"), 1),
    RoleSpec("event_diary", "Event Diary", (r"data\.event_diary", r"event_diary"), 1),
    RoleSpec("heartbeat_watchdog", "Watchdog", (r"tools\.heartbeat_watchdog", r"heartbeat_watchdog"), 1),
    RoleSpec("s34_observer", "S34 Observer (v02 shadow mirror)", (r"s34_v_engine_v02_shadow_mirror",), 1, category="s34"),
    RoleSpec("s34_realtime_shadow_runner", "S34 Realtime Shadow Runner", (r"s34_realtime_shadow_runner",), 1, category="s34"),
    RoleSpec("s34_shadow_paper_runner", "S34 Shadow Paper Runner", (r"s34_shadow_paper_runner",), 1, category="s34"),
    RoleSpec("liquidation_silence_scheduler", "Liquidation-Silence Scheduler", (r"liquidation_silence.*schedul", r"s34.*silence.*schedul", r"liq.*silence.*schedul"), 0, category="gate2"),
    RoleSpec("canonical_dashboard", "Canonical Operator Dashboard (serve)", (r"s34_cascade_navigation_dashboard.*--serve", r"s34_cascade_navigation_dashboard"), 0, category="dashboard"),
    RoleSpec("secondary_live_chart", "Secondary: s34_live_chart", (r"s34_live_chart",), 0, category="dashboard_secondary"),
    RoleSpec("secondary_orderflow", "Secondary: orderflow chart", (r"orderflow_chart",), 0, category="dashboard_secondary"),
    RoleSpec("secondary_replay", "Secondary: S34 replay", (r"s34_replay",), 0, category="dashboard_secondary"),
    RoleSpec("live_executor", "Live Executor", (r"s34_state_machine_live_executor", r"state_machine_live", r"live_executor"), 0, expected_off=True, category="execution"),
)


@dataclass
class ProcMatch:
    pid: int
    cmdline: str
    create_time: float | None
    identity_ok: bool


def _iter_python_procs() -> list[dict[str, Any]]:
    """Return read-only snapshots of python processes. Fail closed if psutil
    is unavailable."""
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = p.info
            name = str(info.get("name") or "")
            cmd = info.get("cmdline") or []
            cmdline = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
            if "python" not in name.lower() and "python" not in cmdline.lower():
                continue
            out.append({
                "pid": int(info.get("pid")),
                "cmdline": cmdline,
                "create_time": info.get("create_time"),
            })
        except Exception:
            continue
    return out


def _compile(patterns: tuple[str, ...]):
    import re
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def scan_topology(procs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the read-only process topology.

    ``procs`` may be injected (fixtures/tests); otherwise a live read-only scan
    is performed. Returns a plain dict (adapter wraps it into a view model).
    """
    read_ts = time.time()
    psutil_available = True
    if procs is None:
        try:
            import psutil  # noqa: F401  # type: ignore
        except Exception:
            psutil_available = False
        procs = _iter_python_procs() if psutil_available else []

    remaining = list(procs)
    roles: list[dict[str, Any]] = []
    matched_pids: set[int] = set()
    scheduler_count = 0
    live_executor_count = 0

    for spec in ROLE_REGISTRY:
        regexes = _compile(spec.patterns)
        hits: list[ProcMatch] = []
        for pr in remaining:
            cmd = str(pr.get("cmdline") or "")
            if any(rx.search(cmd) for rx in regexes):
                pid = int(pr.get("pid"))
                hits.append(ProcMatch(pid=pid, cmdline=cmd, create_time=pr.get("create_time"), identity_ok=True))
                matched_pids.add(pid)
        actual = len(hits)
        if spec.key == "liquidation_silence_scheduler":
            scheduler_count = actual
        if spec.key == "live_executor":
            live_executor_count = actual

        duplicate = max(0, actual - max(spec.expected_count, 1)) if actual > 0 else 0
        # Health classification
        if spec.expected_off:
            health = "OFF_EXPECTED" if actual == 0 else "RUNNING_UNEXPECTED"
        elif spec.expected_count >= 1 and actual == 0:
            health = "MISSING"
        elif actual > max(spec.expected_count, 1):
            health = "DUPLICATE"
        elif actual >= 1:
            health = "RUNNING"
        else:
            health = "ABSENT_OPTIONAL"

        roles.append({
            "key": spec.key,
            "label": spec.label,
            "category": spec.category,
            "expected_count": spec.expected_count,
            "expected_off": spec.expected_off,
            "actual_count": actual,
            "duplicate_count": duplicate,
            "health": health,
            "pids": [h.pid for h in hits],
            "start_times": [h.create_time for h in hits],
            "identity_ok": all(h.identity_ok for h in hits) if hits else None,
            "cmdlines": [h.cmdline[:400] for h in hits],
        })

    # Orphans: python processes matching an Eclipse-ish marker but no known role.
    import re
    eclipse_marker = re.compile(r"eclipse|s34|microstructure|bookticker|oi_spot|event_diary|collector|watchdog", re.IGNORECASE)
    orphans = []
    for pr in remaining:
        pid = int(pr.get("pid"))
        if pid in matched_pids:
            continue
        cmd = str(pr.get("cmdline") or "")
        if eclipse_marker.search(cmd):
            orphans.append({"pid": pid, "cmdline": cmd[:400]})

    return {
        "read_timestamp": read_ts,
        "psutil_available": psutil_available,
        "roles": roles,
        "orphans": orphans,
        "scheduler_count": scheduler_count,
        "live_executor_count": live_executor_count,
        "watchdog_pids": next((r["pids"] for r in roles if r["key"] == "heartbeat_watchdog"), []),
        "total_python_procs": len(procs),
    }
