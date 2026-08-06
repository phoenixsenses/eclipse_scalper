"""Central source-path + freshness-threshold configuration and the read-only
context object passed to every adapter.

All paths are resolved relative to the repository root. Adapters receive a
:class:`DashboardContext`; tests inject a context pointed at fixture dirs so no
adapter ever hard-codes an absolute path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# dashboard/backend/sources.py -> parents[2] == repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Freshness thresholds (seconds) per logical source. Deliberately generous where
# an artifact updates on a slow cadence; tight where it is a live heartbeat.
FRESHNESS = {
    "process_topology": 30.0,
    "s34_state": 300.0,
    "s34_shadow_evidence": 300.0,
    "execution_state": 300.0,
    "gate2_detector": 900.0,
    "native_ws": 120.0,
    "collector_rows": 300.0,
    "storage": 600.0,
    "research": 7 * 24 * 3600.0,
    "governance": 30 * 24 * 3600.0,
    "shadow_ledger": 900.0,
    "shadow_paper_runner": 180.0,
    "v02_mirror_bucket": 540.0,
    "shadow_paper_activity": 60.0,
    "paper_bucket_ledger": 120.0,
    "liq_anomaly": 120.0,
}


@dataclass
class DashboardContext:
    """Read-only context handed to adapters."""

    repo_root: Path = REPO_ROOT
    now: float | None = None  # if set (tests), used as the read clock
    # Optional injected process snapshot (tests / probes); None => live scan.
    proc_snapshot: list[dict[str, Any]] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    # --- path helpers -----------------------------------------------------
    def p(self, *parts: str) -> Path:
        return self.repo_root.joinpath(*parts)

    @property
    def system_state(self) -> Path:
        return self.p("SYSTEM_STATE.md")

    @property
    def env_path(self) -> Path:
        return self.p(".env")

    @property
    def out_dir(self) -> Path:
        return self.p("reports", "research", "s34")

    @property
    def runtime_dir(self) -> Path:
        return self.p("runtime")

    @property
    def shadow_dir(self) -> Path:
        return self.p("reports", "shadow")

    @property
    def microstructure_db(self) -> Path:
        """The db the collectors are writing NOW, resolved through rotation state.

        Hard-coding `data/microstructure.db` made the storage-health panel report the
        FROZEN segment: since the 2026-07-23 cutover it showed 836 GiB of history
        instead of the live file, and once that segment is reclaimed it would have
        gone permanently MISSING/MEDIUM -- the panel meant to report on storage,
        broken by the storage operation it exists to report on.
        """
        try:
            from ami.storage.union_reader import current_live_db_path
            return Path(current_live_db_path())
        except Exception:
            # fail back to the historical path rather than taking the dashboard down;
            # a wrong-but-present path degrades one panel, an exception degrades the page
            return self.p("data", "microstructure.db")

    def threshold(self, key: str) -> float | None:
        return FRESHNESS.get(key)
