"""Panel 7 — Storage & Data Integrity (read-only, bounded).

Never reads a large DB or JSONL wholesale: uses stat/metadata, drive free space,
and a single bounded ``mode=ro`` query with a timeout to confirm accessibility
and the newest row time. No write is ever performed.
"""
from __future__ import annotations

import shutil
import time

from dashboard.backend.readers import open_ro
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)


def _recent_row_ts(db_path) -> tuple[float | None, str | None]:
    """Bounded probe: newest ts across a couple of known tables, mode=ro."""
    try:
        conn = open_ro(db_path)
    except Exception as exc:
        return None, f"open_failed:{type(exc).__name__}"
    try:
        conn.execute("PRAGMA query_only = ON")
        newest = None
        for tbl, col in (("mark_prices", "ts"), ("agg_trades", "ts")):
            try:
                cur = conn.execute(f"SELECT MAX({col}) FROM {tbl}")
                row = cur.fetchone()
                if row and row[0] is not None:
                    v = float(row[0])
                    # ts may be epoch seconds or ms; normalise ms->s if huge.
                    if v > 1e12:
                        v /= 1000.0
                    newest = v if newest is None else max(newest, v)
            except Exception:
                continue
        return newest, "ok"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build(ctx: DashboardContext) -> PanelViewModel:
    db_path = ctx.microstructure_db
    read_ts = ctx.now if ctx.now is not None else time.time()

    vm = PanelViewModel(
        key="storage",
        title="Storage & Data Integrity",
        source_path=str(db_path),
        read_timestamp=read_ts,
        freshness_threshold_seconds=ctx.threshold("storage"),
        provenance_status=ProvenanceStatus.DIRECT,
    )

    exists = db_path.exists()
    size_gb = None
    if exists:
        try:
            size_gb = round(db_path.stat().st_size / (1024 ** 3), 3)
        except OSError:
            size_gb = None

    free_gb = None
    try:
        free_gb = round(shutil.disk_usage(str(db_path.parent if exists else ctx.repo_root)).free / (1024 ** 3), 2)
    except OSError:
        free_gb = None

    if not exists:
        vm.status = "MISSING"
        vm.parse_status = ParseStatus.MISSING
        vm.trust_state = TrustState.MISSING
        vm.severity = Severity.MEDIUM
        vm.fields = {"db_path_exists": False, "drive_free_gb": free_gb}
        vm.summary = "microstructure DB not found at expected path (fail-closed)"
        vm.add_reason("db_missing")
        return vm

    newest, probe_status = _recent_row_ts(db_path)
    vm.source_timestamp = newest
    vm.fields = {
        "db_path_exists": True,
        "db_size_gb": size_gb,
        "drive_free_gb": free_gb,
        "mode_ro_accessible": probe_status == "ok",
        "probe_status": probe_status,
        "newest_row_epoch": newest,
        "bounded_read": True,
        "wholesale_read": False,
    }
    warn_free = free_gb is not None and free_gb < 20.0
    if probe_status != "ok":
        vm.severity = Severity.MEDIUM
        vm.trust_state = TrustState.MALFORMED
        vm.add_reason("db_probe_failed")
    elif warn_free:
        vm.severity = Severity.MEDIUM
        vm.add_reason("low_disk_free")
    else:
        vm.severity = Severity.OK
    vm.status = "ACTIVE"
    vm.summary = (
        f"db={size_gb}GB free={free_gb}GB mode=ro_ok={probe_status=='ok'} "
        f"newest_row={'set' if newest else 'unknown'}"
    )
    return vm
