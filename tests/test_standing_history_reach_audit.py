"""Pin how far back the standing roles reach, so a deeper window cannot land quietly.

SYSTEM_STATE section 265 measured this per-file and got 7 days. The real answer was
14, hidden two imports away, and the wrong number was used to argue an 836 GiB
segment was safe to delete. The point of this test is not the number -- it is that
raising the number must be a deliberate, visible act.
"""

from __future__ import annotations

from tools.audit_standing_history_reach import DAY_MS, audit, standing_roles


# Raising this is a decision, not a detail: every extra day is history the archive
# and the live DB must actually be able to serve. See SYSTEM_STATE section 267-A.
REGISTERED_DEEPEST_DAYS = 14.0


def test_no_standing_role_reaches_further_than_the_registered_maximum():
    report = [r for r in audit() if "deepest_ms" in r]
    assert report, "the audit found no standing roles at all"
    deepest = max(r["deepest_ms"] for r in report) / DAY_MS
    offenders = [f"{r['role']} -> {r['deepest_days']}d at {r['deepest_at']}"
                 for r in report if r["deepest_days"] > REGISTERED_DEEPEST_DAYS]
    assert deepest <= REGISTERED_DEEPEST_DAYS, (
        "a standing role now reads further back than the registered maximum "
        f"({deepest}d > {REGISTERED_DEEPEST_DAYS}d). Confirm the estate can serve it "
        f"before raising REGISTERED_DEEPEST_DAYS: {offenders}")


def test_the_audit_actually_follows_imports():
    """A per-file scan returns 7 days here; only the transitive walk finds 14."""
    rows = {r["role"]: r for r in audit() if "deepest_ms" in r}
    monitor = rows["tools.liq_anomaly_monitor"]
    assert monitor["deepest_days"] == 14.0
    assert monitor["via_import"] is True
    assert "liq_indicator_library" in monitor["deepest_at"]


def test_the_role_list_is_not_silently_empty():
    """A regex that stops matching start_eclipse.ps1 would make every check vacuous."""
    roles = standing_roles()
    assert len(roles) >= 15
    assert "tools.liq_anomaly_monitor" in roles
    assert "tools.s34_cascade_navigation_dashboard" in roles  # needle carries CLI args


def test_unbounded_aggregates_are_reported_not_hidden():
    """These have no time predicate at all, so their reach is the whole estate."""
    rows = {r["role"]: r for r in audit() if "deepest_ms" in r}
    assert rows["tools.liq_tip_forward"]["unbounded_aggregates"]
