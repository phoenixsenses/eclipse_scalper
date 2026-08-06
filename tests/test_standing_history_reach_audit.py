"""Pin how far back the standing roles reach, so a deeper window cannot land quietly.

SYSTEM_STATE section 265 measured this per-file and got 7 days. The real answer was
14, hidden two imports away, and the wrong number was used to argue an 836 GiB
segment was safe to delete. The point of this test is not the number -- it is that
raising the number must be a deliberate, visible act.
"""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# the parser must read the repo's dominant idiom, not just the tidy one
# ---------------------------------------------------------------------------


from tools.audit_standing_history_reach import _chain_ms  # noqa: E402


def test_reads_a_multiplied_chain_not_just_its_inner_pair():
    """`7 * 24 * 3600_000` used to report 1 day: the days pattern missed it entirely
    and the hours pattern matched the inner `24 * 3600_000`. That is how a constant
    literally named BTC_7D_LOOKBACK_MS was audited as 2 days."""
    assert _chain_ms("BTC_7D_LOOKBACK_MS = 7 * 24 * 3600_000") == [7 * DAY_MS]


@pytest.mark.parametrize("src,days", [
    ("x = 14 * 86_400_000", 14),
    ("x = 7 * 24 * 3_600_000", 7),
    ("x = 30 * 24 * 3600 * 1000", 30),
    ("x = 24 * 3_600_000", 1),
])
def test_equivalent_spellings_all_resolve_to_the_same_window(src, days):
    assert _chain_ms(src) == [days * DAY_MS]


def test_sub_hour_products_are_not_mistaken_for_windows():
    assert _chain_ms("FRESH_MS = 5 * 60_000") == []


def test_an_unbounded_aggregate_with_no_where_is_reported():
    """The previous probe skipped exactly this case: with no WHERE, splitting on it
    returned the whole line, so any aggregate selecting ts_ms looked bounded."""
    from tools.audit_standing_history_reach import scan_module
    import tempfile, pathlib, tools.audit_standing_history_reach as A

    with tempfile.TemporaryDirectory() as d:
        mod = pathlib.Path(d) / "probe.py"
        mod.write_text('q = "SELECT MIN(ts_ms) FROM mark_prices"\n', encoding="utf-8")
        original = A.REPO_ROOT
        A.REPO_ROOT = pathlib.Path(d)
        try:
            _windows, unbounded = scan_module("probe")
        finally:
            A.REPO_ROOT = original
    assert unbounded, "an aggregate with no time predicate must be reported"
