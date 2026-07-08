"""Focused tests: ami.storage.job_state + ami.storage.cli."""
from __future__ import annotations

import json

import pytest

from ami.storage import job_state as JS
from ami.storage import cli as CLI


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

def test_job_starts_at_planned():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    assert job.state == JS.PLANNED


def test_legal_transition_sequence():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.EXPORTING_PARTIAL)
    job.transition(JS.EXPORTED_UNVERIFIED)
    job.transition(JS.VERIFYING)
    job.transition(JS.VERIFIED_DISPOSABLE)
    assert job.is_terminal_verified


def test_illegal_transition_rejected():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    with pytest.raises(JS.IllegalJobTransitionError):
        job.transition(JS.VERIFIED_DISPOSABLE)  # cannot skip straight there


def test_verified_disposable_is_terminal_no_outgoing_transitions():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.EXPORTING_PARTIAL)
    job.transition(JS.EXPORTED_UNVERIFIED)
    job.transition(JS.VERIFYING)
    job.transition(JS.VERIFIED_DISPOSABLE)
    with pytest.raises(JS.IllegalJobTransitionError):
        job.transition(JS.PLANNED)  # no silent overwrite/reset of a verified archive


def test_abandoned_partial_detection():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.EXPORTING_PARTIAL)
    assert JS.detect_abandoned_partial(job, process_still_running=False) is True
    assert JS.detect_abandoned_partial(job, process_still_running=True) is False


def test_verified_job_never_abandoned():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.EXPORTING_PARTIAL)
    job.transition(JS.EXPORTED_UNVERIFIED)
    job.transition(JS.VERIFYING)
    job.transition(JS.VERIFIED_DISPOSABLE)
    assert JS.detect_abandoned_partial(job, process_still_running=False) is False


def test_restart_from_abandoned_produces_fresh_planned_job():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.EXPORTING_PARTIAL)
    restarted = JS.restart_from_abandoned(job)
    assert restarted.state == JS.PLANNED
    assert JS.ABANDONED_PARTIAL in restarted.history


def test_restart_rejects_non_partial_state():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    with pytest.raises(JS.IllegalJobTransitionError):
        JS.restart_from_abandoned(job)  # still PLANNED, nothing to restart


def test_same_watermark_is_idempotent():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    assert JS.same_watermark_is_idempotent(job, 100) is True
    assert JS.same_watermark_is_idempotent(job, 101) is False


def test_failed_job_can_retry_via_planned():
    job = JS.ArchiveJob(job_identity="j1", source_watermark_value=100)
    job.transition(JS.FAILED)
    job.transition(JS.PLANNED)
    assert job.state == JS.PLANNED


def test_all_job_states_declared():
    for s in (JS.PLANNED, JS.EXPORTING_PARTIAL, JS.EXPORTED_UNVERIFIED, JS.VERIFYING,
             JS.VERIFIED_DISPOSABLE, JS.FAILED, JS.ABANDONED_PARTIAL):
        assert s in JS.JOB_STATES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_has_expected_commands():
    parser = CLI.build_parser()
    # argparse subparsers choices are accessible via the subparsers action
    subparsers_action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    commands = set(subparsers_action.choices.keys())
    assert commands == {"policy-status", "plan", "disposable-dry-run", "verify", "read", "restore-slice",
                        "production-activation-rehearsal", "production-plan",
                        "production-archive-authorized", "production-verify",
                        "production-catalog-rebuild", "production-health"}


def test_cli_production_activation_rehearsal_accepts_no_partition_args():
    """The command takes NO table/symbol/month/root arguments -- there is
    nothing to parameterize, so it can never become a general
    production-enable command."""
    parser = CLI.build_parser()
    subparsers_action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    sub_parser = subparsers_action.choices["production-activation-rehearsal"]
    option_strings = {opt for action in sub_parser._actions for opt in action.option_strings}
    forbidden = {"--table", "--symbol", "--utc-year", "--utc-month", "--output-root",
                "--production-root", "--archive-version"}
    assert option_strings.isdisjoint(forbidden)


def test_cli_has_no_forbidden_commands():
    parser = CLI.build_parser()
    subparsers_action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    commands = set(subparsers_action.choices.keys())
    for forbidden in CLI.FORBIDDEN_COMMANDS:
        assert forbidden not in commands


def test_cli_unknown_command_fails():
    with pytest.raises(SystemExit):
        CLI.parse_args(["not-a-real-command"])


def test_cli_policy_status_output(capsys):
    rc = CLI.main(["policy-status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["production_activation"] == "DISABLED"
    assert out["scheduler"] == "DISABLED"
    assert out["purge"] == "DISABLED"
    assert out["vacuum"] == "DISABLED"
    assert set(out["allowlisted_tables"]) == {"agg_trades", "book_ticker", "mark_prices"}


def test_cli_plan_requires_all_args():
    with pytest.raises(SystemExit):
        CLI.parse_args(["plan", "--table", "mark_prices"])  # missing symbol/year/month


def test_cli_plan_rejects_unknown_table():
    with pytest.raises(SystemExit):
        CLI.parse_args(["plan", "--table", "not_a_table", "--symbol", "ETHUSDT",
                        "--utc-year", "2026", "--utc-month", "5"])


def test_cli_disposable_dry_run_requires_output_root():
    with pytest.raises(SystemExit):
        CLI.parse_args(["disposable-dry-run", "--table", "mark_prices", "--symbol", "ETHUSDT",
                        "--utc-year", "2026", "--utc-month", "5"])  # missing --output-root


def test_cli_main_reports_errors_as_json_with_exit_1(capsys):
    rc = CLI.main(["verify", "--parquet-path", "does_not_exist.parquet",
                  "--manifest-path", "does_not_exist.json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" in out


def test_cli_machine_readable_output_is_valid_json(capsys):
    CLI.main(["policy-status"])
    captured = capsys.readouterr().out
    json.loads(captured)  # must not raise


# ---------------------------------------------------------------------------
# Health report (pure aggregation)
# ---------------------------------------------------------------------------

def test_health_report_disables_all_automation_by_default():
    from ami.storage.health import build_health_report
    report = build_health_report(
        policy_version="v1", tooling_versions={"pyarrow": "21.0.0"},
        source_registry_tables=("mark_prices",), jobs=[],
        source_gap_blockers=(), repair_blockers=(), research_dependency_blockers=(),
        source_database_size_bytes=1000, wal_size_bytes=10, drive_free_bytes=2000)
    assert report.production_activation == "DISABLED"
    assert report.scheduler == "DISABLED"
    assert report.purge == "DISABLED"
    assert report.vacuum == "DISABLED"


def test_health_report_counts_jobs_by_state():
    from ami.storage.health import build_health_report
    job1 = JS.ArchiveJob(job_identity="j1", source_watermark_value=1)
    job1.transition(JS.EXPORTING_PARTIAL)
    job1.transition(JS.EXPORTED_UNVERIFIED)
    job1.transition(JS.VERIFYING)
    job1.transition(JS.VERIFIED_DISPOSABLE)
    job2 = JS.ArchiveJob(job_identity="j2", source_watermark_value=2)
    job2.transition(JS.FAILED)
    report = build_health_report(
        policy_version="v1", tooling_versions={}, source_registry_tables=("mark_prices",),
        jobs=[job1, job2], source_gap_blockers=(), repair_blockers=(),
        research_dependency_blockers=(), source_database_size_bytes=1, wal_size_bytes=1, drive_free_bytes=1)
    assert report.verified_disposable_count == 1
    assert report.failed_count == 1


def test_health_report_days_to_warning_computed():
    from ami.storage.health import build_health_report
    report = build_health_report(
        policy_version="v1", tooling_versions={}, source_registry_tables=(), jobs=[],
        source_gap_blockers=(), repair_blockers=(), research_dependency_blockers=(),
        source_database_size_bytes=1, wal_size_bytes=1, drive_free_bytes=1_000_000_000,
        growth_rate_bytes_per_day=100_000_000, warning_threshold_free_bytes=200_000_000)
    assert report.estimated_days_to_warning == pytest.approx(8.0)
