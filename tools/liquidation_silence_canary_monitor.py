"""Reusable, inspect-only evidence harness for liquidation-silence scheduler
canary runs (Gate 1 and any future repeat canary).

Round 2: addresses independent-review findings F-01..F-10 against the round-1
implementation (see SYSTEM_STATE.md Gate 1 acceptance entry and the round-1/
round-2 review transcripts). Each section below names the finding(s) it
closes.

  G1-MON-001 (PID-reuse blind spot) / F-01 (StartTime string comparison) /
      F-02 (identity truthiness-only) / F-14 (unanchored substring matching):
      continuity is decided from PID + UTC-normalized StartTime + an actual
      expected-identity comparison (module/script/token match), never PID
      alone and never a raw string StartTime comparison.
  G1-MON-002 (missing exact scoped-stop evidence) / F-10 (unvalidated
      StopEvent): build_stop_event() enforces invariants -- a
      STOP_VERIFIED_ABSENT outcome cannot be constructed without a
      successful StopVerification, and present chronology fields are
      order-checked; contradictory construction raises ValueError rather
      than producing a misleading record.
  G1-MON-003 (unrecoverable monitor source) / F-03 (dishonest git-commit
      provenance) / F-07 (tautological hash test, fixed in the test file):
      determine_source_provenance() separates "what commit is HEAD" from
      "is this exact source file contained in that commit", producing an
      explicit provenance_status rather than implying containment.
  G1-MON-004 (cadence arithmetic) / F-04 (no monotonicity check) / F-05
      (CLI used whole-second ts_utc): compute_cadence_summary() rejects
      non-monotonic/duplicate cycle timestamps instead of silently
      computing gaps from them, and parse_cycle_log() prefers the always-
      present full-precision cycle_started_epoch field over the whole-
      second ts_utc field, recording which precision tier was used.
  F-06 (CLI continuity always null): take_snapshot()/main() can load a
      baseline from a PID-metadata JSON (the same schema
      tools/liquidation_silence_scheduler.py's PID file sibling already
      writes) and emit a concrete continuity result; when no baseline is
      supplied the output is the explicit {"status": "NOT_EVALUATED",
      "reason_codes": ["BASELINE_NOT_PROVIDED"]}, never a bare null.
  F-08/F-09 (sample_artifact() untested / raised on directory paths):
      sample_artifact() is a fail-closed function returning a typed status
      (SAMPLED/MISSING/NOT_A_REGULAR_FILE/PERMISSION_DENIED/READ_FAILED/
      INVALID_ENCODING) for every expected I/O outcome instead of raising.

Round 3: addresses independent-review findings F-STOP-01 and F-BASELINE-01
(blockers) plus the F-READFAIL-01/F-MIXEDPREC-01/F-ROUND-01 low-coverage
gaps against the round-2 implementation.

  F-STOP-01 (StopEvent direct-construction bypass): every StopEvent
      invariant (F-10's original build_stop_event() rules) is now enforced
      in StopEvent.__post_init__ itself, so `StopEvent(...)` constructed
      directly -- bypassing build_stop_event() entirely -- can no longer
      produce a misleading STOP_VERIFIED_ABSENT record.
  F-BASELINE-01 (baseline loading collapsed every failure into one
      generic reason): load_pid_metadata_baseline() now returns a typed
      BaselineLoadResult (BASELINE_STATUS_OK/BASELINE_UNREADABLE/
      BASELINE_INVALID) with a specific reason_code distinguishing read
      failures (missing/directory/permission/encoding) from parse failures
      (empty/malformed JSON) from schema failures (missing fields/
      malformed pid) -- take_snapshot()'s output record carries this as an
      explicit "baseline" field alongside "continuity", never fabricating
      or silently collapsing the reason.
Round 4: addresses independent-review findings F-STOP-COMPAT-01 (the one
remaining MEDIUM from round 3 -- StopEvent's invariants were only enforced
for outcome=STOP_VERIFIED_ABSENT, leaving other outcomes able to carry
contradictory evidence, e.g. STOP_FAILED with an ALL_ABSENT verification)
and F-CYCLELOG-01 (a LOW robustness gap -- an invalid --scheduler-log-path,
e.g. a directory, raised an uncaught exception instead of a typed evidence
state).

  F-STOP-COMPAT-01: StopEvent.__post_init__ now validates every outcome
      against the explicit _STOP_OUTCOME_RULES compatibility matrix above
      the class (which verification values, and which timestamp fields,
      each outcome may or must carry), not merely STOP_VERIFIED_ABSENT.
  F-CYCLELOG-01: parse_cycle_log() is now a thin wrapper around
      load_cycle_log() (new), which returns a typed CycleLogLoadResult
      (CYCLE_LOG_OK/_MISSING/_NOT_A_REGULAR_FILE/_PERMISSION_DENIED/
      _READ_FAILED/_INVALID_ENCODING/_EMPTY/_NO_VALID_CYCLES) instead of
      letting an expected I/O failure (missing file, a directory, a
      permission/read failure, invalid UTF-8, an empty file, or a file
      with no parseable cycle records) raise -- take_snapshot()'s output
      now carries this as an explicit "cycle_log" field alongside
      "continuity" and "baseline", mirroring the same fail-closed,
      never-fabricate pattern F-BASELINE-01 established for the PID-
      metadata baseline. Individually malformed JSON *lines* within an
      otherwise-readable log are still skipped (unchanged partial-success
      policy from round 2's parse_cycle_log, now surfaced via a
      malformed_record_count in the typed result rather than silently).

F-READFAIL-01/F-MIXEDPREC-01/F-ROUND-01: no production-code change was
      required for these -- they were coverage gaps in the round-2 test
      suite (the generic OSError branch of sample_artifact(), mixed-
      precision cadence sequences, and the documented rounding boundaries
      were already correctly implemented but untested); round 3 adds the
      missing tests. The one related production fix is that
      _round_half_up_to_int() now explicitly rejects NaN/infinite/negative
      inputs (previously relied on math.floor's incidental, inconsistent
      exception behavior for NaN/infinity and silently rounded negative
      gaps) -- compute_cadence_summary() catches this and records None for
      an already-invalid (duplicate/regression) gap's rounded value rather
      than a misleading rounded number.

Strictly read-only / inspect-only: nothing in this module starts, stops,
signals, or reconfigures any process, PID file, health artifact, or
Scheduled Task. Process enumeration is dependency-injected (a
`ProcessLister` callable) so tests never need to spawn or touch a real
ECLIPSE runtime role; the default lister uses psutil (already a repository
dependency -- see tools/verify_data_layer.py, tools/telegram_bot.py,
tools/data_layer_probe.py) purely to *read* process tables.

This module close-reads (never imports control paths from)
tools/liquidation_silence_scheduler.py's own log record shape
(ts_utc/cycle_started_epoch/outcome/severity/status/detector_error/
cycle_duration_sec) and that scheduler's PID-metadata JSON sibling file
shape (pid/started_at/cmdline_sig/role) -- see parse_cycle_log and
load_pid_metadata_baseline below -- and never re-implements or
second-guesses that module's detection semantics or accepted thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tools.health_state import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_SCHEMA_VERSION = "liquidation_silence_canary_monitor_v2"
MONITOR_VERSION = "2.2.0"

# --------------------------------------------------------------------------
# Timestamps (F-01: UTC normalization, never a raw string comparison)
# --------------------------------------------------------------------------

TS_VALID = "VALID"
TS_MISSING = "MISSING"
TS_MALFORMED = "MALFORMED"
TS_NAIVE_REJECTED = "NAIVE_REJECTED"


def utc_now_iso() -> str:
    """Microsecond-precision UTC ISO-8601 -- deliberately NOT the coarser
    "%Y-%m-%dT%H:%M:%SZ" seconds-only format used elsewhere in this
    repository's operational logs: cadence math (G1-MON-004/F-05) needs
    sub-second precision, and truncating it here would silently reintroduce
    the same class of rounding ambiguity this module exists to close."""
    return datetime.now(timezone.utc).isoformat()


def normalize_utc_timestamp(raw: Optional[str]) -> Tuple[Optional[datetime], str]:
    """Parses `raw` into a timezone-aware UTC datetime, or reports exactly
    why it could not: MISSING (None/empty), MALFORMED (unparsable), or
    NAIVE_REJECTED (parsed but carries no offset/timezone -- never silently
    assumed to be UTC, per F-01: an ambiguous local time must not be able to
    masquerade as a confirmed instant). Only TS_VALID carries a usable
    datetime; every other status returns None."""
    if raw is None or raw == "":
        return None, TS_MISSING
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None, TS_MALFORMED
    if dt.tzinfo is None:
        return None, TS_NAIVE_REJECTED
    return dt.astimezone(timezone.utc), TS_VALID


def epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Process identity model
# --------------------------------------------------------------------------

# Continuity classifications. "CONTINUOUS" is the only status meaning
# "same process, same identity, observed twice" -- everything else means
# the caller must NOT treat the sample as proof of an unbroken run.
CONTINUITY_CONTINUOUS = "CONTINUOUS"
CONTINUITY_REPLACED_OR_RESTARTED = "REPLACED_OR_RESTARTED"
CONTINUITY_MISSING = "MISSING"
CONTINUITY_DUPLICATE = "DUPLICATE"
CONTINUITY_QUERY_FAILED = "QUERY_FAILED"
CONTINUITY_UNKNOWN_STARTTIME = "UNKNOWN_STARTTIME"
CONTINUITY_MALFORMED_STARTTIME = "MALFORMED_STARTTIME"
CONTINUITY_IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
CONTINUITY_NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class ProcessRecord:
    """One process observation, exactly as returned by a ProcessLister.
    start_time_utc is required to be an ISO-8601 string when known -- a
    lister that cannot determine it must set None, never fabricate "now".
    captured_at_utc (F-15) is the sample-taking instant itself, distinct
    from the process's own StartTime -- used only to detect an
    out-of-order re-sample (a "current" observation claiming to be earlier
    than the baseline it is being compared against)."""

    pid: int
    start_time_utc: Optional[str]
    command_line: Optional[str] = None
    captured_at_utc: Optional[str] = None


# A ProcessLister returns None to signal "the query itself failed" (e.g. the
# underlying OS/psutil call raised) -- distinct from an empty list, which
# means "the query succeeded and found zero matching processes". Callers
# must never collapse these two cases.
ProcessLister = Callable[[], Optional[List[ProcessRecord]]]


def default_process_lister() -> Optional[List[ProcessRecord]]:
    """Read-only process-table enumeration via psutil. Returns None (query
    failure), never an empty list, if the underlying OS query itself could
    not be performed -- e.g. psutil unavailable or iteration raised."""
    try:
        import psutil  # local import: keeps psutil optional for pure-mock test runs
    except ImportError:
        return None
    sampled_at = utc_now_iso()
    try:
        records: List[ProcessRecord] = []
        for proc in psutil.process_iter(["pid", "create_time", "cmdline"]):
            try:
                info = proc.info
                create_time = info.get("create_time")
                start_time_utc = (
                    datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat()
                    if create_time is not None
                    else None
                )
                cmdline = info.get("cmdline") or []
                records.append(
                    ProcessRecord(
                        pid=info.get("pid"),
                        start_time_utc=start_time_utc,
                        command_line=" ".join(cmdline) if cmdline else None,
                        captured_at_utc=sampled_at,
                    )
                )
            except Exception:
                # A single vanished/inaccessible process must not abort the
                # whole enumeration (psutil.NoSuchProcess/AccessDenied etc.).
                continue
        return records
    except Exception:
        return None


@dataclass(frozen=True)
class ProcessQueryResult:
    query_failed: bool
    matches: List[ProcessRecord]

    @property
    def count(self) -> Optional[int]:
        """None (not 0, not 1) when the query itself failed -- there is no
        code path here where "the query failed" and "exactly zero matches"
        can be confused, because they are different states of this
        dataclass, not the same nullable integer (the exact PowerShell
        `(Get-CimInstance ...).Count` ambiguity this module exists to
        prevent structurally)."""
        if self.query_failed:
            return None
        return len(self.matches)


def find_by_pid(records: Optional[List[ProcessRecord]], pid: int) -> ProcessQueryResult:
    if records is None:
        return ProcessQueryResult(query_failed=True, matches=[])
    matches = [r for r in records if r.pid == pid]
    return ProcessQueryResult(query_failed=False, matches=matches)


def find_by_command_needle(
    records: Optional[List[ProcessRecord]], needle: str
) -> ProcessQueryResult:
    """Retained for simple lookups (e.g. live-executor OFF-state checks,
    where an exact ExpectedIdentity model is unnecessary overhead) -- NOT
    used by classify_continuity's identity check, which uses the anchored
    token-based identity_matches() below instead (F-02/F-14: closes the
    unanchored-substring false-positive risk for continuity decisions)."""
    if records is None:
        return ProcessQueryResult(query_failed=True, matches=[])
    matches = [r for r in records if r.command_line and needle in r.command_line]
    return ProcessQueryResult(query_failed=False, matches=matches)


def find_by_expected_identity(
    records: Optional[List[ProcessRecord]], expected: "ExpectedIdentity"
) -> "ProcessQueryResult":
    """Anchored (token-based, via identity_matches) discovery query --
    finds every process whose command line positively matches `expected`,
    regardless of PID. Exposed as a standalone utility; the default
    continuity-sampling path below uses the broader
    find_role_candidates(), which OR-combines this with a PID match so
    that BOTH "same PID now claiming a different identity" and "a
    different PID now claiming the same identity" remain independently
    detectable."""
    if records is None:
        return ProcessQueryResult(query_failed=True, matches=[])
    matches = [r for r in records if identity_matches(r.command_line, expected)[0] == IDENTITY_MATCHED]
    return ProcessQueryResult(query_failed=False, matches=matches)


def find_role_candidates(
    records: Optional[List[ProcessRecord]],
    baseline_pid: int,
    expected_identity: Optional["ExpectedIdentity"],
) -> "ProcessQueryResult":
    """Discovery query used by sample_role_continuity's default path: a
    candidate is any process matching the baseline PID OR matching the
    anchored expected identity (deduplicated by PID). This is
    deliberately broader than either check alone -- searching only by the
    old PID can never surface a PID replacement (the old PID is gone by
    definition), and searching only by identity can never surface "same
    PID, but it now looks like a different program" (F-02's
    IDENTITY_MISMATCH case). Every candidate found here still passes
    through classify_continuity's own PID/StartTime/identity checks --
    this function only decides what counts as worth inspecting at all."""
    if records is None:
        return ProcessQueryResult(query_failed=True, matches=[])
    matches: List[ProcessRecord] = []
    seen_pids = set()
    for r in records:
        is_pid_match = r.pid == baseline_pid
        is_identity_match = expected_identity is not None and identity_matches(r.command_line, expected_identity)[0] == IDENTITY_MATCHED
        if (is_pid_match or is_identity_match) and r.pid not in seen_pids:
            seen_pids.add(r.pid)
            matches.append(r)
    return ProcessQueryResult(query_failed=False, matches=matches)


# --------------------------------------------------------------------------
# Identity matching (F-02, F-14: anchored token comparison, not substring)
# --------------------------------------------------------------------------

IDENTITY_MATCHED = "MATCHED_EXPECTED_IDENTITY"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
IDENTITY_UNAVAILABLE = "CMDLINE_UNAVAILABLE"
IDENTITY_NOT_REQUESTED = "IDENTITY_NOT_REQUESTED"


@dataclass(frozen=True)
class ExpectedIdentity:
    """Describes what a role's command line MUST contain to be believed --
    matched as whole tokens, never substrings, so
    `tools.liquidation_silence_scheduler` cannot match a stray
    `tools.liquidation_silence_scheduler.py.bak` argument or an unrelated
    process that merely mentions the string somewhere (F-14)."""

    role: str
    module_name: Optional[str] = None  # exact "-m" module token, e.g. "tools.liquidation_silence_scheduler"
    script_basename: Optional[str] = None  # exact script filename, e.g. "heartbeat_watchdog.py"
    required_tokens: Tuple[str, ...] = ()


def _tokenize_command_line(command_line: str) -> List[str]:
    normalized = command_line.replace("\\", "/")
    try:
        return shlex.split(normalized, posix=True)
    except ValueError:
        # Unbalanced quotes etc. -- fall back to whitespace splitting rather
        # than raising; identity matching degrades to UNAVAILABLE/MISMATCH,
        # it never crashes the caller.
        return normalized.split()


def identity_matches(
    command_line: Optional[str], expected: Optional[ExpectedIdentity]
) -> Tuple[str, List[str]]:
    """Returns (identity_status, reason_codes). Matching is anchored: a
    module/script/token must appear as an EXACT whitespace-delimited token
    (case-insensitive, path-separator-normalized), never as a substring of
    a longer token -- this is what makes a false positive like
    `...scheduler.py.bak` or `some_other_tools.liquidation_silence_scheduler_helper`
    structurally impossible."""
    if expected is None:
        return IDENTITY_NOT_REQUESTED, []
    if not command_line:
        return IDENTITY_UNAVAILABLE, ["COMMAND_LINE_UNAVAILABLE"]

    tokens = [t.lower() for t in _tokenize_command_line(command_line)]
    has_discriminator = bool(expected.module_name or expected.script_basename or expected.required_tokens)
    if not has_discriminator:
        return IDENTITY_UNAVAILABLE, ["NO_EXPECTED_IDENTITY_FIELDS"]

    ok = True
    if expected.module_name is not None and expected.module_name.lower() not in tokens:
        ok = False
    if expected.script_basename is not None:
        script_match = any(Path(t).name.lower() == expected.script_basename.lower() for t in tokens)
        if not script_match:
            ok = False
    for required in expected.required_tokens:
        if required.lower() not in tokens:
            ok = False

    if ok:
        return IDENTITY_MATCHED, []
    return IDENTITY_MISMATCH, ["IDENTITY_TOKENS_NOT_MATCHED"]


# --------------------------------------------------------------------------
# Continuity classification (G1-MON-001 / F-01 / F-02 / F-15)
# --------------------------------------------------------------------------


@dataclass
class ProcessIdentitySnapshot:
    sampled_at_utc: str
    role: str
    process_count: Optional[int]
    pid: Optional[int]
    start_time_utc: Optional[str]
    start_time_status: str
    identity_status: str
    command_line: Optional[str]
    continuity_status: str
    reason_codes: List[str] = field(default_factory=list)


def classify_continuity(
    baseline: ProcessRecord,
    query: ProcessQueryResult,
    *,
    expect_singular: bool = True,
    expected_identity: Optional[ExpectedIdentity] = None,
) -> ProcessIdentitySnapshot:
    """Core PID+StartTime+identity continuity model.

    Continuity (CONTINUOUS) requires ALL of: exactly one matching process,
    the same PID as baseline, a UTC-normalized StartTime equal to
    baseline's (F-01 -- compared as datetimes, never as raw strings), and
    (when an ExpectedIdentity was supplied) a positive identity match
    (F-02). A PID match alone is NEVER sufficient."""
    sampled_at = utc_now_iso()
    reason_codes: List[str] = []

    def _snapshot(**overrides: Any) -> ProcessIdentitySnapshot:
        base = dict(
            sampled_at_utc=sampled_at,
            role="",
            process_count=None,
            pid=None,
            start_time_utc=None,
            start_time_status=TS_MISSING,
            identity_status=IDENTITY_UNAVAILABLE,
            command_line=None,
            continuity_status=CONTINUITY_MISSING,
            reason_codes=reason_codes,
        )
        base.update(overrides)
        return ProcessIdentitySnapshot(**base)

    if query.query_failed:
        return _snapshot(continuity_status=CONTINUITY_QUERY_FAILED, reason_codes=["PROCESS_QUERY_FAILED"])

    count = len(query.matches)
    if count == 0:
        return _snapshot(process_count=0, continuity_status=CONTINUITY_MISSING, reason_codes=["NO_MATCHING_PROCESS"])

    if expect_singular and count > 1:
        return _snapshot(
            process_count=count,
            continuity_status=CONTINUITY_DUPLICATE,
            reason_codes=["MULTIPLE_MATCHING_PROCESSES"],
        )

    current = query.matches[0]

    # Sample-ordering sanity check (F-15): does not by itself change the
    # continuity verdict, but is surfaced as a reason code so an obviously
    # out-of-order comparison is never silently trusted.
    if baseline.captured_at_utc and current.captured_at_utc:
        baseline_capture, _ = normalize_utc_timestamp(baseline.captured_at_utc)
        current_capture, _ = normalize_utc_timestamp(current.captured_at_utc)
        if baseline_capture is not None and current_capture is not None and current_capture < baseline_capture:
            reason_codes.append("SAMPLE_TIMESTAMP_REGRESSION")

    if current.pid != baseline.pid:
        reason_codes.append("PID_CHANGED")
        return _snapshot(
            process_count=count,
            pid=current.pid,
            start_time_utc=current.start_time_utc,
            command_line=current.command_line,
            continuity_status=CONTINUITY_REPLACED_OR_RESTARTED,
        )

    baseline_dt, baseline_status = normalize_utc_timestamp(baseline.start_time_utc)
    current_dt, current_status = normalize_utc_timestamp(current.start_time_utc)

    if current_status == TS_MISSING or baseline_status == TS_MISSING:
        reason_codes.append("STARTTIME_UNAVAILABLE")
        return _snapshot(
            process_count=count,
            pid=current.pid,
            start_time_utc=current.start_time_utc,
            start_time_status=current_status,
            command_line=current.command_line,
            continuity_status=CONTINUITY_UNKNOWN_STARTTIME,
        )

    if current_status != TS_VALID or baseline_status != TS_VALID:
        bad = current_status if current_status != TS_VALID else baseline_status
        reason_codes.append(f"STARTTIME_{bad}")
        return _snapshot(
            process_count=count,
            pid=current.pid,
            start_time_utc=current.start_time_utc,
            start_time_status=current_status,
            command_line=current.command_line,
            continuity_status=CONTINUITY_MALFORMED_STARTTIME,
        )

    if baseline_dt != current_dt:
        reason_codes.append("STARTTIME_CHANGED")
        return _snapshot(
            process_count=count,
            pid=current.pid,
            start_time_utc=current.start_time_utc,
            start_time_status=current_status,
            command_line=current.command_line,
            continuity_status=CONTINUITY_REPLACED_OR_RESTARTED,
        )

    identity_status, identity_reasons = identity_matches(current.command_line, expected_identity)
    reason_codes.extend(identity_reasons)
    if identity_status == IDENTITY_MISMATCH:
        return _snapshot(
            process_count=count,
            pid=current.pid,
            start_time_utc=current.start_time_utc,
            start_time_status=current_status,
            identity_status=identity_status,
            command_line=current.command_line,
            continuity_status=CONTINUITY_IDENTITY_MISMATCH,
        )

    return _snapshot(
        process_count=count,
        pid=current.pid,
        start_time_utc=current.start_time_utc,
        start_time_status=current_status,
        identity_status=identity_status,
        command_line=current.command_line,
        continuity_status=CONTINUITY_CONTINUOUS,
    )


def sample_role_continuity(
    role: str,
    baseline: ProcessRecord,
    lister: ProcessLister,
    *,
    query: Optional[ProcessQueryResult] = None,
    expect_singular: bool = True,
    expected_identity: Optional[ExpectedIdentity] = None,
) -> ProcessIdentitySnapshot:
    """Samples one role (scheduler, watchdog, or any protected runtime
    role) using the same PID+StartTime+identity continuity principle --
    there is no separate PID-only fallback anywhere in this module. If
    `query` is not supplied, re-discovers candidates via
    find_role_candidates (PID-match OR identity-match), so both a PID
    replacement and a same-PID-different-identity case remain reachable
    through this default path."""
    if query is None:
        query = find_role_candidates(lister(), baseline.pid, expected_identity)
    snapshot = classify_continuity(
        baseline, query, expect_singular=expect_singular, expected_identity=expected_identity
    )
    snapshot.role = role
    return snapshot


def not_evaluated_continuity(reason: str) -> Dict[str, Any]:
    """The explicit, non-ambiguous stand-in for "no baseline was supplied"
    (F-06) -- never a bare `null`."""
    return {
        "sampled_at_utc": utc_now_iso(),
        "continuity_status": CONTINUITY_NOT_EVALUATED,
        "reason_codes": [reason],
    }


# --------------------------------------------------------------------------
# Live-executor OFF-state verification
# --------------------------------------------------------------------------


@dataclass
class LiveExecutorCheck:
    sampled_at_utc: str
    marker_path: str
    marker_value: Optional[str]
    process_count: Optional[int]
    off_state_ok: bool
    reason_codes: List[str] = field(default_factory=list)


def check_live_executor(
    marker_path: Path,
    lister: ProcessLister,
    cmdline_needle: str,
) -> LiveExecutorCheck:
    """OFF means BOTH the marker file reads "0" AND no process matches the
    live-executor's command line -- a marker of "0" with a live matching
    process actually running is a mismatch, not a pass."""
    sampled_at = utc_now_iso()
    reason_codes: List[str] = []
    try:
        marker_value = marker_path.read_text(encoding="utf-8").strip()
    except OSError:
        marker_value = None
        reason_codes.append("MARKER_UNREADABLE")

    query = find_by_command_needle(lister(), cmdline_needle)
    if query.query_failed:
        reason_codes.append("PROCESS_QUERY_FAILED")
        return LiveExecutorCheck(
            sampled_at_utc=sampled_at,
            marker_path=str(marker_path),
            marker_value=marker_value,
            process_count=None,
            off_state_ok=False,
            reason_codes=reason_codes,
        )

    process_count = len(query.matches)
    marker_is_off = marker_value == "0"
    if not marker_is_off:
        reason_codes.append("MARKER_NOT_ZERO")
    if process_count > 0:
        reason_codes.append("LIVE_EXECUTOR_PROCESS_FOUND_DESPITE_OFF_MARKER")

    return LiveExecutorCheck(
        sampled_at_utc=sampled_at,
        marker_path=str(marker_path),
        marker_value=marker_value,
        process_count=process_count,
        off_state_ok=marker_is_off and process_count == 0,
        reason_codes=reason_codes,
    )


# --------------------------------------------------------------------------
# Artifact sampling (F-08 / F-09: typed, fail-closed disk-read states)
# --------------------------------------------------------------------------

ARTIFACT_STATUS_SAMPLED = "SAMPLED"
ARTIFACT_STATUS_MISSING = "MISSING"
ARTIFACT_STATUS_NOT_A_REGULAR_FILE = "NOT_A_REGULAR_FILE"
ARTIFACT_STATUS_PERMISSION_DENIED = "PERMISSION_DENIED"
ARTIFACT_STATUS_READ_FAILED = "READ_FAILED"
ARTIFACT_STATUS_INVALID_ENCODING = "INVALID_ENCODING"

SEMANTIC_TS_PRESENT = "PRESENT"
SEMANTIC_TS_MISSING = "MISSING"
SEMANTIC_TS_INVALID = "INVALID"
SEMANTIC_TS_MALFORMED_JSON = "MALFORMED_JSON"
SEMANTIC_TS_NOT_REQUESTED = "NOT_REQUESTED"


@dataclass(frozen=True)
class ArtifactSample:
    path: str
    status: str
    exists: bool
    size_bytes: Optional[int]
    mtime_utc: Optional[str]
    sha256: Optional[str]
    semantic_ts_utc: Optional[str]
    semantic_ts_status: str
    error: Optional[str] = None


def sample_artifact(path: Path, *, semantic_ts_field: Optional[str] = None) -> ArtifactSample:
    """Fail-closed: every expected I/O outcome (missing file, a directory
    supplied instead of a file, a permission/read failure, invalid UTF-8,
    malformed JSON, a missing or unparsable semantic timestamp field)
    produces a distinct typed `status`/`semantic_ts_status` -- never an
    unhandled exception, never a fabricated hash/size/timestamp (F-08/F-09).
    Only a genuinely unexpected programmer error is allowed to propagate."""
    not_requested = semantic_ts_field is None
    default_semantic_status = SEMANTIC_TS_NOT_REQUESTED if not_requested else SEMANTIC_TS_MISSING

    if not path.exists():
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_MISSING,
            exists=False,
            size_bytes=None,
            mtime_utc=None,
            sha256=None,
            semantic_ts_utc=None,
            semantic_ts_status=default_semantic_status,
        )
    if path.is_dir():
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_NOT_A_REGULAR_FILE,
            exists=True,
            size_bytes=None,
            mtime_utc=None,
            sha256=None,
            semantic_ts_utc=None,
            semantic_ts_status=default_semantic_status,
            error="path is a directory, not a regular file",
        )

    try:
        raw = path.read_bytes()
        stat_result = path.stat()
    except PermissionError as exc:
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_PERMISSION_DENIED,
            exists=True,
            size_bytes=None,
            mtime_utc=None,
            sha256=None,
            semantic_ts_utc=None,
            semantic_ts_status=default_semantic_status,
            error=str(exc),
        )
    except OSError as exc:
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_READ_FAILED,
            exists=True,
            size_bytes=None,
            mtime_utc=None,
            sha256=None,
            semantic_ts_utc=None,
            semantic_ts_status=default_semantic_status,
            error=str(exc),
        )

    sha256 = hashlib.sha256(raw).hexdigest()
    mtime_utc = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat()

    if not_requested:
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_SAMPLED,
            exists=True,
            size_bytes=len(raw),
            mtime_utc=mtime_utc,
            sha256=sha256,
            semantic_ts_utc=None,
            semantic_ts_status=SEMANTIC_TS_NOT_REQUESTED,
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_INVALID_ENCODING,
            exists=True,
            size_bytes=len(raw),
            mtime_utc=mtime_utc,
            sha256=sha256,
            semantic_ts_utc=None,
            semantic_ts_status=SEMANTIC_TS_INVALID,
            error=str(exc),
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ArtifactSample(
            path=str(path),
            status=ARTIFACT_STATUS_SAMPLED,
            exists=True,
            size_bytes=len(raw),
            mtime_utc=mtime_utc,
            sha256=sha256,
            semantic_ts_utc=None,
            semantic_ts_status=SEMANTIC_TS_MALFORMED_JSON,
            error=str(exc),
        )

    value = payload.get(semantic_ts_field) if isinstance(payload, dict) else None
    if value is None:
        semantic_status = SEMANTIC_TS_MISSING
        semantic_ts = None
    else:
        dt, _ts_status = normalize_utc_timestamp(value)
        if dt is None:
            semantic_status = SEMANTIC_TS_INVALID
            semantic_ts = None
        else:
            semantic_status = SEMANTIC_TS_PRESENT
            semantic_ts = dt.isoformat()

    return ArtifactSample(
        path=str(path),
        status=ARTIFACT_STATUS_SAMPLED,
        exists=True,
        size_bytes=len(raw),
        mtime_utc=mtime_utc,
        sha256=sha256,
        semantic_ts_utc=semantic_ts,
        semantic_ts_status=semantic_status,
    )


FRESHNESS_FRESH = "FRESH_HASH_CHANGED_TIMESTAMP_ADVANCED"
FRESHNESS_UNCHANGED = "UNCHANGED_SINCE_PRIOR_SAMPLE"
FRESHNESS_SUSPECT_STALE_REUSE = "SUSPECT_HASH_CHANGED_TIMESTAMP_NOT_ADVANCED"
FRESHNESS_MONOTONICITY_VIOLATION = "MONOTONICITY_VIOLATION_TIMESTAMP_WENT_BACKWARD"
FRESHNESS_NO_PRIOR_SAMPLE = "NO_PRIOR_SAMPLE"
FRESHNESS_SAMPLE_UNAVAILABLE = "SAMPLE_UNAVAILABLE"


def compare_artifact_freshness(prev: Optional[ArtifactSample], curr: ArtifactSample) -> str:
    """Hash change alone is not accepted as proof of a genuine new
    evaluation: if the content hash changed but the artifact's own
    semantic timestamp field did not advance, that is flagged as a
    suspect stale-reuse pattern, not silently reported as fresh."""
    if prev is None or prev.status != ARTIFACT_STATUS_SAMPLED:
        return FRESHNESS_NO_PRIOR_SAMPLE
    if curr.status != ARTIFACT_STATUS_SAMPLED:
        return FRESHNESS_SAMPLE_UNAVAILABLE
    if curr.sha256 == prev.sha256:
        return FRESHNESS_UNCHANGED

    prev_ts, _ = normalize_utc_timestamp(prev.semantic_ts_utc)
    curr_ts, _ = normalize_utc_timestamp(curr.semantic_ts_utc)
    if prev_ts is not None and curr_ts is not None:
        if curr_ts < prev_ts:
            return FRESHNESS_MONOTONICITY_VIOLATION
        if curr_ts == prev_ts:
            return FRESHNESS_SUSPECT_STALE_REUSE
    return FRESHNESS_FRESH


# --------------------------------------------------------------------------
# Cycle log parsing (F-05: full-precision timestamp preference)
# --------------------------------------------------------------------------

TIMESTAMP_SOURCE_EPOCH_FULL = "EPOCH_FULL_PRECISION"
TIMESTAMP_SOURCE_ISO_FULL = "ISO_FULL_PRECISION"
TIMESTAMP_SOURCE_ISO_DEGRADED = "ISO_WHOLE_SECOND_DEGRADED"
TIMESTAMP_SOURCE_INVALID = "INVALID"


def _select_cycle_timestamp(record: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Precedence, highest quality first: (1) `cycle_started_epoch` --
    always present in tools/liquidation_silence_scheduler.py's own log
    format, full float precision; (2) `evaluated_at_utc` if it happens to
    carry sub-second precision (a documented alternate producer shape);
    (3) whole-second `ts_utc` only as an explicitly degraded fallback.
    Never silently represents (3) as equivalent to (1)/(2) -- the caller
    always receives which tier was actually used."""
    epoch = record.get("cycle_started_epoch")
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        try:
            return epoch_to_iso(float(epoch)), TIMESTAMP_SOURCE_EPOCH_FULL
        except (OverflowError, OSError, ValueError):
            pass

    evaluated = record.get("evaluated_at_utc")
    if evaluated:
        dt, status = normalize_utc_timestamp(evaluated)
        if dt is not None and dt.microsecond != 0:
            return dt.isoformat(), TIMESTAMP_SOURCE_ISO_FULL

    ts_utc = record.get("ts_utc")
    if ts_utc:
        dt, status = normalize_utc_timestamp(ts_utc)
        if dt is not None:
            return dt.isoformat(), TIMESTAMP_SOURCE_ISO_DEGRADED

    return None, TIMESTAMP_SOURCE_INVALID


@dataclass
class CycleObservation:
    cycle_number: int
    evaluated_at_utc: Optional[str]
    timestamp_source: str
    duration_sec: Optional[float]
    outcome: Optional[str]
    severity: Optional[str]
    status: Optional[str]
    reason_codes: List[str] = field(default_factory=list)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    freshness_result: Optional[str] = None


# F-CYCLELOG-01: typed, fail-closed cycle-log loading vocabulary --
# mirrors sample_artifact() (F-08/F-09) and load_pid_metadata_baseline()
# (F-BASELINE-01)'s "every expected I/O outcome gets a distinct typed
# status, never an unhandled exception" contract.
CYCLE_LOG_STATUS_OK = "CYCLE_LOG_OK"
CYCLE_LOG_STATUS_MISSING = "CYCLE_LOG_MISSING"
CYCLE_LOG_STATUS_NOT_A_REGULAR_FILE = "CYCLE_LOG_NOT_A_REGULAR_FILE"
CYCLE_LOG_STATUS_PERMISSION_DENIED = "CYCLE_LOG_PERMISSION_DENIED"
CYCLE_LOG_STATUS_READ_FAILED = "CYCLE_LOG_READ_FAILED"
CYCLE_LOG_STATUS_INVALID_ENCODING = "CYCLE_LOG_INVALID_ENCODING"
CYCLE_LOG_STATUS_EMPTY = "CYCLE_LOG_EMPTY"
CYCLE_LOG_STATUS_NO_VALID_CYCLES = "CYCLE_LOG_NO_VALID_CYCLES"


@dataclass(frozen=True)
class CycleLogLoadResult:
    status: str
    reason_code: str
    path: str
    observations: List[CycleObservation]
    record_count: int
    malformed_record_count: int
    error: Optional[str] = None


def load_cycle_log(path: Path) -> CycleLogLoadResult:
    """Fail-closed loader for tools/liquidation_silence_scheduler.py's own
    per-cycle JSONL log (F-CYCLELOG-01). Every expected I/O outcome (a
    missing file, a directory supplied instead of a file, a permission/
    read failure, invalid UTF-8, an empty file, or a file whose every line
    is malformed JSON) produces a distinct typed status/reason_code instead
    of an unhandled exception or a fabricated empty-but-successful result.

    Policy for a MIX of malformed and valid JSON lines within an otherwise
    readable log (explicitly not left ambiguous, per this round's
    requirement): PARTIAL SUCCESS WITH WARNINGS, not whole-file rejection
    -- this is the unchanged round-2 policy ("a malformed line is skipped,
    never silently coerced into a fabricated SUCCESS"), now surfaced via
    malformed_record_count instead of being silently invisible. Only a log
    that is fully unreadable, or that contains zero salvageable cycles at
    all, gets a non-OK top-level status."""
    if not path.exists():
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_MISSING,
            reason_code="CYCLE_LOG_MISSING",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
        )
    if path.is_dir():
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_NOT_A_REGULAR_FILE,
            reason_code="CYCLE_LOG_NOT_A_REGULAR_FILE",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
            error="path is a directory, not a regular file",
        )

    try:
        raw_bytes = path.read_bytes()
    except PermissionError as exc:
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_PERMISSION_DENIED,
            reason_code="CYCLE_LOG_PERMISSION_DENIED",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
            error=str(exc),
        )
    except OSError as exc:
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_READ_FAILED,
            reason_code="CYCLE_LOG_READ_FAILED",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
            error=str(exc),
        )

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_INVALID_ENCODING,
            reason_code="CYCLE_LOG_INVALID_ENCODING",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
            error=str(exc),
        )

    if raw_text.strip() == "":
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_EMPTY,
            reason_code="CYCLE_LOG_EMPTY",
            path=str(path),
            observations=[],
            record_count=0,
            malformed_record_count=0,
        )

    observations: List[CycleObservation] = []
    record_count = 0
    malformed_record_count = 0
    for cycle_number, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed_record_count += 1
            continue
        reason_codes = list(record.get("reason_codes") or [])
        if record.get("detector_error"):
            reason_codes.append("DETECTOR_ERROR_PRESENT")
        if record.get("outcome") == "WRAPPER_EXCEPTION":
            reason_codes.append("WRAPPER_EXCEPTION")
        evaluated_at_utc, timestamp_source = _select_cycle_timestamp(record)
        observations.append(
            CycleObservation(
                cycle_number=cycle_number,
                evaluated_at_utc=evaluated_at_utc,
                timestamp_source=timestamp_source,
                duration_sec=record.get("cycle_duration_sec"),
                outcome=record.get("outcome"),
                severity=record.get("severity"),
                status=record.get("status"),
                reason_codes=reason_codes,
            )
        )

    if not observations:
        return CycleLogLoadResult(
            status=CYCLE_LOG_STATUS_NO_VALID_CYCLES,
            reason_code="CYCLE_LOG_NO_VALID_CYCLES",
            path=str(path),
            observations=[],
            record_count=record_count,
            malformed_record_count=malformed_record_count,
        )

    return CycleLogLoadResult(
        status=CYCLE_LOG_STATUS_OK,
        reason_code="OK",
        path=str(path),
        observations=observations,
        record_count=record_count,
        malformed_record_count=malformed_record_count,
    )


def parse_cycle_log(path: Path) -> List[CycleObservation]:
    """Reads tools/liquidation_silence_scheduler.py's own per-cycle JSONL
    record shape verbatim -- this function does not reinterpret or
    second-guess that module's own outcome semantics, only reshapes each
    line into a CycleObservation for cadence/freshness analysis. A
    malformed line is skipped, never silently coerced into a fabricated
    SUCCESS.

    Backward-compatible wrapper around load_cycle_log() (F-CYCLELOG-01):
    always returns just the observations list, exactly as every existing
    caller/test expects (including returning [] for a missing file, same
    as before this round). Callers that need the typed failure state --
    currently only take_snapshot() -- use load_cycle_log() directly."""
    return load_cycle_log(path).observations
    return observations


# --------------------------------------------------------------------------
# Cadence math (G1-MON-004 / F-04: monotonicity enforced, no silent sort)
# --------------------------------------------------------------------------

CADENCE_REASON_DUPLICATE = "DUPLICATE_CYCLE_TIMESTAMP"
CADENCE_REASON_REGRESSION = "CYCLE_TIMESTAMP_REGRESSION"
CADENCE_REASON_INVALID = "INVALID_CYCLE_TIMESTAMP"

CADENCE_PRECISION_FULL = "FULL_PRECISION"
CADENCE_PRECISION_DEGRADED = "DEGRADED_WHOLE_SECOND"
CADENCE_PRECISION_MIXED = "MIXED"
CADENCE_PRECISION_UNKNOWN = "UNKNOWN"


@dataclass
class CadenceSummary:
    cycle_count: int
    gap_count: int
    exact_gaps_sec: List[float]
    rounded_gaps_sec: List[Optional[int]]
    min_gap_sec: Optional[float]
    max_gap_sec: Optional[float]
    mean_gap_sec: Optional[float]
    median_gap_sec: Optional[float]
    valid: bool
    precision_status: str
    reason_codes: List[str] = field(default_factory=list)
    invalid_gap_indices: List[int] = field(default_factory=list)


def _round_half_up_to_int(value: float) -> int:
    """Whole-second rounding method, explicitly specified: standard
    round-half-up (toward positive infinity for the .5 case) on the exact
    decimal-second float, not Python's built-in round() (which is
    round-half-to-even and would silently change behavior exactly on a .5
    boundary).

    F-ROUND-01: NaN, +/-infinity, and negative values are explicitly
    rejected here rather than relying on math.floor's incidental (and
    inconsistent -- ValueError for NaN, OverflowError for infinity)
    behavior for those inputs; a negative inter-cycle gap is never a
    legitimate whole-second duration, so it is rejected before rounding is
    even attempted rather than being silently rounded to a negative int."""
    if math.isnan(value):
        raise ValueError("cannot round-half-up a NaN value")
    if math.isinf(value):
        raise ValueError("cannot round-half-up an infinite value")
    if value < 0:
        raise ValueError(f"cannot round-half-up a negative gap: {value!r}")
    return int(math.floor(value + 0.5))


def _precision_status(cycles: Sequence[CycleObservation]) -> str:
    sources = {c.timestamp_source for c in cycles if c.evaluated_at_utc is not None}
    if not sources:
        return CADENCE_PRECISION_UNKNOWN
    full = {TIMESTAMP_SOURCE_EPOCH_FULL, TIMESTAMP_SOURCE_ISO_FULL}
    if sources <= full:
        return CADENCE_PRECISION_FULL
    if sources == {TIMESTAMP_SOURCE_ISO_DEGRADED}:
        return CADENCE_PRECISION_DEGRADED
    return CADENCE_PRECISION_MIXED


def compute_cadence_summary(cycles: Sequence[CycleObservation]) -> CadenceSummary:
    """N cycles imply exactly N-1 inter-cycle gaps, computed here as
    len(cycles) - 1 and nowhere else. Before computing statistics, cycle
    timestamps are validated for strict monotonic increase (F-04): a
    duplicate or decreasing adjacent pair is recorded via reason_codes and
    invalid_gap_indices, `valid` is set False, and min/max/mean/median are
    withheld (None) -- exact_gaps_sec is still populated (in original
    order, never silently sorted) so the raw evidence remains available
    for diagnosis, but summary statistics are never presented as trustworthy
    over invalid input."""
    cycle_count = len(cycles)
    timestamps: List[Optional[datetime]] = []
    for cyc in cycles:
        dt, _status = normalize_utc_timestamp(cyc.evaluated_at_utc)
        timestamps.append(dt)

    valid_timestamps = [t for t in timestamps if t is not None]
    reason_codes: List[str] = []
    invalid_indices: List[int] = []

    exact_gaps: List[float] = []
    if len(valid_timestamps) >= 2:
        # Gaps are computed strictly between adjacent VALID timestamps in
        # original order -- a cycle with an unparsable timestamp is simply
        # excluded from the sequence (already flagged via CADENCE_REASON_INVALID
        # if it was expected to be present), never treated as a zero-length gap.
        ordered = [t for t in timestamps if t is not None]
        for i in range(1, len(ordered)):
            prev_t, curr_t = ordered[i - 1], ordered[i]
            gap = (curr_t - prev_t).total_seconds()
            exact_gaps.append(gap)
            if gap == 0:
                reason_codes.append(CADENCE_REASON_DUPLICATE)
                invalid_indices.append(i - 1)
            elif gap < 0:
                reason_codes.append(CADENCE_REASON_REGRESSION)
                invalid_indices.append(i - 1)

    if any(t is None for t in timestamps) and cycle_count > 0:
        reason_codes.append(CADENCE_REASON_INVALID)

    gap_count = len(exact_gaps)
    rounded: List[Optional[int]] = []
    for g in exact_gaps:
        # A gap already flagged invalid (duplicate/regression, i.e. <= 0 in
        # the regression case) is never a legitimate whole-second duration
        # -- _round_half_up_to_int rejects it (F-ROUND-01) rather than
        # producing a misleading rounded value; recorded as None so the
        # list length still reconciles 1:1 with exact_gaps_sec.
        try:
            rounded.append(_round_half_up_to_int(g))
        except ValueError:
            rounded.append(None)
    valid = len(invalid_indices) == 0 and CADENCE_REASON_INVALID not in reason_codes
    precision_status = _precision_status(cycles)

    if gap_count == 0 or not valid:
        return CadenceSummary(
            cycle_count=cycle_count,
            gap_count=gap_count,
            exact_gaps_sec=exact_gaps,
            rounded_gaps_sec=rounded,
            min_gap_sec=None,
            max_gap_sec=None,
            mean_gap_sec=None,
            median_gap_sec=None,
            valid=valid,
            precision_status=precision_status,
            reason_codes=reason_codes,
            invalid_gap_indices=invalid_indices,
        )

    sorted_gaps = sorted(exact_gaps)
    mid = gap_count // 2
    median = (
        sorted_gaps[mid] if gap_count % 2 == 1 else (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2.0
    )

    return CadenceSummary(
        cycle_count=cycle_count,
        gap_count=gap_count,
        exact_gaps_sec=exact_gaps,
        rounded_gaps_sec=rounded,
        min_gap_sec=min(exact_gaps),
        max_gap_sec=max(exact_gaps),
        mean_gap_sec=sum(exact_gaps) / gap_count,
        median_gap_sec=median,
        valid=True,
        precision_status=precision_status,
        reason_codes=reason_codes,
        invalid_gap_indices=invalid_indices,
    )


# --------------------------------------------------------------------------
# Stop-event evidence (G1-MON-002 / F-10: validated construction)
# --------------------------------------------------------------------------

STOP_OUTCOME_REQUESTED = "STOP_REQUESTED"
STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED = "STOP_COMMAND_SUCCEEDED_UNVERIFIED"
STOP_OUTCOME_VERIFIED_ABSENT = "STOP_VERIFIED_ABSENT"
STOP_OUTCOME_FAILED = "STOP_FAILED"
STOP_OUTCOME_VERIFICATION_FAILED = "VERIFICATION_FAILED"
STOP_OUTCOME_NOT_EXECUTED = "NOT_EXECUTED"

VALID_STOP_OUTCOMES = {
    STOP_OUTCOME_REQUESTED,
    STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED,
    STOP_OUTCOME_VERIFIED_ABSENT,
    STOP_OUTCOME_FAILED,
    STOP_OUTCOME_VERIFICATION_FAILED,
    STOP_OUTCOME_NOT_EXECUTED,
}

# F-STOP-COMPAT-01: verification is the collapsed string form of a
# StopVerification (see build_stop_event) -- None means "verification was
# never attempted", the three non-None values are the only outcomes a real
# verify_pids_absent() call can ever produce. Any other string is rejected
# outright, regardless of outcome.
VALID_VERIFICATION_VALUES = {None, "QUERY_FAILED", "ALL_ABSENT", "STILL_PRESENT"}

# The complete outcome/verification/timestamp compatibility matrix. Every
# outcome is validated against this table (STOP_VERIFIED_ABSENT is no longer
# the only outcome with enforced invariants) so a contradictory combination
# on ANY outcome -- e.g. STOP_FAILED carrying an ALL_ABSENT verification, or
# STOP_REQUESTED carrying a completed_at_utc -- is rejected the same way a
# STOP_VERIFIED_ABSENT-without-verification record always was.
#
#   verification_allowed   -- the only verification values compatible with
#                              this outcome (ALL_ABSENT is exclusive to
#                              STOP_VERIFIED_ABSENT: it is the one outcome
#                              allowed to claim success).
#   allow_requested/completed/verified_absent -- False means this outcome
#                              must NEVER carry that timestamp field at all
#                              (not merely "doesn't need it").
#   require_requested/completed/verified_absent -- True means this outcome
#                              cannot be constructed without that field.
#   require_nonempty_pids  -- True means target_pids must be non-empty.
#   require_role            -- True means target_role must be non-empty.
_STOP_OUTCOME_RULES: Dict[str, Dict[str, Any]] = {
    STOP_OUTCOME_NOT_EXECUTED: dict(
        verification_allowed={None, "QUERY_FAILED", "STILL_PRESENT"},
        allow_requested=False,
        allow_completed=False,
        allow_verified_absent=False,
    ),
    STOP_OUTCOME_REQUESTED: dict(
        verification_allowed={None, "QUERY_FAILED", "STILL_PRESENT"},
        allow_requested=True,
        allow_completed=False,
        allow_verified_absent=False,
        require_requested=True,
    ),
    STOP_OUTCOME_COMMAND_SUCCEEDED_UNVERIFIED: dict(
        verification_allowed={None, "QUERY_FAILED", "STILL_PRESENT"},
        allow_requested=True,
        allow_completed=True,
        allow_verified_absent=False,
        require_requested=True,
        require_completed=True,
        require_nonempty_pids=True,
    ),
    STOP_OUTCOME_VERIFIED_ABSENT: dict(
        verification_allowed={"ALL_ABSENT"},
        allow_requested=True,
        allow_completed=True,
        allow_verified_absent=True,
        require_requested=True,
        require_completed=True,
        require_verified_absent=True,
        require_nonempty_pids=True,
        require_role=True,
    ),
    STOP_OUTCOME_FAILED: dict(
        verification_allowed={None, "QUERY_FAILED", "STILL_PRESENT"},
        allow_requested=True,
        allow_completed=True,
        allow_verified_absent=False,
    ),
    STOP_OUTCOME_VERIFICATION_FAILED: dict(
        verification_allowed={None, "QUERY_FAILED", "STILL_PRESENT"},
        allow_requested=True,
        allow_completed=True,
        allow_verified_absent=False,
    ),
}


@dataclass(frozen=True)
class StopVerification:
    checked_at_utc: str
    query_failed: bool
    still_present_pids: List[int]

    @property
    def all_absent(self) -> bool:
        return (not self.query_failed) and len(self.still_present_pids) == 0


def verify_pids_absent(lister: ProcessLister, pids: Sequence[int]) -> StopVerification:
    """Read-only re-check: does NOT stop anything, only samples the process
    table once and reports whether the given PIDs are currently absent."""
    records = lister()
    if records is None:
        return StopVerification(checked_at_utc=utc_now_iso(), query_failed=True, still_present_pids=[])
    present = {r.pid for r in records}
    still_present = [pid for pid in pids if pid in present]
    return StopVerification(checked_at_utc=utc_now_iso(), query_failed=False, still_present_pids=still_present)


@dataclass(frozen=True)
class StopEvent:
    """Every timestamp field defaults to None (missing), never a fabricated
    "now" -- populated only from caller-supplied evidence for that specific
    instant. This module never executes a stop itself.

    F-STOP-01: all invariants are enforced in __post_init__ below, i.e. in
    the data model itself -- NOT merely in build_stop_event(). This closes
    the direct-construction bypass a round-1/round-2 review found (calling
    `StopEvent(outcome="STOP_VERIFIED_ABSENT", verification=None, ...)`
    directly used to skip every check build_stop_event() performed).
    build_stop_event() remains as a convenience factory that turns a
    StopVerification object into the `verification` string field, then
    delegates to this same constructor -- there is exactly one enforcement
    path, so builder and direct construction can never diverge.

    F-1 (accepted round-4 re-review finding, closed here): frozen=True --
    a validated StopEvent can no longer be silently invalidated by
    post-construction attribute assignment (e.g. `event.outcome =
    STOP_OUTCOME_FAILED` on an already-validated STOP_VERIFIED_ABSENT
    instance), which previously bypassed __post_init__ entirely because
    assignment, unlike construction, never re-invokes it.
    dataclasses.replace() is unaffected -- it always constructs a new
    instance via __init__ and therefore always re-runs __post_init__,
    frozen or not."""

    target_role: str
    target_pids: List[int]
    requested_at_utc: Optional[str] = None
    completed_at_utc: Optional[str] = None
    verified_absent_at_utc: Optional[str] = None
    outcome: Optional[str] = None
    verification: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome is not None and self.outcome not in VALID_STOP_OUTCOMES:
            raise ValueError(f"unknown stop outcome: {self.outcome!r}")
        if self.verification not in VALID_VERIFICATION_VALUES:
            raise ValueError(f"unknown stop verification value: {self.verification!r}")

        # F-STOP-COMPAT-01: the full per-outcome compatibility matrix
        # (_STOP_OUTCOME_RULES above) is enforced for every outcome, not
        # merely STOP_VERIFIED_ABSENT -- this is what rejects, e.g., a
        # STOP_FAILED record carrying an ALL_ABSENT verification or a
        # verified_absent_at_utc timestamp, which the round-2/round-3
        # implementation only rejected when outcome was itself
        # STOP_VERIFIED_ABSENT. outcome=None remains unconstrained here
        # (as in prior rounds): it represents "no outcome recorded yet",
        # not a specific claim this matrix needs to police.
        if self.outcome is not None:
            rules = _STOP_OUTCOME_RULES[self.outcome]
            if self.verification not in rules["verification_allowed"]:
                allowed = sorted(v for v in rules["verification_allowed"] if v is not None)
                raise ValueError(
                    f"outcome {self.outcome!r} is not compatible with verification={self.verification!r} "
                    f"-- allowed verification values for this outcome: {allowed or ['<none>']}"
                )
            if not rules.get("allow_requested", True) and self.requested_at_utc is not None:
                raise ValueError(f"outcome {self.outcome!r} must not carry a requested_at_utc timestamp")
            if not rules.get("allow_completed", True) and self.completed_at_utc is not None:
                raise ValueError(f"outcome {self.outcome!r} must not carry a completed_at_utc timestamp")
            if not rules.get("allow_verified_absent", True) and self.verified_absent_at_utc is not None:
                raise ValueError(f"outcome {self.outcome!r} must not carry a verified_absent_at_utc timestamp")
            if rules.get("require_requested") and self.requested_at_utc is None:
                raise ValueError(f"outcome {self.outcome!r} requires a requested_at_utc timestamp")
            if rules.get("require_completed") and self.completed_at_utc is None:
                raise ValueError(f"outcome {self.outcome!r} requires a completed_at_utc timestamp")
            if rules.get("require_verified_absent") and self.verified_absent_at_utc is None:
                raise ValueError(f"outcome {self.outcome!r} requires a verified_absent_at_utc timestamp")
            if rules.get("require_nonempty_pids") and not self.target_pids:
                raise ValueError(
                    f"outcome {self.outcome!r} requires a non-empty target_pids list; "
                    "use STOP_OUTCOME_NOT_EXECUTED for a no-process stop"
                )
            if rules.get("require_role") and not self.target_role:
                raise ValueError(f"outcome {self.outcome!r} requires a non-empty target_role")

        ordered_fields = [
            ("requested_at_utc", self.requested_at_utc),
            ("completed_at_utc", self.completed_at_utc),
            ("verified_absent_at_utc", self.verified_absent_at_utc),
        ]
        parsed: List[Tuple[str, datetime]] = []
        for name, raw in ordered_fields:
            if raw is None:
                continue
            dt, status = normalize_utc_timestamp(raw)
            if dt is None:
                raise ValueError(f"{name} is not a valid timezone-aware UTC timestamp: {raw!r} ({status})")
            parsed.append((name, dt))
        for (prev_name, prev_dt), (curr_name, curr_dt) in zip(parsed, parsed[1:]):
            if curr_dt < prev_dt:
                raise ValueError(
                    f"stop-event chronology violation: {curr_name} ({curr_dt.isoformat()}) "
                    f"precedes {prev_name} ({prev_dt.isoformat()})"
                )


def build_stop_event(
    *,
    target_role: str,
    target_pids: Sequence[int],
    requested_at_utc: Optional[str] = None,
    completed_at_utc: Optional[str] = None,
    verification: Optional[StopVerification] = None,
    outcome: Optional[str] = None,
) -> StopEvent:
    """Convenience factory: reduces a StopVerification object to the
    StopEvent.verification string field (and derives verified_absent_at_utc
    from it, never accepting that timestamp directly), then constructs
    StopEvent -- all invariant enforcement happens inside StopEvent.__post_init__
    (F-STOP-01), identically for this factory and for direct construction."""
    verified_absent_at_utc: Optional[str] = None
    verification_str: Optional[str] = None
    if verification is not None:
        if verification.query_failed:
            verification_str = "QUERY_FAILED"
        elif verification.all_absent:
            verification_str = "ALL_ABSENT"
            verified_absent_at_utc = verification.checked_at_utc
        else:
            verification_str = "STILL_PRESENT"

    return StopEvent(
        target_role=target_role,
        target_pids=list(target_pids),
        requested_at_utc=requested_at_utc,
        completed_at_utc=completed_at_utc,
        verified_absent_at_utc=verified_absent_at_utc,
        outcome=outcome,
        verification=verification_str,
    )


# --------------------------------------------------------------------------
# Source provenance (G1-MON-003 / F-03: honest repo-vs-source containment)
# --------------------------------------------------------------------------

PROVENANCE_TRACKED_MATCHES_HEAD = "TRACKED_MATCHES_HEAD"
PROVENANCE_TRACKED_DIFFERS_FROM_HEAD = "TRACKED_DIFFERS_FROM_HEAD"
PROVENANCE_UNTRACKED_NOT_IN_HEAD = "UNTRACKED_NOT_IN_HEAD"
PROVENANCE_OUTSIDE_REPOSITORY = "OUTSIDE_REPOSITORY"
PROVENANCE_GIT_UNAVAILABLE = "GIT_UNAVAILABLE"
PROVENANCE_SOURCE_READ_FAILED = "SOURCE_READ_FAILED"


@dataclass
class SourceProvenance:
    source_path: str
    source_sha256: Optional[str]
    repository_head_commit: Optional[str]
    source_tracked: Optional[bool]
    source_blob_id: Optional[str]
    source_matches_head: Optional[bool]
    provenance_status: str


def _run_git(args: List[str], *, repo_root: Path, capture_bytes: bool = False):
    kwargs: Dict[str, Any] = dict(cwd=str(repo_root), timeout=5, check=False)
    if capture_bytes:
        kwargs["capture_output"] = True
    else:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(["git", *args], **kwargs)


def _current_git_commit(repo_root: Path = ROOT) -> Optional[str]:
    """Best-effort, read-only `git rev-parse HEAD`, scoped to `repo_root`
    (never an ambient/unrelated cwd) -- never raises; returns None if git
    is unavailable or the call fails."""
    try:
        result = _run_git(["rev-parse", "HEAD"], repo_root=repo_root)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def determine_source_provenance(source_path: Path, *, repo_root: Path = ROOT) -> SourceProvenance:
    """Separates "what is the repository's current HEAD" from "is this
    exact source file's content actually contained in that commit" --
    the manifest must never imply the latter merely because the former is
    known (F-03)."""
    try:
        raw = source_path.read_bytes()
    except OSError:
        return SourceProvenance(
            source_path=str(source_path),
            source_sha256=None,
            repository_head_commit=None,
            source_tracked=None,
            source_blob_id=None,
            source_matches_head=None,
            provenance_status=PROVENANCE_SOURCE_READ_FAILED,
        )
    source_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        rel = source_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return SourceProvenance(
            source_path=str(source_path),
            source_sha256=source_sha256,
            repository_head_commit=_current_git_commit(repo_root),
            source_tracked=None,
            source_blob_id=None,
            source_matches_head=None,
            provenance_status=PROVENANCE_OUTSIDE_REPOSITORY,
        )

    head_commit = _current_git_commit(repo_root)
    if head_commit is None:
        return SourceProvenance(
            source_path=str(source_path),
            source_sha256=source_sha256,
            repository_head_commit=None,
            source_tracked=None,
            source_blob_id=None,
            source_matches_head=None,
            provenance_status=PROVENANCE_GIT_UNAVAILABLE,
        )

    rel_posix = rel.as_posix()
    blob_id: Optional[str] = None
    try:
        blob_result = _run_git(["rev-parse", f"HEAD:{rel_posix}"], repo_root=repo_root)
        if blob_result.returncode == 0:
            blob_id = blob_result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        blob_id = None

    if blob_id is None:
        return SourceProvenance(
            source_path=str(source_path),
            source_sha256=source_sha256,
            repository_head_commit=head_commit,
            source_tracked=False,
            source_blob_id=None,
            source_matches_head=False,
            provenance_status=PROVENANCE_UNTRACKED_NOT_IN_HEAD,
        )

    matches = False
    try:
        # `git diff --quiet` (exit 0 == no diff) is used rather than a raw
        # byte comparison against `git show HEAD:<path>`: this repo's
        # core.autocrlf=true means a clean, unmodified checked-out text
        # file legitimately differs byte-for-byte from its stored LF blob
        # (CRLF conversion) -- git's own porcelain already treats that as
        # "not modified", and this function should agree with git's own
        # notion of "differs from HEAD", not a naive byte comparison that
        # would falsely flag every untouched text file as differing.
        diff_result = _run_git(["diff", "--quiet", "HEAD", "--", rel_posix], repo_root=repo_root)
        matches = diff_result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        matches = False

    status = PROVENANCE_TRACKED_MATCHES_HEAD if matches else PROVENANCE_TRACKED_DIFFERS_FROM_HEAD
    return SourceProvenance(
        source_path=str(source_path),
        source_sha256=source_sha256,
        repository_head_commit=head_commit,
        source_tracked=True,
        source_blob_id=blob_id,
        source_matches_head=matches,
        provenance_status=status,
    )


# --------------------------------------------------------------------------
# Run manifest
# --------------------------------------------------------------------------


@dataclass
class RunManifest:
    schema_version: str
    monitor_version: str
    source_path: str
    source_sha256: Optional[str]
    repository_head_commit: Optional[str]
    source_tracked: Optional[bool]
    source_blob_id: Optional[str]
    source_matches_head: Optional[bool]
    source_provenance_status: str
    start_time_utc: str
    cadence_sec: Optional[int]
    requested_duration_sec: Optional[int]
    role_definitions: Dict[str, Any]
    artifact_paths: Dict[str, str]
    arguments: Dict[str, Any]
    operating_mode: str = "INSPECT_ONLY"


def build_run_manifest(
    *,
    cadence_sec: Optional[int],
    requested_duration_sec: Optional[int],
    role_definitions: Dict[str, Any],
    artifact_paths: Dict[str, str],
    arguments: Dict[str, Any],
    source_provenance: Optional[SourceProvenance] = None,
) -> RunManifest:
    provenance = source_provenance or determine_source_provenance(Path(__file__))
    return RunManifest(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        monitor_version=MONITOR_VERSION,
        source_path=provenance.source_path,
        source_sha256=provenance.source_sha256,
        repository_head_commit=provenance.repository_head_commit,
        source_tracked=provenance.source_tracked,
        source_blob_id=provenance.source_blob_id,
        source_matches_head=provenance.source_matches_head,
        source_provenance_status=provenance.provenance_status,
        start_time_utc=utc_now_iso(),
        cadence_sec=cadence_sec,
        requested_duration_sec=requested_duration_sec,
        role_definitions=role_definitions,
        artifact_paths=artifact_paths,
        arguments=arguments,
    )


# --------------------------------------------------------------------------
# Evidence persistence -- append-only JSONL, dedicated output path only
# --------------------------------------------------------------------------


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Bounded, explicit, one-record-per-call append. Never writes into an
    existing production health file -- callers must always pass a dedicated
    evidence-output path (see main()'s --evidence-dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":"), default=str) + "\n")


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    atomic_write_json(path, asdict(manifest))


# --------------------------------------------------------------------------
# PID-metadata baseline loader (F-06: wires continuity into the CLI)
# --------------------------------------------------------------------------


# F-BASELINE-01: baseline-loading outcome vocabulary. BASELINE_STATUS_OK is
# the only status that yields a usable (baseline, expected_identity) pair;
# the other two distinguish WHY loading failed -- BASELINE_UNREADABLE means
# the file itself could not be obtained as valid UTF-8 text (missing, a
# directory, permission-denied, any other I/O failure, or bad encoding);
# BASELINE_INVALID means the file WAS read successfully but its content is
# not a usable baseline (empty, malformed JSON, or a JSON object missing/
# mistyping the required pid/started_at fields). reason_code is always a
# single specific token distinguishing the exact sub-case (never a single
# catch-all string), matching the required "parsing, schema, and read
# failures" distinction.
BASELINE_STATUS_OK = "OK"
BASELINE_STATUS_UNREADABLE = "BASELINE_UNREADABLE"
BASELINE_STATUS_INVALID = "BASELINE_INVALID"


@dataclass(frozen=True)
class BaselineLoadResult:
    status: str
    reason_code: str
    baseline: Optional[ProcessRecord]
    expected_identity: Optional[ExpectedIdentity]
    error: Optional[str] = None


def load_pid_metadata_baseline(
    pid_meta_path: Path, *, expected_module: Optional[str] = None
) -> BaselineLoadResult:
    """Reads the PID-metadata JSON sibling file that this repository's
    process-startup wiring already writes for
    tools/liquidation_silence_scheduler.py (schema:
    pid/started_at/cmdline_sig/role/mode/cwd) and turns it into a
    BaselineLoadResult. Every expected I/O and schema outcome (missing
    file, a directory supplied instead of a file, a permission/read
    failure, invalid UTF-8, an empty file, malformed JSON, or a JSON object
    missing/mistyping the required pid/started_at fields) produces a
    distinct typed status/reason_code -- never an unhandled exception,
    never a fabricated baseline (F-BASELINE-01, mirrors sample_artifact's
    fail-closed contract for F-08/F-09). A syntactically-present but
    semantically unparsable started_at string (e.g. "not-a-timestamp") is
    NOT rejected here -- it is passed through as BASELINE_STATUS_OK so the
    existing PID+StartTime+identity continuity model in classify_continuity
    can report the more specific CONTINUITY_MALFORMED_STARTTIME, rather
    than this loader pre-empting that with a less specific verdict."""
    if not pid_meta_path.exists():
        return BaselineLoadResult(
            status=BASELINE_STATUS_UNREADABLE,
            reason_code="BASELINE_MISSING",
            baseline=None,
            expected_identity=None,
        )
    if pid_meta_path.is_dir():
        return BaselineLoadResult(
            status=BASELINE_STATUS_UNREADABLE,
            reason_code="BASELINE_NOT_A_REGULAR_FILE",
            baseline=None,
            expected_identity=None,
            error="path is a directory, not a regular file",
        )

    try:
        raw_bytes = pid_meta_path.read_bytes()
    except PermissionError as exc:
        return BaselineLoadResult(
            status=BASELINE_STATUS_UNREADABLE,
            reason_code="BASELINE_PERMISSION_DENIED",
            baseline=None,
            expected_identity=None,
            error=str(exc),
        )
    except OSError as exc:
        return BaselineLoadResult(
            status=BASELINE_STATUS_UNREADABLE,
            reason_code="BASELINE_READ_FAILED",
            baseline=None,
            expected_identity=None,
            error=str(exc),
        )

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return BaselineLoadResult(
            status=BASELINE_STATUS_UNREADABLE,
            reason_code="BASELINE_INVALID_ENCODING",
            baseline=None,
            expected_identity=None,
            error=str(exc),
        )

    if raw_text.strip() == "":
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_EMPTY_FILE",
            baseline=None,
            expected_identity=None,
        )

    try:
        meta = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_MALFORMED_JSON",
            baseline=None,
            expected_identity=None,
            error=str(exc),
        )

    if not isinstance(meta, dict):
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_SCHEMA_NOT_AN_OBJECT",
            baseline=None,
            expected_identity=None,
        )
    if "pid" not in meta or "started_at" not in meta:
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_SCHEMA_MISSING_REQUIRED_FIELDS",
            baseline=None,
            expected_identity=None,
        )

    pid = meta.get("pid")
    started_at = meta.get("started_at")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_SCHEMA_MALFORMED_PID",
            baseline=None,
            expected_identity=None,
        )
    if not started_at or not isinstance(started_at, str):
        return BaselineLoadResult(
            status=BASELINE_STATUS_INVALID,
            reason_code="BASELINE_SCHEMA_MALFORMED_STARTTIME",
            baseline=None,
            expected_identity=None,
        )

    baseline = ProcessRecord(
        pid=pid,
        start_time_utc=started_at,
        command_line=meta.get("cmdline_sig"),
        captured_at_utc=utc_now_iso(),
    )
    expected_identity = ExpectedIdentity(
        role=str(meta.get("role") or "unknown"),
        module_name=expected_module,
    )
    return BaselineLoadResult(
        status=BASELINE_STATUS_OK,
        reason_code="OK",
        baseline=baseline,
        expected_identity=expected_identity,
    )


# --------------------------------------------------------------------------
# CLI -- default inspect-only single snapshot; no process control anywhere
# --------------------------------------------------------------------------


def take_snapshot(
    *,
    scheduler_log_path: Path,
    evidence_dir: Path,
    pid_meta_path: Optional[Path] = None,
    expected_module: str = "tools.liquidation_silence_scheduler",
    lister: ProcessLister = default_process_lister,
) -> Dict[str, Any]:
    """One inspect-only sampling pass: loads the scheduler's own cycle log
    (never mutates it) via the fail-closed load_cycle_log() (F-CYCLELOG-01),
    computes the cadence summary from whatever cycles that yielded (zero for
    any non-OK cycle-log status -- never fabricated), and -- when
    pid_meta_path resolves to a readable baseline -- samples current
    scheduler continuity against it. Writes one evidence JSONL record and
    one run manifest, both under evidence_dir -- never under logs/health or
    logs/pids. continuity is NEVER a bare null: absent a usable baseline it
    is the explicit NOT_EVALUATED/BASELINE_NOT_PROVIDED marker; likewise the
    output's "cycle_log" field is never a bare null -- an unreadable or
    directory --scheduler-log-path never raises, it degrades to a typed
    CYCLE_LOG_* status with zero fabricated cycles/gaps."""
    cycle_log_result = load_cycle_log(scheduler_log_path)
    cycles = cycle_log_result.observations
    cadence = compute_cadence_summary(cycles)

    continuity: Dict[str, Any]
    baseline_result: Dict[str, Any]
    if pid_meta_path is not None:
        loaded = load_pid_metadata_baseline(pid_meta_path, expected_module=expected_module)
        baseline_result = {
            "status": loaded.status,
            "reason_codes": [loaded.reason_code],
        }
        if loaded.status != BASELINE_STATUS_OK or loaded.baseline is None:
            continuity = not_evaluated_continuity(loaded.reason_code)
        else:
            snapshot = sample_role_continuity(
                "liquidation_silence_scheduler",
                loaded.baseline,
                lister,
                expected_identity=loaded.expected_identity,
            )
            continuity = asdict(snapshot)
    else:
        continuity = not_evaluated_continuity("BASELINE_NOT_PROVIDED")
        baseline_result = {"status": "NOT_EVALUATED", "reason_codes": ["BASELINE_NOT_PROVIDED"]}

    manifest = build_run_manifest(
        cadence_sec=None,
        requested_duration_sec=None,
        role_definitions={"scheduler_log_path": str(scheduler_log_path)},
        artifact_paths={"scheduler_log_path": str(scheduler_log_path)},
        arguments={
            "scheduler_log_path": str(scheduler_log_path),
            "evidence_dir": str(evidence_dir),
            "pid_meta_path": str(pid_meta_path) if pid_meta_path else None,
        },
    )
    write_run_manifest(evidence_dir / "run_manifest.json", manifest)

    record: Dict[str, Any] = {
        "sampled_at_utc": utc_now_iso(),
        "cadence_summary": asdict(cadence),
        "continuity": continuity,
        "baseline": baseline_result,
        "cycle_log": {
            "status": cycle_log_result.status,
            "reason_code": cycle_log_result.reason_code,
            "path": cycle_log_result.path,
            "record_count": cycle_log_result.record_count,
            "malformed_record_count": cycle_log_result.malformed_record_count,
            "valid_cycle_count": len(cycle_log_result.observations),
        },
    }
    append_jsonl_record(evidence_dir / "snapshots.jsonl", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect-only evidence harness for liquidation-silence scheduler canary runs. "
            "Never starts, stops, or signals any process; reads the scheduler's own cycle "
            "log and current process table only. Always requires an explicit, dedicated "
            "--evidence-dir -- never writes into logs/health or logs/pids."
        )
    )
    parser.add_argument(
        "--scheduler-log-path",
        type=str,
        default=str(ROOT / "logs" / "liquidation_silence_scheduler.log"),
    )
    parser.add_argument("--evidence-dir", type=str, required=True)
    parser.add_argument(
        "--pid-meta-path",
        type=str,
        default=None,
        help="Optional path to a PID-metadata JSON (pid/started_at/cmdline_sig) used as the "
        "scheduler continuity baseline. If omitted, continuity is reported as NOT_EVALUATED.",
    )
    parser.add_argument("--expected-module", type=str, default="tools.liquidation_silence_scheduler")
    args = parser.parse_args()

    record = take_snapshot(
        scheduler_log_path=Path(args.scheduler_log_path),
        evidence_dir=Path(args.evidence_dir),
        pid_meta_path=Path(args.pid_meta_path) if args.pid_meta_path else None,
        expected_module=args.expected_module,
    )
    print(json.dumps(record, ensure_ascii=True, separators=(",", ":"), default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
