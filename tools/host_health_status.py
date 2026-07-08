"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1.

Read-only CLI: `python -m tools.host_health_status`. Prints a
machine-readable JSON restart-readiness assessment and exits 0/1/2/3.

This command NEVER restarts, shuts down, suspends, or modifies the host,
never stops or restarts a collector process, and exposes no
restart/reboot/shutdown/collector-stop/collector-restart/force-reset
flag of any kind. It only observes and classifies.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from ami.host_health.evaluator import evaluate_restart_readiness
from ami.host_health.observation import build_health_inputs, collect_host_observation

_EXIT_CODE_BY_STATE = {
    "HOST_RESTART_GREEN": 0,
    "HOST_RESTART_YELLOW": 1,
    "HOST_RESTART_RED": 2,
    "HOST_RESTART_UNKNOWN": 3,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read-only host-health / restart-readiness observation and evaluation. "
            "Never restarts, shuts down, or modifies the host or any collector process."
        )
    )
    p.add_argument("--repo-root", default=None, help="Override repo root (default: autodetect).")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return p


def build_status_payload(repo_root: Path | None = None) -> dict:
    obs = collect_host_observation(repo_root=repo_root)
    inputs = build_health_inputs(obs)
    evaluation = evaluate_restart_readiness(inputs)

    return {
        "state": evaluation.state,
        "recommended_action": evaluation.recommended_action,
        "deferred": evaluation.deferred,
        "primary_reason": evaluation.primary_reason,
        "reasons": list(evaluation.reason_codes),
        "observations": dataclasses.asdict(obs),
        "unknown_fields": list(dict.fromkeys(list(evaluation.unknown_fields) + list(obs.unknown_fields))),
        "stale_fields": list(obs.stale_fields),
        "observation_timestamp": obs.observation_ts_utc,
        "no_automatic_action": True,
    }


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(args.repo_root) if args.repo_root else None
    payload = build_status_payload(repo_root=repo_root)
    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, default=str))
    return _EXIT_CODE_BY_STATE.get(payload["state"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
