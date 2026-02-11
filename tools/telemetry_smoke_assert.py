#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _norm_upper(value: Any) -> str:
    return _norm(value).upper()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assert telemetry notifier smoke expectations")
    p.add_argument("--state", default="logs/telemetry_dashboard_notify_state.json")
    p.add_argument("--expected-level", default="", help="Expected notifier level (normal/critical)")
    p.add_argument("--expected-stage", default="", help="Expected recovery stage (e.g. RED_LOCK)")
    p.add_argument(
        "--expected-red-lock-streak",
        default="",
        help="Expected recovery_red_lock_streak integer value",
    )
    args = p.parse_args(argv)

    exp_level = _norm(args.expected_level).lower()
    exp_stage = _norm_upper(args.expected_stage)
    exp_streak_raw = _norm(args.expected_red_lock_streak)
    has_expectation = bool(exp_level or exp_stage or exp_streak_raw)
    if not has_expectation:
        print("smoke_assert: no expectations provided; skipping")
        return 0

    state = _load_state(Path(args.state))
    if not state:
        print("smoke_assert: state missing or unreadable")
        return 2

    mismatches: list[str] = []
    if exp_level:
        actual_level = _norm(state.get("level")).lower()
        if actual_level != exp_level:
            mismatches.append(f"level expected={exp_level} actual={actual_level or 'n/a'}")

    if exp_stage:
        actual_stage = _norm_upper(state.get("recovery_stage_latest"))
        if actual_stage != exp_stage:
            mismatches.append(f"recovery_stage_latest expected={exp_stage} actual={actual_stage or 'n/a'}")

    if exp_streak_raw:
        try:
            exp_streak = int(exp_streak_raw)
        except Exception:
            print(f"smoke_assert: invalid expected-red-lock-streak={exp_streak_raw}")
            return 2
        try:
            actual_streak = int(state.get("recovery_red_lock_streak") or 0)
        except Exception:
            actual_streak = 0
        if actual_streak != exp_streak:
            mismatches.append(f"recovery_red_lock_streak expected={exp_streak} actual={actual_streak}")

    if mismatches:
        print("smoke_assert: FAIL")
        for line in mismatches:
            print(f"- {line}")
        return 1

    print("smoke_assert: PASS")
    if exp_level:
        print(f"- level={exp_level}")
    if exp_stage:
        print(f"- recovery_stage_latest={exp_stage}")
    if exp_streak_raw:
        print(f"- recovery_red_lock_streak={int(exp_streak_raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
