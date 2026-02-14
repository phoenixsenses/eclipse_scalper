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
    p.add_argument(
        "--expected-intent-collision-streak",
        default="",
        help="Expected intent_collision_streak integer value",
    )
    p.add_argument(
        "--min-refresh-budget-blocked",
        default="",
        help="Optional minimum protection_refresh_budget_blocked_count expected in state",
    )
    p.add_argument(
        "--require-entry-clamped-on-refresh-budget",
        action="store_true",
        help="When min refresh budget is met, require belief_allow_entries_latest=false",
    )
    p.add_argument(
        "--require-refresh-consistency",
        action="store_true",
        help=(
            "Fail if refresh budget is elevated while entries are allowed and stage is not "
            "PROTECTION_REFRESH_WARMUP/GREEN."
        ),
    )
    p.add_argument(
        "--require-unlock-fields",
        action="store_true",
        help=(
            "Require unlock-condition numeric fields to exist in state "
            "(healthy ticks/journal coverage/contradiction clear/protection gap)."
        ),
    )
    args = p.parse_args(argv)

    exp_level = _norm(args.expected_level).lower()
    exp_stage = _norm_upper(args.expected_stage)
    exp_streak_raw = _norm(args.expected_red_lock_streak)
    exp_intent_collision_streak_raw = _norm(args.expected_intent_collision_streak)
    exp_refresh_blocked_raw = _norm(args.min_refresh_budget_blocked)
    has_expectation = bool(
        exp_level
        or exp_stage
        or exp_streak_raw
        or exp_intent_collision_streak_raw
        or exp_refresh_blocked_raw
        or bool(args.require_entry_clamped_on_refresh_budget)
        or bool(args.require_refresh_consistency)
        or bool(args.require_unlock_fields)
    )
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

    if exp_intent_collision_streak_raw:
        try:
            exp_collision_streak = int(exp_intent_collision_streak_raw)
        except Exception:
            print(
                "smoke_assert: invalid expected-intent-collision-streak="
                f"{exp_intent_collision_streak_raw}"
            )
            return 2
        try:
            actual_collision_streak = int(state.get("intent_collision_streak") or 0)
        except Exception:
            actual_collision_streak = 0
        if actual_collision_streak != exp_collision_streak:
            mismatches.append(
                "intent_collision_streak "
                f"expected={exp_collision_streak} actual={actual_collision_streak}"
            )

    actual_refresh_blocked = int(state.get("protection_refresh_budget_blocked_count") or 0)
    if exp_refresh_blocked_raw:
        try:
            exp_refresh_blocked = int(exp_refresh_blocked_raw)
        except Exception:
            print(f"smoke_assert: invalid min-refresh-budget-blocked={exp_refresh_blocked_raw}")
            return 2
        if actual_refresh_blocked < exp_refresh_blocked:
            mismatches.append(
                "protection_refresh_budget_blocked_count "
                f"expected>={exp_refresh_blocked} actual={actual_refresh_blocked}"
            )

    if args.require_entry_clamped_on_refresh_budget:
        threshold = 1
        if exp_refresh_blocked_raw:
            try:
                threshold = int(exp_refresh_blocked_raw)
            except Exception:
                threshold = 1
        if int(actual_refresh_blocked) >= int(max(1, threshold)):
            allow_entries = bool(state.get("belief_allow_entries_latest", True))
            if allow_entries:
                mismatches.append(
                    "belief_allow_entries_latest expected=false when refresh budget is elevated"
                )

    if args.require_refresh_consistency:
        allow_entries = bool(state.get("belief_allow_entries_latest", True))
        stage = _norm_upper(state.get("recovery_stage_latest"))
        if int(actual_refresh_blocked) > 0 and allow_entries and stage not in ("PROTECTION_REFRESH_WARMUP", "GREEN"):
            mismatches.append(
                "refresh consistency violated: blocked>0 with allow_entries=true outside warmup/green "
                f"(stage={stage or 'n/a'})"
            )

    if args.require_unlock_fields:
        required_keys = (
            "unlock_next_sec",
            "unlock_healthy_ticks_current",
            "unlock_healthy_ticks_required",
            "unlock_healthy_ticks_remaining",
            "unlock_journal_coverage_current",
            "unlock_journal_coverage_required",
            "unlock_journal_coverage_remaining",
            "unlock_contradiction_clear_current_sec",
            "unlock_contradiction_clear_required_sec",
            "unlock_contradiction_clear_remaining_sec",
            "unlock_protection_gap_current_sec",
            "unlock_protection_gap_max_sec",
            "unlock_protection_gap_remaining_sec",
        )
        missing = [k for k in required_keys if k not in state]
        if missing:
            mismatches.append("unlock fields missing: " + ", ".join(missing))

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
    if exp_intent_collision_streak_raw:
        print(f"- intent_collision_streak={int(exp_intent_collision_streak_raw)}")
    if exp_refresh_blocked_raw:
        print(f"- protection_refresh_budget_blocked_count>={int(exp_refresh_blocked_raw)}")
    if args.require_entry_clamped_on_refresh_budget:
        print("- belief_allow_entries_latest=false when refresh budget elevated")
    if args.require_refresh_consistency:
        print("- refresh consistency check passed")
    if args.require_unlock_fields:
        print("- unlock fields present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
