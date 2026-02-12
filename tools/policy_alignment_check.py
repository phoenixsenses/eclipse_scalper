#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _parse_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = str(raw or "").strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v or "").strip()
    return out


def _validate(kv: dict[str, str]) -> list[str]:
    errs: list[str] = []
    required = (
        "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT",
        "INTENT_COLLISION_CRITICAL_THRESHOLD",
        "INTENT_COLLISION_CRITICAL_STREAK",
        "summary.arg.reliability_max_intent_collision_count",
        "summary.arg.intent_collision_critical_threshold",
        "summary.arg.intent_collision_critical_streak",
        "gate.intent_collision_count",
        "notify.reliability_intent_collision_count",
        "notify.intent_collision_streak",
        "notify.level",
    )
    missing = [k for k in required if k not in kv]
    if missing:
        errs.append("missing keys: " + ", ".join(missing))
        return errs
    pairs = (
        (
            "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT",
            "summary.arg.reliability_max_intent_collision_count",
        ),
        (
            "INTENT_COLLISION_CRITICAL_THRESHOLD",
            "summary.arg.intent_collision_critical_threshold",
        ),
        (
            "INTENT_COLLISION_CRITICAL_STREAK",
            "summary.arg.intent_collision_critical_streak",
        ),
    )
    for lhs, rhs in pairs:
        lv = kv.get(lhs, "")
        rv = kv.get(rhs, "")
        if not lv or not rv or lv != rv:
            errs.append(f"{lhs}={lv!r} != {rhs}={rv!r}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate policy_alignment.txt keys and threshold consistency")
    ap.add_argument("--path", default="logs/policy_alignment.txt")
    args = ap.parse_args(argv)

    path = Path(args.path)
    kv = _parse_kv(path)
    if not kv:
        print("policy_alignment_check: missing or empty policy alignment file")
        return 2
    errs = _validate(kv)
    if errs:
        print("policy_alignment_check: FAIL")
        for e in errs:
            print(f"- {e}")
        return 1
    print("policy_alignment_check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
