#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_telemetry_event(ts: float) -> dict:
    return {
        "event": "execution.belief_state",
        "ts": float(ts),
        "data": {
            "belief_debt_sec": 12.0,
            "belief_debt_symbols": 1,
            "belief_confidence": 0.91,
            "mismatch_streak": 0,
            "guard_mode": "ORANGE",
            "allow_entries": False,
            "guard_recovery_stage": "RUNTIME_GATE_DEGRADED",
            "guard_unlock_conditions": "runtime gate clear required",
            "guard_next_unlock_sec": 45.0,
            "guard_cause_tags": "runtime_gate,runtime_gate_position_peak,runtime_gate_coverage_gap_peak",
            "guard_dominant_contributors": "position=2.0,coverage_gap=1.5",
        },
    }


def build_reliability_gate_text() -> str:
    lines = [
        "Execution Reliability Gate",
        "replay_mismatch_count=0",
        "invalid_transition_count=0",
        "journal_coverage_ratio=1.000",
        "position_mismatch_count=0",
        "position_mismatch_count_peak=2",
        "orphan_count=0",
        "protection_coverage_gap_seconds=0.0",
        "protection_coverage_gap_seconds_peak=12.0",
        "replace_race_count=0",
        "evidence_contradiction_count=0",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate local peak-threshold telemetry fixtures for dashboard/notifier verification."
    )
    ap.add_argument("--telemetry-path", default="logs/telemetry_peak_fixture.jsonl")
    ap.add_argument("--gate-path", default="logs/reliability_gate_peak_fixture.txt")
    ap.add_argument("--ts", type=float, default=1770900000.0)
    args = ap.parse_args(argv)

    telemetry_path = Path(args.telemetry_path)
    gate_path = Path(args.gate_path)

    event = build_telemetry_event(float(args.ts))
    _write_text(telemetry_path, json.dumps(event, ensure_ascii=True) + "\n")
    _write_text(gate_path, build_reliability_gate_text())

    print(f"Wrote telemetry fixture: {telemetry_path}")
    print(f"Wrote reliability gate fixture: {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
