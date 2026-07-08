"""AMI Scientific Constitution — Part I §4 / Part XII §29.

Principles are data (auditable) + a few machine-checkable helpers used by the
governor and stores. Changing PRINCIPLES requires a decision record
(docs/ami/AMI_DECISION_RECORDS/) per Appendix G.
"""
from __future__ import annotations

VERSION = "1.0.0"

PRINCIPLES = [
    "No claim without provenance.",
    "No operational promotion without untouched validation.",
    "No mechanism claim from correlation alone.",
    "No confidence without calibration.",
    "No theory without falsifiable predictions.",
    "No live rule without execution validation.",
    "No failed idea is silently deleted.",
    "No contradiction is ignored.",
    "No unknown is converted into certainty.",
    "No agent may override the evidence hierarchy.",
    "No economic attractiveness substitutes for scientific validity.",
    "Every decision must be traceable to active evidence.",
    "Every model must expose where it is likely to fail.",
    "Every state must be timeframe-aware.",
    "LONG and SHORT must be studied as connected structural phases.",
    "Research, shadow, paper and live permissions remain architecturally separate.",
    "The system must be able to say 'I do not know'.",
]

# Appendix F mandatory guardrails — paths AMI must never write to.
FORBIDDEN_WRITE_PATHS = [
    "tools/s34_state_machine_live_executor.py",
    ".env",
    "execution/",
    "risk/",
    "brain/",
]


class ConstitutionViolation(Exception):
    """Raised when a component attempts an action the constitution forbids."""


def require(condition: bool, principle: str) -> None:
    if not condition:
        raise ConstitutionViolation(principle)
