from __future__ import annotations

from typing import Any, Dict


RUN_SUMMARY_VERSION = "v1"


def build_run_summary(
    *,
    run_type: str,
    inputs: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "version": RUN_SUMMARY_VERSION,
        "run_type": str(run_type),
        "inputs": dict(inputs or {}),
        "metrics": dict(metrics or {}),
        "artifacts": dict(artifacts or {}),
    }
