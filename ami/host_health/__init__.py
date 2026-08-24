"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1.

Read-only Windows host-health observation and restart-readiness
evaluation. No function in this package (or its submodules) restarts,
shuts down, suspends, or otherwise mutates the host, a collector process,
or any repository-governed database. See `evaluator.py` for the pure
deterministic state machine and `observation.py` for the fail-closed
observation collector.
"""
