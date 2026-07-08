"""Canonical warehouse — Phase 1 (Reconstruction Protocol §7 skeleton).

Read-only relative to all existing AMI/S34 stores and running processes.
New file only: data/ami/canonical.sqlite.
"""
from ami.warehouse.schema import CANONICAL_SCHEMA_VERSION, connect

__all__ = ["CANONICAL_SCHEMA_VERSION", "connect"]
