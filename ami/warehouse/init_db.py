"""Idempotent entrypoint: python -m ami.warehouse.init_db

Creates/verifies data/ami/canonical.sqlite schema. Safe to run repeatedly
(second run is a no-op on the schema, only refreshes schema_versions.applied_ms).
Touches no other store and no running process.
"""
from __future__ import annotations
from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema


def main() -> None:
    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        print(f"canonical warehouse ready at {DEFAULT_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
