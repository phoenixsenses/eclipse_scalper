"""Bounded storage-archive CLI (Phase 16), following the repository's
`tools/*.py` convention (`parse_args(argv) -> argparse.Namespace`,
`main(argv=None) -> int`, `sys.exit(main())`).

Commands: `policy-status`, `plan`, `disposable-dry-run`, `verify`, `read`,
`restore-slice`. There is deliberately no `purge`, `delete`, `vacuum`,
`schedule`, `activate-production`, `stop-collector`, or
`restart-collector` command -- none of that functionality exists
anywhere in this package, so no such command could ever be wired up.
"""
from __future__ import annotations

import argparse
import json
import sys

from ami.storage.policy import DEFAULT_POLICY
from ami.storage.registry import allowlisted_tables, get_table_spec, UnknownTableError

# Commands this CLI will never contain -- asserted by a focused test that
# scans the built parser for these strings.
FORBIDDEN_COMMANDS = ("purge", "delete", "vacuum", "schedule", "activate-production",
                      "stop-collector", "restart-collector")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ami-storage", description="Bounded, non-destructive storage-archive tooling "
                                         "(rotation/retention). No purge, VACUUM, or production "
                                         "activation command exists in this CLI.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("policy-status", help="Show the effective storage policy (read-only).")
    sp.set_defaults(func=_cmd_policy_status)

    sp = sub.add_parser("plan", help="Read-only partition plan for one table/symbol/UTC-month.")
    sp.add_argument("--table", required=True, choices=allowlisted_tables())
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--utc-year", required=True, type=int)
    sp.add_argument("--utc-month", required=True, type=int)
    sp.add_argument("--source-db", default=None, help="Override source database path (default: repository standard).")
    sp.set_defaults(func=_cmd_plan)

    sp = sub.add_parser("disposable-dry-run",
                        help="Export+verify one allowlisted partition into a disposable output root.")
    sp.add_argument("--table", required=True, choices=allowlisted_tables())
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--utc-year", required=True, type=int)
    sp.add_argument("--utc-month", required=True, type=int)
    sp.add_argument("--output-root", required=True,
                    help="Must be beneath .runtime_temp or .pytest_temp.")
    sp.add_argument("--source-db", default=None)
    sp.set_defaults(func=_cmd_disposable_dry_run)

    sp = sub.add_parser("verify", help="Verify one disposable archive + manifest pair.")
    sp.add_argument("--parquet-path", required=True)
    sp.add_argument("--manifest-path", required=True)
    sp.set_defaults(func=_cmd_verify)

    sp = sub.add_parser("read", help="Bounded direct read of one verified disposable Parquet partition.")
    sp.add_argument("--parquet-path", required=True)
    sp.add_argument("--manifest-path", required=True)
    sp.add_argument("--symbol", required=True)
    sp.set_defaults(func=_cmd_read)

    sp = sub.add_parser("restore-slice", help="Restore a verified partition into a minimal disposable SQLite slice.")
    sp.add_argument("--parquet-path", required=True)
    sp.add_argument("--manifest-path", required=True)
    sp.add_argument("--table", required=True, choices=allowlisted_tables())
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--destination", required=True,
                    help="Must be beneath .runtime_temp or .pytest_temp.")
    sp.set_defaults(func=_cmd_restore_slice)

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _cmd_policy_status(args: argparse.Namespace) -> dict:
    from dataclasses import asdict
    return {
        "policy": asdict(DEFAULT_POLICY),
        "production_activation": "DISABLED",
        "scheduler": "DISABLED",
        "purge": "DISABLED",
        "vacuum": "DISABLED",
        "allowlisted_tables": list(allowlisted_tables()),
    }


def _cmd_plan(args: argparse.Namespace) -> dict:
    from ami.storage.partition import plan_partition
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean, DEFAULT_SOURCE_PATH

    conn, log = open_read_only(args.source_db or DEFAULT_SOURCE_PATH)
    try:
        plan = plan_partition(conn, table=args.table, symbol=args.symbol,
                               utc_year=args.utc_year, utc_month=args.utc_month)
    finally:
        assert_read_only_session_clean(log)
        conn.close()
    return {
        "plan_state": plan.plan_state, "estimated_row_count": plan.estimated_row_count,
        "estimated_source_bytes": plan.estimated_source_bytes,
        "unresolved_gap_count": plan.unresolved_gap_count, "repair_status": plan.repair_status,
        "archive_rehearsal_eligible": plan.archive_rehearsal_eligible,
        "production_activation_eligible": plan.production_activation_eligible,
        "purge_eligible": plan.purge_eligible, "blockers": list(plan.blockers),
    }


def _cmd_disposable_dry_run(args: argparse.Namespace) -> dict:
    from ami.storage.partition import plan_partition, build_partition_identity
    from ami.storage.archive import export_partition, build_manifest, canonical_row_hash, fetch_partition_rows
    from ami.storage.registry import get_table_spec
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean, DEFAULT_SOURCE_PATH
    import datetime as dt

    allowed_roots = (".runtime_temp", ".pytest_temp")
    conn, log = open_read_only(args.source_db or DEFAULT_SOURCE_PATH)
    try:
        plan = plan_partition(conn, table=args.table, symbol=args.symbol,
                               utc_year=args.utc_year, utc_month=args.utc_month)
        if not plan.archive_rehearsal_eligible:
            return {"status": "BLOCKED", "plan_state": plan.plan_state, "blockers": list(plan.blockers)}
        spec = get_table_spec(args.table)
        result = export_partition(conn, args.table, plan.partition, args.output_root,
                                   allowed_roots, max_output_bytes=DEFAULT_POLICY.max_disposable_output_bytes)
    finally:
        assert_read_only_session_clean(log)
        conn.close()
    return {"status": "EXPORTED", "row_count": result["row_count"],
            "scientific_content_hash": result["scientific_content_hash"],
            "parquet_sha256": result["parquet_sha256"], "final_path": result["final_path"]}


def _cmd_verify(args: argparse.Namespace) -> dict:
    import json as _json
    from ami.storage.verifier import verify_checksum
    import hashlib
    with open(args.manifest_path) as f:
        manifest = _json.load(f)
    h = hashlib.sha256()
    with open(args.parquet_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    result = verify_checksum(expected_sha256=manifest.get("parquet_sha256", ""), actual_sha256=h.hexdigest())
    return {"state": result.state, "reasons": list(result.reasons)}


def _cmd_read(args: argparse.Namespace) -> dict:
    import json as _json
    from ami.storage.reader import read_partition
    with open(args.manifest_path) as f:
        manifest = _json.load(f)
    result = read_partition(parquet_path=args.parquet_path, manifest=manifest, requested_symbol=args.symbol)
    return {"row_count": result.row_count, "partition_id": result.partition_id,
            "verification_state": result.verification_state}


def _cmd_restore_slice(args: argparse.Namespace) -> dict:
    import json as _json
    from ami.storage.reader import read_partition
    from ami.storage.restorer import restore_slice
    from ami.storage.registry import get_table_spec
    with open(args.manifest_path) as f:
        manifest = _json.load(f)
    read_result = read_partition(parquet_path=args.parquet_path, manifest=manifest, requested_symbol=args.symbol)
    spec = get_table_spec(args.table)
    restore_result = restore_slice(
        destination_path=args.destination, spec=spec, rows=read_result.rows, manifest=manifest,
        expected_scientific_hash=manifest.get("ordered_scientific_content_hash", ""))
    return {"destination_path": restore_result.destination_path, "row_count": restore_result.row_count,
            "scientific_content_hash": restore_result.scientific_content_hash}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = args.func(args)
    except Exception as exc:  # fail closed: report, never silently succeed
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
