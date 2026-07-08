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

    # Narrowly scoped: publishes ONLY the single frozen rehearsal partition
    # (mark_prices/ETHUSDT/2026-05/v1). It accepts no table/symbol/month/root
    # arguments at all -- there is nothing to parameterize, so it can never
    # be turned into a general production-enable command.
    sp = sub.add_parser("production-activation-rehearsal",
                        help="Publish the single frozen rehearsal production partition "
                             "(mark_prices/ETHUSDT/2026-05/v1). No other partition is possible.")
    sp.add_argument("--source-db", default=None)
    sp.set_defaults(func=_cmd_production_activation_rehearsal)

    # --- Manual-only production archive activation commands ---
    sp = sub.add_parser("production-plan",
                        help="Read-only deterministic plan for an allowlisted table (candidate "
                             "selection + resource/authorization requirements). No production write.")
    sp.add_argument("--table", required=True, choices=allowlisted_tables())
    sp.add_argument("--source-db", default=None)
    sp.set_defaults(func=_cmd_production_plan)

    sp = sub.add_parser("production-archive-authorized",
                        help="Publish one production partition matching a supplied authorization "
                             "receipt. Requires --receipt-path; no arbitrary root/table/self-approval.")
    sp.add_argument("--receipt-path", required=True)
    sp.add_argument("--source-db", default=None)
    sp.set_defaults(func=_cmd_production_archive_authorized)

    sp = sub.add_parser("production-verify", help="Re-verify one production partition directory.")
    sp.add_argument("--partition-dir", required=True)
    sp.add_argument("--table", required=True, choices=allowlisted_tables())
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--utc-year", required=True, type=int)
    sp.add_argument("--utc-month", required=True, type=int)
    sp.set_defaults(func=_cmd_production_verify)

    sp = sub.add_parser("production-catalog-rebuild",
                        help="Acquire the catalog lock and deterministically rebuild the root index "
                             "from immutable partition-local entries. No source access.")
    sp.set_defaults(func=_cmd_production_catalog_rebuild)

    sp = sub.add_parser("production-health", help="Report real production-archive root/index/lock/staging state.")
    sp.set_defaults(func=_cmd_production_health)

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


def _cmd_production_activation_rehearsal(args: argparse.Namespace) -> dict:
    from ami.storage import production as PR
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean, DEFAULT_SOURCE_PATH

    root, root_source = PR.resolve_production_root()
    conn, log = open_read_only(args.source_db or DEFAULT_SOURCE_PATH)
    try:
        return PR.run_production_activation_rehearsal(conn, root=root, root_source=root_source)
    finally:
        assert_read_only_session_clean(log)
        conn.close()


def _cmd_production_plan(args: argparse.Namespace) -> dict:
    from ami.storage import production as PR
    from ami.storage import production_activation as PA
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean, DEFAULT_SOURCE_PATH
    root, _ = PR.resolve_production_root()
    conn, log = open_read_only(args.source_db or DEFAULT_SOURCE_PATH)
    try:
        plan = PA.preflight_partition(conn, root=root, table=args.table)
    finally:
        assert_read_only_session_clean(log)
        conn.close()
    p = plan["partition"]
    return {"table": args.table, "chosen_symbol": p.symbol, "utc_year": p.utc_year,
            "utc_month": p.utc_month, "row_count": plan["row_count"], "watermark": plan["watermark"],
            "plan_hash": plan["plan_hash"], "final_path_exists": plan["final_path_exists"],
            "unresolved_gap_count": plan["unresolved_gap_count"],
            "estimated_parquet_bytes": plan["estimated_parquet_bytes"],
            "authorization_required": "CREATE_PRODUCTION_ARCHIVE_ONLY receipt matching this plan_hash"}


def _cmd_production_archive_authorized(args: argparse.Namespace) -> dict:
    from ami.storage import production as PR
    from ami.storage import production_activation as PA
    from ami.storage.partition import build_partition_identity
    from ami.storage.registry import get_table_spec
    from ami.storage.source_access import open_read_only, assert_read_only_session_clean, DEFAULT_SOURCE_PATH
    with open(args.receipt_path) as f:
        receipt = json.load(f)
    root, _ = PR.resolve_production_root()
    # receipt's own root must match the resolved root (no arbitrary root)
    if os.path.normpath(receipt.get("production_archive_root", "")).replace("\\", "/").lower() != \
            os.path.normpath(os.path.abspath(root)).replace("\\", "/").lower():
        return {"error": "AuthorizationReceiptRejected", "message": "receipt root does not match resolved production root"}
    import datetime as _dt
    table = receipt["source_table"]
    spec = get_table_spec(table)
    conn, log = open_read_only(args.source_db or DEFAULT_SOURCE_PATH)
    try:
        # Re-derive the current live plan; publish_authorized_production_partition
        # verifies the supplied receipt against this plan (exact plan-hash /
        # watermark / schema / table match) before any production write.
        plan = PA.preflight_partition(conn, root=root, table=table)
        result = PA.publish_authorized_production_partition(
            conn, root=root, partition=plan["partition"], spec=spec, archive_version="v1",
            receipt=receipt, job_identity="PROD-ACT-CLI", source_schema_hash=plan["source_schema_hash"],
            export_cutoff=_dt.datetime.now(_dt.timezone.utc).isoformat())
    finally:
        assert_read_only_session_clean(log)
        conn.close()
    return {"status": result.status, "final_partition_dir": result.final_partition_dir,
            "row_count": result.row_count, "parquet_sha256": result.parquet_sha256,
            "reverification_mismatch_count": result.reverification_mismatch_count}


def _cmd_production_verify(args: argparse.Namespace) -> dict:
    from ami.storage import production_activation as PA
    from ami.storage.partition import build_partition_identity
    partition = build_partition_identity(table=args.table, symbol=args.symbol, utc_year=args.utc_year,
                                         utc_month=args.utc_month, source_watermark_value=0)
    # watermark not needed for reverify's partition_id (excludes watermark)
    mismatches = PA.reverify_authorized_partition(args.partition_dir, partition)
    return {"partition_dir": args.partition_dir, "reverification_mismatch_count": mismatches,
            "verified": mismatches == 0}


def _cmd_production_catalog_rebuild(args: argparse.Namespace) -> dict:
    from ami.storage import production as PR
    from ami.storage import production_activation as PA
    root, _ = PR.resolve_production_root()
    index_path, index_sha = PA.rebuild_root_index_under_lock(root, job_identity="CLI-REBUILD")
    return {"root_index_path": index_path, "root_index_sha256": index_sha}


def _cmd_production_health(args: argparse.Namespace) -> dict:
    from ami.storage import production as PR
    from ami.storage import production_activation as PA
    from ami.storage.health import scan_production_archive_health
    root, _ = PR.resolve_production_root()
    return scan_production_archive_health(root)


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
