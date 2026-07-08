"""Standalone, fresh-process worker for post-publish partition
verification (BATCH-STORAGE-PRODUCTION-ARCHIVE-REVERIFY-MEMORY-
HARDENING-V1).

Root cause this hardens against: `ami.storage.production_activation.
publish_authorized_production_partition_sharded` calls `SHA.
stream_hash_parquet_multi` TWICE within the same long-lived process --
once as the pre-publish reread parity check, once again (via
`reverify_authorized_partition_sharded`) as the post-publish reverify.
Each pass streams in RAM-bounded batches (never a full in-memory read),
but each batch still does millions of small Python allocations
(`repr()` strings, `to_pydict()` per-column lists) that CPython's
allocator does not reliably return to the OS between passes -- so
resident memory climbs across the two back-to-back passes even though
neither pass alone is unbounded. Confirmed against the real book_ticker/
SOLUSDT/2026-04 publish: RSS reached ~4.06GB during this phase (the
export loop itself, which HAS an RSS guard, never exceeded ~1GB).

Running verification as a byte-for-byte FRESH subprocess (started once,
exited once, per verification run) sidesteps allocator fragmentation
entirely -- the OS reclaims 100% of the subprocess's memory on exit
regardless of what Python's own allocator did or didn't return
internally. See `ami.storage.reverify_guard` for the parent-side runner
that spawns this worker and enforces a hard RSS ceiling on it from
outside (so the guard works even if the worker itself is too memory-
starved to check its own RSS).

This worker only ever reads (manifest.json, catalog_entry.json,
authorization_receipt.json, shard Parquet files) and, for the bounded
restore-slice check, writes ONE disposable SQLite file beneath an
approved temp root (cleaned up before exit) -- it never mutates the
source database, the shard files, the manifest, the catalog entry, or
the root catalog index.

Invoke as: python -m ami.storage.reverify_worker <final_dir> <result_path>
                 [--batch-size N] [--no-restore-check] [--restore-temp-root PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_partition(final_dir: str, *, batch_size: int = 1_000_000,
                      do_restore_check: bool = True,
                      restore_temp_root: str = ".runtime_temp") -> dict:
    """Read-only verification of an already-published sharded partition.
    Returns a structured result dict; never raises for a *found*
    mismatch (that is a normal, reportable outcome) -- only raises for a
    genuinely broken/missing input (manifest absent, table unregistered,
    etc.), which the caller (subprocess exit code) surfaces as a crash,
    distinct from a clean mismatch finding."""
    from ami.storage import sharded_archive as SHA
    from ami.storage import restorer as RS
    from ami.storage.registry import get_table_spec

    manifest_path = os.path.join(final_dir, "manifest.json")
    manifest = _read_json(manifest_path)
    spec = get_table_spec(manifest["source_table"])
    shards = sorted(manifest.get("shards") or [], key=lambda s: s["shard_index"])
    if not shards:
        raise ValueError(f"manifest at {manifest_path!r} has no shard inventory")

    shard_paths = [os.path.join(final_dir, s["shard_file"]) for s in shards]

    # 0. Structural/authorization invariants -- the same checks the old
    #    in-process `reverify_authorized_partition_sharded` performed,
    #    reproduced here so this worker is a complete superset (not a
    #    partial replacement) of that function's safety net. Cheap:
    #    JSON reads only, no Parquet I/O.
    structural_mismatches = []
    success_path = os.path.join(final_dir, "_SUCCESS")
    if not os.path.exists(success_path):
        structural_mismatches.append({"reason": "_SUCCESS file missing"})
    receipt_path = os.path.join(final_dir, "authorization_receipt.json")
    catalog_entry_path = os.path.join(final_dir, "catalog_entry.json")
    if os.path.exists(receipt_path) and os.path.exists(catalog_entry_path):
        receipt = _read_json(receipt_path)
        catalog_entry = _read_json(catalog_entry_path)
        receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        import hashlib as _hashlib
        expected_self_hash = _hashlib.sha256(
            json.dumps(receipt_body, indent=2, sort_keys=True, default=str).encode()).hexdigest()
        if "receipt_sha256" in receipt and receipt.get("receipt_sha256") != expected_self_hash:
            structural_mismatches.append({"reason": "receipt self-hash mismatch (tampered)"})
        if catalog_entry.get("authorization_receipt_sha256") != _sha256_file(receipt_path):
            structural_mismatches.append({"reason": "catalog_entry authorization_receipt_sha256 mismatch"})
        if receipt.get("action") != "CREATE_PRODUCTION_ARCHIVE_ONLY":
            structural_mismatches.append({"reason": f"receipt action is {receipt.get('action')!r}"})
        for field in ("purge_authorization", "scheduler_authorization", "vacuum_authorization"):
            if receipt.get(field) != "PROHIBITED":
                structural_mismatches.append({"reason": f"receipt {field} is not PROHIBITED"})
        if catalog_entry.get("purge_authorization") != "PROHIBITED":
            structural_mismatches.append({"reason": "catalog_entry purge_authorization is not PROHIBITED"})
        if catalog_entry.get("source_retention_status") != "SOURCE_PRESENT":
            structural_mismatches.append({"reason": "catalog_entry source_retention_status is not SOURCE_PRESENT"})
    else:
        structural_mismatches.append({"reason": "authorization_receipt.json or catalog_entry.json missing"})
    if manifest.get("production_status") != "PRODUCTION_VERIFIED":
        structural_mismatches.append({"reason": f"manifest production_status is {manifest.get('production_status')!r}"})
    if manifest.get("purge_authorization") != "PROHIBITED":
        structural_mismatches.append({"reason": "manifest purge_authorization is not PROHIBITED"})

    # 1. Direct-read checksum verification -- cheap, file-level, per shard.
    shard_mismatches = []
    for s, path in zip(shards, shard_paths):
        if not os.path.exists(path):
            shard_mismatches.append({"shard_index": s["shard_index"], "reason": "missing file"})
            continue
        actual = _sha256_file(path)
        expected = s.get("sha256")
        if actual != expected:
            shard_mismatches.append({"shard_index": s["shard_index"], "reason": "checksum mismatch",
                                      "expected": expected, "actual": actual})

    # 2. Post-export scientific reverify -- one fresh streaming pass, this
    #    process only (never a second pass in the same process). A shard
    #    corrupted badly enough that pyarrow cannot even parse it (as
    #    opposed to a checksum-mismatch on an otherwise-readable file) is
    #    reported as a finding here too, never allowed to crash the
    #    whole reverify -- catching corruption is this tool's job.
    try:
        reread = SHA.stream_hash_parquet_multi(shard_paths, tuple(spec.preserved_columns))
        reread_error = None
    except Exception as exc:  # noqa: BLE001 -- reported as a finding, not a crash
        reread = {"row_count": None, "scientific_content_hash": None}
        reread_error = f"{type(exc).__name__}: {exc}"
    aggregate_matches = (
        reread_error is None
        and reread["row_count"] == manifest["row_count"]
        and reread["scientific_content_hash"] == manifest["ordered_scientific_content_hash"]
    )

    # 3. Bounded restore-slice verification -- ONE shard only (the
    #    smallest, to keep this bounded regardless of partition size),
    #    into a disposable temp file that is removed before returning.
    restore_ok = None
    restore_shard_index = None
    restore_error = None
    if do_restore_check and shards:
        smallest = min(shards, key=lambda s: s["row_count"])
        restore_shard_index = smallest["shard_index"]
        shard_path = os.path.join(final_dir, smallest["shard_file"])
        os.makedirs(restore_temp_root, exist_ok=True)
        dest = os.path.join(restore_temp_root,
                             f"reverify_restore_check_shard{restore_shard_index:05d}.sqlite")
        if os.path.exists(dest):
            os.remove(dest)
        try:
            shard_hash = SHA.stream_hash_parquet_multi([shard_path], tuple(spec.preserved_columns))
            result = RS.stream_restore_slice_multi(
                destination_path=dest, spec=spec, parquet_paths=[shard_path],
                manifest={"ordered_scientific_content_hash": shard_hash["scientific_content_hash"]},
                expected_scientific_hash=shard_hash["scientific_content_hash"], batch_size=batch_size)
            restore_ok = (result.row_count == smallest["row_count"]
                          and result.scientific_content_hash == shard_hash["scientific_content_hash"])
        except Exception as exc:  # noqa: BLE001 -- reported as a finding, not a crash
            restore_ok = False
            restore_error = f"{type(exc).__name__}: {exc}"
        finally:
            if os.path.exists(dest):
                os.remove(dest)

    mismatch_count = len(structural_mismatches) + len(shard_mismatches) + \
        (0 if aggregate_matches else 1) + (0 if restore_ok in (None, True) else 1)

    return {
        "final_dir": final_dir,
        "shard_count": len(shards),
        "structural_mismatches": structural_mismatches,
        "shard_mismatches": shard_mismatches,
        "aggregate_row_count": reread["row_count"],
        "aggregate_hash": reread["scientific_content_hash"],
        "aggregate_reread_error": reread_error,
        "manifest_row_count": manifest["row_count"],
        "manifest_hash": manifest["ordered_scientific_content_hash"],
        "aggregate_matches_manifest": aggregate_matches,
        "restore_check_performed": do_restore_check,
        "restore_check_ok": restore_ok,
        "restore_check_shard_index": restore_shard_index,
        "restore_check_error": restore_error,
        "mismatch_count": mismatch_count,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_dir")
    parser.add_argument("result_path")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--no-restore-check", action="store_true")
    parser.add_argument("--restore-temp-root", default=".runtime_temp")
    args = parser.parse_args(argv)

    try:
        result = verify_partition(
            args.final_dir, batch_size=args.batch_size,
            do_restore_check=not args.no_restore_check,
            restore_temp_root=args.restore_temp_root)
    except Exception as exc:  # noqa: BLE001 -- genuine crash, surfaced via exit code + traceback
        with open(args.result_path, "w", encoding="utf-8") as f:
            json.dump({"crashed": True, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}, f, indent=2)
        return 1

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
