"""Read-only classification pass over
`range_read_inventory_reconciliation_v1_raw_scan.json` (BATCH-STORAGE-
ROTATION-RETENTION-RANGE-READ-INVENTORY-RECONCILIATION-V1). Produces
`range_read_inventory_reconciliation_v1.json`, the reconciled,
per-file, per-category inventory referenced by
`RANGE_READ_INVENTORY_RECONCILIATION_V1.md`.

Never imports or executes a scanned file. Reads only its own upstream
JSON and (for two named special cases with prior manual review already
recorded in this docstring/comments below) applies a documented,
inline manual override where the automated per-statement regex signal
alone would be ambiguous or wrong. No other manual overrides are
applied -- every other file's classification is 100% derived from the
raw scan's per-statement, table-correlated signals.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "reports" / "governance" / "storage" / "range_read_inventory_reconciliation_v1_raw_scan.json"
OUT_PATH = ROOT / "reports" / "governance" / "storage" / "range_read_inventory_reconciliation_v1.json"

# Manual overrides, each backed by a full manual read of the file (not
# just the static regex signals) during this reconciliation gate.
MANUAL_OVERRIDES = {
    "tools/research_s34_cross_symbol_lag.py": {
        "classification": "NO_OP_NO_ALLOWLISTED_RANGE_PATTERN",
        "recommended_next_action": "none -- confirmed no-op for range-read",
        "recommended_gate_if_any": None,
        "manual_review_note": (
            "Manually read in full. `follower_liq_nearby()` has the file's ONLY bounded "
            "ts_ms>=?/ts_ms<=? range pattern, and it targets `liquidations` (out-of-allowlist), "
            "confirmed by the per-statement scan (`ooa_range_tables=['liquidations']`, "
            "`allowlist_range_tables=[]`). `mark_at()` is the file's only allowlisted-table "
            "(`mark_prices`) query and it is a dynamic-direction ASOF point-lookup "
            "(`order by ts_ms {order}`, called with `before=False` i.e. forward/ASC), not a "
            "range scan. Matches the prior V3 finding stated in the operator's prompt: "
            "'no allowlisted range-read; only ASOF mark_at + liquidations.' Confirmed, not "
            "just trusted."
        ),
    },
    "tools/walkforward_eval.py": {
        "classification": "NO_OP_EXECUTE_READ_CONTRACT_MISMATCH",
        "recommended_next_action": "none -- execute_read requires a single resolved symbol per plan_read() call; this file's query is multi-symbol with client-side post-filtering",
        "recommended_gate_if_any": "would require a NEW multi-symbol-aware reader helper -- explicitly out of scope for any range-read gate to date",
        "manual_review_note": (
            "Manually read in full. `_slice_price_window()` has the file's only allowlisted-"
            "table range pattern (`agg_trades`, confirmed by "
            "`allowlist_range_tables=['agg_trades']`): `SELECT ts_ms, symbol, price FROM "
            "agg_trades WHERE ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC, rowid ASC` with NO "
            "`symbol=?` predicate in the SQL at all -- it fetches every symbol in the window "
            "and filters `symbol_set` client-side in Python afterward. "
            "`ami.storage.research_reader.plan_read()`'s contract requires a single resolved "
            "`symbol` parameter per call (it is the catalog-matching key); this file's shape "
            "does not fit without either calling `plan_read` once per requested symbol (a real "
            "behavior change: current code makes exactly one SQL query per slice regardless of "
            "symbol count) or a new multi-symbol reader primitive (explicitly not authorized in "
            "any range-read gate). Also has `mode_ro=False` (opens via plain `sqlite3.connect`, "
            "no `?mode=ro` URI) and imports from `tools.eval_run`/`tools.replay_strategy` "
            "(both live/replay-adjacent) -- an import-safety/mode=ro prep step would be a "
            "SEPARATE gate even if the contract-mismatch were resolved. Matches the prior V6/V7 "
            "finding referenced in the operator's prompt."
        ),
    },
}

# Files where the ASOF portion is already migrated (lookup_call_present)
# but a genuinely separate, allowlisted-table range-read pattern remains
# -- this is the previously-documented MIXED_PARTIAL_MIGRATION bucket
# (plan doc §4.5): a prior gate deliberately left this specific range
# query on direct SQL (documented inline in that file at the time), not
# an oversight. Still counted as a real, open range-read opportunity.
KNOWN_MIXED_PARTIAL_NOTE = (
    "ASOF portion already migrated in a prior gate (lookup_call_present=True); this file's "
    "remaining allowlisted range-read pattern was deliberately left on direct SQL at that time "
    "(plan doc §4.5 MIXED_PARTIAL_MIGRATION precedent) -- a genuine, separate range-read "
    "migration opportunity, not an overlooked file."
)


def classify(e: dict) -> dict:
    path = e["path"]
    if path in MANUAL_OVERRIDES:
        ov = MANUAL_OVERRIDES[path]
        rec = dict(e)
        rec["classification"] = ov["classification"]
        rec["recommended_next_action"] = ov["recommended_next_action"]
        rec["recommended_gate_if_any"] = ov["recommended_gate_if_any"]
        rec["manual_review_note"] = ov["manual_review_note"]
        rec["classification_basis"] = "MANUAL_REVIEW_OVERRIDE"
        return rec

    rec = dict(e)
    rec["manual_review_note"] = None

    if e["do_not_touch_by_name"]:
        rec["classification"] = "DO_NOT_TOUCH_BY_NAME"
        rec["recommended_next_action"] = "none -- excluded by filename-pattern guardrail"
        rec["recommended_gate_if_any"] = None
        rec["classification_basis"] = "AUTOMATED"
        return rec

    if e["reader_v2_present"]:
        rec["classification"] = "MIGRATED_RANGE_READ"
        rec["recommended_next_action"] = "none -- reader-backed execute_read/plan_read call already present"
        rec["recommended_gate_if_any"] = None
        rec["classification_basis"] = "AUTOMATED"
        return rec

    allow_range = e["allowlist_range_tables"]
    ooa_range = e["ooa_range_tables"]
    allow_desc = e["allowlist_desc_asof_tables"]
    allow_asc = e["allowlist_asc_asof_tables"]
    allow_dyn = e["allowlist_dynamic_asof_tables"]
    other_dbs = e["other_db_refs"]

    if allow_range:
        rec["classification"] = "REMAINING_MIGRATABLE_RANGE_READ_CANDIDATE"
        rec["recommended_next_action"] = (
            f"manual review + migrate: bounded allowlisted range pattern on {allow_range}"
        )
        rec["recommended_gate_if_any"] = "RANGE-READ-CONSUMER-MIGRATION-V8-or-later"
        rec["classification_basis"] = "AUTOMATED"
        if e["lookup_call_present"]:
            rec["manual_review_note"] = KNOWN_MIXED_PARTIAL_NOTE
            rec["mixed_partial_migration"] = True
        else:
            rec["mixed_partial_migration"] = False
        return rec

    if not e["allowlist_tables"]:
        if other_dbs:
            rec["classification"] = "BLOCKED_DIFFERENT_DB"
            rec["recommended_next_action"] = f"none -- targets non-allowlisted DB(s): {other_dbs}"
            rec["recommended_gate_if_any"] = None
        else:
            rec["classification"] = "NO_OP_NO_ALLOWLISTED_RANGE_PATTERN"
            rec["recommended_next_action"] = "none -- no mark_prices/agg_trades/book_ticker reference at all"
            rec["recommended_gate_if_any"] = None
        rec["classification_basis"] = "AUTOMATED"
        return rec

    # Has an allowlisted table somewhere, but no allowlisted-table range
    # pattern -- check for ASOF shapes on that same table.
    if allow_asc or allow_dyn:
        rec["classification"] = "BLOCKED_FORWARD_ASOF_PRIMITIVE"
        rec["recommended_next_action"] = (
            "manual direction check required -- ASC or dynamic-direction ORDER BY ts_ms on an "
            "allowlisted table; if forward (\"first row at-or-after\"), lookup_latest_at_or_before "
            "does not support this direction and no primitive exists yet"
        )
        rec["recommended_gate_if_any"] = "FORWARD-ASOF-PRIMITIVE-PREP (not started, not authorized here)"
        rec["classification_basis"] = "AUTOMATED"
        return rec

    if allow_desc and not e["lookup_call_present"]:
        rec["classification"] = "ASOF_ONLY_NOT_RANGE_READ_SCOPE"
        rec["recommended_next_action"] = (
            "out of scope for a range-read gate -- has a backward ASOF pattern "
            "(ORDER BY ts_ms DESC LIMIT 1) on an allowlisted table but no range scan; would "
            "belong to an ASOF/point-lookup migration gate, not V8"
        )
        rec["recommended_gate_if_any"] = "ASOF-LOOKUP-CONSUMER-MIGRATION (new wave, if operator wants it)"
        rec["classification_basis"] = "AUTOMATED"
        return rec

    if e["unbounded_minmax_count"] > 0:
        rec["classification"] = "OUT_OF_SCOPE_UNBOUNDED_SCAN"
        rec["recommended_next_action"] = "none -- only unbounded MAX/MIN(ts_ms) probe, no bounded window"
        rec["recommended_gate_if_any"] = None
        rec["classification_basis"] = "AUTOMATED"
        return rec

    rec["classification"] = "NO_OP_NO_ALLOWLISTED_RANGE_PATTERN"
    rec["recommended_next_action"] = "none -- allowlisted table referenced but no range/ASOF pattern correlated to it"
    rec["recommended_gate_if_any"] = None
    rec["classification_basis"] = "AUTOMATED"
    return rec


def main() -> int:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    records = [classify(e) for e in raw]

    counts: dict[str, int] = {}
    for r in records:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    payload = {
        "scan_source": str(RAW_PATH.relative_to(ROOT)).replace("\\", "/"),
        "total_scanned": len(records),
        "category_counts": counts,
        "records": records,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"classified {len(records)} files -> {OUT_PATH}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
