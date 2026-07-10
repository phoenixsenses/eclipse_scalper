"""Read-only, static-text reconciliation scanner for the range-read
consumer migration inventory (BATCH-STORAGE-ROTATION-RETENTION-RANGE-
READ-INVENTORY-RECONCILIATION-V1).

Supersedes (does not modify) `tools/storage_migration_inventory_scan.py`
for the specific purpose of answering "what is actually migrated /
no-op / blocked / remaining today" -- the original scanner has no
concept of "already migrated" beyond a hardcoded `ALREADY_MIGRATED` set
that was never updated past its original 5 ASOF pilots, so re-running
it (or reading its last JSON output) makes every one of the 16+ files
migrated across Range V1-V7 look identical to a genuinely-unmigrated
file: the oracle direct-SQL function each migration gate deliberately
KEEPS (as the parity-test reference) still contains the same literal
`ts_ms>=? ... ts_ms<=?` text the regex looks for.

This scanner adds exactly one new signal the old one didn't have:
`reader_v2_present` -- whether the file imports
`ami.storage.research_reader` (or `ami.storage.reader` for the point-
lookup track) at all. That single import is authoritative for "this
file has SOME reader-backed path", which combined with the git-tracked
"is this specific consumer on the known-migrated list" cross-check
(done in the reconciliation report, not here) is what actually answers
"migrated or not" -- static SQL-shape regexes alone cannot, since a
migrated file's oracle is supposed to keep looking exactly like an
unmigrated one.

Never imports or executes any scanned file. Never touches
microstructure.db, the archive, or any consumer file. Writes only its
own JSON output.

Usage: python -m tools.range_read_inventory_reconciliation_v1_scan
Output: reports/governance/storage/range_read_inventory_reconciliation_v1_raw_scan.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

ALLOWLIST_TABLES = {"mark_prices", "agg_trades", "book_ticker"}
OUT_OF_ALLOWLIST_TABLES = {"liquidations", "open_interest", "vol_state", "spot_prices",
                           "funding_history", "gaps", "ami_signal_lifecycle", "s34_trades",
                           "liq_event_features", "liq_event_outcome_labels"}

DO_NOT_TOUCH_NAME_PATTERNS = [
    "state_machine", "live_executor", "shadow_runner", "shadow_mirror", "shadow_observer",
    "shadow_paper", "shadow_control_plane", "shadow_dashboard", "shadow_forward_tracker",
    "live_chart", "dashboard", "scheduler", "live_order_executor", "live_preflight",
    "orderflow_chart", "replay", "runtime_status", "prepare_live_env", "v_engine",
    "v02_", "quarantine_monitor", "prereg_monitor", "guardrail_shadow", "intelligence_ledger",
    "execution_optimizer", "execution_management", "risk_sandbox", "risk_frontier",
    "live_mirror", "live_rule_profile", "live_trade_analysis", "prediction_risk",
]

# Backward ASOF: the direction research_reader.lookup_latest_at_or_before
# already supports ("latest row at-or-before").
DESC_LIMIT1_RE = re.compile(r"ORDER BY\s+ts_ms\s+DESC\s+LIMIT\s+1", re.I)
# Forward ASOF: the opposite, unsupported direction ("first row at-or-after").
ASC_LIMIT1_RE = re.compile(r"ORDER BY\s+ts_ms\s+ASC\s+LIMIT\s+1", re.I)
# Dynamic-direction ASOF (order chosen by an f-string / variable at runtime,
# e.g. `order by ts_ms {order}` -- can't tell direction statically; the
# reconciliation report resolves these by reading the call sites).
DYNAMIC_ORDER_RE = re.compile(r"order\s+by\s+ts_ms\s*\{", re.I)

RANGE_SCAN_RE = re.compile(r"ts_ms\s*[<>]=?\s*\?.{0,120}?ts_ms\s*[<>]=?\s*\?", re.I | re.S)
UNBOUNDED_MINMAX_RE = re.compile(r"SELECT\s+(MAX|MIN)\s*\(\s*ts_ms\s*\)\s+FROM", re.I)
FROM_TABLE_RE = re.compile(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)

# Per-SQL-statement extraction: for each `select` keyword occurrence,
# takes a bounded window of the following text so a range-scan or ASOF
# pattern can be correlated to the SPECIFIC table each statement
# targets -- the whole-file regex approach (matching "has an
# allowlisted table anywhere" and "has a range pattern anywhere"
# independently) produces false positives when a file queries an
# out-of-allowlist table (e.g. `liquidations`) with a bounded range AND
# an allowlisted table (e.g. `mark_prices`) with an unrelated ASOF
# lookup elsewhere -- the whole-file view wrongly looks like an
# allowlisted-table range candidate. Windows may overlap; that is fine
# for a superset/detection scan, not a claim of statement boundaries.
SELECT_KEYWORD_RE = re.compile(r"\bselect\b", re.I)
STATEMENT_WINDOW_CHARS = 400
FROM_TABLE_IN_WINDOW_RE = re.compile(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", re.I)
DB_FILE_RE = re.compile(r'"([a-zA-Z0-9_./\\]+\.(?:db|sqlite))"')
AMI_GOVERNANCE_IMPORT_RE = re.compile(r"from ami\.(knowledge|research|governance)")
READER_MODULE_IMPORTED_RE = re.compile(r"from ami\.storage import research_reader|import ami\.storage\.research_reader")
# Call-level (not just import-level) signal: this is what actually
# distinguishes "has a range-read migration" from "only has an ASOF/
# point-lookup migration but still imports the same reader module" --
# the two tracks share one import line, so import presence alone
# conflates them (a MIXED file could have one without the other).
EXECUTE_READ_CALL_RE = re.compile(r"\b(?:RR|research_reader)\.(?:plan_read|execute_read)\s*\(")
LOOKUP_CALL_RE = re.compile(r"\b(?:RR|research_reader)\.lookup_latest_at_or_before\s*\(|\blookup_latest_at_or_before\s*\(")
MAIN_GUARD_RE = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')
V2_FUNC_DEF_RE = re.compile(r"def\s+\w+_v2\s*\(")


def _tracked(relpath: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", relpath], cwd=ROOT,
                        capture_output=True)
    return r.returncode == 0


def _per_statement_signals(text: str) -> dict:
    """Correlates range-scan / ASOF patterns to the specific table each
    SQL statement targets, so a bounded range on an out-of-allowlist
    table (e.g. `liquidations`) is never counted as an allowlisted-table
    range candidate just because the file ALSO happens to reference
    `mark_prices` somewhere else (e.g. in an unrelated ASOF lookup)."""
    allowlist_range_tables: set[str] = set()
    ooa_range_tables: set[str] = set()
    allowlist_desc_asof_tables: set[str] = set()
    allowlist_asc_asof_tables: set[str] = set()
    allowlist_dynamic_asof_tables: set[str] = set()
    for m in SELECT_KEYWORD_RE.finditer(text):
        window = text[m.start():m.start() + STATEMENT_WINDOW_CHARS]
        table_m = FROM_TABLE_IN_WINDOW_RE.search(window)
        if not table_m:
            continue
        table = table_m.group(1).lower()
        is_allowlist = table in ALLOWLIST_TABLES
        is_ooa = table in OUT_OF_ALLOWLIST_TABLES
        has_range = bool(RANGE_SCAN_RE.search(window))
        has_desc1 = bool(DESC_LIMIT1_RE.search(window))
        has_asc1 = bool(ASC_LIMIT1_RE.search(window))
        has_dyn = bool(DYNAMIC_ORDER_RE.search(window))
        if has_range:
            if is_allowlist:
                allowlist_range_tables.add(table)
            elif is_ooa:
                ooa_range_tables.add(table)
        if is_allowlist and has_desc1:
            allowlist_desc_asof_tables.add(table)
        if is_allowlist and has_asc1:
            allowlist_asc_asof_tables.add(table)
        if is_allowlist and has_dyn:
            allowlist_dynamic_asof_tables.add(table)
    return {
        "allowlist_range_tables": sorted(allowlist_range_tables),
        "ooa_range_tables": sorted(ooa_range_tables),
        "allowlist_desc_asof_tables": sorted(allowlist_desc_asof_tables),
        "allowlist_asc_asof_tables": sorted(allowlist_asc_asof_tables),
        "allowlist_dynamic_asof_tables": sorted(allowlist_dynamic_asof_tables),
    }


def scan_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    tables = sorted(set(m.lower() for m in FROM_TABLE_RE.findall(text)))
    other_dbs = sorted(set(m for m in DB_FILE_RE.findall(text) if "microstructure" not in m))
    stmt = _per_statement_signals(text)
    return {
        "path": f"tools/{path.name}",
        "lines": lines,
        "tracked": _tracked(f"tools/{path.name}"),
        "mode_ro": "mode=ro" in text,
        "other_db_refs": other_dbs,
        "tables": tables,
        "allowlist_tables": sorted(t for t in tables if t in ALLOWLIST_TABLES),
        "out_of_allowlist_tables": sorted(t for t in tables if t in OUT_OF_ALLOWLIST_TABLES),
        "desc_limit1_count": len(DESC_LIMIT1_RE.findall(text)),
        "asc_limit1_count": len(ASC_LIMIT1_RE.findall(text)),
        "dynamic_order_asof_count": len(DYNAMIC_ORDER_RE.findall(text)),
        "range_scan_count": len(RANGE_SCAN_RE.findall(text)),
        "unbounded_minmax_count": len(UNBOUNDED_MINMAX_RE.findall(text)),
        # Per-statement, table-correlated signals (authoritative for
        # classification -- see _per_statement_signals docstring).
        "allowlist_range_tables": stmt["allowlist_range_tables"],
        "ooa_range_tables": stmt["ooa_range_tables"],
        "allowlist_desc_asof_tables": stmt["allowlist_desc_asof_tables"],
        "allowlist_asc_asof_tables": stmt["allowlist_asc_asof_tables"],
        "allowlist_dynamic_asof_tables": stmt["allowlist_dynamic_asof_tables"],
        "has_ami_governance_import": bool(AMI_GOVERNANCE_IMPORT_RE.search(text)),
        "reader_module_imported": bool(READER_MODULE_IMPORTED_RE.search(text)),
        "execute_read_call_present": bool(EXECUTE_READ_CALL_RE.search(text)),
        "lookup_call_present": bool(LOOKUP_CALL_RE.search(text)),
        # Authoritative "range-read migrated" signal for this reconciliation:
        # an actual plan_read/execute_read CALL, not just the shared import.
        "reader_v2_present": bool(EXECUTE_READ_CALL_RE.search(text)),
        "has_main_guard": bool(MAIN_GUARD_RE.search(text)),
        "v2_function_defined": bool(V2_FUNC_DEF_RE.search(text)),
        "do_not_touch_by_name": any(p in path.name for p in DO_NOT_TOUCH_NAME_PATTERNS),
    }


def main() -> int:
    results = []
    for path in sorted(TOOLS.glob("*.py")):
        text_probe = path.read_text(encoding="utf-8", errors="replace")
        if "microstructure.db" not in text_probe:
            continue
        results.append(scan_file(path))

    out_dir = ROOT / "reports" / "governance" / "storage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "range_read_inventory_reconciliation_v1_raw_scan.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"scanned {len(results)} candidate files -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
