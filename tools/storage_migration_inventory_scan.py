"""Read-only, static-text inventory scanner for remaining research-
consumer storage-reader migration candidates (BATCH-STORAGE-ROTATION-
RETENTION-RESEARCH-CONSUMER-MIGRATION-PLAN-V1).

Never imports or executes any scanned file -- pure regex/text analysis
over each tools/*.py source file. No database access, no writes
outside its own JSON output. Produces the raw structural signals used
to classify each remaining candidate in
reports/governance/storage/RESEARCH_CONSUMER_MIGRATION_PLAN_V1.md; the
final category/risk assignment in that document also incorporates
manual spot-checks, not just these heuristics.

Usage: python -m tools.storage_migration_inventory_scan
Output: reports/governance/storage/research_consumer_migration_inventory_v1.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

ALREADY_MIGRATED = {
    "research_ami_mfe50_experiment.py",
    "research_s34_100k_notmon_check.py",
    "research_s34_consensus_composite.py",
    "research_s34_btc_microtrend_eth_quality.py",
    "research_s34_btc_microtrend_sweep.py",
}

ALLOWLIST_TABLES = {"mark_prices", "agg_trades", "book_ticker"}
OUT_OF_ALLOWLIST_TABLES = {"liquidations", "open_interest", "vol_state", "spot_prices",
                           "funding_history", "gaps", "ami_signal_lifecycle", "s34_trades"}

DO_NOT_TOUCH_NAME_PATTERNS = [
    "state_machine", "live_executor", "shadow_runner", "shadow_mirror", "shadow_observer",
    "shadow_paper", "shadow_control_plane", "shadow_dashboard", "shadow_forward_tracker",
    "live_chart", "dashboard", "scheduler", "live_order_executor", "live_preflight",
    "orderflow_chart", "replay", "runtime_status", "prepare_live_env", "v_engine",
    "v02_", "quarantine_monitor", "prereg_monitor", "guardrail_shadow", "intelligence_ledger",
    "execution_optimizer", "execution_management", "risk_sandbox", "risk_frontier",
    "live_mirror", "live_rule_profile", "live_trade_analysis", "prediction_risk",
]

DESC_LIMIT1_RE = re.compile(r"ORDER BY\s+ts_ms\s+DESC\s+LIMIT\s+1", re.I)
RANGE_SCAN_RE = re.compile(r"ts_ms\s*[<>]=?\s*\?.{0,80}?ts_ms\s*[<>]=?\s*\?", re.I | re.S)
FROM_TABLE_RE = re.compile(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)
DB_FILE_RE = re.compile(r'"([a-zA-Z0-9_]+\.db)"')
AMI_GOVERNANCE_IMPORT_RE = re.compile(r"from ami\.(knowledge|research|governance)")


def _tracked(relpath: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", relpath], cwd=ROOT,
                        capture_output=True)
    return r.returncode == 0


def scan_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    tables = sorted(set(m.lower() for m in FROM_TABLE_RE.findall(text)))
    return {
        "path": f"tools/{path.name}",
        "lines": lines,
        "tracked": _tracked(f"tools/{path.name}"),
        "mode_ro": "mode=ro" in text,
        "dbs": sorted(set(DB_FILE_RE.findall(text))),
        "tables": tables,
        "allowlist_tables": sorted(t for t in tables if t in ALLOWLIST_TABLES),
        "out_of_allowlist_tables": sorted(t for t in tables if t in OUT_OF_ALLOWLIST_TABLES),
        "desc_limit1_count": len(DESC_LIMIT1_RE.findall(text)),
        "range_scan_count": len(RANGE_SCAN_RE.findall(text)),
        "has_ami_governance_import": bool(AMI_GOVERNANCE_IMPORT_RE.search(text)),
        "random_without_seed": ("import random" in text) and (".seed(" not in text and "Random(" not in text),
        "writes_report": bool(re.search(r"reports[\\/]research[\\/]s34|\bOJ\s*=|\bOM\s*=", text)),
        "do_not_touch_by_name": any(p in path.name for p in DO_NOT_TOUCH_NAME_PATTERNS),
    }


def main() -> int:
    results = []
    for path in sorted(TOOLS.glob("*.py")):
        if path.name in ALREADY_MIGRATED:
            continue
        text_probe = path.read_text(encoding="utf-8", errors="replace")
        if "microstructure.db" not in text_probe:
            continue
        results.append(scan_file(path))

    out_dir = ROOT / "reports" / "governance" / "storage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "research_consumer_migration_inventory_v1.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"scanned {len(results)} candidate files -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
