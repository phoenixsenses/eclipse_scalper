from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import report_check as rc


def test_report_check_passes_known_reports(monkeypatch) -> None:
    base = Path("reports/test_report_check")
    base.mkdir(parents=True, exist_ok=True)
    canonical = base / "validate_canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "status": "pass",
                "run_id": "abc123",
                "source": "data/canonical/canonical_merged.parquet",
                "violations": [],
                "column_stats": {},
                "invariant_summary": {"rows": 10, "violations": 0},
                "notes": [],
                "run_summary": {
                    "version": "v1",
                    "run_type": "validate_canonical",
                    "inputs": {"source": "data/canonical/canonical_merged.parquet"},
                    "metrics": {"status": "pass", "violation_count": 0, "row_count": 10},
                    "artifacts": {"json": "reports/validate_canonical_abc123.json", "md": "reports/validate_canonical_abc123.md"},
                },
            }
        ),
        encoding="utf-8",
    )
    summary_out = base / "report_check_summary.json"
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(canonical), "--out-json", str(summary_out)])
    assert rc.main() == 0
    payload = json.loads(summary_out.read_text(encoding="utf-8"))
    assert payload["summary"]["checked"] == 1
    assert payload["summary"]["fail_count"] == 0
    assert payload["run_summary"]["run_type"] == "report_check"


def test_report_check_fails_unknown_schema() -> None:
    base = Path("reports/test_report_check")
    base.mkdir(parents=True, exist_ok=True)
    unknown = base / "unknown.json"
    unknown.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    results, summary = rc.check_reports([unknown])
    assert summary["fail_count"] == 1
    assert results[0]["errors"] == ["unknown_schema:auto"]
