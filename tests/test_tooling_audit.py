from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import tooling_audit as ta


def test_tooling_audit_classifies_dev_only_helpers() -> None:
    manifest = ta.build_manifest(Path(".").resolve())
    cut_candidates = set(manifest["cut_candidates"])
    assert "tools/_inspect_rank_json.py" in cut_candidates or "tools/dev/_inspect_rank_json.py" in cut_candidates
    assert manifest["summary"]["tool_count"] >= 1


def test_tooling_audit_detects_run_summary_gap() -> None:
    root = Path("localtests/tooling_audit_fixture")
    tools_dir = root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "report_gap.py").write_text('p.add_argument("--out-json")\n', encoding="utf-8")
    manifest = ta.build_manifest(root.resolve())
    assert "tools/report_gap.py" in manifest["run_summary_gaps"]
