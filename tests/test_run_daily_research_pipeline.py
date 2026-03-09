from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_daily_research_pipeline as rdp


def test_run_pipeline_runs_daily_and_refresh(monkeypatch) -> None:
    out_dir = Path("localtests/test_run_daily_research_pipeline")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_once(args) -> int:
        report_date = "2026-03-13"
        md = reports_dir / f"DAILY_{report_date}.md"
        js = reports_dir / f"DAILY_{report_date}.json"
        md.write_text("# x\n", encoding="utf-8")
        js.write_text(
            json.dumps(
                {
                    "report_date": report_date,
                    "headline": {
                        "event_lanes": "CLEAR",
                        "regime_recovery_prep": "HOLD",
                        "pocket_promotion_checklist": "INCOMPLETE",
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(rdp.drr, "run_once", fake_run_once)
    monkeypatch.setattr(
        rdp.rdr,
        "build_refresh_payload",
        lambda **kwargs: {"summary": {"watchboard_top_lane": "spread_stress", "artifact_count": 15}},
    )

    args = rdp._parse_args(
        [
            "--date",
            "2026-03-13",
            "--reports-dir",
            str(reports_dir),
        ]
    )
    payload = rdp.run_pipeline(args)
    assert payload["report_date"] == "2026-03-13"
    assert payload["daily_report"]["headline"]["event_lanes"] == "CLEAR"
    assert payload["refresh"]["summary"]["artifact_count"] == 15


def test_main_writes_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        rdp,
        "run_pipeline",
        lambda args: {
            "report_date": "2026-03-13",
            "daily_report": {
                "headline": {
                    "event_lanes": "CLEAR",
                    "regime_recovery_prep": "HOLD",
                    "pocket_promotion_checklist": "INCOMPLETE",
                }
            },
            "refresh": {"summary": {"watchboard_top_lane": "spread_stress", "artifact_count": 15}},
        },
    )
    out_dir = Path("localtests/test_run_daily_research_pipeline_main")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "pipeline.json"
    out_md = out_dir / "pipeline.md"
    assert rdp.main(["--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    body = json.loads(out_json.read_text(encoding="utf-8"))
    assert body["refresh"]["summary"]["watchboard_top_lane"] == "spread_stress"
    assert out_md.exists()
