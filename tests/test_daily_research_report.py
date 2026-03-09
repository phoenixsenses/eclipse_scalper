from __future__ import annotations

import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import daily_research_report as drr


def _mk_local_tmp() -> Path:
    path = Path("localtests") / f"daily_research_report_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_daily_research_report_writes_markdown_and_json(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        telemetry = tmp / "telemetry.jsonl"
        now = time.time()
        telemetry.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "ts": now,
                            "event": "execution.belief_state",
                            "data": {
                                "runtime_gate_degraded": True,
                                "allow_entries": False,
                                "guard_mode": "ORANGE",
                                "guard_recovery_stage": "RUNTIME_GATE_DEGRADED",
                                "runtime_gate_reason": "coverage_gap=4",
                            },
                        }
                    ),
                    json.dumps({"ts": now, "event": "entry.blocked", "symbol": "ETHUSDT", "data": {"reason": "signal not present"}}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        telemetry_abs = telemetry.resolve()

        _write_json(
            tmp / "reports" / "ETH_POCKET_B_7D_BASELINE_SPLIT2.json",
            {"pass_count": 1, "per_split": [{"filled_avg_net_mean": -0.0002044, "attempt_fill_rate_mean": 0.7997}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json",
            {"pass_count": 3, "per_split": [{"filled_avg_net_mean": 0.0000902, "attempt_fill_rate_mean": 1.0}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_C_7D_PASSIVE_THEN_TAKER.json",
            {"pass_count": 3, "per_split": [{"filled_avg_net_mean": 0.0003791, "attempt_fill_rate_mean": 1.0}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_SOFT_7D_BASELINE.json",
            {"pass_count": 0, "per_split": [{"filled_avg_net_mean": -0.0002775, "attempt_fill_rate_mean": 0.5591}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_SOFT_7D_PASSIVE_THEN_TAKER.json",
            {"pass_count": 2, "per_split": [{"filled_avg_net_mean": 0.0000049, "attempt_fill_rate_mean": 1.0}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_MID_7D_BASELINE.json",
            {"pass_count": 0, "per_split": [{"filled_avg_net_mean": -0.0004311, "attempt_fill_rate_mean": 0.5776}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_MID_7D_PASSIVE_THEN_TAKER.json",
            {"pass_count": 0, "per_split": [{"filled_avg_net_mean": -0.0000493, "attempt_fill_rate_mean": 1.0}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_TIGHTMID_7D_BASELINE.json",
            {"pass_count": 0, "per_split": [{"filled_avg_net_mean": -0.0002028, "attempt_fill_rate_mean": 0.6073}]},
        )
        _write_json(
            tmp / "reports" / "ETH_POCKET_TIGHTMID_7D_PASSIVE_THEN_TAKER.json",
            {"pass_count": 3, "per_split": [{"filled_avg_net_mean": 0.0002131, "attempt_fill_rate_mean": 1.0}]},
        )

        monkeypatch.chdir(tmp)
        monkeypatch.setattr(
            drr,
            "check_gate",
            lambda **kwargs: {
                "symbol": "ETHUSDT",
                "gate": "BLOCKED",
                "reason": "active_lanes=book_proxy_pressure",
                "blocked_lanes": ["book_proxy_pressure"],
                "pocket": "h=60 imb>=0.85 int>=4000 spr<=0.000150",
                "profile": "event_block_eth_micro_imb085_v1",
            },
        )
        rc = drr.run_once(
            type(
                "Args",
                (),
                {
                    "date": "2026-03-13",
                    "out": "",
                    "db": "data/microstructure.db",
                    "symbol": "ETHUSDT",
                    "telemetry_path": str(telemetry_abs),
                    "recovery_lookback_min": 180,
                    "event_lookback_min": 60,
                    "event_bucket_sec": 5,
                    "event_stale_after_sec": 60,
                },
            )()
        )
        assert rc == 0
        md_out = Path("reports") / "DAILY_2026-03-13.md"
        json_out = Path("reports") / "DAILY_2026-03-13.json"
        assert md_out.exists()
        assert json_out.exists()
        text = md_out.read_text(encoding="utf-8")
        assert "Daily Research Report - 2026-03-13" in text
        assert "event lanes: `BLOCKED`" in text
        assert "regime recovery prep: `HOLD`" in text
        assert "pocket promotion checklist: `GO_EXPERIMENTAL`" in text
        assert "Pocket B" in text
        assert "Tight-mid" in text
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["report_date"] == "2026-03-13"
        assert payload["headline"]["event_lanes"] == "BLOCKED"
        assert payload["headline"]["regime_recovery_prep"] == "HOLD"
        assert payload["headline"]["pocket_promotion_checklist"] == "GO_EXPERIMENTAL"
        assert payload["promotion"]["promotable"] == ["Pocket B", "Pocket C", "Tight-mid"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_daily_research_report_uses_fixture_db_for_event_lane(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        telemetry = tmp / "telemetry.jsonl"
        now = time.time()
        telemetry.write_text(
            json.dumps(
                {
                    "ts": now,
                    "event": "execution.belief_state",
                    "data": {
                        "runtime_gate_degraded": False,
                        "allow_entries": True,
                        "guard_mode": "GREEN",
                        "guard_recovery_stage": "READY",
                        "runtime_gate_reason": "steady",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        fixture_db = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "microstructure_sample.db"
        monkeypatch.chdir(tmp)
        rc = drr.run_once(
            type(
                "Args",
                (),
                {
                    "date": "2026-03-10",
                    "out": "",
                    "db": str(fixture_db),
                    "symbol": "ETHUSDT",
                    "telemetry_path": str(telemetry.resolve()),
                    "recovery_lookback_min": 180,
                    "event_lookback_min": 60,
                    "event_bucket_sec": 5,
                    "event_stale_after_sec": 60,
                },
            )()
        )
        assert rc == 0
        payload = json.loads((Path("reports") / "DAILY_2026-03-10.json").read_text(encoding="utf-8"))
        assert payload["event_lane"]["status"] in {"CLEAR", "BLOCKED"}
        assert "no_data" not in payload["event_lane"]["summary"]
        assert payload["event_lane"]["raw"]["buckets_loaded"] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
