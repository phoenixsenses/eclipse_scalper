from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import activate_online_artifacts as tool


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"activate_hist_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_activate_history_contains_probe_stats(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        live = tmp / "live"
        cal = tmp / "cal.json"
        exe = tmp / "exe.json"
        val = tmp / "validate_report.json"
        cal.write_text(
            json.dumps(
                {
                    "quantiles": {
                        "F_ofi_z": {"0.5000": 0.0, "0.9000": 1.0},
                        "F_intensity_z": {"0.5000": 0.0, "0.9000": 1.0},
                        "spread_z": {"0.1000": -1.0, "0.5000": 0.0},
                    },
                    "nan_ratio": {"F_ofi_z": 0.0, "F_intensity_z": 0.0, "spread_z": 0.0},
                    "sample_count": 100,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        exe.write_text(
            json.dumps(
                {
                    "maker_hazard": {"a": 1.0, "b": -0.5, "c": 0.3, "d": 0.0, "fill_threshold": 0.5, "ttl_bars": 5},
                    "maker_queue": {"queue_frac": 0.2, "ttl_bars": 5, "min_depth": 1.0},
                    "adverse": {"buy_mean": 0.0, "sell_mean": 0.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        val.write_text(
            json.dumps(
                {
                    "calibration": {
                        "probe_sanity": {
                            "ok": True,
                            "errors": [],
                            "summary": {"total_density": 0.123, "probe_stats": [{"probe": "x", "density": 0.1}]},
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "activate_online_artifacts",
                "--calibration",
                str(cal),
                "--execution",
                str(exe),
                "--live-root",
                str(live),
                "--validation-report",
                str(val),
                "--run-id",
                "r_test",
            ],
        )
        assert tool.main() == 0
        rows = [json.loads(x) for x in (live / "calibration_history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert rows
        last = rows[-1]
        ps = dict(last.get("probe_summary", {}) or {})
        assert float(ps.get("total_density", 0.0)) == 0.123
        assert last.get("validation_passed") is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

