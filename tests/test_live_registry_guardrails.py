from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.live.guardrails import validate_calibration_payload, validate_execution_params_payload
from src.microphys.live.registry import activate_artifacts, get_active_artifacts, rollback_to_previous


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"live_registry_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_registry_activate_and_rollback() -> None:
    tmp = _mk_local_tmp()
    try:
        live = tmp / "live"
        c1 = tmp / "cal1.json"
        c2 = tmp / "cal2.json"
        e1 = tmp / "exe1.json"
        c1.write_text(json.dumps({"v": 1}) + "\n", encoding="utf-8")
        c2.write_text(json.dumps({"v": 2}) + "\n", encoding="utf-8")
        e1.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
        activate_artifacts(live_root=live, calibration_path=str(c1), execution_path=str(e1), metadata={"run_id": "r1"})
        a1 = get_active_artifacts(live)
        assert a1.get("calibration_json_path") == str(c1)
        activate_artifacts(live_root=live, calibration_path=str(c2), metadata={"run_id": "r2"})
        a2 = get_active_artifacts(live)
        assert a2.get("calibration_json_path") == str(c2)
        rb = rollback_to_previous("calibration", live_root=live)
        assert rb.get("calibration_json_path") == str(c1)
        assert (live / "calibration_history.jsonl").exists()
        assert (live / "execution_params_history.jsonl").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guardrails_reject_malformed_payloads() -> None:
    bad_cal = {"quantiles": {"F_ofi_z": {"0.5000": 2.0, "0.9000": 1.0}}, "nan_ratio": {"F_ofi_z": 0.1}, "sample_count": 10}
    okc, errc = validate_calibration_payload(bad_cal)
    assert okc is False
    assert any("non_monotone_quantiles" in e for e in errc)

    bad_exe = {"maker_hazard": {"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.0, "fill_threshold": 2.0, "ttl_bars": -1}}
    oke, erre = validate_execution_params_payload(bad_exe)
    assert oke is False
    assert any("fill_threshold_out_of_bounds" in e for e in erre)

