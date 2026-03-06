from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.spec import SignalSpec, specs_to_jsonl
from src.microphys.live.config import LiveSettings
from src.microphys.live.daemon import _resolve_artifacts, _resolve_exec_params, find_latest_completed_run, load_latest_model_specs
from src.microphys.live.registry import activate_artifacts


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"live_model_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mk_run(root: Path, name: str, status: str, signal_name: str) -> None:
    r = root / name
    r.mkdir(parents=True, exist_ok=True)
    (r / "manifest.json").write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")
    spec = SignalSpec(name=signal_name, side="buy", condition={"type": "gt", "op": "gt", "left": "x", "right": 0})
    cand = r / "cand.jsonl"
    cand.write_text(specs_to_jsonl([spec]), encoding="utf-8")
    sel = r / "sel.parquet"
    pd.DataFrame([{"signal": signal_name}]).to_parquet(sel, index=False)
    (r / "pointers.json").write_text(
        json.dumps({"candidates_deduped_jsonl": str(cand), "selected_parquet": str(sel)}) + "\n",
        encoding="utf-8",
    )


def test_latest_completed_run_is_deterministic() -> None:
    tmp = _mk_local_tmp()
    try:
        _mk_run(tmp, "run_20240101_000000_symbol=ETHUSDT_interval=100ms", "completed", "older")
        _mk_run(tmp, "run_20250101_000000_symbol=ETHUSDT_interval=100ms", "completed", "newer")
        _mk_run(tmp, "run_20260101_000000_symbol=ETHUSDT_interval=100ms", "failed", "failed")
        latest = find_latest_completed_run(tmp)
        assert latest is not None
        assert latest.name.startswith("run_20250101")
        specs, _ = load_latest_model_specs(tmp)
        assert len(specs) == 1
        assert specs[0].name == "newer"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_auto_execution_params_resolve_and_fallback() -> None:
    tmp = _mk_local_tmp()
    try:
        _mk_run(tmp, "run_20270101_000000_symbol=ETHUSDT_interval=100ms", "completed", "sig")
        run = tmp / "run_20270101_000000_symbol=ETHUSDT_interval=100ms"
        p = run / "exec_params.json"
        p.write_text(json.dumps({"maker_queue": {"queue_frac": 0.2}}) + "\n", encoding="utf-8")
        pointers = json.loads((run / "pointers.json").read_text(encoding="utf-8"))
        pointers["execution_params_json"] = str(p)
        (run / "pointers.json").write_text(json.dumps(pointers) + "\n", encoding="utf-8")

        cfg = LiveSettings(symbol="ETHUSDT", run_root=str(tmp), execution_params_path="")
        params, path_used, run_id, loaded = _resolve_exec_params(cfg, Path(str(cfg.run_root)))
        assert loaded is True
        assert run_id.startswith("run_20270101")
        assert path_used == str(p)
        assert "maker_queue" in params

        p.unlink(missing_ok=True)
        params2, path_used2, run_id2, loaded2 = _resolve_exec_params(cfg, Path(str(cfg.run_root)))
        assert loaded2 is False
        assert run_id2.startswith("run_20270101")
        assert path_used2 == str(p)
        assert params2 == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_active_artifacts_prefer_live_registry() -> None:
    tmp = _mk_local_tmp()
    try:
        run_root = tmp / "runs"
        live_root = tmp / "live"
        _mk_run(run_root, "run_20270101_000000_symbol=ETHUSDT_interval=100ms", "completed", "sig")
        run = run_root / "run_20270101_000000_symbol=ETHUSDT_interval=100ms"
        p_exec = tmp / "active_exec.json"
        p_exec.write_text(json.dumps({"maker_queue": {"queue_frac": 0.3}}) + "\n", encoding="utf-8")
        p_cal = tmp / "active_cal.json"
        p_cal.write_text(json.dumps({"quantiles": {"F_ofi_z": {"0.5000": 0.0}}, "nan_ratio": {"F_ofi_z": 0.0}, "sample_count": 10}) + "\n", encoding="utf-8")
        activate_artifacts(live_root=live_root, calibration_path=str(p_cal), execution_path=str(p_exec), metadata={"run_id": "active_run"})

        # Put a different pointer in latest run to ensure live registry is preferred.
        p_latest = run / "exec_latest.json"
        p_latest.write_text(json.dumps({"maker_queue": {"queue_frac": 0.9}}) + "\n", encoding="utf-8")
        pointers = json.loads((run / "pointers.json").read_text(encoding="utf-8"))
        pointers["execution_params_json"] = str(p_latest)
        (run / "pointers.json").write_text(json.dumps(pointers) + "\n", encoding="utf-8")

        cfg = LiveSettings(symbol="ETHUSDT", run_root=str(run_root), out_root=str(live_root), execution_params_path="", use_active_artifacts=True)
        art = _resolve_artifacts(cfg, run_root=run_root, live_root=live_root)
        assert str(art.get("execution_path_hint", "")) == str(p_exec)
        assert str(art.get("calibration_path", "")) == str(p_cal)
        assert str(art.get("execution_run_id", "")) == "active_run"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
