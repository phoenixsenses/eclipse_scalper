from __future__ import annotations

import json
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_canary_expansion_gate import evaluate_gate, _parse_daily_scores


def _write_daily(path: Path, score: float, name: str = "Latency Modeling Drift") -> None:
    payload = {
        "causes": [
            {"name": name, "score": float(score), "evidence": [], "actions": []},
            {"name": "Other", "score": 0.1, "evidence": [], "actions": []},
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _mk_tmp_dir() -> Path:
    p = Path("localtests") / f"gate_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_gate_go_when_7_days_below_threshold() -> None:
    tmp_path = _mk_tmp_dir()
    for i in range(1, 8):
        _write_daily(tmp_path / f"2026-03-0{i}_LIVE_FILL_DRIFT_ROOT_CAUSE.json", score=0.42)
    rows = _parse_daily_scores(tmp_path)
    passed, detail = evaluate_gate(rows, window_days=7, max_top_score=0.5)
    assert passed is True
    assert int(detail["days_observed"]) == 7


def test_gate_hold_when_any_day_above_threshold() -> None:
    tmp_path = _mk_tmp_dir()
    for i in range(1, 7):
        _write_daily(tmp_path / f"2026-03-0{i}_LIVE_FILL_DRIFT_ROOT_CAUSE.json", score=0.30)
    _write_daily(tmp_path / "2026-03-07_LIVE_FILL_DRIFT_ROOT_CAUSE.json", score=0.71)
    rows = _parse_daily_scores(tmp_path)
    passed, detail = evaluate_gate(rows, window_days=7, max_top_score=0.5)
    assert passed is False
    assert bool(detail["score_ok"]) is False


def test_main_writes_run_summary(monkeypatch) -> None:
    tmp_path = _mk_tmp_dir()
    for i in range(1, 8):
        _write_daily(tmp_path / f"2026-03-0{i}_LIVE_FILL_DRIFT_ROOT_CAUSE.json", score=0.42)
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--report-dir", str(tmp_path), "--window-days", "7", "--max-top-score", "0.5", "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    from tools import evaluate_canary_expansion_gate as ecg

    assert ecg.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "evaluate_canary_expansion_gate"
