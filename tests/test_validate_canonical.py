from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_canonical as vc


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_pass_clean_synthetic(monkeypatch) -> None:
    src = Path("reports/test_validate_canonical/pass_clean.csv")
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
            "mid": [100.0, 100.1, 200.0, 200.1],
            "spread": [0.1, 0.1, 0.2, 0.2],
            "volume": [1.0, 2.0, 1.5, 1.2],
        }
    )
    _write_csv(src, df)
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(src), "--reports-dir", "reports"])
    rc = vc.main()
    assert rc == 0
    run_id = vc._stable_run_id(str(src), vc.DEFAULT_NAN_THRESHOLD)
    out = Path(f"reports/validate_canonical_{run_id}.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"


def test_fail_duplicate_timestamp(monkeypatch) -> None:
    src = Path("reports/test_validate_canonical/fail_dup.csv")
    df = pd.DataFrame(
        {
            "timestamp": [1, 1, 2],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "mid": [100.0, 100.1, 100.2],
            "spread": [0.1, 0.1, 0.1],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    _write_csv(src, df)
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(src), "--reports-dir", "reports"])
    rc = vc.main()
    assert rc == 3
    run_id = vc._stable_run_id(str(src), vc.DEFAULT_NAN_THRESHOLD)
    payload = json.loads(Path(f"reports/validate_canonical_{run_id}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert any(v.get("code") == "duplicate_timestamp_per_symbol" for v in payload["violations"])


def test_fail_missing_required_column(monkeypatch) -> None:
    src = Path("reports/test_validate_canonical/fail_missing_col.csv")
    df = pd.DataFrame({"timestamp": [1, 2], "mid": [1.0, 1.1]})
    _write_csv(src, df)
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(src), "--reports-dir", "reports"])
    rc = vc.main()
    assert rc == 3
    run_id = vc._stable_run_id(str(src), vc.DEFAULT_NAN_THRESHOLD)
    payload = json.loads(Path(f"reports/validate_canonical_{run_id}.json").read_text(encoding="utf-8"))
    assert any(v.get("code") == "missing_symbol_col" for v in payload["violations"])


def test_fail_nan_threshold(monkeypatch) -> None:
    src = Path("reports/test_validate_canonical/fail_nan.csv")
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "mid": [100.0, None, None, 100.3],
            "spread": [0.1, 0.1, 0.1, 0.1],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    _write_csv(src, df)
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--in", str(src), "--reports-dir", "reports", "--nan-threshold", "0.25"],
    )
    rc = vc.main()
    assert rc == 3
    run_id = vc._stable_run_id(str(src), 0.25)
    payload = json.loads(Path(f"reports/validate_canonical_{run_id}.json").read_text(encoding="utf-8"))
    assert any(v.get("code") == "nan_ratio_above_threshold" for v in payload["violations"])


def test_skip_missing_source(monkeypatch) -> None:
    src = Path("reports/test_validate_canonical/does_not_exist.csv")
    monkeypatch.setattr(sys, "argv", ["x", "--in", str(src), "--reports-dir", "reports"])
    rc = vc.main()
    assert rc == 0
    run_id = vc._stable_run_id(str(src), vc.DEFAULT_NAN_THRESHOLD)
    payload = json.loads(Path(f"reports/validate_canonical_{run_id}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skip"
    assert "skipped_missing_data" in payload.get("notes", [])

