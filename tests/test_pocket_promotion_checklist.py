"""tests/test_pocket_promotion_checklist.py"""
import json
import tempfile
from pathlib import Path
import pytest
import tools.pocket_promotion_checklist as pc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(directory: str, name: str, data: dict) -> str:
    path = Path(directory) / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _patch_all_pass(monkeypatch):
    monkeypatch.setattr(pc, "_call_check_gate", lambda db, symbol: {"gate": "ALLOWED", "lanes": {}})
    monkeypatch.setattr(pc, "check_market_regime", lambda db, symbol: {"name": "market_regime", "result": "PASS", "detail": "ok"})


def _patch_gate_fail(monkeypatch):
    monkeypatch.setattr(pc, "_call_check_gate", lambda db, symbol: {
        "gate": "BLOCKED",
        "lanes": {"book_proxy_pressure": {"active": True}},
    })
    monkeypatch.setattr(pc, "check_market_regime", lambda db, symbol: {"name": "market_regime", "result": "PASS", "detail": "ok"})


# ---------------------------------------------------------------------------
# 1. All PASS -> GO
# ---------------------------------------------------------------------------

def test_all_pass_gives_go(monkeypatch):
    _patch_all_pass(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        reval = _write_json(tmpdir, "reval.json", {"results": [{"pass_rate": 0.6}]})
        density = _write_json(tmpdir, "density.json", {"status": "READY_TO_RANK", "estimated_fills": 40})
        payload = pc.run_checklist(db="/fake/db", revalidation_report=reval, density_report=density)
    assert payload["overall"] == "GO"
    assert payload["fail_count"] == 0
    assert payload["skip_count"] == 0


# ---------------------------------------------------------------------------
# 2. Gate FAIL -> NO-GO
# ---------------------------------------------------------------------------

def test_gate_fail_gives_no_go(monkeypatch):
    _patch_gate_fail(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        reval = _write_json(tmpdir, "reval.json", {"results": [{"pass_rate": 0.6}]})
        density = _write_json(tmpdir, "density.json", {"status": "READY_TO_RANK", "estimated_fills": 40})
        payload = pc.run_checklist(db="/fake/db", revalidation_report=reval, density_report=density)
    assert payload["overall"] == "NO-GO"
    assert payload["fail_count"] >= 1


# ---------------------------------------------------------------------------
# 3. No FAIL but SKIP -> HOLD
# ---------------------------------------------------------------------------

def test_skip_with_no_fail_gives_hold(monkeypatch):
    _patch_all_pass(monkeypatch)
    payload = pc.run_checklist(
        db="/fake/db",
        revalidation_report="/nonexistent/reval.json",
        density_report="/nonexistent/density.json",
    )
    assert payload["overall"] == "HOLD"
    assert payload["skip_count"] >= 1
    assert payload["fail_count"] == 0


# ---------------------------------------------------------------------------
# 4. Missing revalidation file -> SKIP
# ---------------------------------------------------------------------------

def test_missing_revalidation_file_gives_skip():
    result = pc.check_revalidation("/nonexistent/reval.json")
    assert result["result"] == "SKIP"
    assert "not found" in result["detail"]


# ---------------------------------------------------------------------------
# 5. Output schema
# ---------------------------------------------------------------------------

def test_output_schema(monkeypatch):
    _patch_all_pass(monkeypatch)
    payload = pc.run_checklist(
        db="/fake/db",
        revalidation_report="/nonexistent/reval.json",
        density_report="/nonexistent/density.json",
    )
    for key in ("timestamp_utc", "symbol", "pocket", "overall", "fail_count",
                "skip_count", "checklist", "revalidate_cmd"):
        assert key in payload, f"missing key: {key}"
    assert len(payload["checklist"]) == 4
    for item in payload["checklist"]:
        assert "name" in item and "result" in item and "detail" in item


# ---------------------------------------------------------------------------
# 6. Revalidation FAIL when pass_rate low
# ---------------------------------------------------------------------------

def test_revalidation_fail_when_pass_rate_low():
    with tempfile.TemporaryDirectory() as tmpdir:
        reval = _write_json(tmpdir, "reval.json", {"results": [{"pass_rate": 0.3}]})
        result = pc.check_revalidation(reval)
    assert result["result"] == "FAIL"


# ---------------------------------------------------------------------------
# 7. Fill density FAIL when insufficient
# ---------------------------------------------------------------------------

def test_fill_density_fail_when_insufficient():
    with tempfile.TemporaryDirectory() as tmpdir:
        density = _write_json(tmpdir, "density.json", {"status": "INSUFFICIENT", "estimated_fills": 12})
        result = pc.check_fill_density(density)
    assert result["result"] == "FAIL"
    assert "12" in result["detail"]
