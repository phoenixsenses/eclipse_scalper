from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.microstructure import build_collector_schema_fixture
from tools import validate_microstructure_contract as vmc


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"micro_contract_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _mk_collector_db(path: Path) -> None:
    build_collector_schema_fixture(path, symbols=["ETHUSDT", "BTCUSDT"], start_ms=1_700_000_000_000, rows_per_symbol=2)


def _mk_true_book_db(path: Path) -> None:
    build_collector_schema_fixture(
        path,
        symbols=["ETHUSDT", "BTCUSDT"],
        start_ms=1_700_000_000_000,
        rows_per_symbol=2,
        include_true_book=True,
    )


def test_validate_microstructure_contract_warns_without_true_book() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        _mk_collector_db(db)
        payload = vmc.analyze_contract(db, ["ETHUSDT", "BTCUSDT"])
        assert payload["status"] == "warn"
        assert payload["feature_capability"]["trade_flow"] is True
        assert payload["feature_capability"]["trade_plus_liq"] is True
        assert payload["feature_capability"]["requires_book"] is False
        assert "true_top_of_book_missing" in payload["warnings"]
        assert payload["run_summary"]["run_type"] == "validate_microstructure_contract"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_microstructure_contract_passes_with_true_book() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        _mk_true_book_db(db)
        payload = vmc.analyze_contract(db, ["ETHUSDT", "BTCUSDT"])
        assert payload["status"] == "pass"
        assert payload["feature_capability"]["requires_book"] is True
        assert payload["feature_capability"]["tier"] == "full_book"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_microstructure_contract_fails_missing_required_table() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
            conn.commit()
        finally:
            conn.close()
        payload = vmc.analyze_contract(db, ["ETHUSDT"])
        assert payload["status"] == "fail"
        assert "missing_table:agg_trades" in payload["failures"]
        assert "missing_table:liquidations" in payload["failures"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_writes_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        out_json = tmp / "contract.json"
        out_md = tmp / "contract.md"
        _mk_collector_db(db)
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--db", str(db), "--symbols", "ETHUSDT,BTCUSDT", "--out-json", str(out_json), "--out-md", str(out_md)],
        )
        assert vmc.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "warn"
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
