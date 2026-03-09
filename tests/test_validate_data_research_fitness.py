from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_data_research_fitness as vdrf


FIXTURE_DB = Path("tests/fixtures/microstructure_sample.db")


def _mk_local_tmp() -> Path:
    root = Path("localtests") / "research_fitness" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_analyze_research_fitness_warns_on_degraded_but_usable_fixture() -> None:
    tmp = _mk_local_tmp()
    csv_path = tmp / "event_diary.csv"
    csv_path.write_text("ts_ms,symbol,note\n1700000000000,BTCUSDT,ok\n", encoding="utf-8")

    payload = vdrf.analyze_research_fitness(
        db_path=FIXTURE_DB,
        csv_path=csv_path,
        symbols=["BTCUSDT", "ETHUSDT"],
        fresh_sec=9_999_999_999,
        min_trade_rows_per_symbol=10,
        now=1_700_000_100.0,
    )

    assert payload["status"] == "warn"
    assert payload["db_ready"] is True
    assert payload["contract"]["status"] == "warn"
    assert payload["feature_stats"]["BTCUSDT"]["has_mid"] is True
    assert payload["feature_stats"]["BTCUSDT"]["has_trade_intensity"] is True
    assert "contract_warn" in payload["warnings"]
    assert "no_spread:BTCUSDT" in payload["warnings"]
    assert "no_spread:ETHUSDT" in payload["warnings"]


def test_analyze_research_fitness_fails_when_required_inputs_missing() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "broken.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
            conn.commit()
        finally:
            conn.close()

        payload = vdrf.analyze_research_fitness(
            db_path=db,
            csv_path=tmp / "missing.csv",
            symbols=["BTCUSDT"],
            fresh_sec=120,
            now=1_700_000_100.0,
        )
        assert payload["status"] == "fail"
        assert "missing_event_diary_csv" in payload["failures"]
        assert "db_not_ready" in payload["failures"]
        assert "contract_fail" in payload["failures"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_writes_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        csv_path = tmp / "event_diary.csv"
        out_json = tmp / "fitness.json"
        out_md = tmp / "fitness.md"
        csv_path.write_text("ts_ms,symbol,note\n1700000000000,ETHUSDT,ok\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(FIXTURE_DB),
                "--csv",
                str(csv_path),
                "--symbols",
                "BTCUSDT,ETHUSDT",
                "--fresh-sec",
                "9999999999",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert vdrf.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "warn"
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
