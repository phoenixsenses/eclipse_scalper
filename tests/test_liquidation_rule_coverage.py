from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.microstructure import build_collector_schema_fixture, cleanup_temp_path, make_temp_micro_db
from tools import liquidation_rule_coverage as mod
from tools import report_schema_validator as rsv


def test_liquidation_rule_coverage_writes_reports() -> None:
    db = make_temp_micro_db(prefix="liq_coverage")
    out_json = Path("localtests/liquidation_rule_coverage/out.json")
    out_md = Path("localtests/liquidation_rule_coverage/out.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        build_collector_schema_fixture(
            db,
            symbols=["ETHUSDT"],
            start_ms=1_700_000_000_000,
            rows_per_symbol=180,
            include_liquidations=True,
            include_true_book=False,
        )
        old = sys.argv
        try:
            sys.argv = [
                "x",
                "--db",
                str(db),
                "--symbol",
                "ETHUSDT",
                "--lookbacks-min",
                "60,180",
                "--bucket-sec",
                "5",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
            rc = mod.main()
        finally:
            sys.argv = old
        assert rc == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["rule"] == "high_liq_reversal_regime"
        assert len(payload["results"]) == 2
        assert rsv.infer_schema_name(payload) == "liquidation_rule_coverage"
        assert rsv.validate_payload(payload, "liquidation_rule_coverage") == []
    finally:
        cleanup_temp_path(db)
        out_json.unlink(missing_ok=True)
        out_md.unlink(missing_ok=True)
