from __future__ import annotations

import shutil
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.microstructure import build_collector_schema_fixture, build_fixture_manifest, cleanup_temp_path, make_temp_micro_db
from tools.build_micro_features import build_micro_features


def test_build_micro_features_smoke_fixture_db() -> None:
    db = make_temp_micro_db(prefix="micro_features_smoke")
    out_dir = Path("localtests") / "micro_features_smoke" / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        build_collector_schema_fixture(
            db,
            symbols=["ETHUSDT"],
            start_ms=1_700_000_000_000,
            rows_per_symbol=40,
            include_true_book=True,
        )
        manifest_info = build_fixture_manifest(db, ["ETHUSDT"])
        sym = manifest_info["symbols"]["ETHUSDT"]
        start = float(sym["min_ts_ms"]) / 1000.0
        end = float(sym["max_ts_ms"]) / 1000.0

        manifest = build_micro_features(
            db_path=db,
            out_root=out_dir,
            symbol="ETHUSDT",
            interval_ms=100,
            window_sec=300,
            start_ts=start,
            end_ts=end,
            rv_window_sec=5.0,
        )
        assert manifest["dates"]
        total_rows = sum(int(d["rows"]) for d in manifest["dates"])
        assert total_rows > 0
        one_day_dir = out_dir / "interval_ms=100" / "symbol=ETHUSDT" / f"date={manifest['dates'][0]['date']}"
        assert (one_day_dir / "bars.parquet").exists()
    finally:
        cleanup_temp_path(db)
        shutil.rmtree(out_dir, ignore_errors=True)
