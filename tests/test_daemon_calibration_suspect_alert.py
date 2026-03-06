from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.alpha.spec import SignalSpec, specs_to_jsonl
from src.microphys.live.config import LiveSettings
from src.microphys.live.daemon import run_live_cycle
from src.microphys.live.registry import activate_artifacts


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"live_cal_alert_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_db(path: Path, symbol: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE agg_trades (ts INTEGER, symbol TEXT, price REAL, qty REAL, is_buyer_maker INTEGER)")
        conn.execute("CREATE TABLE mark_prices (ts INTEGER, symbol TEXT, mark_price REAL)")
        now_ms = int(time.time() * 1000)
        rows_t = []
        rows_m = []
        for i in range(300):
            ts = now_ms - 30_000 + i * 100
            px = 2000.0 + 0.1 * i
            rows_t.append((ts, symbol, px, 1.0, 1 if i % 2 else 0))
            rows_m.append((ts, symbol, px))
        conn.executemany("INSERT INTO agg_trades(ts,symbol,price,qty,is_buyer_maker) VALUES (?,?,?,?,?)", rows_t)
        conn.executemany("INSERT INTO mark_prices(ts,symbol,mark_price) VALUES (?,?,?)", rows_m)
        conn.commit()
    finally:
        conn.close()


def _build_run(run_root: Path) -> None:
    run = run_root / "run_20990101_000000_symbol=ETHUSDT_interval=100ms"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    spec = SignalSpec(
        name="always_buy",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": -999.0},
        horizon_bars=5,
        cooldown_bars=0,
    )
    cand_path = run / "cand.jsonl"
    cand_path.write_text(specs_to_jsonl([spec]), encoding="utf-8")
    sel = pd.DataFrame([{"signal": "always_buy"}])
    sel_path = run / "selected.parquet"
    sel.to_parquet(sel_path, index=False)
    cal_path = run / "calibration.json"
    cal_path.write_text(
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
    (run / "pointers.json").write_text(
        json.dumps(
            {
                "candidates_deduped_jsonl": str(cand_path),
                "selected_parquet": str(sel_path),
                "calibration_json": str(cal_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_daemon_emits_calibration_suspected_bad(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    old_cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        db = tmp / "micro.db"
        run_root = tmp / "runs"
        out_root = tmp / "live"
        _build_db(db, "ETHUSDT")
        _build_run(run_root)
        active_cal = tmp / "active_cal.json"
        active_cal.write_text(
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
        activate_artifacts(
            live_root=out_root,
            calibration_path=str(active_cal),
            metadata={"run_id": "active_r", "calibration_probe_total_density": 0.01},
        )
        cfg = LiveSettings(
            db_path=str(db),
            symbol="ETHUSDT",
            interval_ms=100,
            lookback_hours=0.01,
            refresh_sec=0.1,
            out_root=str(out_root),
            run_root=str(run_root),
            use_active_artifacts=True,
        )
        res = run_live_cycle(cfg)
        assert "status" in res
        evt = Path("logs/calibration_events.jsonl")
        assert evt.exists()
        rows = [json.loads(x) for x in evt.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert any(str(r.get("event", "")) == "calibration_suspected_bad" for r in rows)
    finally:
        monkeypatch.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
