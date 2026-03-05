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
from src.microphys.live.daemon import load_latest_model_specs, run_live_cycle
import src.microphys.live.daemon as live_daemon


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"live_daemon_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_db(path: Path, symbol: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (ts INTEGER, symbol TEXT, price REAL, qty REAL, is_buyer_maker INTEGER)"
        )
        conn.execute("CREATE TABLE mark_prices (ts INTEGER, symbol TEXT, mark_price REAL)")
        now_ms = int(time.time() * 1000)
        rows_t = []
        rows_m = []
        for i in range(300):
            ts = now_ms - 30_000 + i * 100
            px = 2000.0 + 0.1 * i
            rows_t.append((ts, symbol, px, 1.0 + (i % 3) * 0.1, 1 if i % 2 else 0))
            rows_m.append((ts, symbol, px))
        conn.executemany("INSERT INTO agg_trades(ts,symbol,price,qty,is_buyer_maker) VALUES (?,?,?,?,?)", rows_t)
        conn.executemany("INSERT INTO mark_prices(ts,symbol,mark_price) VALUES (?,?,?)", rows_m)
        conn.commit()
    finally:
        conn.close()


def _build_run(run_root: Path, symbol: str) -> None:
    run = run_root / "run_20990101_000000_symbol=ETHUSDT_interval=100ms"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    spec = SignalSpec(
        name="always_buy",
        side="buy",
        condition={"type": "gt", "op": "gt", "left": "F_ofi_z", "right": -999.0},
        horizon_bars=5,
        cooldown_bars=3,
    )
    cand_path = run / "cand.jsonl"
    cand_path.write_text(specs_to_jsonl([spec]), encoding="utf-8")
    sel = pd.DataFrame([{"signal": "always_buy"}])
    sel_path = run / "selected.parquet"
    sel.to_parquet(sel_path, index=False)
    cal_path = run / "calibration.json"
    cal_path.write_text(json.dumps({"quantiles": {"spread_z": {"0.5": 0.0}, "F_ofi_z": {"0.5": 0.0}}}) + "\n", encoding="utf-8")
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


def test_latest_model_load_and_incremental_cycle() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        run_root = tmp / "runs"
        out_root = tmp / "live"
        _build_db(db, "ETHUSDT")
        _build_run(run_root, "ETHUSDT")

        specs, _ = load_latest_model_specs(run_root)
        assert specs

        cfg = LiveSettings(
            db_path=str(db),
            symbol="ETHUSDT",
            interval_ms=100,
            lookback_hours=0.01,
            refresh_sec=0.1,
            out_root=str(out_root),
            run_root=str(run_root),
            execution_model="maker_hazard",
        )
        params_path = tmp / "exec_params.json"
        params_path.write_text(
            json.dumps({"maker_hazard": {"a": 1.0, "b": -0.5, "c": 0.5, "d": -0.2, "ttl_bars": 5, "fill_threshold": 0.3}}) + "\n",
            encoding="utf-8",
        )
        cfg = cfg.model_copy(update={"execution_params_path": str(params_path)})
        r1 = run_live_cycle(cfg)
        assert Path(out_root / "watermark.json").exists()
        assert Path(out_root / "status.json").exists()
        n1 = int(r1.get("new_trades", 0))
        r2 = run_live_cycle(cfg)
        n2 = int(r2.get("new_trades", 0))
        assert n1 >= 0
        assert n2 <= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_cycle_with_regime_experts_writes_gating() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        run_root = tmp / "runs"
        out_root = tmp / "live"
        _build_db(db, "ETHUSDT")
        _build_run(run_root, "ETHUSDT")

        experts = pd.DataFrame(
            [
                {
                    "symbol": "ETHUSDT",
                    "aligned_regime_id": 0,
                    "signal": "always_buy",
                    "family": "ofi",
                    "trade_count": 10,
                    "mean_net_ret": 0.001,
                    "win_rate": 0.6,
                    "fill_rate": 1.0,
                    "base_weight": 1.0,
                    "penalty": 1.0,
                    "weight": 1.0,
                    "expected_trigger_rate": 0.1,
                    "expected_fill_rate": 1.0,
                    "expert_quality": 0.01,
                    "regime_rows": 100,
                }
            ]
        )
        experts_path = tmp / "experts.parquet"
        experts.to_parquet(experts_path, index=False)

        cfg = LiveSettings(
            db_path=str(db),
            symbol="ETHUSDT",
            interval_ms=100,
            lookback_hours=0.01,
            refresh_sec=0.1,
            out_root=str(out_root),
            run_root=str(run_root),
            execution_model="simple",
            use_regime_experts=True,
            experts_path=str(experts_path),
        )
        res = run_live_cycle(cfg)
        assert Path(out_root / "gating_live.parquet").exists()
        assert "regime_experts_used" in res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_cycle_regime_experts_autoresolve_from_pointers() -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        run_root = tmp / "runs"
        out_root = tmp / "live"
        _build_db(db, "ETHUSDT")
        _build_run(run_root, "ETHUSDT")
        run = run_root / "run_20990101_000000_symbol=ETHUSDT_interval=100ms"

        experts = pd.DataFrame(
            [
                {
                    "symbol": "ETHUSDT",
                    "aligned_regime_id": 0,
                    "signal": "always_buy",
                    "family": "ofi",
                    "trade_count": 10,
                    "mean_net_ret": 0.001,
                    "win_rate": 0.6,
                    "fill_rate": 1.0,
                    "base_weight": 1.0,
                    "penalty": 1.0,
                    "weight": 1.0,
                    "expected_trigger_rate": 0.1,
                    "expected_fill_rate": 1.0,
                    "expert_quality": 0.01,
                    "regime_rows": 100,
                }
            ]
        )
        experts_path = run / "experts.parquet"
        experts.to_parquet(experts_path, index=False)
        aligned = pd.DataFrame([{"ts_ms": i, "symbol": "ETHUSDT", "aligned_regime_id": 0} for i in range(1000)])
        aligned_path = run / "aligned.parquet"
        aligned.to_parquet(aligned_path, index=False)
        ptr = json.loads((run / "pointers.json").read_text(encoding="utf-8"))
        ptr["ensemble_regime_experts_parquet"] = str(experts_path)
        ptr["aligned_regimes_path"] = str(aligned_path)
        (run / "pointers.json").write_text(json.dumps(ptr) + "\n", encoding="utf-8")

        cfg = LiveSettings(
            db_path=str(db),
            symbol="ETHUSDT",
            interval_ms=100,
            lookback_hours=0.01,
            refresh_sec=0.1,
            out_root=str(out_root),
            run_root=str(run_root),
            execution_model="simple",
            use_regime_experts=True,
            experts_path="",
            aligned_regimes_path="",
        )
        res = run_live_cycle(cfg)
        status = json.loads((out_root / "status.json").read_text(encoding="utf-8"))
        assert bool(status.get("experts_loaded")) is True
        assert bool(status.get("aligned_regimes_loaded")) is True
        assert str(status.get("experts_pointer_path", "")).strip()
        assert Path(out_root / "gating_live.parquet").exists()
        assert bool(res.get("experts_loaded")) is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_cycle_with_risk_engine_writes_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        db = tmp / "micro.db"
        run_root = tmp / "runs"
        out_root = tmp / "live"
        _build_db(db, "ETHUSDT")
        _build_run(run_root, "ETHUSDT")

        policy = {
            "starting_equity": 10000.0,
            "base_risk_per_trade": 0.05,
            "min_trade_notional": 10.0,
            "max_trade_notional": 1000.0,
            "drawdown_kill_pct": 0.5,
            "kill_cooldown_minutes": 1,
            "health_skip_on_bad": True,
            "regime_quality_floor": 1.1,
        }
        pol_path = tmp / "risk_policy.json"
        pol_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")

        cfg = LiveSettings(
            db_path=str(db),
            symbol="ETHUSDT",
            interval_ms=100,
            lookback_hours=0.01,
            refresh_sec=0.1,
            out_root=str(out_root),
            run_root=str(run_root),
            execution_model="simple",
            enable_risk_engine=True,
            risk_policy_path=str(pol_path),
            starting_equity=10000.0,
        )

        def _fake_generate_papertrades(*args, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "entry_ts_utc": "2026-03-01T00:00:00Z",
                        "exit_ts_utc": "2026-03-01T00:00:01Z",
                        "side": "buy",
                        "entry_price": 2000.0,
                        "fill_price": 2000.0,
                        "exit_price": 2000.1,
                        "filled": True,
                        "ttl_expired": False,
                        "fill_delay_bars": 0,
                        "pnl_gross": 0.0001,
                        "pnl_net": 0.00009,
                    }
                ]
            )

        monkeypatch.setattr(live_daemon, "generate_papertrades", _fake_generate_papertrades)
        run_live_cycle(cfg)
        assert Path(out_root / "risk_snapshot.json").exists()
        assert Path(out_root / "positions_live.parquet").exists()
        status = json.loads((out_root / "status.json").read_text(encoding="utf-8"))
        assert bool(status.get("risk_engine_enabled")) is True
        assert float(status.get("risk_equity", 0.0)) > 0.0
        pos = pd.read_parquet(out_root / "positions_live.parquet")
        assert not pos.empty

        # policy enforces skip -> event log should be written.
        events_path = Path("logs/risk_events.jsonl")
        assert events_path.exists()
        lines = [ln.strip() for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert any("RISK_SKIP_BAD_REGIME" in ln for ln in lines)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
