from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

from tools import run_scratch_calibration as rsc


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"run_scratch_cal_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _write_report(out_md: Path, n: int) -> None:
    out_md.write_text(f"# SCRATCH_ANALYSIS\n\n## Baseline\n\nn={n}\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(
        json.dumps({"baseline": {"n": float(n)}}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def test_run_with_fallback_improves_sample(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_md = tmp / "SELL.md"
        calls: list[tuple[str, str, int]] = []

        def _fake_run_side(side: str, db: str, symbol: str, out_md_param: Path, **kwargs):
            regime = str(kwargs.get("regime", "UP")).upper()
            lookback = int(kwargs.get("lookback_min", 0))
            calls.append((side, regime, lookback))
            n = 0 if regime == "UP" else 42
            _write_report(out_md_param, n)
            return 0

        monkeypatch.setattr(rsc, "_run_side", _fake_run_side)

        rc, n_primary, n_final = rsc._run_with_fallback(
            "sell",
            db="dummy.db",
            symbol="ETHUSDT",
            out_md=out_md,
            adverse_sweep="2:10",
            trail_sweep="2,3,4,5",
            fee_bps=0.5,
            exec_model="passive_realistic",
            regime="UP",
            lookback_min=100,
            min_trades=10,
            fallback_regime="NONE",
            fallback_lookback_min=200,
        )

        assert rc == 0
        assert n_primary == 0
        assert n_final == 42
        assert len(calls) == 2
        assert calls[0][1] == "UP"
        assert calls[1][1] == "NONE"
        assert "Fallback accepted" in out_md.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_with_fallback_skipped_when_sample_enough(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_md = tmp / "BUY.md"
        calls: list[tuple[str, str, int]] = []

        def _fake_run_side(side: str, db: str, symbol: str, out_md_param: Path, **kwargs):
            calls.append((side, str(kwargs.get("regime", "UP")).upper(), int(kwargs.get("lookback_min", 0))))
            _write_report(out_md_param, 55)
            return 0

        monkeypatch.setattr(rsc, "_run_side", _fake_run_side)

        rc, n_primary, n_final = rsc._run_with_fallback(
            "buy",
            db="dummy.db",
            symbol="ETHUSDT",
            out_md=out_md,
            adverse_sweep="2:10",
            trail_sweep="2,3,4,5",
            fee_bps=0.5,
            exec_model="passive_realistic",
            regime="UP",
            lookback_min=100,
            min_trades=10,
            fallback_regime="NONE",
            fallback_lookback_min=200,
        )

        assert rc == 0
        assert n_primary == 55
        assert n_final == 55
        assert len(calls) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_writes_run_summary(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_sell = tmp / "SELL.md"
        out_buy = tmp / "BUY.md"

        def _fake_run_with_fallback(*args, **kwargs):
            out_md = kwargs["out_md"]
            _write_report(out_md, 25)
            return (0, 10, 25)

        monkeypatch.setattr(rsc, "_run_with_fallback", _fake_run_with_fallback)
        monkeypatch.setattr(
            sys,
            "argv",
            ["x", "--symbol", "ETHUSDT", "--out-sell", str(out_sell), "--out-buy", str(out_buy)],
        )
        assert rsc.main() == 0
        summary = json.loads((tmp / "SCRATCH_CALIBRATION_RUN_SUMMARY.json").read_text(encoding="utf-8"))
        assert summary["run_summary"]["run_type"] == "run_scratch_calibration"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
