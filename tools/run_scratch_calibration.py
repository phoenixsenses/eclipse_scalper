from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.run_summary import build_run_summary

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scratch calibration sweeps and write markdown summaries.")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--out-sell", default="reports/SCRATCH_CALIBRATION_SELL_UP.md")
    p.add_argument("--out-buy", default="reports/SCRATCH_CALIBRATION_BUY_UP.md")
    p.add_argument("--adverse-sweep", default="2.0:10.0")
    p.add_argument("--trail-sweep", default="2.0,3.0,4.0,5.0")
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--exec-model", default="passive_realistic", choices=["passive_realistic", "taker"])
    p.add_argument("--regime", default="UP", help="Primary regime filter (UP/DOWN/NONE).")
    p.add_argument("--lookback-min", type=int, default=13 * 24 * 60)
    p.add_argument("--min-trades", type=int, default=30, help="If baseline n is below this, run fallback pass.")
    p.add_argument("--fallback-regime", default="NONE", help="Fallback regime filter when sample is too low.")
    p.add_argument("--fallback-lookback-min", type=int, default=30 * 24 * 60)
    return p.parse_args()


def _run_side(
    side: str,
    db: str,
    symbol: str,
    out_md: Path,
    *,
    adverse_sweep: str,
    trail_sweep: str,
    fee_bps: float,
    exec_model: str,
    regime: str,
    lookback_min: int,
) -> int:
    out_json = out_md.with_suffix(".json")
    regime_val = str(regime or "UP").strip()
    if regime_val.upper() == "NONE":
        regime_val = "none"
    cmd = [
        sys.executable,
        "-m",
        "tools.backtest_scratch",
        "--db",
        db,
        "--symbol",
        symbol,
        "--side",
        side,
        "--regime",
        str(regime_val),
        "--lookback-min",
        str(int(lookback_min)),
        "--adverse-sweep",
        str(adverse_sweep),
        "--trail-sweep",
        str(trail_sweep),
        "--fee-bps",
        str(float(fee_bps)),
        "--exec-model",
        str(exec_model),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    return int(subprocess.call(cmd))


def _load_baseline_n(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        base = payload.get("baseline") if isinstance(payload, dict) else None
        if isinstance(base, dict):
            return int(float(base.get("n", 0.0) or 0.0))
    except Exception:
        return 0
    return 0


def _append_notes(out_md: Path, notes: list[str]) -> None:
    if not notes:
        return
    text = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
    text = text.rstrip() + "\n\n## Calibration Notes\n\n"
    for n in notes:
        text += f"- {n}\n"
    out_md.write_text(text, encoding="utf-8")


def _run_with_fallback(
    side: str,
    db: str,
    symbol: str,
    out_md: Path,
    *,
    adverse_sweep: str,
    trail_sweep: str,
    fee_bps: float,
    exec_model: str,
    regime: str,
    lookback_min: int,
    min_trades: int,
    fallback_regime: str,
    fallback_lookback_min: int,
) -> tuple[int, int, int]:
    notes: list[str] = []
    rc_primary = _run_side(
        side,
        db,
        symbol,
        out_md,
        adverse_sweep=adverse_sweep,
        trail_sweep=trail_sweep,
        fee_bps=fee_bps,
        exec_model=exec_model,
        regime=regime,
        lookback_min=lookback_min,
    )
    out_json = out_md.with_suffix(".json")
    n_primary = _load_baseline_n(out_json)
    n_final = n_primary
    rc_final = rc_primary

    if rc_primary == 0 and n_primary < int(max(1, min_trades)):
        fb_lookback = int(max(int(lookback_min), int(fallback_lookback_min)))
        notes.append(
            f"Primary sample too low: baseline_n={n_primary} < min_trades={int(min_trades)}. "
            f"Fallback run executed (regime={str(fallback_regime).upper()}, lookback_min={fb_lookback})."
        )
        rc_fb = _run_side(
            side,
            db,
            symbol,
            out_md,
            adverse_sweep=adverse_sweep,
            trail_sweep=trail_sweep,
            fee_bps=fee_bps,
            exec_model=exec_model,
            regime=fallback_regime,
            lookback_min=fb_lookback,
        )
        n_fb = _load_baseline_n(out_json)
        if rc_fb == 0 and n_fb > n_primary:
            rc_final = rc_fb
            n_final = n_fb
            notes.append(f"Fallback accepted: baseline_n improved {n_primary} -> {n_fb}.")
        else:
            notes.append(
                f"Fallback did not improve sample (primary_n={n_primary}, fallback_n={n_fb}, rc_fallback={rc_fb})."
            )
    if n_final < int(max(1, min_trades)):
        notes.append(
            "Insufficient calibration sample remains after fallback. "
            "Action: increase lookback, relax regime filter, or wait for more data."
        )
    _append_notes(out_md, notes)
    return rc_final, n_primary, n_final


def main() -> int:
    args = _args()
    out_sell = Path(args.out_sell)
    out_buy = Path(args.out_buy)
    out_sell.parent.mkdir(parents=True, exist_ok=True)
    out_buy.parent.mkdir(parents=True, exist_ok=True)

    rc1, n1_primary, n1_final = _run_with_fallback(
        "sell",
        db=str(args.db),
        symbol=str(args.symbol),
        out_md=out_sell,
        adverse_sweep=str(args.adverse_sweep),
        trail_sweep=str(args.trail_sweep),
        fee_bps=float(args.fee_bps),
        exec_model=str(args.exec_model),
        regime=str(args.regime),
        lookback_min=int(args.lookback_min),
        min_trades=int(args.min_trades),
        fallback_regime=str(args.fallback_regime),
        fallback_lookback_min=int(args.fallback_lookback_min),
    )
    rc2, n2_primary, n2_final = _run_with_fallback(
        "buy",
        db=str(args.db),
        symbol=str(args.symbol),
        out_md=out_buy,
        adverse_sweep=str(args.adverse_sweep),
        trail_sweep=str(args.trail_sweep),
        fee_bps=float(args.fee_bps),
        exec_model=str(args.exec_model),
        regime=str(args.regime),
        lookback_min=int(args.lookback_min),
        min_trades=int(args.min_trades),
        fallback_regime=str(args.fallback_regime),
        fallback_lookback_min=int(args.fallback_lookback_min),
    )

    if rc1 != 0 and not out_sell.exists():
        out_sell.write_text("# SCRATCH CALIBRATION SELL_UP\n\nbacktest_scratch execution failed.\n", encoding="utf-8")
    if rc2 != 0 and not out_buy.exists():
        out_buy.write_text("# SCRATCH CALIBRATION BUY_UP\n\nbacktest_scratch execution failed.\n", encoding="utf-8")

    summary_path = out_sell.parent / "SCRATCH_CALIBRATION_RUN_SUMMARY.json"
    summary_payload = {
        "symbol": str(args.symbol),
        "sell": {"rc": int(rc1), "baseline_n_primary": int(n1_primary), "baseline_n_final": int(n1_final), "out_md": str(out_sell)},
        "buy": {"rc": int(rc2), "baseline_n_primary": int(n2_primary), "baseline_n_final": int(n2_final), "out_md": str(out_buy)},
    }
    summary_payload["run_summary"] = build_run_summary(
        run_type="run_scratch_calibration",
        inputs={"symbol": str(args.symbol), "db": str(args.db), "exec_model": str(args.exec_model)},
        metrics={"sell_rc": int(rc1), "buy_rc": int(rc2), "sell_n_final": int(n1_final), "buy_n_final": int(n2_final)},
        artifacts={"json": str(summary_path), "sell_md": str(out_sell), "buy_md": str(out_buy)},
    )
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        "run_scratch_calibration: "
        f"rc_sell={rc1} n_sell_primary={n1_primary} n_sell_final={n1_final} "
        f"rc_buy={rc2} n_buy_primary={n2_primary} n_buy_final={n2_final}"
    )
    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
