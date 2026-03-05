from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.microphys.alpha.calibration import CalibrationContext, compute_calibration, load_calibration, save_calibration
from src.microphys.alpha.column_guard import collect_expr_columns
from src.microphys.alpha.generator import generate_candidates
from src.microphys.alpha.spec import specs_to_jsonl
from utils.symbols import canonical_symbol


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate alpha candidate SignalSpec set.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--compression", default="0,1")
    p.add_argument("--vacuum", default="0,1")
    p.add_argument("--regimes", default="", help="optional csv of regime ids")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--use-calibration", dest="use_calibration", action="store_true")
    p.add_argument("--no-use-calibration", dest="use_calibration", action="store_false")
    p.add_argument("--coverage-guarantee", dest="coverage_guarantee", action="store_true")
    p.add_argument("--no-coverage-guarantee", dest="coverage_guarantee", action="store_false")
    p.add_argument("--min-triggered", type=int, default=50)
    p.add_argument("--max-tries", type=int, default=30)
    p.add_argument("--target-triggers-per-day", type=float, default=200.0)
    p.add_argument("--target-trigger-band", type=float, default=0.5)
    p.add_argument("--min-triggers-per-day", type=float, default=50.0)
    p.add_argument("--max-triggers-per-day", type=float, default=500.0)
    p.add_argument("--calibration-days", type=int, default=14)
    p.add_argument("--max-nan-ratio", type=float, default=0.98)
    p.add_argument("--coverage-report", default="")
    p.add_argument("--out", default="data/derived/alpha_candidates")
    p.set_defaults(use_calibration=True, coverage_guarantee=True)
    return p.parse_args()


def _load_recent_physics(root: Path, symbol: str, interval_ms: int, days: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    chosen = files[-max(1, int(days)) :]
    return pd.concat([pd.read_parquet(p) for p in chosen], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        horizons = _parse_int_list(args.horizons)
        comp = [bool(int(x)) for x in _parse_int_list(args.compression)]
        vac = [bool(int(x)) for x in _parse_int_list(args.vacuum)]
        regimes = _parse_int_list(args.regimes) if str(args.regimes).strip() else []
        out_dir = Path(str(args.out)) / f"symbol={symbol}" / f"interval_ms={int(args.interval_ms)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        frame = _load_recent_physics(Path(str(args.physics)), symbol, int(args.interval_ms), int(args.calibration_days))
        if frame.empty:
            raise RuntimeError("physics_missing_for_generation")
        cal_path = out_dir / "calibration.json"
        calibration: CalibrationContext | None = None
        if bool(args.use_calibration) and cal_path.exists():
            calibration = load_calibration(cal_path)
        elif bool(args.use_calibration):
            cols = [c for c in ("F_ofi_z", "F_intensity_z", "spread_z", "rv_short", "rv_z", "top_depth_imbalance", "liq_rate_z", "micro_trend") if c in frame.columns]
            calibration = compute_calibration(frame, columns=cols)
            save_calibration(calibration, cal_path)

        target_tpd = float(args.target_triggers_per_day)
        band = max(0.0, float(args.target_trigger_band))
        band_lo = max(0.0, target_tpd * (1.0 - band))
        band_hi = max(band_lo, target_tpd * (1.0 + band))
        min_tpd = max(float(args.min_triggers_per_day), band_lo)
        max_tpd = min(float(args.max_triggers_per_day), band_hi) if float(args.max_triggers_per_day) > 0 else band_hi

        specs = generate_candidates(
            horizons=horizons,
            compression_options=comp,
            vacuum_options=vac,
            regime_ids=regimes,
            limit=int(args.limit),
            calibration=calibration,
            frame=frame,
            coverage_guarantee=bool(args.coverage_guarantee),
            min_triggered=int(args.min_triggered),
            max_tries=int(args.max_tries),
            target_triggers_per_day=target_tpd,
            min_triggers_per_day=min_tpd,
            max_triggers_per_day=max_tpd,
            available_columns=frame.columns.tolist(),
            max_nan_ratio=float(args.max_nan_ratio),
        )
        payload_hash = hashlib.sha1(
            json.dumps([s.to_dict() for s in specs], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        out_file = out_dir / "candidates.jsonl"
        out_file.write_text(specs_to_jsonl(specs), encoding="utf-8")
        all_cols = sorted({c for s in specs for c in collect_expr_columns(s.condition)})
        missing_cols = [c for c in all_cols if c not in frame.columns]
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "interval_ms": int(args.interval_ms),
                    "count": len(specs),
                    "hash": payload_hash,
                    "horizons": horizons,
                    "compression": comp,
                    "vacuum": vac,
                    "regimes": regimes,
                    "limit": int(args.limit),
                    "use_calibration": bool(args.use_calibration),
                    "coverage_guarantee": bool(args.coverage_guarantee),
                    "min_triggered": int(args.min_triggered),
                    "max_tries": int(args.max_tries),
                    "target_triggers_per_day": target_tpd,
                    "target_trigger_band": float(args.target_trigger_band),
                    "min_triggers_per_day": min_tpd,
                    "max_triggers_per_day": max_tpd,
                    "max_nan_ratio": float(args.max_nan_ratio),
                    "referenced_columns": all_cols,
                    "missing_columns": missing_cols,
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if bool(args.coverage_guarantee):
            cov_rows = []
            for s in specs:
                trig = int((s.meta or {}).get("calibration_triggered", 0) or 0)
                relax = int((s.meta or {}).get("relax_steps", 0) or 0)
                tighten = int((s.meta or {}).get("tighten_steps", 0) or 0)
                tpd = float((s.meta or {}).get("trigger_rate_per_day", 0.0) or 0.0)
                cov_rows.append((s.name, trig, relax, tighten, tpd))
            cov_rows.sort(key=lambda x: (x[4], x[0]))
            report = Path(str(args.coverage_report)) if str(args.coverage_report).strip() else Path(
                f"reports/generator_coverage_guard_{symbol}_{int(args.interval_ms)}ms.md"
            )
            lines = [
                f"# Generator Coverage Guard - {symbol} ({int(args.interval_ms)}ms)",
                "",
                f"- candidates: `{len(specs)}`",
                f"- min_triggered target: `{int(args.min_triggered)}`",
                f"- trigger_rate/day target: `{target_tpd:.2f}`",
                f"- trigger_rate/day bounds: `[{min_tpd:.2f}, {max_tpd:.2f}]`",
                f"- calibration window days: `{int(args.calibration_days)}`",
                "",
                "| signal | calibration_triggered | triggers_per_day | relax_steps | tighten_steps |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, trig, relax, tighten, tpd in cov_rows[:100]:
                lines.append(f"| {name} | {trig} | {tpd:.2f} | {relax} | {tighten} |")
            if cov_rows:
                vals = [r[1] for r in cov_rows]
                rates = [r[4] for r in cov_rows]
                lines.append("")
                lines.append(f"- min_triggered_observed: `{min(vals)}`")
                lines.append(f"- median_triggered_observed: `{sorted(vals)[len(vals)//2]}`")
                lines.append(f"- max_triggered_observed: `{max(vals)}`")
                lines.append(f"- min_triggers_per_day_observed: `{min(rates):.2f}`")
                lines.append(f"- median_triggers_per_day_observed: `{sorted(rates)[len(rates)//2]:.2f}`")
                lines.append(f"- max_triggers_per_day_observed: `{max(rates):.2f}`")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        if missing_cols:
            raise RuntimeError(f"generated_specs_reference_missing_columns: {missing_cols}")
        print(
            f"generate_alpha_candidates ok out={out_file} count={len(specs)} hash={payload_hash} "
            f"use_calibration={int(bool(args.use_calibration))} coverage_guarantee={int(bool(args.coverage_guarantee))}"
        )
        return 0
    except Exception as e:
        print(f"generate_alpha_candidates error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
