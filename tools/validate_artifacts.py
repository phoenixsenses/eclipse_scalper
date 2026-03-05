from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.microphys.live.guardrails import (
    evaluate_probe_directional_sanity,
    evaluate_probe_trigger_sanity,
    validate_calibration_file,
    validate_execution_file,
)
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate online calibration/execution artifacts.")
    p.add_argument("--calibration", default="")
    p.add_argument("--execution", default="")
    p.add_argument("--physics", default="")
    p.add_argument("--symbol", default="")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--sanity-days", type=int, default=1)
    p.add_argument("--max-sanity-rows", type=int, default=50_000)
    p.add_argument("--probe-min-triggers", type=int, default=10)
    p.add_argument("--probe-max-density", type=float, default=0.95)
    p.add_argument("--total-density-min", type=float, default=0.001)
    p.add_argument("--total-density-max", type=float, default=0.60)
    p.add_argument("--directional-sanity", dest="directional_sanity", action="store_true")
    p.add_argument("--no-directional-sanity", dest="directional_sanity", action="store_false")
    p.add_argument("--directional-min-triggers", type=int, default=50)
    p.add_argument("--directional-max-fail-probes", type=int, default=2)
    p.add_argument("--directional-horizons", default="1,5")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-report", default="")
    p.set_defaults(directional_sanity=False)
    return p.parse_args()


def _load_recent_physics(root: Path, symbol: str, interval_ms: int, days: int, max_rows: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if days > 0:
        files = files[-int(days) :]
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    frame = frame.sort_values("ts_ms").reset_index(drop=True) if "ts_ms" in frame.columns else frame
    if len(frame) > int(max_rows):
        frame = frame.tail(int(max_rows)).reset_index(drop=True)
    return frame


def _append_event(payload: dict) -> None:
    p = Path("logs/calibration_events.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    try:
        report = {"ok": True, "calibration": {}, "execution": {}}
        lines = ["# Validate Artifacts", ""]
        dir_horizons = [int(x.strip()) for x in str(args.directional_horizons).split(",") if x.strip()]
        if str(args.calibration).strip():
            cp = Path(str(args.calibration))
            ok, errs, payload = validate_calibration_file(cp)
            report["calibration"] = {"path": str(cp), "ok": bool(ok), "errors": errs}
            report["ok"] = bool(report["ok"] and ok)
            lines += [f"## Calibration `{cp}`", "", f"- structural_ok: `{int(bool(ok))}`", f"- errors: `{';'.join(errs) if errs else 'none'}`"]
            if bool(ok) and str(args.physics).strip() and str(args.symbol).strip():
                frame = _load_recent_physics(
                    Path(str(args.physics)),
                    canonical_symbol(args.symbol),
                    int(args.interval_ms),
                    int(args.sanity_days),
                    int(args.max_sanity_rows),
                )
                ps_ok, ps_errs, ps_sum = evaluate_probe_trigger_sanity(
                    frame,
                    payload,
                    probe_min_triggers=int(args.probe_min_triggers),
                    probe_max_density=float(args.probe_max_density),
                    total_density_min=float(args.total_density_min),
                    total_density_max=float(args.total_density_max),
                )
                report["calibration"]["probe_sanity"] = {"ok": bool(ps_ok), "errors": ps_errs, "summary": ps_sum}
                report["ok"] = bool(report["ok"] and ps_ok)
                _append_event(
                    {
                        "event": "validate_probe_sanity",
                        "ok": bool(ps_ok),
                        "calibration_path": str(cp),
                        "errors": ps_errs,
                        "probe_stats": list(ps_sum.get("probe_stats", []) or []),
                        "total_density": float(ps_sum.get("total_density", 0.0) or 0.0),
                    }
                )
                lines += [
                    f"- probe_sanity_ok: `{int(bool(ps_ok))}`",
                    f"- probe_errors: `{';'.join(ps_errs) if ps_errs else 'none'}`",
                    f"- probe_total_density: `{float(ps_sum.get('total_density', 0.0)):.6f}`",
                    "",
                    "| probe | triggers | triggers_per_day | density |",
                    "|---|---:|---:|---:|",
                ]
                for r in list(ps_sum.get("probe_stats", []) or []):
                    lines.append(
                        f"| {r.get('probe','')} | {int(r.get('triggers',0))} | {float(r.get('triggers_per_day',0.0)):.4f} | {float(r.get('density',0.0)):.6f} |"
                    )
                if bool(args.directional_sanity):
                    ds_ok, ds_errs, ds_sum = evaluate_probe_directional_sanity(
                        frame,
                        payload,
                        horizons=dir_horizons or (1, 5),
                        min_dir_triggers=int(args.directional_min_triggers),
                        max_fail_probes=int(args.directional_max_fail_probes),
                    )
                    report["calibration"]["directional_sanity"] = {"ok": bool(ds_ok), "errors": ds_errs, "summary": ds_sum}
                    report["ok"] = bool(report["ok"] and ds_ok)
                    _append_event(
                        {
                            "event": "validate_directional_sanity",
                            "ok": bool(ds_ok),
                            "calibration_path": str(cp),
                            "errors": ds_errs,
                            "failed_count": int(ds_sum.get("failed_count", 0) or 0),
                            "directional_probe_stats": list(ds_sum.get("directional_probe_stats", []) or []),
                        }
                    )
                    lines += [
                        "",
                        f"- directional_sanity_ok: `{int(bool(ds_ok))}`",
                        f"- directional_errors: `{';'.join(ds_errs) if ds_errs else 'none'}`",
                        f"- directional_failed_count: `{int(ds_sum.get('failed_count', 0) or 0)}`",
                        "",
                        "| dprobe | side | h | n | mean_signed | win_rate | failed |",
                        "|---|---|---:|---:|---:|---:|---:|",
                    ]
                    for r in list(ds_sum.get("directional_probe_stats", []) or []):
                        lines.append(
                            f"| {r.get('probe','')} | {r.get('side','')} | {int(r.get('horizon_bars',0))} | {int(r.get('n_triggers',0))} | "
                            f"{float(r.get('mean_signed_return',0.0)):.8f} | {float(r.get('win_rate',0.0)):.4f} | {int(bool(r.get('failed', False)))} |"
                        )
        if str(args.execution).strip():
            ep = Path(str(args.execution))
            ok, errs, _ = validate_execution_file(ep)
            report["execution"] = {"path": str(ep), "ok": bool(ok), "errors": errs}
            report["ok"] = bool(report["ok"] and ok)
            lines += ["", f"## Execution `{ep}`", "", f"- ok: `{int(bool(ok))}`", f"- errors: `{';'.join(errs) if errs else 'none'}`"]
        if str(args.out_json).strip():
            out = Path(str(args.out_json))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if str(args.out_report).strip():
            out = Path(str(args.out_report))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"validate_artifacts ok={int(bool(report['ok']))}")
        return 0 if bool(report["ok"]) else 2
    except Exception as e:
        print(f"validate_artifacts error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
