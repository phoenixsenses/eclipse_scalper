from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.microphys.alpha.calibration import compute_calibration, save_calibration
from src.microphys.execution.calibration import calibrate_execution_models, save_execution_params
from src.microphys.live.guardrails import (
    evaluate_probe_directional_sanity,
    evaluate_probe_trigger_sanity,
    validate_calibration_file,
    validate_execution_file,
)
from src.microphys.live.registry import activate_artifacts, rollback_to_previous
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate and activate online artifacts with rollback support.")
    p.add_argument("--calibration", default="")
    p.add_argument("--execution", default="")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--run-id", default="")
    p.add_argument("--validation-report", default="")
    p.add_argument("--out-validation-report", default="")
    p.add_argument("--write-validation-event", dest="write_validation_event", action="store_true")
    p.add_argument("--no-write-validation-event", dest="write_validation_event", action="store_false")
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
    p.add_argument("--build-calibration", action="store_true")
    p.add_argument("--build-execution", action="store_true")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--rollback", choices=["", "calibration", "execution"], default="")
    p.set_defaults(write_validation_event=True, directional_sanity=False)
    return p.parse_args()


def _log_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _load_recent_physics(root: Path, symbol: str, interval_ms: int, days: int, max_rows: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if int(days) > 0:
        files = files[-int(days) :]
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "ts_ms" in frame.columns:
        frame = frame.sort_values("ts_ms").reset_index(drop=True)
    if len(frame) > int(max_rows):
        frame = frame.tail(int(max_rows)).reset_index(drop=True)
    return frame


def _default_validation_report_path(live_root: Path, symbol: str, interval_ms: int) -> Path:
    return live_root / "reports" / f"validate_artifacts_{symbol}_{int(interval_ms)}ms.md"


def _write_validation_report(path: Path, report: dict) -> None:
    lines = [
        "# Validate + Activate",
        "",
        f"- ok: `{int(bool(report.get('ok', False)))}`",
    ]
    cal = dict(report.get("calibration", {}) or {})
    if cal:
        lines += [
            "",
            "## Calibration",
            "",
            f"- path: `{cal.get('path', '')}`",
            f"- structural_ok: `{int(bool(cal.get('ok', False)))}`",
            f"- errors: `{';'.join(list(cal.get('errors', []) or [])) or 'none'}`",
        ]
        ps = dict(cal.get("probe_sanity", {}) or {})
        if ps:
            s = dict(ps.get("summary", {}) or {})
            lines += [
                f"- probe_ok: `{int(bool(ps.get('ok', False)))}`",
                f"- probe_errors: `{';'.join(list(ps.get('errors', []) or [])) or 'none'}`",
                f"- probe_total_density: `{float(s.get('total_density', 0.0) or 0.0):.6f}`",
            ]
    ex = dict(report.get("execution", {}) or {})
    if ex:
        lines += [
            "",
            "## Execution",
            "",
            f"- path: `{ex.get('path', '')}`",
            f"- ok: `{int(bool(ex.get('ok', False)))}`",
            f"- errors: `{';'.join(list(ex.get('errors', []) or [])) or 'none'}`",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    live_root = Path(str(args.live_root))
    events = Path("logs/calibration_events.jsonl")
    try:
        if str(args.rollback).strip():
            out = rollback_to_previous(str(args.rollback), live_root=live_root)
            _log_event(events, {"event": "rollback", "kind": str(args.rollback), "ok": True, "active": out})
            print(f"activate_online_artifacts rollback ok kind={args.rollback}")
            return 0

        symbol = canonical_symbol(args.symbol) if str(args.symbol).strip() else ""
        interval_ms = int(args.interval_ms)
        dir_horizons = [int(x.strip()) for x in str(args.directional_horizons).split(",") if x.strip()]
        physics_root = Path(str(args.physics)) if str(args.physics).strip() else None
        cal_path = Path(str(args.calibration)) if str(args.calibration).strip() else None
        exe_path = Path(str(args.execution)) if str(args.execution).strip() else None

        # Optional build mode.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        if bool(args.build_calibration) and cal_path is None:
            if not symbol or physics_root is None:
                raise RuntimeError("build_calibration_requires_physics_symbol")
            frame = _load_recent_physics(physics_root, symbol, interval_ms, int(args.days), int(args.max_sanity_rows))
            if frame.empty:
                raise RuntimeError("build_calibration_physics_missing")
            cols = [c for c in ("F_ofi_z", "F_intensity_z", "spread_z", "rv_short", "rv_z", "top_depth_imbalance", "liq_rate_z") if c in frame.columns]
            ctx = compute_calibration(frame, columns=cols)
            cal_dir = live_root / "artifacts" / "calibration"
            cal_dir.mkdir(parents=True, exist_ok=True)
            cal_path = cal_dir / f"calibration_{ts}.json"
            save_calibration(ctx, cal_path)
        if bool(args.build_execution) and exe_path is None:
            if not symbol or physics_root is None:
                raise RuntimeError("build_execution_requires_physics_symbol")
            frame = _load_recent_physics(physics_root, symbol, interval_ms, int(args.days), int(args.max_sanity_rows))
            if frame.empty:
                raise RuntimeError("build_execution_physics_missing")
            params = calibrate_execution_models(frame)
            exe_dir = live_root / "artifacts" / "execution"
            exe_dir.mkdir(parents=True, exist_ok=True)
            exe_path = exe_dir / f"params_{ts}.json"
            save_execution_params(exe_path, params)

        if cal_path is None and exe_path is None:
            raise RuntimeError("nothing_to_activate")

        report = {"ok": True, "calibration": {}, "execution": {}}
        cal_probe_summary = {}
        cal_probe_errors: list[str] = []
        directional_summary: dict = {}
        directional_errors: list[str] = []
        validation_reasons: list[str] = []
        validation_report_path = (
            Path(str(args.out_validation_report))
            if str(args.out_validation_report).strip()
            else _default_validation_report_path(live_root, (symbol or "UNKNOWN"), interval_ms)
        )
        if cal_path is not None:
            ok, errs, payload = validate_calibration_file(cal_path)
            report["calibration"] = {"path": str(cal_path), "ok": bool(ok), "errors": errs}
            report["ok"] = bool(report["ok"] and ok)
            if not ok:
                validation_reasons.extend(list(errs))
                _write_validation_report(validation_report_path, report)
                _log_event(events, {"event": "activate", "kind": "calibration", "ok": False, "errors": errs, "path": str(cal_path)})
                print(f"activate_online_artifacts reject calibration errors={';'.join(errs)}")
                return 2
            # Integrated probe sanity when physics context is provided.
            if physics_root is not None and symbol:
                probe_frame = _load_recent_physics(
                    physics_root, symbol, interval_ms, int(args.sanity_days), int(args.max_sanity_rows)
                )
                ps_ok, ps_errs, ps_sum = evaluate_probe_trigger_sanity(
                    probe_frame,
                    payload,
                    probe_min_triggers=int(args.probe_min_triggers),
                    probe_max_density=float(args.probe_max_density),
                    total_density_min=float(args.total_density_min),
                    total_density_max=float(args.total_density_max),
                )
                cal_probe_summary = dict(ps_sum or {})
                cal_probe_errors = list(ps_errs or [])
                report["calibration"]["probe_sanity"] = {"ok": bool(ps_ok), "errors": cal_probe_errors, "summary": cal_probe_summary}
                report["ok"] = bool(report["ok"] and ps_ok)
                if not ps_ok:
                    validation_reasons.extend(list(ps_errs))
                    _write_validation_report(validation_report_path, report)
                    if bool(args.write_validation_event):
                        _log_event(
                            events,
                            {
                                "event": "validate_probe_sanity",
                                "ok": False,
                                "calibration_path": str(cal_path),
                                "errors": cal_probe_errors,
                                "probe_stats": list(cal_probe_summary.get("probe_stats", []) or []),
                                "total_density": float(cal_probe_summary.get("total_density", 0.0) or 0.0),
                            },
                        )
                    print(f"activate_online_artifacts reject probe_sanity errors={';'.join(ps_errs)}")
                    return 2
                if bool(args.write_validation_event):
                    _log_event(
                        events,
                        {
                            "event": "validate_probe_sanity",
                            "ok": True,
                            "calibration_path": str(cal_path),
                            "errors": [],
                            "probe_stats": list(cal_probe_summary.get("probe_stats", []) or []),
                            "total_density": float(cal_probe_summary.get("total_density", 0.0) or 0.0),
                        },
                    )
                if bool(args.directional_sanity):
                    ds_ok, ds_errs, ds_sum = evaluate_probe_directional_sanity(
                        probe_frame,
                        payload,
                        horizons=dir_horizons or (1, 5),
                        min_dir_triggers=int(args.directional_min_triggers),
                        max_fail_probes=int(args.directional_max_fail_probes),
                    )
                    directional_summary = dict(ds_sum or {})
                    directional_errors = list(ds_errs or [])
                    report["calibration"]["directional_sanity"] = {
                        "ok": bool(ds_ok),
                        "errors": directional_errors,
                        "summary": directional_summary,
                    }
                    report["ok"] = bool(report["ok"] and ds_ok)
                    if not ds_ok:
                        validation_reasons.extend(list(ds_errs))
                        _write_validation_report(validation_report_path, report)
                        if bool(args.write_validation_event):
                            _log_event(
                                events,
                                {
                                    "event": "validate_directional_sanity",
                                    "ok": False,
                                    "calibration_path": str(cal_path),
                                    "errors": directional_errors,
                                    "failed_count": int(directional_summary.get("failed_count", 0) or 0),
                                    "directional_probe_stats": list(directional_summary.get("directional_probe_stats", []) or []),
                                },
                            )
                        print(f"activate_online_artifacts reject directional_sanity errors={';'.join(ds_errs)}")
                        return 2
                    if bool(args.write_validation_event):
                        _log_event(
                            events,
                            {
                                "event": "validate_directional_sanity",
                                "ok": True,
                                "calibration_path": str(cal_path),
                                "errors": [],
                                "failed_count": int(directional_summary.get("failed_count", 0) or 0),
                                "directional_probe_stats": list(directional_summary.get("directional_probe_stats", []) or []),
                            },
                        )
            # Backward-compatible path: read existing validation report when provided.
            elif str(args.validation_report).strip():
                rpt = Path(str(args.validation_report))
                if rpt.exists():
                    try:
                        rp = json.loads(rpt.read_text(encoding="utf-8"))
                        ps = dict(dict(rp.get("calibration", {}) or {}).get("probe_sanity", {}) or {})
                        cal_probe_summary = dict(ps.get("summary", {}) or {})
                        cal_probe_errors = list(ps.get("errors", []) or [])
                        report["calibration"]["probe_sanity"] = {"ok": bool(ps.get("ok", True)), "errors": cal_probe_errors, "summary": cal_probe_summary}
                    except Exception:
                        cal_probe_summary = {}
                        cal_probe_errors = []
            if not cal_probe_summary:
                # fallback minimal summary for history continuity
                cal_probe_summary = {
                    "total_density": 0.0,
                    "probe_stats": [],
                    "sample_count": int(payload.get("sample_count", 0) or 0),
                }
        if exe_path is not None:
            ok, errs, _ = validate_execution_file(exe_path)
            report["execution"] = {"path": str(exe_path), "ok": bool(ok), "errors": errs}
            report["ok"] = bool(report["ok"] and ok)
            if not ok:
                validation_reasons.extend(list(errs))
                _write_validation_report(validation_report_path, report)
                _log_event(events, {"event": "activate", "kind": "execution", "ok": False, "errors": errs, "path": str(exe_path)})
                print(f"activate_online_artifacts reject execution errors={';'.join(errs)}")
                return 2

        _write_validation_report(validation_report_path, report)
        rec = activate_artifacts(
            live_root=live_root,
            calibration_path=(str(cal_path) if cal_path is not None else ""),
            execution_path=(str(exe_path) if exe_path is not None else ""),
            metadata={
                "run_id": str(args.run_id),
                "validation_report_path": str(validation_report_path),
                "validation_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "validation_passed": bool(report.get("ok", False)),
                "validation_reasons": list(validation_reasons),
                "calibration_probe_summary": cal_probe_summary,
                "calibration_probe_errors": cal_probe_errors,
                "calibration_probe_total_density": float(cal_probe_summary.get("total_density", 0.0) or 0.0),
                "directional_sanity_enabled": bool(args.directional_sanity),
                "directional_probe_summary": directional_summary,
                "directional_failed_count": int(directional_summary.get("failed_count", 0) if directional_summary else 0),
                "directional_probe_errors": directional_errors,
            },
        )
        _log_event(events, {"event": "activate", "ok": True, "active": rec})
        print(f"activate_online_artifacts ok live_root={live_root}")
        return 0
    except Exception as e:
        _log_event(events, {"event": "activate", "ok": False, "error": f"{type(e).__name__}:{e}"})
        print(f"activate_online_artifacts error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
