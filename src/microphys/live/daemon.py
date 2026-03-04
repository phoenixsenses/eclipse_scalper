from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.microphys.alpha.calibration import CalibrationContext, load_calibration
from src.microphys.alpha.ensemble import build_ensemble_scores
from src.microphys.alpha.gating import build_gated_ensemble_scores
from src.microphys.alpha.spec import SignalSpec, signal_from_dict
from src.microphys.execution.calibration import load_execution_params
from src.microphys.io.sqlite_reader import SQLiteMicroReader
from src.microphys.live.alerts import append_alerts, evaluate_alerts
from src.microphys.live.config import LiveSettings
from src.microphys.live.metrics import compute_live_metrics, write_status
from src.microphys.live.registry import get_active_artifacts, rollback_to_previous
from src.microphys.risk.guards import check_kill_switch
from src.microphys.risk.policy import load_risk_policy
from src.microphys.risk.portfolio import apply_fill, init_portfolio_state, mark_to_market
from src.microphys.risk.sizer import compute_risk_decision
from src.microphys.sim.papertrade import PaperTradeConfig, generate_papertrades
from tools.build_micro_features import compute_micro_bars_from_frames
from tools.build_physics_signals import compute_physics_signals_frame
from utils.symbols import canonical_symbol


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return v
    except Exception:
        return float(default)


def _state_from_bars(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ts_ms"] = pd.to_numeric(df.get("ts_ms"), errors="coerce").fillna(0).astype("int64")
    out["ts_utc"] = df.get("ts_utc").astype(str)
    out["symbol"] = df.get("symbol").astype(str)
    out["mid"] = pd.to_numeric(df.get("mid"), errors="coerce").fillna(0.0)
    out["microprice"] = pd.to_numeric(df.get("microprice"), errors="coerce").fillna(out["mid"])
    out["spread"] = pd.to_numeric(df.get("spread"), errors="coerce").fillna(0.0)
    out["ofi"] = pd.to_numeric(df.get("ofi"), errors="coerce").fillna(0.0)
    out["trade_intensity"] = pd.to_numeric(df.get("trade_intensity_qty_per_sec"), errors="coerce").fillna(0.0)
    out["top_depth_imbalance"] = pd.to_numeric(df.get("top_depth_imbalance"), errors="coerce").fillna(0.0)
    out["rv_short"] = pd.to_numeric(df.get("rv_short"), errors="coerce").fillna(0.0)
    liq_qty = pd.to_numeric(df.get("liq_qty"), errors="coerce").fillna(0.0)
    out["liq_rate"] = liq_qty / max(1e-9, float(interval_ms) / 1000.0)
    out["qty_sum"] = pd.to_numeric(df.get("qty_sum"), errors="coerce").fillna(0.0)
    out["trade_count"] = pd.to_numeric(df.get("trade_count"), errors="coerce").fillna(0).astype(int)
    return out.sort_values("ts_ms").reset_index(drop=True)


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _calibration_health(live_density_1h: float, baseline_probe_density: float | None) -> tuple[bool, str]:
    d = float(max(0.0, live_density_1h))
    if baseline_probe_density is None:
        return (0.001 <= d <= 0.95), "no_baseline"
    b = float(max(1e-9, baseline_probe_density))
    low = max(0.001, b * 0.10)
    high = min(0.95, b * 3.0 + 0.05)
    if d < low:
        return False, f"too_low:{d:.6f}<low:{low:.6f}"
    if d > high:
        return False, f"too_high:{d:.6f}>high:{high:.6f}"
    return True, "ok"


def _with_lock(lock_path: Path):
    class _Lock:
        def __init__(self, p: Path):
            self.p = p
            self.fd: int | None = None

        def __enter__(self):
            self.p.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(str(self.p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                if self.fd is not None:
                    os.close(self.fd)
            finally:
                try:
                    self.p.unlink(missing_ok=True)
                except Exception:
                    pass
            return False

    return _Lock(lock_path)


def find_latest_completed_run(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    runs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")], reverse=True)
    for r in runs:
        man = r / "manifest.json"
        if not man.exists():
            continue
        try:
            payload = json.loads(man.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("status", "")).lower() == "completed":
            return r
    return None


def load_latest_model_specs(run_root: Path) -> tuple[list[SignalSpec], Dict[str, Any]]:
    run = find_latest_completed_run(run_root)
    if run is None:
        return [], {}
    pointers = _load_json(run / "pointers.json", {})
    cand_path = Path(str(pointers.get("candidates_deduped_jsonl", "")))
    selected_path = Path(str(pointers.get("selected_parquet", "")))
    calib_path = Path(str(pointers.get("calibration_json", "")))
    if not cand_path.exists() or not selected_path.exists():
        return [], {}
    selected_df = pd.read_parquet(selected_path)
    names = set(selected_df.get("signal", pd.Series([], dtype=str)).astype(str).tolist())
    specs: list[SignalSpec] = []
    for line in cand_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        spec = signal_from_dict(json.loads(s))
        if spec.name in names:
            specs.append(spec)
    baseline: Dict[str, Any] = {}
    if calib_path.exists():
        try:
            c = json.loads(calib_path.read_text(encoding="utf-8"))
            q = c.get("quantiles", {}) if isinstance(c, dict) else {}
            spread_q = q.get("spread_z", {})
            if isinstance(spread_q, dict):
                baseline["spread_median"] = float(spread_q.get("0.5", 0.0) or 0.0)
            # Keep references for shift metrics.
            baseline["ofi_ref"] = []
            ofi_q = q.get("F_ofi_z", {})
            if isinstance(ofi_q, dict):
                for k in sorted(ofi_q.keys()):
                    baseline["ofi_ref"].append(float(ofi_q[k]))
        except Exception:
            pass
    return specs, baseline


def load_latest_regime_experts(run_root: Path, symbol: str, explicit_path: str = "") -> tuple[pd.DataFrame, str, str]:
    if str(explicit_path).strip():
        p = Path(str(explicit_path))
        if p.exists() and p.is_file():
            return pd.read_parquet(p), str(p), ""
        return pd.DataFrame(), str(p), ""
    run = find_latest_completed_run(run_root)
    if run is None:
        return pd.DataFrame(), "", ""
    ptr = _load_json(run / "pointers.json", {})
    p = str(ptr.get("ensemble_regime_experts_parquet", "")).strip()
    if p and Path(p).exists() and Path(p).is_file():
        return pd.read_parquet(Path(p)), p, run.name
    # fallback to conventional artifact location
    candidates = sorted((run / "artifacts").glob(f"interval_ms=*/symbol={symbol}/ensemble_regime_experts.parquet"))
    if candidates:
        return pd.read_parquet(candidates[-1]), str(candidates[-1]), run.name
    return pd.DataFrame(), "", run.name


def _append_parquet(path: Path, df: pd.DataFrame, subset_cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, df], ignore_index=True) if not df.empty else old
        if subset_cols:
            avail = [c for c in subset_cols if c in merged.columns]
            if avail:
                merged = merged.drop_duplicates(subset=avail, keep="last")
        merged.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _resolve_exec_params(cfg: LiveSettings, run_root: Path) -> tuple[Dict[str, Any], str, str, bool]:
    if str(cfg.execution_params_path).strip():
        p = Path(str(cfg.execution_params_path))
        if p.exists():
            return load_execution_params(p), str(p), "", True
        return {}, str(p), "", False
    run = find_latest_completed_run(run_root)
    if run is None:
        return {}, "", "", False
    pointers = _load_json(run / "pointers.json", {})
    p = str(pointers.get("execution_params_json", "")).strip()
    if p and Path(p).exists():
        return load_execution_params(Path(p)), p, run.name, True
    return {}, p, run.name, False


def _resolve_artifacts(
    cfg: LiveSettings,
    *,
    run_root: Path,
    live_root: Path,
    sticky: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    active: Dict[str, Any] = {}
    if sticky is not None:
        active = dict(sticky)
    elif bool(cfg.use_active_artifacts):
        active = get_active_artifacts(live_root)

    run = find_latest_completed_run(run_root)
    pointers = _load_json(run / "pointers.json", {}) if run is not None else {}

    cal_path = str(active.get("calibration_json_path", "")).strip()
    if not cal_path:
        cal_path = str(pointers.get("calibration_json", "")).strip()

    exec_hint = str(active.get("execution_params_json_path", "")).strip()
    exec_run_id = str(active.get("run_id", "")).strip() if exec_hint else ""
    if not exec_hint and run is not None:
        exec_hint = str(pointers.get("execution_params_json", "")).strip()
        exec_run_id = run.name if exec_hint else ""

    experts_hint = str(pointers.get("ensemble_regime_experts_parquet", "")).strip() if run is not None else ""
    aligned_hint = str(pointers.get("aligned_regimes_path", "")).strip() if run is not None else ""
    return {
        "active": active,
        "calibration_path": cal_path,
        "execution_path_hint": exec_hint,
        "execution_run_id": exec_run_id,
        "experts_path_hint": experts_hint,
        "aligned_regimes_path_hint": aligned_hint,
        "model_run_id": (run.name if run is not None else ""),
    }


def _events_to_frames(
    reader: SQLiteMicroReader,
    symbol: str,
    start_ts: float,
    end_ts: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = [asdict(x) for x in reader.read_trades(symbol, start_ts, end_ts)]
    book = [asdict(x) for x in reader.read_top_of_book(symbol, start_ts, end_ts)]
    liq = [asdict(x) for x in reader.read_liquidations(symbol, start_ts, end_ts)]
    return pd.DataFrame(trades), pd.DataFrame(book), pd.DataFrame(liq)


def run_live_cycle(cfg: LiveSettings, *, artifact_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
    symbol = canonical_symbol(cfg.symbol)
    out_root = Path(str(cfg.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    watermark_path = out_root / "watermark.json"
    heartbeat_path = out_root / "heartbeat.json"
    status_path = out_root / "status.json"
    live_trades_path = out_root / "papertrades_live.parquet"
    gating_live_path = out_root / "gating_live.parquet"
    positions_live_path = out_root / "positions_live.parquet"
    risk_snapshot_path = out_root / "risk_snapshot.json"
    alerts_path = Path("logs/live_alerts.jsonl")
    cal_events_path = Path("logs/calibration_events.jsonl")
    risk_events_path = Path("logs/risk_events.jsonl")
    lock_path = out_root / "watermark.lock"

    with _with_lock(lock_path):
        wm = _load_json(watermark_path, {"last_ts_ms": 0})
        last_ts_ms = int(wm.get("last_ts_ms", 0) or 0)
        now = _epoch_now()
        lookback_sec = max(60.0, float(cfg.lookback_hours) * 3600.0)
        start_ts = max(0.0, now - lookback_sec)
        if last_ts_ms > 0:
            # Keep overlap for rolling features.
            start_ts = max(start_ts, (last_ts_ms / 1000.0) - 300.0)

        reader = SQLiteMicroReader(Path(str(cfg.db_path)))
        trades_df, book_df, liq_df = _events_to_frames(reader, symbol, start_ts, now)
        bars = compute_micro_bars_from_frames(
            trades_df,
            book_df,
            liq_df,
            symbol=symbol,
            start_ts=start_ts,
            end_ts=now,
            interval_ms=int(cfg.interval_ms),
        )
        state = _state_from_bars(bars, int(cfg.interval_ms))
        physics = compute_physics_signals_frame(state, horizons=[1, 5, 10, 20], rolling=500)
        physics = physics.sort_values("ts_ms").reset_index(drop=True)
        if "regime_id" not in physics.columns:
            physics["regime_id"] = (pd.to_numeric(physics.get("F_ofi_z"), errors="coerce").fillna(0.0) >= 0.0).astype(int)
        if physics.empty:
            status = {
                "ts_utc": _utc_now(),
                "state": "degraded",
                "reason": "no_physics_rows",
                "data_freshness_sec": float("inf"),
            }
            write_status(status_path, status)
            _write_json(heartbeat_path, {"ts_utc": _utc_now(), "ok": False, "reason": "no_physics_rows"})
            return {"status": status, "new_trades": 0}

        specs, baseline = load_latest_model_specs(Path(str(cfg.run_root)))
        if not specs:
            status = {
                "ts_utc": _utc_now(),
                "state": "degraded",
                "reason": "no_latest_model_specs",
            }
            write_status(status_path, status)
            _write_json(heartbeat_path, {"ts_utc": _utc_now(), "ok": False, "reason": "no_latest_model_specs"})
            return {"status": status, "new_trades": 0}

        art = _resolve_artifacts(
            cfg,
            run_root=Path(str(cfg.run_root)),
            live_root=out_root,
            sticky=artifact_snapshot,
        )
        cal_ctx: CalibrationContext | None = None
        cal_path_used = str(art.get("calibration_path", "")).strip()
        if cal_path_used and Path(cal_path_used).exists():
            try:
                cal_ctx = load_calibration(Path(cal_path_used))
            except Exception:
                cal_ctx = None

        ensemble = build_ensemble_scores(physics, specs, calibration=cal_ctx)
        frame = physics.merge(ensemble[["ts_ms", "ensemble_side", "signal_count", "ensemble_score"]], on="ts_ms", how="left")
        frame["ensemble_side"] = pd.to_numeric(frame.get("ensemble_side"), errors="coerce").fillna(0.0)
        frame["signal_count"] = pd.to_numeric(frame.get("signal_count"), errors="coerce").fillna(0).astype(int)
        frame["ensemble_score"] = pd.to_numeric(frame.get("ensemble_score"), errors="coerce").fillna(0.0)

        experts_used = False
        experts_loaded = False
        experts_source_run_id = ""
        experts_pointer_path = ""
        aligned_regimes_loaded = False
        experts_reason = "disabled"
        gating_df = pd.DataFrame()
        if bool(cfg.use_regime_experts):
            hint = str(art.get("experts_path_hint", "")).strip()
            explicit = str(cfg.experts_path).strip() or hint
            experts_df, experts_pointer_path, experts_source_run_id = load_latest_regime_experts(
                Path(str(cfg.run_root)), symbol, explicit_path=explicit
            )
            experts_loaded = bool(not experts_df.empty)
            if not experts_df.empty:
                regime_col = "aligned_regime_id"
                aligned_hint = str(art.get("aligned_regimes_path_hint", "")).strip()
                aligned_raw = str(cfg.aligned_regimes_path).strip() or aligned_hint
                if aligned_raw:
                    ap = Path(aligned_raw)
                    if ap.exists() and ap.is_file():
                        aligned = pd.read_parquet(ap)
                        amap = aligned[aligned.get("symbol", pd.Series([], dtype=str)).astype(str) == symbol]
                        if not amap.empty and "aligned_regime_id" in amap.columns and "ts_ms" in amap.columns:
                            frame = frame.merge(
                                amap[["ts_ms", "aligned_regime_id"]].drop_duplicates(subset=["ts_ms"], keep="last"),
                                on="ts_ms",
                                how="left",
                            )
                            aligned_regimes_loaded = True
                if "aligned_regime_id" not in frame.columns:
                    frame["aligned_regime_id"] = pd.to_numeric(frame.get("regime_id"), errors="coerce").fillna(-1).astype(int)
                gated_ens, gating_df = build_gated_ensemble_scores(
                    frame,
                    specs,
                    experts_df,
                    calibration=cal_ctx,
                    regime_col=regime_col,
                    global_ensemble=ensemble,
                    data_quality_ok=True,
                )
                if not gated_ens.empty:
                    frame = frame.drop(columns=[c for c in ["ensemble_score", "ensemble_side", "signal_count"] if c in frame.columns]).merge(
                        gated_ens[["ts_ms", "ensemble_score", "ensemble_side", "signal_count"]],
                        on="ts_ms",
                        how="left",
                    )
                    frame["ensemble_side"] = pd.to_numeric(frame.get("ensemble_side"), errors="coerce").fillna(0.0)
                    frame["signal_count"] = pd.to_numeric(frame.get("signal_count"), errors="coerce").fillna(0).astype(int)
                    frame["ensemble_score"] = pd.to_numeric(frame.get("ensemble_score"), errors="coerce").fillna(0.0)
                    experts_used = True
                    experts_reason = "expert_active"
                    if not gating_df.empty:
                        gsave = gating_df.copy()
                        gsave["ts_utc"] = frame.get("ts_utc")
                        _append_parquet(gating_live_path, gsave, subset_cols=["ts_ms", "active_expert_id"])
                else:
                    experts_reason = "gated_ensemble_empty"
            else:
                experts_reason = "experts_missing"
        tsf = pd.to_numeric(frame.get("ts_ms"), errors="coerce")
        if tsf.notna().any():
            max_ts = float(tsf.max())
            frame_1h = frame[tsf >= (max_ts - 3_600_000.0)].copy()
        else:
            frame_1h = frame.tail(min(len(frame), 10_000)).copy()
        live_trade_density_1h = float((pd.to_numeric(frame_1h.get("signal_count"), errors="coerce").fillna(0) > 0).mean()) if not frame_1h.empty else 0.0

        horizon = max(1, int(pd.Series([s.horizon_bars for s in specs], dtype=float).median()))
        exec_params_path_hint = str(art.get("execution_path_hint", "")).strip()
        exec_params_run_id = str(art.get("execution_run_id", "")).strip()
        if str(cfg.execution_params_path).strip():
            exec_params, exec_params_path_used, exec_params_run_id, exec_loaded = _resolve_exec_params(cfg, Path(str(cfg.run_root)))
        elif exec_params_path_hint and Path(exec_params_path_hint).exists():
            exec_params = load_execution_params(Path(exec_params_path_hint))
            exec_params_path_used = exec_params_path_hint
            exec_loaded = True
        else:
            exec_params, exec_params_path_used, exec_params_run_id, exec_loaded = _resolve_exec_params(cfg, Path(str(cfg.run_root)))
        effective_exec_model = str(cfg.execution_model)
        if effective_exec_model != "simple" and not exec_loaded:
            effective_exec_model = "simple"
        trades = generate_papertrades(
            frame,
            horizon_bars=horizon,
            cfg=PaperTradeConfig(
                mode=str(cfg.mode),
                fee_bps=float(cfg.fee_bps),
                execution_model=effective_exec_model,
                execution_params=exec_params,
                ttl_bars=int(cfg.maker_ttl_bars),
            ),
        )
        if not trades.empty:
            entry_ts = pd.to_datetime(trades["entry_ts_utc"], utc=True, errors="coerce")
            entry_ms = pd.Series(0, index=trades.index, dtype="int64")
            valid = entry_ts.notna()
            if bool(valid.any()):
                entry_ms.loc[valid] = (entry_ts.loc[valid].astype("int64") // 10**6).astype("int64")
            trades["_entry_ts_ms"] = entry_ms
            if last_ts_ms > 0:
                trades = trades[trades["_entry_ts_ms"] > int(last_ts_ms)]

        risk_state = _load_json(risk_snapshot_path, {})
        if not risk_state:
            risk_state = init_portfolio_state(float(cfg.starting_equity))
        elif "symbols" not in risk_state:
            risk_state["symbols"] = {}
        now_ts_ms = int(_safe_float(pd.to_numeric(physics.get("ts_ms"), errors="coerce").max(), 0.0))
        mid_map = {
            str(symbol): _safe_float(pd.to_numeric(frame.get("mid"), errors="coerce").iloc[-1] if not frame.empty else 0.0, 0.0)
        }
        mtm_before = mark_to_market(risk_state, mid_map)
        kill_active = False
        kill_reason = "OK"
        policy = None
        risk_skips = 0
        if bool(cfg.enable_risk_engine):
            policy = load_risk_policy(str(cfg.risk_policy_path), starting_equity_override=float(cfg.starting_equity))
            kill_active, kill_reason = check_kill_switch(risk_state, mtm_before, policy, now_ts_ms)
            prev_status = _load_json(status_path, {})
            row_lookup = {}
            if "ts_utc" in frame.columns:
                for _, rr in frame.iterrows():
                    row_lookup[str(rr.get("ts_utc", ""))] = rr.to_dict()
            gate_lookup = {}
            if not gating_df.empty and "ts_ms" in gating_df.columns:
                gate_lookup = {int(_safe_float(r.get("ts_ms"), -1)): r.to_dict() for _, r in gating_df.iterrows()}
            kept = []
            for _, tr in trades.iterrows():
                entry_ts_utc = str(tr.get("entry_ts_utc", ""))
                entry_ts_ms = int(pd.to_datetime(pd.Series([entry_ts_utc]), utc=True, errors="coerce").astype("int64").iloc[0] // 10**6)
                side = str(tr.get("side", "buy"))
                sig_row = dict(row_lookup.get(entry_ts_utc, {}))
                ts_key = int(_safe_float(sig_row.get("ts_ms"), entry_ts_ms))
                gate_row = dict(gate_lookup.get(ts_key, {}))
                desired = "KILL" if kill_active else "TRADE"
                if kill_active:
                    decision = {
                        "action": "KILL",
                        "reason": kill_reason,
                        "final_notional": 0.0,
                        "base_notional": 0.0,
                        "factors": {},
                    }
                else:
                    rd = compute_risk_decision(
                        ts_ms=entry_ts_ms,
                        symbol=symbol,
                        desired_side=side,
                        signal_row=sig_row,
                        gating_row=gate_row,
                        live_status=prev_status,
                        policy=policy,
                        mtm=mtm_before,
                    )
                    decision = {
                        "action": rd.action,
                        "reason": rd.reason,
                        "final_notional": rd.final_notional,
                        "base_notional": rd.base_notional,
                        "factors": rd.factors,
                    }
                if str(decision["action"]) != "TRADE":
                    risk_skips += 1
                    _append_jsonl(
                        risk_events_path,
                        {
                            "ts_utc": _utc_now(),
                            "event": str(decision["reason"]),
                            "symbol": symbol,
                            "entry_ts_utc": entry_ts_utc,
                            "action": str(decision["action"]),
                            "side": side,
                            "factors": dict(decision.get("factors", {}) or {}),
                        },
                    )
                    continue
                notional = float(decision["final_notional"])
                trow = tr.copy()
                trow["trade_notional"] = notional
                trow["risk_reason"] = str(decision["reason"])
                trow["risk_factors_json"] = json.dumps(decision.get("factors", {}), ensure_ascii=True, sort_keys=True)
                trow["pnl_net_notional"] = _safe_float(trow.get("pnl_net"), 0.0) * notional
                trow["pnl_gross_notional"] = _safe_float(trow.get("pnl_gross"), 0.0) * notional
                fee_notional = abs(notional) * float(cfg.fee_bps) / 10000.0
                trow["fee_notional"] = fee_notional
                if bool(trow.get("filled", True)):
                    apply_fill(
                        risk_state,
                        symbol=symbol,
                        side=side,
                        fill_price=_safe_float(trow.get("fill_price", trow.get("entry_price", 0.0)), 0.0),
                        notional=notional,
                        fee_notional=fee_notional,
                    )
                    mtm_before = mark_to_market(risk_state, mid_map)
                kept.append(trow)
            trades = pd.DataFrame(kept) if kept else trades.iloc[:0].copy()

        _append_parquet(live_trades_path, trades, subset_cols=["_entry_ts_ms", "side"])
        live_all = pd.read_parquet(live_trades_path) if live_trades_path.exists() else trades

        mtm_after = mark_to_market(risk_state, mid_map)
        if bool(cfg.enable_risk_engine):
            risk_snapshot = dict(risk_state)
            risk_snapshot.update(
                {
                    "ts_utc": _utc_now(),
                    "equity": float(mtm_after.get("equity", 0.0)),
                    "gross_notional": float(mtm_after.get("gross_notional", 0.0)),
                    "drawdown_pct": float(mtm_after.get("drawdown_pct", 0.0)),
                    "kill_active": bool(kill_active),
                    "kill_reason": str(kill_reason),
                }
            )
            _write_json(risk_snapshot_path, risk_snapshot)
            pos_rows = []
            for sym, row in dict(mtm_after.get("by_symbol", {}) or {}).items():
                pos_rows.append(
                    {
                        "ts_ms": now_ts_ms,
                        "ts_utc": _utc_now(),
                        "symbol": str(sym),
                        "position_qty": float(row.get("qty", 0.0)),
                        "avg_entry_price": float(row.get("avg_entry_price", 0.0)),
                        "mid_price": float(row.get("mid", 0.0)),
                        "realized_pnl": float(risk_state.get("realized_pnl", 0.0)),
                        "unrealized_pnl": float(row.get("unrealized_pnl", 0.0)),
                        "equity": float(mtm_after.get("equity", 0.0)),
                        "gross_notional": float(mtm_after.get("gross_notional", 0.0)),
                        "drawdown_pct": float(mtm_after.get("drawdown_pct", 0.0)),
                    }
                )
            if not pos_rows:
                pos_rows.append(
                    {
                        "ts_ms": now_ts_ms,
                        "ts_utc": _utc_now(),
                        "symbol": symbol,
                        "position_qty": 0.0,
                        "avg_entry_price": 0.0,
                        "mid_price": float(mid_map.get(symbol, 0.0)),
                        "realized_pnl": float(risk_state.get("realized_pnl", 0.0)),
                        "unrealized_pnl": 0.0,
                        "equity": float(mtm_after.get("equity", 0.0)),
                        "gross_notional": float(mtm_after.get("gross_notional", 0.0)),
                        "drawdown_pct": float(mtm_after.get("drawdown_pct", 0.0)),
                    }
                )
            _append_parquet(positions_live_path, pd.DataFrame(pos_rows), subset_cols=["ts_ms", "symbol"])

        tmax = int(pd.to_numeric(physics.get("ts_ms"), errors="coerce").max() or 0)
        _write_json(watermark_path, {"last_ts_ms": tmax, "updated_utc": _utc_now()})

        _, db_last = reader.get_ts_range("trades", symbol)
        metrics = compute_live_metrics(
            physics_recent=physics.tail(max(1000, min(len(physics), 50_000))),
            live_trades=live_all.tail(50_000) if not live_all.empty else live_all,
            baseline=baseline,
            db_last_event_ts=db_last,
            interval_ms=int(cfg.interval_ms),
        )
        metrics["execution_model"] = str(effective_exec_model)
        metrics["execution_params_path"] = str(exec_params_path_used)
        metrics["execution_params_loaded"] = bool(exec_loaded)
        metrics["execution_params_run_id"] = str(exec_params_run_id)
        active_hashes = dict(art.get("active", {}).get("hashes", {}) or {})
        active_probe_density_raw = art.get("active", {}).get("calibration_probe_total_density")
        active_probe_density = float(active_probe_density_raw) if active_probe_density_raw is not None else None
        cal_ok, cal_reason = _calibration_health(live_trade_density_1h, active_probe_density)
        metrics["active_calibration_path"] = str(cal_path_used)
        metrics["active_calibration_sha256"] = str(active_hashes.get("calibration_sha256", ""))
        metrics["active_execution_sha256"] = str(active_hashes.get("execution_sha256", ""))
        metrics["active_artifacts_activated_ts"] = str(art.get("active", {}).get("activated_ts", ""))
        metrics["live_trade_density_1h"] = float(live_trade_density_1h)
        metrics["active_calibration_probe_total_density"] = (float(active_probe_density) if active_probe_density is not None else 0.0)
        metrics["calibration_health_ok"] = bool(cal_ok)
        metrics["calibration_health_reason"] = str(cal_reason)
        metrics["use_regime_experts"] = bool(cfg.use_regime_experts)
        metrics["regime_experts_used"] = bool(experts_used)
        metrics["experts_loaded"] = bool(experts_loaded)
        metrics["aligned_regimes_loaded"] = bool(aligned_regimes_loaded)
        metrics["experts_source_run_id"] = str(experts_source_run_id)
        metrics["experts_pointer_path"] = str(experts_pointer_path)
        metrics["experts_reason"] = str(experts_reason)
        metrics["gating_fallback_rate"] = (
            float(pd.to_numeric(gating_df.get("fallback_used"), errors="coerce").fillna(0.0).mean()) if not gating_df.empty else 0.0
        )
        metrics["gating_confidence_mean"] = (
            float(pd.to_numeric(gating_df.get("confidence_score"), errors="coerce").fillna(0.0).mean()) if not gating_df.empty else 0.0
        )
        metrics["risk_engine_enabled"] = bool(cfg.enable_risk_engine)
        metrics["risk_equity"] = float(mtm_after.get("equity", 0.0)) if bool(cfg.enable_risk_engine) else 0.0
        metrics["risk_drawdown_pct"] = float(mtm_after.get("drawdown_pct", 0.0)) if bool(cfg.enable_risk_engine) else 0.0
        metrics["risk_gross_notional"] = float(mtm_after.get("gross_notional", 0.0)) if bool(cfg.enable_risk_engine) else 0.0
        metrics["risk_kill_active"] = bool(kill_active) if bool(cfg.enable_risk_engine) else False
        metrics["risk_kill_reason"] = str(kill_reason) if bool(cfg.enable_risk_engine) else ""
        metrics["risk_skips_count"] = int(risk_skips) if bool(cfg.enable_risk_engine) else 0
        write_status(status_path, metrics)
        alerts = evaluate_alerts(metrics, cfg)
        if not cal_ok:
            cal_alert = {
                "ts_utc": _utc_now(),
                "code": "CALIBRATION_SUSPECTED_BAD",
                "severity": "warn",
                "detail": f"reason={cal_reason} live_density_1h={live_trade_density_1h:.6f}",
            }
            alerts.append(cal_alert)
            _append_jsonl(
                cal_events_path,
                {
                    "event": "calibration_suspected_bad",
                    "ts_utc": _utc_now(),
                    "reason": cal_reason,
                    "live_trade_density_1h": float(live_trade_density_1h),
                    "active_calibration_probe_total_density": (float(active_probe_density) if active_probe_density is not None else None),
                    "active_calibration_path": str(cal_path_used),
                },
            )
            if bool(cfg.auto_rollback_on_bad_calibration):
                try:
                    rb = rollback_to_previous("calibration", live_root=out_root)
                    _append_jsonl(
                        cal_events_path,
                        {
                            "event": "auto_rollback",
                            "kind": "calibration",
                            "ts_utc": _utc_now(),
                            "ok": True,
                            "active": rb,
                        },
                    )
                except Exception as e:
                    _append_jsonl(
                        cal_events_path,
                        {
                            "event": "auto_rollback",
                            "kind": "calibration",
                            "ts_utc": _utc_now(),
                            "ok": False,
                            "error": f"{type(e).__name__}:{e}",
                        },
                    )
        append_alerts(alerts_path, alerts)
        _write_json(
            heartbeat_path,
            {
                "ts_utc": _utc_now(),
                "ok": True,
                "new_trades": int(len(trades)),
                "last_ts_ms": tmax,
            },
        )
        return {
            "status": metrics,
            "new_trades": int(len(trades)),
            "alerts": alerts,
            "execution_model": effective_exec_model,
            "execution_params_path": exec_params_path_used,
            "execution_params_loaded": exec_loaded,
            "active_calibration_path": cal_path_used,
            "calibration_health_ok": bool(cal_ok),
            "live_trade_density_1h": float(live_trade_density_1h),
            "regime_experts_used": bool(experts_used),
            "experts_loaded": bool(experts_loaded),
            "aligned_regimes_loaded": bool(aligned_regimes_loaded),
            "risk_engine_enabled": bool(cfg.enable_risk_engine),
            "risk_skips_count": int(risk_skips) if bool(cfg.enable_risk_engine) else 0,
        }


def run_daemon(cfg: LiveSettings, *, max_cycles: int | None = None) -> int:
    cycles = 0
    backoff = 1.0
    sticky_artifacts: Dict[str, Any] | None = None
    if bool(cfg.use_active_artifacts) and bool(cfg.disable_online_reload):
        sticky_artifacts = get_active_artifacts(Path(str(cfg.out_root)))
    while True:
        try:
            res = run_live_cycle(cfg, artifact_snapshot=sticky_artifacts)
            print(
                f"[live] cycle ok symbol={canonical_symbol(cfg.symbol)} new_trades={int(res.get('new_trades', 0))} "
                f"state={res.get('status', {}).get('state', 'na')} "
                f"exec_model={res.get('execution_model', 'na')} exec_params_loaded={int(bool(res.get('execution_params_loaded', False)))}"
            )
            backoff = 1.0
        except KeyboardInterrupt:
            print("[live] shutdown requested")
            return 0
        except Exception as e:
            print(f"[live] cycle error {type(e).__name__}:{e}; retry in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(60.0, backoff * 2.0)
        cycles += 1
        if max_cycles is not None and cycles >= int(max_cycles):
            return 0
        time.sleep(max(0.1, float(cfg.refresh_sec)))
