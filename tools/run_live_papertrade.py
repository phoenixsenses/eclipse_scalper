from __future__ import annotations

import argparse

from src.microphys.live.config import LiveSettings
from src.microphys.live.daemon import run_daemon


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live paper-trade daemon (research mode, no real orders).")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--lookback-hours", type=float, default=24.0)
    p.add_argument("--refresh-sec", type=float, default=30.0)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--execution-model", choices=["simple", "maker_queue", "maker_hazard"], default="simple")
    p.add_argument("--execution-params", default="")
    p.add_argument("--use-active-artifacts", dest="use_active_artifacts", action="store_true")
    p.add_argument("--no-use-active-artifacts", dest="use_active_artifacts", action="store_false")
    p.add_argument("--disable-online-reload", action="store_true")
    p.add_argument("--auto-rollback-on-bad-calibration", action="store_true")
    p.add_argument("--use-regime-experts", action="store_true")
    p.add_argument("--experts-path", default="")
    p.add_argument("--aligned-regimes-path", default="")
    p.add_argument("--enable-risk-engine", action="store_true")
    p.add_argument("--risk-policy", default="")
    p.add_argument("--starting-equity", type=float, default=10000.0)
    p.add_argument("--exec-engine-unified", action="store_true")
    p.add_argument("--exec-event-bus-enabled", action="store_true")
    p.add_argument("--exec-runtime-supervisor-enabled", action="store_true")
    p.add_argument("--exec-kill-on-contract-violation", action="store_true")
    p.add_argument("--supervisor-max-feed-age-sec", type=float, default=120.0)
    p.add_argument("--supervisor-max-order-age-sec", type=float, default=600.0)
    p.add_argument("--supervisor-max-loop-errors", type=int, default=3)
    p.add_argument("--maker-ttl-bars", type=int, default=10)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--latency-bars", type=int, default=2)
    p.add_argument("--out-root", default="data/live")
    p.add_argument("--run-root", default="data/runs/alpha")
    p.add_argument("--max-cycles", type=int, default=0, help="0 means run forever")
    p.set_defaults(use_active_artifacts=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = LiveSettings(
        db_path=str(args.db),
        symbol=str(args.symbol),
        interval_ms=int(args.interval_ms),
        lookback_hours=float(args.lookback_hours),
        refresh_sec=float(args.refresh_sec),
        mode=str(args.mode),
        execution_model=str(args.execution_model),
        execution_params_path=str(args.execution_params),
        use_active_artifacts=bool(args.use_active_artifacts),
        disable_online_reload=bool(args.disable_online_reload),
        auto_rollback_on_bad_calibration=bool(args.auto_rollback_on_bad_calibration),
        maker_ttl_bars=int(args.maker_ttl_bars),
        fee_bps=float(args.fee_bps),
        latency_bars=int(args.latency_bars),
        out_root=str(args.out_root),
        run_root=str(args.run_root),
        use_regime_experts=bool(args.use_regime_experts),
        experts_path=str(args.experts_path),
        aligned_regimes_path=str(args.aligned_regimes_path),
        enable_risk_engine=bool(args.enable_risk_engine),
        risk_policy_path=str(args.risk_policy),
        starting_equity=float(args.starting_equity),
        exec_engine_unified=bool(args.exec_engine_unified),
        exec_event_bus_enabled=bool(args.exec_event_bus_enabled),
        exec_runtime_supervisor_enabled=bool(args.exec_runtime_supervisor_enabled),
        exec_kill_on_contract_violation=bool(args.exec_kill_on_contract_violation),
        supervisor_max_feed_age_sec=float(args.supervisor_max_feed_age_sec),
        supervisor_max_order_age_sec=float(args.supervisor_max_order_age_sec),
        supervisor_max_loop_errors=int(args.supervisor_max_loop_errors),
    )
    max_cycles = None if int(args.max_cycles) <= 0 else int(args.max_cycles)
    return run_daemon(cfg, max_cycles=max_cycles)


if __name__ == "__main__":
    raise SystemExit(main())
