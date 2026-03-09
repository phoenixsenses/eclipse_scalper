from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.book_proxy_pressure_alerts import build_payload as build_book_proxy_pressure_alerts
from tools.book_proxy_pressure_state import build_state_payload as build_book_proxy_pressure_state
from tools.book_proxy_pressure_watchlist import build_watchlist_payload as build_book_proxy_pressure_watchlist
from tools.fill_toxicity_state import build_state_payload as build_fill_toxicity_state
from tools.latency_stress_state import build_state_payload as build_latency_stress_state
from tools.liquidation_alert_state import build_state_payload as build_liquidation_alert_state
from tools.liquidation_regime_alerts import build_alert_payload as build_liquidation_alerts
from tools.liquidation_watchlist import build_watchlist_payload as build_liquidation_watchlist
from tools.research_event_watchboard import build_watchboard_payload
from tools.return_shock_alerts import build_payload as build_return_shock_alerts
from tools.return_shock_state import build_state_payload as build_return_shock_state
from tools.return_shock_watchlist import build_watchlist_payload as build_return_shock_watchlist
from tools.run_summary import build_run_summary
from tools.spread_stress_alerts import build_payload as build_spread_stress_alerts
from tools.spread_stress_state import build_state_payload as build_spread_stress_state
from tools.spread_stress_watchlist import build_watchlist_payload as build_spread_stress_watchlist
from tools.toxicity_report import _load as load_toxicity_rows
from tools.toxicity_report import build_toxicity_report
from tools.execution_diagnostics import _load_rows as load_execution_rows
from tools.execution_diagnostics import compute_execution_diagnostics
from tools.volatility_burst_alerts import build_payload as build_volatility_burst_alerts
from tools.volatility_burst_state import build_state_payload as build_volatility_burst_state
from tools.volatility_burst_watchlist import build_watchlist_payload as build_volatility_burst_watchlist
from tools.volume_vacuum_alerts import build_payload as build_volume_vacuum_alerts
from tools.volume_vacuum_state import build_state_payload as build_volume_vacuum_state
from tools.volume_vacuum_watchlist import build_watchlist_payload as build_volume_vacuum_watchlist


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw).split(",") if s.strip()]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_md(path: Path, title: str, payload: Dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        json.dumps(payload.get("run_summary") or {}, indent=2, ensure_ascii=True),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_paths(reports_dir: Path) -> Dict[str, Path]:
    names = [
        "RESEARCH_EVENT_WATCHBOARD_REAL",
        "LIQUIDATION_ALERT_STATE_REAL",
        "LIQUIDATION_WATCHLIST_REAL",
        "SPREAD_STRESS_STATE_REAL",
        "SPREAD_STRESS_WATCHLIST_REAL",
        "FILL_TOXICITY_STATE_REAL",
        "LATENCY_STRESS_STATE_REAL",
        "RETURN_SHOCK_STATE_REAL",
        "RETURN_SHOCK_WATCHLIST_REAL",
        "VOLUME_VACUUM_STATE_REAL",
        "VOLUME_VACUUM_WATCHLIST_REAL",
        "VOLATILITY_BURST_STATE_REAL",
        "VOLATILITY_BURST_WATCHLIST_REAL",
        "BOOK_PROXY_PRESSURE_STATE_REAL",
        "BOOK_PROXY_PRESSURE_WATCHLIST_REAL",
    ]
    out: Dict[str, Path] = {}
    for name in names:
        out[f"{name}_json"] = reports_dir / f"{name}.json"
        out[f"{name}_md"] = reports_dir / f"{name}.md"
    return out


def build_refresh_payload(
    *,
    micro_db: str,
    trade_source: str,
    primary_symbol: str,
    symbols: List[str],
    lookback_min: int,
    bucket_sec: int,
    recent_limit: int,
    top_n: int,
    reports_dir: str,
) -> Dict[str, Any]:
    reports_path = Path(reports_dir)
    artifacts = _artifact_paths(reports_path)

    liquidation_alerts = build_liquidation_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        rule="high_liq_reversal_regime",
        recent_limit=recent_limit,
        min_liq_rate=0.0,
        out_json=str(reports_path / "LIQUIDATION_REGIME_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "LIQUIDATION_REGIME_ALERTS_REAL.tmp.md"),
    )
    liquidation_state = build_liquidation_alert_state(
        alert_payload=liquidation_alerts,
        source_json=str(reports_path / "LIQUIDATION_REGIME_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["LIQUIDATION_ALERT_STATE_REAL_json"]),
        out_md=str(artifacts["LIQUIDATION_ALERT_STATE_REAL_md"]),
    )
    liquidation_watchlist = build_liquidation_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        rule="high_liq_reversal_regime",
        recent_limit=recent_limit,
        min_liq_rate=0.0,
        top_n=top_n,
        out_json=str(artifacts["LIQUIDATION_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["LIQUIDATION_WATCHLIST_REAL_md"]),
    )

    spread_alerts = build_spread_stress_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        out_json=str(reports_path / "SPREAD_STRESS_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "SPREAD_STRESS_ALERTS_REAL.tmp.md"),
    )
    spread_state = build_spread_stress_state(
        alert_payload=spread_alerts,
        source_json=str(reports_path / "SPREAD_STRESS_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["SPREAD_STRESS_STATE_REAL_json"]),
        out_md=str(artifacts["SPREAD_STRESS_STATE_REAL_md"]),
    )
    spread_watchlist = build_spread_stress_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["SPREAD_STRESS_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["SPREAD_STRESS_WATCHLIST_REAL_md"]),
    )

    fill_state = build_fill_toxicity_state(
        source=trade_source,
        report_payload=build_toxicity_report(load_toxicity_rows(Path(trade_source))),
        out_json=str(artifacts["FILL_TOXICITY_STATE_REAL_json"]),
        out_md=str(artifacts["FILL_TOXICITY_STATE_REAL_md"]),
    )
    latency_state = build_latency_stress_state(
        source=trade_source,
        diag=compute_execution_diagnostics(load_execution_rows(Path(trade_source))),
        out_json=str(artifacts["LATENCY_STRESS_STATE_REAL_json"]),
        out_md=str(artifacts["LATENCY_STRESS_STATE_REAL_md"]),
    )

    return_shock_alerts = build_return_shock_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        out_json=str(reports_path / "RETURN_SHOCK_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "RETURN_SHOCK_ALERTS_REAL.tmp.md"),
    )
    return_shock_state = build_return_shock_state(
        alert_payload=return_shock_alerts,
        source_json=str(reports_path / "RETURN_SHOCK_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["RETURN_SHOCK_STATE_REAL_json"]),
        out_md=str(artifacts["RETURN_SHOCK_STATE_REAL_md"]),
    )
    return_shock_watchlist = build_return_shock_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["RETURN_SHOCK_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["RETURN_SHOCK_WATCHLIST_REAL_md"]),
    )

    volume_vacuum_alerts = build_volume_vacuum_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        out_json=str(reports_path / "VOLUME_VACUUM_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "VOLUME_VACUUM_ALERTS_REAL.tmp.md"),
    )
    volume_vacuum_state = build_volume_vacuum_state(
        alert_payload=volume_vacuum_alerts,
        source_json=str(reports_path / "VOLUME_VACUUM_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["VOLUME_VACUUM_STATE_REAL_json"]),
        out_md=str(artifacts["VOLUME_VACUUM_STATE_REAL_md"]),
    )
    volume_vacuum_watchlist = build_volume_vacuum_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["VOLUME_VACUUM_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["VOLUME_VACUUM_WATCHLIST_REAL_md"]),
    )

    volatility_burst_alerts = build_volatility_burst_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        out_json=str(reports_path / "VOLATILITY_BURST_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "VOLATILITY_BURST_ALERTS_REAL.tmp.md"),
    )
    volatility_burst_state = build_volatility_burst_state(
        alert_payload=volatility_burst_alerts,
        source_json=str(reports_path / "VOLATILITY_BURST_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["VOLATILITY_BURST_STATE_REAL_json"]),
        out_md=str(artifacts["VOLATILITY_BURST_STATE_REAL_md"]),
    )
    volatility_burst_watchlist = build_volatility_burst_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["VOLATILITY_BURST_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["VOLATILITY_BURST_WATCHLIST_REAL_md"]),
    )

    book_proxy_pressure_alerts = build_book_proxy_pressure_alerts(
        db=micro_db,
        symbol=primary_symbol,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        out_json=str(reports_path / "BOOK_PROXY_PRESSURE_ALERTS_REAL.tmp.json"),
        out_md=str(reports_path / "BOOK_PROXY_PRESSURE_ALERTS_REAL.tmp.md"),
    )
    book_proxy_pressure_state = build_book_proxy_pressure_state(
        alert_payload=book_proxy_pressure_alerts,
        source_json=str(reports_path / "BOOK_PROXY_PRESSURE_ALERTS_REAL.tmp.json"),
        out_json=str(artifacts["BOOK_PROXY_PRESSURE_STATE_REAL_json"]),
        out_md=str(artifacts["BOOK_PROXY_PRESSURE_STATE_REAL_md"]),
    )
    book_proxy_pressure_watchlist = build_book_proxy_pressure_watchlist(
        db=micro_db,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["BOOK_PROXY_PRESSURE_WATCHLIST_REAL_json"]),
        out_md=str(artifacts["BOOK_PROXY_PRESSURE_WATCHLIST_REAL_md"]),
    )

    watchboard = build_watchboard_payload(
        micro_db=micro_db,
        trade_source=trade_source,
        symbols=symbols,
        lookback_min=lookback_min,
        bucket_sec=bucket_sec,
        recent_limit=recent_limit,
        top_n=top_n,
        out_json=str(artifacts["RESEARCH_EVENT_WATCHBOARD_REAL_json"]),
        out_md=str(artifacts["RESEARCH_EVENT_WATCHBOARD_REAL_md"]),
    )

    payloads = {
        "RESEARCH_EVENT_WATCHBOARD_REAL": watchboard,
        "LIQUIDATION_ALERT_STATE_REAL": liquidation_state,
        "LIQUIDATION_WATCHLIST_REAL": liquidation_watchlist,
        "SPREAD_STRESS_STATE_REAL": spread_state,
        "SPREAD_STRESS_WATCHLIST_REAL": spread_watchlist,
        "FILL_TOXICITY_STATE_REAL": fill_state,
        "LATENCY_STRESS_STATE_REAL": latency_state,
        "RETURN_SHOCK_STATE_REAL": return_shock_state,
        "RETURN_SHOCK_WATCHLIST_REAL": return_shock_watchlist,
        "VOLUME_VACUUM_STATE_REAL": volume_vacuum_state,
        "VOLUME_VACUUM_WATCHLIST_REAL": volume_vacuum_watchlist,
        "VOLATILITY_BURST_STATE_REAL": volatility_burst_state,
        "VOLATILITY_BURST_WATCHLIST_REAL": volatility_burst_watchlist,
        "BOOK_PROXY_PRESSURE_STATE_REAL": book_proxy_pressure_state,
        "BOOK_PROXY_PRESSURE_WATCHLIST_REAL": book_proxy_pressure_watchlist,
    }
    for name, artifact_payload in payloads.items():
        _write_json(artifacts[f"{name}_json"], artifact_payload)
        _write_md(artifacts[f"{name}_md"], name, artifact_payload)

    return {
        "summary": {
            "watchboard_top_lane": str((watchboard.get("summary") or {}).get("top_lane") or ""),
            "primary_symbol": primary_symbol,
            "symbol_count": len(symbols),
            "artifact_count": len(payloads),
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "run_summary": build_run_summary(
            run_type="refresh_dashboard_research_events",
            inputs={
                "micro_db": micro_db,
                "trade_source": trade_source,
                "primary_symbol": primary_symbol,
                "symbols": symbols,
                "lookback_min": lookback_min,
                "bucket_sec": bucket_sec,
                "recent_limit": recent_limit,
                "top_n": top_n,
                "reports_dir": str(reports_path),
            },
            metrics={
                "artifact_count": len(payloads),
                "watchboard_top_lane": str((watchboard.get("summary") or {}).get("top_lane") or ""),
            },
            artifacts={name: str(path) for name, path in artifacts.items()},
        ),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh dashboard-facing research event REAL artifacts.")
    p.add_argument("--micro-db", default="data/microstructure.db")
    p.add_argument("--trade-source", default="data/live/papertrades_live.parquet")
    p.add_argument("--primary-symbol", default="ETHUSDT")
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--recent-limit", type=int, default=20)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--out-json", default="reports/REFRESH_DASHBOARD_RESEARCH_EVENTS.json")
    p.add_argument("--out-md", default="reports/REFRESH_DASHBOARD_RESEARCH_EVENTS.md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    payload = build_refresh_payload(
        micro_db=str(args.micro_db),
        trade_source=str(args.trade_source),
        primary_symbol=str(args.primary_symbol).upper(),
        symbols=_parse_symbols(str(args.symbols)),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        recent_limit=int(args.recent_limit),
        top_n=int(args.top_n),
        reports_dir=str(args.reports_dir),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    _write_json(out_json, payload)
    _write_md(out_md, "REFRESH DASHBOARD RESEARCH EVENTS", payload)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
