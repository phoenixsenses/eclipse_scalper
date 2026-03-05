from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import LiveSettings


def evaluate_alerts(status: Dict[str, Any], cfg: LiveSettings) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(code: str, detail: str, severity: str = "warn") -> None:
        out.append(
            {
                "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "code": code,
                "severity": severity,
                "detail": detail,
            }
        )

    if float(status.get("data_freshness_sec", 0.0) or 0.0) > float(cfg.freshness_warn_sec):
        add("data_stale", f"freshness={float(status.get('data_freshness_sec', 0.0)):.1f}s")
    if float(status.get("missing_bars_pct_1h", 0.0) or 0.0) > float(cfg.missing_bars_warn_pct):
        add("missing_bars", f"missing_pct={float(status.get('missing_bars_pct_1h', 0.0)):.2f}")
    if float(abs(status.get("spread_jump_frac", 0.0) or 0.0)) > float(cfg.spread_jump_warn_frac):
        add("spread_jump", f"spread_jump_frac={float(status.get('spread_jump_frac', 0.0)):.3f}")
    if float(status.get("ofi_shift", 0.0) or 0.0) > float(cfg.ofi_shift_warn):
        add("ofi_shift", f"ofi_shift={float(status.get('ofi_shift', 0.0)):.3f}")
    if float(status.get("regime_shift", 0.0) or 0.0) > float(cfg.regime_shift_warn):
        add("regime_shift", f"regime_shift={float(status.get('regime_shift', 0.0)):.3f}")
    rate = float(status.get("signal_rate_per_hour", 0.0) or 0.0)
    if rate < float(cfg.signal_rate_low_warn):
        add("signal_rate_low", f"signal_rate_per_hour={rate:.4f}")
    if rate > float(cfg.signal_rate_high_warn):
        add("signal_rate_high", f"signal_rate_per_hour={rate:.2f}")
    # Sprint 2 drift alerts
    if abs(float(status.get("replay_fill_rate_delta", 0.0) or 0.0)) > float(cfg.replay_fill_rate_delta_warn):
        add("replay_fill_rate_drift", f"delta={float(status.get('replay_fill_rate_delta', 0.0)):+.4f}")
    if abs(float(status.get("replay_adverse_bps_delta", 0.0) or 0.0)) > float(cfg.replay_adverse_bps_delta_warn):
        add("replay_adverse_drift", f"delta_bps={float(status.get('replay_adverse_bps_delta', 0.0)):+.4f}")
    if float(status.get("replay_match_rate_vs_sim", 1.0) or 1.0) < float(cfg.replay_match_rate_low_warn):
        add("replay_match_rate_low", f"match_rate={float(status.get('replay_match_rate_vs_sim', 0.0)):.2%}")
    if float(status.get("diag_toxicity_score", 0.0) or 0.0) > float(cfg.diagnostics_toxicity_warn):
        add("diagnostics_toxicity_high", f"toxicity={float(status.get('diag_toxicity_score', 0.0)):.4f}")
    if float(status.get("diag_latency_fill_delay_sec_p95", 0.0) or 0.0) > float(cfg.diagnostics_latency_p95_warn_sec):
        add("diagnostics_latency_high", f"p95={float(status.get('diag_latency_fill_delay_sec_p95', 0.0)):.3f}s")
    return out


def append_alerts(path: Path, alerts: List[Dict[str, Any]]) -> None:
    if not alerts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, ensure_ascii=True, sort_keys=True) + "\n")
