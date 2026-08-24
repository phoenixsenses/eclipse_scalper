from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from notifications.events import NotificationEvent, NotificationSeverity


def build_entry_blocked_event(reason: str, risk_state: Dict[str, Any] | None = None) -> NotificationEvent:
    st = dict(risk_state or {})
    cfg = dict(st.get("config") or {})
    body = (
        f"ENTRY BLOCKED: {reason}\n"
        f"Loss: {float(st.get('daily_pnl_bps', 0.0)):.1f}/{float(cfg.get('max_daily_loss_bps', 0.0)):.1f} bps | "
        f"Trades: {int(st.get('daily_trades', 0))}/{int(cfg.get('max_daily_trades', 0))}\n"
        "Resumes: next daily reset (00:00 UTC)"
    )
    return NotificationEvent(
        severity=NotificationSeverity.WARNING,
        category="risk_block",
        title="ENTRY BLOCKED",
        body=body,
        raw_payload={"reason": str(reason)},
        throttle_key=f"risk_block:{reason}",
        throttle_window_sec=600.0,
    )


def build_regime_change_event(old_regime: str, new_regime: str, cooldown_sec: float, open_positions: int) -> NotificationEvent:
    resume = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + float(cooldown_sec or 0.0), tz=timezone.utc)
    body = (
        f"REGIME CHANGE: {str(old_regime).upper()} -> {str(new_regime).upper()}\n"
        f"Cooldown: {int(float(cooldown_sec or 0.0))}s (no entries until {resume.strftime('%H:%M:%S')} UTC)\n"
        f"Open positions: {int(open_positions)}"
    )
    return NotificationEvent(
        NotificationSeverity.INFO,
        "regime_change",
        "REGIME CHANGE",
        body,
        raw_payload={"old_regime": str(old_regime), "new_regime": str(new_regime)},
    )


def build_scratch_pause_event(consecutive: int, pause_sec: float, last_trades_bps: Iterable[float]) -> NotificationEvent:
    last = ", ".join(f"{float(x):+.1f}" for x in list(last_trades_bps)[-3:])
    body = (
        f"SCRATCH CIRCUIT BREAKER: {int(consecutive)} consecutive scratches\n"
        f"Trading paused for {int(float(pause_sec or 0.0))}s\n"
        f"Last 3 trades: {last or 'n/a'} bps"
    )
    return NotificationEvent(
        NotificationSeverity.CRITICAL,
        "scratch_pause",
        "SCRATCH CIRCUIT BREAKER",
        body,
        raw_payload={"consecutive": int(consecutive)},
    )


def build_drawdown_event(max_drawdown_bps: float, peak_pnl_bps: float, current_pnl_bps: float) -> NotificationEvent:
    body = (
        f"DRAWDOWN LIMIT HIT: -{float(max_drawdown_bps):.1f} bps\n"
        "Trading halted until manual reset\n"
        f"Peak: {float(peak_pnl_bps):+.1f} bps | Current: {float(current_pnl_bps):+.1f} bps"
    )
    return NotificationEvent(
        NotificationSeverity.CRITICAL,
        "drawdown_limit",
        "DRAWDOWN LIMIT HIT",
        body,
        raw_payload={
            "max_drawdown_bps": float(max_drawdown_bps),
            "peak_pnl_bps": float(peak_pnl_bps),
            "current_pnl_bps": float(current_pnl_bps),
        },
        throttle_key="drawdown_limit",
        throttle_window_sec=3600.0,
    )

