from __future__ import annotations

from typing import Any, Dict

from notifications.events import NotificationEvent, NotificationSeverity


def _f(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "0.00"


def build_entry_event(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    regime: str,
    regime_age_sec: float,
    rolling_return: float,
    pocket: Dict[str, Any] | None = None,
    risk_state: Dict[str, Any] | None = None,
) -> NotificationEvent:
    p = dict(pocket or {})
    r = dict(risk_state or {})
    body = (
        f"ENTRY: {str(side).upper()} {symbol} @ ${_f(entry_price, 4)}\n"
        f"Regime: {str(regime).upper()} ({int(float(regime_age_sec or 0.0))}s) | Return: {_f(rolling_return * 100.0, 3)}%\n"
        f"Pocket: imb{_f(p.get('min_imbalance', 0.0), 2)} int{int(float(p.get('min_trade_intensity', 0.0) or 0.0))} spr{_f(p.get('max_spread', 0.0), 6)}\n"
        f"Risk: {int(r.get('open_positions', 0))}/{int((r.get('config') or {}).get('max_concurrent_positions', 1))} pos | "
        f"Daily: {_f(r.get('daily_pnl_bps', 0.0), 1)}/{_f((r.get('config') or {}).get('max_daily_loss_bps', 0.0), 1)} bps"
    )
    return NotificationEvent(
        severity=NotificationSeverity.SUCCESS,
        category="trade_entry",
        title="ENTRY",
        body=body,
    )


def build_exit_event(
    *,
    symbol: str,
    side: str,
    pnl_bps: float,
    exit_type: str,
    hold_sec: float,
    peak_bps: float,
    adverse_bps: float,
    session_trades: int = 0,
    session_pnl_bps: float = 0.0,
    session_win_rate: float = 0.0,
) -> NotificationEvent:
    body = (
        f"EXIT: {str(side).upper()} {symbol} {_f(pnl_bps, 2)} bps ({exit_type})\n"
        f"Hold: {int(float(hold_sec or 0.0))}s | Peak: {_f(peak_bps, 2)} bps | Adverse: {_f(adverse_bps, 2)} bps\n"
        f"Session: {int(session_trades)} trades | {_f(session_pnl_bps, 2)} bps | {_f(session_win_rate * 100.0, 1)}% win"
    )
    sev = NotificationSeverity.SUCCESS if float(pnl_bps or 0.0) >= 0 else NotificationSeverity.WARNING
    return NotificationEvent(severity=sev, category="trade_exit", title="EXIT", body=body)


def build_scratch_event(
    *,
    symbol: str,
    side: str,
    pnl_bps: float,
    adverse_limit_bps: float,
    elapsed_sec: float,
    session_trades: int = 0,
    session_pnl_bps: float = 0.0,
    scratches: int = 0,
) -> NotificationEvent:
    body = (
        f"SCRATCH: {str(side).upper()} {symbol} {_f(pnl_bps, 2)} bps\n"
        f"Adverse hit {_f(adverse_limit_bps, 2)} bps limit @ {int(float(elapsed_sec or 0.0))}s\n"
        f"Session: {int(session_trades)} trades | {_f(session_pnl_bps, 2)} bps | {int(scratches)} scratches"
    )
    return NotificationEvent(
        severity=NotificationSeverity.WARNING,
        category="trade_scratch",
        title="SCRATCH",
        body=body,
    )

