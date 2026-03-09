"""Standalone Prometheus metrics exporter for Eclipse Scalper.

Guardian-safe: never raises, returns empty string on error.
Generates Prometheus text exposition format (text/plain; version=0.0.4).

Usage:
    from monitoring.prometheus import PrometheusExporter
    exporter = PrometheusExporter()
    body = exporter.collect_metrics(bot)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Guardian-safe imports from execution layer
# ---------------------------------------------------------------------------
try:
    from execution.guardian import get_step_timings
except Exception:  # pragma: no cover

    def get_step_timings() -> Dict[str, Dict[str, float]]:  # type: ignore[misc]
        return {}


try:
    from execution.guardian import is_exchange_degraded
except Exception:  # pragma: no cover

    def is_exchange_degraded(bot: Any) -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HEARTBEAT_PATH = Path("logs/health/heartbeat.json")
_HEARTBEAT_ALIVE_THRESHOLD_SEC = 30.0


# ---------------------------------------------------------------------------
# Helper: Prometheus line formatter
# ---------------------------------------------------------------------------
def _gauge_line(
    name: str,
    value: float,
    help_text: str = "",
    labels: str = "",
    *,
    buf: list[str],
) -> None:
    """Append a Prometheus gauge metric to *buf*. Never raises."""
    try:
        if help_text:
            buf.append(f"# HELP {name} {help_text}")
            buf.append(f"# TYPE {name} gauge")
        lbl = f"{{{labels}}}" if labels else ""
        buf.append(f"{name}{lbl} {value}")
    except Exception:
        pass


def _counter_line(
    name: str,
    value: float,
    help_text: str = "",
    labels: str = "",
    *,
    buf: list[str],
) -> None:
    """Append a Prometheus counter metric to *buf*. Never raises."""
    try:
        if help_text:
            buf.append(f"# HELP {name} {help_text}")
            buf.append(f"# TYPE {name} counter")
        lbl = f"{{{labels}}}" if labels else ""
        buf.append(f"{name}{lbl} {value}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PrometheusExporter
# ---------------------------------------------------------------------------
class PrometheusExporter:
    """Generates Prometheus text-format metrics from bot state.

    All public methods are guardian-safe: they catch every exception
    internally and return a safe default so the caller never crashes.
    """

    def __init__(self, heartbeat_path: Optional[Path] = None) -> None:
        self._hb_path = heartbeat_path or _HEARTBEAT_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_metrics(self, bot: Any = None) -> str:
        """Return the full Prometheus text exposition body.

        Parameters
        ----------
        bot : object, optional
            The live bot instance (used for ``is_exchange_degraded``).
            May be ``None`` when running without a bot (e.g. dashboard-only).

        Returns
        -------
        str
            Prometheus text-format string, or ``""`` on error.
        """
        try:
            lines: list[str] = []
            now = time.time()

            _gauge_line("eclipse_up", 1, "Whether the exporter is up", buf=lines)
            _gauge_line("eclipse_scrape_ts", now, "Timestamp of this scrape", buf=lines)

            self._collect_heartbeat(lines, now)
            self._collect_exchange_degraded(lines, bot)
            self._collect_step_timings(lines)
            self._collect_counters_from_heartbeat(lines)

            return "\n".join(lines) + "\n" if lines else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Internal collectors — each is independently guardian-safe
    # ------------------------------------------------------------------

    def _read_heartbeat(self) -> Optional[dict]:
        """Read and parse the heartbeat JSON. Returns None on any error."""
        try:
            if not self._hb_path.exists():
                return None
            return json.loads(self._hb_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

    def _collect_heartbeat(self, lines: list[str], now: float) -> None:
        """Emit gauges derived from the guardian heartbeat file."""
        try:
            hb = self._read_heartbeat()
            if hb is None:
                _gauge_line("eclipse_bot_alive", 0, "Bot process alive (heartbeat fresh)", buf=lines)
                return

            hb_ts = float(hb.get("ts", 0) or 0)
            hb_age = max(0.0, now - hb_ts)
            alive = 1 if hb_age < _HEARTBEAT_ALIVE_THRESHOLD_SEC else 0

            _gauge_line("eclipse_bot_heartbeat_age_sec", hb_age, "Seconds since last guardian heartbeat", buf=lines)
            _gauge_line("eclipse_bot_alive", alive, "Bot process alive (heartbeat fresh)", buf=lines)
            _gauge_line("eclipse_uptime_seconds", float(hb.get("uptime_sec", 0) or 0), "Bot uptime in seconds", buf=lines)
            _gauge_line("eclipse_open_positions", float(hb.get("open_positions", 0) or 0), "Number of open positions", buf=lines)
            _gauge_line("eclipse_daily_pnl", float(hb.get("daily_pnl", 0) or 0), "Daily PnL in USDT", buf=lines)
            _gauge_line("eclipse_margin_ratio", float(hb.get("margin_ratio", 0) or 0), "Current margin ratio", buf=lines)
            _gauge_line("eclipse_kill_switch_active", 1 if hb.get("kill_switch_active") else 0, "Kill switch active", buf=lines)
        except Exception:
            pass

    def _collect_exchange_degraded(self, lines: list[str], bot: Any) -> None:
        """Emit the exchange-degraded gauge."""
        try:
            degraded = 0
            if bot is not None:
                degraded = 1 if is_exchange_degraded(bot) else 0
            else:
                # Fall back to heartbeat data
                hb = self._read_heartbeat()
                if hb is not None:
                    degraded = 1 if hb.get("exchange_degraded") else 0
            _gauge_line("eclipse_exchange_degraded", degraded, "Exchange in degraded mode", buf=lines)
        except Exception:
            pass

    def _collect_step_timings(self, lines: list[str]) -> None:
        """Emit per-step guardian profiling gauges."""
        try:
            timings = get_step_timings()
            if not timings:
                return
            for step_name, metrics in timings.items():
                if not isinstance(metrics, dict):
                    continue
                safe_name = step_name.replace(".", "_").replace("-", "_")
                lbl = f'step="{safe_name}"'
                _gauge_line(
                    "eclipse_guardian_step_avg_ms",
                    float(metrics.get("avg_ms", 0)),
                    "Guardian step avg duration ms",
                    lbl,
                    buf=lines,
                )
                _gauge_line(
                    "eclipse_guardian_step_max_ms",
                    float(metrics.get("max_ms", 0)),
                    "Guardian step max duration ms",
                    lbl,
                    buf=lines,
                )
        except Exception:
            pass

    def _collect_counters_from_heartbeat(self, lines: list[str]) -> None:
        """Emit counter metrics from heartbeat cumulative values."""
        try:
            hb = self._read_heartbeat()
            if hb is None:
                return
            orders_total = float(hb.get("orders_total", 0) or 0)
            errors_total = float(hb.get("errors_total", 0) or 0)
            if orders_total > 0:
                _counter_line("eclipse_orders_total", orders_total, "Total orders placed", buf=lines)
            if errors_total > 0:
                _counter_line("eclipse_errors_total", errors_total, "Total errors encountered", buf=lines)
        except Exception:
            pass
