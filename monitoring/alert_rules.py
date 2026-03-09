# monitoring/alert_rules.py — Structured alerting rules engine
# Guardian-safe: never raises, never fatal

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_RULES_PATH = Path(os.environ.get("ALERT_RULES_PATH", "config/alert_rules.json"))

_DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "id": "margin_warning",
        "metric": "margin_ratio",
        "op": "<",
        "threshold": 0.10,
        "severity": "WARNING",
        "message": "Margin ratio {value:.4f} below {threshold}",
        "cooldown_sec": 300,
    },
    {
        "id": "margin_critical",
        "metric": "margin_ratio",
        "op": "<",
        "threshold": 0.05,
        "severity": "CRITICAL",
        "message": "MARGIN CRITICAL: {value:.4f} — liquidation risk!",
        "cooldown_sec": 60,
    },
    {
        "id": "daily_loss_warning",
        "metric": "daily_pnl_pct",
        "op": "<",
        "threshold": -0.05,
        "severity": "WARNING",
        "message": "Daily PnL {value:.2%} exceeds loss threshold {threshold}",
        "cooldown_sec": 600,
    },
    {
        "id": "daily_loss_critical",
        "metric": "daily_pnl_pct",
        "op": "<",
        "threshold": -0.10,
        "severity": "CRITICAL",
        "message": "Daily PnL CRITICAL: {value:.2%}",
        "cooldown_sec": 120,
    },
    {
        "id": "position_count_high",
        "metric": "open_position_count",
        "op": ">",
        "threshold": 5,
        "severity": "WARNING",
        "message": "{value} open positions (max recommended: {threshold})",
        "cooldown_sec": 600,
    },
    {
        "id": "uptime_heartbeat",
        "metric": "uptime_sec",
        "op": ">",
        "threshold": 86400,
        "severity": "INFO",
        "message": "Bot uptime: {value:.0f}s ({hours:.1f}h)",
        "cooldown_sec": 86400,
    },
]


@dataclass
class AlertResult:
    rule_id: str
    severity: str
    message: str
    metric: str
    value: float
    threshold: float


_OPS = {
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
    "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t,
}


class AlertRuleEngine:
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self._rules = rules if rules is not None else list(_DEFAULT_RULES)
        self._last_fired: Dict[str, float] = {}  # rule_id -> last_fire_ts
        self._rules_mtime: float = 0.0

    def reload_rules(self) -> None:
        """Reload rules from file if changed. Falls back to defaults on error."""
        try:
            if not _RULES_PATH.exists():
                return
            mtime = _RULES_PATH.stat().st_mtime
            if mtime <= self._rules_mtime:
                return
            self._rules_mtime = mtime
            raw = _RULES_PATH.read_text(encoding="utf-8").strip()
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                self._rules = loaded
        except Exception:
            pass

    def evaluate(self, metrics: Dict[str, float]) -> List[AlertResult]:
        """Evaluate all rules against provided metrics. Returns triggered alerts."""
        self.reload_rules()
        now = time.time()
        results: List[AlertResult] = []

        for rule in self._rules:
            try:
                rule_id = str(rule.get("id", ""))
                metric_name = str(rule.get("metric", ""))
                op_str = str(rule.get("op", ""))
                threshold = float(rule.get("threshold", 0))
                severity = str(rule.get("severity", "INFO"))
                msg_template = str(rule.get("message", ""))
                cooldown = float(rule.get("cooldown_sec", 60))

                if metric_name not in metrics:
                    continue

                value = float(metrics[metric_name])
                op_fn = _OPS.get(op_str)
                if op_fn is None:
                    continue

                if not op_fn(value, threshold):
                    continue

                # Cooldown check
                last = self._last_fired.get(rule_id, 0.0)
                if (now - last) < cooldown:
                    continue

                self._last_fired[rule_id] = now

                # Cap tracked rules
                if len(self._last_fired) > 500:
                    oldest_k = min(self._last_fired, key=self._last_fired.get)  # type: ignore[arg-type]
                    self._last_fired.pop(oldest_k, None)

                # Format message
                try:
                    msg = msg_template.format(
                        value=value,
                        threshold=threshold,
                        hours=value / 3600.0 if value > 0 else 0,
                    )
                except Exception:
                    msg = f"{metric_name}: {value} {op_str} {threshold}"

                results.append(AlertResult(
                    rule_id=rule_id,
                    severity=severity,
                    message=msg,
                    metric=metric_name,
                    value=value,
                    threshold=threshold,
                ))
            except Exception:
                continue

        return results

    def collect_metrics(self, bot: Any) -> Dict[str, float]:
        """Collect current metrics from bot state for rule evaluation."""
        metrics: Dict[str, float] = {}
        try:
            st = getattr(bot, "state", None)
            if st is None:
                return metrics

            # Margin ratio
            mr = getattr(st, "margin_ratio", None)
            if mr is not None:
                metrics["margin_ratio"] = float(mr)

            # Open positions
            positions = getattr(st, "positions", None)
            if isinstance(positions, dict):
                count = sum(
                    1 for v in positions.values()
                    if isinstance(v, dict) and abs(float(v.get("size") or v.get("qty") or 0)) > 0
                )
                metrics["open_position_count"] = float(count)

            # Uptime
            startup_ts = getattr(bot, "STARTUP_TIMESTAMP", None)
            if startup_ts is not None:
                metrics["uptime_sec"] = time.time() - float(startup_ts)

            # Daily PnL (from run_context if available)
            rc = getattr(st, "run_context", None)
            if isinstance(rc, dict):
                daily_pnl = rc.get("daily_pnl_pct")
                if daily_pnl is not None:
                    metrics["daily_pnl_pct"] = float(daily_pnl)

                # Exchange degraded
                if rc.get("exchange_degraded"):
                    metrics["exchange_degraded"] = 1.0

        except Exception:
            pass
        return metrics


# Singleton instance
_engine: Optional[AlertRuleEngine] = None


def get_engine() -> AlertRuleEngine:
    global _engine
    if _engine is None:
        _engine = AlertRuleEngine()
    return _engine
