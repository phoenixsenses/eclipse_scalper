"""Tests for structured alerting rules engine."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

from monitoring.alert_rules import AlertRuleEngine, AlertResult, _DEFAULT_RULES


class TestAlertRuleEngine:
    def test_default_rules_loaded(self):
        engine = AlertRuleEngine()
        assert len(engine._rules) == len(_DEFAULT_RULES)

    def test_no_metrics_no_alerts(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({})
        assert results == []

    def test_margin_warning_fires(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({"margin_ratio": 0.08})
        ids = [r.rule_id for r in results]
        assert "margin_warning" in ids

    def test_margin_critical_fires(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({"margin_ratio": 0.03})
        ids = [r.rule_id for r in results]
        assert "margin_critical" in ids
        # Warning should also fire since 0.03 < 0.10
        assert "margin_warning" in ids

    def test_margin_ok_no_alert(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({"margin_ratio": 0.50})
        ids = [r.rule_id for r in results]
        assert "margin_warning" not in ids
        assert "margin_critical" not in ids

    def test_daily_loss_warning(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({"daily_pnl_pct": -0.06})
        ids = [r.rule_id for r in results]
        assert "daily_loss_warning" in ids

    def test_position_count_alert(self):
        engine = AlertRuleEngine()
        results = engine.evaluate({"open_position_count": 8.0})
        ids = [r.rule_id for r in results]
        assert "position_count_high" in ids

    def test_cooldown_prevents_refiring(self):
        engine = AlertRuleEngine()
        results1 = engine.evaluate({"margin_ratio": 0.08})
        assert len(results1) > 0
        # Same evaluation immediately should be blocked by cooldown
        results2 = engine.evaluate({"margin_ratio": 0.08})
        margin_results = [r for r in results2 if r.rule_id == "margin_warning"]
        assert len(margin_results) == 0

    def test_cooldown_expires(self):
        engine = AlertRuleEngine()
        engine.evaluate({"margin_ratio": 0.08})
        # Manually expire cooldown
        engine._last_fired["margin_warning"] = time.time() - 999
        results = engine.evaluate({"margin_ratio": 0.08})
        ids = [r.rule_id for r in results]
        assert "margin_warning" in ids

    def test_custom_rules(self):
        custom = [{
            "id": "custom_test",
            "metric": "latency_ms",
            "op": ">",
            "threshold": 500,
            "severity": "WARNING",
            "message": "Latency {value}ms > {threshold}ms",
            "cooldown_sec": 10,
        }]
        engine = AlertRuleEngine(rules=custom)
        results = engine.evaluate({"latency_ms": 800.0})
        assert len(results) == 1
        assert results[0].rule_id == "custom_test"
        assert "800" in results[0].message

    def test_all_operators(self):
        for op, val, threshold, should_fire in [
            ("<", 5, 10, True),
            ("<", 15, 10, False),
            ("<=", 10, 10, True),
            (">", 15, 10, True),
            (">", 5, 10, False),
            (">=", 10, 10, True),
            ("==", 10, 10, True),
            ("!=", 10, 10, False),
            ("!=", 5, 10, True),
        ]:
            rule = [{
                "id": f"test_{op}_{val}",
                "metric": "x",
                "op": op,
                "threshold": threshold,
                "severity": "INFO",
                "message": "test",
                "cooldown_sec": 0,
            }]
            engine = AlertRuleEngine(rules=rule)
            results = engine.evaluate({"x": float(val)})
            assert (len(results) > 0) == should_fire, f"Failed: {val} {op} {threshold}"

    def test_reload_from_file(self, tmp_path):
        import monitoring.alert_rules as ar
        rules_file = tmp_path / "rules.json"
        custom = [{
            "id": "file_rule",
            "metric": "test_val",
            "op": ">",
            "threshold": 0,
            "severity": "INFO",
            "message": "fired",
            "cooldown_sec": 0,
        }]
        rules_file.write_text(json.dumps(custom), encoding="utf-8")

        engine = AlertRuleEngine()
        with patch.object(ar, "_RULES_PATH", rules_file):
            engine.reload_rules()
        assert len(engine._rules) == 1
        assert engine._rules[0]["id"] == "file_rule"

    def test_last_fired_capped(self):
        rules = [{
            "id": f"rule_{i}",
            "metric": f"m_{i}",
            "op": ">",
            "threshold": 0,
            "severity": "INFO",
            "message": "test",
            "cooldown_sec": 0,
        } for i in range(600)]
        engine = AlertRuleEngine(rules=rules)
        metrics = {f"m_{i}": 1.0 for i in range(600)}
        engine.evaluate(metrics)
        assert len(engine._last_fired) <= 501


class TestCollectMetrics:
    def test_collects_margin_ratio(self):
        engine = AlertRuleEngine()
        bot = MagicMock()
        bot.state.margin_ratio = 0.15
        bot.state.positions = {}
        bot.state.run_context = {}
        bot.STARTUP_TIMESTAMP = time.time() - 100
        metrics = engine.collect_metrics(bot)
        assert metrics["margin_ratio"] == 0.15

    def test_collects_position_count(self):
        engine = AlertRuleEngine()
        bot = MagicMock()
        bot.state.margin_ratio = None
        bot.state.positions = {
            "BTCUSDT": {"size": 0.01},
            "ETHUSDT": {"size": 0},
            "SOLUSDT": {"qty": 5.0},
        }
        bot.state.run_context = {}
        bot.STARTUP_TIMESTAMP = None
        metrics = engine.collect_metrics(bot)
        assert metrics["open_position_count"] == 2.0

    def test_survives_missing_state(self):
        engine = AlertRuleEngine()
        bot = MagicMock()
        bot.state = None
        metrics = engine.collect_metrics(bot)
        assert metrics == {}
