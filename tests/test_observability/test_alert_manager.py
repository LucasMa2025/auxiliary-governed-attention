"""
tests/test_observability/test_alert_manager.py — AlertManager 测试
"""
import time
import pytest
from aga.instrumentation import EventBus, Event
from aga_observability.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertEvent,
)


class TestAlertRule:
    """AlertRule 测试"""

    def test_evaluate_greater(self):
        rule = AlertRule(name="test", metric="m", operator=">", threshold=0.5)
        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.4) is False

    def test_evaluate_less(self):
        rule = AlertRule(name="test", metric="m", operator="<", threshold=0.5)
        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.6) is False

    def test_evaluate_gte(self):
        rule = AlertRule(name="test", metric="m", operator=">=", threshold=0.5)
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.4) is False

    def test_evaluate_lte(self):
        rule = AlertRule(name="test", metric="m", operator="<=", threshold=0.5)
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.6) is False

    def test_evaluate_eq(self):
        rule = AlertRule(name="test", metric="m", operator="==", threshold=0.5)
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.50000000001) is True  # 浮点容差
        assert rule.evaluate(0.6) is False

    def test_evaluate_unknown_operator(self):
        rule = AlertRule(name="test", metric="m", operator="!=", threshold=0.5)
        assert rule.evaluate(0.6) is False  # 未知运算符返回 False

    def test_format_message_custom(self):
        rule = AlertRule(
            name="test",
            metric="latency",
            threshold=100.0,
            message="延迟 {value:.1f}μs 超过 {threshold:.0f}μs",
        )
        msg = rule.format_message(150.0)
        assert "150.0" in msg
        assert "100" in msg

    def test_format_message_default(self):
        rule = AlertRule(
            name="test",
            metric="latency",
            operator=">",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
        )
        msg = rule.format_message(150.0)
        assert "WARNING" in msg
        assert "test" in msg


class TestAlertManager:
    """AlertManager 测试"""

    def test_default_rules(self):
        """默认规则加载"""
        manager = AlertManager(use_defaults=True)
        assert len(manager._rules) == len(AlertManager.DEFAULT_RULES)

    def test_no_default_rules(self):
        """不加载默认规则"""
        manager = AlertManager(use_defaults=False)
        assert len(manager._rules) == 0

    def test_add_remove_rule(self):
        """添加和移除规则"""
        manager = AlertManager(use_defaults=False)
        rule = AlertRule(name="test", metric="m", threshold=0.5)
        manager.add_rule(rule)
        assert "test" in manager._rules
        assert manager.remove_rule("test") is True
        assert "test" not in manager._rules
        assert manager.remove_rule("nonexistent") is False

    def test_subscribe_and_forward_event(self):
        """订阅并处理 forward 事件"""
        bus = EventBus()
        manager = AlertManager(use_defaults=False)

        # 添加一个低阈值规则，确保触发
        manager.add_rule(AlertRule(
            name="test_latency",
            metric="latency_p99",
            operator=">",
            threshold=0.0,
            window_seconds=60,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0,  # 无冷却
        ))

        manager.subscribe(bus)

        # 发射事件
        bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.5,
            "layer_idx": 0,
            "latency_us": 100.0,
        })

        # 检查是否有告警
        alerts = manager.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["rule_name"] == "test_latency"

    def test_cooldown(self):
        """冷却期测试"""
        bus = EventBus()
        manager = AlertManager(use_defaults=False)

        manager.add_rule(AlertRule(
            name="test",
            metric="latency_p99",
            operator=">",
            threshold=0.0,
            cooldown_seconds=9999,  # 很长的冷却期
        ))

        manager.subscribe(bus)

        # 第一次触发
        bus.emit("forward", {"latency_us": 100.0, "aga_applied": True, "gate_mean": 0.5})
        alerts1 = manager.get_alerts()

        # 第二次不应触发（冷却期内）
        bus.emit("forward", {"latency_us": 200.0, "aga_applied": True, "gate_mean": 0.5})
        alerts2 = manager.get_alerts()

        assert len(alerts2) == len(alerts1)  # 数量不变

    def test_callback(self):
        """回调通知测试"""
        bus = EventBus()
        manager = AlertManager(use_defaults=False)
        received = []

        manager.add_rule(AlertRule(
            name="test",
            metric="latency_p99",
            operator=">",
            threshold=0.0,
            cooldown_seconds=0,
        ))
        manager.add_callback(lambda alert: received.append(alert))
        manager.subscribe(bus)

        bus.emit("forward", {"latency_us": 100.0, "aga_applied": True, "gate_mean": 0.5})

        assert len(received) >= 1
        assert isinstance(received[0], AlertEvent)

    def test_get_stats(self):
        """统计信息"""
        manager = AlertManager(use_defaults=True)
        stats = manager.get_stats()
        assert "rules_count" in stats
        assert "total_alerts" in stats
        assert "by_severity" in stats

    def test_severity_filter(self):
        """按级别过滤告警"""
        bus = EventBus()
        manager = AlertManager(use_defaults=False)

        manager.add_rule(AlertRule(
            name="warn",
            metric="latency_p99",
            operator=">",
            threshold=0.0,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0,
        ))
        manager.add_rule(AlertRule(
            name="crit",
            metric="gate_mean",
            operator=">",
            threshold=0.0,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=0,
        ))

        manager.subscribe(bus)
        bus.emit("forward", {
            "latency_us": 100.0,
            "gate_mean": 0.5,
            "aga_applied": True,
        })

        warnings = manager.get_alerts(severity=AlertSeverity.WARNING)
        criticals = manager.get_alerts(severity=AlertSeverity.CRITICAL)

        assert all(a["severity"] == "warning" for a in warnings)
        assert all(a["severity"] == "critical" for a in criticals)
