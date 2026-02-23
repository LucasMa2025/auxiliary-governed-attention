"""
tests/test_observability/test_config.py — ObservabilityConfig 测试
"""
import pytest
from aga_observability.config import ObservabilityConfig, AlertRuleConfig


class TestObservabilityConfig:
    """ObservabilityConfig 测试"""

    def test_default_config(self):
        """默认配置"""
        config = ObservabilityConfig()
        assert config.enabled is True
        assert config.prometheus_enabled is True
        assert config.prometheus_port == 9090
        assert config.log_format == "json"
        assert config.audit_storage_backend == "memory"
        assert config.alert_enabled is True
        assert config.health_enabled is True

    def test_from_dict(self):
        """从字典创建"""
        config = ObservabilityConfig.from_dict({
            "enabled": True,
            "prometheus_port": 9091,
            "log_format": "text",
            "audit_storage_backend": "sqlite",
            "audit_retention_days": 30,
        })
        assert config.prometheus_port == 9091
        assert config.log_format == "text"
        assert config.audit_storage_backend == "sqlite"
        assert config.audit_retention_days == 30

    def test_from_dict_with_alert_rules(self):
        """从字典创建（含告警规则）"""
        config = ObservabilityConfig.from_dict({
            "alert_rules": [
                {
                    "name": "test_rule",
                    "metric": "latency_p99",
                    "operator": ">",
                    "threshold": 1000.0,
                    "severity": "critical",
                },
            ],
        })
        assert len(config.alert_rules) == 1
        assert config.alert_rules[0].name == "test_rule"
        assert config.alert_rules[0].threshold == 1000.0

    def test_from_aga_config(self):
        """从 AGAConfig 映射"""
        # 模拟 AGAConfig
        class MockAGAConfig:
            observability_enabled = True
            prometheus_enabled = True
            prometheus_port = 9091
            log_format = "text"
            log_level = "DEBUG"
            audit_storage_backend = "sqlite"
            audit_retention_days = 60

        config = ObservabilityConfig.from_aga_config(MockAGAConfig())
        assert config.enabled is True
        assert config.prometheus_port == 9091
        assert config.log_format == "text"
        assert config.log_level == "DEBUG"
        assert config.audit_storage_backend == "sqlite"
        assert config.audit_retention_days == 60

    def test_from_aga_config_missing_attrs(self):
        """从 AGAConfig 映射（缺少属性使用默认值）"""
        class MinimalConfig:
            pass

        config = ObservabilityConfig.from_aga_config(MinimalConfig())
        assert config.enabled is True
        assert config.prometheus_port == 9090

    def test_unknown_fields_ignored(self):
        """未知字段被忽略"""
        config = ObservabilityConfig.from_dict({
            "unknown_field": "value",
            "prometheus_port": 9091,
        })
        assert config.prometheus_port == 9091
        assert not hasattr(config, "unknown_field")


class TestAlertRuleConfig:
    """AlertRuleConfig 测试"""

    def test_default(self):
        """默认值"""
        rule = AlertRuleConfig()
        assert rule.name == ""
        assert rule.operator == ">"
        assert rule.severity == "warning"
        assert rule.cooldown_seconds == 300

    def test_custom(self):
        """自定义值"""
        rule = AlertRuleConfig(
            name="test",
            metric="latency_p99",
            threshold=500.0,
            severity="critical",
        )
        assert rule.name == "test"
        assert rule.threshold == 500.0
        assert rule.severity == "critical"
