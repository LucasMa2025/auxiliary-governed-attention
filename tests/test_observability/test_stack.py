"""
tests/test_observability/test_stack.py — ObservabilityStack 和 setup_observability 测试
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from aga.instrumentation import EventBus
from aga_observability.config import ObservabilityConfig
from aga_observability.stack import ObservabilityStack
from aga_observability import integration as integration_module
from aga_observability.integration import (
    setup_observability,
    get_global_stack,
    bind_plugin,
    shutdown_observability,
)


class TestObservabilityStack:
    """ObservabilityStack 测试"""

    def test_create_default(self):
        """默认创建"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,  # 避免端口冲突
            health_enabled=False,
            log_enabled=False,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        assert stack.alert_manager is not None
        assert stack.dashboard_generator is not None
        stack.shutdown()

    def test_create_with_log(self):
        """创建含日志导出"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=True,
            log_format="json",
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        assert stack.log_exporter is not None
        stack.shutdown()

    def test_create_with_file_audit(self, tmp_path):
        """创建含文件审计"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
            audit_storage_backend="file",
            audit_storage_path=str(tmp_path / "audit.jsonl"),
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        assert stack.audit_storage is not None
        stack.shutdown()

    def test_create_with_sqlite_audit(self, tmp_path):
        """创建含 SQLite 审计"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
            audit_storage_backend="sqlite",
            audit_storage_path=str(tmp_path / "audit.db"),
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        assert stack.audit_storage is not None
        stack.shutdown()

    def test_start_and_shutdown(self):
        """启动和关闭"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        stack.start()
        assert stack._running is True
        stack.shutdown()
        assert stack._running is False

    def test_event_flow(self):
        """事件流测试"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
            alert_enabled=True,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        stack.start()

        # 发射事件
        bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.5,
            "layer_idx": 0,
            "latency_us": 45.0,
        })

        # AlertManager 应该收到事件
        assert len(stack.alert_manager._metrics_window) > 0

        stack.shutdown()

    def test_bind_plugin(self):
        """绑定 AGAPlugin"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=True,
            log_enabled=False,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)

        # 模拟 plugin
        mock_plugin = MagicMock()
        mock_plugin.config.hidden_dim = 4096
        mock_plugin.config.bottleneck_dim = 64
        mock_plugin.config.max_slots = 256
        mock_plugin.device = "cuda"

        stack.bind_plugin(mock_plugin)
        assert stack._plugin is mock_plugin
        stack.shutdown()

    def test_get_stats(self):
        """获取统计"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)
        stats = stack.get_stats()
        assert "running" in stats
        stack.shutdown()

    def test_generate_dashboard(self, tmp_path):
        """生成 Dashboard"""
        bus = EventBus()
        config = ObservabilityConfig(
            prometheus_enabled=False,
            health_enabled=False,
            log_enabled=False,
        )
        stack = ObservabilityStack(event_bus=bus, config=config)

        # 生成 JSON
        json_str = stack.generate_dashboard()
        assert "dashboard" in json_str

        # 保存到文件
        path = str(tmp_path / "dashboard.json")
        stack.generate_dashboard(path=path)

        import os
        assert os.path.exists(path)
        stack.shutdown()


class TestSetupObservability:
    """setup_observability 测试"""

    def setup_method(self):
        """每个测试前重置全局状态"""
        integration_module._global_stack = None

    def teardown_method(self):
        """每个测试后清理"""
        shutdown_observability()

    def test_setup_creates_stack(self):
        """setup 创建 stack"""
        bus = EventBus()

        class MockConfig:
            observability_enabled = True
            prometheus_enabled = False
            prometheus_port = 9090
            log_format = "json"
            log_level = "INFO"
            audit_storage_backend = "memory"
            audit_retention_days = 90

        stack = setup_observability(bus, MockConfig())
        assert stack is not None
        assert get_global_stack() is stack

    def test_setup_singleton(self):
        """setup 单例"""
        bus = EventBus()

        class MockConfig:
            observability_enabled = True
            prometheus_enabled = False
            prometheus_port = 9090
            log_format = "json"
            log_level = "INFO"
            audit_storage_backend = "memory"
            audit_retention_days = 90

        stack1 = setup_observability(bus, MockConfig())
        stack2 = setup_observability(bus, MockConfig())
        assert stack1 is stack2

    def test_setup_disabled(self):
        """禁用时不创建"""
        bus = EventBus()

        class MockConfig:
            observability_enabled = False
            prometheus_enabled = False
            prometheus_port = 9090
            log_format = "json"
            log_level = "INFO"
            audit_storage_backend = "memory"
            audit_retention_days = 90

        stack = setup_observability(bus, MockConfig())
        assert stack is None

    def test_setup_no_config(self):
        """无配置使用默认"""
        bus = EventBus()
        # 不传 prometheus_client 时可能失败，但应该 Fail-Open
        stack = setup_observability(bus)
        # 可能成功也可能 None（取决于 prometheus_client 是否安装）
        # 关键是不抛异常

    def test_bind_plugin_global(self):
        """全局绑定 plugin"""
        bus = EventBus()

        class MockConfig:
            observability_enabled = True
            prometheus_enabled = False
            prometheus_port = 9090
            log_format = "json"
            log_level = "INFO"
            audit_storage_backend = "memory"
            audit_retention_days = 90

        setup_observability(bus, MockConfig())

        mock_plugin = MagicMock()
        bind_plugin(mock_plugin)
        # 不抛异常即可

    def test_shutdown(self):
        """关闭全局 stack"""
        bus = EventBus()

        class MockConfig:
            observability_enabled = True
            prometheus_enabled = False
            prometheus_port = 9090
            log_format = "json"
            log_level = "INFO"
            audit_storage_backend = "memory"
            audit_retention_days = 90

        setup_observability(bus, MockConfig())
        assert get_global_stack() is not None

        shutdown_observability()
        assert get_global_stack() is None
