"""
tests/test_observability/test_health.py — HealthChecker 测试
"""
import pytest
from unittest.mock import MagicMock, PropertyMock
from aga_observability.health import HealthChecker, HealthStatus


class TestHealthChecker:
    """HealthChecker 测试"""

    def _make_mock_plugin(
        self,
        count=10,
        max_slots=256,
        utilization=0.04,
        pinned=2,
        attached=True,
        hooks=3,
    ):
        """创建模拟 AGAPlugin"""
        plugin = MagicMock()
        plugin.get_store_stats.return_value = {
            "count": count,
            "max_slots": max_slots,
            "utilization": utilization,
            "pinned_count": pinned,
        }
        plugin.is_attached = attached
        plugin._attached_model_name = "TestModel"
        plugin._hooks = [MagicMock()] * hooks

        # GateSystem mock
        import torch
        param = torch.tensor([1.0, 2.0])
        plugin.gate_system.parameters.return_value = [param]

        # Retriever mock
        plugin.retriever.get_stats.return_value = {"type": "NullRetriever"}

        # EventBus mock
        plugin.event_bus.enabled = True
        plugin.event_bus.get_stats.return_value = {
            "buffer_size": 100,
            "buffer_capacity": 10000,
            "total_emitted": 500,
            "total_consumed": 500,
            "subscribers": {},
        }

        return plugin

    def test_check_healthy(self):
        """健康状态"""
        checker = HealthChecker()
        plugin = self._make_mock_plugin()
        checker.bind_plugin(plugin)

        result = checker.check()
        assert result["status"] == "healthy"
        assert "components" in result
        assert "kv_store" in result["components"]
        assert "gate_system" in result["components"]
        assert "retriever" in result["components"]
        assert "event_bus" in result["components"]
        assert "attachment" in result["components"]

    def test_check_degraded_high_utilization(self):
        """高利用率 → degraded"""
        checker = HealthChecker()
        plugin = self._make_mock_plugin(utilization=0.96)
        checker.bind_plugin(plugin)

        result = checker.check()
        kv_status = result["components"]["kv_store"]["status"]
        assert kv_status == "degraded"

    def test_check_not_attached(self):
        """未挂载 → degraded"""
        checker = HealthChecker()
        plugin = self._make_mock_plugin(attached=False)
        checker.bind_plugin(plugin)

        result = checker.check()
        attach_status = result["components"]["attachment"]["status"]
        assert attach_status == "degraded"

    def test_check_no_plugin(self):
        """无 plugin"""
        checker = HealthChecker()
        result = checker.check()
        assert result["status"] == "healthy"
        assert len(result["components"]) == 0

    def test_custom_check(self):
        """自定义健康检查"""
        checker = HealthChecker()
        checker.add_check("custom", lambda: {
            "status": "healthy",
            "detail": "all good",
        })

        result = checker.check()
        assert "custom" in result["components"]
        assert result["components"]["custom"]["status"] == "healthy"

    def test_custom_check_failure(self):
        """自定义检查失败"""
        checker = HealthChecker()
        checker.add_check("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        result = checker.check()
        assert result["components"]["bad"]["status"] == "unhealthy"

    def test_overall_unhealthy(self):
        """任一组件 unhealthy → 整体 unhealthy"""
        checker = HealthChecker()
        checker.add_check("broken", lambda: {"status": "unhealthy"})
        checker.add_check("ok", lambda: {"status": "healthy"})

        result = checker.check()
        assert result["status"] == "unhealthy"

    def test_overall_degraded(self):
        """任一组件 degraded → 整体 degraded"""
        checker = HealthChecker()
        checker.add_check("slow", lambda: {"status": "degraded"})
        checker.add_check("ok", lambda: {"status": "healthy"})

        result = checker.check()
        assert result["status"] == "degraded"

    def test_timestamp(self):
        """检查结果包含时间戳"""
        checker = HealthChecker()
        result = checker.check()
        assert "timestamp" in result
        assert isinstance(result["timestamp"], float)
