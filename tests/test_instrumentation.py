"""
Instrumentation 单元测试 (EventBus, ForwardMetrics, AuditLog)
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aga.instrumentation import EventBus, Event, ForwardMetrics, AuditLog


class TestEventBus:
    """EventBus 测试"""

    @pytest.fixture
    def bus(self):
        return EventBus(buffer_size=100, enabled=True)

    def test_emit_and_query(self, bus):
        """测试发射和查询"""
        bus.emit("forward", {"gate_mean": 0.3})
        bus.emit("forward", {"gate_mean": 0.5})

        events = bus.query(event_type="forward")
        assert len(events) == 2
        assert events[0]["data"]["gate_mean"] == 0.3
        assert events[1]["data"]["gate_mean"] == 0.5

    def test_subscribe(self, bus):
        """测试订阅"""
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("forward", handler)
        bus.emit("forward", {"test": True})

        assert len(received) == 1
        assert received[0].data["test"] is True

    def test_wildcard_subscribe(self, bus):
        """测试通配符订阅"""
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("*", handler)
        bus.emit("forward", {"type": "forward"})
        bus.emit("audit", {"type": "audit"})

        assert len(received) == 2

    def test_unsubscribe(self, bus):
        """测试取消订阅"""
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("forward", handler)
        bus.emit("forward", {"test": 1})
        assert len(received) == 1

        bus.unsubscribe("forward", handler)
        bus.emit("forward", {"test": 2})
        assert len(received) == 1  # 不应增加

    def test_disabled(self):
        """测试禁用状态"""
        bus = EventBus(enabled=False)
        bus.emit("forward", {"test": True})
        events = bus.query()
        assert len(events) == 0

    def test_buffer_overflow(self):
        """测试缓冲区溢出"""
        bus = EventBus(buffer_size=5)
        for i in range(10):
            bus.emit("forward", {"idx": i})

        events = bus.query()
        assert len(events) == 5
        assert events[0]["data"]["idx"] == 5  # 最早的 5 条被丢弃

    def test_query_with_since(self, bus):
        """测试按时间过滤"""
        bus.emit("forward", {"idx": 0})
        time.sleep(0.01)
        cutoff = time.time()
        time.sleep(0.01)
        bus.emit("forward", {"idx": 1})

        events = bus.query(since=cutoff)
        assert len(events) == 1
        assert events[0]["data"]["idx"] == 1

    def test_get_stats(self, bus):
        """测试统计"""
        bus.emit("forward", {"test": True})
        bus.emit("audit", {"test": True})

        stats = bus.get_stats()
        assert stats["total_emitted"] == 2
        assert stats["buffer_size"] == 2


class TestForwardMetrics:
    """ForwardMetrics 测试"""

    @pytest.fixture
    def metrics(self):
        bus = EventBus(buffer_size=100)
        return ForwardMetrics(bus)

    def test_record_applied(self, metrics):
        """测试记录 AGA 应用"""
        metrics.record(aga_applied=True, gate_mean=0.5, entropy_mean=1.2, layer_idx=0)
        summary = metrics.get_summary()
        assert summary["forward_total"] == 1
        assert summary["forward_applied"] == 1
        assert summary["forward_bypassed"] == 0
        assert summary["activation_rate"] == 1.0

    def test_record_bypassed(self, metrics):
        """测试记录旁路"""
        metrics.record(aga_applied=False, gate_mean=0.01, layer_idx=0)
        summary = metrics.get_summary()
        assert summary["forward_total"] == 1
        assert summary["forward_applied"] == 0
        assert summary["forward_bypassed"] == 1
        assert summary["activation_rate"] == 0.0

    def test_mixed_records(self, metrics):
        """测试混合记录"""
        for i in range(10):
            metrics.record(
                aga_applied=(i % 3 == 0),
                gate_mean=0.1 * i,
                entropy_mean=0.5 * i,
                layer_idx=i % 3,
            )

        summary = metrics.get_summary()
        assert summary["forward_total"] == 10
        assert summary["forward_applied"] == 4  # 0, 3, 6, 9
        assert summary["forward_bypassed"] == 6
        assert 0 < summary["activation_rate"] < 1

    def test_layer_stats(self, metrics):
        """测试按层统计"""
        metrics.record(aga_applied=True, gate_mean=0.5, layer_idx=0)
        metrics.record(aga_applied=False, gate_mean=0.01, layer_idx=0)
        metrics.record(aga_applied=True, gate_mean=0.6, layer_idx=1)

        summary = metrics.get_summary()
        assert 0 in summary["layer_stats"]
        assert summary["layer_stats"][0]["total"] == 2
        assert summary["layer_stats"][0]["applied"] == 1
        assert summary["layer_stats"][1]["total"] == 1

    def test_latency_tracking(self, metrics):
        """测试延迟追踪"""
        for i in range(100):
            metrics.record(
                aga_applied=True,
                gate_mean=0.5,
                latency_us=float(i + 10),
            )

        summary = metrics.get_summary()
        assert "latency_p50_us" in summary
        assert "latency_p95_us" in summary
        assert "latency_p99_us" in summary

    def test_reset(self, metrics):
        """测试重置"""
        metrics.record(aga_applied=True, gate_mean=0.5)
        metrics.reset()
        summary = metrics.get_summary()
        assert summary["forward_total"] == 0


class TestAuditLog:
    """AuditLog 测试"""

    @pytest.fixture
    def audit(self):
        bus = EventBus(buffer_size=100)
        return AuditLog(bus, buffer_size=100, log_level="DEBUG")

    def test_record_and_query(self, audit):
        """测试记录和查询"""
        audit.record("register", {"id": "fact_001", "reliability": 0.9})
        audit.record("register", {"id": "fact_002", "reliability": 0.8})

        trail = audit.query()
        assert len(trail) == 2
        assert trail[0]["operation"] == "register"
        assert trail[0]["details"]["id"] == "fact_001"

    def test_query_by_operation(self, audit):
        """测试按操作类型查询"""
        audit.record("register", {"id": "fact_001"})
        audit.record("unregister", {"id": "fact_001"})
        audit.record("register", {"id": "fact_002"})

        trail = audit.query(operation="register")
        assert len(trail) == 2

        trail = audit.query(operation="unregister")
        assert len(trail) == 1

    def test_query_limit(self, audit):
        """测试查询限制"""
        for i in range(20):
            audit.record("register", {"id": f"fact_{i:03d}"})

        trail = audit.query(limit=5)
        assert len(trail) == 5

    def test_query_success_only(self, audit):
        """测试只查询成功的"""
        audit.record("register", {"id": "fact_001"}, success=True)
        audit.record("register", {"id": "fact_002"}, success=False, error="test error")

        trail = audit.query(success_only=True)
        assert len(trail) == 1
        assert trail[0]["success"] is True

    def test_get_stats(self, audit):
        """测试统计"""
        audit.record("register", {"id": "fact_001"})
        audit.record("register", {"id": "fact_002"})
        audit.record("unregister", {"id": "fact_001"})

        stats = audit.get_stats()
        assert stats["total_entries"] == 3
        assert stats["by_operation"]["register"] == 2
        assert stats["by_operation"]["unregister"] == 1
