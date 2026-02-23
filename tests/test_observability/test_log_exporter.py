"""
tests/test_observability/test_log_exporter.py — LogExporter 测试
"""
import pytest
import logging
from aga.instrumentation import EventBus
from aga_observability.log_exporter import LogExporter


class TestLogExporter:
    """LogExporter 测试"""

    def test_init_json_format(self):
        """JSON 格式初始化"""
        exporter = LogExporter(format="json")
        assert exporter._format == "json"
        exporter.shutdown()

    def test_init_text_format(self):
        """Text 格式初始化"""
        exporter = LogExporter(format="text")
        assert exporter._format == "text"
        exporter.shutdown()

    def test_subscribe(self):
        """订阅 EventBus"""
        bus = EventBus()
        exporter = LogExporter(format="json", level="DEBUG")
        exporter.subscribe(bus)
        assert exporter._subscribed is True
        exporter.unsubscribe(bus)
        assert exporter._subscribed is False
        exporter.shutdown()

    def test_forward_event(self):
        """处理 forward 事件"""
        bus = EventBus()
        exporter = LogExporter(format="json", level="DEBUG")
        exporter.subscribe(bus)

        bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.5,
            "layer_idx": 0,
            "latency_us": 45.0,
        })

        assert exporter._event_count == 1
        exporter.shutdown()

    def test_retrieval_event(self):
        """处理 retrieval 事件"""
        bus = EventBus()
        exporter = LogExporter(format="json", level="DEBUG")
        exporter.subscribe(bus)

        bus.emit("retrieval", {
            "layer_idx": 0,
            "results_count": 5,
            "injected_count": 3,
            "budget_remaining": 2,
        })

        assert exporter._event_count == 1
        exporter.shutdown()

    def test_audit_event(self):
        """处理 audit 事件"""
        bus = EventBus()
        exporter = LogExporter(format="json", level="DEBUG")
        exporter.subscribe(bus)

        bus.emit("audit", {
            "operation": "register",
            "details": {"id": "test"},
            "success": True,
        })

        assert exporter._event_count == 1
        exporter.shutdown()

    def test_multiple_events(self):
        """多事件处理"""
        bus = EventBus()
        exporter = LogExporter(format="json", level="DEBUG")
        exporter.subscribe(bus)

        for i in range(100):
            bus.emit("forward", {
                "aga_applied": i % 2 == 0,
                "gate_mean": 0.5,
                "layer_idx": 0,
                "latency_us": 10.0,
            })

        assert exporter._event_count == 100
        exporter.shutdown()

    def test_get_stats(self):
        """统计信息"""
        exporter = LogExporter(format="json")
        stats = exporter.get_stats()
        assert "subscribed" in stats
        assert "format" in stats
        assert "event_count" in stats
        exporter.shutdown()

    def test_file_output(self, tmp_path):
        """文件输出"""
        log_file = str(tmp_path / "test.log")
        exporter = LogExporter(format="json", file=log_file, level="DEBUG")

        bus = EventBus()
        exporter.subscribe(bus)

        bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "layer_idx": 0,
            "latency_us": 10.0,
        })

        exporter.shutdown()

        # 验证文件已创建
        import os
        assert os.path.exists(log_file)
