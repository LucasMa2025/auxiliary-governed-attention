"""
tests/test_streaming.py — StreamingSession 单元测试
"""
import time
import pytest
import torch

from aga import AGAPlugin, AGAConfig
from aga.streaming import StreamingSession


@pytest.fixture
def plugin():
    """创建测试用 AGAPlugin"""
    config = AGAConfig(
        hidden_dim=64,
        bottleneck_dim=16,
        max_slots=32,
        device="cpu",
        instrumentation_enabled=True,
        event_buffer_size=1000,
    )
    p = AGAPlugin(config)
    # 注册一些测试知识
    for i in range(5):
        p.register(
            id=f"test_{i}",
            key=torch.randn(16),
            value=torch.randn(64),
            reliability=0.9,
        )
    return p


class TestStreamingSessionCreation:
    """测试 StreamingSession 创建"""

    def test_create_session(self, plugin):
        session = plugin.create_streaming_session()
        assert session.is_active
        assert session.step_count == 0
        session.close()

    def test_create_session_with_buffer_size(self, plugin):
        session = plugin.create_streaming_session(diagnostics_buffer_size=500)
        assert session.is_active
        assert session._buffer_size == 500
        session.close()

    def test_create_multiple_sessions(self, plugin):
        s1 = plugin.create_streaming_session()
        s2 = plugin.create_streaming_session()
        assert s1.is_active
        assert s2.is_active
        s1.close()
        s2.close()

    def test_session_context_manager(self, plugin):
        with plugin.create_streaming_session() as session:
            assert session.is_active
        assert not session.is_active


class TestStreamingSessionDiagnostics:
    """测试 StreamingSession 诊断功能"""

    def test_initial_diagnostics(self, plugin):
        session = plugin.create_streaming_session()
        diag = session.get_step_diagnostics()
        assert diag["step"] == 0
        assert diag["aga_applied"] is False
        assert diag["gate_mean"] == 0.0
        session.close()

    def test_diagnostics_after_forward_event(self, plugin):
        session = plugin.create_streaming_session()

        # 模拟 forward 事件
        plugin.event_bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.2,
            "latency_us": 50.0,
            "layer_idx": 0,
        })

        diag = session.get_step_diagnostics()
        assert diag["step"] == 1
        assert diag["aga_applied"] is True
        assert diag["gate_mean"] == 0.5
        assert diag["entropy_mean"] == 1.2
        assert diag["latency_us"] == 50.0
        session.close()

    def test_multiple_forward_events(self, plugin):
        session = plugin.create_streaming_session()

        # 模拟多个 forward 事件
        for i in range(10):
            plugin.event_bus.emit("forward", {
                "aga_applied": i % 3 == 0,
                "gate_mean": 0.1 * i,
                "entropy_mean": 0.5 + 0.1 * i,
                "latency_us": 30.0 + i,
                "layer_idx": 0,
            })

        assert session.step_count == 10
        diag = session.get_step_diagnostics()
        assert diag["step"] == 10
        session.close()

    def test_recent_diagnostics(self, plugin):
        session = plugin.create_streaming_session()

        for i in range(20):
            plugin.event_bus.emit("forward", {
                "aga_applied": i % 2 == 0,
                "gate_mean": 0.1 * i,
                "entropy_mean": 1.0,
                "latency_us": 40.0,
                "layer_idx": 0,
            })

        recent = session.get_recent_diagnostics(n=5)
        assert len(recent) == 5
        assert recent[-1]["step"] == 20
        assert recent[0]["step"] == 16
        session.close()


class TestStreamingSessionSummary:
    """测试 StreamingSession 统计摘要"""

    def test_empty_summary(self, plugin):
        session = plugin.create_streaming_session()
        summary = session.get_session_summary()
        assert summary["total_steps"] == 0
        assert summary["injection_count"] == 0
        assert summary["injection_rate"] == 0.0
        assert summary["active"] is True
        session.close()

    def test_summary_with_events(self, plugin):
        session = plugin.create_streaming_session()

        # 10 个事件，其中 4 个注入
        for i in range(10):
            plugin.event_bus.emit("forward", {
                "aga_applied": i < 4,
                "gate_mean": 0.5 if i < 4 else 0.0,
                "entropy_mean": 1.5 if i < 4 else 0.3,
                "latency_us": 50.0,
                "layer_idx": 0,
            })

        summary = session.get_session_summary()
        assert summary["total_steps"] == 10
        assert summary["injection_count"] == 4
        assert summary["injection_rate"] == 0.4
        assert summary["bypass_count"] == 6
        assert summary["bypass_rate"] == 0.6
        assert summary["knowledge_count"] == 5
        assert summary["active"] is True
        session.close()

    def test_summary_after_close(self, plugin):
        session = plugin.create_streaming_session()

        plugin.event_bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.0,
            "latency_us": 40.0,
            "layer_idx": 0,
        })

        session.close()
        summary = session.get_session_summary()
        assert summary["total_steps"] == 1
        assert summary["active"] is False


class TestStreamingSessionKnowledgeUpdate:
    """测试流式会话中的知识热更新"""

    def test_update_knowledge(self, plugin):
        session = plugin.create_streaming_session()
        initial_count = plugin.knowledge_count

        success = session.update_knowledge(
            id="dynamic_001",
            key=torch.randn(16),
            value=torch.randn(64),
            reliability=0.85,
        )
        assert success
        assert plugin.knowledge_count == initial_count + 1
        session.close()

    def test_remove_knowledge(self, plugin):
        session = plugin.create_streaming_session()
        initial_count = plugin.knowledge_count

        success = session.remove_knowledge("test_0")
        assert success
        assert plugin.knowledge_count == initial_count - 1
        session.close()

    def test_update_after_close(self, plugin):
        session = plugin.create_streaming_session()
        session.close()

        success = session.update_knowledge(
            id="late_fact",
            key=torch.randn(16),
            value=torch.randn(64),
        )
        assert not success

    def test_remove_after_close(self, plugin):
        session = plugin.create_streaming_session()
        session.close()

        success = session.remove_knowledge("test_0")
        assert not success


class TestStreamingSessionLifecycle:
    """测试流式会话生命周期"""

    def test_close_resets_decay(self, plugin):
        session = plugin.create_streaming_session()
        # 模拟一些衰减上下文（使用线程隔离的接口）
        contexts = plugin._get_decay_contexts()
        contexts[0] = "some_context"
        contexts[1] = "another_context"

        session.close()
        assert len(plugin._get_decay_contexts()) == 0

    def test_double_close(self, plugin):
        session = plugin.create_streaming_session()
        session.close()
        session.close()  # 不应报错
        assert not session.is_active

    def test_events_not_received_after_close(self, plugin):
        session = plugin.create_streaming_session()

        plugin.event_bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.0,
            "latency_us": 40.0,
            "layer_idx": 0,
        })
        assert session.step_count == 1

        session.close()

        # 关闭后的事件不应被记录
        plugin.event_bus.emit("forward", {
            "aga_applied": True,
            "gate_mean": 0.5,
            "entropy_mean": 1.0,
            "latency_us": 40.0,
            "layer_idx": 0,
        })
        assert session.step_count == 1

    def test_reset_decay(self, plugin):
        session = plugin.create_streaming_session()
        contexts = plugin._get_decay_contexts()
        contexts[0] = "context"
        session.reset_decay()
        assert len(plugin._get_decay_contexts()) == 0
        session.close()

    def test_repr(self, plugin):
        session = plugin.create_streaming_session()
        repr_str = repr(session)
        assert "StreamingSession" in repr_str
        assert "steps=0" in repr_str
        assert "active=True" in repr_str
        session.close()


class TestStreamingSessionAudit:
    """测试流式会话审计日志"""

    def test_session_start_audit(self, plugin):
        session = plugin.create_streaming_session()
        trail = plugin.get_audit_trail(limit=10, operation="streaming_session_start")
        assert len(trail) >= 1
        session.close()

    def test_session_end_audit(self, plugin):
        session = plugin.create_streaming_session()
        session.close()
        trail = plugin.get_audit_trail(limit=10, operation="streaming_session_end")
        assert len(trail) >= 1

    def test_knowledge_update_audit(self, plugin):
        session = plugin.create_streaming_session()
        session.update_knowledge(
            id="audit_test",
            key=torch.randn(16),
            value=torch.randn(64),
        )
        trail = plugin.get_audit_trail(limit=10, operation="streaming_knowledge_update")
        assert len(trail) >= 1
        session.close()


class TestStreamingSessionIntegration:
    """集成测试：模拟完整的流式生成过程"""

    def test_full_streaming_flow(self, plugin):
        """模拟完整的流式生成流程"""
        with plugin.create_streaming_session() as session:
            # 模拟 20 个 token 的生成
            for step in range(20):
                # 模拟 forward 事件（每 3 个 token 一次注入）
                is_injection = step % 3 == 0
                plugin.event_bus.emit("forward", {
                    "aga_applied": is_injection,
                    "gate_mean": 0.6 if is_injection else 0.02,
                    "entropy_mean": 1.8 if is_injection else 0.3,
                    "latency_us": 60.0 if is_injection else 5.0,
                    "layer_idx": 0,
                })

                # 获取实时诊断
                diag = session.get_step_diagnostics()
                assert diag["step"] == step + 1

            # 获取摘要
            summary = session.get_session_summary()
            assert summary["total_steps"] == 20
            assert summary["injection_count"] == 7  # 0,3,6,9,12,15,18
            assert summary["injection_rate"] == pytest.approx(0.35, abs=0.01)

    def test_streaming_with_knowledge_updates(self, plugin):
        """模拟带知识热更新的流式生成"""
        with plugin.create_streaming_session() as session:
            initial_count = plugin.knowledge_count

            # 前 5 步
            for _ in range(5):
                plugin.event_bus.emit("forward", {
                    "aga_applied": True,
                    "gate_mean": 0.5,
                    "entropy_mean": 1.5,
                    "latency_us": 50.0,
                    "layer_idx": 0,
                })

            # 中途添加知识
            session.update_knowledge(
                id="mid_stream_fact",
                key=torch.randn(16),
                value=torch.randn(64),
                reliability=0.9,
            )
            assert plugin.knowledge_count == initial_count + 1

            # 后 5 步
            for _ in range(5):
                plugin.event_bus.emit("forward", {
                    "aga_applied": True,
                    "gate_mean": 0.6,
                    "entropy_mean": 1.8,
                    "latency_us": 55.0,
                    "layer_idx": 0,
                })

            # 移除知识
            session.remove_knowledge("mid_stream_fact")
            assert plugin.knowledge_count == initial_count

            summary = session.get_session_summary()
            assert summary["total_steps"] == 10
            assert summary["injection_count"] == 10

    def test_buffer_overflow(self, plugin):
        """测试诊断缓冲区溢出"""
        session = plugin.create_streaming_session(diagnostics_buffer_size=10)

        # 发射 20 个事件
        for i in range(20):
            plugin.event_bus.emit("forward", {
                "aga_applied": True,
                "gate_mean": 0.1 * i,
                "entropy_mean": 1.0,
                "latency_us": 40.0,
                "layer_idx": 0,
            })

        # 缓冲区只保留最近 10 个
        recent = session.get_recent_diagnostics(n=20)
        assert len(recent) == 10
        assert recent[0]["step"] == 11  # 最早的是第 11 步

        # 但累计统计应该包含所有 20 步
        summary = session.get_session_summary()
        assert summary["total_steps"] == 20
        assert summary["injection_count"] == 20

        session.close()
