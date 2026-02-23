"""
Gate 系统单元测试 (EntropyGateSystem, PersistenceDecay)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aga.config import AGAConfig
from aga.gate.entropy_gate import EntropyGateSystem
from aga.gate.decay import PersistenceDecay, DecayContext, DecayStrategy


class TestEntropyGateSystem:
    """EntropyGateSystem 测试"""

    @pytest.fixture
    def config(self):
        return AGAConfig(
            hidden_dim=128,
            bottleneck_dim=16,
            device="cpu",
        )

    @pytest.fixture
    def gate_system(self, config):
        return EntropyGateSystem(config)

    def test_init(self, gate_system):
        """测试初始化"""
        assert gate_system.tau_low == 0.5
        assert gate_system.tau_high == 2.0
        assert gate_system.max_gate == 0.8

    def test_forward_shape(self, gate_system):
        """测试输出形状"""
        hidden_states = torch.randn(2, 10, 128)
        gate, diagnostics = gate_system(hidden_states)
        assert gate.shape == (2, 10)
        assert diagnostics.entropy_mean >= 0

    def test_forward_gate_range(self, gate_system):
        """测试门控值范围"""
        hidden_states = torch.randn(4, 20, 128)
        gate, _ = gate_system(hidden_states)
        assert gate.min() >= 0.0
        assert gate.max() <= gate_system.max_gate + 0.01  # 允许小误差

    def test_low_entropy_bypass(self):
        """测试低熵旁路"""
        config = AGAConfig(
            hidden_dim=128,
            tau_low=100.0,  # 极高阈值，所有都应该被否决
            tau_high=200.0,
            device="cpu",
        )
        gate_system = EntropyGateSystem(config)
        hidden_states = torch.randn(2, 10, 128)
        gate, diagnostics = gate_system(hidden_states)
        # 大部分 gate 应该为 0（因为 tau_low 极高）
        assert gate.mean().item() < 0.1

    def test_early_exit(self):
        """测试 early exit"""
        config = AGAConfig(
            hidden_dim=128,
            early_exit_enabled=True,
            early_exit_threshold=100.0,  # 极高阈值
            device="cpu",
        )
        gate_system = EntropyGateSystem(config)
        hidden_states = torch.randn(2, 10, 128)
        gate, diagnostics = gate_system(hidden_states)
        assert diagnostics.early_exit is True

    def test_gate0_namespace_filter(self):
        """测试 Gate-0 命名空间过滤"""
        config = AGAConfig(
            hidden_dim=128,
            gate0_enabled=True,
            gate0_disabled_namespaces=["system"],
            device="cpu",
        )
        gate_system = EntropyGateSystem(config)
        hidden_states = torch.randn(2, 10, 128)

        # 被禁用的命名空间
        gate, diagnostics = gate_system(
            hidden_states, context={"namespace": "system"}
        )
        assert diagnostics.gate0_passed is False
        assert gate.sum().item() == 0.0

        # 正常命名空间
        gate, diagnostics = gate_system(
            hidden_states, context={"namespace": "user"}
        )
        assert diagnostics.gate0_passed is True

    def test_update_thresholds(self, gate_system):
        """测试运行时更新阈值"""
        gate_system.update_thresholds(tau_low=0.3, tau_high=1.5)
        assert gate_system.tau_low == 0.3
        assert gate_system.tau_high == 1.5

    def test_diagnostics(self, gate_system):
        """测试诊断信息"""
        hidden_states = torch.randn(2, 10, 128)
        _, diagnostics = gate_system(hidden_states)
        assert isinstance(diagnostics.entropy_mean, float)
        assert isinstance(diagnostics.gate_mean, float)
        assert isinstance(diagnostics.gate_max, float)
        assert isinstance(diagnostics.veto_ratio, float)


class TestPersistenceDecay:
    """PersistenceDecay 测试"""

    @pytest.fixture
    def config(self):
        return AGAConfig(
            decay_strategy="exponential",
            decay_gamma=0.9,
            decay_hard_reset_threshold=3.0,
            device="cpu",
        )

    @pytest.fixture
    def decay(self, config):
        return PersistenceDecay(config)

    def test_init(self, decay):
        """测试初始化"""
        assert decay.strategy == DecayStrategy.EXPONENTIAL
        assert decay.gamma == 0.9

    def test_first_layer_no_decay(self, decay):
        """测试第一层不衰减"""
        gate = torch.ones(2, 10) * 0.5
        effective, context = decay(gate, layer_idx=0)
        # 第一层不应衰减
        assert torch.allclose(effective, gate, atol=0.01)

    def test_subsequent_layers_decay(self, decay):
        """测试后续层衰减"""
        gate = torch.ones(2, 10) * 0.5
        context = DecayContext()

        # 第一层
        effective0, context = decay(gate, context, layer_idx=0)
        # 第二层
        effective1, context = decay(gate, context, layer_idx=1)
        # 第三层
        effective2, context = decay(gate, context, layer_idx=2)

        # 后续层应该越来越小
        assert effective1.mean() < effective0.mean()
        assert effective2.mean() < effective1.mean()

    def test_hard_reset(self):
        """测试硬重置"""
        config = AGAConfig(
            decay_strategy="none",
            decay_hard_reset_threshold=1.0,  # 低阈值
            device="cpu",
        )
        decay = PersistenceDecay(config)
        context = DecayContext()

        gate = torch.ones(2, 10) * 0.5
        # 多次累积应触发硬重置
        for i in range(5):
            effective, context = decay(gate, context, layer_idx=i)

        assert context.hard_reset_triggered is True

    def test_decay_context_clone(self):
        """测试上下文克隆"""
        context = DecayContext(accumulated_gate=1.5, layer_idx=3)
        context.record(0.5)

        cloned = context.clone()
        assert cloned.accumulated_gate == 1.5
        assert cloned.layer_idx == 3
        assert len(cloned.gate_history) == 1

        # 修改原始不影响克隆
        context.accumulated_gate = 2.0
        assert cloned.accumulated_gate == 1.5

    def test_decay_context_reset(self):
        """测试上下文重置"""
        context = DecayContext(accumulated_gate=1.5, layer_idx=3)
        context.record(0.5)
        context.reset()

        assert context.accumulated_gate == 0.0
        assert context.layer_idx == 0
        assert len(context.gate_history) == 0

    def test_linear_decay(self):
        """测试线性衰减"""
        config = AGAConfig(
            decay_strategy="linear",
            decay_gamma=0.9,
            device="cpu",
        )
        decay = PersistenceDecay(config)
        context = DecayContext()

        gate = torch.ones(2, 10) * 0.5
        effective0, context = decay(gate, context, layer_idx=0)
        effective1, context = decay(gate, context, layer_idx=1)

        assert effective1.mean() < effective0.mean()

    def test_diagnostics(self, decay):
        """测试诊断信息"""
        context = DecayContext()
        gate = torch.ones(2, 10) * 0.5
        _, context = decay(gate, context, layer_idx=0)

        diag = decay.get_diagnostics(context)
        assert diag["strategy"] == "exponential"
        assert diag["gamma"] == 0.9
        assert "accumulated_gate" in diag
        assert "gate_history" in diag
