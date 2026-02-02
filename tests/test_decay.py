"""
持久化衰减模块单元测试

测试覆盖：
- 衰减策略
- 衰减上下文
- 硬重置机制
- 衰减管理器
"""
import pytest
import torch

import sys
sys.path.insert(0, '..')

from aga.decay import (
    PersistenceDecay,
    DecayConfig,
    DecayContext,
    DecayStrategy,
    DecayAwareAGAManager,
    create_decay_config,
)


class TestDecayConfig:
    """衰减配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DecayConfig()
        
        assert config.strategy == DecayStrategy.EXPONENTIAL
        assert config.gamma == 0.9
        assert config.hard_reset_threshold == 3.0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DecayConfig(
            strategy=DecayStrategy.LINEAR,
            gamma=0.8,
            delta=0.15,
        )
        
        assert config.strategy == DecayStrategy.LINEAR
        assert config.gamma == 0.8
        assert config.delta == 0.15


class TestDecayContext:
    """衰减上下文测试"""
    
    def test_initial_state(self):
        """测试初始状态"""
        context = DecayContext()
        
        assert context.accumulated_gate == 0.0
        assert context.layer_idx == 0
        assert context.hard_reset_triggered is False
        assert len(context.gate_history) == 0
    
    def test_record(self):
        """测试记录功能"""
        context = DecayContext()
        context.layer_idx = 1
        context.accumulated_gate = 0.5
        
        context.record(0.3)
        
        assert len(context.gate_history) == 1
        assert context.gate_history[0]['layer'] == 1
        assert context.gate_history[0]['gate_mean'] == 0.3
    
    def test_reset(self):
        """测试重置功能"""
        context = DecayContext()
        context.accumulated_gate = 1.5
        context.layer_idx = 3
        context.record(0.5)
        
        context.reset()
        
        assert context.accumulated_gate == 0.0
        assert context.layer_idx == 0
        assert len(context.gate_history) == 0
    
    def test_clone(self):
        """测试克隆功能"""
        context = DecayContext()
        context.accumulated_gate = 1.0
        context.layer_idx = 2
        
        cloned = context.clone()
        
        assert cloned.accumulated_gate == 1.0
        assert cloned.layer_idx == 2
        
        # 修改原始不影响克隆
        context.accumulated_gate = 2.0
        assert cloned.accumulated_gate == 1.0


class TestExponentialDecay:
    """指数衰减测试"""
    
    @pytest.fixture
    def decay(self):
        """创建指数衰减实例"""
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.9,
            enable_hard_reset=False,
        )
        return PersistenceDecay(config)
    
    def test_first_layer_no_decay(self, decay):
        """测试第一层不衰减"""
        raw_gate = torch.ones(2, 4) * 0.5
        context = DecayContext()
        
        effective_gate, context = decay(raw_gate, context, layer_idx=0)
        
        assert torch.allclose(effective_gate, raw_gate)
    
    def test_second_layer_decay(self, decay):
        """测试第二层衰减"""
        raw_gate = torch.ones(2, 4) * 0.5
        context = DecayContext()
        
        # 第一层
        _, context = decay(raw_gate, context, layer_idx=0)
        
        # 第二层
        effective_gate, context = decay(raw_gate, context, layer_idx=1)
        
        # 应该衰减为 0.5 * 0.9 = 0.45
        expected = raw_gate * 0.9
        assert torch.allclose(effective_gate, expected)
    
    def test_multiple_layers_decay(self, decay):
        """测试多层衰减"""
        raw_gate = torch.ones(2, 4) * 1.0
        context = DecayContext()
        
        for layer_idx in range(5):
            effective_gate, context = decay(raw_gate, context, layer_idx)
            
            if layer_idx == 0:
                expected_factor = 1.0
            else:
                expected_factor = 0.9 ** layer_idx
            
            expected = raw_gate * expected_factor
            assert torch.allclose(effective_gate, expected, atol=1e-5)


class TestLinearDecay:
    """线性衰减测试"""
    
    @pytest.fixture
    def decay(self):
        """创建线性衰减实例"""
        config = DecayConfig(
            strategy=DecayStrategy.LINEAR,
            delta=0.2,
            enable_hard_reset=False,
        )
        return PersistenceDecay(config)
    
    def test_linear_decay_progression(self, decay):
        """测试线性衰减进程"""
        raw_gate = torch.ones(2, 4) * 1.0
        context = DecayContext()
        
        # 第一层: factor = 1.0
        effective_gate, context = decay(raw_gate, context, layer_idx=0)
        assert torch.allclose(effective_gate, raw_gate)
        
        # 第二层: factor = 1.0 - 0.2 * 1 = 0.8
        effective_gate, context = decay(raw_gate, context, layer_idx=1)
        assert torch.allclose(effective_gate, raw_gate * 0.8)
        
        # 第三层: factor = 1.0 - 0.2 * 2 = 0.6
        effective_gate, context = decay(raw_gate, context, layer_idx=2)
        assert torch.allclose(effective_gate, raw_gate * 0.6)
    
    def test_linear_decay_floor(self, decay):
        """测试线性衰减下限"""
        raw_gate = torch.ones(2, 4) * 1.0
        context = DecayContext()
        
        # 第 6 层: factor = max(0, 1.0 - 0.2 * 5) = 0.0
        effective_gate, context = decay(raw_gate, context, layer_idx=5)
        
        assert torch.allclose(effective_gate, torch.zeros_like(raw_gate))


class TestHardReset:
    """硬重置测试"""
    
    @pytest.fixture
    def decay(self):
        """创建带硬重置的衰减实例"""
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.95,  # 较慢衰减
            hard_reset_threshold=1.5,
            enable_hard_reset=True,
        )
        return PersistenceDecay(config)
    
    def test_hard_reset_triggered(self, decay):
        """测试硬重置触发"""
        raw_gate = torch.ones(2, 4) * 0.6
        context = DecayContext()
        
        # 累积直到触发硬重置
        for layer_idx in range(10):
            effective_gate, context = decay(raw_gate, context, layer_idx)
            
            if context.hard_reset_triggered:
                break
        
        assert context.hard_reset_triggered is True
        assert context.accumulated_gate == 0.0
        assert torch.allclose(effective_gate, torch.zeros_like(raw_gate))
    
    def test_no_hard_reset_below_threshold(self):
        """测试阈值以下不触发硬重置"""
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.5,  # 快速衰减
            hard_reset_threshold=10.0,  # 高阈值
            enable_hard_reset=True,
        )
        decay = PersistenceDecay(config)
        
        raw_gate = torch.ones(2, 4) * 0.3
        context = DecayContext()
        
        for layer_idx in range(5):
            _, context = decay(raw_gate, context, layer_idx)
        
        assert context.hard_reset_triggered is False


class TestNoDecay:
    """无衰减测试"""
    
    def test_no_decay_strategy(self):
        """测试无衰减策略"""
        config = DecayConfig(strategy=DecayStrategy.NONE)
        decay = PersistenceDecay(config)
        
        raw_gate = torch.ones(2, 4) * 0.5
        context = DecayContext()
        
        for layer_idx in range(5):
            effective_gate, context = decay(raw_gate, context, layer_idx)
            assert torch.allclose(effective_gate, raw_gate)


class TestDecayAwareAGAManager:
    """衰减感知 AGA 管理器测试"""
    
    @pytest.fixture
    def manager(self):
        """创建管理器实例"""
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.9,
        )
        return DecayAwareAGAManager(config)
    
    def test_get_context(self, manager):
        """测试获取上下文"""
        context1 = manager.get_context("request_1")
        context2 = manager.get_context("request_1")
        
        # 同一请求应该返回同一上下文
        assert context1 is context2
        
        # 不同请求应该返回不同上下文
        context3 = manager.get_context("request_2")
        assert context1 is not context3
    
    def test_apply_decay(self, manager):
        """测试应用衰减"""
        raw_gate = torch.ones(2, 4) * 0.5
        
        # 第一层
        effective_gate = manager.apply_decay("request_1", raw_gate, layer_idx=0)
        assert torch.allclose(effective_gate, raw_gate)
        
        # 第二层
        effective_gate = manager.apply_decay("request_1", raw_gate, layer_idx=1)
        assert torch.allclose(effective_gate, raw_gate * 0.9)
    
    def test_finish_request(self, manager):
        """测试完成请求"""
        raw_gate = torch.ones(2, 4) * 0.5
        
        manager.apply_decay("request_1", raw_gate, layer_idx=0)
        manager.apply_decay("request_1", raw_gate, layer_idx=1)
        
        diagnostics = manager.finish_request("request_1")
        
        assert diagnostics is not None
        assert 'strategy' in diagnostics
        assert 'accumulated_gate' in diagnostics
        assert len(diagnostics['gate_history']) == 2
        
        # 请求应该被清理
        assert manager.finish_request("request_1") is None
    
    def test_clear_all(self, manager):
        """测试清理所有上下文"""
        manager.get_context("request_1")
        manager.get_context("request_2")
        
        manager.clear_all()
        
        # 应该创建新的上下文
        context = manager.get_context("request_1")
        assert context.accumulated_gate == 0.0


class TestCreateDecayConfig:
    """便捷函数测试"""
    
    def test_create_exponential(self):
        """测试创建指数衰减配置"""
        config = create_decay_config(strategy="exponential", gamma=0.85)
        
        assert config.strategy == DecayStrategy.EXPONENTIAL
        assert config.gamma == 0.85
    
    def test_create_linear(self):
        """测试创建线性衰减配置"""
        config = create_decay_config(strategy="linear", delta=0.15)
        
        assert config.strategy == DecayStrategy.LINEAR
        assert config.delta == 0.15


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

