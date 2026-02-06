"""
持久化衰减单元测试

测试持久化衰减机制的核心功能。
"""
import pytest
import torch

from aga.decay import PersistenceDecay, DecayConfig, DecayContext, DecayStrategy


@pytest.mark.unit
class TestPersistenceDecay:
    """持久化衰减测试"""
    
    @pytest.fixture
    def decay_module(self, decay_config, device):
        """创建衰减模块"""
        decay = PersistenceDecay(decay_config)
        decay.to(device)
        return decay
    
    def test_initialization(self, decay_module, decay_config):
        """测试初始化"""
        assert decay_module.config.gamma == decay_config.gamma
    
    def test_decay_application(self, decay_module, device):
        """测试衰减应用"""
        gate = torch.ones(2, 8, device=device) * 0.5
        context = DecayContext()
        
        # 第一层不衰减
        effective_gate, context = decay_module(gate, context, layer_idx=0)
        assert torch.allclose(effective_gate, gate)
        
        # 第二层应该衰减
        effective_gate, context = decay_module(gate, context, layer_idx=1)
        assert (effective_gate <= gate).all()
        assert (effective_gate >= 0).all()
    
    def test_decay_increases_with_layer(self, device):
        """测试衰减随层数增加"""
        # 使用禁用硬重置的配置
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.9,
            enable_hard_reset=False,  # 禁用硬重置以测试纯衰减
        )
        decay_module = PersistenceDecay(config).to(device)
        
        gate = torch.ones(1, 1, device=device) * 0.5  # 使用较小的值避免累积过大
        context = DecayContext()
        
        gates = []
        for layer_idx in range(5):
            effective_gate, context = decay_module(gate, context, layer_idx)
            gates.append(effective_gate.item())
        
        # 后面层的衰减应该更强（门控值更小）
        for i in range(1, len(gates)):
            assert gates[i] <= gates[i-1] + 0.01  # 允许小误差
    
    def test_batch_decay(self, decay_module, device):
        """测试批量衰减"""
        batch_gate = torch.rand(4, 16, device=device)
        context = DecayContext()
        
        # 第一层
        _, context = decay_module(batch_gate, context, layer_idx=0)
        # 第二层
        decayed, _ = decay_module(batch_gate, context, layer_idx=1)
        
        assert decayed.shape == batch_gate.shape


@pytest.mark.unit
class TestDecayConfig:
    """衰减配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DecayConfig()
        
        assert config.strategy == DecayStrategy.EXPONENTIAL
        assert 0 < config.gamma < 1
        assert config.hard_reset_threshold >= 0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DecayConfig(
            strategy=DecayStrategy.EXPONENTIAL,
            gamma=0.9,
            enable_hard_reset=True,
            hard_reset_threshold=3.0,
        )
        
        assert config.gamma == 0.9
        assert config.enable_hard_reset is True
        assert config.hard_reset_threshold == 3.0


@pytest.mark.unit
class TestDecayEdgeCases:
    """衰减边界情况测试"""
    
    @pytest.fixture
    def decay_module(self, decay_config, device):
        decay = PersistenceDecay(decay_config)
        decay.to(device)
        return decay
    
    def test_zero_gate(self, decay_module, device):
        """测试零门控值"""
        zero_gate = torch.zeros(1, 1, device=device)
        context = DecayContext()
        
        effective_gate, _ = decay_module(zero_gate, context, layer_idx=0)
        
        assert effective_gate.item() == 0.0
    
    def test_one_gate(self, decay_module, device):
        """测试门控值为 1"""
        one_gate = torch.ones(1, 1, device=device)
        context = DecayContext()
        
        # 第一层不衰减
        _, context = decay_module(one_gate, context, layer_idx=0)
        # 第二层应该衰减
        decayed, _ = decay_module(one_gate, context, layer_idx=1)
        
        # 应该被衰减
        assert decayed.item() < 1.0
    
    def test_numerical_stability(self, decay_module, device):
        """测试数值稳定性"""
        # 极小值
        tiny_gate = torch.tensor([[1e-10]], device=device)
        context = DecayContext()
        
        decayed, _ = decay_module(tiny_gate, context, layer_idx=0)
        
        assert torch.isfinite(decayed).all()
        assert decayed.item() >= 0


@pytest.mark.unit
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
