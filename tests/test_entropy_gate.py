"""
熵门控模块单元测试

测试覆盖：
- 熵计算
- 门控公式
- 熵否决机制
- 自适应阈值
"""
import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '..')

from aga.entropy_gate import (
    EntropyGate,
    EntropyGateConfig,
    EntropyCalculator,
    EntropySource,
    EntropyGateWithDecay,
    create_entropy_gate,
)


class TestEntropyGateConfig:
    """熵门控配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EntropyGateConfig()
        
        assert config.tau_low == 0.5
        assert config.tau_high == 2.0
        assert config.max_gate == 0.8
        assert config.w1_init == 0.5
        assert config.w2_init == 0.3
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EntropyGateConfig(
            tau_low=0.3,
            tau_high=3.0,
            entropy_source=EntropySource.LOGITS,
        )
        
        assert config.tau_low == 0.3
        assert config.tau_high == 3.0
        assert config.entropy_source == EntropySource.LOGITS


class TestEntropyCalculator:
    """熵计算器测试"""
    
    @pytest.fixture
    def hidden_states(self):
        """创建测试隐藏状态"""
        return torch.randn(2, 8, 256)
    
    def test_hidden_variance_entropy(self, hidden_states):
        """测试隐藏状态方差熵"""
        config = EntropyGateConfig(entropy_source=EntropySource.HIDDEN_VARIANCE)
        calculator = EntropyCalculator(config, hidden_dim=256)
        
        entropy = calculator(hidden_states)
        
        assert entropy.shape == (2, 8)
        assert (entropy >= 0).all()
        assert (entropy <= config.entropy_clamp_max).all()
    
    def test_attention_entropy(self, hidden_states):
        """测试注意力熵"""
        config = EntropyGateConfig(entropy_source=EntropySource.ATTENTION)
        calculator = EntropyCalculator(config, hidden_dim=256)
        
        # 创建注意力权重
        attention_weights = F.softmax(torch.randn(2, 8, 8, 8), dim=-1)
        
        entropy = calculator(hidden_states, attention_weights=attention_weights)
        
        assert entropy.shape == (2, 8)
        assert (entropy >= 0).all()
    
    def test_logits_entropy(self, hidden_states):
        """测试 logits 熵"""
        config = EntropyGateConfig(entropy_source=EntropySource.LOGITS)
        calculator = EntropyCalculator(config, hidden_dim=256)
        
        # 创建 logits
        logits = torch.randn(2, 8, 1000)
        
        entropy = calculator(hidden_states, logits=logits)
        
        assert entropy.shape == (2, 8)
        assert (entropy >= 0).all()
    
    def test_fallback_to_hidden_variance(self, hidden_states):
        """测试回退到隐藏状态方差"""
        config = EntropyGateConfig(entropy_source=EntropySource.ATTENTION)
        calculator = EntropyCalculator(config, hidden_dim=256)
        
        # 不提供注意力权重，应该回退
        entropy = calculator(hidden_states, attention_weights=None)
        
        assert entropy.shape == (2, 8)
    
    def test_ensemble_entropy(self, hidden_states):
        """测试集成熵"""
        config = EntropyGateConfig(entropy_source=EntropySource.ENSEMBLE)
        calculator = EntropyCalculator(config, hidden_dim=256)
        
        attention_weights = F.softmax(torch.randn(2, 8, 8, 8), dim=-1)
        logits = torch.randn(2, 8, 1000)
        
        entropy = calculator(hidden_states, attention_weights, logits)
        
        assert entropy.shape == (2, 8)


class TestEntropyGate:
    """熵门控测试"""
    
    @pytest.fixture
    def gate(self):
        """创建熵门控实例"""
        config = EntropyGateConfig(
            tau_low=0.5,
            tau_high=2.0,
            max_gate=0.8,
        )
        return EntropyGate(config, hidden_dim=256)
    
    @pytest.fixture
    def hidden_states(self):
        """创建测试隐藏状态"""
        return torch.randn(2, 8, 256)
    
    def test_forward_basic(self, gate, hidden_states):
        """测试基本前向传播"""
        reliability = torch.ones(2, 8) * 0.8
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        assert gate_values.shape == (2, 8)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()
        assert 'entropy_mean' in diagnostics
        assert 'final_gate_mean' in diagnostics
    
    def test_forward_with_scalar_reliability(self, gate, hidden_states):
        """测试标量可靠性"""
        reliability = torch.tensor(0.8)
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        assert gate_values.shape == (2, 8)
    
    def test_low_entropy_veto(self, gate):
        """测试低熵否决"""
        # 创建低方差（低熵）的隐藏状态
        hidden_states = torch.ones(2, 8, 256) * 0.5 + torch.randn(2, 8, 256) * 0.01
        reliability = torch.ones(2, 8)
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        # 低熵时应该有较多的 veto
        assert diagnostics['veto_ratio'] > 0
    
    def test_high_entropy_cap(self, gate):
        """测试高熵限制"""
        # 创建高方差（高熵）的隐藏状态
        hidden_states = torch.randn(2, 8, 256) * 10
        reliability = torch.ones(2, 8)
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        # 高熵时 gate 应该被限制
        assert (gate_values <= 0.8).all()
    
    def test_diagnostics_content(self, gate, hidden_states):
        """测试诊断信息内容"""
        reliability = torch.ones(2, 8) * 0.8
        
        _, diagnostics = gate(hidden_states, reliability)
        
        required_keys = [
            'entropy_mean', 'entropy_std', 'raw_gate_mean',
            'final_gate_mean', 'tau_low', 'tau_high',
            'w1', 'w2', 'bias', 'veto_ratio'
        ]
        
        for key in required_keys:
            assert key in diagnostics


class TestEntropyGateWithDecay:
    """集成衰减的熵门控测试"""
    
    @pytest.fixture
    def gate_with_decay(self):
        """创建带衰减的熵门控"""
        gate_config = EntropyGateConfig()
        return EntropyGateWithDecay(gate_config, hidden_dim=256)
    
    def test_forward_with_decay(self, gate_with_decay):
        """测试带衰减的前向传播"""
        hidden_states = torch.randn(2, 8, 256)
        reliability = torch.ones(2, 8) * 0.8
        
        # 第一层
        gate_values, context, diagnostics = gate_with_decay(
            hidden_states, reliability, layer_idx=0
        )
        
        assert gate_values.shape == (2, 8)
        assert context is not None
        assert 'decay_accumulated' in diagnostics
        
        # 第二层（应该有衰减）
        gate_values_2, context_2, diagnostics_2 = gate_with_decay(
            hidden_states, reliability, layer_idx=1, decay_context=context
        )
        
        assert context_2.layer_idx == 1


class TestCreateEntropyGate:
    """便捷函数测试"""
    
    def test_create_default(self):
        """测试创建默认熵门控"""
        gate = create_entropy_gate(hidden_dim=256)
        
        assert isinstance(gate, EntropyGate)
        assert gate.config.entropy_source == EntropySource.HIDDEN_VARIANCE
    
    def test_create_with_options(self):
        """测试创建自定义熵门控"""
        gate = create_entropy_gate(
            hidden_dim=512,
            entropy_source="logits",
            tau_low=0.3,
            tau_high=3.0,
        )
        
        assert gate.config.entropy_source == EntropySource.LOGITS
        assert gate.config.tau_low == 0.3
        assert gate.config.tau_high == 3.0


class TestNumericalStability:
    """数值稳定性测试"""
    
    def test_zero_variance_hidden_states(self):
        """测试零方差隐藏状态"""
        config = EntropyGateConfig(entropy_source=EntropySource.HIDDEN_VARIANCE)
        gate = EntropyGate(config, hidden_dim=256)
        
        # 创建常数隐藏状态（零方差）
        hidden_states = torch.ones(2, 8, 256)
        reliability = torch.ones(2, 8)
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        # 不应该有 NaN
        assert not torch.isnan(gate_values).any()
        assert not torch.isinf(gate_values).any()
    
    def test_extreme_values(self):
        """测试极端值"""
        config = EntropyGateConfig()
        gate = EntropyGate(config, hidden_dim=256)
        
        # 创建极端值隐藏状态
        hidden_states = torch.randn(2, 8, 256) * 1000
        reliability = torch.ones(2, 8)
        
        gate_values, diagnostics = gate(hidden_states, reliability)
        
        assert not torch.isnan(gate_values).any()
        assert not torch.isinf(gate_values).any()
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

