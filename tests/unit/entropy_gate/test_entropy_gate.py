"""
熵门控单元测试

测试熵门控机制的核心功能。
"""
import pytest
import torch
import math

from aga import EntropyGate, UncertaintySource
from aga.entropy_gate import EntropyGateConfig


@pytest.mark.unit
class TestEntropyComputation:
    """熵计算测试"""
    
    def test_uniform_distribution_entropy(self, device):
        """测试均匀分布的熵"""
        # 均匀分布应该有最大熵
        uniform = torch.ones(1, 10, device=device) / 10
        
        # 手动计算熵
        entropy = -(uniform * torch.log(uniform + 1e-10)).sum(dim=-1)
        
        expected = math.log(10)  # 最大熵
        assert abs(entropy.item() - expected) < 0.01
    
    def test_peaked_distribution_entropy(self, device):
        """测试尖峰分布的熵"""
        # 尖峰分布应该有低熵
        peaked = torch.zeros(1, 10, device=device)
        peaked[0, 0] = 1.0
        
        # 手动计算熵
        entropy = -(peaked * torch.log(peaked + 1e-10)).sum(dim=-1)
        
        assert entropy.item() < 0.1  # 接近 0


@pytest.mark.unit
class TestEntropyGate:
    """EntropyGate 模块测试"""
    
    @pytest.fixture
    def entropy_gate(self, hidden_dim, device):
        """创建熵门控模块"""
        config = EntropyGateConfig()
        gate = EntropyGate(config, hidden_dim=hidden_dim)
        gate.to(device)
        gate.eval()
        return gate
    
    def test_initialization(self, entropy_gate):
        """测试初始化"""
        assert entropy_gate.config is not None
    
    def test_forward_hidden_variance(self, entropy_gate, random_hidden_states, device):
        """测试基于隐藏状态方差的前向传播"""
        batch_size, seq_len, _ = random_hidden_states.shape
        reliability = torch.ones(batch_size, seq_len, device=device)
        
        gate_value, diagnostics = entropy_gate(
            hidden_states=random_hidden_states,
            reliability=reliability,
        )
        
        assert gate_value.shape == (batch_size, seq_len)
        
        # 门控值应该在有效范围内
        assert (gate_value >= 0).all()
        assert (gate_value <= 1).all()


@pytest.mark.unit
class TestEntropyGateEdgeCases:
    """熵门控边界情况测试"""
    
    @pytest.fixture
    def entropy_gate(self, hidden_dim, device):
        config = EntropyGateConfig()
        gate = EntropyGate(config, hidden_dim=hidden_dim)
        gate.to(device)
        gate.eval()
        return gate
    
    def test_zero_variance_input(self, entropy_gate, hidden_dim, device):
        """测试零方差输入"""
        constant_input = torch.ones(1, 1, hidden_dim, device=device)
        reliability = torch.ones(1, 1, device=device)
        
        gate_value, diagnostics = entropy_gate(
            hidden_states=constant_input,
            reliability=reliability,
        )
        
        # 应该不会产生 NaN
        assert torch.isfinite(gate_value).all()
    
    def test_extreme_values(self, entropy_gate, hidden_dim, device):
        """测试极端值"""
        extreme_input = torch.randn(1, 1, hidden_dim, device=device) * 1000
        reliability = torch.ones(1, 1, device=device)
        
        gate_value, diagnostics = entropy_gate(
            hidden_states=extreme_input,
            reliability=reliability,
        )
        
        assert torch.isfinite(gate_value).all()
    
    def test_single_token(self, entropy_gate, hidden_dim, device):
        """测试单个 token"""
        single_token = torch.randn(1, 1, hidden_dim, device=device)
        reliability = torch.ones(1, 1, device=device)
        
        gate_value, diagnostics = entropy_gate(
            hidden_states=single_token,
            reliability=reliability,
        )
        
        assert gate_value.shape == (1, 1)
