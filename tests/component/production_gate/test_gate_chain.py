"""
三段式门控组件测试

测试生产级门控链的功能。
"""
import pytest
import torch

from aga.production.config import GateConfig, UncertaintySource

# 尝试导入门控模块
try:
    from aga.production.gate import (
        GateChain, Gate0, Gate1, Gate2, 
        GateResult, GateContext, GateDiagnostics
    )
    HAS_GATE = True
except ImportError:
    HAS_GATE = False
    GateChain = None
    Gate0 = None
    Gate1 = None
    Gate2 = None
    GateResult = None
    GateContext = None


@pytest.mark.component
@pytest.mark.skipif(not HAS_GATE, reason="Gate module not available")
class TestGateChain:
    """门控链测试"""
    
    @pytest.fixture
    def gate_config(self):
        """门控配置"""
        return GateConfig(
            gate0_enabled=True,
            gate1_enabled=True,
            gate1_threshold=0.1,
            gate2_top_k=4,
            early_exit_threshold=0.05,
            early_exit_enabled=True,
        )
    
    @pytest.fixture
    def gate_chain(self, gate_config, hidden_dim, bottleneck_dim, num_slots, device):
        """创建门控链"""
        chain = GateChain(gate_config, hidden_dim, bottleneck_dim, num_slots)
        chain.to(device)
        chain.eval()
        return chain
    
    def test_initialization(self, gate_chain, gate_config):
        """测试初始化"""
        assert gate_chain.gate0 is not None
        assert gate_chain.gate1 is not None
        assert gate_chain.gate2 is not None
    
    def test_forward(self, gate_chain, random_hidden_states, bottleneck_dim, num_slots, device):
        """测试前向传播"""
        context = GateContext(namespace="default")
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        
        indices, scores, diagnostics = gate_chain(
            context=context,
            hidden_states=random_hidden_states,
            slot_keys=keys,
        )
        
        assert isinstance(diagnostics, GateDiagnostics)
    
    def test_gate0_disabled_namespace(self, hidden_dim, bottleneck_dim, num_slots, device, random_hidden_states):
        """测试 Gate-0 禁用命名空间"""
        config = GateConfig(
            gate0_enabled=True,
            gate0_disabled_namespaces=["disabled_ns"],
        )
        chain = GateChain(config, hidden_dim, bottleneck_dim, num_slots)
        chain.to(device)
        chain.eval()
        
        context = GateContext(namespace="disabled_ns")
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        
        indices, scores, diagnostics = chain(
            context=context,
            hidden_states=random_hidden_states,
            slot_keys=keys,
        )
        
        # 禁用的命名空间应该直接旁路
        assert diagnostics.gate0_result == GateResult.DISABLED
    
    def test_gate0_required_namespace(self, hidden_dim, bottleneck_dim, num_slots, device, random_hidden_states):
        """测试 Gate-0 强制启用命名空间"""
        config = GateConfig(
            gate0_enabled=True,
            gate0_required_namespaces=["required_ns"],
        )
        chain = GateChain(config, hidden_dim, bottleneck_dim, num_slots)
        chain.to(device)
        chain.eval()
        
        context = GateContext(namespace="required_ns")
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        
        indices, scores, diagnostics = chain(
            context=context,
            hidden_states=random_hidden_states,
            slot_keys=keys,
        )
        
        # 强制启用的命名空间应该通过 Gate-0
        assert diagnostics.gate0_result == GateResult.REQUIRED
    
    def test_early_exit(self, hidden_dim, bottleneck_dim, num_slots, device):
        """测试提前退出"""
        config = GateConfig(
            gate1_enabled=True,
            early_exit_enabled=True,
            early_exit_threshold=0.9,  # 高阈值，容易触发
        )
        chain = GateChain(config, hidden_dim, bottleneck_dim, num_slots)
        chain.to(device)
        chain.eval()
        
        # 低方差输入（高置信度）
        low_variance_input = torch.ones(1, 1, hidden_dim, device=device)
        context = GateContext(namespace="default")
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        
        indices, scores, diagnostics = chain(
            context=context,
            hidden_states=low_variance_input,
            slot_keys=keys,
        )
        
        # 可能触发提前退出
        assert diagnostics is not None


@pytest.mark.component
@pytest.mark.skipif(not HAS_GATE, reason="Gate module not available")
class TestGate0:
    """Gate-0 测试"""
    
    @pytest.fixture
    def gate0(self):
        """创建 Gate-0"""
        config = GateConfig(
            gate0_enabled=True,
            gate0_disabled_namespaces=["disabled"],
            gate0_required_namespaces=["required"],
        )
        return Gate0(config)
    
    def test_normal_namespace(self, gate0):
        """测试普通命名空间"""
        context = GateContext(namespace="normal")
        result = gate0(context)
        assert result == GateResult.POSSIBLE
    
    def test_disabled_namespace(self, gate0):
        """测试禁用命名空间"""
        context = GateContext(namespace="disabled")
        result = gate0(context)
        assert result == GateResult.DISABLED
    
    def test_required_namespace(self, gate0):
        """测试强制命名空间"""
        context = GateContext(namespace="required")
        result = gate0(context)
        assert result == GateResult.REQUIRED


@pytest.mark.component
@pytest.mark.skipif(not HAS_GATE, reason="Gate module not available")
class TestGate1:
    """Gate-1 测试"""
    
    @pytest.fixture
    def gate1(self, hidden_dim, device):
        """创建 Gate-1"""
        config = GateConfig(
            gate1_enabled=True,
            gate1_threshold=0.1,
        )
        gate = Gate1(config, hidden_dim)
        gate.to(device)
        gate.eval()
        return gate
    
    def test_high_confidence(self, gate1, hidden_dim, device):
        """测试高置信度输入"""
        # 低方差输入
        low_variance = torch.ones(1, 1, hidden_dim, device=device)
        
        result, confidence = gate1(low_variance)
        
        # 高置信度应该旁路
        assert result in [GateResult.BYPASS, GateResult.PASS]
    
    def test_low_confidence(self, gate1, hidden_dim, device):
        """测试低置信度输入"""
        # 高方差输入
        high_variance = torch.randn(1, 1, hidden_dim, device=device) * 10
        
        result, confidence = gate1(high_variance)
        
        # 低置信度应该通过
        assert result == GateResult.PASS


@pytest.mark.component
@pytest.mark.skipif(not HAS_GATE, reason="Gate module not available")
class TestGate2:
    """Gate-2 测试"""
    
    @pytest.fixture
    def gate2(self, bottleneck_dim, num_slots, device):
        """创建 Gate-2"""
        config = GateConfig(
            gate2_top_k=4,
            gate2_chunk_size=64,
        )
        gate = Gate2(config, bottleneck_dim, num_slots)
        gate.to(device)
        return gate
    
    def test_top_k_routing(self, gate2, bottleneck_dim, num_slots, device):
        """测试 Top-k 路由"""
        query = torch.randn(1, 8, bottleneck_dim, device=device)
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        
        indices, scores = gate2(query, keys)
        
        assert indices.shape[-1] == min(4, num_slots)  # top_k
        assert scores.shape[-1] == min(4, num_slots)
    
    def test_fewer_slots_than_k(self, bottleneck_dim, device):
        """测试槽位数少于 k"""
        config = GateConfig(
            gate2_top_k=4,
            gate2_chunk_size=64,
        )
        gate = Gate2(config, bottleneck_dim, 2)  # 只有 2 个槽位
        gate.to(device)
        
        query = torch.randn(1, 1, bottleneck_dim, device=device)
        keys = torch.randn(2, bottleneck_dim, device=device)
        
        indices, scores = gate(query, keys)
        
        # 应该返回所有槽位
        assert indices.shape[-1] == 2
