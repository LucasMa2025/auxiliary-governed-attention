"""
Unit tests for FlashAttention-3 integration

测试 FlashAttention-3 集成模块:
- 后端检测
- 配置管理
- 注意力计算
- 知识注入优化
"""

import pytest
import torch
import torch.nn as nn

from aga.operator.flash_attention3 import (
    # 可用性检测
    FLASH_ATTN_AVAILABLE,
    FLASH_ATTN_VERSION,
    FLASH_ATTN_3_AVAILABLE,
    # 后端
    FA3Backend,
    FlashAttention3Backend,
    # 配置
    FA3Config,
    FA2Config,
    FlashAttention3Config,
    # 函数
    scaled_dot_product_attention_standard,
    flash_attention3_forward,
    # 模块
    AGAFlashAttention3,
    AGAKnowledgeInjectionOptimizer,
)


class TestFlashAttention3Backend:
    """测试后端检测"""
    
    def test_detect_best_backend(self):
        """测试自动检测最佳后端"""
        backend = FlashAttention3Backend.detect_best_backend()
        assert backend in ["fa3", "fa2", "standard"]
    
    def test_get_gpu_info(self):
        """测试获取 GPU 信息"""
        info = FlashAttention3Backend.get_gpu_info()
        assert "available" in info
        
        if info["available"]:
            assert "name" in info
            assert "compute_capability" in info
            assert "flash_attn_available" in info
            assert "recommended_backend" in info
    
    def test_validate_backend_auto(self):
        """测试验证后端 - auto"""
        backend = FlashAttention3Backend.validate_backend("auto")
        assert backend in ["fa3", "fa2", "standard"]
    
    def test_validate_backend_standard(self):
        """测试验证后端 - standard"""
        backend = FlashAttention3Backend.validate_backend("standard")
        assert backend == "standard"
    
    def test_supports_fa3(self):
        """测试 FA3 支持检测"""
        result = FlashAttention3Backend.supports_fa3()
        assert isinstance(result, bool)
    
    def test_supports_fa2(self):
        """测试 FA2 支持检测"""
        result = FlashAttention3Backend.supports_fa2()
        assert isinstance(result, bool)


class TestFlashAttention3Config:
    """测试配置管理"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FlashAttention3Config()
        assert config.enabled == True
        assert config.backend == "auto"
        assert config.fa3.use_fp8 == False
        assert config.fa2.causal == False
    
    def test_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "enabled": True,
            "backend": "fa2",
            "fa3": {
                "use_fp8": True,
                "enable_async": False,
            },
            "fa2": {
                "causal": True,
                "dropout": 0.1,
            },
        }
        config = FlashAttention3Config.from_dict(data)
        
        assert config.enabled == True
        assert config.backend == "fa2"
        assert config.fa3.use_fp8 == True
        assert config.fa3.enable_async == False
        assert config.fa2.causal == True
        assert config.fa2.dropout == 0.1
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = FlashAttention3Config(
            enabled=True,
            backend="fa3",
        )
        data = config.to_dict()
        
        assert data["enabled"] == True
        assert data["backend"] == "fa3"
        assert "fa3" in data
        assert "fa2" in data


class TestScaledDotProductAttention:
    """测试标准注意力计算"""
    
    def test_basic_attention(self):
        """测试基本注意力计算"""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = scaled_dot_product_attention_standard(query, key, value)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_attention_with_mask(self):
        """测试带掩码的注意力计算"""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # 创建 causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        output = scaled_dot_product_attention_standard(
            query, key, value, attn_mask=mask
        )
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestFlashAttention3Forward:
    """测试统一前向接口"""
    
    def test_forward_standard_backend(self):
        """测试标准后端前向"""
        batch_size = 2
        seq_len = 8
        num_heads = 4
        head_dim = 16
        
        # FlashAttention 格式: [batch, seq, heads, head_dim]
        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        output = flash_attention3_forward(
            query, key, value,
            backend="standard",
        )
        
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    
    def test_forward_auto_backend(self):
        """测试自动后端前向"""
        batch_size = 2
        seq_len = 8
        num_heads = 4
        head_dim = 16
        
        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        output = flash_attention3_forward(
            query, key, value,
            backend="auto",
        )
        
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)


class TestAGAFlashAttention3:
    """测试 AGA 专用 FlashAttention-3 模块"""
    
    @pytest.fixture
    def module(self):
        """创建测试模块"""
        return AGAFlashAttention3(
            hidden_dim=64,
            bottleneck_dim=16,
            num_slots=10,
            top_k=3,
            config=FlashAttention3Config(backend="standard"),
        )
    
    def test_forward(self, module):
        """测试前向传播"""
        batch_size = 2
        seq_len = 8
        
        query = torch.randn(batch_size, seq_len, 16)  # bottleneck_dim
        keys = torch.randn(10, 16)  # num_slots, bottleneck_dim
        values = torch.randn(10, 64)  # num_slots, hidden_dim
        reliability = torch.ones(10)
        
        output, attn_weights = module(query, keys, values, reliability)
        
        assert output.shape == (batch_size, seq_len, 64)
        assert attn_weights.shape == (batch_size, seq_len, 3)  # top_k
    
    def test_forward_with_precomputed_indices(self, module):
        """测试带预计算索引的前向传播"""
        batch_size = 2
        seq_len = 8
        
        query = torch.randn(batch_size, seq_len, 16)
        keys = torch.randn(10, 16)
        values = torch.randn(10, 64)
        reliability = torch.ones(10)
        
        # 预计算 top-k 索引
        top_indices = torch.randint(0, 10, (batch_size, seq_len, 3))
        
        output, attn_weights = module(
            query, keys, values, reliability,
            top_indices=top_indices,
        )
        
        assert output.shape == (batch_size, seq_len, 64)
    
    def test_get_stats(self, module):
        """测试获取统计信息"""
        # 执行一次前向
        query = torch.randn(1, 4, 16)
        keys = torch.randn(10, 16)
        values = torch.randn(10, 64)
        reliability = torch.ones(10)
        
        module(query, keys, values, reliability)
        
        stats = module.get_stats()
        
        assert stats["forward_count"] == 1
        assert "backend" in stats
        assert "gpu_info" in stats
    
    def test_reset_stats(self, module):
        """测试重置统计信息"""
        # 执行一次前向
        query = torch.randn(1, 4, 16)
        keys = torch.randn(10, 16)
        values = torch.randn(10, 64)
        reliability = torch.ones(10)
        
        module(query, keys, values, reliability)
        
        # 重置
        module.reset_stats()
        
        stats = module.get_stats()
        assert stats["forward_count"] == 0


class TestAGAKnowledgeInjectionOptimizer:
    """测试 AGA 知识注入优化器"""
    
    @pytest.fixture
    def optimizer(self):
        """创建测试优化器"""
        return AGAKnowledgeInjectionOptimizer(
            config=FlashAttention3Config(backend="standard"),
        )
    
    def test_inject(self, optimizer):
        """测试知识注入"""
        batch_size = 2
        seq_len = 8
        hidden_dim = 64
        num_slots = 10
        bottleneck_dim = 16
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        knowledge_keys = torch.randn(num_slots, hidden_dim)
        knowledge_values = torch.randn(num_slots, hidden_dim)
        
        injected, metadata = optimizer.inject(
            hidden_states,
            knowledge_keys,
            knowledge_values,
            alpha=0.5,
        )
        
        assert injected.shape == (batch_size, seq_len, hidden_dim)
        assert metadata["alpha"] == 0.5
        assert "latency_ms" in metadata
    
    def test_inject_with_reliability(self, optimizer):
        """测试带可靠性权重的知识注入"""
        batch_size = 2
        seq_len = 8
        hidden_dim = 64
        num_slots = 10
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        knowledge_keys = torch.randn(num_slots, hidden_dim)
        knowledge_values = torch.randn(num_slots, hidden_dim)
        reliability = torch.rand(num_slots)
        
        injected, metadata = optimizer.inject(
            hidden_states,
            knowledge_keys,
            knowledge_values,
            alpha=0.7,
            reliability=reliability,
        )
        
        assert injected.shape == (batch_size, seq_len, hidden_dim)
    
    def test_get_stats(self, optimizer):
        """测试获取统计信息"""
        # 执行一次注入
        hidden_states = torch.randn(1, 4, 64)
        knowledge_keys = torch.randn(10, 64)
        knowledge_values = torch.randn(10, 64)
        
        optimizer.inject(hidden_states, knowledge_keys, knowledge_values)
        
        stats = optimizer.get_stats()
        
        assert stats["total_injections"] == 1
        assert "backend" in stats
        assert "avg_latency_ms" in stats


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流水线"""
        # 配置
        config = FlashAttention3Config(
            enabled=True,
            backend="standard",
        )
        
        # 创建模块
        attention = AGAFlashAttention3(
            hidden_dim=64,
            bottleneck_dim=16,
            num_slots=10,
            top_k=3,
            config=config,
        )
        
        optimizer = AGAKnowledgeInjectionOptimizer(config=config)
        
        # 模拟数据
        batch_size = 2
        seq_len = 8
        
        query = torch.randn(batch_size, seq_len, 16)
        keys = torch.randn(10, 16)
        values = torch.randn(10, 64)
        reliability = torch.ones(10)
        
        # 执行注意力
        output, attn_weights = attention(query, keys, values, reliability)
        
        # 执行注入
        hidden_states = torch.randn(batch_size, seq_len, 64)
        knowledge_keys = torch.randn(10, 64)
        knowledge_values = torch.randn(10, 64)
        
        injected, metadata = optimizer.inject(
            hidden_states,
            knowledge_keys,
            knowledge_values,
            alpha=0.5,
        )
        
        # 验证
        assert output.shape == (batch_size, seq_len, 64)
        assert injected.shape == (batch_size, seq_len, 64)
        
        # 检查统计
        attn_stats = attention.get_stats()
        opt_stats = optimizer.get_stats()
        
        assert attn_stats["forward_count"] == 1
        assert opt_stats["total_injections"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
