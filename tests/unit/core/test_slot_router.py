"""
SlotRouter 单元测试

测试槽位路由的核心功能。
"""
import pytest
import torch
import math

from aga import AGAConfig
from aga.core import SlotRouter


@pytest.mark.unit
class TestSlotRouter:
    """SlotRouter 测试"""
    
    @pytest.fixture
    def router(self, aga_config, device):
        """创建路由器"""
        router = SlotRouter(
            hidden_dim=aga_config.hidden_dim,
            bottleneck_dim=aga_config.bottleneck_dim,
            num_slots=aga_config.num_slots,
            top_k=aga_config.top_k_routing,
        )
        return router.to(device)
    
    @pytest.fixture
    def sample_keys(self, num_slots, bottleneck_dim, device):
        """示例 key 向量"""
        return torch.randn(num_slots, bottleneck_dim, device=device)
    
    @pytest.fixture
    def sample_query(self, bottleneck_dim, device):
        """示例查询向量"""
        batch_size = 2
        seq_len = 8
        return torch.randn(batch_size, seq_len, bottleneck_dim, device=device)
    
    @pytest.fixture
    def reliability_mask(self, num_slots, device):
        """可靠性掩码"""
        return torch.zeros(num_slots, device=device)  # log(1.0) = 0
    
    def test_router_initialization(self, router, aga_config):
        """测试路由器初始化"""
        assert router.top_k == aga_config.top_k_routing
        assert router.bottleneck_dim == aga_config.bottleneck_dim
        assert router.num_slots == aga_config.num_slots
    
    def test_forward_routing(self, router, sample_query, sample_keys, reliability_mask):
        """测试前向路由"""
        top_indices, top_scores = router(sample_query, sample_keys, reliability_mask)
        
        batch_size, seq_len, _ = sample_query.shape
        k = router.top_k
        
        assert top_indices.shape == (batch_size, seq_len, k)
        assert top_scores.shape == (batch_size, seq_len, k)
        
        # 索引应该在有效范围内
        assert (top_indices >= 0).all()
        assert (top_indices < sample_keys.shape[0]).all()
        
        # 分数应该是有限的
        assert torch.isfinite(top_scores).all()
    
    def test_top_k_selection(self, router, sample_query, sample_keys, reliability_mask):
        """测试 Top-k 选择"""
        top_indices, top_scores = router(sample_query, sample_keys, reliability_mask)
        
        # 验证返回的是 top-k 个槽位
        assert top_indices.shape[-1] == router.top_k
        
        # 验证分数是降序排列的
        for b in range(top_scores.shape[0]):
            for s in range(top_scores.shape[1]):
                scores = top_scores[b, s]
                for i in range(len(scores) - 1):
                    assert scores[i] >= scores[i + 1]


@pytest.mark.unit
class TestSlotRouterEdgeCases:
    """SlotRouter 边界情况测试"""
    
    @pytest.fixture
    def router(self, aga_config, device):
        router = SlotRouter(
            hidden_dim=aga_config.hidden_dim,
            bottleneck_dim=aga_config.bottleneck_dim,
            num_slots=aga_config.num_slots,
            top_k=aga_config.top_k_routing,
        )
        return router.to(device)
    
    def test_single_slot(self, device, hidden_dim, bottleneck_dim):
        """测试单个槽位"""
        router = SlotRouter(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=1,
            top_k=1,
        ).to(device)
        
        single_key = torch.randn(1, bottleneck_dim, device=device)
        query = torch.randn(1, 1, bottleneck_dim, device=device)
        reliability_mask = torch.zeros(1, device=device)
        
        top_indices, top_scores = router(query, single_key, reliability_mask)
        
        assert top_indices.shape == (1, 1, 1)
        assert top_scores.shape == (1, 1, 1)
        assert top_indices[0, 0, 0] == 0  # 唯一的槽位
    
    def test_numerical_stability(self, router, device, bottleneck_dim, num_slots):
        """测试数值稳定性"""
        # 极端值
        extreme_keys = torch.randn(num_slots, bottleneck_dim, device=device) * 1000
        query = torch.randn(1, 1, bottleneck_dim, device=device) * 1000
        reliability_mask = torch.zeros(num_slots, device=device)
        
        top_indices, top_scores = router(query, extreme_keys, reliability_mask)
        
        assert torch.isfinite(top_scores).all()
        assert (top_indices >= 0).all()
        assert (top_indices < num_slots).all()
    
    def test_reliability_mask_effect(self, device, hidden_dim, bottleneck_dim):
        """测试可靠性掩码的影响"""
        num_slots = 10
        router = SlotRouter(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots=num_slots,
            top_k=3,
        ).to(device)
        
        keys = torch.randn(num_slots, bottleneck_dim, device=device)
        query = torch.randn(1, 1, bottleneck_dim, device=device)
        
        # 创建掩码，使某些槽位的可靠性很低
        reliability_mask = torch.zeros(num_slots, device=device)
        reliability_mask[0:3] = -1000  # 这些槽位应该不会被选中
        
        top_indices, _ = router(query, keys, reliability_mask)
        
        # 验证低可靠性的槽位不会被选中
        for idx in top_indices[0, 0]:
            assert idx.item() >= 3
