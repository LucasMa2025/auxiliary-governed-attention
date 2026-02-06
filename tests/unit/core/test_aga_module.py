"""
AGA 核心模块单元测试

测试 AGA 模块的核心功能。
"""
import pytest
import torch

from aga import AGAConfig, LifecycleState
from aga.core import AuxiliaryGovernedAttention


@pytest.mark.unit
class TestAGAModule:
    """AGA 模块测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        """创建 AGA 模块"""
        module = AuxiliaryGovernedAttention(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_initialization(self, aga_module, aga_config):
        """测试初始化"""
        assert aga_module.hidden_dim == aga_config.hidden_dim
        assert aga_module.bottleneck_dim == aga_config.bottleneck_dim
        assert aga_module.num_slots == aga_config.num_slots
    
    def test_forward_without_slots(self, aga_module, random_hidden_states):
        """测试无槽位时的前向传播"""
        primary_output = random_hidden_states.clone()
        
        output, diagnostics = aga_module(
            hidden_states=random_hidden_states,
            primary_attention_output=primary_output,
        )
        
        # 输出形状应该与输入相同
        assert output.shape == random_hidden_states.shape
        # 无槽位时应该返回原始输入
        assert torch.allclose(output, primary_output, atol=1e-5)
    
    def test_inject_knowledge(self, aga_module, random_key_vector, random_value_vector):
        """测试知识注入"""
        success = aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        assert success is True
        assert aga_module.get_active_slots() == 1
        
        # 检查槽位信息
        slot_info = aga_module.get_slot_info(0)
        assert slot_info is not None
        assert slot_info.lu_id == "test_lu_001"
        assert slot_info.lifecycle_state == LifecycleState.PROBATIONARY
    
    def test_forward_with_slots(
        self,
        aga_module,
        random_hidden_states,
        random_key_vector,
        random_value_vector,
    ):
        """测试有槽位时的前向传播"""
        # 注入知识
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        primary_output = random_hidden_states.clone()
        
        output, diagnostics = aga_module(
            hidden_states=random_hidden_states,
            primary_attention_output=primary_output,
            return_diagnostics=True,
        )
        
        assert output.shape == random_hidden_states.shape
        assert diagnostics is not None
        assert diagnostics.active_slots == 1
    
    def test_lifecycle_update(self, aga_module, random_key_vector, random_value_vector):
        """测试生命周期更新"""
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        # 更新状态 - 使用 confirm_slot 方法
        aga_module.confirm_slot(0)
        
        slot_info = aga_module.get_slot_info(0)
        assert slot_info.lifecycle_state == LifecycleState.CONFIRMED
        assert slot_info.reliability == 1.0
    
    def test_quarantine_slot(self, aga_module, random_key_vector, random_value_vector):
        """测试槽位隔离"""
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # 隔离
        aga_module.quarantine_slot(0)
        
        slot_info = aga_module.get_slot_info(0)
        assert slot_info.lifecycle_state == LifecycleState.QUARANTINED
        assert slot_info.reliability == 0.0
    
    def test_clear_slot(self, aga_module, random_key_vector, random_value_vector):
        """测试槽位清除"""
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
        )
        
        assert aga_module.get_active_slots() == 1
        
        # 清除槽位 - 使用 quarantine_slot
        aga_module.quarantine_slot(0)
        
        # 隔离后活跃槽位数应该为 0
        assert aga_module.get_active_slots() == 0
    
    def test_get_statistics(self, aga_module, random_key_vector, random_value_vector):
        """测试统计信息"""
        # 注入多个知识
        for i in range(5):
            aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=random_key_vector,
                value_vector=random_value_vector,
                lu_id=f"test_lu_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY if i % 2 == 0 else LifecycleState.CONFIRMED,
            )
        
        stats = aga_module.get_statistics()
        
        assert stats["active_slots"] == 5
        # 检查状态分布
        assert "state_distribution" in stats
        assert stats["state_distribution"]["probationary"] == 3
        assert stats["state_distribution"]["confirmed"] == 2


@pytest.mark.unit
class TestAGAModuleEdgeCases:
    """AGA 模块边界情况测试"""
    
    @pytest.fixture
    def aga_module(self, aga_config, device):
        module = AuxiliaryGovernedAttention(aga_config)
        module.to(device)
        module.eval()
        return module
    
    def test_batch_size_one(self, aga_module, hidden_dim, device, random_key_vector, random_value_vector):
        """测试批量大小为 1"""
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        single_input = torch.randn(1, 1, hidden_dim, device=device)
        primary_output = single_input.clone()
        
        output, _ = aga_module(
            hidden_states=single_input,
            primary_attention_output=primary_output,
        )
        
        assert output.shape == single_input.shape
    
    def test_long_sequence(self, aga_module, hidden_dim, device, random_key_vector, random_value_vector):
        """测试长序列"""
        aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        long_input = torch.randn(1, 512, hidden_dim, device=device)
        primary_output = long_input.clone()
        
        output, _ = aga_module(
            hidden_states=long_input,
            primary_attention_output=primary_output,
        )
        
        assert output.shape == long_input.shape
    
    def test_multiple_injections(self, aga_module, bottleneck_dim, hidden_dim, device):
        """测试多次注入"""
        for i in range(10):
            key = torch.randn(bottleneck_dim, device=device)
            value = torch.randn(hidden_dim, device=device)
            
            success = aga_module.inject_knowledge(
                slot_idx=i,
                key_vector=key,
                value_vector=value,
                lu_id=f"test_lu_{i:03d}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
            assert success is True
        
        assert aga_module.get_active_slots() == 10
    
    def test_inject_with_dimension_mismatch(self, aga_module, device):
        """测试维度不匹配时的注入（应该自动填充）"""
        # 使用较小的维度
        small_key = torch.randn(32, device=device)  # 小于 bottleneck_dim
        small_value = torch.randn(256, device=device)  # 小于 hidden_dim
        
        success = aga_module.inject_knowledge(
            slot_idx=0,
            key_vector=small_key,
            value_vector=small_value,
            lu_id="test_lu_small",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        assert success is True
        # 验证向量已被正确填充
        assert aga_module.aux_keys[0].shape[0] == aga_module.bottleneck_dim
        assert aga_module.aux_values[0].shape[0] == aga_module.hidden_dim
