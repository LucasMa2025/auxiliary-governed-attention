"""
AGA 核心模块单元测试

测试覆盖：
- 生命周期状态转换
- 知识注入
- 熵门控
- 前向传播
- 槽位管理
"""
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import Mock, patch
from datetime import datetime

# 导入被测模块
import sys
sys.path.insert(0, '..')

from aga.core import (
    AuxiliaryGovernedAttention,
    AGAConfig,
    LifecycleState,
    KnowledgeSlotInfo,
    AGADiagnostics,
    UncertaintySource,
)


class TestLifecycleState:
    """生命周期状态测试"""
    
    def test_lifecycle_values(self):
        """测试生命周期状态值"""
        assert LifecycleState.PROBATIONARY.value == "probationary"
        assert LifecycleState.CONFIRMED.value == "confirmed"
        assert LifecycleState.DEPRECATED.value == "deprecated"
        assert LifecycleState.QUARANTINED.value == "quarantined"
    
    def test_lifecycle_reliability_mapping(self):
        """测试生命周期到可靠性的映射"""
        aga = AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
        )
        
        assert aga.LIFECYCLE_RELIABILITY[LifecycleState.PROBATIONARY] == 0.3
        assert aga.LIFECYCLE_RELIABILITY[LifecycleState.CONFIRMED] == 1.0
        assert aga.LIFECYCLE_RELIABILITY[LifecycleState.DEPRECATED] == 0.1
        assert aga.LIFECYCLE_RELIABILITY[LifecycleState.QUARANTINED] == 0.0


class TestAGAConfig:
    """AGA 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AGAConfig()
        
        assert config.hidden_dim == 4096
        assert config.bottleneck_dim == 64
        assert config.num_slots == 100
        assert config.tau_low == 0.5
        assert config.tau_high == 2.0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AGAConfig(
            hidden_dim=1024,
            bottleneck_dim=128,
            num_slots=50,
            tau_low=0.3,
            tau_high=3.0,
        )
        
        assert config.hidden_dim == 1024
        assert config.bottleneck_dim == 128
        assert config.num_slots == 50


class TestKnowledgeInjection:
    """知识注入测试"""
    
    @pytest.fixture
    def aga(self):
        """创建 AGA 实例"""
        config = AGAConfig(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
            enable_norm_clipping=True,
            key_norm_target=5.0,
            value_norm_target=3.0,
        )
        aga = AuxiliaryGovernedAttention(config=config)
        aga.eval()
        return aga
    
    def test_inject_knowledge_basic(self, aga):
        """测试基本知识注入"""
        key_vector = torch.randn(32)
        value_vector = torch.randn(256)
        
        success = aga.inject_knowledge(
            slot_idx=0,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        assert success is True
        assert aga.slot_lifecycle[0] == LifecycleState.PROBATIONARY
        assert aga.slot_lu_ids[0] == "test_lu_001"
    
    def test_inject_knowledge_with_metadata(self, aga):
        """测试带元数据的知识注入"""
        key_vector = torch.randn(32)
        value_vector = torch.randn(256)
        
        success = aga.inject_knowledge(
            slot_idx=1,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_002",
            lifecycle_state=LifecycleState.CONFIRMED,
            condition="用户询问天气",
            decision="调用天气 API",
        )
        
        assert success is True
        assert aga.slot_conditions[1] == "用户询问天气"
        assert aga.slot_decisions[1] == "调用天气 API"
    
    def test_inject_knowledge_norm_clipping(self, aga):
        """测试范数裁剪"""
        # 创建大范数向量
        key_vector = torch.randn(32) * 100
        value_vector = torch.randn(256) * 100
        
        aga.inject_knowledge(
            slot_idx=2,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_003",
        )
        
        # 检查范数被裁剪
        actual_key_norm = aga.aux_keys[2].norm().item()
        actual_value_norm = aga.aux_values[2].norm().item()
        
        assert abs(actual_key_norm - 5.0) < 0.1
        assert abs(actual_value_norm - 3.0) < 0.1
    
    def test_inject_knowledge_dimension_padding(self, aga):
        """测试维度填充"""
        # 创建小维度向量
        key_vector = torch.randn(16)  # 小于 bottleneck_dim
        value_vector = torch.randn(128)  # 小于 hidden_dim
        
        success = aga.inject_knowledge(
            slot_idx=3,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_004",
        )
        
        assert success is True
        assert aga.aux_keys[3].shape[0] == 32
        assert aga.aux_values[3].shape[0] == 256
    
    def test_inject_knowledge_training_mode_error(self, aga):
        """测试训练模式下注入失败"""
        aga.train()
        
        with pytest.raises(RuntimeError, match="training mode"):
            aga.inject_knowledge(
                slot_idx=0,
                key_vector=torch.randn(32),
                value_vector=torch.randn(256),
                lu_id="test_lu_005",
            )
    
    def test_inject_knowledge_invalid_slot_idx(self, aga):
        """测试无效槽位索引"""
        with pytest.raises(ValueError, match="slot_idx"):
            aga.inject_knowledge(
                slot_idx=100,  # 超出范围
                key_vector=torch.randn(32),
                value_vector=torch.randn(256),
                lu_id="test_lu_006",
            )


class TestLifecycleTransitions:
    """生命周期转换测试"""
    
    @pytest.fixture
    def aga_with_knowledge(self):
        """创建带知识的 AGA 实例"""
        aga = AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
        )
        aga.eval()
        
        # 注入测试知识
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        return aga
    
    def test_confirm_slot(self, aga_with_knowledge):
        """测试确认槽位"""
        aga_with_knowledge.confirm_slot(0)
        assert aga_with_knowledge.slot_lifecycle[0] == LifecycleState.CONFIRMED
    
    def test_deprecate_slot(self, aga_with_knowledge):
        """测试弃用槽位"""
        aga_with_knowledge.deprecate_slot(0)
        assert aga_with_knowledge.slot_lifecycle[0] == LifecycleState.DEPRECATED
    
    def test_quarantine_slot(self, aga_with_knowledge):
        """测试隔离槽位"""
        aga_with_knowledge.quarantine_slot(0)
        
        assert aga_with_knowledge.slot_lifecycle[0] == LifecycleState.QUARANTINED
        # 检查 value 被清零
        assert aga_with_knowledge.aux_values[0].norm().item() == 0.0
    
    def test_quarantine_by_lu_id(self, aga_with_knowledge):
        """测试按 LU ID 隔离"""
        quarantined = aga_with_knowledge.quarantine_by_lu_id("test_lu")
        
        assert 0 in quarantined
        assert aga_with_knowledge.slot_lifecycle[0] == LifecycleState.QUARANTINED


class TestEntropyVeto:
    """熵否决机制测试"""
    
    @pytest.fixture
    def aga(self):
        """创建 AGA 实例"""
        config = AGAConfig(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
            tau_low=0.5,
            tau_high=2.0,
        )
        return AuxiliaryGovernedAttention(config=config)
    
    def test_low_entropy_veto(self, aga):
        """测试低熵否决（主模型确信）"""
        gate = torch.ones(2, 4) * 0.5
        entropy = torch.ones(2, 4) * 0.3  # 低于 tau_low
        
        result = aga._apply_entropy_veto(gate, entropy)
        
        # 低熵时 gate 应该为 0
        assert (result == 0).all()
    
    def test_high_entropy_cap(self, aga):
        """测试高熵限制"""
        gate = torch.ones(2, 4) * 0.9
        entropy = torch.ones(2, 4) * 2.5  # 高于 tau_high
        
        result = aga._apply_entropy_veto(gate, entropy)
        
        # 高熵时 gate 应该被限制在 0.8
        assert (result <= 0.8).all()
    
    def test_normal_entropy_pass(self, aga):
        """测试正常熵通过"""
        gate = torch.ones(2, 4) * 0.6
        entropy = torch.ones(2, 4) * 1.0  # 在 tau_low 和 tau_high 之间
        
        result = aga._apply_entropy_veto(gate, entropy)
        
        # 正常熵时 gate 保持不变
        assert torch.allclose(result, gate)


class TestForwardPass:
    """前向传播测试"""
    
    @pytest.fixture
    def aga_with_knowledge(self):
        """创建带知识的 AGA 实例"""
        config = AGAConfig(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
            uncertainty_source=UncertaintySource.HIDDEN_VARIANCE,
        )
        aga = AuxiliaryGovernedAttention(config=config)
        aga.eval()
        
        # 注入一些知识
        for i in range(3):
            aga.inject_knowledge(
                slot_idx=i,
                key_vector=torch.randn(32),
                value_vector=torch.randn(256),
                lu_id=f"test_lu_{i}",
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        return aga
    
    def test_forward_basic(self, aga_with_knowledge):
        """测试基本前向传播"""
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, 256)
        primary_output = torch.randn(batch_size, seq_len, 256)
        
        fused_output, diagnostics = aga_with_knowledge(
            hidden_states=hidden_states,
            primary_attention_output=primary_output,
            return_diagnostics=True,
        )
        
        assert fused_output.shape == (batch_size, seq_len, 256)
        assert diagnostics is not None
        assert diagnostics.active_slots == 3
    
    def test_forward_no_active_slots(self):
        """测试无活跃槽位时的前向传播"""
        aga = AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=10,
        )
        aga.eval()
        
        hidden_states = torch.randn(2, 8, 256)
        primary_output = torch.randn(2, 8, 256)
        
        fused_output, diagnostics = aga(
            hidden_states=hidden_states,
            primary_attention_output=primary_output,
        )
        
        # 无活跃槽位时应该返回原始输出
        assert torch.allclose(fused_output, primary_output)
    
    def test_forward_with_routing(self, aga_with_knowledge):
        """测试带路由的前向传播"""
        hidden_states = torch.randn(2, 8, 256)
        primary_output = torch.randn(2, 8, 256)
        
        fused_output, diagnostics = aga_with_knowledge(
            hidden_states=hidden_states,
            primary_attention_output=primary_output,
            use_routing=True,
            return_diagnostics=True,
        )
        
        assert fused_output.shape == (2, 8, 256)
        assert diagnostics.routed_slots is not None


class TestSlotManagement:
    """槽位管理测试"""
    
    @pytest.fixture
    def aga(self):
        """创建 AGA 实例"""
        return AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=5,
        )
    
    def test_find_free_slot(self, aga):
        """测试查找空闲槽位"""
        aga.eval()
        
        # 初始时所有槽位都是空闲的
        free_slot = aga.find_free_slot()
        assert free_slot == 0
        
        # 注入知识后
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu",
        )
        
        free_slot = aga.find_free_slot()
        assert free_slot == 1
    
    def test_get_active_slots(self, aga):
        """测试获取活跃槽位数"""
        aga.eval()
        
        assert aga.get_active_slots() == 0
        
        # 注入知识
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu_1",
        )
        aga.inject_knowledge(
            slot_idx=1,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu_2",
        )
        
        assert aga.get_active_slots() == 2
    
    def test_get_slot_info(self, aga):
        """测试获取槽位信息"""
        aga.eval()
        
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu",
            lifecycle_state=LifecycleState.CONFIRMED,
            condition="test condition",
            decision="test decision",
        )
        
        info = aga.get_slot_info(0)
        
        assert isinstance(info, KnowledgeSlotInfo)
        assert info.lu_id == "test_lu"
        assert info.lifecycle_state == LifecycleState.CONFIRMED
        assert info.reliability == 1.0
        assert info.condition == "test condition"
        assert info.decision == "test decision"
    
    def test_get_statistics(self, aga):
        """测试获取统计信息"""
        aga.eval()
        
        # 注入不同状态的知识
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu_1",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        aga.inject_knowledge(
            slot_idx=1,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu_2",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        stats = aga.get_statistics()
        
        assert stats['total_slots'] == 5
        assert stats['active_slots'] == 2
        assert stats['state_distribution']['confirmed'] == 1
        assert stats['state_distribution']['probationary'] == 1


class TestExportImport:
    """导出导入测试"""
    
    @pytest.fixture
    def aga_with_knowledge(self):
        """创建带知识的 AGA 实例"""
        aga = AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=5,
        )
        aga.eval()
        
        aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.randn(32),
            value_vector=torch.randn(256),
            lu_id="test_lu",
            lifecycle_state=LifecycleState.CONFIRMED,
            condition="test condition",
            decision="test decision",
        )
        
        return aga
    
    def test_export_state(self, aga_with_knowledge):
        """测试导出状态"""
        state = aga_with_knowledge.export_state()
        
        assert 'version' in state
        assert 'config' in state
        assert 'aux_keys' in state
        assert 'aux_values' in state
        assert 'slot_lifecycle' in state
        assert 'slot_lu_ids' in state
    
    def test_import_state(self, aga_with_knowledge):
        """测试导入状态"""
        # 导出
        state = aga_with_knowledge.export_state()
        
        # 创建新实例
        new_aga = AuxiliaryGovernedAttention(
            hidden_dim=256,
            bottleneck_dim=32,
            num_slots=5,
        )
        new_aga.eval()
        
        # 导入
        new_aga.import_state(state)
        
        # 验证
        assert new_aga.slot_lu_ids[0] == "test_lu"
        assert new_aga.slot_lifecycle[0] == LifecycleState.CONFIRMED
        assert new_aga.slot_conditions[0] == "test condition"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

