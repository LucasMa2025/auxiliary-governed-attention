"""
单节点集成测试

测试单节点部署场景下的完整流程。
"""
import pytest
import asyncio
import torch
import tempfile
import os

from aga import AGAConfig, LifecycleState, SQLiteAdapter, SlotPoolConfig, KnowledgeRecord
from aga.core import AuxiliaryGovernedAttention as AGA

# 尝试导入生产模块
try:
    from aga.production.slot_pool import SlotPool
    HAS_SLOT_POOL = True
except ImportError:
    HAS_SLOT_POOL = False
    SlotPool = None


@pytest.mark.integration
@pytest.mark.asyncio
class TestSingleNodeIntegration:
    """单节点集成测试"""
    
    @pytest.fixture
    async def setup_single_node(self, aga_config, device, temp_db_path):
        """设置单节点环境"""
        # 创建组件
        aga_module = AGA(aga_config)
        aga_module.to(device)
        aga_module.eval()
        
        persistence = SQLiteAdapter(temp_db_path)
        await persistence.connect()
        
        slot_pool_config = SlotPoolConfig(
            max_slots_per_namespace=aga_config.num_slots,
            hidden_dim=aga_config.hidden_dim,
            bottleneck_dim=aga_config.bottleneck_dim,
        )
        slot_pool = SlotPool("default", slot_pool_config, device)
        
        yield {
            "aga": aga_module,
            "persistence": persistence,
            "slot_pool": slot_pool,
            "config": aga_config,
            "device": device,
        }
        
        await persistence.disconnect()
    
    async def test_inject_and_query(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试注入和查询流程"""
        env = setup_single_node
        
        # 创建知识
        key_vector = torch.randn(bottleneck_dim, device=env["device"])
        value_vector = torch.randn(hidden_dim, device=env["device"])
        
        # 注入到 AGA 模块
        slot_idx = env["aga"].inject_knowledge(
            slot_idx=0,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # 同步到槽位池
        env["slot_pool"].add_slot(
            lu_id="test_lu_001",
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # 持久化
        from aga.persistence.base import KnowledgeRecord
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=key_vector.tolist(),
            value_vector=value_vector.tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        await env["persistence"].save_slot(record)
        
        # 验证
        assert env["aga"].active_slots == 1
        assert env["slot_pool"].active_count == 1
        
        loaded = await env["persistence"].load_slot("default", "test_lu_001")
        assert loaded is not None
    
    async def test_inference_with_knowledge(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试带知识的推理"""
        env = setup_single_node
        
        # 注入知识
        key_vector = torch.randn(bottleneck_dim, device=env["device"])
        value_vector = torch.randn(hidden_dim, device=env["device"])
        
        env["aga"].inject_knowledge(
            slot_idx=0,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # 推理
        input_hidden = torch.randn(1, 8, hidden_dim, device=env["device"])
        output, diagnostics = env["aga"](input_hidden)
        
        assert output.shape == input_hidden.shape
        assert diagnostics is not None
    
    async def test_lifecycle_transition(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试生命周期转换"""
        env = setup_single_node
        
        # 注入为试用状态
        key_vector = torch.randn(bottleneck_dim, device=env["device"])
        value_vector = torch.randn(hidden_dim, device=env["device"])
        
        env["aga"].inject_knowledge(
            slot_idx=0,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        # 验证初始状态
        slot_info = env["aga"].get_slot_info(0)
        assert slot_info.lifecycle_state == LifecycleState.PROBATIONARY
        assert slot_info.reliability == 0.3
        
        # 转换为确认状态
        env["aga"].update_lifecycle(0, LifecycleState.CONFIRMED)
        
        # 验证转换后状态
        slot_info = env["aga"].get_slot_info(0)
        assert slot_info.lifecycle_state == LifecycleState.CONFIRMED
        assert slot_info.reliability == 1.0
    
    async def test_quarantine_and_recovery(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试隔离和恢复"""
        env = setup_single_node
        
        # 注入知识
        key_vector = torch.randn(bottleneck_dim, device=env["device"])
        value_vector = torch.randn(hidden_dim, device=env["device"])
        
        env["aga"].inject_knowledge(
            slot_idx=0,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id="test_lu_001",
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # 隔离
        env["aga"].quarantine_slot(0)
        
        slot_info = env["aga"].get_slot_info(0)
        assert slot_info.lifecycle_state == LifecycleState.QUARANTINED
        assert slot_info.reliability == 0.0
        
        # 推理时隔离槽位不应该参与
        input_hidden = torch.randn(1, 8, hidden_dim, device=env["device"])
        output, diagnostics = env["aga"](input_hidden)
        
        # 输出应该接近输入（因为唯一的槽位被隔离）
        assert torch.allclose(output, input_hidden, atol=0.1)
    
    async def test_persistence_recovery(self, setup_single_node, bottleneck_dim, hidden_dim, temp_db_path):
        """测试持久化恢复"""
        env = setup_single_node
        
        # 注入并持久化
        key_vector = torch.randn(bottleneck_dim, device=env["device"])
        value_vector = torch.randn(hidden_dim, device=env["device"])
        
        record = KnowledgeRecord(
            lu_id="test_lu_001",
            namespace="default",
            condition="测试条件",
            decision="测试决策",
            key_vector=key_vector.tolist(),
            value_vector=value_vector.tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        await env["persistence"].save_slot(record)
        
        # 模拟重启：创建新的 AGA 模块
        new_aga = AGA(env["config"])
        new_aga.to(env["device"])
        new_aga.eval()
        
        # 从持久化恢复
        loaded = await env["persistence"].load_slot("default", "test_lu_001")
        
        new_aga.inject_knowledge(
            slot_idx=0,
            key_vector=torch.tensor(loaded.key_vector, device=env["device"]),
            value_vector=torch.tensor(loaded.value_vector, device=env["device"]),
            lu_id=loaded.lu_id,
            lifecycle_state=loaded.lifecycle_state,
        )
        
        # 验证恢复
        assert new_aga.active_slots == 1
        slot_info = new_aga.get_slot_info(0)
        assert slot_info.lu_id == "test_lu_001"


@pytest.mark.integration
@pytest.mark.asyncio
class TestSingleNodeBatch:
    """单节点批量操作测试"""
    
    @pytest.fixture
    async def setup_single_node(self, aga_config, device, temp_db_path):
        """设置单节点环境"""
        aga_module = AGA(aga_config)
        aga_module.to(device)
        aga_module.eval()
        
        persistence = SQLiteAdapter(temp_db_path)
        await persistence.connect()
        
        yield {
            "aga": aga_module,
            "persistence": persistence,
            "config": aga_config,
            "device": device,
        }
        
        await persistence.disconnect()
    
    async def test_batch_inject(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试批量注入"""
        env = setup_single_node
        
        # 批量创建知识
        records = []
        for i in range(10):
            key_vector = torch.randn(bottleneck_dim, device=env["device"])
            value_vector = torch.randn(hidden_dim, device=env["device"])
            
            env["aga"].inject_knowledge(
                slot_idx=i,
                key_vector=key_vector,
                value_vector=value_vector,
                lu_id=f"test_lu_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
            
            records.append(KnowledgeRecord(
                lu_id=f"test_lu_{i:03d}",
                namespace="default",
                condition=f"条件 {i}",
                decision=f"决策 {i}",
                key_vector=key_vector.tolist(),
                value_vector=value_vector.tolist(),
                lifecycle_state=LifecycleState.PROBATIONARY,
            ))
        
        # 批量持久化
        await env["persistence"].save_batch(records)
        
        # 验证
        assert env["aga"].active_slots == 10
        
        all_slots = await env["persistence"].load_all_slots("default")
        assert len(all_slots) == 10
    
    async def test_batch_lifecycle_update(self, setup_single_node, bottleneck_dim, hidden_dim):
        """测试批量生命周期更新"""
        env = setup_single_node
        
        # 注入知识
        for i in range(5):
            key_vector = torch.randn(bottleneck_dim, device=env["device"])
            value_vector = torch.randn(hidden_dim, device=env["device"])
            
            env["aga"].inject_knowledge(
                slot_idx=i,
                key_vector=key_vector,
                value_vector=value_vector,
                lu_id=f"test_lu_{i:03d}",
                lifecycle_state=LifecycleState.PROBATIONARY,
            )
        
        # 批量更新
        for i in range(5):
            env["aga"].update_lifecycle(i, LifecycleState.CONFIRMED)
        
        # 验证
        stats = env["aga"].get_statistics()
        assert stats["confirmed_count"] == 5
        assert stats["probationary_count"] == 0
