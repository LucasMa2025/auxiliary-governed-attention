"""
SQLite 适配器组件测试

测试 SQLite 持久化适配器的功能。
"""
import pytest
import asyncio
import os
import torch

from aga import SQLiteAdapter, LifecycleState
from aga.persistence.base import KnowledgeRecord


@pytest.mark.component
@pytest.mark.asyncio
class TestSQLiteAdapter:
    """SQLite 适配器测试"""
    
    @pytest.fixture
    async def adapter(self, temp_db_path):
        """创建适配器"""
        adapter = SQLiteAdapter(temp_db_path)
        await adapter.connect()
        yield adapter
        await adapter.disconnect()
    
    async def test_connect_disconnect(self, temp_db_path):
        """测试连接和断开"""
        adapter = SQLiteAdapter(temp_db_path)
        
        await adapter.connect()
        assert await adapter.is_connected()
        
        await adapter.disconnect()
        assert not await adapter.is_connected()
    
    async def test_save_and_load_slot(self, adapter, sample_knowledge_record):
        """测试保存和加载槽位"""
        # 保存
        await adapter.save_slot(sample_knowledge_record.namespace, sample_knowledge_record)
        
        # 加载
        loaded = await adapter.load_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        
        assert loaded is not None
        assert loaded.lu_id == sample_knowledge_record.lu_id
        assert loaded.condition == sample_knowledge_record.condition
        assert loaded.decision == sample_knowledge_record.decision
    
    async def test_save_batch(self, adapter, sample_knowledge_records):
        """测试批量保存"""
        namespace = sample_knowledge_records[0].namespace
        await adapter.save_batch(namespace, sample_knowledge_records)
        
        # 验证所有记录都被保存
        for record in sample_knowledge_records:
            loaded = await adapter.load_slot(record.namespace, record.lu_id)
            assert loaded is not None
            assert loaded.lu_id == record.lu_id
    
    async def test_load_all_slots(self, adapter, sample_knowledge_records):
        """测试加载所有槽位"""
        namespace = sample_knowledge_records[0].namespace
        await adapter.save_batch(namespace, sample_knowledge_records)
        
        all_slots = await adapter.load_all_slots(namespace)
        
        assert len(all_slots) == len(sample_knowledge_records)
    
    async def test_delete_slot(self, adapter, sample_knowledge_record):
        """测试删除槽位"""
        await adapter.save_slot(sample_knowledge_record.namespace, sample_knowledge_record)
        
        # 确认存在
        loaded = await adapter.load_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded is not None
        
        # 删除
        await adapter.delete_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        
        # 确认已删除
        loaded = await adapter.load_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded is None
    
    async def test_update_lifecycle(self, adapter, sample_knowledge_record):
        """测试更新生命周期"""
        from aga.types import LifecycleState as LSType
        
        await adapter.save_slot(sample_knowledge_record.namespace, sample_knowledge_record)
        
        # 更新状态 - 传入枚举
        await adapter.update_lifecycle(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
            LSType.CONFIRMED,
        )
        
        # 验证更新
        loaded = await adapter.load_slot(
            sample_knowledge_record.namespace,
            sample_knowledge_record.lu_id,
        )
        assert loaded.lifecycle_state == LSType.CONFIRMED.value
    
    async def test_get_statistics(self, adapter, sample_knowledge_records):
        """测试获取统计信息"""
        namespace = sample_knowledge_records[0].namespace
        await adapter.save_batch(namespace, sample_knowledge_records)
        
        stats = await adapter.get_statistics(namespace)
        
        # 检查统计信息中的槽位数
        assert "slot_count" in stats or "total" in stats or len(stats) > 0
        assert "state_distribution" in stats
    
    async def test_namespace_isolation(self, adapter, bottleneck_dim, hidden_dim):
        """测试命名空间隔离"""
        # 在不同命名空间保存记录
        record1 = KnowledgeRecord(
            slot_idx=0,
            lu_id="test_001",
            namespace="ns1",
            condition="条件1",
            decision="决策1",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY.value,
        )
        record2 = KnowledgeRecord(
            slot_idx=0,
            lu_id="test_001",  # 相同 lu_id
            namespace="ns2",
            condition="条件2",
            decision="决策2",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY.value,
        )
        
        await adapter.save_slot("ns1", record1)
        await adapter.save_slot("ns2", record2)
        
        # 验证隔离
        loaded1 = await adapter.load_slot("ns1", "test_001")
        loaded2 = await adapter.load_slot("ns2", "test_001")
        
        assert loaded1.condition == "条件1"
        assert loaded2.condition == "条件2"


@pytest.mark.component
@pytest.mark.asyncio
class TestSQLiteAdapterEdgeCases:
    """SQLite 适配器边界情况测试"""
    
    @pytest.fixture
    async def adapter(self, temp_db_path):
        adapter = SQLiteAdapter(temp_db_path)
        await adapter.connect()
        yield adapter
        await adapter.disconnect()
    
    async def test_load_nonexistent(self, adapter):
        """测试加载不存在的记录"""
        loaded = await adapter.load_slot("default", "nonexistent")
        assert loaded is None
    
    async def test_delete_nonexistent(self, adapter):
        """测试删除不存在的记录"""
        # 不应该抛出异常
        await adapter.delete_slot("default", "nonexistent")
    
    async def test_update_nonexistent(self, adapter):
        """测试更新不存在的记录"""
        # 不应该抛出异常
        await adapter.update_lifecycle("default", "nonexistent", LifecycleState.CONFIRMED.value)
    
    async def test_empty_namespace(self, adapter):
        """测试空命名空间"""
        all_slots = await adapter.load_all_slots("empty_namespace")
        assert len(all_slots) == 0
    
    async def test_concurrent_access(self, adapter, sample_knowledge_records):
        """测试并发访问"""
        namespace = sample_knowledge_records[0].namespace
        
        # 并发保存
        tasks = [adapter.save_slot(namespace, record) for record in sample_knowledge_records]
        await asyncio.gather(*tasks)
        
        # 验证所有记录都被保存
        all_slots = await adapter.load_all_slots(namespace)
        assert len(all_slots) == len(sample_knowledge_records)
    
    async def test_large_vectors(self, adapter, hidden_dim):
        """测试大向量"""
        large_record = KnowledgeRecord(
            slot_idx=0,
            lu_id="large_001",
            namespace="default",
            condition="大向量测试",
            decision="决策",
            key_vector=torch.randn(1024).tolist(),
            value_vector=torch.randn(4096).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY.value,
        )
        
        await adapter.save_slot("default", large_record)
        loaded = await adapter.load_slot("default", "large_001")
        
        assert len(loaded.key_vector) == 1024
        assert len(loaded.value_vector) == 4096
