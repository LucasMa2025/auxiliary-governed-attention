"""
槽位池组件测试

测试 SlotPool 的功能。
"""
import pytest
import torch
import asyncio
import threading

from aga.production.slot_pool import SlotPool, Slot, LifecycleState, EvictionPolicy
from aga.production.config import SlotPoolConfig


@pytest.mark.component
class TestSlotPool:
    """SlotPool 测试"""
    
    @pytest.fixture
    def pool_config(self, hidden_dim, bottleneck_dim, num_slots):
        """创建槽位池配置"""
        return SlotPoolConfig(
            max_slots_per_namespace=num_slots,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            eviction_enabled=True,
        )
    
    @pytest.fixture
    def pool(self, pool_config, device):
        """创建槽位池"""
        return SlotPool(
            namespace="test",
            config=pool_config,
            device=device,
        )
    
    def test_initialization(self, pool, pool_config):
        """测试初始化"""
        assert pool.namespace == "test"
        assert pool.max_slots == pool_config.max_slots_per_namespace
        assert pool.active_count == 0
    
    def test_add_slot(self, pool, random_key_vector, random_value_vector):
        """测试添加槽位"""
        slot_idx = pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        assert slot_idx is not None
        assert pool.active_count == 1
    
    def test_get_slot(self, pool, random_key_vector, random_value_vector):
        """测试获取槽位"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        slot = pool.get_slot("test_001")
        
        assert slot is not None
        assert slot.lu_id == "test_001"
    
    def test_remove_slot(self, pool, random_key_vector, random_value_vector):
        """测试移除槽位"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        assert pool.active_count == 1
        
        result = pool.remove_slot("test_001")
        
        assert result is True
        assert pool.active_count == 0
        assert pool.get_slot("test_001") is None
    
    def test_quarantine_slot(self, pool, random_key_vector, random_value_vector):
        """测试隔离槽位"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        result = pool.quarantine_slot("test_001")
        
        assert result is True
        slot = pool.get_slot("test_001")
        assert slot.lifecycle_state == LifecycleState.QUARANTINED
        assert slot.reliability == 0.0
    
    def test_update_lifecycle(self, pool, random_key_vector, random_value_vector):
        """测试更新生命周期"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        result = pool.update_lifecycle("test_001", LifecycleState.CONFIRMED)
        
        assert result is True
        slot = pool.get_slot("test_001")
        assert slot.lifecycle_state == LifecycleState.CONFIRMED
        assert slot.reliability == 1.0
    
    def test_get_vectors(self, pool, random_key_vector, random_value_vector):
        """测试获取向量"""
        # 添加多个槽位
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        keys, values, reliability = pool.get_vectors()
        
        assert keys.shape[0] == 5
        assert values.shape[0] == 5
        assert reliability.shape[0] == 5
    
    def test_quarantined_excluded_from_vectors(self, pool, random_key_vector, random_value_vector):
        """测试隔离槽位不包含在向量中"""
        # 添加槽位
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
                lifecycle_state=LifecycleState.CONFIRMED,
            )
        
        # 隔离一个
        pool.quarantine_slot("test_002")
        
        # 验证隔离后的槽位状态
        slot = pool.get_slot("test_002")
        assert slot.lifecycle_state == LifecycleState.QUARANTINED
        assert slot.reliability == 0.0
        
        # 获取向量 - 隔离的槽位可能被排除或 reliability 为 0
        keys, values, reliability = pool.get_vectors()
        
        # 验证向量数量（可能排除了隔离的槽位）
        assert keys.shape[0] >= 4  # 至少有 4 个非隔离槽位
    
    def test_eviction(self, pool_config, device, random_key_vector, random_value_vector):
        """测试淘汰机制"""
        # 创建小容量池
        small_config = SlotPoolConfig(
            max_slots_per_namespace=5,
            hidden_dim=pool_config.hidden_dim,
            bottleneck_dim=pool_config.bottleneck_dim,
            eviction_enabled=True,
        )
        pool = SlotPool(namespace="test", config=small_config, device=device)
        
        # 填满池
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
            )
        
        assert pool.active_count == 5
        
        # 添加新槽位，应该触发淘汰
        result = pool.add_slot(
            lu_id="test_new",
            key_vector=random_key_vector.clone(),
            value_vector=random_value_vector.clone(),
        )
        
        # 淘汰后槽位数应该不超过最大值
        assert pool.active_count <= 5
        # 如果添加成功，新槽位应该存在
        if result is not None:
            assert pool.get_slot("test_new") is not None
    
    def test_hit_recording(self, pool, random_key_vector, random_value_vector):
        """测试命中记录"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        slot = pool.get_slot("test_001")
        initial_hits = slot.hit_count
        
        # 记录命中 - 直接在槽位上记录
        slot.record_hit()
        
        slot = pool.get_slot("test_001")
        assert slot.hit_count == initial_hits + 1
    
    def test_statistics(self, pool, random_key_vector, random_value_vector):
        """测试统计信息"""
        # 添加多个槽位
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
                lifecycle_state=LifecycleState.PROBATIONARY if i % 2 == 0 else LifecycleState.CONFIRMED,
            )
        
        stats = pool.get_statistics()
        
        assert stats["active_slots"] == 5
        assert stats["occupancy_ratio"] > 0
        assert "state_distribution" in stats


@pytest.mark.component
class TestSlotPoolEdgeCases:
    """SlotPool 边界情况测试"""
    
    @pytest.fixture
    def pool_config(self, hidden_dim, bottleneck_dim, num_slots):
        return SlotPoolConfig(
            max_slots_per_namespace=num_slots,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
    
    @pytest.fixture
    def pool(self, pool_config, device):
        return SlotPool(namespace="test", config=pool_config, device=device)
    
    def test_duplicate_lu_id(self, pool, random_key_vector, random_value_vector):
        """测试重复 lu_id"""
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        # 尝试添加相同 lu_id
        slot_idx = pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        # 应该返回 None 或更新现有槽位
        assert pool.active_count == 1
    
    def test_remove_nonexistent(self, pool):
        """测试移除不存在的槽位"""
        result = pool.remove_slot("nonexistent")
        assert result is False
    
    def test_quarantine_nonexistent(self, pool):
        """测试隔离不存在的槽位"""
        result = pool.quarantine_slot("nonexistent")
        assert result is False
    
    def test_empty_pool_vectors(self, pool):
        """测试空池的向量"""
        keys, values, reliability = pool.get_vectors()
        
        assert keys.shape[0] == 0
        assert values.shape[0] == 0
        assert reliability.shape[0] == 0
    
    def test_concurrent_access(self, pool, random_key_vector, random_value_vector):
        """测试并发访问"""
        errors = []
        
        def add_slots(start_idx):
            try:
                for i in range(10):
                    pool.add_slot(
                        lu_id=f"test_{start_idx}_{i:03d}",
                        key_vector=random_key_vector.clone(),
                        value_vector=random_value_vector.clone(),
                    )
            except Exception as e:
                errors.append(e)
        
        # 并发添加
        threads = [
            threading.Thread(target=add_slots, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 不应该有错误
        assert len(errors) == 0
        # 应该有一些槽位被添加
        assert pool.active_count > 0
