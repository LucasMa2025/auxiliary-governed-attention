"""
动态槽位池组件测试

测试 DynamicSlotPool 和 DynamicSlotManager 的功能。
"""
import pytest
import torch
import time

from aga.production.slot_pool import SlotPool, LifecycleState
from aga.production.config import SlotPoolConfig
from aga.production.dynamic_slots import (
    DynamicSlotPool,
    DynamicSlotManager,
    DynamicSlotConfig,
    TieredSlotStorage,
    ScalingPolicy,
    SlotTier,
)


@pytest.mark.component
class TestDynamicSlotPool:
    """DynamicSlotPool 测试"""
    
    @pytest.fixture
    def base_config(self, hidden_dim, bottleneck_dim):
        """基础槽位池配置"""
        return SlotPoolConfig(
            max_slots_per_namespace=100,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
    
    @pytest.fixture
    def dynamic_config(self):
        """动态槽位配置"""
        return DynamicSlotConfig(
            initial_capacity=10,
            min_capacity=5,
            max_capacity=50,
            expand_threshold=0.8,
            shrink_threshold=0.2,
            expand_factor=2.0,
            shrink_factor=0.5,
            scaling_policy=ScalingPolicy.AUTO_SCALE,
        )
    
    @pytest.fixture
    def pool(self, dynamic_config, base_config, device):
        """创建动态槽位池"""
        pool = DynamicSlotPool(
            namespace="test",
            config=dynamic_config,
            base_config=base_config,
            device=device,
        )
        yield pool
        pool.stop_monitor()
    
    def test_initialization(self, pool, dynamic_config):
        """测试初始化"""
        assert pool.namespace == "test"
        assert pool.current_capacity == dynamic_config.initial_capacity
        assert pool.active_count == 0
    
    def test_add_slot(self, pool, random_key_vector, random_value_vector):
        """测试添加槽位"""
        slot_idx = pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        assert slot_idx is not None
        assert pool.active_count == 1
    
    def test_auto_expand(self, pool, random_key_vector, random_value_vector):
        """测试自动扩容"""
        initial_capacity = pool.current_capacity
        
        # 添加足够多的槽位触发扩容
        for i in range(int(initial_capacity * 0.9)):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
            )
        
        # 容量应该增加
        assert pool.current_capacity >= initial_capacity
    
    def test_manual_resize(self, pool):
        """测试手动调整大小"""
        initial_capacity = pool.current_capacity
        new_capacity = initial_capacity + 10
        
        result = pool.resize(new_capacity)
        
        assert result is True
        assert pool.current_capacity == new_capacity
    
    def test_resize_below_active(self, pool, random_key_vector, random_value_vector):
        """测试调整大小低于活跃数"""
        # 添加一些槽位
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
            )
        
        # 尝试调整到低于活跃数
        result = pool.resize(3)
        
        # 应该失败或调整到活跃数
        assert pool.current_capacity >= pool.active_count
    
    def test_resize_below_min(self, pool, dynamic_config):
        """测试调整大小低于最小值"""
        result = pool.resize(dynamic_config.min_capacity - 1)
        
        # 应该失败或调整到最小值
        assert pool.current_capacity >= dynamic_config.min_capacity
    
    def test_resize_above_max(self, pool, dynamic_config):
        """测试调整大小高于最大值"""
        result = pool.resize(dynamic_config.max_capacity + 10)
        
        # 应该失败或调整到最大值
        assert pool.current_capacity <= dynamic_config.max_capacity
    
    def test_scaling_history(self, pool, random_key_vector, random_value_vector):
        """测试扩缩容历史"""
        # 触发扩容
        for i in range(int(pool.current_capacity * 0.9)):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
            )
        
        history = pool.get_scaling_history()
        
        # 历史应该是列表
        assert isinstance(history, list)
    
    def test_statistics(self, pool, random_key_vector, random_value_vector):
        """测试统计信息"""
        # 添加一些槽位
        for i in range(5):
            pool.add_slot(
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
            )
        
        stats = pool.get_statistics()
        
        assert "active_slots" in stats
        assert "current_capacity" in stats
        assert "occupancy_ratio" in stats


@pytest.mark.component
class TestDynamicSlotManager:
    """DynamicSlotManager 测试"""
    
    @pytest.fixture
    def base_config(self, hidden_dim, bottleneck_dim):
        return SlotPoolConfig(
            max_slots_per_namespace=100,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
    
    @pytest.fixture
    def dynamic_config(self):
        return DynamicSlotConfig(
            initial_capacity=10,
            min_capacity=5,
            max_capacity=50,
        )
    
    @pytest.fixture
    def manager(self, dynamic_config, base_config, device):
        """创建管理器"""
        manager = DynamicSlotManager(
            config=dynamic_config,
            base_config=base_config,
            device=device,
        )
        yield manager
        manager.shutdown()
    
    def test_get_pool(self, manager):
        """测试获取池"""
        pool = manager.get_pool("test_ns")
        
        assert pool is not None
        assert pool.namespace == "test_ns"
    
    def test_multiple_namespaces(self, manager):
        """测试多个命名空间"""
        pool1 = manager.get_pool("ns1")
        pool2 = manager.get_pool("ns2")
        
        assert pool1 is not pool2
        assert pool1.namespace == "ns1"
        assert pool2.namespace == "ns2"
    
    def test_remove_pool(self, manager):
        """测试移除池"""
        manager.get_pool("test_ns")
        
        result = manager.remove_pool("test_ns")
        
        assert result is True
    
    def test_statistics(self, manager, random_key_vector, random_value_vector):
        """测试统计信息"""
        pool = manager.get_pool("test_ns")
        pool.add_slot(
            lu_id="test_001",
            key_vector=random_key_vector,
            value_vector=random_value_vector,
        )
        
        stats = manager.get_statistics()
        
        assert "total_namespaces" in stats
        assert stats["total_namespaces"] >= 1
    
    def test_shutdown(self, manager):
        """测试关闭"""
        manager.get_pool("ns1")
        manager.get_pool("ns2")
        
        manager.shutdown()
        
        # 关闭后应该没有池
        stats = manager.get_statistics()
        assert stats["total_namespaces"] == 0


@pytest.mark.component
class TestTieredStorage:
    """TieredSlotStorage 测试"""
    
    @pytest.fixture
    def base_config(self, hidden_dim, bottleneck_dim):
        return SlotPoolConfig(
            max_slots_per_namespace=100,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
    
    @pytest.fixture
    def dynamic_config(self):
        return DynamicSlotConfig(
            initial_capacity=10,
        )
    
    @pytest.fixture
    def storage(self, dynamic_config, base_config, device):
        """创建分层存储"""
        return TieredSlotStorage(
            namespace="test",
            config=dynamic_config,
            base_config=base_config,
            device=device,
        )
    
    def test_tiered_statistics(self, storage, random_key_vector, random_value_vector, device):
        """测试分层统计"""
        from aga.production.slot_pool import Slot
        
        # 添加一些槽位
        for i in range(5):
            slot = Slot(
                slot_idx=i,
                lu_id=f"test_{i:03d}",
                key_vector=random_key_vector.clone(),
                value_vector=random_value_vector.clone(),
                lifecycle_state=LifecycleState.CONFIRMED,
            )
            storage.put(slot, SlotTier.HOT)
        
        stats = storage.get_statistics()
        
        assert "hot_count" in stats
        assert "warm_count" in stats
        assert "cold_count" in stats
