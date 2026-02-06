"""
Runtime 本地缓存单元测试

测试 LocalCache 和 CachedSlot 的核心功能。
"""
import pytest
import time
from unittest.mock import Mock, patch

from aga.runtime.cache import LocalCache, CachedSlot


@pytest.mark.unit
class TestCachedSlot:
    """CachedSlot 测试"""
    
    def test_create_cached_slot(self):
        """测试创建缓存槽位"""
        slot = CachedSlot(
            lu_id="LU_001",
            slot_idx=0,
            namespace="default",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
            condition="条件",
            decision="决策",
        )
        
        assert slot.lu_id == "LU_001"
        assert slot.slot_idx == 0
        assert slot.namespace == "default"
        assert slot.lifecycle_state == "probationary"
    
    def test_record_hit(self):
        """测试记录命中"""
        slot = CachedSlot(
            lu_id="LU_001",
            slot_idx=0,
            namespace="default",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        old_ts = slot.last_hit_ts
        time.sleep(0.01)
        
        slot.record_hit()
        
        assert slot.hit_count == 1
        assert slot.last_hit_ts > old_ts
    
    def test_multiple_hits(self):
        """测试多次命中"""
        slot = CachedSlot(
            lu_id="LU_001",
            slot_idx=0,
            namespace="default",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        for _ in range(5):
            slot.record_hit()
        
        assert slot.hit_count == 5


@pytest.mark.unit
class TestLocalCacheBasic:
    """LocalCache 基本功能测试"""
    
    def test_create_cache(self):
        """测试创建缓存"""
        cache = LocalCache(
            max_slots=100,
            device="cpu",
            dtype="float32",
        )
        
        assert cache.max_slots == 100
        assert cache.device == "cpu"
    
    def test_add_slot(self):
        """测试添加槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        slot_idx = cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
            namespace="default",
            condition="条件",
            decision="决策",
        )
        
        assert slot_idx is not None
        assert "LU_001" in cache._slots
    
    def test_add_duplicate(self):
        """测试添加重复槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        slot_idx1 = cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        slot_idx2 = cache.add(
            lu_id="LU_001",
            key_vector=[0.3] * 64,
            value_vector=[0.4] * 256,
        )
        
        # 应该返回相同的槽位索引
        assert slot_idx1 == slot_idx2
    
    def test_get_slot(self):
        """测试获取槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        slot = cache.get("LU_001")
        
        assert slot is not None
        assert slot.lu_id == "LU_001"
    
    def test_get_nonexistent(self):
        """测试获取不存在的槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        slot = cache.get("NONEXISTENT")
        
        assert slot is None
    
    def test_remove_slot(self):
        """测试移除槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        result = cache.remove("LU_001")
        
        assert result is True
        assert cache.get("LU_001") is None
    
    def test_remove_nonexistent(self):
        """测试移除不存在的槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        result = cache.remove("NONEXISTENT")
        
        assert result is False
    
    def test_contains(self):
        """测试包含检查"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        assert cache.contains("LU_001") is True
        assert cache.contains("LU_002") is False
    
    def test_get_all(self):
        """测试获取所有槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(lu_id="LU_001", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        cache.add(lu_id="LU_002", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        
        all_slots = cache.get_all()
        
        assert len(all_slots) == 2
    
    def test_clear(self):
        """测试清空"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(lu_id="LU_001", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        cache.add(lu_id="LU_002", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        
        cache.clear()
        
        assert len(cache.get_all()) == 0


@pytest.mark.unit
class TestLocalCacheCapacity:
    """LocalCache 容量测试"""
    
    def test_cache_full(self):
        """测试缓存已满"""
        cache = LocalCache(max_slots=3, device="cpu")
        
        # 添加 3 个槽位
        for i in range(3):
            cache.add(
                lu_id=f"LU_{i:03d}",
                key_vector=[0.1] * 64,
                value_vector=[0.2] * 256,
            )
        
        # 第 4 个应该触发淘汰
        slot_idx = cache.add(
            lu_id="LU_NEW",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
        )
        
        # 应该成功（淘汰了一个旧的）
        assert slot_idx is not None
        assert len(cache.get_all()) == 3


@pytest.mark.unit
class TestLocalCacheUpdate:
    """LocalCache 更新测试"""
    
    def test_update_lifecycle(self):
        """测试更新生命周期"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(
            lu_id="LU_001",
            key_vector=[0.1] * 64,
            value_vector=[0.2] * 256,
            lifecycle_state="probationary",
        )
        
        result = cache.update("LU_001", lifecycle_state="confirmed")
        
        assert result is True
        assert cache.get("LU_001").lifecycle_state == "confirmed"
    
    def test_update_nonexistent(self):
        """测试更新不存在的槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        result = cache.update("NONEXISTENT", lifecycle_state="confirmed")
        
        assert result is False


@pytest.mark.unit
class TestLocalCacheQuery:
    """LocalCache 查询测试"""
    
    def test_get_all_by_namespace(self):
        """测试按命名空间获取"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(lu_id="LU_001", key_vector=[0.1] * 64, value_vector=[0.2] * 256, namespace="ns1")
        cache.add(lu_id="LU_002", key_vector=[0.1] * 64, value_vector=[0.2] * 256, namespace="ns2")
        cache.add(lu_id="LU_003", key_vector=[0.1] * 64, value_vector=[0.2] * 256, namespace="ns1")
        
        ns1_slots = cache.get_all("ns1")
        
        assert len(ns1_slots) == 2
    
    def test_get_active(self):
        """测试获取活跃槽位"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(lu_id="LU_001", key_vector=[0.1] * 64, value_vector=[0.2] * 256, lifecycle_state="confirmed")
        cache.add(lu_id="LU_002", key_vector=[0.1] * 64, value_vector=[0.2] * 256, lifecycle_state="quarantined")
        
        active = cache.get_active()
        
        # 只有非隔离的
        assert len(active) == 1
        assert active[0].lu_id == "LU_001"
    
    def test_get_stats(self):
        """测试获取统计"""
        cache = LocalCache(max_slots=10, device="cpu")
        
        cache.add(lu_id="LU_001", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        cache.add(lu_id="LU_002", key_vector=[0.1] * 64, value_vector=[0.2] * 256)
        
        stats = cache.get_stats()
        
        assert stats["total_slots"] == 2
        assert stats["max_slots"] == 10
        assert stats["available_slots"] == 8
