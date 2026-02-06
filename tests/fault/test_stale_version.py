"""
版本过期测试

测试版本冲突和过期数据场景。
"""
import pytest
import asyncio
import torch
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from aga import AGAConfig, LifecycleState, KnowledgeRecord
from aga.core import AuxiliaryGovernedAttention as AGA


# ==================== 版本化记录 ====================

@dataclass
class VersionedRecord:
    """版本化记录"""
    record: KnowledgeRecord
    version: int
    timestamp: float = field(default_factory=time.time)
    
    def is_newer_than(self, other: 'VersionedRecord') -> bool:
        """是否比另一个记录更新"""
        return self.version > other.version


# ==================== 版本化存储 ====================

class VersionedStorage:
    """版本化存储"""
    
    def __init__(self):
        self._records: Dict[str, VersionedRecord] = {}
        self._version_counter: Dict[str, int] = {}
    
    def _make_key(self, namespace: str, lu_id: str) -> str:
        return f"{namespace}:{lu_id}"
    
    def save(self, record: KnowledgeRecord, expected_version: Optional[int] = None) -> int:
        """
        保存记录
        
        Args:
            record: 知识记录
            expected_version: 期望的版本号（用于乐观锁）
            
        Returns:
            新版本号
            
        Raises:
            VersionConflictError: 版本冲突
        """
        key = self._make_key(record.namespace, record.lu_id)
        
        current = self._records.get(key)
        
        if expected_version is not None:
            if current is not None and current.version != expected_version:
                raise VersionConflictError(
                    f"Version conflict: expected {expected_version}, got {current.version}"
                )
        
        new_version = self._version_counter.get(key, 0) + 1
        self._version_counter[key] = new_version
        
        self._records[key] = VersionedRecord(
            record=record,
            version=new_version,
        )
        
        return new_version
    
    def load(self, namespace: str, lu_id: str) -> Optional[VersionedRecord]:
        """加载记录"""
        key = self._make_key(namespace, lu_id)
        return self._records.get(key)
    
    def get_version(self, namespace: str, lu_id: str) -> int:
        """获取当前版本"""
        key = self._make_key(namespace, lu_id)
        versioned = self._records.get(key)
        return versioned.version if versioned else 0


class VersionConflictError(Exception):
    """版本冲突错误"""
    pass


# ==================== 测试 ====================

@pytest.mark.fault
class TestVersionConflict:
    """版本冲突测试"""
    
    @pytest.fixture
    def storage(self):
        return VersionedStorage()
    
    @pytest.fixture
    def sample_record(self, bottleneck_dim, hidden_dim):
        return KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
    
    def test_optimistic_locking_success(self, storage, sample_record):
        """测试乐观锁成功"""
        # 首次保存
        v1 = storage.save(sample_record)
        assert v1 == 1
        
        # 使用正确版本更新
        v2 = storage.save(sample_record, expected_version=1)
        assert v2 == 2
    
    def test_optimistic_locking_conflict(self, storage, sample_record):
        """测试乐观锁冲突"""
        # 首次保存
        storage.save(sample_record)
        
        # 模拟另一个客户端更新
        storage.save(sample_record)  # 版本变为 2
        
        # 使用过期版本尝试更新
        with pytest.raises(VersionConflictError):
            storage.save(sample_record, expected_version=1)
    
    def test_concurrent_updates(self, storage, sample_record):
        """测试并发更新"""
        # 首次保存
        storage.save(sample_record)
        
        # 两个客户端同时读取
        v1 = storage.get_version("default", "test_001")
        v2 = storage.get_version("default", "test_001")
        
        assert v1 == v2 == 1
        
        # 客户端 1 先更新
        storage.save(sample_record, expected_version=v1)
        
        # 客户端 2 尝试更新（应该失败）
        with pytest.raises(VersionConflictError):
            storage.save(sample_record, expected_version=v2)
    
    def test_version_history(self, storage, sample_record):
        """测试版本历史"""
        versions = []
        
        for i in range(5):
            v = storage.save(sample_record)
            versions.append(v)
        
        # 版本应该递增
        assert versions == [1, 2, 3, 4, 5]


@pytest.mark.fault
class TestStaleData:
    """过期数据测试"""
    
    @pytest.fixture
    def storage(self):
        return VersionedStorage()
    
    def test_stale_read_detection(self, storage, bottleneck_dim, hidden_dim):
        """测试过期读取检测"""
        record = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件 v1",
            decision="决策 v1",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        # 保存 v1
        storage.save(record)
        
        # 读取 v1
        read_v1 = storage.load("default", "test_001")
        
        # 更新到 v2
        record.condition = "条件 v2"
        storage.save(record)
        
        # read_v1 现在是过期的
        current = storage.load("default", "test_001")
        
        assert read_v1.version < current.version
        assert read_v1.record.condition == "条件 v1"
        assert current.record.condition == "条件 v2"
    
    def test_stale_write_prevention(self, storage, bottleneck_dim, hidden_dim):
        """测试过期写入预防"""
        record = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        # 保存初始版本
        storage.save(record)
        
        # 读取（模拟缓存）
        cached = storage.load("default", "test_001")
        cached_version = cached.version
        
        # 其他地方更新了记录
        storage.save(record)
        storage.save(record)
        
        # 尝试基于缓存版本更新（应该失败）
        with pytest.raises(VersionConflictError):
            storage.save(record, expected_version=cached_version)


@pytest.mark.fault
class TestVersionReconciliation:
    """版本协调测试"""
    
    @pytest.fixture
    def storage_a(self):
        return VersionedStorage()
    
    @pytest.fixture
    def storage_b(self):
        return VersionedStorage()
    
    def test_version_comparison(self, storage_a, storage_b, bottleneck_dim, hidden_dim):
        """测试版本比较"""
        record = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件",
            decision="决策",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        # 两个存储独立更新
        storage_a.save(record)
        storage_a.save(record)
        storage_a.save(record)  # v3
        
        storage_b.save(record)
        storage_b.save(record)  # v2
        
        versioned_a = storage_a.load("default", "test_001")
        versioned_b = storage_b.load("default", "test_001")
        
        # A 的版本更高
        assert versioned_a.is_newer_than(versioned_b)
    
    def test_last_write_wins(self, storage_a, storage_b, bottleneck_dim, hidden_dim):
        """测试最后写入胜出"""
        record_a = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件 A",
            decision="决策 A",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
        
        record_b = KnowledgeRecord(
            lu_id="test_001",
            namespace="default",
            condition="条件 B",
            decision="决策 B",
            key_vector=torch.randn(bottleneck_dim).tolist(),
            value_vector=torch.randn(hidden_dim).tolist(),
            lifecycle_state=LifecycleState.CONFIRMED,
        )
        
        # A 先写入
        storage_a.save(record_a)
        time.sleep(0.01)
        
        # B 后写入
        storage_b.save(record_b)
        
        versioned_a = storage_a.load("default", "test_001")
        versioned_b = storage_b.load("default", "test_001")
        
        # B 的时间戳更新
        assert versioned_b.timestamp > versioned_a.timestamp
        
        # 协调：选择时间戳更新的
        if versioned_b.timestamp > versioned_a.timestamp:
            winner = versioned_b
        else:
            winner = versioned_a
        
        assert winner.record.condition == "条件 B"
