"""
KVStore 单元测试
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from aga.kv_store import KVStore


class TestKVStore:
    """KVStore 测试"""

    @pytest.fixture
    def store(self):
        """创建测试用 KVStore"""
        return KVStore(
            max_slots=10,
            key_dim=64,
            value_dim=128,
            device=torch.device("cpu"),
        )

    def test_init(self, store):
        """测试初始化"""
        assert store.count == 0
        assert store.max_slots == 10
        assert store.key_dim == 64
        assert store.value_dim == 128

    def test_put_and_count(self, store):
        """测试写入和计数"""
        key = torch.randn(64)
        value = torch.randn(128)
        assert store.put("test_001", key, value, reliability=0.9)
        assert store.count == 1

    def test_put_multiple(self, store):
        """测试多次写入"""
        for i in range(5):
            key = torch.randn(64)
            value = torch.randn(128)
            assert store.put(f"test_{i:03d}", key, value)
        assert store.count == 5

    def test_put_update_existing(self, store):
        """测试更新已有知识"""
        key1 = torch.randn(64)
        value1 = torch.randn(128)
        store.put("test_001", key1, value1, reliability=0.5)
        assert store.count == 1

        key2 = torch.randn(64)
        value2 = torch.randn(128)
        store.put("test_001", key2, value2, reliability=0.9)
        assert store.count == 1  # 不应增加

    def test_remove(self, store):
        """测试移除"""
        key = torch.randn(64)
        value = torch.randn(128)
        store.put("test_001", key, value)
        assert store.count == 1

        assert store.remove("test_001")
        assert store.count == 0

    def test_remove_nonexistent(self, store):
        """测试移除不存在的知识"""
        assert not store.remove("nonexistent")

    def test_get(self, store):
        """测试获取"""
        key = torch.randn(64)
        value = torch.randn(128)
        store.put("test_001", key, value, reliability=0.8)

        result = store.get("test_001")
        assert result is not None
        k, v, r = result
        assert k.shape == (64,)
        assert v.shape == (128,)
        assert abs(r - 0.8) < 0.01

    def test_get_nonexistent(self, store):
        """测试获取不存在的知识"""
        assert store.get("nonexistent") is None

    def test_get_active(self, store):
        """测试获取活跃知识"""
        for i in range(3):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value, reliability=0.5 + i * 0.1)

        keys, values, reliability = store.get_active()
        assert keys.shape == (3, 64)
        assert values.shape == (3, 128)
        assert reliability.shape == (3,)

    def test_get_active_empty(self, store):
        """测试空存储获取活跃知识"""
        keys, values, reliability = store.get_active()
        assert keys.shape == (0, 64)
        assert values.shape == (0, 128)
        assert reliability.shape == (0,)

    def test_contains(self, store):
        """测试存在性检查"""
        key = torch.randn(64)
        value = torch.randn(128)
        store.put("test_001", key, value)

        assert store.contains("test_001")
        assert not store.contains("test_002")

    def test_clear(self, store):
        """测试清空"""
        for i in range(5):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value)
        assert store.count == 5

        store.clear()
        assert store.count == 0

    def test_clear_namespace(self, store):
        """测试按命名空间清空"""
        for i in range(3):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"ns1_{i}", key, value, metadata={"namespace": "ns1"})
        for i in range(2):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"ns2_{i}", key, value, metadata={"namespace": "ns2"})

        assert store.count == 5
        store.clear(namespace="ns1")
        assert store.count == 2

    def test_lru_eviction(self, store):
        """测试 LRU 淘汰"""
        # 填满所有槽位
        for i in range(10):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value)
        assert store.count == 10

        # 再写入一个，应该淘汰最早的
        key = torch.randn(64)
        value = torch.randn(128)
        store.put("test_new", key, value)
        assert store.count == 10
        assert store.contains("test_new")
        assert not store.contains("test_000")  # 最早的被淘汰

    def test_utilization(self, store):
        """测试使用率"""
        assert store.utilization == 0.0

        for i in range(5):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value)

        assert store.utilization == 0.5

    def test_get_stats(self, store):
        """测试统计信息"""
        for i in range(3):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value)

        stats = store.get_stats()
        assert stats["count"] == 3
        assert stats["max_slots"] == 10
        assert stats["utilization"] == 0.3
        assert stats["free_slots"] == 7
        assert stats["vram_bytes"] > 0

    def test_get_all_ids(self, store):
        """测试获取所有 ID"""
        for i in range(3):
            key = torch.randn(64)
            value = torch.randn(128)
            store.put(f"test_{i:03d}", key, value)

        ids = store.get_all_ids()
        assert len(ids) == 3
        assert "test_000" in ids
        assert "test_001" in ids
        assert "test_002" in ids
