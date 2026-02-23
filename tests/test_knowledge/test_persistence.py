"""
aga-knowledge 持久化适配器测试
"""

import pytest
import asyncio
import os
import tempfile

from aga_knowledge.persistence import MemoryAdapter, SQLiteAdapter


class TestMemoryAdapter:
    """测试内存持久化适配器"""

    @pytest.fixture
    def adapter(self):
        return MemoryAdapter(max_slots_per_namespace=10)

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, adapter):
        assert not await adapter.is_connected()
        await adapter.connect()
        assert await adapter.is_connected()
        await adapter.disconnect()
        assert not await adapter.is_connected()

    @pytest.mark.asyncio
    async def test_save_and_load(self, adapter):
        await adapter.connect()
        data = {
            "condition": "when X",
            "decision": "do Y",
            "lifecycle_state": "probationary",
        }
        result = await adapter.save_knowledge("default", "lu_001", data)
        assert result is True

        loaded = await adapter.load_knowledge("default", "lu_001")
        assert loaded is not None
        assert loaded["condition"] == "when X"
        assert loaded["decision"] == "do Y"
        assert loaded["lu_id"] == "lu_001"

    @pytest.mark.asyncio
    async def test_delete(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {"condition": "c", "decision": "d"})
        assert await adapter.knowledge_exists("default", "lu_001")

        await adapter.delete_knowledge("default", "lu_001")
        assert not await adapter.knowledge_exists("default", "lu_001")

    @pytest.mark.asyncio
    async def test_lru_eviction(self, adapter):
        """测试 LRU 淘汰"""
        await adapter.connect()
        # 填满 10 个槽位
        for i in range(10):
            await adapter.save_knowledge("default", f"lu_{i:03d}", {
                "condition": f"c{i}", "decision": f"d{i}",
            })

        # 第 11 个应触发淘汰
        await adapter.save_knowledge("default", "lu_new", {
            "condition": "new", "decision": "new",
        })

        # 最早的应被淘汰
        assert not await adapter.knowledge_exists("default", "lu_000")
        assert await adapter.knowledge_exists("default", "lu_new")

    @pytest.mark.asyncio
    async def test_query_with_filters(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {
            "condition": "c1", "decision": "d1", "lifecycle_state": "confirmed",
        })
        await adapter.save_knowledge("default", "lu_002", {
            "condition": "c2", "decision": "d2", "lifecycle_state": "probationary",
        })

        confirmed = await adapter.query_knowledge("default", lifecycle_states=["confirmed"])
        assert len(confirmed) == 1
        assert confirmed[0]["lu_id"] == "lu_001"

    @pytest.mark.asyncio
    async def test_batch_save(self, adapter):
        await adapter.connect()
        records = [
            {"lu_id": f"batch_{i}", "condition": f"c{i}", "decision": f"d{i}"}
            for i in range(5)
        ]
        count = await adapter.save_batch("default", records)
        assert count == 5

    @pytest.mark.asyncio
    async def test_lifecycle_update(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {
            "condition": "c", "decision": "d", "lifecycle_state": "probationary",
        })
        await adapter.update_lifecycle("default", "lu_001", "confirmed")
        record = await adapter.load_knowledge("default", "lu_001")
        assert record["lifecycle_state"] == "confirmed"

    @pytest.mark.asyncio
    async def test_statistics(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {
            "condition": "c", "decision": "d", "lifecycle_state": "confirmed",
        })
        await adapter.save_knowledge("default", "lu_002", {
            "condition": "c", "decision": "d", "lifecycle_state": "probationary",
        })
        stats = await adapter.get_statistics("default")
        assert stats["total_knowledge"] == 2
        assert stats["state_distribution"]["confirmed"] == 1
        assert stats["state_distribution"]["probationary"] == 1

    @pytest.mark.asyncio
    async def test_namespaces(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("ns1", "lu_001", {"condition": "c", "decision": "d"})
        await adapter.save_knowledge("ns2", "lu_002", {"condition": "c", "decision": "d"})
        namespaces = await adapter.get_namespaces()
        assert set(namespaces) == {"ns1", "ns2"}

    @pytest.mark.asyncio
    async def test_audit_log(self, adapter):
        await adapter.connect()
        await adapter.save_audit_log({
            "action": "INJECT",
            "lu_id": "lu_001",
            "namespace": "default",
        })
        logs = await adapter.query_audit_log(namespace="default")
        assert len(logs) == 1
        assert logs[0]["action"] == "INJECT"

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        await adapter.connect()
        health = await adapter.health_check()
        assert health["status"] == "healthy"
        assert health["adapter"] == "memory"

    @pytest.mark.asyncio
    async def test_hit_count(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {"condition": "c", "decision": "d"})
        await adapter.increment_hit_count("default", ["lu_001"])
        record = await adapter.load_knowledge("default", "lu_001")
        assert record["hit_count"] == 1


class TestSQLiteAdapter:
    """测试 SQLite 持久化适配器"""

    @pytest.fixture
    def adapter(self, tmp_path):
        db_path = str(tmp_path / "test_knowledge.db")
        return SQLiteAdapter(db_path=db_path, enable_audit=True)

    @pytest.mark.asyncio
    async def test_connect_and_init(self, adapter):
        result = await adapter.connect()
        assert result is True
        assert await adapter.is_connected()
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_load(self, adapter):
        await adapter.connect()
        data = {
            "condition": "when X happens",
            "decision": "do Y",
            "lifecycle_state": "probationary",
            "trust_tier": "standard",
        }
        result = await adapter.save_knowledge("default", "lu_001", data)
        assert result is True

        loaded = await adapter.load_knowledge("default", "lu_001")
        assert loaded is not None
        assert loaded["condition"] == "when X happens"
        assert loaded["decision"] == "do Y"
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_delete(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {"condition": "c", "decision": "d"})
        assert await adapter.knowledge_exists("default", "lu_001")

        await adapter.delete_knowledge("default", "lu_001")
        assert not await adapter.knowledge_exists("default", "lu_001")
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_query_with_filters(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {
            "condition": "c1", "decision": "d1", "lifecycle_state": "confirmed",
        })
        await adapter.save_knowledge("default", "lu_002", {
            "condition": "c2", "decision": "d2", "lifecycle_state": "probationary",
        })

        confirmed = await adapter.query_knowledge("default", lifecycle_states=["confirmed"])
        assert len(confirmed) == 1
        assert confirmed[0]["lu_id"] == "lu_001"
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_version_increment(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {"condition": "v1", "decision": "d"})
        r1 = await adapter.load_knowledge("default", "lu_001")
        assert r1["version"] == 1

        await adapter.save_knowledge("default", "lu_001", {"condition": "v2", "decision": "d"})
        r2 = await adapter.load_knowledge("default", "lu_001")
        assert r2["version"] == 2
        assert r2["condition"] == "v2"
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_audit_log(self, adapter):
        await adapter.connect()
        await adapter.save_audit_log({
            "action": "INJECT",
            "lu_id": "lu_001",
            "namespace": "default",
        })
        logs = await adapter.query_audit_log(namespace="default")
        assert len(logs) >= 1
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_statistics(self, adapter):
        await adapter.connect()
        await adapter.save_knowledge("default", "lu_001", {
            "condition": "c", "decision": "d", "lifecycle_state": "confirmed",
        })
        stats = await adapter.get_statistics("default")
        assert stats["total_knowledge"] == 1
        await adapter.disconnect()
