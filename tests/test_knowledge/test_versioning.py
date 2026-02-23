"""
aga-knowledge 版本控制模块测试
"""

import pytest
from datetime import datetime

from aga_knowledge.persistence import MemoryAdapter
from aga_knowledge.persistence.versioning import (
    KnowledgeVersion,
    VersionDiff,
    VersionedKnowledgeStore,
)


class TestKnowledgeVersion:
    """测试 KnowledgeVersion 数据类"""

    def test_to_dict(self):
        version = KnowledgeVersion(
            version=1,
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
            lifecycle_state="probationary",
            trust_tier="standard",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            created_by="admin",
            change_reason="Initial creation",
        )
        d = version.to_dict()
        assert d["version"] == 1
        assert d["lu_id"] == "lu_001"
        assert d["condition"] == "when X"
        assert d["decision"] == "do Y"
        assert d["created_by"] == "admin"

    def test_from_dict(self):
        data = {
            "version": 2,
            "lu_id": "lu_002",
            "condition": "when A",
            "decision": "do B",
            "lifecycle_state": "confirmed",
            "trust_tier": "high",
            "created_at": "2025-01-01T12:00:00",
            "created_by": "system",
            "change_reason": "Update",
        }
        version = KnowledgeVersion.from_dict(data)
        assert version.version == 2
        assert version.lu_id == "lu_002"
        assert version.condition == "when A"
        assert version.trust_tier == "high"

    def test_from_dict_defaults(self):
        data = {"lu_id": "lu_003"}
        version = KnowledgeVersion.from_dict(data)
        assert version.version == 1
        assert version.lifecycle_state == "probationary"
        assert version.trust_tier == "standard"
        assert version.created_by == "system"

    def test_roundtrip(self):
        original = KnowledgeVersion(
            version=3,
            lu_id="lu_004",
            condition="condition text",
            decision="decision text",
            lifecycle_state="confirmed",
            trust_tier="high",
            created_at=datetime(2025, 6, 15, 10, 30, 0),
            created_by="admin",
            change_reason="test roundtrip",
            metadata={"source": "test"},
        )
        d = original.to_dict()
        restored = KnowledgeVersion.from_dict(d)
        assert restored.version == original.version
        assert restored.lu_id == original.lu_id
        assert restored.condition == original.condition
        assert restored.decision == original.decision
        assert restored.metadata == original.metadata


class TestVersionDiff:
    """测试 VersionDiff"""

    def test_to_dict(self):
        diff = VersionDiff(
            field="condition",
            old_value="old text",
            new_value="new text",
        )
        d = diff.to_dict()
        assert d["field"] == "condition"
        assert d["old_value"] == "old text"
        assert d["new_value"] == "new text"


class TestVersionedKnowledgeStore:
    """测试版本化知识存储"""

    @pytest.fixture
    async def store(self):
        adapter = MemoryAdapter(max_slots_per_namespace=128)
        await adapter.connect()
        return VersionedKnowledgeStore(
            persistence_adapter=adapter,
            max_versions_per_knowledge=5,
        )

    @pytest.mark.asyncio
    async def test_create_knowledge(self, store):
        store = await store
        version = await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
            created_by="admin",
        )
        assert version.version == 1
        assert version.lu_id == "lu_001"
        assert version.condition == "when X"
        assert version.decision == "do Y"
        assert version.lifecycle_state == "probationary"

    @pytest.mark.asyncio
    async def test_update_knowledge(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
            created_by="admin",
        )

        new_version = await store.update_knowledge(
            namespace="default",
            lu_id="lu_001",
            decision="do Z",
            updated_by="admin",
            change_reason="Improved decision",
        )
        assert new_version.version == 2
        assert new_version.condition == "when X"  # 未变
        assert new_version.decision == "do Z"  # 已更新

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, store):
        store = await store
        with pytest.raises(ValueError, match="not found"):
            await store.update_knowledge(
                namespace="default",
                lu_id="nonexistent",
                decision="new",
            )

    @pytest.mark.asyncio
    async def test_get_latest_version(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="d",
            created_by="admin",
        )
        await store.update_knowledge(
            namespace="default",
            lu_id="lu_001",
            decision="d2",
        )

        latest = await store.get_latest_version("default", "lu_001")
        assert latest is not None
        assert latest.version == 2
        assert latest.decision == "d2"

    @pytest.mark.asyncio
    async def test_get_version(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="d1",
            created_by="admin",
        )
        await store.update_knowledge(
            namespace="default",
            lu_id="lu_001",
            decision="d2",
        )

        v1 = await store.get_version("default", "lu_001", 1)
        assert v1 is not None
        assert v1.decision == "d1"

        v2 = await store.get_version("default", "lu_001", 2)
        assert v2 is not None
        assert v2.decision == "d2"

    @pytest.mark.asyncio
    async def test_rollback(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="original",
            created_by="admin",
        )
        await store.update_knowledge(
            namespace="default",
            lu_id="lu_001",
            decision="modified",
        )

        rollback = await store.rollback(
            namespace="default",
            lu_id="lu_001",
            target_version=1,
            rolled_back_by="admin",
        )
        assert rollback.version == 3
        assert rollback.decision == "original"
        assert "rollback_from" in rollback.metadata

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_version(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="d",
            created_by="admin",
        )

        with pytest.raises(ValueError, match="not found"):
            await store.rollback(
                namespace="default",
                lu_id="lu_001",
                target_version=99,
            )

    @pytest.mark.asyncio
    async def test_compare_versions(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c1",
            decision="d1",
            created_by="admin",
        )
        await store.update_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c2",
            decision="d2",
        )

        diffs = await store.compare_versions(
            "default", "lu_001", 1, 2
        )
        assert len(diffs) >= 2

        field_names = [d.field for d in diffs]
        assert "condition" in field_names
        assert "decision" in field_names

    @pytest.mark.asyncio
    async def test_version_history_limit(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="d0",
            created_by="admin",
        )

        for i in range(1, 8):
            await store.update_knowledge(
                namespace="default",
                lu_id="lu_001",
                decision=f"d{i}",
            )

        # max_versions_per_knowledge=5，缓存中应最多 5 个
        history = await store.get_version_history("default", "lu_001")
        assert len(history) <= 5

    @pytest.mark.asyncio
    async def test_clear_cache(self, store):
        store = await store
        await store.create_knowledge(
            namespace="default",
            lu_id="lu_001",
            condition="c",
            decision="d",
            created_by="admin",
        )

        # 确认缓存中有数据
        assert "default" in store._version_cache

        store.clear_cache()
        assert len(store._version_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_cache_namespace(self, store):
        store = await store
        await store.create_knowledge(
            namespace="ns1",
            lu_id="lu_001",
            condition="c",
            decision="d",
            created_by="admin",
        )
        await store.create_knowledge(
            namespace="ns2",
            lu_id="lu_002",
            condition="c",
            decision="d",
            created_by="admin",
        )

        store.clear_cache("ns1")
        assert "ns1" not in store._version_cache
        assert "ns2" in store._version_cache
