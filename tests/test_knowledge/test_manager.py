"""
aga-knowledge 知识管理器测试
"""

import pytest
import asyncio

from aga_knowledge.config import PortalConfig
from aga_knowledge.manager import KnowledgeManager
from aga_knowledge.sync.backends import MemoryBackend


class TestKnowledgeManager:
    """测试知识管理器"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        MemoryBackend.clear_all()
        yield
        MemoryBackend.clear_all()

    @pytest.fixture
    def config(self):
        return PortalConfig.for_development()

    @pytest.mark.asyncio
    async def test_start_stop(self, config):
        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()
        assert manager._running is True

        stats = manager.get_stats()
        assert stats["instance_id"] == "test-001"
        assert stats["running"] is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_get_knowledge_empty(self, config):
        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        result = await manager.get_knowledge("default", "nonexistent")
        assert result is None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_active_knowledge(self, config):
        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        # 手动注入到缓存
        manager._cache["default"] = {
            "lu_001": {
                "lu_id": "lu_001",
                "condition": "c1",
                "decision": "d1",
                "lifecycle_state": "confirmed",
            },
            "lu_002": {
                "lu_id": "lu_002",
                "condition": "c2",
                "decision": "d2",
                "lifecycle_state": "quarantined",
            },
        }

        active = await manager.get_active_knowledge("default")
        assert len(active) == 1
        assert active[0]["lu_id"] == "lu_001"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_knowledge_for_injection(self, config):
        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        manager._cache["default"] = {
            "lu_001": {
                "lu_id": "lu_001",
                "condition": "c1",
                "decision": "d1",
                "lifecycle_state": "confirmed",
            },
        }

        injection_list = await manager.get_knowledge_for_injection("default")
        assert len(injection_list) == 1
        assert injection_list[0]["reliability"] == 1.0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_cache_summary(self, config):
        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        manager._cache["default"] = {
            "lu_001": {"lu_id": "lu_001", "lifecycle_state": "confirmed"},
            "lu_002": {"lu_id": "lu_002", "lifecycle_state": "probationary"},
        }

        summary = manager.get_cache_summary()
        assert summary["default"]["total"] == 2
        assert summary["default"]["state_distribution"]["confirmed"] == 1
        assert summary["default"]["state_distribution"]["probationary"] == 1

        await manager.stop()

    @pytest.mark.asyncio
    async def test_sync_message_inject(self, config):
        """测试通过同步消息注入知识"""
        from aga_knowledge.sync import SyncMessage, MessageType

        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        # 模拟注入消息
        msg = SyncMessage.inject(
            lu_id="lu_sync_001",
            condition="sync condition",
            decision="sync decision",
            namespace="default",
        )
        await manager._handle_sync_message(msg)

        record = await manager.get_knowledge("default", "lu_sync_001")
        assert record is not None
        assert record["condition"] == "sync condition"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_sync_message_delete(self, config):
        """测试通过同步消息删除知识"""
        from aga_knowledge.sync import SyncMessage

        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        # 先注入
        manager._cache["default"] = {
            "lu_001": {"lu_id": "lu_001", "condition": "c", "decision": "d"},
        }

        # 删除
        msg = SyncMessage.delete(lu_id="lu_001", namespace="default")
        await manager._handle_sync_message(msg)

        record = await manager.get_knowledge("default", "lu_001")
        assert record is None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_sync_message_quarantine(self, config):
        """测试通过同步消息隔离知识"""
        from aga_knowledge.sync import SyncMessage

        manager = KnowledgeManager(config, instance_id="test-001")
        await manager.start()

        manager._cache["default"] = {
            "lu_001": {"lu_id": "lu_001", "condition": "c", "decision": "d"},
        }

        msg = SyncMessage.quarantine(lu_id="lu_001", reason="error", namespace="default")
        await manager._handle_sync_message(msg)

        # 隔离后应从缓存移除
        assert "lu_001" not in manager._cache.get("default", {})

        await manager.stop()

    @pytest.mark.asyncio
    async def test_on_knowledge_update_callback(self, config):
        """测试知识更新回调"""
        from aga_knowledge.sync import SyncMessage

        updates = []

        def on_update(namespace, lu_id, record):
            updates.append((namespace, lu_id, record))

        manager = KnowledgeManager(
            config, instance_id="test-001",
            on_knowledge_update=on_update,
        )
        await manager.start()

        msg = SyncMessage.inject(
            lu_id="lu_cb_001",
            condition="callback test",
            decision="callback decision",
            namespace="default",
        )
        await manager._handle_sync_message(msg)

        assert len(updates) == 1
        assert updates[0][0] == "default"
        assert updates[0][1] == "lu_cb_001"

        await manager.stop()
