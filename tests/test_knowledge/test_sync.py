"""
aga-knowledge 同步协议测试
"""

import pytest
import asyncio
import json

from aga_knowledge.sync import (
    SyncMessage,
    MessageType,
    SyncAck,
    SyncPublisher,
    MemoryBackend,
)


class TestSyncMessage:
    """测试同步消息"""

    def test_create_inject(self):
        msg = SyncMessage.inject(
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
            namespace="default",
        )
        assert msg.message_type == MessageType.INJECT
        assert msg.lu_id == "lu_001"
        assert msg.condition == "when X"
        assert msg.decision == "do Y"
        assert msg.require_ack is True

    def test_create_update(self):
        msg = SyncMessage.update_lifecycle(
            lu_id="lu_001",
            new_state="confirmed",
            reason="passed review",
        )
        assert msg.message_type == MessageType.UPDATE
        assert msg.lifecycle_state == "confirmed"
        assert msg.reason == "passed review"

    def test_create_delete(self):
        msg = SyncMessage.delete(lu_id="lu_001", reason="obsolete")
        assert msg.message_type == MessageType.DELETE
        assert msg.reason == "obsolete"

    def test_create_quarantine(self):
        msg = SyncMessage.quarantine(lu_id="lu_001", reason="error detected")
        assert msg.message_type == MessageType.QUARANTINE
        assert msg.lifecycle_state == "quarantined"

    def test_create_batch(self):
        items = [
            {"lu_id": "lu_001", "condition": "c1", "decision": "d1"},
            {"lu_id": "lu_002", "condition": "c2", "decision": "d2"},
        ]
        msg = SyncMessage.batch_inject(items=items, namespace="default")
        assert msg.message_type == MessageType.BATCH_INJECT
        assert len(msg.batch_items) == 2

    def test_json_roundtrip(self):
        msg = SyncMessage.inject(
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
        )
        json_str = msg.to_json()
        restored = SyncMessage.from_json(json_str)
        assert restored.lu_id == msg.lu_id
        assert restored.condition == msg.condition
        assert restored.message_type == MessageType.INJECT


class TestSyncAck:
    """测试同步确认"""

    def test_create_ack(self):
        ack = SyncAck(
            message_id="msg_001",
            instance_id="runtime-001",
            success=True,
        )
        assert ack.success is True

    def test_json_roundtrip(self):
        ack = SyncAck(
            message_id="msg_001",
            instance_id="runtime-001",
            success=False,
            error="timeout",
        )
        json_str = ack.to_json()
        restored = SyncAck.from_json(json_str)
        assert restored.message_id == "msg_001"
        assert restored.success is False
        assert restored.error == "timeout"


class TestMemoryBackend:
    """测试内存后端"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        MemoryBackend.clear_all()
        yield
        MemoryBackend.clear_all()

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        backend = MemoryBackend()
        await backend.connect()

        received = []

        async def callback(msg):
            received.append(msg)

        await backend.subscribe("test_channel", callback)

        msg = SyncMessage.inject(lu_id="lu_001", condition="c", decision="d")
        await backend.publish("test_channel", msg)

        assert len(received) == 1
        assert received[0].lu_id == "lu_001"

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        backend = MemoryBackend()
        await backend.connect()

        received = []

        async def callback(msg):
            received.append(msg)

        await backend.subscribe("test_channel", callback)
        await backend.unsubscribe("test_channel")

        msg = SyncMessage.inject(lu_id="lu_001", condition="c", decision="d")
        await backend.publish("test_channel", msg)

        assert len(received) == 0
        await backend.disconnect()


class TestSyncPublisher:
    """测试同步发布器"""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        MemoryBackend.clear_all()
        yield
        MemoryBackend.clear_all()

    @pytest.mark.asyncio
    async def test_publish_inject(self):
        publisher = SyncPublisher(
            backend_type="memory",
            channel="test:sync",
            require_ack=False,
        )
        await publisher.connect()

        result = await publisher.publish_inject(
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
        )
        assert result["success"] is True
        assert publisher.get_stats()["messages_sent"] == 1

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_publish_update(self):
        publisher = SyncPublisher(
            backend_type="memory",
            channel="test:sync",
            require_ack=False,
        )
        await publisher.connect()

        result = await publisher.publish_update(
            lu_id="lu_001",
            new_state="confirmed",
        )
        assert result["success"] is True

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_publish_delete(self):
        publisher = SyncPublisher(
            backend_type="memory",
            channel="test:sync",
            require_ack=False,
        )
        await publisher.connect()

        result = await publisher.publish_delete(
            lu_id="lu_001",
            reason="obsolete",
        )
        assert result["success"] is True

        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_stats(self):
        publisher = SyncPublisher(
            backend_type="memory",
            channel="test:sync",
            require_ack=False,
        )
        await publisher.connect()

        stats = publisher.get_stats()
        assert stats["messages_sent"] == 0
        assert stats["channel"] == "test:sync"
        assert stats["connected"] is True

        await publisher.disconnect()
