"""
aga-knowledge 同步发布器

Portal 使用此组件发布消息到 Runtime（明文 KV 版本）。
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .protocol import SyncMessage, MessageType, SyncAck
from .backends import SyncBackend, create_backend

logger = logging.getLogger(__name__)


class SyncPublisher:
    """
    同步发布器（明文 KV 版本）

    Portal 用于向 Runtime 发布同步消息。
    不包含向量数据，只传递 condition/decision 文本。
    """

    MAX_ACK_TIMEOUT = 60

    def __init__(
        self,
        backend_type: str = "redis",
        channel: str = "aga:sync",
        require_ack: bool = True,
        ack_timeout: int = 10,
        **backend_kwargs,
    ):
        valid_backends = {"memory", "redis"}
        if backend_type not in valid_backends:
            raise ValueError(f"Unknown backend_type '{backend_type}'. Valid: {valid_backends}")

        self.channel = channel
        self.require_ack = require_ack
        self.ack_timeout = min(ack_timeout, self.MAX_ACK_TIMEOUT)
        self._backend = create_backend(backend_type, **backend_kwargs)
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._ack_results: Dict[str, List[SyncAck]] = {}

        self._stats = {
            "messages_sent": 0,
            "acks_received": 0,
            "acks_timeout": 0,
            "errors": 0,
        }

    async def connect(self):
        """连接到后端"""
        await self._backend.connect()
        ack_channel = f"{self.channel}:ack"
        await self._backend.subscribe(ack_channel, self._handle_ack)
        logger.info(f"SyncPublisher connected, channel={self.channel}")

    async def disconnect(self):
        """断开连接"""
        await self._backend.disconnect()

    async def _handle_ack(self, message: SyncMessage):
        """处理 ACK 消息"""
        if message.message_type not in (MessageType.ACK, MessageType.NACK):
            return
        correlation_id = message.correlation_id
        if not correlation_id or correlation_id not in self._pending_acks:
            return
        ack = SyncAck(
            message_id=correlation_id,
            instance_id=message.source_instance or "unknown",
            success=(message.message_type == MessageType.ACK),
            metadata=message.metadata,
        )
        if correlation_id not in self._ack_results:
            self._ack_results[correlation_id] = []
        self._ack_results[correlation_id].append(ack)
        event = self._pending_acks.get(correlation_id)
        if event:
            event.set()
        self._stats["acks_received"] += 1

    async def publish(self, message: SyncMessage, wait_for_ack: bool = None) -> Dict[str, Any]:
        """发布消息"""
        wait_ack = wait_for_ack if wait_for_ack is not None else self.require_ack
        if wait_ack:
            message.require_ack = True
            self._pending_acks[message.message_id] = asyncio.Event()

        try:
            await self._backend.publish(self.channel, message)
            self._stats["messages_sent"] += 1
            result: Dict[str, Any] = {
                "success": True,
                "message_id": message.message_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if wait_ack:
                event = self._pending_acks[message.message_id]
                try:
                    await asyncio.wait_for(event.wait(), timeout=self.ack_timeout)
                    result["acks"] = [
                        ack.to_dict()
                        for ack in self._ack_results.get(message.message_id, [])
                    ]
                except asyncio.TimeoutError:
                    self._stats["acks_timeout"] += 1
                    result["ack_timeout"] = True
                finally:
                    self._pending_acks.pop(message.message_id, None)
                    self._ack_results.pop(message.message_id, None)
            return result
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Publish error: {e}")
            if wait_ack:
                self._pending_acks.pop(message.message_id, None)
                self._ack_results.pop(message.message_id, None)
            return {"success": False, "message_id": message.message_id, "error": str(e)}

    # ==================== 便捷方法（明文 KV 版本） ====================

    async def publish_inject(
        self, lu_id: str, condition: str, decision: str,
        namespace: str = "default", lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None, source_instance: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布知识注入消息（明文）"""
        message = SyncMessage.inject(
            lu_id=lu_id, condition=condition, decision=decision,
            namespace=namespace, lifecycle_state=lifecycle_state,
            trust_tier=trust_tier, source_instance=source_instance or "portal",
            metadata=metadata,
        )
        return await self.publish(message, wait_for_ack)

    async def publish_update(
        self, lu_id: str, new_state: str,
        namespace: str = "default", reason: Optional[str] = None,
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布生命周期更新消息"""
        message = SyncMessage.update_lifecycle(
            lu_id=lu_id, new_state=new_state,
            namespace=namespace, reason=reason,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)

    async def publish_delete(
        self, lu_id: str, namespace: str = "default",
        reason: Optional[str] = None, source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布删除消息"""
        message = SyncMessage.delete(
            lu_id=lu_id, namespace=namespace,
            reason=reason, source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "channel": self.channel,
            "connected": self._backend.is_connected,
        }
