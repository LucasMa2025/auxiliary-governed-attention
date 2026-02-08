"""
AGA 同步发布器

Portal 使用此组件发布消息到 Runtime。
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
    同步发布器
    
    Portal 用于向 Runtime 发布同步消息。
    
    示例:
        publisher = SyncPublisher(
            backend_type="redis",
            redis_host="localhost",
            channel="aga:sync",
        )
        await publisher.connect()
        
        # 发布注入消息
        await publisher.publish_inject(
            lu_id="knowledge_001",
            key_vector=[...],
            value_vector=[...],
            condition="...",
            decision="...",
        )
        
        await publisher.disconnect()
    """
    
    # 最大超时保护
    MAX_ACK_TIMEOUT = 60
    
    def __init__(
        self,
        backend_type: str = "redis",
        channel: str = "aga:sync",
        require_ack: bool = True,
        ack_timeout: int = 10,
        **backend_kwargs
    ):
        """
        初始化发布器
        
        Args:
            backend_type: 后端类型 (memory, redis, kafka)
            channel: 发布通道
            require_ack: 是否需要确认
            ack_timeout: 确认超时（秒）
            **backend_kwargs: 后端配置
        """
        # 验证 backend_type
        valid_backends = {"memory", "redis", "kafka"}
        if backend_type not in valid_backends:
            raise ValueError(f"Unknown backend_type '{backend_type}'. Valid options: {valid_backends}")
        
        self.channel = channel
        self.require_ack = require_ack
        # 限制最大超时
        self.ack_timeout = min(ack_timeout, self.MAX_ACK_TIMEOUT)
        
        self._backend = create_backend(backend_type, **backend_kwargs)
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._ack_results: Dict[str, List[SyncAck]] = {}
        self._ack_lock = asyncio.Lock()  # 保护 ACK 结果
        
        # 统计
        self._stats = {
            "messages_sent": 0,
            "acks_received": 0,
            "acks_timeout": 0,
            "errors": 0,
        }
    
    async def connect(self):
        """连接到后端"""
        await self._backend.connect()
        
        # 订阅 ACK 通道
        ack_channel = f"{self.channel}:ack"
        await self._backend.subscribe(ack_channel, self._handle_ack)
        
        logger.info(f"SyncPublisher connected, channel={self.channel}")
    
    async def disconnect(self):
        """断开连接"""
        await self._backend.disconnect()
        logger.info("SyncPublisher disconnected")
    
    async def _handle_ack(self, message: SyncMessage):
        """处理 ACK 消息"""
        if message.message_type not in (MessageType.ACK, MessageType.NACK):
            return
        
        correlation_id = message.correlation_id
        if not correlation_id:
            logger.debug("Received ACK without correlation_id, ignoring")
            return
        
        if correlation_id not in self._pending_acks:
            # 可能是超时后收到的 ACK，或者是其他实例的 ACK
            logger.debug(f"Received ACK for unknown correlation_id: {correlation_id[:8]}...")
            return
        
        async with self._ack_lock:
            ack = SyncAck(
                message_id=correlation_id,
                instance_id=message.source_instance or "unknown",
                success=(message.message_type == MessageType.ACK),
                metadata=message.metadata,
            )
            
            if correlation_id not in self._ack_results:
                self._ack_results[correlation_id] = []
            self._ack_results[correlation_id].append(ack)
        
        # 设置事件（如果只需要一个 ACK）
        event = self._pending_acks.get(correlation_id)
        if event:
            event.set()
        
        self._stats["acks_received"] += 1
    
    async def publish(
        self,
        message: SyncMessage,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """
        发布消息
        
        Args:
            message: 同步消息
            wait_for_ack: 是否等待确认（覆盖默认设置）
        
        Returns:
            发布结果
        """
        wait_ack = wait_for_ack if wait_for_ack is not None else self.require_ack
        
        if wait_ack:
            message.require_ack = True
            self._pending_acks[message.message_id] = asyncio.Event()
        
        try:
            await self._backend.publish(self.channel, message)
            self._stats["messages_sent"] += 1
            
            result = {
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
            return {
                "success": False,
                "message_id": message.message_id,
                "error": str(e),
            }
    
    # ==================== 便捷方法 ====================
    
    async def publish_inject(
        self,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        condition: str,
        decision: str,
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布知识注入消息"""
        # 验证向量
        if not key_vector or not value_vector:
            return {
                "success": False,
                "error": "key_vector and value_vector cannot be empty",
            }
        
        message = SyncMessage.inject(
            lu_id=lu_id,
            key_vector=key_vector,
            value_vector=value_vector,
            condition=condition,
            decision=decision,
            namespace=namespace,
            lifecycle_state=lifecycle_state,
            trust_tier=trust_tier,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)
    
    async def publish_update(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布生命周期更新消息"""
        message = SyncMessage.update_lifecycle(
            lu_id=lu_id,
            new_state=new_state,
            namespace=namespace,
            reason=reason,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)
    
    async def publish_quarantine(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布隔离消息"""
        message = SyncMessage.quarantine(
            lu_id=lu_id,
            reason=reason,
            namespace=namespace,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)
    
    async def publish_delete(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布删除消息"""
        message = SyncMessage.delete(
            lu_id=lu_id,
            namespace=namespace,
            reason=reason,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)
    
    async def publish_batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        source_instance: Optional[str] = None,
        wait_for_ack: bool = None,
    ) -> Dict[str, Any]:
        """发布批量注入消息"""
        message = SyncMessage.batch_inject(
            items=items,
            namespace=namespace,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack)
    
    async def request_full_sync(
        self,
        namespace: str = "default",
        source_instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """请求全量同步"""
        message = SyncMessage.full_sync_request(
            namespace=namespace,
            source_instance=source_instance or "portal",
        )
        return await self.publish(message, wait_for_ack=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        connected = self._backend.is_connected
        return {
            **self._stats,
            "channel": self.channel,
            "connected": connected,
            "status": "healthy" if connected else "degraded",
            "pending_acks_count": len(self._pending_acks),
        }
