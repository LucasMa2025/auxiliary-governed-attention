"""
AGA 同步订阅器

Runtime 使用此组件订阅来自 Portal 的消息。
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime

from .protocol import SyncMessage, MessageType, SyncAck
from .backends import SyncBackend, create_backend

logger = logging.getLogger(__name__)


# 消息处理器类型
MessageHandler = Callable[[SyncMessage], None]


class SyncSubscriber:
    """
    同步订阅器
    
    Runtime 用于订阅来自 Portal 的同步消息。
    
    示例:
        subscriber = SyncSubscriber(
            backend_type="redis",
            redis_host="localhost",
            channel="aga:sync",
            instance_id="runtime-001",
        )
        await subscriber.connect()
        
        # 注册处理器
        subscriber.on_inject(handle_inject)
        subscriber.on_update(handle_update)
        subscriber.on_quarantine(handle_quarantine)
        
        # 开始监听
        await subscriber.start()
        
        # 停止
        await subscriber.stop()
    """
    
    def __init__(
        self,
        backend_type: str = "redis",
        channel: str = "aga:sync",
        instance_id: str = "runtime-unknown",
        send_ack: bool = True,
        **backend_kwargs
    ):
        """
        初始化订阅器
        
        Args:
            backend_type: 后端类型 (memory, redis, kafka)
            channel: 订阅通道
            instance_id: Runtime 实例 ID
            send_ack: 是否发送 ACK
            **backend_kwargs: 后端配置
        """
        self.channel = channel
        self.instance_id = instance_id
        self.send_ack = send_ack
        
        self._backend = create_backend(backend_type, **backend_kwargs)
        self._handlers: Dict[MessageType, List[MessageHandler]] = {}
        self._default_handler: Optional[MessageHandler] = None
        self._running = False
        
        # 统计
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "acks_sent": 0,
            "errors": 0,
            "by_type": {},
        }
    
    async def connect(self):
        """连接到后端"""
        await self._backend.connect()
        logger.info(f"SyncSubscriber connected, channel={self.channel}, instance={self.instance_id}")
    
    async def disconnect(self):
        """断开连接"""
        self._running = False
        await self._backend.disconnect()
        logger.info("SyncSubscriber disconnected")
    
    async def start(self):
        """开始监听消息"""
        if self._running:
            return
        
        self._running = True
        await self._backend.subscribe(self.channel, self._handle_message)
        logger.info(f"SyncSubscriber started listening on {self.channel}")
    
    async def stop(self):
        """停止监听"""
        self._running = False
        await self._backend.unsubscribe(self.channel)
        logger.info("SyncSubscriber stopped")
    
    async def _handle_message(self, message: SyncMessage):
        """处理收到的消息"""
        self._stats["messages_received"] += 1
        
        msg_type = message.message_type
        type_name = msg_type.value if isinstance(msg_type, MessageType) else msg_type
        
        # 更新类型统计
        if type_name not in self._stats["by_type"]:
            self._stats["by_type"][type_name] = 0
        self._stats["by_type"][type_name] += 1
        
        logger.debug(f"Received message: {type_name}, id={message.message_id}")
        
        try:
            # 获取处理器
            handlers = self._handlers.get(msg_type, [])
            
            if handlers:
                for handler in handlers:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
            elif self._default_handler:
                if asyncio.iscoroutinefunction(self._default_handler):
                    await self._default_handler(message)
                else:
                    self._default_handler(message)
            else:
                logger.warning(f"No handler for message type: {type_name}")
            
            self._stats["messages_processed"] += 1
            
            # 发送 ACK
            if self.send_ack and message.require_ack:
                await self._send_ack(message, success=True)
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Message handling error: {e}")
            
            # 发送 NACK
            if self.send_ack and message.require_ack:
                await self._send_ack(message, success=False, error=str(e))
    
    async def _send_ack(
        self,
        original_message: SyncMessage,
        success: bool = True,
        error: Optional[str] = None
    ):
        """发送 ACK/NACK"""
        ack_message = SyncMessage(
            message_type=MessageType.ACK if success else MessageType.NACK,
            correlation_id=original_message.message_id,
            source_instance=self.instance_id,
            metadata={
                "success": success,
                "error": error,
            } if error else {"success": success},
        )
        
        ack_channel = f"{self.channel}:ack"
        await self._backend.publish(ack_channel, ack_message)
        self._stats["acks_sent"] += 1
    
    # ==================== 处理器注册 ====================
    
    def on(self, message_type: MessageType, handler: MessageHandler):
        """
        注册消息处理器
        
        Args:
            message_type: 消息类型
            handler: 处理函数
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
        return handler
    
    def on_default(self, handler: MessageHandler):
        """注册默认处理器"""
        self._default_handler = handler
        return handler
    
    def on_inject(self, handler: MessageHandler):
        """注册 INJECT 处理器"""
        return self.on(MessageType.INJECT, handler)
    
    def on_update(self, handler: MessageHandler):
        """注册 UPDATE 处理器"""
        return self.on(MessageType.UPDATE, handler)
    
    def on_quarantine(self, handler: MessageHandler):
        """注册 QUARANTINE 处理器"""
        return self.on(MessageType.QUARANTINE, handler)
    
    def on_delete(self, handler: MessageHandler):
        """注册 DELETE 处理器"""
        return self.on(MessageType.DELETE, handler)
    
    def on_batch_inject(self, handler: MessageHandler):
        """注册 BATCH_INJECT 处理器"""
        return self.on(MessageType.BATCH_INJECT, handler)
    
    def on_full_sync(self, handler: MessageHandler):
        """注册 FULL_SYNC 处理器"""
        return self.on(MessageType.FULL_SYNC, handler)
    
    def on_heartbeat(self, handler: MessageHandler):
        """注册 HEARTBEAT 处理器"""
        return self.on(MessageType.HEARTBEAT, handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "channel": self.channel,
            "instance_id": self.instance_id,
            "connected": self._backend.is_connected,
            "running": self._running,
        }


class SyncSubscriberBuilder:
    """
    订阅器构建器（流式 API）
    
    示例:
        subscriber = (
            SyncSubscriberBuilder()
            .backend("redis", host="localhost")
            .channel("aga:sync")
            .instance("runtime-001")
            .on_inject(handle_inject)
            .on_update(handle_update)
            .build()
        )
    """
    
    def __init__(self):
        self._backend_type = "memory"
        self._backend_kwargs = {}
        self._channel = "aga:sync"
        self._instance_id = "runtime-unknown"
        self._send_ack = True
        self._handlers: Dict[MessageType, List[MessageHandler]] = {}
        self._default_handler: Optional[MessageHandler] = None
    
    def backend(self, backend_type: str, **kwargs) -> "SyncSubscriberBuilder":
        """设置后端"""
        self._backend_type = backend_type
        self._backend_kwargs = kwargs
        return self
    
    def channel(self, channel: str) -> "SyncSubscriberBuilder":
        """设置通道"""
        self._channel = channel
        return self
    
    def instance(self, instance_id: str) -> "SyncSubscriberBuilder":
        """设置实例 ID"""
        self._instance_id = instance_id
        return self
    
    def send_ack(self, enabled: bool = True) -> "SyncSubscriberBuilder":
        """设置是否发送 ACK"""
        self._send_ack = enabled
        return self
    
    def on_inject(self, handler: MessageHandler) -> "SyncSubscriberBuilder":
        """注册 INJECT 处理器"""
        if MessageType.INJECT not in self._handlers:
            self._handlers[MessageType.INJECT] = []
        self._handlers[MessageType.INJECT].append(handler)
        return self
    
    def on_update(self, handler: MessageHandler) -> "SyncSubscriberBuilder":
        """注册 UPDATE 处理器"""
        if MessageType.UPDATE not in self._handlers:
            self._handlers[MessageType.UPDATE] = []
        self._handlers[MessageType.UPDATE].append(handler)
        return self
    
    def on_quarantine(self, handler: MessageHandler) -> "SyncSubscriberBuilder":
        """注册 QUARANTINE 处理器"""
        if MessageType.QUARANTINE not in self._handlers:
            self._handlers[MessageType.QUARANTINE] = []
        self._handlers[MessageType.QUARANTINE].append(handler)
        return self
    
    def on_delete(self, handler: MessageHandler) -> "SyncSubscriberBuilder":
        """注册 DELETE 处理器"""
        if MessageType.DELETE not in self._handlers:
            self._handlers[MessageType.DELETE] = []
        self._handlers[MessageType.DELETE].append(handler)
        return self
    
    def on_default(self, handler: MessageHandler) -> "SyncSubscriberBuilder":
        """注册默认处理器"""
        self._default_handler = handler
        return self
    
    def build(self) -> SyncSubscriber:
        """构建订阅器"""
        subscriber = SyncSubscriber(
            backend_type=self._backend_type,
            channel=self._channel,
            instance_id=self._instance_id,
            send_ack=self._send_ack,
            **self._backend_kwargs,
        )
        
        # 注册处理器
        for msg_type, handlers in self._handlers.items():
            for handler in handlers:
                subscriber.on(msg_type, handler)
        
        if self._default_handler:
            subscriber.on_default(self._default_handler)
        
        return subscriber
