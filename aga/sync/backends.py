"""
AGA 同步后端实现

提供不同的消息队列后端。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, List
from collections import deque

from .protocol import SyncMessage, SyncAck

logger = logging.getLogger(__name__)


class SyncBackend(ABC):
    """同步后端抽象基类"""
    
    @abstractmethod
    async def connect(self):
        """连接到后端"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def publish(self, channel: str, message: SyncMessage):
        """发布消息"""
        pass
    
    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        """订阅消息"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str):
        """取消订阅"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""
        pass


class MemoryBackend(SyncBackend):
    """
    内存后端（测试用）
    
    用于单进程内的测试，不支持跨进程通信。
    """
    
    # 全局消息队列（模拟 Pub/Sub）
    _channels: Dict[str, List[Callable]] = {}
    _message_queue: Dict[str, deque] = {}
    
    def __init__(self):
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}
    
    async def connect(self):
        self._connected = True
        logger.info("MemoryBackend connected")
    
    async def disconnect(self):
        self._connected = False
        for channel in list(self._subscriptions.keys()):
            await self.unsubscribe(channel)
        logger.info("MemoryBackend disconnected")
    
    async def publish(self, channel: str, message: SyncMessage):
        if not self._connected:
            raise RuntimeError("Not connected")
        
        # 存储到队列
        if channel not in MemoryBackend._message_queue:
            MemoryBackend._message_queue[channel] = deque(maxlen=1000)
        MemoryBackend._message_queue[channel].append(message)
        
        # 通知订阅者
        if channel in MemoryBackend._channels:
            for callback in MemoryBackend._channels[channel]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        logger.debug(f"Published to {channel}: {message.message_type}")
    
    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        if not self._connected:
            raise RuntimeError("Not connected")
        
        if channel not in MemoryBackend._channels:
            MemoryBackend._channels[channel] = []
        
        MemoryBackend._channels[channel].append(callback)
        self._subscriptions[channel] = callback
        
        logger.info(f"Subscribed to {channel}")
    
    async def unsubscribe(self, channel: str):
        if channel in self._subscriptions:
            callback = self._subscriptions.pop(channel)
            if channel in MemoryBackend._channels:
                try:
                    MemoryBackend._channels[channel].remove(callback)
                except ValueError:
                    pass
        logger.info(f"Unsubscribed from {channel}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def get_pending_messages(self, channel: str) -> List[SyncMessage]:
        """获取待处理消息（测试用）"""
        return list(MemoryBackend._message_queue.get(channel, []))
    
    @classmethod
    def clear_all(cls):
        """清空所有数据（测试用）"""
        cls._channels.clear()
        cls._message_queue.clear()


class RedisBackend(SyncBackend):
    """
    Redis Pub/Sub 后端
    
    推荐用于生产环境，低延迟。
    
    需要安装: pip install redis
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        **kwargs
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.kwargs = kwargs
        
        self._redis = None
        self._pubsub = None
        self._subscriptions: Dict[str, Callable] = {}
        self._listener_task = None
    
    async def connect(self):
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("需要安装 redis: pip install redis")
        
        self._redis = aioredis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
            **self.kwargs
        )
        
        # 测试连接
        await self._redis.ping()
        
        self._pubsub = self._redis.pubsub()
        logger.info(f"RedisBackend connected to {self.host}:{self.port}")
    
    async def disconnect(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
        
        self._redis = None
        self._pubsub = None
        logger.info("RedisBackend disconnected")
    
    async def publish(self, channel: str, message: SyncMessage):
        if not self._redis:
            raise RuntimeError("Not connected")
        
        await self._redis.publish(channel, message.to_json())
        logger.debug(f"Published to {channel}: {message.message_type}")
    
    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        if not self._pubsub:
            raise RuntimeError("Not connected")
        
        self._subscriptions[channel] = callback
        await self._pubsub.subscribe(channel)
        
        # 启动监听任务
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())
        
        logger.info(f"Subscribed to {channel}")
    
    async def unsubscribe(self, channel: str):
        if channel in self._subscriptions:
            del self._subscriptions[channel]
            if self._pubsub:
                await self._pubsub.unsubscribe(channel)
        logger.info(f"Unsubscribed from {channel}")
    
    async def _listen(self):
        """监听消息"""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    
                    if channel in self._subscriptions:
                        try:
                            sync_msg = SyncMessage.from_json(message["data"])
                            callback = self._subscriptions[channel]
                            
                            if asyncio.iscoroutinefunction(callback):
                                await callback(sync_msg)
                            else:
                                callback(sync_msg)
                        except Exception as e:
                            logger.error(f"Message handling error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Listener error: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._redis is not None


class KafkaBackend(SyncBackend):
    """
    Kafka 后端
    
    适合大规模部署，支持消息持久化和回放。
    
    需要安装: pip install aiokafka
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "aga-consumers",
        **kwargs
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.kwargs = kwargs
        
        self._producer = None
        self._consumer = None
        self._subscriptions: Dict[str, Callable] = {}
        self._listener_task = None
    
    async def connect(self):
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
        except ImportError:
            raise ImportError("需要安装 aiokafka: pip install aiokafka")
        
        # 创建生产者
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
        )
        await self._producer.start()
        
        logger.info(f"KafkaBackend connected to {self.bootstrap_servers}")
    
    async def disconnect(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._consumer:
            await self._consumer.stop()
        
        if self._producer:
            await self._producer.stop()
        
        self._producer = None
        self._consumer = None
        logger.info("KafkaBackend disconnected")
    
    async def publish(self, channel: str, message: SyncMessage):
        if not self._producer:
            raise RuntimeError("Not connected")
        
        await self._producer.send(channel, message.to_json())
        logger.debug(f"Published to {channel}: {message.message_type}")
    
    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError("需要安装 aiokafka: pip install aiokafka")
        
        self._subscriptions[channel] = callback
        
        # 创建消费者
        if not self._consumer:
            self._consumer = AIOKafkaConsumer(
                channel,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: v.decode() if isinstance(v, bytes) else v,
            )
            await self._consumer.start()
            
            # 启动监听任务
            self._listener_task = asyncio.create_task(self._listen())
        
        logger.info(f"Subscribed to {channel}")
    
    async def unsubscribe(self, channel: str):
        if channel in self._subscriptions:
            del self._subscriptions[channel]
        logger.info(f"Unsubscribed from {channel}")
    
    async def _listen(self):
        """监听消息"""
        try:
            async for message in self._consumer:
                topic = message.topic
                
                if topic in self._subscriptions:
                    try:
                        sync_msg = SyncMessage.from_json(message.value)
                        callback = self._subscriptions[topic]
                        
                        if asyncio.iscoroutinefunction(callback):
                            await callback(sync_msg)
                        else:
                            callback(sync_msg)
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Listener error: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._producer is not None


def create_backend(
    backend_type: str,
    **kwargs
) -> SyncBackend:
    """
    创建同步后端
    
    Args:
        backend_type: 后端类型 (memory, redis, kafka)
        **kwargs: 后端配置参数
    
    Returns:
        SyncBackend 实例
    
    示例:
        # 内存后端
        backend = create_backend("memory")
        
        # Redis 后端
        backend = create_backend("redis", host="localhost", port=6379)
        
        # Kafka 后端
        backend = create_backend("kafka", bootstrap_servers="localhost:9092")
    """
    backends = {
        "memory": MemoryBackend,
        "redis": RedisBackend,
        "kafka": KafkaBackend,
    }
    
    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type](**kwargs)
