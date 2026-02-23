"""
aga-knowledge 同步后端实现

提供不同的消息队列后端（明文 KV 版本）。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, List
from collections import deque

from .protocol import SyncMessage

logger = logging.getLogger(__name__)


class SyncBackend(ABC):
    """同步后端抽象基类"""

    @abstractmethod
    async def connect(self):
        ...

    @abstractmethod
    async def disconnect(self):
        ...

    @abstractmethod
    async def publish(self, channel: str, message: SyncMessage):
        ...

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        ...

    @abstractmethod
    async def unsubscribe(self, channel: str):
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...


class MemoryBackend(SyncBackend):
    """
    内存后端（测试用）

    用于单进程内的测试，不支持跨进程通信。
    """

    _channels: Dict[str, List[Callable]] = {}
    _message_queue: Dict[str, deque] = {}

    def __init__(self):
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False
        for channel in list(self._subscriptions.keys()):
            await self.unsubscribe(channel)

    async def publish(self, channel: str, message: SyncMessage):
        if not self._connected:
            raise RuntimeError("Not connected")
        if channel not in MemoryBackend._message_queue:
            MemoryBackend._message_queue[channel] = deque(maxlen=1000)
        MemoryBackend._message_queue[channel].append(message)

        if channel in MemoryBackend._channels:
            for callback in MemoryBackend._channels[channel]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        if not self._connected:
            raise RuntimeError("Not connected")
        if channel not in MemoryBackend._channels:
            MemoryBackend._channels[channel] = []
        MemoryBackend._channels[channel].append(callback)
        self._subscriptions[channel] = callback

    async def unsubscribe(self, channel: str):
        if channel in self._subscriptions:
            callback = self._subscriptions.pop(channel)
            if channel in MemoryBackend._channels:
                try:
                    MemoryBackend._channels[channel].remove(callback)
                except ValueError:
                    pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @classmethod
    def clear_all(cls):
        """清空所有数据（测试用）"""
        cls._channels.clear()
        cls._message_queue.clear()


class RedisBackend(SyncBackend):
    """
    Redis Pub/Sub 后端

    需要安装: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        **kwargs,
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
            host=self.host, port=self.port, db=self.db,
            password=self.password, decode_responses=True,
            **self.kwargs,
        )
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

    async def publish(self, channel: str, message: SyncMessage):
        if not self._redis:
            raise RuntimeError("Not connected")
        await self._redis.publish(channel, message.to_json())

    async def subscribe(self, channel: str, callback: Callable[[SyncMessage], None]):
        if not self._pubsub:
            raise RuntimeError("Not connected")
        self._subscriptions[channel] = callback
        await self._pubsub.subscribe(channel)
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: str):
        if channel in self._subscriptions:
            del self._subscriptions[channel]
            if self._pubsub:
                await self._pubsub.unsubscribe(channel)

    async def _listen(self):
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


def create_backend(backend_type: str, **kwargs) -> SyncBackend:
    """
    创建同步后端

    Args:
        backend_type: "memory" | "redis"
        **kwargs: 后端配置参数

    Returns:
        SyncBackend 实例
    """
    backends = {
        "memory": MemoryBackend,
        "redis": RedisBackend,
    }
    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")
    return backends[backend_type](**kwargs)
