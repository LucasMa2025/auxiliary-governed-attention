"""
aga-knowledge 同步协议模块

提供 Portal 与 Runtime 之间的消息同步机制（明文 KV 版本）。

组件说明：
- SyncMessage: 同步消息协议
- SyncPublisher: 消息发布器（Portal 使用）
- SyncSubscriber: 消息订阅器（Runtime 使用）
- MemoryBackend: 内存实现（测试用）
- RedisBackend: Redis Pub/Sub 实现
"""

from .protocol import SyncMessage, MessageType, SyncAck
from .publisher import SyncPublisher
from .backends import SyncBackend, MemoryBackend, RedisBackend, create_backend

__all__ = [
    "SyncMessage",
    "MessageType",
    "SyncAck",
    "SyncPublisher",
    "SyncBackend",
    "MemoryBackend",
    "RedisBackend",
    "create_backend",
]
