"""
AGA 同步协议模块

提供 Portal 与 Runtime 之间的消息同步机制。

组件说明：
- SyncMessage: 同步消息协议
- SyncPublisher: 消息发布器（Portal 使用）
- SyncSubscriber: 消息订阅器（Runtime 使用）
- RedisBackend: Redis Pub/Sub 实现
- KafkaBackend: Kafka 实现（可选）
- MemoryBackend: 内存实现（测试用）
"""

from .protocol import (
    SyncMessage,
    MessageType,
    SyncAck,
)
from .publisher import SyncPublisher
from .subscriber import SyncSubscriber
from .backends import (
    SyncBackend,
    RedisBackend,
    MemoryBackend,
    create_backend,
)

__all__ = [
    # 协议
    "SyncMessage",
    "MessageType",
    "SyncAck",
    # 发布/订阅
    "SyncPublisher",
    "SyncSubscriber",
    # 后端
    "SyncBackend",
    "RedisBackend",
    "MemoryBackend",
    "create_backend",
]
