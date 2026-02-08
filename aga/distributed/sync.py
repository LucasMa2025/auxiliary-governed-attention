"""
AGA 分布式同步器

实现多实例间的知识同步。

版本: v3.1

v3.1 更新：
- 集成治理裁决器
- 集成传播节流器
- 支持语义主权分区
- 实现"少数即生效"的隔离机制

核心原则：
1. 错误传播速度必须 < 隔离速度
2. 默认拒绝传播未注册的知识
3. 治理指令优先于同步指令
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
import logging

from ..types import LifecycleState

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """同步消息类型"""
    # 知识同步
    KNOWLEDGE_INJECT = "knowledge_inject"
    KNOWLEDGE_UPDATE = "knowledge_update"
    LIFECYCLE_UPDATE = "lifecycle_update"
    BATCH_SYNC = "batch_sync"
    
    # 治理消息（v3.1）
    QUARANTINE = "quarantine"
    QUARANTINE_VOTE = "quarantine_vote"      # 隔离投票
    PROPAGATION_BLOCK = "propagation_block"  # 传播阻止
    TRUST_UPDATE = "trust_update"            # 信任层级更新
    APPROVAL_REQUEST = "approval_request"    # 审批请求
    APPROVAL_RESPONSE = "approval_response"  # 审批响应
    
    # 实例管理
    HEARTBEAT = "heartbeat"
    INSTANCE_JOIN = "instance_join"
    INSTANCE_LEAVE = "instance_leave"


@dataclass
class SyncMessage:
    """
    同步消息
    
    用于实例间的知识同步通信。
    """
    message_type: MessageType
    namespace: str
    instance_id: str
    timestamp: float = field(default_factory=time.time)
    
    # 知识相关
    lu_id: Optional[str] = None
    slot_idx: Optional[int] = None
    key_vector: Optional[List[float]] = None
    value_vector: Optional[List[float]] = None
    lifecycle_state: Optional[str] = None
    condition: Optional[str] = None
    decision: Optional[str] = None
    
    # 批量同步
    records: Optional[List[Dict[str, Any]]] = None
    
    # 治理相关（v3.1）
    trust_tier: Optional[str] = None          # 信任层级
    propagation_policy: Optional[str] = None  # 传播策略
    quarantine_reason: Optional[str] = None   # 隔离原因
    approval_id: Optional[str] = None         # 审批 ID
    approval_result: Optional[bool] = None    # 审批结果
    priority: int = 0                         # 消息优先级（治理消息优先）
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        data = {
            "message_type": self.message_type.value,
            "namespace": self.namespace,
            "instance_id": self.instance_id,
            "timestamp": self.timestamp,
        }
        
        if self.lu_id:
            data["lu_id"] = self.lu_id
        if self.slot_idx is not None:
            data["slot_idx"] = self.slot_idx
        if self.key_vector is not None:
            data["key_vector"] = self.key_vector
        if self.value_vector is not None:
            data["value_vector"] = self.value_vector
        if self.lifecycle_state:
            data["lifecycle_state"] = self.lifecycle_state
        if self.condition:
            data["condition"] = self.condition
        if self.decision:
            data["decision"] = self.decision
        if self.records is not None:
            data["records"] = self.records
        if self.trust_tier:
            data["trust_tier"] = self.trust_tier
        if self.propagation_policy:
            data["propagation_policy"] = self.propagation_policy
        if self.quarantine_reason:
            data["quarantine_reason"] = self.quarantine_reason
        if self.approval_id:
            data["approval_id"] = self.approval_id
        if self.approval_result is not None:
            data["approval_result"] = self.approval_result
        if self.priority:
            data["priority"] = self.priority
        if self.metadata:
            data["metadata"] = self.metadata
        
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SyncMessage":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        return cls(
            message_type=MessageType(data["message_type"]),
            namespace=data["namespace"],
            instance_id=data["instance_id"],
            timestamp=data.get("timestamp", time.time()),
            lu_id=data.get("lu_id"),
            slot_idx=data.get("slot_idx"),
            key_vector=data.get("key_vector"),
            value_vector=data.get("value_vector"),
            lifecycle_state=data.get("lifecycle_state"),
            condition=data.get("condition"),
            decision=data.get("decision"),
            records=data.get("records"),
            trust_tier=data.get("trust_tier"),
            propagation_policy=data.get("propagation_policy"),
            quarantine_reason=data.get("quarantine_reason"),
            approval_id=data.get("approval_id"),
            approval_result=data.get("approval_result"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata"),
        )


class DistributedSynchronizer:
    """
    分布式同步器
    
    负责多实例间的知识同步，支持：
    - Kafka 消息总线
    - Redis Pub/Sub
    - 本地事件（单机多进程）
    """
    
    def __init__(
        self,
        instance_id: str,
        namespace: str = "default",
        backend: str = "redis",  # redis, kafka, local
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化同步器
        
        Args:
            instance_id: 实例 ID
            namespace: 命名空间
            backend: 后端类型
            config: 后端配置
        """
        self.instance_id = instance_id
        self.namespace = namespace
        self.backend = backend
        self.config = config or {}
        
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._running = False
        self._client = None
    
    async def start(self) -> bool:
        """启动同步器"""
        try:
            if self.backend == "redis":
                await self._start_redis()
            elif self.backend == "kafka":
                await self._start_kafka()
            else:
                await self._start_local()
            
            self._running = True
            logger.info(f"Synchronizer started: instance={self.instance_id}, backend={self.backend}")
            
            # 发送加入消息
            await self.publish(SyncMessage(
                message_type=MessageType.INSTANCE_JOIN,
                namespace=self.namespace,
                instance_id=self.instance_id,
            ))
            
            return True
        except Exception as e:
            logger.error(f"Failed to start synchronizer: {e}")
            return False
    
    async def stop(self):
        """停止同步器"""
        if not self._running:
            return
        
        # 发送离开消息
        await self.publish(SyncMessage(
            message_type=MessageType.INSTANCE_LEAVE,
            namespace=self.namespace,
            instance_id=self.instance_id,
        ))
        
        self._running = False
        
        if self._client:
            if self.backend == "redis":
                await self._client.close()
            elif self.backend == "kafka":
                await self._client.stop()
        
        logger.info(f"Synchronizer stopped: instance={self.instance_id}")
    
    async def _start_redis(self):
        """启动 Redis 后端"""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package")
        
        self._client = aioredis.Redis(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 6379),
            db=self.config.get("db", 0),
            password=self.config.get("password"),
        )
        
        # 启动订阅
        asyncio.create_task(self._redis_subscribe())
    
    async def _redis_subscribe(self):
        """Redis 订阅循环"""
        channel = f"aga:{self.namespace}:sync"
        pubsub = self._client.pubsub()
        await pubsub.subscribe(channel)
        
        async for message in pubsub.listen():
            if not self._running:
                break
            
            if message["type"] == "message":
                try:
                    sync_msg = SyncMessage.from_json(message["data"])
                    
                    # 忽略自己的消息
                    if sync_msg.instance_id == self.instance_id:
                        continue
                    
                    await self._handle_message(sync_msg)
                except Exception as e:
                    logger.error(f"Failed to handle message: {e}")
    
    async def _start_kafka(self):
        """启动 Kafka 后端"""
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
        except ImportError:
            raise ImportError("Kafka backend requires 'aiokafka' package")
        
        bootstrap_servers = self.config.get("bootstrap_servers", "localhost:9092")
        topic = f"aga-{self.namespace}-sync"
        
        # 生产者
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
        )
        await self._producer.start()
        
        # 消费者
        self._consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=f"aga-{self.instance_id}",
        )
        await self._consumer.start()
        
        # 启动消费循环
        asyncio.create_task(self._kafka_consume())
        
        self._client = self._producer
    
    async def _kafka_consume(self):
        """Kafka 消费循环"""
        async for msg in self._consumer:
            if not self._running:
                break
            
            try:
                sync_msg = SyncMessage.from_json(msg.value.decode())
                
                if sync_msg.instance_id == self.instance_id:
                    continue
                
                await self._handle_message(sync_msg)
            except Exception as e:
                logger.error(f"Failed to handle Kafka message: {e}")
    
    async def _start_local(self):
        """启动本地后端（用于测试）"""
        self._local_queue = asyncio.Queue()
        asyncio.create_task(self._local_consume())
    
    async def _local_consume(self):
        """本地消费循环"""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._local_queue.get(), timeout=1.0)
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                continue
    
    async def _handle_message(self, message: SyncMessage):
        """处理同步消息"""
        handlers = self._handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    def register_handler(
        self, 
        message_type: MessageType, 
        handler: Callable[[SyncMessage], Any]
    ):
        """注册消息处理器"""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    async def publish(self, message: SyncMessage) -> bool:
        """发布同步消息"""
        if not self._running and message.message_type not in (
            MessageType.INSTANCE_JOIN, MessageType.INSTANCE_LEAVE
        ):
            return False
        
        try:
            if self.backend == "redis":
                channel = f"aga:{self.namespace}:sync"
                await self._client.publish(channel, message.to_json())
            elif self.backend == "kafka":
                topic = f"aga-{self.namespace}-sync"
                await self._producer.send(topic, message.to_json().encode())
            else:
                await self._local_queue.put(message)
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    # ==================== 便捷方法 ====================
    
    async def sync_knowledge_inject(
        self,
        lu_id: str,
        slot_idx: int,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: LifecycleState,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> bool:
        """同步知识注入"""
        return await self.publish(SyncMessage(
            message_type=MessageType.KNOWLEDGE_INJECT,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            slot_idx=slot_idx,
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state=lifecycle_state.value,
            condition=condition,
            decision=decision,
        ))
    
    async def sync_lifecycle_update(
        self,
        lu_id: str,
        new_state: LifecycleState,
    ) -> bool:
        """同步生命周期更新"""
        return await self.publish(SyncMessage(
            message_type=MessageType.LIFECYCLE_UPDATE,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            lifecycle_state=new_state.value,
        ))
    
    async def sync_quarantine(self, lu_id: str) -> bool:
        """同步隔离"""
        return await self.publish(SyncMessage(
            message_type=MessageType.QUARANTINE,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
        ))
    
    async def sync_batch(self, records: List[Dict[str, Any]]) -> bool:
        """批量同步"""
        return await self.publish(SyncMessage(
            message_type=MessageType.BATCH_SYNC,
            namespace=self.namespace,
            instance_id=self.instance_id,
            records=records,
        ))
    
    # ==================== 治理方法（v3.1） ====================
    
    async def broadcast_quarantine_vote(
        self,
        lu_id: str,
        reason: str,
    ) -> bool:
        """
        广播隔离投票
        
        实现"少数即生效"原则：任何实例都可以发起隔离投票
        """
        return await self.publish(SyncMessage(
            message_type=MessageType.QUARANTINE_VOTE,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            quarantine_reason=reason,
            priority=10,  # 高优先级
        ))
    
    async def broadcast_propagation_block(
        self,
        lu_id: str,
        reason: str,
    ) -> bool:
        """
        广播传播阻止
        
        立即阻止该知识的进一步传播
        """
        return await self.publish(SyncMessage(
            message_type=MessageType.PROPAGATION_BLOCK,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            quarantine_reason=reason,
            priority=10,  # 高优先级
        ))
    
    async def request_approval(
        self,
        lu_id: str,
        approval_id: str,
        trust_tier: str,
    ) -> bool:
        """请求传播审批（用于 GATED 策略）"""
        return await self.publish(SyncMessage(
            message_type=MessageType.APPROVAL_REQUEST,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            approval_id=approval_id,
            trust_tier=trust_tier,
        ))
    
    async def respond_approval(
        self,
        lu_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        """响应传播审批"""
        return await self.publish(SyncMessage(
            message_type=MessageType.APPROVAL_RESPONSE,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            approval_id=approval_id,
            approval_result=approved,
        ))
    
    async def update_trust_tier(
        self,
        lu_id: str,
        trust_tier: str,
        propagation_policy: str,
    ) -> bool:
        """更新信任层级"""
        return await self.publish(SyncMessage(
            message_type=MessageType.TRUST_UPDATE,
            namespace=self.namespace,
            instance_id=self.instance_id,
            lu_id=lu_id,
            trust_tier=trust_tier,
            propagation_policy=propagation_policy,
        ))
