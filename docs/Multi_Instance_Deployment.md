# AGA 多实例部署改造方案

**版本**: v1.0  
**日期**: 2026-02-01

---

## 目录

1. [背景与挑战](#1-背景与挑战)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [实施方案](#4-实施方案)
5. [一致性保证](#5-一致性保证)
6. [故障处理](#6-故障处理)
7. [监控与运维](#7-监控与运维)

---

## 1. 背景与挑战

### 1.1 当前单机架构限制

```
┌─────────────────────────────────────────────────────────────┐
│                    单机 AGA 架构                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   LLM 1     │    │   LLM 2     │    │   LLM 3     │     │
│  │   + AGA     │    │   + AGA     │    │   + AGA     │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │   SQLite DB   │  ← 单点瓶颈             │
│                    └───────────────┘                        │
└─────────────────────────────────────────────────────────────┘

问题：
1. 知识不同步：实例 A 注入的知识，实例 B 不可见
2. 无法水平扩展：单 DB 成为瓶颈
3. 单点故障：DB 故障导致全部实例不可用
4. 状态不一致：隔离操作无法即时生效
```

### 1.2 多实例部署目标

| 目标 | 描述 | 优先级 |
|-----|------|-------|
| **知识同步** | 任意实例的知识变更在 <1s 内同步到所有实例 | P0 |
| **水平扩展** | 支持 10+ 实例无性能退化 | P0 |
| **高可用** | 单实例故障不影响整体服务 | P0 |
| **一致性** | 最终一致性，关键操作（隔离）强一致 | P1 |
| **可观测** | 全链路追踪，跨实例诊断 | P1 |

---

## 2. 架构设计

### 2.1 目标架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGA 多实例分布式架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   LLM 1     │    │   LLM 2     │    │   LLM 3     │    │   LLM N     │  │
│  │   + AGA     │    │   + AGA     │    │   + AGA     │    │   + AGA     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┼──────────────────┼──────────────────┘          │
│                            │                  │                             │
│  ┌─────────────────────────┼──────────────────┼─────────────────────────┐   │
│  │                    Message Bus (Kafka)                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ knowledge   │  │ lifecycle   │  │ quarantine  │                   │   │
│  │  │ _changes    │  │ _updates    │  │ _commands   │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                            │                  │                             │
│         ┌──────────────────┼──────────────────┼──────────────────┐          │
│         │                  │                  │                  │          │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐  │
│  │ Redis Cluster│    │ PostgreSQL │    │ Vector DB  │    │ Prometheus  │  │
│  │ (Hot Cache)  │    │ (Primary)  │    │ (Optional) │    │ (Metrics)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **事件驱动同步**：所有知识变更通过 Kafka 广播
2. **本地缓存优先**：每个实例维护本地热缓存
3. **最终一致性**：普通操作允许短暂不一致
4. **关键操作强一致**：隔离命令使用分布式锁

### 2.3 数据流设计

```
知识注入流程：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Client  │───▶│ LLM+AGA │───▶│  Kafka  │───▶│ 其他实例 │
└─────────┘    └────┬────┘    └────┬────┘    └─────────┘
                    │              │
                    ▼              ▼
              ┌─────────┐    ┌─────────┐
              │  Redis  │    │PostgreSQL│
              └─────────┘    └─────────┘

隔离命令流程（强一致）：
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Client  │───▶│ Redis   │───▶│  Kafka  │───▶│ 所有实例 │
│         │    │ Lock    │    │ (sync)  │    │ (ack)   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

---

## 3. 核心组件

### 3.1 分布式同步器 (DistributedSynchronizer)

```python
"""
分布式同步器

负责跨实例的知识同步和状态一致性。
"""
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型"""
    KNOWLEDGE_INJECT = "knowledge_inject"
    KNOWLEDGE_UPDATE = "knowledge_update"
    LIFECYCLE_UPDATE = "lifecycle_update"
    QUARANTINE_COMMAND = "quarantine_command"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"


@dataclass
class SyncMessage:
    """同步消息"""
    message_type: MessageType
    namespace: str
    lu_id: Optional[str]
    payload: Dict[str, Any]
    source_instance: str
    timestamp: datetime
    sequence_number: int
    require_ack: bool = False
    
    def to_json(self) -> str:
        return json.dumps({
            'message_type': self.message_type.value,
            'namespace': self.namespace,
            'lu_id': self.lu_id,
            'payload': self.payload,
            'source_instance': self.source_instance,
            'timestamp': self.timestamp.isoformat(),
            'sequence_number': self.sequence_number,
            'require_ack': self.require_ack,
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'SyncMessage':
        d = json.loads(data)
        return cls(
            message_type=MessageType(d['message_type']),
            namespace=d['namespace'],
            lu_id=d.get('lu_id'),
            payload=d['payload'],
            source_instance=d['source_instance'],
            timestamp=datetime.fromisoformat(d['timestamp']),
            sequence_number=d['sequence_number'],
            require_ack=d.get('require_ack', False),
        )


class DistributedSynchronizer:
    """
    分布式同步器
    
    功能：
    - 知识变更广播
    - 消息消费和本地应用
    - 序列号追踪（检测丢失）
    - 心跳和健康检查
    """
    
    def __init__(
        self,
        instance_id: str,
        kafka_config: Dict[str, Any],
        redis_config: Dict[str, Any],
        aga_manager: 'AGAManager',
    ):
        self.instance_id = instance_id
        self.aga_manager = aga_manager
        
        # Kafka 生产者/消费者
        self.producer = None
        self.consumer = None
        self.kafka_config = kafka_config
        
        # Redis 客户端（用于分布式锁）
        self.redis = None
        self.redis_config = redis_config
        
        # 序列号追踪
        self._sequence_number = 0
        self._received_sequences: Dict[str, int] = {}  # instance_id -> last_seq
        
        # 消息处理器
        self._handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()
        
        # 状态
        self._running = False
    
    async def start(self):
        """启动同步器"""
        from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
        import aioredis
        
        # 初始化 Kafka
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: v.encode('utf-8'),
        )
        await self.producer.start()
        
        self.consumer = AIOKafkaConsumer(
            'aga_sync',
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            group_id=f'aga_sync_{self.instance_id}',
            value_deserializer=lambda v: v.decode('utf-8'),
        )
        await self.consumer.start()
        
        # 初始化 Redis
        self.redis = await aioredis.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}"
        )
        
        self._running = True
        
        # 启动消费循环
        asyncio.create_task(self._consume_loop())
        
        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"DistributedSynchronizer started: {self.instance_id}")
    
    async def stop(self):
        """停止同步器"""
        self._running = False
        
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        if self.redis:
            await self.redis.close()
        
        logger.info(f"DistributedSynchronizer stopped: {self.instance_id}")
    
    async def broadcast_knowledge_inject(
        self,
        namespace: str,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: str,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ):
        """广播知识注入"""
        message = SyncMessage(
            message_type=MessageType.KNOWLEDGE_INJECT,
            namespace=namespace,
            lu_id=lu_id,
            payload={
                'key_vector': key_vector,
                'value_vector': value_vector,
                'lifecycle_state': lifecycle_state,
                'condition': condition,
                'decision': decision,
            },
            source_instance=self.instance_id,
            timestamp=datetime.now(),
            sequence_number=self._next_sequence(),
        )
        
        await self._send_message(message)
    
    async def broadcast_lifecycle_update(
        self,
        namespace: str,
        lu_id: str,
        new_state: str,
    ):
        """广播生命周期更新"""
        message = SyncMessage(
            message_type=MessageType.LIFECYCLE_UPDATE,
            namespace=namespace,
            lu_id=lu_id,
            payload={'new_state': new_state},
            source_instance=self.instance_id,
            timestamp=datetime.now(),
            sequence_number=self._next_sequence(),
        )
        
        await self._send_message(message)
    
    async def broadcast_quarantine(
        self,
        namespace: str,
        lu_id: str,
        reason: str,
    ) -> bool:
        """
        广播隔离命令（强一致）
        
        使用分布式锁确保所有实例同时执行。
        """
        lock_key = f"aga:quarantine:{namespace}:{lu_id}"
        
        # 获取分布式锁
        lock = await self.redis.lock(lock_key, timeout=30)
        
        try:
            async with lock:
                message = SyncMessage(
                    message_type=MessageType.QUARANTINE_COMMAND,
                    namespace=namespace,
                    lu_id=lu_id,
                    payload={'reason': reason},
                    source_instance=self.instance_id,
                    timestamp=datetime.now(),
                    sequence_number=self._next_sequence(),
                    require_ack=True,
                )
                
                await self._send_message(message)
                
                # 等待所有实例确认
                ack_count = await self._wait_for_acks(message, timeout=10)
                
                return ack_count > 0
        except Exception as e:
            logger.error(f"Quarantine broadcast failed: {e}")
            return False
    
    async def _send_message(self, message: SyncMessage):
        """发送消息到 Kafka"""
        await self.producer.send(
            'aga_sync',
            value=message.to_json(),
            key=message.namespace.encode('utf-8'),
        )
    
    async def _consume_loop(self):
        """消息消费循环"""
        while self._running:
            try:
                async for msg in self.consumer:
                    if not self._running:
                        break
                    
                    try:
                        sync_message = SyncMessage.from_json(msg.value)
                        
                        # 跳过自己发送的消息
                        if sync_message.source_instance == self.instance_id:
                            continue
                        
                        # 检查序列号
                        self._check_sequence(sync_message)
                        
                        # 处理消息
                        await self._handle_message(sync_message)
                        
                    except Exception as e:
                        logger.error(f"Failed to process message: {e}")
                        
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: SyncMessage):
        """处理消息"""
        handler = self._handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self._handlers[MessageType.KNOWLEDGE_INJECT] = self._handle_knowledge_inject
        self._handlers[MessageType.LIFECYCLE_UPDATE] = self._handle_lifecycle_update
        self._handlers[MessageType.QUARANTINE_COMMAND] = self._handle_quarantine
        self._handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
    
    async def _handle_knowledge_inject(self, message: SyncMessage):
        """处理知识注入消息"""
        import torch
        
        payload = message.payload
        operator = self.aga_manager.get_operator(message.namespace)
        
        operator.inject_knowledge(
            lu_id=message.lu_id,
            key_vector=torch.tensor(payload['key_vector']),
            value_vector=torch.tensor(payload['value_vector']),
            lifecycle_state=payload['lifecycle_state'],
            condition=payload.get('condition'),
            decision=payload.get('decision'),
        )
        
        logger.info(f"Applied knowledge inject from {message.source_instance}: {message.lu_id}")
    
    async def _handle_lifecycle_update(self, message: SyncMessage):
        """处理生命周期更新消息"""
        operator = self.aga_manager.get_operator(message.namespace)
        operator.update_lifecycle(message.lu_id, message.payload['new_state'])
        
        logger.info(f"Applied lifecycle update from {message.source_instance}: {message.lu_id}")
    
    async def _handle_quarantine(self, message: SyncMessage):
        """处理隔离命令"""
        operator = self.aga_manager.get_operator(message.namespace)
        operator.quarantine_knowledge(message.lu_id)
        
        logger.warning(
            f"Applied quarantine from {message.source_instance}: "
            f"{message.lu_id}, reason: {message.payload.get('reason')}"
        )
        
        # 发送确认
        if message.require_ack:
            await self._send_ack(message)
    
    async def _handle_heartbeat(self, message: SyncMessage):
        """处理心跳消息"""
        # 更新实例状态
        pass
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            message = SyncMessage(
                message_type=MessageType.HEARTBEAT,
                namespace='_system',
                lu_id=None,
                payload={'status': 'alive'},
                source_instance=self.instance_id,
                timestamp=datetime.now(),
                sequence_number=self._next_sequence(),
            )
            
            await self._send_message(message)
            await asyncio.sleep(10)
    
    def _next_sequence(self) -> int:
        """获取下一个序列号"""
        self._sequence_number += 1
        return self._sequence_number
    
    def _check_sequence(self, message: SyncMessage):
        """检查序列号（检测丢失）"""
        source = message.source_instance
        seq = message.sequence_number
        
        last_seq = self._received_sequences.get(source, 0)
        
        if seq > last_seq + 1:
            logger.warning(
                f"Sequence gap detected from {source}: "
                f"expected {last_seq + 1}, got {seq}"
            )
            # 触发同步请求
            asyncio.create_task(self._request_sync(source, last_seq + 1, seq - 1))
        
        self._received_sequences[source] = seq
    
    async def _request_sync(self, source: str, from_seq: int, to_seq: int):
        """请求同步丢失的消息"""
        # 实现略
        pass
    
    async def _send_ack(self, original_message: SyncMessage):
        """发送确认"""
        # 实现略
        pass
    
    async def _wait_for_acks(self, message: SyncMessage, timeout: float) -> int:
        """等待确认"""
        # 实现略
        return 1
```

### 3.2 分布式槽位池 (DistributedSlotPool)

```python
"""
分布式槽位池

支持跨实例的槽位同步和一致性。
"""
from typing import Optional, Dict, Any, List
import asyncio
import logging

logger = logging.getLogger(__name__)


class DistributedSlotPool:
    """
    分布式槽位池
    
    特性：
    - 本地热缓存 + Redis 共享缓存
    - 写时广播
    - 读时本地优先
    - 版本控制防止冲突
    """
    
    def __init__(
        self,
        namespace: str,
        local_pool: 'SlotPool',
        redis_client,
        synchronizer: 'DistributedSynchronizer',
    ):
        self.namespace = namespace
        self.local_pool = local_pool
        self.redis = redis_client
        self.synchronizer = synchronizer
        
        # Redis key 前缀
        self.key_prefix = f"aga:slots:{namespace}"
    
    async def add_slot(
        self,
        lu_id: str,
        key_vector,
        value_vector,
        lifecycle_state: str,
        **kwargs
    ) -> Optional[int]:
        """
        添加槽位（分布式）
        
        流程：
        1. 写入本地
        2. 写入 Redis
        3. 广播到其他实例
        """
        # 1. 写入本地
        slot_idx = self.local_pool.add_slot(
            lu_id=lu_id,
            key_vector=key_vector,
            value_vector=value_vector,
            lifecycle_state=lifecycle_state,
            **kwargs
        )
        
        if slot_idx is None:
            return None
        
        # 2. 写入 Redis
        await self._write_to_redis(lu_id, slot_idx, key_vector, value_vector, lifecycle_state, kwargs)
        
        # 3. 广播
        await self.synchronizer.broadcast_knowledge_inject(
            namespace=self.namespace,
            lu_id=lu_id,
            key_vector=key_vector.tolist() if hasattr(key_vector, 'tolist') else key_vector,
            value_vector=value_vector.tolist() if hasattr(value_vector, 'tolist') else value_vector,
            lifecycle_state=lifecycle_state,
            **kwargs
        )
        
        return slot_idx
    
    async def quarantine_slot(self, lu_id: str, reason: str) -> bool:
        """
        隔离槽位（强一致）
        """
        # 使用分布式锁和广播
        success = await self.synchronizer.broadcast_quarantine(
            namespace=self.namespace,
            lu_id=lu_id,
            reason=reason,
        )
        
        if success:
            # 更新 Redis
            await self._update_lifecycle_in_redis(lu_id, 'quarantined')
        
        return success
    
    async def sync_from_redis(self):
        """
        从 Redis 同步槽位（启动时调用）
        """
        import torch
        
        # 获取所有槽位 key
        pattern = f"{self.key_prefix}:*"
        keys = await self.redis.keys(pattern)
        
        for key in keys:
            data = await self.redis.hgetall(key)
            if data:
                lu_id = data[b'lu_id'].decode()
                key_vector = torch.tensor(eval(data[b'key_vector'].decode()))
                value_vector = torch.tensor(eval(data[b'value_vector'].decode()))
                lifecycle_state = data[b'lifecycle_state'].decode()
                
                self.local_pool.add_slot(
                    lu_id=lu_id,
                    key_vector=key_vector,
                    value_vector=value_vector,
                    lifecycle_state=lifecycle_state,
                )
        
        logger.info(f"Synced {len(keys)} slots from Redis for namespace {self.namespace}")
    
    async def _write_to_redis(
        self,
        lu_id: str,
        slot_idx: int,
        key_vector,
        value_vector,
        lifecycle_state: str,
        metadata: Dict,
    ):
        """写入 Redis"""
        key = f"{self.key_prefix}:{lu_id}"
        
        await self.redis.hset(key, mapping={
            'lu_id': lu_id,
            'slot_idx': slot_idx,
            'key_vector': str(key_vector.tolist() if hasattr(key_vector, 'tolist') else key_vector),
            'value_vector': str(value_vector.tolist() if hasattr(value_vector, 'tolist') else value_vector),
            'lifecycle_state': lifecycle_state,
            'condition': metadata.get('condition', ''),
            'decision': metadata.get('decision', ''),
        })
        
        # 设置 TTL（7 天）
        await self.redis.expire(key, 7 * 24 * 3600)
    
    async def _update_lifecycle_in_redis(self, lu_id: str, new_state: str):
        """更新 Redis 中的生命周期状态"""
        key = f"{self.key_prefix}:{lu_id}"
        await self.redis.hset(key, 'lifecycle_state', new_state)
```

### 3.3 实例协调器 (InstanceCoordinator)

```python
"""
实例协调器

管理多实例的注册、发现和健康检查。
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str
    host: str
    port: int
    status: str  # 'active', 'inactive', 'unhealthy'
    last_heartbeat: datetime
    namespaces: List[str]
    slot_count: int
    version: str


class InstanceCoordinator:
    """
    实例协调器
    
    功能：
    - 实例注册和发现
    - 健康检查
    - 负载均衡建议
    - 故障检测
    """
    
    def __init__(
        self,
        instance_id: str,
        redis_client,
        heartbeat_interval: float = 10.0,
        unhealthy_threshold: float = 30.0,
    ):
        self.instance_id = instance_id
        self.redis = redis_client
        self.heartbeat_interval = heartbeat_interval
        self.unhealthy_threshold = unhealthy_threshold
        
        self._running = False
        self._instances: Dict[str, InstanceInfo] = {}
    
    async def start(self):
        """启动协调器"""
        self._running = True
        
        # 注册自己
        await self._register()
        
        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())
        
        # 启动健康检查
        asyncio.create_task(self._health_check_loop())
        
        logger.info(f"InstanceCoordinator started: {self.instance_id}")
    
    async def stop(self):
        """停止协调器"""
        self._running = False
        await self._deregister()
        logger.info(f"InstanceCoordinator stopped: {self.instance_id}")
    
    async def get_active_instances(self) -> List[InstanceInfo]:
        """获取所有活跃实例"""
        await self._refresh_instances()
        return [i for i in self._instances.values() if i.status == 'active']
    
    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        """获取指定实例信息"""
        await self._refresh_instances()
        return self._instances.get(instance_id)
    
    async def _register(self):
        """注册实例"""
        key = f"aga:instances:{self.instance_id}"
        
        await self.redis.hset(key, mapping={
            'instance_id': self.instance_id,
            'host': 'localhost',  # 实际应从配置获取
            'port': 8000,
            'status': 'active',
            'last_heartbeat': datetime.now().isoformat(),
            'namespaces': '[]',
            'slot_count': 0,
            'version': '2.1.0',
        })
        
        await self.redis.expire(key, int(self.unhealthy_threshold * 2))
    
    async def _deregister(self):
        """注销实例"""
        key = f"aga:instances:{self.instance_id}"
        await self.redis.delete(key)
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            await self._update_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _update_heartbeat(self):
        """更新心跳"""
        key = f"aga:instances:{self.instance_id}"
        await self.redis.hset(key, 'last_heartbeat', datetime.now().isoformat())
        await self.redis.expire(key, int(self.unhealthy_threshold * 2))
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            await self._check_instances_health()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _check_instances_health(self):
        """检查所有实例健康状态"""
        await self._refresh_instances()
        
        now = datetime.now()
        for instance in self._instances.values():
            elapsed = (now - instance.last_heartbeat).total_seconds()
            
            if elapsed > self.unhealthy_threshold:
                if instance.status != 'unhealthy':
                    logger.warning(f"Instance {instance.instance_id} is unhealthy")
                    instance.status = 'unhealthy'
    
    async def _refresh_instances(self):
        """刷新实例列表"""
        pattern = "aga:instances:*"
        keys = await self.redis.keys(pattern)
        
        self._instances.clear()
        
        for key in keys:
            data = await self.redis.hgetall(key)
            if data:
                instance = InstanceInfo(
                    instance_id=data[b'instance_id'].decode(),
                    host=data[b'host'].decode(),
                    port=int(data[b'port']),
                    status=data[b'status'].decode(),
                    last_heartbeat=datetime.fromisoformat(data[b'last_heartbeat'].decode()),
                    namespaces=eval(data[b'namespaces'].decode()),
                    slot_count=int(data[b'slot_count']),
                    version=data[b'version'].decode(),
                )
                self._instances[instance.instance_id] = instance
```

---

## 4. 实施方案

### 4.1 Phase 1: 基础设施准备 (1-2 周)

**任务清单**：

1. **部署 Kafka 集群**
   - 3 节点 Kafka 集群
   - 创建 `aga_sync` topic（分区数 = 实例数 × 2）
   - 配置消息保留策略（7 天）

2. **部署 Redis 集群**
   - 3 主 3 从 Redis Cluster
   - 配置持久化（AOF + RDB）
   - 配置最大内存和淘汰策略

3. **升级 PostgreSQL**
   - 主从复制配置
   - 连接池配置（PgBouncer）

### 4.2 Phase 2: 代码改造 (2-3 周)

**改造清单**：

1. **添加分布式同步模块**
   ```
   aga/
   ├── distributed/
   │   ├── __init__.py
   │   ├── synchronizer.py
   │   ├── distributed_pool.py
   │   ├── coordinator.py
   │   └── config.py
   ```

2. **修改 AGAManager**
   - 添加 `DistributedSynchronizer` 集成
   - 修改知识注入流程
   - 添加启动时同步逻辑

3. **修改 SlotPool**
   - 添加版本控制
   - 添加分布式锁支持

### 4.3 Phase 3: 测试验证 (1-2 周)

**测试场景**：

| 场景 | 验证点 | 通过标准 |
|-----|-------|---------|
| 知识同步 | 实例 A 注入，实例 B 可见 | <1s 延迟 |
| 隔离命令 | 所有实例同时隔离 | 100% 一致 |
| 实例故障 | 单实例宕机不影响服务 | 服务可用 |
| 网络分区 | 分区恢复后数据一致 | 最终一致 |

### 4.4 Phase 4: 灰度上线 (1-2 周)

**灰度策略**：

1. **10% 流量**：单实例 + 分布式组件
2. **30% 流量**：2 实例
3. **50% 流量**：3 实例
4. **100% 流量**：全量

---

## 5. 一致性保证

### 5.1 一致性级别

| 操作类型 | 一致性级别 | 实现方式 |
|---------|-----------|---------|
| 知识注入 | 最终一致 | Kafka 广播 |
| 生命周期更新 | 最终一致 | Kafka 广播 |
| 隔离命令 | 强一致 | 分布式锁 + 同步广播 |
| 读取知识 | 本地一致 | 本地缓存 |

### 5.2 冲突解决

```python
class ConflictResolver:
    """冲突解决器"""
    
    def resolve(self, local_slot: Slot, remote_slot: Slot) -> Slot:
        """
        解决冲突
        
        策略：
        1. 版本号高的优先
        2. 版本号相同时，时间戳新的优先
        3. 隔离状态优先（安全第一）
        """
        # 隔离状态优先
        if remote_slot.lifecycle_state == 'quarantined':
            return remote_slot
        if local_slot.lifecycle_state == 'quarantined':
            return local_slot
        
        # 版本号比较
        if remote_slot.version > local_slot.version:
            return remote_slot
        if local_slot.version > remote_slot.version:
            return local_slot
        
        # 时间戳比较
        if remote_slot.updated_at > local_slot.updated_at:
            return remote_slot
        
        return local_slot
```

---

## 6. 故障处理

### 6.1 故障场景和处理

| 故障场景 | 检测方式 | 处理策略 |
|---------|---------|---------|
| 单实例宕机 | 心跳超时 | 自动摘除，流量转移 |
| Kafka 不可用 | 生产者超时 | 降级为本地模式 |
| Redis 不可用 | 连接失败 | 使用本地缓存 |
| 网络分区 | 心跳丢失 | 分区内独立运行 |

### 6.2 降级策略

```python
class DegradationManager:
    """降级管理器"""
    
    def __init__(self):
        self.kafka_available = True
        self.redis_available = True
    
    def get_mode(self) -> str:
        """获取当前运行模式"""
        if self.kafka_available and self.redis_available:
            return 'distributed'
        elif self.redis_available:
            return 'redis_only'
        else:
            return 'local_only'
    
    def should_broadcast(self) -> bool:
        """是否应该广播"""
        return self.kafka_available
    
    def should_sync_redis(self) -> bool:
        """是否应该同步 Redis"""
        return self.redis_available
```

---

## 7. 监控与运维

### 7.1 关键指标

| 指标 | 说明 | 告警阈值 |
|-----|------|---------|
| `aga_sync_lag_seconds` | 同步延迟 | > 5s |
| `aga_instance_count` | 活跃实例数 | < 预期数量 |
| `aga_kafka_produce_errors` | Kafka 生产错误 | > 0 |
| `aga_redis_connection_errors` | Redis 连接错误 | > 0 |
| `aga_conflict_count` | 冲突次数 | > 10/min |

### 7.2 运维命令

```bash
# 查看实例状态
aga-cli instances list

# 强制同步
aga-cli sync --namespace=my_app --force

# 隔离知识
aga-cli quarantine --namespace=my_app --lu-id=LU_001 --reason="发现问题"

# 查看同步状态
aga-cli sync-status --namespace=my_app
```

---

## 附录

### A. 配置示例

```yaml
# config/distributed.yaml
distributed:
  enabled: true
  instance_id: "${HOSTNAME}"
  
  kafka:
    bootstrap_servers: "kafka-1:9092,kafka-2:9092,kafka-3:9092"
    topic: "aga_sync"
    consumer_group: "aga_sync_${HOSTNAME}"
  
  redis:
    cluster:
      - "redis-1:6379"
      - "redis-2:6379"
      - "redis-3:6379"
    password: "${REDIS_PASSWORD}"
  
  coordinator:
    heartbeat_interval: 10
    unhealthy_threshold: 30
  
  sync:
    broadcast_timeout: 5
    ack_timeout: 10
```

### B. 部署拓扑示例

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kubernetes Cluster                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        AGA Deployment (3 replicas)                   │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │   Pod 1     │    │   Pod 2     │    │   Pod 3     │              │   │
│  │  │   LLM+AGA   │    │   LLM+AGA   │    │   LLM+AGA   │              │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Kafka StatefulSet (3 replicas)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Redis Cluster (6 nodes)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PostgreSQL (Primary + Replica)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*文档结束*

