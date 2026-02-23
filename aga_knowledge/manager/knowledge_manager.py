"""
AGA Knowledge Manager（明文 KV 版本）

面向 Runtime 的知识管理器，负责：
- 从持久化层加载知识到内存
- 监听同步消息并更新本地缓存
- 向 aga-core 提供知识查询接口
- 知识生命周期管理

使用示例：
    from aga_knowledge.manager import KnowledgeManager
    from aga_knowledge.config import PortalConfig

    config = PortalConfig.for_development()
    manager = KnowledgeManager(config)
    await manager.start()

    # 获取活跃知识
    knowledge = await manager.get_active_knowledge("default")

    # 按 lu_id 获取
    record = await manager.get_knowledge("default", "rule_001")
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable

from ..config import PortalConfig
from ..persistence import PersistenceAdapter, create_adapter
from ..sync import SyncMessage, MessageType
from ..sync.backends import SyncBackend, create_backend
from ..types import KnowledgeRecord, LifecycleState, LIFECYCLE_RELIABILITY

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    知识管理器（明文 KV 版本）

    面向 Runtime 的知识管理接口。

    职责：
    - 从持久化层加载知识
    - 监听同步消息并更新本地缓存
    - 提供知识查询接口
    - 管理知识生命周期

    特性：
    - 本地内存缓存（快速查询）
    - 异步消息订阅
    - 自动同步
    - 线程安全
    """

    def __init__(
        self,
        config: PortalConfig,
        instance_id: str = "runtime-default",
        namespaces: Optional[List[str]] = None,
        on_knowledge_update: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        """
        初始化知识管理器

        Args:
            config: Portal 配置
            instance_id: Runtime 实例 ID
            namespaces: 订阅的命名空间列表
            on_knowledge_update: 知识更新回调 (namespace, lu_id, record)
        """
        self.config = config
        self.instance_id = instance_id
        self.namespaces = namespaces or ["default"]
        self.on_knowledge_update = on_knowledge_update

        # 组件
        self._persistence: Optional[PersistenceAdapter] = None
        self._backend: Optional[SyncBackend] = None

        # 本地缓存: {namespace: {lu_id: record}}
        self._cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # 状态
        self._running = False
        self._start_time: Optional[float] = None

        # 统计
        self._stats = {
            "messages_received": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "sync_errors": 0,
        }

    async def start(self):
        """启动知识管理器"""
        if self._running:
            return

        self._start_time = time.time()

        # 初始化持久化
        await self._init_persistence()

        # 加载初始知识
        await self._load_initial_knowledge()

        # 初始化消息订阅
        await self._init_subscriber()

        self._running = True
        logger.info(
            f"KnowledgeManager started: instance={self.instance_id}, "
            f"namespaces={self.namespaces}"
        )

    async def stop(self):
        """停止知识管理器"""
        self._running = False

        if self._backend:
            await self._backend.disconnect()

        if self._persistence:
            await self._persistence.disconnect()

        logger.info(f"KnowledgeManager stopped: instance={self.instance_id}")

    async def _init_persistence(self):
        """初始化持久化"""
        persistence_config = self.config.persistence
        adapter_dict = {
            "type": persistence_config.type,
            "sqlite_path": persistence_config.sqlite_path,
            "postgres_url": persistence_config.postgres_url,
            "enable_audit": persistence_config.enable_audit,
        }
        self._persistence = create_adapter(adapter_dict)
        await self._persistence.connect()

    async def _load_initial_knowledge(self):
        """从持久化层加载初始知识"""
        for namespace in self.namespaces:
            records = await self._persistence.load_active_knowledge(namespace)
            self._cache[namespace] = {}
            for record in records:
                lu_id = record.get("lu_id")
                if lu_id:
                    self._cache[namespace][lu_id] = record
            logger.info(
                f"Loaded {len(records)} knowledge records for namespace={namespace}"
            )

    async def _init_subscriber(self):
        """初始化消息订阅"""
        messaging_config = self.config.messaging

        kwargs = {}
        if messaging_config.backend == "redis":
            kwargs = {
                "host": messaging_config.redis_host,
                "port": messaging_config.redis_port,
                "db": messaging_config.redis_db,
                "password": messaging_config.redis_password,
            }

        self._backend = create_backend(messaging_config.backend, **kwargs)
        await self._backend.connect()
        await self._backend.subscribe(
            messaging_config.redis_channel,
            self._handle_sync_message,
        )
        logger.info(f"Subscribed to sync channel: {messaging_config.redis_channel}")

    async def _handle_sync_message(self, message: SyncMessage):
        """处理同步消息"""
        self._stats["messages_received"] += 1

        try:
            namespace = message.namespace
            if namespace not in self.namespaces:
                return

            if message.message_type == MessageType.INJECT:
                await self._handle_inject(message)
            elif message.message_type == MessageType.UPDATE:
                await self._handle_update(message)
            elif message.message_type == MessageType.QUARANTINE:
                await self._handle_quarantine(message)
            elif message.message_type == MessageType.DELETE:
                await self._handle_delete(message)
            elif message.message_type == MessageType.BATCH_INJECT:
                await self._handle_batch_inject(message)
            elif message.message_type == MessageType.FULL_SYNC:
                await self._handle_full_sync(message)

            # 发送 ACK
            if message.require_ack:
                await self._send_ack(message)

        except Exception as e:
            self._stats["sync_errors"] += 1
            logger.error(f"Error handling sync message: {e}")
            if message.require_ack:
                await self._send_nack(message, str(e))

    async def _handle_inject(self, message: SyncMessage):
        """处理注入消息"""
        namespace = message.namespace
        lu_id = message.lu_id
        if not lu_id:
            return

        record = {
            "lu_id": lu_id,
            "namespace": namespace,
            "condition": message.condition or "",
            "decision": message.decision or "",
            "lifecycle_state": message.lifecycle_state or "probationary",
            "trust_tier": message.trust_tier or "standard",
            "hit_count": 0,
            "metadata": message.metadata,
        }

        if namespace not in self._cache:
            self._cache[namespace] = {}
        self._cache[namespace][lu_id] = record

        # 回调通知
        if self.on_knowledge_update:
            try:
                self.on_knowledge_update(namespace, lu_id, record)
            except Exception as e:
                logger.error(f"Knowledge update callback error: {e}")

        logger.debug(f"Knowledge injected: {namespace}/{lu_id}")

    async def _handle_update(self, message: SyncMessage):
        """处理更新消息"""
        namespace = message.namespace
        lu_id = message.lu_id
        if not lu_id or namespace not in self._cache:
            return

        if lu_id in self._cache[namespace]:
            self._cache[namespace][lu_id]["lifecycle_state"] = message.lifecycle_state
            if self.on_knowledge_update:
                try:
                    self.on_knowledge_update(namespace, lu_id, self._cache[namespace][lu_id])
                except Exception as e:
                    logger.error(f"Knowledge update callback error: {e}")

    async def _handle_quarantine(self, message: SyncMessage):
        """处理隔离消息"""
        namespace = message.namespace
        lu_id = message.lu_id
        if not lu_id or namespace not in self._cache:
            return

        # 从缓存中移除（隔离的知识不参与推理）
        if lu_id in self._cache[namespace]:
            del self._cache[namespace][lu_id]
            logger.info(f"Knowledge quarantined: {namespace}/{lu_id}")

    async def _handle_delete(self, message: SyncMessage):
        """处理删除消息"""
        namespace = message.namespace
        lu_id = message.lu_id
        if not lu_id or namespace not in self._cache:
            return

        if lu_id in self._cache[namespace]:
            del self._cache[namespace][lu_id]
            logger.info(f"Knowledge deleted: {namespace}/{lu_id}")

    async def _handle_batch_inject(self, message: SyncMessage):
        """处理批量注入消息"""
        if not message.batch_items:
            return

        namespace = message.namespace
        if namespace not in self._cache:
            self._cache[namespace] = {}

        for item in message.batch_items:
            lu_id = item.get("lu_id")
            if not lu_id:
                continue
            record = {
                "lu_id": lu_id,
                "namespace": item.get("namespace", namespace),
                "condition": item.get("condition", ""),
                "decision": item.get("decision", ""),
                "lifecycle_state": item.get("lifecycle_state", "probationary"),
                "trust_tier": item.get("trust_tier", "standard"),
                "hit_count": 0,
                "metadata": item.get("metadata"),
            }
            self._cache[namespace][lu_id] = record

    async def _handle_full_sync(self, message: SyncMessage):
        """处理全量同步请求"""
        namespace = message.namespace
        if namespace in self.namespaces:
            records = await self._persistence.load_active_knowledge(namespace)
            self._cache[namespace] = {}
            for record in records:
                lu_id = record.get("lu_id")
                if lu_id:
                    self._cache[namespace][lu_id] = record
            logger.info(f"Full sync completed for namespace={namespace}: {len(records)} records")

    async def _send_ack(self, message: SyncMessage):
        """发送 ACK"""
        ack = SyncMessage(
            message_type=MessageType.ACK,
            correlation_id=message.message_id,
            source_instance=self.instance_id,
        )
        ack_channel = f"{self.config.messaging.redis_channel}:ack"
        try:
            await self._backend.publish(ack_channel, ack)
        except Exception as e:
            logger.error(f"Failed to send ACK: {e}")

    async def _send_nack(self, message: SyncMessage, error: str):
        """发送 NACK"""
        nack = SyncMessage(
            message_type=MessageType.NACK,
            correlation_id=message.message_id,
            source_instance=self.instance_id,
            metadata={"error": error},
        )
        ack_channel = f"{self.config.messaging.redis_channel}:ack"
        try:
            await self._backend.publish(ack_channel, nack)
        except Exception as e:
            logger.error(f"Failed to send NACK: {e}")

    # ==================== 查询接口 ====================

    async def get_knowledge(
        self,
        namespace: str,
        lu_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取单个知识

        优先从缓存获取，缓存未命中则从持久化层加载。

        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID

        Returns:
            知识记录字典或 None
        """
        # 缓存查找
        if namespace in self._cache and lu_id in self._cache[namespace]:
            self._stats["cache_hits"] += 1
            return self._cache[namespace][lu_id]

        # 持久化层查找
        self._stats["cache_misses"] += 1
        if self._persistence:
            record = await self._persistence.load_knowledge(namespace, lu_id)
            if record:
                # 更新缓存
                if namespace not in self._cache:
                    self._cache[namespace] = {}
                self._cache[namespace][lu_id] = record
                return record

        return None

    async def get_active_knowledge(
        self,
        namespace: str,
    ) -> List[Dict[str, Any]]:
        """
        获取命名空间下所有活跃知识

        Args:
            namespace: 命名空间

        Returns:
            活跃知识列表
        """
        if namespace in self._cache:
            return [
                record for record in self._cache[namespace].values()
                if record.get("lifecycle_state") != LifecycleState.QUARANTINED.value
            ]
        return []

    async def get_knowledge_for_injection(
        self,
        namespace: str,
    ) -> List[Dict[str, Any]]:
        """
        获取可用于注入的知识列表

        返回所有活跃且非隔离的知识，附带可靠性分数。

        Args:
            namespace: 命名空间

        Returns:
            知识列表，每项包含 reliability 字段
        """
        active = await self.get_active_knowledge(namespace)
        result = []
        for record in active:
            # 计算可靠性
            try:
                state = LifecycleState(record.get("lifecycle_state", "probationary"))
                reliability = LIFECYCLE_RELIABILITY.get(state, 0.5)
            except ValueError:
                reliability = 0.5

            enriched = dict(record)
            enriched["reliability"] = reliability
            result.append(enriched)

        return result

    async def increment_hit_count(
        self,
        namespace: str,
        lu_ids: List[str],
    ):
        """
        增加命中计数

        Args:
            namespace: 命名空间
            lu_ids: 命中的知识 ID 列表
        """
        # 更新缓存
        if namespace in self._cache:
            for lu_id in lu_ids:
                if lu_id in self._cache[namespace]:
                    self._cache[namespace][lu_id]["hit_count"] = \
                        self._cache[namespace][lu_id].get("hit_count", 0) + 1

        # 更新持久化层
        if self._persistence:
            try:
                await self._persistence.increment_hit_count(namespace, lu_ids)
            except Exception as e:
                logger.error(f"Failed to increment hit count: {e}")

    # ==================== 统计 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_stats = {}
        total_cached = 0
        for ns, records in self._cache.items():
            cache_stats[ns] = len(records)
            total_cached += len(records)

        return {
            "instance_id": self.instance_id,
            "namespaces": self.namespaces,
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "total_cached": total_cached,
            "cache_by_namespace": cache_stats,
            **self._stats,
        }

    def get_cache_summary(self) -> Dict[str, Any]:
        """获取缓存摘要"""
        summary = {}
        for ns, records in self._cache.items():
            state_counts: Dict[str, int] = {}
            for record in records.values():
                state = record.get("lifecycle_state", "unknown")
                state_counts[state] = state_counts.get(state, 0) + 1
            summary[ns] = {
                "total": len(records),
                "state_distribution": state_counts,
            }
        return summary
