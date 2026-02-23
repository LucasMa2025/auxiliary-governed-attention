"""
AGA Knowledge Portal 服务层（明文 KV 版本）

核心业务逻辑，不包含 AGA 推理实例，不包含向量化编码。

架构说明：
    Portal 是知识管理的"计算层"，负责：
    - 接收治理系统的明文 condition/decision 文本对
    - 存储到持久化层
    - 同步到 Runtime（通过消息队列）
    - 审计日志记录

    向量化编码由 aga-core 在推理时按需处理。
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..config import PortalConfig, PersistenceDBConfig, MessagingConfig
from ..persistence import PersistenceAdapter, create_adapter
from ..sync import SyncPublisher, SyncMessage, MessageType

logger = logging.getLogger(__name__)


class PortalService:
    """
    Portal 服务（明文 KV 版本）

    无 GPU 依赖的知识管理服务。

    职责：
    - 知识元数据的 CRUD（condition/decision 文本对）
    - 生命周期状态管理
    - 审计日志
    - 消息发布到 Runtime

    关键特性：
    - 不持有 AGA 推理实例
    - 不进行向量化编码
    - 通过消息队列同步到 Runtime
    - Runtime 在推理时按需将文本编码为 KV 向量
    """

    def __init__(self, config: PortalConfig):
        """
        初始化 Portal 服务

        Args:
            config: Portal 配置
        """
        self.config = config

        # 组件（延迟初始化）
        self._persistence: Optional[PersistenceAdapter] = None
        self._publisher: Optional[SyncPublisher] = None

        # 状态
        self._initialized = False
        self._start_time = time.time()

        # 统计
        self._stats = {
            "inject_count": 0,
            "batch_inject_count": 0,
            "update_count": 0,
            "quarantine_count": 0,
            "delete_count": 0,
            "query_count": 0,
        }

    async def initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        # 初始化持久化
        await self._init_persistence()

        # 初始化消息发布器
        await self._init_publisher()

        self._initialized = True
        logger.info("PortalService initialized (plaintext KV mode)")

    async def shutdown(self):
        """关闭服务"""
        if self._publisher:
            await self._publisher.disconnect()

        if self._persistence:
            await self._persistence.disconnect()

        self._initialized = False
        logger.info("PortalService shutdown")

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
        logger.info(f"Persistence initialized: {persistence_config.type}")

    async def _init_publisher(self):
        """初始化消息发布器"""
        messaging_config = self.config.messaging

        kwargs = {}
        if messaging_config.backend == "redis":
            kwargs = {
                "host": messaging_config.redis_host,
                "port": messaging_config.redis_port,
                "db": messaging_config.redis_db,
                "password": messaging_config.redis_password,
            }

        self._publisher = SyncPublisher(
            backend_type=messaging_config.backend,
            channel=messaging_config.redis_channel,
        **kwargs,
        )
        await self._publisher.connect()
        logger.info(f"Publisher initialized: {messaging_config.backend}")

    # ==================== 知识管理（明文 KV） ====================

    async def inject_knowledge_text(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        注入知识（明文 condition/decision）

        ✅ 推荐 API：外部系统应使用此方法。

        流程：
        1. 验证输入
        2. 存储到持久化层（明文）
        3. 发布同步消息到 Runtime
        4. 记录审计日志

        Args:
            lu_id: Learning Unit ID
            condition: 触发条件描述（明文）
            decision: 决策描述（明文）
            namespace: 命名空间
            lifecycle_state: 初始生命周期状态
            trust_tier: 信任层级
            metadata: 扩展元数据

        Returns:
            注入结果
        """
        self._stats["inject_count"] += 1

        # 输入验证
        if not lu_id or not lu_id.strip():
            return {"success": False, "error": "lu_id cannot be empty"}
        if not condition or not condition.strip():
            return {"success": False, "error": "condition cannot be empty"}
        if not decision or not decision.strip():
            return {"success": False, "error": "decision cannot be empty"}

        # 1. 存储明文数据
        record = {
            "lu_id": lu_id,
            "namespace": namespace,
            "condition": condition,
            "decision": decision,
            "lifecycle_state": lifecycle_state,
            "trust_tier": trust_tier or "standard",
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "hit_count": 0,
        }

        await self._persistence.save_knowledge(namespace, lu_id, record)

        # 2. 发布同步消息（明文）
        sync_result = await self._publisher.publish_inject(
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            namespace=namespace,
            lifecycle_state=lifecycle_state,
            trust_tier=trust_tier,
            source_instance="portal",
            metadata=metadata,
        )

        # 3. 记录审计日志
        await self._log_audit(
            action="INJECT",
            lu_id=lu_id,
            namespace=namespace,
            new_state=lifecycle_state,
            details={"trust_tier": trust_tier},
        )

        return {
            "success": True,
            "lu_id": lu_id,
            "namespace": namespace,
            "lifecycle_state": lifecycle_state,
            "sync_result": sync_result,
        }

    async def batch_inject_text(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        批量注入知识（明文）

        ✅ 推荐 API：外部系统应使用此方法。

        Args:
            items: 知识列表，每项包含 lu_id, condition, decision 等
            namespace: 默认命名空间
            skip_duplicates: 是否跳过重复

        Returns:
            批量注入结果
        """
        self._stats["batch_inject_count"] += 1

        results = []
        success_count = 0
        failed_count = 0

        for item in items:
            try:
                result = await self.inject_knowledge_text(
                    lu_id=item["lu_id"],
                    condition=item["condition"],
                    decision=item["decision"],
                    namespace=item.get("namespace", namespace),
                    lifecycle_state=item.get("lifecycle_state", "probationary"),
                    trust_tier=item.get("trust_tier"),
                    metadata=item.get("metadata"),
                )
                results.append(result)
                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                if not skip_duplicates:
                    raise
                results.append({
                    "success": False,
                    "lu_id": item.get("lu_id", "unknown"),
                    "error": str(e),
                })
                failed_count += 1

        return {
            "total": len(items),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
        }

    async def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        self._stats["update_count"] += 1

        # 获取当前记录
        record = await self._persistence.load_knowledge(namespace, lu_id)
        if not record:
            return {
                "success": False,
                "error": f"Knowledge not found: {lu_id}",
            }

        old_state = record.get("lifecycle_state")

        # 更新状态
        await self._persistence.update_lifecycle(namespace, lu_id, new_state)

        # 发布同步消息
        sync_result = await self._publisher.publish_update(
            lu_id=lu_id,
            new_state=new_state,
            namespace=namespace,
            reason=reason,
        )

        # 记录审计日志
        await self._log_audit(
            action="UPDATE_LIFECYCLE",
            lu_id=lu_id,
            namespace=namespace,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
        )

        return {
            "success": True,
            "lu_id": lu_id,
            "old_state": old_state,
            "new_state": new_state,
            "sync_result": sync_result,
        }

    async def quarantine(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """隔离知识"""
        self._stats["quarantine_count"] += 1

        return await self.update_lifecycle(
            lu_id=lu_id,
            new_state="quarantined",
            namespace=namespace,
            reason=reason,
        )

    async def delete_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """删除知识"""
        self._stats["delete_count"] += 1

        record = await self._persistence.load_knowledge(namespace, lu_id)
        if not record:
            return {
                "success": False,
                "error": f"Knowledge not found: {lu_id}",
            }

        await self._persistence.delete_knowledge(namespace, lu_id)

        sync_result = await self._publisher.publish_delete(
            lu_id=lu_id,
            namespace=namespace,
            reason=reason,
        )

        await self._log_audit(
            action="DELETE",
            lu_id=lu_id,
            namespace=namespace,
            reason=reason,
        )

        return {
            "success": True,
            "lu_id": lu_id,
            "sync_result": sync_result,
        }

    # ==================== 查询 ====================

    async def get_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """获取单个知识"""
        self._stats["query_count"] += 1
        return await self._persistence.load_knowledge(namespace, lu_id)

    async def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """查询知识列表"""
        self._stats["query_count"] += 1

        records = await self._persistence.query_knowledge(
            namespace=namespace,
            lifecycle_states=lifecycle_states,
            trust_tiers=trust_tiers,
            limit=limit,
            offset=offset,
        )

        return {
            "items": records,
            "count": len(records),
            "limit": limit,
            "offset": offset,
        }

    async def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        return await self._persistence.get_namespaces()

    # ==================== 统计 ====================

    async def get_statistics(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取统计信息"""
        if namespace:
            return await self._persistence.get_statistics(namespace)
        else:
            namespaces = await self.get_namespaces()
            all_stats = {}
            total_knowledge = 0

            for ns in namespaces:
                stats = await self._persistence.get_statistics(ns)
                all_stats[ns] = stats
                total_knowledge += stats.get("total_knowledge", 0)

            return {
                "namespaces": all_stats,
                "total_namespaces": len(namespaces),
                "total_knowledge": total_knowledge,
            }

    # ==================== 审计 ====================

    async def _log_audit(
        self,
        action: str,
        lu_id: Optional[str] = None,
        namespace: str = "default",
        old_state: Optional[str] = None,
        new_state: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """记录审计日志"""
        import json
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "lu_id": lu_id,
            "namespace": namespace,
            "old_state": old_state,
            "new_state": new_state,
            "reason": reason,
            "details": details,
            "source": "portal",
        }

        try:
            await self._persistence.save_audit_log(log_entry)
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")

    async def get_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """获取审计日志"""
        logs = await self._persistence.query_audit_log(
            namespace=namespace,
            lu_id=lu_id,
            limit=limit,
            offset=offset,
        )

        return {
            "entries": logs,
            "count": len(logs),
            "limit": limit,
            "offset": offset,
        }

    # ==================== 健康检查 ====================

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        persistence_health = None
        if self._persistence:
            persistence_health = await self._persistence.health_check()

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "version": self.config.version,
            "environment": self.config.environment,
            "mode": "plaintext_kv",
            "uptime_seconds": time.time() - self._start_time,
            "stats": self._stats,
            "persistence": persistence_health,
            "publisher": self._publisher.get_stats() if self._publisher else None,
        }
