"""
AGA Portal 服务层

核心业务逻辑，不包含 AGA 推理实例。
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..config.portal import PortalConfig
from ..persistence import PersistenceAdapter, SQLiteAdapter, create_adapter
from ..sync import SyncPublisher, SyncMessage, MessageType

logger = logging.getLogger(__name__)


class PortalService:
    """
    Portal 服务
    
    无 GPU 依赖的知识管理服务。
    
    职责：
    - 知识元数据的 CRUD
    - 生命周期状态管理
    - 审计日志
    - 消息发布到 Runtime
    
    关键特性：
    - 不持有 AGA 推理实例
    - 向量数据存储在数据库
    - 通过消息队列同步到 Runtime
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
            "update_count": 0,
            "quarantine_count": 0,
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
        logger.info("PortalService initialized")
    
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
        
        if persistence_config.type == "sqlite":
            self._persistence = SQLiteAdapter(persistence_config.sqlite_path)
        elif persistence_config.type == "postgres":
            # PostgreSQL 适配器
            from ..persistence import PostgresAdapter
            self._persistence = PostgresAdapter(
                dsn=persistence_config.postgres_url,
                pool_size=persistence_config.postgres_pool_size,
            )
        else:
            # 默认使用内存
            from ..persistence import MemoryAdapter
            self._persistence = MemoryAdapter()
        
        await self._persistence.connect()
        logger.info(f"Persistence initialized: {persistence_config.type}")
    
    async def _init_publisher(self):
        """初始化消息发布器"""
        messaging_config = self.config.messaging
        
        if messaging_config.backend == "memory":
            self._publisher = SyncPublisher(
                backend_type="memory",
                channel=messaging_config.redis_channel,
            )
        elif messaging_config.backend == "redis":
            self._publisher = SyncPublisher(
                backend_type="redis",
                channel=messaging_config.redis_channel,
                host=messaging_config.redis_host,
                port=messaging_config.redis_port,
                db=messaging_config.redis_db,
                password=messaging_config.redis_password,
            )
        elif messaging_config.backend == "kafka":
            self._publisher = SyncPublisher(
                backend_type="kafka",
                channel=messaging_config.kafka_topic,
                bootstrap_servers=messaging_config.kafka_bootstrap_servers,
            )
        
        await self._publisher.connect()
        logger.info(f"Publisher initialized: {messaging_config.backend}")
    
    # ==================== 知识管理 ====================
    
    async def inject_knowledge(
        self,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        condition: str,
        decision: str,
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        注入知识
        
        流程：
        1. 验证输入
        2. 存储到持久化层
        3. 发布同步消息到 Runtime
        4. 记录审计日志
        
        Args:
            lu_id: Learning Unit ID
            key_vector: 条件编码向量
            value_vector: 决策编码向量
            condition: 触发条件描述
            decision: 决策描述
            namespace: 命名空间
            lifecycle_state: 初始生命周期状态
            trust_tier: 信任层级
            metadata: 扩展元数据
        
        Returns:
            注入结果
        """
        self._stats["inject_count"] += 1
        
        # 1. 存储元数据
        record = {
            "lu_id": lu_id,
            "namespace": namespace,
            "key_vector": key_vector,
            "value_vector": value_vector,
            "condition": condition,
            "decision": decision,
            "lifecycle_state": lifecycle_state,
            "trust_tier": trust_tier,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "hit_count": 0,
        }
        
        await self._persistence.save_knowledge(namespace, lu_id, record)
        
        # 2. 发布同步消息
        sync_result = await self._publisher.publish_inject(
            lu_id=lu_id,
            key_vector=key_vector,
            value_vector=value_vector,
            condition=condition,
            decision=decision,
            namespace=namespace,
            lifecycle_state=lifecycle_state,
            trust_tier=trust_tier,
            source_instance="portal",
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
    
    async def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """批量注入知识"""
        results = []
        success_count = 0
        failed_count = 0
        
        for item in items:
            try:
                result = await self.inject_knowledge(
                    lu_id=item["lu_id"],
                    key_vector=item["key_vector"],
                    value_vector=item["value_vector"],
                    condition=item["condition"],
                    decision=item["decision"],
                    namespace=item.get("namespace", namespace),
                    lifecycle_state=item.get("lifecycle_state", "probationary"),
                    trust_tier=item.get("trust_tier"),
                    metadata=item.get("metadata"),
                )
                results.append(result)
                success_count += 1
            except Exception as e:
                if not skip_duplicates:
                    raise
                results.append({
                    "success": False,
                    "lu_id": item["lu_id"],
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
        record["lifecycle_state"] = new_state
        record["updated_at"] = datetime.utcnow().isoformat()
        
        await self._persistence.save_knowledge(namespace, lu_id, record)
        
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
        
        result = await self.update_lifecycle(
            lu_id=lu_id,
            new_state="quarantined",
            namespace=namespace,
            reason=reason,
        )
        
        return result
    
    async def delete_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """删除知识"""
        # 获取当前记录
        record = await self._persistence.load_knowledge(namespace, lu_id)
        if not record:
            return {
                "success": False,
                "error": f"Knowledge not found: {lu_id}",
            }
        
        # 删除
        await self._persistence.delete_knowledge(namespace, lu_id)
        
        # 发布同步消息
        sync_result = await self._publisher.publish_delete(
            lu_id=lu_id,
            namespace=namespace,
            reason=reason,
        )
        
        # 记录审计日志
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
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单个知识"""
        self._stats["query_count"] += 1
        
        record = await self._persistence.load_knowledge(namespace, lu_id)
        
        if record and not include_vectors:
            record.pop("key_vector", None)
            record.pop("value_vector", None)
        
        return record
    
    async def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
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
        
        if not include_vectors:
            for record in records:
                record.pop("key_vector", None)
                record.pop("value_vector", None)
        
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
            return await self._get_namespace_stats(namespace)
        else:
            namespaces = await self.get_namespaces()
            all_stats = {}
            total_knowledge = 0
            
            for ns in namespaces:
                stats = await self._get_namespace_stats(ns)
                all_stats[ns] = stats
                total_knowledge += stats.get("total_knowledge", 0)
            
            return {
                "namespaces": all_stats,
                "total_namespaces": len(namespaces),
                "total_knowledge": total_knowledge,
            }
    
    async def _get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """获取命名空间统计"""
        records = await self._persistence.query_knowledge(
            namespace=namespace,
            limit=10000,
        )
        
        state_distribution = {}
        trust_tier_distribution = {}
        total_hits = 0
        
        for record in records:
            # 状态分布
            state = record.get("lifecycle_state", "unknown")
            state_distribution[state] = state_distribution.get(state, 0) + 1
            
            # 信任层级分布
            tier = record.get("trust_tier", "unset")
            trust_tier_distribution[tier] = trust_tier_distribution.get(tier, 0) + 1
            
            # 总命中数
            total_hits += record.get("hit_count", 0)
        
        return {
            "namespace": namespace,
            "total_knowledge": len(records),
            "state_distribution": state_distribution,
            "trust_tier_distribution": trust_tier_distribution,
            "total_hits": total_hits,
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
        
        await self._persistence.save_audit_log(log_entry)
    
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
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "version": self.config.version,
            "environment": self.config.environment,
            "uptime_seconds": time.time() - self._start_time,
            "stats": self._stats,
            "persistence": self._persistence.__class__.__name__ if self._persistence else None,
            "publisher": self._publisher.get_stats() if self._publisher else None,
        }
