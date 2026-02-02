"""
AGA API 服务层

核心业务逻辑实现，集成现有的写入器和持久化管理器。

设计原则：
- 服务层负责所有业务逻辑
- 路由层只负责 HTTP 协议转换
- 复用现有的 KnowledgeWriter 和 PersistenceManager
"""
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

import torch

from ..types import LifecycleState, Slot, KnowledgeSlotInfo
from ..core import AuxiliaryGovernedAttention

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """服务配置"""
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    num_slots: int = 100
    num_heads: int = 32
    # 持久化
    persistence_enabled: bool = True
    persistence_type: str = "sqlite"  # sqlite, postgres, redis
    persistence_path: str = "./aga_api_data.db"
    # 写入器
    writer_enabled: bool = True
    writer_num_workers: int = 2
    writer_max_queue_size: int = 1000
    enable_quality_assessment: bool = True


class AGAService:
    """
    AGA 核心服务
    
    提供统一的业务逻辑层，集成：
    - AGA 模块管理
    - KnowledgeWriter（异步写入）
    - PersistenceManager（持久化）
    """
    
    _instance: Optional["AGAService"] = None
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig()
        
        # AGA 实例管理（按命名空间）
        self._aga_instances: Dict[str, AuxiliaryGovernedAttention] = {}
        
        # 持久化适配器和管理器
        self._persistence_adapter = None
        self._persistence_managers: Dict[str, Any] = {}
        
        # 写入器
        self._writer = None
        self._aga_manager = None
        
        # 启动时间（用于健康检查）
        self._start_time = time.time()
        
        # 锁
        self._lock = asyncio.Lock()
        
        # 初始化
        self._initialized = False
    
    @classmethod
    def get_instance(cls, config: ServiceConfig = None) -> "AGAService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试）"""
        if cls._instance:
            # 清理资源
            if cls._instance._writer:
                cls._instance._writer.stop()
        cls._instance = None
    
    async def initialize(self) -> bool:
        """初始化服务"""
        if self._initialized:
            return True
        
        try:
            # 1. 初始化持久化
            if self.config.persistence_enabled:
                await self._init_persistence()
            
            # 2. 初始化写入器
            if self.config.writer_enabled:
                self._init_writer()
            
            self._initialized = True
            logger.info("AGA Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AGA Service: {e}")
            return False
    
    async def shutdown(self):
        """关闭服务"""
        # 停止写入器
        if self._writer:
            self._writer.stop()
        
        # 断开持久化连接
        for pm in self._persistence_managers.values():
            try:
                await pm.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting persistence: {e}")
        
        logger.info("AGA Service shutdown")
    
    async def _init_persistence(self):
        """初始化持久化层"""
        from ..persistence import create_adapter, PersistenceManager
        
        self._persistence_adapter = create_adapter(
            self.config.persistence_type,
            db_path=self.config.persistence_path,
        )
        await self._persistence_adapter.connect()
        logger.info(f"Persistence adapter connected: {self.config.persistence_type}")
    
    def _init_writer(self):
        """初始化写入器"""
        from ..production.operator import ConcurrentAGAManager
        from ..production.config import ProductionAGAConfig, SlotPoolConfig, PersistenceConfig
        from ..production.writer import KnowledgeWriter
        
        # 创建默认配置
        default_config = ProductionAGAConfig(
            slot_pool=SlotPoolConfig(
                max_slots_per_namespace=self.config.num_slots,
                hidden_dim=self.config.hidden_dim,
                bottleneck_dim=self.config.bottleneck_dim,
            ),
            persistence=PersistenceConfig(
                redis_enabled=False,
                postgres_enabled=False,
            ),
        )
        
        # 创建并发 AGA 管理器
        self._aga_manager = ConcurrentAGAManager(default_config)
        
        # 创建写入器
        self._writer = KnowledgeWriter(
            aga_manager=self._aga_manager,
            persistence=None,  # 我们使用自己的持久化管理
            num_workers=self.config.writer_num_workers,
            max_queue_size=self.config.writer_max_queue_size,
            enable_quality_assessment=self.config.enable_quality_assessment,
        )
        self._writer.start()
        logger.info("KnowledgeWriter started")
    
    def _get_or_create_aga(self, namespace: str) -> AuxiliaryGovernedAttention:
        """获取或创建 AGA 实例"""
        if namespace not in self._aga_instances:
            aga = AuxiliaryGovernedAttention(
                hidden_dim=self.config.hidden_dim,
                bottleneck_dim=self.config.bottleneck_dim,
                num_slots=self.config.num_slots,
                num_heads=self.config.num_heads,
            )
            aga.eval()
            self._aga_instances[namespace] = aga
            logger.info(f"Created AGA instance for namespace: {namespace}")
        
        return self._aga_instances[namespace]
    
    async def _get_persistence_manager(self, namespace: str):
        """获取持久化管理器"""
        if not self.config.persistence_enabled:
            return None
        
        if namespace not in self._persistence_managers:
            from ..persistence import PersistenceManager
            
            pm = PersistenceManager(
                adapter=self._persistence_adapter,
                namespace=namespace,
            )
            await pm.connect()
            self._persistence_managers[namespace] = pm
        
        return self._persistence_managers[namespace]
    
    # ============================================================
    # 知识管理服务
    # ============================================================
    
    async def inject_knowledge(
        self,
        namespace: str,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = True,
    ) -> Dict[str, Any]:
        """
        注入知识
        
        Args:
            namespace: 命名空间
            lu_id: Learning Unit ID
            condition: 触发条件
            decision: 决策描述
            key_vector: 条件编码向量
            value_vector: 决策编码向量
            lifecycle_state: 生命周期状态
            trust_tier: 信任层级
            metadata: 扩展元数据
            sync: 是否同步写入
        
        Returns:
            注入结果
        """
        async with self._lock:
            # 验证向量维度
            if len(key_vector) != self.config.bottleneck_dim:
                raise ValueError(
                    f"Key vector dimension mismatch: expected {self.config.bottleneck_dim}, "
                    f"got {len(key_vector)}"
                )
            if len(value_vector) != self.config.hidden_dim:
                raise ValueError(
                    f"Value vector dimension mismatch: expected {self.config.hidden_dim}, "
                    f"got {len(value_vector)}"
                )
            
            # 转换为 tensor
            key_tensor = torch.tensor(key_vector, dtype=torch.float32)
            value_tensor = torch.tensor(value_vector, dtype=torch.float32)
            
            # 转换生命周期状态
            lc_state = LifecycleState(lifecycle_state)
            
            # 获取 AGA 实例
            aga = self._get_or_create_aga(namespace)
            
            # 查找空闲槽位
            slot_idx = aga.find_free_slot()
            if slot_idx is None:
                raise RuntimeError(f"No free slots available in namespace {namespace}")
            
            # 注入知识到 AGA
            aga.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=key_tensor,
                value_vector=value_tensor,
                lu_id=lu_id,
                lifecycle_state=lc_state,
                condition=condition,
                decision=decision,
            )
            
            # 持久化
            if self.config.persistence_enabled:
                pm = await self._get_persistence_manager(namespace)
                if pm:
                    from ..persistence.base import KnowledgeRecord
                    record = KnowledgeRecord(
                        slot_idx=slot_idx,
                        lu_id=lu_id,
                        condition=condition,
                        decision=decision,
                        key_vector=key_vector,
                        value_vector=value_vector,
                        lifecycle_state=lifecycle_state,
                        namespace=namespace,
                        trust_tier=trust_tier,
                        metadata=metadata or {},
                    )
                    await self._persistence_adapter.save_slot(namespace, record)
            
            return {
                "success": True,
                "lu_id": lu_id,
                "namespace": namespace,
                "slot_idx": slot_idx,
                "lifecycle_state": lifecycle_state,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def batch_inject_knowledge(
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
            item_ns = item.get("namespace", namespace)
            try:
                result = await self.inject_knowledge(
                    namespace=item_ns,
                    lu_id=item["lu_id"],
                    condition=item["condition"],
                    decision=item["decision"],
                    key_vector=item["key_vector"],
                    value_vector=item["value_vector"],
                    lifecycle_state=item.get("lifecycle_state", "probationary"),
                    trust_tier=item.get("trust_tier"),
                    metadata=item.get("metadata"),
                )
                results.append(result)
                if result.get("success"):
                    success_count += 1
            except Exception as e:
                failed_count += 1
                results.append({
                    "success": False,
                    "lu_id": item.get("lu_id"),
                    "error": str(e),
                })
        
        return {
            "total": len(items),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
        }
    
    async def update_lifecycle(
        self,
        namespace: str,
        lu_id: str,
        new_state: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        async with self._lock:
            aga = self._get_or_create_aga(namespace)
            new_lc_state = LifecycleState(new_state)
            
            # 更新 AGA
            slots = aga.get_slot_by_lu_id(lu_id)
            if not slots:
                raise ValueError(f"Knowledge {lu_id} not found in namespace {namespace}")
            
            for slot_idx in slots:
                aga.update_lifecycle(slot_idx, new_lc_state)
            
            # 更新持久化
            if self.config.persistence_enabled:
                await self._persistence_adapter.update_lifecycle(namespace, lu_id, new_lc_state)
            
            return {
                "success": True,
                "lu_id": lu_id,
                "namespace": namespace,
                "old_state": None,  # TODO: 记录旧状态
                "new_state": new_state,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def quarantine_knowledge(
        self,
        namespace: str,
        lu_id: str,
        reason: str,
        source_instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """隔离知识"""
        async with self._lock:
            aga = self._get_or_create_aga(namespace)
            
            # 隔离 AGA
            quarantined_slots = aga.quarantine_by_lu_id(lu_id)
            
            # 更新持久化
            if self.config.persistence_enabled:
                await self._persistence_adapter.update_lifecycle(
                    namespace, lu_id, LifecycleState.QUARANTINED
                )
            
            return {
                "success": True,
                "lu_id": lu_id,
                "namespace": namespace,
                "quarantined_slots": quarantined_slots,
                "reason": reason,
                "source_instance": source_instance,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def delete_knowledge(
        self,
        namespace: str,
        lu_id: str,
    ) -> Dict[str, Any]:
        """删除知识"""
        async with self._lock:
            aga = self._get_or_create_aga(namespace)
            
            # 从 AGA 删除
            slots = aga.get_slot_by_lu_id(lu_id)
            for slot_idx in slots:
                aga.clear_slot(slot_idx)
            
            # 从持久化删除
            if self.config.persistence_enabled:
                await self._persistence_adapter.delete_slot(namespace, lu_id)
            
            return {
                "success": True,
                "lu_id": lu_id,
                "namespace": namespace,
                "deleted_slots": slots,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    # ============================================================
    # 查询服务
    # ============================================================
    
    async def get_knowledge(
        self,
        namespace: str,
        lu_id: str,
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单条知识"""
        aga = self._get_or_create_aga(namespace)
        
        slots = aga.get_slot_by_lu_id(lu_id)
        if not slots:
            return None
        
        slot_idx = slots[0]
        info = aga.get_slot_info(slot_idx)
        
        result = {
            "lu_id": info.lu_id,
            "namespace": namespace,
            "slot_idx": info.slot_idx,
            "condition": info.condition,
            "decision": info.decision,
            "lifecycle_state": info.lifecycle_state.value,
            "reliability": info.reliability,
            "hit_count": info.hit_count,
            "key_norm": info.key_norm,
            "value_norm": info.value_norm,
            "created_at": info.created_at.isoformat() if info.created_at else None,
        }
        
        if include_vectors:
            result["key_vector"] = aga.aux_keys[slot_idx].cpu().tolist()
            result["value_vector"] = aga.aux_values[slot_idx].cpu().tolist()
        
        return result
    
    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """查询知识列表"""
        aga = self._get_or_create_aga(namespace)
        
        # 获取所有活跃知识
        all_knowledge = aga.get_active_knowledge()
        
        # 筛选
        results = []
        for k in all_knowledge:
            # 状态筛选
            if lifecycle_states:
                if k.lifecycle_state.value not in lifecycle_states:
                    continue
            
            item = {
                "lu_id": k.lu_id,
                "namespace": namespace,
                "slot_idx": k.slot_idx,
                "condition": k.condition,
                "decision": k.decision,
                "lifecycle_state": k.lifecycle_state.value,
                "reliability": k.reliability,
                "hit_count": k.hit_count,
            }
            
            if include_vectors:
                item["key_vector"] = aga.aux_keys[k.slot_idx].cpu().tolist()
                item["value_vector"] = aga.aux_values[k.slot_idx].cpu().tolist()
            
            results.append(item)
        
        # 分页
        return results[offset:offset + limit]
    
    async def find_free_slot(self, namespace: str) -> Optional[int]:
        """查找空闲槽位"""
        aga = self._get_or_create_aga(namespace)
        return aga.find_free_slot()
    
    async def get_slot_info(self, namespace: str, slot_idx: int) -> Dict[str, Any]:
        """获取槽位信息"""
        aga = self._get_or_create_aga(namespace)
        info = aga.get_slot_info(slot_idx)
        
        return {
            "slot_idx": info.slot_idx,
            "lu_id": info.lu_id,
            "namespace": namespace,
            "lifecycle_state": info.lifecycle_state.value,
            "reliability": info.reliability,
            "key_norm": info.key_norm,
            "value_norm": info.value_norm,
            "condition": info.condition,
            "decision": info.decision,
            "hit_count": info.hit_count,
            "is_active": info.lu_id is not None,
        }
    
    # ============================================================
    # 统计服务
    # ============================================================
    
    async def get_namespace_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取命名空间统计"""
        aga = self._get_or_create_aga(namespace)
        stats = aga.get_statistics()
        
        return {
            "namespace": namespace,
            "total_slots": stats.get("total_slots", 0),
            "active_slots": stats.get("active_slots", 0),
            "free_slots": stats.get("total_slots", 0) - stats.get("active_slots", 0),
            "state_distribution": stats.get("state_distribution", {}),
            "total_hits": stats.get("total_hits", 0),
            "avg_reliability": stats.get("avg_reliability", 0.0),
            "avg_key_norm": stats.get("avg_key_norm", 0.0),
            "avg_value_norm": stats.get("avg_value_norm", 0.0),
        }
    
    async def get_all_statistics(self) -> Dict[str, Any]:
        """获取所有命名空间统计"""
        namespaces_stats = {}
        total_knowledge = 0
        
        for ns in self._aga_instances:
            stats = await self.get_namespace_statistics(ns)
            namespaces_stats[ns] = stats
            total_knowledge += stats.get("active_slots", 0)
        
        return {
            "namespaces": namespaces_stats,
            "total_namespaces": len(self._aga_instances),
            "total_knowledge": total_knowledge,
        }
    
    async def get_writer_statistics(self) -> Dict[str, Any]:
        """获取写入器统计"""
        if self._writer:
            return self._writer.get_statistics()
        return {}
    
    # ============================================================
    # 审计服务
    # ============================================================
    
    async def get_audit_log(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """获取审计日志"""
        if not self.config.persistence_enabled or not self._persistence_adapter:
            return {"entries": [], "total": 0, "limit": limit, "offset": offset}
        
        # 从持久化层获取审计日志
        if hasattr(self._persistence_adapter, 'get_audit_log'):
            entries = await self._persistence_adapter.get_audit_log(namespace, limit + offset)
            paged_entries = entries[offset:offset + limit]
            return {
                "entries": paged_entries,
                "total": len(entries),
                "limit": limit,
                "offset": offset,
            }
        
        return {"entries": [], "total": 0, "limit": limit, "offset": offset}
    
    # ============================================================
    # 命名空间服务
    # ============================================================
    
    def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        return list(self._aga_instances.keys())
    
    async def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """删除命名空间"""
        if namespace not in self._aga_instances:
            return {"success": False, "error": f"Namespace {namespace} not found"}
        
        async with self._lock:
            # 删除 AGA 实例
            del self._aga_instances[namespace]
            
            # 删除持久化管理器
            if namespace in self._persistence_managers:
                pm = self._persistence_managers[namespace]
                await pm.disconnect()
                del self._persistence_managers[namespace]
        
        return {
            "success": True,
            "namespace": namespace,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ============================================================
    # 健康检查服务
    # ============================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        total_knowledge = sum(
            len(aga.get_active_knowledge())
            for aga in self._aga_instances.values()
        )
        
        # 持久化健康检查
        persistence_status = "disabled"
        if self.config.persistence_enabled and self._persistence_adapter:
            try:
                await self._persistence_adapter.health_check()
                persistence_status = "healthy"
            except Exception as e:
                persistence_status = f"unhealthy: {e}"
        
        # 写入器状态
        writer_status = "disabled"
        if self._writer:
            writer_stats = self._writer.get_statistics()
            writer_status = f"healthy (queue: {writer_stats.get('queue_size', 0)})"
        
        return {
            "status": "healthy",
            "version": "3.1.0",
            "initialized": self._initialized,
            "namespaces": list(self._aga_instances.keys()),
            "total_knowledge": total_knowledge,
            "persistence": persistence_status,
            "writer": writer_status,
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ============================================================
    # 配置服务
    # ============================================================
    
    def get_config(self) -> Dict[str, Any]:
        """获取服务配置"""
        return {
            "hidden_dim": self.config.hidden_dim,
            "bottleneck_dim": self.config.bottleneck_dim,
            "num_slots": self.config.num_slots,
            "num_heads": self.config.num_heads,
            "persistence_enabled": self.config.persistence_enabled,
            "persistence_type": self.config.persistence_type,
            "writer_enabled": self.config.writer_enabled,
            "enable_quality_assessment": self.config.enable_quality_assessment,
        }
