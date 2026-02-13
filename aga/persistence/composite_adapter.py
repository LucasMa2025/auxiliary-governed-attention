"""
AGA 组合持久化适配器

实现 L0 (Memory) -> L1 (Redis) -> L2 (PostgreSQL) 分层缓存。

版本: v3.0
"""
import asyncio
from typing import Optional, List, Dict, Any
import logging

from .base import PersistenceAdapter, KnowledgeRecord
from ..types import LifecycleState

logger = logging.getLogger(__name__)


class CompositeAdapter(PersistenceAdapter):
    """
    组合持久化适配器
    
    实现分层缓存架构：
    - L0 (Memory): 最热数据，GPU 内存友好
    - L1 (Redis): 热数据缓存，跨实例共享
    - L2 (PostgreSQL): 冷存储，完整持久化
    
    读取策略：从最近的层读取，miss 时向下查找并提升
    写入策略：写入所有层（write-through）
    """
    
    def __init__(
        self,
        l0_adapter: Optional[PersistenceAdapter] = None,  # Memory
        l1_adapter: Optional[PersistenceAdapter] = None,  # Redis
        l2_adapter: Optional[PersistenceAdapter] = None,  # PostgreSQL
        write_through: bool = True,
        promote_on_read: bool = True,
    ):
        """
        初始化组合适配器
        
        Args:
            l0_adapter: L0 层适配器（内存）
            l1_adapter: L1 层适配器（Redis）
            l2_adapter: L2 层适配器（PostgreSQL）
            write_through: 是否写入所有层
            promote_on_read: 读取时是否提升到更高层
        """
        self.l0 = l0_adapter
        self.l1 = l1_adapter
        self.l2 = l2_adapter
        self.write_through = write_through
        self.promote_on_read = promote_on_read
        
        self._adapters = [a for a in [l0_adapter, l1_adapter, l2_adapter] if a]
        self._connected = False
    
    # ==================== 连接管理 ====================
    
    async def connect(self) -> bool:
        if not self._adapters:
            logger.warning("No adapters configured for CompositeAdapter")
            return False
        
        results = await asyncio.gather(
            *[a.connect() for a in self._adapters],
            return_exceptions=True
        )
        
        success_count = sum(1 for r in results if r is True)
        self._connected = success_count > 0
        
        if self._connected:
            logger.info(f"CompositeAdapter connected: {success_count}/{len(self._adapters)} adapters")
        else:
            logger.error("CompositeAdapter failed to connect any adapter")
        
        return self._connected
    
    async def disconnect(self):
        await asyncio.gather(
            *[a.disconnect() for a in self._adapters],
            return_exceptions=True
        )
        self._connected = False
    
    async def is_connected(self) -> bool:
        if not self._adapters:
            return False
        
        results = await asyncio.gather(
            *[a.is_connected() for a in self._adapters],
            return_exceptions=True
        )
        
        return any(r is True for r in results)
    
    def _get_layer_name(self, adapter: PersistenceAdapter) -> str:
        """获取适配器对应的层名称"""
        if adapter is self.l0:
            return "L0"
        elif adapter is self.l1:
            return "L1"
        elif adapter is self.l2:
            return "L2"
        return f"L?"
    
    async def health_check(self) -> Dict[str, Any]:
        health = {
            "status": "healthy" if self._connected else "disconnected",
            "adapter": "composite",
            "layers": {},
        }
        
        for adapter in self._adapters:
            level = self._get_layer_name(adapter)
            try:
                layer_health = await adapter.health_check()
                health["layers"][level] = layer_health
            except Exception as e:
                health["layers"][level] = {"status": "error", "error": str(e)}
        
        # 如果任何层不健康，整体状态为 degraded
        if any(
            h.get("status") != "healthy" 
            for h in health["layers"].values()
        ):
            health["status"] = "degraded"
        
        return health
    
    # ==================== 槽位操作 ====================
    
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        """写入所有层（write-through）"""
        if not self._adapters:
            return False
        
        if self.write_through:
            results = await asyncio.gather(
                *[a.save_slot(namespace, record) for a in self._adapters],
                return_exceptions=True
            )
            return any(r is True for r in results if not isinstance(r, Exception))
        else:
            # 只写入最高层（L2）
            if self.l2:
                return await self.l2.save_slot(namespace, record)
            elif self.l1:
                return await self.l1.save_slot(namespace, record)
            elif self.l0:
                return await self.l0.save_slot(namespace, record)
            return False
    
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        """从最近的层读取，miss 时向下查找并提升"""
        # L0 查找
        if self.l0:
            record = await self.l0.load_slot(namespace, lu_id)
            if record:
                return record
        
        # L1 查找
        if self.l1:
            record = await self.l1.load_slot(namespace, lu_id)
            if record:
                # 提升到 L0
                if self.promote_on_read and self.l0:
                    await self.l0.save_slot(namespace, record)
                return record
        
        # L2 查找
        if self.l2:
            record = await self.l2.load_slot(namespace, lu_id)
            if record:
                # 提升到 L0 和 L1
                if self.promote_on_read:
                    if self.l1:
                        await self.l1.save_slot(namespace, record)
                    if self.l0:
                        await self.l0.save_slot(namespace, record)
                return record
        
        return None
    
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """从所有层删除"""
        if not self._adapters:
            return False
        
        results = await asyncio.gather(
            *[a.delete_slot(namespace, lu_id) for a in self._adapters],
            return_exceptions=True
        )
        
        return any(r is True for r in results if not isinstance(r, Exception))
    
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        """检查任意层是否存在"""
        for adapter in self._adapters:
            if await adapter.slot_exists(namespace, lu_id):
                return True
        return False
    
    # ==================== 批量操作 ====================
    
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        """批量保存到所有层"""
        if not self._adapters or not records:
            return 0
        
        if self.write_through:
            results = await asyncio.gather(
                *[a.save_batch(namespace, records) for a in self._adapters],
                return_exceptions=True
            )
            return max(
                (r for r in results if isinstance(r, int)),
                default=0
            )
        else:
            # 只写入最高层
            if self.l2:
                return await self.l2.save_batch(namespace, records)
            elif self.l1:
                return await self.l1.save_batch(namespace, records)
            elif self.l0:
                return await self.l0.save_batch(namespace, records)
            return 0
    
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        """从最完整的层加载（L2 优先）"""
        # 优先从 L2 加载（最完整）
        if self.l2:
            records = await self.l2.load_active_slots(namespace)
            if records:
                # 预热到 L0 和 L1
                if self.promote_on_read:
                    if self.l1:
                        await self.l1.save_batch(namespace, records)
                    if self.l0:
                        await self.l0.save_batch(namespace, records)
                return records
        
        # 尝试 L1
        if self.l1:
            records = await self.l1.load_active_slots(namespace)
            if records:
                if self.promote_on_read and self.l0:
                    await self.l0.save_batch(namespace, records)
                return records
        
        # 最后尝试 L0
        if self.l0:
            return await self.l0.load_active_slots(namespace)
        
        return []
    
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        """从最完整的层加载所有槽位"""
        if self.l2:
            records = await self.l2.load_all_slots(namespace)
            if records:
                return records
        
        if self.l1:
            records = await self.l1.load_all_slots(namespace)
            if records:
                return records
        
        if self.l0:
            return await self.l0.load_all_slots(namespace)
        
        return []
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        """更新所有层的生命周期"""
        if not self._adapters:
            return False
        
        results = await asyncio.gather(
            *[a.update_lifecycle(namespace, lu_id, new_state) for a in self._adapters],
            return_exceptions=True
        )
        
        return any(r is True for r in results if not isinstance(r, Exception))
    
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]
    ) -> int:
        """批量更新所有层的生命周期"""
        if not self._adapters or not updates:
            return 0
        
        results = await asyncio.gather(
            *[a.update_lifecycle_batch(namespace, updates) for a in self._adapters],
            return_exceptions=True
        )
        
        return max(
            (r for r in results if isinstance(r, int)),
            default=0
        )
    
    # ==================== 统计查询 ====================
    
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        """从最完整的层获取数量"""
        if self.l2:
            return await self.l2.get_slot_count(namespace, state)
        if self.l1:
            return await self.l1.get_slot_count(namespace, state)
        if self.l0:
            return await self.l0.get_slot_count(namespace, state)
        return 0
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """聚合所有层的统计"""
        stats = {
            "namespace": namespace,
            "adapter": "composite",
            "layers": {},
        }
        
        for adapter in self._adapters:
            level = self._get_layer_name(adapter)
            try:
                layer_stats = await adapter.get_statistics(namespace)
                stats["layers"][level] = layer_stats
            except Exception as e:
                stats["layers"][level] = {"error": str(e)}
        
        # 汇总统计
        if self.l2:
            l2_stats = stats["layers"].get("L2", {})
            stats["total_slots"] = l2_stats.get("total_slots", 0)
            stats["state_distribution"] = l2_stats.get("state_distribution", {})
        
        return stats
    
    # ==================== 命中计数 ====================
    
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        """更新所有层的命中计数"""
        if not self._adapters or not lu_ids:
            return True
        
        results = await asyncio.gather(
            *[a.increment_hit_count(namespace, lu_ids) for a in self._adapters],
            return_exceptions=True
        )
        
        return any(r is True for r in results if not isinstance(r, Exception))
    
    # ==================== 层管理 ====================
    
    def get_layer(self, level: int) -> Optional[PersistenceAdapter]:
        """获取指定层的适配器"""
        if level == 0:
            return self.l0
        elif level == 1:
            return self.l1
        elif level == 2:
            return self.l2
        return None
    
    async def warm_cache(self, namespace: str) -> int:
        """
        预热缓存：从 L2 加载到 L0/L1
        
        Returns:
            预热的记录数
        """
        if not self.l2:
            return 0
        
        records = await self.l2.load_active_slots(namespace)
        if not records:
            return 0
        
        count = 0
        if self.l1:
            count = await self.l1.save_batch(namespace, records)
        if self.l0:
            await self.l0.save_batch(namespace, records)
        
        logger.info(f"Warmed cache with {len(records)} records for namespace={namespace}")
        return len(records)
    
    async def flush_to_l2(self, namespace: str) -> int:
        """
        刷新到 L2：将 L0/L1 的数据持久化到 L2
        
        Returns:
            刷新的记录数
        """
        if not self.l2:
            return 0
        
        records = []
        
        # 从 L0 收集
        if self.l0:
            l0_records = await self.l0.load_all_slots(namespace)
            records.extend(l0_records)
        
        # 从 L1 收集（去重）
        if self.l1:
            l1_records = await self.l1.load_all_slots(namespace)
            existing_ids = {r.lu_id for r in records}
            for r in l1_records:
                if r.lu_id not in existing_ids:
                    records.append(r)
        
        if not records:
            return 0
        
        count = await self.l2.save_batch(namespace, records)
        logger.info(f"Flushed {count} records to L2 for namespace={namespace}")
        return count
