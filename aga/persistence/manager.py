"""
AGA 持久化管理器

提供高层 API 来管理 AGA 状态的持久化。

版本: v3.0
"""
import asyncio
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
import logging

import torch

from .base import PersistenceAdapter, KnowledgeRecord
from ..types import LifecycleState, Slot

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    AGA 持久化管理器
    
    核心设计原则：
    - DB 是治理层：管理 lifecycle、LU 元数据、审计日志
    - AGA 是执行层：只负责推理时的知识检索和融合
    - 增量同步：只同步变更，减少 IO
    """
    
    def __init__(
        self,
        adapter: PersistenceAdapter,
        namespace: str = "default",
        auto_sync: bool = True,
    ):
        """
        初始化管理器
        
        Args:
            adapter: 持久化适配器
            namespace: 命名空间
            auto_sync: 是否自动同步变更
        """
        self.adapter = adapter
        self.namespace = namespace
        self.auto_sync = auto_sync
        self._dirty_lu_ids: Set[str] = set()
        self._connected = False
        self._dirty_lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """连接到持久化层"""
        self._connected = await self.adapter.connect()
        return self._connected
    
    async def disconnect(self):
        """断开连接"""
        await self.adapter.disconnect()
        self._connected = False
    
    # ==================== AGA 集成 ====================
    
    async def save_aga_state(self, aga_module) -> bool:
        """
        保存 AGA 模块状态到持久化层
        
        Args:
            aga_module: AuxiliaryGovernedAttention 实例
        
        Returns:
            是否成功
        """
        try:
            records = []
            
            for i in range(aga_module.num_slots):
                if aga_module.slot_lifecycle[i] == LifecycleState.QUARANTINED:
                    continue
                
                lu_id = aga_module.slot_lu_ids[i]
                if not lu_id:
                    continue
                
                record = KnowledgeRecord(
                    slot_idx=i,
                    lu_id=lu_id,
                    condition=aga_module.slot_conditions[i] or "",
                    decision=aga_module.slot_decisions[i] or "",
                    key_vector=aga_module.aux_keys[i].cpu().tolist(),
                    value_vector=aga_module.aux_values[i].cpu().tolist(),
                    lifecycle_state=aga_module.slot_lifecycle[i].value,
                    namespace=self.namespace,
                    hit_count=aga_module.slot_hit_counts[i],
                    consecutive_misses=aga_module.slot_consecutive_misses[i],
                )
                records.append(record)
            
            if records:
                count = await self.adapter.save_batch(self.namespace, records)
                logger.info(f"Saved {count} slots to persistence for namespace={self.namespace}")
            
            self._dirty_lu_ids.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to save AGA state: {e}")
            return False
    
    async def load_aga_state(self, aga_module) -> int:
        """
        从持久化层加载 AGA 状态
        
        Args:
            aga_module: AuxiliaryGovernedAttention 实例
        
        Returns:
            加载的槽位数量
        """
        try:
            records = await self.adapter.load_active_slots(self.namespace)
            
            if not records:
                logger.info(f"No active slots found for namespace={self.namespace}")
                return 0
            
            device = aga_module.aux_keys.device
            dtype = aga_module.aux_keys.dtype
            
            for record in records:
                slot_idx = record.slot_idx
                if slot_idx >= aga_module.num_slots:
                    continue
                
                key_vector = torch.tensor(record.key_vector, device=device, dtype=dtype)
                value_vector = torch.tensor(record.value_vector, device=device, dtype=dtype)
                
                aga_module.inject_knowledge(
                    slot_idx=slot_idx,
                    key_vector=key_vector,
                    value_vector=value_vector,
                    lu_id=record.lu_id,
                    lifecycle_state=LifecycleState(record.lifecycle_state),
                    condition=record.condition,
                    decision=record.decision,
                )
                
                # 恢复统计信息
                aga_module.slot_hit_counts[slot_idx] = record.hit_count
                aga_module.slot_consecutive_misses[slot_idx] = record.consecutive_misses
            
            logger.info(f"Loaded {len(records)} slots from persistence for namespace={self.namespace}")
            return len(records)
        except Exception as e:
            logger.error(f"Failed to load AGA state: {e}")
            return 0
    
    # ==================== 知识同步 ====================
    
    async def sync_inject_knowledge(
        self,
        aga_module,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        condition: str = "",
        decision: str = "",
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
    ) -> Optional[int]:
        """
        同步注入知识（同时写入 AGA 和持久化层）
        
        Returns:
            槽位索引，失败返回 None
        """
        # 查找空闲槽位
        slot_idx = aga_module.find_free_slot()
        if slot_idx is None:
            logger.error("No free slot available")
            return None
        
        # 写入 AGA
        aga_module.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id=lu_id,
            lifecycle_state=lifecycle_state,
            condition=condition,
            decision=decision,
        )
        
        # 写入持久化层
        record = KnowledgeRecord(
            slot_idx=slot_idx,
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            key_vector=key_vector.cpu().tolist(),
            value_vector=value_vector.cpu().tolist(),
            lifecycle_state=lifecycle_state.value,
            namespace=self.namespace,
        )
        
        await self.adapter.save_slot(self.namespace, record)
        
        return slot_idx
    
    async def sync_update_lifecycle(
        self,
        aga_module,
        lu_id: str,
        new_state: LifecycleState,
    ) -> bool:
        """
        同步更新生命周期
        """
        # 更新 AGA
        slots = aga_module.get_slot_by_lu_id(lu_id)
        for slot_idx in slots:
            aga_module.update_lifecycle(slot_idx, new_state)
            if self.auto_sync:
                async with self._dirty_lock:
                    self._dirty_lu_ids.add(lu_id)
        
        # 更新持久化层
        return await self.adapter.update_lifecycle(self.namespace, lu_id, new_state)
    
    async def sync_quarantine(
        self,
        aga_module,
        lu_id: str,
    ) -> bool:
        """
        同步隔离知识
        """
        # 隔离 AGA
        quarantined = aga_module.quarantine_by_lu_id(lu_id)
        if self.auto_sync:
            async with self._dirty_lock:
                self._dirty_lu_ids.add(lu_id)
        
        # 更新持久化层
        return await self.adapter.update_lifecycle(
            self.namespace, lu_id, LifecycleState.QUARANTINED
        )
    
    # ==================== 增量同步 ====================
    
    async def flush_dirty(self, aga_module) -> int:
        """
        刷新脏数据到持久化层
        
        Returns:
            同步的槽位数量
        """
        async with self._dirty_lock:
            if not self._dirty_lu_ids:
                return 0
            dirty_ids = set(self._dirty_lu_ids)
            self._dirty_lu_ids.clear()

        records = []
        for lu_id in dirty_ids:
            slots = aga_module.get_slot_by_lu_id(lu_id)
            for slot_idx in slots:
                if slot_idx < aga_module.num_slots:
                    record = KnowledgeRecord(
                        slot_idx=slot_idx,
                        lu_id=lu_id,
                        condition=aga_module.slot_conditions[slot_idx] or "",
                        decision=aga_module.slot_decisions[slot_idx] or "",
                        key_vector=aga_module.aux_keys[slot_idx].cpu().tolist(),
                        value_vector=aga_module.aux_values[slot_idx].cpu().tolist(),
                        lifecycle_state=aga_module.slot_lifecycle[slot_idx].value,
                        namespace=self.namespace,
                        hit_count=aga_module.slot_hit_counts[slot_idx],
                        consecutive_misses=aga_module.slot_consecutive_misses[slot_idx],
                    )
                    records.append(record)
        
        count = await self.adapter.save_batch(self.namespace, records)
        return count
    
    async def sync_hit_counts(self, aga_module) -> bool:
        """
        同步命中计数到持久化层
        
        建议定期调用，而非每次推理后调用
        """
        lu_ids_with_hits = []
        for i in range(aga_module.num_slots):
            if aga_module.slot_hit_counts[i] > 0 and aga_module.slot_lu_ids[i]:
                lu_ids_with_hits.append(aga_module.slot_lu_ids[i])
        
        if lu_ids_with_hits:
            return await self.adapter.increment_hit_count(self.namespace, lu_ids_with_hits)
        return True
    
    # ==================== 统计和查询 ====================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取持久化层统计"""
        return await self.adapter.get_statistics(self.namespace)
    
    async def get_slot_count(self, state: Optional[LifecycleState] = None) -> int:
        """获取槽位数量"""
        return await self.adapter.get_slot_count(self.namespace, state)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self.adapter.health_check()
    
    # ==================== Slot 对象支持 ====================
    
    async def save_slot(self, slot: Slot) -> bool:
        """
        保存 Slot 对象
        
        Args:
            slot: Slot 数据类实例
        
        Returns:
            是否成功
        """
        record = KnowledgeRecord(
            slot_idx=slot.slot_idx,
            lu_id=slot.lu_id,
            condition=slot.condition or "",
            decision=slot.decision or "",
            key_vector=slot.key_vector.cpu().tolist() if isinstance(slot.key_vector, torch.Tensor) else slot.key_vector,
            value_vector=slot.value_vector.cpu().tolist() if isinstance(slot.value_vector, torch.Tensor) else slot.value_vector,
            lifecycle_state=slot.lifecycle_state.value,
            namespace=slot.namespace,
            hit_count=slot.hit_count,
            consecutive_misses=slot.consecutive_misses,
            version=slot.version,
        )
        return await self.adapter.save_slot(self.namespace, record)
    
    async def load_slot(self, lu_id: str, device: torch.device = None) -> Optional[Slot]:
        """
        加载 Slot 对象
        
        Args:
            lu_id: 知识单元 ID
            device: 目标设备
        
        Returns:
            Slot 实例或 None
        """
        record = await self.adapter.load_slot(self.namespace, lu_id)
        if not record:
            return None
        
        device = device or torch.device("cpu")
        return Slot(
            slot_idx=record.slot_idx,
            lu_id=record.lu_id,
            key_vector=torch.tensor(record.key_vector, device=device),
            value_vector=torch.tensor(record.value_vector, device=device),
            lifecycle_state=LifecycleState(record.lifecycle_state),
            namespace=record.namespace,
            condition=record.condition,
            decision=record.decision,
            hit_count=record.hit_count,
            consecutive_misses=record.consecutive_misses,
            version=record.version,
        )
    
    async def load_active_slots(self, device: torch.device = None) -> List[Slot]:
        """
        加载所有活跃 Slot 对象
        
        Args:
            device: 目标设备
        
        Returns:
            Slot 列表
        """
        records = await self.adapter.load_active_slots(self.namespace)
        device = device or torch.device("cpu")
        
        slots = []
        for record in records:
            slot = Slot(
                slot_idx=record.slot_idx,
                lu_id=record.lu_id,
                key_vector=torch.tensor(record.key_vector, device=device),
                value_vector=torch.tensor(record.value_vector, device=device),
                lifecycle_state=LifecycleState(record.lifecycle_state),
                namespace=record.namespace,
                condition=record.condition,
                decision=record.decision,
                hit_count=record.hit_count,
                consecutive_misses=record.consecutive_misses,
                version=record.version,
            )
            slots.append(slot)
        
        return slots
