"""
AGA 内存持久化适配器

用于测试和作为 L0 缓存层。

版本: v3.0
"""
import asyncio
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import OrderedDict

from .base import PersistenceAdapter, KnowledgeRecord
from ..types import LifecycleState


class MemoryAdapter(PersistenceAdapter):
    """
    内存持久化适配器
    
    特性：
    - 线程安全
    - LRU 淘汰
    - 适合作为 L0 缓存或测试
    """
    
    def __init__(
        self,
        max_slots_per_namespace: int = 128,
        enable_lru: bool = True,
    ):
        """
        初始化内存适配器
        
        Args:
            max_slots_per_namespace: 每个命名空间的最大槽位数
            enable_lru: 是否启用 LRU 淘汰
        """
        self.max_slots = max_slots_per_namespace
        self.enable_lru = enable_lru
        
        # 存储: namespace -> lu_id -> KnowledgeRecord
        self._storage: Dict[str, OrderedDict[str, KnowledgeRecord]] = {}
        self._lock = threading.RLock()
        self._connected = False
    
    # ==================== 连接管理 ====================
    
    async def connect(self) -> bool:
        self._connected = True
        return True
    
    async def disconnect(self):
        self._connected = False
    
    async def is_connected(self) -> bool:
        return self._connected
    
    async def health_check(self) -> Dict[str, Any]:
        total_slots = sum(len(ns) for ns in self._storage.values())
        return {
            "status": "healthy" if self._connected else "disconnected",
            "adapter": "memory",
            "namespaces": len(self._storage),
            "total_slots": total_slots,
        }
    
    # ==================== 槽位操作 ====================
    
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        with self._lock:
            if namespace not in self._storage:
                self._storage[namespace] = OrderedDict()
            
            ns_storage = self._storage[namespace]
            
            # LRU 淘汰
            if self.enable_lru and len(ns_storage) >= self.max_slots:
                if record.lu_id not in ns_storage:
                    # 淘汰最旧的
                    ns_storage.popitem(last=False)
            
            # 更新时间戳
            now = datetime.now().isoformat()
            if record.created_at is None:
                record.created_at = now
            record.updated_at = now
            
            # 保存
            ns_storage[record.lu_id] = record
            
            # 移动到末尾（LRU）
            if self.enable_lru:
                ns_storage.move_to_end(record.lu_id)
            
            return True
    
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        with self._lock:
            if namespace not in self._storage:
                return None
            
            ns_storage = self._storage[namespace]
            record = ns_storage.get(lu_id)
            
            if record and self.enable_lru:
                ns_storage.move_to_end(lu_id)
            
            return record
    
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            
            ns_storage = self._storage[namespace]
            if lu_id in ns_storage:
                del ns_storage[lu_id]
                return True
            return False
    
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            return lu_id in self._storage[namespace]
    
    # ==================== 批量操作 ====================
    
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        count = 0
        for record in records:
            if await self.save_slot(namespace, record):
                count += 1
        return count
    
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        with self._lock:
            if namespace not in self._storage:
                return []
            
            return [
                record for record in self._storage[namespace].values()
                if record.lifecycle_state != LifecycleState.QUARANTINED.value
            ]
    
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        with self._lock:
            if namespace not in self._storage:
                return []
            return list(self._storage[namespace].values())
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            
            ns_storage = self._storage[namespace]
            if lu_id not in ns_storage:
                return False
            
            record = ns_storage[lu_id]
            record.lifecycle_state = new_state.value
            record.updated_at = datetime.now().isoformat()
            return True
    
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]
    ) -> int:
        count = 0
        for lu_id, new_state in updates:
            if await self.update_lifecycle(namespace, lu_id, new_state):
                count += 1
        return count
    
    # ==================== 统计查询 ====================
    
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        with self._lock:
            if namespace not in self._storage:
                return 0
            
            if state is None:
                return len(self._storage[namespace])
            
            return sum(
                1 for record in self._storage[namespace].values()
                if record.lifecycle_state == state.value
            )
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        with self._lock:
            if namespace not in self._storage:
                return {
                    "namespace": namespace,
                    "total_slots": 0,
                    "state_distribution": {},
                }
            
            ns_storage = self._storage[namespace]
            
            state_counts = {}
            total_hits = 0
            for record in ns_storage.values():
                state = record.lifecycle_state
                state_counts[state] = state_counts.get(state, 0) + 1
                total_hits += record.hit_count
            
            return {
                "namespace": namespace,
                "total_slots": len(ns_storage),
                "max_slots": self.max_slots,
                "occupancy_ratio": len(ns_storage) / self.max_slots,
                "state_distribution": state_counts,
                "total_hits": total_hits,
                "enable_lru": self.enable_lru,
            }
    
    # ==================== 命中计数 ====================
    
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            
            ns_storage = self._storage[namespace]
            for lu_id in lu_ids:
                if lu_id in ns_storage:
                    ns_storage[lu_id].hit_count += 1
            
            return True
    
    # ==================== 额外方法 ====================
    
    def clear(self, namespace: Optional[str] = None):
        """清空存储"""
        with self._lock:
            if namespace:
                if namespace in self._storage:
                    self._storage[namespace].clear()
            else:
                self._storage.clear()
    
    def get_all_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        with self._lock:
            return list(self._storage.keys())
