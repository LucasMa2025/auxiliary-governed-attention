"""
aga-knowledge 内存持久化适配器

用于测试和开发环境。明文 KV 版本。
"""

import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import OrderedDict

from .base import PersistenceAdapter
from ..types import LifecycleState


class MemoryAdapter(PersistenceAdapter):
    """
    内存持久化适配器（明文 KV 版本）

    特性：
    - 线程安全
    - LRU 淘汰
    - 适合作为测试或开发环境
    - 不包含向量数据
    """

    def __init__(
        self,
        max_slots_per_namespace: int = 128,
        enable_lru: bool = True,
    ):
        self.max_slots = max_slots_per_namespace
        self.enable_lru = enable_lru
        self._storage: Dict[str, OrderedDict[str, Dict[str, Any]]] = {}
        self._audit_log: List[Dict[str, Any]] = []
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
        total = sum(len(ns) for ns in self._storage.values())
        return {
            "status": "healthy" if self._connected else "disconnected",
            "adapter": "memory",
            "namespaces": len(self._storage),
            "total_knowledge": total,
        }

    # ==================== 知识 CRUD ====================

    async def save_knowledge(self, namespace: str, lu_id: str, data: Dict[str, Any]) -> bool:
        with self._lock:
            if namespace not in self._storage:
                self._storage[namespace] = OrderedDict()

            ns_storage = self._storage[namespace]

            # LRU 淘汰
            if self.enable_lru and len(ns_storage) >= self.max_slots:
                if lu_id not in ns_storage:
                    ns_storage.popitem(last=False)

            now = datetime.utcnow().isoformat()
            existing = ns_storage.get(lu_id)

            record = {
                "lu_id": lu_id,
                "namespace": namespace,
                "condition": data.get("condition", ""),
                "decision": data.get("decision", ""),
                "lifecycle_state": data.get("lifecycle_state", "probationary"),
                "trust_tier": data.get("trust_tier", "standard"),
                "hit_count": data.get("hit_count", existing.get("hit_count", 0) if existing else 0),
                "version": (existing.get("version", 0) + 1) if existing else 1,
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
                "metadata": data.get("metadata"),
            }

            ns_storage[lu_id] = record

            if self.enable_lru:
                ns_storage.move_to_end(lu_id)

            return True

    async def load_knowledge(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if namespace not in self._storage:
                return None
            ns_storage = self._storage[namespace]
            record = ns_storage.get(lu_id)
            if record and self.enable_lru:
                ns_storage.move_to_end(lu_id)
            return dict(record) if record else None

    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            ns_storage = self._storage[namespace]
            if lu_id in ns_storage:
                del ns_storage[lu_id]
                return True
            return False

    async def knowledge_exists(self, namespace: str, lu_id: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            return lu_id in self._storage[namespace]

    # ==================== 批量操作 ====================

    async def save_batch(self, namespace: str, records: List[Dict[str, Any]]) -> int:
        count = 0
        for record in records:
            lu_id = record.get("lu_id", "")
            if lu_id and await self.save_knowledge(namespace, lu_id, record):
                count += 1
        return count

    async def load_active_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        with self._lock:
            if namespace not in self._storage:
                return []
            return [
                dict(r) for r in self._storage[namespace].values()
                if r.get("lifecycle_state") != LifecycleState.QUARANTINED.value
            ]

    async def load_all_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        with self._lock:
            if namespace not in self._storage:
                return []
            return [dict(r) for r in self._storage[namespace].values()]

    # ==================== 查询 ====================

    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if namespace not in self._storage:
                return []

            records = list(self._storage[namespace].values())

            if lifecycle_states:
                records = [r for r in records if r.get("lifecycle_state") in lifecycle_states]
            if trust_tiers:
                records = [r for r in records if r.get("trust_tier") in trust_tiers]

            records = records[offset:offset + limit]
            return [dict(r) for r in records]

    # ==================== 生命周期管理 ====================

    async def update_lifecycle(self, namespace: str, lu_id: str, new_state: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            ns_storage = self._storage[namespace]
            if lu_id not in ns_storage:
                return False
            ns_storage[lu_id]["lifecycle_state"] = new_state
            ns_storage[lu_id]["updated_at"] = datetime.utcnow().isoformat()
            return True

    async def update_trust_tier(self, namespace: str, lu_id: str, new_tier: str) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            ns_storage = self._storage[namespace]
            if lu_id not in ns_storage:
                return False
            ns_storage[lu_id]["trust_tier"] = new_tier
            ns_storage[lu_id]["updated_at"] = datetime.utcnow().isoformat()
            return True

    # ==================== 统计 ====================

    async def get_knowledge_count(self, namespace: str, state: Optional[str] = None) -> int:
        with self._lock:
            if namespace not in self._storage:
                return 0
            if state is None:
                return len(self._storage[namespace])
            return sum(
                1 for r in self._storage[namespace].values()
                if r.get("lifecycle_state") == state
            )

    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        with self._lock:
            if namespace not in self._storage:
                return {"namespace": namespace, "total_knowledge": 0, "state_distribution": {}}

            ns_storage = self._storage[namespace]
            state_counts: Dict[str, int] = {}
            total_hits = 0
            for record in ns_storage.values():
                state = record.get("lifecycle_state", "unknown")
                state_counts[state] = state_counts.get(state, 0) + 1
                total_hits += record.get("hit_count", 0)

            return {
                "namespace": namespace,
                "total_knowledge": len(ns_storage),
                "max_slots": self.max_slots,
                "occupancy_ratio": len(ns_storage) / self.max_slots if self.max_slots > 0 else 0,
                "state_distribution": state_counts,
                "total_hits": total_hits,
            }

    async def increment_hit_count(self, namespace: str, lu_ids: List[str]) -> bool:
        with self._lock:
            if namespace not in self._storage:
                return False
            ns_storage = self._storage[namespace]
            for lu_id in lu_ids:
                if lu_id in ns_storage:
                    ns_storage[lu_id]["hit_count"] = ns_storage[lu_id].get("hit_count", 0) + 1
            return True

    # ==================== 命名空间 ====================

    async def get_namespaces(self) -> List[str]:
        with self._lock:
            return list(self._storage.keys())

    # ==================== 审计日志 ====================

    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        with self._lock:
            self._audit_log.append(entry)
            # 限制内存审计日志大小
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-5000:]
            return True

    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            logs = self._audit_log[:]

            if namespace:
                logs = [l for l in logs if l.get("namespace") == namespace]
            if lu_id:
                logs = [l for l in logs if l.get("lu_id") == lu_id]

            # 按时间倒序
            logs.reverse()
            return logs[offset:offset + limit]

    # ==================== 工具方法 ====================

    def clear(self, namespace: Optional[str] = None):
        """清空存储"""
        with self._lock:
            if namespace:
                if namespace in self._storage:
                    self._storage[namespace].clear()
            else:
                self._storage.clear()
