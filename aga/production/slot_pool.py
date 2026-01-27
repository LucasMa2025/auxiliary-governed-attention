"""
AGA 槽位池管理

核心设计：
- 物理隔离的子空间池（per-namespace）
- 保证 O(1) 推理复杂度
- LRU + hit_count 混合淘汰策略
- 线程安全
"""
import time
import threading
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .config import SlotPoolConfig

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    """知识槽位生命周期状态"""
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)


class EvictionPolicy(str, Enum):
    """淘汰策略"""
    LRU = "lru"                    # 最近最少使用
    HIT_COUNT = "hit_count"        # 命中次数最低
    HYBRID = "hybrid"              # 混合策略


# 生命周期到可靠性的映射
LIFECYCLE_RELIABILITY = {
    LifecycleState.PROBATIONARY: 0.3,
    LifecycleState.CONFIRMED: 1.0,
    LifecycleState.DEPRECATED: 0.1,
    LifecycleState.QUARANTINED: 0.0,
}


@dataclass
class Slot:
    """
    知识槽位
    
    包含 key/value 向量和元数据。
    """
    slot_idx: int
    lu_id: str
    key_vector: torch.Tensor
    value_vector: torch.Tensor
    lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY
    
    # 元数据
    condition: Optional[str] = None
    decision: Optional[str] = None
    namespace: str = "default"
    
    # 统计信息
    hit_count: int = 0
    consecutive_misses: int = 0
    last_hit_ts: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 版本控制
    version: int = 1
    
    @property
    def reliability(self) -> float:
        """获取可靠性值"""
        return LIFECYCLE_RELIABILITY.get(self.lifecycle_state, 0.0)
    
    @property
    def age_days(self) -> float:
        """获取槽位年龄（天）"""
        return (time.time() - self.created_at) / 86400
    
    def record_hit(self):
        """记录命中"""
        self.hit_count += 1
        self.consecutive_misses = 0
        self.last_hit_ts = time.time()
    
    def record_miss(self):
        """记录未命中"""
        self.consecutive_misses += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "slot_idx": self.slot_idx,
            "lu_id": self.lu_id,
            "key_vector": self.key_vector.cpu().tolist(),
            "value_vector": self.value_vector.cpu().tolist(),
            "lifecycle_state": self.lifecycle_state.value,
            "condition": self.condition,
            "decision": self.decision,
            "namespace": self.namespace,
            "hit_count": self.hit_count,
            "consecutive_misses": self.consecutive_misses,
            "last_hit_ts": self.last_hit_ts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device = None) -> "Slot":
        """从字典创建"""
        device = device or torch.device("cpu")
        return cls(
            slot_idx=data["slot_idx"],
            lu_id=data["lu_id"],
            key_vector=torch.tensor(data["key_vector"], device=device),
            value_vector=torch.tensor(data["value_vector"], device=device),
            lifecycle_state=LifecycleState(data["lifecycle_state"]),
            condition=data.get("condition"),
            decision=data.get("decision"),
            namespace=data.get("namespace", "default"),
            hit_count=data.get("hit_count", 0),
            consecutive_misses=data.get("consecutive_misses", 0),
            last_hit_ts=data.get("last_hit_ts", time.time()),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            version=data.get("version", 1),
        )


class SlotPool:
    """
    单个命名空间的槽位池
    
    特性：
    - 固定最大容量（保证 O(1)）
    - 线程安全
    - 支持淘汰策略
    """
    
    def __init__(
        self,
        namespace: str,
        config: SlotPoolConfig,
        device: torch.device = None,
    ):
        self.namespace = namespace
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.max_slots = config.max_slots_per_namespace
        self.hidden_dim = config.hidden_dim
        self.bottleneck_dim = config.bottleneck_dim
        
        # 槽位存储
        self._slots: Dict[int, Slot] = {}
        self._lu_id_to_slot: Dict[str, int] = {}  # lu_id -> slot_idx
        self._free_indices: List[int] = list(range(self.max_slots))
        
        # 向量缓存（用于批量计算）
        self._keys_cache: Optional[torch.Tensor] = None
        self._values_cache: Optional[torch.Tensor] = None
        self._reliability_cache: Optional[torch.Tensor] = None
        self._cache_dirty = True
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._total_hits = 0
        self._total_misses = 0
        self._eviction_count = 0
    
    @property
    def active_count(self) -> int:
        """活跃槽位数"""
        with self._lock:
            return len(self._slots)
    
    @property
    def occupancy_ratio(self) -> float:
        """占用率"""
        return self.active_count / self.max_slots
    
    def add_slot(
        self,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[int]:
        """
        添加槽位
        
        Returns:
            slot_idx 或 None（如果没有空闲槽位）
        """
        with self._lock:
            # 检查是否已存在
            if lu_id in self._lu_id_to_slot:
                return self._update_slot(lu_id, key_vector, value_vector, lifecycle_state)
            
            # 检查是否需要淘汰
            if not self._free_indices:
                if self.config.eviction_enabled:
                    self._evict()
                else:
                    return None
            
            if not self._free_indices:
                return None
            
            # 分配槽位
            slot_idx = self._free_indices.pop(0)
            
            # 处理向量维度
            key_vector = self._normalize_vector(key_vector, self.bottleneck_dim, is_key=True)
            value_vector = self._normalize_vector(value_vector, self.hidden_dim, is_key=False)
            
            # 创建槽位
            slot = Slot(
                slot_idx=slot_idx,
                lu_id=lu_id,
                key_vector=key_vector.to(self.device),
                value_vector=value_vector.to(self.device),
                lifecycle_state=lifecycle_state,
                condition=condition,
                decision=decision,
                namespace=self.namespace,
            )
            
            self._slots[slot_idx] = slot
            self._lu_id_to_slot[lu_id] = slot_idx
            self._cache_dirty = True
            
            logger.debug(f"Added slot {slot_idx} for lu_id={lu_id} in namespace={self.namespace}")
            return slot_idx
    
    def _update_slot(
        self,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lifecycle_state: LifecycleState,
    ) -> int:
        """更新已存在的槽位"""
        slot_idx = self._lu_id_to_slot[lu_id]
        slot = self._slots[slot_idx]
        
        key_vector = self._normalize_vector(key_vector, self.bottleneck_dim, is_key=True)
        value_vector = self._normalize_vector(value_vector, self.hidden_dim, is_key=False)
        
        slot.key_vector = key_vector.to(self.device)
        slot.value_vector = value_vector.to(self.device)
        slot.lifecycle_state = lifecycle_state
        slot.updated_at = time.time()
        slot.version += 1
        
        self._cache_dirty = True
        return slot_idx
    
    def _normalize_vector(
        self,
        vector: torch.Tensor,
        target_dim: int,
        is_key: bool,
    ) -> torch.Tensor:
        """归一化向量"""
        if vector.dim() > 1:
            vector = vector.flatten()
        
        # 调整维度
        if vector.shape[0] < target_dim:
            vector = F.pad(vector, (0, target_dim - vector.shape[0]))
        elif vector.shape[0] > target_dim:
            vector = vector[:target_dim]
        
        # 范数裁剪
        if self.config.enable_norm_clipping:
            norm = vector.norm()
            if norm > 0:
                target_norm = self.config.key_norm_target if is_key else self.config.value_norm_target
                vector = vector / (norm + 1e-8) * target_norm
        
        return vector
    
    def remove_slot(self, lu_id: str) -> bool:
        """移除槽位"""
        with self._lock:
            if lu_id not in self._lu_id_to_slot:
                return False
            
            slot_idx = self._lu_id_to_slot.pop(lu_id)
            del self._slots[slot_idx]
            self._free_indices.append(slot_idx)
            self._cache_dirty = True
            
            logger.debug(f"Removed slot {slot_idx} for lu_id={lu_id}")
            return True
    
    def quarantine_slot(self, lu_id: str) -> bool:
        """隔离槽位（标记为 QUARANTINED 并清零 value）"""
        with self._lock:
            if lu_id not in self._lu_id_to_slot:
                return False
            
            slot_idx = self._lu_id_to_slot[lu_id]
            slot = self._slots[slot_idx]
            slot.lifecycle_state = LifecycleState.QUARANTINED
            slot.value_vector.zero_()
            slot.updated_at = time.time()
            self._cache_dirty = True
            
            logger.info(f"Quarantined slot {slot_idx} for lu_id={lu_id}")
            return True
    
    def update_lifecycle(self, lu_id: str, new_state: LifecycleState) -> bool:
        """更新生命周期状态"""
        with self._lock:
            if lu_id not in self._lu_id_to_slot:
                return False
            
            slot_idx = self._lu_id_to_slot[lu_id]
            slot = self._slots[slot_idx]
            slot.lifecycle_state = new_state
            slot.updated_at = time.time()
            self._cache_dirty = True
            
            return True
    
    def get_slot(self, lu_id: str) -> Optional[Slot]:
        """获取槽位"""
        with self._lock:
            if lu_id not in self._lu_id_to_slot:
                return None
            slot_idx = self._lu_id_to_slot[lu_id]
            return self._slots.get(slot_idx)
    
    def get_slot_by_idx(self, slot_idx: int) -> Optional[Slot]:
        """通过索引获取槽位"""
        with self._lock:
            return self._slots.get(slot_idx)
    
    def get_vectors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取所有活跃槽位的向量（用于批量计算）
        
        Returns:
            keys: [active_count, bottleneck_dim]
            values: [active_count, hidden_dim]
            reliability: [active_count]
        """
        with self._lock:
            if self._cache_dirty or self._keys_cache is None:
                self._rebuild_cache()
            return self._keys_cache, self._values_cache, self._reliability_cache
    
    def _rebuild_cache(self):
        """重建向量缓存"""
        if not self._slots:
            self._keys_cache = torch.zeros(0, self.bottleneck_dim, device=self.device)
            self._values_cache = torch.zeros(0, self.hidden_dim, device=self.device)
            self._reliability_cache = torch.zeros(0, device=self.device)
            self._cache_dirty = False
            return
        
        # 只包含非隔离的槽位
        active_slots = [
            s for s in self._slots.values()
            if s.lifecycle_state != LifecycleState.QUARANTINED
        ]
        
        if not active_slots:
            self._keys_cache = torch.zeros(0, self.bottleneck_dim, device=self.device)
            self._values_cache = torch.zeros(0, self.hidden_dim, device=self.device)
            self._reliability_cache = torch.zeros(0, device=self.device)
        else:
            self._keys_cache = torch.stack([s.key_vector for s in active_slots])
            self._values_cache = torch.stack([s.value_vector for s in active_slots])
            self._reliability_cache = torch.tensor(
                [s.reliability for s in active_slots],
                device=self.device
            )
        
        self._cache_dirty = False
    
    def record_hits(self, slot_indices: List[int]):
        """记录命中"""
        with self._lock:
            hit_set = set(slot_indices)
            for slot_idx, slot in self._slots.items():
                if slot.lifecycle_state == LifecycleState.QUARANTINED:
                    continue
                if slot_idx in hit_set:
                    slot.record_hit()
                    self._total_hits += 1
                else:
                    slot.record_miss()
                    self._total_misses += 1
    
    def _evict(self):
        """执行淘汰"""
        if not self._slots:
            return
        
        # 计算需要淘汰的数量
        current_count = len(self._slots)
        target_count = int(self.max_slots * self.config.eviction_target_ratio)
        evict_count = current_count - target_count
        
        if evict_count <= 0:
            return
        
        # 计算每个槽位的淘汰分数（分数越高越容易被淘汰）
        scores = []
        for slot_idx, slot in self._slots.items():
            # 跳过 CONFIRMED 状态的槽位（除非严重过期）
            if slot.lifecycle_state == LifecycleState.CONFIRMED and slot.age_days < self.config.eviction_max_age_days:
                score = -float('inf')  # 不淘汰
            else:
                # 混合评分
                # 1. hit_count 越低分数越高
                hit_score = 1.0 / (slot.hit_count + 1)
                # 2. 最后命中时间越久分数越高
                time_score = time.time() - slot.last_hit_ts
                # 3. consecutive_misses 越高分数越高
                miss_score = slot.consecutive_misses
                # 4. 年龄越大分数越高
                age_score = slot.age_days
                
                score = hit_score * 0.3 + time_score * 0.3 + miss_score * 0.2 + age_score * 0.2
            
            scores.append((slot_idx, score))
        
        # 按分数排序，淘汰分数最高的
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(evict_count, len(scores))):
            slot_idx, score = scores[i]
            if score == -float('inf'):
                break
            
            slot = self._slots[slot_idx]
            lu_id = slot.lu_id
            
            del self._slots[slot_idx]
            del self._lu_id_to_slot[lu_id]
            self._free_indices.append(slot_idx)
            self._eviction_count += 1
            
            logger.info(f"Evicted slot {slot_idx} (lu_id={lu_id}, score={score:.4f})")
        
        self._cache_dirty = True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            state_counts = {}
            for state in LifecycleState:
                state_counts[state.value] = sum(
                    1 for s in self._slots.values() if s.lifecycle_state == state
                )
            
            hit_counts = [s.hit_count for s in self._slots.values() if s.hit_count > 0]
            
            return {
                "namespace": self.namespace,
                "max_slots": self.max_slots,
                "active_slots": len(self._slots),
                "free_slots": len(self._free_indices),
                "occupancy_ratio": self.occupancy_ratio,
                "state_distribution": state_counts,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "eviction_count": self._eviction_count,
                "avg_hit_count": sum(hit_counts) / len(hit_counts) if hit_counts else 0,
                "max_hit_count": max(hit_counts) if hit_counts else 0,
            }
    
    def export_all(self) -> List[Dict[str, Any]]:
        """导出所有槽位"""
        with self._lock:
            return [slot.to_dict() for slot in self._slots.values()]
    
    def import_slots(self, slots_data: List[Dict[str, Any]]):
        """导入槽位"""
        with self._lock:
            for data in slots_data:
                slot = Slot.from_dict(data, self.device)
                if slot.slot_idx < self.max_slots:
                    self._slots[slot.slot_idx] = slot
                    self._lu_id_to_slot[slot.lu_id] = slot.slot_idx
                    if slot.slot_idx in self._free_indices:
                        self._free_indices.remove(slot.slot_idx)
            self._cache_dirty = True


class SlotSubspacePool:
    """
    子空间槽位池管理器
    
    管理多个 namespace 的槽位池，提供物理隔离。
    """
    
    def __init__(self, config: SlotPoolConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cpu")
        
        self._pools: Dict[str, SlotPool] = {}
        self._lock = threading.RLock()
    
    def get_pool(self, namespace: str) -> SlotPool:
        """获取或创建命名空间的槽位池"""
        with self._lock:
            if namespace not in self._pools:
                self._pools[namespace] = SlotPool(namespace, self.config, self.device)
            return self._pools[namespace]
    
    def remove_pool(self, namespace: str) -> bool:
        """移除命名空间的槽位池"""
        with self._lock:
            if namespace in self._pools:
                del self._pools[namespace]
                return True
            return False
    
    def get_all_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        with self._lock:
            return list(self._pools.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取所有槽位池的统计信息"""
        with self._lock:
            return {
                "total_namespaces": len(self._pools),
                "per_namespace": {
                    ns: pool.get_statistics()
                    for ns, pool in self._pools.items()
                },
            }

