"""
AGA Runtime 本地缓存

管理本地知识槽位数据。
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class CachedSlot:
    """缓存的槽位数据"""
    lu_id: str
    slot_idx: int
    namespace: str
    
    # 向量数据
    key_vector: Any  # torch.Tensor
    value_vector: Any  # torch.Tensor
    
    # 元数据
    condition: Optional[str] = None
    decision: Optional[str] = None
    lifecycle_state: str = "probationary"
    trust_tier: Optional[str] = None
    
    # 统计
    hit_count: int = 0
    last_hit_ts: float = field(default_factory=time.time)
    cached_at: float = field(default_factory=time.time)
    
    def record_hit(self):
        """记录命中"""
        self.hit_count += 1
        self.last_hit_ts = time.time()


class LocalCache:
    """
    本地知识缓存
    
    管理 AGA Runtime 的本地知识槽位。
    """
    
    def __init__(
        self,
        max_slots: int = 100,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """
        初始化缓存
        
        Args:
            max_slots: 最大槽位数
            device: 计算设备
            dtype: 数据类型
        """
        self.max_slots = max_slots
        self.device = device
        self.dtype_str = dtype
        
        # 槽位存储
        self._slots: Dict[str, CachedSlot] = {}  # lu_id -> CachedSlot
        self._slot_idx_map: Dict[int, str] = {}  # slot_idx -> lu_id
        self._next_slot_idx = 0
        self._free_indices: List[int] = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 设备和类型
        if HAS_TORCH:
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
            self._dtype = getattr(torch, dtype, torch.float32)
        else:
            self._device = None
            self._dtype = None
    
    def add(
        self,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = "default",
        condition: Optional[str] = None,
        decision: Optional[str] = None,
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
    ) -> Optional[int]:
        """
        添加知识到缓存
        
        Args:
            lu_id: Learning Unit ID
            key_vector: 条件向量
            value_vector: 决策向量
            namespace: 命名空间
            condition: 条件描述
            decision: 决策描述
            lifecycle_state: 生命周期状态
            trust_tier: 信任层级
        
        Returns:
            分配的槽位索引，如果已满返回 None
        """
        with self._lock:
            # 输入验证
            if not lu_id or not lu_id.strip():
                logger.warning("Invalid lu_id: empty or whitespace")
                return None
            
            if not key_vector or not value_vector:
                logger.warning(f"Invalid vectors for {lu_id}: empty key or value vector")
                return None
            
            if len(key_vector) != len(value_vector):
                logger.warning(f"Vector length mismatch for {lu_id}: key={len(key_vector)}, value={len(value_vector)}")
                # 允许不同长度，只记录警告
            
            # 检查是否已存在
            if lu_id in self._slots:
                return self._slots[lu_id].slot_idx
            
            # 检查是否已满
            if len(self._slots) >= self.max_slots:
                # 尝试淘汰
                evicted = self._evict_one()
                if not evicted:
                    logger.warning(f"Cache full, cannot add {lu_id}")
                    return None
            
            # 分配槽位
            slot_idx = self._allocate_slot_idx()
            if slot_idx is None:
                logger.warning(f"Cache full, cannot allocate slot for {lu_id}")
                return None
            
            # 转换向量
            try:
                if HAS_TORCH:
                    key_tensor = torch.tensor(key_vector, device=self._device, dtype=self._dtype)
                    value_tensor = torch.tensor(value_vector, device=self._device, dtype=self._dtype)
                    
                    # 检查 NaN/Inf
                    if torch.isnan(key_tensor).any() or torch.isinf(key_tensor).any():
                        logger.warning(f"Invalid key_vector for {lu_id}: contains NaN or Inf")
                        self._free_indices.append(slot_idx)
                        return None
                    if torch.isnan(value_tensor).any() or torch.isinf(value_tensor).any():
                        logger.warning(f"Invalid value_vector for {lu_id}: contains NaN or Inf")
                        self._free_indices.append(slot_idx)
                        return None
                else:
                    key_tensor = key_vector
                    value_tensor = value_vector
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to convert vectors for {lu_id}: {e}")
                self._free_indices.append(slot_idx)
                return None
            
            # 创建槽位
            slot = CachedSlot(
                lu_id=lu_id,
                slot_idx=slot_idx,
                namespace=namespace,
                key_vector=key_tensor,
                value_vector=value_tensor,
                condition=condition,
                decision=decision,
                lifecycle_state=lifecycle_state,
                trust_tier=trust_tier,
            )
            
            self._slots[lu_id] = slot
            self._slot_idx_map[slot_idx] = lu_id
            
            logger.debug(f"Added {lu_id} to slot {slot_idx}")
            return slot_idx
    
    def update(
        self,
        lu_id: str,
        lifecycle_state: Optional[str] = None,
        trust_tier: Optional[str] = None,
    ) -> bool:
        """更新槽位状态"""
        with self._lock:
            if lu_id not in self._slots:
                return False
            
            slot = self._slots[lu_id]
            
            if lifecycle_state:
                slot.lifecycle_state = lifecycle_state
            if trust_tier:
                slot.trust_tier = trust_tier
            
            return True
    
    def remove(self, lu_id: str) -> bool:
        """移除槽位"""
        with self._lock:
            if lu_id not in self._slots:
                return False
            
            slot = self._slots.pop(lu_id)
            self._slot_idx_map.pop(slot.slot_idx, None)
            if slot.slot_idx not in self._free_indices:
                self._free_indices.append(slot.slot_idx)
            
            logger.debug(f"Removed {lu_id} from slot {slot.slot_idx}")
            return True
    
    def get(self, lu_id: str) -> Optional[CachedSlot]:
        """获取槽位"""
        with self._lock:
            return self._slots.get(lu_id)
    
    def get_by_idx(self, slot_idx: int) -> Optional[CachedSlot]:
        """通过索引获取槽位"""
        with self._lock:
            lu_id = self._slot_idx_map.get(slot_idx)
            if lu_id:
                return self._slots.get(lu_id)
            return None
    
    def contains(self, lu_id: str) -> bool:
        """检查是否包含"""
        with self._lock:
            return lu_id in self._slots
    
    def get_all(self, namespace: Optional[str] = None) -> List[CachedSlot]:
        """获取所有槽位"""
        with self._lock:
            slots = list(self._slots.values())
            if namespace:
                slots = [s for s in slots if s.namespace == namespace]
            return slots
    
    def get_active(self, namespace: Optional[str] = None) -> List[CachedSlot]:
        """获取活跃槽位（非隔离）"""
        slots = self.get_all(namespace)
        return [s for s in slots if s.lifecycle_state != "quarantined"]
    
    def get_vectors(self, namespace: Optional[str] = None):
        """
        获取所有向量（用于 AGA 模块）
        
        Returns:
            (key_matrix, value_matrix, reliability_vector)
        """
        if not HAS_TORCH:
            raise RuntimeError("需要 PyTorch")
        
        slots = self.get_active(namespace)
        if not slots:
            return None, None, None
        
        # 收集向量
        keys = []
        values = []
        reliabilities = []
        
        reliability_map = {
            "probationary": 0.3,
            "confirmed": 1.0,
            "deprecated": 0.1,
            "quarantined": 0.0,
        }
        
        for slot in slots:
            keys.append(slot.key_vector)
            values.append(slot.value_vector)
            # 未知状态 fallback 到 0.0（安全隔离）
            reliabilities.append(reliability_map.get(slot.lifecycle_state, 0.0))
        
        # 堆叠
        key_matrix = torch.stack(keys)  # [num_slots, dim]
        value_matrix = torch.stack(values)  # [num_slots, dim]
        reliability_vector = torch.tensor(reliabilities, device=self._device, dtype=self._dtype)
        
        return key_matrix, value_matrix, reliability_vector
    
    def clear(self, namespace: Optional[str] = None):
        """清空缓存"""
        with self._lock:
            if namespace:
                to_remove = [lu_id for lu_id, slot in self._slots.items() if slot.namespace == namespace]
                if not to_remove:
                    logger.debug(f"No slots to clear for namespace: {namespace}")
                    return
                for lu_id in to_remove:
                    self.remove(lu_id)
                logger.info(f"Cleared {len(to_remove)} slots for namespace: {namespace}")
            else:
                count = len(self._slots)
                self._slots.clear()
                self._slot_idx_map.clear()
                self._free_indices.clear()
                self._next_slot_idx = 0
                logger.info(f"Cleared all {count} slots from cache")
    
    def _allocate_slot_idx(self) -> Optional[int]:
        """分配槽位索引"""
        if self._free_indices:
            return self._free_indices.pop(0)
        if self._next_slot_idx < self.max_slots:
            idx = self._next_slot_idx
            self._next_slot_idx += 1
            return idx
        return None
    
    def _evict_one(self) -> bool:
        """淘汰一个槽位（LRU）"""
        if not self._slots:
            return False
        
        # 找到最久未使用的
        oldest_lu_id = None
        oldest_ts = float('inf')
        
        for lu_id, slot in self._slots.items():
            # 跳过已确认的
            if slot.lifecycle_state == "confirmed":
                continue
            
            if slot.last_hit_ts < oldest_ts:
                oldest_ts = slot.last_hit_ts
                oldest_lu_id = lu_id
        
        if oldest_lu_id:
            self.remove(oldest_lu_id)
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            slots = list(self._slots.values())
            
            state_dist = {}
            total_hits = 0
            
            for slot in slots:
                state = slot.lifecycle_state
                state_dist[state] = state_dist.get(state, 0) + 1
                total_hits += slot.hit_count
            
            return {
                "total_slots": len(slots),
                "max_slots": self.max_slots,
                "available_slots": self.max_slots - len(slots),
                "state_distribution": state_dist,
                "total_hits": total_hits,
                "device": str(self._device) if self._device else None,
            }
