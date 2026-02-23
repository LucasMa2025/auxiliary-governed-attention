"""
aga/kv_store.py — GPU 常驻 KV 存储

源码映射:
  - 核心存储: 来自 core.py 第 378-386 行 (aux_keys, aux_values buffer)
  - 槽位管理: 来自 runtime/cache.py LocalCache + CachedSlot
  - LRU 淘汰: 来自 production/slot_pool.py SlotPool

设计要点:
  - 预分配 GPU 内存，避免运行时分配
  - LRU 淘汰策略（跳过 pinned 知识）
  - Pin/Unpin 锁定机制，核心知识不可被淘汰
  - 线程安全
  - 支持命名空间隔离

v4.5 变更:
  - 新增 pin/unpin 锁定机制
  - LRU 淘汰跳过 pinned 知识
  - 新增 source 标记（区分 register / retriever 来源）
  - 新增 pinned_count / unpinned_count 统计

显存占用 (以 hidden_dim=4096, bottleneck_dim=64, FP16 为例):
  256 slots:  keys=32KB  + values=2MB   + reliability=512B  ≈ 2.03 MB
  1000 slots: keys=125KB + values=8MB   + reliability=2KB   ≈ 8.13 MB
  5000 slots: keys=625KB + values=40MB  + reliability=10KB  ≈ 40.6 MB
"""
import threading
from typing import Dict, Optional, Tuple, List, Any
from collections import OrderedDict

import torch

from .exceptions import KVStoreError


class KVStore:
    """
    GPU 常驻 KV 存储

    使用方式:
        store = KVStore(max_slots=256, key_dim=64, value_dim=4096)
        store.put("fact_001", key_tensor, value_tensor, reliability=0.9)
        keys, values, reliability = store.get_active()
        store.remove("fact_001")
    """

    def __init__(
        self,
        max_slots: int,
        key_dim: int,
        value_dim: int,
        device: torch.device = None,
    ):
        self.max_slots = max_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 预分配 GPU 内存
        self.keys = torch.zeros(
            max_slots, key_dim, dtype=torch.float16, device=self.device
        )
        self.values = torch.zeros(
            max_slots, value_dim, dtype=torch.float16, device=self.device
        )
        self.reliability = torch.zeros(
            max_slots, dtype=torch.float16, device=self.device
        )
        self.active = torch.zeros(
            max_slots, dtype=torch.bool, device=self.device
        )

        # 索引管理 (CPU)
        self._id_to_slot: Dict[str, int] = {}
        self._slot_to_id: Dict[int, str] = {}
        self._metadata: Dict[str, Dict] = {}
        self._access_order = OrderedDict()  # LRU 跟踪
        self._free_slots: List[int] = list(range(max_slots))
        self._lock = threading.Lock()

        # Pin 锁定机制 — 核心知识不可被 LRU 淘汰
        self._pinned: set = set()

        # get_active() 缓存（避免每次 forward 都创建新 tensor）
        self._active_cache_valid = False
        self._cached_keys: Optional[torch.Tensor] = None
        self._cached_values: Optional[torch.Tensor] = None
        self._cached_reliability: Optional[torch.Tensor] = None

    def put(
        self,
        id: str,
        key: torch.Tensor,
        value: torch.Tensor,
        reliability: float = 1.0,
        metadata: Optional[Dict] = None,
        pinned: bool = False,
    ) -> bool:
        """
        写入知识到 GPU

        Args:
            id: 知识唯一标识
            key: 检索键向量 [key_dim]
            value: 知识值向量 [value_dim]
            reliability: 可靠性分数 (0.0-1.0)
            metadata: 可选元数据
            pinned: 是否锁定（锁定的知识不会被 LRU 淘汰）

        Returns:
            是否写入成功
        """
        with self._lock:
            # 如果 ID 已存在，更新
            if id in self._id_to_slot:
                slot_idx = self._id_to_slot[id]
            else:
                # 分配新槽位
                if not self._free_slots:
                    slot_idx = self._evict()
                    if slot_idx is None:
                        return False
                else:
                    slot_idx = self._free_slots.pop()

            # 写入 GPU
            self.keys[slot_idx] = key.to(self.device, dtype=torch.float16)
            self.values[slot_idx] = value.to(self.device, dtype=torch.float16)
            self.reliability[slot_idx] = reliability
            self.active[slot_idx] = True

            # 更新索引
            self._id_to_slot[id] = slot_idx
            self._slot_to_id[slot_idx] = id
            if metadata:
                self._metadata[id] = metadata
            else:
                self._metadata.setdefault(id, {})
            self._access_order[id] = slot_idx
            self._access_order.move_to_end(id)

            # Pin 锁定
            if pinned:
                self._pinned.add(id)

            # 使缓存失效
            self._active_cache_valid = False

            return True

    def remove(self, id: str) -> bool:
        """
        移除知识（包括 pinned 的知识）

        Args:
            id: 知识唯一标识

        Returns:
            是否移除成功
        """
        with self._lock:
            if id not in self._id_to_slot:
                return False

            slot_idx = self._id_to_slot[id]
            self.active[slot_idx] = False
            self.keys[slot_idx] = 0
            self.values[slot_idx] = 0
            self.reliability[slot_idx] = 0

            del self._id_to_slot[id]
            del self._slot_to_id[slot_idx]
            self._metadata.pop(id, None)
            self._access_order.pop(id, None)
            self._pinned.discard(id)
            self._free_slots.append(slot_idx)

            # 使缓存失效
            self._active_cache_valid = False

            return True

    def get(self, id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        获取单条知识

        Returns:
            (key, value, reliability) 或 None
        """
        with self._lock:
            if id not in self._id_to_slot:
                return None
            slot_idx = self._id_to_slot[id]
            # 更新 LRU
            self._access_order.move_to_end(id)
            return (
                self.keys[slot_idx],
                self.values[slot_idx],
                self.reliability[slot_idx].item(),
            )

    def get_active(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取所有活跃的 (keys, values, reliability)

        使用缓存机制避免每次 forward 都创建新 tensor。
        缓存在 put/remove/clear 时自动失效。

        Returns:
            keys: [active_count, key_dim]
            values: [active_count, value_dim]
            reliability: [active_count]
        """
        if self._active_cache_valid and self._cached_keys is not None:
            return self._cached_keys, self._cached_values, self._cached_reliability

        mask = self.active
        if not mask.any():
            empty_keys = torch.empty(0, self.key_dim, device=self.device, dtype=torch.float16)
            empty_values = torch.empty(0, self.value_dim, device=self.device, dtype=torch.float16)
            empty_rel = torch.empty(0, device=self.device, dtype=torch.float16)
            return empty_keys, empty_values, empty_rel

        self._cached_keys = self.keys[mask]
        self._cached_values = self.values[mask]
        self._cached_reliability = self.reliability[mask]
        self._active_cache_valid = True

        return self._cached_keys, self._cached_values, self._cached_reliability

    def contains(self, id: str) -> bool:
        """检查知识是否存在"""
        return id in self._id_to_slot

    def get_metadata(self, id: str) -> Optional[Dict]:
        """获取知识元数据"""
        return self._metadata.get(id)

    def get_all_ids(self) -> List[str]:
        """获取所有知识 ID"""
        return list(self._id_to_slot.keys())

    # ========== Pin 锁定机制 ==========

    def pin(self, id: str) -> bool:
        """
        锁定知识，防止被 LRU 淘汰

        Args:
            id: 知识唯一标识

        Returns:
            是否锁定成功
        """
        with self._lock:
            if id in self._id_to_slot:
                self._pinned.add(id)
                return True
            return False

    def unpin(self, id: str) -> bool:
        """
        解锁知识，允许被 LRU 淘汰

        Args:
            id: 知识唯一标识

        Returns:
            是否解锁成功
        """
        with self._lock:
            self._pinned.discard(id)
            return True

    def is_pinned(self, id: str) -> bool:
        """检查知识是否被锁定"""
        return id in self._pinned

    @property
    def pinned_count(self) -> int:
        """当前锁定的知识数量"""
        return len(self._pinned)

    @property
    def unpinned_count(self) -> int:
        """当前未锁定的活跃知识数量"""
        return self.count - self.pinned_count

    def clear(self, namespace: Optional[str] = None):
        """
        清空知识

        Args:
            namespace: 如果指定，只清空该命名空间的知识
        """
        with self._lock:
            if namespace is None:
                self.active.fill_(False)
                self.keys.zero_()
                self.values.zero_()
                self.reliability.zero_()
                self._id_to_slot.clear()
                self._slot_to_id.clear()
                self._metadata.clear()
                self._access_order.clear()
                self._pinned.clear()
                self._free_slots = list(range(self.max_slots))
                self._active_cache_valid = False
            else:
                to_remove = [
                    id for id, meta in self._metadata.items()
                    if meta.get("namespace") == namespace
                ]
                for id in to_remove:
                    self._remove_internal(id)

    @property
    def count(self) -> int:
        """当前活跃知识数量"""
        return int(self.active.sum().item())

    @property
    def capacity(self) -> int:
        """最大容量"""
        return self.max_slots

    @property
    def utilization(self) -> float:
        """使用率"""
        return self.count / max(self.max_slots, 1)

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        return {
            "count": self.count,
            "max_slots": self.max_slots,
            "utilization": self.utilization,
            "free_slots": len(self._free_slots),
            "pinned_count": self.pinned_count,
            "unpinned_count": self.unpinned_count,
            "evictable_count": self.unpinned_count,
            "vram_bytes": self._estimate_vram(),
        }

    def _evict(self) -> Optional[int]:
        """
        LRU 淘汰（跳过 pinned 知识）

        淘汰策略:
          1. 遍历 LRU 队列（从最久未访问开始）
          2. 跳过 pinned 的知识
          3. 淘汰第一个未锁定的知识
          4. 如果所有知识都被锁定，返回 None（无法淘汰）
        """
        # 遍历 access_order，找到第一个未锁定的知识
        for candidate_id in list(self._access_order.keys()):
            if candidate_id not in self._pinned:
                slot_idx = self._access_order.pop(candidate_id)
                self.active[slot_idx] = False
                self.keys[slot_idx] = 0
                self.values[slot_idx] = 0
                self.reliability[slot_idx] = 0
                del self._id_to_slot[candidate_id]
                del self._slot_to_id[slot_idx]
                self._metadata.pop(candidate_id, None)
                return slot_idx

        # 所有知识都被锁定，无法淘汰
        return None

    def _remove_internal(self, id: str):
        """内部移除（不加锁）"""
        if id not in self._id_to_slot:
            return
        slot_idx = self._id_to_slot[id]
        self.active[slot_idx] = False
        self.keys[slot_idx] = 0
        self.values[slot_idx] = 0
        self.reliability[slot_idx] = 0
        del self._id_to_slot[id]
        del self._slot_to_id[slot_idx]
        self._metadata.pop(id, None)
        self._access_order.pop(id, None)
        self._pinned.discard(id)
        self._free_slots.append(slot_idx)
        self._active_cache_valid = False

    def _estimate_vram(self) -> int:
        """估算 VRAM 占用（字节）"""
        # FP16 = 2 bytes per element
        key_bytes = self.max_slots * self.key_dim * 2
        value_bytes = self.max_slots * self.value_dim * 2
        reliability_bytes = self.max_slots * 2
        active_bytes = self.max_slots  # bool = 1 byte
        return key_bytes + value_bytes + reliability_bytes + active_bytes
