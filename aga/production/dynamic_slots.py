"""
AGA 动态槽位扩展模块

实现运行时槽位容量的动态调整，支持：
- 自动扩容：当占用率超过阈值时自动扩展
- 自动缩容：当占用率低于阈值时自动收缩
- 分层存储：热/温/冷槽位分层管理
- 内存预算：基于内存预算的容量控制

版本: v1.0
"""
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .config import SlotPoolConfig
from .slot_pool import Slot, SlotPool, LifecycleState

logger = logging.getLogger(__name__)


class ScalingPolicy(str, Enum):
    """扩缩容策略"""
    FIXED = "fixed"              # 固定容量
    AUTO_SCALE = "auto_scale"    # 自动扩缩容
    MEMORY_BUDGET = "memory_budget"  # 基于内存预算


class SlotTier(str, Enum):
    """槽位层级"""
    HOT = "hot"      # 热槽位 (GPU 显存)
    WARM = "warm"    # 温槽位 (CPU 内存)
    COLD = "cold"    # 冷槽位 (持久化存储)


@dataclass
class DynamicSlotConfig:
    """动态槽位配置"""
    # 基础配置
    initial_capacity: int = 64
    min_capacity: int = 16
    max_capacity: int = 512
    
    # 扩缩容策略
    scaling_policy: ScalingPolicy = ScalingPolicy.AUTO_SCALE
    
    # 扩容配置
    expand_threshold: float = 0.85      # 占用率超过此值触发扩容
    expand_factor: float = 1.5          # 扩容倍数
    expand_cooldown_seconds: float = 60.0  # 扩容冷却时间
    
    # 缩容配置
    shrink_threshold: float = 0.3       # 占用率低于此值触发缩容
    shrink_factor: float = 0.7          # 缩容倍数
    shrink_cooldown_seconds: float = 300.0  # 缩容冷却时间
    shrink_min_age_seconds: float = 600.0   # 最小存活时间后才能缩容
    
    # 分层配置
    hot_tier_capacity: int = 128        # 热层容量 (GPU)
    warm_tier_capacity: int = 512       # 温层容量 (CPU)
    cold_tier_enabled: bool = True      # 是否启用冷层
    
    # 内存预算 (bytes)
    memory_budget_bytes: Optional[int] = None  # None 表示不限制
    slot_memory_estimate_bytes: int = 32768    # 单个槽位估计内存 (32KB)
    
    # 预热配置
    preload_on_startup: bool = True
    preload_top_k: int = 64             # 启动时预加载 top-k 热门槽位


@dataclass
class ScalingEvent:
    """扩缩容事件"""
    timestamp: float
    event_type: str  # "expand" | "shrink" | "tier_promote" | "tier_demote"
    old_capacity: int
    new_capacity: int
    trigger_reason: str
    namespace: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "old_capacity": self.old_capacity,
            "new_capacity": self.new_capacity,
            "trigger_reason": self.trigger_reason,
            "namespace": self.namespace,
        }


class TieredSlotStorage:
    """
    分层槽位存储
    
    实现三层存储架构：
    - Hot (GPU): 高频访问槽位，常驻显存
    - Warm (CPU): 中频访问槽位，CPU 内存
    - Cold (Disk): 低频访问槽位，持久化存储
    """
    
    def __init__(
        self,
        namespace: str,
        config: DynamicSlotConfig,
        base_config: SlotPoolConfig,
        device: torch.device = None,
    ):
        self.namespace = namespace
        self.config = config
        self.base_config = base_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # 分层存储
        self._hot_slots: Dict[str, Slot] = {}      # lu_id -> Slot (GPU)
        self._warm_slots: Dict[str, Slot] = {}     # lu_id -> Slot (CPU)
        self._cold_refs: Dict[str, Dict] = {}      # lu_id -> metadata (仅引用)
        
        # 访问统计
        self._access_counts: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        
        # LRU 顺序
        self._hot_lru: OrderedDict = OrderedDict()
        self._warm_lru: OrderedDict = OrderedDict()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计
        self._stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_hits": 0,
            "promotions": 0,
            "demotions": 0,
        }
    
    def get(self, lu_id: str) -> Optional[Slot]:
        """获取槽位（自动提升层级）"""
        with self._lock:
            # 更新访问统计
            self._access_counts[lu_id] = self._access_counts.get(lu_id, 0) + 1
            self._last_access[lu_id] = time.time()
            
            # 检查热层
            if lu_id in self._hot_slots:
                self._stats["hot_hits"] += 1
                # 更新 LRU
                self._hot_lru.move_to_end(lu_id)
                return self._hot_slots[lu_id]
            
            # 检查温层
            if lu_id in self._warm_slots:
                self._stats["warm_hits"] += 1
                slot = self._warm_slots[lu_id]
                # 提升到热层
                self._promote_to_hot(lu_id, slot)
                return slot
            
            # 检查冷层
            if lu_id in self._cold_refs:
                self._stats["cold_hits"] += 1
                # 需要从持久化加载
                return None  # 调用者需要从持久化层加载
            
            return None
    
    def put(self, slot: Slot, tier: SlotTier = SlotTier.HOT):
        """存入槽位"""
        with self._lock:
            lu_id = slot.lu_id
            
            # 移除旧位置
            self._remove_from_all_tiers(lu_id)
            
            if tier == SlotTier.HOT:
                self._put_hot(lu_id, slot)
            elif tier == SlotTier.WARM:
                self._put_warm(lu_id, slot)
            else:
                self._put_cold(lu_id, slot)
    
    def _put_hot(self, lu_id: str, slot: Slot):
        """存入热层"""
        # 检查容量
        while len(self._hot_slots) >= self.config.hot_tier_capacity:
            self._demote_oldest_hot()
        
        # 确保在 GPU
        slot.key_vector = slot.key_vector.to(self.device)
        slot.value_vector = slot.value_vector.to(self.device)
        
        self._hot_slots[lu_id] = slot
        self._hot_lru[lu_id] = True
    
    def _put_warm(self, lu_id: str, slot: Slot):
        """存入温层"""
        # 检查容量
        while len(self._warm_slots) >= self.config.warm_tier_capacity:
            self._demote_oldest_warm()
        
        # 确保在 CPU
        slot.key_vector = slot.key_vector.to(self.cpu_device)
        slot.value_vector = slot.value_vector.to(self.cpu_device)
        
        self._warm_slots[lu_id] = slot
        self._warm_lru[lu_id] = True
    
    def _put_cold(self, lu_id: str, slot: Slot):
        """存入冷层（仅保存引用）"""
        self._cold_refs[lu_id] = {
            "lu_id": lu_id,
            "lifecycle_state": slot.lifecycle_state.value,
            "hit_count": slot.hit_count,
            "created_at": slot.created_at,
        }
    
    def _promote_to_hot(self, lu_id: str, slot: Slot):
        """提升到热层"""
        # 从温层移除
        if lu_id in self._warm_slots:
            del self._warm_slots[lu_id]
            if lu_id in self._warm_lru:
                del self._warm_lru[lu_id]
        
        # 加入热层
        self._put_hot(lu_id, slot)
        self._stats["promotions"] += 1
    
    def _demote_oldest_hot(self):
        """降级最老的热槽位"""
        if not self._hot_lru:
            return
        
        # 获取最老的
        lu_id = next(iter(self._hot_lru))
        slot = self._hot_slots.pop(lu_id)
        del self._hot_lru[lu_id]
        
        # 降级到温层
        self._put_warm(lu_id, slot)
        self._stats["demotions"] += 1
    
    def _demote_oldest_warm(self):
        """降级最老的温槽位"""
        if not self._warm_lru:
            return
        
        # 获取最老的
        lu_id = next(iter(self._warm_lru))
        slot = self._warm_slots.pop(lu_id)
        del self._warm_lru[lu_id]
        
        # 降级到冷层
        self._put_cold(lu_id, slot)
        self._stats["demotions"] += 1
    
    def _remove_from_all_tiers(self, lu_id: str):
        """从所有层移除"""
        if lu_id in self._hot_slots:
            del self._hot_slots[lu_id]
            if lu_id in self._hot_lru:
                del self._hot_lru[lu_id]
        
        if lu_id in self._warm_slots:
            del self._warm_slots[lu_id]
            if lu_id in self._warm_lru:
                del self._warm_lru[lu_id]
        
        if lu_id in self._cold_refs:
            del self._cold_refs[lu_id]
    
    def remove(self, lu_id: str) -> bool:
        """移除槽位"""
        with self._lock:
            existed = (
                lu_id in self._hot_slots or
                lu_id in self._warm_slots or
                lu_id in self._cold_refs
            )
            self._remove_from_all_tiers(lu_id)
            return existed
    
    def get_hot_vectors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取热层向量（用于推理）"""
        with self._lock:
            if not self._hot_slots:
                return (
                    torch.zeros(0, self.base_config.bottleneck_dim, device=self.device),
                    torch.zeros(0, self.base_config.hidden_dim, device=self.device),
                    torch.zeros(0, device=self.device),
                )
            
            active_slots = [
                s for s in self._hot_slots.values()
                if s.lifecycle_state != LifecycleState.QUARANTINED
            ]
            
            if not active_slots:
                return (
                    torch.zeros(0, self.base_config.bottleneck_dim, device=self.device),
                    torch.zeros(0, self.base_config.hidden_dim, device=self.device),
                    torch.zeros(0, device=self.device),
                )
            
            keys = torch.stack([s.key_vector for s in active_slots])
            values = torch.stack([s.value_vector for s in active_slots])
            reliability = torch.tensor(
                [s.reliability for s in active_slots],
                device=self.device
            )
            
            return keys, values, reliability
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "hot_count": len(self._hot_slots),
                "warm_count": len(self._warm_slots),
                "cold_count": len(self._cold_refs),
                "total_count": len(self._hot_slots) + len(self._warm_slots) + len(self._cold_refs),
                "hot_capacity": self.config.hot_tier_capacity,
                "warm_capacity": self.config.warm_tier_capacity,
                **self._stats,
            }


class DynamicSlotPool:
    """
    动态槽位池
    
    支持运行时容量调整的槽位池实现。
    
    特性：
    - 自动扩缩容
    - 分层存储
    - 内存预算控制
    - 扩缩容事件回调
    """
    
    def __init__(
        self,
        namespace: str,
        config: DynamicSlotConfig,
        base_config: SlotPoolConfig,
        device: torch.device = None,
        on_scaling_event: Optional[Callable[[ScalingEvent], None]] = None,
    ):
        self.namespace = namespace
        self.config = config
        self.base_config = base_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 回调
        self._on_scaling_event = on_scaling_event
        
        # 当前容量
        self._current_capacity = config.initial_capacity
        
        # 分层存储
        self._tiered_storage = TieredSlotStorage(
            namespace=namespace,
            config=config,
            base_config=base_config,
            device=device,
        )
        
        # 主槽位池（用于兼容现有接口）
        self._slot_pool = SlotPool(
            namespace=namespace,
            config=base_config,
            device=device,
        )
        
        # 扩缩容状态
        self._last_expand_time = 0.0
        self._last_shrink_time = 0.0
        self._created_at = time.time()
        
        # 扩缩容历史
        self._scaling_history: List[ScalingEvent] = []
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 后台监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
    
    @property
    def current_capacity(self) -> int:
        """当前容量"""
        return self._current_capacity
    
    @property
    def active_count(self) -> int:
        """活跃槽位数"""
        return self._slot_pool.active_count
    
    @property
    def occupancy_ratio(self) -> float:
        """占用率"""
        if self._current_capacity == 0:
            return 0.0
        return self.active_count / self._current_capacity
    
    def start_monitor(self, interval_seconds: float = 30.0):
        """启动后台监控"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(f"Started dynamic slot monitor for namespace={self.namespace}")
    
    def stop_monitor(self):
        """停止后台监控"""
        if self._monitor_thread is None:
            return
        
        self._stop_monitor.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        logger.info(f"Stopped dynamic slot monitor for namespace={self.namespace}")
    
    def _monitor_loop(self, interval_seconds: float):
        """监控循环"""
        while not self._stop_monitor.wait(interval_seconds):
            try:
                self._check_and_scale()
            except Exception as e:
                logger.error(f"Error in scaling monitor: {e}")
    
    def _check_and_scale(self):
        """检查并执行扩缩容"""
        if self.config.scaling_policy == ScalingPolicy.FIXED:
            return
        
        with self._lock:
            current_time = time.time()
            occupancy = self.occupancy_ratio
            
            # 检查扩容
            if occupancy >= self.config.expand_threshold:
                if current_time - self._last_expand_time >= self.config.expand_cooldown_seconds:
                    self._expand()
            
            # 检查缩容
            elif occupancy <= self.config.shrink_threshold:
                age = current_time - self._created_at
                if (age >= self.config.shrink_min_age_seconds and
                    current_time - self._last_shrink_time >= self.config.shrink_cooldown_seconds):
                    self._shrink()
    
    def _expand(self):
        """执行扩容"""
        old_capacity = self._current_capacity
        new_capacity = min(
            int(old_capacity * self.config.expand_factor),
            self.config.max_capacity
        )
        
        if new_capacity <= old_capacity:
            logger.warning(f"Cannot expand: already at max capacity {self.config.max_capacity}")
            return
        
        # 检查内存预算
        if self.config.memory_budget_bytes is not None:
            estimated_memory = new_capacity * self.config.slot_memory_estimate_bytes
            if estimated_memory > self.config.memory_budget_bytes:
                new_capacity = self.config.memory_budget_bytes // self.config.slot_memory_estimate_bytes
                if new_capacity <= old_capacity:
                    logger.warning("Cannot expand: memory budget exceeded")
                    return
        
        # 执行扩容
        self._current_capacity = new_capacity
        self._slot_pool.max_slots = new_capacity
        self._slot_pool._free_indices.extend(range(old_capacity, new_capacity))
        self._slot_pool._cache_dirty = True
        self._last_expand_time = time.time()
        
        # 记录事件
        event = ScalingEvent(
            timestamp=time.time(),
            event_type="expand",
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            trigger_reason=f"occupancy={self.occupancy_ratio:.2%}",
            namespace=self.namespace,
        )
        self._scaling_history.append(event)
        
        if self._on_scaling_event:
            self._on_scaling_event(event)
        
        logger.info(f"Expanded slot pool: {old_capacity} -> {new_capacity} "
                   f"(namespace={self.namespace})")
    
    def _shrink(self):
        """执行缩容"""
        old_capacity = self._current_capacity
        new_capacity = max(
            int(old_capacity * self.config.shrink_factor),
            self.config.min_capacity,
            self.active_count + 1  # 至少保留当前活跃槽位
        )
        
        if new_capacity >= old_capacity:
            return
        
        # 执行缩容
        self._current_capacity = new_capacity
        self._slot_pool.max_slots = new_capacity
        
        # 清理超出范围的空闲索引
        self._slot_pool._free_indices = [
            idx for idx in self._slot_pool._free_indices
            if idx < new_capacity
        ]
        self._slot_pool._cache_dirty = True
        
        self._last_shrink_time = time.time()
        
        # 记录事件
        event = ScalingEvent(
            timestamp=time.time(),
            event_type="shrink",
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            trigger_reason=f"occupancy={self.occupancy_ratio:.2%}",
            namespace=self.namespace,
        )
        self._scaling_history.append(event)
        
        if self._on_scaling_event:
            self._on_scaling_event(event)
        
        logger.info(f"Shrunk slot pool: {old_capacity} -> {new_capacity} "
                   f"(namespace={self.namespace})")
    
    def add_slot(
        self,
        lu_id: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[int]:
        """添加槽位"""
        with self._lock:
            # 检查是否需要扩容
            if self.occupancy_ratio >= self.config.expand_threshold:
                self._expand()
            
            # 委托给底层槽位池
            slot_idx = self._slot_pool.add_slot(
                lu_id=lu_id,
                key_vector=key_vector,
                value_vector=value_vector,
                lifecycle_state=lifecycle_state,
                condition=condition,
                decision=decision,
            )
            
            # 同步到分层存储
            if slot_idx is not None:
                slot = self._slot_pool.get_slot(lu_id)
                if slot:
                    self._tiered_storage.put(slot, SlotTier.HOT)
            
            return slot_idx
    
    def remove_slot(self, lu_id: str) -> bool:
        """移除槽位"""
        with self._lock:
            result = self._slot_pool.remove_slot(lu_id)
            self._tiered_storage.remove(lu_id)
            return result
    
    def get_slot(self, lu_id: str) -> Optional[Slot]:
        """获取槽位"""
        return self._slot_pool.get_slot(lu_id)
    
    def get_vectors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取向量（用于推理）"""
        return self._slot_pool.get_vectors()
    
    def update_lifecycle(self, lu_id: str, new_state: LifecycleState) -> bool:
        """更新生命周期状态"""
        return self._slot_pool.update_lifecycle(lu_id, new_state)
    
    def quarantine_slot(self, lu_id: str) -> bool:
        """隔离槽位"""
        return self._slot_pool.quarantine_slot(lu_id)
    
    def resize(self, new_capacity: int) -> bool:
        """
        手动调整容量
        
        Args:
            new_capacity: 新容量
            
        Returns:
            是否成功
        """
        with self._lock:
            if new_capacity < self.config.min_capacity:
                logger.warning(f"Cannot resize below min_capacity={self.config.min_capacity}")
                return False
            
            if new_capacity > self.config.max_capacity:
                logger.warning(f"Cannot resize above max_capacity={self.config.max_capacity}")
                return False
            
            if new_capacity < self.active_count:
                logger.warning(f"Cannot resize below active_count={self.active_count}")
                return False
            
            old_capacity = self._current_capacity
            self._current_capacity = new_capacity
            self._slot_pool.max_slots = new_capacity
            
            if new_capacity > old_capacity:
                self._slot_pool._free_indices.extend(range(old_capacity, new_capacity))
            else:
                self._slot_pool._free_indices = [
                    idx for idx in self._slot_pool._free_indices
                    if idx < new_capacity
                ]
            
            # 记录事件
            event = ScalingEvent(
                timestamp=time.time(),
                event_type="resize",
                old_capacity=old_capacity,
                new_capacity=new_capacity,
                trigger_reason="manual_resize",
                namespace=self.namespace,
            )
            self._scaling_history.append(event)
            
            if self._on_scaling_event:
                self._on_scaling_event(event)
            
            logger.info(f"Resized slot pool: {old_capacity} -> {new_capacity}")
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            base_stats = self._slot_pool.get_statistics()
            tiered_stats = self._tiered_storage.get_statistics()
            
            return {
                **base_stats,
                "current_capacity": self._current_capacity,
                "min_capacity": self.config.min_capacity,
                "max_capacity": self.config.max_capacity,
                "scaling_policy": self.config.scaling_policy.value,
                "tiered_storage": tiered_stats,
                "scaling_events_count": len(self._scaling_history),
                "last_expand_time": self._last_expand_time,
                "last_shrink_time": self._last_shrink_time,
            }
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取扩缩容历史"""
        with self._lock:
            return [e.to_dict() for e in self._scaling_history[-limit:]]


class DynamicSlotManager:
    """
    动态槽位管理器
    
    管理多个命名空间的动态槽位池。
    """
    
    def __init__(
        self,
        config: DynamicSlotConfig,
        base_config: SlotPoolConfig,
        device: torch.device = None,
    ):
        self.config = config
        self.base_config = base_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._pools: Dict[str, DynamicSlotPool] = {}
        self._lock = threading.RLock()
        
        # 全局扩缩容回调
        self._scaling_callbacks: List[Callable[[ScalingEvent], None]] = []
    
    def get_pool(self, namespace: str) -> DynamicSlotPool:
        """获取或创建动态槽位池"""
        with self._lock:
            if namespace not in self._pools:
                pool = DynamicSlotPool(
                    namespace=namespace,
                    config=self.config,
                    base_config=self.base_config,
                    device=self.device,
                    on_scaling_event=self._on_scaling_event,
                )
                self._pools[namespace] = pool
                pool.start_monitor()
            return self._pools[namespace]
    
    def remove_pool(self, namespace: str) -> bool:
        """移除动态槽位池"""
        with self._lock:
            if namespace in self._pools:
                self._pools[namespace].stop_monitor()
                del self._pools[namespace]
                return True
            return False
    
    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]):
        """添加扩缩容回调"""
        self._scaling_callbacks.append(callback)
    
    def _on_scaling_event(self, event: ScalingEvent):
        """处理扩缩容事件"""
        for callback in self._scaling_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in scaling callback: {e}")
    
    def get_all_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        with self._lock:
            return list(self._pools.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取所有池的统计信息"""
        with self._lock:
            total_capacity = sum(p.current_capacity for p in self._pools.values())
            total_active = sum(p.active_count for p in self._pools.values())
            
            return {
                "total_namespaces": len(self._pools),
                "total_capacity": total_capacity,
                "total_active": total_active,
                "overall_occupancy": total_active / total_capacity if total_capacity > 0 else 0,
                "per_namespace": {
                    ns: pool.get_statistics()
                    for ns, pool in self._pools.items()
                },
            }
    
    def shutdown(self):
        """关闭所有池"""
        with self._lock:
            for pool in self._pools.values():
                pool.stop_monitor()
            self._pools.clear()
            logger.info("Shutdown all dynamic slot pools")
