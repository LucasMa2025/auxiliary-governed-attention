"""
AGA 动态知识加载器

实现冷知识的动态加载，支持从分层存储 (Hot/Warm/Cold) 按需加载知识到 Hot Pool。

核心功能:
- 分层缓存查找 (Hot → Warm → Cold)
- 批量加载优化
- 超时保护
- 预取策略
- Fail-Open 机制

版本: v1.1
更新:
- 添加配置校验
- 增强异常处理
- 使用 public API 访问 TieredSlotStorage
- 添加向量范数验证
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..production.slot_pool import Slot
    from ..production.dynamic_slots import TieredSlotStorage, SlotTier
    from ..persistence.manager import PersistenceManager


@dataclass
class DynamicLoaderConfig:
    """动态加载器配置"""
    # 基础配置
    enabled: bool = True
    
    # 加载配置
    max_cold_load_per_request: int = 50  # 单次最大冷加载数
    cold_load_timeout_ms: float = 10.0  # 冷加载超时
    batch_load_size: int = 20  # 批量加载大小
    
    # 预取配置
    prefetch_enabled: bool = True
    prefetch_threshold: int = 3  # 访问次数阈值触发预取
    prefetch_batch_size: int = 10  # 预取批量大小
    
    # Warm 缓存配置
    warm_cache_size: int = 2000  # Warm 层容量
    warm_cache_ttl_seconds: float = 3600.0  # Warm 缓存 TTL
    
    # 性能配置
    enable_async_load: bool = True  # 启用异步加载
    load_retry_count: int = 2  # 加载重试次数
    load_retry_delay_ms: float = 5.0  # 重试延迟
    
    # 向量验证
    validate_vectors: bool = True  # 验证加载的向量
    max_vector_norm: float = 10.0  # 最大向量范数
    
    def __post_init__(self):
        """配置校验"""
        # 批量大小校验
        if self.batch_load_size <= 0:
            raise ValueError("batch_load_size must be positive")
        
        # 最大冷加载数校验（防止 overload）
        if self.max_cold_load_per_request > 100:
            logger.warning("max_cold_load_per_request capped at 100 for safety")
            self.max_cold_load_per_request = 100
        
        if self.max_cold_load_per_request <= 0:
            self.max_cold_load_per_request = 50
        
        # Warm 缓存大小校验
        if self.warm_cache_size <= 0:
            self.warm_cache_size = 2000
        
        # 超时校验
        if self.cold_load_timeout_ms <= 0:
            self.cold_load_timeout_ms = 10.0


@dataclass
class LoadResult:
    """加载结果"""
    loaded_slots: List[Any]  # Slot 列表
    hot_hits: int  # Hot 层命中数
    warm_hits: int  # Warm 层命中数
    cold_loads: int  # Cold 层加载数
    load_time_ms: float  # 加载耗时
    failed_lu_ids: List[str]  # 加载失败的 lu_id
    
    @property
    def total_loaded(self) -> int:
        """总加载数"""
        return self.hot_hits + self.warm_hits + self.cold_loads
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率 (Hot + Warm)"""
        total = self.total_loaded + len(self.failed_lu_ids)
        if total == 0:
            return 0.0
        return (self.hot_hits + self.warm_hits) / total


class WarmCache:
    """
    Warm 层缓存
    
    LRU 缓存，用于存储从 Cold 层加载的槽位，
    减少重复的 Cold 层访问。
    """
    
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def get(self, lu_id: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            if lu_id not in self._cache:
                self._stats["misses"] += 1
                return None
            
            # 检查 TTL
            if time.time() - self._timestamps[lu_id] > self.ttl_seconds:
                self._remove(lu_id)
                self._stats["misses"] += 1
                return None
            
            # 移动到末尾 (LRU)
            self._cache.move_to_end(lu_id)
            self._stats["hits"] += 1
            return self._cache[lu_id]
    
    def put(self, lu_id: str, slot: Any):
        """添加缓存项"""
        with self._lock:
            # 如果已存在，更新
            if lu_id in self._cache:
                self._cache.move_to_end(lu_id)
                self._cache[lu_id] = slot
                self._timestamps[lu_id] = time.time()
                return
            
            # 检查容量
            while len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # 添加
            self._cache[lu_id] = slot
            self._timestamps[lu_id] = time.time()
    
    def remove(self, lu_id: str) -> bool:
        """移除缓存项"""
        with self._lock:
            return self._remove(lu_id)
    
    def _remove(self, lu_id: str) -> bool:
        """内部移除（无锁）"""
        if lu_id not in self._cache:
            return False
        del self._cache[lu_id]
        del self._timestamps[lu_id]
        return True
    
    def _evict_oldest(self):
        """淘汰最老的项"""
        if not self._cache:
            return
        oldest_id = next(iter(self._cache))
        self._remove(oldest_id)
        self._stats["evictions"] += 1
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self._stats["hits"] / max(1, total),
            }


class DynamicKnowledgeLoader:
    """
    动态知识加载器
    
    负责从分层存储中按需加载知识槽位到 Hot Pool。
    
    加载流程:
    1. 检查 Hot 层 (TieredSlotStorage._hot_slots)
    2. 检查 Warm 层 (本地 LRU 缓存)
    3. 从 Cold 层加载 (持久化存储)
    4. 提升到 Hot 层
    """
    
    def __init__(
        self,
        config: DynamicLoaderConfig,
        tiered_storage: Optional["TieredSlotStorage"] = None,
        persistence: Optional["PersistenceManager"] = None,
    ):
        self.config = config
        self.tiered_storage = tiered_storage
        self.persistence = persistence
        
        # Warm 缓存
        self._warm_cache = WarmCache(
            max_size=config.warm_cache_size,
            ttl_seconds=config.warm_cache_ttl_seconds,
        )
        
        # 访问计数（用于预取决策）
        self._access_counts: Dict[str, int] = {}
        self._access_lock = threading.Lock()
        
        # 统计
        self._stats = {
            "total_loads": 0,
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_loads": 0,
            "cold_failures": 0,
            "total_load_time_ms": 0.0,
            "avg_load_time_ms": 0.0,
            "prefetch_count": 0,
        }
        
        # 线程锁
        self._lock = threading.RLock()
    
    def set_tiered_storage(self, tiered_storage: "TieredSlotStorage"):
        """设置分层存储"""
        self.tiered_storage = tiered_storage
    
    def set_persistence(self, persistence: "PersistenceManager"):
        """设置持久化管理器"""
        self.persistence = persistence
    
    def load_candidates(
        self, 
        candidate_lu_ids: List[str],
        timeout_ms: Optional[float] = None,
    ) -> LoadResult:
        """
        加载候选槽位
        
        Args:
            candidate_lu_ids: 候选 lu_id 列表
            timeout_ms: 超时时间（毫秒）
            
        Returns:
            LoadResult: 加载结果
        """
        start_time = time.perf_counter()
        timeout_ms = timeout_ms or self.config.cold_load_timeout_ms
        
        loaded_slots = []
        hot_hits = 0
        warm_hits = 0
        cold_loads = 0
        failed_lu_ids = []
        
        # 需要从 Cold 加载的
        to_load_cold = []
        
        with self._lock:
            for lu_id in candidate_lu_ids:
                # 更新访问计数
                self._record_access(lu_id)
                
                # 1. 检查 Hot 层
                slot = self._get_from_hot(lu_id)
                if slot is not None:
                    loaded_slots.append(slot)
                    hot_hits += 1
                    continue
                
                # 2. 检查 Warm 层
                slot = self._warm_cache.get(lu_id)
                if slot is not None:
                    # 提升到 Hot 层
                    self._promote_to_hot(lu_id, slot)
                    loaded_slots.append(slot)
                    warm_hits += 1
                    continue
                
                # 3. 需要从 Cold 加载
                to_load_cold.append(lu_id)
            
            # 批量加载 Cold 层
            if to_load_cold:
                # 限制单次加载数量
                to_load_cold = to_load_cold[:self.config.max_cold_load_per_request]
                
                cold_slots, failures = self._batch_load_cold(
                    to_load_cold, 
                    timeout_ms
                )
                
                for slot in cold_slots:
                    # 添加到 Warm 缓存
                    self._warm_cache.put(slot.lu_id, slot)
                    # 提升到 Hot 层
                    self._promote_to_hot(slot.lu_id, slot)
                    loaded_slots.append(slot)
                    cold_loads += 1
                
                failed_lu_ids.extend(failures)
        
        # 更新统计
        elapsed = (time.perf_counter() - start_time) * 1000
        self._update_stats(hot_hits, warm_hits, cold_loads, len(failed_lu_ids), elapsed)
        
        return LoadResult(
            loaded_slots=loaded_slots,
            hot_hits=hot_hits,
            warm_hits=warm_hits,
            cold_loads=cold_loads,
            load_time_ms=elapsed,
            failed_lu_ids=failed_lu_ids,
        )
    
    def _get_from_hot(self, lu_id: str) -> Optional[Any]:
        """从 Hot 层获取"""
        if self.tiered_storage is None:
            return None
        
        try:
            # 使用 public API: get_slot 方法（如果存在）
            if hasattr(self.tiered_storage, 'get_slot'):
                return self.tiered_storage.get_slot(lu_id, tier_hint='hot')
            
            # 回退: 使用 get 方法但不更新统计
            if hasattr(self.tiered_storage, 'get'):
                return self.tiered_storage.get(lu_id)
            
            return None
        except Exception as e:
            logger.debug(f"Failed to get from hot: {e}")
            return None
    
    def _promote_to_hot(self, lu_id: str, slot: Any):
        """提升到 Hot 层"""
        if self.tiered_storage is None:
            return
        
        try:
            # 使用 public API: promote 方法（如果存在）
            if hasattr(self.tiered_storage, 'promote'):
                from ..production.dynamic_slots import SlotTier
                self.tiered_storage.promote(lu_id, SlotTier.HOT)
            elif hasattr(self.tiered_storage, 'put'):
                from ..production.dynamic_slots import SlotTier
                self.tiered_storage.put(slot, SlotTier.HOT)
        except Exception as e:
            logger.warning(f"Failed to promote slot {lu_id} to hot: {e}")
    
    def _batch_load_cold(
        self, 
        lu_ids: List[str], 
        timeout_ms: float
    ) -> Tuple[List[Any], List[str]]:
        """批量加载 Cold 层"""
        if self.persistence is None:
            return [], lu_ids
        
        loaded = []
        failed = []
        
        start_time = time.perf_counter()
        batch_size = self.config.batch_load_size
        
        for i in range(0, len(lu_ids), batch_size):
            # 检查超时
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed >= timeout_ms:
                logger.warning(f"Cold load timeout after {elapsed:.1f}ms")
                failed.extend(lu_ids[i:])
                break
            
            batch = lu_ids[i:i + batch_size]
            
            for retry in range(self.config.load_retry_count + 1):
                try:
                    slots_data = self._load_slots_from_persistence(batch)
                    for data in slots_data:
                        slot = self._create_slot_from_data(data)
                        if slot is not None:
                            loaded.append(slot)
                        else:
                            failed.append(data.get("lu_id", "unknown"))
                    break
                except Exception as e:
                    if retry < self.config.load_retry_count:
                        time.sleep(self.config.load_retry_delay_ms / 1000)
                    else:
                        logger.warning(f"Failed to load cold slots after retries: {e}")
                        failed.extend(batch)
        
        return loaded, failed
    
    def _load_slots_from_persistence(self, lu_ids: List[str]) -> List[Dict]:
        """从持久化层加载槽位数据"""
        if self.persistence is None:
            return []
        
        # 尝试批量加载
        if hasattr(self.persistence, 'load_slots_batch'):
            return self.persistence.load_slots_batch(lu_ids)
        
        # 逐个加载
        results = []
        for lu_id in lu_ids:
            if hasattr(self.persistence, 'load_slot'):
                data = self.persistence.load_slot(lu_id)
                if data:
                    results.append(data)
        return results
    
    def _create_slot_from_data(self, data: Dict) -> Optional[Any]:
        """从数据创建 Slot 对象"""
        try:
            from ..production.slot_pool import Slot
            slot = Slot.from_dict(data)
            
            # 向量验证（与 core.py 范数裁剪对齐）
            if self.config.validate_vectors and slot is not None:
                if hasattr(slot, 'key_vector') and slot.key_vector is not None:
                    if not self._validate_vector(slot.key_vector):
                        logger.warning(f"Invalid vector for slot {data.get('lu_id', 'unknown')}")
                        return None
            
            return slot
        except Exception as e:
            logger.warning(f"Failed to create slot from data: {e}")
            return None
    
    def _validate_vector(self, vector: np.ndarray) -> bool:
        """验证向量有效性"""
        try:
            # NaN/Inf 检查
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                return False
            
            # 范数检查
            norm = np.linalg.norm(vector)
            if norm > self.config.max_vector_norm:
                logger.debug(f"Vector norm {norm:.2f} exceeds max {self.config.max_vector_norm}")
                return False
            
            return True
        except Exception:
            return False
    
    def _record_access(self, lu_id: str):
        """记录访问"""
        with self._access_lock:
            self._access_counts[lu_id] = self._access_counts.get(lu_id, 0) + 1
    
    def _update_stats(
        self, 
        hot: int, 
        warm: int, 
        cold: int, 
        failed: int, 
        elapsed: float
    ):
        """更新统计"""
        self._stats["total_loads"] += 1
        self._stats["hot_hits"] += hot
        self._stats["warm_hits"] += warm
        self._stats["cold_loads"] += cold
        self._stats["cold_failures"] += failed
        self._stats["total_load_time_ms"] += elapsed
        self._stats["avg_load_time_ms"] = (
            self._stats["total_load_time_ms"] / self._stats["total_loads"]
        )
    
    def prefetch(self, lu_ids: List[str]):
        """
        预取槽位到 Warm 缓存
        
        用于预测性加载，减少后续请求的 Cold 加载延迟。
        """
        if not self.config.prefetch_enabled:
            return
        
        # 过滤已在缓存中的
        to_prefetch = []
        for lu_id in lu_ids[:self.config.prefetch_batch_size]:
            if self._warm_cache.get(lu_id) is None:
                to_prefetch.append(lu_id)
        
        if not to_prefetch:
            return
        
        # 异步预取
        if self.config.enable_async_load:
            import threading
            thread = threading.Thread(
                target=self._do_prefetch,
                args=(to_prefetch,),
                daemon=True,
            )
            thread.start()
        else:
            self._do_prefetch(to_prefetch)
    
    def _do_prefetch(self, lu_ids: List[str]):
        """执行预取"""
        try:
            slots, _ = self._batch_load_cold(
                lu_ids, 
                self.config.cold_load_timeout_ms * 2  # 预取可以更宽松
            )
            for slot in slots:
                self._warm_cache.put(slot.lu_id, slot)
            self._stats["prefetch_count"] += len(slots)
        except Exception as e:
            logger.debug(f"Prefetch failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._stats["hot_hits"] + self._stats["warm_hits"] + self._stats["cold_loads"]
            return {
                **self._stats,
                "hot_hit_rate": self._stats["hot_hits"] / max(1, total),
                "warm_hit_rate": self._stats["warm_hits"] / max(1, total),
                "cold_load_rate": self._stats["cold_loads"] / max(1, total),
                "warm_cache": self._warm_cache.get_statistics(),
            }
    
    def clear_warm_cache(self):
        """清空 Warm 缓存"""
        self._warm_cache.clear()
    
    def invalidate(self, lu_id: str):
        """使缓存失效"""
        self._warm_cache.remove(lu_id)
