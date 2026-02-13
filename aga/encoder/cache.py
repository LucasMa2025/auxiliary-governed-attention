"""
AGA 编码器缓存模块

提供编码器结果缓存，避免重复编码相同文本。

特性：
- LRU 缓存策略
- 批量编码优化
- 缓存命中率统计
- 可选持久化缓存

版本: v1.1

v1.1 更新:
- 改进缓存键生成（使用更快的哈希）
- 增强持久化缓存的错误处理
- 优化批量编码的并行处理
- 添加缓存统计的 NaN 保护
"""
import hashlib
import threading
import time
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .base import BaseEncoder

logger = logging.getLogger(__name__)

# 尝试使用更快的哈希库
try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_encode_time_ms: float = 0.0
    total_cache_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率（带 NaN 保护）"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        rate = self.hits / total
        # NaN 保护
        return rate if rate == rate else 0.0  # NaN != NaN
    
    @property
    def avg_encode_time_ms(self) -> float:
        """平均编码时间（带 NaN 保护）"""
        if self.misses == 0:
            return 0.0
        avg = self.total_encode_time_ms / self.misses
        return avg if avg == avg else 0.0
    
    @property
    def avg_cache_time_ms(self) -> float:
        """平均缓存查询时间（带 NaN 保护）"""
        if self.hits == 0:
            return 0.0
        avg = self.total_cache_time_ms / self.hits
        return avg if avg == avg else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'avg_encode_time_ms': self.avg_encode_time_ms,
            'avg_cache_time_ms': self.avg_cache_time_ms,
        }


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 10000           # 最大缓存条目数
    ttl_seconds: Optional[int] = None  # 缓存过期时间（秒），None 表示不过期
    enable_stats: bool = True       # 启用统计
    thread_safe: bool = True        # 线程安全


class LRUCache:
    """
    LRU 缓存实现
    
    使用 OrderedDict 实现 O(1) 的 get/put 操作。
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            # 检查过期
            if self.ttl_seconds and time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return value
    
    def put(self, key: str, value: Any) -> bool:
        """存入缓存"""
        with self._lock:
            # 如果已存在，更新并移动到末尾
            if key in self._cache:
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
                return False
            
            # 检查容量
            evicted = False
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                evicted = True
            
            # 添加新条目
            self._cache[key] = (value, time.time())
            return evicted
    
    def contains(self, key: str) -> bool:
        """检查是否存在"""
        with self._lock:
            if key not in self._cache:
                return False
            
            # 检查过期
            if self.ttl_seconds:
                _, timestamp = self._cache[key]
                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    return False
            
            return True
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class CachedEncoder:
    """
    带缓存的编码器包装
    
    包装任意 BaseEncoder，提供透明的缓存层。
    
    使用示例：
        ```python
        from aga.encoder import EncoderFactory, CachedEncoder
        
        # 创建基础编码器
        base_encoder = EncoderFactory.create("openai_compatible", ...)
        
        # 包装为缓存编码器
        encoder = CachedEncoder(base_encoder, max_size=10000)
        
        # 使用（自动缓存）
        vec1 = encoder.encode("Hello")  # 首次编码
        vec2 = encoder.encode("Hello")  # 缓存命中
        
        # 查看统计
        print(encoder.get_stats())
        ```
    """
    
    def __init__(
        self,
        encoder: 'BaseEncoder',
        config: Optional[CacheConfig] = None,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None,
    ):
        """
        初始化缓存编码器
        
        Args:
            encoder: 基础编码器
            config: 缓存配置（优先级高于单独参数）
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间
        """
        self.encoder = encoder
        self.config = config or CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds)
        
        # 缓存实例
        self._cache = LRUCache(
            max_size=self.config.max_size,
            ttl_seconds=self.config.ttl_seconds,
        )
        
        # 统计
        self._stats = CacheStats() if self.config.enable_stats else None
        
        # 线程锁
        self._lock = threading.RLock() if self.config.thread_safe else None
    
    def _make_cache_key(self, text: str, target_dim: Optional[int] = None) -> str:
        """生成缓存键（使用更快的哈希）"""
        content = f"{text}:{target_dim}" if target_dim else text
        content_bytes = content.encode('utf-8')
        
        # 优先使用 xxhash（更快）
        if _HAS_XXHASH:
            return xxhash.xxh64(content_bytes).hexdigest()
        else:
            return hashlib.md5(content_bytes).hexdigest()

    def _lock_context(self):
        return self._lock if self._lock else nullcontext()
    
    def encode(self, text: str) -> List[float]:
        """
        编码文本（带缓存）
        
        Args:
            text: 输入文本
            
        Returns:
            编码向量（原生维度）
        """
        cache_key = self._make_cache_key(text)
        
        # 尝试从缓存获取
        with self._lock_context():
            start_time = time.perf_counter()
            cached = self._cache.get(cache_key)
            
            if cached is not None:
                if self._stats:
                    self._stats.hits += 1
                    self._stats.total_cache_time_ms += (time.perf_counter() - start_time) * 1000
                return cached
        
        # 缓存未命中，执行编码
        encode_start = time.perf_counter()
        vector = self.encoder.encode(text)
        encode_time = (time.perf_counter() - encode_start) * 1000
        
        # 存入缓存
        with self._lock_context():
            evicted = self._cache.put(cache_key, vector)
            
            if self._stats:
                self._stats.misses += 1
                self._stats.total_encode_time_ms += encode_time
                if evicted:
                    self._stats.evictions += 1
        
        return vector
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本（带缓存优化）
        
        Args:
            texts: 文本列表
            
        Returns:
            编码向量列表
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        
        # 第一遍：检查缓存
        with self._lock_context():
            for i, text in enumerate(texts):
                cache_key = self._make_cache_key(text)
                cached = self._cache.get(cache_key)
                
                if cached is not None:
                    results[i] = cached
                    if self._stats:
                        self._stats.hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    if self._stats:
                        self._stats.misses += 1
        
        # 批量编码未缓存的文本
        if uncached_texts:
            encode_start = time.perf_counter()
            
            # 使用基础编码器的批量编码
            if hasattr(self.encoder, 'encode_batch'):
                encoded = self.encoder.encode_batch(uncached_texts)
            else:
                encoded = [self.encoder.encode(t) for t in uncached_texts]
            
            encode_time = (time.perf_counter() - encode_start) * 1000
            if self._stats:
                self._stats.total_encode_time_ms += encode_time
            
            # 存入缓存并填充结果
            with self._lock_context():
                for idx, text, vec in zip(uncached_indices, uncached_texts, encoded):
                    cache_key = self._make_cache_key(text)
                    evicted = self._cache.put(cache_key, vec)
                    results[idx] = vec
                    
                    if self._stats and evicted:
                        self._stats.evictions += 1
        
        return results  # type: ignore
    
    def encode_to_key(self, text: str) -> List[float]:
        """编码为 Key 向量（带缓存）"""
        cache_key = self._make_cache_key(text, self.encoder.key_dim)

        with self._lock_context():
            cached = self._cache.get(cache_key)
            if cached is not None:
                if self._stats:
                    self._stats.hits += 1
                return cached
        
        vector = self.encoder.encode_to_key(text)
        
        with self._lock_context():
            evicted = self._cache.put(cache_key, vector)
            if self._stats:
                self._stats.misses += 1
                if evicted:
                    self._stats.evictions += 1
        return vector
    
    def encode_to_value(self, text: str) -> List[float]:
        """编码为 Value 向量（带缓存）"""
        cache_key = self._make_cache_key(text, self.encoder.value_dim)

        with self._lock_context():
            cached = self._cache.get(cache_key)
            if cached is not None:
                if self._stats:
                    self._stats.hits += 1
                return cached
        
        vector = self.encoder.encode_to_value(text)
        
        with self._lock_context():
            evicted = self._cache.put(cache_key, vector)
            if self._stats:
                self._stats.misses += 1
                if evicted:
                    self._stats.evictions += 1
        return vector
    
    def encode_constraint(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """编码约束对（带缓存）"""
        key_vector = self.encode_to_key(condition)
        value_vector = self.encode_to_value(decision)
        return key_vector, value_vector
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if self._stats is None:
            return {'enabled': False}
        
        with self._lock_context():
            return {
                'enabled': True,
                'cache_size': len(self._cache),
                'max_size': self.config.max_size,
                **self._stats.to_dict(),
            }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock_context():
            self._cache.clear()
            if self._stats:
                self._stats = CacheStats()
    
    def warm_cache(self, texts: List[str]):
        """
        预热缓存
        
        Args:
            texts: 要预热的文本列表
        """
        logger.info(f"Warming cache with {len(texts)} texts...")
        self.encode_batch(texts)
        logger.info(f"Cache warmed. Size: {len(self._cache)}")
    
    # 代理基础编码器的属性
    @property
    def encoder_type(self):
        return self.encoder.encoder_type
    
    @property
    def native_dim(self) -> int:
        return self.encoder.native_dim
    
    @property
    def key_dim(self) -> int:
        return self.encoder.key_dim
    
    @property
    def value_dim(self) -> int:
        return self.encoder.value_dim
    
    @property
    def model_name(self) -> str:
        return self.encoder.model_name
    
    def get_signature(self):
        return self.encoder.get_signature()
    
    def get_info(self) -> Dict[str, Any]:
        info = self.encoder.get_info()
        info['cached'] = True
        info['cache_stats'] = self.get_stats()
        return info
    
    def __repr__(self) -> str:
        return f"<CachedEncoder wrapping {self.encoder} cache_size={len(self._cache)}>"


class PersistentCachedEncoder(CachedEncoder):
    """
    持久化缓存编码器
    
    支持将缓存持久化到磁盘，实现跨进程/重启的缓存复用。
    """
    
    def __init__(
        self,
        encoder: 'BaseEncoder',
        cache_path: str,
        config: Optional[CacheConfig] = None,
        **kwargs,
    ):
        """
        初始化持久化缓存编码器
        
        Args:
            encoder: 基础编码器
            cache_path: 缓存文件路径
            config: 缓存配置
        """
        super().__init__(encoder, config, **kwargs)
        self.cache_path = cache_path
        self._load_cache()
    
    def _load_cache(self):
        """从磁盘加载缓存（带错误恢复）"""
        import json
        import os
        import shutil
        
        if not os.path.exists(self.cache_path):
            return
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_count = 0
            for key, value in data.items():
                try:
                    self._cache.put(key, value)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
            
            logger.info(f"Loaded {loaded_count}/{len(data)} cache entries from {self.cache_path}")
            
        except json.JSONDecodeError as e:
            # 缓存文件损坏，备份并重建
            backup_path = f"{self.cache_path}.corrupted"
            logger.warning(f"Cache file corrupted, backing up to {backup_path}: {e}")
            try:
                shutil.move(self.cache_path, backup_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
    
    def save_cache(self):
        """保存缓存到磁盘"""
        import json
        import os
        
        try:
            # 提取缓存数据
            data = {}
            for key, (value, _) in self._cache._cache.items():
                data[key] = value

            tmp_path = f"{self.cache_path}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)

            os.replace(tmp_path, self.cache_path)
            logger.info(f"Saved {len(data)} cache entries to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_path}: {e}")
    
    def __del__(self):
        """析构时保存缓存"""
        try:
            self.save_cache()
        except Exception as e:
            # 析构时不应抛出异常
            try:
                logger.warning(f"Failed to save cache on destruction: {e}")
            except Exception:
                pass  # 日志系统可能已关闭
