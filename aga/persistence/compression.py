"""
AGA KV 向量压缩模块

提供高效的向量压缩和解压功能，支持：
- 多种精度级别 (FP32, FP16, BF16, INT8)
- 多种压缩算法 (zlib, lz4, zstd)
- 分层压缩策略
- 延迟解压缓存

版本: v1.0
"""
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional, Dict, Any, List, Union
import threading
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import lz4.frame as lz4
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False

try:
    import zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


class Precision(IntEnum):
    """精度级别"""
    FP32 = 0    # 32位浮点 (无损)
    FP16 = 1    # 16位浮点 (推荐)
    BF16 = 2    # Brain Float 16
    INT8 = 3    # 8位整数量化


class Compression(IntEnum):
    """压缩算法"""
    NONE = 0    # 无压缩
    ZLIB = 1    # zlib (通用，压缩比高)
    LZ4 = 2     # lz4 (速度快，压缩比中等)
    ZSTD = 3    # zstd (平衡)


@dataclass
class CompressionConfig:
    """压缩配置"""
    precision: Precision = Precision.FP16
    compression: Compression = Compression.ZLIB
    zlib_level: int = 6           # zlib 压缩级别 (1-9)
    lz4_level: int = 0            # lz4 压缩级别 (0-16)
    zstd_level: int = 3           # zstd 压缩级别 (1-22)
    
    # 量化参数 (INT8)
    quantize_symmetric: bool = True
    quantize_per_channel: bool = False
    
    @classmethod
    def for_speed(cls) -> 'CompressionConfig':
        """速度优先配置"""
        return cls(
            precision=Precision.FP16,
            compression=Compression.LZ4 if _HAS_LZ4 else Compression.NONE,
        )
    
    @classmethod
    def for_size(cls) -> 'CompressionConfig':
        """压缩比优先配置"""
        return cls(
            precision=Precision.FP16,
            compression=Compression.ZSTD if _HAS_ZSTD else Compression.ZLIB,
            zstd_level=6,
            zlib_level=9,
        )
    
    @classmethod
    def for_quality(cls) -> 'CompressionConfig':
        """质量优先配置（无损）"""
        return cls(
            precision=Precision.FP32,
            compression=Compression.ZLIB,
            zlib_level=6,
        )


class VectorCompressor:
    """
    向量压缩器
    
    提供 KV 向量的压缩和解压功能。
    
    使用示例：
        ```python
        compressor = VectorCompressor(CompressionConfig(
            precision=Precision.FP16,
            compression=Compression.LZ4,
        ))
        
        # 压缩
        compressed = compressor.compress_vectors(key_vector, value_vector)
        
        # 解压
        key_arr, value_arr = compressor.decompress_vectors(compressed)
        
        # 获取压缩比
        ratio = compressor.get_compression_ratio(key_vector, value_vector)
        ```
    """
    
    # 魔数和版本
    MAGIC = b'CVF1'
    VERSION = 1
    HEADER_SIZE = 8
    METADATA_SIZE = 16
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        
        # 验证依赖
        if self.config.compression == Compression.LZ4 and not _HAS_LZ4:
            logger.warning("LZ4 not available, falling back to ZLIB")
            self.config.compression = Compression.ZLIB
        if self.config.compression == Compression.ZSTD and not _HAS_ZSTD:
            logger.warning("ZSTD not available, falling back to ZLIB")
            self.config.compression = Compression.ZLIB
    
    def compress_vectors(
        self,
        key_vector: Union[List[float], np.ndarray],
        value_vector: Union[List[float], np.ndarray],
    ) -> bytes:
        """
        压缩 KV 向量对
        
        Args:
            key_vector: Key 向量 (bottleneck_dim)
            value_vector: Value 向量 (hidden_dim)
        
        Returns:
            压缩后的字节数据
            
        Raises:
            SerializationError: 向量转换或压缩失败
        """
        from .base import SerializationError
        
        # 转换为 numpy 数组
        try:
            key_arr = np.array(key_vector, dtype=np.float32)
            value_arr = np.array(value_vector, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to convert vectors to numpy array: {e}")
        
        # 验证向量有效性
        if key_arr.size == 0:
            raise SerializationError("key_vector is empty")
        if value_arr.size == 0:
            raise SerializationError("value_vector is empty")
        
        # 检查 NaN/Inf
        if np.isnan(key_arr).any() or np.isinf(key_arr).any():
            raise SerializationError("key_vector contains NaN or Inf values")
        if np.isnan(value_arr).any() or np.isinf(value_arr).any():
            raise SerializationError("value_vector contains NaN or Inf values")
        
        # 精度转换
        key_bytes = self._convert_precision(key_arr)
        value_bytes = self._convert_precision(value_arr)
        
        # 压缩
        key_compressed = self._compress_bytes(key_bytes)
        value_compressed = self._compress_bytes(value_bytes)
        
        # 构建输出
        return self._build_output(
            key_dim=len(key_vector),
            value_dim=len(value_vector),
            key_data=key_compressed,
            value_data=value_compressed,
        )
    
    def decompress_vectors(
        self,
        data: bytes,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        解压 KV 向量对
        
        Args:
            data: 压缩后的字节数据
        
        Returns:
            (key_vector, value_vector) 元组
            
        Raises:
            SerializationError: 数据格式无效或解压失败
        """
        from .base import SerializationError
        
        if not data or len(data) < self.HEADER_SIZE + self.METADATA_SIZE:
            raise SerializationError("Invalid compressed data: too short")
        
        try:
            # 解析头部
            header = self._parse_header(data)
            
            # 验证版本兼容性
            if header['version'] > self.VERSION:
                raise SerializationError(
                    f"Unsupported version: {header['version']} (max supported: {self.VERSION})"
                )
            
            # 解析元数据
            metadata = self._parse_metadata(data)
            
            # 提取压缩数据
            offset = self.HEADER_SIZE + self.METADATA_SIZE
            key_data = data[offset:offset + metadata['key_compressed_size']]
            offset += metadata['key_compressed_size']
            value_data = data[offset:offset + metadata['value_compressed_size']]
            
            # 解压
            key_bytes = self._decompress_bytes(key_data, header['compression'])
            value_bytes = self._decompress_bytes(value_data, header['compression'])
            
            # 精度恢复
            key_arr = self._restore_precision(key_bytes, metadata['key_dim'], header['precision'])
            value_arr = self._restore_precision(value_bytes, metadata['value_dim'], header['precision'])
            
            return key_arr, value_arr
            
        except (struct.error, ValueError) as e:
            raise SerializationError(f"Failed to decompress vectors: {e}")
    
    def _convert_precision(self, arr: np.ndarray) -> bytes:
        """转换精度"""
        if self.config.precision == Precision.FP32:
            return arr.astype(np.float32).tobytes()
        elif self.config.precision == Precision.FP16:
            return arr.astype(np.float16).tobytes()
        elif self.config.precision == Precision.BF16:
            return self._to_bfloat16(arr).tobytes()
        elif self.config.precision == Precision.INT8:
            return self._quantize_int8(arr)
        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")
    
    def _restore_precision(self, data: bytes, dim: int, precision: int) -> np.ndarray:
        """恢复精度"""
        if precision == Precision.FP32:
            return np.frombuffer(data, dtype=np.float32)
        elif precision == Precision.FP16:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)
        elif precision == Precision.BF16:
            return self._from_bfloat16(data)
        elif precision == Precision.INT8:
            return self._dequantize_int8(data, dim)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    
    def _compress_bytes(self, data: bytes) -> bytes:
        """压缩字节数据"""
        if self.config.compression == Compression.NONE:
            return data
        elif self.config.compression == Compression.ZLIB:
            return zlib.compress(data, level=self.config.zlib_level)
        elif self.config.compression == Compression.LZ4:
            return lz4.compress(data, compression_level=self.config.lz4_level)
        elif self.config.compression == Compression.ZSTD:
            return zstd.compress(data, self.config.zstd_level)
        else:
            raise ValueError(f"Unknown compression: {self.config.compression}")
    
    def _decompress_bytes(self, data: bytes, compression: int) -> bytes:
        """解压字节数据"""
        if compression == Compression.NONE:
            return data
        elif compression == Compression.ZLIB:
            return zlib.decompress(data)
        elif compression == Compression.LZ4:
            if not _HAS_LZ4:
                raise ImportError("LZ4 required for decompression")
            return lz4.decompress(data)
        elif compression == Compression.ZSTD:
            if not _HAS_ZSTD:
                raise ImportError("ZSTD required for decompression")
            return zstd.decompress(data)
        else:
            raise ValueError(f"Unknown compression: {compression}")
    
    def _build_output(
        self,
        key_dim: int,
        value_dim: int,
        key_data: bytes,
        value_data: bytes,
    ) -> bytes:
        """构建输出数据"""
        # Header
        header = struct.pack(
            '4sBBBB',
            self.MAGIC,
            self.VERSION,
            self.config.precision,
            self.config.compression,
            0,  # reserved
        )
        
        # Metadata
        metadata = struct.pack(
            'IIII',
            key_dim,
            value_dim,
            len(key_data),
            len(value_data),
        )
        
        return header + metadata + key_data + value_data
    
    def _parse_header(self, data: bytes) -> dict:
        """解析头部"""
        magic, version, precision, compression, _ = struct.unpack(
            '4sBBBB',
            data[:self.HEADER_SIZE]
        )
        
        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        
        return {
            'version': version,
            'precision': precision,
            'compression': compression,
        }
    
    def _parse_metadata(self, data: bytes) -> dict:
        """解析元数据"""
        offset = self.HEADER_SIZE
        key_dim, value_dim, key_size, value_size = struct.unpack(
            'IIII',
            data[offset:offset + self.METADATA_SIZE]
        )
        
        return {
            'key_dim': key_dim,
            'value_dim': value_dim,
            'key_compressed_size': key_size,
            'value_compressed_size': value_size,
        }
    
    def _to_bfloat16(self, arr: np.ndarray) -> np.ndarray:
        """转换为 BFloat16"""
        # BF16 = FP32 的高 16 位
        fp32_bytes = arr.astype(np.float32).view(np.uint32)
        bf16_bytes = (fp32_bytes >> 16).astype(np.uint16)
        return bf16_bytes
    
    def _from_bfloat16(self, data: bytes) -> np.ndarray:
        """从 BFloat16 恢复"""
        bf16_arr = np.frombuffer(data, dtype=np.uint16)
        fp32_bytes = bf16_arr.astype(np.uint32) << 16
        return fp32_bytes.view(np.float32)
    
    def _quantize_int8(self, arr: np.ndarray) -> bytes:
        """INT8 量化（带溢出保护）"""
        if self.config.quantize_symmetric:
            # 对称量化
            max_abs = np.abs(arr).max()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            
            # 防止溢出：确保 scale 不会太小
            scale = max(scale, 1e-10)
            
            quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            # 存储 scale 和量化数据
            return struct.pack('f', scale) + quantized.tobytes()
        else:
            # 非对称量化
            min_val, max_val = arr.min(), arr.max()
            range_val = max_val - min_val
            scale = range_val / 255.0 if range_val > 0 else 1.0
            
            # 防止溢出
            scale = max(scale, 1e-10)
            
            zero_point = -min_val / scale if scale > 0 else 0.0
            # 确保 zero_point 在有效范围内
            zero_point = np.clip(zero_point, 0, 255)
            
            quantized = np.clip(np.round(arr / scale + zero_point), 0, 255).astype(np.uint8)
            return struct.pack('ff', scale, zero_point) + quantized.tobytes()
    
    def _dequantize_int8(self, data: bytes, dim: int) -> np.ndarray:
        """INT8 反量化"""
        if self.config.quantize_symmetric:
            scale = struct.unpack('f', data[:4])[0]
            quantized = np.frombuffer(data[4:], dtype=np.int8)
            return quantized.astype(np.float32) * scale
        else:
            scale, zero_point = struct.unpack('ff', data[:8])
            quantized = np.frombuffer(data[8:], dtype=np.uint8)
            return (quantized.astype(np.float32) - zero_point) * scale
    
    # ==================== 便捷方法 ====================
    
    def get_compression_ratio(
        self,
        key_vector: List[float],
        value_vector: List[float],
    ) -> float:
        """计算压缩比"""
        original_size = (len(key_vector) + len(value_vector)) * 4  # FP32
        compressed = self.compress_vectors(key_vector, value_vector)
        return original_size / len(compressed)
    
    def estimate_size(self, key_dim: int, value_dim: int) -> dict:
        """估算压缩后大小"""
        # 基于经验值估算
        precision_factor = {
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.BF16: 0.5,
            Precision.INT8: 0.25,
        }
        
        compression_factor = {
            Compression.NONE: 1.0,
            Compression.ZLIB: 0.3,
            Compression.LZ4: 0.5,
            Compression.ZSTD: 0.35,
        }
        
        base_size = (key_dim + value_dim) * 4  # FP32 bytes
        estimated = base_size * precision_factor[self.config.precision] * compression_factor[self.config.compression]
        
        return {
            'original_size': base_size,
            'estimated_compressed_size': int(estimated),
            'estimated_ratio': base_size / max(1, estimated),
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取压缩器信息"""
        return {
            'precision': Precision(self.config.precision).name,
            'compression': Compression(self.config.compression).name,
            'zlib_level': self.config.zlib_level,
            'lz4_available': _HAS_LZ4,
            'zstd_available': _HAS_ZSTD,
        }


class DecompressionCache:
    """
    延迟解压缓存
    
    实现三级缓存策略：
    - L1: 热槽位 (常驻解压)
    - L2: 温槽位 (LRU 缓存)
    - L3: 冷槽位 (压缩存储)
    """
    
    def __init__(
        self,
        compressor: VectorCompressor,
        hot_threshold: int = 10,
        warm_cache_size: int = 64,
    ):
        """
        初始化缓存
        
        Args:
            compressor: 向量压缩器
            hot_threshold: 热槽位命中阈值
            warm_cache_size: 温槽位缓存大小
        """
        self.compressor = compressor
        self.hot_threshold = hot_threshold
        self.warm_cache_size = warm_cache_size
        
        # L1: 热槽位 (常驻)
        self.hot_slots: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # L2: 温槽位 (LRU)
        self.warm_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        
        # L3: 冷槽位 (压缩)
        self.cold_storage: Dict[str, bytes] = {}
        
        # 访问统计
        self.access_counts: Dict[str, int] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计
        self._stats = {
            'hot_hits': 0,
            'warm_hits': 0,
            'cold_hits': 0,
            'decompress_count': 0,
            'total_decompress_time_ms': 0.0,
        }
    
    def put(self, lu_id: str, compressed_data: bytes):
        """存入压缩数据"""
        with self._lock:
            self.cold_storage[lu_id] = compressed_data
            self.access_counts[lu_id] = 0
    
    def get(self, lu_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取解压后的向量"""
        with self._lock:
            # 更新访问计数
            self.access_counts[lu_id] = self.access_counts.get(lu_id, 0) + 1
            
            # L1: 检查热槽位
            if lu_id in self.hot_slots:
                self._stats['hot_hits'] += 1
                return self.hot_slots[lu_id]
            
            # L2: 检查温槽位
            if lu_id in self.warm_cache:
                vectors = self.warm_cache.pop(lu_id)
                self.warm_cache[lu_id] = vectors  # 移到末尾
                self._stats['warm_hits'] += 1
                
                # 检查是否应该提升为热槽位
                if self.access_counts[lu_id] >= self.hot_threshold:
                    self.hot_slots[lu_id] = vectors
                    del self.warm_cache[lu_id]
                
                return vectors
            
            # L3: 从冷存储解压
            if lu_id in self.cold_storage:
                self._stats['cold_hits'] += 1
                return self._decompress_and_cache(lu_id)
            
            return None
    
    def _decompress_and_cache(self, lu_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """解压并缓存"""
        if lu_id not in self.cold_storage:
            return None
        
        start_time = time.time()
        
        # 解压
        compressed_data = self.cold_storage[lu_id]
        key_arr, value_arr = self.compressor.decompress_vectors(compressed_data)
        
        # 统计
        decompress_time = (time.time() - start_time) * 1000
        self._stats['decompress_count'] += 1
        self._stats['total_decompress_time_ms'] += decompress_time
        
        # 缓存到温槽位
        vectors = (key_arr, value_arr)
        
        # 检查容量
        while len(self.warm_cache) >= self.warm_cache_size:
            self.warm_cache.popitem(last=False)
        
        self.warm_cache[lu_id] = vectors
        
        return vectors
    
    def preload_hot_slots(self, lu_ids: List[str]):
        """预加载热槽位"""
        with self._lock:
            for lu_id in lu_ids:
                if lu_id in self.cold_storage:
                    vectors = self._decompress_and_cache(lu_id)
                    if vectors:
                        self.hot_slots[lu_id] = vectors
                        if lu_id in self.warm_cache:
                            del self.warm_cache[lu_id]
                        self.access_counts[lu_id] = self.hot_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_hits = (
                self._stats['hot_hits'] + 
                self._stats['warm_hits'] + 
                self._stats['cold_hits']
            )
            
            avg_decompress_time = (
                self._stats['total_decompress_time_ms'] / 
                max(1, self._stats['decompress_count'])
            )
            
            return {
                'hot_slots_count': len(self.hot_slots),
                'warm_slots_count': len(self.warm_cache),
                'cold_slots_count': len(self.cold_storage),
                'hot_hit_rate': self._stats['hot_hits'] / max(1, total_hits),
                'warm_hit_rate': self._stats['warm_hits'] / max(1, total_hits),
                'cold_hit_rate': self._stats['cold_hits'] / max(1, total_hits),
                'avg_decompress_time_ms': avg_decompress_time,
                'total_decompress_count': self._stats['decompress_count'],
            }
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.hot_slots.clear()
            self.warm_cache.clear()
            self.cold_storage.clear()
            self.access_counts.clear()
