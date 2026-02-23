"""
aga-knowledge 文本压缩模块

明文 KV 版本：提供 condition/decision 文本的压缩和解压功能。
不处理向量数据，只压缩文本内容以减少存储空间。

特性:
- 多种压缩算法 (zlib, lz4, zstd)
- 延迟解压缓存
- 批量压缩/解压
- 压缩比统计

依赖:
    - zlib: 内置
    - lz4: pip install lz4 (可选)
    - zstd: pip install zstd (可选)
"""

import struct
import zlib
import json
import time
import threading
import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional, Dict, Any, List, Union

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


class CompressionAlgorithm(IntEnum):
    """压缩算法"""
    NONE = 0    # 无压缩
    ZLIB = 1    # zlib (通用，压缩比高)
    LZ4 = 2     # lz4 (速度快，压缩比中等)
    ZSTD = 3    # zstd (平衡)


@dataclass
class TextCompressionConfig:
    """文本压缩配置"""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB
    zlib_level: int = 6           # zlib 压缩级别 (1-9)
    lz4_level: int = 0            # lz4 压缩级别 (0-16)
    zstd_level: int = 3           # zstd 压缩级别 (1-22)
    encoding: str = "utf-8"       # 文本编码

    @classmethod
    def for_speed(cls) -> "TextCompressionConfig":
        """速度优先配置"""
        return cls(
            algorithm=(
                CompressionAlgorithm.LZ4
                if _HAS_LZ4
                else CompressionAlgorithm.NONE
            ),
        )

    @classmethod
    def for_size(cls) -> "TextCompressionConfig":
        """压缩比优先配置"""
        return cls(
            algorithm=(
                CompressionAlgorithm.ZSTD
                if _HAS_ZSTD
                else CompressionAlgorithm.ZLIB
            ),
            zstd_level=6,
            zlib_level=9,
        )

    @classmethod
    def for_balanced(cls) -> "TextCompressionConfig":
        """平衡配置"""
        return cls(
            algorithm=CompressionAlgorithm.ZLIB,
            zlib_level=6,
        )


class TextCompressor:
    """
    文本压缩器

    提供 condition/decision 文本的压缩和解压功能。

    使用示例:
        ```python
        compressor = TextCompressor(TextCompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
        ))

        # 压缩知识记录
        compressed = compressor.compress_knowledge(
            condition="当用户询问天气时",
            decision="提供天气信息",
        )

        # 解压
        condition, decision = compressor.decompress_knowledge(compressed)

        # 获取压缩比
        ratio = compressor.get_compression_ratio(
            "当用户询问天气时", "提供天气信息"
        )
        ```
    """

    # 魔数和版本
    MAGIC = b"TKC1"  # Text Knowledge Compression v1
    VERSION = 1
    HEADER_SIZE = 8
    METADATA_SIZE = 8

    def __init__(self, config: Optional[TextCompressionConfig] = None):
        self.config = config or TextCompressionConfig()

        # 验证依赖
        if (
            self.config.algorithm == CompressionAlgorithm.LZ4
            and not _HAS_LZ4
        ):
            logger.warning("LZ4 not available, falling back to ZLIB")
            self.config.algorithm = CompressionAlgorithm.ZLIB
        if (
            self.config.algorithm == CompressionAlgorithm.ZSTD
            and not _HAS_ZSTD
        ):
            logger.warning("ZSTD not available, falling back to ZLIB")
            self.config.algorithm = CompressionAlgorithm.ZLIB

        # 统计
        self._stats = {
            "compress_count": 0,
            "decompress_count": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "total_compress_time_ms": 0.0,
            "total_decompress_time_ms": 0.0,
        }

    def compress_knowledge(
        self,
        condition: str,
        decision: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        压缩知识记录

        Args:
            condition: 条件文本
            decision: 决策文本
            metadata: 可选元数据（JSON 序列化后一起压缩）

        Returns:
            压缩后的字节数据
        """
        start_time = time.perf_counter()

        # 编码文本
        cond_bytes = condition.encode(self.config.encoding)
        dec_bytes = decision.encode(self.config.encoding)

        # 压缩
        cond_compressed = self._compress_bytes(cond_bytes)
        dec_compressed = self._compress_bytes(dec_bytes)

        # 构建输出
        result = self._build_output(
            cond_data=cond_compressed,
            dec_data=dec_compressed,
        )

        # 统计
        original_size = len(cond_bytes) + len(dec_bytes)
        self._stats["compress_count"] += 1
        self._stats["total_original_bytes"] += original_size
        self._stats["total_compressed_bytes"] += len(result)
        self._stats["total_compress_time_ms"] += (
            (time.perf_counter() - start_time) * 1000
        )

        return result

    def decompress_knowledge(
        self, data: bytes
    ) -> Tuple[str, str]:
        """
        解压知识记录

        Args:
            data: 压缩后的字节数据

        Returns:
            (condition, decision) 元组
        """
        start_time = time.perf_counter()

        if not data or len(data) < self.HEADER_SIZE + self.METADATA_SIZE:
            raise ValueError("Invalid compressed data: too short")

        try:
            # 解析头部
            header = self._parse_header(data)

            if header["version"] > self.VERSION:
                raise ValueError(
                    f"Unsupported version: {header['version']} "
                    f"(max supported: {self.VERSION})"
                )

            # 解析元数据
            metadata = self._parse_metadata(data)

            # 提取压缩数据
            offset = self.HEADER_SIZE + self.METADATA_SIZE
            cond_data = data[
                offset:offset + metadata["cond_compressed_size"]
            ]
            offset += metadata["cond_compressed_size"]
            dec_data = data[
                offset:offset + metadata["dec_compressed_size"]
            ]

            # 解压
            cond_bytes = self._decompress_bytes(
                cond_data, header["algorithm"]
            )
            dec_bytes = self._decompress_bytes(
                dec_data, header["algorithm"]
            )

            # 解码
            condition = cond_bytes.decode(self.config.encoding)
            decision = dec_bytes.decode(self.config.encoding)

            # 统计
            self._stats["decompress_count"] += 1
            self._stats["total_decompress_time_ms"] += (
                (time.perf_counter() - start_time) * 1000
            )

            return condition, decision

        except (struct.error, ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decompress knowledge: {e}")

    def compress_batch(
        self, records: List[Dict[str, str]]
    ) -> List[bytes]:
        """
        批量压缩

        Args:
            records: 知识记录列表，每项需包含 condition 和 decision

        Returns:
            压缩后的字节数据列表
        """
        return [
            self.compress_knowledge(
                condition=r.get("condition", ""),
                decision=r.get("decision", ""),
            )
            for r in records
        ]

    def decompress_batch(
        self, compressed_list: List[bytes]
    ) -> List[Tuple[str, str]]:
        """
        批量解压

        Args:
            compressed_list: 压缩后的字节数据列表

        Returns:
            (condition, decision) 元组列表
        """
        return [
            self.decompress_knowledge(data) for data in compressed_list
        ]

    def _compress_bytes(self, data: bytes) -> bytes:
        """压缩字节数据"""
        if self.config.algorithm == CompressionAlgorithm.NONE:
            return data
        elif self.config.algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(data, level=self.config.zlib_level)
        elif self.config.algorithm == CompressionAlgorithm.LZ4:
            return lz4.compress(
                data, compression_level=self.config.lz4_level
            )
        elif self.config.algorithm == CompressionAlgorithm.ZSTD:
            return zstd.compress(data, self.config.zstd_level)
        else:
            raise ValueError(
                f"Unknown compression algorithm: {self.config.algorithm}"
            )

    def _decompress_bytes(self, data: bytes, algorithm: int) -> bytes:
        """解压字节数据"""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        elif algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(data)
        elif algorithm == CompressionAlgorithm.LZ4:
            if not _HAS_LZ4:
                raise ImportError("LZ4 required for decompression")
            return lz4.decompress(data)
        elif algorithm == CompressionAlgorithm.ZSTD:
            if not _HAS_ZSTD:
                raise ImportError("ZSTD required for decompression")
            return zstd.decompress(data)
        else:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")

    def _build_output(
        self,
        cond_data: bytes,
        dec_data: bytes,
    ) -> bytes:
        """构建输出数据"""
        # Header: magic(4) + version(1) + algorithm(1) + reserved(2)
        header = struct.pack(
            "4sBBH",
            self.MAGIC,
            self.VERSION,
            self.config.algorithm,
            0,  # reserved
        )

        # Metadata: cond_size(4) + dec_size(4)
        metadata = struct.pack(
            "II",
            len(cond_data),
            len(dec_data),
        )

        return header + metadata + cond_data + dec_data

    def _parse_header(self, data: bytes) -> dict:
        """解析头部"""
        magic, version, algorithm, _ = struct.unpack(
            "4sBBH", data[: self.HEADER_SIZE]
        )

        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        return {
            "version": version,
            "algorithm": algorithm,
        }

    def _parse_metadata(self, data: bytes) -> dict:
        """解析元数据"""
        offset = self.HEADER_SIZE
        cond_size, dec_size = struct.unpack(
            "II", data[offset:offset + self.METADATA_SIZE]
        )

        return {
            "cond_compressed_size": cond_size,
            "dec_compressed_size": dec_size,
        }

    def get_compression_ratio(
        self, condition: str, decision: str
    ) -> float:
        """计算压缩比"""
        original_size = (
            len(condition.encode(self.config.encoding))
            + len(decision.encode(self.config.encoding))
        )
        compressed = self.compress_knowledge(condition, decision)
        return original_size / max(1, len(compressed))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_ratio = (
            self._stats["total_original_bytes"]
            / max(1, self._stats["total_compressed_bytes"])
        )
        avg_compress_time = (
            self._stats["total_compress_time_ms"]
            / max(1, self._stats["compress_count"])
        )
        avg_decompress_time = (
            self._stats["total_decompress_time_ms"]
            / max(1, self._stats["decompress_count"])
        )

        return {
            "algorithm": CompressionAlgorithm(
                self.config.algorithm
            ).name,
            "compress_count": self._stats["compress_count"],
            "decompress_count": self._stats["decompress_count"],
            "total_original_bytes": self._stats["total_original_bytes"],
            "total_compressed_bytes": self._stats[
                "total_compressed_bytes"
            ],
            "avg_compression_ratio": f"{avg_ratio:.2f}",
            "avg_compress_time_ms": f"{avg_compress_time:.3f}",
            "avg_decompress_time_ms": f"{avg_decompress_time:.3f}",
            "lz4_available": _HAS_LZ4,
            "zstd_available": _HAS_ZSTD,
        }

    def get_info(self) -> Dict[str, Any]:
        """获取压缩器信息"""
        return {
            "algorithm": CompressionAlgorithm(
                self.config.algorithm
            ).name,
            "zlib_level": self.config.zlib_level,
            "lz4_available": _HAS_LZ4,
            "zstd_available": _HAS_ZSTD,
        }


class DecompressionCache:
    """
    延迟解压缓存

    实现两级缓存策略:
    - L1: 热数据 (常驻解压)
    - L2: 温数据 (LRU 缓存)
    """

    def __init__(
        self,
        compressor: TextCompressor,
        hot_threshold: int = 10,
        warm_cache_size: int = 256,
    ):
        """
        初始化缓存

        Args:
            compressor: 文本压缩器
            hot_threshold: 热数据命中阈值
            warm_cache_size: 温数据缓存大小
        """
        self.compressor = compressor
        self.hot_threshold = hot_threshold
        self.warm_cache_size = warm_cache_size

        # L1: 热数据 (常驻)
        self.hot_data: Dict[str, Tuple[str, str]] = {}

        # L2: 温数据 (LRU)
        self.warm_cache: OrderedDict[str, Tuple[str, str]] = OrderedDict()

        # 冷存储 (压缩)
        self.cold_storage: Dict[str, bytes] = {}

        # 访问统计
        self.access_counts: Dict[str, int] = {}

        # 线程锁
        self._lock = threading.RLock()

        # 统计
        self._stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_hits": 0,
            "misses": 0,
            "decompress_count": 0,
            "total_decompress_time_ms": 0.0,
        }

    def put(self, lu_id: str, compressed_data: bytes):
        """存入压缩数据"""
        with self._lock:
            self.cold_storage[lu_id] = compressed_data
            self.access_counts[lu_id] = 0

    def put_text(self, lu_id: str, condition: str, decision: str):
        """存入明文数据（自动压缩到冷存储）"""
        compressed = self.compressor.compress_knowledge(
            condition, decision
        )
        self.put(lu_id, compressed)

    def get(self, lu_id: str) -> Optional[Tuple[str, str]]:
        """获取解压后的文本"""
        with self._lock:
            # 更新访问计数
            self.access_counts[lu_id] = (
                self.access_counts.get(lu_id, 0) + 1
            )

            # L1: 检查热数据
            if lu_id in self.hot_data:
                self._stats["hot_hits"] += 1
                return self.hot_data[lu_id]

            # L2: 检查温数据
            if lu_id in self.warm_cache:
                texts = self.warm_cache.pop(lu_id)
                self.warm_cache[lu_id] = texts  # 移到末尾
                self._stats["warm_hits"] += 1

                # 检查是否应该提升为热数据
                if self.access_counts[lu_id] >= self.hot_threshold:
                    self.hot_data[lu_id] = texts
                    del self.warm_cache[lu_id]

                return texts

            # L3: 从冷存储解压
            if lu_id in self.cold_storage:
                self._stats["cold_hits"] += 1
                return self._decompress_and_cache(lu_id)

            self._stats["misses"] += 1
            return None

    def _decompress_and_cache(
        self, lu_id: str
    ) -> Optional[Tuple[str, str]]:
        """解压并缓存"""
        if lu_id not in self.cold_storage:
            return None

        start_time = time.perf_counter()

        compressed_data = self.cold_storage[lu_id]
        condition, decision = self.compressor.decompress_knowledge(
            compressed_data
        )

        decompress_time = (time.perf_counter() - start_time) * 1000
        self._stats["decompress_count"] += 1
        self._stats["total_decompress_time_ms"] += decompress_time

        texts = (condition, decision)

        # 缓存到温数据
        if self.warm_cache_size > 0:
            while len(self.warm_cache) >= self.warm_cache_size:
                self.warm_cache.popitem(last=False)
            self.warm_cache[lu_id] = texts

        return texts

    def preload_hot(self, lu_ids: List[str]):
        """预加载热数据"""
        with self._lock:
            for lu_id in lu_ids:
                if lu_id in self.cold_storage:
                    texts = self._decompress_and_cache(lu_id)
                    if texts:
                        self.hot_data[lu_id] = texts
                        if lu_id in self.warm_cache:
                            del self.warm_cache[lu_id]
                        self.access_counts[lu_id] = self.hot_threshold

    def remove(self, lu_id: str):
        """移除数据"""
        with self._lock:
            self.hot_data.pop(lu_id, None)
            self.warm_cache.pop(lu_id, None)
            self.cold_storage.pop(lu_id, None)
            self.access_counts.pop(lu_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_hits = (
                self._stats["hot_hits"]
                + self._stats["warm_hits"]
                + self._stats["cold_hits"]
            )

            avg_decompress_time = (
                self._stats["total_decompress_time_ms"]
                / max(1, self._stats["decompress_count"])
            )

            return {
                "hot_count": len(self.hot_data),
                "warm_count": len(self.warm_cache),
                "cold_count": len(self.cold_storage),
                "hot_hit_rate": (
                    self._stats["hot_hits"] / max(1, total_hits)
                ),
                "warm_hit_rate": (
                    self._stats["warm_hits"] / max(1, total_hits)
                ),
                "cold_hit_rate": (
                    self._stats["cold_hits"] / max(1, total_hits)
                ),
                "miss_count": self._stats["misses"],
                "avg_decompress_time_ms": avg_decompress_time,
                "total_decompress_count": self._stats[
                    "decompress_count"
                ],
            }

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.hot_data.clear()
            self.warm_cache.clear()
            self.cold_storage.clear()
            self.access_counts.clear()
