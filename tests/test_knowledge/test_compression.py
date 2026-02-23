"""
aga-knowledge 文本压缩模块测试
"""

import pytest

from aga_knowledge.persistence.compression import (
    CompressionAlgorithm,
    TextCompressionConfig,
    TextCompressor,
    DecompressionCache,
)


class TestTextCompressionConfig:
    """测试压缩配置"""

    def test_default_config(self):
        config = TextCompressionConfig()
        assert config.algorithm == CompressionAlgorithm.ZLIB
        assert config.zlib_level == 6
        assert config.encoding == "utf-8"

    def test_for_speed(self):
        config = TextCompressionConfig.for_speed()
        # LZ4 或 NONE（取决于是否安装了 lz4）
        assert config.algorithm in (
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.NONE,
        )

    def test_for_size(self):
        config = TextCompressionConfig.for_size()
        assert config.algorithm in (
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.ZLIB,
        )

    def test_for_balanced(self):
        config = TextCompressionConfig.for_balanced()
        assert config.algorithm == CompressionAlgorithm.ZLIB


class TestTextCompressor:
    """测试文本压缩器"""

    @pytest.fixture
    def compressor(self):
        return TextCompressor(TextCompressionConfig())

    @pytest.fixture
    def compressor_none(self):
        return TextCompressor(
            TextCompressionConfig(algorithm=CompressionAlgorithm.NONE)
        )

    def test_compress_decompress_zlib(self, compressor):
        condition = "当用户询问天气时"
        decision = "提供详细天气预报信息"

        compressed = compressor.compress_knowledge(condition, decision)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        cond_out, dec_out = compressor.decompress_knowledge(compressed)
        assert cond_out == condition
        assert dec_out == decision

    def test_compress_decompress_none(self, compressor_none):
        condition = "test condition"
        decision = "test decision"

        compressed = compressor_none.compress_knowledge(condition, decision)
        cond_out, dec_out = compressor_none.decompress_knowledge(compressed)
        assert cond_out == condition
        assert dec_out == decision

    def test_compress_empty_strings(self, compressor):
        compressed = compressor.compress_knowledge("", "")
        cond_out, dec_out = compressor.decompress_knowledge(compressed)
        assert cond_out == ""
        assert dec_out == ""

    def test_compress_long_text(self, compressor):
        condition = "A" * 10000
        decision = "B" * 10000

        compressed = compressor.compress_knowledge(condition, decision)
        cond_out, dec_out = compressor.decompress_knowledge(compressed)
        assert cond_out == condition
        assert dec_out == decision

        # 压缩后应该比原始数据小
        original_size = len(condition.encode()) + len(decision.encode())
        assert len(compressed) < original_size

    def test_compress_unicode(self, compressor):
        condition = "条件：当模型遇到高熵 token 时"
        decision = "决策：注入领域知识以降低不确定性"

        compressed = compressor.compress_knowledge(condition, decision)
        cond_out, dec_out = compressor.decompress_knowledge(compressed)
        assert cond_out == condition
        assert dec_out == decision

    def test_compress_batch(self, compressor):
        records = [
            {"condition": f"condition_{i}", "decision": f"decision_{i}"}
            for i in range(5)
        ]

        compressed_list = compressor.compress_batch(records)
        assert len(compressed_list) == 5

        decompressed_list = compressor.decompress_batch(compressed_list)
        assert len(decompressed_list) == 5

        for i, (cond, dec) in enumerate(decompressed_list):
            assert cond == f"condition_{i}"
            assert dec == f"decision_{i}"

    def test_compression_ratio(self, compressor):
        condition = "A" * 1000
        decision = "B" * 1000
        ratio = compressor.get_compression_ratio(condition, decision)
        assert ratio > 1.0  # 压缩后应该更小

    def test_stats(self, compressor):
        compressor.compress_knowledge("c", "d")
        stats = compressor.get_stats()
        assert stats["compress_count"] == 1
        assert stats["algorithm"] == "ZLIB"

    def test_info(self, compressor):
        info = compressor.get_info()
        assert info["algorithm"] == "ZLIB"
        assert "lz4_available" in info
        assert "zstd_available" in info

    def test_invalid_data_decompress(self, compressor):
        with pytest.raises(ValueError):
            compressor.decompress_knowledge(b"invalid data")

    def test_too_short_data(self, compressor):
        with pytest.raises(ValueError):
            compressor.decompress_knowledge(b"short")


class TestDecompressionCache:
    """测试延迟解压缓存"""

    @pytest.fixture
    def cache(self):
        compressor = TextCompressor(TextCompressionConfig())
        return DecompressionCache(
            compressor=compressor,
            hot_threshold=3,
            warm_cache_size=5,
        )

    def test_put_and_get(self, cache):
        compressed = cache.compressor.compress_knowledge("cond", "dec")
        cache.put("lu_001", compressed)

        result = cache.get("lu_001")
        assert result is not None
        assert result == ("cond", "dec")

    def test_put_text(self, cache):
        cache.put_text("lu_001", "condition", "decision")
        result = cache.get("lu_001")
        assert result == ("condition", "decision")

    def test_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_hot_promotion(self, cache):
        cache.put_text("lu_001", "c", "d")

        # 访问 3 次（hot_threshold=3）
        for _ in range(3):
            cache.get("lu_001")

        # 应该在热数据中
        assert "lu_001" in cache.hot_data

    def test_warm_lru_eviction(self, cache):
        # 填满温缓存（warm_cache_size=5）
        for i in range(6):
            cache.put_text(f"lu_{i:03d}", f"c{i}", f"d{i}")
            cache.get(f"lu_{i:03d}")  # 触发从冷到温

        # 最早的应该被淘汰出温缓存
        assert "lu_000" not in cache.warm_cache

    def test_preload_hot(self, cache):
        for i in range(3):
            cache.put_text(f"lu_{i:03d}", f"c{i}", f"d{i}")

        cache.preload_hot(["lu_000", "lu_001"])
        assert "lu_000" in cache.hot_data
        assert "lu_001" in cache.hot_data

    def test_remove(self, cache):
        cache.put_text("lu_001", "c", "d")
        cache.get("lu_001")  # 移到温缓存

        cache.remove("lu_001")
        assert cache.get("lu_001") is None

    def test_clear(self, cache):
        cache.put_text("lu_001", "c", "d")
        cache.get("lu_001")

        cache.clear()
        assert cache.get("lu_001") is None
        assert len(cache.hot_data) == 0
        assert len(cache.warm_cache) == 0
        assert len(cache.cold_storage) == 0

    def test_stats(self, cache):
        cache.put_text("lu_001", "c", "d")
        cache.get("lu_001")  # cold hit
        cache.get("lu_001")  # warm hit

        stats = cache.get_stats()
        assert stats["cold_count"] == 1
        assert stats["cold_hit_rate"] > 0
        assert stats["warm_hit_rate"] > 0
