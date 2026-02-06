"""
向量压缩单元测试

测试 KV 向量压缩和解压功能。
"""
import pytest
import torch
import numpy as np

# 尝试导入压缩模块
try:
    from aga.persistence.compression import (
        VectorCompressor,
        CompressionConfig,
        Compression,
        Precision,
    )
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False


@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMPRESSION, reason="Compression module not available")
class TestVectorCompressor:
    """向量压缩器测试"""
    
    @pytest.fixture
    def compressor(self):
        """创建压缩器"""
        config = CompressionConfig(
            compression=Compression.ZLIB,
            precision=Precision.FP16,
            zlib_level=6,
        )
        return VectorCompressor(config)
    
    @pytest.fixture
    def sample_vectors(self, bottleneck_dim, hidden_dim):
        """示例向量"""
        key = np.random.randn(bottleneck_dim).astype(np.float32)
        value = np.random.randn(hidden_dim).astype(np.float32)
        return key, value
    
    def test_compress_decompress_roundtrip(self, compressor, sample_vectors):
        """测试压缩-解压往返"""
        key, value = sample_vectors
        
        # 压缩
        compressed = compressor.compress_vectors(key, value)
        
        # 解压
        decompressed_key, decompressed_value = compressor.decompress_vectors(compressed)
        
        # 检查形状
        assert decompressed_key.shape == key.shape
        assert decompressed_value.shape == value.shape
        
        # 检查值（FP16 会有精度损失）
        np.testing.assert_allclose(decompressed_key, key, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(decompressed_value, value, rtol=1e-2, atol=1e-3)
    
    def test_compression_ratio(self, compressor, sample_vectors):
        """测试压缩率"""
        key, value = sample_vectors
        
        # 原始大小
        original_size = key.nbytes + value.nbytes
        
        # 压缩
        compressed = compressor.compress_vectors(key, value)
        compressed_size = len(compressed)
        
        # 压缩率应该大于 1
        ratio = original_size / compressed_size
        assert ratio > 1.0
    
    def test_empty_vectors(self, compressor):
        """测试空向量"""
        empty_key = np.array([], dtype=np.float32)
        empty_value = np.array([], dtype=np.float32)
        
        compressed = compressor.compress_vectors(empty_key, empty_value)
        decompressed_key, decompressed_value = compressor.decompress_vectors(compressed)
        
        assert len(decompressed_key) == 0
        assert len(decompressed_value) == 0
    
    def test_large_vectors(self, compressor):
        """测试大向量"""
        large_key = np.random.randn(1024).astype(np.float32)
        large_value = np.random.randn(4096).astype(np.float32)
        
        compressed = compressor.compress_vectors(large_key, large_value)
        decompressed_key, decompressed_value = compressor.decompress_vectors(compressed)
        
        np.testing.assert_allclose(decompressed_key, large_key, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(decompressed_value, large_value, rtol=1e-2, atol=1e-3)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMPRESSION, reason="Compression module not available")
class TestCompressionConfig:
    """压缩配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = CompressionConfig()
        
        assert config.compression is not None
        assert config.precision is not None
    
    def test_compression_level_range(self):
        """测试压缩级别范围"""
        # 有效范围
        for level in [1, 5, 9]:
            config = CompressionConfig(zlib_level=level)
            assert config.zlib_level == level
    
    def test_preset_configs(self):
        """测试预设配置"""
        # 速度优先
        speed_config = CompressionConfig.for_speed()
        assert speed_config.precision == Precision.FP16
        
        # 压缩比优先
        size_config = CompressionConfig.for_size()
        assert size_config.precision == Precision.FP16
        
        # 质量优先
        quality_config = CompressionConfig.for_quality()
        assert quality_config.precision == Precision.FP32


@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMPRESSION, reason="Compression module not available")
class TestCompressionEdgeCases:
    """压缩边界情况测试"""
    
    @pytest.fixture
    def compressor(self):
        config = CompressionConfig(
            compression=Compression.ZLIB,
            precision=Precision.FP16,
        )
        return VectorCompressor(config)
    
    def test_special_values(self, compressor):
        """测试特殊值"""
        # 包含 0 和接近 0 的值
        special_key = np.array([0.0, 1e-10, -1e-10], dtype=np.float32)
        special_value = np.array([0.0] * 10, dtype=np.float32)
        
        compressed = compressor.compress_vectors(special_key, special_value)
        decompressed_key, decompressed_value = compressor.decompress_vectors(compressed)
        
        assert np.isfinite(decompressed_key).all()
    
    def test_repeated_compression(self, compressor, bottleneck_dim, hidden_dim):
        """测试重复压缩"""
        key = np.random.randn(bottleneck_dim).astype(np.float32)
        value = np.random.randn(hidden_dim).astype(np.float32)
        
        # 多次压缩解压
        for _ in range(10):
            compressed = compressor.compress_vectors(key, value)
            key, value = compressor.decompress_vectors(compressed)
        
        # 不应该崩溃
        assert key is not None
        assert value is not None
