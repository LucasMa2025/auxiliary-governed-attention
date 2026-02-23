"""
aga-knowledge 编码器模块测试

注意: SentenceTransformerEncoder 测试需要安装 sentence-transformers，
在 CI 环境中可能跳过。SimpleHashEncoder 不需要额外依赖。
"""

import pytest
from unittest.mock import patch, MagicMock

from aga_knowledge.encoder.base import (
    BaseEncoder,
    EncoderConfig,
    EncodedKnowledge,
)
from aga_knowledge.encoder.simple_encoder import SimpleHashEncoder
from aga_knowledge.encoder import create_encoder


class TestEncoderConfig:
    """测试编码器配置"""

    def test_default_config(self):
        config = EncoderConfig()
        assert config.backend == "sentence_transformer"
        assert config.key_dim == 64
        assert config.value_dim == 4096
        assert config.device == "cpu"
        assert config.normalize is True

    def test_from_dict(self):
        data = {
            "backend": "simple_hash",
            "key_dim": 32,
            "value_dim": 2048,
            "device": "cuda",
            "batch_size": 16,
        }
        config = EncoderConfig.from_dict(data)
        assert config.backend == "simple_hash"
        assert config.key_dim == 32
        assert config.value_dim == 2048
        assert config.device == "cuda"
        assert config.batch_size == 16

    def test_from_dict_ignores_unknown(self):
        data = {
            "backend": "simple_hash",
            "unknown_field": "value",
        }
        config = EncoderConfig.from_dict(data)
        assert config.backend == "simple_hash"
        assert not hasattr(config, "unknown_field")

    def test_validate_valid(self):
        config = EncoderConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid(self):
        config = EncoderConfig(key_dim=0, value_dim=-1, batch_size=0)
        errors = config.validate()
        assert len(errors) == 3


class TestEncodedKnowledge:
    """测试编码后的知识数据类"""

    def test_creation(self):
        ek = EncodedKnowledge(
            lu_id="lu_001",
            condition="when X",
            decision="do Y",
            key_vector=[0.1, 0.2, 0.3],
            value_vector=[0.4, 0.5, 0.6],
            reliability=0.9,
            metadata={"source": "test"},
        )
        assert ek.lu_id == "lu_001"
        assert len(ek.key_vector) == 3
        assert len(ek.value_vector) == 3
        assert ek.reliability == 0.9


class TestSimpleHashEncoder:
    """测试简单哈希编码器"""

    @pytest.fixture
    def encoder(self):
        config = EncoderConfig(
            backend="simple_hash",
            key_dim=16,
            value_dim=32,
            normalize=True,
        )
        enc = SimpleHashEncoder(config)
        enc.warmup()
        return enc

    @pytest.fixture
    def encoder_no_normalize(self):
        config = EncoderConfig(
            backend="simple_hash",
            key_dim=16,
            value_dim=32,
            normalize=False,
        )
        enc = SimpleHashEncoder(config)
        enc.warmup()
        return enc

    def test_encode_single(self, encoder):
        result = encoder.encode(
            condition="when X happens",
            decision="do Y",
            lu_id="lu_001",
        )
        assert isinstance(result, EncodedKnowledge)
        assert result.lu_id == "lu_001"
        assert len(result.key_vector) == 16
        assert len(result.value_vector) == 32

    def test_encode_deterministic(self, encoder):
        r1 = encoder.encode("cond", "dec", "lu_001")
        r2 = encoder.encode("cond", "dec", "lu_002")
        # 相同文本应产生相同向量
        assert r1.key_vector == r2.key_vector
        assert r1.value_vector == r2.value_vector

    def test_encode_different_texts(self, encoder):
        r1 = encoder.encode("condition A", "decision A", "lu_001")
        r2 = encoder.encode("condition B", "decision B", "lu_002")
        # 不同文本应产生不同向量
        assert r1.key_vector != r2.key_vector

    def test_encode_empty_text(self, encoder):
        result = encoder.encode("", "", "lu_empty")
        assert len(result.key_vector) == 16
        assert len(result.value_vector) == 32

    def test_encode_batch(self, encoder):
        records = [
            {"condition": f"c{i}", "decision": f"d{i}", "lu_id": f"lu_{i}"}
            for i in range(5)
        ]
        results = encoder.encode_batch(records)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.lu_id == f"lu_{i}"
            assert len(r.key_vector) == 16
            assert len(r.value_vector) == 32

    def test_encode_batch_empty(self, encoder):
        results = encoder.encode_batch([])
        assert results == []

    def test_cache_hit(self, encoder):
        # 第一次编码
        encoder.encode("cond", "dec", "lu_001")
        # 第二次应命中缓存
        encoder.encode("cond", "dec", "lu_002")

        stats = encoder.get_stats()
        assert stats["cache_size"] >= 1

    def test_cache_eviction(self):
        config = EncoderConfig(
            backend="simple_hash",
            key_dim=8,
            value_dim=16,
            cache_max_size=3,
        )
        enc = SimpleHashEncoder(config)
        enc.warmup()

        # 填满缓存
        for i in range(5):
            enc.encode(f"cond_{i}", f"dec_{i}", f"lu_{i}")

        # 缓存大小不应超过 max_size
        assert enc.get_stats()["cache_size"] <= 3

    def test_cache_disabled(self):
        config = EncoderConfig(
            backend="simple_hash",
            key_dim=8,
            value_dim=16,
            cache_enabled=False,
        )
        enc = SimpleHashEncoder(config)
        enc.warmup()

        enc.encode("cond", "dec", "lu_001")
        assert enc.get_stats()["cache_size"] == 0

    def test_get_stats(self, encoder):
        encoder.encode("c", "d", "lu_001")
        stats = encoder.get_stats()
        assert stats["type"] == "SimpleHashEncoder"
        assert stats["backend"] == "simple_hash"
        assert stats["initialized"] is True
        assert stats["encode_count"] >= 1

    def test_shutdown(self, encoder):
        encoder.encode("c", "d", "lu_001")
        encoder.shutdown()
        assert encoder.get_stats()["initialized"] is False
        assert encoder.get_stats()["cache_size"] == 0

    def test_repr(self, encoder):
        r = repr(encoder)
        assert "SimpleHashEncoder" in r
        assert "simple_hash" in r

    def test_no_normalize(self, encoder_no_normalize):
        result = encoder_no_normalize.encode("cond", "dec", "lu_001")
        assert len(result.key_vector) == 16
        assert len(result.value_vector) == 32


class TestCreateEncoder:
    """测试编码器工厂函数"""

    def test_create_simple_hash(self):
        config = EncoderConfig(backend="simple_hash", key_dim=8, value_dim=16)
        encoder = create_encoder(config)
        assert isinstance(encoder, SimpleHashEncoder)

    def test_create_none_raises(self):
        config = EncoderConfig(backend="none")
        with pytest.raises(ValueError, match="不需要编码器"):
            create_encoder(config)

    def test_create_unknown_raises(self):
        config = EncoderConfig(backend="unknown_backend")
        with pytest.raises(ValueError, match="未知的编码器后端"):
            create_encoder(config)

    def test_create_sentence_transformer(self):
        """测试创建 SentenceTransformer 编码器（可能需要安装依赖）"""
        config = EncoderConfig(
            backend="sentence_transformer",
            key_dim=64,
            value_dim=4096,
        )
        try:
            encoder = create_encoder(config)
            from aga_knowledge.encoder.sentence_transformer_encoder import (
                SentenceTransformerEncoder,
            )
            assert isinstance(encoder, SentenceTransformerEncoder)
        except ImportError:
            pytest.skip("sentence-transformers not installed")
